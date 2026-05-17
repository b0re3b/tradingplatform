from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from core.logger import get_logger

from .config import LiquidityConfig
from .enums import LiquidityLevelType, LiquiditySide
from .models import (
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    EqualLevel,
    LiquidityLevel,
    StopCluster,
)
from .scoring import LiquidityScorer
from .utils import (
    clamp,
    get_candle_close,
    get_candle_high,
    get_candle_low,
    get_candle_volume,
    get_first_value,
    merge_price_ranges,
    midpoint,
    pct_distance,
    safe_float,
    safe_mean,
)


@dataclass(slots=True)
class OrderbookLevel:
    """
    Нормалізований orderbook level для stop-cluster enhancement.
    """

    price: float
    size: float
    side: str  # "bid" | "ask"

    def __post_init__(self) -> None:
        self.price = safe_float(self.price)
        self.size = safe_float(self.size)


@dataclass(slots=True)
class StopClusterCandidate:
    """
    Внутрішня модель-кандидат для формування StopCluster.

    Це не публічна модель і не event payload.
    Використовується тільки всередині StopClustersDetector.

    Важливо:
    - exchange + market_type зберігаються в candidate, щоб merge/dedup
      не змішували різні біржі або різні типи ринку;
    - swept_at переноситься із source LiquidityLevel у StopCluster;
    - partially/fully swept source levels знижують density/confidence, але не губляться;
    - це дає strategy-шару змогу відрізнити swept cluster від звичайного cluster.
    """

    symbol: str
    timeframe: str
    side: LiquiditySide

    low_price: float
    high_price: float

    exchange: str = DEFAULT_EXCHANGE
    market_type: str = DEFAULT_MARKET_TYPE

    source_levels: list[LiquidityLevel] = field(default_factory=list)

    created_at: datetime | None = None
    updated_at: datetime | None = None
    swept_at: datetime | None = None
    invalidated_at: datetime | None = None

    volume_score: float = 0.0
    orderbook_score: float = 0.0
    compression_score: float = 0.0
    time_decay_factor: float = 1.0
    partial_sweep_factor: float = 1.0

    swept_source_count: int = 0
    partially_swept_source_count: int = 0

    def __post_init__(self) -> None:
        self.exchange = self._normalize_scope_value(self.exchange, DEFAULT_EXCHANGE)
        self.market_type = self._normalize_scope_value(self.market_type, DEFAULT_MARKET_TYPE)
        self.symbol = str(self.symbol or "").strip().upper()
        self.timeframe = str(self.timeframe or "").strip()

        self.low_price = safe_float(self.low_price)
        self.high_price = safe_float(self.high_price)

        if self.low_price > self.high_price:
            self.low_price, self.high_price = self.high_price, self.low_price

        self.volume_score = clamp(self.volume_score, 0.0, 1.0)
        self.orderbook_score = clamp(self.orderbook_score, 0.0, 1.0)
        self.compression_score = clamp(self.compression_score, 0.0, 1.0)
        self.time_decay_factor = clamp(self.time_decay_factor, 0.0, 1.0)
        self.partial_sweep_factor = clamp(self.partial_sweep_factor, 0.0, 1.0)

        self.swept_source_count = max(0, int(self.swept_source_count))
        self.partially_swept_source_count = max(
            0,
            int(self.partially_swept_source_count),
        )

    @property
    def scope_key(self) -> str:
        return (
            f"{self.exchange.lower()}:"
            f"{self.market_type.lower()}:"
            f"{self.symbol}:"
            f"{self.timeframe}"
        )

    @property
    def center_price(self) -> float:
        return midpoint(self.low_price, self.high_price)

    def width(self) -> float:
        return max(0.0, self.high_price - self.low_price)

    def overlaps(self, other: "StopClusterCandidate") -> bool:
        return not (
            self.high_price < other.low_price
            or other.high_price < self.low_price
        )

    def same_scope(self, other: "StopClusterCandidate") -> bool:
        return (
            self.exchange == other.exchange
            and self.market_type == other.market_type
            and self.symbol == other.symbol
            and self.timeframe == other.timeframe
        )

    def is_swept(self) -> bool:
        """
        Swept-context candidate.

        Partial sweep також вважається swept-context для downstream reversal logic.
        Це не означає, що cluster повністю invalidated; це означає, що він має
        sweep/reversal context.
        """
        return (
            self.swept_at is not None
            or self.swept_source_count > 0
            or self.partially_swept_source_count > 0
        )

    @staticmethod
    def _normalize_scope_value(value: Any, default: str) -> str:
        normalized = str(value or default).strip()
        return normalized if normalized else default


class StopClustersDetector:
    """
    Production-ready detector stop/liquidity clusters.

    Відповідальність:
    - перетворити liquidity levels у stop-cluster candidates;
    - врахувати partial/full sweep;
    - підсилити density через volume/orderbook/compression;
    - застосувати time decay;
    - merge/deduplicate candidates;
    - повернути scored StopCluster.

    Архітектурні правила:
    - не приймає EventBus;
    - не приймає Scheduler;
    - не публікує події;
    - не керує lifecycle;
    - використовується LiquidityMap як чистий domain detector.

    Multi-exchange behavior:
    - detector не отримує дані напряму з бірж;
    - exchange/market_type передаються як scope metadata;
    - merge/dedup є scope-aware і не змішує різні біржі/market_type.
    """

    def __init__(
        self,
        config: LiquidityConfig,
        scorer: LiquidityScorer | None = None,
    ) -> None:
        self._config = config
        self._config.validate()

        self._scorer = scorer or LiquidityScorer(config=config)

        self._logger = get_logger(
            __name__,
            service_name="analytics_liquidity",
            event_type="stop_clusters_detector",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_from_levels(
        self,
        symbol: str,
        timeframe: str,
        levels: Sequence[LiquidityLevel],
        current_price: float,
        candles: Sequence[Any] | None = None,
        orderbook: dict[str, Sequence[Any]] | None = None,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
    ) -> list[StopCluster]:
        """
        Будує stop clusters з liquidity levels.

        Returns
        -------
        list[StopCluster]
            Scored, merged і deduplicated stop clusters.
        """
        exchange = self._normalize_scope_value(exchange, DEFAULT_EXCHANGE)
        market_type = self._normalize_scope_value(market_type, DEFAULT_MARKET_TYPE)
        symbol = self._normalize_symbol(symbol)
        timeframe = self._normalize_timeframe(timeframe)

        if not self._config.enabled:
            self._logger.debug(
                "Stop cluster detection skipped: liquidity module disabled",
                extra={
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                    "timeframe": timeframe,
                },
            )
            return []

        if not symbol or not timeframe:
            raise ValueError("symbol and timeframe are required")

        current_price = safe_float(current_price)
        if current_price <= 0:
            raise ValueError("current_price must be > 0")

        levels_list = list(levels)
        if not levels_list:
            return []

        self._scope_levels(
            levels=levels_list,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        candidates = self._build_candidates(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            levels=levels_list,
            candles=list(candles or []),
            orderbook=orderbook,
        )

        if not candidates:
            return []

        merged_candidates = self._merge_candidates(candidates)

        clusters = self._build_clusters(
            candidates=merged_candidates,
            current_price=current_price,
        )

        clusters.sort(
            key=lambda cluster: (
                cluster.exchange,
                cluster.market_type,
                cluster.symbol,
                cluster.timeframe,
                cluster.side.value,
                cluster.center_price,
            )
        )

        self._logger.info(
            "Stop clusters detected",
            extra={
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "input_levels": len(levels_list),
                "candidates": len(candidates),
                "merged_candidates": len(merged_candidates),
                "clusters": len(clusters),
                "swept_clusters": sum(1 for cluster in clusters if cluster.is_swept()),
            },
        )

        return clusters

    def detect_from_equal_levels(
        self,
        symbol: str,
        timeframe: str,
        equal_levels: Sequence[EqualLevel],
        current_price: float,
        candles: Sequence[Any] | None = None,
        orderbook: dict[str, Sequence[Any]] | None = None,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
    ) -> list[StopCluster]:
        """
        Зручний wrapper для EqualLevel sequence.
        """
        return self.detect_from_levels(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            levels=list(equal_levels),
            current_price=current_price,
            candles=candles,
            orderbook=orderbook,
        )

    def build_stop_zones(
        self,
        levels: Sequence[LiquidityLevel],
    ) -> list[tuple[float, float]]:
        """
        Повертає сирі price zones без scoring.

        Корисно для dashboard/debug/strategy pre-filter.
        """
        ranges: list[tuple[float, float]] = []

        for level in levels:
            if not self._should_use_level(level):
                continue

            candidate = self._level_to_candidate(
                exchange=level.exchange,
                market_type=level.market_type,
                symbol=level.symbol,
                timeframe=level.timeframe,
                level=level,
            )

            if candidate is None:
                continue

            ranges.append((candidate.low_price, candidate.high_price))

        return merge_price_ranges(
            ranges=ranges,
            merge_distance_pct=self._config.cluster_merge_distance_pct,
        )

    # ------------------------------------------------------------------
    # Candidate building
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
        levels: Sequence[LiquidityLevel],
        candles: Sequence[Any],
        orderbook: dict[str, Sequence[Any]] | None,
    ) -> list[StopClusterCandidate]:
        candidates: list[StopClusterCandidate] = []

        for level in levels:
            if not self._should_use_level(level):
                continue

            candidate = self._level_to_candidate(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
                level=level,
            )

            if candidate is None:
                continue

            candidate.partial_sweep_factor = self._calculate_partial_sweep_factor(level)

            if self._config.use_volume_in_scoring:
                candidate.volume_score = self._calculate_volume_score(
                    level=level,
                    candles=candles,
                )

            if self._config.use_orderbook_in_stop_clusters:
                candidate.orderbook_score = self._calculate_orderbook_score(
                    candidate=candidate,
                    orderbook=orderbook,
                )

            if self._config.use_reaction_strength_in_scoring:
                candidate.compression_score = self._calculate_compression_score(
                    level=level,
                    candles=candles,
                )

            if self._config.use_time_decay:
                candidate.time_decay_factor = self._calculate_time_decay_factor(level)

            candidates.append(self._apply_candidate_adjustments(candidate))

        return candidates

    def _should_use_level(self, level: LiquidityLevel) -> bool:
        """
        Вирішує, чи можна використовувати рівень як source для stop cluster.

        Invalidated/expired рівні не використовуються.
        Fully/partially swept рівні можна використовувати, але вони переносять
        swept_at у StopCluster і знижують confidence/density через sweep factor.
        Це потрібно для коректної stop-hunt reversal логіки в strategy layer.
        """
        if level.price <= 0:
            return False

        if self._level_is_invalidated_or_expired(level):
            return False

        if not level.is_active() and not level.is_partially_swept() and not level.is_swept():
            return False

        return level.level_type in {
            LiquidityLevelType.EQUAL_HIGHS,
            LiquidityLevelType.EQUAL_LOWS,
            LiquidityLevelType.SWING_HIGH,
            LiquidityLevelType.SWING_LOW,
            LiquidityLevelType.RANGE_HIGH,
            LiquidityLevelType.RANGE_LOW,
            LiquidityLevelType.ORDERBOOK_WALL,
            LiquidityLevelType.LIQUIDATION_ZONE,
        }

    def _level_is_invalidated_or_expired(self, level: LiquidityLevel) -> bool:
        if getattr(level, "invalidated_at", None) is not None:
            return True

        if getattr(level, "expired_at", None) is not None:
            return True

        is_invalidated = getattr(level, "is_invalidated", None)
        if callable(is_invalidated) and is_invalidated():
            return True

        is_expired = getattr(level, "is_expired", None)
        if callable(is_expired) and is_expired():
            return True

        return False

    def _level_to_candidate(
        self,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
        level: LiquidityLevel,
    ) -> StopClusterCandidate | None:
        padding_pct = self._resolve_padding_pct(level)

        side = self._resolve_candidate_side(level)
        if side == LiquiditySide.UNKNOWN:
            return None

        if side == LiquiditySide.BUY_SIDE:
            low_price = level.price
            high_price = level.price * (1.0 + padding_pct)

        elif side == LiquiditySide.SELL_SIDE:
            low_price = level.price * (1.0 - padding_pct)
            high_price = level.price

        else:
            low_price = level.price * (1.0 - padding_pct)
            high_price = level.price * (1.0 + padding_pct)

        swept_at = self._resolve_level_sweep_timestamp(level)
        invalidated_at = (
            self._normalize_timestamp(level.invalidated_at)
            if level.invalidated_at
            else None
        )

        return StopClusterCandidate(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            low_price=low_price,
            high_price=high_price,
            source_levels=[level],
            created_at=self._normalize_timestamp(level.first_seen_at),
            updated_at=self._normalize_timestamp(level.last_seen_at),
            swept_at=swept_at,
            invalidated_at=invalidated_at,
            swept_source_count=1 if level.is_swept() else 0,
            partially_swept_source_count=1 if level.is_partially_swept() else 0,
        )

    def _resolve_candidate_side(self, level: LiquidityLevel) -> LiquiditySide:
        if level.side in {LiquiditySide.BUY_SIDE, LiquiditySide.SELL_SIDE}:
            return level.side

        if level.side == LiquiditySide.BOTH:
            return LiquiditySide.BOTH

        inferred_side = level.level_type.infer_side()

        if inferred_side in {
            LiquiditySide.BUY_SIDE,
            LiquiditySide.SELL_SIDE,
            LiquiditySide.BOTH,
        }:
            return inferred_side

        return LiquiditySide.UNKNOWN

    def _resolve_padding_pct(self, level: LiquidityLevel) -> float:
        base_padding = self._config.stop_cluster_padding_pct
        confidence = clamp(level.confidence, 0.0, 1.0)

        scale = 0.75 + 0.50 * confidence
        padding = base_padding * scale

        if level.is_swept():
            padding *= 1.25
        elif level.is_partially_swept():
            padding *= 1.15

        return max(padding, base_padding * 0.5)

    # ------------------------------------------------------------------
    # Sweep helpers
    # ------------------------------------------------------------------

    def _level_is_sweep_affected(self, level: LiquidityLevel) -> bool:
        return level.is_swept() or level.is_partially_swept()

    def _resolve_level_sweep_timestamp(
        self,
        level: LiquidityLevel,
    ) -> datetime | None:
        """
        Повертає timestamp sweep-контексту.

        Якщо level має sweep_status, але swept_at не заповнений,
        використовує last_seen_at / first_seen_at як fallback.
        """
        if not self._level_is_sweep_affected(level):
            return None

        return (
            self._normalize_timestamp(level.swept_at)
            or self._normalize_timestamp(level.last_seen_at)
            or self._normalize_timestamp(level.first_seen_at)
        )

    # ------------------------------------------------------------------
    # Candidate enhancement
    # ------------------------------------------------------------------

    def _calculate_partial_sweep_factor(self, level: LiquidityLevel) -> float:
        if not self._config.use_partial_sweep_penalty:
            return 1.0

        if level.is_swept():
            return 0.55

        if level.is_partially_swept():
            return 0.82

        return 1.0

    def _calculate_volume_score(
        self,
        level: LiquidityLevel,
        candles: Sequence[Any],
    ) -> float:
        if len(candles) < 5:
            return 0.0

        tolerance_value = safe_float(
            level.metadata.get("tolerance_pct"),
            default=self._config.equal_level_tolerance_pct,
        )

        tolerance_pct = max(
            tolerance_value,
            self._config.equal_level_tolerance_pct,
        ) * 2.0

        nearby_candles = self._find_candles_near_level(
            candles=candles,
            level_price=level.price,
            tolerance_pct=tolerance_pct,
        )

        if not nearby_candles:
            return 0.0

        near_volumes = [
            volume
            for volume in (
                safe_float(get_candle_volume(candle), default=0.0)
                for candle in nearby_candles
            )
            if volume > 0
        ]

        all_volumes = [
            volume
            for volume in (
                safe_float(get_candle_volume(candle), default=0.0)
                for candle in candles
            )
            if volume > 0
        ]

        if not near_volumes or not all_volumes:
            return 0.0

        avg_near = safe_mean(near_volumes)
        avg_all = safe_mean(all_volumes)

        if avg_all <= 0:
            return 0.0

        volume_ratio = avg_near / avg_all
        return clamp((volume_ratio - 1.0) / 1.5, 0.0, 1.0)

    def _calculate_orderbook_score(
        self,
        candidate: StopClusterCandidate,
        orderbook: dict[str, Sequence[Any]] | None,
    ) -> float:
        if not orderbook:
            return 0.0

        if candidate.side == LiquiditySide.BUY_SIDE:
            relevant_levels = self._parse_orderbook_levels(
                orderbook.get("asks", []),
                side="ask",
            )
        elif candidate.side == LiquiditySide.SELL_SIDE:
            relevant_levels = self._parse_orderbook_levels(
                orderbook.get("bids", []),
                side="bid",
            )
        else:
            relevant_levels = [
                *self._parse_orderbook_levels(orderbook.get("bids", []), side="bid"),
                *self._parse_orderbook_levels(orderbook.get("asks", []), side="ask"),
            ]

        if not relevant_levels:
            return 0.0

        all_sizes = [level.size for level in relevant_levels if level.size > 0]
        nearby_sizes = [
            level.size
            for level in relevant_levels
            if level.size > 0
            and candidate.low_price <= level.price <= candidate.high_price
        ]

        if not all_sizes or not nearby_sizes:
            return 0.0

        avg_all = safe_mean(all_sizes)
        max_nearby = max(nearby_sizes)

        if avg_all <= 0:
            return 0.0

        wall_ratio = max_nearby / avg_all
        return clamp((wall_ratio - 1.0) / 4.0, 0.0, 1.0)

    def _calculate_compression_score(
        self,
        level: LiquidityLevel,
        candles: Sequence[Any],
    ) -> float:
        if len(candles) < 10:
            return 0.0

        pivot_indexes = self._extract_int_list(
            level.metadata.get("pivot_indexes")
        )

        if not pivot_indexes:
            return 0.0

        last_index = max(pivot_indexes)
        if last_index < 5:
            return 0.0

        lookback_slice = candles[max(0, last_index - 5) : last_index]

        if len(lookback_slice) < 4:
            return 0.0

        ranges_pct: list[float] = []

        for candle in lookback_slice:
            high = get_candle_high(candle)
            low = get_candle_low(candle)
            close = get_candle_close(candle)

            if close <= 0 or high < low:
                continue

            ranges_pct.append((high - low) / abs(close))

        if len(ranges_pct) < 3:
            return 0.0

        midpoint_index = len(ranges_pct) // 2
        first_half = ranges_pct[:midpoint_index]
        second_half = ranges_pct[midpoint_index:]

        if not first_half or not second_half:
            return 0.0

        avg_first = safe_mean(first_half)
        avg_second = safe_mean(second_half)

        if avg_first <= 0:
            return 0.0

        compression_ratio = 1.0 - (avg_second / avg_first)
        return clamp(compression_ratio, 0.0, 1.0)

    def _calculate_time_decay_factor(self, level: LiquidityLevel) -> float:
        anchor = self._normalize_timestamp(level.last_seen_at or level.first_seen_at)

        if anchor is None:
            return 1.0

        now = datetime.now(timezone.utc)
        age_hours = max((now - anchor).total_seconds(), 0.0) / 3600.0

        if age_hours <= 6:
            return 1.0

        if age_hours <= 24:
            return 0.95

        if age_hours <= 72:
            return 0.88

        if age_hours <= 168:
            return 0.78

        return 0.65

    def _apply_candidate_adjustments(
        self,
        candidate: StopClusterCandidate,
    ) -> StopClusterCandidate:
        low_price = candidate.low_price
        high_price = candidate.high_price

        if candidate.partial_sweep_factor < 1.0:
            extra_width = candidate.width() * (
                1.0 - candidate.partial_sweep_factor
            ) * 0.75

            if candidate.side == LiquiditySide.BUY_SIDE:
                high_price += extra_width
            elif candidate.side == LiquiditySide.SELL_SIDE:
                low_price -= extra_width
            else:
                low_price -= extra_width
                high_price += extra_width

        return StopClusterCandidate(
            exchange=candidate.exchange,
            market_type=candidate.market_type,
            symbol=candidate.symbol,
            timeframe=candidate.timeframe,
            side=candidate.side,
            low_price=low_price,
            high_price=high_price,
            source_levels=list(candidate.source_levels),
            created_at=candidate.created_at,
            updated_at=candidate.updated_at,
            swept_at=candidate.swept_at,
            invalidated_at=candidate.invalidated_at,
            volume_score=candidate.volume_score,
            orderbook_score=candidate.orderbook_score,
            compression_score=candidate.compression_score,
            time_decay_factor=candidate.time_decay_factor,
            partial_sweep_factor=candidate.partial_sweep_factor,
            swept_source_count=candidate.swept_source_count,
            partially_swept_source_count=candidate.partially_swept_source_count,
        )

    # ------------------------------------------------------------------
    # Merge candidates
    # ------------------------------------------------------------------

    def _merge_candidates(
        self,
        candidates: Sequence[StopClusterCandidate],
    ) -> list[StopClusterCandidate]:
        grouped: dict[tuple[str, str, str, str, LiquiditySide], list[StopClusterCandidate]] = {}

        for candidate in candidates:
            key = (
                candidate.exchange,
                candidate.market_type,
                candidate.symbol,
                candidate.timeframe,
                candidate.side,
            )
            grouped.setdefault(key, []).append(candidate)

        merged: list[StopClusterCandidate] = []

        for side_candidates in grouped.values():
            merged.extend(self._merge_candidates_by_side(side_candidates))

        return merged

    def _merge_candidates_by_side(
        self,
        candidates: Sequence[StopClusterCandidate],
    ) -> list[StopClusterCandidate]:
        if not candidates:
            return []

        sorted_candidates = sorted(
            candidates,
            key=lambda candidate: candidate.low_price,
        )

        merged: list[StopClusterCandidate] = [sorted_candidates[0]]

        for candidate in sorted_candidates[1:]:
            current = merged[-1]

            if self._should_merge(current, candidate):
                merged[-1] = self._merge_two_candidates(current, candidate)
            else:
                merged.append(candidate)

        return merged

    def _should_merge(
        self,
        left: StopClusterCandidate,
        right: StopClusterCandidate,
    ) -> bool:
        if not left.same_scope(right):
            return False

        if left.side != right.side:
            return False

        if left.overlaps(right):
            return True

        gap = right.low_price - left.high_price
        reference_price = max(left.center_price, right.center_price, 1e-12)
        gap_pct = gap / reference_price

        return gap_pct <= self._config.cluster_merge_distance_pct

    def _merge_two_candidates(
        self,
        left: StopClusterCandidate,
        right: StopClusterCandidate,
    ) -> StopClusterCandidate:
        source_levels = self._deduplicate_source_levels(
            [*left.source_levels, *right.source_levels]
        )

        swept_at = self._max_datetime(left.swept_at, right.swept_at)

        if swept_at is None:
            swept_timestamps = [
                self._resolve_level_sweep_timestamp(level)
                for level in source_levels
                if self._level_is_sweep_affected(level)
            ]
            swept_timestamps = [ts for ts in swept_timestamps if ts is not None]
            swept_at = max(swept_timestamps) if swept_timestamps else None

        return StopClusterCandidate(
            exchange=left.exchange,
            market_type=left.market_type,
            symbol=left.symbol,
            timeframe=left.timeframe,
            side=left.side,
            low_price=min(left.low_price, right.low_price),
            high_price=max(left.high_price, right.high_price),
            source_levels=source_levels,
            created_at=self._min_datetime(left.created_at, right.created_at),
            updated_at=self._max_datetime(left.updated_at, right.updated_at),
            swept_at=swept_at,
            invalidated_at=self._max_datetime(left.invalidated_at, right.invalidated_at),
            volume_score=max(left.volume_score, right.volume_score),
            orderbook_score=max(left.orderbook_score, right.orderbook_score),
            compression_score=max(left.compression_score, right.compression_score),
            time_decay_factor=min(left.time_decay_factor, right.time_decay_factor),
            partial_sweep_factor=min(
                left.partial_sweep_factor,
                right.partial_sweep_factor,
            ),
            swept_source_count=sum(1 for level in source_levels if level.is_swept()),
            partially_swept_source_count=sum(
                1 for level in source_levels if level.is_partially_swept()
            ),
        )

    # ------------------------------------------------------------------
    # Build final clusters
    # ------------------------------------------------------------------

    def _build_clusters(
        self,
        candidates: Sequence[StopClusterCandidate],
        current_price: float,
    ) -> list[StopCluster]:
        clusters: list[StopCluster] = []

        for candidate in candidates:
            cluster = self._candidate_to_cluster(
                candidate=candidate,
                current_price=current_price,
            )

            if cluster.confidence < self._config.min_confidence:
                # Keep genuinely swept clusters if they are close enough and not zero-quality,
                # because stop-hunt reversal strategies may need them as context.
                if not cluster.is_swept() or cluster.confidence <= 0.0:
                    continue

            clusters.append(cluster)

        return self._deduplicate_clusters(clusters)

    def _candidate_to_cluster(
        self,
        candidate: StopClusterCandidate,
        current_price: float,
    ) -> StopCluster:
        base_density = self._scorer.estimate_stop_density(
            candidate.source_levels
        )

        enhanced_density = self._enhance_density(
            base_density=base_density,
            volume_score=candidate.volume_score,
            orderbook_score=candidate.orderbook_score,
            compression_score=candidate.compression_score,
            partial_sweep_factor=candidate.partial_sweep_factor,
            time_decay_factor=candidate.time_decay_factor,
        )

        touches_count = sum(
            max(level.touches_count, 1)
            for level in candidate.source_levels
        )

        dominant_level_type = self._resolve_dominant_level_type(
            candidate.source_levels
        )

        swept_source_count = sum(
            1 for level in candidate.source_levels if level.is_swept()
        )
        partially_swept_source_count = sum(
            1 for level in candidate.source_levels if level.is_partially_swept()
        )

        cluster = StopCluster(
            exchange=candidate.exchange,
            market_type=candidate.market_type,
            symbol=candidate.symbol,
            timeframe=candidate.timeframe,
            side=candidate.side,
            low_price=candidate.low_price,
            high_price=candidate.high_price,
            center_price=candidate.center_price,
            confidence=0.0,
            estimated_stop_density=enhanced_density,
            touches_count=touches_count,
            source_level_type=dominant_level_type,
            created_at=candidate.created_at,
            updated_at=candidate.updated_at,
            swept_at=candidate.swept_at,
            invalidated_at=candidate.invalidated_at,
            source_levels=list(candidate.source_levels),
            metadata={
                "detector": self.__class__.__name__,
                "exchange": candidate.exchange,
                "market_type": candidate.market_type,
                "source_count": len(candidate.source_levels),
                "source_keys": [level.key for level in candidate.source_levels],
                "source_prices": [
                    level.price
                    for level in candidate.source_levels
                ],
                "source_confidences": [
                    level.confidence
                    for level in candidate.source_levels
                ],
                "volume_score": candidate.volume_score,
                "orderbook_score": candidate.orderbook_score,
                "compression_score": candidate.compression_score,
                "time_decay_factor": candidate.time_decay_factor,
                "partial_sweep_factor": candidate.partial_sweep_factor,
                "swept_source_count": swept_source_count,
                "partially_swept_source_count": partially_swept_source_count,
                "is_swept_cluster": candidate.is_swept(),
            },
        )

        base_score = self._scorer.score_stop_cluster(
            cluster=cluster,
            current_price=current_price,
        )

        final_score = self._enhance_cluster_score(
            base_score=base_score,
            volume_score=candidate.volume_score,
            orderbook_score=candidate.orderbook_score,
            compression_score=candidate.compression_score,
            partial_sweep_factor=candidate.partial_sweep_factor,
            time_decay_factor=candidate.time_decay_factor,
        )

        cluster.confidence = final_score
        cluster.strength = self._scorer.classify_cluster_strength(
            cluster.confidence
        )

        return cluster

    def _enhance_density(
        self,
        base_density: float,
        volume_score: float,
        orderbook_score: float,
        compression_score: float,
        partial_sweep_factor: float,
        time_decay_factor: float,
    ) -> float:
        density = clamp(base_density, 0.0, 1.0)

        density += 0.18 * clamp(volume_score, 0.0, 1.0)
        density += 0.22 * clamp(orderbook_score, 0.0, 1.0)
        density += 0.14 * clamp(compression_score, 0.0, 1.0)

        density *= clamp(partial_sweep_factor, 0.3, 1.0)
        density *= clamp(time_decay_factor, 0.4, 1.0)

        return clamp(density, 0.0, 1.0)

    def _enhance_cluster_score(
        self,
        base_score: float,
        volume_score: float,
        orderbook_score: float,
        compression_score: float,
        partial_sweep_factor: float,
        time_decay_factor: float,
    ) -> float:
        score = clamp(base_score, 0.0, 1.0)

        score += 0.10 * clamp(volume_score, 0.0, 1.0)
        score += 0.14 * clamp(orderbook_score, 0.0, 1.0)
        score += 0.08 * clamp(compression_score, 0.0, 1.0)

        score *= clamp(partial_sweep_factor, 0.35, 1.0)
        score *= clamp(time_decay_factor, 0.5, 1.0)

        return clamp(score, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Deduplication / classification
    # ------------------------------------------------------------------

    def _resolve_dominant_level_type(
        self,
        source_levels: Sequence[LiquidityLevel],
    ) -> LiquidityLevelType:
        if not source_levels:
            return LiquidityLevelType.STOP_CLUSTER

        priority = (
            LiquidityLevelType.EQUAL_HIGHS,
            LiquidityLevelType.EQUAL_LOWS,
            LiquidityLevelType.SWING_HIGH,
            LiquidityLevelType.SWING_LOW,
            LiquidityLevelType.RANGE_HIGH,
            LiquidityLevelType.RANGE_LOW,
            LiquidityLevelType.ORDERBOOK_WALL,
            LiquidityLevelType.LIQUIDATION_ZONE,
        )

        type_counts: dict[LiquidityLevelType, int] = {}

        for level in source_levels:
            type_counts[level.level_type] = type_counts.get(level.level_type, 0) + 1

        for level_type in priority:
            if level_type in type_counts:
                return level_type

        return source_levels[0].level_type

    def _deduplicate_clusters(
        self,
        clusters: Sequence[StopCluster],
    ) -> list[StopCluster]:
        if not clusters:
            return []

        sorted_clusters = sorted(
            clusters,
            key=lambda cluster: (
                cluster.exchange,
                cluster.market_type,
                cluster.symbol,
                cluster.timeframe,
                cluster.side.value,
                cluster.center_price,
            ),
        )

        result: list[StopCluster] = [sorted_clusters[0]]

        for cluster in sorted_clusters[1:]:
            previous = result[-1]

            same_scope = (
                cluster.exchange == previous.exchange
                and cluster.market_type == previous.market_type
                and cluster.symbol == previous.symbol
                and cluster.timeframe == previous.timeframe
            )

            if not same_scope or cluster.side != previous.side:
                result.append(cluster)
                continue

            if self._clusters_are_near(previous, cluster):
                if self._is_better_cluster(cluster, previous):
                    result[-1] = cluster
                continue

            result.append(cluster)

        return result

    def _clusters_are_near(
        self,
        left: StopCluster,
        right: StopCluster,
    ) -> bool:
        if left.overlaps(right):
            return True

        return (
            pct_distance(left.center_price, right.center_price)
            <= self._config.cluster_merge_distance_pct
        )

    def _is_better_cluster(
        self,
        candidate: StopCluster,
        current: StopCluster,
    ) -> bool:
        candidate_score = (
            1 if candidate.is_swept() else 0,
            candidate.confidence,
            candidate.estimated_stop_density,
            candidate.touches_count,
            -candidate.width_pct(),
        )
        current_score = (
            1 if current.is_swept() else 0,
            current.confidence,
            current.estimated_stop_density,
            current.touches_count,
            -current.width_pct(),
        )

        return candidate_score > current_score

    def _deduplicate_source_levels(
        self,
        levels: Sequence[LiquidityLevel],
    ) -> list[LiquidityLevel]:
        """
        Deduplicate source levels without losing swept / partially swept context.

        Важливо:
        - level.key містить exchange + market_type + symbol + timeframe;
        - active і swept level на тій самій ціні можуть мати однаковий key;
        - для reversal strategies swept/partial swept source важливіший за active
          source з вищим confidence.
        """
        result: dict[str, LiquidityLevel] = {}

        for level in levels:
            existing = result.get(level.key)

            if existing is None:
                result[level.key] = level
                continue

            if self._source_level_rank(level) > self._source_level_rank(existing):
                result[level.key] = level

        return list(result.values())

    def _source_level_rank(self, level: LiquidityLevel) -> tuple[int, int, float, int, int]:
        """
        Ranking для deduplication source levels.

        Пріоритет:
        1. swept рівень;
        2. partially swept рівень;
        3. active рівень;
        4. explicit swept_at;
        5. confidence;
        6. touches_count;
        7. reaction_count.
        """
        if level.is_swept():
            sweep_rank = 3
        elif level.is_partially_swept():
            sweep_rank = 2
        elif level.is_active():
            sweep_rank = 1
        else:
            sweep_rank = 0

        return (
            sweep_rank,
            1 if level.swept_at is not None else 0,
            clamp(level.confidence, 0.0, 1.0),
            max(level.touches_count, 0),
            max(level.reaction_count, 0),
        )

    # ------------------------------------------------------------------
    # Candle / orderbook helpers
    # ------------------------------------------------------------------

    def _find_candles_near_level(
        self,
        candles: Sequence[Any],
        level_price: float,
        tolerance_pct: float,
    ) -> list[Any]:
        result: list[Any] = []

        for candle in candles:
            high = get_candle_high(candle)
            low = get_candle_low(candle)

            if low <= level_price <= high:
                result.append(candle)
                continue

            if pct_distance(high, level_price) <= tolerance_pct:
                result.append(candle)
                continue

            if pct_distance(low, level_price) <= tolerance_pct:
                result.append(candle)

        return result

    def _parse_orderbook_levels(
        self,
        levels: Sequence[Any],
        side: str,
    ) -> list[OrderbookLevel]:
        parsed: list[OrderbookLevel] = []

        for item in levels:
            price = self._extract_orderbook_price(item)
            size = self._extract_orderbook_size(item)

            if price is None or size is None:
                continue

            if price <= 0 or size <= 0:
                continue

            parsed.append(
                OrderbookLevel(
                    price=price,
                    size=size,
                    side=side,
                )
            )

        return parsed

    def _extract_orderbook_price(self, item: Any) -> float | None:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            value = item[0]
        else:
            value = get_first_value(
                item,
                ("price", "p", "px"),
            )

        price = safe_float(value, default=0.0)
        return price if price > 0 else None

    def _extract_orderbook_size(self, item: Any) -> float | None:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            value = item[1]
        else:
            value = get_first_value(
                item,
                ("size", "qty", "quantity", "q", "amount"),
            )

        size = safe_float(value, default=0.0)
        return size if size > 0 else None

    def _extract_int_list(self, value: Any) -> list[int]:
        if not isinstance(value, (list, tuple)):
            return []

        result: list[int] = []

        for item in value:
            if isinstance(item, bool):
                continue

            if isinstance(item, int):
                result.append(item)
                continue

            if isinstance(item, float) and item.is_integer():
                result.append(int(item))

        return result

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    def _scope_levels(
        self,
        *,
        levels: Sequence[LiquidityLevel],
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
    ) -> None:
        for level in levels:
            level.exchange = exchange
            level.market_type = market_type
            level.symbol = symbol
            level.timeframe = timeframe

    @staticmethod
    def _normalize_scope_value(value: Any, default: str) -> str:
        normalized = str(value or default).strip()
        return normalized if normalized else default

    @staticmethod
    def _normalize_symbol(value: Any) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _normalize_timeframe(value: Any) -> str:
        return str(value or "").strip()

    # ------------------------------------------------------------------
    # Datetime helpers
    # ------------------------------------------------------------------

    def _normalize_timestamp(
        self,
        value: datetime | None,
    ) -> datetime | None:
        if value is None:
            return None

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)

    def _min_datetime(
        self,
        left: datetime | None,
        right: datetime | None,
    ) -> datetime | None:
        left = self._normalize_timestamp(left)
        right = self._normalize_timestamp(right)

        if left is None:
            return right

        if right is None:
            return left

        return min(left, right)

    def _max_datetime(
        self,
        left: datetime | None,
        right: datetime | None,
    ) -> datetime | None:
        left = self._normalize_timestamp(left)
        right = self._normalize_timestamp(right)

        if left is None:
            return right

        if right is None:
            return left

        return max(left, right)