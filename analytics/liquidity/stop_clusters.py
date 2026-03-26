from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from core.logger import get_logger

from .config import LiquidityConfig
from .enums import LiquidityLevelType, LiquiditySide, SweepStatus
from .models import EqualLevel, LiquidityLevel, StopCluster
from .scoring import LiquidityScorer
from .utils import clamp, merge_price_ranges, midpoint, pct_distance


@dataclass(slots=True)
class OrderbookLevel:
    price: float
    size: float
    side: str  # "bid" | "ask"


@dataclass(slots=True)
class StopClusterCandidate:
    """
    Внутрішня модель для формування stop cluster.
    """

    symbol: str
    timeframe: str
    side: LiquiditySide
    low_price: float
    high_price: float
    source_levels: list[LiquidityLevel] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    # додаткові фактори
    volume_score: float = 0.0
    orderbook_score: float = 0.0
    compression_score: float = 0.0
    time_decay_factor: float = 1.0
    partial_sweep_factor: float = 1.0

    @property
    def center_price(self) -> float:
        return midpoint(self.low_price, self.high_price)

    def width(self) -> float:
        return max(0.0, self.high_price - self.low_price)


class StopClustersDetector:
    """
    Детектор stop clusters.

    Підтримує:
    - partial sweep awareness
    - volume-aware density
    - orderbook-aware enhancement
    - range compression factor
    - time decay
    """

    def __init__(
        self,
        config: LiquidityConfig,
        scorer: LiquidityScorer | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self._config = config
        self._config.validate()

        self._scorer = scorer or LiquidityScorer(config=config)
        self._event_bus = event_bus
        self._logger = get_logger(__name__, service_name="liquidity")

    def detect_from_levels(
        self,
        symbol: str,
        timeframe: str,
        levels: Sequence[LiquidityLevel],
        current_price: float,
        candles: Sequence[Any] | None = None,
        orderbook: dict[str, Sequence[Any]] | None = None,
    ) -> list[StopCluster]:
        """
        Основний метод побудови stop clusters.

        Parameters
        ----------
        symbol:
            Торговий символ.
        timeframe:
            Таймфрейм.
        levels:
            Список liquidity levels.
        current_price:
            Поточна ціна.
        candles:
            Свічки для аналізу volume/compression.
        orderbook:
            Опціонально:
            {
                "bids": [...],
                "asks": [...]
            }

            Кожен елемент може бути:
            - dict {"price": ..., "size": ...}
            - tuple/list [price, size]
            - object.price / object.size
        """
        if not levels:
            return []

        candidates = self._build_candidates(
            symbol=symbol,
            timeframe=timeframe,
            levels=levels,
            candles=candles,
            orderbook=orderbook,
        )
        if not candidates:
            return []

        merged_candidates = self._merge_candidates(candidates)
        clusters = self._build_clusters(
            candidates=merged_candidates,
            current_price=current_price,
        )
        clusters.sort(key=lambda c: c.center_price)

        self._logger.info(
            "Stop clusters detected",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "input_levels": len(levels),
                "candidates": len(candidates),
                "merged_candidates": len(merged_candidates),
                "clusters": len(clusters),
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
    ) -> list[StopCluster]:
        return self.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=equal_levels,
            current_price=current_price,
            candles=candles,
            orderbook=orderbook,
        )

    def build_stop_zones(
        self,
        levels: Sequence[LiquidityLevel],
    ) -> list[tuple[float, float]]:
        ranges: list[tuple[float, float]] = []

        for level in levels:
            candidate = self._level_to_candidate(
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

    def _build_candidates(
        self,
        symbol: str,
        timeframe: str,
        levels: Sequence[LiquidityLevel],
        candles: Sequence[Any] | None,
        orderbook: dict[str, Sequence[Any]] | None,
    ) -> list[StopClusterCandidate]:
        candidates: list[StopClusterCandidate] = []

        for level in levels:
            if not level.is_active() and level.sweep_status != SweepStatus.PARTIALLY_SWEPT:
                continue

            candidate = self._level_to_candidate(
                symbol=symbol,
                timeframe=timeframe,
                level=level,
            )
            if candidate is None:
                continue

            candidate.partial_sweep_factor = self._calculate_partial_sweep_factor(level)
            candidate.volume_score = self._calculate_volume_score(
                level=level,
                candles=candles,
            )
            candidate.orderbook_score = self._calculate_orderbook_score(
                candidate=candidate,
                orderbook=orderbook,
            )
            candidate.compression_score = self._calculate_compression_score(
                level=level,
                candles=candles,
            )
            candidate.time_decay_factor = self._calculate_time_decay_factor(level)

            candidate = self._apply_candidate_adjustments(candidate)
            candidates.append(candidate)

        return candidates

    def _level_to_candidate(
        self,
        symbol: str,
        timeframe: str,
        level: LiquidityLevel,
    ) -> StopClusterCandidate | None:
        padding_pct = self._resolve_padding_pct(level)

        if level.level_type in {
            LiquidityLevelType.EQUAL_HIGHS,
            LiquidityLevelType.SWING_HIGH,
            LiquidityLevelType.RANGE_HIGH,
        }:
            low_price = level.price
            high_price = level.price * (1.0 + padding_pct)
            side = LiquiditySide.BUY_SIDE

        elif level.level_type in {
            LiquidityLevelType.EQUAL_LOWS,
            LiquidityLevelType.SWING_LOW,
            LiquidityLevelType.RANGE_LOW,
        }:
            low_price = level.price * (1.0 - padding_pct)
            high_price = level.price
            side = LiquiditySide.SELL_SIDE

        else:
            return None

        return StopClusterCandidate(
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            low_price=min(low_price, high_price),
            high_price=max(low_price, high_price),
            source_levels=[level],
            created_at=level.first_seen_at,
            updated_at=level.last_seen_at,
        )

    def _resolve_padding_pct(self, level: LiquidityLevel) -> float:
        base_padding = self._config.stop_cluster_padding_pct
        confidence = clamp(level.confidence, 0.0, 1.0)

        scale = 0.75 + 0.50 * confidence
        padding = base_padding * scale

        return max(padding, base_padding * 0.5)

    def _calculate_partial_sweep_factor(self, level: LiquidityLevel) -> float:
        """
        Частково sweeped рівень:
        - stop cluster залишається релевантним
        - але трохи слабшає
        - іноді зона трохи розширюється, бо ліквідність стала "розмазана"
        """
        if level.sweep_status == SweepStatus.PARTIALLY_SWEPT:
            return 0.82
        if level.sweep_status == SweepStatus.SWEPT:
            return 0.45
        return 1.0

    def _calculate_volume_score(
        self,
        level: LiquidityLevel,
        candles: Sequence[Any] | None,
    ) -> float:
        """
        Оцінка volume pressure біля рівня.
        """
        if not candles or len(candles) < 5:
            return 0.0

        level_price = level.price
        window = self._find_candles_near_level(
            candles=candles,
            level_price=level_price,
            tolerance_pct=max(level.metadata.get("tolerance_pct", 0.0), self._config.equal_level_tolerance_pct) * 2.0,
        )
        if not window:
            return 0.0

        near_volumes = [self._get_candle_volume(c) for c in window if self._get_candle_volume(c) is not None]
        all_volumes = [self._get_candle_volume(c) for c in candles if self._get_candle_volume(c) is not None]

        if not near_volumes or not all_volumes:
            return 0.0

        avg_near = sum(near_volumes) / len(near_volumes)
        avg_all = sum(all_volumes) / len(all_volumes)
        if avg_all <= 0:
            return 0.0

        ratio = avg_near / avg_all
        return clamp((ratio - 1.0) / 1.5, 0.0, 1.0)

    def _calculate_orderbook_score(
        self,
        candidate: StopClusterCandidate,
        orderbook: dict[str, Sequence[Any]] | None,
    ) -> float:
        """
        Якщо біля stop-zone є помітна resting liquidity / wall, посилюємо кластер.
        """
        if not orderbook:
            return 0.0

        if candidate.side == LiquiditySide.BUY_SIDE:
            relevant_levels = self._parse_orderbook_levels(orderbook.get("asks", []), side="ask")
        else:
            relevant_levels = self._parse_orderbook_levels(orderbook.get("bids", []), side="bid")

        if not relevant_levels:
            return 0.0

        nearby_sizes: list[float] = []
        all_sizes: list[float] = []

        for level in relevant_levels:
            all_sizes.append(level.size)
            if candidate.low_price <= level.price <= candidate.high_price:
                nearby_sizes.append(level.size)

        if not nearby_sizes or not all_sizes:
            return 0.0

        avg_all = sum(all_sizes) / len(all_sizes)
        max_nearby = max(nearby_sizes)

        if avg_all <= 0:
            return 0.0

        wall_ratio = max_nearby / avg_all
        return clamp((wall_ratio - 1.0) / 4.0, 0.0, 1.0)

    def _calculate_compression_score(
        self,
        level: LiquidityLevel,
        candles: Sequence[Any] | None,
    ) -> float:
        """
        Якщо перед рівнем волатильність стискається, stop hunt стає більш імовірним.
        """
        if not candles or len(candles) < 10:
            return 0.0

        pivot_indexes = level.metadata.get("pivot_indexes")
        if not pivot_indexes:
            return 0.0

        last_idx = max(pivot_indexes)
        if last_idx < 5:
            return 0.0

        lookback_slice = candles[max(0, last_idx - 5):last_idx]
        if len(lookback_slice) < 4:
            return 0.0

        ranges = []
        for candle in lookback_slice:
            high = self._get_candle_high(candle)
            low = self._get_candle_low(candle)
            close = self._get_candle_close(candle)
            if close <= 0:
                continue
            ranges.append((high - low) / close)

        if len(ranges) < 3:
            return 0.0

        first_half = ranges[: len(ranges) // 2]
        second_half = ranges[len(ranges) // 2 :]

        if not first_half or not second_half:
            return 0.0

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        if avg_first <= 0:
            return 0.0

        compression_ratio = 1.0 - (avg_second / avg_first)
        return clamp(compression_ratio, 0.0, 1.0)

    def _calculate_time_decay_factor(self, level: LiquidityLevel) -> float:
        """
        Старі рівні поступово послаблюємо.
        """
        anchor = level.last_seen_at or level.first_seen_at
        if anchor is None:
            return 1.0

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if anchor.tzinfo is not None:
            anchor = anchor.astimezone(timezone.utc).replace(tzinfo=None)

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
        """
        Застосовує:
        - partial sweep → трохи ширша зона, слабший кластер
        - compression/orderbook/volume → можуть посилити кластер
        """
        low_price = candidate.low_price
        high_price = candidate.high_price

        # partial sweep -> зона трохи розширюється
        if candidate.partial_sweep_factor < 1.0:
            extra_width = candidate.width() * (1.0 - candidate.partial_sweep_factor) * 0.75
            if candidate.side == LiquiditySide.BUY_SIDE:
                high_price += extra_width
            elif candidate.side == LiquiditySide.SELL_SIDE:
                low_price -= extra_width

        return StopClusterCandidate(
            symbol=candidate.symbol,
            timeframe=candidate.timeframe,
            side=candidate.side,
            low_price=min(low_price, high_price),
            high_price=max(low_price, high_price),
            source_levels=list(candidate.source_levels),
            created_at=candidate.created_at,
            updated_at=candidate.updated_at,
            volume_score=candidate.volume_score,
            orderbook_score=candidate.orderbook_score,
            compression_score=candidate.compression_score,
            time_decay_factor=candidate.time_decay_factor,
            partial_sweep_factor=candidate.partial_sweep_factor,
        )

    def _merge_candidates(
        self,
        candidates: Sequence[StopClusterCandidate],
    ) -> list[StopClusterCandidate]:
        buy_candidates = [c for c in candidates if c.side == LiquiditySide.BUY_SIDE]
        sell_candidates = [c for c in candidates if c.side == LiquiditySide.SELL_SIDE]

        merged_buy = self._merge_candidates_by_side(buy_candidates)
        merged_sell = self._merge_candidates_by_side(sell_candidates)

        return merged_buy + merged_sell

    def _merge_candidates_by_side(
        self,
        candidates: Sequence[StopClusterCandidate],
    ) -> list[StopClusterCandidate]:
        if not candidates:
            return []

        sorted_candidates = sorted(candidates, key=lambda c: c.low_price)
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
        if left.side != right.side:
            return False

        if right.low_price <= left.high_price:
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
        created_at = left.created_at
        if created_at is None or (right.created_at is not None and right.created_at < created_at):
            created_at = right.created_at

        updated_at = left.updated_at
        if updated_at is None or (right.updated_at is not None and right.updated_at > updated_at):
            updated_at = right.updated_at

        merged_sources = [*left.source_levels, *right.source_levels]

        return StopClusterCandidate(
            symbol=left.symbol,
            timeframe=left.timeframe,
            side=left.side,
            low_price=min(left.low_price, right.low_price),
            high_price=max(left.high_price, right.high_price),
            source_levels=merged_sources,
            created_at=created_at,
            updated_at=updated_at,
            volume_score=max(left.volume_score, right.volume_score),
            orderbook_score=max(left.orderbook_score, right.orderbook_score),
            compression_score=max(left.compression_score, right.compression_score),
            time_decay_factor=min(left.time_decay_factor, right.time_decay_factor),
            partial_sweep_factor=min(left.partial_sweep_factor, right.partial_sweep_factor),
        )

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
                continue

            clusters.append(cluster)

        return self._deduplicate_clusters(clusters)

    def _candidate_to_cluster(
        self,
        candidate: StopClusterCandidate,
        current_price: float,
    ) -> StopCluster:
        base_density = self._scorer.estimate_stop_density(candidate.source_levels)

        enhanced_density = self._enhance_density(
            base_density=base_density,
            volume_score=candidate.volume_score,
            orderbook_score=candidate.orderbook_score,
            compression_score=candidate.compression_score,
            partial_sweep_factor=candidate.partial_sweep_factor,
            time_decay_factor=candidate.time_decay_factor,
        )

        touches_count = sum(max(level.touches_count, 1) for level in candidate.source_levels)
        dominant_level_type = self._resolve_dominant_level_type(candidate.source_levels)

        cluster = StopCluster(
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
            source_levels=list(candidate.source_levels),
            metadata={
                "source_count": len(candidate.source_levels),
                "source_prices": [level.price for level in candidate.source_levels],
                "source_confidences": [level.confidence for level in candidate.source_levels],
                "volume_score": candidate.volume_score,
                "orderbook_score": candidate.orderbook_score,
                "compression_score": candidate.compression_score,
                "time_decay_factor": candidate.time_decay_factor,
                "partial_sweep_factor": candidate.partial_sweep_factor,
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
        cluster.strength = self._scorer.classify_cluster_strength(cluster.confidence)

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
        density = base_density

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
        score = base_score

        score += 0.10 * clamp(volume_score, 0.0, 1.0)
        score += 0.14 * clamp(orderbook_score, 0.0, 1.0)
        score += 0.08 * clamp(compression_score, 0.0, 1.0)

        score *= clamp(partial_sweep_factor, 0.35, 1.0)
        score *= clamp(time_decay_factor, 0.5, 1.0)

        return clamp(score, 0.0, 1.0)

    def _resolve_dominant_level_type(
        self,
        source_levels: Sequence[LiquidityLevel],
    ) -> LiquidityLevelType:
        if not source_levels:
            return LiquidityLevelType.STOP_CLUSTER

        priority = [
            LiquidityLevelType.EQUAL_HIGHS,
            LiquidityLevelType.EQUAL_LOWS,
            LiquidityLevelType.SWING_HIGH,
            LiquidityLevelType.SWING_LOW,
            LiquidityLevelType.RANGE_HIGH,
            LiquidityLevelType.RANGE_LOW,
        ]

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
            key=lambda c: (c.side.value, c.center_price),
        )

        result: list[StopCluster] = [sorted_clusters[0]]

        for cluster in sorted_clusters[1:]:
            prev = result[-1]

            if cluster.side != prev.side:
                result.append(cluster)
                continue

            if self._clusters_are_near(prev, cluster):
                if cluster.confidence > prev.confidence:
                    result[-1] = cluster
            else:
                result.append(cluster)

        return result

    def _clusters_are_near(
        self,
        left: StopCluster,
        right: StopCluster,
    ) -> bool:
        if left.overlaps(right):
            return True

        return pct_distance(left.center_price, right.center_price) <= self._config.cluster_merge_distance_pct

    def _find_candles_near_level(
        self,
        candles: Sequence[Any],
        level_price: float,
        tolerance_pct: float,
    ) -> list[Any]:
        result: list[Any] = []

        for candle in candles:
            high = self._get_candle_high(candle)
            low = self._get_candle_low(candle)

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
            if isinstance(item, dict):
                price = item.get("price")
                size = item.get("size") or item.get("qty") or item.get("quantity")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                price = item[0]
                size = item[1]
            else:
                price = getattr(item, "price", None)
                size = getattr(item, "size", None)

            if price is None or size is None:
                continue

            try:
                parsed.append(
                    OrderbookLevel(
                        price=float(price),
                        size=float(size),
                        side=side,
                    )
                )
            except (TypeError, ValueError):
                continue

        return parsed

    def _get_candle_value(self, candle: Any, field: str, default: Any = None) -> Any:
        if isinstance(candle, dict):
            return candle.get(field, default)
        return getattr(candle, field, default)

    def _get_candle_high(self, candle: Any) -> float:
        value = self._get_candle_value(candle, "high")
        if value is None:
            raise ValueError("Candle must contain 'high'")
        return float(value)

    def _get_candle_low(self, candle: Any) -> float:
        value = self._get_candle_value(candle, "low")
        if value is None:
            raise ValueError("Candle must contain 'low'")
        return float(value)

    def _get_candle_close(self, candle: Any) -> float:
        value = self._get_candle_value(candle, "close")
        if value is None:
            raise ValueError("Candle must contain 'close'")
        return float(value)

    def _get_candle_volume(self, candle: Any) -> float | None:
        value = self._get_candle_value(candle, "volume")
        if value is None:
            return None
        return float(value)