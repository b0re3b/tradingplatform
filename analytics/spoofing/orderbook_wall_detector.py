from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable

from core.event_bus import EventBus
from core.scheduler import Scheduler

from .base import BaseSpoofingDetector
from .config import SpoofingConfig
from .enums import (
    DetectorDecision,
    SpoofingComponent,
    SpoofingPattern,
    SpoofingSide,
)
from .models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    DetectorResult,
    OrderbookLevelSnapshot,
    SpoofingFeatures,
    SpoofingKey,
    TrackedWall,
    WallCandidateContext,
    spoofing_key_to_dict,
)
from .persistence_tracker import PersistenceTracker


class OrderbookWallDetector(BaseSpoofingDetector):
    """
    Detector великих стін у стакані.

    Відповідає за:
    - виявлення аномально великих bid/ask рівнів;
    - порівняння рівня з локальним baseline стакана;
    - оцінку близькості до mid / best quote;
    - побудову первинних spoofing features для подальших detector-ів.

    Correct scope:
        exchange + market_type + symbol + timeframe

    Correct production input flow:
        exchange adapters
            -> market.orderbook
            -> OrderBookCache
            -> market.orderbook.updated
            -> SpoofingAnalyzer
            -> OrderbookWallDetector

    Важливо:
    - detector не визначає spoofing остаточно;
    - не підписується на EventBus;
    - не публікує події;
    - не запускає Scheduler jobs;
    - не читає exchange adapters напряму;
    - працює з normalized OrderbookLevelSnapshot;
    - повертає тільки DetectorResult або None.
    """

    component = SpoofingComponent.ORDERBOOK_WALL_DETECTOR

    def __init__(
        self,
        *,
        event_bus: EventBus | None,
        scheduler: Scheduler | None,
        config: SpoofingConfig,
        persistence_tracker: PersistenceTracker | None = None,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config,
        )
        self.persistence_tracker = persistence_tracker

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def analyze(
        self,
        snapshot: OrderbookLevelSnapshot,
        *,
        baseline_size: float | None = None,
        tracked_wall: TrackedWall | None = None,
    ) -> DetectorResult | None:
        """
        Аналізує один конкретний normalized orderbook level як wall candidate.

        Production snapshots мають походити з OrderBookCache /
        market.orderbook.updated, а не напряму з raw exchange payload.
        """
        candidate = self._evaluate_snapshot_candidate(
            snapshot=snapshot,
            baseline_size=baseline_size,
        )
        if candidate is None:
            return None

        features = self._build_features(
            candidate=candidate,
            tracked_wall=tracked_wall,
        )

        wall_id = self._resolve_wall_id(snapshot)

        return DetectorResult(
            detector=self.component,
            decision=DetectorDecision.POSITIVE,
            score=candidate.score,
            confidence=candidate.confidence,
            reason=candidate.reason,
            features=features,
            wall_id=wall_id,
            pattern=SpoofingPattern.SINGLE_LEVEL_SPOOF,
            metadata={
                "scope": spoofing_key_to_dict(snapshot.key),
                "exchange_symbol": snapshot.exchange_symbol,
                "baseline_size": candidate.baseline_size,
                "size_ratio": candidate.size_ratio,
                "distance_from_mid_bps": candidate.distance_from_mid_bps,
                "near_best_quote": candidate.near_best_quote,
                "notional": candidate.notional,
            },
        )

    def analyze_key(
        self,
        snapshots: Iterable[OrderbookLevelSnapshot],
        *,
        key: SpoofingKey,
        side: SpoofingSide | None = None,
    ) -> list[DetectorResult]:
        """
        Key-first API для scoped futures market.

        key:
            exchange + market_type + symbol + timeframe
        """
        return self.analyze_many(
            snapshots=snapshots,
            key=key,
            side=side,
        )

    def analyze_many(
        self,
        snapshots: Iterable[OrderbookLevelSnapshot],
        *,
        key: SpoofingKey | None = None,
        symbol: str | None = None,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        side: SpoofingSide | None = None,
    ) -> list[DetectorResult]:
        """
        Аналізує набір рівнів стакана та повертає позитивні wall-кандидати.

        New code should pass key=SpoofingKey.
        Legacy filters exchange/symbol/market_type/timeframe залишені для міграції.
        """
        if not self.config.enabled or not self.config.wall_detection.enabled:
            return []

        levels = self._filter_snapshots(
            snapshots=snapshots,
            key=key,
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            side=side,
        )
        if not levels:
            return []

        baseline = self._estimate_baseline_size(levels)
        results: list[DetectorResult] = []

        for snapshot in levels:
            tracked_wall = self._resolve_tracked_wall(snapshot)

            result = self.analyze(
                snapshot,
                baseline_size=baseline,
                tracked_wall=tracked_wall,
            )
            if result is not None and result.is_positive():
                results.append(result)

        results.sort(key=lambda item: (item.score, item.confidence), reverse=True)
        return results

    def select_top_candidates(
        self,
        snapshots: Iterable[OrderbookLevelSnapshot],
        *,
        limit: int = 10,
        key: SpoofingKey | None = None,
        symbol: str | None = None,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        side: SpoofingSide | None = None,
    ) -> list[DetectorResult]:
        """
        Повертає top-N найсильніших wall-кандидатів.
        """
        if limit <= 0:
            return []

        results = self.analyze_many(
            snapshots=snapshots,
            key=key,
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            side=side,
        )
        return results[:limit]

    def is_wall_candidate(
        self,
        snapshot: OrderbookLevelSnapshot,
        *,
        baseline_size: float | None = None,
    ) -> bool:
        """
        Boolean helper для швидкої перевірки одного рівня.
        """
        return self._evaluate_snapshot_candidate(
            snapshot=snapshot,
            baseline_size=baseline_size,
        ) is not None

    def build_snapshot_levels_from_orderbook(
        self,
        *,
        symbol: str,
        exchange: str,
        bids: Iterable[Any],
        asks: Iterable[Any],
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
        best_bid: float | None = None,
        best_ask: float | None = None,
        sequence_id: int | None = None,
        timestamp: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[OrderbookLevelSnapshot]:
        """
        Manual/test helper.

        Перетворює bids/asks у normalized OrderbookLevelSnapshot.

        Production runtime не має напряму передавати сюди raw exchange payload.
        Production path:
            OrderBookCache -> market.orderbook.updated -> SpoofingAnalyzer

        Robustness:
        - malformed raw levels не мають валити pipeline;
        - non-numeric, NaN, inf, -inf, price <= 0, size <= 0 пропускаються;
        - OrderbookLevelSnapshot створюється тільки після pre-validation.
        """
        ts = self.ensure_utc(timestamp)

        safe_best_bid = self._coerce_optional_positive_float(best_bid)
        safe_best_ask = self._coerce_optional_positive_float(best_ask)

        mid_price = self._resolve_mid_price(
            best_bid=safe_best_bid,
            best_ask=safe_best_ask,
        )
        spread = self._resolve_spread(
            best_bid=safe_best_bid,
            best_ask=safe_best_ask,
        )

        base_metadata = {
            **dict(metadata or {}),
            "source": "manual_or_test_helper",
        }

        levels: list[OrderbookLevelSnapshot] = []

        for price, size in self._iter_valid_raw_levels(bids):
            snapshot = self._safe_build_level_snapshot(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                side=SpoofingSide.BID,
                price=price,
                size=size,
                best_bid=safe_best_bid,
                best_ask=safe_best_ask,
                mid_price=mid_price,
                spread=spread,
                sequence_id=sequence_id,
                timestamp=ts,
                metadata=base_metadata,
            )
            if snapshot is not None:
                levels.append(snapshot)

        for price, size in self._iter_valid_raw_levels(asks):
            snapshot = self._safe_build_level_snapshot(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                side=SpoofingSide.ASK,
                price=price,
                size=size,
                best_bid=safe_best_bid,
                best_ask=safe_best_ask,
                mid_price=mid_price,
                spread=spread,
                sequence_id=sequence_id,
                timestamp=ts,
                metadata=base_metadata,
            )
            if snapshot is not None:
                levels.append(snapshot)

        return levels

    def analyze_orderbook_side(
        self,
        *,
        symbol: str,
        exchange: str,
        side: SpoofingSide,
        levels: Iterable[Any],
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
        best_bid: float | None = None,
        best_ask: float | None = None,
        sequence_id: int | None = None,
        timestamp: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[DetectorResult]:
        """
        Manual/test helper для аналізу однієї сторони стакана.

        New production code should use analyze_key() with snapshots prepared by
        SpoofingAnalyzer from OrderBookCache.
        """
        resolved_side = self.parse_spoofing_side(side)
        if resolved_side == SpoofingSide.UNKNOWN:
            return []

        if resolved_side == SpoofingSide.BID:
            snapshots = self.build_snapshot_levels_from_orderbook(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                bids=list(levels),
                asks=[],
                best_bid=best_bid,
                best_ask=best_ask,
                sequence_id=sequence_id,
                timestamp=timestamp,
                metadata=metadata,
            )
        elif resolved_side == SpoofingSide.ASK:
            snapshots = self.build_snapshot_levels_from_orderbook(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                bids=[],
                asks=list(levels),
                best_bid=best_bid,
                best_ask=best_ask,
                sequence_id=sequence_id,
                timestamp=timestamp,
                metadata=metadata,
            )
        else:
            return []

        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        return self.analyze_many(
            snapshots=snapshots,
            key=key,
            side=resolved_side,
        )

    # -------------------------------------------------------------------------
    # Core detection logic
    # -------------------------------------------------------------------------

    def _evaluate_snapshot_candidate(
        self,
        *,
        snapshot: OrderbookLevelSnapshot,
        baseline_size: float | None = None,
    ) -> WallCandidateContext | None:
        if not self.config.enabled or not self.config.wall_detection.enabled:
            return None

        if not self.should_process_key(snapshot.key):
            return None

        if snapshot.side == SpoofingSide.UNKNOWN:
            return None

        if not self._is_valid_snapshot_numeric_values(snapshot):
            return None

        notional = snapshot.price * snapshot.size
        if not math.isfinite(notional) or notional <= 0.0:
            return None

        if notional < self.config.wall_detection.min_wall_size_abs:
            return None

        effective_baseline = self._resolve_effective_baseline(
            snapshot=snapshot,
            baseline_size=baseline_size,
        )
        if not math.isfinite(effective_baseline) or effective_baseline <= 0.0:
            return None

        size_ratio = snapshot.size / effective_baseline
        if not math.isfinite(size_ratio):
            return None

        if size_ratio < self.config.wall_detection.min_wall_size_ratio:
            return None

        mid_price = self._resolve_snapshot_mid_price(snapshot)
        distance_from_mid_bps = (
            self.bps_distance(snapshot.price, mid_price)
            if mid_price is not None and mid_price > 0
            else 0.0
        )
        if not math.isfinite(distance_from_mid_bps):
            return None

        if distance_from_mid_bps > self.config.wall_detection.max_distance_from_mid_bps:
            return None

        near_best_quote = self._is_near_best_quote(snapshot)

        score = self._compute_candidate_score(
            size_ratio=size_ratio,
            distance_from_mid_bps=distance_from_mid_bps,
            near_best_quote=near_best_quote,
            notional=notional,
        )

        confidence = self._compute_candidate_confidence(
            snapshot=snapshot,
            size_ratio=size_ratio,
            distance_from_mid_bps=distance_from_mid_bps,
            near_best_quote=near_best_quote,
            notional=notional,
        )

        reason = self._build_reason(
            snapshot=snapshot,
            size_ratio=size_ratio,
            distance_from_mid_bps=distance_from_mid_bps,
            near_best_quote=near_best_quote,
            notional=notional,
        )

        return WallCandidateContext(
            snapshot=snapshot,
            baseline_size=effective_baseline,
            size_ratio=size_ratio,
            distance_from_mid_bps=distance_from_mid_bps,
            near_best_quote=near_best_quote,
            notional=notional,
            confidence=confidence,
            score=score,
            reason=reason,
        )

    def _build_features(
        self,
        *,
        candidate: WallCandidateContext,
        tracked_wall: TrackedWall | None,
    ) -> SpoofingFeatures:
        snapshot = candidate.snapshot

        repetition_count = 0
        lifetime_ms = 0.0
        updates_count = 0
        fill_ratio = 0.0
        pull_ratio = 0.0

        if tracked_wall is not None:
            lifetime_ms = tracked_wall.lifetime_ms
            updates_count = tracked_wall.updates_count
            fill_ratio = tracked_wall.fill_ratio
            pull_ratio = tracked_wall.pull_ratio

            if self.persistence_tracker is not None:
                history = self.persistence_tracker.get_recent_history(
                    exchange=tracked_wall.exchange,
                    market_type=tracked_wall.market_type,
                    symbol=tracked_wall.symbol,
                    timeframe=tracked_wall.timeframe,
                    side=tracked_wall.side,
                    price=tracked_wall.price,
                    limit=50,
                )
                repetition_count = len(history)

        cancel_to_fill_ratio = self._compute_cancel_to_fill_ratio(
            pull_ratio=pull_ratio,
            fill_ratio=fill_ratio,
        )

        return SpoofingFeatures(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            timeframe=snapshot.timeframe,
            exchange_symbol=snapshot.exchange_symbol,
            side=snapshot.side,
            price=snapshot.price,
            wall_size=snapshot.size,
            wall_size_ratio=candidate.size_ratio,
            distance_from_mid_bps=candidate.distance_from_mid_bps,
            lifetime_ms=lifetime_ms,
            updates_count=updates_count,
            repetition_count=repetition_count,
            fill_ratio=fill_ratio,
            pull_ratio=pull_ratio,
            cancel_to_fill_ratio=cancel_to_fill_ratio,
            is_near_best_quote=candidate.near_best_quote,
            is_fast_pull=False,
            is_fake_liquidity=False,
            is_layering=False,
            metadata={
                "scope": spoofing_key_to_dict(snapshot.key),
                "notional": candidate.notional,
                "baseline_size": candidate.baseline_size,
                "detector": self.component.value,
                "tracked_wall_id": tracked_wall.wall_id if tracked_wall else None,
            },
        )

    # -------------------------------------------------------------------------
    # Scoring / confidence
    # -------------------------------------------------------------------------

    def _compute_candidate_score(
        self,
        *,
        size_ratio: float,
        distance_from_mid_bps: float,
        near_best_quote: bool,
        notional: float,
    ) -> float:
        """
        Первинний wall score в [0, 1].

        Це ще не фінальний spoofing score, а тільки score кандидата.
        """
        min_ratio = self.config.wall_detection.min_wall_size_ratio
        ratio_component = 0.0
        if min_ratio > 0:
            ratio_component = (size_ratio - min_ratio) / max(min_ratio, 1e-12)
        ratio_component = self.clamp(ratio_component, 0.0, 1.0)

        min_notional = self.config.wall_detection.min_wall_size_abs
        notional_component = 0.0
        if min_notional > 0:
            notional_component = (notional - min_notional) / max(min_notional * 2.0, 1e-12)
        notional_component = self.clamp(notional_component, 0.0, 1.0)

        max_distance = max(self.config.wall_detection.max_distance_from_mid_bps, 1e-12)
        distance_component = 1.0 - self.clamp(
            distance_from_mid_bps / max_distance,
            0.0,
            1.0,
        )

        proximity_bonus = 1.0 if near_best_quote else 0.5

        raw_score = (
            0.40 * ratio_component
            + 0.25 * notional_component
            + 0.25 * distance_component
            + 0.10 * proximity_bonus
        )

        return self.clamp(raw_score, 0.0, 1.0)

    def _compute_candidate_confidence(
        self,
        *,
        snapshot: OrderbookLevelSnapshot,
        size_ratio: float,
        distance_from_mid_bps: float,
        near_best_quote: bool,
        notional: float,
    ) -> float:
        """
        Confidence для wall candidate.
        """
        confidence = 0.35

        min_ratio = self.config.wall_detection.min_wall_size_ratio
        if size_ratio >= min_ratio * 1.25:
            confidence += 0.15
        if size_ratio >= min_ratio * 1.75:
            confidence += 0.10

        if near_best_quote:
            confidence += 0.15

        max_distance = self.config.wall_detection.max_distance_from_mid_bps
        if max_distance > 0:
            closeness = 1.0 - self.clamp(distance_from_mid_bps / max_distance, 0.0, 1.0)
            confidence += 0.15 * closeness

        min_notional = self.config.wall_detection.min_wall_size_abs
        if min_notional > 0 and notional >= min_notional * 2:
            confidence += 0.10

        if snapshot.best_bid is not None and snapshot.best_ask is not None:
            confidence += 0.05

        return self.clamp(confidence, 0.0, 0.99)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _filter_snapshots(
        self,
        *,
        snapshots: Iterable[OrderbookLevelSnapshot],
        key: SpoofingKey | None = None,
        symbol: str | None,
        exchange: str | None,
        market_type: str | None,
        timeframe: str | None,
        side: SpoofingSide | None,
    ) -> list[OrderbookLevelSnapshot]:
        resolved_side = self.parse_spoofing_side(side) if side is not None else None

        normalized_exchange = (
            self.normalize_exchange(exchange)
            if exchange is not None
            else None
        )
        normalized_symbol = (
            self.normalize_symbol(symbol)
            if symbol is not None
            else None
        )
        normalized_market_type = (
            self.normalize_market_type(market_type)
            if market_type is not None
            else None
        )
        normalized_timeframe = (
            self.normalize_timeframe(timeframe)
            if timeframe is not None
            else None
        )

        levels: list[OrderbookLevelSnapshot] = []

        for item in snapshots:
            if item is None:
                continue

            if not self._has_snapshot_contract(item):
                continue

            if key is not None and item.key != key:
                continue
            if normalized_symbol is not None and item.symbol != normalized_symbol:
                continue
            if normalized_exchange is not None and item.exchange != normalized_exchange:
                continue
            if normalized_market_type is not None and item.market_type != normalized_market_type:
                continue
            if normalized_timeframe is not None and item.timeframe != normalized_timeframe:
                continue
            if resolved_side is not None and item.side != resolved_side:
                continue
            if item.side == SpoofingSide.UNKNOWN:
                continue
            if not self.should_process_key(item.key):
                continue
            if not self._is_valid_snapshot_numeric_values(item):
                continue

            levels.append(item)

        max_levels = self.config.wall_detection.max_levels_to_scan
        if max_levels > 0:
            return levels[:max_levels]

        return levels

    def _resolve_tracked_wall(
        self,
        snapshot: OrderbookLevelSnapshot,
    ) -> TrackedWall | None:
        if self.persistence_tracker is None:
            return None

        return self.persistence_tracker.get_wall_by_level(
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            side=snapshot.side,
            price=snapshot.price,
        )

    def _resolve_wall_id(self, snapshot: OrderbookLevelSnapshot) -> str | None:
        if self.persistence_tracker is None:
            return None

        return self.persistence_tracker.build_wall_id(
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            side=snapshot.side,
            price=snapshot.price,
        )

    @staticmethod
    def _resolve_mid_price(
        *,
        best_bid: float | None,
        best_ask: float | None,
    ) -> float | None:
        if best_bid is None or best_ask is None:
            return None
        if not math.isfinite(best_bid) or not math.isfinite(best_ask):
            return None
        if best_bid <= 0 or best_ask <= 0:
            return None
        return (best_bid + best_ask) / 2.0

    @staticmethod
    def _resolve_spread(
        *,
        best_bid: float | None,
        best_ask: float | None,
    ) -> float | None:
        if best_bid is None or best_ask is None:
            return None
        if not math.isfinite(best_bid) or not math.isfinite(best_ask):
            return None
        if best_bid <= 0 or best_ask <= 0:
            return None
        return max(0.0, best_ask - best_bid)

    def _resolve_snapshot_mid_price(
        self,
        snapshot: OrderbookLevelSnapshot,
    ) -> float | None:
        if (
            snapshot.mid_price is not None
            and math.isfinite(snapshot.mid_price)
            and snapshot.mid_price > 0
        ):
            return snapshot.mid_price

        return self._resolve_mid_price(
            best_bid=snapshot.best_bid,
            best_ask=snapshot.best_ask,
        )

    @staticmethod
    def _resolve_effective_baseline(
        *,
        snapshot: OrderbookLevelSnapshot,
        baseline_size: float | None,
    ) -> float:
        if baseline_size is None:
            return max(snapshot.size, 1.0)
        if not math.isfinite(baseline_size) or baseline_size <= 0.0:
            return max(snapshot.size, 1.0)
        return max(baseline_size, 1e-12)

    @staticmethod
    def _compute_cancel_to_fill_ratio(
        *,
        pull_ratio: float,
        fill_ratio: float,
    ) -> float:
        if fill_ratio > 0:
            return pull_ratio / fill_ratio
        if pull_ratio > 0:
            return pull_ratio
        return 0.0

    def _is_near_best_quote(self, snapshot: OrderbookLevelSnapshot) -> bool:
        near_bps = self.config.wall_detection.near_best_quote_bps

        if near_bps <= 0:
            return False

        if (
            snapshot.side == SpoofingSide.BID
            and snapshot.best_bid is not None
            and math.isfinite(snapshot.best_bid)
            and snapshot.best_bid > 0
        ):
            distance = self.bps_distance(snapshot.price, snapshot.best_bid)
            return math.isfinite(distance) and distance <= near_bps

        if (
            snapshot.side == SpoofingSide.ASK
            and snapshot.best_ask is not None
            and math.isfinite(snapshot.best_ask)
            and snapshot.best_ask > 0
        ):
            distance = self.bps_distance(snapshot.price, snapshot.best_ask)
            return math.isfinite(distance) and distance <= near_bps

        return False

    @staticmethod
    def _estimate_baseline_size(
        snapshots: Iterable[OrderbookLevelSnapshot],
    ) -> float:
        """
        Оцінює локальний baseline size для рівнів стакана.

        Бере median size, щоб один великий wall не зіпсував baseline.
        """
        sizes = [
            item.size
            for item in snapshots
            if item is not None
            and isinstance(item.size, (int, float))
            and math.isfinite(float(item.size))
            and float(item.size) > 0.0
        ]
        if not sizes:
            return 1.0

        return max(median(sizes), 1e-12)

    @staticmethod
    def _build_reason(
        *,
        snapshot: OrderbookLevelSnapshot,
        size_ratio: float,
        distance_from_mid_bps: float,
        near_best_quote: bool,
        notional: float,
    ) -> str:
        side = snapshot.side.value.upper()
        near_txt = "near_best_quote" if near_best_quote else "not_near_best_quote"

        return (
            f"{side} wall candidate detected: "
            f"exchange={snapshot.exchange}, "
            f"market_type={snapshot.market_type}, "
            f"symbol={snapshot.symbol}, "
            f"timeframe={snapshot.timeframe}, "
            f"notional={notional:.2f}, "
            f"size_ratio={size_ratio:.2f}, "
            f"distance_from_mid_bps={distance_from_mid_bps:.2f}, "
            f"{near_txt}"
        )

    # -------------------------------------------------------------------------
    # Raw level robustness helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _coerce_optional_positive_float(value: Any) -> float | None:
        if value is None:
            return None

        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None

        if not math.isfinite(result) or result <= 0.0:
            return None

        return result

    @staticmethod
    def _coerce_raw_price_size(raw_level: Any) -> tuple[float, float] | None:
        """
        Безпечно дістає price/size з raw level.

        Підтримує:
        - tuple/list: (price, size, ...)
        - dict: {"price": ..., "size": ...}
        - dict aliases: qty/quantity/amount
        """
        if raw_level is None:
            return None

        raw_price: Any
        raw_size: Any

        if isinstance(raw_level, dict):
            raw_price = raw_level.get("price")
            raw_size = (
                raw_level.get("size")
                if "size" in raw_level
                else raw_level.get("qty", raw_level.get("quantity", raw_level.get("amount")))
            )
        elif isinstance(raw_level, (list, tuple)):
            if len(raw_level) < 2:
                return None
            raw_price = raw_level[0]
            raw_size = raw_level[1]
        else:
            return None

        try:
            price = float(raw_price)
            size = float(raw_size)
        except (TypeError, ValueError, OverflowError):
            return None

        if not OrderbookWallDetector._is_valid_raw_level_value(price=price, size=size):
            return None

        return price, size

    @staticmethod
    def _is_valid_raw_level_value(*, price: float, size: float) -> bool:
        return (
            math.isfinite(price)
            and math.isfinite(size)
            and price > 0.0
            and size > 0.0
        )

    @classmethod
    def _iter_valid_raw_levels(
        cls,
        levels: Iterable[Any],
    ) -> Iterable[tuple[float, float]]:
        for raw_level in levels or ():
            parsed = cls._coerce_raw_price_size(raw_level)
            if parsed is None:
                continue
            yield parsed

    def _safe_build_level_snapshot(
        self,
        *,
        symbol: str,
        exchange: str,
        market_type: str,
        timeframe: str,
        exchange_symbol: str | None,
        side: SpoofingSide,
        price: float,
        size: float,
        best_bid: float | None,
        best_ask: float | None,
        mid_price: float | None,
        spread: float | None,
        sequence_id: int | None,
        timestamp: Any,
        metadata: dict[str, Any],
    ) -> OrderbookLevelSnapshot | None:
        """
        Створює OrderbookLevelSnapshot тільки для попередньо валідованих values.

        Додатково захищає helper від майбутніх змін у model validation:
        manual/test helper не має валити весь pipeline через один поганий level.
        """
        if not self._is_valid_raw_level_value(price=price, size=size):
            return None

        try:
            return self.build_level_snapshot(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                side=side,
                price=price,
                size=size,
                best_bid=best_bid,
                best_ask=best_ask,
                mid_price=mid_price,
                spread=spread,
                sequence_id=sequence_id,
                timestamp=timestamp,
                metadata=metadata,
            )
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _has_snapshot_contract(item: Any) -> bool:
        required_attrs = (
            "key",
            "symbol",
            "exchange",
            "market_type",
            "timeframe",
            "side",
            "price",
            "size",
        )
        return all(hasattr(item, attr) for attr in required_attrs)

    @staticmethod
    def _is_valid_snapshot_numeric_values(item: Any) -> bool:
        try:
            price = float(item.price)
            size = float(item.size)
        except (TypeError, ValueError, OverflowError):
            return False

        return (
            math.isfinite(price)
            and math.isfinite(size)
            and price > 0.0
            and size > 0.0
        )


__all__ = ["OrderbookWallDetector"]