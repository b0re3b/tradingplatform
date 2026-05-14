from __future__ import annotations

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
    DetectorResult,
    OrderbookLevelSnapshot,
    SpoofingFeatures,
    TrackedWall,
    WallCandidateContext,
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

    Важливо:
    - detector не визначає spoofing остаточно;
    - не підписується на EventBus;
    - не публікує події;
    - не запускає Scheduler jobs;
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
        Аналізує один конкретний рівень стакана як кандидата на wall.

        Цей метод зручний, коли SpoofingAnalyzer уже ітерується по
        нормалізованих OrderbookLevelSnapshot окремо.
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
                "baseline_size": candidate.baseline_size,
                "size_ratio": candidate.size_ratio,
                "distance_from_mid_bps": candidate.distance_from_mid_bps,
                "near_best_quote": candidate.near_best_quote,
                "notional": candidate.notional,
            },
        )

    def analyze_many(
        self,
        snapshots: Iterable[OrderbookLevelSnapshot],
        *,
        symbol: str | None = None,
        exchange: str | None = None,
        side: SpoofingSide | None = None,
    ) -> list[DetectorResult]:
        """
        Аналізує набір рівнів стакана та повертає позитивні wall-кандидати.
        """
        if not self.config.enabled or not self.config.wall_detection.enabled:
            return []

        levels = self._filter_snapshots(
            snapshots=snapshots,
            symbol=symbol,
            exchange=exchange,
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
        symbol: str | None = None,
        exchange: str | None = None,
        side: SpoofingSide | None = None,
    ) -> list[DetectorResult]:
        """
        Повертає top-N найсильніших wall-кандидатів.
        """
        if limit <= 0:
            return []

        results = self.analyze_many(
            snapshots=snapshots,
            symbol=symbol,
            exchange=exchange,
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
        bids: Iterable[tuple[float, float]],
        asks: Iterable[tuple[float, float]],
        best_bid: float | None = None,
        best_ask: float | None = None,
        sequence_id: int | None = None,
        timestamp: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[OrderbookLevelSnapshot]:
        """
        Перетворює raw bids/asks у normalized OrderbookLevelSnapshot.

        bids/asks очікуються у форматі:
            [(price, size), ...]
        """
        ts = self.ensure_utc(timestamp)
        mid_price = self._resolve_mid_price(best_bid=best_bid, best_ask=best_ask)
        spread = self._resolve_spread(best_bid=best_bid, best_ask=best_ask)

        levels: list[OrderbookLevelSnapshot] = []

        for price, size in bids:
            levels.append(
                self.build_level_snapshot(
                    symbol=symbol,
                    exchange=exchange,
                    side=SpoofingSide.BID,
                    price=price,
                    size=size,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    mid_price=mid_price,
                    spread=spread,
                    sequence_id=sequence_id,
                    timestamp=ts,
                    metadata=dict(metadata or {}),
                )
            )

        for price, size in asks:
            levels.append(
                self.build_level_snapshot(
                    symbol=symbol,
                    exchange=exchange,
                    side=SpoofingSide.ASK,
                    price=price,
                    size=size,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    mid_price=mid_price,
                    spread=spread,
                    sequence_id=sequence_id,
                    timestamp=ts,
                    metadata=dict(metadata or {}),
                )
            )

        return levels

    def analyze_orderbook_side(
        self,
        *,
        symbol: str,
        exchange: str,
        side: SpoofingSide,
        levels: Iterable[tuple[float, float]],
        best_bid: float | None = None,
        best_ask: float | None = None,
        sequence_id: int | None = None,
        timestamp: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[DetectorResult]:
        """
        Аналізує одну сторону стакана.
        """
        resolved_side = self.parse_spoofing_side(side)
        if resolved_side == SpoofingSide.UNKNOWN:
            return []

        if resolved_side == SpoofingSide.BID:
            snapshots = self.build_snapshot_levels_from_orderbook(
                symbol=symbol,
                exchange=exchange,
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

        return self.analyze_many(
            snapshots=snapshots,
            symbol=symbol,
            exchange=exchange,
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

        if snapshot.side == SpoofingSide.UNKNOWN:
            return None

        if snapshot.size <= 0.0 or snapshot.price <= 0.0:
            return None

        notional = snapshot.price * snapshot.size

        if notional < self.config.wall_detection.min_wall_size_abs:
            return None

        effective_baseline = self._resolve_effective_baseline(
            snapshot=snapshot,
            baseline_size=baseline_size,
        )
        size_ratio = snapshot.size / effective_baseline

        if size_ratio < self.config.wall_detection.min_wall_size_ratio:
            return None

        mid_price = self._resolve_snapshot_mid_price(snapshot)
        distance_from_mid_bps = (
            self.bps_distance(snapshot.price, mid_price)
            if mid_price is not None and mid_price > 0
            else 0.0
        )

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
                    symbol=tracked_wall.symbol,
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
                "notional": candidate.notional,
                "baseline_size": candidate.baseline_size,
                "detector": self.component.value,
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
        symbol: str | None,
        exchange: str | None,
        side: SpoofingSide | None,
    ) -> list[OrderbookLevelSnapshot]:
        levels = [
            item
            for item in snapshots
            if (symbol is None or item.symbol == symbol)
            and (exchange is None or item.exchange == exchange)
            and (side is None or item.side == side)
        ]

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
            symbol=snapshot.symbol,
            side=snapshot.side,
            price=snapshot.price,
        )

    def _resolve_wall_id(self, snapshot: OrderbookLevelSnapshot) -> str | None:
        if self.persistence_tracker is None:
            return None

        return self.persistence_tracker.build_wall_id(
            exchange=snapshot.exchange,
            symbol=snapshot.symbol,
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
        if best_bid <= 0 or best_ask <= 0:
            return None
        return max(0.0, best_ask - best_bid)

    def _resolve_snapshot_mid_price(
        self,
        snapshot: OrderbookLevelSnapshot,
    ) -> float | None:
        if snapshot.mid_price is not None and snapshot.mid_price > 0:
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
            and snapshot.best_bid > 0
        ):
            distance = self.bps_distance(snapshot.price, snapshot.best_bid)
            return distance <= near_bps

        if (
            snapshot.side == SpoofingSide.ASK
            and snapshot.best_ask is not None
            and snapshot.best_ask > 0
        ):
            distance = self.bps_distance(snapshot.price, snapshot.best_ask)
            return distance <= near_bps

        return False

    @staticmethod
    def _estimate_baseline_size(
        snapshots: Iterable[OrderbookLevelSnapshot],
    ) -> float:
        """
        Оцінює локальний baseline size для рівнів стакана.

        Бере median size, щоб один великий wall не зіпсував baseline.
        """
        sizes = [item.size for item in snapshots if item.size > 0]
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
            f"notional={notional:.2f}, "
            f"size_ratio={size_ratio:.2f}, "
            f"distance_from_mid_bps={distance_from_mid_bps:.2f}, "
            f"{near_txt}"
        )


__all__ = ["OrderbookWallDetector"]