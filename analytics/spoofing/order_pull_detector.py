from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .base import BaseSpoofingDetector
from .config import SpoofingConfig
from .enums import (
    DetectorDecision,
    OrderbookWallState,
    SpoofingComponent,
    SpoofingPattern,
    SpoofingType,
)
from .models import (
    DetectorResult,
    SpoofingFeatures,
    TrackedWall,
)
from .persistence_tracker import PersistenceTracker


@dataclass(slots=True)
class PullCandidateContext:
    """
    Внутрішній контейнер для оцінки кандидата на pull-event.
    """
    wall: TrackedWall
    pulled_notional: float
    pulled_size_ratio: float
    fill_ratio: float
    pull_ratio: float
    lifetime_ms: float
    is_fast_pull: bool
    is_strong_pull: bool
    confidence: float
    score: float
    reason: str


class OrderPullDetector(BaseSpoofingDetector):
    """
    Detector швидкого зняття ліквідності.

    Призначення:
    - знайти стінки, які:
        1) були достатньо великими,
        2) існували недовго,
        3) були значно або повністю зняті,
        4) при цьому майже не були виконані.

    Важливо:
    - цей detector працює вже поверх stateful persistence tracking
    - він не дивиться на сирий orderbook сам по собі
    - він аналізує еволюцію конкретного рівня
    """

    component = SpoofingComponent.ORDER_PULL_DETECTOR

    def __init__(
        self,
        event_bus: Any | None,
        config: SpoofingConfig,
        persistence_tracker: PersistenceTracker,
    ) -> None:
        super().__init__(event_bus=event_bus, config=config)
        self.persistence_tracker = persistence_tracker

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def analyze(
        self,
        wall: TrackedWall,
        *,
        current_mid_price: float | None = None,
        repetition_count: int | None = None,
    ) -> DetectorResult | None:
        """
        Аналізує один tracked wall на предмет підозрілого pull.
        """
        candidate = self._evaluate_pull_candidate(wall=wall)
        if candidate is None:
            return None

        features = self._build_features(
            candidate=candidate,
            current_mid_price=current_mid_price,
            repetition_count=repetition_count,
        )

        return DetectorResult(
            detector=self.component,
            decision=DetectorDecision.POSITIVE,
            score=candidate.score,
            confidence=candidate.confidence,
            reason=candidate.reason,
            features=features,
            wall_id=wall.wall_id,
            pattern=SpoofingPattern.PULL_AND_REVERSAL,
            metadata={
                "spoofing_type": SpoofingType.ORDER_PULL.value,
                "pulled_notional": candidate.pulled_notional,
                "pulled_size_ratio": candidate.pulled_size_ratio,
                "pull_ratio": candidate.pull_ratio,
                "fill_ratio": candidate.fill_ratio,
                "lifetime_ms": candidate.lifetime_ms,
                "is_fast_pull": candidate.is_fast_pull,
                "is_strong_pull": candidate.is_strong_pull,
                "wall_state": wall.state.value,
            },
        )

    def analyze_many(
        self,
        walls: Iterable[TrackedWall],
        *,
        exchange: str | None = None,
        symbol: str | None = None,
    ) -> list[DetectorResult]:
        """
        Аналізує набір tracked walls і повертає позитивні pull-candidates.
        """
        results: list[DetectorResult] = []

        for wall in walls:
            if exchange is not None and wall.exchange != exchange:
                continue
            if symbol is not None and wall.symbol != symbol:
                continue

            repetition_count = self._estimate_repetition_count(wall)
            result = self.analyze(
                wall,
                repetition_count=repetition_count,
            )
            if result is not None and result.decision == DetectorDecision.POSITIVE:
                results.append(result)

        results.sort(key=lambda item: (item.score, item.confidence), reverse=True)
        return results

    def analyze_symbol(
        self,
        *,
        exchange: str,
        symbol: str,
    ) -> list[DetectorResult]:
        """
        Зручний helper для аналізу всіх tracked walls одного символу.
        """
        walls = self.persistence_tracker.get_walls_for_symbol(
            exchange=exchange,
            symbol=symbol,
        )
        return self.analyze_many(
            walls=walls,
            exchange=exchange,
            symbol=symbol,
        )

    def is_pull_candidate(self, wall: TrackedWall) -> bool:
        """
        Простий boolean helper.
        """
        return self._evaluate_pull_candidate(wall=wall) is not None

    # -------------------------------------------------------------------------
    # Core detection logic
    # -------------------------------------------------------------------------

    def _evaluate_pull_candidate(
        self,
        *,
        wall: TrackedWall,
    ) -> PullCandidateContext | None:
        if not self.config.pull_detection.enabled:
            return None

        if wall.max_size <= 0.0:
            return None

        # Має бути достатньо велика стінка
        wall_notional = wall.price * wall.max_size
        if wall_notional < self.config.wall_detection.min_wall_size_abs:
            return None

        lifetime_ms = wall.lifetime_ms
        if lifetime_ms <= 0:
            return None

        pull_ratio = wall.pull_ratio
        fill_ratio = wall.fill_ratio
        pulled_notional = wall.price * wall.estimated_pulled_size
        pulled_size_ratio = self.normalize_ratio(wall.estimated_pulled_size, wall.max_size)

        # Основні фільтри
        if pulled_notional < self.config.pull_detection.min_removed_notional:
            return None

        if pull_ratio < self.config.pull_detection.min_pull_ratio:
            return None

        if fill_ratio > self.config.pull_detection.max_fill_ratio_for_pull:
            return None

        if lifetime_ms > self.config.pull_detection.max_pull_lifetime_ms:
            return None

        # Стан також має бути релевантним
        if wall.state not in {
            OrderbookWallState.PULLED,
            OrderbookWallState.WEAKENING,
            OrderbookWallState.EXPIRED,
            OrderbookWallState.FILLED,  # допускаємо, бо іноді евристики могли частково спотворити state
        }:
            return None

        # Якщо явно filled і pull_ratio слабкий — відсікаємо
        if wall.state == OrderbookWallState.FILLED and pull_ratio < self.config.pull_detection.strong_pull_ratio:
            return None

        is_fast_pull = lifetime_ms <= self.config.pull_detection.fast_pull_lifetime_ms
        is_strong_pull = pull_ratio >= self.config.pull_detection.strong_pull_ratio

        score = self._compute_pull_score(
            wall=wall,
            lifetime_ms=lifetime_ms,
            pull_ratio=pull_ratio,
            fill_ratio=fill_ratio,
            pulled_notional=pulled_notional,
            is_fast_pull=is_fast_pull,
            is_strong_pull=is_strong_pull,
        )

        confidence = self._compute_pull_confidence(
            wall=wall,
            lifetime_ms=lifetime_ms,
            pull_ratio=pull_ratio,
            fill_ratio=fill_ratio,
            pulled_notional=pulled_notional,
            is_fast_pull=is_fast_pull,
            is_strong_pull=is_strong_pull,
        )

        reason = self._build_reason(
            wall=wall,
            lifetime_ms=lifetime_ms,
            pull_ratio=pull_ratio,
            fill_ratio=fill_ratio,
            pulled_notional=pulled_notional,
            is_fast_pull=is_fast_pull,
            is_strong_pull=is_strong_pull,
        )

        return PullCandidateContext(
            wall=wall,
            pulled_notional=pulled_notional,
            pulled_size_ratio=pulled_size_ratio,
            fill_ratio=fill_ratio,
            pull_ratio=pull_ratio,
            lifetime_ms=lifetime_ms,
            is_fast_pull=is_fast_pull,
            is_strong_pull=is_strong_pull,
            confidence=confidence,
            score=score,
            reason=reason,
        )

    def _build_features(
        self,
        *,
        candidate: PullCandidateContext,
        current_mid_price: float | None = None,
        repetition_count: int | None = None,
    ) -> SpoofingFeatures:
        wall = candidate.wall

        distance_from_mid_bps = 0.0
        if current_mid_price is not None and current_mid_price > 0:
            distance_from_mid_bps = self.bps_distance(wall.price, current_mid_price)
        elif wall.mid_price_at_creation is not None and wall.mid_price_at_creation > 0:
            distance_from_mid_bps = self.bps_distance(wall.price, wall.mid_price_at_creation)

        cancel_to_fill_ratio = 0.0
        if candidate.fill_ratio > 0:
            cancel_to_fill_ratio = candidate.pull_ratio / candidate.fill_ratio
        elif candidate.pull_ratio > 0:
            cancel_to_fill_ratio = candidate.pull_ratio

        repetition = repetition_count if repetition_count is not None else self._estimate_repetition_count(wall)

        is_near_best_quote = wall.near_touch_count > 0 or wall.touch_count > 0

        return SpoofingFeatures(
            symbol=wall.symbol,
            exchange=wall.exchange,
            side=wall.side,
            price=wall.price,
            wall_size=wall.max_size,
            wall_size_ratio=self.normalize_ratio(wall.max_size, max(wall.initial_size, 1e-12)),
            distance_from_mid_bps=distance_from_mid_bps,
            lifetime_ms=candidate.lifetime_ms,
            updates_count=wall.updates_count,
            repetition_count=repetition,
            fill_ratio=candidate.fill_ratio,
            pull_ratio=candidate.pull_ratio,
            cancel_to_fill_ratio=cancel_to_fill_ratio,
            price_reaction_bps=0.0,
            pressure_flip_strength=0.0,
            layering_score=0.0,
            is_near_best_quote=is_near_best_quote,
            is_fast_pull=candidate.is_fast_pull,
            is_fake_liquidity=False,
            is_layering=False,
            metadata={
                "pulled_notional": candidate.pulled_notional,
                "pulled_size_ratio": candidate.pulled_size_ratio,
                "estimated_pulled_size": wall.estimated_pulled_size,
                "estimated_filled_size": wall.estimated_filled_size,
                "current_to_max_ratio": wall.current_to_max_ratio,
                "detector": self.component.value,
            },
        )

    # -------------------------------------------------------------------------
    # Score / confidence
    # -------------------------------------------------------------------------

    def _compute_pull_score(
        self,
        *,
        wall: TrackedWall,
        lifetime_ms: float,
        pull_ratio: float,
        fill_ratio: float,
        pulled_notional: float,
        is_fast_pull: bool,
        is_strong_pull: bool,
    ) -> float:
        """
        Первинний score pull-candidate в [0, 1].
        """
        cfg = self.config.pull_detection

        # 1. lifetime component: чим коротше життя, тим підозріліше
        max_lifetime = max(float(cfg.max_pull_lifetime_ms), 1.0)
        lifetime_component = 1.0 - self.clamp(lifetime_ms / max_lifetime, 0.0, 1.0)

        # 2. pull ratio component
        min_pull_ratio = max(cfg.min_pull_ratio, 1e-12)
        pull_component = (pull_ratio - min_pull_ratio) / max(1.0 - min_pull_ratio, 1e-12)
        pull_component = self.clamp(pull_component, 0.0, 1.0)

        # 3. fill ratio component: чим менше fill, тим підозріліше
        max_fill = max(cfg.max_fill_ratio_for_pull, 1e-12)
        fill_component = 1.0 - self.clamp(fill_ratio / max_fill, 0.0, 1.0)

        # 4. removed notional component
        min_removed_notional = max(cfg.min_removed_notional, 1e-12)
        notional_component = (pulled_notional - min_removed_notional) / max(min_removed_notional * 2.0, 1e-12)
        notional_component = self.clamp(notional_component, 0.0, 1.0)

        # 5. state / behavior bonus
        behavior_bonus = 0.0
        if is_fast_pull:
            behavior_bonus += 0.08
        if is_strong_pull:
            behavior_bonus += 0.08
        if wall.state == OrderbookWallState.PULLED:
            behavior_bonus += 0.06

        raw_score = (
            0.30 * lifetime_component +
            0.28 * pull_component +
            0.18 * fill_component +
            0.16 * notional_component +
            behavior_bonus
        )

        return self.clamp(raw_score, 0.0, 1.0)

    def _compute_pull_confidence(
        self,
        *,
        wall: TrackedWall,
        lifetime_ms: float,
        pull_ratio: float,
        fill_ratio: float,
        pulled_notional: float,
        is_fast_pull: bool,
        is_strong_pull: bool,
    ) -> float:
        """
        Confidence для order pull detection.
        """
        confidence = 0.40

        if is_fast_pull:
            confidence += 0.16

        if is_strong_pull:
            confidence += 0.16

        if fill_ratio <= self.config.pull_detection.max_fill_ratio_for_pull * 0.5:
            confidence += 0.10

        if pulled_notional >= self.config.pull_detection.min_removed_notional * 2.0:
            confidence += 0.08

        if wall.state == OrderbookWallState.PULLED:
            confidence += 0.05

        if wall.near_touch_count == 0 and wall.touch_count == 0:
            confidence += 0.04

        if lifetime_ms <= self.config.pull_detection.fast_pull_lifetime_ms * 0.5:
            confidence += 0.04

        return self.clamp(confidence, 0.0, 0.99)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _estimate_repetition_count(self, wall: TrackedWall) -> int:
        history = self.persistence_tracker.get_recent_history(
            exchange=wall.exchange,
            symbol=wall.symbol,
            side=wall.side,
            price=wall.price,
            limit=100,
        )
        return len(history)

    def _build_reason(
        self,
        *,
        wall: TrackedWall,
        lifetime_ms: float,
        pull_ratio: float,
        fill_ratio: float,
        pulled_notional: float,
        is_fast_pull: bool,
        is_strong_pull: bool,
    ) -> str:
        parts = [
            f"pull candidate detected for {wall.side.value.upper()} wall",
            f"pulled_notional={pulled_notional:.2f}",
            f"pull_ratio={pull_ratio:.4f}",
            f"fill_ratio={fill_ratio:.4f}",
            f"lifetime_ms={lifetime_ms:.2f}",
            f"state={wall.state.value}",
        ]

        if is_fast_pull:
            parts.append("fast_pull=true")
        if is_strong_pull:
            parts.append("strong_pull=true")

        return ", ".join(parts)