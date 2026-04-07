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
)
from .models import (
    DetectorResult,
    SpoofingFeatures,
    TrackedWall,
)
from .persistence_tracker import PersistenceTracker


@dataclass(slots=True)
class FakeLiquidityCandidateContext:
    """
    Внутрішній контейнер для оцінки кандидата на fake liquidity.
    """
    wall: TrackedWall
    wall_notional: float
    pulled_notional: float
    lifetime_ms: float
    fill_ratio: float
    pull_ratio: float
    price_reaction_bps: float
    distance_from_mid_bps: float
    is_short_lived: bool
    is_low_fill: bool
    is_high_pull: bool
    has_market_reaction: bool
    confidence: float
    score: float
    reason: str


class FakeLiquidityDetector(BaseSpoofingDetector):
    """
    Detector фейкової ліквідності.

    Основна ідея:
    - велика ліквідність з'явилась у стакані
    - простояла недовго
    - майже не була виконана
    - значною мірою була знята
    - після цього ринок відреагував у релевантний бік

    Важливо:
    - це advanced detector поверх persistence state
    - він не працює як сирий orderbook parser
    - для найкращої якості йому бажано мати current_mid_price
      або інший reference price після pull-події
    """

    component = SpoofingComponent.FAKE_LIQUIDITY_DETECTOR

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
        Аналізує один tracked wall як кандидата на fake liquidity.
        """
        candidate = self._evaluate_candidate(
            wall=wall,
            current_mid_price=current_mid_price,
        )
        if candidate is None:
            return None

        features = self._build_features(
            candidate=candidate,
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
            pattern=SpoofingPattern.FAKE_ABSORPTION,
            metadata={
                "wall_notional": candidate.wall_notional,
                "pulled_notional": candidate.pulled_notional,
                "lifetime_ms": candidate.lifetime_ms,
                "fill_ratio": candidate.fill_ratio,
                "pull_ratio": candidate.pull_ratio,
                "price_reaction_bps": candidate.price_reaction_bps,
                "distance_from_mid_bps": candidate.distance_from_mid_bps,
                "is_short_lived": candidate.is_short_lived,
                "is_low_fill": candidate.is_low_fill,
                "is_high_pull": candidate.is_high_pull,
                "has_market_reaction": candidate.has_market_reaction,
                "wall_state": wall.state.value,
            },
        )

    def analyze_many(
        self,
        walls: Iterable[TrackedWall],
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        current_mid_price: float | None = None,
    ) -> list[DetectorResult]:
        """
        Аналізує набір tracked walls.
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
                current_mid_price=current_mid_price,
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
        current_mid_price: float | None = None,
    ) -> list[DetectorResult]:
        """
        Зручний helper для аналізу всіх tracked walls символу.
        """
        walls = self.persistence_tracker.get_walls_for_symbol(
            exchange=exchange,
            symbol=symbol,
        )
        return self.analyze_many(
            walls=walls,
            exchange=exchange,
            symbol=symbol,
            current_mid_price=current_mid_price,
        )

    def is_fake_liquidity_candidate(
        self,
        wall: TrackedWall,
        *,
        current_mid_price: float | None = None,
    ) -> bool:
        return self._evaluate_candidate(
            wall=wall,
            current_mid_price=current_mid_price,
        ) is not None

    # -------------------------------------------------------------------------
    # Core detection logic
    # -------------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        *,
        wall: TrackedWall,
        current_mid_price: float | None = None,
    ) -> FakeLiquidityCandidateContext | None:
        if not self.config.fake_liquidity.enabled:
            return None

        if wall.max_size <= 0.0 or wall.price <= 0.0:
            return None

        wall_notional = wall.price * wall.max_size
        pulled_notional = wall.price * wall.estimated_pulled_size
        lifetime_ms = wall.lifetime_ms
        fill_ratio = wall.fill_ratio
        pull_ratio = wall.pull_ratio

        # Базова відсічка: стінка має бути суттєвою
        if wall_notional < self.config.wall_detection.min_wall_size_abs:
            return None

        # Стінка повинна мати ознаки "нереального наміру"
        if fill_ratio > self.config.fake_liquidity.max_fill_ratio:
            return None

        if pull_ratio < self.config.fake_liquidity.min_pull_ratio:
            return None

        if lifetime_ms > self.config.fake_liquidity.max_lifetime_ms:
            return None

        if wall.state not in {
            OrderbookWallState.PULLED,
            OrderbookWallState.WEAKENING,
            OrderbookWallState.EXPIRED,
            OrderbookWallState.FILLED,
        }:
            return None

        price_reaction_bps = self._estimate_price_reaction_bps(
            wall=wall,
            current_mid_price=current_mid_price,
        )

        if price_reaction_bps < self.config.fake_liquidity.min_price_reaction_bps:
            return None

        distance_from_mid_bps = self._estimate_distance_from_mid_bps(
            wall=wall,
            current_mid_price=current_mid_price,
        )

        is_short_lived = lifetime_ms <= self.config.fake_liquidity.max_lifetime_ms * 0.5
        is_low_fill = fill_ratio <= self.config.fake_liquidity.max_fill_ratio * 0.5
        is_high_pull = pull_ratio >= max(
            self.config.fake_liquidity.min_pull_ratio,
            self.config.pull_detection.strong_pull_ratio,
        )
        has_market_reaction = price_reaction_bps >= self.config.fake_liquidity.min_price_reaction_bps

        score = self._compute_score(
            wall=wall,
            pulled_notional=pulled_notional,
            lifetime_ms=lifetime_ms,
            fill_ratio=fill_ratio,
            pull_ratio=pull_ratio,
            price_reaction_bps=price_reaction_bps,
            is_short_lived=is_short_lived,
            is_low_fill=is_low_fill,
            is_high_pull=is_high_pull,
        )

        confidence = self._compute_confidence(
            wall=wall,
            pulled_notional=pulled_notional,
            lifetime_ms=lifetime_ms,
            fill_ratio=fill_ratio,
            pull_ratio=pull_ratio,
            price_reaction_bps=price_reaction_bps,
            is_short_lived=is_short_lived,
            is_low_fill=is_low_fill,
            is_high_pull=is_high_pull,
        )

        reason = self._build_reason(
            wall=wall,
            pulled_notional=pulled_notional,
            lifetime_ms=lifetime_ms,
            fill_ratio=fill_ratio,
            pull_ratio=pull_ratio,
            price_reaction_bps=price_reaction_bps,
            is_short_lived=is_short_lived,
            is_low_fill=is_low_fill,
            is_high_pull=is_high_pull,
        )

        return FakeLiquidityCandidateContext(
            wall=wall,
            wall_notional=wall_notional,
            pulled_notional=pulled_notional,
            lifetime_ms=lifetime_ms,
            fill_ratio=fill_ratio,
            pull_ratio=pull_ratio,
            price_reaction_bps=price_reaction_bps,
            distance_from_mid_bps=distance_from_mid_bps,
            is_short_lived=is_short_lived,
            is_low_fill=is_low_fill,
            is_high_pull=is_high_pull,
            has_market_reaction=has_market_reaction,
            confidence=confidence,
            score=score,
            reason=reason,
        )

    def _build_features(
        self,
        *,
        candidate: FakeLiquidityCandidateContext,
        repetition_count: int | None = None,
    ) -> SpoofingFeatures:
        wall = candidate.wall

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
            distance_from_mid_bps=candidate.distance_from_mid_bps,
            lifetime_ms=candidate.lifetime_ms,
            updates_count=wall.updates_count,
            repetition_count=repetition,
            fill_ratio=candidate.fill_ratio,
            pull_ratio=candidate.pull_ratio,
            cancel_to_fill_ratio=cancel_to_fill_ratio,
            price_reaction_bps=candidate.price_reaction_bps,
            pressure_flip_strength=0.0,
            layering_score=0.0,
            is_near_best_quote=is_near_best_quote,
            is_fast_pull=candidate.is_short_lived,
            is_fake_liquidity=True,
            is_layering=False,
            metadata={
                "wall_notional": candidate.wall_notional,
                "pulled_notional": candidate.pulled_notional,
                "estimated_pulled_size": wall.estimated_pulled_size,
                "estimated_filled_size": wall.estimated_filled_size,
                "current_to_max_ratio": wall.current_to_max_ratio,
                "detector": self.component.value,
            },
        )

    # -------------------------------------------------------------------------
    # Score / confidence
    # -------------------------------------------------------------------------

    def _compute_score(
        self,
        *,
        wall: TrackedWall,
        pulled_notional: float,
        lifetime_ms: float,
        fill_ratio: float,
        pull_ratio: float,
        price_reaction_bps: float,
        is_short_lived: bool,
        is_low_fill: bool,
        is_high_pull: bool,
    ) -> float:
        cfg = self.config.fake_liquidity

        # 1. Коротке життя = підозріло
        max_lifetime = max(float(cfg.max_lifetime_ms), 1.0)
        lifetime_component = 1.0 - self.clamp(lifetime_ms / max_lifetime, 0.0, 1.0)

        # 2. Низький fill = підозріло
        max_fill = max(cfg.max_fill_ratio, 1e-12)
        fill_component = 1.0 - self.clamp(fill_ratio / max_fill, 0.0, 1.0)

        # 3. Високий pull = підозріло
        min_pull = max(cfg.min_pull_ratio, 1e-12)
        pull_component = (pull_ratio - min_pull) / max(1.0 - min_pull, 1e-12)
        pull_component = self.clamp(pull_component, 0.0, 1.0)

        # 4. Реакція ринку
        reaction_component = self.clamp(
            price_reaction_bps / max(cfg.min_price_reaction_bps * 3.0, 1e-12),
            0.0,
            1.0,
        )

        # 5. Розмір знятої ліквідності
        min_removed = max(self.config.pull_detection.min_removed_notional, 1e-12)
        notional_component = (pulled_notional - min_removed) / max(min_removed * 2.0, 1e-12)
        notional_component = self.clamp(notional_component, 0.0, 1.0)

        bonus = 0.0
        if is_short_lived:
            bonus += 0.05
        if is_low_fill:
            bonus += 0.05
        if is_high_pull:
            bonus += 0.05

        raw_score = (
            0.22 * lifetime_component +
            0.22 * fill_component +
            0.22 * pull_component +
            0.22 * reaction_component +
            0.12 * notional_component +
            bonus
        )

        if wall.state == OrderbookWallState.PULLED:
            raw_score += 0.03

        return self.clamp(raw_score, 0.0, 1.0)

    def _compute_confidence(
        self,
        *,
        wall: TrackedWall,
        pulled_notional: float,
        lifetime_ms: float,
        fill_ratio: float,
        pull_ratio: float,
        price_reaction_bps: float,
        is_short_lived: bool,
        is_low_fill: bool,
        is_high_pull: bool,
    ) -> float:
        confidence = 0.42

        if is_short_lived:
            confidence += 0.12

        if is_low_fill:
            confidence += 0.12

        if is_high_pull:
            confidence += 0.10

        if price_reaction_bps >= self.config.fake_liquidity.min_price_reaction_bps * 2.0:
            confidence += 0.10

        if pulled_notional >= self.config.pull_detection.min_removed_notional * 2.0:
            confidence += 0.06

        if wall.state == OrderbookWallState.PULLED:
            confidence += 0.04

        if wall.touch_count == 0 and wall.near_touch_count == 0:
            confidence += 0.03

        return self.clamp(confidence, 0.0, 0.99)

    # -------------------------------------------------------------------------
    # Price reaction helpers
    # -------------------------------------------------------------------------

    def _estimate_price_reaction_bps(
        self,
        *,
        wall: TrackedWall,
        current_mid_price: float | None,
    ) -> float:
        """
        Евристично оцінює реакцію ціни після поведінки ліквідності.

        Логіка:
        - ASK wall fake liquidity підозріла, якщо після її зникнення ціна пішла ВГОРУ
        - BID wall fake liquidity підозріла, якщо після її зникнення ціна пішла ВНИЗ

        Повертається абсолютна релевантна реакція в bps.
        """
        reference = wall.mid_price_at_creation
        current = current_mid_price

        if reference is None or reference <= 0 or current is None or current <= 0:
            return 0.0

        signed_move = self.signed_bps_move(current, reference)

        if wall.side.value == "ask":
            # ask wall тиснула зверху; якщо після зникнення ціна росте — це релевантно
            return max(0.0, signed_move)

        if wall.side.value == "bid":
            # bid wall підтримувала знизу; якщо після зникнення ціна падає — це релевантно
            return max(0.0, -signed_move)

        return 0.0

    def _estimate_distance_from_mid_bps(
        self,
        *,
        wall: TrackedWall,
        current_mid_price: float | None,
    ) -> float:
        if current_mid_price is not None and current_mid_price > 0:
            return self.bps_distance(wall.price, current_mid_price)

        if wall.mid_price_at_creation is not None and wall.mid_price_at_creation > 0:
            return self.bps_distance(wall.price, wall.mid_price_at_creation)

        return 0.0

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
        pulled_notional: float,
        lifetime_ms: float,
        fill_ratio: float,
        pull_ratio: float,
        price_reaction_bps: float,
        is_short_lived: bool,
        is_low_fill: bool,
        is_high_pull: bool,
    ) -> str:
        parts = [
            f"fake liquidity candidate detected for {wall.side.value.upper()} wall",
            f"pulled_notional={pulled_notional:.2f}",
            f"pull_ratio={pull_ratio:.4f}",
            f"fill_ratio={fill_ratio:.4f}",
            f"lifetime_ms={lifetime_ms:.2f}",
            f"price_reaction_bps={price_reaction_bps:.4f}",
            f"state={wall.state.value}",
        ]

        if is_short_lived:
            parts.append("short_lived=true")
        if is_low_fill:
            parts.append("low_fill=true")
        if is_high_pull:
            parts.append("high_pull=true")

        return ", ".join(parts)