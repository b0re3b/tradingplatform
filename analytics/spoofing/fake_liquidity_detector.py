from __future__ import annotations

from typing import Iterable

from core.event_bus import EventBus
from core.scheduler import Scheduler

from .base import BaseSpoofingDetector
from .config import SpoofingConfig
from .enums import (
    DetectorDecision,
    OrderbookWallState,
    SpoofingComponent,
    SpoofingPattern,
    SpoofingSide,
)
from .models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    DetectorResult,
    FakeLiquidityCandidateContext,
    SpoofingFeatures,
    SpoofingKey,
    TrackedWall,
    spoofing_key_to_dict,
)
from .persistence_tracker import PersistenceTracker


class FakeLiquidityDetector(BaseSpoofingDetector):
    """
    Detector фейкової ліквідності.

    Основна ідея:
    - велика ліквідність з'явилась у стакані;
    - простояла недовго;
    - майже не була виконана;
    - значною мірою була знята;
    - після цього ринок відреагував у релевантний бік.

    Correct scope:
        exchange + market_type + symbol + timeframe

    Correct production input flow:
        exchange adapters
            -> market.orderbook
            -> OrderBookCache
            -> market.orderbook.updated
            -> SpoofingAnalyzer
            -> PersistenceTracker
            -> FakeLiquidityDetector

    Важливо:
    - працює поверх PersistenceTracker state;
    - не працює як raw orderbook parser;
    - не читає exchange adapters напряму;
    - не підписується на EventBus;
    - не публікує події;
    - не запускає Scheduler jobs;
    - повертає тільки DetectorResult або None.
    """

    component = SpoofingComponent.FAKE_LIQUIDITY_DETECTOR

    def __init__(
        self,
        *,
        event_bus: EventBus | None,
        scheduler: Scheduler | None,
        config: SpoofingConfig,
        persistence_tracker: PersistenceTracker,
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
                "scope": spoofing_key_to_dict(wall.key),
                "exchange_symbol": wall.exchange_symbol,
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

    def analyze_key(
        self,
        *,
        key: SpoofingKey,
        current_mid_price: float | None = None,
    ) -> list[DetectorResult]:
        """
        Key-first API для scoped futures market.

        key:
            exchange + market_type + symbol + timeframe
        """
        if not self.config.enabled or not self.config.fake_liquidity.enabled:
            return []

        walls = self.persistence_tracker.get_walls_for_key(key)

        return self.analyze_many(
            walls=walls,
            key=key,
            current_mid_price=current_mid_price,
        )

    def analyze_many(
        self,
        walls: Iterable[TrackedWall],
        *,
        key: SpoofingKey | None = None,
        exchange: str | None = None,
        symbol: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        current_mid_price: float | None = None,
    ) -> list[DetectorResult]:
        """
        Аналізує набір tracked walls і повертає позитивні fake-liquidity candidates.

        New code should pass key=SpoofingKey.
        Legacy filters exchange/symbol/market_type/timeframe залишені для міграції.
        """
        if not self.config.enabled or not self.config.fake_liquidity.enabled:
            return []

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

        results: list[DetectorResult] = []

        for wall in walls:
            if key is not None and wall.key != key:
                continue
            if normalized_exchange is not None and wall.exchange != normalized_exchange:
                continue
            if normalized_market_type is not None and wall.market_type != normalized_market_type:
                continue
            if normalized_symbol is not None and wall.symbol != normalized_symbol:
                continue
            if normalized_timeframe is not None and wall.timeframe != normalized_timeframe:
                continue
            if not self.should_process_key(wall.key):
                continue

            repetition_count = self._estimate_repetition_count(wall)
            result = self.analyze(
                wall,
                current_mid_price=current_mid_price,
                repetition_count=repetition_count,
            )
            if result is not None and result.is_positive():
                results.append(result)

        results.sort(key=lambda item: (item.score, item.confidence), reverse=True)
        return results

    def analyze_scope(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        current_mid_price: float | None = None,
    ) -> list[DetectorResult]:
        """
        Аналізує всі tracked walls одного scoped futures market.
        """
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        return self.analyze_key(
            key=key,
            current_mid_price=current_mid_price,
        )

    def analyze_symbol(
        self,
        *,
        exchange: str,
        symbol: str,
        current_mid_price: float | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> list[DetectorResult]:
        """
        Backward-compatible helper.

        New code should use analyze_key() або analyze_scope().
        Якщо market_type/timeframe не передані, аналізує всі scope-и для
        exchange + symbol.
        """
        walls = self.persistence_tracker.get_walls_for_symbol(
            exchange=exchange,
            symbol=symbol,
            market_type=market_type,
            timeframe=timeframe,
        )

        return self.analyze_many(
            walls=walls,
            exchange=exchange,
            symbol=symbol,
            market_type=market_type,
            timeframe=timeframe,
            current_mid_price=current_mid_price,
        )

    def is_fake_liquidity_candidate(
        self,
        wall: TrackedWall,
        *,
        current_mid_price: float | None = None,
    ) -> bool:
        """
        Boolean helper для швидкої перевірки tracked wall.
        """
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
        if not self.config.enabled or not self.config.fake_liquidity.enabled:
            return None

        if not self.should_process_key(wall.key):
            return None

        if wall.max_size <= 0.0 or wall.price <= 0.0:
            return None

        wall_notional = wall.price * wall.max_size
        pulled_notional = wall.price * wall.estimated_pulled_size
        lifetime_ms = wall.lifetime_ms
        fill_ratio = wall.fill_ratio
        pull_ratio = wall.pull_ratio

        if not self._passes_basic_filters(
            wall=wall,
            wall_notional=wall_notional,
            lifetime_ms=lifetime_ms,
            fill_ratio=fill_ratio,
            pull_ratio=pull_ratio,
        ):
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
        has_market_reaction = (
            price_reaction_bps >= self.config.fake_liquidity.min_price_reaction_bps
        )

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
            fill_ratio=fill_ratio,
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

    def _passes_basic_filters(
        self,
        *,
        wall: TrackedWall,
        wall_notional: float,
        lifetime_ms: float,
        fill_ratio: float,
        pull_ratio: float,
    ) -> bool:
        cfg = self.config.fake_liquidity

        if wall_notional < self.config.wall_detection.min_wall_size_abs:
            return False

        if fill_ratio > cfg.max_fill_ratio:
            return False

        if pull_ratio < cfg.min_pull_ratio:
            return False

        if lifetime_ms <= 0 or lifetime_ms > cfg.max_lifetime_ms:
            return False

        if wall.state not in {
            OrderbookWallState.PULLED,
            OrderbookWallState.WEAKENING,
            OrderbookWallState.EXPIRED,
            OrderbookWallState.FILLED,
        }:
            return False

        return True

    def _build_features(
        self,
        *,
        candidate: FakeLiquidityCandidateContext,
        repetition_count: int | None = None,
    ) -> SpoofingFeatures:
        wall = candidate.wall

        cancel_to_fill_ratio = self._compute_cancel_to_fill_ratio(
            pull_ratio=candidate.pull_ratio,
            fill_ratio=candidate.fill_ratio,
        )

        repetition = (
            repetition_count
            if repetition_count is not None
            else self._estimate_repetition_count(wall)
        )

        is_near_best_quote = wall.near_touch_count > 0 or wall.touch_count > 0

        return SpoofingFeatures(
            symbol=wall.symbol,
            exchange=wall.exchange,
            market_type=wall.market_type,
            timeframe=wall.timeframe,
            exchange_symbol=wall.exchange_symbol,
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
                "scope": spoofing_key_to_dict(wall.key),
                "wall_id": wall.wall_id,
                "wall_notional": candidate.wall_notional,
                "pulled_notional": candidate.pulled_notional,
                "estimated_pulled_size": wall.estimated_pulled_size,
                "estimated_filled_size": wall.estimated_filled_size,
                "current_to_max_ratio": wall.current_to_max_ratio,
                "wall_state": wall.state.value,
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

        max_lifetime = max(float(cfg.max_lifetime_ms), 1.0)
        lifetime_component = 1.0 - self.clamp(lifetime_ms / max_lifetime, 0.0, 1.0)

        max_fill = max(cfg.max_fill_ratio, 1e-12)
        fill_component = 1.0 - self.clamp(fill_ratio / max_fill, 0.0, 1.0)

        min_pull = max(cfg.min_pull_ratio, 1e-12)
        pull_component = (pull_ratio - min_pull) / max(1.0 - min_pull, 1e-12)
        pull_component = self.clamp(pull_component, 0.0, 1.0)

        reaction_component = self.clamp(
            price_reaction_bps / max(cfg.min_price_reaction_bps * 3.0, 1e-12),
            0.0,
            1.0,
        )

        min_removed = max(self.config.pull_detection.min_removed_notional, 1e-12)
        notional_component = (pulled_notional - min_removed) / max(
            min_removed * 2.0,
            1e-12,
        )
        notional_component = self.clamp(notional_component, 0.0, 1.0)

        bonus = 0.0
        if is_short_lived:
            bonus += 0.05
        if is_low_fill:
            bonus += 0.05
        if is_high_pull:
            bonus += 0.05
        if wall.state == OrderbookWallState.PULLED:
            bonus += 0.03

        raw_score = (
            0.22 * lifetime_component
            + 0.22 * fill_component
            + 0.22 * pull_component
            + 0.22 * reaction_component
            + 0.12 * notional_component
            + bonus
        )

        return self.clamp(raw_score, 0.0, 1.0)

    def _compute_confidence(
        self,
        *,
        wall: TrackedWall,
        pulled_notional: float,
        fill_ratio: float,
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

        ASK wall fake liquidity підозріла, якщо після її зникнення ціна йде вгору.
        BID wall fake liquidity підозріла, якщо після її зникнення ціна йде вниз.
        """
        reference = wall.mid_price_at_creation
        current = current_mid_price

        if reference is None or reference <= 0 or current is None or current <= 0:
            return 0.0

        signed_move = self.signed_bps_move(current, reference)

        if wall.side == SpoofingSide.ASK:
            return max(0.0, signed_move)

        if wall.side == SpoofingSide.BID:
            return max(0.0, -signed_move)

        return 0.0

    def _estimate_distance_from_mid_bps(
        self,
        *,
        wall: TrackedWall,
        current_mid_price: float | None,
    ) -> float:
        reference_mid = current_mid_price or wall.mid_price_at_creation
        if reference_mid is None or reference_mid <= 0:
            return 0.0
        return self.bps_distance(wall.price, reference_mid)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _estimate_repetition_count(self, wall: TrackedWall) -> int:
        history = self.persistence_tracker.get_recent_history(
            exchange=wall.exchange,
            market_type=wall.market_type,
            symbol=wall.symbol,
            timeframe=wall.timeframe,
            side=wall.side,
            price=wall.price,
            limit=100,
        )
        return len(history)

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

    @staticmethod
    def _build_reason(
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
            f"exchange={wall.exchange}",
            f"market_type={wall.market_type}",
            f"symbol={wall.symbol}",
            f"timeframe={wall.timeframe}",
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


__all__ = ["FakeLiquidityDetector"]