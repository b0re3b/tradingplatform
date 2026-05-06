from __future__ import annotations

from dataclasses import dataclass

from core.event_bus import EventBus
from core.scheduler import Scheduler

from analytics.spoofing import (
    SpoofingPattern,
    SpoofingSeverity,
    SpoofingSide,
    SpoofingSignal,
    SpoofingType,
)

from .base_spoofing_strategy import (
    BaseSpoofingStrategy,
    BaseSpoofingStrategyConfig,
    SetupStatus,
    SpoofingTradeSetup,
    StrategyDirection,
)


@dataclass(slots=True)
class FakeLiquidityTrapStrategyConfig(BaseSpoofingStrategyConfig):
    """
    Конфіг для fake liquidity trap strategy.

    Логіка:
    - приймаємо тільки дуже специфічні сигнали fake liquidity / fake absorption;
    - хочемо бачити короткий lifetime, low fill, high pull, вже наявну реакцію ринку;
    - торгуємо continuation / trap unwind після зникнення фейкової ліквідності.
    """

    # accepted signals
    allow_fake_liquidity_type: bool = True
    allow_fake_absorption_pattern: bool = True
    allow_composite_if_fake_liquidity_flag: bool = True

    # stronger filtering than generic reversal
    min_trap_score: float = 0.72
    min_trap_confidence: float = 0.62
    min_pull_ratio: float = 0.70
    max_fill_ratio: float = 0.20
    min_price_reaction_bps: float = 2.0
    max_lifetime_ms: float = 4_500.0

    # require features / trap semantics
    require_fake_liquidity_flag: bool = False
    require_short_lived_wall: bool = True
    require_market_reaction: bool = True

    # confirmation
    confirmation_move_bps: float = 1.0
    high_severity_confirmation_move_bps: float = 0.7
    require_price_beyond_entry: bool = True
    require_reaction_not_fading: bool = True

    # trap retest logic
    allow_retest_entry: bool = True
    max_retest_distance_bps: float = 1.2
    invalidate_if_deep_reentry_bps: float = 1.8

    # invalidation
    max_adverse_move_bps_trap: float = 2.0
    invalidate_on_signal_confidence_drop_below: float = 0.50
    invalidate_on_signal_score_drop_below: float = 0.55

    # pricing
    entry_offset_bps_trap: float = 0.10
    stop_buffer_bps_trap: float = 2.5
    take_profit_bps_trap: float = 8.0
    trap_tp_multiplier: float = 1.50
    min_take_profit_bps: float = 5.0
    max_take_profit_bps: float = 20.0

    # metadata
    keep_source_wall_reference: bool = True


class FakeLiquidityTrapStrategy(BaseSpoofingStrategy):
    """
    Strategy: trade fake-liquidity trap continuation.

    Типова інтерпретація:
    - ASK fake liquidity vanished -> market can rip up -> LONG;
    - BID fake liquidity vanished -> market can flush down -> SHORT.

    Це не просто spoofing reversal, а більш вузький сценарій:
    натовп повірив у ліквідність / тиск, ліквідність зникла, ринок рухається
    в протилежний бік, і ми хочемо забрати continuation цього move.
    """

    strategy_name = "fake_liquidity_trap_strategy"

    def __init__(
        self,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        config: FakeLiquidityTrapStrategyConfig | None = None,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config or FakeLiquidityTrapStrategyConfig(),
        )
        self.config: FakeLiquidityTrapStrategyConfig

    # -------------------------------------------------------------------------
    # pattern support
    # -------------------------------------------------------------------------

    def supports_pattern(self, signal: SpoofingSignal) -> bool:
        """
        Приймаємо тільки fake liquidity style signals.
        """
        if (
            self.config.allow_fake_liquidity_type
            and signal.spoofing_type == SpoofingType.FAKE_LIQUIDITY
        ):
            return True

        if (
            self.config.allow_fake_absorption_pattern
            and signal.pattern == SpoofingPattern.FAKE_ABSORPTION
        ):
            return True

        if (
            self.config.allow_composite_if_fake_liquidity_flag
            and signal.spoofing_type == SpoofingType.COMPOSITE
            and self._feature_bool(signal.features, "is_fake_liquidity")
        ):
            return True

        return False

    def accepts_signal(self, signal: SpoofingSignal) -> bool:
        """
        Trap-specific acceptance filter.
        """
        if not super().accepts_signal(signal):
            return False

        if signal.score < self.config.min_trap_score:
            return False

        if signal.confidence < self.config.min_trap_confidence:
            return False

        features = signal.features
        if features is None:
            return False

        pull_ratio = self._feature_float(features, "pull_ratio")
        fill_ratio = self._feature_float(features, "fill_ratio")
        price_reaction_bps = abs(self._feature_float(features, "price_reaction_bps"))
        lifetime_ms = self._feature_float(features, "lifetime_ms")
        is_fake_liquidity = self._feature_bool(features, "is_fake_liquidity")
        has_market_reaction = self._feature_bool(features, "has_market_reaction")
        is_fast_pull = self._feature_bool(features, "is_fast_pull")

        if self.config.require_fake_liquidity_flag and not is_fake_liquidity:
            return False

        if pull_ratio < self.config.min_pull_ratio:
            return False

        if fill_ratio > self.config.max_fill_ratio:
            return False

        if self.config.require_market_reaction:
            if price_reaction_bps < self.config.min_price_reaction_bps:
                if not has_market_reaction:
                    return False

        if self.config.require_short_lived_wall:
            if lifetime_ms > 0 and lifetime_ms > self.config.max_lifetime_ms:
                return False

        # Додатковий quality gate:
        # якщо реакція ще слабка, хочемо хоча б fast pull.
        if price_reaction_bps < self.config.min_price_reaction_bps:
            if not is_fast_pull:
                return False

        return True

    # -------------------------------------------------------------------------
    # setup building
    # -------------------------------------------------------------------------

    def build_setup(self, signal: SpoofingSignal) -> SpoofingTradeSetup | None:
        """
        Створює trap setup.
        """
        setup = super().build_setup(signal)
        if setup is None:
            return None

        direction = setup.direction
        reference_price = setup.reference_price

        setup.entry_price = self.compute_entry_price(signal, direction, reference_price)
        setup.stop_price = self.compute_stop_price(signal, direction, reference_price)
        setup.take_profit_price = self.compute_take_profit_price(
            signal=signal,
            direction=direction,
            entry_price=setup.entry_price,
            stop_price=setup.stop_price,
            reference_price=reference_price,
        )

        retest_price = self._compute_retest_price(
            direction=direction,
            reference_price=reference_price,
            entry_price=setup.entry_price,
        )

        setup.metadata["trap_reason"] = self._build_trap_reason(signal)
        setup.metadata["trap_mode"] = "fake_liquidity_unwind"
        setup.metadata["retest_price"] = retest_price
        setup.metadata["allow_retest_entry"] = self.config.allow_retest_entry
        setup.metadata["pull_ratio"] = self._feature_float(signal.features, "pull_ratio")
        setup.metadata["fill_ratio"] = self._feature_float(signal.features, "fill_ratio")
        setup.metadata["price_reaction_bps"] = self._feature_float(
            signal.features,
            "price_reaction_bps",
        )
        setup.metadata["lifetime_ms"] = self._feature_float(signal.features, "lifetime_ms")
        setup.metadata["is_fake_liquidity"] = self._feature_bool(
            signal.features,
            "is_fake_liquidity",
        )
        setup.metadata["has_market_reaction"] = self._feature_bool(
            signal.features,
            "has_market_reaction",
        )
        setup.metadata["is_fast_pull"] = self._feature_bool(signal.features, "is_fast_pull")

        if self.config.keep_source_wall_reference:
            setup.metadata["wall_id"] = signal.wall_id

        return setup

    def enrich_setup(self, setup: SpoofingTradeSetup, signal: SpoofingSignal) -> None:
        """
        Trap-specific metadata.
        """
        setup.metadata["signal_side"] = signal.side.value
        setup.metadata["expected_direction"] = setup.direction.value
        setup.metadata["pattern"] = signal.pattern.value
        setup.metadata["spoofing_type"] = signal.spoofing_type.value
        setup.metadata["severity"] = signal.severity.value

    # -------------------------------------------------------------------------
    # pricing
    # -------------------------------------------------------------------------

    def compute_entry_price(
        self,
        signal: SpoofingSignal,
        direction: StrategyDirection,
        reference_price: float,
    ) -> float:
        """
        Entry трохи за trap-zone в бік unwind continuation.
        """
        offset_ratio = self.config.entry_offset_bps_trap / 10_000.0

        if direction == StrategyDirection.LONG:
            return reference_price * (1.0 + offset_ratio)

        if direction == StrategyDirection.SHORT:
            return reference_price * (1.0 - offset_ratio)

        return reference_price

    def compute_stop_price(
        self,
        signal: SpoofingSignal,
        direction: StrategyDirection,
        reference_price: float,
    ) -> float:
        """
        Stop за fake-liquidity zone.
        """
        buffer_ratio = self.config.stop_buffer_bps_trap / 10_000.0

        if direction == StrategyDirection.LONG:
            return reference_price * (1.0 - buffer_ratio)

        if direction == StrategyDirection.SHORT:
            return reference_price * (1.0 + buffer_ratio)

        return reference_price

    def compute_take_profit_price(
        self,
        *,
        signal: SpoofingSignal,
        direction: StrategyDirection,
        entry_price: float,
        stop_price: float,
        reference_price: float,
    ) -> float:
        """
        TP для trap strategy:
        - базовий RR;
        - optional scaling від сили reaction.
        """
        base_tp = super().compute_take_profit_price(
            signal=signal,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            reference_price=reference_price,
        )

        price_reaction_bps = abs(self._feature_float(signal.features, "price_reaction_bps"))
        if price_reaction_bps <= 0:
            fallback_ratio = self.config.take_profit_bps_trap / 10_000.0

            if direction == StrategyDirection.LONG:
                fallback_tp = entry_price * (1.0 + fallback_ratio)
                return max(base_tp, fallback_tp)

            if direction == StrategyDirection.SHORT:
                fallback_tp = entry_price * (1.0 - fallback_ratio)
                return min(base_tp, fallback_tp)

            return base_tp

        target_bps = price_reaction_bps * self.config.trap_tp_multiplier
        target_bps = max(self.config.min_take_profit_bps, target_bps)
        target_bps = min(self.config.max_take_profit_bps, target_bps)

        target_ratio = target_bps / 10_000.0

        if direction == StrategyDirection.LONG:
            reaction_tp = entry_price * (1.0 + target_ratio)
            return max(base_tp, reaction_tp)

        if direction == StrategyDirection.SHORT:
            reaction_tp = entry_price * (1.0 - target_ratio)
            return min(base_tp, reaction_tp)

        return base_tp

    # -------------------------------------------------------------------------
    # signal updates
    # -------------------------------------------------------------------------

    def apply_signal_update(self, *, setup: SpoofingTradeSetup, signal: SpoofingSignal) -> None:
        """
        Setup updates on stronger / fresher fake-liquidity context.
        """
        super().apply_signal_update(setup=setup, signal=signal)

        if setup.status in {SetupStatus.PENDING, SetupStatus.CONFIRMED}:
            setup.take_profit_price = self.compute_take_profit_price(
                signal=signal,
                direction=setup.direction,
                entry_price=setup.entry_price,
                stop_price=setup.stop_price,
                reference_price=setup.reference_price,
            )

        setup.metadata["updated_pull_ratio"] = self._feature_float(signal.features, "pull_ratio")
        setup.metadata["updated_fill_ratio"] = self._feature_float(signal.features, "fill_ratio")
        setup.metadata["updated_price_reaction_bps"] = self._feature_float(
            signal.features,
            "price_reaction_bps",
        )
        setup.metadata["updated_lifetime_ms"] = self._feature_float(
            signal.features,
            "lifetime_ms",
        )

    def should_invalidate_from_signal_update(
        self,
        *,
        setup: SpoofingTradeSetup,
        signal: SpoofingSignal,
    ) -> bool:
        """
        Invalidate якщо fake-liquidity thesis більше не виглядає сильною.
        """
        if signal.confidence < self.config.invalidate_on_signal_confidence_drop_below:
            return True

        if signal.score < self.config.invalidate_on_signal_score_drop_below:
            return True

        features = signal.features
        if features is None:
            return False

        pull_ratio = self._feature_float(features, "pull_ratio")
        fill_ratio = self._feature_float(features, "fill_ratio")
        lifetime_ms = self._feature_float(features, "lifetime_ms")
        price_reaction_bps = abs(self._feature_float(features, "price_reaction_bps"))

        if pull_ratio > 0 and pull_ratio < self.config.min_pull_ratio * 0.85:
            return True

        if fill_ratio > max(self.config.max_fill_ratio, 0.30):
            return True

        if self.config.require_short_lived_wall:
            if lifetime_ms > 0 and lifetime_ms > self.config.max_lifetime_ms * 1.20:
                return True

        if self.config.require_reaction_not_fading:
            if price_reaction_bps < self.config.min_price_reaction_bps * 0.60:
                return True

        return False

    # -------------------------------------------------------------------------
    # confirmation / trigger / invalidation
    # -------------------------------------------------------------------------

    def confirm_setup(
        self,
        *,
        setup: SpoofingTradeSetup,
        current_price: float,
        signal: SpoofingSignal,
    ) -> bool:
        """
        Trap-specific confirmation.

        Ми хочемо:
        - вже наявний рух у бік trap unwind;
        - бажано, щоб ціна не просто торкнула reference, а пішла за entry zone;
        - fake-liquidity semantics мають залишатись валідними.
        """
        if setup.status != SetupStatus.PENDING:
            return False

        if current_price <= 0 or setup.reference_price <= 0:
            return False

        required_bps = self.config.confirmation_move_bps
        if setup.severity in {SpoofingSeverity.HIGH, SpoofingSeverity.CRITICAL}:
            required_bps = min(required_bps, self.config.high_severity_confirmation_move_bps)

        move_bps = self.signed_bps_move(
            current_price=current_price,
            reference_price=setup.reference_price,
        )

        if setup.direction == StrategyDirection.LONG:
            if self.config.require_price_beyond_entry and current_price < setup.entry_price:
                return False
            passed = move_bps >= required_bps

        elif setup.direction == StrategyDirection.SHORT:
            if self.config.require_price_beyond_entry and current_price > setup.entry_price:
                return False
            passed = move_bps <= -required_bps

        else:
            passed = False

        if not passed:
            return False

        pull_ratio = self._feature_float(signal.features, "pull_ratio")
        fill_ratio = self._feature_float(signal.features, "fill_ratio")
        price_reaction_bps = abs(self._feature_float(signal.features, "price_reaction_bps"))

        if pull_ratio < self.config.min_pull_ratio:
            return False

        if fill_ratio > self.config.max_fill_ratio:
            return False

        if (
            self.config.require_market_reaction
            and price_reaction_bps < self.config.min_price_reaction_bps
        ):
            return False

        setup.status = SetupStatus.CONFIRMED
        setup.confirmed_at = self.now()
        setup.confirmation_price = current_price
        setup.metadata["confirmation_move_bps"] = move_bps
        setup.metadata["confirmation_required_bps"] = required_bps

        self._stats["setups_confirmed"] += 1

        self.log_info(
            "Fake liquidity trap setup confirmed",
            setup_id=setup.setup_id,
            symbol=setup.symbol,
            direction=setup.direction.value,
            current_price=current_price,
            move_bps=move_bps,
            required_bps=required_bps,
        )
        return True

    def should_trigger_entry(
        self,
        *,
        setup: SpoofingTradeSetup,
        current_price: float,
    ) -> bool:
        """
        Trigger logic:
        - confirmed continuation;
        - або retest logic, якщо увімкнено.
        """
        if setup.status != SetupStatus.CONFIRMED:
            return False

        retest_price = self._safe_float(setup.metadata.get("retest_price"))

        if not self.config.allow_retest_entry:
            if setup.direction == StrategyDirection.LONG:
                return current_price >= setup.entry_price
            if setup.direction == StrategyDirection.SHORT:
                return current_price <= setup.entry_price
            return False

        if setup.direction == StrategyDirection.LONG:
            return current_price >= min(setup.entry_price, retest_price or setup.entry_price)

        if setup.direction == StrategyDirection.SHORT:
            return current_price <= max(setup.entry_price, retest_price or setup.entry_price)

        return False

    def should_invalidate_on_price(
        self,
        *,
        setup: SpoofingTradeSetup,
        current_price: float,
    ) -> bool:
        """
        Invalidation:
        - deep re-entry у trap zone;
        - adverse move.
        """
        adverse_bps = self._compute_adverse_move_bps(
            setup=setup,
            current_price=current_price,
        )
        if adverse_bps >= self.config.max_adverse_move_bps_trap:
            return True

        deep_reentry_ratio = self.config.invalidate_if_deep_reentry_bps / 10_000.0

        if setup.direction == StrategyDirection.LONG:
            deep_reentry_price = setup.reference_price * (1.0 - deep_reentry_ratio)
            if current_price <= deep_reentry_price:
                return True

        elif setup.direction == StrategyDirection.SHORT:
            deep_reentry_price = setup.reference_price * (1.0 + deep_reentry_ratio)
            if current_price >= deep_reentry_price:
                return True

        return False

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------

    def _compute_retest_price(
        self,
        *,
        direction: StrategyDirection,
        reference_price: float,
        entry_price: float,
    ) -> float:
        """
        Розрахунок trap retest zone.
        """
        if reference_price <= 0 or entry_price <= 0:
            return entry_price

        max_retest_ratio = self.config.max_retest_distance_bps / 10_000.0

        if direction == StrategyDirection.LONG:
            candidate = entry_price * (1.0 - max_retest_ratio)
            return max(candidate, reference_price)

        if direction == StrategyDirection.SHORT:
            candidate = entry_price * (1.0 + max_retest_ratio)
            return min(candidate, reference_price)

        return entry_price

    def _build_trap_reason(self, signal: SpoofingSignal) -> str:
        """
        Людинозрозумілий опис trap-ідеї.
        """
        if signal.side == SpoofingSide.ASK:
            base = "ask-side fake liquidity vanished -> upside trap unwind"
        elif signal.side == SpoofingSide.BID:
            base = "bid-side fake liquidity vanished -> downside trap unwind"
        else:
            base = "fake liquidity trap"

        return f"{base}; pattern={signal.pattern.value}; type={signal.spoofing_type.value}"