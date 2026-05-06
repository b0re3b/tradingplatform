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
class SpoofingReversalStrategyConfig(BaseSpoofingStrategyConfig):
    """
    Конфіг для reversal-стратегії поверх spoofing-сигналів.

    Ідея:
    - торгуємо реверс після зняття фейкового тиску;
    - prefer signal-и з вираженим price reaction / pull / pressure flip.
    """

    # accepted patterns / types
    allow_pull_and_reversal: bool = True
    allow_pressure_bluff: bool = True
    allow_multi_level_layering: bool = True
    allow_composite: bool = True

    # stronger filters for reversal
    min_reversal_score: float = 0.68
    min_reversal_confidence: float = 0.58
    min_price_reaction_bps: float = 1.5
    min_pull_ratio: float = 0.55
    max_fill_ratio: float = 0.35

    # optional feature-based filters
    require_fast_pull_or_reaction: bool = False
    require_reaction_for_pressure_bluff: bool = True

    # confirmation
    confirmation_move_bps: float = 1.2
    confirmation_move_bps_high_severity: float = 0.8
    require_price_beyond_reference: bool = True

    # invalidation
    max_adverse_move_bps_reversal: float = 2.2
    invalidate_on_signal_confidence_drop_below: float = 0.45
    invalidate_on_signal_score_drop_below: float = 0.50

    # pricing
    entry_offset_bps_reversal: float = 0.15
    stop_buffer_bps_reversal: float = 2.8
    take_profit_bps_reversal: float = 7.0

    # target shaping
    use_reaction_scaled_take_profit: bool = True
    reaction_tp_multiplier: float = 1.35
    min_take_profit_bps: float = 4.0
    max_take_profit_bps: float = 18.0

    # metadata / state
    keep_reference_to_source_wall: bool = True


class SpoofingReversalStrategy(BaseSpoofingStrategy):
    """
    Concrete strategy: reversal after spoofing pressure removal.

    Типові кейси:
    - ASK spoof wall disappears -> LONG;
    - BID spoof wall disappears -> SHORT.

    Підтримувані джерела:
    - ORDER_PULL / PULL_AND_REVERSAL;
    - FLIP_PRESSURE / PRESSURE_BLUFF;
    - LAYERING / MULTI_LEVEL_LAYERING;
    - COMPOSITE spoofing signals.
    """

    strategy_name = "spoofing_reversal_strategy"

    def __init__(
        self,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        config: SpoofingReversalStrategyConfig | None = None,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config or SpoofingReversalStrategyConfig(),
        )
        self.config: SpoofingReversalStrategyConfig

    # -------------------------------------------------------------------------
    # pattern support
    # -------------------------------------------------------------------------

    def supports_pattern(self, signal: SpoofingSignal) -> bool:
        """
        Чи є цей spoofing signal релевантним для reversal-ідеї.
        """
        pattern = signal.pattern
        spoofing_type = signal.spoofing_type

        if (
            self.config.allow_pull_and_reversal
            and (
                pattern == SpoofingPattern.PULL_AND_REVERSAL
                or spoofing_type == SpoofingType.ORDER_PULL
            )
        ):
            return True

        if (
            self.config.allow_pressure_bluff
            and (
                pattern == SpoofingPattern.PRESSURE_BLUFF
                or spoofing_type == SpoofingType.FLIP_PRESSURE
            )
        ):
            return True

        if (
            self.config.allow_multi_level_layering
            and (
                pattern == SpoofingPattern.MULTI_LEVEL_LAYERING
                or spoofing_type == SpoofingType.LAYERING
            )
        ):
            return True

        if self.config.allow_composite and spoofing_type == SpoofingType.COMPOSITE:
            return True

        return False

    def accepts_signal(self, signal: SpoofingSignal) -> bool:
        """
        Reversal-specific signal filter.
        """
        if not super().accepts_signal(signal):
            return False

        if signal.score < self.config.min_reversal_score:
            return False

        if signal.confidence < self.config.min_reversal_confidence:
            return False

        features = signal.features
        if features is None:
            return False

        price_reaction_bps = abs(self._feature_float(features, "price_reaction_bps"))
        pull_ratio = self._feature_float(features, "pull_ratio")
        fill_ratio = self._feature_float(features, "fill_ratio")
        is_fast_pull = self._feature_bool(features, "is_fast_pull")
        pressure_flip_strength = self._feature_float(features, "pressure_flip_strength")
        layering_score = self._feature_float(features, "layering_score")

        if price_reaction_bps < self.config.min_price_reaction_bps:
            # Для деяких setup-ів достатньо сильного pull/flip навіть без великої реакції.
            if not (
                pull_ratio >= self.config.min_pull_ratio
                or pressure_flip_strength > 0.0
                or layering_score > 0.0
            ):
                return False

        if fill_ratio > self.config.max_fill_ratio:
            return False

        if pull_ratio > 0 and pull_ratio < self.config.min_pull_ratio:
            if pressure_flip_strength <= 0 and layering_score <= 0:
                return False

        if self.config.require_fast_pull_or_reaction:
            if not is_fast_pull and price_reaction_bps < self.config.min_price_reaction_bps:
                return False

        if self.config.require_reaction_for_pressure_bluff:
            if signal.pattern == SpoofingPattern.PRESSURE_BLUFF:
                if price_reaction_bps < self.config.min_price_reaction_bps:
                    return False

        return True

    # -------------------------------------------------------------------------
    # setup building
    # -------------------------------------------------------------------------

    def build_setup(self, signal: SpoofingSignal) -> SpoofingTradeSetup | None:
        """
        Створює reversal-setup на базі spoofing signal.
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

        setup.metadata["reversal_reason"] = self._build_reversal_reason(signal)
        setup.metadata["reversal_mode"] = "post_spoof_reversal"
        setup.metadata["price_reaction_bps"] = self._feature_float(
            signal.features,
            "price_reaction_bps",
        )
        setup.metadata["pull_ratio"] = self._feature_float(signal.features, "pull_ratio")
        setup.metadata["fill_ratio"] = self._feature_float(signal.features, "fill_ratio")
        setup.metadata["pressure_flip_strength"] = self._feature_float(
            signal.features,
            "pressure_flip_strength",
        )
        setup.metadata["layering_score"] = self._feature_float(
            signal.features,
            "layering_score",
        )

        if self.config.keep_reference_to_source_wall:
            setup.metadata["wall_id"] = signal.wall_id

        return setup

    def enrich_setup(self, setup: SpoofingTradeSetup, signal: SpoofingSignal) -> None:
        """
        Додає reversal-specific metadata.
        """
        features = signal.features

        setup.metadata["signal_side"] = signal.side.value
        setup.metadata["expected_reversal_direction"] = setup.direction.value
        setup.metadata["spoofing_type"] = signal.spoofing_type.value
        setup.metadata["pattern"] = signal.pattern.value
        setup.metadata["severity"] = signal.severity.value

        if features is not None:
            setup.metadata["is_fast_pull"] = self._feature_bool(features, "is_fast_pull")
            setup.metadata["is_fake_liquidity"] = self._feature_bool(
                features,
                "is_fake_liquidity",
            )
            setup.metadata["is_layering"] = self._feature_bool(features, "is_layering")

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
        Entry трохи за reference level у бік continuation.
        """
        offset_ratio = self.config.entry_offset_bps_reversal / 10_000.0

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
        Stop за spoof reference zone.
        """
        buffer_ratio = self.config.stop_buffer_bps_reversal / 10_000.0

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
        TP:
        1. базово через RR;
        2. optional: масштабуємо від фактичної price reaction.
        """
        base_tp = super().compute_take_profit_price(
            signal=signal,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            reference_price=reference_price,
        )

        if not self.config.use_reaction_scaled_take_profit:
            return base_tp

        price_reaction_bps = abs(self._feature_float(signal.features, "price_reaction_bps"))
        if price_reaction_bps <= 0:
            return base_tp

        target_bps = price_reaction_bps * self.config.reaction_tp_multiplier
        target_bps = max(self.config.min_take_profit_bps, target_bps)
        target_bps = min(self.config.max_take_profit_bps, target_bps)

        target_ratio = target_bps / 10_000.0

        if direction == StrategyDirection.LONG:
            reaction_tp = entry_price * (1.0 + target_ratio)
        elif direction == StrategyDirection.SHORT:
            reaction_tp = entry_price * (1.0 - target_ratio)
        else:
            reaction_tp = entry_price

        # Беремо дальшу ціль, щоб не "обрізати" reversal.
        if direction == StrategyDirection.LONG:
            return max(base_tp, reaction_tp)

        if direction == StrategyDirection.SHORT:
            return min(base_tp, reaction_tp)

        return base_tp

    # -------------------------------------------------------------------------
    # signal update handling
    # -------------------------------------------------------------------------

    def apply_signal_update(self, *, setup: SpoofingTradeSetup, signal: SpoofingSignal) -> None:
        """
        Оновлюємо setup сильнішим spoofing update.
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

        setup.metadata["updated_price_reaction_bps"] = self._feature_float(
            signal.features,
            "price_reaction_bps",
        )
        setup.metadata["updated_pull_ratio"] = self._feature_float(
            signal.features,
            "pull_ratio",
        )
        setup.metadata["updated_fill_ratio"] = self._feature_float(
            signal.features,
            "fill_ratio",
        )

    def should_invalidate_from_signal_update(
        self,
        *,
        setup: SpoofingTradeSetup,
        signal: SpoofingSignal,
    ) -> bool:
        """
        Setup invalidate, якщо апдейт ослабив setup нижче мінімально прийнятного рівня.
        """
        if signal.confidence < self.config.invalidate_on_signal_confidence_drop_below:
            return True

        if signal.score < self.config.invalidate_on_signal_score_drop_below:
            return True

        features = signal.features
        if features is None:
            return False

        fill_ratio = self._feature_float(features, "fill_ratio")
        if fill_ratio > max(self.config.max_fill_ratio, 0.45):
            return True

        # Якщо це pressure bluff, але немає реакції ринку — reversal слабкий.
        if signal.pattern == SpoofingPattern.PRESSURE_BLUFF:
            price_reaction_bps = abs(self._feature_float(features, "price_reaction_bps"))
            if self.config.require_reaction_for_pressure_bluff:
                if price_reaction_bps < self.config.min_price_reaction_bps * 0.75:
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
        Reversal-specific confirmation.

        Ідея:
        - ціна має зміститись у напрямку reversal;
        - для HIGH/CRITICAL severity можна вимагати трохи менший рух;
        - optional: ціна має перейти reference_price в правильний бік.
        """
        if setup.status != SetupStatus.PENDING:
            return False

        if current_price <= 0 or setup.reference_price <= 0:
            return False

        required_bps = self.config.confirmation_move_bps
        if setup.severity in {SpoofingSeverity.HIGH, SpoofingSeverity.CRITICAL}:
            required_bps = min(
                required_bps,
                self.config.confirmation_move_bps_high_severity,
            )

        move_bps = self.signed_bps_move(
            current_price=current_price,
            reference_price=setup.reference_price,
        )

        if setup.direction == StrategyDirection.LONG:
            if self.config.require_price_beyond_reference and current_price <= setup.entry_price:
                return False
            passed = move_bps >= required_bps

        elif setup.direction == StrategyDirection.SHORT:
            if self.config.require_price_beyond_reference and current_price >= setup.entry_price:
                return False
            passed = move_bps <= -required_bps

        else:
            passed = False

        if not passed:
            return False

        price_reaction_bps = abs(self._feature_float(signal.features, "price_reaction_bps"))
        pull_ratio = self._feature_float(signal.features, "pull_ratio")
        pressure_flip_strength = self._feature_float(
            signal.features,
            "pressure_flip_strength",
        )

        if (
            price_reaction_bps <= 0
            and pull_ratio < self.config.min_pull_ratio
            and pressure_flip_strength <= 0
        ):
            return False

        setup.status = SetupStatus.CONFIRMED
        setup.confirmed_at = self.now()
        setup.confirmation_price = current_price
        setup.metadata["confirmation_move_bps"] = move_bps
        setup.metadata["confirmation_required_bps"] = required_bps

        self._stats["setups_confirmed"] += 1

        self.log_info(
            "Reversal setup confirmed",
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
        Для reversal-стратегії confirmed setup можна trigger-ити одразу,
        коли ціна не відкотилась назад під/над entry zone.
        """
        if setup.status != SetupStatus.CONFIRMED:
            return False

        if setup.direction == StrategyDirection.LONG:
            return current_price >= setup.entry_price

        if setup.direction == StrategyDirection.SHORT:
            return current_price <= setup.entry_price

        return False

    def should_invalidate_on_price(
        self,
        *,
        setup: SpoofingTradeSetup,
        current_price: float,
    ) -> bool:
        """
        Invalidation якщо ринок рухається проти reversal-гіпотези.
        """
        adverse_bps = self._compute_adverse_move_bps(
            setup=setup,
            current_price=current_price,
        )
        if adverse_bps >= self.config.max_adverse_move_bps_reversal:
            return True

        # Більш жорстка логіка:
        # якщо confirmed setup повернувся через reference zone.
        if setup.status == SetupStatus.CONFIRMED:
            if setup.direction == StrategyDirection.LONG and current_price < setup.reference_price:
                return True

            if setup.direction == StrategyDirection.SHORT and current_price > setup.reference_price:
                return True

        return False

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------

    def _build_reversal_reason(self, signal: SpoofingSignal) -> str:
        """
        Людинозрозумілий опис логіки reversal setup.
        """
        if signal.side == SpoofingSide.ASK:
            base = "ask-side fake pressure removed -> bullish reversal"
        elif signal.side == SpoofingSide.BID:
            base = "bid-side fake support removed -> bearish reversal"
        else:
            base = "spoofing reversal"

        return f"{base}; pattern={signal.pattern.value}; type={signal.spoofing_type.value}"