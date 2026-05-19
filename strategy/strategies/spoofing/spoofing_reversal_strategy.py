from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler

from analytics.spoofing import (
    SpoofingComponent,
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
    Конфіг для reversal-стратегії поверх analytics.spoofing.

    Ідея:
    - торгуємо directional reversal / continuation після зняття spoofing-тиску;
    - використовуємо не тільки merged SpoofingFeatures, а й:
      detector_results, score_breakdown, analytics metadata.
    """

    # ---------------------------------------------------------------------
    # Accepted analytics patterns / types
    # ---------------------------------------------------------------------

    allow_pull_and_reversal: bool = True
    allow_pressure_bluff: bool = True
    allow_multi_level_layering: bool = True
    allow_composite: bool = True

    # ---------------------------------------------------------------------
    # Base reversal filters
    # ---------------------------------------------------------------------

    min_reversal_score: float = 0.68
    min_reversal_confidence: float = 0.58
    min_price_reaction_bps: float = 1.5
    min_pull_ratio: float = 0.55
    max_fill_ratio: float = 0.35

    # Якщо signal прийшов як COMPOSITE, хочемо бачити не менше N detector-ів.
    min_composite_detector_count: int = 2
    min_composite_agreement_ratio: float = 0.50

    # Optional feature-based filters.
    require_fast_pull_or_reaction: bool = False
    require_directional_reaction_alignment: bool = False

    # ---------------------------------------------------------------------
    # ORDER_PULL / PULL_AND_REVERSAL-specific filters
    # ---------------------------------------------------------------------

    require_order_pull_detector_for_pull_pattern: bool = False
    min_order_pull_detector_score: float = 0.0
    min_order_pull_detector_confidence: float = 0.0

    # ---------------------------------------------------------------------
    # FLIP_PRESSURE / PRESSURE_BLUFF-specific filters
    # ---------------------------------------------------------------------

    require_flip_pressure_detector_for_pressure_bluff: bool = False
    require_reaction_for_pressure_bluff: bool = True
    require_has_reversal_for_pressure_bluff: bool = False

    min_pressure_flip_strength: float = 0.0
    min_flip_pressure_detector_score: float = 0.0
    min_flip_pressure_detector_confidence: float = 0.0
    max_distance_from_mid_bps_for_flip: float = 0.0
    # 0.0 = disabled

    # ---------------------------------------------------------------------
    # LAYERING-specific filters
    # ---------------------------------------------------------------------

    require_layering_detector_for_layering_pattern: bool = False
    min_layering_score: float = 0.0
    min_layering_detector_score: float = 0.0
    min_layering_detector_confidence: float = 0.0

    min_layers: int = 0
    min_layer_total_notional: float = 0.0
    min_synchronized_pull_ratio: float = 0.0
    max_layer_price_span_bps: float = 0.0
    # 0.0 = disabled

    # ---------------------------------------------------------------------
    # Analytics score metadata filters
    # ---------------------------------------------------------------------

    min_reversal_detector_count: int = 1
    min_reversal_agreement_ratio: float = 0.0
    min_reversal_average_confidence: float = 0.0
    require_analytics_score_passed: bool = False

    # ---------------------------------------------------------------------
    # Confirmation
    # ---------------------------------------------------------------------

    confirmation_move_bps: float = 1.2
    confirmation_move_bps_high_severity: float = 0.8
    confirmation_move_bps_composite: float = 0.8
    require_price_beyond_reference: bool = True
    require_confirmation_analytics_still_valid: bool = True

    # ---------------------------------------------------------------------
    # Invalidation
    # ---------------------------------------------------------------------

    max_adverse_move_bps_reversal: float = 2.2
    invalidate_on_signal_confidence_drop_below: float = 0.45
    invalidate_on_signal_score_drop_below: float = 0.50
    invalidate_if_pull_ratio_drops_below_factor: float = 0.75
    invalidate_if_reaction_drops_below_factor: float = 0.60

    # ---------------------------------------------------------------------
    # Pricing
    # ---------------------------------------------------------------------

    entry_offset_bps_reversal: float = 0.15
    stop_buffer_bps_reversal: float = 2.8
    take_profit_bps_reversal: float = 7.0

    # Target shaping.
    use_reaction_scaled_take_profit: bool = True
    reaction_tp_multiplier: float = 1.35
    composite_tp_multiplier: float = 1.50
    min_take_profit_bps: float = 4.0
    max_take_profit_bps: float = 18.0

    # Metadata / state.
    keep_reference_to_source_wall: bool = True
    store_detector_metadata: bool = True


class SpoofingReversalStrategy(BaseSpoofingStrategy):
    """
    Strategy: reversal / pressure-release trade після spoofing-сигналів.

    Підтримувані analytics sources:
    - ORDER_PULL / PULL_AND_REVERSAL;
    - FLIP_PRESSURE / PRESSURE_BLUFF;
    - LAYERING / MULTI_LEVEL_LAYERING;
    - COMPOSITE spoofing signals.

    Важливо:
    - detection не робиться тут;
    - strategy працює поверх готового SpoofingSignal;
    - максимально використовує detector_results / score_breakdown / metadata.
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
    # Pattern support
    # -------------------------------------------------------------------------

    def supports_pattern(self, signal: SpoofingSignal) -> bool:
        """
        Перевіряє, чи сигнал релевантний саме для reversal-ідеї.

        Тут враховуємо і high-level pattern/type, і фактичну наявність
        detector_results, якщо сигнал прийшов як COMPOSITE.
        """
        pattern = signal.pattern
        spoofing_type = signal.spoofing_type

        if (
            self.config.allow_pull_and_reversal
            and (
                pattern == SpoofingPattern.PULL_AND_REVERSAL
                or spoofing_type == SpoofingType.ORDER_PULL
                or self.has_detector(signal, SpoofingComponent.ORDER_PULL_DETECTOR)
            )
        ):
            return True

        if (
            self.config.allow_pressure_bluff
            and (
                pattern == SpoofingPattern.PRESSURE_BLUFF
                or spoofing_type == SpoofingType.FLIP_PRESSURE
                or self.has_detector(signal, SpoofingComponent.FLIP_PRESSURE_DETECTOR)
            )
        ):
            return True

        if (
            self.config.allow_multi_level_layering
            and (
                pattern == SpoofingPattern.MULTI_LEVEL_LAYERING
                or spoofing_type == SpoofingType.LAYERING
                or self.has_detector(signal, SpoofingComponent.LAYERING_DETECTOR)
            )
        ):
            return True

        if self.config.allow_composite and spoofing_type == SpoofingType.COMPOSITE:
            return self._has_reversal_relevant_composite(signal)

        return False

    def accepts_signal(self, signal: SpoofingSignal) -> bool:
        """
        Reversal-specific acceptance filter.

        На відміну від старої версії, цей фільтр використовує:
        - SpoofingSignal fields;
        - SpoofingFeatures;
        - detector_results;
        - score_breakdown;
        - analytics metadata.
        """
        if not super().accepts_signal(signal):
            return False

        if signal.score < self.config.min_reversal_score:
            return False

        if signal.confidence < self.config.min_reversal_confidence:
            return False

        if signal.features is None:
            return False

        if not self._passes_reversal_analytics_contract(signal):
            return False

        if not self._passes_common_feature_filters(signal):
            return False

        if self._is_pull_reversal_signal(signal):
            if not self._passes_pull_reversal_filters(signal):
                return False

        if self._is_pressure_bluff_signal(signal):
            if not self._passes_pressure_bluff_filters(signal):
                return False

        if self._is_layering_signal(signal):
            if not self._passes_layering_filters(signal):
                return False

        if signal.spoofing_type == SpoofingType.COMPOSITE:
            if not self._passes_composite_filters(signal):
                return False

        return True

    # -------------------------------------------------------------------------
    # Setup building
    # -------------------------------------------------------------------------

    def build_setup(self, signal: SpoofingSignal) -> SpoofingTradeSetup | None:
        """
        Створює reversal setup на базі актуального analytics.spoofing signal.
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
        setup.metadata["reversal_mode"] = self._resolve_reversal_mode(signal)
        setup.metadata["expected_reversal_direction"] = direction.value

        setup.metadata.update(self._build_reversal_feature_metadata(signal))
        setup.metadata.update(self._build_reversal_analytics_metadata(signal))

        if self.config.store_detector_metadata:
            setup.metadata["detectors"] = self._build_detector_metadata(signal)

        if self.config.keep_reference_to_source_wall:
            setup.metadata["wall_id"] = signal.wall_id

        return setup

    def enrich_setup(self, setup: SpoofingTradeSetup, signal: SpoofingSignal) -> None:
        """
        Додає reversal-specific metadata без втрати analytics scope.
        """
        setup.metadata["signal_side"] = signal.side.value
        setup.metadata["spoofing_type"] = signal.spoofing_type.value
        setup.metadata["pattern"] = signal.pattern.value
        setup.metadata["severity"] = signal.severity.value

        setup.metadata["scope"] = {
            "exchange": setup.exchange,
            "market_type": setup.market_type,
            "symbol": setup.symbol,
            "timeframe": setup.timeframe,
            "exchange_symbol": setup.exchange_symbol,
        }

    # -------------------------------------------------------------------------
    # Pricing
    # -------------------------------------------------------------------------

    def compute_entry_price(
        self,
        signal: SpoofingSignal,
        direction: StrategyDirection,
        reference_price: float,
    ) -> float:
        """
        Entry трохи за reference level у бік очікуваного reversal move.
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
        2. optional reaction scaling;
        3. composite signals можуть отримати трохи ширшу ціль.
        """
        base_tp = super().compute_take_profit_price(
            signal=signal,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            reference_price=reference_price,
        )

        fallback_ratio = self.config.take_profit_bps_reversal / 10_000.0
        if direction == StrategyDirection.LONG:
            fallback_tp = entry_price * (1.0 + fallback_ratio)
            base_tp = max(base_tp, fallback_tp)
        elif direction == StrategyDirection.SHORT:
            fallback_tp = entry_price * (1.0 - fallback_ratio)
            base_tp = min(base_tp, fallback_tp)

        if not self.config.use_reaction_scaled_take_profit:
            return base_tp

        reaction_bps = abs(self._feature_float(signal.features, "price_reaction_bps"))
        if reaction_bps <= 0:
            reaction_bps = abs(
                self.first_detector_metadata_float(
                    signal,
                    names=("price_reaction_bps", "reaction_bps"),
                    default=0.0,
                )
            )

        if reaction_bps <= 0:
            return base_tp

        multiplier = self.config.reaction_tp_multiplier
        if signal.spoofing_type == SpoofingType.COMPOSITE:
            multiplier = max(multiplier, self.config.composite_tp_multiplier)

        target_bps = reaction_bps * multiplier
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
    # Signal update handling
    # -------------------------------------------------------------------------

    def apply_signal_update(self, *, setup: SpoofingTradeSetup, signal: SpoofingSignal) -> None:
        """
        Оновлюємо setup сильнішим / свіжішим spoofing update.
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

        setup.metadata["reversal_reason"] = self._build_reversal_reason(signal)
        setup.metadata["reversal_mode"] = self._resolve_reversal_mode(signal)
        setup.metadata.update(
            {
                f"updated_{key}": value
                for key, value in self._build_reversal_feature_metadata(signal).items()
            }
        )
        setup.metadata["updated_analytics"] = self._build_reversal_analytics_metadata(signal)

        if self.config.store_detector_metadata:
            setup.metadata["updated_detectors"] = self._build_detector_metadata(signal)

    def should_invalidate_from_signal_update(
        self,
        *,
        setup: SpoofingTradeSetup,
        signal: SpoofingSignal,
    ) -> bool:
        """
        Invalidate setup, якщо analytics update ослабив reversal hypothesis.
        """
        if signal.confidence < self.config.invalidate_on_signal_confidence_drop_below:
            return True

        if signal.score < self.config.invalidate_on_signal_score_drop_below:
            return True

        if signal.features is None:
            return False

        fill_ratio = self._feature_float(signal.features, "fill_ratio")
        if fill_ratio > max(self.config.max_fill_ratio, 0.45):
            return True

        pull_ratio = self._feature_float(signal.features, "pull_ratio")
        if pull_ratio > 0:
            threshold = self.config.min_pull_ratio * self.config.invalidate_if_pull_ratio_drops_below_factor
            if pull_ratio < threshold and not self._has_strong_flip_or_layering(signal):
                return True

        reaction_bps = abs(self._feature_float(signal.features, "price_reaction_bps"))
        if reaction_bps > 0:
            threshold = (
                self.config.min_price_reaction_bps
                * self.config.invalidate_if_reaction_drops_below_factor
            )
            if reaction_bps < threshold and self._requires_reaction(signal):
                return True

        if self._is_pressure_bluff_signal(signal):
            if self.config.require_reaction_for_pressure_bluff:
                if reaction_bps < self.config.min_price_reaction_bps * 0.75:
                    return True

        if self.config.require_analytics_score_passed:
            if not self.analytics_bool(signal, "passed", default=True):
                return True

        return False

    # -------------------------------------------------------------------------
    # Confirmation / trigger / invalidation
    # -------------------------------------------------------------------------

    def confirm_setup(
        self,
        *,
        setup: SpoofingTradeSetup,
        current_price: float,
        signal: SpoofingSignal,
    ) -> bool:
        """
        Reversal confirmation.

        Ціна має зміститись у напрямку reversal. Для HIGH/CRITICAL або
        COMPOSITE можна дозволити нижчий confirmation threshold.
        """
        if setup.status != SetupStatus.PENDING:
            return False

        if current_price <= 0 or setup.reference_price <= 0:
            return False

        if self.config.require_confirmation_analytics_still_valid:
            if not self.accepts_signal(signal):
                return False

        required_bps = self._resolve_confirmation_bps(signal=signal, setup=setup)

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

        if not self._has_minimum_confirmation_evidence(signal):
            return False

        setup.status = SetupStatus.CONFIRMED
        setup.confirmed_at = self.now()
        setup.confirmation_price = current_price
        setup.metadata["confirmation_move_bps"] = move_bps
        setup.metadata["confirmation_required_bps"] = required_bps
        setup.metadata["confirmation_analytics"] = self._build_reversal_analytics_metadata(signal)

        self._stats["setups_confirmed"] += 1

        self.log_info(
            "Reversal setup confirmed",
            setup_id=setup.setup_id,
            exchange=setup.exchange,
            market_type=setup.market_type,
            symbol=setup.symbol,
            timeframe=setup.timeframe,
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
        Для reversal-стратегії confirmed setup trigger-иться, якщо ціна
        не відкотилась назад за entry zone.
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
        Invalidation якщо ринок рухається проти reversal hypothesis.
        """
        adverse_bps = self._compute_adverse_move_bps(
            setup=setup,
            current_price=current_price,
        )
        if adverse_bps >= self.config.max_adverse_move_bps_reversal:
            return True

        return False

    # -------------------------------------------------------------------------
    # Signal classification helpers
    # -------------------------------------------------------------------------

    def _is_pull_reversal_signal(self, signal: SpoofingSignal) -> bool:
        return (
            signal.pattern == SpoofingPattern.PULL_AND_REVERSAL
            or signal.spoofing_type == SpoofingType.ORDER_PULL
            or self.has_detector(signal, SpoofingComponent.ORDER_PULL_DETECTOR)
        )

    def _is_pressure_bluff_signal(self, signal: SpoofingSignal) -> bool:
        return (
            signal.pattern == SpoofingPattern.PRESSURE_BLUFF
            or signal.spoofing_type == SpoofingType.FLIP_PRESSURE
            or self.has_detector(signal, SpoofingComponent.FLIP_PRESSURE_DETECTOR)
        )

    def _is_layering_signal(self, signal: SpoofingSignal) -> bool:
        return (
            signal.pattern == SpoofingPattern.MULTI_LEVEL_LAYERING
            or signal.spoofing_type == SpoofingType.LAYERING
            or self.has_detector(signal, SpoofingComponent.LAYERING_DETECTOR)
        )

    def _has_reversal_relevant_composite(self, signal: SpoofingSignal) -> bool:
        relevant_detectors = 0

        if self.has_detector(signal, SpoofingComponent.ORDER_PULL_DETECTOR):
            relevant_detectors += 1

        if self.has_detector(signal, SpoofingComponent.FLIP_PRESSURE_DETECTOR):
            relevant_detectors += 1

        if self.has_detector(signal, SpoofingComponent.LAYERING_DETECTOR):
            relevant_detectors += 1

        if relevant_detectors > 0:
            return True

        pattern = signal.pattern
        return pattern in {
            SpoofingPattern.PULL_AND_REVERSAL,
            SpoofingPattern.PRESSURE_BLUFF,
            SpoofingPattern.MULTI_LEVEL_LAYERING,
        }

    def _resolve_reversal_mode(self, signal: SpoofingSignal) -> str:
        if self._is_pressure_bluff_signal(signal):
            return "pressure_bluff_reversal"

        if self._is_layering_signal(signal):
            return "layering_unwind_reversal"

        if self._is_pull_reversal_signal(signal):
            return "order_pull_reversal"

        if signal.spoofing_type == SpoofingType.COMPOSITE:
            return "composite_spoofing_reversal"

        return "post_spoof_reversal"

    # -------------------------------------------------------------------------
    # Acceptance filter helpers
    # -------------------------------------------------------------------------

    def _passes_reversal_analytics_contract(self, signal: SpoofingSignal) -> bool:
        detector_count = self.analytics_int(
            signal,
            "detector_count",
            default=len(signal.detector_results or []),
        )
        if detector_count < self.config.min_reversal_detector_count:
            return False

        agreement_ratio = self.analytics_float(signal, "agreement_ratio", default=0.0)
        if agreement_ratio < self.config.min_reversal_agreement_ratio:
            return False

        average_confidence = self.analytics_float(signal, "average_confidence", default=0.0)
        if average_confidence < self.config.min_reversal_average_confidence:
            return False

        if self.config.require_analytics_score_passed:
            if not self.analytics_bool(signal, "passed", default=False):
                return False

        return True

    def _passes_common_feature_filters(self, signal: SpoofingSignal) -> bool:
        features = signal.features
        if features is None:
            return False

        reaction_bps = abs(self._feature_float(features, "price_reaction_bps"))
        signed_reaction_bps = self._feature_float(features, "price_reaction_bps")
        pull_ratio = self._feature_float(features, "pull_ratio")
        fill_ratio = self._feature_float(features, "fill_ratio")
        is_fast_pull = self._feature_bool(features, "is_fast_pull")
        pressure_flip_strength = self._feature_float(features, "pressure_flip_strength")
        layering_score = self._feature_float(features, "layering_score")

        if fill_ratio > self.config.max_fill_ratio:
            return False

        if self.config.require_directional_reaction_alignment:
            direction = self.resolve_direction(signal)
            if direction == StrategyDirection.LONG and signed_reaction_bps < 0:
                return False
            if direction == StrategyDirection.SHORT and signed_reaction_bps > 0:
                return False

        if reaction_bps < self.config.min_price_reaction_bps:
            has_alternative_strength = (
                pull_ratio >= self.config.min_pull_ratio
                or pressure_flip_strength >= self.config.min_pressure_flip_strength > 0
                or layering_score >= self.config.min_layering_score > 0
                or self.has_detector(signal, SpoofingComponent.FLIP_PRESSURE_DETECTOR)
                or self.has_detector(signal, SpoofingComponent.LAYERING_DETECTOR)
            )
            if not has_alternative_strength:
                return False

        if pull_ratio > 0 and pull_ratio < self.config.min_pull_ratio:
            if not self._has_strong_flip_or_layering(signal):
                return False

        if self.config.require_fast_pull_or_reaction:
            if not is_fast_pull and reaction_bps < self.config.min_price_reaction_bps:
                return False

        return True

    def _passes_pull_reversal_filters(self, signal: SpoofingSignal) -> bool:
        if self.config.require_order_pull_detector_for_pull_pattern:
            if not self.has_detector(signal, SpoofingComponent.ORDER_PULL_DETECTOR):
                return False

        detector_score = self.detector_score(
            signal,
            SpoofingComponent.ORDER_PULL_DETECTOR,
            default=0.0,
        )
        if detector_score < self.config.min_order_pull_detector_score:
            return False

        detector_confidence = self.detector_confidence(
            signal,
            SpoofingComponent.ORDER_PULL_DETECTOR,
            default=0.0,
        )
        if detector_confidence < self.config.min_order_pull_detector_confidence:
            return False

        pull_ratio = self._feature_float(signal.features, "pull_ratio")
        if pull_ratio > 0 and pull_ratio < self.config.min_pull_ratio:
            return False

        return True

    def _passes_pressure_bluff_filters(self, signal: SpoofingSignal) -> bool:
        if self.config.require_flip_pressure_detector_for_pressure_bluff:
            if not self.has_detector(signal, SpoofingComponent.FLIP_PRESSURE_DETECTOR):
                return False

        detector_score = self.detector_score(
            signal,
            SpoofingComponent.FLIP_PRESSURE_DETECTOR,
            default=0.0,
        )
        if detector_score < self.config.min_flip_pressure_detector_score:
            return False

        detector_confidence = self.detector_confidence(
            signal,
            SpoofingComponent.FLIP_PRESSURE_DETECTOR,
            default=0.0,
        )
        if detector_confidence < self.config.min_flip_pressure_detector_confidence:
            return False

        reaction_bps = abs(self._feature_float(signal.features, "price_reaction_bps"))
        pressure_flip_strength = self._feature_float(signal.features, "pressure_flip_strength")

        if pressure_flip_strength < self.config.min_pressure_flip_strength:
            if self.config.min_pressure_flip_strength > 0:
                return False

        if self.config.require_reaction_for_pressure_bluff:
            if reaction_bps < self.config.min_price_reaction_bps:
                return False

        flip_metadata = self.detector_metadata(
            signal,
            SpoofingComponent.FLIP_PRESSURE_DETECTOR,
        )

        if self.config.require_has_reversal_for_pressure_bluff:
            has_reversal = self._metadata_bool(flip_metadata, "has_reversal", default=False)
            if not has_reversal:
                return False

        if self.config.max_distance_from_mid_bps_for_flip > 0:
            distance = self._feature_float(signal.features, "distance_from_mid_bps")
            if distance <= 0:
                distance = self._metadata_float(flip_metadata, "distance_from_mid_bps", default=0.0)
            if distance > self.config.max_distance_from_mid_bps_for_flip:
                return False

        return True

    def _passes_layering_filters(self, signal: SpoofingSignal) -> bool:
        if self.config.require_layering_detector_for_layering_pattern:
            if not self.has_detector(signal, SpoofingComponent.LAYERING_DETECTOR):
                return False

        detector_score = self.detector_score(
            signal,
            SpoofingComponent.LAYERING_DETECTOR,
            default=0.0,
        )
        if detector_score < self.config.min_layering_detector_score:
            return False

        detector_confidence = self.detector_confidence(
            signal,
            SpoofingComponent.LAYERING_DETECTOR,
            default=0.0,
        )
        if detector_confidence < self.config.min_layering_detector_confidence:
            return False

        layering_score = self._feature_float(signal.features, "layering_score")
        if layering_score < self.config.min_layering_score:
            if self.config.min_layering_score > 0:
                return False

        metadata = self.detector_metadata(signal, SpoofingComponent.LAYERING_DETECTOR)

        if self.config.min_layers > 0:
            layers = int(self._metadata_float(metadata, "layers", default=0.0))
            if layers < self.config.min_layers:
                return False

        if self.config.min_layer_total_notional > 0:
            total_notional = self._first_metadata_float(
                metadata,
                names=("total_notional", "total_layer_notional", "layer_notional"),
                default=0.0,
            )
            if total_notional < self.config.min_layer_total_notional:
                return False

        if self.config.min_synchronized_pull_ratio > 0:
            sync_pull = self._metadata_float(
                metadata,
                "synchronized_pull_ratio",
                default=0.0,
            )
            if sync_pull < self.config.min_synchronized_pull_ratio:
                return False

        if self.config.max_layer_price_span_bps > 0:
            price_span = self._metadata_float(metadata, "price_span_bps", default=0.0)
            if price_span > self.config.max_layer_price_span_bps:
                return False

        return True

    def _passes_composite_filters(self, signal: SpoofingSignal) -> bool:
        detector_count = self.analytics_int(
            signal,
            "detector_count",
            default=len(signal.detector_results or []),
        )
        if detector_count < self.config.min_composite_detector_count:
            return False

        agreement_ratio = self.analytics_float(signal, "agreement_ratio", default=0.0)
        if agreement_ratio < self.config.min_composite_agreement_ratio:
            return False

        return self._has_reversal_relevant_composite(signal)

    def _has_strong_flip_or_layering(self, signal: SpoofingSignal) -> bool:
        pressure_flip_strength = self._feature_float(signal.features, "pressure_flip_strength")
        layering_score = self._feature_float(signal.features, "layering_score")

        if self.config.min_pressure_flip_strength > 0:
            if pressure_flip_strength >= self.config.min_pressure_flip_strength:
                return True
        elif pressure_flip_strength > 0:
            return True

        if self.config.min_layering_score > 0:
            if layering_score >= self.config.min_layering_score:
                return True
        elif layering_score > 0:
            return True

        if self.has_detector(signal, SpoofingComponent.FLIP_PRESSURE_DETECTOR):
            return True

        if self.has_detector(signal, SpoofingComponent.LAYERING_DETECTOR):
            return True

        return False

    def _requires_reaction(self, signal: SpoofingSignal) -> bool:
        if self._is_pressure_bluff_signal(signal):
            return self.config.require_reaction_for_pressure_bluff
        return True

    def _has_minimum_confirmation_evidence(self, signal: SpoofingSignal) -> bool:
        reaction_bps = abs(self._feature_float(signal.features, "price_reaction_bps"))
        pull_ratio = self._feature_float(signal.features, "pull_ratio")
        pressure_flip_strength = self._feature_float(signal.features, "pressure_flip_strength")
        layering_score = self._feature_float(signal.features, "layering_score")

        if reaction_bps >= self.config.min_price_reaction_bps:
            return True

        if pull_ratio >= self.config.min_pull_ratio:
            return True

        if pressure_flip_strength > 0:
            return True

        if layering_score > 0:
            return True

        if self.has_detector(signal, SpoofingComponent.FLIP_PRESSURE_DETECTOR):
            return True

        if self.has_detector(signal, SpoofingComponent.LAYERING_DETECTOR):
            return True

        return False

    def _resolve_confirmation_bps(
        self,
        *,
        signal: SpoofingSignal,
        setup: SpoofingTradeSetup,
    ) -> float:
        required_bps = self.config.confirmation_move_bps

        if setup.severity in {SpoofingSeverity.HIGH, SpoofingSeverity.CRITICAL}:
            required_bps = min(
                required_bps,
                self.config.confirmation_move_bps_high_severity,
            )

        if signal.spoofing_type == SpoofingType.COMPOSITE:
            required_bps = min(
                required_bps,
                self.config.confirmation_move_bps_composite,
            )

        return required_bps

    # -------------------------------------------------------------------------
    # Metadata builders
    # -------------------------------------------------------------------------

    def _build_reversal_feature_metadata(self, signal: SpoofingSignal) -> dict[str, Any]:
        features = signal.features

        return {
            "price_reaction_bps": self._feature_float(features, "price_reaction_bps"),
            "abs_price_reaction_bps": abs(self._feature_float(features, "price_reaction_bps")),
            "pull_ratio": self._feature_float(features, "pull_ratio"),
            "fill_ratio": self._feature_float(features, "fill_ratio"),
            "lifetime_ms": self._feature_float(features, "lifetime_ms"),
            "pressure_flip_strength": self._feature_float(features, "pressure_flip_strength"),
            "layering_score": self._feature_float(features, "layering_score"),
            "distance_from_mid_bps": self._feature_float(features, "distance_from_mid_bps"),
            "wall_size": self._feature_float(features, "wall_size"),
            "wall_size_ratio": self._feature_float(features, "wall_size_ratio"),
            "is_fast_pull": self._feature_bool(features, "is_fast_pull"),
            "is_fake_liquidity": self._feature_bool(features, "is_fake_liquidity"),
            "is_layering": self._feature_bool(features, "is_layering"),
            "is_near_best_quote": self._feature_bool(features, "is_near_best_quote"),
            "repetition_count": self._feature_float(features, "repetition_count"),
        }

    def _build_reversal_analytics_metadata(self, signal: SpoofingSignal) -> dict[str, Any]:
        return {
            "detector_count": self.analytics_int(
                signal,
                "detector_count",
                default=len(signal.detector_results or []),
            ),
            "agreement_ratio": self.analytics_float(signal, "agreement_ratio", default=0.0),
            "average_confidence": self.analytics_float(signal, "average_confidence", default=0.0),
            "threshold": self.analytics_float(signal, "threshold", default=0.0),
            "passed": self.analytics_bool(signal, "passed", default=False),
            "order_pull_detector_score": self.detector_score(
                signal,
                SpoofingComponent.ORDER_PULL_DETECTOR,
                default=0.0,
            ),
            "order_pull_detector_confidence": self.detector_confidence(
                signal,
                SpoofingComponent.ORDER_PULL_DETECTOR,
                default=0.0,
            ),
            "flip_pressure_detector_score": self.detector_score(
                signal,
                SpoofingComponent.FLIP_PRESSURE_DETECTOR,
                default=0.0,
            ),
            "flip_pressure_detector_confidence": self.detector_confidence(
                signal,
                SpoofingComponent.FLIP_PRESSURE_DETECTOR,
                default=0.0,
            ),
            "layering_detector_score": self.detector_score(
                signal,
                SpoofingComponent.LAYERING_DETECTOR,
                default=0.0,
            ),
            "layering_detector_confidence": self.detector_confidence(
                signal,
                SpoofingComponent.LAYERING_DETECTOR,
                default=0.0,
            ),
            "has_order_pull_detector": self.has_detector(
                signal,
                SpoofingComponent.ORDER_PULL_DETECTOR,
            ),
            "has_flip_pressure_detector": self.has_detector(
                signal,
                SpoofingComponent.FLIP_PRESSURE_DETECTOR,
            ),
            "has_layering_detector": self.has_detector(
                signal,
                SpoofingComponent.LAYERING_DETECTOR,
            ),
        }

    def _build_detector_metadata(self, signal: SpoofingSignal) -> dict[str, Any]:
        return {
            "order_pull": self.detector_metadata(
                signal,
                SpoofingComponent.ORDER_PULL_DETECTOR,
            ),
            "flip_pressure": self.detector_metadata(
                signal,
                SpoofingComponent.FLIP_PRESSURE_DETECTOR,
            ),
            "layering": self.detector_metadata(
                signal,
                SpoofingComponent.LAYERING_DETECTOR,
            ),
        }

    def _build_reversal_reason(self, signal: SpoofingSignal) -> str:
        parts: list[str] = []

        if self._is_pull_reversal_signal(signal):
            pull_ratio = self._feature_float(signal.features, "pull_ratio")
            parts.append(f"order_pull:pull_ratio={pull_ratio:.3f}")

        if self._is_pressure_bluff_signal(signal):
            strength = self._feature_float(signal.features, "pressure_flip_strength")
            reaction = self._feature_float(signal.features, "price_reaction_bps")
            parts.append(
                f"pressure_bluff:strength={strength:.3f},reaction_bps={reaction:.2f}"
            )

        if self._is_layering_signal(signal):
            layering_score = self._feature_float(signal.features, "layering_score")
            metadata = self.detector_metadata(signal, SpoofingComponent.LAYERING_DETECTOR)
            layers = int(self._metadata_float(metadata, "layers", default=0.0))
            parts.append(f"layering:score={layering_score:.3f},layers={layers}")

        if signal.spoofing_type == SpoofingType.COMPOSITE:
            detector_count = self.analytics_int(
                signal,
                "detector_count",
                default=len(signal.detector_results or []),
            )
            agreement_ratio = self.analytics_float(signal, "agreement_ratio", default=0.0)
            parts.append(
                f"composite:detectors={detector_count},agreement={agreement_ratio:.3f}"
            )

        if not parts:
            parts.append(
                f"{signal.spoofing_type.value}:{signal.pattern.value}:score={signal.score:.3f}"
            )

        return " | ".join(parts)

    # -------------------------------------------------------------------------
    # Metadata small helpers
    # -------------------------------------------------------------------------

    def _metadata_float(
        self,
        metadata: dict[str, Any] | None,
        name: str,
        *,
        default: float = 0.0,
    ) -> float:
        if not metadata:
            return default
        return self._safe_float(metadata.get(name), default)

    def _metadata_bool(
        self,
        metadata: dict[str, Any] | None,
        name: str,
        *,
        default: bool = False,
    ) -> bool:
        if not metadata:
            return default

        value = metadata.get(name)
        if value is None:
            return default

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}

        return default

    def _first_metadata_float(
        self,
        metadata: dict[str, Any] | None,
        names: tuple[str, ...],
        *,
        default: float = 0.0,
    ) -> float:
        if not metadata:
            return default

        for name in names:
            value = self._safe_float(metadata.get(name), default)
            if value != default:
                return value

        return default