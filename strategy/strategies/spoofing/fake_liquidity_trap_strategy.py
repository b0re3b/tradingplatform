from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler

from analytics.spoofing import (
    SpoofingComponent,
    SpoofingPattern,
    SpoofingSeverity,
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
    Конфіг для fake-liquidity trap strategy.

    Strategy працює поверх готового analytics.spoofing.SpoofingSignal.

    Основна ідея:
    - велика ліквідність виглядала як реальна підтримка/тиск;
    - швидко зникла;
    - майже не була виконана;
    - ринок відреагував у напрямку unwind;
    - strategy забирає continuation після trap/unwind.
    """

    # ---------------------------------------------------------------------
    # Accepted signals
    # ---------------------------------------------------------------------

    allow_fake_liquidity_type: bool = True
    allow_fake_absorption_pattern: bool = True
    allow_composite_if_fake_liquidity_flag: bool = True
    allow_composite_if_fake_liquidity_detector: bool = True

    # ---------------------------------------------------------------------
    # Base trap filters
    # ---------------------------------------------------------------------

    min_trap_score: float = 0.72
    min_trap_confidence: float = 0.62

    min_pull_ratio: float = 0.70
    max_fill_ratio: float = 0.20
    min_price_reaction_bps: float = 2.0
    max_lifetime_ms: float = 4_500.0

    # ---------------------------------------------------------------------
    # Fake-liquidity semantics
    # ---------------------------------------------------------------------

    require_fake_liquidity_flag: bool = False
    require_fake_liquidity_detector: bool = False
    require_short_lived_wall: bool = True
    require_market_reaction: bool = True

    # Якщо price_reaction_bps ще слабкий, допускаємо сигнал лише якщо є
    # достатньо сильний pull / detector evidence.
    allow_fast_pull_without_reaction: bool = True

    # ---------------------------------------------------------------------
    # Detector-specific quality filters
    # ---------------------------------------------------------------------

    min_fake_liquidity_detector_score: float = 0.0
    min_fake_liquidity_detector_confidence: float = 0.0

    require_order_pull_confirmation: bool = False
    min_order_pull_detector_score: float = 0.0
    min_order_pull_detector_confidence: float = 0.0

    # ---------------------------------------------------------------------
    # Deep fake-liquidity feature filters
    # ---------------------------------------------------------------------

    min_cancel_to_fill_ratio: float = 0.0
    min_wall_notional: float = 0.0
    min_pulled_notional: float = 0.0
    max_distance_from_mid_bps: float = 0.0
    # 0.0 = disabled

    prefer_near_best_quote: bool = False
    require_near_best_quote: bool = False

    min_repetition_count: int = 0
    min_repetition_count_for_composite: int = 0

    allowed_wall_states: tuple[str, ...] = (
        "pulled",
        "weakening",
        "expired",
        "filled",
    )

    # ---------------------------------------------------------------------
    # Analytics score metadata filters
    # ---------------------------------------------------------------------

    min_trap_detector_count: int = 1
    min_trap_agreement_ratio: float = 0.0
    min_trap_average_confidence: float = 0.0
    require_analytics_score_passed: bool = False

    # ---------------------------------------------------------------------
    # Confirmation
    # ---------------------------------------------------------------------

    confirmation_move_bps: float = 1.0
    high_severity_confirmation_move_bps: float = 0.7
    composite_confirmation_move_bps: float = 0.7

    require_price_beyond_entry: bool = True
    require_reaction_not_fading: bool = True
    require_confirmation_analytics_still_valid: bool = True
    require_directional_reaction_alignment: bool = False

    # ---------------------------------------------------------------------
    # Trap retest logic
    # ---------------------------------------------------------------------

    allow_retest_entry: bool = True
    max_retest_distance_bps: float = 1.2
    invalidate_if_deep_reentry_bps: float = 1.8

    # ---------------------------------------------------------------------
    # Invalidation
    # ---------------------------------------------------------------------

    max_adverse_move_bps_trap: float = 2.0
    invalidate_on_signal_confidence_drop_below: float = 0.50
    invalidate_on_signal_score_drop_below: float = 0.55

    invalidate_if_pull_ratio_drops_below_factor: float = 0.85
    invalidate_if_reaction_drops_below_factor: float = 0.60
    invalidate_if_fill_ratio_above: float = 0.30

    # ---------------------------------------------------------------------
    # Pricing
    # ---------------------------------------------------------------------

    entry_offset_bps_trap: float = 0.10
    stop_buffer_bps_trap: float = 2.5
    take_profit_bps_trap: float = 8.0

    trap_tp_multiplier: float = 1.50
    composite_tp_multiplier: float = 1.65
    repeated_trap_tp_multiplier: float = 1.75

    min_take_profit_bps: float = 5.0
    max_take_profit_bps: float = 20.0

    # ---------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------

    keep_source_wall_reference: bool = True
    store_detector_metadata: bool = True


class FakeLiquidityTrapStrategy(BaseSpoofingStrategy):
    """
    Strategy: fake-liquidity trap continuation.

    Типова інтерпретація:
    - ASK fake liquidity vanished -> market can rip up -> LONG;
    - BID fake liquidity vanished -> market can flush down -> SHORT.

    Strategy НЕ виконує detection.
    Вона працює поверх готового SpoofingSignal від analytics.spoofing і
    максимально використовує:
    - SpoofingFeatures;
    - detector_results;
    - score_breakdown;
    - analytics metadata;
    - FakeLiquidityDetector metadata.
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
    # Pattern support
    # -------------------------------------------------------------------------

    def supports_pattern(self, signal: SpoofingSignal) -> bool:
        """
        Приймаємо тільки fake-liquidity style signals.
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

        if signal.spoofing_type == SpoofingType.COMPOSITE:
            if (
                self.config.allow_composite_if_fake_liquidity_flag
                and self._feature_bool(signal.features, "is_fake_liquidity")
            ):
                return True

            if (
                self.config.allow_composite_if_fake_liquidity_detector
                and self.has_detector(signal, SpoofingComponent.FAKE_LIQUIDITY_DETECTOR)
            ):
                return True

        if self.has_detector(signal, SpoofingComponent.FAKE_LIQUIDITY_DETECTOR):
            return True

        return False

    def accepts_signal(self, signal: SpoofingSignal) -> bool:
        """
        Trap-specific acceptance filter.

        На відміну від старої версії, тут враховується не тільки merged
        SpoofingFeatures, а й detector-specific metadata.
        """
        if not super().accepts_signal(signal):
            return False

        if signal.score < self.config.min_trap_score:
            return False

        if signal.confidence < self.config.min_trap_confidence:
            return False

        if signal.features is None:
            return False

        if not self._passes_trap_analytics_contract(signal):
            return False

        if not self._passes_fake_liquidity_detector_filters(signal):
            return False

        if not self._passes_order_pull_confirmation_filters(signal):
            return False

        if not self._passes_feature_filters(signal):
            return False

        if not self._passes_detector_metadata_filters(signal):
            return False

        if not self._passes_directional_reaction_filter(signal):
            return False

        return True

    # -------------------------------------------------------------------------
    # Setup building
    # -------------------------------------------------------------------------

    def build_setup(self, signal: SpoofingSignal) -> SpoofingTradeSetup | None:
        """
        Створює trap setup з повним analytics context.
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
        setup.metadata["trap_mode"] = self._resolve_trap_mode(signal)
        setup.metadata["expected_direction"] = direction.value
        setup.metadata["retest_price"] = retest_price
        setup.metadata["allow_retest_entry"] = self.config.allow_retest_entry

        setup.metadata.update(self._build_trap_feature_metadata(signal))
        setup.metadata.update(self._build_trap_analytics_metadata(signal))

        if self.config.store_detector_metadata:
            setup.metadata["detectors"] = self._build_detector_metadata(signal)

        if self.config.keep_source_wall_reference:
            setup.metadata["wall_id"] = signal.wall_id

        return setup

    def enrich_setup(self, setup: SpoofingTradeSetup, signal: SpoofingSignal) -> None:
        """
        Trap-specific metadata.
        """
        setup.metadata["signal_side"] = signal.side.value
        setup.metadata["pattern"] = signal.pattern.value
        setup.metadata["spoofing_type"] = signal.spoofing_type.value
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
        Entry трохи за trap-zone у напрямку unwind continuation.
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
        - fallback trap TP;
        - scaling від price reaction;
        - сильніший scaling для composite / repeated trap.
        """
        base_tp = super().compute_take_profit_price(
            signal=signal,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            reference_price=reference_price,
        )

        fallback_ratio = self.config.take_profit_bps_trap / 10_000.0

        if direction == StrategyDirection.LONG:
            fallback_tp = entry_price * (1.0 + fallback_ratio)
            base_tp = max(base_tp, fallback_tp)

        elif direction == StrategyDirection.SHORT:
            fallback_tp = entry_price * (1.0 - fallback_ratio)
            base_tp = min(base_tp, fallback_tp)

        reaction_bps = abs(self._feature_float(signal.features, "price_reaction_bps"))
        if reaction_bps <= 0:
            reaction_bps = abs(
                self.first_detector_metadata_float(
                    signal,
                    names=("price_reaction_bps", "reaction_bps"),
                    detector=SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
                    default=0.0,
                )
            )

        if reaction_bps <= 0:
            return base_tp

        multiplier = self.config.trap_tp_multiplier

        if signal.spoofing_type == SpoofingType.COMPOSITE:
            multiplier = max(multiplier, self.config.composite_tp_multiplier)

        repetition_count = int(self._feature_float(signal.features, "repetition_count"))
        if (
            self.config.min_repetition_count > 0
            and repetition_count >= self.config.min_repetition_count
        ):
            multiplier = max(multiplier, self.config.repeated_trap_tp_multiplier)

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
    # Signal updates
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

        setup.metadata["trap_reason"] = self._build_trap_reason(signal)
        setup.metadata["trap_mode"] = self._resolve_trap_mode(signal)

        setup.metadata.update(
            {
                f"updated_{key}": value
                for key, value in self._build_trap_feature_metadata(signal).items()
            }
        )
        setup.metadata["updated_analytics"] = self._build_trap_analytics_metadata(signal)

        if self.config.store_detector_metadata:
            setup.metadata["updated_detectors"] = self._build_detector_metadata(signal)

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

        if signal.features is None:
            return False

        pull_ratio = self._feature_float(signal.features, "pull_ratio")
        fill_ratio = self._feature_float(signal.features, "fill_ratio")
        lifetime_ms = self._feature_float(signal.features, "lifetime_ms")
        reaction_bps = abs(self._feature_float(signal.features, "price_reaction_bps"))

        if pull_ratio > 0:
            min_pull = self.config.min_pull_ratio * self.config.invalidate_if_pull_ratio_drops_below_factor
            if pull_ratio < min_pull:
                return True

        if fill_ratio > self.config.invalidate_if_fill_ratio_above:
            return True

        if self.config.require_short_lived_wall:
            if lifetime_ms > 0 and lifetime_ms > self.config.max_lifetime_ms * 1.20:
                return True

        if self.config.require_reaction_not_fading:
            min_reaction = (
                self.config.min_price_reaction_bps
                * self.config.invalidate_if_reaction_drops_below_factor
            )
            if reaction_bps < min_reaction:
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
        Trap-specific confirmation.

        Потрібно:
        - рух у напрямку trap unwind;
        - ціна бажано за entry zone;
        - fake-liquidity semantics досі валідні;
        - detector / feature evidence не розвалились.
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

        if not self._has_minimum_confirmation_evidence(signal):
            return False

        setup.status = SetupStatus.CONFIRMED
        setup.confirmed_at = self.now()
        setup.confirmation_price = current_price
        setup.metadata["confirmation_move_bps"] = move_bps
        setup.metadata["confirmation_required_bps"] = required_bps
        setup.metadata["confirmation_analytics"] = self._build_trap_analytics_metadata(signal)

        self._stats["setups_confirmed"] += 1

        self.log_info(
            "Fake liquidity trap setup confirmed",
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
        Trigger logic:
        - confirmed continuation;
        - або retest entry, якщо увімкнено.
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
            trigger_price = min(setup.entry_price, retest_price or setup.entry_price)
            return current_price >= trigger_price

        if setup.direction == StrategyDirection.SHORT:
            trigger_price = max(setup.entry_price, retest_price or setup.entry_price)
            return current_price <= trigger_price

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
    # Acceptance helpers
    # -------------------------------------------------------------------------

    def _passes_trap_analytics_contract(self, signal: SpoofingSignal) -> bool:
        detector_count = self.analytics_int(
            signal,
            "detector_count",
            default=len(signal.detector_results or []),
        )
        if detector_count < self.config.min_trap_detector_count:
            return False

        agreement_ratio = self.analytics_float(signal, "agreement_ratio", default=0.0)
        if agreement_ratio < self.config.min_trap_agreement_ratio:
            return False

        average_confidence = self.analytics_float(signal, "average_confidence", default=0.0)
        if average_confidence < self.config.min_trap_average_confidence:
            return False

        if self.config.require_analytics_score_passed:
            if not self.analytics_bool(signal, "passed", default=False):
                return False

        if signal.spoofing_type == SpoofingType.COMPOSITE:
            repetition_count = int(self._feature_float(signal.features, "repetition_count"))
            if repetition_count < self.config.min_repetition_count_for_composite:
                if self.config.min_repetition_count_for_composite > 0:
                    return False

        return True

    def _passes_fake_liquidity_detector_filters(self, signal: SpoofingSignal) -> bool:
        has_detector = self.has_detector(signal, SpoofingComponent.FAKE_LIQUIDITY_DETECTOR)

        if self.config.require_fake_liquidity_detector and not has_detector:
            return False

        if has_detector:
            detector_score = self.detector_score(
                signal,
                SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
                default=0.0,
            )
            if detector_score < self.config.min_fake_liquidity_detector_score:
                return False

            detector_confidence = self.detector_confidence(
                signal,
                SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
                default=0.0,
            )
            if detector_confidence < self.config.min_fake_liquidity_detector_confidence:
                return False

        return True

    def _passes_order_pull_confirmation_filters(self, signal: SpoofingSignal) -> bool:
        if not self.config.require_order_pull_confirmation:
            return True

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

        return True

    def _passes_feature_filters(self, signal: SpoofingSignal) -> bool:
        features = signal.features
        if features is None:
            return False

        pull_ratio = self._feature_float(features, "pull_ratio")
        fill_ratio = self._feature_float(features, "fill_ratio")
        reaction_bps = abs(self._feature_float(features, "price_reaction_bps"))
        lifetime_ms = self._feature_float(features, "lifetime_ms")
        cancel_to_fill_ratio = self._feature_float(features, "cancel_to_fill_ratio")
        distance_from_mid_bps = self._feature_float(features, "distance_from_mid_bps")
        is_fake_liquidity = self._feature_bool(features, "is_fake_liquidity")
        is_fast_pull = self._feature_bool(features, "is_fast_pull")
        is_near_best_quote = self._feature_bool(features, "is_near_best_quote")
        repetition_count = int(self._feature_float(features, "repetition_count"))

        if self.config.require_fake_liquidity_flag and not is_fake_liquidity:
            return False

        if pull_ratio < self.config.min_pull_ratio:
            return False

        if fill_ratio > self.config.max_fill_ratio:
            return False

        if self.config.min_cancel_to_fill_ratio > 0:
            if cancel_to_fill_ratio < self.config.min_cancel_to_fill_ratio:
                return False

        if self.config.require_market_reaction:
            if reaction_bps < self.config.min_price_reaction_bps:
                if not self._has_market_reaction(signal):
                    if not (self.config.allow_fast_pull_without_reaction and is_fast_pull):
                        return False

        if self.config.require_short_lived_wall:
            if lifetime_ms > 0 and lifetime_ms > self.config.max_lifetime_ms:
                return False

        if self.config.max_distance_from_mid_bps > 0:
            if distance_from_mid_bps > self.config.max_distance_from_mid_bps:
                return False

        if self.config.require_near_best_quote and not is_near_best_quote:
            return False

        if self.config.min_repetition_count > 0:
            if repetition_count < self.config.min_repetition_count:
                return False

        # Якщо реакція ще слабка, потрібен хоча б швидкий pull або detector evidence.
        if reaction_bps < self.config.min_price_reaction_bps:
            if not is_fast_pull and not self.has_detector(signal, SpoofingComponent.FAKE_LIQUIDITY_DETECTOR):
                return False

        return True

    def _passes_detector_metadata_filters(self, signal: SpoofingSignal) -> bool:
        metadata = self.detector_metadata(signal, SpoofingComponent.FAKE_LIQUIDITY_DETECTOR)

        if not metadata:
            # Якщо detector metadata немає, покладаємося на merged features.
            return True

        if self.config.min_wall_notional > 0:
            wall_notional = self._metadata_float(metadata, "wall_notional", default=0.0)
            if wall_notional < self.config.min_wall_notional:
                return False

        if self.config.min_pulled_notional > 0:
            pulled_notional = self._metadata_float(metadata, "pulled_notional", default=0.0)
            if pulled_notional < self.config.min_pulled_notional:
                return False

        if self.config.max_distance_from_mid_bps > 0:
            distance = self._metadata_float(metadata, "distance_from_mid_bps", default=0.0)
            if distance > self.config.max_distance_from_mid_bps:
                return False

        wall_state = str(metadata.get("wall_state") or "").strip().lower()
        if wall_state and self.config.allowed_wall_states:
            allowed = {item.strip().lower() for item in self.config.allowed_wall_states}
            if wall_state not in allowed:
                return False

        return True

    def _passes_directional_reaction_filter(self, signal: SpoofingSignal) -> bool:
        if not self.config.require_directional_reaction_alignment:
            return True

        reaction_bps = self._feature_float(signal.features, "price_reaction_bps")
        direction = self.resolve_direction(signal)

        if direction == StrategyDirection.LONG and reaction_bps < 0:
            return False

        if direction == StrategyDirection.SHORT and reaction_bps > 0:
            return False

        return True

    # -------------------------------------------------------------------------
    # Confirmation helpers
    # -------------------------------------------------------------------------

    def _resolve_confirmation_bps(
        self,
        *,
        signal: SpoofingSignal,
        setup: SpoofingTradeSetup,
    ) -> float:
        required_bps = self.config.confirmation_move_bps

        if setup.severity in {SpoofingSeverity.HIGH, SpoofingSeverity.CRITICAL}:
            required_bps = min(required_bps, self.config.high_severity_confirmation_move_bps)

        if signal.spoofing_type == SpoofingType.COMPOSITE:
            required_bps = min(required_bps, self.config.composite_confirmation_move_bps)

        return required_bps

    def _has_minimum_confirmation_evidence(self, signal: SpoofingSignal) -> bool:
        pull_ratio = self._feature_float(signal.features, "pull_ratio")
        fill_ratio = self._feature_float(signal.features, "fill_ratio")
        reaction_bps = abs(self._feature_float(signal.features, "price_reaction_bps"))
        is_fast_pull = self._feature_bool(signal.features, "is_fast_pull")

        if pull_ratio < self.config.min_pull_ratio:
            return False

        if fill_ratio > self.config.max_fill_ratio:
            return False

        if self.config.require_market_reaction:
            if reaction_bps < self.config.min_price_reaction_bps:
                if not self._has_market_reaction(signal):
                    if not (self.config.allow_fast_pull_without_reaction and is_fast_pull):
                        return False

        if self.config.require_fake_liquidity_detector:
            if not self.has_detector(signal, SpoofingComponent.FAKE_LIQUIDITY_DETECTOR):
                return False

        return True

    # -------------------------------------------------------------------------
    # Metadata builders
    # -------------------------------------------------------------------------

    def _build_trap_feature_metadata(self, signal: SpoofingSignal) -> dict[str, Any]:
        features = signal.features

        return {
            "pull_ratio": self._feature_float(features, "pull_ratio"),
            "fill_ratio": self._feature_float(features, "fill_ratio"),
            "cancel_to_fill_ratio": self._feature_float(features, "cancel_to_fill_ratio"),
            "price_reaction_bps": self._feature_float(features, "price_reaction_bps"),
            "abs_price_reaction_bps": abs(self._feature_float(features, "price_reaction_bps")),
            "lifetime_ms": self._feature_float(features, "lifetime_ms"),
            "distance_from_mid_bps": self._feature_float(features, "distance_from_mid_bps"),
            "wall_size": self._feature_float(features, "wall_size"),
            "wall_size_ratio": self._feature_float(features, "wall_size_ratio"),
            "repetition_count": int(self._feature_float(features, "repetition_count")),
            "is_fake_liquidity": self._feature_bool(features, "is_fake_liquidity"),
            "is_fast_pull": self._feature_bool(features, "is_fast_pull"),
            "is_near_best_quote": self._feature_bool(features, "is_near_best_quote"),
            "has_market_reaction": self._has_market_reaction(signal),
        }

    def _build_trap_analytics_metadata(self, signal: SpoofingSignal) -> dict[str, Any]:
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
            "fake_liquidity_detector_score": self.detector_score(
                signal,
                SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
                default=0.0,
            ),
            "fake_liquidity_detector_confidence": self.detector_confidence(
                signal,
                SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
                default=0.0,
            ),
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
            "has_fake_liquidity_detector": self.has_detector(
                signal,
                SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
            ),
            "has_order_pull_detector": self.has_detector(
                signal,
                SpoofingComponent.ORDER_PULL_DETECTOR,
            ),
            "has_flip_pressure_detector": self.has_detector(
                signal,
                SpoofingComponent.FLIP_PRESSURE_DETECTOR,
            ),
        }

    def _build_detector_metadata(self, signal: SpoofingSignal) -> dict[str, Any]:
        return {
            "fake_liquidity": self.detector_metadata(
                signal,
                SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
            ),
            "order_pull": self.detector_metadata(
                signal,
                SpoofingComponent.ORDER_PULL_DETECTOR,
            ),
            "flip_pressure": self.detector_metadata(
                signal,
                SpoofingComponent.FLIP_PRESSURE_DETECTOR,
            ),
        }

    def _build_trap_reason(self, signal: SpoofingSignal) -> str:
        features = signal.features
        metadata = self.detector_metadata(signal, SpoofingComponent.FAKE_LIQUIDITY_DETECTOR)

        pull_ratio = self._feature_float(features, "pull_ratio")
        fill_ratio = self._feature_float(features, "fill_ratio")
        reaction_bps = self._feature_float(features, "price_reaction_bps")
        lifetime_ms = self._feature_float(features, "lifetime_ms")
        cancel_to_fill_ratio = self._feature_float(features, "cancel_to_fill_ratio")
        repetition_count = int(self._feature_float(features, "repetition_count"))

        wall_notional = self._metadata_float(metadata, "wall_notional", default=0.0)
        pulled_notional = self._metadata_float(metadata, "pulled_notional", default=0.0)
        wall_state = str(metadata.get("wall_state") or "")

        parts = [
            "fake_liquidity_trap",
            f"pull_ratio={pull_ratio:.3f}",
            f"fill_ratio={fill_ratio:.3f}",
            f"cancel_to_fill_ratio={cancel_to_fill_ratio:.3f}",
            f"reaction_bps={reaction_bps:.2f}",
            f"lifetime_ms={lifetime_ms:.0f}",
        ]

        if wall_notional > 0:
            parts.append(f"wall_notional={wall_notional:.2f}")

        if pulled_notional > 0:
            parts.append(f"pulled_notional={pulled_notional:.2f}")

        if repetition_count > 0:
            parts.append(f"repetition_count={repetition_count}")

        if wall_state:
            parts.append(f"wall_state={wall_state}")

        if signal.spoofing_type == SpoofingType.COMPOSITE:
            detector_count = self.analytics_int(
                signal,
                "detector_count",
                default=len(signal.detector_results or []),
            )
            agreement_ratio = self.analytics_float(signal, "agreement_ratio", default=0.0)
            parts.append(f"composite_detectors={detector_count}")
            parts.append(f"agreement={agreement_ratio:.3f}")

        return ", ".join(parts)

    def _resolve_trap_mode(self, signal: SpoofingSignal) -> str:
        if signal.spoofing_type == SpoofingType.COMPOSITE:
            if self.has_detector(signal, SpoofingComponent.FLIP_PRESSURE_DETECTOR):
                return "composite_fake_liquidity_pressure_unwind"
            return "composite_fake_liquidity_unwind"

        if self.has_detector(signal, SpoofingComponent.FAKE_LIQUIDITY_DETECTOR):
            return "fake_liquidity_detector_unwind"

        if signal.pattern == SpoofingPattern.FAKE_ABSORPTION:
            return "fake_absorption_unwind"

        return "fake_liquidity_unwind"

    # -------------------------------------------------------------------------
    # Retest / market reaction helpers
    # -------------------------------------------------------------------------

    def _compute_retest_price(
        self,
        *,
        direction: StrategyDirection,
        reference_price: float,
        entry_price: float,
    ) -> float:
        """
        Retest price біля entry/reference zone.

        Для LONG дозволяємо легкий retest нижче entry.
        Для SHORT дозволяємо легкий retest вище entry.
        """
        retest_ratio = self.config.max_retest_distance_bps / 10_000.0

        if direction == StrategyDirection.LONG:
            return max(reference_price, entry_price * (1.0 - retest_ratio))

        if direction == StrategyDirection.SHORT:
            return min(reference_price, entry_price * (1.0 + retest_ratio))

        return reference_price

    def _has_market_reaction(self, signal: SpoofingSignal) -> bool:
        """
        has_market_reaction може бути в detector metadata, але не обов'язково
        як top-level SpoofingFeatures field. Тому перевіряємо:
        1. features.metadata/top-level через _feature_bool;
        2. FakeLiquidityDetector metadata;
        3. price_reaction_bps threshold.
        """
        if self._feature_bool(signal.features, "has_market_reaction"):
            return True

        metadata = self.detector_metadata(signal, SpoofingComponent.FAKE_LIQUIDITY_DETECTOR)
        if self._metadata_bool(metadata, "has_market_reaction", default=False):
            return True

        reaction_bps = abs(self._feature_float(signal.features, "price_reaction_bps"))
        return reaction_bps >= self.config.min_price_reaction_bps

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