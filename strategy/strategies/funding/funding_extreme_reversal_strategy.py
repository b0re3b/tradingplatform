from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analytics.funding.enums import (
    FundingBias,
    FundingDivergenceType,
    FundingExtremeType,
    FundingFlipType,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingSignalType,
)
from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from .base import (
    BaseFundingStrategy,
    BaseFundingStrategyConfig,
    FundingSetupStatus,
    FundingStrategyDirection,
    FundingStrategyState,
)


_PRESSURE_NEUTRALIZATION_THRESHOLD_RATIO: float = 0.70
_SIGNAL_CONFIRMATION_SCORE_WEIGHT: float = 0.35
_DIVERGENCE_CONFIRMATION_SCORE_WEIGHT: float = 0.30
_CONTEXT_CONFIRMATION_SCORE_WEIGHT: float = 0.35


@dataclass(slots=True)
class FundingExtremeReversalStrategyConfig(BaseFundingStrategyConfig):
    """
    Funding extreme reversal strategy config.

    Strategy idea:
    - identify overcrowded funding extremes;
    - require regime + pressure confluence;
    - build contrarian reversal setup;
    - confirm via flip, pressure release, divergence, or normalized funding signal;
    - invalidate when analytics context contradicts the setup.

    Scope:
        exchange + market_type + symbol + timeframe
    """

    strategy_namespace: str = "strategy.funding.extreme_reversal"
    source_name: str = "funding_extreme_reversal_strategy"
    service_name: str = "funding_extreme_reversal_strategy"

    enable_funding_updated_subscription: bool = True
    enable_funding_signal_subscription: bool = True
    funding_updated_event_name: str = "analytics.funding.updated"
    funding_signal_event_name: str = "analytics.funding.signal"

    regime_event_name: str = "analytics.funding.regime"
    pressure_event_name: str = "analytics.funding.pressure"
    extreme_event_name: str = "analytics.funding.extreme"
    flip_event_name: str = "analytics.funding.flip"
    divergence_event_name: str = "analytics.funding.divergence"

    min_extreme_severity: float = 0.60
    min_pressure_score: float = 0.55
    min_regime_confidence: float = 0.15
    min_mean_reversion_probability: float = 0.50
    min_squeeze_probability: float = 0.50
    min_divergence_confidence: float = 0.45
    min_signal_confidence: float = 0.45
    min_signal_abs_score: float = 0.35

    require_reversal_risk: bool = True
    require_squeeze_risk_or_reversion_probability: bool = True
    require_high_pressure_level: bool = True

    allow_flip_confirmation: bool = True
    allow_pressure_release_confirmation: bool = True
    allow_divergence_confirmation: bool = True
    allow_signal_confirmation: bool = True
    allow_updated_context_setup: bool = True

    confirm_on_pressure_drop_levels: int = 1
    pressure_release_min_score_drop: float = 0.10

    invalidate_on_opposite_flip: bool = True
    invalidate_on_opposite_divergence: bool = True
    invalidate_on_opposite_signal: bool = True
    invalidate_on_pressure_neutralization: bool = True
    invalidate_on_regime_neutral: bool = False
    invalidate_on_regime_conflict: bool = True
    invalidate_on_extreme_no_longer_reversal_risk: bool = True

    bearish_setup_type: str = "extreme_positive_reversal"
    bullish_setup_type: str = "extreme_negative_reversal"

    tag_extreme: str = "funding_extreme"
    tag_reversal: str = "reversal"
    tag_crowding: str = "crowding"
    tag_squeeze: str = "squeeze_risk"
    tag_divergence: str = "funding_divergence"
    tag_signal: str = "funding_signal"
    tag_global_extreme: str = "global_extreme"
    tag_percentile_extreme: str = "percentile_extreme"
    tag_zscore_extreme: str = "zscore_extreme"
    tag_local_extreme: str = "local_extreme"

    tag_confirmed_by_flip: str = "confirmed_by_flip"
    tag_confirmed_by_release: str = "confirmed_by_pressure_release"
    tag_confirmed_by_divergence: str = "confirmed_by_divergence"
    tag_confirmed_by_signal: str = "confirmed_by_funding_signal"
    tag_atomic_context: str = "atomic_funding_context"

    global_extreme_bonus: float = 0.12
    percentile_extreme_bonus: float = 0.09
    zscore_extreme_bonus: float = 0.08
    local_extreme_bonus: float = 0.04
    liquidation_divergence_bonus: float = 0.10
    cvd_divergence_bonus: float = 0.08
    oi_divergence_bonus: float = 0.06
    price_divergence_bonus: float = 0.04

    preferred_signal_origins_for_confirmation: tuple[str, ...] = (
        "extreme_reversion",
        "pressure_reversion",
        "divergence",
        "flip",
        "regime",
    )
    preferred_signal_origins_for_invalidation: tuple[str, ...] = (
        "extreme_squeeze",
        "pressure",
        "divergence",
        "flip",
        "regime",
    )

    signal_origin_confirmation_weight: dict[str, float] = field(
        default_factory=lambda: {
            "extreme_reversion": 1.00,
            "pressure_reversion": 0.95,
            "divergence": 0.90,
            "flip": 0.85,
            "regime": 0.65,
            "extreme": 0.55,
            "pressure": 0.45,
            "extreme_squeeze": 0.35,
        }
    )
    signal_origin_alignment_weight: dict[str, float] = field(
        default_factory=lambda: {
            "extreme_reversion": 1.00,
            "pressure_reversion": 0.95,
            "divergence": 0.90,
            "flip": 0.80,
            "regime": 0.60,
            "extreme": 0.50,
            "pressure": 0.45,
            "extreme_squeeze": 0.30,
        }
    )

    def validate(self) -> None:
        BaseFundingStrategyConfig.validate(self)

        bounded_fields = {
            "min_extreme_severity": self.min_extreme_severity,
            "min_pressure_score": self.min_pressure_score,
            "min_regime_confidence": self.min_regime_confidence,
            "min_mean_reversion_probability": self.min_mean_reversion_probability,
            "min_squeeze_probability": self.min_squeeze_probability,
            "min_divergence_confidence": self.min_divergence_confidence,
            "min_signal_confidence": self.min_signal_confidence,
            "min_signal_abs_score": self.min_signal_abs_score,
            "pressure_release_min_score_drop": self.pressure_release_min_score_drop,
            "global_extreme_bonus": self.global_extreme_bonus,
            "percentile_extreme_bonus": self.percentile_extreme_bonus,
            "zscore_extreme_bonus": self.zscore_extreme_bonus,
            "local_extreme_bonus": self.local_extreme_bonus,
            "liquidation_divergence_bonus": self.liquidation_divergence_bonus,
            "cvd_divergence_bonus": self.cvd_divergence_bonus,
            "oi_divergence_bonus": self.oi_divergence_bonus,
            "price_divergence_bonus": self.price_divergence_bonus,
        }

        for field_name, value in bounded_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")

        if self.confirm_on_pressure_drop_levels < 0:
            raise ValueError("confirm_on_pressure_drop_levels must be >= 0")

        for mapping_name in (
            "signal_origin_confirmation_weight",
            "signal_origin_alignment_weight",
        ):
            mapping = getattr(self, mapping_name)
            for key, value in mapping.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(f"{mapping_name} keys must be non-empty strings")
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError(f"{mapping_name}[{key!r}] must be between 0 and 1")


class FundingExtremeReversalStrategy(BaseFundingStrategy):
    """
    Contrarian reversal strategy for funding extremes.

    Event flow:
    - regime/pressure/extreme/flip/divergence update local scoped strategy context;
    - extreme or atomic funding.updated may create a contrarian setup;
    - flip, pressure release, aligned divergence, or funding.signal may confirm;
    - opposite flip/divergence/signal or context breakdown invalidates.

    Full scope:
        exchange + market_type + symbol + timeframe
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: FundingExtremeReversalStrategyConfig | None = None,
        scheduler: Scheduler | None = None,
        service_name: str | None = None,
        parquet_storage: Any | None = None,
    ) -> None:
        resolved_config = config or FundingExtremeReversalStrategyConfig()
        super().__init__(
            event_bus=event_bus,
            config=resolved_config,
            scheduler=scheduler,
            service_name=service_name or resolved_config.service_name,
            parquet_storage=parquet_storage,
        )
        self.config: FundingExtremeReversalStrategyConfig = resolved_config

    @property
    def strategy_name(self) -> str:
        return "funding_extreme_reversal"

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_subscriptions(self) -> None:
        self.subscribe(
            self.config.regime_event_name,
            self.on_regime,
            name=f"{self.strategy_name}.on_regime",
        )
        self.subscribe(
            self.config.pressure_event_name,
            self.on_pressure,
            name=f"{self.strategy_name}.on_pressure",
        )
        self.subscribe(
            self.config.extreme_event_name,
            self.on_extreme,
            name=f"{self.strategy_name}.on_extreme",
        )
        self.subscribe(
            self.config.flip_event_name,
            self.on_flip,
            name=f"{self.strategy_name}.on_flip",
        )
        self.subscribe(
            self.config.divergence_event_name,
            self.on_divergence,
            name=f"{self.strategy_name}.on_divergence",
        )

    # ------------------------------------------------------------------
    # Base hooks: analytics.funding.updated / analytics.funding.signal
    # ------------------------------------------------------------------

    async def on_after_funding_updated(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
        event: Event,
    ) -> None:
        if not self.config.allow_updated_context_setup:
            return

        if self.is_in_cooldown(state):
            return

        if state.is_active():
            await self._evaluate_active_state_after_atomic_update(
                state=state,
                correlation_id=event.correlation_id,
            )
            return

        extreme_event = state.last_extreme
        if extreme_event is None:
            return

        event_time = self._extract_event_time_from_normalized(extreme_event)
        if self.is_stale_event(event_time):
            return

        setup_candidate = self._build_setup_from_extreme(
            state=state,
            extreme_event=extreme_event,
        )
        if setup_candidate is None:
            return

        self._apply_setup_candidate(
            state=state,
            setup_candidate=setup_candidate,
            event_time=event_time,
        )
        await self.emit_setup(
            state,
            extra_payload={
                "trigger": "funding_updated",
                "correlation_id": event.correlation_id,
            },
        )

    async def on_after_funding_signal(
        self,
        state: FundingStrategyState,
        signal: Any,
        event: Event,
    ) -> None:
        if self.is_in_cooldown(state) or not state.is_active():
            return

        if self._should_invalidate_by_signal(state, signal):
            self.set_invalidated(
                state,
                reason="opposite_funding_signal_invalidated_reversal_setup",
                cooldown=True,
                metadata={
                    "invalidation_source": "funding_signal",
                    "signal_origin": self._signal_origin(signal),
                    "scope": state.scope.to_dict(),
                },
            )
            await self.emit_invalidated(
                state,
                extra_payload={
                    "trigger": "funding_signal",
                    "correlation_id": event.correlation_id,
                },
            )
            return

        if self._can_confirm_by_signal(state, signal):
            self.set_confirmed(
                state,
                score=self._compute_confirmation_score_from_signal(state, signal),
                confidence=self._compute_confirmation_confidence_from_signal(state, signal),
                reason="funding_signal_confirmed_reversal_setup",
                tags=[self.config.tag_confirmed_by_signal, self.config.tag_signal],
                event_time=self._extract_event_time_from_normalized(signal),
                metadata={
                    "confirmation_source": "funding_signal",
                    "signal_origin": self._signal_origin(signal),
                    "scope": state.scope.to_dict(),
                },
            )
            await self.emit_confirmed(
                state,
                extra_payload={
                    "trigger": "funding_signal",
                    "correlation_id": event.correlation_id,
                },
            )

    # ------------------------------------------------------------------
    # Scoped event handlers
    # ------------------------------------------------------------------

    async def on_regime(self, event: Event) -> None:
        payload = self.extract_payload(event)
        scope = self.extract_funding_scope(payload)
        if scope is None:
            return

        lock = await self.acquire_scope_lock(scope)
        if lock is None:
            return

        try:
            state = self.get_state_for_scope(scope)
            regime_state = self._normalize_regime_payload(payload)

            self.attach_regime(state, regime_state)
            self._expire_state_if_needed(state)

            if self.is_in_cooldown(state) or not state.is_active():
                return

            if self._should_invalidate_by_regime(state, regime_state):
                self.set_invalidated(
                    state,
                    reason="regime_context_invalidated_reversal_setup",
                    cooldown=True,
                    metadata={
                        "invalidation_source": "regime",
                        "scope": state.scope.to_dict(),
                    },
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={
                        "trigger": "regime",
                        "correlation_id": event.correlation_id,
                    },
                )

        except Exception:
            self.logger.exception(
                "Failed to process regime event | strategy=%s scope=%s",
                self.strategy_name,
                scope.to_dict(),
            )
        finally:
            self.release_symbol_lock(lock)

    async def on_pressure(self, event: Event) -> None:
        payload = self.extract_payload(event)
        scope = self.extract_funding_scope(payload)
        if scope is None:
            return

        lock = await self.acquire_scope_lock(scope)
        if lock is None:
            return

        try:
            state = self.get_state_for_scope(scope)
            previous_pressure = state.last_pressure
            pressure_state = self._normalize_pressure_payload(payload)

            self.attach_pressure(state, pressure_state)
            self._expire_state_if_needed(state)

            if self.is_in_cooldown(state) or not state.is_active():
                return

            if self._should_invalidate_by_pressure(state, pressure_state):
                self.set_invalidated(
                    state,
                    reason="pressure_context_invalidated_reversal_setup",
                    cooldown=True,
                    metadata={
                        "invalidation_source": "pressure",
                        "scope": state.scope.to_dict(),
                    },
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={
                        "trigger": "pressure",
                        "correlation_id": event.correlation_id,
                    },
                )
                return

            if self._can_confirm_by_pressure_release(
                state=state,
                previous_pressure=previous_pressure,
                current_pressure=pressure_state,
            ):
                self.set_confirmed(
                    state,
                    score=self._compute_confirmation_score_from_release(
                        state=state,
                        current_pressure=pressure_state,
                    ),
                    confidence=self._compute_confirmation_confidence_from_release(
                        state=state,
                        current_pressure=pressure_state,
                    ),
                    reason="pressure_release_confirmed_reversal_setup",
                    tags=[self.config.tag_confirmed_by_release],
                    event_time=self._extract_event_time_from_normalized(pressure_state),
                    metadata={
                        "confirmation_source": "pressure_release",
                        "scope": state.scope.to_dict(),
                    },
                )
                await self.emit_confirmed(
                    state,
                    extra_payload={
                        "trigger": "pressure_release",
                        "correlation_id": event.correlation_id,
                    },
                )

        except Exception:
            self.logger.exception(
                "Failed to process pressure event | strategy=%s scope=%s",
                self.strategy_name,
                scope.to_dict(),
            )
        finally:
            self.release_symbol_lock(lock)

    async def on_extreme(self, event: Event) -> None:
        payload = self.extract_payload(event)
        scope = self.extract_funding_scope(payload)
        if scope is None:
            return

        lock = await self.acquire_scope_lock(scope)
        if lock is None:
            return

        try:
            state = self.get_state_for_scope(scope)
            extreme_event = self._normalize_extreme_payload(payload)

            self.attach_extreme(state, extreme_event)
            self._expire_state_if_needed(state)

            if self.is_in_cooldown(state):
                return

            event_time = self._extract_event_time_from_normalized(extreme_event)
            if self.is_stale_event(event_time):
                self.logger.debug(
                    "Stale funding extreme event ignored | strategy=%s scope=%s",
                    self.strategy_name,
                    state.scope.to_dict(),
                )
                return

            if state.is_active():
                if self._should_invalidate_by_extreme_context(state, extreme_event):
                    self.set_invalidated(
                        state,
                        reason="extreme_context_invalidated_reversal_setup",
                        cooldown=True,
                        metadata={
                            "invalidation_source": "extreme",
                            "scope": state.scope.to_dict(),
                        },
                    )
                    await self.emit_invalidated(
                        state,
                        extra_payload={
                            "trigger": "extreme",
                            "correlation_id": event.correlation_id,
                        },
                    )
                return

            setup_candidate = self._build_setup_from_extreme(
                state=state,
                extreme_event=extreme_event,
            )
            if setup_candidate is None:
                return

            self._apply_setup_candidate(
                state=state,
                setup_candidate=setup_candidate,
                event_time=event_time,
            )
            await self.emit_setup(
                state,
                extra_payload={
                    "trigger": "extreme",
                    "correlation_id": event.correlation_id,
                },
            )

        except Exception:
            self.logger.exception(
                "Failed to process extreme event | strategy=%s scope=%s",
                self.strategy_name,
                scope.to_dict(),
            )
        finally:
            self.release_symbol_lock(lock)

    async def on_flip(self, event: Event) -> None:
        payload = self.extract_payload(event)
        scope = self.extract_funding_scope(payload)
        if scope is None:
            return

        lock = await self.acquire_scope_lock(scope)
        if lock is None:
            return

        try:
            state = self.get_state_for_scope(scope)
            flip_event = self._normalize_flip_payload(payload)

            self.attach_flip(state, flip_event)
            self._expire_state_if_needed(state)

            if self.is_in_cooldown(state) or not state.is_active():
                return

            if self._should_invalidate_by_flip(state, flip_event):
                self.set_invalidated(
                    state,
                    reason="opposite_flip_invalidated_reversal_setup",
                    cooldown=True,
                    metadata={
                        "invalidation_source": "flip",
                        "scope": state.scope.to_dict(),
                    },
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={
                        "trigger": "flip",
                        "correlation_id": event.correlation_id,
                    },
                )
                return

            if self._can_confirm_by_flip(state, flip_event):
                self.set_confirmed(
                    state,
                    score=self._compute_confirmation_score_from_flip(
                        state=state,
                        flip_event=flip_event,
                    ),
                    confidence=self._compute_confirmation_confidence_from_flip(
                        state=state,
                        flip_event=flip_event,
                    ),
                    reason="flip_confirmed_reversal_setup",
                    tags=[self.config.tag_confirmed_by_flip],
                    event_time=self._extract_event_time_from_normalized(flip_event),
                    metadata={
                        "confirmation_source": "flip",
                        "scope": state.scope.to_dict(),
                    },
                )
                await self.emit_confirmed(
                    state,
                    extra_payload={
                        "trigger": "flip",
                        "correlation_id": event.correlation_id,
                    },
                )

        except Exception:
            self.logger.exception(
                "Failed to process flip event | strategy=%s scope=%s",
                self.strategy_name,
                scope.to_dict(),
            )
        finally:
            self.release_symbol_lock(lock)

    async def on_divergence(self, event: Event) -> None:
        payload = self.extract_payload(event)
        scope = self.extract_funding_scope(payload)
        if scope is None:
            return

        lock = await self.acquire_scope_lock(scope)
        if lock is None:
            return

        try:
            state = self.get_state_for_scope(scope)
            divergence_event = self._normalize_divergence_payload(payload)

            self.attach_divergence(state, divergence_event)
            self._expire_state_if_needed(state)

            if self.is_in_cooldown(state) or not state.is_active():
                return

            event_time = self._extract_event_time_from_normalized(divergence_event)
            if self.is_stale_event(event_time):
                self.logger.debug(
                    "Stale funding divergence event ignored | strategy=%s scope=%s",
                    self.strategy_name,
                    state.scope.to_dict(),
                )
                return

            if self._should_invalidate_by_divergence(state, divergence_event):
                self.set_invalidated(
                    state,
                    reason="opposite_divergence_invalidated_reversal_setup",
                    cooldown=True,
                    metadata={
                        "invalidation_source": "divergence",
                        "scope": state.scope.to_dict(),
                    },
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={
                        "trigger": "divergence",
                        "correlation_id": event.correlation_id,
                    },
                )
                return

            if self._can_confirm_by_divergence(state, divergence_event):
                self.set_confirmed(
                    state,
                    score=self._compute_confirmation_score_from_divergence(
                        state=state,
                        divergence_event=divergence_event,
                    ),
                    confidence=self._compute_confirmation_confidence_from_divergence(
                        state=state,
                        divergence_event=divergence_event,
                    ),
                    reason="divergence_confirmed_reversal_setup",
                    tags=[
                        self.config.tag_confirmed_by_divergence,
                        self.config.tag_divergence,
                    ],
                    event_time=event_time,
                    metadata={
                        "confirmation_source": "divergence",
                        "scope": state.scope.to_dict(),
                    },
                )
                await self.emit_confirmed(
                    state,
                    extra_payload={
                        "trigger": "divergence",
                        "correlation_id": event.correlation_id,
                    },
                )

        except Exception:
            self.logger.exception(
                "Failed to process divergence event | strategy=%s scope=%s",
                self.strategy_name,
                scope.to_dict(),
            )
        finally:
            self.release_symbol_lock(lock)

    # ------------------------------------------------------------------
    # Setup creation
    # ------------------------------------------------------------------

    def _build_setup_from_extreme(
        self,
        state: FundingStrategyState,
        extreme_event: Any,
    ) -> dict[str, Any] | None:
        regime = state.last_regime
        pressure = state.last_pressure

        if regime is None or pressure is None:
            return None

        severity = self._to_float(
            self._get_value(extreme_event, "severity"),
            default=0.0,
        )
        if severity < self.config.min_extreme_severity:
            return None

        if self.config.require_reversal_risk and not bool(
            self._get_value(extreme_event, "is_reversal_risk", False)
        ):
            return None

        regime_confidence = self._to_float(
            self._get_value(regime, "confidence"),
            default=0.0,
        )
        if regime_confidence < self.config.min_regime_confidence:
            return None

        pressure_score = self._to_float(
            self._get_value(pressure, "pressure_score"),
            default=0.0,
        )
        if pressure_score < self.config.min_pressure_score:
            return None

        pressure_direction = self._enum_str(self._get_value(pressure, "direction"))
        pressure_level = self._enum_str(self._get_value(pressure, "level"))

        if self.config.require_high_pressure_level and pressure_level not in {
            FundingPressureLevel.HIGH.value,
            FundingPressureLevel.EXTREME.value,
        }:
            return None

        squeeze_probability = self._to_float(
            self._get_value(pressure, "squeeze_probability"),
            default=0.0,
        )
        mean_reversion_probability = self._to_float(
            self._get_value(pressure, "mean_reversion_probability"),
            default=0.0,
        )

        is_squeeze_risk = bool(self._get_value(extreme_event, "is_squeeze_risk", False))

        if self.config.require_squeeze_risk_or_reversion_probability:
            if (
                not is_squeeze_risk
                and squeeze_probability < self.config.min_squeeze_probability
                and mean_reversion_probability < self.config.min_mean_reversion_probability
            ):
                return None

        extreme_type = self._enum_str(self._get_value(extreme_event, "extreme_type"))
        bias = self._enum_str(self._get_value(regime, "bias"))
        regime_name = self._enum_str(self._get_value(regime, "regime"))

        divergence_alignment_bonus = self._calc_divergence_alignment_bonus(state)
        signal_alignment_bonus = self._calc_signal_alignment_bonus(state)
        extreme_type_bonus = self._extreme_type_bonus(extreme_type)

        adjusted_severity = self._clip_score(severity + extreme_type_bonus)

        if self._is_positive_extreme(extreme_type):
            if pressure_direction != FundingPressureDirection.LONG.value:
                return None
            if bias not in {
                FundingBias.LONG_BIAS.value,
                FundingBias.OVERCROWDED_LONGS.value,
                FundingBias.SQUEEZE_RISK_LONGS.value,
            }:
                return None

            score = self._compute_setup_score(
                severity=adjusted_severity,
                pressure_score=pressure_score,
                squeeze_probability=squeeze_probability,
                mean_reversion_probability=mean_reversion_probability,
                regime_confidence=regime_confidence,
                directional_alignment_bonus=1.0,
                divergence_alignment_bonus=divergence_alignment_bonus,
                signal_alignment_bonus=signal_alignment_bonus,
            )
            confidence = self._compute_setup_confidence(
                score=score,
                severity=adjusted_severity,
                regime_confidence=regime_confidence,
            )

            return {
                "direction": FundingStrategyDirection.SHORT,
                "setup_type": self.config.bearish_setup_type,
                "score": score,
                "confidence": confidence,
                "reason": "positive_funding_extreme_with_crowded_longs",
                "reasons": [
                    "positive_funding_extreme_with_crowded_longs",
                    "contrarian_short_reversal_setup",
                    f"extreme_type:{extreme_type}",
                    f"scope:{state.scope.key}",
                ],
                "tags": self._build_setup_tags(state, extreme_event),
                "metadata": {
                    "scope": state.scope.to_dict(),
                    "extreme_type": extreme_type,
                    "regime": regime_name,
                    "bias": bias,
                    "pressure_direction": pressure_direction,
                    "pressure_level": pressure_level,
                    "pressure_score": pressure_score,
                    "squeeze_probability": squeeze_probability,
                    "mean_reversion_probability": mean_reversion_probability,
                    "is_reversal_risk": bool(self._get_value(extreme_event, "is_reversal_risk", False)),
                    "is_squeeze_risk": is_squeeze_risk,
                    "raw_extreme_severity": severity,
                    "adjusted_extreme_severity": adjusted_severity,
                    "extreme_type_bonus": extreme_type_bonus,
                    "divergence_alignment_bonus": divergence_alignment_bonus,
                    "signal_alignment_bonus": signal_alignment_bonus,
                    "signal_origins_available": sorted(state.last_signals_by_origin.keys()),
                },
            }

        if self._is_negative_extreme(extreme_type):
            if pressure_direction != FundingPressureDirection.SHORT.value:
                return None
            if bias not in {
                FundingBias.SHORT_BIAS.value,
                FundingBias.OVERCROWDED_SHORTS.value,
                FundingBias.SQUEEZE_RISK_SHORTS.value,
            }:
                return None

            score = self._compute_setup_score(
                severity=adjusted_severity,
                pressure_score=pressure_score,
                squeeze_probability=squeeze_probability,
                mean_reversion_probability=mean_reversion_probability,
                regime_confidence=regime_confidence,
                directional_alignment_bonus=1.0,
                divergence_alignment_bonus=divergence_alignment_bonus,
                signal_alignment_bonus=signal_alignment_bonus,
            )
            confidence = self._compute_setup_confidence(
                score=score,
                severity=adjusted_severity,
                regime_confidence=regime_confidence,
            )

            return {
                "direction": FundingStrategyDirection.LONG,
                "setup_type": self.config.bullish_setup_type,
                "score": score,
                "confidence": confidence,
                "reason": "negative_funding_extreme_with_crowded_shorts",
                "reasons": [
                    "negative_funding_extreme_with_crowded_shorts",
                    "contrarian_long_reversal_setup",
                    f"extreme_type:{extreme_type}",
                    f"scope:{state.scope.key}",
                ],
                "tags": self._build_setup_tags(state, extreme_event),
                "metadata": {
                    "scope": state.scope.to_dict(),
                    "extreme_type": extreme_type,
                    "regime": regime_name,
                    "bias": bias,
                    "pressure_direction": pressure_direction,
                    "pressure_level": pressure_level,
                    "pressure_score": pressure_score,
                    "squeeze_probability": squeeze_probability,
                    "mean_reversion_probability": mean_reversion_probability,
                    "is_reversal_risk": bool(self._get_value(extreme_event, "is_reversal_risk", False)),
                    "is_squeeze_risk": is_squeeze_risk,
                    "raw_extreme_severity": severity,
                    "adjusted_extreme_severity": adjusted_severity,
                    "extreme_type_bonus": extreme_type_bonus,
                    "divergence_alignment_bonus": divergence_alignment_bonus,
                    "signal_alignment_bonus": signal_alignment_bonus,
                    "signal_origins_available": sorted(state.last_signals_by_origin.keys()),
                },
            }

        return None

    def _apply_setup_candidate(
        self,
        *,
        state: FundingStrategyState,
        setup_candidate: dict[str, Any],
        event_time: Any,
    ) -> None:
        self.set_setup_detected(
            state,
            direction=setup_candidate["direction"],
            setup_type=setup_candidate["setup_type"],
            score=setup_candidate["score"],
            confidence=setup_candidate["confidence"],
            reason=setup_candidate["reason"],
            reasons=setup_candidate["reasons"],
            tags=setup_candidate["tags"],
            event_time=event_time,
            metadata=setup_candidate["metadata"],
        )

    # ------------------------------------------------------------------
    # Confirmation logic
    # ------------------------------------------------------------------

    def _can_confirm_by_flip(
        self,
        state: FundingStrategyState,
        flip_event: Any,
    ) -> bool:
        if not self.config.allow_flip_confirmation:
            return False
        if state.status not in {
            FundingSetupStatus.SETUP_DETECTED,
            FundingSetupStatus.CONFIRMED,
        }:
            return False

        flip_type = self._enum_str(self._get_value(flip_event, "flip_type"))

        return (
            state.direction == FundingStrategyDirection.SHORT
            and flip_type == FundingFlipType.POSITIVE_TO_NEGATIVE.value
        ) or (
            state.direction == FundingStrategyDirection.LONG
            and flip_type == FundingFlipType.NEGATIVE_TO_POSITIVE.value
        )

    def _can_confirm_by_pressure_release(
        self,
        state: FundingStrategyState,
        previous_pressure: Any | None,
        current_pressure: Any,
    ) -> bool:
        if not self.config.allow_pressure_release_confirmation:
            return False
        if previous_pressure is None or state.status != FundingSetupStatus.SETUP_DETECTED:
            return False

        prev_score = self._to_float(
            self._get_value(previous_pressure, "pressure_score"),
            default=0.0,
        )
        curr_score = self._to_float(
            self._get_value(current_pressure, "pressure_score"),
            default=0.0,
        )

        if prev_score <= curr_score:
            return False
        if (prev_score - curr_score) < self.config.pressure_release_min_score_drop:
            return False

        prev_level = self._enum_str(self._get_value(previous_pressure, "level"))
        curr_level = self._enum_str(self._get_value(current_pressure, "level"))

        previous_rank = self._pressure_level_rank(prev_level)
        current_rank = self._pressure_level_rank(curr_level)

        if (previous_rank - current_rank) < self.config.confirm_on_pressure_drop_levels:
            return False

        direction = self._enum_str(self._get_value(current_pressure, "direction"))

        if state.direction == FundingStrategyDirection.SHORT:
            return direction in {
                FundingPressureDirection.LONG.value,
                FundingPressureDirection.NEUTRAL.value,
            }

        if state.direction == FundingStrategyDirection.LONG:
            return direction in {
                FundingPressureDirection.SHORT.value,
                FundingPressureDirection.NEUTRAL.value,
            }

        return False

    def _can_confirm_by_divergence(
        self,
        state: FundingStrategyState,
        divergence_event: Any,
    ) -> bool:
        if not self.config.allow_divergence_confirmation:
            return False
        if state.status != FundingSetupStatus.SETUP_DETECTED:
            return False

        confidence = self._to_float(
            self._get_value(divergence_event, "confidence"),
            default=0.0,
        )
        if confidence < self.config.min_divergence_confidence:
            return False

        return self._divergence_direction(divergence_event) == state.direction

    def _can_confirm_by_signal(
        self,
        state: FundingStrategyState,
        signal: Any,
    ) -> bool:
        if not self.config.allow_signal_confirmation:
            return False
        if state.status != FundingSetupStatus.SETUP_DETECTED:
            return False

        confidence = self._to_float(
            self._get_value(signal, "confidence"),
            default=0.0,
        )
        if confidence < self.config.min_signal_confidence:
            return False

        score = self._to_float(self._get_value(signal, "score"), default=0.0)
        if abs(score) < self.config.min_signal_abs_score:
            return False

        origin = self._signal_origin(signal)
        if (
            self.config.preferred_signal_origins_for_confirmation
            and origin
            and origin not in self.config.preferred_signal_origins_for_confirmation
        ):
            origin_weight = self.config.signal_origin_confirmation_weight.get(origin, 0.0)
            if origin_weight < 0.50:
                return False

        return self._signal_direction(signal) == state.direction

    # ------------------------------------------------------------------
    # Invalidation logic
    # ------------------------------------------------------------------

    def _should_invalidate_by_flip(
        self,
        state: FundingStrategyState,
        flip_event: Any,
    ) -> bool:
        if not self.config.invalidate_on_opposite_flip:
            return False

        flip_type = self._enum_str(self._get_value(flip_event, "flip_type"))

        return (
            state.direction == FundingStrategyDirection.SHORT
            and flip_type == FundingFlipType.NEGATIVE_TO_POSITIVE.value
        ) or (
            state.direction == FundingStrategyDirection.LONG
            and flip_type == FundingFlipType.POSITIVE_TO_NEGATIVE.value
        )

    def _should_invalidate_by_pressure(
        self,
        state: FundingStrategyState,
        pressure_state: Any,
    ) -> bool:
        direction = self._enum_str(self._get_value(pressure_state, "direction"))
        level = self._enum_str(self._get_value(pressure_state, "level"))
        pressure_score = self._to_float(
            self._get_value(pressure_state, "pressure_score"),
            default=0.0,
        )

        if self.config.invalidate_on_pressure_neutralization:
            if level in {
                FundingPressureLevel.LOW.value,
                FundingPressureLevel.UNKNOWN.value,
            }:
                return True
            if pressure_score < (
                self.config.min_pressure_score * _PRESSURE_NEUTRALIZATION_THRESHOLD_RATIO
            ):
                return True

        if state.direction == FundingStrategyDirection.SHORT:
            return direction not in {
                FundingPressureDirection.LONG.value,
                FundingPressureDirection.NEUTRAL.value,
            }

        if state.direction == FundingStrategyDirection.LONG:
            return direction not in {
                FundingPressureDirection.SHORT.value,
                FundingPressureDirection.NEUTRAL.value,
            }

        return False

    def _should_invalidate_by_regime(
        self,
        state: FundingStrategyState,
        regime_state: Any,
    ) -> bool:
        regime = self._enum_str(self._get_value(regime_state, "regime"))
        bias = self._enum_str(self._get_value(regime_state, "bias"))

        if self.config.invalidate_on_regime_neutral and regime in {
            FundingRegime.NEUTRAL.value,
            FundingRegime.UNKNOWN.value,
        }:
            return True

        if not self.config.invalidate_on_regime_conflict:
            return False

        if state.direction == FundingStrategyDirection.SHORT:
            return bias in {
                FundingBias.SHORT_BIAS.value,
                FundingBias.OVERCROWDED_SHORTS.value,
                FundingBias.SQUEEZE_RISK_SHORTS.value,
            }

        if state.direction == FundingStrategyDirection.LONG:
            return bias in {
                FundingBias.LONG_BIAS.value,
                FundingBias.OVERCROWDED_LONGS.value,
                FundingBias.SQUEEZE_RISK_LONGS.value,
            }

        return False

    def _should_invalidate_by_divergence(
        self,
        state: FundingStrategyState,
        divergence_event: Any,
    ) -> bool:
        if not self.config.invalidate_on_opposite_divergence:
            return False

        direction = self._divergence_direction(divergence_event)
        return direction != FundingStrategyDirection.NEUTRAL and direction != state.direction

    def _should_invalidate_by_signal(
        self,
        state: FundingStrategyState,
        signal: Any,
    ) -> bool:
        if not self.config.invalidate_on_opposite_signal:
            return False

        direction = self._signal_direction(signal)
        if direction == FundingStrategyDirection.NEUTRAL:
            return False

        origin = self._signal_origin(signal)
        if (
            self.config.preferred_signal_origins_for_invalidation
            and origin
            and origin not in self.config.preferred_signal_origins_for_invalidation
        ):
            return False

        return direction != state.direction

    def _should_invalidate_by_extreme_context(
        self,
        state: FundingStrategyState,
        extreme_event: Any,
    ) -> bool:
        if not self.config.invalidate_on_extreme_no_longer_reversal_risk:
            return False

        if bool(self._get_value(extreme_event, "is_reversal_risk", False)):
            return False

        extreme_direction = self._extreme_reversal_direction(extreme_event)

        return (
            extreme_direction != FundingStrategyDirection.NEUTRAL
            and extreme_direction == state.direction
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_setup_score(
        self,
        *,
        severity: float,
        pressure_score: float,
        squeeze_probability: float,
        mean_reversion_probability: float,
        regime_confidence: float,
        directional_alignment_bonus: float,
        divergence_alignment_bonus: float = 0.0,
        signal_alignment_bonus: float = 0.0,
    ) -> float:
        return self._weighted_average(
            [
                (severity, 0.24),
                (pressure_score, 0.22),
                (max(squeeze_probability, mean_reversion_probability), 0.18),
                (regime_confidence, 0.14),
                (directional_alignment_bonus, 0.10),
                (divergence_alignment_bonus, 0.07),
                (signal_alignment_bonus, 0.05),
            ]
        )

    def _compute_setup_confidence(
        self,
        *,
        score: float,
        severity: float,
        regime_confidence: float,
    ) -> float:
        return self._weighted_average(
            [
                (score, 0.45),
                (severity, 0.35),
                (regime_confidence, 0.20),
            ]
        )

    def _compute_confirmation_score_from_flip(
        self,
        state: FundingStrategyState,
        flip_event: Any,
    ) -> float:
        flip_confidence = self._to_float(
            self._get_value(flip_event, "confidence"),
            default=0.0,
        )
        return self._weighted_average(
            [
                (state.score, 0.35),
                (flip_confidence, 0.40),
                (1.0, 0.25),
            ]
        )

    def _compute_confirmation_confidence_from_flip(
        self,
        state: FundingStrategyState,
        flip_event: Any,
    ) -> float:
        flip_confidence = self._to_float(
            self._get_value(flip_event, "confidence"),
            default=0.0,
        )
        return self._weighted_average(
            [
                (state.confidence, 0.40),
                (flip_confidence, 0.60),
            ]
        )

    def _compute_confirmation_score_from_release(
        self,
        state: FundingStrategyState,
        current_pressure: Any,
    ) -> float:
        pressure_score = self._to_float(
            self._get_value(current_pressure, "pressure_score"),
            default=0.0,
        )
        return self._weighted_average(
            [
                (state.score, 0.40),
                (1.0 - min(pressure_score, 1.0), 0.35),
                (1.0, 0.25),
            ]
        )

    def _compute_confirmation_confidence_from_release(
        self,
        state: FundingStrategyState,
        current_pressure: Any,
    ) -> float:
        mean_reversion_probability = self._to_float(
            self._get_value(current_pressure, "mean_reversion_probability"),
            default=0.0,
        )
        squeeze_probability = self._to_float(
            self._get_value(current_pressure, "squeeze_probability"),
            default=0.0,
        )

        return self._weighted_average(
            [
                (state.confidence, 0.35),
                (mean_reversion_probability, 0.35),
                (squeeze_probability, 0.30),
            ]
        )

    def _compute_confirmation_score_from_divergence(
        self,
        state: FundingStrategyState,
        divergence_event: Any,
    ) -> float:
        divergence_confidence = self._to_float(
            self._get_value(divergence_event, "confidence"),
            default=0.0,
        )
        divergence_bonus = self._divergence_type_bonus(divergence_event)

        return self._weighted_average(
            [
                (state.score, _CONTEXT_CONFIRMATION_SCORE_WEIGHT),
                (
                    self._clip_score(divergence_confidence + divergence_bonus),
                    _DIVERGENCE_CONFIRMATION_SCORE_WEIGHT,
                ),
                (
                    1.0,
                    1.0
                    - _CONTEXT_CONFIRMATION_SCORE_WEIGHT
                    - _DIVERGENCE_CONFIRMATION_SCORE_WEIGHT,
                ),
            ]
        )

    def _compute_confirmation_confidence_from_divergence(
        self,
        state: FundingStrategyState,
        divergence_event: Any,
    ) -> float:
        divergence_confidence = self._to_float(
            self._get_value(divergence_event, "confidence"),
            default=0.0,
        )
        return self._weighted_average(
            [
                (state.confidence, 0.45),
                (divergence_confidence, 0.55),
            ]
        )

    def _compute_confirmation_score_from_signal(
        self,
        state: FundingStrategyState,
        signal: Any,
    ) -> float:
        signal_score = abs(
            self._to_float(self._get_value(signal, "score"), default=0.0)
        )
        origin_weight = self._signal_origin_confirmation_weight(signal)

        return self._weighted_average(
            [
                (state.score, 1.0 - _SIGNAL_CONFIRMATION_SCORE_WEIGHT),
                (
                    signal_score * origin_weight,
                    _SIGNAL_CONFIRMATION_SCORE_WEIGHT,
                ),
            ]
        )

    def _compute_confirmation_confidence_from_signal(
        self,
        state: FundingStrategyState,
        signal: Any,
    ) -> float:
        signal_confidence = self._to_float(
            self._get_value(signal, "confidence"),
            default=0.0,
        )
        origin_weight = self._signal_origin_confirmation_weight(signal)

        return self._weighted_average(
            [
                (state.confidence, 0.45),
                (signal_confidence * origin_weight, 0.55),
            ]
        )

    # ------------------------------------------------------------------
    # Atomic update evaluation
    # ------------------------------------------------------------------

    async def _evaluate_active_state_after_atomic_update(
        self,
        *,
        state: FundingStrategyState,
        correlation_id: str | None,
    ) -> None:
        best_opposite_signal = self._best_signal_for_direction(
            state,
            target_direction=self._opposite_direction(state.direction),
        )
        if best_opposite_signal is not None and self._should_invalidate_by_signal(
            state,
            best_opposite_signal,
        ):
            self.set_invalidated(
                state,
                reason="atomic_update_opposite_signal_invalidated_reversal_setup",
                cooldown=True,
                metadata={
                    "invalidation_source": "funding_updated.signal",
                    "signal_origin": self._signal_origin(best_opposite_signal),
                    "scope": state.scope.to_dict(),
                },
            )
            await self.emit_invalidated(
                state,
                extra_payload={
                    "trigger": "funding_updated",
                    "correlation_id": correlation_id,
                },
            )
            return

        if state.last_divergence is not None and self._should_invalidate_by_divergence(
            state,
            state.last_divergence,
        ):
            self.set_invalidated(
                state,
                reason="atomic_update_opposite_divergence_invalidated_reversal_setup",
                cooldown=True,
                metadata={
                    "invalidation_source": "funding_updated.divergence",
                    "scope": state.scope.to_dict(),
                },
            )
            await self.emit_invalidated(
                state,
                extra_payload={
                    "trigger": "funding_updated",
                    "correlation_id": correlation_id,
                },
            )
            return

        if state.last_extreme is not None and self._should_invalidate_by_extreme_context(
            state,
            state.last_extreme,
        ):
            self.set_invalidated(
                state,
                reason="atomic_update_extreme_context_invalidated_reversal_setup",
                cooldown=True,
                metadata={
                    "invalidation_source": "funding_updated.extreme",
                    "scope": state.scope.to_dict(),
                },
            )
            await self.emit_invalidated(
                state,
                extra_payload={
                    "trigger": "funding_updated",
                    "correlation_id": correlation_id,
                },
            )
            return

        if state.status != FundingSetupStatus.SETUP_DETECTED:
            return

        best_aligned_signal = self._best_signal_for_direction(
            state,
            target_direction=state.direction,
        )
        if best_aligned_signal is not None and self._can_confirm_by_signal(
            state,
            best_aligned_signal,
        ):
            self.set_confirmed(
                state,
                score=self._compute_confirmation_score_from_signal(
                    state,
                    best_aligned_signal,
                ),
                confidence=self._compute_confirmation_confidence_from_signal(
                    state,
                    best_aligned_signal,
                ),
                reason="atomic_update_signal_confirmed_reversal_setup",
                tags=[
                    self.config.tag_confirmed_by_signal,
                    self.config.tag_atomic_context,
                ],
                event_time=self._extract_event_time_from_normalized(best_aligned_signal),
                metadata={
                    "confirmation_source": "funding_updated.signal",
                    "signal_origin": self._signal_origin(best_aligned_signal),
                    "scope": state.scope.to_dict(),
                },
            )
            await self.emit_confirmed(
                state,
                extra_payload={
                    "trigger": "funding_updated",
                    "correlation_id": correlation_id,
                },
            )
            return

        if state.last_divergence is not None and self._can_confirm_by_divergence(
            state,
            state.last_divergence,
        ):
            self.set_confirmed(
                state,
                score=self._compute_confirmation_score_from_divergence(
                    state,
                    state.last_divergence,
                ),
                confidence=self._compute_confirmation_confidence_from_divergence(
                    state,
                    state.last_divergence,
                ),
                reason="atomic_update_divergence_confirmed_reversal_setup",
                tags=[
                    self.config.tag_confirmed_by_divergence,
                    self.config.tag_atomic_context,
                ],
                event_time=self._extract_event_time_from_normalized(state.last_divergence),
                metadata={
                    "confirmation_source": "funding_updated.divergence",
                    "scope": state.scope.to_dict(),
                },
            )
            await self.emit_confirmed(
                state,
                extra_payload={
                    "trigger": "funding_updated",
                    "correlation_id": correlation_id,
                },
            )

    # ------------------------------------------------------------------
    # Direction helpers
    # ------------------------------------------------------------------

    def _divergence_direction(
        self,
        divergence_event: Any,
    ) -> FundingStrategyDirection:
        divergence_type = self._enum_str(
            self._get_value(divergence_event, "divergence_type")
        )

        bullish = {
            FundingDivergenceType.PRICE_UP_FUNDING_DOWN.value,
            FundingDivergenceType.OI_UP_FUNDING_DOWN.value,
            FundingDivergenceType.CVD_UP_FUNDING_DOWN.value,
            FundingDivergenceType.LIQUIDATIONS_SHORTS_WITH_NEGATIVE_FUNDING.value,
        }
        bearish = {
            FundingDivergenceType.PRICE_DOWN_FUNDING_UP.value,
            FundingDivergenceType.OI_UP_FUNDING_UP_PRICE_STALLED.value,
            FundingDivergenceType.CVD_DOWN_FUNDING_UP.value,
            FundingDivergenceType.LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING.value,
        }

        if divergence_type in bullish:
            return FundingStrategyDirection.LONG
        if divergence_type in bearish:
            return FundingStrategyDirection.SHORT
        return FundingStrategyDirection.NEUTRAL

    def _signal_direction(
        self,
        signal: Any,
    ) -> FundingStrategyDirection:
        score = self._to_float(self._get_value(signal, "score"), default=0.0)
        signal_type = self._enum_str(self._get_value(signal, "signal_type"))
        bias = self._enum_str(self._get_value(signal, "bias"))

        if score >= self.config.min_signal_abs_score:
            return FundingStrategyDirection.LONG
        if score <= -self.config.min_signal_abs_score:
            return FundingStrategyDirection.SHORT

        directional_types = {
            FundingSignalType.REVERSION_SETUP.value,
            FundingSignalType.DIVERGENCE_DETECTED.value,
            FundingSignalType.FLIP_DETECTED.value,
        }

        if (
            signal_type in directional_types
            and bias
            in {
                FundingBias.SHORT_BIAS.value,
                FundingBias.OVERCROWDED_SHORTS.value,
                FundingBias.SQUEEZE_RISK_SHORTS.value,
            }
        ):
            return FundingStrategyDirection.LONG

        if (
            signal_type in directional_types
            and bias
            in {
                FundingBias.LONG_BIAS.value,
                FundingBias.OVERCROWDED_LONGS.value,
                FundingBias.SQUEEZE_RISK_LONGS.value,
            }
        ):
            return FundingStrategyDirection.SHORT

        return FundingStrategyDirection.NEUTRAL

    def _extreme_reversal_direction(
        self,
        extreme_event: Any,
    ) -> FundingStrategyDirection:
        extreme_type = self._enum_str(self._get_value(extreme_event, "extreme_type"))

        if self._is_positive_extreme(extreme_type):
            return FundingStrategyDirection.SHORT
        if self._is_negative_extreme(extreme_type):
            return FundingStrategyDirection.LONG
        return FundingStrategyDirection.NEUTRAL

    @staticmethod
    def _opposite_direction(
        direction: FundingStrategyDirection,
    ) -> FundingStrategyDirection:
        if direction == FundingStrategyDirection.LONG:
            return FundingStrategyDirection.SHORT
        if direction == FundingStrategyDirection.SHORT:
            return FundingStrategyDirection.LONG
        return FundingStrategyDirection.NEUTRAL

    def _calc_divergence_alignment_bonus(
        self,
        state: FundingStrategyState,
    ) -> float:
        if state.last_divergence is None:
            return 0.0

        direction = self._divergence_direction(state.last_divergence)
        confidence = self._to_float(
            self._get_value(state.last_divergence, "confidence"),
            default=0.0,
        )
        type_bonus = self._divergence_type_bonus(state.last_divergence)

        extreme_type = self._enum_str(self._get_value(state.last_extreme, "extreme_type"))

        if self._is_positive_extreme(extreme_type) and direction == FundingStrategyDirection.SHORT:
            return self._clip_score(confidence + type_bonus)

        if self._is_negative_extreme(extreme_type) and direction == FundingStrategyDirection.LONG:
            return self._clip_score(confidence + type_bonus)

        return 0.0

    def _calc_signal_alignment_bonus(
        self,
        state: FundingStrategyState,
    ) -> float:
        target_direction = self._extreme_reversal_direction(state.last_extreme)
        best_signal = self._best_signal_for_direction(state, target_direction)

        if best_signal is None:
            return 0.0

        confidence = self._to_float(
            self._get_value(best_signal, "confidence"),
            default=0.0,
        )
        origin_weight = self._signal_origin_alignment_weight(best_signal)

        return self._clip_score(confidence * origin_weight)

    def _best_signal_for_direction(
        self,
        state: FundingStrategyState,
        target_direction: FundingStrategyDirection,
    ) -> Any | None:
        if target_direction == FundingStrategyDirection.NEUTRAL:
            return None

        candidates = list(state.recent_signals)
        if state.last_signal is not None and state.last_signal not in candidates:
            candidates.append(state.last_signal)

        if not candidates:
            return None

        best_signal = None
        best_score = -1.0

        for signal in candidates:
            if self._signal_direction(signal) != target_direction:
                continue

            confidence = self._to_float(
                self._get_value(signal, "confidence"),
                default=0.0,
            )
            score = abs(self._to_float(self._get_value(signal, "score"), default=0.0))
            origin_weight = self._signal_origin_alignment_weight(signal)

            rank = self._clip_score(
                (confidence * 0.55) + (score * 0.30) + (origin_weight * 0.15)
            )

            if rank > best_score:
                best_score = rank
                best_signal = signal

        return best_signal

    def _signal_origin_alignment_weight(
        self,
        signal: Any,
    ) -> float:
        origin = self._signal_origin(signal)
        if not origin:
            return 0.50
        return self._clip_score(
            self.config.signal_origin_alignment_weight.get(origin, 0.50)
        )

    def _signal_origin_confirmation_weight(
        self,
        signal: Any,
    ) -> float:
        origin = self._signal_origin(signal)
        if not origin:
            return 0.50
        return self._clip_score(
            self.config.signal_origin_confirmation_weight.get(origin, 0.50)
        )

    def _divergence_type_bonus(
        self,
        divergence_event: Any,
    ) -> float:
        divergence_type = self._enum_str(
            self._get_value(divergence_event, "divergence_type")
        )

        if divergence_type in {
            FundingDivergenceType.LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING.value,
            FundingDivergenceType.LIQUIDATIONS_SHORTS_WITH_NEGATIVE_FUNDING.value,
        }:
            return self.config.liquidation_divergence_bonus

        if divergence_type in {
            FundingDivergenceType.CVD_UP_FUNDING_DOWN.value,
            FundingDivergenceType.CVD_DOWN_FUNDING_UP.value,
        }:
            return self.config.cvd_divergence_bonus

        if divergence_type in {
            FundingDivergenceType.OI_UP_FUNDING_DOWN.value,
            FundingDivergenceType.OI_UP_FUNDING_UP_PRICE_STALLED.value,
        }:
            return self.config.oi_divergence_bonus

        if divergence_type in {
            FundingDivergenceType.PRICE_UP_FUNDING_DOWN.value,
            FundingDivergenceType.PRICE_DOWN_FUNDING_UP.value,
        }:
            return self.config.price_divergence_bonus

        return 0.0

    def _extreme_type_bonus(
        self,
        extreme_type: str | None,
    ) -> float:
        if extreme_type in {
            FundingExtremeType.GLOBAL_HIGH.value,
            FundingExtremeType.GLOBAL_LOW.value,
        }:
            return self.config.global_extreme_bonus

        if extreme_type in {
            FundingExtremeType.PERCENTILE_HIGH.value,
            FundingExtremeType.PERCENTILE_LOW.value,
        }:
            return self.config.percentile_extreme_bonus

        if extreme_type in {
            FundingExtremeType.ZSCORE_HIGH.value,
            FundingExtremeType.ZSCORE_LOW.value,
        }:
            return self.config.zscore_extreme_bonus

        if extreme_type in {
            FundingExtremeType.LOCAL_HIGH.value,
            FundingExtremeType.LOCAL_LOW.value,
        }:
            return self.config.local_extreme_bonus

        return 0.0

    # ------------------------------------------------------------------
    # Emit hooks
    # ------------------------------------------------------------------

    def on_before_setup_emit(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "extreme_reversal"
        payload["signal_class"] = "contrarian_reversal"
        payload["scope"] = state.scope.to_dict()
        return payload

    def on_before_confirmation_emit(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "extreme_reversal"
        payload["signal_class"] = "contrarian_reversal"
        payload["is_tradeable"] = True
        payload["scope"] = state.scope.to_dict()
        return payload

    def on_before_invalidation_emit(
            self,
            state: FundingStrategyState,
            payload: dict[str, Any],
    ) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "extreme_reversal"
        payload["signal_class"] = "contrarian_reversal"
        payload["scope"] = state.scope.to_dict()
        return payload

    def on_before_expiration_emit(
            self,
            state: FundingStrategyState,
            payload: dict[str, Any],
    ) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "extreme_reversal"
        payload["signal_class"] = "contrarian_reversal"
        payload["scope"] = state.scope.to_dict()
        return payload

    # ------------------------------------------------------------------
    # Internal utils
    # ------------------------------------------------------------------

    def _build_setup_tags(
        self,
        state: FundingStrategyState,
        extreme_event: Any,
    ) -> list[str]:
        tags = [
            self.config.tag_extreme,
            self.config.tag_reversal,
            self.config.tag_crowding,
            self.config.tag_squeeze,
        ]

        extreme_type = self._enum_str(self._get_value(extreme_event, "extreme_type"))

        if extreme_type in {
            FundingExtremeType.GLOBAL_HIGH.value,
            FundingExtremeType.GLOBAL_LOW.value,
        }:
            tags.append(self.config.tag_global_extreme)

        if extreme_type in {
            FundingExtremeType.PERCENTILE_HIGH.value,
            FundingExtremeType.PERCENTILE_LOW.value,
        }:
            tags.append(self.config.tag_percentile_extreme)

        if extreme_type in {
            FundingExtremeType.ZSCORE_HIGH.value,
            FundingExtremeType.ZSCORE_LOW.value,
        }:
            tags.append(self.config.tag_zscore_extreme)

        if extreme_type in {
            FundingExtremeType.LOCAL_HIGH.value,
            FundingExtremeType.LOCAL_LOW.value,
        }:
            tags.append(self.config.tag_local_extreme)

        if state.last_divergence is not None:
            tags.append(self.config.tag_divergence)

        if state.last_signal is not None:
            tags.append(self.config.tag_signal)

        return tags

    @staticmethod
    def _is_positive_extreme(extreme_type: str | None) -> bool:
        return extreme_type in {
            FundingExtremeType.LOCAL_HIGH.value,
            FundingExtremeType.GLOBAL_HIGH.value,
            FundingExtremeType.ZSCORE_HIGH.value,
            FundingExtremeType.PERCENTILE_HIGH.value,
        }

    @staticmethod
    def _is_negative_extreme(extreme_type: str | None) -> bool:
        return extreme_type in {
            FundingExtremeType.LOCAL_LOW.value,
            FundingExtremeType.GLOBAL_LOW.value,
            FundingExtremeType.ZSCORE_LOW.value,
            FundingExtremeType.PERCENTILE_LOW.value,
        }


__all__ = [
    "FundingExtremeReversalStrategy",
    "FundingExtremeReversalStrategyConfig",
]