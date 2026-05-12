from __future__ import annotations

from dataclasses import dataclass
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


_ALIGNMENT_BONUS_BASE: float = 0.50
_ALIGNMENT_BONUS_PER_DIMENSION: float = 0.25
_FLIP_CONFIRMATION_PERFECT_WEIGHT: float = 1.0
_PRESSURE_BREAKDOWN_THRESHOLD_RATIO: float = 0.70
_PRESSURE_RELEASE_TARGET_CONFIDENCE: float = 0.85
_EXTREME_ALIGNMENT_WEIGHT: float = 0.08
_SIGNAL_ALIGNMENT_WEIGHT: float = 0.07
_SIGNAL_CONFIRMATION_WEIGHT: float = 0.35
_EXTREME_CONFIRMATION_WEIGHT: float = 0.30


@dataclass(slots=True)
class FundingDivergenceStrategyConfig(BaseFundingStrategyConfig):
    """
    Production-grade funding divergence strategy config.

    Strategy idea:
    - detect directional funding dislocation from analytics.funding.divergence;
    - use regime and pressure as context filters;
    - use flip, repeated divergence, pressure release, regime shift, extreme context,
      and normalized funding signal as confirmation/invalidation layers.
    """

    strategy_namespace: str = "strategy.funding.divergence"
    source_name: str = "funding_divergence_strategy"
    service_name: str = "funding_divergence_strategy"

    enable_funding_updated_subscription: bool = True
    enable_funding_signal_subscription: bool = True
    funding_updated_event_name: str = "analytics.funding.updated"
    funding_signal_event_name: str = "analytics.funding.signal"

    regime_event_name: str = "analytics.funding.regime"
    pressure_event_name: str = "analytics.funding.pressure"
    divergence_event_name: str = "analytics.funding.divergence"
    flip_event_name: str = "analytics.funding.flip"
    extreme_event_name: str = "analytics.funding.extreme"

    min_divergence_confidence: float = 0.50
    min_pressure_score: float = 0.35
    min_regime_confidence: float = 0.10
    min_extreme_severity: float = 0.45
    min_signal_confidence: float = 0.45
    min_signal_abs_score: float = 0.30

    require_non_neutral_regime: bool = True
    require_pressure_alignment: bool = False
    require_pressure_present: bool = False

    score_weight_divergence: float = 0.40
    score_weight_pressure: float = 0.22
    score_weight_regime: float = 0.18
    score_weight_alignment: float = 0.15
    score_weight_extreme_alignment: float = _EXTREME_ALIGNMENT_WEIGHT
    score_weight_signal_alignment: float = _SIGNAL_ALIGNMENT_WEIGHT

    allow_flip_confirmation: bool = True
    allow_repeat_divergence_confirmation: bool = True
    allow_pressure_release_confirmation: bool = True
    allow_regime_shift_confirmation: bool = True
    allow_extreme_confirmation: bool = True
    allow_signal_confirmation: bool = True
    allow_updated_context_setup: bool = True

    repeat_divergence_confirmation_bonus: float = 0.12
    regime_shift_confirmation_bonus: float = 0.10
    extreme_confirmation_bonus: float = 0.10

    pressure_release_min_score_drop: float = 0.08
    confirm_on_pressure_drop_levels: int = 1

    invalidate_on_opposite_flip: bool = True
    invalidate_on_opposite_divergence: bool = True
    invalidate_on_opposite_signal: bool = True
    invalidate_on_opposite_extreme: bool = False
    invalidate_on_pressure_breakdown: bool = True
    invalidate_on_regime_conflict: bool = True

    bullish_setup_type: str = "funding_bullish_divergence"
    bearish_setup_type: str = "funding_bearish_divergence"

    tag_divergence: str = "funding_divergence"
    tag_dislocation: str = "dislocation"
    tag_reversal: str = "reversal"
    tag_extreme: str = "funding_extreme"
    tag_signal: str = "funding_signal"
    tag_atomic_context: str = "atomic_funding_context"
    tag_confirmed_by_flip: str = "confirmed_by_flip"
    tag_confirmed_by_repeat: str = "confirmed_by_repeat_divergence"
    tag_confirmed_by_release: str = "confirmed_by_pressure_release"
    tag_confirmed_by_regime: str = "confirmed_by_regime_shift"
    tag_confirmed_by_extreme: str = "confirmed_by_extreme"
    tag_confirmed_by_signal: str = "confirmed_by_funding_signal"

    def validate(self) -> None:
        BaseFundingStrategyConfig.validate(self)

        bounded_fields = {
            "min_divergence_confidence": self.min_divergence_confidence,
            "min_pressure_score": self.min_pressure_score,
            "min_regime_confidence": self.min_regime_confidence,
            "min_extreme_severity": self.min_extreme_severity,
            "min_signal_confidence": self.min_signal_confidence,
            "min_signal_abs_score": self.min_signal_abs_score,
            "repeat_divergence_confirmation_bonus": self.repeat_divergence_confirmation_bonus,
            "regime_shift_confirmation_bonus": self.regime_shift_confirmation_bonus,
            "extreme_confirmation_bonus": self.extreme_confirmation_bonus,
            "pressure_release_min_score_drop": self.pressure_release_min_score_drop,
        }
        for field_name, value in bounded_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")

        if self.confirm_on_pressure_drop_levels < 0:
            raise ValueError("confirm_on_pressure_drop_levels must be >= 0")

        for attr in (
                "score_weight_divergence",
                "score_weight_pressure",
                "score_weight_regime",
                "score_weight_alignment",
                "score_weight_extreme_alignment",
                "score_weight_signal_alignment",
        ):
            value = getattr(self, attr)
            if value < 0.0:
                raise ValueError(f"{attr} must be >= 0")


class FundingDivergenceStrategy(BaseFundingStrategy):
    """
    Event-driven strategy for funding divergence setups.

    Runtime flow:
    - divergence creates directional setup;
    - regime/pressure filter and confirm/invalidate context;
    - flip, repeat divergence, pressure release, regime shift, extreme alignment,
      and funding.signal may confirm;
    - opposite analytics context invalidates stale setups.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: FundingDivergenceStrategyConfig | None = None,
        scheduler: Scheduler | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_config = config or FundingDivergenceStrategyConfig()
        super().__init__(
            event_bus=event_bus,
            config=resolved_config,
            scheduler=scheduler,
            service_name=service_name or resolved_config.service_name,
        )
        self.config: FundingDivergenceStrategyConfig = resolved_config

    @property
    def strategy_name(self) -> str:
        return "funding_divergence"

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
            self.config.divergence_event_name,
            self.on_divergence,
            name=f"{self.strategy_name}.on_divergence",
        )
        self.subscribe(
            self.config.flip_event_name,
            self.on_flip,
            name=f"{self.strategy_name}.on_flip",
        )
        self.subscribe(
            self.config.extreme_event_name,
            self.on_extreme,
            name=f"{self.strategy_name}.on_extreme",
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
        if self.is_in_cooldown(state):
            return

        if state.is_active():
            await self._evaluate_active_state_after_atomic_update(
                state=state,
                correlation_id=event.correlation_id,
            )
            return

        if not self.config.allow_updated_context_setup:
            return

        divergence_event = state.last_divergence
        if divergence_event is None:
            return

        event_time = self._extract_event_time_from_normalized(divergence_event)
        if self.is_stale_event(event_time):
            return

        if not self._can_create_setup_from_divergence(state=state, divergence_event=divergence_event):
            return

        setup_candidate = self._build_setup_from_divergence(
            state=state,
            divergence_event=divergence_event,
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
                reason="opposite_funding_signal_invalidated_divergence_setup",
                cooldown=True,
                metadata={"invalidation_source": "funding_signal"},
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
                reason="funding_signal_confirmed_divergence_setup",
                tags=[self.config.tag_confirmed_by_signal, self.config.tag_signal],
                event_time=self._extract_event_time_from_normalized(signal),
                metadata={"confirmation_source": "funding_signal"},
            )
            await self.emit_confirmed(
                state,
                extra_payload={
                    "trigger": "funding_signal",
                    "correlation_id": event.correlation_id,
                },
            )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _is_opposite_divergence_direction_for_state(
            self,
            state: FundingStrategyState,
            divergence_event: Any,
    ) -> bool:
        direction = self._derive_direction_from_divergence(divergence_event)
        return (
                direction != FundingStrategyDirection.NEUTRAL
                and state.direction != FundingStrategyDirection.NEUTRAL
                and direction != state.direction
        )

    async def on_regime(self, event: Event) -> None:
        payload = self.extract_payload(event)
        symbol, exchange = self.extract_symbol_exchange(payload)
        if not symbol:
            return

        lock = await self.acquire_symbol_lock(symbol, exchange)
        if lock is None:
            return

        try:
            state = self.get_state(symbol, exchange)
            previous_regime = state.last_regime
            regime_state = self._normalize_regime_payload(payload)

            self.attach_regime(state, regime_state)
            self._expire_state_if_needed(state)

            if self.is_in_cooldown(state) or not state.is_active():
                return

            if self._should_invalidate_by_regime(state, regime_state):
                self.set_invalidated(
                    state,
                    reason="regime_context_invalidated_divergence_setup",
                    cooldown=True,
                    metadata={"invalidation_source": "regime"},
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={"trigger": "regime", "correlation_id": event.correlation_id},
                )
                return

            if self._can_confirm_by_regime_shift(
                state=state,
                previous_regime=previous_regime,
                current_regime=regime_state,
            ):
                self.set_confirmed(
                    state,
                    score=self._compute_confirmation_score_from_regime_shift(
                        state=state,
                        regime_state=regime_state,
                    ),
                    confidence=self._compute_confirmation_confidence_from_regime_shift(
                        state=state,
                        regime_state=regime_state,
                    ),
                    reason="regime_shift_confirmed_divergence_setup",
                    tags=[self.config.tag_confirmed_by_regime],
                    event_time=self._extract_event_time_from_normalized(regime_state),
                    metadata={"confirmation_source": "regime_shift"},
                )
                await self.emit_confirmed(
                    state,
                    extra_payload={"trigger": "regime_shift", "correlation_id": event.correlation_id},
                )

        except Exception:
            self.logger.exception(
                "Failed to process funding divergence regime event | symbol=%s exchange=%s",
                symbol,
                exchange,
            )
        finally:
            self.release_symbol_lock(lock)

    async def on_pressure(self, event: Event) -> None:
        payload = self.extract_payload(event)
        symbol, exchange = self.extract_symbol_exchange(payload)
        if not symbol:
            return

        lock = await self.acquire_symbol_lock(symbol, exchange)
        if lock is None:
            return

        try:
            state = self.get_state(symbol, exchange)
            previous_pressure = state.last_pressure
            pressure_state = self._normalize_pressure_payload(payload)

            self.attach_pressure(state, pressure_state)
            self._expire_state_if_needed(state)

            if self.is_in_cooldown(state) or not state.is_active():
                return

            if self._should_invalidate_by_pressure(state, pressure_state):
                self.set_invalidated(
                    state,
                    reason="pressure_context_invalidated_divergence_setup",
                    cooldown=True,
                    metadata={"invalidation_source": "pressure"},
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={"trigger": "pressure", "correlation_id": event.correlation_id},
                )
                return

            if self._can_confirm_by_pressure_release(
                state=state,
                previous_pressure=previous_pressure,
                current_pressure=pressure_state,
            ):
                self.set_confirmed(
                    state,
                    score=self._compute_confirmation_score_from_pressure_release(
                        state=state,
                        pressure_state=pressure_state,
                    ),
                    confidence=self._compute_confirmation_confidence_from_pressure_release(
                        state=state,
                        pressure_state=pressure_state,
                    ),
                    reason="pressure_release_confirmed_divergence_setup",
                    tags=[self.config.tag_confirmed_by_release],
                    event_time=self._extract_event_time_from_normalized(pressure_state),
                    metadata={"confirmation_source": "pressure_release"},
                )
                await self.emit_confirmed(
                    state,
                    extra_payload={"trigger": "pressure_release", "correlation_id": event.correlation_id},
                )

        except Exception:
            self.logger.exception(
                "Failed to process funding divergence pressure event | symbol=%s exchange=%s",
                symbol,
                exchange,
            )
        finally:
            self.release_symbol_lock(lock)

    async def on_divergence(self, event: Event) -> None:
        payload = self.extract_payload(event)
        symbol, exchange = self.extract_symbol_exchange(payload)
        if not symbol:
            return

        lock = await self.acquire_symbol_lock(symbol, exchange)
        if lock is None:
            return

        try:
            state = self.get_state(symbol, exchange)
            previous_divergence = state.last_divergence
            divergence_event = self._normalize_divergence_payload(payload)

            self.attach_divergence(state, divergence_event)
            self._expire_state_if_needed(state)

            if self.is_in_cooldown(state):
                return

            event_time = self._extract_event_time_from_normalized(divergence_event)
            if self.is_stale_event(event_time):
                self.logger.debug(
                    "Stale funding divergence event ignored | symbol=%s exchange=%s",
                    symbol,
                    exchange,
                )
                return

            current_direction = self._derive_direction_from_divergence(divergence_event)
            if current_direction == FundingStrategyDirection.NEUTRAL:
                return

            # ------------------------------------------------------------------
            # Active setup handling must happen BEFORE setup creation.
            #
            # Production rule:
            # - aligned repeat divergence confirms an active setup;
            # - opposite divergence invalidates if enabled;
            # - opposite divergence is ignored if invalidation is disabled;
            # - active setup is never silently flipped LONG -> SHORT or SHORT -> LONG.
            # ------------------------------------------------------------------
            if state.is_active():
                if self._is_opposite_divergence_direction_for_state(state, divergence_event):
                    if self.config.invalidate_on_opposite_divergence:
                        self.set_invalidated(
                            state,
                            reason="opposite_divergence_invalidated_setup",
                            cooldown=True,
                            metadata={"invalidation_source": "divergence"},
                        )
                        await self.emit_invalidated(
                            state,
                            extra_payload={
                                "trigger": "divergence",
                                "correlation_id": event.correlation_id,
                            },
                        )
                    return

                if self._can_confirm_by_repeat_divergence(
                        state=state,
                        previous_divergence=previous_divergence,
                        current_divergence=divergence_event,
                ):
                    self.set_confirmed(
                        state,
                        score=self._compute_confirmation_score_from_repeat_divergence(
                            state=state,
                            divergence_event=divergence_event,
                        ),
                        confidence=self._compute_confirmation_confidence_from_repeat_divergence(
                            state=state,
                            divergence_event=divergence_event,
                        ),
                        reason="repeat_divergence_confirmed_setup",
                        tags=[self.config.tag_confirmed_by_repeat],
                        event_time=event_time,
                        metadata={"confirmation_source": "repeat_divergence"},
                    )
                    await self.emit_confirmed(
                        state,
                        extra_payload={
                            "trigger": "repeat_divergence",
                            "correlation_id": event.correlation_id,
                        },
                    )
                    return

                # Active setup exists, divergence is aligned, but not strong/valid enough
                # to confirm. Do not refresh setup repeatedly because that hides lifecycle
                # transitions and can extend TTL without a confirmation event.
                return

            if self._can_create_setup_from_divergence(
                    state=state,
                    divergence_event=divergence_event,
            ):
                setup_candidate = self._build_setup_from_divergence(
                    state=state,
                    divergence_event=divergence_event,
                )
                if setup_candidate is not None:
                    self._apply_setup_candidate(
                        state=state,
                        setup_candidate=setup_candidate,
                        event_time=event_time,
                    )
                    await self.emit_setup(
                        state,
                        extra_payload={
                            "trigger": "divergence",
                            "correlation_id": event.correlation_id,
                        },
                    )

        except Exception:
            self.logger.exception(
                "Failed to process funding divergence event | symbol=%s exchange=%s",
                symbol,
                exchange,
            )
        finally:
            self.release_symbol_lock(lock)

    async def on_flip(self, event: Event) -> None:
        payload = self.extract_payload(event)
        symbol, exchange = self.extract_symbol_exchange(payload)
        if not symbol:
            return

        lock = await self.acquire_symbol_lock(symbol, exchange)
        if lock is None:
            return

        try:
            state = self.get_state(symbol, exchange)
            flip_event = self._normalize_flip_payload(payload)

            self.attach_flip(state, flip_event)
            self._expire_state_if_needed(state)

            if self.is_in_cooldown(state) or not state.is_active():
                return

            if self._should_invalidate_by_flip(state, flip_event):
                self.set_invalidated(
                    state,
                    reason="opposite_flip_invalidated_divergence_setup",
                    cooldown=True,
                    metadata={"invalidation_source": "flip"},
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={"trigger": "flip", "correlation_id": event.correlation_id},
                )
                return

            if self._can_confirm_by_flip(state, flip_event):
                self.set_confirmed(
                    state,
                    score=self._compute_confirmation_score_from_flip(state=state, flip_event=flip_event),
                    confidence=self._compute_confirmation_confidence_from_flip(state=state, flip_event=flip_event),
                    reason="flip_confirmed_divergence_setup",
                    tags=[self.config.tag_confirmed_by_flip],
                    event_time=self._extract_event_time_from_normalized(flip_event),
                    metadata={"confirmation_source": "flip"},
                )
                await self.emit_confirmed(
                    state,
                    extra_payload={"trigger": "flip", "correlation_id": event.correlation_id},
                )

        except Exception:
            self.logger.exception(
                "Failed to process funding divergence flip event | symbol=%s exchange=%s",
                symbol,
                exchange,
            )
        finally:
            self.release_symbol_lock(lock)

    async def on_extreme(self, event: Event) -> None:
        payload = self.extract_payload(event)
        symbol, exchange = self.extract_symbol_exchange(payload)
        if not symbol:
            return

        lock = await self.acquire_symbol_lock(symbol, exchange)
        if lock is None:
            return

        try:
            state = self.get_state(symbol, exchange)
            extreme_event = self._normalize_extreme_payload(payload)
            self.attach_extreme(state, extreme_event)
            self._expire_state_if_needed(state)

            if self.is_in_cooldown(state) or not state.is_active():
                return

            event_time = self._extract_event_time_from_normalized(extreme_event)
            if self.is_stale_event(event_time):
                return

            if self._should_invalidate_by_extreme(state, extreme_event):
                self.set_invalidated(
                    state,
                    reason="opposite_extreme_invalidated_divergence_setup",
                    cooldown=True,
                    metadata={"invalidation_source": "extreme"},
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={"trigger": "extreme", "correlation_id": event.correlation_id},
                )
                return

            if self._can_confirm_by_extreme(state, extreme_event):
                self.set_confirmed(
                    state,
                    score=self._compute_confirmation_score_from_extreme(state, extreme_event),
                    confidence=self._compute_confirmation_confidence_from_extreme(state, extreme_event),
                    reason="extreme_context_confirmed_divergence_setup",
                    tags=[self.config.tag_confirmed_by_extreme, self.config.tag_extreme],
                    event_time=event_time,
                    metadata={"confirmation_source": "extreme"},
                )
                await self.emit_confirmed(
                    state,
                    extra_payload={"trigger": "extreme", "correlation_id": event.correlation_id},
                )

        except Exception:
            self.logger.exception(
                "Failed to process funding extreme event | strategy=%s symbol=%s exchange=%s",
                self.strategy_name,
                symbol,
                exchange,
            )
        finally:
            self.release_symbol_lock(lock)

    # ------------------------------------------------------------------
    # Setup creation
    # ------------------------------------------------------------------

    def _can_create_setup_from_divergence(self, state: FundingStrategyState, divergence_event: Any) -> bool:
        divergence_confidence = self._to_float(self._get_value(divergence_event, "confidence"), default=0.0)
        if divergence_confidence < self.config.min_divergence_confidence:
            return False

        target_direction = self._derive_direction_from_divergence(divergence_event)
        if target_direction == FundingStrategyDirection.NEUTRAL:
            return False

        regime = state.last_regime
        if self.config.require_non_neutral_regime:
            if regime is None:
                return False
            regime_name = self._enum_str(self._get_value(regime, "regime"))
            regime_confidence = self._to_float(self._get_value(regime, "confidence"), default=0.0)
            if regime_confidence < self.config.min_regime_confidence:
                return False
            if regime_name in {FundingRegime.NEUTRAL.value, FundingRegime.UNKNOWN.value}:
                return False

        pressure = state.last_pressure
        if pressure is None:
            return not self.config.require_pressure_present

        pressure_score = self._to_float(self._get_value(pressure, "pressure_score"), default=0.0)
        if pressure_score < self.config.min_pressure_score:
            return False

        if self.config.require_pressure_alignment:
            direction = self._enum_str(self._get_value(pressure, "direction"))
            if target_direction == FundingStrategyDirection.LONG:
                return direction in {FundingPressureDirection.SHORT.value, FundingPressureDirection.NEUTRAL.value}
            if target_direction == FundingStrategyDirection.SHORT:
                return direction in {FundingPressureDirection.LONG.value, FundingPressureDirection.NEUTRAL.value}

        return True

    def _build_setup_from_divergence(self, state: FundingStrategyState, divergence_event: Any) -> dict[str, Any] | None:
        divergence_type = self._enum_str(self._get_value(divergence_event, "divergence_type"))
        target_direction = self._derive_direction_from_divergence(divergence_event)
        if target_direction == FundingStrategyDirection.NEUTRAL:
            return None

        divergence_confidence = self._to_float(self._get_value(divergence_event, "confidence"), default=0.0)
        regime = state.last_regime
        pressure = state.last_pressure

        regime_confidence = self._to_float(self._get_value(regime, "confidence"), default=0.0)
        bias = self._enum_str(self._get_value(regime, "bias"))
        pressure_score = self._to_float(self._get_value(pressure, "pressure_score"), default=0.0)
        pressure_direction = self._enum_str(self._get_value(pressure, "direction"))

        directional_alignment_bonus = self._calc_directional_alignment_bonus(
            target_direction=target_direction,
            regime_bias=bias,
            pressure_direction=pressure_direction,
        )
        extreme_alignment_bonus = self._calc_extreme_alignment_bonus(state, target_direction)
        signal_alignment_bonus = self._calc_signal_alignment_bonus(state, target_direction)

        score = self._compute_setup_score(
            divergence_confidence=divergence_confidence,
            pressure_score=pressure_score,
            regime_confidence=regime_confidence,
            directional_alignment_bonus=directional_alignment_bonus,
            extreme_alignment_bonus=extreme_alignment_bonus,
            signal_alignment_bonus=signal_alignment_bonus,
        )
        confidence = self._compute_setup_confidence(
            score=score,
            divergence_confidence=divergence_confidence,
            regime_confidence=regime_confidence,
        )

        setup_type = self.config.bullish_setup_type if target_direction == FundingStrategyDirection.LONG else self.config.bearish_setup_type
        side_label = "bullish_funding_divergence_setup" if target_direction == FundingStrategyDirection.LONG else "bearish_funding_divergence_setup"

        return {
            "direction": target_direction,
            "setup_type": setup_type,
            "score": score,
            "confidence": confidence,
            "reason": side_label,
            "reasons": [side_label, f"divergence_type:{divergence_type}"],
            "tags": self._build_setup_tags(state),
            "metadata": {
                "funding_context": self._build_divergence_context(
                    divergence_event=divergence_event,
                    regime=regime,
                    pressure=pressure,
                ),
                "extreme_alignment_bonus": extreme_alignment_bonus,
                "signal_alignment_bonus": signal_alignment_bonus,
            },
        }

    def _apply_setup_candidate(self, *, state: FundingStrategyState, setup_candidate: dict[str, Any], event_time: Any) -> None:
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
    # Confirmation guards
    # ------------------------------------------------------------------

    def _can_confirm_by_flip(self, state: FundingStrategyState, flip_event: Any) -> bool:
        if not self.config.allow_flip_confirmation:
            return False
        if state.status not in {FundingSetupStatus.SETUP_DETECTED, FundingSetupStatus.CONFIRMED}:
            return False
        flip_type = self._enum_str(self._get_value(flip_event, "flip_type"))
        return (
            state.direction == FundingStrategyDirection.LONG
            and flip_type == FundingFlipType.NEGATIVE_TO_POSITIVE.value
        ) or (
            state.direction == FundingStrategyDirection.SHORT
            and flip_type == FundingFlipType.POSITIVE_TO_NEGATIVE.value
        )

    def _can_confirm_by_repeat_divergence(self, state: FundingStrategyState, previous_divergence: Any | None, current_divergence: Any) -> bool:
        if not self.config.allow_repeat_divergence_confirmation:
            return False
        if state.status != FundingSetupStatus.SETUP_DETECTED or previous_divergence is None:
            return False
        prev_direction = self._derive_direction_from_divergence(previous_divergence)
        curr_direction = self._derive_direction_from_divergence(current_divergence)
        if prev_direction == FundingStrategyDirection.NEUTRAL:
            return False
        if curr_direction != prev_direction or curr_direction != state.direction:
            return False
        curr_confidence = self._to_float(self._get_value(current_divergence, "confidence"), default=0.0)
        return curr_confidence >= self.config.min_divergence_confidence

    def _can_confirm_by_pressure_release(self, state: FundingStrategyState, previous_pressure: Any | None, current_pressure: Any) -> bool:
        if not self.config.allow_pressure_release_confirmation:
            return False
        if state.status != FundingSetupStatus.SETUP_DETECTED or previous_pressure is None:
            return False
        prev_score = self._to_float(self._get_value(previous_pressure, "pressure_score"), default=0.0)
        curr_score = self._to_float(self._get_value(current_pressure, "pressure_score"), default=0.0)
        if prev_score <= curr_score:
            return False
        if (prev_score - curr_score) < self.config.pressure_release_min_score_drop:
            return False
        prev_level = self._enum_str(self._get_value(previous_pressure, "level"))
        curr_level = self._enum_str(self._get_value(current_pressure, "level"))
        return self._has_pressure_level_dropped_enough(prev_level, curr_level)

    def _can_confirm_by_regime_shift(self, state: FundingStrategyState, previous_regime: Any | None, current_regime: Any) -> bool:
        if not self.config.allow_regime_shift_confirmation:
            return False
        if state.status != FundingSetupStatus.SETUP_DETECTED or previous_regime is None:
            return False
        curr_bias = self._enum_str(self._get_value(current_regime, "bias"))
        if state.direction == FundingStrategyDirection.LONG:
            return curr_bias in {FundingBias.SHORT_BIAS.value, FundingBias.OVERCROWDED_SHORTS.value, FundingBias.SQUEEZE_RISK_SHORTS.value}
        if state.direction == FundingStrategyDirection.SHORT:
            return curr_bias in {FundingBias.LONG_BIAS.value, FundingBias.OVERCROWDED_LONGS.value, FundingBias.SQUEEZE_RISK_LONGS.value}
        return False

    def _can_confirm_by_extreme(self, state: FundingStrategyState, extreme_event: Any) -> bool:
        if not self.config.allow_extreme_confirmation:
            return False
        if state.status != FundingSetupStatus.SETUP_DETECTED:
            return False
        severity = self._to_float(self._get_value(extreme_event, "severity"), default=0.0)
        if severity < self.config.min_extreme_severity:
            return False
        return self._extreme_direction(extreme_event) == state.direction

    def _can_confirm_by_signal(self, state: FundingStrategyState, signal: Any) -> bool:
        if not self.config.allow_signal_confirmation:
            return False
        if state.status != FundingSetupStatus.SETUP_DETECTED:
            return False
        confidence = self._to_float(self._get_value(signal, "confidence"), default=0.0)
        score = self._to_float(self._get_value(signal, "score"), default=0.0)
        if confidence < self.config.min_signal_confidence or abs(score) < self.config.min_signal_abs_score:
            return False
        return self._signal_direction(signal) == state.direction

    # ------------------------------------------------------------------
    # Invalidation guards
    # ------------------------------------------------------------------

    def _should_invalidate_by_flip(self, state: FundingStrategyState, flip_event: Any) -> bool:
        if not self.config.invalidate_on_opposite_flip:
            return False
        flip_type = self._enum_str(self._get_value(flip_event, "flip_type"))
        return (
            state.direction == FundingStrategyDirection.LONG
            and flip_type == FundingFlipType.POSITIVE_TO_NEGATIVE.value
        ) or (
            state.direction == FundingStrategyDirection.SHORT
            and flip_type == FundingFlipType.NEGATIVE_TO_POSITIVE.value
        )

    def _should_invalidate_by_pressure(self, state: FundingStrategyState, pressure_state: Any) -> bool:
        if not self.config.invalidate_on_pressure_breakdown:
            return False
        pressure_score = self._to_float(self._get_value(pressure_state, "pressure_score"), default=0.0)
        level = self._enum_str(self._get_value(pressure_state, "level"))
        if level in {FundingPressureLevel.UNKNOWN.value, FundingPressureLevel.LOW.value} and pressure_score < (self.config.min_pressure_score * _PRESSURE_BREAKDOWN_THRESHOLD_RATIO):
            return True
        direction = self._enum_str(self._get_value(pressure_state, "direction"))
        if state.direction == FundingStrategyDirection.LONG:
            return direction == FundingPressureDirection.LONG.value
        if state.direction == FundingStrategyDirection.SHORT:
            return direction == FundingPressureDirection.SHORT.value
        return False

    def _should_invalidate_by_regime(self, state: FundingStrategyState, regime_state: Any) -> bool:
        if not self.config.invalidate_on_regime_conflict:
            return False
        bias = self._enum_str(self._get_value(regime_state, "bias"))
        regime_name = self._enum_str(self._get_value(regime_state, "regime"))
        if regime_name == FundingRegime.UNKNOWN.value:
            return True
        if state.direction == FundingStrategyDirection.LONG:
            return bias in {FundingBias.LONG_BIAS.value, FundingBias.OVERCROWDED_LONGS.value, FundingBias.SQUEEZE_RISK_LONGS.value}
        if state.direction == FundingStrategyDirection.SHORT:
            return bias in {FundingBias.SHORT_BIAS.value, FundingBias.OVERCROWDED_SHORTS.value, FundingBias.SQUEEZE_RISK_SHORTS.value}
        return False

    def _is_opposite_divergence_for_state(
            self,
            state: FundingStrategyState,
            divergence_event: Any,
    ) -> bool:
        if not self.config.invalidate_on_opposite_divergence:
            return False
        return self._is_opposite_divergence_direction_for_state(state, divergence_event)

    def _should_invalidate_by_signal(self, state: FundingStrategyState, signal: Any) -> bool:
        if not self.config.invalidate_on_opposite_signal:
            return False
        direction = self._signal_direction(signal)
        return direction != FundingStrategyDirection.NEUTRAL and direction != state.direction

    def _should_invalidate_by_extreme(self, state: FundingStrategyState, extreme_event: Any) -> bool:
        if not self.config.invalidate_on_opposite_extreme:
            return False
        direction = self._extreme_direction(extreme_event)
        return direction != FundingStrategyDirection.NEUTRAL and direction != state.direction

    # ------------------------------------------------------------------
    # Score / confidence computation
    # ------------------------------------------------------------------

    def _compute_setup_score(
        self,
        *,
        divergence_confidence: float,
        pressure_score: float,
        regime_confidence: float,
        directional_alignment_bonus: float,
        extreme_alignment_bonus: float = 0.0,
        signal_alignment_bonus: float = 0.0,
    ) -> float:
        cfg = self.config
        return self._clip_score(
            self._weighted_average(
                [
                    (divergence_confidence, cfg.score_weight_divergence),
                    (pressure_score, cfg.score_weight_pressure),
                    (regime_confidence, cfg.score_weight_regime),
                    (directional_alignment_bonus, cfg.score_weight_alignment),
                    (extreme_alignment_bonus, cfg.score_weight_extreme_alignment),
                    (signal_alignment_bonus, cfg.score_weight_signal_alignment),
                ]
            )
        )

    def _compute_setup_confidence(self, *, score: float, divergence_confidence: float, regime_confidence: float) -> float:
        return self._clip_score(self._average_scores(score, divergence_confidence, regime_confidence))

    def _boost_metric(self, base: float, signal: float, bonus: float) -> float:
        return self._clip_score(self._average_scores(base, signal, min(1.0, base + bonus)))

    def _compute_confirmation_score_from_flip(self, state: FundingStrategyState, flip_event: Any) -> float:
        flip_confidence = self._to_float(self._get_value(flip_event, "confidence"), default=0.0)
        return self._clip_score(self._average_scores(state.score, flip_confidence, _FLIP_CONFIRMATION_PERFECT_WEIGHT))

    def _compute_confirmation_confidence_from_flip(self, state: FundingStrategyState, flip_event: Any) -> float:
        flip_confidence = self._to_float(self._get_value(flip_event, "confidence"), default=0.0)
        return self._clip_score(self._average_scores(state.confidence, flip_confidence, _FLIP_CONFIRMATION_PERFECT_WEIGHT))

    def _compute_confirmation_score_from_repeat_divergence(self, state: FundingStrategyState, divergence_event: Any) -> float:
        divergence_confidence = self._to_float(self._get_value(divergence_event, "confidence"), default=0.0)
        return self._boost_metric(base=state.score, signal=divergence_confidence, bonus=self.config.repeat_divergence_confirmation_bonus)

    def _compute_confirmation_confidence_from_repeat_divergence(self, state: FundingStrategyState, divergence_event: Any) -> float:
        divergence_confidence = self._to_float(self._get_value(divergence_event, "confidence"), default=0.0)
        return self._boost_metric(base=state.confidence, signal=divergence_confidence, bonus=self.config.repeat_divergence_confirmation_bonus)

    def _compute_confirmation_score_from_pressure_release(self, state: FundingStrategyState, pressure_state: Any) -> float:
        curr_pressure_score = self._to_float(self._get_value(pressure_state, "pressure_score"), default=0.0)
        return self._clip_score(self._average_scores(state.score, 1.0 - min(curr_pressure_score, 1.0), _PRESSURE_RELEASE_TARGET_CONFIDENCE))

    def _compute_confirmation_confidence_from_pressure_release(self, state: FundingStrategyState, pressure_state: Any) -> float:
        mean_reversion_probability = self._to_float(self._get_value(pressure_state, "mean_reversion_probability"), default=0.0)
        squeeze_probability = self._to_float(self._get_value(pressure_state, "squeeze_probability"), default=0.0)
        return self._clip_score(self._average_scores(state.confidence, mean_reversion_probability, squeeze_probability))

    def _compute_confirmation_score_from_regime_shift(self, state: FundingStrategyState, regime_state: Any) -> float:
        regime_confidence = self._to_float(self._get_value(regime_state, "confidence"), default=0.0)
        return self._boost_metric(base=state.score, signal=regime_confidence, bonus=self.config.regime_shift_confirmation_bonus)

    def _compute_confirmation_confidence_from_regime_shift(self, state: FundingStrategyState, regime_state: Any) -> float:
        regime_confidence = self._to_float(self._get_value(regime_state, "confidence"), default=0.0)
        return self._boost_metric(base=state.confidence, signal=regime_confidence, bonus=self.config.regime_shift_confirmation_bonus)

    def _compute_confirmation_score_from_extreme(self, state: FundingStrategyState, extreme_event: Any) -> float:
        severity = self._to_float(self._get_value(extreme_event, "severity"), default=0.0)
        return self._boost_metric(base=state.score, signal=severity, bonus=self.config.extreme_confirmation_bonus)

    def _compute_confirmation_confidence_from_extreme(self, state: FundingStrategyState, extreme_event: Any) -> float:
        severity = self._to_float(self._get_value(extreme_event, "severity"), default=0.0)
        return self._boost_metric(base=state.confidence, signal=severity, bonus=self.config.extreme_confirmation_bonus)

    def _compute_confirmation_score_from_signal(self, state: FundingStrategyState, signal: Any) -> float:
        signal_score = abs(self._to_float(self._get_value(signal, "score"), default=0.0))
        return self._weighted_average([(state.score, 1.0 - _SIGNAL_CONFIRMATION_WEIGHT), (signal_score, _SIGNAL_CONFIRMATION_WEIGHT)])

    def _compute_confirmation_confidence_from_signal(self, state: FundingStrategyState, signal: Any) -> float:
        signal_confidence = self._to_float(self._get_value(signal, "confidence"), default=0.0)
        return self._weighted_average([(state.confidence, 0.45), (signal_confidence, 0.55)])

    # ------------------------------------------------------------------
    # Atomic update evaluation
    # ------------------------------------------------------------------

    async def _evaluate_active_state_after_atomic_update(self, *, state: FundingStrategyState, correlation_id: str | None) -> None:
        if state.last_signal is not None and self._should_invalidate_by_signal(state, state.last_signal):
            self.set_invalidated(state, reason="atomic_update_opposite_signal_invalidated_divergence_setup", cooldown=True, metadata={"invalidation_source": "funding_updated.signal"})
            await self.emit_invalidated(state, extra_payload={"trigger": "funding_updated", "correlation_id": correlation_id})
            return

        if state.last_divergence is not None and self._is_opposite_divergence_for_state(state, state.last_divergence):
            self.set_invalidated(state, reason="atomic_update_opposite_divergence_invalidated_divergence_setup", cooldown=True, metadata={"invalidation_source": "funding_updated.divergence"})
            await self.emit_invalidated(state, extra_payload={"trigger": "funding_updated", "correlation_id": correlation_id})
            return

        if state.status != FundingSetupStatus.SETUP_DETECTED:
            return

        if state.last_signal is not None and self._can_confirm_by_signal(state, state.last_signal):
            self.set_confirmed(
                state,
                score=self._compute_confirmation_score_from_signal(state, state.last_signal),
                confidence=self._compute_confirmation_confidence_from_signal(state, state.last_signal),
                reason="atomic_update_signal_confirmed_divergence_setup",
                tags=[self.config.tag_confirmed_by_signal, self.config.tag_atomic_context],
                event_time=self._extract_event_time_from_normalized(state.last_signal),
                metadata={"confirmation_source": "funding_updated.signal"},
            )
            await self.emit_confirmed(state, extra_payload={"trigger": "funding_updated", "correlation_id": correlation_id})
            return

        if state.last_extreme is not None and self._can_confirm_by_extreme(state, state.last_extreme):
            self.set_confirmed(
                state,
                score=self._compute_confirmation_score_from_extreme(state, state.last_extreme),
                confidence=self._compute_confirmation_confidence_from_extreme(state, state.last_extreme),
                reason="atomic_update_extreme_confirmed_divergence_setup",
                tags=[self.config.tag_confirmed_by_extreme, self.config.tag_atomic_context],
                event_time=self._extract_event_time_from_normalized(state.last_extreme),
                metadata={"confirmation_source": "funding_updated.extreme"},
            )
            await self.emit_confirmed(state, extra_payload={"trigger": "funding_updated", "correlation_id": correlation_id})

    # ------------------------------------------------------------------
    # Direction & alignment helpers
    # ------------------------------------------------------------------

    def _derive_direction_from_divergence(self, divergence_event: Any) -> FundingStrategyDirection:
        divergence_type = self._enum_str(self._get_value(divergence_event, "divergence_type"))
        bullish_types = {
            FundingDivergenceType.PRICE_UP_FUNDING_DOWN.value,
            FundingDivergenceType.OI_UP_FUNDING_DOWN.value,
            FundingDivergenceType.CVD_UP_FUNDING_DOWN.value,
            FundingDivergenceType.LIQUIDATIONS_SHORTS_WITH_NEGATIVE_FUNDING.value,
        }
        bearish_types = {
            FundingDivergenceType.PRICE_DOWN_FUNDING_UP.value,
            FundingDivergenceType.CVD_DOWN_FUNDING_UP.value,
            FundingDivergenceType.LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING.value,
            FundingDivergenceType.OI_UP_FUNDING_UP_PRICE_STALLED.value,
        }
        if divergence_type in bullish_types:
            return FundingStrategyDirection.LONG
        if divergence_type in bearish_types:
            return FundingStrategyDirection.SHORT
        return FundingStrategyDirection.NEUTRAL

    def _extreme_direction(self, extreme_event: Any) -> FundingStrategyDirection:
        extreme_type = self._enum_str(self._get_value(extreme_event, "extreme_type"))
        positive = {
            FundingExtremeType.LOCAL_HIGH.value,
            FundingExtremeType.GLOBAL_HIGH.value,
            FundingExtremeType.ZSCORE_HIGH.value,
            FundingExtremeType.PERCENTILE_HIGH.value,
        }
        negative = {
            FundingExtremeType.LOCAL_LOW.value,
            FundingExtremeType.GLOBAL_LOW.value,
            FundingExtremeType.ZSCORE_LOW.value,
            FundingExtremeType.PERCENTILE_LOW.value,
        }
        if extreme_type in positive:
            return FundingStrategyDirection.SHORT
        if extreme_type in negative:
            return FundingStrategyDirection.LONG
        return FundingStrategyDirection.NEUTRAL

    def _signal_direction(self, signal: Any) -> FundingStrategyDirection:
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
        if signal_type in directional_types and bias in {FundingBias.SHORT_BIAS.value, FundingBias.OVERCROWDED_SHORTS.value, FundingBias.SQUEEZE_RISK_SHORTS.value}:
            return FundingStrategyDirection.LONG
        if signal_type in directional_types and bias in {FundingBias.LONG_BIAS.value, FundingBias.OVERCROWDED_LONGS.value, FundingBias.SQUEEZE_RISK_LONGS.value}:
            return FundingStrategyDirection.SHORT

        return FundingStrategyDirection.NEUTRAL

    def _calc_directional_alignment_bonus(self, *, target_direction: FundingStrategyDirection, regime_bias: str | None, pressure_direction: str | None) -> float:
        bonus = _ALIGNMENT_BONUS_BASE
        if target_direction == FundingStrategyDirection.LONG:
            if regime_bias in {FundingBias.SHORT_BIAS.value, FundingBias.OVERCROWDED_SHORTS.value, FundingBias.SQUEEZE_RISK_SHORTS.value}:
                bonus += _ALIGNMENT_BONUS_PER_DIMENSION
            if pressure_direction in {FundingPressureDirection.SHORT.value, FundingPressureDirection.NEUTRAL.value}:
                bonus += _ALIGNMENT_BONUS_PER_DIMENSION
        elif target_direction == FundingStrategyDirection.SHORT:
            if regime_bias in {FundingBias.LONG_BIAS.value, FundingBias.OVERCROWDED_LONGS.value, FundingBias.SQUEEZE_RISK_LONGS.value}:
                bonus += _ALIGNMENT_BONUS_PER_DIMENSION
            if pressure_direction in {FundingPressureDirection.LONG.value, FundingPressureDirection.NEUTRAL.value}:
                bonus += _ALIGNMENT_BONUS_PER_DIMENSION
        return self._clip_score(bonus)

    def _calc_extreme_alignment_bonus(self, state: FundingStrategyState, target_direction: FundingStrategyDirection) -> float:
        if state.last_extreme is None:
            return 0.0
        direction = self._extreme_direction(state.last_extreme)
        severity = self._to_float(self._get_value(state.last_extreme, "severity"), default=0.0)
        return severity if direction == target_direction else 0.0

    def _calc_signal_alignment_bonus(self, state: FundingStrategyState, target_direction: FundingStrategyDirection) -> float:
        if state.last_signal is None:
            return 0.0
        direction = self._signal_direction(state.last_signal)
        confidence = self._to_float(self._get_value(state.last_signal, "confidence"), default=0.0)
        return confidence if direction == target_direction else 0.0

    # ------------------------------------------------------------------
    # Emit hooks
    # ------------------------------------------------------------------

    def on_before_setup_emit(self, state: FundingStrategyState, payload: dict[str, Any]) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "divergence"
        payload["signal_class"] = "directional_dislocation"
        return payload

    def on_before_confirmation_emit(self, state: FundingStrategyState, payload: dict[str, Any]) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "divergence"
        payload["signal_class"] = "directional_dislocation"
        payload["is_tradeable"] = True
        return payload

    def on_before_invalidation_emit(self, state: FundingStrategyState, payload: dict[str, Any]) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "divergence"
        return payload

    def on_before_expiration_emit(self, state: FundingStrategyState, payload: dict[str, Any]) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "divergence"
        return payload

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    def _build_setup_tags(self, state: FundingStrategyState) -> list[str]:
        tags = [self.config.tag_divergence, self.config.tag_dislocation, self.config.tag_reversal]
        if state.last_extreme is not None:
            tags.append(self.config.tag_extreme)
        if state.last_signal is not None:
            tags.append(self.config.tag_signal)
        return tags

    def _build_divergence_context(self, *, divergence_event: Any, regime: Any | None, pressure: Any | None) -> dict[str, Any]:
        return {
            "divergence": {
                "divergence_type": self._enum_str(self._get_value(divergence_event, "divergence_type")),
                "confidence": self._to_float(self._get_value(divergence_event, "confidence"), default=0.0),
                "funding_rate": self._get_value(divergence_event, "funding_rate"),
                "price_change_pct": self._get_value(divergence_event, "price_change_pct"),
                "oi_change_pct": self._get_value(divergence_event, "oi_change_pct"),
                "cvd_change": self._get_value(divergence_event, "cvd_change"),
                "long_liquidations": self._get_value(divergence_event, "long_liquidations"),
                "short_liquidations": self._get_value(divergence_event, "short_liquidations"),
            },
            "regime": {
                "regime": self._enum_str(self._get_value(regime, "regime")),
                "bias": self._enum_str(self._get_value(regime, "bias")),
                "confidence": self._to_float(self._get_value(regime, "confidence"), default=0.0),
            },
            "pressure": {
                "direction": self._enum_str(self._get_value(pressure, "direction")),
                "level": self._enum_str(self._get_value(pressure, "level")),
                "pressure_score": self._to_float(self._get_value(pressure, "pressure_score"), default=0.0),
            },
        }
