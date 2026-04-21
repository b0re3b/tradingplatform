from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.logger import get_logger

from analytics.funding.enums import (
    FundingBias,
    FundingDivergenceType,
    FundingFlipType,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
)

from .base import (
    BaseFundingStrategy,
    BaseFundingStrategyConfig,
    FundingSetupStatus,
    FundingStrategyDirection,
    FundingStrategyState,
)


@dataclass(slots=True)
class FundingDivergenceStrategyConfig(BaseFundingStrategyConfig):
    """
    Конфіг funding divergence strategy.

    Ідея:
    - funding / positioning не узгоджується з price / OI / CVD / liquidations
    - strategy створює directional setup
    - потім чекає confirm через flip / repeated divergence / pressure release / regime shift
    """

    strategy_namespace: str = "strategy.funding.divergence"
    source_name: str = "funding_divergence_strategy"

    regime_event_name: str = "analytics.funding.regime"
    pressure_event_name: str = "analytics.funding.pressure"
    divergence_event_name: str = "analytics.funding.divergence"
    flip_event_name: str = "analytics.funding.flip"

    min_divergence_confidence: float = 0.50
    min_pressure_score: float = 0.35
    min_regime_confidence: float = 0.10

    require_non_neutral_regime: bool = True
    require_pressure_alignment: bool = False

    allow_flip_confirmation: bool = True
    allow_repeat_divergence_confirmation: bool = True
    allow_pressure_release_confirmation: bool = True
    allow_regime_shift_confirmation: bool = True

    repeat_divergence_confirmation_bonus: float = 0.12
    regime_shift_confirmation_bonus: float = 0.10

    pressure_release_min_score_drop: float = 0.08
    confirm_on_pressure_drop_levels: int = 1

    invalidate_on_opposite_flip: bool = True
    invalidate_on_opposite_divergence: bool = True
    invalidate_on_pressure_breakdown: bool = True
    invalidate_on_regime_conflict: bool = True

    bullish_setup_type: str = "funding_bullish_divergence"
    bearish_setup_type: str = "funding_bearish_divergence"

    tag_divergence: str = "funding_divergence"
    tag_dislocation: str = "dislocation"
    tag_reversal: str = "reversal"
    tag_confirmed_by_flip: str = "confirmed_by_flip"
    tag_confirmed_by_repeat: str = "confirmed_by_repeat_divergence"
    tag_confirmed_by_release: str = "confirmed_by_pressure_release"
    tag_confirmed_by_regime: str = "confirmed_by_regime_shift"


class FundingDivergenceStrategy(BaseFundingStrategy):
    """
    Стратегія на divergence між funding та іншими компонентами ринку.

    Потік:
    1. На divergence event створює setup
    2. Використовує pressure/regime як фільтри якості
    3. Confirm через flip / повторну divergence / pressure release / regime shift
    4. Invalidate при конфліктному контексті
    """

    def __init__(
        self,
        event_bus: Any,
        config: FundingDivergenceStrategyConfig | None = None,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            config=config or FundingDivergenceStrategyConfig(),
        )
        self.config: FundingDivergenceStrategyConfig = (
            config or FundingDivergenceStrategyConfig()
        )
        self.logger = get_logger(__name__)

    @property
    def strategy_name(self) -> str:
        return "funding_divergence"

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_subscriptions(self) -> None:
        self.event_bus.subscribe(self.config.regime_event_name, self.on_regime)
        self.event_bus.subscribe(self.config.pressure_event_name, self.on_pressure)
        self.event_bus.subscribe(self.config.divergence_event_name, self.on_divergence)
        self.event_bus.subscribe(self.config.flip_event_name, self.on_flip)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_regime(self, event: Any) -> None:
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

            if self.is_in_cooldown(state):
                return

            if not state.is_active():
                return

            if self._should_invalidate_by_regime(state, regime_state):
                self.set_invalidated(
                    state,
                    reason="regime_context_invalidated_divergence_setup",
                    cooldown=True,
                    metadata={
                        "invalidation_source": "regime",
                    },
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={"trigger": "regime"},
                )
                return

            if self._can_confirm_by_regime_shift(
                state=state,
                previous_regime=previous_regime,
                current_regime=regime_state,
            ):
                new_score = self._compute_confirmation_score_from_regime_shift(
                    state=state,
                    regime_state=regime_state,
                )
                new_confidence = self._compute_confirmation_confidence_from_regime_shift(
                    state=state,
                    regime_state=regime_state,
                )

                self.set_confirmed(
                    state,
                    score=new_score,
                    confidence=new_confidence,
                    reason="regime_shift_confirmed_divergence_setup",
                    tags=[self.config.tag_confirmed_by_regime],
                    event_time=self._extract_event_time_from_normalized(regime_state),
                    metadata={
                        "confirmation_source": "regime_shift",
                    },
                )

                await self.emit_confirmed(
                    state,
                    extra_payload={"trigger": "regime_shift"},
                )

        except Exception:
            self.logger.exception(
                "Failed to process regime event in %s: symbol=%s exchange=%s",
                self.strategy_name,
                symbol,
                exchange,
            )
        finally:
            self.release_symbol_lock(lock)

    async def on_pressure(self, event: Any) -> None:
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

            if self.is_in_cooldown(state):
                return

            if not state.is_active():
                return

            if self._should_invalidate_by_pressure(state, pressure_state):
                self.set_invalidated(
                    state,
                    reason="pressure_context_invalidated_divergence_setup",
                    cooldown=True,
                    metadata={
                        "invalidation_source": "pressure",
                    },
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={"trigger": "pressure"},
                )
                return

            if self._can_confirm_by_pressure_release(
                state=state,
                previous_pressure=previous_pressure,
                current_pressure=pressure_state,
            ):
                new_score = self._compute_confirmation_score_from_pressure_release(
                    state=state,
                    pressure_state=pressure_state,
                )
                new_confidence = (
                    self._compute_confirmation_confidence_from_pressure_release(
                        state=state,
                        pressure_state=pressure_state,
                    )
                )

                self.set_confirmed(
                    state,
                    score=new_score,
                    confidence=new_confidence,
                    reason="pressure_release_confirmed_divergence_setup",
                    tags=[self.config.tag_confirmed_by_release],
                    event_time=self._extract_event_time_from_normalized(pressure_state),
                    metadata={
                        "confirmation_source": "pressure_release",
                    },
                )

                await self.emit_confirmed(
                    state,
                    extra_payload={"trigger": "pressure_release"},
                )

        except Exception:
            self.logger.exception(
                "Failed to process pressure event in %s: symbol=%s exchange=%s",
                self.strategy_name,
                symbol,
                exchange,
            )
        finally:
            self.release_symbol_lock(lock)

    async def on_divergence(self, event: Any) -> None:
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
                    "Stale divergence event ignored: symbol=%s exchange=%s",
                    symbol,
                    exchange,
                )
                return

            if (
                state.is_active()
                and self.config.invalidate_on_opposite_divergence
                and self._is_opposite_divergence_for_state(state, divergence_event)
            ):
                self.set_invalidated(
                    state,
                    reason="opposite_divergence_invalidated_setup",
                    cooldown=True,
                    metadata={
                        "invalidation_source": "divergence",
                    },
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={"trigger": "divergence"},
                )
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

                    await self.emit_setup(
                        state,
                        extra_payload={"trigger": "divergence"},
                    )
                    return

            if self._can_confirm_by_repeat_divergence(
                state=state,
                previous_divergence=previous_divergence,
                current_divergence=divergence_event,
            ):
                new_score = self._compute_confirmation_score_from_repeat_divergence(
                    state=state,
                    divergence_event=divergence_event,
                )
                new_confidence = (
                    self._compute_confirmation_confidence_from_repeat_divergence(
                        state=state,
                        divergence_event=divergence_event,
                    )
                )

                self.set_confirmed(
                    state,
                    score=new_score,
                    confidence=new_confidence,
                    reason="repeat_divergence_confirmed_setup",
                    tags=[self.config.tag_confirmed_by_repeat],
                    event_time=event_time,
                    metadata={
                        "confirmation_source": "repeat_divergence",
                    },
                )

                await self.emit_confirmed(
                    state,
                    extra_payload={"trigger": "repeat_divergence"},
                )

        except Exception:
            self.logger.exception(
                "Failed to process divergence event in %s: symbol=%s exchange=%s",
                self.strategy_name,
                symbol,
                exchange,
            )
        finally:
            self.release_symbol_lock(lock)

    async def on_flip(self, event: Any) -> None:
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

            if self.is_in_cooldown(state):
                return

            if not state.is_active():
                return

            if self._should_invalidate_by_flip(state, flip_event):
                self.set_invalidated(
                    state,
                    reason="opposite_flip_invalidated_divergence_setup",
                    cooldown=True,
                    metadata={
                        "invalidation_source": "flip",
                    },
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={"trigger": "flip"},
                )
                return

            if self._can_confirm_by_flip(state, flip_event):
                new_score = self._compute_confirmation_score_from_flip(
                    state=state,
                    flip_event=flip_event,
                )
                new_confidence = self._compute_confirmation_confidence_from_flip(
                    state=state,
                    flip_event=flip_event,
                )

                self.set_confirmed(
                    state,
                    score=new_score,
                    confidence=new_confidence,
                    reason="flip_confirmed_divergence_setup",
                    tags=[self.config.tag_confirmed_by_flip],
                    event_time=self._extract_event_time_from_normalized(flip_event),
                    metadata={
                        "confirmation_source": "flip",
                    },
                )

                await self.emit_confirmed(
                    state,
                    extra_payload={"trigger": "flip"},
                )

        except Exception:
            self.logger.exception(
                "Failed to process flip event in %s: symbol=%s exchange=%s",
                self.strategy_name,
                symbol,
                exchange,
            )
        finally:
            self.release_symbol_lock(lock)

    # ------------------------------------------------------------------
    # Setup creation
    # ------------------------------------------------------------------

    def _can_create_setup_from_divergence(
        self,
        state: FundingStrategyState,
        divergence_event: Any,
    ) -> bool:
        divergence_confidence = self._to_float(
            self._get_value(divergence_event, "confidence"),
            default=0.0,
        )
        if divergence_confidence < self.config.min_divergence_confidence:
            return False

        regime = state.last_regime
        if self.config.require_non_neutral_regime:
            if regime is None:
                return False

            regime_name = self._enum_str(self._get_value(regime, "regime"))
            regime_confidence = self._to_float(
                self._get_value(regime, "confidence"),
                default=0.0,
            )

            if regime_confidence < self.config.min_regime_confidence:
                return False

            if regime_name in {
                FundingRegime.NEUTRAL.value,
                FundingRegime.UNKNOWN.value,
            }:
                return False

        pressure = state.last_pressure
        if pressure is not None:
            pressure_score = self._to_float(
                self._get_value(pressure, "pressure_score"),
                default=0.0,
            )
            if pressure_score < self.config.min_pressure_score:
                return False

            if self.config.require_pressure_alignment:
                direction = self._enum_str(self._get_value(pressure, "direction"))
                target_direction = self._derive_direction_from_divergence(divergence_event)
                if target_direction == FundingStrategyDirection.LONG:
                    if direction not in {
                        FundingPressureDirection.SHORT.value,
                        FundingPressureDirection.NEUTRAL.value,
                    }:
                        return False
                elif target_direction == FundingStrategyDirection.SHORT:
                    if direction not in {
                        FundingPressureDirection.LONG.value,
                        FundingPressureDirection.NEUTRAL.value,
                    }:
                        return False

        return True

    def _build_setup_from_divergence(
        self,
        state: FundingStrategyState,
        divergence_event: Any,
    ) -> dict[str, Any] | None:
        divergence_type = self._enum_str(
            self._get_value(divergence_event, "divergence_type")
        )
        target_direction = self._derive_direction_from_divergence(divergence_event)

        if target_direction == FundingStrategyDirection.NEUTRAL:
            return None

        divergence_confidence = self._to_float(
            self._get_value(divergence_event, "confidence"),
            default=0.0,
        )

        regime = state.last_regime
        regime_confidence = self._to_float(
            self._get_value(regime, "confidence") if regime is not None else None,
            default=0.0,
        )
        regime_name = self._enum_str(
            self._get_value(regime, "regime") if regime is not None else None
        )
        bias = self._enum_str(
            self._get_value(regime, "bias") if regime is not None else None
        )

        pressure = state.last_pressure
        pressure_score = self._to_float(
            self._get_value(pressure, "pressure_score") if pressure is not None else None,
            default=0.0,
        )
        pressure_direction = self._enum_str(
            self._get_value(pressure, "direction") if pressure is not None else None
        )
        pressure_level = self._enum_str(
            self._get_value(pressure, "level") if pressure is not None else None
        )

        directional_alignment_bonus = self._calc_directional_alignment_bonus(
            target_direction=target_direction,
            regime_bias=bias,
            pressure_direction=pressure_direction,
        )

        score = self._compute_setup_score(
            divergence_confidence=divergence_confidence,
            pressure_score=pressure_score,
            regime_confidence=regime_confidence,
            directional_alignment_bonus=directional_alignment_bonus,
        )

        confidence = self._compute_setup_confidence(
            score=score,
            divergence_confidence=divergence_confidence,
            regime_confidence=regime_confidence,
        )

        setup_type = (
            self.config.bullish_setup_type
            if target_direction == FundingStrategyDirection.LONG
            else self.config.bearish_setup_type
        )

        side_label = (
            "bullish_funding_divergence_setup"
            if target_direction == FundingStrategyDirection.LONG
            else "bearish_funding_divergence_setup"
        )

        return {
            "direction": target_direction,
            "setup_type": setup_type,
            "score": score,
            "confidence": confidence,
            "reason": side_label,
            "reasons": [
                side_label,
                f"divergence_type:{divergence_type}",
            ],
            "tags": [
                self.config.tag_divergence,
                self.config.tag_dislocation,
                self.config.tag_reversal,
            ],
            "metadata": {
                "divergence_type": divergence_type,
                "regime": regime_name,
                "bias": bias,
                "regime_confidence": regime_confidence,
                "pressure_score": pressure_score,
                "pressure_direction": pressure_direction,
                "pressure_level": pressure_level,
                "divergence_confidence": divergence_confidence,
            },
        }

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

        if (
            state.direction == FundingStrategyDirection.LONG
            and flip_type == FundingFlipType.NEGATIVE_TO_POSITIVE.value
        ):
            return True

        if (
            state.direction == FundingStrategyDirection.SHORT
            and flip_type == FundingFlipType.POSITIVE_TO_NEGATIVE.value
        ):
            return True

        return False

    def _can_confirm_by_repeat_divergence(
        self,
        state: FundingStrategyState,
        previous_divergence: Any | None,
        current_divergence: Any,
    ) -> bool:
        if not self.config.allow_repeat_divergence_confirmation:
            return False

        if state.status != FundingSetupStatus.SETUP_DETECTED:
            return False

        if previous_divergence is None:
            return False

        prev_direction = self._derive_direction_from_divergence(previous_divergence)
        curr_direction = self._derive_direction_from_divergence(current_divergence)

        if prev_direction == FundingStrategyDirection.NEUTRAL:
            return False

        if curr_direction != prev_direction:
            return False

        if curr_direction != state.direction:
            return False

        curr_confidence = self._to_float(
            self._get_value(current_divergence, "confidence"),
            default=0.0,
        )
        return curr_confidence >= self.config.min_divergence_confidence

    def _can_confirm_by_pressure_release(
        self,
        state: FundingStrategyState,
        previous_pressure: Any | None,
        current_pressure: Any,
    ) -> bool:
        if not self.config.allow_pressure_release_confirmation:
            return False

        if state.status != FundingSetupStatus.SETUP_DETECTED:
            return False

        if previous_pressure is None:
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

        score_drop = prev_score - curr_score
        if score_drop < self.config.pressure_release_min_score_drop:
            return False

        prev_level = self._enum_str(self._get_value(previous_pressure, "level"))
        curr_level = self._enum_str(self._get_value(current_pressure, "level"))
        if not self._has_pressure_level_dropped_enough(prev_level, curr_level):
            return False

        return True

    def _can_confirm_by_regime_shift(
        self,
        state: FundingStrategyState,
        previous_regime: Any | None,
        current_regime: Any,
    ) -> bool:
        if not self.config.allow_regime_shift_confirmation:
            return False

        if state.status != FundingSetupStatus.SETUP_DETECTED:
            return False

        if previous_regime is None:
            return False

        prev_bias = self._enum_str(self._get_value(previous_regime, "bias"))
        curr_bias = self._enum_str(self._get_value(current_regime, "bias"))

        if curr_bias is None or prev_bias is None:
            return False

        if state.direction == FundingStrategyDirection.LONG:
            if curr_bias in {
                FundingBias.SHORT_BIAS.value,
                FundingBias.OVERCROWDED_SHORTS.value,
                FundingBias.SQUEEZE_RISK_SHORTS.value,
            }:
                return True

        if state.direction == FundingStrategyDirection.SHORT:
            if curr_bias in {
                FundingBias.LONG_BIAS.value,
                FundingBias.OVERCROWDED_LONGS.value,
                FundingBias.SQUEEZE_RISK_LONGS.value,
            }:
                return True

        return False

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

        if (
            state.direction == FundingStrategyDirection.LONG
            and flip_type == FundingFlipType.POSITIVE_TO_NEGATIVE.value
        ):
            return True

        if (
            state.direction == FundingStrategyDirection.SHORT
            and flip_type == FundingFlipType.NEGATIVE_TO_POSITIVE.value
        ):
            return True

        return False

    def _should_invalidate_by_pressure(
        self,
        state: FundingStrategyState,
        pressure_state: Any,
    ) -> bool:
        if not self.config.invalidate_on_pressure_breakdown:
            return False

        pressure_score = self._to_float(
            self._get_value(pressure_state, "pressure_score"),
            default=0.0,
        )
        level = self._enum_str(self._get_value(pressure_state, "level"))

        if level in {
            FundingPressureLevel.UNKNOWN.value,
            FundingPressureLevel.LOW.value,
        } and pressure_score < (self.config.min_pressure_score * 0.70):
            return True

        direction = self._enum_str(self._get_value(pressure_state, "direction"))

        if state.direction == FundingStrategyDirection.LONG:
            if direction == FundingPressureDirection.LONG.value:
                return True

        if state.direction == FundingStrategyDirection.SHORT:
            if direction == FundingPressureDirection.SHORT.value:
                return True

        return False

    def _should_invalidate_by_regime(
        self,
        state: FundingStrategyState,
        regime_state: Any,
    ) -> bool:
        if not self.config.invalidate_on_regime_conflict:
            return False

        bias = self._enum_str(self._get_value(regime_state, "bias"))
        regime_name = self._enum_str(self._get_value(regime_state, "regime"))

        if regime_name in {
            FundingRegime.UNKNOWN.value,
        }:
            return True

        if state.direction == FundingStrategyDirection.LONG:
            if bias in {
                FundingBias.LONG_BIAS.value,
                FundingBias.OVERCROWDED_LONGS.value,
                FundingBias.SQUEEZE_RISK_LONGS.value,
            }:
                return True

        if state.direction == FundingStrategyDirection.SHORT:
            if bias in {
                FundingBias.SHORT_BIAS.value,
                FundingBias.OVERCROWDED_SHORTS.value,
                FundingBias.SQUEEZE_RISK_SHORTS.value,
            }:
                return True

        return False

    def _is_opposite_divergence_for_state(
        self,
        state: FundingStrategyState,
        divergence_event: Any,
    ) -> bool:
        direction = self._derive_direction_from_divergence(divergence_event)
        if direction == FundingStrategyDirection.NEUTRAL:
            return False
        return direction != state.direction

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_setup_score(
        self,
        *,
        divergence_confidence: float,
        pressure_score: float,
        regime_confidence: float,
        directional_alignment_bonus: float,
    ) -> float:
        return self._clip_score(
            self._average_scores(
                divergence_confidence,
                pressure_score,
                regime_confidence,
                directional_alignment_bonus,
            )
        )

    def _compute_setup_confidence(
        self,
        *,
        score: float,
        divergence_confidence: float,
        regime_confidence: float,
    ) -> float:
        return self._clip_score(
            self._average_scores(
                score,
                divergence_confidence,
                regime_confidence,
            )
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
        return self._clip_score(
            self._average_scores(
                state.score,
                flip_confidence,
                1.0,
            )
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
        return self._clip_score(
            self._average_scores(
                state.confidence,
                flip_confidence,
                1.0,
            )
        )

    def _compute_confirmation_score_from_repeat_divergence(
        self,
        state: FundingStrategyState,
        divergence_event: Any,
    ) -> float:
        divergence_confidence = self._to_float(
            self._get_value(divergence_event, "confidence"),
            default=0.0,
        )
        return self._clip_score(
            self._average_scores(
                state.score,
                divergence_confidence,
                min(1.0, state.score + self.config.repeat_divergence_confirmation_bonus),
            )
        )

    def _compute_confirmation_confidence_from_repeat_divergence(
        self,
        state: FundingStrategyState,
        divergence_event: Any,
    ) -> float:
        divergence_confidence = self._to_float(
            self._get_value(divergence_event, "confidence"),
            default=0.0,
        )
        return self._clip_score(
            self._average_scores(
                state.confidence,
                divergence_confidence,
                min(
                    1.0,
                    state.confidence + self.config.repeat_divergence_confirmation_bonus,
                ),
            )
        )

    def _compute_confirmation_score_from_pressure_release(
        self,
        state: FundingStrategyState,
        pressure_state: Any,
    ) -> float:
        curr_pressure_score = self._to_float(
            self._get_value(pressure_state, "pressure_score"),
            default=0.0,
        )
        return self._clip_score(
            self._average_scores(
                state.score,
                1.0 - min(curr_pressure_score, 1.0),
                0.85,
            )
        )

    def _compute_confirmation_confidence_from_pressure_release(
        self,
        state: FundingStrategyState,
        pressure_state: Any,
    ) -> float:
        mean_reversion_probability = self._to_float(
            self._get_value(pressure_state, "mean_reversion_probability"),
            default=0.0,
        )
        squeeze_probability = self._to_float(
            self._get_value(pressure_state, "squeeze_probability"),
            default=0.0,
        )
        return self._clip_score(
            self._average_scores(
                state.confidence,
                mean_reversion_probability,
                squeeze_probability,
            )
        )

    def _compute_confirmation_score_from_regime_shift(
        self,
        state: FundingStrategyState,
        regime_state: Any,
    ) -> float:
        regime_confidence = self._to_float(
            self._get_value(regime_state, "confidence"),
            default=0.0,
        )
        return self._clip_score(
            self._average_scores(
                state.score,
                regime_confidence,
                min(1.0, state.score + self.config.regime_shift_confirmation_bonus),
            )
        )

    def _compute_confirmation_confidence_from_regime_shift(
        self,
        state: FundingStrategyState,
        regime_state: Any,
    ) -> float:
        regime_confidence = self._to_float(
            self._get_value(regime_state, "confidence"),
            default=0.0,
        )
        return self._clip_score(
            self._average_scores(
                state.confidence,
                regime_confidence,
                min(1.0, state.confidence + self.config.regime_shift_confirmation_bonus),
            )
        )

    # ------------------------------------------------------------------
    # Divergence interpretation
    # ------------------------------------------------------------------

    def _derive_direction_from_divergence(
        self,
        divergence_event: Any,
    ) -> FundingStrategyDirection:
        divergence_type = self._enum_str(
            self._get_value(divergence_event, "divergence_type")
        )

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

    def _calc_directional_alignment_bonus(
        self,
        *,
        target_direction: FundingStrategyDirection,
        regime_bias: str | None,
        pressure_direction: str | None,
    ) -> float:
        bonus = 0.50

        if target_direction == FundingStrategyDirection.LONG:
            if regime_bias in {
                FundingBias.SHORT_BIAS.value,
                FundingBias.OVERCROWDED_SHORTS.value,
                FundingBias.SQUEEZE_RISK_SHORTS.value,
            }:
                bonus += 0.25
            if pressure_direction in {
                FundingPressureDirection.SHORT.value,
                FundingPressureDirection.NEUTRAL.value,
            }:
                bonus += 0.25

        elif target_direction == FundingStrategyDirection.SHORT:
            if regime_bias in {
                FundingBias.LONG_BIAS.value,
                FundingBias.OVERCROWDED_LONGS.value,
                FundingBias.SQUEEZE_RISK_LONGS.value,
            }:
                bonus += 0.25
            if pressure_direction in {
                FundingPressureDirection.LONG.value,
                FundingPressureDirection.NEUTRAL.value,
            }:
                bonus += 0.25

        return self._clip_score(bonus)

    # ------------------------------------------------------------------
    # Payload normalization
    # ------------------------------------------------------------------

    def _normalize_regime_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "symbol": str(payload.get("symbol", "")).upper().strip(),
            "exchange": str(payload.get("exchange", "unknown")).lower().strip(),
            "regime": payload.get("regime"),
            "bias": payload.get("bias"),
            "confidence": payload.get("confidence"),
            "event_time": self.extract_event_time(payload),
            "metadata": payload.get("metadata", {}),
        }

    def _normalize_pressure_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "symbol": str(payload.get("symbol", "")).upper().strip(),
            "exchange": str(payload.get("exchange", "unknown")).lower().strip(),
            "direction": payload.get("direction"),
            "level": payload.get("level"),
            "pressure_score": payload.get("pressure_score"),
            "squeeze_probability": payload.get("squeeze_probability"),
            "mean_reversion_probability": payload.get("mean_reversion_probability"),
            "event_time": self.extract_event_time(payload),
            "metadata": payload.get("metadata", {}),
        }

    def _normalize_divergence_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "symbol": str(payload.get("symbol", "")).upper().strip(),
            "exchange": str(payload.get("exchange", "unknown")).lower().strip(),
            "divergence_type": payload.get("divergence_type"),
            "funding_rate": payload.get("funding_rate"),
            "price_change_pct": payload.get("price_change_pct"),
            "oi_change_pct": payload.get("oi_change_pct"),
            "cvd_change": payload.get("cvd_change"),
            "long_liquidations": payload.get("long_liquidations"),
            "short_liquidations": payload.get("short_liquidations"),
            "confidence": payload.get("confidence"),
            "event_time": self.extract_event_time(payload),
            "metadata": payload.get("metadata", {}),
        }

    def _normalize_flip_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "symbol": str(payload.get("symbol", "")).upper().strip(),
            "exchange": str(payload.get("exchange", "unknown")).lower().strip(),
            "flip_type": payload.get("flip_type"),
            "confidence": payload.get("confidence"),
            "previous_rate": payload.get("previous_rate"),
            "current_rate": payload.get("current_rate"),
            "flip_magnitude": payload.get("flip_magnitude"),
            "event_time": self.extract_event_time(payload),
            "metadata": payload.get("metadata", {}),
        }

    # ------------------------------------------------------------------
    # Emit hooks
    # ------------------------------------------------------------------

    def on_before_setup_emit(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "divergence"
        payload["signal_class"] = "directional_dislocation"
        return payload

    def on_before_confirmation_emit(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "divergence"
        payload["signal_class"] = "directional_dislocation"
        payload["is_tradeable"] = True
        return payload

    def on_before_invalidation_emit(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "divergence"
        return payload

    def on_before_expiration_emit(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "divergence"
        return payload

    # ------------------------------------------------------------------
    # Internal utils
    # ------------------------------------------------------------------

    def _has_pressure_level_dropped_enough(
        self,
        previous_level: str | None,
        current_level: str | None,
    ) -> bool:
        rank = {
            FundingPressureLevel.UNKNOWN.value: 0,
            FundingPressureLevel.LOW.value: 1,
            FundingPressureLevel.MODERATE.value: 2,
            FundingPressureLevel.HIGH.value: 3,
            FundingPressureLevel.EXTREME.value: 4,
        }
        prev_rank = rank.get(previous_level or "", 0)
        curr_rank = rank.get(current_level or "", 0)
        return (prev_rank - curr_rank) >= self.config.confirm_on_pressure_drop_levels

    def _extract_event_time_from_normalized(self, obj: Any) -> datetime | None:
        value = self._get_value(obj, "event_time")
        if isinstance(value, datetime):
            return value
        return None

    def _get_value(self, obj: Any, field: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(field)
        return getattr(obj, field, None)

    def _enum_str(self, value: Any) -> str | None:
        if value is None:
            return None
        if hasattr(value, "value"):
            raw = getattr(value, "value", None)
            if raw is not None:
                return str(raw)
        return str(value)

    def _to_float(
        self,
        value: Any,
        *,
        default: float = 0.0,
    ) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default