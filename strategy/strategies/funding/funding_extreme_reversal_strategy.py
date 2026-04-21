from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.logger import get_logger

from analytics.funding.enums import (
    FundingBias,
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
class FundingExtremeReversalStrategyConfig(BaseFundingStrategyConfig):
    """
    Конфіг funding extreme reversal strategy.

    Ідея стратегії:
    - коли funding стає екстремальним
    - crowding/pressure підтверджують перекіс позиціонування
    - стратегія формує contrarian setup
    - далі чекає release / flip для confirm
    """

    strategy_namespace: str = "strategy.funding.extreme_reversal"
    source_name: str = "funding_extreme_reversal_strategy"

    regime_event_name: str = "analytics.funding.regime"
    pressure_event_name: str = "analytics.funding.pressure"
    extreme_event_name: str = "analytics.funding.extreme"
    flip_event_name: str = "analytics.funding.flip"

    min_extreme_severity: float = 0.60
    min_pressure_score: float = 0.55
    min_regime_confidence: float = 0.15

    min_mean_reversion_probability: float = 0.50
    min_squeeze_probability: float = 0.50

    allow_flip_confirmation: bool = True
    allow_pressure_release_confirmation: bool = True

    confirm_on_pressure_drop_levels: int = 1
    pressure_release_min_score_drop: float = 0.10

    invalidate_on_opposite_flip: bool = True
    invalidate_on_pressure_neutralization: bool = True
    invalidate_on_regime_neutral: bool = False

    bearish_setup_type: str = "extreme_positive_reversal"
    bullish_setup_type: str = "extreme_negative_reversal"

    tag_extreme: str = "funding_extreme"
    tag_reversal: str = "reversal"
    tag_crowding: str = "crowding"
    tag_squeeze: str = "squeeze_risk"
    tag_confirmed_by_flip: str = "confirmed_by_flip"
    tag_confirmed_by_release: str = "confirmed_by_release"


class FundingExtremeReversalStrategy(BaseFundingStrategy):
    """
    Стратегія contrarian reversal на funding extremes.

    Потік:
    1. Слухає regime / pressure / extreme / flip
    2. На extreme будує contrarian setup
    3. На flip або pressure release підтверджує setup
    4. На протилежному контексті або зникненні перекосу — інвалідує
    """

    def __init__(
        self,
        event_bus: Any,
        config: FundingExtremeReversalStrategyConfig | None = None,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            config=config or FundingExtremeReversalStrategyConfig(),
        )
        self.config: FundingExtremeReversalStrategyConfig = (
            config or FundingExtremeReversalStrategyConfig()
        )
        self.logger = get_logger(__name__)

    @property
    def strategy_name(self) -> str:
        return "funding_extreme_reversal"

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_subscriptions(self) -> None:
        self.event_bus.subscribe(self.config.regime_event_name, self.on_regime)
        self.event_bus.subscribe(self.config.pressure_event_name, self.on_pressure)
        self.event_bus.subscribe(self.config.extreme_event_name, self.on_extreme)
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
            regime_state = self._normalize_regime_payload(payload)
            self.attach_regime(state, regime_state)

            self._expire_state_if_needed(state)

            if not state.is_active():
                return

            if self._should_invalidate_by_regime(state, regime_state):
                self.set_invalidated(
                    state,
                    reason="regime_context_invalidated_setup",
                    cooldown=True,
                    metadata={
                        "invalidation_source": "regime",
                    },
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={
                        "trigger": "regime",
                    },
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
                    reason="pressure_context_invalidated_setup",
                    cooldown=True,
                    metadata={
                        "invalidation_source": "pressure",
                    },
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={
                        "trigger": "pressure",
                    },
                )
                return

            if self._can_confirm_by_pressure_release(
                state=state,
                previous_pressure=previous_pressure,
                current_pressure=pressure_state,
            ):
                new_score = self._compute_confirmation_score_from_release(
                    state=state,
                    current_pressure=pressure_state,
                )
                new_confidence = self._compute_confirmation_confidence_from_release(
                    state=state,
                    current_pressure=pressure_state,
                )

                self.set_confirmed(
                    state,
                    score=new_score,
                    confidence=new_confidence,
                    reason="pressure_release_confirmed_reversal_setup",
                    tags=[self.config.tag_confirmed_by_release],
                    event_time=self._extract_event_time_from_normalized(pressure_state),
                    metadata={
                        "confirmation_source": "pressure_release",
                    },
                )

                await self.emit_confirmed(
                    state,
                    extra_payload={
                        "trigger": "pressure_release",
                    },
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

    async def on_extreme(self, event: Any) -> None:
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

            if self.is_in_cooldown(state):
                return

            event_time = self._extract_event_time_from_normalized(extreme_event)
            if self.is_stale_event(event_time):
                self.logger.debug(
                    "Stale extreme event ignored: symbol=%s exchange=%s",
                    symbol,
                    exchange,
                )
                return

            setup_candidate = self._build_setup_from_extreme(
                state=state,
                extreme_event=extreme_event,
            )
            if setup_candidate is None:
                return

            direction = setup_candidate["direction"]
            setup_type = setup_candidate["setup_type"]
            score = setup_candidate["score"]
            confidence = setup_candidate["confidence"]
            reason = setup_candidate["reason"]
            reasons = setup_candidate["reasons"]
            tags = setup_candidate["tags"]
            metadata = setup_candidate["metadata"]

            self.set_setup_detected(
                state,
                direction=direction,
                setup_type=setup_type,
                score=score,
                confidence=confidence,
                reason=reason,
                reasons=reasons,
                tags=tags,
                event_time=event_time,
                metadata=metadata,
            )

            await self.emit_setup(
                state,
                extra_payload={
                    "trigger": "extreme",
                },
            )

        except Exception:
            self.logger.exception(
                "Failed to process extreme event in %s: symbol=%s exchange=%s",
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
                    reason="opposite_flip_invalidated_setup",
                    cooldown=True,
                    metadata={
                        "invalidation_source": "flip",
                    },
                )
                await self.emit_invalidated(
                    state,
                    extra_payload={
                        "trigger": "flip",
                    },
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
                    reason="flip_confirmed_reversal_setup",
                    tags=[self.config.tag_confirmed_by_flip],
                    event_time=self._extract_event_time_from_normalized(flip_event),
                    metadata={
                        "confirmation_source": "flip",
                    },
                )

                await self.emit_confirmed(
                    state,
                    extra_payload={
                        "trigger": "flip",
                    },
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
    # Core setup logic
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

        severity = self._to_float(self._get_value(extreme_event, "severity"), default=0.0)
        if severity < self.config.min_extreme_severity:
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
        if pressure_level not in {
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
        if (
            squeeze_probability < self.config.min_squeeze_probability
            and mean_reversion_probability < self.config.min_mean_reversion_probability
        ):
            return None

        extreme_type = self._enum_str(self._get_value(extreme_event, "extreme_type"))
        bias = self._enum_str(self._get_value(regime, "bias"))
        regime_name = self._enum_str(self._get_value(regime, "regime"))

        # ------------------------------------------------------------------
        # Bearish reversal setup:
        # extreme positive + crowded longs + long pressure
        # ------------------------------------------------------------------
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
                severity=severity,
                pressure_score=pressure_score,
                squeeze_probability=squeeze_probability,
                mean_reversion_probability=mean_reversion_probability,
                regime_confidence=regime_confidence,
                directional_alignment_bonus=1.0,
            )

            confidence = self._compute_setup_confidence(
                score=score,
                severity=severity,
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
                ],
                "tags": [
                    self.config.tag_extreme,
                    self.config.tag_reversal,
                    self.config.tag_crowding,
                    self.config.tag_squeeze,
                ],
                "metadata": {
                    "extreme_type": extreme_type,
                    "regime": regime_name,
                    "bias": bias,
                    "pressure_direction": pressure_direction,
                    "pressure_level": pressure_level,
                    "pressure_score": pressure_score,
                    "squeeze_probability": squeeze_probability,
                    "mean_reversion_probability": mean_reversion_probability,
                    "extreme_severity": severity,
                },
            }

        # ------------------------------------------------------------------
        # Bullish reversal setup:
        # extreme negative + crowded shorts + short pressure
        # ------------------------------------------------------------------
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
                severity=severity,
                pressure_score=pressure_score,
                squeeze_probability=squeeze_probability,
                mean_reversion_probability=mean_reversion_probability,
                regime_confidence=regime_confidence,
                directional_alignment_bonus=1.0,
            )

            confidence = self._compute_setup_confidence(
                score=score,
                severity=severity,
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
                ],
                "tags": [
                    self.config.tag_extreme,
                    self.config.tag_reversal,
                    self.config.tag_crowding,
                    self.config.tag_squeeze,
                ],
                "metadata": {
                    "extreme_type": extreme_type,
                    "regime": regime_name,
                    "bias": bias,
                    "pressure_direction": pressure_direction,
                    "pressure_level": pressure_level,
                    "pressure_score": pressure_score,
                    "squeeze_probability": squeeze_probability,
                    "mean_reversion_probability": mean_reversion_probability,
                    "extreme_severity": severity,
                },
            }

        return None

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
            state.direction == FundingStrategyDirection.SHORT
            and flip_type == FundingFlipType.POSITIVE_TO_NEGATIVE.value
        ):
            return True

        if (
            state.direction == FundingStrategyDirection.LONG
            and flip_type == FundingFlipType.NEGATIVE_TO_POSITIVE.value
        ):
            return True

        return False

    def _can_confirm_by_pressure_release(
        self,
        state: FundingStrategyState,
        previous_pressure: Any | None,
        current_pressure: Any,
    ) -> bool:
        if not self.config.allow_pressure_release_confirmation:
            return False

        if previous_pressure is None:
            return False

        if state.status != FundingSetupStatus.SETUP_DETECTED:
            return False

        prev_score = self._to_float(
            self._get_value(previous_pressure, "pressure_score"),
            default=0.0,
        )
        curr_score = self._to_float(
            self._get_value(current_pressure, "pressure_score"),
            default=0.0,
        )

        prev_level = self._enum_str(self._get_value(previous_pressure, "level"))
        curr_level = self._enum_str(self._get_value(current_pressure, "level"))

        if prev_score <= curr_score:
            return False

        score_drop = prev_score - curr_score
        if score_drop < self.config.pressure_release_min_score_drop:
            return False

        if not self._has_pressure_level_dropped_enough(prev_level, curr_level):
            return False

        direction = self._enum_str(self._get_value(current_pressure, "direction"))
        if (
            state.direction == FundingStrategyDirection.SHORT
            and direction != FundingPressureDirection.LONG.value
        ):
            return False

        if (
            state.direction == FundingStrategyDirection.LONG
            and direction != FundingPressureDirection.SHORT.value
        ):
            return False

        return True

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
            state.direction == FundingStrategyDirection.SHORT
            and flip_type == FundingFlipType.NEGATIVE_TO_POSITIVE.value
        ):
            return True

        if (
            state.direction == FundingStrategyDirection.LONG
            and flip_type == FundingFlipType.POSITIVE_TO_NEGATIVE.value
        ):
            return True

        return False

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

            if pressure_score < self.config.min_pressure_score * 0.70:
                return True

        if state.direction == FundingStrategyDirection.SHORT:
            if direction not in {
                FundingPressureDirection.LONG.value,
                FundingPressureDirection.NEUTRAL.value,
            }:
                return True

        if state.direction == FundingStrategyDirection.LONG:
            if direction not in {
                FundingPressureDirection.SHORT.value,
                FundingPressureDirection.NEUTRAL.value,
            }:
                return True

        return False

    def _should_invalidate_by_regime(
        self,
        state: FundingStrategyState,
        regime_state: Any,
    ) -> bool:
        regime = self._enum_str(self._get_value(regime_state, "regime"))
        bias = self._enum_str(self._get_value(regime_state, "bias"))

        if self.config.invalidate_on_regime_neutral:
            if regime in {
                FundingRegime.NEUTRAL.value,
                FundingRegime.UNKNOWN.value,
            }:
                return True

        if state.direction == FundingStrategyDirection.SHORT:
            if bias in {
                FundingBias.SHORT_BIAS.value,
                FundingBias.OVERCROWDED_SHORTS.value,
                FundingBias.SQUEEZE_RISK_SHORTS.value,
            }:
                return True

        if state.direction == FundingStrategyDirection.LONG:
            if bias in {
                FundingBias.LONG_BIAS.value,
                FundingBias.OVERCROWDED_LONGS.value,
                FundingBias.SQUEEZE_RISK_LONGS.value,
            }:
                return True

        return False

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
    ) -> float:
        score = self._average_scores(
            severity,
            pressure_score,
            squeeze_probability,
            mean_reversion_probability,
            regime_confidence,
            directional_alignment_bonus,
        )
        return self._clip_score(score)

    def _compute_setup_confidence(
        self,
        *,
        score: float,
        severity: float,
        regime_confidence: float,
    ) -> float:
        return self._clip_score(
            self._average_scores(score, severity, regime_confidence)
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

    def _compute_confirmation_score_from_release(
        self,
        state: FundingStrategyState,
        current_pressure: Any,
    ) -> float:
        current_pressure_score = self._to_float(
            self._get_value(current_pressure, "pressure_score"),
            default=0.0,
        )
        return self._clip_score(
            self._average_scores(
                state.score,
                1.0 - min(current_pressure_score, 1.0),
                0.90,
            )
        )

    def _compute_confirmation_confidence_from_release(
        self,
        state: FundingStrategyState,
        current_pressure: Any,
    ) -> float:
        squeeze_probability = self._to_float(
            self._get_value(current_pressure, "squeeze_probability"),
            default=0.0,
        )
        mean_reversion_probability = self._to_float(
            self._get_value(current_pressure, "mean_reversion_probability"),
            default=0.0,
        )
        return self._clip_score(
            self._average_scores(
                state.confidence,
                squeeze_probability,
                mean_reversion_probability,
            )
        )

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

    def _normalize_extreme_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "symbol": str(payload.get("symbol", "")).upper().strip(),
            "exchange": str(payload.get("exchange", "unknown")).lower().strip(),
            "extreme_type": payload.get("extreme_type"),
            "severity": payload.get("severity"),
            "is_reversal_risk": payload.get("is_reversal_risk"),
            "is_squeeze_risk": payload.get("is_squeeze_risk"),
            "funding_rate": payload.get("funding_rate"),
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
        payload["strategy_variant"] = "extreme_reversal"
        payload["signal_class"] = "contrarian_reversal"
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
        return payload

    def on_before_invalidation_emit(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "extreme_reversal"
        return payload

    def on_before_expiration_emit(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        payload["strategy_family"] = "funding"
        payload["strategy_variant"] = "extreme_reversal"
        return payload

    # ------------------------------------------------------------------
    # Internal utils
    # ------------------------------------------------------------------

    def _is_positive_extreme(self, extreme_type: str | None) -> bool:
        if not extreme_type:
            return False
        return extreme_type in {
            "local_high",
            "global_high",
            "zscore_high",
            "percentile_high",
        }

    def _is_negative_extreme(self, extreme_type: str | None) -> bool:
        if not extreme_type:
            return False
        return extreme_type in {
            "local_low",
            "global_low",
            "zscore_low",
            "percentile_low",
        }

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