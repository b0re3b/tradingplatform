from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from analytics.funding.enums import (
    FundingBias,
    FundingDivergenceType,
    FundingFlipType,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
)
from core.event_bus import EventBus
from core.scheduler import Scheduler
from .base import (
    BaseFundingStrategy,
    BaseFundingStrategyConfig,
    FundingSetupStatus,
    FundingStrategyDirection,
    FundingStrategyState,
)

# ---------------------------------------------------------------------------
# ВИПРАВЛЕННЯ #5: Магічні числа замінені на іменовані константи
# ---------------------------------------------------------------------------

# Базова нейтральна складова directional_alignment_bonus.
# Встановлена в 0.5 щоб бонус завжди знаходився у межах [0.5, 1.0],
# тобто ніколи не пеналізує, але й не є домінуючим компонентом score.
_ALIGNMENT_BONUS_BASE: float = 0.50

# Додаткова вага за кожен з двох вимірів вирівнювання
# (режим + pressure). Разом дають максимум 1.0.
_ALIGNMENT_BONUS_PER_DIMENSION: float = 0.25

# При підтвердженні через flip 1.0 означає «ідеальний» третій компонент,
# оскільки flip є бінарним підтвердженням без градації.
_FLIP_CONFIRMATION_PERFECT_WEIGHT: float = 1.0

# Якщо pressure score падає нижче цієї частки від мінімального порогу —
# стан вважається зламаним (pressure breakdown).
_PRESSURE_BREAKDOWN_THRESHOLD_RATIO: float = 0.70

# Цільова впевненість при підтвердженні через pressure release.
# 0.85 відображає, що release є сильним, але не ідеальним сигналом.
_PRESSURE_RELEASE_TARGET_CONFIDENCE: float = 0.85


@dataclass(slots=True)
class FundingDivergenceStrategyConfig(BaseFundingStrategyConfig):
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

    # ВИПРАВЛЕННЯ #5: явна документація ваг скорингу
    # Ваги для _compute_setup_score. Сума не зобов'язана дорівнювати 1 —
    # _weighted_average нормалізує автоматично.
    score_weight_divergence: float = 0.40   # головний сигнал
    score_weight_pressure: float = 0.25     # ринковий контекст
    score_weight_regime: float = 0.20       # макро-режим
    score_weight_alignment: float = 0.15    # бонус за напрямок

    allow_flip_confirmation: bool = True
    allow_repeat_divergence_confirmation: bool = True
    allow_pressure_release_confirmation: bool = True
    allow_regime_shift_confirmation: bool = True

    repeat_divergence_confirmation_bonus: float = 0.12
    regime_shift_confirmation_bonus: float = 0.10

    pressure_release_min_score_drop: float = 0.08
    confirm_on_pressure_drop_levels: int = 1

    # ВИПРАВЛЕННЯ #3: явний прапор — чи блокувати setup за відсутності pressure
    require_pressure_present: bool = False

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

    def validate(self) -> None:
        super().validate()

        if not (0.0 <= self.min_divergence_confidence <= 1.0):
            raise ValueError("min_divergence_confidence must be between 0 and 1")

        if not (0.0 <= self.min_pressure_score <= 1.0):
            raise ValueError("min_pressure_score must be between 0 and 1")

        if not (0.0 <= self.min_regime_confidence <= 1.0):
            raise ValueError("min_regime_confidence must be between 0 and 1")

        if not (0.0 <= self.repeat_divergence_confirmation_bonus <= 1.0):
            raise ValueError("repeat_divergence_confirmation_bonus must be between 0 and 1")

        if not (0.0 <= self.regime_shift_confirmation_bonus <= 1.0):
            raise ValueError("regime_shift_confirmation_bonus must be between 0 and 1")

        if self.pressure_release_min_score_drop < 0:
            raise ValueError("pressure_release_min_score_drop must be >= 0")

        if self.confirm_on_pressure_drop_levels < 0:
            raise ValueError("confirm_on_pressure_drop_levels must be >= 0")

        # ВИПРАВЛЕННЯ #5: валідація нових вагових параметрів
        for attr in (
            "score_weight_divergence",
            "score_weight_pressure",
            "score_weight_regime",
            "score_weight_alignment",
        ):
            v = getattr(self, attr)
            if v < 0.0:
                raise ValueError(f"{attr} must be >= 0")


class FundingDivergenceStrategy(BaseFundingStrategy):
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
            service_name=service_name or resolved_config.source_name,
        )

        self.config: FundingDivergenceStrategyConfig = resolved_config

    @property
    def strategy_name(self) -> str:
        return "funding_divergence"

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

            if self.is_in_cooldown(state) or not state.is_active():
                return

            if self._should_invalidate_by_regime(state, regime_state):
                self.set_invalidated(
                    state,
                    reason="regime_context_invalidated_divergence_setup",
                    cooldown=True,
                    metadata={"invalidation_source": "regime"},
                )
                await self.emit_invalidated(state, extra_payload={"trigger": "regime"})
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
                    extra_payload={"trigger": "regime_shift"},
                )

        except Exception:
            self.logger.exception(
                "Failed to process funding divergence regime event | symbol=%s exchange=%s",
                symbol,
                exchange,
                extra={"exchange": exchange, "symbol": symbol},
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

            if self.is_in_cooldown(state) or not state.is_active():
                return

            if self._should_invalidate_by_pressure(state, pressure_state):
                self.set_invalidated(
                    state,
                    reason="pressure_context_invalidated_divergence_setup",
                    cooldown=True,
                    metadata={"invalidation_source": "pressure"},
                )
                await self.emit_invalidated(state, extra_payload={"trigger": "pressure"})
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
                    extra_payload={"trigger": "pressure_release"},
                )

        except Exception:
            self.logger.exception(
                "Failed to process funding divergence pressure event | symbol=%s exchange=%s",
                symbol,
                exchange,
                extra={"exchange": exchange, "symbol": symbol},
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
                    "Stale funding divergence event ignored | symbol=%s exchange=%s",
                    symbol,
                    exchange,
                    extra={"exchange": exchange, "symbol": symbol},
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
                    metadata={"invalidation_source": "divergence"},
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
                    extra_payload={"trigger": "repeat_divergence"},
                )

        except Exception:
            self.logger.exception(
                "Failed to process funding divergence event | symbol=%s exchange=%s",
                symbol,
                exchange,
                extra={"exchange": exchange, "symbol": symbol},
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

            if self.is_in_cooldown(state) or not state.is_active():
                return

            if self._should_invalidate_by_flip(state, flip_event):
                self.set_invalidated(
                    state,
                    reason="opposite_flip_invalidated_divergence_setup",
                    cooldown=True,
                    metadata={"invalidation_source": "flip"},
                )
                await self.emit_invalidated(state, extra_payload={"trigger": "flip"})
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
                    reason="flip_confirmed_divergence_setup",
                    tags=[self.config.tag_confirmed_by_flip],
                    event_time=self._extract_event_time_from_normalized(flip_event),
                    metadata={"confirmation_source": "flip"},
                )
                await self.emit_confirmed(state, extra_payload={"trigger": "flip"})

        except Exception:
            self.logger.exception(
                "Failed to process funding divergence flip event | symbol=%s exchange=%s",
                symbol,
                exchange,
                extra={"exchange": exchange, "symbol": symbol},
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

        # ВИПРАВЛЕННЯ #3: явна поведінка при відсутності pressure.
        # Якщо require_pressure_present=True — setup без pressure блокується,
        # аналогічно до поведінки зі слабким pressure (< min_pressure_score).
        # За замовчуванням (False) — відсутність pressure не блокує setup,
        # але слабкий pressure блокує. Така асиметрія тепер задокументована
        # і керується конфігом, а не прихована в умові `if pressure is not None`.
        if pressure is None:
            if self.config.require_pressure_present:
                return False
        else:
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
                    return direction in {
                        FundingPressureDirection.SHORT.value,
                        FundingPressureDirection.NEUTRAL.value,
                    }

                if target_direction == FundingStrategyDirection.SHORT:
                    return direction in {
                        FundingPressureDirection.LONG.value,
                        FundingPressureDirection.NEUTRAL.value,
                    }

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
        pressure = state.last_pressure

        regime_confidence = self._to_float(
            self._get_value(regime, "confidence") if regime is not None else None,
            default=0.0,
        )
        bias = self._enum_str(
            self._get_value(regime, "bias") if regime is not None else None
        )

        pressure_score = self._to_float(
            self._get_value(pressure, "pressure_score") if pressure is not None else None,
            default=0.0,
        )
        pressure_direction = self._enum_str(
            self._get_value(pressure, "direction") if pressure is not None else None
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
                "funding_context": self._build_divergence_context(
                    divergence_event=divergence_event,
                    regime=regime,
                    pressure=pressure,
                ),
            },
        }

    # ------------------------------------------------------------------
    # Confirmation guards
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
            state.direction == FundingStrategyDirection.LONG
            and flip_type == FundingFlipType.NEGATIVE_TO_POSITIVE.value
        ) or (
            state.direction == FundingStrategyDirection.SHORT
            and flip_type == FundingFlipType.POSITIVE_TO_NEGATIVE.value
        )

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

        if curr_direction != prev_direction or curr_direction != state.direction:
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

        if (prev_score - curr_score) < self.config.pressure_release_min_score_drop:
            return False

        prev_level = self._enum_str(self._get_value(previous_pressure, "level"))
        curr_level = self._enum_str(self._get_value(current_pressure, "level"))

        return self._has_pressure_level_dropped_enough(prev_level, curr_level)

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
            return curr_bias in {
                FundingBias.SHORT_BIAS.value,
                FundingBias.OVERCROWDED_SHORTS.value,
                FundingBias.SQUEEZE_RISK_SHORTS.value,
            }

        if state.direction == FundingStrategyDirection.SHORT:
            return curr_bias in {
                FundingBias.LONG_BIAS.value,
                FundingBias.OVERCROWDED_LONGS.value,
                FundingBias.SQUEEZE_RISK_LONGS.value,
            }

        return False

    # ------------------------------------------------------------------
    # Invalidation guards
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
            state.direction == FundingStrategyDirection.LONG
            and flip_type == FundingFlipType.POSITIVE_TO_NEGATIVE.value
        ) or (
            state.direction == FundingStrategyDirection.SHORT
            and flip_type == FundingFlipType.NEGATIVE_TO_POSITIVE.value
        )

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

        # ВИПРАВЛЕННЯ #5: _PRESSURE_BREAKDOWN_THRESHOLD_RATIO замість 0.70
        if level in {
            FundingPressureLevel.UNKNOWN.value,
            FundingPressureLevel.LOW.value,
        } and pressure_score < (
            self.config.min_pressure_score * _PRESSURE_BREAKDOWN_THRESHOLD_RATIO
        ):
            return True

        direction = self._enum_str(self._get_value(pressure_state, "direction"))

        if state.direction == FundingStrategyDirection.LONG:
            return direction == FundingPressureDirection.LONG.value

        if state.direction == FundingStrategyDirection.SHORT:
            return direction == FundingPressureDirection.SHORT.value

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

        if regime_name == FundingRegime.UNKNOWN.value:
            return True

        if state.direction == FundingStrategyDirection.LONG:
            return bias in {
                FundingBias.LONG_BIAS.value,
                FundingBias.OVERCROWDED_LONGS.value,
                FundingBias.SQUEEZE_RISK_LONGS.value,
            }

        if state.direction == FundingStrategyDirection.SHORT:
            return bias in {
                FundingBias.SHORT_BIAS.value,
                FundingBias.OVERCROWDED_SHORTS.value,
                FundingBias.SQUEEZE_RISK_SHORTS.value,
            }

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
    # ВИПРАВЛЕННЯ #1: Score / confidence computation
    # Замість простого average з dominantним alignment_bonus
    # використовується weighted average з конфігурованими вагами.
    # ВИПРАВЛЕННЯ #2: Спільний приватний хелпер _boost_metric
    # замість дублювання у чотирьох парах методів.
    # ------------------------------------------------------------------

    def _compute_setup_score(
        self,
        *,
        divergence_confidence: float,
        pressure_score: float,
        regime_confidence: float,
        directional_alignment_bonus: float,
    ) -> float:
        """Зважений score setup-у.

        Divergence confidence отримує найбільшу вагу як первинний сигнал.
        Alignment bonus більше не домінує через просте включення у середнє —
        його вплив обмежений вагою score_weight_alignment.
        """
        cfg = self.config
        return self._clip_score(
            self._weighted_average(
                (divergence_confidence, cfg.score_weight_divergence),
                (pressure_score,            cfg.score_weight_pressure),
                (regime_confidence,         cfg.score_weight_regime),
                (directional_alignment_bonus, cfg.score_weight_alignment),
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
            self._average_scores(score, divergence_confidence, regime_confidence)
        )

    def _boost_metric(
        self,
        base: float,
        signal: float,
        bonus: float,
    ) -> float:
        """ВИПРАВЛЕННЯ #2: Загальний хелпер для підйому score/confidence
        при підтвердженні. Усуває дублювання між чотирма парами методів.

        Розраховує average(base, signal, base + bonus), де третій компонент
        відображає «куди ми хочемо прийти», а signal — силу поточного підтвердження.
        """
        return self._clip_score(
            self._average_scores(base, signal, min(1.0, base + bonus))
        )

    def _compute_confirmation_score_from_flip(
        self,
        state: FundingStrategyState,
        flip_event: Any,
    ) -> float:
        flip_confidence = self._to_float(
            self._get_value(flip_event, "confidence"), default=0.0
        )
        # ВИПРАВЛЕННЯ #5: _FLIP_CONFIRMATION_PERFECT_WEIGHT замість 1.0
        return self._clip_score(
            self._average_scores(
                state.score, flip_confidence, _FLIP_CONFIRMATION_PERFECT_WEIGHT
            )
        )

    def _compute_confirmation_confidence_from_flip(
        self,
        state: FundingStrategyState,
        flip_event: Any,
    ) -> float:
        flip_confidence = self._to_float(
            self._get_value(flip_event, "confidence"), default=0.0
        )
        return self._clip_score(
            self._average_scores(
                state.confidence, flip_confidence, _FLIP_CONFIRMATION_PERFECT_WEIGHT
            )
        )

    def _compute_confirmation_score_from_repeat_divergence(
        self,
        state: FundingStrategyState,
        divergence_event: Any,
    ) -> float:
        # ВИПРАВЛЕННЯ #2: використовуємо _boost_metric замість дублювання
        divergence_confidence = self._to_float(
            self._get_value(divergence_event, "confidence"), default=0.0
        )
        return self._boost_metric(
            base=state.score,
            signal=divergence_confidence,
            bonus=self.config.repeat_divergence_confirmation_bonus,
        )

    def _compute_confirmation_confidence_from_repeat_divergence(
        self,
        state: FundingStrategyState,
        divergence_event: Any,
    ) -> float:
        divergence_confidence = self._to_float(
            self._get_value(divergence_event, "confidence"), default=0.0
        )
        return self._boost_metric(
            base=state.confidence,
            signal=divergence_confidence,
            bonus=self.config.repeat_divergence_confirmation_bonus,
        )

    def _compute_confirmation_score_from_pressure_release(
        self,
        state: FundingStrategyState,
        pressure_state: Any,
    ) -> float:
        curr_pressure_score = self._to_float(
            self._get_value(pressure_state, "pressure_score"), default=0.0
        )
        # ВИПРАВЛЕННЯ #5: _PRESSURE_RELEASE_TARGET_CONFIDENCE замість 0.85
        return self._clip_score(
            self._average_scores(
                state.score,
                1.0 - min(curr_pressure_score, 1.0),
                _PRESSURE_RELEASE_TARGET_CONFIDENCE,
            )
        )

    def _compute_confirmation_confidence_from_pressure_release(
        self,
        state: FundingStrategyState,
        pressure_state: Any,
    ) -> float:
        mean_reversion_probability = self._to_float(
            self._get_value(pressure_state, "mean_reversion_probability"), default=0.0
        )
        squeeze_probability = self._to_float(
            self._get_value(pressure_state, "squeeze_probability"), default=0.0
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
        # ВИПРАВЛЕННЯ #2: використовуємо _boost_metric
        regime_confidence = self._to_float(
            self._get_value(regime_state, "confidence"), default=0.0
        )
        return self._boost_metric(
            base=state.score,
            signal=regime_confidence,
            bonus=self.config.regime_shift_confirmation_bonus,
        )

    def _compute_confirmation_confidence_from_regime_shift(
        self,
        state: FundingStrategyState,
        regime_state: Any,
    ) -> float:
        regime_confidence = self._to_float(
            self._get_value(regime_state, "confidence"), default=0.0
        )
        return self._boost_metric(
            base=state.confidence,
            signal=regime_confidence,
            bonus=self.config.regime_shift_confirmation_bonus,
        )

    # ------------------------------------------------------------------
    # Direction & alignment helpers
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
        # ВИПРАВЛЕННЯ #5: магічні числа замінені на іменовані константи
        bonus = _ALIGNMENT_BONUS_BASE

        if target_direction == FundingStrategyDirection.LONG:
            if regime_bias in {
                FundingBias.SHORT_BIAS.value,
                FundingBias.OVERCROWDED_SHORTS.value,
                FundingBias.SQUEEZE_RISK_SHORTS.value,
            }:
                bonus += _ALIGNMENT_BONUS_PER_DIMENSION
            if pressure_direction in {
                FundingPressureDirection.SHORT.value,
                FundingPressureDirection.NEUTRAL.value,
            }:
                bonus += _ALIGNMENT_BONUS_PER_DIMENSION

        elif target_direction == FundingStrategyDirection.SHORT:
            if regime_bias in {
                FundingBias.LONG_BIAS.value,
                FundingBias.OVERCROWDED_LONGS.value,
                FundingBias.SQUEEZE_RISK_LONGS.value,
            }:
                bonus += _ALIGNMENT_BONUS_PER_DIMENSION
            if pressure_direction in {
                FundingPressureDirection.LONG.value,
                FundingPressureDirection.NEUTRAL.value,
            }:
                bonus += _ALIGNMENT_BONUS_PER_DIMENSION

        return self._clip_score(bonus)

    # ------------------------------------------------------------------
    # Payload normalizers
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
    # Private utilities
    # ------------------------------------------------------------------

    def _build_divergence_context(
        self,
        *,
        divergence_event: Any,
        regime: Any | None,
        pressure: Any | None,
    ) -> dict[str, Any]:
        return {
            "divergence": {
                "divergence_type": self._enum_str(
                    self._get_value(divergence_event, "divergence_type")
                ),
                "confidence": self._to_float(
                    self._get_value(divergence_event, "confidence"),
                    default=0.0,
                ),
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
                "confidence": self._to_float(
                    self._get_value(regime, "confidence"),
                    default=0.0,
                ),
            },
            "pressure": {
                "direction": self._enum_str(self._get_value(pressure, "direction")),
                "level": self._enum_str(self._get_value(pressure, "level")),
                "pressure_score": self._to_float(
                    self._get_value(pressure, "pressure_score"),
                    default=0.0,
                ),
            },
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

    def _weighted_average(self, *pairs: tuple[float, float]) -> float:
        """Зважене середнє. Кожен pair = (value, weight).

        ВИПРАВЛЕННЯ #1: замінює простий _average_scores для _compute_setup_score,
        щоб alignment_bonus не отримував рівну вагу з divergence_confidence.
        """
        total_weight = sum(w for _, w in pairs)
        if total_weight == 0.0:
            return 0.0
        return sum(v * w for v, w in pairs) / total_weight

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