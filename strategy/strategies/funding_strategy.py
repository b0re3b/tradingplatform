from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.logger import get_logger

from analytics.funding.enums import (
    FundingBias,
    FundingDivergenceType,
    FundingRegime,
    FundingSignalType,
    FundingTimeframe,
)
from analytics.funding.models import (
    FundingDivergenceEvent,
    FundingExtremeEvent,
    FundingFlipEvent,
    FundingPressureState,
    FundingRegimeState,
    FundingSignal,
    FundingSnapshot,
)


@dataclass(slots=True)
class FundingStrategyConfig:
    """
    Конфігурація агрегуючої funding strategy.

    Ідея:
    - regime дає базовий structural context
    - pressure дає crowding/squeeze context
    - extreme дає reversion/squeeze context
    - flip дає regime transition context
    - divergence дає confirmation / contradiction context
    """

    default_timeframe: FundingTimeframe = FundingTimeframe.H1

    # Ваги компонентів
    regime_weight: float = 0.25
    pressure_weight: float = 0.30
    extreme_weight: float = 0.20
    flip_weight: float = 0.10
    divergence_weight: float = 0.15

    # Пороги для фінального рішення
    weak_signal_threshold: float = 0.15
    moderate_signal_threshold: float = 0.35
    strong_signal_threshold: float = 0.60

    # Мінімальна впевненість, нижче якої краще не формувати meaningful signal
    min_confidence_threshold: float = 0.15

    # Додаткові коефіцієнти
    squeeze_boost: float = 1.15
    extreme_reversion_boost: float = 1.10
    divergence_confirmation_boost: float = 1.10
    conflicting_signal_penalty: float = 0.75

    # Обмеження
    clamp_score_min: float = -1.0
    clamp_score_max: float = 1.0


@dataclass(slots=True)
class FundingStrategyDecision:
    """
    Внутрішнє узагальнене рішення funding strategy.

    Це не заміняє FundingSignal, а є більш багатим внутрішнім container,
    з якого вже можна побудувати FundingSignal для strategy layer.
    """

    symbol: str
    timeframe: FundingTimeframe

    score: float
    confidence: float

    bullish: bool
    bearish: bool
    neutral: bool

    signal_type: FundingSignalType
    bias: FundingBias
    regime: FundingRegime

    summary: str
    factors: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_signal(
        self,
        snapshot: FundingSnapshot,
        exchange=None,
        event_time=None,
    ) -> FundingSignal:
        return FundingSignal(
            symbol=self.symbol,
            exchange=exchange or snapshot.exchange,
            timeframe=self.timeframe,
            signal_type=self.signal_type,
            bias=self.bias,
            regime=self.regime,
            score=self.score,
            confidence=self.confidence,
            description=self.summary,
            supporting_factors=list(self.factors),
            tags=list(self.tags),
            event_time=event_time or snapshot.event_time,
            metadata=dict(self.metadata),
        )


class FundingStrategy:
    """
    Узагальнюючий decision layer для funding analytics.

    Клас приймає вже готові результати:
    - FundingRegimeState
    - FundingPressureState
    - FundingFlipEvent
    - FundingExtremeEvent
    - FundingDivergenceEvent

    І формує єдину funding-інтерпретацію для strategy engine.
    """

    def __init__(
        self,
        config: FundingStrategyConfig | None = None,
    ) -> None:
        self.config = config or FundingStrategyConfig()
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        snapshot: FundingSnapshot,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState | None = None,
        flip_event: FundingFlipEvent | None = None,
        extreme_event: FundingExtremeEvent | None = None,
        divergence_event: FundingDivergenceEvent | None = None,
        timeframe: FundingTimeframe | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> FundingStrategyDecision:
        """
        Побудова інтегрального funding decision.

        Parameters
        ----------
        snapshot:
            Поточний funding snapshot.
        regime_state:
            Базовий regime state.
        pressure_state:
            Поточний pressure state.
        flip_event:
            Flip event, якщо був.
        extreme_event:
            Extreme event, якщо був.
        divergence_event:
            Divergence event, якщо був.
        timeframe:
            Таймфрейм оцінки.
        extra_metadata:
            Додаткові metadata.

        Returns
        -------
        FundingStrategyDecision
        """
        tf = timeframe or regime_state.timeframe or self.config.default_timeframe

        regime_component = self._score_regime(regime_state)
        pressure_component = self._score_pressure(pressure_state)
        extreme_component = self._score_extreme(extreme_event)
        flip_component = self._score_flip(flip_event)
        divergence_component = self._score_divergence(divergence_event)

        raw_score = (
            self.config.regime_weight * regime_component
            + self.config.pressure_weight * pressure_component
            + self.config.extreme_weight * extreme_component
            + self.config.flip_weight * flip_component
            + self.config.divergence_weight * divergence_component
        )

        adjusted_score = self._apply_interaction_rules(
            raw_score=raw_score,
            regime_state=regime_state,
            pressure_state=pressure_state,
            extreme_event=extreme_event,
            divergence_event=divergence_event,
        )

        final_score = self._clamp(
            adjusted_score,
            self.config.clamp_score_min,
            self.config.clamp_score_max,
        )

        confidence = self._calculate_confidence(
            regime_state=regime_state,
            pressure_state=pressure_state,
            flip_event=flip_event,
            extreme_event=extreme_event,
            divergence_event=divergence_event,
            final_score=final_score,
        )

        signal_type = self._resolve_signal_type(
            final_score=final_score,
            regime_state=regime_state,
            pressure_state=pressure_state,
            extreme_event=extreme_event,
            divergence_event=divergence_event,
            flip_event=flip_event,
            confidence=confidence,
        )

        bullish = final_score > 0
        bearish = final_score < 0
        neutral = final_score == 0

        factors = self._build_factors(
            snapshot=snapshot,
            regime_state=regime_state,
            pressure_state=pressure_state,
            flip_event=flip_event,
            extreme_event=extreme_event,
            divergence_event=divergence_event,
            final_score=final_score,
            confidence=confidence,
        )

        tags = self._build_tags(
            regime_state=regime_state,
            pressure_state=pressure_state,
            flip_event=flip_event,
            extreme_event=extreme_event,
            divergence_event=divergence_event,
            signal_type=signal_type,
        )

        summary = self._build_summary(
            symbol=snapshot.symbol,
            signal_type=signal_type,
            final_score=final_score,
            confidence=confidence,
            factors=factors,
        )

        metadata: dict[str, Any] = {
            "regime_component": regime_component,
            "pressure_component": pressure_component,
            "extreme_component": extreme_component,
            "flip_component": flip_component,
            "divergence_component": divergence_component,
            "raw_score": raw_score,
            "adjusted_score": adjusted_score,
            "final_score": final_score,
            "funding_rate": snapshot.funding_rate,
            "regime": regime_state.regime.value,
            "bias": regime_state.bias.value,
        }

        if pressure_state is not None:
            metadata["pressure_score"] = pressure_state.pressure_score
            metadata["pressure_level"] = pressure_state.level.value
            metadata["pressure_direction"] = pressure_state.direction.value

        if flip_event is not None:
            metadata["flip_type"] = flip_event.flip_type.value
            metadata["flip_confidence"] = flip_event.confidence

        if extreme_event is not None:
            metadata["extreme_type"] = extreme_event.extreme_type.value
            metadata["extreme_severity"] = extreme_event.severity

        if divergence_event is not None:
            metadata["divergence_type"] = divergence_event.divergence_type.value
            metadata["divergence_confidence"] = divergence_event.confidence

        if extra_metadata:
            metadata.update(extra_metadata)

        decision = FundingStrategyDecision(
            symbol=snapshot.symbol,
            timeframe=tf,
            score=final_score,
            confidence=confidence,
            bullish=bullish,
            bearish=bearish,
            neutral=neutral,
            signal_type=signal_type,
            bias=regime_state.bias,
            regime=regime_state.regime,
            summary=summary,
            factors=factors,
            tags=tags,
            metadata=metadata,
        )

        self.logger.debug(
            "Funding strategy evaluated: symbol=%s regime=%s bias=%s score=%.4f "
            "confidence=%.4f signal_type=%s",
            decision.symbol,
            decision.regime.value,
            decision.bias.value,
            decision.score,
            decision.confidence,
            decision.signal_type.value,
        )

        return decision

    def build_signal(
        self,
        snapshot: FundingSnapshot,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState | None = None,
        flip_event: FundingFlipEvent | None = None,
        extreme_event: FundingExtremeEvent | None = None,
        divergence_event: FundingDivergenceEvent | None = None,
        timeframe: FundingTimeframe | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> FundingSignal:
        """
        Shortcut для побудови фінального FundingSignal.
        """
        decision = self.evaluate(
            snapshot=snapshot,
            regime_state=regime_state,
            pressure_state=pressure_state,
            flip_event=flip_event,
            extreme_event=extreme_event,
            divergence_event=divergence_event,
            timeframe=timeframe,
            extra_metadata=extra_metadata,
        )
        return decision.to_signal(snapshot=snapshot)

    # ------------------------------------------------------------------
    # Component scoring
    # ------------------------------------------------------------------

    def _score_regime(self, regime_state: FundingRegimeState) -> float:
        """
        Regime score:
        позитивний funding частіше bearish contrarian context,
        негативний funding частіше bullish contrarian context.
        """
        confidence = regime_state.confidence

        if regime_state.regime == FundingRegime.NEUTRAL:
            return 0.0

        if regime_state.regime in {FundingRegime.POSITIVE, FundingRegime.EXTREME_POSITIVE}:
            return -confidence

        if regime_state.regime in {FundingRegime.NEGATIVE, FundingRegime.EXTREME_NEGATIVE}:
            return confidence

        return 0.0

    def _score_pressure(self, pressure_state: FundingPressureState | None) -> float:
        if pressure_state is None:
            return 0.0

        score = pressure_state.pressure_score

        if pressure_state.direction.value == "long":
            return -score
        if pressure_state.direction.value == "short":
            return score
        return 0.0

    def _score_extreme(self, extreme_event: FundingExtremeEvent | None) -> float:
        if extreme_event is None:
            return 0.0

        severity = extreme_event.severity

        if extreme_event.funding_rate > 0:
            return -severity
        if extreme_event.funding_rate < 0:
            return severity
        return 0.0

    def _score_flip(self, flip_event: FundingFlipEvent | None) -> float:
        if flip_event is None:
            return 0.0

        if flip_event.flip_type.value == "negative_to_positive":
            return -flip_event.confidence

        if flip_event.flip_type.value == "positive_to_negative":
            return flip_event.confidence

        return 0.0

    def _score_divergence(self, divergence_event: FundingDivergenceEvent | None) -> float:
        if divergence_event is None:
            return 0.0

        bullish_types = {
            FundingDivergenceType.PRICE_UP_FUNDING_DOWN,
            FundingDivergenceType.OI_UP_FUNDING_DOWN,
            FundingDivergenceType.CVD_UP_FUNDING_DOWN,
            FundingDivergenceType.LIQUIDATIONS_SHORTS_WITH_NEGATIVE_FUNDING,
        }

        bearish_types = {
            FundingDivergenceType.PRICE_DOWN_FUNDING_UP,
            FundingDivergenceType.OI_UP_FUNDING_UP_PRICE_STALLED,
            FundingDivergenceType.CVD_DOWN_FUNDING_UP,
            FundingDivergenceType.LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING,
        }

        if divergence_event.divergence_type in bullish_types:
            return divergence_event.confidence

        if divergence_event.divergence_type in bearish_types:
            return -divergence_event.confidence

        return 0.0

    # ------------------------------------------------------------------
    # Interaction rules
    # ------------------------------------------------------------------

    def _apply_interaction_rules(
        self,
        raw_score: float,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
    ) -> float:
        score = raw_score

        if pressure_state is not None:
            if (pressure_state.squeeze_probability or 0.0) >= 0.65:
                score *= self.config.squeeze_boost

        if extreme_event is not None:
            if extreme_event.is_reversal_risk:
                score *= self.config.extreme_reversion_boost

        if divergence_event is not None:
            divergence_score = self._score_divergence(divergence_event)

            if score == 0.0:
                score += divergence_score * 0.25
            elif self._same_direction(score, divergence_score):
                score *= self.config.divergence_confirmation_boost
            else:
                score *= self.config.conflicting_signal_penalty

        # Якщо regime neutral, але інші компоненти слабкі — приглушуємо
        if regime_state.regime == FundingRegime.NEUTRAL and abs(score) < self.config.moderate_signal_threshold:
            score *= 0.7

        return score

    # ------------------------------------------------------------------
    # Confidence / signal type
    # ------------------------------------------------------------------

    def _calculate_confidence(
        self,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState | None,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
        final_score: float,
    ) -> float:
        components: list[float] = [regime_state.confidence]

        if pressure_state is not None:
            components.append(max(
                pressure_state.squeeze_probability or 0.0,
                pressure_state.mean_reversion_probability or 0.0,
                pressure_state.pressure_score,
            ))

        if flip_event is not None:
            components.append(flip_event.confidence)

        if extreme_event is not None:
            components.append(extreme_event.severity)

        if divergence_event is not None:
            components.append(divergence_event.confidence)

        if not components:
            return 0.0

        base_confidence = sum(components) / len(components)
        directional_bonus = min(abs(final_score), 1.0) * 0.20
        confidence = base_confidence * 0.80 + directional_bonus

        return self._clamp(confidence, 0.0, 1.0)

    def _resolve_signal_type(
        self,
        final_score: float,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
        flip_event: FundingFlipEvent | None,
        confidence: float,
    ) -> FundingSignalType:
        if confidence < self.config.min_confidence_threshold:
            return FundingSignalType.TREND_CONFIRMATION

        if divergence_event is not None and abs(final_score) >= self.config.weak_signal_threshold:
            return FundingSignalType.DIVERGENCE_DETECTED

        if extreme_event is not None:
            if extreme_event.is_squeeze_risk:
                return FundingSignalType.SQUEEZE_WARNING
            if extreme_event.is_reversal_risk:
                return FundingSignalType.REVERSION_SETUP

        if pressure_state is not None:
            if (pressure_state.squeeze_probability or 0.0) >= 0.65:
                return FundingSignalType.SQUEEZE_WARNING
            if pressure_state.level.value in {"high", "extreme"}:
                return FundingSignalType.CROWDING_WARNING

        if flip_event is not None:
            return FundingSignalType.FLIP_DETECTED

        if regime_state.changed:
            return FundingSignalType.REGIME_CHANGE

        if abs(final_score) >= self.config.moderate_signal_threshold:
            return FundingSignalType.REVERSION_SETUP

        return FundingSignalType.TREND_CONFIRMATION

    # ------------------------------------------------------------------
    # Description builders
    # ------------------------------------------------------------------

    def _build_factors(
        self,
        snapshot: FundingSnapshot,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState | None,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
        final_score: float,
        confidence: float,
    ) -> list[str]:
        factors: list[str] = [
            f"funding_rate={snapshot.funding_rate:.8f}",
            f"regime={regime_state.regime.value}",
            f"bias={regime_state.bias.value}",
            f"regime_confidence={regime_state.confidence:.4f}",
            f"final_score={final_score:.4f}",
            f"confidence={confidence:.4f}",
        ]

        if regime_state.percentile is not None:
            factors.append(f"percentile={regime_state.percentile:.2f}")

        if regime_state.zscore is not None:
            factors.append(f"zscore={regime_state.zscore:.4f}")

        if pressure_state is not None:
            factors.extend([
                f"pressure_level={pressure_state.level.value}",
                f"pressure_direction={pressure_state.direction.value}",
                f"pressure_score={pressure_state.pressure_score:.4f}",
            ])
            if pressure_state.squeeze_probability is not None:
                factors.append(f"squeeze_probability={pressure_state.squeeze_probability:.4f}")
            if pressure_state.mean_reversion_probability is not None:
                factors.append(
                    f"mean_reversion_probability={pressure_state.mean_reversion_probability:.4f}"
                )

        if flip_event is not None:
            factors.extend([
                f"flip_type={flip_event.flip_type.value}",
                f"flip_magnitude={flip_event.flip_magnitude:.8f}",
                f"flip_confidence={flip_event.confidence:.4f}",
            ])

        if extreme_event is not None:
            factors.extend([
                f"extreme_type={extreme_event.extreme_type.value}",
                f"extreme_severity={extreme_event.severity:.4f}",
                f"reversal_risk={extreme_event.is_reversal_risk}",
                f"squeeze_risk={extreme_event.is_squeeze_risk}",
            ])

        if divergence_event is not None:
            factors.extend([
                f"divergence_type={divergence_event.divergence_type.value}",
                f"divergence_confidence={divergence_event.confidence:.4f}",
            ])

        return factors

    def _build_tags(
        self,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState | None,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
        signal_type: FundingSignalType,
    ) -> list[str]:
        tags = [
            "funding",
            signal_type.value,
            regime_state.regime.value,
            regime_state.bias.value,
        ]

        if pressure_state is not None:
            tags.extend([
                "pressure",
                pressure_state.level.value,
                pressure_state.direction.value,
            ])

        if flip_event is not None:
            tags.append(flip_event.flip_type.value)

        if extreme_event is not None:
            tags.append(extreme_event.extreme_type.value)

        if divergence_event is not None:
            tags.append(divergence_event.divergence_type.value)

        return list(dict.fromkeys(tags))

    def _build_summary(
        self,
        symbol: str,
        signal_type: FundingSignalType,
        final_score: float,
        confidence: float,
        factors: list[str],
    ) -> str:
        direction = "bullish" if final_score > 0 else "bearish" if final_score < 0 else "neutral"
        strength = self._strength_label(abs(final_score))

        key_factors = ", ".join(factors[:4])

        return (
            f"Funding strategy for {symbol}: "
            f"{direction} {strength} signal, "
            f"type={signal_type.value}, "
            f"score={final_score:.4f}, "
            f"confidence={confidence:.4f}. "
            f"Key factors: {key_factors}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strength_label(self, abs_score: float) -> str:
        if abs_score >= self.config.strong_signal_threshold:
            return "strong"
        if abs_score >= self.config.moderate_signal_threshold:
            return "moderate"
        if abs_score >= self.config.weak_signal_threshold:
            return "weak"
        return "very_weak"

    def _same_direction(self, a: float, b: float) -> bool:
        if a == 0.0 or b == 0.0:
            return False
        return (a > 0 and b > 0) or (a < 0 and b < 0)

    def _clamp(self, value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))