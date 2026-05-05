from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analytics.open_interest.enums import OIDivergenceType
from analytics.open_interest.models import OIFeatures
from strategy.config import StrategyConfig
from strategy.enums import (
    MarketRegime,
    SetupType,
    SignalOrigin,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    Timeframe,
    TriggerType,
)
from strategy.exceptions import StrategyEvaluationError
from strategy.models import SignalContext, StrategyEvaluation, StrategyMetadata, StrategySignal

from .base import OpenInterestStrategyBase


@dataclass(slots=True)
class OIDivergencePayload:
    divergence_type: OIDivergenceType = OIDivergenceType.NONE
    detected: bool = False
    confidence: float = 0.0
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    window_size: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class OIDivergenceStrategy(OpenInterestStrategyBase):
    """
    Strategy, що інтерпретує divergence сигнали з analytics/open_interest
    і перетворює їх у StrategySignal strategy layer.

    Strategy не рахує divergence самостійно.
    Вона використовує вже нормалізовані дані з analytics.open_interest.
    """

    STRATEGY_NAME = "oi_divergence_strategy"
    DEFAULT_PRIORITY = 85

    REQUIRED_FEATURES: set[str] = {
        "oi.divergence.type",
        "oi.divergence.detected",
        "oi.divergence.confidence",
    }

    MINIMUM_OI_CONTEXT_KEYS: tuple[str, ...] = (
        "oi.divergence.type",
        "oi.divergence.detected",
        "oi.divergence.confidence",
    )

    BULLISH_DIVERGENCES: set[OIDivergenceType] = {
        OIDivergenceType.BULLISH,
        OIDivergenceType.EXHAUSTION_DOWN,
    }

    BEARISH_DIVERGENCES: set[OIDivergenceType] = {
        OIDivergenceType.BEARISH,
        OIDivergenceType.EXHAUSTION_UP,
        OIDivergenceType.PRICE_UP_OI_DOWN,
        OIDivergenceType.PRICE_UP_OI_FLAT,
        OIDivergenceType.WEAK_BREAKOUT_UP,
    }

    NEUTRAL_OR_CONTEXTUAL_DIVERGENCES: set[OIDivergenceType] = {
        OIDivergenceType.PRICE_DOWN_OI_DOWN,
        OIDivergenceType.PRICE_DOWN_OI_FLAT,
        OIDivergenceType.WEAK_BREAKOUT_DOWN,
        OIDivergenceType.NONE,
    }

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: Any | None = None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus)

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.STRATEGY_NAME,
            category=StrategyCategory.OPEN_INTEREST,
            timeframe=Timeframe.M1,
            tags=[
                "open_interest",
                "divergence",
                "reversal",
                "exhaustion",
                "contextual",
            ],
            version="1.1.0",
            description=(
                "Інтерпретує divergence між price та open interest і будує "
                "directional strategy signals."
            ),
            required_features=set(self.REQUIRED_FEATURES),
            supported_regimes={
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
                MarketRegime.BREAKOUT,
                MarketRegime.SQUEEZE,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.LOW_VOLATILITY,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "analytics.open_interest",
                "strategy_type": "single_factor_contextual",
                "base_class": "OpenInterestStrategyBase",
            },
        )

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        try:
            self.validate_context(context)

            evaluation = StrategyEvaluation(
                strategy_name=self.STRATEGY_NAME,
                symbol=context.symbol,
                timestamp=context.timestamp,
                passed=False,
                score=0.0,
                confidence=0.0,
                reasons=[],
                metadata={},
            )

            strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)

            runtime_allowed, runtime_reason = self.is_strategy_runtime_allowed(context)
            if not runtime_allowed:
                evaluation.reasons.append(runtime_reason or "strategy_runtime_not_allowed")
                return evaluation

            if strategy_cfg is None:
                evaluation.reasons.append("strategy_config_not_found")
                return evaluation

            if not self.context_has_any_oi_data(context, self.MINIMUM_OI_CONTEXT_KEYS):
                evaluation.reasons.append("missing_open_interest_context")
                return evaluation

            if self.has_stale_required_features(context):
                evaluation.reasons.append("required_oi_features_are_stale")
                return evaluation

            payload = self._extract_divergence_payload(context)
            regime = self.get_market_regime(context)

            evaluation.metadata.update(
                {
                    "divergence": payload.raw,
                    "divergence_type": payload.divergence_type.value,
                    "regime": regime.value,
                }
            )

            if not payload.detected:
                evaluation.reasons.append("divergence_not_detected")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            if payload.divergence_type == OIDivergenceType.NONE:
                evaluation.reasons.append("divergence_type_none")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            regime_allowed, regime_reason = self.is_market_regime_allowed(context)
            if not regime_allowed:
                evaluation.reasons.append(regime_reason or "regime_not_allowed")
                return evaluation

            side = self._map_divergence_to_side(payload.divergence_type)
            if side == SignalSide.UNKNOWN:
                evaluation.reasons.append("divergence_not_directional")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            setup_type = self._map_divergence_to_setup_type(payload.divergence_type)
            score = self._compute_score(context=context, payload=payload)
            confidence = self._compute_confidence(context=context, payload=payload)

            evaluation.score = score
            evaluation.confidence = confidence

            if confidence < strategy_cfg.runtime.min_confidence:
                evaluation.reasons.append("confidence_below_strategy_threshold")
                evaluation.metadata["min_confidence_required"] = strategy_cfg.runtime.min_confidence
                return evaluation

            if score < strategy_cfg.runtime.min_score:
                evaluation.reasons.append("score_below_strategy_threshold")
                evaluation.metadata["min_score_required"] = strategy_cfg.runtime.min_score
                return evaluation

            signal = self._build_signal(
                context=context,
                payload=payload,
                side=side,
                setup_type=setup_type,
                score=score,
                confidence=confidence,
            )
            signal.validate()

            evaluation.signal = signal
            evaluation.passed = True
            evaluation.reasons.extend(payload.reasons)
            evaluation.reasons.append("oi_divergence_signal_generated")
            evaluation.metadata.update(
                {
                    "signal_side": side.value,
                    "setup_type": setup_type.value,
                }
            )
            return evaluation

        except StrategyEvaluationError:
            raise
        except Exception as exc:
            self.logger.exception(
                "Strategy evaluation failed | strategy=%s symbol=%s",
                self.STRATEGY_NAME,
                getattr(context, "symbol", "UNKNOWN"),
            )
            raise StrategyEvaluationError(
                f"{self.STRATEGY_NAME}: failed to evaluate context for "
                f"{getattr(context, 'symbol', 'UNKNOWN')}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_divergence_payload(self, context: SignalContext) -> OIDivergencePayload:
        """
        Підтримує кілька форматів:
        1. context.open_interest["divergence"] = {...}
        2. flat keys у context.open_interest
        3. feature_map keys типу oi.divergence.*
        """
        domain = context.open_interest or {}
        raw = self.extract_domain_section(context, "divergence")

        self.merge_domain_keys(
            raw=raw,
            domain=domain,
            keys=(
                "divergence_type",
                "detected",
                "confidence",
                "score",
                "reasons",
                "window_size",
            ),
        )

        self.merge_aliases(
            raw=raw,
            domain=domain,
            aliases={
                "type": "divergence_type",
                "oi_divergence_type": "divergence_type",
                "open_interest_divergence_type": "divergence_type",
                "is_detected": "detected",
                "oi_divergence_detected": "detected",
                "oi_divergence_confidence": "confidence",
                "oi_divergence_score": "score",
                "oi_divergence_reasons": "reasons",
            },
        )

        self.merge_feature_aliases(
            raw=raw,
            context=context,
            feature_aliases={
                "oi.divergence.type": "divergence_type",
                "oi.divergence.detected": "detected",
                "oi.divergence.confidence": "confidence",
                "oi.divergence.score": "score",
                "oi.divergence.reasons": "reasons",
                "oi.divergence.window_size": "window_size",
                "open_interest.divergence.type": "divergence_type",
                "open_interest.divergence.detected": "detected",
                "open_interest.divergence.confidence": "confidence",
                "open_interest.divergence.score": "score",
                "open_interest.divergence.reasons": "reasons",
                "open_interest.divergence.window_size": "window_size",
            },
        )

        divergence_type = self.parse_divergence_type(raw.get("divergence_type"))
        detected = self.safe_bool(
            raw.get("detected"),
            default=divergence_type != OIDivergenceType.NONE,
        )
        confidence = self.clamp(self.safe_float(raw.get("confidence"), 0.0))
        score = self.clamp(self.safe_float(raw.get("score"), confidence))
        reasons = self.normalize_reasons(raw.get("reasons"))
        window_size = self.safe_int(raw.get("window_size"))

        return OIDivergencePayload(
            divergence_type=divergence_type,
            detected=detected,
            confidence=confidence,
            score=score,
            reasons=reasons,
            window_size=window_size,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def _map_divergence_to_side(self, divergence_type: OIDivergenceType) -> SignalSide:
        if divergence_type in self.BULLISH_DIVERGENCES:
            return SignalSide.LONG

        if divergence_type in self.BEARISH_DIVERGENCES:
            return SignalSide.SHORT

        if divergence_type in {
            OIDivergenceType.PRICE_DOWN_OI_FLAT,
            OIDivergenceType.WEAK_BREAKOUT_DOWN,
            OIDivergenceType.PRICE_DOWN_OI_DOWN,
        }:
            return SignalSide.LONG

        return SignalSide.UNKNOWN

    @staticmethod
    def _map_divergence_to_setup_type(divergence_type: OIDivergenceType) -> SetupType:
        if divergence_type in {
            OIDivergenceType.BULLISH,
            OIDivergenceType.BEARISH,
        }:
            return SetupType.REVERSAL

        if divergence_type in {
            OIDivergenceType.EXHAUSTION_UP,
            OIDivergenceType.EXHAUSTION_DOWN,
        }:
            return SetupType.EXHAUSTION

        if divergence_type in {
            OIDivergenceType.WEAK_BREAKOUT_UP,
            OIDivergenceType.WEAK_BREAKOUT_DOWN,
        }:
            return SetupType.BREAKOUT

        if divergence_type in {
            OIDivergenceType.PRICE_UP_OI_DOWN,
            OIDivergenceType.PRICE_UP_OI_FLAT,
            OIDivergenceType.PRICE_DOWN_OI_DOWN,
            OIDivergenceType.PRICE_DOWN_OI_FLAT,
        }:
            return SetupType.MEAN_REVERSION

        return SetupType.UNKNOWN

    # ------------------------------------------------------------------
    # Scoring / confidence
    # ------------------------------------------------------------------

    def _compute_score(self, context: SignalContext, payload: OIDivergencePayload) -> float:
        regime = self.get_market_regime(context)
        base_score = payload.score if payload.score > 0 else payload.confidence
        score = self.weighted_score(base_score, regime)

        features = self.extract_oi_features(context)
        if features is not None:
            if features.volume_ratio is not None and features.volume_ratio >= 1.0:
                score += 0.05
            if features.oi_price_efficiency is not None and abs(features.oi_price_efficiency) < 0.50:
                score += 0.05
            if features.oi_pressure_score is not None and abs(features.oi_pressure_score) < 0.20:
                score += 0.04
            if features.liquidation_imbalance is not None and abs(features.liquidation_imbalance) >= 0.20:
                score += 0.03
            if features.aggressive_flow_imbalance is not None and abs(features.aggressive_flow_imbalance) < 0.10:
                score += 0.03

        strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
        if strategy_cfg is not None and payload.window_size is not None and payload.window_size >= 5:
            score += 0.02

        return self.clamp(score)

    def _compute_confidence(self, context: SignalContext, payload: OIDivergencePayload) -> float:
        confidence = payload.confidence if payload.confidence > 0 else payload.score

        features = self.extract_oi_features(context)
        if features is not None:
            if features.volume_ratio is not None and features.volume_ratio >= 1.0:
                confidence += 0.04
            if features.oi_zscore is not None and abs(features.oi_zscore) >= 1.0:
                confidence += 0.03
            if features.oi_price_efficiency is not None and abs(features.oi_price_efficiency) < 0.50:
                confidence += 0.03

        return self.clamp(confidence)

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        *,
        context: SignalContext,
        payload: OIDivergencePayload,
        side: SignalSide,
        setup_type: SetupType,
        score: float,
        confidence: float,
    ) -> StrategySignal:
        features = self.extract_oi_features(context)

        signal = StrategySignal(
            symbol=context.symbol,
            side=side,
            strategy_name=self.STRATEGY_NAME,
            category=StrategyCategory.OPEN_INTEREST,
            timeframe=context.timeframe,
            setup_type=setup_type,
            timestamp=context.timestamp,
            confidence=confidence,
            score=score,
            strength=self.strength_from_score(score),
            confidence_grade=self.confidence_grade(confidence),
            status=SignalStatus.NEW,
            trigger_type=TriggerType.PRIMARY,
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=self.priority_from_score(score),
            regime=self.get_market_regime(context),
            metadata={
                "oi_divergence_type": payload.divergence_type.value,
                "oi_divergence_window_size": payload.window_size,
                "oi_raw_payload": payload.raw,
            },
        )

        for reason in payload.reasons:
            signal.add_reason(reason)

        signal.add_reason(f"oi_divergence_type:{payload.divergence_type.value}")
        signal.add_reason(f"setup_type:{setup_type.value}")

        self.add_required_source_features(signal, context)

        if features is not None:
            self.append_oi_feature_reasons(signal, features)
            self.append_oi_feature_metadata(signal, features)

        for confirmation in self._build_confirmations(
            side=side,
            divergence_type=payload.divergence_type,
            features=features,
        ):
            signal.add_confirmation(confirmation)

        return signal

    def _build_confirmations(
        self,
        *,
        side: SignalSide,
        divergence_type: OIDivergenceType,
        features: OIFeatures | None,
    ) -> list[str]:
        confirmations: list[str] = [f"divergence:{divergence_type.value}"]

        if features is None:
            return confirmations

        if side == SignalSide.LONG:
            if features.liquidation_imbalance is not None and features.liquidation_imbalance <= -0.20:
                confirmations.append("long_flush_context")
            if features.aggressive_flow_imbalance is not None and features.aggressive_flow_imbalance > -0.05:
                confirmations.append("sell_aggression_fading")
            if features.funding_rate is not None and features.funding_rate < 0:
                confirmations.append("negative_funding_context")

        if side == SignalSide.SHORT:
            if features.aggressive_flow_imbalance is not None and features.aggressive_flow_imbalance < 0.05:
                confirmations.append("buy_aggression_fading")
            if features.funding_rate is not None and features.funding_rate > 0:
                confirmations.append("positive_funding_context")
            if features.oi_price_efficiency is not None and abs(features.oi_price_efficiency) < 0.50:
                confirmations.append("price_move_not_supported_by_oi")

        if features.volume_ratio is not None and features.volume_ratio >= 1.0:
            confirmations.append("volume_context_present")

        return confirmations
