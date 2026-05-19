from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analytics.open_interest.enums import (
    OIAnomalyType,
    OIRegime,
)
from analytics.open_interest.models import (
    OIAnomalyResult,
    OIDivergenceResult,
    OIFeatures,
    OIRegimeResult,
)
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
from strategy.models import (
    SignalContext,
    StrategyEvaluation,
    StrategyMetadata,
    StrategySignal,
)
from .base import OpenInterestStrategyBase


@dataclass(slots=True)
class OIAnomalyStrategyPayload:
    """
    Normalized strategy-level payload для OI anomaly.

    Source of truth:
        analytics.open_interest.models.OIAnomalyResult

    Додатковий context:
        - OIFeatures;
        - OIRegimeResult;
        - OIDivergenceResult;
        - full OI analysis confidence через OpenInterestStrategyBase.
    """

    anomaly: OIAnomalyResult
    features: OIFeatures | None = None
    regime: OIRegimeResult | None = None
    divergence: OIDivergenceResult | None = None

    analysis_confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def anomaly_type(self) -> OIAnomalyType:
        return self.anomaly.anomaly_type

    @property
    def detected(self) -> bool:
        return self.anomaly.detected

    @property
    def confidence(self) -> float:
        return self.anomaly.confidence

    @property
    def score(self) -> float:
        return (
            self.anomaly.score
            if self.anomaly.score is not None
            else self.anomaly.confidence
        )


class OIAnomalyStrategy(OpenInterestStrategyBase):
    """
    Strategy, що інтерпретує anomaly сигнали з analytics.open_interest.

    Важливо:
    - strategy не рахує anomaly самостійно;
    - strategy не читає raw market data;
    - strategy працює з OIAnomalyResult, OIFeatures, OIRegimeResult,
      OIDivergenceResult, які вже підготовлені analytics.open_interest;
    - основний input — context.open_interest = OIAnalysisResult.to_dict();
    - specialized analytics.oi.anomaly payload також підтримується через base.
    """

    STRATEGY_NAME = "oi_anomaly_strategy"
    DEFAULT_PRIORITY = 90

    REQUIRED_FEATURES: set[str] = {
        "analytics.open_interest.anomaly",
    }

    MINIMUM_OI_CONTEXT_KEYS: tuple[str, ...] = (
        "oi.anomaly.type",
        "oi.anomaly.detected",
        "oi.anomaly.confidence",
        "open_interest.anomaly.type",
        "open_interest.anomaly.detected",
        "open_interest.anomaly.confidence",
    )

    REVERSAL_ANOMALIES: set[OIAnomalyType] = {
        OIAnomalyType.OI_COLLAPSE,
        OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
        OIAnomalyType.SUDDEN_DELEVERAGING,
        OIAnomalyType.OVERHEATED_BUILDUP,
        OIAnomalyType.EXTREME_CROWDING,
        OIAnomalyType.FUNDING_OI_IMBALANCE,
        OIAnomalyType.OI_PRICE_DISLOCATION,
    }

    CONTINUATION_ANOMALIES: set[OIAnomalyType] = {
        OIAnomalyType.OI_SPIKE,
        OIAnomalyType.OI_VOLUME_DISLOCATION,
    }

    RISK_CRITICAL_ANOMALIES: set[OIAnomalyType] = {
        OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
        OIAnomalyType.SUDDEN_DELEVERAGING,
        OIAnomalyType.OI_COLLAPSE,
        OIAnomalyType.EXTREME_CROWDING,
    }

    CROWDING_ANOMALIES: set[OIAnomalyType] = {
        OIAnomalyType.OVERHEATED_BUILDUP,
        OIAnomalyType.EXTREME_CROWDING,
        OIAnomalyType.FUNDING_OI_IMBALANCE,
    }

    DISLOCATION_ANOMALIES: set[OIAnomalyType] = {
        OIAnomalyType.OI_PRICE_DISLOCATION,
        OIAnomalyType.OI_VOLUME_DISLOCATION,
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
                "anomaly",
                "risk",
                "reversal",
                "crowding",
                "deleveraging",
                "liquidations",
                "analytics_open_interest",
            ],
            version="1.0.0",
            description=(
                "Інтерпретує OI anomaly з analytics.open_interest і будує "
                "directional signals з урахуванням regime, divergence, funding, "
                "liquidations, orderflow та повного OIFeatures context."
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
                "strategy_type": "risk_anomaly_contextual",
                "base_class": "OpenInterestStrategyBase",
                "canonical_payload": "OIAnalysisResult",
                "primary_result": "OIAnomalyResult",
                "uses_features": True,
                "uses_regime": True,
                "uses_divergence": True,
                "uses_anomaly": True,
            },
        )

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        try:
            self.validate_context(context)

            evaluation = StrategyEvaluation(
                strategy_name=self.STRATEGY_NAME,
                symbol=context.symbol,
                timestamp=context.timestamp,
                signal=None,
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

            payload = self._extract_payload(context)
            if payload is None:
                evaluation.reasons.append("missing_oi_anomaly_result")
                return evaluation

            evaluation.metadata.update(
                {
                    "oi_payload": payload.raw,
                    "oi_anomaly_detected": payload.detected,
                    "oi_anomaly_type": payload.anomaly_type.value,
                    "oi_anomaly_strength": payload.anomaly.strength.value,
                    "oi_anomaly_confidence": payload.confidence,
                    "oi_anomaly_score": payload.score,
                    "oi_analysis_confidence": payload.analysis_confidence,
                    "strategy_market_regime": self.get_market_regime(context).value,
                }
            )

            if payload.regime is not None:
                evaluation.metadata.update(
                    {
                        "oi_regime": payload.regime.regime.value,
                        "oi_regime_confidence": payload.regime.confidence,
                        "oi_regime_score": payload.regime.score,
                    }
                )

            if payload.divergence is not None:
                evaluation.metadata.update(
                    {
                        "oi_divergence_detected": payload.divergence.detected,
                        "oi_divergence_type": payload.divergence.divergence_type.value,
                        "oi_divergence_confidence": payload.divergence.confidence,
                        "oi_divergence_score": payload.divergence.score,
                    }
                )

            if not payload.detected:
                evaluation.reasons.append("anomaly_not_detected")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            if payload.anomaly_type is OIAnomalyType.NONE:
                evaluation.reasons.append("anomaly_type_none")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            regime_allowed, regime_reason = self.is_market_regime_allowed(context)
            if not regime_allowed:
                evaluation.reasons.append(regime_reason or "regime_not_allowed")
                return evaluation

            side = self._map_anomaly_to_side(payload)
            if side is SignalSide.UNKNOWN:
                evaluation.reasons.append("anomaly_not_directional")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            blocked_reason = self._block_reason(payload=payload, side=side)
            if blocked_reason is not None:
                evaluation.reasons.append(blocked_reason)
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            setup_type = self._map_anomaly_to_setup_type(payload)
            score = self._compute_score(
                context=context,
                payload=payload,
                side=side,
            )
            confidence = self._compute_confidence(
                payload=payload,
                side=side,
            )

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
            evaluation.reasons.append("oi_anomaly_signal_generated")
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

    def _extract_payload(
        self,
        context: SignalContext,
    ) -> OIAnomalyStrategyPayload | None:
        anomaly = self.extract_oi_anomaly_result(context)
        if anomaly is None:
            return None

        features = self.extract_oi_features(context)
        regime = self.extract_oi_regime_result(context)
        divergence = self.extract_oi_divergence_result(context)

        reasons: list[str] = []
        reasons.extend(anomaly.reasons)

        if regime is not None:
            reasons.extend(f"regime:{reason}" for reason in regime.reasons)

        if divergence is not None and divergence.detected:
            reasons.extend(f"divergence:{reason}" for reason in divergence.reasons)

        raw = self.extract_oi_domain(context)

        return OIAnomalyStrategyPayload(
            anomaly=anomaly,
            features=features,
            regime=regime,
            divergence=divergence,
            analysis_confidence=self.oi_analysis_confidence(context),
            reasons=list(dict.fromkeys(reasons)),
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Direction / setup mapping
    # ------------------------------------------------------------------

    def _map_anomaly_to_side(
        self,
        payload: OIAnomalyStrategyPayload,
    ) -> SignalSide:
        anomaly_type = payload.anomaly_type
        features = payload.features
        regime = payload.regime.regime if payload.regime is not None else None

        if anomaly_type is OIAnomalyType.OI_SPIKE:
            return self._trend_side_from_regime_or_features(
                regime=regime,
                features=features,
            )

        if anomaly_type in {
            OIAnomalyType.OI_COLLAPSE,
            OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
            OIAnomalyType.SUDDEN_DELEVERAGING,
        }:
            return self._reversal_side_from_flush(features)

        if anomaly_type in {
            OIAnomalyType.OVERHEATED_BUILDUP,
            OIAnomalyType.EXTREME_CROWDING,
            OIAnomalyType.FUNDING_OI_IMBALANCE,
        }:
            return self._contrarian_side_from_crowding(
                regime=regime,
                features=features,
            )

        if anomaly_type is OIAnomalyType.OI_PRICE_DISLOCATION:
            return self._contrarian_side_from_price_dislocation(features)

        if anomaly_type is OIAnomalyType.OI_VOLUME_DISLOCATION:
            return self._trend_side_from_regime_or_features(
                regime=regime,
                features=features,
            )

        return SignalSide.UNKNOWN

    def _map_anomaly_to_setup_type(
        self,
        payload: OIAnomalyStrategyPayload,
    ) -> SetupType:
        anomaly_type = payload.anomaly_type
        regime = payload.regime.regime if payload.regime is not None else None

        if anomaly_type in {
            OIAnomalyType.OI_COLLAPSE,
            OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
            OIAnomalyType.SUDDEN_DELEVERAGING,
        }:
            return SetupType.REVERSAL

        if anomaly_type in self.CROWDING_ANOMALIES:
            if regime is OIRegime.SQUEEZE_SETUP:
                return SetupType.SQUEEZE
            return SetupType.REVERSAL

        if anomaly_type is OIAnomalyType.OI_PRICE_DISLOCATION:
            return SetupType.MEAN_REVERSION

        if anomaly_type in {
            OIAnomalyType.OI_SPIKE,
            OIAnomalyType.OI_VOLUME_DISLOCATION,
        }:
            if regime is OIRegime.SQUEEZE_SETUP:
                return SetupType.SQUEEZE
            return SetupType.BREAKOUT

        return SetupType.UNKNOWN

    def _trend_side_from_regime_or_features(
        self,
        *,
        regime: OIRegime | None,
        features: OIFeatures | None,
    ) -> SignalSide:
        if regime in {
            OIRegime.LONG_BUILDUP,
            OIRegime.SHORT_COVERING,
        }:
            return SignalSide.LONG

        if regime in {
            OIRegime.SHORT_BUILDUP,
            OIRegime.LONG_UNWIND,
        }:
            return SignalSide.SHORT

        if features is None:
            return SignalSide.UNKNOWN

        if features.price_delta_pct is not None:
            if features.price_delta_pct > 0:
                if (
                    features.oi_delta_pct is None
                    or features.oi_delta_pct >= 0
                ):
                    return SignalSide.LONG

            if features.price_delta_pct < 0:
                if (
                    features.oi_delta_pct is None
                    or features.oi_delta_pct >= 0
                ):
                    return SignalSide.SHORT

        if features.oi_pressure_score is not None:
            if features.oi_pressure_score >= 0.25:
                return SignalSide.LONG
            if features.oi_pressure_score <= -0.25:
                return SignalSide.SHORT

        if features.aggressive_flow_imbalance is not None:
            if features.aggressive_flow_imbalance >= 0.15:
                return SignalSide.LONG
            if features.aggressive_flow_imbalance <= -0.15:
                return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _reversal_side_from_flush(
        self,
        features: OIFeatures | None,
    ) -> SignalSide:
        if features is None:
            return SignalSide.UNKNOWN

        if features.liquidation_imbalance is not None:
            if features.liquidation_imbalance <= -0.20:
                return SignalSide.LONG
            if features.liquidation_imbalance >= 0.20:
                return SignalSide.SHORT

        if features.price_delta_pct is not None:
            if features.price_delta_pct <= -0.50:
                return SignalSide.LONG
            if features.price_delta_pct >= 0.50:
                return SignalSide.SHORT

        if features.oi_pressure_score is not None:
            if features.oi_pressure_score <= -0.50:
                return SignalSide.LONG
            if features.oi_pressure_score >= 0.50:
                return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _contrarian_side_from_crowding(
        self,
        *,
        regime: OIRegime | None,
        features: OIFeatures | None,
    ) -> SignalSide:
        if features is not None:
            if features.funding_rate is not None:
                if features.funding_rate > 0:
                    return SignalSide.SHORT
                if features.funding_rate < 0:
                    return SignalSide.LONG

            if features.oi_pressure_score is not None:
                if features.oi_pressure_score >= 0.50:
                    return SignalSide.SHORT
                if features.oi_pressure_score <= -0.50:
                    return SignalSide.LONG

            if features.price_delta_pct is not None:
                if features.price_delta_pct > 0:
                    return SignalSide.SHORT
                if features.price_delta_pct < 0:
                    return SignalSide.LONG

        if regime in {
            OIRegime.LONG_BUILDUP,
            OIRegime.TREND_CONFIRMATION,
            OIRegime.OVERHEATED,
        }:
            return SignalSide.SHORT

        if regime in {
            OIRegime.SHORT_BUILDUP,
            OIRegime.SQUEEZE_SETUP,
        }:
            return SignalSide.LONG

        return SignalSide.UNKNOWN

    def _contrarian_side_from_price_dislocation(
        self,
        features: OIFeatures | None,
    ) -> SignalSide:
        if features is None:
            return SignalSide.UNKNOWN

        if features.price_delta_pct is not None:
            if features.price_delta_pct > 0:
                return SignalSide.SHORT
            if features.price_delta_pct < 0:
                return SignalSide.LONG

        if features.oi_price_efficiency is not None:
            if features.oi_price_efficiency >= 1.25:
                if features.oi_pressure_score is not None:
                    if features.oi_pressure_score > 0:
                        return SignalSide.SHORT
                    if features.oi_pressure_score < 0:
                        return SignalSide.LONG

        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Blockers
    # ------------------------------------------------------------------

    def _block_reason(
        self,
        *,
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
    ) -> str | None:
        features = payload.features

        if payload.anomaly_type is OIAnomalyType.NONE:
            return "anomaly_type_none"

        if features is None:
            return None

        if side is SignalSide.LONG:
            if features.price_delta_pct is not None and features.price_delta_pct > 0:
                if payload.anomaly_type in {
                    OIAnomalyType.OI_COLLAPSE,
                    OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
                    OIAnomalyType.SUDDEN_DELEVERAGING,
                }:
                    return "long_anomaly_reversal_blocked_by_positive_price_delta"

            if (
                features.aggressive_flow_imbalance is not None
                and features.aggressive_flow_imbalance <= -0.35
                and payload.anomaly_type not in {
                    OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
                    OIAnomalyType.SUDDEN_DELEVERAGING,
                }
            ):
                return "long_anomaly_blocked_by_extreme_sell_aggression"

        if side is SignalSide.SHORT:
            if features.price_delta_pct is not None and features.price_delta_pct < 0:
                if payload.anomaly_type in {
                    OIAnomalyType.OI_COLLAPSE,
                    OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
                    OIAnomalyType.SUDDEN_DELEVERAGING,
                }:
                    return "short_anomaly_reversal_blocked_by_negative_price_delta"

            if (
                features.aggressive_flow_imbalance is not None
                and features.aggressive_flow_imbalance >= 0.35
                and payload.anomaly_type not in {
                    OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
                    OIAnomalyType.SUDDEN_DELEVERAGING,
                }
            ):
                return "short_anomaly_blocked_by_extreme_buy_aggression"

        divergence = payload.divergence
        if divergence is not None and divergence.detected:
            hint = self.divergence_to_side_hint(divergence.divergence_type)

            if side is SignalSide.LONG and hint == "bearish":
                return f"long_anomaly_signal_blocked_by_divergence:{divergence.divergence_type.value}"

            if side is SignalSide.SHORT and hint == "bullish":
                return f"short_anomaly_signal_blocked_by_divergence:{divergence.divergence_type.value}"

        return None

    # ------------------------------------------------------------------
    # Scoring / confidence
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        *,
        context: SignalContext,
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
    ) -> float:
        market_regime = self.get_market_regime(context)
        base_score = payload.score if payload.score > 0 else payload.confidence
        score = self.weighted_score(base_score, market_regime)

        if payload.analysis_confidence >= 0.75:
            score += 0.04
        elif payload.analysis_confidence >= 0.50:
            score += 0.02

        if payload.anomaly_type in self.RISK_CRITICAL_ANOMALIES:
            score += 0.07
        elif payload.anomaly_type in self.CROWDING_ANOMALIES:
            score += 0.05
        elif payload.anomaly_type in self.DISLOCATION_ANOMALIES:
            score += 0.04

        if payload.regime is not None:
            score += self._regime_score_adjustment(payload.regime)

        if payload.features is not None:
            score += self._feature_score_adjustment(
                features=payload.features,
                side=side,
                anomaly_type=payload.anomaly_type,
            )

        if payload.divergence is not None and payload.divergence.detected:
            hint = self.divergence_to_side_hint(payload.divergence.divergence_type)

            if side is SignalSide.LONG and hint == "bullish":
                score += min(0.05, payload.divergence.score * 0.08)
            elif side is SignalSide.SHORT and hint == "bearish":
                score += min(0.05, payload.divergence.score * 0.08)
            else:
                score -= min(0.10, payload.divergence.score * 0.12)

        return self.clamp(score)

    def _compute_confidence(
        self,
        *,
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
    ) -> float:
        confidence = payload.confidence if payload.confidence > 0 else payload.score

        if payload.analysis_confidence >= 0.75:
            confidence += 0.04
        elif payload.analysis_confidence >= 0.50:
            confidence += 0.02

        if payload.regime is not None and payload.regime.confidence >= 0.75:
            confidence += 0.03

        if payload.features is not None:
            confidence += self._feature_confidence_adjustment(
                features=payload.features,
                side=side,
                anomaly_type=payload.anomaly_type,
            )

        if payload.divergence is not None and payload.divergence.detected:
            hint = self.divergence_to_side_hint(payload.divergence.divergence_type)

            if side is SignalSide.LONG and hint == "bullish":
                confidence += min(0.03, payload.divergence.confidence * 0.05)
            elif side is SignalSide.SHORT and hint == "bearish":
                confidence += min(0.03, payload.divergence.confidence * 0.05)
            else:
                confidence -= min(0.08, payload.divergence.confidence * 0.10)

        return self.clamp(confidence)

    def _regime_score_adjustment(self, regime: OIRegimeResult) -> float:
        if regime.regime in {
            OIRegime.CAPITULATION,
            OIRegime.OVERHEATED,
            OIRegime.TREND_EXHAUSTION,
        }:
            return 0.06

        if regime.regime is OIRegime.SQUEEZE_SETUP:
            return 0.04

        if regime.regime in {
            OIRegime.LONG_BUILDUP,
            OIRegime.SHORT_BUILDUP,
            OIRegime.TREND_CONFIRMATION,
        }:
            return 0.03

        return 0.0

    def _feature_score_adjustment(
        self,
        *,
        features: OIFeatures,
        side: SignalSide,
        anomaly_type: OIAnomalyType,
    ) -> float:
        adjustment = 0.0

        if features.oi_zscore is not None:
            if abs(features.oi_zscore) >= 3.0:
                adjustment += 0.07
            elif abs(features.oi_zscore) >= 2.0:
                adjustment += 0.05
            elif abs(features.oi_zscore) >= 1.0:
                adjustment += 0.03

        if features.volume_ratio is not None:
            if features.volume_ratio >= 1.50:
                adjustment += 0.05
            elif features.volume_ratio >= 1.00:
                adjustment += 0.03

        if features.oi_delta_pct is not None:
            if abs(features.oi_delta_pct) >= 1.00:
                adjustment += 0.05
            elif abs(features.oi_delta_pct) >= 0.50:
                adjustment += 0.03

        if features.liquidation_imbalance is not None:
            if side is SignalSide.LONG and features.liquidation_imbalance <= -0.20:
                adjustment += 0.05
            elif side is SignalSide.SHORT and features.liquidation_imbalance >= 0.20:
                adjustment += 0.05
            elif abs(features.liquidation_imbalance) >= 0.30:
                adjustment += 0.03

        if features.funding_rate is not None:
            if side is SignalSide.LONG and features.funding_rate < 0:
                adjustment += 0.04
            elif side is SignalSide.SHORT and features.funding_rate > 0:
                adjustment += 0.04

        if features.oi_price_efficiency is not None:
            if anomaly_type is OIAnomalyType.OI_PRICE_DISLOCATION:
                if abs(features.oi_price_efficiency) >= 1.25:
                    adjustment += 0.05
            elif abs(features.oi_price_efficiency) >= 1.00:
                adjustment += 0.03

        if features.aggressive_flow_imbalance is not None:
            if side is SignalSide.LONG and features.aggressive_flow_imbalance > -0.05:
                adjustment += 0.03
            elif side is SignalSide.SHORT and features.aggressive_flow_imbalance < 0.05:
                adjustment += 0.03

        return adjustment

    def _feature_confidence_adjustment(
        self,
        *,
        features: OIFeatures,
        side: SignalSide,
        anomaly_type: OIAnomalyType,
    ) -> float:
        adjustment = 0.0

        if features.oi_zscore is not None and abs(features.oi_zscore) >= 2.0:
            adjustment += 0.04

        if features.volume_ratio is not None and features.volume_ratio >= 1.0:
            adjustment += 0.03

        if features.liquidation_imbalance is not None:
            if side is SignalSide.LONG and features.liquidation_imbalance <= -0.20:
                adjustment += 0.03
            elif side is SignalSide.SHORT and features.liquidation_imbalance >= 0.20:
                adjustment += 0.03

        if features.funding_rate is not None:
            if anomaly_type in self.CROWDING_ANOMALIES:
                if side is SignalSide.LONG and features.funding_rate < 0:
                    adjustment += 0.03
                elif side is SignalSide.SHORT and features.funding_rate > 0:
                    adjustment += 0.03

        if features.oi_price_efficiency is not None:
            if anomaly_type is OIAnomalyType.OI_PRICE_DISLOCATION:
                if abs(features.oi_price_efficiency) >= 1.0:
                    adjustment += 0.03

        return adjustment

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        *,
        context: SignalContext,
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
        setup_type: SetupType,
        score: float,
        confidence: float,
    ) -> StrategySignal:
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
                "oi_anomaly_detected": payload.detected,
                "oi_anomaly_type": payload.anomaly_type.value,
                "oi_anomaly_strength": payload.anomaly.strength.value,
                "oi_anomaly_confidence": payload.confidence,
                "oi_anomaly_score": payload.score,
                "oi_anomaly_confidence_band": payload.anomaly.confidence_band.value,
                "oi_analysis_confidence": payload.analysis_confidence,
                "oi_raw_payload": payload.raw,
            },
        )

        if payload.regime is not None:
            signal.metadata.update(
                {
                    "oi_regime": payload.regime.regime.value,
                    "oi_regime_confidence": payload.regime.confidence,
                    "oi_regime_score": payload.regime.score,
                    "oi_regime_confidence_band": payload.regime.confidence_band.value,
                }
            )

        if payload.divergence is not None:
            signal.metadata.update(
                {
                    "oi_divergence_detected": payload.divergence.detected,
                    "oi_divergence_type": payload.divergence.divergence_type.value,
                    "oi_divergence_confidence": payload.divergence.confidence,
                    "oi_divergence_score": payload.divergence.score,
                    "oi_divergence_confidence_band": payload.divergence.confidence_band.value,
                }
            )

        signal.add_reason(f"oi_anomaly:{payload.anomaly_type.value}")
        signal.add_reason(f"setup_type:{setup_type.value}")

        for reason in payload.reasons:
            signal.add_reason(reason)

        self.add_oi_source_features(signal, context)
        self.append_oi_analysis_metadata(signal, context)
        self.append_oi_analysis_reasons(signal, context)

        if payload.features is not None:
            self.append_oi_feature_reasons(signal, payload.features)

        for confirmation in self._build_confirmations(
            side=side,
            payload=payload,
        ):
            signal.add_confirmation(confirmation)

        return signal

    def _build_confirmations(
        self,
        *,
        side: SignalSide,
        payload: OIAnomalyStrategyPayload,
    ) -> list[str]:
        confirmations: list[str] = [
            f"anomaly:{payload.anomaly_type.value}",
            f"anomaly_strength:{payload.anomaly.strength.value}",
        ]

        if payload.anomaly_type in self.RISK_CRITICAL_ANOMALIES:
            confirmations.append("risk_critical_oi_anomaly")

        if payload.anomaly_type in self.CROWDING_ANOMALIES:
            confirmations.append("crowding_anomaly_context")

        if payload.anomaly_type in self.DISLOCATION_ANOMALIES:
            confirmations.append("dislocation_anomaly_context")

        if payload.analysis_confidence >= 0.75:
            confirmations.append("high_confidence_oi_analysis")

        if payload.regime is not None:
            confirmations.append(f"oi_regime:{payload.regime.regime.value}")

            if payload.regime.regime is OIRegime.CAPITULATION:
                confirmations.append("capitulation_regime_context")

            if payload.regime.regime is OIRegime.OVERHEATED:
                confirmations.append("overheated_regime_context")

            if payload.regime.regime is OIRegime.SQUEEZE_SETUP:
                confirmations.append("squeeze_regime_context")

        if payload.divergence is not None and payload.divergence.detected:
            hint = self.divergence_to_side_hint(payload.divergence.divergence_type)

            if side is SignalSide.LONG and hint == "bullish":
                confirmations.append("bullish_divergence_support")

            elif side is SignalSide.SHORT and hint == "bearish":
                confirmations.append("bearish_divergence_support")

            else:
                confirmations.append("divergence_risk_context")

        features = payload.features
        if features is None:
            return list(dict.fromkeys(confirmations))

        if features.oi_zscore is not None and abs(features.oi_zscore) >= 2.0:
            confirmations.append("extreme_oi_zscore_context")

        if features.volume_ratio is not None and features.volume_ratio >= 1.0:
            confirmations.append("volume_context_present")

        if features.oi_delta_pct is not None and abs(features.oi_delta_pct) >= 0.50:
            confirmations.append("large_oi_delta_context")

        if features.oi_price_efficiency is not None and abs(features.oi_price_efficiency) >= 1.0:
            confirmations.append("oi_price_efficiency_context")

        if side is SignalSide.LONG:
            if features.liquidation_imbalance is not None and features.liquidation_imbalance <= -0.20:
                confirmations.append("long_flush_context")

            if features.funding_rate is not None and features.funding_rate < 0:
                confirmations.append("negative_funding_context")

            if features.aggressive_flow_imbalance is not None and features.aggressive_flow_imbalance > -0.05:
                confirmations.append("sell_aggression_fading")

        elif side is SignalSide.SHORT:
            if features.liquidation_imbalance is not None and features.liquidation_imbalance >= 0.20:
                confirmations.append("short_flush_context")

            if features.funding_rate is not None and features.funding_rate > 0:
                confirmations.append("positive_funding_context")

            if features.aggressive_flow_imbalance is not None and features.aggressive_flow_imbalance < 0.05:
                confirmations.append("buy_aggression_fading")

        return list(dict.fromkeys(confirmations))