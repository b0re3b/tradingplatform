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
class OICapitulationPayload:
    """
    Normalized strategy-level payload для OI capitulation.

    Source of truth:
        analytics.open_interest:
        - OIRegimeResult(regime=CAPITULATION)
        - OIAnomalyResult із deleveraging/liquidation-driven anomaly
        - OIFeatures
        - optional OIDivergenceResult

    Стратегія не детектить capitulation самостійно. Вона інтерпретує
    готовий analytics context і будує reversal/risk signal лише коли
    capitulation має достатній directional context.
    """

    regime: OIRegimeResult | None = None
    anomaly: OIAnomalyResult | None = None
    features: OIFeatures | None = None
    divergence: OIDivergenceResult | None = None

    analysis_confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_capitulation_regime(self) -> bool:
        return self.regime is not None and self.regime.regime is OIRegime.CAPITULATION

    @property
    def has_capitulation_anomaly(self) -> bool:
        return (
            self.anomaly is not None
            and self.anomaly.detected
            and self.anomaly.anomaly_type
            in {
                OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
                OIAnomalyType.SUDDEN_DELEVERAGING,
                OIAnomalyType.OI_COLLAPSE,
            }
        )

    @property
    def detected(self) -> bool:
        return self.has_capitulation_regime or self.has_capitulation_anomaly

    @property
    def confidence(self) -> float:
        values: list[float] = []

        if self.regime is not None:
            values.append(self.regime.confidence)

        if self.anomaly is not None and self.anomaly.detected:
            values.append(self.anomaly.confidence)

        if self.analysis_confidence > 0:
            values.append(self.analysis_confidence)

        if not values:
            return 0.0

        return sum(values) / len(values)

    @property
    def score(self) -> float:
        values: list[float] = []

        if self.regime is not None:
            values.append(self.regime.score)

        if self.anomaly is not None and self.anomaly.detected:
            values.append(self.anomaly.score)

        if not values:
            return self.confidence

        return max(values)


class OICapitulationStrategy(OpenInterestStrategyBase):
    """
    Strategy для capitulation / forced deleveraging reversal context.

    Важливо:
    - strategy не читає raw market data;
    - strategy не рахує regime/anomaly/features самостійно;
    - strategy працює з готовими результатами analytics.open_interest;
    - основний input — context.open_interest = OIAnalysisResult.to_dict();
    - specialized analytics.oi.capitulation payload також підтримується через base.
    """

    STRATEGY_NAME = "oi_capitulation_strategy"
    DEFAULT_PRIORITY = 95

    REQUIRED_FEATURES: set[str] = {
        "analytics.open_interest.regime",
        "analytics.open_interest.anomaly",
        "analytics.open_interest.features",
    }

    MINIMUM_OI_CONTEXT_KEYS: tuple[str, ...] = (
        "oi.regime.type",
        "oi.anomaly.type",
        "oi.anomaly.detected",
        "oi.features.price_delta_pct",
        "oi.features.oi_delta_pct",
        "open_interest.regime.type",
        "open_interest.anomaly.type",
    )

    CAPITULATION_ANOMALIES: set[OIAnomalyType] = {
        OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
        OIAnomalyType.SUDDEN_DELEVERAGING,
        OIAnomalyType.OI_COLLAPSE,
    }

    SUPPORTING_RISK_ANOMALIES: set[OIAnomalyType] = {
        OIAnomalyType.OVERHEATED_BUILDUP,
        OIAnomalyType.EXTREME_CROWDING,
        OIAnomalyType.FUNDING_OI_IMBALANCE,
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
                "capitulation",
                "deleveraging",
                "liquidations",
                "forced_unwind",
                "reversal",
                "risk",
                "analytics_open_interest",
            ],
            version="1.0.0",
            description=(
                "Інтерпретує capitulation / forced deleveraging context з "
                "analytics.open_interest і будує reversal signals з урахуванням "
                "liquidations, OI collapse, price shock, funding, orderflow, "
                "divergence та full OIFeatures context."
            ),
            required_features=set(self.REQUIRED_FEATURES),
            supported_regimes={
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
                MarketRegime.BREAKOUT,
                MarketRegime.SQUEEZE,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "analytics.open_interest",
                "strategy_type": "capitulation_reversal",
                "base_class": "OpenInterestStrategyBase",
                "canonical_payload": "OIAnalysisResult",
                "primary_regime": OIRegime.CAPITULATION.value,
                "uses_features": True,
                "uses_regime": True,
                "uses_anomaly": True,
                "uses_divergence": True,
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
                evaluation.reasons.append("missing_oi_capitulation_context")
                return evaluation

            evaluation.metadata.update(
                {
                    "oi_payload": payload.raw,
                    "oi_capitulation_detected": payload.detected,
                    "oi_has_capitulation_regime": payload.has_capitulation_regime,
                    "oi_has_capitulation_anomaly": payload.has_capitulation_anomaly,
                    "oi_capitulation_confidence": payload.confidence,
                    "oi_capitulation_score": payload.score,
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

            if payload.anomaly is not None:
                evaluation.metadata.update(
                    {
                        "oi_anomaly_detected": payload.anomaly.detected,
                        "oi_anomaly_type": payload.anomaly.anomaly_type.value,
                        "oi_anomaly_strength": payload.anomaly.strength.value,
                        "oi_anomaly_confidence": payload.anomaly.confidence,
                        "oi_anomaly_score": payload.anomaly.score,
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
                evaluation.reasons.append("capitulation_not_detected")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            regime_allowed, regime_reason = self.is_market_regime_allowed(context)
            if not regime_allowed:
                evaluation.reasons.append(regime_reason or "regime_not_allowed")
                return evaluation

            side = self._map_capitulation_to_side(payload)
            if side is SignalSide.UNKNOWN:
                evaluation.reasons.append("capitulation_not_directional")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            blocked_reason = self._block_reason(payload=payload, side=side)
            if blocked_reason is not None:
                evaluation.reasons.append(blocked_reason)
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            setup_type = self._map_setup_type(payload)
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
            evaluation.reasons.append("oi_capitulation_signal_generated")
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
    ) -> OICapitulationPayload | None:
        regime = self.extract_oi_regime_result(context)
        anomaly = self.extract_oi_anomaly_result(context)
        features = self.extract_oi_features(context)
        divergence = self.extract_oi_divergence_result(context)

        if regime is None and anomaly is None:
            return None

        reasons: list[str] = []

        if regime is not None:
            reasons.extend(regime.reasons)

        if anomaly is not None and anomaly.detected:
            reasons.extend(f"anomaly:{reason}" for reason in anomaly.reasons)

        if divergence is not None and divergence.detected:
            reasons.extend(f"divergence:{reason}" for reason in divergence.reasons)

        return OICapitulationPayload(
            regime=regime,
            anomaly=anomaly,
            features=features,
            divergence=divergence,
            analysis_confidence=self.oi_analysis_confidence(context),
            reasons=list(dict.fromkeys(reasons)),
            raw=self.extract_oi_domain(context),
        )

    # ------------------------------------------------------------------
    # Direction / setup mapping
    # ------------------------------------------------------------------

    def _map_capitulation_to_side(
        self,
        payload: OICapitulationPayload,
    ) -> SignalSide:
        """
        Capitulation зазвичай є reversal context:
        - сильний down flush / long liquidation -> LONG reversal;
        - blow-off / short liquidation / upside squeeze exhaustion -> SHORT reversal.
        """
        features = payload.features

        if features is None:
            return SignalSide.UNKNOWN

        if features.liquidation_imbalance is not None:
            if features.liquidation_imbalance <= -0.20:
                return SignalSide.LONG

            if features.liquidation_imbalance >= 0.20:
                return SignalSide.SHORT

        if features.price_delta_pct is not None:
            if features.price_delta_pct <= -0.75:
                return SignalSide.LONG

            if features.price_delta_pct >= 0.75:
                return SignalSide.SHORT

        if features.oi_pressure_score is not None:
            if features.oi_pressure_score <= -0.60:
                return SignalSide.LONG

            if features.oi_pressure_score >= 0.60:
                return SignalSide.SHORT

        if features.aggressive_flow_imbalance is not None:
            if features.aggressive_flow_imbalance <= -0.30:
                return SignalSide.LONG

            if features.aggressive_flow_imbalance >= 0.30:
                return SignalSide.SHORT

        return SignalSide.UNKNOWN

    @staticmethod
    def _map_setup_type(payload: OICapitulationPayload) -> SetupType:
        if payload.has_capitulation_regime:
            return SetupType.REVERSAL

        if payload.has_capitulation_anomaly:
            return SetupType.REVERSAL

        return SetupType.EXHAUSTION

    # ------------------------------------------------------------------
    # Blockers
    # ------------------------------------------------------------------

    def _block_reason(
        self,
        *,
        payload: OICapitulationPayload,
        side: SignalSide,
    ) -> str | None:
        features = payload.features

        if not payload.detected:
            return "capitulation_context_not_detected"

        if features is None:
            return "capitulation_missing_features"

        if not self._has_flush_confirmation(features):
            return "capitulation_missing_flush_confirmation"

        if side is SignalSide.LONG:
            if features.price_delta_pct is not None and features.price_delta_pct > 0.25:
                return "long_capitulation_blocked_by_positive_price_delta"

            if (
                features.liquidation_imbalance is not None
                and features.liquidation_imbalance > 0.20
            ):
                return "long_capitulation_blocked_by_short_liquidation_context"

        if side is SignalSide.SHORT:
            if features.price_delta_pct is not None and features.price_delta_pct < -0.25:
                return "short_capitulation_blocked_by_negative_price_delta"

            if (
                features.liquidation_imbalance is not None
                and features.liquidation_imbalance < -0.20
            ):
                return "short_capitulation_blocked_by_long_liquidation_context"

        divergence = payload.divergence
        if divergence is not None and divergence.detected:
            hint = self.divergence_to_side_hint(divergence.divergence_type)

            if side is SignalSide.LONG and hint == "bearish":
                return f"long_capitulation_blocked_by_divergence:{divergence.divergence_type.value}"

            if side is SignalSide.SHORT and hint == "bullish":
                return f"short_capitulation_blocked_by_divergence:{divergence.divergence_type.value}"

        return None

    @staticmethod
    def _has_flush_confirmation(features: OIFeatures) -> bool:
        if features.price_delta_pct is not None and abs(features.price_delta_pct) >= 0.75:
            return True

        if features.oi_delta_pct is not None and features.oi_delta_pct <= -0.75:
            return True

        if features.liquidation_imbalance is not None and abs(features.liquidation_imbalance) >= 0.20:
            return True

        if features.oi_zscore is not None and abs(features.oi_zscore) >= 2.0:
            return True

        if features.volume_ratio is not None and features.volume_ratio >= 1.25:
            return True

        return False

    # ------------------------------------------------------------------
    # Scoring / confidence
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        *,
        context: SignalContext,
        payload: OICapitulationPayload,
        side: SignalSide,
    ) -> float:
        market_regime = self.get_market_regime(context)
        base_score = payload.score if payload.score > 0 else payload.confidence
        score = self.weighted_score(base_score, market_regime)

        if payload.has_capitulation_regime:
            score += 0.08

        if payload.has_capitulation_anomaly:
            score += 0.08

        if payload.analysis_confidence >= 0.75:
            score += 0.04
        elif payload.analysis_confidence >= 0.50:
            score += 0.02

        if payload.features is not None:
            score += self._feature_score_adjustment(
                features=payload.features,
                side=side,
            )

        if payload.divergence is not None and payload.divergence.detected:
            hint = self.divergence_to_side_hint(payload.divergence.divergence_type)

            if side is SignalSide.LONG and hint == "bullish":
                score += min(0.06, payload.divergence.score * 0.10)
            elif side is SignalSide.SHORT and hint == "bearish":
                score += min(0.06, payload.divergence.score * 0.10)
            else:
                score -= min(0.10, payload.divergence.score * 0.12)

        return self.clamp(score)

    def _compute_confidence(
        self,
        *,
        payload: OICapitulationPayload,
        side: SignalSide,
    ) -> float:
        confidence = payload.confidence

        if payload.has_capitulation_regime:
            confidence += 0.04

        if payload.has_capitulation_anomaly:
            confidence += 0.04

        if payload.analysis_confidence >= 0.75:
            confidence += 0.04
        elif payload.analysis_confidence >= 0.50:
            confidence += 0.02

        if payload.features is not None:
            confidence += self._feature_confidence_adjustment(
                features=payload.features,
                side=side,
            )

        if payload.divergence is not None and payload.divergence.detected:
            hint = self.divergence_to_side_hint(payload.divergence.divergence_type)

            if side is SignalSide.LONG and hint == "bullish":
                confidence += min(0.04, payload.divergence.confidence * 0.06)
            elif side is SignalSide.SHORT and hint == "bearish":
                confidence += min(0.04, payload.divergence.confidence * 0.06)
            else:
                confidence -= min(0.08, payload.divergence.confidence * 0.10)

        return self.clamp(confidence)

    def _feature_score_adjustment(
        self,
        *,
        features: OIFeatures,
        side: SignalSide,
    ) -> float:
        adjustment = 0.0

        if features.price_delta_pct is not None:
            if abs(features.price_delta_pct) >= 1.50:
                adjustment += 0.07
            elif abs(features.price_delta_pct) >= 0.75:
                adjustment += 0.05

        if features.oi_delta_pct is not None:
            if features.oi_delta_pct <= -1.50:
                adjustment += 0.07
            elif features.oi_delta_pct <= -0.75:
                adjustment += 0.05

        if features.volume_ratio is not None:
            if features.volume_ratio >= 1.75:
                adjustment += 0.06
            elif features.volume_ratio >= 1.25:
                adjustment += 0.04

        if features.oi_zscore is not None:
            if abs(features.oi_zscore) >= 3.0:
                adjustment += 0.06
            elif abs(features.oi_zscore) >= 2.0:
                adjustment += 0.04

        if features.liquidation_imbalance is not None:
            if side is SignalSide.LONG and features.liquidation_imbalance <= -0.20:
                adjustment += 0.07
            elif side is SignalSide.SHORT and features.liquidation_imbalance >= 0.20:
                adjustment += 0.07

        if features.funding_rate is not None:
            if side is SignalSide.LONG and features.funding_rate < 0:
                adjustment += 0.04
            elif side is SignalSide.SHORT and features.funding_rate > 0:
                adjustment += 0.04

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
    ) -> float:
        adjustment = 0.0

        if features.price_delta_pct is not None and abs(features.price_delta_pct) >= 0.75:
            adjustment += 0.04

        if features.oi_delta_pct is not None and features.oi_delta_pct <= -0.75:
            adjustment += 0.04

        if features.volume_ratio is not None and features.volume_ratio >= 1.25:
            adjustment += 0.03

        if features.liquidation_imbalance is not None:
            if side is SignalSide.LONG and features.liquidation_imbalance <= -0.20:
                adjustment += 0.04
            elif side is SignalSide.SHORT and features.liquidation_imbalance >= 0.20:
                adjustment += 0.04

        if features.oi_zscore is not None and abs(features.oi_zscore) >= 2.0:
            adjustment += 0.03

        return adjustment

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        *,
        context: SignalContext,
        payload: OICapitulationPayload,
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
                "oi_capitulation_detected": payload.detected,
                "oi_has_capitulation_regime": payload.has_capitulation_regime,
                "oi_has_capitulation_anomaly": payload.has_capitulation_anomaly,
                "oi_capitulation_confidence": payload.confidence,
                "oi_capitulation_score": payload.score,
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

        if payload.anomaly is not None:
            signal.metadata.update(
                {
                    "oi_anomaly_detected": payload.anomaly.detected,
                    "oi_anomaly_type": payload.anomaly.anomaly_type.value,
                    "oi_anomaly_strength": payload.anomaly.strength.value,
                    "oi_anomaly_confidence": payload.anomaly.confidence,
                    "oi_anomaly_score": payload.anomaly.score,
                    "oi_anomaly_confidence_band": payload.anomaly.confidence_band.value,
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

        signal.add_reason("oi_capitulation_context_detected")
        signal.add_reason(f"setup_type:{setup_type.value}")

        if payload.regime is not None:
            signal.add_reason(f"oi_regime:{payload.regime.regime.value}")

        if payload.anomaly is not None and payload.anomaly.detected:
            signal.add_reason(f"oi_anomaly:{payload.anomaly.anomaly_type.value}")

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
        payload: OICapitulationPayload,
    ) -> list[str]:
        confirmations: list[str] = ["capitulation_context"]

        if payload.has_capitulation_regime:
            confirmations.append("capitulation_regime")

        if payload.has_capitulation_anomaly:
            confirmations.append("capitulation_anomaly")

        if payload.analysis_confidence >= 0.75:
            confirmations.append("high_confidence_oi_analysis")

        if payload.regime is not None:
            confirmations.append(f"oi_regime:{payload.regime.regime.value}")

        if payload.anomaly is not None and payload.anomaly.detected:
            confirmations.append(f"oi_anomaly:{payload.anomaly.anomaly_type.value}")

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

        if features.price_delta_pct is not None and abs(features.price_delta_pct) >= 0.75:
            confirmations.append("price_shock_confirmation")

        if features.oi_delta_pct is not None and features.oi_delta_pct <= -0.75:
            confirmations.append("oi_collapse_confirmation")

        if features.volume_ratio is not None and features.volume_ratio >= 1.25:
            confirmations.append("high_volume_flush")

        if features.oi_zscore is not None and abs(features.oi_zscore) >= 2.0:
            confirmations.append("extreme_oi_zscore_context")

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