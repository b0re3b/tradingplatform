from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analytics.open_interest.enums import (
    OIAnomalyType,
    OIDivergenceType,
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
class OIBreakoutConfirmationPayload:
    """
    Normalized strategy-level payload для OI breakout confirmation.

    Source of truth:
        analytics.open_interest:
        - OIRegimeResult
        - OIDivergenceResult
        - OIAnomalyResult
        - OIFeatures

    Стратегія не шукає breakout самостійно. Вона підтверджує directional
    breakout / continuation context через open interest, volume, pressure,
    aggressive flow, funding, liquidations, divergence і anomaly context.
    """

    regime: OIRegimeResult
    features: OIFeatures | None = None
    divergence: OIDivergenceResult | None = None
    anomaly: OIAnomalyResult | None = None

    analysis_confidence: float = 0.0
    explicit_side: SignalSide = SignalSide.UNKNOWN
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def oi_regime(self) -> OIRegime:
        return self.regime.regime

    @property
    def regime_confidence(self) -> float:
        return self.regime.confidence

    @property
    def regime_score(self) -> float:
        return self.regime.score

    @property
    def divergence_type(self) -> OIDivergenceType:
        if self.divergence is None:
            return OIDivergenceType.NONE
        return self.divergence.divergence_type

    @property
    def divergence_detected(self) -> bool:
        return self.divergence is not None and self.divergence.detected

    @property
    def anomaly_detected(self) -> bool:
        return self.anomaly is not None and self.anomaly.detected


class OIBreakoutConfirmationStrategy(OpenInterestStrategyBase):
    """
    Стратегія підтвердження breakout / continuation через Open Interest analytics.

    Важливо:
    - strategy не читає raw market data;
    - strategy не рахує OI features/regime/divergence/anomaly самостійно;
    - strategy використовує canonical context.open_interest payload від
      analytics.open_interest;
    - старі oi.* feature keys підтримуються тільки через base fallback.
    """

    STRATEGY_NAME = "oi_breakout_confirmation_strategy"
    DEFAULT_PRIORITY = 80

    REQUIRED_FEATURES: set[str] = {
        "analytics.open_interest.regime",
        "analytics.open_interest.features",
    }

    MINIMUM_OI_CONTEXT_KEYS: tuple[str, ...] = (
        "oi.regime.type",
        "oi.regime.confidence",
        "oi.features.oi_delta_pct",
        "oi.features.price_delta_pct",
        "open_interest.regime.type",
        "open_interest.regime.confidence",
    )

    LONG_CONFIRMATION_REGIMES: set[OIRegime] = {
        OIRegime.LONG_BUILDUP,
        OIRegime.SHORT_COVERING,
        OIRegime.TREND_CONFIRMATION,
        OIRegime.SQUEEZE_SETUP,
    }

    SHORT_CONFIRMATION_REGIMES: set[OIRegime] = {
        OIRegime.SHORT_BUILDUP,
        OIRegime.LONG_UNWIND,
        OIRegime.TREND_CONFIRMATION,
        OIRegime.SQUEEZE_SETUP,
    }

    HARD_BLOCKING_ANOMALIES: set[OIAnomalyType] = {
        OIAnomalyType.OI_COLLAPSE,
        OIAnomalyType.SUDDEN_DELEVERAGING,
        OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
    }

    SOFT_RISK_ANOMALIES: set[OIAnomalyType] = {
        OIAnomalyType.OVERHEATED_BUILDUP,
        OIAnomalyType.EXTREME_CROWDING,
        OIAnomalyType.FUNDING_OI_IMBALANCE,
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
                "breakout",
                "confirmation",
                "continuation",
                "pressure",
                "volume",
                "aggressive_flow",
                "analytics_open_interest",
            ],
            version="2.0.0",
            description=(
                "Підтверджує breakout/continuation через OI regime, OI growth, "
                "volume, pressure, aggressive flow, funding, liquidations, "
                "divergence rejection та anomaly/risk context."
            ),
            required_features=set(self.REQUIRED_FEATURES),
            supported_regimes={
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.BREAKOUT,
                MarketRegime.SQUEEZE,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "analytics.open_interest",
                "strategy_type": "confirmation",
                "base_class": "OpenInterestStrategyBase",
                "canonical_payload": "OIAnalysisResult",
                "primary_result": "OIRegimeResult",
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
                evaluation.reasons.append("missing_oi_regime_result")
                return evaluation

            market_regime = self.get_market_regime(context)

            evaluation.metadata.update(
                {
                    "oi_payload": payload.raw,
                    "oi_regime": payload.oi_regime.value,
                    "oi_regime_confidence": payload.regime_confidence,
                    "oi_regime_score": payload.regime_score,
                    "oi_analysis_confidence": payload.analysis_confidence,
                    "strategy_market_regime": market_regime.value,
                    "explicit_side": payload.explicit_side.value,
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

            if payload.anomaly is not None:
                evaluation.metadata.update(
                    {
                        "oi_anomaly_detected": payload.anomaly.detected,
                        "oi_anomaly_type": payload.anomaly.anomaly_type.value,
                        "oi_anomaly_confidence": payload.anomaly.confidence,
                        "oi_anomaly_score": payload.anomaly.score,
                    }
                )

            regime_allowed, regime_reason = self.is_market_regime_allowed(context)
            if not regime_allowed:
                evaluation.reasons.append(regime_reason or "regime_not_allowed")
                return evaluation

            side = self._infer_side(payload)
            if side is SignalSide.UNKNOWN:
                evaluation.reasons.append("breakout_direction_not_confirmed")
                evaluation.score = payload.regime_score
                evaluation.confidence = payload.regime_confidence
                return evaluation

            blocked_reason = self._block_reason(payload=payload, side=side)
            if blocked_reason is not None:
                evaluation.reasons.append(blocked_reason)
                evaluation.score = payload.regime_score
                evaluation.confidence = payload.regime_confidence
                return evaluation

            setup_type = self._infer_setup_type(payload)
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
            evaluation.reasons.append("oi_breakout_confirmation_signal_generated")
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
    ) -> OIBreakoutConfirmationPayload | None:
        regime = self.extract_oi_regime_result(context)
        if regime is None:
            return None

        features = self.extract_oi_features(context)
        divergence = self.extract_oi_divergence_result(context)
        anomaly = self.extract_oi_anomaly_result(context)
        raw = self.extract_oi_domain(context)

        reasons: list[str] = []
        reasons.extend(regime.reasons)

        if divergence is not None:
            reasons.extend(f"divergence:{reason}" for reason in divergence.reasons)

        if anomaly is not None and anomaly.detected:
            reasons.extend(f"anomaly:{reason}" for reason in anomaly.reasons)

        return OIBreakoutConfirmationPayload(
            regime=regime,
            features=features,
            divergence=divergence,
            anomaly=anomaly,
            analysis_confidence=self.oi_analysis_confidence(context),
            explicit_side=self._extract_explicit_side(raw),
            reasons=list(dict.fromkeys(reasons)),
            raw=raw,
        )

    def _extract_explicit_side(self, raw: dict[str, Any]) -> SignalSide:
        """
        Best-effort extraction для випадку, коли StrategyContextBuilder
        уже передав breakout direction у context.open_interest або metadata.
        """
        for key in (
            "side",
            "signal_side",
            "breakout_side",
            "breakout_direction",
            "direction",
            "trend_direction",
        ):
            value = raw.get(key)
            side = self._parse_side(value)
            if side is not SignalSide.UNKNOWN:
                return side

        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            for key in (
                "side",
                "signal_side",
                "breakout_side",
                "breakout_direction",
                "direction",
                "trend_direction",
            ):
                value = metadata.get(key)
                side = self._parse_side(value)
                if side is not SignalSide.UNKNOWN:
                    return side

        return SignalSide.UNKNOWN

    @staticmethod
    def _parse_side(value: Any) -> SignalSide:
        if isinstance(value, SignalSide):
            return value

        if value is None:
            return SignalSide.UNKNOWN

        text = str(value).strip().upper()
        if text in {"LONG", "BUY", "BULL", "BULLISH", "UP", "UPSIDE"}:
            return SignalSide.LONG

        if text in {"SHORT", "SELL", "BEAR", "BEARISH", "DOWN", "DOWNSIDE"}:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Direction / setup
    # ------------------------------------------------------------------

    def _infer_side(self, payload: OIBreakoutConfirmationPayload) -> SignalSide:
        if payload.explicit_side is not SignalSide.UNKNOWN:
            if self._side_supported_by_oi(payload=payload, side=payload.explicit_side):
                return payload.explicit_side
            return SignalSide.UNKNOWN

        regime = payload.oi_regime
        features = payload.features

        if regime is OIRegime.LONG_BUILDUP:
            return SignalSide.LONG

        if regime is OIRegime.SHORT_BUILDUP:
            return SignalSide.SHORT

        if regime is OIRegime.SHORT_COVERING:
            return SignalSide.LONG

        if regime is OIRegime.LONG_UNWIND:
            return SignalSide.SHORT

        if regime is OIRegime.TREND_CONFIRMATION:
            return self._infer_side_from_features(features)

        if regime is OIRegime.SQUEEZE_SETUP:
            return self._infer_squeeze_side(features)

        inferred = self._infer_side_from_features(features)
        if inferred is not SignalSide.UNKNOWN:
            return inferred

        return SignalSide.UNKNOWN

    def _infer_side_from_features(
        self,
        features: OIFeatures | None,
    ) -> SignalSide:
        if features is None:
            return SignalSide.UNKNOWN

        price_delta = features.price_delta_pct
        oi_delta = features.oi_delta_pct
        pressure = features.oi_pressure_score
        flow = features.aggressive_flow_imbalance

        if price_delta is not None and price_delta > 0:
            if oi_delta is not None and oi_delta > 0:
                if pressure is None or pressure >= -0.10:
                    if flow is None or flow >= -0.10:
                        return SignalSide.LONG

        if price_delta is not None and price_delta < 0:
            if oi_delta is not None and oi_delta > 0:
                if pressure is None or pressure <= 0.10:
                    if flow is None or flow <= 0.10:
                        return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _infer_squeeze_side(
        self,
        features: OIFeatures | None,
    ) -> SignalSide:
        if features is None:
            return SignalSide.UNKNOWN

        if features.funding_rate is not None:
            if features.funding_rate < 0:
                return SignalSide.LONG
            if features.funding_rate > 0:
                return SignalSide.SHORT

        if features.liquidation_imbalance is not None:
            if features.liquidation_imbalance <= -0.25:
                return SignalSide.LONG
            if features.liquidation_imbalance >= 0.25:
                return SignalSide.SHORT

        return self._infer_side_from_features(features)

    def _infer_setup_type(
        self,
        payload: OIBreakoutConfirmationPayload,
    ) -> SetupType:
        if payload.oi_regime is OIRegime.SQUEEZE_SETUP:
            return SetupType.SQUEEZE

        if payload.oi_regime in {
            OIRegime.LONG_BUILDUP,
            OIRegime.SHORT_BUILDUP,
            OIRegime.SHORT_COVERING,
            OIRegime.LONG_UNWIND,
            OIRegime.TREND_CONFIRMATION,
        }:
            return SetupType.CONTINUATION

        return SetupType.BREAKOUT

    # ------------------------------------------------------------------
    # Blockers
    # ------------------------------------------------------------------

    def _block_reason(
        self,
        *,
        payload: OIBreakoutConfirmationPayload,
        side: SignalSide,
    ) -> str | None:
        anomaly_reason = self._anomaly_block_reason(payload)
        if anomaly_reason is not None:
            return anomaly_reason

        divergence_reason = self._divergence_block_reason(payload=payload, side=side)
        if divergence_reason is not None:
            return divergence_reason

        feature_reason = self._feature_block_reason(payload=payload, side=side)
        if feature_reason is not None:
            return feature_reason

        if not self._side_supported_by_oi(payload=payload, side=side):
            return f"{side.value.lower()}_breakout_not_supported_by_oi_context"

        return None

    def _anomaly_block_reason(
        self,
        payload: OIBreakoutConfirmationPayload,
    ) -> str | None:
        anomaly = payload.anomaly
        if anomaly is None or not anomaly.detected:
            return None

        if anomaly.anomaly_type in self.HARD_BLOCKING_ANOMALIES:
            return f"breakout_blocked_by_hard_oi_anomaly:{anomaly.anomaly_type.value}"

        if anomaly.anomaly_type is OIAnomalyType.EXTREME_CROWDING:
            if anomaly.confidence >= 0.75:
                return "breakout_blocked_by_extreme_crowding"

        return None

    def _divergence_block_reason(
        self,
        *,
        payload: OIBreakoutConfirmationPayload,
        side: SignalSide,
    ) -> str | None:
        divergence = payload.divergence
        if divergence is None or not divergence.detected:
            return None

        if divergence.divergence_type is OIDivergenceType.NONE:
            return None

        hint = self.divergence_to_side_hint(divergence.divergence_type)

        if side is SignalSide.LONG and hint == "bearish":
            return f"long_breakout_rejected_by_divergence:{divergence.divergence_type.value}"

        if side is SignalSide.SHORT and hint == "bullish":
            return f"short_breakout_rejected_by_divergence:{divergence.divergence_type.value}"

        return None

    def _feature_block_reason(
        self,
        *,
        payload: OIBreakoutConfirmationPayload,
        side: SignalSide,
    ) -> str | None:
        features = payload.features
        if features is None:
            return None

        if features.oi_delta_pct is not None and features.oi_delta_pct <= 0:
            return f"{side.value.lower()}_breakout_not_supported_by_oi_growth"

        if features.oi_price_efficiency is not None and features.oi_price_efficiency <= 0:
            return f"{side.value.lower()}_breakout_not_supported_by_oi_price_efficiency"

        if side is SignalSide.LONG:
            if features.price_delta_pct is not None and features.price_delta_pct < 0:
                return "long_breakout_negative_price_delta"

            if features.oi_pressure_score is not None and features.oi_pressure_score < -0.10:
                return "long_breakout_negative_oi_pressure"

            if (
                features.aggressive_flow_imbalance is not None
                and features.aggressive_flow_imbalance < -0.10
            ):
                return "long_breakout_negative_aggressive_flow"

        if side is SignalSide.SHORT:
            if features.price_delta_pct is not None and features.price_delta_pct > 0:
                return "short_breakout_positive_price_delta"

            if features.oi_pressure_score is not None and features.oi_pressure_score > 0.10:
                return "short_breakout_positive_oi_pressure"

            if (
                features.aggressive_flow_imbalance is not None
                and features.aggressive_flow_imbalance > 0.10
            ):
                return "short_breakout_positive_aggressive_flow"

        return None

    def _side_supported_by_oi(
        self,
        *,
        payload: OIBreakoutConfirmationPayload,
        side: SignalSide,
    ) -> bool:
        regime = payload.oi_regime

        if side is SignalSide.LONG and regime in self.LONG_CONFIRMATION_REGIMES:
            return True

        if side is SignalSide.SHORT and regime in self.SHORT_CONFIRMATION_REGIMES:
            return True

        features = payload.features
        if features is None:
            return False

        if side is SignalSide.LONG:
            return self._positive_breakout_support(features)

        if side is SignalSide.SHORT:
            return self._negative_breakout_support(features)

        return False

    @staticmethod
    def _positive_breakout_support(features: OIFeatures) -> bool:
        return (
            features.oi_delta_pct is not None
            and features.oi_delta_pct > 0
            and (features.price_delta_pct is None or features.price_delta_pct >= 0)
            and (features.volume_ratio is None or features.volume_ratio >= 1.0)
            and (features.oi_pressure_score is None or features.oi_pressure_score >= 0.10)
            and (
                features.aggressive_flow_imbalance is None
                or features.aggressive_flow_imbalance >= -0.05
            )
        )

    @staticmethod
    def _negative_breakout_support(features: OIFeatures) -> bool:
        return (
            features.oi_delta_pct is not None
            and features.oi_delta_pct > 0
            and (features.price_delta_pct is None or features.price_delta_pct <= 0)
            and (features.volume_ratio is None or features.volume_ratio >= 1.0)
            and (features.oi_pressure_score is None or features.oi_pressure_score <= -0.10)
            and (
                features.aggressive_flow_imbalance is None
                or features.aggressive_flow_imbalance <= 0.05
            )
        )

    # ------------------------------------------------------------------
    # Scoring / confidence
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        *,
        context: SignalContext,
        payload: OIBreakoutConfirmationPayload,
        side: SignalSide,
    ) -> float:
        market_regime = self.get_market_regime(context)
        base_score = payload.regime_score if payload.regime_score > 0 else payload.regime_confidence
        score = self.weighted_score(base_score, market_regime)

        if payload.analysis_confidence >= 0.75:
            score += 0.04
        elif payload.analysis_confidence >= 0.50:
            score += 0.02

        features = payload.features
        if features is not None:
            score += self._feature_score_adjustment(features=features, side=side)

        divergence = payload.divergence
        if divergence is not None and divergence.detected:
            hint = self.divergence_to_side_hint(divergence.divergence_type)
            if side is SignalSide.LONG and hint == "bullish":
                score += min(0.05, divergence.score * 0.08)
            elif side is SignalSide.SHORT and hint == "bearish":
                score += min(0.05, divergence.score * 0.08)
            else:
                score -= min(0.12, divergence.score * 0.16)

        anomaly = payload.anomaly
        if anomaly is not None and anomaly.detected:
            if anomaly.anomaly_type in self.SOFT_RISK_ANOMALIES:
                score -= min(0.12, anomaly.score * 0.12)

            if anomaly.anomaly_type is OIAnomalyType.OI_SPIKE:
                score += min(0.04, anomaly.score * 0.05)

        if payload.oi_regime is OIRegime.SQUEEZE_SETUP:
            score += 0.04

        if payload.oi_regime is OIRegime.TREND_CONFIRMATION:
            score += 0.03

        return self.clamp(score)

    def _compute_confidence(
        self,
        *,
        payload: OIBreakoutConfirmationPayload,
        side: SignalSide,
    ) -> float:
        confidence = (
            payload.regime_confidence
            if payload.regime_confidence > 0
            else payload.regime_score
        )

        if payload.analysis_confidence >= 0.75:
            confidence += 0.04
        elif payload.analysis_confidence >= 0.50:
            confidence += 0.02

        features = payload.features
        if features is not None:
            confidence += self._feature_confidence_adjustment(features=features, side=side)

        divergence = payload.divergence
        if divergence is not None and divergence.detected:
            hint = self.divergence_to_side_hint(divergence.divergence_type)
            if side is SignalSide.LONG and hint == "bullish":
                confidence += min(0.03, divergence.confidence * 0.05)
            elif side is SignalSide.SHORT and hint == "bearish":
                confidence += min(0.03, divergence.confidence * 0.05)
            else:
                confidence -= min(0.10, divergence.confidence * 0.15)

        anomaly = payload.anomaly
        if anomaly is not None and anomaly.detected:
            if anomaly.anomaly_type in self.SOFT_RISK_ANOMALIES:
                confidence -= min(0.10, anomaly.confidence * 0.10)

        return self.clamp(confidence)

    def _feature_score_adjustment(
        self,
        *,
        features: OIFeatures,
        side: SignalSide,
    ) -> float:
        adjustment = 0.0

        if features.volume_ratio is not None:
            if features.volume_ratio >= 1.50:
                adjustment += 0.06
            elif features.volume_ratio >= 1.00:
                adjustment += 0.04

        if features.oi_delta_pct is not None:
            if features.oi_delta_pct >= 0.75:
                adjustment += 0.07
            elif features.oi_delta_pct >= 0.25:
                adjustment += 0.05

        if features.oi_zscore is not None and abs(features.oi_zscore) >= 1.0:
            adjustment += 0.03

        if features.oi_price_efficiency is not None:
            if features.oi_price_efficiency >= 1.0:
                adjustment += 0.06
            elif features.oi_price_efficiency >= 0.50:
                adjustment += 0.04

        if side is SignalSide.LONG:
            if features.oi_pressure_score is not None and features.oi_pressure_score >= 0.25:
                adjustment += 0.06

            if (
                features.aggressive_flow_imbalance is not None
                and features.aggressive_flow_imbalance >= 0.10
            ):
                adjustment += 0.04

            if features.funding_rate is not None and 0 <= features.funding_rate < 0.01:
                adjustment += 0.02

            if features.liquidation_imbalance is not None and features.liquidation_imbalance <= -0.20:
                adjustment += 0.03

        elif side is SignalSide.SHORT:
            if features.oi_pressure_score is not None and features.oi_pressure_score <= -0.25:
                adjustment += 0.06

            if (
                features.aggressive_flow_imbalance is not None
                and features.aggressive_flow_imbalance <= -0.10
            ):
                adjustment += 0.04

            if features.funding_rate is not None and -0.01 < features.funding_rate <= 0:
                adjustment += 0.02

            if features.liquidation_imbalance is not None and features.liquidation_imbalance >= 0.20:
                adjustment += 0.03

        return adjustment

    def _feature_confidence_adjustment(
        self,
        *,
        features: OIFeatures,
        side: SignalSide,
    ) -> float:
        adjustment = 0.0

        if features.volume_ratio is not None and features.volume_ratio >= 1.0:
            adjustment += 0.03

        if features.oi_delta_pct is not None and features.oi_delta_pct >= 0.25:
            adjustment += 0.04

        if features.oi_ma_fast is not None and features.oi_ma_slow is not None:
            if features.oi_ma_fast > features.oi_ma_slow:
                adjustment += 0.03

        if features.oi_price_efficiency is not None and features.oi_price_efficiency > 0.50:
            adjustment += 0.03

        if side is SignalSide.LONG:
            if features.oi_pressure_score is not None and features.oi_pressure_score >= 0.25:
                adjustment += 0.04

            if (
                features.aggressive_flow_imbalance is not None
                and features.aggressive_flow_imbalance >= 0.10
            ):
                adjustment += 0.03

        elif side is SignalSide.SHORT:
            if features.oi_pressure_score is not None and features.oi_pressure_score <= -0.25:
                adjustment += 0.04

            if (
                features.aggressive_flow_imbalance is not None
                and features.aggressive_flow_imbalance <= -0.10
            ):
                adjustment += 0.03

        return adjustment

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        *,
        context: SignalContext,
        payload: OIBreakoutConfirmationPayload,
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
            trigger_type=TriggerType.CONFIRMATION,
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=self.priority_from_score(score),
            regime=self.get_market_regime(context),
            metadata={
                "oi_regime": payload.oi_regime.value,
                "oi_regime_confidence": payload.regime_confidence,
                "oi_regime_score": payload.regime_score,
                "oi_regime_confidence_band": payload.regime.confidence_band.value,
                "oi_analysis_confidence": payload.analysis_confidence,
                "oi_raw_payload": payload.raw,
            },
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

        signal.add_reason(f"oi_regime:{payload.oi_regime.value}")
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
        payload: OIBreakoutConfirmationPayload,
    ) -> list[str]:
        confirmations: list[str] = [
            f"regime:{payload.oi_regime.value}",
        ]

        if payload.analysis_confidence >= 0.75:
            confirmations.append("high_confidence_oi_analysis")

        if payload.regime_confidence >= 0.75:
            confirmations.append("high_confidence_oi_regime")

        if payload.oi_regime is OIRegime.TREND_CONFIRMATION:
            confirmations.append("trend_confirmation_regime")

        if payload.oi_regime is OIRegime.SQUEEZE_SETUP:
            confirmations.append("squeeze_setup_regime")

        if payload.divergence is not None and payload.divergence.detected:
            hint = self.divergence_to_side_hint(payload.divergence.divergence_type)

            if side is SignalSide.LONG and hint == "bullish":
                confirmations.append("bullish_divergence_support")
            elif side is SignalSide.SHORT and hint == "bearish":
                confirmations.append("bearish_divergence_support")
            else:
                confirmations.append("divergence_risk_context")

        if payload.anomaly is not None and payload.anomaly.detected:
            confirmations.append(f"oi_anomaly_context:{payload.anomaly.anomaly_type.value}")

            if payload.anomaly.anomaly_type is OIAnomalyType.OI_SPIKE:
                confirmations.append("oi_spike_confirmation")

            if payload.anomaly.anomaly_type in self.SOFT_RISK_ANOMALIES:
                confirmations.append("soft_risk_anomaly_context")

        features = payload.features
        if features is None:
            return list(dict.fromkeys(confirmations))

        if features.volume_ratio is not None and features.volume_ratio >= 1.0:
            confirmations.append("volume_confirmation")

        if features.oi_delta_pct is not None and features.oi_delta_pct >= 0.25:
            confirmations.append("oi_build_confirmation")

        if features.oi_ma_fast is not None and features.oi_ma_slow is not None:
            if features.oi_ma_fast > features.oi_ma_slow:
                confirmations.append("fast_oi_above_slow_oi")

        if features.oi_price_efficiency is not None and features.oi_price_efficiency > 0.50:
            confirmations.append("price_move_supported_by_oi")

        if features.oi_zscore is not None and abs(features.oi_zscore) >= 1.0:
            confirmations.append("oi_zscore_context")

        if side is SignalSide.LONG:
            if features.oi_pressure_score is not None and features.oi_pressure_score >= 0.25:
                confirmations.append("positive_pressure_confirmation")

            if (
                features.aggressive_flow_imbalance is not None
                and features.aggressive_flow_imbalance >= 0.10
            ):
                confirmations.append("aggressive_buy_confirmation")

            if features.funding_rate is not None and 0 <= features.funding_rate < 0.01:
                confirmations.append("healthy_long_funding_context")

            if features.liquidation_imbalance is not None and features.liquidation_imbalance <= -0.20:
                confirmations.append("short_liquidation_support")

        elif side is SignalSide.SHORT:
            if features.oi_pressure_score is not None and features.oi_pressure_score <= -0.25:
                confirmations.append("negative_pressure_confirmation")

            if (
                features.aggressive_flow_imbalance is not None
                and features.aggressive_flow_imbalance <= -0.10
            ):
                confirmations.append("aggressive_sell_confirmation")

            if features.funding_rate is not None and -0.01 < features.funding_rate <= 0:
                confirmations.append("healthy_short_funding_context")

            if features.liquidation_imbalance is not None and features.liquidation_imbalance >= 0.20:
                confirmations.append("long_liquidation_support")

        return list(dict.fromkeys(confirmations))