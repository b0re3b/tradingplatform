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
class OIDivergencePayload:
    """
    Normalized strategy-level payload для OI divergence.

    Source of truth:
        analytics.open_interest.models.OIDivergenceResult

    Додатковий context:
        - OIFeatures;
        - OIRegimeResult;
        - OIAnomalyResult;
        - full OI analysis confidence через OpenInterestStrategyBase.
    """

    divergence: OIDivergenceResult
    features: OIFeatures | None = None
    regime: OIRegimeResult | None = None
    anomaly: OIAnomalyResult | None = None

    analysis_confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def divergence_type(self) -> OIDivergenceType:
        return self.divergence.divergence_type

    @property
    def detected(self) -> bool:
        return self.divergence.detected

    @property
    def confidence(self) -> float:
        return self.divergence.confidence

    @property
    def score(self) -> float:
        return self.divergence.score

    @property
    def window_size(self) -> int | None:
        return self.divergence.window_size


class OIDivergenceStrategy(OpenInterestStrategyBase):
    """
    Strategy, що інтерпретує divergence сигнали з analytics.open_interest.

    Важливо:
    - strategy не рахує divergence самостійно;
    - strategy не читає raw market data;
    - strategy працює з OIDivergenceResult, OIFeatures, OIRegimeResult,
      OIAnomalyResult, які вже підготовлені analytics.open_interest;
    - основний input — context.open_interest = OIAnalysisResult.to_dict();
    - старі oi.* feature keys підтримуються тільки через base fallback.
    """

    STRATEGY_NAME = "oi_divergence_strategy"
    DEFAULT_PRIORITY = 85

    REQUIRED_FEATURES: set[str] = {
        "analytics.open_interest.divergence",
    }

    MINIMUM_OI_CONTEXT_KEYS: tuple[str, ...] = (
        "oi.divergence.type",
        "oi.divergence.detected",
        "oi.divergence.confidence",
        "open_interest.divergence.type",
        "open_interest.divergence.detected",
        "open_interest.divergence.confidence",
    )

    HARD_BLOCKING_ANOMALIES: set[OIAnomalyType] = {
        OIAnomalyType.SUDDEN_DELEVERAGING,
        OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
        OIAnomalyType.OI_COLLAPSE,
    }

    SOFT_RISK_ANOMALIES: set[OIAnomalyType] = {
        OIAnomalyType.OVERHEATED_BUILDUP,
        OIAnomalyType.EXTREME_CROWDING,
        OIAnomalyType.FUNDING_OI_IMBALANCE,
        OIAnomalyType.OI_PRICE_DISLOCATION,
        OIAnomalyType.OI_VOLUME_DISLOCATION,
    }

    REVERSAL_REGIMES: set[OIRegime] = {
        OIRegime.TREND_EXHAUSTION,
        OIRegime.CAPITULATION,
        OIRegime.OVERHEATED,
    }

    CONTEXTUAL_REGIMES: set[OIRegime] = {
        OIRegime.SQUEEZE_SETUP,
        OIRegime.LONG_UNWIND,
        OIRegime.SHORT_COVERING,
        OIRegime.NEUTRAL,
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
                "mean_reversion",
                "contextual",
                "analytics_open_interest",
            ],
            version="2.0.0",
            description=(
                "Інтерпретує OI divergence з analytics.open_interest і будує "
                "directional strategy signals з урахуванням OI regime, anomaly, "
                "funding, liquidations, orderflow та повного OIFeatures context."
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
                "canonical_payload": "OIAnalysisResult",
                "primary_result": "OIDivergenceResult",
                "uses_features": True,
                "uses_regime": True,
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
                evaluation.reasons.append("missing_oi_divergence_result")
                return evaluation

            evaluation.metadata.update(
                {
                    "oi_payload": payload.raw,
                    "oi_divergence_type": payload.divergence_type.value,
                    "oi_divergence_detected": payload.detected,
                    "oi_divergence_confidence": payload.confidence,
                    "oi_divergence_score": payload.score,
                    "oi_divergence_window_size": payload.window_size,
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
                        "oi_anomaly_confidence": payload.anomaly.confidence,
                        "oi_anomaly_score": payload.anomaly.score,
                    }
                )

            if not payload.detected:
                evaluation.reasons.append("divergence_not_detected")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            if payload.divergence_type is OIDivergenceType.NONE:
                evaluation.reasons.append("divergence_type_none")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            regime_allowed, regime_reason = self.is_market_regime_allowed(context)
            if not regime_allowed:
                evaluation.reasons.append(regime_reason or "regime_not_allowed")
                return evaluation

            side = self._map_divergence_to_side(payload)
            if side is SignalSide.UNKNOWN:
                evaluation.reasons.append("divergence_not_directional")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            blocked_reason = self._risk_block_reason(payload=payload, side=side)
            if blocked_reason is not None:
                evaluation.reasons.append(blocked_reason)
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            setup_type = self._map_divergence_to_setup_type(payload)
            score = self._compute_score(
                context=context,
                payload=payload,
                side=side,
            )
            confidence = self._compute_confidence(
                context=context,
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

    def _extract_payload(self, context: SignalContext) -> OIDivergencePayload | None:
        divergence = self.extract_oi_divergence_result(context)
        if divergence is None:
            return None

        features = self.extract_oi_features(context)
        regime = self.extract_oi_regime_result(context)
        anomaly = self.extract_oi_anomaly_result(context)

        reasons: list[str] = []
        reasons.extend(divergence.reasons)

        if regime is not None:
            reasons.extend(f"regime:{reason}" for reason in regime.reasons)

        if anomaly is not None and anomaly.detected:
            reasons.extend(f"anomaly:{reason}" for reason in anomaly.reasons)

        raw = self.extract_oi_domain(context)

        return OIDivergencePayload(
            divergence=divergence,
            features=features,
            regime=regime,
            anomaly=anomaly,
            analysis_confidence=self.oi_analysis_confidence(context),
            reasons=list(dict.fromkeys(reasons)),
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def _map_divergence_to_side(self, payload: OIDivergencePayload) -> SignalSide:
        """
        Direction mapping базується на enum semantics із analytics.open_interest.

        Додатково використовує OIFeatures як tie-breaker для contextual cases.
        """
        divergence_type = payload.divergence_type
        semantic_hint = self.divergence_to_side_hint(divergence_type)

        if semantic_hint == "bullish":
            return SignalSide.LONG

        if semantic_hint == "bearish":
            return SignalSide.SHORT

        features = payload.features
        if features is None:
            return SignalSide.UNKNOWN

        if divergence_type in {
            OIDivergenceType.PRICE_DOWN_OI_DOWN,
            OIDivergenceType.PRICE_DOWN_OI_FLAT,
            OIDivergenceType.WEAK_BREAKOUT_DOWN,
        }:
            return SignalSide.LONG

        if divergence_type in {
            OIDivergenceType.PRICE_UP_OI_DOWN,
            OIDivergenceType.PRICE_UP_OI_FLAT,
            OIDivergenceType.WEAK_BREAKOUT_UP,
        }:
            return SignalSide.SHORT

        if features.price_delta_pct is not None:
            if features.price_delta_pct < 0 and features.oi_delta_pct is not None and features.oi_delta_pct <= 0:
                return SignalSide.LONG
            if features.price_delta_pct > 0 and features.oi_delta_pct is not None and features.oi_delta_pct <= 0:
                return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _map_divergence_to_setup_type(self, payload: OIDivergencePayload) -> SetupType:
        divergence_type = payload.divergence_type
        regime = payload.regime.regime if payload.regime is not None else None

        if divergence_type in {
            OIDivergenceType.BULLISH,
            OIDivergenceType.BEARISH,
            OIDivergenceType.PRICE_UP_OI_DOWN,
            OIDivergenceType.PRICE_UP_OI_FLAT,
            OIDivergenceType.PRICE_DOWN_OI_DOWN,
            OIDivergenceType.PRICE_DOWN_OI_FLAT,
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

        if regime in {
            OIRegime.TREND_EXHAUSTION,
            OIRegime.OVERHEATED,
            OIRegime.CAPITULATION,
        }:
            return SetupType.REVERSAL

        return SetupType.MEAN_REVERSION

    # ------------------------------------------------------------------
    # Risk / anomaly filters
    # ------------------------------------------------------------------

    def _risk_block_reason(
        self,
        *,
        payload: OIDivergencePayload,
        side: SignalSide,
    ) -> str | None:
        anomaly = payload.anomaly
        if anomaly is None or not anomaly.detected:
            return None

        if anomaly.anomaly_type not in self.HARD_BLOCKING_ANOMALIES:
            return None

        # Capitulation/deleveraging can support reversal only when the
        # directional context matches the flush direction.
        features = payload.features

        if side is SignalSide.LONG:
            if (
                features is not None
                and features.price_delta_pct is not None
                and features.price_delta_pct < 0
                and features.liquidation_imbalance is not None
                and features.liquidation_imbalance <= -0.20
            ):
                return None

        if side is SignalSide.SHORT:
            if (
                features is not None
                and features.price_delta_pct is not None
                and features.price_delta_pct > 0
                and features.liquidation_imbalance is not None
                and features.liquidation_imbalance >= 0.20
            ):
                return None

        return f"divergence_blocked_by_risk_anomaly:{anomaly.anomaly_type.value}"

    # ------------------------------------------------------------------
    # Scoring / confidence
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        *,
        context: SignalContext,
        payload: OIDivergencePayload,
        side: SignalSide,
    ) -> float:
        market_regime = self.get_market_regime(context)
        base_score = payload.score if payload.score > 0 else payload.confidence
        score = self.weighted_score(base_score, market_regime)

        features = payload.features
        regime = payload.regime
        anomaly = payload.anomaly

        if payload.window_size is not None:
            if payload.window_size >= 8:
                score += 0.04
            elif payload.window_size >= 5:
                score += 0.02

        if payload.analysis_confidence >= 0.75:
            score += 0.04
        elif payload.analysis_confidence >= 0.50:
            score += 0.02

        if regime is not None:
            if regime.regime in self.REVERSAL_REGIMES:
                score += 0.07
            elif regime.regime in self.CONTEXTUAL_REGIMES:
                score += 0.03

            if regime.confidence >= 0.75:
                score += 0.03

        if features is not None:
            score += self._feature_score_adjustment(
                features=features,
                side=side,
                divergence_type=payload.divergence_type,
            )

        if anomaly is not None and anomaly.detected:
            if anomaly.anomaly_type in self.SOFT_RISK_ANOMALIES:
                score -= min(0.10, anomaly.score * 0.10)

            if anomaly.anomaly_type in {
                OIAnomalyType.OI_PRICE_DISLOCATION,
                OIAnomalyType.OI_VOLUME_DISLOCATION,
            }:
                score += 0.02

        return self.clamp(score)

    def _compute_confidence(
        self,
        *,
        context: SignalContext,
        payload: OIDivergencePayload,
        side: SignalSide,
    ) -> float:
        confidence = payload.confidence if payload.confidence > 0 else payload.score

        features = payload.features
        regime = payload.regime
        anomaly = payload.anomaly

        if payload.analysis_confidence >= 0.75:
            confidence += 0.04
        elif payload.analysis_confidence >= 0.50:
            confidence += 0.02

        if payload.window_size is not None and payload.window_size >= 5:
            confidence += 0.02

        if regime is not None:
            if regime.regime in self.REVERSAL_REGIMES:
                confidence += 0.04
            elif regime.regime in self.CONTEXTUAL_REGIMES:
                confidence += 0.02

        if features is not None:
            confidence += self._feature_confidence_adjustment(
                features=features,
                side=side,
            )

        if anomaly is not None and anomaly.detected:
            if anomaly.anomaly_type in self.SOFT_RISK_ANOMALIES:
                confidence -= min(0.08, anomaly.confidence * 0.08)

        return self.clamp(confidence)

    def _feature_score_adjustment(
        self,
        *,
        features: OIFeatures,
        side: SignalSide,
        divergence_type: OIDivergenceType,
    ) -> float:
        adjustment = 0.0

        if features.volume_ratio is not None:
            if features.volume_ratio >= 1.50:
                adjustment += 0.05
            elif features.volume_ratio >= 1.00:
                adjustment += 0.03

        if features.oi_zscore is not None and abs(features.oi_zscore) >= 1.0:
            adjustment += 0.03

        if features.oi_price_efficiency is not None:
            if abs(features.oi_price_efficiency) < 0.50:
                adjustment += 0.05
            elif abs(features.oi_price_efficiency) < 0.80:
                adjustment += 0.03

        if features.oi_pressure_score is not None:
            if abs(features.oi_pressure_score) < 0.20:
                adjustment += 0.03

        if features.liquidation_imbalance is not None:
            if side is SignalSide.LONG and features.liquidation_imbalance <= -0.20:
                adjustment += 0.04
            elif side is SignalSide.SHORT and features.liquidation_imbalance >= 0.20:
                adjustment += 0.04
            elif abs(features.liquidation_imbalance) >= 0.20:
                adjustment += 0.02

        if features.aggressive_flow_imbalance is not None:
            if side is SignalSide.LONG and features.aggressive_flow_imbalance > -0.05:
                adjustment += 0.03
            elif side is SignalSide.SHORT and features.aggressive_flow_imbalance < 0.05:
                adjustment += 0.03

        if features.funding_rate is not None:
            if side is SignalSide.LONG and features.funding_rate < 0:
                adjustment += 0.03
            elif side is SignalSide.SHORT and features.funding_rate > 0:
                adjustment += 0.03

        if divergence_type in {
            OIDivergenceType.EXHAUSTION_UP,
            OIDivergenceType.EXHAUSTION_DOWN,
        }:
            if features.oi_price_efficiency is not None and abs(features.oi_price_efficiency) >= 1.0:
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

        if features.oi_zscore is not None and abs(features.oi_zscore) >= 1.0:
            adjustment += 0.03

        if features.oi_price_efficiency is not None and abs(features.oi_price_efficiency) < 0.50:
            adjustment += 0.03

        if side is SignalSide.LONG:
            if features.liquidation_imbalance is not None and features.liquidation_imbalance <= -0.20:
                adjustment += 0.03
            if features.funding_rate is not None and features.funding_rate < 0:
                adjustment += 0.02

        elif side is SignalSide.SHORT:
            if features.liquidation_imbalance is not None and features.liquidation_imbalance >= 0.20:
                adjustment += 0.03
            if features.funding_rate is not None and features.funding_rate > 0:
                adjustment += 0.02

        return adjustment

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
                "oi_divergence_detected": payload.detected,
                "oi_divergence_confidence": payload.confidence,
                "oi_divergence_score": payload.score,
                "oi_divergence_window_size": payload.window_size,
                "oi_divergence_confidence_band": payload.divergence.confidence_band.value,
                "oi_analysis_confidence": payload.analysis_confidence,
                "oi_raw_payload": payload.raw,
            },
        )

        signal.add_reason(f"oi_divergence_type:{payload.divergence_type.value}")
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
        payload: OIDivergencePayload,
    ) -> list[str]:
        confirmations: list[str] = [
            f"divergence:{payload.divergence_type.value}",
        ]

        if payload.window_size is not None:
            confirmations.append(f"divergence_window:{payload.window_size}")

        if payload.regime is not None:
            confirmations.append(f"oi_regime:{payload.regime.regime.value}")

            if payload.regime.regime in self.REVERSAL_REGIMES:
                confirmations.append("reversal_regime_context")

            if payload.regime.regime is OIRegime.SQUEEZE_SETUP:
                confirmations.append("squeeze_context")

            if payload.regime.confidence >= 0.75:
                confirmations.append("high_confidence_oi_regime")

        if payload.anomaly is not None and payload.anomaly.detected:
            confirmations.append(f"oi_anomaly:{payload.anomaly.anomaly_type.value}")

            if payload.anomaly.anomaly_type in self.SOFT_RISK_ANOMALIES:
                confirmations.append("soft_risk_anomaly_context")

        features = payload.features
        if features is None:
            return confirmations

        if features.volume_ratio is not None and features.volume_ratio >= 1.0:
            confirmations.append("volume_context_present")

        if features.oi_zscore is not None and abs(features.oi_zscore) >= 1.0:
            confirmations.append("oi_zscore_context")

        if features.oi_price_efficiency is not None and abs(features.oi_price_efficiency) < 0.50:
            confirmations.append("price_move_not_supported_by_oi")

        if side is SignalSide.LONG:
            if features.liquidation_imbalance is not None and features.liquidation_imbalance <= -0.20:
                confirmations.append("long_flush_context")
            if features.aggressive_flow_imbalance is not None and features.aggressive_flow_imbalance > -0.05:
                confirmations.append("sell_aggression_fading")
            if features.funding_rate is not None and features.funding_rate < 0:
                confirmations.append("negative_funding_context")

        elif side is SignalSide.SHORT:
            if features.liquidation_imbalance is not None and features.liquidation_imbalance >= 0.20:
                confirmations.append("short_flush_context")
            if features.aggressive_flow_imbalance is not None and features.aggressive_flow_imbalance < 0.05:
                confirmations.append("buy_aggression_fading")
            if features.funding_rate is not None and features.funding_rate > 0:
                confirmations.append("positive_funding_context")

        return list(dict.fromkeys(confirmations))