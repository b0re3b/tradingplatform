# trading_system/strategy/strategies/open_interest/oi_breakout_confirmation_strategy.py

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
from core.event_bus import EventBus
from core.scheduler import Scheduler

from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    MarketRegime,
    SetupType,
    SignalPriority,
    SignalSide,
    StrategyCategory,
    Timeframe,
)
from ...exceptions import StrategyConfigError
from ...models import StrategyContext, StrategyMetadata, StrategySignal
from .base import (
    OPEN_INTEREST_FEATURES,
    OpenInterestStrategyConfig,
    OpenInterestTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    confidence_from_components,
    divergence_side_hint,
    extract_aggressive_flow_imbalance,
    extract_confidence,
    extract_event_time,
    extract_funding_rate,
    extract_liquidation_pressure,
    extract_oi_delta_pct,
    extract_oi_pressure_score,
    extract_price_delta_pct,
    extract_reasons,
    extract_score,
    freshness_score,
    get_attr_or_key,
    is_directional_side,
    is_stale,
    parse_side,
    serialize_for_metadata,
    side_from_oi_features,
    unit_score,
    weighted_score,
)


@dataclass(slots=True)
class OIBreakoutConfirmationPayload:
    """
    Normalized strategy-level payload для OI breakout confirmation.

    Source of truth:
        analytics.open_interest:
        - OIRegimeResult;
        - OIFeatures;
        - optional OIDivergenceResult;
        - optional OIAnomalyResult.

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
        return unit_score(getattr(self.regime, "confidence", 0.0))

    @property
    def regime_score(self) -> float:
        return unit_score(getattr(self.regime, "score", self.regime_confidence))

    @property
    def divergence_type(self) -> OIDivergenceType:
        if self.divergence is None:
            return OIDivergenceType.NONE
        return self.divergence.divergence_type

    @property
    def divergence_detected(self) -> bool:
        return self.divergence is not None and bool(self.divergence.detected)

    @property
    def anomaly_detected(self) -> bool:
        return self.anomaly is not None and bool(self.anomaly.detected)


@dataclass(slots=True)
class OIBreakoutConfirmationStrategyConfig(OpenInterestStrategyConfig):
    """
    Unified OI breakout / continuation confirmation strategy config.

    Strategy idea:
    - read normalized OI regime/features context from StrategyContext;
    - confirm breakout / continuation side using OI regime and OIFeatures;
    - reject hard risk anomalies and opposing divergences;
    - build internal StrategySignal only;
    - leave routing, filtering, confluence, portfolio coordination and
      risk-ready conversion to SignalProcessor.
    """

    require_regime_result: bool = True
    require_features: bool = False
    require_actionable_side: bool = True
    require_fresh_regime: bool = True

    min_regime_confidence: float = 0.58
    min_regime_score: float = 0.35
    min_analysis_confidence_bonus_threshold: float = 0.50

    require_positive_oi_growth: bool = True
    require_positive_efficiency: bool = True

    min_oi_delta_pct: float = 0.0
    min_volume_ratio: float = 1.0
    strong_volume_ratio: float = 1.5
    min_abs_oi_zscore_for_confirmation: float = 1.0

    min_long_pressure_score: float = -0.10
    max_short_pressure_score: float = 0.10
    min_long_aggressive_flow: float = -0.10
    max_short_aggressive_flow: float = 0.10

    squeeze_funding_long_threshold: float = 0.0
    squeeze_funding_short_threshold: float = 0.0
    squeeze_liquidation_imbalance_threshold: float = 0.25

    block_hard_anomalies: bool = True
    block_extreme_crowding: bool = True
    extreme_crowding_block_confidence: float = 0.75

    soft_risk_anomaly_penalty: float = 0.10
    max_soft_risk_penalty: float = 0.12
    opposing_divergence_penalty: float = 0.14
    aligned_divergence_bonus: float = 0.05
    oi_spike_bonus: float = 0.04

    score_regime_weight: float = 0.42
    score_features_weight: float = 0.30
    score_analysis_weight: float = 0.10
    score_divergence_weight: float = 0.10
    score_freshness_weight: float = 0.08

    confidence_regime_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    tag_oi_breakout: str = "oi_breakout"
    tag_confirmation: str = "confirmation"
    tag_continuation: str = "continuation"
    tag_squeeze: str = "squeeze"
    tag_pressure: str = "pressure"
    tag_aggressive_flow: str = "aggressive_flow"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.CONTINUATION

    required_open_interest_features: tuple[str, ...] = (
        OPEN_INTEREST_FEATURES.REGIME,
        OPEN_INTEREST_FEATURES.FEATURES,
    )

    def validate(self) -> None:
        OpenInterestStrategyConfig.validate(self)

        bounded_fields = {
            "min_regime_confidence": self.min_regime_confidence,
            "min_regime_score": self.min_regime_score,
            "min_analysis_confidence_bonus_threshold": self.min_analysis_confidence_bonus_threshold,
            "min_oi_delta_pct": self.min_oi_delta_pct,
            "min_long_pressure_score": self.min_long_pressure_score,
            "max_short_pressure_score": self.max_short_pressure_score,
            "min_long_aggressive_flow": self.min_long_aggressive_flow,
            "max_short_aggressive_flow": self.max_short_aggressive_flow,
            "squeeze_liquidation_imbalance_threshold": self.squeeze_liquidation_imbalance_threshold,
            "extreme_crowding_block_confidence": self.extreme_crowding_block_confidence,
            "soft_risk_anomaly_penalty": self.soft_risk_anomaly_penalty,
            "max_soft_risk_penalty": self.max_soft_risk_penalty,
            "opposing_divergence_penalty": self.opposing_divergence_penalty,
            "aligned_divergence_bonus": self.aligned_divergence_bonus,
            "oi_spike_bonus": self.oi_spike_bonus,
        }

        for field_name, value in bounded_fields.items():
            if not -1.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between -1.0 and 1.0")

        unit_fields = {
            "min_regime_confidence": self.min_regime_confidence,
            "min_regime_score": self.min_regime_score,
            "min_analysis_confidence_bonus_threshold": self.min_analysis_confidence_bonus_threshold,
            "squeeze_liquidation_imbalance_threshold": self.squeeze_liquidation_imbalance_threshold,
            "extreme_crowding_block_confidence": self.extreme_crowding_block_confidence,
            "soft_risk_anomaly_penalty": self.soft_risk_anomaly_penalty,
            "max_soft_risk_penalty": self.max_soft_risk_penalty,
            "opposing_divergence_penalty": self.opposing_divergence_penalty,
            "aligned_divergence_bonus": self.aligned_divergence_bonus,
            "oi_spike_bonus": self.oi_spike_bonus,
        }

        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.min_volume_ratio < 0:
            raise StrategyConfigError("min_volume_ratio must be >= 0")

        if self.strong_volume_ratio < self.min_volume_ratio:
            raise StrategyConfigError("strong_volume_ratio must be >= min_volume_ratio")

        if self.min_abs_oi_zscore_for_confirmation < 0:
            raise StrategyConfigError(
                "min_abs_oi_zscore_for_confirmation must be >= 0"
            )

        score_weights = {
            "score_regime_weight": self.score_regime_weight,
            "score_features_weight": self.score_features_weight,
            "score_analysis_weight": self.score_analysis_weight,
            "score_divergence_weight": self.score_divergence_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }

        confidence_weights = {
            "confidence_regime_weight": self.confidence_regime_weight,
            "confidence_context_weight": self.confidence_context_weight,
            "confidence_confirmation_weight": self.confidence_confirmation_weight,
            "confidence_freshness_weight": self.confidence_freshness_weight,
        }

        for field_name, value in {**score_weights, **confidence_weights}.items():
            if value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if sum(score_weights.values()) <= 0:
            raise StrategyConfigError("score weights sum must be > 0")

        if sum(confidence_weights.values()) <= 0:
            raise StrategyConfigError("confidence weights sum must be > 0")

        for attr in (
            "tag_oi_breakout",
            "tag_confirmation",
            "tag_continuation",
            "tag_squeeze",
            "tag_pressure",
            "tag_aggressive_flow",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_open_interest_features:
            raise StrategyConfigError("required_open_interest_features cannot be empty")

        for feature in self.required_open_interest_features:
            if not isinstance(feature, str) or not feature.strip():
                raise StrategyConfigError(
                    "required_open_interest_features cannot contain empty feature names"
                )


class OIBreakoutConfirmationStrategy(OpenInterestTradingStrategy):
    """
    Unified OI breakout / continuation confirmation strategy.

    Input:
        StrategyContext with FeatureSource.OPEN_INTEREST domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.open_interest.breakout_confirmation"
    category: StrategyCategory = StrategyCategory.OPEN_INTEREST
    default_setup_type: SetupType = SetupType.CONTINUATION

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
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        open_interest_config: OIBreakoutConfirmationStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_open_interest_config = (
            open_interest_config or OIBreakoutConfirmationStrategyConfig()
        )
        resolved_open_interest_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            open_interest_config=resolved_open_interest_config,
            service_name=service_name,
        )

        self.breakout_config: OIBreakoutConfirmationStrategyConfig = (
            resolved_open_interest_config
        )

    @property
    def strategy_name(self) -> str:
        return "oi_breakout_confirmation"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.OPEN_INTEREST,
            timeframe=Timeframe.M1,
            tags=[
                self.breakout_config.tag_open_interest,
                self.breakout_config.tag_oi_breakout,
                self.breakout_config.tag_confirmation,
                self.breakout_config.tag_continuation,
                self.breakout_config.tag_pressure,
                self.breakout_config.tag_aggressive_flow,
                "analytics_open_interest",
            ],
            version="2.0.0",
            description=(
                "Підтверджує breakout/continuation через OI regime, OI growth, "
                "volume, pressure, aggressive flow, funding, liquidations, "
                "divergence rejection та anomaly/risk context."
            ),
            required_features=set(self.required_features()),
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
                "base_class": "OpenInterestTradingStrategy",
                "canonical_payload": "OIAnalysisResult",
                "primary_result": "OIRegimeResult",
                "uses_features": True,
                "uses_regime": True,
                "uses_divergence": True,
                "uses_anomaly": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(
            self.breakout_config.required_open_interest_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        if not self.has_any_open_interest_data(
            context,
            tuple(self.breakout_config.required_open_interest_features),
        ):
            return None

        if self.has_stale_open_interest_features(
            context,
            tuple(self.breakout_config.required_open_interest_features),
        ):
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        event_time = extract_event_time(payload.regime)
        if (
            self.breakout_config.require_fresh_regime
            and is_stale(
                event_time=event_time,
                now=context.timestamp,
                stale_after_seconds=self.breakout_config.stale_feature_max_age_seconds,
            )
        ):
            return None

        if payload.regime_confidence < self.breakout_config.min_regime_confidence:
            return None

        if payload.regime_score < self.breakout_config.min_regime_score:
            return None

        if self.breakout_config.require_features and payload.features is None:
            return None

        side = self._infer_side(payload)
        if self.breakout_config.require_actionable_side and not is_directional_side(side):
            return None

        blocked_reason = self._block_reason(payload=payload, side=side)
        if blocked_reason is not None:
            return None

        setup_type = self._infer_setup_type(payload)

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
            side=side,
            event_time=event_time,
        )

        if breakdown.score < self.breakout_config.min_signal_score:
            return None

        if breakdown.confidence < self.breakout_config.min_signal_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload=payload, setup_type=setup_type)

        reasons = list(
            dict.fromkeys(
                [
                    "oi_breakout_confirmation_signal",
                    f"side:{side.value}",
                    f"oi_regime:{payload.oi_regime.value}",
                    f"setup_type:{setup_type.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "open_interest_setup_family": "oi_breakout_confirmation",
            "open_interest_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "event_time": event_time.isoformat() if event_time else None,
            "regime": serialize_for_metadata(payload.regime),
            "features": serialize_for_metadata(payload.features),
            "divergence": serialize_for_metadata(payload.divergence),
            "anomaly": serialize_for_metadata(payload.anomaly),
            "raw": serialize_for_metadata(payload.raw),
            "oi_regime": payload.oi_regime.value,
            "oi_regime_confidence": payload.regime_confidence,
            "oi_regime_score": payload.regime_score,
            "analysis_confidence": payload.analysis_confidence,
            "explicit_side": payload.explicit_side.value,
            "mapped_side": side.value,
            "setup_type": setup_type.value,
            "divergence_detected": payload.divergence_detected,
            "divergence_type": payload.divergence_type.value,
            "anomaly_detected": payload.anomaly_detected,
        }

        if payload.divergence is not None:
            metadata.update(
                {
                    "oi_divergence_confidence": payload.divergence.confidence,
                    "oi_divergence_score": payload.divergence.score,
                }
            )

        if payload.anomaly is not None:
            metadata.update(
                {
                    "oi_anomaly_type": payload.anomaly.anomaly_type.value,
                    "oi_anomaly_confidence": payload.anomaly.confidence,
                    "oi_anomaly_score": payload.anomaly.score,
                }
            )

        return self.build_open_interest_signal(
            context=context,
            side=side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.breakout_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> OIBreakoutConfirmationPayload | None:
        regime = self.extract_oi_regime_result(context)
        if regime is None:
            return None

        features = self.extract_oi_features(context)
        divergence = self.extract_oi_divergence_result(context)
        anomaly = self.extract_oi_anomaly_result(context)
        raw = self.open_interest_domain(context)

        reasons: list[str] = []
        reasons.extend(extract_reasons(regime))

        if divergence is not None:
            reasons.extend(
                f"divergence:{reason}"
                for reason in extract_reasons(divergence)
            )

        if anomaly is not None and anomaly.detected:
            reasons.extend(
                f"anomaly:{reason}"
                for reason in extract_reasons(anomaly)
            )

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
        уже передав breakout direction у FeatureSource.OPEN_INTEREST domain.
        """
        for key in (
            "side",
            "signal_side",
            "breakout_side",
            "breakout_direction",
            "direction",
            "trend_direction",
        ):
            side = parse_side(raw.get(key))
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
                side = parse_side(metadata.get(key))
                if side is not SignalSide.UNKNOWN:
                    return side

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

        price_delta = extract_price_delta_pct(features)
        oi_delta = extract_oi_delta_pct(features)
        pressure = extract_oi_pressure_score(features)
        flow = extract_aggressive_flow_imbalance(features)

        if price_delta > 0:
            if oi_delta > self.breakout_config.min_oi_delta_pct:
                if pressure >= self.breakout_config.min_long_pressure_score:
                    if flow >= self.breakout_config.min_long_aggressive_flow:
                        return SignalSide.LONG

        if price_delta < 0:
            if oi_delta > self.breakout_config.min_oi_delta_pct:
                if pressure <= self.breakout_config.max_short_pressure_score:
                    if flow <= self.breakout_config.max_short_aggressive_flow:
                        return SignalSide.SHORT

        utility_side = side_from_oi_features(features)
        if is_directional_side(utility_side):
            return utility_side

        return SignalSide.UNKNOWN

    def _infer_squeeze_side(
        self,
        features: OIFeatures | None,
    ) -> SignalSide:
        if features is None:
            return SignalSide.UNKNOWN

        funding_rate = extract_funding_rate(features)
        if funding_rate is not None:
            if funding_rate < self.breakout_config.squeeze_funding_long_threshold:
                return SignalSide.LONG
            if funding_rate > self.breakout_config.squeeze_funding_short_threshold:
                return SignalSide.SHORT

        liquidation_pressure = extract_liquidation_pressure(features)
        threshold = self.breakout_config.squeeze_liquidation_imbalance_threshold

        if liquidation_pressure <= -threshold:
            return SignalSide.LONG

        if liquidation_pressure >= threshold:
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

        if self.breakout_config.block_hard_anomalies:
            if anomaly.anomaly_type in self.HARD_BLOCKING_ANOMALIES:
                return (
                    "breakout_blocked_by_hard_oi_anomaly:"
                    f"{anomaly.anomaly_type.value}"
                )

        if (
            self.breakout_config.block_extreme_crowding
            and anomaly.anomaly_type is OIAnomalyType.EXTREME_CROWDING
            and anomaly.confidence >= self.breakout_config.extreme_crowding_block_confidence
        ):
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

        hint = divergence_side_hint(divergence.divergence_type)

        if side is SignalSide.LONG and hint == "bearish":
            return (
                "long_breakout_rejected_by_divergence:"
                f"{divergence.divergence_type.value}"
            )

        if side is SignalSide.SHORT and hint == "bullish":
            return (
                "short_breakout_rejected_by_divergence:"
                f"{divergence.divergence_type.value}"
            )

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

        oi_delta = extract_oi_delta_pct(features)
        efficiency = get_attr_or_key(features, "oi_price_efficiency")
        price_delta = extract_price_delta_pct(features)
        pressure = extract_oi_pressure_score(features)
        flow = extract_aggressive_flow_imbalance(features)

        if self.breakout_config.require_positive_oi_growth:
            if oi_delta <= self.breakout_config.min_oi_delta_pct:
                return f"{side.value.lower()}_breakout_not_supported_by_oi_growth"

        if self.breakout_config.require_positive_efficiency:
            if efficiency is not None and float(efficiency) <= 0:
                return (
                    f"{side.value.lower()}_breakout_not_supported_by_oi_price_efficiency"
                )

        if side is SignalSide.LONG:
            if price_delta < 0:
                return "long_breakout_negative_price_delta"

            if pressure < self.breakout_config.min_long_pressure_score:
                return "long_breakout_negative_oi_pressure"

            if flow < self.breakout_config.min_long_aggressive_flow:
                return "long_breakout_negative_aggressive_flow"

        if side is SignalSide.SHORT:
            if price_delta > 0:
                return "short_breakout_positive_price_delta"

            if pressure > self.breakout_config.max_short_pressure_score:
                return "short_breakout_positive_oi_pressure"

            if flow > self.breakout_config.max_short_aggressive_flow:
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

    def _positive_breakout_support(self, features: OIFeatures) -> bool:
        volume_ratio = get_attr_or_key(features, "volume_ratio")
        oi_delta = extract_oi_delta_pct(features)
        price_delta = extract_price_delta_pct(features)
        pressure = extract_oi_pressure_score(features)
        flow = extract_aggressive_flow_imbalance(features)

        return (
            oi_delta > self.breakout_config.min_oi_delta_pct
            and price_delta >= 0
            and (
                volume_ratio is None
                or float(volume_ratio) >= self.breakout_config.min_volume_ratio
            )
            and pressure >= self.breakout_config.min_long_pressure_score
            and flow >= self.breakout_config.min_long_aggressive_flow
        )

    def _negative_breakout_support(self, features: OIFeatures) -> bool:
        volume_ratio = get_attr_or_key(features, "volume_ratio")
        oi_delta = extract_oi_delta_pct(features)
        price_delta = extract_price_delta_pct(features)
        pressure = extract_oi_pressure_score(features)
        flow = extract_aggressive_flow_imbalance(features)

        return (
            oi_delta > self.breakout_config.min_oi_delta_pct
            and price_delta <= 0
            and (
                volume_ratio is None
                or float(volume_ratio) >= self.breakout_config.min_volume_ratio
            )
            and pressure <= self.breakout_config.max_short_pressure_score
            and flow <= self.breakout_config.max_short_aggressive_flow
        )

    # ------------------------------------------------------------------
    # Scoring / confidence
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: OIBreakoutConfirmationPayload,
        side: SignalSide,
        event_time: Any,
    ) -> ScoreBreakdown:
        base_score = (
            payload.regime_score
            if payload.regime_score > 0
            else payload.regime_confidence
        )
        regime_score = unit_score(base_score)
        features_score = self._feature_context_score(payload=payload, side=side)
        analysis_score = unit_score(payload.analysis_confidence)
        divergence_score = self._divergence_context_score(payload=payload, side=side)
        fresh_score = freshness_score(
            event_time=event_time,
            now=context.timestamp,
            stale_after_seconds=self.breakout_config.stale_feature_max_age_seconds,
        )

        components = {
            "regime": regime_score,
            "features": features_score,
            "analysis": analysis_score,
            "divergence": divergence_score,
            "freshness": fresh_score,
        }
        weights = {
            "regime": self.breakout_config.score_regime_weight,
            "features": self.breakout_config.score_features_weight,
            "analysis": self.breakout_config.score_analysis_weight,
            "divergence": self.breakout_config.score_divergence_weight,
            "freshness": self.breakout_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=regime_score)
        confidence = confidence_from_components(
            primary=payload.regime_confidence,
            context=features_score,
            confirmation=divergence_score,
            freshness=fresh_score,
            primary_weight=self.breakout_config.confidence_regime_weight,
            context_weight=self.breakout_config.confidence_context_weight,
            confirmation_weight=self.breakout_config.confidence_confirmation_weight,
            freshness_weight=self.breakout_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            f"oi_regime:{payload.oi_regime.value}",
            f"side:{side.value}",
        ]

        if payload.analysis_confidence >= 0.75:
            score += 0.04
            confidence += 0.04
            confirmations.append("high_analysis_confidence")
        elif (
            payload.analysis_confidence
            >= self.breakout_config.min_analysis_confidence_bonus_threshold
        ):
            score += 0.02
            confidence += 0.02
            confirmations.append("moderate_analysis_confidence")

        if payload.oi_regime is OIRegime.SQUEEZE_SETUP:
            score += 0.04
            confidence += 0.02
            confirmations.append("squeeze_setup_context")

        if payload.oi_regime is OIRegime.TREND_CONFIRMATION:
            score += 0.03
            confidence += 0.02
            confirmations.append("trend_confirmation_context")

        feature_adjustment = self._feature_score_adjustment(
            features=payload.features,
            side=side,
        )
        score += feature_adjustment

        feature_confidence_adjustment = self._feature_confidence_adjustment(
            features=payload.features,
            side=side,
        )
        confidence += feature_confidence_adjustment

        divergence = payload.divergence
        if divergence is not None and divergence.detected:
            hint = divergence_side_hint(divergence.divergence_type)

            if side is SignalSide.LONG and hint == "bullish":
                bonus = min(
                    self.breakout_config.aligned_divergence_bonus,
                    extract_score(divergence) * 0.08,
                )
                score += bonus
                confidence += min(0.03, extract_confidence(divergence) * 0.05)
                confirmations.append("aligned_bullish_divergence")

            elif side is SignalSide.SHORT and hint == "bearish":
                bonus = min(
                    self.breakout_config.aligned_divergence_bonus,
                    extract_score(divergence) * 0.08,
                )
                score += bonus
                confidence += min(0.03, extract_confidence(divergence) * 0.05)
                confirmations.append("aligned_bearish_divergence")

            else:
                penalty = min(
                    self.breakout_config.opposing_divergence_penalty,
                    extract_score(divergence) * 0.16,
                )
                score -= penalty
                confidence -= min(0.08, extract_confidence(divergence) * 0.10)
                reasons.append("opposing_divergence_penalty")

        anomaly = payload.anomaly
        if anomaly is not None and anomaly.detected:
            confirmations.append(f"oi_anomaly:{anomaly.anomaly_type.value}")

            if anomaly.anomaly_type in self.SOFT_RISK_ANOMALIES:
                penalty = min(
                    self.breakout_config.max_soft_risk_penalty,
                    extract_score(anomaly) * self.breakout_config.soft_risk_anomaly_penalty,
                )
                score -= penalty
                confidence -= min(0.08, penalty)
                reasons.append("soft_risk_anomaly_penalty")

            if anomaly.anomaly_type is OIAnomalyType.OI_SPIKE:
                bonus = min(
                    self.breakout_config.oi_spike_bonus,
                    extract_score(anomaly) * 0.05,
                )
                score += bonus
                confidence += min(0.02, extract_confidence(anomaly) * 0.04)
                confirmations.append("oi_spike_confirms_breakout")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    def _feature_context_score(
        self,
        *,
        payload: OIBreakoutConfirmationPayload,
        side: SignalSide,
    ) -> float:
        features = payload.features
        if features is None:
            return 0.0

        components: dict[str, float] = {}

        oi_delta = extract_oi_delta_pct(features)
        components["oi_growth"] = unit_score(max(0.0, oi_delta))

        price_delta = extract_price_delta_pct(features)
        if side is SignalSide.LONG:
            components["price_alignment"] = unit_score(max(0.0, price_delta))
        elif side is SignalSide.SHORT:
            components["price_alignment"] = unit_score(abs(min(0.0, price_delta)))

        pressure = extract_oi_pressure_score(features)
        if side is SignalSide.LONG:
            components["pressure"] = unit_score((pressure + 1.0) / 2.0)
        elif side is SignalSide.SHORT:
            components["pressure"] = unit_score((1.0 - pressure) / 2.0)

        flow = extract_aggressive_flow_imbalance(features)
        if side is SignalSide.LONG:
            components["aggressive_flow"] = unit_score((flow + 1.0) / 2.0)
        elif side is SignalSide.SHORT:
            components["aggressive_flow"] = unit_score((1.0 - flow) / 2.0)

        volume_ratio = get_attr_or_key(features, "volume_ratio")
        if volume_ratio is not None:
            components["volume"] = unit_score(float(volume_ratio) / 2.0)

        oi_zscore = get_attr_or_key(features, "oi_zscore")
        if oi_zscore is not None:
            components["oi_zscore"] = unit_score(abs(float(oi_zscore)) / 3.0)

        efficiency = get_attr_or_key(features, "oi_price_efficiency")
        if efficiency is not None:
            components["efficiency"] = unit_score(abs(float(efficiency)))

        if not components:
            return 0.0

        weights = {key: 1.0 for key in components}
        return weighted_score(components, weights)

    def _divergence_context_score(
        self,
        *,
        payload: OIBreakoutConfirmationPayload,
        side: SignalSide,
    ) -> float:
        divergence = payload.divergence
        if divergence is None or not divergence.detected:
            return 0.0

        hint = divergence_side_hint(divergence.divergence_type)
        base = extract_score(divergence, default=extract_confidence(divergence))

        if side is SignalSide.LONG and hint == "bullish":
            return unit_score(base)

        if side is SignalSide.SHORT and hint == "bearish":
            return unit_score(base)

        if hint in {"bullish", "bearish"}:
            return unit_score(1.0 - base)

        return unit_score(base * 0.5)

    def _feature_score_adjustment(
        self,
        *,
        features: OIFeatures | None,
        side: SignalSide,
    ) -> float:
        if features is None:
            return 0.0

        adjustment = 0.0

        volume_ratio = get_attr_or_key(features, "volume_ratio")
        if volume_ratio is not None:
            volume = float(volume_ratio)
            if volume >= self.breakout_config.strong_volume_ratio:
                adjustment += 0.05
            elif volume >= self.breakout_config.min_volume_ratio:
                adjustment += 0.03

        oi_zscore = get_attr_or_key(features, "oi_zscore")
        if (
            oi_zscore is not None
            and abs(float(oi_zscore))
            >= self.breakout_config.min_abs_oi_zscore_for_confirmation
        ):
            adjustment += 0.03

        efficiency = get_attr_or_key(features, "oi_price_efficiency")
        if efficiency is not None and float(efficiency) > 0:
            adjustment += min(0.05, abs(float(efficiency)) * 0.04)

        pressure = extract_oi_pressure_score(features)
        if side is SignalSide.LONG and pressure >= 0.10:
            adjustment += 0.04
        elif side is SignalSide.SHORT and pressure <= -0.10:
            adjustment += 0.04

        flow = extract_aggressive_flow_imbalance(features)
        if side is SignalSide.LONG and flow >= 0.10:
            adjustment += 0.03
        elif side is SignalSide.SHORT and flow <= -0.10:
            adjustment += 0.03

        funding_rate = extract_funding_rate(features)
        if funding_rate is not None and payload_like_squeeze(side, funding_rate):
            adjustment += 0.02

        liquidation_pressure = extract_liquidation_pressure(features)
        if side is SignalSide.LONG and liquidation_pressure <= -0.20:
            adjustment += 0.02
        elif side is SignalSide.SHORT and liquidation_pressure >= 0.20:
            adjustment += 0.02

        return adjustment

    def _feature_confidence_adjustment(
        self,
        *,
        features: OIFeatures | None,
        side: SignalSide,
    ) -> float:
        if features is None:
            return 0.0

        adjustment = 0.0

        volume_ratio = get_attr_or_key(features, "volume_ratio")
        if (
            volume_ratio is not None
            and float(volume_ratio) >= self.breakout_config.min_volume_ratio
        ):
            adjustment += 0.03

        oi_zscore = get_attr_or_key(features, "oi_zscore")
        if (
            oi_zscore is not None
            and abs(float(oi_zscore))
            >= self.breakout_config.min_abs_oi_zscore_for_confirmation
        ):
            adjustment += 0.03

        efficiency = get_attr_or_key(features, "oi_price_efficiency")
        if efficiency is not None and float(efficiency) > 0:
            adjustment += 0.03

        pressure = extract_oi_pressure_score(features)
        flow = extract_aggressive_flow_imbalance(features)

        if side is SignalSide.LONG:
            if pressure >= 0.10:
                adjustment += 0.02
            if flow >= 0.10:
                adjustment += 0.02

        elif side is SignalSide.SHORT:
            if pressure <= -0.10:
                adjustment += 0.02
            if flow <= -0.10:
                adjustment += 0.02

        return adjustment

    # ------------------------------------------------------------------
    # Source features / tags
    # ------------------------------------------------------------------

    def _source_features(self, payload: OIBreakoutConfirmationPayload) -> list[str]:
        features = [
            OPEN_INTEREST_FEATURES.REGIME,
            OPEN_INTEREST_FEATURES.REGIME_TYPE,
            OPEN_INTEREST_FEATURES.REGIME_CONFIDENCE,
            OPEN_INTEREST_FEATURES.REGIME_SCORE,
        ]

        if payload.features is not None:
            features.extend(
                [
                    OPEN_INTEREST_FEATURES.FEATURES,
                    OPEN_INTEREST_FEATURES.OI_DELTA_PCT,
                    OPEN_INTEREST_FEATURES.PRICE_DELTA_PCT,
                    OPEN_INTEREST_FEATURES.OI_PRESSURE_SCORE,
                    OPEN_INTEREST_FEATURES.AGGRESSIVE_FLOW_IMBALANCE,
                ]
            )

        if payload.divergence is not None:
            features.extend(
                [
                    OPEN_INTEREST_FEATURES.DIVERGENCE,
                    OPEN_INTEREST_FEATURES.DIVERGENCE_TYPE,
                    OPEN_INTEREST_FEATURES.DIVERGENCE_CONFIDENCE,
                ]
            )

        if payload.anomaly is not None:
            features.extend(
                [
                    OPEN_INTEREST_FEATURES.ANOMALY,
                    OPEN_INTEREST_FEATURES.ANOMALY_TYPE,
                    OPEN_INTEREST_FEATURES.ANOMALY_CONFIDENCE,
                ]
            )

        return list(dict.fromkeys(features))

    def _tags(
        self,
        *,
        payload: OIBreakoutConfirmationPayload,
        setup_type: SetupType,
    ) -> list[str]:
        tags = [
            self.breakout_config.tag_open_interest,
            self.breakout_config.tag_oi_breakout,
            self.breakout_config.tag_confirmation,
            setup_type.value,
            payload.oi_regime.value,
        ]

        if setup_type is SetupType.CONTINUATION:
            tags.append(self.breakout_config.tag_continuation)

        if setup_type is SetupType.SQUEEZE:
            tags.append(self.breakout_config.tag_squeeze)

        if payload.features is not None:
            tags.extend(
                [
                    self.breakout_config.tag_pressure,
                    self.breakout_config.tag_aggressive_flow,
                ]
            )

        if payload.divergence is not None and payload.divergence.detected:
            tags.append(f"oi_divergence:{payload.divergence.divergence_type.value}")

        if payload.anomaly is not None and payload.anomaly.detected:
            tags.append(f"oi_anomaly:{payload.anomaly.anomaly_type.value}")

        return list(dict.fromkeys(tags))


def payload_like_squeeze(side: SignalSide, funding_rate: float) -> bool:
    """
    Small local helper for squeeze-like funding confirmation.

    Negative funding can support LONG squeeze continuation.
    Positive funding can support SHORT squeeze continuation.
    """
    if side is SignalSide.LONG:
        return funding_rate < 0

    if side is SignalSide.SHORT:
        return funding_rate > 0

    return False