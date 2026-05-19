# trading_system/strategy/strategies/open_interest/oi_anomaly_strategy.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
from core.event_bus import EventBus
from core.scheduler import Scheduler
from .base import (
    OPEN_INTEREST_FEATURES,
    OpenInterestStrategyConfig,
    OpenInterestTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    anomaly_filter_reason,
    anomaly_setup_hint,
    confidence_from_components,
    contrarian_side_from_crowding,
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
    reversal_side_from_flush,
    serialize_for_metadata,
    side_from_oi_anomaly,
    side_from_oi_features,
    side_from_oi_regime,
    unit_score,
    weighted_score,
)
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
        - full OI analysis confidence через OpenInterestTradingStrategy.
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
        return bool(self.anomaly.detected)

    @property
    def confidence(self) -> float:
        return unit_score(getattr(self.anomaly, "confidence", 0.0))

    @property
    def score(self) -> float:
        return unit_score(getattr(self.anomaly, "score", self.confidence))


@dataclass(slots=True)
class OIAnomalyStrategyConfig(OpenInterestStrategyConfig):
    """
    Unified OI anomaly strategy config.

    Strategy idea:
    - read normalized OI anomaly context from StrategyContext;
    - interpret detected anomaly as reversal / continuation / squeeze /
      mean-reversion setup depending on anomaly type, regime and features;
    - use regime/divergence/features only as context;
    - return internal StrategySignal only;
    - leave routing, filtering, confluence, portfolio coordination and
      risk-ready conversion to SignalProcessor.
    """

    require_detected_result: bool = True
    require_actionable_side: bool = True
    require_fresh_anomaly: bool = True

    min_anomaly_confidence: float = 0.58
    min_anomaly_score: float = 0.35
    min_analysis_confidence_bonus_threshold: float = 0.50

    block_none_anomaly: bool = True
    block_non_actionable_anomaly: bool = True

    hard_risk_requires_directional_flush: bool = True
    block_extreme_crowding_without_reversal_context: bool = False

    opposing_divergence_penalty: float = 0.12
    aligned_divergence_bonus: float = 0.06
    regime_alignment_bonus: float = 0.05
    dislocation_bonus: float = 0.04
    capitulation_bonus: float = 0.05
    crowding_reversal_bonus: float = 0.04
    oi_spike_continuation_bonus: float = 0.04

    min_abs_oi_delta_for_spike: float = 0.0
    min_abs_price_delta_for_dislocation: float = 0.0
    liquidation_flush_threshold: float = 0.20
    crowding_pressure_threshold: float = 0.15
    flow_confirmation_threshold: float = 0.05

    score_anomaly_weight: float = 0.44
    score_features_weight: float = 0.25
    score_regime_weight: float = 0.14
    score_divergence_weight: float = 0.09
    score_analysis_weight: float = 0.05
    score_freshness_weight: float = 0.03

    confidence_anomaly_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    tag_oi_anomaly: str = "oi_anomaly"
    tag_reversal: str = "reversal"
    tag_continuation: str = "continuation"
    tag_mean_reversion: str = "mean_reversion"
    tag_squeeze: str = "squeeze"
    tag_risk: str = "risk"
    tag_crowding: str = "crowding"
    tag_dislocation: str = "dislocation"
    tag_deleveraging: str = "deleveraging"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.REVERSAL

    required_open_interest_features: tuple[str, ...] = (
        OPEN_INTEREST_FEATURES.ANOMALY,
    )

    def validate(self) -> None:
        OpenInterestStrategyConfig.validate(self)

        unit_fields = {
            "min_anomaly_confidence": self.min_anomaly_confidence,
            "min_anomaly_score": self.min_anomaly_score,
            "min_analysis_confidence_bonus_threshold": self.min_analysis_confidence_bonus_threshold,
            "opposing_divergence_penalty": self.opposing_divergence_penalty,
            "aligned_divergence_bonus": self.aligned_divergence_bonus,
            "regime_alignment_bonus": self.regime_alignment_bonus,
            "dislocation_bonus": self.dislocation_bonus,
            "capitulation_bonus": self.capitulation_bonus,
            "crowding_reversal_bonus": self.crowding_reversal_bonus,
            "oi_spike_continuation_bonus": self.oi_spike_continuation_bonus,
            "liquidation_flush_threshold": self.liquidation_flush_threshold,
            "crowding_pressure_threshold": self.crowding_pressure_threshold,
            "flow_confirmation_threshold": self.flow_confirmation_threshold,
        }

        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.min_abs_oi_delta_for_spike < 0:
            raise StrategyConfigError("min_abs_oi_delta_for_spike must be >= 0")

        if self.min_abs_price_delta_for_dislocation < 0:
            raise StrategyConfigError("min_abs_price_delta_for_dislocation must be >= 0")

        score_weights = {
            "score_anomaly_weight": self.score_anomaly_weight,
            "score_features_weight": self.score_features_weight,
            "score_regime_weight": self.score_regime_weight,
            "score_divergence_weight": self.score_divergence_weight,
            "score_analysis_weight": self.score_analysis_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }

        confidence_weights = {
            "confidence_anomaly_weight": self.confidence_anomaly_weight,
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
            "tag_oi_anomaly",
            "tag_reversal",
            "tag_continuation",
            "tag_mean_reversion",
            "tag_squeeze",
            "tag_risk",
            "tag_crowding",
            "tag_dislocation",
            "tag_deleveraging",
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


class OIAnomalyStrategy(OpenInterestTradingStrategy):
    """
    Unified OI anomaly strategy.

    Input:
        StrategyContext with FeatureSource.OPEN_INTEREST domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.open_interest.anomaly"
    category: StrategyCategory = StrategyCategory.OPEN_INTEREST
    default_setup_type: SetupType = SetupType.REVERSAL

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

    DELEVERAGING_ANOMALIES: set[OIAnomalyType] = {
        OIAnomalyType.OI_COLLAPSE,
        OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
        OIAnomalyType.SUDDEN_DELEVERAGING,
    }

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        open_interest_config: OIAnomalyStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_open_interest_config = (
            open_interest_config or OIAnomalyStrategyConfig()
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

        self.anomaly_config: OIAnomalyStrategyConfig = resolved_open_interest_config

    @property
    def strategy_name(self) -> str:
        return "oi_anomaly"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.OPEN_INTEREST,
            timeframe=Timeframe.M1,
            tags=[
                self.anomaly_config.tag_open_interest,
                self.anomaly_config.tag_oi_anomaly,
                self.anomaly_config.tag_risk,
                self.anomaly_config.tag_reversal,
                self.anomaly_config.tag_crowding,
                self.anomaly_config.tag_deleveraging,
                "analytics_open_interest",
            ],
            version="2.0.0",
            description=(
                "Інтерпретує OI anomaly з analytics.open_interest і будує "
                "internal StrategySignal з урахуванням OI regime, divergence, "
                "features і futures context."
            ),
            required_features=set(self.required_features()),
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
                "base_class": "OpenInterestTradingStrategy",
                "canonical_payload": "OIAnalysisResult",
                "primary_result": "OIAnomalyResult",
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
            self.anomaly_config.required_open_interest_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        if not self.has_any_open_interest_data(
            context,
            tuple(self.anomaly_config.required_open_interest_features),
        ):
            return None

        if self.has_stale_open_interest_features(
            context,
            tuple(self.anomaly_config.required_open_interest_features),
        ):
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        event_time = extract_event_time(payload.anomaly)
        if (
            self.anomaly_config.require_fresh_anomaly
            and is_stale(
                event_time=event_time,
                now=context.timestamp,
                stale_after_seconds=self.anomaly_config.stale_feature_max_age_seconds,
            )
        ):
            return None

        common_rejection = anomaly_filter_reason(
            payload.anomaly,
            min_confidence=self.anomaly_config.min_anomaly_confidence,
            min_score=self.anomaly_config.min_anomaly_score,
            require_detected=self.anomaly_config.require_detected_result,
            require_actionable=self.anomaly_config.block_non_actionable_anomaly,
        )
        if common_rejection is not None:
            return None

        if (
            self.anomaly_config.block_none_anomaly
            and payload.anomaly_type is OIAnomalyType.NONE
        ):
            return None

        side = self._map_anomaly_to_side(payload)
        if self.anomaly_config.require_actionable_side and not is_directional_side(side):
            return None

        blocked_reason = self._block_reason(payload=payload, side=side)
        if blocked_reason is not None:
            return None

        setup_type = self._map_anomaly_to_setup_type(payload)

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
            side=side,
            event_time=event_time,
        )

        if breakdown.score < self.anomaly_config.min_signal_score:
            return None

        if breakdown.confidence < self.anomaly_config.min_signal_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload=payload, setup_type=setup_type)

        reasons = list(
            dict.fromkeys(
                [
                    "oi_anomaly_signal",
                    f"side:{side.value}",
                    f"anomaly:{payload.anomaly_type.value}",
                    f"setup_type:{setup_type.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "open_interest_setup_family": "oi_anomaly",
            "open_interest_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "event_time": event_time.isoformat() if event_time else None,
            "anomaly": serialize_for_metadata(payload.anomaly),
            "regime": serialize_for_metadata(payload.regime),
            "divergence": serialize_for_metadata(payload.divergence),
            "features": serialize_for_metadata(payload.features),
            "raw": serialize_for_metadata(payload.raw),
            "anomaly_type": payload.anomaly_type.value,
            "anomaly_detected": payload.detected,
            "anomaly_confidence": payload.confidence,
            "anomaly_score": payload.score,
            "anomaly_strength": self._anomaly_strength(payload),
            "analysis_confidence": payload.analysis_confidence,
            "mapped_side": side.value,
            "setup_type": setup_type.value,
            "is_risk_critical": payload.anomaly_type in self.RISK_CRITICAL_ANOMALIES,
            "is_deleveraging": payload.anomaly_type in self.DELEVERAGING_ANOMALIES,
            "is_crowding": payload.anomaly_type in self.CROWDING_ANOMALIES,
            "is_dislocation": payload.anomaly_type in self.DISLOCATION_ANOMALIES,
        }

        if payload.regime is not None:
            metadata.update(
                {
                    "oi_regime": payload.regime.regime.value,
                    "oi_regime_confidence": payload.regime.confidence,
                    "oi_regime_score": payload.regime.score,
                }
            )

        if payload.divergence is not None:
            metadata.update(
                {
                    "oi_divergence_detected": payload.divergence.detected,
                    "oi_divergence_type": payload.divergence.divergence_type.value,
                    "oi_divergence_confidence": payload.divergence.confidence,
                    "oi_divergence_score": payload.divergence.score,
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
            priority=self.anomaly_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> OIAnomalyStrategyPayload | None:
        anomaly = self.extract_oi_anomaly_result(context)
        if anomaly is None:
            return None

        features = self.extract_oi_features(context)
        regime = self.extract_oi_regime_result(context)
        divergence = self.extract_oi_divergence_result(context)

        reasons: list[str] = []
        reasons.extend(extract_reasons(anomaly))

        if regime is not None:
            reasons.extend(f"regime:{reason}" for reason in extract_reasons(regime))

        if divergence is not None and divergence.detected:
            reasons.extend(
                f"divergence:{reason}"
                for reason in extract_reasons(divergence)
            )

        return OIAnomalyStrategyPayload(
            anomaly=anomaly,
            features=features,
            regime=regime,
            divergence=divergence,
            analysis_confidence=self.oi_analysis_confidence(context),
            reasons=list(dict.fromkeys(reasons)),
            raw=self.open_interest_domain(context),
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

        utility_side = side_from_oi_anomaly(
            anomaly_type,
            features=features,
            regime=regime,
        )
        if is_directional_side(utility_side):
            return utility_side

        if anomaly_type is OIAnomalyType.OI_SPIKE:
            side = side_from_oi_regime(regime, features=features)
            if is_directional_side(side):
                return side
            return side_from_oi_features(features)

        if anomaly_type in self.DELEVERAGING_ANOMALIES:
            return reversal_side_from_flush(features)

        if anomaly_type in self.CROWDING_ANOMALIES:
            return contrarian_side_from_crowding(
                features=features,
                regime=regime,
            )

        if anomaly_type is OIAnomalyType.OI_PRICE_DISLOCATION:
            return contrarian_side_from_crowding(
                features=features,
                regime=regime,
            )

        if anomaly_type is OIAnomalyType.OI_VOLUME_DISLOCATION:
            side = side_from_oi_regime(regime, features=features)
            if is_directional_side(side):
                return side
            return side_from_oi_features(features)

        return SignalSide.UNKNOWN

    def _map_anomaly_to_setup_type(
        self,
        payload: OIAnomalyStrategyPayload,
    ) -> SetupType:
        anomaly_type = payload.anomaly_type
        regime = payload.regime.regime if payload.regime is not None else None

        if anomaly_type in self.DELEVERAGING_ANOMALIES:
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

        hint = anomaly_setup_hint(anomaly_type)
        if hint == "reversal":
            return SetupType.REVERSAL
        if hint == "continuation":
            return SetupType.CONTINUATION

        return self.anomaly_config.default_setup_type

    # ------------------------------------------------------------------
    # Blockers
    # ------------------------------------------------------------------

    def _block_reason(
        self,
        *,
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
    ) -> str | None:
        if payload.anomaly_type is OIAnomalyType.NONE:
            return "anomaly_type_none"

        if (
            self.anomaly_config.hard_risk_requires_directional_flush
            and payload.anomaly_type in self.DELEVERAGING_ANOMALIES
            and not self._hard_risk_flush_supports_side(payload=payload, side=side)
        ):
            return (
                "hard_risk_anomaly_without_directional_flush:"
                f"{payload.anomaly_type.value}"
            )

        if (
            self.anomaly_config.block_extreme_crowding_without_reversal_context
            and payload.anomaly_type is OIAnomalyType.EXTREME_CROWDING
            and not self._crowding_supports_reversal(payload=payload, side=side)
        ):
            return "extreme_crowding_without_reversal_context"

        return None

    def _hard_risk_flush_supports_side(
        self,
        *,
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
    ) -> bool:
        features = payload.features
        if features is None:
            return False

        price_delta = extract_price_delta_pct(features)
        liquidation_pressure = extract_liquidation_pressure(features)
        pressure = extract_oi_pressure_score(features)

        threshold = self.anomaly_config.liquidation_flush_threshold

        if side is SignalSide.LONG:
            return (
                price_delta <= 0
                and (
                    liquidation_pressure <= -threshold
                    or pressure <= -self.anomaly_config.crowding_pressure_threshold
                )
            )

        if side is SignalSide.SHORT:
            return (
                price_delta >= 0
                and (
                    liquidation_pressure >= threshold
                    or pressure >= self.anomaly_config.crowding_pressure_threshold
                )
            )

        return False

    def _crowding_supports_reversal(
        self,
        *,
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
    ) -> bool:
        features = payload.features
        if features is None:
            return False

        regime = payload.regime.regime if payload.regime is not None else None
        regime_side = side_from_oi_regime(regime, features=features)

        if is_directional_side(regime_side) and side is self.opposite_side(regime_side):
            return True

        pressure = extract_oi_pressure_score(features)
        flow = extract_aggressive_flow_imbalance(features)

        if side is SignalSide.LONG:
            return pressure <= -self.anomaly_config.crowding_pressure_threshold or flow <= -self.anomaly_config.flow_confirmation_threshold

        if side is SignalSide.SHORT:
            return pressure >= self.anomaly_config.crowding_pressure_threshold or flow >= self.anomaly_config.flow_confirmation_threshold

        return False

    # ------------------------------------------------------------------
    # Scoring / confidence
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
        event_time: datetime | None,
    ) -> ScoreBreakdown:
        base_score = payload.score if payload.score > 0 else payload.confidence
        anomaly_score = unit_score(base_score)
        features_score = self._feature_context_score(payload=payload, side=side)
        regime_score = self._regime_context_score(payload=payload, side=side)
        divergence_score = self._divergence_context_score(payload=payload, side=side)
        analysis_score = unit_score(payload.analysis_confidence)
        fresh_score = freshness_score(
            event_time=event_time,
            now=context.timestamp,
            stale_after_seconds=self.anomaly_config.stale_feature_max_age_seconds,
        )

        components = {
            "anomaly": anomaly_score,
            "features": features_score,
            "regime": regime_score,
            "divergence": divergence_score,
            "analysis": analysis_score,
            "freshness": fresh_score,
        }
        weights = {
            "anomaly": self.anomaly_config.score_anomaly_weight,
            "features": self.anomaly_config.score_features_weight,
            "regime": self.anomaly_config.score_regime_weight,
            "divergence": self.anomaly_config.score_divergence_weight,
            "analysis": self.anomaly_config.score_analysis_weight,
            "freshness": self.anomaly_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=anomaly_score)
        confidence = confidence_from_components(
            primary=payload.confidence,
            context=regime_score,
            confirmation=max(features_score, divergence_score),
            freshness=fresh_score,
            primary_weight=self.anomaly_config.confidence_anomaly_weight,
            context_weight=self.anomaly_config.confidence_context_weight,
            confirmation_weight=self.anomaly_config.confidence_confirmation_weight,
            freshness_weight=self.anomaly_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            f"anomaly:{payload.anomaly_type.value}",
            f"side:{side.value}",
        ]

        if payload.analysis_confidence >= 0.75:
            score += 0.04
            confidence += 0.04
            confirmations.append("high_analysis_confidence")
        elif (
            payload.analysis_confidence
            >= self.anomaly_config.min_analysis_confidence_bonus_threshold
        ):
            score += 0.02
            confidence += 0.02
            confirmations.append("moderate_analysis_confidence")

        if payload.anomaly_type in self.DELEVERAGING_ANOMALIES:
            score += self.anomaly_config.capitulation_bonus
            confidence += 0.03
            confirmations.append("deleveraging_anomaly_context")

        if payload.anomaly_type in self.CROWDING_ANOMALIES:
            score += self.anomaly_config.crowding_reversal_bonus
            confirmations.append("crowding_reversal_context")

        if payload.anomaly_type in self.DISLOCATION_ANOMALIES:
            score += self.anomaly_config.dislocation_bonus
            confirmations.append("dislocation_context")

        if payload.anomaly_type is OIAnomalyType.OI_SPIKE:
            score += self.anomaly_config.oi_spike_continuation_bonus
            confirmations.append("oi_spike_continuation_context")

        if payload.regime is not None:
            confirmations.append(f"oi_regime:{payload.regime.regime.value}")

            if self._regime_aligns_with_side(payload=payload, side=side):
                score += self.anomaly_config.regime_alignment_bonus
                confidence += 0.03
                confirmations.append("regime_aligned_with_anomaly_side")

        divergence_adjustment = self._divergence_score_adjustment(
            payload=payload,
            side=side,
        )
        score += divergence_adjustment

        if divergence_adjustment < 0:
            confidence -= min(0.06, abs(divergence_adjustment))
            reasons.append("opposing_divergence_penalty")
        elif divergence_adjustment > 0:
            confidence += min(0.04, divergence_adjustment)
            confirmations.append("aligned_divergence_context")

        feature_adjustment = self._feature_score_adjustment(
            payload=payload,
            side=side,
        )
        score += feature_adjustment

        confidence += self._feature_confidence_adjustment(
            payload=payload,
            side=side,
        )

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
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
    ) -> float:
        features = payload.features
        if features is None:
            return 0.0

        components: dict[str, float] = {}

        price_delta = extract_price_delta_pct(features)
        oi_delta = extract_oi_delta_pct(features)
        pressure = extract_oi_pressure_score(features)
        flow = extract_aggressive_flow_imbalance(features)
        liquidation_pressure = extract_liquidation_pressure(features)

        components["oi_delta"] = unit_score(abs(oi_delta))
        components["price_delta"] = unit_score(abs(price_delta))
        components["pressure"] = unit_score(abs(pressure))
        components["flow"] = unit_score(abs(flow))
        components["liquidation_pressure"] = unit_score(abs(liquidation_pressure))

        volume_ratio = get_attr_or_key(features, "volume_ratio")
        if volume_ratio is not None:
            components["volume"] = unit_score(float(volume_ratio) / 2.0)

        oi_zscore = get_attr_or_key(features, "oi_zscore")
        if oi_zscore is not None:
            components["oi_zscore"] = unit_score(abs(float(oi_zscore)) / 3.0)

        if side is SignalSide.LONG:
            components["side_alignment"] = unit_score(
                max(0.0, -liquidation_pressure) + max(0.0, -pressure)
            )
        elif side is SignalSide.SHORT:
            components["side_alignment"] = unit_score(
                max(0.0, liquidation_pressure) + max(0.0, pressure)
            )

        weights = {key: 1.0 for key in components}
        return weighted_score(components, weights)

    def _regime_context_score(
        self,
        *,
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
    ) -> float:
        regime = payload.regime
        if regime is None:
            return 0.0

        score = unit_score(getattr(regime, "score", regime.confidence))

        if self._regime_aligns_with_side(payload=payload, side=side):
            return unit_score(score + 0.20)

        if regime.regime in {
            OIRegime.SQUEEZE_SETUP,
            OIRegime.TREND_EXHAUSTION,
            OIRegime.CAPITULATION,
            OIRegime.OVERHEATED,
        }:
            return unit_score(score + 0.10)

        return score

    def _divergence_context_score(
        self,
        *,
        payload: OIAnomalyStrategyPayload,
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
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
    ) -> float:
        features = payload.features
        if features is None:
            return 0.0

        adjustment = 0.0

        price_delta = extract_price_delta_pct(features)
        oi_delta = extract_oi_delta_pct(features)
        liquidation_pressure = extract_liquidation_pressure(features)
        pressure = extract_oi_pressure_score(features)
        flow = extract_aggressive_flow_imbalance(features)
        funding_rate = extract_funding_rate(features)

        if payload.anomaly_type is OIAnomalyType.OI_SPIKE:
            if abs(oi_delta) >= self.anomaly_config.min_abs_oi_delta_for_spike:
                adjustment += 0.03

        if payload.anomaly_type in self.DISLOCATION_ANOMALIES:
            if abs(price_delta) >= self.anomaly_config.min_abs_price_delta_for_dislocation:
                adjustment += 0.03

        if side is SignalSide.LONG:
            if liquidation_pressure <= -self.anomaly_config.liquidation_flush_threshold:
                adjustment += 0.04
            if pressure <= -self.anomaly_config.crowding_pressure_threshold:
                adjustment += 0.03
            if flow >= -self.anomaly_config.flow_confirmation_threshold:
                adjustment += 0.02
            if funding_rate is not None and funding_rate < 0:
                adjustment += 0.02

        elif side is SignalSide.SHORT:
            if liquidation_pressure >= self.anomaly_config.liquidation_flush_threshold:
                adjustment += 0.04
            if pressure >= self.anomaly_config.crowding_pressure_threshold:
                adjustment += 0.03
            if flow <= self.anomaly_config.flow_confirmation_threshold:
                adjustment += 0.02
            if funding_rate is not None and funding_rate > 0:
                adjustment += 0.02

        return adjustment

    def _feature_confidence_adjustment(
        self,
        *,
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
    ) -> float:
        features = payload.features
        if features is None:
            return 0.0

        adjustment = 0.0

        volume_ratio = get_attr_or_key(features, "volume_ratio")
        if volume_ratio is not None and float(volume_ratio) >= 1.0:
            adjustment += 0.02

        oi_zscore = get_attr_or_key(features, "oi_zscore")
        if oi_zscore is not None and abs(float(oi_zscore)) >= 1.0:
            adjustment += 0.03

        pressure = extract_oi_pressure_score(features)
        flow = extract_aggressive_flow_imbalance(features)

        if side is SignalSide.LONG:
            if pressure <= -self.anomaly_config.crowding_pressure_threshold:
                adjustment += 0.02
            if flow >= -self.anomaly_config.flow_confirmation_threshold:
                adjustment += 0.02

        elif side is SignalSide.SHORT:
            if pressure >= self.anomaly_config.crowding_pressure_threshold:
                adjustment += 0.02
            if flow <= self.anomaly_config.flow_confirmation_threshold:
                adjustment += 0.02

        return adjustment

    def _divergence_score_adjustment(
        self,
        *,
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
    ) -> float:
        divergence = payload.divergence
        if divergence is None or not divergence.detected:
            return 0.0

        hint = divergence_side_hint(divergence.divergence_type)
        divergence_strength = extract_score(divergence, default=extract_confidence(divergence))

        if side is SignalSide.LONG and hint == "bullish":
            return min(
                self.anomaly_config.aligned_divergence_bonus,
                divergence_strength * 0.08,
            )

        if side is SignalSide.SHORT and hint == "bearish":
            return min(
                self.anomaly_config.aligned_divergence_bonus,
                divergence_strength * 0.08,
            )

        if hint in {"bullish", "bearish"}:
            return -min(
                self.anomaly_config.opposing_divergence_penalty,
                divergence_strength * 0.16,
            )

        return 0.0

    def _regime_aligns_with_side(
        self,
        *,
        payload: OIAnomalyStrategyPayload,
        side: SignalSide,
    ) -> bool:
        if payload.regime is None:
            return False

        regime_side = side_from_oi_regime(
            payload.regime.regime,
            features=payload.features,
        )

        if not is_directional_side(regime_side):
            return payload.regime.regime in {
                OIRegime.SQUEEZE_SETUP,
                OIRegime.TREND_EXHAUSTION,
                OIRegime.CAPITULATION,
                OIRegime.OVERHEATED,
            }

        if payload.anomaly_type in self.CROWDING_ANOMALIES:
            return side is self.opposite_side(regime_side)

        return side is regime_side

    # ------------------------------------------------------------------
    # Source features / tags
    # ------------------------------------------------------------------

    def _source_features(self, payload: OIAnomalyStrategyPayload) -> list[str]:
        features = [
            OPEN_INTEREST_FEATURES.ANOMALY,
            OPEN_INTEREST_FEATURES.ANOMALY_TYPE,
            OPEN_INTEREST_FEATURES.ANOMALY_DETECTED,
            OPEN_INTEREST_FEATURES.ANOMALY_CONFIDENCE,
            OPEN_INTEREST_FEATURES.ANOMALY_SCORE,
        ]

        if payload.regime is not None:
            features.extend(
                [
                    OPEN_INTEREST_FEATURES.REGIME,
                    OPEN_INTEREST_FEATURES.REGIME_TYPE,
                    OPEN_INTEREST_FEATURES.REGIME_CONFIDENCE,
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

        return list(dict.fromkeys(features))

    def _tags(
        self,
        *,
        payload: OIAnomalyStrategyPayload,
        setup_type: SetupType,
    ) -> list[str]:
        tags = [
            self.anomaly_config.tag_open_interest,
            self.anomaly_config.tag_oi_anomaly,
            setup_type.value,
            payload.anomaly_type.value,
        ]

        if setup_type is SetupType.REVERSAL:
            tags.append(self.anomaly_config.tag_reversal)

        if setup_type is SetupType.CONTINUATION:
            tags.append(self.anomaly_config.tag_continuation)

        if setup_type is SetupType.MEAN_REVERSION:
            tags.append(self.anomaly_config.tag_mean_reversion)

        if setup_type is SetupType.SQUEEZE:
            tags.append(self.anomaly_config.tag_squeeze)

        if payload.anomaly_type in self.RISK_CRITICAL_ANOMALIES:
            tags.append(self.anomaly_config.tag_risk)

        if payload.anomaly_type in self.CROWDING_ANOMALIES:
            tags.append(self.anomaly_config.tag_crowding)

        if payload.anomaly_type in self.DISLOCATION_ANOMALIES:
            tags.append(self.anomaly_config.tag_dislocation)

        if payload.anomaly_type in self.DELEVERAGING_ANOMALIES:
            tags.append(self.anomaly_config.tag_deleveraging)

        if payload.regime is not None:
            tags.append(f"oi_regime:{payload.regime.regime.value}")

        if payload.divergence is not None and payload.divergence.detected:
            tags.append(f"oi_divergence:{payload.divergence.divergence_type.value}")

        return list(dict.fromkeys(tags))

    @staticmethod
    def _anomaly_strength(payload: OIAnomalyStrategyPayload) -> str | None:
        strength = getattr(payload.anomaly, "strength", None)

        if strength is None:
            return None

        value = getattr(strength, "value", None)
        if isinstance(value, str):
            return value

        return str(strength)