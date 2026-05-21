# trading_system/strategy/strategies/open_interest/oi_capitulation_strategy.py

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
from core.event_bus import EventBus
from core.scheduler import Scheduler
from .base import (
    OPEN_INTEREST_FEATURES,
    OpenInterestStrategyConfig,
    OpenInterestTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
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
class OICapitulationPayload:
    """
    Normalized strategy-level payload для OI capitulation.

    Source of truth:
        analytics.open_interest:
        - OIRegimeResult(regime=CAPITULATION);
        - OIAnomalyResult із forced deleveraging / liquidation-driven anomaly;
        - OIFeatures;
        - optional OIDivergenceResult.

    Стратегія не детектить capitulation самостійно. Вона інтерпретує готовий
    analytics context і будує reversal/risk signal лише коли capitulation має
    достатній directional context.
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
            and bool(self.anomaly.detected)
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
            values.append(unit_score(self.regime.confidence))

        if self.anomaly is not None and self.anomaly.detected:
            values.append(unit_score(self.anomaly.confidence))

        if self.analysis_confidence > 0:
            values.append(unit_score(self.analysis_confidence))

        if not values:
            return 0.0

        return unit_score(sum(values) / len(values))

    @property
    def score(self) -> float:
        values: list[float] = []

        if self.regime is not None:
            values.append(extract_score(self.regime, default=self.regime.confidence))

        if self.anomaly is not None and self.anomaly.detected:
            values.append(extract_score(self.anomaly, default=self.anomaly.confidence))

        if not values:
            return self.confidence

        return unit_score(max(values))


@dataclass(slots=True)
class OICapitulationStrategyConfig(OpenInterestStrategyConfig):
    """
    Unified OI capitulation / forced deleveraging reversal strategy config.

    Strategy idea:
    - read normalized OI regime/anomaly/features context from StrategyContext;
    - accept capitulation regime or liquidation-driven/deleveraging anomaly;
    - generate reversal signal after forced unwind / OI collapse;
    - use divergence, funding, liquidations and flow only as context;
    - return internal StrategySignal only;
    - leave routing, filtering, confluence, portfolio coordination and
      risk-ready conversion to SignalProcessor.
    """

    require_detected_context: bool = True
    require_actionable_side: bool = True
    require_fresh_capitulation: bool = True
    require_features_for_direction: bool = True

    min_capitulation_confidence: float = 0.62
    min_capitulation_score: float = 0.40
    min_analysis_confidence_bonus_threshold: float = 0.50

    min_regime_confidence: float = 0.55
    min_anomaly_confidence: float = 0.55

    min_abs_price_delta_for_flush: float = 0.0
    min_abs_oi_delta_for_flush: float = 0.0
    liquidation_flush_threshold: float = 0.20
    pressure_flush_threshold: float = 0.15
    flow_stabilization_threshold: float = 0.05

    allow_regime_only_capitulation: bool = True
    allow_anomaly_only_capitulation: bool = True
    allow_divergence_confirmation: bool = True

    aligned_divergence_bonus: float = 0.06
    opposing_divergence_penalty: float = 0.10
    capitulation_regime_bonus: float = 0.08
    capitulation_anomaly_bonus: float = 0.08
    liquidation_flush_bonus: float = 0.06
    funding_extreme_bonus: float = 0.03
    stabilization_bonus: float = 0.04

    score_capitulation_weight: float = 0.46
    score_features_weight: float = 0.28
    score_divergence_weight: float = 0.10
    score_analysis_weight: float = 0.08
    score_freshness_weight: float = 0.08

    confidence_capitulation_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    tag_oi_capitulation: str = "oi_capitulation"
    tag_capitulation: str = "capitulation"
    tag_deleveraging: str = "deleveraging"
    tag_liquidations: str = "liquidations"
    tag_forced_unwind: str = "forced_unwind"
    tag_reversal: str = "reversal"
    tag_risk: str = "risk"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.REVERSAL

    required_open_interest_features: tuple[str, ...] = (
        OPEN_INTEREST_FEATURES.REGIME,
        OPEN_INTEREST_FEATURES.ANOMALY,
        OPEN_INTEREST_FEATURES.FEATURES,
    )

    def validate(self) -> None:
        OpenInterestStrategyConfig.validate(self)

        unit_fields = {
            "min_capitulation_confidence": self.min_capitulation_confidence,
            "min_capitulation_score": self.min_capitulation_score,
            "min_analysis_confidence_bonus_threshold": self.min_analysis_confidence_bonus_threshold,
            "min_regime_confidence": self.min_regime_confidence,
            "min_anomaly_confidence": self.min_anomaly_confidence,
            "liquidation_flush_threshold": self.liquidation_flush_threshold,
            "pressure_flush_threshold": self.pressure_flush_threshold,
            "flow_stabilization_threshold": self.flow_stabilization_threshold,
            "aligned_divergence_bonus": self.aligned_divergence_bonus,
            "opposing_divergence_penalty": self.opposing_divergence_penalty,
            "capitulation_regime_bonus": self.capitulation_regime_bonus,
            "capitulation_anomaly_bonus": self.capitulation_anomaly_bonus,
            "liquidation_flush_bonus": self.liquidation_flush_bonus,
            "funding_extreme_bonus": self.funding_extreme_bonus,
            "stabilization_bonus": self.stabilization_bonus,
        }

        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.min_abs_price_delta_for_flush < 0:
            raise StrategyConfigError("min_abs_price_delta_for_flush must be >= 0")

        if self.min_abs_oi_delta_for_flush < 0:
            raise StrategyConfigError("min_abs_oi_delta_for_flush must be >= 0")

        score_weights = {
            "score_capitulation_weight": self.score_capitulation_weight,
            "score_features_weight": self.score_features_weight,
            "score_divergence_weight": self.score_divergence_weight,
            "score_analysis_weight": self.score_analysis_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }

        confidence_weights = {
            "confidence_capitulation_weight": self.confidence_capitulation_weight,
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
            "tag_oi_capitulation",
            "tag_capitulation",
            "tag_deleveraging",
            "tag_liquidations",
            "tag_forced_unwind",
            "tag_reversal",
            "tag_risk",
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


class OICapitulationStrategy(OpenInterestTradingStrategy):
    """
    Unified OI capitulation / forced deleveraging strategy.

    Input:
        StrategyContext with FeatureSource.OPEN_INTEREST domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.open_interest.capitulation"
    category: StrategyCategory = StrategyCategory.OPEN_INTEREST
    default_setup_type: SetupType = SetupType.REVERSAL

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
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        open_interest_config: OICapitulationStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_open_interest_config = (
            open_interest_config or OICapitulationStrategyConfig()
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

        self.capitulation_config: OICapitulationStrategyConfig = (
            resolved_open_interest_config
        )

    @property
    def strategy_name(self) -> str:
        return "oi_capitulation"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.OPEN_INTEREST,
            timeframe=Timeframe.M1,
            tags=[
                self.capitulation_config.tag_open_interest,
                self.capitulation_config.tag_oi_capitulation,
                self.capitulation_config.tag_capitulation,
                self.capitulation_config.tag_deleveraging,
                self.capitulation_config.tag_liquidations,
                self.capitulation_config.tag_forced_unwind,
                self.capitulation_config.tag_reversal,
                self.capitulation_config.tag_risk,
                "analytics_open_interest",
            ],
            version="2.0.0",
            description=(
                "Інтерпретує OI capitulation / forced deleveraging context з "
                "analytics.open_interest і будує internal reversal StrategySignal "
                "з урахуванням OI collapse, liquidation pressure, funding, "
                "divergence та futures context."
            ),
            required_features=set(self.required_features()),
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
                "base_class": "OpenInterestTradingStrategy",
                "canonical_payload": "OIAnalysisResult",
                "primary_regime": OIRegime.CAPITULATION.value,
                "primary_anomalies": [
                    item.value for item in sorted(
                        self.CAPITULATION_ANOMALIES,
                        key=lambda enum_item: enum_item.value,
                    )
                ],
                "uses_features": True,
                "uses_regime": True,
                "uses_anomaly": True,
                "uses_divergence": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(
            self.capitulation_config.required_open_interest_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        if not self.has_any_open_interest_data(
            context,
            tuple(self.capitulation_config.required_open_interest_features),
        ):
            self.remember_no_signal(
                "missing_open_interest_capitulation_contract",
                open_interest_domain_keys=sorted(self.open_interest_domain(context).keys()),
                required_features=sorted(self.required_features()),
            )
            return None

        if self.has_stale_open_interest_features(
            context,
            tuple(self.capitulation_config.required_open_interest_features),
        ):
            self.remember_no_signal(
                "stale_open_interest_capitulation_features",
                required_features=sorted(self.required_features()),
            )
            return None

        payload = self._extract_payload(context)
        if payload is None:
            self.remember_no_signal(
                "open_interest_capitulation_payload_not_resolved",
                open_interest_domain=self.open_interest_domain(context),
                required_features=sorted(self.required_features()),
            )
            return None

        event_time = self._event_time(payload)
        if (
            self.capitulation_config.require_fresh_capitulation
            and is_stale(
                event_time=event_time,
                now=context.timestamp,
                stale_after_seconds=self.capitulation_config.stale_feature_max_age_seconds,
            )
        ):
            self.remember_no_signal(
                "stale_open_interest_capitulation",
                event_time=event_time.isoformat() if event_time else None,
                context_timestamp=context.timestamp.isoformat(),
                stale_after_seconds=self.capitulation_config.stale_feature_max_age_seconds,
            )
            return None

        if self.capitulation_config.require_detected_context and not payload.detected:
            self.remember_no_signal(
                "open_interest_capitulation_not_detected",
                regime=serialize_for_metadata(payload.regime),
                anomaly=serialize_for_metadata(payload.anomaly),
            )
            return None

        if not self._passes_capitulation_thresholds(payload):
            self.remember_no_signal(
                "open_interest_capitulation_thresholds_failed",
                regime=serialize_for_metadata(payload.regime),
                anomaly=serialize_for_metadata(payload.anomaly),
                confidence=payload.confidence,
                score=payload.score,
                min_capitulation_confidence=self.capitulation_config.min_capitulation_confidence,
                min_capitulation_score=self.capitulation_config.min_capitulation_score,
                min_regime_confidence=self.capitulation_config.min_regime_confidence,
                min_anomaly_confidence=self.capitulation_config.min_anomaly_confidence,
            )
            return None

        side = self._map_capitulation_to_side(payload)
        if self.capitulation_config.require_actionable_side and not is_directional_side(side):
            self.remember_no_signal(
                "open_interest_capitulation_side_not_directional",
                regime=serialize_for_metadata(payload.regime),
                anomaly=serialize_for_metadata(payload.anomaly),
                features=serialize_for_metadata(payload.features),
                divergence=serialize_for_metadata(payload.divergence),
            )
            return None

        blocked_reason = self._block_reason(payload=payload, side=side)
        if blocked_reason is not None:
            self.remember_no_signal(
                "open_interest_capitulation_blocked",
                blocked_reason=blocked_reason,
                side=side.value,
                regime=serialize_for_metadata(payload.regime),
                anomaly=serialize_for_metadata(payload.anomaly),
                features=serialize_for_metadata(payload.features),
                divergence=serialize_for_metadata(payload.divergence),
            )
            return None

        setup_type = self._map_setup_type(payload)

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
            side=side,
            event_time=event_time,
        )

        if breakdown.score < self.capitulation_config.min_signal_score:
            self.remember_no_signal(
                "open_interest_capitulation_score_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_signal_score=self.capitulation_config.min_signal_score,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        if breakdown.confidence < self.capitulation_config.min_signal_confidence:
            self.remember_no_signal(
                "open_interest_capitulation_confidence_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_signal_confidence=self.capitulation_config.min_signal_confidence,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload=payload, setup_type=setup_type)

        reasons = list(
            dict.fromkeys(
                [
                    "oi_capitulation_signal",
                    f"side:{side.value}",
                    f"setup_type:{setup_type.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "open_interest_setup_family": "oi_capitulation",
            "open_interest_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "event_time": event_time.isoformat() if event_time else None,
            "regime": serialize_for_metadata(payload.regime),
            "anomaly": serialize_for_metadata(payload.anomaly),
            "features": serialize_for_metadata(payload.features),
            "divergence": serialize_for_metadata(payload.divergence),
            "raw": serialize_for_metadata(payload.raw),
            "capitulation_detected": payload.detected,
            "has_capitulation_regime": payload.has_capitulation_regime,
            "has_capitulation_anomaly": payload.has_capitulation_anomaly,
            "capitulation_confidence": payload.confidence,
            "capitulation_score": payload.score,
            "analysis_confidence": payload.analysis_confidence,
            "mapped_side": side.value,
            "setup_type": setup_type.value,
        }

        if payload.regime is not None:
            metadata.update(
                {
                    "oi_regime": payload.regime.regime.value,
                    "oi_regime_confidence": payload.regime.confidence,
                    "oi_regime_score": payload.regime.score,
                }
            )

        if payload.anomaly is not None:
            metadata.update(
                {
                    "oi_anomaly_detected": payload.anomaly.detected,
                    "oi_anomaly_type": payload.anomaly.anomaly_type.value,
                    "oi_anomaly_confidence": payload.anomaly.confidence,
                    "oi_anomaly_score": payload.anomaly.score,
                    "oi_anomaly_strength": self._anomaly_strength(payload),
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
            priority=self.capitulation_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> OICapitulationPayload | None:
        regime = self.extract_oi_regime_result(context)
        anomaly = self.extract_oi_anomaly_result(context)
        features = self.extract_oi_features(context)
        divergence = self.extract_oi_divergence_result(context)

        if regime is None and anomaly is None:
            return None

        reasons: list[str] = []

        if regime is not None:
            reasons.extend(extract_reasons(regime))

        if anomaly is not None and anomaly.detected:
            reasons.extend(
                f"anomaly:{reason}"
                for reason in extract_reasons(anomaly)
            )

        if divergence is not None and divergence.detected:
            reasons.extend(
                f"divergence:{reason}"
                for reason in extract_reasons(divergence)
            )

        return OICapitulationPayload(
            regime=regime,
            anomaly=anomaly,
            features=features,
            divergence=divergence,
            analysis_confidence=self.oi_analysis_confidence(context),
            reasons=list(dict.fromkeys(reasons)),
            raw=self.open_interest_domain(context),
        )

    def _event_time(self, payload: OICapitulationPayload) -> Any:
        if payload.anomaly is not None:
            anomaly_time = extract_event_time(payload.anomaly)
            if anomaly_time is not None:
                return anomaly_time

        if payload.regime is not None:
            regime_time = extract_event_time(payload.regime)
            if regime_time is not None:
                return regime_time

        return None

    # ------------------------------------------------------------------
    # Detection / threshold checks
    # ------------------------------------------------------------------

    def _passes_capitulation_thresholds(self, payload: OICapitulationPayload) -> bool:
        if not payload.detected:
            return False

        if payload.confidence < self.capitulation_config.min_capitulation_confidence:
            return False

        if payload.score < self.capitulation_config.min_capitulation_score:
            return False

        if payload.regime is not None:
            if payload.regime.regime is OIRegime.CAPITULATION:
                if payload.regime.confidence < self.capitulation_config.min_regime_confidence:
                    return False

        if payload.anomaly is not None and payload.anomaly.detected:
            if payload.anomaly.anomaly_type in self.CAPITULATION_ANOMALIES:
                if payload.anomaly.confidence < self.capitulation_config.min_anomaly_confidence:
                    return False

        if payload.has_capitulation_regime and not self.capitulation_config.allow_regime_only_capitulation:
            return payload.has_capitulation_anomaly

        if payload.has_capitulation_anomaly and not self.capitulation_config.allow_anomaly_only_capitulation:
            return payload.has_capitulation_regime

        return True

    # ------------------------------------------------------------------
    # Direction / setup mapping
    # ------------------------------------------------------------------

    def _map_capitulation_to_side(
        self,
        payload: OICapitulationPayload,
    ) -> SignalSide:
        """
        Capitulation зазвичай є reversal context:
        - downside OI flush / long liquidation pressure -> LONG;
        - upside squeeze exhaustion / short liquidation pressure -> SHORT.
        """
        features = payload.features

        if features is not None:
            side = reversal_side_from_flush(features)
            if is_directional_side(side):
                return side

        if payload.anomaly is not None and payload.anomaly.detected:
            side = self._side_from_capitulation_anomaly(payload)
            if is_directional_side(side):
                return side

        if payload.regime is not None:
            regime_side = side_from_oi_regime(
                payload.regime.regime,
                features=payload.features,
            )
            if is_directional_side(regime_side):
                return self.opposite_side(regime_side)

        if payload.divergence is not None and payload.divergence.detected:
            side = self._side_from_divergence(payload.divergence)
            if is_directional_side(side):
                return side

        return SignalSide.UNKNOWN

    def _side_from_capitulation_anomaly(
        self,
        payload: OICapitulationPayload,
    ) -> SignalSide:
        anomaly = payload.anomaly
        if anomaly is None:
            return SignalSide.UNKNOWN

        features = payload.features

        if anomaly.anomaly_type in self.CAPITULATION_ANOMALIES:
            side = reversal_side_from_flush(features)
            if is_directional_side(side):
                return side

        if anomaly.anomaly_type in self.SUPPORTING_RISK_ANOMALIES:
            return contrarian_side_from_crowding(
                features=features,
                regime=payload.regime.regime if payload.regime is not None else None,
            )

        return SignalSide.UNKNOWN

    def _side_from_divergence(
        self,
        divergence: OIDivergenceResult,
    ) -> SignalSide:
        hint = divergence_side_hint(divergence.divergence_type)

        if hint == "bullish":
            return SignalSide.LONG

        if hint == "bearish":
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _map_setup_type(
        self,
        payload: OICapitulationPayload,
    ) -> SetupType:
        if payload.has_capitulation_regime or payload.has_capitulation_anomaly:
            return SetupType.REVERSAL

        if payload.regime is not None and payload.regime.regime is OIRegime.SQUEEZE_SETUP:
            return SetupType.SQUEEZE

        return self.capitulation_config.default_setup_type

    # ------------------------------------------------------------------
    # Blockers
    # ------------------------------------------------------------------

    def _block_reason(
        self,
        *,
        payload: OICapitulationPayload,
        side: SignalSide,
    ) -> str | None:
        if self.capitulation_config.require_features_for_direction and payload.features is None:
            return "capitulation_requires_features_for_direction"

        if not payload.has_capitulation_regime and not payload.has_capitulation_anomaly:
            return "capitulation_not_detected"

        if payload.features is not None:
            if not self._features_support_reversal_side(payload=payload, side=side):
                return f"capitulation_features_do_not_support_side:{side.value}"

        if payload.divergence is not None and payload.divergence.detected:
            hint = divergence_side_hint(payload.divergence.divergence_type)

            if side is SignalSide.LONG and hint == "bearish":
                return "long_capitulation_rejected_by_bearish_divergence"

            if side is SignalSide.SHORT and hint == "bullish":
                return "short_capitulation_rejected_by_bullish_divergence"

        return None

    def _features_support_reversal_side(
        self,
        *,
        payload: OICapitulationPayload,
        side: SignalSide,
    ) -> bool:
        features = payload.features
        if features is None:
            return False

        price_delta = extract_price_delta_pct(features)
        oi_delta = extract_oi_delta_pct(features)
        liquidation_pressure = extract_liquidation_pressure(features)
        pressure = extract_oi_pressure_score(features)

        if abs(price_delta) < self.capitulation_config.min_abs_price_delta_for_flush:
            return False

        if abs(oi_delta) < self.capitulation_config.min_abs_oi_delta_for_flush:
            return False

        if side is SignalSide.LONG:
            return (
                price_delta <= 0
                or liquidation_pressure <= -self.capitulation_config.liquidation_flush_threshold
                or pressure <= -self.capitulation_config.pressure_flush_threshold
            )

        if side is SignalSide.SHORT:
            return (
                price_delta >= 0
                or liquidation_pressure >= self.capitulation_config.liquidation_flush_threshold
                or pressure >= self.capitulation_config.pressure_flush_threshold
            )

        return False

    # ------------------------------------------------------------------
    # Scoring / confidence
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: OICapitulationPayload,
        side: SignalSide,
        event_time: Any,
    ) -> ScoreBreakdown:
        capitulation_score = unit_score(payload.score)
        features_score = self._feature_context_score(payload=payload, side=side)
        divergence_score = self._divergence_context_score(payload=payload, side=side)
        analysis_score = unit_score(payload.analysis_confidence)
        fresh_score = freshness_score(
            event_time=event_time,
            now=context.timestamp,
            stale_after_seconds=self.capitulation_config.stale_feature_max_age_seconds,
        )

        components = {
            "capitulation": capitulation_score,
            "features": features_score,
            "divergence": divergence_score,
            "analysis": analysis_score,
            "freshness": fresh_score,
        }
        weights = {
            "capitulation": self.capitulation_config.score_capitulation_weight,
            "features": self.capitulation_config.score_features_weight,
            "divergence": self.capitulation_config.score_divergence_weight,
            "analysis": self.capitulation_config.score_analysis_weight,
            "freshness": self.capitulation_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=capitulation_score)
        confidence = confidence_from_components(
            primary=payload.confidence,
            context=features_score,
            confirmation=divergence_score,
            freshness=fresh_score,
            primary_weight=self.capitulation_config.confidence_capitulation_weight,
            context_weight=self.capitulation_config.confidence_context_weight,
            confirmation_weight=self.capitulation_config.confidence_confirmation_weight,
            freshness_weight=self.capitulation_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            f"side:{side.value}",
            "capitulation_context",
        ]

        if payload.has_capitulation_regime:
            score += self.capitulation_config.capitulation_regime_bonus
            confidence += 0.04
            confirmations.append("capitulation_regime")

        if payload.has_capitulation_anomaly:
            score += self.capitulation_config.capitulation_anomaly_bonus
            confidence += 0.04
            confirmations.append("capitulation_anomaly")

        if payload.analysis_confidence >= 0.75:
            score += 0.04
            confidence += 0.04
            confirmations.append("high_analysis_confidence")
        elif (
            payload.analysis_confidence
            >= self.capitulation_config.min_analysis_confidence_bonus_threshold
        ):
            score += 0.02
            confidence += 0.02
            confirmations.append("moderate_analysis_confidence")

        feature_adjustment = self._feature_score_adjustment(
            payload=payload,
            side=side,
        )
        score += feature_adjustment

        confidence += self._feature_confidence_adjustment(
            payload=payload,
            side=side,
        )

        divergence_adjustment = self._divergence_score_adjustment(
            payload=payload,
            side=side,
        )
        score += divergence_adjustment

        if divergence_adjustment > 0:
            confidence += min(0.04, divergence_adjustment)
            confirmations.append("aligned_divergence_context")
        elif divergence_adjustment < 0:
            confidence -= min(0.06, abs(divergence_adjustment))
            reasons.append("opposing_divergence_penalty")

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
        payload: OICapitulationPayload,
        side: SignalSide,
    ) -> float:
        features = payload.features
        if features is None:
            return 0.0

        price_delta = extract_price_delta_pct(features)
        oi_delta = extract_oi_delta_pct(features)
        liquidation_pressure = extract_liquidation_pressure(features)
        pressure = extract_oi_pressure_score(features)
        flow = extract_aggressive_flow_imbalance(features)

        components: dict[str, float] = {
            "price_flush": unit_score(abs(price_delta)),
            "oi_flush": unit_score(abs(oi_delta)),
            "liquidation_pressure": unit_score(abs(liquidation_pressure)),
            "oi_pressure": unit_score(abs(pressure)),
            "flow": unit_score(abs(flow)),
        }

        volume_ratio = get_attr_or_key(features, "volume_ratio")
        if volume_ratio is not None:
            components["volume"] = unit_score(float(volume_ratio) / 2.0)

        oi_zscore = get_attr_or_key(features, "oi_zscore")
        if oi_zscore is not None:
            components["oi_zscore"] = unit_score(abs(float(oi_zscore)) / 3.0)

        if side is SignalSide.LONG:
            components["side_alignment"] = unit_score(
                max(0.0, -price_delta)
                + max(0.0, -liquidation_pressure)
                + max(0.0, -pressure)
            )

        elif side is SignalSide.SHORT:
            components["side_alignment"] = unit_score(
                max(0.0, price_delta)
                + max(0.0, liquidation_pressure)
                + max(0.0, pressure)
            )

        weights = {key: 1.0 for key in components}
        return weighted_score(components, weights)

    def _divergence_context_score(
        self,
        *,
        payload: OICapitulationPayload,
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
        payload: OICapitulationPayload,
        side: SignalSide,
    ) -> float:
        features = payload.features
        if features is None:
            return 0.0

        adjustment = 0.0

        price_delta = extract_price_delta_pct(features)
        liquidation_pressure = extract_liquidation_pressure(features)
        pressure = extract_oi_pressure_score(features)
        flow = extract_aggressive_flow_imbalance(features)
        funding_rate = extract_funding_rate(features)

        if side is SignalSide.LONG:
            if liquidation_pressure <= -self.capitulation_config.liquidation_flush_threshold:
                adjustment += self.capitulation_config.liquidation_flush_bonus

            if pressure <= -self.capitulation_config.pressure_flush_threshold:
                adjustment += 0.03

            if price_delta < 0:
                adjustment += 0.03

            if flow >= -self.capitulation_config.flow_stabilization_threshold:
                adjustment += self.capitulation_config.stabilization_bonus

            if funding_rate is not None and funding_rate < 0:
                adjustment += self.capitulation_config.funding_extreme_bonus

        elif side is SignalSide.SHORT:
            if liquidation_pressure >= self.capitulation_config.liquidation_flush_threshold:
                adjustment += self.capitulation_config.liquidation_flush_bonus

            if pressure >= self.capitulation_config.pressure_flush_threshold:
                adjustment += 0.03

            if price_delta > 0:
                adjustment += 0.03

            if flow <= self.capitulation_config.flow_stabilization_threshold:
                adjustment += self.capitulation_config.stabilization_bonus

            if funding_rate is not None and funding_rate > 0:
                adjustment += self.capitulation_config.funding_extreme_bonus

        return adjustment

    def _feature_confidence_adjustment(
        self,
        *,
        payload: OICapitulationPayload,
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

        liquidation_pressure = extract_liquidation_pressure(features)
        pressure = extract_oi_pressure_score(features)

        if side is SignalSide.LONG:
            if liquidation_pressure <= -self.capitulation_config.liquidation_flush_threshold:
                adjustment += 0.03
            if pressure <= -self.capitulation_config.pressure_flush_threshold:
                adjustment += 0.02

        elif side is SignalSide.SHORT:
            if liquidation_pressure >= self.capitulation_config.liquidation_flush_threshold:
                adjustment += 0.03
            if pressure >= self.capitulation_config.pressure_flush_threshold:
                adjustment += 0.02

        return adjustment

    def _divergence_score_adjustment(
        self,
        *,
        payload: OICapitulationPayload,
        side: SignalSide,
    ) -> float:
        divergence = payload.divergence
        if (
            divergence is None
            or not divergence.detected
            or not self.capitulation_config.allow_divergence_confirmation
        ):
            return 0.0

        hint = divergence_side_hint(divergence.divergence_type)
        divergence_strength = extract_score(
            divergence,
            default=extract_confidence(divergence),
        )

        if side is SignalSide.LONG and hint == "bullish":
            return min(
                self.capitulation_config.aligned_divergence_bonus,
                divergence_strength * 0.08,
            )

        if side is SignalSide.SHORT and hint == "bearish":
            return min(
                self.capitulation_config.aligned_divergence_bonus,
                divergence_strength * 0.08,
            )

        if hint in {"bullish", "bearish"}:
            return -min(
                self.capitulation_config.opposing_divergence_penalty,
                divergence_strength * 0.14,
            )

        return 0.0

    # ------------------------------------------------------------------
    # Source features / tags
    # ------------------------------------------------------------------

    def _source_features(self, payload: OICapitulationPayload) -> list[str]:
        features: list[str] = []

        if payload.regime is not None:
            features.extend(
                [
                    OPEN_INTEREST_FEATURES.REGIME,
                    OPEN_INTEREST_FEATURES.REGIME_TYPE,
                    OPEN_INTEREST_FEATURES.REGIME_CONFIDENCE,
                    OPEN_INTEREST_FEATURES.REGIME_SCORE,
                ]
            )

        if payload.anomaly is not None:
            features.extend(
                [
                    OPEN_INTEREST_FEATURES.ANOMALY,
                    OPEN_INTEREST_FEATURES.ANOMALY_TYPE,
                    OPEN_INTEREST_FEATURES.ANOMALY_DETECTED,
                    OPEN_INTEREST_FEATURES.ANOMALY_CONFIDENCE,
                    OPEN_INTEREST_FEATURES.ANOMALY_SCORE,
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

        if payload.divergence is not None:
            features.extend(
                [
                    OPEN_INTEREST_FEATURES.DIVERGENCE,
                    OPEN_INTEREST_FEATURES.DIVERGENCE_TYPE,
                    OPEN_INTEREST_FEATURES.DIVERGENCE_CONFIDENCE,
                ]
            )

        return list(dict.fromkeys(features))

    def _tags(
        self,
        *,
        payload: OICapitulationPayload,
        setup_type: SetupType,
    ) -> list[str]:
        tags = [
            self.capitulation_config.tag_open_interest,
            self.capitulation_config.tag_oi_capitulation,
            self.capitulation_config.tag_capitulation,
            self.capitulation_config.tag_reversal,
            self.capitulation_config.tag_risk,
            setup_type.value,
        ]

        if payload.has_capitulation_anomaly:
            tags.extend(
                [
                    self.capitulation_config.tag_deleveraging,
                    self.capitulation_config.tag_liquidations,
                    self.capitulation_config.tag_forced_unwind,
                ]
            )

        if payload.regime is not None:
            tags.append(f"oi_regime:{payload.regime.regime.value}")

        if payload.anomaly is not None and payload.anomaly.detected:
            tags.append(f"oi_anomaly:{payload.anomaly.anomaly_type.value}")

        if payload.divergence is not None and payload.divergence.detected:
            tags.append(f"oi_divergence:{payload.divergence.divergence_type.value}")

        return list(dict.fromkeys(tags))

    @staticmethod
    def _anomaly_strength(payload: OICapitulationPayload) -> str | None:
        if payload.anomaly is None:
            return None

        strength = getattr(payload.anomaly, "strength", None)

        if strength is None:
            return None

        value = getattr(strength, "value", None)
        if isinstance(value, str):
            return value

        return str(strength)