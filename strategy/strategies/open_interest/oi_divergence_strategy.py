# trading_system/strategy/strategies/open_interest/oi_divergence_strategy.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
from .base import (
    OPEN_INTEREST_FEATURES,
    OpenInterestStrategyConfig,
    OpenInterestTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    anomaly_is_risk_critical,
    confidence_from_components,
    divergence_filter_reason,
    divergence_side_hint,
    extract_aggressive_flow_imbalance,
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
    serialize_for_metadata,
    side_from_oi_divergence,
    side_from_oi_features,
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
class OIDivergencePayload:
    """
    Normalized strategy-level payload для OI divergence.

    Source of truth:
        analytics.open_interest.models.OIDivergenceResult

    Додатковий context:
        - OIFeatures;
        - OIRegimeResult;
        - OIAnomalyResult;
        - full OI analysis confidence через OpenInterestTradingStrategy.
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
        return bool(self.divergence.detected)

    @property
    def confidence(self) -> float:
        return unit_score(getattr(self.divergence, "confidence", 0.0))

    @property
    def score(self) -> float:
        return unit_score(getattr(self.divergence, "score", self.confidence))

    @property
    def window_size(self) -> int | None:
        return getattr(self.divergence, "window_size", None)


@dataclass(slots=True)
class OIDivergenceStrategyConfig(OpenInterestStrategyConfig):
    """
    Unified OI divergence strategy config.

    Strategy idea:
    - read normalized OI divergence context from StrategyContext;
    - accept detected directional OIDivergenceResult;
    - use OIFeatures, OIRegimeResult and OIAnomalyResult only as context;
    - generate reversal / mean-reversion / exhaustion signal;
    - leave routing, filtering, confluence, portfolio coordination and
      risk-ready conversion to SignalProcessor.
    """

    require_detected_result: bool = True
    require_actionable_side: bool = True
    require_fresh_divergence: bool = True

    min_divergence_confidence: float = 0.58
    min_divergence_score: float = 0.35
    min_analysis_confidence_bonus_threshold: float = 0.50

    block_hard_risk_anomalies: bool = True
    allow_hard_anomaly_when_flush_aligned: bool = True

    soft_risk_anomaly_penalty: float = 0.08
    max_soft_risk_penalty: float = 0.10

    min_window_for_bonus: int = 5
    strong_window_for_bonus: int = 8

    min_volume_ratio_for_confirmation: float = 1.0
    strong_volume_ratio: float = 1.5
    min_abs_oi_zscore_for_confirmation: float = 1.0
    inefficient_price_move_threshold: float = 0.50
    moderate_inefficiency_threshold: float = 0.80
    low_pressure_threshold: float = 0.20
    liquidation_flush_threshold: float = 0.20
    flow_fading_threshold: float = 0.05

    score_divergence_weight: float = 0.42
    score_regime_weight: float = 0.18
    score_features_weight: float = 0.25
    score_analysis_weight: float = 0.10
    score_freshness_weight: float = 0.05

    confidence_divergence_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    tag_oi_divergence: str = "oi_divergence"
    tag_reversal: str = "reversal"
    tag_mean_reversion: str = "mean_reversion"
    tag_exhaustion: str = "exhaustion"
    tag_contextual: str = "contextual"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

    required_open_interest_features: tuple[str, ...] = (
        OPEN_INTEREST_FEATURES.DIVERGENCE,
    )

    def validate(self) -> None:
        OpenInterestStrategyConfig.validate(self)

        bounded_fields = {
            "min_divergence_confidence": self.min_divergence_confidence,
            "min_divergence_score": self.min_divergence_score,
            "min_analysis_confidence_bonus_threshold": self.min_analysis_confidence_bonus_threshold,
            "soft_risk_anomaly_penalty": self.soft_risk_anomaly_penalty,
            "max_soft_risk_penalty": self.max_soft_risk_penalty,
            "inefficient_price_move_threshold": self.inefficient_price_move_threshold,
            "moderate_inefficiency_threshold": self.moderate_inefficiency_threshold,
            "low_pressure_threshold": self.low_pressure_threshold,
            "liquidation_flush_threshold": self.liquidation_flush_threshold,
            "flow_fading_threshold": self.flow_fading_threshold,
        }

        for field_name, value in bounded_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.min_window_for_bonus < 0:
            raise StrategyConfigError("min_window_for_bonus must be >= 0")

        if self.strong_window_for_bonus < self.min_window_for_bonus:
            raise StrategyConfigError(
                "strong_window_for_bonus must be >= min_window_for_bonus"
            )

        if self.min_volume_ratio_for_confirmation < 0:
            raise StrategyConfigError(
                "min_volume_ratio_for_confirmation must be >= 0"
            )

        if self.strong_volume_ratio < self.min_volume_ratio_for_confirmation:
            raise StrategyConfigError(
                "strong_volume_ratio must be >= min_volume_ratio_for_confirmation"
            )

        if self.min_abs_oi_zscore_for_confirmation < 0:
            raise StrategyConfigError(
                "min_abs_oi_zscore_for_confirmation must be >= 0"
            )

        score_weights = {
            "score_divergence_weight": self.score_divergence_weight,
            "score_regime_weight": self.score_regime_weight,
            "score_features_weight": self.score_features_weight,
            "score_analysis_weight": self.score_analysis_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }

        confidence_weights = {
            "confidence_divergence_weight": self.confidence_divergence_weight,
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
            "tag_oi_divergence",
            "tag_reversal",
            "tag_mean_reversion",
            "tag_exhaustion",
            "tag_contextual",
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


class OIDivergenceStrategy(OpenInterestTradingStrategy):
    """
    Unified OI divergence strategy.

    Input:
        StrategyContext with FeatureSource.OPEN_INTEREST domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.open_interest.divergence"
    category: StrategyCategory = StrategyCategory.OPEN_INTEREST
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

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
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        open_interest_config: OIDivergenceStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_open_interest_config = (
            open_interest_config or OIDivergenceStrategyConfig()
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

        self.divergence_config: OIDivergenceStrategyConfig = (
            resolved_open_interest_config
        )

    @property
    def strategy_name(self) -> str:
        return "oi_divergence"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.OPEN_INTEREST,
            timeframe=Timeframe.M1,
            tags=[
                self.divergence_config.tag_open_interest,
                self.divergence_config.tag_oi_divergence,
                self.divergence_config.tag_reversal,
                self.divergence_config.tag_mean_reversion,
                self.divergence_config.tag_exhaustion,
                self.divergence_config.tag_contextual,
                "analytics_open_interest",
            ],
            version="2.0.0",
            description=(
                "Інтерпретує OI divergence з analytics.open_interest і будує "
                "internal StrategySignal з урахуванням OI regime, anomaly, "
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
                "strategy_type": "single_factor_contextual",
                "base_class": "OpenInterestTradingStrategy",
                "canonical_payload": "OIAnalysisResult",
                "primary_result": "OIDivergenceResult",
                "uses_features": True,
                "uses_regime": True,
                "uses_anomaly": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(
            self.divergence_config.required_open_interest_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        if not self.has_any_open_interest_data(
            context,
            tuple(self.divergence_config.required_open_interest_features),
        ):
            self.remember_no_signal(
                "missing_open_interest_divergence_contract",
                open_interest_domain_keys=sorted(self.open_interest_domain(context).keys()),
                required_features=sorted(self.required_features()),
            )
            return None

        if self.has_stale_open_interest_features(
            context,
            tuple(self.divergence_config.required_open_interest_features),
        ):
            self.remember_no_signal(
                "stale_open_interest_divergence_features",
                required_features=sorted(self.required_features()),
            )
            return None

        payload = self._extract_payload(context)
        if payload is None:
            self.remember_no_signal(
                "open_interest_divergence_payload_not_resolved",
                open_interest_domain=self.open_interest_domain(context),
                required_features=sorted(self.required_features()),
            )
            return None

        event_time = extract_event_time(payload.divergence)
        if (
            self.divergence_config.require_fresh_divergence
            and is_stale(
                event_time=event_time,
                now=context.timestamp,
                stale_after_seconds=self.divergence_config.stale_feature_max_age_seconds,
            )
        ):
            self.remember_no_signal(
                "stale_open_interest_divergence",
                event_time=event_time.isoformat() if event_time else None,
                context_timestamp=context.timestamp.isoformat(),
                stale_after_seconds=self.divergence_config.stale_feature_max_age_seconds,
            )
            return None

        common_rejection = divergence_filter_reason(
            payload.divergence,
            min_confidence=self.divergence_config.min_divergence_confidence,
            min_score=self.divergence_config.min_divergence_score,
            require_detected=self.divergence_config.require_detected_result,
            require_directional=False,
        )
        if common_rejection is not None:
            self.remember_no_signal(
                "open_interest_divergence_filter_failed",
                filter_reason=common_rejection,
                divergence=serialize_for_metadata(payload.divergence),
                min_divergence_confidence=self.divergence_config.min_divergence_confidence,
                min_divergence_score=self.divergence_config.min_divergence_score,
            )
            return None

        side = self._map_divergence_to_side(payload)
        if self.divergence_config.require_actionable_side and not is_directional_side(side):
            self.remember_no_signal(
                "open_interest_divergence_side_not_directional",
                divergence=serialize_for_metadata(payload.divergence),
                features=serialize_for_metadata(payload.features),
                regime=serialize_for_metadata(payload.regime),
                anomaly=serialize_for_metadata(payload.anomaly),
            )
            return None

        blocked_reason = self._risk_block_reason(payload=payload, side=side)
        if blocked_reason is not None:
            self.remember_no_signal(
                "open_interest_divergence_risk_blocked",
                blocked_reason=blocked_reason,
                side=side.value,
                divergence=serialize_for_metadata(payload.divergence),
                anomaly=serialize_for_metadata(payload.anomaly),
            )
            return None

        setup_type = self._map_divergence_to_setup_type(payload)

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
            side=side,
            event_time=event_time,
        )

        if breakdown.score < self.divergence_config.min_signal_score:
            self.remember_no_signal(
                "open_interest_divergence_score_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_signal_score=self.divergence_config.min_signal_score,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        if breakdown.confidence < self.divergence_config.min_signal_confidence:
            self.remember_no_signal(
                "open_interest_divergence_confidence_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_signal_confidence=self.divergence_config.min_signal_confidence,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload=payload, setup_type=setup_type)

        reasons = list(
            dict.fromkeys(
                [
                    "oi_divergence_signal",
                    f"side:{side.value}",
                    f"divergence:{payload.divergence_type.value}",
                    f"setup_type:{setup_type.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "open_interest_setup_family": "oi_divergence",
            "open_interest_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "event_time": event_time.isoformat() if event_time else None,
            "divergence": serialize_for_metadata(payload.divergence),
            "regime": serialize_for_metadata(payload.regime),
            "anomaly": serialize_for_metadata(payload.anomaly),
            "features": serialize_for_metadata(payload.features),
            "raw": serialize_for_metadata(payload.raw),
            "divergence_type": payload.divergence_type.value,
            "divergence_detected": payload.detected,
            "divergence_confidence": payload.confidence,
            "divergence_score": payload.score,
            "divergence_window_size": payload.window_size,
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
            priority=self.divergence_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(self, context: StrategyContext) -> OIDivergencePayload | None:
        divergence = self.extract_oi_divergence_result(context)
        if divergence is None:
            return None

        features = self.extract_oi_features(context)
        regime = self.extract_oi_regime_result(context)
        anomaly = self.extract_oi_anomaly_result(context)

        reasons: list[str] = []
        reasons.extend(extract_reasons(divergence))

        if regime is not None:
            reasons.extend(f"regime:{reason}" for reason in extract_reasons(regime))

        if anomaly is not None and anomaly.detected:
            reasons.extend(f"anomaly:{reason}" for reason in extract_reasons(anomaly))

        return OIDivergencePayload(
            divergence=divergence,
            features=features,
            regime=regime,
            anomaly=anomaly,
            analysis_confidence=self.oi_analysis_confidence(context),
            reasons=list(dict.fromkeys(reasons)),
            raw=self.open_interest_domain(context),
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
        semantic_hint = divergence_side_hint(divergence_type)

        if semantic_hint == "bullish":
            return SignalSide.LONG

        if semantic_hint == "bearish":
            return SignalSide.SHORT

        utility_side = side_from_oi_divergence(
            divergence_type,
            features=payload.features,
        )
        if is_directional_side(utility_side):
            return utility_side

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

        feature_side = side_from_oi_features(features)
        if is_directional_side(feature_side):
            return self.opposite_side(feature_side)

        price_delta = extract_price_delta_pct(features)
        oi_delta = extract_oi_delta_pct(features)

        if price_delta < 0 and oi_delta <= 0:
            return SignalSide.LONG

        if price_delta > 0 and oi_delta <= 0:
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

        return self.divergence_config.default_setup_type

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

        if not self.divergence_config.block_hard_risk_anomalies:
            return None

        if anomaly.anomaly_type not in self.HARD_BLOCKING_ANOMALIES:
            return None

        if (
            self.divergence_config.allow_hard_anomaly_when_flush_aligned
            and self._hard_anomaly_flush_supports_side(payload=payload, side=side)
        ):
            return None

        return f"divergence_blocked_by_risk_anomaly:{anomaly.anomaly_type.value}"

    def _hard_anomaly_flush_supports_side(
        self,
        *,
        payload: OIDivergencePayload,
        side: SignalSide,
    ) -> bool:
        features = payload.features
        if features is None:
            return False

        price_delta = extract_price_delta_pct(features)
        liquidation_pressure = extract_liquidation_pressure(features)

        threshold = self.divergence_config.liquidation_flush_threshold

        if side is SignalSide.LONG:
            return price_delta < 0 and liquidation_pressure <= -threshold

        if side is SignalSide.SHORT:
            return price_delta > 0 and liquidation_pressure >= threshold

        return False

    # ------------------------------------------------------------------
    # Scoring / confidence
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: OIDivergencePayload,
        side: SignalSide,
        event_time: datetime | None,
    ) -> ScoreBreakdown:
        base_score = payload.score if payload.score > 0 else payload.confidence
        divergence_score = unit_score(base_score)

        regime_score = self._regime_context_score(payload)
        feature_score = self._feature_context_score(payload=payload, side=side)
        analysis_score = unit_score(payload.analysis_confidence)
        fresh_score = freshness_score(
            event_time=event_time,
            now=context.timestamp,
            stale_after_seconds=self.divergence_config.stale_feature_max_age_seconds,
        )

        components = {
            "divergence": divergence_score,
            "regime": regime_score,
            "features": feature_score,
            "analysis": analysis_score,
            "freshness": fresh_score,
        }
        weights = {
            "divergence": self.divergence_config.score_divergence_weight,
            "regime": self.divergence_config.score_regime_weight,
            "features": self.divergence_config.score_features_weight,
            "analysis": self.divergence_config.score_analysis_weight,
            "freshness": self.divergence_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=divergence_score)
        confidence = confidence_from_components(
            primary=payload.confidence,
            context=regime_score,
            confirmation=feature_score,
            freshness=fresh_score,
            primary_weight=self.divergence_config.confidence_divergence_weight,
            context_weight=self.divergence_config.confidence_context_weight,
            confirmation_weight=self.divergence_config.confidence_confirmation_weight,
            freshness_weight=self.divergence_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            f"divergence:{payload.divergence_type.value}",
        ]

        if payload.window_size is not None:
            confirmations.append(f"divergence_window:{payload.window_size}")

            if payload.window_size >= self.divergence_config.strong_window_for_bonus:
                score += 0.04
                confidence += 0.02
                reasons.append("strong_divergence_window")

            elif payload.window_size >= self.divergence_config.min_window_for_bonus:
                score += 0.02
                confidence += 0.01
                reasons.append("valid_divergence_window")

        if payload.analysis_confidence >= 0.75:
            score += 0.04
            confidence += 0.04
            confirmations.append("high_analysis_confidence")

        elif (
            payload.analysis_confidence
            >= self.divergence_config.min_analysis_confidence_bonus_threshold
        ):
            score += 0.02
            confidence += 0.02
            confirmations.append("moderate_analysis_confidence")

        if payload.regime is not None:
            confirmations.append(f"oi_regime:{payload.regime.regime.value}")

            if payload.regime.regime in self.REVERSAL_REGIMES:
                score += 0.07
                confidence += 0.04
                confirmations.append("reversal_regime_context")

            elif payload.regime.regime in self.CONTEXTUAL_REGIMES:
                score += 0.03
                confidence += 0.02
                confirmations.append("contextual_oi_regime")

            if payload.regime.confidence >= 0.75:
                score += 0.03
                confirmations.append("high_confidence_oi_regime")

        feature_adjustment = self._feature_score_adjustment(
            payload=payload,
            side=side,
        )
        score += feature_adjustment

        feature_confidence_adjustment = self._feature_confidence_adjustment(
            payload=payload,
            side=side,
        )
        confidence += feature_confidence_adjustment

        anomaly = payload.anomaly
        if anomaly is not None and anomaly.detected:
            confirmations.append(f"oi_anomaly:{anomaly.anomaly_type.value}")

            if anomaly.anomaly_type in self.SOFT_RISK_ANOMALIES:
                penalty = min(
                    self.divergence_config.max_soft_risk_penalty,
                    extract_score(anomaly) * self.divergence_config.soft_risk_anomaly_penalty,
                )
                score -= penalty
                confidence -= min(0.08, penalty)
                reasons.append("soft_risk_anomaly_penalty")
                confirmations.append("soft_risk_anomaly_context")

            if anomaly.anomaly_type in {
                OIAnomalyType.OI_PRICE_DISLOCATION,
                OIAnomalyType.OI_VOLUME_DISLOCATION,
            }:
                score += 0.02
                confirmations.append("dislocation_context")

            if anomaly_is_risk_critical(anomaly.anomaly_type):
                confirmations.append("risk_critical_anomaly_context")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    def _regime_context_score(self, payload: OIDivergencePayload) -> float:
        regime = payload.regime
        if regime is None:
            return 0.0

        score = unit_score(getattr(regime, "score", regime.confidence))

        if regime.regime in self.REVERSAL_REGIMES:
            return unit_score(score + 0.20)

        if regime.regime in self.CONTEXTUAL_REGIMES:
            return unit_score(score + 0.10)

        return score

    def _feature_context_score(
        self,
        *,
        payload: OIDivergencePayload,
        side: SignalSide,
    ) -> float:
        features = payload.features
        if features is None:
            return 0.0

        components: dict[str, float] = {}

        volume_ratio = get_attr_or_key(features, "volume_ratio")
        if volume_ratio is not None:
            components["volume"] = unit_score(float(volume_ratio) / 2.0)

        oi_zscore = get_attr_or_key(features, "oi_zscore")
        if oi_zscore is not None:
            components["oi_zscore"] = unit_score(abs(float(oi_zscore)) / 3.0)

        efficiency = get_attr_or_key(features, "oi_price_efficiency")
        if efficiency is not None:
            components["inefficiency"] = unit_score(1.0 - min(abs(float(efficiency)), 1.0))

        pressure = abs(extract_oi_pressure_score(features))
        components["pressure_neutrality"] = unit_score(1.0 - pressure)

        liquidation_pressure = extract_liquidation_pressure(features)
        if side is SignalSide.LONG:
            components["flush_alignment"] = unit_score(abs(min(0.0, liquidation_pressure)))
        elif side is SignalSide.SHORT:
            components["flush_alignment"] = unit_score(max(0.0, liquidation_pressure))

        flow = extract_aggressive_flow_imbalance(features)
        if side is SignalSide.LONG:
            components["flow_fading"] = unit_score(flow + 1.0, 0.0) / 2.0
        elif side is SignalSide.SHORT:
            components["flow_fading"] = unit_score(1.0 - flow, 0.0) / 2.0

        if not components:
            return 0.0

        weights = {key: 1.0 for key in components}
        return weighted_score(components, weights)

    def _feature_score_adjustment(
        self,
        *,
        payload: OIDivergencePayload,
        side: SignalSide,
    ) -> float:
        features = payload.features
        if features is None:
            return 0.0

        adjustment = 0.0

        volume_ratio = get_attr_or_key(features, "volume_ratio")
        if volume_ratio is not None:
            volume = float(volume_ratio)
            if volume >= self.divergence_config.strong_volume_ratio:
                adjustment += 0.05
            elif volume >= self.divergence_config.min_volume_ratio_for_confirmation:
                adjustment += 0.03

        oi_zscore = get_attr_or_key(features, "oi_zscore")
        if (
            oi_zscore is not None
            and abs(float(oi_zscore))
            >= self.divergence_config.min_abs_oi_zscore_for_confirmation
        ):
            adjustment += 0.03

        efficiency = get_attr_or_key(features, "oi_price_efficiency")
        if efficiency is not None:
            abs_efficiency = abs(float(efficiency))

            if abs_efficiency < self.divergence_config.inefficient_price_move_threshold:
                adjustment += 0.05
            elif abs_efficiency < self.divergence_config.moderate_inefficiency_threshold:
                adjustment += 0.03

        pressure = abs(extract_oi_pressure_score(features))
        if pressure < self.divergence_config.low_pressure_threshold:
            adjustment += 0.03

        liquidation_pressure = extract_liquidation_pressure(features)
        flush_threshold = self.divergence_config.liquidation_flush_threshold

        if side is SignalSide.LONG and liquidation_pressure <= -flush_threshold:
            adjustment += 0.04
        elif side is SignalSide.SHORT and liquidation_pressure >= flush_threshold:
            adjustment += 0.04
        elif abs(liquidation_pressure) >= flush_threshold:
            adjustment += 0.02

        flow = extract_aggressive_flow_imbalance(features)
        flow_threshold = self.divergence_config.flow_fading_threshold

        if side is SignalSide.LONG and flow > -flow_threshold:
            adjustment += 0.03
        elif side is SignalSide.SHORT and flow < flow_threshold:
            adjustment += 0.03

        funding_rate = extract_funding_rate(features)
        if funding_rate is not None:
            if side is SignalSide.LONG and funding_rate < 0:
                adjustment += 0.03
            elif side is SignalSide.SHORT and funding_rate > 0:
                adjustment += 0.03

        if payload.divergence_type in {
            OIDivergenceType.EXHAUSTION_UP,
            OIDivergenceType.EXHAUSTION_DOWN,
        }:
            if efficiency is not None and abs(float(efficiency)) >= 1.0:
                adjustment += 0.03

        return adjustment

    def _feature_confidence_adjustment(
        self,
        *,
        payload: OIDivergencePayload,
        side: SignalSide,
    ) -> float:
        features = payload.features
        if features is None:
            return 0.0

        adjustment = 0.0

        volume_ratio = get_attr_or_key(features, "volume_ratio")
        if (
            volume_ratio is not None
            and float(volume_ratio)
            >= self.divergence_config.min_volume_ratio_for_confirmation
        ):
            adjustment += 0.03

        oi_zscore = get_attr_or_key(features, "oi_zscore")
        if (
            oi_zscore is not None
            and abs(float(oi_zscore))
            >= self.divergence_config.min_abs_oi_zscore_for_confirmation
        ):
            adjustment += 0.03

        efficiency = get_attr_or_key(features, "oi_price_efficiency")
        if (
            efficiency is not None
            and abs(float(efficiency))
            < self.divergence_config.inefficient_price_move_threshold
        ):
            adjustment += 0.03

        liquidation_pressure = extract_liquidation_pressure(features)
        funding_rate = extract_funding_rate(features)

        if side is SignalSide.LONG:
            if liquidation_pressure <= -self.divergence_config.liquidation_flush_threshold:
                adjustment += 0.03
            if funding_rate is not None and funding_rate < 0:
                adjustment += 0.02

        elif side is SignalSide.SHORT:
            if liquidation_pressure >= self.divergence_config.liquidation_flush_threshold:
                adjustment += 0.03
            if funding_rate is not None and funding_rate > 0:
                adjustment += 0.02

        return adjustment

    # ------------------------------------------------------------------
    # Source features / tags
    # ------------------------------------------------------------------

    def _source_features(self, payload: OIDivergencePayload) -> list[str]:
        features = [
            OPEN_INTEREST_FEATURES.DIVERGENCE,
            OPEN_INTEREST_FEATURES.DIVERGENCE_TYPE,
            OPEN_INTEREST_FEATURES.DIVERGENCE_CONFIDENCE,
            OPEN_INTEREST_FEATURES.DIVERGENCE_SCORE,
        ]

        if payload.regime is not None:
            features.extend(
                [
                    OPEN_INTEREST_FEATURES.REGIME,
                    OPEN_INTEREST_FEATURES.REGIME_TYPE,
                    OPEN_INTEREST_FEATURES.REGIME_CONFIDENCE,
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
        payload: OIDivergencePayload,
        setup_type: SetupType,
    ) -> list[str]:
        tags = [
            self.divergence_config.tag_open_interest,
            self.divergence_config.tag_oi_divergence,
            setup_type.value,
            payload.divergence_type.value,
        ]

        if setup_type is SetupType.REVERSAL:
            tags.append(self.divergence_config.tag_reversal)

        if setup_type is SetupType.MEAN_REVERSION:
            tags.append(self.divergence_config.tag_mean_reversion)

        if setup_type is SetupType.EXHAUSTION:
            tags.append(self.divergence_config.tag_exhaustion)

        if payload.regime is not None:
            tags.append(f"oi_regime:{payload.regime.regime.value}")

        if payload.anomaly is not None and payload.anomaly.detected:
            tags.append(f"oi_anomaly:{payload.anomaly.anomaly_type.value}")

        return list(dict.fromkeys(tags))