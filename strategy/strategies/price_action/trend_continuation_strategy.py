# trading_system/strategy/strategies/price_action/trend_continuation_strategy.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from analytics.price_action.enums import (
    StructureLayer,
    TrendDirection,
    TrendEventType,
    TrendRegime,
)
from core.event_bus import EventBus
from core.scheduler import Scheduler
from .base import (
    PRICE_ACTION_FEATURES,
    PriceActionStrategyConfig,
    PriceActionTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    average_score,
    confidence_from_components,
    extract_last_event,
    extract_last_update,
    get_path,
    is_directional_side,
    is_stale,
    layer_confidence,
    layer_strength,
    normalize_label,
    parse_datetime,
    parse_structure_layer,
    parse_trend_direction,
    parse_trend_event_type,
    parse_trend_regime,
    quality_filter_reason,
    select_primary_layer,
    select_secondary_layer,
    serialize_for_metadata,
    to_bool,
    trend_direction_to_side,
    trend_source_features,
    unit_score,
    weighted_score,
    freshness_score,
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
class TrendEventContext:
    """
    Normalized trend lifecycle/signal event.

    Strategy-layer DTO only. Analytics models remain in analytics.price_action.
    """

    event_type: TrendEventType | None = None
    direction: TrendDirection | None = None
    regime: TrendRegime | None = None
    layer: StructureLayer | None = None

    confidence: float = 0.0
    score: float = 0.0
    continuation_probability: float = 0.0
    reversal_risk: float = 0.0
    exhaustion_score: float = 0.0

    timestamp: datetime | None = None
    is_confirmed: bool = False

    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        fallback_layer: StructureLayer | None = None,
    ) -> TrendEventContext | None:
        if payload is None:
            return None

        event_type = parse_trend_event_type(
            get_path(payload, "event_type")
            or get_path(payload, "type")
            or get_path(payload, "kind")
        )
        direction = parse_trend_direction(
            get_path(payload, "direction")
            or get_path(payload, "trend_direction")
            or get_path(payload, "side")
        )
        regime = parse_trend_regime(
            get_path(payload, "regime")
            or get_path(payload, "trend_regime")
            or get_path(payload, "state")
        )
        layer = parse_structure_layer(
            get_path(payload, "layer")
            or get_path(payload, "structure_layer"),
            default=fallback_layer,
        )

        if event_type is None and direction is None and regime is None:
            return None

        confidence = unit_score(
            get_path(payload, "confidence")
            or get_path(payload, "event_confidence")
        )
        score = unit_score(
            get_path(payload, "score")
            or get_path(payload, "event_score")
            or confidence
        )

        reasons_raw = (
            get_path(payload, "reasons")
            or get_path(payload, "reason")
            or get_path(payload, "confirmations")
            or []
        )
        reasons: list[str] = []
        if isinstance(reasons_raw, str):
            reasons = [reasons_raw] if reasons_raw.strip() else []
        elif isinstance(reasons_raw, (list, tuple, set)):
            reasons = [str(item).strip() for item in reasons_raw if str(item).strip()]

        return cls(
            event_type=event_type,
            direction=direction,
            regime=regime,
            layer=layer,
            confidence=confidence,
            score=score,
            continuation_probability=unit_score(
                get_path(payload, "continuation_probability")
                or get_path(payload, "continuation_prob")
                or get_path(payload, "probability")
            ),
            reversal_risk=unit_score(
                get_path(payload, "reversal_risk")
                or get_path(payload, "reversal_probability")
                or get_path(payload, "countertrend_risk")
            ),
            exhaustion_score=unit_score(
                get_path(payload, "exhaustion_score")
                or get_path(payload, "exhaustion")
            ),
            timestamp=parse_datetime(
                get_path(payload, "timestamp")
                or get_path(payload, "event_time")
                or get_path(payload, "created_at")
                or get_path(payload, "time")
            ),
            is_confirmed=to_bool(
                get_path(payload, "confirmed")
                or get_path(payload, "is_confirmed")
                or get_path(payload, "valid"),
                default=confidence > 0.0,
            ),
            reasons=list(dict.fromkeys(reasons)),
            raw=serialize_for_metadata(payload)
            if isinstance(payload, dict)
            else {"raw": serialize_for_metadata(payload)},
        )


@dataclass(slots=True)
class TrendLayerContext:
    """
    Normalized trend layer view.
    """

    direction: TrendDirection | None = None
    regime: TrendRegime | None = None
    layer: StructureLayer | None = None

    confidence: float = 0.0
    score: float = 0.0
    trend_strength: float = 0.0
    momentum_score: float = 0.0
    slope_score: float = 0.0
    continuation_probability: float = 0.0
    reversal_risk: float = 0.0
    exhaustion_score: float = 0.0
    pullback_quality: float = 0.0
    structure_score: float = 0.0

    is_exhausted: bool = False
    is_pullback: bool = False
    is_countertrend: bool = False

    updated_at: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        fallback_layer: StructureLayer | None = None,
    ) -> TrendLayerContext | None:
        if payload is None:
            return None

        direction = parse_trend_direction(
            get_path(payload, "direction")
            or get_path(payload, "trend_direction")
            or get_path(payload, "side")
        )
        regime = parse_trend_regime(
            get_path(payload, "regime")
            or get_path(payload, "trend_regime")
            or get_path(payload, "state")
        )
        layer = parse_structure_layer(
            get_path(payload, "layer")
            or get_path(payload, "structure_layer"),
            default=fallback_layer,
        )

        if direction is None and regime is None:
            return None

        confidence = unit_score(
            get_path(payload, "confidence")
            or get_path(payload, "trend_confidence")
        )
        trend_strength = unit_score(
            get_path(payload, "trend_strength")
            or get_path(payload, "strength")
            or get_path(payload, "directional_strength")
        )
        score = unit_score(
            get_path(payload, "score")
            or get_path(payload, "trend_score")
            or get_path(payload, "overall_score")
            or trend_strength
            or confidence
        )

        return cls(
            direction=direction,
            regime=regime,
            layer=layer,
            confidence=confidence,
            score=score,
            trend_strength=trend_strength,
            momentum_score=unit_score(
                get_path(payload, "momentum_score")
                or get_path(payload, "momentum")
                or get_path(payload, "directional_momentum")
            ),
            slope_score=unit_score(
                get_path(payload, "slope_score")
                or get_path(payload, "slope")
                or get_path(payload, "trend_slope")
            ),
            continuation_probability=unit_score(
                get_path(payload, "continuation_probability")
                or get_path(payload, "continuation_prob")
            ),
            reversal_risk=unit_score(
                get_path(payload, "reversal_risk")
                or get_path(payload, "reversal_probability")
                or get_path(payload, "countertrend_risk")
            ),
            exhaustion_score=unit_score(
                get_path(payload, "exhaustion_score")
                or get_path(payload, "exhaustion")
            ),
            pullback_quality=unit_score(
                get_path(payload, "pullback_quality")
                or get_path(payload, "pullback_score")
                or get_path(payload, "retracement_quality")
            ),
            structure_score=unit_score(
                get_path(payload, "structure_score")
                or get_path(payload, "market_structure_score")
                or get_path(payload, "structure_alignment_score")
            ),
            is_exhausted=to_bool(
                get_path(payload, "is_exhausted")
                or get_path(payload, "exhausted"),
                default=False,
            ),
            is_pullback=to_bool(
                get_path(payload, "is_pullback")
                or get_path(payload, "pullback"),
                default=False,
            ),
            is_countertrend=to_bool(
                get_path(payload, "is_countertrend")
                or get_path(payload, "countertrend"),
                default=False,
            ),
            updated_at=parse_datetime(
                get_path(payload, "updated_at")
                or get_path(payload, "last_update")
                or get_path(payload, "timestamp")
                or get_path(payload, "time")
            ),
            raw=serialize_for_metadata(payload)
            if isinstance(payload, dict)
            else {"raw": serialize_for_metadata(payload)},
        )


@dataclass(slots=True)
class TrendContinuationContextView:
    """
    Normalized trend-continuation view consumed by TrendContinuationStrategy.
    """

    module: dict[str, Any]
    primary_layer: dict[str, Any]
    secondary_layer: dict[str, Any]

    primary_trend: TrendLayerContext | None = None
    secondary_trend: TrendLayerContext | None = None
    last_event: TrendEventContext | None = None

    primary_layer_name: StructureLayer | None = None
    secondary_layer_name: StructureLayer | None = None

    internal_external_alignment: float = 0.0
    higher_timeframe_alignment: float = 0.0
    overall_trend_score: float = 0.0
    layer_confidence: float = 0.0
    layer_strength: float = 0.0

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrendContinuationStrategyConfig(PriceActionStrategyConfig):
    """
    Unified trend-continuation strategy config.

    Strategy idea:
    - read normalized trend context from StrategyContext;
    - require directional trend, strength, continuation probability and alignment;
    - optionally allow pullback continuation entries;
    - return internal StrategySignal only;
    - leave routing, filtering, confluence, portfolio coordination and
      risk-ready conversion to SignalProcessor.
    """

    prefer_external_layer: bool = True

    require_fresh_trend: bool = True
    require_internal_confirmation: bool = True
    allow_cross_layer_fallback: bool = True
    require_direction_alignment: bool = True

    allow_pullback_entries: bool = True
    block_exhausted_trend: bool = True
    block_high_reversal_risk: bool = True
    block_countertrend_context: bool = True

    min_layer_confidence: float = 0.50
    min_trend_strength: float = 0.55
    min_continuation_probability: float = 0.55
    min_directional_momentum: float = 0.10
    min_slope_score: float = 0.05
    min_overall_trend_score: float = 0.35
    min_internal_external_alignment: float = 0.30
    min_higher_timeframe_alignment: float = 0.20

    max_reversal_risk: float = 0.45
    max_exhaustion_score: float = 0.65

    pullback_bonus: float = 0.05
    alignment_bonus: float = 0.05
    higher_timeframe_bonus: float = 0.04
    strong_trend_bonus: float = 0.05
    momentum_bonus: float = 0.04
    event_confirmation_bonus: float = 0.03

    strong_trend_threshold: float = 0.75
    strong_momentum_threshold: float = 0.60
    strong_continuation_probability_threshold: float = 0.70

    score_trend_weight: float = 0.28
    score_probability_weight: float = 0.22
    score_momentum_weight: float = 0.16
    score_alignment_weight: float = 0.14
    score_structure_weight: float = 0.10
    score_freshness_weight: float = 0.10

    confidence_trend_weight: float = 0.52
    confidence_context_weight: float = 0.28
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    tag_trend_continuation: str = "trend_continuation"
    tag_uptrend_continuation: str = "uptrend_continuation"
    tag_downtrend_continuation: str = "downtrend_continuation"
    tag_pullback_entry: str = "pullback_entry"
    tag_momentum: str = "momentum"
    tag_alignment: str = "alignment"
    tag_higher_timeframe: str = "higher_timeframe"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.CONTINUATION

    required_price_action_features: tuple[str, ...] = (
        PRICE_ACTION_FEATURES.TREND,
    )

    def validate(self) -> None:
        PriceActionStrategyConfig.validate(self)

        unit_fields = {
            "min_layer_confidence": self.min_layer_confidence,
            "min_trend_strength": self.min_trend_strength,
            "min_continuation_probability": self.min_continuation_probability,
            "min_directional_momentum": self.min_directional_momentum,
            "min_slope_score": self.min_slope_score,
            "min_overall_trend_score": self.min_overall_trend_score,
            "min_internal_external_alignment": self.min_internal_external_alignment,
            "min_higher_timeframe_alignment": self.min_higher_timeframe_alignment,
            "max_reversal_risk": self.max_reversal_risk,
            "max_exhaustion_score": self.max_exhaustion_score,
            "pullback_bonus": self.pullback_bonus,
            "alignment_bonus": self.alignment_bonus,
            "higher_timeframe_bonus": self.higher_timeframe_bonus,
            "strong_trend_bonus": self.strong_trend_bonus,
            "momentum_bonus": self.momentum_bonus,
            "event_confirmation_bonus": self.event_confirmation_bonus,
            "strong_trend_threshold": self.strong_trend_threshold,
            "strong_momentum_threshold": self.strong_momentum_threshold,
            "strong_continuation_probability_threshold": self.strong_continuation_probability_threshold,
        }

        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        score_weights = {
            "score_trend_weight": self.score_trend_weight,
            "score_probability_weight": self.score_probability_weight,
            "score_momentum_weight": self.score_momentum_weight,
            "score_alignment_weight": self.score_alignment_weight,
            "score_structure_weight": self.score_structure_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }
        confidence_weights = {
            "confidence_trend_weight": self.confidence_trend_weight,
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
            "tag_trend_continuation",
            "tag_uptrend_continuation",
            "tag_downtrend_continuation",
            "tag_pullback_entry",
            "tag_momentum",
            "tag_alignment",
            "tag_higher_timeframe",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_price_action_features:
            raise StrategyConfigError("required_price_action_features cannot be empty")

        for feature in self.required_price_action_features:
            if not isinstance(feature, str) or not feature.strip():
                raise StrategyConfigError(
                    "required_price_action_features cannot contain empty feature names"
                )


class TrendContinuationStrategy(PriceActionTradingStrategy):
    """
    Unified trend-continuation strategy.

    Input:
        StrategyContext with FeatureSource.PRICE_ACTION domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.price_action.trend_continuation"
    category: StrategyCategory = StrategyCategory.PRICE_ACTION
    default_setup_type: SetupType = SetupType.CONTINUATION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        price_action_config: TrendContinuationStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_price_action_config = (
            price_action_config or TrendContinuationStrategyConfig()
        )
        resolved_price_action_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            price_action_config=resolved_price_action_config,
            service_name=service_name,
        )

        self.trend_config: TrendContinuationStrategyConfig = (
            resolved_price_action_config
        )

    @property
    def strategy_name(self) -> str:
        return "trend_continuation"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.PRICE_ACTION,
            timeframe=Timeframe.M1,
            tags=[
                self.trend_config.tag_price_action,
                self.trend_config.tag_trend,
                self.trend_config.tag_trend_continuation,
                self.trend_config.tag_continuation,
                self.trend_config.tag_momentum,
                self.trend_config.tag_alignment,
                "analytics_price_action",
            ],
            version="2.0.0",
            description=(
                "Interprets trend direction, strength, continuation probability, "
                "momentum, slope and layer alignment from normalized price-action "
                "StrategyContext and returns internal StrategySignal."
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
                "source": "analytics.price_action",
                "strategy_type": "trend_continuation",
                "base_class": "PriceActionTradingStrategy",
                "canonical_payload": "PriceActionCompositeSnapshot",
                "uses_trend": True,
                "uses_alignment": True,
                "uses_pullback_entries": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(
            self.trend_config.required_price_action_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        if not self.has_any_price_action_data(
            context,
            tuple(self.trend_config.required_price_action_features),
        ):
            self.remember_no_signal(
                "missing_price_action_trend_context",
                price_action_domain_keys=sorted(self.price_action_domain(context).keys()),
                required_features=sorted(self.required_features()),
            )
            return None

        if self.has_stale_price_action_features(
            context,
            tuple(self.trend_config.required_price_action_features),
        ):
            self.remember_no_signal(
                "stale_price_action_trend_features",
                required_features=sorted(self.trend_config.required_price_action_features),
            )
            return None

        view = self._extract_view(context)
        if view is None:
            self.remember_no_signal(
                "trend_view_not_resolved",
                trend=serialize_for_metadata(self.price_action_item(context, "trend")),
                price_action_domain_keys=sorted(self.price_action_domain(context).keys()),
            )
            return None

        if view.primary_trend is None:
            self.remember_no_signal(
                "missing_primary_trend_context",
                trend=serialize_for_metadata(self.price_action_item(context, "trend")),
                last_event=serialize_for_metadata(view.last_event),
            )
            return None

        if (
            self.trend_config.require_fresh_trend
            and is_stale(
                event_time=view.event_time,
                now=context.timestamp,
                stale_after_seconds=self.trend_config.stale_feature_max_age_seconds,
            )
        ):
            self.remember_no_signal(
                "stale_trend_event",
                event_time=view.event_time.isoformat() if view.event_time else None,
                context_timestamp=context.timestamp.isoformat(),
                stale_after_seconds=self.trend_config.stale_feature_max_age_seconds,
            )
            return None

        common_rejection = quality_filter_reason(
            view.primary_layer,
            min_confidence=self.trend_config.min_layer_confidence,
            min_score=self.trend_config.min_trend_strength,
            stale_after_seconds=self.trend_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )
        if common_rejection is not None:
            self.remember_no_signal(
                "trend_primary_layer_rejected",
                rejection=common_rejection,
                primary_layer=serialize_for_metadata(view.primary_layer),
                min_layer_confidence=self.trend_config.min_layer_confidence,
                min_trend_strength=self.trend_config.min_trend_strength,
            )
            return None

        side = self._infer_side(view)
        if not is_directional_side(side):
            self.remember_no_signal(
                "trend_side_not_directional",
                primary_trend=serialize_for_metadata(view.primary_trend),
                last_event=serialize_for_metadata(view.last_event),
            )
            return None

        if not self._passes_filters(view=view, side=side):
            self.remember_no_signal(
                "trend_filters_failed",
                side=side.value,
                primary_trend=serialize_for_metadata(view.primary_trend),
                secondary_trend=serialize_for_metadata(view.secondary_trend),
                last_event=serialize_for_metadata(view.last_event),
                internal_external_alignment=view.internal_external_alignment,
                higher_timeframe_alignment=view.higher_timeframe_alignment,
                overall_trend_score=view.overall_trend_score,
                layer_confidence=view.layer_confidence,
                layer_strength=view.layer_strength,
            )
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            view=view,
            side=side,
        )

        if breakdown.score < self.trend_config.min_signal_score:
            self.remember_no_signal(
                "trend_score_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_signal_score=self.trend_config.min_signal_score,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        if breakdown.confidence < self.trend_config.min_signal_confidence:
            self.remember_no_signal(
                "trend_confidence_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_signal_confidence=self.trend_config.min_signal_confidence,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        source_features = self._source_features(view)
        tags = self._tags(view=view)

        event_label = (
            normalize_label(view.last_event.event_type)
            if view.last_event is not None
            else "trend_state"
        )

        reasons = list(
            dict.fromkeys(
                [
                    "trend_continuation_signal",
                    f"side:{side.value}",
                    f"setup_type:{SetupType.CONTINUATION.value}",
                    f"event:{event_label}",
                    *view.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "price_action_setup_family": "trend_continuation",
            "price_action_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "event_time": view.event_time.isoformat() if view.event_time else None,
            "primary_layer_name": normalize_label(view.primary_layer_name),
            "secondary_layer_name": normalize_label(view.secondary_layer_name),
            "primary_trend": serialize_for_metadata(view.primary_trend),
            "secondary_trend": serialize_for_metadata(view.secondary_trend),
            "last_event": serialize_for_metadata(view.last_event),
            "primary_layer": serialize_for_metadata(view.primary_layer),
            "secondary_layer": serialize_for_metadata(view.secondary_layer),
            "internal_external_alignment": view.internal_external_alignment,
            "higher_timeframe_alignment": view.higher_timeframe_alignment,
            "overall_trend_score": view.overall_trend_score,
            "layer_confidence": view.layer_confidence,
            "layer_strength": view.layer_strength,
            "mapped_side": side.value,
            "setup_type": SetupType.CONTINUATION.value,
            "raw": serialize_for_metadata(view.raw),
        }

        return self.build_price_action_signal(
            context=context,
            side=side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=SetupType.CONTINUATION,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.trend_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_view(
        self,
        context: StrategyContext,
    ) -> TrendContinuationContextView | None:
        module = self.resolve_price_action_module(
            context,
            "trend",
        )
        if not module:
            return None

        primary = select_primary_layer(
            module,
            prefer_external_layer=self.trend_config.prefer_external_layer,
        )
        secondary = select_secondary_layer(
            module,
            prefer_external_layer=self.trend_config.prefer_external_layer,
        )

        if not primary:
            return None

        primary_layer_name = self._extract_layer_name(
            primary,
            fallback=StructureLayer.EXTERNAL
            if self.trend_config.prefer_external_layer
            else StructureLayer.INTERNAL,
        )
        secondary_layer_name = self._extract_layer_name(
            secondary,
            fallback=StructureLayer.INTERNAL
            if self.trend_config.prefer_external_layer
            else StructureLayer.EXTERNAL,
        )

        primary_trend = TrendLayerContext.from_payload(
            primary,
            fallback_layer=primary_layer_name,
        )
        secondary_trend = TrendLayerContext.from_payload(
            secondary,
            fallback_layer=secondary_layer_name,
        )

        last_event_payload = (
            get_path(primary, "last_signal")
            or get_path(primary, "last_event")
            or get_path(module, "last_signal")
            or get_path(module, "last_event")
            or extract_last_event(module)
        )
        last_event = TrendEventContext.from_payload(
            last_event_payload,
            fallback_layer=primary_layer_name,
        )

        if primary_trend is None:
            return None

        internal_external_alignment = unit_score(
            get_path(module, "internal_external_alignment")
            or get_path(module, "internal_external_alignment_score")
            or get_path(module, "alignment_score")
            or get_path(primary, "alignment_score")
        )
        higher_timeframe_alignment = unit_score(
            get_path(module, "higher_timeframe_alignment")
            or get_path(module, "higher_timeframe_alignment_score")
            or get_path(module, "htf_alignment")
            or get_path(primary, "higher_timeframe_alignment")
        )
        overall_trend_score = unit_score(
            get_path(module, "overall_trend_score")
            or get_path(module, "trend_score")
            or get_path(primary, "overall_trend_score")
            or primary_trend.score
        )

        event_time = (
            last_event.timestamp if last_event is not None else None
        ) or primary_trend.updated_at or extract_last_update(primary) or extract_last_update(module)

        reasons = self._extract_reasons(module, primary, last_event)

        return TrendContinuationContextView(
            module=module,
            primary_layer=primary,
            secondary_layer=secondary,
            primary_trend=primary_trend,
            secondary_trend=secondary_trend,
            last_event=last_event,
            primary_layer_name=primary_layer_name,
            secondary_layer_name=secondary_layer_name,
            internal_external_alignment=internal_external_alignment,
            higher_timeframe_alignment=higher_timeframe_alignment,
            overall_trend_score=overall_trend_score,
            layer_confidence=layer_confidence(primary),
            layer_strength=layer_strength(primary),
            event_time=event_time,
            reasons=reasons,
            raw=module,
        )

    @staticmethod
    def _extract_layer_name(
        layer: dict[str, Any],
        *,
        fallback: StructureLayer,
    ) -> StructureLayer:
        return (
            parse_structure_layer(
                get_path(layer, "layer")
                or get_path(layer, "structure_layer")
                or get_path(layer, "name"),
                default=fallback,
            )
            or fallback
        )

    @staticmethod
    def _extract_reasons(
        module: dict[str, Any],
        primary: dict[str, Any],
        event: TrendEventContext | None,
    ) -> list[str]:
        reasons: list[str] = []

        for value in (
            get_path(module, "reasons"),
            get_path(primary, "reasons"),
            get_path(module, "confirmations"),
            get_path(primary, "confirmations"),
        ):
            if isinstance(value, str) and value.strip():
                reasons.append(value.strip())
            elif isinstance(value, (list, tuple, set)):
                reasons.extend(str(item).strip() for item in value if str(item).strip())

        if event is not None:
            reasons.extend(event.reasons)

        return list(dict.fromkeys(reasons))

    # ------------------------------------------------------------------
    # Mapping / filters
    # ------------------------------------------------------------------

    def _infer_side(self, view: TrendContinuationContextView) -> SignalSide:
        trend = view.primary_trend
        if trend is None:
            return SignalSide.UNKNOWN

        side = trend_direction_to_side(trend.direction)
        if is_directional_side(side):
            return side

        if view.last_event is not None and view.last_event.direction is not None:
            event_side = trend_direction_to_side(view.last_event.direction)
            if is_directional_side(event_side):
                return event_side

        return SignalSide.UNKNOWN

    def _passes_filters(
        self,
        *,
        view: TrendContinuationContextView,
        side: SignalSide,
    ) -> bool:
        trend = view.primary_trend
        if trend is None:
            return False

        if trend.confidence < self.trend_config.min_layer_confidence:
            return False

        if trend.trend_strength < self.trend_config.min_trend_strength:
            return False

        if trend.continuation_probability < self.trend_config.min_continuation_probability:
            return False

        if trend.momentum_score < self.trend_config.min_directional_momentum:
            return False

        if trend.slope_score < self.trend_config.min_slope_score:
            return False

        if view.overall_trend_score < self.trend_config.min_overall_trend_score:
            return False

        if self.trend_config.require_direction_alignment:
            if not self._direction_alignment_ok(view=view, side=side):
                return False

        if self.trend_config.require_internal_confirmation:
            if view.internal_external_alignment < self.trend_config.min_internal_external_alignment:
                if not self.trend_config.allow_cross_layer_fallback:
                    return False

        if view.higher_timeframe_alignment < self.trend_config.min_higher_timeframe_alignment:
            return False

        if self.trend_config.block_exhausted_trend:
            if trend.is_exhausted or trend.exhaustion_score > self.trend_config.max_exhaustion_score:
                return False

        if self.trend_config.block_high_reversal_risk:
            if trend.reversal_risk > self.trend_config.max_reversal_risk:
                return False

        if self.trend_config.block_countertrend_context and trend.is_countertrend:
            return False

        if trend.is_pullback and not self.trend_config.allow_pullback_entries:
            return False

        if not is_directional_side(side):
            return False

        return True

    def _direction_alignment_ok(
        self,
        *,
        view: TrendContinuationContextView,
        side: SignalSide,
    ) -> bool:
        secondary = view.secondary_trend
        if secondary is None:
            return True

        secondary_side = trend_direction_to_side(secondary.direction)
        if not is_directional_side(secondary_side):
            return True

        return secondary_side is side

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        view: TrendContinuationContextView,
        side: SignalSide,
    ) -> ScoreBreakdown:
        trend = view.primary_trend
        if trend is None:
            return ScoreBreakdown()

        trend_component = average_score(
            trend.score,
            trend.confidence,
            trend.trend_strength,
        )
        probability_component = trend.continuation_probability
        momentum_component = average_score(
            trend.momentum_score,
            trend.slope_score,
        )
        alignment_component = average_score(
            view.internal_external_alignment,
            view.higher_timeframe_alignment,
        )
        structure_component = average_score(
            trend.structure_score,
            view.overall_trend_score,
            view.layer_strength,
        )
        fresh_component = freshness_score(
            event_time=view.event_time,
            now=context.timestamp,
            stale_after_seconds=self.trend_config.stale_feature_max_age_seconds,
        )

        components = {
            "trend": trend_component,
            "probability": probability_component,
            "momentum": momentum_component,
            "alignment": alignment_component,
            "structure": structure_component,
            "freshness": fresh_component,
        }
        weights = {
            "trend": self.trend_config.score_trend_weight,
            "probability": self.trend_config.score_probability_weight,
            "momentum": self.trend_config.score_momentum_weight,
            "alignment": self.trend_config.score_alignment_weight,
            "structure": self.trend_config.score_structure_weight,
            "freshness": self.trend_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=trend_component)
        confidence = confidence_from_components(
            primary=trend_component,
            context=weighted_score(
                {
                    "structure": structure_component,
                    "alignment": alignment_component,
                    "probability": probability_component,
                },
                {
                    "structure": 0.30,
                    "alignment": 0.35,
                    "probability": 0.35,
                },
            ),
            confirmation=momentum_component,
            freshness=fresh_component,
            primary_weight=self.trend_config.confidence_trend_weight,
            context_weight=self.trend_config.confidence_context_weight,
            confirmation_weight=self.trend_config.confidence_confirmation_weight,
            freshness_weight=self.trend_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            f"side:{side.value}",
            "trend_continuation_context",
            f"trend_direction:{normalize_label(trend.direction)}",
        ]

        if trend.trend_strength >= self.trend_config.strong_trend_threshold:
            score += self.trend_config.strong_trend_bonus
            confirmations.append("strong_trend_strength")

        if trend.momentum_score >= self.trend_config.strong_momentum_threshold:
            score += self.trend_config.momentum_bonus
            confirmations.append("strong_directional_momentum")

        if trend.continuation_probability >= self.trend_config.strong_continuation_probability_threshold:
            score += self.trend_config.strong_trend_bonus
            confirmations.append("high_continuation_probability")

        if trend.is_pullback:
            score += self.trend_config.pullback_bonus
            confirmations.append("pullback_continuation_entry")

        if view.internal_external_alignment >= self.trend_config.min_internal_external_alignment:
            score += self.trend_config.alignment_bonus
            confirmations.append("internal_external_alignment_confirmed")

        if view.higher_timeframe_alignment >= self.trend_config.min_higher_timeframe_alignment:
            score += self.trend_config.higher_timeframe_bonus
            confirmations.append("higher_timeframe_alignment_confirmed")

        if view.last_event is not None:
            score += self.trend_config.event_confirmation_bonus
            confirmations.append(
                f"trend_event:{normalize_label(view.last_event.event_type)}"
            )

        if trend.exhaustion_score > 0:
            reasons.append(f"exhaustion_score:{trend.exhaustion_score:.4f}")

        if trend.reversal_risk > 0:
            reasons.append(f"reversal_risk:{trend.reversal_risk:.4f}")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    # ------------------------------------------------------------------
    # Source features / tags
    # ------------------------------------------------------------------

    def _source_features(self, view: TrendContinuationContextView) -> list[str]:
        features = [
            *trend_source_features(),
            PRICE_ACTION_FEATURES.TREND,
            PRICE_ACTION_FEATURES.TREND_INTERNAL,
            PRICE_ACTION_FEATURES.TREND_EXTERNAL,
            PRICE_ACTION_FEATURES.TREND_LAST_SIGNAL,
            PRICE_ACTION_FEATURES.TREND_INTERNAL_EXTERNAL_ALIGNMENT,
            PRICE_ACTION_FEATURES.TREND_HIGHER_TIMEFRAME_ALIGNMENT,
            PRICE_ACTION_FEATURES.TREND_OVERALL_SCORE,
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        *,
        view: TrendContinuationContextView,
    ) -> list[str]:
        tags = [
            self.trend_config.tag_price_action,
            self.trend_config.tag_trend,
            self.trend_config.tag_trend_continuation,
            self.trend_config.tag_continuation,
            self.trend_config.tag_momentum,
            self.trend_config.tag_alignment,
        ]

        trend = view.primary_trend
        if trend is not None:
            side = trend_direction_to_side(trend.direction)

            if side is SignalSide.LONG:
                tags.append(self.trend_config.tag_uptrend_continuation)

            if side is SignalSide.SHORT:
                tags.append(self.trend_config.tag_downtrend_continuation)

            if trend.is_pullback:
                tags.append(self.trend_config.tag_pullback_entry)

            if trend.layer is not None:
                tags.append(f"layer:{normalize_label(trend.layer)}")

            if trend.regime is not None:
                tags.append(f"regime:{normalize_label(trend.regime)}")

        if view.higher_timeframe_alignment > 0:
            tags.append(self.trend_config.tag_higher_timeframe)

        return list(dict.fromkeys(tags))