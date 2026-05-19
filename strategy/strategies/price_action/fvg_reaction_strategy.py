# trading_system/strategy/strategies/price_action/fvg_reaction_strategy.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from analytics.price_action.enums import (
    FVGDirection,
    FVGEventType,
    FVGStatus,
    StructureLayer,
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
    PRICE_ACTION_FEATURES,
    PriceActionStrategyConfig,
    PriceActionTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    average_score,
    confidence_from_components,
    distance_score,
    extract_last_event,
    extract_last_update,
    fvg_direction_to_side,
    fvg_source_features,
    get_path,
    is_directional_side,
    is_stale,
    layer_confidence,
    layer_strength,
    normalize_label,
    parse_datetime,
    parse_fvg_direction,
    parse_fvg_event_type,
    parse_fvg_status,
    parse_structure_layer,
    quality_filter_reason,
    select_primary_layer,
    select_secondary_layer,
    serialize_for_metadata,
    to_bool,
    to_float,
    unit_score,
    weighted_score,
    freshness_score,
)


@dataclass(slots=True)
class FVGContext:
    """
    Normalized strategy-level FVG context.

    This DTO belongs to strategy layer only. Analytics models remain in
    analytics.price_action.
    """

    direction: FVGDirection | None = None
    status: FVGStatus | None = None
    layer: StructureLayer | None = None

    upper_price: float | None = None
    lower_price: float | None = None
    mid_price: float | None = None
    current_price: float | None = None

    fill_pct: float = 0.0
    gap_size_pct: float = 0.0
    distance_to_mid_pct: float = 0.0
    distance_to_nearest_edge_pct: float = 0.0

    strength: float = 0.0
    confidence: float = 0.0
    score: float = 0.0
    touches: int = 0

    created_at: datetime | None = None
    updated_at: datetime | None = None

    is_mitigated: bool = False
    is_valid: bool = True

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        fallback_layer: StructureLayer | None = None,
    ) -> FVGContext | None:
        if payload is None:
            return None

        direction = parse_fvg_direction(
            get_path(payload, "direction")
            or get_path(payload, "fvg_direction")
            or get_path(payload, "side")
        )
        status = parse_fvg_status(
            get_path(payload, "status")
            or get_path(payload, "fvg_status")
            or get_path(payload, "state")
        )
        layer = parse_structure_layer(
            get_path(payload, "layer")
            or get_path(payload, "structure_layer"),
            default=fallback_layer,
        )

        upper_price = to_float(
            get_path(payload, "upper_price")
            or get_path(payload, "upper")
            or get_path(payload, "high")
        )
        lower_price = to_float(
            get_path(payload, "lower_price")
            or get_path(payload, "lower")
            or get_path(payload, "low")
        )
        mid_price = to_float(
            get_path(payload, "mid_price")
            or get_path(payload, "mid")
        )

        if mid_price is None and upper_price is not None and lower_price is not None:
            mid_price = (upper_price + lower_price) / 2.0

        if direction is None and upper_price is None and lower_price is None:
            return None

        confidence = unit_score(
            get_path(payload, "confidence")
            or get_path(payload, "event_confidence")
            or get_path(payload, "gap_confidence")
        )
        strength = unit_score(
            get_path(payload, "strength")
            or get_path(payload, "gap_strength")
            or get_path(payload, "quality")
            or confidence
        )
        score = unit_score(
            get_path(payload, "score")
            or get_path(payload, "gap_score")
            or strength
            or confidence
        )

        return cls(
            direction=direction,
            status=status,
            layer=layer,
            upper_price=upper_price,
            lower_price=lower_price,
            mid_price=mid_price,
            current_price=to_float(
                get_path(payload, "current_price")
                or get_path(payload, "price")
                or get_path(payload, "last_price")
            ),
            fill_pct=unit_score(
                get_path(payload, "fill_pct")
                or get_path(payload, "filled_pct")
                or get_path(payload, "mitigation_pct")
                or get_path(payload, "fill_ratio")
            ),
            gap_size_pct=abs(
                to_float(
                    get_path(payload, "gap_size_pct")
                    or get_path(payload, "size_pct")
                    or get_path(payload, "width_pct"),
                    0.0,
                )
                or 0.0
            ),
            distance_to_mid_pct=abs(
                to_float(
                    get_path(payload, "distance_to_mid_pct")
                    or get_path(payload, "mid_distance_pct")
                    or get_path(payload, "distance_pct"),
                    0.0,
                )
                or 0.0
            ),
            distance_to_nearest_edge_pct=abs(
                to_float(
                    get_path(payload, "distance_to_nearest_edge_pct")
                    or get_path(payload, "edge_distance_pct")
                    or get_path(payload, "distance_to_gap_pct"),
                    0.0,
                )
                or 0.0
            ),
            strength=strength,
            confidence=confidence,
            score=score,
            touches=int(
                to_float(
                    get_path(payload, "touches")
                    or get_path(payload, "touch_count"),
                    0,
                )
                or 0
            ),
            created_at=parse_datetime(
                get_path(payload, "created_at")
                or get_path(payload, "timestamp")
                or get_path(payload, "time")
            ),
            updated_at=parse_datetime(
                get_path(payload, "updated_at")
                or get_path(payload, "last_update")
                or get_path(payload, "event_time")
            ),
            is_mitigated=to_bool(
                get_path(payload, "is_mitigated")
                or get_path(payload, "mitigated"),
                default=False,
            ),
            is_valid=to_bool(
                get_path(payload, "is_valid")
                or get_path(payload, "valid"),
                default=True,
            ),
            raw=serialize_for_metadata(payload)
            if isinstance(payload, dict)
            else {"raw": serialize_for_metadata(payload)},
        )


@dataclass(slots=True)
class FVGEventContext:
    """
    Normalized last FVG lifecycle event.
    """

    event_type: FVGEventType | None = None
    direction: FVGDirection | None = None
    status: FVGStatus | None = None
    layer: StructureLayer | None = None

    confidence: float = 0.0
    score: float = 0.0
    fill_pct: float = 0.0
    distance_to_mid_pct: float = 0.0

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
    ) -> FVGEventContext | None:
        if payload is None:
            return None

        event_type = parse_fvg_event_type(
            get_path(payload, "event_type")
            or get_path(payload, "type")
            or get_path(payload, "kind")
        )
        direction = parse_fvg_direction(
            get_path(payload, "direction")
            or get_path(payload, "fvg_direction")
            or get_path(payload, "side")
        )
        status = parse_fvg_status(
            get_path(payload, "status")
            or get_path(payload, "fvg_status")
            or get_path(payload, "state")
        )
        layer = parse_structure_layer(
            get_path(payload, "layer")
            or get_path(payload, "structure_layer"),
            default=fallback_layer,
        )

        if event_type is None and direction is None and status is None:
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
            status=status,
            layer=layer,
            confidence=confidence,
            score=score,
            fill_pct=unit_score(
                get_path(payload, "fill_pct")
                or get_path(payload, "filled_pct")
                or get_path(payload, "mitigation_pct")
            ),
            distance_to_mid_pct=abs(
                to_float(
                    get_path(payload, "distance_to_mid_pct")
                    or get_path(payload, "mid_distance_pct")
                    or get_path(payload, "distance_pct"),
                    0.0,
                )
                or 0.0
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
class FVGReactionContext:
    """
    Normalized FVG reaction view consumed by FVGReactionStrategy.
    """

    module: dict[str, Any]
    primary_layer: dict[str, Any]
    secondary_layer: dict[str, Any]

    primary_layer_name: StructureLayer | None = None
    secondary_layer_name: StructureLayer | None = None

    reaction_gap: FVGContext | None = None
    last_event: FVGEventContext | None = None

    layer_confidence: float = 0.0
    layer_strength: float = 0.0
    layer_alignment_score: float = 0.0
    proximity_score: float = 0.0
    fill_quality_score: float = 0.0

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FVGReactionStrategyConfig(PriceActionStrategyConfig):
    """
    Unified FVG reaction strategy config.

    Strategy idea:
    - read normalized fair_value_gap context from StrategyContext;
    - select active/respected/retested/partially filled FVG;
    - generate reaction / continuation / retest signal;
    - return internal StrategySignal only;
    - leave routing, filtering, confluence, portfolio coordination and
      risk-ready conversion to SignalProcessor.
    """

    prefer_external_layer: bool = True

    require_fresh_fvg: bool = True
    require_recent_event: bool = True
    require_directional_gap: bool = True
    require_respected_or_retested: bool = False
    require_primary_layer_eligible: bool = True

    allow_active_gap_proximity_entry: bool = True
    allow_fill_started_reaction: bool = True
    allow_partial_fill_reaction: bool = True
    allow_respected_reaction: bool = True
    allow_retested_reaction: bool = True
    allow_created_gap_continuation: bool = False

    block_invalidated_gaps: bool = True
    block_filled_gaps: bool = True

    min_layer_confidence: float = 0.40
    min_gap_strength: float = 0.45
    min_event_confidence: float = 0.45
    min_gap_fill_for_reaction: float = 0.05
    max_gap_fill_for_entry: float = 0.90
    max_distance_to_mid_pct: float = 0.0035
    max_distance_to_edge_pct: float = 0.0035

    respected_bonus: float = 0.05
    retested_bonus: float = 0.06
    partial_fill_bonus: float = 0.04
    proximity_bonus: float = 0.04
    layer_alignment_bonus: float = 0.03

    score_gap_weight: float = 0.30
    score_event_weight: float = 0.24
    score_proximity_weight: float = 0.16
    score_fill_weight: float = 0.14
    score_layer_weight: float = 0.10
    score_freshness_weight: float = 0.06

    confidence_gap_weight: float = 0.52
    confidence_context_weight: float = 0.28
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    tag_fvg_reaction: str = "fvg_reaction"
    tag_fvg_respected: str = "fvg_respected"
    tag_fvg_retested: str = "fvg_retested"
    tag_fvg_partial_fill: str = "fvg_partial_fill"
    tag_fvg_fill_started: str = "fvg_fill_started"
    tag_fvg_proximity: str = "fvg_proximity"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.RETEST

    required_price_action_features: tuple[str, ...] = (
        PRICE_ACTION_FEATURES.FAIR_VALUE_GAP,
    )

    def validate(self) -> None:
        PriceActionStrategyConfig.validate(self)

        unit_fields = {
            "min_layer_confidence": self.min_layer_confidence,
            "min_gap_strength": self.min_gap_strength,
            "min_event_confidence": self.min_event_confidence,
            "min_gap_fill_for_reaction": self.min_gap_fill_for_reaction,
            "max_gap_fill_for_entry": self.max_gap_fill_for_entry,
            "respected_bonus": self.respected_bonus,
            "retested_bonus": self.retested_bonus,
            "partial_fill_bonus": self.partial_fill_bonus,
            "proximity_bonus": self.proximity_bonus,
            "layer_alignment_bonus": self.layer_alignment_bonus,
        }

        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.max_distance_to_mid_pct < 0:
            raise StrategyConfigError("max_distance_to_mid_pct must be >= 0")

        if self.max_distance_to_edge_pct < 0:
            raise StrategyConfigError("max_distance_to_edge_pct must be >= 0")

        score_weights = {
            "score_gap_weight": self.score_gap_weight,
            "score_event_weight": self.score_event_weight,
            "score_proximity_weight": self.score_proximity_weight,
            "score_fill_weight": self.score_fill_weight,
            "score_layer_weight": self.score_layer_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }
        confidence_weights = {
            "confidence_gap_weight": self.confidence_gap_weight,
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
            "tag_fvg_reaction",
            "tag_fvg_respected",
            "tag_fvg_retested",
            "tag_fvg_partial_fill",
            "tag_fvg_fill_started",
            "tag_fvg_proximity",
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


class FVGReactionStrategy(PriceActionTradingStrategy):
    """
    Unified FVG reaction strategy.

    Input:
        StrategyContext with FeatureSource.PRICE_ACTION domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.price_action.fvg_reaction"
    category: StrategyCategory = StrategyCategory.PRICE_ACTION
    default_setup_type: SetupType = SetupType.RETEST

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        price_action_config: FVGReactionStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_price_action_config = (
            price_action_config or FVGReactionStrategyConfig()
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

        self.fvg_config: FVGReactionStrategyConfig = resolved_price_action_config

    @property
    def strategy_name(self) -> str:
        return "fvg_reaction"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.PRICE_ACTION,
            timeframe=Timeframe.M1,
            tags=[
                self.fvg_config.tag_price_action,
                self.fvg_config.tag_fvg,
                self.fvg_config.tag_fvg_reaction,
                self.fvg_config.tag_reaction,
                self.fvg_config.tag_retest,
                "analytics_price_action",
            ],
            version="2.0.0",
            description=(
                "Interprets Fair Value Gap lifecycle context from normalized "
                "price-action StrategyContext and returns internal StrategySignal."
            ),
            required_features=set(self.required_features()),
            supported_regimes={
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.BREAKOUT,
                MarketRegime.SQUEEZE,
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "analytics.price_action",
                "strategy_type": "fvg_reaction",
                "base_class": "PriceActionTradingStrategy",
                "canonical_payload": "PriceActionCompositeSnapshot",
                "uses_fvg": True,
                "uses_fvg_lifecycle_events": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(
            self.fvg_config.required_price_action_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        if not self.has_any_price_action_data(
            context,
            tuple(self.fvg_config.required_price_action_features),
        ):
            return None

        if self.has_stale_price_action_features(
            context,
            tuple(self.fvg_config.required_price_action_features),
        ):
            return None

        view = self._extract_view(context)
        if view is None or view.reaction_gap is None:
            return None

        if (
            self.fvg_config.require_fresh_fvg
            and is_stale(
                event_time=view.event_time,
                now=context.timestamp,
                stale_after_seconds=self.fvg_config.stale_feature_max_age_seconds,
            )
        ):
            return None

        common_rejection = quality_filter_reason(
            view.primary_layer,
            min_confidence=self.fvg_config.min_layer_confidence,
            min_score=self.fvg_config.min_gap_strength,
            stale_after_seconds=self.fvg_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )
        if common_rejection is not None and self.fvg_config.require_primary_layer_eligible:
            return None

        side = self._infer_side(view)
        if self.fvg_config.require_directional_gap and not is_directional_side(side):
            return None

        setup_type = self._infer_setup_type(view)

        if not self._passes_filters(view=view, side=side, setup_type=setup_type):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            view=view,
            side=side,
            setup_type=setup_type,
        )

        if breakdown.score < self.fvg_config.min_signal_score:
            return None

        if breakdown.confidence < self.fvg_config.min_signal_confidence:
            return None

        source_features = self._source_features(view)
        tags = self._tags(view=view, setup_type=setup_type)

        event_label = (
            normalize_label(view.last_event.event_type)
            if view.last_event is not None
            else "active_gap_proximity"
        )

        reasons = list(
            dict.fromkeys(
                [
                    "fvg_reaction_signal",
                    f"side:{side.value}",
                    f"setup_type:{setup_type.value}",
                    f"event:{event_label}",
                    *view.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "price_action_setup_family": "fvg_reaction",
            "price_action_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "event_time": view.event_time.isoformat() if view.event_time else None,
            "primary_layer_name": normalize_label(view.primary_layer_name),
            "secondary_layer_name": normalize_label(view.secondary_layer_name),
            "reaction_gap": serialize_for_metadata(view.reaction_gap),
            "last_event": serialize_for_metadata(view.last_event),
            "primary_layer": serialize_for_metadata(view.primary_layer),
            "secondary_layer": serialize_for_metadata(view.secondary_layer),
            "layer_confidence": view.layer_confidence,
            "layer_strength": view.layer_strength,
            "layer_alignment_score": view.layer_alignment_score,
            "proximity_score": view.proximity_score,
            "fill_quality_score": view.fill_quality_score,
            "mapped_side": side.value,
            "setup_type": setup_type.value,
            "raw": serialize_for_metadata(view.raw),
        }

        return self.build_price_action_signal(
            context=context,
            side=side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.fvg_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_view(
        self,
        context: StrategyContext,
    ) -> FVGReactionContext | None:
        module = self.resolve_price_action_module(
            context,
            "fair_value_gap",
            aliases=("fvg",),
        )
        if not module:
            return None

        primary = select_primary_layer(
            module,
            prefer_external_layer=self.fvg_config.prefer_external_layer,
        )
        secondary = select_secondary_layer(
            module,
            prefer_external_layer=self.fvg_config.prefer_external_layer,
        )

        if not primary:
            return None

        primary_layer_name = self._extract_layer_name(
            primary,
            fallback=StructureLayer.EXTERNAL
            if self.fvg_config.prefer_external_layer
            else StructureLayer.INTERNAL,
        )
        secondary_layer_name = self._extract_layer_name(
            secondary,
            fallback=StructureLayer.INTERNAL
            if self.fvg_config.prefer_external_layer
            else StructureLayer.EXTERNAL,
        )

        last_event_payload = (
            get_path(primary, "last_event")
            or get_path(module, "last_event")
            or extract_last_event(module)
        )
        last_event = FVGEventContext.from_payload(
            last_event_payload,
            fallback_layer=primary_layer_name,
        )

        reaction_gap = self._select_reaction_gap(
            module=module,
            primary=primary,
            last_event=last_event,
            fallback_layer=primary_layer_name,
        )
        if reaction_gap is None:
            return None

        layer_alignment_score = unit_score(
            get_path(module, "layer_alignment_score")
            or get_path(module, "internal_external_alignment")
            or get_path(module, "alignment_score")
            or get_path(primary, "alignment_score")
        )
        proximity_score = self._proximity_score(reaction_gap)
        fill_quality_score = self._fill_quality_score(reaction_gap)

        event_time = (
            last_event.timestamp if last_event is not None else None
        ) or reaction_gap.updated_at or reaction_gap.created_at or extract_last_update(primary)

        reasons = self._extract_reasons(module, primary, last_event)

        return FVGReactionContext(
            module=module,
            primary_layer=primary,
            secondary_layer=secondary,
            primary_layer_name=primary_layer_name,
            secondary_layer_name=secondary_layer_name,
            reaction_gap=reaction_gap,
            last_event=last_event,
            layer_confidence=layer_confidence(primary),
            layer_strength=layer_strength(primary),
            layer_alignment_score=layer_alignment_score,
            proximity_score=proximity_score,
            fill_quality_score=fill_quality_score,
            event_time=event_time,
            reasons=reasons,
            raw=module,
        )

    def _select_reaction_gap(
        self,
        *,
        module: dict[str, Any],
        primary: dict[str, Any],
        last_event: FVGEventContext | None,
        fallback_layer: StructureLayer | None,
    ) -> FVGContext | None:
        candidates: list[Any] = []

        if last_event is not None:
            event_gap = (
                get_path(last_event.raw, "gap")
                or get_path(last_event.raw, "fvg")
                or get_path(last_event.raw, "fair_value_gap")
            )
            if event_gap is not None:
                candidates.append(event_gap)

        candidates.extend(
            [
                get_path(primary, "reaction_gap"),
                get_path(primary, "active_gap"),
                get_path(primary, "nearest_gap"),
                get_path(primary, "nearest_bullish_gap"),
                get_path(primary, "nearest_bearish_gap"),
                get_path(module, "reaction_gap"),
                get_path(module, "active_gap"),
                get_path(module, "nearest_gap"),
                get_path(module, "nearest_bullish_gap"),
                get_path(module, "nearest_bearish_gap"),
            ]
        )

        gaps = get_path(primary, "active_gaps") or get_path(module, "active_gaps")
        if isinstance(gaps, (list, tuple)):
            candidates.extend(gaps)

        normalized: list[FVGContext] = []
        for candidate in candidates:
            gap = FVGContext.from_payload(candidate, fallback_layer=fallback_layer)
            if gap is not None:
                normalized.append(gap)

        if not normalized:
            return None

        return max(
            normalized,
            key=lambda gap: (
                gap.score,
                gap.confidence,
                gap.strength,
                self._proximity_score(gap),
            ),
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
        event: FVGEventContext | None,
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

    def _infer_side(self, view: FVGReactionContext) -> SignalSide:
        gap = view.reaction_gap
        if gap is None:
            return SignalSide.UNKNOWN

        if view.last_event is not None and view.last_event.direction is not None:
            event_side = fvg_direction_to_side(view.last_event.direction)
            if is_directional_side(event_side):
                return event_side

        return fvg_direction_to_side(gap.direction)

    def _infer_setup_type(
        self,
        view: FVGReactionContext,
    ) -> SetupType:
        event = view.last_event
        gap = view.reaction_gap

        if event is not None and event.event_type is not None:
            event_label = normalize_label(event.event_type)

            if event_label in {"created", "gap_created", "new_gap"}:
                return SetupType.CONTINUATION

            if event_label in {"retested", "retest"}:
                return SetupType.RETEST

            if event_label in {"respected", "rejected", "reaction"}:
                return SetupType.RETEST

            if event_label in {"fill_started", "partial_fill", "mitigation_started"}:
                return SetupType.RETEST

        if gap is not None:
            status_label = normalize_label(gap.status)

            if status_label in {"active", "open"}:
                return SetupType.RETEST

            if status_label in {"respected", "retested", "partial_fill", "partially_filled"}:
                return SetupType.RETEST

        return self.fvg_config.default_setup_type

    def _passes_filters(
        self,
        *,
        view: FVGReactionContext,
        side: SignalSide,
        setup_type: SetupType,
    ) -> bool:
        gap = view.reaction_gap
        if gap is None:
            return False

        if self.fvg_config.require_directional_gap and not is_directional_side(side):
            return False

        if not gap.is_valid:
            return False

        status_label = normalize_label(gap.status)

        if self.fvg_config.block_invalidated_gaps:
            if status_label in {"invalidated", "expired", "broken"}:
                return False

        if self.fvg_config.block_filled_gaps:
            if status_label in {"filled", "fully_filled", "mitigated"} or gap.is_mitigated:
                return False

        if gap.strength < self.fvg_config.min_gap_strength:
            return False

        if view.layer_confidence < self.fvg_config.min_layer_confidence:
            return False

        if gap.fill_pct > self.fvg_config.max_gap_fill_for_entry:
            return False

        if gap.distance_to_mid_pct > self.fvg_config.max_distance_to_mid_pct:
            if gap.distance_to_nearest_edge_pct > self.fvg_config.max_distance_to_edge_pct:
                return False

        event = view.last_event
        if self.fvg_config.require_recent_event and event is None:
            if not self.fvg_config.allow_active_gap_proximity_entry:
                return False

        if event is not None:
            if event.confidence < self.fvg_config.min_event_confidence:
                return False

            event_label = normalize_label(event.event_type)

            if event_label in {"created", "gap_created", "new_gap"}:
                if not self.fvg_config.allow_created_gap_continuation:
                    return False

            if event_label in {"fill_started", "mitigation_started"}:
                if not self.fvg_config.allow_fill_started_reaction:
                    return False

            if event_label in {"partial_fill", "partially_filled"}:
                if not self.fvg_config.allow_partial_fill_reaction:
                    return False

            if event_label in {"respected", "rejected", "reaction"}:
                if not self.fvg_config.allow_respected_reaction:
                    return False

            if event_label in {"retested", "retest"}:
                if not self.fvg_config.allow_retested_reaction:
                    return False

        if self.fvg_config.require_respected_or_retested:
            event_label = normalize_label(event.event_type) if event else ""
            if event_label not in {"respected", "retested", "retest", "rejected", "reaction"}:
                return False

        if gap.fill_pct < self.fvg_config.min_gap_fill_for_reaction:
            if not self.fvg_config.allow_active_gap_proximity_entry:
                return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        view: FVGReactionContext,
        side: SignalSide,
        setup_type: SetupType,
    ) -> ScoreBreakdown:
        gap = view.reaction_gap
        event = view.last_event

        if gap is None:
            return ScoreBreakdown()

        gap_component = average_score(gap.score, gap.confidence, gap.strength)
        event_component = self._event_component(event)
        proximity_component = view.proximity_score
        fill_component = view.fill_quality_score
        layer_component = average_score(view.layer_confidence, view.layer_strength)
        fresh_component = freshness_score(
            event_time=view.event_time,
            now=context.timestamp,
            stale_after_seconds=self.fvg_config.stale_feature_max_age_seconds,
        )

        components = {
            "gap": gap_component,
            "event": event_component,
            "proximity": proximity_component,
            "fill": fill_component,
            "layer": layer_component,
            "freshness": fresh_component,
        }
        weights = {
            "gap": self.fvg_config.score_gap_weight,
            "event": self.fvg_config.score_event_weight,
            "proximity": self.fvg_config.score_proximity_weight,
            "fill": self.fvg_config.score_fill_weight,
            "layer": self.fvg_config.score_layer_weight,
            "freshness": self.fvg_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=gap_component)
        confidence = confidence_from_components(
            primary=max(gap_component, event_component),
            context=weighted_score(
                {
                    "layer": layer_component,
                    "alignment": view.layer_alignment_score,
                    "proximity": proximity_component,
                },
                {
                    "layer": 0.35,
                    "alignment": 0.30,
                    "proximity": 0.35,
                },
            ),
            confirmation=fill_component,
            freshness=fresh_component,
            primary_weight=self.fvg_config.confidence_gap_weight,
            context_weight=self.fvg_config.confidence_context_weight,
            confirmation_weight=self.fvg_config.confidence_confirmation_weight,
            freshness_weight=self.fvg_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            f"side:{side.value}",
            f"setup_type:{setup_type.value}",
            f"fvg_direction:{normalize_label(gap.direction)}",
        ]

        if event is not None and event.event_type is not None:
            event_label = normalize_label(event.event_type)
            confirmations.append(f"fvg_event:{event_label}")

            if event_label in {"respected", "rejected", "reaction"}:
                score += self.fvg_config.respected_bonus
                confirmations.append("fvg_respected")

            elif event_label in {"retested", "retest"}:
                score += self.fvg_config.retested_bonus
                confirmations.append("fvg_retested")

            elif event_label in {"partial_fill", "partially_filled"}:
                score += self.fvg_config.partial_fill_bonus
                confirmations.append("fvg_partial_fill")

            elif event_label in {"fill_started", "mitigation_started"}:
                score += self.fvg_config.partial_fill_bonus
                confirmations.append("fvg_fill_started")

        else:
            reasons.append("no_recent_fvg_event")
            confirmations.append("active_gap_proximity_entry")

        if proximity_component >= 0.70:
            score += self.fvg_config.proximity_bonus
            confirmations.append("price_near_fvg_reaction_zone")

        if view.layer_alignment_score > 0:
            score += self.fvg_config.layer_alignment_bonus
            confirmations.append("fvg_layer_alignment_context")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    def _event_component(self, event: FVGEventContext | None) -> float:
        if event is None:
            return 0.0

        return weighted_score(
            {
                "confidence": event.confidence,
                "score": event.score,
                "fill": event.fill_pct,
                "distance": distance_score(
                    event.distance_to_mid_pct,
                    max_distance_pct=max(self.fvg_config.max_distance_to_mid_pct, 0.0001),
                ),
            },
            {
                "confidence": 0.40,
                "score": 0.35,
                "fill": 0.15,
                "distance": 0.10,
            },
        )

    def _proximity_score(self, gap: FVGContext) -> float:
        mid_score = distance_score(
            gap.distance_to_mid_pct,
            max_distance_pct=max(self.fvg_config.max_distance_to_mid_pct, 0.0001),
        )
        edge_score = distance_score(
            gap.distance_to_nearest_edge_pct,
            max_distance_pct=max(self.fvg_config.max_distance_to_edge_pct, 0.0001),
        )
        return max(mid_score, edge_score)

    def _fill_quality_score(self, gap: FVGContext) -> float:
        if gap.fill_pct <= 0:
            return 0.0

        if gap.fill_pct < self.fvg_config.min_gap_fill_for_reaction:
            return unit_score(gap.fill_pct / max(self.fvg_config.min_gap_fill_for_reaction, 0.0001))

        if gap.fill_pct <= self.fvg_config.max_gap_fill_for_entry:
            return unit_score(gap.fill_pct)

        return unit_score(1.0 - gap.fill_pct)

    # ------------------------------------------------------------------
    # Source features / tags
    # ------------------------------------------------------------------

    def _source_features(self, view: FVGReactionContext) -> list[str]:
        features = [
            *fvg_source_features(),
            PRICE_ACTION_FEATURES.FAIR_VALUE_GAP,
            PRICE_ACTION_FEATURES.FVG,
            PRICE_ACTION_FEATURES.FVG_INTERNAL,
            PRICE_ACTION_FEATURES.FVG_EXTERNAL,
            PRICE_ACTION_FEATURES.FVG_LAST_EVENT,
            PRICE_ACTION_FEATURES.FVG_NEAREST_BULLISH_GAP,
            PRICE_ACTION_FEATURES.FVG_NEAREST_BEARISH_GAP,
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        *,
        view: FVGReactionContext,
        setup_type: SetupType,
    ) -> list[str]:
        tags = [
            self.fvg_config.tag_price_action,
            self.fvg_config.tag_fvg,
            self.fvg_config.tag_fvg_reaction,
            self.fvg_config.tag_reaction,
            setup_type.value,
        ]

        event = view.last_event
        if event is not None and event.event_type is not None:
            event_label = normalize_label(event.event_type)

            if event_label in {"respected", "rejected", "reaction"}:
                tags.append(self.fvg_config.tag_fvg_respected)

            if event_label in {"retested", "retest"}:
                tags.append(self.fvg_config.tag_fvg_retested)

            if event_label in {"partial_fill", "partially_filled"}:
                tags.append(self.fvg_config.tag_fvg_partial_fill)

            if event_label in {"fill_started", "mitigation_started"}:
                tags.append(self.fvg_config.tag_fvg_fill_started)

        else:
            tags.append(self.fvg_config.tag_fvg_proximity)

        gap = view.reaction_gap
        if gap is not None:
            if gap.layer is not None:
                tags.append(f"layer:{normalize_label(gap.layer)}")

            if gap.direction is not None:
                tags.append(f"direction:{normalize_label(gap.direction)}")

            if gap.status is not None:
                tags.append(f"status:{normalize_label(gap.status)}")

        if setup_type is SetupType.RETEST:
            tags.append(self.fvg_config.tag_retest)

        if setup_type is SetupType.CONTINUATION:
            tags.append(self.fvg_config.tag_continuation)

        if setup_type is SetupType.REVERSAL:
            tags.append(self.fvg_config.tag_reversal)

        return list(dict.fromkeys(tags))