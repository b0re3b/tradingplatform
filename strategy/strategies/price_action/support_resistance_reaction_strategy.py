# trading_system/strategy/strategies/price_action/support_resistance_reaction_strategy.py

from __future__ import annotations
import logging

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from analytics.price_action.enums import (
    LevelStatus,
    LevelType,
    SREventType,
    StructureLayer,
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
    distance_score,
    extract_last_event,
    extract_last_update,
    get_path,
    is_directional_side,
    is_stale,
    layer_confidence,
    layer_strength,
    level_reaction_to_side,
    normalize_label,
    parse_datetime,
    parse_level_status,
    parse_level_type,
    parse_sr_event_type,
    parse_structure_layer,
    quality_filter_reason,
    select_primary_layer,
    select_secondary_layer,
    serialize_for_metadata,
    support_resistance_source_features,
    to_bool,
    to_float,
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
class SupportResistanceLevelContext:
    """
    Normalized support/resistance level context.

    This DTO belongs to strategy layer only. Analytics models remain in
    analytics.price_action.
    """
    _logger = logging.getLogger(__name__ + ".SupportResistanceLevelContext")

    level_type: LevelType | None = None
    status: LevelStatus | None = None
    layer: StructureLayer | None = None

    price: float | None = None
    current_price: float | None = None
    distance_pct: float = 0.0

    strength: float = 0.0
    confidence: float = 0.0
    score: float = 0.0
    touch_count: int = 0
    reaction_count: int = 0
    break_count: int = 0

    is_active: bool = True
    is_broken: bool = False
    is_flipped: bool = False
    is_retested: bool = False

    created_at: datetime | None = None
    updated_at: datetime | None = None

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        fallback_layer: StructureLayer | None = None,
    ) -> SupportResistanceLevelContext | None:
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".SupportResistanceLevelContext")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceLevelContext.from_payload")
        if payload is None:
            return None

        level_type = parse_level_type(
            get_path(payload, "level_type")
            or get_path(payload, "type")
            or get_path(payload, "kind")
        )
        status = parse_level_status(
            get_path(payload, "status")
            or get_path(payload, "level_status")
            or get_path(payload, "state")
        )
        layer = parse_structure_layer(
            get_path(payload, "layer")
            or get_path(payload, "structure_layer"),
            default=fallback_layer,
        )

        price = to_float(
            get_path(payload, "price")
            or get_path(payload, "level")
            or get_path(payload, "value")
        )

        if level_type is None and price is None:
            return None

        confidence = unit_score(
            get_path(payload, "confidence")
            or get_path(payload, "level_confidence")
        )
        strength = unit_score(
            get_path(payload, "strength")
            or get_path(payload, "level_strength")
            or get_path(payload, "quality")
            or confidence
        )
        score = unit_score(
            get_path(payload, "score")
            or get_path(payload, "level_score")
            or strength
            or confidence
        )

        return cls(
            level_type=level_type,
            status=status,
            layer=layer,
            price=price,
            current_price=to_float(
                get_path(payload, "current_price")
                or get_path(payload, "last_price")
                or get_path(payload, "close")
            ),
            distance_pct=abs(
                to_float(
                    get_path(payload, "distance_pct")
                    or get_path(payload, "distance_to_price_pct")
                    or get_path(payload, "distance_to_level_pct"),
                    0.0,
                )
                or 0.0
            ),
            strength=strength,
            confidence=confidence,
            score=score,
            touch_count=to_int_safe(
                get_path(payload, "touch_count")
                or get_path(payload, "touches")
            ),
            reaction_count=to_int_safe(
                get_path(payload, "reaction_count")
                or get_path(payload, "reactions")
            ),
            break_count=to_int_safe(
                get_path(payload, "break_count")
                or get_path(payload, "breaks")
            ),
            is_active=to_bool(
                get_path(payload, "is_active")
                or get_path(payload, "active"),
                default=True,
            ),
            is_broken=to_bool(
                get_path(payload, "is_broken")
                or get_path(payload, "broken"),
                default=False,
            ),
            is_flipped=to_bool(
                get_path(payload, "is_flipped")
                or get_path(payload, "flipped"),
                default=False,
            ),
            is_retested=to_bool(
                get_path(payload, "is_retested")
                or get_path(payload, "retested"),
                default=False,
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
            raw=serialize_for_metadata(payload)
            if isinstance(payload, dict)
            else {"raw": serialize_for_metadata(payload)},
        )


@dataclass(slots=True)
class SupportResistanceEventContext:
    """
    Normalized support/resistance lifecycle event.
    """
    _logger = logging.getLogger(__name__ + ".SupportResistanceEventContext")

    event_type: SREventType | None = None
    level_type: LevelType | None = None
    status: LevelStatus | None = None
    layer: StructureLayer | None = None

    confidence: float = 0.0
    score: float = 0.0
    distance_pct: float = 0.0
    price: float | None = None
    level_price: float | None = None

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
    ) -> SupportResistanceEventContext | None:
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".SupportResistanceEventContext")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceEventContext.from_payload")
        if payload is None:
            return None

        event_type = parse_sr_event_type(
            get_path(payload, "event_type")
            or get_path(payload, "type")
            or get_path(payload, "kind")
        )
        level_type = parse_level_type(
            get_path(payload, "level_type")
            or get_path(payload, "type")
            or get_path(payload, "level.kind")
        )
        status = parse_level_status(
            get_path(payload, "status")
            or get_path(payload, "level_status")
            or get_path(payload, "state")
        )
        layer = parse_structure_layer(
            get_path(payload, "layer")
            or get_path(payload, "structure_layer"),
            default=fallback_layer,
        )

        if event_type is None and level_type is None and status is None:
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
            level_type=level_type,
            status=status,
            layer=layer,
            confidence=confidence,
            score=score,
            distance_pct=abs(
                to_float(
                    get_path(payload, "distance_pct")
                    or get_path(payload, "distance_to_level_pct"),
                    0.0,
                )
                or 0.0
            ),
            price=to_float(
                get_path(payload, "price")
                or get_path(payload, "current_price")
                or get_path(payload, "last_price")
            ),
            level_price=to_float(
                get_path(payload, "level_price")
                or get_path(payload, "level")
                or get_path(payload, "level.price")
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
class SupportResistanceReactionContext:
    """
    Normalized SR reaction view consumed by SupportResistanceReactionStrategy.
    """
    _logger = logging.getLogger(__name__ + ".SupportResistanceReactionContext")

    module: dict[str, Any]
    primary_layer: dict[str, Any]
    secondary_layer: dict[str, Any]

    primary_layer_name: StructureLayer | None = None
    secondary_layer_name: StructureLayer | None = None

    reaction_level: SupportResistanceLevelContext | None = None
    last_event: SupportResistanceEventContext | None = None

    layer_confidence: float = 0.0
    layer_strength: float = 0.0
    layer_alignment_score: float = 0.0
    proximity_score: float = 0.0
    level_quality_score: float = 0.0

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SupportResistanceReactionStrategyConfig(PriceActionStrategyConfig):
    """
    Unified support/resistance reaction strategy config.

    Strategy idea:
    - read normalized support_resistance context from StrategyContext;
    - select support/resistance reaction level;
    - interpret rejection / break / flip / retest;
    - return internal StrategySignal only;
    - leave routing, filtering, confluence, portfolio coordination and
      risk-ready conversion to SignalProcessor.
    """
    _logger = logging.getLogger(__name__ + ".SupportResistanceReactionStrategyConfig")

    prefer_external_layer: bool = True

    require_fresh_sr: bool = True
    require_recent_event: bool = True
    require_level_strength: bool = True
    require_primary_layer_eligible: bool = True

    allow_support_rejection_long: bool = True
    allow_resistance_rejection_short: bool = True
    allow_support_break_short: bool = True
    allow_resistance_break_long: bool = True
    allow_flip_support_long: bool = True
    allow_flip_resistance_short: bool = True
    allow_retest_entries: bool = True
    allow_nearest_level_fallback: bool = True

    block_inactive_levels: bool = True
    block_broken_non_flip_levels: bool = True

    min_layer_confidence: float = 0.40
    min_level_strength: float = 0.45
    min_event_confidence: float = 0.45
    min_touch_count: int = 1
    max_distance_to_level_pct: float = 0.0035

    rejection_bonus: float = 0.05
    break_bonus: float = 0.05
    flip_bonus: float = 0.06
    retest_bonus: float = 0.06
    proximity_bonus: float = 0.04
    touch_count_bonus: float = 0.03
    layer_alignment_bonus: float = 0.03

    score_level_weight: float = 0.30
    score_event_weight: float = 0.24
    score_proximity_weight: float = 0.16
    score_reaction_weight: float = 0.14
    score_layer_weight: float = 0.10
    score_freshness_weight: float = 0.06

    confidence_level_weight: float = 0.52
    confidence_context_weight: float = 0.28
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    tag_sr_reaction: str = "support_resistance_reaction"
    tag_support_rejection: str = "support_rejection"
    tag_resistance_rejection: str = "resistance_rejection"
    tag_support_break: str = "support_break"
    tag_resistance_break: str = "resistance_break"
    tag_flip: str = "flip"
    tag_level_retest: str = "level_retest"
    tag_level_proximity: str = "level_proximity"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.RETEST

    required_price_action_features: tuple[str, ...] = (
        PRICE_ACTION_FEATURES.SUPPORT_RESISTANCE,
    )

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategyConfig.validate")
        PriceActionStrategyConfig.validate(self)

        unit_fields = {
            "min_layer_confidence": self.min_layer_confidence,
            "min_level_strength": self.min_level_strength,
            "min_event_confidence": self.min_event_confidence,
            "rejection_bonus": self.rejection_bonus,
            "break_bonus": self.break_bonus,
            "flip_bonus": self.flip_bonus,
            "retest_bonus": self.retest_bonus,
            "proximity_bonus": self.proximity_bonus,
            "touch_count_bonus": self.touch_count_bonus,
            "layer_alignment_bonus": self.layer_alignment_bonus,
        }

        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.min_touch_count < 0:
            raise StrategyConfigError("min_touch_count must be >= 0")

        if self.max_distance_to_level_pct < 0:
            raise StrategyConfigError("max_distance_to_level_pct must be >= 0")

        score_weights = {
            "score_level_weight": self.score_level_weight,
            "score_event_weight": self.score_event_weight,
            "score_proximity_weight": self.score_proximity_weight,
            "score_reaction_weight": self.score_reaction_weight,
            "score_layer_weight": self.score_layer_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }
        confidence_weights = {
            "confidence_level_weight": self.confidence_level_weight,
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
            "tag_sr_reaction",
            "tag_support_rejection",
            "tag_resistance_rejection",
            "tag_support_break",
            "tag_resistance_break",
            "tag_flip",
            "tag_level_retest",
            "tag_level_proximity",
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


class SupportResistanceReactionStrategy(PriceActionTradingStrategy):
    """
    Unified support/resistance reaction strategy.

    Input:
        StrategyContext with FeatureSource.PRICE_ACTION domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """
    _logger = logging.getLogger(__name__ + ".SupportResistanceReactionStrategy")

    component_namespace = "strategy.price_action.support_resistance_reaction"
    category: StrategyCategory = StrategyCategory.PRICE_ACTION
    default_setup_type: SetupType = SetupType.RETEST

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        price_action_config: SupportResistanceReactionStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy.__init__")
        resolved_price_action_config = (
            price_action_config or SupportResistanceReactionStrategyConfig()
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

        self.sr_config: SupportResistanceReactionStrategyConfig = (
            resolved_price_action_config
        )

    @property
    def strategy_name(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy.strategy_name")
        return "support_resistance_reaction"

    @property
    def metadata(self) -> StrategyMetadata:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy.metadata")
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.PRICE_ACTION,
            timeframe=Timeframe.M1,
            tags=[
                self.sr_config.tag_price_action,
                self.sr_config.tag_support_resistance,
                self.sr_config.tag_sr_reaction,
                self.sr_config.tag_retest,
                self.sr_config.tag_reaction,
                "analytics_price_action",
            ],
            version="2.0.0",
            description=(
                "Interprets support/resistance rejection, break, flip and retest "
                "context from normalized price-action StrategyContext and returns "
                "internal StrategySignal."
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
                "strategy_type": "support_resistance_reaction",
                "base_class": "PriceActionTradingStrategy",
                "canonical_payload": "PriceActionCompositeSnapshot",
                "uses_support_resistance": True,
                "uses_level_events": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy.required_features")
        base_required = super().required_features()
        return set(base_required).union(
            self.sr_config.required_price_action_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy.generate_signal")
        self.validate_context_requirements(context)

        if not self.has_any_price_action_data(
            context,
            tuple(self.sr_config.required_price_action_features),
        ):
            self.remember_no_signal(
                "missing_price_action_support_resistance_context",
                price_action_domain_keys=sorted(self.price_action_domain(context).keys()),
                required_features=sorted(self.required_features()),
            )
            return None

        if self.has_stale_price_action_features(
            context,
            tuple(self.sr_config.required_price_action_features),
        ):
            self.remember_no_signal(
                "stale_price_action_support_resistance_features",
                required_features=sorted(self.sr_config.required_price_action_features),
            )
            return None

        view = self._extract_view(context)
        if view is None:
            self.remember_no_signal(
                "support_resistance_view_not_resolved",
                support_resistance=serialize_for_metadata(
                    self.price_action_item(context, "support_resistance")
                ),
                price_action_domain_keys=sorted(self.price_action_domain(context).keys()),
            )
            return None

        if view.reaction_level is None:
            self.remember_no_signal(
                "missing_support_resistance_reaction_level",
                support_resistance=serialize_for_metadata(
                    self.price_action_item(context, "support_resistance")
                ),
                last_event=serialize_for_metadata(view.last_event),
            )
            return None

        if (
            self.sr_config.require_fresh_sr
            and is_stale(
                event_time=view.event_time,
                now=context.timestamp,
                stale_after_seconds=self.sr_config.stale_feature_max_age_seconds,
            )
        ):
            self.remember_no_signal(
                "stale_support_resistance_event",
                event_time=view.event_time.isoformat() if view.event_time else None,
                context_timestamp=context.timestamp.isoformat(),
                stale_after_seconds=self.sr_config.stale_feature_max_age_seconds,
            )
            return None

        common_rejection = quality_filter_reason(
            view.primary_layer,
            min_confidence=self.sr_config.min_layer_confidence,
            min_score=self.sr_config.min_level_strength,
            stale_after_seconds=self.sr_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )
        if common_rejection is not None and self.sr_config.require_primary_layer_eligible:
            self.remember_no_signal(
                "support_resistance_primary_layer_rejected",
                rejection=common_rejection,
                primary_layer=serialize_for_metadata(view.primary_layer),
                min_layer_confidence=self.sr_config.min_layer_confidence,
                min_level_strength=self.sr_config.min_level_strength,
            )
            return None

        side = self._infer_side(view)
        if not is_directional_side(side):
            self.remember_no_signal(
                "support_resistance_side_not_directional",
                reaction_level=serialize_for_metadata(view.reaction_level),
                last_event=serialize_for_metadata(view.last_event),
            )
            return None

        setup_type = self._infer_setup_type(view)

        if not self._passes_filters(view=view, side=side, setup_type=setup_type):
            self.remember_no_signal(
                "support_resistance_filters_failed",
                side=side.value,
                setup_type=setup_type.value,
                reaction_level=serialize_for_metadata(view.reaction_level),
                last_event=serialize_for_metadata(view.last_event),
                proximity_score=view.proximity_score,
                level_quality_score=view.level_quality_score,
                layer_confidence=view.layer_confidence,
                layer_strength=view.layer_strength,
            )
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            view=view,
            side=side,
            setup_type=setup_type,
        )

        if breakdown.score < self.sr_config.min_signal_score:
            self.remember_no_signal(
                "support_resistance_score_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_signal_score=self.sr_config.min_signal_score,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        if breakdown.confidence < self.sr_config.min_signal_confidence:
            self.remember_no_signal(
                "support_resistance_confidence_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_signal_confidence=self.sr_config.min_signal_confidence,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        source_features = self._source_features(view)
        tags = self._tags(view=view, setup_type=setup_type)

        event_label = (
            normalize_label(view.last_event.event_type)
            if view.last_event is not None
            else "nearest_level_proximity"
        )

        reasons = list(
            dict.fromkeys(
                [
                    "support_resistance_reaction_signal",
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
            "price_action_setup_family": "support_resistance_reaction",
            "price_action_strategy_version": "2.0.0",
            "contract": "price_action",
            "contract_version": "strategy-domain-v1",
            "primary_section": "support_resistance",
            "strategy_contract_role": "decision_module",
            "risk_ready_payload_owner": "SignalProcessor",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "event_time": view.event_time.isoformat() if view.event_time else None,
            "primary_layer_name": normalize_label(view.primary_layer_name),
            "secondary_layer_name": normalize_label(view.secondary_layer_name),
            "reaction_level": serialize_for_metadata(view.reaction_level),
            "last_event": serialize_for_metadata(view.last_event),
            "primary_layer": serialize_for_metadata(view.primary_layer),
            "secondary_layer": serialize_for_metadata(view.secondary_layer),
            "layer_confidence": view.layer_confidence,
            "layer_strength": view.layer_strength,
            "layer_alignment_score": view.layer_alignment_score,
            "proximity_score": view.proximity_score,
            "level_quality_score": view.level_quality_score,
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
            priority=self.sr_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_view(
        self,
        context: StrategyContext,
    ) -> SupportResistanceReactionContext | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._extract_view")
        module = self.resolve_price_action_module(
            context,
            "support_resistance",
            aliases=("sr",),
        )
        if not module:
            return None

        primary = select_primary_layer(
            module,
            prefer_external_layer=self.sr_config.prefer_external_layer,
        )
        secondary = select_secondary_layer(
            module,
            prefer_external_layer=self.sr_config.prefer_external_layer,
        )

        if not primary:
            return None

        primary_layer_name = self._extract_layer_name(
            primary,
            fallback=StructureLayer.EXTERNAL
            if self.sr_config.prefer_external_layer
            else StructureLayer.INTERNAL,
        )
        secondary_layer_name = self._extract_layer_name(
            secondary,
            fallback=StructureLayer.INTERNAL
            if self.sr_config.prefer_external_layer
            else StructureLayer.EXTERNAL,
        )

        last_event_payload = (
            get_path(primary, "last_event")
            or get_path(module, "last_event")
            or extract_last_event(module)
        )
        last_event = SupportResistanceEventContext.from_payload(
            last_event_payload,
            fallback_layer=primary_layer_name,
        )

        reaction_level = self._select_reaction_level(
            module=module,
            primary=primary,
            last_event=last_event,
            fallback_layer=primary_layer_name,
        )
        if reaction_level is None:
            return None

        layer_alignment_score = unit_score(
            get_path(module, "layer_alignment_score")
            or get_path(module, "internal_external_alignment")
            or get_path(module, "alignment_score")
            or get_path(primary, "alignment_score")
        )
        proximity_score = self._proximity_score(reaction_level)
        level_quality_score = self._level_quality_score(reaction_level)

        event_time = (
            last_event.timestamp if last_event is not None else None
        ) or reaction_level.updated_at or reaction_level.created_at or extract_last_update(primary)

        reasons = self._extract_reasons(module, primary, last_event)

        return SupportResistanceReactionContext(
            module=module,
            primary_layer=primary,
            secondary_layer=secondary,
            primary_layer_name=primary_layer_name,
            secondary_layer_name=secondary_layer_name,
            reaction_level=reaction_level,
            last_event=last_event,
            layer_confidence=layer_confidence(primary),
            layer_strength=layer_strength(primary),
            layer_alignment_score=layer_alignment_score,
            proximity_score=proximity_score,
            level_quality_score=level_quality_score,
            event_time=event_time,
            reasons=reasons,
            raw=module,
        )

    def _select_reaction_level(
        self,
        *,
        module: dict[str, Any],
        primary: dict[str, Any],
        last_event: SupportResistanceEventContext | None,
        fallback_layer: StructureLayer | None,
    ) -> SupportResistanceLevelContext | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._select_reaction_level")
        candidates: list[Any] = []

        if last_event is not None:
            event_level = (
                get_path(last_event.raw, "level")
                or get_path(last_event.raw, "support_resistance_level")
                or get_path(last_event.raw, "sr_level")
            )
            if event_level is not None:
                candidates.append(event_level)

        candidates.extend(
            [
                get_path(primary, "reaction_level"),
                get_path(primary, "nearest_level"),
                get_path(primary, "nearest_support"),
                get_path(primary, "nearest_resistance"),
                get_path(primary, "active_level"),
                get_path(module, "reaction_level"),
                get_path(module, "nearest_level"),
                get_path(module, "nearest_support"),
                get_path(module, "nearest_resistance"),
                get_path(module, "active_level"),
            ]
        )

        levels = get_path(primary, "levels") or get_path(module, "levels")
        if isinstance(levels, (list, tuple)):
            candidates.extend(levels)

        normalized: list[SupportResistanceLevelContext] = []
        for candidate in candidates:
            level = SupportResistanceLevelContext.from_payload(
                candidate,
                fallback_layer=fallback_layer,
            )
            if level is not None:
                normalized.append(level)

        if not normalized:
            return None

        return max(
            normalized,
            key=lambda level: (
                level.score,
                level.confidence,
                level.strength,
                self._proximity_score(level),
                level.touch_count,
            ),
        )

    @staticmethod
    def _extract_layer_name(
        layer: dict[str, Any],
        *,
        fallback: StructureLayer,
    ) -> StructureLayer:
        _strategy_logger = logging.getLogger(__name__ + ".SupportResistanceReactionStrategy._extract_layer_name")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._extract_layer_name")
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
        event: SupportResistanceEventContext | None,
    ) -> list[str]:
        _strategy_logger = logging.getLogger(__name__ + ".SupportResistanceReactionStrategy._extract_reasons")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._extract_reasons")
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

    def _infer_side(self, view: SupportResistanceReactionContext) -> SignalSide:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._infer_side")
        if view.reaction_level is None:
            return SignalSide.UNKNOWN

        if view.last_event is not None:
            event_side = level_reaction_to_side(
                view.reaction_level.raw,
                view.last_event.raw,
            )
            if is_directional_side(event_side):
                return event_side

        return level_reaction_to_side(view.reaction_level.raw)

    def _infer_setup_type(
        self,
        view: SupportResistanceReactionContext,
    ) -> SetupType:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._infer_setup_type")
        event = view.last_event

        if event is not None and event.event_type is not None:
            event_label = normalize_label(event.event_type)

            if event_label in {
                "support_break",
                "support_breakdown",
                "resistance_break",
                "resistance_breakout",
                "break",
                "breakout",
                "breakdown",
            }:
                return SetupType.BREAKOUT

            if event_label in {
                "flip_support",
                "flip_resistance",
                "support_retest",
                "resistance_retest",
                "retest",
            }:
                return SetupType.RETEST

            if event_label in {
                "support_rejection",
                "support_hold",
                "resistance_rejection",
                "resistance_hold",
                "rejection",
                "hold",
            }:
                return SetupType.REVERSAL

        level = view.reaction_level
        if level is not None:
            if level.is_retested or level.is_flipped:
                return SetupType.RETEST

            if level.is_broken:
                return SetupType.BREAKOUT

        return self.sr_config.default_setup_type

    def _passes_filters(
        self,
        *,
        view: SupportResistanceReactionContext,
        side: SignalSide,
        setup_type: SetupType,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._passes_filters")
        level = view.reaction_level
        if level is None:
            return False

        if self.sr_config.block_inactive_levels and not level.is_active:
            return False

        if self.sr_config.block_broken_non_flip_levels:
            if level.is_broken and not level.is_flipped and setup_type is not SetupType.BREAKOUT:
                return False

        if self.sr_config.require_level_strength:
            if level.strength < self.sr_config.min_level_strength:
                return False

        if level.touch_count < self.sr_config.min_touch_count:
            return False

        if view.layer_confidence < self.sr_config.min_layer_confidence:
            return False

        if level.distance_pct > self.sr_config.max_distance_to_level_pct:
            return False

        event = view.last_event
        if self.sr_config.require_recent_event and event is None:
            if not self.sr_config.allow_nearest_level_fallback:
                return False

        if event is not None:
            if event.confidence < self.sr_config.min_event_confidence:
                return False

            event_label = normalize_label(event.event_type)

            if event_label in {"support_rejection", "support_hold"}:
                if side is SignalSide.LONG and not self.sr_config.allow_support_rejection_long:
                    return False

            if event_label in {"resistance_rejection", "resistance_hold"}:
                if side is SignalSide.SHORT and not self.sr_config.allow_resistance_rejection_short:
                    return False

            if event_label in {"support_break", "support_breakdown"}:
                if side is SignalSide.SHORT and not self.sr_config.allow_support_break_short:
                    return False

            if event_label in {"resistance_break", "resistance_breakout"}:
                if side is SignalSide.LONG and not self.sr_config.allow_resistance_break_long:
                    return False

            if event_label == "flip_support":
                if side is SignalSide.LONG and not self.sr_config.allow_flip_support_long:
                    return False

            if event_label == "flip_resistance":
                if side is SignalSide.SHORT and not self.sr_config.allow_flip_resistance_short:
                    return False

            if event_label in {"support_retest", "resistance_retest", "retest"}:
                if not self.sr_config.allow_retest_entries:
                    return False

        if not is_directional_side(side):
            return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        view: SupportResistanceReactionContext,
        side: SignalSide,
        setup_type: SetupType,
    ) -> ScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._build_score_breakdown")
        level = view.reaction_level
        event = view.last_event

        if level is None:
            return ScoreBreakdown()

        level_component = average_score(level.score, level.confidence, level.strength)
        event_component = self._event_component(event)
        proximity_component = view.proximity_score
        reaction_component = self._reaction_component(view, side, setup_type)
        layer_component = average_score(view.layer_confidence, view.layer_strength)
        fresh_component = freshness_score(
            event_time=view.event_time,
            now=context.timestamp,
            stale_after_seconds=self.sr_config.stale_feature_max_age_seconds,
        )

        components = {
            "level": level_component,
            "event": event_component,
            "proximity": proximity_component,
            "reaction": reaction_component,
            "layer": layer_component,
            "freshness": fresh_component,
        }
        weights = {
            "level": self.sr_config.score_level_weight,
            "event": self.sr_config.score_event_weight,
            "proximity": self.sr_config.score_proximity_weight,
            "reaction": self.sr_config.score_reaction_weight,
            "layer": self.sr_config.score_layer_weight,
            "freshness": self.sr_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=level_component)
        confidence = confidence_from_components(
            primary=max(level_component, event_component),
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
            confirmation=reaction_component,
            freshness=fresh_component,
            primary_weight=self.sr_config.confidence_level_weight,
            context_weight=self.sr_config.confidence_context_weight,
            confirmation_weight=self.sr_config.confidence_confirmation_weight,
            freshness_weight=self.sr_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            f"side:{side.value}",
            f"setup_type:{setup_type.value}",
            f"level_type:{normalize_label(level.level_type)}",
        ]

        if event is not None and event.event_type is not None:
            event_label = normalize_label(event.event_type)
            confirmations.append(f"sr_event:{event_label}")

            if event_label in {"support_rejection", "support_hold", "resistance_rejection", "resistance_hold"}:
                score += self.sr_config.rejection_bonus
                confirmations.append("sr_rejection_context")

            elif event_label in {"support_break", "support_breakdown", "resistance_break", "resistance_breakout"}:
                score += self.sr_config.break_bonus
                confirmations.append("sr_break_context")

            elif event_label in {"flip_support", "flip_resistance"}:
                score += self.sr_config.flip_bonus
                confirmations.append("sr_flip_context")

            elif event_label in {"support_retest", "resistance_retest", "retest"}:
                score += self.sr_config.retest_bonus
                confirmations.append("sr_retest_context")

        else:
            reasons.append("no_recent_sr_event")
            confirmations.append("nearest_level_proximity_entry")

        if proximity_component >= 0.70:
            score += self.sr_config.proximity_bonus
            confirmations.append("price_near_sr_level")

        if level.touch_count >= self.sr_config.min_touch_count:
            score += self.sr_config.touch_count_bonus
            confirmations.append("level_touch_count_confirmed")

        if view.layer_alignment_score > 0:
            score += self.sr_config.layer_alignment_bonus
            confirmations.append("sr_layer_alignment_context")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    def _event_component(
        self,
        event: SupportResistanceEventContext | None,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._event_component")
        if event is None:
            return 0.0

        return weighted_score(
            {
                "confidence": event.confidence,
                "score": event.score,
                "distance": distance_score(
                    event.distance_pct,
                    max_distance_pct=max(
                        self.sr_config.max_distance_to_level_pct,
                        0.0001,
                    ),
                ),
            },
            {
                "confidence": 0.45,
                "score": 0.40,
                "distance": 0.15,
            },
        )

    def _reaction_component(
        self,
        view: SupportResistanceReactionContext,
        side: SignalSide,
        setup_type: SetupType,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._reaction_component")
        level = view.reaction_level
        event = view.last_event

        if level is None:
            return 0.0

        base = average_score(level.score, level.strength, view.proximity_score)

        event_label = normalize_label(event.event_type) if event else ""
        level_label = normalize_label(level.level_type)

        if side is SignalSide.LONG:
            if "support" in level_label:
                base += 0.08
            if event_label in {
                "support_rejection",
                "support_hold",
                "support_retest",
                "flip_support",
                "resistance_break",
                "resistance_breakout",
            }:
                base += 0.10

        elif side is SignalSide.SHORT:
            if "resistance" in level_label:
                base += 0.08
            if event_label in {
                "resistance_rejection",
                "resistance_hold",
                "resistance_retest",
                "flip_resistance",
                "support_break",
                "support_breakdown",
            }:
                base += 0.10

        if setup_type is SetupType.RETEST and level.is_retested:
            base += 0.05

        if level.is_flipped:
            base += 0.05

        return unit_score(base)

    def _proximity_score(
        self,
        level: SupportResistanceLevelContext,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._proximity_score")
        return distance_score(
            level.distance_pct,
            max_distance_pct=max(self.sr_config.max_distance_to_level_pct, 0.0001),
        )

    def _level_quality_score(
        self,
        level: SupportResistanceLevelContext,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._level_quality_score")
        touch_component = unit_score(
            level.touch_count / max(self.sr_config.min_touch_count * 3, 1)
        )
        reaction_component = unit_score(level.reaction_count / 5.0)
        break_penalty = 0.15 if level.is_broken and not level.is_flipped else 0.0

        return unit_score(
            weighted_score(
                {
                    "strength": level.strength,
                    "confidence": level.confidence,
                    "score": level.score,
                    "touches": touch_component,
                    "reactions": reaction_component,
                },
                {
                    "strength": 0.28,
                    "confidence": 0.24,
                    "score": 0.24,
                    "touches": 0.14,
                    "reactions": 0.10,
                },
            )
            - break_penalty
        )

    # ------------------------------------------------------------------
    # Source features / tags
    # ------------------------------------------------------------------

    def _source_features(self, view: SupportResistanceReactionContext) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._source_features")
        features = [
            *support_resistance_source_features(),
            PRICE_ACTION_FEATURES.SUPPORT_RESISTANCE,
            PRICE_ACTION_FEATURES.SUPPORT_RESISTANCE_INTERNAL,
            PRICE_ACTION_FEATURES.SUPPORT_RESISTANCE_EXTERNAL,
            PRICE_ACTION_FEATURES.SUPPORT_RESISTANCE_LAST_EVENT,
            PRICE_ACTION_FEATURES.SUPPORT_RESISTANCE_NEAREST_SUPPORT,
            PRICE_ACTION_FEATURES.SUPPORT_RESISTANCE_NEAREST_RESISTANCE,
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        *,
        view: SupportResistanceReactionContext,
        setup_type: SetupType,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SupportResistanceReactionStrategy._tags")
        tags = [
            self.sr_config.tag_price_action,
            self.sr_config.tag_support_resistance,
            self.sr_config.tag_sr_reaction,
            setup_type.value,
        ]

        event = view.last_event
        if event is not None and event.event_type is not None:
            event_label = normalize_label(event.event_type)

            if event_label in {"support_rejection", "support_hold"}:
                tags.append(self.sr_config.tag_support_rejection)

            if event_label in {"resistance_rejection", "resistance_hold"}:
                tags.append(self.sr_config.tag_resistance_rejection)

            if event_label in {"support_break", "support_breakdown"}:
                tags.append(self.sr_config.tag_support_break)

            if event_label in {"resistance_break", "resistance_breakout"}:
                tags.append(self.sr_config.tag_resistance_break)

            if event_label in {"flip_support", "flip_resistance"}:
                tags.append(self.sr_config.tag_flip)

            if event_label in {"support_retest", "resistance_retest", "retest"}:
                tags.append(self.sr_config.tag_level_retest)

        else:
            tags.append(self.sr_config.tag_level_proximity)

        level = view.reaction_level
        if level is not None:
            if level.layer is not None:
                tags.append(f"layer:{normalize_label(level.layer)}")

            if level.level_type is not None:
                tags.append(f"level:{normalize_label(level.level_type)}")

            if level.status is not None:
                tags.append(f"status:{normalize_label(level.status)}")

        if setup_type is SetupType.RETEST:
            tags.append(self.sr_config.tag_retest)

        if setup_type is SetupType.BREAKOUT:
            tags.append(self.sr_config.tag_breakout)

        if setup_type is SetupType.REVERSAL:
            tags.append(self.sr_config.tag_reversal)

        return list(dict.fromkeys(tags))


def to_int_safe(value: Any, default: int = 0) -> int:
    parsed = to_float(value, default)
    if parsed is None:
        return default
    return int(parsed)