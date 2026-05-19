# trading_system/strategy/strategies/price_action/market_structure_strategy.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from analytics.price_action.enums import (
    MarketBias,
    StructureEventType,
    StructureLayer,
    SwingType,
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
    market_bias_to_side,
    market_structure_source_features,
    normalize_label,
    parse_datetime,
    parse_market_bias,
    parse_structure_event_type,
    parse_structure_layer,
    parse_swing_type,
    quality_filter_reason,
    select_primary_layer,
    select_secondary_layer,
    serialize_for_metadata,
    structure_event_to_side,
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
class SwingContext:
    """
    Normalized swing context for market-structure strategy.

    This is strategy-layer DTO only. Analytics models remain in
    analytics.price_action.
    """

    swing_type: SwingType | None = None
    price: float | None = None
    timestamp: datetime | None = None
    strength: float = 0.0
    distance_pct: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> SwingContext | None:
        if payload is None:
            return None

        swing_type = parse_swing_type(
            get_path(payload, "swing_type")
            or get_path(payload, "type")
            or get_path(payload, "kind")
        )
        price = to_float(
            get_path(payload, "price")
            or get_path(payload, "level")
            or get_path(payload, "value")
        )

        if swing_type is None and price is None:
            return None

        return cls(
            swing_type=swing_type,
            price=price,
            timestamp=parse_datetime(
                get_path(payload, "timestamp")
                or get_path(payload, "time")
                or get_path(payload, "created_at")
            ),
            strength=unit_score(
                get_path(payload, "strength")
                or get_path(payload, "score")
                or get_path(payload, "confidence")
            ),
            distance_pct=abs(
                to_float(
                    get_path(payload, "distance_pct")
                    or get_path(payload, "distance_to_price_pct"),
                    0.0,
                )
                or 0.0
            ),
            metadata=serialize_for_metadata(payload)
            if isinstance(payload, dict)
            else {"raw": serialize_for_metadata(payload)},
        )


@dataclass(slots=True)
class StructureEventContext:
    """
    Normalized BOS/CHOCH/MSS event context.
    """

    event_type: StructureEventType | None = None
    side: SignalSide = SignalSide.UNKNOWN
    layer: StructureLayer | None = None

    confidence: float = 0.0
    score: float = 0.0
    break_distance_pct: float = 0.0
    price: float | None = None
    level: float | None = None
    timestamp: datetime | None = None

    is_confirmed: bool = False
    is_liquidity_sweep: bool = False

    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        fallback_layer: StructureLayer | None = None,
        reverse_on_choch: bool = True,
        reverse_on_mss: bool = True,
    ) -> StructureEventContext | None:
        if payload is None:
            return None

        event_type = parse_structure_event_type(
            get_path(payload, "event_type")
            or get_path(payload, "type")
            or get_path(payload, "kind")
        )
        side = structure_event_to_side(
            payload,
            reverse_on_choch=reverse_on_choch,
            reverse_on_mss=reverse_on_mss,
        )

        layer = parse_structure_layer(
            get_path(payload, "layer")
            or get_path(payload, "structure_layer"),
            default=fallback_layer,
        )

        if event_type is None and not is_directional_side(side):
            return None

        confidence = unit_score(
            get_path(payload, "confidence")
            or get_path(payload, "event_confidence")
            or get_path(payload, "break_confidence")
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
            side=side,
            layer=layer,
            confidence=confidence,
            score=score,
            break_distance_pct=abs(
                to_float(
                    get_path(payload, "break_distance_pct")
                    or get_path(payload, "distance_pct")
                    or get_path(payload, "distance_to_level_pct"),
                    0.0,
                )
                or 0.0
            ),
            price=to_float(
                get_path(payload, "price")
                or get_path(payload, "break_price")
                or get_path(payload, "close")
            ),
            level=to_float(
                get_path(payload, "level")
                or get_path(payload, "broken_level")
                or get_path(payload, "swing_level")
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
            is_liquidity_sweep=to_bool(
                get_path(payload, "is_liquidity_sweep")
                or get_path(payload, "liquidity_sweep")
                or get_path(payload, "sweep"),
                default=False,
            ),
            reasons=list(dict.fromkeys(reasons)),
            raw=serialize_for_metadata(payload)
            if isinstance(payload, dict)
            else {"raw": serialize_for_metadata(payload)},
        )


@dataclass(slots=True)
class MarketStructureContextView:
    """
    Normalized view consumed by MarketStructureStrategy.
    """

    module: dict[str, Any]
    primary_layer: dict[str, Any]
    secondary_layer: dict[str, Any]

    primary_layer_name: StructureLayer | None = None
    secondary_layer_name: StructureLayer | None = None

    bias: MarketBias = MarketBias.UNKNOWN
    secondary_bias: MarketBias = MarketBias.UNKNOWN

    last_event: StructureEventContext | None = None
    last_swing_high: SwingContext | None = None
    last_swing_low: SwingContext | None = None

    mtf_alignment_score: float = 0.0
    swing_progression_score: float = 0.0
    trend_strength: float = 0.0
    layer_confidence: float = 0.0
    layer_strength: float = 0.0

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MarketStructureStrategyConfig(PriceActionStrategyConfig):
    """
    Unified market-structure strategy config.

    Strategy idea:
    - read normalized market_structure context from StrategyContext;
    - react to BOS / CHOCH / MSS;
    - optionally use market-bias continuation fallback;
    - build internal StrategySignal only;
    - leave routing, filtering, confluence, portfolio coordination and
      risk-ready conversion to SignalProcessor.
    """

    prefer_external_layer: bool = True

    require_fresh_structure: bool = True
    require_primary_layer_eligible: bool = True
    require_break_event: bool = True
    require_alignment: bool = False

    allow_bos_entries: bool = True
    allow_choch_reversals: bool = True
    allow_mss_reversals: bool = True
    allow_bias_continuation_fallback: bool = True

    reverse_on_choch: bool = True
    reverse_on_mss: bool = True

    min_layer_confidence: float = 0.45
    min_layer_strength: float = 0.25
    min_alignment_score: float = 0.30
    min_break_confidence: float = 0.45
    min_break_score: float = 0.35
    min_trend_strength: float = 0.20
    min_swing_strength: float = 0.25
    min_swing_progression_score: float = 0.20
    min_break_distance_pct: float = 0.0002

    bos_score_bonus: float = 0.05
    choch_score_bonus: float = 0.06
    mss_score_bonus: float = 0.07
    alignment_score_bonus: float = 0.04
    liquidity_sweep_bonus: float = 0.03
    swing_strength_bonus: float = 0.03

    score_event_weight: float = 0.34
    score_layer_weight: float = 0.20
    score_alignment_weight: float = 0.16
    score_trend_weight: float = 0.12
    score_swing_weight: float = 0.10
    score_freshness_weight: float = 0.08

    confidence_event_weight: float = 0.52
    confidence_context_weight: float = 0.28
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    tag_market_structure_bos: str = "bos"
    tag_market_structure_choch: str = "choch"
    tag_market_structure_mss: str = "mss"
    tag_structure_break: str = "structure_break"
    tag_bias_continuation: str = "bias_continuation"
    tag_liquidity_sweep: str = "liquidity_sweep"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.BREAKOUT

    required_price_action_features: tuple[str, ...] = (
        PRICE_ACTION_FEATURES.MARKET_STRUCTURE,
    )

    def validate(self) -> None:
        PriceActionStrategyConfig.validate(self)

        unit_fields = {
            "min_layer_confidence": self.min_layer_confidence,
            "min_layer_strength": self.min_layer_strength,
            "min_alignment_score": self.min_alignment_score,
            "min_break_confidence": self.min_break_confidence,
            "min_break_score": self.min_break_score,
            "min_trend_strength": self.min_trend_strength,
            "min_swing_strength": self.min_swing_strength,
            "min_swing_progression_score": self.min_swing_progression_score,
            "bos_score_bonus": self.bos_score_bonus,
            "choch_score_bonus": self.choch_score_bonus,
            "mss_score_bonus": self.mss_score_bonus,
            "alignment_score_bonus": self.alignment_score_bonus,
            "liquidity_sweep_bonus": self.liquidity_sweep_bonus,
            "swing_strength_bonus": self.swing_strength_bonus,
        }

        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.min_break_distance_pct < 0:
            raise StrategyConfigError("min_break_distance_pct must be >= 0")

        score_weights = {
            "score_event_weight": self.score_event_weight,
            "score_layer_weight": self.score_layer_weight,
            "score_alignment_weight": self.score_alignment_weight,
            "score_trend_weight": self.score_trend_weight,
            "score_swing_weight": self.score_swing_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }
        confidence_weights = {
            "confidence_event_weight": self.confidence_event_weight,
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
            "tag_market_structure_bos",
            "tag_market_structure_choch",
            "tag_market_structure_mss",
            "tag_structure_break",
            "tag_bias_continuation",
            "tag_liquidity_sweep",
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


class MarketStructureStrategy(PriceActionTradingStrategy):
    """
    Unified market-structure strategy.

    Input:
        StrategyContext with FeatureSource.PRICE_ACTION domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.price_action.market_structure"
    category: StrategyCategory = StrategyCategory.PRICE_ACTION
    default_setup_type: SetupType = SetupType.BREAKOUT

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        price_action_config: MarketStructureStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_price_action_config = (
            price_action_config or MarketStructureStrategyConfig()
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

        self.structure_config: MarketStructureStrategyConfig = (
            resolved_price_action_config
        )

    @property
    def strategy_name(self) -> str:
        return "market_structure"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.PRICE_ACTION,
            timeframe=Timeframe.M1,
            tags=[
                self.structure_config.tag_price_action,
                self.structure_config.tag_market_structure,
                self.structure_config.tag_structure_break,
                self.structure_config.tag_breakout,
                self.structure_config.tag_reversal,
                "analytics_price_action",
            ],
            version="2.0.0",
            description=(
                "Interprets BOS/CHOCH/MSS and market-bias context from "
                "normalized price-action StrategyContext and returns internal "
                "StrategySignal."
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
                "strategy_type": "market_structure",
                "base_class": "PriceActionTradingStrategy",
                "canonical_payload": "PriceActionCompositeSnapshot",
                "uses_market_structure": True,
                "uses_bos": True,
                "uses_choch": True,
                "uses_mss": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(
            self.structure_config.required_price_action_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        if not self.has_any_price_action_data(
            context,
            tuple(self.structure_config.required_price_action_features),
        ):
            return None

        if self.has_stale_price_action_features(
            context,
            tuple(self.structure_config.required_price_action_features),
        ):
            return None

        view = self._extract_view(context)
        if view is None:
            return None

        if (
            self.structure_config.require_fresh_structure
            and is_stale(
                event_time=view.event_time,
                now=context.timestamp,
                stale_after_seconds=self.structure_config.stale_feature_max_age_seconds,
            )
        ):
            return None

        common_rejection = quality_filter_reason(
            view.primary_layer,
            min_confidence=self.structure_config.min_layer_confidence,
            min_score=self.structure_config.min_layer_strength,
            stale_after_seconds=self.structure_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )
        if common_rejection is not None and self.structure_config.require_primary_layer_eligible:
            return None

        side = self._infer_side(view)
        if not is_directional_side(side):
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

        if breakdown.score < self.structure_config.min_signal_score:
            return None

        if breakdown.confidence < self.structure_config.min_signal_confidence:
            return None

        source_features = self._source_features(view)
        tags = self._tags(view=view, setup_type=setup_type)

        event_label = (
            normalize_label(view.last_event.event_type)
            if view.last_event is not None
            else "bias_continuation"
        )

        reasons = list(
            dict.fromkeys(
                [
                    "market_structure_signal",
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
            "price_action_setup_family": "market_structure",
            "price_action_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "event_time": view.event_time.isoformat() if view.event_time else None,
            "primary_layer_name": normalize_label(view.primary_layer_name),
            "secondary_layer_name": normalize_label(view.secondary_layer_name),
            "bias": normalize_label(view.bias),
            "secondary_bias": normalize_label(view.secondary_bias),
            "last_event": serialize_for_metadata(view.last_event),
            "last_swing_high": serialize_for_metadata(view.last_swing_high),
            "last_swing_low": serialize_for_metadata(view.last_swing_low),
            "primary_layer": serialize_for_metadata(view.primary_layer),
            "secondary_layer": serialize_for_metadata(view.secondary_layer),
            "mtf_alignment_score": view.mtf_alignment_score,
            "swing_progression_score": view.swing_progression_score,
            "trend_strength": view.trend_strength,
            "layer_confidence": view.layer_confidence,
            "layer_strength": view.layer_strength,
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
            priority=self.structure_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_view(
        self,
        context: StrategyContext,
    ) -> MarketStructureContextView | None:
        module = self.resolve_price_action_module(
            context,
            "market_structure",
            aliases=("structure",),
        )
        if not module:
            return None

        primary = select_primary_layer(
            module,
            prefer_external_layer=self.structure_config.prefer_external_layer,
        )
        secondary = select_secondary_layer(
            module,
            prefer_external_layer=self.structure_config.prefer_external_layer,
        )

        if not primary:
            return None

        primary_layer_name = self._extract_layer_name(
            primary,
            fallback=StructureLayer.EXTERNAL
            if self.structure_config.prefer_external_layer
            else StructureLayer.INTERNAL,
        )
        secondary_layer_name = self._extract_layer_name(
            secondary,
            fallback=StructureLayer.INTERNAL
            if self.structure_config.prefer_external_layer
            else StructureLayer.EXTERNAL,
        )

        bias = parse_market_bias(
            get_path(primary, "bias")
            or get_path(primary, "market_bias")
            or get_path(module, "bias")
        )
        secondary_bias = parse_market_bias(
            get_path(secondary, "bias")
            or get_path(secondary, "market_bias")
        )

        last_event_payload = (
            get_path(primary, "last_break_event")
            or get_path(primary, "last_event")
            or get_path(module, "last_break_event")
            or get_path(module, "last_event")
            or extract_last_event(module)
        )
        last_event = StructureEventContext.from_payload(
            last_event_payload,
            fallback_layer=primary_layer_name,
            reverse_on_choch=self.structure_config.reverse_on_choch,
            reverse_on_mss=self.structure_config.reverse_on_mss,
        )

        last_swing_high = SwingContext.from_payload(
            get_path(primary, "last_swing_high")
            or get_path(primary, "swing_high")
            or get_path(module, "last_swing_high")
        )
        last_swing_low = SwingContext.from_payload(
            get_path(primary, "last_swing_low")
            or get_path(primary, "swing_low")
            or get_path(module, "last_swing_low")
        )

        mtf_alignment_score = unit_score(
            get_path(module, "mtf_alignment_score")
            or get_path(module, "mtf_alignment.score")
            or get_path(module, "alignment_score")
            or get_path(primary, "alignment_score")
        )
        swing_progression_score = unit_score(
            get_path(primary, "swing_progression_score")
            or get_path(module, "swing_progression_score")
            or get_path(module, "swing_progression.score")
        )
        trend_strength = unit_score(
            get_path(primary, "trend_strength")
            or get_path(module, "trend_strength")
            or get_path(module, "trend.strength")
        )

        event_time = (
            last_event.timestamp if last_event is not None else None
        ) or extract_last_update(primary) or extract_last_update(module)

        reasons = self._extract_reasons(module, primary, last_event)

        return MarketStructureContextView(
            module=module,
            primary_layer=primary,
            secondary_layer=secondary,
            primary_layer_name=primary_layer_name,
            secondary_layer_name=secondary_layer_name,
            bias=bias,
            secondary_bias=secondary_bias,
            last_event=last_event,
            last_swing_high=last_swing_high,
            last_swing_low=last_swing_low,
            mtf_alignment_score=mtf_alignment_score,
            swing_progression_score=swing_progression_score,
            trend_strength=trend_strength,
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
        event: StructureEventContext | None,
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

    def _infer_side(self, view: MarketStructureContextView) -> SignalSide:
        event = view.last_event

        if event is not None:
            event_type = normalize_label(event.event_type)

            if event_type == "bos" and self.structure_config.allow_bos_entries:
                if is_directional_side(event.side):
                    return event.side

            if event_type == "choch" and self.structure_config.allow_choch_reversals:
                if is_directional_side(event.side):
                    return event.side

            if event_type == "mss" and self.structure_config.allow_mss_reversals:
                if is_directional_side(event.side):
                    return event.side

        if self.structure_config.allow_bias_continuation_fallback:
            return market_bias_to_side(view.bias)

        return SignalSide.UNKNOWN

    def _infer_setup_type(
        self,
        view: MarketStructureContextView,
    ) -> SetupType:
        event = view.last_event
        if event is None or event.event_type is None:
            return SetupType.CONTINUATION

        event_type = normalize_label(event.event_type)

        if event_type == "bos":
            return SetupType.BREAKOUT

        if event_type in {"choch", "mss"}:
            return SetupType.REVERSAL

        return self.structure_config.default_setup_type

    def _passes_filters(
        self,
        *,
        view: MarketStructureContextView,
        side: SignalSide,
        setup_type: SetupType,
    ) -> bool:
        if self.structure_config.require_break_event and view.last_event is None:
            if not self.structure_config.allow_bias_continuation_fallback:
                return False

        if view.layer_confidence < self.structure_config.min_layer_confidence:
            return False

        if view.layer_strength < self.structure_config.min_layer_strength:
            return False

        if self.structure_config.require_alignment:
            if view.mtf_alignment_score < self.structure_config.min_alignment_score:
                return False

        if view.trend_strength < self.structure_config.min_trend_strength:
            return False

        if view.swing_progression_score < self.structure_config.min_swing_progression_score:
            if setup_type is SetupType.CONTINUATION:
                return False

        event = view.last_event
        if event is not None:
            if event.confidence < self.structure_config.min_break_confidence:
                return False

            if event.score < self.structure_config.min_break_score:
                return False

            if event.break_distance_pct < self.structure_config.min_break_distance_pct:
                return False

            event_type = normalize_label(event.event_type)
            if event_type == "bos" and not self.structure_config.allow_bos_entries:
                return False

            if event_type == "choch" and not self.structure_config.allow_choch_reversals:
                return False

            if event_type == "mss" and not self.structure_config.allow_mss_reversals:
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
        view: MarketStructureContextView,
        side: SignalSide,
        setup_type: SetupType,
    ) -> ScoreBreakdown:
        event_component = self._event_component(view)
        layer_component = average_score(view.layer_confidence, view.layer_strength)
        alignment_component = view.mtf_alignment_score
        trend_component = view.trend_strength
        swing_component = self._swing_component(view)
        fresh_component = freshness_score(
            event_time=view.event_time,
            now=context.timestamp,
            stale_after_seconds=self.structure_config.stale_feature_max_age_seconds,
        )

        components = {
            "event": event_component,
            "layer": layer_component,
            "alignment": alignment_component,
            "trend": trend_component,
            "swing": swing_component,
            "freshness": fresh_component,
        }
        weights = {
            "event": self.structure_config.score_event_weight,
            "layer": self.structure_config.score_layer_weight,
            "alignment": self.structure_config.score_alignment_weight,
            "trend": self.structure_config.score_trend_weight,
            "swing": self.structure_config.score_swing_weight,
            "freshness": self.structure_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=event_component)
        confidence = confidence_from_components(
            primary=max(event_component, view.layer_confidence),
            context=weighted_score(
                {
                    "layer": layer_component,
                    "trend": trend_component,
                    "alignment": alignment_component,
                },
                {
                    "layer": 0.40,
                    "trend": 0.30,
                    "alignment": 0.30,
                },
            ),
            confirmation=swing_component,
            freshness=fresh_component,
            primary_weight=self.structure_config.confidence_event_weight,
            context_weight=self.structure_config.confidence_context_weight,
            confirmation_weight=self.structure_config.confidence_confirmation_weight,
            freshness_weight=self.structure_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            f"side:{side.value}",
            f"setup_type:{setup_type.value}",
        ]

        event = view.last_event
        if event is not None and event.event_type is not None:
            event_type = normalize_label(event.event_type)
            confirmations.append(f"structure_event:{event_type}")

            if event_type == "bos":
                score += self.structure_config.bos_score_bonus
                confirmations.append("bos_confirms_breakout")

            elif event_type == "choch":
                score += self.structure_config.choch_score_bonus
                confirmations.append("choch_confirms_reversal")

            elif event_type == "mss":
                score += self.structure_config.mss_score_bonus
                confirmations.append("mss_confirms_reversal")

            if event.is_liquidity_sweep:
                score += self.structure_config.liquidity_sweep_bonus
                confirmations.append("liquidity_sweep_context")

        else:
            confirmations.append("bias_continuation_fallback")
            reasons.append("no_recent_structure_break_event")

        if view.mtf_alignment_score >= self.structure_config.min_alignment_score:
            score += self.structure_config.alignment_score_bonus
            confirmations.append("mtf_alignment_confirmed")

        swing_bonus = self._swing_bonus(view)
        if swing_bonus > 0:
            score += swing_bonus
            confirmations.append("swing_context_confirmed")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    def _event_component(self, view: MarketStructureContextView) -> float:
        event = view.last_event
        if event is None:
            return average_score(view.layer_confidence, view.layer_strength)

        distance_component = unit_score(
            event.break_distance_pct
            / max(self.structure_config.min_break_distance_pct * 5.0, 0.0001)
        )

        return weighted_score(
            {
                "confidence": event.confidence,
                "score": event.score,
                "distance": distance_component,
            },
            {
                "confidence": 0.45,
                "score": 0.40,
                "distance": 0.15,
            },
        )

    def _swing_component(self, view: MarketStructureContextView) -> float:
        swing_scores = []

        if view.last_swing_high is not None:
            swing_scores.append(view.last_swing_high.strength)

        if view.last_swing_low is not None:
            swing_scores.append(view.last_swing_low.strength)

        swing_scores.append(view.swing_progression_score)

        return average_score(*swing_scores)

    def _swing_bonus(self, view: MarketStructureContextView) -> float:
        if self._swing_component(view) >= self.structure_config.min_swing_strength:
            return self.structure_config.swing_strength_bonus

        return 0.0

    # ------------------------------------------------------------------
    # Source features / tags
    # ------------------------------------------------------------------

    def _source_features(self, view: MarketStructureContextView) -> list[str]:
        features = [
            *market_structure_source_features(),
            PRICE_ACTION_FEATURES.MARKET_STRUCTURE,
            PRICE_ACTION_FEATURES.MARKET_STRUCTURE_INTERNAL,
            PRICE_ACTION_FEATURES.MARKET_STRUCTURE_EXTERNAL,
            PRICE_ACTION_FEATURES.MARKET_STRUCTURE_LAST_BREAK_EVENT,
            PRICE_ACTION_FEATURES.MARKET_STRUCTURE_MTF_ALIGNMENT,
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        *,
        view: MarketStructureContextView,
        setup_type: SetupType,
    ) -> list[str]:
        tags = [
            self.structure_config.tag_price_action,
            self.structure_config.tag_market_structure,
            setup_type.value,
        ]

        event = view.last_event
        if event is not None and event.event_type is not None:
            event_type = normalize_label(event.event_type)

            if event_type == "bos":
                tags.append(self.structure_config.tag_market_structure_bos)
                tags.append(self.structure_config.tag_structure_break)

            elif event_type == "choch":
                tags.append(self.structure_config.tag_market_structure_choch)
                tags.append(self.structure_config.tag_reversal)

            elif event_type == "mss":
                tags.append(self.structure_config.tag_market_structure_mss)
                tags.append(self.structure_config.tag_reversal)

            if event.is_liquidity_sweep:
                tags.append(self.structure_config.tag_liquidity_sweep)

        else:
            tags.append(self.structure_config.tag_bias_continuation)

        if setup_type is SetupType.BREAKOUT:
            tags.append(self.structure_config.tag_breakout)

        if setup_type is SetupType.REVERSAL:
            tags.append(self.structure_config.tag_reversal)

        if setup_type is SetupType.CONTINUATION:
            tags.append(self.structure_config.tag_continuation)

        if view.primary_layer_name is not None:
            tags.append(f"layer:{normalize_label(view.primary_layer_name)}")

        return list(dict.fromkeys(tags))