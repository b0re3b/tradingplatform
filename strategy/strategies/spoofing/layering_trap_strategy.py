# trading_system/strategy/strategies/spoofing/layering_trap_strategy.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from analytics.spoofing.enums import SpoofingComponent, SpoofingPattern, SpoofingType
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
    SPOOFING_FEATURES,
    SpoofingCompositeSnapshot,
    SpoofingStrategyConfig,
    SpoofingTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    average_score,
    confidence_from_components,
    detector_agreement_ratio,
    detector_average_confidence,
    detector_confidence,
    detector_count,
    detector_passed,
    detector_score,
    extract_event_time,
    extract_fill_ratio,
    extract_layer_count,
    extract_layer_price_span_bps,
    extract_price_reaction_bps,
    extract_pull_ratio,
    extract_pulled_notional,
    extract_score,
    extract_wall_notional,
    freshness_score,
    is_directional_side,
    is_layering_signal,
    is_stale,
    layering_source_features,
    normalize_label,
    quality_filter_reason,
    reaction_aligns_with_side,
    serialize_for_metadata,
    spoofing_side_to_signal_side,
    unit_score,
    weighted_score,
)


@dataclass(slots=True)
class LayeringTrapPayload:
    """
    Normalized strategy-level payload для layering trap.

    Direction convention:
    - multi-level fake ASK supply removed -> fake resistance disappears -> LONG;
    - multi-level fake BID demand removed -> fake support disappears -> SHORT.
    """

    snapshot: SpoofingCompositeSnapshot
    side: SignalSide

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def layer_count(self) -> int:
        return extract_layer_count(self.snapshot.raw_signal)

    @property
    def layer_price_span_bps(self) -> float:
        return extract_layer_price_span_bps(self.snapshot.raw_signal)

    @property
    def pull_ratio(self) -> float:
        return extract_pull_ratio(self.snapshot.raw_signal)

    @property
    def fill_ratio(self) -> float:
        return extract_fill_ratio(self.snapshot.raw_signal)

    @property
    def price_reaction_bps(self) -> float:
        return extract_price_reaction_bps(self.snapshot.raw_signal)

    @property
    def wall_notional(self) -> float:
        return extract_wall_notional(self.snapshot.raw_signal)

    @property
    def pulled_notional(self) -> float:
        return extract_pulled_notional(self.snapshot.raw_signal)


@dataclass(slots=True)
class LayeringTrapStrategyConfig(SpoofingStrategyConfig):
    """
    Unified layering trap strategy config.

    Strategy idea:
    - analytics.spoofing detects multi-level layering;
    - layers are mostly pulled / low-filled;
    - layer span is compact enough to be one fake liquidity zone;
    - optional reaction confirms unwind direction;
    - strategy returns internal StrategySignal only.
    """

    min_layering_score: float = 0.70
    min_layering_confidence: float = 0.60

    min_layers: int = 2
    min_layer_total_notional: float = 0.0
    min_pulled_notional: float = 0.0
    min_synchronized_pull_ratio: float = 0.50
    max_fill_ratio: float = 0.40
    max_layer_price_span_bps: float = 8.0
    min_price_reaction_bps: float = 1.0

    require_layering_detector: bool = False
    require_layering_detector_passed: bool = False
    require_compact_layer_span: bool = True
    require_market_reaction: bool = False
    require_directional_reaction_alignment: bool = False

    min_layering_detector_score: float = 0.0
    min_layering_detector_confidence: float = 0.0

    score_base_weight: float = 0.24
    score_layers_weight: float = 0.20
    score_pull_weight: float = 0.18
    score_span_weight: float = 0.14
    score_detector_weight: float = 0.10
    score_reaction_weight: float = 0.08
    score_notional_weight: float = 0.03
    score_freshness_weight: float = 0.03

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    layer_count_bonus: float = 0.04
    synchronized_pull_bonus: float = 0.04
    compact_span_bonus: float = 0.03
    detector_bonus: float = 0.04
    reaction_bonus: float = 0.03
    notional_bonus: float = 0.03
    low_fill_bonus: float = 0.03

    strong_layer_count_threshold: int = 4
    strong_pull_ratio_threshold: float = 0.70
    low_fill_ratio_threshold: float = 0.25
    compact_span_threshold_bps: float = 4.0

    entry_offset_bps_hint: float | None = None
    stop_buffer_bps_hint: float | None = None
    take_profit_bps_hint: float | None = None
    trap_tp_multiplier_hint: float | None = None

    tag_layering_trap: str = "layering_trap"
    tag_multi_level_layering: str = "multi_level_layering"
    tag_synchronized_pull: str = "synchronized_pull"
    tag_compact_layers: str = "compact_layers"
    tag_unwind: str = "unwind"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.REVERSAL

    required_spoofing_features: tuple[str, ...] = (
        SPOOFING_FEATURES.SIGNAL,
    )

    def validate(self) -> None:
        SpoofingStrategyConfig.validate(self)

        unit_fields = {
            "min_layering_score": self.min_layering_score,
            "min_layering_confidence": self.min_layering_confidence,
            "min_synchronized_pull_ratio": self.min_synchronized_pull_ratio,
            "max_fill_ratio": self.max_fill_ratio,
            "min_layering_detector_score": self.min_layering_detector_score,
            "min_layering_detector_confidence": self.min_layering_detector_confidence,
            "layer_count_bonus": self.layer_count_bonus,
            "synchronized_pull_bonus": self.synchronized_pull_bonus,
            "compact_span_bonus": self.compact_span_bonus,
            "detector_bonus": self.detector_bonus,
            "reaction_bonus": self.reaction_bonus,
            "notional_bonus": self.notional_bonus,
            "low_fill_bonus": self.low_fill_bonus,
            "strong_pull_ratio_threshold": self.strong_pull_ratio_threshold,
            "low_fill_ratio_threshold": self.low_fill_ratio_threshold,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        non_negative_fields = {
            "min_layer_total_notional": self.min_layer_total_notional,
            "min_pulled_notional": self.min_pulled_notional,
            "max_layer_price_span_bps": self.max_layer_price_span_bps,
            "min_price_reaction_bps": self.min_price_reaction_bps,
            "compact_span_threshold_bps": self.compact_span_threshold_bps,
        }
        for field_name, value in non_negative_fields.items():
            if float(value) < 0.0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if self.min_layers < 1:
            raise StrategyConfigError("min_layers must be >= 1")

        if self.strong_layer_count_threshold < 1:
            raise StrategyConfigError("strong_layer_count_threshold must be >= 1")

        hint_fields = {
            "entry_offset_bps_hint": self.entry_offset_bps_hint,
            "stop_buffer_bps_hint": self.stop_buffer_bps_hint,
            "take_profit_bps_hint": self.take_profit_bps_hint,
            "trap_tp_multiplier_hint": self.trap_tp_multiplier_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        score_weights = {
            "score_base_weight": self.score_base_weight,
            "score_layers_weight": self.score_layers_weight,
            "score_pull_weight": self.score_pull_weight,
            "score_span_weight": self.score_span_weight,
            "score_detector_weight": self.score_detector_weight,
            "score_reaction_weight": self.score_reaction_weight,
            "score_notional_weight": self.score_notional_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }
        confidence_weights = {
            "confidence_primary_weight": self.confidence_primary_weight,
            "confidence_context_weight": self.confidence_context_weight,
            "confidence_confirmation_weight": self.confidence_confirmation_weight,
            "confidence_freshness_weight": self.confidence_freshness_weight,
        }

        for field_name, value in {**score_weights, **confidence_weights}.items():
            if float(value) < 0.0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if sum(score_weights.values()) <= 0:
            raise StrategyConfigError("score weights sum must be > 0")

        if sum(confidence_weights.values()) <= 0:
            raise StrategyConfigError("confidence weights sum must be > 0")

        for attr in (
            "tag_layering_trap",
            "tag_multi_level_layering",
            "tag_synchronized_pull",
            "tag_compact_layers",
            "tag_unwind",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_spoofing_features:
            raise StrategyConfigError("required_spoofing_features cannot be empty")


class LayeringTrapStrategy(SpoofingTradingStrategy):
    """
    Unified layering trap strategy.

    Input:
        StrategyContext with FeatureSource.SPOOFING domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.spoofing.layering_trap"
    category: StrategyCategory = StrategyCategory.SPOOFING
    default_setup_type: SetupType = SetupType.REVERSAL

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        spoofing_config: LayeringTrapStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_spoofing_config = spoofing_config or LayeringTrapStrategyConfig()
        resolved_spoofing_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            spoofing_config=resolved_spoofing_config,
            service_name=service_name,
        )

        self.layering_config: LayeringTrapStrategyConfig = resolved_spoofing_config

    @property
    def strategy_name(self) -> str:
        return "layering_trap"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.SPOOFING,
            timeframe=Timeframe.M1,
            tags=[
                self.layering_config.tag_spoofing,
                self.layering_config.tag_layering,
                self.layering_config.tag_layering_trap,
                self.layering_config.tag_multi_level_layering,
                self.layering_config.tag_reversal,
                "analytics_spoofing",
            ],
            version="2.0.0",
            description=(
                "Interprets multi-level layering spoofing traps from normalized "
                "StrategyContext and returns internal StrategySignal."
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
                "source": "analytics.spoofing",
                "strategy_type": "layering_trap",
                "base_class": "SpoofingTradingStrategy",
                "canonical_payload": "SpoofingCompositeSnapshot",
                "uses_layering_detector": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(
            self.layering_config.required_spoofing_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        if not self.has_any_spoofing_data(
            context,
            tuple(self.layering_config.required_spoofing_features),
        ):
            return None

        if self.has_stale_spoofing_features(
            context,
            tuple(self.layering_config.required_spoofing_features),
        ):
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        if is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.layering_config.stale_feature_max_age_seconds,
        ):
            return None

        rejection = quality_filter_reason(
            payload.snapshot.raw_signal,
            min_score=max(
                self.layering_config.min_score,
                self.layering_config.min_layering_score,
            ),
            min_confidence=max(
                self.layering_config.min_confidence,
                self.layering_config.min_layering_confidence,
            ),
            allowed_severities=self.layering_config.allowed_severities,
            min_detector_count=self.layering_config.min_detector_count,
            min_agreement_ratio=self.layering_config.min_agreement_ratio,
            min_average_confidence=self.layering_config.min_average_confidence,
            require_score_passed=self.layering_config.require_score_passed,
            stale_after_seconds=self.layering_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )
        if rejection is not None:
            return None

        if not self.accepts_spoofing_snapshot(payload.snapshot):
            return None

        if not self._supports_snapshot(payload.snapshot):
            return None

        if not self._passes_layering_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.layering_config.min_layering_score:
            return None

        if breakdown.confidence < self.layering_config.min_layering_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "layering_trap_signal",
                    f"side:{payload.side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "spoofing_setup_family": "layering_trap",
            "spoofing_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "snapshot": serialize_for_metadata(payload.snapshot.to_dict()),
            "raw": serialize_for_metadata(payload.raw),
            "event_time": payload.event_time.isoformat() if payload.event_time else None,
            "spoofing_type": normalize_label(payload.snapshot.spoofing_type),
            "pattern": normalize_label(payload.snapshot.pattern),
            "spoofing_side": normalize_label(payload.snapshot.side),
            "mapped_side": payload.side.value,
            "score": payload.snapshot.score,
            "confidence": payload.snapshot.confidence,
            "layer_count": payload.layer_count,
            "layer_price_span_bps": payload.layer_price_span_bps,
            "pull_ratio": payload.pull_ratio,
            "fill_ratio": payload.fill_ratio,
            "price_reaction_bps": payload.price_reaction_bps,
            "signed_price_reaction_bps": payload.snapshot.signed_price_reaction_bps,
            "wall_notional": payload.wall_notional,
            "pulled_notional": payload.pulled_notional,
            "detector_count": detector_count(payload.snapshot.raw_signal),
            "detector_agreement_ratio": detector_agreement_ratio(payload.snapshot.raw_signal),
            "detector_average_confidence": detector_average_confidence(payload.snapshot.raw_signal),
            "execution_hints": self._execution_hints(),
        }

        return self.build_spoofing_signal(
            context=context,
            side=payload.side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.layering_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.layering_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> LayeringTrapPayload | None:
        snapshot = self.resolve_spoofing_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        side = spoofing_side_to_signal_side(snapshot.side)
        if not is_directional_side(side):
            return None

        event_time = (
            extract_event_time(snapshot.raw_signal)
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "layering_trap_context",
            f"spoofing_type:{normalize_label(snapshot.spoofing_type)}",
            f"pattern:{normalize_label(snapshot.pattern)}",
            f"spoofing_side:{normalize_label(snapshot.side)}",
            f"score:{snapshot.score:.4f}",
            f"confidence:{snapshot.confidence:.4f}",
        ]

        if is_layering_signal(snapshot.raw_signal):
            reasons.append("layering_signal_detected")

        return LayeringTrapPayload(
            snapshot=snapshot,
            side=side,
            event_time=event_time,
            reasons=list(dict.fromkeys(reasons)),
            raw=snapshot.raw_signal,
        )

    # ------------------------------------------------------------------
    # Support / filters
    # ------------------------------------------------------------------

    def _supports_snapshot(
        self,
        snapshot: SpoofingCompositeSnapshot,
    ) -> bool:
        return (
            snapshot.spoofing_type is SpoofingType.LAYERING
            or snapshot.pattern is SpoofingPattern.MULTI_LEVEL_LAYERING
            or snapshot.has_detector(SpoofingComponent.LAYERING_DETECTOR)
            or is_layering_signal(snapshot.raw_signal)
        )

    def _passes_layering_filters(
        self,
        payload: LayeringTrapPayload,
    ) -> bool:
        snapshot = payload.snapshot

        if self.layering_config.require_layering_detector:
            if not snapshot.has_detector(SpoofingComponent.LAYERING_DETECTOR):
                return False

        if self.layering_config.require_layering_detector_passed:
            if not detector_passed(snapshot.raw_signal, SpoofingComponent.LAYERING_DETECTOR):
                return False

        if payload.layer_count < self.layering_config.min_layers:
            return False

        if payload.wall_notional < self.layering_config.min_layer_total_notional:
            return False

        if payload.pulled_notional < self.layering_config.min_pulled_notional:
            return False

        if payload.pull_ratio < self.layering_config.min_synchronized_pull_ratio:
            return False

        if payload.fill_ratio > self.layering_config.max_fill_ratio:
            return False

        if self.layering_config.require_compact_layer_span:
            if payload.layer_price_span_bps > self.layering_config.max_layer_price_span_bps:
                return False

        if self.layering_config.require_market_reaction:
            if payload.price_reaction_bps < self.layering_config.min_price_reaction_bps:
                return False

        if self.layering_config.require_directional_reaction_alignment:
            if not reaction_aligns_with_side(
                signed_reaction_bps=snapshot.signed_price_reaction_bps,
                side=payload.side,
                min_reaction_bps=self.layering_config.min_price_reaction_bps,
            ):
                return False

        if detector_score(
            snapshot.raw_signal,
            SpoofingComponent.LAYERING_DETECTOR,
        ) < self.layering_config.min_layering_detector_score:
            if self.layering_config.require_layering_detector:
                return False

        if detector_confidence(
            snapshot.raw_signal,
            SpoofingComponent.LAYERING_DETECTOR,
        ) < self.layering_config.min_layering_detector_confidence:
            if self.layering_config.require_layering_detector:
                return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: LayeringTrapPayload,
    ) -> ScoreBreakdown:
        snapshot = payload.snapshot

        base_component = average_score(
            extract_score(snapshot.raw_signal),
            snapshot.score,
            snapshot.confidence,
        )
        layers_component = unit_score(
            payload.layer_count / max(self.layering_config.min_layers * 3, 1)
        )
        pull_component = payload.pull_ratio
        span_component = self._span_component(payload)
        detector_component = average_score(
            detector_score(snapshot.raw_signal, SpoofingComponent.LAYERING_DETECTOR),
            detector_confidence(snapshot.raw_signal, SpoofingComponent.LAYERING_DETECTOR),
            detector_agreement_ratio(snapshot.raw_signal),
            detector_average_confidence(snapshot.raw_signal),
        )
        reaction_component = unit_score(
            payload.price_reaction_bps
            / max(self.layering_config.min_price_reaction_bps * 4.0, 0.01)
        )
        notional_component = self._notional_component(payload)
        fresh_component = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.layering_config.stale_feature_max_age_seconds,
        )

        components = {
            "base": base_component,
            "layers": layers_component,
            "pull": pull_component,
            "span": span_component,
            "detector": detector_component,
            "reaction": reaction_component,
            "notional": notional_component,
            "freshness": fresh_component,
        }
        weights = {
            "base": self.layering_config.score_base_weight,
            "layers": self.layering_config.score_layers_weight,
            "pull": self.layering_config.score_pull_weight,
            "span": self.layering_config.score_span_weight,
            "detector": self.layering_config.score_detector_weight,
            "reaction": self.layering_config.score_reaction_weight,
            "notional": self.layering_config.score_notional_weight,
            "freshness": self.layering_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=base_component)
        confidence = confidence_from_components(
            primary=base_component,
            context=average_score(layers_component, span_component, detector_component),
            confirmation=average_score(pull_component, reaction_component, notional_component),
            freshness=fresh_component,
            primary_weight=self.layering_config.confidence_primary_weight,
            context_weight=self.layering_config.confidence_context_weight,
            confirmation_weight=self.layering_config.confidence_confirmation_weight,
            freshness_weight=self.layering_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "layering_trap_context",
            f"side:{payload.side.value}",
            f"layer_count:{payload.layer_count}",
            f"layer_price_span_bps:{payload.layer_price_span_bps:.4f}",
            f"pull_ratio:{payload.pull_ratio:.4f}",
            f"fill_ratio:{payload.fill_ratio:.4f}",
        ]

        if payload.layer_count >= self.layering_config.strong_layer_count_threshold:
            score += self.layering_config.layer_count_bonus
            confirmations.append("strong_layer_count")

        if payload.pull_ratio >= self.layering_config.strong_pull_ratio_threshold:
            score += self.layering_config.synchronized_pull_bonus
            confirmations.append("synchronized_layer_pull")

        if payload.layer_price_span_bps <= self.layering_config.compact_span_threshold_bps:
            score += self.layering_config.compact_span_bonus
            confirmations.append("compact_layering_zone")

        if payload.fill_ratio <= self.layering_config.low_fill_ratio_threshold:
            score += self.layering_config.low_fill_bonus
            confirmations.append("low_fill_layering")

        if snapshot.has_detector(SpoofingComponent.LAYERING_DETECTOR):
            score += self.layering_config.detector_bonus
            confirmations.append("layering_detector_context")

        if payload.price_reaction_bps >= self.layering_config.min_price_reaction_bps:
            score += self.layering_config.reaction_bonus
            confirmations.append("layering_reaction_context")

        if payload.pulled_notional > 0:
            score += self.layering_config.notional_bonus
            confirmations.append("layering_pulled_notional_confirmed")

        if reaction_aligns_with_side(
            signed_reaction_bps=snapshot.signed_price_reaction_bps,
            side=payload.side,
            min_reaction_bps=self.layering_config.min_price_reaction_bps,
        ):
            confirmations.append("directional_layering_unwind")

        if payload.wall_notional > 0:
            reasons.append(f"wall_notional:{payload.wall_notional:.4f}")

        if payload.pulled_notional > 0:
            reasons.append(f"pulled_notional:{payload.pulled_notional:.4f}")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    def _span_component(
        self,
        payload: LayeringTrapPayload,
    ) -> float:
        max_span = max(self.layering_config.max_layer_price_span_bps, 0.0001)
        return unit_score(1.0 - (payload.layer_price_span_bps / max_span))

    def _notional_component(
        self,
        payload: LayeringTrapPayload,
    ) -> float:
        if payload.wall_notional <= 0:
            return 0.0

        return unit_score(payload.pulled_notional / max(payload.wall_notional, 1.0))

    # ------------------------------------------------------------------
    # Source features / tags / execution hints
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: LayeringTrapPayload,
    ) -> list[str]:
        features = [
            *layering_source_features(),
            SPOOFING_FEATURES.SIGNAL,
            SPOOFING_FEATURES.SPOOFING_TYPE,
            SPOOFING_FEATURES.PATTERN,
            SPOOFING_FEATURES.SIDE,
            SPOOFING_FEATURES.SCORE,
            SPOOFING_FEATURES.CONFIDENCE,
            SPOOFING_FEATURES.LAYER_COUNT,
            SPOOFING_FEATURES.LAYER_PRICE_SPAN_BPS,
            SPOOFING_FEATURES.PULL_RATIO,
            SPOOFING_FEATURES.FILL_RATIO,
            SPOOFING_FEATURES.PRICE_REACTION_BPS,
            SPOOFING_FEATURES.WALL_NOTIONAL,
            SPOOFING_FEATURES.PULLED_NOTIONAL,
            SPOOFING_FEATURES.DETECTOR_RESULTS,
            SPOOFING_FEATURES.SCORE_BREAKDOWN,
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: LayeringTrapPayload,
    ) -> list[str]:
        tags = [
            self.layering_config.tag_spoofing,
            self.layering_config.tag_layering,
            self.layering_config.tag_layering_trap,
            self.layering_config.tag_multi_level_layering,
            self.layering_config.tag_reversal,
            self.layering_config.tag_unwind,
            f"side:{payload.side.value}",
        ]

        if payload.pull_ratio >= self.layering_config.strong_pull_ratio_threshold:
            tags.append(self.layering_config.tag_synchronized_pull)

        if payload.layer_price_span_bps <= self.layering_config.compact_span_threshold_bps:
            tags.append(self.layering_config.tag_compact_layers)

        if payload.snapshot.spoofing_type is not None:
            tags.append(f"type:{normalize_label(payload.snapshot.spoofing_type)}")

        if payload.snapshot.pattern is not None:
            tags.append(f"pattern:{normalize_label(payload.snapshot.pattern)}")

        return list(dict.fromkeys(tags))

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        return {
            "entry_offset_bps": self.layering_config.entry_offset_bps_hint,
            "stop_buffer_bps": self.layering_config.stop_buffer_bps_hint,
            "take_profit_bps": self.layering_config.take_profit_bps_hint,
            "trap_tp_multiplier": self.layering_config.trap_tp_multiplier_hint,
        }