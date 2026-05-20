# trading_system/strategy/strategies/spoofing/spoofing_reversal_strategy.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from analytics.spoofing.enums import (
    SpoofingComponent,
)
from core.event_bus import EventBus
from core.scheduler import Scheduler
from .base import (
    SPOOFING_FEATURES,
    SpoofingCompositeSnapshot,
    SpoofingStrategyConfig,
    SpoofingTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    average_score,
    base_spoofing_source_features,
    confidence_from_components,
    composite_spoofing_source_features,
    detector_agreement_ratio,
    detector_average_confidence,
    detector_count,
    detector_passed,
    detector_score,
    extract_cancel_to_fill_ratio,
    extract_event_time,
    extract_fill_ratio,
    extract_price_reaction_bps,
    extract_pull_ratio,
    extract_pulled_notional,
    extract_score,
    extract_wall_notional,
    freshness_score,
    is_composite_signal,
    is_directional_side,
    is_layering_signal,
    is_order_pull_signal,
    is_pressure_bluff_signal,
    is_stale,
    layering_source_features,
    normalize_label,
    order_pull_source_features,
    pressure_bluff_source_features,
    quality_filter_reason,
    reaction_aligns_with_side,
    serialize_for_metadata,
    spoofing_side_to_signal_side,
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
class SpoofingReversalPayload:
    """
    Normalized strategy-level payload для spoofing reversal.

    Strategy idea:
    - fake ASK pressure / pulled ask wall -> fake resistance removed -> LONG;
    - fake BID pressure / pulled bid wall -> fake support removed -> SHORT.
    """

    snapshot: SpoofingCompositeSnapshot
    side: SignalSide

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

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

    @property
    def cancel_to_fill_ratio(self) -> float:
        return extract_cancel_to_fill_ratio(self.snapshot.raw_signal)


@dataclass(slots=True)
class SpoofingReversalStrategyConfig(SpoofingStrategyConfig):
    """
    Unified spoofing reversal config.

    This is an umbrella/fallback reversal strategy for analytics-level spoofing
    signals. Narrower strategies can still handle order-pull, pressure-bluff,
    layering and fake-liquidity cases separately.
    """

    allow_order_pull: bool = True
    allow_pressure_bluff: bool = True
    allow_layering: bool = True
    allow_composite: bool = True

    min_reversal_score: float = 0.68
    min_reversal_confidence: float = 0.58

    min_pull_ratio: float = 0.55
    max_fill_ratio: float = 0.35
    min_price_reaction_bps: float = 1.5

    min_wall_notional: float = 0.0
    min_pulled_notional: float = 0.0
    min_cancel_to_fill_ratio: float = 0.0
    max_lifetime_ms: float | None = None

    min_pressure_flip_strength: float = 0.0
    min_layer_count: int = 0
    max_layer_price_span_bps: float | None = None

    min_composite_detector_count: int = 2
    min_composite_agreement_ratio: float = 0.50
    min_composite_average_confidence: float = 0.50

    require_fast_pull_or_reaction: bool = False
    require_directional_reaction_alignment: bool = False
    require_detector_passed_for_known_type: bool = False

    score_base_weight: float = 0.30
    score_pull_weight: float = 0.18
    score_fill_weight: float = 0.14
    score_reaction_weight: float = 0.16
    score_detector_weight: float = 0.12
    score_context_weight: float = 0.06
    score_freshness_weight: float = 0.04

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    order_pull_bonus: float = 0.04
    pressure_bluff_bonus: float = 0.04
    layering_bonus: float = 0.04
    composite_bonus: float = 0.05
    directional_reaction_bonus: float = 0.04
    high_pull_bonus: float = 0.03

    execution_entry_offset_bps_hint: float | None = None
    execution_stop_buffer_bps_hint: float | None = None
    execution_take_profit_bps_hint: float | None = None
    reaction_tp_multiplier_hint: float | None = None

    tag_spoofing_reversal: str = "spoofing_reversal"
    tag_order_pull_reversal: str = "order_pull_reversal"
    tag_pressure_bluff_reversal: str = "pressure_bluff_reversal"
    tag_layering_reversal: str = "layering_reversal"
    tag_composite_reversal: str = "composite_reversal"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.REVERSAL

    required_spoofing_features: tuple[str, ...] = (
        SPOOFING_FEATURES.SIGNAL,
    )

    def validate(self) -> None:
        SpoofingStrategyConfig.validate(self)

        unit_fields = {
            "min_reversal_score": self.min_reversal_score,
            "min_reversal_confidence": self.min_reversal_confidence,
            "min_pull_ratio": self.min_pull_ratio,
            "max_fill_ratio": self.max_fill_ratio,
            "min_cancel_to_fill_ratio": self.min_cancel_to_fill_ratio,
            "min_pressure_flip_strength": self.min_pressure_flip_strength,
            "min_composite_agreement_ratio": self.min_composite_agreement_ratio,
            "min_composite_average_confidence": self.min_composite_average_confidence,
            "order_pull_bonus": self.order_pull_bonus,
            "pressure_bluff_bonus": self.pressure_bluff_bonus,
            "layering_bonus": self.layering_bonus,
            "composite_bonus": self.composite_bonus,
            "directional_reaction_bonus": self.directional_reaction_bonus,
            "high_pull_bonus": self.high_pull_bonus,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        non_negative = {
            "min_price_reaction_bps": self.min_price_reaction_bps,
            "min_wall_notional": self.min_wall_notional,
            "min_pulled_notional": self.min_pulled_notional,
        }
        for field_name, value in non_negative.items():
            if float(value) < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if self.min_layer_count < 0:
            raise StrategyConfigError("min_layer_count must be >= 0")

        if self.min_composite_detector_count < 0:
            raise StrategyConfigError("min_composite_detector_count must be >= 0")

        if self.max_lifetime_ms is not None and self.max_lifetime_ms <= 0:
            raise StrategyConfigError("max_lifetime_ms must be > 0")

        if self.max_layer_price_span_bps is not None and self.max_layer_price_span_bps < 0:
            raise StrategyConfigError("max_layer_price_span_bps must be >= 0")

        hint_fields = {
            "execution_entry_offset_bps_hint": self.execution_entry_offset_bps_hint,
            "execution_stop_buffer_bps_hint": self.execution_stop_buffer_bps_hint,
            "execution_take_profit_bps_hint": self.execution_take_profit_bps_hint,
            "reaction_tp_multiplier_hint": self.reaction_tp_multiplier_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        score_weights = {
            "score_base_weight": self.score_base_weight,
            "score_pull_weight": self.score_pull_weight,
            "score_fill_weight": self.score_fill_weight,
            "score_reaction_weight": self.score_reaction_weight,
            "score_detector_weight": self.score_detector_weight,
            "score_context_weight": self.score_context_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }
        confidence_weights = {
            "confidence_primary_weight": self.confidence_primary_weight,
            "confidence_context_weight": self.confidence_context_weight,
            "confidence_confirmation_weight": self.confidence_confirmation_weight,
            "confidence_freshness_weight": self.confidence_freshness_weight,
        }

        for field_name, value in {**score_weights, **confidence_weights}.items():
            if float(value) < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if sum(score_weights.values()) <= 0:
            raise StrategyConfigError("score weights sum must be > 0")

        if sum(confidence_weights.values()) <= 0:
            raise StrategyConfigError("confidence weights sum must be > 0")

        for attr in (
            "tag_spoofing_reversal",
            "tag_order_pull_reversal",
            "tag_pressure_bluff_reversal",
            "tag_layering_reversal",
            "tag_composite_reversal",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_spoofing_features:
            raise StrategyConfigError("required_spoofing_features cannot be empty")


class SpoofingReversalStrategy(SpoofingTradingStrategy):
    """
    Unified spoofing reversal strategy.

    Input:
        StrategyContext with FeatureSource.SPOOFING domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.spoofing.reversal"
    category: StrategyCategory = StrategyCategory.SPOOFING
    default_setup_type: SetupType = SetupType.REVERSAL

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        spoofing_config: SpoofingReversalStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_spoofing_config = spoofing_config or SpoofingReversalStrategyConfig()
        resolved_spoofing_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            spoofing_config=resolved_spoofing_config,
            service_name=service_name,
        )

        self.reversal_config: SpoofingReversalStrategyConfig = resolved_spoofing_config

    @property
    def strategy_name(self) -> str:
        return "spoofing_reversal"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.SPOOFING,
            timeframe=Timeframe.M1,
            tags=[
                self.reversal_config.tag_spoofing,
                self.reversal_config.tag_reversal,
                self.reversal_config.tag_spoofing_reversal,
                self.reversal_config.tag_order_pull,
                self.reversal_config.tag_pressure_bluff,
                self.reversal_config.tag_layering,
                self.reversal_config.tag_composite,
                "analytics_spoofing",
            ],
            version="2.0.0",
            description=(
                "Interprets analytics-level spoofing reversal signals from "
                "normalized StrategyContext and returns internal StrategySignal."
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
                "strategy_type": "spoofing_reversal",
                "base_class": "SpoofingTradingStrategy",
                "canonical_payload": "SpoofingCompositeSnapshot",
                "uses_order_pull": True,
                "uses_pressure_bluff": True,
                "uses_layering": True,
                "uses_composite": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(self.reversal_config.required_spoofing_features)

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        if not self.has_any_spoofing_data(
            context,
            tuple(self.reversal_config.required_spoofing_features),
        ):
            return None

        if self.has_stale_spoofing_features(
            context,
            tuple(self.reversal_config.required_spoofing_features),
        ):
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        if is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.reversal_config.stale_feature_max_age_seconds,
        ):
            return None

        rejection = quality_filter_reason(
            payload.snapshot.raw_signal,
            min_score=max(self.reversal_config.min_score, self.reversal_config.min_reversal_score),
            min_confidence=max(
                self.reversal_config.min_confidence,
                self.reversal_config.min_reversal_confidence,
            ),
            allowed_severities=self.reversal_config.allowed_severities,
            min_detector_count=self.reversal_config.min_detector_count,
            min_agreement_ratio=self.reversal_config.min_agreement_ratio,
            min_average_confidence=self.reversal_config.min_average_confidence,
            require_score_passed=self.reversal_config.require_score_passed,
            stale_after_seconds=self.reversal_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )
        if rejection is not None:
            return None

        if not self.accepts_spoofing_snapshot(payload.snapshot):
            return None

        if not self._supports_snapshot(payload.snapshot):
            return None

        if not self._passes_reversal_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.reversal_config.min_reversal_score:
            return None

        if breakdown.confidence < self.reversal_config.min_reversal_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "spoofing_reversal_signal",
                    f"side:{payload.side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "spoofing_setup_family": "spoofing_reversal",
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
            "pull_ratio": payload.pull_ratio,
            "fill_ratio": payload.fill_ratio,
            "price_reaction_bps": payload.price_reaction_bps,
            "signed_price_reaction_bps": payload.snapshot.signed_price_reaction_bps,
            "wall_notional": payload.wall_notional,
            "pulled_notional": payload.pulled_notional,
            "cancel_to_fill_ratio": payload.cancel_to_fill_ratio,
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
            setup_type=self.reversal_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.reversal_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> SpoofingReversalPayload | None:
        snapshot = self.resolve_spoofing_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        side = spoofing_side_to_signal_side(snapshot.side)
        if not is_directional_side(side):
            return None

        event_time = extract_event_time(snapshot.raw_signal) or snapshot.timestamp or context.timestamp

        reasons = [
            f"spoofing_type:{normalize_label(snapshot.spoofing_type)}",
            f"pattern:{normalize_label(snapshot.pattern)}",
            f"spoofing_side:{normalize_label(snapshot.side)}",
            f"score:{snapshot.score:.4f}",
            f"confidence:{snapshot.confidence:.4f}",
        ]

        if is_order_pull_signal(snapshot.raw_signal):
            reasons.append("order_pull_reversal_context")

        if is_pressure_bluff_signal(snapshot.raw_signal):
            reasons.append("pressure_bluff_reversal_context")

        if is_layering_signal(snapshot.raw_signal):
            reasons.append("layering_reversal_context")

        if is_composite_signal(snapshot.raw_signal):
            reasons.append("composite_spoofing_reversal_context")

        return SpoofingReversalPayload(
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
        if self.reversal_config.allow_order_pull and is_order_pull_signal(snapshot.raw_signal):
            return True

        if self.reversal_config.allow_pressure_bluff and is_pressure_bluff_signal(snapshot.raw_signal):
            return True

        if self.reversal_config.allow_layering and is_layering_signal(snapshot.raw_signal):
            return True

        if self.reversal_config.allow_composite and is_composite_signal(snapshot.raw_signal):
            return True

        return False

    def _passes_reversal_filters(
        self,
        payload: SpoofingReversalPayload,
    ) -> bool:
        snapshot = payload.snapshot

        if payload.pull_ratio < self.reversal_config.min_pull_ratio:
            return False

        if payload.fill_ratio > self.reversal_config.max_fill_ratio:
            return False

        if payload.price_reaction_bps < self.reversal_config.min_price_reaction_bps:
            if self.reversal_config.require_fast_pull_or_reaction:
                return False

        if payload.wall_notional < self.reversal_config.min_wall_notional:
            return False

        if payload.pulled_notional < self.reversal_config.min_pulled_notional:
            return False

        if payload.cancel_to_fill_ratio < self.reversal_config.min_cancel_to_fill_ratio:
            return False

        if self.reversal_config.max_lifetime_ms is not None:
            if snapshot.lifetime_ms > self.reversal_config.max_lifetime_ms:
                return False

        if self.reversal_config.require_directional_reaction_alignment:
            if not reaction_aligns_with_side(
                signed_reaction_bps=snapshot.signed_price_reaction_bps,
                side=payload.side,
                min_reaction_bps=self.reversal_config.min_price_reaction_bps,
            ):
                return False

        if is_pressure_bluff_signal(snapshot.raw_signal):
            if snapshot.pressure_flip_strength < self.reversal_config.min_pressure_flip_strength:
                return False

        if is_layering_signal(snapshot.raw_signal):
            if snapshot.layer_count < self.reversal_config.min_layer_count:
                return False

            if self.reversal_config.max_layer_price_span_bps is not None:
                if snapshot.layer_price_span_bps > self.reversal_config.max_layer_price_span_bps:
                    return False

        if is_composite_signal(snapshot.raw_signal):
            if detector_count(snapshot.raw_signal) < self.reversal_config.min_composite_detector_count:
                return False

            if detector_agreement_ratio(snapshot.raw_signal) < self.reversal_config.min_composite_agreement_ratio:
                return False

            if detector_average_confidence(snapshot.raw_signal) < self.reversal_config.min_composite_average_confidence:
                return False

        if self.reversal_config.require_detector_passed_for_known_type:
            component = self._primary_detector_component(snapshot)
            if component is not None and not detector_passed(snapshot.raw_signal, component):
                return False

        return True

    def _primary_detector_component(
        self,
        snapshot: SpoofingCompositeSnapshot,
    ) -> SpoofingComponent | None:
        if is_order_pull_signal(snapshot.raw_signal):
            return SpoofingComponent.ORDER_PULL_DETECTOR

        if is_pressure_bluff_signal(snapshot.raw_signal):
            return SpoofingComponent.FLIP_PRESSURE_DETECTOR

        if is_layering_signal(snapshot.raw_signal):
            return SpoofingComponent.LAYERING_DETECTOR

        return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: SpoofingReversalPayload,
    ) -> ScoreBreakdown:
        snapshot = payload.snapshot

        base_component = average_score(
            extract_score(snapshot.raw_signal),
            snapshot.score,
            snapshot.confidence,
        )
        pull_component = payload.pull_ratio
        fill_component = unit_score(1.0 - payload.fill_ratio)
        reaction_component = unit_score(
            payload.price_reaction_bps
            / max(self.reversal_config.min_price_reaction_bps * 4.0, 0.01)
        )
        detector_component = average_score(
            detector_agreement_ratio(snapshot.raw_signal),
            detector_average_confidence(snapshot.raw_signal),
            detector_score(snapshot.raw_signal, self._primary_detector_component(snapshot))
            if self._primary_detector_component(snapshot) is not None
            else snapshot.detector_average_confidence,
        )
        context_component = self._context_component(snapshot)
        fresh_component = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.reversal_config.stale_feature_max_age_seconds,
        )

        components = {
            "base": base_component,
            "pull": pull_component,
            "fill": fill_component,
            "reaction": reaction_component,
            "detector": detector_component,
            "context": context_component,
            "freshness": fresh_component,
        }
        weights = {
            "base": self.reversal_config.score_base_weight,
            "pull": self.reversal_config.score_pull_weight,
            "fill": self.reversal_config.score_fill_weight,
            "reaction": self.reversal_config.score_reaction_weight,
            "detector": self.reversal_config.score_detector_weight,
            "context": self.reversal_config.score_context_weight,
            "freshness": self.reversal_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=base_component)
        confidence = confidence_from_components(
            primary=base_component,
            context=context_component,
            confirmation=average_score(pull_component, fill_component, reaction_component),
            freshness=fresh_component,
            primary_weight=self.reversal_config.confidence_primary_weight,
            context_weight=self.reversal_config.confidence_context_weight,
            confirmation_weight=self.reversal_config.confidence_confirmation_weight,
            freshness_weight=self.reversal_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "spoofing_reversal_context",
            f"side:{payload.side.value}",
            f"pull_ratio:{payload.pull_ratio:.4f}",
            f"fill_ratio:{payload.fill_ratio:.4f}",
            f"price_reaction_bps:{payload.price_reaction_bps:.4f}",
        ]

        if is_order_pull_signal(snapshot.raw_signal):
            score += self.reversal_config.order_pull_bonus
            confirmations.append("order_pull_detector_context")

        if is_pressure_bluff_signal(snapshot.raw_signal):
            score += self.reversal_config.pressure_bluff_bonus
            confirmations.append("pressure_bluff_detector_context")

        if is_layering_signal(snapshot.raw_signal):
            score += self.reversal_config.layering_bonus
            confirmations.append("layering_detector_context")

        if is_composite_signal(snapshot.raw_signal):
            score += self.reversal_config.composite_bonus
            confirmations.append("composite_spoofing_context")

        if payload.pull_ratio >= max(self.reversal_config.min_pull_ratio * 1.25, self.reversal_config.min_pull_ratio):
            score += self.reversal_config.high_pull_bonus
            confirmations.append("strong_pull_ratio")

        if reaction_aligns_with_side(
            signed_reaction_bps=snapshot.signed_price_reaction_bps,
            side=payload.side,
            min_reaction_bps=self.reversal_config.min_price_reaction_bps,
        ):
            score += self.reversal_config.directional_reaction_bonus
            confidence += min(0.03, self.reversal_config.directional_reaction_bonus)
            confirmations.append("directional_reaction_alignment")

        if snapshot.lifetime_ms > 0:
            reasons.append(f"lifetime_ms:{snapshot.lifetime_ms:.2f}")

        if snapshot.distance_from_mid_bps > 0:
            reasons.append(f"distance_from_mid_bps:{snapshot.distance_from_mid_bps:.4f}")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    def _context_component(
        self,
        snapshot: SpoofingCompositeSnapshot,
    ) -> float:
        components = {
            "notional": unit_score(
                snapshot.pulled_notional / max(snapshot.wall_notional, 1.0)
            )
            if snapshot.wall_notional > 0
            else 0.0,
            "cancel_to_fill": snapshot.cancel_to_fill_ratio,
            "detectors": average_score(
                snapshot.detector_agreement_ratio,
                snapshot.detector_average_confidence,
            ),
            "pressure": snapshot.pressure_flip_strength,
            "layers": unit_score(snapshot.layer_count / 5.0),
        }

        return weighted_score(
            components,
            {
                "notional": 0.25,
                "cancel_to_fill": 0.20,
                "detectors": 0.30,
                "pressure": 0.15,
                "layers": 0.10,
            },
        )

    # ------------------------------------------------------------------
    # Source features / tags / metadata hints
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: SpoofingReversalPayload,
    ) -> list[str]:
        features = [
            *base_spoofing_source_features(),
            SPOOFING_FEATURES.SIGNAL,
            SPOOFING_FEATURES.SPOOFING_TYPE,
            SPOOFING_FEATURES.PATTERN,
            SPOOFING_FEATURES.SIDE,
            SPOOFING_FEATURES.SCORE,
            SPOOFING_FEATURES.CONFIDENCE,
            SPOOFING_FEATURES.PULL_RATIO,
            SPOOFING_FEATURES.FILL_RATIO,
            SPOOFING_FEATURES.PRICE_REACTION_BPS,
            SPOOFING_FEATURES.WALL_NOTIONAL,
            SPOOFING_FEATURES.PULLED_NOTIONAL,
            SPOOFING_FEATURES.DETECTOR_RESULTS,
            SPOOFING_FEATURES.SCORE_BREAKDOWN,
        ]

        if is_order_pull_signal(payload.snapshot.raw_signal):
            features.extend(order_pull_source_features())

        if is_pressure_bluff_signal(payload.snapshot.raw_signal):
            features.extend(pressure_bluff_source_features())

        if is_layering_signal(payload.snapshot.raw_signal):
            features.extend(layering_source_features())

        if is_composite_signal(payload.snapshot.raw_signal):
            features.extend(composite_spoofing_source_features())

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: SpoofingReversalPayload,
    ) -> list[str]:
        tags = [
            self.reversal_config.tag_spoofing,
            self.reversal_config.tag_reversal,
            self.reversal_config.tag_spoofing_reversal,
        ]

        if is_order_pull_signal(payload.snapshot.raw_signal):
            tags.extend(
                [
                    self.reversal_config.tag_order_pull,
                    self.reversal_config.tag_order_pull_reversal,
                ]
            )

        if is_pressure_bluff_signal(payload.snapshot.raw_signal):
            tags.extend(
                [
                    self.reversal_config.tag_pressure_bluff,
                    self.reversal_config.tag_pressure_bluff_reversal,
                ]
            )

        if is_layering_signal(payload.snapshot.raw_signal):
            tags.extend(
                [
                    self.reversal_config.tag_layering,
                    self.reversal_config.tag_layering_reversal,
                ]
            )

        if is_composite_signal(payload.snapshot.raw_signal):
            tags.extend(
                [
                    self.reversal_config.tag_composite,
                    self.reversal_config.tag_composite_reversal,
                ]
            )

        tags.append(f"side:{payload.side.value}")

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
            "entry_offset_bps": self.reversal_config.execution_entry_offset_bps_hint,
            "stop_buffer_bps": self.reversal_config.execution_stop_buffer_bps_hint,
            "take_profit_bps": self.reversal_config.execution_take_profit_bps_hint,
            "reaction_tp_multiplier": self.reversal_config.reaction_tp_multiplier_hint,
        }