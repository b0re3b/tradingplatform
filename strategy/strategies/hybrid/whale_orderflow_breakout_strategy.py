# trading_system/strategy/strategies/hybrid/whale_orderflow_breakout_strategy.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler
from .base import (
    HYBRID_FEATURES,
    HybridCompositeSnapshot,
    HybridStrategyConfig,
    HybridTradingStrategy,
)
from .utils import (
    DirectionVote,
    HybridScoreBreakdown,
    average_score,
    confidence_from_components,
    conflicting_source_names,
    extract_domain_score,
    get_path,
    hybrid_freshness_score,
    is_directional_side,
    is_stale,
    latest_timestamp_from_payloads,
    serialize_for_metadata,
    side_to_signal_side,
    unit_score,
    votes_against_side,
    votes_for_side,
    weighted_score,
    whale_orderflow_breakout_source_features,
)
from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    FeatureSource,
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
class WhaleOrderflowBreakoutPayload:
    """
    Normalized strategy-level payload для whale + orderflow breakout.

    Direction convention:
    - whale buy activity/pressure + buy orderflow continuation -> LONG;
    - whale sell activity/pressure + sell orderflow continuation -> SHORT.
    """

    snapshot: HybridCompositeSnapshot
    side: SignalSide

    whale_side: SignalSide = SignalSide.UNKNOWN
    orderflow_side: SignalSide = SignalSide.UNKNOWN
    breakout_side: SignalSide = SignalSide.UNKNOWN

    aligned_votes: list[DirectionVote] = field(default_factory=list)
    conflicting_votes: list[DirectionVote] = field(default_factory=list)

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WhaleOrderflowBreakoutStrategyConfig(HybridStrategyConfig):
    """
    Hybrid whale + orderflow breakout config.

    Strategy idea:
    - whales domain identifies large-player activity / pressure / large trades;
    - orderflow domain confirms aggressive same-side continuation;
    - optional price action confirms breakout / breakdown;
    - strategy returns internal StrategySignal only.
    """

    min_whale_orderflow_score: float = 0.64
    min_whale_orderflow_confidence: float = 0.58

    min_whale_score: float = 0.60
    min_orderflow_score: float = 0.58
    min_breakout_score: float = 0.58
    min_large_trade_score: float = 0.50
    min_pressure_score: float = 0.50
    min_price_action_score: float = 0.50

    require_whales: bool = True
    require_orderflow: bool = True
    require_price_action_confirmation: bool = False

    require_whale_side: bool = True
    require_orderflow_side: bool = True
    require_whale_orderflow_same_side: bool = True
    require_price_action_same_side: bool = False

    use_large_trade_confirmation: bool = True
    use_whale_pressure_confirmation: bool = True
    use_price_action_confirmation: bool = True

    reject_high_conflict: bool = True
    reject_opposite_price_action: bool = True

    max_conflict_score: float = 0.38
    min_alignment_score: float = 0.54
    min_confluence_score: float = 0.52

    whales_vote_weight: float = 1.25
    orderflow_vote_weight: float = 1.20
    price_action_vote_weight: float = 0.90

    score_whales_weight: float = 0.30
    score_orderflow_weight: float = 0.30
    score_large_trade_weight: float = 0.12
    score_pressure_weight: float = 0.10
    score_price_action_weight: float = 0.08
    score_alignment_weight: float = 0.05
    score_freshness_weight: float = 0.05

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    strong_whale_bonus: float = 0.05
    strong_orderflow_bonus: float = 0.05
    large_trade_confirmation_bonus: float = 0.04
    pressure_confirmation_bonus: float = 0.03
    price_action_confirmation_bonus: float = 0.03
    same_side_flow_bonus: float = 0.05
    low_conflict_bonus: float = 0.03

    strong_whale_threshold: float = 0.72
    strong_orderflow_threshold: float = 0.72
    strong_large_trade_threshold: float = 0.68
    strong_pressure_threshold: float = 0.68
    strong_price_action_threshold: float = 0.68
    low_conflict_threshold: float = 0.15

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.BREAKOUT

    tag_whale_orderflow: str = "whale_orderflow"
    tag_whale_orderflow_breakout: str = "whale_orderflow_breakout"
    tag_whale_activity: str = "whale_activity"
    tag_orderflow_continuation: str = "orderflow_continuation"
    tag_large_trade: str = "large_trade"
    tag_whale_pressure: str = "whale_pressure"
    tag_price_action_breakout: str = "price_action_breakout"

    execution_entry_offset_bps_hint: float | None = None
    execution_stop_buffer_bps_hint: float | None = None
    execution_take_profit_bps_hint: float | None = None
    whale_orderflow_rr_hint: float | None = None

    required_hybrid_features: tuple[str, ...] = (
        HYBRID_FEATURES.DOMINANT_SIDE,
        HYBRID_FEATURES.ALIGNMENT_SCORE,
        HYBRID_FEATURES.CONFLUENCE_SCORE,
    )

    def validate(self) -> None:
        HybridStrategyConfig.validate(self)

        unit_fields = {
            "min_whale_orderflow_score": self.min_whale_orderflow_score,
            "min_whale_orderflow_confidence": self.min_whale_orderflow_confidence,
            "min_whale_score": self.min_whale_score,
            "min_orderflow_score": self.min_orderflow_score,
            "min_breakout_score": self.min_breakout_score,
            "min_large_trade_score": self.min_large_trade_score,
            "min_pressure_score": self.min_pressure_score,
            "min_price_action_score": self.min_price_action_score,
            "max_conflict_score": self.max_conflict_score,
            "min_alignment_score": self.min_alignment_score,
            "min_confluence_score": self.min_confluence_score,
            "strong_whale_bonus": self.strong_whale_bonus,
            "strong_orderflow_bonus": self.strong_orderflow_bonus,
            "large_trade_confirmation_bonus": self.large_trade_confirmation_bonus,
            "pressure_confirmation_bonus": self.pressure_confirmation_bonus,
            "price_action_confirmation_bonus": self.price_action_confirmation_bonus,
            "same_side_flow_bonus": self.same_side_flow_bonus,
            "low_conflict_bonus": self.low_conflict_bonus,
            "strong_whale_threshold": self.strong_whale_threshold,
            "strong_orderflow_threshold": self.strong_orderflow_threshold,
            "strong_large_trade_threshold": self.strong_large_trade_threshold,
            "strong_pressure_threshold": self.strong_pressure_threshold,
            "strong_price_action_threshold": self.strong_price_action_threshold,
            "low_conflict_threshold": self.low_conflict_threshold,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        vote_weights = {
            "whales_vote_weight": self.whales_vote_weight,
            "orderflow_vote_weight": self.orderflow_vote_weight,
            "price_action_vote_weight": self.price_action_vote_weight,
        }
        score_weights = {
            "score_whales_weight": self.score_whales_weight,
            "score_orderflow_weight": self.score_orderflow_weight,
            "score_large_trade_weight": self.score_large_trade_weight,
            "score_pressure_weight": self.score_pressure_weight,
            "score_price_action_weight": self.score_price_action_weight,
            "score_alignment_weight": self.score_alignment_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }
        confidence_weights = {
            "confidence_primary_weight": self.confidence_primary_weight,
            "confidence_context_weight": self.confidence_context_weight,
            "confidence_confirmation_weight": self.confidence_confirmation_weight,
            "confidence_freshness_weight": self.confidence_freshness_weight,
        }

        for field_name, value in {
            **vote_weights,
            **score_weights,
            **confidence_weights,
        }.items():
            if float(value) < 0.0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if sum(score_weights.values()) <= 0:
            raise StrategyConfigError("score weights sum must be > 0")

        if sum(confidence_weights.values()) <= 0:
            raise StrategyConfigError("confidence weights sum must be > 0")

        hint_fields = {
            "execution_entry_offset_bps_hint": self.execution_entry_offset_bps_hint,
            "execution_stop_buffer_bps_hint": self.execution_stop_buffer_bps_hint,
            "execution_take_profit_bps_hint": self.execution_take_profit_bps_hint,
            "whale_orderflow_rr_hint": self.whale_orderflow_rr_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        for attr in (
            "tag_whale_orderflow",
            "tag_whale_orderflow_breakout",
            "tag_whale_activity",
            "tag_orderflow_continuation",
            "tag_large_trade",
            "tag_whale_pressure",
            "tag_price_action_breakout",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


class WhaleOrderflowBreakoutStrategy(HybridTradingStrategy):
    """
    Hybrid whale + orderflow breakout / continuation strategy.

    Input:
        StrategyContext with FeatureSource.WHALES and FeatureSource.ORDERFLOW,
        optionally FeatureSource.PRICE_ACTION.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    It does not duplicate SignalProcessor.ConfluenceEngine.
    SignalProcessor owns global routing, confluence, filters, portfolio coordination,
    building and risk payloads.
    """

    component_namespace = "strategy.hybrid.whale_orderflow_breakout"
    category: StrategyCategory = StrategyCategory.HYBRID
    default_setup_type: SetupType = SetupType.BREAKOUT

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        hybrid_config: WhaleOrderflowBreakoutStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_hybrid_config = (
            hybrid_config or WhaleOrderflowBreakoutStrategyConfig()
        )
        resolved_hybrid_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            hybrid_config=resolved_hybrid_config,
            service_name=service_name,
        )

        self.wo_config: WhaleOrderflowBreakoutStrategyConfig = (
            resolved_hybrid_config
        )

    @property
    def strategy_name(self) -> str:
        return "whale_orderflow_breakout"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.HYBRID,
            timeframe=Timeframe.M1,
            tags=[
                self.wo_config.tag_hybrid,
                self.wo_config.tag_whale_orderflow,
                self.wo_config.tag_whale_orderflow_breakout,
                self.wo_config.tag_whale_activity,
                self.wo_config.tag_orderflow_continuation,
                "strategy_context",
            ],
            version="2.0.0",
            description=(
                "Builds a specialized whale + orderflow breakout signal from "
                "same-side whale activity and aggressive orderflow continuation."
            ),
            required_features=set(self.required_features()),
            supported_regimes={
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.BREAKOUT,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.SQUEEZE,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "strategy_context.domains",
                "strategy_type": "whale_orderflow_breakout",
                "base_class": "HybridTradingStrategy",
                "canonical_payload": "HybridCompositeSnapshot",
                "uses_whales": True,
                "uses_orderflow": True,
                "uses_price_action_confirmation": True,
                "requires_whales_orderflow": True,
                "duplicates_signal_processor_confluence": False,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(self.wo_config.required_hybrid_features)

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        sources = self._enabled_sources()
        required_sources = self._required_sources()

        required_domains_available = self.required_domains_available(
            context,
            required_sources,
            allow_missing=self.wo_config.allow_missing_required_domains,
        )
        if not required_domains_available:
            return None

        snapshot = self.resolve_hybrid_snapshot(
            context,
            sources=sources,
            vote_weights=self._vote_weights(),
        )
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        payload = self._extract_payload(
            context=context,
            snapshot=snapshot,
            sources=sources,
            required_sources=required_sources,
        )
        if payload is None:
            return None

        if is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.wo_config.stale_feature_max_age_seconds,
        ):
            return None

        if not self._passes_whale_orderflow_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.wo_config.min_whale_orderflow_score:
            return None

        if breakdown.confidence < self.wo_config.min_whale_orderflow_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "whale_orderflow_breakout_signal",
                    f"side:{payload.side.value}",
                    f"whale_side:{payload.whale_side.value}",
                    f"orderflow_side:{payload.orderflow_side.value}",
                    f"breakout_side:{payload.breakout_side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "hybrid_setup_family": "whale_orderflow_breakout",
            "hybrid_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "snapshot": serialize_for_metadata(payload.snapshot.to_dict()),
            "votes": [vote.to_dict() for vote in payload.snapshot.votes],
            "aligned_votes": [vote.to_dict() for vote in payload.aligned_votes],
            "conflicting_votes": [vote.to_dict() for vote in payload.conflicting_votes],
            "raw": serialize_for_metadata(payload.raw),
            "event_time": payload.event_time.isoformat() if payload.event_time else None,
            "mapped_side": payload.side.value,
            "whale_side": payload.whale_side.value,
            "orderflow_side": payload.orderflow_side.value,
            "breakout_side": payload.breakout_side.value,
            "alignment_score": payload.snapshot.alignment_score,
            "conflict_score": payload.snapshot.conflict_score,
            "confluence_score": payload.snapshot.confluence_score,
            "confidence": payload.snapshot.confidence,
            "available_domains": payload.snapshot.available_domains,
            "aligned_domains": payload.snapshot.aligned_domains,
            "conflicting_domains": payload.snapshot.conflicting_domains,
            "required_domains": [source.value for source in required_sources],
            "enabled_domains": [source.value for source in sources],
            "execution_hints": self._execution_hints(),
        }

        return self.build_hybrid_signal(
            context=context,
            side=payload.side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.wo_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.wo_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Sources / weights
    # ------------------------------------------------------------------

    def _enabled_sources(self) -> tuple[FeatureSource, ...]:
        sources: list[FeatureSource] = [
            FeatureSource.WHALES,
            FeatureSource.ORDERFLOW,
        ]

        if self.wo_config.use_price_action_confirmation:
            sources.append(FeatureSource.PRICE_ACTION)

        return tuple(dict.fromkeys(sources))

    def _required_sources(self) -> tuple[FeatureSource, ...]:
        sources: list[FeatureSource] = []

        if self.wo_config.require_whales:
            sources.append(FeatureSource.WHALES)

        if self.wo_config.require_orderflow:
            sources.append(FeatureSource.ORDERFLOW)

        if self.wo_config.require_price_action_confirmation:
            sources.append(FeatureSource.PRICE_ACTION)

        return tuple(dict.fromkeys(sources))

    def _vote_weights(self) -> dict[FeatureSource, float]:
        return {
            FeatureSource.WHALES: self.wo_config.whales_vote_weight,
            FeatureSource.ORDERFLOW: self.wo_config.orderflow_vote_weight,
            FeatureSource.PRICE_ACTION: self.wo_config.price_action_vote_weight,
        }

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        *,
        context: StrategyContext,
        snapshot: HybridCompositeSnapshot,
        sources: tuple[FeatureSource, ...],
        required_sources: tuple[FeatureSource, ...],
    ) -> WhaleOrderflowBreakoutPayload | None:
        payloads = snapshot.payloads()

        whales = payloads.get(FeatureSource.WHALES, {})
        orderflow = payloads.get(FeatureSource.ORDERFLOW, {})
        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})

        if not whales or not orderflow:
            return None

        whale_side = self._extract_whale_side(whales)
        if self.wo_config.require_whale_side and not is_directional_side(whale_side):
            return None

        orderflow_side = self._extract_orderflow_side(orderflow)
        if self.wo_config.require_orderflow_side and not is_directional_side(orderflow_side):
            return None

        breakout_side = self._extract_breakout_side(price_action)

        side = whale_side
        if not is_directional_side(side):
            side = orderflow_side

        if not is_directional_side(side):
            side = snapshot.dominant_side

        if not is_directional_side(side):
            return None

        aligned_votes = votes_for_side(snapshot.votes, side)
        conflicting_votes = votes_against_side(snapshot.votes, side)

        event_time = (
            latest_timestamp_from_payloads(payloads, fallback=context.timestamp)
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "whale_orderflow_breakout_context",
            f"side:{side.value}",
            f"whale_side:{whale_side.value}",
            f"orderflow_side:{orderflow_side.value}",
            f"breakout_side:{breakout_side.value}",
            f"alignment_score:{snapshot.alignment_score:.4f}",
            f"conflict_score:{snapshot.conflict_score:.4f}",
            f"confluence_score:{snapshot.confluence_score:.4f}",
        ]

        return WhaleOrderflowBreakoutPayload(
            snapshot=snapshot,
            side=side,
            whale_side=whale_side,
            orderflow_side=orderflow_side,
            breakout_side=breakout_side,
            aligned_votes=aligned_votes,
            conflicting_votes=conflicting_votes,
            event_time=event_time,
            reasons=list(dict.fromkeys(reasons)),
            raw={
                "payloads": payloads,
                "sources": [source.value for source in sources],
                "required_sources": [source.value for source in required_sources],
            },
        )

    def _extract_whale_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "whale_side",
            "activity_side",
            "pressure_side",
            "dominant_side",
            "cluster_side",
            "breakout_side",
            "continuation_side",
            "signal_side",
            "side",
            "direction",
            "metadata.whale_side",
            "metadata.side",
        ):
            side = side_to_signal_side(get_path(payload, path))
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    def _extract_orderflow_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "continuation_side",
            "delta_side",
            "pressure_side",
            "aggression_side",
            "dominant_side",
            "signal_side",
            "side",
            "direction",
            "bias",
            "metadata.continuation_side",
            "metadata.side",
        ):
            side = side_to_signal_side(get_path(payload, path))
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    def _extract_breakout_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "breakout_side",
            "breakdown_side",
            "continuation_side",
            "trend_side",
            "signal_side",
            "side",
            "direction",
            "bias",
            "metadata.breakout_side",
            "metadata.side",
        ):
            side = side_to_signal_side(get_path(payload, path))
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_whale_orderflow_filters(
        self,
        payload: WhaleOrderflowBreakoutPayload,
    ) -> bool:
        snapshot = payload.snapshot
        payloads = snapshot.payloads()

        whales = payloads.get(FeatureSource.WHALES, {})
        orderflow = payloads.get(FeatureSource.ORDERFLOW, {})
        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})

        if self.wo_config.reject_high_conflict:
            if snapshot.conflict_score > self.wo_config.max_conflict_score:
                return False

        if snapshot.alignment_score < self.wo_config.min_alignment_score:
            return False

        if snapshot.confluence_score < self.wo_config.min_confluence_score:
            return False

        if extract_domain_score(whales) < self.wo_config.min_whale_score:
            return False

        if extract_domain_score(orderflow) < self.wo_config.min_orderflow_score:
            return False

        if self.wo_config.require_whale_orderflow_same_side:
            if payload.whale_side != payload.orderflow_side:
                return False

            if payload.whale_side != payload.side:
                return False

        if self.wo_config.use_large_trade_confirmation:
            large_trade_score = self._large_trade_score(whales)
            if large_trade_score > 0.0 and large_trade_score < self.wo_config.min_large_trade_score:
                return False

        if self.wo_config.use_whale_pressure_confirmation:
            pressure_score = self._whale_pressure_score(whales)
            if pressure_score > 0.0 and pressure_score < self.wo_config.min_pressure_score:
                return False

        if self.wo_config.use_price_action_confirmation and price_action:
            price_action_score = extract_domain_score(price_action)
            if price_action_score > 0.0 and price_action_score < self.wo_config.min_price_action_score:
                return False

            if self.wo_config.require_price_action_same_side:
                if not is_directional_side(payload.breakout_side):
                    return False

                if payload.breakout_side != payload.side:
                    return False

            if self.wo_config.reject_opposite_price_action:
                if is_directional_side(payload.breakout_side):
                    if payload.breakout_side != payload.side:
                        return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: WhaleOrderflowBreakoutPayload,
    ) -> HybridScoreBreakdown:
        snapshot = payload.snapshot
        payloads = snapshot.payloads()

        whales = payloads.get(FeatureSource.WHALES, {})
        orderflow = payloads.get(FeatureSource.ORDERFLOW, {})
        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})

        whales_component = extract_domain_score(whales)
        orderflow_component = extract_domain_score(orderflow)
        large_trade_component = (
            self._large_trade_score(whales)
            if self.wo_config.use_large_trade_confirmation
            else 0.0
        )
        pressure_component = (
            self._whale_pressure_score(whales)
            if self.wo_config.use_whale_pressure_confirmation
            else 0.0
        )
        price_action_component = (
            extract_domain_score(price_action)
            if price_action and self.wo_config.use_price_action_confirmation
            else 0.0
        )
        alignment_component = snapshot.alignment_score
        freshness_component = hybrid_freshness_score(
            payloads,
            now=context.timestamp,
            stale_after_seconds=self.wo_config.stale_feature_max_age_seconds,
        )

        components = {
            "whales": whales_component,
            "orderflow": orderflow_component,
            "large_trade": large_trade_component,
            "pressure": pressure_component,
            "price_action": price_action_component,
            "alignment": alignment_component,
            "freshness": freshness_component,
        }
        weights = {
            "whales": self.wo_config.score_whales_weight,
            "orderflow": self.wo_config.score_orderflow_weight,
            "large_trade": self.wo_config.score_large_trade_weight,
            "pressure": self.wo_config.score_pressure_weight,
            "price_action": self.wo_config.score_price_action_weight,
            "alignment": self.wo_config.score_alignment_weight,
            "freshness": self.wo_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=whales_component)
        confidence = confidence_from_components(
            primary=average_score(whales_component, orderflow_component),
            context=average_score(large_trade_component, pressure_component),
            confirmation=average_score(price_action_component, 1.0 - snapshot.conflict_score),
            freshness=freshness_component,
            primary_weight=self.wo_config.confidence_primary_weight,
            context_weight=self.wo_config.confidence_context_weight,
            confirmation_weight=self.wo_config.confidence_confirmation_weight,
            freshness_weight=self.wo_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "whale_orderflow_breakout_context",
            f"side:{payload.side.value}",
            f"whale_side:{payload.whale_side.value}",
            f"orderflow_side:{payload.orderflow_side.value}",
        ]

        conflicts = conflicting_source_names(snapshot.votes, payload.side)

        if whales_component >= self.wo_config.strong_whale_threshold:
            score += self.wo_config.strong_whale_bonus
            confirmations.append("strong_whale_context")

        if orderflow_component >= self.wo_config.strong_orderflow_threshold:
            score += self.wo_config.strong_orderflow_bonus
            confirmations.append("strong_orderflow_continuation")

        if large_trade_component >= self.wo_config.strong_large_trade_threshold:
            score += self.wo_config.large_trade_confirmation_bonus
            confirmations.append("large_trade_confirmation")

        if pressure_component >= self.wo_config.strong_pressure_threshold:
            score += self.wo_config.pressure_confirmation_bonus
            confirmations.append("whale_pressure_confirmation")

        if price_action_component >= self.wo_config.strong_price_action_threshold:
            score += self.wo_config.price_action_confirmation_bonus
            confirmations.append("price_action_breakout_confirmation")

        if (
            is_directional_side(payload.whale_side)
            and is_directional_side(payload.orderflow_side)
            and payload.whale_side == payload.orderflow_side
            and payload.whale_side == payload.side
        ):
            score += self.wo_config.same_side_flow_bonus
            confirmations.append("whale_orderflow_same_side")

        if snapshot.conflict_score <= self.wo_config.low_conflict_threshold:
            score += self.wo_config.low_conflict_bonus
            confirmations.append("low_domain_conflict")

        if is_directional_side(payload.breakout_side) and payload.breakout_side == payload.side:
            confirmations.append("price_action_same_as_breakout_side")

        reasons.extend(
            [
                f"whales_score:{whales_component:.4f}",
                f"orderflow_score:{orderflow_component:.4f}",
                f"large_trade_score:{large_trade_component:.4f}",
                f"pressure_score:{pressure_component:.4f}",
                f"price_action_score:{price_action_component:.4f}",
                f"alignment_score:{snapshot.alignment_score:.4f}",
                f"conflict_score:{snapshot.conflict_score:.4f}",
                f"confluence_score:{snapshot.confluence_score:.4f}",
            ]
        )

        if conflicts:
            reasons.append(f"conflicts:{','.join(conflicts)}")

        return HybridScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            votes=list(snapshot.votes),
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
            conflicts=conflicts,
        ).normalize()

    def _large_trade_score(
        self,
        whales: dict[str, Any],
    ) -> float:
        candidates = [
            get_path(whales, "large_trade_score"),
            get_path(whales, "large_trade.zscore_score"),
            get_path(whales, "large_trade.score"),
            get_path(whales, "large_trade_notional_score"),
            get_path(whales, "activity_score"),
            get_path(whales, "metadata.large_trade_score"),
        ]

        for value in candidates:
            if value is not None:
                return unit_score(value)

        return 0.0

    def _whale_pressure_score(
        self,
        whales: dict[str, Any],
    ) -> float:
        candidates = [
            get_path(whales, "pressure_score"),
            get_path(whales, "whale_pressure_score"),
            get_path(whales, "imbalance_score"),
            get_path(whales, "imbalance_ratio"),
            get_path(whales, "context_strength"),
            get_path(whales, "metadata.pressure_score"),
            get_path(whales, "metadata.imbalance_ratio"),
        ]

        for value in candidates:
            if value is not None:
                return unit_score(value)

        return 0.0

    # ------------------------------------------------------------------
    # Source features / tags / metadata helpers
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: WhaleOrderflowBreakoutPayload,
    ) -> list[str]:
        features = [
            *whale_orderflow_breakout_source_features(),
            HYBRID_FEATURES.DOMINANT_SIDE,
            HYBRID_FEATURES.ALIGNMENT_SCORE,
            HYBRID_FEATURES.CONFLICT_SCORE,
            HYBRID_FEATURES.CONFLUENCE_SCORE,
            HYBRID_FEATURES.CONFIDENCE,
            HYBRID_FEATURES.VOTES,
            "whales.*",
            "orderflow.*",
            "price_action.*",
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: WhaleOrderflowBreakoutPayload,
    ) -> list[str]:
        tags = [
            self.wo_config.tag_hybrid,
            self.wo_config.tag_whale_orderflow,
            self.wo_config.tag_whale_orderflow_breakout,
            self.wo_config.tag_whale_activity,
            self.wo_config.tag_orderflow_continuation,
            f"side:{payload.side.value}",
            f"whale_side:{payload.whale_side.value}",
            f"orderflow_side:{payload.orderflow_side.value}",
        ]

        whales = payload.snapshot.payloads().get(FeatureSource.WHALES, {})
        price_action = payload.snapshot.payloads().get(FeatureSource.PRICE_ACTION, {})

        if self._large_trade_score(whales) > 0.0:
            tags.append(self.wo_config.tag_large_trade)

        if self._whale_pressure_score(whales) > 0.0:
            tags.append(self.wo_config.tag_whale_pressure)

        if price_action and is_directional_side(payload.breakout_side):
            tags.append(self.wo_config.tag_price_action_breakout)

        for source in payload.snapshot.aligned_domains:
            tags.append(f"aligned:{source}")

        return list(dict.fromkeys(tags))

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        return {
            "entry_offset_bps": self.wo_config.execution_entry_offset_bps_hint,
            "stop_buffer_bps": self.wo_config.execution_stop_buffer_bps_hint,
            "take_profit_bps": self.wo_config.execution_take_profit_bps_hint,
            "risk_reward": self.wo_config.whale_orderflow_rr_hint,
        }