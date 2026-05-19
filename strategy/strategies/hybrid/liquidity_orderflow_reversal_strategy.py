# trading_system/strategy/strategies/hybrid/liquidity_orderflow_reversal_strategy.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler

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
from .base import (
    HYBRID_FEATURES,
    HybridCompositeSnapshot,
    HybridStrategyConfig,
    HybridTradingStrategy,
)
from .utils import (
    DirectionVote,
    HybridScoreBreakdown,
    LIQUIDITY_ORDERFLOW_REVERSAL_SOURCES,
    average_score,
    confidence_from_components,
    conflicting_source_names,
    extract_domain_score,
    get_path,
    hybrid_freshness_score,
    is_directional_side,
    is_stale,
    latest_timestamp_from_payloads,
    liquidity_orderflow_reversal_source_features,
    normalize_label,
    opposite_signal_side,
    serialize_for_metadata,
    side_to_signal_side,
    unit_score,
    votes_against_side,
    votes_for_side,
    weighted_score,
)


@dataclass(slots=True)
class LiquidityOrderflowReversalPayload:
    """
    Normalized strategy-level payload для liquidity + orderflow reversal.

    Direction convention:
    - sweep / stop-hunt down + sell exhaustion / absorption -> LONG;
    - sweep / stop-hunt up + buy exhaustion / absorption -> SHORT.
    """

    snapshot: HybridCompositeSnapshot
    side: SignalSide

    sweep_side: SignalSide = SignalSide.UNKNOWN
    orderflow_exhaustion_side: SignalSide = SignalSide.UNKNOWN
    absorption_side: SignalSide = SignalSide.UNKNOWN
    rejection_side: SignalSide = SignalSide.UNKNOWN

    aligned_votes: list[DirectionVote] = field(default_factory=list)
    conflicting_votes: list[DirectionVote] = field(default_factory=list)

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LiquidityOrderflowReversalStrategyConfig(HybridStrategyConfig):
    """
    Hybrid liquidity + orderflow reversal config.

    Strategy idea:
    - liquidity sweep / stop-hunt indicates stop-run extension;
    - orderflow exhaustion / absorption confirms fading forced flow;
    - optional price action rejection confirms reversal side;
    - strategy returns internal StrategySignal only.
    """

    min_liquidity_orderflow_score: float = 0.64
    min_liquidity_orderflow_confidence: float = 0.58

    min_liquidity_sweep_score: float = 0.60
    min_orderflow_exhaustion_score: float = 0.55
    min_absorption_score: float = 0.50
    min_rejection_score: float = 0.50

    require_liquidity: bool = True
    require_orderflow: bool = True
    require_price_action_confirmation: bool = False

    require_sweep_side: bool = True
    require_orderflow_exhaustion_side: bool = True
    require_orderflow_same_as_sweep: bool = True
    require_absorption_same_as_reversal: bool = False
    require_rejection_same_as_reversal: bool = False

    use_absorption_confirmation: bool = True
    use_price_action_confirmation: bool = True

    reject_same_side_momentum: bool = True
    reject_high_conflict: bool = True

    max_conflict_score: float = 0.38
    min_alignment_score: float = 0.52
    min_confluence_score: float = 0.50

    liquidity_vote_weight: float = 1.25
    orderflow_vote_weight: float = 1.20
    price_action_vote_weight: float = 0.90

    score_liquidity_weight: float = 0.34
    score_orderflow_weight: float = 0.32
    score_absorption_weight: float = 0.12
    score_price_action_weight: float = 0.10
    score_alignment_weight: float = 0.06
    score_freshness_weight: float = 0.06

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    strong_liquidity_bonus: float = 0.05
    strong_orderflow_bonus: float = 0.05
    absorption_confirmation_bonus: float = 0.04
    price_rejection_bonus: float = 0.04
    opposite_flow_bonus: float = 0.04
    low_conflict_bonus: float = 0.03

    strong_liquidity_threshold: float = 0.76
    strong_orderflow_threshold: float = 0.72
    strong_absorption_threshold: float = 0.68
    strong_rejection_threshold: float = 0.68
    low_conflict_threshold: float = 0.15

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.REVERSAL

    tag_liquidity_orderflow: str = "liquidity_orderflow"
    tag_liquidity_orderflow_reversal: str = "liquidity_orderflow_reversal"
    tag_liquidity_sweep: str = "liquidity_sweep"
    tag_stop_hunt: str = "stop_hunt"
    tag_orderflow_exhaustion: str = "orderflow_exhaustion"
    tag_orderflow_absorption: str = "orderflow_absorption"
    tag_price_rejection: str = "price_rejection"

    execution_entry_offset_bps_hint: float | None = None
    execution_stop_buffer_bps_hint: float | None = None
    execution_take_profit_bps_hint: float | None = None
    liquidity_orderflow_rr_hint: float | None = None

    required_hybrid_features: tuple[str, ...] = (
        HYBRID_FEATURES.DOMINANT_SIDE,
        HYBRID_FEATURES.ALIGNMENT_SCORE,
        HYBRID_FEATURES.CONFLUENCE_SCORE,
    )

    def validate(self) -> None:
        HybridStrategyConfig.validate(self)

        unit_fields = {
            "min_liquidity_orderflow_score": self.min_liquidity_orderflow_score,
            "min_liquidity_orderflow_confidence": self.min_liquidity_orderflow_confidence,
            "min_liquidity_sweep_score": self.min_liquidity_sweep_score,
            "min_orderflow_exhaustion_score": self.min_orderflow_exhaustion_score,
            "min_absorption_score": self.min_absorption_score,
            "min_rejection_score": self.min_rejection_score,
            "max_conflict_score": self.max_conflict_score,
            "min_alignment_score": self.min_alignment_score,
            "min_confluence_score": self.min_confluence_score,
            "strong_liquidity_bonus": self.strong_liquidity_bonus,
            "strong_orderflow_bonus": self.strong_orderflow_bonus,
            "absorption_confirmation_bonus": self.absorption_confirmation_bonus,
            "price_rejection_bonus": self.price_rejection_bonus,
            "opposite_flow_bonus": self.opposite_flow_bonus,
            "low_conflict_bonus": self.low_conflict_bonus,
            "strong_liquidity_threshold": self.strong_liquidity_threshold,
            "strong_orderflow_threshold": self.strong_orderflow_threshold,
            "strong_absorption_threshold": self.strong_absorption_threshold,
            "strong_rejection_threshold": self.strong_rejection_threshold,
            "low_conflict_threshold": self.low_conflict_threshold,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        vote_weights = {
            "liquidity_vote_weight": self.liquidity_vote_weight,
            "orderflow_vote_weight": self.orderflow_vote_weight,
            "price_action_vote_weight": self.price_action_vote_weight,
        }
        score_weights = {
            "score_liquidity_weight": self.score_liquidity_weight,
            "score_orderflow_weight": self.score_orderflow_weight,
            "score_absorption_weight": self.score_absorption_weight,
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
            "liquidity_orderflow_rr_hint": self.liquidity_orderflow_rr_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        for attr in (
            "tag_liquidity_orderflow",
            "tag_liquidity_orderflow_reversal",
            "tag_liquidity_sweep",
            "tag_stop_hunt",
            "tag_orderflow_exhaustion",
            "tag_orderflow_absorption",
            "tag_price_rejection",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


class LiquidityOrderflowReversalStrategy(HybridTradingStrategy):
    """
    Hybrid liquidity sweep + orderflow exhaustion reversal strategy.

    Input:
        StrategyContext with FeatureSource.LIQUIDITY and FeatureSource.ORDERFLOW,
        optionally FeatureSource.PRICE_ACTION.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    It does not duplicate SignalProcessor.ConfluenceEngine.
    SignalProcessor owns global routing, confluence, filters, portfolio coordination,
    building and risk payloads.
    """

    component_namespace = "strategy.hybrid.liquidity_orderflow_reversal"
    category: StrategyCategory = StrategyCategory.HYBRID
    default_setup_type: SetupType = SetupType.REVERSAL

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        hybrid_config: LiquidityOrderflowReversalStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_hybrid_config = (
            hybrid_config or LiquidityOrderflowReversalStrategyConfig()
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

        self.lo_config: LiquidityOrderflowReversalStrategyConfig = (
            resolved_hybrid_config
        )

    @property
    def strategy_name(self) -> str:
        return "liquidity_orderflow_reversal"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.HYBRID,
            timeframe=Timeframe.M1,
            tags=[
                self.lo_config.tag_hybrid,
                self.lo_config.tag_liquidity_orderflow,
                self.lo_config.tag_liquidity_orderflow_reversal,
                self.lo_config.tag_liquidity_sweep,
                self.lo_config.tag_orderflow_exhaustion,
                "strategy_context",
            ],
            version="2.0.0",
            description=(
                "Builds a specialized liquidity sweep + orderflow exhaustion "
                "reversal signal from StrategyContext."
            ),
            required_features=set(self.required_features()),
            supported_regimes={
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.SQUEEZE,
                MarketRegime.BREAKOUT,
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "strategy_context.domains",
                "strategy_type": "liquidity_orderflow_reversal",
                "base_class": "HybridTradingStrategy",
                "canonical_payload": "HybridCompositeSnapshot",
                "uses_liquidity": True,
                "uses_orderflow": True,
                "uses_price_action_confirmation": True,
                "requires_liquidity_orderflow": True,
                "duplicates_signal_processor_confluence": False,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(self.lo_config.required_hybrid_features)

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
            allow_missing=self.lo_config.allow_missing_required_domains,
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
            stale_after_seconds=self.lo_config.stale_feature_max_age_seconds,
        ):
            return None

        if not self._passes_liquidity_orderflow_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.lo_config.min_liquidity_orderflow_score:
            return None

        if breakdown.confidence < self.lo_config.min_liquidity_orderflow_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "liquidity_orderflow_reversal_signal",
                    f"side:{payload.side.value}",
                    f"sweep_side:{payload.sweep_side.value}",
                    f"orderflow_exhaustion_side:{payload.orderflow_exhaustion_side.value}",
                    f"absorption_side:{payload.absorption_side.value}",
                    f"rejection_side:{payload.rejection_side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "hybrid_setup_family": "liquidity_orderflow_reversal",
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
            "sweep_side": payload.sweep_side.value,
            "orderflow_exhaustion_side": payload.orderflow_exhaustion_side.value,
            "absorption_side": payload.absorption_side.value,
            "rejection_side": payload.rejection_side.value,
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
            setup_type=self.lo_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.lo_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Sources / weights
    # ------------------------------------------------------------------

    def _enabled_sources(self) -> tuple[FeatureSource, ...]:
        sources: list[FeatureSource] = [
            FeatureSource.LIQUIDITY,
            FeatureSource.ORDERFLOW,
        ]

        if self.lo_config.use_price_action_confirmation:
            sources.append(FeatureSource.PRICE_ACTION)

        return tuple(dict.fromkeys(sources))

    def _required_sources(self) -> tuple[FeatureSource, ...]:
        sources: list[FeatureSource] = []

        if self.lo_config.require_liquidity:
            sources.append(FeatureSource.LIQUIDITY)

        if self.lo_config.require_orderflow:
            sources.append(FeatureSource.ORDERFLOW)

        if self.lo_config.require_price_action_confirmation:
            sources.append(FeatureSource.PRICE_ACTION)

        return tuple(dict.fromkeys(sources))

    def _vote_weights(self) -> dict[FeatureSource, float]:
        return {
            FeatureSource.LIQUIDITY: self.lo_config.liquidity_vote_weight,
            FeatureSource.ORDERFLOW: self.lo_config.orderflow_vote_weight,
            FeatureSource.PRICE_ACTION: self.lo_config.price_action_vote_weight,
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
    ) -> LiquidityOrderflowReversalPayload | None:
        payloads = snapshot.payloads()

        liquidity = payloads.get(FeatureSource.LIQUIDITY, {})
        orderflow = payloads.get(FeatureSource.ORDERFLOW, {})
        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})

        if not liquidity or not orderflow:
            return None

        sweep_side = self._extract_sweep_side(liquidity)
        if self.lo_config.require_sweep_side and not is_directional_side(sweep_side):
            return None

        side = opposite_signal_side(sweep_side)
        if not is_directional_side(side):
            side = self._extract_reversal_side(orderflow, price_action)

        if not is_directional_side(side):
            return None

        orderflow_exhaustion_side = self._extract_orderflow_exhaustion_side(orderflow)
        absorption_side = self._extract_absorption_side(orderflow)
        rejection_side = self._extract_rejection_side(price_action)

        aligned_votes = votes_for_side(snapshot.votes, side)
        conflicting_votes = votes_against_side(snapshot.votes, side)

        event_time = (
            latest_timestamp_from_payloads(payloads, fallback=context.timestamp)
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "liquidity_orderflow_reversal_context",
            f"side:{side.value}",
            f"sweep_side:{sweep_side.value}",
            f"orderflow_exhaustion_side:{orderflow_exhaustion_side.value}",
            f"absorption_side:{absorption_side.value}",
            f"rejection_side:{rejection_side.value}",
            f"alignment_score:{snapshot.alignment_score:.4f}",
            f"conflict_score:{snapshot.conflict_score:.4f}",
            f"confluence_score:{snapshot.confluence_score:.4f}",
        ]

        return LiquidityOrderflowReversalPayload(
            snapshot=snapshot,
            side=side,
            sweep_side=sweep_side,
            orderflow_exhaustion_side=orderflow_exhaustion_side,
            absorption_side=absorption_side,
            rejection_side=rejection_side,
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

    def _extract_sweep_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "sweep_side",
            "stop_hunt_side",
            "liquidity_sweep_side",
            "taken_side",
            "swept_side",
            "side",
            "direction",
            "metadata.sweep_side",
            "metadata.stop_hunt_side",
            "metadata.side",
        ):
            side = side_to_signal_side(get_path(payload, path))
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    def _extract_orderflow_exhaustion_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "exhaustion_side",
            "exhausted_side",
            "pressure_side",
            "delta_side",
            "aggression_side",
            "dominant_side",
            "side",
            "direction",
            "metadata.exhaustion_side",
            "metadata.side",
        ):
            side = side_to_signal_side(get_path(payload, path))
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    def _extract_absorption_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "absorption_side",
            "absorbing_side",
            "absorber_side",
            "supporting_side",
            "reversal_side",
            "signal_side",
            "metadata.absorption_side",
            "metadata.reversal_side",
        ):
            side = side_to_signal_side(get_path(payload, path))
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    def _extract_rejection_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "rejection_side",
            "reversal_side",
            "signal_side",
            "side",
            "direction",
            "bias",
            "metadata.rejection_side",
            "metadata.side",
        ):
            side = side_to_signal_side(get_path(payload, path))
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    def _extract_reversal_side(
        self,
        orderflow: dict[str, Any],
        price_action: dict[str, Any],
    ) -> SignalSide:
        for extractor, payload in (
            (self._extract_absorption_side, orderflow),
            (self._extract_rejection_side, price_action),
        ):
            side = extractor(payload)
            if is_directional_side(side):
                return side

        exhaustion_side = self._extract_orderflow_exhaustion_side(orderflow)
        if is_directional_side(exhaustion_side):
            return opposite_signal_side(exhaustion_side)

        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_liquidity_orderflow_filters(
        self,
        payload: LiquidityOrderflowReversalPayload,
    ) -> bool:
        snapshot = payload.snapshot
        payloads = snapshot.payloads()

        liquidity = payloads.get(FeatureSource.LIQUIDITY, {})
        orderflow = payloads.get(FeatureSource.ORDERFLOW, {})
        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})

        if self.lo_config.reject_high_conflict:
            if snapshot.conflict_score > self.lo_config.max_conflict_score:
                return False

        if snapshot.alignment_score < self.lo_config.min_alignment_score:
            return False

        if snapshot.confluence_score < self.lo_config.min_confluence_score:
            return False

        if extract_domain_score(liquidity) < self.lo_config.min_liquidity_sweep_score:
            return False

        if extract_domain_score(orderflow) < self.lo_config.min_orderflow_exhaustion_score:
            return False

        if self.lo_config.require_orderflow_exhaustion_side:
            if not is_directional_side(payload.orderflow_exhaustion_side):
                return False

        if self.lo_config.require_orderflow_same_as_sweep:
            if payload.orderflow_exhaustion_side != payload.sweep_side:
                return False

        if self.lo_config.use_absorption_confirmation:
            absorption_score = self._absorption_score(orderflow)
            if absorption_score > 0.0 and absorption_score < self.lo_config.min_absorption_score:
                return False

            if self.lo_config.require_absorption_same_as_reversal:
                if not is_directional_side(payload.absorption_side):
                    return False
                if payload.absorption_side != payload.side:
                    return False

        if self.lo_config.use_price_action_confirmation and price_action:
            rejection_score = extract_domain_score(price_action)
            if rejection_score > 0.0 and rejection_score < self.lo_config.min_rejection_score:
                return False

            if self.lo_config.require_rejection_same_as_reversal:
                if not is_directional_side(payload.rejection_side):
                    return False
                if payload.rejection_side != payload.side:
                    return False

        if self.lo_config.reject_same_side_momentum:
            if payload.sweep_side == payload.side:
                return False

            if is_directional_side(payload.orderflow_exhaustion_side):
                if payload.orderflow_exhaustion_side == payload.side:
                    return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: LiquidityOrderflowReversalPayload,
    ) -> HybridScoreBreakdown:
        snapshot = payload.snapshot
        payloads = snapshot.payloads()

        liquidity = payloads.get(FeatureSource.LIQUIDITY, {})
        orderflow = payloads.get(FeatureSource.ORDERFLOW, {})
        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})

        liquidity_component = extract_domain_score(liquidity)
        orderflow_component = extract_domain_score(orderflow)
        absorption_component = (
            self._absorption_score(orderflow)
            if self.lo_config.use_absorption_confirmation
            else 0.0
        )
        price_action_component = (
            extract_domain_score(price_action)
            if price_action and self.lo_config.use_price_action_confirmation
            else 0.0
        )
        alignment_component = snapshot.alignment_score
        freshness_component = hybrid_freshness_score(
            payloads,
            now=context.timestamp,
            stale_after_seconds=self.lo_config.stale_feature_max_age_seconds,
        )

        components = {
            "liquidity": liquidity_component,
            "orderflow": orderflow_component,
            "absorption": absorption_component,
            "price_action": price_action_component,
            "alignment": alignment_component,
            "freshness": freshness_component,
        }
        weights = {
            "liquidity": self.lo_config.score_liquidity_weight,
            "orderflow": self.lo_config.score_orderflow_weight,
            "absorption": self.lo_config.score_absorption_weight,
            "price_action": self.lo_config.score_price_action_weight,
            "alignment": self.lo_config.score_alignment_weight,
            "freshness": self.lo_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=liquidity_component)
        confidence = confidence_from_components(
            primary=average_score(liquidity_component, orderflow_component),
            context=average_score(absorption_component, snapshot.confluence_score),
            confirmation=average_score(price_action_component, 1.0 - snapshot.conflict_score),
            freshness=freshness_component,
            primary_weight=self.lo_config.confidence_primary_weight,
            context_weight=self.lo_config.confidence_context_weight,
            confirmation_weight=self.lo_config.confidence_confirmation_weight,
            freshness_weight=self.lo_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "liquidity_orderflow_reversal_context",
            f"side:{payload.side.value}",
            f"sweep_side:{payload.sweep_side.value}",
            f"orderflow_exhaustion_side:{payload.orderflow_exhaustion_side.value}",
        ]

        conflicts = conflicting_source_names(snapshot.votes, payload.side)

        if liquidity_component >= self.lo_config.strong_liquidity_threshold:
            score += self.lo_config.strong_liquidity_bonus
            confirmations.append("strong_liquidity_sweep")

        if orderflow_component >= self.lo_config.strong_orderflow_threshold:
            score += self.lo_config.strong_orderflow_bonus
            confirmations.append("strong_orderflow_exhaustion")

        if absorption_component >= self.lo_config.strong_absorption_threshold:
            score += self.lo_config.absorption_confirmation_bonus
            confirmations.append("orderflow_absorption_confirmed")

        if price_action_component >= self.lo_config.strong_rejection_threshold:
            score += self.lo_config.price_rejection_bonus
            confirmations.append("price_action_rejection_confirmed")

        if payload.side == opposite_signal_side(payload.sweep_side):
            score += self.lo_config.opposite_flow_bonus
            confirmations.append("opposite_sweep_reversal")

        if snapshot.conflict_score <= self.lo_config.low_conflict_threshold:
            score += self.lo_config.low_conflict_bonus
            confirmations.append("low_domain_conflict")

        if payload.orderflow_exhaustion_side == payload.sweep_side:
            confirmations.append("orderflow_exhaustion_same_as_sweep_side")

        if payload.absorption_side == payload.side:
            confirmations.append("orderflow_absorption_same_as_reversal_side")

        if payload.rejection_side == payload.side:
            confirmations.append("price_rejection_same_as_reversal_side")

        reasons.extend(
            [
                f"liquidity_score:{liquidity_component:.4f}",
                f"orderflow_score:{orderflow_component:.4f}",
                f"absorption_score:{absorption_component:.4f}",
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

    def _absorption_score(
        self,
        orderflow: dict[str, Any],
    ) -> float:
        candidates = [
            get_path(orderflow, "absorption_score"),
            get_path(orderflow, "orderflow_absorption_score"),
            get_path(orderflow, "exhaustion_score"),
            get_path(orderflow, "delta_exhaustion_score"),
            get_path(orderflow, "context_strength"),
            get_path(orderflow, "metadata.absorption_score"),
            get_path(orderflow, "metadata.exhaustion_score"),
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
        payload: LiquidityOrderflowReversalPayload,
    ) -> list[str]:
        features = [
            *liquidity_orderflow_reversal_source_features(),
            HYBRID_FEATURES.DOMINANT_SIDE,
            HYBRID_FEATURES.ALIGNMENT_SCORE,
            HYBRID_FEATURES.CONFLICT_SCORE,
            HYBRID_FEATURES.CONFLUENCE_SCORE,
            HYBRID_FEATURES.CONFIDENCE,
            HYBRID_FEATURES.VOTES,
            "liquidity.*",
            "orderflow.*",
            "price_action.*",
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: LiquidityOrderflowReversalPayload,
    ) -> list[str]:
        tags = [
            self.lo_config.tag_hybrid,
            self.lo_config.tag_liquidity_orderflow,
            self.lo_config.tag_liquidity_orderflow_reversal,
            self.lo_config.tag_liquidity_sweep,
            self.lo_config.tag_stop_hunt,
            self.lo_config.tag_orderflow_exhaustion,
            f"side:{payload.side.value}",
            f"sweep_side:{payload.sweep_side.value}",
        ]

        if is_directional_side(payload.absorption_side):
            tags.append(self.lo_config.tag_orderflow_absorption)

        if is_directional_side(payload.rejection_side):
            tags.append(self.lo_config.tag_price_rejection)

        for source in payload.snapshot.aligned_domains:
            tags.append(f"aligned:{source}")

        return list(dict.fromkeys(tags))

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        return {
            "entry_offset_bps": self.lo_config.execution_entry_offset_bps_hint,
            "stop_buffer_bps": self.lo_config.execution_stop_buffer_bps_hint,
            "take_profit_bps": self.lo_config.execution_take_profit_bps_hint,
            "risk_reward": self.lo_config.liquidity_orderflow_rr_hint,
        }