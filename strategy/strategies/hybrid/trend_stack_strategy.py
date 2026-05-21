# trading_system/strategy/strategies/hybrid/trend_stack_strategy.py

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
    normalize_label,
    serialize_for_metadata,
    side_to_signal_side,
    trend_stack_source_features,
    unit_score,
    votes_against_side,
    votes_for_side,
    weighted_score,
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
class TrendStackPayload:
    """
    Normalized strategy-level payload для hybrid trend continuation stack.

    Direction convention:
    - price-action uptrend/breakout + same-side orderflow/OI/whales -> LONG;
    - price-action downtrend/breakdown + same-side orderflow/OI/whales -> SHORT.
    """

    snapshot: HybridCompositeSnapshot
    side: SignalSide

    trend_side: SignalSide = SignalSide.UNKNOWN
    orderflow_side: SignalSide = SignalSide.UNKNOWN
    oi_side: SignalSide = SignalSide.UNKNOWN
    whale_side: SignalSide = SignalSide.UNKNOWN
    funding_side: SignalSide = SignalSide.UNKNOWN

    aligned_votes: list[DirectionVote] = field(default_factory=list)
    conflicting_votes: list[DirectionVote] = field(default_factory=list)

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrendStackStrategyConfig(HybridStrategyConfig):
    """
    Hybrid trend-continuation stack config.

    Strategy idea:
    - price action defines trend / breakout direction;
    - orderflow confirms continuation;
    - open interest expansion confirms participation;
    - whales may confirm aggressive same-side flow;
    - funding can act as a guard against overcrowded/extreme opposite context;
    - strategy returns internal StrategySignal only.
    """

    min_trend_stack_score: float = 0.64
    min_trend_stack_confidence: float = 0.58

    min_price_action_score: float = 0.60
    min_orderflow_score: float = 0.55
    min_oi_score: float = 0.50
    min_whale_score: float = 0.50
    min_funding_score: float = 0.0

    require_price_action: bool = True
    require_orderflow: bool = True
    require_open_interest: bool = False
    require_whales: bool = False
    require_funding: bool = False

    use_open_interest_confirmation: bool = True
    use_whales_confirmation: bool = True
    use_funding_guard: bool = True

    require_price_action_side: bool = True
    require_orderflow_same_side: bool = True
    require_oi_same_side: bool = False
    require_whales_same_side: bool = False

    block_extreme_opposite_funding: bool = True
    block_funding_same_side_overcrowding: bool = False
    max_funding_extreme_score: float = 0.82

    reject_high_conflict: bool = True
    max_conflict_score: float = 0.38
    min_alignment_score: float = 0.58
    min_confluence_score: float = 0.56
    min_aligned_confirmations: int = 2

    price_action_vote_weight: float = 1.25
    orderflow_vote_weight: float = 1.15
    open_interest_vote_weight: float = 0.95
    whales_vote_weight: float = 0.95
    funding_vote_weight: float = 0.65

    score_price_action_weight: float = 0.28
    score_orderflow_weight: float = 0.24
    score_open_interest_weight: float = 0.16
    score_whales_weight: float = 0.10
    score_funding_guard_weight: float = 0.08
    score_alignment_weight: float = 0.08
    score_freshness_weight: float = 0.06

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    strong_price_action_bonus: float = 0.04
    strong_orderflow_bonus: float = 0.04
    oi_confirmation_bonus: float = 0.03
    whale_confirmation_bonus: float = 0.03
    healthy_funding_bonus: float = 0.03
    low_conflict_bonus: float = 0.03

    strong_price_action_threshold: float = 0.75
    strong_orderflow_threshold: float = 0.72
    strong_oi_threshold: float = 0.68
    strong_whale_threshold: float = 0.68
    low_conflict_threshold: float = 0.15

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.CONTINUATION

    tag_trend_stack: str = "trend_stack"
    tag_trend_continuation: str = "trend_continuation"
    tag_price_action_trend: str = "price_action_trend"
    tag_orderflow_continuation: str = "orderflow_continuation"
    tag_oi_expansion: str = "oi_expansion"
    tag_whale_confirmation: str = "whale_confirmation"
    tag_funding_guard: str = "funding_guard"

    execution_entry_offset_bps_hint: float | None = None
    execution_stop_buffer_bps_hint: float | None = None
    execution_take_profit_bps_hint: float | None = None
    trend_rr_hint: float | None = None

    required_hybrid_features: tuple[str, ...] = (
        HYBRID_FEATURES.DOMINANT_SIDE,
        HYBRID_FEATURES.ALIGNMENT_SCORE,
        HYBRID_FEATURES.CONFLUENCE_SCORE,
    )

    def validate(self) -> None:
        HybridStrategyConfig.validate(self)

        unit_fields = {
            "min_trend_stack_score": self.min_trend_stack_score,
            "min_trend_stack_confidence": self.min_trend_stack_confidence,
            "min_price_action_score": self.min_price_action_score,
            "min_orderflow_score": self.min_orderflow_score,
            "min_oi_score": self.min_oi_score,
            "min_whale_score": self.min_whale_score,
            "min_funding_score": self.min_funding_score,
            "max_funding_extreme_score": self.max_funding_extreme_score,
            "max_conflict_score": self.max_conflict_score,
            "min_alignment_score": self.min_alignment_score,
            "min_confluence_score": self.min_confluence_score,
            "strong_price_action_bonus": self.strong_price_action_bonus,
            "strong_orderflow_bonus": self.strong_orderflow_bonus,
            "oi_confirmation_bonus": self.oi_confirmation_bonus,
            "whale_confirmation_bonus": self.whale_confirmation_bonus,
            "healthy_funding_bonus": self.healthy_funding_bonus,
            "low_conflict_bonus": self.low_conflict_bonus,
            "strong_price_action_threshold": self.strong_price_action_threshold,
            "strong_orderflow_threshold": self.strong_orderflow_threshold,
            "strong_oi_threshold": self.strong_oi_threshold,
            "strong_whale_threshold": self.strong_whale_threshold,
            "low_conflict_threshold": self.low_conflict_threshold,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.min_aligned_confirmations <= 0:
            raise StrategyConfigError("min_aligned_confirmations must be > 0")

        vote_weights = {
            "price_action_vote_weight": self.price_action_vote_weight,
            "orderflow_vote_weight": self.orderflow_vote_weight,
            "open_interest_vote_weight": self.open_interest_vote_weight,
            "whales_vote_weight": self.whales_vote_weight,
            "funding_vote_weight": self.funding_vote_weight,
        }
        score_weights = {
            "score_price_action_weight": self.score_price_action_weight,
            "score_orderflow_weight": self.score_orderflow_weight,
            "score_open_interest_weight": self.score_open_interest_weight,
            "score_whales_weight": self.score_whales_weight,
            "score_funding_guard_weight": self.score_funding_guard_weight,
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
            "trend_rr_hint": self.trend_rr_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        for attr in (
            "tag_trend_stack",
            "tag_trend_continuation",
            "tag_price_action_trend",
            "tag_orderflow_continuation",
            "tag_oi_expansion",
            "tag_whale_confirmation",
            "tag_funding_guard",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


class TrendStackStrategy(HybridTradingStrategy):
    """
    Hybrid trend-continuation stack.

    Input:
        StrategyContext with FeatureSource.PRICE_ACTION, ORDERFLOW,
        and optional OPEN_INTEREST/WHALES/FUNDING domain sections.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    It does not duplicate SignalProcessor.ConfluenceEngine.
    SignalProcessor owns global routing, confluence, filters, portfolio coordination,
    building and risk payloads.
    """

    component_namespace = "strategy.hybrid.trend_stack"
    category: StrategyCategory = StrategyCategory.HYBRID
    default_setup_type: SetupType = SetupType.CONTINUATION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        hybrid_config: TrendStackStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_hybrid_config = hybrid_config or TrendStackStrategyConfig()
        resolved_hybrid_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            hybrid_config=resolved_hybrid_config,
            service_name=service_name,
        )

        self.trend_config: TrendStackStrategyConfig = resolved_hybrid_config

    @property
    def strategy_name(self) -> str:
        return "trend_stack"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.HYBRID,
            timeframe=Timeframe.M1,
            tags=[
                self.trend_config.tag_hybrid,
                self.trend_config.tag_trend_stack,
                self.trend_config.tag_trend_continuation,
                self.trend_config.tag_price_action_trend,
                self.trend_config.tag_orderflow_continuation,
                "strategy_context",
            ],
            version="2.0.0",
            description=(
                "Builds a local trend-continuation stack signal from price action, "
                "orderflow, optional open interest, whales and funding guard."
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
                "strategy_type": "trend_stack",
                "base_class": "HybridTradingStrategy",
                "canonical_payload": "HybridCompositeSnapshot",
                "uses_price_action": True,
                "uses_orderflow": True,
                "uses_open_interest": True,
                "uses_whales": True,
                "uses_funding_guard": True,
                "duplicates_signal_processor_confluence": False,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(self.trend_config.required_hybrid_features)

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
            allow_missing=self.trend_config.allow_missing_required_domains,
        )
        if not required_domains_available:
            return None

        if not self._has_minimum_available_domains(context, sources):
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
            stale_after_seconds=self.trend_config.stale_feature_max_age_seconds,
        ):
            return None

        if not self._passes_trend_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.trend_config.min_trend_stack_score:
            return None

        if breakdown.confidence < self.trend_config.min_trend_stack_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "trend_stack_signal",
                    f"side:{payload.side.value}",
                    f"trend_side:{payload.trend_side.value}",
                    f"orderflow_side:{payload.orderflow_side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "hybrid_setup_family": "trend_stack",
            "hybrid_strategy_version": "2.0.0",
            "contract": "hybrid",
            "contract_version": "strategy-domain-v1",
            "primary_section": "trend_stack",
            "strategy_contract_role": "decision_module",
            "risk_ready_payload_owner": "SignalProcessor",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "snapshot": serialize_for_metadata(payload.snapshot.to_dict()),
            "votes": [vote.to_dict() for vote in payload.snapshot.votes],
            "aligned_votes": [vote.to_dict() for vote in payload.aligned_votes],
            "conflicting_votes": [vote.to_dict() for vote in payload.conflicting_votes],
            "raw": serialize_for_metadata(payload.raw),
            "event_time": payload.event_time.isoformat() if payload.event_time else None,
            "mapped_side": payload.side.value,
            "trend_side": payload.trend_side.value,
            "orderflow_side": payload.orderflow_side.value,
            "oi_side": payload.oi_side.value,
            "whale_side": payload.whale_side.value,
            "funding_side": payload.funding_side.value,
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
            setup_type=self.trend_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.trend_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Sources / weights
    # ------------------------------------------------------------------

    def _enabled_sources(self) -> tuple[FeatureSource, ...]:
        sources: list[FeatureSource] = [
            FeatureSource.PRICE_ACTION,
            FeatureSource.ORDERFLOW,
        ]

        if self.trend_config.use_open_interest_confirmation:
            sources.append(FeatureSource.OPEN_INTEREST)

        if self.trend_config.use_whales_confirmation:
            sources.append(FeatureSource.WHALES)

        if self.trend_config.use_funding_guard:
            sources.append(FeatureSource.FUNDING)

        return tuple(dict.fromkeys(sources))

    def _required_sources(self) -> tuple[FeatureSource, ...]:
        sources: list[FeatureSource] = []

        if self.trend_config.require_price_action:
            sources.append(FeatureSource.PRICE_ACTION)

        if self.trend_config.require_orderflow:
            sources.append(FeatureSource.ORDERFLOW)

        if self.trend_config.require_open_interest:
            sources.append(FeatureSource.OPEN_INTEREST)

        if self.trend_config.require_whales:
            sources.append(FeatureSource.WHALES)

        if self.trend_config.require_funding:
            sources.append(FeatureSource.FUNDING)

        return tuple(dict.fromkeys(sources))

    def _vote_weights(self) -> dict[FeatureSource, float]:
        return {
            FeatureSource.PRICE_ACTION: self.trend_config.price_action_vote_weight,
            FeatureSource.ORDERFLOW: self.trend_config.orderflow_vote_weight,
            FeatureSource.OPEN_INTEREST: self.trend_config.open_interest_vote_weight,
            FeatureSource.WHALES: self.trend_config.whales_vote_weight,
            FeatureSource.FUNDING: self.trend_config.funding_vote_weight,
        }

    def _has_minimum_available_domains(
        self,
        context: StrategyContext,
        sources: tuple[FeatureSource, ...],
    ) -> bool:
        available_count = len(self.available_domain_sources(context, sources))
        return available_count >= self.trend_config.min_required_domains

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
    ) -> TrendStackPayload | None:
        payloads = snapshot.payloads()

        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})
        orderflow = payloads.get(FeatureSource.ORDERFLOW, {})
        open_interest = payloads.get(FeatureSource.OPEN_INTEREST, {})
        whales = payloads.get(FeatureSource.WHALES, {})
        funding = payloads.get(FeatureSource.FUNDING, {})

        trend_side = self._extract_trend_side(price_action)
        if self.trend_config.require_price_action_side and not is_directional_side(trend_side):
            return None

        side = trend_side
        if not is_directional_side(side):
            side = snapshot.dominant_side

        if not is_directional_side(side):
            return None

        orderflow_side = self._extract_orderflow_side(orderflow)
        oi_side = self._extract_oi_side(open_interest)
        whale_side = self._extract_whale_side(whales)
        funding_side = self._extract_funding_side(funding)

        aligned_votes = votes_for_side(snapshot.votes, side)
        conflicting_votes = votes_against_side(snapshot.votes, side)

        event_time = (
            latest_timestamp_from_payloads(payloads, fallback=context.timestamp)
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "trend_stack_context",
            f"side:{side.value}",
            f"trend_side:{trend_side.value}",
            f"orderflow_side:{orderflow_side.value}",
            f"oi_side:{oi_side.value}",
            f"whale_side:{whale_side.value}",
            f"funding_side:{funding_side.value}",
            f"alignment_score:{snapshot.alignment_score:.4f}",
            f"conflict_score:{snapshot.conflict_score:.4f}",
            f"confluence_score:{snapshot.confluence_score:.4f}",
        ]

        return TrendStackPayload(
            snapshot=snapshot,
            side=side,
            trend_side=trend_side,
            orderflow_side=orderflow_side,
            oi_side=oi_side,
            whale_side=whale_side,
            funding_side=funding_side,
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

    def _extract_trend_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "trend_side",
            "trend_direction",
            "breakout_side",
            "continuation_side",
            "signal_side",
            "side",
            "direction",
            "bias",
            "metadata.trend_side",
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

    def _extract_oi_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "oi_side",
            "open_interest_side",
            "breakout_side",
            "confirmation_side",
            "regime_side",
            "signal_side",
            "side",
            "direction",
            "bias",
            "metadata.oi_side",
            "metadata.side",
        ):
            side = side_to_signal_side(get_path(payload, path))
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    def _extract_whale_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "whale_side",
            "activity_side",
            "pressure_side",
            "dominant_side",
            "breakout_side",
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

    def _extract_funding_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "funding_side",
            "pressure_side",
            "crowded_side",
            "signal_side",
            "side",
            "direction",
            "bias",
            "metadata.funding_side",
            "metadata.side",
        ):
            side = side_to_signal_side(get_path(payload, path))
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_trend_filters(
        self,
        payload: TrendStackPayload,
    ) -> bool:
        snapshot = payload.snapshot
        payloads = snapshot.payloads()

        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})
        orderflow = payloads.get(FeatureSource.ORDERFLOW, {})
        open_interest = payloads.get(FeatureSource.OPEN_INTEREST, {})
        whales = payloads.get(FeatureSource.WHALES, {})
        funding = payloads.get(FeatureSource.FUNDING, {})

        if self.trend_config.reject_high_conflict:
            if snapshot.conflict_score > self.trend_config.max_conflict_score:
                return False

        if snapshot.alignment_score < self.trend_config.min_alignment_score:
            return False

        if snapshot.confluence_score < self.trend_config.min_confluence_score:
            return False

        if extract_domain_score(price_action) < self.trend_config.min_price_action_score:
            return False

        if extract_domain_score(orderflow) < self.trend_config.min_orderflow_score:
            return False

        if self.trend_config.require_orderflow_same_side:
            if is_directional_side(payload.orderflow_side):
                if payload.orderflow_side != payload.side:
                    return False
            else:
                return False

        if self.trend_config.use_open_interest_confirmation and open_interest:
            oi_score = extract_domain_score(open_interest)
            if oi_score > 0.0 and oi_score < self.trend_config.min_oi_score:
                return False

            if self.trend_config.require_oi_same_side:
                if is_directional_side(payload.oi_side):
                    if payload.oi_side != payload.side:
                        return False
                else:
                    return False

        if self.trend_config.use_whales_confirmation and whales:
            whale_score = extract_domain_score(whales)
            if whale_score > 0.0 and whale_score < self.trend_config.min_whale_score:
                return False

            if self.trend_config.require_whales_same_side:
                if is_directional_side(payload.whale_side):
                    if payload.whale_side != payload.side:
                        return False
                else:
                    return False

        if self.trend_config.use_funding_guard and funding:
            if not self._passes_funding_guard(payload):
                return False

        aligned_confirmations = self._aligned_confirmation_count(payload)
        if aligned_confirmations < self.trend_config.min_aligned_confirmations:
            return False

        return True

    def _passes_funding_guard(
        self,
        payload: TrendStackPayload,
    ) -> bool:
        funding = payload.snapshot.payloads().get(FeatureSource.FUNDING, {})
        funding_score = extract_domain_score(funding)

        if funding_score < self.trend_config.min_funding_score:
            return True

        extreme_label = normalize_label(
            get_path(funding, "extreme")
            or get_path(funding, "is_extreme")
            or get_path(funding, "regime")
            or get_path(funding, "metadata.regime")
        )
        is_extreme = extreme_label in {
            "true",
            "extreme",
            "overheated",
            "crowded",
            "positive_extreme",
            "negative_extreme",
            "funding_extreme",
        }

        if not is_extreme:
            return True

        if funding_score < self.trend_config.max_funding_extreme_score:
            return True

        if self.trend_config.block_extreme_opposite_funding:
            if is_directional_side(payload.funding_side) and payload.funding_side != payload.side:
                return False

        if self.trend_config.block_funding_same_side_overcrowding:
            if is_directional_side(payload.funding_side) and payload.funding_side == payload.side:
                return False

        return True

    def _aligned_confirmation_count(
        self,
        payload: TrendStackPayload,
    ) -> int:
        count = 0

        if is_directional_side(payload.trend_side) and payload.trend_side == payload.side:
            count += 1

        if is_directional_side(payload.orderflow_side) and payload.orderflow_side == payload.side:
            count += 1

        if is_directional_side(payload.oi_side) and payload.oi_side == payload.side:
            count += 1

        if is_directional_side(payload.whale_side) and payload.whale_side == payload.side:
            count += 1

        if is_directional_side(payload.funding_side) and payload.funding_side == payload.side:
            count += 1

        return count

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: TrendStackPayload,
    ) -> HybridScoreBreakdown:
        snapshot = payload.snapshot
        payloads = snapshot.payloads()

        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})
        orderflow = payloads.get(FeatureSource.ORDERFLOW, {})
        open_interest = payloads.get(FeatureSource.OPEN_INTEREST, {})
        whales = payloads.get(FeatureSource.WHALES, {})
        funding = payloads.get(FeatureSource.FUNDING, {})

        price_action_component = extract_domain_score(price_action)
        orderflow_component = extract_domain_score(orderflow)
        oi_component = (
            extract_domain_score(open_interest)
            if open_interest and self.trend_config.use_open_interest_confirmation
            else 0.0
        )
        whales_component = (
            extract_domain_score(whales)
            if whales and self.trend_config.use_whales_confirmation
            else 0.0
        )
        funding_guard_component = (
            self._funding_guard_score(payload)
            if funding and self.trend_config.use_funding_guard
            else 1.0
        )
        alignment_component = snapshot.alignment_score
        freshness_component = hybrid_freshness_score(
            payloads,
            now=context.timestamp,
            stale_after_seconds=self.trend_config.stale_feature_max_age_seconds,
        )

        components = {
            "price_action": price_action_component,
            "orderflow": orderflow_component,
            "open_interest": oi_component,
            "whales": whales_component,
            "funding_guard": funding_guard_component,
            "alignment": alignment_component,
            "freshness": freshness_component,
        }
        weights = {
            "price_action": self.trend_config.score_price_action_weight,
            "orderflow": self.trend_config.score_orderflow_weight,
            "open_interest": self.trend_config.score_open_interest_weight,
            "whales": self.trend_config.score_whales_weight,
            "funding_guard": self.trend_config.score_funding_guard_weight,
            "alignment": self.trend_config.score_alignment_weight,
            "freshness": self.trend_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=price_action_component)
        confidence = confidence_from_components(
            primary=average_score(price_action_component, orderflow_component),
            context=average_score(oi_component, snapshot.confluence_score),
            confirmation=average_score(whales_component, funding_guard_component),
            freshness=freshness_component,
            primary_weight=self.trend_config.confidence_primary_weight,
            context_weight=self.trend_config.confidence_context_weight,
            confirmation_weight=self.trend_config.confidence_confirmation_weight,
            freshness_weight=self.trend_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "trend_stack_context",
            f"side:{payload.side.value}",
            f"trend_side:{payload.trend_side.value}",
            f"orderflow_side:{payload.orderflow_side.value}",
        ]

        conflicts = conflicting_source_names(snapshot.votes, payload.side)

        if price_action_component >= self.trend_config.strong_price_action_threshold:
            score += self.trend_config.strong_price_action_bonus
            confirmations.append("strong_price_action_trend")

        if orderflow_component >= self.trend_config.strong_orderflow_threshold:
            score += self.trend_config.strong_orderflow_bonus
            confirmations.append("strong_orderflow_continuation")

        if oi_component >= self.trend_config.strong_oi_threshold:
            score += self.trend_config.oi_confirmation_bonus
            confirmations.append("open_interest_confirms_trend")

        if whales_component >= self.trend_config.strong_whale_threshold:
            score += self.trend_config.whale_confirmation_bonus
            confirmations.append("whales_confirm_trend")

        if funding_guard_component >= 0.80:
            score += self.trend_config.healthy_funding_bonus
            confirmations.append("funding_guard_passed")

        if snapshot.conflict_score <= self.trend_config.low_conflict_threshold:
            score += self.trend_config.low_conflict_bonus
            confirmations.append("low_domain_conflict")

        if is_directional_side(payload.trend_side) and payload.trend_side == payload.side:
            confirmations.append("price_action_same_as_trend_side")

        if is_directional_side(payload.orderflow_side) and payload.orderflow_side == payload.side:
            confirmations.append("orderflow_same_as_trend_side")

        if is_directional_side(payload.oi_side) and payload.oi_side == payload.side:
            confirmations.append("open_interest_same_as_trend_side")

        if is_directional_side(payload.whale_side) and payload.whale_side == payload.side:
            confirmations.append("whales_same_as_trend_side")

        reasons.extend(
            [
                f"price_action_score:{price_action_component:.4f}",
                f"orderflow_score:{orderflow_component:.4f}",
                f"open_interest_score:{oi_component:.4f}",
                f"whales_score:{whales_component:.4f}",
                f"funding_guard_score:{funding_guard_component:.4f}",
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

    def _funding_guard_score(
        self,
        payload: TrendStackPayload,
    ) -> float:
        funding = payload.snapshot.payloads().get(FeatureSource.FUNDING, {})
        if not funding:
            return 1.0

        funding_score = extract_domain_score(funding)
        if funding_score <= 0.0:
            return 1.0

        extreme_label = normalize_label(
            get_path(funding, "extreme")
            or get_path(funding, "is_extreme")
            or get_path(funding, "regime")
            or get_path(funding, "metadata.regime")
        )
        is_extreme = extreme_label in {
            "true",
            "extreme",
            "overheated",
            "crowded",
            "positive_extreme",
            "negative_extreme",
            "funding_extreme",
        }

        if not is_extreme:
            return 1.0

        if is_directional_side(payload.funding_side):
            if payload.funding_side != payload.side and self.trend_config.block_extreme_opposite_funding:
                return unit_score(1.0 - funding_score)

            if payload.funding_side == payload.side and self.trend_config.block_funding_same_side_overcrowding:
                return unit_score(1.0 - funding_score)

        return unit_score(1.0 - max(0.0, funding_score - self.trend_config.max_funding_extreme_score))

    # ------------------------------------------------------------------
    # Source features / tags / metadata helpers
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: TrendStackPayload,
    ) -> list[str]:
        features = [
            *trend_stack_source_features(),
            HYBRID_FEATURES.DOMINANT_SIDE,
            HYBRID_FEATURES.ALIGNMENT_SCORE,
            HYBRID_FEATURES.CONFLICT_SCORE,
            HYBRID_FEATURES.CONFLUENCE_SCORE,
            HYBRID_FEATURES.CONFIDENCE,
            HYBRID_FEATURES.VOTES,
            "price_action.*",
            "orderflow.*",
            "open_interest.*",
            "whales.*",
            "funding.*",
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: TrendStackPayload,
    ) -> list[str]:
        tags = [
            self.trend_config.tag_hybrid,
            self.trend_config.tag_trend_stack,
            self.trend_config.tag_trend_continuation,
            self.trend_config.tag_price_action_trend,
            self.trend_config.tag_orderflow_continuation,
            f"side:{payload.side.value}",
            f"trend_side:{payload.trend_side.value}",
        ]

        if is_directional_side(payload.oi_side):
            tags.append(self.trend_config.tag_oi_expansion)

        if is_directional_side(payload.whale_side):
            tags.append(self.trend_config.tag_whale_confirmation)

        if self.trend_config.use_funding_guard:
            tags.append(self.trend_config.tag_funding_guard)

        for source in payload.snapshot.aligned_domains:
            tags.append(f"aligned:{source}")

        return list(dict.fromkeys(tags))

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        return {
            "entry_offset_bps": self.trend_config.execution_entry_offset_bps_hint,
            "stop_buffer_bps": self.trend_config.execution_stop_buffer_bps_hint,
            "take_profit_bps": self.trend_config.execution_take_profit_bps_hint,
            "risk_reward": self.trend_config.trend_rr_hint,
        }