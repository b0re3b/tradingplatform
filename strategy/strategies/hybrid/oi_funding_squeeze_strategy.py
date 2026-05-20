# trading_system/strategy/strategies/hybrid/oi_funding_squeeze_strategy.py

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
    oi_funding_squeeze_source_features,
    opposite_signal_side,
    serialize_for_metadata,
    side_to_signal_side,
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
class OIFundingSqueezePayload:
    """
    Normalized strategy-level payload для OI + funding squeeze / unwind.

    Direction convention:
    - crowded shorts + negative/extreme funding + squeeze confirmation -> LONG;
    - crowded longs + positive/extreme funding + unwind confirmation -> SHORT.
    """

    snapshot: HybridCompositeSnapshot
    side: SignalSide

    oi_side: SignalSide = SignalSide.UNKNOWN
    funding_side: SignalSide = SignalSide.UNKNOWN
    crowded_side: SignalSide = SignalSide.UNKNOWN
    squeeze_side: SignalSide = SignalSide.UNKNOWN
    price_action_side: SignalSide = SignalSide.UNKNOWN

    aligned_votes: list[DirectionVote] = field(default_factory=list)
    conflicting_votes: list[DirectionVote] = field(default_factory=list)

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OIFundingSqueezeStrategyConfig(HybridStrategyConfig):
    """
    Hybrid OI + funding squeeze config.

    Strategy idea:
    - open interest identifies crowding / participation / squeeze pressure;
    - funding identifies extreme/crowded side;
    - squeeze side is usually opposite to crowded/funding side;
    - optional price action confirms reclaim/rejection;
    - strategy returns internal StrategySignal only.
    """

    min_squeeze_score: float = 0.64
    min_squeeze_confidence: float = 0.58

    min_oi_score: float = 0.58
    min_funding_extreme_score: float = 0.58
    min_squeeze_probability: float = 0.60
    min_price_action_score: float = 0.50

    require_open_interest: bool = True
    require_funding: bool = True
    require_price_action_confirmation: bool = False

    require_funding_extreme: bool = True
    require_crowded_side: bool = True
    require_squeeze_side: bool = False
    require_squeeze_opposite_crowded: bool = True
    require_price_action_same_as_squeeze: bool = False

    use_price_action_confirmation: bool = True
    reject_same_side_crowding_momentum: bool = True
    reject_high_conflict: bool = True

    max_conflict_score: float = 0.38
    min_alignment_score: float = 0.50
    min_confluence_score: float = 0.50

    open_interest_vote_weight: float = 1.20
    funding_vote_weight: float = 1.25
    price_action_vote_weight: float = 0.90

    score_oi_weight: float = 0.32
    score_funding_weight: float = 0.32
    score_squeeze_probability_weight: float = 0.14
    score_price_action_weight: float = 0.10
    score_alignment_weight: float = 0.06
    score_freshness_weight: float = 0.06

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    strong_oi_bonus: float = 0.04
    strong_funding_bonus: float = 0.05
    squeeze_probability_bonus: float = 0.04
    opposite_crowding_bonus: float = 0.05
    price_action_confirmation_bonus: float = 0.03
    low_conflict_bonus: float = 0.03

    strong_oi_threshold: float = 0.72
    strong_funding_threshold: float = 0.72
    strong_squeeze_probability_threshold: float = 0.75
    low_conflict_threshold: float = 0.15

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.SQUEEZE

    tag_oi_funding: str = "oi_funding"
    tag_oi_funding_squeeze: str = "oi_funding_squeeze"
    tag_open_interest: str = "open_interest"
    tag_funding_extreme: str = "funding_extreme"
    tag_squeeze: str = "squeeze"
    tag_crowded_positioning: str = "crowded_positioning"
    tag_price_action_confirmation: str = "price_action_confirmation"

    execution_entry_offset_bps_hint: float | None = None
    execution_stop_buffer_bps_hint: float | None = None
    execution_take_profit_bps_hint: float | None = None
    squeeze_rr_hint: float | None = None

    required_hybrid_features: tuple[str, ...] = (
        HYBRID_FEATURES.DOMINANT_SIDE,
        HYBRID_FEATURES.ALIGNMENT_SCORE,
        HYBRID_FEATURES.CONFLUENCE_SCORE,
    )

    def validate(self) -> None:
        HybridStrategyConfig.validate(self)

        unit_fields = {
            "min_squeeze_score": self.min_squeeze_score,
            "min_squeeze_confidence": self.min_squeeze_confidence,
            "min_oi_score": self.min_oi_score,
            "min_funding_extreme_score": self.min_funding_extreme_score,
            "min_squeeze_probability": self.min_squeeze_probability,
            "min_price_action_score": self.min_price_action_score,
            "max_conflict_score": self.max_conflict_score,
            "min_alignment_score": self.min_alignment_score,
            "min_confluence_score": self.min_confluence_score,
            "strong_oi_bonus": self.strong_oi_bonus,
            "strong_funding_bonus": self.strong_funding_bonus,
            "squeeze_probability_bonus": self.squeeze_probability_bonus,
            "opposite_crowding_bonus": self.opposite_crowding_bonus,
            "price_action_confirmation_bonus": self.price_action_confirmation_bonus,
            "low_conflict_bonus": self.low_conflict_bonus,
            "strong_oi_threshold": self.strong_oi_threshold,
            "strong_funding_threshold": self.strong_funding_threshold,
            "strong_squeeze_probability_threshold": self.strong_squeeze_probability_threshold,
            "low_conflict_threshold": self.low_conflict_threshold,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        vote_weights = {
            "open_interest_vote_weight": self.open_interest_vote_weight,
            "funding_vote_weight": self.funding_vote_weight,
            "price_action_vote_weight": self.price_action_vote_weight,
        }
        score_weights = {
            "score_oi_weight": self.score_oi_weight,
            "score_funding_weight": self.score_funding_weight,
            "score_squeeze_probability_weight": self.score_squeeze_probability_weight,
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
            "squeeze_rr_hint": self.squeeze_rr_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        for attr in (
            "tag_oi_funding",
            "tag_oi_funding_squeeze",
            "tag_open_interest",
            "tag_funding_extreme",
            "tag_squeeze",
            "tag_crowded_positioning",
            "tag_price_action_confirmation",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


class OIFundingSqueezeStrategy(HybridTradingStrategy):
    """
    Hybrid open-interest + funding squeeze strategy.

    Input:
        StrategyContext with FeatureSource.OPEN_INTEREST and FeatureSource.FUNDING,
        optionally FeatureSource.PRICE_ACTION.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    It does not duplicate SignalProcessor.ConfluenceEngine.
    SignalProcessor owns global routing, confluence, filters, portfolio coordination,
    building and risk payloads.
    """

    component_namespace = "strategy.hybrid.oi_funding_squeeze"
    category: StrategyCategory = StrategyCategory.HYBRID
    default_setup_type: SetupType = SetupType.SQUEEZE

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        hybrid_config: OIFundingSqueezeStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_hybrid_config = hybrid_config or OIFundingSqueezeStrategyConfig()
        resolved_hybrid_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            hybrid_config=resolved_hybrid_config,
            service_name=service_name,
        )

        self.squeeze_config: OIFundingSqueezeStrategyConfig = resolved_hybrid_config

    @property
    def strategy_name(self) -> str:
        return "oi_funding_squeeze"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.HYBRID,
            timeframe=Timeframe.M1,
            tags=[
                self.squeeze_config.tag_hybrid,
                self.squeeze_config.tag_oi_funding,
                self.squeeze_config.tag_oi_funding_squeeze,
                self.squeeze_config.tag_open_interest,
                self.squeeze_config.tag_funding_extreme,
                self.squeeze_config.tag_squeeze,
                "strategy_context",
            ],
            version="2.0.0",
            description=(
                "Builds a specialized open-interest + funding squeeze/unwind signal "
                "from crowded positioning and funding extremes."
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
                "strategy_type": "oi_funding_squeeze",
                "base_class": "HybridTradingStrategy",
                "canonical_payload": "HybridCompositeSnapshot",
                "uses_open_interest": True,
                "uses_funding": True,
                "uses_price_action_confirmation": True,
                "requires_oi_funding": True,
                "duplicates_signal_processor_confluence": False,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(self.squeeze_config.required_hybrid_features)

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
            allow_missing=self.squeeze_config.allow_missing_required_domains,
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
            stale_after_seconds=self.squeeze_config.stale_feature_max_age_seconds,
        ):
            return None

        if not self._passes_squeeze_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.squeeze_config.min_squeeze_score:
            return None

        if breakdown.confidence < self.squeeze_config.min_squeeze_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "oi_funding_squeeze_signal",
                    f"side:{payload.side.value}",
                    f"crowded_side:{payload.crowded_side.value}",
                    f"squeeze_side:{payload.squeeze_side.value}",
                    f"funding_side:{payload.funding_side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "hybrid_setup_family": "oi_funding_squeeze",
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
            "oi_side": payload.oi_side.value,
            "funding_side": payload.funding_side.value,
            "crowded_side": payload.crowded_side.value,
            "squeeze_side": payload.squeeze_side.value,
            "price_action_side": payload.price_action_side.value,
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
            setup_type=self.squeeze_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.squeeze_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Sources / weights
    # ------------------------------------------------------------------

    def _enabled_sources(self) -> tuple[FeatureSource, ...]:
        sources: list[FeatureSource] = [
            FeatureSource.OPEN_INTEREST,
            FeatureSource.FUNDING,
        ]

        if self.squeeze_config.use_price_action_confirmation:
            sources.append(FeatureSource.PRICE_ACTION)

        return tuple(dict.fromkeys(sources))

    def _required_sources(self) -> tuple[FeatureSource, ...]:
        sources: list[FeatureSource] = []

        if self.squeeze_config.require_open_interest:
            sources.append(FeatureSource.OPEN_INTEREST)

        if self.squeeze_config.require_funding:
            sources.append(FeatureSource.FUNDING)

        if self.squeeze_config.require_price_action_confirmation:
            sources.append(FeatureSource.PRICE_ACTION)

        return tuple(dict.fromkeys(sources))

    def _vote_weights(self) -> dict[FeatureSource, float]:
        return {
            FeatureSource.OPEN_INTEREST: self.squeeze_config.open_interest_vote_weight,
            FeatureSource.FUNDING: self.squeeze_config.funding_vote_weight,
            FeatureSource.PRICE_ACTION: self.squeeze_config.price_action_vote_weight,
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
    ) -> OIFundingSqueezePayload | None:
        payloads = snapshot.payloads()

        open_interest = payloads.get(FeatureSource.OPEN_INTEREST, {})
        funding = payloads.get(FeatureSource.FUNDING, {})
        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})

        if not open_interest or not funding:
            return None

        oi_side = self._extract_oi_side(open_interest)
        funding_side = self._extract_funding_side(funding)
        crowded_side = self._extract_crowded_side(open_interest, funding)
        squeeze_side = self._extract_squeeze_side(open_interest, funding)
        price_action_side = self._extract_price_action_side(price_action)

        if self.squeeze_config.require_crowded_side and not is_directional_side(crowded_side):
            return None

        side = squeeze_side
        if not is_directional_side(side) and is_directional_side(crowded_side):
            side = opposite_signal_side(crowded_side)

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
            "oi_funding_squeeze_context",
            f"side:{side.value}",
            f"oi_side:{oi_side.value}",
            f"funding_side:{funding_side.value}",
            f"crowded_side:{crowded_side.value}",
            f"squeeze_side:{squeeze_side.value}",
            f"price_action_side:{price_action_side.value}",
            f"alignment_score:{snapshot.alignment_score:.4f}",
            f"conflict_score:{snapshot.conflict_score:.4f}",
            f"confluence_score:{snapshot.confluence_score:.4f}",
        ]

        return OIFundingSqueezePayload(
            snapshot=snapshot,
            side=side,
            oi_side=oi_side,
            funding_side=funding_side,
            crowded_side=crowded_side,
            squeeze_side=squeeze_side,
            price_action_side=price_action_side,
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

    def _extract_oi_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "oi_side",
            "open_interest_side",
            "crowded_side",
            "squeeze_side",
            "regime_side",
            "divergence_side",
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

    def _extract_funding_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "funding_side",
            "crowded_side",
            "pressure_side",
            "extreme_side",
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

    def _extract_crowded_side(
        self,
        open_interest: dict[str, Any],
        funding: dict[str, Any],
    ) -> SignalSide:
        for payload in (funding, open_interest):
            for path in (
                "crowded_side",
                "overcrowded_side",
                "positioning_side",
                "funding_side",
                "oi_side",
                "extreme_side",
                "side",
                "metadata.crowded_side",
                "metadata.positioning_side",
                "metadata.side",
            ):
                side = side_to_signal_side(get_path(payload, path))
                if is_directional_side(side):
                    return side
        return SignalSide.UNKNOWN

    def _extract_squeeze_side(
        self,
        open_interest: dict[str, Any],
        funding: dict[str, Any],
    ) -> SignalSide:
        for payload in (open_interest, funding):
            for path in (
                "squeeze_side",
                "unwind_side",
                "reversal_side",
                "expected_side",
                "signal_side",
                "side",
                "direction",
                "metadata.squeeze_side",
                "metadata.reversal_side",
            ):
                side = side_to_signal_side(get_path(payload, path))
                if is_directional_side(side):
                    return side
        return SignalSide.UNKNOWN

    def _extract_price_action_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "confirmation_side",
            "reclaim_side",
            "rejection_side",
            "breakout_side",
            "signal_side",
            "side",
            "direction",
            "bias",
            "metadata.confirmation_side",
            "metadata.side",
        ):
            side = side_to_signal_side(get_path(payload, path))
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_squeeze_filters(
        self,
        payload: OIFundingSqueezePayload,
    ) -> bool:
        snapshot = payload.snapshot
        payloads = snapshot.payloads()

        open_interest = payloads.get(FeatureSource.OPEN_INTEREST, {})
        funding = payloads.get(FeatureSource.FUNDING, {})
        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})

        if self.squeeze_config.reject_high_conflict:
            if snapshot.conflict_score > self.squeeze_config.max_conflict_score:
                return False

        if snapshot.alignment_score < self.squeeze_config.min_alignment_score:
            return False

        if snapshot.confluence_score < self.squeeze_config.min_confluence_score:
            return False

        if extract_domain_score(open_interest) < self.squeeze_config.min_oi_score:
            return False

        if extract_domain_score(funding) < self.squeeze_config.min_funding_extreme_score:
            return False

        if self.squeeze_config.require_funding_extreme:
            if not self._funding_is_extreme(funding):
                return False

        squeeze_probability = self._squeeze_probability(open_interest, funding)
        if squeeze_probability < self.squeeze_config.min_squeeze_probability:
            return False

        if self.squeeze_config.require_squeeze_side:
            if not is_directional_side(payload.squeeze_side):
                return False

        if self.squeeze_config.require_squeeze_opposite_crowded:
            if is_directional_side(payload.crowded_side):
                if payload.side != opposite_signal_side(payload.crowded_side):
                    return False

        if self.squeeze_config.reject_same_side_crowding_momentum:
            if is_directional_side(payload.crowded_side):
                if payload.side == payload.crowded_side:
                    return False

        if self.squeeze_config.use_price_action_confirmation and price_action:
            if extract_domain_score(price_action) < self.squeeze_config.min_price_action_score:
                return False

            if self.squeeze_config.require_price_action_same_as_squeeze:
                if not is_directional_side(payload.price_action_side):
                    return False

                if payload.price_action_side != payload.side:
                    return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: OIFundingSqueezePayload,
    ) -> HybridScoreBreakdown:
        snapshot = payload.snapshot
        payloads = snapshot.payloads()

        open_interest = payloads.get(FeatureSource.OPEN_INTEREST, {})
        funding = payloads.get(FeatureSource.FUNDING, {})
        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})

        oi_component = extract_domain_score(open_interest)
        funding_component = extract_domain_score(funding)
        squeeze_probability_component = self._squeeze_probability(open_interest, funding)
        price_action_component = (
            extract_domain_score(price_action)
            if price_action and self.squeeze_config.use_price_action_confirmation
            else 0.0
        )
        alignment_component = snapshot.alignment_score
        freshness_component = hybrid_freshness_score(
            payloads,
            now=context.timestamp,
            stale_after_seconds=self.squeeze_config.stale_feature_max_age_seconds,
        )

        components = {
            "open_interest": oi_component,
            "funding": funding_component,
            "squeeze_probability": squeeze_probability_component,
            "price_action": price_action_component,
            "alignment": alignment_component,
            "freshness": freshness_component,
        }
        weights = {
            "open_interest": self.squeeze_config.score_oi_weight,
            "funding": self.squeeze_config.score_funding_weight,
            "squeeze_probability": self.squeeze_config.score_squeeze_probability_weight,
            "price_action": self.squeeze_config.score_price_action_weight,
            "alignment": self.squeeze_config.score_alignment_weight,
            "freshness": self.squeeze_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=oi_component)
        confidence = confidence_from_components(
            primary=average_score(oi_component, funding_component),
            context=average_score(squeeze_probability_component, snapshot.confluence_score),
            confirmation=average_score(price_action_component, 1.0 - snapshot.conflict_score),
            freshness=freshness_component,
            primary_weight=self.squeeze_config.confidence_primary_weight,
            context_weight=self.squeeze_config.confidence_context_weight,
            confirmation_weight=self.squeeze_config.confidence_confirmation_weight,
            freshness_weight=self.squeeze_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "oi_funding_squeeze_context",
            f"side:{payload.side.value}",
            f"crowded_side:{payload.crowded_side.value}",
            f"squeeze_side:{payload.squeeze_side.value}",
        ]

        conflicts = conflicting_source_names(snapshot.votes, payload.side)

        if oi_component >= self.squeeze_config.strong_oi_threshold:
            score += self.squeeze_config.strong_oi_bonus
            confirmations.append("strong_open_interest_context")

        if funding_component >= self.squeeze_config.strong_funding_threshold:
            score += self.squeeze_config.strong_funding_bonus
            confirmations.append("strong_funding_extreme")

        if squeeze_probability_component >= self.squeeze_config.strong_squeeze_probability_threshold:
            score += self.squeeze_config.squeeze_probability_bonus
            confirmations.append("squeeze_probability_confirmed")

        if (
            is_directional_side(payload.crowded_side)
            and payload.side == opposite_signal_side(payload.crowded_side)
        ):
            score += self.squeeze_config.opposite_crowding_bonus
            confirmations.append("opposite_crowded_side_squeeze")

        if (
            is_directional_side(payload.price_action_side)
            and payload.price_action_side == payload.side
        ):
            score += self.squeeze_config.price_action_confirmation_bonus
            confirmations.append("price_action_confirms_squeeze")

        if snapshot.conflict_score <= self.squeeze_config.low_conflict_threshold:
            score += self.squeeze_config.low_conflict_bonus
            confirmations.append("low_domain_conflict")

        if self._funding_is_extreme(funding):
            confirmations.append("funding_extreme_confirmed")

        reasons.extend(
            [
                f"open_interest_score:{oi_component:.4f}",
                f"funding_score:{funding_component:.4f}",
                f"squeeze_probability:{squeeze_probability_component:.4f}",
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

    def _funding_is_extreme(
        self,
        funding: dict[str, Any],
    ) -> bool:
        label = normalize_label(
            get_path(funding, "extreme")
            or get_path(funding, "is_extreme")
            or get_path(funding, "funding_extreme")
            or get_path(funding, "regime")
            or get_path(funding, "metadata.extreme")
            or get_path(funding, "metadata.regime")
        )

        if label in {
            "true",
            "extreme",
            "funding_extreme",
            "positive_extreme",
            "negative_extreme",
            "overheated",
            "crowded",
        }:
            return True

        return extract_domain_score(funding) >= self.squeeze_config.min_funding_extreme_score

    def _squeeze_probability(
        self,
        open_interest: dict[str, Any],
        funding: dict[str, Any],
    ) -> float:
        candidates = [
            get_path(open_interest, "squeeze_probability"),
            get_path(open_interest, "short_squeeze_probability"),
            get_path(open_interest, "long_squeeze_probability"),
            get_path(open_interest, "unwind_probability"),
            get_path(open_interest, "crowding_score"),
            get_path(open_interest, "metadata.squeeze_probability"),
            get_path(funding, "squeeze_probability"),
            get_path(funding, "unwind_probability"),
            get_path(funding, "crowding_score"),
            get_path(funding, "metadata.squeeze_probability"),
        ]

        for value in candidates:
            if value is not None:
                return unit_score(value)

        return average_score(
            extract_domain_score(open_interest),
            extract_domain_score(funding),
            default=0.0,
        )

    # ------------------------------------------------------------------
    # Source features / tags / metadata helpers
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: OIFundingSqueezePayload,
    ) -> list[str]:
        features = [
            *oi_funding_squeeze_source_features(),
            HYBRID_FEATURES.DOMINANT_SIDE,
            HYBRID_FEATURES.ALIGNMENT_SCORE,
            HYBRID_FEATURES.CONFLICT_SCORE,
            HYBRID_FEATURES.CONFLUENCE_SCORE,
            HYBRID_FEATURES.CONFIDENCE,
            HYBRID_FEATURES.VOTES,
            "open_interest.*",
            "funding.*",
            "price_action.*",
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: OIFundingSqueezePayload,
    ) -> list[str]:
        tags = [
            self.squeeze_config.tag_hybrid,
            self.squeeze_config.tag_oi_funding,
            self.squeeze_config.tag_oi_funding_squeeze,
            self.squeeze_config.tag_open_interest,
            self.squeeze_config.tag_funding_extreme,
            self.squeeze_config.tag_squeeze,
            self.squeeze_config.tag_crowded_positioning,
            f"side:{payload.side.value}",
            f"crowded_side:{payload.crowded_side.value}",
        ]

        if is_directional_side(payload.price_action_side):
            tags.append(self.squeeze_config.tag_price_action_confirmation)

        for source in payload.snapshot.aligned_domains:
            tags.append(f"aligned:{source}")

        return list(dict.fromkeys(tags))

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        return {
            "entry_offset_bps": self.squeeze_config.execution_entry_offset_bps_hint,
            "stop_buffer_bps": self.squeeze_config.execution_stop_buffer_bps_hint,
            "take_profit_bps": self.squeeze_config.execution_take_profit_bps_hint,
            "risk_reward": self.squeeze_config.squeeze_rr_hint,
        }