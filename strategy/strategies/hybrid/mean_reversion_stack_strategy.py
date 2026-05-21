# trading_system/strategy/strategies/hybrid/mean_reversion_stack_strategy.py

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
    mean_reversion_stack_source_features,
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
class MeanReversionStackPayload:
    """
    Normalized strategy-level payload для hybrid mean-reversion stack.

    Direction convention:
    - sweep/forced-flow/exhaustion down -> reversal LONG;
    - sweep/forced-flow/exhaustion up -> reversal SHORT.
    """

    snapshot: HybridCompositeSnapshot
    side: SignalSide

    sweep_side: SignalSide = SignalSide.UNKNOWN
    exhaustion_side: SignalSide = SignalSide.UNKNOWN
    rejection_side: SignalSide = SignalSide.UNKNOWN
    liquidation_side: SignalSide = SignalSide.UNKNOWN
    whale_side: SignalSide = SignalSide.UNKNOWN

    aligned_votes: list[DirectionVote] = field(default_factory=list)
    conflicting_votes: list[DirectionVote] = field(default_factory=list)

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MeanReversionStackStrategyConfig(HybridStrategyConfig):
    """
    Hybrid mean-reversion stack config.

    Strategy idea:
    - liquidity sweep / stop hunt creates forced-flow extension;
    - orderflow shows exhaustion or absorption;
    - price action confirms rejection;
    - liquidations / whales may strengthen reversal context;
    - strategy returns internal StrategySignal only.
    """

    min_reversion_score: float = 0.64
    min_reversion_confidence: float = 0.58

    min_liquidity_sweep_score: float = 0.58
    min_orderflow_exhaustion_score: float = 0.55
    min_price_rejection_score: float = 0.55
    min_liquidation_score: float = 0.50
    min_whale_absorption_score: float = 0.50

    require_liquidity: bool = True
    require_orderflow: bool = True
    require_price_action: bool = True
    require_liquidations: bool = False
    require_whales: bool = False

    use_liquidations_confirmation: bool = True
    use_whales_confirmation: bool = True

    require_sweep_side: bool = True
    require_orderflow_exhaustion: bool = True
    require_price_rejection: bool = True
    require_rejection_same_as_reversal: bool = True
    require_liquidation_same_as_sweep: bool = False
    require_whale_same_as_reversal: bool = False

    reject_same_side_momentum: bool = True
    reject_high_conflict: bool = True

    max_conflict_score: float = 0.40
    min_alignment_score: float = 0.55
    min_confluence_score: float = 0.55
    min_aligned_confirmations: int = 2

    liquidity_vote_weight: float = 1.20
    orderflow_vote_weight: float = 1.10
    price_action_vote_weight: float = 1.15
    liquidations_vote_weight: float = 0.95
    whales_vote_weight: float = 1.00

    score_liquidity_weight: float = 0.26
    score_orderflow_weight: float = 0.24
    score_price_action_weight: float = 0.20
    score_liquidations_weight: float = 0.10
    score_whales_weight: float = 0.10
    score_alignment_weight: float = 0.05
    score_freshness_weight: float = 0.05

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    strong_liquidity_bonus: float = 0.04
    strong_orderflow_bonus: float = 0.04
    price_rejection_bonus: float = 0.04
    liquidation_confirmation_bonus: float = 0.03
    whale_confirmation_bonus: float = 0.03
    low_conflict_bonus: float = 0.03

    strong_liquidity_threshold: float = 0.75
    strong_orderflow_threshold: float = 0.72
    strong_price_rejection_threshold: float = 0.72
    low_conflict_threshold: float = 0.15

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

    tag_mean_reversion_stack: str = "mean_reversion_stack"
    tag_liquidity_sweep: str = "liquidity_sweep"
    tag_orderflow_exhaustion: str = "orderflow_exhaustion"
    tag_price_rejection: str = "price_rejection"
    tag_liquidation_confirmation: str = "liquidation_confirmation"
    tag_whale_absorption: str = "whale_absorption"
    tag_reversal_stack: str = "reversal_stack"

    execution_entry_offset_bps_hint: float | None = None
    execution_stop_buffer_bps_hint: float | None = None
    execution_take_profit_bps_hint: float | None = None
    mean_reversion_rr_hint: float | None = None

    required_hybrid_features: tuple[str, ...] = (
        HYBRID_FEATURES.DOMINANT_SIDE,
        HYBRID_FEATURES.ALIGNMENT_SCORE,
        HYBRID_FEATURES.CONFLUENCE_SCORE,
    )

    def validate(self) -> None:
        HybridStrategyConfig.validate(self)

        unit_fields = {
            "min_reversion_score": self.min_reversion_score,
            "min_reversion_confidence": self.min_reversion_confidence,
            "min_liquidity_sweep_score": self.min_liquidity_sweep_score,
            "min_orderflow_exhaustion_score": self.min_orderflow_exhaustion_score,
            "min_price_rejection_score": self.min_price_rejection_score,
            "min_liquidation_score": self.min_liquidation_score,
            "min_whale_absorption_score": self.min_whale_absorption_score,
            "max_conflict_score": self.max_conflict_score,
            "min_alignment_score": self.min_alignment_score,
            "min_confluence_score": self.min_confluence_score,
            "strong_liquidity_bonus": self.strong_liquidity_bonus,
            "strong_orderflow_bonus": self.strong_orderflow_bonus,
            "price_rejection_bonus": self.price_rejection_bonus,
            "liquidation_confirmation_bonus": self.liquidation_confirmation_bonus,
            "whale_confirmation_bonus": self.whale_confirmation_bonus,
            "low_conflict_bonus": self.low_conflict_bonus,
            "strong_liquidity_threshold": self.strong_liquidity_threshold,
            "strong_orderflow_threshold": self.strong_orderflow_threshold,
            "strong_price_rejection_threshold": self.strong_price_rejection_threshold,
            "low_conflict_threshold": self.low_conflict_threshold,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.min_aligned_confirmations <= 0:
            raise StrategyConfigError("min_aligned_confirmations must be > 0")

        vote_weights = {
            "liquidity_vote_weight": self.liquidity_vote_weight,
            "orderflow_vote_weight": self.orderflow_vote_weight,
            "price_action_vote_weight": self.price_action_vote_weight,
            "liquidations_vote_weight": self.liquidations_vote_weight,
            "whales_vote_weight": self.whales_vote_weight,
        }
        score_weights = {
            "score_liquidity_weight": self.score_liquidity_weight,
            "score_orderflow_weight": self.score_orderflow_weight,
            "score_price_action_weight": self.score_price_action_weight,
            "score_liquidations_weight": self.score_liquidations_weight,
            "score_whales_weight": self.score_whales_weight,
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
            "mean_reversion_rr_hint": self.mean_reversion_rr_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        for attr in (
            "tag_mean_reversion_stack",
            "tag_liquidity_sweep",
            "tag_orderflow_exhaustion",
            "tag_price_rejection",
            "tag_liquidation_confirmation",
            "tag_whale_absorption",
            "tag_reversal_stack",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


class MeanReversionStackStrategy(HybridTradingStrategy):
    """
    Hybrid liquidity/orderflow/price-action mean-reversion stack.

    Input:
        StrategyContext with FeatureSource.LIQUIDITY, ORDERFLOW, PRICE_ACTION,
        and optional LIQUIDATIONS/WHALES domain sections.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    It does not duplicate SignalProcessor.ConfluenceEngine.
    SignalProcessor owns global routing, confluence, filters, portfolio coordination,
    building and risk payloads.
    """

    component_namespace = "strategy.hybrid.mean_reversion_stack"
    category: StrategyCategory = StrategyCategory.HYBRID
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        hybrid_config: MeanReversionStackStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_hybrid_config = hybrid_config or MeanReversionStackStrategyConfig()
        resolved_hybrid_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            hybrid_config=resolved_hybrid_config,
            service_name=service_name,
        )

        self.reversion_config: MeanReversionStackStrategyConfig = (
            resolved_hybrid_config
        )

    @property
    def strategy_name(self) -> str:
        return "mean_reversion_stack"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.HYBRID,
            timeframe=Timeframe.M1,
            tags=[
                self.reversion_config.tag_hybrid,
                self.reversion_config.tag_mean_reversion_stack,
                self.reversion_config.tag_reversal_stack,
                self.reversion_config.tag_liquidity_sweep,
                self.reversion_config.tag_orderflow_exhaustion,
                self.reversion_config.tag_price_rejection,
                "strategy_context",
            ],
            version="2.0.0",
            description=(
                "Builds a local mean-reversion stack signal from liquidity sweep, "
                "orderflow exhaustion, price-action rejection and optional "
                "liquidation/whale confirmation."
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
                "strategy_type": "mean_reversion_stack",
                "base_class": "HybridTradingStrategy",
                "canonical_payload": "HybridCompositeSnapshot",
                "uses_liquidity": True,
                "uses_orderflow": True,
                "uses_price_action": True,
                "uses_liquidations": True,
                "uses_whales": True,
                "duplicates_signal_processor_confluence": False,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(
            self.reversion_config.required_hybrid_features
        )

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
            allow_missing=self.reversion_config.allow_missing_required_domains,
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
            stale_after_seconds=self.reversion_config.stale_feature_max_age_seconds,
        ):
            return None

        if not self._passes_stack_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.reversion_config.min_reversion_score:
            return None

        if breakdown.confidence < self.reversion_config.min_reversion_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "mean_reversion_stack_signal",
                    f"side:{payload.side.value}",
                    f"sweep_side:{payload.sweep_side.value}",
                    f"exhaustion_side:{payload.exhaustion_side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "hybrid_setup_family": "mean_reversion_stack",
            "hybrid_strategy_version": "2.0.0",
            "contract": "hybrid",
            "contract_version": "strategy-domain-v1",
            "primary_section": "mean_reversion_stack",
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
            "sweep_side": payload.sweep_side.value,
            "exhaustion_side": payload.exhaustion_side.value,
            "rejection_side": payload.rejection_side.value,
            "liquidation_side": payload.liquidation_side.value,
            "whale_side": payload.whale_side.value,
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
            setup_type=self.reversion_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.reversion_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Sources / weights
    # ------------------------------------------------------------------

    def _enabled_sources(self) -> tuple[FeatureSource, ...]:
        sources: list[FeatureSource] = [
            FeatureSource.LIQUIDITY,
            FeatureSource.ORDERFLOW,
            FeatureSource.PRICE_ACTION,
        ]

        if self.reversion_config.use_liquidations_confirmation:
            sources.append(FeatureSource.LIQUIDATIONS)

        if self.reversion_config.use_whales_confirmation:
            sources.append(FeatureSource.WHALES)

        return tuple(dict.fromkeys(sources))

    def _required_sources(self) -> tuple[FeatureSource, ...]:
        sources: list[FeatureSource] = []

        if self.reversion_config.require_liquidity:
            sources.append(FeatureSource.LIQUIDITY)

        if self.reversion_config.require_orderflow:
            sources.append(FeatureSource.ORDERFLOW)

        if self.reversion_config.require_price_action:
            sources.append(FeatureSource.PRICE_ACTION)

        if self.reversion_config.require_liquidations:
            sources.append(FeatureSource.LIQUIDATIONS)

        if self.reversion_config.require_whales:
            sources.append(FeatureSource.WHALES)

        return tuple(dict.fromkeys(sources))

    def _vote_weights(self) -> dict[FeatureSource, float]:
        return {
            FeatureSource.LIQUIDITY: self.reversion_config.liquidity_vote_weight,
            FeatureSource.ORDERFLOW: self.reversion_config.orderflow_vote_weight,
            FeatureSource.PRICE_ACTION: self.reversion_config.price_action_vote_weight,
            FeatureSource.LIQUIDATIONS: self.reversion_config.liquidations_vote_weight,
            FeatureSource.WHALES: self.reversion_config.whales_vote_weight,
        }

    def _has_minimum_available_domains(
        self,
        context: StrategyContext,
        sources: tuple[FeatureSource, ...],
    ) -> bool:
        available_count = len(self.available_domain_sources(context, sources))
        return available_count >= self.reversion_config.min_required_domains

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
    ) -> MeanReversionStackPayload | None:
        payloads = snapshot.payloads()

        liquidity = payloads.get(FeatureSource.LIQUIDITY, {})
        orderflow = payloads.get(FeatureSource.ORDERFLOW, {})
        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})
        liquidations = payloads.get(FeatureSource.LIQUIDATIONS, {})
        whales = payloads.get(FeatureSource.WHALES, {})

        sweep_side = self._extract_sweep_side(liquidity)
        if self.reversion_config.require_sweep_side and not is_directional_side(sweep_side):
            return None

        side = opposite_signal_side(sweep_side)
        if not is_directional_side(side):
            side = self._extract_reversal_side(price_action, orderflow, whales)

        if not is_directional_side(side):
            return None

        exhaustion_side = self._extract_exhaustion_side(orderflow)
        rejection_side = self._extract_rejection_side(price_action)
        liquidation_side = self._extract_liquidation_side(liquidations)
        whale_side = self._extract_whale_side(whales)

        aligned_votes = votes_for_side(snapshot.votes, side)
        conflicting_votes = votes_against_side(snapshot.votes, side)

        event_time = (
            latest_timestamp_from_payloads(payloads, fallback=context.timestamp)
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "mean_reversion_stack_context",
            f"side:{side.value}",
            f"sweep_side:{sweep_side.value}",
            f"exhaustion_side:{exhaustion_side.value}",
            f"rejection_side:{rejection_side.value}",
            f"liquidation_side:{liquidation_side.value}",
            f"whale_side:{whale_side.value}",
            f"alignment_score:{snapshot.alignment_score:.4f}",
            f"conflict_score:{snapshot.conflict_score:.4f}",
            f"confluence_score:{snapshot.confluence_score:.4f}",
        ]

        return MeanReversionStackPayload(
            snapshot=snapshot,
            side=side,
            sweep_side=sweep_side,
            exhaustion_side=exhaustion_side,
            rejection_side=rejection_side,
            liquidation_side=liquidation_side,
            whale_side=whale_side,
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
            "metadata.side",
        ):
            side = side_to_signal_side(get_path(payload, path))
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    def _extract_exhaustion_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "exhaustion_side",
            "absorbed_side",
            "pressure_side",
            "delta_side",
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

    def _extract_liquidation_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "liquidation_side",
            "liquidated_side",
            "cascade_side",
            "forced_flow_side",
            "side",
            "direction",
            "metadata.liquidation_side",
            "metadata.side",
        ):
            side = side_to_signal_side(get_path(payload, path))
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    def _extract_whale_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "whale_side",
            "absorption_side",
            "dominant_side",
            "cluster_side",
            "side",
            "direction",
            "metadata.whale_side",
            "metadata.side",
        ):
            side = side_to_signal_side(get_path(payload, path))
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    def _extract_reversal_side(
        self,
        price_action: dict[str, Any],
        orderflow: dict[str, Any],
        whales: dict[str, Any],
    ) -> SignalSide:
        for payload, extractor in (
            (price_action, self._extract_rejection_side),
            (whales, self._extract_whale_side),
        ):
            side = extractor(payload)
            if is_directional_side(side):
                return side

        exhaustion_side = self._extract_exhaustion_side(orderflow)
        if is_directional_side(exhaustion_side):
            return opposite_signal_side(exhaustion_side)

        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_stack_filters(
        self,
        payload: MeanReversionStackPayload,
    ) -> bool:
        snapshot = payload.snapshot
        payloads = snapshot.payloads()

        liquidity = payloads.get(FeatureSource.LIQUIDITY, {})
        orderflow = payloads.get(FeatureSource.ORDERFLOW, {})
        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})
        liquidations = payloads.get(FeatureSource.LIQUIDATIONS, {})
        whales = payloads.get(FeatureSource.WHALES, {})

        if self.reversion_config.reject_high_conflict:
            if snapshot.conflict_score > self.reversion_config.max_conflict_score:
                return False

        if snapshot.alignment_score < self.reversion_config.min_alignment_score:
            return False

        if snapshot.confluence_score < self.reversion_config.min_confluence_score:
            return False

        if extract_domain_score(liquidity) < self.reversion_config.min_liquidity_sweep_score:
            return False

        if self.reversion_config.require_orderflow_exhaustion:
            if extract_domain_score(orderflow) < self.reversion_config.min_orderflow_exhaustion_score:
                return False

            if is_directional_side(payload.exhaustion_side):
                if payload.exhaustion_side != payload.sweep_side:
                    return False

        if self.reversion_config.require_price_rejection:
            if extract_domain_score(price_action) < self.reversion_config.min_price_rejection_score:
                return False

            if self.reversion_config.require_rejection_same_as_reversal:
                if is_directional_side(payload.rejection_side):
                    if payload.rejection_side != payload.side:
                        return False

        if self.reversion_config.use_liquidations_confirmation and liquidations:
            liquidation_score = extract_domain_score(liquidations)
            if liquidation_score > 0.0 and liquidation_score < self.reversion_config.min_liquidation_score:
                return False

            if self.reversion_config.require_liquidation_same_as_sweep:
                if is_directional_side(payload.liquidation_side):
                    if payload.liquidation_side != payload.sweep_side:
                        return False

        if self.reversion_config.use_whales_confirmation and whales:
            whale_score = extract_domain_score(whales)
            if whale_score > 0.0 and whale_score < self.reversion_config.min_whale_absorption_score:
                return False

            if self.reversion_config.require_whale_same_as_reversal:
                if is_directional_side(payload.whale_side):
                    if payload.whale_side != payload.side:
                        return False

        if self.reversion_config.reject_same_side_momentum:
            if is_directional_side(payload.exhaustion_side):
                if payload.exhaustion_side == payload.side:
                    return False

            if is_directional_side(payload.sweep_side):
                if payload.sweep_side == payload.side:
                    return False

        aligned_confirmations = self._aligned_confirmation_count(payload)
        if aligned_confirmations < self.reversion_config.min_aligned_confirmations:
            return False

        return True

    def _aligned_confirmation_count(
        self,
        payload: MeanReversionStackPayload,
    ) -> int:
        count = 0

        if is_directional_side(payload.sweep_side) and payload.sweep_side == opposite_signal_side(payload.side):
            count += 1

        if is_directional_side(payload.exhaustion_side) and payload.exhaustion_side == opposite_signal_side(payload.side):
            count += 1

        if is_directional_side(payload.rejection_side) and payload.rejection_side == payload.side:
            count += 1

        if is_directional_side(payload.liquidation_side) and payload.liquidation_side == opposite_signal_side(payload.side):
            count += 1

        if is_directional_side(payload.whale_side) and payload.whale_side == payload.side:
            count += 1

        return count

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: MeanReversionStackPayload,
    ) -> HybridScoreBreakdown:
        snapshot = payload.snapshot
        payloads = snapshot.payloads()

        liquidity = payloads.get(FeatureSource.LIQUIDITY, {})
        orderflow = payloads.get(FeatureSource.ORDERFLOW, {})
        price_action = payloads.get(FeatureSource.PRICE_ACTION, {})
        liquidations = payloads.get(FeatureSource.LIQUIDATIONS, {})
        whales = payloads.get(FeatureSource.WHALES, {})

        liquidity_component = extract_domain_score(liquidity)
        orderflow_component = extract_domain_score(orderflow)
        price_action_component = extract_domain_score(price_action)
        liquidations_component = (
            extract_domain_score(liquidations)
            if liquidations and self.reversion_config.use_liquidations_confirmation
            else 0.0
        )
        whales_component = (
            extract_domain_score(whales)
            if whales and self.reversion_config.use_whales_confirmation
            else 0.0
        )
        alignment_component = snapshot.alignment_score
        freshness_component = hybrid_freshness_score(
            payloads,
            now=context.timestamp,
            stale_after_seconds=self.reversion_config.stale_feature_max_age_seconds,
        )

        components = {
            "liquidity": liquidity_component,
            "orderflow": orderflow_component,
            "price_action": price_action_component,
            "liquidations": liquidations_component,
            "whales": whales_component,
            "alignment": alignment_component,
            "freshness": freshness_component,
        }
        weights = {
            "liquidity": self.reversion_config.score_liquidity_weight,
            "orderflow": self.reversion_config.score_orderflow_weight,
            "price_action": self.reversion_config.score_price_action_weight,
            "liquidations": self.reversion_config.score_liquidations_weight,
            "whales": self.reversion_config.score_whales_weight,
            "alignment": self.reversion_config.score_alignment_weight,
            "freshness": self.reversion_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=liquidity_component)
        confidence = confidence_from_components(
            primary=average_score(liquidity_component, orderflow_component),
            context=average_score(price_action_component, snapshot.confluence_score),
            confirmation=average_score(liquidations_component, whales_component),
            freshness=freshness_component,
            primary_weight=self.reversion_config.confidence_primary_weight,
            context_weight=self.reversion_config.confidence_context_weight,
            confirmation_weight=self.reversion_config.confidence_confirmation_weight,
            freshness_weight=self.reversion_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "mean_reversion_stack_context",
            f"side:{payload.side.value}",
            f"sweep_side:{payload.sweep_side.value}",
            f"exhaustion_side:{payload.exhaustion_side.value}",
        ]

        conflicts = conflicting_source_names(snapshot.votes, payload.side)

        if liquidity_component >= self.reversion_config.strong_liquidity_threshold:
            score += self.reversion_config.strong_liquidity_bonus
            confirmations.append("strong_liquidity_sweep")

        if orderflow_component >= self.reversion_config.strong_orderflow_threshold:
            score += self.reversion_config.strong_orderflow_bonus
            confirmations.append("strong_orderflow_exhaustion")

        if price_action_component >= self.reversion_config.strong_price_rejection_threshold:
            score += self.reversion_config.price_rejection_bonus
            confirmations.append("price_action_rejection_confirmed")

        if liquidations_component >= self.reversion_config.min_liquidation_score:
            score += self.reversion_config.liquidation_confirmation_bonus
            confirmations.append("liquidations_confirm_reversal")

        if whales_component >= self.reversion_config.min_whale_absorption_score:
            score += self.reversion_config.whale_confirmation_bonus
            confirmations.append("whales_confirm_reversal")

        if snapshot.conflict_score <= self.reversion_config.low_conflict_threshold:
            score += self.reversion_config.low_conflict_bonus
            confirmations.append("low_domain_conflict")

        if is_directional_side(payload.rejection_side) and payload.rejection_side == payload.side:
            confirmations.append("rejection_same_as_reversal_side")

        if is_directional_side(payload.whale_side) and payload.whale_side == payload.side:
            confirmations.append("whale_absorption_same_as_reversal_side")

        if is_directional_side(payload.liquidation_side) and payload.liquidation_side == opposite_signal_side(payload.side):
            confirmations.append("liquidation_same_as_sweep_side")

        reasons.extend(
            [
                f"liquidity_score:{liquidity_component:.4f}",
                f"orderflow_score:{orderflow_component:.4f}",
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

    # ------------------------------------------------------------------
    # Source features / tags / metadata helpers
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: MeanReversionStackPayload,
    ) -> list[str]:
        features = [
            *mean_reversion_stack_source_features(),
            HYBRID_FEATURES.DOMINANT_SIDE,
            HYBRID_FEATURES.ALIGNMENT_SCORE,
            HYBRID_FEATURES.CONFLICT_SCORE,
            HYBRID_FEATURES.CONFLUENCE_SCORE,
            HYBRID_FEATURES.CONFIDENCE,
            HYBRID_FEATURES.VOTES,
            "liquidity.*",
            "orderflow.*",
            "price_action.*",
            "liquidations.*",
            "whales.*",
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: MeanReversionStackPayload,
    ) -> list[str]:
        tags = [
            self.reversion_config.tag_hybrid,
            self.reversion_config.tag_mean_reversion_stack,
            self.reversion_config.tag_reversal_stack,
            self.reversion_config.tag_liquidity_sweep,
            self.reversion_config.tag_orderflow_exhaustion,
            self.reversion_config.tag_price_rejection,
            f"side:{payload.side.value}",
            f"sweep_side:{payload.sweep_side.value}",
        ]

        if is_directional_side(payload.liquidation_side):
            tags.append(self.reversion_config.tag_liquidation_confirmation)

        if is_directional_side(payload.whale_side):
            tags.append(self.reversion_config.tag_whale_absorption)

        for source in payload.snapshot.aligned_domains:
            tags.append(f"aligned:{source}")

        return list(dict.fromkeys(tags))

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        return {
            "entry_offset_bps": self.reversion_config.execution_entry_offset_bps_hint,
            "stop_buffer_bps": self.reversion_config.execution_stop_buffer_bps_hint,
            "take_profit_bps": self.reversion_config.execution_take_profit_bps_hint,
            "risk_reward": self.reversion_config.mean_reversion_rr_hint,
        }