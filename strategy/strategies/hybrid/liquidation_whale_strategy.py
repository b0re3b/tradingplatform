# trading_system/strategy/strategies/hybrid/liquidation_whale_strategy.py

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
    LIQUIDATION_WHALE_SOURCES,
    aligned_source_names,
    average_score,
    confidence_from_components,
    conflicting_source_names,
    extract_domain_score,
    get_path,
    hybrid_freshness_score,
    is_directional_side,
    is_stale,
    latest_timestamp_from_payloads,
    liquidation_whale_source_features,
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
class LiquidationWhalePayload:
    """
    Normalized strategy-level payload для liquidation + whale reversal.

    Direction convention:
    - sell-side liquidation cascade / forced flow + buy-side whale absorption -> LONG;
    - buy-side liquidation cascade / forced flow + sell-side whale absorption -> SHORT.
    """

    snapshot: HybridCompositeSnapshot
    side: SignalSide

    liquidation_side: SignalSide = SignalSide.UNKNOWN
    whale_side: SignalSide = SignalSide.UNKNOWN
    exhausted_side: SignalSide = SignalSide.UNKNOWN

    aligned_votes: list[DirectionVote] = field(default_factory=list)
    conflicting_votes: list[DirectionVote] = field(default_factory=list)

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LiquidationWhaleStrategyConfig(HybridStrategyConfig):
    """
    Hybrid liquidation + whale reversal config.

    Strategy idea:
    - liquidations domain identifies forced-flow / liquidation side;
    - whales domain confirms opposite-side absorption / cluster support;
    - reversal side is opposite to liquidation side;
    - strategy returns internal StrategySignal only.

    This is stricter than whales.WhaleLiquidationReversalStrategy because both
    LIQUIDATIONS and WHALES domains are required by default.
    """

    min_liquidation_whale_score: float = 0.66
    min_liquidation_whale_confidence: float = 0.60

    min_liquidation_score: float = 0.60
    min_whale_score: float = 0.60
    min_absorption_score: float = 0.55
    min_exhaustion_score: float = 0.50

    require_liquidations: bool = True
    require_whales: bool = True
    require_liquidation_side: bool = True
    require_whale_side: bool = True

    require_opposite_forced_flow: bool = True
    require_whale_same_as_reversal: bool = True
    require_exhaustion_same_as_liquidation: bool = False

    reject_same_side_whale_pressure: bool = True
    reject_high_conflict: bool = True

    max_conflict_score: float = 0.35
    min_alignment_score: float = 0.52
    min_confluence_score: float = 0.52

    liquidations_vote_weight: float = 1.25
    whales_vote_weight: float = 1.20

    score_liquidations_weight: float = 0.36
    score_whales_weight: float = 0.32
    score_absorption_weight: float = 0.12
    score_exhaustion_weight: float = 0.08
    score_alignment_weight: float = 0.06
    score_freshness_weight: float = 0.06

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    strong_liquidation_bonus: float = 0.05
    strong_whale_bonus: float = 0.05
    absorption_confirmation_bonus: float = 0.04
    exhaustion_confirmation_bonus: float = 0.03
    opposite_flow_bonus: float = 0.05
    low_conflict_bonus: float = 0.03

    strong_liquidation_threshold: float = 0.75
    strong_whale_threshold: float = 0.72
    strong_absorption_threshold: float = 0.70
    strong_exhaustion_threshold: float = 0.65
    low_conflict_threshold: float = 0.12

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.REVERSAL

    tag_liquidation_whale: str = "liquidation_whale"
    tag_forced_flow_reversal: str = "forced_flow_reversal"
    tag_liquidation_cascade: str = "liquidation_cascade"
    tag_whale_absorption: str = "whale_absorption"
    tag_opposite_flow: str = "opposite_forced_flow"
    tag_exhaustion_confirmed: str = "exhaustion_confirmed"

    execution_entry_offset_bps_hint: float | None = None
    execution_stop_buffer_bps_hint: float | None = None
    execution_take_profit_bps_hint: float | None = None
    liquidation_whale_rr_hint: float | None = None

    required_hybrid_features: tuple[str, ...] = (
        HYBRID_FEATURES.DOMINANT_SIDE,
        HYBRID_FEATURES.ALIGNMENT_SCORE,
        HYBRID_FEATURES.CONFLUENCE_SCORE,
    )

    def validate(self) -> None:
        HybridStrategyConfig.validate(self)

        unit_fields = {
            "min_liquidation_whale_score": self.min_liquidation_whale_score,
            "min_liquidation_whale_confidence": self.min_liquidation_whale_confidence,
            "min_liquidation_score": self.min_liquidation_score,
            "min_whale_score": self.min_whale_score,
            "min_absorption_score": self.min_absorption_score,
            "min_exhaustion_score": self.min_exhaustion_score,
            "max_conflict_score": self.max_conflict_score,
            "min_alignment_score": self.min_alignment_score,
            "min_confluence_score": self.min_confluence_score,
            "strong_liquidation_bonus": self.strong_liquidation_bonus,
            "strong_whale_bonus": self.strong_whale_bonus,
            "absorption_confirmation_bonus": self.absorption_confirmation_bonus,
            "exhaustion_confirmation_bonus": self.exhaustion_confirmation_bonus,
            "opposite_flow_bonus": self.opposite_flow_bonus,
            "low_conflict_bonus": self.low_conflict_bonus,
            "strong_liquidation_threshold": self.strong_liquidation_threshold,
            "strong_whale_threshold": self.strong_whale_threshold,
            "strong_absorption_threshold": self.strong_absorption_threshold,
            "strong_exhaustion_threshold": self.strong_exhaustion_threshold,
            "low_conflict_threshold": self.low_conflict_threshold,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        vote_weights = {
            "liquidations_vote_weight": self.liquidations_vote_weight,
            "whales_vote_weight": self.whales_vote_weight,
        }
        score_weights = {
            "score_liquidations_weight": self.score_liquidations_weight,
            "score_whales_weight": self.score_whales_weight,
            "score_absorption_weight": self.score_absorption_weight,
            "score_exhaustion_weight": self.score_exhaustion_weight,
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
            "liquidation_whale_rr_hint": self.liquidation_whale_rr_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        for attr in (
            "tag_liquidation_whale",
            "tag_forced_flow_reversal",
            "tag_liquidation_cascade",
            "tag_whale_absorption",
            "tag_opposite_flow",
            "tag_exhaustion_confirmed",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


class LiquidationWhaleStrategy(HybridTradingStrategy):
    """
    Hybrid liquidation + whale reversal strategy.

    Input:
        StrategyContext with FeatureSource.LIQUIDATIONS and FeatureSource.WHALES.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    It does not duplicate SignalProcessor.ConfluenceEngine.
    SignalProcessor owns global routing, confluence, filters, portfolio coordination,
    building and risk payloads.
    """

    component_namespace = "strategy.hybrid.liquidation_whale"
    category: StrategyCategory = StrategyCategory.HYBRID
    default_setup_type: SetupType = SetupType.REVERSAL

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        hybrid_config: LiquidationWhaleStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_hybrid_config = hybrid_config or LiquidationWhaleStrategyConfig()
        resolved_hybrid_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            hybrid_config=resolved_hybrid_config,
            service_name=service_name,
        )

        self.liq_whale_config: LiquidationWhaleStrategyConfig = resolved_hybrid_config

    @property
    def strategy_name(self) -> str:
        return "liquidation_whale"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.HYBRID,
            timeframe=Timeframe.M1,
            tags=[
                self.liq_whale_config.tag_hybrid,
                self.liq_whale_config.tag_liquidation_whale,
                self.liq_whale_config.tag_forced_flow_reversal,
                self.liq_whale_config.tag_liquidation_cascade,
                self.liq_whale_config.tag_whale_absorption,
                "strategy_context",
            ],
            version="2.0.0",
            description=(
                "Builds a specialized liquidation + whale reversal signal from "
                "liquidation forced-flow and opposite whale absorption."
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
                "strategy_type": "liquidation_whale",
                "base_class": "HybridTradingStrategy",
                "canonical_payload": "HybridCompositeSnapshot",
                "uses_liquidations": True,
                "uses_whales": True,
                "requires_both_domains": True,
                "duplicates_signal_processor_confluence": False,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(
            self.liq_whale_config.required_hybrid_features
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
            allow_missing=self.liq_whale_config.allow_missing_required_domains,
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
            stale_after_seconds=self.liq_whale_config.stale_feature_max_age_seconds,
        ):
            return None

        if not self._passes_liquidation_whale_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.liq_whale_config.min_liquidation_whale_score:
            return None

        if breakdown.confidence < self.liq_whale_config.min_liquidation_whale_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "liquidation_whale_signal",
                    f"side:{payload.side.value}",
                    f"liquidation_side:{payload.liquidation_side.value}",
                    f"whale_side:{payload.whale_side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "hybrid_setup_family": "liquidation_whale",
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
            "liquidation_side": payload.liquidation_side.value,
            "whale_side": payload.whale_side.value,
            "exhausted_side": payload.exhausted_side.value,
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
            setup_type=self.liq_whale_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.liq_whale_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Sources / weights
    # ------------------------------------------------------------------

    def _enabled_sources(self) -> tuple[FeatureSource, ...]:
        return LIQUIDATION_WHALE_SOURCES

    def _required_sources(self) -> tuple[FeatureSource, ...]:
        sources: list[FeatureSource] = []

        if self.liq_whale_config.require_liquidations:
            sources.append(FeatureSource.LIQUIDATIONS)

        if self.liq_whale_config.require_whales:
            sources.append(FeatureSource.WHALES)

        return tuple(dict.fromkeys(sources))

    def _vote_weights(self) -> dict[FeatureSource, float]:
        return {
            FeatureSource.LIQUIDATIONS: self.liq_whale_config.liquidations_vote_weight,
            FeatureSource.WHALES: self.liq_whale_config.whales_vote_weight,
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
    ) -> LiquidationWhalePayload | None:
        payloads = snapshot.payloads()
        liquidations = payloads.get(FeatureSource.LIQUIDATIONS, {})
        whales = payloads.get(FeatureSource.WHALES, {})

        if not liquidations or not whales:
            return None

        liquidation_side = self._extract_liquidation_side(liquidations)
        if self.liq_whale_config.require_liquidation_side and not is_directional_side(liquidation_side):
            return None

        whale_side = self._extract_whale_side(whales)
        if self.liq_whale_config.require_whale_side and not is_directional_side(whale_side):
            return None

        exhausted_side = self._extract_exhausted_side(whales, liquidations)

        side = opposite_signal_side(liquidation_side)
        if not is_directional_side(side):
            side = whale_side

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
            "liquidation_whale_context",
            f"side:{side.value}",
            f"liquidation_side:{liquidation_side.value}",
            f"whale_side:{whale_side.value}",
            f"exhausted_side:{exhausted_side.value}",
            f"alignment_score:{snapshot.alignment_score:.4f}",
            f"conflict_score:{snapshot.conflict_score:.4f}",
            f"confluence_score:{snapshot.confluence_score:.4f}",
        ]

        return LiquidationWhalePayload(
            snapshot=snapshot,
            side=side,
            liquidation_side=liquidation_side,
            whale_side=whale_side,
            exhausted_side=exhausted_side,
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

    def _extract_liquidation_side(self, payload: dict[str, Any]) -> SignalSide:
        for path in (
            "liquidation_side",
            "liquidated_side",
            "cascade_side",
            "forced_flow_side",
            "exhausted_side",
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
            "reversal_side",
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

    def _extract_exhausted_side(
        self,
        whales: dict[str, Any],
        liquidations: dict[str, Any],
    ) -> SignalSide:
        for payload in (whales, liquidations):
            for path in (
                "exhausted_side",
                "liquidation_side",
                "liquidated_side",
                "forced_flow_side",
                "side",
                "metadata.exhausted_side",
                "metadata.liquidation_side",
            ):
                side = side_to_signal_side(get_path(payload, path))
                if is_directional_side(side):
                    return side
        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_liquidation_whale_filters(
        self,
        payload: LiquidationWhalePayload,
    ) -> bool:
        snapshot = payload.snapshot
        payloads = snapshot.payloads()

        liquidations = payloads.get(FeatureSource.LIQUIDATIONS, {})
        whales = payloads.get(FeatureSource.WHALES, {})

        if snapshot.domain_count < self.liq_whale_config.min_required_domains:
            return False

        if self.liq_whale_config.reject_high_conflict:
            if snapshot.conflict_score > self.liq_whale_config.max_conflict_score:
                return False

        if snapshot.alignment_score < self.liq_whale_config.min_alignment_score:
            return False

        if snapshot.confluence_score < self.liq_whale_config.min_confluence_score:
            return False

        if extract_domain_score(liquidations) < self.liq_whale_config.min_liquidation_score:
            return False

        if extract_domain_score(whales) < self.liq_whale_config.min_whale_score:
            return False

        if self.liq_whale_config.require_opposite_forced_flow:
            if payload.side != opposite_signal_side(payload.liquidation_side):
                return False

        if self.liq_whale_config.require_whale_same_as_reversal:
            if payload.whale_side != payload.side:
                return False

        if self.liq_whale_config.reject_same_side_whale_pressure:
            if payload.whale_side == payload.liquidation_side:
                return False

        if self.liq_whale_config.require_exhaustion_same_as_liquidation:
            if is_directional_side(payload.exhausted_side):
                if payload.exhausted_side != payload.liquidation_side:
                    return False
            else:
                return False

        absorption_score = self._absorption_score(whales)
        if absorption_score > 0.0 and absorption_score < self.liq_whale_config.min_absorption_score:
            return False

        exhaustion_score = self._exhaustion_score(liquidations, whales)
        if exhaustion_score > 0.0 and exhaustion_score < self.liq_whale_config.min_exhaustion_score:
            return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: LiquidationWhalePayload,
    ) -> HybridScoreBreakdown:
        snapshot = payload.snapshot
        payloads = snapshot.payloads()

        liquidations = payloads.get(FeatureSource.LIQUIDATIONS, {})
        whales = payloads.get(FeatureSource.WHALES, {})

        liquidations_component = extract_domain_score(liquidations)
        whales_component = extract_domain_score(whales)
        absorption_component = self._absorption_score(whales)
        exhaustion_component = self._exhaustion_score(liquidations, whales)
        alignment_component = snapshot.alignment_score
        freshness_component = hybrid_freshness_score(
            payloads,
            now=context.timestamp,
            stale_after_seconds=self.liq_whale_config.stale_feature_max_age_seconds,
        )

        components = {
            "liquidations": liquidations_component,
            "whales": whales_component,
            "absorption": absorption_component,
            "exhaustion": exhaustion_component,
            "alignment": alignment_component,
            "freshness": freshness_component,
        }
        weights = {
            "liquidations": self.liq_whale_config.score_liquidations_weight,
            "whales": self.liq_whale_config.score_whales_weight,
            "absorption": self.liq_whale_config.score_absorption_weight,
            "exhaustion": self.liq_whale_config.score_exhaustion_weight,
            "alignment": self.liq_whale_config.score_alignment_weight,
            "freshness": self.liq_whale_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=liquidations_component)
        confidence = confidence_from_components(
            primary=average_score(liquidations_component, whales_component),
            context=average_score(absorption_component, exhaustion_component),
            confirmation=average_score(snapshot.alignment_score, 1.0 - snapshot.conflict_score),
            freshness=freshness_component,
            primary_weight=self.liq_whale_config.confidence_primary_weight,
            context_weight=self.liq_whale_config.confidence_context_weight,
            confirmation_weight=self.liq_whale_config.confidence_confirmation_weight,
            freshness_weight=self.liq_whale_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "liquidation_whale_context",
            f"side:{payload.side.value}",
            f"liquidation_side:{payload.liquidation_side.value}",
            f"whale_side:{payload.whale_side.value}",
        ]

        conflicts = conflicting_source_names(snapshot.votes, payload.side)

        if liquidations_component >= self.liq_whale_config.strong_liquidation_threshold:
            score += self.liq_whale_config.strong_liquidation_bonus
            confirmations.append("strong_liquidation_cascade")

        if whales_component >= self.liq_whale_config.strong_whale_threshold:
            score += self.liq_whale_config.strong_whale_bonus
            confirmations.append("strong_whale_context")

        if absorption_component >= self.liq_whale_config.strong_absorption_threshold:
            score += self.liq_whale_config.absorption_confirmation_bonus
            confirmations.append("whale_absorption_confirmed")

        if exhaustion_component >= self.liq_whale_config.strong_exhaustion_threshold:
            score += self.liq_whale_config.exhaustion_confirmation_bonus
            confirmations.append("forced_flow_exhaustion_confirmed")

        if payload.side == opposite_signal_side(payload.liquidation_side):
            score += self.liq_whale_config.opposite_flow_bonus
            confirmations.append("opposite_forced_flow_reversal")

        if snapshot.conflict_score <= self.liq_whale_config.low_conflict_threshold:
            score += self.liq_whale_config.low_conflict_bonus
            confirmations.append("low_domain_conflict")

        if payload.whale_side == payload.side:
            confirmations.append("whale_same_as_reversal_side")

        if payload.exhausted_side == payload.liquidation_side:
            confirmations.append("exhausted_side_matches_liquidation_side")

        reasons.extend(
            [
                f"liquidations_score:{liquidations_component:.4f}",
                f"whales_score:{whales_component:.4f}",
                f"absorption_score:{absorption_component:.4f}",
                f"exhaustion_score:{exhaustion_component:.4f}",
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
        whales: dict[str, Any],
    ) -> float:
        candidates = [
            get_path(whales, "absorption_score"),
            get_path(whales, "whale_absorption_score"),
            get_path(whales, "context_strength"),
            get_path(whales, "pressure_score"),
            get_path(whales, "score"),
            get_path(whales, "metadata.absorption_score"),
            get_path(whales, "metadata.context_strength"),
        ]

        for value in candidates:
            if value is not None:
                return unit_score(value)

        return 0.0

    def _exhaustion_score(
        self,
        liquidations: dict[str, Any],
        whales: dict[str, Any],
    ) -> float:
        candidates = [
            get_path(liquidations, "exhaustion_score"),
            get_path(liquidations, "exhaustion_probability"),
            get_path(liquidations, "cascade_exhaustion_score"),
            get_path(whales, "exhaustion_score"),
            get_path(whales, "exhaustion_probability"),
            get_path(whales, "metadata.exhaustion_score"),
            get_path(liquidations, "metadata.exhaustion_score"),
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
        payload: LiquidationWhalePayload,
    ) -> list[str]:
        features = [
            *liquidation_whale_source_features(),
            HYBRID_FEATURES.DOMINANT_SIDE,
            HYBRID_FEATURES.ALIGNMENT_SCORE,
            HYBRID_FEATURES.CONFLICT_SCORE,
            HYBRID_FEATURES.CONFLUENCE_SCORE,
            HYBRID_FEATURES.CONFIDENCE,
            HYBRID_FEATURES.VOTES,
            "liquidations.*",
            "whales.*",
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: LiquidationWhalePayload,
    ) -> list[str]:
        tags = [
            self.liq_whale_config.tag_hybrid,
            self.liq_whale_config.tag_liquidation_whale,
            self.liq_whale_config.tag_forced_flow_reversal,
            self.liq_whale_config.tag_liquidation_cascade,
            self.liq_whale_config.tag_whale_absorption,
            f"side:{payload.side.value}",
            f"liquidation_side:{payload.liquidation_side.value}",
            f"whale_side:{payload.whale_side.value}",
        ]

        if payload.side == opposite_signal_side(payload.liquidation_side):
            tags.append(self.liq_whale_config.tag_opposite_flow)

        if payload.exhausted_side == payload.liquidation_side:
            tags.append(self.liq_whale_config.tag_exhaustion_confirmed)

        for source in payload.snapshot.aligned_domains:
            tags.append(f"aligned:{source}")

        return list(dict.fromkeys(tags))

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        return {
            "entry_offset_bps": self.liq_whale_config.execution_entry_offset_bps_hint,
            "stop_buffer_bps": self.liq_whale_config.execution_stop_buffer_bps_hint,
            "take_profit_bps": self.liq_whale_config.execution_take_profit_bps_hint,
            "risk_reward": self.liq_whale_config.liquidation_whale_rr_hint,
        }