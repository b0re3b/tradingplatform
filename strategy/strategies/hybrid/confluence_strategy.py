# trading_system/strategy/strategies/hybrid/confluence_strategy.py

from __future__ import annotations
import logging

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
    aligned_source_names,
    available_domain_sources,
    confluence_source_features,
    hybrid_freshness_score,
    is_directional_side,
    is_stale,
    latest_timestamp_from_payloads,
    serialize_for_metadata,
    strong_votes,
    unit_score,
    votes_against_side,
    votes_for_side,
    weighted_score, )
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
class ConfluencePayload:
    """
    Normalized strategy-level payload для generic hybrid confluence.

    Це локальний payload конкретної стратегії, не глобальний ConfluenceEngine.
    """
    _logger = logging.getLogger(__name__ + ".ConfluencePayload")

    snapshot: HybridCompositeSnapshot
    side: SignalSide
    votes: list[DirectionVote]

    aligned_votes: list[DirectionVote] = field(default_factory=list)
    conflicting_votes: list[DirectionVote] = field(default_factory=list)

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConfluenceStrategyConfig(HybridStrategyConfig):
    """
    Generic hybrid confluence strategy config.

    Strategy idea:
    - read several already-normalized domain sections from StrategyContext;
    - build local direction votes;
    - require enough same-side domain alignment;
    - reject direct conflicts if configured;
    - return internal StrategySignal only.

    This class is not SignalProcessor.ConfluenceEngine.
    """
    _logger = logging.getLogger(__name__ + ".ConfluenceStrategyConfig")

    min_confluence_strategy_score: float = 0.64
    min_confluence_strategy_confidence: float = 0.58

    min_required_domains: int = 3
    min_aligned_domains: int = 2
    min_strong_votes: int = 2

    min_vote_score: float = 0.50
    min_vote_confidence: float = 0.50
    strong_vote_min_score: float = 0.62
    strong_vote_min_confidence: float = 0.58

    use_orderflow: bool = True
    use_liquidity: bool = True
    use_liquidations: bool = True
    use_whales: bool = True
    use_open_interest: bool = True
    use_funding: bool = False
    use_price_action: bool = True
    use_spoofing: bool = False
    use_spreads: bool = False

    require_orderflow: bool = False
    require_liquidity: bool = False
    require_liquidations: bool = False
    require_whales: bool = False
    require_open_interest: bool = False
    require_funding: bool = False
    require_price_action: bool = False
    require_spoofing: bool = False
    require_spreads: bool = False

    allow_missing_required_domains: int = 0
    reject_direct_conflicts: bool = True
    allow_single_conflict: bool = False

    orderflow_vote_weight: float = 1.05
    liquidity_vote_weight: float = 1.05
    liquidations_vote_weight: float = 1.00
    whales_vote_weight: float = 1.00
    open_interest_vote_weight: float = 0.90
    funding_vote_weight: float = 0.75
    price_action_vote_weight: float = 1.15
    spoofing_vote_weight: float = 0.80
    spreads_vote_weight: float = 0.75

    score_confluence_weight: float = 0.34
    score_alignment_weight: float = 0.24
    score_vote_strength_weight: float = 0.16
    score_domain_count_weight: float = 0.10
    score_freshness_weight: float = 0.08
    score_conflict_inverse_weight: float = 0.08

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    aligned_domains_bonus: float = 0.04
    strong_votes_bonus: float = 0.04
    price_action_confirmation_bonus: float = 0.03
    orderflow_confirmation_bonus: float = 0.03
    whale_confirmation_bonus: float = 0.03
    low_conflict_bonus: float = 0.03

    low_conflict_threshold: float = 0.15

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.HYBRID

    tag_generic_confluence: str = "generic_confluence"
    tag_multi_domain: str = "multi_domain"
    tag_same_side_alignment: str = "same_side_alignment"
    tag_strong_votes: str = "strong_votes"
    tag_low_conflict: str = "low_conflict"

    execution_entry_offset_bps_hint: float | None = None
    execution_stop_buffer_bps_hint: float | None = None
    execution_take_profit_bps_hint: float | None = None
    confluence_rr_hint: float | None = None

    required_hybrid_features: tuple[str, ...] = (
        HYBRID_FEATURES.DOMINANT_SIDE,
        HYBRID_FEATURES.ALIGNMENT_SCORE,
        HYBRID_FEATURES.CONFLUENCE_SCORE,
    )

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategyConfig.validate")
        HybridStrategyConfig.validate(self)

        unit_fields = {
            "min_confluence_strategy_score": self.min_confluence_strategy_score,
            "min_confluence_strategy_confidence": self.min_confluence_strategy_confidence,
            "min_vote_score": self.min_vote_score,
            "min_vote_confidence": self.min_vote_confidence,
            "strong_vote_min_score": self.strong_vote_min_score,
            "strong_vote_min_confidence": self.strong_vote_min_confidence,
            "aligned_domains_bonus": self.aligned_domains_bonus,
            "strong_votes_bonus": self.strong_votes_bonus,
            "price_action_confirmation_bonus": self.price_action_confirmation_bonus,
            "orderflow_confirmation_bonus": self.orderflow_confirmation_bonus,
            "whale_confirmation_bonus": self.whale_confirmation_bonus,
            "low_conflict_bonus": self.low_conflict_bonus,
            "low_conflict_threshold": self.low_conflict_threshold,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.min_required_domains <= 0:
            raise StrategyConfigError("min_required_domains must be > 0")

        if self.min_aligned_domains <= 0:
            raise StrategyConfigError("min_aligned_domains must be > 0")

        if self.min_strong_votes < 0:
            raise StrategyConfigError("min_strong_votes must be >= 0")

        if self.allow_missing_required_domains < 0:
            raise StrategyConfigError("allow_missing_required_domains must be >= 0")

        vote_weights = {
            "orderflow_vote_weight": self.orderflow_vote_weight,
            "liquidity_vote_weight": self.liquidity_vote_weight,
            "liquidations_vote_weight": self.liquidations_vote_weight,
            "whales_vote_weight": self.whales_vote_weight,
            "open_interest_vote_weight": self.open_interest_vote_weight,
            "funding_vote_weight": self.funding_vote_weight,
            "price_action_vote_weight": self.price_action_vote_weight,
            "spoofing_vote_weight": self.spoofing_vote_weight,
            "spreads_vote_weight": self.spreads_vote_weight,
        }
        score_weights = {
            "score_confluence_weight": self.score_confluence_weight,
            "score_alignment_weight": self.score_alignment_weight,
            "score_vote_strength_weight": self.score_vote_strength_weight,
            "score_domain_count_weight": self.score_domain_count_weight,
            "score_freshness_weight": self.score_freshness_weight,
            "score_conflict_inverse_weight": self.score_conflict_inverse_weight,
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
            "confluence_rr_hint": self.confluence_rr_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        for attr in (
            "tag_generic_confluence",
            "tag_multi_domain",
            "tag_same_side_alignment",
            "tag_strong_votes",
            "tag_low_conflict",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


class ConfluenceStrategy(HybridTradingStrategy):
    """
    Generic local hybrid confluence strategy.

    Input:
        StrategyContext with several FeatureSource.* domain sections.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    It does not replace SignalProcessor.ConfluenceEngine.
    SignalProcessor owns routing, global confluence, filters, portfolio coordination,
    building and risk payloads.
    """
    _logger = logging.getLogger(__name__ + ".ConfluenceStrategy")

    component_namespace = "strategy.hybrid.confluence"
    category: StrategyCategory = StrategyCategory.HYBRID
    default_setup_type: SetupType = SetupType.HYBRID

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        hybrid_config: ConfluenceStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy.__init__")
        resolved_hybrid_config = hybrid_config or ConfluenceStrategyConfig()
        resolved_hybrid_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            hybrid_config=resolved_hybrid_config,
            service_name=service_name,
        )

        self.confluence_config: ConfluenceStrategyConfig = resolved_hybrid_config

    @property
    def strategy_name(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy.strategy_name")
        return "confluence"

    @property
    def metadata(self) -> StrategyMetadata:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy.metadata")
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.HYBRID,
            timeframe=Timeframe.M1,
            tags=[
                self.confluence_config.tag_hybrid,
                self.confluence_config.tag_confluence,
                self.confluence_config.tag_generic_confluence,
                self.confluence_config.tag_multi_domain,
                "strategy_context",
            ],
            version="2.0.0",
            description=(
                "Builds a local multi-domain confluence signal from StrategyContext "
                "without replacing SignalProcessor.ConfluenceEngine."
            ),
            required_features=set(self.required_features()),
            supported_regimes={
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.BREAKOUT,
                MarketRegime.SQUEEZE,
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "strategy_context.domains",
                "strategy_type": "hybrid_confluence",
                "base_class": "HybridTradingStrategy",
                "canonical_payload": "HybridCompositeSnapshot",
                "uses_local_direction_votes": True,
                "duplicates_signal_processor_confluence": False,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy.required_features")
        base_required = super().required_features()
        return set(base_required).union(self.confluence_config.required_hybrid_features)

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy.generate_signal")
        self.validate_context_requirements(context)

        sources = self._enabled_sources()
        required_sources = self._required_sources()

        required_domains_available = self.required_domains_available(
            context,
            required_sources,
            allow_missing=self.confluence_config.allow_missing_required_domains,
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
            stale_after_seconds=self.confluence_config.stale_feature_max_age_seconds,
        ):
            return None

        if not self.accepts_hybrid_snapshot(
            payload.snapshot,
            required_sources=required_sources,
            min_score=self.confluence_config.min_score,
            min_confidence=self.confluence_config.min_confidence,
            min_alignment_score=self.confluence_config.min_alignment_score,
            min_confluence_score=self.confluence_config.min_confluence_score,
            max_conflict_score=self.confluence_config.max_conflict_score,
            allow_missing=self.confluence_config.allow_missing_required_domains,
        ):
            return None

        if not self._passes_confluence_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.confluence_config.min_confluence_strategy_score:
            return None

        if breakdown.confidence < self.confluence_config.min_confluence_strategy_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "hybrid_confluence_signal",
                    f"side:{payload.side.value}",
                    f"aligned_domains:{len(payload.aligned_votes)}",
                    f"conflicting_domains:{len(payload.conflicting_votes)}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "hybrid_setup_family": "confluence",
            "hybrid_strategy_version": "2.0.0",
            "contract": "hybrid",
            "contract_version": "strategy-domain-v1",
            "primary_section": "generic_confluence",
            "strategy_contract_role": "decision_module",
            "risk_ready_payload_owner": "SignalProcessor",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "snapshot": serialize_for_metadata(payload.snapshot.to_dict()),
            "votes": [vote.to_dict() for vote in payload.votes],
            "aligned_votes": [vote.to_dict() for vote in payload.aligned_votes],
            "conflicting_votes": [vote.to_dict() for vote in payload.conflicting_votes],
            "raw": serialize_for_metadata(payload.raw),
            "event_time": payload.event_time.isoformat() if payload.event_time else None,
            "mapped_side": payload.side.value,
            "dominant_side": payload.snapshot.dominant_side.value,
            "alignment_score": payload.snapshot.alignment_score,
            "conflict_score": payload.snapshot.conflict_score,
            "confluence_score": payload.snapshot.confluence_score,
            "confidence": payload.snapshot.confidence,
            "available_domains": payload.snapshot.available_domains,
            "aligned_domains": payload.snapshot.aligned_domains,
            "conflicting_domains": payload.snapshot.conflicting_domains,
            "required_domains": [source.value for source in self._required_sources()],
            "enabled_domains": [source.value for source in self._enabled_sources()],
            "execution_hints": self._execution_hints(),
        }

        return self.build_hybrid_signal(
            context=context,
            side=payload.side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.confluence_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.confluence_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Sources / weights
    # ------------------------------------------------------------------

    def _enabled_sources(self) -> tuple[FeatureSource, ...]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy._enabled_sources")
        sources: list[FeatureSource] = []

        if self.confluence_config.use_orderflow:
            sources.append(FeatureSource.ORDERFLOW)

        if self.confluence_config.use_liquidity:
            sources.append(FeatureSource.LIQUIDITY)

        if self.confluence_config.use_liquidations:
            sources.append(FeatureSource.LIQUIDATIONS)

        if self.confluence_config.use_whales:
            sources.append(FeatureSource.WHALES)

        if self.confluence_config.use_open_interest:
            sources.append(FeatureSource.OPEN_INTEREST)

        if self.confluence_config.use_funding:
            sources.append(FeatureSource.FUNDING)

        if self.confluence_config.use_price_action:
            sources.append(FeatureSource.PRICE_ACTION)

        if self.confluence_config.use_spoofing:
            sources.append(FeatureSource.SPOOFING)

        if self.confluence_config.use_spreads:
            sources.append(FeatureSource.SPREADS)

        return tuple(dict.fromkeys(sources))

    def _required_sources(self) -> tuple[FeatureSource, ...]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy._required_sources")
        sources: list[FeatureSource] = []

        if self.confluence_config.require_orderflow:
            sources.append(FeatureSource.ORDERFLOW)

        if self.confluence_config.require_liquidity:
            sources.append(FeatureSource.LIQUIDITY)

        if self.confluence_config.require_liquidations:
            sources.append(FeatureSource.LIQUIDATIONS)

        if self.confluence_config.require_whales:
            sources.append(FeatureSource.WHALES)

        if self.confluence_config.require_open_interest:
            sources.append(FeatureSource.OPEN_INTEREST)

        if self.confluence_config.require_funding:
            sources.append(FeatureSource.FUNDING)

        if self.confluence_config.require_price_action:
            sources.append(FeatureSource.PRICE_ACTION)

        if self.confluence_config.require_spoofing:
            sources.append(FeatureSource.SPOOFING)

        if self.confluence_config.require_spreads:
            sources.append(FeatureSource.SPREADS)

        return tuple(dict.fromkeys(sources))

    def _vote_weights(self) -> dict[FeatureSource, float]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy._vote_weights")
        return {
            FeatureSource.ORDERFLOW: self.confluence_config.orderflow_vote_weight,
            FeatureSource.LIQUIDITY: self.confluence_config.liquidity_vote_weight,
            FeatureSource.LIQUIDATIONS: self.confluence_config.liquidations_vote_weight,
            FeatureSource.WHALES: self.confluence_config.whales_vote_weight,
            FeatureSource.OPEN_INTEREST: self.confluence_config.open_interest_vote_weight,
            FeatureSource.FUNDING: self.confluence_config.funding_vote_weight,
            FeatureSource.PRICE_ACTION: self.confluence_config.price_action_vote_weight,
            FeatureSource.SPOOFING: self.confluence_config.spoofing_vote_weight,
            FeatureSource.SPREADS: self.confluence_config.spreads_vote_weight,
        }

    def _has_minimum_available_domains(
        self,
        context: StrategyContext,
        sources: tuple[FeatureSource, ...],
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy._has_minimum_available_domains")
        available = available_domain_sources(context, sources)
        return len(available) >= self.confluence_config.min_required_domains

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
    ) -> ConfluencePayload | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy._extract_payload")
        if not snapshot.directional:
            return None

        side = snapshot.dominant_side
        if not is_directional_side(side):
            return None

        aligned_votes = votes_for_side(snapshot.votes, side)
        conflicting_votes = votes_against_side(snapshot.votes, side)

        event_time = (
            latest_timestamp_from_payloads(snapshot.payloads(), fallback=context.timestamp)
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "hybrid_confluence_context",
            f"side:{side.value}",
            f"available_domains:{snapshot.domain_count}",
            f"aligned_domains:{len(aligned_votes)}",
            f"conflicting_domains:{len(conflicting_votes)}",
            f"alignment_score:{snapshot.alignment_score:.4f}",
            f"conflict_score:{snapshot.conflict_score:.4f}",
            f"confluence_score:{snapshot.confluence_score:.4f}",
        ]

        return ConfluencePayload(
            snapshot=snapshot,
            side=side,
            votes=list(snapshot.votes),
            aligned_votes=aligned_votes,
            conflicting_votes=conflicting_votes,
            event_time=event_time,
            reasons=list(dict.fromkeys(reasons)),
            raw={
                "payloads": snapshot.payloads(),
                "sources": [source.value for source in sources],
                "required_sources": [source.value for source in required_sources],
            },
        )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_confluence_filters(
        self,
        payload: ConfluencePayload,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy._passes_confluence_filters")
        if len(payload.aligned_votes) < self.confluence_config.min_aligned_domains:
            return False

        strong = strong_votes(
            payload.votes,
            min_score=self.confluence_config.strong_vote_min_score,
            min_confidence=self.confluence_config.strong_vote_min_confidence,
        )
        strong_aligned = [
            vote
            for vote in strong
            if vote.side is payload.side
        ]

        if len(strong_aligned) < self.confluence_config.min_strong_votes:
            return False

        qualified_votes = [
            vote
            for vote in payload.aligned_votes
            if vote.score >= self.confluence_config.min_vote_score
            and vote.confidence >= self.confluence_config.min_vote_confidence
        ]
        if len(qualified_votes) < self.confluence_config.min_aligned_domains:
            return False

        if self.confluence_config.reject_direct_conflicts:
            if payload.conflicting_votes and not self.confluence_config.allow_single_conflict:
                return False

            if (
                payload.conflicting_votes
                and len(payload.conflicting_votes) > 1
                and self.confluence_config.allow_single_conflict
            ):
                return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: ConfluencePayload,
    ) -> HybridScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy._build_score_breakdown")
        snapshot = payload.snapshot

        domain_count_component = unit_score(
            snapshot.domain_count / max(self.confluence_config.min_required_domains, 1)
        )
        aligned_vote_strength = unit_score(
            sum(vote.weighted_strength for vote in payload.aligned_votes)
            / max(len(payload.aligned_votes), 1)
        )
        freshness_component = hybrid_freshness_score(
            snapshot.payloads(),
            now=context.timestamp,
            stale_after_seconds=self.confluence_config.stale_feature_max_age_seconds,
        )
        conflict_inverse = 1.0 - snapshot.conflict_score

        components = {
            "confluence": snapshot.confluence_score,
            "alignment": snapshot.alignment_score,
            "vote_strength": aligned_vote_strength,
            "domain_count": domain_count_component,
            "freshness": freshness_component,
            "conflict_inverse": conflict_inverse,
        }
        weights = {
            "confluence": self.confluence_config.score_confluence_weight,
            "alignment": self.confluence_config.score_alignment_weight,
            "vote_strength": self.confluence_config.score_vote_strength_weight,
            "domain_count": self.confluence_config.score_domain_count_weight,
            "freshness": self.confluence_config.score_freshness_weight,
            "conflict_inverse": self.confluence_config.score_conflict_inverse_weight,
        }

        score = weighted_score(
            components,
            weights,
            default=snapshot.confluence_score,
        )
        confidence = weighted_score(
            {
                "primary": unit_score((snapshot.confluence_score + aligned_vote_strength) / 2),
                "context": snapshot.alignment_score,
                "confirmation": conflict_inverse,
                "freshness": freshness_component,
            },
            {
                "primary": self.confluence_config.confidence_primary_weight,
                "context": self.confluence_config.confidence_context_weight,
                "confirmation": self.confluence_config.confidence_confirmation_weight,
                "freshness": self.confluence_config.confidence_freshness_weight,
            },
            default=snapshot.confidence,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "hybrid_confluence_context",
            f"dominant_side:{payload.side.value}",
            f"aligned_domains:{len(payload.aligned_votes)}",
            f"available_domains:{snapshot.domain_count}",
        ]

        conflicts = [
            vote.source.value
            for vote in payload.conflicting_votes
        ]

        if len(payload.aligned_votes) >= self.confluence_config.min_aligned_domains:
            score += self.confluence_config.aligned_domains_bonus
            confirmations.append("minimum_aligned_domains_confirmed")

        strong = strong_votes(
            payload.aligned_votes,
            min_score=self.confluence_config.strong_vote_min_score,
            min_confidence=self.confluence_config.strong_vote_min_confidence,
        )
        if len(strong) >= self.confluence_config.min_strong_votes:
            score += self.confluence_config.strong_votes_bonus
            confirmations.append("strong_votes_confirmed")

        aligned_sources = aligned_source_names(payload.votes, payload.side)

        if FeatureSource.PRICE_ACTION.value in aligned_sources:
            score += self.confluence_config.price_action_confirmation_bonus
            confirmations.append("price_action_aligned")

        if FeatureSource.ORDERFLOW.value in aligned_sources:
            score += self.confluence_config.orderflow_confirmation_bonus
            confirmations.append("orderflow_aligned")

        if FeatureSource.WHALES.value in aligned_sources:
            score += self.confluence_config.whale_confirmation_bonus
            confirmations.append("whales_aligned")

        if snapshot.conflict_score <= self.confluence_config.low_conflict_threshold:
            score += self.confluence_config.low_conflict_bonus
            confirmations.append("low_domain_conflict")

        if conflicts:
            reasons.append(f"conflicts:{','.join(conflicts)}")

        reasons.extend(
            [
                f"alignment_score:{snapshot.alignment_score:.4f}",
                f"conflict_score:{snapshot.conflict_score:.4f}",
                f"confluence_score:{snapshot.confluence_score:.4f}",
                f"domain_count:{snapshot.domain_count}",
            ]
        )

        return HybridScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            votes=list(payload.votes),
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
            conflicts=conflicts,
        ).normalize()

    # ------------------------------------------------------------------
    # Source features / tags / metadata helpers
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: ConfluencePayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy._source_features")
        features = [
            *confluence_source_features(),
            HYBRID_FEATURES.DOMINANT_SIDE,
            HYBRID_FEATURES.ALIGNMENT_SCORE,
            HYBRID_FEATURES.CONFLICT_SCORE,
            HYBRID_FEATURES.CONFLUENCE_SCORE,
            HYBRID_FEATURES.CONFIDENCE,
            HYBRID_FEATURES.VOTES,
        ]

        for source in self._enabled_sources():
            features.append(f"{source.value}.*")

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: ConfluencePayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy._tags")
        tags = [
            self.confluence_config.tag_hybrid,
            self.confluence_config.tag_confluence,
            self.confluence_config.tag_generic_confluence,
            self.confluence_config.tag_multi_domain,
            self.confluence_config.tag_same_side_alignment,
            f"side:{payload.side.value}",
            f"domains:{payload.snapshot.domain_count}",
            f"aligned:{len(payload.aligned_votes)}",
        ]

        strong = strong_votes(
            payload.aligned_votes,
            min_score=self.confluence_config.strong_vote_min_score,
            min_confidence=self.confluence_config.strong_vote_min_confidence,
        )
        if len(strong) >= self.confluence_config.min_strong_votes:
            tags.append(self.confluence_config.tag_strong_votes)

        if payload.snapshot.conflict_score <= self.confluence_config.low_conflict_threshold:
            tags.append(self.confluence_config.tag_low_conflict)

        for source in payload.snapshot.aligned_domains:
            tags.append(f"aligned:{source}")

        return list(dict.fromkeys(tags))

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceStrategy._execution_hints")
        return {
            "entry_offset_bps": self.confluence_config.execution_entry_offset_bps_hint,
            "stop_buffer_bps": self.confluence_config.execution_stop_buffer_bps_hint,
            "take_profit_bps": self.confluence_config.execution_take_profit_bps_hint,
            "risk_reward": self.confluence_config.confluence_rr_hint,
        }