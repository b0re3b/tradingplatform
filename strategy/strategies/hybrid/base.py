# trading_system/strategy/strategies/hybrid/base.py

from __future__ import annotations
import logging

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler
from .utils import (
    DEFAULT_HYBRID_STALE_AFTER_SECONDS,
    DOMAIN_FEATURE_SOURCES,
    DirectionVote,
    HYBRID_CONTEXT_VERSION,
    HybridScoreBreakdown,
    aligned_source_names,
    alignment_score,
    as_dict,
    available_domain_sources,
    build_direction_vote,
    build_direction_votes,
    build_hybrid_score_breakdown,
    conflict_score,
    conflicting_source_names,
    domain_available,
    domain_confluence_score,
    domain_dict,
    domain_is_stale,
    domain_path,
    dominant_side_from_votes,
    extract_context_domain_payloads,
    extract_domain_exchange,
    extract_domain_market_type,
    extract_domain_symbol,
    extract_domain_timeframe,
    hybrid_base_source_features,
    hybrid_freshness_score,
    is_directional_side,
    latest_timestamp_from_payloads,
    missing_domain_sources,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    parse_datetime,
    required_domains_available,
    serialize_for_metadata,
    timestamp_ms,
    to_bool,
    to_float,
    to_int,
    to_str,
    unit_score,
)
from ..base_strategy import TradingStrategy
from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    EntryType,
    ExitType,
    FeatureSource,
    SetupType,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    StrategyMarginMode,
    StrategyMarketType,
    StrategyOrderIntent,
    StrategyTradeTier,
    Timeframe,
    TriggerType,
)
from ...exceptions import StrategyConfigError, StrategyEvaluationError
from ...models import FeatureSnapshot, StrategyContext, StrategySignal


# =============================================================================
# Feature contract
# =============================================================================


@dataclass(frozen=True, slots=True)
class HybridFeatureNames:
    """
    Stable hybrid feature names expected in StrategyContext.

    Hybrid strategies mostly read native domain sections:
    FeatureSource.ORDERFLOW, FeatureSource.LIQUIDITY, etc.
    These feature names are optional normalized summary outputs that
    StrategyContextBuilder / SignalNormalizer may also provide.
    """
    _logger = logging.getLogger(__name__ + ".HybridFeatureNames")

    ORDERFLOW: str = "hybrid.orderflow"
    LIQUIDITY: str = "hybrid.liquidity"
    LIQUIDATIONS: str = "hybrid.liquidations"
    WHALES: str = "hybrid.whales"
    OPEN_INTEREST: str = "hybrid.open_interest"
    FUNDING: str = "hybrid.funding"
    PRICE_ACTION: str = "hybrid.price_action"
    SPOOFING: str = "hybrid.spoofing"
    SPREADS: str = "hybrid.spreads"

    DOMINANT_SIDE: str = "hybrid.dominant_side"
    ALIGNMENT_SCORE: str = "hybrid.alignment_score"
    CONFLICT_SCORE: str = "hybrid.conflict_score"
    CONFLUENCE_SCORE: str = "hybrid.confluence_score"
    CONFIDENCE: str = "hybrid.confidence"
    VOTES: str = "hybrid.votes"

    SYMBOL: str = "hybrid.symbol"
    EXCHANGE: str = "hybrid.exchange"
    MARKET_TYPE: str = "hybrid.market_type"
    TIMEFRAME: str = "hybrid.timeframe"
    EXCHANGE_SYMBOL: str = "hybrid.exchange_symbol"

    TIMESTAMP: str = "hybrid.timestamp"
    METADATA: str = "hybrid.metadata"

    @classmethod
    def all(cls) -> set[str]:
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".HybridFeatureNames")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridFeatureNames.all")
        instance = cls()
        return {
            getattr(instance, item.name)
            for item in fields(cls)
            if isinstance(getattr(instance, item.name), str)
            and getattr(instance, item.name).strip()
        }


HYBRID_FEATURES = HybridFeatureNames()


# =============================================================================
# Scope
# =============================================================================


@dataclass(frozen=True, slots=True)
class HybridStrategyScope:
    _logger = logging.getLogger(__name__ + ".HybridStrategyScope")
    exchange: str
    market_type: str
    symbol: str
    timeframe: str = Timeframe.M1.value
    exchange_symbol: str | None = None

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridStrategyScope.__post_init__")
        exchange = normalize_exchange(self.exchange) or "unknown"
        market_type = normalize_market_type(self.market_type) or "unknown"
        symbol = normalize_symbol(self.symbol)
        timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()
        exchange_symbol = to_str(self.exchange_symbol) or symbol

        if not symbol:
            raise StrategyEvaluationError("HybridStrategyScope.symbol cannot be empty")

        object.__setattr__(self, "exchange", exchange)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "exchange_symbol", exchange_symbol)

    @property
    def key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridStrategyScope.key")
        return (
            f"{self.exchange}:"
            f"{self.market_type}:"
            f"{self.symbol}:"
            f"{self.timeframe}"
        )

    @property
    def legacy_key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridStrategyScope.legacy_key")
        return f"{self.exchange}:{self.symbol}"

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridStrategyScope.to_dict")
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "key": self.key,
            "legacy_key": self.legacy_key,
        }


# =============================================================================
# Config
# =============================================================================


@dataclass(slots=True)
class HybridStrategyConfig:
    """
    Stateless config shared by concrete hybrid strategies.

    Hybrid strategies are decision modules only:
    StrategyContext -> StrategySignal | None.

    They must not duplicate SignalProcessor / ConfluenceEngine /
    PortfolioCoordinator / SignalBuilder.
    """
    _logger = logging.getLogger(__name__ + ".HybridStrategyConfig")

    min_score: float = 0.62
    min_confidence: float = 0.58
    min_alignment_score: float = 0.60
    min_confluence_score: float = 0.58
    max_conflict_score: float = 0.35

    stale_feature_max_age_seconds: float | None = DEFAULT_HYBRID_STALE_AFTER_SECONDS

    min_required_domains: int = 3
    allow_missing_required_domains: int = 0
    require_same_side_alignment: bool = True
    reject_direct_conflicts: bool = True
    allow_unknown_side: bool = False

    default_market_type: StrategyMarketType = StrategyMarketType.USDM_FUTURES
    default_margin_mode: StrategyMarginMode = StrategyMarginMode.ISOLATED
    default_order_intent: StrategyOrderIntent = StrategyOrderIntent.OPEN
    default_trade_tier: StrategyTradeTier = StrategyTradeTier.T2

    default_entry_type: EntryType = EntryType.MARKET
    default_exit_types: tuple[ExitType, ...] = (
        ExitType.TAKE_PROFIT,
        ExitType.STOP_LOSS,
        ExitType.INVALIDATION,
    )

    requested_leverage: float | None = None
    max_slippage_bps: float | None = None
    entry_timeout_seconds: int | None = None
    max_holding_seconds: int | None = None

    attach_hybrid_context_metadata: bool = True
    attach_domain_snapshots_metadata: bool = True
    attach_votes_metadata: bool = True
    attach_scope_metadata: bool = True

    tag_hybrid: str = "hybrid"
    tag_confluence: str = "confluence"
    tag_trend_stack: str = "trend_stack"
    tag_mean_reversion_stack: str = "mean_reversion_stack"
    tag_liquidation_whale: str = "liquidation_whale"
    tag_liquidity_orderflow: str = "liquidity_orderflow"
    tag_oi_funding: str = "oi_funding"
    tag_whale_orderflow: str = "whale_orderflow"

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridStrategyConfig.validate")
        bounded = {
            "min_score": self.min_score,
            "min_confidence": self.min_confidence,
            "min_alignment_score": self.min_alignment_score,
            "min_confluence_score": self.min_confluence_score,
            "max_conflict_score": self.max_conflict_score,
        }
        for name, value in bounded.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{name} must be between 0.0 and 1.0")

        if (
            self.stale_feature_max_age_seconds is not None
            and self.stale_feature_max_age_seconds <= 0
        ):
            raise StrategyConfigError("stale_feature_max_age_seconds must be > 0")

        if self.min_required_domains <= 1:
            raise StrategyConfigError("min_required_domains must be > 1")

        if self.allow_missing_required_domains < 0:
            raise StrategyConfigError("allow_missing_required_domains must be >= 0")

        if self.requested_leverage is not None and self.requested_leverage <= 0:
            raise StrategyConfigError("requested_leverage must be > 0")

        if self.max_slippage_bps is not None and self.max_slippage_bps < 0:
            raise StrategyConfigError("max_slippage_bps must be >= 0")

        if self.entry_timeout_seconds is not None and self.entry_timeout_seconds <= 0:
            raise StrategyConfigError("entry_timeout_seconds must be > 0")

        if self.max_holding_seconds is not None and self.max_holding_seconds <= 0:
            raise StrategyConfigError("max_holding_seconds must be > 0")

        for attr in (
            "tag_hybrid",
            "tag_confluence",
            "tag_trend_stack",
            "tag_mean_reversion_stack",
            "tag_liquidation_whale",
            "tag_liquidity_orderflow",
            "tag_oi_funding",
            "tag_whale_orderflow",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


# =============================================================================
# Composite snapshot
# =============================================================================


@dataclass(slots=True)
class HybridCompositeSnapshot:
    """
    Strategy-side normalized projection over multiple domain payloads.

    This DTO is local to hybrid strategies. It does not replace SignalProcessor's
    global ConfluenceEngine.
    """
    _logger = logging.getLogger(__name__ + ".HybridCompositeSnapshot")

    symbol: str
    exchange: str = "unknown"
    market_type: str = "unknown"
    timeframe: str = Timeframe.M1.value
    exchange_symbol: str | None = None

    orderflow: dict[str, Any] = field(default_factory=dict)
    liquidity: dict[str, Any] = field(default_factory=dict)
    liquidations: dict[str, Any] = field(default_factory=dict)
    whales: dict[str, Any] = field(default_factory=dict)
    open_interest: dict[str, Any] = field(default_factory=dict)
    funding: dict[str, Any] = field(default_factory=dict)
    price_action: dict[str, Any] = field(default_factory=dict)
    spoofing: dict[str, Any] = field(default_factory=dict)
    spreads: dict[str, Any] = field(default_factory=dict)

    votes: list[DirectionVote] = field(default_factory=list)
    dominant_side: SignalSide = SignalSide.UNKNOWN

    alignment_score: float = 0.0
    conflict_score: float = 0.0
    confluence_score: float = 0.0
    confidence: float = 0.0

    timestamp: datetime | None = None
    source: str = "context.domains"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.__post_init__")
        self.symbol = normalize_symbol(self.symbol)
        self.exchange = normalize_exchange(self.exchange) or "unknown"
        self.market_type = normalize_market_type(self.market_type) or "unknown"
        self.timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()
        self.exchange_symbol = to_str(self.exchange_symbol) or self.symbol

        if not self.symbol:
            raise StrategyEvaluationError("HybridCompositeSnapshot.symbol cannot be empty")

        self.orderflow = as_dict(self.orderflow)
        self.liquidity = as_dict(self.liquidity)
        self.liquidations = as_dict(self.liquidations)
        self.whales = as_dict(self.whales)
        self.open_interest = as_dict(self.open_interest)
        self.funding = as_dict(self.funding)
        self.price_action = as_dict(self.price_action)
        self.spoofing = as_dict(self.spoofing)
        self.spreads = as_dict(self.spreads)

        self.alignment_score = unit_score(self.alignment_score)
        self.conflict_score = unit_score(self.conflict_score)
        self.confluence_score = unit_score(self.confluence_score)
        self.confidence = unit_score(self.confidence)
        self.timestamp = parse_datetime(self.timestamp)
        self.metadata = as_dict(self.metadata)

    @property
    def scope(self) -> HybridStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.scope")
        return HybridStrategyScope(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange_symbol=self.exchange_symbol,
        )

    @property
    def scope_key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.scope_key")
        return self.scope.key

    @property
    def directional(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.directional")
        return is_directional_side(self.dominant_side)

    @property
    def available_domains(self) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.available_domains")
        return [
            source.value
            for source, payload in self.payloads().items()
            if bool(payload)
        ]

    @property
    def domain_count(self) -> int:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.domain_count")
        return len(self.available_domains)

    @property
    def aligned_domains(self) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.aligned_domains")
        if not self.directional:
            return []
        return aligned_source_names(self.votes, self.dominant_side)

    @property
    def conflicting_domains(self) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.conflicting_domains")
        if not self.directional:
            return []
        return conflicting_source_names(self.votes, self.dominant_side)

    def payloads(self) -> dict[FeatureSource, dict[str, Any]]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.payloads")
        return {
            FeatureSource.ORDERFLOW: self.orderflow,
            FeatureSource.LIQUIDITY: self.liquidity,
            FeatureSource.LIQUIDATIONS: self.liquidations,
            FeatureSource.WHALES: self.whales,
            FeatureSource.OPEN_INTEREST: self.open_interest,
            FeatureSource.FUNDING: self.funding,
            FeatureSource.PRICE_ACTION: self.price_action,
            FeatureSource.SPOOFING: self.spoofing,
            FeatureSource.SPREADS: self.spreads,
        }

    def has_domain(self, source: FeatureSource) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.has_domain")
        return bool(self.payloads().get(source))

    def has_any_domain(self, sources: Sequence[FeatureSource]) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.has_any_domain")
        return any(self.has_domain(source) for source in sources)

    def has_all_domains(
        self,
        sources: Sequence[FeatureSource],
        *,
        allow_missing: int = 0,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.has_all_domains")
        missing = sum(1 for source in sources if not self.has_domain(source))
        return missing <= max(0, allow_missing)

    def has_minimum_data(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.has_minimum_data")
        return self.domain_count > 0 or bool(self.votes) or self.confidence > 0.0

    def to_signal_payload(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.to_signal_payload")
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "market_type": self.market_type,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "dominant_side": self.dominant_side,
            "alignment_score": self.alignment_score,
            "conflict_score": self.conflict_score,
            "confluence_score": self.confluence_score,
            "confidence": self.confidence,
            "available_domains": self.available_domains,
            "aligned_domains": self.aligned_domains,
            "conflicting_domains": self.conflicting_domains,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridCompositeSnapshot.to_dict")
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "market_type": self.market_type,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "scope": self.scope.to_dict(),
            "scope_key": self.scope_key,
            "dominant_side": self.dominant_side.value,
            "directional": self.directional,
            "alignment_score": self.alignment_score,
            "conflict_score": self.conflict_score,
            "confluence_score": self.confluence_score,
            "confidence": self.confidence,
            "available_domains": self.available_domains,
            "domain_count": self.domain_count,
            "aligned_domains": self.aligned_domains,
            "conflicting_domains": self.conflicting_domains,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "timestamp_ms": timestamp_ms(self.timestamp),
            "source": self.source,
            "votes": [vote.to_dict() for vote in self.votes],
            "payloads": {
                source.value: serialize_for_metadata(payload)
                for source, payload in self.payloads().items()
                if payload
            },
            "metadata": serialize_for_metadata(self.metadata),
        }


# =============================================================================
# Base hybrid strategy
# =============================================================================


class HybridTradingStrategy(TradingStrategy):
    """
    Base class for concrete strategy/strategies/hybrid/* classes.

    Responsibilities:
    - read multiple domain sections from StrategyContext;
    - build local DirectionVote list;
    - compute local alignment/conflict/confluence metrics;
    - build internal StrategySignal objects.

    Forbidden:
    - no replacement for SignalProcessor;
    - no global ConfluenceEngine duplication;
    - no SignalRouter / PortfolioCoordinator / SignalBuilder duplication;
    - no direct signal.generated publishing;
    - no RiskManager / Execution calls;
    - no raw market data reads.
    """
    _logger = logging.getLogger(__name__ + ".HybridTradingStrategy")

    component_namespace = "strategy.hybrid"
    category: StrategyCategory = StrategyCategory.HYBRID
    default_setup_type: SetupType = SetupType.HYBRID
    default_timeframe: Timeframe = Timeframe.M1

    feature_names = HYBRID_FEATURES

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        hybrid_config: HybridStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.__init__")
        self.hybrid_config = hybrid_config or HybridStrategyConfig()
        self.hybrid_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            service_name=service_name,
        )

    def validate_config(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.validate_config")
        super().validate_config()
        self.hybrid_config.validate()

    # ------------------------------------------------------------------
    # Domain access
    # ------------------------------------------------------------------

    def hybrid_domain(self, context: StrategyContext) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.hybrid_domain")
        self.validate_context(context)
        return dict(context.domain_dict(FeatureSource.EXTERNAL))

    def domain_dict(
        self,
        context: StrategyContext,
        source: FeatureSource,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.domain_dict")
        self.validate_context(context)
        return domain_dict(context, source)

    def domain_path(
        self,
        context: StrategyContext,
        source: FeatureSource,
        path: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.domain_path")
        self.validate_context(context)

        if not isinstance(path, str) or not path.strip():
            raise StrategyEvaluationError("domain path cannot be empty")

        return domain_path(context, source, path, default)

    def domain_float(
        self,
        context: StrategyContext,
        source: FeatureSource,
        path: str,
        *,
        default: float | None = None,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.domain_float")
        return to_float(self.domain_path(context, source, path, default), default)

    def domain_int(
        self,
        context: StrategyContext,
        source: FeatureSource,
        path: str,
        *,
        default: int | None = None,
    ) -> int | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.domain_int")
        return to_int(self.domain_path(context, source, path, default), default)

    def domain_bool(
        self,
        context: StrategyContext,
        source: FeatureSource,
        path: str,
        *,
        default: bool = False,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.domain_bool")
        return to_bool(self.domain_path(context, source, path, default), default)

    def domain_str(
        self,
        context: StrategyContext,
        source: FeatureSource,
        path: str,
        *,
        default: str | None = None,
    ) -> str | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.domain_str")
        return to_str(self.domain_path(context, source, path, default), default)

    def domain_feature_snapshot(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> FeatureSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.domain_feature_snapshot")
        self.validate_context(context)

        if not isinstance(feature_name, str) or not feature_name.strip():
            raise StrategyEvaluationError("feature_name cannot be empty")

        features_map = getattr(context, "features", None)
        if isinstance(features_map, Mapping):
            raw = features_map.get(feature_name)
            if isinstance(raw, FeatureSnapshot):
                return raw

        return None

    def domain_feature_age_seconds(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.domain_feature_age_seconds")
        snapshot = self.domain_feature_snapshot(context, feature_name)
        if snapshot is None:
            return None
        return snapshot.age_seconds(context.timestamp)

    def domain_available(
        self,
        context: StrategyContext,
        source: FeatureSource,
        *,
        required_paths: Sequence[str] = (),
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.domain_available")
        self.validate_context(context)
        return domain_available(context, source, required_paths=required_paths)

    def required_domains_available(
        self,
        context: StrategyContext,
        sources: Sequence[FeatureSource],
        *,
        allow_missing: int | None = None,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.required_domains_available")
        return required_domains_available(
            context,
            sources,
            allow_missing=(
                self.hybrid_config.allow_missing_required_domains
                if allow_missing is None
                else allow_missing
            ),
        )

    def available_domain_sources(
        self,
        context: StrategyContext,
        sources: Sequence[FeatureSource] = DOMAIN_FEATURE_SOURCES,
    ) -> list[FeatureSource]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.available_domain_sources")
        return available_domain_sources(context, sources)

    def missing_domain_sources(
        self,
        context: StrategyContext,
        sources: Sequence[FeatureSource],
    ) -> list[FeatureSource]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.missing_domain_sources")
        return missing_domain_sources(context, sources)

    def has_stale_domains(
        self,
        context: StrategyContext,
        sources: Sequence[FeatureSource],
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.has_stale_domains")
        payloads = self.resolve_domain_payloads(context, sources)
        return any(
            domain_is_stale(
                payload,
                now=context.timestamp,
                stale_after_seconds=self.hybrid_config.stale_feature_max_age_seconds,
            )
            for payload in payloads.values()
        )

    # ------------------------------------------------------------------
    # Scope / payload / snapshot
    # ------------------------------------------------------------------

    def hybrid_scope(
        self,
        context: StrategyContext,
        sources: Sequence[FeatureSource] = DOMAIN_FEATURE_SOURCES,
    ) -> HybridStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.hybrid_scope")
        payloads = self.resolve_domain_payloads(context, sources)

        symbol = self._first_non_empty_symbol(payloads) or normalize_symbol(context.symbol)
        exchange = (
            self._first_non_empty_exchange(payloads)
            or normalize_exchange(context.metadata.get("exchange"))
            or "unknown"
        )
        market_type = (
            self._first_non_empty_market_type(payloads)
            or normalize_market_type(context.metadata.get("market_type"))
            or self.hybrid_config.default_market_type.value
        )
        timeframe = (
            self._first_non_empty_timeframe(payloads)
            or str(getattr(context.timeframe, "value", context.timeframe) or Timeframe.M1.value)
        )

        return HybridStrategyScope(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            exchange_symbol=symbol,
        )

    def resolve_domain_payloads(
        self,
        context: StrategyContext,
        sources: Sequence[FeatureSource] = DOMAIN_FEATURE_SOURCES,
    ) -> dict[FeatureSource, dict[str, Any]]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.resolve_domain_payloads")
        self.validate_context(context)
        return extract_context_domain_payloads(context, sources)

    def resolve_hybrid_snapshot(
        self,
        context: StrategyContext,
        *,
        sources: Sequence[FeatureSource] = DOMAIN_FEATURE_SOURCES,
        vote_weights: Mapping[FeatureSource, float] | None = None,
    ) -> HybridCompositeSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.resolve_hybrid_snapshot")
        self.validate_context(context)

        payloads = self.resolve_domain_payloads(context, sources)
        if not payloads:
            feature_snapshot = self._build_snapshot_from_hybrid_features(context)
            if feature_snapshot is not None and feature_snapshot.has_minimum_data():
                return feature_snapshot
            return None

        votes = self.build_direction_votes(
            context,
            sources=sources,
            vote_weights=vote_weights,
        )
        dominant_side = dominant_side_from_votes(votes)
        align_score = alignment_score(votes, side=dominant_side)
        conf_score = conflict_score(votes, side=dominant_side)
        confl_score = domain_confluence_score(
            votes,
            side=dominant_side,
            min_domains=self.hybrid_config.min_required_domains,
        )
        freshness = hybrid_freshness_score(
            payloads,
            now=context.timestamp,
            stale_after_seconds=self.hybrid_config.stale_feature_max_age_seconds,
        )

        scope = self.hybrid_scope(context, sources=sources)
        timestamp = (
            latest_timestamp_from_payloads(payloads, fallback=context.timestamp)
            or context.timestamp
        )

        confidence = unit_score(
            (align_score * 0.35)
            + (confl_score * 0.35)
            + ((1.0 - conf_score) * 0.15)
            + (freshness * 0.15)
        )

        snapshot = HybridCompositeSnapshot(
            symbol=scope.symbol,
            exchange=scope.exchange,
            market_type=scope.market_type,
            timeframe=scope.timeframe,
            exchange_symbol=scope.exchange_symbol,
            orderflow=payloads.get(FeatureSource.ORDERFLOW, {}),
            liquidity=payloads.get(FeatureSource.LIQUIDITY, {}),
            liquidations=payloads.get(FeatureSource.LIQUIDATIONS, {}),
            whales=payloads.get(FeatureSource.WHALES, {}),
            open_interest=payloads.get(FeatureSource.OPEN_INTEREST, {}),
            funding=payloads.get(FeatureSource.FUNDING, {}),
            price_action=payloads.get(FeatureSource.PRICE_ACTION, {}),
            spoofing=payloads.get(FeatureSource.SPOOFING, {}),
            spreads=payloads.get(FeatureSource.SPREADS, {}),
            votes=votes,
            dominant_side=dominant_side,
            alignment_score=align_score,
            conflict_score=conf_score,
            confluence_score=confl_score,
            confidence=confidence,
            timestamp=timestamp,
            source="context.domains",
            metadata={
                "version": HYBRID_CONTEXT_VERSION,
                "sources": [source.value for source in sources],
                "available_sources": [source.value for source in payloads.keys()],
                "missing_sources": [
                    source.value
                    for source in sources
                    if source not in payloads
                ],
                "scope": scope.to_dict(),
            },
        )

        if snapshot.has_minimum_data():
            return snapshot

        return None

    def _build_snapshot_from_hybrid_features(
        self,
        context: StrategyContext,
    ) -> HybridCompositeSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy._build_snapshot_from_hybrid_features")
        payload = {
            "symbol": self._feature_value(context, HYBRID_FEATURES.SYMBOL),
            "exchange": self._feature_value(context, HYBRID_FEATURES.EXCHANGE),
            "market_type": self._feature_value(context, HYBRID_FEATURES.MARKET_TYPE),
            "timeframe": self._feature_value(context, HYBRID_FEATURES.TIMEFRAME),
            "exchange_symbol": self._feature_value(
                context,
                HYBRID_FEATURES.EXCHANGE_SYMBOL,
            ),
            "dominant_side": self._feature_value(
                context,
                HYBRID_FEATURES.DOMINANT_SIDE,
            ),
            "alignment_score": self._feature_value(
                context,
                HYBRID_FEATURES.ALIGNMENT_SCORE,
            ),
            "conflict_score": self._feature_value(
                context,
                HYBRID_FEATURES.CONFLICT_SCORE,
            ),
            "confluence_score": self._feature_value(
                context,
                HYBRID_FEATURES.CONFLUENCE_SCORE,
            ),
            "confidence": self._feature_value(context, HYBRID_FEATURES.CONFIDENCE),
            "timestamp": self._feature_value(context, HYBRID_FEATURES.TIMESTAMP),
            "metadata": self._feature_value(context, HYBRID_FEATURES.METADATA),
        }

        if not self._has_any_value(payload):
            return None

        side_value = payload.get("dominant_side")
        dominant_side = (
            side_value
            if isinstance(side_value, SignalSide)
            else SignalSide.UNKNOWN
        )
        if not is_directional_side(dominant_side):
            from .utils import side_to_signal_side

            dominant_side = side_to_signal_side(side_value)

        symbol = normalize_symbol(payload.get("symbol")) or normalize_symbol(context.symbol)
        exchange = normalize_exchange(payload.get("exchange")) or normalize_exchange(
            context.metadata.get("exchange")
        ) or "unknown"
        market_type = normalize_market_type(payload.get("market_type")) or normalize_market_type(
            context.metadata.get("market_type")
        ) or self.hybrid_config.default_market_type.value
        timeframe = (
            to_str(payload.get("timeframe"))
            or str(getattr(context.timeframe, "value", context.timeframe) or Timeframe.M1.value)
        )

        snapshot = HybridCompositeSnapshot(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            exchange_symbol=to_str(payload.get("exchange_symbol")) or symbol,
            dominant_side=dominant_side,
            alignment_score=to_float(payload.get("alignment_score"), 0.0) or 0.0,
            conflict_score=to_float(payload.get("conflict_score"), 0.0) or 0.0,
            confluence_score=to_float(payload.get("confluence_score"), 0.0) or 0.0,
            confidence=to_float(payload.get("confidence"), 0.0) or 0.0,
            timestamp=parse_datetime(payload.get("timestamp")) or context.timestamp,
            source="context.features",
            metadata={
                **as_dict(payload.get("metadata")),
                "source": "context.features",
            },
        )

        return snapshot if snapshot.has_minimum_data() else None

    # ------------------------------------------------------------------
    # Votes / acceptance / scoring
    # ------------------------------------------------------------------

    def build_direction_votes(
        self,
        context: StrategyContext,
        *,
        sources: Sequence[FeatureSource],
        vote_weights: Mapping[FeatureSource, float] | None = None,
    ) -> list[DirectionVote]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.build_direction_votes")
        self.validate_context(context)
        return build_direction_votes(
            context,
            sources=sources,
            weights=vote_weights,
        )

    def build_domain_vote(
        self,
        *,
        source: FeatureSource,
        payload: Mapping[str, Any],
        weight: float = 1.0,
        reason: str | None = None,
    ) -> DirectionVote:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.build_domain_vote")
        return build_direction_vote(
            source=source,
            payload=payload,
            weight=weight,
            reason=reason,
        )

    def accepts_hybrid_snapshot(
        self,
        snapshot: HybridCompositeSnapshot,
        *,
        required_sources: Sequence[FeatureSource] = (),
        min_score: float | None = None,
        min_confidence: float | None = None,
        min_alignment_score: float | None = None,
        min_confluence_score: float | None = None,
        max_conflict_score: float | None = None,
        allow_missing: int | None = None,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.accepts_hybrid_snapshot")
        if not snapshot.has_minimum_data():
            return False

        if required_sources:
            if not snapshot.has_all_domains(
                required_sources,
                allow_missing=(
                    self.hybrid_config.allow_missing_required_domains
                    if allow_missing is None
                    else allow_missing
                ),
            ):
                return False

        if snapshot.domain_count < self.hybrid_config.min_required_domains:
            return False

        if self.hybrid_config.require_same_side_alignment:
            if not is_directional_side(snapshot.dominant_side):
                return self.hybrid_config.allow_unknown_side

        if self.hybrid_config.reject_direct_conflicts:
            threshold = (
                self.hybrid_config.max_conflict_score
                if max_conflict_score is None
                else max_conflict_score
            )
            if snapshot.conflict_score > threshold:
                return False

        if snapshot.alignment_score < (
            self.hybrid_config.min_alignment_score
            if min_alignment_score is None
            else min_alignment_score
        ):
            return False

        if snapshot.confluence_score < (
            self.hybrid_config.min_confluence_score
            if min_confluence_score is None
            else min_confluence_score
        ):
            return False

        if snapshot.confidence < (
            self.hybrid_config.min_confidence
            if min_confidence is None
            else min_confidence
        ):
            return False

        if min_score is not None:
            base_score = unit_score(
                snapshot.confluence_score * 0.5
                + snapshot.alignment_score * 0.3
                + (1.0 - snapshot.conflict_score) * 0.2
            )
            if base_score < min_score:
                return False

        return True

    def build_hybrid_score_breakdown(
        self,
        *,
        context: StrategyContext,
        snapshot: HybridCompositeSnapshot,
        side: SignalSide,
        min_domains: int | None = None,
        weights: Mapping[str, float] | None = None,
        reasons: list[str] | None = None,
        confirmations: list[str] | None = None,
    ) -> HybridScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.build_hybrid_score_breakdown")
        return build_hybrid_score_breakdown(
            votes=snapshot.votes,
            side=side,
            payloads=snapshot.payloads(),
            now=context.timestamp,
            stale_after_seconds=self.hybrid_config.stale_feature_max_age_seconds,
            min_domains=min_domains or self.hybrid_config.min_required_domains,
            weights=weights,
            reasons=reasons,
            confirmations=confirmations,
        )

    # ------------------------------------------------------------------
    # Signal builder
    # ------------------------------------------------------------------

    def build_hybrid_signal(
        self,
        *,
        context: StrategyContext,
        side: SignalSide,
        confidence: float,
        score: float,
        setup_type: SetupType | None = None,
        reasons: list[str] | None = None,
        confirmations: list[str] | None = None,
        source_features: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        priority: SignalPriority = SignalPriority.HIGH,
        trigger_type: TriggerType = TriggerType.PRIMARY,
        origin: SignalOrigin = SignalOrigin.SINGLE_STRATEGY,
        status: SignalStatus = SignalStatus.NEW,
    ) -> StrategySignal:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.build_hybrid_signal")
        if not is_directional_side(side):
            raise StrategyEvaluationError(
                f"{self.strategy_name}: hybrid signal side must be LONG or SHORT"
            )

        scope = self.hybrid_scope(context)

        signal_metadata = dict(metadata or {})
        signal_metadata.setdefault("domain", FeatureSource.EXTERNAL.value)
        signal_metadata.setdefault("hybrid_strategy_version", HYBRID_CONTEXT_VERSION)
        signal_metadata.setdefault(
            "order_intent",
            self.hybrid_config.default_order_intent.value,
        )
        signal_metadata.setdefault(
            "margin_mode",
            self.hybrid_config.default_margin_mode.value,
        )
        signal_metadata.setdefault(
            "market_type",
            self.hybrid_config.default_market_type.value,
        )
        signal_metadata.setdefault(
            "tier",
            self.hybrid_config.default_trade_tier.value,
        )
        signal_metadata.setdefault("processor_boundary", "SignalProcessor")

        if self.hybrid_config.requested_leverage is not None:
            signal_metadata.setdefault(
                "requested_leverage",
                float(self.hybrid_config.requested_leverage),
            )

        if self.hybrid_config.max_slippage_bps is not None:
            signal_metadata.setdefault(
                "max_slippage_bps",
                float(self.hybrid_config.max_slippage_bps),
            )

        if self.hybrid_config.entry_timeout_seconds is not None:
            signal_metadata.setdefault(
                "entry_timeout_seconds",
                int(self.hybrid_config.entry_timeout_seconds),
            )

        if self.hybrid_config.max_holding_seconds is not None:
            signal_metadata.setdefault(
                "max_holding_seconds",
                int(self.hybrid_config.max_holding_seconds),
            )

        if self.hybrid_config.attach_scope_metadata:
            signal_metadata.setdefault("scope", scope.to_dict())

        if self.hybrid_config.attach_hybrid_context_metadata:
            signal_metadata.setdefault(
                "hybrid_context",
                self.hybrid_context_metadata(context),
            )

        if self.hybrid_config.metadata:
            signal_metadata.setdefault(
                "hybrid_config_metadata",
                serialize_for_metadata(self.hybrid_config.metadata),
            )

        final_reasons = list(
            dict.fromkeys(
                [
                    "hybrid_strategy_signal",
                    *(reasons or []),
                ]
            )
        )
        final_confirmations = list(dict.fromkeys(confirmations or []))
        final_features = list(
            dict.fromkeys(source_features or hybrid_base_source_features())
        )

        signal = self.build_signal(
            context=context,
            side=side,
            confidence=confidence,
            score=score,
            setup_type=setup_type or self.default_setup_type,
            reasons=final_reasons,
            confirmations=final_confirmations,
            source_features=final_features,
            metadata=serialize_for_metadata(signal_metadata),
            trigger_type=trigger_type,
            origin=origin,
            priority=priority,
            status=status,
        )

        signal.validate()
        return signal

    def hybrid_context_metadata(
        self,
        context: StrategyContext,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy.hybrid_context_metadata")
        metadata: dict[str, Any] = {}

        snapshot = self.resolve_hybrid_snapshot(context)
        if snapshot is not None:
            metadata["snapshot"] = snapshot.to_dict()

            if self.hybrid_config.attach_domain_snapshots_metadata:
                metadata["domain_payloads"] = {
                    source.value: serialize_for_metadata(payload)
                    for source, payload in snapshot.payloads().items()
                    if payload
                }

            if self.hybrid_config.attach_votes_metadata:
                metadata["votes"] = [vote.to_dict() for vote in snapshot.votes]

        metadata["feature_values"] = {
            "dominant_side": self._feature_value(context, HYBRID_FEATURES.DOMINANT_SIDE),
            "alignment_score": self._feature_value(
                context,
                HYBRID_FEATURES.ALIGNMENT_SCORE,
            ),
            "conflict_score": self._feature_value(
                context,
                HYBRID_FEATURES.CONFLICT_SCORE,
            ),
            "confluence_score": self._feature_value(
                context,
                HYBRID_FEATURES.CONFLUENCE_SCORE,
            ),
            "confidence": self._feature_value(context, HYBRID_FEATURES.CONFIDENCE),
        }

        return serialize_for_metadata(metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_value(context: StrategyContext, feature_name: str) -> Any:
        _strategy_logger = logging.getLogger(__name__ + ".HybridTradingStrategy._feature_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy._feature_value")
        if not isinstance(feature_name, str) or not feature_name.strip():
            return None

        if not context.has_feature(feature_name):
            return None

        value = context.get_feature(feature_name)

        if isinstance(value, FeatureSnapshot):
            return value.value

        return value

    @staticmethod
    def _has_any_value(value: Any) -> bool:
        _strategy_logger = logging.getLogger(__name__ + ".HybridTradingStrategy._has_any_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy._has_any_value")
        if value is None:
            return False

        if isinstance(value, Mapping):
            return any(
                HybridTradingStrategy._has_any_value(item)
                for item in value.values()
            )

        if isinstance(value, (list, tuple, set)):
            return any(
                HybridTradingStrategy._has_any_value(item)
                for item in value
            )

        return value is not None

    @staticmethod
    def _first_non_empty_symbol(
        payloads: Mapping[FeatureSource, Mapping[str, Any]],
    ) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".HybridTradingStrategy._first_non_empty_symbol")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy._first_non_empty_symbol")
        for payload in payloads.values():
            symbol = extract_domain_symbol(payload)
            if symbol:
                return symbol
        return ""

    @staticmethod
    def _first_non_empty_exchange(
        payloads: Mapping[FeatureSource, Mapping[str, Any]],
    ) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".HybridTradingStrategy._first_non_empty_exchange")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy._first_non_empty_exchange")
        for payload in payloads.values():
            exchange = extract_domain_exchange(payload)
            if exchange:
                return exchange
        return ""

    @staticmethod
    def _first_non_empty_market_type(
        payloads: Mapping[FeatureSource, Mapping[str, Any]],
    ) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".HybridTradingStrategy._first_non_empty_market_type")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy._first_non_empty_market_type")
        for payload in payloads.values():
            market_type = extract_domain_market_type(payload)
            if market_type:
                return market_type
        return ""

    @staticmethod
    def _first_non_empty_timeframe(
        payloads: Mapping[FeatureSource, Mapping[str, Any]],
    ) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".HybridTradingStrategy._first_non_empty_timeframe")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridTradingStrategy._first_non_empty_timeframe")
        for payload in payloads.values():
            timeframe = extract_domain_timeframe(payload, "")
            if timeframe:
                return timeframe
        return ""


# Backward-compatible aliases while concrete hybrid strategies are migrated.
HybridStrategyBase = HybridTradingStrategy
BaseHybridStrategy = HybridTradingStrategy
BaseHybridStrategyConfig = HybridStrategyConfig