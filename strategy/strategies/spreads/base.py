# trading_system/strategy/strategies/spreads/base.py

from __future__ import annotations
import logging

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime
from decimal import Decimal
from typing import Any

from analytics.spreads.enums import (
    OpportunityStatus,
    QuoteValidity,
    SpreadDirection,
    SpreadRegime,
    SpreadSignalType,
    SpreadType,
)
from core.event_bus import EventBus
from core.scheduler import Scheduler
from .utils import (
    DECIMAL_ZERO,
    as_dict,
    base_spreads_source_features,
    basis_to_bias,
    basis_to_signal_side,
    cross_exchange_leg_metadata,
    cross_exchange_to_signal_side,
    extract_arbitrage_opportunity_payload,
    extract_basis,
    extract_confidence,
    extract_direction,
    extract_exchange_a,
    extract_exchange_b,
    extract_exchange_symbol_a,
    extract_exchange_symbol_b,
    extract_funding_adjusted_spread,
    extract_has_edge,
    extract_market_type_a,
    extract_market_type_b,
    extract_metadata,
    extract_net_edge,
    extract_net_edge_bps,
    extract_opportunity_key,
    extract_persistence_ms,
    extract_quote_validity,
    extract_regime,
    extract_signal_type,
    extract_spread_bps,
    extract_spread_signal_payload,
    extract_spread_snapshot_payload,
    extract_spread_type,
    extract_status,
    extract_symbol,
    extract_timestamp,
    extract_timeframe,
    extract_zscore,
    is_directional_side,
    normalize_exchange,
    normalize_label,
    normalize_symbol,
    opportunity_is_active,
    opportunity_is_tradeable,
    parse_datetime,
    quote_is_valid,
    serialize_for_metadata,
    spread_quality_filter_reason,
    spreads_domain,
    spreads_item,
    spreads_path,
    to_bool,
    to_decimal,
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
class SpreadsFeatureNames:
    """
    Stable spreads feature names expected in StrategyContext.

    StrategyContextBuilder / SignalNormalizer should populate these from
    analytics.spreads.* payloads. Concrete strategies may also read equivalent
    values from FeatureSource.SPREADS domain_data aliases.
    """
    _logger = logging.getLogger(__name__ + ".SpreadsFeatureNames")

    SNAPSHOT: str = "spreads.snapshot"
    SIGNAL: str = "spreads.signal"
    OPPORTUNITY: str = "spreads.opportunity"

    SPREAD_TYPE: str = "spreads.type"
    SYMBOL: str = "spreads.symbol"

    EXCHANGE_A: str = "spreads.exchange_a"
    EXCHANGE_B: str = "spreads.exchange_b"
    MARKET_TYPE_A: str = "spreads.market_type_a"
    MARKET_TYPE_B: str = "spreads.market_type_b"
    EXCHANGE_SYMBOL_A: str = "spreads.exchange_symbol_a"
    EXCHANGE_SYMBOL_B: str = "spreads.exchange_symbol_b"

    SPREAD_BPS: str = "spreads.spread_bps"
    BASIS: str = "spreads.basis"
    FUNDING_ADJUSTED_SPREAD: str = "spreads.funding_adjusted_spread"
    NET_EDGE: str = "spreads.net_edge"
    NET_EDGE_BPS: str = "spreads.net_edge_bps"
    ZSCORE: str = "spreads.zscore"

    REGIME: str = "spreads.regime"
    DIRECTION: str = "spreads.direction"
    SIGNAL_TYPE: str = "spreads.signal_type"
    QUOTE_VALIDITY: str = "spreads.quote_validity"
    HAS_EDGE: str = "spreads.has_edge"
    CONFIDENCE: str = "spreads.confidence"

    OPPORTUNITY_KEY: str = "spreads.opportunity_key"
    OPPORTUNITY_STATUS: str = "spreads.opportunity_status"
    PERSISTENCE_MS: str = "spreads.persistence_ms"

    BUY_EXCHANGE: str = "spreads.buy_exchange"
    SELL_EXCHANGE: str = "spreads.sell_exchange"
    BUY_MARKET_TYPE: str = "spreads.buy_market_type"
    SELL_MARKET_TYPE: str = "spreads.sell_market_type"

    LEG_A_INSTRUMENT_TYPE: str = "spreads.leg_a.instrument_type"
    LEG_B_INSTRUMENT_TYPE: str = "spreads.leg_b.instrument_type"
    INSTRUMENT_TYPE: str = "spreads.instrument_type"

    METADATA: str = "spreads.metadata"

    @classmethod
    def all(cls) -> set[str]:
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".SpreadsFeatureNames")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsFeatureNames.all")
        instance = cls()
        return {
            getattr(instance, item.name)
            for item in fields(cls)
            if isinstance(getattr(instance, item.name), str)
            and getattr(instance, item.name).strip()
        }


SPREADS_FEATURES = SpreadsFeatureNames()


# =============================================================================
# Scope
# =============================================================================


@dataclass(frozen=True, slots=True)
class SpreadsStrategyScope:
    """
    Spread scope used only for strategy metadata and normalization.

    Concrete strategies still make decisions from StrategyContext.
    """
    _logger = logging.getLogger(__name__ + ".SpreadsStrategyScope")

    spread_type: str
    symbol: str
    exchange_a: str
    exchange_b: str
    market_type_a: str | None = None
    market_type_b: str | None = None
    timeframe: str = Timeframe.M1.value
    exchange_symbol_a: str | None = None
    exchange_symbol_b: str | None = None

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsStrategyScope.__post_init__")
        spread_type = str(self.spread_type or "unknown").strip().lower()
        symbol = normalize_symbol(self.symbol)
        exchange_a = normalize_exchange(self.exchange_a)
        exchange_b = normalize_exchange(self.exchange_b)
        market_type_a = to_str(self.market_type_a)
        market_type_b = to_str(self.market_type_b)
        timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()

        exchange_symbol_a = (
            to_str(self.exchange_symbol_a)
            or symbol
        )
        exchange_symbol_b = (
            to_str(self.exchange_symbol_b)
            or symbol
        )

        if not symbol:
            raise StrategyEvaluationError("SpreadsStrategyScope.symbol cannot be empty")

        object.__setattr__(self, "spread_type", spread_type)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "exchange_a", exchange_a or "unknown")
        object.__setattr__(self, "exchange_b", exchange_b or "unknown")
        object.__setattr__(self, "market_type_a", market_type_a)
        object.__setattr__(self, "market_type_b", market_type_b)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "exchange_symbol_a", exchange_symbol_a)
        object.__setattr__(self, "exchange_symbol_b", exchange_symbol_b)

    @property
    def key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsStrategyScope.key")
        return (
            f"{self.spread_type}:"
            f"{self.exchange_a}:{self.market_type_a or 'na'}:"
            f"{self.exchange_b}:{self.market_type_b or 'na'}:"
            f"{self.symbol}:{self.timeframe}"
        )

    @property
    def legacy_key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsStrategyScope.legacy_key")
        return f"{self.symbol}:{self.exchange_a}:{self.exchange_b}"

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsStrategyScope.to_dict")
        return {
            "spread_type": self.spread_type,
            "symbol": self.symbol,
            "exchange_a": self.exchange_a,
            "exchange_b": self.exchange_b,
            "market_type_a": self.market_type_a,
            "market_type_b": self.market_type_b,
            "timeframe": self.timeframe,
            "exchange_symbol_a": self.exchange_symbol_a,
            "exchange_symbol_b": self.exchange_symbol_b,
            "key": self.key,
            "legacy_key": self.legacy_key,
        }


# =============================================================================
# Config
# =============================================================================


@dataclass(slots=True)
class SpreadsStrategyConfig:
    """
    Stateless domain config shared by concrete spreads strategies.

    No EventBus topics, no local setup lifecycle, no cleanup jobs, no pending/open
    state indexes. Those concerns belong to engine/processor/state layers.
    """
    _logger = logging.getLogger(__name__ + ".SpreadsStrategyConfig")

    min_score: float = 0.60
    min_confidence: float = 0.55
    stale_feature_max_age_seconds: float | None = None

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

    attach_spread_context_metadata: bool = True
    attach_scope_metadata: bool = True
    attach_leg_metadata: bool = True
    attach_raw_payload_metadata: bool = True

    tag_spreads: str = "spreads"
    tag_spot_futures: str = "spot_futures"
    tag_cross_exchange: str = "cross_exchange"
    tag_basis: str = "basis"
    tag_arbitrage: str = "arbitrage"
    tag_mean_reversion: str = "mean_reversion"
    tag_momentum: str = "momentum"
    tag_funding_adjusted: str = "funding_adjusted"

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsStrategyConfig.validate")
        bounded = {
            "min_score": self.min_score,
            "min_confidence": self.min_confidence,
        }
        for name, value in bounded.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{name} must be between 0.0 and 1.0")

        if (
            self.stale_feature_max_age_seconds is not None
            and self.stale_feature_max_age_seconds <= 0
        ):
            raise StrategyConfigError("stale_feature_max_age_seconds must be > 0")

        if self.requested_leverage is not None and self.requested_leverage <= 0:
            raise StrategyConfigError("requested_leverage must be > 0")

        if self.max_slippage_bps is not None and self.max_slippage_bps < 0:
            raise StrategyConfigError("max_slippage_bps must be >= 0")

        if self.entry_timeout_seconds is not None and self.entry_timeout_seconds <= 0:
            raise StrategyConfigError("entry_timeout_seconds must be > 0")

        if self.max_holding_seconds is not None and self.max_holding_seconds <= 0:
            raise StrategyConfigError("max_holding_seconds must be > 0")

        for attr in (
            "tag_spreads",
            "tag_spot_futures",
            "tag_cross_exchange",
            "tag_basis",
            "tag_arbitrage",
            "tag_mean_reversion",
            "tag_momentum",
            "tag_funding_adjusted",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


# =============================================================================
# Composite snapshot
# =============================================================================


@dataclass(slots=True)
class SpreadCompositeSnapshot:
    """
    Strategy-level normalized view over analytics.spreads payloads.

    This is intentionally not an analytics model. It is a strategy-side
    projection that lets concrete spread strategies consume snapshot, signal and
    opportunity payloads through one stable contract.
    """
    _logger = logging.getLogger(__name__ + ".SpreadCompositeSnapshot")

    spread_type: SpreadType | None
    symbol: str

    exchange_a: str
    exchange_b: str
    market_type_a: str | None = None
    market_type_b: str | None = None
    exchange_symbol_a: str | None = None
    exchange_symbol_b: str | None = None
    timeframe: str = Timeframe.M1.value

    spread_bps: Decimal | None = None
    basis: Decimal | None = None
    funding_adjusted_spread: Decimal | None = None
    net_edge: Decimal | None = None
    net_edge_bps: Decimal | None = None
    zscore: Decimal | None = None

    regime: SpreadRegime | None = None
    direction: SpreadDirection | None = None
    signal_type: SpreadSignalType | None = None
    quote_validity: QuoteValidity | None = None
    has_edge: bool | None = None

    opportunity_status: OpportunityStatus | None = None
    opportunity_key: str | None = None
    persistence_ms: int = 0

    confidence: float = 0.0
    timestamp: datetime | None = None
    source: str = "unknown"

    raw_snapshot: dict[str, Any] = field(default_factory=dict)
    raw_signal: dict[str, Any] = field(default_factory=dict)
    raw_opportunity: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.__post_init__")
        self.symbol = normalize_symbol(self.symbol)
        self.exchange_a = normalize_exchange(self.exchange_a) or "unknown"
        self.exchange_b = normalize_exchange(self.exchange_b) or "unknown"

        self.market_type_a = to_str(self.market_type_a)
        self.market_type_b = to_str(self.market_type_b)

        self.exchange_symbol_a = to_str(self.exchange_symbol_a) or self.symbol
        self.exchange_symbol_b = to_str(self.exchange_symbol_b) or self.symbol
        self.timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()

        if not self.symbol:
            raise StrategyEvaluationError("SpreadCompositeSnapshot.symbol cannot be empty")

        self.spread_bps = to_decimal(self.spread_bps)
        self.basis = to_decimal(self.basis)
        self.funding_adjusted_spread = to_decimal(self.funding_adjusted_spread)
        self.net_edge = to_decimal(self.net_edge)
        self.net_edge_bps = to_decimal(self.net_edge_bps)
        self.zscore = to_decimal(self.zscore)

        self.has_edge = None if self.has_edge is None else bool(self.has_edge)
        self.persistence_ms = max(0, int(to_int(self.persistence_ms, 0) or 0))
        self.confidence = unit_score(self.confidence)

        self.timestamp = parse_datetime(self.timestamp)

        self.raw_snapshot = as_dict(self.raw_snapshot)
        self.raw_signal = as_dict(self.raw_signal)
        self.raw_opportunity = as_dict(self.raw_opportunity)
        self.metadata = as_dict(self.metadata)

    @property
    def spread_type_label(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.spread_type_label")
        return normalize_label(self.spread_type)

    @property
    def scope(self) -> SpreadsStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.scope")
        return SpreadsStrategyScope(
            spread_type=self.spread_type_label or "unknown",
            symbol=self.symbol,
            exchange_a=self.exchange_a,
            exchange_b=self.exchange_b,
            market_type_a=self.market_type_a,
            market_type_b=self.market_type_b,
            timeframe=self.timeframe,
            exchange_symbol_a=self.exchange_symbol_a,
            exchange_symbol_b=self.exchange_symbol_b,
        )

    @property
    def scope_key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.scope_key")
        return self.scope.key

    @property
    def basis_bias(self) -> str | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.basis_bias")
        return basis_to_bias(self.to_signal_payload())

    @property
    def basis_side(self) -> SignalSide:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.basis_side")
        return basis_to_signal_side(self.to_signal_payload())

    @property
    def cross_exchange_side(self) -> SignalSide:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.cross_exchange_side")
        return cross_exchange_to_signal_side(self.raw_opportunity or self.to_signal_payload())

    @property
    def is_quote_valid(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.is_quote_valid")
        return quote_is_valid(self.to_signal_payload())

    @property
    def tradeable_edge(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.tradeable_edge")
        if self.has_edge is not None:
            return self.has_edge

        return any(
            value is not None and value != DECIMAL_ZERO
            for value in (
                self.funding_adjusted_spread,
                self.net_edge,
                self.net_edge_bps,
                self.basis,
                self.spread_bps,
            )
        )

    @property
    def abs_zscore(self) -> Decimal:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.abs_zscore")
        return abs(self.zscore) if self.zscore is not None else DECIMAL_ZERO

    @property
    def abs_edge(self) -> Decimal:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.abs_edge")
        for value in (
            self.funding_adjusted_spread,
            self.net_edge,
            self.net_edge_bps,
            self.basis,
            self.spread_bps,
        ):
            if value is not None:
                return abs(value)
        return DECIMAL_ZERO

    @property
    def opportunity_active(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.opportunity_active")
        if self.raw_opportunity:
            return opportunity_is_active(self.raw_opportunity)

        if self.opportunity_status is None:
            return True

        label = normalize_label(self.opportunity_status)
        return label in {"active", "open", "detected", "tradeable"}

    @property
    def opportunity_tradeable(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.opportunity_tradeable")
        if self.raw_opportunity:
            return opportunity_is_tradeable(self.raw_opportunity)
        return self.opportunity_active

    def has_minimum_data(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.has_minimum_data")
        return (
            self.spread_type is not None
            or self.spread_bps is not None
            or self.basis is not None
            or self.funding_adjusted_spread is not None
            or self.net_edge is not None
            or self.net_edge_bps is not None
            or self.zscore is not None
            or self.confidence > 0.0
            or bool(self.raw_snapshot)
            or bool(self.raw_signal)
            or bool(self.raw_opportunity)
        )

    def to_signal_payload(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.to_signal_payload")
        return {
            "spread_type": self.spread_type,
            "symbol": self.symbol,
            "exchange_a": self.exchange_a,
            "exchange_b": self.exchange_b,
            "market_type_a": self.market_type_a,
            "market_type_b": self.market_type_b,
            "exchange_symbol_a": self.exchange_symbol_a,
            "exchange_symbol_b": self.exchange_symbol_b,
            "timeframe": self.timeframe,
            "spread_bps": self.spread_bps,
            "basis": self.basis,
            "funding_adjusted_spread": self.funding_adjusted_spread,
            "net_edge": self.net_edge,
            "net_edge_bps": self.net_edge_bps,
            "zscore": self.zscore,
            "regime": self.regime,
            "direction": self.direction,
            "signal_type": self.signal_type,
            "quote_validity": self.quote_validity,
            "has_edge": self.has_edge,
            "opportunity_status": self.opportunity_status,
            "opportunity_key": self.opportunity_key,
            "persistence_ms": self.persistence_ms,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadCompositeSnapshot.to_dict")
        return {
            "spread_type": self.spread_type.value if self.spread_type else None,
            "symbol": self.symbol,
            "exchange_a": self.exchange_a,
            "exchange_b": self.exchange_b,
            "market_type_a": self.market_type_a,
            "market_type_b": self.market_type_b,
            "exchange_symbol_a": self.exchange_symbol_a,
            "exchange_symbol_b": self.exchange_symbol_b,
            "timeframe": self.timeframe,
            "spread_bps": str(self.spread_bps) if self.spread_bps is not None else None,
            "basis": str(self.basis) if self.basis is not None else None,
            "funding_adjusted_spread": (
                str(self.funding_adjusted_spread)
                if self.funding_adjusted_spread is not None
                else None
            ),
            "net_edge": str(self.net_edge) if self.net_edge is not None else None,
            "net_edge_bps": (
                str(self.net_edge_bps) if self.net_edge_bps is not None else None
            ),
            "zscore": str(self.zscore) if self.zscore is not None else None,
            "regime": self.regime.value if self.regime else None,
            "direction": self.direction.value if self.direction else None,
            "signal_type": self.signal_type.value if self.signal_type else None,
            "quote_validity": self.quote_validity.value if self.quote_validity else None,
            "has_edge": self.has_edge,
            "opportunity_status": (
                self.opportunity_status.value if self.opportunity_status else None
            ),
            "opportunity_key": self.opportunity_key,
            "persistence_ms": self.persistence_ms,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
            "scope": self.scope.to_dict(),
            "scope_key": self.scope_key,
            "basis_bias": self.basis_bias,
            "basis_side": self.basis_side.value,
            "cross_exchange_side": self.cross_exchange_side.value,
            "is_quote_valid": self.is_quote_valid,
            "tradeable_edge": self.tradeable_edge,
            "abs_zscore": str(self.abs_zscore),
            "abs_edge": str(self.abs_edge),
            "opportunity_active": self.opportunity_active,
            "opportunity_tradeable": self.opportunity_tradeable,
            "raw_snapshot": serialize_for_metadata(self.raw_snapshot),
            "raw_signal": serialize_for_metadata(self.raw_signal),
            "raw_opportunity": serialize_for_metadata(self.raw_opportunity),
            "metadata": serialize_for_metadata(self.metadata),
        }


# =============================================================================
# Base spreads strategy
# =============================================================================


class SpreadsTradingStrategy(TradingStrategy):
    """
    Base class for concrete strategy/strategies/spreads/* classes.

    Responsibilities:
    - read spread analytics data from StrategyContext only;
    - provide helpers for snapshots, signals, opportunities and scoring metadata;
    - build internal StrategySignal objects through TradingStrategy helpers;
    - attach spread/futures metadata for SignalProcessor.

    Forbidden:
    - no direct analytics.spreads.* EventBus subscriptions;
    - no local latest snapshot/signal/opportunity caches;
    - no pending/open/closed setup lifecycle state;
    - no direct signal.generated / signal.updated / signal.closed emission;
    - no RiskManager / Execution calls;
    - no raw market data reads.
    """
    _logger = logging.getLogger(__name__ + ".SpreadsTradingStrategy")

    component_namespace = "strategy.spreads"
    category: StrategyCategory = StrategyCategory.SPREADS
    default_setup_type: SetupType = SetupType.UNKNOWN
    default_timeframe: Timeframe = Timeframe.M1

    feature_names = SPREADS_FEATURES

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        spreads_config: SpreadsStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.__init__")
        self.spreads_config = spreads_config or SpreadsStrategyConfig()
        self.spreads_config.validate()

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
            _strategy_logger.debug("Entering SpreadsTradingStrategy.validate_config")
        super().validate_config()
        self.spreads_config.validate()

    # ------------------------------------------------------------------
    # Context / domain access
    # ------------------------------------------------------------------

    def spreads_domain(self, context: StrategyContext) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_domain")
        self.validate_context(context)
        return spreads_domain(context)

    def spreads_item(
        self,
        context: StrategyContext,
        key: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_item")
        self.validate_context(context)
        return spreads_item(context, key, default)

    def spreads_path(
        self,
        context: StrategyContext,
        path: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_path")
        self.validate_context(context)

        if not isinstance(path, str) or not path.strip():
            raise StrategyEvaluationError("spreads path cannot be empty")

        return spreads_path(context, path, default)

    def spreads_decimal(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: Decimal | None = None,
    ) -> Decimal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_decimal")
        return to_decimal(self.spreads_path(context, path, default), default)

    def spreads_float(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float | None = None,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_float")
        return to_float(self.spreads_path(context, path, default), default)

    def spreads_int(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: int | None = None,
    ) -> int | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_int")
        return to_int(self.spreads_path(context, path, default), default)

    def spreads_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_score")
        return unit_score(self.spreads_path(context, path, default), default)

    def spreads_bool(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: bool = False,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_bool")
        return to_bool(self.spreads_path(context, path, default), default)

    def spreads_str(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: str | None = None,
    ) -> str | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_str")
        return to_str(self.spreads_path(context, path, default), default)

    def spreads_datetime(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: datetime | None = None,
    ) -> datetime | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_datetime")
        return parse_datetime(self.spreads_path(context, path, default))

    def spreads_feature_snapshot(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> FeatureSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_feature_snapshot")
        self.validate_context(context)

        if not isinstance(feature_name, str) or not feature_name.strip():
            raise StrategyEvaluationError("feature_name cannot be empty")

        features_map = getattr(context, "features", None)
        if isinstance(features_map, Mapping):
            raw = features_map.get(feature_name)
            if isinstance(raw, FeatureSnapshot):
                return raw

        return None

    def spreads_feature_age_seconds(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_feature_age_seconds")
        snapshot = self.spreads_feature_snapshot(context, feature_name)
        if snapshot is None:
            return None
        return snapshot.age_seconds(context.timestamp)

    def spreads_feature_is_stale(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_feature_is_stale")
        max_age = self.spreads_config.stale_feature_max_age_seconds
        if max_age is None:
            return False

        age = self.spreads_feature_age_seconds(context, feature_name)
        if age is None:
            return False

        return age > max_age

    def has_any_spreads_data(
        self,
        context: StrategyContext,
        feature_names: tuple[str, ...] = (),
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.has_any_spreads_data")
        self.validate_context(context)

        if self.spreads_domain(context):
            return True

        return any(context.has_feature(name) for name in feature_names)

    def has_stale_spreads_features(
        self,
        context: StrategyContext,
        feature_names: tuple[str, ...] | None = None,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.has_stale_spreads_features")
        names = feature_names or tuple(self.required_features())

        return any(
            self.spreads_feature_is_stale(context, feature_name)
            for feature_name in names
        )

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------

    def spreads_scope(self, context: StrategyContext) -> SpreadsStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spreads_scope")
        domain = self.spreads_domain(context)

        snapshot_candidate = (
            self.spreads_item(context, "snapshot")
            or self.spreads_item(context, "opportunity")
            or self.spreads_item(context, "signal")
            or domain
        )

        spread_type = (
            extract_spread_type(snapshot_candidate)
            or extract_spread_type(domain)
        )
        symbol = (
            extract_symbol(snapshot_candidate)
            or normalize_symbol(context.symbol)
        )
        exchange_a = (
            extract_exchange_a(snapshot_candidate)
            or normalize_exchange(context.metadata.get("exchange_a"))
            or "unknown"
        )
        exchange_b = (
            extract_exchange_b(snapshot_candidate)
            or normalize_exchange(context.metadata.get("exchange_b"))
            or "unknown"
        )

        market_type_a = (
            extract_market_type_a(snapshot_candidate)
            or to_str(context.metadata.get("market_type_a"))
        )
        market_type_b = (
            extract_market_type_b(snapshot_candidate)
            or to_str(context.metadata.get("market_type_b"))
        )

        timeframe_value = getattr(context.timeframe, "value", context.timeframe)

        return SpreadsStrategyScope(
            spread_type=normalize_label(spread_type) or "unknown",
            symbol=symbol,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            market_type_a=market_type_a,
            market_type_b=market_type_b,
            timeframe=str(timeframe_value or Timeframe.M1.value),
            exchange_symbol_a=extract_exchange_symbol_a(snapshot_candidate) or symbol,
            exchange_symbol_b=extract_exchange_symbol_b(snapshot_candidate) or symbol,
        )

    # ------------------------------------------------------------------
    # Snapshot / signal / opportunity resolution
    # ------------------------------------------------------------------

    def resolve_spread_snapshot(
        self,
        context: StrategyContext,
    ) -> SpreadCompositeSnapshot | None:
        """
        Resolve normalized spread snapshot from StrategyContext only.

        Preferred:
            FeatureSource.SPREADS domain["snapshot"] / domain["signal"] /
            domain["opportunity"].

        Fallback:
            individual spreads.* features.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.resolve_spread_snapshot")
        self.validate_context(context)

        scope = self.spreads_scope(context)

        for candidate, source in (
            (self.spreads_item(context, "snapshot"), "context.domain.snapshot"),
            (self.spreads_item(context, "signal"), "context.domain.signal"),
            (self.spreads_item(context, "opportunity"), "context.domain.opportunity"),
            (self.spreads_path(context, "snapshot", None), "context.path.snapshot"),
            (self.spreads_path(context, "signal", None), "context.path.signal"),
            (self.spreads_path(context, "opportunity", None), "context.path.opportunity"),
        ):
            snapshot = self._coerce_spread_snapshot(
                candidate,
                context=context,
                scope=scope,
                source=source,
            )
            if snapshot is not None and snapshot.has_minimum_data():
                return snapshot

        domain_snapshot = self._build_snapshot_from_domain(
            context=context,
            scope=scope,
        )
        if domain_snapshot is not None and domain_snapshot.has_minimum_data():
            return domain_snapshot

        feature_snapshot = self._build_snapshot_from_features(
            context=context,
            scope=scope,
        )
        if feature_snapshot is not None and feature_snapshot.has_minimum_data():
            return feature_snapshot

        return domain_snapshot or feature_snapshot

    def resolve_spread_signal(
        self,
        context: StrategyContext,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.resolve_spread_signal")
        self.validate_context(context)

        candidate = (
            self.spreads_item(context, "signal")
            or self.spreads_path(context, "signal", None)
            or self._feature_value(context, SPREADS_FEATURES.SIGNAL)
        )

        return extract_spread_signal_payload(candidate) if candidate is not None else {}

    def resolve_arbitrage_opportunity(
        self,
        context: StrategyContext,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.resolve_arbitrage_opportunity")
        self.validate_context(context)

        candidate = (
            self.spreads_item(context, "opportunity")
            or self.spreads_path(context, "opportunity", None)
            or self._feature_value(context, SPREADS_FEATURES.OPPORTUNITY)
        )

        return (
            extract_arbitrage_opportunity_payload(candidate)
            if candidate is not None
            else {}
        )

    def accepts_spread_snapshot(
        self,
        snapshot: SpreadCompositeSnapshot,
        *,
        require_valid_quote: bool = False,
        require_edge: bool = False,
    ) -> bool:
        """
        Shared base acceptance gate.

        Concrete strategies should add spread-type-specific checks.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.accepts_spread_snapshot")
        if not snapshot.has_minimum_data():
            return False

        if snapshot.confidence < self.spreads_config.min_confidence:
            return False

        if require_valid_quote and not snapshot.is_quote_valid:
            return False

        if require_edge and not snapshot.tradeable_edge:
            return False

        if snapshot.timestamp is not None:
            rejection = spread_quality_filter_reason(
                snapshot.to_signal_payload(),
                min_score=0.0,
                min_confidence=self.spreads_config.min_confidence,
                require_valid_quote=require_valid_quote,
                require_edge=require_edge,
                stale_after_seconds=self.spreads_config.stale_feature_max_age_seconds,
            )
            if rejection is not None:
                return False

        return True

    def _coerce_spread_snapshot(
        self,
        value: Any,
        *,
        context: StrategyContext,
        scope: SpreadsStrategyScope,
        source: str,
    ) -> SpreadCompositeSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy._coerce_spread_snapshot")
        if value is None:
            return None

        if isinstance(value, SpreadCompositeSnapshot):
            return value

        snapshot_payload = extract_spread_snapshot_payload(value)
        signal_payload = extract_spread_signal_payload(value)
        opportunity_payload = extract_arbitrage_opportunity_payload(value)

        if not snapshot_payload and not signal_payload and not opportunity_payload:
            return None

        return self._snapshot_from_payloads(
            snapshot_payload=snapshot_payload,
            signal_payload=signal_payload,
            opportunity_payload=opportunity_payload,
            context=context,
            scope=scope,
            source=source,
        )

    def _build_snapshot_from_domain(
        self,
        *,
        context: StrategyContext,
        scope: SpreadsStrategyScope,
    ) -> SpreadCompositeSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy._build_snapshot_from_domain")
        domain = self.spreads_domain(context)
        if not domain:
            return None

        snapshot_payload = extract_spread_snapshot_payload(
            self.spreads_item(context, "snapshot") or domain
        )
        signal_payload = extract_spread_signal_payload(
            self.spreads_item(context, "signal") or {}
        )
        opportunity_payload = extract_arbitrage_opportunity_payload(
            self.spreads_item(context, "opportunity") or {}
        )

        return self._snapshot_from_payloads(
            snapshot_payload=snapshot_payload,
            signal_payload=signal_payload,
            opportunity_payload=opportunity_payload,
            context=context,
            scope=scope,
            source="context.domain",
        )

    def _build_snapshot_from_features(
        self,
        *,
        context: StrategyContext,
        scope: SpreadsStrategyScope,
    ) -> SpreadCompositeSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy._build_snapshot_from_features")
        payload = {
            "spread_type": self._feature_value(context, SPREADS_FEATURES.SPREAD_TYPE),
            "symbol": self._feature_value(context, SPREADS_FEATURES.SYMBOL),
            "exchange_a": self._feature_value(context, SPREADS_FEATURES.EXCHANGE_A),
            "exchange_b": self._feature_value(context, SPREADS_FEATURES.EXCHANGE_B),
            "market_type_a": self._feature_value(context, SPREADS_FEATURES.MARKET_TYPE_A),
            "market_type_b": self._feature_value(context, SPREADS_FEATURES.MARKET_TYPE_B),
            "exchange_symbol_a": self._feature_value(
                context,
                SPREADS_FEATURES.EXCHANGE_SYMBOL_A,
            ),
            "exchange_symbol_b": self._feature_value(
                context,
                SPREADS_FEATURES.EXCHANGE_SYMBOL_B,
            ),
            "spread_bps": self._feature_value(context, SPREADS_FEATURES.SPREAD_BPS),
            "basis": self._feature_value(context, SPREADS_FEATURES.BASIS),
            "funding_adjusted_spread": self._feature_value(
                context,
                SPREADS_FEATURES.FUNDING_ADJUSTED_SPREAD,
            ),
            "net_edge": self._feature_value(context, SPREADS_FEATURES.NET_EDGE),
            "net_edge_bps": self._feature_value(context, SPREADS_FEATURES.NET_EDGE_BPS),
            "zscore": self._feature_value(context, SPREADS_FEATURES.ZSCORE),
            "regime": self._feature_value(context, SPREADS_FEATURES.REGIME),
            "direction": self._feature_value(context, SPREADS_FEATURES.DIRECTION),
            "signal_type": self._feature_value(context, SPREADS_FEATURES.SIGNAL_TYPE),
            "quote_validity": self._feature_value(
                context,
                SPREADS_FEATURES.QUOTE_VALIDITY,
            ),
            "has_edge": self._feature_value(context, SPREADS_FEATURES.HAS_EDGE),
            "confidence": self._feature_value(context, SPREADS_FEATURES.CONFIDENCE),
            "opportunity_key": self._feature_value(
                context,
                SPREADS_FEATURES.OPPORTUNITY_KEY,
            ),
            "opportunity_status": self._feature_value(
                context,
                SPREADS_FEATURES.OPPORTUNITY_STATUS,
            ),
            "persistence_ms": self._feature_value(
                context,
                SPREADS_FEATURES.PERSISTENCE_MS,
            ),
            "metadata": self._feature_value(context, SPREADS_FEATURES.METADATA),
        }

        if not self._has_any_value(payload):
            return None

        return self._snapshot_from_payloads(
            snapshot_payload=payload,
            signal_payload={},
            opportunity_payload={},
            context=context,
            scope=scope,
            source="context.features",
        )

    def _snapshot_from_payloads(
        self,
        *,
        snapshot_payload: Mapping[str, Any],
        signal_payload: Mapping[str, Any],
        opportunity_payload: Mapping[str, Any],
        context: StrategyContext,
        scope: SpreadsStrategyScope,
        source: str,
    ) -> SpreadCompositeSnapshot:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy._snapshot_from_payloads")
        snapshot_data = dict(snapshot_payload or {})
        signal_data = dict(signal_payload or {})
        opportunity_data = dict(opportunity_payload or {})

        merged: dict[str, Any] = {}
        merged.update(snapshot_data)
        merged.update(signal_data)
        merged.update(opportunity_data)

        metadata = {}
        metadata.update(extract_metadata(snapshot_data))
        metadata.update(extract_metadata(signal_data))
        metadata.update(extract_metadata(opportunity_data))

        timestamp = (
            extract_timestamp(merged)
            or extract_timestamp(snapshot_data)
            or extract_timestamp(signal_data)
            or extract_timestamp(opportunity_data)
            or context.timestamp
        )

        symbol = (
            extract_symbol(merged)
            or scope.symbol
            or normalize_symbol(context.symbol)
        )

        exchange_a = (
            extract_exchange_a(merged)
            or scope.exchange_a
        )
        exchange_b = (
            extract_exchange_b(merged)
            or scope.exchange_b
        )

        return SpreadCompositeSnapshot(
            spread_type=extract_spread_type(merged),
            symbol=symbol,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            market_type_a=extract_market_type_a(merged) or scope.market_type_a,
            market_type_b=extract_market_type_b(merged) or scope.market_type_b,
            exchange_symbol_a=extract_exchange_symbol_a(merged) or scope.exchange_symbol_a,
            exchange_symbol_b=extract_exchange_symbol_b(merged) or scope.exchange_symbol_b,
            timeframe=extract_timeframe(merged, scope.timeframe),
            spread_bps=extract_spread_bps(merged),
            basis=extract_basis(merged),
            funding_adjusted_spread=extract_funding_adjusted_spread(merged),
            net_edge=extract_net_edge(merged),
            net_edge_bps=extract_net_edge_bps(merged),
            zscore=extract_zscore(merged),
            regime=extract_regime(merged),
            direction=extract_direction(merged),
            signal_type=extract_signal_type(merged),
            quote_validity=extract_quote_validity(merged),
            has_edge=extract_has_edge(merged),
            opportunity_status=extract_status(merged),
            opportunity_key=extract_opportunity_key(merged),
            persistence_ms=extract_persistence_ms(merged),
            confidence=extract_confidence(merged),
            timestamp=timestamp,
            source=source,
            raw_snapshot=snapshot_data,
            raw_signal=signal_data,
            raw_opportunity=opportunity_data,
            metadata={
                **metadata,
                "scope": scope.to_dict(),
                "source": source,
            },
        )

    # ------------------------------------------------------------------
    # Signal builder
    # ------------------------------------------------------------------

    def build_spread_signal(
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
        priority: SignalPriority = SignalPriority.MEDIUM,
        trigger_type: TriggerType = TriggerType.PRIMARY,
        origin: SignalOrigin = SignalOrigin.SINGLE_STRATEGY,
        status: SignalStatus = SignalStatus.NEW,
    ) -> StrategySignal:
        """
        Build internal StrategySignal with spreads/futures metadata.

        Final risk-ready payload conversion belongs to SignalProcessor /
        SignalBuilder, not to this domain strategy.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.build_spread_signal")
        if not is_directional_side(side):
            raise StrategyEvaluationError(
                f"{self.strategy_name}: spread signal side must be LONG or SHORT"
            )

        scope = self.spreads_scope(context)

        signal_metadata = dict(metadata or {})
        signal_metadata.setdefault("domain", FeatureSource.SPREADS.value)
        signal_metadata.setdefault("spreads_strategy_version", "2.0.0")
        signal_metadata.setdefault(
            "order_intent",
            self.spreads_config.default_order_intent.value,
        )
        signal_metadata.setdefault(
            "margin_mode",
            self.spreads_config.default_margin_mode.value,
        )
        signal_metadata.setdefault(
            "market_type",
            self.spreads_config.default_market_type.value,
        )
        signal_metadata.setdefault(
            "tier",
            self.spreads_config.default_trade_tier.value,
        )

        if self.spreads_config.requested_leverage is not None:
            signal_metadata.setdefault(
                "requested_leverage",
                float(self.spreads_config.requested_leverage),
            )

        if self.spreads_config.max_slippage_bps is not None:
            signal_metadata.setdefault(
                "max_slippage_bps",
                float(self.spreads_config.max_slippage_bps),
            )

        if self.spreads_config.entry_timeout_seconds is not None:
            signal_metadata.setdefault(
                "entry_timeout_seconds",
                int(self.spreads_config.entry_timeout_seconds),
            )

        if self.spreads_config.max_holding_seconds is not None:
            signal_metadata.setdefault(
                "max_holding_seconds",
                int(self.spreads_config.max_holding_seconds),
            )

        if self.spreads_config.attach_scope_metadata:
            signal_metadata.setdefault("scope", scope.to_dict())

        if self.spreads_config.attach_spread_context_metadata:
            signal_metadata.setdefault(
                "spread_context",
                self.spread_context_metadata(context),
            )

        if self.spreads_config.metadata:
            signal_metadata.setdefault(
                "spreads_config_metadata",
                serialize_for_metadata(self.spreads_config.metadata),
            )

        final_reasons = list(
            dict.fromkeys(
                [
                    "spreads_strategy_signal",
                    *(reasons or []),
                ]
            )
        )
        final_confirmations = list(dict.fromkeys(confirmations or []))
        final_features = list(dict.fromkeys(source_features or base_spreads_source_features()))

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

    def spread_context_metadata(
        self,
        context: StrategyContext,
    ) -> dict[str, Any]:
        """
        Compact serialized spread context for StrategySignal.metadata.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy.spread_context_metadata")
        metadata: dict[str, Any] = {}

        snapshot = self.resolve_spread_snapshot(context)
        if snapshot is not None:
            metadata["snapshot"] = snapshot.to_dict()

            if self.spreads_config.attach_leg_metadata:
                if snapshot.spread_type is SpreadType.CROSS_EXCHANGE:
                    metadata["legs"] = cross_exchange_leg_metadata(
                        snapshot.raw_opportunity or snapshot.to_signal_payload()
                    )
                else:
                    metadata["legs"] = {
                        "leg_a": {
                            "exchange": snapshot.exchange_a,
                            "market_type": snapshot.market_type_a,
                            "symbol": snapshot.exchange_symbol_a,
                        },
                        "leg_b": {
                            "exchange": snapshot.exchange_b,
                            "market_type": snapshot.market_type_b,
                            "symbol": snapshot.exchange_symbol_b,
                        },
                    }

            if self.spreads_config.attach_raw_payload_metadata:
                metadata["raw"] = {
                    "snapshot": snapshot.raw_snapshot,
                    "signal": snapshot.raw_signal,
                    "opportunity": snapshot.raw_opportunity,
                }

        metadata["feature_values"] = {
            "spread_type": self.spreads_path(context, "type", None),
            "symbol": self.spreads_path(context, "symbol", None),
            "spread_bps": self.spreads_path(context, "spread_bps", None),
            "basis": self.spreads_path(context, "basis", None),
            "funding_adjusted_spread": self.spreads_path(
                context,
                "funding_adjusted_spread",
                None,
            ),
            "net_edge": self.spreads_path(context, "net_edge", None),
            "net_edge_bps": self.spreads_path(context, "net_edge_bps", None),
            "zscore": self.spreads_path(context, "zscore", None),
            "regime": self.spreads_path(context, "regime", None),
            "direction": self.spreads_path(context, "direction", None),
            "quote_validity": self.spreads_path(context, "quote_validity", None),
            "has_edge": self.spreads_path(context, "has_edge", None),
            "confidence": self.spreads_path(context, "confidence", None),
        }

        return serialize_for_metadata(metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_value(context: StrategyContext, feature_name: str) -> Any:
        _strategy_logger = logging.getLogger(__name__ + ".SpreadsTradingStrategy._feature_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy._feature_value")
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
        _strategy_logger = logging.getLogger(__name__ + ".SpreadsTradingStrategy._has_any_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadsTradingStrategy._has_any_value")
        if value is None:
            return False

        if isinstance(value, Mapping):
            return any(
                SpreadsTradingStrategy._has_any_value(item)
                for item in value.values()
            )

        if isinstance(value, (list, tuple, set)):
            return any(
                SpreadsTradingStrategy._has_any_value(item)
                for item in value
            )

        return value is not None


# Backward-compatible aliases while concrete spreads strategies are migrated.
BaseSpreadStrategy = SpreadsTradingStrategy
BaseSpreadStrategyConfig = SpreadsStrategyConfig