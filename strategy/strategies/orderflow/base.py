# trading_system/strategy/strategies/orderflow/base.py

from __future__ import annotations
import logging

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler
from .utils import (
    as_dict,
    extract_aggressive_burst_score,
    extract_aggressive_buy_ratio,
    extract_aggressive_net_notional_delta,
    extract_aggressive_net_volume_delta,
    extract_aggressive_sell_ratio,
    extract_buy_volume,
    extract_cumulative_notional_delta,
    extract_cumulative_volume_delta,
    extract_cvd_change_pct,
    extract_cvd_delta_ratio,
    extract_cvd_slope,
    extract_cvd_value,
    extract_event_time,
    extract_large_buy_trades,
    extract_large_sell_trades,
    extract_last_price,
    extract_notional_delta,
    extract_orderbook_imbalance_diff,
    extract_orderbook_imbalance_ratio,
    extract_price_change_pct,
    extract_sell_volume,
    extract_total_notional,
    extract_total_volume,
    extract_trades_count,
    extract_volume_delta,
    extract_volume_delta_ratio,
    get_path,
    orderflow_domain,
    orderflow_item,
    orderflow_path,
    parse_datetime,
    serialize_for_metadata,
    signed_imbalance_from_ratio,
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
from ...models import (
    FeatureSnapshot,
    StrategyContext,
    StrategySignal,
    clamp,
)


# =============================================================================
# Orderflow feature contract
# =============================================================================


@dataclass(frozen=True, slots=True)
class OrderflowFeatureNames:
    """
    Stable feature names expected in StrategyContext.

    Generic SignalNormalizer / StrategyContextBuilder should populate these
    names from analytics.orderflow.* payloads, while strategies can also read
    equivalent values from FeatureSource.ORDERFLOW domain_data aliases.
    """
    _logger = logging.getLogger(__name__ + ".OrderflowFeatureNames")

    COMPOSITE: str = "orderflow.composite"

    CVD: str = "orderflow.cvd"
    CVD_VALUE: str = "orderflow.cvd.value"
    CVD_DELTA_RATIO: str = "orderflow.cvd.delta_ratio"
    CVD_CHANGE_PCT: str = "orderflow.cvd.cvd_change_pct"
    CVD_SLOPE: str = "orderflow.cvd.cvd_slope"
    CVD_PRICE_CHANGE_PCT: str = "orderflow.cvd.price_change_pct"

    VOLUME_DELTA: str = "orderflow.volume_delta"
    VOLUME_DELTA_VALUE: str = "orderflow.volume_delta.volume_delta"
    VOLUME_DELTA_RATIO: str = "orderflow.volume_delta.delta_ratio"
    CUMULATIVE_VOLUME_DELTA: str = (
        "orderflow.volume_delta.cumulative_volume_delta"
    )
    NOTIONAL_DELTA: str = "orderflow.volume_delta.notional_delta"
    CUMULATIVE_NOTIONAL_DELTA: str = (
        "orderflow.volume_delta.cumulative_notional_delta"
    )

    AGGRESSIVE_TRADES: str = "orderflow.aggressive_trades"
    AGGRESSIVE_BUY_RATIO: str = "orderflow.aggressive_trades.buy_ratio"
    AGGRESSIVE_SELL_RATIO: str = "orderflow.aggressive_trades.sell_ratio"
    AGGRESSIVE_BURST_SCORE: str = "orderflow.aggressive_trades.burst_score"
    AGGRESSIVE_NET_VOLUME_DELTA: str = (
        "orderflow.aggressive_trades.net_volume_delta"
    )
    AGGRESSIVE_NET_NOTIONAL_DELTA: str = (
        "orderflow.aggressive_trades.net_notional_delta"
    )
    LARGE_BUY_TRADES: str = "orderflow.aggressive_trades.large_buy_trades"
    LARGE_SELL_TRADES: str = "orderflow.aggressive_trades.large_sell_trades"

    ORDERBOOK_IMBALANCE: str = "orderflow.orderbook_imbalance"
    ORDERBOOK_IMBALANCE_RATIO: str = "orderflow.orderbook_imbalance.ratio"
    ORDERBOOK_IMBALANCE_DIFF: str = "orderflow.orderbook_imbalance.diff"

    TRADES_COUNT: str = "orderflow.trades_count"
    TOTAL_VOLUME: str = "orderflow.total_volume"
    TOTAL_NOTIONAL: str = "orderflow.total_notional"
    LAST_PRICE: str = "orderflow.last_price"
    PRICE_CHANGE_PCT: str = "orderflow.price_change_pct"

    @classmethod
    def all(cls) -> set[str]:
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".OrderflowFeatureNames")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowFeatureNames.all")
        instance = cls()
        return {
            getattr(instance, item.name)
            for item in fields(cls)
            if isinstance(getattr(instance, item.name), str)
            and getattr(instance, item.name).strip()
        }


ORDERFLOW_FEATURES = OrderflowFeatureNames()


# =============================================================================
# Scope
# =============================================================================


@dataclass(frozen=True, slots=True)
class OrderflowStrategyScope:
    """
    Futures orderflow scope used only for metadata and normalization.

    Concrete strategies still make decisions from StrategyContext.
    """
    _logger = logging.getLogger(__name__ + ".OrderflowStrategyScope")

    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    exchange_symbol: str | None = None

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowStrategyScope.__post_init__")
        exchange = str(self.exchange or "unknown").strip().lower()
        market_type = str(
            self.market_type or StrategyMarketType.USDM_FUTURES.value
        ).strip()
        symbol = str(self.symbol or "").strip().upper()
        timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()
        exchange_symbol = str(self.exchange_symbol or symbol).strip().upper()

        if not symbol:
            raise StrategyEvaluationError(
                "OrderflowStrategyScope.symbol cannot be empty"
            )

        object.__setattr__(self, "exchange", exchange)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "exchange_symbol", exchange_symbol)

    @property
    def key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowStrategyScope.key")
        return f"{self.exchange}:{self.market_type}:{self.symbol}:{self.timeframe}"

    @property
    def legacy_key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowStrategyScope.legacy_key")
        return f"{self.symbol}:{self.exchange}"

    def to_dict(self) -> dict[str, str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowStrategyScope.to_dict")
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol or self.symbol,
            "key": self.key,
            "legacy_key": self.legacy_key,
        }


# =============================================================================
# Config
# =============================================================================


@dataclass(slots=True)
class OrderflowStrategyConfig:
    """
    Domain config shared by concrete orderflow strategies.

    Runtime enabled/symbol/timeframe/regime checks belong to StrategyConfig /
    StrategyDefinitionConfig. This config keeps orderflow-specific defaults and
    quality thresholds.
    """
    _logger = logging.getLogger(__name__ + ".OrderflowStrategyConfig")

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

    min_context_confidence: float = 0.0
    min_signal_confidence: float = 0.50
    min_signal_score: float = 0.35

    min_trades_count: int = 0
    min_total_volume: float = 0.0
    min_total_notional: float = 0.0

    stale_feature_max_age_seconds: float | None = None

    requested_leverage: float | None = None
    max_slippage_bps: float | None = None
    entry_timeout_seconds: int | None = None
    max_holding_seconds: int | None = None

    attach_orderflow_context_metadata: bool = True
    attach_scope_metadata: bool = True
    attach_feature_values_metadata: bool = True

    tag_orderflow: str = "orderflow"
    tag_cvd: str = "cvd"
    tag_volume_delta: str = "volume_delta"
    tag_aggressive_flow: str = "aggressive_flow"
    tag_orderbook: str = "orderbook"
    tag_continuation: str = "continuation"
    tag_reversal: str = "reversal"
    tag_divergence: str = "divergence"

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowStrategyConfig.validate")
        bounded = {
            "min_context_confidence": self.min_context_confidence,
            "min_signal_confidence": self.min_signal_confidence,
            "min_signal_score": self.min_signal_score,
        }

        for name, value in bounded.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{name} must be between 0.0 and 1.0")

        if self.min_trades_count < 0:
            raise StrategyConfigError("min_trades_count must be >= 0")

        if self.min_total_volume < 0:
            raise StrategyConfigError("min_total_volume must be >= 0")

        if self.min_total_notional < 0:
            raise StrategyConfigError("min_total_notional must be >= 0")

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
            "tag_orderflow",
            "tag_cvd",
            "tag_volume_delta",
            "tag_aggressive_flow",
            "tag_orderbook",
            "tag_continuation",
            "tag_reversal",
            "tag_divergence",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


# =============================================================================
# Composite snapshot
# =============================================================================


@dataclass(slots=True)
class OrderflowCompositeSnapshot:
    """
    Strategy-level normalized view over analytics.orderflow metrics.

    This is intentionally not an analytics model. It is a strategy-side
    projection that lets concrete orderflow strategies consume CVD,
    volume delta, aggressive trades and orderbook imbalance through one
    stable contract.

    Canonical futures scope:
        exchange + market_type + symbol + timeframe
    """
    _logger = logging.getLogger(__name__ + ".OrderflowCompositeSnapshot")

    exchange: str
    market_type: str
    symbol: str
    timeframe: str

    exchange_symbol: str | None = None
    timestamp: datetime | None = None
    source: str = "unknown"

    last_price: float | None = None
    price_change: float | None = None
    price_change_pct: float = 0.0

    window_seconds: float = 0.0
    trades_count: int = 0
    total_volume: float = 0.0
    total_notional: float = 0.0

    # CVD
    cvd_value: float = 0.0
    cvd_open: float = 0.0
    cvd_high: float = 0.0
    cvd_low: float = 0.0
    cvd_close: float = 0.0
    cvd_change: float = 0.0
    cvd_change_pct: float = 0.0
    cvd_slope: float = 0.0
    cvd_delta_ratio: float = 0.0
    cvd_buy_ratio: float = 0.0
    cvd_sell_ratio: float = 0.0

    # Volume delta
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    volume_delta: float = 0.0
    notional_delta: float = 0.0
    volume_delta_ratio: float = 0.0
    cumulative_volume_delta: float = 0.0
    cumulative_notional_delta: float = 0.0
    volume_buy_ratio: float = 0.0
    volume_sell_ratio: float = 0.0
    avg_trade_size: float = 0.0
    avg_trade_notional: float = 0.0

    # Aggressive trades
    aggressive_buy_count: int = 0
    aggressive_sell_count: int = 0
    aggressive_buy_volume: float = 0.0
    aggressive_sell_volume: float = 0.0
    aggressive_buy_notional: float = 0.0
    aggressive_sell_notional: float = 0.0
    aggressive_net_volume_delta: float = 0.0
    aggressive_net_notional_delta: float = 0.0
    aggressive_buy_ratio: float = 0.0
    aggressive_sell_ratio: float = 0.0
    aggressive_burst_score: float = 0.0
    large_buy_trades: int = 0
    large_sell_trades: int = 0
    aggressive_avg_trade_size: float = 0.0
    aggressive_avg_trade_notional: float = 0.0

    # Orderbook imbalance
    orderbook_bid_volume: float = 0.0
    orderbook_ask_volume: float = 0.0
    orderbook_imbalance_ratio: float = 0.0
    orderbook_imbalance_diff: float = 0.0
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    mid_price: float | None = None
    depth_levels_used: int = 0

    raw_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowCompositeSnapshot.__post_init__")
        exchange = str(self.exchange or "unknown").strip().lower()
        market_type = str(
            self.market_type or StrategyMarketType.USDM_FUTURES.value
        ).strip()
        symbol = str(self.symbol or "").strip().upper()
        timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()
        exchange_symbol = str(self.exchange_symbol or symbol).strip().upper()

        if not symbol:
            raise StrategyEvaluationError(
                "OrderflowCompositeSnapshot.symbol cannot be empty"
            )

        self.exchange = exchange
        self.market_type = market_type
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange_symbol = exchange_symbol

        self.timestamp = parse_datetime(self.timestamp)

        self.last_price = to_float(self.last_price)
        self.price_change = to_float(self.price_change)
        self.best_bid = to_float(self.best_bid)
        self.best_ask = to_float(self.best_ask)
        self.spread = to_float(self.spread)
        self.mid_price = to_float(self.mid_price)

        self.price_change_pct = float(to_float(self.price_change_pct, 0.0) or 0.0)
        self.window_seconds = max(float(to_float(self.window_seconds, 0.0) or 0.0), 0.0)
        self.trades_count = max(int(to_int(self.trades_count, 0) or 0), 0)
        self.total_volume = max(float(to_float(self.total_volume, 0.0) or 0.0), 0.0)
        self.total_notional = max(float(to_float(self.total_notional, 0.0) or 0.0), 0.0)

        for attr in (
            "cvd_value",
            "cvd_open",
            "cvd_high",
            "cvd_low",
            "cvd_close",
            "cvd_change",
            "cvd_change_pct",
            "cvd_slope",
            "cvd_delta_ratio",
            "cvd_buy_ratio",
            "cvd_sell_ratio",
            "buy_volume",
            "sell_volume",
            "buy_notional",
            "sell_notional",
            "volume_delta",
            "notional_delta",
            "volume_delta_ratio",
            "cumulative_volume_delta",
            "cumulative_notional_delta",
            "volume_buy_ratio",
            "volume_sell_ratio",
            "avg_trade_size",
            "avg_trade_notional",
            "aggressive_buy_volume",
            "aggressive_sell_volume",
            "aggressive_buy_notional",
            "aggressive_sell_notional",
            "aggressive_net_volume_delta",
            "aggressive_net_notional_delta",
            "aggressive_buy_ratio",
            "aggressive_sell_ratio",
            "aggressive_burst_score",
            "aggressive_avg_trade_size",
            "aggressive_avg_trade_notional",
            "orderbook_bid_volume",
            "orderbook_ask_volume",
            "orderbook_imbalance_ratio",
            "orderbook_imbalance_diff",
        ):
            setattr(self, attr, float(to_float(getattr(self, attr), 0.0) or 0.0))

        for attr in (
            "aggressive_buy_count",
            "aggressive_sell_count",
            "large_buy_trades",
            "large_sell_trades",
            "depth_levels_used",
        ):
            setattr(self, attr, max(int(to_int(getattr(self, attr), 0) or 0), 0))

        self.orderbook_imbalance_ratio = clamp(
            self.orderbook_imbalance_ratio,
            -1.0,
            1.0,
        )
        self.orderbook_imbalance_diff = clamp(
            self.orderbook_imbalance_diff,
            -1.0,
            1.0,
        )
        self.cvd_delta_ratio = clamp(self.cvd_delta_ratio, -1.0, 1.0)
        self.volume_delta_ratio = clamp(self.volume_delta_ratio, -1.0, 1.0)
        self.aggressive_buy_ratio = clamp(self.aggressive_buy_ratio, 0.0, 1.0)
        self.aggressive_sell_ratio = clamp(self.aggressive_sell_ratio, 0.0, 1.0)
        self.aggressive_burst_score = clamp(self.aggressive_burst_score, 0.0, 1.0)

    @property
    def scope(self) -> OrderflowStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowCompositeSnapshot.scope")
        return OrderflowStrategyScope(
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
            _strategy_logger.debug("Entering OrderflowCompositeSnapshot.scope_key")
        return self.scope.key

    @property
    def aggressive_large_trade_count(self) -> int:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowCompositeSnapshot.aggressive_large_trade_count")
        return self.large_buy_trades + self.large_sell_trades

    @property
    def signed_orderbook_imbalance(self) -> float:
        """
        Signed orderbook value for strategy logic.

        Prefer this over raw orderbook_imbalance_ratio because analytics can
        configure ratio as either 0..1 or -1..1.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowCompositeSnapshot.signed_orderbook_imbalance")
        return self.orderbook_imbalance_diff

    @property
    def has_orderbook(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowCompositeSnapshot.has_orderbook")
        return (
            self.orderbook_bid_volume > 0.0
            or self.orderbook_ask_volume > 0.0
            or self.depth_levels_used > 0
            or self.best_bid is not None
            or self.best_ask is not None
        )

    @property
    def has_aggressive_flow(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowCompositeSnapshot.has_aggressive_flow")
        return (
            self.aggressive_buy_count > 0
            or self.aggressive_sell_count > 0
            or self.aggressive_buy_volume > 0.0
            or self.aggressive_sell_volume > 0.0
            or self.aggressive_net_volume_delta != 0.0
            or self.aggressive_net_notional_delta != 0.0
        )

    def has_minimum_data(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowCompositeSnapshot.has_minimum_data")
        return (
            self.trades_count > 0
            or self.total_volume > 0.0
            or self.total_notional > 0.0
            or self.has_orderbook
            or self.has_aggressive_flow
        )

    def directional_large_trades(self, side: SignalSide | str) -> int:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowCompositeSnapshot.directional_large_trades")
        normalized = (
            side.value if isinstance(side, SignalSide) else str(side)
        ).strip().lower()

        if normalized in {"long", "buy", "bullish"}:
            return self.large_buy_trades

        if normalized in {"short", "sell", "bearish"}:
            return self.large_sell_trades

        return 0

    def directional_aggressive_ratio(self, side: SignalSide | str) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowCompositeSnapshot.directional_aggressive_ratio")
        normalized = (
            side.value if isinstance(side, SignalSide) else str(side)
        ).strip().lower()

        if normalized in {"long", "buy", "bullish"}:
            return self.aggressive_buy_ratio

        if normalized in {"short", "sell", "bearish"}:
            return self.aggressive_sell_ratio

        return 0.0

    def directional_aggressive_notional_delta(
        self,
        side: SignalSide | str,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowCompositeSnapshot.directional_aggressive_notional_delta")
        normalized = (
            side.value if isinstance(side, SignalSide) else str(side)
        ).strip().lower()

        if normalized in {"long", "buy", "bullish"}:
            return self.aggressive_net_notional_delta

        if normalized in {"short", "sell", "bearish"}:
            return -self.aggressive_net_notional_delta

        return 0.0

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowCompositeSnapshot.to_dict")
        payload = asdict(self)
        payload["timestamp"] = (
            self.timestamp.isoformat() if self.timestamp is not None else None
        )
        payload["scope"] = self.scope.to_dict()
        payload["scope_key"] = self.scope_key
        payload["aggressive_large_trade_count"] = self.aggressive_large_trade_count
        payload["signed_orderbook_imbalance"] = self.signed_orderbook_imbalance
        payload["has_orderbook"] = self.has_orderbook
        payload["has_aggressive_flow"] = self.has_aggressive_flow
        payload["has_minimum_data"] = self.has_minimum_data()
        return payload


# =============================================================================
# Base orderflow strategy
# =============================================================================


class OrderflowTradingStrategy(TradingStrategy):
    """
    Base class for concrete strategy/strategies/orderflow/* classes.

    Responsibilities:
    - read orderflow analytics data from StrategyContext only;
    - provide helper methods for orderflow domain extraction and scoring;
    - build internal StrategySignal objects through TradingStrategy helpers;
    - attach futures/orderflow metadata for SignalProcessor.

    Forbidden:
    - no direct analytics.orderflow.* EventBus subscriptions;
    - no direct OrderFlowAnalyzer / analytics facade reads;
    - no local signal/rejection state machine;
    - no diagnostics scheduler jobs;
    - no EventBus emit of signal.generated;
    - no RiskManager / Execution calls;
    - no raw market data reads.
    """
    _logger = logging.getLogger(__name__ + ".OrderflowTradingStrategy")

    component_namespace = "strategy.orderflow"
    category: StrategyCategory = StrategyCategory.ORDERFLOW
    default_setup_type: SetupType = SetupType.UNKNOWN
    default_timeframe: Timeframe = Timeframe.M1

    feature_names = ORDERFLOW_FEATURES

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        orderflow_config: OrderflowStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.__init__")
        self.orderflow_config = orderflow_config or OrderflowStrategyConfig()
        self.orderflow_config.validate()

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
            _strategy_logger.debug("Entering OrderflowTradingStrategy.validate_config")
        super().validate_config()
        self.orderflow_config.validate()

    # ------------------------------------------------------------------
    # Context / domain access
    # ------------------------------------------------------------------

    def orderflow_domain(self, context: StrategyContext) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_domain")
        self.validate_context(context)
        return orderflow_domain(context)

    def orderflow_item(
        self,
        context: StrategyContext,
        key: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_item")
        self.validate_context(context)
        return orderflow_item(context, key, default)

    def orderflow_path(
        self,
        context: StrategyContext,
        path: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_path")
        self.validate_context(context)

        if not isinstance(path, str) or not path.strip():
            raise StrategyEvaluationError("orderflow path cannot be empty")

        return orderflow_path(context, path, default)

    def orderflow_float(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float | None = None,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_float")
        return to_float(self.orderflow_path(context, path, default), default)

    def orderflow_int(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: int | None = None,
    ) -> int | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_int")
        return to_int(self.orderflow_path(context, path, default), default)

    def orderflow_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_score")
        return unit_score(self.orderflow_path(context, path, default), default)

    def orderflow_signed_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_signed_score")
        value = self.orderflow_float(context, path, default=default)
        return clamp(float(value if value is not None else default), -1.0, 1.0)

    def orderflow_bool(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: bool = False,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_bool")
        return to_bool(self.orderflow_path(context, path, default), default)

    def orderflow_str(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: str | None = None,
    ) -> str | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_str")
        return to_str(self.orderflow_path(context, path, default), default)

    def orderflow_datetime(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: datetime | None = None,
    ) -> datetime | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_datetime")
        return parse_datetime(self.orderflow_path(context, path, default))

    def orderflow_feature_snapshot(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> FeatureSnapshot | None:
        """
        Return full FeatureSnapshot if StrategyContext stores one.

        Best-effort helper because StrategyContext may store raw values or
        FeatureSnapshot objects depending on normalization.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_feature_snapshot")
        self.validate_context(context)

        if not isinstance(feature_name, str) or not feature_name.strip():
            raise StrategyEvaluationError("feature_name cannot be empty")

        features_map = getattr(context, "features", None)
        if isinstance(features_map, Mapping):
            raw = features_map.get(feature_name)
            if isinstance(raw, FeatureSnapshot):
                return raw

        return None

    def orderflow_feature_age_seconds(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_feature_age_seconds")
        snapshot = self.orderflow_feature_snapshot(context, feature_name)
        if snapshot is None:
            return None
        return snapshot.age_seconds(context.timestamp)

    def orderflow_feature_is_stale(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_feature_is_stale")
        max_age = self.orderflow_config.stale_feature_max_age_seconds
        if max_age is None:
            return False

        age = self.orderflow_feature_age_seconds(context, feature_name)
        if age is None:
            return False

        return age > max_age

    def has_any_orderflow_data(
        self,
        context: StrategyContext,
        feature_names: tuple[str, ...] = (),
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.has_any_orderflow_data")
        self.validate_context(context)

        if self.orderflow_domain(context):
            return True

        return any(context.has_feature(name) for name in feature_names)

    def has_stale_orderflow_features(
        self,
        context: StrategyContext,
        feature_names: tuple[str, ...] | None = None,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.has_stale_orderflow_features")
        names = feature_names or tuple(self.required_features())

        return any(
            self.orderflow_feature_is_stale(context, feature_name)
            for feature_name in names
        )

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------

    def orderflow_scope(self, context: StrategyContext) -> OrderflowStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_scope")
        domain = self.orderflow_domain(context)

        exchange = (
            to_str(domain.get("exchange"))
            or to_str(get_path(domain, "scope.exchange"))
            or to_str(context.metadata.get("exchange"))
            or "unknown"
        )
        market_type = (
            to_str(domain.get("market_type"))
            or to_str(get_path(domain, "scope.market_type"))
            or to_str(context.metadata.get("market_type"))
            or self.orderflow_config.default_market_type.value
        )
        exchange_symbol = (
            to_str(domain.get("exchange_symbol"))
            or to_str(get_path(domain, "scope.exchange_symbol"))
            or to_str(context.metadata.get("exchange_symbol"))
            or context.symbol
        )

        timeframe_value = getattr(context.timeframe, "value", context.timeframe)

        return OrderflowStrategyScope(
            exchange=exchange,
            market_type=market_type,
            symbol=context.symbol,
            timeframe=str(timeframe_value or Timeframe.M1.value),
            exchange_symbol=exchange_symbol,
        )

    # ------------------------------------------------------------------
    # Snapshot resolution
    # ------------------------------------------------------------------

    def resolve_orderflow_snapshot(
        self,
        context: StrategyContext,
    ) -> OrderflowCompositeSnapshot | None:
        """
        Resolve normalized composite snapshot from StrategyContext only.

        Preferred:
            FeatureSource.ORDERFLOW domain["composite"] / domain["snapshot"]

        Fallback:
            domain metrics: cvd, volume_delta, aggressive_trades,
            orderbook_imbalance.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.resolve_orderflow_snapshot")
        self.validate_context(context)

        scope = self.orderflow_scope(context)

        composite = self.orderflow_item(context, "composite")
        if composite is not None:
            snapshot = self._coerce_composite_snapshot(
                composite,
                context=context,
                scope=scope,
                source="context.composite",
            )
            if snapshot is not None and snapshot.has_minimum_data():
                return snapshot

        domain_snapshot = self._build_composite_snapshot_from_domain(
            context=context,
            scope=scope,
        )
        if domain_snapshot is not None and domain_snapshot.has_minimum_data():
            return domain_snapshot

        feature_snapshot = self._build_composite_snapshot_from_features(
            context=context,
            scope=scope,
        )
        if feature_snapshot is not None and feature_snapshot.has_minimum_data():
            return feature_snapshot

        return domain_snapshot or feature_snapshot

    def resolve_cvd_payload(self, context: StrategyContext) -> Any | None:
        """
        Resolve CVD payload from StrategyContext only.

        CVD divergence strategy can use this directly, but most orderflow
        strategies should prefer resolve_orderflow_snapshot().
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.resolve_cvd_payload")
        self.validate_context(context)

        cvd = self.orderflow_item(context, "cvd")
        if cvd is not None:
            return cvd

        snapshot = self.resolve_orderflow_snapshot(context)
        if snapshot is not None:
            return snapshot

        return None

    def _coerce_composite_snapshot(
        self,
        value: Any,
        *,
        context: StrategyContext,
        scope: OrderflowStrategyScope,
        source: str,
    ) -> OrderflowCompositeSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy._coerce_composite_snapshot")
        if isinstance(value, OrderflowCompositeSnapshot):
            return value

        data = as_dict(value)
        if not data:
            return None

        return self._snapshot_from_mapping(
            data,
            context=context,
            scope=scope,
            source=source,
        )

    def _build_composite_snapshot_from_domain(
        self,
        *,
        context: StrategyContext,
        scope: OrderflowStrategyScope,
    ) -> OrderflowCompositeSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy._build_composite_snapshot_from_domain")
        domain = self.orderflow_domain(context)

        cvd = self.orderflow_item(context, "cvd")
        volume_delta = self.orderflow_item(context, "volume_delta")
        aggressive = self.orderflow_item(context, "aggressive_trades")
        imbalance = self.orderflow_item(context, "orderbook_imbalance")

        if not any((cvd, volume_delta, aggressive, imbalance, domain)):
            return None

        merged = {
            **as_dict(domain),
            "cvd": as_dict(cvd),
            "volume_delta": as_dict(volume_delta),
            "aggressive_trades": as_dict(aggressive),
            "orderbook_imbalance": as_dict(imbalance),
        }

        return self._snapshot_from_mapping(
            merged,
            context=context,
            scope=scope,
            source="context.domain",
        )

    def _build_composite_snapshot_from_features(
        self,
        *,
        context: StrategyContext,
        scope: OrderflowStrategyScope,
    ) -> OrderflowCompositeSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy._build_composite_snapshot_from_features")
        feature_payload = {
            "price_change_pct": self._feature_value(
                context,
                ORDERFLOW_FEATURES.PRICE_CHANGE_PCT,
            ),
            "last_price": self._feature_value(
                context,
                ORDERFLOW_FEATURES.LAST_PRICE,
            ),
            "trades_count": self._feature_value(
                context,
                ORDERFLOW_FEATURES.TRADES_COUNT,
            ),
            "total_volume": self._feature_value(
                context,
                ORDERFLOW_FEATURES.TOTAL_VOLUME,
            ),
            "total_notional": self._feature_value(
                context,
                ORDERFLOW_FEATURES.TOTAL_NOTIONAL,
            ),
            "cvd": {
                "delta_ratio": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.CVD_DELTA_RATIO,
                ),
                "cvd_change_pct": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.CVD_CHANGE_PCT,
                ),
                "cvd_slope": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.CVD_SLOPE,
                ),
                "price_change_pct": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.CVD_PRICE_CHANGE_PCT,
                ),
                "value": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.CVD_VALUE,
                ),
            },
            "volume_delta": {
                "delta_ratio": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.VOLUME_DELTA_RATIO,
                ),
                "volume_delta": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.VOLUME_DELTA_VALUE,
                ),
                "cumulative_volume_delta": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.CUMULATIVE_VOLUME_DELTA,
                ),
                "notional_delta": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.NOTIONAL_DELTA,
                ),
                "cumulative_notional_delta": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.CUMULATIVE_NOTIONAL_DELTA,
                ),
            },
            "aggressive_trades": {
                "buy_ratio": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.AGGRESSIVE_BUY_RATIO,
                ),
                "sell_ratio": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.AGGRESSIVE_SELL_RATIO,
                ),
                "burst_score": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.AGGRESSIVE_BURST_SCORE,
                ),
                "net_volume_delta": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.AGGRESSIVE_NET_VOLUME_DELTA,
                ),
                "net_notional_delta": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.AGGRESSIVE_NET_NOTIONAL_DELTA,
                ),
                "large_buy_trades": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.LARGE_BUY_TRADES,
                ),
                "large_sell_trades": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.LARGE_SELL_TRADES,
                ),
            },
            "orderbook_imbalance": {
                "ratio": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.ORDERBOOK_IMBALANCE_RATIO,
                ),
                "diff": self._feature_value(
                    context,
                    ORDERFLOW_FEATURES.ORDERBOOK_IMBALANCE_DIFF,
                ),
            },
        }

        if not self._has_any_value(feature_payload):
            return None

        return self._snapshot_from_mapping(
            feature_payload,
            context=context,
            scope=scope,
            source="context.features",
        )

    def _snapshot_from_mapping(
        self,
        data: Mapping[str, Any],
        *,
        context: StrategyContext,
        scope: OrderflowStrategyScope,
        source: str,
    ) -> OrderflowCompositeSnapshot:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy._snapshot_from_mapping")
        cvd = get_path(data, "cvd", {}) or {}
        volume_delta = get_path(data, "volume_delta", {}) or {}
        aggressive = get_path(data, "aggressive_trades", {}) or {}
        imbalance = get_path(data, "orderbook_imbalance", {}) or {}

        timestamp = (
            extract_event_time(data)
            or extract_event_time(cvd)
            or extract_event_time(volume_delta)
            or extract_event_time(aggressive)
            or extract_event_time(imbalance)
            or context.timestamp
        )

        orderbook_ratio = extract_orderbook_imbalance_ratio(data)
        orderbook_diff = extract_orderbook_imbalance_diff(data)

        if orderbook_diff == 0.0 and orderbook_ratio != 0.0:
            orderbook_diff = signed_imbalance_from_ratio(orderbook_ratio)

        buy_volume = extract_buy_volume(data)
        sell_volume = extract_sell_volume(data)
        total_volume = extract_total_volume(data) or max(buy_volume + sell_volume, 0.0)

        buy_notional = to_float(
            get_path(data, "buy_notional", None)
            or get_path(volume_delta, "buy_notional", None)
            or get_path(cvd, "buy_notional", None),
            0.0,
        ) or 0.0
        sell_notional = to_float(
            get_path(data, "sell_notional", None)
            or get_path(volume_delta, "sell_notional", None)
            or get_path(cvd, "sell_notional", None),
            0.0,
        ) or 0.0

        return OrderflowCompositeSnapshot(
            exchange=scope.exchange,
            market_type=scope.market_type,
            symbol=scope.symbol,
            timeframe=scope.timeframe,
            exchange_symbol=scope.exchange_symbol,
            timestamp=timestamp,
            source=source,
            last_price=extract_last_price(data),
            price_change=to_float(
                get_path(data, "price_change", None)
                or get_path(cvd, "price_change", None),
            ),
            price_change_pct=extract_price_change_pct(data),
            window_seconds=to_float(
                get_path(data, "window_seconds", None)
                or get_path(cvd, "window_seconds", None)
                or get_path(volume_delta, "window_seconds", None)
                or get_path(aggressive, "window_seconds", None),
                0.0,
            ) or 0.0,
            trades_count=extract_trades_count(data),
            total_volume=total_volume,
            total_notional=extract_total_notional(data),
            cvd_value=extract_cvd_value(data),
            cvd_open=to_float(get_path(cvd, "cvd_open", None) or get_path(cvd, "open", None), 0.0) or 0.0,
            cvd_high=to_float(get_path(cvd, "cvd_high", None) or get_path(cvd, "high", None), 0.0) or 0.0,
            cvd_low=to_float(get_path(cvd, "cvd_low", None) or get_path(cvd, "low", None), 0.0) or 0.0,
            cvd_close=to_float(get_path(cvd, "cvd_close", None) or get_path(cvd, "close", None), 0.0) or 0.0,
            cvd_change=to_float(
                get_path(data, "cvd_change", None)
                or get_path(cvd, "cvd_change", None)
                or get_path(cvd, "change", None),
                0.0,
            ) or 0.0,
            cvd_change_pct=extract_cvd_change_pct(data),
            cvd_slope=extract_cvd_slope(data),
            cvd_delta_ratio=extract_cvd_delta_ratio(data),
            cvd_buy_ratio=unit_score(get_path(cvd, "buy_ratio", 0.0)),
            cvd_sell_ratio=unit_score(get_path(cvd, "sell_ratio", 0.0)),
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            buy_notional=buy_notional,
            sell_notional=sell_notional,
            volume_delta=extract_volume_delta(data),
            notional_delta=extract_notional_delta(data),
            volume_delta_ratio=extract_volume_delta_ratio(data),
            cumulative_volume_delta=extract_cumulative_volume_delta(data),
            cumulative_notional_delta=extract_cumulative_notional_delta(data),
            volume_buy_ratio=unit_score(
                get_path(data, "volume_buy_ratio", None)
                or get_path(volume_delta, "buy_ratio", None),
                0.0,
            ),
            volume_sell_ratio=unit_score(
                get_path(data, "volume_sell_ratio", None)
                or get_path(volume_delta, "sell_ratio", None),
                0.0,
            ),
            avg_trade_size=to_float(
                get_path(data, "avg_trade_size", None)
                or get_path(volume_delta, "avg_trade_size", None)
                or get_path(cvd, "avg_trade_size", None),
                0.0,
            ) or 0.0,
            avg_trade_notional=to_float(
                get_path(data, "avg_trade_notional", None)
                or get_path(volume_delta, "avg_trade_notional", None)
                or get_path(cvd, "avg_trade_notional", None),
                0.0,
            ) or 0.0,
            aggressive_buy_count=to_int(
                get_path(aggressive, "aggressive_buy_count", None)
                or get_path(aggressive, "buy_count", None),
                0,
            ) or 0,
            aggressive_sell_count=to_int(
                get_path(aggressive, "aggressive_sell_count", None)
                or get_path(aggressive, "sell_count", None),
                0,
            ) or 0,
            aggressive_buy_volume=to_float(
                get_path(aggressive, "aggressive_buy_volume", None)
                or get_path(aggressive, "buy_volume", None),
                0.0,
            ) or 0.0,
            aggressive_sell_volume=to_float(
                get_path(aggressive, "aggressive_sell_volume", None)
                or get_path(aggressive, "sell_volume", None),
                0.0,
            ) or 0.0,
            aggressive_buy_notional=to_float(
                get_path(aggressive, "aggressive_buy_notional", None)
                or get_path(aggressive, "buy_notional", None),
                0.0,
            ) or 0.0,
            aggressive_sell_notional=to_float(
                get_path(aggressive, "aggressive_sell_notional", None)
                or get_path(aggressive, "sell_notional", None),
                0.0,
            ) or 0.0,
            aggressive_net_volume_delta=extract_aggressive_net_volume_delta(data),
            aggressive_net_notional_delta=extract_aggressive_net_notional_delta(data),
            aggressive_buy_ratio=extract_aggressive_buy_ratio(data),
            aggressive_sell_ratio=extract_aggressive_sell_ratio(data),
            aggressive_burst_score=extract_aggressive_burst_score(data),
            large_buy_trades=extract_large_buy_trades(data),
            large_sell_trades=extract_large_sell_trades(data),
            aggressive_avg_trade_size=to_float(
                get_path(aggressive, "avg_trade_size", None)
                or get_path(aggressive, "aggressive_avg_trade_size", None),
                0.0,
            ) or 0.0,
            aggressive_avg_trade_notional=to_float(
                get_path(aggressive, "avg_trade_notional", None)
                or get_path(aggressive, "aggressive_avg_trade_notional", None),
                0.0,
            ) or 0.0,
            orderbook_bid_volume=to_float(
                get_path(imbalance, "bid_volume", None)
                or get_path(imbalance, "orderbook_bid_volume", None),
                0.0,
            ) or 0.0,
            orderbook_ask_volume=to_float(
                get_path(imbalance, "ask_volume", None)
                or get_path(imbalance, "orderbook_ask_volume", None),
                0.0,
            ) or 0.0,
            orderbook_imbalance_ratio=orderbook_ratio,
            orderbook_imbalance_diff=orderbook_diff,
            best_bid=to_float(get_path(imbalance, "best_bid", None)),
            best_ask=to_float(get_path(imbalance, "best_ask", None)),
            spread=to_float(get_path(imbalance, "spread", None)),
            mid_price=to_float(get_path(imbalance, "mid_price", None)),
            depth_levels_used=to_int(
                get_path(imbalance, "depth_levels_used", None)
                or get_path(imbalance, "levels_used", None),
                0,
            ) or 0,
            raw_metrics={
                "domain": serialize_for_metadata(data),
                "cvd": serialize_for_metadata(cvd),
                "volume_delta": serialize_for_metadata(volume_delta),
                "aggressive_trades": serialize_for_metadata(aggressive),
                "orderbook_imbalance": serialize_for_metadata(imbalance),
            },
            metadata={
                "source": source,
                "scope": scope.to_dict(),
            },
        )

    # ------------------------------------------------------------------
    # Signal builder
    # ------------------------------------------------------------------

    def build_orderflow_signal(
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
        Build internal StrategySignal with orderflow/futures metadata.

        Final risk-ready payload conversion belongs to SignalProcessor /
        SignalBuilder, not to this domain strategy.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.build_orderflow_signal")
        if side not in {SignalSide.LONG, SignalSide.SHORT}:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: orderflow signal side must be LONG or SHORT"
            )

        scope = self.orderflow_scope(context)

        signal_metadata = dict(metadata or {})
        signal_metadata.setdefault("domain", FeatureSource.ORDERFLOW.value)
        signal_metadata.setdefault("orderflow_strategy_version", "2.0.0")
        signal_metadata.setdefault(
            "order_intent",
            self.orderflow_config.default_order_intent.value,
        )
        signal_metadata.setdefault(
            "margin_mode",
            self.orderflow_config.default_margin_mode.value,
        )
        signal_metadata.setdefault(
            "market_type",
            self.orderflow_config.default_market_type.value,
        )
        signal_metadata.setdefault(
            "tier",
            self.orderflow_config.default_trade_tier.value,
        )

        if self.orderflow_config.requested_leverage is not None:
            signal_metadata.setdefault(
                "requested_leverage",
                float(self.orderflow_config.requested_leverage),
            )

        if self.orderflow_config.max_slippage_bps is not None:
            signal_metadata.setdefault(
                "max_slippage_bps",
                float(self.orderflow_config.max_slippage_bps),
            )

        if self.orderflow_config.entry_timeout_seconds is not None:
            signal_metadata.setdefault(
                "entry_timeout_seconds",
                int(self.orderflow_config.entry_timeout_seconds),
            )

        if self.orderflow_config.max_holding_seconds is not None:
            signal_metadata.setdefault(
                "max_holding_seconds",
                int(self.orderflow_config.max_holding_seconds),
            )

        if self.orderflow_config.attach_scope_metadata:
            signal_metadata.setdefault("scope", scope.to_dict())

        if self.orderflow_config.attach_orderflow_context_metadata:
            signal_metadata.setdefault(
                "orderflow_context",
                self.orderflow_context_metadata(context),
            )

        if self.orderflow_config.metadata:
            signal_metadata.setdefault(
                "orderflow_config_metadata",
                serialize_for_metadata(self.orderflow_config.metadata),
            )

        final_reasons = list(
            dict.fromkeys(
                [
                    "orderflow_strategy_signal",
                    *(reasons or []),
                ]
            )
        )
        final_confirmations = list(dict.fromkeys(confirmations or []))
        final_features = list(dict.fromkeys(source_features or []))

        signal = self.build_signal(
            context=context,
            side=side,
            confidence=confidence,
            score=score,
            setup_type=setup_type or self.default_setup_type,
            reasons=final_reasons,
            confirmations=final_confirmations,
            source_features=final_features,
            metadata=signal_metadata,
            trigger_type=trigger_type,
            origin=origin,
            priority=priority,
            status=status,
        )

        signal.validate()
        return signal

    def orderflow_context_metadata(
        self,
        context: StrategyContext,
    ) -> dict[str, Any]:
        """
        Compact serialized orderflow context for StrategySignal.metadata.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy.orderflow_context_metadata")
        metadata: dict[str, Any] = {}

        snapshot = self.resolve_orderflow_snapshot(context)
        if snapshot is not None:
            metadata["snapshot"] = snapshot.to_dict()

        if self.orderflow_config.attach_feature_values_metadata:
            metadata["feature_values"] = {
                "price_change_pct": self.orderflow_path(
                    context,
                    "price_change_pct",
                    None,
                ),
                "cvd_delta_ratio": self.orderflow_path(
                    context,
                    "cvd.delta_ratio",
                    None,
                ),
                "cvd_change_pct": self.orderflow_path(
                    context,
                    "cvd.cvd_change_pct",
                    None,
                ),
                "cvd_slope": self.orderflow_path(
                    context,
                    "cvd.cvd_slope",
                    None,
                ),
                "volume_delta_ratio": self.orderflow_path(
                    context,
                    "volume_delta.delta_ratio",
                    None,
                ),
                "aggressive_buy_ratio": self.orderflow_path(
                    context,
                    "aggressive_trades.buy_ratio",
                    None,
                ),
                "aggressive_sell_ratio": self.orderflow_path(
                    context,
                    "aggressive_trades.sell_ratio",
                    None,
                ),
                "orderbook_imbalance_diff": self.orderflow_path(
                    context,
                    "orderbook_imbalance.diff",
                    None,
                ),
            }

        return serialize_for_metadata(metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_value(context: StrategyContext, feature_name: str) -> Any:
        _strategy_logger = logging.getLogger(__name__ + ".OrderflowTradingStrategy._feature_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy._feature_value")
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
        _strategy_logger = logging.getLogger(__name__ + ".OrderflowTradingStrategy._has_any_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowTradingStrategy._has_any_value")
        if value is None:
            return False

        if isinstance(value, Mapping):
            return any(OrderflowTradingStrategy._has_any_value(item) for item in value.values())

        if isinstance(value, (list, tuple, set)):
            return any(OrderflowTradingStrategy._has_any_value(item) for item in value)

        return value is not None


# Backward-compatible alias while concrete orderflow strategies are migrated.
OrderflowStrategyBase = OrderflowTradingStrategy