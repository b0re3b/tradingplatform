from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, TypeAlias

from .enums import (
    OrderFlowMetricType,
    OrderFlowSide,
    OrderFlowSignalType,
    OrderFlowSourceType,
)


DEFAULT_MARKET_TYPE = "perpetual"
DEFAULT_TIMEFRAME = "1m"

OrderFlowKey: TypeAlias = tuple[str, str, str, str]
# exchange, market_type, symbol, timeframe


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _safe_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int | None = None) -> int | None:
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: object, low: float = 0.0, high: float = 1.0) -> float:
    parsed = _safe_float(value, default=0.0)
    assert parsed is not None
    return max(low, min(high, parsed))


def _normalize_symbol(symbol: object) -> str:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized


def _normalize_exchange(exchange: object) -> str:
    normalized = str(exchange or "").strip().lower()
    if not normalized:
        raise ValueError("exchange must not be empty")
    return normalized


def _normalize_market_type(market_type: object) -> str:
    normalized = str(market_type or DEFAULT_MARKET_TYPE).strip().lower()
    return normalized if normalized else DEFAULT_MARKET_TYPE


def _normalize_timeframe(timeframe: object) -> str:
    normalized = str(timeframe or DEFAULT_TIMEFRAME).strip()
    return normalized if normalized else DEFAULT_TIMEFRAME


def _normalize_exchange_symbol(
    exchange_symbol: object,
    *,
    fallback_symbol: str,
) -> str:
    normalized = str(exchange_symbol or "").strip()
    return normalized if normalized else fallback_symbol


def make_orderflow_key(
    *,
    exchange: object,
    market_type: object = DEFAULT_MARKET_TYPE,
    symbol: object,
    timeframe: object = DEFAULT_TIMEFRAME,
) -> OrderFlowKey:
    return (
        _normalize_exchange(exchange),
        _normalize_market_type(market_type),
        _normalize_symbol(symbol),
        _normalize_timeframe(timeframe),
    )


def orderflow_key_to_dict(key: OrderFlowKey) -> dict[str, str]:
    exchange, market_type, symbol, timeframe = key
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def _coerce_metric_type(value: OrderFlowMetricType | str) -> OrderFlowMetricType:
    if isinstance(value, OrderFlowMetricType):
        return value
    return OrderFlowMetricType(str(value))


def _coerce_source_type(value: OrderFlowSourceType | str) -> OrderFlowSourceType:
    if isinstance(value, OrderFlowSourceType):
        return value
    return OrderFlowSourceType(str(value))


def _coerce_signal_type(value: OrderFlowSignalType | str) -> OrderFlowSignalType:
    if isinstance(value, OrderFlowSignalType):
        return value
    return OrderFlowSignalType(str(value))


def _coerce_side(value: OrderFlowSide | str | None) -> OrderFlowSide:
    return OrderFlowSide.from_value(value)


def _serialize_value(value: Any) -> Any:
    """
    Convert dataclasses and enums into EventBus-safe plain Python values.

    EventBus payloads should be JSON-friendly dictionaries. This helper keeps
    order-flow models independent from EventBus while still making their output
    safe for analytics.* events.
    """
    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value):
        return {
            key: _serialize_value(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]

    if isinstance(value, set):
        return sorted(_serialize_value(item) for item in value)

    return value


# ---------------------------------------------------------------------
# Base input models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class NormalizedTrade:
    """
    Canonical futures trade model consumed by analytics.orderflow modules.

    Source:
        TradesCache -> market.trades.updated -> analytics.orderflow

    Scope:
        exchange + market_type + symbol + timeframe

    The exchange adapter/data layer should normalize raw exchange payloads
    before order-flow analyzers calculate metrics. This model should not know
    about raw Binance/Bybit/OKX/MEXC field names.
    """

    symbol: str
    side: OrderFlowSide
    price: float
    quantity: float
    notional: float
    timestamp: float

    exchange: str
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    trade_id: str | None = None
    is_aggressive: bool = False
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.symbol = _normalize_symbol(self.symbol)
        self.exchange = _normalize_exchange(self.exchange)
        self.market_type = _normalize_market_type(self.market_type)
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.side = _coerce_side(self.side)
        self.price = float(self.price)
        self.quantity = float(self.quantity)
        self.notional = float(self.notional)
        self.timestamp = float(self.timestamp)

        self.trade_id = str(self.trade_id) if self.trade_id is not None else None
        self.is_aggressive = bool(self.is_aggressive)
        self.raw = dict(self.raw or {})

        if self.price <= 0:
            raise ValueError("NormalizedTrade.price must be > 0")
        if self.quantity <= 0:
            raise ValueError("NormalizedTrade.quantity must be > 0")
        if self.notional < 0:
            raise ValueError("NormalizedTrade.notional must be >= 0")
        if self.timestamp <= 0:
            raise ValueError("NormalizedTrade.timestamp must be > 0")

    @property
    def key(self) -> OrderFlowKey:
        return make_orderflow_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def market_key(self) -> tuple[str, str, str]:
        return self.exchange, self.market_type, self.symbol

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        side: str | OrderFlowSide,
        price: float,
        quantity: float,
        timestamp: float,
        exchange: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
        trade_id: str | None = None,
        is_aggressive: bool = False,
        raw: dict[str, Any] | None = None,
        notional: float | None = None,
    ) -> NormalizedTrade:
        side_enum = OrderFlowSide.from_value(side)
        price_f = float(price)
        quantity_f = float(quantity)
        notional_f = float(notional) if notional is not None else price_f * quantity_f

        return cls(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            exchange_symbol=exchange_symbol,
            timeframe=timeframe,
            side=side_enum,
            price=price_f,
            quantity=quantity_f,
            notional=notional_f,
            timestamp=float(timestamp),
            trade_id=str(trade_id) if trade_id is not None else None,
            is_aggressive=bool(is_aggressive),
            raw=raw,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedTrade:
        return cls.create(
            exchange=str(data["exchange"]),
            market_type=str(data.get("market_type") or DEFAULT_MARKET_TYPE),
            symbol=str(data["symbol"]),
            exchange_symbol=data.get("exchange_symbol"),
            timeframe=str(data.get("timeframe") or DEFAULT_TIMEFRAME),
            side=data.get("side"),
            price=float(data["price"]),
            quantity=float(data.get("quantity", data.get("qty", data.get("size")))),
            notional=data.get("notional") or data.get("quote_qty"),
            timestamp=float(data.get("timestamp", data.get("timestamp_ms", time.time()))),
            trade_id=data.get("trade_id"),
            is_aggressive=bool(data.get("is_aggressive", False)),
            raw=dict(data.get("raw") or {}),
        )

    @property
    def signed_volume(self) -> float:
        if self.side == OrderFlowSide.BUY:
            return self.quantity
        if self.side == OrderFlowSide.SELL:
            return -self.quantity
        return 0.0

    @property
    def signed_notional(self) -> float:
        if self.side == OrderFlowSide.BUY:
            return self.notional
        if self.side == OrderFlowSide.SELL:
            return -self.notional
        return 0.0

    @property
    def is_valid(self) -> bool:
        return (
            bool(self.symbol)
            and bool(self.exchange)
            and bool(self.market_type)
            and self.side.is_known
            and self.price > 0
            and self.quantity > 0
            and self.timestamp > 0
        )

    def to_dict(self) -> dict[str, Any]:
        payload = model_to_dict(self)
        payload["key"] = list(self.key)
        return payload


@dataclass(slots=True)
class OrderbookLevel:
    """
    Single normalized orderbook level.
    """

    price: float
    size: float

    def __post_init__(self) -> None:
        self.price = float(self.price)
        self.size = float(self.size)

    @classmethod
    def from_raw(cls, raw: Any) -> OrderbookLevel | None:
        if raw is None:
            return None

        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            price = _safe_float(raw[0])
            size = _safe_float(raw[1])

            if price is None or size is None:
                return None

            return cls(price=price, size=size)

        if isinstance(raw, dict):
            price = _safe_float(raw.get("price", raw.get("p")))
            size = _safe_float(
                raw.get(
                    "size",
                    raw.get("quantity", raw.get("qty", raw.get("q"))),
                )
            )

            if price is None or size is None:
                return None

            return cls(price=price, size=size)

        return None

    @property
    def notional(self) -> float:
        return self.price * self.size

    @property
    def is_valid(self) -> bool:
        return self.price > 0 and self.size > 0

    def to_dict(self) -> dict[str, Any]:
        return model_to_dict(self)


@dataclass(slots=True)
class OrderbookSnapshot:
    """
    Canonical futures orderbook snapshot consumed by orderbook-based analyzers.

    Source:
        OrderbookCache -> market.orderbook.updated -> analytics.orderflow

    Scope:
        exchange + market_type + symbol + timeframe

    Orderbook itself is not naturally timeframe-based, but analytics output
    still carries timeframe/window scope for downstream consumers.
    """

    symbol: str
    bids: list[OrderbookLevel]
    asks: list[OrderbookLevel]
    timestamp: float

    exchange: str
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    sequence_id: str | None = None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.symbol = _normalize_symbol(self.symbol)
        self.exchange = _normalize_exchange(self.exchange)
        self.market_type = _normalize_market_type(self.market_type)
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.timestamp = float(self.timestamp)
        self.sequence_id = str(self.sequence_id) if self.sequence_id is not None else None
        self.raw = dict(self.raw or {})

        self.bids = [
            level if isinstance(level, OrderbookLevel) else OrderbookLevel.from_raw(level)
            for level in self.bids
        ]
        self.asks = [
            level if isinstance(level, OrderbookLevel) else OrderbookLevel.from_raw(level)
            for level in self.asks
        ]

        self.bids = [level for level in self.bids if level is not None and level.is_valid]
        self.asks = [level for level in self.asks if level is not None and level.is_valid]

        self.bids.sort(key=lambda item: item.price, reverse=True)
        self.asks.sort(key=lambda item: item.price)

        if self.timestamp <= 0:
            raise ValueError("OrderbookSnapshot.timestamp must be > 0")

    @property
    def key(self) -> OrderFlowKey:
        return make_orderflow_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def market_key(self) -> tuple[str, str, str]:
        return self.exchange, self.market_type, self.symbol

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        bids: list[Any],
        asks: list[Any],
        exchange: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
        timestamp: float | None = None,
        sequence_id: str | int | None = None,
        raw: dict[str, Any] | None = None,
    ) -> OrderbookSnapshot:
        return cls(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            exchange_symbol=exchange_symbol,
            timeframe=timeframe,
            bids=[
                level
                for item in bids
                if (level := OrderbookLevel.from_raw(item)) is not None and level.is_valid
            ],
            asks=[
                level
                for item in asks
                if (level := OrderbookLevel.from_raw(item)) is not None and level.is_valid
            ],
            timestamp=float(timestamp if timestamp is not None else time.time()),
            sequence_id=str(sequence_id) if sequence_id is not None else None,
            raw=raw,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrderbookSnapshot:
        return cls.create(
            exchange=str(data["exchange"]),
            market_type=str(data.get("market_type") or DEFAULT_MARKET_TYPE),
            symbol=str(data["symbol"]),
            exchange_symbol=data.get("exchange_symbol"),
            timeframe=str(data.get("timeframe") or DEFAULT_TIMEFRAME),
            bids=list(data.get("bids") or []),
            asks=list(data.get("asks") or []),
            timestamp=float(
                data.get(
                    "timestamp",
                    data.get("timestamp_ms", data.get("last_update_ts_ms", time.time())),
                )
            ),
            sequence_id=data.get("sequence_id") or data.get("sequence"),
            raw=dict(data.get("raw") or {}),
        )

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> float | None:
        best_bid = self.best_bid
        best_ask = self.best_ask

        if best_bid is None or best_ask is None:
            return None

        return best_ask - best_bid

    @property
    def mid_price(self) -> float | None:
        best_bid = self.best_bid
        best_ask = self.best_ask

        if best_bid is None or best_ask is None:
            return None

        return (best_bid + best_ask) / 2.0

    @property
    def is_valid(self) -> bool:
        return (
            bool(self.exchange)
            and bool(self.market_type)
            and bool(self.symbol)
            and bool(self.bids)
            and bool(self.asks)
            and self.timestamp > 0
        )

    def to_dict(self) -> dict[str, Any]:
        payload = model_to_dict(self)
        payload["key"] = list(self.key)
        payload["best_bid"] = self.best_bid
        payload["best_ask"] = self.best_ask
        payload["spread"] = self.spread
        payload["mid_price"] = self.mid_price
        return payload


# ---------------------------------------------------------------------
# Base output models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class BaseOrderFlowStats:
    """
    Base stats contract for all analytics.orderflow metrics.

    Every stats object must carry futures scope:
        exchange + market_type + symbol + timeframe
    """

    symbol: str
    metric: OrderFlowMetricType
    source_type: OrderFlowSourceType

    exchange: str
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = _normalize_symbol(self.symbol)
        self.exchange = _normalize_exchange(self.exchange)
        self.market_type = _normalize_market_type(self.market_type)
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.metric = _coerce_metric_type(self.metric)
        self.source_type = _coerce_source_type(self.source_type)
        self.timestamp = float(self.timestamp)
        self.metadata = dict(self.metadata or {})

        if self.timestamp <= 0:
            raise ValueError("BaseOrderFlowStats.timestamp must be > 0")

    @property
    def key(self) -> OrderFlowKey:
        return make_orderflow_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = model_to_dict(self)
        payload["key"] = list(self.key)
        return payload


@dataclass(slots=True)
class OrderFlowUpdate:
    """
    Generic update payload emitted to analytics.orderflow.*.updated topics.
    """

    symbol: str
    metric: OrderFlowMetricType
    source_type: OrderFlowSourceType
    stats: dict[str, Any]

    exchange: str
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.symbol = _normalize_symbol(self.symbol)
        self.exchange = _normalize_exchange(self.exchange)
        self.market_type = _normalize_market_type(self.market_type)
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.metric = _coerce_metric_type(self.metric)
        self.source_type = _coerce_source_type(self.source_type)
        self.stats = dict(self.stats or {})
        self.timestamp = float(self.timestamp)

        if self.timestamp <= 0:
            raise ValueError("OrderFlowUpdate.timestamp must be > 0")

    @property
    def key(self) -> OrderFlowKey:
        return make_orderflow_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @classmethod
    def from_stats(cls, stats: BaseOrderFlowStats) -> OrderFlowUpdate:
        return cls(
            exchange=stats.exchange,
            market_type=stats.market_type,
            symbol=stats.symbol,
            exchange_symbol=stats.exchange_symbol,
            timeframe=stats.timeframe,
            metric=stats.metric,
            source_type=stats.source_type,
            stats=stats.to_dict(),
            timestamp=stats.timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = model_to_dict(self)
        payload["key"] = list(self.key)
        return payload


@dataclass(slots=True)
class OrderFlowSignal:
    """
    Generic signal payload emitted to analytics.orderflow.*.signal topics.

    Keep this payload contract aligned with OrderFlowUpdate:
    - exchange / market_type / symbol / timeframe define futures scope;
    - metric: metric that produced the signal;
    - source_type: source data category used by the analyzer;
    - signal_type: bullish/bearish/neutral/info;
    - side: buy/sell/unknown semantic side;
    - strength: normalized confidence/strength in [0.0, 1.0];
    - reason: machine-readable reason for downstream strategy/risk modules;
    - context: JSON-friendly analyzer-specific metadata.
    """

    symbol: str
    metric: OrderFlowMetricType
    source_type: OrderFlowSourceType
    signal_type: OrderFlowSignalType
    side: OrderFlowSide
    strength: float
    reason: str

    exchange: str
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.symbol = _normalize_symbol(self.symbol)
        self.exchange = _normalize_exchange(self.exchange)
        self.market_type = _normalize_market_type(self.market_type)
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.metric = _coerce_metric_type(self.metric)
        self.source_type = _coerce_source_type(self.source_type)
        self.signal_type = _coerce_signal_type(self.signal_type)
        self.side = _coerce_side(self.side)
        self.strength = _clamp(self.strength)
        self.reason = str(self.reason or "").strip()
        self.context = dict(self.context or {})
        self.timestamp = float(self.timestamp)

        if not self.reason:
            raise ValueError("OrderFlowSignal.reason must not be empty")
        if self.timestamp <= 0:
            raise ValueError("OrderFlowSignal.timestamp must be > 0")

    @property
    def key(self) -> OrderFlowKey:
        return make_orderflow_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def is_directional(self) -> bool:
        return self.signal_type.is_directional

    def to_dict(self) -> dict[str, Any]:
        payload = model_to_dict(self)
        payload["key"] = list(self.key)
        return payload


# ---------------------------------------------------------------------
# Metric-specific models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class CvdPoint:
    timestamp: float
    value: float

    exchange: str
    market_type: str
    symbol: str
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    price: float | None = None

    def __post_init__(self) -> None:
        self.exchange = _normalize_exchange(self.exchange)
        self.market_type = _normalize_market_type(self.market_type)
        self.symbol = _normalize_symbol(self.symbol)
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.timestamp = float(self.timestamp)
        self.value = float(self.value)
        self.price = _safe_float(self.price)

        if self.timestamp <= 0:
            raise ValueError("CvdPoint.timestamp must be > 0")

    @property
    def key(self) -> OrderFlowKey:
        return make_orderflow_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = model_to_dict(self)
        payload["key"] = list(self.key)
        return payload


@dataclass(slots=True)
class CvdStats(BaseOrderFlowStats):
    window_seconds: float = 0.0
    trades_count: int = 0

    buy_volume: float = 0.0
    sell_volume: float = 0.0
    volume_delta: float = 0.0

    buy_notional: float = 0.0
    sell_notional: float = 0.0
    notional_delta: float = 0.0

    cvd_value: float = 0.0
    cvd_open: float = 0.0
    cvd_high: float = 0.0
    cvd_low: float = 0.0
    cvd_close: float = 0.0
    cvd_change: float = 0.0
    cvd_change_pct: float = 0.0
    cvd_slope: float = 0.0

    delta_ratio: float = 0.0
    buy_ratio: float = 0.0
    sell_ratio: float = 0.0

    avg_trade_size: float = 0.0
    avg_trade_notional: float = 0.0

    last_price: float | None = None
    price_change: float | None = None
    price_change_pct: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.window_seconds = float(self.window_seconds)
        self.trades_count = int(self.trades_count)
        self.last_price = _safe_float(self.last_price)
        self.price_change = _safe_float(self.price_change)
        self.price_change_pct = _safe_float(self.price_change_pct)


@dataclass(slots=True)
class VolumeDeltaStats(BaseOrderFlowStats):
    window_seconds: float = 0.0
    trades_count: int = 0

    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_notional: float = 0.0
    sell_notional: float = 0.0

    volume_delta: float = 0.0
    notional_delta: float = 0.0
    delta_ratio: float = 0.0

    cumulative_volume_delta: float = 0.0
    cumulative_notional_delta: float = 0.0

    buy_ratio: float = 0.0
    sell_ratio: float = 0.0

    avg_trade_size: float = 0.0
    avg_trade_notional: float = 0.0

    last_price: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.window_seconds = float(self.window_seconds)
        self.trades_count = int(self.trades_count)
        self.last_price = _safe_float(self.last_price)


@dataclass(slots=True)
class AggressiveTradesStats(BaseOrderFlowStats):
    window_seconds: float = 0.0
    trades_count: int = 0

    aggressive_buy_count: int = 0
    aggressive_sell_count: int = 0

    aggressive_buy_volume: float = 0.0
    aggressive_sell_volume: float = 0.0

    aggressive_buy_notional: float = 0.0
    aggressive_sell_notional: float = 0.0

    net_volume_delta: float = 0.0
    net_notional_delta: float = 0.0

    buy_ratio: float = 0.0
    sell_ratio: float = 0.0

    burst_score: float = 0.0
    large_buy_trades: int = 0
    large_sell_trades: int = 0

    avg_trade_size: float = 0.0
    avg_trade_notional: float = 0.0

    last_price: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.window_seconds = float(self.window_seconds)
        self.trades_count = int(self.trades_count)
        self.aggressive_buy_count = int(self.aggressive_buy_count)
        self.aggressive_sell_count = int(self.aggressive_sell_count)
        self.large_buy_trades = int(self.large_buy_trades)
        self.large_sell_trades = int(self.large_sell_trades)
        self.burst_score = _clamp(self.burst_score)
        self.last_price = _safe_float(self.last_price)


@dataclass(slots=True)
class OrderbookImbalanceStats(BaseOrderFlowStats):
    bid_volume: float = 0.0
    ask_volume: float = 0.0

    imbalance_ratio: float = 0.0
    imbalance_diff: float = 0.0

    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    mid_price: float | None = None

    depth_levels_used: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.best_bid = _safe_float(self.best_bid)
        self.best_ask = _safe_float(self.best_ask)
        self.spread = _safe_float(self.spread)
        self.mid_price = _safe_float(self.mid_price)
        self.depth_levels_used = int(self.depth_levels_used)


# ---------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------


def model_to_dict(model: Any) -> dict[str, Any]:
    """
    Convert an order-flow dataclass model to a JSON/EventBus-friendly dict.
    """
    if not is_dataclass(model):
        raise TypeError(f"Expected dataclass instance, got: {type(model)!r}")

    return {
        key: _serialize_value(value)
        for key, value in asdict(model).items()
    }


def stats_to_dict(stats: BaseOrderFlowStats) -> dict[str, Any]:
    return stats.to_dict()


def signal_to_dict(signal: OrderFlowSignal) -> dict[str, Any]:
    return signal.to_dict()


def update_to_dict(update: OrderFlowUpdate) -> dict[str, Any]:
    return update.to_dict()


def orderbook_snapshot_to_dict(snapshot: OrderbookSnapshot) -> dict[str, Any]:
    return snapshot.to_dict()


def trade_to_dict(trade: NormalizedTrade) -> dict[str, Any]:
    return trade.to_dict()