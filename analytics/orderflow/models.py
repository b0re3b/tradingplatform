from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any

from .enums import (
    OrderFlowMetricType,
    OrderFlowSide,
    OrderFlowSignalType,
    OrderFlowSourceType,
)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _safe_float(value: object) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_symbol(symbol: object) -> str:
    return str(symbol).strip().upper()


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
        return tuple(_serialize_value(item) for item in value)

    if isinstance(value, set):
        return sorted(_serialize_value(item) for item in value)

    return value


# ---------------------------------------------------------------------
# Base input models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class NormalizedTrade:
    """
    Canonical trade model consumed by analytics.orderflow modules.

    Exchange adapters / data layer should normalize raw exchange payloads into
    this structure before order-flow analyzers calculate metrics.
    """

    symbol: str
    side: OrderFlowSide
    price: float
    quantity: float
    notional: float
    timestamp: float
    trade_id: str | None = None
    exchange: str | None = None
    is_aggressive: bool = False
    raw: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        side: str | OrderFlowSide,
        price: float,
        quantity: float,
        timestamp: float,
        trade_id: str | None = None,
        exchange: str | None = None,
        is_aggressive: bool = False,
        raw: dict[str, Any] | None = None,
    ) -> "NormalizedTrade":
        side_enum = OrderFlowSide.from_value(side)

        price_f = float(price)
        quantity_f = float(quantity)

        return cls(
            symbol=_normalize_symbol(symbol),
            side=side_enum,
            price=price_f,
            quantity=quantity_f,
            notional=price_f * quantity_f,
            timestamp=float(timestamp),
            trade_id=str(trade_id) if trade_id is not None else None,
            exchange=str(exchange) if exchange is not None else None,
            is_aggressive=bool(is_aggressive),
            raw=raw,
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
            and self.side.is_known
            and self.price > 0
            and self.quantity > 0
            and self.timestamp > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return model_to_dict(self)


@dataclass(slots=True)
class OrderbookLevel:
    """
    Single normalized orderbook level.
    """

    price: float
    size: float

    @classmethod
    def from_raw(cls, raw: Any) -> "OrderbookLevel | None":
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
    Canonical orderbook snapshot consumed by orderbook-based analyzers.
    """

    symbol: str
    bids: list[OrderbookLevel]
    asks: list[OrderbookLevel]
    timestamp: float
    exchange: str | None = None
    sequence_id: str | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        bids: list[Any],
        asks: list[Any],
        timestamp: float | None = None,
        exchange: str | None = None,
        sequence_id: str | int | None = None,
        raw: dict[str, Any] | None = None,
    ) -> "OrderbookSnapshot":
        normalized_bids = [
            level
            for item in bids
            if (level := OrderbookLevel.from_raw(item)) is not None and level.is_valid
        ]
        normalized_asks = [
            level
            for item in asks
            if (level := OrderbookLevel.from_raw(item)) is not None and level.is_valid
        ]

        normalized_bids.sort(key=lambda item: item.price, reverse=True)
        normalized_asks.sort(key=lambda item: item.price)

        return cls(
            symbol=_normalize_symbol(symbol),
            bids=normalized_bids,
            asks=normalized_asks,
            timestamp=float(timestamp if timestamp is not None else time.time()),
            exchange=str(exchange) if exchange is not None else None,
            sequence_id=str(sequence_id) if sequence_id is not None else None,
            raw=raw,
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
        return bool(self.symbol) and bool(self.bids) and bool(self.asks) and self.timestamp > 0

    def to_dict(self) -> dict[str, Any]:
        return model_to_dict(self)


# ---------------------------------------------------------------------
# Base output models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class BaseOrderFlowStats:
    """
    Base stats contract for all analytics.orderflow metrics.
    """

    symbol: str
    metric: OrderFlowMetricType
    source_type: OrderFlowSourceType
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return model_to_dict(self)


@dataclass(slots=True)
class OrderFlowUpdate:
    """
    Generic update payload emitted to analytics.orderflow.*.updated topics.
    """

    symbol: str
    metric: OrderFlowMetricType
    source_type: OrderFlowSourceType
    stats: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return model_to_dict(self)


@dataclass(slots=True)
class OrderFlowSignal:
    """
    Generic signal payload emitted to analytics.orderflow.*.signal topics.
    """

    symbol: str
    metric: OrderFlowMetricType
    signal_type: OrderFlowSignalType
    side: OrderFlowSide
    strength: float
    reason: str
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.symbol = _normalize_symbol(self.symbol)
        self.strength = max(0.0, min(float(self.strength), 1.0))

    @property
    def is_directional(self) -> bool:
        return self.signal_type.is_directional

    def to_dict(self) -> dict[str, Any]:
        return model_to_dict(self)


# ---------------------------------------------------------------------
# Metric-specific models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class CvdPoint:
    timestamp: float
    value: float
    price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return model_to_dict(self)


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
    return model_to_dict(stats)


def signal_to_dict(signal: OrderFlowSignal) -> dict[str, Any]:
    return model_to_dict(signal)


def update_to_dict(update: OrderFlowUpdate) -> dict[str, Any]:
    return model_to_dict(update)


def orderbook_snapshot_to_dict(snapshot: OrderbookSnapshot) -> dict[str, Any]:
    return model_to_dict(snapshot)


def trade_to_dict(trade: NormalizedTrade) -> dict[str, Any]:
    return model_to_dict(trade)