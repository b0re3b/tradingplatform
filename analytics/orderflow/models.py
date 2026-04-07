from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

from .enums import (
    OrderFlowMetricType,
    OrderFlowSide,
    OrderFlowSignalType,
    OrderFlowSourceType,
)


# ---------------------------------------------------------------------
# Base input models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class NormalizedTrade:
    symbol: str
    side: OrderFlowSide
    price: float
    quantity: float
    notional: float
    timestamp: float
    trade_id: Optional[str] = None
    exchange: Optional[str] = None
    is_aggressive: bool = False
    raw: Optional[dict[str, Any]] = None

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        side: str | OrderFlowSide,
        price: float,
        quantity: float,
        timestamp: float,
        trade_id: Optional[str] = None,
        exchange: Optional[str] = None,
        is_aggressive: bool = False,
        raw: Optional[dict[str, Any]] = None,
    ) -> "NormalizedTrade":
        if isinstance(side, OrderFlowSide):
            side_enum = side
        else:
            side_str = str(side).lower()
            allowed_values = {item.value for item in OrderFlowSide}
            side_enum = (
                OrderFlowSide(side_str)
                if side_str in allowed_values
                else OrderFlowSide.UNKNOWN
            )

        price_f = float(price)
        quantity_f = float(quantity)
        notional = price_f * quantity_f

        return cls(
            symbol=str(symbol).upper(),
            side=side_enum,
            price=price_f,
            quantity=quantity_f,
            notional=notional,
            timestamp=float(timestamp),
            trade_id=trade_id,
            exchange=exchange,
            is_aggressive=bool(is_aggressive),
            raw=raw,
        )


from dataclasses import dataclass
from typing import Any, Optional


def _safe_float(value: object) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class OrderbookLevel:
    price: float
    size: float

    @classmethod
    def from_raw(cls, raw: Any) -> Optional["OrderbookLevel"]:
        if raw is None:
            return None

        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            price = _safe_float(raw[0])
            size = _safe_float(raw[1])

            if price is None or size is None:
                return None

            return cls(price=price, size=size)

        if isinstance(raw, dict):
            raw_dict: dict[object, object] = raw

            price = _safe_float(raw_dict.get("price"))
            size = _safe_float(
                raw_dict.get("size", raw_dict.get("quantity", raw_dict.get("qty")))
            )

            if price is None or size is None:
                return None

            return cls(price=price, size=size)

        return None


@dataclass(slots=True)
class OrderbookSnapshot:
    symbol: str
    bids: list[OrderbookLevel]
    asks: list[OrderbookLevel]
    timestamp: float
    exchange: Optional[str] = None
    sequence_id: Optional[str] = None
    raw: Optional[dict[str, Any]] = None

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        best_bid = self.best_bid
        best_ask = self.best_ask

        if best_bid is None or best_ask is None:
            return None

        return best_ask - best_bid

    @property
    def mid_price(self) -> Optional[float]:
        best_bid = self.best_bid
        best_ask = self.best_ask

        if best_bid is None or best_ask is None:
            return None

        return (best_bid + best_ask) / 2.0


# ---------------------------------------------------------------------
# Base output models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class BaseOrderFlowStats:
    symbol: str
    metric: OrderFlowMetricType
    source_type: OrderFlowSourceType
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class OrderFlowUpdate:
    symbol: str
    metric: OrderFlowMetricType
    source_type: OrderFlowSourceType
    stats: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class OrderFlowSignal:
    symbol: str
    metric: OrderFlowMetricType
    signal_type: OrderFlowSignalType
    side: OrderFlowSide
    strength: float
    reason: str
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------
# Metric-specific models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class CvdPoint:
    timestamp: float
    value: float
    price: Optional[float] = None


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

    last_price: Optional[float] = None
    price_change: Optional[float] = None
    price_change_pct: Optional[float] = None


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

    last_price: Optional[float] = None


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

    last_price: Optional[float] = None


@dataclass(slots=True)
class OrderbookImbalanceStats(BaseOrderFlowStats):
    bid_volume: float = 0.0
    ask_volume: float = 0.0

    imbalance_ratio: float = 0.0
    imbalance_diff: float = 0.0

    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None
    mid_price: Optional[float] = None

    depth_levels_used: int = 0


# ---------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------


def stats_to_dict(stats: BaseOrderFlowStats) -> dict[str, Any]:
    return asdict(stats)


def signal_to_dict(signal: OrderFlowSignal) -> dict[str, Any]:
    return asdict(signal)