from __future__ import annotations

from enum import Enum


class OrderFlowSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class OrderFlowMetricType(str, Enum):
    CVD = "cvd"
    VOLUME_DELTA = "volume_delta"
    AGGRESSIVE_TRADES = "aggressive_trades"
    ORDERBOOK_IMBALANCE = "orderbook_imbalance"


class OrderFlowSignalType(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    INFO = "info"


class OrderFlowSourceType(str, Enum):
    TRADES = "trades"
    ORDERBOOK = "orderbook"
    UNKNOWN = "unknown"


class OrderFlowEventTopic(str, Enum):
    CVD_UPDATED = "analytics.orderflow.cvd.updated"
    CVD_SIGNAL = "analytics.orderflow.cvd.signal"

    VOLUME_DELTA_UPDATED = "analytics.orderflow.volume_delta.updated"
    VOLUME_DELTA_SIGNAL = "analytics.orderflow.volume_delta.signal"

    AGGRESSIVE_TRADES_UPDATED = "analytics.orderflow.aggressive_trades.updated"
    AGGRESSIVE_TRADES_SIGNAL = "analytics.orderflow.aggressive_trades.signal"

    ORDERBOOK_IMBALANCE_UPDATED = "analytics.orderflow.orderbook_imbalance.updated"
    ORDERBOOK_IMBALANCE_SIGNAL = "analytics.orderflow.orderbook_imbalance.signal"