from __future__ import annotations

from enum import Enum


class OrderFlowSide(str, Enum):
    """
    Normalized aggressor / flow side used by order-flow analytics.

    This enum is intentionally independent from exchange-specific naming.
    Exchange adapters and data normalizers should map raw values into these
    canonical values before analytics modules process them.
    """

    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"

    @classmethod
    def from_value(cls, value: object) -> "OrderFlowSide":
        if isinstance(value, cls):
            return value

        if value is None:
            return cls.UNKNOWN

        normalized = str(value).strip().lower()

        buy_aliases = {
            "buy",
            "bid",
            "b",
            "long",
            "maker_sell_false",
            "aggressive_buy",
        }
        sell_aliases = {
            "sell",
            "ask",
            "s",
            "short",
            "maker_sell_true",
            "aggressive_sell",
        }

        if normalized in buy_aliases:
            return cls.BUY

        if normalized in sell_aliases:
            return cls.SELL

        return cls.UNKNOWN

    @property
    def is_buy(self) -> bool:
        return self is self.BUY

    @property
    def is_sell(self) -> bool:
        return self is self.SELL

    @property
    def is_known(self) -> bool:
        return self in {self.BUY, self.SELL}


class OrderFlowMetricType(str, Enum):
    """
    Order-flow metric identifiers.

    These values are used in:
    - stats models
    - update payloads
    - signal payloads
    - module metrics
    - logs
    """

    CVD = "cvd"
    VOLUME_DELTA = "volume_delta"
    AGGRESSIVE_TRADES = "aggressive_trades"
    ORDERBOOK_IMBALANCE = "orderbook_imbalance"


class OrderFlowSignalType(str, Enum):
    """
    Canonical signal direction / semantic type produced by order-flow analyzers.
    """

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    INFO = "info"

    @property
    def is_directional(self) -> bool:
        return self in {self.BULLISH, self.BEARISH}


class OrderFlowSourceType(str, Enum):
    """
    Source data category consumed by an order-flow analyzer.
    """

    TRADES = "trades"
    ORDERBOOK = "orderbook"
    UNKNOWN = "unknown"


class OrderFlowEventTopic(str, Enum):
    """
    EventBus topics emitted by analytics.orderflow modules.

    Topic convention:
        analytics.orderflow.<metric>.<event>

    Data layer should publish market.* events.
    Order-flow analyzers should consume market.* events and emit these
    analytics.orderflow.* events.
    Strategy modules should subscribe to these topics instead of calling
    order-flow analyzers directly.
    """

    # ------------------------------------------------------------------
    # Package lifecycle / service-level events
    # ------------------------------------------------------------------

    STARTED = "analytics.orderflow.started"
    STOPPED = "analytics.orderflow.stopped"
    HEALTH = "analytics.orderflow.health"
    ERROR = "analytics.orderflow.error"

    # ------------------------------------------------------------------
    # CVD events
    # ------------------------------------------------------------------

    CVD_UPDATED = "analytics.orderflow.cvd.updated"
    CVD_SIGNAL = "analytics.orderflow.cvd.signal"

    # ------------------------------------------------------------------
    # Volume delta events
    # ------------------------------------------------------------------

    VOLUME_DELTA_UPDATED = "analytics.orderflow.volume_delta.updated"
    VOLUME_DELTA_SIGNAL = "analytics.orderflow.volume_delta.signal"

    # ------------------------------------------------------------------
    # Aggressive trades events
    # ------------------------------------------------------------------

    AGGRESSIVE_TRADES_UPDATED = "analytics.orderflow.aggressive_trades.updated"
    AGGRESSIVE_TRADES_SIGNAL = "analytics.orderflow.aggressive_trades.signal"

    # ------------------------------------------------------------------
    # Orderbook imbalance events
    # ------------------------------------------------------------------

    ORDERBOOK_IMBALANCE_UPDATED = "analytics.orderflow.orderbook_imbalance.updated"
    ORDERBOOK_IMBALANCE_SIGNAL = "analytics.orderflow.orderbook_imbalance.signal"


class OrderFlowInputTopic(str, Enum):
    """
    Canonical market-data topics consumed by analytics.orderflow modules.

    These are not emitted by orderflow itself. They are expected to come from:
    - data.market_stream
    - exchange websocket adapters
    - cache update publishers
    """

    MARKET_TRADE = "market.trade"
    MARKET_TRADE_WILDCARD = "market.trade.*"
    MARKET_TRADES_UPDATED = "market.trades.updated"

    MARKET_ORDERBOOK_UPDATED = "market.orderbook.updated"
    MARKET_ORDERBOOK_SNAPSHOT = "market.orderbook.snapshot"
    MARKET_ORDERBOOK_WILDCARD = "market.orderbook.*"


TRADE_INPUT_TOPICS: tuple[str, ...] = (
    OrderFlowInputTopic.MARKET_TRADE.value,
    OrderFlowInputTopic.MARKET_TRADE_WILDCARD.value,
    OrderFlowInputTopic.MARKET_TRADES_UPDATED.value,
)

ORDERBOOK_INPUT_TOPICS: tuple[str, ...] = (
    OrderFlowInputTopic.MARKET_ORDERBOOK_UPDATED.value,
    OrderFlowInputTopic.MARKET_ORDERBOOK_SNAPSHOT.value,
    OrderFlowInputTopic.MARKET_ORDERBOOK_WILDCARD.value,
)


METRIC_UPDATE_TOPICS: dict[OrderFlowMetricType, OrderFlowEventTopic] = {
    OrderFlowMetricType.CVD: OrderFlowEventTopic.CVD_UPDATED,
    OrderFlowMetricType.VOLUME_DELTA: OrderFlowEventTopic.VOLUME_DELTA_UPDATED,
    OrderFlowMetricType.AGGRESSIVE_TRADES: OrderFlowEventTopic.AGGRESSIVE_TRADES_UPDATED,
    OrderFlowMetricType.ORDERBOOK_IMBALANCE: OrderFlowEventTopic.ORDERBOOK_IMBALANCE_UPDATED,
}


METRIC_SIGNAL_TOPICS: dict[OrderFlowMetricType, OrderFlowEventTopic] = {
    OrderFlowMetricType.CVD: OrderFlowEventTopic.CVD_SIGNAL,
    OrderFlowMetricType.VOLUME_DELTA: OrderFlowEventTopic.VOLUME_DELTA_SIGNAL,
    OrderFlowMetricType.AGGRESSIVE_TRADES: OrderFlowEventTopic.AGGRESSIVE_TRADES_SIGNAL,
    OrderFlowMetricType.ORDERBOOK_IMBALANCE: OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL,
}


def get_update_topic(metric_type: OrderFlowMetricType) -> str:
    """
    Return canonical analytics.orderflow.*.updated topic for a metric.
    """
    return METRIC_UPDATE_TOPICS[metric_type].value


def get_signal_topic(metric_type: OrderFlowMetricType) -> str:
    """
    Return canonical analytics.orderflow.*.signal topic for a metric.
    """
    return METRIC_SIGNAL_TOPICS[metric_type].value