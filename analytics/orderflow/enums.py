from __future__ import annotations
from core.logger import get_logger

from enum import Enum


class OrderFlowSide(str, Enum):
    """
    Normalized aggressor / flow side used by futures order-flow analytics.

    Exchange adapters and data caches should normalize raw exchange values into
    these canonical values before analytics modules process them.

    Futures-only note:
    - "long" is treated as aggressive buy context;
    - "short" is treated as aggressive sell context;
    - this enum is still side/flow-oriented, not position-management logic.
    """

    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"

    @classmethod
    def from_value(cls, value: object) -> OrderFlowSide:
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "OrderFlowSide"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_value", _analytics_args)
        except Exception:
            pass
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
            "taker_buy",
            "aggressive_buy",
            "maker_sell_false",
            "is_buyer_maker_false",
            "buyer_maker_false",
        }
        sell_aliases = {
            "sell",
            "ask",
            "s",
            "short",
            "taker_sell",
            "aggressive_sell",
            "maker_sell_true",
            "is_buyer_maker_true",
            "buyer_maker_true",
        }

        if normalized in buy_aliases:
            return cls.BUY

        if normalized in sell_aliases:
            return cls.SELL

        return cls.UNKNOWN

    @property
    def is_buy(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_buy", _analytics_args)
        except Exception:
            pass
        return self is OrderFlowSide.BUY

    @property
    def is_sell(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_sell", _analytics_args)
        except Exception:
            pass
        return self is OrderFlowSide.SELL

    @property
    def is_known(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_known", _analytics_args)
        except Exception:
            pass
        return self in {OrderFlowSide.BUY, OrderFlowSide.SELL}

    @property
    def sign(self) -> int:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "sign", _analytics_args)
        except Exception:
            pass
        if self is OrderFlowSide.BUY:
            return 1
        if self is OrderFlowSide.SELL:
            return -1
        return 0


class OrderFlowMetricType(str, Enum):
    """
    Order-flow metric identifiers.

    These values are used in:
    - stats models;
    - update payloads;
    - signal payloads;
    - module metrics;
    - logs;
    - dashboard/storage routing.
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
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_directional", _analytics_args)
        except Exception:
            pass
        return self in {
            OrderFlowSignalType.BULLISH,
            OrderFlowSignalType.BEARISH,
        }


class OrderFlowSourceType(str, Enum):
    """
    Source data category consumed by an order-flow analyzer.

    Important:
    these are logical source categories, not exchange adapter names.
    """

    TRADES = "trades"
    ORDERBOOK = "orderbook"
    UNKNOWN = "unknown"


class OrderFlowEventTopic(str, Enum):
    """
    EventBus topics emitted by analytics.orderflow modules.

    Topic convention:
        analytics.orderflow.<metric>.<event>

    Data layer publishes:
        market.trades.updated
        market.orderbook.updated

    Order-flow analyzers consume those data-layer topics and emit these
    analytics.orderflow.* topics.

    Strategy, Open Interest, Liquidity, Dashboard, Bots and Storage should
    subscribe to analytics.orderflow.* topics instead of calling analyzers
    directly.
    """

    # ------------------------------------------------------------------
    # Package lifecycle / service-level events
    # ------------------------------------------------------------------

    STARTED = "analytics.orderflow.started"
    STOPPED = "analytics.orderflow.stopped"
    HEALTH = "analytics.orderflow.health"
    ERROR = "analytics.orderflow.error"

    # Optional service-level aggregate update for downstream packages that
    # want a single orderflow topic instead of metric-specific topics.
    UPDATED = "analytics.orderflow.updated"

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

    @property
    def topic(self) -> str:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "topic", _analytics_args)
        except Exception:
            pass
        return self.value


class OrderFlowInputTopic(str, Enum):
    """
    Canonical data-layer topics consumed by analytics.orderflow modules.

    Orderflow must NOT consume raw exchange adapter events by default.

    Correct flow:
        exchange adapters
            -> market.trade / market.orderbook
            -> TradesCache / OrderbookCache
            -> market.trades.updated / market.orderbook.updated
            -> analytics.orderflow
            -> analytics.orderflow.*

    Futures scope expected in payload:
        exchange + market_type + symbol + timeframe/window

    Futures market_type examples:
        binance: usdm_futures
        bybit: linear
        okx: swap
        mexc: usdm_futures
    """

    # Main trade context from TradesCache.
    MARKET_TRADES_UPDATED = "market.trades.updated"

    # Main orderbook context from OrderbookCache.
    MARKET_ORDERBOOK_UPDATED = "market.orderbook.updated"

    # ------------------------------------------------------------------
    # Legacy raw topics.
    # ------------------------------------------------------------------
    # These are intentionally NOT included in default TRADE_INPUT_TOPICS or
    # ORDERBOOK_INPUT_TOPICS. They can be used only explicitly in tests or
    # compatibility adapters if needed.
    MARKET_TRADE_LEGACY = "market.trade"
    MARKET_TRADE_WILDCARD_LEGACY = "market.trade.*"
    MARKET_ORDERBOOK_SNAPSHOT_LEGACY = "market.orderbook.snapshot"
    MARKET_ORDERBOOK_WILDCARD_LEGACY = "market.orderbook.*"

    @property
    def topic(self) -> str:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "topic", _analytics_args)
        except Exception:
            pass
        return self.value

    @property
    def is_data_layer_topic(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_data_layer_topic", _analytics_args)
        except Exception:
            pass
        return self in {
            OrderFlowInputTopic.MARKET_TRADES_UPDATED,
            OrderFlowInputTopic.MARKET_ORDERBOOK_UPDATED,
        }

    @property
    def is_legacy_raw_topic(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_legacy_raw_topic", _analytics_args)
        except Exception:
            pass
        return self in {
            OrderFlowInputTopic.MARKET_TRADE_LEGACY,
            OrderFlowInputTopic.MARKET_TRADE_WILDCARD_LEGACY,
            OrderFlowInputTopic.MARKET_ORDERBOOK_SNAPSHOT_LEGACY,
            OrderFlowInputTopic.MARKET_ORDERBOOK_WILDCARD_LEGACY,
        }


# ---------------------------------------------------------------------
# Default input topics
# ---------------------------------------------------------------------
# These are the only default topics analyzers should subscribe to.
# Raw exchange topics are intentionally excluded.
# ---------------------------------------------------------------------


TRADE_INPUT_TOPICS: tuple[str, ...] = (
    OrderFlowInputTopic.MARKET_TRADES_UPDATED.value,
)

ORDERBOOK_INPUT_TOPICS: tuple[str, ...] = (
    OrderFlowInputTopic.MARKET_ORDERBOOK_UPDATED.value,
)


# Explicit legacy groups for compatibility tests or temporary migration only.
# Do not use these as production defaults.

LEGACY_RAW_TRADE_INPUT_TOPICS: tuple[str, ...] = (
    OrderFlowInputTopic.MARKET_TRADE_LEGACY.value,
    OrderFlowInputTopic.MARKET_TRADE_WILDCARD_LEGACY.value,
)

LEGACY_RAW_ORDERBOOK_INPUT_TOPICS: tuple[str, ...] = (
    OrderFlowInputTopic.MARKET_ORDERBOOK_SNAPSHOT_LEGACY.value,
    OrderFlowInputTopic.MARKET_ORDERBOOK_WILDCARD_LEGACY.value,
)


# ---------------------------------------------------------------------
# Metric -> output topic maps
# ---------------------------------------------------------------------


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


def normalize_metric_type(metric_type: OrderFlowMetricType | str) -> OrderFlowMetricType:
    """
    Normalize metric enum/string into OrderFlowMetricType.
    """
    if isinstance(metric_type, OrderFlowMetricType):
        return metric_type

    return OrderFlowMetricType(str(metric_type))


def get_update_topic(metric_type: OrderFlowMetricType | str) -> str:
    """
    Return canonical analytics.orderflow.*.updated topic for a metric.
    """
    return METRIC_UPDATE_TOPICS[normalize_metric_type(metric_type)].value


def get_signal_topic(metric_type: OrderFlowMetricType | str) -> str:
    """
    Return canonical analytics.orderflow.*.signal topic for a metric.
    """
    return METRIC_SIGNAL_TOPICS[normalize_metric_type(metric_type)].value


def is_orderflow_update_topic(topic: str) -> bool:
    """
    Check whether a topic is one of metric-specific orderflow update topics.
    """
    return topic in {event_topic.value for event_topic in METRIC_UPDATE_TOPICS.values()}


def is_orderflow_signal_topic(topic: str) -> bool:
    """
    Check whether a topic is one of metric-specific orderflow signal topics.
    """
    return topic in {event_topic.value for event_topic in METRIC_SIGNAL_TOPICS.values()}