from __future__ import annotations
from core.logger import get_logger

from enum import Enum


class WhaleTradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"

    @classmethod
    def normalize(cls, value: object) -> "WhaleTradeSide":
        """
        Normalize common exchange-side formats into a whale trade side.

        Підтримує:
        - buy / sell
        - bid / ask
        - long / short
        - b / s
        - boolean maker flag у вигляді рядка
        """
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "WhaleTradeSide"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalize", _analytics_args)
        except Exception:
            pass
        if value is None:
            return cls.UNKNOWN

        normalized = str(value).strip().lower()

        if normalized in {"buy", "bid", "long", "b"}:
            return cls.BUY

        if normalized in {"sell", "ask", "short", "s"}:
            return cls.SELL

        return cls.UNKNOWN

    @classmethod
    def from_maker_flag(cls, maker_flag: object) -> "WhaleTradeSide":
        """
        Convert trade maker flag into taker-side approximation.

        Для багатьох бірж:
        - maker_flag=True означає buyer is maker, отже taker side = sell;
        - maker_flag=False означає seller is maker, отже taker side = buy.
        """
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "WhaleTradeSide"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_maker_flag", _analytics_args)
        except Exception:
            pass
        if maker_flag is None:
            return cls.UNKNOWN

        if isinstance(maker_flag, bool):
            return cls.SELL if maker_flag else cls.BUY

        normalized = str(maker_flag).strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return cls.SELL
        if normalized in {"false", "0", "no", "n"}:
            return cls.BUY

        return cls.UNKNOWN

    @property
    def opposite(self) -> "WhaleTradeSide":
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "opposite", _analytics_args)
        except Exception:
            pass
        if self is WhaleTradeSide.BUY:
            return WhaleTradeSide.SELL
        if self is WhaleTradeSide.SELL:
            return WhaleTradeSide.BUY
        return WhaleTradeSide.UNKNOWN


class LargeTradeTriggerType(str, Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    ABSOLUTE_AND_RELATIVE = "absolute_and_relative"
    UNKNOWN = "unknown"

    @classmethod
    def from_flags(
        cls,
        *,
        absolute_triggered: bool,
        relative_triggered: bool,
    ) -> "LargeTradeTriggerType":
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "LargeTradeTriggerType"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_flags", _analytics_args)
        except Exception:
            pass
        if absolute_triggered and relative_triggered:
            return cls.ABSOLUTE_AND_RELATIVE
        if absolute_triggered:
            return cls.ABSOLUTE
        if relative_triggered:
            return cls.RELATIVE
        return cls.UNKNOWN


class WhaleEventType(str, Enum):
    LARGE_TRADE = "large_trade"
    WHALE_ACTIVITY = "whale_activity"
    WHALE_PRESSURE = "whale_pressure"
    WHALE_LIQUIDATION_CONTEXT = "whale_liquidation_context"
    WHALE_CLUSTER = "whale_cluster"
    WHALE_CLUSTER_UPDATE = "whale_cluster_update"
    WHALE_CLUSTER_EXHAUSTION = "whale_cluster_exhaustion"


class WhaleEventTopic(str, Enum):
    """
    Canonical EventBus topics for analytics.whales.

    Runtime-компоненти мають використовувати ці значення як дефолтні topics
    у config-класах, а самі підписки/публікації виконувати через core EventBus.
    """

    MARKET_TRADE = "market.trade"
    MARKET_LIQUIDATION = "market.liquidation"

    LARGE_TRADE = "analytics.whales.large_trade"
    WHALE_ACTIVITY = "analytics.whales.whale_activity"
    WHALE_PRESSURE = "analytics.whales.whale_pressure"
    WHALE_LIQUIDATION_CONTEXT = "analytics.whales.whale_liquidation_context"
    WHALE_CLUSTER = "analytics.whales.whale_cluster"
    WHALE_CLUSTER_UPDATE = "analytics.whales.whale_cluster_update"
    WHALE_CLUSTER_EXHAUSTION = "analytics.whales.whale_cluster_exhaustion"


class WhaleComponentName(str, Enum):
    LARGE_TRADE_DETECTOR = "large_trade_detector"
    WHALE_TRACKER = "whale_tracker"
    WHALE_CLUSTER_ANALYZER = "whale_cluster_analyzer"
    WHALE_ANALYZER = "analyzer"


class WhaleBias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"

    @classmethod
    def from_side(cls, side: WhaleTradeSide | str) -> "WhaleBias":
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "WhaleBias"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_side", _analytics_args)
        except Exception:
            pass
        normalized_side = (
            side
            if isinstance(side, WhaleTradeSide)
            else WhaleTradeSide.normalize(side)
        )

        if normalized_side is WhaleTradeSide.BUY:
            return cls.BULLISH
        if normalized_side is WhaleTradeSide.SELL:
            return cls.BEARISH
        return cls.UNKNOWN


class WhaleClusterStateType(str, Enum):
    FORMING = "forming"
    ACTIVE = "active"
    EXHAUSTING = "exhausting"
    INACTIVE = "inactive"

    @classmethod
    def from_scores(
        cls,
        *,
        cluster_score: float,
        exhaustion_probability: float,
        active_threshold: float = 0.60,
        exhaustion_threshold: float = 0.70,
    ) -> "WhaleClusterStateType":
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "WhaleClusterStateType"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_scores", _analytics_args)
        except Exception:
            pass
        if exhaustion_probability >= exhaustion_threshold:
            return cls.EXHAUSTING
        if cluster_score >= active_threshold:
            return cls.ACTIVE
        if cluster_score > 0.0:
            return cls.FORMING
        return cls.INACTIVE


class WhalePressureType(str, Enum):
    BUY_PRESSURE = "buy_pressure"
    SELL_PRESSURE = "sell_pressure"
    BALANCED = "balanced"
    UNKNOWN = "unknown"

    @classmethod
    def from_notional(
        cls,
        *,
        buy_notional: float,
        sell_notional: float,
    ) -> "WhalePressureType":
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "WhalePressureType"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_notional", _analytics_args)
        except Exception:
            pass
        if buy_notional > sell_notional:
            return cls.BUY_PRESSURE
        if sell_notional > buy_notional:
            return cls.SELL_PRESSURE
        if buy_notional == sell_notional:
            return cls.BALANCED
        return cls.UNKNOWN


class WhaleNormalizationStatus(str, Enum):
    NORMALIZED = "normalized"
    INVALID = "invalid"
    FILTERED = "filtered"


class WhaleDataSource(str, Enum):
    """
    Semantic source names used in payloads/logs.

    Для EventBus topics краще використовувати WhaleEventTopic.
    Цей enum залишений для сумісності з моделями, payload metadata і логами.
    """

    MARKET_TRADE = WhaleEventTopic.MARKET_TRADE.value
    MARKET_LIQUIDATION = WhaleEventTopic.MARKET_LIQUIDATION.value

    ANALYTICS_WHALES_LARGE_TRADE = WhaleEventTopic.LARGE_TRADE.value
    ANALYTICS_WHALES_ACTIVITY = WhaleEventTopic.WHALE_ACTIVITY.value
    ANALYTICS_WHALES_PRESSURE = WhaleEventTopic.WHALE_PRESSURE.value
    ANALYTICS_WHALES_LIQUIDATION_CONTEXT = (
        WhaleEventTopic.WHALE_LIQUIDATION_CONTEXT.value
    )
    ANALYTICS_WHALES_CLUSTER = WhaleEventTopic.WHALE_CLUSTER.value
    ANALYTICS_WHALES_CLUSTER_UPDATE = WhaleEventTopic.WHALE_CLUSTER_UPDATE.value
    ANALYTICS_WHALES_CLUSTER_EXHAUSTION = (
        WhaleEventTopic.WHALE_CLUSTER_EXHAUSTION.value
    )


__all__ = [
    "WhaleTradeSide",
    "LargeTradeTriggerType",
    "WhaleEventType",
    "WhaleEventTopic",
    "WhaleComponentName",
    "WhaleBias",
    "WhaleClusterStateType",
    "WhalePressureType",
    "WhaleNormalizationStatus",
    "WhaleDataSource",
]