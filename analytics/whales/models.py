from __future__ import annotations
from core.logger import get_logger

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias

from analytics.whales.enums import (
    LargeTradeTriggerType,
    WhaleBias,
    WhaleClusterStateType,
    WhaleEventType,
    WhalePressureType,
    WhaleTradeSide,
)


# =============================================================================
# Common constants / helpers
# =============================================================================

DEFAULT_MARKET_TYPE = "perpetual"
DEFAULT_TIMEFRAME = "realtime"
UNKNOWN_EXCHANGE = "unknown"

WhaleKey: TypeAlias = tuple[str, str, str, str]
# exchange, market_type, symbol, timeframe


def utc_now_ms() -> int:
    return int(time.time() * 1000)


def _safe_non_negative(value: float) -> float:
    return max(0.0, float(value))


def _clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_exchange(exchange: object | None) -> str:
    value = str(exchange or UNKNOWN_EXCHANGE).strip().lower()
    return value or UNKNOWN_EXCHANGE


def normalize_market_type(market_type: object | None = None) -> str:
    value = str(market_type or DEFAULT_MARKET_TYPE).strip().lower()
    return value or DEFAULT_MARKET_TYPE


def normalize_symbol(symbol: object) -> str:
    value = (
        str(symbol or "")
        .replace("-", "")
        .replace("/", "")
        .replace("_", "")
        .upper()
        .strip()
    )
    if not value:
        raise ValueError("symbol must not be empty")
    return value


def normalize_timeframe(timeframe: object | None = None) -> str:
    value = str(timeframe or DEFAULT_TIMEFRAME).strip()
    return value or DEFAULT_TIMEFRAME


def normalize_exchange_symbol(
    exchange_symbol: object | None,
    *,
    fallback_symbol: str,
) -> str:
    value = str(exchange_symbol or "").strip()
    return value or fallback_symbol


def normalize_side(side: object | None) -> str:
    return WhaleTradeSide.normalize(side).value


def make_whale_key(
    *,
    exchange: object | None,
    market_type: object | None,
    symbol: object,
    timeframe: object | None = DEFAULT_TIMEFRAME,
) -> WhaleKey:
    return (
        normalize_exchange(exchange),
        normalize_market_type(market_type),
        normalize_symbol(symbol),
        normalize_timeframe(timeframe),
    )


def whale_key_to_dict(key: WhaleKey) -> dict[str, str]:
    exchange, market_type, symbol, timeframe = key
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def scope_payload(
    *,
    exchange: object | None,
    market_type: object | None,
    symbol: object,
    timeframe: object | None = DEFAULT_TIMEFRAME,
    exchange_symbol: object | None = None,
) -> dict[str, str]:
    key = make_whale_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    data = whale_key_to_dict(key)
    data["exchange_symbol"] = normalize_exchange_symbol(
        exchange_symbol,
        fallback_symbol=data["symbol"],
    )
    return data


def _metadata_copy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _payload_with_scope(
    payload: dict[str, Any],
    *,
    key: WhaleKey,
    exchange_symbol: str | None,
) -> dict[str, Any]:
    scope = whale_key_to_dict(key)
    payload.update(scope)
    payload["exchange_symbol"] = exchange_symbol or scope["symbol"]
    payload["scope"] = scope
    return payload


# =============================================================================
# Base signal model
# =============================================================================


@dataclass(slots=True)
class WhaleBaseSignalModel:
    """
    Базова модель для всіх whale-сигналів.

    Це не core.event_bus.Event.
    Runtime-компонент сам публікує payload через EventBus.emit().
    """

    detector_name: str
    event_type: str
    schema_version: int = 2
    created_at_ms: int = field(default_factory=utc_now_ms)

    def to_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "detector": self.detector_name,
            "created_at_ms": self.created_at_ms,
        }

    def to_event(self) -> dict[str, Any]:
        """
        Backward-compatible alias.
        """
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_event", _analytics_args)
        except Exception:
            pass
        return self.to_payload()


WhaleBaseEventModel = WhaleBaseSignalModel


# =============================================================================
# Raw / normalized records
# =============================================================================


@dataclass(slots=True)
class TradeRecord:
    """
    Нормалізований trade record після прийому data-layer trade payload.

    Production source:
        exchange adapters -> market.trade -> TradesCache -> market.trades.updated
        -> analytics.whales.LargeTradeDetector

    Scope:
        exchange + market_type + symbol + timeframe
    """

    symbol: str
    price: float
    quantity: float
    side: str
    timestamp_ms: int

    trade_id: str | None = None
    exchange: str | None = None
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    raw_event: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.side = normalize_side(self.side)
        self.price = float(self.price)
        self.quantity = float(self.quantity)
        self.timestamp_ms = int(self.timestamp_ms)
        self.metadata = _metadata_copy(self.metadata)

        if self.price <= 0:
            raise ValueError("price must be > 0")
        if self.quantity <= 0:
            raise ValueError("quantity must be > 0")
        if self.timestamp_ms <= 0:
            raise ValueError("timestamp_ms must be > 0")

    @property
    def key(self) -> WhaleKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "key", _analytics_args)
        except Exception:
            pass
        return make_whale_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def notional(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "notional", _analytics_args)
        except Exception:
            pass
        return self.price * self.quantity

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        payload = {
            "symbol": self.symbol,
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side,
            "timestamp_ms": self.timestamp_ms,
            "trade_id": self.trade_id,
            "notional": self.notional,
            "metadata": dict(self.metadata),
        }
        _payload_with_scope(
            payload,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )
        if include_raw:
            payload["raw_event"] = self.raw_event
        return payload


@dataclass(slots=True)
class WhaleTradeRecord:
    """
    Нормалізований record для вже виявленого великого трейду.
    """

    symbol: str
    side: str
    notional: float
    price: float
    quantity: float
    timestamp_ms: int

    zscore: float = 0.0
    trigger_type: str = LargeTradeTriggerType.UNKNOWN.value
    trade_id: str | None = None

    exchange: str | None = None
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    raw_event: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.side = normalize_side(self.side)
        self.notional = _safe_non_negative(self.notional)
        self.price = float(self.price)
        self.quantity = float(self.quantity)
        self.timestamp_ms = int(self.timestamp_ms)
        self.zscore = float(self.zscore)
        self.metadata = _metadata_copy(self.metadata)

    @property
    def key(self) -> WhaleKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "key", _analytics_args)
        except Exception:
            pass
        return make_whale_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @classmethod
    def from_large_trade_signal(cls, signal: "LargeTradeSignal") -> "WhaleTradeRecord":
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "WhaleTradeRecord"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_large_trade_signal", _analytics_args)
        except Exception:
            pass
        return cls(
            symbol=signal.symbol,
            side=signal.side,
            notional=signal.notional,
            price=signal.price,
            quantity=signal.quantity,
            timestamp_ms=signal.timestamp_ms,
            zscore=signal.zscore,
            trigger_type=signal.trigger_type,
            trade_id=signal.trade_id,
            exchange=signal.exchange,
            market_type=signal.market_type,
            timeframe=signal.timeframe,
            exchange_symbol=signal.exchange_symbol,
            metadata=dict(signal.metadata),
        )

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        payload = {
            "symbol": self.symbol,
            "side": self.side,
            "notional": self.notional,
            "price": self.price,
            "quantity": self.quantity,
            "timestamp_ms": self.timestamp_ms,
            "zscore": self.zscore,
            "trigger_type": self.trigger_type,
            "trade_id": self.trade_id,
            "metadata": dict(self.metadata),
        }
        _payload_with_scope(
            payload,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )
        if include_raw:
            payload["raw_event"] = self.raw_event
        return payload


@dataclass(slots=True)
class LiquidationRecord:
    """
    Нормалізований liquidation record.

    Production source:
        liquidation stream/cache -> market.liquidations.updated
        або analytics.liquidations.* -> WhaleTracker
    """

    symbol: str
    side: str
    notional: float
    price: float
    quantity: float
    timestamp_ms: int

    liquidation_id: str | None = None
    exchange: str | None = None
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    raw_event: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.side = normalize_side(self.side)
        self.notional = _safe_non_negative(self.notional)
        self.price = float(self.price)
        self.quantity = float(self.quantity)
        self.timestamp_ms = int(self.timestamp_ms)
        self.metadata = _metadata_copy(self.metadata)

    @property
    def key(self) -> WhaleKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "key", _analytics_args)
        except Exception:
            pass
        return make_whale_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        payload = {
            "symbol": self.symbol,
            "side": self.side,
            "notional": self.notional,
            "price": self.price,
            "quantity": self.quantity,
            "timestamp_ms": self.timestamp_ms,
            "liquidation_id": self.liquidation_id,
            "metadata": dict(self.metadata),
        }
        _payload_with_scope(
            payload,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )
        if include_raw:
            payload["raw_event"] = self.raw_event
        return payload


@dataclass(slots=True)
class WhaleActivityRecord:
    symbol: str
    side: str
    trade_count: int
    total_notional: float
    avg_notional: float
    max_notional: float
    window_sec: int
    timestamp_ms: int

    exchange: str | None = None
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    raw_event: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.side = normalize_side(self.side)
        self.trade_count = max(0, int(self.trade_count))
        self.total_notional = _safe_non_negative(self.total_notional)
        self.avg_notional = _safe_non_negative(self.avg_notional)
        self.max_notional = _safe_non_negative(self.max_notional)
        self.window_sec = max(0, int(self.window_sec))
        self.timestamp_ms = int(self.timestamp_ms)
        self.metadata = _metadata_copy(self.metadata)

    @property
    def key(self) -> WhaleKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "key", _analytics_args)
        except Exception:
            pass
        return make_whale_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @classmethod
    def from_signal(cls, signal: "WhaleActivitySignal") -> "WhaleActivityRecord":
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "WhaleActivityRecord"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_signal", _analytics_args)
        except Exception:
            pass
        return cls(
            symbol=signal.symbol,
            side=signal.side,
            trade_count=signal.trade_count,
            total_notional=signal.total_notional,
            avg_notional=signal.avg_notional,
            max_notional=signal.max_notional,
            window_sec=signal.window_sec,
            timestamp_ms=signal.timestamp_ms,
            exchange=signal.exchange,
            market_type=signal.market_type,
            timeframe=signal.timeframe,
            exchange_symbol=signal.exchange_symbol,
            metadata=dict(signal.metadata),
        )

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        payload = {
            "symbol": self.symbol,
            "side": self.side,
            "trade_count": self.trade_count,
            "total_notional": self.total_notional,
            "avg_notional": self.avg_notional,
            "max_notional": self.max_notional,
            "window_sec": self.window_sec,
            "timestamp_ms": self.timestamp_ms,
            "metadata": dict(self.metadata),
        }
        _payload_with_scope(
            payload,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )
        if include_raw:
            payload["raw_event"] = self.raw_event
        return payload


@dataclass(slots=True)
class WhalePressureRecord:
    symbol: str
    dominant_side: str
    buy_trade_count: int
    sell_trade_count: int
    buy_notional: float
    sell_notional: float
    total_notional: float
    imbalance_ratio: float
    net_flow_notional: float
    window_sec: int
    timestamp_ms: int

    exchange: str | None = None
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    raw_event: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.dominant_side = normalize_side(self.dominant_side)
        self.buy_trade_count = max(0, int(self.buy_trade_count))
        self.sell_trade_count = max(0, int(self.sell_trade_count))
        self.buy_notional = _safe_non_negative(self.buy_notional)
        self.sell_notional = _safe_non_negative(self.sell_notional)
        self.total_notional = _safe_non_negative(self.total_notional)
        self.imbalance_ratio = _clamp_0_1(self.imbalance_ratio)
        self.net_flow_notional = float(self.net_flow_notional)
        self.window_sec = max(0, int(self.window_sec))
        self.timestamp_ms = int(self.timestamp_ms)
        self.metadata = _metadata_copy(self.metadata)

    @property
    def key(self) -> WhaleKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "key", _analytics_args)
        except Exception:
            pass
        return make_whale_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def pressure_type(self) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "pressure_type", _analytics_args)
        except Exception:
            pass
        return WhalePressureType.from_notional(
            buy_notional=self.buy_notional,
            sell_notional=self.sell_notional,
        ).value

    @classmethod
    def from_signal(cls, signal: "WhalePressureSignal") -> "WhalePressureRecord":
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "WhalePressureRecord"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_signal", _analytics_args)
        except Exception:
            pass
        return cls(
            symbol=signal.symbol,
            dominant_side=signal.dominant_side,
            buy_trade_count=signal.buy_trade_count,
            sell_trade_count=signal.sell_trade_count,
            buy_notional=signal.buy_notional,
            sell_notional=signal.sell_notional,
            total_notional=signal.total_notional,
            imbalance_ratio=signal.imbalance_ratio,
            net_flow_notional=signal.net_flow_notional,
            window_sec=signal.window_sec,
            timestamp_ms=signal.timestamp_ms,
            exchange=signal.exchange,
            market_type=signal.market_type,
            timeframe=signal.timeframe,
            exchange_symbol=signal.exchange_symbol,
            metadata=dict(signal.metadata),
        )

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        payload = {
            "symbol": self.symbol,
            "dominant_side": self.dominant_side,
            "buy_trade_count": self.buy_trade_count,
            "sell_trade_count": self.sell_trade_count,
            "buy_notional": self.buy_notional,
            "sell_notional": self.sell_notional,
            "total_notional": self.total_notional,
            "imbalance_ratio": self.imbalance_ratio,
            "net_flow_notional": self.net_flow_notional,
            "pressure_type": self.pressure_type,
            "window_sec": self.window_sec,
            "timestamp_ms": self.timestamp_ms,
            "metadata": dict(self.metadata),
        }
        _payload_with_scope(
            payload,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )
        if include_raw:
            payload["raw_event"] = self.raw_event
        return payload


@dataclass(slots=True)
class WhaleLiquidationContextRecord:
    symbol: str
    whale_side: str
    whale_total_notional: float
    whale_trade_count: int
    liquidation_side: str
    liquidation_total_notional: float
    liquidation_count: int
    context_strength: float
    timestamp_ms: int

    exchange: str | None = None
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    raw_event: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.whale_side = normalize_side(self.whale_side)
        self.liquidation_side = normalize_side(self.liquidation_side)
        self.whale_total_notional = _safe_non_negative(self.whale_total_notional)
        self.whale_trade_count = max(0, int(self.whale_trade_count))
        self.liquidation_total_notional = _safe_non_negative(self.liquidation_total_notional)
        self.liquidation_count = max(0, int(self.liquidation_count))
        self.context_strength = _clamp_0_1(self.context_strength)
        self.timestamp_ms = int(self.timestamp_ms)
        self.metadata = _metadata_copy(self.metadata)

    @property
    def key(self) -> WhaleKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "key", _analytics_args)
        except Exception:
            pass
        return make_whale_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @classmethod
    def from_signal(
        cls,
        signal: "WhaleLiquidationContextSignal",
    ) -> "WhaleLiquidationContextRecord":
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "WhaleLiquidationContextRecord"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_signal", _analytics_args)
        except Exception:
            pass
        return cls(
            symbol=signal.symbol,
            whale_side=signal.whale_side,
            whale_total_notional=signal.whale_total_notional,
            whale_trade_count=signal.whale_trade_count,
            liquidation_side=signal.liquidation_side,
            liquidation_total_notional=signal.liquidation_total_notional,
            liquidation_count=signal.liquidation_count,
            context_strength=signal.context_strength,
            timestamp_ms=signal.timestamp_ms,
            exchange=signal.exchange,
            market_type=signal.market_type,
            timeframe=signal.timeframe,
            exchange_symbol=signal.exchange_symbol,
            metadata=dict(signal.metadata),
        )

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        payload = {
            "symbol": self.symbol,
            "whale_side": self.whale_side,
            "whale_total_notional": self.whale_total_notional,
            "whale_trade_count": self.whale_trade_count,
            "liquidation_side": self.liquidation_side,
            "liquidation_total_notional": self.liquidation_total_notional,
            "liquidation_count": self.liquidation_count,
            "context_strength": self.context_strength,
            "timestamp_ms": self.timestamp_ms,
            "metadata": dict(self.metadata),
        }
        _payload_with_scope(
            payload,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )
        if include_raw:
            payload["raw_event"] = self.raw_event
        return payload


# =============================================================================
# Scoped signal base
# =============================================================================


@dataclass(slots=True)
class WhaleScopedSignalModel(WhaleBaseSignalModel):
    symbol: str = ""
    exchange: str | None = None
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.metadata = _metadata_copy(self.metadata)

    @property
    def key(self) -> WhaleKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "key", _analytics_args)
        except Exception:
            pass
        return make_whale_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    def scoped_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "scoped_payload", _analytics_args)
        except Exception:
            pass
        payload = WhaleBaseSignalModel.to_payload(self)
        _payload_with_scope(
            payload,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )
        payload["metadata"] = dict(self.metadata)
        return payload


# =============================================================================
# Signal models
# =============================================================================


@dataclass(slots=True)
class LargeTradeSignal(WhaleScopedSignalModel):
    side: str = WhaleTradeSide.UNKNOWN.value
    price: float = 0.0
    quantity: float = 0.0
    notional: float = 0.0
    timestamp_ms: int = 0

    abs_threshold: float = 0.0
    mean_notional: float = 0.0
    std_notional: float = 0.0
    zscore: float = 0.0

    trigger_type: str = LargeTradeTriggerType.UNKNOWN.value
    trade_id: str | None = None

    detector_name: str = "LargeTradeDetector"
    event_type: str = WhaleEventType.LARGE_TRADE.value

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        WhaleScopedSignalModel.__post_init__(self)
        self.side = normalize_side(self.side)
        self.price = float(self.price)
        self.quantity = float(self.quantity)
        self.notional = _safe_non_negative(self.notional)
        self.timestamp_ms = int(self.timestamp_ms)
        self.abs_threshold = _safe_non_negative(self.abs_threshold)
        self.mean_notional = _safe_non_negative(self.mean_notional)
        self.std_notional = _safe_non_negative(self.std_notional)
        self.zscore = float(self.zscore)

    def to_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        payload = self.scoped_payload()
        payload.update(
            {
                "side": self.side,
                "price": self.price,
                "quantity": self.quantity,
                "notional": self.notional,
                "timestamp_ms": self.timestamp_ms,
                "abs_threshold": self.abs_threshold,
                "mean_notional": self.mean_notional,
                "std_notional": self.std_notional,
                "zscore": self.zscore,
                "trigger_type": self.trigger_type,
                "trade_id": self.trade_id,
            }
        )
        return payload

    @classmethod
    def from_trade(
        cls,
        *,
        trade: TradeRecord,
        abs_threshold: float,
        mean_notional: float,
        std_notional: float,
        zscore: float,
        absolute_triggered: bool,
        relative_triggered: bool,
    ) -> LargeTradeSignal:
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "LargeTradeSignal"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_trade", _analytics_args)
        except Exception:
            pass
        return cls(
            symbol=trade.symbol,
            exchange=trade.exchange,
            market_type=trade.market_type,
            timeframe=trade.timeframe,
            exchange_symbol=trade.exchange_symbol,
            side=trade.side,
            price=trade.price,
            quantity=trade.quantity,
            notional=trade.notional,
            timestamp_ms=trade.timestamp_ms,
            abs_threshold=abs_threshold,
            mean_notional=mean_notional,
            std_notional=std_notional,
            zscore=zscore,
            trigger_type=LargeTradeTriggerType.from_flags(
                absolute_triggered=absolute_triggered,
                relative_triggered=relative_triggered,
            ).value,
            trade_id=trade.trade_id,
            metadata=dict(trade.metadata),
        )


@dataclass(slots=True)
class WhaleActivitySignal(WhaleScopedSignalModel):
    side: str = WhaleTradeSide.UNKNOWN.value
    trade_count: int = 0
    total_notional: float = 0.0
    avg_notional: float = 0.0
    max_notional: float = 0.0
    window_sec: int = 0
    timestamp_ms: int = 0

    detector_name: str = "WhaleTracker"
    event_type: str = WhaleEventType.WHALE_ACTIVITY.value

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        WhaleScopedSignalModel.__post_init__(self)
        self.side = normalize_side(self.side)
        self.trade_count = max(0, int(self.trade_count))
        self.total_notional = _safe_non_negative(self.total_notional)
        self.avg_notional = _safe_non_negative(self.avg_notional)
        self.max_notional = _safe_non_negative(self.max_notional)
        self.window_sec = max(0, int(self.window_sec))
        self.timestamp_ms = int(self.timestamp_ms)

    def to_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        payload = self.scoped_payload()
        payload.update(
            {
                "side": self.side,
                "trade_count": self.trade_count,
                "total_notional": self.total_notional,
                "avg_notional": self.avg_notional,
                "max_notional": self.max_notional,
                "window_sec": self.window_sec,
                "timestamp_ms": self.timestamp_ms,
            }
        )
        return payload


@dataclass(slots=True)
class WhalePressureSignal(WhaleScopedSignalModel):
    dominant_side: str = WhaleTradeSide.UNKNOWN.value
    buy_trade_count: int = 0
    sell_trade_count: int = 0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    total_notional: float = 0.0
    imbalance_ratio: float = 0.0
    net_flow_notional: float = 0.0
    window_sec: int = 0
    timestamp_ms: int = 0

    detector_name: str = "WhaleTracker"
    event_type: str = WhaleEventType.WHALE_PRESSURE.value

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        WhaleScopedSignalModel.__post_init__(self)
        self.dominant_side = normalize_side(self.dominant_side)
        self.buy_trade_count = max(0, int(self.buy_trade_count))
        self.sell_trade_count = max(0, int(self.sell_trade_count))
        self.buy_notional = _safe_non_negative(self.buy_notional)
        self.sell_notional = _safe_non_negative(self.sell_notional)
        self.total_notional = _safe_non_negative(self.total_notional)
        self.imbalance_ratio = _clamp_0_1(self.imbalance_ratio)
        self.net_flow_notional = float(self.net_flow_notional)
        self.window_sec = max(0, int(self.window_sec))
        self.timestamp_ms = int(self.timestamp_ms)

    @property
    def pressure_type(self) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "pressure_type", _analytics_args)
        except Exception:
            pass
        return WhalePressureType.from_notional(
            buy_notional=self.buy_notional,
            sell_notional=self.sell_notional,
        ).value

    def to_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        payload = self.scoped_payload()
        payload.update(
            {
                "dominant_side": self.dominant_side,
                "buy_trade_count": self.buy_trade_count,
                "sell_trade_count": self.sell_trade_count,
                "buy_notional": self.buy_notional,
                "sell_notional": self.sell_notional,
                "total_notional": self.total_notional,
                "imbalance_ratio": self.imbalance_ratio,
                "net_flow_notional": self.net_flow_notional,
                "pressure_type": self.pressure_type,
                "window_sec": self.window_sec,
                "timestamp_ms": self.timestamp_ms,
            }
        )
        return payload


@dataclass(slots=True)
class WhaleLiquidationContextSignal(WhaleScopedSignalModel):
    whale_side: str = WhaleTradeSide.UNKNOWN.value
    whale_total_notional: float = 0.0
    whale_trade_count: int = 0
    liquidation_side: str = WhaleTradeSide.UNKNOWN.value
    liquidation_total_notional: float = 0.0
    liquidation_count: int = 0
    context_strength: float = 0.0
    timestamp_ms: int = 0

    detector_name: str = "WhaleTracker"
    event_type: str = WhaleEventType.WHALE_LIQUIDATION_CONTEXT.value

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        WhaleScopedSignalModel.__post_init__(self)
        self.whale_side = normalize_side(self.whale_side)
        self.liquidation_side = normalize_side(self.liquidation_side)
        self.whale_total_notional = _safe_non_negative(self.whale_total_notional)
        self.whale_trade_count = max(0, int(self.whale_trade_count))
        self.liquidation_total_notional = _safe_non_negative(self.liquidation_total_notional)
        self.liquidation_count = max(0, int(self.liquidation_count))
        self.context_strength = _clamp_0_1(self.context_strength)
        self.timestamp_ms = int(self.timestamp_ms)

    def to_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        payload = self.scoped_payload()
        payload.update(
            {
                "whale_side": self.whale_side,
                "whale_total_notional": self.whale_total_notional,
                "whale_trade_count": self.whale_trade_count,
                "liquidation_side": self.liquidation_side,
                "liquidation_total_notional": self.liquidation_total_notional,
                "liquidation_count": self.liquidation_count,
                "context_strength": self.context_strength,
                "timestamp_ms": self.timestamp_ms,
            }
        )
        return payload


@dataclass(slots=True)
class WhaleClusterSignal(WhaleScopedSignalModel):
    cluster_side: str = WhaleTradeSide.UNKNOWN.value
    cluster_score: float = 0.0
    persistence_score: float = 0.0
    directional_bias: float = 0.0
    continuation_probability: float = 0.0
    exhaustion_probability: float = 0.0

    activity_signal_count: int = 0
    pressure_signal_count: int = 0
    liquidation_context_count: int = 0

    total_activity_notional: float = 0.0
    total_pressure_notional: float = 0.0
    total_liquidation_context_notional: float = 0.0

    first_seen_ts_ms: int = 0
    last_seen_ts_ms: int = 0
    timestamp_ms: int = 0

    detector_name: str = "WhaleClusterAnalyzer"
    event_type: str = WhaleEventType.WHALE_CLUSTER.value

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        WhaleScopedSignalModel.__post_init__(self)
        self.cluster_side = normalize_side(self.cluster_side)
        self.cluster_score = _clamp_0_1(self.cluster_score)
        self.persistence_score = _clamp_0_1(self.persistence_score)
        self.directional_bias = _clamp_0_1(self.directional_bias)
        self.continuation_probability = _clamp_0_1(self.continuation_probability)
        self.exhaustion_probability = _clamp_0_1(self.exhaustion_probability)
        self.activity_signal_count = max(0, int(self.activity_signal_count))
        self.pressure_signal_count = max(0, int(self.pressure_signal_count))
        self.liquidation_context_count = max(0, int(self.liquidation_context_count))
        self.total_activity_notional = _safe_non_negative(self.total_activity_notional)
        self.total_pressure_notional = _safe_non_negative(self.total_pressure_notional)
        self.total_liquidation_context_notional = _safe_non_negative(
            self.total_liquidation_context_notional
        )

    @property
    def bias(self) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "bias", _analytics_args)
        except Exception:
            pass
        return WhaleBias.from_side(self.cluster_side).value

    @property
    def state(self) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "state", _analytics_args)
        except Exception:
            pass
        return WhaleClusterStateType.from_scores(
            cluster_score=self.cluster_score,
            exhaustion_probability=self.exhaustion_probability,
        ).value

    def to_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        payload = self.scoped_payload()
        payload.update(
            {
                "cluster_side": self.cluster_side,
                "cluster_score": self.cluster_score,
                "persistence_score": self.persistence_score,
                "directional_bias": self.directional_bias,
                "continuation_probability": self.continuation_probability,
                "exhaustion_probability": self.exhaustion_probability,
                "activity_signal_count": self.activity_signal_count,
                "pressure_signal_count": self.pressure_signal_count,
                "liquidation_context_count": self.liquidation_context_count,
                "total_activity_notional": self.total_activity_notional,
                "total_pressure_notional": self.total_pressure_notional,
                "total_liquidation_context_notional": self.total_liquidation_context_notional,
                "first_seen_ts_ms": self.first_seen_ts_ms,
                "last_seen_ts_ms": self.last_seen_ts_ms,
                "timestamp_ms": self.timestamp_ms,
                "bias": self.bias,
                "cluster_state": self.state,
            }
        )
        return payload


@dataclass(slots=True)
class WhaleClusterUpdateSignal(WhaleScopedSignalModel):
    cluster_side: str = WhaleTradeSide.UNKNOWN.value
    cluster_score: float = 0.0
    persistence_score: float = 0.0
    continuation_probability: float = 0.0
    exhaustion_probability: float = 0.0
    activity_signal_count: int = 0
    pressure_signal_count: int = 0
    liquidation_context_count: int = 0
    timestamp_ms: int = 0

    detector_name: str = "WhaleClusterAnalyzer"
    event_type: str = WhaleEventType.WHALE_CLUSTER_UPDATE.value

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        WhaleScopedSignalModel.__post_init__(self)
        self.cluster_side = normalize_side(self.cluster_side)
        self.cluster_score = _clamp_0_1(self.cluster_score)
        self.persistence_score = _clamp_0_1(self.persistence_score)
        self.continuation_probability = _clamp_0_1(self.continuation_probability)
        self.exhaustion_probability = _clamp_0_1(self.exhaustion_probability)

    def to_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        payload = self.scoped_payload()
        payload.update(
            {
                "cluster_side": self.cluster_side,
                "cluster_score": self.cluster_score,
                "persistence_score": self.persistence_score,
                "continuation_probability": self.continuation_probability,
                "exhaustion_probability": self.exhaustion_probability,
                "activity_signal_count": self.activity_signal_count,
                "pressure_signal_count": self.pressure_signal_count,
                "liquidation_context_count": self.liquidation_context_count,
                "timestamp_ms": self.timestamp_ms,
            }
        )
        return payload


@dataclass(slots=True)
class WhaleClusterExhaustionSignal(WhaleScopedSignalModel):
    cluster_side: str = WhaleTradeSide.UNKNOWN.value
    cluster_score: float = 0.0
    exhaustion_probability: float = 0.0
    reversal_risk: float = 0.0
    timestamp_ms: int = 0

    detector_name: str = "WhaleClusterAnalyzer"
    event_type: str = WhaleEventType.WHALE_CLUSTER_EXHAUSTION.value

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        WhaleScopedSignalModel.__post_init__(self)
        self.cluster_side = normalize_side(self.cluster_side)
        self.cluster_score = _clamp_0_1(self.cluster_score)
        self.exhaustion_probability = _clamp_0_1(self.exhaustion_probability)
        self.reversal_risk = _clamp_0_1(self.reversal_risk)

    def to_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        payload = self.scoped_payload()
        payload.update(
            {
                "cluster_side": self.cluster_side,
                "cluster_score": self.cluster_score,
                "exhaustion_probability": self.exhaustion_probability,
                "reversal_risk": self.reversal_risk,
                "timestamp_ms": self.timestamp_ms,
            }
        )
        return payload


# =============================================================================
# Internal rolling / scoped states
# =============================================================================


@dataclass(slots=True)
class SymbolStats:
    """
    Rolling-статистика для LargeTradeDetector.

    Назва залишена backward-compatible, але state тепер scoped через WhaleKey.
    """

    notionals: deque[float]

    exchange: str | None = None
    market_type: str = DEFAULT_MARKET_TYPE
    symbol: str = ""
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    trades_processed: int = 0
    signals_emitted: int = 0
    last_signal_ts_monotonic: float = 0.0
    last_update_ts_monotonic: float = field(default_factory=time.monotonic)

    running_sum: float = 0.0
    running_sum_sq: float = 0.0
    updates_since_recalibration: int = 0

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol or "UNKNOWN")
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

    @property
    def key(self) -> WhaleKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "key", _analytics_args)
        except Exception:
            pass
        return make_whale_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    def touch(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "touch", _analytics_args)
        except Exception:
            pass
        self.last_update_ts_monotonic = time.monotonic()

    @property
    def sample_size(self) -> int:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "sample_size", _analytics_args)
        except Exception:
            pass
        return len(self.notionals)

    def add(self, value: float, recalibration_interval: int) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "add", _analytics_args)
        except Exception:
            pass
        value = _safe_non_negative(value)

        if self.notionals.maxlen is not None and len(self.notionals) == self.notionals.maxlen:
            evicted = self.notionals[0]
            self.running_sum -= evicted
            self.running_sum_sq -= evicted * evicted

        self.notionals.append(value)
        self.running_sum += value
        self.running_sum_sq += value * value
        self.updates_since_recalibration += 1
        self.touch()

        if recalibration_interval > 0 and self.updates_since_recalibration >= recalibration_interval:
            self.recalibrate()

    def recalibrate(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "recalibrate", _analytics_args)
        except Exception:
            pass
        values = list(self.notionals)
        self.running_sum = math.fsum(values)
        self.running_sum_sq = math.fsum(value * value for value in values)
        self.updates_since_recalibration = 0

    def mean(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "mean", _analytics_args)
        except Exception:
            pass
        sample_size = len(self.notionals)
        if sample_size == 0:
            return 0.0
        return self.running_sum / sample_size

    def std(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "std", _analytics_args)
        except Exception:
            pass
        sample_size = len(self.notionals)
        if sample_size < 2:
            return 0.0

        mean_value = self.running_sum / sample_size
        numerator = self.running_sum_sq - sample_size * mean_value * mean_value
        numerator = max(numerator, 0.0)

        variance = numerator / (sample_size - 1)
        return math.sqrt(max(variance, 0.0))

    def to_dict(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        payload = {
            "sample_size": self.sample_size,
            "trades_processed": self.trades_processed,
            "signals_emitted": self.signals_emitted,
            "mean_notional": self.mean(),
            "std_notional": self.std(),
            "last_signal_ts_monotonic": self.last_signal_ts_monotonic,
            "last_update_ts_monotonic": self.last_update_ts_monotonic,
        }
        _payload_with_scope(
            payload,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )
        return payload


@dataclass(slots=True)
class SymbolTrackerState:
    """
    Rolling-state для WhaleTracker.

    Назва залишена backward-compatible, але state тепер scoped через WhaleKey.
    """

    large_trades: deque[WhaleTradeRecord]
    liquidations: deque[LiquidationRecord]

    exchange: str | None = None
    market_type: str = DEFAULT_MARKET_TYPE
    symbol: str = ""
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    total_large_trades_seen: int = 0
    total_liquidations_seen: int = 0

    whale_activity_signals_emitted: int = 0
    whale_pressure_signals_emitted: int = 0
    whale_liquidation_context_signals_emitted: int = 0

    last_whale_activity_signal_ts_monotonic: float = 0.0
    last_whale_pressure_signal_ts_monotonic: float = 0.0
    last_whale_liquidation_context_signal_ts_monotonic: float = 0.0

    last_update_ts_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol or "UNKNOWN")
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

    @property
    def key(self) -> WhaleKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "key", _analytics_args)
        except Exception:
            pass
        return make_whale_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    def touch(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "touch", _analytics_args)
        except Exception:
            pass
        self.last_update_ts_monotonic = time.monotonic()

    def to_dict(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        payload = {
            "large_trades_buffer_size": len(self.large_trades),
            "liquidations_buffer_size": len(self.liquidations),
            "total_large_trades_seen": self.total_large_trades_seen,
            "total_liquidations_seen": self.total_liquidations_seen,
            "whale_activity_signals_emitted": self.whale_activity_signals_emitted,
            "whale_pressure_signals_emitted": self.whale_pressure_signals_emitted,
            "whale_liquidation_context_signals_emitted": (
                self.whale_liquidation_context_signals_emitted
            ),
            "last_update_ts_monotonic": self.last_update_ts_monotonic,
        }
        _payload_with_scope(
            payload,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )
        return payload


@dataclass(slots=True)
class SymbolClusterState:
    """
    Rolling-state для WhaleClusterAnalyzer.

    Назва залишена backward-compatible, але state тепер scoped через WhaleKey.
    """

    activity_records: deque[WhaleActivityRecord]
    pressure_records: deque[WhalePressureRecord]
    liquidation_context_records: deque[WhaleLiquidationContextRecord]

    exchange: str | None = None
    market_type: str = DEFAULT_MARKET_TYPE
    symbol: str = ""
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    total_events_seen: int = 0
    total_clusters_emitted: int = 0
    total_cluster_updates_emitted: int = 0
    total_cluster_exhaustions_emitted: int = 0

    cluster_first_seen_ts_ms: int | None = None
    cluster_last_seen_ts_ms: int | None = None

    last_cluster_emit_ts_monotonic: float = 0.0
    last_cluster_update_emit_ts_monotonic: float = 0.0
    last_cluster_exhaustion_emit_ts_monotonic: float = 0.0

    last_update_ts_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol or "UNKNOWN")
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

    @property
    def key(self) -> WhaleKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "key", _analytics_args)
        except Exception:
            pass
        return make_whale_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    def touch(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "touch", _analytics_args)
        except Exception:
            pass
        self.last_update_ts_monotonic = time.monotonic()

    def to_dict(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        payload = {
            "activity_records_size": len(self.activity_records),
            "pressure_records_size": len(self.pressure_records),
            "liquidation_context_records_size": len(self.liquidation_context_records),
            "total_events_seen": self.total_events_seen,
            "total_clusters_emitted": self.total_clusters_emitted,
            "total_cluster_updates_emitted": self.total_cluster_updates_emitted,
            "total_cluster_exhaustions_emitted": self.total_cluster_exhaustions_emitted,
            "cluster_first_seen_ts_ms": self.cluster_first_seen_ts_ms,
            "cluster_last_seen_ts_ms": self.cluster_last_seen_ts_ms,
            "last_update_ts_monotonic": self.last_update_ts_monotonic,
        }
        _payload_with_scope(
            payload,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )
        return payload


# =============================================================================
# Aggregate result models
# =============================================================================


@dataclass(slots=True)
class WhaleTrackerResult:
    whale_activity_signal: WhaleActivitySignal | None = None
    whale_pressure_signal: WhalePressureSignal | None = None
    whale_liquidation_context_signal: WhaleLiquidationContextSignal | None = None

    @property
    def has_signals(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "has_signals", _analytics_args)
        except Exception:
            pass
        return any(signal is not None for signal in self.iter_signals())

    def iter_signals(self) -> tuple[WhaleBaseSignalModel, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "iter_signals", _analytics_args)
        except Exception:
            pass
        return tuple(
            signal
            for signal in (
                self.whale_activity_signal,
                self.whale_pressure_signal,
                self.whale_liquidation_context_signal,
            )
            if signal is not None
        )

    def to_dict(self) -> dict[str, dict[str, Any] | None]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        return {
            "whale_activity_signal": (
                self.whale_activity_signal.to_payload()
                if self.whale_activity_signal is not None
                else None
            ),
            "whale_pressure_signal": (
                self.whale_pressure_signal.to_payload()
                if self.whale_pressure_signal is not None
                else None
            ),
            "whale_liquidation_context_signal": (
                self.whale_liquidation_context_signal.to_payload()
                if self.whale_liquidation_context_signal is not None
                else None
            ),
        }


@dataclass(slots=True)
class WhaleClusterAnalysisResult:
    whale_cluster_signal: WhaleClusterSignal | None = None
    whale_cluster_update_signal: WhaleClusterUpdateSignal | None = None
    whale_cluster_exhaustion_signal: WhaleClusterExhaustionSignal | None = None

    @property
    def has_signals(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "has_signals", _analytics_args)
        except Exception:
            pass
        return any(signal is not None for signal in self.iter_signals())

    def iter_signals(self) -> tuple[WhaleBaseSignalModel, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "iter_signals", _analytics_args)
        except Exception:
            pass
        return tuple(
            signal
            for signal in (
                self.whale_cluster_signal,
                self.whale_cluster_update_signal,
                self.whale_cluster_exhaustion_signal,
            )
            if signal is not None
        )

    def to_dict(self) -> dict[str, dict[str, Any] | None]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        return {
            "whale_cluster_signal": (
                self.whale_cluster_signal.to_payload()
                if self.whale_cluster_signal is not None
                else None
            ),
            "whale_cluster_update_signal": (
                self.whale_cluster_update_signal.to_payload()
                if self.whale_cluster_update_signal is not None
                else None
            ),
            "whale_cluster_exhaustion_signal": (
                self.whale_cluster_exhaustion_signal.to_payload()
                if self.whale_cluster_exhaustion_signal is not None
                else None
            ),
        }


# =============================================================================
# Factory helpers
# =============================================================================


def make_symbol_stats(
    window_size: int,
    *,
    exchange: str | None = None,
    market_type: str = DEFAULT_MARKET_TYPE,
    symbol: str = "UNKNOWN",
    timeframe: str = DEFAULT_TIMEFRAME,
    exchange_symbol: str | None = None,
) -> SymbolStats:
    if window_size <= 1:
        raise ValueError("window_size must be > 1")

    return SymbolStats(
        notionals=deque(maxlen=window_size),
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
    )


def make_symbol_tracker_state(
    large_trade_window_size: int,
    liquidation_window_size: int,
    *,
    exchange: str | None = None,
    market_type: str = DEFAULT_MARKET_TYPE,
    symbol: str = "UNKNOWN",
    timeframe: str = DEFAULT_TIMEFRAME,
    exchange_symbol: str | None = None,
) -> SymbolTrackerState:
    if large_trade_window_size <= 0:
        raise ValueError("large_trade_window_size must be > 0")
    if liquidation_window_size <= 0:
        raise ValueError("liquidation_window_size must be > 0")

    return SymbolTrackerState(
        large_trades=deque(maxlen=large_trade_window_size),
        liquidations=deque(maxlen=liquidation_window_size),
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
    )


def make_symbol_cluster_state(
    activity_window_size: int,
    pressure_window_size: int,
    liquidation_context_window_size: int,
    *,
    exchange: str | None = None,
    market_type: str = DEFAULT_MARKET_TYPE,
    symbol: str = "UNKNOWN",
    timeframe: str = DEFAULT_TIMEFRAME,
    exchange_symbol: str | None = None,
) -> SymbolClusterState:
    if activity_window_size <= 0:
        raise ValueError("activity_window_size must be > 0")
    if pressure_window_size <= 0:
        raise ValueError("pressure_window_size must be > 0")
    if liquidation_context_window_size <= 0:
        raise ValueError("liquidation_context_window_size must be > 0")

    return SymbolClusterState(
        activity_records=deque(maxlen=activity_window_size),
        pressure_records=deque(maxlen=pressure_window_size),
        liquidation_context_records=deque(maxlen=liquidation_context_window_size),
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
    )


__all__ = [
    # constants / keys
    "DEFAULT_MARKET_TYPE",
    "DEFAULT_TIMEFRAME",
    "UNKNOWN_EXCHANGE",
    "WhaleKey",
    "make_whale_key",
    "whale_key_to_dict",
    "scope_payload",
    "normalize_exchange",
    "normalize_market_type",
    "normalize_symbol",
    "normalize_timeframe",
    "normalize_exchange_symbol",

    # base
    "WhaleBaseSignalModel",
    "WhaleBaseEventModel",
    "WhaleScopedSignalModel",

    # normalized records
    "TradeRecord",
    "WhaleTradeRecord",
    "LiquidationRecord",
    "WhaleActivityRecord",
    "WhalePressureRecord",
    "WhaleLiquidationContextRecord",

    # signals
    "LargeTradeSignal",
    "WhaleActivitySignal",
    "WhalePressureSignal",
    "WhaleLiquidationContextSignal",
    "WhaleClusterSignal",
    "WhaleClusterUpdateSignal",
    "WhaleClusterExhaustionSignal",

    # states
    "SymbolStats",
    "SymbolTrackerState",
    "SymbolClusterState",

    # results
    "WhaleTrackerResult",
    "WhaleClusterAnalysisResult",

    # factories
    "make_symbol_stats",
    "make_symbol_tracker_state",
    "make_symbol_cluster_state",

    # helpers
    "utc_now_ms",
]