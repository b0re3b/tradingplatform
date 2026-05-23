from __future__ import annotations
from core.logger import get_logger

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, NewType, TypeAlias

from analytics.price_action.enums import (
    FVGDirection,
    FVGEventType,
    FVGStatus,
    LevelStatus,
    LevelType,
    LiquidityEventType,
    LiquidityLevelStatus,
    LiquidityLevelType,
    MarketBias,
    SREventType,
    StructureEventType,
    StructureLayer,
    SwingType,
    TrendDirection,
    TrendEventType,
    TrendRegime,
)


SignedScore = NewType("SignedScore", float)   # expected range [-1.0, 1.0]
UnitScore = NewType("UnitScore", float)       # expected range [0.0, 1.0]

Metadata = dict[str, Any]

DEFAULT_EXCHANGE = "unknown"
DEFAULT_MARKET_TYPE = "usdm_futures"
DEFAULT_TIMEFRAME = "1m"

PriceActionKey: TypeAlias = tuple[str, str, str, str]
# exchange, market_type, symbol, timeframe


# ---------------------------------------------------------------------------
# Shared validation / normalization helpers
# ---------------------------------------------------------------------------

def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def ensure_non_negative(value: float, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def ensure_bounds(*, upper_bound: float, lower_bound: float) -> None:
    if lower_bound > upper_bound:
        raise ValueError("lower_bound cannot be greater than upper_bound")


def normalize_exchange(value: Any) -> str:
    normalized = str(value or DEFAULT_EXCHANGE).strip().lower()
    return normalized if normalized else DEFAULT_EXCHANGE


def normalize_market_type(value: Any) -> str:
    normalized = str(value or DEFAULT_MARKET_TYPE).strip().lower()
    return normalized if normalized else DEFAULT_MARKET_TYPE


def normalize_symbol(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized


def normalize_timeframe(value: Any) -> str:
    normalized = str(value or DEFAULT_TIMEFRAME).strip()
    return normalized if normalized else DEFAULT_TIMEFRAME


def normalize_exchange_symbol(
    value: Any,
    *,
    fallback_symbol: str,
) -> str:
    normalized = str(value or "").strip()
    return normalized if normalized else fallback_symbol


def make_price_action_key(
    *,
    exchange: Any = DEFAULT_EXCHANGE,
    market_type: Any = DEFAULT_MARKET_TYPE,
    symbol: Any,
    timeframe: Any = DEFAULT_TIMEFRAME,
) -> PriceActionKey:
    return (
        normalize_exchange(exchange),
        normalize_market_type(market_type),
        normalize_symbol(symbol),
        normalize_timeframe(timeframe),
    )


def price_action_key_to_dict(key: PriceActionKey) -> dict[str, str]:
    exchange, market_type, symbol, timeframe = key
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def serialize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if is_dataclass(value):
        return {
            key: serialize_value(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            str(key): serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [serialize_value(item) for item in value]

    if isinstance(value, tuple):
        return [serialize_value(item) for item in value]

    if isinstance(value, set):
        return sorted(serialize_value(item) for item in value)

    return value


def model_to_dict(model: Any) -> dict[str, Any]:
    if not is_dataclass(model):
        raise TypeError(f"Expected dataclass instance, got: {type(model)!r}")

    return {
        key: serialize_value(value)
        for key, value in asdict(model).items()
    }


@dataclass(slots=True)
class PriceActionScope:
    """
    Shared futures market scope.

    Scope:
        exchange + market_type + symbol + timeframe

    Futures examples:
        ("binance", "usdm_futures", "BTCUSDT", "1m")
        ("bybit", "linear", "BTCUSDT", "1m")
        ("okx", "swap", "BTCUSDT", "1m")
        ("mexc", "usdm_futures", "BTCUSDT", "1m")
    """

    symbol: str
    timeframe: str = DEFAULT_TIMEFRAME
    exchange: str = DEFAULT_EXCHANGE
    market_type: str = DEFAULT_MARKET_TYPE
    exchange_symbol: str | None = None

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
        self.symbol = normalize_symbol(self.symbol)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

    @property
    def key(self) -> PriceActionKey:
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
        return make_price_action_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def scope_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "scope_payload", _analytics_args)
        except Exception:
            pass
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "exchange_symbol": self.exchange_symbol,
            "timeframe": self.timeframe,
            "key": list(self.key),
        }


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Candle(PriceActionScope):
    timestamp: datetime = field(default_factory=utc_now)
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    quote_volume: float | None = None
    trades_count: int | None = None
    is_closed: bool = True
    index: int = 0
    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        self.timestamp = normalize_datetime(self.timestamp) or utc_now()

        self.open = float(self.open)
        self.high = float(self.high)
        self.low = float(self.low)
        self.close = float(self.close)
        self.volume = float(self.volume)

        if self.quote_volume is not None:
            self.quote_volume = float(self.quote_volume)
            ensure_non_negative(self.quote_volume, "quote_volume")

        if self.trades_count is not None:
            self.trades_count = int(self.trades_count)
            if self.trades_count < 0:
                raise ValueError("trades_count must be >= 0")

        self.index = int(self.index)
        self.is_closed = bool(self.is_closed)
        self.metadata = dict(self.metadata or {})

        if self.low > self.high:
            raise ValueError("Invalid candle: low cannot be greater than high")
        if min(self.open, self.high, self.low, self.close) < 0:
            raise ValueError("Invalid candle: OHLC cannot be negative")
        if self.high < max(self.open, self.close):
            raise ValueError("Invalid candle: high must be >= max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError("Invalid candle: low must be <= min(open, close)")
        if self.volume < 0:
            raise ValueError("Invalid candle: volume cannot be negative")
        if self.index < 0:
            raise ValueError("Invalid candle: index cannot be negative")

    @property
    def range_size(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "range_size", _analytics_args)
        except Exception:
            pass
        return max(self.high - self.low, 0.0)

    @property
    def body_high(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "body_high", _analytics_args)
        except Exception:
            pass
        return max(self.open, self.close)

    @property
    def body_low(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "body_low", _analytics_args)
        except Exception:
            pass
        return min(self.open, self.close)

    @property
    def body_size(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "body_size", _analytics_args)
        except Exception:
            pass
        return abs(self.close - self.open)

    @property
    def body_ratio(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "body_ratio", _analytics_args)
        except Exception:
            pass
        if self.range_size <= 0:
            return 0.0
        return self.body_size / self.range_size

    @property
    def upper_wick(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "upper_wick", _analytics_args)
        except Exception:
            pass
        return max(self.high - self.body_high, 0.0)

    @property
    def lower_wick(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "lower_wick", _analytics_args)
        except Exception:
            pass
        return max(self.body_low - self.low, 0.0)

    @property
    def upper_wick_ratio(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "upper_wick_ratio", _analytics_args)
        except Exception:
            pass
        if self.range_size <= 0:
            return 0.0
        return self.upper_wick / self.range_size

    @property
    def lower_wick_ratio(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "lower_wick_ratio", _analytics_args)
        except Exception:
            pass
        if self.range_size <= 0:
            return 0.0
        return self.lower_wick / self.range_size

    @property
    def is_bullish(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_bullish", _analytics_args)
        except Exception:
            pass
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_bearish", _analytics_args)
        except Exception:
            pass
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_doji", _analytics_args)
        except Exception:
            pass
        return self.close == self.open

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
        payload = model_to_dict(self)
        payload["key"] = list(self.key)
        return payload


# ---------------------------------------------------------------------------
# Market structure models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SwingPoint(PriceActionScope):
    swing_id: str = ""
    timestamp: datetime = field(default_factory=utc_now)
    price: float = 0.0
    swing_type: SwingType = SwingType.HIGH
    layer: StructureLayer = StructureLayer.INTERNAL
    index: int = 0
    candle_open: float = 0.0
    candle_high: float = 0.0
    candle_low: float = 0.0
    candle_close: float = 0.0
    strength: float = 0.0
    is_confirmed: bool = True
    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        self.timestamp = normalize_datetime(self.timestamp) or utc_now()

        if not self.swing_id:
            raise ValueError("swing_id must not be empty")

        self.price = float(self.price)
        self.index = int(self.index)
        self.candle_open = float(self.candle_open)
        self.candle_high = float(self.candle_high)
        self.candle_low = float(self.candle_low)
        self.candle_close = float(self.candle_close)
        self.strength = clamp_unit(self.strength)
        self.is_confirmed = bool(self.is_confirmed)
        self.metadata = dict(self.metadata or {})

        ensure_non_negative(self.price, "price")

        if self.index < 0:
            raise ValueError("index must be >= 0")


@dataclass(slots=True)
class StructureEvent(PriceActionScope):
    event_id: str = ""
    event_type: StructureEventType = StructureEventType.SWING_HIGH
    timestamp: datetime = field(default_factory=utc_now)
    price: float = 0.0
    layer: StructureLayer = StructureLayer.INTERNAL
    direction: MarketBias | None = None
    swing_id: str | None = None
    reference_price: float | None = None
    reference_swing_id: str | None = None
    confidence: float = 0.0
    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        self.timestamp = normalize_datetime(self.timestamp) or utc_now()

        if not self.event_id:
            raise ValueError("event_id must not be empty")

        self.price = float(self.price)
        self.confidence = clamp_unit(self.confidence)
        self.metadata = dict(self.metadata or {})

        ensure_non_negative(self.price, "price")

        if self.reference_price is not None:
            self.reference_price = float(self.reference_price)
            ensure_non_negative(self.reference_price, "reference_price")


@dataclass(slots=True)
class StructureLayerState:
    layer: StructureLayer
    bias: MarketBias = MarketBias.UNKNOWN
    confidence: float = 0.0
    trend_strength: float = 0.0
    in_breakout: bool = False

    last_swing_high: SwingPoint | None = None
    previous_swing_high: SwingPoint | None = None
    last_swing_low: SwingPoint | None = None
    previous_swing_low: SwingPoint | None = None

    last_hh: StructureEvent | None = None
    last_hl: StructureEvent | None = None
    last_lh: StructureEvent | None = None
    last_ll: StructureEvent | None = None
    last_bos: StructureEvent | None = None
    last_choch: StructureEvent | None = None
    last_mss: StructureEvent | None = None

    swing_count: int = 0
    event_count: int = 0
    sequence: list[str] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)

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
        self.confidence = clamp_unit(self.confidence)
        self.trend_strength = clamp_unit(self.trend_strength)
        self.swing_count = int(self.swing_count)
        self.event_count = int(self.event_count)
        self.sequence = list(self.sequence or [])
        self.metadata = dict(self.metadata or {})

        if self.swing_count < 0:
            raise ValueError("swing_count must be >= 0")
        if self.event_count < 0:
            raise ValueError("event_count must be >= 0")


@dataclass(slots=True)
class MultiTimeframeAlignment:
    higher_timeframe: str | None = None
    higher_timeframe_bias: MarketBias = MarketBias.UNKNOWN
    higher_timeframe_confidence: float = 0.0

    internal_bias_aligned: bool = False
    external_bias_aligned: bool = False
    internal_with_external_aligned: bool = False

    alignment_score: float = 0.0
    last_updated: datetime | None = None
    metadata: Metadata = field(default_factory=dict)

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
        self.higher_timeframe = (
            str(self.higher_timeframe).strip()
            if self.higher_timeframe is not None
            else None
        )
        self.higher_timeframe_confidence = clamp_unit(
            self.higher_timeframe_confidence
        )
        self.alignment_score = clamp_unit(self.alignment_score)
        self.last_updated = normalize_datetime(self.last_updated)
        self.metadata = dict(self.metadata or {})


@dataclass(slots=True)
class MarketStructureState(PriceActionScope):
    last_price: float | None = None
    last_update: datetime | None = None

    internal: StructureLayerState = field(
        default_factory=lambda: StructureLayerState(layer=StructureLayer.INTERNAL)
    )
    external: StructureLayerState = field(
        default_factory=lambda: StructureLayerState(layer=StructureLayer.EXTERNAL)
    )
    mtf_alignment: MultiTimeframeAlignment = field(default_factory=MultiTimeframeAlignment)

    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        self.last_update = normalize_datetime(self.last_update)
        self.metadata = dict(self.metadata or {})

        if self.last_price is not None:
            self.last_price = float(self.last_price)
            ensure_non_negative(self.last_price, "last_price")

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
        payload = model_to_dict(self)
        payload["key"] = list(self.key)
        return payload


# ---------------------------------------------------------------------------
# Support / Resistance models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SupportResistanceLevel(PriceActionScope):
    level_id: str = ""
    layer: StructureLayer = StructureLayer.INTERNAL
    level_type: LevelType = LevelType.SUPPORT
    price: float = 0.0
    upper_bound: float = 0.0
    lower_bound: float = 0.0
    strength: float = 0.0
    status: LevelStatus = LevelStatus.ACTIVE

    created_at: datetime | None = None
    updated_at: datetime | None = None
    broken_at: datetime | None = None
    flipped_at: datetime | None = None
    last_tested_at: datetime | None = None
    last_rejected_at: datetime | None = None
    last_broken_at: datetime | None = None
    last_retested_at: datetime | None = None

    touch_count: int = 0
    rejection_count: int = 0
    break_count: int = 0
    retest_count: int = 0
    source_count: int = 0

    source_swing_ids: list[str] = field(default_factory=list)
    source_prices: list[float] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        if not self.level_id:
            raise ValueError("level_id must not be empty")

        self.price = float(self.price)
        self.upper_bound = float(self.upper_bound)
        self.lower_bound = float(self.lower_bound)
        self.strength = clamp_unit(self.strength)

        ensure_non_negative(self.price, "price")
        ensure_bounds(upper_bound=self.upper_bound, lower_bound=self.lower_bound)

        self.created_at = normalize_datetime(self.created_at)
        self.updated_at = normalize_datetime(self.updated_at)
        self.broken_at = normalize_datetime(self.broken_at)
        self.flipped_at = normalize_datetime(self.flipped_at)
        self.last_tested_at = normalize_datetime(self.last_tested_at)
        self.last_rejected_at = normalize_datetime(self.last_rejected_at)
        self.last_broken_at = normalize_datetime(self.last_broken_at)
        self.last_retested_at = normalize_datetime(self.last_retested_at)

        self.source_swing_ids = list(self.source_swing_ids or [])
        self.source_prices = [float(price) for price in self.source_prices or []]
        self.metadata = dict(self.metadata or {})

        for field_name in (
            "touch_count",
            "rejection_count",
            "break_count",
            "retest_count",
            "source_count",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0")
            setattr(self, field_name, value)


@dataclass(slots=True)
class SupportResistanceEvent(PriceActionScope):
    event_id: str = ""
    event_type: SREventType = SREventType.LEVEL_CREATED
    timestamp: datetime = field(default_factory=utc_now)
    layer: StructureLayer = StructureLayer.INTERNAL
    level_id: str = ""
    level_type: LevelType = LevelType.SUPPORT
    price: float = 0.0
    confidence: float = 0.0
    reference_price: float | None = None
    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        self.timestamp = normalize_datetime(self.timestamp) or utc_now()

        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not self.level_id:
            raise ValueError("level_id must not be empty")

        self.price = float(self.price)
        self.confidence = clamp_unit(self.confidence)
        self.metadata = dict(self.metadata or {})

        ensure_non_negative(self.price, "price")

        if self.reference_price is not None:
            self.reference_price = float(self.reference_price)
            ensure_non_negative(self.reference_price, "reference_price")


@dataclass(slots=True)
class LayerSRState:
    layer: StructureLayer
    total_levels: int = 0
    active_supports: int = 0
    active_resistances: int = 0
    active_flip_supports: int = 0
    active_flip_resistances: int = 0

    strongest_support: SupportResistanceLevel | None = None
    strongest_resistance: SupportResistanceLevel | None = None
    nearest_support: SupportResistanceLevel | None = None
    nearest_resistance: SupportResistanceLevel | None = None

    last_event: SupportResistanceEvent | None = None
    metadata: Metadata = field(default_factory=dict)

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
        self.metadata = dict(self.metadata or {})

        for field_name in (
            "total_levels",
            "active_supports",
            "active_resistances",
            "active_flip_supports",
            "active_flip_resistances",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0")
            setattr(self, field_name, value)


@dataclass(slots=True)
class SupportResistanceState(PriceActionScope):
    last_price: float | None = None
    last_update: datetime | None = None
    internal: LayerSRState = field(
        default_factory=lambda: LayerSRState(layer=StructureLayer.INTERNAL)
    )
    external: LayerSRState = field(
        default_factory=lambda: LayerSRState(layer=StructureLayer.EXTERNAL)
    )
    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        self.last_update = normalize_datetime(self.last_update)
        self.metadata = dict(self.metadata or {})

        if self.last_price is not None:
            self.last_price = float(self.last_price)
            ensure_non_negative(self.last_price, "last_price")

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
        payload = model_to_dict(self)
        payload["key"] = list(self.key)
        return payload


# ---------------------------------------------------------------------------
# Fair Value Gap models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FairValueGap(PriceActionScope):
    gap_id: str = ""
    layer: StructureLayer = StructureLayer.INTERNAL
    direction: FVGDirection = FVGDirection.BULLISH

    upper_bound: float = 0.0
    lower_bound: float = 0.0
    mid_price: float = 0.0
    size: float = 0.0
    size_pct: float = 0.0
    strength: float = 0.0

    status: FVGStatus = FVGStatus.ACTIVE
    fill_percentage: float = 0.0
    touch_count: int = 0
    retest_count: int = 0

    created_at: datetime | None = None
    updated_at: datetime | None = None
    first_touch_at: datetime | None = None
    filled_at: datetime | None = None
    respected_at: datetime | None = None
    invalidated_at: datetime | None = None

    created_index: int | None = None
    last_touch_index: int | None = None
    last_fill_index: int | None = None

    source_candle_indices: list[int] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        if not self.gap_id:
            raise ValueError("gap_id must not be empty")

        self.upper_bound = float(self.upper_bound)
        self.lower_bound = float(self.lower_bound)
        self.mid_price = float(self.mid_price)
        self.size = float(self.size)
        self.size_pct = float(self.size_pct)
        self.strength = clamp_unit(self.strength)
        self.fill_percentage = clamp_unit(self.fill_percentage)

        ensure_bounds(upper_bound=self.upper_bound, lower_bound=self.lower_bound)
        ensure_non_negative(self.mid_price, "mid_price")
        ensure_non_negative(self.size, "size")
        ensure_non_negative(self.size_pct, "size_pct")

        self.created_at = normalize_datetime(self.created_at)
        self.updated_at = normalize_datetime(self.updated_at)
        self.first_touch_at = normalize_datetime(self.first_touch_at)
        self.filled_at = normalize_datetime(self.filled_at)
        self.respected_at = normalize_datetime(self.respected_at)
        self.invalidated_at = normalize_datetime(self.invalidated_at)

        self.source_candle_indices = [
            int(index)
            for index in self.source_candle_indices or []
        ]
        self.metadata = dict(self.metadata or {})

        for field_name in ("touch_count", "retest_count"):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0")
            setattr(self, field_name, value)


@dataclass(slots=True)
class FVGEvent(PriceActionScope):
    event_id: str = ""
    event_type: FVGEventType = FVGEventType.FVG_CREATED
    timestamp: datetime = field(default_factory=utc_now)
    layer: StructureLayer = StructureLayer.INTERNAL
    gap_id: str = ""
    direction: FVGDirection = FVGDirection.BULLISH
    upper_bound: float = 0.0
    lower_bound: float = 0.0
    fill_percentage: float = 0.0
    confidence: float = 0.0
    reference_price: float | None = None
    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        self.timestamp = normalize_datetime(self.timestamp) or utc_now()

        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not self.gap_id:
            raise ValueError("gap_id must not be empty")

        self.upper_bound = float(self.upper_bound)
        self.lower_bound = float(self.lower_bound)
        self.fill_percentage = clamp_unit(self.fill_percentage)
        self.confidence = clamp_unit(self.confidence)
        self.metadata = dict(self.metadata or {})

        ensure_bounds(upper_bound=self.upper_bound, lower_bound=self.lower_bound)

        if self.reference_price is not None:
            self.reference_price = float(self.reference_price)
            ensure_non_negative(self.reference_price, "reference_price")


@dataclass(slots=True)
class LayerFVGState:
    layer: StructureLayer
    total_gaps: int = 0
    active_gaps: int = 0
    partially_filled_gaps: int = 0
    filled_gaps: int = 0
    respected_gaps: int = 0
    invalidated_gaps: int = 0

    nearest_bullish_gap: FairValueGap | None = None
    nearest_bearish_gap: FairValueGap | None = None
    strongest_bullish_gap: FairValueGap | None = None
    strongest_bearish_gap: FairValueGap | None = None

    recent_fill_activity: float = 0.0
    last_event: FVGEvent | None = None
    metadata: Metadata = field(default_factory=dict)

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
        self.recent_fill_activity = clamp_unit(self.recent_fill_activity)
        self.metadata = dict(self.metadata or {})

        for field_name in (
            "total_gaps",
            "active_gaps",
            "partially_filled_gaps",
            "filled_gaps",
            "respected_gaps",
            "invalidated_gaps",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0")
            setattr(self, field_name, value)


@dataclass(slots=True)
class FairValueGapState(PriceActionScope):
    last_price: float | None = None
    last_update: datetime | None = None
    internal: LayerFVGState = field(
        default_factory=lambda: LayerFVGState(layer=StructureLayer.INTERNAL)
    )
    external: LayerFVGState = field(
        default_factory=lambda: LayerFVGState(layer=StructureLayer.EXTERNAL)
    )
    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        self.last_update = normalize_datetime(self.last_update)
        self.metadata = dict(self.metadata or {})

        if self.last_price is not None:
            self.last_price = float(self.last_price)
            ensure_non_negative(self.last_price, "last_price")

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
        payload = model_to_dict(self)
        payload["key"] = list(self.key)
        return payload


# ---------------------------------------------------------------------------
# Liquidity models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LiquidityLevel(PriceActionScope):
    level_id: str = ""
    layer: StructureLayer = StructureLayer.INTERNAL
    level_type: LiquidityLevelType = LiquidityLevelType.BUY_SIDE_LIQUIDITY
    price: float = 0.0
    upper_bound: float = 0.0
    lower_bound: float = 0.0
    strength: float = 0.0

    status: LiquidityLevelStatus = LiquidityLevelStatus.ACTIVE
    touch_count: int = 0
    sweep_count: int = 0
    reclaim_count: int = 0

    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_touched_at: datetime | None = None
    swept_at: datetime | None = None
    reclaimed_at: datetime | None = None
    invalidated_at: datetime | None = None

    last_sweep_side: str | None = None
    last_sweep_price: float | None = None
    last_sweep_index: int | None = None

    source_swing_ids: list[str] = field(default_factory=list)
    source_prices: list[float] = field(default_factory=list)
    source_count: int = 0

    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        if not self.level_id:
            raise ValueError("level_id must not be empty")

        self.price = float(self.price)
        self.upper_bound = float(self.upper_bound)
        self.lower_bound = float(self.lower_bound)
        self.strength = clamp_unit(self.strength)

        ensure_non_negative(self.price, "price")
        ensure_bounds(upper_bound=self.upper_bound, lower_bound=self.lower_bound)

        self.created_at = normalize_datetime(self.created_at)
        self.updated_at = normalize_datetime(self.updated_at)
        self.last_touched_at = normalize_datetime(self.last_touched_at)
        self.swept_at = normalize_datetime(self.swept_at)
        self.reclaimed_at = normalize_datetime(self.reclaimed_at)
        self.invalidated_at = normalize_datetime(self.invalidated_at)

        if self.last_sweep_price is not None:
            self.last_sweep_price = float(self.last_sweep_price)
            ensure_non_negative(self.last_sweep_price, "last_sweep_price")

        self.source_swing_ids = list(self.source_swing_ids or [])
        self.source_prices = [float(price) for price in self.source_prices or []]
        self.metadata = dict(self.metadata or {})

        for field_name in (
            "touch_count",
            "sweep_count",
            "reclaim_count",
            "source_count",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0")
            setattr(self, field_name, value)


@dataclass(slots=True)
class LiquidityEvent(PriceActionScope):
    event_id: str = ""
    event_type: LiquidityEventType = LiquidityEventType.LEVEL_CREATED
    timestamp: datetime = field(default_factory=utc_now)
    layer: StructureLayer = StructureLayer.INTERNAL
    level_id: str = ""
    level_type: LiquidityLevelType = LiquidityLevelType.BUY_SIDE_LIQUIDITY
    price: float = 0.0
    confidence: float = 0.0
    reference_price: float | None = None
    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        self.timestamp = normalize_datetime(self.timestamp) or utc_now()

        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not self.level_id:
            raise ValueError("level_id must not be empty")

        self.price = float(self.price)
        self.confidence = clamp_unit(self.confidence)
        self.metadata = dict(self.metadata or {})

        ensure_non_negative(self.price, "price")

        if self.reference_price is not None:
            self.reference_price = float(self.reference_price)
            ensure_non_negative(self.reference_price, "reference_price")


@dataclass(slots=True)
class LayerLiquidityState:
    layer: StructureLayer
    total_levels: int = 0
    active_levels: int = 0
    swept_levels: int = 0
    reclaimed_levels: int = 0
    invalidated_levels: int = 0

    nearest_buy_side: LiquidityLevel | None = None
    nearest_sell_side: LiquidityLevel | None = None
    strongest_buy_side: LiquidityLevel | None = None
    strongest_sell_side: LiquidityLevel | None = None

    recent_sweep_count: int = 0
    last_event: LiquidityEvent | None = None
    metadata: Metadata = field(default_factory=dict)

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
        self.metadata = dict(self.metadata or {})

        for field_name in (
            "total_levels",
            "active_levels",
            "swept_levels",
            "reclaimed_levels",
            "invalidated_levels",
            "recent_sweep_count",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0")
            setattr(self, field_name, value)


@dataclass(slots=True)
class LiquidityState(PriceActionScope):
    last_price: float | None = None
    last_update: datetime | None = None
    internal: LayerLiquidityState = field(
        default_factory=lambda: LayerLiquidityState(layer=StructureLayer.INTERNAL)
    )
    external: LayerLiquidityState = field(
        default_factory=lambda: LayerLiquidityState(layer=StructureLayer.EXTERNAL)
    )
    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        self.last_update = normalize_datetime(self.last_update)
        self.metadata = dict(self.metadata or {})

        if self.last_price is not None:
            self.last_price = float(self.last_price)
            ensure_non_negative(self.last_price, "last_price")

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
        payload = model_to_dict(self)
        payload["key"] = list(self.key)
        return payload


# ---------------------------------------------------------------------------
# Trend models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrendSignal(PriceActionScope):
    signal_id: str = ""
    timestamp: datetime = field(default_factory=utc_now)
    layer: StructureLayer = StructureLayer.INTERNAL
    event_type: TrendEventType = TrendEventType.TREND_STARTED
    direction: TrendDirection = TrendDirection.UNKNOWN
    strength: float = 0.0
    confidence: float = 0.0
    regime: TrendRegime = TrendRegime.UNKNOWN
    price: float | None = None
    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        self.timestamp = normalize_datetime(self.timestamp) or utc_now()

        if not self.signal_id:
            raise ValueError("signal_id must not be empty")

        self.strength = clamp_unit(self.strength)
        self.confidence = clamp_unit(self.confidence)
        self.metadata = dict(self.metadata or {})

        if self.price is not None:
            self.price = float(self.price)
            ensure_non_negative(self.price, "price")


@dataclass(slots=True)
class TrendLayerState:
    layer: StructureLayer
    direction: TrendDirection = TrendDirection.UNKNOWN
    regime: TrendRegime = TrendRegime.UNKNOWN

    strength: UnitScore = UnitScore(0.0)
    confidence: UnitScore = UnitScore(0.0)

    momentum_direction_score: SignedScore = SignedScore(0.0)
    slope_direction_score: SignedScore = SignedScore(0.0)

    structure_score: UnitScore = UnitScore(0.0)
    continuation_probability: UnitScore = UnitScore(0.0)
    reversal_risk: UnitScore = UnitScore(0.0)
    exhaustion_score: UnitScore = UnitScore(0.0)
    pullback_depth: UnitScore = UnitScore(0.0)
    consolidation_score: UnitScore = UnitScore(0.0)

    is_accelerating: bool = False
    is_exhausted: bool = False
    in_pullback: bool = False
    is_aligned_with_structure: bool = False

    last_signal: TrendSignal | None = None
    metadata: Metadata = field(default_factory=dict)

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
        self.strength = UnitScore(clamp_unit(float(self.strength)))
        self.confidence = UnitScore(clamp_unit(float(self.confidence)))

        self.momentum_direction_score = SignedScore(
            clamp_signed(float(self.momentum_direction_score))
        )
        self.slope_direction_score = SignedScore(
            clamp_signed(float(self.slope_direction_score))
        )

        self.structure_score = UnitScore(clamp_unit(float(self.structure_score)))
        self.continuation_probability = UnitScore(
            clamp_unit(float(self.continuation_probability))
        )
        self.reversal_risk = UnitScore(clamp_unit(float(self.reversal_risk)))
        self.exhaustion_score = UnitScore(clamp_unit(float(self.exhaustion_score)))
        self.pullback_depth = UnitScore(clamp_unit(float(self.pullback_depth)))
        self.consolidation_score = UnitScore(clamp_unit(float(self.consolidation_score)))

        self.metadata = dict(self.metadata or {})


@dataclass(slots=True)
class TrendState(PriceActionScope):
    last_price: float | None = None
    last_update: datetime | None = None

    internal: TrendLayerState = field(
        default_factory=lambda: TrendLayerState(layer=StructureLayer.INTERNAL)
    )
    external: TrendLayerState = field(
        default_factory=lambda: TrendLayerState(layer=StructureLayer.EXTERNAL)
    )

    internal_external_alignment: UnitScore = UnitScore(0.0)
    higher_timeframe_alignment: UnitScore = UnitScore(0.0)
    overall_trend_score: UnitScore = UnitScore(0.0)

    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        self.last_update = normalize_datetime(self.last_update)
        self.metadata = dict(self.metadata or {})

        if self.last_price is not None:
            self.last_price = float(self.last_price)
            ensure_non_negative(self.last_price, "last_price")

        self.internal_external_alignment = UnitScore(
            clamp_unit(float(self.internal_external_alignment))
        )
        self.higher_timeframe_alignment = UnitScore(
            clamp_unit(float(self.higher_timeframe_alignment))
        )
        self.overall_trend_score = UnitScore(
            clamp_unit(float(self.overall_trend_score))
        )

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
        payload = model_to_dict(self)
        payload["key"] = list(self.key)
        return payload


# ---------------------------------------------------------------------------
# Facade / aggregate models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PriceActionCompositeState(PriceActionScope):
    """
    Aggregated state for PriceActionAnalyzer facade.

    This is only a typed state container.
    It does not orchestrate EventBus, Scheduler, or calculations.
    """

    last_price: float | None = None
    last_update: datetime | None = None

    market_structure: MarketStructureState | None = None
    support_resistance: SupportResistanceState | None = None
    fair_value_gap: FairValueGapState | None = None
    liquidity: LiquidityState | None = None
    trend: TrendState | None = None

    metadata: Metadata = field(default_factory=dict)

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
        PriceActionScope.__post_init__(self)

        self.last_update = normalize_datetime(self.last_update)
        self.metadata = dict(self.metadata or {})

        if self.last_price is not None:
            self.last_price = float(self.last_price)
            ensure_non_negative(self.last_price, "last_price")

        self._validate_child_scope("market_structure", self.market_structure)
        self._validate_child_scope("support_resistance", self.support_resistance)
        self._validate_child_scope("fair_value_gap", self.fair_value_gap)
        self._validate_child_scope("liquidity", self.liquidity)
        self._validate_child_scope("trend", self.trend)

    def _validate_child_scope(
        self,
        name: str,
        child: PriceActionScope | None,
    ) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_child_scope", _analytics_args)
        except Exception:
            pass
        if child is None:
            return

        if child.key != self.key:
            raise ValueError(
                f"{name} scope does not match composite scope: "
                f"child={child.key}, composite={self.key}"
            )

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
        payload = model_to_dict(self)
        payload["key"] = list(self.key)
        return payload


__all__ = [
    "SignedScore",
    "UnitScore",
    "Metadata",
    "DEFAULT_EXCHANGE",
    "DEFAULT_MARKET_TYPE",
    "DEFAULT_TIMEFRAME",
    "PriceActionKey",
    "PriceActionScope",
    "clamp_unit",
    "clamp_signed",
    "ensure_non_negative",
    "ensure_bounds",
    "normalize_exchange",
    "normalize_market_type",
    "normalize_symbol",
    "normalize_timeframe",
    "normalize_exchange_symbol",
    "make_price_action_key",
    "price_action_key_to_dict",
    "utc_now",
    "normalize_datetime",
    "serialize_value",
    "model_to_dict",
    "Candle",
    "SwingPoint",
    "StructureEvent",
    "StructureLayerState",
    "MultiTimeframeAlignment",
    "MarketStructureState",
    "SupportResistanceLevel",
    "SupportResistanceEvent",
    "LayerSRState",
    "SupportResistanceState",
    "FairValueGap",
    "FVGEvent",
    "LayerFVGState",
    "FairValueGapState",
    "LiquidityLevel",
    "LiquidityEvent",
    "LayerLiquidityState",
    "LiquidityState",
    "TrendSignal",
    "TrendLayerState",
    "TrendState",
    "PriceActionCompositeState",
]