from __future__ import annotations
from core.logger import get_logger

import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypeAlias

from .enums import (
    CandidateStatus,
    DetectorDecision,
    LiquidityEventType,
    OrderbookWallState,
    ScoreComponent,
    SpoofingComponent,
    SpoofingPattern,
    SpoofingSeverity,
    SpoofingSide,
    SpoofingStatus,
    SpoofingType,
    TradeSide,
)


DEFAULT_MARKET_TYPE = "perpetual"
DEFAULT_TIMEFRAME = "realtime"

SpoofingKey: TypeAlias = tuple[str, str, str, str]
# exchange, market_type, symbol, timeframe


# =============================================================================
# Shared helpers
# =============================================================================


def utc_now() -> datetime:
    """
    Єдина точка для UTC timestamps у spoofing-пакеті.
    """
    return datetime.now(timezone.utc)


def unix_ts() -> float:
    """
    Єдина точка для float UNIX timestamps у runtime metrics/state.
    """
    return time.time()


def ensure_utc(dt: datetime | int | float | str | None = None) -> datetime:
    """
    Нормалізує datetime / epoch timestamp до timezone-aware UTC.

    Market state snapshots can carry timestamps as epoch milliseconds, while
    historical/manual code often passes datetime.  Keep this helper permissive
    so dataclass post-init normalization never crashes on scheduler payloads.
    """
    if dt is None:
        return utc_now()

    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    if isinstance(dt, (int, float)) and not isinstance(dt, bool):
        value = float(dt)
        abs_value = abs(value)
        if abs_value >= 1_000_000_000_000_000_000:
            value /= 1_000_000_000.0
        elif abs_value >= 1_000_000_000_000_000:
            value /= 1_000_000.0
        elif abs_value >= 1_000_000_000_000:
            value /= 1_000.0
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if isinstance(dt, str):
        text = dt.strip()
        if not text:
            return utc_now()
        try:
            return ensure_utc(float(text))
        except ValueError:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

    return utc_now()


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


def _normalize_market_type(market_type: object = DEFAULT_MARKET_TYPE) -> str:
    normalized = str(market_type or DEFAULT_MARKET_TYPE).strip().lower()
    return normalized if normalized else DEFAULT_MARKET_TYPE


def _normalize_timeframe(timeframe: object = DEFAULT_TIMEFRAME) -> str:
    normalized = str(timeframe or DEFAULT_TIMEFRAME).strip()
    return normalized if normalized else DEFAULT_TIMEFRAME


def _normalize_exchange_symbol(
    exchange_symbol: object,
    *,
    fallback_symbol: str,
) -> str:
    normalized = str(exchange_symbol or "").strip()
    return normalized if normalized else fallback_symbol


def make_spoofing_key(
    *,
    exchange: object,
    symbol: object,
    market_type: object = DEFAULT_MARKET_TYPE,
    timeframe: object = DEFAULT_TIMEFRAME,
) -> SpoofingKey:
    """
    Canonical key для multi-exchange futures spoofing analytics.

    Scope:
        exchange + market_type + symbol + timeframe
    """
    return (
        _normalize_exchange(exchange),
        _normalize_market_type(market_type),
        _normalize_symbol(symbol),
        _normalize_timeframe(timeframe),
    )


def spoofing_key_to_dict(key: SpoofingKey) -> dict[str, str]:
    exchange, market_type, symbol, timeframe = key
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def make_market_key(
    *,
    exchange: object,
    symbol: object,
    market_type: object = DEFAULT_MARKET_TYPE,
) -> tuple[str, str, str]:
    """
    Short market key без timeframe для aggregation, де timeframe не потрібен.
    """
    return (
        _normalize_exchange(exchange),
        _normalize_market_type(market_type),
        _normalize_symbol(symbol),
    )


def scoped_metadata(
    *,
    exchange: object,
    symbol: object,
    market_type: object = DEFAULT_MARKET_TYPE,
    timeframe: object = DEFAULT_TIMEFRAME,
    exchange_symbol: object | None = None,
) -> dict[str, str]:
    """
    JSON-friendly scope metadata для EventBus payloads.
    """
    key = make_spoofing_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    data = spoofing_key_to_dict(key)
    data["exchange_symbol"] = _normalize_exchange_symbol(
        exchange_symbol,
        fallback_symbol=data["symbol"],
    )
    return data


def _coerce_side(value: SpoofingSide | str | None) -> SpoofingSide:
    if isinstance(value, SpoofingSide):
        return value

    if value is None:
        return SpoofingSide.UNKNOWN

    normalized = str(value).strip().lower()
    if normalized in {"bid", "buy", "b", "long"}:
        return SpoofingSide.BID
    if normalized in {"ask", "sell", "s", "short"}:
        return SpoofingSide.ASK

    return SpoofingSide.UNKNOWN


def _coerce_trade_side(value: TradeSide | str | None) -> TradeSide:
    if isinstance(value, TradeSide):
        return value

    if value is None:
        return TradeSide.UNKNOWN

    normalized = str(value).strip().lower()
    if normalized in {"buy", "bid", "b", "long", "taker_buy", "aggressive_buy"}:
        return TradeSide.BUY
    if normalized in {"sell", "ask", "s", "short", "taker_sell", "aggressive_sell"}:
        return TradeSide.SELL

    return TradeSide.UNKNOWN


def _serialize_value(value: Any) -> Any:
    """
    Convert dataclasses/enums у EventBus-safe plain Python values.
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

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def model_to_dict(value: Any) -> dict[str, Any]:
    serialized = _serialize_value(value)
    if not isinstance(serialized, dict):
        raise TypeError(f"Expected dataclass/dict serializable to dict, got {type(value)!r}")
    return serialized


# =============================================================================
# Raw / normalized market models
# =============================================================================


@dataclass(slots=True)
class OrderbookLevel:
    """
    Простий рівень стакана.

    Використовується як lightweight model для normalized OrderBookCache snapshots.
    Production path:
        exchange adapters
            -> market.orderbook
            -> OrderBookCache
            -> market.orderbook.updated
            -> SpoofingAnalyzer
    """

    price: float
    size: float

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
        self.price = float(self.price)
        self.size = float(self.size)

        if self.price <= 0:
            raise ValueError("OrderbookLevel.price must be > 0")
        if self.size < 0:
            raise ValueError("OrderbookLevel.size must be >= 0")


# Backward-compatible alias for legacy naming from old spoofing_detector.py.
OrderBookLevel = OrderbookLevel


@dataclass(slots=True)
class TradeTick:
    """
    Нормалізована trade-flow подія.

    Для spoofing це secondary/confirmation input. Основний production source
    має приходити з TradesCache через market.trades.updated, а не напряму з біржі.
    """

    symbol: str
    price: float
    qty: float
    side: TradeSide
    ts_ms: int

    exchange: str | None = None
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    trade_id: str | None = None
    notional: float | None = None
    is_aggressive: bool = False
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
        self.symbol = _normalize_symbol(self.symbol)
        self.market_type = _normalize_market_type(self.market_type)
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        if self.exchange is not None:
            self.exchange = _normalize_exchange(self.exchange)

        self.side = _coerce_trade_side(self.side)
        self.price = float(self.price)
        self.qty = float(self.qty)
        self.ts_ms = int(self.ts_ms)
        self.notional = (
            float(self.notional)
            if self.notional is not None
            else self.price * self.qty
        )
        self.is_aggressive = bool(self.is_aggressive)
        self.metadata = dict(self.metadata or {})

        if self.price <= 0:
            raise ValueError("TradeTick.price must be > 0")
        if self.qty <= 0:
            raise ValueError("TradeTick.qty must be > 0")
        if self.ts_ms <= 0:
            raise ValueError("TradeTick.ts_ms must be > 0")
        if self.notional < 0:
            raise ValueError("TradeTick.notional must be >= 0")

    @property
    def key(self) -> SpoofingKey | None:
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
        if self.exchange is None:
            return None
        return make_spoofing_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def market_key(self) -> tuple[str, str, str] | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "market_key", _analytics_args)
        except Exception:
            pass
        if self.exchange is None:
            return None
        return make_market_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
        )


@dataclass(slots=True)
class OrderbookLevelSnapshot:
    """
    Нормалізований знімок конкретного рівня стакана.

    Основна input-модель для OrderbookWallDetector.
    Має бути побудована з OrderBookCache / market.orderbook.updated, а не з raw
    exchange adapter payload у production runtime.
    """

    symbol: str
    exchange: str
    side: SpoofingSide
    price: float
    size: float

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    best_bid: float | None = None
    best_ask: float | None = None
    mid_price: float | None = None
    spread: float | None = None
    sequence_id: int | None = None
    timestamp: datetime = field(default_factory=utc_now)
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
        self.exchange = _normalize_exchange(self.exchange)
        self.market_type = _normalize_market_type(self.market_type)
        self.symbol = _normalize_symbol(self.symbol)
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.side = _coerce_side(self.side)
        self.price = float(self.price)
        self.size = float(self.size)

        self.best_bid = float(self.best_bid) if self.best_bid is not None else None
        self.best_ask = float(self.best_ask) if self.best_ask is not None else None
        self.mid_price = float(self.mid_price) if self.mid_price is not None else None
        self.spread = float(self.spread) if self.spread is not None else None
        self.sequence_id = int(self.sequence_id) if self.sequence_id is not None else None
        self.timestamp = ensure_utc(self.timestamp)
        self.metadata = dict(self.metadata or {})

        if self.price <= 0:
            raise ValueError("OrderbookLevelSnapshot.price must be > 0")
        if self.size < 0:
            raise ValueError("OrderbookLevelSnapshot.size must be >= 0")

    @property
    def key(self) -> SpoofingKey:
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
        return make_spoofing_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def market_key(self) -> tuple[str, str, str]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "market_key", _analytics_args)
        except Exception:
            pass
        return make_market_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
        )

    @property
    def scope(self) -> dict[str, str]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "scope", _analytics_args)
        except Exception:
            pass
        return scoped_metadata(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange_symbol=self.exchange_symbol,
        )

    @property
    def level_key(self) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "level_key", _analytics_args)
        except Exception:
            pass
        return (
            f"{self.exchange}:{self.market_type}:{self.symbol}:{self.timeframe}:"
            f"{self.side.value}:{self.price:.12f}"
        )


# =============================================================================
# Stateful lifecycle models
# =============================================================================


@dataclass(slots=True)
class TrackedWall:
    """
    Внутрішня модель життєвого циклу великої стінки в стакані.

    Створюється та оновлюється PersistenceTracker.
    Detector-и працюють із цією stateful-моделлю, а не з raw orderbook.
    """

    wall_id: str
    symbol: str
    exchange: str
    side: SpoofingSide
    price: float

    first_seen_at: datetime
    last_seen_at: datetime

    initial_size: float
    current_size: float
    max_size: float
    min_size: float

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    best_bid_at_creation: float | None = None
    best_ask_at_creation: float | None = None
    mid_price_at_creation: float | None = None

    total_added_size: float = 0.0
    total_removed_size: float = 0.0
    estimated_filled_size: float = 0.0
    estimated_pulled_size: float = 0.0

    updates_count: int = 0
    touch_count: int = 0
    near_touch_count: int = 0

    state: OrderbookWallState = OrderbookWallState.ACTIVE
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
        self.wall_id = str(self.wall_id)
        self.exchange = _normalize_exchange(self.exchange)
        self.market_type = _normalize_market_type(self.market_type)
        self.symbol = _normalize_symbol(self.symbol)
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.side = _coerce_side(self.side)
        self.price = float(self.price)

        self.first_seen_at = ensure_utc(self.first_seen_at)
        self.last_seen_at = ensure_utc(self.last_seen_at)

        self.initial_size = float(self.initial_size)
        self.current_size = float(self.current_size)
        self.max_size = float(self.max_size)
        self.min_size = float(self.min_size)

        self.best_bid_at_creation = (
            float(self.best_bid_at_creation)
            if self.best_bid_at_creation is not None
            else None
        )
        self.best_ask_at_creation = (
            float(self.best_ask_at_creation)
            if self.best_ask_at_creation is not None
            else None
        )
        self.mid_price_at_creation = (
            float(self.mid_price_at_creation)
            if self.mid_price_at_creation is not None
            else None
        )

        self.total_added_size = float(self.total_added_size)
        self.total_removed_size = float(self.total_removed_size)
        self.estimated_filled_size = float(self.estimated_filled_size)
        self.estimated_pulled_size = float(self.estimated_pulled_size)

        self.updates_count = int(self.updates_count)
        self.touch_count = int(self.touch_count)
        self.near_touch_count = int(self.near_touch_count)

        if not isinstance(self.state, OrderbookWallState):
            self.state = OrderbookWallState(str(self.state))

        self.metadata = dict(self.metadata or {})

        if self.price <= 0:
            raise ValueError("TrackedWall.price must be > 0")
        if self.initial_size < 0 or self.current_size < 0:
            raise ValueError("TrackedWall sizes must be >= 0")
        if self.max_size < 0 or self.min_size < 0:
            raise ValueError("TrackedWall sizes must be >= 0")

    @property
    def key(self) -> SpoofingKey:
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
        return make_spoofing_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def market_key(self) -> tuple[str, str, str]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "market_key", _analytics_args)
        except Exception:
            pass
        return make_market_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
        )

    @property
    def level_key(self) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "level_key", _analytics_args)
        except Exception:
            pass
        return (
            f"{self.exchange}:{self.market_type}:{self.symbol}:{self.timeframe}:"
            f"{self.side.value}:{self.price:.12f}"
        )

    @property
    def lifetime_ms(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "lifetime_ms", _analytics_args)
        except Exception:
            pass
        return max(
            0.0,
            (self.last_seen_at - self.first_seen_at).total_seconds() * 1000.0,
        )

    @property
    def fill_ratio(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "fill_ratio", _analytics_args)
        except Exception:
            pass
        if self.max_size <= 0:
            return 0.0
        return max(0.0, min(1.0, self.estimated_filled_size / self.max_size))

    @property
    def pull_ratio(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "pull_ratio", _analytics_args)
        except Exception:
            pass
        if self.max_size <= 0:
            return 0.0
        return max(0.0, min(1.0, self.estimated_pulled_size / self.max_size))

    @property
    def current_to_max_ratio(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "current_to_max_ratio", _analytics_args)
        except Exception:
            pass
        if self.max_size <= 0:
            return 0.0
        return max(0.0, min(1.0, self.current_size / self.max_size))

    @property
    def size_delta(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "size_delta", _analytics_args)
        except Exception:
            pass
        return self.current_size - self.initial_size


@dataclass(slots=True)
class SpoofingCandidate:
    """
    Legacy-compatible внутрішній кандидат spoofing-події.

    Новий production flow має використовувати TrackedWall + SpoofingSignal.
    Ця модель залишена для міграції старої candidate-based логіки.
    """

    candidate_id: str
    symbol: str
    side: SpoofingSide
    price: float

    initial_size: float
    peak_size: float

    detected_ts_ms: int
    last_seen_ts_ms: int

    best_bid_at_detection: float
    best_ask_at_detection: float
    mid_at_detection: float

    avg_same_side_size_at_detection: float
    distance_bps_at_detection: float
    size_multiple_at_detection: float

    exchange: str | None = None
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    status: CandidateStatus = CandidateStatus.ACTIVE

    removed_ts_ms: int | None = None
    remaining_size: float | None = None
    cancel_ratio: float | None = None

    confirmation_ts_ms: int | None = None
    confirmation_price_move_bps: float | None = None
    opposite_pressure_ratio: float | None = None

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
        self.candidate_id = str(self.candidate_id)
        self.symbol = _normalize_symbol(self.symbol)
        self.side = _coerce_side(self.side)
        self.price = float(self.price)
        self.initial_size = float(self.initial_size)
        self.peak_size = float(self.peak_size)

        if self.exchange is not None:
            self.exchange = _normalize_exchange(self.exchange)

        self.market_type = _normalize_market_type(self.market_type)
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        if not isinstance(self.status, CandidateStatus):
            self.status = CandidateStatus(str(self.status))

        self.metadata = dict(self.metadata or {})

    @property
    def id(self) -> str:
        """
        Backward-compatible accessor for old code that used candidate.id.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "id", _analytics_args)
        except Exception:
            pass
        return self.candidate_id

    @property
    def key(self) -> SpoofingKey | None:
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
        if self.exchange is None:
            return None
        return make_spoofing_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )


@dataclass(slots=True)
class LiquidityLifecycleEvent:
    """
    Подія життєвого циклу стінки/ліквідності.

    Генерується PersistenceTracker під час створення, оновлення, pull,
    fill або expiry tracked wall.
    """

    wall_id: str
    symbol: str
    exchange: str
    side: SpoofingSide
    event_type: LiquidityEventType
    price: float
    size_before: float
    size_after: float
    delta_size: float

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    timestamp: datetime = field(default_factory=utc_now)
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
        self.wall_id = str(self.wall_id)
        self.exchange = _normalize_exchange(self.exchange)
        self.market_type = _normalize_market_type(self.market_type)
        self.symbol = _normalize_symbol(self.symbol)
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.side = _coerce_side(self.side)
        if not isinstance(self.event_type, LiquidityEventType):
            self.event_type = LiquidityEventType(str(self.event_type))

        self.price = float(self.price)
        self.size_before = float(self.size_before)
        self.size_after = float(self.size_after)
        self.delta_size = float(self.delta_size)
        self.timestamp = ensure_utc(self.timestamp)
        self.metadata = dict(self.metadata or {})

    @property
    def key(self) -> SpoofingKey:
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
        return make_spoofing_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )


# =============================================================================
# Detector features and results
# =============================================================================


@dataclass(slots=True)
class SpoofingFeatures:
    """
    Уніфікований набір ознак, з яких формується detector decision і score.
    """

    symbol: str
    exchange: str
    side: SpoofingSide
    price: float

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    wall_size: float = 0.0
    wall_size_ratio: float = 0.0
    distance_from_mid_bps: float = 0.0

    lifetime_ms: float = 0.0
    updates_count: int = 0
    repetition_count: int = 0

    fill_ratio: float = 0.0
    pull_ratio: float = 0.0
    cancel_to_fill_ratio: float = 0.0

    price_reaction_bps: float = 0.0
    pressure_flip_strength: float = 0.0
    layering_score: float = 0.0

    is_near_best_quote: bool = False
    is_fast_pull: bool = False
    is_fake_liquidity: bool = False
    is_layering: bool = False

    timestamp: datetime = field(default_factory=utc_now)
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
        self.exchange = _normalize_exchange(self.exchange)
        self.market_type = _normalize_market_type(self.market_type)
        self.symbol = _normalize_symbol(self.symbol)
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.side = _coerce_side(self.side)
        self.price = float(self.price)

        self.wall_size = float(self.wall_size)
        self.wall_size_ratio = float(self.wall_size_ratio)
        self.distance_from_mid_bps = float(self.distance_from_mid_bps)
        self.lifetime_ms = float(self.lifetime_ms)
        self.updates_count = int(self.updates_count)
        self.repetition_count = int(self.repetition_count)
        self.fill_ratio = float(self.fill_ratio)
        self.pull_ratio = float(self.pull_ratio)
        self.cancel_to_fill_ratio = float(self.cancel_to_fill_ratio)
        self.price_reaction_bps = float(self.price_reaction_bps)
        self.pressure_flip_strength = float(self.pressure_flip_strength)
        self.layering_score = float(self.layering_score)
        self.timestamp = ensure_utc(self.timestamp)
        self.metadata = dict(self.metadata or {})

    @property
    def key(self) -> SpoofingKey:
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
        return make_spoofing_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )


@dataclass(slots=True)
class DetectorResult:
    """
    Уніфікований результат окремого detector-а.
    """

    detector: SpoofingComponent
    decision: DetectorDecision
    score: float
    confidence: float
    reason: str
    features: SpoofingFeatures | None = None
    wall_id: str | None = None
    pattern: SpoofingPattern = SpoofingPattern.UNKNOWN
    timestamp: datetime = field(default_factory=utc_now)
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
        if not isinstance(self.detector, SpoofingComponent):
            self.detector = SpoofingComponent(str(self.detector))
        if not isinstance(self.decision, DetectorDecision):
            self.decision = DetectorDecision(str(self.decision))
        if not isinstance(self.pattern, SpoofingPattern):
            self.pattern = SpoofingPattern(str(self.pattern))

        self.score = max(0.0, min(1.0, float(self.score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.reason = str(self.reason)
        self.timestamp = ensure_utc(self.timestamp)
        self.metadata = dict(self.metadata or {})

    @property
    def key(self) -> SpoofingKey | None:
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
        return self.features.key if self.features is not None else None

    def is_positive(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_positive", _analytics_args)
        except Exception:
            pass
        return self.decision == DetectorDecision.POSITIVE


# =============================================================================
# Detector-local context models moved here from detector files
# =============================================================================


@dataclass(slots=True)
class WallCandidateContext:
    """
    Контекст оцінки конкретного orderbook level як потенційної стінки.
    """

    snapshot: OrderbookLevelSnapshot
    baseline_size: float
    size_ratio: float
    distance_from_mid_bps: float
    near_best_quote: bool
    notional: float
    confidence: float
    score: float
    reason: str

    @property
    def key(self) -> SpoofingKey:
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
        return self.snapshot.key


@dataclass(slots=True)
class PullCandidateContext:
    """
    Контекст оцінки tracked wall як pull-event кандидата.
    """

    wall: TrackedWall
    pulled_notional: float
    pulled_size_ratio: float
    fill_ratio: float
    pull_ratio: float
    lifetime_ms: float
    is_fast_pull: bool
    is_strong_pull: bool
    confidence: float
    score: float
    reason: str

    @property
    def key(self) -> SpoofingKey:
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
        return self.wall.key


@dataclass(slots=True)
class FakeLiquidityCandidateContext:
    """
    Контекст оцінки tracked wall як fake-liquidity кандидата.
    """

    wall: TrackedWall
    wall_notional: float
    pulled_notional: float
    lifetime_ms: float
    fill_ratio: float
    pull_ratio: float
    price_reaction_bps: float
    distance_from_mid_bps: float
    is_short_lived: bool
    is_low_fill: bool
    is_high_pull: bool
    has_market_reaction: bool
    confidence: float
    score: float
    reason: str

    @property
    def key(self) -> SpoofingKey:
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
        return self.wall.key


@dataclass(slots=True)
class FlipPressureCandidateContext:
    """
    Контекст оцінки tracked wall як pressure-flip / pressure-bluff кандидата.
    """

    wall: TrackedWall
    wall_notional: float
    pulled_notional: float
    lifetime_ms: float
    fill_ratio: float
    pull_ratio: float
    price_reaction_bps: float
    pressure_flip_strength: float
    distance_from_mid_bps: float
    is_pressure_removed: bool
    is_short_lived: bool
    is_low_fill: bool
    has_reversal: bool
    confidence: float
    score: float
    reason: str

    @property
    def key(self) -> SpoofingKey:
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
        return self.wall.key


@dataclass(slots=True)
class LayeringCluster:
    """
    Кластер потенційного multi-level layering патерну.
    """

    exchange: str
    symbol: str
    side: SpoofingSide
    walls: list[TrackedWall]
    total_notional: float
    average_pull_ratio: float
    average_fill_ratio: float
    average_lifetime_ms: float
    synchronized_pull_ratio: float
    price_span_bps: float
    layering_score: float
    cluster_price: float
    cluster_wall_id: str | None

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
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
        if self.walls:
            first = self.walls[0]
            self.exchange = first.exchange
            self.market_type = first.market_type
            self.symbol = first.symbol
            self.timeframe = first.timeframe
            self.exchange_symbol = first.exchange_symbol
            self.side = first.side
        else:
            self.exchange = _normalize_exchange(self.exchange)
            self.market_type = _normalize_market_type(self.market_type)
            self.symbol = _normalize_symbol(self.symbol)
            self.timeframe = _normalize_timeframe(self.timeframe)
            self.exchange_symbol = _normalize_exchange_symbol(
                self.exchange_symbol,
                fallback_symbol=self.symbol,
            )
            self.side = _coerce_side(self.side)

        self.total_notional = float(self.total_notional)
        self.average_pull_ratio = float(self.average_pull_ratio)
        self.average_fill_ratio = float(self.average_fill_ratio)
        self.average_lifetime_ms = float(self.average_lifetime_ms)
        self.synchronized_pull_ratio = float(self.synchronized_pull_ratio)
        self.price_span_bps = float(self.price_span_bps)
        self.layering_score = float(self.layering_score)
        self.cluster_price = float(self.cluster_price)

    @property
    def key(self) -> SpoofingKey:
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
        return make_spoofing_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )


@dataclass(slots=True)
class LayeringCandidateContext:
    """
    Контекст оцінки LayeringCluster як detector result.
    """

    cluster: LayeringCluster
    confidence: float
    score: float
    reason: str
    price_reaction_bps: float

    @property
    def key(self) -> SpoofingKey:
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
        return self.cluster.key


# =============================================================================
# Scoring / signal models
# =============================================================================


@dataclass(slots=True)
class ScoreContribution:
    """
    Внесок окремої ознаки в загальний spoofing score.
    """

    component: ScoreComponent
    raw_value: float
    normalized_value: float
    weight: float
    contribution: float
    description: str = ""

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
        if not isinstance(self.component, ScoreComponent):
            self.component = ScoreComponent(str(self.component))

        self.raw_value = float(self.raw_value)
        self.normalized_value = max(0.0, min(1.0, float(self.normalized_value)))
        self.weight = float(self.weight)
        self.contribution = float(self.contribution)
        self.description = str(self.description)


@dataclass(slots=True)
class SpoofingScore:
    """
    Підсумковий score із деталізацією.
    """

    total_score: float
    confidence: float
    severity: SpoofingSeverity
    contributions: list[ScoreContribution] = field(default_factory=list)
    threshold: float = 0.0
    passed: bool = False
    timestamp: datetime = field(default_factory=utc_now)
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
        self.total_score = max(0.0, min(1.0, float(self.total_score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

        if not isinstance(self.severity, SpoofingSeverity):
            self.severity = SpoofingSeverity(str(self.severity))

        self.threshold = float(self.threshold)
        self.passed = bool(self.passed)
        self.timestamp = ensure_utc(self.timestamp)
        self.metadata = dict(self.metadata or {})


@dataclass(slots=True)
class AggregationContext:
    """
    Агрегований контекст перед побудовою фінального score/signal.
    """

    symbol: str
    exchange: str
    price: float
    features: SpoofingFeatures
    detector_results: list[DetectorResult]
    agreement_ratio: float
    average_confidence: float
    primary_pattern: SpoofingPattern
    spoofing_type: SpoofingType
    wall_id: str | None

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
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
        if self.features is not None:
            self.exchange = self.features.exchange
            self.market_type = self.features.market_type
            self.symbol = self.features.symbol
            self.timeframe = self.features.timeframe
            self.exchange_symbol = self.features.exchange_symbol
        else:
            self.exchange = _normalize_exchange(self.exchange)
            self.market_type = _normalize_market_type(self.market_type)
            self.symbol = _normalize_symbol(self.symbol)
            self.timeframe = _normalize_timeframe(self.timeframe)
            self.exchange_symbol = _normalize_exchange_symbol(
                self.exchange_symbol,
                fallback_symbol=self.symbol,
            )

        self.price = float(self.price)
        self.agreement_ratio = max(0.0, min(1.0, float(self.agreement_ratio)))
        self.average_confidence = max(0.0, min(1.0, float(self.average_confidence)))

        if not isinstance(self.primary_pattern, SpoofingPattern):
            self.primary_pattern = SpoofingPattern(str(self.primary_pattern))
        if not isinstance(self.spoofing_type, SpoofingType):
            self.spoofing_type = SpoofingType(str(self.spoofing_type))

    @property
    def key(self) -> SpoofingKey:
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
        return make_spoofing_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )


@dataclass(slots=True)
class SpoofingSignal:
    """
    Фінальний spoofing-сигнал, який analyzer публікує через EventBus.

    Має повний futures scope, щоб downstream strategy/risk/dashboard не
    плутали однаковий symbol на різних біржах або market_type.
    """

    signal_id: str
    symbol: str
    exchange: str
    side: SpoofingSide
    spoofing_type: SpoofingType
    pattern: SpoofingPattern
    status: SpoofingStatus

    price_level: float
    wall_id: str | None

    score: float
    confidence: float
    severity: SpoofingSeverity

    first_seen_at: datetime

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    detected_at: datetime = field(default_factory=utc_now)

    features: SpoofingFeatures | None = None
    detector_results: list[DetectorResult] = field(default_factory=list)
    score_breakdown: SpoofingScore | None = None
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
        if self.features is not None:
            self.exchange = self.features.exchange
            self.market_type = self.features.market_type
            self.symbol = self.features.symbol
            self.timeframe = self.features.timeframe
            self.exchange_symbol = self.features.exchange_symbol
        else:
            self.exchange = _normalize_exchange(self.exchange)
            self.market_type = _normalize_market_type(self.market_type)
            self.symbol = _normalize_symbol(self.symbol)
            self.timeframe = _normalize_timeframe(self.timeframe)
            self.exchange_symbol = _normalize_exchange_symbol(
                self.exchange_symbol,
                fallback_symbol=self.symbol,
            )

        self.signal_id = str(self.signal_id)
        self.side = _coerce_side(self.side)

        if not isinstance(self.spoofing_type, SpoofingType):
            self.spoofing_type = SpoofingType(str(self.spoofing_type))
        if not isinstance(self.pattern, SpoofingPattern):
            self.pattern = SpoofingPattern(str(self.pattern))
        if not isinstance(self.status, SpoofingStatus):
            self.status = SpoofingStatus(str(self.status))
        if not isinstance(self.severity, SpoofingSeverity):
            self.severity = SpoofingSeverity(str(self.severity))

        self.price_level = float(self.price_level)
        self.score = max(0.0, min(1.0, float(self.score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

        self.first_seen_at = ensure_utc(self.first_seen_at)
        self.detected_at = ensure_utc(self.detected_at)
        self.metadata = dict(self.metadata or {})
        self.metadata.setdefault("scope", spoofing_key_to_dict(self.key))
        self.metadata.setdefault("exchange_symbol", self.exchange_symbol)

    @property
    def key(self) -> SpoofingKey:
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
        return make_spoofing_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )


@dataclass(slots=True)
class AnalyzerOutput:
    """
    Результат роботи SpoofingAnalyzer за один scoped processing cycle.

    Scope:
        exchange + market_type + symbol + timeframe
    """

    symbol: str
    exchange: str
    signal: SpoofingSignal | None

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    detector_results: list[DetectorResult] = field(default_factory=list)
    tracked_walls: list[TrackedWall] = field(default_factory=list)
    lifecycle_events: list[LiquidityLifecycleEvent] = field(default_factory=list)
    timestamp: datetime = field(default_factory=utc_now)
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
        if self.signal is not None:
            self.exchange = self.signal.exchange
            self.market_type = self.signal.market_type
            self.symbol = self.signal.symbol
            self.timeframe = self.signal.timeframe
            self.exchange_symbol = self.signal.exchange_symbol
        else:
            self.exchange = _normalize_exchange(self.exchange)
            self.market_type = _normalize_market_type(self.market_type)
            self.symbol = _normalize_symbol(self.symbol)
            self.timeframe = _normalize_timeframe(self.timeframe)
            self.exchange_symbol = _normalize_exchange_symbol(
                self.exchange_symbol,
                fallback_symbol=self.symbol,
            )

        self.timestamp = ensure_utc(self.timestamp)
        self.metadata = dict(self.metadata or {})
        self.metadata.setdefault("scope", spoofing_key_to_dict(self.key))
        self.metadata.setdefault("exchange_symbol", self.exchange_symbol)

    @property
    def key(self) -> SpoofingKey:
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
        return make_spoofing_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )


# =============================================================================
# Serialization helpers
# =============================================================================


def orderbook_level_to_dict(level: OrderbookLevel) -> dict[str, Any]:
    return model_to_dict(level)


def trade_tick_to_dict(trade: TradeTick) -> dict[str, Any]:
    return model_to_dict(trade)


def snapshot_to_dict(snapshot: OrderbookLevelSnapshot) -> dict[str, Any]:
    return model_to_dict(snapshot)


def wall_to_dict(wall: TrackedWall) -> dict[str, Any]:
    return model_to_dict(wall)


def lifecycle_event_to_dict(event: LiquidityLifecycleEvent) -> dict[str, Any]:
    return model_to_dict(event)


def features_to_dict(features: SpoofingFeatures) -> dict[str, Any]:
    return model_to_dict(features)


def detector_result_to_dict(result: DetectorResult) -> dict[str, Any]:
    return model_to_dict(result)


def score_to_dict(score: SpoofingScore) -> dict[str, Any]:
    return model_to_dict(score)


def signal_to_dict(signal: SpoofingSignal) -> dict[str, Any]:
    return model_to_dict(signal)


def analyzer_output_to_dict(output: AnalyzerOutput) -> dict[str, Any]:
    return model_to_dict(output)


__all__ = [
    # defaults / key helpers
    "DEFAULT_MARKET_TYPE",
    "DEFAULT_TIMEFRAME",
    "SpoofingKey",
    "utc_now",
    "unix_ts",
    "ensure_utc",
    "make_spoofing_key",
    "spoofing_key_to_dict",
    "make_market_key",
    "scoped_metadata",

    # raw / normalized market models
    "OrderbookLevel",
    "OrderBookLevel",
    "TradeTick",
    "OrderbookLevelSnapshot",

    # lifecycle models
    "TrackedWall",
    "SpoofingCandidate",
    "LiquidityLifecycleEvent",

    # detector features/results
    "SpoofingFeatures",
    "DetectorResult",

    # detector contexts
    "WallCandidateContext",
    "PullCandidateContext",
    "FakeLiquidityCandidateContext",
    "FlipPressureCandidateContext",
    "LayeringCluster",
    "LayeringCandidateContext",

    # scoring/signal models
    "ScoreContribution",
    "SpoofingScore",
    "AggregationContext",
    "SpoofingSignal",
    "AnalyzerOutput",

    # serialization helpers
    "model_to_dict",
    "orderbook_level_to_dict",
    "trade_tick_to_dict",
    "snapshot_to_dict",
    "wall_to_dict",
    "lifecycle_event_to_dict",
    "features_to_dict",
    "detector_result_to_dict",
    "score_to_dict",
    "signal_to_dict",
    "analyzer_output_to_dict",
]