from __future__ import annotations
from core.logger import get_logger

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, TypeAlias

from analytics.funding.enums import (
    FundingBias,
    FundingDataSource,
    FundingDivergenceType,
    FundingEventType,
    FundingExtremeType,
    FundingFlipType,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingSignalType,
    FundingTimeframe,
)


DEFAULT_MARKET_TYPE = "perpetual"
DEFAULT_EXCHANGE_SYMBOL = ""
DEFAULT_TIMEFRAME = FundingTimeframe.H1

FundingKey: TypeAlias = tuple[str, str, str, str]
# exchange, market_type, symbol, timeframe


# =============================================================================
# Time helpers
# =============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# =============================================================================
# Scope / normalization helpers
# =============================================================================


def normalize_symbol(symbol: object) -> str:
    value = (
        str(symbol or "")
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
        .upper()
        .strip()
    )
    if not value:
        raise ValueError("symbol must not be empty")
    return value


def normalize_exchange_symbol(
    exchange_symbol: object | None,
    *,
    fallback_symbol: str,
) -> str:
    value = str(exchange_symbol or "").strip()
    return value or fallback_symbol


def normalize_market_type(market_type: object | None = None) -> str:
    value = str(market_type or DEFAULT_MARKET_TYPE).strip().lower()
    return value or DEFAULT_MARKET_TYPE


def normalize_timeframe(
    timeframe: FundingTimeframe | str | None = None,
) -> FundingTimeframe:
    if isinstance(timeframe, FundingTimeframe):
        return timeframe

    if timeframe is None:
        return DEFAULT_TIMEFRAME

    raw = str(timeframe).strip()
    if not raw:
        return DEFAULT_TIMEFRAME

    for item in FundingTimeframe:
        if item.value == raw:
            return item

    raise ValueError(f"Unsupported funding timeframe: {timeframe!r}")


def normalize_exchange(
    exchange: FundingDataSource | str | None = None,
) -> FundingDataSource:
    if isinstance(exchange, FundingDataSource):
        return exchange

    if exchange is None:
        return FundingDataSource.UNKNOWN

    raw = str(exchange).strip().lower()
    if not raw:
        return FundingDataSource.UNKNOWN

    for item in FundingDataSource:
        if item.value == raw:
            return item

    return FundingDataSource.UNKNOWN


def exchange_value(exchange: FundingDataSource | str | None) -> str:
    if isinstance(exchange, FundingDataSource):
        return exchange.value
    return normalize_exchange(exchange).value


def make_funding_key(
    *,
    exchange: FundingDataSource | str | None,
    market_type: str | None,
    symbol: str,
    timeframe: FundingTimeframe | str | None = None,
) -> FundingKey:
    normalized_timeframe = normalize_timeframe(timeframe)
    return (
        exchange_value(exchange),
        normalize_market_type(market_type),
        normalize_symbol(symbol),
        normalized_timeframe.value,
    )


def funding_key_to_dict(key: FundingKey) -> dict[str, str]:
    exchange, market_type, symbol, timeframe = key
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def scoped_metadata(
    *,
    exchange: FundingDataSource | str | None,
    market_type: str | None,
    symbol: str,
    timeframe: FundingTimeframe | str | None,
    exchange_symbol: str | None = None,
) -> dict[str, str]:
    key = make_funding_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    payload = funding_key_to_dict(key)
    payload["exchange_symbol"] = normalize_exchange_symbol(
        exchange_symbol,
        fallback_symbol=payload["symbol"],
    )
    return payload


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _datetime_to_payload(value: datetime | None) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _metadata_copy(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def _inject_scope(
    payload: dict[str, Any],
    *,
    key: FundingKey,
    exchange_symbol: str,
) -> dict[str, Any]:
    scope = funding_key_to_dict(key)
    payload.update(scope)
    payload["exchange_symbol"] = exchange_symbol
    payload["scope"] = scope
    return payload


def _serialize_value(value: Any, *, _seen: set[int] | None = None) -> Any:
    """
    Safely serialize funding models/events into EventBus/storage payloads.

    This helper intentionally avoids dataclasses.asdict() because asdict() performs
    recursive deep conversion before we can protect against cycles or non-payload
    runtime attributes. Funding events may contain nested analytics payloads,
    FeatureSnapshot-like objects, logger references, or even accidental circular
    dictionaries from upstream contract adapters.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "value"):
        return _serialize_value(getattr(value, "value"), _seen=_seen)

    if _seen is None:
        _seen = set()

    value_id = id(value)
    if isinstance(value, (Mapping, list, tuple, set, frozenset)) or (
        is_dataclass(value) and not isinstance(value, type)
    ):
        if value_id in _seen:
            return "<circular>"
        _seen.add(value_id)

    try:
        if is_dataclass(value) and not isinstance(value, type):
            payload: dict[str, Any] = {}
            for item in fields(value):
                name = item.name
                if name in {"logger", "_logger"} or name.startswith("_analytics"):
                    continue
                payload[name] = _serialize_value(getattr(value, name), _seen=_seen)
            return payload

        if isinstance(value, Mapping):
            payload: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(_serialize_value(key, _seen=_seen))
                if key_text in {"logger", "_logger"} or key_text.startswith("_analytics"):
                    continue
                payload[key_text] = _serialize_value(item, _seen=_seen)
            return payload

        if isinstance(value, (list, tuple, set, frozenset)):
            return [_serialize_value(item, _seen=_seen) for item in value]

        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return _serialize_value(value.to_dict(), _seen=_seen)
            except Exception:
                return str(value)

        if hasattr(value, "to_payload") and callable(value.to_payload):
            try:
                return _serialize_value(value.to_payload(), _seen=_seen)
            except Exception:
                return str(value)

        return value
    finally:
        if value_id in _seen:
            _seen.discard(value_id)


# =============================================================================
# Base scoped model
# =============================================================================


@dataclass(slots=True)
class FundingScopedModel:
    """
    Базовий scoped model для analytics.funding.

    Scope:
        exchange + market_type + symbol + timeframe

    Важливо:
    - exchange лишається FundingDataSource для backward compatibility;
    - market_type є string, бо це біржовий futures/perpetual scope:
      perpetual, futures, linear, inverse, swap, usdm_futures, coinm_futures;
    - exchange_symbol зберігає нативний символ біржі.
    """

    symbol: str
    exchange: FundingDataSource = FundingDataSource.UNKNOWN
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: FundingTimeframe = DEFAULT_TIMEFRAME
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
        self.metadata.setdefault("scope", funding_key_to_dict(self.key))
        self.metadata.setdefault("exchange_symbol", self.exchange_symbol)

    @property
    def key(self) -> FundingKey:
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
        return make_funding_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
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

    def _base_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_base_payload", _analytics_args)
        except Exception:
            pass
        payload = {
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "market_type": self.market_type,
            "timeframe": self.timeframe.value,
            "exchange_symbol": self.exchange_symbol,
            "metadata": dict(self.metadata),
        }
        payload["scope"] = funding_key_to_dict(self.key)
        return payload


# =============================================================================
# Funding data models
# =============================================================================


@dataclass(slots=True)
class FundingSnapshot(FundingScopedModel):
    """
    Нормалізований funding snapshot для одного futures/perpetual instrument.

    Production source:
        exchange adapter -> market.funding
        -> FundingCache
        -> market.funding.updated
        -> FundingAnalyzer

    Ця модель не має напряму залежати від EventBus.
    """

    funding_rate: float = 0.0
    predicted_funding_rate: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    open_interest: float | None = None
    volume_24h: float | None = None
    next_funding_time: datetime | None = None
    event_time: datetime = field(default_factory=utc_now)
    received_at: datetime = field(default_factory=utc_now)

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
        FundingScopedModel.__post_init__(self)

        self.funding_rate = float(self.funding_rate)

        if self.predicted_funding_rate is not None:
            self.predicted_funding_rate = float(self.predicted_funding_rate)

        if self.mark_price is not None:
            self.mark_price = float(self.mark_price)

        if self.index_price is not None:
            self.index_price = float(self.index_price)

        if self.open_interest is not None:
            self.open_interest = float(self.open_interest)

        if self.volume_24h is not None:
            self.volume_24h = float(self.volume_24h)

        self.event_time = ensure_utc(self.event_time)
        self.received_at = ensure_utc(self.received_at)

        if self.next_funding_time is not None:
            self.next_funding_time = ensure_utc(self.next_funding_time)

    @property
    def basis(self) -> float | None:
        """
        Відносне відхилення mark від index, якщо обидва значення доступні.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "basis", _analytics_args)
        except Exception:
            pass
        if self.mark_price is None or self.index_price is None or self.index_price == 0:
            return None
        return (self.mark_price - self.index_price) / self.index_price

    @property
    def funding_sign(self) -> int:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "funding_sign", _analytics_args)
        except Exception:
            pass
        if self.funding_rate > 0:
            return 1
        if self.funding_rate < 0:
            return -1
        return 0

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
        payload = self._base_payload()
        payload.update(
            {
                "funding_rate": self.funding_rate,
                "predicted_funding_rate": self.predicted_funding_rate,
                "mark_price": self.mark_price,
                "index_price": self.index_price,
                "open_interest": self.open_interest,
                "volume_24h": self.volume_24h,
                "next_funding_time": _datetime_to_payload(self.next_funding_time),
                "event_time": _datetime_to_payload(self.event_time),
                "received_at": _datetime_to_payload(self.received_at),
                "basis": self.basis,
                "funding_sign": self.funding_sign,
            }
        )
        return payload


@dataclass(slots=True)
class FundingStatistics(FundingScopedModel):
    """
    Агрегована статистика funding на заданому вікні.
    """

    current_rate: float = 0.0
    mean_rate: float = 0.0
    median_rate: float = 0.0
    std_rate: float = 0.0
    min_rate: float = 0.0
    max_rate: float = 0.0

    zscore: float | None = None
    percentile: float | None = None

    sample_size: int = 0
    window_start: datetime | None = None
    window_end: datetime | None = None
    updated_at: datetime = field(default_factory=utc_now)

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
        FundingScopedModel.__post_init__(self)

        self.current_rate = float(self.current_rate)
        self.mean_rate = float(self.mean_rate)
        self.median_rate = float(self.median_rate)
        self.std_rate = max(0.0, float(self.std_rate))
        self.min_rate = float(self.min_rate)
        self.max_rate = float(self.max_rate)
        self.sample_size = max(0, int(self.sample_size))

        if self.zscore is not None:
            self.zscore = float(self.zscore)
        if self.percentile is not None:
            self.percentile = max(0.0, min(100.0, float(self.percentile)))

        self.updated_at = ensure_utc(self.updated_at)

        if self.window_start is not None:
            self.window_start = ensure_utc(self.window_start)
        if self.window_end is not None:
            self.window_end = ensure_utc(self.window_end)

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
        payload = self._base_payload()
        payload.update(
            {
                "current_rate": self.current_rate,
                "mean_rate": self.mean_rate,
                "median_rate": self.median_rate,
                "std_rate": self.std_rate,
                "min_rate": self.min_rate,
                "max_rate": self.max_rate,
                "zscore": self.zscore,
                "percentile": self.percentile,
                "sample_size": self.sample_size,
                "window_start": _datetime_to_payload(self.window_start),
                "window_end": _datetime_to_payload(self.window_end),
                "updated_at": _datetime_to_payload(self.updated_at),
            }
        )
        return payload


@dataclass(slots=True)
class FundingRegimeState(FundingScopedModel):
    """
    Стан funding regime для одного scoped futures/perpetual instrument.
    """

    regime: FundingRegime = FundingRegime.UNKNOWN
    bias: FundingBias = FundingBias.NEUTRAL

    current_rate: float = 0.0
    mean_rate: float | None = None
    zscore: float | None = None
    percentile: float | None = None

    confidence: float = 0.0
    changed: bool = False
    previous_regime: FundingRegime | None = None

    event_time: datetime = field(default_factory=utc_now)

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
        FundingScopedModel.__post_init__(self)

        self.current_rate = float(self.current_rate)

        if self.mean_rate is not None:
            self.mean_rate = float(self.mean_rate)
        if self.zscore is not None:
            self.zscore = float(self.zscore)
        if self.percentile is not None:
            self.percentile = max(0.0, min(100.0, float(self.percentile)))

        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.event_time = ensure_utc(self.event_time)

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
        payload = self._base_payload()
        payload.update(
            {
                "regime": self.regime.value,
                "bias": self.bias.value,
                "current_rate": self.current_rate,
                "mean_rate": self.mean_rate,
                "zscore": self.zscore,
                "percentile": self.percentile,
                "confidence": self.confidence,
                "changed": self.changed,
                "previous_regime": (
                    self.previous_regime.value
                    if self.previous_regime is not None
                    else None
                ),
                "event_time": _datetime_to_payload(self.event_time),
            }
        )
        return payload


@dataclass(slots=True)
class FundingExtremeEvent(FundingScopedModel):
    """
    Подія екстремального funding.
    """

    extreme_type: FundingExtremeType = FundingExtremeType.NONE
    regime: FundingRegime = FundingRegime.UNKNOWN
    funding_rate: float = 0.0

    zscore: float | None = None
    percentile: float | None = None

    severity: float = 0.0
    is_reversal_risk: bool = False
    is_squeeze_risk: bool = False

    event_time: datetime = field(default_factory=utc_now)

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
        FundingScopedModel.__post_init__(self)

        self.funding_rate = float(self.funding_rate)
        if self.zscore is not None:
            self.zscore = float(self.zscore)
        if self.percentile is not None:
            self.percentile = max(0.0, min(100.0, float(self.percentile)))
        self.severity = max(0.0, min(1.0, float(self.severity)))
        self.event_time = ensure_utc(self.event_time)

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
        payload = self._base_payload()
        payload.update(
            {
                "extreme_type": self.extreme_type.value,
                "regime": self.regime.value,
                "funding_rate": self.funding_rate,
                "zscore": self.zscore,
                "percentile": self.percentile,
                "severity": self.severity,
                "is_reversal_risk": self.is_reversal_risk,
                "is_squeeze_risk": self.is_squeeze_risk,
                "event_time": _datetime_to_payload(self.event_time),
            }
        )
        return payload


@dataclass(slots=True)
class FundingDivergenceEvent(FundingScopedModel):
    """
    Подія дивергенції funding з ціною, OI, CVD або ліквідаціями.
    """

    divergence_type: FundingDivergenceType = FundingDivergenceType.NONE
    funding_rate: float = 0.0

    price_change_pct: float | None = None
    oi_change_pct: float | None = None
    cvd_change: float | None = None
    long_liquidations: float | None = None
    short_liquidations: float | None = None

    confidence: float = 0.0
    event_time: datetime = field(default_factory=utc_now)

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
        FundingScopedModel.__post_init__(self)

        self.funding_rate = float(self.funding_rate)

        if self.price_change_pct is not None:
            self.price_change_pct = float(self.price_change_pct)
        if self.oi_change_pct is not None:
            self.oi_change_pct = float(self.oi_change_pct)
        if self.cvd_change is not None:
            self.cvd_change = float(self.cvd_change)
        if self.long_liquidations is not None:
            self.long_liquidations = max(0.0, float(self.long_liquidations))
        if self.short_liquidations is not None:
            self.short_liquidations = max(0.0, float(self.short_liquidations))

        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.event_time = ensure_utc(self.event_time)

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
        payload = self._base_payload()
        payload.update(
            {
                "divergence_type": self.divergence_type.value,
                "funding_rate": self.funding_rate,
                "price_change_pct": self.price_change_pct,
                "oi_change_pct": self.oi_change_pct,
                "cvd_change": self.cvd_change,
                "long_liquidations": self.long_liquidations,
                "short_liquidations": self.short_liquidations,
                "confidence": self.confidence,
                "event_time": _datetime_to_payload(self.event_time),
            }
        )
        return payload


@dataclass(slots=True)
class FundingFlipEvent(FundingScopedModel):
    """
    Подія зміни знаку funding.
    """

    flip_type: FundingFlipType = FundingFlipType.NONE
    previous_rate: float = 0.0
    current_rate: float = 0.0

    flip_magnitude: float = 0.0
    confidence: float = 0.0

    event_time: datetime = field(default_factory=utc_now)

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
        FundingScopedModel.__post_init__(self)

        self.previous_rate = float(self.previous_rate)
        self.current_rate = float(self.current_rate)
        self.flip_magnitude = abs(self.current_rate - self.previous_rate)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.event_time = ensure_utc(self.event_time)

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
        payload = self._base_payload()
        payload.update(
            {
                "flip_type": self.flip_type.value,
                "previous_rate": self.previous_rate,
                "current_rate": self.current_rate,
                "flip_magnitude": self.flip_magnitude,
                "confidence": self.confidence,
                "event_time": _datetime_to_payload(self.event_time),
            }
        )
        return payload


@dataclass(slots=True)
class FundingPressureState(FundingScopedModel):
    """
    Оцінка накопиченого funding pressure / crowded positioning.
    """

    direction: FundingPressureDirection = FundingPressureDirection.NEUTRAL
    level: FundingPressureLevel = FundingPressureLevel.UNKNOWN
    bias: FundingBias = FundingBias.NEUTRAL

    funding_rate: float = 0.0
    pressure_score: float = 0.0
    oi_confirmation: bool = False
    price_stall_confirmation: bool = False

    squeeze_probability: float | None = None
    mean_reversion_probability: float | None = None

    event_time: datetime = field(default_factory=utc_now)

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
        FundingScopedModel.__post_init__(self)

        self.funding_rate = float(self.funding_rate)
        self.pressure_score = max(0.0, min(1.0, float(self.pressure_score)))

        if self.squeeze_probability is not None:
            self.squeeze_probability = max(0.0, min(1.0, float(self.squeeze_probability)))
        if self.mean_reversion_probability is not None:
            self.mean_reversion_probability = max(
                0.0,
                min(1.0, float(self.mean_reversion_probability)),
            )

        self.event_time = ensure_utc(self.event_time)

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
        payload = self._base_payload()
        payload.update(
            {
                "direction": self.direction.value,
                "level": self.level.value,
                "bias": self.bias.value,
                "funding_rate": self.funding_rate,
                "pressure_score": self.pressure_score,
                "oi_confirmation": self.oi_confirmation,
                "price_stall_confirmation": self.price_stall_confirmation,
                "squeeze_probability": self.squeeze_probability,
                "mean_reversion_probability": self.mean_reversion_probability,
                "event_time": _datetime_to_payload(self.event_time),
            }
        )
        return payload


@dataclass(slots=True)
class FundingSignal(FundingScopedModel):
    """
    Нормалізований funding-сигнал для strategy layer.

    Strategy має слухати analytics.funding.signal і отримувати payload
    з повним exchange + market_type + symbol + timeframe scope.
    """

    signal_type: FundingSignalType = FundingSignalType.REGIME_CHANGE
    bias: FundingBias = FundingBias.NEUTRAL
    regime: FundingRegime = FundingRegime.UNKNOWN

    score: float = 0.0
    confidence: float = 0.0
    description: str = ""

    supporting_factors: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    event_time: datetime = field(default_factory=utc_now)

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
        FundingScopedModel.__post_init__(self)

        self.score = max(-1.0, min(1.0, float(self.score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.supporting_factors = list(self.supporting_factors or [])
        self.tags = list(self.tags or [])
        self.event_time = ensure_utc(self.event_time)

    @property
    def bullish(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "bullish", _analytics_args)
        except Exception:
            pass
        return self.score > 0

    @property
    def bearish(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "bearish", _analytics_args)
        except Exception:
            pass
        return self.score < 0

    @property
    def neutral(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "neutral", _analytics_args)
        except Exception:
            pass
        return self.score == 0

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
        payload = self._base_payload()
        payload.update(
            {
                "signal_type": self.signal_type.value,
                "bias": self.bias.value,
                "regime": self.regime.value,
                "score": self.score,
                "confidence": self.confidence,
                "description": self.description,
                "supporting_factors": list(self.supporting_factors),
                "tags": list(self.tags),
                "event_time": _datetime_to_payload(self.event_time),
                "bullish": self.bullish,
                "bearish": self.bearish,
                "neutral": self.neutral,
            }
        )
        return payload


@dataclass(slots=True)
class FundingAnalyticsEvent(FundingScopedModel):
    """
    Уніфікована обгортка для подій, які можна напряму публікувати в EventBus.
    """

    event_type: FundingEventType = FundingEventType.SNAPSHOT
    payload: dict[str, Any] = field(default_factory=dict)
    event_time: datetime = field(default_factory=utc_now)
    source: str = "analytics.funding"

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
        FundingScopedModel.__post_init__(self)

        self.payload = dict(self.payload or {})
        self.event_time = ensure_utc(self.event_time)

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
        payload = self._base_payload()
        payload.update(
            {
                "event_type": self.event_type.value,
                "payload": _serialize_value(self.payload),
                "event_time": _datetime_to_payload(self.event_time),
                "source": self.source,
            }
        )
        return payload


# =============================================================================
# Generic serialization helper
# =============================================================================


def model_to_payload(model: Any) -> dict[str, Any]:
    """
    Єдиний helper для EventBus/storage/dashboard payload serialization.
    """
    if hasattr(model, "to_dict") and callable(model.to_dict):
        return model.to_dict()

    if hasattr(model, "to_payload") and callable(model.to_payload):
        payload = model.to_payload()
        if isinstance(payload, Mapping):
            return dict(payload)

    if is_dataclass(model):
        return _serialize_value(model)

    if isinstance(model, Mapping):
        return _serialize_value(model)

    raise TypeError(f"Unsupported funding model type: {type(model)!r}")


__all__ = [
    "DEFAULT_MARKET_TYPE",
    "DEFAULT_EXCHANGE_SYMBOL",
    "DEFAULT_TIMEFRAME",
    "FundingKey",
    "utc_now",
    "ensure_utc",
    "normalize_symbol",
    "normalize_exchange_symbol",
    "normalize_market_type",
    "normalize_timeframe",
    "normalize_exchange",
    "exchange_value",
    "make_funding_key",
    "funding_key_to_dict",
    "scoped_metadata",
    "FundingScopedModel",
    "FundingSnapshot",
    "FundingStatistics",
    "FundingRegimeState",
    "FundingExtremeEvent",
    "FundingDivergenceEvent",
    "FundingFlipEvent",
    "FundingPressureState",
    "FundingSignal",
    "FundingAnalyticsEvent",
    "model_to_payload",
]