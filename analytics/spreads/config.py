from __future__ import annotations
from core.logger import get_logger

from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Mapping

from .enums import InstrumentType, parse_instrument_type
from .models import (
    DEFAULT_FUTURES_MARKET_TYPE,
    DEFAULT_PERPETUAL_MARKET_TYPE,
    DEFAULT_SPOT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    SpreadKey,
    make_spread_key,
    spread_key_to_dict,
)


# ============================================================
# Constants
# ============================================================

# Production input topics.
# Важливо: price/quote source для spreads тепер іде з OrderBookCache,
# тобто з data-layer updated event, а не з окремого QuoteCache.
DEFAULT_ORDERBOOK_EVENT_TOPIC = "market.orderbook.updated"
DEFAULT_FUNDING_EVENT_TOPIC = "market.funding.updated"

# Backward-compatible alias для старого коду, який ще імпортує
# DEFAULT_QUOTE_EVENT_TOPIC або читає quote_event_topic.
# Значення навмисно вказує на production orderbook topic.
DEFAULT_QUOTE_EVENT_TOPIC = DEFAULT_ORDERBOOK_EVENT_TOPIC

# Legacy topics. Не використовувати в production analyzer subscriptions.
DEFAULT_LEGACY_QUOTE_EVENT_TOPIC = "market.quote.updated"
DEFAULT_RAW_ORDERBOOK_EVENT_TOPIC = "market.orderbook"
DEFAULT_RAW_QUOTE_EVENT_TOPIC = "market.quote"
DEFAULT_RAW_FUNDING_EVENT_TOPIC = "market.funding"

DEFAULT_SPOT_FUTURES_SNAPSHOT_TOPIC = "analytics.spreads.spot_futures.updated"
DEFAULT_CROSS_EXCHANGE_SNAPSHOT_TOPIC = "analytics.spreads.cross_exchange.updated"
DEFAULT_SPREAD_SIGNAL_TOPIC = "analytics.spreads.signal.generated"
DEFAULT_ARBITRAGE_OPPORTUNITY_TOPIC = "analytics.spreads.arbitrage.opportunity"

DEFAULT_ANALYZER_STARTED_TOPIC = "analytics.spreads.analyzer.started"
DEFAULT_ANALYZER_STOPPED_TOPIC = "analytics.spreads.analyzer.stopped"
DEFAULT_ANALYZER_HEARTBEAT_TOPIC = "analytics.spreads.analyzer.heartbeat"

DECIMAL_ZERO = Decimal("0")

# Project runtime scope.
# Market data exchanges used by the project: Binance, Bybit, OKX, MEXC.
# Bitget is intentionally not included.
PROJECT_DEFAULT_EXCHANGE = "binance"
PROJECT_EXCHANGES: frozenset[str] = frozenset({"binance", "bybit", "okx", "mexc"})
PROJECT_FUTURES_MARKET_TYPES: frozenset[str] = frozenset({"usdm_futures", "linear", "swap"})
PROJECT_SPOT_MARKET_TYPES: frozenset[str] = frozenset({"spot"})
PROJECT_SPREAD_SYMBOLS: frozenset[str] = frozenset({"BTCUSDT", "ETHUSDT", "RIVERUSDT"})
PROJECT_TIMEFRAMES: frozenset[str] = frozenset({"1m", "15m"})
PROJECT_DERIVATIVE_INSTRUMENT_TYPES: frozenset[InstrumentType] = frozenset(
    {InstrumentType.PERPETUAL, InstrumentType.FUTURES}
)


# ============================================================
# Helpers
# ============================================================

def _normalize_exchange(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip().lower()
    return normalized or None


def _normalize_symbol(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = (
        str(value)
        .replace("-", "")
        .replace("/", "")
        .replace("_", "")
        .upper()
        .strip()
    )
    return normalized or None


def _normalize_market_type(
    value: str | None,
    *,
    fallback: str = DEFAULT_PERPETUAL_MARKET_TYPE,
) -> str:
    normalized = str(value or fallback).strip().lower()
    return normalized or fallback


def _normalize_timeframe(value: str | None) -> str:
    normalized = str(value or DEFAULT_TIMEFRAME).strip()
    return normalized or DEFAULT_TIMEFRAME


def _normalize_exchange_set(
    values: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str]:
    if not values:
        return set()

    return {
        normalized
        for item in values
        if (normalized := _normalize_exchange(str(item))) is not None
    }


def _normalize_symbol_set(
    values: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str]:
    if not values:
        return set()

    return {
        normalized
        for item in values
        if (normalized := _normalize_symbol(str(item))) is not None
    }


def _normalize_market_type_set(
    values: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str]:
    if not values:
        return set()

    return {
        _normalize_market_type(str(item))
        for item in values
        if str(item).strip()
    }


def _normalize_timeframe_set(
    values: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str]:
    if not values:
        return set()

    return {
        _normalize_timeframe(str(item))
        for item in values
        if str(item).strip()
    }


def _normalize_topic_patterns(
    values: set[str] | list[str] | tuple[str, ...] | None,
    *,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    if not values:
        return fallback

    normalized = tuple(
        str(item).strip()
        for item in values
        if str(item).strip()
    )

    return normalized or fallback


def _normalize_instrument_type_set(
    values: set[InstrumentType | str] | list[InstrumentType | str] | tuple[InstrumentType | str, ...] | None,
) -> set[InstrumentType]:
    if not values:
        return set()

    return {
        parsed
        for item in values
        if (parsed := parse_instrument_type(item)) is not InstrumentType.UNKNOWN
    }


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_positive_decimal(name: str, value: Decimal) -> None:
    if value <= DECIMAL_ZERO:
        raise ValueError(f"{name} must be > 0")


def _validate_non_negative_decimal(name: str, value: Decimal) -> None:
    if value < DECIMAL_ZERO:
        raise ValueError(f"{name} must be >= 0")


def _validate_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_non_empty_str(name: str, value: str | None) -> None:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must not be empty")


# ============================================================
# Base Config
# ============================================================

@dataclass(slots=True)
class BaseSpreadConfig:
    """
    Базовий config-контракт для analytics.spreads.

    Production input flow:
        exchange adapters
            -> market.orderbook / market.funding
            -> OrderBookCache / FundingCache
            -> market.orderbook.updated / market.funding.updated
            -> analytics.spreads.*

    QuoteSnapshot у spreads залишається внутрішньою normalized top-of-book
    моделлю, але окремий QuoteCache не потрібен.
    """

    # Runtime
    enabled: bool = True
    service_name: str = "spread_analyzer"

    # Production input EventBus topics: data-layer updated events.
    orderbook_event_topic: str = DEFAULT_ORDERBOOK_EVENT_TOPIC
    funding_event_topic: str = DEFAULT_FUNDING_EVENT_TOPIC

    orderbook_event_topic_patterns: tuple[str, ...] = (DEFAULT_ORDERBOOK_EVENT_TOPIC,)
    funding_event_topic_patterns: tuple[str, ...] = (DEFAULT_FUNDING_EVENT_TOPIC,)

    # Backward-compatible aliases for old analyzer/base code.
    # До переписування base.py вони вже будуть вести на market.orderbook.updated.
    quote_event_topic: str = DEFAULT_ORDERBOOK_EVENT_TOPIC
    quote_event_topic_patterns: tuple[str, ...] = (DEFAULT_ORDERBOOK_EVENT_TOPIC,)

    # Legacy/raw topics. Вимкнені за замовчуванням.
    legacy_quote_event_topic: str = DEFAULT_LEGACY_QUOTE_EVENT_TOPIC
    raw_orderbook_event_topic: str = DEFAULT_RAW_ORDERBOOK_EVENT_TOPIC
    raw_quote_event_topic: str = DEFAULT_RAW_QUOTE_EVENT_TOPIC
    raw_funding_event_topic: str = DEFAULT_RAW_FUNDING_EVENT_TOPIC
    allow_legacy_quote_topics: bool = False
    allow_legacy_raw_topics: bool = False

    # Common output EventBus topics
    signal_event_topic: str = DEFAULT_SPREAD_SIGNAL_TOPIC
    analyzer_started_event_topic: str = DEFAULT_ANALYZER_STARTED_TOPIC
    analyzer_stopped_event_topic: str = DEFAULT_ANALYZER_STOPPED_TOPIC
    analyzer_heartbeat_event_topic: str = DEFAULT_ANALYZER_HEARTBEAT_TOPIC

    # Scoped defaults
    default_timeframe: str = "1m"

    # Optional scoped filters.
    # Filled with the project runtime scope by default so analyzers can bootstrap
    # without receiving empty allowlists from package-level config.
    allowed_symbols: set[str] = field(default_factory=lambda: set(PROJECT_SPREAD_SYMBOLS))
    allowed_timeframes: set[str] = field(default_factory=lambda: set(PROJECT_TIMEFRAMES))
    allowed_market_types: set[str] = field(default_factory=lambda: set(PROJECT_FUTURES_MARKET_TYPES))

    # Top-of-book freshness / alignment
    max_quote_age_ms: int = 2_000
    max_quote_skew_ms: int = 1_000

    # Rolling stats
    rolling_window_size: int = 500
    ema_alpha: Decimal = Decimal("0.2")

    # Emit throttling / signal cooldown
    min_emit_interval_ms: int = 250
    cooldown_seconds: int = 10

    # Scheduler / maintenance
    cleanup_interval_seconds: float = 30.0
    heartbeat_interval_seconds: float = 60.0
    stale_state_ttl_seconds: float = 300.0

    # Cache safety limits
    max_cached_quotes: int = 50_000
    max_cached_snapshots: int = 25_000
    max_cached_windows: int = 25_000

    # Signal thresholds
    anomaly_zscore_threshold: Decimal = Decimal("2.5")
    widening_bps_threshold: Decimal = Decimal("8")

    # Extensibility
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
        self.default_timeframe = _normalize_timeframe(self.default_timeframe)

        self.allowed_symbols = _normalize_symbol_set(self.allowed_symbols)
        self.allowed_timeframes = _normalize_timeframe_set(self.allowed_timeframes)
        self.allowed_market_types = _normalize_market_type_set(self.allowed_market_types)

        self.orderbook_event_topic_patterns = _normalize_topic_patterns(
            self.orderbook_event_topic_patterns,
            fallback=(self.orderbook_event_topic,),
        )
        self.funding_event_topic_patterns = _normalize_topic_patterns(
            self.funding_event_topic_patterns,
            fallback=(self.funding_event_topic,),
        )

        self.orderbook_event_topic = self.orderbook_event_topic_patterns[0]
        self.funding_event_topic = self.funding_event_topic_patterns[0]

        # Синхронізуємо старі quote-поля з production orderbook input.
        # Це дає можливість міняти base.py/analyzer-и поступово.
        if not self.allow_legacy_quote_topics:
            self.quote_event_topic = self.orderbook_event_topic
            self.quote_event_topic_patterns = self.orderbook_event_topic_patterns
        else:
            self.quote_event_topic_patterns = _normalize_topic_patterns(
                self.quote_event_topic_patterns,
                fallback=(self.quote_event_topic,),
            )
            self.quote_event_topic = self.quote_event_topic_patterns[0]

        self.metadata = dict(self.metadata or {})

        self.validate()

    def validate(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "validate", _analytics_args)
        except Exception:
            pass
        _validate_positive_int("max_quote_age_ms", self.max_quote_age_ms)
        _validate_positive_int("max_quote_skew_ms", self.max_quote_skew_ms)
        _validate_positive_int("rolling_window_size", self.rolling_window_size)
        _validate_non_negative_int("min_emit_interval_ms", self.min_emit_interval_ms)
        _validate_non_negative_int("cooldown_seconds", self.cooldown_seconds)

        _validate_positive_float("cleanup_interval_seconds", self.cleanup_interval_seconds)
        _validate_positive_float("heartbeat_interval_seconds", self.heartbeat_interval_seconds)
        _validate_positive_float("stale_state_ttl_seconds", self.stale_state_ttl_seconds)

        _validate_positive_int("max_cached_quotes", self.max_cached_quotes)
        _validate_positive_int("max_cached_snapshots", self.max_cached_snapshots)
        _validate_positive_int("max_cached_windows", self.max_cached_windows)

        _validate_positive_decimal("ema_alpha", self.ema_alpha)
        if self.ema_alpha > Decimal("1"):
            raise ValueError("ema_alpha must be <= 1")

        _validate_positive_decimal("anomaly_zscore_threshold", self.anomaly_zscore_threshold)
        _validate_positive_decimal("widening_bps_threshold", self.widening_bps_threshold)

        self._validate_topic("orderbook_event_topic", self.orderbook_event_topic)
        self._validate_topic("funding_event_topic", self.funding_event_topic)
        self._validate_topic("quote_event_topic", self.quote_event_topic)
        self._validate_topic("legacy_quote_event_topic", self.legacy_quote_event_topic)
        self._validate_topic("signal_event_topic", self.signal_event_topic)
        self._validate_topic("analyzer_started_event_topic", self.analyzer_started_event_topic)
        self._validate_topic("analyzer_stopped_event_topic", self.analyzer_stopped_event_topic)
        self._validate_topic("analyzer_heartbeat_event_topic", self.analyzer_heartbeat_event_topic)
        self._validate_topic("raw_orderbook_event_topic", self.raw_orderbook_event_topic)
        self._validate_topic("raw_quote_event_topic", self.raw_quote_event_topic)
        self._validate_topic("raw_funding_event_topic", self.raw_funding_event_topic)

        for topic in self.orderbook_event_topic_patterns:
            self._validate_topic("orderbook_event_topic_patterns item", topic)

        for topic in self.funding_event_topic_patterns:
            self._validate_topic("funding_event_topic_patterns item", topic)

        for topic in self.quote_event_topic_patterns:
            self._validate_topic("quote_event_topic_patterns item", topic)

        if not self.allow_legacy_raw_topics:
            production_topics = set(self.production_input_topics)
            raw_topics = {
                self.raw_orderbook_event_topic,
                self.raw_quote_event_topic,
                self.raw_funding_event_topic,
            }
            used_raw_topics = production_topics.intersection(raw_topics)

            if used_raw_topics:
                raise ValueError(
                    "Spread analyzer production input topics must use data-layer "
                    f"updated topics, not raw topics: {sorted(used_raw_topics)}"
                )

        if not self.allow_legacy_quote_topics:
            production_topics = set(self.production_price_input_topics)
            legacy_quote_topics = {
                self.legacy_quote_event_topic,
                DEFAULT_LEGACY_QUOTE_EVENT_TOPIC,
            }
            used_legacy_quote_topics = production_topics.intersection(legacy_quote_topics)

            if used_legacy_quote_topics:
                raise ValueError(
                    "Spread analyzer production price input must use "
                    "market.orderbook.updated from OrderBookCache, not legacy "
                    f"quote topics: {sorted(used_legacy_quote_topics)}"
                )

        _validate_non_empty_str("default_timeframe", self.default_timeframe)

    @staticmethod
    def _validate_topic(field_name: str, value: str) -> None:
        try:
            _analytics_class_name = "BaseSpreadConfig"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_topic", _analytics_args)
        except Exception:
            pass
        if not value or not value.strip():
            raise ValueError(f"{field_name} must not be empty")

    @property
    def production_price_input_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "production_price_input_topics", _analytics_args)
        except Exception:
            pass
        return self.orderbook_event_topic_patterns

    @property
    def production_input_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "production_input_topics", _analytics_args)
        except Exception:
            pass
        return (
            *self.orderbook_event_topic_patterns,
            *self.funding_event_topic_patterns,
        )

    @property
    def legacy_raw_input_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "legacy_raw_input_topics", _analytics_args)
        except Exception:
            pass
        return (
            self.raw_orderbook_event_topic,
            self.raw_quote_event_topic,
            self.raw_funding_event_topic,
        )

    @property
    def legacy_quote_input_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "legacy_quote_input_topics", _analytics_args)
        except Exception:
            pass
        return (self.legacy_quote_event_topic,)

    def should_process_scope(
        self,
        *,
        symbol: str,
        market_type: str,
        timeframe: str | None = None,
    ) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "should_process_scope", _analytics_args)
        except Exception:
            pass
        normalized_symbol = _normalize_symbol(symbol)
        normalized_market_type = _normalize_market_type(market_type)
        normalized_timeframe = _normalize_timeframe(timeframe or self.default_timeframe)

        if normalized_symbol is None:
            return False

        if self.allowed_symbols and normalized_symbol not in self.allowed_symbols:
            return False

        if self.allowed_market_types and normalized_market_type not in self.allowed_market_types:
            return False

        if self.allowed_timeframes and normalized_timeframe not in self.allowed_timeframes:
            return False

        return True

    def should_process_key(self, key: SpreadKey) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "should_process_key", _analytics_args)
        except Exception:
            pass
        scope = spread_key_to_dict(key)
        return self.should_process_scope(
            symbol=scope["symbol"],
            market_type=scope["market_type"],
            timeframe=scope["timeframe"],
        )

    def make_key(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str | None = None,
    ) -> SpreadKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "make_key", _analytics_args)
        except Exception:
            pass
        return make_spread_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe or self.default_timeframe,
        )

    def with_metadata(self, **metadata: Any) -> BaseSpreadConfig:
        """
        Повертає копію config з оновленим metadata.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "with_metadata", _analytics_args)
        except Exception:
            pass
        return replace(
            self,
            metadata={
                **self.metadata,
                **metadata,
            },
        )


# ============================================================
# Spot-Futures Config
# ============================================================

@dataclass(slots=True)
class SpotFuturesSpreadConfig(BaseSpreadConfig):
    """
    Config для SpotFuturesSpreadAnalyzer.

    Цей analyzer залишається production-компонентом:
    - spot leg: InstrumentType.SPOT / market_type="spot";
    - futures leg: InstrumentType.PERPETUAL або InstrumentType.FUTURES;
    - funding leg: зазвичай futures/perpetual market_type.

    Price/top-of-book дані мають приходити тільки через:
        market.orderbook.updated
    Funding context має приходити через:
        market.funding.updated
    """

    service_name: str = "spot_futures_spread_analyzer"

    # Output topics
    snapshot_event_topic: str = DEFAULT_SPOT_FUTURES_SNAPSHOT_TOPIC

    # Strategy/signal thresholds
    mean_reversion_zscore_threshold: Decimal = Decimal("2.0")
    regime_shift_zscore_threshold: Decimal = Decimal("3.0")

    # Funding adjustment
    notional_for_funding_adjustment: Decimal | None = None

    # Optional filters/default routing
    default_spot_exchange: str | None = PROJECT_DEFAULT_EXCHANGE
    default_futures_exchange: str | None = PROJECT_DEFAULT_EXCHANGE

    default_spot_market_type: str = DEFAULT_SPOT_MARKET_TYPE
    default_futures_market_type: str = DEFAULT_PERPETUAL_MARKET_TYPE

    allowed_spot_exchanges: set[str] = field(default_factory=lambda: set(PROJECT_EXCHANGES))
    allowed_futures_exchanges: set[str] = field(default_factory=lambda: set(PROJECT_EXCHANGES))

    allowed_spot_market_types: set[str] = field(
        default_factory=lambda: {DEFAULT_SPOT_MARKET_TYPE}
    )
    allowed_futures_market_types: set[str] = field(
        default_factory=lambda: set(PROJECT_FUTURES_MARKET_TYPES)
    )

    allowed_spot_instrument_types: set[InstrumentType] = field(
        default_factory=lambda: {InstrumentType.SPOT}
    )
    allowed_futures_instrument_types: set[InstrumentType] = field(
        default_factory=lambda: {
            InstrumentType.PERPETUAL,
            InstrumentType.FUTURES,
        }
    )

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
        self.allowed_spot_exchanges = _normalize_exchange_set(self.allowed_spot_exchanges)
        self.allowed_futures_exchanges = _normalize_exchange_set(self.allowed_futures_exchanges)

        self.allowed_spot_market_types = _normalize_market_type_set(
            self.allowed_spot_market_types
        )
        self.allowed_futures_market_types = _normalize_market_type_set(
            self.allowed_futures_market_types
        )

        self.allowed_spot_instrument_types = _normalize_instrument_type_set(
            self.allowed_spot_instrument_types
        )
        self.allowed_futures_instrument_types = _normalize_instrument_type_set(
            self.allowed_futures_instrument_types
        )

        self.default_spot_exchange = _normalize_exchange(self.default_spot_exchange)
        self.default_futures_exchange = _normalize_exchange(self.default_futures_exchange)
        self.default_spot_market_type = _normalize_market_type(
            self.default_spot_market_type,
            fallback=DEFAULT_SPOT_MARKET_TYPE,
        )
        self.default_futures_market_type = _normalize_market_type(
            self.default_futures_market_type,
            fallback=DEFAULT_PERPETUAL_MARKET_TYPE,
        )

        if not self.allowed_spot_market_types:
            raise ValueError("allowed_spot_market_types must not be empty")
        if not self.allowed_futures_market_types:
            raise ValueError("allowed_futures_market_types must not be empty")
        if not self.allowed_spot_instrument_types:
            raise ValueError("allowed_spot_instrument_types must not be empty")
        if not self.allowed_futures_instrument_types:
            raise ValueError("allowed_futures_instrument_types must not be empty")

        if InstrumentType.UNKNOWN in self.allowed_spot_instrument_types:
            raise ValueError("allowed_spot_instrument_types must not include UNKNOWN")
        if InstrumentType.UNKNOWN in self.allowed_futures_instrument_types:
            raise ValueError("allowed_futures_instrument_types must not include UNKNOWN")

        if any(item is not InstrumentType.SPOT for item in self.allowed_spot_instrument_types):
            raise ValueError(
                "allowed_spot_instrument_types must include only InstrumentType.SPOT"
            )

        if any(not item.is_derivative for item in self.allowed_futures_instrument_types):
            raise ValueError(
                "allowed_futures_instrument_types must include only derivative instrument types"
            )

        # Не використовуємо zero-argument super() у dataclass(slots=True) inheritance.
        BaseSpreadConfig.__post_init__(self)

        self._validate_topic("snapshot_event_topic", self.snapshot_event_topic)

        _validate_positive_decimal(
            "mean_reversion_zscore_threshold",
            self.mean_reversion_zscore_threshold,
        )
        _validate_positive_decimal(
            "regime_shift_zscore_threshold",
            self.regime_shift_zscore_threshold,
        )

        if self.notional_for_funding_adjustment is not None:
            _validate_positive_decimal(
                "notional_for_funding_adjustment",
                self.notional_for_funding_adjustment,
            )

    def is_spot_exchange_allowed(self, exchange: str) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_spot_exchange_allowed", _analytics_args)
        except Exception:
            pass
        normalized = _normalize_exchange(exchange)
        if normalized is None:
            return False
        return not self.allowed_spot_exchanges or normalized in self.allowed_spot_exchanges

    def is_futures_exchange_allowed(self, exchange: str) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_futures_exchange_allowed", _analytics_args)
        except Exception:
            pass
        normalized = _normalize_exchange(exchange)
        if normalized is None:
            return False
        return (
            not self.allowed_futures_exchanges
            or normalized in self.allowed_futures_exchanges
        )

    def is_spot_market_type_allowed(self, market_type: str | None) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_spot_market_type_allowed", _analytics_args)
        except Exception:
            pass
        normalized = _normalize_market_type(
            market_type,
            fallback=self.default_spot_market_type,
        )
        return (
            not self.allowed_spot_market_types
            or normalized in self.allowed_spot_market_types
        )

    def is_futures_market_type_allowed(self, market_type: str | None) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_futures_market_type_allowed", _analytics_args)
        except Exception:
            pass
        normalized = _normalize_market_type(
            market_type,
            fallback=self.default_futures_market_type,
        )
        return (
            not self.allowed_futures_market_types
            or normalized in self.allowed_futures_market_types
        )

    def is_spot_instrument_type_allowed(
        self,
        instrument_type: InstrumentType | str | None,
    ) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_spot_instrument_type_allowed", _analytics_args)
        except Exception:
            pass
        parsed = parse_instrument_type(instrument_type)
        return parsed in self.allowed_spot_instrument_types

    def is_futures_instrument_type_allowed(
        self,
        instrument_type: InstrumentType | str | None,
    ) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_futures_instrument_type_allowed", _analytics_args)
        except Exception:
            pass
        parsed = parse_instrument_type(instrument_type)
        return parsed in self.allowed_futures_instrument_types

    def is_spot_quote_allowed(
        self,
        *,
        exchange: str,
        market_type: str | None,
        instrument_type: InstrumentType | str | None,
        symbol: str,
        timeframe: str | None = None,
    ) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_spot_quote_allowed", _analytics_args)
        except Exception:
            pass
        return (
            self.is_spot_exchange_allowed(exchange)
            and self.is_spot_market_type_allowed(market_type)
            and self.is_spot_instrument_type_allowed(instrument_type)
            and self.should_process_scope(
                symbol=symbol,
                market_type=market_type or self.default_spot_market_type,
                timeframe=timeframe or self.default_timeframe,
            )
        )

    def is_futures_quote_allowed(
        self,
        *,
        exchange: str,
        market_type: str | None,
        instrument_type: InstrumentType | str | None,
        symbol: str,
        timeframe: str | None = None,
    ) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_futures_quote_allowed", _analytics_args)
        except Exception:
            pass
        return (
            self.is_futures_exchange_allowed(exchange)
            and self.is_futures_market_type_allowed(market_type)
            and self.is_futures_instrument_type_allowed(instrument_type)
            and self.should_process_scope(
                symbol=symbol,
                market_type=market_type or self.default_futures_market_type,
                timeframe=timeframe or self.default_timeframe,
            )
        )


# ============================================================
# Cross-Exchange Config
# ============================================================

@dataclass(slots=True)
class CrossExchangeSpreadConfig(BaseSpreadConfig):
    """
    Config для CrossExchangeSpreadAnalyzer.

    Cross-exchange spread/arbitrage може працювати як зі spot, так і з
    perpetual/futures venues. Price source так само має бути top-of-book
    із OrderBookCache через market.orderbook.updated.
    """

    service_name: str = "cross_exchange_spread_analyzer"

    # Output topics
    snapshot_event_topic: str = DEFAULT_CROSS_EXCHANGE_SNAPSHOT_TOPIC
    opportunity_event_topic: str = DEFAULT_ARBITRAGE_OPPORTUNITY_TOPIC

    # Arbitrage threshold
    arbitrage_min_bps: Decimal = Decimal("5")

    # Trade sizing
    default_trade_size: Decimal = Decimal("1")
    min_trade_size: Decimal | None = None
    max_trade_size: Decimal | None = None

    # Cost model
    slippage_max_bps: Decimal = Decimal("5")
    safety_buffer_bps: Decimal = Decimal("1")
    default_taker_fee_rate: Decimal = Decimal("0.001")
    default_maker_fee_rate: Decimal = Decimal("0.0005")

    # Opportunity lifecycle
    opportunity_ttl_seconds: float = 10.0
    max_cached_opportunities: int = 10_000

    # Filters
    allowed_instrument_types: set[InstrumentType] = field(
        default_factory=lambda: set(PROJECT_DERIVATIVE_INSTRUMENT_TYPES)
    )
    allowed_exchanges: set[str] = field(default_factory=lambda: set(PROJECT_EXCHANGES))
    preferred_exchanges: set[str] = field(default_factory=lambda: set(PROJECT_EXCHANGES))

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
        self.allowed_exchanges = _normalize_exchange_set(self.allowed_exchanges)
        self.preferred_exchanges = _normalize_exchange_set(self.preferred_exchanges)

        self.allowed_instrument_types = _normalize_instrument_type_set(
            self.allowed_instrument_types
        )

        if not self.allowed_instrument_types:
            raise ValueError("allowed_instrument_types must not be empty")

        if InstrumentType.UNKNOWN in self.allowed_instrument_types:
            raise ValueError(
                "allowed_instrument_types must not include InstrumentType.UNKNOWN"
            )

        # Не використовуємо zero-argument super() у dataclass(slots=True) inheritance.
        BaseSpreadConfig.__post_init__(self)

        self._validate_topic("snapshot_event_topic", self.snapshot_event_topic)
        self._validate_topic("opportunity_event_topic", self.opportunity_event_topic)

        _validate_positive_decimal("arbitrage_min_bps", self.arbitrage_min_bps)
        _validate_positive_decimal("default_trade_size", self.default_trade_size)

        _validate_non_negative_decimal("slippage_max_bps", self.slippage_max_bps)
        _validate_non_negative_decimal("safety_buffer_bps", self.safety_buffer_bps)
        _validate_non_negative_decimal("default_taker_fee_rate", self.default_taker_fee_rate)
        _validate_non_negative_decimal("default_maker_fee_rate", self.default_maker_fee_rate)

        _validate_positive_float("opportunity_ttl_seconds", self.opportunity_ttl_seconds)
        _validate_positive_int("max_cached_opportunities", self.max_cached_opportunities)

        if self.min_trade_size is not None:
            _validate_positive_decimal("min_trade_size", self.min_trade_size)

        if self.max_trade_size is not None:
            _validate_positive_decimal("max_trade_size", self.max_trade_size)

        if (
            self.min_trade_size is not None
            and self.max_trade_size is not None
            and self.min_trade_size > self.max_trade_size
        ):
            raise ValueError("min_trade_size must be <= max_trade_size")

        if self.default_trade_size is not None:
            if (
                self.min_trade_size is not None
                and self.default_trade_size < self.min_trade_size
            ):
                raise ValueError("default_trade_size must be >= min_trade_size")

            if (
                self.max_trade_size is not None
                and self.default_trade_size > self.max_trade_size
            ):
                raise ValueError("default_trade_size must be <= max_trade_size")

    def is_exchange_allowed(self, exchange: str) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_exchange_allowed", _analytics_args)
        except Exception:
            pass
        normalized = _normalize_exchange(exchange)
        if normalized is None:
            return False

        return not self.allowed_exchanges or normalized in self.allowed_exchanges

    def is_exchange_preferred(self, exchange: str) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_exchange_preferred", _analytics_args)
        except Exception:
            pass
        normalized = _normalize_exchange(exchange)
        if normalized is None:
            return False

        return not self.preferred_exchanges or normalized in self.preferred_exchanges

    def is_instrument_type_allowed(
        self,
        instrument_type: InstrumentType | str | None,
    ) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_instrument_type_allowed", _analytics_args)
        except Exception:
            pass
        parsed = parse_instrument_type(instrument_type)
        return parsed in self.allowed_instrument_types

    def is_quote_allowed(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        instrument_type: InstrumentType | str | None,
        timeframe: str | None = None,
    ) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_quote_allowed", _analytics_args)
        except Exception:
            pass
        return (
            self.is_exchange_allowed(exchange)
            and self.is_instrument_type_allowed(instrument_type)
            and self.should_process_scope(
                symbol=symbol,
                market_type=market_type,
                timeframe=timeframe or self.default_timeframe,
            )
        )

    def fee_rates_from_metadata(self) -> Mapping[str, Any]:
        """
        Повертає fee override map з metadata.

        Очікуваний формат:
            metadata = {
                "fee_rates": {
                    "binance": {"buy": "0.001", "sell": "0.001"},
                    "bybit": {"buy": "0.0008", "sell": "0.0008"},
                }
            }
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "fee_rates_from_metadata", _analytics_args)
        except Exception:
            pass
        value = self.metadata.get("fee_rates", {})
        if isinstance(value, Mapping):
            return value
        return {}


__all__ = [
    "PROJECT_DEFAULT_EXCHANGE",
    "PROJECT_EXCHANGES",
    "PROJECT_FUTURES_MARKET_TYPES",
    "PROJECT_SPOT_MARKET_TYPES",
    "PROJECT_SPREAD_SYMBOLS",
    "PROJECT_TIMEFRAMES",
    "PROJECT_DERIVATIVE_INSTRUMENT_TYPES",
    "DEFAULT_ORDERBOOK_EVENT_TOPIC",
    "DEFAULT_QUOTE_EVENT_TOPIC",
    "DEFAULT_FUNDING_EVENT_TOPIC",
    "DEFAULT_LEGACY_QUOTE_EVENT_TOPIC",
    "DEFAULT_RAW_ORDERBOOK_EVENT_TOPIC",
    "DEFAULT_RAW_QUOTE_EVENT_TOPIC",
    "DEFAULT_RAW_FUNDING_EVENT_TOPIC",
    "DEFAULT_SPOT_FUTURES_SNAPSHOT_TOPIC",
    "DEFAULT_CROSS_EXCHANGE_SNAPSHOT_TOPIC",
    "DEFAULT_SPREAD_SIGNAL_TOPIC",
    "DEFAULT_ARBITRAGE_OPPORTUNITY_TOPIC",
    "DEFAULT_ANALYZER_STARTED_TOPIC",
    "DEFAULT_ANALYZER_STOPPED_TOPIC",
    "DEFAULT_ANALYZER_HEARTBEAT_TOPIC",
    "BaseSpreadConfig",
    "SpotFuturesSpreadConfig",
    "CrossExchangeSpreadConfig",
]