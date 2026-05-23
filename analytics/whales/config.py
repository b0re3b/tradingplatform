from __future__ import annotations
from core.logger import get_logger

from dataclasses import dataclass, field
from typing import Mapping

from analytics.whales.models import (
    WhaleKey,
    make_whale_key,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    whale_key_to_dict,
)

# =============================================================================
# Canonical topics
# =============================================================================

# Production data-layer input topics.
# Важливо: це updated/cache-layer events, а не raw exchange-adapter events.
DEFAULT_TRADES_UPDATED_TOPIC = "market.trades.updated"
DEFAULT_LIQUIDATIONS_UPDATED_TOPIC = "market.liquidations.updated"

# Legacy/raw topics. Не використовувати в production, якщо allow_legacy_raw_topics=False.
DEFAULT_RAW_TRADE_TOPIC = "market.trade"
DEFAULT_RAW_LIQUIDATION_TOPIC = "market.liquidation"

# Analytics output topics.
DEFAULT_LARGE_TRADE_TOPIC = "analytics.whales.large_trade"
DEFAULT_WHALE_ACTIVITY_TOPIC = "analytics.whales.whale_activity"
DEFAULT_WHALE_PRESSURE_TOPIC = "analytics.whales.whale_pressure"
DEFAULT_WHALE_LIQUIDATION_CONTEXT_TOPIC = "analytics.whales.whale_liquidation_context"
DEFAULT_WHALE_CLUSTER_TOPIC = "analytics.whales.whale_cluster"
DEFAULT_WHALE_CLUSTER_UPDATE_TOPIC = "analytics.whales.whale_cluster_update"
DEFAULT_WHALE_CLUSTER_EXHAUSTION_TOPIC = "analytics.whales.whale_cluster_exhaustion"

# Project runtime scope.
# Market data exchanges used by the project. Bitget is intentionally not included.
PROJECT_EXCHANGES: set[str] = {"binance", "bybit", "okx", "mexc"}
PROJECT_DEFAULT_EXCHANGE: str = "binance"

# Futures/perpetual market types used across project adapters/normalizers.
PROJECT_FUTURES_MARKET_TYPES: set[str] = {"usdm_futures", "linear", "swap"}
PROJECT_DEFAULT_MARKET_TYPE: str = "usdm_futures"

# Backtest/runtime symbols and timeframes used by the project.
PROJECT_SYMBOLS: set[str] = {"BTCUSDT", "ETHUSDT", "RIVERUSDT"}
PROJECT_TIMEFRAMES: set[str] = {"1m", "15m"}
PROJECT_DEFAULT_TIMEFRAME: str = "1m"

# Whale detector defaults tuned as safe notional gates per tracked symbol.
# Values are intentionally conservative and can be overridden from runtime config.
PROJECT_SYMBOL_ABS_THRESHOLDS: dict[str, float] = {
    "BTCUSDT": 250_000.0,
    "ETHUSDT": 150_000.0,
    "RIVERUSDT": 25_000.0,
}

PROJECT_SYMBOL_COOLDOWNS_SEC: dict[str, float] = {
    "BTCUSDT": 2.0,
    "ETHUSDT": 2.0,
    "RIVERUSDT": 3.0,
}



# =============================================================================
# Validation / normalization helpers
# =============================================================================

def _validate_positive_number(name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_non_negative_number(name: str, value: float | int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_min_int(name: str, value: int, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in range [0, 1]")


def _validate_non_empty_topic(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty event topic string")


def _normalize_topic_patterns(
    values: tuple[str, ...] | list[str] | set[str] | None,
    *,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    if not values:
        return fallback

    normalized = tuple(str(item).strip() for item in values if str(item).strip())
    return normalized or fallback


def _normalize_symbol_set(values: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    if not values:
        return set()
    return {normalize_symbol(item) for item in values if str(item).strip()}


def _normalize_exchange_set(values: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    if not values:
        return set()
    return {normalize_exchange(item) for item in values if str(item).strip()}


def _normalize_market_type_set(values: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    if not values:
        return set()
    return {normalize_market_type(item) for item in values if str(item).strip()}


def _normalize_timeframe_set(values: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    if not values:
        return set()
    return {normalize_timeframe(item) for item in values if str(item).strip()}


def _normalize_threshold_mapping(values: Mapping[str, float] | None) -> dict[str, float]:
    if not values:
        return {}

    normalized: dict[str, float] = {}
    for key, value in values.items():
        normalized_key = _normalize_threshold_key(key)
        normalized[normalized_key] = float(value)

    return normalized


def _normalize_threshold_key(key: str) -> str:
    """
    Підтримує два формати:
    - symbol-only: BTCUSDT
    - scoped: exchange:market_type:symbol:timeframe
    """
    raw = str(key).strip()
    if not raw:
        raise ValueError("mapping key must be non-empty")

    parts = raw.split(":")
    if len(parts) == 4:
        whale_key = make_whale_key(
            exchange=parts[0],
            market_type=parts[1],
            symbol=parts[2],
            timeframe=parts[3],
        )
        scope = whale_key_to_dict(whale_key)
        return (
            f"{scope['exchange']}:"
            f"{scope['market_type']}:"
            f"{scope['symbol']}:"
            f"{scope['timeframe']}"
        )

    return normalize_symbol(raw)


def _validate_non_negative_mapping(name: str, values: Mapping[str, float]) -> None:
    for key, value in values.items():
        if not key or not isinstance(key, str):
            raise ValueError(f"{name} keys must be non-empty strings")
        _validate_non_negative_number(f"{name}[{key!r}]", value)


def _validate_positive_mapping(name: str, values: Mapping[str, float]) -> None:
    for key, value in values.items():
        if not key or not isinstance(key, str):
            raise ValueError(f"{name} keys must be non-empty strings")
        _validate_positive_number(f"{name}[{key!r}]", value)


# =============================================================================
# Shared scoped config mixin
# =============================================================================

@dataclass(slots=True)
class WhaleScopedConfigMixin:
    """
    Спільні scoped-фільтри для analytics.whales.

    Scope:
        exchange + market_type + symbol + timeframe

    Якщо allowlist порожній — значення не фільтрується.
    """

    default_exchange: str = PROJECT_DEFAULT_EXCHANGE
    default_market_type: str = PROJECT_DEFAULT_MARKET_TYPE
    default_timeframe: str = PROJECT_DEFAULT_TIMEFRAME

    allowed_exchanges: set[str] = field(default_factory=set)
    allowed_market_types: set[str] = field(
        default_factory=lambda: set(PROJECT_FUTURES_MARKET_TYPES)
    )
    allowed_symbols: set[str] = field(default_factory=lambda: set(PROJECT_SYMBOLS))
    allowed_timeframes: set[str] = field(default_factory=lambda: set(PROJECT_TIMEFRAMES))

    def _normalize_scope_fields(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_scope_fields", _analytics_args)
        except Exception:
            pass
        self.default_exchange = normalize_exchange(self.default_exchange)
        self.default_market_type = normalize_market_type(self.default_market_type)
        self.default_timeframe = normalize_timeframe(self.default_timeframe)

        self.allowed_exchanges = _normalize_exchange_set(self.allowed_exchanges)
        self.allowed_market_types = _normalize_market_type_set(self.allowed_market_types)
        self.allowed_symbols = _normalize_symbol_set(self.allowed_symbols)
        self.allowed_timeframes = _normalize_timeframe_set(self.allowed_timeframes)

    def make_key(
        self,
        *,
        exchange: str | None,
        market_type: str | None,
        symbol: str,
        timeframe: str | None = None,
    ) -> WhaleKey:
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
        return make_whale_key(
            exchange=exchange or self.default_exchange,
            market_type=market_type or self.default_market_type,
            symbol=symbol,
            timeframe=timeframe or self.default_timeframe,
        )

    def should_process_scope(
        self,
        *,
        exchange: str | None,
        market_type: str | None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "should_process_scope", _analytics_args)
        except Exception:
            pass
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        return self.should_process_key(key)

    def should_process_key(self, key: WhaleKey) -> bool:
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
        scope = whale_key_to_dict(key)

        if self.allowed_exchanges and scope["exchange"] not in self.allowed_exchanges:
            return False

        if self.allowed_market_types and scope["market_type"] not in self.allowed_market_types:
            return False

        if self.allowed_symbols and scope["symbol"] not in self.allowed_symbols:
            return False

        if self.allowed_timeframes and scope["timeframe"] not in self.allowed_timeframes:
            return False

        return True

    @staticmethod
    def scoped_mapping_key(key: WhaleKey) -> str:
        try:
            _analytics_class_name = "WhaleScopedConfigMixin"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "scoped_mapping_key", _analytics_args)
        except Exception:
            pass
        scope = whale_key_to_dict(key)
        return (
            f"{scope['exchange']}:"
            f"{scope['market_type']}:"
            f"{scope['symbol']}:"
            f"{scope['timeframe']}"
        )


# =============================================================================
# Large trade detector config
# =============================================================================

@dataclass(slots=True)
class LargeTradeDetectorConfig(WhaleScopedConfigMixin):
    """
    Config для low-level detector великих трейдів.

    Production input:
        TradesCache -> market.trades.updated -> LargeTradeDetector

    Legacy raw input:
        market.trade

    Raw input дозволений тільки якщо allow_legacy_raw_topics=True.
    """

    enabled: bool = True

    # Absolute thresholds
    default_abs_notional_threshold: float = 100_000.0

    # Backward-compatible symbol-only thresholds:
    # {"BTCUSDT": 250000}
    symbol_abs_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(PROJECT_SYMBOL_ABS_THRESHOLDS)
    )

    # New scoped thresholds:
    # {"binance:perpetual:BTCUSDT:realtime": 250000}
    scoped_abs_thresholds: dict[str, float] = field(default_factory=dict)

    # Relative detection
    use_relative_detection: bool = True
    rolling_window_size: int = 300
    min_samples_for_relative_detection: int = 30
    zscore_threshold: float = 3.0

    # Basic filters
    min_notional_filter: float = 10_000.0
    side_filter: str | None = None  # "buy" / "sell" / None

    # Cooldowns
    signal_cooldown_sec: float = 2.0
    symbol_cooldown_sec: dict[str, float] = field(
        default_factory=lambda: dict(PROJECT_SYMBOL_COOLDOWNS_SEC)
    )
    scoped_cooldown_sec: dict[str, float] = field(default_factory=dict)

    # Cleanup / lifecycle
    cleanup_interval_sec: float = 60.0
    stats_ttl_sec: float = 60.0 * 60.0
    recalibration_interval: int = 2_000

    # Production EventBus topics
    input_event_name: str = DEFAULT_TRADES_UPDATED_TOPIC
    input_event_patterns: tuple[str, ...] = (DEFAULT_TRADES_UPDATED_TOPIC,)
    output_event_name: str = DEFAULT_LARGE_TRADE_TOPIC

    # Legacy/raw EventBus topics
    raw_input_event_name: str = DEFAULT_RAW_TRADE_TOPIC
    allow_legacy_raw_topics: bool = False

    # Behavior
    emit_on_bus: bool = True
    log_signals: bool = True

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
        self._normalize_scope_fields()

        self.symbol_abs_thresholds = _normalize_threshold_mapping(self.symbol_abs_thresholds)
        self.scoped_abs_thresholds = _normalize_threshold_mapping(self.scoped_abs_thresholds)
        self.symbol_cooldown_sec = _normalize_threshold_mapping(self.symbol_cooldown_sec)
        self.scoped_cooldown_sec = _normalize_threshold_mapping(self.scoped_cooldown_sec)

        self.input_event_patterns = _normalize_topic_patterns(
            self.input_event_patterns,
            fallback=(self.input_event_name,),
        )
        self.input_event_name = self.input_event_patterns[0]

        self.validate()

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
        return self.input_event_patterns

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
        return (self.raw_input_event_name,)

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
        _validate_non_negative_number(
            "large_trade_detector.default_abs_notional_threshold",
            self.default_abs_notional_threshold,
        )
        _validate_non_negative_mapping(
            "large_trade_detector.symbol_abs_thresholds",
            self.symbol_abs_thresholds,
        )
        _validate_non_negative_mapping(
            "large_trade_detector.scoped_abs_thresholds",
            self.scoped_abs_thresholds,
        )

        _validate_min_int(
            "large_trade_detector.rolling_window_size",
            self.rolling_window_size,
            minimum=2,
        )
        _validate_min_int(
            "large_trade_detector.min_samples_for_relative_detection",
            self.min_samples_for_relative_detection,
            minimum=2,
        )
        if self.min_samples_for_relative_detection > self.rolling_window_size:
            raise ValueError(
                "large_trade_detector.min_samples_for_relative_detection "
                "must be <= rolling_window_size"
            )

        _validate_non_negative_number(
            "large_trade_detector.zscore_threshold",
            self.zscore_threshold,
        )
        _validate_non_negative_number(
            "large_trade_detector.min_notional_filter",
            self.min_notional_filter,
        )

        if self.side_filter not in {None, "buy", "sell"}:
            raise ValueError(
                "large_trade_detector.side_filter must be one of: None, 'buy', 'sell'"
            )

        _validate_non_negative_number(
            "large_trade_detector.signal_cooldown_sec",
            self.signal_cooldown_sec,
        )
        _validate_non_negative_mapping(
            "large_trade_detector.symbol_cooldown_sec",
            self.symbol_cooldown_sec,
        )
        _validate_non_negative_mapping(
            "large_trade_detector.scoped_cooldown_sec",
            self.scoped_cooldown_sec,
        )

        _validate_positive_number(
            "large_trade_detector.cleanup_interval_sec",
            self.cleanup_interval_sec,
        )
        _validate_positive_number(
            "large_trade_detector.stats_ttl_sec",
            self.stats_ttl_sec,
        )
        _validate_positive_int(
            "large_trade_detector.recalibration_interval",
            self.recalibration_interval,
        )

        for topic in self.input_event_patterns:
            _validate_non_empty_topic("large_trade_detector.input_event_patterns item", topic)

        _validate_non_empty_topic(
            "large_trade_detector.input_event_name",
            self.input_event_name,
        )
        _validate_non_empty_topic(
            "large_trade_detector.output_event_name",
            self.output_event_name,
        )
        _validate_non_empty_topic(
            "large_trade_detector.raw_input_event_name",
            self.raw_input_event_name,
        )

        if not self.allow_legacy_raw_topics and self.raw_input_event_name in self.input_event_patterns:
            raise ValueError(
                "large_trade_detector.input_event_patterns must use data-layer updated "
                "topics in production, not raw market.trade"
            )

    def get_symbol_abs_threshold(self, symbol: str) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_symbol_abs_threshold", _analytics_args)
        except Exception:
            pass
        normalized_symbol = normalize_symbol(symbol)
        return self.symbol_abs_thresholds.get(
            normalized_symbol,
            self.default_abs_notional_threshold,
        )

    def get_key_abs_threshold(self, key: WhaleKey) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_key_abs_threshold", _analytics_args)
        except Exception:
            pass
        scoped_key = self.scoped_mapping_key(key)
        if scoped_key in self.scoped_abs_thresholds:
            return self.scoped_abs_thresholds[scoped_key]

        scope = whale_key_to_dict(key)
        return self.get_symbol_abs_threshold(scope["symbol"])

    def get_symbol_cooldown(self, symbol: str) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_symbol_cooldown", _analytics_args)
        except Exception:
            pass
        normalized_symbol = normalize_symbol(symbol)
        return self.symbol_cooldown_sec.get(
            normalized_symbol,
            self.signal_cooldown_sec,
        )

    def get_key_cooldown(self, key: WhaleKey) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_key_cooldown", _analytics_args)
        except Exception:
            pass
        scoped_key = self.scoped_mapping_key(key)
        if scoped_key in self.scoped_cooldown_sec:
            return self.scoped_cooldown_sec[scoped_key]

        scope = whale_key_to_dict(key)
        return self.get_symbol_cooldown(scope["symbol"])


# =============================================================================
# Whale tracker config
# =============================================================================

@dataclass(slots=True)
class WhaleTrackerConfig(WhaleScopedConfigMixin):
    """
    Config для high-level whale activity / pressure / liquidation context tracker.

    Production input:
        analytics.whales.large_trade
        market.liquidations.updated або analytics.liquidations.*

    Legacy raw liquidation input:
        market.liquidation

    Raw liquidation input дозволений тільки якщо allow_legacy_raw_topics=True.
    """

    enabled: bool = True

    # Input events
    large_trade_event_name: str = DEFAULT_LARGE_TRADE_TOPIC
    liquidation_event_name: str = DEFAULT_LIQUIDATIONS_UPDATED_TOPIC
    liquidation_event_patterns: tuple[str, ...] = (DEFAULT_LIQUIDATIONS_UPDATED_TOPIC,)

    raw_liquidation_event_name: str = DEFAULT_RAW_LIQUIDATION_TOPIC
    allow_legacy_raw_topics: bool = False

    # Output events
    whale_activity_event_name: str = DEFAULT_WHALE_ACTIVITY_TOPIC
    whale_pressure_event_name: str = DEFAULT_WHALE_PRESSURE_TOPIC
    whale_liquidation_context_event_name: str = DEFAULT_WHALE_LIQUIDATION_CONTEXT_TOPIC

    # Windows
    cluster_window_sec: int = 30
    pressure_window_sec: int = 60
    liquidation_window_sec: int = 60

    # Thresholds
    cluster_min_trades: int = 3
    cluster_min_total_notional: float = 300_000.0

    pressure_min_trades: int = 4
    pressure_min_total_notional: float = 500_000.0
    pressure_imbalance_ratio_threshold: float = 0.65

    liquidation_context_min_notional: float = 100_000.0

    # Cooldowns
    whale_activity_cooldown_sec: float = 5.0
    whale_pressure_cooldown_sec: float = 5.0
    whale_liquidation_context_cooldown_sec: float = 5.0

    # Cleanup
    cleanup_interval_sec: float = 60.0
    stats_ttl_sec: float = 60.0 * 60.0

    # Behavior
    emit_on_bus: bool = True
    log_signals: bool = True
    subscribe_liquidations: bool = True

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
        self._normalize_scope_fields()

        self.liquidation_event_patterns = _normalize_topic_patterns(
            self.liquidation_event_patterns,
            fallback=(self.liquidation_event_name,),
        )
        self.liquidation_event_name = self.liquidation_event_patterns[0]

        self.validate()

    @property
    def large_trade_buffer_size(self) -> int:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "large_trade_buffer_size", _analytics_args)
        except Exception:
            pass
        return max(self.cluster_window_sec, self.pressure_window_sec) * 10

    @property
    def liquidation_buffer_size(self) -> int:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "liquidation_buffer_size", _analytics_args)
        except Exception:
            pass
        return self.liquidation_window_sec * 10

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
        topics = [self.large_trade_event_name]
        if self.subscribe_liquidations:
            topics.extend(self.liquidation_event_patterns)
        return tuple(topics)

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
        return (self.raw_liquidation_event_name,)

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
        _validate_non_empty_topic(
            "whale_tracker.large_trade_event_name",
            self.large_trade_event_name,
        )

        for topic in self.liquidation_event_patterns:
            _validate_non_empty_topic("whale_tracker.liquidation_event_patterns item", topic)

        _validate_non_empty_topic(
            "whale_tracker.liquidation_event_name",
            self.liquidation_event_name,
        )
        _validate_non_empty_topic(
            "whale_tracker.raw_liquidation_event_name",
            self.raw_liquidation_event_name,
        )
        _validate_non_empty_topic(
            "whale_tracker.whale_activity_event_name",
            self.whale_activity_event_name,
        )
        _validate_non_empty_topic(
            "whale_tracker.whale_pressure_event_name",
            self.whale_pressure_event_name,
        )
        _validate_non_empty_topic(
            "whale_tracker.whale_liquidation_context_event_name",
            self.whale_liquidation_context_event_name,
        )

        if (
            not self.allow_legacy_raw_topics
            and self.raw_liquidation_event_name in self.liquidation_event_patterns
        ):
            raise ValueError(
                "whale_tracker.liquidation_event_patterns must use data-layer updated "
                "topics in production, not raw market.liquidation"
            )

        _validate_positive_int("whale_tracker.cluster_window_sec", self.cluster_window_sec)
        _validate_positive_int("whale_tracker.pressure_window_sec", self.pressure_window_sec)
        _validate_positive_int(
            "whale_tracker.liquidation_window_sec",
            self.liquidation_window_sec,
        )

        _validate_positive_int("whale_tracker.cluster_min_trades", self.cluster_min_trades)
        _validate_non_negative_number(
            "whale_tracker.cluster_min_total_notional",
            self.cluster_min_total_notional,
        )

        _validate_positive_int("whale_tracker.pressure_min_trades", self.pressure_min_trades)
        _validate_non_negative_number(
            "whale_tracker.pressure_min_total_notional",
            self.pressure_min_total_notional,
        )
        _validate_ratio(
            "whale_tracker.pressure_imbalance_ratio_threshold",
            self.pressure_imbalance_ratio_threshold,
        )

        _validate_non_negative_number(
            "whale_tracker.liquidation_context_min_notional",
            self.liquidation_context_min_notional,
        )

        _validate_non_negative_number(
            "whale_tracker.whale_activity_cooldown_sec",
            self.whale_activity_cooldown_sec,
        )
        _validate_non_negative_number(
            "whale_tracker.whale_pressure_cooldown_sec",
            self.whale_pressure_cooldown_sec,
        )
        _validate_non_negative_number(
            "whale_tracker.whale_liquidation_context_cooldown_sec",
            self.whale_liquidation_context_cooldown_sec,
        )

        _validate_positive_number(
            "whale_tracker.cleanup_interval_sec",
            self.cleanup_interval_sec,
        )
        _validate_positive_number(
            "whale_tracker.stats_ttl_sec",
            self.stats_ttl_sec,
        )


# =============================================================================
# Whale cluster analyzer config
# =============================================================================

@dataclass(slots=True)
class WhaleClusterAnalyzerConfig(WhaleScopedConfigMixin):
    """
    Config для третього шару whale-аналітики.

    Input:
        analytics.whales.whale_activity
        analytics.whales.whale_pressure
        analytics.whales.whale_liquidation_context

    Output:
        analytics.whales.whale_cluster
        analytics.whales.whale_cluster_update
        analytics.whales.whale_cluster_exhaustion
    """

    enabled: bool = True

    # Input events
    whale_activity_event_name: str = DEFAULT_WHALE_ACTIVITY_TOPIC
    whale_pressure_event_name: str = DEFAULT_WHALE_PRESSURE_TOPIC
    whale_liquidation_context_event_name: str = DEFAULT_WHALE_LIQUIDATION_CONTEXT_TOPIC

    # Output events
    whale_cluster_event_name: str = DEFAULT_WHALE_CLUSTER_TOPIC
    whale_cluster_update_event_name: str = DEFAULT_WHALE_CLUSTER_UPDATE_TOPIC
    whale_cluster_exhaustion_event_name: str = DEFAULT_WHALE_CLUSTER_EXHAUSTION_TOPIC

    # Analysis windows / ttl
    analysis_window_sec: int = 180
    cluster_ttl_sec: int = 300

    # Formation thresholds
    min_activity_signals: int = 2
    min_total_activity_notional: float = 500_000.0

    # Score weights
    activity_weight: float = 0.35
    pressure_weight: float = 0.35
    liquidation_context_weight: float = 0.20
    persistence_weight: float = 0.10

    # Score thresholds
    min_cluster_score_to_emit: float = 0.55
    min_continuation_probability_to_emit: float = 0.60
    min_exhaustion_probability_to_emit: float = 0.65

    # Cooldowns
    cluster_emit_cooldown_sec: float = 5.0
    cluster_update_cooldown_sec: float = 5.0
    cluster_exhaustion_cooldown_sec: float = 5.0

    # Cleanup
    cleanup_interval_sec: float = 60.0
    stats_ttl_sec: float = 60.0 * 60.0

    # Behavior
    emit_on_bus: bool = True
    log_signals: bool = True

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
        self._normalize_scope_fields()
        self.validate()

    @property
    def activity_buffer_size(self) -> int:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "activity_buffer_size", _analytics_args)
        except Exception:
            pass
        return self.analysis_window_sec * 2

    @property
    def pressure_buffer_size(self) -> int:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "pressure_buffer_size", _analytics_args)
        except Exception:
            pass
        return self.analysis_window_sec * 2

    @property
    def liquidation_context_buffer_size(self) -> int:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "liquidation_context_buffer_size", _analytics_args)
        except Exception:
            pass
        return self.analysis_window_sec * 2

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
            self.whale_activity_event_name,
            self.whale_pressure_event_name,
            self.whale_liquidation_context_event_name,
        )

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
        _validate_non_empty_topic(
            "whale_cluster_analyzer.whale_activity_event_name",
            self.whale_activity_event_name,
        )
        _validate_non_empty_topic(
            "whale_cluster_analyzer.whale_pressure_event_name",
            self.whale_pressure_event_name,
        )
        _validate_non_empty_topic(
            "whale_cluster_analyzer.whale_liquidation_context_event_name",
            self.whale_liquidation_context_event_name,
        )
        _validate_non_empty_topic(
            "whale_cluster_analyzer.whale_cluster_event_name",
            self.whale_cluster_event_name,
        )
        _validate_non_empty_topic(
            "whale_cluster_analyzer.whale_cluster_update_event_name",
            self.whale_cluster_update_event_name,
        )
        _validate_non_empty_topic(
            "whale_cluster_analyzer.whale_cluster_exhaustion_event_name",
            self.whale_cluster_exhaustion_event_name,
        )

        _validate_positive_int(
            "whale_cluster_analyzer.analysis_window_sec",
            self.analysis_window_sec,
        )
        _validate_positive_int(
            "whale_cluster_analyzer.cluster_ttl_sec",
            self.cluster_ttl_sec,
        )

        _validate_positive_int(
            "whale_cluster_analyzer.min_activity_signals",
            self.min_activity_signals,
        )
        _validate_non_negative_number(
            "whale_cluster_analyzer.min_total_activity_notional",
            self.min_total_activity_notional,
        )

        _validate_ratio("whale_cluster_analyzer.activity_weight", self.activity_weight)
        _validate_ratio("whale_cluster_analyzer.pressure_weight", self.pressure_weight)
        _validate_ratio(
            "whale_cluster_analyzer.liquidation_context_weight",
            self.liquidation_context_weight,
        )
        _validate_ratio(
            "whale_cluster_analyzer.persistence_weight",
            self.persistence_weight,
        )

        total_weight = (
            self.activity_weight
            + self.pressure_weight
            + self.liquidation_context_weight
            + self.persistence_weight
        )
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError("whale_cluster_analyzer weights must sum to 1.0")

        _validate_ratio(
            "whale_cluster_analyzer.min_cluster_score_to_emit",
            self.min_cluster_score_to_emit,
        )
        _validate_ratio(
            "whale_cluster_analyzer.min_continuation_probability_to_emit",
            self.min_continuation_probability_to_emit,
        )
        _validate_ratio(
            "whale_cluster_analyzer.min_exhaustion_probability_to_emit",
            self.min_exhaustion_probability_to_emit,
        )

        _validate_non_negative_number(
            "whale_cluster_analyzer.cluster_emit_cooldown_sec",
            self.cluster_emit_cooldown_sec,
        )
        _validate_non_negative_number(
            "whale_cluster_analyzer.cluster_update_cooldown_sec",
            self.cluster_update_cooldown_sec,
        )
        _validate_non_negative_number(
            "whale_cluster_analyzer.cluster_exhaustion_cooldown_sec",
            self.cluster_exhaustion_cooldown_sec,
        )

        _validate_positive_number(
            "whale_cluster_analyzer.cleanup_interval_sec",
            self.cleanup_interval_sec,
        )
        _validate_positive_number(
            "whale_cluster_analyzer.stats_ttl_sec",
            self.stats_ttl_sec,
        )


# =============================================================================
# Unified package config
# =============================================================================

@dataclass(slots=True)
class WhalesConfig:
    """
    Верхньорівневий unified config для всього analytics.whales пакета.

    Цей config передається у WhaleAnalyzer, а той уже передає підконфіги
    у LargeTradeDetector, WhaleTracker і WhaleClusterAnalyzer.
    """

    enabled: bool = True
    auto_start_components: bool = True

    large_trade_detector: LargeTradeDetectorConfig = field(
        default_factory=LargeTradeDetectorConfig
    )
    whale_tracker: WhaleTrackerConfig = field(
        default_factory=WhaleTrackerConfig
    )
    whale_cluster_analyzer: WhaleClusterAnalyzerConfig = field(
        default_factory=WhaleClusterAnalyzerConfig
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
        self.large_trade_detector.validate()
        self.whale_tracker.validate()
        self.whale_cluster_analyzer.validate()
        self._validate_pipeline_topics()

    @property
    def production_input_topics(self) -> dict[str, tuple[str, ...]]:
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
        return {
            "large_trade_detector": self.large_trade_detector.production_input_topics,
            "whale_tracker": self.whale_tracker.production_input_topics,
            "whale_cluster_analyzer": self.whale_cluster_analyzer.production_input_topics,
        }

    @property
    def legacy_raw_input_topics(self) -> dict[str, tuple[str, ...]]:
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
        return {
            "large_trade_detector": self.large_trade_detector.legacy_raw_input_topics,
            "whale_tracker": self.whale_tracker.legacy_raw_input_topics,
        }

    def _validate_pipeline_topics(self) -> None:
        """
        Перевіряє, що внутрішні output/input topics між whale-компонентами
        узгоджені між собою.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_pipeline_topics", _analytics_args)
        except Exception:
            pass
        if (
            self.large_trade_detector.output_event_name
            != self.whale_tracker.large_trade_event_name
        ):
            raise ValueError(
                "Pipeline topic mismatch: "
                "large_trade_detector.output_event_name must equal "
                "whale_tracker.large_trade_event_name"
            )

        if (
            self.whale_tracker.whale_activity_event_name
            != self.whale_cluster_analyzer.whale_activity_event_name
        ):
            raise ValueError(
                "Pipeline topic mismatch: "
                "whale_tracker.whale_activity_event_name must equal "
                "whale_cluster_analyzer.whale_activity_event_name"
            )

        if (
            self.whale_tracker.whale_pressure_event_name
            != self.whale_cluster_analyzer.whale_pressure_event_name
        ):
            raise ValueError(
                "Pipeline topic mismatch: "
                "whale_tracker.whale_pressure_event_name must equal "
                "whale_cluster_analyzer.whale_pressure_event_name"
            )

        if (
            self.whale_tracker.whale_liquidation_context_event_name
            != self.whale_cluster_analyzer.whale_liquidation_context_event_name
        ):
            raise ValueError(
                "Pipeline topic mismatch: "
                "whale_tracker.whale_liquidation_context_event_name must equal "
                "whale_cluster_analyzer.whale_liquidation_context_event_name"
            )


__all__ = [
    # project scope
    "PROJECT_EXCHANGES",
    "PROJECT_DEFAULT_EXCHANGE",
    "PROJECT_FUTURES_MARKET_TYPES",
    "PROJECT_DEFAULT_MARKET_TYPE",
    "PROJECT_SYMBOLS",
    "PROJECT_TIMEFRAMES",
    "PROJECT_DEFAULT_TIMEFRAME",
    "PROJECT_SYMBOL_ABS_THRESHOLDS",
    "PROJECT_SYMBOL_COOLDOWNS_SEC",

    # topics
    "DEFAULT_TRADES_UPDATED_TOPIC",
    "DEFAULT_LIQUIDATIONS_UPDATED_TOPIC",
    "DEFAULT_RAW_TRADE_TOPIC",
    "DEFAULT_RAW_LIQUIDATION_TOPIC",
    "DEFAULT_LARGE_TRADE_TOPIC",
    "DEFAULT_WHALE_ACTIVITY_TOPIC",
    "DEFAULT_WHALE_PRESSURE_TOPIC",
    "DEFAULT_WHALE_LIQUIDATION_CONTEXT_TOPIC",
    "DEFAULT_WHALE_CLUSTER_TOPIC",
    "DEFAULT_WHALE_CLUSTER_UPDATE_TOPIC",
    "DEFAULT_WHALE_CLUSTER_EXHAUSTION_TOPIC",

    # configs
    "WhaleScopedConfigMixin",
    "LargeTradeDetectorConfig",
    "WhaleTrackerConfig",
    "WhaleClusterAnalyzerConfig",
    "WhalesConfig",
]