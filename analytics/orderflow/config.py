from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, TypeAlias

from core.event_bus import EventPriority

from .enums import (
    ORDERBOOK_INPUT_TOPICS,
    TRADE_INPUT_TOPICS,
    OrderFlowEventTopic,
)


# =============================================================================
# Scope defaults / key
# =============================================================================

DEFAULT_EXCHANGE = "unknown"
DEFAULT_MARKET_TYPE = "perpetual"
DEFAULT_TIMEFRAME = "1m"

OrderFlowKey: TypeAlias = tuple[str, str, str, str]
# exchange, market_type, symbol, timeframe


# =============================================================================
# Canonical input topics
# =============================================================================

DEFAULT_TRADES_UPDATED_TOPIC = "market.trades.updated"
DEFAULT_ORDERBOOK_UPDATED_TOPIC = "market.orderbook.updated"

DEFAULT_RAW_TRADE_TOPIC = "market.trade"
DEFAULT_RAW_ORDERBOOK_TOPIC = "market.orderbook"

RAW_ORDERFLOW_MARKET_TOPICS = {
    DEFAULT_RAW_TRADE_TOPIC,
    DEFAULT_RAW_ORDERBOOK_TOPIC,
}


# =============================================================================
# Helpers
# =============================================================================

_MISSING = object()


def normalize_exchange(value: object | None) -> str:
    normalized = str(value or DEFAULT_EXCHANGE).strip().lower()
    return normalized or DEFAULT_EXCHANGE


def normalize_market_type(value: object | None) -> str:
    normalized = str(value or DEFAULT_MARKET_TYPE).strip().lower()
    return normalized or DEFAULT_MARKET_TYPE


def normalize_symbol(value: object | None) -> str:
    normalized = str(value or "").strip().upper()
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized


def normalize_timeframe(value: object | None) -> str:
    normalized = str(value or DEFAULT_TIMEFRAME).strip()
    return normalized or DEFAULT_TIMEFRAME


def make_orderflow_key(
    *,
    exchange: object | None,
    market_type: object | None,
    symbol: object,
    timeframe: object | None,
) -> OrderFlowKey:
    return (
        normalize_exchange(exchange),
        normalize_market_type(market_type),
        normalize_symbol(symbol),
        normalize_timeframe(timeframe),
    )


def orderflow_key_to_dict(key: OrderFlowKey) -> dict[str, str]:
    exchange, market_type, symbol, timeframe = key
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def orderflow_key_to_string(key: OrderFlowKey) -> str:
    scope = orderflow_key_to_dict(key)
    return (
        f"{scope['exchange']}:"
        f"{scope['market_type']}:"
        f"{scope['symbol']}:"
        f"{scope['timeframe']}"
    )


def _get_nested_config(root: Any, *path: str) -> Any:
    """
    Safely read nested config attributes.

    Supports app-level config objects like:
        app_config.analytics.orderflow.cvd
    """
    current = root

    for item in path:
        if current is None:
            return None

        current = getattr(current, item, None)

    return current


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default

    return getattr(obj, name, default)


def _get_bool(obj: Any, name: str, default: bool) -> bool:
    value = _get_attr(obj, name, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False

    return bool(default if value is None else value)


def _get_int(obj: Any, name: str, default: int) -> int:
    value = _get_attr(obj, name, default)

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_float(obj: Any, name: str, default: float) -> float:
    value = _get_attr(obj, name, default)

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_str(obj: Any, name: str, default: str) -> str:
    value = _get_attr(obj, name, default)

    if value is None:
        return default

    return str(value)


def _get_list_str(obj: Any, name: str, default: tuple[str, ...] | list[str]) -> list[str]:
    value = _get_attr(obj, name, _MISSING)

    if value is _MISSING or value is None:
        return list(default)

    if isinstance(value, str):
        return [value]

    try:
        return [str(item) for item in value]
    except TypeError:
        return list(default)


def _normalize_topics(values: list[str] | tuple[str, ...] | set[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value).strip()
        )
    )


def _normalize_symbol_allowlist(value: Any) -> set[str] | None:
    if value is None:
        return None

    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    else:
        try:
            raw_items = list(value)
        except TypeError:
            return None

    normalized = {
        normalize_symbol(item)
        for item in raw_items
        if str(item).strip()
    }

    return normalized or None


def _normalize_exchange_set(values: Any) -> set[str]:
    if values is None:
        return set()

    if isinstance(values, str):
        raw_items = [item.strip() for item in values.split(",")]
    else:
        try:
            raw_items = list(values)
        except TypeError:
            return set()

    return {
        normalize_exchange(item)
        for item in raw_items
        if str(item).strip()
    }


def _normalize_market_type_set(values: Any) -> set[str]:
    if values is None:
        return set()

    if isinstance(values, str):
        raw_items = [item.strip() for item in values.split(",")]
    else:
        try:
            raw_items = list(values)
        except TypeError:
            return set()

    return {
        normalize_market_type(item)
        for item in raw_items
        if str(item).strip()
    }


def _normalize_symbol_set(values: Any) -> set[str]:
    if values is None:
        return set()

    if isinstance(values, str):
        raw_items = [item.strip() for item in values.split(",")]
    else:
        try:
            raw_items = list(values)
        except TypeError:
            return set()

    return {
        normalize_symbol(item)
        for item in raw_items
        if str(item).strip()
    }


def _normalize_timeframe_set(values: Any) -> set[str]:
    if values is None:
        return set()

    if isinstance(values, str):
        raw_items = [item.strip() for item in values.split(",")]
    else:
        try:
            raw_items = list(values)
        except TypeError:
            return set()

    return {
        normalize_timeframe(item)
        for item in raw_items
        if str(item).strip()
    }


def _normalize_priority(value: Any, default: EventPriority = EventPriority.NORMAL) -> EventPriority:
    if isinstance(value, EventPriority):
        return value

    if isinstance(value, Enum):
        value = value.value

    if isinstance(value, int):
        try:
            return EventPriority(value)
        except ValueError:
            return default

    if isinstance(value, str):
        normalized = value.strip().upper()

        if normalized in EventPriority.__members__:
            return EventPriority[normalized]

        lowered = value.strip().lower()
        for item in EventPriority:
            if item.name.lower() == lowered or str(item.value).lower() == lowered:
                return item

    return default


def _validate_topic(topic: str, field_name: str) -> None:
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError(f"{field_name} must not be empty")

    if " " in topic:
        raise ValueError(f"{field_name} must not contain spaces")


def _validate_job_name(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")

    if " " in value:
        raise ValueError(f"{field_name} must not contain spaces")


def _base_kwargs_from_section(
    section: Any,
    *,
    source_name: str,
    update_topic: str,
    signal_topic: str,
    default_min_signal_interval_sec: float = 0.75,
) -> dict[str, Any]:
    return {
        "enabled": _get_bool(section, "enabled", True),
        "emit_updates": _get_bool(section, "emit_updates", True),
        "emit_signals": _get_bool(section, "emit_signals", True),
        "min_signal_interval_sec": _get_float(
            section,
            "min_signal_interval_sec",
            default_min_signal_interval_sec,
        ),
        "health_log_interval_sec": _get_float(section, "health_log_interval_sec", 30.0),
        "cleanup_interval_sec": _get_float(section, "cleanup_interval_sec", 15.0),
        "scheduler_job_timeout_sec": _get_float(section, "scheduler_job_timeout_sec", 10.0),
        "scheduler_job_retry_delay_sec": _get_float(
            section,
            "scheduler_job_retry_delay_sec",
            1.0,
        ),
        "scheduler_job_max_retries": _get_int(section, "scheduler_job_max_retries", 1),
        "symbol_allowlist": _normalize_symbol_allowlist(
            _get_attr(section, "symbol_allowlist", None)
        ),
        "allowed_exchanges": _normalize_exchange_set(
            _get_attr(section, "allowed_exchanges", None)
        ),
        "allowed_market_types": _normalize_market_type_set(
            _get_attr(section, "allowed_market_types", None)
        ),
        "allowed_symbols": _normalize_symbol_set(
            _get_attr(section, "allowed_symbols", None)
        ),
        "allowed_timeframes": _normalize_timeframe_set(
            _get_attr(section, "allowed_timeframes", None)
        ),
        "publish_priority": _normalize_priority(
            _get_attr(section, "publish_priority", EventPriority.NORMAL)
        ),
        "source_name": _get_str(section, "source_name", source_name),
        "update_topic": _get_str(section, "update_topic", update_topic),
        "signal_topic": _get_str(section, "signal_topic", signal_topic),
    }


# =============================================================================
# Base sub-config
# =============================================================================


@dataclass(slots=True)
class BaseOrderFlowSubConfig:
    """
    Shared config contract for all analytics.orderflow analyzers.

    Runtime dependencies such as EventBus and Scheduler are intentionally not
    stored here. They must be injected into analyzer constructors.

    Canonical scope:
        exchange + market_type + symbol + timeframe
    """

    enabled: bool = True
    emit_updates: bool = True
    emit_signals: bool = True

    min_signal_interval_sec: float = 0.75

    health_log_interval_sec: float = 30.0
    cleanup_interval_sec: float = 15.0

    scheduler_job_timeout_sec: float = 10.0
    scheduler_job_retry_delay_sec: float = 1.0
    scheduler_job_max_retries: int = 1

    # Backward-compatible symbol-only allowlist.
    symbol_allowlist: set[str] | None = None

    # New scoped allowlists.
    allowed_exchanges: set[str] = field(default_factory=set)
    allowed_market_types: set[str] = field(default_factory=set)
    allowed_symbols: set[str] = field(default_factory=set)
    allowed_timeframes: set[str] = field(default_factory=set)

    publish_priority: EventPriority = EventPriority.NORMAL

    source_name: str = "orderflow_analyzer"
    update_topic: str = ""
    signal_topic: str = ""

    def __post_init__(self) -> None:
        self.symbol_allowlist = _normalize_symbol_allowlist(self.symbol_allowlist)
        self.allowed_exchanges = _normalize_exchange_set(self.allowed_exchanges)
        self.allowed_market_types = _normalize_market_type_set(self.allowed_market_types)
        self.allowed_symbols = _normalize_symbol_set(self.allowed_symbols)
        self.allowed_timeframes = _normalize_timeframe_set(self.allowed_timeframes)
        self.publish_priority = _normalize_priority(self.publish_priority)

        if self.symbol_allowlist:
            self.allowed_symbols.update(self.symbol_allowlist)

    def make_key(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> OrderFlowKey:
        return make_orderflow_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    def should_process_symbol(self, symbol: str | None) -> bool:
        if not symbol:
            return False

        if self.allowed_symbols:
            return normalize_symbol(symbol) in self.allowed_symbols

        return True

    def should_process_key(self, key: OrderFlowKey) -> bool:
        scope = orderflow_key_to_dict(key)

        if self.allowed_exchanges and scope["exchange"] not in self.allowed_exchanges:
            return False

        if self.allowed_market_types and scope["market_type"] not in self.allowed_market_types:
            return False

        if self.allowed_symbols and scope["symbol"] not in self.allowed_symbols:
            return False

        if self.allowed_timeframes and scope["timeframe"] not in self.allowed_timeframes:
            return False

        return True

    def should_process_scope(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> bool:
        return self.should_process_key(
            self.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    @property
    def output_topics(self) -> tuple[str, ...]:
        topics: list[str] = []

        if self.emit_updates:
            topics.append(self.update_topic)

        if self.emit_signals:
            topics.append(self.signal_topic)

        return tuple(dict.fromkeys(topic for topic in topics if topic))

    @property
    def scheduler_job_names(self) -> tuple[str, ...]:
        source = self.source_name or self.__class__.__name__
        return (
            f"analytics.orderflow.{source}.health",
            f"analytics.orderflow.{source}.cleanup",
        )

    def validate(self) -> None:
        errors: list[str] = []

        if self.min_signal_interval_sec < 0:
            errors.append("min_signal_interval_sec must be >= 0")

        if self.health_log_interval_sec <= 0:
            errors.append("health_log_interval_sec must be > 0")

        if self.cleanup_interval_sec <= 0:
            errors.append("cleanup_interval_sec must be > 0")

        if self.scheduler_job_timeout_sec <= 0:
            errors.append("scheduler_job_timeout_sec must be > 0")

        if self.scheduler_job_retry_delay_sec < 0:
            errors.append("scheduler_job_retry_delay_sec must be >= 0")

        if self.scheduler_job_max_retries < 0:
            errors.append("scheduler_job_max_retries must be >= 0")

        if self.emit_updates and not self.update_topic:
            errors.append("update_topic is required when emit_updates=True")

        if self.emit_signals and not self.signal_topic:
            errors.append("signal_topic is required when emit_signals=True")

        for topic in self.output_topics:
            try:
                _validate_topic(topic, "orderflow output topic")
            except ValueError as exc:
                errors.append(str(exc))

        for job_name in self.scheduler_job_names:
            try:
                _validate_job_name(job_name, "orderflow scheduler job name")
            except ValueError as exc:
                errors.append(str(exc))

        if errors:
            raise ValueError(
                f"{self.__class__.__name__} validation failed: "
                + "; ".join(errors)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "emit_updates": self.emit_updates,
            "emit_signals": self.emit_signals,
            "min_signal_interval_sec": self.min_signal_interval_sec,
            "health_log_interval_sec": self.health_log_interval_sec,
            "cleanup_interval_sec": self.cleanup_interval_sec,
            "scheduler_job_timeout_sec": self.scheduler_job_timeout_sec,
            "scheduler_job_retry_delay_sec": self.scheduler_job_retry_delay_sec,
            "scheduler_job_max_retries": self.scheduler_job_max_retries,
            "symbol_allowlist": sorted(self.symbol_allowlist) if self.symbol_allowlist else None,
            "allowed_exchanges": sorted(self.allowed_exchanges),
            "allowed_market_types": sorted(self.allowed_market_types),
            "allowed_symbols": sorted(self.allowed_symbols),
            "allowed_timeframes": sorted(self.allowed_timeframes),
            "publish_priority": self.publish_priority.name,
            "source_name": self.source_name,
            "update_topic": self.update_topic,
            "signal_topic": self.signal_topic,
            "output_topics": list(self.output_topics),
            "scheduler_job_names": list(self.scheduler_job_names),
        }


# =============================================================================
# Metric-specific configs
# =============================================================================


@dataclass(slots=True)
class CvdConfig(BaseOrderFlowSubConfig):
    window_seconds: float = 20.0
    max_trades_per_symbol: int = 8000
    max_cvd_points_per_symbol: int = 5000

    min_trades_in_window: int = 12
    min_total_volume: float = 0.0

    bullish_delta_ratio_threshold: float = 0.15
    bearish_delta_ratio_threshold: float = -0.15

    bullish_cvd_change_threshold: float = 0.0
    bearish_cvd_change_threshold: float = 0.0

    bullish_cvd_slope_threshold: float = 0.0
    bearish_cvd_slope_threshold: float = 0.0

    bullish_impulse_threshold_pct: float = 0.0
    bearish_impulse_threshold_pct: float = 0.0

    require_delta_confirmation: bool = True
    require_slope_confirmation: bool = True

    source_name: str = "cvd"
    update_topic: str = OrderFlowEventTopic.CVD_UPDATED.value
    signal_topic: str = OrderFlowEventTopic.CVD_SIGNAL.value

    @classmethod
    def from_app_config(cls, app_config: Any) -> CvdConfig:
        section = _get_nested_config(app_config, "analytics", "orderflow", "cvd")

        return cls(
            **_base_kwargs_from_section(
                section,
                source_name="cvd",
                update_topic=OrderFlowEventTopic.CVD_UPDATED.value,
                signal_topic=OrderFlowEventTopic.CVD_SIGNAL.value,
                default_min_signal_interval_sec=0.75,
            ),
            window_seconds=_get_float(section, "window_seconds", 20.0),
            max_trades_per_symbol=_get_int(section, "max_trades_per_symbol", 8000),
            max_cvd_points_per_symbol=_get_int(section, "max_cvd_points_per_symbol", 5000),
            min_trades_in_window=_get_int(section, "min_trades_in_window", 12),
            min_total_volume=_get_float(section, "min_total_volume", 0.0),
            bullish_delta_ratio_threshold=_get_float(
                section,
                "bullish_delta_ratio_threshold",
                0.15,
            ),
            bearish_delta_ratio_threshold=_get_float(
                section,
                "bearish_delta_ratio_threshold",
                -0.15,
            ),
            bullish_cvd_change_threshold=_get_float(
                section,
                "bullish_cvd_change_threshold",
                0.0,
            ),
            bearish_cvd_change_threshold=_get_float(
                section,
                "bearish_cvd_change_threshold",
                0.0,
            ),
            bullish_cvd_slope_threshold=_get_float(
                section,
                "bullish_cvd_slope_threshold",
                0.0,
            ),
            bearish_cvd_slope_threshold=_get_float(
                section,
                "bearish_cvd_slope_threshold",
                0.0,
            ),
            bullish_impulse_threshold_pct=_get_float(
                section,
                "bullish_impulse_threshold_pct",
                0.0,
            ),
            bearish_impulse_threshold_pct=_get_float(
                section,
                "bearish_impulse_threshold_pct",
                0.0,
            ),
            require_delta_confirmation=_get_bool(section, "require_delta_confirmation", True),
            require_slope_confirmation=_get_bool(section, "require_slope_confirmation", True),
        )

    def validate(self) -> None:
        BaseOrderFlowSubConfig.validate(self)

        errors: list[str] = []

        if self.window_seconds <= 0:
            errors.append("window_seconds must be > 0")

        if self.max_trades_per_symbol <= 0:
            errors.append("max_trades_per_symbol must be > 0")

        if self.max_cvd_points_per_symbol <= 0:
            errors.append("max_cvd_points_per_symbol must be > 0")

        if self.min_trades_in_window <= 0:
            errors.append("min_trades_in_window must be > 0")

        if self.min_total_volume < 0:
            errors.append("min_total_volume must be >= 0")

        if errors:
            raise ValueError(
                f"{self.__class__.__name__} validation failed: "
                + "; ".join(errors)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = BaseOrderFlowSubConfig.to_dict(self)
        payload.update(
            {
                "window_seconds": self.window_seconds,
                "max_trades_per_symbol": self.max_trades_per_symbol,
                "max_cvd_points_per_symbol": self.max_cvd_points_per_symbol,
                "min_trades_in_window": self.min_trades_in_window,
                "min_total_volume": self.min_total_volume,
                "bullish_delta_ratio_threshold": self.bullish_delta_ratio_threshold,
                "bearish_delta_ratio_threshold": self.bearish_delta_ratio_threshold,
                "bullish_cvd_change_threshold": self.bullish_cvd_change_threshold,
                "bearish_cvd_change_threshold": self.bearish_cvd_change_threshold,
                "bullish_cvd_slope_threshold": self.bullish_cvd_slope_threshold,
                "bearish_cvd_slope_threshold": self.bearish_cvd_slope_threshold,
                "bullish_impulse_threshold_pct": self.bullish_impulse_threshold_pct,
                "bearish_impulse_threshold_pct": self.bearish_impulse_threshold_pct,
                "require_delta_confirmation": self.require_delta_confirmation,
                "require_slope_confirmation": self.require_slope_confirmation,
            }
        )
        return payload


@dataclass(slots=True)
class VolumeDeltaConfig(BaseOrderFlowSubConfig):
    window_seconds: float = 10.0
    max_trades_per_symbol: int = 6000

    min_trades_in_window: int = 10
    min_total_volume: float = 0.0

    bullish_delta_ratio_threshold: float = 0.18
    bearish_delta_ratio_threshold: float = -0.18

    bullish_volume_delta_threshold: float = 0.0
    bearish_volume_delta_threshold: float = 0.0

    bullish_cumulative_delta_threshold: float = 0.0
    bearish_cumulative_delta_threshold: float = 0.0

    require_ratio_and_absolute_confirmation: bool = True

    source_name: str = "volume_delta"
    update_topic: str = OrderFlowEventTopic.VOLUME_DELTA_UPDATED.value
    signal_topic: str = OrderFlowEventTopic.VOLUME_DELTA_SIGNAL.value

    @classmethod
    def from_app_config(cls, app_config: Any) -> VolumeDeltaConfig:
        section = _get_nested_config(app_config, "analytics", "orderflow", "volume_delta")

        return cls(
            **_base_kwargs_from_section(
                section,
                source_name="volume_delta",
                update_topic=OrderFlowEventTopic.VOLUME_DELTA_UPDATED.value,
                signal_topic=OrderFlowEventTopic.VOLUME_DELTA_SIGNAL.value,
                default_min_signal_interval_sec=0.50,
            ),
            window_seconds=_get_float(section, "window_seconds", 10.0),
            max_trades_per_symbol=_get_int(section, "max_trades_per_symbol", 6000),
            min_trades_in_window=_get_int(section, "min_trades_in_window", 10),
            min_total_volume=_get_float(section, "min_total_volume", 0.0),
            bullish_delta_ratio_threshold=_get_float(
                section,
                "bullish_delta_ratio_threshold",
                0.18,
            ),
            bearish_delta_ratio_threshold=_get_float(
                section,
                "bearish_delta_ratio_threshold",
                -0.18,
            ),
            bullish_volume_delta_threshold=_get_float(
                section,
                "bullish_volume_delta_threshold",
                0.0,
            ),
            bearish_volume_delta_threshold=_get_float(
                section,
                "bearish_volume_delta_threshold",
                0.0,
            ),
            bullish_cumulative_delta_threshold=_get_float(
                section,
                "bullish_cumulative_delta_threshold",
                0.0,
            ),
            bearish_cumulative_delta_threshold=_get_float(
                section,
                "bearish_cumulative_delta_threshold",
                0.0,
            ),
            require_ratio_and_absolute_confirmation=_get_bool(
                section,
                "require_ratio_and_absolute_confirmation",
                True,
            ),
        )

    def validate(self) -> None:
        BaseOrderFlowSubConfig.validate(self)

        errors: list[str] = []

        if self.window_seconds <= 0:
            errors.append("window_seconds must be > 0")

        if self.max_trades_per_symbol <= 0:
            errors.append("max_trades_per_symbol must be > 0")

        if self.min_trades_in_window <= 0:
            errors.append("min_trades_in_window must be > 0")

        if self.min_total_volume < 0:
            errors.append("min_total_volume must be >= 0")

        if errors:
            raise ValueError(
                f"{self.__class__.__name__} validation failed: "
                + "; ".join(errors)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = BaseOrderFlowSubConfig.to_dict(self)
        payload.update(
            {
                "window_seconds": self.window_seconds,
                "max_trades_per_symbol": self.max_trades_per_symbol,
                "min_trades_in_window": self.min_trades_in_window,
                "min_total_volume": self.min_total_volume,
                "bullish_delta_ratio_threshold": self.bullish_delta_ratio_threshold,
                "bearish_delta_ratio_threshold": self.bearish_delta_ratio_threshold,
                "bullish_volume_delta_threshold": self.bullish_volume_delta_threshold,
                "bearish_volume_delta_threshold": self.bearish_volume_delta_threshold,
                "bullish_cumulative_delta_threshold": self.bullish_cumulative_delta_threshold,
                "bearish_cumulative_delta_threshold": self.bearish_cumulative_delta_threshold,
                "require_ratio_and_absolute_confirmation": self.require_ratio_and_absolute_confirmation,
            }
        )
        return payload


@dataclass(slots=True)
class AggressiveTradesConfig(BaseOrderFlowSubConfig):
    window_seconds: float = 8.0
    max_trades_per_symbol: int = 5000

    min_trades_in_window: int = 8

    bullish_buy_ratio_threshold: float = 0.68
    bearish_sell_ratio_threshold: float = 0.68

    bullish_delta_threshold: float = 0.0
    bearish_delta_threshold: float = 0.0

    large_trade_notional_threshold: float = 25_000.0
    min_large_trades_for_signal: int = 1

    burst_trades_threshold: int = 12
    burst_volume_threshold: float = 0.0
    burst_score_threshold: float = 1.15

    source_name: str = "aggressive_trades"
    update_topic: str = OrderFlowEventTopic.AGGRESSIVE_TRADES_UPDATED.value
    signal_topic: str = OrderFlowEventTopic.AGGRESSIVE_TRADES_SIGNAL.value

    @classmethod
    def from_app_config(cls, app_config: Any) -> AggressiveTradesConfig:
        section = _get_nested_config(
            app_config,
            "analytics",
            "orderflow",
            "aggressive_trades",
        )

        return cls(
            **_base_kwargs_from_section(
                section,
                source_name="aggressive_trades",
                update_topic=OrderFlowEventTopic.AGGRESSIVE_TRADES_UPDATED.value,
                signal_topic=OrderFlowEventTopic.AGGRESSIVE_TRADES_SIGNAL.value,
                default_min_signal_interval_sec=0.50,
            ),
            window_seconds=_get_float(section, "window_seconds", 8.0),
            max_trades_per_symbol=_get_int(section, "max_trades_per_symbol", 5000),
            min_trades_in_window=_get_int(section, "min_trades_in_window", 8),
            bullish_buy_ratio_threshold=_get_float(
                section,
                "bullish_buy_ratio_threshold",
                0.68,
            ),
            bearish_sell_ratio_threshold=_get_float(
                section,
                "bearish_sell_ratio_threshold",
                0.68,
            ),
            bullish_delta_threshold=_get_float(section, "bullish_delta_threshold", 0.0),
            bearish_delta_threshold=_get_float(section, "bearish_delta_threshold", 0.0),
            large_trade_notional_threshold=_get_float(
                section,
                "large_trade_notional_threshold",
                25_000.0,
            ),
            min_large_trades_for_signal=_get_int(section, "min_large_trades_for_signal", 1),
            burst_trades_threshold=_get_int(section, "burst_trades_threshold", 12),
            burst_volume_threshold=_get_float(section, "burst_volume_threshold", 0.0),
            burst_score_threshold=_get_float(section, "burst_score_threshold", 1.15),
        )

    def validate(self) -> None:
        BaseOrderFlowSubConfig.validate(self)

        errors: list[str] = []

        if self.window_seconds <= 0:
            errors.append("window_seconds must be > 0")

        if self.max_trades_per_symbol <= 0:
            errors.append("max_trades_per_symbol must be > 0")

        if self.min_trades_in_window <= 0:
            errors.append("min_trades_in_window must be > 0")

        if not 0.0 <= self.bullish_buy_ratio_threshold <= 1.0:
            errors.append("bullish_buy_ratio_threshold must be between 0 and 1")

        if not 0.0 <= self.bearish_sell_ratio_threshold <= 1.0:
            errors.append("bearish_sell_ratio_threshold must be between 0 and 1")

        if self.large_trade_notional_threshold < 0:
            errors.append("large_trade_notional_threshold must be >= 0")

        if self.min_large_trades_for_signal < 0:
            errors.append("min_large_trades_for_signal must be >= 0")

        if self.burst_trades_threshold < 0:
            errors.append("burst_trades_threshold must be >= 0")

        if self.burst_volume_threshold < 0:
            errors.append("burst_volume_threshold must be >= 0")

        if self.burst_score_threshold < 0:
            errors.append("burst_score_threshold must be >= 0")

        if errors:
            raise ValueError(
                f"{self.__class__.__name__} validation failed: "
                + "; ".join(errors)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = BaseOrderFlowSubConfig.to_dict(self)
        payload.update(
            {
                "window_seconds": self.window_seconds,
                "max_trades_per_symbol": self.max_trades_per_symbol,
                "min_trades_in_window": self.min_trades_in_window,
                "bullish_buy_ratio_threshold": self.bullish_buy_ratio_threshold,
                "bearish_sell_ratio_threshold": self.bearish_sell_ratio_threshold,
                "bullish_delta_threshold": self.bullish_delta_threshold,
                "bearish_delta_threshold": self.bearish_delta_threshold,
                "large_trade_notional_threshold": self.large_trade_notional_threshold,
                "min_large_trades_for_signal": self.min_large_trades_for_signal,
                "burst_trades_threshold": self.burst_trades_threshold,
                "burst_volume_threshold": self.burst_volume_threshold,
                "burst_score_threshold": self.burst_score_threshold,
            }
        )
        return payload


@dataclass(slots=True)
class OrderbookImbalanceConfig(BaseOrderFlowSubConfig):
    depth_levels: int = 10
    min_total_volume: float = 0.0

    bullish_ratio_threshold: float = 0.60
    bearish_ratio_threshold: float = 0.40

    normalize_ratio_to_minus_one_one: bool = False
    smooth_window: int = 5

    source_name: str = "orderbook_imbalance"
    update_topic: str = OrderFlowEventTopic.ORDERBOOK_IMBALANCE_UPDATED.value
    signal_topic: str = OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL.value

    @classmethod
    def from_app_config(cls, app_config: Any) -> OrderbookImbalanceConfig:
        section = _get_nested_config(
            app_config,
            "analytics",
            "orderflow",
            "orderbook_imbalance",
        )

        return cls(
            **_base_kwargs_from_section(
                section,
                source_name="orderbook_imbalance",
                update_topic=OrderFlowEventTopic.ORDERBOOK_IMBALANCE_UPDATED.value,
                signal_topic=OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL.value,
                default_min_signal_interval_sec=0.30,
            ),
            depth_levels=_get_int(section, "depth_levels", 10),
            min_total_volume=_get_float(section, "min_total_volume", 0.0),
            bullish_ratio_threshold=_get_float(section, "bullish_ratio_threshold", 0.60),
            bearish_ratio_threshold=_get_float(section, "bearish_ratio_threshold", 0.40),
            normalize_ratio_to_minus_one_one=_get_bool(
                section,
                "normalize_ratio_to_minus_one_one",
                False,
            ),
            smooth_window=_get_int(section, "smooth_window", 5),
        )

    def validate(self) -> None:
        BaseOrderFlowSubConfig.validate(self)

        errors: list[str] = []

        if self.depth_levels <= 0:
            errors.append("depth_levels must be > 0")

        if self.min_total_volume < 0:
            errors.append("min_total_volume must be >= 0")

        if not 0.0 <= self.bullish_ratio_threshold <= 1.0:
            errors.append("bullish_ratio_threshold must be between 0 and 1")

        if not 0.0 <= self.bearish_ratio_threshold <= 1.0:
            errors.append("bearish_ratio_threshold must be between 0 and 1")

        if self.bearish_ratio_threshold >= self.bullish_ratio_threshold:
            errors.append("bearish_ratio_threshold must be < bullish_ratio_threshold")

        if self.smooth_window <= 0:
            errors.append("smooth_window must be > 0")

        if errors:
            raise ValueError(
                f"{self.__class__.__name__} validation failed: "
                + "; ".join(errors)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = BaseOrderFlowSubConfig.to_dict(self)
        payload.update(
            {
                "depth_levels": self.depth_levels,
                "min_total_volume": self.min_total_volume,
                "bullish_ratio_threshold": self.bullish_ratio_threshold,
                "bearish_ratio_threshold": self.bearish_ratio_threshold,
                "normalize_ratio_to_minus_one_one": self.normalize_ratio_to_minus_one_one,
                "smooth_window": self.smooth_window,
            }
        )
        return payload


# =============================================================================
# Package-level config
# =============================================================================


@dataclass(slots=True)
class OrderFlowConfig:
    """
    Top-level config for analytics.orderflow package.

    This object is passed to OrderFlowAnalyzer facade, which then injects
    metric-specific sub-configs into concrete analyzers.

    Correct production input topics:
        market.trades.updated
        market.orderbook.updated

    Raw market topics are forbidden by default:
        market.trade
        market.orderbook

    Canonical scope:
        exchange + market_type + symbol + timeframe
    """

    enabled: bool = True

    # ------------------------------------------------------------------
    # Scope defaults / filters
    # ------------------------------------------------------------------

    default_exchange: str = DEFAULT_EXCHANGE
    default_market_type: str = DEFAULT_MARKET_TYPE
    default_timeframe: str = DEFAULT_TIMEFRAME

    allowed_exchanges: set[str] = field(default_factory=set)
    allowed_market_types: set[str] = field(
        default_factory=lambda: {
            "perpetual",
            "futures",
            "linear",
            "inverse",
            "swap",
            "usdm_futures",
            "coinm_futures",
        }
    )
    allowed_symbols: set[str] = field(default_factory=set)
    allowed_timeframes: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Input topics
    # ------------------------------------------------------------------

    trade_input_topics: tuple[str, ...] = tuple(TRADE_INPUT_TOPICS)
    orderbook_input_topics: tuple[str, ...] = tuple(ORDERBOOK_INPUT_TOPICS)

    # Backward-compatible names used by current facade/base.
    source_topic_patterns_trades: list[str] = field(
        default_factory=lambda: list(TRADE_INPUT_TOPICS)
    )
    source_topic_patterns_orderbook: list[str] = field(
        default_factory=lambda: list(ORDERBOOK_INPUT_TOPICS)
    )

    allow_raw_market_topics: bool = False

    # ------------------------------------------------------------------
    # Sub-configs
    # ------------------------------------------------------------------

    cvd: CvdConfig = field(default_factory=CvdConfig)
    volume_delta: VolumeDeltaConfig = field(default_factory=VolumeDeltaConfig)
    aggressive_trades: AggressiveTradesConfig = field(default_factory=AggressiveTradesConfig)
    orderbook_imbalance: OrderbookImbalanceConfig = field(default_factory=OrderbookImbalanceConfig)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.default_exchange = normalize_exchange(self.default_exchange)
        self.default_market_type = normalize_market_type(self.default_market_type)
        self.default_timeframe = normalize_timeframe(self.default_timeframe)

        self.allowed_exchanges = _normalize_exchange_set(self.allowed_exchanges)
        self.allowed_market_types = _normalize_market_type_set(self.allowed_market_types)
        self.allowed_symbols = _normalize_symbol_set(self.allowed_symbols)
        self.allowed_timeframes = _normalize_timeframe_set(self.allowed_timeframes)

        self.trade_input_topics = _normalize_topics(self.trade_input_topics)
        self.orderbook_input_topics = _normalize_topics(self.orderbook_input_topics)

        # Keep backward-compatible fields in sync with canonical topic fields.
        if self.source_topic_patterns_trades:
            self.source_topic_patterns_trades = list(
                _normalize_topics(self.source_topic_patterns_trades)
            )
        else:
            self.source_topic_patterns_trades = list(self.trade_input_topics)

        if self.source_topic_patterns_orderbook:
            self.source_topic_patterns_orderbook = list(
                _normalize_topics(self.source_topic_patterns_orderbook)
            )
        else:
            self.source_topic_patterns_orderbook = list(self.orderbook_input_topics)

        # If caller provided legacy fields but not canonical fields, honor legacy.
        if not self.trade_input_topics:
            self.trade_input_topics = tuple(self.source_topic_patterns_trades)

        if not self.orderbook_input_topics:
            self.orderbook_input_topics = tuple(self.source_topic_patterns_orderbook)

        self.metadata = dict(self.metadata or {})

        self._propagate_scope_filters_to_subconfigs()
        self.validate()

    @classmethod
    def from_app_config(cls, app_config: Any) -> OrderFlowConfig:
        section = _get_nested_config(app_config, "analytics", "orderflow")

        config = cls(
            enabled=_get_bool(section, "enabled", True),
            default_exchange=_get_str(section, "default_exchange", DEFAULT_EXCHANGE),
            default_market_type=_get_str(section, "default_market_type", DEFAULT_MARKET_TYPE),
            default_timeframe=_get_str(section, "default_timeframe", DEFAULT_TIMEFRAME),
            allowed_exchanges=_normalize_exchange_set(
                _get_attr(section, "allowed_exchanges", None)
            ),
            allowed_market_types=_normalize_market_type_set(
                _get_attr(section, "allowed_market_types", None)
            ) or {
                "perpetual",
                "futures",
                "linear",
                "inverse",
                "swap",
                "usdm_futures",
                "coinm_futures",
            },
            allowed_symbols=_normalize_symbol_set(
                _get_attr(section, "allowed_symbols", None)
            ),
            allowed_timeframes=_normalize_timeframe_set(
                _get_attr(section, "allowed_timeframes", None)
            ),
            trade_input_topics=tuple(
                _get_list_str(
                    section,
                    "trade_input_topics",
                    TRADE_INPUT_TOPICS,
                )
            ),
            orderbook_input_topics=tuple(
                _get_list_str(
                    section,
                    "orderbook_input_topics",
                    ORDERBOOK_INPUT_TOPICS,
                )
            ),
            source_topic_patterns_trades=_get_list_str(
                section,
                "source_topic_patterns_trades",
                TRADE_INPUT_TOPICS,
            ),
            source_topic_patterns_orderbook=_get_list_str(
                section,
                "source_topic_patterns_orderbook",
                ORDERBOOK_INPUT_TOPICS,
            ),
            allow_raw_market_topics=_get_bool(section, "allow_raw_market_topics", False),
            cvd=CvdConfig.from_app_config(app_config),
            volume_delta=VolumeDeltaConfig.from_app_config(app_config),
            aggressive_trades=AggressiveTradesConfig.from_app_config(app_config),
            orderbook_imbalance=OrderbookImbalanceConfig.from_app_config(app_config),
            metadata=dict(_get_attr(section, "metadata", {}) or {}),
        )
        config.validate()
        return config

    # ------------------------------------------------------------------
    # Topic groups
    # ------------------------------------------------------------------

    @property
    def trades_topics(self) -> tuple[str, ...]:
        return self.trade_input_topics

    @property
    def orderbook_topics(self) -> tuple[str, ...]:
        return self.orderbook_input_topics

    @property
    def production_input_topics(self) -> tuple[str, ...]:
        topics: list[str] = []
        topics.extend(self.trade_input_topics)
        topics.extend(self.orderbook_input_topics)
        return tuple(dict.fromkeys(topics))

    @property
    def output_topics(self) -> tuple[str, ...]:
        topics: list[str] = []

        for config in (
            self.cvd,
            self.volume_delta,
            self.aggressive_trades,
            self.orderbook_imbalance,
        ):
            topics.extend(config.output_topics)

        topics.extend(
            [
                OrderFlowEventTopic.STARTED.value,
                OrderFlowEventTopic.STOPPED.value,
                OrderFlowEventTopic.HEALTH.value,
                OrderFlowEventTopic.ERROR.value,
                OrderFlowEventTopic.UPDATED.value,
            ]
        )

        return tuple(dict.fromkeys(topic for topic in topics if topic))

    @property
    def scheduler_job_names(self) -> tuple[str, ...]:
        names: list[str] = []

        for config in (
            self.cvd,
            self.volume_delta,
            self.aggressive_trades,
            self.orderbook_imbalance,
        ):
            names.extend(config.scheduler_job_names)

        return tuple(dict.fromkeys(names))

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    def make_key(
        self,
        *,
        symbol: str,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> OrderFlowKey:
        return make_orderflow_key(
            exchange=exchange or self.default_exchange,
            market_type=market_type or self.default_market_type,
            symbol=symbol,
            timeframe=timeframe or self.default_timeframe,
        )

    def should_process_key(self, key: OrderFlowKey) -> bool:
        scope = orderflow_key_to_dict(key)

        if self.allowed_exchanges and scope["exchange"] not in self.allowed_exchanges:
            return False

        if self.allowed_market_types and scope["market_type"] not in self.allowed_market_types:
            return False

        if self.allowed_symbols and scope["symbol"] not in self.allowed_symbols:
            return False

        if self.allowed_timeframes and scope["timeframe"] not in self.allowed_timeframes:
            return False

        return True

    def should_process_scope(
        self,
        *,
        symbol: str,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> bool:
        return self.should_process_key(
            self.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    def scoped_mapping_key(self, key: OrderFlowKey) -> str:
        return orderflow_key_to_string(key)

    # ------------------------------------------------------------------
    # Topic guards
    # ------------------------------------------------------------------

    def is_raw_market_topic(self, topic: str) -> bool:
        return topic in RAW_ORDERFLOW_MARKET_TOPICS

    def assert_input_topic_allowed(self, topic: str) -> None:
        _validate_topic(topic, "orderflow input topic")

        if self.is_raw_market_topic(topic) and not self.allow_raw_market_topics:
            raise ValueError(
                f"Raw market topic {topic!r} is not allowed for OrderFlowAnalyzer. "
                "Use data/cache-layer topics such as market.trades.updated "
                "or market.orderbook.updated."
            )

    def assert_production_topics_allowed(self) -> None:
        for topic in self.production_input_topics:
            self.assert_input_topic_allowed(topic)

    # ------------------------------------------------------------------
    # Module helpers
    # ------------------------------------------------------------------

    def enabled_modules(self) -> tuple[str, ...]:
        modules: list[str] = []

        if self.cvd.enabled:
            modules.append("cvd")

        if self.volume_delta.enabled:
            modules.append("volume_delta")

        if self.aggressive_trades.enabled:
            modules.append("aggressive_trades")

        if self.orderbook_imbalance.enabled:
            modules.append("orderbook_imbalance")

        return tuple(modules)

    def _propagate_scope_filters_to_subconfigs(self) -> None:
        for config in (
            self.cvd,
            self.volume_delta,
            self.aggressive_trades,
            self.orderbook_imbalance,
        ):
            config.allowed_exchanges.update(self.allowed_exchanges)
            config.allowed_market_types.update(self.allowed_market_types)
            config.allowed_symbols.update(self.allowed_symbols)
            config.allowed_timeframes.update(self.allowed_timeframes)

    # ------------------------------------------------------------------
    # Validation / diagnostics
    # ------------------------------------------------------------------

    def validate(self) -> None:
        errors: list[str] = []

        if not self.default_exchange:
            errors.append("default_exchange must not be empty")

        if not self.default_market_type:
            errors.append("default_market_type must not be empty")

        if not self.default_timeframe:
            errors.append("default_timeframe must not be empty")

        if not self.allowed_market_types:
            errors.append("allowed_market_types must not be empty")

        if not self.trade_input_topics:
            errors.append("trade_input_topics must not be empty")

        if not self.orderbook_input_topics:
            errors.append("orderbook_input_topics must not be empty")

        try:
            for topic in self.production_input_topics:
                self.assert_input_topic_allowed(topic)

            for topic in self.output_topics:
                _validate_topic(topic, "orderflow output topic")

            for job_name in self.scheduler_job_names:
                _validate_job_name(job_name, "orderflow scheduler job name")

        except ValueError as exc:
            errors.append(str(exc))

        for config in (
            self.cvd,
            self.volume_delta,
            self.aggressive_trades,
            self.orderbook_imbalance,
        ):
            try:
                config.validate()
            except ValueError as exc:
                errors.append(str(exc))

        if errors:
            raise ValueError(
                f"{self.__class__.__name__} validation failed: "
                + "; ".join(errors)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "scope": "exchange:market_type:symbol:timeframe",
            "default_exchange": self.default_exchange,
            "default_market_type": self.default_market_type,
            "default_timeframe": self.default_timeframe,
            "allowed_exchanges": sorted(self.allowed_exchanges),
            "allowed_market_types": sorted(self.allowed_market_types),
            "allowed_symbols": sorted(self.allowed_symbols),
            "allowed_timeframes": sorted(self.allowed_timeframes),
            "trade_input_topics": list(self.trade_input_topics),
            "orderbook_input_topics": list(self.orderbook_input_topics),
            "source_topic_patterns_trades": list(self.source_topic_patterns_trades),
            "source_topic_patterns_orderbook": list(self.source_topic_patterns_orderbook),
            "production_input_topics": list(self.production_input_topics),
            "allow_raw_market_topics": self.allow_raw_market_topics,
            "output_topics": list(self.output_topics),
            "scheduler_job_names": list(self.scheduler_job_names),
            "enabled_modules": list(self.enabled_modules()),
            "cvd": self.cvd.to_dict(),
            "volume_delta": self.volume_delta.to_dict(),
            "aggressive_trades": self.aggressive_trades.to_dict(),
            "orderbook_imbalance": self.orderbook_imbalance.to_dict(),
            "metadata": dict(self.metadata),
        }


__all__ = [
    # scope
    "DEFAULT_EXCHANGE",
    "DEFAULT_MARKET_TYPE",
    "DEFAULT_TIMEFRAME",
    "OrderFlowKey",
    "normalize_exchange",
    "normalize_market_type",
    "normalize_symbol",
    "normalize_timeframe",
    "make_orderflow_key",
    "orderflow_key_to_dict",
    "orderflow_key_to_string",

    # topics
    "DEFAULT_TRADES_UPDATED_TOPIC",
    "DEFAULT_ORDERBOOK_UPDATED_TOPIC",
    "DEFAULT_RAW_TRADE_TOPIC",
    "DEFAULT_RAW_ORDERBOOK_TOPIC",
    "RAW_ORDERFLOW_MARKET_TOPICS",

    # configs
    "BaseOrderFlowSubConfig",
    "CvdConfig",
    "VolumeDeltaConfig",
    "AggressiveTradesConfig",
    "OrderbookImbalanceConfig",
    "OrderFlowConfig",
]