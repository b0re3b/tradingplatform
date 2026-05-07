from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.event_bus import EventPriority

from .enums import (
    ORDERBOOK_INPUT_TOPICS,
    TRADE_INPUT_TOPICS,
    OrderFlowEventTopic,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


_MISSING = object()


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
        str(item).strip().upper()
        for item in raw_items
        if str(item).strip()
    }

    return normalized or None


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
        "publish_priority": _normalize_priority(
            _get_attr(section, "publish_priority", EventPriority.NORMAL)
        ),
        "source_name": _get_str(section, "source_name", source_name),
        "update_topic": _get_str(section, "update_topic", update_topic),
        "signal_topic": _get_str(section, "signal_topic", signal_topic),
    }


# ---------------------------------------------------------------------
# Base sub-config
# ---------------------------------------------------------------------


@dataclass(slots=True)
class BaseOrderFlowSubConfig:
    """
    Shared config contract for all analytics.orderflow analyzers.

    Runtime dependencies such as EventBus and Scheduler are intentionally not
    stored here. They must be injected into analyzer constructors.
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

    symbol_allowlist: set[str] | None = None
    publish_priority: EventPriority = EventPriority.NORMAL

    source_name: str = "orderflow_analyzer"
    update_topic: str = ""
    signal_topic: str = ""

    def __post_init__(self) -> None:
        self.symbol_allowlist = _normalize_symbol_allowlist(self.symbol_allowlist)
        self.publish_priority = _normalize_priority(self.publish_priority)

    def should_process_symbol(self, symbol: str | None) -> bool:
        if not symbol:
            return False

        if self.symbol_allowlist is None:
            return True

        return str(symbol).strip().upper() in self.symbol_allowlist

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

        if errors:
            raise ValueError(f"{self.__class__.__name__} validation failed: " + "; ".join(errors))


# ---------------------------------------------------------------------
# Metric-specific configs
# ---------------------------------------------------------------------


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
        super().validate()

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
            raise ValueError(f"{self.__class__.__name__} validation failed: " + "; ".join(errors))


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
        super().validate()

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
            raise ValueError(f"{self.__class__.__name__} validation failed: " + "; ".join(errors))


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
        super().validate()

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
            raise ValueError(f"{self.__class__.__name__} validation failed: " + "; ".join(errors))


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
        super().validate()

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
            raise ValueError(f"{self.__class__.__name__} validation failed: " + "; ".join(errors))


# ---------------------------------------------------------------------
# Package-level config
# ---------------------------------------------------------------------


@dataclass(slots=True)
class OrderFlowConfig:
    """
    Top-level config for analytics.orderflow package.

    This object is passed to OrderFlowAnalyzer facade, which then injects
    metric-specific sub-configs into concrete analyzers.
    """

    enabled: bool = True

    source_topic_patterns_trades: list[str] = field(
        default_factory=lambda: list(TRADE_INPUT_TOPICS)
    )
    source_topic_patterns_orderbook: list[str] = field(
        default_factory=lambda: list(ORDERBOOK_INPUT_TOPICS)
    )

    cvd: CvdConfig = field(default_factory=CvdConfig)
    volume_delta: VolumeDeltaConfig = field(default_factory=VolumeDeltaConfig)
    aggressive_trades: AggressiveTradesConfig = field(default_factory=AggressiveTradesConfig)
    orderbook_imbalance: OrderbookImbalanceConfig = field(default_factory=OrderbookImbalanceConfig)

    @classmethod
    def from_app_config(cls, app_config: Any) -> OrderFlowConfig:
        section = _get_nested_config(app_config, "analytics", "orderflow")

        config = cls(
            enabled=_get_bool(section, "enabled", True),
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
            cvd=CvdConfig.from_app_config(app_config),
            volume_delta=VolumeDeltaConfig.from_app_config(app_config),
            aggressive_trades=AggressiveTradesConfig.from_app_config(app_config),
            orderbook_imbalance=OrderbookImbalanceConfig.from_app_config(app_config),
        )
        config.validate()
        return config

    def validate(self) -> None:
        errors: list[str] = []

        if not self.source_topic_patterns_trades:
            errors.append("source_topic_patterns_trades must not be empty")

        if not self.source_topic_patterns_orderbook:
            errors.append("source_topic_patterns_orderbook must not be empty")

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
            raise ValueError(f"{self.__class__.__name__} validation failed: " + "; ".join(errors))

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