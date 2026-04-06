from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.event_bus import EventPriority

from .enums import OrderFlowEventTopic


@dataclass(slots=True)
class BaseOrderFlowSubConfig:
    enabled: bool = True
    emit_updates: bool = True
    emit_signals: bool = True

    min_signal_interval_sec: float = 0.75

    health_log_interval_sec: float = 30.0
    cleanup_interval_sec: float = 15.0

    scheduler_job_timeout_sec: float = 10.0
    scheduler_job_retry_delay_sec: float = 1.0
    scheduler_job_max_retries: int = 1

    symbol_allowlist: Optional[set[str]] = None
    publish_priority: EventPriority = EventPriority.NORMAL

    source_name: str = "orderflow_analyzer"
    update_topic: str = ""
    signal_topic: str = ""


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
    def from_app_config(cls, app_config: Any) -> "CvdConfig":
        section = _get_nested_config(app_config, "analytics", "orderflow", "cvd")
        return cls(
            enabled=getattr(section, "enabled", True),
            emit_updates=getattr(section, "emit_updates", True),
            emit_signals=getattr(section, "emit_signals", True),
            min_signal_interval_sec=getattr(section, "min_signal_interval_sec", 0.75),
            health_log_interval_sec=getattr(section, "health_log_interval_sec", 30.0),
            cleanup_interval_sec=getattr(section, "cleanup_interval_sec", 15.0),
            scheduler_job_timeout_sec=getattr(section, "scheduler_job_timeout_sec", 10.0),
            scheduler_job_retry_delay_sec=getattr(section, "scheduler_job_retry_delay_sec", 1.0),
            scheduler_job_max_retries=getattr(section, "scheduler_job_max_retries", 1),
            symbol_allowlist=_normalize_symbol_allowlist(getattr(section, "symbol_allowlist", None)),
            publish_priority=getattr(section, "publish_priority", EventPriority.NORMAL),
            source_name=getattr(section, "source_name", "cvd"),
            update_topic=getattr(section, "update_topic", OrderFlowEventTopic.CVD_UPDATED.value),
            signal_topic=getattr(section, "signal_topic", OrderFlowEventTopic.CVD_SIGNAL.value),
            window_seconds=getattr(section, "window_seconds", 20.0),
            max_trades_per_symbol=getattr(section, "max_trades_per_symbol", 8000),
            max_cvd_points_per_symbol=getattr(section, "max_cvd_points_per_symbol", 5000),
            min_trades_in_window=getattr(section, "min_trades_in_window", 12),
            min_total_volume=getattr(section, "min_total_volume", 0.0),
            bullish_delta_ratio_threshold=getattr(section, "bullish_delta_ratio_threshold", 0.15),
            bearish_delta_ratio_threshold=getattr(section, "bearish_delta_ratio_threshold", -0.15),
            bullish_cvd_change_threshold=getattr(section, "bullish_cvd_change_threshold", 0.0),
            bearish_cvd_change_threshold=getattr(section, "bearish_cvd_change_threshold", 0.0),
            bullish_cvd_slope_threshold=getattr(section, "bullish_cvd_slope_threshold", 0.0),
            bearish_cvd_slope_threshold=getattr(section, "bearish_cvd_slope_threshold", 0.0),
            bullish_impulse_threshold_pct=getattr(section, "bullish_impulse_threshold_pct", 0.0),
            bearish_impulse_threshold_pct=getattr(section, "bearish_impulse_threshold_pct", 0.0),
            require_delta_confirmation=getattr(section, "require_delta_confirmation", True),
            require_slope_confirmation=getattr(section, "require_slope_confirmation", True),
        )


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
    def from_app_config(cls, app_config: Any) -> "VolumeDeltaConfig":
        section = _get_nested_config(app_config, "analytics", "orderflow", "volume_delta")
        return cls(
            enabled=getattr(section, "enabled", True),
            emit_updates=getattr(section, "emit_updates", True),
            emit_signals=getattr(section, "emit_signals", True),
            min_signal_interval_sec=getattr(section, "min_signal_interval_sec", 0.50),
            health_log_interval_sec=getattr(section, "health_log_interval_sec", 30.0),
            cleanup_interval_sec=getattr(section, "cleanup_interval_sec", 15.0),
            scheduler_job_timeout_sec=getattr(section, "scheduler_job_timeout_sec", 10.0),
            scheduler_job_retry_delay_sec=getattr(section, "scheduler_job_retry_delay_sec", 1.0),
            scheduler_job_max_retries=getattr(section, "scheduler_job_max_retries", 1),
            symbol_allowlist=_normalize_symbol_allowlist(getattr(section, "symbol_allowlist", None)),
            publish_priority=getattr(section, "publish_priority", EventPriority.NORMAL),
            source_name=getattr(section, "source_name", "volume_delta"),
            update_topic=getattr(
                section,
                "update_topic",
                OrderFlowEventTopic.VOLUME_DELTA_UPDATED.value,
            ),
            signal_topic=getattr(
                section,
                "signal_topic",
                OrderFlowEventTopic.VOLUME_DELTA_SIGNAL.value,
            ),
            window_seconds=getattr(section, "window_seconds", 10.0),
            max_trades_per_symbol=getattr(section, "max_trades_per_symbol", 6000),
            min_trades_in_window=getattr(section, "min_trades_in_window", 10),
            min_total_volume=getattr(section, "min_total_volume", 0.0),
            bullish_delta_ratio_threshold=getattr(section, "bullish_delta_ratio_threshold", 0.18),
            bearish_delta_ratio_threshold=getattr(section, "bearish_delta_ratio_threshold", -0.18),
            bullish_volume_delta_threshold=getattr(section, "bullish_volume_delta_threshold", 0.0),
            bearish_volume_delta_threshold=getattr(section, "bearish_volume_delta_threshold", 0.0),
            bullish_cumulative_delta_threshold=getattr(
                section,
                "bullish_cumulative_delta_threshold",
                0.0,
            ),
            bearish_cumulative_delta_threshold=getattr(
                section,
                "bearish_cumulative_delta_threshold",
                0.0,
            ),
            require_ratio_and_absolute_confirmation=getattr(
                section,
                "require_ratio_and_absolute_confirmation",
                True,
            ),
        )


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
    def from_app_config(cls, app_config: Any) -> "AggressiveTradesConfig":
        section = _get_nested_config(app_config, "analytics", "orderflow", "aggressive_trades")
        return cls(
            enabled=getattr(section, "enabled", True),
            emit_updates=getattr(section, "emit_updates", True),
            emit_signals=getattr(section, "emit_signals", True),
            min_signal_interval_sec=getattr(section, "min_signal_interval_sec", 0.50),
            health_log_interval_sec=getattr(section, "health_log_interval_sec", 30.0),
            cleanup_interval_sec=getattr(section, "cleanup_interval_sec", 15.0),
            scheduler_job_timeout_sec=getattr(section, "scheduler_job_timeout_sec", 10.0),
            scheduler_job_retry_delay_sec=getattr(section, "scheduler_job_retry_delay_sec", 1.0),
            scheduler_job_max_retries=getattr(section, "scheduler_job_max_retries", 1),
            symbol_allowlist=_normalize_symbol_allowlist(getattr(section, "symbol_allowlist", None)),
            publish_priority=getattr(section, "publish_priority", EventPriority.NORMAL),
            source_name=getattr(section, "source_name", "aggressive_trades"),
            update_topic=getattr(
                section,
                "update_topic",
                OrderFlowEventTopic.AGGRESSIVE_TRADES_UPDATED.value,
            ),
            signal_topic=getattr(
                section,
                "signal_topic",
                OrderFlowEventTopic.AGGRESSIVE_TRADES_SIGNAL.value,
            ),
            window_seconds=getattr(section, "window_seconds", 8.0),
            max_trades_per_symbol=getattr(section, "max_trades_per_symbol", 5000),
            min_trades_in_window=getattr(section, "min_trades_in_window", 8),
            bullish_buy_ratio_threshold=getattr(section, "bullish_buy_ratio_threshold", 0.68),
            bearish_sell_ratio_threshold=getattr(section, "bearish_sell_ratio_threshold", 0.68),
            bullish_delta_threshold=getattr(section, "bullish_delta_threshold", 0.0),
            bearish_delta_threshold=getattr(section, "bearish_delta_threshold", 0.0),
            large_trade_notional_threshold=getattr(
                section,
                "large_trade_notional_threshold",
                25_000.0,
            ),
            min_large_trades_for_signal=getattr(section, "min_large_trades_for_signal", 1),
            burst_trades_threshold=getattr(section, "burst_trades_threshold", 12),
            burst_volume_threshold=getattr(section, "burst_volume_threshold", 0.0),
            burst_score_threshold=getattr(section, "burst_score_threshold", 1.15),
        )


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
    def from_app_config(cls, app_config: Any) -> "OrderbookImbalanceConfig":
        section = _get_nested_config(app_config, "analytics", "orderflow", "orderbook_imbalance")
        return cls(
            enabled=getattr(section, "enabled", True),
            emit_updates=getattr(section, "emit_updates", True),
            emit_signals=getattr(section, "emit_signals", True),
            min_signal_interval_sec=getattr(section, "min_signal_interval_sec", 0.30),
            health_log_interval_sec=getattr(section, "health_log_interval_sec", 30.0),
            cleanup_interval_sec=getattr(section, "cleanup_interval_sec", 15.0),
            scheduler_job_timeout_sec=getattr(section, "scheduler_job_timeout_sec", 10.0),
            scheduler_job_retry_delay_sec=getattr(section, "scheduler_job_retry_delay_sec", 1.0),
            scheduler_job_max_retries=getattr(section, "scheduler_job_max_retries", 1),
            symbol_allowlist=_normalize_symbol_allowlist(getattr(section, "symbol_allowlist", None)),
            publish_priority=getattr(section, "publish_priority", EventPriority.NORMAL),
            source_name=getattr(section, "source_name", "orderbook_imbalance"),
            update_topic=getattr(
                section,
                "update_topic",
                OrderFlowEventTopic.ORDERBOOK_IMBALANCE_UPDATED.value,
            ),
            signal_topic=getattr(
                section,
                "signal_topic",
                OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL.value,
            ),
            depth_levels=getattr(section, "depth_levels", 10),
            min_total_volume=getattr(section, "min_total_volume", 0.0),
            bullish_ratio_threshold=getattr(section, "bullish_ratio_threshold", 0.60),
            bearish_ratio_threshold=getattr(section, "bearish_ratio_threshold", 0.40),
            normalize_ratio_to_minus_one_one=getattr(
                section,
                "normalize_ratio_to_minus_one_one",
                False,
            ),
            smooth_window=getattr(section, "smooth_window", 5),
        )


@dataclass(slots=True)
class OrderFlowConfig:
    enabled: bool = True

    source_topic_patterns_trades: list[str] = field(
        default_factory=lambda: [
            "market.trade",
            "market.trade.*",
            "market.trades.updated",
            "trades.*",
        ]
    )

    source_topic_patterns_orderbook: list[str] = field(
        default_factory=lambda: [
            "market.orderbook.updated",
            "market.orderbook.snapshot",
            "orderbook.updated",
            "orderbook.*",
        ]
    )

    cvd: CvdConfig = field(default_factory=CvdConfig)
    volume_delta: VolumeDeltaConfig = field(default_factory=VolumeDeltaConfig)
    aggressive_trades: AggressiveTradesConfig = field(default_factory=AggressiveTradesConfig)
    orderbook_imbalance: OrderbookImbalanceConfig = field(default_factory=OrderbookImbalanceConfig)

    @classmethod
    def from_app_config(cls, app_config: Any) -> "OrderFlowConfig":
        section = _get_nested_config(app_config, "analytics", "orderflow")
        return cls(
            enabled=getattr(section, "enabled", True),
            source_topic_patterns_trades=list(
                getattr(
                    section,
                    "source_topic_patterns_trades",
                    [
                        "market.trade",
                        "market.trade.*",
                        "market.trades.updated",
                        "trades.*",
                    ],
                )
            ),
            source_topic_patterns_orderbook=list(
                getattr(
                    section,
                    "source_topic_patterns_orderbook",
                    [
                        "market.orderbook.updated",
                        "market.orderbook.snapshot",
                        "orderbook.updated",
                        "orderbook.*",
                    ],
                )
            ),
            cvd=CvdConfig.from_app_config(app_config),
            volume_delta=VolumeDeltaConfig.from_app_config(app_config),
            aggressive_trades=AggressiveTradesConfig.from_app_config(app_config),
            orderbook_imbalance=OrderbookImbalanceConfig.from_app_config(app_config),
        )


def _get_nested_config(root: Any, *path: str) -> Any:
    current = root
    for item in path:
        current = getattr(current, item, None)
        if current is None:
            break
    return current


def _normalize_symbol_allowlist(value: Any) -> Optional[set[str]]:
    if not value:
        return None
    return {str(item).upper() for item in value}