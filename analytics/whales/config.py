from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(slots=True)
class LargeTradeDetectorConfig:
    """
    Конфігурація low-level detector для великих трейдів.
    """

    enabled: bool = True

    # Absolute thresholds
    default_abs_notional_threshold: float = 100_000.0
    symbol_abs_thresholds: Dict[str, float] = field(default_factory=dict)

    # Relative detection
    use_relative_detection: bool = True
    rolling_window_size: int = 300
    min_samples_for_relative_detection: int = 30
    zscore_threshold: float = 3.0

    # Basic filters
    min_notional_filter: float = 10_000.0
    side_filter: Optional[str] = None  # "buy" / "sell" / None

    # Cooldowns
    signal_cooldown_sec: float = 2.0
    symbol_cooldown_sec: Dict[str, float] = field(default_factory=dict)

    # Cleanup / lifecycle
    cleanup_interval_sec: int = 60
    stats_ttl_sec: int = 60 * 60
    recalibration_interval: int = 2_000

    # Event names
    input_event_name: str = "market.trade"
    output_event_name: str = "analytics.whales.large_trade"

    # Behavior
    emit_on_bus: bool = True
    log_signals: bool = True

    def get_symbol_abs_threshold(self, symbol: str) -> float:
        return self.symbol_abs_thresholds.get(symbol, self.default_abs_notional_threshold)

    def get_symbol_cooldown(self, symbol: str) -> float:
        return self.symbol_cooldown_sec.get(symbol, self.signal_cooldown_sec)


@dataclass(slots=True)
class WhaleTrackerConfig:
    """
    Конфігурація high-level tracker для whale activity / pressure / liquidation context.
    """

    enabled: bool = True

    # Input events
    large_trade_event_name: str = "analytics.whales.large_trade"
    liquidation_event_name: str = "market.liquidation"

    # Output events
    whale_activity_event_name: str = "analytics.whales.whale_activity"
    whale_pressure_event_name: str = "analytics.whales.whale_pressure"
    whale_liquidation_context_event_name: str = "analytics.whales.whale_liquidation_context"

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
    cleanup_interval_sec: int = 60
    stats_ttl_sec: int = 60 * 60

    # Behavior
    emit_on_bus: bool = True
    log_signals: bool = True
    subscribe_liquidations: bool = True

    @property
    def large_trade_buffer_size(self) -> int:
        return max(self.cluster_window_sec, self.pressure_window_sec) * 10

    @property
    def liquidation_buffer_size(self) -> int:
        return self.liquidation_window_sec * 10


@dataclass(slots=True)
class WhaleClusterAnalyzerConfig:
    """
    Конфігурація третього шару whale-аналітики.
    """

    enabled: bool = True

    # Input events
    whale_activity_event_name: str = "analytics.whales.whale_activity"
    whale_pressure_event_name: str = "analytics.whales.whale_pressure"
    whale_liquidation_context_event_name: str = "analytics.whales.whale_liquidation_context"

    # Output events
    whale_cluster_event_name: str = "analytics.whales.whale_cluster"
    whale_cluster_update_event_name: str = "analytics.whales.whale_cluster_update"
    whale_cluster_exhaustion_event_name: str = "analytics.whales.whale_cluster_exhaustion"

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
    cleanup_interval_sec: int = 60
    stats_ttl_sec: int = 60 * 60

    # Behavior
    emit_on_bus: bool = True
    log_signals: bool = True

    @property
    def activity_buffer_size(self) -> int:
        return self.analysis_window_sec * 2

    @property
    def pressure_buffer_size(self) -> int:
        return self.analysis_window_sec * 2

    @property
    def liquidation_context_buffer_size(self) -> int:
        return self.analysis_window_sec * 2


@dataclass(slots=True)
class WhalesConfig:
    """
    Верхньорівневий unified config для всього analytics.whales пакета.
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

    def validate(self) -> None:
        if self.large_trade_detector.default_abs_notional_threshold < 0:
            raise ValueError("default_abs_notional_threshold must be >= 0")

        if self.large_trade_detector.rolling_window_size <= 1:
            raise ValueError("rolling_window_size must be > 1")

        if self.large_trade_detector.min_samples_for_relative_detection < 2:
            raise ValueError("min_samples_for_relative_detection must be >= 2")

        if not 0.0 <= self.whale_tracker.pressure_imbalance_ratio_threshold <= 1.0:
            raise ValueError("pressure_imbalance_ratio_threshold must be in range [0, 1]")

        total_weight = (
            self.whale_cluster_analyzer.activity_weight
            + self.whale_cluster_analyzer.pressure_weight
            + self.whale_cluster_analyzer.liquidation_context_weight
            + self.whale_cluster_analyzer.persistence_weight
        )
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError(
                "whale_cluster_analyzer weights must sum to 1.0"
            )