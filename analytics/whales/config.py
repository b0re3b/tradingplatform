from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


# =============================================================================
# Validation helpers
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


def _validate_non_negative_mapping(
    name: str,
    values: Mapping[str, float],
) -> None:
    for key, value in values.items():
        if not key or not isinstance(key, str):
            raise ValueError(f"{name} keys must be non-empty strings")
        _validate_non_negative_number(f"{name}[{key!r}]", value)


def _validate_positive_mapping(
    name: str,
    values: Mapping[str, float],
) -> None:
    for key, value in values.items():
        if not key or not isinstance(key, str):
            raise ValueError(f"{name} keys must be non-empty strings")
        _validate_positive_number(f"{name}[{key!r}]", value)


# =============================================================================
# Large trade detector config
# =============================================================================


@dataclass(slots=True)
class LargeTradeDetectorConfig:
    """
    Config для low-level detector великих трейдів.

    Runtime-клас LargeTradeDetector має:
    - слухати input_event_name через EventBus.subscribe();
    - публікувати output_event_name через EventBus.emit();
    - запускати cleanup через Scheduler.add_interval_job().
    """

    enabled: bool = True

    # Absolute thresholds
    default_abs_notional_threshold: float = 100_000.0
    symbol_abs_thresholds: dict[str, float] = field(default_factory=dict)

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
    symbol_cooldown_sec: dict[str, float] = field(default_factory=dict)

    # Cleanup / lifecycle
    cleanup_interval_sec: float = 60.0
    stats_ttl_sec: float = 60.0 * 60.0
    recalibration_interval: int = 2_000

    # Event names
    input_event_name: str = "market.trade"
    output_event_name: str = "analytics.whales.large_trade"

    # Behavior
    emit_on_bus: bool = True
    log_signals: bool = True

    def validate(self) -> None:
        _validate_non_negative_number(
            "large_trade_detector.default_abs_notional_threshold",
            self.default_abs_notional_threshold,
        )
        _validate_non_negative_mapping(
            "large_trade_detector.symbol_abs_thresholds",
            self.symbol_abs_thresholds,
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

        _validate_non_empty_topic(
            "large_trade_detector.input_event_name",
            self.input_event_name,
        )
        _validate_non_empty_topic(
            "large_trade_detector.output_event_name",
            self.output_event_name,
        )

    def get_symbol_abs_threshold(self, symbol: str) -> float:
        return self.symbol_abs_thresholds.get(
            symbol,
            self.default_abs_notional_threshold,
        )

    def get_symbol_cooldown(self, symbol: str) -> float:
        return self.symbol_cooldown_sec.get(
            symbol,
            self.signal_cooldown_sec,
        )


# =============================================================================
# Whale tracker config
# =============================================================================


@dataclass(slots=True)
class WhaleTrackerConfig:
    """
    Config для high-level whale activity / pressure / liquidation context tracker.

    Runtime-клас WhaleTracker має:
    - слухати large_trade_event_name;
    - опційно слухати liquidation_event_name;
    - публікувати whale_activity / whale_pressure / whale_liquidation_context;
    - запускати cleanup через Scheduler.add_interval_job().
    """

    enabled: bool = True

    # Input events
    large_trade_event_name: str = "analytics.whales.large_trade"
    liquidation_event_name: str = "market.liquidation"

    # Output events
    whale_activity_event_name: str = "analytics.whales.whale_activity"
    whale_pressure_event_name: str = "analytics.whales.whale_pressure"
    whale_liquidation_context_event_name: str = (
        "analytics.whales.whale_liquidation_context"
    )

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

    @property
    def large_trade_buffer_size(self) -> int:
        return max(self.cluster_window_sec, self.pressure_window_sec) * 10

    @property
    def liquidation_buffer_size(self) -> int:
        return self.liquidation_window_sec * 10

    def validate(self) -> None:
        _validate_non_empty_topic(
            "whale_tracker.large_trade_event_name",
            self.large_trade_event_name,
        )
        _validate_non_empty_topic(
            "whale_tracker.liquidation_event_name",
            self.liquidation_event_name,
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
class WhaleClusterAnalyzerConfig:
    """
    Config для третього шару whale-аналітики.

    Runtime-клас WhaleClusterAnalyzer має:
    - слухати whale_activity / whale_pressure / whale_liquidation_context;
    - публікувати whale_cluster / whale_cluster_update / whale_cluster_exhaustion;
    - запускати cleanup через Scheduler.add_interval_job().
    """

    enabled: bool = True

    # Input events
    whale_activity_event_name: str = "analytics.whales.whale_activity"
    whale_pressure_event_name: str = "analytics.whales.whale_pressure"
    whale_liquidation_context_event_name: str = (
        "analytics.whales.whale_liquidation_context"
    )

    # Output events
    whale_cluster_event_name: str = "analytics.whales.whale_cluster"
    whale_cluster_update_event_name: str = "analytics.whales.whale_cluster_update"
    whale_cluster_exhaustion_event_name: str = (
        "analytics.whales.whale_cluster_exhaustion"
    )

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

    @property
    def activity_buffer_size(self) -> int:
        return self.analysis_window_sec * 2

    @property
    def pressure_buffer_size(self) -> int:
        return self.analysis_window_sec * 2

    @property
    def liquidation_context_buffer_size(self) -> int:
        return self.analysis_window_sec * 2

    def validate(self) -> None:
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
            raise ValueError(
                "whale_cluster_analyzer weights must sum to 1.0"
            )

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

    def validate(self) -> None:
        self.large_trade_detector.validate()
        self.whale_tracker.validate()
        self.whale_cluster_analyzer.validate()

        self._validate_pipeline_topics()

    def _validate_pipeline_topics(self) -> None:
        """
        Перевіряє, що внутрішні output/input topics між whale-компонентами
        узгоджені між собою.
        """
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