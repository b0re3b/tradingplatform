from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(slots=True)
class LiquidationStreamConfig:
    """
    Конфігурація ingestion/stream-рівня liquidation events.

    Відповідальність:
    - керує підключенням liquidation stream;
    - задає symbol/exchange scope;
    - задає publish topics для market.liquidation.*;
    - задає scheduler jobs для healthcheck / snapshot / cleanup;
    - не містить runtime state і не створює EventBus/Scheduler.
    """

    enabled: bool = True

    exchanges: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()

    max_buffer_size_per_symbol: int = 5000

    emit_raw_events: bool = False
    emit_large_events: bool = True

    large_liquidation_threshold_usd: Decimal = Decimal("100000")
    stale_event_threshold_seconds: int = 15

    publish_topic_raw: str = "market.liquidation.raw"
    publish_topic_normalized: str = "market.liquidation.normalized"
    publish_topic_large: str = "market.liquidation.large"
    publish_topic_health: str = "system.analytics.liquidations.stream.health"
    publish_topic_snapshot: str = "analytics.liquidation.stream.snapshot"

    healthcheck_interval_seconds: float = 10.0
    snapshot_interval_seconds: float = 30.0
    cleanup_interval_seconds: float = 60.0

    healthcheck_job_name: str = "liquidation_stream_healthcheck"
    snapshot_job_name: str = "liquidation_stream_snapshot"
    cleanup_job_name: str = "liquidation_stream_cleanup"

    scheduler_job_timeout_seconds: float = 5.0
    scheduler_job_max_retries: int = 1
    scheduler_job_retry_delay_seconds: float = 1.0

    reconnect_on_health_degraded: bool = True
    reconnect_cooldown_seconds: float = 10.0

    consumer_idle_sleep_seconds: float = 0.01
    consumer_error_sleep_seconds: float = 1.0

    deduplication_enabled: bool = True
    recent_payload_fingerprints_size: int = 10_000
    recent_large_events_size: int = 500


@dataclass(slots=True)
class CascadeDetectorConfig:
    """
    Конфігурація detector-а liquidation cascades.

    Відповідальність:
    - задає input topic для EventBus.subscribe();
    - задає publish topics для analytics.liquidation.*;
    - задає thresholds/scoring/cooldown;
    - задає scheduler jobs для detector snapshot / cleanup / healthcheck.
    """

    enabled: bool = True

    input_topic: str = "market.liquidation.normalized"

    window_seconds: int = 10
    min_events: int = 5
    min_total_notional_usd: Decimal = Decimal("250000")
    min_side_imbalance_ratio: float = 0.75

    cooldown_seconds: int = 15

    acceleration_enabled: bool = True
    min_acceleration_ratio: float = 1.20

    price_compaction_enabled: bool = True
    max_price_range_pct: float = 0.75

    continuation_score_weight: float = 0.40
    imbalance_score_weight: float = 0.25
    notional_score_weight: float = 0.20
    acceleration_score_weight: float = 0.15

    low_severity_threshold: float = 0.30
    medium_severity_threshold: float = 0.55
    high_severity_threshold: float = 0.75
    extreme_severity_threshold: float = 0.90

    publish_topic_detected: str = "analytics.liquidation.cascade_detected"
    publish_topic_exhaustion: str = "analytics.liquidation.exhaustion_detected"
    publish_topic_snapshot: str = "analytics.liquidation.detector.snapshot"
    publish_topic_health: str = "system.analytics.liquidations.detector.health"

    snapshot_interval_seconds: float = 30.0
    healthcheck_interval_seconds: float = 15.0
    cleanup_interval_seconds: float = 60.0

    snapshot_job_name: str = "liquidation_cascade_detector_snapshot"
    healthcheck_job_name: str = "liquidation_cascade_detector_healthcheck"
    cleanup_job_name: str = "liquidation_cascade_detector_cleanup"

    scheduler_job_timeout_seconds: float = 5.0
    scheduler_job_max_retries: int = 1
    scheduler_job_retry_delay_seconds: float = 1.0

    recent_signals_limit: int = 200


@dataclass(slots=True)
class LiquidationMetricsConfig:
    """
    Конфігурація runtime metrics liquidation-модуля.

    Metrics-клас має бути pure accumulator.
    Публікація snapshots має виконуватись runtime-класами через EventBus/Scheduler.
    """

    enabled: bool = True

    keep_symbol_level_counters: bool = True
    keep_exchange_level_counters: bool = True

    latency_buckets_ms: tuple[int, ...] = (
        1,
        5,
        10,
        25,
        50,
        100,
        250,
        500,
        1000,
        2500,
        5000,
    )


@dataclass(slots=True)
class LiquidationsConfig:
    """
    Кореневий конфіг liquidation-модуля.

    Цей config передається через dependency injection у bootstrap/container.
    Сам config не створює EventBus, Scheduler, Stream або Detector.
    """

    stream: LiquidationStreamConfig = field(default_factory=LiquidationStreamConfig)
    cascade: CascadeDetectorConfig = field(default_factory=CascadeDetectorConfig)
    metrics: LiquidationMetricsConfig = field(default_factory=LiquidationMetricsConfig)

    def validate(self) -> None:
        self._validate_stream()
        self._validate_cascade()
        self._validate_metrics()

    def _validate_stream(self) -> None:
        if self.stream.max_buffer_size_per_symbol <= 0:
            raise ValueError("stream.max_buffer_size_per_symbol must be > 0")

        if self.stream.large_liquidation_threshold_usd <= 0:
            raise ValueError("stream.large_liquidation_threshold_usd must be > 0")

        if self.stream.stale_event_threshold_seconds <= 0:
            raise ValueError("stream.stale_event_threshold_seconds must be > 0")

        if self.stream.healthcheck_interval_seconds <= 0:
            raise ValueError("stream.healthcheck_interval_seconds must be > 0")

        if self.stream.snapshot_interval_seconds <= 0:
            raise ValueError("stream.snapshot_interval_seconds must be > 0")

        if self.stream.cleanup_interval_seconds <= 0:
            raise ValueError("stream.cleanup_interval_seconds must be > 0")

        if self.stream.scheduler_job_timeout_seconds <= 0:
            raise ValueError("stream.scheduler_job_timeout_seconds must be > 0")

        if self.stream.scheduler_job_max_retries < 0:
            raise ValueError("stream.scheduler_job_max_retries must be >= 0")

        if self.stream.scheduler_job_retry_delay_seconds < 0:
            raise ValueError("stream.scheduler_job_retry_delay_seconds must be >= 0")

        if self.stream.reconnect_cooldown_seconds < 0:
            raise ValueError("stream.reconnect_cooldown_seconds must be >= 0")

        if self.stream.consumer_idle_sleep_seconds < 0:
            raise ValueError("stream.consumer_idle_sleep_seconds must be >= 0")

        if self.stream.consumer_error_sleep_seconds < 0:
            raise ValueError("stream.consumer_error_sleep_seconds must be >= 0")

        if self.stream.recent_payload_fingerprints_size <= 0:
            raise ValueError("stream.recent_payload_fingerprints_size must be > 0")

        if self.stream.recent_large_events_size <= 0:
            raise ValueError("stream.recent_large_events_size must be > 0")

        self._validate_topic(self.stream.publish_topic_raw, "stream.publish_topic_raw")
        self._validate_topic(self.stream.publish_topic_normalized, "stream.publish_topic_normalized")
        self._validate_topic(self.stream.publish_topic_large, "stream.publish_topic_large")
        self._validate_topic(self.stream.publish_topic_health, "stream.publish_topic_health")
        self._validate_topic(self.stream.publish_topic_snapshot, "stream.publish_topic_snapshot")

        self._validate_job_name(self.stream.healthcheck_job_name, "stream.healthcheck_job_name")
        self._validate_job_name(self.stream.snapshot_job_name, "stream.snapshot_job_name")
        self._validate_job_name(self.stream.cleanup_job_name, "stream.cleanup_job_name")

    def _validate_cascade(self) -> None:
        if self.cascade.window_seconds <= 0:
            raise ValueError("cascade.window_seconds must be > 0")

        if self.cascade.min_events <= 0:
            raise ValueError("cascade.min_events must be > 0")

        if self.cascade.min_total_notional_usd <= 0:
            raise ValueError("cascade.min_total_notional_usd must be > 0")

        if not (0.0 <= self.cascade.min_side_imbalance_ratio <= 1.0):
            raise ValueError("cascade.min_side_imbalance_ratio must be between 0 and 1")

        if self.cascade.cooldown_seconds < 0:
            raise ValueError("cascade.cooldown_seconds must be >= 0")

        if self.cascade.min_acceleration_ratio < 0:
            raise ValueError("cascade.min_acceleration_ratio must be >= 0")

        if self.cascade.max_price_range_pct < 0:
            raise ValueError("cascade.max_price_range_pct must be >= 0")

        weights_sum = (
            self.cascade.continuation_score_weight
            + self.cascade.imbalance_score_weight
            + self.cascade.notional_score_weight
            + self.cascade.acceleration_score_weight
        )
        if weights_sum <= 0:
            raise ValueError("cascade score weights sum must be > 0")

        for field_name, value in (
            ("cascade.continuation_score_weight", self.cascade.continuation_score_weight),
            ("cascade.imbalance_score_weight", self.cascade.imbalance_score_weight),
            ("cascade.notional_score_weight", self.cascade.notional_score_weight),
            ("cascade.acceleration_score_weight", self.cascade.acceleration_score_weight),
        ):
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0")

        thresholds = (
            self.cascade.low_severity_threshold,
            self.cascade.medium_severity_threshold,
            self.cascade.high_severity_threshold,
            self.cascade.extreme_severity_threshold,
        )

        if any(not (0.0 <= value <= 1.0) for value in thresholds):
            raise ValueError("cascade severity thresholds must be between 0 and 1")

        if list(thresholds) != sorted(thresholds):
            raise ValueError("cascade severity thresholds must be sorted ascending")

        if self.cascade.snapshot_interval_seconds <= 0:
            raise ValueError("cascade.snapshot_interval_seconds must be > 0")

        if self.cascade.healthcheck_interval_seconds <= 0:
            raise ValueError("cascade.healthcheck_interval_seconds must be > 0")

        if self.cascade.cleanup_interval_seconds <= 0:
            raise ValueError("cascade.cleanup_interval_seconds must be > 0")

        if self.cascade.scheduler_job_timeout_seconds <= 0:
            raise ValueError("cascade.scheduler_job_timeout_seconds must be > 0")

        if self.cascade.scheduler_job_max_retries < 0:
            raise ValueError("cascade.scheduler_job_max_retries must be >= 0")

        if self.cascade.scheduler_job_retry_delay_seconds < 0:
            raise ValueError("cascade.scheduler_job_retry_delay_seconds must be >= 0")

        if self.cascade.recent_signals_limit <= 0:
            raise ValueError("cascade.recent_signals_limit must be > 0")

        self._validate_topic(self.cascade.input_topic, "cascade.input_topic")
        self._validate_topic(self.cascade.publish_topic_detected, "cascade.publish_topic_detected")
        self._validate_topic(self.cascade.publish_topic_exhaustion, "cascade.publish_topic_exhaustion")
        self._validate_topic(self.cascade.publish_topic_snapshot, "cascade.publish_topic_snapshot")
        self._validate_topic(self.cascade.publish_topic_health, "cascade.publish_topic_health")

        self._validate_job_name(self.cascade.snapshot_job_name, "cascade.snapshot_job_name")
        self._validate_job_name(self.cascade.healthcheck_job_name, "cascade.healthcheck_job_name")
        self._validate_job_name(self.cascade.cleanup_job_name, "cascade.cleanup_job_name")

    def _validate_metrics(self) -> None:
        if not self.metrics.latency_buckets_ms:
            raise ValueError("metrics.latency_buckets_ms must not be empty")

        if any(bucket <= 0 for bucket in self.metrics.latency_buckets_ms):
            raise ValueError("metrics.latency_buckets_ms values must be > 0")

        if tuple(sorted(self.metrics.latency_buckets_ms)) != self.metrics.latency_buckets_ms:
            raise ValueError("metrics.latency_buckets_ms must be sorted ascending")

    @staticmethod
    def _validate_topic(value: str, field_name: str) -> None:
        if not value or not value.strip():
            raise ValueError(f"{field_name} must not be empty")

        if " " in value:
            raise ValueError(f"{field_name} must not contain spaces")

    @staticmethod
    def _validate_job_name(value: str, field_name: str) -> None:
        if not value or not value.strip():
            raise ValueError(f"{field_name} must not be empty")

        if " " in value:
            raise ValueError(f"{field_name} must not contain spaces")