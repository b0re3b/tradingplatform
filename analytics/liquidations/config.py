from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(slots=True)
class LiquidationStreamConfig:
    """
    Конфігурація ingestion/stream-рівня liquidation events.
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


@dataclass(slots=True)
class CascadeDetectorConfig:
    """
    Конфігурація detector-а liquidation cascades.
    """

    enabled: bool = True

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


@dataclass(slots=True)
class LiquidationMetricsConfig:
    """
    Конфігурація runtime metrics liquidation-модуля.
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
    """

    stream: LiquidationStreamConfig = field(default_factory=LiquidationStreamConfig)
    cascade: CascadeDetectorConfig = field(default_factory=CascadeDetectorConfig)
    metrics: LiquidationMetricsConfig = field(default_factory=LiquidationMetricsConfig)

    def validate(self) -> None:
        if self.stream.max_buffer_size_per_symbol <= 0:
            raise ValueError("stream.max_buffer_size_per_symbol must be > 0")

        if self.stream.large_liquidation_threshold_usd <= 0:
            raise ValueError("stream.large_liquidation_threshold_usd must be > 0")

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