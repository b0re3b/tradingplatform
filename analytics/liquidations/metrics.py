from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from .enums import CascadeSeverity, LiquidationSide
from .models import CascadeDetectionResult, LiquidationEvent
from .utils import utc_now


@dataclass(slots=True)
class LatencyHistogram:
    """
    Простий latency histogram без зовнішніх залежностей.
    """

    buckets_ms: tuple[int, ...]
    counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for bucket in self.buckets_ms:
            self.counts[f"le_{bucket}ms"] = 0
        self.counts["gt_max"] = 0

    def observe(self, latency_ms: float) -> None:
        for bucket in self.buckets_ms:
            if latency_ms <= bucket:
                self.counts[f"le_{bucket}ms"] += 1
                return
        self.counts["gt_max"] += 1

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)


@dataclass(slots=True)
class LiquidationMetricsSnapshot:
    created_at: datetime

    total_events_seen: int
    total_valid_events: int
    total_invalid_events: int
    total_stale_events: int

    total_large_events: int
    total_cascades_detected: int
    total_exhaustions_detected: int

    total_long_events: int
    total_short_events: int

    total_long_notional_usd: Decimal
    total_short_notional_usd: Decimal

    symbol_event_counts: dict[str, int]
    exchange_event_counts: dict[str, int]
    cascade_by_symbol: dict[str, int]
    cascade_by_exchange: dict[str, int]
    severity_counts: dict[str, int]

    latency_histogram: dict[str, int]


@dataclass(slots=True)
class LiquidationMetrics:
    """
    Runtime metrics для ingestion та detection pipeline.
    """

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

    total_events_seen: int = 0
    total_valid_events: int = 0
    total_invalid_events: int = 0
    total_stale_events: int = 0

    total_large_events: int = 0
    total_cascades_detected: int = 0
    total_exhaustions_detected: int = 0

    total_long_events: int = 0
    total_short_events: int = 0

    total_long_notional_usd: Decimal = Decimal("0")
    total_short_notional_usd: Decimal = Decimal("0")

    symbol_event_counts: dict[str, int] = field(default_factory=dict)
    exchange_event_counts: dict[str, int] = field(default_factory=dict)
    cascade_by_symbol: dict[str, int] = field(default_factory=dict)
    cascade_by_exchange: dict[str, int] = field(default_factory=dict)
    severity_counts: dict[str, int] = field(default_factory=lambda: {
        CascadeSeverity.LOW.value: 0,
        CascadeSeverity.MEDIUM.value: 0,
        CascadeSeverity.HIGH.value: 0,
        CascadeSeverity.EXTREME.value: 0,
    })

    def __post_init__(self) -> None:
        self._latency_histogram = LatencyHistogram(self.latency_buckets_ms)

    def _symbol_key(self, exchange: str, symbol: str) -> str:
        return f"{exchange.lower()}:{symbol.upper()}"

    def _increment_counter(self, mapping: dict[str, int], key: str, value: int = 1) -> None:
        mapping[key] = mapping.get(key, 0) + value

    def observe_event(
        self,
        event: LiquidationEvent,
        *,
        is_valid: bool = True,
        is_stale: bool = False,
        is_large: bool = False,
    ) -> None:
        self.total_events_seen += 1

        if is_valid:
            self.total_valid_events += 1
        else:
            self.total_invalid_events += 1

        if is_stale:
            self.total_stale_events += 1

        if is_large:
            self.total_large_events += 1

        if event.side == LiquidationSide.LONG:
            self.total_long_events += 1
            self.total_long_notional_usd += event.notional_usd
        elif event.side == LiquidationSide.SHORT:
            self.total_short_events += 1
            self.total_short_notional_usd += event.notional_usd

        if self.keep_symbol_level_counters:
            self._increment_counter(
                self.symbol_event_counts,
                self._symbol_key(event.exchange, event.symbol),
            )

        if self.keep_exchange_level_counters:
            self._increment_counter(self.exchange_event_counts, event.exchange.lower())

    def observe_latency_ms(self, latency_ms: float) -> None:
        self._latency_histogram.observe(latency_ms)

    def observe_cascade(self, result: CascadeDetectionResult) -> None:
        self.total_cascades_detected += 1

        if self.keep_symbol_level_counters:
            self._increment_counter(
                self.cascade_by_symbol,
                self._symbol_key(result.exchange, result.symbol),
            )

        if self.keep_exchange_level_counters:
            self._increment_counter(self.cascade_by_exchange, result.exchange.lower())

        self._increment_counter(self.severity_counts, result.severity.value)

    def observe_exhaustion(self, result: CascadeDetectionResult) -> None:
        self.total_exhaustions_detected += 1
        self.observe_cascade(result)

    def snapshot(self) -> LiquidationMetricsSnapshot:
        return LiquidationMetricsSnapshot(
            created_at=utc_now(),
            total_events_seen=self.total_events_seen,
            total_valid_events=self.total_valid_events,
            total_invalid_events=self.total_invalid_events,
            total_stale_events=self.total_stale_events,
            total_large_events=self.total_large_events,
            total_cascades_detected=self.total_cascades_detected,
            total_exhaustions_detected=self.total_exhaustions_detected,
            total_long_events=self.total_long_events,
            total_short_events=self.total_short_events,
            total_long_notional_usd=self.total_long_notional_usd,
            total_short_notional_usd=self.total_short_notional_usd,
            symbol_event_counts=dict(self.symbol_event_counts),
            exchange_event_counts=dict(self.exchange_event_counts),
            cascade_by_symbol=dict(self.cascade_by_symbol),
            cascade_by_exchange=dict(self.cascade_by_exchange),
            severity_counts=dict(self.severity_counts),
            latency_histogram=self._latency_histogram.snapshot(),
        )

    def reset(self) -> None:
        self.total_events_seen = 0
        self.total_valid_events = 0
        self.total_invalid_events = 0
        self.total_stale_events = 0
        self.total_large_events = 0
        self.total_cascades_detected = 0
        self.total_exhaustions_detected = 0
        self.total_long_events = 0
        self.total_short_events = 0
        self.total_long_notional_usd = Decimal("0")
        self.total_short_notional_usd = Decimal("0")
        self.symbol_event_counts.clear()
        self.exchange_event_counts.clear()
        self.cascade_by_symbol.clear()
        self.cascade_by_exchange.clear()
        self.severity_counts = {
            CascadeSeverity.LOW.value: 0,
            CascadeSeverity.MEDIUM.value: 0,
            CascadeSeverity.HIGH.value: 0,
            CascadeSeverity.EXTREME.value: 0,
        }
        self._latency_histogram = LatencyHistogram(self.latency_buckets_ms)