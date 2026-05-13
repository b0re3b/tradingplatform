from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from .enums import CascadeSeverity, LiquidationSide
from .models import CascadeDetectionResult, LiquidationEvent
from .utils import build_symbol_key, utc_now


DECIMAL_ZERO = Decimal("0")


def _serialize_value(value: Any) -> Any:
    """
    JSON-friendly serializer для Decimal / datetime / dict / list.
    """
    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_serialize_value(item) for item in value)

    return value


@dataclass(slots=True)
class LatencyHistogram:
    """
    Простий latency histogram без зовнішніх залежностей.

    Це pure helper для runtime metrics.
    """

    buckets_ms: tuple[int, ...]
    counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.buckets_ms:
            raise ValueError("buckets_ms must not be empty")

        if any(bucket <= 0 for bucket in self.buckets_ms):
            raise ValueError("buckets_ms values must be > 0")

        if tuple(sorted(self.buckets_ms)) != self.buckets_ms:
            raise ValueError("buckets_ms must be sorted ascending")

        for bucket in self.buckets_ms:
            self.counts[f"le_{bucket}ms"] = 0

        self.counts["gt_max"] = 0

    def observe(self, latency_ms: float) -> None:
        if latency_ms < 0:
            latency_ms = 0.0

        for bucket in self.buckets_ms:
            if latency_ms <= bucket:
                self.counts[f"le_{bucket}ms"] += 1
                return

        self.counts["gt_max"] += 1

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)

    def reset(self) -> None:
        for key in self.counts:
            self.counts[key] = 0


@dataclass(slots=True)
class LiquidationMetricsSnapshot:
    """
    Immutable-style snapshot поточного стану metrics.

    Snapshot можна безпечно передавати у dashboard/storage через EventBus payload.
    """

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

    exhaustion_by_symbol: dict[str, int]
    exhaustion_by_exchange: dict[str, int]

    severity_counts: dict[str, int]
    latency_histogram: dict[str, int]

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_notional_usd(self) -> Decimal:
        return self.total_long_notional_usd + self.total_short_notional_usd

    @property
    def valid_ratio(self) -> float:
        if self.total_events_seen <= 0:
            return 0.0
        return self.total_valid_events / self.total_events_seen

    @property
    def invalid_ratio(self) -> float:
        if self.total_events_seen <= 0:
            return 0.0
        return self.total_invalid_events / self.total_events_seen

    @property
    def stale_ratio(self) -> float:
        if self.total_events_seen <= 0:
            return 0.0
        return self.total_stale_events / self.total_events_seen

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["total_notional_usd"] = self.total_notional_usd
        data["valid_ratio"] = self.valid_ratio
        data["invalid_ratio"] = self.invalid_ratio
        data["stale_ratio"] = self.stale_ratio

        if serialize:
            return _serialize_value(data)

        return data


@dataclass(slots=True)
class LiquidationMetrics:
    """
    Runtime metrics для liquidation ingestion/detection pipeline.

    Відповідальність:
    - рахувати ingestion counters;
    - рахувати valid/invalid/stale/large events;
    - рахувати cascade/exhaustion detections;
    - тримати symbol/exchange counters;
    - давати snapshot() для dashboard/storage/monitoring.

    Цей клас не має залежати від EventBus, Scheduler або logger.
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

    total_long_notional_usd: Decimal = DECIMAL_ZERO
    total_short_notional_usd: Decimal = DECIMAL_ZERO

    symbol_event_counts: dict[str, int] = field(default_factory=dict)
    exchange_event_counts: dict[str, int] = field(default_factory=dict)

    cascade_by_symbol: dict[str, int] = field(default_factory=dict)
    cascade_by_exchange: dict[str, int] = field(default_factory=dict)

    exhaustion_by_symbol: dict[str, int] = field(default_factory=dict)
    exhaustion_by_exchange: dict[str, int] = field(default_factory=dict)

    severity_counts: dict[str, int] = field(
        default_factory=lambda: {
            CascadeSeverity.LOW.value: 0,
            CascadeSeverity.MEDIUM.value: 0,
            CascadeSeverity.HIGH.value: 0,
            CascadeSeverity.EXTREME.value: 0,
        }
    )

    _latency_histogram: LatencyHistogram = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate()
        self._latency_histogram = LatencyHistogram(self.latency_buckets_ms)

    @property
    def total_notional_usd(self) -> Decimal:
        return self.total_long_notional_usd + self.total_short_notional_usd

    @property
    def valid_ratio(self) -> float:
        if self.total_events_seen <= 0:
            return 0.0
        return self.total_valid_events / self.total_events_seen

    @property
    def invalid_ratio(self) -> float:
        if self.total_events_seen <= 0:
            return 0.0
        return self.total_invalid_events / self.total_events_seen

    @property
    def stale_ratio(self) -> float:
        if self.total_events_seen <= 0:
            return 0.0
        return self.total_stale_events / self.total_events_seen

    def observe_event(
        self,
        event: LiquidationEvent,
        *,
        is_valid: bool = True,
        is_stale: bool = False,
        is_large: bool = False,
    ) -> None:
        """
        Спостерігає liquidation event на ingestion-рівні.

        Важливо:
        - total_events_seen збільшується один раз на кожен event, який metrics отримав;
        - invalid/stale/large рахуються як окремі класифікації;
        - side notional рахується тільки для валідних known-side events.
        """
        self.total_events_seen += 1

        if is_valid:
            self.total_valid_events += 1
        else:
            self.total_invalid_events += 1

        if is_stale:
            self.total_stale_events += 1

        if is_large:
            self.total_large_events += 1

        if is_valid and event.side is LiquidationSide.LONG:
            self.total_long_events += 1
            self.total_long_notional_usd += event.notional_usd
        elif is_valid and event.side is LiquidationSide.SHORT:
            self.total_short_events += 1
            self.total_short_notional_usd += event.notional_usd

        if self.keep_symbol_level_counters:
            self._increment_counter(
                self.symbol_event_counts,
                self._symbol_key(event.exchange, event.symbol),
            )

        if self.keep_exchange_level_counters:
            self._increment_counter(
                self.exchange_event_counts,
                self._exchange_key(event.exchange),
            )

    def observe_invalid_event(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
    ) -> None:
        """
        Для випадків, коли LiquidationEvent ще не створено,
        але raw payload уже визначено як invalid.
        """
        self.total_events_seen += 1
        self.total_invalid_events += 1

        if exchange and symbol and self.keep_symbol_level_counters:
            self._increment_counter(
                self.symbol_event_counts,
                self._symbol_key(exchange, symbol),
            )

        if exchange and self.keep_exchange_level_counters:
            self._increment_counter(
                self.exchange_event_counts,
                self._exchange_key(exchange),
            )

    def observe_latency_ms(self, latency_ms: float) -> None:
        self._latency_histogram.observe(latency_ms)

    def observe_cascade(self, result: CascadeDetectionResult) -> None:
        """
        Рахує підтверджений cascade detection.
        """
        self.total_cascades_detected += 1

        if self.keep_symbol_level_counters:
            self._increment_counter(
                self.cascade_by_symbol,
                self._symbol_key(result.exchange, result.symbol),
            )

        if self.keep_exchange_level_counters:
            self._increment_counter(
                self.cascade_by_exchange,
                self._exchange_key(result.exchange),
            )

        self._increment_counter(self.severity_counts, result.severity.value)

    def observe_exhaustion(self, result: CascadeDetectionResult) -> None:
        """
        Рахує exhaustion detection окремо від cascade.

        Не викликає observe_cascade(), щоб не подвоювати
        total_cascades_detected і cascade_by_* counters.
        """
        self.total_exhaustions_detected += 1

        if self.keep_symbol_level_counters:
            self._increment_counter(
                self.exhaustion_by_symbol,
                self._symbol_key(result.exchange, result.symbol),
            )

        if self.keep_exchange_level_counters:
            self._increment_counter(
                self.exhaustion_by_exchange,
                self._exchange_key(result.exchange),
            )

        self._increment_counter(self.severity_counts, result.severity.value)

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
            exhaustion_by_symbol=dict(self.exhaustion_by_symbol),
            exhaustion_by_exchange=dict(self.exhaustion_by_exchange),
            severity_counts=dict(self.severity_counts),
            latency_histogram=self._latency_histogram.snapshot(),
            metadata={
                "total_notional_usd": str(self.total_notional_usd),
                "valid_ratio": self.valid_ratio,
                "invalid_ratio": self.invalid_ratio,
                "stale_ratio": self.stale_ratio,
            },
        )

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        return self.snapshot().to_dict(serialize=serialize)

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

        self.total_long_notional_usd = DECIMAL_ZERO
        self.total_short_notional_usd = DECIMAL_ZERO

        self.symbol_event_counts.clear()
        self.exchange_event_counts.clear()

        self.cascade_by_symbol.clear()
        self.cascade_by_exchange.clear()

        self.exhaustion_by_symbol.clear()
        self.exhaustion_by_exchange.clear()

        self.severity_counts = self._default_severity_counts()
        self._latency_histogram = LatencyHistogram(self.latency_buckets_ms)

    def _validate(self) -> None:
        if not self.latency_buckets_ms:
            raise ValueError("latency_buckets_ms must not be empty")

        if any(bucket <= 0 for bucket in self.latency_buckets_ms):
            raise ValueError("latency_buckets_ms values must be > 0")

        if tuple(sorted(self.latency_buckets_ms)) != self.latency_buckets_ms:
            raise ValueError("latency_buckets_ms must be sorted ascending")

    @staticmethod
    def _default_severity_counts() -> dict[str, int]:
        return {
            CascadeSeverity.LOW.value: 0,
            CascadeSeverity.MEDIUM.value: 0,
            CascadeSeverity.HIGH.value: 0,
            CascadeSeverity.EXTREME.value: 0,
        }

    @staticmethod
    def _increment_counter(mapping: dict[str, int], key: str, value: int = 1) -> None:
        mapping[key] = mapping.get(key, 0) + value

    @staticmethod
    def _symbol_key(exchange: str, symbol: str) -> str:
        normalized_exchange, normalized_symbol = build_symbol_key(exchange, symbol)
        return f"{normalized_exchange}:{normalized_symbol}"

    @staticmethod
    def _exchange_key(exchange: str) -> str:
        return exchange.strip().lower()