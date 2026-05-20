from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from .enums import CascadeSeverity, LiquidationSide
from .models import (
    CascadeDetectionResult,
    LiquidationEvent,
    LiquidationKey,
    liquidation_key_to_dict,
    make_liquidation_key,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
)
from .utils import utc_now

DECIMAL_ZERO = Decimal("0")


def _serialize_value(value: Any) -> Any:
    """
    JSON-friendly serializer для Decimal / datetime / enum / dataclass / dict / list.
    """
    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "value"):
        return value.value

    if is_dataclass(value):
        return _serialize_value(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]

    return value


def _scope_key_to_string(key: LiquidationKey) -> str:
    scope = liquidation_key_to_dict(key)
    return (
        f"{scope['exchange']}:"
        f"{scope['market_type']}:"
        f"{scope['symbol']}:"
        f"{scope['timeframe']}"
    )


def _exchange_symbol_key(
    *,
    exchange: str,
    symbol: str,
) -> str:
    return f"{normalize_exchange(exchange)}:{normalize_symbol(symbol)}"


def _market_type_key(market_type: str) -> str:
    return normalize_market_type(market_type)


def _exchange_key(exchange: str) -> str:
    return normalize_exchange(exchange)


@dataclass(slots=True)
class LatencyHistogram:
    """
    Простий latency histogram без зовнішніх залежностей.

    Це pure helper для runtime metrics:
    - не має EventBus;
    - не має Scheduler;
    - не має logger;
    - не виконує side effects.
    """

    buckets_ms: tuple[int, ...]
    counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.buckets_ms = tuple(int(bucket) for bucket in self.buckets_ms)
        self._validate()

        for bucket in self.buckets_ms:
            self.counts.setdefault(f"le_{bucket}ms", 0)

        self.counts.setdefault("gt_max", 0)

    def observe(self, latency_ms: float) -> None:
        if latency_ms < 0:
            latency_ms = 0.0

        for bucket in self.buckets_ms:
            if latency_ms <= bucket:
                self.counts[f"le_{bucket}ms"] = self.counts.get(f"le_{bucket}ms", 0) + 1
                return

        self.counts["gt_max"] = self.counts.get("gt_max", 0) + 1

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)

    def reset(self) -> None:
        for key in self.counts:
            self.counts[key] = 0

    def _validate(self) -> None:
        if not self.buckets_ms:
            raise ValueError("buckets_ms must not be empty")

        if any(bucket <= 0 for bucket in self.buckets_ms):
            raise ValueError("buckets_ms values must be > 0")

        if tuple(sorted(self.buckets_ms)) != self.buckets_ms:
            raise ValueError("buckets_ms must be sorted ascending")


@dataclass(slots=True)
class LiquidationMetricsSnapshot:
    """
    Immutable-style snapshot поточного стану liquidation metrics.

    Snapshot можна безпечно передавати у dashboard/storage через EventBus payload.

    Scope:
        exchange + market_type + symbol + timeframe
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

    # Legacy / aggregated dimensions
    symbol_event_counts: dict[str, int]
    exchange_event_counts: dict[str, int]

    cascade_by_symbol: dict[str, int]
    cascade_by_exchange: dict[str, int]

    exhaustion_by_symbol: dict[str, int]
    exhaustion_by_exchange: dict[str, int]

    # New scoped dimensions
    market_type_event_counts: dict[str, int] = field(default_factory=dict)
    scope_event_counts: dict[str, int] = field(default_factory=dict)

    cascade_by_market_type: dict[str, int] = field(default_factory=dict)
    cascade_by_scope: dict[str, int] = field(default_factory=dict)

    exhaustion_by_market_type: dict[str, int] = field(default_factory=dict)
    exhaustion_by_scope: dict[str, int] = field(default_factory=dict)

    severity_counts: dict[str, int] = field(default_factory=dict)
    latency_histogram: dict[str, int] = field(default_factory=dict)

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

    @property
    def large_ratio(self) -> float:
        if self.total_valid_events <= 0:
            return 0.0
        return self.total_large_events / self.total_valid_events

    @property
    def long_notional_ratio(self) -> float:
        total = self.total_notional_usd
        if total <= DECIMAL_ZERO:
            return 0.0
        return float(self.total_long_notional_usd / total)

    @property
    def short_notional_ratio(self) -> float:
        total = self.total_notional_usd
        if total <= DECIMAL_ZERO:
            return 0.0
        return float(self.total_short_notional_usd / total)

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["total_notional_usd"] = self.total_notional_usd
        data["valid_ratio"] = self.valid_ratio
        data["invalid_ratio"] = self.invalid_ratio
        data["stale_ratio"] = self.stale_ratio
        data["large_ratio"] = self.large_ratio
        data["long_notional_ratio"] = self.long_notional_ratio
        data["short_notional_ratio"] = self.short_notional_ratio

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
    - тримати exchange/symbol/market_type/scope counters;
    - давати snapshot() для dashboard/storage/monitoring.

    Цей клас не має залежати від EventBus, Scheduler або logger.
    Runtime-класи самі вирішують, коли публікувати metrics snapshot.

    Canonical scope:
        exchange + market_type + symbol + timeframe
    """

    keep_symbol_level_counters: bool = True
    keep_exchange_level_counters: bool = True
    keep_market_type_level_counters: bool = True
    keep_scope_level_counters: bool = True

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

    # Legacy / aggregated dimensions
    symbol_event_counts: dict[str, int] = field(default_factory=dict)
    exchange_event_counts: dict[str, int] = field(default_factory=dict)

    cascade_by_symbol: dict[str, int] = field(default_factory=dict)
    cascade_by_exchange: dict[str, int] = field(default_factory=dict)

    exhaustion_by_symbol: dict[str, int] = field(default_factory=dict)
    exhaustion_by_exchange: dict[str, int] = field(default_factory=dict)

    # New scoped dimensions
    market_type_event_counts: dict[str, int] = field(default_factory=dict)
    scope_event_counts: dict[str, int] = field(default_factory=dict)

    cascade_by_market_type: dict[str, int] = field(default_factory=dict)
    cascade_by_scope: dict[str, int] = field(default_factory=dict)

    exhaustion_by_market_type: dict[str, int] = field(default_factory=dict)
    exhaustion_by_scope: dict[str, int] = field(default_factory=dict)

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

        self.total_long_notional_usd = Decimal(str(self.total_long_notional_usd))
        self.total_short_notional_usd = Decimal(str(self.total_short_notional_usd))

        # Гарантуємо, що всі severity keys існують навіть після custom init.
        for key, value in self._default_severity_counts().items():
            self.severity_counts.setdefault(key, value)

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

    @property
    def large_ratio(self) -> float:
        if self.total_valid_events <= 0:
            return 0.0
        return self.total_large_events / self.total_valid_events

    @property
    def long_notional_ratio(self) -> float:
        total = self.total_notional_usd
        if total <= DECIMAL_ZERO:
            return 0.0
        return float(self.total_long_notional_usd / total)

    @property
    def short_notional_ratio(self) -> float:
        total = self.total_notional_usd
        if total <= DECIMAL_ZERO:
            return 0.0
        return float(self.total_short_notional_usd / total)

    # ------------------------------------------------------------------
    # Ingestion metrics
    # ------------------------------------------------------------------

    def observe_event(
        self,
        event: LiquidationEvent,
        *,
        is_valid: bool = True,
        is_stale: bool = False,
        is_large: bool = False,
    ) -> None:
        """
        Спостерігає liquidation event на ingestion/data-layer рівні.

        Важливо:
        - total_events_seen збільшується один раз на кожен event;
        - invalid/stale/large рахуються як окремі класифікації;
        - side notional рахується тільки для валідних known-side events;
        - scoped counters рахуються через event.key.
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
            self.total_long_notional_usd += Decimal(str(event.notional_usd))
        elif is_valid and event.side is LiquidationSide.SHORT:
            self.total_short_events += 1
            self.total_short_notional_usd += Decimal(str(event.notional_usd))

        self._observe_scope_counters(
            key=event.key,
            symbol_event_counts=self.symbol_event_counts,
            exchange_event_counts=self.exchange_event_counts,
            market_type_event_counts=self.market_type_event_counts,
            scope_event_counts=self.scope_event_counts,
        )

    def observe_invalid_event(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> None:
        """
        Для випадків, коли LiquidationEvent ще не створено,
        але raw payload уже визначено як invalid.

        Якщо доступний повний scope — counters оновлюються scoped.
        Якщо доступні тільки exchange/symbol — оновлюються legacy counters.
        """
        self.total_events_seen += 1
        self.total_invalid_events += 1

        key = self._try_make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        if key is not None:
            self._observe_scope_counters(
                key=key,
                symbol_event_counts=self.symbol_event_counts,
                exchange_event_counts=self.exchange_event_counts,
                market_type_event_counts=self.market_type_event_counts,
                scope_event_counts=self.scope_event_counts,
            )
            return

        if exchange and symbol and self.keep_symbol_level_counters:
            self._increment_counter(
                self.symbol_event_counts,
                self._symbol_key(exchange=exchange, symbol=symbol),
            )

        if exchange and self.keep_exchange_level_counters:
            self._increment_counter(
                self.exchange_event_counts,
                self._exchange_key(exchange),
            )

        if market_type and self.keep_market_type_level_counters:
            self._increment_counter(
                self.market_type_event_counts,
                _market_type_key(market_type),
            )

    def observe_latency_ms(self, latency_ms: float) -> None:
        self._latency_histogram.observe(latency_ms)

    # ------------------------------------------------------------------
    # Detection metrics
    # ------------------------------------------------------------------

    def observe_cascade(self, result: CascadeDetectionResult) -> None:
        """
        Рахує підтверджений cascade detection.
        """
        self.total_cascades_detected += 1

        self._observe_scope_counters(
            key=result.key,
            symbol_event_counts=self.cascade_by_symbol,
            exchange_event_counts=self.cascade_by_exchange,
            market_type_event_counts=self.cascade_by_market_type,
            scope_event_counts=self.cascade_by_scope,
        )

        self._increment_counter(self.severity_counts, result.severity.value)

    def observe_exhaustion(self, result: CascadeDetectionResult) -> None:
        """
        Рахує exhaustion detection окремо від cascade.

        Не викликає observe_cascade(), щоб не подвоювати:
        - total_cascades_detected;
        - cascade_by_* counters.
        """
        self.total_exhaustions_detected += 1

        self._observe_scope_counters(
            key=result.key,
            symbol_event_counts=self.exhaustion_by_symbol,
            exchange_event_counts=self.exhaustion_by_exchange,
            market_type_event_counts=self.exhaustion_by_market_type,
            scope_event_counts=self.exhaustion_by_scope,
        )

        self._increment_counter(self.severity_counts, result.severity.value)

    # ------------------------------------------------------------------
    # Snapshot / serialization
    # ------------------------------------------------------------------

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
            market_type_event_counts=dict(self.market_type_event_counts),
            scope_event_counts=dict(self.scope_event_counts),
            cascade_by_market_type=dict(self.cascade_by_market_type),
            cascade_by_scope=dict(self.cascade_by_scope),
            exhaustion_by_market_type=dict(self.exhaustion_by_market_type),
            exhaustion_by_scope=dict(self.exhaustion_by_scope),
            severity_counts=dict(self.severity_counts),
            latency_histogram=self._latency_histogram.snapshot(),
            metadata={
                "scope": "exchange:market_type:symbol:timeframe",
                "total_notional_usd": str(self.total_notional_usd),
                "valid_ratio": self.valid_ratio,
                "invalid_ratio": self.invalid_ratio,
                "stale_ratio": self.stale_ratio,
                "large_ratio": self.large_ratio,
                "long_notional_ratio": self.long_notional_ratio,
                "short_notional_ratio": self.short_notional_ratio,
                "tracked_symbols": len(self.symbol_event_counts),
                "tracked_exchanges": len(self.exchange_event_counts),
                "tracked_market_types": len(self.market_type_event_counts),
                "tracked_scopes": len(self.scope_event_counts),
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
        self.market_type_event_counts.clear()
        self.scope_event_counts.clear()

        self.cascade_by_symbol.clear()
        self.cascade_by_exchange.clear()
        self.cascade_by_market_type.clear()
        self.cascade_by_scope.clear()

        self.exhaustion_by_symbol.clear()
        self.exhaustion_by_exchange.clear()
        self.exhaustion_by_market_type.clear()
        self.exhaustion_by_scope.clear()

        self.severity_counts = self._default_severity_counts()
        self._latency_histogram = LatencyHistogram(self.latency_buckets_ms)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
    def _increment_counter(
        mapping: dict[str, int],
        key: str,
        value: int = 1,
    ) -> None:
        mapping[key] = mapping.get(key, 0) + value

    def _observe_scope_counters(
        self,
        *,
        key: LiquidationKey,
        symbol_event_counts: dict[str, int],
        exchange_event_counts: dict[str, int],
        market_type_event_counts: dict[str, int],
        scope_event_counts: dict[str, int],
    ) -> None:
        scope = liquidation_key_to_dict(key)

        if self.keep_symbol_level_counters:
            self._increment_counter(
                symbol_event_counts,
                self._symbol_key(
                    exchange=scope["exchange"],
                    symbol=scope["symbol"],
                ),
            )

        if self.keep_exchange_level_counters:
            self._increment_counter(
                exchange_event_counts,
                scope["exchange"],
            )

        if self.keep_market_type_level_counters:
            self._increment_counter(
                market_type_event_counts,
                scope["market_type"],
            )

        if self.keep_scope_level_counters:
            self._increment_counter(
                scope_event_counts,
                _scope_key_to_string(key),
            )

    @staticmethod
    def _try_make_key(
        *,
        exchange: str | None,
        market_type: str | None,
        symbol: str | None,
        timeframe: str | None,
    ) -> LiquidationKey | None:
        if not exchange or not symbol:
            return None

        try:
            return make_liquidation_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        except ValueError:
            return None

    @staticmethod
    def _symbol_key(
        *,
        exchange: str,
        symbol: str,
    ) -> str:
        return _exchange_symbol_key(exchange=exchange, symbol=symbol)

    @staticmethod
    def _exchange_key(exchange: str) -> str:
        return _exchange_key(exchange)


__all__ = [
    "DECIMAL_ZERO",
    "LatencyHistogram",
    "LiquidationMetricsSnapshot",
    "LiquidationMetrics",
]