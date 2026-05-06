from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from .enums import (
    CascadeDirection,
    CascadeSeverity,
    LiquidationEventType,
    LiquidationSide,
    LiquidationStatus,
)


DECIMAL_ZERO = Decimal("0")
DEFAULT_LARGE_LIQUIDATION_THRESHOLD_USD = Decimal("100000")


def _utc_now() -> datetime:
    """
    Локальний default_factory для timestamp-полів.

    Це не runtime helper і не I/O. Для основної логіки часу краще використовувати
    analytics/liquidations/utils.py або root utils/time_utils.py.
    """
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    """
    Мінімальна нормалізація datetime для model-level properties.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _decimal_to_str(value: Any) -> Any:
    """
    Допоміжний serializer для Decimal/datetime у вкладених структурах.
    """
    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, datetime):
        return _ensure_utc(value).isoformat()

    if isinstance(value, dict):
        return {key: _decimal_to_str(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_decimal_to_str(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_decimal_to_str(item) for item in value)

    return value


@dataclass(slots=True, frozen=True)
class LiquidationEvent:
    """
    Нормалізована атомарна liquidation-подія.

    Створюється на stream/ingestion рівні після парсингу raw payload біржі.
    Ця модель не публікує події самостійно і не знає про EventBus.
    """

    exchange: str
    symbol: str
    side: LiquidationSide
    price: Decimal
    quantity: Decimal
    notional_usd: Decimal
    timestamp: datetime

    event_type: LiquidationEventType = LiquidationEventType.NORMALIZED

    trade_id: str | None = None
    order_id: str | None = None
    event_id: str | None = None
    correlation_id: str | None = None

    source: str | None = None
    received_at: datetime = field(default_factory=_utc_now)

    raw_payload_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_exchange(self) -> str:
        return self.exchange.strip().lower()

    @property
    def normalized_symbol(self) -> str:
        return self.symbol.strip().upper().replace("-", "").replace("/", "")

    @property
    def symbol_key(self) -> tuple[str, str]:
        return self.normalized_exchange, self.normalized_symbol

    @property
    def is_large(self) -> bool:
        """
        Backward-compatible helper.

        Для production threshold краще використовувати:
        LiquidationStreamConfig.large_liquidation_threshold_usd.
        """
        return self.notional_usd >= DEFAULT_LARGE_LIQUIDATION_THRESHOLD_USD

    def is_large_at(self, threshold_usd: Decimal) -> bool:
        return threshold_usd > DECIMAL_ZERO and self.notional_usd >= threshold_usd

    @property
    def is_valid(self) -> bool:
        return (
            bool(self.normalized_exchange)
            and bool(self.normalized_symbol)
            and self.side.is_known
            and self.price > DECIMAL_ZERO
            and self.quantity > DECIMAL_ZERO
            and self.notional_usd > DECIMAL_ZERO
        )

    @property
    def pressure_direction(self) -> CascadeDirection:
        return CascadeDirection.from_side(self.side)

    @property
    def age_seconds(self) -> float:
        now = _utc_now()
        return max(0.0, (now - _ensure_utc(self.timestamp)).total_seconds())

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["side"] = self.side.value
        data["event_type"] = self.event_type.value
        data["pressure_direction"] = self.pressure_direction.value

        if serialize:
            return _decimal_to_str(data)

        return data


@dataclass(slots=True)
class LiquidationCluster:
    """
    Агрегований кластер liquidation events у часовому вікні.

    Це доменна модель для detector-а. Вона не виконує detection самостійно,
    а лише зберігає результат агрегації.
    """

    exchange: str
    symbol: str
    side: LiquidationSide
    start_time: datetime
    end_time: datetime

    event_count: int
    total_notional_usd: Decimal
    total_quantity: Decimal

    avg_price: Decimal
    min_price: Decimal
    max_price: Decimal

    direction: CascadeDirection

    severity: CascadeSeverity = CascadeSeverity.LOW
    status: LiquidationStatus = LiquidationStatus.NEW

    cluster_id: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_exchange(self) -> str:
        return self.exchange.strip().lower()

    @property
    def normalized_symbol(self) -> str:
        return self.symbol.strip().upper().replace("-", "").replace("/", "")

    @property
    def symbol_key(self) -> tuple[str, str]:
        return self.normalized_exchange, self.normalized_symbol

    @property
    def duration_seconds(self) -> float:
        return max(
            0.0,
            (_ensure_utc(self.end_time) - _ensure_utc(self.start_time)).total_seconds(),
        )

    @property
    def price_range(self) -> Decimal:
        return max(DECIMAL_ZERO, self.max_price - self.min_price)

    @property
    def price_range_pct(self) -> float:
        if self.min_price <= DECIMAL_ZERO:
            return 0.0
        return float((self.max_price - self.min_price) / self.min_price) * 100.0

    @property
    def avg_notional_per_event(self) -> Decimal:
        if self.event_count <= 0:
            return DECIMAL_ZERO
        return self.total_notional_usd / Decimal(self.event_count)

    @property
    def is_confirmed(self) -> bool:
        return self.status is LiquidationStatus.CONFIRMED

    @property
    def is_actionable_severity(self) -> bool:
        return self.severity.is_actionable

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["side"] = self.side.value
        data["direction"] = self.direction.value
        data["severity"] = self.severity.value
        data["status"] = self.status.value
        data["duration_seconds"] = self.duration_seconds
        data["price_range"] = self.price_range
        data["price_range_pct"] = self.price_range_pct
        data["avg_notional_per_event"] = self.avg_notional_per_event

        if serialize:
            return _decimal_to_str(data)

        return data


@dataclass(slots=True)
class CascadeDetectionResult:
    """
    Результат детекції liquidation cascade.

    Це analytics-level висновок detector-а. Strategy/Risk мають сприймати його
    як вхідний аналітичний сигнал, а не як готове торгове рішення.
    """

    exchange: str
    symbol: str
    side: LiquidationSide
    direction: CascadeDirection
    detected_at: datetime
    cluster: LiquidationCluster

    intensity_score: float
    confidence: float
    continuation_bias: float
    exhaustion_bias: float

    event_count: int
    total_notional_usd: Decimal
    window_seconds: int
    price_range_pct: float

    severity: CascadeSeverity = CascadeSeverity.LOW
    status: LiquidationStatus = LiquidationStatus.CONFIRMED

    signal_id: str | None = None
    correlation_id: str | None = None
    source: str | None = "cascade_detector"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_exchange(self) -> str:
        return self.exchange.strip().lower()

    @property
    def normalized_symbol(self) -> str:
        return self.symbol.strip().upper().replace("-", "").replace("/", "")

    @property
    def symbol_key(self) -> tuple[str, str]:
        return self.normalized_exchange, self.normalized_symbol

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.8

    @property
    def is_confirmed(self) -> bool:
        return self.status is LiquidationStatus.CONFIRMED

    @property
    def is_actionable_severity(self) -> bool:
        return self.severity.is_actionable

    @property
    def favors_continuation(self) -> bool:
        return self.continuation_bias > self.exhaustion_bias

    @property
    def favors_exhaustion(self) -> bool:
        return self.exhaustion_bias > self.continuation_bias

    @property
    def bias_delta(self) -> float:
        return abs(self.continuation_bias - self.exhaustion_bias)

    @property
    def event_type(self) -> LiquidationEventType:
        if self.favors_exhaustion:
            return LiquidationEventType.EXHAUSTION
        return LiquidationEventType.CASCADE

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["side"] = self.side.value
        data["direction"] = self.direction.value
        data["severity"] = self.severity.value
        data["status"] = self.status.value
        data["event_type"] = self.event_type.value
        data["is_high_confidence"] = self.is_high_confidence
        data["is_actionable_severity"] = self.is_actionable_severity
        data["favors_continuation"] = self.favors_continuation
        data["favors_exhaustion"] = self.favors_exhaustion
        data["bias_delta"] = self.bias_delta

        if serialize:
            return _decimal_to_str(data)

        return data


@dataclass(slots=True)
class LiquidationWindowStats:
    """
    Статистика по liquidation events у конкретному sliding window.
    """

    exchange: str
    symbol: str
    window_start: datetime
    window_end: datetime

    total_events: int = 0
    long_events: int = 0
    short_events: int = 0

    total_notional_usd: Decimal = DECIMAL_ZERO
    long_notional_usd: Decimal = DECIMAL_ZERO
    short_notional_usd: Decimal = DECIMAL_ZERO

    min_price: Decimal | None = None
    max_price: Decimal | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_exchange(self) -> str:
        return self.exchange.strip().lower()

    @property
    def normalized_symbol(self) -> str:
        return self.symbol.strip().upper().replace("-", "").replace("/", "")

    @property
    def symbol_key(self) -> tuple[str, str]:
        return self.normalized_exchange, self.normalized_symbol

    @property
    def duration_seconds(self) -> float:
        return max(
            0.0,
            (_ensure_utc(self.window_end) - _ensure_utc(self.window_start)).total_seconds(),
        )

    @property
    def dominant_side(self) -> LiquidationSide:
        if self.long_notional_usd > self.short_notional_usd:
            return LiquidationSide.LONG
        if self.short_notional_usd > self.long_notional_usd:
            return LiquidationSide.SHORT
        return LiquidationSide.UNKNOWN

    @property
    def dominant_notional_usd(self) -> Decimal:
        if self.dominant_side is LiquidationSide.LONG:
            return self.long_notional_usd
        if self.dominant_side is LiquidationSide.SHORT:
            return self.short_notional_usd
        return DECIMAL_ZERO

    @property
    def dominant_events_count(self) -> int:
        if self.dominant_side is LiquidationSide.LONG:
            return self.long_events
        if self.dominant_side is LiquidationSide.SHORT:
            return self.short_events
        return 0

    @property
    def side_imbalance_ratio(self) -> float:
        if self.total_notional_usd <= DECIMAL_ZERO:
            return 0.0
        return float(self.dominant_notional_usd / self.total_notional_usd)

    @property
    def event_imbalance_ratio(self) -> float:
        if self.total_events <= 0:
            return 0.0
        return self.dominant_events_count / self.total_events

    @property
    def price_range(self) -> Decimal:
        if self.min_price is None or self.max_price is None:
            return DECIMAL_ZERO
        return max(DECIMAL_ZERO, self.max_price - self.min_price)

    @property
    def price_range_pct(self) -> float:
        if self.min_price is None or self.min_price <= DECIMAL_ZERO or self.max_price is None:
            return 0.0
        return float((self.max_price - self.min_price) / self.min_price) * 100.0

    @property
    def avg_notional_per_event(self) -> Decimal:
        if self.total_events <= 0:
            return DECIMAL_ZERO
        return self.total_notional_usd / Decimal(self.total_events)

    @property
    def has_known_dominant_side(self) -> bool:
        return self.dominant_side.is_known

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["dominant_side"] = self.dominant_side.value
        data["dominant_notional_usd"] = self.dominant_notional_usd
        data["dominant_events_count"] = self.dominant_events_count
        data["side_imbalance_ratio"] = self.side_imbalance_ratio
        data["event_imbalance_ratio"] = self.event_imbalance_ratio
        data["price_range"] = self.price_range
        data["price_range_pct"] = self.price_range_pct
        data["avg_notional_per_event"] = self.avg_notional_per_event
        data["duration_seconds"] = self.duration_seconds

        if serialize:
            return _decimal_to_str(data)

        return data


@dataclass(slots=True)
class LiquidationBufferSnapshot:
    """
    Знімок буфера/state для діагностики, dashboard, storage та metrics.
    """

    exchange: str
    symbol: str

    total_buffered_events: int
    long_buffered_events: int
    short_buffered_events: int

    first_event_at: datetime | None
    last_event_at: datetime | None
    last_cascade_at: datetime | None
    cooldown_until: datetime | None

    max_events: int | None = None
    total_events_seen: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_exchange(self) -> str:
        return self.exchange.strip().lower()

    @property
    def normalized_symbol(self) -> str:
        return self.symbol.strip().upper().replace("-", "").replace("/", "")

    @property
    def symbol_key(self) -> tuple[str, str]:
        return self.normalized_exchange, self.normalized_symbol

    @property
    def is_empty(self) -> bool:
        return self.total_buffered_events <= 0

    @property
    def is_in_cooldown(self) -> bool:
        if self.cooldown_until is None:
            return False
        return _utc_now() < _ensure_utc(self.cooldown_until)

    @property
    def dominant_buffer_side(self) -> LiquidationSide:
        if self.long_buffered_events > self.short_buffered_events:
            return LiquidationSide.LONG
        if self.short_buffered_events > self.long_buffered_events:
            return LiquidationSide.SHORT
        return LiquidationSide.UNKNOWN

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["dominant_buffer_side"] = self.dominant_buffer_side.value
        data["is_empty"] = self.is_empty
        data["is_in_cooldown"] = self.is_in_cooldown

        if serialize:
            return _decimal_to_str(data)

        return data