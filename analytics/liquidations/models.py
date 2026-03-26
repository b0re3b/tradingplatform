from __future__ import annotations

from dataclasses import dataclass, field
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


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True, frozen=True)
class LiquidationEvent:
    """
    Нормалізована liquidation event-подія.

    Це базова атомарна подія, яку створює liquidation stream після
    парсингу біржового payload.
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
    source: str | None = None
    received_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_large(self) -> bool:
        return self.notional_usd > Decimal("100000")

    @property
    def is_valid(self) -> bool:
        return (
            bool(self.exchange)
            and bool(self.symbol)
            and self.side != LiquidationSide.UNKNOWN
            and self.price > 0
            and self.quantity > 0
            and self.notional_usd > 0
        )


@dataclass(slots=True)
class LiquidationCluster:
    """
    Агрегований кластер ліквідацій у часовому вікні.
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
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.end_time - self.start_time).total_seconds())

    @property
    def price_range(self) -> Decimal:
        return max(Decimal("0"), self.max_price - self.min_price)

    @property
    def avg_notional_per_event(self) -> Decimal:
        if self.event_count <= 0:
            return Decimal("0")
        return self.total_notional_usd / Decimal(self.event_count)


@dataclass(slots=True)
class CascadeDetectionResult:
    """
    Результат детекції каскаду.

    Це вже не сирі події, а аналітичний висновок detector-а.
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
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.8

    @property
    def favors_continuation(self) -> bool:
        return self.continuation_bias > self.exhaustion_bias

    @property
    def favors_exhaustion(self) -> bool:
        return self.exhaustion_bias > self.continuation_bias


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

    total_notional_usd: Decimal = Decimal("0")
    long_notional_usd: Decimal = Decimal("0")
    short_notional_usd: Decimal = Decimal("0")

    min_price: Decimal | None = None
    max_price: Decimal | None = None

    @property
    def dominant_side(self) -> LiquidationSide:
        if self.long_notional_usd > self.short_notional_usd:
            return LiquidationSide.LONG
        if self.short_notional_usd > self.long_notional_usd:
            return LiquidationSide.SHORT
        return LiquidationSide.UNKNOWN

    @property
    def dominant_notional_usd(self) -> Decimal:
        if self.dominant_side == LiquidationSide.LONG:
            return self.long_notional_usd
        if self.dominant_side == LiquidationSide.SHORT:
            return self.short_notional_usd
        return Decimal("0")

    @property
    def side_imbalance_ratio(self) -> float:
        if self.total_notional_usd <= 0:
            return 0.0
        return float(self.dominant_notional_usd / self.total_notional_usd)

    @property
    def price_range(self) -> Decimal:
        if self.min_price is None or self.max_price is None:
            return Decimal("0")
        return max(Decimal("0"), self.max_price - self.min_price)

    @property
    def price_range_pct(self) -> float:
        if self.min_price is None or self.min_price <= 0 or self.max_price is None:
            return 0.0
        return float((self.max_price - self.min_price) / self.min_price) * 100.0


@dataclass(slots=True)
class LiquidationBufferSnapshot:
    """
    Знімок буфера/state для діагностики, дебагу та метрик.
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