from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence

from .enums import CascadeDirection, CascadeSeverity, LiquidationSide
from .models import LiquidationCluster, LiquidationEvent, LiquidationWindowStats


DECIMAL_ZERO = Decimal("0")
DECIMAL_ONE = Decimal("1")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """
    Гарантує timezone-aware UTC datetime.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def safe_decimal(value: object, default: Decimal = DECIMAL_ZERO) -> Decimal:
    """
    Безпечне приведення до Decimal.
    """
    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def normalize_symbol(symbol: str) -> str:
    """
    Нормалізація symbol до єдиного вигляду.
    """
    return symbol.strip().upper().replace("-", "").replace("/", "")


def build_symbol_key(exchange: str, symbol: str) -> tuple[str, str]:
    return exchange.strip().lower(), normalize_symbol(symbol)


def side_to_direction(side: LiquidationSide) -> CascadeDirection:
    return CascadeDirection.from_side(side)


def prune_events_older_than(
    events: Sequence[LiquidationEvent],
    min_timestamp: datetime,
) -> list[LiquidationEvent]:
    """
    Повертає лише події, що новіші або рівні min_timestamp.
    """
    min_timestamp = ensure_utc(min_timestamp)
    return [event for event in events if ensure_utc(event.timestamp) >= min_timestamp]


def filter_events_by_side(
    events: Iterable[LiquidationEvent],
    side: LiquidationSide,
) -> list[LiquidationEvent]:
    return [event for event in events if event.side == side]


def sum_notional(events: Iterable[LiquidationEvent]) -> Decimal:
    total = DECIMAL_ZERO
    for event in events:
        total += event.notional_usd
    return total


def sum_quantity(events: Iterable[LiquidationEvent]) -> Decimal:
    total = DECIMAL_ZERO
    for event in events:
        total += event.quantity
    return total


def compute_window_stats(
    exchange: str,
    symbol: str,
    events: Sequence[LiquidationEvent],
) -> LiquidationWindowStats:
    """
    Агрегує статистику для поточного sliding window.
    """
    if not events:
        now = utc_now()
        return LiquidationWindowStats(
            exchange=exchange,
            symbol=symbol,
            window_start=now,
            window_end=now,
        )

    sorted_events = sorted(events, key=lambda event: ensure_utc(event.timestamp))

    stats = LiquidationWindowStats(
        exchange=exchange,
        symbol=symbol,
        window_start=ensure_utc(sorted_events[0].timestamp),
        window_end=ensure_utc(sorted_events[-1].timestamp),
    )

    min_price: Decimal | None = None
    max_price: Decimal | None = None

    for event in sorted_events:
        stats.total_events += 1
        stats.total_notional_usd += event.notional_usd

        if event.side == LiquidationSide.LONG:
            stats.long_events += 1
            stats.long_notional_usd += event.notional_usd
        elif event.side == LiquidationSide.SHORT:
            stats.short_events += 1
            stats.short_notional_usd += event.notional_usd

        if min_price is None or event.price < min_price:
            min_price = event.price
        if max_price is None or event.price > max_price:
            max_price = event.price

    stats.min_price = min_price
    stats.max_price = max_price
    return stats


def split_events_in_halves(
    events: Sequence[LiquidationEvent],
) -> tuple[list[LiquidationEvent], list[LiquidationEvent]]:
    """
    Ділить послідовність подій на дві половини для оцінки acceleration.
    """
    if not events:
        return [], []

    midpoint = len(events) // 2
    if midpoint == 0:
        return list(events), []

    return list(events[:midpoint]), list(events[midpoint:])


def compute_acceleration_ratio(events: Sequence[LiquidationEvent]) -> float:
    """
    Порівнює силу другої половини window з першою.
    """
    first_half, second_half = split_events_in_halves(events)

    first_notional = sum_notional(first_half)
    second_notional = sum_notional(second_half)

    if first_notional <= 0:
        return float(second_notional) if second_notional > 0 else 0.0

    return float(second_notional / first_notional)


def clamp_float(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def normalize_score(value: float, reference: float) -> float:
    """
    Нормалізація метрики до [0, 1].
    """
    if reference <= 0:
        return 0.0
    return clamp_float(value / reference)


def infer_severity(
    intensity_score: float,
    low_threshold: float,
    medium_threshold: float,
    high_threshold: float,
    extreme_threshold: float,
) -> CascadeSeverity:
    """
    Перетворює intensity score у дискретний severity-рівень.
    """
    if intensity_score >= extreme_threshold:
        return CascadeSeverity.EXTREME
    if intensity_score >= high_threshold:
        return CascadeSeverity.HIGH
    if intensity_score >= medium_threshold:
        return CascadeSeverity.MEDIUM
    if intensity_score >= low_threshold:
        return CascadeSeverity.LOW
    return CascadeSeverity.LOW


def is_stale_event(event: LiquidationEvent, stale_after_seconds: int, now: datetime | None = None) -> bool:
    now = ensure_utc(now or utc_now())
    event_ts = ensure_utc(event.timestamp)
    return (now - event_ts) > timedelta(seconds=stale_after_seconds)


def build_cluster_from_events(
    exchange: str,
    symbol: str,
    side: LiquidationSide,
    events: Sequence[LiquidationEvent],
    severity: CascadeSeverity = CascadeSeverity.LOW,
) -> LiquidationCluster | None:
    """
    Будує LiquidationCluster із набору подій однієї домінантної сторони.
    """
    if not events:
        return None

    sorted_events = sorted(events, key=lambda event: ensure_utc(event.timestamp))
    prices = [event.price for event in sorted_events]

    total_notional_usd = sum_notional(sorted_events)
    total_quantity = sum_quantity(sorted_events)

    weighted_price_numerator = DECIMAL_ZERO
    for event in sorted_events:
        weighted_price_numerator += event.price * event.quantity

    if total_quantity > 0:
        avg_price = weighted_price_numerator / total_quantity
    else:
        avg_price = sorted_events[-1].price

    return LiquidationCluster(
        exchange=exchange,
        symbol=symbol,
        side=side,
        start_time=ensure_utc(sorted_events[0].timestamp),
        end_time=ensure_utc(sorted_events[-1].timestamp),
        event_count=len(sorted_events),
        total_notional_usd=total_notional_usd,
        total_quantity=total_quantity,
        avg_price=avg_price,
        min_price=min(prices),
        max_price=max(prices),
        direction=side_to_direction(side),
        severity=severity,
    )