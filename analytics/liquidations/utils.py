from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence

from .enums import CascadeDirection, CascadeSeverity, LiquidationSide, LiquidationStatus
from .models import (
    DECIMAL_ZERO,
    LiquidationCluster,
    LiquidationEvent,
    LiquidationWindowStats,
)


DECIMAL_ONE = Decimal("1")


def utc_now() -> datetime:
    """
    Поточний UTC datetime.

    Це pure helper. Для production runtime він не створює side effects,
    тому може використовуватись у models/state/detectors.
    """
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
    Безпечне приведення значення до Decimal.

    Не кидає exception назовні, бо використовується на ingestion-рівні,
    де raw exchange payload може бути нестабільним.
    """
    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def normalize_exchange(exchange: str) -> str:
    """
    Нормалізація exchange name до єдиного вигляду.
    """
    return exchange.strip().lower()


def normalize_symbol(symbol: str) -> str:
    """
    Нормалізація symbol до єдиного вигляду.

    Приклади:
    - BTC-USDT -> BTCUSDT
    - BTC/USDT -> BTCUSDT
    - btcusdt  -> BTCUSDT
    """
    return symbol.strip().upper().replace("-", "").replace("/", "")


def build_symbol_key(exchange: str, symbol: str) -> tuple[str, str]:
    """
    Єдиний ключ для state/metrics/cache.
    """
    return normalize_exchange(exchange), normalize_symbol(symbol)


def side_to_direction(side: LiquidationSide) -> CascadeDirection:
    """
    Перетворює liquidation side у напрям очікуваного pressure/cascade.
    """
    return CascadeDirection.from_side(side)


def prune_events_older_than(
    events: Sequence[LiquidationEvent],
    min_timestamp: datetime,
) -> list[LiquidationEvent]:
    """
    Повертає лише події, що новіші або рівні min_timestamp.
    """
    min_timestamp = ensure_utc(min_timestamp)
    return [
        event
        for event in events
        if ensure_utc(event.timestamp) >= min_timestamp
    ]


def prune_events_by_window(
    events: Sequence[LiquidationEvent],
    *,
    now: datetime,
    window_seconds: int | float,
) -> list[LiquidationEvent]:
    """
    Повертає events у межах sliding window.
    """
    if window_seconds <= 0:
        return []

    min_timestamp = ensure_utc(now) - timedelta(seconds=float(window_seconds))
    return prune_events_older_than(events, min_timestamp=min_timestamp)


def filter_events_by_side(
    events: Iterable[LiquidationEvent],
    side: LiquidationSide,
) -> list[LiquidationEvent]:
    """
    Фільтрує liquidation events за стороною.
    """
    return [event for event in events if event.side is side]


def filter_valid_events(
    events: Iterable[LiquidationEvent],
) -> list[LiquidationEvent]:
    """
    Повертає тільки валідні liquidation events.
    """
    return [event for event in events if event.is_valid]


def sort_events_by_timestamp(
    events: Sequence[LiquidationEvent],
) -> list[LiquidationEvent]:
    """
    Сортує events за UTC timestamp.
    """
    return sorted(events, key=lambda event: ensure_utc(event.timestamp))


def sum_notional(events: Iterable[LiquidationEvent]) -> Decimal:
    """
    Сума notional_usd по events.
    """
    total = DECIMAL_ZERO
    for event in events:
        total += event.notional_usd
    return total


def sum_quantity(events: Iterable[LiquidationEvent]) -> Decimal:
    """
    Сума quantity по events.
    """
    total = DECIMAL_ZERO
    for event in events:
        total += event.quantity
    return total


def compute_weighted_average_price(
    events: Sequence[LiquidationEvent],
) -> Decimal:
    """
    Рахує weighted average price за quantity.

    Якщо quantity нульова або список порожній — повертає 0.
    """
    if not events:
        return DECIMAL_ZERO

    total_quantity = sum_quantity(events)
    if total_quantity <= DECIMAL_ZERO:
        return DECIMAL_ZERO

    numerator = DECIMAL_ZERO
    for event in events:
        numerator += event.price * event.quantity

    return numerator / total_quantity


def compute_window_stats(
    exchange: str,
    symbol: str,
    events: Sequence[LiquidationEvent],
) -> LiquidationWindowStats:
    """
    Агрегує статистику для поточного liquidation sliding window.
    """
    normalized_exchange, normalized_symbol = build_symbol_key(exchange, symbol)

    if not events:
        now = utc_now()
        return LiquidationWindowStats(
            exchange=normalized_exchange,
            symbol=normalized_symbol,
            window_start=now,
            window_end=now,
        )

    sorted_events = sort_events_by_timestamp(events)

    stats = LiquidationWindowStats(
        exchange=normalized_exchange,
        symbol=normalized_symbol,
        window_start=ensure_utc(sorted_events[0].timestamp),
        window_end=ensure_utc(sorted_events[-1].timestamp),
    )

    min_price: Decimal | None = None
    max_price: Decimal | None = None

    for event in sorted_events:
        stats.total_events += 1
        stats.total_notional_usd += event.notional_usd

        if event.side is LiquidationSide.LONG:
            stats.long_events += 1
            stats.long_notional_usd += event.notional_usd
        elif event.side is LiquidationSide.SHORT:
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
    if midpoint <= 0:
        return list(events), []

    return list(events[:midpoint]), list(events[midpoint:])


def compute_acceleration_ratio(events: Sequence[LiquidationEvent]) -> float:
    """
    Порівнює notional другої половини window з першою.

    Значення:
    - 0.0: немає активності;
    - 1.0: друга половина така сама за notional, як перша;
    - >1.0: прискорення liquidation flow.
    """
    if not events:
        return 0.0

    sorted_events = sort_events_by_timestamp(events)
    first_half, second_half = split_events_in_halves(sorted_events)

    first_notional = sum_notional(first_half)
    second_notional = sum_notional(second_half)

    if first_notional <= DECIMAL_ZERO:
        return float(second_notional) if second_notional > DECIMAL_ZERO else 0.0

    return float(second_notional / first_notional)


def clamp_float(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """
    Обмежує float у заданому діапазоні.
    """
    if min_value > max_value:
        raise ValueError("min_value must be <= max_value")

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

    Використовує той самий порядок threshold-ів, що й CascadeDetectorConfig.
    """
    return CascadeSeverity.from_score(
        intensity_score,
        low_threshold=low_threshold,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
        extreme_threshold=extreme_threshold,
    )


def is_stale_event(
    event: LiquidationEvent,
    stale_after_seconds: int | float,
    now: datetime | None = None,
) -> bool:
    """
    Перевіряє, чи liquidation event застарів.
    """
    if stale_after_seconds <= 0:
        return False

    current_time = ensure_utc(now or utc_now())
    event_ts = ensure_utc(event.timestamp)
    return (current_time - event_ts) > timedelta(seconds=float(stale_after_seconds))


def build_cluster_from_events(
    exchange: str,
    symbol: str,
    side: LiquidationSide,
    events: Sequence[LiquidationEvent],
    severity: CascadeSeverity = CascadeSeverity.LOW,
    status: LiquidationStatus = LiquidationStatus.CANDIDATE,
    cluster_id: str | None = None,
    source: str | None = "cascade_detector",
) -> LiquidationCluster | None:
    """
    Будує LiquidationCluster із набору подій однієї домінантної сторони.

    Функція не визначає, чи є cascade підтвердженим.
    Вона тільки агрегує події у cluster model.
    """
    if not events:
        return None

    if not side.is_known:
        return None

    sorted_events = sort_events_by_timestamp(events)
    prices = [event.price for event in sorted_events]

    total_notional_usd = sum_notional(sorted_events)
    total_quantity = sum_quantity(sorted_events)
    avg_price = compute_weighted_average_price(sorted_events)

    if avg_price <= DECIMAL_ZERO:
        avg_price = sorted_events[-1].price

    normalized_exchange, normalized_symbol = build_symbol_key(exchange, symbol)

    return LiquidationCluster(
        exchange=normalized_exchange,
        symbol=normalized_symbol,
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
        status=status,
        cluster_id=cluster_id,
        source=source,
        metadata={
            "first_event_id": sorted_events[0].event_id,
            "last_event_id": sorted_events[-1].event_id,
            "trade_ids": [
                event.trade_id
                for event in sorted_events
                if event.trade_id is not None
            ],
            "order_ids": [
                event.order_id
                for event in sorted_events
                if event.order_id is not None
            ],
        },
    )