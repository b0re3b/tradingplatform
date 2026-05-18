from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Iterable, Mapping, Sequence

from .enums import CascadeDirection, CascadeSeverity, LiquidationSide, LiquidationStatus
from .models import (
    DECIMAL_ZERO,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    LiquidationCluster,
    LiquidationEvent,
    LiquidationKey,
    LiquidationWindowStats,
    liquidation_key_to_dict,
    make_liquidation_key,
    normalize_exchange,
    normalize_exchange_symbol,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
)


DECIMAL_ONE = Decimal("1")


# =============================================================================
# Time / parsing helpers
# =============================================================================

def utc_now() -> datetime:
    """
    Поточний UTC datetime.

    Pure helper без side effects. Може використовуватись у models/state/detectors.
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


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if result != result:  # NaN
        return default

    return result


# =============================================================================
# Scope / key helpers
# =============================================================================

def build_liquidation_key(
    *,
    exchange: str,
    symbol: str,
    market_type: str | None = None,
    timeframe: str | None = None,
) -> LiquidationKey:
    """
    Canonical key для state/metrics/cache.

    Scope:
        exchange + market_type + symbol + timeframe
    """
    return make_liquidation_key(
        exchange=exchange,
        market_type=market_type or DEFAULT_MARKET_TYPE,
        symbol=symbol,
        timeframe=timeframe or DEFAULT_TIMEFRAME,
    )


def build_key_from_event(event: LiquidationEvent) -> LiquidationKey:
    """
    Витягує canonical LiquidationKey із LiquidationEvent.
    """
    return event.key


def build_key_from_payload(
    payload: Mapping[str, object],
    *,
    default_market_type: str = DEFAULT_MARKET_TYPE,
    default_timeframe: str = DEFAULT_TIMEFRAME,
) -> LiquidationKey | None:
    """
    Будує LiquidationKey із raw/data-layer payload.

    Підтримує поля:
    - exchange
    - symbol / s / instrument
    - market_type / category / market
    - timeframe
    """
    symbol = (
        payload.get("symbol")
        or payload.get("s")
        or payload.get("instrument")
    )
    exchange = payload.get("exchange")

    if not exchange or not symbol:
        return None

    try:
        return make_liquidation_key(
            exchange=exchange,
            market_type=(
                payload.get("market_type")
                or payload.get("category")
                or payload.get("market")
                or default_market_type
            ),
            symbol=symbol,
            timeframe=payload.get("timeframe") or default_timeframe,
        )
    except ValueError:
        return None


def build_symbol_key(exchange: str, symbol: str) -> tuple[str, str]:
    """
    Backward-compatible legacy key.

    Новий код має використовувати build_liquidation_key() або event.key,
    бо symbol_key не розділяє market_type/timeframe.
    """
    return normalize_exchange(exchange), normalize_symbol(symbol)


def key_to_scope(key: LiquidationKey) -> dict[str, str]:
    return liquidation_key_to_dict(key)


def scoped_key_to_string(key: LiquidationKey) -> str:
    scope = liquidation_key_to_dict(key)
    return (
        f"{scope['exchange']}:"
        f"{scope['market_type']}:"
        f"{scope['symbol']}:"
        f"{scope['timeframe']}"
    )


def ensure_same_scope(events: Sequence[LiquidationEvent]) -> LiquidationKey | None:
    """
    Перевіряє, що всі events належать одному LiquidationKey.

    Повертає key, якщо список не порожній.
    """
    if not events:
        return None

    first_key = events[0].key

    for event in events[1:]:
        if event.key != first_key:
            raise ValueError(
                "Mixed liquidation scopes are not allowed in one aggregation window: "
                f"first={liquidation_key_to_dict(first_key)} "
                f"other={liquidation_key_to_dict(event.key)}"
            )

    return first_key


def infer_scope_from_events(
    events: Sequence[LiquidationEvent],
    *,
    fallback_exchange: str | None = None,
    fallback_symbol: str | None = None,
    fallback_market_type: str | None = None,
    fallback_timeframe: str | None = None,
) -> tuple[str, str, str, str, str]:
    """
    Повертає normalized:
        exchange, market_type, symbol, timeframe, exchange_symbol

    Якщо events не порожні — scope береться з першого event і перевіряється,
    що всі events мають той самий key.
    """
    if events:
        key = ensure_same_scope(events)
        assert key is not None

        first = events[0]
        return (
            first.exchange,
            first.market_type,
            first.symbol,
            first.timeframe,
            first.exchange_symbol or first.symbol,
        )

    if fallback_exchange is None or fallback_symbol is None:
        raise ValueError("fallback_exchange and fallback_symbol are required for empty events")

    normalized_symbol = normalize_symbol(fallback_symbol)
    return (
        normalize_exchange(fallback_exchange),
        normalize_market_type(fallback_market_type),
        normalized_symbol,
        normalize_timeframe(fallback_timeframe),
        normalize_exchange_symbol(None, fallback_symbol=normalized_symbol),
    )


# =============================================================================
# Direction / filtering helpers
# =============================================================================

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


def filter_events_by_key(
    events: Iterable[LiquidationEvent],
    key: LiquidationKey,
) -> list[LiquidationEvent]:
    """
    Фільтрує events за повним liquidation scope.
    """
    return [event for event in events if event.key == key]


def filter_events_by_scope(
    events: Iterable[LiquidationEvent],
    *,
    exchange: str,
    symbol: str,
    market_type: str | None = None,
    timeframe: str | None = None,
) -> list[LiquidationEvent]:
    """
    Фільтрує events за exchange + market_type + symbol + timeframe.
    """
    key = build_liquidation_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    return filter_events_by_key(events, key)


def sort_events_by_timestamp(
    events: Sequence[LiquidationEvent],
) -> list[LiquidationEvent]:
    """
    Сортує events за UTC timestamp.
    """
    return sorted(events, key=lambda event: ensure_utc(event.timestamp))


# =============================================================================
# Aggregation helpers
# =============================================================================

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
    *,
    market_type: str | None = None,
    timeframe: str | None = None,
    exchange_symbol: str | None = None,
) -> LiquidationWindowStats:
    """
    Агрегує статистику для поточного liquidation sliding window.

    Backward-compatible signature з exchange/symbol збережено, але canonical
    scope тепер:
        exchange + market_type + symbol + timeframe

    Якщо events не порожні, scope береться з events і перевіряється,
    що всі вони мають один LiquidationKey.
    """
    if not events:
        now = utc_now()
        normalized_symbol = normalize_symbol(symbol)

        return LiquidationWindowStats(
            exchange=normalize_exchange(exchange),
            market_type=normalize_market_type(market_type),
            symbol=normalized_symbol,
            timeframe=normalize_timeframe(timeframe),
            exchange_symbol=normalize_exchange_symbol(
                exchange_symbol,
                fallback_symbol=normalized_symbol,
            ),
            window_start=now,
            window_end=now,
            metadata={
                "scope": liquidation_key_to_dict(
                    make_liquidation_key(
                        exchange=exchange,
                        market_type=market_type,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                ),
            },
        )

    (
        normalized_exchange,
        normalized_market_type,
        normalized_symbol,
        normalized_timeframe,
        normalized_exchange_symbol,
    ) = infer_scope_from_events(
        events,
        fallback_exchange=exchange,
        fallback_symbol=symbol,
        fallback_market_type=market_type,
        fallback_timeframe=timeframe,
    )

    sorted_events = sort_events_by_timestamp(events)

    stats = LiquidationWindowStats(
        exchange=normalized_exchange,
        market_type=normalized_market_type,
        symbol=normalized_symbol,
        timeframe=normalized_timeframe,
        exchange_symbol=normalized_exchange_symbol,
        window_start=ensure_utc(sorted_events[0].timestamp),
        window_end=ensure_utc(sorted_events[-1].timestamp),
        metadata={
            "scope": liquidation_key_to_dict(sorted_events[0].key),
            "first_event_id": sorted_events[0].event_id,
            "last_event_id": sorted_events[-1].event_id,
        },
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


def compute_window_stats_for_key(
    key: LiquidationKey,
    events: Sequence[LiquidationEvent],
) -> LiquidationWindowStats:
    """
    Canonical API для статистики по конкретному LiquidationKey.
    """
    scope = liquidation_key_to_dict(key)
    scoped_events = filter_events_by_key(events, key)

    return compute_window_stats(
        exchange=scope["exchange"],
        market_type=scope["market_type"],
        symbol=scope["symbol"],
        timeframe=scope["timeframe"],
        events=scoped_events,
    )


def group_events_by_key(
    events: Iterable[LiquidationEvent],
) -> dict[LiquidationKey, list[LiquidationEvent]]:
    """
    Групує events за canonical LiquidationKey.
    """
    grouped: dict[LiquidationKey, list[LiquidationEvent]] = {}

    for event in events:
        grouped.setdefault(event.key, []).append(event)

    return grouped


def compute_window_stats_by_key(
    events: Sequence[LiquidationEvent],
) -> dict[LiquidationKey, LiquidationWindowStats]:
    """
    Рахує LiquidationWindowStats окремо для кожного LiquidationKey.
    """
    result: dict[LiquidationKey, LiquidationWindowStats] = {}

    for key, scoped_events in group_events_by_key(events).items():
        result[key] = compute_window_stats_for_key(key, scoped_events)

    return result


# =============================================================================
# Acceleration / scores
# =============================================================================

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

    ensure_same_scope(events)

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

    return max(min_value, min(max_value, float(value)))


def normalize_score(value: float, reference: float) -> float:
    """
    Нормалізація метрики до [0, 1].
    """
    if reference <= 0:
        return 0.0
    return clamp_float(float(value) / float(reference))


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


# =============================================================================
# Staleness / cluster helpers
# =============================================================================

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
    *,
    market_type: str | None = None,
    timeframe: str | None = None,
    exchange_symbol: str | None = None,
) -> LiquidationCluster | None:
    """
    Будує LiquidationCluster із набору подій однієї домінантної сторони.

    Backward-compatible signature з exchange/symbol збережено, але cluster
    тепер завжди отримує повний scope:
        exchange + market_type + symbol + timeframe

    Функція не визначає, чи є cascade підтвердженим.
    Вона тільки агрегує події у cluster model.
    """
    if not events:
        return None

    if not side.is_known:
        return None

    scoped_key = ensure_same_scope(events)
    if scoped_key is None:
        return None

    sorted_events = sort_events_by_timestamp(events)
    prices = [event.price for event in sorted_events]

    total_notional_usd = sum_notional(sorted_events)
    total_quantity = sum_quantity(sorted_events)
    avg_price = compute_weighted_average_price(sorted_events)

    if avg_price <= DECIMAL_ZERO:
        avg_price = sorted_events[-1].price

    first_event = sorted_events[0]

    return LiquidationCluster(
        exchange=first_event.exchange,
        market_type=first_event.market_type,
        symbol=first_event.symbol,
        timeframe=first_event.timeframe,
        exchange_symbol=first_event.exchange_symbol,
        side=side,
        start_time=ensure_utc(first_event.timestamp),
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
            "scope": liquidation_key_to_dict(scoped_key),
            "exchange_symbol": first_event.exchange_symbol,
            "first_event_id": first_event.event_id,
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


def build_cluster_from_events_for_key(
    key: LiquidationKey,
    side: LiquidationSide,
    events: Sequence[LiquidationEvent],
    severity: CascadeSeverity = CascadeSeverity.LOW,
    status: LiquidationStatus = LiquidationStatus.CANDIDATE,
    cluster_id: str | None = None,
    source: str | None = "cascade_detector",
) -> LiquidationCluster | None:
    """
    Canonical API для побудови cluster по конкретному LiquidationKey.
    """
    scoped_events = filter_events_by_key(events, key)
    if not scoped_events:
        return None

    scope = liquidation_key_to_dict(key)

    return build_cluster_from_events(
        exchange=scope["exchange"],
        market_type=scope["market_type"],
        symbol=scope["symbol"],
        timeframe=scope["timeframe"],
        side=side,
        events=scoped_events,
        severity=severity,
        status=status,
        cluster_id=cluster_id,
        source=source,
    )


__all__ = [
    "DECIMAL_ONE",
    "utc_now",
    "ensure_utc",
    "safe_decimal",
    "safe_float",

    # scope/key
    "normalize_exchange",
    "normalize_symbol",
    "normalize_market_type",
    "normalize_timeframe",
    "normalize_exchange_symbol",
    "build_liquidation_key",
    "build_key_from_event",
    "build_key_from_payload",
    "build_symbol_key",
    "key_to_scope",
    "scoped_key_to_string",
    "ensure_same_scope",
    "infer_scope_from_events",

    # filtering
    "side_to_direction",
    "prune_events_older_than",
    "prune_events_by_window",
    "filter_events_by_side",
    "filter_valid_events",
    "filter_events_by_key",
    "filter_events_by_scope",
    "sort_events_by_timestamp",

    # aggregation
    "sum_notional",
    "sum_quantity",
    "compute_weighted_average_price",
    "compute_window_stats",
    "compute_window_stats_for_key",
    "group_events_by_key",
    "compute_window_stats_by_key",

    # scores
    "split_events_in_halves",
    "compute_acceleration_ratio",
    "clamp_float",
    "normalize_score",
    "infer_severity",

    # stale / cluster
    "is_stale_event",
    "build_cluster_from_events",
    "build_cluster_from_events_for_key",
]