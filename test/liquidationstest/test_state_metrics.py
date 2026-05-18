# tests/analytics/liquidations/test_state_metrics.py

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from analytics.liquidations.enums import (
    CascadeDirection,
    CascadeSeverity,
    LiquidationSide,
    LiquidationStatus,
)
from analytics.liquidations.metrics import (
    LatencyHistogram,
    LiquidationMetrics,
    LiquidationMetricsSnapshot,
)
from analytics.liquidations.models import (
    CascadeDetectionResult,
    LiquidationCluster,
    LiquidationEvent,
    LiquidationKey,
    liquidation_key_to_dict,
    make_liquidation_key,
)
from analytics.liquidations.state import LiquidationState, SymbolLiquidationState


# =============================================================================
# Helpers
# =============================================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _scope_key_to_string(key: LiquidationKey) -> str:
    scope = liquidation_key_to_dict(key)
    return (
        f"{scope['exchange']}:"
        f"{scope['market_type']}:"
        f"{scope['symbol']}:"
        f"{scope['timeframe']}"
    )


def _make_event(
    *,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    market_type: str = "usdm_futures",
    timeframe: str = "realtime",
    exchange_symbol: str | None = None,
    side: LiquidationSide = LiquidationSide.LONG,
    price: Decimal = Decimal("65000"),
    quantity: Decimal = Decimal("1"),
    notional_usd: Decimal | None = None,
    timestamp: datetime | None = None,
    trade_id: str | None = None,
    order_id: str | None = None,
    correlation_id: str | None = "test-correlation-id",
    source: str = "test",
) -> LiquidationEvent:
    resolved_notional = notional_usd if notional_usd is not None else price * quantity

    return LiquidationEvent(
        exchange=exchange,
        symbol=symbol,
        market_type=market_type,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol or symbol,
        side=side,
        price=price,
        quantity=quantity,
        notional_usd=resolved_notional,
        timestamp=timestamp or _utc_now(),
        trade_id=trade_id,
        order_id=order_id,
        correlation_id=correlation_id,
        source=source,
    )


def _make_cluster(
    *,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    market_type: str = "usdm_futures",
    timeframe: str = "realtime",
    exchange_symbol: str | None = None,
    side: LiquidationSide = LiquidationSide.LONG,
    severity: CascadeSeverity = CascadeSeverity.HIGH,
    status: LiquidationStatus = LiquidationStatus.CONFIRMED,
) -> LiquidationCluster:
    now = _utc_now()

    return LiquidationCluster(
        exchange=exchange,
        symbol=symbol,
        market_type=market_type,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol or symbol,
        side=side,
        start_time=now - timedelta(seconds=3),
        end_time=now,
        event_count=3,
        total_notional_usd=Decimal("195000"),
        total_quantity=Decimal("3"),
        avg_price=Decimal("65000"),
        min_price=Decimal("64900"),
        max_price=Decimal("65100"),
        direction=(
            CascadeDirection.DOWN
            if side is LiquidationSide.LONG
            else CascadeDirection.UP
        ),
        severity=severity,
        status=status,
        source="test",
    )


def _make_detection_result(
    *,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    market_type: str = "usdm_futures",
    timeframe: str = "realtime",
    exchange_symbol: str | None = None,
    side: LiquidationSide = LiquidationSide.LONG,
    severity: CascadeSeverity = CascadeSeverity.HIGH,
    correlation_id: str | None = "test-correlation-id",
) -> CascadeDetectionResult:
    cluster = _make_cluster(
        exchange=exchange,
        symbol=symbol,
        market_type=market_type,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
        side=side,
        severity=severity,
    )

    return CascadeDetectionResult(
        exchange=exchange,
        symbol=symbol,
        market_type=market_type,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol or symbol,
        side=side,
        direction=cluster.direction,
        detected_at=_utc_now(),
        cluster=cluster,
        intensity_score=0.85,
        confidence=0.80,
        continuation_bias=0.70,
        exhaustion_bias=0.20,
        event_count=cluster.event_count,
        total_notional_usd=cluster.total_notional_usd,
        window_seconds=10,
        price_range_pct=cluster.price_range_pct,
        severity=severity,
        status=LiquidationStatus.CONFIRMED,
        correlation_id=correlation_id,
        source="test",
    )


# =============================================================================
# SymbolLiquidationState
# =============================================================================

def test_symbol_state_normalizes_full_scope() -> None:
    state = SymbolLiquidationState(
        exchange="Binance",
        symbol="btc-usdt",
        market_type="USDM_FUTURES",
        timeframe="Realtime",
        exchange_symbol="BTCUSDT",
        max_events=10,
    )

    assert state.exchange == "binance"
    assert state.symbol == "BTCUSDT"
    assert state.market_type == "usdm_futures"
    assert state.timeframe == "realtime"
    assert state.exchange_symbol == "BTCUSDT"
    assert state.key == ("binance", "usdm_futures", "BTCUSDT", "realtime")
    assert state.scope == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
        "exchange_symbol": "BTCUSDT",
    }


def test_symbol_state_add_event_updates_counters_for_same_full_scope() -> None:
    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        max_events=10,
    )

    long_event = _make_event(side=LiquidationSide.LONG)
    short_event = _make_event(side=LiquidationSide.SHORT)

    state.add_event(long_event)
    state.add_event(short_event)

    assert state.total_buffered_events == 2
    assert state.total_events_seen == 2
    assert state.long_events_count == 1
    assert state.short_events_count == 1
    assert state.last_event_at == short_event.timestamp
    assert state.last_long_event_at == long_event.timestamp
    assert state.last_short_event_at == short_event.timestamp
    assert state.buffered_notional_usd == Decimal("130000")


def test_symbol_state_rejects_same_symbol_different_market_type() -> None:
    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        max_events=10,
    )

    coinm_event = _make_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
        exchange_symbol="BTCUSD_PERP",
    )

    with pytest.raises(ValueError, match="Event key mismatch"):
        state.add_event(coinm_event)


def test_symbol_state_rejects_same_symbol_different_timeframe() -> None:
    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        max_events=10,
    )

    one_minute_event = _make_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="1m",
    )

    with pytest.raises(ValueError, match="Event key mismatch"):
        state.add_event(one_minute_event)


def test_symbol_state_bounded_buffer_evicts_oldest_and_updates_side_counters() -> None:
    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        max_events=2,
    )

    first = _make_event(side=LiquidationSide.LONG, trade_id="1")
    second = _make_event(side=LiquidationSide.SHORT, trade_id="2")
    third = _make_event(side=LiquidationSide.SHORT, trade_id="3")

    state.add_event(first)
    state.add_event(second)
    state.add_event(third)

    assert list(state.events) == [second, third]
    assert state.total_buffered_events == 2
    assert state.total_events_seen == 3
    assert state.long_events_count == 0
    assert state.short_events_count == 2
    assert state.buffered_notional_usd == Decimal("130000")


def test_symbol_state_get_recent_events_returns_newest_first_and_filters_side() -> None:
    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        max_events=10,
    )

    now = _utc_now()
    first = _make_event(
        side=LiquidationSide.LONG,
        timestamp=now - timedelta(seconds=3),
        trade_id="1",
    )
    second = _make_event(
        side=LiquidationSide.SHORT,
        timestamp=now - timedelta(seconds=2),
        trade_id="2",
    )
    third = _make_event(
        side=LiquidationSide.LONG,
        timestamp=now - timedelta(seconds=1),
        trade_id="3",
    )

    state.extend_events([first, second, third])

    recent_longs = state.get_recent_events(
        side=LiquidationSide.LONG,
        limit=2,
    )

    assert recent_longs == [third, first]


def test_symbol_state_get_window_events_returns_only_events_inside_window() -> None:
    now = _utc_now()

    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        max_events=10,
    )

    old_event = _make_event(timestamp=now - timedelta(minutes=5), trade_id="old")
    fresh_event = _make_event(timestamp=now - timedelta(seconds=5), trade_id="fresh")

    state.extend_events([old_event, fresh_event])

    window_events = state.get_window_events(
        min_timestamp=now - timedelta(seconds=30),
    )

    assert window_events == [fresh_event]


def test_symbol_state_prune_before_removes_old_events_but_preserves_total_seen() -> None:
    now = _utc_now()

    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        max_events=10,
    )

    old_event = _make_event(timestamp=now - timedelta(minutes=10), trade_id="old")
    fresh_event = _make_event(timestamp=now, trade_id="fresh")

    state.extend_events([old_event, fresh_event])

    removed = state.prune_before(now - timedelta(minutes=1))

    assert removed == 1
    assert list(state.events) == [fresh_event]
    assert state.total_buffered_events == 1
    assert state.total_events_seen == 2
    assert state.long_events_count == 1


def test_symbol_state_cooldown_lifecycle_is_scope_local() -> None:
    now = _utc_now()

    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        max_events=10,
    )

    state.set_cascade_detected(
        detected_at=now,
        cooldown_until=now + timedelta(seconds=10),
    )

    assert state.last_cascade_at == now
    assert state.is_in_cooldown(now) is True
    assert state.is_in_cooldown(now + timedelta(seconds=11)) is False

    state.clear_cooldown()

    assert state.cooldown_until is None
    assert state.is_in_cooldown(now) is False


def test_symbol_state_clear_can_preserve_total_seen() -> None:
    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        max_events=10,
    )

    state.extend_events(
        [
            _make_event(trade_id="1"),
            _make_event(trade_id="2"),
        ]
    )

    state.clear(reset_total_seen=False)

    assert state.total_buffered_events == 0
    assert state.total_events_seen == 2
    assert state.long_events_count == 0
    assert state.short_events_count == 0
    assert state.last_event_at is None


def test_symbol_state_snapshot_contains_full_scope_and_runtime_metadata() -> None:
    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        exchange_symbol="BTCUSDT",
        max_events=10,
    )

    state.add_event(
        _make_event(
            quantity=Decimal("2"),
            trade_id="snapshot-event",
        )
    )

    snapshot = state.snapshot()

    assert snapshot.exchange == "binance"
    assert snapshot.market_type == "usdm_futures"
    assert snapshot.symbol == "BTCUSDT"
    assert snapshot.timeframe == "realtime"
    assert snapshot.exchange_symbol == "BTCUSDT"
    assert snapshot.total_buffered_events == 1
    assert snapshot.long_buffered_events == 1
    assert snapshot.short_buffered_events == 0
    assert snapshot.max_events == 10
    assert snapshot.metadata["scope"] == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
    }
    assert snapshot.metadata["exchange_symbol"] == "BTCUSDT"
    assert snapshot.metadata["is_in_cooldown"] is False
    assert snapshot.metadata["buffered_notional_usd"] == "130000"


# =============================================================================
# LiquidationState
# =============================================================================

def test_liquidation_state_rejects_invalid_max_events_per_symbol() -> None:
    with pytest.raises(ValueError, match="max_events_per_symbol must be > 0"):
        LiquidationState(max_events_per_symbol=0)


def test_liquidation_state_add_event_creates_state_for_full_scope() -> None:
    state = LiquidationState(max_events_per_symbol=10)
    event = _make_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    symbol_state = state.add_event(event)

    assert symbol_state is state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    assert symbol_state is state.get_key(event.key)
    assert symbol_state.total_buffered_events == 1
    assert state.symbols_count == 1
    assert state.scopes_count == 1
    assert state.total_buffered_events == 1
    assert state.total_events_seen == 1


def test_liquidation_state_legacy_get_without_market_type_does_not_match_futures_scope() -> None:
    state = LiquidationState(max_events_per_symbol=10)
    event = _make_event(
        market_type="usdm_futures",
        timeframe="realtime",
    )

    state.add_event(event)

    assert state.get("binance", "BTCUSDT") is None
    assert state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    ) is not None


def test_liquidation_state_get_or_create_is_idempotent_for_same_full_scope() -> None:
    state = LiquidationState(max_events_per_symbol=10)

    first = state.get_or_create(
        "binance",
        "btc-usdt",
        market_type="USDM_FUTURES",
        timeframe="Realtime",
    )
    second = state.get_or_create(
        "BINANCE",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert first is second
    assert state.symbols_count == 1
    assert first.key == ("binance", "usdm_futures", "BTCUSDT", "realtime")


def test_liquidation_state_separates_same_symbol_by_market_type() -> None:
    state = LiquidationState(max_events_per_symbol=10)

    usdm_event = _make_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        exchange_symbol="BTCUSDT",
        trade_id="usdm",
    )
    coinm_event = _make_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
        exchange_symbol="BTCUSD_PERP",
        trade_id="coinm",
    )

    usdm_state = state.add_event(usdm_event)
    coinm_state = state.add_event(coinm_event)

    assert usdm_state is not coinm_state
    assert state.scopes_count == 2
    assert state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    ) is usdm_state
    assert state.get(
        "binance",
        "BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
    ) is coinm_state
    assert usdm_state.exchange_symbol == "BTCUSDT"
    assert coinm_state.exchange_symbol == "BTCUSD_PERP"


def test_liquidation_state_separates_same_symbol_by_timeframe() -> None:
    state = LiquidationState(max_events_per_symbol=10)

    realtime_event = _make_event(
        market_type="usdm_futures",
        timeframe="realtime",
        trade_id="realtime",
    )
    one_minute_event = _make_event(
        market_type="usdm_futures",
        timeframe="1m",
        trade_id="1m",
    )

    realtime_state = state.add_event(realtime_event)
    one_minute_state = state.add_event(one_minute_event)

    assert realtime_state is not one_minute_state
    assert state.scopes_count == 2
    assert state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    ) is realtime_state
    assert state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="1m",
    ) is one_minute_state


def test_liquidation_state_get_recent_events_can_aggregate_across_matching_scopes() -> None:
    now = _utc_now()
    state = LiquidationState(max_events_per_symbol=10)

    older_usdm = _make_event(
        market_type="usdm_futures",
        timeframe="realtime",
        timestamp=now - timedelta(seconds=3),
        trade_id="older-usdm",
    )
    newer_coinm = _make_event(
        market_type="coinm_futures",
        timeframe="realtime",
        exchange_symbol="BTCUSD_PERP",
        timestamp=now - timedelta(seconds=1),
        trade_id="newer-coinm",
    )

    state.add_events([older_usdm, newer_coinm])

    aggregated = state.get_recent_events(
        exchange="binance",
        symbol="BTCUSDT",
        limit=10,
    )

    usdm_only = state.get_recent_events(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        limit=10,
    )

    assert aggregated == [newer_coinm, older_usdm]
    assert usdm_only == [older_usdm]


def test_liquidation_state_get_recent_events_filters_by_side_and_limit() -> None:
    now = _utc_now()
    state = LiquidationState(max_events_per_symbol=10)

    first_long = _make_event(
        side=LiquidationSide.LONG,
        timestamp=now - timedelta(seconds=3),
        trade_id="first-long",
    )
    short = _make_event(
        side=LiquidationSide.SHORT,
        timestamp=now - timedelta(seconds=2),
        trade_id="short",
    )
    second_long = _make_event(
        side=LiquidationSide.LONG,
        timestamp=now - timedelta(seconds=1),
        trade_id="second-long",
    )

    state.add_events([first_long, short, second_long])

    recent_longs = state.get_recent_events(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        side=LiquidationSide.LONG,
        limit=1,
    )

    assert recent_longs == [second_long]


def test_liquidation_state_prune_before_removes_empty_scopes() -> None:
    now = _utc_now()
    state = LiquidationState(max_events_per_symbol=10)

    old_event = _make_event(timestamp=now - timedelta(minutes=10), trade_id="old")
    fresh_event = _make_event(
        symbol="ETHUSDT",
        timestamp=now,
        trade_id="fresh",
    )

    state.add_events([old_event, fresh_event])

    removed = state.prune_before(now - timedelta(minutes=1))

    assert removed == 1
    assert state.scopes_count == 1
    assert state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    ) is None
    assert state.get(
        "binance",
        "ETHUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    ) is not None


def test_liquidation_state_remove_key_and_remove_empty() -> None:
    state = LiquidationState(max_events_per_symbol=10)

    event = _make_event()
    symbol_state = state.add_event(event)

    symbol_state.clear()
    removed_empty = state.remove_empty()

    assert removed_empty == 1
    assert state.scopes_count == 0

    state.add_event(event)
    state.remove_key(event.key)

    assert state.scopes_count == 0


def test_liquidation_state_snapshot_by_key_and_snapshot_by_symbol_use_full_scope() -> None:
    state = LiquidationState(max_events_per_symbol=10)
    event = _make_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    state.add_event(event)

    snapshot_by_key = state.snapshot_by_key(event.key)
    snapshot_by_symbol = state.snapshot_by_symbol(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    legacy_snapshot = state.snapshot_by_symbol("binance", "BTCUSDT")

    assert snapshot_by_key is not None
    assert snapshot_by_symbol is not None
    assert legacy_snapshot is None
    assert snapshot_by_key.market_type == "usdm_futures"
    assert snapshot_by_symbol.timeframe == "realtime"


def test_liquidation_state_to_dict_contains_scoped_keys() -> None:
    state = LiquidationState(max_events_per_symbol=10)
    event = _make_event()

    state.add_event(event)

    data = state.to_dict()
    scope_key = _scope_key_to_string(event.key)

    assert data["scopes_count"] == 1
    assert data["symbols_count"] == 1
    assert data["total_buffered_events"] == 1
    assert data["total_events_seen"] == 1
    assert scope_key in data["scopes"]


def test_liquidation_state_clear_reset_total_seen_removes_all_scopes() -> None:
    state = LiquidationState(max_events_per_symbol=10)

    state.add_event(_make_event(symbol="BTCUSDT"))
    state.add_event(_make_event(symbol="ETHUSDT"))

    assert state.scopes_count == 2

    state.clear(reset_total_seen=True)

    assert state.scopes_count == 0
    assert state.symbols_count == 0
    assert state.snapshots() == []


def test_liquidation_state_clear_without_reset_preserves_scopes_and_total_seen() -> None:
    state = LiquidationState(max_events_per_symbol=10)

    state.add_event(_make_event(symbol="BTCUSDT"))
    state.add_event(_make_event(symbol="ETHUSDT"))

    state.clear(reset_total_seen=False)

    assert state.scopes_count == 2
    assert state.total_buffered_events == 0
    assert state.total_events_seen == 2
    assert all(snapshot.total_buffered_events == 0 for snapshot in state.snapshots())


# =============================================================================
# LatencyHistogram
# =============================================================================

def test_latency_histogram_observe_and_snapshot() -> None:
    histogram = LatencyHistogram(
        buckets_ms=(10, 50, 100),
    )

    histogram.observe(-5)
    histogram.observe(5)
    histogram.observe(50)
    histogram.observe(250)

    snapshot = histogram.snapshot()

    assert snapshot["le_10ms"] == 2
    assert snapshot["le_50ms"] == 1
    assert snapshot["le_100ms"] == 0
    assert snapshot["gt_max"] == 1


def test_latency_histogram_snapshot_returns_copy() -> None:
    histogram = LatencyHistogram(
        buckets_ms=(10, 50, 100),
    )

    histogram.observe(5)

    snapshot = histogram.snapshot()
    snapshot["le_10ms"] = 999

    assert histogram.snapshot()["le_10ms"] == 1


def test_latency_histogram_reset() -> None:
    histogram = LatencyHistogram(
        buckets_ms=(10, 50, 100),
    )

    histogram.observe(5)
    histogram.observe(250)

    histogram.reset()

    assert histogram.snapshot() == {
        "le_10ms": 0,
        "le_50ms": 0,
        "le_100ms": 0,
        "gt_max": 0,
    }


@pytest.mark.parametrize(
    "buckets, expected_message",
    [
        ((), "must not be empty"),
        ((0, 10), "values must be > 0"),
        ((50, 10), "sorted ascending"),
    ],
)
def test_latency_histogram_rejects_invalid_buckets(
    buckets: tuple[int, ...],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        LatencyHistogram(buckets_ms=buckets)


# =============================================================================
# LiquidationMetrics ingestion counters
# =============================================================================

def test_metrics_observe_valid_long_event_updates_legacy_and_scoped_counters() -> None:
    metrics = LiquidationMetrics()
    event = _make_event(
        side=LiquidationSide.LONG,
        quantity=Decimal("2"),
        market_type="usdm_futures",
        timeframe="realtime",
    )

    metrics.observe_event(
        event,
        is_valid=True,
        is_stale=False,
        is_large=True,
    )

    scope_key = "binance:usdm_futures:BTCUSDT:realtime"

    assert metrics.total_events_seen == 1
    assert metrics.total_valid_events == 1
    assert metrics.total_invalid_events == 0
    assert metrics.total_stale_events == 0
    assert metrics.total_large_events == 1
    assert metrics.total_long_events == 1
    assert metrics.total_short_events == 0
    assert metrics.total_long_notional_usd == Decimal("130000")
    assert metrics.total_short_notional_usd == Decimal("0")
    assert metrics.total_notional_usd == Decimal("130000")

    assert metrics.symbol_event_counts["binance:BTCUSDT"] == 1
    assert metrics.exchange_event_counts["binance"] == 1
    assert metrics.market_type_event_counts["usdm_futures"] == 1
    assert metrics.scope_event_counts[scope_key] == 1


def test_metrics_observe_valid_short_event_updates_short_notional_and_scope() -> None:
    metrics = LiquidationMetrics()
    event = _make_event(
        side=LiquidationSide.SHORT,
        quantity=Decimal("2"),
        market_type="linear",
        timeframe="realtime",
    )

    metrics.observe_event(
        event,
        is_valid=True,
        is_stale=False,
        is_large=False,
    )

    assert metrics.total_events_seen == 1
    assert metrics.total_valid_events == 1
    assert metrics.total_short_events == 1
    assert metrics.total_short_notional_usd == Decimal("130000")
    assert metrics.total_large_events == 0
    assert metrics.market_type_event_counts["linear"] == 1
    assert metrics.scope_event_counts["binance:linear:BTCUSDT:realtime"] == 1


def test_metrics_observe_invalid_event_counts_scope_but_not_notional() -> None:
    metrics = LiquidationMetrics()
    event = _make_event(
        side=LiquidationSide.LONG,
        quantity=Decimal("2"),
        market_type="usdm_futures",
    )

    metrics.observe_event(
        event,
        is_valid=False,
        is_stale=False,
        is_large=False,
    )

    assert metrics.total_events_seen == 1
    assert metrics.total_valid_events == 0
    assert metrics.total_invalid_events == 1
    assert metrics.total_long_events == 0
    assert metrics.total_long_notional_usd == Decimal("0")
    assert metrics.scope_event_counts["binance:usdm_futures:BTCUSDT:realtime"] == 1


def test_metrics_observe_stale_event_can_be_valid_and_stale() -> None:
    metrics = LiquidationMetrics()
    event = _make_event()

    metrics.observe_event(
        event,
        is_valid=True,
        is_stale=True,
        is_large=False,
    )

    assert metrics.total_events_seen == 1
    assert metrics.total_valid_events == 1
    assert metrics.total_invalid_events == 0
    assert metrics.total_stale_events == 1
    assert metrics.stale_ratio == 1.0


def test_metrics_observe_invalid_event_without_event_updates_full_scope_if_available() -> None:
    metrics = LiquidationMetrics()

    metrics.observe_invalid_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert metrics.total_events_seen == 1
    assert metrics.total_invalid_events == 1
    assert metrics.symbol_event_counts["binance:BTCUSDT"] == 1
    assert metrics.exchange_event_counts["binance"] == 1
    assert metrics.market_type_event_counts["usdm_futures"] == 1
    assert metrics.scope_event_counts["binance:usdm_futures:BTCUSDT:realtime"] == 1


def test_metrics_observe_invalid_event_without_symbol_updates_partial_dimensions() -> None:
    metrics = LiquidationMetrics()

    metrics.observe_invalid_event(
        exchange="binance",
        market_type="usdm_futures",
    )

    assert metrics.total_events_seen == 1
    assert metrics.total_invalid_events == 1
    assert metrics.exchange_event_counts["binance"] == 1
    assert metrics.market_type_event_counts["usdm_futures"] == 1
    assert metrics.symbol_event_counts == {}
    assert metrics.scope_event_counts == {}


def test_metrics_respects_disabled_dimension_counters() -> None:
    metrics = LiquidationMetrics(
        keep_symbol_level_counters=False,
        keep_exchange_level_counters=False,
        keep_market_type_level_counters=False,
        keep_scope_level_counters=False,
    )
    event = _make_event()

    metrics.observe_event(
        event,
        is_valid=True,
        is_stale=False,
        is_large=False,
    )

    assert metrics.total_events_seen == 1
    assert metrics.total_valid_events == 1
    assert metrics.symbol_event_counts == {}
    assert metrics.exchange_event_counts == {}
    assert metrics.market_type_event_counts == {}
    assert metrics.scope_event_counts == {}


def test_metrics_observe_latency_ms_updates_histogram_snapshot() -> None:
    metrics = LiquidationMetrics(
        latency_buckets_ms=(10, 50, 100),
    )

    metrics.observe_latency_ms(25)
    metrics.observe_latency_ms(250)

    snapshot = metrics.snapshot()

    assert snapshot.latency_histogram["le_50ms"] == 1
    assert snapshot.latency_histogram["gt_max"] == 1


# =============================================================================
# LiquidationMetrics detection counters
# =============================================================================

def test_metrics_observe_cascade_updates_full_scope_counters() -> None:
    metrics = LiquidationMetrics()
    result = _make_detection_result(
        market_type="usdm_futures",
        timeframe="realtime",
        severity=CascadeSeverity.HIGH,
    )

    metrics.observe_cascade(result)

    assert metrics.total_cascades_detected == 1
    assert metrics.cascade_by_symbol["binance:BTCUSDT"] == 1
    assert metrics.cascade_by_exchange["binance"] == 1
    assert metrics.cascade_by_market_type["usdm_futures"] == 1
    assert metrics.cascade_by_scope["binance:usdm_futures:BTCUSDT:realtime"] == 1
    assert metrics.severity_counts[CascadeSeverity.HIGH.value] == 1


def test_metrics_observe_exhaustion_updates_full_scope_counters_without_incrementing_cascade() -> None:
    metrics = LiquidationMetrics()
    result = _make_detection_result(
        market_type="linear",
        timeframe="realtime",
        severity=CascadeSeverity.EXTREME,
    )

    metrics.observe_exhaustion(result)

    assert metrics.total_cascades_detected == 0
    assert metrics.total_exhaustions_detected == 1
    assert metrics.exhaustion_by_symbol["binance:BTCUSDT"] == 1
    assert metrics.exhaustion_by_exchange["binance"] == 1
    assert metrics.exhaustion_by_market_type["linear"] == 1
    assert metrics.exhaustion_by_scope["binance:linear:BTCUSDT:realtime"] == 1
    assert metrics.severity_counts[CascadeSeverity.EXTREME.value] == 1


def test_metrics_observe_multiple_scopes_does_not_collapse_same_symbol() -> None:
    metrics = LiquidationMetrics()

    usdm_event = _make_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        quantity=Decimal("1"),
    )
    coinm_event = _make_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
        exchange_symbol="BTCUSD_PERP",
        quantity=Decimal("1"),
    )
    one_minute_event = _make_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="1m",
        quantity=Decimal("1"),
    )

    for event in (usdm_event, coinm_event, one_minute_event):
        metrics.observe_event(
            event,
            is_valid=True,
            is_stale=False,
            is_large=False,
        )

    assert metrics.total_events_seen == 3
    assert metrics.symbol_event_counts["binance:BTCUSDT"] == 3
    assert metrics.exchange_event_counts["binance"] == 3
    assert metrics.market_type_event_counts["usdm_futures"] == 2
    assert metrics.market_type_event_counts["coinm_futures"] == 1

    assert metrics.scope_event_counts["binance:usdm_futures:BTCUSDT:realtime"] == 1
    assert metrics.scope_event_counts["binance:coinm_futures:BTCUSDT:realtime"] == 1
    assert metrics.scope_event_counts["binance:usdm_futures:BTCUSDT:1m"] == 1


# =============================================================================
# LiquidationMetricsSnapshot / serialization / ratios / reset
# =============================================================================

def test_metrics_snapshot_returns_snapshot_model_with_full_scope_metadata() -> None:
    metrics = LiquidationMetrics()

    metrics.observe_event(
        _make_event(quantity=Decimal("2")),
        is_valid=True,
        is_stale=False,
        is_large=True,
    )
    metrics.observe_invalid_event(
        exchange="binance",
        symbol="ETHUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    snapshot = metrics.snapshot()

    assert isinstance(snapshot, LiquidationMetricsSnapshot)
    assert snapshot.total_events_seen == 2
    assert snapshot.total_valid_events == 1
    assert snapshot.total_invalid_events == 1
    assert snapshot.total_large_events == 1
    assert snapshot.total_notional_usd == Decimal("130000")
    assert snapshot.valid_ratio == 0.5
    assert snapshot.invalid_ratio == 0.5
    assert snapshot.large_ratio == 1.0
    assert snapshot.metadata["scope"] == "exchange:market_type:symbol:timeframe"
    assert snapshot.metadata["tracked_scopes"] == 2


def test_metrics_snapshot_to_dict_serializes_decimal_datetime_and_nested_counters() -> None:
    metrics = LiquidationMetrics()

    metrics.observe_event(
        _make_event(quantity=Decimal("2")),
        is_valid=True,
        is_stale=False,
        is_large=False,
    )

    data = metrics.snapshot().to_dict()

    assert isinstance(data["created_at"], str)
    assert data["total_long_notional_usd"] == "130000"
    assert data["total_notional_usd"] == "130000"
    assert data["scope_event_counts"] == {
        "binance:usdm_futures:BTCUSDT:realtime": 1,
    }
    assert data["metadata"]["scope"] == "exchange:market_type:symbol:timeframe"


def test_metrics_snapshot_to_dict_can_return_raw_decimal_values() -> None:
    metrics = LiquidationMetrics()

    metrics.observe_event(
        _make_event(quantity=Decimal("2")),
        is_valid=True,
        is_stale=False,
        is_large=False,
    )

    data = metrics.snapshot().to_dict(serialize=False)

    assert data["total_long_notional_usd"] == Decimal("130000")
    assert data["total_notional_usd"] == Decimal("130000")


def test_metrics_ratios_are_zero_when_no_events_seen() -> None:
    metrics = LiquidationMetrics()

    assert metrics.total_notional_usd == Decimal("0")
    assert metrics.valid_ratio == 0.0
    assert metrics.invalid_ratio == 0.0
    assert metrics.stale_ratio == 0.0
    assert metrics.large_ratio == 0.0
    assert metrics.long_notional_ratio == 0.0
    assert metrics.short_notional_ratio == 0.0


def test_metrics_notional_ratios_split_long_and_short_notional() -> None:
    metrics = LiquidationMetrics()

    metrics.observe_event(
        _make_event(
            side=LiquidationSide.LONG,
            quantity=Decimal("1"),
        ),
        is_valid=True,
        is_stale=False,
        is_large=False,
    )
    metrics.observe_event(
        _make_event(
            side=LiquidationSide.SHORT,
            quantity=Decimal("3"),
        ),
        is_valid=True,
        is_stale=False,
        is_large=False,
    )

    assert metrics.total_long_notional_usd == Decimal("65000")
    assert metrics.total_short_notional_usd == Decimal("195000")
    assert metrics.total_notional_usd == Decimal("260000")
    assert metrics.long_notional_ratio == 0.25
    assert metrics.short_notional_ratio == 0.75


def test_metrics_reset_clears_all_counters_and_histogram() -> None:
    metrics = LiquidationMetrics(
        latency_buckets_ms=(10, 50, 100),
    )

    metrics.observe_event(
        _make_event(quantity=Decimal("2")),
        is_valid=True,
        is_stale=False,
        is_large=True,
    )
    metrics.observe_cascade(
        _make_detection_result(severity=CascadeSeverity.HIGH),
    )
    metrics.observe_exhaustion(
        _make_detection_result(severity=CascadeSeverity.EXTREME),
    )
    metrics.observe_latency_ms(25)

    metrics.reset()

    assert metrics.total_events_seen == 0
    assert metrics.total_valid_events == 0
    assert metrics.total_invalid_events == 0
    assert metrics.total_stale_events == 0
    assert metrics.total_large_events == 0
    assert metrics.total_cascades_detected == 0
    assert metrics.total_exhaustions_detected == 0
    assert metrics.total_long_events == 0
    assert metrics.total_short_events == 0
    assert metrics.total_long_notional_usd == Decimal("0")
    assert metrics.total_short_notional_usd == Decimal("0")

    assert metrics.symbol_event_counts == {}
    assert metrics.exchange_event_counts == {}
    assert metrics.market_type_event_counts == {}
    assert metrics.scope_event_counts == {}

    assert metrics.cascade_by_symbol == {}
    assert metrics.cascade_by_exchange == {}
    assert metrics.cascade_by_market_type == {}
    assert metrics.cascade_by_scope == {}

    assert metrics.exhaustion_by_symbol == {}
    assert metrics.exhaustion_by_exchange == {}
    assert metrics.exhaustion_by_market_type == {}
    assert metrics.exhaustion_by_scope == {}

    assert metrics.severity_counts == {
        CascadeSeverity.LOW.value: 0,
        CascadeSeverity.MEDIUM.value: 0,
        CascadeSeverity.HIGH.value: 0,
        CascadeSeverity.EXTREME.value: 0,
    }

    latency_snapshot = metrics.snapshot().latency_histogram
    assert latency_snapshot == {
        "le_10ms": 0,
        "le_50ms": 0,
        "le_100ms": 0,
        "gt_max": 0,
    }


# =============================================================================
# Model scope safety checks used by metrics/state tests
# =============================================================================

def test_detection_result_rejects_cluster_with_different_scope() -> None:
    cluster = _make_cluster(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
        exchange_symbol="BTCUSD_PERP",
    )

    with pytest.raises(ValueError, match="cluster scope mismatch"):
        CascadeDetectionResult(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            exchange_symbol="BTCUSDT",
            side=LiquidationSide.LONG,
            direction=CascadeDirection.DOWN,
            detected_at=_utc_now(),
            cluster=cluster,
            intensity_score=0.85,
            confidence=0.80,
            continuation_bias=0.70,
            exhaustion_bias=0.20,
            event_count=3,
            total_notional_usd=Decimal("195000"),
            window_seconds=10,
            price_range_pct=0.30,
            severity=CascadeSeverity.HIGH,
            status=LiquidationStatus.CONFIRMED,
            source="test",
        )


def test_make_liquidation_key_normalizes_futures_scope() -> None:
    key = make_liquidation_key(
        exchange="BINANCE",
        market_type="USDM_FUTURES",
        symbol="btc-usdt",
        timeframe="Realtime",
    )

    assert key == ("binance", "usdm_futures", "BTCUSDT", "realtime")
    assert _scope_key_to_string(key) == "binance:usdm_futures:BTCUSDT:realtime"