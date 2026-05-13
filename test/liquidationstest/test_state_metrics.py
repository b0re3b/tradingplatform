# test/liquidationstest/test_state_metrics.py

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from analytics.liquidations.enums import (
    CascadeDirection,
    CascadeSeverity,
    LiquidationSide,
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
)
from analytics.liquidations.state import LiquidationState, SymbolLiquidationState


# ============================================================
# Helpers
# ============================================================

def _make_event(
    *,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    side: LiquidationSide = LiquidationSide.LONG,
    price: Decimal = Decimal("65000"),
    quantity: Decimal = Decimal("1"),
    timestamp: datetime | None = None,
    source: str = "test",
) -> LiquidationEvent:
    notional_usd = price * quantity

    return LiquidationEvent(
        exchange=exchange,
        symbol=symbol,
        side=side,
        price=price,
        quantity=quantity,
        notional_usd=notional_usd,
        timestamp=timestamp or datetime.now(timezone.utc),
        source=source,
    )


def _make_cluster(
    *,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    side: LiquidationSide = LiquidationSide.LONG,
    severity: CascadeSeverity = CascadeSeverity.HIGH,
) -> LiquidationCluster:
    now = datetime.now(timezone.utc)

    return LiquidationCluster(
        exchange=exchange,
        symbol=symbol,
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
        source="test",
    )


def _make_detection_result(
    *,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    side: LiquidationSide = LiquidationSide.LONG,
    severity: CascadeSeverity = CascadeSeverity.HIGH,
) -> CascadeDetectionResult:
    cluster = _make_cluster(
        exchange=exchange,
        symbol=symbol,
        side=side,
        severity=severity,
    )

    return CascadeDetectionResult(
        exchange=exchange,
        symbol=symbol,
        side=side,
        direction=cluster.direction,
        detected_at=datetime.now(timezone.utc),
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
        source="test",
    )


# ============================================================
# SymbolLiquidationState
# ============================================================

def test_symbol_state_add_event_updates_counters() -> None:
    state = SymbolLiquidationState(
        exchange="Binance",
        symbol="btc-usdt",
        max_events=10,
    )

    long_event = _make_event(side=LiquidationSide.LONG)
    short_event = _make_event(side=LiquidationSide.SHORT)

    state.add_event(long_event)
    state.add_event(short_event)

    assert state.exchange == "binance"
    assert state.symbol == "BTCUSDT"
    assert state.total_buffered_events == 2
    assert state.total_events_seen == 2
    assert state.long_events_count == 1
    assert state.short_events_count == 1
    assert state.last_event_at is not None
    assert state.last_long_event_at is not None
    assert state.last_short_event_at is not None


def test_symbol_state_rejects_wrong_symbol_key() -> None:
    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        max_events=10,
    )

    wrong_event = _make_event(
        exchange="bybit",
        symbol="ETHUSDT",
    )

    try:
        state.add_event(wrong_event)
    except ValueError as exc:
        assert "Event key mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched event key")


def test_symbol_state_bounded_buffer_evicts_oldest_and_updates_counters() -> None:
    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        max_events=2,
    )

    first = _make_event(side=LiquidationSide.LONG)
    second = _make_event(side=LiquidationSide.SHORT)
    third = _make_event(side=LiquidationSide.SHORT)

    state.add_event(first)
    state.add_event(second)
    state.add_event(third)

    assert state.total_buffered_events == 2
    assert state.total_events_seen == 3
    assert state.long_events_count == 0
    assert state.short_events_count == 2
    assert list(state.events) == [second, third]


def test_symbol_state_get_recent_events_filters_by_side() -> None:
    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        max_events=10,
    )

    first = _make_event(side=LiquidationSide.LONG)
    second = _make_event(side=LiquidationSide.SHORT)
    third = _make_event(side=LiquidationSide.LONG)

    state.extend_events([first, second, third])

    recent_longs = state.get_recent_events(
        side=LiquidationSide.LONG,
        limit=2,
    )

    assert recent_longs == [third, first]


def test_symbol_state_prune_before_removes_old_events() -> None:
    now = datetime.now(timezone.utc)

    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        max_events=10,
    )

    old_event = _make_event(timestamp=now - timedelta(minutes=10))
    fresh_event = _make_event(timestamp=now)

    state.extend_events([old_event, fresh_event])

    removed = state.prune_before(now - timedelta(minutes=1))

    assert removed == 1
    assert state.total_buffered_events == 1
    assert list(state.events) == [fresh_event]


def test_symbol_state_cooldown_lifecycle() -> None:
    now = datetime.now(timezone.utc)

    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        max_events=10,
    )

    state.set_cascade_detected(
        detected_at=now,
        cooldown_until=now + timedelta(seconds=10),
    )

    assert state.is_in_cooldown(now) is True

    state.clear_cooldown()

    assert state.is_in_cooldown(now) is False


def test_symbol_state_snapshot_contains_runtime_state() -> None:
    state = SymbolLiquidationState(
        exchange="binance",
        symbol="BTCUSDT",
        max_events=10,
    )

    state.add_event(_make_event())

    snapshot = state.snapshot()

    assert snapshot.exchange == "binance"
    assert snapshot.symbol == "BTCUSDT"
    assert snapshot.total_buffered_events == 1
    assert snapshot.max_events == 10
    assert snapshot.metadata["is_in_cooldown"] is False


# ============================================================
# LiquidationState
# ============================================================

def test_liquidation_state_add_event_creates_symbol_state() -> None:
    state = LiquidationState(max_events_per_symbol=10)
    event = _make_event()

    symbol_state = state.add_event(event)

    assert symbol_state is state.get("binance", "BTCUSDT")
    assert symbol_state.total_buffered_events == 1
    assert state.symbols_count == 1


def test_liquidation_state_get_or_create_is_idempotent() -> None:
    state = LiquidationState(max_events_per_symbol=10)

    first = state.get_or_create("binance", "BTCUSDT")
    second = state.get_or_create("BINANCE", "btc-usdt")

    assert first is second
    assert state.symbols_count == 1


def test_liquidation_state_prune_all_removes_old_events() -> None:
    now = datetime.now(timezone.utc)

    state = LiquidationState(max_events_per_symbol=10)

    old_event = _make_event(timestamp=now - timedelta(minutes=10))
    fresh_event = _make_event(timestamp=now)

    state.add_event(old_event)
    state.add_event(fresh_event)

    removed = state.prune_before(
        now - timedelta(minutes=1),
    )

    symbol_state = state.get("binance", "BTCUSDT")

    assert removed == 1
    assert symbol_state is not None
    assert symbol_state.total_buffered_events == 1


def test_liquidation_state_snapshots_returns_all_symbol_snapshots() -> None:
    state = LiquidationState(max_events_per_symbol=10)

    state.add_event(_make_event(symbol="BTCUSDT"))
    state.add_event(_make_event(symbol="ETHUSDT"))

    snapshots = state.snapshots()

    assert len(snapshots) == 2

    symbols = {snapshot.symbol for snapshot in snapshots}
    assert symbols == {"BTCUSDT", "ETHUSDT"}


def test_liquidation_state_clear_removes_all_symbols() -> None:
    state = LiquidationState(max_events_per_symbol=10)

    state.add_event(_make_event(symbol="BTCUSDT"))
    state.add_event(_make_event(symbol="ETHUSDT"))

    assert state.symbols_count == 2

    state.clear()

    assert state.symbols_count == 0
    assert state.snapshots() == []


# ============================================================
# LatencyHistogram
# ============================================================

def test_latency_histogram_observe_and_snapshot() -> None:
    histogram = LatencyHistogram(
        buckets_ms=(10, 50, 100),
    )

    histogram.observe(5)
    histogram.observe(50)
    histogram.observe(250)

    snapshot = histogram.snapshot()

    assert snapshot["le_10ms"] == 1
    assert snapshot["le_50ms"] == 1
    assert snapshot["le_100ms"] == 0
    assert snapshot["gt_max"] == 1


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


def test_latency_histogram_rejects_invalid_buckets() -> None:
    try:
        LatencyHistogram(buckets_ms=(50, 10))
    except ValueError as exc:
        assert "sorted ascending" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsorted buckets")


# ============================================================
# LiquidationMetrics
# ============================================================

def test_metrics_observe_valid_long_event() -> None:
    metrics = LiquidationMetrics()
    event = _make_event(
        side=LiquidationSide.LONG,
        quantity=Decimal("2"),
    )

    metrics.observe_event(
        event,
        is_valid=True,
        is_stale=False,
        is_large=True,
    )

    assert metrics.total_events_seen == 1
    assert metrics.total_valid_events == 1
    assert metrics.total_invalid_events == 0
    assert metrics.total_stale_events == 0
    assert metrics.total_large_events == 1
    assert metrics.total_long_events == 1
    assert metrics.total_short_events == 0
    assert metrics.total_long_notional_usd == Decimal("130000")
    assert metrics.total_notional_usd == Decimal("130000")
    assert metrics.symbol_event_counts["binance:BTCUSDT"] == 1
    assert metrics.exchange_event_counts["binance"] == 1


def test_metrics_observe_valid_short_event() -> None:
    metrics = LiquidationMetrics()
    event = _make_event(
        side=LiquidationSide.SHORT,
        quantity=Decimal("2"),
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


def test_metrics_observe_invalid_event() -> None:
    metrics = LiquidationMetrics()
    event = _make_event()

    metrics.observe_event(
        event,
        is_valid=False,
        is_stale=False,
        is_large=False,
    )

    assert metrics.total_events_seen == 1
    assert metrics.total_valid_events == 0
    assert metrics.total_invalid_events == 1


def test_metrics_observe_stale_event() -> None:
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


def test_metrics_observe_invalid_event_without_liquidation_event() -> None:
    metrics = LiquidationMetrics()

    metrics.observe_invalid_event(exchange="binance")

    assert metrics.total_events_seen == 1
    assert metrics.total_invalid_events == 1
    assert metrics.exchange_event_counts["binance"] == 1


def test_metrics_observe_latency_ms() -> None:
    metrics = LiquidationMetrics(
        latency_buckets_ms=(10, 50, 100),
    )

    metrics.observe_latency_ms(25)

    snapshot = metrics.snapshot()

    assert snapshot.latency_histogram["le_50ms"] == 1


def test_metrics_observe_cascade_updates_counters() -> None:
    metrics = LiquidationMetrics()
    result = _make_detection_result(
        severity=CascadeSeverity.HIGH,
    )

    metrics.observe_cascade(result)

    assert metrics.total_cascades_detected == 1
    assert metrics.cascade_by_symbol["binance:BTCUSDT"] == 1
    assert metrics.cascade_by_exchange["binance"] == 1
    assert metrics.severity_counts[CascadeSeverity.HIGH.value] == 1


def test_metrics_observe_exhaustion_updates_counters() -> None:
    metrics = LiquidationMetrics()
    result = _make_detection_result(
        severity=CascadeSeverity.EXTREME,
    )

    metrics.observe_exhaustion(result)

    assert metrics.total_exhaustions_detected == 1
    assert metrics.exhaustion_by_symbol["binance:BTCUSDT"] == 1
    assert metrics.exhaustion_by_exchange["binance"] == 1
    assert metrics.severity_counts[CascadeSeverity.EXTREME.value] == 1


def test_metrics_snapshot_returns_snapshot_model() -> None:
    metrics = LiquidationMetrics()

    metrics.observe_event(
        _make_event(),
        is_valid=True,
        is_stale=False,
        is_large=True,
    )

    snapshot = metrics.snapshot()

    assert isinstance(snapshot, LiquidationMetricsSnapshot)
    assert snapshot.total_events_seen == 1
    assert snapshot.total_valid_events == 1
    assert snapshot.total_large_events == 1
    assert snapshot.total_notional_usd == Decimal("65000")


def test_metrics_snapshot_to_dict_serializes_decimal_and_datetime() -> None:
    metrics = LiquidationMetrics()

    metrics.observe_event(
        _make_event(),
        is_valid=True,
        is_stale=False,
        is_large=False,
    )

    data = metrics.snapshot().to_dict()

    assert isinstance(data["created_at"], str)
    assert data["total_long_notional_usd"] == "65000"
    assert data["total_notional_usd"] == "65000"


def test_metrics_reset_clears_counters() -> None:
    metrics = LiquidationMetrics()

    metrics.observe_event(
        _make_event(),
        is_valid=True,
        is_stale=False,
        is_large=True,
    )
    metrics.observe_latency_ms(25)

    metrics.reset()

    assert metrics.total_events_seen == 0
    assert metrics.total_valid_events == 0
    assert metrics.total_large_events == 0
    assert metrics.total_long_notional_usd == Decimal("0")
    assert metrics.symbol_event_counts == {}
    assert metrics.exchange_event_counts == {}

    latency_snapshot = metrics.snapshot().latency_histogram
    assert all(value == 0 for value in latency_snapshot.values())