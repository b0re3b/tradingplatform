from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable

import pytest
import pytest_asyncio

from analytics.spreads import (
    FundingSnapshot,
    InstrumentType,
    QuoteSnapshot,
    SpotFuturesSpreadAnalyzer,
    SpotFuturesSpreadConfig,
    SpreadSignal,
    SpreadSnapshot,
    SpreadSignalType,
    SpreadType,
)
from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler


pytestmark = pytest.mark.asyncio


# ============================================================
# Topics / constants
# ============================================================

QUOTE_TOPIC = "market.quote.updated"
FUNDING_TOPIC = "market.funding.updated"

SNAPSHOT_TOPIC = "analytics.spreads.spot_futures.updated"
SIGNAL_TOPIC = "analytics.spreads.signal.generated"

SERVICE_NAME = "test_spot_futures_pipeline"


# ============================================================
# Real infrastructure adapters
# ============================================================


def _build_event_bus() -> EventBus:
    try:
        return EventBus()
    except TypeError:
        pass

    try:
        from core.config import EventBusConfig

        try:
            return EventBus(config=EventBusConfig())
        except TypeError:
            return EventBus(EventBusConfig())
    except Exception as exc:
        raise AssertionError(
            "Could not construct real EventBus. "
            "Update _build_event_bus() to match your core.EventBus constructor."
        ) from exc


def _build_scheduler(event_bus: EventBus) -> Scheduler:
    try:
        return Scheduler(event_bus=event_bus)
    except TypeError:
        pass

    try:
        from core.config import SchedulerConfig

        try:
            return Scheduler(config=SchedulerConfig(), event_bus=event_bus)
        except TypeError:
            try:
                return Scheduler(SchedulerConfig(), event_bus=event_bus)
            except TypeError:
                return Scheduler(SchedulerConfig())
    except Exception as exc:
        raise AssertionError(
            "Could not construct real Scheduler. "
            "Update _build_scheduler() to match your core.Scheduler constructor."
        ) from exc


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _maybe_start(component: Any) -> None:
    start = getattr(component, "start", None)
    if callable(start):
        await _maybe_await(start())


async def _maybe_stop(component: Any) -> None:
    stop = getattr(component, "stop", None)
    if callable(stop):
        await _maybe_await(stop())


async def _subscribe(
    event_bus: EventBus,
    topic_pattern: str,
    handler: Callable[[Event], Any],
    *,
    name: str,
) -> Any:
    subscription = event_bus.subscribe(
        topic_pattern,
        handler,
        name=name,
    )
    return await _maybe_await(subscription)


async def _unsubscribe(event_bus: EventBus, subscription: Any) -> None:
    unsubscribe = getattr(event_bus, "unsubscribe", None)
    if callable(unsubscribe):
        await _maybe_await(unsubscribe(subscription))


async def _emit_market_event(
    event_bus: EventBus,
    topic: str,
    payload: Any,
    *,
    source: str = "test.market",
    correlation_id: str | None = None,
) -> Any:
    """
    Emits through the real core.EventBus.

    The fallback branches are here only to keep the tests resilient to small
    signature differences in EventBus.emit()/publish().
    """
    emit = getattr(event_bus, "emit", None)
    if callable(emit):
        try:
            return await _maybe_await(
                emit(
                    topic,
                    payload,
                    priority=EventPriority.NORMAL,
                    source=source,
                    correlation_id=correlation_id,
                )
            )
        except TypeError:
            try:
                return await _maybe_await(
                    emit(
                        topic,
                        payload,
                        source=source,
                        correlation_id=correlation_id,
                    )
                )
            except TypeError:
                return await _maybe_await(emit(topic, payload))

    publish = getattr(event_bus, "publish", None)
    if callable(publish):
        event = Event(
            topic=topic,
            payload=payload,
            priority=EventPriority.NORMAL,
            source=source,
            correlation_id=correlation_id,
        )
        return await _maybe_await(publish(event))

    raise AssertionError("Real EventBus exposes neither emit() nor publish().")


async def _drain_event_bus() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0.01)


async def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 1.5,
    interval: float = 0.01,
    message: str = "Condition was not satisfied in time",
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout

    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return

        await asyncio.sleep(interval)

    assert predicate(), message


# ============================================================
# Payload helpers
# ============================================================


def _payload_value(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)

    return getattr(payload, key, default)


def _payload_decimal(payload: Any, key: str) -> Decimal | None:
    value = _payload_value(payload, key)

    if value is None:
        return None

    if isinstance(value, Decimal):
        return value

    return Decimal(str(value))


def _payload_metadata(payload: Any) -> dict[str, Any]:
    metadata = _payload_value(payload, "metadata", {})
    return dict(metadata or {})


def _snapshot_payloads(events: list[Event]) -> list[Any]:
    return [event.payload for event in events]


def _signal_payloads(events: list[Event]) -> list[Any]:
    return [event.payload for event in events]


def _signal_type_value(signal: Any) -> str | None:
    signal_type = _payload_value(signal, "signal_type")

    if signal_type is None:
        return None

    if hasattr(signal_type, "value"):
        return str(signal_type.value)

    return str(signal_type)


def _assert_decimal_equal(
    actual: Decimal | None,
    expected: Decimal,
    *,
    quant: Decimal = Decimal("0.00000001"),
) -> None:
    assert actual is not None
    assert actual.quantize(quant) == expected.quantize(quant)


# ============================================================
# Event recorder
# ============================================================


class EventRecorder:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._changed = asyncio.Event()

    async def handler(self, event: Event) -> None:
        self.events.append(event)
        self._changed.set()

    async def wait_for_count(
        self,
        expected_count: int,
        *,
        timeout: float = 1.5,
    ) -> list[Event]:
        async def _wait() -> list[Event]:
            while len(self.events) < expected_count:
                self._changed.clear()
                await self._changed.wait()
            return self.events

        return await asyncio.wait_for(_wait(), timeout=timeout)

    async def assert_no_new_events(
        self,
        previous_count: int,
        *,
        settle_time: float = 0.05,
    ) -> None:
        await asyncio.sleep(settle_time)
        assert len(self.events) == previous_count

    def payloads(self) -> list[Any]:
        return [event.payload for event in self.events]

    def clear(self) -> None:
        self.events.clear()
        self._changed.clear()


# ============================================================
# Config / model factories
# ============================================================


def _spot_config(**overrides: Any) -> SpotFuturesSpreadConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "service_name": SERVICE_NAME,
        "quote_event_topic": QUOTE_TOPIC,
        "funding_event_topic": FUNDING_TOPIC,
        "snapshot_event_topic": SNAPSHOT_TOPIC,
        "signal_event_topic": SIGNAL_TOPIC,
        "max_quote_age_ms": 60_000,
        "max_quote_skew_ms": 1_000,
        "rolling_window_size": 20,
        "ema_alpha": Decimal("0.2"),
        "min_emit_interval_ms": 0,
        "cooldown_seconds": 0,
        "cleanup_interval_seconds": 3_600.0,
        "heartbeat_interval_seconds": 3_600.0,
        "stale_state_ttl_seconds": 3_600.0,
        "max_cached_quotes": 10_000,
        "max_cached_snapshots": 10_000,
        "max_cached_windows": 10_000,
        "anomaly_zscore_threshold": Decimal("2.5"),
        "widening_bps_threshold": Decimal("10"),
        "mean_reversion_zscore_threshold": Decimal("2.0"),
        "regime_shift_zscore_threshold": Decimal("3.0"),
        "notional_for_funding_adjustment": None,
        "metadata": {"test": "spot_futures_pipeline"},
    }
    values.update(overrides)
    return SpotFuturesSpreadConfig(**values)


def _quote(
    *,
    exchange: str,
    symbol: str = "BTCUSDT",
    instrument_type: InstrumentType,
    bid: Decimal | str | None,
    ask: Decimal | str | None,
    bid_size: Decimal | str | None = "10",
    ask_size: Decimal | str | None = "10",
    timestamp: datetime | None = None,
    received_at: datetime | None = None,
    sequence_id: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> QuoteSnapshot:
    now = datetime.utcnow()

    return QuoteSnapshot(
        exchange=exchange,
        symbol=symbol,
        instrument_type=instrument_type,
        bid=Decimal(str(bid)) if bid is not None else None,
        ask=Decimal(str(ask)) if ask is not None else None,
        bid_size=Decimal(str(bid_size)) if bid_size is not None else None,
        ask_size=Decimal(str(ask_size)) if ask_size is not None else None,
        timestamp=timestamp or now,
        received_at=received_at or now,
        sequence_id=sequence_id,
        metadata=dict(metadata or {}),
    )


def _spot_quote(
    *,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    bid: Decimal | str | None = "99.9",
    ask: Decimal | str | None = "100.1",
    timestamp: datetime | None = None,
    sequence_id: int | None = 1,
) -> QuoteSnapshot:
    return _quote(
        exchange=exchange,
        symbol=symbol,
        instrument_type=InstrumentType.SPOT,
        bid=bid,
        ask=ask,
        timestamp=timestamp,
        sequence_id=sequence_id,
    )


def _futures_quote(
    *,
    exchange: str = "bybit",
    symbol: str = "BTCUSDT",
    bid: Decimal | str | None = "100.9",
    ask: Decimal | str | None = "101.1",
    timestamp: datetime | None = None,
    sequence_id: int | None = 2,
    instrument_type: InstrumentType = InstrumentType.PERPETUAL,
) -> QuoteSnapshot:
    return _quote(
        exchange=exchange,
        symbol=symbol,
        instrument_type=instrument_type,
        bid=bid,
        ask=ask,
        timestamp=timestamp,
        sequence_id=sequence_id,
    )


def _funding(
    *,
    exchange: str = "bybit",
    symbol: str = "BTCUSDT",
    funding_rate: Decimal | str = "0.01",
    timestamp: datetime | None = None,
) -> FundingSnapshot:
    return FundingSnapshot(
        exchange=exchange,
        symbol=symbol,
        funding_rate=Decimal(str(funding_rate)),
        timestamp=timestamp or datetime.utcnow(),
        interval_hours=8,
        metadata={"source": "test"},
    )


# ============================================================
# Fixtures
# ============================================================


@pytest_asyncio.fixture
async def real_runtime() -> tuple[EventBus, Scheduler]:
    event_bus = _build_event_bus()
    scheduler = _build_scheduler(event_bus)

    await _maybe_start(event_bus)
    await _maybe_start(scheduler)

    try:
        yield event_bus, scheduler
    finally:
        await _maybe_stop(scheduler)
        await _maybe_stop(event_bus)


@pytest_asyncio.fixture
async def event_bus(real_runtime: tuple[EventBus, Scheduler]) -> EventBus:
    return real_runtime[0]


@pytest_asyncio.fixture
async def scheduler(real_runtime: tuple[EventBus, Scheduler]) -> Scheduler:
    return real_runtime[1]


@pytest_asyncio.fixture
async def analyzer(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> SpotFuturesSpreadAnalyzer:
    instance = SpotFuturesSpreadAnalyzer(
        config=_spot_config(),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    try:
        yield instance
    finally:
        if instance.is_running:
            await instance.stop()

        if instance.is_registered:
            instance.unregister()


# ============================================================
# Basic lifecycle / subscription pipeline
# ============================================================


async def test_start_registers_real_eventbus_subscriptions_and_scheduler_jobs(
    analyzer: SpotFuturesSpreadAnalyzer,
) -> None:
    assert analyzer.is_running is False
    assert analyzer.is_registered is False
    assert analyzer._subscriptions == []

    await analyzer.start()

    assert analyzer.is_running is True
    assert analyzer.is_registered is True

    assert len(analyzer._subscriptions) == 2
    assert len(analyzer._scheduler_job_ids) == 2

    stats = analyzer.get_stats()

    assert stats["running"] is True
    assert stats["registered"] is True
    assert stats["enabled"] is True
    assert stats["spot_quotes_cached"] == 0
    assert stats["futures_quotes_cached"] == 0
    assert stats["funding_cached"] == 0
    assert stats["latest_snapshots"] == 0


async def test_quote_event_is_ignored_when_analyzer_not_running(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    analyzer.register()

    quote = _spot_quote()

    await _emit_market_event(event_bus, QUOTE_TOPIC, quote)
    await _drain_event_bus()

    stats = analyzer.get_stats()

    assert stats["quote_events_received"] == 0
    assert stats["quotes_received"] == 0
    assert stats["quotes_stored"] == 0
    assert stats["spot_quotes_cached"] == 0


# ============================================================
# Quote processing / validation
# ============================================================


async def test_rejects_invalid_quote_payload_and_updates_stats(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    await _emit_market_event(event_bus, QUOTE_TOPIC, {"bad": "payload"})

    await _wait_until(
        lambda: analyzer.get_stats()["invalid_payloads"] == 1,
        message="Invalid quote payload was not counted",
    )

    stats = analyzer.get_stats()

    assert stats["quote_events_received"] == 1
    assert stats["invalid_payloads"] == 1
    assert stats["quotes_received"] == 0
    assert stats["quotes_stored"] == 0


async def test_rejects_stale_quote_and_does_not_cache_it(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    stale_timestamp = datetime.utcnow() - timedelta(minutes=10)

    stale_quote = _spot_quote(
        exchange="binance",
        timestamp=stale_timestamp,
    )

    await _emit_market_event(event_bus, QUOTE_TOPIC, stale_quote)

    await _wait_until(
        lambda: analyzer.get_stats()["stale_quotes"] == 1,
        message="Stale quote was not counted",
    )

    stats = analyzer.get_stats()

    assert stats["quote_events_received"] == 1
    assert stats["quotes_received"] == 1
    assert stats["stale_quotes"] == 1
    assert stats["quotes_stored"] == 0
    assert stats["spot_quotes_cached"] == 0


async def test_rejects_incomplete_quote_and_does_not_cache_it(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    incomplete_quote = _spot_quote(
        exchange="binance",
        bid=None,
        ask="100.1",
    )

    await _emit_market_event(event_bus, QUOTE_TOPIC, incomplete_quote)

    await _wait_until(
        lambda: analyzer.get_stats()["invalid_quotes"] + analyzer.get_stats()["incomplete_quotes"] >= 1,
        message="Incomplete quote was not rejected",
    )

    stats = analyzer.get_stats()

    assert stats["quote_events_received"] == 1
    assert stats["quotes_received"] == 1
    assert stats["quotes_stored"] == 0
    assert stats["spot_quotes_cached"] == 0


async def test_stores_spot_quote_without_snapshot_until_futures_quote_arrives(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )

    try:
        await analyzer.start()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _spot_quote(exchange="binance"),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["spot_quotes_cached"] == 1,
            message="Spot quote was not cached",
        )

        stats = analyzer.get_stats()

        assert stats["quotes_stored"] == 1
        assert stats["futures_quotes_cached"] == 0
        assert stats["snapshots_built"] == 0

        await snapshot_recorder.assert_no_new_events(0)
    finally:
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_stores_futures_quote_without_snapshot_until_spot_quote_arrives(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )

    try:
        await analyzer.start()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _futures_quote(exchange="bybit"),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["futures_quotes_cached"] == 1,
            message="Futures quote was not cached",
        )

        stats = analyzer.get_stats()

        assert stats["quotes_stored"] == 1
        assert stats["spot_quotes_cached"] == 0
        assert stats["snapshots_built"] == 0

        await snapshot_recorder.assert_no_new_events(0)
    finally:
        await _unsubscribe(event_bus, snapshot_subscription)


# ============================================================
# Snapshot building
# ============================================================


async def test_real_eventbus_quote_flow_builds_spot_futures_snapshot(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _spot_quote(
                exchange="Binance",
                symbol="BTC/USDT",
                bid="99.9",
                ask="100.1",
                timestamp=now,
                sequence_id=101,
            ),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _futures_quote(
                exchange="Bybit",
                symbol="BTC-USDT",
                bid="100.9",
                ask="101.1",
                timestamp=now + timedelta(milliseconds=20),
                sequence_id=202,
            ),
        )

        await snapshot_recorder.wait_for_count(1)

        payload = snapshot_recorder.payloads()[0]

        assert isinstance(payload, SpreadSnapshot)

        assert payload.spread_type == SpreadType.SPOT_FUTURES
        assert payload.symbol == "BTCUSDT"

        assert payload.leg_a_exchange == "binance"
        assert payload.leg_b_exchange == "bybit"

        assert payload.leg_a_type == InstrumentType.SPOT
        assert payload.leg_b_type == InstrumentType.PERPETUAL

        assert payload.leg_a_mid == Decimal("100.0")
        assert payload.leg_b_mid == Decimal("101.0")

        _assert_decimal_equal(payload.raw_spread, Decimal("1.0"))
        _assert_decimal_equal(payload.basis, Decimal("1.0"))
        _assert_decimal_equal(payload.spread_pct, Decimal("1.0"))
        _assert_decimal_equal(payload.spread_bps, Decimal("100.0"))
        _assert_decimal_equal(payload.funding_adjusted_spread, Decimal("1.0"))
        _assert_decimal_equal(payload.net_spread, Decimal("1.0"))

        assert payload.stats is not None
        assert payload.stats.count == 1
        assert payload.stats.last_value == Decimal("1.0")

        metadata = payload.metadata

        assert metadata["spot_exchange"] == "binance"
        assert metadata["futures_exchange"] == "bybit"
        assert metadata["spot_sequence_id"] == 101
        assert metadata["futures_sequence_id"] == 202

        latest = analyzer.get_latest_snapshot(
            "BTC_USDT",
            "BINANCE",
            "BYBIT",
        )

        assert latest is payload

        stats = analyzer.get_stats()

        assert stats["quotes_received"] == 2
        assert stats["quotes_stored"] == 2
        assert stats["spot_quotes_cached"] == 1
        assert stats["futures_quotes_cached"] == 1
        assert stats["snapshots_built"] == 1
        assert stats["latest_snapshots"] == 1
        assert stats["active_windows"] == 1
    finally:
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_unaligned_quotes_are_skipped_and_no_snapshot_is_published(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = SpotFuturesSpreadAnalyzer(
        config=_spot_config(max_quote_skew_ms=50),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _spot_quote(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _futures_quote(timestamp=now + timedelta(seconds=2)),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["unaligned_quotes"] == 1,
            message="Unaligned quote pair was not counted",
        )

        stats = analyzer.get_stats()

        assert stats["snapshots_built"] == 0
        assert stats["latest_snapshots"] == 0

        await snapshot_recorder.assert_no_new_events(0)
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)


# ============================================================
# Funding pipeline
# ============================================================


async def test_funding_update_is_stored_without_snapshot_until_quotes_exist(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )

    try:
        await analyzer.start()

        await _emit_market_event(
            event_bus,
            FUNDING_TOPIC,
            _funding(exchange="bybit", symbol="BTCUSDT", funding_rate="0.05"),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["funding_cached"] == 1,
            message="Funding was not cached",
        )

        stats = analyzer.get_stats()

        assert stats["funding_events_received"] == 1
        assert stats["funding_updates"] == 1
        assert stats["funding_stored"] == 1
        assert stats["snapshots_built"] == 0

        await snapshot_recorder.assert_no_new_events(0)
    finally:
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_real_eventbus_funding_flow_affects_funding_adjusted_spread(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = SpotFuturesSpreadAnalyzer(
        config=_spot_config(
            notional_for_funding_adjustment=None,
            widening_bps_threshold=Decimal("1000"),
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            FUNDING_TOPIC,
            _funding(
                exchange="bybit",
                symbol="BTCUSDT",
                funding_rate="0.25",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _spot_quote(
                exchange="binance",
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _futures_quote(
                exchange="bybit",
                bid="100.9",
                ask="101.1",
                timestamp=now,
            ),
        )

        await snapshot_recorder.wait_for_count(1)

        payload = snapshot_recorder.payloads()[0]

        assert isinstance(payload, SpreadSnapshot)

        _assert_decimal_equal(payload.raw_spread, Decimal("1.0"))
        _assert_decimal_equal(payload.funding_adjusted_spread, Decimal("0.75"))
        _assert_decimal_equal(payload.net_spread, Decimal("0.75"))

        assert payload.metadata["funding_rate"] == "0.25"
        assert payload.metadata["funding_timestamp"] is not None

        stats = analyzer.get_stats()

        assert stats["funding_updates"] == 1
        assert stats["funding_stored"] == 1
        assert stats["funding_cached"] == 1
        assert stats["snapshots_built"] == 1
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_funding_update_after_quotes_recalculates_and_publishes_new_snapshot(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = SpotFuturesSpreadAnalyzer(
        config=_spot_config(
            notional_for_funding_adjustment=None,
            widening_bps_threshold=Decimal("1000"),
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _spot_quote(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _futures_quote(timestamp=now + timedelta(milliseconds=5)),
        )

        await snapshot_recorder.wait_for_count(1)

        first_snapshot = snapshot_recorder.payloads()[0]

        assert isinstance(first_snapshot, SpreadSnapshot)
        _assert_decimal_equal(first_snapshot.funding_adjusted_spread, Decimal("1.0"))

        await _emit_market_event(
            event_bus,
            FUNDING_TOPIC,
            _funding(
                exchange="bybit",
                symbol="BTCUSDT",
                funding_rate="0.40",
                timestamp=now + timedelta(milliseconds=10),
            ),
        )

        await snapshot_recorder.wait_for_count(2)

        second_snapshot = snapshot_recorder.payloads()[1]

        assert isinstance(second_snapshot, SpreadSnapshot)
        _assert_decimal_equal(second_snapshot.funding_adjusted_spread, Decimal("0.60"))

        stats = analyzer.get_stats()

        assert stats["snapshots_built"] == 2
        assert stats["funding_updates"] == 1
        assert stats["funding_stored"] == 1
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)


# ============================================================
# Signal generation
# ============================================================


async def test_widening_signal_is_generated_when_threshold_is_crossed(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = SpotFuturesSpreadAnalyzer(
        config=_spot_config(
            widening_bps_threshold=Decimal("10"),
            anomaly_zscore_threshold=Decimal("100"),
            mean_reversion_zscore_threshold=Decimal("100"),
            regime_shift_zscore_threshold=Decimal("100"),
            cooldown_seconds=0,
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    signal_recorder = EventRecorder()

    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )
    signal_subscription = await _subscribe(
        event_bus,
        SIGNAL_TOPIC,
        signal_recorder.handler,
        name="test.signal.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _spot_quote(
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _futures_quote(
                bid="100.9",
                ask="101.1",
                timestamp=now,
            ),
        )

        await snapshot_recorder.wait_for_count(1)
        await signal_recorder.wait_for_count(1)

        signal = signal_recorder.payloads()[0]

        assert isinstance(signal, SpreadSignal)
        assert signal.spread_type == SpreadType.SPOT_FUTURES
        assert signal.signal_type == SpreadSignalType.WIDENING
        assert signal.symbol == "BTCUSDT"
        assert signal.exchange_a == "binance"
        assert signal.exchange_b == "bybit"
        assert signal.value == Decimal("100.00") or signal.value == Decimal("100.0")

        assert signal.threshold == Decimal("10")
        assert signal.metadata["reason"] == "signal_built"

        stats = analyzer.get_stats()

        assert stats["signals_built"] >= 1
        assert stats["signals_published"] >= 1
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)
        await _unsubscribe(event_bus, signal_subscription)


async def test_no_signal_is_generated_below_widening_threshold(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = SpotFuturesSpreadAnalyzer(
        config=_spot_config(
            widening_bps_threshold=Decimal("1000"),
            anomaly_zscore_threshold=Decimal("100"),
            mean_reversion_zscore_threshold=Decimal("100"),
            regime_shift_zscore_threshold=Decimal("100"),
            cooldown_seconds=0,
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    signal_recorder = EventRecorder()

    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )
    signal_subscription = await _subscribe(
        event_bus,
        SIGNAL_TOPIC,
        signal_recorder.handler,
        name="test.signal.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _spot_quote(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _futures_quote(timestamp=now),
        )

        await snapshot_recorder.wait_for_count(1)

        await signal_recorder.assert_no_new_events(0)

        stats = analyzer.get_stats()

        assert stats["snapshots_built"] == 1
        assert stats["signals_published"] == 0
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)
        await _unsubscribe(event_bus, signal_subscription)


# ============================================================
# Filters
# ============================================================


async def test_disallowed_spot_exchange_is_stored_neither_as_cache_nor_snapshot(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = SpotFuturesSpreadAnalyzer(
        config=_spot_config(
            allowed_spot_exchanges={"okx"},
            allowed_futures_exchanges={"bybit"},
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )

    try:
        await analyzer.start()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _spot_quote(exchange="binance"),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _futures_quote(exchange="bybit"),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["futures_quotes_cached"] == 1,
            message="Allowed futures quote was not cached",
        )

        stats = analyzer.get_stats()

        assert stats["spot_quotes_cached"] == 0
        assert stats["futures_quotes_cached"] == 1
        assert stats["snapshots_built"] == 0

        await snapshot_recorder.assert_no_new_events(0)
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_disallowed_futures_exchange_is_stored_neither_as_cache_nor_snapshot(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = SpotFuturesSpreadAnalyzer(
        config=_spot_config(
            allowed_spot_exchanges={"binance"},
            allowed_futures_exchanges={"okx"},
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )

    try:
        await analyzer.start()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _spot_quote(exchange="binance"),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _futures_quote(exchange="bybit"),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["spot_quotes_cached"] == 1,
            message="Allowed spot quote was not cached",
        )

        stats = analyzer.get_stats()

        assert stats["spot_quotes_cached"] == 1
        assert stats["futures_quotes_cached"] == 0
        assert stats["snapshots_built"] == 0

        await snapshot_recorder.assert_no_new_events(0)
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)


# ============================================================
# Rolling stats
# ============================================================


async def test_rolling_window_updates_after_multiple_snapshots(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = SpotFuturesSpreadAnalyzer(
        config=_spot_config(
            widening_bps_threshold=Decimal("1000"),
            rolling_window_size=5,
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _spot_quote(timestamp=now, sequence_id=1),
        )

        futures_values = [
            ("100.9", "101.1"),
            ("101.9", "102.1"),
            ("102.9", "103.1"),
        ]

        for index, (bid, ask) in enumerate(futures_values, start=2):
            await _emit_market_event(
                event_bus,
                QUOTE_TOPIC,
                _futures_quote(
                    bid=bid,
                    ask=ask,
                    timestamp=now + timedelta(milliseconds=index * 10),
                    sequence_id=index,
                ),
            )

        await snapshot_recorder.wait_for_count(3)

        last_snapshot = snapshot_recorder.payloads()[-1]

        assert isinstance(last_snapshot, SpreadSnapshot)
        assert last_snapshot.stats is not None
        assert last_snapshot.stats.count == 3
        assert last_snapshot.stats.last_value == Decimal("3.0")
        assert last_snapshot.stats.mean is not None
        assert last_snapshot.stats.min_value == Decimal("1.0")
        assert last_snapshot.stats.max_value == Decimal("3.0")

        stats = analyzer.get_stats()

        assert stats["snapshots_built"] == 3
        assert stats["active_windows"] == 1
        assert stats["latest_snapshots"] == 1
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)


# ============================================================
# Cleanup
# ============================================================


async def test_cleanup_stale_state_removes_quotes_funding_snapshots_and_windows(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = SpotFuturesSpreadAnalyzer(
        config=_spot_config(
            stale_state_ttl_seconds=0.01,
            cleanup_interval_seconds=3_600.0,
            widening_bps_threshold=Decimal("1000"),
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            FUNDING_TOPIC,
            _funding(
                exchange="bybit",
                symbol="BTCUSDT",
                funding_rate="0.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _spot_quote(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _futures_quote(timestamp=now),
        )

        await snapshot_recorder.wait_for_count(1)

        assert analyzer.get_stats()["spot_quotes_cached"] == 1
        assert analyzer.get_stats()["futures_quotes_cached"] == 1
        assert analyzer.get_stats()["funding_cached"] == 1
        assert analyzer.get_stats()["latest_snapshots"] == 1
        assert analyzer.get_stats()["active_windows"] == 1

        await asyncio.sleep(0.03)

        await analyzer.cleanup_stale_state()

        stats = analyzer.get_stats()

        assert stats["cleanup_runs"] == 1
        assert stats["spot_quotes_cached"] == 0
        assert stats["futures_quotes_cached"] == 0
        assert stats["funding_cached"] == 0
        assert stats["latest_snapshots"] == 0
        assert stats["active_windows"] == 0

        assert stats["cleanup_removed_quotes"] >= 2
        assert stats["cleanup_removed_funding"] >= 1
        assert stats["cleanup_removed_snapshots"] >= 1
        assert stats["cleanup_removed_windows"] >= 1
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)


# ============================================================
# Stats contract
# ============================================================


async def test_stats_reflect_complete_spot_futures_pipeline(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = SpotFuturesSpreadAnalyzer(
        config=_spot_config(
            widening_bps_threshold=Decimal("10"),
            anomaly_zscore_threshold=Decimal("100"),
            mean_reversion_zscore_threshold=Decimal("100"),
            regime_shift_zscore_threshold=Decimal("100"),
            cooldown_seconds=0,
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    signal_recorder = EventRecorder()

    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.recorder",
    )
    signal_subscription = await _subscribe(
        event_bus,
        SIGNAL_TOPIC,
        signal_recorder.handler,
        name="test.signal.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            FUNDING_TOPIC,
            _funding(
                exchange="bybit",
                symbol="BTCUSDT",
                funding_rate="0.20",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _spot_quote(
                exchange="binance",
                symbol="BTCUSDT",
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _futures_quote(
                exchange="bybit",
                symbol="BTCUSDT",
                bid="100.9",
                ask="101.1",
                timestamp=now,
            ),
        )

        await snapshot_recorder.wait_for_count(1)
        await signal_recorder.wait_for_count(1)

        stats = analyzer.get_stats()

        assert stats["running"] is True
        assert stats["registered"] is True
        assert stats["enabled"] is True

        assert stats["quote_events_received"] == 2
        assert stats["funding_events_received"] == 1

        assert stats["quotes_received"] == 2
        assert stats["funding_updates"] == 1

        assert stats["invalid_payloads"] == 0
        assert stats["invalid_quotes"] == 0
        assert stats["stale_quotes"] == 0
        assert stats["unaligned_quotes"] == 0

        assert stats["quotes_stored"] == 2
        assert stats["funding_stored"] == 1

        assert stats["spot_quotes_cached"] == 1
        assert stats["futures_quotes_cached"] == 1
        assert stats["funding_cached"] == 1

        assert stats["snapshots_built"] == 1
        assert stats["snapshots_published"] == 1
        assert stats["latest_snapshots"] == 1
        assert stats["active_windows"] == 1

        assert stats["signals_built"] >= 1
        assert stats["signals_published"] >= 1

        snapshot = snapshot_recorder.payloads()[0]
        signal = signal_recorder.payloads()[0]

        assert isinstance(snapshot, SpreadSnapshot)
        assert isinstance(signal, SpreadSignal)

        assert snapshot.symbol == "BTCUSDT"
        assert signal.symbol == "BTCUSDT"
        assert signal.signal_type == SpreadSignalType.WIDENING
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)
        await _unsubscribe(event_bus, signal_subscription)