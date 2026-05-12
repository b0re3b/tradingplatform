from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable

import pytest
import pytest_asyncio

from analytics.spreads import (
    ArbitrageOpportunity,
    CrossExchangeSpreadAnalyzer,
    CrossExchangeSpreadConfig,
    InstrumentType,
    OpportunityDetectionReason,
    OpportunityStatus,
    QuoteSnapshot,
    SpreadOpportunityDetector,
    SpreadSignal,
    SpreadSignalType,
    SpreadSnapshot,
    SpreadType,
)
from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler


pytestmark = pytest.mark.asyncio


# ============================================================
# Topics / constants
# ============================================================

QUOTE_TOPIC = "market.quote.updated"

SNAPSHOT_TOPIC = "analytics.spreads.cross_exchange.updated"
SIGNAL_TOPIC = "analytics.spreads.signal.generated"
OPPORTUNITY_TOPIC = "analytics.spreads.arbitrage.opportunity"

SERVICE_NAME = "test_cross_exchange_pipeline"


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
# Assertions / helpers
# ============================================================


def _assert_decimal_equal(
    actual: Decimal | None,
    expected: Decimal,
    *,
    quant: Decimal = Decimal("0.00000001"),
) -> None:
    assert actual is not None
    assert actual.quantize(quant) == expected.quantize(quant)


def _assert_profitable_opportunity(opportunity: ArbitrageOpportunity) -> None:
    assert opportunity.status == OpportunityStatus.ACTIVE
    assert opportunity.is_profitable is True
    assert opportunity.gross_edge > Decimal("0")
    assert opportunity.net_edge > Decimal("0")
    assert opportunity.expires_at is not None
    assert opportunity.expires_at > opportunity.timestamp


def _signal_types(payloads: list[Any]) -> set[SpreadSignalType]:
    return {
        payload.signal_type
        for payload in payloads
        if isinstance(payload, SpreadSignal)
    }


# ============================================================
# Config / model factories
# ============================================================


def _cross_config(**overrides: Any) -> CrossExchangeSpreadConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "service_name": SERVICE_NAME,
        "quote_event_topic": QUOTE_TOPIC,
        "snapshot_event_topic": SNAPSHOT_TOPIC,
        "signal_event_topic": SIGNAL_TOPIC,
        "opportunity_event_topic": OPPORTUNITY_TOPIC,
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
        "anomaly_zscore_threshold": Decimal("100"),
        "widening_bps_threshold": Decimal("10000"),
        "arbitrage_min_bps": Decimal("1"),
        "default_trade_size": Decimal("1"),
        "min_trade_size": None,
        "max_trade_size": None,
        "slippage_max_bps": Decimal("0"),
        "safety_buffer_bps": Decimal("0"),
        "default_taker_fee_rate": Decimal("0"),
        "default_maker_fee_rate": Decimal("0"),
        "opportunity_ttl_seconds": 60.0,
        "max_cached_opportunities": 10_000,
        "allowed_instrument_types": {
            InstrumentType.SPOT,
            InstrumentType.PERPETUAL,
            InstrumentType.FUTURES,
        },
        "preferred_exchanges": set(),
        "metadata": {"test": "cross_exchange_pipeline"},
    }
    values.update(overrides)
    return CrossExchangeSpreadConfig(**values)


def _quote(
    *,
    exchange: str,
    symbol: str = "BTCUSDT",
    instrument_type: InstrumentType = InstrumentType.PERPETUAL,
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


def _binance_quote(
    *,
    symbol: str = "BTCUSDT",
    instrument_type: InstrumentType = InstrumentType.PERPETUAL,
    bid: Decimal | str | None = "99.9",
    ask: Decimal | str | None = "100.1",
    timestamp: datetime | None = None,
    sequence_id: int | None = 1,
) -> QuoteSnapshot:
    return _quote(
        exchange="binance",
        symbol=symbol,
        instrument_type=instrument_type,
        bid=bid,
        ask=ask,
        timestamp=timestamp,
        sequence_id=sequence_id,
    )


def _bybit_quote(
    *,
    symbol: str = "BTCUSDT",
    instrument_type: InstrumentType = InstrumentType.PERPETUAL,
    bid: Decimal | str | None = "104.9",
    ask: Decimal | str | None = "105.1",
    timestamp: datetime | None = None,
    sequence_id: int | None = 2,
) -> QuoteSnapshot:
    return _quote(
        exchange="bybit",
        symbol=symbol,
        instrument_type=instrument_type,
        bid=bid,
        ask=ask,
        timestamp=timestamp,
        sequence_id=sequence_id,
    )


def _okx_quote(
    *,
    symbol: str = "BTCUSDT",
    instrument_type: InstrumentType = InstrumentType.PERPETUAL,
    bid: Decimal | str | None = "109.9",
    ask: Decimal | str | None = "110.1",
    timestamp: datetime | None = None,
    sequence_id: int | None = 3,
) -> QuoteSnapshot:
    return _quote(
        exchange="okx",
        symbol=symbol,
        instrument_type=instrument_type,
        bid=bid,
        ask=ask,
        timestamp=timestamp,
        sequence_id=sequence_id,
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
) -> CrossExchangeSpreadAnalyzer:
    instance = CrossExchangeSpreadAnalyzer(
        config=_cross_config(),
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
# Lifecycle / base pipeline
# ============================================================


async def test_start_registers_real_eventbus_subscription_and_scheduler_jobs(
    analyzer: CrossExchangeSpreadAnalyzer,
) -> None:
    assert analyzer.is_running is False
    assert analyzer.is_registered is False
    assert analyzer._subscriptions == []

    await analyzer.start()

    assert analyzer.is_running is True
    assert analyzer.is_registered is True

    assert len(analyzer._subscriptions) == 1
    assert len(analyzer._scheduler_job_ids) == 2

    stats = analyzer.get_stats()

    assert stats["running"] is True
    assert stats["registered"] is True
    assert stats["enabled"] is True
    assert stats["quotes_cached"] == 0
    assert stats["active_windows"] == 0
    assert stats["latest_snapshots"] == 0
    assert stats["latest_opportunities"] == 0


async def test_quote_event_is_ignored_when_analyzer_not_running(
    analyzer: CrossExchangeSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    analyzer.register()

    await _emit_market_event(
        event_bus,
        QUOTE_TOPIC,
        _binance_quote(),
    )
    await _drain_event_bus()

    stats = analyzer.get_stats()

    assert stats["quote_events_received"] == 0
    assert stats["quotes_received"] == 0
    assert stats["quotes_stored"] == 0
    assert stats["quotes_cached"] == 0


# ============================================================
# Quote validation / rejection
# ============================================================


async def test_rejects_invalid_payload_and_updates_stats(
    analyzer: CrossExchangeSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    await _emit_market_event(
        event_bus,
        QUOTE_TOPIC,
        {"bad": "payload"},
    )

    await _wait_until(
        lambda: analyzer.get_stats()["invalid_payloads"] == 1,
        message="Invalid quote payload was not counted",
    )

    stats = analyzer.get_stats()

    assert stats["quote_events_received"] == 1
    assert stats["invalid_payloads"] == 1
    assert stats["quotes_received"] == 0
    assert stats["quotes_stored"] == 0
    assert stats["quotes_cached"] == 0


async def test_rejects_stale_quote_and_does_not_cache_it(
    analyzer: CrossExchangeSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    stale_timestamp = datetime.utcnow() - timedelta(minutes=10)

    await _emit_market_event(
        event_bus,
        QUOTE_TOPIC,
        _binance_quote(timestamp=stale_timestamp),
    )

    await _wait_until(
        lambda: analyzer.get_stats()["stale_quotes"] == 1,
        message="Stale quote was not counted",
    )

    stats = analyzer.get_stats()

    assert stats["quote_events_received"] == 1
    assert stats["quotes_received"] == 1
    assert stats["stale_quotes"] == 1
    assert stats["quotes_stored"] == 0
    assert stats["quotes_cached"] == 0


async def test_rejects_incomplete_quote_and_does_not_cache_it(
    analyzer: CrossExchangeSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    await _emit_market_event(
        event_bus,
        QUOTE_TOPIC,
        _binance_quote(bid=None, ask="100.1"),
    )

    await _wait_until(
        lambda: analyzer.get_stats()["incomplete_quotes"] + analyzer.get_stats()["invalid_quotes"] >= 1,
        message="Incomplete quote was not rejected",
    )

    stats = analyzer.get_stats()

    assert stats["quote_events_received"] == 1
    assert stats["quotes_received"] == 1
    assert stats["quotes_stored"] == 0
    assert stats["quotes_cached"] == 0


async def test_skips_disallowed_instrument_type_before_caching(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            allowed_instrument_types={InstrumentType.PERPETUAL},
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    try:
        await analyzer.start()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _binance_quote(instrument_type=InstrumentType.SPOT),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["instrument_type_skips"] == 1,
            message="Disallowed instrument type was not skipped",
        )

        stats = analyzer.get_stats()

        assert stats["quote_events_received"] == 1
        assert stats["quotes_received"] == 1
        assert stats["instrument_type_skips"] == 1
        assert stats["quotes_stored"] == 0
        assert stats["quotes_cached"] == 0
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()


# ============================================================
# Snapshot pipeline
# ============================================================


async def test_stores_first_exchange_quote_without_snapshot_until_second_exchange_arrives(
    analyzer: CrossExchangeSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.cross.snapshot.recorder",
    )

    try:
        await analyzer.start()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _binance_quote(),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["quotes_cached"] == 1,
            message="First exchange quote was not cached",
        )

        stats = analyzer.get_stats()

        assert stats["quotes_stored"] == 1
        assert stats["snapshots_built"] == 0
        assert stats["latest_snapshots"] == 0
        assert stats["latest_opportunities"] == 0

        await snapshot_recorder.assert_no_new_events(0)
    finally:
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_real_eventbus_cross_exchange_quotes_build_snapshot(
    analyzer: CrossExchangeSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.cross.snapshot.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _binance_quote(
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
            _bybit_quote(
                symbol="BTC-USDT",
                bid="104.9",
                ask="105.1",
                timestamp=now + timedelta(milliseconds=20),
                sequence_id=202,
            ),
        )

        await snapshot_recorder.wait_for_count(1)

        snapshot = snapshot_recorder.payloads()[0]

        assert isinstance(snapshot, SpreadSnapshot)

        assert snapshot.spread_type == SpreadType.CROSS_EXCHANGE
        assert snapshot.symbol == "BTCUSDT"

        assert snapshot.leg_a_exchange == "binance"
        assert snapshot.leg_b_exchange == "bybit"

        assert snapshot.leg_a_type == InstrumentType.PERPETUAL
        assert snapshot.leg_b_type == InstrumentType.PERPETUAL

        assert snapshot.leg_a_mid == Decimal("100.0")
        assert snapshot.leg_b_mid == Decimal("105.0")

        _assert_decimal_equal(snapshot.raw_spread, Decimal("5.0"))
        _assert_decimal_equal(snapshot.spread_pct, Decimal("5.0"))
        _assert_decimal_equal(snapshot.spread_bps, Decimal("500.0"))

        assert snapshot.stats is not None
        assert snapshot.stats.count == 1
        assert snapshot.stats.last_value == Decimal("5.0")

        assert snapshot.metadata["instrument_type"] == InstrumentType.PERPETUAL.value
        assert snapshot.metadata["quote_a_sequence_id"] == 101
        assert snapshot.metadata["quote_b_sequence_id"] == 202
        assert snapshot.metadata["buy_exchange"] == "binance"
        assert snapshot.metadata["sell_exchange"] == "bybit"
        assert snapshot.metadata["buy_price"] == "100.1"
        assert snapshot.metadata["sell_price"] == "104.9"
        assert snapshot.metadata["gross_edge"] == "4.8"

        latest = analyzer.get_latest_snapshot(
            "BTC_USDT",
            "BYBIT",
            "BINANCE",
            InstrumentType.PERPETUAL,
        )

        assert latest is snapshot

        stats = analyzer.get_stats()

        assert stats["quote_events_received"] == 2
        assert stats["quotes_received"] == 2
        assert stats["quotes_stored"] == 2
        assert stats["quotes_cached"] == 2
        assert stats["snapshots_built"] == 1
        assert stats["snapshots_published"] == 1
        assert stats["latest_snapshots"] == 1
        assert stats["active_windows"] == 1
    finally:
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_snapshot_key_is_exchange_order_independent(
    analyzer: CrossExchangeSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.cross.snapshot.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _bybit_quote(timestamp=now, sequence_id=1),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _binance_quote(timestamp=now, sequence_id=2),
        )

        await snapshot_recorder.wait_for_count(1)

        snapshot = snapshot_recorder.payloads()[0]

        assert isinstance(snapshot, SpreadSnapshot)

        latest_a = analyzer.get_latest_snapshot(
            "BTCUSDT",
            "binance",
            "bybit",
            InstrumentType.PERPETUAL,
        )
        latest_b = analyzer.get_latest_snapshot(
            "BTC/USDT",
            "BYBIT",
            "BINANCE",
            InstrumentType.PERPETUAL,
        )

        assert latest_a is snapshot
        assert latest_b is snapshot
    finally:
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_unaligned_quotes_are_skipped_and_no_snapshot_is_published(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(max_quote_skew_ms=50),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.cross.snapshot.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _binance_quote(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _bybit_quote(timestamp=now + timedelta(seconds=2)),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["unaligned_quotes"] == 1,
            message="Unaligned quote pair was not counted",
        )

        stats = analyzer.get_stats()

        assert stats["quotes_cached"] == 2
        assert stats["snapshots_built"] == 0
        assert stats["latest_snapshots"] == 0

        await snapshot_recorder.assert_no_new_events(0)
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_preferred_exchange_filter_skips_disallowed_pair(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            preferred_exchanges={"binance", "okx"},
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.cross.snapshot.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _binance_quote(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _bybit_quote(timestamp=now),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["preferred_exchange_skips"] == 1,
            message="Preferred exchange skip was not counted",
        )

        stats = analyzer.get_stats()

        assert stats["quotes_cached"] == 2
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
# Opportunity detector direct checks
# ============================================================


async def test_opportunity_detector_selects_lower_ask_as_buy_and_higher_bid_as_sell() -> None:
    config = _cross_config(
        arbitrage_min_bps=Decimal("1"),
        default_taker_fee_rate=Decimal("0"),
        slippage_max_bps=Decimal("0"),
        safety_buffer_bps=Decimal("0"),
    )
    detector = SpreadOpportunityDetector(config)

    now = datetime.utcnow()

    result = detector.detect_from_quotes(
        _binance_quote(
            bid="99.9",
            ask="100.1",
            timestamp=now,
        ),
        _bybit_quote(
            bid="104.9",
            ask="105.1",
            timestamp=now,
        ),
        timestamp=now,
    )

    assert result.found is True
    assert result.reason == OpportunityDetectionReason.OPPORTUNITY_DETECTED
    assert result.opportunity is not None
    assert result.costs is not None

    opportunity = result.opportunity

    _assert_profitable_opportunity(opportunity)

    assert opportunity.buy_exchange == "binance"
    assert opportunity.sell_exchange == "bybit"
    assert opportunity.buy_price == Decimal("100.1")
    assert opportunity.sell_price == Decimal("104.9")

    _assert_decimal_equal(opportunity.gross_edge, Decimal("4.8"))
    _assert_decimal_equal(opportunity.net_edge, Decimal("4.8"))
    _assert_decimal_equal(result.quantity, Decimal("1"))

    assert result.net_edge_bps is not None
    assert result.net_edge_bps > Decimal("1")


async def test_opportunity_detector_costs_reduce_gross_edge_to_net_edge() -> None:
    config = _cross_config(
        arbitrage_min_bps=Decimal("1"),
        default_trade_size=Decimal("2"),
        default_taker_fee_rate=Decimal("0.001"),
        slippage_max_bps=Decimal("5"),
        safety_buffer_bps=Decimal("2"),
    )
    detector = SpreadOpportunityDetector(config)

    now = datetime.utcnow()

    result = detector.detect_from_quotes(
        _binance_quote(
            bid="99.9",
            ask="100.1",
            bid_size="100",
            ask_size="100",
            timestamp=now,
        ),
        _bybit_quote(
            bid="104.9",
            ask="105.1",
            bid_size="100",
            ask_size="100",
            timestamp=now,
        ),
        timestamp=now,
    )

    assert result.found is True
    assert result.costs is not None
    assert result.opportunity is not None

    costs = result.costs
    opportunity = result.opportunity

    assert costs.gross_edge > Decimal("0")
    assert costs.total_costs > Decimal("0")
    assert costs.net_edge < costs.gross_edge
    assert opportunity.net_edge == costs.net_edge
    assert opportunity.estimated_fees == costs.estimated_fees
    assert opportunity.estimated_slippage == costs.estimated_slippage
    assert opportunity.net_edge > Decimal("0")


async def test_opportunity_detector_does_not_detect_when_net_edge_below_threshold() -> None:
    config = _cross_config(
        arbitrage_min_bps=Decimal("10000"),
        default_taker_fee_rate=Decimal("0"),
        slippage_max_bps=Decimal("0"),
        safety_buffer_bps=Decimal("0"),
    )
    detector = SpreadOpportunityDetector(config)

    now = datetime.utcnow()

    result = detector.detect_from_quotes(
        _binance_quote(
            bid="99.9",
            ask="100.1",
            timestamp=now,
        ),
        _bybit_quote(
            bid="100.2",
            ask="100.4",
            timestamp=now,
        ),
        timestamp=now,
    )

    assert result.found is False
    assert result.opportunity is None
    assert result.reason == OpportunityDetectionReason.NET_EDGE_BPS_BELOW_THRESHOLD
    assert result.net_edge_bps is not None
    assert result.net_edge_bps < Decimal("10000")


async def test_opportunity_detector_rejects_symbol_and_instrument_mismatch() -> None:
    config = _cross_config()
    detector = SpreadOpportunityDetector(config)

    now = datetime.utcnow()

    symbol_result = detector.detect_from_quotes(
        _binance_quote(symbol="BTCUSDT", timestamp=now),
        _bybit_quote(symbol="ETHUSDT", timestamp=now),
        timestamp=now,
    )

    assert symbol_result.found is False
    assert symbol_result.reason == OpportunityDetectionReason.SYMBOL_MISMATCH

    instrument_result = detector.detect_from_quotes(
        _binance_quote(
            instrument_type=InstrumentType.SPOT,
            timestamp=now,
        ),
        _bybit_quote(
            instrument_type=InstrumentType.PERPETUAL,
            timestamp=now,
        ),
        timestamp=now,
    )

    assert instrument_result.found is False
    assert instrument_result.reason == OpportunityDetectionReason.INSTRUMENT_TYPE_MISMATCH


# ============================================================
# Opportunity + signal pipeline
# ============================================================


async def test_real_eventbus_profitable_spread_publishes_opportunity_and_arbitrage_signal(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            arbitrage_min_bps=Decimal("1"),
            widening_bps_threshold=Decimal("10000"),
            anomaly_zscore_threshold=Decimal("100"),
            default_taker_fee_rate=Decimal("0"),
            slippage_max_bps=Decimal("0"),
            safety_buffer_bps=Decimal("0"),
            cooldown_seconds=0,
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    opportunity_recorder = EventRecorder()
    signal_recorder = EventRecorder()

    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.cross.snapshot.recorder",
    )
    opportunity_subscription = await _subscribe(
        event_bus,
        OPPORTUNITY_TOPIC,
        opportunity_recorder.handler,
        name="test.cross.opportunity.recorder",
    )
    signal_subscription = await _subscribe(
        event_bus,
        SIGNAL_TOPIC,
        signal_recorder.handler,
        name="test.cross.signal.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _binance_quote(
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _bybit_quote(
                bid="104.9",
                ask="105.1",
                timestamp=now,
            ),
        )

        await snapshot_recorder.wait_for_count(1)
        await opportunity_recorder.wait_for_count(1)
        await signal_recorder.wait_for_count(1)

        snapshot = snapshot_recorder.payloads()[0]
        opportunity = opportunity_recorder.payloads()[0]

        assert isinstance(snapshot, SpreadSnapshot)
        assert isinstance(opportunity, ArbitrageOpportunity)

        _assert_profitable_opportunity(opportunity)

        assert opportunity.buy_exchange == "binance"
        assert opportunity.sell_exchange == "bybit"
        assert opportunity.buy_price == Decimal("100.1")
        assert opportunity.sell_price == Decimal("104.9")
        assert opportunity.net_edge == Decimal("4.8")

        assert snapshot.net_spread == opportunity.net_edge
        assert snapshot.estimated_fees == Decimal("0.0") or snapshot.estimated_fees == Decimal("0")
        assert snapshot.estimated_slippage == Decimal("0.0") or snapshot.estimated_slippage == Decimal("0")
        assert snapshot.metadata["opportunity_reason"] == "opportunity_detected"
        assert snapshot.metadata["opportunity_status"] == OpportunityStatus.ACTIVE.value
        assert snapshot.metadata["opportunity_net_edge"] == str(opportunity.net_edge)

        signal_payloads = signal_recorder.payloads()

        assert SpreadSignalType.ARBITRAGE in _signal_types(signal_payloads)

        arbitrage_signal = next(
            signal
            for signal in signal_payloads
            if isinstance(signal, SpreadSignal)
            and signal.signal_type == SpreadSignalType.ARBITRAGE
        )

        assert arbitrage_signal.spread_type == SpreadType.CROSS_EXCHANGE
        assert arbitrage_signal.symbol == "BTCUSDT"
        assert arbitrage_signal.exchange_a == "binance"
        assert arbitrage_signal.exchange_b == "bybit"
        assert arbitrage_signal.value == opportunity.net_edge
        assert arbitrage_signal.threshold == Decimal("1")
        assert arbitrage_signal.metadata["reason"] == "signal_built"

        stats = analyzer.get_stats()

        assert stats["snapshots_built"] == 1
        assert stats["snapshots_published"] == 1
        assert stats["opportunities_detected"] == 1
        assert stats["opportunities_published"] == 1
        assert stats["latest_opportunities"] == 1
        assert stats["signals_built"] >= 1
        assert stats["signals_published"] >= 1
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)
        await _unsubscribe(event_bus, opportunity_subscription)
        await _unsubscribe(event_bus, signal_subscription)


async def test_pipeline_does_not_publish_opportunity_when_net_edge_is_not_profitable(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            arbitrage_min_bps=Decimal("1"),
            default_taker_fee_rate=Decimal("0.01"),
            slippage_max_bps=Decimal("10"),
            safety_buffer_bps=Decimal("10"),
            widening_bps_threshold=Decimal("10000"),
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    opportunity_recorder = EventRecorder()

    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.cross.snapshot.recorder",
    )
    opportunity_subscription = await _subscribe(
        event_bus,
        OPPORTUNITY_TOPIC,
        opportunity_recorder.handler,
        name="test.cross.opportunity.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _binance_quote(
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _bybit_quote(
                bid="100.2",
                ask="100.4",
                timestamp=now,
            ),
        )

        await snapshot_recorder.wait_for_count(1)
        await opportunity_recorder.assert_no_new_events(0)

        snapshot = snapshot_recorder.payloads()[0]

        assert isinstance(snapshot, SpreadSnapshot)
        assert snapshot.metadata["opportunity_reason"] in {
            "net_edge_not_profitable",
            "net_edge_bps_below_threshold",
            "no_positive_gross_edge",
            "non_positive_gross_edge_after_leg_selection",
        }

        stats = analyzer.get_stats()

        assert stats["snapshots_built"] == 1
        assert stats["opportunity_detection_misses"] == 1
        assert stats["opportunities_detected"] == 0
        assert stats["opportunities_published"] == 0
        assert stats["latest_opportunities"] == 0
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)
        await _unsubscribe(event_bus, opportunity_subscription)


# ============================================================
# get_best_opportunities
# ============================================================


async def test_get_best_opportunities_filters_sorts_and_limits_results(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            arbitrage_min_bps=Decimal("1"),
            default_taker_fee_rate=Decimal("0"),
            slippage_max_bps=Decimal("0"),
            safety_buffer_bps=Decimal("0"),
            cooldown_seconds=0,
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    opportunity_recorder = EventRecorder()
    opportunity_subscription = await _subscribe(
        event_bus,
        OPPORTUNITY_TOPIC,
        opportunity_recorder.handler,
        name="test.cross.opportunity.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        # BTC opportunity, net edge around 4.8
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _binance_quote(
                symbol="BTCUSDT",
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _bybit_quote(
                symbol="BTCUSDT",
                bid="104.9",
                ask="105.1",
                timestamp=now,
            ),
        )

        # ETH opportunity, net edge around 9.8, should sort before BTC.
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _quote(
                exchange="binance",
                symbol="ETHUSDT",
                instrument_type=InstrumentType.PERPETUAL,
                bid="199.9",
                ask="200.1",
                timestamp=now + timedelta(milliseconds=10),
                sequence_id=10,
            ),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _quote(
                exchange="okx",
                symbol="ETHUSDT",
                instrument_type=InstrumentType.PERPETUAL,
                bid="209.9",
                ask="210.1",
                timestamp=now + timedelta(milliseconds=15),
                sequence_id=11,
            ),
        )

        await opportunity_recorder.wait_for_count(2)

        all_opportunities = analyzer.get_best_opportunities(
            profitable_only=True,
            active_only=True,
        )

        assert len(all_opportunities) == 2
        assert all_opportunities[0].net_edge >= all_opportunities[1].net_edge
        assert all_opportunities[0].symbol == "ETHUSDT"
        assert all_opportunities[1].symbol == "BTCUSDT"

        btc_only = analyzer.get_best_opportunities(symbol="BTC/USDT")

        assert len(btc_only) == 1
        assert btc_only[0].symbol == "BTCUSDT"

        perpetual_only = analyzer.get_best_opportunities(
            instrument_type=InstrumentType.PERPETUAL,
        )

        assert len(perpetual_only) == 2

        limited = analyzer.get_best_opportunities(limit=1)

        assert len(limited) == 1
        assert limited[0].symbol == "ETHUSDT"
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, opportunity_subscription)


async def test_get_best_opportunities_expires_old_items_when_active_only(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            arbitrage_min_bps=Decimal("1"),
            default_taker_fee_rate=Decimal("0"),
            slippage_max_bps=Decimal("0"),
            safety_buffer_bps=Decimal("0"),
            opportunity_ttl_seconds=0.01,
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    opportunity_recorder = EventRecorder()
    opportunity_subscription = await _subscribe(
        event_bus,
        OPPORTUNITY_TOPIC,
        opportunity_recorder.handler,
        name="test.cross.opportunity.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _binance_quote(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _bybit_quote(timestamp=now),
        )

        await opportunity_recorder.wait_for_count(1)

        assert len(analyzer.get_best_opportunities(active_only=True)) == 1

        await asyncio.sleep(0.03)

        active = analyzer.get_best_opportunities(active_only=True)
        all_items = analyzer.get_best_opportunities(active_only=False)

        assert active == []
        assert len(all_items) == 1
        assert all_items[0].status == OpportunityStatus.EXPIRED

        stats = analyzer.get_stats()
        assert stats["latest_opportunities"] == 1
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, opportunity_subscription)


# ============================================================
# Rolling stats
# ============================================================


async def test_rolling_window_updates_after_multiple_cross_exchange_snapshots(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            arbitrage_min_bps=Decimal("10000"),
            widening_bps_threshold=Decimal("10000"),
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
        name="test.cross.snapshot.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _binance_quote(timestamp=now),
        )

        bybit_values = [
            ("104.9", "105.1"),
            ("106.9", "107.1"),
            ("108.9", "109.1"),
        ]

        for index, (bid, ask) in enumerate(bybit_values, start=2):
            await _emit_market_event(
                event_bus,
                QUOTE_TOPIC,
                _bybit_quote(
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
        assert last_snapshot.stats.last_value == Decimal("9.0")
        assert last_snapshot.stats.min_value == Decimal("5.0")
        assert last_snapshot.stats.max_value == Decimal("9.0")

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


async def test_cleanup_stale_state_removes_quotes_snapshots_opportunities_and_windows(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            stale_state_ttl_seconds=0.01,
            opportunity_ttl_seconds=0.01,
            cleanup_interval_seconds=3_600.0,
            arbitrage_min_bps=Decimal("1"),
            default_taker_fee_rate=Decimal("0"),
            slippage_max_bps=Decimal("0"),
            safety_buffer_bps=Decimal("0"),
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    opportunity_recorder = EventRecorder()

    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.cross.snapshot.recorder",
    )
    opportunity_subscription = await _subscribe(
        event_bus,
        OPPORTUNITY_TOPIC,
        opportunity_recorder.handler,
        name="test.cross.opportunity.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _binance_quote(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _bybit_quote(timestamp=now),
        )

        await snapshot_recorder.wait_for_count(1)
        await opportunity_recorder.wait_for_count(1)

        assert analyzer.get_stats()["quotes_cached"] == 2
        assert analyzer.get_stats()["latest_snapshots"] == 1
        assert analyzer.get_stats()["latest_opportunities"] == 1
        assert analyzer.get_stats()["active_windows"] == 1

        await asyncio.sleep(0.03)

        await analyzer.cleanup_stale_state()

        stats = analyzer.get_stats()

        assert stats["cleanup_runs"] == 1
        assert stats["quotes_cached"] == 0
        assert stats["latest_snapshots"] == 0
        assert stats["latest_opportunities"] == 0

        # Якщо цей assert впаде, це така сама orphan-window проблема,
        # яку вже знайшли в spot/futures analyzer.
        assert stats["active_windows"] == 0

        assert stats["cleanup_removed_quotes"] >= 2
        assert stats["cleanup_removed_snapshots"] >= 1
        assert stats["cleanup_removed_opportunities"] >= 1
        assert stats["cleanup_removed_windows"] >= 1
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)
        await _unsubscribe(event_bus, opportunity_subscription)


# ============================================================
# Stats contract
# ============================================================


async def test_stats_reflect_complete_cross_exchange_pipeline(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            arbitrage_min_bps=Decimal("1"),
            widening_bps_threshold=Decimal("10000"),
            anomaly_zscore_threshold=Decimal("100"),
            default_taker_fee_rate=Decimal("0"),
            slippage_max_bps=Decimal("0"),
            safety_buffer_bps=Decimal("0"),
            cooldown_seconds=0,
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    opportunity_recorder = EventRecorder()
    signal_recorder = EventRecorder()

    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.cross.snapshot.recorder",
    )
    opportunity_subscription = await _subscribe(
        event_bus,
        OPPORTUNITY_TOPIC,
        opportunity_recorder.handler,
        name="test.cross.opportunity.recorder",
    )
    signal_subscription = await _subscribe(
        event_bus,
        SIGNAL_TOPIC,
        signal_recorder.handler,
        name="test.cross.signal.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _binance_quote(
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            QUOTE_TOPIC,
            _bybit_quote(
                bid="104.9",
                ask="105.1",
                timestamp=now,
            ),
        )

        await snapshot_recorder.wait_for_count(1)
        await opportunity_recorder.wait_for_count(1)
        await signal_recorder.wait_for_count(1)

        stats = analyzer.get_stats()

        assert stats["running"] is True
        assert stats["registered"] is True
        assert stats["enabled"] is True

        assert stats["quote_events_received"] == 2
        assert stats["quotes_received"] == 2
        assert stats["invalid_payloads"] == 0
        assert stats["invalid_quotes"] == 0
        assert stats["incomplete_quotes"] == 0
        assert stats["stale_quotes"] == 0
        assert stats["unaligned_quotes"] == 0
        assert stats["instrument_type_skips"] == 0
        assert stats["preferred_exchange_skips"] == 0

        assert stats["quotes_stored"] == 2
        assert stats["quotes_cached"] == 2

        assert stats["snapshots_built"] == 1
        assert stats["snapshots_published"] == 1
        assert stats["latest_snapshots"] == 1
        assert stats["active_windows"] == 1

        assert stats["opportunities_detected"] == 1
        assert stats["opportunities_published"] == 1
        assert stats["opportunity_detection_misses"] == 0
        assert stats["latest_opportunities"] == 1

        assert stats["signals_built"] >= 1
        assert stats["signals_published"] >= 1

        snapshot = snapshot_recorder.payloads()[0]
        opportunity = opportunity_recorder.payloads()[0]

        assert isinstance(snapshot, SpreadSnapshot)
        assert isinstance(opportunity, ArbitrageOpportunity)

        assert snapshot.symbol == "BTCUSDT"
        assert opportunity.symbol == "BTCUSDT"

        assert opportunity.buy_exchange == "binance"
        assert opportunity.sell_exchange == "bybit"
        assert opportunity.net_edge == Decimal("4.8")
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)
        await _unsubscribe(event_bus, opportunity_subscription)
        await _unsubscribe(event_bus, signal_subscription)