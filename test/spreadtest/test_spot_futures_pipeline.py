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
    SpreadSignalType,
    SpreadType,
)
from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler


pytestmark = pytest.mark.asyncio


# ============================================================
# Topics / constants
# ============================================================

ORDERBOOK_TOPIC = "market.orderbook.updated"
LEGACY_QUOTE_TOPIC = "market.quote.updated"
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
# Generic payload helpers
# ============================================================

def _payload_value(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _payload_metadata(payload: Any) -> dict[str, Any]:
    metadata = _payload_value(payload, "metadata", {})
    return dict(metadata or {})


def _payload_enum_value(payload: Any, key: str) -> str | None:
    value = _payload_value(payload, key)
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _payload_decimal(payload: Any, key: str) -> Decimal | None:
    value = _payload_value(payload, key)
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _signal_type_value(signal: Any) -> str | None:
    signal_type = _payload_value(signal, "signal_type")
    if signal_type is None:
        return None
    if hasattr(signal_type, "value"):
        return str(signal_type.value)
    return str(signal_type)


def _assert_decimal_equal(
    actual: Decimal | str | None,
    expected: Decimal,
    *,
    quant: Decimal = Decimal("0.00000001"),
) -> None:
    assert actual is not None
    actual_decimal = actual if isinstance(actual, Decimal) else Decimal(str(actual))
    assert actual_decimal.quantize(quant) == expected.quantize(quant)


def _assert_stats_orderbook_aliases(
    stats: dict[str, Any],
    expected_orderbook_events: int,
) -> None:
    assert stats["orderbook_events_received"] == expected_orderbook_events

    # Backward-compatible alias. Його можна буде прибрати пізніше,
    # але поки він корисний для плавного переходу.
    assert stats["quote_events_received"] == expected_orderbook_events


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
        "orderbook_event_topic": ORDERBOOK_TOPIC,
        "orderbook_event_topic_patterns": (ORDERBOOK_TOPIC,),
        "funding_event_topic": FUNDING_TOPIC,
        "funding_event_topic_patterns": (FUNDING_TOPIC,),
        "snapshot_event_topic": SNAPSHOT_TOPIC,
        "signal_event_topic": SIGNAL_TOPIC,
        "allow_legacy_quote_topics": False,
        "allow_legacy_raw_topics": False,
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


def _market_type_for_instrument(instrument_type: InstrumentType | str) -> str:
    value = instrument_type.value if hasattr(instrument_type, "value") else str(instrument_type)
    value = value.lower()

    if value == InstrumentType.SPOT.value:
        return "spot"
    if value == InstrumentType.FUTURES.value:
        return "futures"
    return "perpetual"


def _ts_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _orderbook_payload(
    *,
    exchange: str,
    symbol: str = "BTCUSDT",
    instrument_type: InstrumentType | str,
    market_type: str | None = None,
    bid: Decimal | str | None = "99.9",
    ask: Decimal | str | None = "100.1",
    bid_size: Decimal | str | None = "10",
    ask_size: Decimal | str | None = "10",
    timestamp: datetime | None = None,
    received_at: datetime | None = None,
    sequence_id: int | None = None,
    timeframe: str = "realtime",
    exchange_symbol: str | None = None,
    shape: str = "best_bid_best_ask",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Emulates market.orderbook.updated payload from OrderBookCache.

    Shapes intentionally cover multiple realistic top-of-book formats:
    - best_bid_best_ask
    - best_bid_price_best_ask_price
    - bid_ask
    - levels_dict_price_quantity
    - levels_dict_p_q
    - levels_tuple
    - levels_list
    """
    now = timestamp or datetime.utcnow()
    received = received_at or now

    resolved_market_type = market_type or _market_type_for_instrument(instrument_type)

    payload: dict[str, Any] = {
        "exchange": exchange,
        "symbol": symbol,
        "exchange_symbol": exchange_symbol or symbol,
        "instrument_type": instrument_type.value if hasattr(instrument_type, "value") else instrument_type,
        "market_type": resolved_market_type,
        "timeframe": timeframe,
        "timestamp": now,
        "timestamp_ms": _ts_ms(now),
        "received_at": received,
        "received_at_ms": _ts_ms(received),
        "sequence_id": sequence_id,
        "metadata": {
            "source": "test_orderbook_cache",
            "shape": shape,
            **dict(metadata or {}),
        },
    }

    if shape == "best_bid_best_ask":
        payload.update(
            {
                "best_bid": bid,
                "best_ask": ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
            }
        )
    elif shape == "best_bid_price_best_ask_price":
        payload.update(
            {
                "best_bid_price": bid,
                "best_ask_price": ask,
                "best_bid_size": bid_size,
                "best_ask_size": ask_size,
            }
        )
    elif shape == "bid_ask":
        payload.update(
            {
                "bid": bid,
                "ask": ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
            }
        )
    elif shape == "levels_dict_price_quantity":
        payload.update(
            {
                "bids": [{"price": bid, "quantity": bid_size}],
                "asks": [{"price": ask, "quantity": ask_size}],
            }
        )
    elif shape == "levels_dict_p_q":
        payload.update(
            {
                "bids": [{"p": bid, "q": bid_size}],
                "asks": [{"p": ask, "q": ask_size}],
            }
        )
    elif shape == "levels_tuple":
        payload.update(
            {
                "bids": [(bid, bid_size)],
                "asks": [(ask, ask_size)],
            }
        )
    elif shape == "levels_list":
        payload.update(
            {
                "bids": [[bid, bid_size]],
                "asks": [[ask, ask_size]],
            }
        )
    elif shape == "empty_levels":
        payload.update({"bids": [], "asks": []})
    elif shape == "missing_prices":
        payload.update({"bid_size": bid_size, "ask_size": ask_size})
    else:
        raise ValueError(f"Unknown orderbook payload shape: {shape}")

    return payload


def _spot_orderbook(
    *,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    bid: Decimal | str | None = "99.9",
    ask: Decimal | str | None = "100.1",
    timestamp: datetime | None = None,
    sequence_id: int | None = 1,
    shape: str = "best_bid_best_ask",
    market_type: str = "spot",
) -> dict[str, Any]:
    return _orderbook_payload(
        exchange=exchange,
        symbol=symbol,
        instrument_type=InstrumentType.SPOT,
        market_type=market_type,
        bid=bid,
        ask=ask,
        timestamp=timestamp,
        sequence_id=sequence_id,
        shape=shape,
    )


def _futures_orderbook(
    *,
    exchange: str = "bybit",
    symbol: str = "BTCUSDT",
    bid: Decimal | str | None = "100.9",
    ask: Decimal | str | None = "101.1",
    timestamp: datetime | None = None,
    sequence_id: int | None = 2,
    shape: str = "best_bid_best_ask",
    instrument_type: InstrumentType = InstrumentType.PERPETUAL,
    market_type: str = "perpetual",
) -> dict[str, Any]:
    return _orderbook_payload(
        exchange=exchange,
        symbol=symbol,
        instrument_type=instrument_type,
        market_type=market_type,
        bid=bid,
        ask=ask,
        timestamp=timestamp,
        sequence_id=sequence_id,
        shape=shape,
    )


def _quote_snapshot(
    *,
    exchange: str,
    symbol: str = "BTCUSDT",
    instrument_type: InstrumentType,
    bid: Decimal | str | None,
    ask: Decimal | str | None,
    bid_size: Decimal | str | None = "10",
    ask_size: Decimal | str | None = "10",
    timestamp: datetime | None = None,
    sequence_id: int | None = None,
) -> QuoteSnapshot:
    now = timestamp or datetime.utcnow()
    return QuoteSnapshot(
        exchange=exchange,
        symbol=symbol,
        instrument_type=instrument_type,
        market_type=_market_type_for_instrument(instrument_type),
        bid=Decimal(str(bid)) if bid is not None else None,
        ask=Decimal(str(ask)) if ask is not None else None,
        bid_size=Decimal(str(bid_size)) if bid_size is not None else None,
        ask_size=Decimal(str(ask_size)) if ask_size is not None else None,
        timestamp=now,
        received_at=now,
        sequence_id=sequence_id,
        metadata={"source": "legacy_direct_quote_snapshot"},
    )


def _funding(
    *,
    exchange: str = "bybit",
    symbol: str = "BTCUSDT",
    market_type: str = "perpetual",
    funding_rate: Decimal | str = "0.01",
    timestamp: datetime | None = None,
    timeframe: str = "realtime",
) -> FundingSnapshot:
    return FundingSnapshot(
        exchange=exchange,
        symbol=symbol,
        market_type=market_type,
        timeframe=timeframe,
        funding_rate=Decimal(str(funding_rate)),
        timestamp=timestamp or datetime.utcnow(),
        interval_hours=8,
        metadata={"source": "test_funding_cache"},
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
    assert stats["price_input_source"] == ORDERBOOK_TOPIC
    assert stats["funding_input_source"] == FUNDING_TOPIC
    assert stats["spot_quotes_cached"] == 0
    assert stats["futures_quotes_cached"] == 0
    assert stats["funding_cached"] == 0
    assert stats["latest_snapshots"] == 0


async def test_orderbook_event_is_ignored_when_analyzer_not_running(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    analyzer.register()

    await _emit_market_event(
        event_bus,
        ORDERBOOK_TOPIC,
        _spot_orderbook(),
    )
    await _drain_event_bus()

    stats = analyzer.get_stats()

    assert stats["orderbook_events_received"] == 0
    assert stats["quote_events_received"] == 0
    assert stats["quotes_received"] == 0
    assert stats["quotes_stored"] == 0
    assert stats["spot_quotes_cached"] == 0


async def test_direct_legacy_quote_snapshot_handler_still_accepts_quote_snapshot(
    analyzer: SpotFuturesSpreadAnalyzer,
) -> None:
    await analyzer.start()

    now = datetime.utcnow()

    spot_event = Event(
        topic=LEGACY_QUOTE_TOPIC,
        payload=_quote_snapshot(
            exchange="binance",
            instrument_type=InstrumentType.SPOT,
            bid="99.9",
            ask="100.1",
            timestamp=now,
            sequence_id=101,
        ),
        priority=EventPriority.NORMAL,
        source="test.legacy",
    )
    futures_event = Event(
        topic=LEGACY_QUOTE_TOPIC,
        payload=_quote_snapshot(
            exchange="bybit",
            instrument_type=InstrumentType.PERPETUAL,
            bid="100.9",
            ask="101.1",
            timestamp=now,
            sequence_id=202,
        ),
        priority=EventPriority.NORMAL,
        source="test.legacy",
    )

    await analyzer.on_quote_update(spot_event)
    await analyzer.on_quote_update(futures_event)

    stats = analyzer.get_stats()

    assert stats["quotes_received"] == 2
    assert stats["quotes_stored"] == 2
    assert stats["spot_quotes_cached"] == 1
    assert stats["futures_quotes_cached"] == 1
    assert stats["snapshots_built"] == 1


# ============================================================
# Orderbook processing / validation
# ============================================================

async def test_rejects_invalid_orderbook_payload_and_updates_stats(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    await _emit_market_event(event_bus, ORDERBOOK_TOPIC, {"bad": "payload"})

    await _wait_until(
        lambda: analyzer.get_stats()["invalid_payloads"] == 1,
        message="Invalid orderbook payload was not counted",
    )

    stats = analyzer.get_stats()

    _assert_stats_orderbook_aliases(stats, 1)
    assert stats["invalid_payloads"] == 1
    assert stats["quotes_received"] == 0
    assert stats["quotes_stored"] == 0


@pytest.mark.parametrize(
    "shape",
    [
        "best_bid_best_ask",
        "best_bid_price_best_ask_price",
        "bid_ask",
        "levels_dict_price_quantity",
        "levels_dict_p_q",
        "levels_tuple",
        "levels_list",
    ],
)
async def test_orderbook_payload_shapes_are_normalized_and_cached(
    event_bus: EventBus,
    scheduler: Scheduler,
    shape: str,
) -> None:
    analyzer = SpotFuturesSpreadAnalyzer(
        config=_spot_config(
            widening_bps_threshold=Decimal("1000"),
            anomaly_zscore_threshold=Decimal("100"),
            mean_reversion_zscore_threshold=Decimal("100"),
            regime_shift_zscore_threshold=Decimal("100"),
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _spot_orderbook(
                exchange="binance",
                bid="99.9",
                ask="100.1",
                timestamp=now,
                sequence_id=1,
                shape=shape,
            ),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(
                exchange="bybit",
                bid="100.9",
                ask="101.1",
                timestamp=now,
                sequence_id=2,
                shape=shape,
            ),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["snapshots_built"] == 1,
            message=f"Orderbook shape {shape!r} did not build a snapshot",
        )

        stats = analyzer.get_stats()

        _assert_stats_orderbook_aliases(stats, 2)
        assert stats["quotes_received"] == 2
        assert stats["quotes_stored"] == 2
        assert stats["spot_quotes_cached"] == 1
        assert stats["futures_quotes_cached"] == 1
        assert stats["invalid_payloads"] == 0
        assert stats["invalid_quotes"] == 0
        assert stats["snapshots_built"] == 1
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"exchange": "binance", "symbol": "BTCUSDT", "instrument_type": "spot"},
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "instrument_type": "spot",
            "market_type": "spot",
            "bids": [],
            "asks": [],
        },
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "instrument_type": "spot",
            "market_type": "spot",
            "best_bid": "not-a-decimal",
            "best_ask": "100.1",
        },
    ],
)
async def test_rejects_bad_orderbook_payload_shapes(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
    bad_payload: dict[str, Any],
) -> None:
    await analyzer.start()

    await _emit_market_event(event_bus, ORDERBOOK_TOPIC, bad_payload)

    await _wait_until(
        lambda: analyzer.get_stats()["invalid_payloads"] >= 1
        or analyzer.get_stats()["invalid_quotes"] >= 1
        or analyzer.get_stats()["incomplete_quotes"] >= 1,
        message="Bad orderbook payload was not rejected",
    )

    stats = analyzer.get_stats()

    assert stats["quotes_stored"] == 0
    assert stats["spot_quotes_cached"] == 0
    assert stats["futures_quotes_cached"] == 0


async def test_rejects_stale_orderbook_and_does_not_cache_it(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    stale_timestamp = datetime.utcnow() - timedelta(minutes=10)

    await _emit_market_event(
        event_bus,
        ORDERBOOK_TOPIC,
        _spot_orderbook(
            exchange="binance",
            timestamp=stale_timestamp,
        ),
    )

    await _wait_until(
        lambda: analyzer.get_stats()["stale_quotes"] == 1,
        message="Stale orderbook-derived quote was not counted",
    )

    stats = analyzer.get_stats()

    _assert_stats_orderbook_aliases(stats, 1)
    assert stats["quotes_received"] == 1
    assert stats["stale_quotes"] == 1
    assert stats["quotes_stored"] == 0
    assert stats["spot_quotes_cached"] == 0


async def test_rejects_incomplete_orderbook_and_does_not_cache_it(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    await _emit_market_event(
        event_bus,
        ORDERBOOK_TOPIC,
        _spot_orderbook(
            exchange="binance",
            bid=None,
            ask="100.1",
        ),
    )

    await _wait_until(
        lambda: analyzer.get_stats()["invalid_quotes"]
        + analyzer.get_stats()["incomplete_quotes"]
        + analyzer.get_stats()["invalid_payloads"] >= 1,
        message="Incomplete orderbook was not rejected",
    )

    stats = analyzer.get_stats()

    _assert_stats_orderbook_aliases(stats, 1)
    assert stats["quotes_stored"] == 0
    assert stats["spot_quotes_cached"] == 0


async def test_rejects_crossed_orderbook_and_does_not_cache_it(
    analyzer: SpotFuturesSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    await _emit_market_event(
        event_bus,
        ORDERBOOK_TOPIC,
        _spot_orderbook(
            exchange="binance",
            bid="101.0",
            ask="100.0",
        ),
    )

    await _wait_until(
        lambda: analyzer.get_stats()["invalid_quotes"]
        + analyzer.get_stats()["incomplete_quotes"] >= 1,
        message="Crossed orderbook was not rejected",
    )

    stats = analyzer.get_stats()

    assert stats["quotes_stored"] == 0
    assert stats["spot_quotes_cached"] == 0


async def test_stores_spot_orderbook_without_snapshot_until_futures_orderbook_arrives(
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
            ORDERBOOK_TOPIC,
            _spot_orderbook(exchange="binance"),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["spot_quotes_cached"] == 1,
            message="Spot orderbook quote was not cached",
        )

        stats = analyzer.get_stats()

        assert stats["quotes_stored"] == 1
        assert stats["futures_quotes_cached"] == 0
        assert stats["snapshots_built"] == 0

        await snapshot_recorder.assert_no_new_events(0)
    finally:
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_stores_futures_orderbook_without_snapshot_until_spot_orderbook_arrives(
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
            ORDERBOOK_TOPIC,
            _futures_orderbook(exchange="bybit"),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["futures_quotes_cached"] == 1,
            message="Futures orderbook quote was not cached",
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

async def test_real_eventbus_orderbook_flow_builds_spot_futures_snapshot(
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
            ORDERBOOK_TOPIC,
            _spot_orderbook(
                exchange="Binance",
                symbol="BTC/USDT",
                bid="99.9",
                ask="100.1",
                timestamp=now,
                sequence_id=101,
                shape="levels_dict_price_quantity",
            ),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(
                exchange="Bybit",
                symbol="BTC-USDT",
                bid="100.9",
                ask="101.1",
                timestamp=now + timedelta(milliseconds=20),
                sequence_id=202,
                shape="levels_tuple",
            ),
        )

        await snapshot_recorder.wait_for_count(1)

        payload = snapshot_recorder.payloads()[0]

        assert _payload_enum_value(payload, "spread_type") == SpreadType.SPOT_FUTURES.value
        assert _payload_value(payload, "symbol") == "BTCUSDT"

        assert _payload_value(payload, "leg_a_exchange") == "binance"
        assert _payload_value(payload, "leg_b_exchange") == "bybit"

        assert _payload_enum_value(payload, "leg_a_type") == InstrumentType.SPOT.value
        assert _payload_enum_value(payload, "leg_b_type") == InstrumentType.PERPETUAL.value

        _assert_decimal_equal(_payload_decimal(payload, "leg_a_mid"), Decimal("100.0"))
        _assert_decimal_equal(_payload_decimal(payload, "leg_b_mid"), Decimal("101.0"))

        _assert_decimal_equal(_payload_decimal(payload, "raw_spread"), Decimal("1.0"))
        _assert_decimal_equal(_payload_decimal(payload, "basis"), Decimal("1.0"))
        _assert_decimal_equal(_payload_decimal(payload, "spread_pct"), Decimal("1.0"))
        _assert_decimal_equal(_payload_decimal(payload, "spread_bps"), Decimal("100.0"))
        _assert_decimal_equal(_payload_decimal(payload, "funding_adjusted_spread"), Decimal("1.0"))
        _assert_decimal_equal(_payload_decimal(payload, "net_spread"), Decimal("1.0"))

        metadata = _payload_metadata(payload)

        assert metadata["price_input_source"] == ORDERBOOK_TOPIC
        assert metadata["spot_exchange"] == "binance"
        assert metadata["futures_exchange"] == "bybit"
        assert metadata["spot_sequence_id"] == 101
        assert metadata["futures_sequence_id"] == 202

        latest = analyzer.get_latest_snapshot(
            "BTC_USDT",
            "BINANCE",
            "BYBIT",
        )

        assert latest is not None
        assert latest.symbol == "BTCUSDT"
        assert latest.metadata["price_input_source"] == ORDERBOOK_TOPIC

        stats = analyzer.get_stats()

        _assert_stats_orderbook_aliases(stats, 2)
        assert stats["quotes_received"] == 2
        assert stats["quotes_stored"] == 2
        assert stats["spot_quotes_cached"] == 1
        assert stats["futures_quotes_cached"] == 1
        assert stats["snapshots_built"] == 1
        assert stats["latest_snapshots"] == 1
        assert stats["active_windows"] == 1
    finally:
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_spot_premium_over_futures_builds_negative_basis_snapshot(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = SpotFuturesSpreadAnalyzer(
        config=_spot_config(
            widening_bps_threshold=Decimal("10000"),
            anomaly_zscore_threshold=Decimal("100"),
            mean_reversion_zscore_threshold=Decimal("100"),
            regime_shift_zscore_threshold=Decimal("100"),
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.spot_premium.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _spot_orderbook(
                exchange="binance",
                bid="104.9",
                ask="105.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(
                exchange="bybit",
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )

        await snapshot_recorder.wait_for_count(1)

        payload = snapshot_recorder.payloads()[0]

        _assert_decimal_equal(_payload_decimal(payload, "leg_a_mid"), Decimal("105.0"))
        _assert_decimal_equal(_payload_decimal(payload, "leg_b_mid"), Decimal("100.0"))
        _assert_decimal_equal(_payload_decimal(payload, "raw_spread"), Decimal("-5.0"))
        _assert_decimal_equal(_payload_decimal(payload, "basis"), Decimal("-5.0"))
        _assert_decimal_equal(
            _payload_decimal(payload, "spread_bps"),
            Decimal("-476.19047619"),
        )
        _assert_decimal_equal(
            _payload_decimal(payload, "spread_pct"),
            Decimal("-4.76190476"),
        )

        metadata = _payload_metadata(payload)
        assert metadata["price_input_source"] == ORDERBOOK_TOPIC

        stats = analyzer.get_stats()
        assert stats["snapshots_built"] == 1
        assert stats["invalid_quotes"] == 0
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_unaligned_orderbooks_are_skipped_and_no_snapshot_is_published(
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
            ORDERBOOK_TOPIC,
            _spot_orderbook(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(timestamp=now + timedelta(seconds=2)),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["unaligned_quotes"] == 1,
            message="Unaligned orderbook pair was not counted",
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

async def test_funding_update_is_stored_without_snapshot_until_orderbooks_exist(
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
            ORDERBOOK_TOPIC,
            _spot_orderbook(
                exchange="binance",
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(
                exchange="bybit",
                bid="100.9",
                ask="101.1",
                timestamp=now,
            ),
        )

        await snapshot_recorder.wait_for_count(1)

        payload = snapshot_recorder.payloads()[0]

        _assert_decimal_equal(_payload_decimal(payload, "raw_spread"), Decimal("1.0"))
        _assert_decimal_equal(_payload_decimal(payload, "funding_adjusted_spread"), Decimal("0.75"))
        _assert_decimal_equal(_payload_decimal(payload, "net_spread"), Decimal("0.75"))

        metadata = _payload_metadata(payload)
        assert metadata["funding_rate"] == "0.25"
        assert metadata["funding_timestamp"] is not None
        assert metadata["price_input_source"] == ORDERBOOK_TOPIC

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


async def test_funding_update_after_orderbooks_recalculates_and_publishes_new_snapshot(
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
            ORDERBOOK_TOPIC,
            _spot_orderbook(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(timestamp=now + timedelta(milliseconds=5)),
        )

        await snapshot_recorder.wait_for_count(1)

        first_snapshot = snapshot_recorder.payloads()[0]
        _assert_decimal_equal(
            _payload_decimal(first_snapshot, "funding_adjusted_spread"),
            Decimal("1.0"),
        )

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
        _assert_decimal_equal(
            _payload_decimal(second_snapshot, "funding_adjusted_spread"),
            Decimal("0.60"),
        )

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


async def test_funding_for_different_exchange_does_not_affect_snapshot(
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
        name="test.snapshot.funding_mismatch.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            FUNDING_TOPIC,
            _funding(
                exchange="okx",
                symbol="BTCUSDT",
                funding_rate="0.90",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _spot_orderbook(exchange="binance", timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(exchange="bybit", timestamp=now),
        )

        await snapshot_recorder.wait_for_count(1)

        payload = snapshot_recorder.payloads()[0]

        _assert_decimal_equal(_payload_decimal(payload, "funding_adjusted_spread"), Decimal("1.0"))

        metadata = _payload_metadata(payload)
        assert metadata["funding_rate"] is None
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
            ORDERBOOK_TOPIC,
            _spot_orderbook(
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(
                bid="100.9",
                ask="101.1",
                timestamp=now,
            ),
        )

        await snapshot_recorder.wait_for_count(1)
        await signal_recorder.wait_for_count(1)

        signal = signal_recorder.payloads()[0]

        assert _payload_enum_value(signal, "spread_type") == SpreadType.SPOT_FUTURES.value
        assert _signal_type_value(signal) == SpreadSignalType.WIDENING.value
        assert _payload_value(signal, "symbol") == "BTCUSDT"
        assert _payload_value(signal, "exchange_a") == "binance"
        assert _payload_value(signal, "exchange_b") == "bybit"

        _assert_decimal_equal(_payload_decimal(signal, "value"), Decimal("100.0"))
        _assert_decimal_equal(_payload_decimal(signal, "threshold"), Decimal("10"))

        metadata = _payload_metadata(signal)
        assert metadata["reason"] == "signal_built"

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
            ORDERBOOK_TOPIC,
            _spot_orderbook(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(timestamp=now),
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
            ORDERBOOK_TOPIC,
            _spot_orderbook(exchange="binance"),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(exchange="bybit"),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["futures_quotes_cached"] == 1,
            message="Allowed futures orderbook was not cached",
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
            ORDERBOOK_TOPIC,
            _spot_orderbook(exchange="binance"),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(exchange="bybit"),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["spot_quotes_cached"] == 1,
            message="Allowed spot orderbook was not cached",
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


async def test_different_symbols_do_not_pair(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = SpotFuturesSpreadAnalyzer(
        config=_spot_config(),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.snapshot.symbol_mismatch.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _spot_orderbook(symbol="BTCUSDT", timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(symbol="ETHUSDT", timestamp=now),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["quotes_stored"] == 2,
            message="Both valid quotes should be cached even if they do not pair",
        )

        stats = analyzer.get_stats()

        assert stats["spot_quotes_cached"] == 1
        assert stats["futures_quotes_cached"] == 1
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
            ORDERBOOK_TOPIC,
            _spot_orderbook(timestamp=now, sequence_id=1),
        )

        futures_values = [
            ("100.9", "101.1"),
            ("101.9", "102.1"),
            ("102.9", "103.1"),
        ]

        for index, (bid, ask) in enumerate(futures_values, start=2):
            await _emit_market_event(
                event_bus,
                ORDERBOOK_TOPIC,
                _futures_orderbook(
                    bid=bid,
                    ask=ask,
                    timestamp=now + timedelta(milliseconds=index * 10),
                    sequence_id=index,
                ),
            )

        await snapshot_recorder.wait_for_count(3)

        last_snapshot = snapshot_recorder.payloads()[-1]

        stats_payload = _payload_value(last_snapshot, "stats")

        if isinstance(stats_payload, dict):
            assert stats_payload["count"] == 3
            _assert_decimal_equal(stats_payload["last_value"], Decimal("3.0"))
            _assert_decimal_equal(stats_payload["min_value"], Decimal("1.0"))
            _assert_decimal_equal(stats_payload["max_value"], Decimal("3.0"))
            assert stats_payload["mean"] is not None
        else:
            assert stats_payload is not None
            assert stats_payload.count == 3
            assert stats_payload.last_value == Decimal("3.0")
            assert stats_payload.min_value == Decimal("1.0")
            assert stats_payload.max_value == Decimal("3.0")
            assert stats_payload.mean is not None

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
            ORDERBOOK_TOPIC,
            _spot_orderbook(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(timestamp=now),
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

async def test_stats_reflect_complete_spot_futures_orderbook_pipeline(
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
            ORDERBOOK_TOPIC,
            _spot_orderbook(
                exchange="binance",
                symbol="BTCUSDT",
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _futures_orderbook(
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
        assert stats["price_input_source"] == ORDERBOOK_TOPIC
        assert stats["funding_input_source"] == FUNDING_TOPIC

        _assert_stats_orderbook_aliases(stats, 2)
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

        assert _payload_value(snapshot, "symbol") == "BTCUSDT"
        assert _payload_value(signal, "symbol") == "BTCUSDT"
        assert _signal_type_value(signal) == SpreadSignalType.WIDENING.value

        snapshot_metadata = _payload_metadata(snapshot)
        assert snapshot_metadata["price_input_source"] == ORDERBOOK_TOPIC
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)
        await _unsubscribe(event_bus, signal_subscription)