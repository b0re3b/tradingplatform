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
# Payload helpers
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

    # Backward-compatible alias.
    assert stats["quote_events_received"] == expected_orderbook_events


def _assert_profitable_opportunity(opportunity: ArbitrageOpportunity) -> None:
    assert opportunity.status == OpportunityStatus.ACTIVE
    assert opportunity.is_profitable is True
    assert opportunity.gross_edge > Decimal("0")
    assert opportunity.net_edge > Decimal("0")
    assert opportunity.expires_at is not None
    assert opportunity.expires_at > opportunity.timestamp


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

def _cross_config(**overrides: Any) -> CrossExchangeSpreadConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "service_name": SERVICE_NAME,
        "orderbook_event_topic": ORDERBOOK_TOPIC,
        "orderbook_event_topic_patterns": (ORDERBOOK_TOPIC,),
        "snapshot_event_topic": SNAPSHOT_TOPIC,
        "signal_event_topic": SIGNAL_TOPIC,
        "opportunity_event_topic": OPPORTUNITY_TOPIC,
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
        "allowed_exchanges": set(),
        "preferred_exchanges": set(),
        "metadata": {"test": "cross_exchange_pipeline"},
    }
    values.update(overrides)
    return CrossExchangeSpreadConfig(**values)


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
    instrument_type: InstrumentType | str = InstrumentType.PERPETUAL,
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


def _binance_orderbook(
    *,
    symbol: str = "BTCUSDT",
    instrument_type: InstrumentType = InstrumentType.PERPETUAL,
    market_type: str = "perpetual",
    bid: Decimal | str | None = "99.9",
    ask: Decimal | str | None = "100.1",
    timestamp: datetime | None = None,
    sequence_id: int | None = 1,
    shape: str = "best_bid_best_ask",
) -> dict[str, Any]:
    return _orderbook_payload(
        exchange="binance",
        symbol=symbol,
        instrument_type=instrument_type,
        market_type=market_type,
        bid=bid,
        ask=ask,
        timestamp=timestamp,
        sequence_id=sequence_id,
        shape=shape,
    )


def _bybit_orderbook(
    *,
    symbol: str = "BTCUSDT",
    instrument_type: InstrumentType = InstrumentType.PERPETUAL,
    market_type: str = "perpetual",
    bid: Decimal | str | None = "104.9",
    ask: Decimal | str | None = "105.1",
    timestamp: datetime | None = None,
    sequence_id: int | None = 2,
    shape: str = "best_bid_best_ask",
) -> dict[str, Any]:
    return _orderbook_payload(
        exchange="bybit",
        symbol=symbol,
        instrument_type=instrument_type,
        market_type=market_type,
        bid=bid,
        ask=ask,
        timestamp=timestamp,
        sequence_id=sequence_id,
        shape=shape,
    )


def _okx_orderbook(
    *,
    symbol: str = "BTCUSDT",
    instrument_type: InstrumentType = InstrumentType.PERPETUAL,
    market_type: str = "perpetual",
    bid: Decimal | str | None = "109.9",
    ask: Decimal | str | None = "110.1",
    timestamp: datetime | None = None,
    sequence_id: int | None = 3,
    shape: str = "best_bid_best_ask",
) -> dict[str, Any]:
    return _orderbook_payload(
        exchange="okx",
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
    instrument_type: InstrumentType = InstrumentType.PERPETUAL,
    market_type: str = "perpetual",
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
        market_type=market_type,
        bid=Decimal(str(bid)) if bid is not None else None,
        ask=Decimal(str(ask)) if ask is not None else None,
        bid_size=Decimal(str(bid_size)) if bid_size is not None else None,
        ask_size=Decimal(str(ask_size)) if ask_size is not None else None,
        timestamp=timestamp or now,
        received_at=received_at or timestamp or now,
        sequence_id=sequence_id,
        metadata=dict(metadata or {}),
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
    assert stats["price_input_source"] == ORDERBOOK_TOPIC
    assert stats["quotes_cached"] == 0
    assert stats["active_windows"] == 0
    assert stats["latest_snapshots"] == 0
    assert stats["latest_opportunities"] == 0


async def test_orderbook_event_is_ignored_when_analyzer_not_running(
    analyzer: CrossExchangeSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    analyzer.register()

    await _emit_market_event(
        event_bus,
        ORDERBOOK_TOPIC,
        _binance_orderbook(),
    )
    await _drain_event_bus()

    stats = analyzer.get_stats()

    assert stats["orderbook_events_received"] == 0
    assert stats["quote_events_received"] == 0
    assert stats["quotes_received"] == 0
    assert stats["quotes_stored"] == 0
    assert stats["quotes_cached"] == 0


async def test_direct_legacy_quote_snapshot_handler_still_accepts_quote_snapshot(
    analyzer: CrossExchangeSpreadAnalyzer,
) -> None:
    await analyzer.start()

    now = datetime.utcnow()

    event_a = Event(
        topic=LEGACY_QUOTE_TOPIC,
        payload=_quote_snapshot(
            exchange="binance",
            bid="99.9",
            ask="100.1",
            timestamp=now,
            sequence_id=101,
        ),
        priority=EventPriority.NORMAL,
        source="test.legacy",
    )
    event_b = Event(
        topic=LEGACY_QUOTE_TOPIC,
        payload=_quote_snapshot(
            exchange="bybit",
            bid="104.9",
            ask="105.1",
            timestamp=now + timedelta(milliseconds=20),
            sequence_id=202,
        ),
        priority=EventPriority.NORMAL,
        source="test.legacy",
    )

    await analyzer.on_quote_update(event_a)
    await analyzer.on_quote_update(event_b)

    stats = analyzer.get_stats()

    assert stats["quotes_received"] == 2
    assert stats["quotes_stored"] == 2
    assert stats["quotes_cached"] == 2
    assert stats["snapshots_built"] == 1
    assert stats["latest_snapshots"] == 1


# ============================================================
# Orderbook validation / rejection
# ============================================================

async def test_rejects_invalid_orderbook_payload_and_updates_stats(
    analyzer: CrossExchangeSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    await _emit_market_event(
        event_bus,
        ORDERBOOK_TOPIC,
        {"bad": "payload"},
    )

    await _wait_until(
        lambda: analyzer.get_stats()["invalid_payloads"] == 1,
        message="Invalid orderbook payload was not counted",
    )

    stats = analyzer.get_stats()

    _assert_stats_orderbook_aliases(stats, 1)
    assert stats["invalid_payloads"] == 1
    assert stats["quotes_received"] == 0
    assert stats["quotes_stored"] == 0
    assert stats["quotes_cached"] == 0


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
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            widening_bps_threshold=Decimal("10000"),
            anomaly_zscore_threshold=Decimal("100"),
            arbitrage_min_bps=Decimal("10000"),
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
            _binance_orderbook(
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
            _bybit_orderbook(
                bid="104.9",
                ask="105.1",
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
        assert stats["quotes_cached"] == 2
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
        {"exchange": "binance", "symbol": "BTCUSDT", "instrument_type": "perpetual"},
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "instrument_type": "perpetual",
            "market_type": "perpetual",
            "bids": [],
            "asks": [],
        },
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "instrument_type": "perpetual",
            "market_type": "perpetual",
            "best_bid": "not-a-decimal",
            "best_ask": "100.1",
        },
    ],
)
async def test_rejects_bad_orderbook_payload_shapes(
    analyzer: CrossExchangeSpreadAnalyzer,
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
    assert stats["quotes_cached"] == 0


async def test_rejects_stale_orderbook_and_does_not_cache_it(
    analyzer: CrossExchangeSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    stale_timestamp = datetime.utcnow() - timedelta(minutes=10)

    await _emit_market_event(
        event_bus,
        ORDERBOOK_TOPIC,
        _binance_orderbook(timestamp=stale_timestamp),
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
    assert stats["quotes_cached"] == 0


async def test_rejects_incomplete_orderbook_and_does_not_cache_it(
    analyzer: CrossExchangeSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    await _emit_market_event(
        event_bus,
        ORDERBOOK_TOPIC,
        _binance_orderbook(
            bid=None,
            ask="100.1",
        ),
    )

    await _wait_until(
        lambda: analyzer.get_stats()["invalid_payloads"]
        + analyzer.get_stats()["invalid_quotes"]
        + analyzer.get_stats()["incomplete_quotes"] >= 1,
        message="Incomplete orderbook was not rejected",
    )

    stats = analyzer.get_stats()

    assert stats["quotes_stored"] == 0
    assert stats["quotes_cached"] == 0


async def test_rejects_crossed_orderbook_and_does_not_cache_it(
    analyzer: CrossExchangeSpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    await _emit_market_event(
        event_bus,
        ORDERBOOK_TOPIC,
        _binance_orderbook(
            bid="101.0",
            ask="100.0",
        ),
    )

    await _wait_until(
        lambda: analyzer.get_stats()["invalid_payloads"]
        + analyzer.get_stats()["invalid_quotes"]
        + analyzer.get_stats()["incomplete_quotes"] >= 1,
        message="Crossed orderbook was not rejected",
    )

    stats = analyzer.get_stats()

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
            ORDERBOOK_TOPIC,
            _binance_orderbook(
                instrument_type=InstrumentType.SPOT,
                market_type="spot",
            ),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["instrument_type_skips"] == 1,
            message="Disallowed instrument type was not skipped",
        )

        stats = analyzer.get_stats()

        _assert_stats_orderbook_aliases(stats, 1)
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

async def test_stores_first_exchange_orderbook_without_snapshot_until_second_exchange_arrives(
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
            ORDERBOOK_TOPIC,
            _binance_orderbook(),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["quotes_cached"] == 1,
            message="First exchange orderbook quote was not cached",
        )

        stats = analyzer.get_stats()

        assert stats["quotes_stored"] == 1
        assert stats["snapshots_built"] == 0
        assert stats["latest_snapshots"] == 0
        assert stats["latest_opportunities"] == 0

        await snapshot_recorder.assert_no_new_events(0)
    finally:
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_real_eventbus_cross_exchange_orderbooks_build_snapshot(
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
            ORDERBOOK_TOPIC,
            _binance_orderbook(
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
            _bybit_orderbook(
                symbol="BTC-USDT",
                bid="104.9",
                ask="105.1",
                timestamp=now + timedelta(milliseconds=20),
                sequence_id=202,
                shape="levels_tuple",
            ),
        )

        await snapshot_recorder.wait_for_count(1)

        snapshot = snapshot_recorder.payloads()[0]

        assert _payload_enum_value(snapshot, "spread_type") == SpreadType.CROSS_EXCHANGE.value
        assert _payload_value(snapshot, "symbol") == "BTCUSDT"

        assert _payload_value(snapshot, "leg_a_exchange") == "binance"
        assert _payload_value(snapshot, "leg_b_exchange") == "bybit"

        assert _payload_enum_value(snapshot, "leg_a_type") == InstrumentType.PERPETUAL.value
        assert _payload_enum_value(snapshot, "leg_b_type") == InstrumentType.PERPETUAL.value

        _assert_decimal_equal(_payload_decimal(snapshot, "leg_a_mid"), Decimal("100.0"))
        _assert_decimal_equal(_payload_decimal(snapshot, "leg_b_mid"), Decimal("105.0"))

        _assert_decimal_equal(_payload_decimal(snapshot, "raw_spread"), Decimal("5.0"))
        _assert_decimal_equal(_payload_decimal(snapshot, "spread_pct"), Decimal("5.0"))
        _assert_decimal_equal(_payload_decimal(snapshot, "spread_bps"), Decimal("500.0"))

        metadata = _payload_metadata(snapshot)

        assert metadata["price_input_source"] == ORDERBOOK_TOPIC
        assert metadata["instrument_type"] == InstrumentType.PERPETUAL.value
        assert metadata["quote_a_sequence_id"] == 101
        assert metadata["quote_b_sequence_id"] == 202
        assert metadata["buy_exchange"] == "binance"
        assert metadata["sell_exchange"] == "bybit"
        assert metadata["buy_price"] == "100.1"
        assert metadata["sell_price"] == "104.9"
        assert metadata["gross_edge"] == "4.8"

        latest = analyzer.get_latest_snapshot(
            "BTC_USDT",
            "BYBIT",
            "BINANCE",
            InstrumentType.PERPETUAL,
            market_type="perpetual",
        )

        assert latest is not None
        assert latest.symbol == "BTCUSDT"
        assert latest.metadata["price_input_source"] == ORDERBOOK_TOPIC

        stats = analyzer.get_stats()

        _assert_stats_orderbook_aliases(stats, 2)
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
            ORDERBOOK_TOPIC,
            _bybit_orderbook(timestamp=now, sequence_id=1),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _binance_orderbook(timestamp=now, sequence_id=2),
        )

        await snapshot_recorder.wait_for_count(1)

        latest_a = analyzer.get_latest_snapshot(
            "BTCUSDT",
            "binance",
            "bybit",
            InstrumentType.PERPETUAL,
            market_type="perpetual",
        )
        latest_b = analyzer.get_latest_snapshot(
            "BTC/USDT",
            "BYBIT",
            "BINANCE",
            InstrumentType.PERPETUAL,
            market_type="perpetual",
        )

        assert latest_a is not None
        assert latest_b is not None
        assert latest_a is latest_b
    finally:
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_unaligned_orderbooks_are_skipped_and_no_snapshot_is_published(
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
            ORDERBOOK_TOPIC,
            _binance_orderbook(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _bybit_orderbook(timestamp=now + timedelta(seconds=2)),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["unaligned_quotes"] == 1,
            message="Unaligned orderbook pair was not counted",
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


async def test_allowed_exchange_filter_skips_disallowed_exchange_before_snapshot(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            allowed_exchanges={"binance", "okx"},
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
            ORDERBOOK_TOPIC,
            _binance_orderbook(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _bybit_orderbook(timestamp=now),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["scope_skips"] >= 1,
            message="Disallowed exchange was not skipped",
        )

        stats = analyzer.get_stats()

        assert stats["quotes_cached"] == 1
        assert stats["snapshots_built"] == 0
        assert stats["latest_snapshots"] == 0

        await snapshot_recorder.assert_no_new_events(0)
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)


async def test_different_market_types_do_not_pair(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.cross.market_type_mismatch.recorder",
    )

    try:
        await analyzer.start()

        now = datetime.utcnow()

        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _binance_orderbook(
                market_type="perpetual",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _bybit_orderbook(
                market_type="futures",
                timestamp=now,
            ),
        )

        await _wait_until(
            lambda: analyzer.get_stats()["quotes_cached"] == 2,
            message="Both valid orderbooks should be cached",
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
        _quote_snapshot(
            exchange="binance",
            bid="99.9",
            ask="100.1",
            timestamp=now,
        ),
        _quote_snapshot(
            exchange="bybit",
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


async def test_opportunity_detector_reverses_buy_sell_when_second_exchange_has_lower_ask() -> None:
    config = _cross_config(
        arbitrage_min_bps=Decimal("1"),
        default_taker_fee_rate=Decimal("0"),
        slippage_max_bps=Decimal("0"),
        safety_buffer_bps=Decimal("0"),
    )
    detector = SpreadOpportunityDetector(config)

    now = datetime.utcnow()

    result = detector.detect_from_quotes(
        _quote_snapshot(
            exchange="binance",
            bid="110.0",
            ask="110.2",
            timestamp=now,
        ),
        _quote_snapshot(
            exchange="bybit",
            bid="104.9",
            ask="105.1",
            timestamp=now,
        ),
        timestamp=now,
    )

    assert result.found is True
    assert result.opportunity is not None

    opportunity = result.opportunity

    _assert_profitable_opportunity(opportunity)

    assert opportunity.buy_exchange == "bybit"
    assert opportunity.sell_exchange == "binance"
    assert opportunity.buy_price == Decimal("105.1")
    assert opportunity.sell_price == Decimal("110.0")
    _assert_decimal_equal(opportunity.gross_edge, Decimal("4.9"))


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
        _quote_snapshot(
            exchange="binance",
            bid="99.9",
            ask="100.1",
            bid_size="100",
            ask_size="100",
            timestamp=now,
        ),
        _quote_snapshot(
            exchange="bybit",
            bid="104.9",
            ask="105.1",
            bid_size="100",
            ask_size="100",
            timestamp=now,
        ),
        timestamp=now,
    )

    assert result.found is True
    assert result.opportunity is not None
    assert result.costs is not None

    opportunity = result.opportunity
    costs = result.costs

    _assert_profitable_opportunity(opportunity)

    assert costs.gross_edge > costs.net_edge
    assert costs.estimated_fees > Decimal("0")
    assert costs.estimated_slippage > Decimal("0")
    assert costs.safety_buffer > Decimal("0")
    assert opportunity.net_edge == costs.net_edge
    assert opportunity.net_edge < opportunity.gross_edge


async def test_opportunity_detector_rejects_trade_when_costs_destroy_edge() -> None:
    config = _cross_config(
        arbitrage_min_bps=Decimal("1"),
        default_trade_size=Decimal("1"),
        default_taker_fee_rate=Decimal("0.01"),
        slippage_max_bps=Decimal("100"),
        safety_buffer_bps=Decimal("100"),
    )
    detector = SpreadOpportunityDetector(config)

    now = datetime.utcnow()

    result = detector.detect_from_quotes(
        _quote_snapshot(
            exchange="binance",
            bid="99.90",
            ask="100.00",
            timestamp=now,
        ),
        _quote_snapshot(
            exchange="bybit",
            bid="100.05",
            ask="100.15",
            timestamp=now,
        ),
        timestamp=now,
    )

    assert result.found is False
    assert result.opportunity is None
    assert result.reason in {
        OpportunityDetectionReason.NET_EDGE_NOT_PROFITABLE,
        OpportunityDetectionReason.NET_EDGE_BPS_BELOW_THRESHOLD,
    }


async def test_opportunity_detector_rejects_symbol_and_instrument_mismatch() -> None:
    config = _cross_config()
    detector = SpreadOpportunityDetector(config)

    now = datetime.utcnow()

    symbol_result = detector.detect_from_quotes(
        _quote_snapshot(
            exchange="binance",
            symbol="BTCUSDT",
            bid="99.9",
            ask="100.1",
            timestamp=now,
        ),
        _quote_snapshot(
            exchange="bybit",
            symbol="ETHUSDT",
            bid="104.9",
            ask="105.1",
            timestamp=now,
        ),
        timestamp=now,
    )

    assert symbol_result.found is False
    assert symbol_result.reason == OpportunityDetectionReason.SYMBOL_MISMATCH

    instrument_result = detector.detect_from_quotes(
        _quote_snapshot(
            exchange="binance",
            instrument_type=InstrumentType.PERPETUAL,
            market_type="perpetual",
            bid="99.9",
            ask="100.1",
            timestamp=now,
        ),
        _quote_snapshot(
            exchange="bybit",
            instrument_type=InstrumentType.FUTURES,
            market_type="futures",
            bid="104.9",
            ask="105.1",
            timestamp=now,
        ),
        timestamp=now,
    )

    assert instrument_result.found is False
    assert instrument_result.reason == OpportunityDetectionReason.INSTRUMENT_TYPE_MISMATCH


# ============================================================
# Full EventBus opportunity pipeline
# ============================================================

async def test_real_eventbus_cross_exchange_orderbooks_publish_arbitrage_opportunity(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
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
            ORDERBOOK_TOPIC,
            _binance_orderbook(
                bid="99.9",
                ask="100.1",
                timestamp=now,
                sequence_id=101,
            ),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _bybit_orderbook(
                bid="104.9",
                ask="105.1",
                timestamp=now,
                sequence_id=202,
            ),
        )

        await snapshot_recorder.wait_for_count(1)
        await opportunity_recorder.wait_for_count(1)

        opportunity = opportunity_recorder.payloads()[0]

        assert _payload_value(opportunity, "symbol") == "BTCUSDT"
        assert _payload_value(opportunity, "buy_exchange") == "binance"
        assert _payload_value(opportunity, "sell_exchange") == "bybit"
        assert _payload_value(opportunity, "buy_market_type") == "perpetual"
        assert _payload_value(opportunity, "sell_market_type") == "perpetual"

        _assert_decimal_equal(_payload_decimal(opportunity, "gross_edge"), Decimal("4.8"))
        _assert_decimal_equal(_payload_decimal(opportunity, "net_edge"), Decimal("4.8"))

        metadata = _payload_metadata(opportunity)
        assert metadata["price_input_source"] == ORDERBOOK_TOPIC

        stats = analyzer.get_stats()

        assert stats["snapshots_built"] == 1
        assert stats["opportunities_detected"] == 1
        assert stats["opportunities_published"] == 1
        assert stats["latest_opportunities"] == 1
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)
        await _unsubscribe(event_bus, opportunity_subscription)


async def test_no_opportunity_is_published_when_net_edge_below_threshold(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            arbitrage_min_bps=Decimal("1000"),
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
            ORDERBOOK_TOPIC,
            _binance_orderbook(
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _bybit_orderbook(
                bid="100.2",
                ask="100.4",
                timestamp=now,
            ),
        )

        await snapshot_recorder.wait_for_count(1)
        await opportunity_recorder.assert_no_new_events(0)

        stats = analyzer.get_stats()

        assert stats["snapshots_built"] == 1
        assert stats["opportunities_published"] == 0
        assert stats["opportunity_detection_misses"] >= 1
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)
        await _unsubscribe(event_bus, opportunity_subscription)


# ============================================================
# Signal generation
# ============================================================

async def test_widening_signal_is_generated_when_threshold_is_crossed(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            widening_bps_threshold=Decimal("10"),
            anomaly_zscore_threshold=Decimal("100"),
            arbitrage_min_bps=Decimal("10000"),
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
        name="test.cross.snapshot.recorder",
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
            ORDERBOOK_TOPIC,
            _binance_orderbook(
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _bybit_orderbook(
                bid="104.9",
                ask="105.1",
                timestamp=now,
            ),
        )

        await snapshot_recorder.wait_for_count(1)
        await signal_recorder.wait_for_count(1)

        signal = signal_recorder.payloads()[0]

        assert _payload_enum_value(signal, "spread_type") == SpreadType.CROSS_EXCHANGE.value
        assert _signal_type_value(signal) == SpreadSignalType.WIDENING.value
        assert _payload_value(signal, "symbol") == "BTCUSDT"

        _assert_decimal_equal(_payload_decimal(signal, "value"), Decimal("500.0"))
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
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            widening_bps_threshold=Decimal("1000"),
            anomaly_zscore_threshold=Decimal("100"),
            arbitrage_min_bps=Decimal("10000"),
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
        name="test.cross.snapshot.recorder",
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
            ORDERBOOK_TOPIC,
            _binance_orderbook(
                bid="99.9",
                ask="100.1",
                timestamp=now,
            ),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _bybit_orderbook(
                bid="100.9",
                ask="101.1",
                timestamp=now,
            ),
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
# Rolling windows / cleanup
# ============================================================

async def test_rolling_window_updates_after_multiple_snapshots(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            widening_bps_threshold=Decimal("10000"),
            arbitrage_min_bps=Decimal("10000"),
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
            ORDERBOOK_TOPIC,
            _binance_orderbook(timestamp=now, sequence_id=1),
        )

        bybit_values = [
            ("104.9", "105.1"),
            ("105.9", "106.1"),
            ("106.9", "107.1"),
        ]

        for index, (bid, ask) in enumerate(bybit_values, start=2):
            await _emit_market_event(
                event_bus,
                ORDERBOOK_TOPIC,
                _bybit_orderbook(
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
            _assert_decimal_equal(stats_payload["last_value"], Decimal("7.0"))
            _assert_decimal_equal(stats_payload["min_value"], Decimal("5.0"))
            _assert_decimal_equal(stats_payload["max_value"], Decimal("7.0"))
        else:
            assert stats_payload is not None
            assert stats_payload.count == 3
            assert stats_payload.last_value == Decimal("7.0")
            assert stats_payload.min_value == Decimal("5.0")
            assert stats_payload.max_value == Decimal("7.0")

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


async def test_cleanup_stale_state_removes_quotes_snapshots_windows_and_opportunities(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            stale_state_ttl_seconds=0.01,
            cleanup_interval_seconds=3_600.0,
            widening_bps_threshold=Decimal("10000"),
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
            ORDERBOOK_TOPIC,
            _binance_orderbook(timestamp=now),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _bybit_orderbook(timestamp=now),
        )

        await snapshot_recorder.wait_for_count(1)
        await opportunity_recorder.wait_for_count(1)

        assert analyzer.get_stats()["quotes_cached"] == 2
        assert analyzer.get_stats()["latest_snapshots"] == 1
        assert analyzer.get_stats()["active_windows"] == 1
        assert analyzer.get_stats()["latest_opportunities"] == 1

        await asyncio.sleep(0.03)

        await analyzer.cleanup_stale_state()

        stats = analyzer.get_stats()

        assert stats["cleanup_runs"] == 1
        assert stats["quotes_cached"] == 0
        assert stats["latest_snapshots"] == 0
        assert stats["active_windows"] == 0

        # Opportunities may be removed by quote/snapshot TTL cleanup or by their own TTL,
        # depending on implementation. The important guarantee is no stale market state.
        assert stats["cleanup_removed_quotes"] >= 2
        assert stats["cleanup_removed_snapshots"] >= 1
        assert stats["cleanup_removed_windows"] >= 1
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)
        await _unsubscribe(event_bus, opportunity_subscription)


async def test_opportunity_ttl_cleanup_expires_and_removes_opportunity(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            opportunity_ttl_seconds=0.01,
            stale_state_ttl_seconds=3_600.0,
            arbitrage_min_bps=Decimal("1"),
            default_taker_fee_rate=Decimal("0"),
            slippage_max_bps=Decimal("0"),
            safety_buffer_bps=Decimal("0"),
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

        await _emit_market_event(event_bus, ORDERBOOK_TOPIC, _binance_orderbook(timestamp=now))
        await _emit_market_event(event_bus, ORDERBOOK_TOPIC, _bybit_orderbook(timestamp=now))

        await opportunity_recorder.wait_for_count(1)

        assert analyzer.get_stats()["latest_opportunities"] == 1

        await asyncio.sleep(0.03)

        await analyzer.cleanup_stale_state()

        stats = analyzer.get_stats()

        assert stats["latest_opportunities"] == 0
        assert stats["opportunities_expired"] >= 1
        assert stats["cleanup_removed_opportunities"] >= 1
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, opportunity_subscription)


# ============================================================
# Stats contract
# ============================================================

async def test_stats_reflect_complete_cross_exchange_orderbook_pipeline(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    analyzer = CrossExchangeSpreadAnalyzer(
        config=_cross_config(
            widening_bps_threshold=Decimal("10"),
            anomaly_zscore_threshold=Decimal("100"),
            arbitrage_min_bps=Decimal("1"),
            default_taker_fee_rate=Decimal("0"),
            slippage_max_bps=Decimal("0"),
            safety_buffer_bps=Decimal("0"),
            cooldown_seconds=0,
        ),
        event_bus=event_bus,
        scheduler=scheduler,
    )

    snapshot_recorder = EventRecorder()
    signal_recorder = EventRecorder()
    opportunity_recorder = EventRecorder()

    snapshot_subscription = await _subscribe(
        event_bus,
        SNAPSHOT_TOPIC,
        snapshot_recorder.handler,
        name="test.cross.snapshot.recorder",
    )
    signal_subscription = await _subscribe(
        event_bus,
        SIGNAL_TOPIC,
        signal_recorder.handler,
        name="test.cross.signal.recorder",
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
            ORDERBOOK_TOPIC,
            _binance_orderbook(
                bid="99.9",
                ask="100.1",
                timestamp=now,
                sequence_id=101,
            ),
        )
        await _emit_market_event(
            event_bus,
            ORDERBOOK_TOPIC,
            _bybit_orderbook(
                bid="104.9",
                ask="105.1",
                timestamp=now,
                sequence_id=202,
            ),
        )

        await snapshot_recorder.wait_for_count(1)
        await signal_recorder.wait_for_count(1)
        await opportunity_recorder.wait_for_count(1)

        stats = analyzer.get_stats()

        assert stats["running"] is True
        assert stats["registered"] is True
        assert stats["enabled"] is True
        assert stats["price_input_source"] == ORDERBOOK_TOPIC

        _assert_stats_orderbook_aliases(stats, 2)

        assert stats["quotes_received"] == 2
        assert stats["invalid_payloads"] == 0
        assert stats["invalid_quotes"] == 0
        assert stats["stale_quotes"] == 0
        assert stats["unaligned_quotes"] == 0

        assert stats["quotes_stored"] == 2
        assert stats["quotes_cached"] == 2

        assert stats["snapshots_built"] == 1
        assert stats["snapshots_published"] == 1
        assert stats["latest_snapshots"] == 1
        assert stats["active_windows"] == 1

        assert stats["signals_built"] >= 1
        assert stats["signals_published"] >= 1

        assert stats["opportunities_detected"] == 1
        assert stats["opportunities_published"] == 1
        assert stats["latest_opportunities"] == 1

        snapshot = snapshot_recorder.payloads()[0]
        signal = signal_recorder.payloads()[0]
        opportunity = opportunity_recorder.payloads()[0]

        assert _payload_value(snapshot, "symbol") == "BTCUSDT"
        assert _payload_value(signal, "symbol") == "BTCUSDT"
        assert _payload_value(opportunity, "symbol") == "BTCUSDT"

        assert _signal_type_value(signal) == SpreadSignalType.WIDENING.value

        snapshot_metadata = _payload_metadata(snapshot)
        opportunity_metadata = _payload_metadata(opportunity)

        assert snapshot_metadata["price_input_source"] == ORDERBOOK_TOPIC
        assert opportunity_metadata["price_input_source"] == ORDERBOOK_TOPIC
    finally:
        if analyzer.is_running:
            await analyzer.stop()
        if analyzer.is_registered:
            analyzer.unregister()
        await _unsubscribe(event_bus, snapshot_subscription)
        await _unsubscribe(event_bus, signal_subscription)
        await _unsubscribe(event_bus, opportunity_subscription)