"""
Integration tests for the market-data event chain.

These tests verify that the runtime link:

    exchange/raw market events -> EventBus -> data caches -> normalized market.*.updated events

actually publishes the downstream topics expected by analytics.

Run:
    pytest -q tests/test_market_data_event_flow.py
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio

from core.config import Config
from core.event_bus import Event, EventBus, EventPriority, HandlerDispatchMode, QueueFullPolicy
from data.candles_cache import CandlesCache
from data.funding_cache import FundingCache
from data.market_stream import MarketStream, MarketStreamConfig
from data.open_interest_cache import OpenInterestCache
from data.orderbook_cache import OrderBookCache
from data.trades_cache import TradesCache


SYMBOL = "BTCUSDT"
EXCHANGE = "binance"
MARKET_TYPE = "usdm_futures"
TIMEFRAME = "1m"


@dataclass(slots=True)
class EventCollector:
    events: list[Event]

    async def __call__(self, event: Event) -> None:
        self.events.append(event)

    def topics(self) -> list[str]:
        return [event.topic for event in self.events]

    def payloads(self, topic: str) -> list[dict[str, Any]]:
        return [event.payload for event in self.events if event.topic == topic and isinstance(event.payload, dict)]

    def count(self, topic: str) -> int:
        return self.topics().count(topic)


class FakeExchangeClient:
    """A deterministic exchange adapter that publishes one event for every cache path."""

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.registered = False
        self.started = False
        self.stopped = False

    def register(self) -> None:
        self.registered = True

    async def start(self) -> None:
        self.started = True
        now_ms = int(time.time() * 1000)
        open_time_ms = now_ms - 60_000
        close_time_ms = now_ms

        await self.event_bus.emit(
            "market.trade",
            {
                "exchange": EXCHANGE,
                "symbol": SYMBOL,
                "market_type": MARKET_TYPE,
                "timestamp_ms": now_ms,
                "received_at_ms": now_ms,
                "trade_id": "t-1",
                "price": 100.5,
                "quantity": 0.25,
                "side": "buy",
                "aggressor_side": "buy",
            },
            source="fake_exchange",
            priority=EventPriority.LOW,
        )

        await self.event_bus.emit(
            "market.candle",
            {
                "exchange": EXCHANGE,
                "symbol": SYMBOL,
                "market_type": MARKET_TYPE,
                "timeframe": TIMEFRAME,
                "timestamp_ms": close_time_ms,
                "received_at_ms": now_ms,
                "open_time_ms": open_time_ms,
                "close_time_ms": close_time_ms,
                "open": 99.0,
                "high": 101.0,
                "low": 98.5,
                "close": 100.5,
                "volume": 1234.0,
                "is_closed": True,
            },
            source="fake_exchange",
            priority=EventPriority.NORMAL,
        )

        await self.event_bus.emit(
            "market.orderbook.snapshot",
            {
                "exchange": EXCHANGE,
                "symbol": SYMBOL,
                "market_type": MARKET_TYPE,
                "timestamp_ms": now_ms,
                "received_at_ms": now_ms,
                "sequence": 100,
                "bids": [[100.4, 1.0], [100.3, 2.0]],
                "asks": [[100.6, 1.5], [100.7, 2.5]],
            },
            source="fake_exchange",
            priority=EventPriority.LOW,
        )

        await self.event_bus.emit(
            "market.funding",
            {
                "exchange": EXCHANGE,
                "symbol": SYMBOL,
                "market_type": MARKET_TYPE,
                "timestamp_ms": now_ms,
                "received_at_ms": now_ms,
                "funding_rate": 0.0001,
                "next_funding_time_ms": now_ms + 8 * 60 * 60 * 1000,
                "mark_price": 100.5,
                "index_price": 100.45,
            },
            source="fake_exchange",
            priority=EventPriority.NORMAL,
        )

        await self.event_bus.emit(
            "market.open_interest",
            {
                "exchange": EXCHANGE,
                "symbol": SYMBOL,
                "market_type": MARKET_TYPE,
                "timestamp_ms": now_ms,
                "received_at_ms": now_ms,
                "open_interest": 123456.0,
                "open_interest_value": 12_345_600.0,
                "mark_price": 100.5,
            },
            source="fake_exchange",
            priority=EventPriority.NORMAL,
        )

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def config() -> Config:
    return Config()


@pytest_asyncio.fixture
async def event_bus() -> EventBus:
    bus = EventBus(
        max_queue_size=10_000,
        worker_count=4,
        queue_full_policy=QueueFullPolicy.DROP_OLDEST,
        max_retries=0,
        handler_dispatch_mode=HandlerDispatchMode.CONCURRENT,
        handler_timeout=2.0,
    )
    await bus.start()
    try:
        yield bus
    finally:
        await bus.stop(drain=True, timeout=5.0)


async def wait_for_topics(
    collector: EventCollector,
    required_topics: set[str],
    *,
    timeout: float = 3.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        seen = set(collector.topics())
        if required_topics <= seen:
            return
        await asyncio.sleep(0.02)

    counts = Counter(collector.topics())
    missing = sorted(required_topics - set(collector.topics()))
    raise AssertionError(f"Missing topics: {missing}; seen counts: {dict(counts)}")


def build_caches(config: Config, event_bus: EventBus) -> list[Any]:
    return [
        TradesCache(
            config=config,
            event_bus=event_bus,
            scheduler=None,
            # Disable throttling so this test is deterministic and does not wait 500ms.
            trades_updated_emit_min_interval_ms=0,
        ),
        CandlesCache(config=config, event_bus=event_bus, scheduler=None),
        OrderBookCache(config=config, event_bus=event_bus, scheduler=None),
        FundingCache(config=config, event_bus=event_bus, scheduler=None),
        OpenInterestCache(config=config, event_bus=event_bus, scheduler=None),
    ]


@pytest.mark.asyncio
async def test_raw_market_events_are_normalized_into_analytics_input_topics(
    config: Config,
    event_bus: EventBus,
) -> None:
    """Directly verifies cache subscriptions and downstream market.*.updated topics."""
    collector = EventCollector([])
    event_bus.subscribe("*", collector, name="test-event-collector")

    caches = build_caches(config, event_bus)
    for cache in caches:
        cache.register()

    fake_exchange = FakeExchangeClient(event_bus)
    await fake_exchange.start()

    required_topics = {
        # Raw exchange/input topics.
        "market.trade",
        "market.candle",
        "market.orderbook.snapshot",
        "market.funding",
        "market.open_interest",
        # Cache-normalized topics consumed by analytics.
        "market.trades.updated",
        "market.candles.updated",
        "market.candle.closed",
        "market.orderbook.updated",
        "market.funding.updated",
        "market.open_interest.updated",
    }
    await wait_for_topics(collector, required_topics)

    # Payload contract checks: analytics must receive enough scope + price data.
    trades_payload = collector.payloads("market.trades.updated")[-1]
    assert trades_payload["exchange"] == EXCHANGE
    assert trades_payload["symbol"] == SYMBOL
    assert trades_payload["market_type"] == MARKET_TYPE
    # Current TradesCache publishes price inside the nested trade object.
    # If you later require top-level last_price/current_price, change this assertion
    # to enforce that contract explicitly.
    assert trades_payload["trade"]["price"] == pytest.approx(100.5)

    candle_payload = collector.payloads("market.candles.updated")[-1]
    assert candle_payload["exchange"] == EXCHANGE
    assert candle_payload["symbol"] == SYMBOL
    assert candle_payload["timeframe"] == TIMEFRAME
    assert candle_payload["close"] == pytest.approx(100.5)
    assert candle_payload["is_closed"] is True

    orderbook_payload = collector.payloads("market.orderbook.updated")[-1]
    assert orderbook_payload["exchange"] == EXCHANGE
    assert orderbook_payload["symbol"] == SYMBOL
    assert orderbook_payload["best_bid"][0] == pytest.approx(100.4)
    assert orderbook_payload["best_ask"][0] == pytest.approx(100.6)
    assert orderbook_payload["mid_price"] == pytest.approx(100.5)

    funding_payload = collector.payloads("market.funding.updated")[-1]
    assert funding_payload["funding_rate"] == pytest.approx(0.0001)
    assert funding_payload["mark_price"] == pytest.approx(100.5)

    oi_payload = collector.payloads("market.open_interest.updated")[-1]
    assert oi_payload["open_interest"] == pytest.approx(123456.0)
    assert oi_payload["mark_price"] == pytest.approx(100.5)


@pytest.mark.asyncio
async def test_market_stream_start_registers_caches_starts_exchange_and_publishes_lifecycle(
    config: Config,
    event_bus: EventBus,
) -> None:
    """
    Full MarketStream-level test.

    It catches startup/lifecycle mismatches where the exchange client emits raw events,
    but MarketStream incorrectly marks the client as failed or does not register caches.
    """
    collector = EventCollector([])
    event_bus.subscribe("*", collector, name="test-event-collector")

    fake_exchange = FakeExchangeClient(event_bus)
    market_stream = MarketStream(
        config=config,
        event_bus=event_bus,
        exchange_clients={EXCHANGE: fake_exchange},
        scheduler=None,
        stream_config=MarketStreamConfig(
            start_clients_on_start=True,
            register_caches_on_start=True,
            emit_lifecycle_events=True,
            startup_timeout_seconds=2.0,
        ),
        caches=build_caches(config, event_bus),
    )

    await market_stream.start()

    required_topics = {
        "system.market_stream.started",
        "system.market_stream.exchange_started",
        "market.trades.updated",
        "market.candles.updated",
        "market.candle.closed",
        "market.orderbook.updated",
        "market.funding.updated",
        "market.open_interest.updated",
    }
    await wait_for_topics(collector, required_topics)

    assert fake_exchange.registered is True
    assert fake_exchange.started is True
    assert market_stream.stats()["exchanges_started"] == 1
    assert collector.count("system.market_stream.exchange_start_failed") == 0

    await market_stream.stop()
    await wait_for_topics(collector, {"system.market_stream.stopped"})
    assert fake_exchange.stopped is True