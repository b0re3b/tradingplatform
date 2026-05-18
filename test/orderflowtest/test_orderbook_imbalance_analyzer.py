# tests/analytics/orderflow/test_orderbook_imbalance_analyzer.py

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pytest

from core.event_bus import Event, EventPriority

from analytics.orderflow.config import OrderbookImbalanceConfig
from analytics.orderflow.enums import OrderFlowEventTopic
from analytics.orderflow.models import (
    OrderFlowKey,
    OrderbookSnapshot,
    make_orderflow_key,
    orderflow_key_to_dict,
    orderflow_key_to_string,
)
from analytics.orderflow.orderbook_imbalance import OrderbookImbalanceAnalyzer


# =============================================================================
# Constants
# =============================================================================

DEFAULT_KEY: OrderFlowKey = make_orderflow_key(
    exchange="binance",
    market_type="usdm_futures",
    symbol="BTCUSDT",
    timeframe="1m",
)

BYBIT_KEY: OrderFlowKey = make_orderflow_key(
    exchange="bybit",
    market_type="linear",
    symbol="BTCUSDT",
    timeframe="1m",
)

ETH_KEY: OrderFlowKey = make_orderflow_key(
    exchange="binance",
    market_type="usdm_futures",
    symbol="ETHUSDT",
    timeframe="1m",
)

HIGHER_TF_KEY: OrderFlowKey = make_orderflow_key(
    exchange="binance",
    market_type="usdm_futures",
    symbol="BTCUSDT",
    timeframe="5m",
)

ORDERBOOK_TOPIC = "market.orderbook.updated"


# =============================================================================
# Fakes
# =============================================================================


@dataclass(slots=True)
class FakeSubscription:
    pattern: str
    handler: Any
    name: str
    enabled: bool = True


class FakeEventBus:
    """
    Strict EventBus fake.

    It records subscriptions and emitted events and can simulate temporary
    EventBus failures. It intentionally keeps the same surface used by
    BaseOrderFlowAnalyzer: subscribe(), unsubscribe(), emit().
    """

    def __init__(self) -> None:
        self.subscriptions: list[FakeSubscription] = []
        self.unsubscribed: list[FakeSubscription] = []
        self.emitted: list[dict[str, Any]] = []
        self.fail_emit_topics: set[str] = set()

    def subscribe(
        self,
        pattern: str,
        handler: Any,
        *,
        name: str | None = None,
    ) -> FakeSubscription:
        subscription = FakeSubscription(
            pattern=pattern,
            handler=handler,
            name=name or getattr(handler, "__name__", "anonymous_handler"),
        )
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: FakeSubscription) -> None:
        if subscription not in self.subscriptions:
            raise RuntimeError(f"Unknown subscription: {subscription!r}")

        self.subscriptions.remove(subscription)
        self.unsubscribed.append(subscription)

    async def emit(
        self,
        topic: str,
        payload: Any,
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str | None = None,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        if topic in self.fail_emit_topics:
            raise RuntimeError(f"Simulated EventBus failure for topic={topic}")

        self.emitted.append(
            {
                "topic": topic,
                "payload": payload,
                "priority": priority,
                "source": source,
                "correlation_id": correlation_id,
                "headers": headers or {},
            }
        )
        return True


class FakeScheduler:
    """
    Strict Scheduler fake.

    This catches analyzers that start uncontrolled asyncio loops instead of
    registering health/cleanup jobs through Scheduler.add_interval_job().
    """

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.disabled_job_ids: list[str] = []
        self.removed_job_ids: list[str] = []

    def get_job_by_name(self, name: str) -> dict[str, Any] | None:
        for job in self.jobs.values():
            if job["name"] == name:
                return job
        return None

    def add_interval_job(
        self,
        name: str,
        func: Any,
        *,
        interval: float,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        run_immediately: bool = False,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        timeout: float | None = None,
        allow_overlap: bool = False,
        enabled: bool = True,
    ) -> str:
        if interval <= 0:
            raise ValueError("interval must be > 0")

        job_id = f"job-{len(self.jobs) + 1}"
        self.jobs[job_id] = {
            "job_id": job_id,
            "name": name,
            "func": func,
            "interval": interval,
            "args": args,
            "kwargs": kwargs or {},
            "run_immediately": run_immediately,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "timeout": timeout,
            "allow_overlap": allow_overlap,
            "enabled": enabled,
        }
        return job_id

    def disable_job(self, job_id: str) -> None:
        if job_id not in self.jobs:
            raise KeyError(job_id)

        self.jobs[job_id]["enabled"] = False
        self.disabled_job_ids.append(job_id)

    def remove_job(self, job_id: str) -> None:
        if job_id not in self.jobs:
            raise KeyError(job_id)

        self.removed_job_ids.append(job_id)
        self.jobs.pop(job_id)


class StrictOrderbookCache:
    """
    Scoped data-layer cache fake.

    Production contract tested here:
        get_snapshot(
            exchange=...,
            market_type=...,
            symbol=...,
            timeframe=...,
        )

    The fake can return snapshots for multiple futures scopes and records calls
    to catch symbol-only or unscoped cache access.
    """

    def __init__(
        self,
        snapshots: dict[OrderFlowKey, Any] | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.snapshots: dict[OrderFlowKey, Any] = dict(snapshots or {})
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def set_snapshot(self, key: OrderFlowKey, snapshot: Any) -> None:
        self.snapshots[key] = snapshot

    def get_snapshot(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str | None = None,
    ) -> Any:
        call = {
            "method": "get_snapshot",
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "timeframe": timeframe,
        }
        self.calls.append(call)

        if self.fail:
            raise RuntimeError("Simulated orderbook cache failure")

        key = make_orderflow_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe or "1m",
        )
        return self.snapshots.get(key)


class StrictGetBookOrderbookCache:
    """
    Data cache fake exposing only get_book().

    This verifies the analyzer can use the canonical orderbook cache style
    without relying on symbol-only legacy fallbacks.
    """

    def __init__(self, snapshots: dict[OrderFlowKey, Any]) -> None:
        self.snapshots = dict(snapshots)
        self.calls: list[dict[str, Any]] = []

    def get_book(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
    ) -> Any:
        self.calls.append(
            {
                "method": "get_book",
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
            }
        )

        matching = [
            snapshot
            for key, snapshot in self.snapshots.items()
            if key[0] == exchange and key[1] == market_type and key[2] == symbol
        ]
        return matching[0] if matching else None


class AsyncOrderbookCache:
    """
    Async cache fake.

    Protects the analyzer from assuming sync-only data cache methods.
    """

    def __init__(self, snapshots: dict[OrderFlowKey, Any]) -> None:
        self.snapshots = dict(snapshots)
        self.calls: list[dict[str, Any]] = []

    async def get_snapshot(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str | None = None,
    ) -> Any:
        await asyncio.sleep(0)
        self.calls.append(
            {
                "method": "get_snapshot",
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
            }
        )

        key = make_orderflow_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe or "1m",
        )
        return self.snapshots.get(key)


class LegacySymbolOnlyOrderbookCache:
    """
    Compatibility fake exposing only symbol-only get().

    This is not the production contract. It exists only to keep the migration
    wrapper behavior explicitly tested.
    """

    def __init__(self, snapshot: Any) -> None:
        self.snapshot = snapshot
        self.calls: list[tuple[str, str]] = []

    def get(self, symbol: str) -> Any:
        self.calls.append(("get", symbol))
        return self.snapshot


# =============================================================================
# Factories
# =============================================================================


def now_ts(offset: float = 0.0) -> float:
    return time.time() + offset


def level(price: Any, size: Any) -> list[Any]:
    return [price, size]


def key(
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
) -> OrderFlowKey:
    return make_orderflow_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


def scope_payload(orderflow_key: OrderFlowKey = DEFAULT_KEY) -> dict[str, str]:
    return orderflow_key_to_dict(orderflow_key)


def snapshot_dict(
    *,
    orderflow_key: OrderFlowKey = DEFAULT_KEY,
    bids: list[Any] | None = None,
    asks: list[Any] | None = None,
    ts: float | None = None,
    sequence_id: str = "seq-1",
) -> dict[str, Any]:
    scope = orderflow_key_to_dict(orderflow_key)
    return {
        **scope,
        "exchange_symbol": scope["symbol"],
        "bids": bids if bids is not None else [],
        "asks": asks if asks is not None else [],
        "timestamp": now_ts() if ts is None else ts,
        "sequence_id": sequence_id,
    }


def bullish_snapshot(
    *,
    orderflow_key: OrderFlowKey = DEFAULT_KEY,
    ts: float | None = None,
) -> dict[str, Any]:
    return snapshot_dict(
        orderflow_key=orderflow_key,
        ts=ts,
        bids=[
            level(100.0, 50.0),
            level(99.5, 30.0),
            level(99.0, 20.0),
        ],
        asks=[
            level(100.5, 10.0),
            level(101.0, 10.0),
            level(101.5, 5.0),
        ],
    )


def bearish_snapshot(
    *,
    orderflow_key: OrderFlowKey = DEFAULT_KEY,
    ts: float | None = None,
) -> dict[str, Any]:
    return snapshot_dict(
        orderflow_key=orderflow_key,
        ts=ts,
        bids=[
            level(100.0, 5.0),
            level(99.5, 10.0),
            level(99.0, 10.0),
        ],
        asks=[
            level(100.5, 40.0),
            level(101.0, 30.0),
            level(101.5, 30.0),
        ],
    )


def neutral_snapshot(
    *,
    orderflow_key: OrderFlowKey = DEFAULT_KEY,
    ts: float | None = None,
) -> dict[str, Any]:
    return snapshot_dict(
        orderflow_key=orderflow_key,
        ts=ts,
        bids=[
            level(100.0, 25.0),
            level(99.5, 25.0),
        ],
        asks=[
            level(100.5, 25.0),
            level(101.0, 25.0),
        ],
    )


def malformed_snapshots(orderflow_key: OrderFlowKey = DEFAULT_KEY) -> list[Any]:
    scope = orderflow_key_to_dict(orderflow_key)
    return [
        None,
        "bad-snapshot",
        {},
        {**scope},
        {**scope, "bids": [], "asks": []},
        {**scope, "bids": [[100, 1]], "asks": []},
        {**scope, "bids": [], "asks": [[101, 1]]},
        {**scope, "bids": [[0, 1]], "asks": [[101, 1]]},
        {**scope, "bids": [[100, -1]], "asks": [[101, 1]]},
        {**scope, "bids": [["nan", 1]], "asks": [[101, 1]]},
        {**scope, "bids": [[100, float("inf")]], "asks": [[101, 1]]},
        {**scope, "bids": [[100, 1]], "asks": [["bad", 1]]},
        {**scope, "bids": [[100, 1]], "asks": [[101, 0]], "timestamp": -1},
    ]


def make_config(**overrides: Any) -> OrderbookImbalanceConfig:
    values = {
        "enabled": True,
        "emit_updates": True,
        "emit_signals": True,
        "min_signal_interval_sec": 3600.0,
        "health_log_interval_sec": 5.0,
        "cleanup_interval_sec": 5.0,
        "scheduler_job_timeout_sec": 0.5,
        "scheduler_job_retry_delay_sec": 0.1,
        "scheduler_job_max_retries": 1,
        "publish_priority": EventPriority.HIGH,
        "allowed_market_types": {"usdm_futures", "linear", "swap"},
        "depth_levels": 3,
        "min_total_volume": 0.0,
        "bullish_ratio_threshold": 0.60,
        "bearish_ratio_threshold": 0.40,
        "normalize_ratio_to_minus_one_one": False,
        "smooth_window": 1,
    }
    values.update(overrides)

    config = OrderbookImbalanceConfig(**values)
    config.validate()
    return config


def make_analyzer(
    snapshot: Any | None = None,
    *,
    orderflow_key: OrderFlowKey = DEFAULT_KEY,
    event_bus: FakeEventBus | None = None,
    cache: Any | None = None,
    config: OrderbookImbalanceConfig | None = None,
    scheduler: FakeScheduler | None = None,
    default_exchange: str = "binance",
    default_market_type: str = "usdm_futures",
    default_timeframe: str = "1m",
) -> tuple[OrderbookImbalanceAnalyzer, FakeEventBus, Any, FakeScheduler]:
    bus = event_bus or FakeEventBus()
    scheduler_obj = scheduler or FakeScheduler()

    if cache is None:
        cache = StrictOrderbookCache({orderflow_key: snapshot})

    analyzer = OrderbookImbalanceAnalyzer(
        event_bus=bus,  # type: ignore[arg-type]
        scheduler=scheduler_obj,  # type: ignore[arg-type]
        orderbook_cache=cache,
        config=config or make_config(),
        source_topic_patterns=(ORDERBOOK_TOPIC,),
        default_exchange=default_exchange,
        default_market_type=default_market_type,
        default_timeframe=default_timeframe,
    )
    return analyzer, bus, cache, scheduler_obj


# =============================================================================
# Assertions
# =============================================================================


def emitted_events(event_bus: FakeEventBus, topic: str) -> list[dict[str, Any]]:
    return [item for item in event_bus.emitted if item["topic"] == topic]


def assert_scope_payload(payload: dict[str, Any], expected_key: OrderFlowKey) -> None:
    scope = orderflow_key_to_dict(expected_key)

    assert payload["exchange"] == scope["exchange"]
    assert payload["market_type"] == scope["market_type"]
    assert payload["symbol"] == scope["symbol"]
    assert payload["timeframe"] == scope["timeframe"]
    assert payload["scope"] == scope
    assert payload["scope_key"] == orderflow_key_to_string(expected_key)
    assert payload["key"] == list(expected_key)
    assert tuple(payload["orderflow_key"]) == expected_key

def assert_update_emitted(
    event_bus: FakeEventBus,
    *,
    expected_key: OrderFlowKey = DEFAULT_KEY,
) -> dict[str, Any]:
    events = emitted_events(
        event_bus,
        OrderFlowEventTopic.ORDERBOOK_IMBALANCE_UPDATED.value,
    )
    assert len(events) >= 1

    emitted = events[-1]
    payload = emitted["payload"]

    assert emitted["source"] == "orderbook_imbalance"
    assert emitted["priority"] == EventPriority.HIGH

    assert payload["metric"] == "orderbook_imbalance"
    assert payload["source_type"] == "orderbook"
    assert_scope_payload(payload, expected_key)

    stats = payload["stats"]
    assert stats["metric"] == "orderbook_imbalance"
    assert stats["source_type"] == "orderbook"
    assert_scope_payload(stats, expected_key)

    return payload


def assert_signal_emitted(
    event_bus: FakeEventBus,
    *,
    expected_key: OrderFlowKey = DEFAULT_KEY,
    signal_type: str,
    side: str,
    reason: str,
) -> dict[str, Any]:
    events = emitted_events(
        event_bus,
        OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL.value,
    )
    assert len(events) >= 1

    payload = events[-1]["payload"]

    assert payload["metric"] == "orderbook_imbalance"
    assert payload["source_type"] == "orderbook"
    assert payload["signal_type"] == signal_type
    assert payload["side"] == side
    assert payload["reason"] == reason
    assert 0.0 <= payload["strength"] <= 1.0
    assert_scope_payload(payload, expected_key)

    context = payload["context"]
    assert context["scope"] == orderflow_key_to_dict(expected_key)
    assert context["scope_key"] == orderflow_key_to_string(expected_key)
    assert context["key"] == list(expected_key)
    assert "stats" in context
    assert "imbalance_ratio" in context
    assert "imbalance_diff" in context
    assert "bid_volume" in context
    assert "ask_volume" in context
    assert "depth_levels_used" in context

    return payload


# =============================================================================
# Lifecycle / topic contract tests
# =============================================================================


def test_register_subscribes_only_to_data_layer_orderbook_updated_topic_and_schedules_jobs() -> None:
    analyzer, event_bus, _, scheduler = make_analyzer()

    analyzer.register()

    assert analyzer.is_running is True
    assert [subscription.pattern for subscription in event_bus.subscriptions] == [
        ORDERBOOK_TOPIC
    ]
    assert all(
        "OrderbookImbalanceAnalyzer" in subscription.name
        for subscription in event_bus.subscriptions
    )

    assert len(scheduler.jobs) == 2
    job_names = {job["name"] for job in scheduler.jobs.values()}
    assert "analytics.orderflow.orderbook_imbalance.health" in job_names
    assert "analytics.orderflow.orderbook_imbalance.cleanup" in job_names

    for job in scheduler.jobs.values():
        assert job["enabled"] is True
        assert job["allow_overlap"] is False
        assert job["max_retries"] == 1
        assert job["retry_delay"] == pytest.approx(0.1)
        assert job["timeout"] == pytest.approx(0.5)


def test_register_rejects_raw_market_orderbook_topic_by_default() -> None:
    event_bus = FakeEventBus()
    cache = StrictOrderbookCache({DEFAULT_KEY: bullish_snapshot()})

    analyzer = OrderbookImbalanceAnalyzer(
        event_bus=event_bus,  # type: ignore[arg-type]
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        orderbook_cache=cache,
        config=make_config(),
        source_topic_patterns=("market.orderbook",),
        default_exchange="binance",
        default_market_type="usdm_futures",
        default_timeframe="1m",
    )

    with pytest.raises(ValueError, match="Raw market topic"):
        analyzer.register()

    assert event_bus.subscriptions == []


def test_register_is_idempotent_and_does_not_duplicate_jobs_or_subscriptions() -> None:
    analyzer, event_bus, _, scheduler = make_analyzer()

    analyzer.register()
    analyzer.register()

    assert len(event_bus.subscriptions) == 1
    assert len(scheduler.jobs) == 2
    assert analyzer.is_running is True


def test_stop_unsubscribes_and_removes_scheduler_jobs() -> None:
    analyzer, event_bus, _, scheduler = make_analyzer()
    analyzer.register()

    created_subscriptions = list(event_bus.subscriptions)
    created_job_ids = set(scheduler.jobs)

    analyzer.stop()

    assert analyzer.is_running is False
    assert event_bus.subscriptions == []
    assert event_bus.unsubscribed == created_subscriptions
    assert set(scheduler.removed_job_ids) == created_job_ids
    assert scheduler.disabled_job_ids == []
    assert scheduler.jobs == {}


# =============================================================================
# Core calculation tests
# =============================================================================


@pytest.mark.asyncio
async def test_process_key_calculates_bid_dominant_imbalance_and_bullish_signal() -> None:
    analyzer, event_bus, cache, _ = make_analyzer(
        bullish_snapshot(),
        config=make_config(
            depth_levels=3,
            bullish_ratio_threshold=0.60,
            bearish_ratio_threshold=0.40,
            smooth_window=1,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.key == DEFAULT_KEY
    assert stats.bid_volume == pytest.approx(100.0)
    assert stats.ask_volume == pytest.approx(25.0)
    assert stats.imbalance_ratio == pytest.approx(100.0 / 125.0)
    assert stats.imbalance_diff == pytest.approx((100.0 - 25.0) / 125.0)
    assert stats.best_bid == pytest.approx(100.0)
    assert stats.best_ask == pytest.approx(100.5)
    assert stats.spread == pytest.approx(0.5)
    assert stats.mid_price == pytest.approx(100.25)
    assert stats.depth_levels_used == 3

    assert cache.calls[-1] == {
        "method": "get_snapshot",
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "1m",
    }

    assert analyzer.get_latest_stats_by_key(DEFAULT_KEY) == stats
    assert_update_emitted(event_bus, expected_key=DEFAULT_KEY)
    assert_signal_emitted(
        event_bus,
        expected_key=DEFAULT_KEY,
        signal_type="bullish",
        side="buy",
        reason="orderbook_bid_imbalance",
    )


@pytest.mark.asyncio
async def test_process_key_calculates_ask_dominant_imbalance_and_bearish_signal() -> None:
    analyzer, event_bus, _, _ = make_analyzer(
        bearish_snapshot(),
        config=make_config(
            depth_levels=3,
            bullish_ratio_threshold=0.60,
            bearish_ratio_threshold=0.40,
            smooth_window=1,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.bid_volume == pytest.approx(25.0)
    assert stats.ask_volume == pytest.approx(100.0)
    assert stats.imbalance_ratio == pytest.approx(25.0 / 125.0)
    assert stats.imbalance_diff == pytest.approx((25.0 - 100.0) / 125.0)
    assert stats.best_bid == pytest.approx(100.0)
    assert stats.best_ask == pytest.approx(100.5)
    assert stats.spread == pytest.approx(0.5)

    assert_update_emitted(event_bus, expected_key=DEFAULT_KEY)
    assert_signal_emitted(
        event_bus,
        expected_key=DEFAULT_KEY,
        signal_type="bearish",
        side="sell",
        reason="orderbook_ask_imbalance",
    )


@pytest.mark.asyncio
async def test_neutral_imbalance_emits_update_but_no_signal() -> None:
    analyzer, event_bus, _, _ = make_analyzer(
        neutral_snapshot(),
        config=make_config(
            bullish_ratio_threshold=0.60,
            bearish_ratio_threshold=0.40,
            smooth_window=1,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.imbalance_ratio == pytest.approx(0.5)
    assert stats.imbalance_diff == pytest.approx(0.0)

    assert_update_emitted(event_bus, expected_key=DEFAULT_KEY)
    assert emitted_events(event_bus, OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL.value) == []


@pytest.mark.asyncio
async def test_depth_levels_limit_is_respected_and_ignores_deeper_liquidity() -> None:
    """
    Vulnerability test.

    Deep liquidity outside configured top-N depth must not distort the signal.
    """
    analyzer, _, _, _ = make_analyzer(
        snapshot_dict(
            bids=[
                level(100.0, 10.0),
                level(99.5, 10.0),
                level(90.0, 10_000.0),
            ],
            asks=[
                level(100.5, 5.0),
                level(101.0, 5.0),
                level(110.0, 10_000.0),
            ],
        ),
        config=make_config(depth_levels=2, smooth_window=1),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.depth_levels_used == 2
    assert stats.bid_volume == pytest.approx(20.0)
    assert stats.ask_volume == pytest.approx(10.0)
    assert stats.imbalance_ratio == pytest.approx(20.0 / 30.0)


@pytest.mark.asyncio
async def test_orderbook_levels_are_sorted_before_calculation() -> None:
    """
    Vulnerability test.

    Cache snapshots can be unsorted. Analyzer must sort bids descending and
    asks ascending before selecting best prices and top depth.
    """
    analyzer, _, _, _ = make_analyzer(
        snapshot_dict(
            bids=[
                level(99.0, 100.0),
                level(100.0, 10.0),
                level(99.5, 20.0),
            ],
            asks=[
                level(102.0, 100.0),
                level(100.5, 10.0),
                level(101.0, 20.0),
            ],
        ),
        config=make_config(depth_levels=2, smooth_window=1),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.best_bid == pytest.approx(100.0)
    assert stats.best_ask == pytest.approx(100.5)
    assert stats.bid_volume == pytest.approx(30.0)
    assert stats.ask_volume == pytest.approx(30.0)
    assert stats.depth_levels_used == 2


@pytest.mark.asyncio
async def test_invalid_orderbook_levels_are_filtered_before_calculation() -> None:
    analyzer, _, _, _ = make_analyzer(
        snapshot_dict(
            bids=[
                level(0.0, 999.0),
                level(100.0, -1.0),
                level(99.5, 10.0),
            ],
            asks=[
                level(0.0, 999.0),
                level(101.0, -1.0),
                level(100.5, 5.0),
            ],
        ),
        config=make_config(depth_levels=3, smooth_window=1),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.bid_volume == pytest.approx(10.0)
    assert stats.ask_volume == pytest.approx(5.0)
    assert stats.best_bid == pytest.approx(99.5)
    assert stats.best_ask == pytest.approx(100.5)
    assert stats.depth_levels_used == 1


@pytest.mark.asyncio
async def test_min_total_volume_blocks_noise_and_emits_nothing() -> None:
    analyzer, event_bus, _, _ = make_analyzer(
        snapshot_dict(
            bids=[level(100.0, 1.0)],
            asks=[level(100.5, 1.0)],
        ),
        config=make_config(min_total_volume=10.0, smooth_window=1),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 0
    assert snapshot["metrics"]["skipped"] >= 1
    assert analyzer.get_latest_stats_by_key(DEFAULT_KEY) is None


# =============================================================================
# Scope / EventBus handling tests
# =============================================================================


@pytest.mark.asyncio
async def test_handle_event_extracts_scoped_key_from_data_layer_payload() -> None:
    cache = StrictOrderbookCache({BYBIT_KEY: bullish_snapshot(orderflow_key=BYBIT_KEY)})
    analyzer, event_bus, _, _ = make_analyzer(
        cache=cache,
        default_exchange="binance",
        default_market_type="usdm_futures",
        default_timeframe="1m",
    )

    event = Event(
        topic=ORDERBOOK_TOPIC,
        payload={
            "data": {
                "scope": {
                    "exchange": "bybit",
                    "market_type": "linear",
                    "symbol": "btcusdt",
                    "timeframe": "1m",
                }
            }
        },
    )

    await analyzer._handle_event(event)  # noqa: SLF001

    assert cache.calls[-1]["exchange"] == "bybit"
    assert cache.calls[-1]["market_type"] == "linear"
    assert cache.calls[-1]["symbol"] == "BTCUSDT"
    assert_update_emitted(event_bus, expected_key=BYBIT_KEY)


@pytest.mark.asyncio
async def test_handle_event_without_symbol_or_key_is_skipped_without_cache_call() -> None:
    cache = StrictOrderbookCache({DEFAULT_KEY: bullish_snapshot()})
    analyzer, event_bus, _, _ = make_analyzer(cache=cache)

    await analyzer._handle_event(  # noqa: SLF001
        Event(
            topic=ORDERBOOK_TOPIC,
            payload={"data": {"price": 100.0}},
        )
    )

    assert cache.calls == []
    assert event_bus.emitted == []
    assert analyzer.stats()["metrics"]["skipped"] == 1


@pytest.mark.asyncio
async def test_snapshot_scope_mismatch_is_rejected_without_emitting() -> None:
    """
    Vulnerability test.

    Cache returning a valid snapshot for the wrong exchange/symbol/timeframe must
    not leak into the requested scoped market.
    """
    wrong_snapshot = bullish_snapshot(orderflow_key=BYBIT_KEY)
    cache = StrictOrderbookCache({DEFAULT_KEY: wrong_snapshot})
    analyzer, event_bus, _, _ = make_analyzer(cache=cache)

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is None
    assert event_bus.emitted == []
    assert analyzer.get_latest_stats_by_key(DEFAULT_KEY) is None


@pytest.mark.asyncio
async def test_market_type_filter_blocks_spot_scope_in_futures_only_mode() -> None:
    spot_key = key(exchange="binance", market_type="spot", symbol="BTCUSDT", timeframe="1m")
    cache = StrictOrderbookCache({spot_key: bullish_snapshot(orderflow_key=spot_key)})
    analyzer, event_bus, _, _ = make_analyzer(
        cache=cache,
        config=make_config(allowed_market_types={"usdm_futures", "linear", "swap"}),
    )

    stats = await analyzer.process_key(spot_key)

    assert stats is None
    assert cache.calls == []
    assert event_bus.emitted == []
    assert analyzer.stats()["metrics"]["skipped"] == 1


@pytest.mark.asyncio
async def test_allowed_exchange_symbol_and_timeframe_filters_are_enforced() -> None:
    analyzer, event_bus, cache, _ = make_analyzer(
        cache=StrictOrderbookCache(
            {
                DEFAULT_KEY: bullish_snapshot(orderflow_key=DEFAULT_KEY),
                ETH_KEY: bullish_snapshot(orderflow_key=ETH_KEY),
                HIGHER_TF_KEY: bullish_snapshot(orderflow_key=HIGHER_TF_KEY),
                BYBIT_KEY: bullish_snapshot(orderflow_key=BYBIT_KEY),
            }
        ),
        config=make_config(
            allowed_exchanges={"binance"},
            allowed_market_types={"usdm_futures"},
            allowed_symbols={"BTCUSDT"},
            allowed_timeframes={"1m"},
        ),
    )

    assert await analyzer.process_key(DEFAULT_KEY) is not None
    assert await analyzer.process_key(ETH_KEY) is None
    assert await analyzer.process_key(HIGHER_TF_KEY) is None
    assert await analyzer.process_key(BYBIT_KEY) is None

    assert len(cache.calls) == 1
    assert cache.calls[0]["symbol"] == "BTCUSDT"
    assert cache.calls[0]["timeframe"] == "1m"

    update_events = emitted_events(
        event_bus,
        OrderFlowEventTopic.ORDERBOOK_IMBALANCE_UPDATED.value,
    )
    assert len(update_events) == 1
    assert_scope_payload(update_events[0]["payload"], DEFAULT_KEY)


# =============================================================================
# Ratio normalization / smoothing tests
# =============================================================================


@pytest.mark.asyncio
async def test_minus_one_to_one_normalization_changes_ratio_but_signal_uses_denormalized_ratio() -> None:
    analyzer, event_bus, _, _ = make_analyzer(
        bullish_snapshot(),
        config=make_config(
            normalize_ratio_to_minus_one_one=True,
            bullish_ratio_threshold=0.60,
            bearish_ratio_threshold=0.40,
            smooth_window=1,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None

    raw_ratio = 100.0 / 125.0
    expected_normalized = (raw_ratio * 2.0) - 1.0

    assert stats.imbalance_ratio == pytest.approx(expected_normalized)
    assert stats.imbalance_diff == pytest.approx(expected_normalized)

    assert_update_emitted(event_bus, expected_key=DEFAULT_KEY)
    assert_signal_emitted(
        event_bus,
        expected_key=DEFAULT_KEY,
        signal_type="bullish",
        side="buy",
        reason="orderbook_bid_imbalance",
    )


@pytest.mark.asyncio
async def test_smoothing_window_averages_recent_ratios_and_can_prevent_signal_spike() -> None:
    """
    Vulnerability test.

    A single spoof-like bid spike should be smoothed after subsequent neutral
    books when smooth_window > 1.
    """
    cache = StrictOrderbookCache(
        {
            DEFAULT_KEY: snapshot_dict(
                bids=[level(100.0, 90.0)],
                asks=[level(100.5, 10.0)],
            )
        }
    )

    analyzer, event_bus, _, _ = make_analyzer(
        cache=cache,
        config=make_config(
            bullish_ratio_threshold=0.80,
            bearish_ratio_threshold=0.20,
            smooth_window=3,
        ),
    )

    first_stats = await analyzer.process_key(DEFAULT_KEY)
    assert first_stats is not None
    assert first_stats.imbalance_ratio == pytest.approx(0.90)

    cache.set_snapshot(
        DEFAULT_KEY,
        snapshot_dict(
            bids=[level(100.0, 50.0)],
            asks=[level(100.5, 50.0)],
        ),
    )
    second_stats = await analyzer.process_key(DEFAULT_KEY)
    assert second_stats is not None
    assert second_stats.imbalance_ratio == pytest.approx((0.90 + 0.50) / 2.0)

    cache.set_snapshot(
        DEFAULT_KEY,
        snapshot_dict(
            bids=[level(100.0, 50.0)],
            asks=[level(100.5, 50.0)],
        ),
    )
    third_stats = await analyzer.process_key(DEFAULT_KEY)
    assert third_stats is not None
    assert third_stats.imbalance_ratio == pytest.approx((0.90 + 0.50 + 0.50) / 3.0)

    signals = emitted_events(
        event_bus,
        OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL.value,
    )
    assert len(signals) == 1
    assert signals[0]["payload"]["signal_type"] == "bullish"


@pytest.mark.asyncio
async def test_smoothing_history_is_bounded_to_configured_window_per_key() -> None:
    cache = StrictOrderbookCache(
        {
            DEFAULT_KEY: snapshot_dict(
                bids=[level(100.0, 90.0)],
                asks=[level(100.5, 10.0)],
            )
        }
    )
    analyzer, _, _, _ = make_analyzer(
        cache=cache,
        config=make_config(smooth_window=2),
    )

    for bid_size, ask_size in [(90, 10), (80, 20), (70, 30), (60, 40)]:
        cache.set_snapshot(
            DEFAULT_KEY,
            snapshot_dict(
                bids=[level(100.0, bid_size)],
                asks=[level(100.5, ask_size)],
            ),
        )
        stats = await analyzer.process_key(DEFAULT_KEY)
        assert stats is not None

    assert analyzer._ratio_history_by_key[DEFAULT_KEY] == pytest.approx([0.70, 0.60])  # noqa: SLF001


@pytest.mark.asyncio
async def test_smoothing_state_is_isolated_between_exchange_scopes() -> None:
    cache = StrictOrderbookCache(
        {
            DEFAULT_KEY: snapshot_dict(
                orderflow_key=DEFAULT_KEY,
                bids=[level(100.0, 90.0)],
                asks=[level(100.5, 10.0)],
            ),
            BYBIT_KEY: snapshot_dict(
                orderflow_key=BYBIT_KEY,
                bids=[level(100.0, 10.0)],
                asks=[level(100.5, 90.0)],
            ),
        }
    )
    analyzer, _, _, _ = make_analyzer(
        cache=cache,
        config=make_config(smooth_window=3),
    )

    binance_stats = await analyzer.process_key(DEFAULT_KEY)
    bybit_stats = await analyzer.process_key(BYBIT_KEY)

    assert binance_stats is not None
    assert bybit_stats is not None
    assert binance_stats.imbalance_ratio == pytest.approx(0.90)
    assert bybit_stats.imbalance_ratio == pytest.approx(0.10)

    assert analyzer._ratio_history_by_key[DEFAULT_KEY] == pytest.approx([0.90])  # noqa: SLF001
    assert analyzer._ratio_history_by_key[BYBIT_KEY] == pytest.approx([0.10])  # noqa: SLF001


# =============================================================================
# Cache compatibility / error handling tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_book_cache_method_is_supported_with_scoped_kwargs() -> None:
    cache = StrictGetBookOrderbookCache({DEFAULT_KEY: bullish_snapshot()})
    analyzer, event_bus, _, _ = make_analyzer(cache=cache)

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert cache.calls == [
        {
            "method": "get_book",
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
        }
    ]
    assert_update_emitted(event_bus, expected_key=DEFAULT_KEY)


@pytest.mark.asyncio
async def test_async_cache_method_is_awaited() -> None:
    cache = AsyncOrderbookCache({DEFAULT_KEY: bullish_snapshot()})
    analyzer, event_bus, _, _ = make_analyzer(cache=cache)

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert cache.calls[-1] == {
        "method": "get_snapshot",
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "1m",
    }
    assert_update_emitted(event_bus, expected_key=DEFAULT_KEY)


@pytest.mark.asyncio
async def test_legacy_symbol_only_get_fallback_still_works_but_is_not_primary_contract() -> None:
    cache = LegacySymbolOnlyOrderbookCache(bullish_snapshot())
    analyzer, _, _, _ = make_analyzer(cache=cache)

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert cache.calls == [("get", "BTCUSDT")]


@pytest.mark.asyncio
async def test_cache_exception_is_handled_without_crashing_or_emitting() -> None:
    cache = StrictOrderbookCache({DEFAULT_KEY: bullish_snapshot()}, fail=True)
    analyzer, event_bus, _, _ = make_analyzer(cache=cache)

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 0
    assert snapshot["metrics"]["skipped"] >= 1 or snapshot["metrics"]["errors"] >= 1


@pytest.mark.asyncio
async def test_emit_update_failure_is_captured_without_crashing() -> None:
    event_bus = FakeEventBus()
    event_bus.fail_emit_topics.add(
        OrderFlowEventTopic.ORDERBOOK_IMBALANCE_UPDATED.value
    )

    analyzer, _, _, _ = make_analyzer(
        bullish_snapshot(),
        event_bus=event_bus,
        config=make_config(emit_signals=False),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert event_bus.emitted == []
    assert analyzer.get_latest_stats_by_key(DEFAULT_KEY) == stats

    metrics = analyzer.stats()["metrics"]
    assert metrics["emit_errors"] == 1
    assert metrics["updates_emitted"] == 0


@pytest.mark.asyncio
async def test_signal_emit_failure_does_not_rollback_latest_stats_or_update_event() -> None:
    event_bus = FakeEventBus()
    event_bus.fail_emit_topics.add(
        OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL.value
    )

    analyzer, _, _, _ = make_analyzer(
        bullish_snapshot(),
        event_bus=event_bus,
        config=make_config(),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert analyzer.get_latest_stats_by_key(DEFAULT_KEY) == stats
    assert len(emitted_events(event_bus, OrderFlowEventTopic.ORDERBOOK_IMBALANCE_UPDATED.value)) == 1
    assert len(emitted_events(event_bus, OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL.value)) == 0

    metrics = analyzer.stats()["metrics"]
    assert metrics["updates_emitted"] == 1
    assert metrics["signals_emitted"] == 0
    assert metrics["emit_errors"] == 1


# =============================================================================
# Malformed / dirty payload tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", malformed_snapshots())
async def test_malformed_snapshots_are_rejected_without_emitting(snapshot: Any) -> None:
    analyzer, event_bus, _, _ = make_analyzer(
        snapshot,
        config=make_config(min_total_volume=0.0),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is None
    assert event_bus.emitted == []


@pytest.mark.asyncio
async def test_snapshot_model_from_cache_is_supported_and_rescoped_exactly() -> None:
    model = OrderbookSnapshot.create(
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe="1m",
        bids=[[100.0, 2.0], [99.5, 1.0]],
        asks=[[100.5, 1.0], [101.0, 1.0]],
        timestamp=now_ts(),
        sequence_id="model-seq",
    )
    cache = StrictOrderbookCache({DEFAULT_KEY: {"data": model}})
    analyzer, event_bus, _, _ = make_analyzer(cache=cache)

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.bid_volume == pytest.approx(3.0)
    assert stats.ask_volume == pytest.approx(2.0)
    assert_update_emitted(event_bus, expected_key=DEFAULT_KEY)


@pytest.mark.asyncio
async def test_negative_spread_snapshot_is_detected_as_market_data_problem() -> None:
    """
    Vulnerability test.

    A crossed book usually means stale/corrupt orderbook state. The desired
    behavior is to reject it instead of emitting a clean directional signal.

    If this fails, the analyzer currently needs an explicit crossed-book guard.
    """
    analyzer, event_bus, _, _ = make_analyzer(
        snapshot_dict(
            bids=[level(101.0, 10.0)],
            asks=[level(100.0, 10.0)],
        )
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is None
    assert event_bus.emitted == []


# =============================================================================
# Signal throttling / cleanup / concurrency tests
# =============================================================================


@pytest.mark.asyncio
async def test_signal_throttling_is_per_scoped_key_not_per_symbol() -> None:
    cache = StrictOrderbookCache(
        {
            DEFAULT_KEY: bullish_snapshot(orderflow_key=DEFAULT_KEY),
            BYBIT_KEY: bullish_snapshot(orderflow_key=BYBIT_KEY),
        }
    )
    analyzer, event_bus, _, _ = make_analyzer(
        cache=cache,
        config=make_config(min_signal_interval_sec=3600.0),
    )

    first = await analyzer.process_key(DEFAULT_KEY)
    second = await analyzer.process_key(DEFAULT_KEY)
    third = await analyzer.process_key(BYBIT_KEY)

    assert first is not None
    assert second is not None
    assert third is not None

    signals = emitted_events(
        event_bus,
        OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL.value,
    )

    assert len(signals) == 2
    assert_scope_payload(signals[0]["payload"], DEFAULT_KEY)
    assert_scope_payload(signals[1]["payload"], BYBIT_KEY)

    metrics = analyzer.stats()["metrics"]
    assert metrics["signals_emitted"] == 2
    assert metrics["skipped"] >= 1


@pytest.mark.asyncio
async def test_cleanup_removes_only_stale_scoped_state() -> None:
    cache = StrictOrderbookCache(
        {
            DEFAULT_KEY: bullish_snapshot(orderflow_key=DEFAULT_KEY),
            BYBIT_KEY: bearish_snapshot(orderflow_key=BYBIT_KEY),
        }
    )
    analyzer, _, _, _ = make_analyzer(
        cache=cache,
        config=make_config(cleanup_interval_sec=5.0, smooth_window=2),
    )

    assert await analyzer.process_key(DEFAULT_KEY) is not None
    assert await analyzer.process_key(BYBIT_KEY) is not None

    analyzer._last_snapshot_ts_by_key[DEFAULT_KEY] = time.time() - 10_000  # noqa: SLF001
    analyzer._last_snapshot_ts_by_key[BYBIT_KEY] = time.time()  # noqa: SLF001

    await analyzer.cleanup()

    assert analyzer.get_latest_stats_by_key(DEFAULT_KEY) is None
    assert analyzer.get_latest_stats_by_key(BYBIT_KEY) is not None
    assert DEFAULT_KEY not in analyzer._ratio_history_by_key  # noqa: SLF001
    assert BYBIT_KEY in analyzer._ratio_history_by_key or make_config().smooth_window == 1


@pytest.mark.asyncio
async def test_concurrent_process_key_calls_do_not_corrupt_state() -> None:
    cache = StrictOrderbookCache({DEFAULT_KEY: bullish_snapshot()})
    analyzer, event_bus, _, _ = make_analyzer(cache=cache)

    results = await asyncio.gather(
        analyzer.process_key(DEFAULT_KEY),
        analyzer.process_key(DEFAULT_KEY),
        analyzer.process_key(DEFAULT_KEY),
    )

    assert all(result is not None for result in results)
    assert analyzer.get_latest_stats_by_key(DEFAULT_KEY) is not None

    tracked_markets = analyzer.stats()["tracked_markets"]
    assert len(tracked_markets) == 1
    assert tracked_markets[0] == {
        **orderflow_key_to_dict(DEFAULT_KEY),
        "has_stats": True,
        "ratio_history_size": 0,
        "last_snapshot_ts": pytest.approx(tracked_markets[0]["last_snapshot_ts"]),
    }

    update_events = emitted_events(
        event_bus,
        OrderFlowEventTopic.ORDERBOOK_IMBALANCE_UPDATED.value,
    )
    assert len(update_events) == 3


# =============================================================================
# Backward compatibility test
# =============================================================================


@pytest.mark.asyncio
async def test_process_symbol_wrapper_uses_explicit_default_futures_scope() -> None:
    """
    Compatibility only.

    New tests should use process_key(), but this protects old callers during
    migration and makes sure defaults are not silently spot/symbol-only.
    """
    analyzer, event_bus, cache, _ = make_analyzer(
        bullish_snapshot(),
        default_exchange="binance",
        default_market_type="usdm_futures",
        default_timeframe="1m",
    )

    stats = await analyzer.process_symbol("btcusdt")

    assert stats is not None
    assert stats.key == DEFAULT_KEY
    assert cache.calls[-1]["exchange"] == "binance"
    assert cache.calls[-1]["market_type"] == "usdm_futures"
    assert cache.calls[-1]["symbol"] == "BTCUSDT"
    assert cache.calls[-1]["timeframe"] == "1m"
    assert_update_emitted(event_bus, expected_key=DEFAULT_KEY)