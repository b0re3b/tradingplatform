# tests/analytics/orderflow/test_orderflow_lifecycle.py

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from core.event_bus import Event, EventPriority

from analytics.orderflow.analyzer import OrderFlowAnalyzer
from analytics.orderflow.base import BaseOrderFlowAnalyzer
from analytics.orderflow.config import BaseOrderFlowSubConfig, OrderFlowConfig
from analytics.orderflow.enums import (
    OrderFlowEventTopic,
    OrderFlowMetricType,
    OrderFlowSide,
    OrderFlowSignalType,
    OrderFlowSourceType,
)
from analytics.orderflow.models import (
    BaseOrderFlowStats,
    OrderFlowKey,
    OrderFlowSignal,
    make_orderflow_key,
    orderflow_key_to_dict,
    orderflow_key_to_string,
)


# =============================================================================
# Constants
# =============================================================================

TRADE_TOPIC = "market.trades.updated"
ORDERBOOK_TOPIC = "market.orderbook.updated"
RAW_TRADE_TOPIC = "market.trade"
RAW_ORDERBOOK_TOPIC = "market.orderbook"

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
SPOT_KEY: OrderFlowKey = make_orderflow_key(
    exchange="binance",
    market_type="spot",
    symbol="BTCUSDT",
    timeframe="1m",
)
ETH_KEY: OrderFlowKey = make_orderflow_key(
    exchange="binance",
    market_type="usdm_futures",
    symbol="ETHUSDT",
    timeframe="1m",
)


# =============================================================================
# Fakes
# =============================================================================


@dataclass(slots=True)
class FakeSubscription:
    pattern: str
    handler: Any
    name: str
    enabled: bool = True


@dataclass(slots=True)
class FakeScheduledJob:
    job_id: str
    name: str
    func: Any
    interval: float
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    run_immediately: bool
    max_retries: int
    retry_delay: float
    timeout: float | None
    allow_overlap: bool
    enabled: bool = True


class FakeEventBus:
    """
    Strict fake for core.event_bus.EventBus.

    It records subscriptions, unsubscriptions and emitted events.
    It can simulate EventBus.emit() failures for selected topics.
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
    Strict fake for core.scheduler.Scheduler.

    BaseOrderFlowAnalyzer uses get_job_by_name() before add_interval_job().
    This fake mirrors that contract and also tracks removed jobs to catch
    lifecycle leaks.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, FakeScheduledJob] = {}
        self.disabled_job_ids: list[str] = []
        self.removed_job_ids: list[str] = []

    def add_interval_job(
        self,
        name: str,
        func: Any,
        *,
        interval: float,
        args: tuple[Any, ...] = (),
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

        existing = self.get_job_by_name(name)
        if existing is not None:
            return existing.job_id

        job_id = f"job-{len(self.jobs) + 1}"
        self.jobs[job_id] = FakeScheduledJob(
            job_id=job_id,
            name=name,
            func=func,
            interval=interval,
            args=args,
            kwargs=kwargs or {},
            run_immediately=run_immediately,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            allow_overlap=allow_overlap,
            enabled=enabled,
        )
        return job_id

    def get_job_by_name(self, name: str) -> FakeScheduledJob | None:
        for job in self.jobs.values():
            if job.name == name:
                return job
        return None

    def disable_job(self, job_id: str) -> None:
        if job_id not in self.jobs:
            raise KeyError(job_id)

        self.jobs[job_id].enabled = False
        self.disabled_job_ids.append(job_id)

    def remove_job(self, job_id: str) -> None:
        if job_id not in self.jobs:
            raise KeyError(job_id)

        self.removed_job_ids.append(job_id)
        self.jobs.pop(job_id)


class AsyncRemoveScheduler(FakeScheduler):
    """
    Vulnerability fake.

    BaseOrderFlowAnalyzer.stop() is sync. If Scheduler.remove_job() becomes async,
    base currently cannot await it and jobs remain in this fake.
    """

    async def remove_job(self, job_id: str) -> None:  # type: ignore[override]
        await asyncio.sleep(0)
        if job_id not in self.jobs:
            raise KeyError(job_id)
        self.removed_job_ids.append(job_id)
        self.jobs.pop(job_id)


class FakeTradesCache:
    pass


class FakeOrderbookCache:
    pass


# =============================================================================
# Dummy analyzer for testing BaseOrderFlowAnalyzer directly
# =============================================================================


class DummyOrderFlowAnalyzer(BaseOrderFlowAnalyzer):
    def __init__(
        self,
        *,
        event_bus: FakeEventBus,
        scheduler: FakeScheduler | None,
        config: BaseOrderFlowSubConfig | None = None,
        source_topic_patterns: list[str] | tuple[str, ...] = (TRADE_TOPIC,),
        default_exchange: str = "binance",
        default_market_type: str = "usdm_futures",
        default_timeframe: str = "1m",
    ) -> None:
        super().__init__(
            event_bus=event_bus,  # type: ignore[arg-type]
            scheduler=scheduler,  # type: ignore[arg-type]
            config=config or make_base_config(),
            metric_type=OrderFlowMetricType.CVD,
            source_type=OrderFlowSourceType.TRADES,
            source_topic_patterns=source_topic_patterns,
            component_module="orderflow_test",
            default_exchange=default_exchange,
            default_market_type=default_market_type,
            default_timeframe=default_timeframe,
        )
        self.handled_events: list[Event] = []
        self.processed_keys: list[OrderFlowKey] = []
        self.cleaned = False
        self.latest_by_key: dict[OrderFlowKey, BaseOrderFlowStats] = {}

    async def process_key(self, key: OrderFlowKey) -> BaseOrderFlowStats | None:
        normalized = self.make_key(
            exchange=key[0],
            market_type=key[1],
            symbol=key[2],
            timeframe=key[3],
        )
        self.processed_keys.append(normalized)
        stats = make_stats(key=normalized)
        self.latest_by_key[normalized] = stats
        return stats

    def get_latest_stats_by_key(self, key: OrderFlowKey) -> BaseOrderFlowStats | None:
        normalized = self.make_key(
            exchange=key[0],
            market_type=key[1],
            symbol=key[2],
            timeframe=key[3],
        )
        return self.latest_by_key.get(normalized) or make_stats(key=normalized)

    async def _handle_event(self, event: Event) -> None:
        self.handled_events.append(event)

    async def cleanup(self) -> None:
        self.cleaned = True


# =============================================================================
# Facade stub modules
# =============================================================================


class StubModule:
    def __init__(
        self,
        *,
        name: str,
        result: BaseOrderFlowStats | None = None,
        raise_on_register: bool = False,
        raise_on_stop: bool = False,
        raise_on_process: bool = False,
        raise_on_cleanup: bool = False,
    ) -> None:
        self.name = name
        self.result = result
        self.raise_on_register = raise_on_register
        self.raise_on_stop = raise_on_stop
        self.raise_on_process = raise_on_process
        self.raise_on_cleanup = raise_on_cleanup

        self.register_calls = 0
        self.stop_calls = 0
        self.processed_keys: list[OrderFlowKey] = []
        self.cleanup_calls = 0

    def register(self) -> None:
        self.register_calls += 1
        if self.raise_on_register:
            raise RuntimeError(f"{self.name} register failed")

    def stop(self) -> None:
        self.stop_calls += 1
        if self.raise_on_stop:
            raise RuntimeError(f"{self.name} stop failed")

    async def process_key(self, key: OrderFlowKey) -> BaseOrderFlowStats | None:
        self.processed_keys.append(key)
        if self.raise_on_process:
            raise RuntimeError(f"{self.name} process failed")
        return self.result

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.raise_on_cleanup:
            raise RuntimeError(f"{self.name} cleanup failed")

    def get_latest_stats_by_key(self, key: OrderFlowKey) -> BaseOrderFlowStats | None:
        return self.result

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "register_calls": self.register_calls,
            "stop_calls": self.stop_calls,
            "cleanup_calls": self.cleanup_calls,
            "processed_keys": [list(item) for item in self.processed_keys],
        }


# =============================================================================
# Factories
# =============================================================================


def make_key(
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


def make_base_config(**overrides: Any) -> BaseOrderFlowSubConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "emit_updates": True,
        "emit_signals": True,
        "min_signal_interval_sec": 60.0,
        "health_log_interval_sec": 5.0,
        "cleanup_interval_sec": 7.0,
        "scheduler_job_timeout_sec": 0.5,
        "scheduler_job_retry_delay_sec": 0.1,
        "scheduler_job_max_retries": 2,
        "publish_priority": EventPriority.HIGH,
        "source_name": "dummy_orderflow",
        "update_topic": "analytics.orderflow.dummy.updated",
        "signal_topic": "analytics.orderflow.dummy.signal",
        "allowed_market_types": {"usdm_futures", "linear", "swap"},
    }
    values.update(overrides)
    config = BaseOrderFlowSubConfig(**values)
    config.validate()
    return config


def make_orderflow_config(**overrides: Any) -> OrderFlowConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "default_exchange": "binance",
        "default_market_type": "usdm_futures",
        "default_timeframe": "1m",
        "allowed_market_types": {"usdm_futures", "linear", "swap"},
        "source_topic_patterns_trades": [TRADE_TOPIC],
        "source_topic_patterns_orderbook": [ORDERBOOK_TOPIC],
        "trade_input_topics": (TRADE_TOPIC,),
        "orderbook_input_topics": (ORDERBOOK_TOPIC,),
    }
    values.update(overrides)
    return OrderFlowConfig(**values)


def make_stats(
    *,
    key: OrderFlowKey = DEFAULT_KEY,
    metric: OrderFlowMetricType = OrderFlowMetricType.CVD,
    source_type: OrderFlowSourceType = OrderFlowSourceType.TRADES,
) -> BaseOrderFlowStats:
    return BaseOrderFlowStats(
        exchange=key[0],
        market_type=key[1],
        symbol=key[2],
        timeframe=key[3],
        metric=metric,
        source_type=source_type,
    )


def make_signal(key: OrderFlowKey = DEFAULT_KEY) -> OrderFlowSignal:
    return OrderFlowSignal(
        exchange=key[0],
        market_type=key[1],
        symbol=key[2],
        timeframe=key[3],
        metric=OrderFlowMetricType.CVD,
        source_type=OrderFlowSourceType.TRADES,
        signal_type=OrderFlowSignalType.BULLISH,
        side=OrderFlowSide.BUY,
        strength=0.85,
        reason="test_signal",
        context={
            "case": "lifecycle",
            "scope": orderflow_key_to_dict(key),
            "scope_key": orderflow_key_to_string(key),
        },
    )


def make_facade(
    *,
    event_bus: FakeEventBus | None = None,
    scheduler: FakeScheduler | None = None,
    config: OrderFlowConfig | None = None,
) -> OrderFlowAnalyzer:
    return OrderFlowAnalyzer(
        event_bus=event_bus or FakeEventBus(),  # type: ignore[arg-type]
        scheduler=scheduler or FakeScheduler(),  # type: ignore[arg-type]
        config=config or make_orderflow_config(),
        trades_cache=FakeTradesCache(),
        orderbook_cache=FakeOrderbookCache(),
    )


def emitted_events(event_bus: FakeEventBus, topic: str) -> list[dict[str, Any]]:
    return [item for item in event_bus.emitted if item["topic"] == topic]


def assert_key_payload(payload: dict[str, Any], expected_key: OrderFlowKey) -> None:
    assert payload["key"] == list(expected_key)
    assert tuple(payload["orderflow_key"]) == expected_key
    assert payload["scope"] == orderflow_key_to_dict(expected_key)
    assert payload["scope_key"] == orderflow_key_to_string(expected_key)


# =============================================================================
# BaseOrderFlowAnalyzer lifecycle tests
# =============================================================================


def test_base_register_subscribes_to_data_layer_event_bus_and_schedules_jobs() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        source_topic_patterns=(TRADE_TOPIC, "market.trades.*"),
    )

    analyzer.register()

    assert analyzer.is_running is True
    assert [sub.pattern for sub in event_bus.subscriptions] == [
        TRADE_TOPIC,
        "market.trades.*",
    ]
    assert all(sub.handler == analyzer._handle_event for sub in event_bus.subscriptions)
    assert all("DummyOrderFlowAnalyzer" in sub.name for sub in event_bus.subscriptions)

    assert len(scheduler.jobs) == 2
    job_names = {job.name for job in scheduler.jobs.values()}
    assert "analytics.orderflow.dummy_orderflow.health" in job_names
    assert "analytics.orderflow.dummy_orderflow.cleanup" in job_names

    for job in scheduler.jobs.values():
        assert job.enabled is True
        assert job.allow_overlap is False
        assert job.max_retries == analyzer.stats()["config"]["scheduler_job_max_retries"]
        assert job.retry_delay == analyzer.stats()["config"]["scheduler_job_retry_delay_sec"]
        assert job.timeout == analyzer.stats()["config"]["scheduler_job_timeout_sec"]


def test_base_register_is_idempotent_and_does_not_duplicate_subscriptions_or_jobs() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        source_topic_patterns=(TRADE_TOPIC, "market.trades.*"),
    )

    analyzer.register()
    analyzer.register()

    assert len(event_bus.subscriptions) == 2
    assert len(scheduler.jobs) == 2
    assert analyzer.is_running is True


def test_base_stop_unsubscribes_all_handlers_and_removes_scheduler_jobs() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        source_topic_patterns=(TRADE_TOPIC, "market.trades.*"),
    )

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


def test_base_register_disabled_analyzer_creates_no_subscriptions_and_no_jobs() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        config=make_base_config(enabled=False),
        source_topic_patterns=(TRADE_TOPIC,),
    )

    analyzer.register()

    assert analyzer.is_running is False
    assert event_bus.subscriptions == []
    assert scheduler.jobs == {}


def test_base_register_rejects_raw_market_topics_by_default() -> None:
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=FakeEventBus(),
        scheduler=FakeScheduler(),
        source_topic_patterns=(RAW_TRADE_TOPIC,),
    )

    with pytest.raises(ValueError, match="Raw market topic"):
        analyzer.register()


def test_base_scheduler_get_job_by_name_prevents_duplicate_job_registration_after_partial_state_reset() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    analyzer = DummyOrderFlowAnalyzer(event_bus=event_bus, scheduler=scheduler)

    analyzer.register()
    first_job_ids = set(scheduler.jobs)

    # Simulate a bad external state flip. Scheduler should still deduplicate by job name.
    analyzer._running = False  # noqa: SLF001
    analyzer._subscriptions.clear()  # noqa: SLF001
    analyzer.register()

    assert set(scheduler.jobs) == first_job_ids
    assert len(scheduler.jobs) == 2


@pytest.mark.xfail(
    reason=(
        "BaseOrderFlowAnalyzer.stop() is sync and currently cannot await an async "
        "Scheduler.remove_job(); keep as lifecycle vulnerability test."
    ),
    strict=False,
)
def test_base_stop_with_async_remove_job_would_leave_orphan_jobs_until_base_becomes_async_safe() -> None:
    scheduler = AsyncRemoveScheduler()
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=FakeEventBus(),
        scheduler=scheduler,  # type: ignore[arg-type]
    )

    analyzer.register()
    analyzer.stop()

    assert scheduler.jobs == {}


# =============================================================================
# BaseOrderFlowAnalyzer emit / extraction / normalization tests
# =============================================================================


@pytest.mark.asyncio
async def test_base_emit_update_publishes_json_safe_scoped_payload_and_updates_metrics() -> None:
    event_bus = FakeEventBus()
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=FakeScheduler(),
    )

    await analyzer.emit_update(make_stats(key=DEFAULT_KEY))

    assert len(event_bus.emitted) == 1
    emitted = event_bus.emitted[0]
    assert emitted["topic"] == "analytics.orderflow.dummy.updated"
    assert emitted["source"] == "dummy_orderflow"
    assert emitted["priority"] == EventPriority.HIGH

    payload = emitted["payload"]
    assert payload["metric"] == "cvd"
    assert payload["source_type"] == "trades"
    assert_key_payload(payload, DEFAULT_KEY)
    assert_key_payload(payload["stats"], DEFAULT_KEY)

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["updates_emitted"] == 1
    assert snapshot["metrics"]["keys"][orderflow_key_to_string(DEFAULT_KEY)]["updates_emitted"] == 1


@pytest.mark.asyncio
async def test_base_emit_signal_throttles_repeated_signal_for_same_scoped_key() -> None:
    event_bus = FakeEventBus()
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=FakeScheduler(),
        config=make_base_config(min_signal_interval_sec=3600.0),
    )

    await analyzer.emit_signal(make_signal(DEFAULT_KEY))
    await analyzer.emit_signal(make_signal(DEFAULT_KEY))
    await analyzer.emit_signal(make_signal(BYBIT_KEY))

    signal_events = emitted_events(event_bus, "analytics.orderflow.dummy.signal")
    assert len(signal_events) == 2
    assert_key_payload(signal_events[0]["payload"], DEFAULT_KEY)
    assert_key_payload(signal_events[1]["payload"], BYBIT_KEY)

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["signals_emitted"] == 2
    assert snapshot["metrics"]["skipped"] == 1
    assert snapshot["metrics"]["keys"][orderflow_key_to_string(DEFAULT_KEY)]["signals_emitted"] == 1
    assert snapshot["metrics"]["keys"][orderflow_key_to_string(DEFAULT_KEY)]["skipped"] == 1
    assert snapshot["metrics"]["keys"][orderflow_key_to_string(BYBIT_KEY)]["signals_emitted"] == 1


@pytest.mark.asyncio
async def test_base_emit_failure_is_captured_as_metric_without_crashing() -> None:
    event_bus = FakeEventBus()
    event_bus.fail_emit_topics.add("analytics.orderflow.dummy.updated")

    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=FakeScheduler(),
    )

    await analyzer.emit_update(make_stats(key=ETH_KEY))

    assert event_bus.emitted == []
    snapshot = analyzer.stats()
    assert snapshot["metrics"]["emit_errors"] == 1
    assert snapshot["metrics"]["updates_emitted"] == 0


def test_base_extract_key_from_flat_nested_alias_and_tuple_payloads() -> None:
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=FakeEventBus(),
        scheduler=FakeScheduler(),
    )

    flat_event = Event(
        topic=TRADE_TOPIC,
        payload={
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "btcusdt",
            "timeframe": "1m",
        },
    )
    nested_event = Event(
        topic=TRADE_TOPIC,
        payload={
            "data": {
                "scope": {
                    "venue": "bybit",
                    "category": "linear",
                    "s": "btcusdt",
                    "tf": "1m",
                }
            }
        },
    )
    key_event = Event(topic=TRADE_TOPIC, payload={"orderflow_key": list(ETH_KEY)})
    missing_symbol_event = Event(topic=TRADE_TOPIC, payload={"data": {"price": 300.0}})

    assert analyzer.extract_key_from_event(flat_event) == DEFAULT_KEY
    assert analyzer.extract_key_from_event(nested_event) == BYBIT_KEY
    assert analyzer.extract_key_from_event(key_event) == ETH_KEY
    assert analyzer.extract_key_from_event(missing_symbol_event) is None
    assert analyzer.extract_symbol_from_event(flat_event) == "BTCUSDT"


def test_base_should_process_key_enforces_futures_scope_filters() -> None:
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=FakeEventBus(),
        scheduler=FakeScheduler(),
        config=make_base_config(
            allowed_exchanges={"binance"},
            allowed_market_types={"usdm_futures"},
            allowed_symbols={"BTCUSDT"},
            allowed_timeframes={"1m"},
        ),
    )

    assert analyzer.should_process_key(DEFAULT_KEY) is True
    assert analyzer.should_process_key(BYBIT_KEY) is False
    assert analyzer.should_process_key(SPOT_KEY) is False
    assert analyzer.should_process_key(ETH_KEY) is False


@pytest.mark.asyncio
async def test_base_process_symbol_wrapper_uses_explicit_default_futures_scope() -> None:
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=FakeEventBus(),
        scheduler=FakeScheduler(),
        default_exchange="binance",
        default_market_type="usdm_futures",
        default_timeframe="1m",
    )

    result = await analyzer.process_symbol("btcusdt")

    assert result is not None
    assert result.key == DEFAULT_KEY
    assert analyzer.processed_keys == [DEFAULT_KEY]


def test_base_normalize_trade_rejects_malformed_payloads_and_normalizes_valid_scoped_aliases() -> None:
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=FakeEventBus(),
        scheduler=FakeScheduler(),
    )

    valid = analyzer.normalize_trade(
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "s": "btcusdt",
            "timeframe": "1m",
            "side": "buy",
            "price": "100.5",
            "qty": "2",
            "timestamp": 123456.0,
            "trade_id": 42,
        }
    )

    assert valid is not None
    assert valid.key == DEFAULT_KEY
    assert valid.side == OrderFlowSide.BUY
    assert valid.price == 100.5
    assert valid.quantity == 2.0
    assert valid.notional == 201.0
    assert valid.trade_id == "42"

    assert analyzer.normalize_trade(None) is None
    assert analyzer.normalize_trade("bad") is None
    assert analyzer.normalize_trade({"symbol": "BTCUSDT", "side": "buy"}) is None
    assert analyzer.normalize_trade(
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "side": "unknown",
            "price": 100,
            "quantity": 1,
            "timestamp": 1,
        }
    ) is None
    assert analyzer.normalize_trade(
        {
            "exchange": "binance",
            "market_type": "spot",
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "side": "buy",
            "price": 100,
            "quantity": 1,
            "timestamp": 1,
        }
    ) is None


# =============================================================================
# OrderFlowConfig topic / scope tests
# =============================================================================


def test_orderflow_config_uses_data_layer_topics_and_rejects_raw_topics_by_default() -> None:
    config = make_orderflow_config()

    assert config.production_input_topics == (TRADE_TOPIC, ORDERBOOK_TOPIC)
    assert config.trades_topics == (TRADE_TOPIC,)
    assert config.orderbook_topics == (ORDERBOOK_TOPIC,)

    with pytest.raises(ValueError, match="Raw market topic"):
        OrderFlowConfig(
            source_topic_patterns_trades=[RAW_TRADE_TOPIC],
            source_topic_patterns_orderbook=[ORDERBOOK_TOPIC],
            trade_input_topics=(RAW_TRADE_TOPIC,),
            orderbook_input_topics=(ORDERBOOK_TOPIC,),
            allow_raw_market_topics=False,
        )


def test_orderflow_config_allows_raw_topics_only_when_explicitly_enabled_for_migration() -> None:
    config = OrderFlowConfig(
        source_topic_patterns_trades=[RAW_TRADE_TOPIC],
        source_topic_patterns_orderbook=[RAW_ORDERBOOK_TOPIC],
        trade_input_topics=(RAW_TRADE_TOPIC,),
        orderbook_input_topics=(RAW_ORDERBOOK_TOPIC,),
        allow_raw_market_topics=True,
    )

    assert config.production_input_topics == (RAW_TRADE_TOPIC, RAW_ORDERBOOK_TOPIC)
    assert config.allow_raw_market_topics is True


def test_orderflow_config_propagates_scope_filters_to_subconfigs() -> None:
    config = make_orderflow_config(
        allowed_exchanges={"binance"},
        allowed_market_types={"usdm_futures"},
        allowed_symbols={"BTCUSDT"},
        allowed_timeframes={"1m"},
    )

    assert config.should_process_key(DEFAULT_KEY) is True
    assert config.should_process_key(BYBIT_KEY) is False
    assert config.should_process_key(SPOT_KEY) is False
    assert config.should_process_key(ETH_KEY) is False

    for subconfig in (
        config.cvd,
        config.volume_delta,
        config.aggressive_trades,
        config.orderbook_imbalance,
    ):
        assert subconfig.should_process_key(DEFAULT_KEY) is True
        assert subconfig.should_process_key(BYBIT_KEY) is False
        assert subconfig.should_process_key(SPOT_KEY) is False
        assert subconfig.should_process_key(ETH_KEY) is False


# =============================================================================
# OrderFlowAnalyzer facade lifecycle tests
# =============================================================================


@pytest.mark.asyncio
async def test_facade_register_registers_only_enabled_modules_and_emits_started_event() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()

    config = make_orderflow_config()
    config.volume_delta.enabled = False
    config.aggressive_trades.enabled = False
    config.validate()

    analyzer = make_facade(
        event_bus=event_bus,
        scheduler=scheduler,
        config=config,
    )

    await analyzer.register()

    assert analyzer.is_running is True

    subscription_names = [sub.name for sub in event_bus.subscriptions]
    assert any("CvdAnalyzer" in name for name in subscription_names)
    assert any("OrderbookImbalanceAnalyzer" in name for name in subscription_names)
    assert not any("VolumeDeltaAnalyzer" in name for name in subscription_names)
    assert not any("AggressiveTradesAnalyzer" in name for name in subscription_names)

    started_events = emitted_events(event_bus, OrderFlowEventTopic.STARTED.value)
    assert len(started_events) == 1
    assert started_events[0]["source"] == "orderflow_analyzer"

    payload = started_events[0]["payload"]
    assert payload["enabled"] is True
    assert payload["modules"] == ["cvd", "orderbook_imbalance"]
    assert payload["enabled_modules"] == ["cvd", "orderbook_imbalance"]
    assert payload["trades_topic_patterns"] == [TRADE_TOPIC]
    assert payload["orderbook_topic_patterns"] == [ORDERBOOK_TOPIC]
    assert payload["input_topics"] == [TRADE_TOPIC, ORDERBOOK_TOPIC]
    assert payload["scope"] == "exchange:market_type:symbol:timeframe"
    assert payload["defaults"] == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "timeframe": "1m",
    }


@pytest.mark.asyncio
async def test_facade_register_is_idempotent_and_does_not_double_subscribe_or_emit() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    analyzer = make_facade(event_bus=event_bus, scheduler=scheduler)

    await analyzer.register()
    first_subscription_count = len(event_bus.subscriptions)
    first_job_count = len(scheduler.jobs)
    first_event_count = len(event_bus.emitted)

    await analyzer.register()

    assert len(event_bus.subscriptions) == first_subscription_count
    assert len(scheduler.jobs) == first_job_count
    assert len(event_bus.emitted) == first_event_count


@pytest.mark.asyncio
async def test_facade_stop_stops_modules_in_reverse_order_and_emits_stopped_event() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    analyzer = make_facade(event_bus=event_bus, scheduler=scheduler)

    await analyzer.register()
    assert analyzer.is_running is True
    assert scheduler.jobs

    await analyzer.stop()

    assert analyzer.is_running is False
    assert event_bus.subscriptions == []
    assert scheduler.jobs == {}

    stopped_events = emitted_events(event_bus, OrderFlowEventTopic.STOPPED.value)
    assert len(stopped_events) == 1
    assert stopped_events[0]["source"] == "orderflow_analyzer"
    assert stopped_events[0]["payload"]["modules"] == [
        "orderbook_imbalance",
        "aggressive_trades",
        "volume_delta",
        "cvd",
    ]
    assert stopped_events[0]["payload"]["scope"] == "exchange:market_type:symbol:timeframe"


@pytest.mark.asyncio
async def test_facade_register_disabled_package_emits_stopped_event_without_subscriptions() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    config = make_orderflow_config(enabled=False)

    analyzer = make_facade(
        event_bus=event_bus,
        scheduler=scheduler,
        config=config,
    )

    await analyzer.register()

    assert analyzer.is_running is False
    assert event_bus.subscriptions == []
    assert scheduler.jobs == {}

    stopped_events = emitted_events(event_bus, OrderFlowEventTopic.STOPPED.value)
    assert len(stopped_events) == 1
    payload = stopped_events[0]["payload"]
    assert payload["reason"] == "disabled_by_config"
    assert payload["enabled"] is False
    assert payload["scope"] == "exchange:market_type:symbol:timeframe"
    assert payload["input_topics"] == [TRADE_TOPIC, ORDERBOOK_TOPIC]


@pytest.mark.asyncio
async def test_facade_constructor_rejects_raw_topic_overrides_by_default() -> None:
    with pytest.raises(ValueError, match="Raw market topic"):
        make_facade(
            config=make_orderflow_config(),
            event_bus=FakeEventBus(),
            scheduler=FakeScheduler(),
        ).__class__(
            event_bus=FakeEventBus(),  # type: ignore[arg-type]
            scheduler=FakeScheduler(),  # type: ignore[arg-type]
            config=make_orderflow_config(),
            trades_cache=FakeTradesCache(),
            orderbook_cache=FakeOrderbookCache(),
            trades_topic_patterns=[RAW_TRADE_TOPIC],
            orderbook_topic_patterns=[ORDERBOOK_TOPIC],
        )


@pytest.mark.asyncio
async def test_facade_stats_exposes_scope_topics_scheduler_and_module_stats() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    analyzer = make_facade(event_bus=event_bus, scheduler=scheduler)

    await analyzer.register()
    snapshot = analyzer.stats()

    assert snapshot["running"] is True
    assert snapshot["enabled"] is True
    assert snapshot["enabled_modules"] == (
        "cvd",
        "volume_delta",
        "aggressive_trades",
        "orderbook_imbalance",
    )
    assert snapshot["scope"] == "exchange:market_type:symbol:timeframe"
    assert snapshot["defaults"] == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "timeframe": "1m",
    }
    assert snapshot["trades_topic_patterns"] == [TRADE_TOPIC]
    assert snapshot["orderbook_topic_patterns"] == [ORDERBOOK_TOPIC]
    assert snapshot["input_topics"] == [TRADE_TOPIC, ORDERBOOK_TOPIC]
    assert snapshot["scheduler_attached"] is True
    assert set(snapshot["modules"]) == {
        "cvd",
        "volume_delta",
        "aggressive_trades",
        "orderbook_imbalance",
    }


# =============================================================================
# Facade scoped manual API tests with stub modules
# =============================================================================


@pytest.mark.asyncio
async def test_facade_process_key_calls_only_enabled_modules_and_returns_scoped_result() -> None:
    analyzer = make_facade(event_bus=FakeEventBus(), scheduler=FakeScheduler())
    config = analyzer._config  # noqa: SLF001
    config.aggressive_trades.enabled = False
    config.orderbook_imbalance.enabled = False

    cvd_result = make_stats(key=DEFAULT_KEY, metric=OrderFlowMetricType.CVD)
    volume_result = make_stats(
        key=DEFAULT_KEY,
        metric=OrderFlowMetricType.VOLUME_DELTA,
    )
    cvd = StubModule(name="cvd", result=cvd_result)
    volume_delta = StubModule(name="volume_delta", result=volume_result)
    aggressive = StubModule(name="aggressive_trades")
    orderbook = StubModule(name="orderbook_imbalance")

    analyzer.cvd = cvd  # type: ignore[assignment]
    analyzer.volume_delta = volume_delta  # type: ignore[assignment]
    analyzer.aggressive_trades = aggressive  # type: ignore[assignment]
    analyzer.orderbook_imbalance = orderbook  # type: ignore[assignment]
    analyzer._modules = {  # noqa: SLF001
        "cvd": cvd,
        "volume_delta": volume_delta,
        "aggressive_trades": aggressive,
        "orderbook_imbalance": orderbook,
    }

    result = await analyzer.process_key(DEFAULT_KEY)

    assert_key_payload(result, DEFAULT_KEY)
    assert result["cvd"] == cvd_result
    assert result["volume_delta"] == volume_result
    assert result["aggressive_trades"] is None
    assert result["orderbook_imbalance"] is None
    assert cvd.processed_keys == [DEFAULT_KEY]
    assert volume_delta.processed_keys == [DEFAULT_KEY]
    assert aggressive.processed_keys == []
    assert orderbook.processed_keys == []


@pytest.mark.asyncio
async def test_facade_process_key_blocks_scope_filtered_market_without_calling_modules() -> None:
    analyzer = make_facade(
        event_bus=FakeEventBus(),
        scheduler=FakeScheduler(),
        config=make_orderflow_config(
            allowed_exchanges={"binance"},
            allowed_market_types={"usdm_futures"},
            allowed_symbols={"BTCUSDT"},
            allowed_timeframes={"1m"},
        ),
    )
    modules = {
        name: StubModule(name=name, result=make_stats(key=DEFAULT_KEY))
        for name in ("cvd", "volume_delta", "aggressive_trades", "orderbook_imbalance")
    }
    analyzer.cvd = modules["cvd"]  # type: ignore[assignment]
    analyzer.volume_delta = modules["volume_delta"]  # type: ignore[assignment]
    analyzer.aggressive_trades = modules["aggressive_trades"]  # type: ignore[assignment]
    analyzer.orderbook_imbalance = modules["orderbook_imbalance"]  # type: ignore[assignment]
    analyzer._modules = modules  # noqa: SLF001

    result = await analyzer.process_key(BYBIT_KEY)

    assert_key_payload(result, BYBIT_KEY)
    assert result["cvd"] is None
    assert result["volume_delta"] is None
    assert result["aggressive_trades"] is None
    assert result["orderbook_imbalance"] is None
    assert all(module.processed_keys == [] for module in modules.values())


@pytest.mark.asyncio
async def test_facade_process_key_emits_error_event_when_one_module_fails() -> None:
    event_bus = FakeEventBus()
    analyzer = make_facade(event_bus=event_bus, scheduler=FakeScheduler())

    cvd = StubModule(name="cvd", result=make_stats(key=DEFAULT_KEY))
    volume_delta = StubModule(name="volume_delta", raise_on_process=True)
    aggressive = StubModule(name="aggressive_trades", result=None)
    orderbook = StubModule(name="orderbook_imbalance", result=None)

    analyzer.cvd = cvd  # type: ignore[assignment]
    analyzer.volume_delta = volume_delta  # type: ignore[assignment]
    analyzer.aggressive_trades = aggressive  # type: ignore[assignment]
    analyzer.orderbook_imbalance = orderbook  # type: ignore[assignment]
    analyzer._modules = {  # noqa: SLF001
        "cvd": cvd,
        "volume_delta": volume_delta,
        "aggressive_trades": aggressive,
        "orderbook_imbalance": orderbook,
    }

    result = await analyzer.process_key(DEFAULT_KEY)

    assert result["cvd"] == cvd.result
    assert result["volume_delta"] is None

    error_events = emitted_events(event_bus, OrderFlowEventTopic.ERROR.value)
    assert len(error_events) == 1
    payload = error_events[0]["payload"]
    assert payload["module"] == "volume_delta"
    assert payload["reason"] == "manual_process_failed"
    assert_key_payload(payload, DEFAULT_KEY)


@pytest.mark.asyncio
async def test_facade_get_latest_stats_by_key_returns_scoped_module_results() -> None:
    analyzer = make_facade(event_bus=FakeEventBus(), scheduler=FakeScheduler())

    cvd_result = make_stats(key=DEFAULT_KEY, metric=OrderFlowMetricType.CVD)
    volume_result = make_stats(
        key=DEFAULT_KEY,
        metric=OrderFlowMetricType.VOLUME_DELTA,
    )
    modules = {
        "cvd": StubModule(name="cvd", result=cvd_result),
        "volume_delta": StubModule(name="volume_delta", result=volume_result),
        "aggressive_trades": StubModule(name="aggressive_trades", result=None),
        "orderbook_imbalance": StubModule(name="orderbook_imbalance", result=None),
    }
    analyzer.cvd = modules["cvd"]  # type: ignore[assignment]
    analyzer.volume_delta = modules["volume_delta"]  # type: ignore[assignment]
    analyzer.aggressive_trades = modules["aggressive_trades"]  # type: ignore[assignment]
    analyzer.orderbook_imbalance = modules["orderbook_imbalance"]  # type: ignore[assignment]
    analyzer._modules = modules  # noqa: SLF001

    result = analyzer.get_latest_stats_by_key(DEFAULT_KEY)

    assert_key_payload(result, DEFAULT_KEY)
    assert result["cvd"] == cvd_result
    assert result["volume_delta"] == volume_result
    assert result["aggressive_trades"] is None
    assert result["orderbook_imbalance"] is None


@pytest.mark.asyncio
async def test_facade_cleanup_runs_all_modules_and_emits_error_for_failed_cleanup() -> None:
    event_bus = FakeEventBus()
    analyzer = make_facade(event_bus=event_bus, scheduler=FakeScheduler())

    cvd = StubModule(name="cvd")
    volume_delta = StubModule(name="volume_delta", raise_on_cleanup=True)
    aggressive = StubModule(name="aggressive_trades")
    orderbook = StubModule(name="orderbook_imbalance")

    analyzer.cvd = cvd  # type: ignore[assignment]
    analyzer.volume_delta = volume_delta  # type: ignore[assignment]
    analyzer.aggressive_trades = aggressive  # type: ignore[assignment]
    analyzer.orderbook_imbalance = orderbook  # type: ignore[assignment]
    analyzer._modules = {  # noqa: SLF001
        "cvd": cvd,
        "volume_delta": volume_delta,
        "aggressive_trades": aggressive,
        "orderbook_imbalance": orderbook,
    }

    await analyzer.cleanup()

    assert cvd.cleanup_calls == 1
    assert volume_delta.cleanup_calls == 1
    assert aggressive.cleanup_calls == 1
    assert orderbook.cleanup_calls == 1

    error_events = emitted_events(event_bus, OrderFlowEventTopic.ERROR.value)
    assert len(error_events) == 1
    assert error_events[0]["payload"]["module"] == "volume_delta"
    assert error_events[0]["payload"]["reason"] == "cleanup_failed"
    assert error_events[0]["payload"]["scope"] == "exchange:market_type:symbol:timeframe"


@pytest.mark.asyncio
async def test_facade_lifecycle_emit_failure_does_not_crash_register() -> None:
    event_bus = FakeEventBus()
    event_bus.fail_emit_topics.add(OrderFlowEventTopic.STARTED.value)

    analyzer = make_facade(event_bus=event_bus, scheduler=FakeScheduler())

    await analyzer.register()

    assert analyzer.is_running is True
    assert emitted_events(event_bus, OrderFlowEventTopic.STARTED.value) == []


@pytest.mark.asyncio
async def test_facade_register_failure_rolls_exception_from_module_registration() -> None:
    analyzer = make_facade(event_bus=FakeEventBus(), scheduler=FakeScheduler())

    bad_module = StubModule(name="cvd", raise_on_register=True)
    analyzer.cvd = bad_module  # type: ignore[assignment]
    analyzer._modules["cvd"] = bad_module  # noqa: SLF001

    with pytest.raises(RuntimeError, match="cvd register failed"):
        await analyzer.register()

    assert analyzer.is_running is False
    assert bad_module.register_calls == 1
