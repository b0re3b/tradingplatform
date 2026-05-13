# tests/analytics/orderflow/test_orderflow_lifecycle.py

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
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
from analytics.orderflow.models import BaseOrderFlowStats, OrderFlowSignal


# ---------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------


@dataclass(slots=True)
class FakeSubscription:
    pattern: str
    handler: Any
    name: str
    enabled: bool = True


class FakeEventBus:
    """
    Strict fake for core.event_bus.EventBus.

    It records subscriptions, unsubscriptions and emitted events.
    It can also simulate EventBus.emit() failures for selected topics.
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

    It deliberately tracks disabled and removed jobs separately.
    This lets tests catch lifecycle leaks where jobs are only disabled
    instead of being removed from the scheduler.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.disabled_job_ids: list[str] = []
        self.removed_job_ids: list[str] = []

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


class FakeTradesCache:
    pass


class FakeOrderbookCache:
    pass


# ---------------------------------------------------------------------
# Dummy analyzer for testing BaseOrderFlowAnalyzer directly
# ---------------------------------------------------------------------


class DummyOrderFlowAnalyzer(BaseOrderFlowAnalyzer):
    def __init__(
        self,
        *,
        event_bus: FakeEventBus,
        scheduler: FakeScheduler | None,
        config: BaseOrderFlowSubConfig | None = None,
        source_topic_patterns: list[str] | tuple[str, ...] = ("market.trade",),
    ) -> None:
        super().__init__(
            event_bus=event_bus,  # type: ignore[arg-type]
            scheduler=scheduler,  # type: ignore[arg-type]
            config=config or make_base_config(),
            metric_type=OrderFlowMetricType.CVD,
            source_type=OrderFlowSourceType.TRADES,
            source_topic_patterns=source_topic_patterns,
            component_module="orderflow_test",
        )
        self.handled_events: list[Event] = []
        self.processed_symbols: list[str] = []
        self.cleaned = False

    async def process_symbol(self, symbol: str) -> BaseOrderFlowStats | None:
        normalized = str(symbol).strip().upper()
        self.processed_symbols.append(normalized)
        return make_stats(symbol=normalized)

    def get_latest_stats(self, symbol: str) -> BaseOrderFlowStats | None:
        return make_stats(symbol=str(symbol).strip().upper())

    async def _handle_event(self, event: Event) -> None:
        self.handled_events.append(event)

    async def cleanup(self) -> None:
        self.cleaned = True


# ---------------------------------------------------------------------
# Facade stub modules
# ---------------------------------------------------------------------


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
        self.processed_symbols: list[str] = []
        self.cleanup_calls = 0

    def register(self) -> None:
        self.register_calls += 1
        if self.raise_on_register:
            raise RuntimeError(f"{self.name} register failed")

    def stop(self) -> None:
        self.stop_calls += 1
        if self.raise_on_stop:
            raise RuntimeError(f"{self.name} stop failed")

    async def process_symbol(self, symbol: str) -> BaseOrderFlowStats | None:
        self.processed_symbols.append(symbol)
        if self.raise_on_process:
            raise RuntimeError(f"{self.name} process failed")
        return self.result

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.raise_on_cleanup:
            raise RuntimeError(f"{self.name} cleanup failed")

    def get_latest_stats(self, symbol: str) -> BaseOrderFlowStats | None:
        return self.result

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "register_calls": self.register_calls,
            "stop_calls": self.stop_calls,
            "cleanup_calls": self.cleanup_calls,
        }


# ---------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------


def make_base_config(**overrides: Any) -> BaseOrderFlowSubConfig:
    values = {
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
        "symbol_allowlist": None,
    }
    values.update(overrides)
    config = BaseOrderFlowSubConfig(**values)
    config.validate()
    return config


def make_orderflow_config(**overrides: Any) -> OrderFlowConfig:
    config = OrderFlowConfig(
        source_topic_patterns_trades=["market.trade"],
        source_topic_patterns_orderbook=["market.orderbook"],
    )

    for key, value in overrides.items():
        setattr(config, key, value)

    config.validate()
    return config


def make_stats(symbol: str = "BTCUSDT") -> BaseOrderFlowStats:
    return BaseOrderFlowStats(
        symbol=str(symbol).strip().upper(),
        metric=OrderFlowMetricType.CVD,
        source_type=OrderFlowSourceType.TRADES,
    )


def make_signal(symbol: str = "BTCUSDT") -> OrderFlowSignal:
    return OrderFlowSignal(
        symbol=symbol,
        metric=OrderFlowMetricType.CVD,
        signal_type=OrderFlowSignalType.BULLISH,
        side=OrderFlowSide.BUY,
        strength=0.85,
        reason="test_signal",
        context={"case": "lifecycle"},
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


# ---------------------------------------------------------------------
# BaseOrderFlowAnalyzer lifecycle tests
# ---------------------------------------------------------------------


def test_base_register_subscribes_to_event_bus_and_schedules_health_cleanup_jobs() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        source_topic_patterns=("market.trade", "market.trade.*"),
    )

    analyzer.register()

    assert analyzer.is_running is True

    assert [sub.pattern for sub in event_bus.subscriptions] == [
        "market.trade",
        "market.trade.*",
    ]
    assert all(sub.handler == analyzer._handle_event for sub in event_bus.subscriptions)
    assert all("DummyOrderFlowAnalyzer" in sub.name for sub in event_bus.subscriptions)

    assert len(scheduler.jobs) == 2

    job_names = {job["name"] for job in scheduler.jobs.values()}
    assert any("health" in name.lower() for name in job_names)
    assert any("cleanup" in name.lower() for name in job_names)

    for job in scheduler.jobs.values():
        assert job["enabled"] is True
        assert job["allow_overlap"] is False
        assert job["max_retries"] == analyzer.stats()["config"]["scheduler_job_max_retries"]
        assert job["retry_delay"] == analyzer.stats()["config"]["scheduler_job_retry_delay_sec"]
        assert job["timeout"] == analyzer.stats()["config"]["scheduler_job_timeout_sec"]


def test_base_register_is_idempotent_and_does_not_duplicate_subscriptions_or_jobs() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        source_topic_patterns=("market.trade", "market.trade.*"),
    )

    analyzer.register()
    analyzer.register()

    assert len(event_bus.subscriptions) == 2
    assert len(scheduler.jobs) == 2
    assert analyzer.is_running is True


def test_base_stop_unsubscribes_all_eventbus_handlers() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        source_topic_patterns=("market.trade", "market.trade.*"),
    )

    analyzer.register()
    created_subscriptions = list(event_bus.subscriptions)

    analyzer.stop()

    assert analyzer.is_running is False
    assert event_bus.subscriptions == []
    assert event_bus.unsubscribed == created_subscriptions


def test_base_stop_should_remove_scheduler_jobs_instead_of_leaving_orphans() -> None:
    """
    Vulnerability test.

    For a long-running trading system, stop()/register() cycles must not leave
    stale disabled jobs inside Scheduler. This test intentionally expects
    Scheduler.remove_job() semantics.

    If current implementation only calls disable_job(), this test should fail
    and force the lifecycle fix.
    """
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        source_topic_patterns=("market.trade",),
    )

    analyzer.register()
    created_job_ids = set(scheduler.jobs)

    analyzer.stop()

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
        source_topic_patterns=("market.trade",),
    )

    analyzer.register()

    assert analyzer.is_running is False
    assert event_bus.subscriptions == []
    assert scheduler.jobs == {}


# ---------------------------------------------------------------------
# BaseOrderFlowAnalyzer emit / extraction tests
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_base_emit_update_publishes_json_safe_payload_and_updates_metrics() -> None:
    event_bus = FakeEventBus()
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=FakeScheduler(),
    )

    await analyzer.emit_update(make_stats("btcusdt"))

    assert len(event_bus.emitted) == 1

    emitted = event_bus.emitted[0]
    assert emitted["topic"] == "analytics.orderflow.dummy.updated"
    assert emitted["source"] == "dummy_orderflow"
    assert emitted["priority"] == EventPriority.HIGH

    payload = emitted["payload"]
    assert payload["symbol"] == "BTCUSDT"
    assert payload["metric"] == "cvd"
    assert payload["source_type"] == "trades"
    assert payload["stats"]["symbol"] == "BTCUSDT"
    assert payload["stats"]["metric"] == "cvd"

    stats = analyzer.stats()
    assert stats["metrics"]["updates_emitted"] == 1
    assert stats["metrics"]["symbols"]["BTCUSDT"]["updates_emitted"] == 1


@pytest.mark.asyncio
async def test_base_emit_signal_throttles_repeated_signal_for_same_symbol() -> None:
    event_bus = FakeEventBus()
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=FakeScheduler(),
        config=make_base_config(min_signal_interval_sec=3600.0),
    )

    await analyzer.emit_signal(make_signal("btcusdt"))
    await analyzer.emit_signal(make_signal("BTCUSDT"))

    signal_events = [
        item for item in event_bus.emitted
        if item["topic"] == "analytics.orderflow.dummy.signal"
    ]

    assert len(signal_events) == 1

    stats = analyzer.stats()
    assert stats["metrics"]["signals_emitted"] == 1
    assert stats["metrics"]["skipped"] == 1
    assert stats["metrics"]["symbols"]["BTCUSDT"]["signals_emitted"] == 1
    assert stats["metrics"]["symbols"]["BTCUSDT"]["skipped"] == 1


@pytest.mark.asyncio
async def test_base_emit_failure_is_captured_as_metric_without_crashing() -> None:
    event_bus = FakeEventBus()
    event_bus.fail_emit_topics.add("analytics.orderflow.dummy.updated")

    analyzer = DummyOrderFlowAnalyzer(
        event_bus=event_bus,
        scheduler=FakeScheduler(),
    )

    await analyzer.emit_update(make_stats("ETHUSDT"))

    assert event_bus.emitted == []

    stats = analyzer.stats()
    assert stats["metrics"]["emit_errors"] == 1
    assert stats["metrics"]["updates_emitted"] == 0


def test_base_extract_symbol_from_flat_and_nested_payloads() -> None:
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=FakeEventBus(),
        scheduler=FakeScheduler(),
    )

    flat_event = Event(
        topic="market.trade",
        payload={"symbol": "btcusdt", "price": 100.0},
    )
    nested_event = Event(
        topic="market.trade",
        payload={"data": {"s": "ethusdt", "price": 200.0}},
    )
    missing_symbol_event = Event(
        topic="market.trade",
        payload={"data": {"price": 300.0}},
    )

    assert analyzer.extract_symbol_from_event(flat_event) == "BTCUSDT"
    assert analyzer.extract_symbol_from_event(nested_event) == "ETHUSDT"
    assert analyzer.extract_symbol_from_event(missing_symbol_event) is None


def test_base_normalize_trade_rejects_malformed_payloads_and_normalizes_valid_aliases() -> None:
    analyzer = DummyOrderFlowAnalyzer(
        event_bus=FakeEventBus(),
        scheduler=FakeScheduler(),
    )

    valid = analyzer.normalize_trade(
        {
            "s": "btcusdt",
            "side": "buy",
            "price": "100.5",
            "qty": "2",
            "timestamp": 123456.0,
            "trade_id": 42,
            "exchange": "binance",
        }
    )

    assert valid is not None
    assert valid.symbol == "BTCUSDT"
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
            "symbol": "BTCUSDT",
            "side": "unknown",
            "price": 100,
            "quantity": 1,
            "timestamp": 1,
        }
    ) is None


# ---------------------------------------------------------------------
# OrderFlowAnalyzer facade lifecycle tests
# ---------------------------------------------------------------------


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

    started_events = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.STARTED.value
    ]
    assert len(started_events) == 1
    assert started_events[0]["source"] == "orderflow_analyzer"
    assert started_events[0]["payload"]["modules"] == ["cvd", "orderbook_imbalance"]


@pytest.mark.asyncio
async def test_facade_register_is_idempotent_and_does_not_double_subscribe_modules() -> None:
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
async def test_facade_register_disabled_package_emits_stopped_event_without_subscriptions() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()

    config = make_orderflow_config()
    config.enabled = False
    config.validate()

    analyzer = make_facade(
        event_bus=event_bus,
        scheduler=scheduler,
        config=config,
    )

    await analyzer.register()

    assert analyzer.is_running is False
    assert event_bus.subscriptions == []
    assert scheduler.jobs == {}

    stopped_events = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.STOPPED.value
    ]
    assert len(stopped_events) == 1
    assert stopped_events[0]["payload"] == {
        "reason": "disabled_by_config",
        "enabled": False,
    }


@pytest.mark.asyncio
async def test_facade_stop_unsubscribes_modules_and_emits_stopped_event() -> None:
    event_bus = FakeEventBus()
    scheduler = FakeScheduler()
    analyzer = make_facade(event_bus=event_bus, scheduler=scheduler)

    await analyzer.register()
    subscriptions_before_stop = list(event_bus.subscriptions)

    await analyzer.stop()

    assert analyzer.is_running is False
    assert event_bus.subscriptions == []
    assert event_bus.unsubscribed == subscriptions_before_stop

    stopped_events = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.STOPPED.value
    ]
    assert len(stopped_events) == 1
    assert stopped_events[0]["payload"]["enabled"] is True
    assert stopped_events[0]["payload"]["modules"] == [
        "orderbook_imbalance",
        "aggressive_trades",
        "volume_delta",
        "cvd",
    ]


@pytest.mark.asyncio
async def test_facade_process_symbol_calls_only_enabled_modules_and_normalizes_symbol() -> None:
    event_bus = FakeEventBus()
    analyzer = make_facade(event_bus=event_bus, scheduler=FakeScheduler())

    analyzer._modules = {
        "cvd": StubModule(name="cvd", result=make_stats("BTCUSDT")),  # type: ignore[dict-item]
        "volume_delta": StubModule(name="volume_delta", result=make_stats("BTCUSDT")),  # type: ignore[dict-item]
        "aggressive_trades": StubModule(name="aggressive_trades", result=make_stats("BTCUSDT")),  # type: ignore[dict-item]
        "orderbook_imbalance": StubModule(name="orderbook_imbalance", result=make_stats("BTCUSDT")),  # type: ignore[dict-item]
    }

    analyzer._config.volume_delta.enabled = False
    analyzer._config.aggressive_trades.enabled = False

    result = await analyzer.process_symbol("  btcusdt ")

    assert result["symbol"] == "BTCUSDT"
    assert result["cvd"] is not None
    assert result["orderbook_imbalance"] is not None
    assert result["volume_delta"] is None
    assert result["aggressive_trades"] is None

    assert analyzer._modules["cvd"].processed_symbols == ["BTCUSDT"]  # type: ignore[attr-defined]
    assert analyzer._modules["orderbook_imbalance"].processed_symbols == ["BTCUSDT"]  # type: ignore[attr-defined]
    assert analyzer._modules["volume_delta"].processed_symbols == []  # type: ignore[attr-defined]
    assert analyzer._modules["aggressive_trades"].processed_symbols == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_facade_process_symbol_emits_error_event_when_enabled_module_raises() -> None:
    event_bus = FakeEventBus()
    analyzer = make_facade(event_bus=event_bus, scheduler=FakeScheduler())

    analyzer._modules = {
        "cvd": StubModule(name="cvd", result=make_stats("BTCUSDT")),  # type: ignore[dict-item]
        "volume_delta": StubModule(name="volume_delta", raise_on_process=True),  # type: ignore[dict-item]
        "aggressive_trades": StubModule(name="aggressive_trades", result=None),  # type: ignore[dict-item]
        "orderbook_imbalance": StubModule(name="orderbook_imbalance", result=None),  # type: ignore[dict-item]
    }

    result = await analyzer.process_symbol("btcusdt")

    assert result["symbol"] == "BTCUSDT"
    assert result["cvd"] is not None
    assert result["volume_delta"] is None

    error_events = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.ERROR.value
    ]
    assert len(error_events) == 1
    assert error_events[0]["priority"] == EventPriority.HIGH
    assert error_events[0]["payload"] == {
        "module": "volume_delta",
        "symbol": "BTCUSDT",
        "reason": "manual_process_failed",
    }


@pytest.mark.asyncio
async def test_facade_cleanup_continues_after_module_failure_and_emits_error_event() -> None:
    event_bus = FakeEventBus()
    analyzer = make_facade(event_bus=event_bus, scheduler=FakeScheduler())

    analyzer._modules = {
        "cvd": StubModule(name="cvd"),
        "volume_delta": StubModule(name="volume_delta", raise_on_cleanup=True),
        "aggressive_trades": StubModule(name="aggressive_trades"),
        "orderbook_imbalance": StubModule(name="orderbook_imbalance"),
    }

    await analyzer.cleanup()

    assert analyzer._modules["cvd"].cleanup_calls == 1  # type: ignore[attr-defined]
    assert analyzer._modules["volume_delta"].cleanup_calls == 1  # type: ignore[attr-defined]
    assert analyzer._modules["aggressive_trades"].cleanup_calls == 1  # type: ignore[attr-defined]
    assert analyzer._modules["orderbook_imbalance"].cleanup_calls == 1  # type: ignore[attr-defined]

    error_events = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.ERROR.value
    ]
    assert len(error_events) == 1
    assert error_events[0]["payload"] == {
        "module": "volume_delta",
        "reason": "cleanup_failed",
    }


@pytest.mark.asyncio
async def test_facade_lifecycle_emit_failure_does_not_crash_register_or_stop() -> None:
    event_bus = FakeEventBus()
    event_bus.fail_emit_topics.update(
        {
            OrderFlowEventTopic.STARTED.value,
            OrderFlowEventTopic.STOPPED.value,
        }
    )

    analyzer = make_facade(event_bus=event_bus, scheduler=FakeScheduler())

    await analyzer.register()
    assert analyzer.is_running is True

    await analyzer.stop()
    assert analyzer.is_running is False

    assert event_bus.emitted == []


def test_facade_stats_exposes_core_integration_state() -> None:
    analyzer = make_facade(
        event_bus=FakeEventBus(),
        scheduler=FakeScheduler(),
    )

    stats = analyzer.stats()

    assert stats["running"] is False
    assert stats["enabled"] is True
    assert stats["scheduler_attached"] is True
    assert stats["trades_topic_patterns"] == ["market.trade"]
    assert stats["orderbook_topic_patterns"] == ["market.orderbook"]

    assert set(stats["modules"]) == {
        "cvd",
        "volume_delta",
        "aggressive_trades",
        "orderbook_imbalance",
    }


def test_facade_rejects_empty_symbol_before_calling_modules() -> None:
    analyzer = make_facade(
        event_bus=FakeEventBus(),
        scheduler=FakeScheduler(),
    )

    with pytest.raises(ValueError, match="symbol must not be empty"):
        analyzer.get_latest_stats("   ")