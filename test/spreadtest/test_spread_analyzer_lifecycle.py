from __future__ import annotations

import asyncio
import inspect
from decimal import Decimal
from typing import Any, Callable

import pytest
import pytest_asyncio

from analytics.spreads import (
    CrossExchangeSpreadConfig,
    SpreadAnalyzer,
    SpotFuturesSpreadConfig,
)
from core.event_bus import Event, EventBus
from core.scheduler import Scheduler


pytestmark = pytest.mark.asyncio


# ============================================================
# Constants
# ============================================================

ORDERBOOK_TOPIC = "market.orderbook.updated"
FUNDING_TOPIC = "market.funding.updated"

STARTED_TOPIC = "analytics.spreads.analyzer.started"
STOPPED_TOPIC = "analytics.spreads.analyzer.stopped"

SPOT_SERVICE_NAME = "test_spot_futures_spread"
CROSS_SERVICE_NAME = "test_cross_exchange_spread"

EXPECTED_CHILD_ANALYZERS = {
    "SpotFuturesSpreadAnalyzer",
    "CrossExchangeSpreadAnalyzer",
}


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


async def _drain_event_bus() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0.01)


# ============================================================
# Scheduler inspection helpers
# ============================================================

async def _scheduler_jobs(scheduler: Scheduler) -> list[Any]:
    list_jobs = getattr(scheduler, "list_jobs", None)
    if callable(list_jobs):
        jobs = await _maybe_await(list_jobs())

        if isinstance(jobs, dict):
            return list(jobs.values())

        return list(jobs)

    jobs = getattr(scheduler, "_jobs", None)
    if isinstance(jobs, dict):
        return list(jobs.values())

    if jobs is not None:
        return list(jobs)

    return []


def _job_name(job: Any) -> str | None:
    if isinstance(job, dict):
        return job.get("name")

    return getattr(job, "name", None)


def _job_id(job: Any) -> str | None:
    if isinstance(job, dict):
        return job.get("job_id") or job.get("id")

    return (
        getattr(job, "job_id", None)
        or getattr(job, "id", None)
    )


def _job_enabled(job: Any) -> bool | None:
    if isinstance(job, dict):
        value = job.get("enabled")
        if value is not None:
            return bool(value)

        status = job.get("status")
        if status is not None:
            return "disabled" not in str(status).lower()

        return None

    value = getattr(job, "enabled", None)
    if value is not None:
        return bool(value)

    status = getattr(job, "status", None)
    if status is not None:
        return "disabled" not in str(status).lower()

    return None


async def _scheduler_job_names(scheduler: Scheduler) -> list[str]:
    jobs = await _scheduler_jobs(scheduler)
    return [
        name
        for job in jobs
        if (name := _job_name(job)) is not None
    ]


def _assert_no_duplicate_names(names: list[str]) -> None:
    duplicates = {
        name
        for name in names
        if names.count(name) > 1
    }
    assert not duplicates, f"Duplicate scheduler job names detected: {duplicates}"


# ============================================================
# Analyzer inspection helpers
# ============================================================

def _subscription_identity(subscription: Any) -> Any:
    return (
        getattr(subscription, "subscription_id", None)
        or getattr(subscription, "id", None)
        or id(subscription)
    )


def _child_subscription_ids(analyzer: SpreadAnalyzer) -> dict[str, list[Any]]:
    return {
        "spot_futures": [
            _subscription_identity(subscription)
            for subscription in analyzer.spot_futures._subscriptions
        ],
        "cross_exchange": [
            _subscription_identity(subscription)
            for subscription in analyzer.cross_exchange._subscriptions
        ],
    }


def _child_subscription_counts(analyzer: SpreadAnalyzer) -> dict[str, int]:
    return {
        "spot_futures": len(analyzer.spot_futures._subscriptions),
        "cross_exchange": len(analyzer.cross_exchange._subscriptions),
    }


def _child_scheduler_job_ids(analyzer: SpreadAnalyzer) -> dict[str, list[str]]:
    return {
        "spot_futures": list(analyzer.spot_futures._scheduler_job_ids),
        "cross_exchange": list(analyzer.cross_exchange._scheduler_job_ids),
    }


def _assert_facade_consistency(analyzer: SpreadAnalyzer) -> None:
    assert analyzer.is_registered == (
        analyzer.spot_futures.is_registered
        and analyzer.cross_exchange.is_registered
    )

    assert analyzer.is_running == (
        analyzer.spot_futures.is_running
        and analyzer.cross_exchange.is_running
    )


# ============================================================
# Event recording
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

    def analyzer_names(self) -> set[str]:
        names: set[str] = set()
        for payload in self.payloads():
            if isinstance(payload, dict) and "analyzer" in payload:
                names.add(str(payload["analyzer"]))
        return names


# ============================================================
# Config factories
# ============================================================

def _spot_config(**overrides: Any) -> SpotFuturesSpreadConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "service_name": SPOT_SERVICE_NAME,
        "orderbook_event_topic": ORDERBOOK_TOPIC,
        "orderbook_event_topic_patterns": (ORDERBOOK_TOPIC,),
        "funding_event_topic": FUNDING_TOPIC,
        "funding_event_topic_patterns": (FUNDING_TOPIC,),
        "allow_legacy_quote_topics": False,
        "allow_legacy_raw_topics": False,
        "max_quote_age_ms": 60_000,
        "max_quote_skew_ms": 60_000,
        "rolling_window_size": 10,
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
        "widening_bps_threshold": Decimal("5"),
        "mean_reversion_zscore_threshold": Decimal("2.0"),
        "regime_shift_zscore_threshold": Decimal("3.0"),
        "metadata": {"test": "spread_analyzer_lifecycle"},
    }
    values.update(overrides)
    return SpotFuturesSpreadConfig(**values)


def _cross_config(**overrides: Any) -> CrossExchangeSpreadConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "service_name": CROSS_SERVICE_NAME,
        "orderbook_event_topic": ORDERBOOK_TOPIC,
        "orderbook_event_topic_patterns": (ORDERBOOK_TOPIC,),
        "allow_legacy_quote_topics": False,
        "allow_legacy_raw_topics": False,
        "max_quote_age_ms": 60_000,
        "max_quote_skew_ms": 60_000,
        "rolling_window_size": 10,
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
        "widening_bps_threshold": Decimal("5"),
        "arbitrage_min_bps": Decimal("1"),
        "default_trade_size": Decimal("1"),
        "slippage_max_bps": Decimal("0"),
        "safety_buffer_bps": Decimal("0"),
        "default_taker_fee_rate": Decimal("0"),
        "default_maker_fee_rate": Decimal("0"),
        "opportunity_ttl_seconds": 60.0,
        "max_cached_opportunities": 10_000,
        "metadata": {"test": "spread_analyzer_lifecycle"},
    }
    values.update(overrides)
    return CrossExchangeSpreadConfig(**values)


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
) -> SpreadAnalyzer:
    instance = SpreadAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        spot_futures_config=_spot_config(),
        cross_exchange_config=_cross_config(),
    )

    try:
        yield instance
    finally:
        if instance.is_running:
            await instance.stop()

        if instance.is_registered:
            instance.unregister()


# ============================================================
# Initial state / construction
# ============================================================

async def test_facade_initializes_children_with_real_infrastructure(
    analyzer: SpreadAnalyzer,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    assert analyzer.is_running is False
    assert analyzer.is_registered is False

    assert analyzer.spot_futures is not None
    assert analyzer.cross_exchange is not None

    assert analyzer.spot_futures.is_running is False
    assert analyzer.cross_exchange.is_running is False

    assert analyzer.spot_futures.is_registered is False
    assert analyzer.cross_exchange.is_registered is False

    assert analyzer.spot_futures._event_bus is event_bus
    assert analyzer.cross_exchange._event_bus is event_bus

    assert analyzer.spot_futures._scheduler is scheduler
    assert analyzer.cross_exchange._scheduler is scheduler

    assert analyzer.spot_futures._config.service_name == SPOT_SERVICE_NAME
    assert analyzer.cross_exchange._config.service_name == CROSS_SERVICE_NAME

    assert analyzer.spot_futures._config.orderbook_event_topic == ORDERBOOK_TOPIC
    assert analyzer.cross_exchange._config.orderbook_event_topic == ORDERBOOK_TOPIC
    assert analyzer.spot_futures._config.funding_event_topic == FUNDING_TOPIC

    stats = analyzer.get_stats()

    assert stats["running"] is False
    assert stats["registered"] is False
    assert stats["price_input_source"] == ORDERBOOK_TOPIC
    assert stats["funding_input_source"] == FUNDING_TOPIC
    assert stats["uses_quote_cache"] is False

    assert "OrderBookCache" in stats["production_flow"]["price"]
    assert "FundingCache" in stats["production_flow"]["funding"]

    assert stats["spot_futures"]["running"] is False
    assert stats["spot_futures"]["registered"] is False
    assert stats["cross_exchange"]["running"] is False
    assert stats["cross_exchange"]["registered"] is False

    assert stats["configs"]["spot_futures_price_topics"] == [ORDERBOOK_TOPIC]
    assert stats["configs"]["cross_exchange_price_topics"] == [ORDERBOOK_TOPIC]


async def test_auto_register_constructor_registers_children_once(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    instance = SpreadAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        spot_futures_config=_spot_config(),
        cross_exchange_config=_cross_config(),
        auto_register=True,
    )

    try:
        assert instance.is_registered is True
        assert instance.spot_futures.is_registered is True
        assert instance.cross_exchange.is_registered is True

        assert _child_subscription_counts(instance) == {
            "spot_futures": 2,
            "cross_exchange": 1,
        }

        first_ids = _child_subscription_ids(instance)

        instance.register()

        assert _child_subscription_counts(instance) == {
            "spot_futures": 2,
            "cross_exchange": 1,
        }
        assert _child_subscription_ids(instance) == first_ids
    finally:
        if instance.is_running:
            await instance.stop()
        if instance.is_registered:
            instance.unregister()


async def test_can_disable_individual_child_analyzers_at_facade_level(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    only_cross = SpreadAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        spot_futures_config=_spot_config(),
        cross_exchange_config=_cross_config(),
        enable_spot_futures=False,
        enable_cross_exchange=True,
    )

    try:
        assert only_cross.spot_futures_enabled is False
        assert only_cross.cross_exchange_enabled is True

        with pytest.raises(RuntimeError):
            _ = only_cross.spot_futures

        only_cross.register()

        assert only_cross.is_registered is True
        assert only_cross.cross_exchange.is_registered is True
        assert len(only_cross.cross_exchange._subscriptions) == 1

        await only_cross.start()

        assert only_cross.is_running is True
        assert only_cross.cross_exchange.is_running is True
    finally:
        if only_cross.is_running:
            await only_cross.stop()
        if only_cross.is_registered:
            only_cross.unregister()

    only_spot = SpreadAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        spot_futures_config=_spot_config(),
        cross_exchange_config=_cross_config(),
        enable_spot_futures=True,
        enable_cross_exchange=False,
    )

    try:
        assert only_spot.spot_futures_enabled is True
        assert only_spot.cross_exchange_enabled is False

        with pytest.raises(RuntimeError):
            _ = only_spot.cross_exchange

        only_spot.register()

        assert only_spot.is_registered is True
        assert only_spot.spot_futures.is_registered is True
        assert len(only_spot.spot_futures._subscriptions) == 2

        await only_spot.start()

        assert only_spot.is_running is True
        assert only_spot.spot_futures.is_running is True
    finally:
        if only_spot.is_running:
            await only_spot.stop()
        if only_spot.is_registered:
            only_spot.unregister()


# ============================================================
# Registration / unregistration
# ============================================================

async def test_register_is_idempotent_and_does_not_duplicate_eventbus_subscriptions(
    analyzer: SpreadAnalyzer,
) -> None:
    analyzer.register()

    assert analyzer.is_registered is True
    assert analyzer.spot_futures.is_registered is True
    assert analyzer.cross_exchange.is_registered is True

    assert _child_subscription_counts(analyzer) == {
        "spot_futures": 2,
        "cross_exchange": 1,
    }

    first_subscription_ids = _child_subscription_ids(analyzer)

    analyzer.register()
    analyzer.register()

    assert _child_subscription_counts(analyzer) == {
        "spot_futures": 2,
        "cross_exchange": 1,
    }
    assert _child_subscription_ids(analyzer) == first_subscription_ids

    _assert_facade_consistency(analyzer)


async def test_unregister_removes_all_child_eventbus_subscriptions(
    analyzer: SpreadAnalyzer,
) -> None:
    analyzer.register()

    assert _child_subscription_counts(analyzer) == {
        "spot_futures": 2,
        "cross_exchange": 1,
    }

    analyzer.unregister()

    assert analyzer.is_registered is False
    assert analyzer.spot_futures.is_registered is False
    assert analyzer.cross_exchange.is_registered is False

    assert _child_subscription_counts(analyzer) == {
        "spot_futures": 0,
        "cross_exchange": 0,
    }

    analyzer.unregister()

    assert analyzer.is_registered is False
    assert _child_subscription_counts(analyzer) == {
        "spot_futures": 0,
        "cross_exchange": 0,
    }


async def test_unregister_after_stop_fully_detaches_facade_from_eventbus(
    analyzer: SpreadAnalyzer,
) -> None:
    await analyzer.start()

    assert analyzer.is_running is True
    assert analyzer.is_registered is True

    await analyzer.stop()

    assert analyzer.is_running is False
    assert analyzer.is_registered is True

    analyzer.unregister()

    assert analyzer.is_registered is False
    assert analyzer.spot_futures.is_registered is False
    assert analyzer.cross_exchange.is_registered is False

    assert analyzer.spot_futures._subscriptions == []
    assert analyzer.cross_exchange._subscriptions == []


async def test_shutdown_stops_and_unregisters_everything(
    analyzer: SpreadAnalyzer,
) -> None:
    await analyzer.start()

    assert analyzer.is_running is True
    assert analyzer.is_registered is True

    await analyzer.shutdown()

    assert analyzer.is_running is False
    assert analyzer.is_registered is False

    assert analyzer.spot_futures.is_running is False
    assert analyzer.cross_exchange.is_running is False

    assert analyzer.spot_futures.is_registered is False
    assert analyzer.cross_exchange.is_registered is False

    assert analyzer.spot_futures._subscriptions == []
    assert analyzer.cross_exchange._subscriptions == []


# ============================================================
# Start / stop lifecycle events
# ============================================================

async def test_start_auto_registers_starts_children_and_emits_started_events(
    analyzer: SpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    recorder = EventRecorder()
    subscription = await _subscribe(
        event_bus,
        STARTED_TOPIC,
        recorder.handler,
        name="test.started.recorder",
    )

    try:
        await analyzer.start()
        await recorder.wait_for_count(2)

        assert analyzer.is_running is True
        assert analyzer.is_registered is True

        assert analyzer.spot_futures.is_running is True
        assert analyzer.cross_exchange.is_running is True

        assert analyzer.spot_futures.is_registered is True
        assert analyzer.cross_exchange.is_registered is True

        assert recorder.analyzer_names() == EXPECTED_CHILD_ANALYZERS

        payloads = recorder.payloads()
        assert all(isinstance(payload, dict) for payload in payloads)

        for payload in payloads:
            assert payload["analyzer"] in EXPECTED_CHILD_ANALYZERS
            assert payload["service_name"] in {
                SPOT_SERVICE_NAME,
                CROSS_SERVICE_NAME,
            }
            assert payload["scope"] == "exchange:market_type:symbol:timeframe"
            assert ORDERBOOK_TOPIC in payload["production_price_input_topics"]
            assert ORDERBOOK_TOPIC in payload["production_input_topics"]

        _assert_facade_consistency(analyzer)
    finally:
        await _unsubscribe(event_bus, subscription)


async def test_stop_stops_children_but_preserves_registration_and_emits_stopped_events(
    analyzer: SpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    recorder = EventRecorder()
    subscription = await _subscribe(
        event_bus,
        STOPPED_TOPIC,
        recorder.handler,
        name="test.stopped.recorder",
    )

    try:
        await analyzer.stop()
        await recorder.wait_for_count(2)

        assert analyzer.is_running is False
        assert analyzer.is_registered is True

        assert analyzer.spot_futures.is_running is False
        assert analyzer.cross_exchange.is_running is False

        assert analyzer.spot_futures.is_registered is True
        assert analyzer.cross_exchange.is_registered is True

        assert recorder.analyzer_names() == EXPECTED_CHILD_ANALYZERS

        for payload in recorder.payloads():
            assert isinstance(payload, dict)
            assert payload["analyzer"] in EXPECTED_CHILD_ANALYZERS
            assert payload["service_name"] in {
                SPOT_SERVICE_NAME,
                CROSS_SERVICE_NAME,
            }
            assert isinstance(payload["stats"], dict)

        _assert_facade_consistency(analyzer)
    finally:
        await _unsubscribe(event_bus, subscription)


async def test_stop_when_already_stopped_does_not_emit_child_stopped_events(
    analyzer: SpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    recorder = EventRecorder()
    subscription = await _subscribe(
        event_bus,
        STOPPED_TOPIC,
        recorder.handler,
        name="test.stopped.noop.recorder",
    )

    try:
        await analyzer.stop()
        await recorder.assert_no_new_events(0)

        assert analyzer.is_running is False
        assert analyzer.is_registered is False
        assert analyzer.spot_futures.is_running is False
        assert analyzer.cross_exchange.is_running is False
    finally:
        await _unsubscribe(event_bus, subscription)


async def test_double_start_does_not_emit_duplicate_started_events_or_duplicate_jobs(
    analyzer: SpreadAnalyzer,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    await analyzer.start()
    await _drain_event_bus()

    first_subscription_ids = _child_subscription_ids(analyzer)
    first_job_ids = _child_scheduler_job_ids(analyzer)
    first_job_names = await _scheduler_job_names(scheduler)

    recorder = EventRecorder()
    subscription = await _subscribe(
        event_bus,
        STARTED_TOPIC,
        recorder.handler,
        name="test.started.double_start.recorder",
    )

    try:
        await analyzer.start()
        await analyzer.start()
        await recorder.assert_no_new_events(0)

        assert _child_subscription_ids(analyzer) == first_subscription_ids
        assert _child_scheduler_job_ids(analyzer) == first_job_ids

        second_job_names = await _scheduler_job_names(scheduler)

        assert second_job_names == first_job_names
        _assert_no_duplicate_names(second_job_names)

        assert analyzer.is_running is True
        assert analyzer.spot_futures.is_running is True
        assert analyzer.cross_exchange.is_running is True
    finally:
        await _unsubscribe(event_bus, subscription)


# ============================================================
# Scheduler lifecycle integration
# ============================================================

async def test_start_registers_expected_real_scheduler_jobs_for_both_children(
    analyzer: SpreadAnalyzer,
    scheduler: Scheduler,
) -> None:
    jobs_before = await _scheduler_jobs(scheduler)
    names_before = {
        name
        for job in jobs_before
        if (name := _job_name(job)) is not None
    }

    await analyzer.start()

    jobs_after = await _scheduler_jobs(scheduler)
    names_after = [
        name
        for job in jobs_after
        if (name := _job_name(job)) is not None
    ]

    expected_names = {
        f"{SPOT_SERVICE_NAME}.cleanup",
        f"{SPOT_SERVICE_NAME}.heartbeat",
        f"{CROSS_SERVICE_NAME}.cleanup",
        f"{CROSS_SERVICE_NAME}.heartbeat",
    }

    assert expected_names.issubset(set(names_after))
    assert expected_names.isdisjoint(names_before)

    _assert_no_duplicate_names(names_after)

    assert len(analyzer.spot_futures._scheduler_job_ids) == 2
    assert len(analyzer.cross_exchange._scheduler_job_ids) == 2

    child_job_ids = {
        *analyzer.spot_futures._scheduler_job_ids,
        *analyzer.cross_exchange._scheduler_job_ids,
    }

    actual_job_ids = {
        job_id
        for job in jobs_after
        if (job_id := _job_id(job)) is not None
    }

    assert child_job_ids.issubset(actual_job_ids)


async def test_stop_removes_or_disables_scheduler_jobs_and_restart_does_not_duplicate(
    analyzer: SpreadAnalyzer,
    scheduler: Scheduler,
) -> None:
    await analyzer.start()

    first_job_names = await _scheduler_job_names(scheduler)
    _assert_no_duplicate_names(first_job_names)

    expected_names = {
        f"{SPOT_SERVICE_NAME}.cleanup",
        f"{SPOT_SERVICE_NAME}.heartbeat",
        f"{CROSS_SERVICE_NAME}.cleanup",
        f"{CROSS_SERVICE_NAME}.heartbeat",
    }

    assert expected_names.issubset(set(first_job_names))

    await analyzer.stop()

    jobs_after_stop = await _scheduler_jobs(scheduler)
    names_after_stop = [
        name
        for job in jobs_after_stop
        if (name := _job_name(job)) is not None
    ]

    # Scheduler implementation may remove jobs or disable them.
    for job in jobs_after_stop:
        name = _job_name(job)
        if name in expected_names:
            enabled = _job_enabled(job)
            assert enabled is False or enabled is None

    assert analyzer.spot_futures._scheduler_job_ids == []
    assert analyzer.cross_exchange._scheduler_job_ids == []

    await analyzer.start()

    names_after_restart = await _scheduler_job_names(scheduler)

    assert expected_names.issubset(set(names_after_restart))
    _assert_no_duplicate_names(names_after_restart)

    assert len(analyzer.spot_futures._scheduler_job_ids) == 2
    assert len(analyzer.cross_exchange._scheduler_job_ids) == 2

    # If stop() removed jobs, names_after_stop may not contain expected names.
    # If stop() disabled jobs, restart should not create duplicate names.
    assert names_after_restart.count(f"{SPOT_SERVICE_NAME}.cleanup") == 1
    assert names_after_restart.count(f"{SPOT_SERVICE_NAME}.heartbeat") == 1
    assert names_after_restart.count(f"{CROSS_SERVICE_NAME}.cleanup") == 1
    assert names_after_restart.count(f"{CROSS_SERVICE_NAME}.heartbeat") == 1


# ============================================================
# Disabled child config behavior
# ============================================================

async def test_disabled_child_configs_do_not_start_or_register_scheduler_jobs(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    instance = SpreadAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        spot_futures_config=_spot_config(enabled=False),
        cross_exchange_config=_cross_config(enabled=False),
    )

    recorder = EventRecorder()
    subscription = await _subscribe(
        event_bus,
        STARTED_TOPIC,
        recorder.handler,
        name="test.disabled.started.recorder",
    )

    try:
        await instance.start()
        await recorder.assert_no_new_events(0)

        # Facade currently marks itself running after delegating start(),
        # even if children skipped because config.enabled=False.
        # The important child contract is stricter:
        # disabled children must not run and must not add scheduler jobs.
        assert instance.spot_futures.is_running is False
        assert instance.cross_exchange.is_running is False

        assert instance.spot_futures._scheduler_job_ids == []
        assert instance.cross_exchange._scheduler_job_ids == []

        stats = instance.get_stats()

        assert stats["spot_futures"]["running"] is False
        assert stats["cross_exchange"]["running"] is False
        assert stats["configs"]["spot_futures_enabled"] is False
        assert stats["configs"]["cross_exchange_enabled"] is False
        assert stats["uses_quote_cache"] is False
        assert stats["price_input_source"] == ORDERBOOK_TOPIC
    finally:
        if instance.is_running:
            await instance.stop()
        if instance.is_registered:
            instance.unregister()
        await _unsubscribe(event_bus, subscription)


# ============================================================
# Facade stats contract
# ============================================================

async def test_facade_stats_expose_new_orderbook_flow_contract(
    analyzer: SpreadAnalyzer,
) -> None:
    stats = analyzer.get_stats()

    assert stats["running"] is False
    assert stats["registered"] is False
    assert stats["scope"] == "exchange:market_type:symbol:timeframe"

    assert stats["price_input_source"] == ORDERBOOK_TOPIC
    assert stats["funding_input_source"] == FUNDING_TOPIC
    assert stats["uses_quote_cache"] is False

    assert stats["production_flow"]["price"] == (
        "exchange adapters -> market.orderbook -> "
        "OrderBookCache -> market.orderbook.updated -> spreads"
    )
    assert stats["production_flow"]["funding"] == (
        "exchange adapters -> market.funding -> "
        "FundingCache -> market.funding.updated -> spreads"
    )

    assert stats["enabled_components"] == {
        "spot_futures": True,
        "cross_exchange": True,
    }

    assert stats["configs"]["spot_futures_topics"] == [
        ORDERBOOK_TOPIC,
        FUNDING_TOPIC,
    ]
    assert stats["configs"]["spot_futures_price_topics"] == [ORDERBOOK_TOPIC]
    assert stats["configs"]["cross_exchange_topics"] == [
        ORDERBOOK_TOPIC,
        FUNDING_TOPIC,
    ]
    assert stats["configs"]["cross_exchange_price_topics"] == [ORDERBOOK_TOPIC]

    assert stats["spot_futures"]["price_input_source"] == ORDERBOOK_TOPIC
    assert stats["spot_futures"]["funding_input_source"] == FUNDING_TOPIC
    assert stats["cross_exchange"]["price_input_source"] == ORDERBOOK_TOPIC


async def test_facade_stats_after_start_and_stop_remain_consistent(
    analyzer: SpreadAnalyzer,
) -> None:
    await analyzer.start()

    started_stats = analyzer.get_stats()

    assert started_stats["running"] is True
    assert started_stats["registered"] is True
    assert started_stats["spot_futures"]["running"] is True
    assert started_stats["cross_exchange"]["running"] is True
    assert started_stats["uses_quote_cache"] is False
    assert started_stats["price_input_source"] == ORDERBOOK_TOPIC

    await analyzer.stop()

    stopped_stats = analyzer.get_stats()

    assert stopped_stats["running"] is False
    assert stopped_stats["registered"] is True
    assert stopped_stats["spot_futures"]["running"] is False
    assert stopped_stats["cross_exchange"]["running"] is False
    assert stopped_stats["spot_futures"]["registered"] is True
    assert stopped_stats["cross_exchange"]["registered"] is True
    assert stopped_stats["uses_quote_cache"] is False
    assert stopped_stats["price_input_source"] == ORDERBOOK_TOPIC

    _assert_facade_consistency(analyzer)