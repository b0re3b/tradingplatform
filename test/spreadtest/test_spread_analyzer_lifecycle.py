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
#
# These tests intentionally use the real core.EventBus and core.Scheduler.
# The adapters below only protect the test file from minor constructor
# differences while the core layer evolves.
#
# If your constructors are already stable, you can simplify these helpers to:
#
#   return EventBus()
#   return Scheduler(event_bus=event_bus)
#


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
    """
    Gives the real EventBus worker/queue a chance to dispatch async events.
    """
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
        "metadata": {"test": "spread_analyzer_lifecycle"},
    }
    values.update(overrides)
    return SpotFuturesSpreadConfig(**values)


def _cross_config(**overrides: Any) -> CrossExchangeSpreadConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "service_name": CROSS_SERVICE_NAME,
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

    stats = analyzer.get_stats()

    assert stats["running"] is False
    assert stats["registered"] is False
    assert stats["spot_futures"]["running"] is False
    assert stats["spot_futures"]["registered"] is False
    assert stats["cross_exchange"]["running"] is False
    assert stats["cross_exchange"]["registered"] is False


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


# ============================================================
# Start / stop lifecycle
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
    scheduler_job_ids = {
        job_id
        for job in jobs_after
        if (job_id := _job_id(job)) is not None
    }

    assert child_job_ids.issubset(scheduler_job_ids)


async def test_restart_cycle_does_not_duplicate_eventbus_subscriptions_or_scheduler_jobs(
    analyzer: SpreadAnalyzer,
    scheduler: Scheduler,
) -> None:
    await analyzer.start()

    first_subscription_ids = _child_subscription_ids(analyzer)
    first_subscription_counts = _child_subscription_counts(analyzer)
    first_job_names = await _scheduler_job_names(scheduler)

    assert first_subscription_counts == {
        "spot_futures": 2,
        "cross_exchange": 1,
    }
    _assert_no_duplicate_names(first_job_names)

    await analyzer.stop()

    assert analyzer.is_running is False
    assert analyzer.is_registered is True

    await analyzer.start()

    second_subscription_ids = _child_subscription_ids(analyzer)
    second_subscription_counts = _child_subscription_counts(analyzer)
    second_job_names = await _scheduler_job_names(scheduler)

    assert second_subscription_counts == first_subscription_counts
    assert second_subscription_ids == first_subscription_ids

    _assert_no_duplicate_names(second_job_names)

    expected_names = {
        f"{SPOT_SERVICE_NAME}.cleanup",
        f"{SPOT_SERVICE_NAME}.heartbeat",
        f"{CROSS_SERVICE_NAME}.cleanup",
        f"{CROSS_SERVICE_NAME}.heartbeat",
    }

    for expected_name in expected_names:
        assert second_job_names.count(expected_name) == 1, (
            f"Scheduler job was duplicated after restart: {expected_name}"
        )


async def test_disabled_configs_still_start_lifecycle_but_register_disabled_scheduler_jobs(
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    instance = SpreadAnalyzer(
        event_bus=event_bus,
        scheduler=scheduler,
        spot_futures_config=_spot_config(
            enabled=False,
            service_name="test_disabled_spot_futures_spread",
        ),
        cross_exchange_config=_cross_config(
            enabled=False,
            service_name="test_disabled_cross_exchange_spread",
        ),
    )

    try:
        await instance.start()

        assert instance.is_running is True
        assert instance.is_registered is True
        assert instance.spot_futures.is_running is True
        assert instance.cross_exchange.is_running is True

        assert instance.spot_futures.get_stats()["enabled"] is False
        assert instance.cross_exchange.get_stats()["enabled"] is False

        job_names = await _scheduler_job_names(scheduler)

        expected_names = {
            "test_disabled_spot_futures_spread.cleanup",
            "test_disabled_spot_futures_spread.heartbeat",
            "test_disabled_cross_exchange_spread.cleanup",
            "test_disabled_cross_exchange_spread.heartbeat",
        }

        assert expected_names.issubset(set(job_names))

        jobs = await _scheduler_jobs(scheduler)
        expected_jobs = [
            job
            for job in jobs
            if _job_name(job) in expected_names
        ]

        assert len(expected_jobs) == 4

        # If Scheduler exposes enabled/status, disabled configs should not
        # silently create enabled periodic jobs.
        exposed_enabled_flags = [
            enabled
            for job in expected_jobs
            if (enabled := _job_enabled(job)) is not None
        ]

        if exposed_enabled_flags:
            assert exposed_enabled_flags == [False] * len(exposed_enabled_flags)
    finally:
        if instance.is_running:
            await instance.stop()
        if instance.is_registered:
            instance.unregister()


# ============================================================
# Shutdown contract
# ============================================================


async def test_shutdown_stops_running_analyzer_and_unregisters_all_children(
    analyzer: SpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    started_recorder = EventRecorder()
    stopped_recorder = EventRecorder()

    started_subscription = await _subscribe(
        event_bus,
        STARTED_TOPIC,
        started_recorder.handler,
        name="test.shutdown.started.recorder",
    )
    stopped_subscription = await _subscribe(
        event_bus,
        STOPPED_TOPIC,
        stopped_recorder.handler,
        name="test.shutdown.stopped.recorder",
    )

    try:
        await analyzer.start()
        await started_recorder.wait_for_count(2)

        assert analyzer.is_running is True
        assert analyzer.is_registered is True

        await analyzer.shutdown()
        await stopped_recorder.wait_for_count(2)

        assert analyzer.is_running is False
        assert analyzer.is_registered is False

        assert analyzer.spot_futures.is_running is False
        assert analyzer.cross_exchange.is_running is False

        assert analyzer.spot_futures.is_registered is False
        assert analyzer.cross_exchange.is_registered is False

        assert analyzer.spot_futures._subscriptions == []
        assert analyzer.cross_exchange._subscriptions == []

        assert stopped_recorder.analyzer_names() == EXPECTED_CHILD_ANALYZERS
    finally:
        await _unsubscribe(event_bus, started_subscription)
        await _unsubscribe(event_bus, stopped_subscription)


async def test_shutdown_is_idempotent_after_completed_shutdown(
    analyzer: SpreadAnalyzer,
    event_bus: EventBus,
) -> None:
    await analyzer.start()

    recorder = EventRecorder()
    subscription = await _subscribe(
        event_bus,
        STOPPED_TOPIC,
        recorder.handler,
        name="test.shutdown.idempotent.recorder",
    )

    try:
        await analyzer.shutdown()

        await recorder.wait_for_count(2)

        assert analyzer.is_running is False
        assert analyzer.is_registered is False

        assert analyzer.spot_futures.is_running is False
        assert analyzer.cross_exchange.is_running is False

        assert analyzer.spot_futures.is_registered is False
        assert analyzer.cross_exchange.is_registered is False

        assert analyzer.spot_futures._subscriptions == []
        assert analyzer.cross_exchange._subscriptions == []

        assert recorder.analyzer_names() == EXPECTED_CHILD_ANALYZERS

        events_after_first_shutdown = len(recorder.events)

        await analyzer.shutdown()
        await analyzer.shutdown()

        await recorder.assert_no_new_events(events_after_first_shutdown)

        assert analyzer.is_running is False
        assert analyzer.is_registered is False

        assert analyzer.spot_futures.is_running is False
        assert analyzer.cross_exchange.is_running is False

        assert analyzer.spot_futures.is_registered is False
        assert analyzer.cross_exchange.is_registered is False

        assert analyzer.spot_futures._subscriptions == []
        assert analyzer.cross_exchange._subscriptions == []
    finally:
        await _unsubscribe(event_bus, subscription)


# ============================================================
# Stats contract
# ============================================================


async def test_get_stats_tracks_facade_and_child_runtime_state(
    analyzer: SpreadAnalyzer,
) -> None:
    initial_stats = analyzer.get_stats()

    assert initial_stats["running"] is False
    assert initial_stats["registered"] is False
    assert initial_stats["spot_futures"]["running"] is False
    assert initial_stats["spot_futures"]["registered"] is False
    assert initial_stats["cross_exchange"]["running"] is False
    assert initial_stats["cross_exchange"]["registered"] is False

    await analyzer.start()

    started_stats = analyzer.get_stats()

    assert started_stats["running"] is True
    assert started_stats["registered"] is True

    assert started_stats["spot_futures"]["running"] is True
    assert started_stats["spot_futures"]["registered"] is True
    assert started_stats["spot_futures"]["enabled"] is True

    assert started_stats["cross_exchange"]["running"] is True
    assert started_stats["cross_exchange"]["registered"] is True
    assert started_stats["cross_exchange"]["enabled"] is True

    expected_spot_keys = {
        "quote_events_received",
        "funding_events_received",
        "quotes_received",
        "funding_updates",
        "invalid_payloads",
        "invalid_quotes",
        "incomplete_quotes",
        "stale_quotes",
        "unaligned_quotes",
        "quotes_stored",
        "funding_stored",
        "snapshots_built",
        "snapshots_skipped",
        "signals_built",
        "cleanup_runs",
        "spot_quotes_cached",
        "futures_quotes_cached",
        "funding_cached",
        "active_windows",
        "latest_snapshots",
    }

    expected_cross_keys = {
        "quote_events_received",
        "quotes_received",
        "invalid_payloads",
        "invalid_quotes",
        "incomplete_quotes",
        "stale_quotes",
        "unaligned_quotes",
        "instrument_type_skips",
        "preferred_exchange_skips",
        "quotes_stored",
        "snapshots_built",
        "snapshots_skipped",
        "signals_built",
        "opportunities_detected",
        "opportunities_published",
        "opportunities_expired",
        "opportunity_detection_misses",
        "cleanup_runs",
        "quotes_cached",
        "active_windows",
        "latest_snapshots",
        "latest_opportunities",
    }

    assert expected_spot_keys.issubset(started_stats["spot_futures"].keys())
    assert expected_cross_keys.issubset(started_stats["cross_exchange"].keys())

    await analyzer.stop()

    stopped_stats = analyzer.get_stats()

    assert stopped_stats["running"] is False
    assert stopped_stats["registered"] is True
    assert stopped_stats["spot_futures"]["running"] is False
    assert stopped_stats["cross_exchange"]["running"] is False

    analyzer.unregister()

    unregistered_stats = analyzer.get_stats()

    assert unregistered_stats["running"] is False
    assert unregistered_stats["registered"] is False
    assert unregistered_stats["spot_futures"]["registered"] is False
    assert unregistered_stats["cross_exchange"]["registered"] is False


# ============================================================
# Failure-oriented / vulnerability-oriented lifecycle scenarios
# ============================================================


async def test_unregister_while_running_does_not_silently_stop_runtime_state(
    analyzer: SpreadAnalyzer,
) -> None:
    """
    This test documents a dangerous lifecycle edge case.

    Facade.unregister() currently warns if called while running, but still
    unregisters children. The analyzer remains marked as running at facade level
    until stop() is called. This test makes that behavior explicit so future
    changes either preserve it intentionally or fix it deliberately.
    """
    await analyzer.start()

    assert analyzer.is_running is True
    assert analyzer.is_registered is True

    analyzer.unregister()

    assert analyzer.is_running is True
    assert analyzer.is_registered is False

    assert analyzer.spot_futures.is_registered is False
    assert analyzer.cross_exchange.is_registered is False

    assert analyzer.spot_futures._subscriptions == []
    assert analyzer.cross_exchange._subscriptions == []

    await analyzer.stop()

    assert analyzer.is_running is False
    assert analyzer.is_registered is False


async def test_register_after_unregister_recreates_exact_subscription_contract(
    analyzer: SpreadAnalyzer,
) -> None:
    analyzer.register()

    first_counts = _child_subscription_counts(analyzer)
    first_ids = _child_subscription_ids(analyzer)

    assert first_counts == {
        "spot_futures": 2,
        "cross_exchange": 1,
    }

    analyzer.unregister()

    assert _child_subscription_counts(analyzer) == {
        "spot_futures": 0,
        "cross_exchange": 0,
    }

    analyzer.register()

    second_counts = _child_subscription_counts(analyzer)
    second_ids = _child_subscription_ids(analyzer)

    assert second_counts == first_counts

    # After unregister/register, subscriptions should be recreated, not reused.
    assert second_ids != first_ids

    assert analyzer.is_registered is True
    assert analyzer.spot_futures.is_registered is True
    assert analyzer.cross_exchange.is_registered is True


async def test_start_after_manual_unregister_recovers_subscriptions_and_runtime(
    analyzer: SpreadAnalyzer,
) -> None:
    analyzer.register()
    analyzer.unregister()

    assert analyzer.is_registered is False
    assert _child_subscription_counts(analyzer) == {
        "spot_futures": 0,
        "cross_exchange": 0,
    }

    await analyzer.start()

    assert analyzer.is_running is True
    assert analyzer.is_registered is True

    assert analyzer.spot_futures.is_running is True
    assert analyzer.cross_exchange.is_running is True

    assert _child_subscription_counts(analyzer) == {
        "spot_futures": 2,
        "cross_exchange": 1,
    }


async def test_partial_child_registration_state_is_detectable_as_inconsistent(
    analyzer: SpreadAnalyzer,
) -> None:
    """
    This is intentionally defensive.

    If a future change accidentally leaves facade._registered=True while one
    child is not registered, this test forces the inconsistency to be visible.
    """
    analyzer.spot_futures.register()

    assert analyzer.spot_futures.is_registered is True
    assert analyzer.cross_exchange.is_registered is False
    assert analyzer.is_registered is False

    analyzer.register()

    assert analyzer.is_registered is True
    assert analyzer.spot_futures.is_registered is True
    assert analyzer.cross_exchange.is_registered is True

    assert _child_subscription_counts(analyzer) == {
        "spot_futures": 2,
        "cross_exchange": 1,
    }