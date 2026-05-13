# tests/analytics/liquidations/test_liquidation_stream.py

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from core.event_bus import Event
from core.scheduler import Scheduler

from analytics.liquidations.liquidation_stream import LiquidationStream
from analytics.liquidations.metrics import LiquidationMetrics
from analytics.liquidations.models import LiquidationEvent
from analytics.liquidations.state import LiquidationState


pytestmark = pytest.mark.asyncio


# ============================================================
# Helpers
# ============================================================

async def _wait_for_events(
    received: list[Event],
    *,
    expected_count: int = 1,
    timeout: float = 0.5,
) -> None:
    """
    Дає EventBus worker-у час обробити async queue.
    """
    deadline = asyncio.get_running_loop().time() + timeout

    while len(received) < expected_count:
        if asyncio.get_running_loop().time() >= deadline:
            break
        await asyncio.sleep(0.01)


# ============================================================
# Construction / registration
# ============================================================

async def test_liquidation_stream_uses_core_dependencies(
    event_bus,
    scheduler: Scheduler,
    stream_config,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    fake_exchange_adapter,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    assert stream.event_bus is event_bus
    assert stream.scheduler is scheduler
    assert stream.exchange_adapter is fake_exchange_adapter
    assert stream.config is stream_config
    assert stream.state is liquidation_state
    assert stream.metrics is liquidation_metrics


async def test_register_adds_scheduler_jobs(
    event_bus,
    scheduler: Scheduler,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    stream.register()

    jobs = scheduler.list_jobs()
    job_names = {job["name"] for job in jobs.values()}

    assert stream_config.healthcheck_job_name in job_names
    assert stream_config.snapshot_job_name in job_names
    assert stream_config.cleanup_job_name in job_names


async def test_register_is_idempotent(
    event_bus,
    scheduler: Scheduler,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    stream.register()
    jobs_after_first_register = scheduler.list_jobs()

    stream.register()
    jobs_after_second_register = scheduler.list_jobs()

    assert len(jobs_after_second_register) == len(jobs_after_first_register)


# ============================================================
# Normalization
# ============================================================

async def test_normalize_flat_payload_returns_liquidation_event(
    event_bus,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
    raw_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    event = stream.normalize_event(raw_liquidation_payload)

    assert isinstance(event, LiquidationEvent)
    assert event.exchange == "binance"
    assert event.symbol == "BTCUSDT"
    assert event.price == Decimal("65000")
    assert event.quantity == Decimal("2")
    assert event.notional_usd == Decimal("130000")
    assert event.is_valid is True


async def test_normalize_binance_force_order_payload_returns_liquidation_event(
    event_bus,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
    binance_force_order_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    event = stream.normalize_event(binance_force_order_payload)

    assert isinstance(event, LiquidationEvent)
    assert event.exchange == "binance"
    assert event.symbol == "BTCUSDT"
    assert event.price == Decimal("65000")
    assert event.quantity == Decimal("2")
    assert event.notional_usd == Decimal("130000")
    assert event.is_valid is True


async def test_normalize_invalid_payload_returns_none(
    event_bus,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
    raw_invalid_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    event = stream.normalize_event(raw_invalid_liquidation_payload)

    assert event is None


# ============================================================
# Raw handling / state / metrics
# ============================================================

async def test_handle_raw_message_adds_event_to_state(
    event_bus,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
    raw_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    event = await stream.handle_raw_message(raw_liquidation_payload)

    assert event is not None

    symbol_state = liquidation_state.get("binance", "BTCUSDT")

    assert symbol_state is not None
    assert symbol_state.total_buffered_events == 1
    assert symbol_state.total_events_seen == 1


async def test_handle_raw_message_updates_metrics(
    event_bus,
    stream_config,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    fake_exchange_adapter,
    raw_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    event = await stream.handle_raw_message(raw_liquidation_payload)

    assert event is not None
    assert liquidation_metrics.total_events_seen == 1
    assert liquidation_metrics.total_valid_events == 1
    assert liquidation_metrics.total_invalid_events == 0
    assert liquidation_metrics.total_long_events == 1


async def test_handle_invalid_payload_is_dropped_and_updates_metrics(
    event_bus,
    stream_config,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    fake_exchange_adapter,
    raw_invalid_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    event = await stream.handle_raw_message(raw_invalid_liquidation_payload)

    assert event is None
    assert liquidation_metrics.total_invalid_events == 1

    symbol_state = liquidation_state.get("binance", "BTCUSDT")
    assert symbol_state is None


async def test_handle_stale_payload_is_dropped(
    event_bus,
    stream_config,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    fake_exchange_adapter,
    raw_stale_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    event = await stream.handle_raw_message(raw_stale_liquidation_payload)

    assert event is None
    assert liquidation_metrics.total_stale_events == 1

    symbol_state = liquidation_state.get("binance", "BTCUSDT")
    assert symbol_state is None


async def test_duplicate_payload_is_dropped(
    event_bus,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
    raw_liquidation_payload: dict[str, Any],
) -> None:
    stream_config.deduplication_enabled = True

    stream = LiquidationStream(
        event_bus=event_bus,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    first_event = await stream.handle_raw_message(raw_liquidation_payload)
    second_event = await stream.handle_raw_message(raw_liquidation_payload)

    assert first_event is not None
    assert second_event is None

    symbol_state = liquidation_state.get("binance", "BTCUSDT")

    assert symbol_state is not None
    assert symbol_state.total_buffered_events == 1


# ============================================================
# EventBus publishing
# ============================================================

async def test_handle_raw_message_publishes_raw_event_when_enabled(
    event_bus,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
    raw_liquidation_payload: dict[str, Any],
) -> None:
    stream_config.emit_raw_events = True

    received: list[Event] = []

    event_bus.subscribe(
        stream_config.publish_topic_raw,
        lambda event: received.append(event),
        name="test.raw_liquidation_collector",
    )

    stream = LiquidationStream(
        event_bus=event_bus,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.handle_raw_message(raw_liquidation_payload)
    await _wait_for_events(received)

    assert len(received) == 1
    assert received[0].topic == stream_config.publish_topic_raw
    assert isinstance(received[0].payload, dict)
    assert received[0].source == stream.service_name


async def test_handle_raw_message_publishes_normalized_event(
    event_bus,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
    raw_liquidation_payload: dict[str, Any],
) -> None:
    received: list[Event] = []

    event_bus.subscribe(
        stream_config.publish_topic_normalized,
        lambda event: received.append(event),
        name="test.normalized_liquidation_collector",
    )

    stream = LiquidationStream(
        event_bus=event_bus,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.handle_raw_message(raw_liquidation_payload)
    await _wait_for_events(received)

    assert len(received) == 1
    assert received[0].topic == stream_config.publish_topic_normalized
    assert isinstance(received[0].payload, LiquidationEvent)
    assert received[0].payload.symbol == "BTCUSDT"
    assert received[0].payload.notional_usd == Decimal("130000")
    assert received[0].source == stream.service_name


async def test_handle_large_liquidation_publishes_large_event(
    event_bus,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
    raw_large_liquidation_payload: dict[str, Any],
) -> None:
    stream_config.emit_large_events = True
    stream_config.large_liquidation_threshold_usd = Decimal("100000")

    received: list[Event] = []

    event_bus.subscribe(
        stream_config.publish_topic_large,
        lambda event: received.append(event),
        name="test.large_liquidation_collector",
    )

    stream = LiquidationStream(
        event_bus=event_bus,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    event = await stream.handle_raw_message(raw_large_liquidation_payload)
    await _wait_for_events(received)

    assert event is not None
    assert event.notional_usd >= stream_config.large_liquidation_threshold_usd

    assert len(received) == 1
    assert received[0].topic == stream_config.publish_topic_large
    assert isinstance(received[0].payload, LiquidationEvent)


async def test_small_liquidation_does_not_publish_large_event(
    event_bus,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
    raw_small_liquidation_payload: dict[str, Any],
) -> None:
    stream_config.emit_large_events = True
    stream_config.large_liquidation_threshold_usd = Decimal("100000")

    received: list[Event] = []

    event_bus.subscribe(
        stream_config.publish_topic_large,
        lambda event: received.append(event),
        name="test.large_liquidation_collector",
    )

    stream = LiquidationStream(
        event_bus=event_bus,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    event = await stream.handle_raw_message(raw_small_liquidation_payload)
    await asyncio.sleep(0.05)

    assert event is not None
    assert event.notional_usd < stream_config.large_liquidation_threshold_usd
    assert received == []


# ============================================================
# Lifecycle
# ============================================================

async def test_start_connects_exchange_adapter_and_creates_consumer_task(
    event_bus,
    scheduler: Scheduler,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()

    assert fake_exchange_adapter.connected is True
    assert fake_exchange_adapter.connected_symbols == stream_config.symbols

    await stream.stop()

    assert fake_exchange_adapter.connected is False
    assert fake_exchange_adapter.disconnected is True


async def test_start_does_nothing_when_config_disabled(
    event_bus,
    scheduler: Scheduler,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
) -> None:
    stream_config.enabled = False

    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()

    assert fake_exchange_adapter.connected is False


async def test_stop_is_idempotent(
    event_bus,
    scheduler: Scheduler,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.stop()
    await stream.stop()

    assert fake_exchange_adapter.connected is False


async def test_close_stops_and_unregisters_scheduler_jobs(
    event_bus,
    scheduler: Scheduler,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    stream.register()

    assert scheduler.list_jobs()

    await stream.close()

    assert scheduler.list_jobs() == {}


async def test_restart_stops_and_starts_stream_again(
    event_bus,
    scheduler: Scheduler,
    stream_config,
    liquidation_state: LiquidationState,
    fake_exchange_adapter,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        exchange_adapter=fake_exchange_adapter,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()
    assert fake_exchange_adapter.connected is True

    await stream.restart()
    assert fake_exchange_adapter.connected is True

    await stream.stop()
    assert fake_exchange_adapter.connected is False