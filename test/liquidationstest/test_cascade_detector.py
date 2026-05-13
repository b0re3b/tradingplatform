# test/liquidationstest/test_cascade_detector.py

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.event_bus import Event
from core.scheduler import Scheduler

from analytics.liquidations.cascade_detector import CascadeDetector
from analytics.liquidations.metrics import LiquidationMetrics
from analytics.liquidations.models import CascadeDetectionResult, LiquidationEvent
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


def _seed_state_with_events(
    state: LiquidationState,
    events: list[LiquidationEvent],
) -> None:
    for event in events:
        state.add_event(event)


# ============================================================
# Construction / registration
# ============================================================

async def test_cascade_detector_uses_core_dependencies(
    event_bus,
    scheduler: Scheduler,
    cascade_config,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        scheduler=scheduler,
        config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    assert detector.event_bus is event_bus
    assert detector.scheduler is scheduler
    assert detector.config is cascade_config
    assert detector.state is liquidation_state
    assert detector.metrics is liquidation_metrics


async def test_register_subscribes_to_input_topic_and_adds_scheduler_jobs(
    event_bus,
    scheduler: Scheduler,
    cascade_config,
    liquidation_state: LiquidationState,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        scheduler=scheduler,
        config=cascade_config,
        state=liquidation_state,
    )

    detector.register()

    event_bus_stats = event_bus.stats()
    assert event_bus_stats["subscriptions"] == 1

    jobs = scheduler.list_jobs()
    job_names = {job["name"] for job in jobs.values()}

    assert cascade_config.healthcheck_job_name in job_names
    assert cascade_config.snapshot_job_name in job_names
    assert cascade_config.cleanup_job_name in job_names

    assert detector.is_registered is True


async def test_register_is_idempotent(
    event_bus,
    scheduler: Scheduler,
    cascade_config,
    liquidation_state: LiquidationState,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        scheduler=scheduler,
        config=cascade_config,
        state=liquidation_state,
    )

    detector.register()

    subscriptions_after_first_register = event_bus.stats()["subscriptions"]
    jobs_after_first_register = scheduler.list_jobs()

    detector.register()

    subscriptions_after_second_register = event_bus.stats()["subscriptions"]
    jobs_after_second_register = scheduler.list_jobs()

    assert subscriptions_after_second_register == subscriptions_after_first_register
    assert len(jobs_after_second_register) == len(jobs_after_first_register)


# ============================================================
# Lifecycle
# ============================================================

async def test_start_auto_registers_detector(
    event_bus,
    scheduler: Scheduler,
    cascade_config,
    liquidation_state: LiquidationState,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        scheduler=scheduler,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    assert detector.is_running is True
    assert detector.is_registered is True
    assert event_bus.stats()["subscriptions"] == 1

    await detector.stop()

    assert detector.is_running is False


async def test_start_does_nothing_when_config_disabled(
    event_bus,
    scheduler: Scheduler,
    cascade_config,
    liquidation_state: LiquidationState,
) -> None:
    cascade_config.enabled = False

    detector = CascadeDetector(
        event_bus=event_bus,
        scheduler=scheduler,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    assert detector.is_running is False
    assert detector.is_registered is False
    assert event_bus.stats()["subscriptions"] == 0


async def test_stop_is_idempotent(
    event_bus,
    scheduler: Scheduler,
    cascade_config,
    liquidation_state: LiquidationState,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        scheduler=scheduler,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.stop()
    await detector.stop()

    assert detector.is_running is False


async def test_restart_stops_and_starts_detector_again(
    event_bus,
    scheduler: Scheduler,
    cascade_config,
    liquidation_state: LiquidationState,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        scheduler=scheduler,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()
    assert detector.is_running is True

    await detector.restart()
    assert detector.is_running is True
    assert detector.is_registered is True

    await detector.stop()
    assert detector.is_running is False


# ============================================================
# Event handling
# ============================================================

async def test_on_liquidation_event_ignores_events_when_not_running(
    event_bus,
    cascade_config,
    liquidation_state: LiquidationState,
    liquidation_event: LiquidationEvent,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    event = Event(
        topic=cascade_config.input_topic,
        payload=liquidation_event,
    )

    await detector.on_liquidation_event(event)

    stats = detector.get_stats()

    assert stats["processed_events"] == 0
    assert stats["cascade_signals_emitted"] == 0


async def test_on_liquidation_event_ignores_invalid_payload(
    event_bus,
    cascade_config,
    liquidation_state: LiquidationState,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    event = Event(
        topic=cascade_config.input_topic,
        payload={"not": "a LiquidationEvent"},
    )

    await detector.on_liquidation_event(event)

    stats = detector.get_stats()

    assert stats["processed_events"] == 0
    assert stats["invalid_payload_skips"] == 1
    assert stats["cascade_signals_emitted"] == 0


async def test_on_liquidation_event_skips_empty_state_window(
    event_bus,
    cascade_config,
    liquidation_state: LiquidationState,
    liquidation_event: LiquidationEvent,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    event = Event(
        topic=cascade_config.input_topic,
        payload=liquidation_event,
    )

    await detector.on_liquidation_event(event)

    stats = detector.get_stats()

    assert stats["processed_events"] == 1
    assert stats["empty_window_skips"] == 1
    assert stats["cascade_signals_emitted"] == 0


# ============================================================
# Detection
# ============================================================

async def test_detect_now_returns_cascade_detection_result(
    event_bus,
    cascade_config,
    liquidation_state: LiquidationState,
    liquidation_events_for_cascade: list[LiquidationEvent],
) -> None:
    _seed_state_with_events(
        liquidation_state,
        liquidation_events_for_cascade,
    )

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    result = await detector.detect_now("binance", "BTCUSDT")

    assert isinstance(result, CascadeDetectionResult)
    assert result.exchange == "binance"
    assert result.symbol == "BTCUSDT"
    assert result.event_count >= cascade_config.min_events
    assert result.total_notional_usd >= cascade_config.min_total_notional_usd
    assert result.status.value == "confirmed"

    stats = detector.get_stats()

    assert stats["cascade_signals_emitted"] == 1
    assert stats["latest_signals_buffered"] == 1


async def test_on_liquidation_event_detects_cascade_from_state(
    event_bus,
    cascade_config,
    liquidation_state: LiquidationState,
    liquidation_events_for_cascade: list[LiquidationEvent],
) -> None:
    _seed_state_with_events(
        liquidation_state,
        liquidation_events_for_cascade,
    )

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    trigger_event = liquidation_events_for_cascade[-1]

    envelope = Event(
        topic=cascade_config.input_topic,
        payload=trigger_event,
        correlation_id="test-correlation-id",
    )

    await detector.on_liquidation_event(envelope)

    last_signal = detector.get_symbol_last_signal("binance", "BTCUSDT")
    stats = detector.get_stats()

    assert last_signal is not None
    assert last_signal.correlation_id == "test-correlation-id"
    assert stats["processed_events"] == 1
    assert stats["cascade_signals_emitted"] == 1


async def test_detection_respects_cooldown(
    event_bus,
    cascade_config,
    liquidation_state: LiquidationState,
    liquidation_events_for_cascade: list[LiquidationEvent],
) -> None:
    _seed_state_with_events(
        liquidation_state,
        liquidation_events_for_cascade,
    )

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    trigger_event = liquidation_events_for_cascade[-1]

    first_envelope = Event(
        topic=cascade_config.input_topic,
        payload=trigger_event,
    )
    second_envelope = Event(
        topic=cascade_config.input_topic,
        payload=trigger_event,
    )

    await detector.on_liquidation_event(first_envelope)
    await detector.on_liquidation_event(second_envelope)

    stats = detector.get_stats()

    assert stats["cascade_signals_emitted"] == 1
    assert stats["cooldown_skips"] == 1


async def test_detection_below_threshold_does_not_emit_signal(
    event_bus,
    cascade_config,
    liquidation_state: LiquidationState,
    liquidation_event: LiquidationEvent,
) -> None:
    liquidation_state.add_event(liquidation_event)

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    envelope = Event(
        topic=cascade_config.input_topic,
        payload=liquidation_event,
    )

    await detector.on_liquidation_event(envelope)

    stats = detector.get_stats()

    assert stats["processed_events"] == 1
    assert stats["threshold_skips"] == 1
    assert stats["cascade_signals_emitted"] == 0


# ============================================================
# EventBus publishing
# ============================================================

async def test_detect_now_publishes_cascade_event(
    event_bus,
    cascade_config,
    liquidation_state: LiquidationState,
    liquidation_events_for_cascade: list[LiquidationEvent],
) -> None:
    _seed_state_with_events(
        liquidation_state,
        liquidation_events_for_cascade,
    )

    received: list[Event] = []

    event_bus.subscribe(
        cascade_config.publish_topic_detected,
        lambda event: received.append(event),
        name="test.cascade_detected_collector",
    )

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    result = await detector.detect_now("binance", "BTCUSDT")
    await _wait_for_events(received)

    assert result is not None
    assert len(received) == 1
    assert received[0].topic == cascade_config.publish_topic_detected
    assert isinstance(received[0].payload, CascadeDetectionResult)
    assert received[0].payload.symbol == "BTCUSDT"
    assert received[0].source == detector.service_name
    assert received[0].headers["event_type"] == "cascade"


async def test_event_bus_input_topic_triggers_detector(
    event_bus,
    cascade_config,
    liquidation_state: LiquidationState,
    liquidation_events_for_cascade: list[LiquidationEvent],
) -> None:
    _seed_state_with_events(
        liquidation_state,
        liquidation_events_for_cascade,
    )

    received: list[Event] = []

    event_bus.subscribe(
        cascade_config.publish_topic_detected,
        lambda event: received.append(event),
        name="test.cascade_detected_collector",
    )

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    trigger_event = liquidation_events_for_cascade[-1]

    accepted = await event_bus.emit(
        cascade_config.input_topic,
        trigger_event,
        source="test",
        correlation_id="test-correlation-id",
    )

    await _wait_for_events(received)

    assert accepted is True
    assert len(received) == 1
    assert isinstance(received[0].payload, CascadeDetectionResult)
    assert received[0].payload.correlation_id == "test-correlation-id"


# ============================================================
# Health / snapshot
# ============================================================

async def test_emit_health_publishes_health_event(
    event_bus,
    cascade_config,
    liquidation_state: LiquidationState,
) -> None:
    received: list[Event] = []

    event_bus.subscribe(
        cascade_config.publish_topic_health,
        lambda event: received.append(event),
        name="test.detector_health_collector",
    )

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    accepted = await detector.emit_health()
    await _wait_for_events(received)

    assert accepted is True
    assert len(received) == 1
    assert received[0].topic == cascade_config.publish_topic_health
    assert isinstance(received[0].payload, dict)
    assert received[0].payload["status"] == "healthy"
    assert received[0].headers["event_type"] == "health"


async def test_emit_runtime_snapshot_publishes_snapshot_event(
    event_bus,
    cascade_config,
    liquidation_state: LiquidationState,
) -> None:
    received: list[Event] = []

    event_bus.subscribe(
        cascade_config.publish_topic_snapshot,
        lambda event: received.append(event),
        name="test.detector_snapshot_collector",
    )

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    accepted = await detector.emit_runtime_snapshot()
    await _wait_for_events(received)

    assert accepted is True
    assert len(received) == 1
    assert received[0].topic == cascade_config.publish_topic_snapshot
    assert isinstance(received[0].payload, dict)
    assert received[0].payload["service"] == detector.service_name
    assert received[0].payload["running"] is True
    assert received[0].headers["event_type"] == "snapshot"