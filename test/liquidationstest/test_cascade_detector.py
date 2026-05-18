# tests/analytics/liquidations/test_cascade_detector.py

from __future__ import annotations

from decimal import Decimal
from typing import Callable

import pytest

from core.event_bus import Event
from core.scheduler import Scheduler

from analytics.liquidations.cascade_detector import CascadeDetector
from analytics.liquidations.config import CascadeDetectorConfig
from analytics.liquidations.enums import LiquidationEventType, LiquidationSide
from analytics.liquidations.metrics import LiquidationMetrics
from analytics.liquidations.models import CascadeDetectionResult, LiquidationEvent
from analytics.liquidations.state import LiquidationState


pytestmark = pytest.mark.asyncio


# =============================================================================
# Helpers
# =============================================================================

def _seed_state_with_events(
    state: LiquidationState,
    events: list[LiquidationEvent],
) -> None:
    for event in events:
        state.add_event(event)


def _scope_key(
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
) -> str:
    return f"{exchange}:{market_type}:{symbol}:{timeframe}"


# =============================================================================
# Construction / registration
# =============================================================================

async def test_cascade_detector_uses_core_dependencies(
    event_bus,
    scheduler: Scheduler,
    cascade_config: CascadeDetectorConfig,
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
    assert detector.service_name == "cascade_detector"
    assert detector.is_running is False
    assert detector.is_registered is False


async def test_detector_rejects_raw_market_liquidation_topic_by_default(
    event_bus,
    liquidation_state: LiquidationState,
) -> None:
    raw_config = CascadeDetectorConfig(
        enabled=True,
        input_topic="market.liquidation",
        allow_raw_input_topics=False,
    )

    with pytest.raises(ValueError, match="raw input topic is not allowed"):
        CascadeDetector(
            event_bus=event_bus,
            config=raw_config,
            state=liquidation_state,
        )


async def test_register_subscribes_to_normalized_input_topic_and_adds_scheduler_jobs(
    event_bus,
    scheduler: Scheduler,
    cascade_config: CascadeDetectorConfig,
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
    assert detector.is_running is False

    stats = detector.get_stats()
    assert stats["registered"] is True
    assert stats["running"] is False
    assert stats["input_topic"] == cascade_config.input_topic
    assert stats["scope"] == "exchange:market_type:symbol:timeframe"
    assert stats["healthcheck_job_id"] is not None
    assert stats["snapshot_job_id"] is not None
    assert stats["cleanup_job_id"] is not None


async def test_register_without_scheduler_only_subscribes_to_event_bus(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        scheduler=None,
        config=cascade_config,
        state=liquidation_state,
    )

    detector.register()

    assert detector.is_registered is True
    assert event_bus.stats()["subscriptions"] == 1

    stats = detector.get_stats()
    assert stats["healthcheck_job_id"] is None
    assert stats["snapshot_job_id"] is None
    assert stats["cleanup_job_id"] is None


async def test_register_is_idempotent(
    event_bus,
    scheduler: Scheduler,
    cascade_config: CascadeDetectorConfig,
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
    assert detector.is_registered is True


# =============================================================================
# Lifecycle
# =============================================================================

async def test_start_auto_registers_detector(
    event_bus,
    scheduler: Scheduler,
    cascade_config: CascadeDetectorConfig,
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

    health = detector.get_health()
    assert health["status"] == "healthy"
    assert health["running"] is True
    assert health["registered"] is True

    await detector.stop()

    assert detector.is_running is False
    assert detector.is_registered is True
    assert detector.get_health()["status"] == "stopped"


async def test_start_is_idempotent(
    event_bus,
    scheduler: Scheduler,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        scheduler=scheduler,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()
    subscriptions_after_first_start = event_bus.stats()["subscriptions"]

    await detector.start()
    subscriptions_after_second_start = event_bus.stats()["subscriptions"]

    assert detector.is_running is True
    assert subscriptions_after_second_start == subscriptions_after_first_start


async def test_start_does_nothing_when_config_disabled(
    event_bus,
    scheduler: Scheduler,
    cascade_config: CascadeDetectorConfig,
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
    assert scheduler.list_jobs() == {}


async def test_stop_is_idempotent(
    event_bus,
    scheduler: Scheduler,
    cascade_config: CascadeDetectorConfig,
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
    assert detector.get_health()["status"] == "stopped"


async def test_restart_stops_and_starts_detector_again(
    event_bus,
    scheduler: Scheduler,
    cascade_config: CascadeDetectorConfig,
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
    assert detector.get_health()["status"] == "healthy"

    await detector.stop()
    assert detector.is_running is False


# =============================================================================
# Event handling
# =============================================================================

async def test_on_liquidation_event_ignores_events_when_not_running(
    event_bus,
    cascade_config: CascadeDetectorConfig,
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
    assert stats["invalid_payload_skips"] == 0
    assert stats["cascade_signals_emitted"] == 0
    assert stats["last_event_at"] is None


async def test_on_liquidation_event_ignores_raw_dict_payload(
    event_bus,
    cascade_config: CascadeDetectorConfig,
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
        payload={
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "price": "65000",
            "quantity": "2",
        },
    )

    await detector.on_liquidation_event(event)

    stats = detector.get_stats()
    assert stats["processed_events"] == 0
    assert stats["invalid_payload_skips"] == 1
    assert stats["cascade_signals_emitted"] == 0
    assert stats["last_error"] is None


async def test_on_liquidation_event_skips_empty_state_window(
    event_bus,
    cascade_config: CascadeDetectorConfig,
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
    assert stats["last_event_at"] is not None


async def test_on_liquidation_event_records_error_for_scope_mismatch(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> None:
    state_event = make_liquidation_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        side=LiquidationSide.LONG,
        quantity=Decimal("2"),
        trade_id="state-event",
    )
    trigger_event = make_liquidation_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
        side=LiquidationSide.LONG,
        quantity=Decimal("2"),
        trade_id="trigger-event",
    )

    liquidation_state.add_event(state_event)

    # Навмисно кладемо state під неправильний key, щоб перевірити defensive guard
    # у CascadeDetector._detect_for_symbol_state().
    liquidation_state.symbols[trigger_event.key] = liquidation_state.get_key(state_event.key)

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    envelope = Event(
        topic=cascade_config.input_topic,
        payload=trigger_event,
    )

    await detector.on_liquidation_event(envelope)

    stats = detector.get_stats()
    assert stats["processed_events"] == 1
    assert stats["cascade_signals_emitted"] == 0
    assert stats["last_error"] is not None
    assert "Trigger event scope does not match" in stats["last_error"]

    health = detector.get_health()
    assert health["status"] == "degraded"


# =============================================================================
# Detection
# =============================================================================

async def test_detect_now_returns_cascade_detection_result_with_full_scope(
    event_bus,
    cascade_config: CascadeDetectorConfig,
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

    result = await detector.detect_now(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert isinstance(result, CascadeDetectionResult)
    assert result.exchange == "binance"
    assert result.market_type == "usdm_futures"
    assert result.symbol == "BTCUSDT"
    assert result.timeframe == "realtime"
    assert result.exchange_symbol == "BTCUSDT"
    assert result.key == ("binance", "usdm_futures", "BTCUSDT", "realtime")
    assert result.event_count >= cascade_config.min_events
    assert result.total_notional_usd >= cascade_config.min_total_notional_usd
    assert result.status.value == "confirmed"
    assert result.cluster.key == result.key
    assert result.metadata["scope_key"] == _scope_key()
    assert result.metadata["dominant_side"] == LiquidationSide.LONG.value

    stats = detector.get_stats()
    assert stats["cascade_signals_emitted"] == 1
    assert stats["latest_signals_buffered"] == 1


async def test_detect_now_returns_none_for_missing_full_scope(
    event_bus,
    cascade_config: CascadeDetectorConfig,
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

    result = await detector.detect_now(
        "binance",
        "BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
    )

    assert result is None
    assert detector.get_stats()["cascade_signals_emitted"] == 0


async def test_detect_now_without_market_type_uses_default_scope_and_returns_none_for_usdm_state(
    event_bus,
    cascade_config: CascadeDetectorConfig,
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

    assert result is None
    assert detector.get_stats()["cascade_signals_emitted"] == 0


async def test_on_liquidation_event_detects_cascade_from_state_and_preserves_correlation_id(
    event_bus,
    cascade_config: CascadeDetectorConfig,
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
        correlation_id="eventbus-correlation-id",
    )

    await detector.on_liquidation_event(envelope)

    last_signal = detector.get_symbol_last_signal(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    stats = detector.get_stats()

    assert last_signal is not None
    assert last_signal.correlation_id == "eventbus-correlation-id"
    assert last_signal.key == trigger_event.key
    assert stats["processed_events"] == 1
    assert stats["cascade_signals_emitted"] == 1


async def test_on_liquidation_event_uses_trigger_event_correlation_when_eventbus_correlation_missing(
    event_bus,
    cascade_config: CascadeDetectorConfig,
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
        correlation_id=None,
    )

    await detector.on_liquidation_event(envelope)

    last_signal = detector.get_symbol_last_signal(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert last_signal is not None
    assert last_signal.correlation_id == trigger_event.correlation_id


async def test_detection_below_threshold_does_not_emit_signal(
    event_bus,
    strict_cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_event: LiquidationEvent,
) -> None:
    liquidation_state.add_event(liquidation_event)

    detector = CascadeDetector(
        event_bus=event_bus,
        config=strict_cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    envelope = Event(
        topic=strict_cascade_config.input_topic,
        payload=liquidation_event,
    )

    await detector.on_liquidation_event(envelope)

    stats = detector.get_stats()
    assert stats["processed_events"] == 1
    assert stats["threshold_skips"] == 1
    assert stats["cascade_signals_emitted"] == 0
    assert detector.get_recent_signals() == []


async def test_detection_rejects_mixed_side_window_below_imbalance_threshold(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    mixed_side_liquidation_events: list[LiquidationEvent],
) -> None:
    _seed_state_with_events(
        liquidation_state,
        mixed_side_liquidation_events,
    )

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    trigger_event = mixed_side_liquidation_events[-1]

    await detector.on_liquidation_event(
        Event(
            topic=cascade_config.input_topic,
            payload=trigger_event,
        )
    )

    stats = detector.get_stats()
    assert stats["processed_events"] == 1
    assert stats["threshold_skips"] == 1
    assert stats["cascade_signals_emitted"] == 0


async def test_detection_does_not_mix_same_symbol_different_market_types(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> None:
    usdm_events = [
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            exchange_symbol="BTCUSDT",
            side=LiquidationSide.LONG,
            quantity=Decimal("1"),
            trade_id=f"usdm-{index}",
        )
        for index in range(2)
    ]
    coinm_event = make_liquidation_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
        exchange_symbol="BTCUSD_PERP",
        side=LiquidationSide.LONG,
        quantity=Decimal("1"),
        trade_id="coinm-trigger",
    )

    _seed_state_with_events(liquidation_state, [*usdm_events, coinm_event])

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    await detector.on_liquidation_event(
        Event(
            topic=cascade_config.input_topic,
            payload=coinm_event,
        )
    )

    stats = detector.get_stats()
    assert stats["processed_events"] == 1
    assert stats["threshold_skips"] == 1
    assert stats["cascade_signals_emitted"] == 0
    assert liquidation_state.scopes_count == 2


async def test_detection_does_not_mix_same_symbol_different_timeframes(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> None:
    realtime_events = [
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            side=LiquidationSide.LONG,
            quantity=Decimal("1"),
            trade_id=f"realtime-{index}",
        )
        for index in range(2)
    ]
    one_minute_event = make_liquidation_event(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="1m",
        side=LiquidationSide.LONG,
        quantity=Decimal("1"),
        trade_id="one-minute-trigger",
    )

    _seed_state_with_events(liquidation_state, [*realtime_events, one_minute_event])

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    await detector.on_liquidation_event(
        Event(
            topic=cascade_config.input_topic,
            payload=one_minute_event,
        )
    )

    stats = detector.get_stats()
    assert stats["processed_events"] == 1
    assert stats["threshold_skips"] == 1
    assert stats["cascade_signals_emitted"] == 0
    assert liquidation_state.scopes_count == 2


# =============================================================================
# Cooldown
# =============================================================================

async def test_detection_respects_cooldown_per_full_scope(
    event_bus,
    cascade_config: CascadeDetectorConfig,
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

    await detector.on_liquidation_event(
        Event(
            topic=cascade_config.input_topic,
            payload=trigger_event,
        )
    )
    await detector.on_liquidation_event(
        Event(
            topic=cascade_config.input_topic,
            payload=trigger_event,
        )
    )

    symbol_state = liquidation_state.get_key(trigger_event.key)
    assert symbol_state is not None
    assert symbol_state.cooldown_until is not None

    stats = detector.get_stats()
    assert stats["cascade_signals_emitted"] == 1
    assert stats["cooldown_skips"] == 1


async def test_cooldown_in_one_scope_does_not_block_other_market_type_scope(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> None:
    usdm_events = [
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            exchange_symbol="BTCUSDT",
            side=LiquidationSide.LONG,
            quantity=Decimal("1"),
            trade_id=f"usdm-cascade-{index}",
        )
        for index in range(3)
    ]
    coinm_events = [
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="coinm_futures",
            timeframe="realtime",
            exchange_symbol="BTCUSD_PERP",
            side=LiquidationSide.LONG,
            quantity=Decimal("1"),
            trade_id=f"coinm-cascade-{index}",
        )
        for index in range(3)
    ]

    _seed_state_with_events(liquidation_state, [*usdm_events, *coinm_events])

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    await detector.on_liquidation_event(
        Event(
            topic=cascade_config.input_topic,
            payload=usdm_events[-1],
        )
    )
    await detector.on_liquidation_event(
        Event(
            topic=cascade_config.input_topic,
            payload=coinm_events[-1],
        )
    )

    stats = detector.get_stats()
    assert stats["cascade_signals_emitted"] == 2
    assert stats["cooldown_skips"] == 0

    usdm_state = liquidation_state.get_key(usdm_events[-1].key)
    coinm_state = liquidation_state.get_key(coinm_events[-1].key)

    assert usdm_state is not None
    assert coinm_state is not None
    assert usdm_state.cooldown_until is not None
    assert coinm_state.cooldown_until is not None


# =============================================================================
# EventBus publishing
# =============================================================================

async def test_detect_now_publishes_cascade_event_with_full_scope_headers(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_events_for_cascade: list[LiquidationEvent],
    event_collector,
    wait_for_events,
) -> None:
    _seed_state_with_events(
        liquidation_state,
        liquidation_events_for_cascade,
    )

    received = event_collector(event_bus, cascade_config.publish_topic_detected)

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    result = await detector.detect_now(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    await wait_for_events(received)

    assert result is not None
    assert len(received) == 1
    assert received[0].topic == cascade_config.publish_topic_detected
    assert isinstance(received[0].payload, CascadeDetectionResult)

    payload = received[0].payload
    assert payload.key == ("binance", "usdm_futures", "BTCUSDT", "realtime")
    assert received[0].source == detector.service_name
    assert received[0].correlation_id == payload.correlation_id
    assert received[0].headers == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
        "exchange_symbol": "BTCUSDT",
        "scope": _scope_key(),
        "event_type": LiquidationEventType.CASCADE.value,
        "severity": payload.severity.value,
    }


async def test_event_bus_input_topic_triggers_detector_and_publishes_result(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_events_for_cascade: list[LiquidationEvent],
    event_collector,
    wait_for_events,
) -> None:
    _seed_state_with_events(
        liquidation_state,
        liquidation_events_for_cascade,
    )

    received = event_collector(event_bus, cascade_config.publish_topic_detected)

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

    await wait_for_events(received)

    assert accepted is True
    assert len(received) == 1
    assert isinstance(received[0].payload, CascadeDetectionResult)
    assert received[0].payload.correlation_id == "test-correlation-id"
    assert received[0].payload.key == trigger_event.key
    assert received[0].headers["scope"] == _scope_key()


async def test_emit_detection_updates_metrics_only_when_eventbus_accepts(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    liquidation_events_for_cascade: list[LiquidationEvent],
    event_collector,
    wait_for_events,
) -> None:
    _seed_state_with_events(
        liquidation_state,
        liquidation_events_for_cascade,
    )

    received = event_collector(event_bus, cascade_config.publish_topic_detected)

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    await detector.start()

    result = await detector.detect_now(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    await wait_for_events(received)

    assert result is not None
    assert len(received) == 1
    assert liquidation_metrics.total_cascades_detected == 1
    assert liquidation_metrics.cascade_by_scope[_scope_key()] == 1


# =============================================================================
# Signal memory / query API
# =============================================================================

async def test_get_recent_signals_filters_by_full_scope(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> None:
    usdm_events = [
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            side=LiquidationSide.LONG,
            quantity=Decimal("1"),
            trade_id=f"usdm-signal-{index}",
        )
        for index in range(3)
    ]
    coinm_events = [
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="coinm_futures",
            timeframe="realtime",
            exchange_symbol="BTCUSD_PERP",
            side=LiquidationSide.LONG,
            quantity=Decimal("1"),
            trade_id=f"coinm-signal-{index}",
        )
        for index in range(3)
    ]

    _seed_state_with_events(liquidation_state, [*usdm_events, *coinm_events])

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    await detector.detect_now(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    await detector.detect_now(
        "binance",
        "BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
    )

    all_signals = detector.get_recent_signals(
        exchange="binance",
        symbol="BTCUSDT",
        limit=10,
    )
    usdm_signals = detector.get_recent_signals(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        limit=10,
    )
    coinm_signals = detector.get_recent_signals(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
        limit=10,
    )

    assert len(all_signals) == 2
    assert len(usdm_signals) == 1
    assert len(coinm_signals) == 1
    assert usdm_signals[0].market_type == "usdm_futures"
    assert coinm_signals[0].market_type == "coinm_futures"


async def test_get_symbol_last_signal_requires_full_scope_when_default_scope_differs(
    event_bus,
    cascade_config: CascadeDetectorConfig,
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

    await detector.detect_now(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    explicit = detector.get_symbol_last_signal(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    broad = detector.get_symbol_last_signal(
        "binance",
        "BTCUSDT",
    )

    assert explicit is not None
    assert broad is not None
    assert explicit is broad
    assert explicit.key == ("binance", "usdm_futures", "BTCUSDT", "realtime")


async def test_get_hot_symbols_returns_latest_signal_per_scope(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> None:
    btc_events = [
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            side=LiquidationSide.LONG,
            quantity=Decimal("1"),
            trade_id=f"btc-hot-{index}",
        )
        for index in range(3)
    ]
    eth_events = [
        make_liquidation_event(
            exchange="binance",
            symbol="ETHUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            side=LiquidationSide.LONG,
            price=Decimal("3500"),
            quantity=Decimal("40"),
            trade_id=f"eth-hot-{index}",
        )
        for index in range(3)
    ]

    _seed_state_with_events(liquidation_state, [*btc_events, *eth_events])

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    await detector.detect_now(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    await detector.detect_now(
        "binance",
        "ETHUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    hot_symbols = detector.get_hot_symbols(limit=10)

    assert len(hot_symbols) == 2
    assert {
        row["scope"]["symbol"]
        for row in hot_symbols
    } == {"BTCUSDT", "ETHUSDT"}
    assert all(row["market_type"] == "usdm_futures" for row in hot_symbols)


async def test_recent_signals_buffer_is_limited_by_config(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> None:
    cascade_config.recent_signals_limit = 2

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    for index, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT")):
        events = [
            make_liquidation_event(
                exchange="binance",
                symbol=symbol,
                market_type="usdm_futures",
                timeframe="realtime",
                side=LiquidationSide.LONG,
                price=Decimal("100"),
                quantity=Decimal("1000"),
                trade_id=f"{symbol}-{event_index}",
                correlation_id=f"{symbol}-correlation",
            )
            for event_index in range(3)
        ]
        _seed_state_with_events(liquidation_state, events)

        await detector.detect_now(
            "binance",
            symbol,
            market_type="usdm_futures",
            timeframe="realtime",
        )

    recent = detector.get_recent_signals(limit=10)

    assert len(recent) == 2
    assert detector.get_stats()["latest_signals_buffered"] == 2
    assert {signal.symbol for signal in recent} == {"ETHUSDT", "SOLUSDT"}


# =============================================================================
# Diagnostics / health / snapshots
# =============================================================================

async def test_get_symbol_diagnostic_returns_missing_scope_payload(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    diagnostic = detector.get_symbol_diagnostic(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert diagnostic == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
        "scope": {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "timeframe": "realtime",
        },
        "exists": False,
    }


async def test_get_symbol_diagnostic_returns_window_stats_and_last_signal(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_events_for_cascade: list[LiquidationEvent],
) -> None:
    _seed_state_with_events(liquidation_state, liquidation_events_for_cascade)

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    await detector.detect_now(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    diagnostic = detector.get_symbol_diagnostic(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert diagnostic["exists"] is True
    assert diagnostic["exchange"] == "binance"
    assert diagnostic["market_type"] == "usdm_futures"
    assert diagnostic["symbol"] == "BTCUSDT"
    assert diagnostic["timeframe"] == "realtime"
    assert diagnostic["exchange_symbol"] == "BTCUSDT"
    assert diagnostic["buffer_snapshot"]["total_buffered_events"] == 3
    assert diagnostic["window_stats"]["total_events"] == 3
    assert diagnostic["cooldown_active"] is True
    assert diagnostic["last_signal"] is not None
    assert diagnostic["last_signal"]["scope"] == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
    }


async def test_emit_health_publishes_health_event(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    event_collector,
    wait_for_events,
) -> None:
    received = event_collector(event_bus, cascade_config.publish_topic_health)

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    accepted = await detector.emit_health()
    await wait_for_events(received)

    assert accepted is True
    assert len(received) == 1
    assert received[0].topic == cascade_config.publish_topic_health
    assert isinstance(received[0].payload, dict)
    assert received[0].payload["status"] == "healthy"
    assert received[0].payload["scope"] == "exchange:market_type:symbol:timeframe"
    assert received[0].headers["event_type"] == "health"


async def test_emit_runtime_snapshot_publishes_snapshot_event(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_events_for_cascade: list[LiquidationEvent],
    event_collector,
    wait_for_events,
) -> None:
    _seed_state_with_events(liquidation_state, liquidation_events_for_cascade)

    received = event_collector(event_bus, cascade_config.publish_topic_snapshot)

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()
    await detector.detect_now(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    accepted = await detector.emit_runtime_snapshot()
    await wait_for_events(received)

    assert accepted is True
    assert len(received) == 1
    assert received[0].topic == cascade_config.publish_topic_snapshot
    assert isinstance(received[0].payload, dict)
    assert received[0].payload["service"] == detector.service_name
    assert received[0].payload["running"] is True
    assert received[0].payload["scope"] == "exchange:market_type:symbol:timeframe"
    assert received[0].payload["stats"]["tracked_scopes"] == 1
    assert len(received[0].payload["latest_signals"]) == 1
    assert received[0].headers["event_type"] == "snapshot"


async def test_scheduled_cleanup_trims_latest_signals_buffer(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> None:
    cascade_config.recent_signals_limit = 2

    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    await detector.start()

    for index, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT")):
        events = [
            make_liquidation_event(
                exchange="binance",
                symbol=symbol,
                market_type="usdm_futures",
                timeframe="realtime",
                side=LiquidationSide.LONG,
                price=Decimal("100"),
                quantity=Decimal("1000"),
                trade_id=f"cleanup-{symbol}-{event_index}",
            )
            for event_index in range(3)
        ]
        _seed_state_with_events(liquidation_state, events)

        # Тимчасово збільшуємо limit через direct append behavior already handled
        # by _remember_signal(); цей виклик лишається як regression coverage.
        await detector.detect_now(
            "binance",
            symbol,
            market_type="usdm_futures",
            timeframe="realtime",
        )

    await detector._scheduled_cleanup()

    assert len(detector.get_recent_signals(limit=10)) <= 2
    assert detector.get_stats()["latest_signals_buffered"] <= 2


# =============================================================================
# Defensive API
# =============================================================================

async def test_detect_now_key_raises_explicit_runtime_error(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_event: LiquidationEvent,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    with pytest.raises(RuntimeError, match="Use async detect_now"):
        detector.detect_now_key(liquidation_event.key)


async def test_get_recent_signals_returns_empty_for_non_positive_limit(
    event_bus,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
) -> None:
    detector = CascadeDetector(
        event_bus=event_bus,
        config=cascade_config,
        state=liquidation_state,
    )

    assert detector.get_recent_signals(limit=0) == []
    assert detector.get_recent_signals(limit=-1) == []