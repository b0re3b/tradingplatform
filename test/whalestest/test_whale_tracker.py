from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any

import pytest

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from analytics.whales.config import WhaleTrackerConfig
from analytics.whales.whale_tracker import WhaleTracker


pytestmark = pytest.mark.asyncio


# =============================================================================
# Topics
# =============================================================================

LARGE_TRADE_TOPIC = "analytics.whales.large_trade"

LIQUIDATIONS_UPDATED_TOPIC = "market.liquidations.updated"
RAW_LIQUIDATION_TOPIC = "market.liquidation"

WHALE_ACTIVITY_TOPIC = "analytics.whales.whale_activity"
WHALE_PRESSURE_TOPIC = "analytics.whales.whale_pressure"
WHALE_LIQUIDATION_CONTEXT_TOPIC = "analytics.whales.whale_liquidation_context"


# =============================================================================
# Local helpers
# =============================================================================

def _build_tracker(
    *,
    config: WhaleTrackerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> WhaleTracker:
    return WhaleTracker(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )


def _default_scope_state(
    tracker: WhaleTracker,
    symbol: str = "BTCUSDT",
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    timeframe: str = "realtime",
) -> dict[str, Any]:
    return tracker.get_symbol_state(
        symbol,
        exchange=exchange,
        market_type=market_type,
        timeframe=timeframe,
    )


async def _feed_large_trades(
    tracker: WhaleTracker,
    large_trade_payload_factory,
    *,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    timeframe: str = "realtime",
    side: str = "buy",
    count: int = 2,
    notional: float = 60_000.0,
    start_ts_ms: int | None = None,
) -> list[Any]:
    base_ts = start_ts_ms if start_ts_ms is not None else int(time.time() * 1000)
    results: list[Any] = []

    for index in range(count):
        result = await tracker.process_large_trade_payload(
            large_trade_payload_factory(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                side=side,
                price=100.0,
                quantity=notional / 100.0,
                notional=notional,
                timestamp_ms=base_ts + index,
                trade_id=f"{exchange}-{market_type}-{symbol}-{side}-{index}",
            )
        )
        results.append(result)

    return results


def _assert_tracker_state_empty(tracker: WhaleTracker) -> None:
    assert tracker.get_all_states() == {}


# =============================================================================
# Lifecycle / core behavior
# =============================================================================

async def test_tracker_register_is_idempotent_and_subscribes_to_production_topics(
    whale_tracker: WhaleTracker,
) -> None:
    await whale_tracker.register()
    await whale_tracker.register()
    await whale_tracker.register()

    assert whale_tracker.is_registered is True
    assert whale_tracker.is_started is False

    patterns = {subscription.pattern for subscription in whale_tracker.subscriptions}

    assert LARGE_TRADE_TOPIC in patterns
    assert LIQUIDATIONS_UPDATED_TOPIC in patterns
    assert RAW_LIQUIDATION_TOPIC not in patterns
    assert len(patterns) == 2


async def test_tracker_legacy_register_subscribes_to_raw_liquidation_topic(
    whale_tracker_legacy: WhaleTracker,
) -> None:
    await whale_tracker_legacy.register()
    await whale_tracker_legacy.register()

    assert whale_tracker_legacy.is_registered is True

    patterns = {
        subscription.pattern
        for subscription in whale_tracker_legacy.subscriptions
    }

    assert LARGE_TRADE_TOPIC in patterns
    assert LIQUIDATIONS_UPDATED_TOPIC in patterns
    assert RAW_LIQUIDATION_TOPIC in patterns
    assert len(patterns) == 3


async def test_tracker_register_without_liquidations_subscribes_only_large_trade(
    whale_tracker_config_fast: WhaleTrackerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    config = replace(
        whale_tracker_config_fast,
        subscribe_liquidations=False,
    )
    tracker = _build_tracker(config=config, event_bus=event_bus, scheduler=scheduler)

    await tracker.register()

    assert tracker.is_registered is True

    patterns = {subscription.pattern for subscription in tracker.subscriptions}
    assert patterns == {LARGE_TRADE_TOPIC}


async def test_tracker_start_adds_one_cleanup_job_and_stop_removes_runtime_state(
    whale_tracker: WhaleTracker,
) -> None:
    await whale_tracker.start()
    await whale_tracker.start()
    await whale_tracker.start()

    assert whale_tracker.is_started is True
    assert whale_tracker.is_registered is True

    assert len(whale_tracker.subscriptions) == 2
    assert len(whale_tracker.scheduler_job_ids) == 1

    await whale_tracker.stop()
    await whale_tracker.stop()

    assert whale_tracker.is_started is False
    assert whale_tracker.is_registered is False
    assert len(whale_tracker.subscriptions) == 0
    assert len(whale_tracker.scheduler_job_ids) == 0


async def test_disabled_tracker_drops_everything_without_state_or_emission(
    whale_tracker_config_fast: WhaleTrackerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    large_trade_payload_factory,
    liquidations_updated_payload_factory,
) -> None:
    config = replace(whale_tracker_config_fast, enabled=False)
    tracker = _build_tracker(config=config, event_bus=event_bus, scheduler=scheduler)

    await tracker.register()
    await tracker.start()

    assert tracker.is_registered is False
    assert tracker.is_started is False

    large_trade_result = await tracker.process_large_trade_payload(
        large_trade_payload_factory(notional=200_000.0)
    )
    liquidation_result = await tracker.process_liquidation_payload(
        liquidations_updated_payload_factory(notional=200_000.0),
        source_topic=LIQUIDATIONS_UPDATED_TOPIC,
    )

    assert large_trade_result.has_signals is False
    assert liquidation_result is None
    _assert_tracker_state_empty(tracker)

    await asyncio.sleep(0.05)
    assert event_collector.by_topic(WHALE_ACTIVITY_TOPIC) == []
    assert event_collector.by_topic(WHALE_PRESSURE_TOPIC) == []
    assert event_collector.by_topic(WHALE_LIQUIDATION_CONTEXT_TOPIC) == []


# =============================================================================
# Normalization / hostile payloads
# =============================================================================

@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "not-a-dict",
        {"data": None},
        {"data": []},
        {"symbol": ""},
        {"symbol": "   "},
        {"symbol": "BTCUSDT", "side": "buy"},
        {"symbol": "BTCUSDT", "side": "buy", "notional": None},
        {"symbol": "BTCUSDT", "side": "buy", "notional": "bad"},
        {"symbol": "BTCUSDT", "side": "buy", "notional": float("nan")},
        {"symbol": "BTCUSDT", "side": "buy", "notional": -1},
        {"symbol": "BTCUSDT", "side": "teleport", "notional": 100_000},
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "notional": 100_000,
            "price": 0,
            "quantity": 0,
        },
    ],
)
async def test_tracker_rejects_malformed_large_trade_payloads_without_state_mutation(
    whale_tracker: WhaleTracker,
    payload: Any,
) -> None:
    result = await whale_tracker.process_large_trade_payload(payload)

    assert result.has_signals is False
    _assert_tracker_state_empty(whale_tracker)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "not-a-dict",
        {"data": None},
        {"data": []},
        {"symbol": ""},
        {"symbol": "   "},
        {"symbol": "BTCUSDT", "side": "bad", "notional": 100_000},
        {"symbol": "BTCUSDT", "side": "sell", "notional": None},
        {"symbol": "BTCUSDT", "side": "sell", "notional": float("nan")},
        {"symbol": "BTCUSDT", "side": "sell", "notional": -1},
        {"symbol": "BTCUSDT", "side": "sell", "price": 0, "quantity": 0},
    ],
)
async def test_tracker_rejects_malformed_liquidation_payloads_without_state_mutation(
    whale_tracker: WhaleTracker,
    payload: Any,
) -> None:
    result = await whale_tracker.process_liquidation_payload(
        payload,
        source_topic=LIQUIDATIONS_UPDATED_TOPIC,
    )

    assert result is None
    _assert_tracker_state_empty(whale_tracker)


async def test_tracker_event_handlers_do_not_raise_on_hostile_payloads(
    whale_tracker: WhaleTracker,
) -> None:
    hostile_payloads: list[Any] = [
        None,
        [],
        "bad",
        {"data": None},
        {"symbol": "", "side": "buy", "notional": 100_000},
        {"symbol": "BTCUSDT", "side": "bad", "notional": 100_000},
    ]

    for payload in hostile_payloads:
        await whale_tracker.handle_large_trade_event(
            Event(topic=LARGE_TRADE_TOPIC, payload=payload, source="test")
        )
        await whale_tracker.handle_liquidation_event(
            Event(topic=LIQUIDATIONS_UPDATED_TOPIC, payload=payload, source="test")
        )

    _assert_tracker_state_empty(whale_tracker)


# =============================================================================
# Whale activity / pressure detection
# =============================================================================

async def test_single_large_trade_creates_state_but_does_not_emit_activity_or_pressure(
    whale_tracker: WhaleTracker,
    large_trade_payload_factory,
) -> None:
    result = await whale_tracker.process_large_trade_payload(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=100_000.0,
            price=100.0,
            quantity=1_000.0,
            trade_id="single-large-trade",
        )
    )

    assert result.has_signals is False

    state = _default_scope_state(whale_tracker, "BTCUSDT")

    assert state["exists"] is True
    assert state["large_trades_buffer_size"] == 1
    assert state["total_large_trades_seen"] == 1
    assert state["whale_activity_signals_emitted"] == 0
    assert state["whale_pressure_signals_emitted"] == 0


async def test_two_same_side_large_trades_emit_activity_and_pressure(
    whale_tracker: WhaleTracker,
    large_trade_payload_factory,
) -> None:
    first, second = await _feed_large_trades(
        whale_tracker,
        large_trade_payload_factory,
        side="buy",
        count=2,
        notional=60_000.0,
    )

    assert first.has_signals is False
    assert second.has_signals is True

    assert second.whale_activity_signal is not None
    assert second.whale_pressure_signal is not None
    assert second.whale_liquidation_context_signal is None

    activity = second.whale_activity_signal
    pressure = second.whale_pressure_signal

    assert activity.symbol == "BTCUSDT"
    assert activity.exchange == "binance"
    assert activity.market_type == "usdm_futures"
    assert activity.timeframe == "realtime"
    assert activity.side == "buy"
    assert activity.trade_count == 2
    assert activity.total_notional == 120_000.0

    assert pressure.symbol == "BTCUSDT"
    assert pressure.dominant_side == "buy"
    assert pressure.buy_trade_count == 2
    assert pressure.sell_trade_count == 0
    assert pressure.buy_notional == 120_000.0
    assert pressure.net_flow_notional == 120_000.0
    assert 0.0 <= pressure.imbalance_ratio <= 1.0

    state = _default_scope_state(whale_tracker, "BTCUSDT")
    assert state["whale_activity_signals_emitted"] == 1
    assert state["whale_pressure_signals_emitted"] == 1


async def test_balanced_buy_sell_flow_emits_no_activity_and_no_pressure(
    whale_tracker: WhaleTracker,
    large_trade_payload_factory,
) -> None:
    first = await whale_tracker.process_large_trade_payload(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=60_000.0,
            price=100.0,
            quantity=600.0,
            trade_id="balanced-buy",
        )
    )
    second = await whale_tracker.process_large_trade_payload(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=60_000.0,
            price=100.0,
            quantity=600.0,
            trade_id="balanced-sell",
        )
    )

    assert first.has_signals is False
    assert second.has_signals is False

    state = _default_scope_state(whale_tracker, "BTCUSDT")

    assert state["exists"] is True
    assert state["large_trades_buffer_size"] == 2
    assert state["whale_activity_signals_emitted"] == 0
    assert state["whale_pressure_signals_emitted"] == 0


async def test_pressure_detects_dominant_sell_flow_after_mixed_sequence(
    whale_tracker: WhaleTracker,
    large_trade_payload_factory,
) -> None:
    await whale_tracker.process_large_trade_payload(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=20_000.0,
            price=100.0,
            quantity=200.0,
            trade_id="mixed-buy-small",
        )
    )

    result = await whale_tracker.process_large_trade_payload(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=120_000.0,
            price=100.0,
            quantity=1_200.0,
            trade_id="mixed-sell-large",
        )
    )

    assert result.whale_activity_signal is None
    assert result.whale_pressure_signal is not None

    pressure = result.whale_pressure_signal

    assert pressure.dominant_side == "sell"
    assert pressure.sell_notional == 120_000.0
    assert pressure.buy_notional == 20_000.0
    assert pressure.net_flow_notional < 0
    assert pressure.imbalance_ratio >= 0.60


async def test_activity_and_pressure_cooldowns_block_duplicates_but_state_keeps_growing(
    whale_tracker_config_fast: WhaleTrackerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    large_trade_payload_factory,
) -> None:
    config = replace(
        whale_tracker_config_fast,
        whale_activity_cooldown_sec=60.0,
        whale_pressure_cooldown_sec=60.0,
    )
    tracker = _build_tracker(config=config, event_bus=event_bus, scheduler=scheduler)

    await _feed_large_trades(
        tracker,
        large_trade_payload_factory,
        side="buy",
        count=2,
        notional=60_000.0,
    )

    duplicate_result = await tracker.process_large_trade_payload(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=70_000.0,
            price=100.0,
            quantity=700.0,
            trade_id="cooldown-duplicate",
        )
    )

    assert duplicate_result.has_signals is False

    state = _default_scope_state(tracker, "BTCUSDT")

    assert state["large_trades_buffer_size"] == 3
    assert state["total_large_trades_seen"] == 3
    assert state["whale_activity_signals_emitted"] == 1
    assert state["whale_pressure_signals_emitted"] == 1


# =============================================================================
# Liquidation context / production and legacy topics
# =============================================================================

async def test_liquidation_alone_is_stored_but_context_is_not_emitted(
    whale_tracker: WhaleTracker,
    liquidations_updated_payload_factory,
) -> None:
    result = await whale_tracker.process_liquidation_payload(
        liquidations_updated_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=100_000.0,
            price=100.0,
            quantity=1_000.0,
        ),
        source_topic=LIQUIDATIONS_UPDATED_TOPIC,
    )

    assert result is None

    state = _default_scope_state(whale_tracker, "BTCUSDT")

    assert state["exists"] is True
    assert state["liquidations_buffer_size"] == 1
    assert state["total_liquidations_seen"] == 1
    assert state["whale_liquidation_context_signals_emitted"] == 0


async def test_whale_trades_plus_opposite_production_liquidation_emit_context(
    whale_tracker: WhaleTracker,
    large_trade_payload_factory,
    liquidations_updated_payload_factory,
) -> None:
    await _feed_large_trades(
        whale_tracker,
        large_trade_payload_factory,
        side="buy",
        count=2,
        notional=80_000.0,
    )

    context = await whale_tracker.process_liquidation_payload(
        liquidations_updated_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=90_000.0,
            price=100.0,
            quantity=900.0,
        ),
        source_topic=LIQUIDATIONS_UPDATED_TOPIC,
    )

    assert context is not None
    assert context.symbol == "BTCUSDT"
    assert context.exchange == "binance"
    assert context.market_type == "usdm_futures"
    assert context.timeframe == "realtime"
    assert context.whale_side == "buy"
    assert context.liquidation_side == "sell"
    assert context.whale_total_notional >= 160_000.0
    assert context.liquidation_total_notional >= 90_000.0
    assert 0.0 <= context.context_strength <= 1.0

    state = _default_scope_state(whale_tracker, "BTCUSDT")
    assert state["whale_liquidation_context_signals_emitted"] == 1


async def test_liquidation_context_not_emitted_when_liquidations_same_side_as_whales(
    whale_tracker: WhaleTracker,
    large_trade_payload_factory,
    liquidations_updated_payload_factory,
) -> None:
    await _feed_large_trades(
        whale_tracker,
        large_trade_payload_factory,
        side="buy",
        count=2,
        notional=80_000.0,
    )

    context = await whale_tracker.process_liquidation_payload(
        liquidations_updated_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=120_000.0,
            price=100.0,
            quantity=1_200.0,
        ),
        source_topic=LIQUIDATIONS_UPDATED_TOPIC,
    )

    assert context is None

    state = _default_scope_state(whale_tracker, "BTCUSDT")

    assert state["liquidations_buffer_size"] == 1
    assert state["whale_liquidation_context_signals_emitted"] == 0


async def test_raw_market_liquidation_is_ignored_when_legacy_topics_disabled(
    whale_tracker: WhaleTracker,
    event_bus: EventBus,
    event_collector,
    raw_liquidation_payload_factory,
) -> None:
    await whale_tracker.start()

    accepted = await event_bus.emit(
        RAW_LIQUIDATION_TOPIC,
        raw_liquidation_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=100_000.0,
            price=100.0,
            quantity=1_000.0,
        ),
        source="tests.liquidation_stream",
        correlation_id="corr-raw-liquidation-disabled",
    )

    assert accepted is True

    await asyncio.sleep(0.05)

    assert event_collector.by_topic(WHALE_LIQUIDATION_CONTEXT_TOPIC) == []
    _assert_tracker_state_empty(whale_tracker)


async def test_raw_market_liquidation_is_processed_when_legacy_topics_enabled(
    whale_tracker_legacy: WhaleTracker,
    event_bus: EventBus,
    raw_liquidation_payload_factory,
) -> None:
    await whale_tracker_legacy.start()

    accepted = await event_bus.emit(
        RAW_LIQUIDATION_TOPIC,
        raw_liquidation_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=100_000.0,
            price=100.0,
            quantity=1_000.0,
        ),
        source="tests.liquidation_stream",
        correlation_id="corr-raw-liquidation-enabled",
    )

    assert accepted is True

    await asyncio.sleep(0.05)

    state = _default_scope_state(whale_tracker_legacy, "BTCUSDT")

    assert state["exists"] is True
    assert state["liquidations_buffer_size"] == 1
    assert state["total_liquidations_seen"] == 1


async def test_handle_raw_liquidation_event_respects_legacy_guard_even_if_called_directly(
    whale_tracker: WhaleTracker,
    raw_liquidation_payload_factory,
) -> None:
    event = Event(
        topic=RAW_LIQUIDATION_TOPIC,
        payload=raw_liquidation_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=100_000.0,
        ),
        source="test",
    )

    await whale_tracker.handle_raw_liquidation_event(event)

    _assert_tracker_state_empty(whale_tracker)


@pytest.mark.xfail(
    reason=(
        "Current WhaleTracker normalizes only one liquidation from "
        "liquidations=[...]. This documents desired future batch behavior."
    ),
    strict=False,
)
async def test_liquidations_updated_batch_should_process_all_valid_liquidations_in_future(
    whale_tracker: WhaleTracker,
    large_trade_payload_factory,
    liquidations_updated_payload_factory,
    raw_liquidation_payload_factory,
) -> None:
    await _feed_large_trades(
        whale_tracker,
        large_trade_payload_factory,
        side="buy",
        count=2,
        notional=80_000.0,
    )

    liquidations = [
        raw_liquidation_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=90_000.0,
            liquidation_id=f"future-batch-liq-{index}",
        )
        for index in range(3)
    ]

    await whale_tracker.process_liquidation_payload(
        liquidations_updated_payload_factory(
            symbol="BTCUSDT",
            liquidations=liquidations,
            batch_id="future-liquidation-batch",
        ),
        source_topic=LIQUIDATIONS_UPDATED_TOPIC,
    )

    state = _default_scope_state(whale_tracker, "BTCUSDT")

    assert state["liquidations_buffer_size"] == 3
    assert state["total_liquidations_seen"] == 3


# =============================================================================
# EventBus emissions
# =============================================================================

async def test_direct_processing_publishes_activity_pressure_and_context_events(
    whale_tracker: WhaleTracker,
    event_collector,
    large_trade_payload_factory,
    liquidations_updated_payload_factory,
) -> None:
    await _feed_large_trades(
        whale_tracker,
        large_trade_payload_factory,
        side="buy",
        count=2,
        notional=80_000.0,
    )

    await whale_tracker.process_liquidation_payload(
        liquidations_updated_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=90_000.0,
            price=100.0,
            quantity=900.0,
        ),
        source_topic=LIQUIDATIONS_UPDATED_TOPIC,
        correlation_id="corr-direct-tracker",
        source_event_id="liq-event-1",
    )

    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1, timeout=1.0)
    await event_collector.wait_for_topic(WHALE_PRESSURE_TOPIC, count=1, timeout=1.0)
    await event_collector.wait_for_topic(
        WHALE_LIQUIDATION_CONTEXT_TOPIC,
        count=1,
        timeout=1.0,
    )

    activity = event_collector.by_topic(WHALE_ACTIVITY_TOPIC)[0]
    pressure = event_collector.by_topic(WHALE_PRESSURE_TOPIC)[0]
    context = event_collector.by_topic(WHALE_LIQUIDATION_CONTEXT_TOPIC)[0]

    assert activity.payload["symbol"] == "BTCUSDT"
    assert activity.payload["side"] == "buy"

    assert pressure.payload["symbol"] == "BTCUSDT"
    assert pressure.payload["dominant_side"] == "buy"

    assert context.payload["symbol"] == "BTCUSDT"
    assert context.payload["whale_side"] == "buy"
    assert context.payload["liquidation_side"] == "sell"
    assert context.correlation_id == "corr-direct-tracker"


async def test_eventbus_large_trade_input_emits_activity_and_pressure(
    whale_tracker: WhaleTracker,
    event_bus: EventBus,
    event_collector,
    large_trade_payload_factory,
) -> None:
    await whale_tracker.start()

    for index in range(2):
        accepted = await event_bus.emit(
            LARGE_TRADE_TOPIC,
            large_trade_payload_factory(
                symbol="BTCUSDT",
                side="buy",
                notional=80_000.0,
                price=100.0,
                quantity=800.0,
                trade_id=f"eventbus-large-trade-{index}",
            ),
            source="tests.large_trade_detector",
            correlation_id="corr-large-trade-input",
        )
        assert accepted is True

    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1, timeout=1.0)
    await event_collector.wait_for_topic(WHALE_PRESSURE_TOPIC, count=1, timeout=1.0)

    activity = event_collector.by_topic(WHALE_ACTIVITY_TOPIC)[0]
    pressure = event_collector.by_topic(WHALE_PRESSURE_TOPIC)[0]

    assert activity.correlation_id == "corr-large-trade-input"
    assert pressure.correlation_id == "corr-large-trade-input"


async def test_eventbus_production_liquidation_input_emits_context_after_whale_trades(
    whale_tracker: WhaleTracker,
    event_bus: EventBus,
    event_collector,
    large_trade_payload_factory,
    liquidations_updated_payload_factory,
) -> None:
    await whale_tracker.start()

    await _feed_large_trades(
        whale_tracker,
        large_trade_payload_factory,
        side="buy",
        count=2,
        notional=80_000.0,
    )

    accepted = await event_bus.emit(
        LIQUIDATIONS_UPDATED_TOPIC,
        liquidations_updated_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=100_000.0,
            price=100.0,
            quantity=1_000.0,
        ),
        source="tests.liquidation_cache",
        correlation_id="corr-production-liquidation",
    )

    assert accepted is True

    await event_collector.wait_for_topic(
        WHALE_LIQUIDATION_CONTEXT_TOPIC,
        count=1,
        timeout=1.0,
    )

    context = event_collector.by_topic(WHALE_LIQUIDATION_CONTEXT_TOPIC)[0]

    assert context.correlation_id == "corr-production-liquidation"
    assert context.payload["symbol"] == "BTCUSDT"
    assert context.payload["whale_side"] == "buy"
    assert context.payload["liquidation_side"] == "sell"


async def test_emit_on_bus_false_returns_signals_without_publishing_events(
    whale_tracker_config_fast: WhaleTrackerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    large_trade_payload_factory,
    liquidations_updated_payload_factory,
) -> None:
    config = replace(
        whale_tracker_config_fast,
        emit_on_bus=False,
        log_signals=False,
    )
    tracker = _build_tracker(config=config, event_bus=event_bus, scheduler=scheduler)

    await _feed_large_trades(
        tracker,
        large_trade_payload_factory,
        side="buy",
        count=2,
        notional=80_000.0,
    )
    context = await tracker.process_liquidation_payload(
        liquidations_updated_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=100_000.0,
        ),
        source_topic=LIQUIDATIONS_UPDATED_TOPIC,
    )

    assert context is not None

    await asyncio.sleep(0.05)

    assert event_collector.by_topic(WHALE_ACTIVITY_TOPIC) == []
    assert event_collector.by_topic(WHALE_PRESSURE_TOPIC) == []
    assert event_collector.by_topic(WHALE_LIQUIDATION_CONTEXT_TOPIC) == []


# =============================================================================
# Window pruning / concurrency / scoped isolation
# =============================================================================

async def test_tracker_prunes_old_large_trades_outside_windows(
    whale_tracker_config_fast: WhaleTrackerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    large_trade_payload_factory,
) -> None:
    config = replace(
        whale_tracker_config_fast,
        cluster_window_sec=1,
        pressure_window_sec=1,
    )
    tracker = _build_tracker(config=config, event_bus=event_bus, scheduler=scheduler)

    base_ts = int(time.time() * 1000)

    await tracker.process_large_trade_payload(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=100_000.0,
            timestamp_ms=base_ts,
            trade_id="old-trade",
        )
    )

    result = await tracker.process_large_trade_payload(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=100_000.0,
            timestamp_ms=base_ts + 2_000,
            trade_id="new-trade",
        )
    )

    assert result.has_signals is False

    state = _default_scope_state(tracker, "BTCUSDT")

    assert state["exists"] is True
    assert state["large_trades_buffer_size"] == 1
    assert state["total_large_trades_seen"] == 2


async def test_concurrent_same_scope_inputs_do_not_corrupt_state_or_duplicate_signals(
    whale_tracker_config_fast: WhaleTrackerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    large_trade_payload_factory,
) -> None:
    config = replace(
        whale_tracker_config_fast,
        whale_activity_cooldown_sec=60.0,
        whale_pressure_cooldown_sec=60.0,
    )
    tracker = _build_tracker(config=config, event_bus=event_bus, scheduler=scheduler)

    payloads = [
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=60_000.0 + index,
            price=100.0,
            quantity=(60_000.0 + index) / 100.0,
            trade_id=f"race-same-scope-{index}",
        )
        for index in range(50)
    ]

    results = await asyncio.gather(
        *(tracker.process_large_trade_payload(payload) for payload in payloads)
    )

    state = _default_scope_state(tracker, "BTCUSDT")

    assert state["exists"] is True
    assert state["total_large_trades_seen"] == 50
    assert state["large_trades_buffer_size"] == 50

    assert state["whale_activity_signals_emitted"] <= 1
    assert state["whale_pressure_signals_emitted"] <= 1

    assert sum(result.whale_activity_signal is not None for result in results) <= 1
    assert sum(result.whale_pressure_signal is not None for result in results) <= 1


async def test_concurrent_different_scopes_do_not_cross_contaminate_state(
    whale_tracker: WhaleTracker,
    large_trade_payload_factory,
) -> None:
    payloads = [
        large_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="realtime",
            side="buy",
            notional=80_000.0,
            trade_id="binance-btc-1",
        ),
        large_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="realtime",
            side="buy",
            notional=80_000.0,
            trade_id="binance-btc-2",
        ),
        large_trade_payload_factory(
            exchange="okx",
            market_type="swap",
            symbol="BTCUSDT",
            timeframe="realtime",
            side="sell",
            notional=80_000.0,
            trade_id="okx-btc-1",
        ),
        large_trade_payload_factory(
            exchange="okx",
            market_type="swap",
            symbol="BTCUSDT",
            timeframe="realtime",
            side="sell",
            notional=80_000.0,
            trade_id="okx-btc-2",
        ),
        large_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="ETHUSDT",
            timeframe="realtime",
            side="buy",
            notional=80_000.0,
            trade_id="binance-eth-1",
        ),
        large_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="ETHUSDT",
            timeframe="realtime",
            side="buy",
            notional=80_000.0,
            trade_id="binance-eth-2",
        ),
    ]

    await asyncio.gather(
        *(whale_tracker.process_large_trade_payload(payload) for payload in payloads)
    )

    all_states = whale_tracker.get_all_states()

    assert len(all_states) == 3

    binance_btc = _default_scope_state(
        whale_tracker,
        "BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    okx_btc = _default_scope_state(
        whale_tracker,
        "BTCUSDT",
        exchange="okx",
        market_type="swap",
        timeframe="realtime",
    )
    binance_eth = _default_scope_state(
        whale_tracker,
        "ETHUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert binance_btc["exists"] is True
    assert okx_btc["exists"] is True
    assert binance_eth["exists"] is True

    assert binance_btc["whale_pressure_signals_emitted"] == 1
    assert okx_btc["whale_pressure_signals_emitted"] == 1
    assert binance_eth["whale_pressure_signals_emitted"] == 1


# =============================================================================
# Cleanup / reset / health
# =============================================================================

async def test_tracker_cleanup_removes_stale_scoped_states(
    whale_tracker_config_fast: WhaleTrackerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    large_trade_payload_factory,
) -> None:
    config = replace(whale_tracker_config_fast, stats_ttl_sec=0.01)
    tracker = _build_tracker(config=config, event_bus=event_bus, scheduler=scheduler)

    await tracker.process_large_trade_payload(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=100_000.0,
        )
    )
    await tracker.process_large_trade_payload(
        large_trade_payload_factory(
            symbol="ETHUSDT",
            side="sell",
            notional=100_000.0,
        )
    )

    assert len(tracker.get_all_states()) == 2

    await asyncio.sleep(0.02)
    await tracker.cleanup()

    assert tracker.get_all_states() == {}


async def test_tracker_reset_symbol_is_scoped_when_scope_is_provided(
    whale_tracker: WhaleTracker,
    large_trade_payload_factory,
) -> None:
    await whale_tracker.process_large_trade_payload(
        large_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            side="buy",
            notional=100_000.0,
            trade_id="reset-binance-btc",
        )
    )
    await whale_tracker.process_large_trade_payload(
        large_trade_payload_factory(
            exchange="okx",
            market_type="swap",
            symbol="BTCUSDT",
            side="sell",
            notional=100_000.0,
            trade_id="reset-okx-btc",
        )
    )
    await whale_tracker.process_large_trade_payload(
        large_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="ETHUSDT",
            side="buy",
            notional=100_000.0,
            trade_id="reset-binance-eth",
        )
    )

    assert len(whale_tracker.get_all_states()) == 3

    await whale_tracker.reset_symbol(
        "BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert (
        _default_scope_state(
            whale_tracker,
            "BTCUSDT",
            exchange="binance",
            market_type="usdm_futures",
            timeframe="realtime",
        )["exists"]
        is False
    )
    assert (
        _default_scope_state(
            whale_tracker,
            "BTCUSDT",
            exchange="okx",
            market_type="swap",
            timeframe="realtime",
        )["exists"]
        is True
    )
    assert _default_scope_state(whale_tracker, "ETHUSDT")["exists"] is True

    await whale_tracker.reset_symbol("BTCUSDT")

    assert whale_tracker.get_symbol_state("BTCUSDT")["exists"] is False
    assert _default_scope_state(whale_tracker, "ETHUSDT")["exists"] is True

    await whale_tracker.reset_all()

    assert whale_tracker.get_all_states() == {}


async def test_tracker_invalid_symbol_state_api_is_safe(
    whale_tracker: WhaleTracker,
) -> None:
    assert whale_tracker.get_symbol_state("")["error"] == "invalid_symbol"
    assert whale_tracker.get_symbol_state("   ")["error"] == "invalid_symbol"


async def test_tracker_healthcheck_reports_per_key_locking_and_topics(
    whale_tracker: WhaleTracker,
    large_trade_payload_factory,
) -> None:
    await whale_tracker.process_large_trade_payload(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=100_000.0,
        )
    )

    health = whale_tracker.get_healthcheck()

    assert health["component"] == "whale_tracker"
    assert health["event_bus_available"] is True
    assert health["scheduler_available"] is True
    assert health["enabled"] is True
    assert health["tracked_scopes"] >= 1
    assert health["state_locks"] >= 1
    assert health["locking"] == "per_whale_key"
    assert health["scope"] == "exchange:market_type:symbol:timeframe"
    assert LARGE_TRADE_TOPIC in health["production_input_topics"]
    assert health["subscribe_liquidations"] is True
    assert health["allow_legacy_raw_topics"] is False