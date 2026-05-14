from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any

import pytest

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from analytics.whales.config import WhaleClusterAnalyzerConfig, WhaleTrackerConfig
from analytics.whales.whale_cluster_analyzer import WhaleClusterAnalyzer
from analytics.whales.whale_tracker import WhaleTracker


pytestmark = pytest.mark.asyncio


LARGE_TRADE_TOPIC = "analytics.whales.large_trade"
MARKET_LIQUIDATION_TOPIC = "market.liquidation"

WHALE_ACTIVITY_TOPIC = "analytics.whales.whale_activity"
WHALE_PRESSURE_TOPIC = "analytics.whales.whale_pressure"
WHALE_LIQUIDATION_CONTEXT_TOPIC = "analytics.whales.whale_liquidation_context"

WHALE_CLUSTER_TOPIC = "analytics.whales.whale_cluster"
WHALE_CLUSTER_UPDATE_TOPIC = "analytics.whales.whale_cluster_update"
WHALE_CLUSTER_EXHAUSTION_TOPIC = "analytics.whales.whale_cluster_exhaustion"


# =============================================================================
# Local builders
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


def _build_cluster_analyzer(
    *,
    config: WhaleClusterAnalyzerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> WhaleClusterAnalyzer:
    return WhaleClusterAnalyzer(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )


async def _feed_large_trades(
    tracker: WhaleTracker,
    large_trade_payload_factory,
    *,
    symbol: str = "BTCUSDT",
    side: str = "buy",
    count: int = 2,
    notional: float = 60_000.0,
    start_ts_ms: int | None = None,
) -> list[Any]:
    ts = start_ts_ms or int(time.time() * 1000)
    results: list[Any] = []

    for index in range(count):
        result = await tracker.process_large_trade_payload(
            large_trade_payload_factory(
                symbol=symbol,
                side=side,
                price=100.0,
                quantity=notional / 100.0,
                notional=notional,
                timestamp_ms=ts + index,
                trade_id=f"{symbol}-{side}-{index}",
            )
        )
        results.append(result)

    return results


# =============================================================================
# WhaleTracker lifecycle / core behavior
# =============================================================================


async def test_tracker_register_is_idempotent_and_subscribes_expected_topics(
    whale_tracker: WhaleTracker,
) -> None:
    await whale_tracker.register()
    await whale_tracker.register()
    await whale_tracker.register()

    assert whale_tracker.is_registered is True
    assert len(whale_tracker.subscriptions) == 2

    patterns = {subscription.pattern for subscription in whale_tracker.subscriptions}
    assert patterns == {LARGE_TRADE_TOPIC, MARKET_LIQUIDATION_TOPIC}


async def test_tracker_register_without_liquidations_subscribes_only_large_trade(
    whale_tracker_config_fast: WhaleTrackerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    config = replace(whale_tracker_config_fast, subscribe_liquidations=False)
    tracker = _build_tracker(config=config, event_bus=event_bus, scheduler=scheduler)

    await tracker.register()

    assert tracker.is_registered is True
    assert len(tracker.subscriptions) == 1
    assert tracker.subscriptions[0].pattern == LARGE_TRADE_TOPIC


async def test_tracker_start_adds_exactly_one_cleanup_job_and_stop_removes_it(
    whale_tracker: WhaleTracker,
) -> None:
    await whale_tracker.start()
    await whale_tracker.start()

    assert whale_tracker.is_started is True
    assert whale_tracker.is_registered is True
    assert len(whale_tracker.subscriptions) == 2
    assert len(whale_tracker.scheduler_job_ids) == 1

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
    liquidation_payload_factory,
) -> None:
    config = replace(whale_tracker_config_fast, enabled=False)
    tracker = _build_tracker(config=config, event_bus=event_bus, scheduler=scheduler)

    await tracker.register()
    await tracker.start()

    assert tracker.is_registered is False
    assert tracker.is_started is False

    result = await tracker.process_large_trade_payload(
        large_trade_payload_factory(notional=200_000.0)
    )
    liquidation_result = await tracker.process_liquidation_payload(
        liquidation_payload_factory(notional=200_000.0)
    )

    assert result.has_signals is False
    assert liquidation_result is None
    assert tracker.get_all_states() == {}

    await asyncio.sleep(0.05)
    assert event_collector.by_topic(WHALE_ACTIVITY_TOPIC) == []
    assert event_collector.by_topic(WHALE_PRESSURE_TOPIC) == []
    assert event_collector.by_topic(WHALE_LIQUIDATION_CONTEXT_TOPIC) == []


# =============================================================================
# WhaleTracker normalization / hostile payloads
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
    assert tracker_state_is_empty(whale_tracker)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "not-a-dict",
        {"data": None},
        {"symbol": ""},
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
    result = await whale_tracker.process_liquidation_payload(payload)

    assert result is None
    assert tracker_state_is_empty(whale_tracker)


def tracker_state_is_empty(tracker: WhaleTracker) -> bool:
    return tracker.get_all_states() == {}


# =============================================================================
# WhaleTracker activity / pressure / liquidation context
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
        )
    )

    assert result.has_signals is False

    state = whale_tracker.get_symbol_state("btcusdt")
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

    activity = second.whale_activity_signal
    pressure = second.whale_pressure_signal

    assert activity.symbol == "BTCUSDT"
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

    state = whale_tracker.get_symbol_state("BTCUSDT")
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

    state = whale_tracker.get_symbol_state("BTCUSDT")
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

    state = tracker.get_symbol_state("BTCUSDT")
    assert state["large_trades_buffer_size"] == 3
    assert state["total_large_trades_seen"] == 3
    assert state["whale_activity_signals_emitted"] == 1
    assert state["whale_pressure_signals_emitted"] == 1


async def test_liquidation_alone_is_stored_but_context_is_not_emitted(
    whale_tracker: WhaleTracker,
    liquidation_payload_factory,
) -> None:
    result = await whale_tracker.process_liquidation_payload(
        liquidation_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=100_000.0,
            price=100.0,
            quantity=1_000.0,
        )
    )

    assert result is None

    state = whale_tracker.get_symbol_state("BTCUSDT")
    assert state["exists"] is True
    assert state["liquidations_buffer_size"] == 1
    assert state["total_liquidations_seen"] == 1
    assert state["whale_liquidation_context_signals_emitted"] == 0


async def test_whale_trades_plus_opposite_liquidations_emit_liquidation_context(
    whale_tracker: WhaleTracker,
    large_trade_payload_factory,
    liquidation_payload_factory,
) -> None:
    await _feed_large_trades(
        whale_tracker,
        large_trade_payload_factory,
        side="buy",
        count=2,
        notional=80_000.0,
    )

    context = await whale_tracker.process_liquidation_payload(
        liquidation_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=90_000.0,
            price=100.0,
            quantity=900.0,
        )
    )

    assert context is not None
    assert context.symbol == "BTCUSDT"
    assert context.whale_side == "buy"
    assert context.liquidation_side == "sell"
    assert context.whale_total_notional >= 160_000.0
    assert context.liquidation_total_notional >= 90_000.0
    assert 0.0 <= context.context_strength <= 1.0

    state = whale_tracker.get_symbol_state("BTCUSDT")
    assert state["whale_liquidation_context_signals_emitted"] == 1


async def test_liquidation_context_not_emitted_when_liquidations_same_side_as_whales(
    whale_tracker: WhaleTracker,
    large_trade_payload_factory,
    liquidation_payload_factory,
) -> None:
    await _feed_large_trades(
        whale_tracker,
        large_trade_payload_factory,
        side="buy",
        count=2,
        notional=80_000.0,
    )

    context = await whale_tracker.process_liquidation_payload(
        liquidation_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=120_000.0,
            price=100.0,
            quantity=1_200.0,
        )
    )

    assert context is None

    state = whale_tracker.get_symbol_state("BTCUSDT")
    assert state["liquidations_buffer_size"] == 1
    assert state["whale_liquidation_context_signals_emitted"] == 0


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

    state = tracker.get_symbol_state("BTCUSDT")
    assert state["exists"] is True
    assert state["large_trades_buffer_size"] == 1
    assert state["total_large_trades_seen"] == 2


async def test_tracker_cleanup_reset_and_invalid_symbol_state_api(
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

    assert tracker.get_healthcheck()["tracked_symbols"] == 2

    assert tracker.get_symbol_state("")["error"] == "invalid_symbol"
    assert tracker.get_symbol_state("   ")["error"] == "invalid_symbol"

    await tracker.reset_symbol("btcusdt")
    assert tracker.get_symbol_state("BTCUSDT")["exists"] is False
    assert tracker.get_symbol_state("ETHUSDT")["exists"] is True

    await asyncio.sleep(0.02)
    await tracker.cleanup()

    assert tracker.get_all_states() == {}

    await tracker.process_large_trade_payload(
        large_trade_payload_factory(symbol="SOLUSDT", notional=100_000.0)
    )
    assert tracker.get_symbol_state("SOLUSDT")["exists"] is True

    await tracker.reset_all()
    assert tracker.get_all_states() == {}


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
            Event(topic=MARKET_LIQUIDATION_TOPIC, payload=payload, source="test")
        )

    assert tracker_state_is_empty(whale_tracker)


async def test_tracker_concurrent_same_symbol_inputs_do_not_corrupt_state_or_duplicate_context(
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
            trade_id=f"race-{index}",
        )
        for index in range(50)
    ]

    results = await asyncio.gather(
        *(tracker.process_large_trade_payload(payload) for payload in payloads)
    )

    state = tracker.get_symbol_state("BTCUSDT")
    assert state["exists"] is True
    assert state["total_large_trades_seen"] == 50
    assert state["large_trades_buffer_size"] == 50

    # Через cooldown має бути не більше одного activity/pressure signal.
    assert state["whale_activity_signals_emitted"] <= 1
    assert state["whale_pressure_signals_emitted"] <= 1

    assert sum(result.whale_activity_signal is not None for result in results) <= 1
    assert sum(result.whale_pressure_signal is not None for result in results) <= 1


# =============================================================================
# WhaleTracker EventBus emissions
# =============================================================================


async def test_tracker_direct_processing_publishes_activity_pressure_and_context_events(
    whale_tracker: WhaleTracker,
    event_collector,
    large_trade_payload_factory,
    liquidation_payload_factory,
) -> None:
    await _feed_large_trades(
        whale_tracker,
        large_trade_payload_factory,
        side="buy",
        count=2,
        notional=80_000.0,
    )

    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1)
    await event_collector.wait_for_topic(WHALE_PRESSURE_TOPIC, count=1)

    context = await whale_tracker.process_liquidation_payload(
        liquidation_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=90_000.0,
        )
    )

    assert context is not None
    await event_collector.wait_for_topic(WHALE_LIQUIDATION_CONTEXT_TOPIC, count=1)

    activity_payload = event_collector.payloads_by_topic(WHALE_ACTIVITY_TOPIC)[0]
    pressure_payload = event_collector.payloads_by_topic(WHALE_PRESSURE_TOPIC)[0]
    context_payload = event_collector.payloads_by_topic(
        WHALE_LIQUIDATION_CONTEXT_TOPIC
    )[0]

    assert activity_payload["symbol"] == "BTCUSDT"
    assert activity_payload["side"] == "buy"
    assert pressure_payload["dominant_side"] == "buy"
    assert context_payload["whale_side"] == "buy"
    assert context_payload["liquidation_side"] == "sell"


async def test_tracker_registered_eventbus_handler_preserves_correlation_id(
    whale_tracker: WhaleTracker,
    event_bus: EventBus,
    event_collector,
    large_trade_payload_factory,
) -> None:
    await whale_tracker.register()

    for index in range(2):
        accepted = await event_bus.emit(
            LARGE_TRADE_TOPIC,
            large_trade_payload_factory(
                symbol="BTCUSDT",
                side="buy",
                notional=80_000.0,
                trade_id=f"bus-tracker-{index}",
            ),
            source="tests.large_trade_detector",
            correlation_id="corr-tracker-1",
        )
        assert accepted is True

    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1)

    emitted = event_collector.by_topic(WHALE_ACTIVITY_TOPIC)[0]
    assert emitted.correlation_id == "corr-tracker-1"
    assert emitted.payload["symbol"] == "BTCUSDT"


# =============================================================================
# WhaleClusterAnalyzer lifecycle
# =============================================================================


async def test_cluster_analyzer_register_is_idempotent_and_subscribes_three_topics(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
) -> None:
    await whale_cluster_analyzer.register()
    await whale_cluster_analyzer.register()
    await whale_cluster_analyzer.register()

    assert whale_cluster_analyzer.is_registered is True
    assert len(whale_cluster_analyzer.subscriptions) == 3

    patterns = {
        subscription.pattern for subscription in whale_cluster_analyzer.subscriptions
    }
    assert patterns == {
        WHALE_ACTIVITY_TOPIC,
        WHALE_PRESSURE_TOPIC,
        WHALE_LIQUIDATION_CONTEXT_TOPIC,
    }


async def test_cluster_analyzer_start_adds_cleanup_job_and_stop_removes_it(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
) -> None:
    await whale_cluster_analyzer.start()
    await whale_cluster_analyzer.start()

    assert whale_cluster_analyzer.is_started is True
    assert whale_cluster_analyzer.is_registered is True
    assert len(whale_cluster_analyzer.subscriptions) == 3
    assert len(whale_cluster_analyzer.scheduler_job_ids) == 1

    await whale_cluster_analyzer.stop()

    assert whale_cluster_analyzer.is_started is False
    assert whale_cluster_analyzer.is_registered is False
    assert len(whale_cluster_analyzer.subscriptions) == 0
    assert len(whale_cluster_analyzer.scheduler_job_ids) == 0


async def test_disabled_cluster_analyzer_does_not_register_or_emit(
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    whale_activity_payload_factory,
) -> None:
    config = replace(whale_cluster_analyzer_config_fast, enabled=False)
    analyzer = _build_cluster_analyzer(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    await analyzer.register()
    await analyzer.start()

    result = await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(total_notional=1_000_000.0)
    )

    assert analyzer.is_registered is False
    assert analyzer.is_started is False
    assert result.has_signals is False
    assert analyzer.get_all_states() == {}

    await asyncio.sleep(0.05)
    assert event_collector.by_topic(WHALE_CLUSTER_TOPIC) == []


# =============================================================================
# WhaleClusterAnalyzer normalization / hostile payloads
# =============================================================================


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "bad",
        {"data": None},
        {"symbol": ""},
        {"symbol": "BTCUSDT", "side": "bad", "trade_count": 2, "total_notional": 100_000},
        {"symbol": "BTCUSDT", "side": "buy", "trade_count": 0, "total_notional": 100_000},
        {"symbol": "BTCUSDT", "side": "buy", "trade_count": 2, "total_notional": -1},
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "trade_count": 2,
            "total_notional": 100_000,
            "avg_notional": 0,
            "max_notional": 50_000,
            "window_sec": 30,
        },
    ],
)
async def test_cluster_analyzer_rejects_malformed_activity_payloads_without_state(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    payload: Any,
) -> None:
    result = await whale_cluster_analyzer.process_whale_activity_payload(payload)

    assert result.has_signals is False
    assert whale_cluster_analyzer.get_all_states() == {}


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "bad",
        {"data": None},
        {"symbol": ""},
        {"symbol": "BTCUSDT", "dominant_side": "bad"},
        {"symbol": "BTCUSDT", "dominant_side": "buy", "total_notional": -1},
        {
            "symbol": "BTCUSDT",
            "dominant_side": "buy",
            "buy_trade_count": 1,
            "sell_trade_count": 1,
            "buy_notional": float("nan"),
            "sell_notional": 10_000,
            "total_notional": 20_000,
            "imbalance_ratio": 0.5,
            "net_flow_notional": 0,
            "window_sec": 30,
        },
    ],
)
async def test_cluster_analyzer_rejects_malformed_pressure_payloads_without_state(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    payload: Any,
) -> None:
    result = await whale_cluster_analyzer.process_whale_pressure_payload(payload)

    assert result.has_signals is False
    assert whale_cluster_analyzer.get_all_states() == {}


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "bad",
        {"data": None},
        {"symbol": ""},
        {"symbol": "BTCUSDT", "whale_side": "bad", "liquidation_side": "sell"},
        {
            "symbol": "BTCUSDT",
            "whale_side": "buy",
            "whale_total_notional": -1,
            "whale_trade_count": 2,
            "liquidation_side": "sell",
            "liquidation_total_notional": 100_000,
            "liquidation_count": 1,
            "context_strength": 0.5,
        },
        {
            "symbol": "BTCUSDT",
            "whale_side": "buy",
            "whale_total_notional": 100_000,
            "whale_trade_count": 0,
            "liquidation_side": "sell",
            "liquidation_total_notional": 100_000,
            "liquidation_count": 1,
            "context_strength": 0.5,
        },
    ],
)
async def test_cluster_analyzer_rejects_malformed_liquidation_context_payloads_without_state(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    payload: Any,
) -> None:
    result = await whale_cluster_analyzer.process_whale_liquidation_context_payload(
        payload
    )

    assert result.has_signals is False
    assert whale_cluster_analyzer.get_all_states() == {}


# =============================================================================
# WhaleClusterAnalyzer scoring / cluster/update/exhaustion
# =============================================================================


async def test_cluster_requires_minimum_activity_count_and_notional(
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    whale_activity_payload_factory,
) -> None:
    config = replace(
        whale_cluster_analyzer_config_fast,
        min_activity_signals=2,
        min_total_activity_notional=500_000.0,
    )
    analyzer = _build_cluster_analyzer(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    first = await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=100_000.0,
            trade_count=2,
        )
    )
    second = await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=100_000.0,
            trade_count=2,
        )
    )

    assert first.has_signals is False
    assert second.has_signals is False

    state = analyzer.get_symbol_state("BTCUSDT")
    assert state["exists"] is True
    assert state["activity_records_size"] == 2
    assert state["total_clusters_emitted"] == 0


async def test_activity_above_threshold_emits_cluster_signal_with_probabilities(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    whale_activity_payload_factory,
) -> None:
    result = await whale_cluster_analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            trade_count=3,
            total_notional=200_000.0,
            avg_notional=66_666.0,
            max_notional=90_000.0,
        )
    )

    assert result.whale_cluster_signal is not None

    signal = result.whale_cluster_signal
    assert signal.symbol == "BTCUSDT"
    assert signal.cluster_side == "buy"
    assert 0.0 <= signal.cluster_score <= 1.0
    assert 0.0 <= signal.persistence_score <= 1.0
    assert 0.0 <= signal.directional_bias <= 1.0
    assert 0.0 <= signal.continuation_probability <= 1.0
    assert 0.0 <= signal.exhaustion_probability <= 1.0
    assert signal.activity_signal_count == 1
    assert signal.total_activity_notional >= 200_000.0

    state = whale_cluster_analyzer.get_symbol_state("BTCUSDT")
    assert state["total_clusters_emitted"] == 1


async def test_pressure_and_liquidation_context_increase_cluster_context_counts(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    whale_activity_payload_factory,
    whale_pressure_payload_factory,
    whale_liquidation_context_payload_factory,
) -> None:
    await whale_cluster_analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=200_000.0,
        )
    )

    pressure_result = await whale_cluster_analyzer.process_whale_pressure_payload(
        whale_pressure_payload_factory(
            symbol="BTCUSDT",
            dominant_side="buy",
            buy_notional=220_000.0,
            sell_notional=20_000.0,
            imbalance_ratio=0.92,
        )
    )
    context_result = (
        await whale_cluster_analyzer.process_whale_liquidation_context_payload(
            whale_liquidation_context_payload_factory(
                symbol="BTCUSDT",
                whale_side="buy",
                liquidation_side="sell",
                context_strength=0.95,
            )
        )
    )

    assert pressure_result.has_signals is True
    assert context_result.has_signals is True

    latest_signal = (
        context_result.whale_cluster_signal
        or context_result.whale_cluster_update_signal
        or context_result.whale_cluster_exhaustion_signal
    )
    assert latest_signal is not None

    state = whale_cluster_analyzer.get_symbol_state("BTCUSDT")
    assert state["activity_records_size"] == 1
    assert state["pressure_records_size"] == 1
    assert state["liquidation_context_records_size"] == 1
    assert state["total_events_seen"] == 3


async def test_cluster_update_is_emitted_after_initial_cluster_when_update_cooldown_allows_it(
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    whale_activity_payload_factory,
) -> None:
    config = replace(
        whale_cluster_analyzer_config_fast,
        cluster_emit_cooldown_sec=60.0,
        cluster_update_cooldown_sec=0.0,
        min_cluster_score_to_emit=0.20,
        min_continuation_probability_to_emit=0.20,
    )
    analyzer = _build_cluster_analyzer(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    first = await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=200_000.0,
        )
    )
    second = await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=220_000.0,
        )
    )

    assert first.whale_cluster_signal is not None
    assert second.whale_cluster_signal is None
    assert second.whale_cluster_update_signal is not None

    state = analyzer.get_symbol_state("BTCUSDT")
    assert state["total_clusters_emitted"] == 1
    assert state["total_cluster_updates_emitted"] >= 1


async def test_cluster_exhaustion_can_be_emitted_when_continuation_is_weak_and_context_is_high(
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    whale_activity_payload_factory,
    whale_pressure_payload_factory,
    whale_liquidation_context_payload_factory,
) -> None:
    config = replace(
        whale_cluster_analyzer_config_fast,
        min_cluster_score_to_emit=0.95,
        min_continuation_probability_to_emit=0.95,
        min_exhaustion_probability_to_emit=0.30,
        cluster_exhaustion_cooldown_sec=0.0,
    )
    analyzer = _build_cluster_analyzer(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=120_000.0,
        )
    )
    await analyzer.process_whale_pressure_payload(
        whale_pressure_payload_factory(
            symbol="BTCUSDT",
            dominant_side="sell",
            buy_notional=10_000.0,
            sell_notional=160_000.0,
            imbalance_ratio=0.94,
            net_flow_notional=-150_000.0,
        )
    )
    result = await analyzer.process_whale_liquidation_context_payload(
        whale_liquidation_context_payload_factory(
            symbol="BTCUSDT",
            whale_side="buy",
            liquidation_side="sell",
            context_strength=1.0,
        )
    )

    assert result.whale_cluster_signal is None
    assert result.whale_cluster_exhaustion_signal is not None

    exhaustion = result.whale_cluster_exhaustion_signal
    assert exhaustion.symbol == "BTCUSDT"
    assert 0.0 <= exhaustion.exhaustion_probability <= 1.0
    assert 0.0 <= exhaustion.reversal_risk <= 1.0

    state = analyzer.get_symbol_state("BTCUSDT")
    assert state["total_cluster_exhaustions_emitted"] >= 1


async def test_cluster_cooldowns_block_duplicate_cluster_but_not_state_updates(
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    whale_activity_payload_factory,
) -> None:
    config = replace(
        whale_cluster_analyzer_config_fast,
        cluster_emit_cooldown_sec=60.0,
        cluster_update_cooldown_sec=60.0,
        cluster_exhaustion_cooldown_sec=60.0,
    )
    analyzer = _build_cluster_analyzer(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    first = await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=200_000.0,
            timestamp_ms=1_700_000_000_000,
        )
    )
    second = await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=240_000.0,
            timestamp_ms=1_700_000_001_000,
        )
    )

    assert first.has_signals is True
    assert second.has_signals is False

    state = analyzer.get_symbol_state("BTCUSDT")
    assert state["activity_records_size"] == 2
    assert state["total_events_seen"] == 2
    assert state["total_clusters_emitted"] == 1


async def test_cluster_prunes_records_outside_analysis_window(
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    whale_activity_payload_factory,
) -> None:
    config = replace(
        whale_cluster_analyzer_config_fast,
        analysis_window_sec=1,
        cluster_emit_cooldown_sec=0.0,
        cluster_update_cooldown_sec=0.0,
    )
    analyzer = _build_cluster_analyzer(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    base_ts = 1_700_000_000_000

    await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=100_000.0,
            timestamp_ms=base_ts,
        )
    )
    result = await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=100_000.0,
            timestamp_ms=base_ts + 2_000,
        )
    )

    assert result.has_signals is True

    state = analyzer.get_symbol_state("BTCUSDT")
    assert state["exists"] is True
    assert state["activity_records_size"] == 1
    assert state["total_events_seen"] == 2
    assert state["cluster_first_seen_ts_ms"] == base_ts + 2_000
    assert state["cluster_last_seen_ts_ms"] == base_ts + 2_000


async def test_cluster_cleanup_reset_and_invalid_symbol_state_api(
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    whale_activity_payload_factory,
) -> None:
    config = replace(whale_cluster_analyzer_config_fast, stats_ttl_sec=0.01)
    analyzer = _build_cluster_analyzer(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(symbol="BTCUSDT", total_notional=200_000.0)
    )
    await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(symbol="ETHUSDT", total_notional=200_000.0)
    )

    assert analyzer.get_healthcheck()["tracked_symbols"] == 2

    assert analyzer.get_symbol_state("")["error"] == "invalid_symbol"
    assert analyzer.get_symbol_state("   ")["error"] == "invalid_symbol"

    await analyzer.reset_symbol("btcusdt")
    assert analyzer.get_symbol_state("BTCUSDT")["exists"] is False
    assert analyzer.get_symbol_state("ETHUSDT")["exists"] is True

    await asyncio.sleep(0.02)
    await analyzer.cleanup()
    assert analyzer.get_all_states() == {}

    await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(symbol="SOLUSDT", total_notional=200_000.0)
    )
    assert analyzer.get_symbol_state("SOLUSDT")["exists"] is True

    await analyzer.reset_all()
    assert analyzer.get_all_states() == {}


async def test_cluster_event_handlers_do_not_raise_on_hostile_payloads(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
) -> None:
    hostile_payloads: list[Any] = [
        None,
        [],
        "bad",
        {"data": None},
        {"symbol": "", "side": "buy"},
        {"symbol": "BTCUSDT", "side": "bad", "total_notional": 100_000},
    ]

    for payload in hostile_payloads:
        await whale_cluster_analyzer.handle_whale_activity_event(
            Event(topic=WHALE_ACTIVITY_TOPIC, payload=payload, source="test")
        )
        await whale_cluster_analyzer.handle_whale_pressure_event(
            Event(topic=WHALE_PRESSURE_TOPIC, payload=payload, source="test")
        )
        await whale_cluster_analyzer.handle_whale_liquidation_context_event(
            Event(
                topic=WHALE_LIQUIDATION_CONTEXT_TOPIC,
                payload=payload,
                source="test",
            )
        )

    assert whale_cluster_analyzer.get_all_states() == {}


async def test_cluster_concurrent_same_symbol_events_do_not_corrupt_state(
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    whale_activity_payload_factory,
) -> None:
    config = replace(
        whale_cluster_analyzer_config_fast,
        cluster_emit_cooldown_sec=60.0,
        cluster_update_cooldown_sec=60.0,
        cluster_exhaustion_cooldown_sec=60.0,
    )
    analyzer = _build_cluster_analyzer(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    payloads = [
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            trade_count=2,
            total_notional=100_000.0 + index,
            avg_notional=50_000.0,
            max_notional=60_000.0,
            timestamp_ms=1_700_000_000_000 + index,
        )
        for index in range(40)
    ]

    results = await asyncio.gather(
        *(analyzer.process_whale_activity_payload(payload) for payload in payloads)
    )

    state = analyzer.get_symbol_state("BTCUSDT")
    assert state["exists"] is True
    assert state["total_events_seen"] == 40
    assert state["activity_records_size"] == 40

    # Через cooldown не має бути лавини однакових сигналів.
    assert state["total_clusters_emitted"] <= 1
    assert state["total_cluster_updates_emitted"] <= 1

    assert sum(result.whale_cluster_signal is not None for result in results) <= 1
    assert sum(result.whale_cluster_update_signal is not None for result in results) <= 1


# =============================================================================
# WhaleClusterAnalyzer EventBus emissions
# =============================================================================


async def test_cluster_direct_processing_publishes_cluster_event(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    event_collector,
    whale_activity_payload_factory,
) -> None:
    result = await whale_cluster_analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=200_000.0,
        )
    )

    assert result.whale_cluster_signal is not None

    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1)

    payload = event_collector.payloads_by_topic(WHALE_CLUSTER_TOPIC)[0]
    assert payload["symbol"] == "BTCUSDT"
    assert payload["cluster_side"] == "buy"
    assert 0.0 <= payload["cluster_score"] <= 1.0
    assert 0.0 <= payload["continuation_probability"] <= 1.0
    assert 0.0 <= payload["exhaustion_probability"] <= 1.0


async def test_cluster_registered_eventbus_handler_preserves_correlation_id(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    event_bus: EventBus,
    event_collector,
    whale_activity_payload_factory,
) -> None:
    await whale_cluster_analyzer.register()

    accepted = await event_bus.emit(
        WHALE_ACTIVITY_TOPIC,
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=200_000.0,
        ),
        source="tests.whale_tracker",
        correlation_id="corr-cluster-1",
    )

    assert accepted is True

    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1)

    emitted = event_collector.by_topic(WHALE_CLUSTER_TOPIC)[0]
    assert emitted.correlation_id == "corr-cluster-1"
    assert emitted.payload["symbol"] == "BTCUSDT"


# =============================================================================
# Tracker + Cluster combined integration
# =============================================================================


async def test_tracker_output_can_feed_cluster_analyzer_directly(
    whale_tracker: WhaleTracker,
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    large_trade_payload_factory,
    liquidation_payload_factory,
) -> None:
    await _feed_large_trades(
        whale_tracker,
        large_trade_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=80_000.0,
    )

    tracker_state = whale_tracker.get_symbol_state("BTCUSDT")
    assert tracker_state["whale_activity_signals_emitted"] == 1
    assert tracker_state["whale_pressure_signals_emitted"] == 1

    tracker_result = await whale_tracker.process_liquidation_payload(
        liquidation_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=90_000.0,
        )
    )

    assert tracker_result is not None

    # Directly feed the latest tracker outputs to cluster analyzer.
    whale_activity_payload = whale_tracker.get_all_states()
    assert "BTCUSDT" in whale_activity_payload

    # Не дістаємо приватний internal buffer тут: використовуємо фабрики,
    # бо інтеграція прямих моделей перевіряється окремо через EventBus.
    cluster_result = await whale_cluster_analyzer.process_whale_activity_payload(
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "trade_count": 2,
            "total_notional": 160_000.0,
            "avg_notional": 80_000.0,
            "max_notional": 80_000.0,
            "window_sec": 30,
            "timestamp_ms": int(time.time() * 1000),
        }
    )

    assert cluster_result.has_signals is True
    assert whale_cluster_analyzer.get_symbol_state("BTCUSDT")["exists"] is True


async def test_tracker_to_cluster_full_eventbus_chain_from_large_trades(
    whale_tracker: WhaleTracker,
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    event_bus: EventBus,
    event_collector,
    large_trade_payload_factory,
) -> None:
    await whale_tracker.register()
    await whale_cluster_analyzer.register()

    for index in range(2):
        accepted = await event_bus.emit(
            LARGE_TRADE_TOPIC,
            large_trade_payload_factory(
                symbol="BTCUSDT",
                side="buy",
                notional=90_000.0,
                trade_id=f"chain-{index}",
            ),
            source="tests.large_trade_detector",
            correlation_id="corr-chain-1",
        )
        assert accepted is True

    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1)
    await event_collector.wait_for_topic(WHALE_PRESSURE_TOPIC, count=1)
    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1)

    activity_event = event_collector.by_topic(WHALE_ACTIVITY_TOPIC)[0]
    cluster_event = event_collector.by_topic(WHALE_CLUSTER_TOPIC)[0]

    assert activity_event.correlation_id == "corr-chain-1"
    assert cluster_event.correlation_id == "corr-chain-1"
    assert cluster_event.payload["symbol"] == "BTCUSDT"


async def test_tracker_to_cluster_chain_handles_invalid_then_valid_events_without_poisoning_state(
    whale_tracker: WhaleTracker,
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    event_bus: EventBus,
    event_collector,
    large_trade_payload_factory,
) -> None:
    await whale_tracker.register()
    await whale_cluster_analyzer.register()

    invalid_events = [
        {"symbol": "", "side": "buy", "notional": 100_000},
        {"symbol": "BTCUSDT", "side": "bad", "notional": 100_000},
        {"data": None},
        [],
    ]

    for payload in invalid_events:
        await event_bus.emit(
            LARGE_TRADE_TOPIC,
            payload,
            source="tests.bad_source",
            correlation_id="corr-invalid",
        )

    await asyncio.sleep(0.05)

    assert whale_tracker.get_all_states() == {}
    assert whale_cluster_analyzer.get_all_states() == {}

    for index in range(2):
        await event_bus.emit(
            LARGE_TRADE_TOPIC,
            large_trade_payload_factory(
                symbol="BTCUSDT",
                side="buy",
                notional=90_000.0,
                trade_id=f"valid-after-invalid-{index}",
            ),
            source="tests.large_trade_detector",
            correlation_id="corr-valid-after-invalid",
        )

    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1)

    assert whale_tracker.get_symbol_state("BTCUSDT")["exists"] is True
    assert whale_cluster_analyzer.get_symbol_state("BTCUSDT")["exists"] is True

    cluster_event = event_collector.by_topic(WHALE_CLUSTER_TOPIC)[0]
    assert cluster_event.correlation_id == "corr-valid-after-invalid"