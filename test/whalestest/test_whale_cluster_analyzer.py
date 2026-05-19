from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any

import pytest

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from analytics.whales.config import WhaleClusterAnalyzerConfig
from analytics.whales.whale_cluster_analyzer import WhaleClusterAnalyzer


pytestmark = pytest.mark.asyncio


# =============================================================================
# Topics
# =============================================================================

WHALE_ACTIVITY_TOPIC = "analytics.whales.whale_activity"
WHALE_PRESSURE_TOPIC = "analytics.whales.whale_pressure"
WHALE_LIQUIDATION_CONTEXT_TOPIC = "analytics.whales.whale_liquidation_context"

WHALE_CLUSTER_TOPIC = "analytics.whales.whale_cluster"
WHALE_CLUSTER_UPDATE_TOPIC = "analytics.whales.whale_cluster_update"
WHALE_CLUSTER_EXHAUSTION_TOPIC = "analytics.whales.whale_cluster_exhaustion"


# =============================================================================
# Local helpers
# =============================================================================

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


def _default_scope_state(
    analyzer: WhaleClusterAnalyzer,
    symbol: str = "BTCUSDT",
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    timeframe: str = "realtime",
) -> dict[str, Any]:
    return analyzer.get_symbol_state(
        symbol,
        exchange=exchange,
        market_type=market_type,
        timeframe=timeframe,
    )


def _assert_cluster_state_empty(analyzer: WhaleClusterAnalyzer) -> None:
    assert analyzer.get_all_states() == {}


def _first_non_none_signal(result: Any) -> Any | None:
    return (
        result.whale_cluster_signal
        or result.whale_cluster_update_signal
        or result.whale_cluster_exhaustion_signal
    )


async def _feed_activity(
    analyzer: WhaleClusterAnalyzer,
    whale_activity_payload_factory,
    *,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    timeframe: str = "realtime",
    side: str = "buy",
    count: int = 1,
    total_notional: float = 200_000.0,
    start_ts_ms: int | None = None,
) -> list[Any]:
    base_ts = start_ts_ms if start_ts_ms is not None else int(time.time() * 1000)
    results: list[Any] = []

    for index in range(count):
        result = await analyzer.process_whale_activity_payload(
            whale_activity_payload_factory(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                side=side,
                trade_count=3,
                total_notional=total_notional + index,
                avg_notional=(total_notional + index) / 3,
                max_notional=total_notional + index,
                timestamp_ms=base_ts + index,
            )
        )
        results.append(result)

    return results


# =============================================================================
# Lifecycle / core behavior
# =============================================================================

async def test_cluster_register_is_idempotent_and_subscribes_expected_topics(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
) -> None:
    await whale_cluster_analyzer.register()
    await whale_cluster_analyzer.register()
    await whale_cluster_analyzer.register()

    assert whale_cluster_analyzer.is_registered is True
    assert whale_cluster_analyzer.is_started is False

    patterns = {
        subscription.pattern
        for subscription in whale_cluster_analyzer.subscriptions
    }

    assert patterns == {
        WHALE_ACTIVITY_TOPIC,
        WHALE_PRESSURE_TOPIC,
        WHALE_LIQUIDATION_CONTEXT_TOPIC,
    }
    assert len(whale_cluster_analyzer.subscriptions) == 3


async def test_cluster_start_adds_one_cleanup_job_and_stop_removes_runtime_state(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
) -> None:
    await whale_cluster_analyzer.start()
    await whale_cluster_analyzer.start()
    await whale_cluster_analyzer.start()

    assert whale_cluster_analyzer.is_started is True
    assert whale_cluster_analyzer.is_registered is True
    assert len(whale_cluster_analyzer.subscriptions) == 3
    assert len(whale_cluster_analyzer.scheduler_job_ids) == 1

    await whale_cluster_analyzer.stop()
    await whale_cluster_analyzer.stop()

    assert whale_cluster_analyzer.is_started is False
    assert whale_cluster_analyzer.is_registered is False
    assert len(whale_cluster_analyzer.subscriptions) == 0
    assert len(whale_cluster_analyzer.scheduler_job_ids) == 0


async def test_disabled_cluster_analyzer_does_not_register_start_process_or_emit(
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    whale_activity_payload_factory,
    whale_pressure_payload_factory,
    whale_liquidation_context_payload_factory,
) -> None:
    config = replace(whale_cluster_analyzer_config_fast, enabled=False)
    analyzer = _build_cluster_analyzer(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    await analyzer.register()
    await analyzer.start()

    assert analyzer.is_registered is False
    assert analyzer.is_started is False
    assert len(analyzer.subscriptions) == 0
    assert len(analyzer.scheduler_job_ids) == 0

    activity_result = await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(total_notional=300_000.0)
    )
    pressure_result = await analyzer.process_whale_pressure_payload(
        whale_pressure_payload_factory(total_notional=300_000.0)
    )
    context_result = await analyzer.process_whale_liquidation_context_payload(
        whale_liquidation_context_payload_factory(
            whale_total_notional=300_000.0,
            liquidation_total_notional=200_000.0,
        )
    )

    assert activity_result.has_signals is False
    assert pressure_result.has_signals is False
    assert context_result.has_signals is False
    _assert_cluster_state_empty(analyzer)

    await asyncio.sleep(0.05)

    assert event_collector.by_topic(WHALE_CLUSTER_TOPIC) == []
    assert event_collector.by_topic(WHALE_CLUSTER_UPDATE_TOPIC) == []
    assert event_collector.by_topic(WHALE_CLUSTER_EXHAUSTION_TOPIC) == []


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
        {"symbol": "BTCUSDT", "side": "bad", "total_notional": 200_000.0},
        {"symbol": "BTCUSDT", "side": "buy", "total_notional": None},
        {"symbol": "BTCUSDT", "side": "buy", "total_notional": "bad"},
        {"symbol": "BTCUSDT", "side": "buy", "total_notional": float("nan")},
        {"symbol": "BTCUSDT", "side": "buy", "total_notional": -1},
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "trade_count": 0,
            "total_notional": 200_000.0,
        },
    ],
)
async def test_cluster_rejects_malformed_activity_payloads_without_state(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    payload: Any,
) -> None:
    result = await whale_cluster_analyzer.process_whale_activity_payload(payload)

    assert result.has_signals is False
    _assert_cluster_state_empty(whale_cluster_analyzer)


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
        {"symbol": "BTCUSDT", "dominant_side": "bad", "total_notional": 200_000.0},
        {"symbol": "BTCUSDT", "dominant_side": "buy", "total_notional": None},
        {"symbol": "BTCUSDT", "dominant_side": "buy", "total_notional": "bad"},
        {"symbol": "BTCUSDT", "dominant_side": "buy", "total_notional": float("nan")},
        {
            "symbol": "BTCUSDT",
            "dominant_side": "buy",
            "buy_notional": 0,
            "sell_notional": 0,
            "total_notional": 0,
        },
    ],
)
async def test_cluster_rejects_malformed_pressure_payloads_without_state(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    payload: Any,
) -> None:
    result = await whale_cluster_analyzer.process_whale_pressure_payload(payload)

    assert result.has_signals is False
    _assert_cluster_state_empty(whale_cluster_analyzer)


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
        {"symbol": "BTCUSDT", "whale_side": "bad", "whale_total_notional": 100_000.0},
        {"symbol": "BTCUSDT", "whale_side": "buy", "whale_total_notional": None},
        {"symbol": "BTCUSDT", "whale_side": "buy", "whale_total_notional": "bad"},
        {"symbol": "BTCUSDT", "whale_side": "buy", "whale_total_notional": float("nan")},
        {
            "symbol": "BTCUSDT",
            "whale_side": "buy",
            "whale_total_notional": 100_000.0,
            "whale_trade_count": 0,
            "liquidation_side": "sell",
            "liquidation_total_notional": 100_000.0,
            "liquidation_count": 1,
            "context_strength": 0.5,
        },
    ],
)
async def test_cluster_rejects_malformed_liquidation_context_payloads_without_state(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    payload: Any,
) -> None:
    result = await whale_cluster_analyzer.process_whale_liquidation_context_payload(
        payload
    )

    assert result.has_signals is False
    _assert_cluster_state_empty(whale_cluster_analyzer)


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

    _assert_cluster_state_empty(whale_cluster_analyzer)


# =============================================================================
# Cluster scoring / emissions
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

    state = _default_scope_state(analyzer, "BTCUSDT")

    assert state["exists"] is True
    assert state["activity_records_size"] == 2
    assert state["total_events_seen"] == 2
    assert state["total_clusters_emitted"] == 0
    assert state["total_cluster_updates_emitted"] == 0


async def test_activity_above_threshold_emits_cluster_and_update_signal(
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
    assert result.whale_cluster_update_signal is not None

    signal = result.whale_cluster_signal

    assert signal.symbol == "BTCUSDT"
    assert signal.exchange == "binance"
    assert signal.market_type == "usdm_futures"
    assert signal.timeframe == "realtime"
    assert signal.cluster_side == "buy"
    assert 0.0 <= signal.cluster_score <= 1.0
    assert 0.0 <= signal.persistence_score <= 1.0
    assert 0.0 <= signal.directional_bias <= 1.0
    assert 0.0 <= signal.continuation_probability <= 1.0
    assert 0.0 <= signal.exhaustion_probability <= 1.0
    assert signal.activity_signal_count == 1
    assert signal.pressure_signal_count == 0
    assert signal.liquidation_context_count == 0
    assert signal.total_activity_notional >= 200_000.0

    state = _default_scope_state(whale_cluster_analyzer, "BTCUSDT")

    assert state["activity_records_size"] == 1
    assert state["pressure_records_size"] == 0
    assert state["liquidation_context_records_size"] == 0
    assert state["total_events_seen"] == 1
    assert state["total_clusters_emitted"] == 1
    assert state["total_cluster_updates_emitted"] == 1


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
            total_notional=240_000.0,
            imbalance_ratio=0.92,
            net_flow_notional=200_000.0,
        )
    )
    context_result = (
        await whale_cluster_analyzer.process_whale_liquidation_context_payload(
            whale_liquidation_context_payload_factory(
                symbol="BTCUSDT",
                whale_side="buy",
                whale_total_notional=180_000.0,
                whale_trade_count=2,
                liquidation_side="sell",
                liquidation_total_notional=100_000.0,
                liquidation_count=1,
                context_strength=0.95,
            )
        )
    )

    assert pressure_result.has_signals is True
    assert context_result.has_signals is True

    latest_signal = _first_non_none_signal(context_result)

    assert latest_signal is not None
    assert latest_signal.symbol == "BTCUSDT"
    assert latest_signal.cluster_side == "buy"

    state = _default_scope_state(whale_cluster_analyzer, "BTCUSDT")

    assert state["activity_records_size"] == 1
    assert state["pressure_records_size"] == 1
    assert state["liquidation_context_records_size"] == 1
    assert state["total_events_seen"] == 3
    assert state["total_clusters_emitted"] >= 1
    assert state["total_cluster_updates_emitted"] >= 1


async def test_sell_side_activity_emits_sell_cluster(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    whale_activity_payload_factory,
    whale_pressure_payload_factory,
) -> None:
    await whale_cluster_analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            total_notional=220_000.0,
            trade_count=3,
        )
    )
    result = await whale_cluster_analyzer.process_whale_pressure_payload(
        whale_pressure_payload_factory(
            symbol="BTCUSDT",
            dominant_side="sell",
            buy_trade_count=0,
            sell_trade_count=3,
            buy_notional=10_000.0,
            sell_notional=240_000.0,
            total_notional=250_000.0,
            imbalance_ratio=0.96,
            net_flow_notional=-230_000.0,
        )
    )

    latest_signal = _first_non_none_signal(result)

    assert latest_signal is not None
    assert latest_signal.cluster_side == "sell"

    state = _default_scope_state(whale_cluster_analyzer, "BTCUSDT")
    assert state["exists"] is True
    assert state["activity_records_size"] == 1
    assert state["pressure_records_size"] == 1


async def test_exhaustion_signal_can_emit_under_weak_direction_and_liquidation_context(
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    whale_activity_payload_factory,
    whale_pressure_payload_factory,
    whale_liquidation_context_payload_factory,
) -> None:
    config = replace(
        whale_cluster_analyzer_config_fast,
        min_exhaustion_probability_to_emit=0.20,
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
            trade_count=2,
        )
    )
    await analyzer.process_whale_pressure_payload(
        whale_pressure_payload_factory(
            symbol="BTCUSDT",
            dominant_side="sell",
            buy_notional=20_000.0,
            sell_notional=220_000.0,
            total_notional=240_000.0,
            imbalance_ratio=0.92,
            net_flow_notional=-200_000.0,
        )
    )
    result = await analyzer.process_whale_liquidation_context_payload(
        whale_liquidation_context_payload_factory(
            symbol="BTCUSDT",
            whale_side="sell",
            whale_total_notional=200_000.0,
            whale_trade_count=2,
            liquidation_side="buy",
            liquidation_total_notional=160_000.0,
            liquidation_count=2,
            context_strength=1.0,
        )
    )

    assert result.whale_cluster_exhaustion_signal is not None

    exhaustion = result.whale_cluster_exhaustion_signal

    assert exhaustion.symbol == "BTCUSDT"
    assert exhaustion.cluster_side in {"buy", "sell"}
    assert 0.0 <= exhaustion.cluster_score <= 1.0
    assert 0.0 <= exhaustion.exhaustion_probability <= 1.0
    assert 0.0 <= exhaustion.reversal_risk <= 1.0

    state = _default_scope_state(analyzer, "BTCUSDT")
    assert state["total_cluster_exhaustions_emitted"] >= 1


async def test_cluster_emit_cooldown_blocks_duplicate_cluster_but_state_keeps_growing(
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

    state = _default_scope_state(analyzer, "BTCUSDT")

    assert state["activity_records_size"] == 2
    assert state["total_events_seen"] == 2
    assert state["total_clusters_emitted"] == 1
    assert state["total_cluster_updates_emitted"] == 1


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

    state = _default_scope_state(analyzer, "BTCUSDT")

    assert state["exists"] is True
    assert state["activity_records_size"] == 1
    assert state["total_events_seen"] == 2
    assert state["cluster_first_seen_ts_ms"] == base_ts + 2_000
    assert state["cluster_last_seen_ts_ms"] == base_ts + 2_000


# =============================================================================
# EventBus emissions
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

    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1, timeout=1.0)

    payload = event_collector.payloads_by_topic(WHALE_CLUSTER_TOPIC)[0]

    assert payload["symbol"] == "BTCUSDT"
    assert payload["exchange"] == "binance"
    assert payload["market_type"] == "usdm_futures"
    assert payload["timeframe"] == "realtime"
    assert payload["cluster_side"] == "buy"
    assert 0.0 <= payload["cluster_score"] <= 1.0
    assert 0.0 <= payload["continuation_probability"] <= 1.0
    assert 0.0 <= payload["exhaustion_probability"] <= 1.0
    assert "scope" in payload


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

    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1, timeout=1.0)

    emitted = event_collector.by_topic(WHALE_CLUSTER_TOPIC)[0]

    assert emitted.correlation_id == "corr-cluster-1"
    assert emitted.payload["symbol"] == "BTCUSDT"
    assert emitted.payload["cluster_side"] == "buy"


async def test_cluster_eventbus_pressure_and_context_update_existing_state(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    event_bus: EventBus,
    event_collector,
    whale_activity_payload_factory,
    whale_pressure_payload_factory,
    whale_liquidation_context_payload_factory,
) -> None:
    await whale_cluster_analyzer.start()

    accepted_activity = await event_bus.emit(
        WHALE_ACTIVITY_TOPIC,
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=200_000.0,
        ),
        source="tests.whale_tracker",
        correlation_id="corr-cluster-context",
    )
    accepted_pressure = await event_bus.emit(
        WHALE_PRESSURE_TOPIC,
        whale_pressure_payload_factory(
            symbol="BTCUSDT",
            dominant_side="buy",
            buy_notional=220_000.0,
            sell_notional=20_000.0,
            total_notional=240_000.0,
            imbalance_ratio=0.92,
        ),
        source="tests.whale_tracker",
        correlation_id="corr-cluster-context",
    )
    accepted_context = await event_bus.emit(
        WHALE_LIQUIDATION_CONTEXT_TOPIC,
        whale_liquidation_context_payload_factory(
            symbol="BTCUSDT",
            whale_side="buy",
            liquidation_side="sell",
            context_strength=0.95,
        ),
        source="tests.whale_tracker",
        correlation_id="corr-cluster-context",
    )

    assert accepted_activity is True
    assert accepted_pressure is True
    assert accepted_context is True

    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1, timeout=1.0)
    await event_collector.wait_for_topic(
        WHALE_CLUSTER_UPDATE_TOPIC,
        count=1,
        timeout=1.0,
    )

    state = _default_scope_state(whale_cluster_analyzer, "BTCUSDT")

    assert state["exists"] is True
    assert state["activity_records_size"] == 1
    assert state["pressure_records_size"] == 1
    assert state["liquidation_context_records_size"] == 1
    assert state["total_events_seen"] == 3


async def test_emit_on_bus_false_returns_signals_without_publishing_events(
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    whale_activity_payload_factory,
) -> None:
    config = replace(
        whale_cluster_analyzer_config_fast,
        emit_on_bus=False,
        log_signals=False,
    )
    analyzer = _build_cluster_analyzer(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    result = await analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=200_000.0,
        )
    )

    assert result.has_signals is True
    assert result.whale_cluster_signal is not None

    await asyncio.sleep(0.05)

    assert event_collector.by_topic(WHALE_CLUSTER_TOPIC) == []
    assert event_collector.by_topic(WHALE_CLUSTER_UPDATE_TOPIC) == []
    assert event_collector.by_topic(WHALE_CLUSTER_EXHAUSTION_TOPIC) == []


# =============================================================================
# Scoped isolation / concurrency
# =============================================================================

async def test_concurrent_same_scope_events_do_not_corrupt_state_or_duplicate_signals(
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

    state = _default_scope_state(analyzer, "BTCUSDT")

    assert state["exists"] is True
    assert state["total_events_seen"] == 40
    assert state["activity_records_size"] == 40

    assert state["total_clusters_emitted"] <= 1
    assert state["total_cluster_updates_emitted"] <= 1

    assert sum(result.whale_cluster_signal is not None for result in results) <= 1
    assert sum(result.whale_cluster_update_signal is not None for result in results) <= 1


async def test_concurrent_different_scopes_do_not_cross_contaminate_state(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    whale_activity_payload_factory,
    whale_pressure_payload_factory,
) -> None:
    payloads = [
        whale_activity_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="realtime",
            side="buy",
            total_notional=200_000.0,
        ),
        whale_pressure_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="realtime",
            dominant_side="buy",
            buy_notional=220_000.0,
            sell_notional=20_000.0,
            total_notional=240_000.0,
        ),
        whale_activity_payload_factory(
            exchange="okx",
            market_type="swap",
            symbol="BTCUSDT",
            timeframe="realtime",
            side="sell",
            total_notional=210_000.0,
        ),
        whale_pressure_payload_factory(
            exchange="okx",
            market_type="swap",
            symbol="BTCUSDT",
            timeframe="realtime",
            dominant_side="sell",
            buy_notional=10_000.0,
            sell_notional=220_000.0,
            total_notional=230_000.0,
        ),
        whale_activity_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="ETHUSDT",
            timeframe="realtime",
            side="buy",
            total_notional=190_000.0,
        ),
        whale_activity_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="1m",
            side="buy",
            total_notional=180_000.0,
        ),
    ]

    async def _process(payload: dict[str, Any]) -> Any:
        if "dominant_side" in payload:
            return await whale_cluster_analyzer.process_whale_pressure_payload(payload)
        return await whale_cluster_analyzer.process_whale_activity_payload(payload)

    await asyncio.gather(*(_process(payload) for payload in payloads))

    all_states = whale_cluster_analyzer.get_all_states()

    assert len(all_states) == 4

    binance_btc = _default_scope_state(
        whale_cluster_analyzer,
        "BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    okx_btc = _default_scope_state(
        whale_cluster_analyzer,
        "BTCUSDT",
        exchange="okx",
        market_type="swap",
        timeframe="realtime",
    )
    binance_eth = _default_scope_state(
        whale_cluster_analyzer,
        "ETHUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    binance_btc_1m = _default_scope_state(
        whale_cluster_analyzer,
        "BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="1m",
    )

    assert binance_btc["exists"] is True
    assert okx_btc["exists"] is True
    assert binance_eth["exists"] is True
    assert binance_btc_1m["exists"] is True

    assert binance_btc["activity_records_size"] == 1
    assert binance_btc["pressure_records_size"] == 1
    assert okx_btc["activity_records_size"] == 1
    assert okx_btc["pressure_records_size"] == 1
    assert binance_eth["activity_records_size"] == 1
    assert binance_eth["pressure_records_size"] == 0
    assert binance_btc_1m["activity_records_size"] == 1
    assert binance_btc_1m["pressure_records_size"] == 0


# =============================================================================
# Cleanup / reset / health
# =============================================================================

async def test_cluster_cleanup_removes_stale_scoped_states(
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

    assert len(analyzer.get_all_states()) == 2
    assert analyzer.get_healthcheck()["tracked_scopes"] == 2

    await asyncio.sleep(0.02)
    await analyzer.cleanup()

    assert analyzer.get_all_states() == {}


async def test_cluster_reset_symbol_is_scoped_when_scope_is_provided(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    whale_activity_payload_factory,
) -> None:
    await whale_cluster_analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="realtime",
            side="buy",
            total_notional=200_000.0,
        )
    )
    await whale_cluster_analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            exchange="okx",
            market_type="swap",
            symbol="BTCUSDT",
            timeframe="realtime",
            side="sell",
            total_notional=210_000.0,
        )
    )
    await whale_cluster_analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="ETHUSDT",
            timeframe="realtime",
            side="buy",
            total_notional=190_000.0,
        )
    )

    assert len(whale_cluster_analyzer.get_all_states()) == 3

    await whale_cluster_analyzer.reset_symbol(
        "BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert (
        _default_scope_state(
            whale_cluster_analyzer,
            "BTCUSDT",
            exchange="binance",
            market_type="usdm_futures",
            timeframe="realtime",
        )["exists"]
        is False
    )
    assert (
        _default_scope_state(
            whale_cluster_analyzer,
            "BTCUSDT",
            exchange="okx",
            market_type="swap",
            timeframe="realtime",
        )["exists"]
        is True
    )
    assert _default_scope_state(whale_cluster_analyzer, "ETHUSDT")["exists"] is True

    await whale_cluster_analyzer.reset_symbol("BTCUSDT")

    assert whale_cluster_analyzer.get_symbol_state("BTCUSDT")["exists"] is False
    assert _default_scope_state(whale_cluster_analyzer, "ETHUSDT")["exists"] is True

    await whale_cluster_analyzer.reset_all()

    assert whale_cluster_analyzer.get_all_states() == {}


async def test_cluster_invalid_symbol_state_api_is_safe(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
) -> None:
    assert whale_cluster_analyzer.get_symbol_state("")["error"] == "invalid_symbol"
    assert whale_cluster_analyzer.get_symbol_state("   ")["error"] == "invalid_symbol"


async def test_cluster_healthcheck_reports_scoped_runtime_shape(
    whale_cluster_analyzer: WhaleClusterAnalyzer,
    whale_activity_payload_factory,
) -> None:
    await whale_cluster_analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=200_000.0,
        )
    )

    health = whale_cluster_analyzer.get_healthcheck()

    assert health["component"] == "whale_cluster_analyzer"
    assert health["event_bus_available"] is True
    assert health["scheduler_available"] is True
    assert health["enabled"] is True
    assert health["tracked_scopes"] >= 1
    assert health["scope"] == "exchange:market_type:symbol:timeframe"

    # Очікування під оновлений per-key locking class.
    # Якщо впаде з KeyError — треба додати state_locks/locking у get_healthcheck().
    assert health["state_locks"] >= 1
    assert health["locking"] == "per_whale_key"

    assert WHALE_ACTIVITY_TOPIC in health["production_input_topics"]
    assert WHALE_PRESSURE_TOPIC in health["production_input_topics"]
    assert WHALE_LIQUIDATION_CONTEXT_TOPIC in health["production_input_topics"]
    assert health["whale_cluster_event_name"] == WHALE_CLUSTER_TOPIC
    assert health["whale_cluster_update_event_name"] == WHALE_CLUSTER_UPDATE_TOPIC
    assert health["whale_cluster_exhaustion_event_name"] == WHALE_CLUSTER_EXHAUSTION_TOPIC