from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from core.event_bus import EventBus
from core.scheduler import Scheduler

from analytics.whales.analyzer import WhaleAnalyzer
from analytics.whales.config import WhalesConfig


pytestmark = pytest.mark.asyncio


# =============================================================================
# Topics
# =============================================================================

TRADES_UPDATED_TOPIC = "market.trades.updated"
RAW_TRADE_TOPIC = "market.trade"

LIQUIDATIONS_UPDATED_TOPIC = "market.liquidations.updated"
RAW_LIQUIDATION_TOPIC = "market.liquidation"

LARGE_TRADE_TOPIC = "analytics.whales.large_trade"
WHALE_ACTIVITY_TOPIC = "analytics.whales.whale_activity"
WHALE_PRESSURE_TOPIC = "analytics.whales.whale_pressure"
WHALE_LIQUIDATION_CONTEXT_TOPIC = "analytics.whales.whale_liquidation_context"

WHALE_CLUSTER_TOPIC = "analytics.whales.whale_cluster"
WHALE_CLUSTER_UPDATE_TOPIC = "analytics.whales.whale_cluster_update"
WHALE_CLUSTER_EXHAUSTION_TOPIC = "analytics.whales.whale_cluster_exhaustion"


# =============================================================================
# Local helpers
# =============================================================================

def _build_analyzer(
    *,
    config: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> WhaleAnalyzer:
    return WhaleAnalyzer(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )


def _assert_children_registered(analyzer: WhaleAnalyzer) -> None:
    assert analyzer.large_trade_detector.is_registered is True
    assert analyzer.whale_tracker.is_registered is True
    assert analyzer.whale_cluster_analyzer.is_registered is True


def _assert_children_not_registered(analyzer: WhaleAnalyzer) -> None:
    assert analyzer.large_trade_detector.is_registered is False
    assert analyzer.whale_tracker.is_registered is False
    assert analyzer.whale_cluster_analyzer.is_registered is False


def _assert_children_started(analyzer: WhaleAnalyzer) -> None:
    assert analyzer.large_trade_detector.is_started is True
    assert analyzer.whale_tracker.is_started is True
    assert analyzer.whale_cluster_analyzer.is_started is True


def _assert_children_not_started(analyzer: WhaleAnalyzer) -> None:
    assert analyzer.large_trade_detector.is_started is False
    assert analyzer.whale_tracker.is_started is False
    assert analyzer.whale_cluster_analyzer.is_started is False


def _assert_no_child_subscriptions(analyzer: WhaleAnalyzer) -> None:
    assert len(analyzer.large_trade_detector.subscriptions) == 0
    assert len(analyzer.whale_tracker.subscriptions) == 0
    assert len(analyzer.whale_cluster_analyzer.subscriptions) == 0


def _assert_no_child_scheduler_jobs(analyzer: WhaleAnalyzer) -> None:
    assert len(analyzer.large_trade_detector.scheduler_job_ids) == 0
    assert len(analyzer.whale_tracker.scheduler_job_ids) == 0
    assert len(analyzer.whale_cluster_analyzer.scheduler_job_ids) == 0


def _assert_auto_started_child_runtime(analyzer: WhaleAnalyzer) -> None:
    assert len(analyzer.large_trade_detector.subscriptions) == 1
    assert len(analyzer.whale_tracker.subscriptions) == 2
    assert len(analyzer.whale_cluster_analyzer.subscriptions) == 3

    assert len(analyzer.large_trade_detector.scheduler_job_ids) == 1
    assert len(analyzer.whale_tracker.scheduler_job_ids) == 1
    assert len(analyzer.whale_cluster_analyzer.scheduler_job_ids) == 1


def _assert_registered_only_child_runtime(analyzer: WhaleAnalyzer) -> None:
    assert len(analyzer.large_trade_detector.subscriptions) == 1
    assert len(analyzer.whale_tracker.subscriptions) == 2
    assert len(analyzer.whale_cluster_analyzer.subscriptions) == 3

    assert len(analyzer.large_trade_detector.scheduler_job_ids) == 0
    assert len(analyzer.whale_tracker.scheduler_job_ids) == 0
    assert len(analyzer.whale_cluster_analyzer.scheduler_job_ids) == 0


def _default_key_stats(analyzer: WhaleAnalyzer, symbol: str = "BTCUSDT") -> dict[str, Any]:
    return analyzer.get_symbol_stats(
        symbol,
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )


async def _emit_trade_batch(
    *,
    event_bus: EventBus,
    trades_updated_payload_factory,
    symbol: str = "BTCUSDT",
    side: str = "buy",
    count: int = 2,
    notional: float = 90_000.0,
    correlation_id: str = "corr-trade-batch",
    batch_id: str = "trade-batch",
) -> None:
    accepted = await event_bus.emit(
        TRADES_UPDATED_TOPIC,
        trades_updated_payload_factory(
            symbol=symbol,
            side=side,
            count=count,
            notional=notional,
            batch_id=batch_id,
        ),
        source="tests.trades_cache",
        correlation_id=correlation_id,
    )
    assert accepted is True


async def _force_direct_whale_context(
    *,
    analyzer: WhaleAnalyzer,
    large_trade_payload_factory,
    liquidations_updated_payload_factory,
    symbol: str = "BTCUSDT",
    whale_side: str = "buy",
    liquidation_side: str = "sell",
) -> Any:
    await analyzer.process_large_trade_signal(
        large_trade_payload_factory(
            symbol=symbol,
            side=whale_side,
            notional=90_000.0,
            price=100.0,
            quantity=900.0,
            trade_id=f"{symbol}-lt-1",
        )
    )
    await analyzer.process_large_trade_signal(
        large_trade_payload_factory(
            symbol=symbol,
            side=whale_side,
            notional=90_000.0,
            price=100.0,
            quantity=900.0,
            trade_id=f"{symbol}-lt-2",
        )
    )
    return await analyzer.process_liquidation(
        liquidations_updated_payload_factory(
            symbol=symbol,
            side=liquidation_side,
            notional=100_000.0,
            price=100.0,
            quantity=1_000.0,
        )
    )


# =============================================================================
# Construction / dependency injection / config validation
# =============================================================================

async def test_analyzer_constructs_all_children_with_same_core_dependencies(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    components = whale_analyzer.get_components()

    assert set(components) == {
        "large_trade_detector",
        "whale_tracker",
        "whale_cluster_analyzer",
    }

    assert components["large_trade_detector"] is whale_analyzer.large_trade_detector
    assert components["whale_tracker"] is whale_analyzer.whale_tracker
    assert components["whale_cluster_analyzer"] is whale_analyzer.whale_cluster_analyzer

    for component in components.values():
        assert component.event_bus is event_bus
        assert component.scheduler is scheduler

    assert whale_analyzer.event_bus is event_bus
    assert whale_analyzer.scheduler is scheduler


async def test_analyzer_validates_pipeline_topic_mismatches_on_construction(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    broken_config = WhalesConfig(
        enabled=True,
        auto_start_components=True,
        large_trade_detector=replace(
            whales_config_fast.large_trade_detector,
            output_event_name="analytics.whales.large_trade.WRONG",
        ),
        whale_tracker=whales_config_fast.whale_tracker,
        whale_cluster_analyzer=whales_config_fast.whale_cluster_analyzer,
    )

    with pytest.raises(ValueError, match="Pipeline topic mismatch"):
        _build_analyzer(
            config=broken_config,
            event_bus=event_bus,
            scheduler=scheduler,
        )


# =============================================================================
# Lifecycle / orchestration
# =============================================================================

async def test_analyzer_register_is_idempotent_and_registers_child_components(
    whale_analyzer: WhaleAnalyzer,
) -> None:
    await whale_analyzer.register()
    await whale_analyzer.register()
    await whale_analyzer.register()

    assert whale_analyzer.is_registered is True
    assert whale_analyzer.is_started is False

    _assert_children_registered(whale_analyzer)
    _assert_children_not_started(whale_analyzer)

    assert len(whale_analyzer.large_trade_detector.subscriptions) == 1
    assert len(whale_analyzer.whale_tracker.subscriptions) == 2
    assert len(whale_analyzer.whale_cluster_analyzer.subscriptions) == 3

    assert len(whale_analyzer.large_trade_detector.scheduler_job_ids) == 0
    assert len(whale_analyzer.whale_tracker.scheduler_job_ids) == 0
    assert len(whale_analyzer.whale_cluster_analyzer.scheduler_job_ids) == 0

    # Facade сам не має власних EventBus subscriptions.
    assert len(whale_analyzer.subscriptions) == 0


async def test_analyzer_start_is_idempotent_and_starts_children_once(
    whale_analyzer: WhaleAnalyzer,
) -> None:
    await whale_analyzer.start()
    await whale_analyzer.start()
    await whale_analyzer.start()

    assert whale_analyzer.is_started is True
    assert whale_analyzer.is_registered is True

    _assert_children_registered(whale_analyzer)
    _assert_children_started(whale_analyzer)
    _assert_auto_started_child_runtime(whale_analyzer)

    assert len(whale_analyzer.subscriptions) == 0


async def test_analyzer_stop_is_idempotent_and_stops_children(
    whale_analyzer: WhaleAnalyzer,
) -> None:
    await whale_analyzer.start()

    await whale_analyzer.stop()
    await whale_analyzer.stop()

    assert whale_analyzer.is_started is False
    assert whale_analyzer.is_registered is False

    _assert_children_not_started(whale_analyzer)
    _assert_children_not_registered(whale_analyzer)
    _assert_no_child_subscriptions(whale_analyzer)
    _assert_no_child_scheduler_jobs(whale_analyzer)


async def test_analyzer_auto_start_false_registers_children_but_does_not_start_scheduler_jobs(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> None:
    config = replace(whales_config_fast, auto_start_components=False)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    await analyzer.start()

    assert analyzer.is_started is True
    assert analyzer.is_registered is True

    _assert_children_registered(analyzer)
    _assert_children_not_started(analyzer)
    _assert_registered_only_child_runtime(analyzer)

    health = analyzer.get_healthcheck()

    assert health["children_registered"] is True
    assert health["children_started"] is False
    assert health["lifecycle_mode"] == "registered_components_only"


async def test_analyzer_auto_start_false_pipeline_still_processes_eventbus_events(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    trades_updated_payload_factory,
) -> None:
    config = replace(whales_config_fast, auto_start_components=False)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    await analyzer.start()

    _assert_children_registered(analyzer)
    _assert_children_not_started(analyzer)

    await _emit_trade_batch(
        event_bus=event_bus,
        trades_updated_payload_factory=trades_updated_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=90_000.0,
        correlation_id="corr-auto-start-false",
        batch_id="auto-start-false-batch",
    )

    await event_collector.wait_for_topic(LARGE_TRADE_TOPIC, count=2, timeout=1.0)
    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1, timeout=1.0)
    await event_collector.wait_for_topic(WHALE_PRESSURE_TOPIC, count=1, timeout=1.0)
    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1, timeout=1.0)

    assert len(analyzer.large_trade_detector.scheduler_job_ids) == 0
    assert len(analyzer.whale_tracker.scheduler_job_ids) == 0
    assert len(analyzer.whale_cluster_analyzer.scheduler_job_ids) == 0


async def test_disabled_analyzer_does_not_register_or_start_children(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    trades_updated_payload_factory,
) -> None:
    config = replace(whales_config_fast, enabled=False)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    await analyzer.register()
    await analyzer.start()

    assert analyzer.is_registered is False
    assert analyzer.is_started is False

    _assert_children_not_registered(analyzer)
    _assert_children_not_started(analyzer)
    _assert_no_child_subscriptions(analyzer)
    _assert_no_child_scheduler_jobs(analyzer)

    accepted = await event_bus.emit(
        TRADES_UPDATED_TOPIC,
        trades_updated_payload_factory(
            symbol="BTCUSDT",
            count=2,
            notional=90_000.0,
            batch_id="disabled-analyzer-batch",
        ),
        source="tests.trades_cache",
        correlation_id="corr-disabled-analyzer",
    )
    assert accepted is True

    await asyncio.sleep(0.05)

    assert event_collector.by_topic(LARGE_TRADE_TOPIC) == []
    assert event_collector.by_topic(WHALE_CLUSTER_TOPIC) == []


@pytest.mark.xfail(
    reason=(
        "Current WhaleAnalyzer direct API is backward-compatible and may still "
        "process even when top-level enabled=False unless allow_direct_raw_api "
        "is added to WhalesConfig and enforced."
    ),
    strict=False,
)
async def test_disabled_analyzer_direct_api_should_be_blocked_in_future(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    trades_updated_payload_factory,
) -> None:
    config = replace(whales_config_fast, enabled=False)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    with pytest.raises(RuntimeError):
        await analyzer.process_trade(
            trades_updated_payload_factory(
                symbol="BTCUSDT",
                count=1,
                notional=90_000.0,
            )
        )


async def test_analyzer_start_failure_rolls_back_started_children(
    whale_analyzer: WhaleAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_tracker_start() -> None:
        raise RuntimeError("synthetic tracker startup failure")

    monkeypatch.setattr(whale_analyzer.whale_tracker, "start", failing_tracker_start)

    with pytest.raises(RuntimeError, match="synthetic tracker startup failure"):
        await whale_analyzer.start()

    assert whale_analyzer.is_started is False
    assert whale_analyzer.is_registered is False

    _assert_children_not_started(whale_analyzer)
    _assert_children_not_registered(whale_analyzer)

    await whale_analyzer.stop()

    _assert_children_not_started(whale_analyzer)
    _assert_children_not_registered(whale_analyzer)


# =============================================================================
# Full EventBus production pipeline
# =============================================================================

async def test_full_eventbus_pipeline_trades_updated_batch_emits_large_activity_pressure_and_cluster(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    trades_updated_payload_factory,
) -> None:
    await whale_analyzer.start()

    await _emit_trade_batch(
        event_bus=event_bus,
        trades_updated_payload_factory=trades_updated_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=90_000.0,
        correlation_id="corr-full-pipeline-1",
        batch_id="full-pipeline-batch",
    )

    await event_collector.wait_for_topic(LARGE_TRADE_TOPIC, count=2, timeout=1.0)
    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1, timeout=1.0)
    await event_collector.wait_for_topic(WHALE_PRESSURE_TOPIC, count=1, timeout=1.0)
    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1, timeout=1.0)

    large_trade_events = event_collector.by_topic(LARGE_TRADE_TOPIC)
    activity_events = event_collector.by_topic(WHALE_ACTIVITY_TOPIC)
    pressure_events = event_collector.by_topic(WHALE_PRESSURE_TOPIC)
    cluster_events = event_collector.by_topic(WHALE_CLUSTER_TOPIC)

    assert len(large_trade_events) == 2
    assert [event.payload["trade_id"] for event in large_trade_events] == [
        "full-pipeline-batch-0",
        "full-pipeline-batch-1",
    ]

    assert activity_events[0].payload["symbol"] == "BTCUSDT"
    assert activity_events[0].payload["side"] == "buy"

    assert pressure_events[0].payload["symbol"] == "BTCUSDT"
    assert pressure_events[0].payload["dominant_side"] == "buy"

    assert cluster_events[0].payload["symbol"] == "BTCUSDT"
    assert cluster_events[0].payload["cluster_side"] == "buy"

    assert all(event.correlation_id == "corr-full-pipeline-1" for event in large_trade_events)
    assert activity_events[0].correlation_id == "corr-full-pipeline-1"
    assert pressure_events[0].correlation_id == "corr-full-pipeline-1"
    assert cluster_events[0].correlation_id == "corr-full-pipeline-1"


async def test_full_eventbus_pipeline_ignores_noise_before_valid_batch(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    trades_updated_payload_factory,
    raw_trade_payload_factory,
) -> None:
    await whale_analyzer.start()

    noisy_payloads: list[Any] = [
        None,
        [],
        "bad",
        {"data": None},
        {"trades": None},
        {"trades": []},
        {"trades": [None, [], "bad", {"symbol": ""}]},
        trades_updated_payload_factory(
            trades=[
                raw_trade_payload_factory(
                    symbol="BTCUSDT",
                    price=100.0,
                    quantity=0.01,
                    side="buy",
                    trade_id="tiny-noise",
                ),
                raw_trade_payload_factory(
                    symbol="BTCUSDT",
                    price=100.0,
                    quantity=1_000.0,
                    side="teleport",
                    trade_id="bad-side",
                ),
            ],
            batch_id="noise-batch",
        ),
    ]

    for index, payload in enumerate(noisy_payloads):
        await event_bus.emit(
            TRADES_UPDATED_TOPIC,
            payload,
            source="tests.noisy_trades_cache",
            correlation_id=f"corr-noise-{index}",
        )

    await asyncio.sleep(0.05)

    assert event_collector.by_topic(LARGE_TRADE_TOPIC) == []
    assert event_collector.by_topic(WHALE_CLUSTER_TOPIC) == []

    await _emit_trade_batch(
        event_bus=event_bus,
        trades_updated_payload_factory=trades_updated_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=90_000.0,
        correlation_id="corr-after-noise",
        batch_id="valid-after-noise",
    )

    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1, timeout=1.0)

    stats = _default_key_stats(whale_analyzer, "BTCUSDT")

    assert stats["large_trade_detector"]["exists"] is True
    assert stats["whale_tracker"]["exists"] is True
    assert stats["whale_cluster_analyzer"]["exists"] is True

    cluster_event = event_collector.by_topic(WHALE_CLUSTER_TOPIC)[0]
    assert cluster_event.correlation_id == "corr-after-noise"


async def test_full_eventbus_pipeline_liquidation_updated_flow_emits_context_and_feeds_cluster(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    trades_updated_payload_factory,
    liquidations_updated_payload_factory,
) -> None:
    await whale_analyzer.start()

    await _emit_trade_batch(
        event_bus=event_bus,
        trades_updated_payload_factory=trades_updated_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=90_000.0,
        correlation_id="corr-liquidation-flow",
        batch_id="liquidation-flow-trades",
    )

    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1, timeout=1.0)

    accepted = await event_bus.emit(
        LIQUIDATIONS_UPDATED_TOPIC,
        liquidations_updated_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=100_000.0,
            price=100.0,
            quantity=1_000.0,
            batch_id="liquidation-flow-liqs",
        ),
        source="tests.liquidation_cache",
        correlation_id="corr-liquidation-flow",
    )
    assert accepted is True

    await event_collector.wait_for_topic(
        WHALE_LIQUIDATION_CONTEXT_TOPIC,
        count=1,
        timeout=1.0,
    )
    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1, timeout=1.0)

    context_event = event_collector.by_topic(WHALE_LIQUIDATION_CONTEXT_TOPIC)[0]

    assert context_event.correlation_id == "corr-liquidation-flow"
    assert context_event.payload["symbol"] == "BTCUSDT"
    assert context_event.payload["whale_side"] == "buy"
    assert context_event.payload["liquidation_side"] == "sell"


async def test_pipeline_does_not_process_raw_market_trade_in_production_config(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    await whale_analyzer.start()

    accepted = await event_bus.emit(
        RAW_TRADE_TOPIC,
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=1_000.0,
            side="buy",
            trade_id="raw-trade-ignored",
        ),
        source="tests.market_stream",
        correlation_id="corr-raw-trade-ignored",
    )
    assert accepted is True

    await asyncio.sleep(0.05)

    assert event_collector.by_topic(LARGE_TRADE_TOPIC) == []
    assert event_collector.by_topic(WHALE_ACTIVITY_TOPIC) == []
    assert event_collector.by_topic(WHALE_CLUSTER_TOPIC) == []


async def test_pipeline_does_not_process_raw_liquidation_in_production_config(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_liquidation_payload_factory,
) -> None:
    await whale_analyzer.start()

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
        correlation_id="corr-raw-liquidation-ignored",
    )
    assert accepted is True

    await asyncio.sleep(0.05)

    assert event_collector.by_topic(WHALE_LIQUIDATION_CONTEXT_TOPIC) == []


async def test_legacy_analyzer_pipeline_can_process_raw_market_trade_when_enabled(
    whale_analyzer_legacy: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    await whale_analyzer_legacy.start()

    for index in range(2):
        accepted = await event_bus.emit(
            RAW_TRADE_TOPIC,
            raw_trade_payload_factory(
                symbol="BTCUSDT",
                price=100.0,
                quantity=900.0,
                side="buy",
                trade_id=f"legacy-raw-trade-{index}",
            ),
            source="tests.market_stream",
            correlation_id="corr-legacy-raw-trades",
        )
        assert accepted is True

    await event_collector.wait_for_topic(LARGE_TRADE_TOPIC, count=2, timeout=1.0)
    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1, timeout=1.0)
    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1, timeout=1.0)


async def test_legacy_analyzer_pipeline_can_process_raw_liquidation_when_enabled(
    whale_analyzer_legacy: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    trades_updated_payload_factory,
    raw_liquidation_payload_factory,
) -> None:
    await whale_analyzer_legacy.start()

    await _emit_trade_batch(
        event_bus=event_bus,
        trades_updated_payload_factory=trades_updated_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=90_000.0,
        correlation_id="corr-legacy-raw-liquidation",
        batch_id="legacy-liquidation-trades",
    )

    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1, timeout=1.0)

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
        correlation_id="corr-legacy-raw-liquidation",
    )
    assert accepted is True

    await event_collector.wait_for_topic(
        WHALE_LIQUIDATION_CONTEXT_TOPIC,
        count=1,
        timeout=1.0,
    )


# =============================================================================
# Direct API paths
# =============================================================================

async def test_direct_process_trade_batch_returns_large_trade_signals(
    whale_analyzer: WhaleAnalyzer,
    trades_updated_payload_factory,
) -> None:
    signals = await whale_analyzer.process_trade(
        trades_updated_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            count=2,
            notional=90_000.0,
            batch_id="direct-trade-batch",
        )
    )

    assert isinstance(signals, list)
    assert len(signals) == 2
    assert [signal.trade_id for signal in signals] == [
        "direct-trade-batch-0",
        "direct-trade-batch-1",
    ]


async def test_direct_large_trade_signal_can_feed_tracker_and_cluster(
    whale_analyzer: WhaleAnalyzer,
    event_collector,
    large_trade_payload_factory,
) -> None:
    first = await whale_analyzer.process_large_trade_signal(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=90_000.0,
            price=100.0,
            quantity=900.0,
            trade_id="direct-lt-1",
        )
    )
    second = await whale_analyzer.process_large_trade_signal(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=90_000.0,
            price=100.0,
            quantity=900.0,
            trade_id="direct-lt-2",
        )
    )

    assert first.has_signals is False
    assert second.has_signals is True

    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1, timeout=1.0)
    await event_collector.wait_for_topic(WHALE_PRESSURE_TOPIC, count=1, timeout=1.0)
    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1, timeout=1.0)


async def test_direct_liquidation_can_emit_context_after_direct_large_trades(
    whale_analyzer: WhaleAnalyzer,
    event_collector,
    large_trade_payload_factory,
    liquidations_updated_payload_factory,
) -> None:
    context = await _force_direct_whale_context(
        analyzer=whale_analyzer,
        large_trade_payload_factory=large_trade_payload_factory,
        liquidations_updated_payload_factory=liquidations_updated_payload_factory,
        symbol="BTCUSDT",
        whale_side="buy",
        liquidation_side="sell",
    )

    assert context is not None
    assert context.symbol == "BTCUSDT"
    assert context.whale_side == "buy"
    assert context.liquidation_side == "sell"

    await event_collector.wait_for_topic(
        WHALE_LIQUIDATION_CONTEXT_TOPIC,
        count=1,
        timeout=1.0,
    )


async def test_direct_cluster_inputs_reach_cluster_analyzer(
    whale_analyzer: WhaleAnalyzer,
    event_collector,
    whale_activity_payload_factory,
    whale_pressure_payload_factory,
    whale_liquidation_context_payload_factory,
) -> None:
    activity_result = await whale_analyzer.process_whale_activity_signal(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=200_000.0,
        )
    )
    pressure_result = await whale_analyzer.process_whale_pressure_signal(
        whale_pressure_payload_factory(
            symbol="BTCUSDT",
            dominant_side="buy",
            buy_notional=220_000.0,
            sell_notional=20_000.0,
            total_notional=240_000.0,
            imbalance_ratio=0.92,
        )
    )
    context_result = await whale_analyzer.process_whale_liquidation_context_signal(
        whale_liquidation_context_payload_factory(
            symbol="BTCUSDT",
            whale_side="buy",
            liquidation_side="sell",
            context_strength=0.95,
        )
    )

    assert activity_result.has_signals is True
    assert pressure_result.has_signals is True
    assert context_result.has_signals is True

    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1, timeout=1.0)


# =============================================================================
# Scoped isolation / concurrency
# =============================================================================

async def test_pipeline_does_not_cross_contaminate_symbols_or_scopes_under_interleaved_batches(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    trades_updated_payload_factory,
) -> None:
    await whale_analyzer.start()

    batches = [
        ("binance", "usdm_futures", "BTCUSDT", "realtime", "buy", 90_000.0),
        ("okx", "swap", "BTCUSDT", "realtime", "sell", 90_000.0),
        ("binance", "usdm_futures", "ETHUSDT", "realtime", "buy", 40_000.0),
        ("binance", "usdm_futures", "SOLUSDT", "realtime", "buy", 20_000.0),
        ("binance", "usdm_futures", "BTCUSDT", "1m", "buy", 90_000.0),
    ]

    for index, (exchange, market_type, symbol, timeframe, side, notional) in enumerate(batches):
        accepted = await event_bus.emit(
            TRADES_UPDATED_TOPIC,
            trades_updated_payload_factory(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
                side=side,
                count=2,
                notional=notional,
                batch_id=f"interleaved-{index}-{exchange}-{symbol}-{timeframe}",
            ),
            source="tests.trades_cache",
            correlation_id=f"corr-{exchange}-{symbol}-{timeframe}",
        )
        assert accepted is True

    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=5, timeout=1.0)

    binance_btc = whale_analyzer.get_symbol_stats(
        "BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    okx_btc = whale_analyzer.get_symbol_stats(
        "BTCUSDT",
        exchange="okx",
        market_type="swap",
        timeframe="realtime",
    )
    binance_btc_1m = whale_analyzer.get_symbol_stats(
        "BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="1m",
    )

    assert binance_btc["large_trade_detector"]["exists"] is True
    assert binance_btc["whale_tracker"]["exists"] is True
    assert binance_btc["whale_cluster_analyzer"]["exists"] is True

    assert okx_btc["large_trade_detector"]["exists"] is True
    assert okx_btc["whale_tracker"]["exists"] is True
    assert okx_btc["whale_cluster_analyzer"]["exists"] is True

    assert binance_btc_1m["large_trade_detector"]["exists"] is True
    assert binance_btc_1m["whale_tracker"]["exists"] is True
    assert binance_btc_1m["whale_cluster_analyzer"]["exists"] is True


async def test_full_pipeline_concurrent_batches_with_cooldowns_does_not_flood_clusters(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    trades_updated_payload_factory,
) -> None:
    config = WhalesConfig(
        enabled=True,
        auto_start_components=True,
        large_trade_detector=replace(
            whales_config_fast.large_trade_detector,
            signal_cooldown_sec=60.0,
            emit_on_bus=True,
            log_signals=False,
        ),
        whale_tracker=replace(
            whales_config_fast.whale_tracker,
            whale_activity_cooldown_sec=60.0,
            whale_pressure_cooldown_sec=60.0,
            emit_on_bus=True,
            log_signals=False,
        ),
        whale_cluster_analyzer=replace(
            whales_config_fast.whale_cluster_analyzer,
            cluster_emit_cooldown_sec=60.0,
            cluster_update_cooldown_sec=60.0,
            cluster_exhaustion_cooldown_sec=60.0,
            emit_on_bus=True,
            log_signals=False,
        ),
    )
    config.validate()

    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)
    await analyzer.start()

    payloads = [
        trades_updated_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            count=1,
            notional=90_000.0 + index,
            batch_id=f"concurrent-cooldown-{index}",
        )
        for index in range(50)
    ]

    await asyncio.gather(
        *[
            event_bus.emit(
                TRADES_UPDATED_TOPIC,
                payload,
                source="tests.concurrent_trades_cache",
                correlation_id="corr-concurrent-pipeline",
            )
            for payload in payloads
        ]
    )

    await asyncio.sleep(0.25)

    stats = analyzer.get_symbol_stats(
        "BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert stats["large_trade_detector"]["exists"] is True
    assert stats["large_trade_detector"]["trades_processed"] == 50
    assert stats["large_trade_detector"]["signals_emitted"] <= 1

    # Якщо detector cooldown пропустив тільки один large_trade,
    # tracker не має достатньо events для activity/pressure.
    assert len(event_collector.by_topic(LARGE_TRADE_TOPIC)) <= 1
    assert len(event_collector.by_topic(WHALE_ACTIVITY_TOPIC)) <= 1
    assert len(event_collector.by_topic(WHALE_PRESSURE_TOPIC)) <= 1
    assert len(event_collector.by_topic(WHALE_CLUSTER_TOPIC)) <= 1


# =============================================================================
# Health / stats / reset
# =============================================================================

async def test_analyzer_healthcheck_reports_pipeline_components_and_topics(
    whale_analyzer: WhaleAnalyzer,
    trades_updated_payload_factory,
) -> None:
    await whale_analyzer.start()

    await whale_analyzer.process_trade(
        trades_updated_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            count=2,
            notional=90_000.0,
            batch_id="health-direct-batch",
        )
    )

    health = whale_analyzer.get_healthcheck()

    assert health["component"] == "analyzer"
    assert health["enabled"] is True
    assert health["auto_start_components"] is True
    assert health["children_registered"] is True
    assert health["children_started"] is True
    assert health["scope"] == "exchange:market_type:symbol:timeframe"

    assert "large_trade_detector" in health["components"]
    assert "whale_tracker" in health["components"]
    assert "whale_cluster_analyzer" in health["components"]

    assert health["pipeline_topics"]["large_trade_input"] == [TRADES_UPDATED_TOPIC]
    assert health["pipeline_topics"]["large_trade_output"] == LARGE_TRADE_TOPIC
    assert LARGE_TRADE_TOPIC in health["pipeline_topics"]["whale_tracker_input"]
    assert health["pipeline_topics"]["whale_cluster_output"] == WHALE_CLUSTER_TOPIC


async def test_analyzer_get_stats_and_get_symbol_stats_are_scoped(
    whale_analyzer: WhaleAnalyzer,
    trades_updated_payload_factory,
) -> None:
    await whale_analyzer.process_trade(
        trades_updated_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="realtime",
            side="buy",
            count=2,
            notional=90_000.0,
            batch_id="stats-btc",
        )
    )
    await whale_analyzer.process_trade(
        trades_updated_payload_factory(
            exchange="okx",
            market_type="swap",
            symbol="BTCUSDT",
            timeframe="realtime",
            side="sell",
            count=2,
            notional=90_000.0,
            batch_id="stats-btc-okx",
        )
    )

    stats = whale_analyzer.get_stats()

    assert stats["enabled"] is True
    assert stats["scope"] == "exchange:market_type:symbol:timeframe"
    assert "large_trade_detector" in stats
    assert "whale_tracker" in stats
    assert "whale_cluster_analyzer" in stats

    binance_btc = whale_analyzer.get_symbol_stats(
        "BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    okx_btc = whale_analyzer.get_symbol_stats(
        "BTCUSDT",
        exchange="okx",
        market_type="swap",
        timeframe="realtime",
    )
    aggregate_btc = whale_analyzer.get_symbol_stats("BTCUSDT")

    assert binance_btc["large_trade_detector"]["exists"] is True
    assert okx_btc["large_trade_detector"]["exists"] is True

    assert aggregate_btc["symbol"] == "BTCUSDT"
    assert aggregate_btc["scope"] == "symbol-only aggregate"
    assert aggregate_btc["large_trade_detector"]["exists"] is True
    assert aggregate_btc["whale_tracker"]["exists"] is True
    assert aggregate_btc["whale_cluster_analyzer"]["exists"] is True


async def test_analyzer_reset_symbol_is_scoped_when_scope_is_provided(
    whale_analyzer: WhaleAnalyzer,
    trades_updated_payload_factory,
) -> None:
    await whale_analyzer.process_trade(
        trades_updated_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="realtime",
            side="buy",
            count=2,
            notional=90_000.0,
            batch_id="reset-binance-btc",
        )
    )
    await whale_analyzer.process_trade(
        trades_updated_payload_factory(
            exchange="okx",
            market_type="swap",
            symbol="BTCUSDT",
            timeframe="realtime",
            side="sell",
            count=2,
            notional=90_000.0,
            batch_id="reset-okx-btc",
        )
    )
    await whale_analyzer.process_trade(
        trades_updated_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="ETHUSDT",
            timeframe="realtime",
            side="buy",
            count=2,
            notional=90_000.0,
            batch_id="reset-binance-eth",
        )
    )

    await whale_analyzer.reset_symbol(
        "BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    binance_btc = whale_analyzer.get_symbol_stats(
        "BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    okx_btc = whale_analyzer.get_symbol_stats(
        "BTCUSDT",
        exchange="okx",
        market_type="swap",
        timeframe="realtime",
    )
    binance_eth = whale_analyzer.get_symbol_stats(
        "ETHUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert binance_btc["large_trade_detector"]["exists"] is False
    assert binance_btc["whale_tracker"]["exists"] is False
    assert binance_btc["whale_cluster_analyzer"]["exists"] is False

    assert okx_btc["large_trade_detector"]["exists"] is True
    assert okx_btc["whale_tracker"]["exists"] is True
    assert okx_btc["whale_cluster_analyzer"]["exists"] is True

    assert binance_eth["large_trade_detector"]["exists"] is True
    assert binance_eth["whale_tracker"]["exists"] is True
    assert binance_eth["whale_cluster_analyzer"]["exists"] is True

    await whale_analyzer.reset_symbol("BTCUSDT")

    aggregate_btc = whale_analyzer.get_symbol_stats("BTCUSDT")
    assert aggregate_btc["large_trade_detector"]["exists"] is False
    assert aggregate_btc["whale_tracker"]["exists"] is False
    assert aggregate_btc["whale_cluster_analyzer"]["exists"] is False

    await whale_analyzer.reset_all()

    assert whale_analyzer.large_trade_detector.get_all_stats() == {}
    assert whale_analyzer.whale_tracker.get_all_states() == {}
    assert whale_analyzer.whale_cluster_analyzer.get_all_states() == {}


async def test_analyzer_invalid_symbol_stats_api_is_safe(
    whale_analyzer: WhaleAnalyzer,
) -> None:
    assert whale_analyzer.get_symbol_stats("")["error"] == "invalid_symbol"
    assert whale_analyzer.get_symbol_stats("   ")["error"] == "invalid_symbol"