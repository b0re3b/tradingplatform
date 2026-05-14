from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any

import pytest

from core.event_bus import EventBus
from core.scheduler import Scheduler

from analytics.whales.analyzer import WhaleAnalyzer
from analytics.whales.config import (
    LargeTradeDetectorConfig,
    WhaleClusterAnalyzerConfig,
    WhaleTrackerConfig,
    WhalesConfig,
)


pytestmark = pytest.mark.asyncio


MARKET_TRADE_TOPIC = "market.trade"
MARKET_LIQUIDATION_TOPIC = "market.liquidation"

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


def _disable_bus_emission(config: WhalesConfig) -> WhalesConfig:
    isolated = WhalesConfig(
        enabled=config.enabled,
        auto_start_components=config.auto_start_components,
        large_trade_detector=replace(
            config.large_trade_detector,
            emit_on_bus=False,
            log_signals=False,
        ),
        whale_tracker=replace(
            config.whale_tracker,
            emit_on_bus=False,
            log_signals=False,
        ),
        whale_cluster_analyzer=replace(
            config.whale_cluster_analyzer,
            emit_on_bus=False,
            log_signals=False,
        ),
    )
    isolated.validate()
    return isolated


async def _emit_trade_burst(
    *,
    event_bus: EventBus,
    raw_trade_payload_factory,
    symbol: str = "BTCUSDT",
    side: str = "buy",
    count: int = 2,
    notional: float = 90_000.0,
    correlation_id: str = "corr-trade-burst",
    source: str = "tests.market_stream",
) -> None:
    for index in range(count):
        accepted = await event_bus.emit(
            MARKET_TRADE_TOPIC,
            raw_trade_payload_factory(
                symbol=symbol,
                price=100.0,
                quantity=notional / 100.0,
                side=side,
                trade_id=f"{symbol}-{side}-burst-{index}",
            ),
            source=source,
            correlation_id=correlation_id,
        )
        assert accepted is True


async def _force_direct_whale_context(
    *,
    analyzer: WhaleAnalyzer,
    large_trade_payload_factory,
    liquidation_payload_factory,
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
        liquidation_payload_factory(
            symbol=symbol,
            side=liquidation_side,
            notional=100_000.0,
            price=100.0,
            quantity=1_000.0,
        )
    )


# =============================================================================
# Construction / dependency injection
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

    assert whale_analyzer.large_trade_detector.is_registered is True
    assert whale_analyzer.whale_tracker.is_registered is True
    assert whale_analyzer.whale_cluster_analyzer.is_registered is True

    assert len(whale_analyzer.large_trade_detector.subscriptions) == 1
    assert len(whale_analyzer.whale_tracker.subscriptions) == 2
    assert len(whale_analyzer.whale_cluster_analyzer.subscriptions) == 3

    # Facade сам не має власних EventBus subscriptions.
    assert len(whale_analyzer.subscriptions) == 0


async def test_analyzer_start_is_idempotent_and_starts_children_once(
    whale_analyzer: WhaleAnalyzer,
) -> None:
    await whale_analyzer.start()
    await whale_analyzer.start()
    await whale_analyzer.start()

    assert whale_analyzer.is_started is True

    assert whale_analyzer.large_trade_detector.is_started is True
    assert whale_analyzer.whale_tracker.is_started is True
    assert whale_analyzer.whale_cluster_analyzer.is_started is True

    assert len(whale_analyzer.large_trade_detector.subscriptions) == 1
    assert len(whale_analyzer.whale_tracker.subscriptions) == 2
    assert len(whale_analyzer.whale_cluster_analyzer.subscriptions) == 3

    assert len(whale_analyzer.large_trade_detector.scheduler_job_ids) == 1
    assert len(whale_analyzer.whale_tracker.scheduler_job_ids) == 1
    assert len(whale_analyzer.whale_cluster_analyzer.scheduler_job_ids) == 1


async def test_analyzer_stop_is_idempotent_and_stops_children(
    whale_analyzer: WhaleAnalyzer,
) -> None:
    await whale_analyzer.start()

    await whale_analyzer.stop()
    await whale_analyzer.stop()

    assert whale_analyzer.is_started is False
    assert whale_analyzer.is_registered is False

    assert whale_analyzer.large_trade_detector.is_started is False
    assert whale_analyzer.whale_tracker.is_started is False
    assert whale_analyzer.whale_cluster_analyzer.is_started is False

    assert len(whale_analyzer.large_trade_detector.subscriptions) == 0
    assert len(whale_analyzer.whale_tracker.subscriptions) == 0
    assert len(whale_analyzer.whale_cluster_analyzer.subscriptions) == 0

    assert len(whale_analyzer.large_trade_detector.scheduler_job_ids) == 0
    assert len(whale_analyzer.whale_tracker.scheduler_job_ids) == 0
    assert len(whale_analyzer.whale_cluster_analyzer.scheduler_job_ids) == 0


async def test_analyzer_auto_start_false_starts_facade_but_not_child_subscriptions(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    config = replace(whales_config_fast, auto_start_components=False)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    await analyzer.start()

    assert analyzer.is_started is True
    assert analyzer.is_registered is True

    assert analyzer.large_trade_detector.is_started is False
    assert analyzer.whale_tracker.is_started is False
    assert analyzer.whale_cluster_analyzer.is_started is False

    assert len(analyzer.large_trade_detector.subscriptions) == 0
    assert len(analyzer.whale_tracker.subscriptions) == 0
    assert len(analyzer.whale_cluster_analyzer.subscriptions) == 0

    accepted = await event_bus.emit(
        MARKET_TRADE_TOPIC,
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=1_000.0,
            side="buy",
        ),
        source="tests.market_stream",
        correlation_id="corr-auto-start-false",
    )

    assert accepted is True
    await asyncio.sleep(0.05)

    assert event_collector.by_topic(LARGE_TRADE_TOPIC) == []
    assert event_collector.by_topic(WHALE_CLUSTER_TOPIC) == []


async def test_disabled_analyzer_does_not_start_register_or_process_children(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    config = replace(whales_config_fast, enabled=False)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    await analyzer.register()
    await analyzer.start()

    assert analyzer.is_registered is False
    assert analyzer.is_started is False

    assert analyzer.large_trade_detector.is_registered is False
    assert analyzer.whale_tracker.is_registered is False
    assert analyzer.whale_cluster_analyzer.is_registered is False

    direct_signal = await analyzer.process_trade(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=1_000.0,
            side="buy",
        )
    )

    # Важлива поведінка: top-level enabled=False не вимикає автоматично
    # підконфіги дочірніх компонентів для direct API. Якщо це небажано,
    # цей тест підсвітить архітектурний розрив.
    assert direct_signal is not None

    await asyncio.sleep(0.05)
    assert event_collector.by_topic(LARGE_TRADE_TOPIC) != []


async def test_analyzer_start_failure_does_not_mark_facade_started_but_can_leave_partial_child_state(
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

    # Після виправлення WhaleAnalyzer.start() має rollback-ити вже запущені children.
    assert whale_analyzer.large_trade_detector.is_started is False
    assert whale_analyzer.whale_tracker.is_started is False
    assert whale_analyzer.whale_cluster_analyzer.is_started is False

    assert whale_analyzer.large_trade_detector.is_registered is False
    assert whale_analyzer.whale_tracker.is_registered is False
    assert whale_analyzer.whale_cluster_analyzer.is_registered is False

    await whale_analyzer.stop()

    assert whale_analyzer.large_trade_detector.is_started is False
    assert whale_analyzer.whale_tracker.is_started is False
    assert whale_analyzer.whale_cluster_analyzer.is_started is False


# =============================================================================
# Full EventBus pipeline
# =============================================================================


async def test_full_eventbus_pipeline_market_trade_burst_emits_large_activity_pressure_and_cluster(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    await whale_analyzer.start()

    await _emit_trade_burst(
        event_bus=event_bus,
        raw_trade_payload_factory=raw_trade_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=90_000.0,
        correlation_id="corr-full-pipeline-1",
    )

    await event_collector.wait_for_topic(LARGE_TRADE_TOPIC, count=2)
    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1)
    await event_collector.wait_for_topic(WHALE_PRESSURE_TOPIC, count=1)
    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1)

    large_trade_events = event_collector.by_topic(LARGE_TRADE_TOPIC)
    activity_events = event_collector.by_topic(WHALE_ACTIVITY_TOPIC)
    pressure_events = event_collector.by_topic(WHALE_PRESSURE_TOPIC)
    cluster_events = event_collector.by_topic(WHALE_CLUSTER_TOPIC)

    assert len(large_trade_events) == 2
    assert activity_events[0].payload["symbol"] == "BTCUSDT"
    assert pressure_events[0].payload["dominant_side"] == "buy"
    assert cluster_events[0].payload["symbol"] == "BTCUSDT"

    assert large_trade_events[0].correlation_id == "corr-full-pipeline-1"
    assert activity_events[0].correlation_id == "corr-full-pipeline-1"
    assert pressure_events[0].correlation_id == "corr-full-pipeline-1"
    assert cluster_events[0].correlation_id == "corr-full-pipeline-1"


async def test_full_eventbus_pipeline_ignores_noise_before_valid_burst(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    await whale_analyzer.start()

    noisy_payloads: list[Any] = [
        None,
        [],
        "bad",
        {"data": None},
        {"symbol": "", "price": 100, "quantity": 1, "side": "buy"},
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
    ]

    for index, payload in enumerate(noisy_payloads):
        await event_bus.emit(
            MARKET_TRADE_TOPIC,
            payload,
            source="tests.noisy_market",
            correlation_id=f"corr-noise-{index}",
        )

    await asyncio.sleep(0.05)

    assert event_collector.by_topic(LARGE_TRADE_TOPIC) == []
    assert event_collector.by_topic(WHALE_CLUSTER_TOPIC) == []

    await _emit_trade_burst(
        event_bus=event_bus,
        raw_trade_payload_factory=raw_trade_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=90_000.0,
        correlation_id="corr-after-noise",
    )

    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1)

    stats = whale_analyzer.get_symbol_stats("BTCUSDT")
    assert stats["large_trade_detector"]["exists"] is True
    assert stats["whale_tracker"]["exists"] is True
    assert stats["whale_cluster_analyzer"]["exists"] is True

    cluster_event = event_collector.by_topic(WHALE_CLUSTER_TOPIC)[0]
    assert cluster_event.correlation_id == "corr-after-noise"


async def test_full_eventbus_pipeline_liquidation_flow_emits_context_and_can_feed_cluster(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
    liquidation_payload_factory,
) -> None:
    await whale_analyzer.start()

    await _emit_trade_burst(
        event_bus=event_bus,
        raw_trade_payload_factory=raw_trade_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=90_000.0,
        correlation_id="corr-liquidation-flow",
    )

    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1)

    accepted = await event_bus.emit(
        MARKET_LIQUIDATION_TOPIC,
        liquidation_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=100_000.0,
            price=100.0,
            quantity=1_000.0,
        ),
        source="tests.liquidation_stream",
        correlation_id="corr-liquidation-flow",
    )
    assert accepted is True

    await event_collector.wait_for_topic(WHALE_LIQUIDATION_CONTEXT_TOPIC, count=1)
    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1)

    context_event = event_collector.by_topic(WHALE_LIQUIDATION_CONTEXT_TOPIC)[0]
    assert context_event.correlation_id == "corr-liquidation-flow"
    assert context_event.payload["symbol"] == "BTCUSDT"
    assert context_event.payload["whale_side"] == "buy"
    assert context_event.payload["liquidation_side"] == "sell"


async def test_pipeline_does_not_cross_contaminate_symbols_under_interleaved_events(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    await whale_analyzer.start()

    interleaved = [
        ("BTCUSDT", "buy", 90_000.0),
        ("ETHUSDT", "sell", 40_000.0),
        ("BTCUSDT", "buy", 90_000.0),
        ("ETHUSDT", "sell", 40_000.0),
        ("SOLUSDT", "buy", 20_000.0),
        ("SOLUSDT", "buy", 20_000.0),
    ]

    for index, (symbol, side, notional) in enumerate(interleaved):
        await event_bus.emit(
            MARKET_TRADE_TOPIC,
            raw_trade_payload_factory(
                symbol=symbol,
                price=100.0,
                quantity=notional / 100.0,
                side=side,
                trade_id=f"interleaved-{symbol}-{index}",
            ),
            source="tests.interleaved_market",
            correlation_id=f"corr-{symbol}",
        )

    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=2)

    cluster_symbols = {
        event.payload["symbol"] for event in event_collector.by_topic(WHALE_CLUSTER_TOPIC)
    }

    assert "BTCUSDT" in cluster_symbols
    assert "ETHUSDT" in cluster_symbols

    btc_stats = whale_analyzer.get_symbol_stats("BTCUSDT")
    eth_stats = whale_analyzer.get_symbol_stats("ETHUSDT")
    sol_stats = whale_analyzer.get_symbol_stats("SOLUSDT")

    assert btc_stats["whale_tracker"]["exists"] is True
    assert eth_stats["whale_tracker"]["exists"] is True

    # SOLUSDT має symbol-specific threshold 10_000 у fast config,
    # тому теж може пройти detector. Перевіряємо не конкретний сигнал,
    # а ізоляцію state по symbol.
    assert sol_stats["symbol"] == "SOLUSDT"


async def test_full_pipeline_concurrent_market_trades_with_cooldowns_does_not_flood_clusters(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    raw_trade_payload_factory,
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
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=(90_000.0 + index) / 100.0,
            side="buy",
            trade_id=f"concurrent-pipeline-{index}",
        )
        for index in range(50)
    ]

    await asyncio.gather(
        *[
            event_bus.emit(
                MARKET_TRADE_TOPIC,
                payload,
                source="tests.concurrent_market",
                correlation_id="corr-concurrent-pipeline",
            )
            for payload in payloads
        ]
    )

    await asyncio.sleep(0.25)

    stats = analyzer.get_symbol_stats("BTCUSDT")

    assert stats["large_trade_detector"]["exists"] is True
    assert stats["large_trade_detector"]["trades_processed"] == 50
    assert stats["large_trade_detector"]["signals_emitted"] <= 1

    # Якщо detector cooldown пропустив лише один large_trade, tracker не має
    # формувати activity/pressure, бо для fast config потрібно 2 large trades.
    assert len(event_collector.by_topic(WHALE_ACTIVITY_TOPIC)) <= 1
    assert len(event_collector.by_topic(WHALE_PRESSURE_TOPIC)) <= 1
    assert len(event_collector.by_topic(WHALE_CLUSTER_TOPIC)) <= 1


async def test_pipeline_after_stop_no_longer_reacts_to_market_events(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    await whale_analyzer.start()
    await whale_analyzer.stop()

    await _emit_trade_burst(
        event_bus=event_bus,
        raw_trade_payload_factory=raw_trade_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=90_000.0,
        correlation_id="corr-after-stop",
    )

    await asyncio.sleep(0.05)

    assert event_collector.by_topic(LARGE_TRADE_TOPIC) == []
    assert event_collector.by_topic(WHALE_ACTIVITY_TOPIC) == []
    assert event_collector.by_topic(WHALE_CLUSTER_TOPIC) == []


# =============================================================================
# Direct API / backtesting behavior
# =============================================================================


async def test_direct_process_trade_returns_only_large_trade_signal_not_full_cascade(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = _disable_bus_emission(whales_config_fast)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    signal = await analyzer.process_trade(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=900.0,
            side="buy",
            trade_id="direct-trade-only",
        )
    )

    assert signal is not None
    assert signal.symbol == "BTCUSDT"
    assert signal.notional == 90_000.0

    stats = analyzer.get_symbol_stats("BTCUSDT")
    assert stats["large_trade_detector"]["exists"] is True

    # Direct process_trade делегує лише в LargeTradeDetector.
    assert stats["whale_tracker"]["exists"] is False
    assert stats["whale_cluster_analyzer"]["exists"] is False


async def test_direct_large_trade_signal_feeds_tracker_but_not_cluster_without_bus(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    large_trade_payload_factory,
) -> None:
    config = _disable_bus_emission(whales_config_fast)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    first = await analyzer.process_large_trade_signal(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=90_000.0,
            trade_id="direct-large-1",
        )
    )
    second = await analyzer.process_large_trade_signal(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=90_000.0,
            trade_id="direct-large-2",
        )
    )

    assert first.has_signals is False
    assert second.has_signals is True
    assert second.whale_activity_signal is not None
    assert second.whale_pressure_signal is not None

    stats = analyzer.get_symbol_stats("BTCUSDT")
    assert stats["whale_tracker"]["exists"] is True
    assert stats["whale_cluster_analyzer"]["exists"] is False


async def test_direct_whale_activity_signal_feeds_cluster_analyzer(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    whale_activity_payload_factory,
) -> None:
    config = _disable_bus_emission(whales_config_fast)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    result = await analyzer.process_whale_activity_signal(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            trade_count=3,
            total_notional=200_000.0,
            avg_notional=66_666.0,
            max_notional=90_000.0,
        )
    )

    assert result.has_signals is True
    assert result.whale_cluster_signal is not None

    stats = analyzer.get_symbol_stats("BTCUSDT")
    assert stats["large_trade_detector"]["exists"] is False
    assert stats["whale_tracker"]["exists"] is False
    assert stats["whale_cluster_analyzer"]["exists"] is True


async def test_direct_liquidation_without_prior_whale_trades_returns_none(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    liquidation_payload_factory,
) -> None:
    config = _disable_bus_emission(whales_config_fast)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    result = await analyzer.process_liquidation(
        liquidation_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=100_000.0,
        )
    )

    assert result is None

    stats = analyzer.get_symbol_stats("BTCUSDT")
    assert stats["large_trade_detector"]["exists"] is False
    assert stats["whale_tracker"]["exists"] is True
    assert stats["whale_cluster_analyzer"]["exists"] is False


async def test_direct_liquidation_context_requires_prior_tracker_state(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    large_trade_payload_factory,
    liquidation_payload_factory,
) -> None:
    config = _disable_bus_emission(whales_config_fast)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    context = await _force_direct_whale_context(
        analyzer=analyzer,
        large_trade_payload_factory=large_trade_payload_factory,
        liquidation_payload_factory=liquidation_payload_factory,
        symbol="BTCUSDT",
        whale_side="buy",
        liquidation_side="sell",
    )

    assert context is not None
    assert context.symbol == "BTCUSDT"
    assert context.whale_side == "buy"
    assert context.liquidation_side == "sell"


async def test_direct_cluster_pressure_and_liquidation_context_paths_update_same_cluster_state(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    whale_activity_payload_factory,
    whale_pressure_payload_factory,
    whale_liquidation_context_payload_factory,
) -> None:
    config = _disable_bus_emission(whales_config_fast)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    activity_result = await analyzer.process_whale_activity_signal(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=200_000.0,
        )
    )
    pressure_result = await analyzer.process_whale_pressure_signal(
        whale_pressure_payload_factory(
            symbol="BTCUSDT",
            dominant_side="buy",
            buy_notional=220_000.0,
            sell_notional=20_000.0,
            imbalance_ratio=0.92,
        )
    )
    context_result = await analyzer.process_whale_liquidation_context_signal(
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

    stats = analyzer.get_symbol_stats("BTCUSDT")
    cluster_state = stats["whale_cluster_analyzer"]

    assert cluster_state["exists"] is True
    assert cluster_state["activity_records_size"] == 1
    assert cluster_state["pressure_records_size"] == 1
    assert cluster_state["liquidation_context_records_size"] == 1
    assert cluster_state["total_events_seen"] == 3


async def test_direct_api_invalid_payloads_do_not_poison_other_layers(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
    whale_activity_payload_factory,
) -> None:
    config = _disable_bus_emission(whales_config_fast)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    invalid_payloads: list[Any] = [
        None,
        [],
        "bad",
        {"data": None},
        {"symbol": "", "side": "buy", "notional": 100_000},
        {"symbol": "BTCUSDT", "side": "teleport", "notional": 100_000},
    ]

    for payload in invalid_payloads:
        trade_result = await analyzer.process_trade(payload)
        large_trade_result = await analyzer.process_large_trade_signal(payload)
        activity_result = await analyzer.process_whale_activity_signal(payload)

        assert trade_result is None
        assert large_trade_result.has_signals is False
        assert activity_result.has_signals is False

    assert analyzer.get_stats()["large_trade_detector"] == {}
    assert analyzer.get_stats()["whale_tracker"] == {}
    assert analyzer.get_stats()["whale_cluster_analyzer"] == {}

    valid = await analyzer.process_trade(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=900.0,
            side="buy",
        )
    )
    assert valid is not None

    cluster_result = await analyzer.process_whale_activity_signal(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=200_000.0,
        )
    )
    assert cluster_result.has_signals is True


# =============================================================================
# Health / stats / reset
# =============================================================================


async def test_healthcheck_and_stats_include_all_children_and_runtime_flags(
    whale_analyzer: WhaleAnalyzer,
    raw_trade_payload_factory,
    large_trade_payload_factory,
    whale_activity_payload_factory,
) -> None:
    await whale_analyzer.start()

    await whale_analyzer.process_trade(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=900.0,
            side="buy",
        )
    )
    await whale_analyzer.process_large_trade_signal(
        large_trade_payload_factory(
            symbol="ETHUSDT",
            side="sell",
            notional=90_000.0,
        )
    )
    await whale_analyzer.process_whale_activity_signal(
        whale_activity_payload_factory(
            symbol="SOLUSDT",
            side="buy",
            total_notional=200_000.0,
        )
    )

    health = whale_analyzer.get_healthcheck()
    stats = whale_analyzer.get_stats()

    assert health["component"] == "analyzer"
    assert health["started"] is True
    assert health["registered"] is False or isinstance(health["registered"], bool)
    assert health["enabled"] is True
    assert health["auto_start_components"] is True
    assert set(health["components"]) == {
        "large_trade_detector",
        "whale_tracker",
        "whale_cluster_analyzer",
    }

    assert stats["analyzer_started"] is True
    assert "BTCUSDT" in stats["large_trade_detector"]
    assert "ETHUSDT" in stats["whale_tracker"]
    assert "SOLUSDT" in stats["whale_cluster_analyzer"]


async def test_get_symbol_stats_normalizes_symbol_and_handles_invalid_symbols(
    whale_analyzer: WhaleAnalyzer,
    raw_trade_payload_factory,
) -> None:
    await whale_analyzer.process_trade(
        raw_trade_payload_factory(
            symbol="btcusdt",
            price=100.0,
            quantity=900.0,
            side="buy",
        )
    )

    stats = whale_analyzer.get_symbol_stats("  btcusdt  ")

    assert stats["symbol"] == "BTCUSDT"
    assert stats["large_trade_detector"]["exists"] is True
    assert stats["whale_tracker"]["exists"] is False
    assert stats["whale_cluster_analyzer"]["exists"] is False

    assert whale_analyzer.get_symbol_stats("") == {
        "symbol": "",
        "exists": False,
        "error": "invalid_symbol",
    }
    assert whale_analyzer.get_symbol_stats("   ") == {
        "symbol": "   ",
        "exists": False,
        "error": "invalid_symbol",
    }


async def test_reset_symbol_clears_all_child_states_for_only_that_symbol(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
    large_trade_payload_factory,
    whale_activity_payload_factory,
) -> None:
    config = _disable_bus_emission(whales_config_fast)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    await analyzer.process_trade(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=900.0,
            side="buy",
        )
    )
    await analyzer.process_large_trade_signal(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=90_000.0,
        )
    )
    await analyzer.process_whale_activity_signal(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=200_000.0,
        )
    )

    await analyzer.process_trade(
        raw_trade_payload_factory(
            symbol="ETHUSDT",
            price=100.0,
            quantity=900.0,
            side="sell",
        )
    )
    await analyzer.process_large_trade_signal(
        large_trade_payload_factory(
            symbol="ETHUSDT",
            side="sell",
            notional=90_000.0,
        )
    )
    await analyzer.process_whale_activity_signal(
        whale_activity_payload_factory(
            symbol="ETHUSDT",
            side="sell",
            total_notional=200_000.0,
        )
    )

    await analyzer.reset_symbol("btcusdt")

    btc = analyzer.get_symbol_stats("BTCUSDT")
    eth = analyzer.get_symbol_stats("ETHUSDT")

    assert btc["large_trade_detector"]["exists"] is False
    assert btc["whale_tracker"]["exists"] is False
    assert btc["whale_cluster_analyzer"]["exists"] is False

    assert eth["large_trade_detector"]["exists"] is True
    assert eth["whale_tracker"]["exists"] is True
    assert eth["whale_cluster_analyzer"]["exists"] is True


async def test_reset_all_clears_every_child_state_but_not_lifecycle(
    whale_analyzer: WhaleAnalyzer,
    raw_trade_payload_factory,
    large_trade_payload_factory,
    whale_activity_payload_factory,
) -> None:
    await whale_analyzer.start()

    await whale_analyzer.process_trade(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=900.0,
            side="buy",
        )
    )
    await whale_analyzer.process_large_trade_signal(
        large_trade_payload_factory(
            symbol="ETHUSDT",
            side="sell",
            notional=90_000.0,
        )
    )
    await whale_analyzer.process_whale_activity_signal(
        whale_activity_payload_factory(
            symbol="SOLUSDT",
            side="buy",
            total_notional=200_000.0,
        )
    )

    assert whale_analyzer.get_stats()["large_trade_detector"] != {}
    assert whale_analyzer.get_stats()["whale_tracker"] != {}
    assert whale_analyzer.get_stats()["whale_cluster_analyzer"] != {}

    await whale_analyzer.reset_all()

    stats = whale_analyzer.get_stats()
    assert stats["large_trade_detector"] == {}
    assert stats["whale_tracker"] == {}
    assert stats["whale_cluster_analyzer"] == {}

    assert whale_analyzer.is_started is True
    assert whale_analyzer.large_trade_detector.is_started is True
    assert whale_analyzer.whale_tracker.is_started is True
    assert whale_analyzer.whale_cluster_analyzer.is_started is True


# =============================================================================
# Cleanup / scheduler jobs
# =============================================================================


async def test_scheduler_cleanup_jobs_can_remove_stale_child_states(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
    large_trade_payload_factory,
    whale_activity_payload_factory,
) -> None:
    config = WhalesConfig(
        enabled=True,
        auto_start_components=True,
        large_trade_detector=replace(
            whales_config_fast.large_trade_detector,
            stats_ttl_sec=0.01,
            cleanup_interval_sec=0.01,
            emit_on_bus=False,
            log_signals=False,
        ),
        whale_tracker=replace(
            whales_config_fast.whale_tracker,
            stats_ttl_sec=0.01,
            cleanup_interval_sec=0.01,
            emit_on_bus=False,
            log_signals=False,
        ),
        whale_cluster_analyzer=replace(
            whales_config_fast.whale_cluster_analyzer,
            stats_ttl_sec=0.01,
            cleanup_interval_sec=0.01,
            emit_on_bus=False,
            log_signals=False,
        ),
    )
    config.validate()

    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)
    await analyzer.start()

    await analyzer.process_trade(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=900.0,
            side="buy",
        )
    )
    await analyzer.process_large_trade_signal(
        large_trade_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            notional=90_000.0,
        )
    )
    await analyzer.process_whale_activity_signal(
        whale_activity_payload_factory(
            symbol="BTCUSDT",
            side="buy",
            total_notional=200_000.0,
        )
    )

    assert analyzer.get_symbol_stats("BTCUSDT")["large_trade_detector"]["exists"] is True
    assert analyzer.get_symbol_stats("BTCUSDT")["whale_tracker"]["exists"] is True
    assert analyzer.get_symbol_stats("BTCUSDT")["whale_cluster_analyzer"]["exists"] is True

    await asyncio.sleep(0.03)

    # Не покладаємось на timing scheduler-а: викликаємо cleanup прямо,
    # але перевіряємо, що jobs були зареєстровані через start().
    assert len(analyzer.large_trade_detector.scheduler_job_ids) == 1
    assert len(analyzer.whale_tracker.scheduler_job_ids) == 1
    assert len(analyzer.whale_cluster_analyzer.scheduler_job_ids) == 1

    await analyzer.large_trade_detector.cleanup()
    await analyzer.whale_tracker.cleanup()
    await analyzer.whale_cluster_analyzer.cleanup()

    stats = analyzer.get_symbol_stats("BTCUSDT")
    assert stats["large_trade_detector"]["exists"] is False
    assert stats["whale_tracker"]["exists"] is False
    assert stats["whale_cluster_analyzer"]["exists"] is False


# =============================================================================
# Adversarial EventBus / correlation / duplicate subscriptions
# =============================================================================


async def test_repeated_register_start_does_not_create_duplicate_pipeline_events(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    await whale_analyzer.register()
    await whale_analyzer.register()
    await whale_analyzer.start()
    await whale_analyzer.start()

    await _emit_trade_burst(
        event_bus=event_bus,
        raw_trade_payload_factory=raw_trade_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=90_000.0,
        correlation_id="corr-no-duplicate-subscriptions",
    )

    await event_collector.wait_for_topic(LARGE_TRADE_TOPIC, count=2)
    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1)
    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1)

    assert len(event_collector.by_topic(LARGE_TRADE_TOPIC)) == 2
    assert len(event_collector.by_topic(WHALE_ACTIVITY_TOPIC)) == 1
    assert len(event_collector.by_topic(WHALE_CLUSTER_TOPIC)) == 1


async def test_pipeline_handles_mixed_correlation_ids_without_overwriting_payload_state(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    await whale_analyzer.start()

    await event_bus.emit(
        MARKET_TRADE_TOPIC,
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=900.0,
            side="buy",
            trade_id="corr-a-1",
        ),
        source="tests.market_stream",
        correlation_id="corr-a",
    )
    await event_bus.emit(
        MARKET_TRADE_TOPIC,
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=900.0,
            side="buy",
            trade_id="corr-b-1",
        ),
        source="tests.market_stream",
        correlation_id="corr-b",
    )

    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1)
    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1)

    large_trade_corrs = {
        event.correlation_id for event in event_collector.by_topic(LARGE_TRADE_TOPIC)
    }
    assert large_trade_corrs == {"corr-a", "corr-b"}

    # Aggregated activity/cluster формується на другій події, тому correlation_id
    # має відповідати event, який спричинив emission, а не випадково першому.
    activity_event = event_collector.by_topic(WHALE_ACTIVITY_TOPIC)[0]
    cluster_event = event_collector.by_topic(WHALE_CLUSTER_TOPIC)[0]

    assert activity_event.correlation_id == "corr-b"
    assert cluster_event.correlation_id == "corr-b"


async def test_one_symbol_invalid_liquidation_does_not_break_other_symbol_trade_pipeline(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    await whale_analyzer.start()

    invalid_liquidations = [
        {"symbol": "BTCUSDT", "side": "bad", "notional": 100_000},
        {"symbol": "BTCUSDT", "side": "sell", "notional": -1},
        {"symbol": "", "side": "sell", "notional": 100_000},
    ]

    for payload in invalid_liquidations:
        await event_bus.emit(
            MARKET_LIQUIDATION_TOPIC,
            payload,
            source="tests.bad_liquidations",
            correlation_id="corr-bad-liquidation",
        )

    await _emit_trade_burst(
        event_bus=event_bus,
        raw_trade_payload_factory=raw_trade_payload_factory,
        symbol="ETHUSDT",
        side="sell",
        count=2,
        notional=40_000.0,
        correlation_id="corr-eth-after-bad-liq",
    )

    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1)

    eth_stats = whale_analyzer.get_symbol_stats("ETHUSDT")
    btc_stats = whale_analyzer.get_symbol_stats("BTCUSDT")

    assert eth_stats["whale_tracker"]["exists"] is True
    assert eth_stats["whale_cluster_analyzer"]["exists"] is True

    assert btc_stats["large_trade_detector"]["exists"] is False
    assert btc_stats["whale_tracker"]["exists"] is False
    assert btc_stats["whale_cluster_analyzer"]["exists"] is False


async def test_pipeline_restart_after_stop_does_not_reuse_old_subscriptions_or_duplicate_jobs(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    await whale_analyzer.start()
    await whale_analyzer.stop()
    await whale_analyzer.start()

    assert len(whale_analyzer.large_trade_detector.subscriptions) == 1
    assert len(whale_analyzer.whale_tracker.subscriptions) == 2
    assert len(whale_analyzer.whale_cluster_analyzer.subscriptions) == 3

    assert len(whale_analyzer.large_trade_detector.scheduler_job_ids) == 1
    assert len(whale_analyzer.whale_tracker.scheduler_job_ids) == 1
    assert len(whale_analyzer.whale_cluster_analyzer.scheduler_job_ids) == 1

    await _emit_trade_burst(
        event_bus=event_bus,
        raw_trade_payload_factory=raw_trade_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=90_000.0,
        correlation_id="corr-after-restart",
    )

    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1)

    assert len(event_collector.by_topic(WHALE_ACTIVITY_TOPIC)) == 1
    assert len(event_collector.by_topic(WHALE_CLUSTER_TOPIC)) == 1


# =============================================================================
# Component accessors / mutation vulnerability
# =============================================================================


async def test_get_components_exposes_live_components_and_mutation_affects_pipeline(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    components = whale_analyzer.get_components()

    # Це навмисний vulnerability-test: get_components() повертає live object-и,
    # отже зовнішній код може змінити behavior компонента.
    components["large_trade_detector"].config.emit_on_bus = False

    await whale_analyzer.start()

    await _emit_trade_burst(
        event_bus=event_bus,
        raw_trade_payload_factory=raw_trade_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=90_000.0,
        correlation_id="corr-mutated-component",
    )

    await asyncio.sleep(0.05)

    assert event_collector.by_topic(LARGE_TRADE_TOPIC) == []
    assert event_collector.by_topic(WHALE_ACTIVITY_TOPIC) == []
    assert event_collector.by_topic(WHALE_CLUSTER_TOPIC) == []

    stats = whale_analyzer.get_symbol_stats("BTCUSDT")
    assert stats["large_trade_detector"]["exists"] is True
    assert stats["large_trade_detector"]["signals_emitted"] == 2
    assert stats["whale_tracker"]["exists"] is False


# =============================================================================
# Backward-compatible package-level expectations
# =============================================================================


async def test_package_facade_is_consistent_with_direct_child_methods(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
    large_trade_payload_factory,
    whale_activity_payload_factory,
) -> None:
    config = _disable_bus_emission(whales_config_fast)
    analyzer = _build_analyzer(config=config, event_bus=event_bus, scheduler=scheduler)

    trade_payload = raw_trade_payload_factory(
        symbol="BTCUSDT",
        price=100.0,
        quantity=900.0,
        side="buy",
        trade_id="facade-child-trade",
    )
    large_trade_payload = large_trade_payload_factory(
        symbol="ETHUSDT",
        side="sell",
        notional=90_000.0,
        trade_id="facade-child-large",
    )
    activity_payload = whale_activity_payload_factory(
        symbol="SOLUSDT",
        side="buy",
        total_notional=200_000.0,
    )

    facade_trade = await analyzer.process_trade(trade_payload)
    child_trade = await analyzer.large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BNBUSDT",
            price=100.0,
            quantity=900.0,
            side="buy",
            trade_id="child-trade",
        )
    )

    facade_tracker = await analyzer.process_large_trade_signal(large_trade_payload)
    child_tracker = await analyzer.whale_tracker.process_large_trade_payload(
        large_trade_payload_factory(
            symbol="ETHUSDT",
            side="sell",
            notional=90_000.0,
            trade_id="child-large-2",
        )
    )

    facade_cluster = await analyzer.process_whale_activity_signal(activity_payload)
    child_cluster = await analyzer.whale_cluster_analyzer.process_whale_activity_payload(
        whale_activity_payload_factory(
            symbol="SOLUSDT",
            side="buy",
            total_notional=220_000.0,
        )
    )

    assert facade_trade is not None
    assert child_trade is not None

    assert facade_tracker.has_signals is False
    assert child_tracker.has_signals is True

    assert facade_cluster.has_signals is True
    assert child_cluster.has_signals is True


# =============================================================================
# Regression guard for known model serialization issue
# =============================================================================


async def test_full_pipeline_signal_models_are_serializable_when_emitted_on_bus(
    whale_analyzer: WhaleAnalyzer,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
    liquidation_payload_factory,
) -> None:
    """
    Regression-test для бага, який уже показали попередні тести:
    slotted dataclass signal models не мають падати в to_payload() під час EventBus emit.
    """

    await whale_analyzer.start()

    await _emit_trade_burst(
        event_bus=event_bus,
        raw_trade_payload_factory=raw_trade_payload_factory,
        symbol="BTCUSDT",
        side="buy",
        count=2,
        notional=90_000.0,
        correlation_id="corr-serialization-regression",
    )

    await event_bus.emit(
        MARKET_LIQUIDATION_TOPIC,
        liquidation_payload_factory(
            symbol="BTCUSDT",
            side="sell",
            notional=100_000.0,
        ),
        source="tests.liquidation_stream",
        correlation_id="corr-serialization-regression",
    )

    await event_collector.wait_for_topic(LARGE_TRADE_TOPIC, count=2)
    await event_collector.wait_for_topic(WHALE_ACTIVITY_TOPIC, count=1)
    await event_collector.wait_for_topic(WHALE_PRESSURE_TOPIC, count=1)
    await event_collector.wait_for_topic(WHALE_LIQUIDATION_CONTEXT_TOPIC, count=1)
    await event_collector.wait_for_topic(WHALE_CLUSTER_TOPIC, count=1)

    for topic in {
        LARGE_TRADE_TOPIC,
        WHALE_ACTIVITY_TOPIC,
        WHALE_PRESSURE_TOPIC,
        WHALE_LIQUIDATION_CONTEXT_TOPIC,
        WHALE_CLUSTER_TOPIC,
    }:
        payloads = event_collector.payloads_by_topic(topic)
        assert payloads, f"No payloads collected for {topic}"

        for payload in payloads:
            assert payload["schema_version"] == 1
            assert isinstance(payload["event_type"], str)
            assert isinstance(payload["detector"], str)
            assert isinstance(payload["created_at_ms"], int)
            assert payload["symbol"] == "BTCUSDT"