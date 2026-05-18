# tests/analytics/spoofing/test_spoofing_analyzer_pipeline.py

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import pytest

from analytics.spoofing import (
    DetectorDecision,
    DetectorResult,
    OrderbookLevelSnapshot,
    SpoofingAnalyzer,
    SpoofingComponent,
    SpoofingPattern,
    SpoofingSide,
)


# =============================================================================
# Local adversarial helpers
# =============================================================================


class EventWithId:
    """
    Event-like object для SpoofingAnalyzer._handle_event() /
    SpoofingAnalyzer._handle_legacy_raw_event().

    Analyzer зараз бере correlation_id саме з event.event_id.
    Metadata/header correlation_id навмисно відрізняється, щоб тест ловив
    accidental contract drift.
    """

    def __init__(
        self,
        payload: Any,
        *,
        topic: str = "market.orderbook.updated",
        event_id: str | None = "event-id-1",
        correlation_id: str | None = "correlation-id-that-should-not-be-used",
        source: str = "tests",
    ) -> None:
        self.payload = payload
        self.topic = topic
        self.source = source
        self.event_id = event_id
        self.correlation_id = correlation_id
        self.metadata = {"correlation_id": correlation_id}
        self.headers = {"correlation_id": correlation_id}


@dataclass(slots=True)
class FailingSubscription:
    topic: str
    handler: Any
    active: bool = True

    def unsubscribe(self) -> None:
        self.active = False


class FailingEventBus:
    """
    EventBus, який падає на emit().

    Потрібен для перевірки, що analyzer не ковтає publish-layer failures.
    """

    def __init__(self) -> None:
        self.subscriptions: list[FailingSubscription] = []
        self.emit_attempts: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def subscribe(self, topic: str, handler: Any, **kwargs: Any) -> FailingSubscription:
        subscription = FailingSubscription(topic=topic, handler=handler)
        self.subscriptions.append(subscription)
        return subscription

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        self.emit_attempts.append((topic, payload, kwargs))
        raise RuntimeError(f"emit failed for topic={topic}")


class ExplodingOptionalDetector:
    """
    Optional detector, який падає.

    Analyzer має ізолювати optional detector failures через _safe_run_detector()
    і продовжити pipeline.
    """

    def analyze_many(
        self,
        walls: Iterable[Any],
        *,
        key: tuple[str, str, str, str] | None = None,
        exchange: str | None = None,
        symbol: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        current_mid_price: float | None = None,
    ) -> list[DetectorResult]:
        raise RuntimeError("optional detector exploded")


class FakeOrderBookCache:
    """
    Read-only fake під SupportsOrderBookCache.

    Дає можливість тестувати production API process_key() без прямого
    process_orderbook()/raw payload path.
    """

    def __init__(self, book: dict[str, Any] | None = None) -> None:
        self.book = book or {}
        self.calls: list[dict[str, Any]] = []

    def get_book(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
        depth: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "market_type": market_type,
                "depth": depth,
            }
        )
        return dict(self.book)


def make_key(
    analyzer: SpoofingAnalyzer,
    *,
    exchange: str = "binance",
    market_type: str = "perpetual",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
) -> tuple[str, str, str, str]:
    return analyzer.make_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


# =============================================================================
# Registration / infrastructure contract
# =============================================================================


def test_register_subscribes_once_to_production_topics_and_registers_cleanup_job(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    """
    Analyzer має підписуватись на production data-layer topics, а не на raw
    market.orderbook / market.trade, якщо legacy mode явно не дозволений.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    analyzer.register()
    analyzer.register()

    expected_topics = set(spoofing_config.production_source_topics)
    actual_topics = {subscription.topic for subscription in mock_event_bus.subscriptions}

    assert actual_topics == expected_topics
    assert "market.orderbook" not in actual_topics
    assert "market.trade" not in actual_topics
    assert len(mock_event_bus.subscriptions) == len(expected_topics)

    for subscription in mock_event_bus.subscriptions:
        assert subscription.handler == analyzer._handle_event
        assert subscription.active is True

    assert len(mock_scheduler.interval_jobs) == 1
    cleanup_job = mock_scheduler.interval_jobs[0]

    assert cleanup_job.name == spoofing_config.analyzer.scheduler_cleanup_job_name
    assert cleanup_job.func == analyzer.cleanup_job
    assert cleanup_job.interval == spoofing_config.cleanup_interval_seconds
    assert cleanup_job.run_immediately is False
    assert cleanup_job.max_retries == 1
    assert cleanup_job.retry_delay == 0.5
    assert cleanup_job.timeout == 5.0
    assert cleanup_job.allow_overlap is False
    assert cleanup_job.enabled is True

    stats = analyzer.stats()
    assert stats["registered"] is True
    assert stats["cleanup_job_id"] == cleanup_job.job_id
    assert stats["config"]["production_source_topics"] == list(
        spoofing_config.production_source_topics
    )


def test_register_includes_legacy_raw_topics_only_when_explicitly_enabled(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    spoofing_config.analyzer.allow_legacy_raw_topics = True
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    analyzer.register()

    topics = {subscription.topic for subscription in mock_event_bus.subscriptions}

    assert set(spoofing_config.production_source_topics).issubset(topics)
    assert set(spoofing_config.analyzer.legacy_raw_topic_patterns).issubset(topics)

    for subscription in mock_event_bus.subscriptions:
        if subscription.topic in spoofing_config.analyzer.legacy_raw_topic_patterns:
            assert subscription.handler == analyzer._handle_legacy_raw_event
        else:
            assert subscription.handler == analyzer._handle_event


def test_register_does_not_mutate_infrastructure_when_analyzer_is_disabled(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    spoofing_config.analyzer.enabled = False
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    analyzer.register()

    assert mock_event_bus.subscriptions == []
    assert mock_scheduler.interval_jobs == []

    stats = analyzer.stats()
    assert stats["registered"] is False
    assert stats["cleanup_job_id"] is None


def test_register_requires_event_bus_for_integration_analyzer(spoofing_config) -> None:
    """
    Detector-и можуть жити без EventBus, але SpoofingAnalyzer — integration point,
    тому register() без EventBus має падати, а не робити вигляд registered.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
    )

    with pytest.raises(RuntimeError, match="requires EventBus"):
        analyzer.register()

    assert analyzer.stats()["registered"] is False


def test_register_does_not_create_cleanup_job_when_scheduler_cleanup_disabled(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    spoofing_config.analyzer.scheduler_cleanup_enabled = False
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    analyzer.register()

    assert len(mock_event_bus.subscriptions) == len(spoofing_config.production_source_topics)
    assert mock_scheduler.interval_jobs == []
    assert analyzer.stats()["cleanup_job_id"] is None


def test_stop_after_register_should_disable_subscriptions_and_cleanup_job(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    """
    Вразливість: у пакеті є ризик lifecycle-конфлікту _registered vs _running.
    Цей тест має впасти, якщо SpoofingAnalyzer.register() не синхронізує
    lifecycle зі stop() із BaseSpoofingAnalyzer.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    analyzer.register()
    assert analyzer.stats()["registered"] is True
    assert any(subscription.active for subscription in mock_event_bus.subscriptions)

    analyzer.stop()

    assert all(not subscription.active for subscription in mock_event_bus.subscriptions)
    assert mock_scheduler.interval_jobs[0].enabled is False
    assert analyzer.is_running is False


# =============================================================================
# Early exits / invalid input behavior
# =============================================================================


@pytest.mark.asyncio
async def test_process_snapshots_disabled_returns_output_without_touching_event_bus(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    wall_snapshot_set_factory,
) -> None:
    spoofing_config.enabled = False

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    key = make_key(analyzer)

    output = await analyzer.process_snapshots(
        snapshots=wall_snapshot_set_factory(),
        key=key,
        metadata={"source": "disabled-test"},
        correlation_id="corr-disabled",
    )

    assert output.key == key
    assert output.symbol == "BTCUSDT"
    assert output.exchange == "binance"
    assert output.market_type == "perpetual"
    assert output.timeframe == "realtime"
    assert output.signal is None
    assert output.detector_results == []
    assert output.tracked_walls == []
    assert output.lifecycle_events == []
    assert output.metadata["reason"] == "analyzer_disabled"
    assert output.metadata["source"] == "disabled-test"

    assert mock_event_bus.emitted == []


@pytest.mark.asyncio
async def test_process_snapshots_empty_input_is_noop_and_preserves_metadata(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    key = make_key(analyzer)

    output = await analyzer.process_snapshots(
        snapshots=[],
        key=key,
        metadata={"batch_id": "empty-batch"},
        correlation_id="corr-empty",
    )

    assert output.key == key
    assert output.signal is None
    assert output.detector_results == []
    assert output.tracked_walls == []
    assert output.metadata == {
        "reason": "no_levels_for_key",
        "batch_id": "empty-batch",
    }
    assert analyzer.get_latest_output_by_key(key) is output
    assert mock_event_bus.emitted == []


@pytest.mark.asyncio
async def test_process_snapshots_wrong_key_does_not_fall_back_to_payload_market(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    orderbook_snapshot_factory,
) -> None:
    """
    Якщо caller явно передав key для BTCUSDT/binance/perpetual/realtime,
    analyzer не має silently fallback-нутися на snapshots ETHUSDT.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    requested_key = make_key(analyzer, symbol="BTCUSDT")

    snapshots = [
        orderbook_snapshot_factory(
            symbol="ETHUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            price=1999.0,
            size=1000.0,
        ),
        orderbook_snapshot_factory(
            symbol="ETHUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            price=1998.0,
            size=1200.0,
        ),
    ]

    output = await analyzer.process_snapshots(
        snapshots=snapshots,
        key=requested_key,
        current_mid_price=2000.0,
        correlation_id="corr-market-mismatch",
    )

    assert output.key == requested_key
    assert output.signal is None
    assert output.detector_results == []
    assert output.tracked_walls == []
    assert output.metadata["reason"] == "no_levels_for_key"

    assert mock_event_bus.emitted == []


@pytest.mark.asyncio
async def test_process_snapshots_filters_poison_levels_unknown_side_and_bad_prices(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    orderbook_snapshot_factory,
) -> None:
    """
    Analyzer мусить відсікти невалідні рівні до tracker/detectors:
    - price <= 0;
    - size <= 0;
    - SpoofingSide.UNKNOWN.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    key = make_key(analyzer)

    snapshots = [
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            side=SpoofingSide.BID,
            price=0.0,
            size=1000.0,
        ),
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            side=SpoofingSide.ASK,
            price=100.2,
            size=0.0,
        ),
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            side=SpoofingSide.UNKNOWN,
            price=100.0,
            size=1000.0,
        ),
    ]

    output = await analyzer.process_snapshots(
        snapshots=snapshots,
        key=key,
        current_mid_price=100.0,
    )

    assert output.signal is None
    assert output.detector_results == []
    assert output.tracked_walls == []
    assert output.metadata["reason"] == "no_levels_after_filtering"
    assert output.metadata["input_levels_count"] == 3

    assert analyzer.persistence_tracker.stats()["tracked_walls"] == 0
    assert mock_event_bus.emitted == []


@pytest.mark.asyncio
async def test_process_snapshots_key_allowlist_rejects_disallowed_market_scope(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    wall_snapshot_set_factory,
) -> None:
    """
    Вразливість: allowlist має працювати на повному key, а не тільки на symbol.
    """

    spoofing_config.exchange_allowlist = {"bybit"}
    spoofing_config.market_type_allowlist = {"perpetual"}
    spoofing_config.symbol_allowlist = {"BTCUSDT"}
    spoofing_config.timeframe_allowlist = {"realtime"}
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    binance_key = make_key(analyzer, exchange="binance")

    output = await analyzer.process_snapshots(
        snapshots=wall_snapshot_set_factory(exchange="binance"),
        key=binance_key,
        current_mid_price=100.0,
    )

    assert output.signal is None
    assert output.detector_results == []
    assert output.metadata["reason"] == "key_not_allowed"
    assert mock_event_bus.emitted == []


@pytest.mark.asyncio
async def test_process_snapshots_below_min_levels_to_scan_is_not_analyzed(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    orderbook_snapshot_factory,
) -> None:
    """
    Надто короткий orderbook slice не має створювати signal.
    """

    spoofing_config.wall_detection.min_levels_to_scan = 5
    spoofing_config.wall_detection.max_levels_to_scan = 50
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    key = make_key(analyzer)

    snapshots = [
        orderbook_snapshot_factory(
            side=SpoofingSide.BID,
            price=99.9,
            size=50_000.0,
        ),
        orderbook_snapshot_factory(
            side=SpoofingSide.ASK,
            price=100.1,
            size=50_000.0,
        ),
    ]

    output = await analyzer.process_snapshots(
        snapshots=snapshots,
        key=key,
        current_mid_price=100.0,
    )

    assert output.signal is None
    assert output.detector_results == []
    assert output.tracked_walls == []
    assert output.metadata["reason"] == "no_levels_after_filtering"
    assert output.metadata["input_levels_count"] == 2

    assert mock_event_bus.emitted == []


# =============================================================================
# Real analyzer pipeline
# =============================================================================


@pytest.mark.asyncio
async def test_process_snapshots_real_pipeline_detects_large_wall_tracks_state_and_publishes_consistent_payloads(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    wall_snapshot_set_factory,
    assert_event_emitted,
) -> None:
    """
    Реальний happy-path без monkeypatch detector-ів.

    Перевіряє повний шлях:
    normalized snapshots -> filtering -> PersistenceTracker -> wall detector ->
    score engine -> AnalyzerOutput -> EventBus payloads.
    """

    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.analyzer.publish_updates = True
    spoofing_config.analyzer.publish_detected_only = False
    spoofing_config.analyzer.publish_lifecycle_events = True
    spoofing_config.analyzer.publish_score_updates = True
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    key = make_key(analyzer)

    snapshots = wall_snapshot_set_factory(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
        wall_price=99.9,
        wall_size=2_000.0,
        normal_size=1.0,
        mid_price=100.0,
    )

    output = await analyzer.process_snapshots(
        snapshots=snapshots,
        key=key,
        current_mid_price=100.0,
        metadata={"batch_id": "real-pipeline"},
        correlation_id="corr-real-pipeline",
    )

    assert output.key == key
    assert output.symbol == "BTCUSDT"
    assert output.exchange == "binance"
    assert output.market_type == "perpetual"
    assert output.timeframe == "realtime"
    assert output.tracked_walls
    assert output.lifecycle_events
    assert output.detector_results
    assert output.signal is not None

    assert output.metadata["scope"] == {
        "exchange": "binance",
        "market_type": "perpetual",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
    }
    assert output.metadata["input_levels_count"] == len(snapshots)
    assert output.metadata["filtered_levels_count"] == len(snapshots)
    assert output.metadata["current_mid_price"] == 100.0
    assert output.metadata["batch_id"] == "real-pipeline"

    assert analyzer.get_latest_output_by_key(key) is output
    assert analyzer.get_latest_signal_by_key(key) is output.signal

    updated_events = assert_event_emitted(
        mock_event_bus,
        spoofing_config.analyzer.event_topic_updated,
    )
    detected_events = assert_event_emitted(
        mock_event_bus,
        spoofing_config.analyzer.event_topic_detected,
    )
    score_events = assert_event_emitted(
        mock_event_bus,
        spoofing_config.analyzer.event_topic_score_updated,
    )
    lifecycle_events = assert_event_emitted(
        mock_event_bus,
        spoofing_config.analyzer.event_topic_lifecycle,
    )

    for event in updated_events + detected_events + score_events + lifecycle_events:
        assert event.correlation_id == "corr-real-pipeline"
        assert event.source == "analytics.spoofing.analyzer"

    updated_payload = updated_events[-1].payload
    assert updated_payload["scope"] == output.metadata["scope"]
    assert updated_payload["symbol"] == "BTCUSDT"
    assert updated_payload["exchange"] == "binance"
    assert updated_payload["market_type"] == "perpetual"
    assert updated_payload["timeframe"] == "realtime"
    assert updated_payload["tracked_walls"]
    assert updated_payload["detector_results"]
    assert updated_payload["metadata"]["batch_id"] == "real-pipeline"

    detected_payload = detected_events[-1].payload
    assert detected_payload["scope"] == output.metadata["scope"]
    assert detected_payload["score"] == output.signal.score
    assert detected_payload["confidence"] == output.signal.confidence
    assert detected_payload["signal"]["signal_id"] == output.signal.signal_id

    score_payload = score_events[-1].payload
    assert score_payload["scope"] == output.metadata["scope"]
    assert score_payload["signal_id"] == output.signal.signal_id
    assert score_payload["passed"] is True
    assert score_payload["contributions"]


@pytest.mark.asyncio
async def test_process_orderbook_manual_helper_ignores_malformed_levels_and_uses_valid_liquidity_only(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    """
    Manual/test helper має пережити частково биті bids/asks.
    Production runtime все одно має йти через market.orderbook.updated.
    """

    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    output = await analyzer.process_orderbook(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
        bids=[
            (99.9, 1.0),
            ("not-a-price", 10.0),
            (99.8, "not-a-size"),
            (99.7, -5.0),
            (99.6, 2_000.0),
        ],
        asks=[
            (100.1, 1.0),
            (100.2, 1.1),
            [100.3, 1.2],
            ("bad", "bad"),
        ],
        best_bid=99.9,
        best_ask=100.1,
        current_mid_price=100.0,
        correlation_id="corr-raw-helper",
    )

    assert output.symbol == "BTCUSDT"
    assert output.exchange == "binance"
    assert output.market_type == "perpetual"
    assert output.timeframe == "realtime"
    assert output.metadata["filtered_levels_count"] >= 3
    assert output.tracked_walls
    assert all(wall.current_size > 0 for wall in output.tracked_walls)


@pytest.mark.asyncio
async def test_process_event_payload_accepts_production_orderbook_updated_payload(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    orderbook_updated_payload_factory,
) -> None:
    """
    Основний production input: OrderBookCache -> market.orderbook.updated.
    """

    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    payload = orderbook_updated_payload_factory(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
        bids=[
            (99.9, 1.0),
            (99.8, 1.1),
            (99.7, 3_000.0),
        ],
        asks=[
            (100.1, 1.0),
            (100.2, 1.2),
            (100.3, 1.3),
        ],
        best_bid=99.9,
        best_ask=100.1,
        sequence_id=1001,
    )
    payload["current_mid_price"] = "100.0"

    output = await analyzer.process_event_payload(
        payload,
        correlation_id="corr-production-payload",
    )

    assert output.key == make_key(analyzer)
    assert output.tracked_walls
    assert output.detector_results
    assert output.metadata["event_payload"] == {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "market_type": "perpetual",
        "timeframe": "realtime",
        "sequence_id": 1001,
    }


@pytest.mark.asyncio
async def test_process_event_payload_rejects_raw_exchange_adapter_payload_by_default(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    raw_orderbook_payload_factory,
) -> None:
    """
    Raw exchange adapter payload не має бути production path.
    Без allow_raw_payload=True analyzer мусить відмовитись.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    payload = raw_orderbook_payload_factory(
        source="exchange_adapter",
        symbol="BTCUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
    )
    payload["source"] = "exchange_adapter"

    with pytest.raises(ValueError, match="Raw exchange adapter orderbook payload is not allowed"):
        await analyzer.process_event_payload(payload)


@pytest.mark.asyncio
async def test_legacy_raw_event_allows_raw_exchange_payload_when_legacy_mode_enabled(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    raw_orderbook_payload_factory,
) -> None:
    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.analyzer.allow_legacy_raw_topics = True
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    payload = raw_orderbook_payload_factory(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
        bids=[
            (99.9, 1.0),
            (99.8, 1.1),
            (99.7, 3_000.0),
        ],
        asks=[
            (100.1, 1.0),
            (100.2, 1.2),
            (100.3, 1.3),
        ],
    )
    payload["source"] = "exchange_adapter"

    await analyzer._handle_legacy_raw_event(
        EventWithId(
            payload=payload,
            topic="market.orderbook",
            event_id="legacy-event-id",
        )
    )

    assert mock_event_bus.emitted
    assert all(event.correlation_id == "legacy-event-id" for event in mock_event_bus.emitted)


@pytest.mark.asyncio
async def test_process_event_payload_normalizes_snapshot_dicts_and_skips_garbage(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    fixed_now,
) -> None:
    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    payload = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "market_type": "perpetual",
        "timeframe": "realtime",
        "sequence_id": 1001,
        "current_mid_price": "100.0",
        "snapshots": [
            {
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "market_type": "perpetual",
                "timeframe": "realtime",
                "side": "bid",
                "price": 99.9,
                "size": 1.0,
                "best_bid": 99.9,
                "best_ask": 100.1,
                "mid_price": 100.0,
                "timestamp": fixed_now,
            },
            {
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "market_type": "perpetual",
                "timeframe": "realtime",
                "side": "ask",
                "price": 100.1,
                "size": 1.1,
                "best_bid": 99.9,
                "best_ask": 100.1,
                "mid_price": 100.0,
                "timestamp": fixed_now,
            },
            {
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "market_type": "perpetual",
                "timeframe": "realtime",
                "side": "bid",
                "price": 99.8,
                "size": 3_000.0,
                "best_bid": 99.9,
                "best_ask": 100.1,
                "mid_price": 100.0,
                "timestamp": fixed_now,
            },
            {
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "market_type": "perpetual",
                "timeframe": "realtime",
                "side": "invalid-side",
                "price": 99.7,
                "size": 9_999.0,
            },
            "garbage-snapshot",
            {"symbol": "BTCUSDT"},
        ],
    }

    output = await analyzer.process_event_payload(
        payload,
        correlation_id="corr-snapshot-dicts",
    )

    assert output.key == make_key(analyzer)
    assert output.metadata["input_levels_count"] == 3
    assert output.metadata["filtered_levels_count"] == 3
    assert output.metadata["current_mid_price"] == 100.0
    assert output.metadata["event_payload"] == {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "market_type": "perpetual",
        "timeframe": "realtime",
        "sequence_id": 1001,
    }

    assert output.tracked_walls
    assert all(wall.side in {SpoofingSide.BID, SpoofingSide.ASK} for wall in output.tracked_walls)


@pytest.mark.asyncio
async def test_process_key_reads_from_orderbook_cache_without_direct_data_dependency(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    cache = FakeOrderBookCache(
        {
            "exchange": "binance",
            "market_type": "perpetual",
            "symbol": "BTCUSDT",
            "timeframe": "realtime",
            "bids": [
                (99.9, 1.0),
                (99.8, 1.1),
                (99.7, 4_000.0),
            ],
            "asks": [
                (100.1, 1.0),
                (100.2, 1.2),
                (100.3, 1.3),
            ],
            "best_bid": 99.9,
            "best_ask": 100.1,
            "sequence_id": 2001,
        }
    )

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
        orderbook_cache=cache,
    )
    key = make_key(analyzer)

    output = await analyzer.process_key(
        key,
        current_mid_price=100.0,
        correlation_id="corr-process-key-cache",
    )

    assert cache.calls == [
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "market_type": "perpetual",
            "depth": spoofing_config.wall_detection.max_levels_to_scan,
        }
    ]

    assert output.key == key
    assert output.tracked_walls
    assert output.detector_results
    assert output.metadata["scope"] == {
        "exchange": "binance",
        "market_type": "perpetual",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
    }


@pytest.mark.asyncio
async def test_process_key_empty_orderbook_cache_snapshot_is_noop(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
        orderbook_cache=FakeOrderBookCache({}),
    )
    key = make_key(analyzer)

    output = await analyzer.process_key(key)

    assert output.key == key
    assert output.signal is None
    assert output.detector_results == []
    assert output.tracked_walls == []
    assert output.metadata["reason"] == "empty_orderbook_cache_snapshot"
    assert mock_event_bus.emitted == []


# =============================================================================
# Key isolation / scope vulnerabilities
# =============================================================================


@pytest.mark.asyncio
async def test_same_symbol_different_exchange_market_type_and_timeframe_are_isolated(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    wall_snapshot_set_factory,
) -> None:
    """
    Однаковий BTCUSDT на різних exchange/market_type/timeframe не має змішуватись.
    """

    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    key_binance_perp_rt = make_key(
        analyzer,
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )
    key_bybit_perp_rt = make_key(
        analyzer,
        exchange="bybit",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )
    key_binance_linear_rt = make_key(
        analyzer,
        exchange="binance",
        market_type="linear",
        symbol="BTCUSDT",
        timeframe="realtime",
    )
    key_binance_perp_1m = make_key(
        analyzer,
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="1m",
    )

    snapshots = []
    snapshots.extend(
        wall_snapshot_set_factory(
            exchange="binance",
            market_type="perpetual",
            symbol="BTCUSDT",
            timeframe="realtime",
            wall_price=99.7,
            wall_size=2_000.0,
        )
    )
    snapshots.extend(
        wall_snapshot_set_factory(
            exchange="bybit",
            market_type="perpetual",
            symbol="BTCUSDT",
            timeframe="realtime",
            wall_price=99.6,
            wall_size=3_000.0,
        )
    )
    snapshots.extend(
        wall_snapshot_set_factory(
            exchange="binance",
            market_type="linear",
            symbol="BTCUSDT",
            timeframe="realtime",
            wall_price=99.5,
            wall_size=4_000.0,
        )
    )
    snapshots.extend(
        wall_snapshot_set_factory(
            exchange="binance",
            market_type="perpetual",
            symbol="BTCUSDT",
            timeframe="1m",
            wall_price=99.4,
            wall_size=5_000.0,
        )
    )

    output = await analyzer.process_snapshots(
        snapshots=snapshots,
        key=key_binance_perp_rt,
        current_mid_price=100.0,
        correlation_id="corr-key-isolation",
    )

    assert output.key == key_binance_perp_rt
    assert output.tracked_walls
    assert all(wall.key == key_binance_perp_rt for wall in output.tracked_walls)
    assert all(result.features.key == key_binance_perp_rt for result in output.detector_results)

    assert analyzer.persistence_tracker.get_walls_for_key(key_binance_perp_rt)
    assert analyzer.persistence_tracker.get_walls_for_key(key_bybit_perp_rt) == []
    assert analyzer.persistence_tracker.get_walls_for_key(key_binance_linear_rt) == []
    assert analyzer.persistence_tracker.get_walls_for_key(key_binance_perp_1m) == []


@pytest.mark.asyncio
async def test_event_payload_explicit_key_prevents_payload_scope_poisoning(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    orderbook_updated_payload_factory,
) -> None:
    """
    Якщо caller передав explicit key, payload із підміненою scope-інформацією
    не має примусово змінити цільовий market.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    explicit_key = make_key(
        analyzer,
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )

    payload = orderbook_updated_payload_factory(
        symbol="ETHUSDT",
        exchange="bybit",
        market_type="linear",
        timeframe="1m",
        bids=[(1999.0, 5_000.0)],
        asks=[(2001.0, 1.0)],
        best_bid=1999.0,
        best_ask=2001.0,
    )

    output = await analyzer.process_event_payload(
        payload,
        key=explicit_key,
        correlation_id="corr-scope-poisoning",
    )

    assert output.key == explicit_key
    assert output.signal is None
    assert output.detector_results == []
    assert output.metadata["reason"] == "no_levels_for_key"
    assert mock_event_bus.emitted == []


# =============================================================================
# EventBus handler behavior
# =============================================================================


@pytest.mark.asyncio
async def test_handle_event_non_mapping_payload_is_ignored_without_error_event(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    await analyzer._handle_event(
        EventWithId(
            payload=["not", "a", "dict"],
            topic="market.orderbook.updated",
            event_id="event-non-dict",
        )
    )

    assert mock_event_bus.emitted == []


@pytest.mark.asyncio
async def test_handle_event_missing_required_key_is_skipped_without_error_noise(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    """
    Missing key у production event — це skipped event, не publish error.
    Інакше dashboard/error channel буде шуміти на кожному неповному payload.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    await analyzer._handle_event(
        EventWithId(
            payload={
                "bids": [(99.9, 1.0)],
                "asks": [(100.1, 1.0)],
            },
            topic="market.orderbook.updated",
            event_id="event-missing-key",
        )
    )

    assert mock_event_bus.emitted == []


@pytest.mark.asyncio
async def test_handle_event_uses_event_id_as_correlation_id(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    orderbook_updated_payload_factory,
) -> None:
    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    payload = orderbook_updated_payload_factory(
        bids=[
            (99.9, 1.0),
            (99.8, 1.1),
            (99.7, 3_000.0),
        ],
        asks=[
            (100.1, 1.0),
            (100.2, 1.2),
            (100.3, 1.3),
        ],
    )

    await analyzer._handle_event(
        EventWithId(
            payload=payload,
            topic="market.orderbook.updated",
            event_id="event-id-used-as-correlation",
            correlation_id="metadata-correlation-ignored",
        )
    )

    assert mock_event_bus.emitted
    assert {
        event.correlation_id for event in mock_event_bus.emitted
    } == {"event-id-used-as-correlation"}


@pytest.mark.asyncio
async def test_handle_trade_updated_event_reprocesses_existing_key_from_cache(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    """
    Trade update не будує raw orderbook snapshots. Він має reprocess key через
    process_key(), тобто через orderbook_cache.
    """

    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    cache = FakeOrderBookCache(
        {
            "exchange": "binance",
            "market_type": "perpetual",
            "symbol": "BTCUSDT",
            "timeframe": "realtime",
            "bids": [
                (99.9, 1.0),
                (99.8, 1.1),
                (99.7, 3_000.0),
            ],
            "asks": [
                (100.1, 1.0),
                (100.2, 1.2),
                (100.3, 1.3),
            ],
            "best_bid": 99.9,
            "best_ask": 100.1,
        }
    )

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
        orderbook_cache=cache,
    )

    await analyzer._handle_event(
        EventWithId(
            topic="market.trades.updated",
            event_id="trade-event-id",
            payload={
                "exchange": "binance",
                "market_type": "perpetual",
                "symbol": "BTCUSDT",
                "timeframe": "realtime",
                "current_mid_price": 100.0,
            },
        )
    )

    assert cache.calls
    assert mock_event_bus.emitted
    assert all(event.correlation_id == "trade-event-id" for event in mock_event_bus.emitted)


# =============================================================================
# Publishing contract / failure propagation
# =============================================================================


@pytest.mark.asyncio
async def test_publish_layer_failure_is_not_swallowed(
    spoofing_config,
    mock_scheduler,
    wall_snapshot_set_factory,
) -> None:
    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    failing_bus = FailingEventBus()

    analyzer = SpoofingAnalyzer(
        event_bus=failing_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    key = make_key(analyzer)

    with pytest.raises(RuntimeError, match="emit failed"):
        await analyzer.process_snapshots(
            snapshots=wall_snapshot_set_factory(),
            key=key,
            current_mid_price=100.0,
            correlation_id="corr-failing-bus",
        )

    assert failing_bus.emit_attempts
    assert failing_bus.emit_attempts[0][0] in {
        spoofing_config.analyzer.event_topic_lifecycle,
        spoofing_config.analyzer.event_topic_updated,
        spoofing_config.analyzer.event_topic_score_updated,
        spoofing_config.analyzer.event_topic_detected,
    }


@pytest.mark.asyncio
async def test_handle_event_publish_error_event_and_reraises_processing_exception(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    raw_orderbook_payload_factory,
    assert_event_emitted,
) -> None:
    """
    _handle_event() має publish error event і re-raise, якщо processing впав.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    payload = raw_orderbook_payload_factory()
    payload["source"] = "exchange_adapter"

    with pytest.raises(ValueError, match="Raw exchange adapter orderbook payload is not allowed"):
        await analyzer._handle_event(
            EventWithId(
                payload=payload,
                topic="market.orderbook.updated",
                event_id="event-raw-forbidden",
            )
        )

    error_events = assert_event_emitted(
        mock_event_bus,
        spoofing_config.analyzer.event_topic_error,
    )
    error_payload = error_events[-1].payload

    assert error_payload["error_type"] == "ValueError"
    assert "Raw exchange adapter orderbook payload is not allowed" in error_payload["error"]
    assert error_payload["context"]["handler"] == "_handle_event"
    assert error_payload["payload_keys"]
    assert error_events[-1].correlation_id == "event-raw-forbidden"


@pytest.mark.asyncio
async def test_publish_detected_only_suppresses_update_when_no_signal_even_if_detector_results_exist(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    wall_snapshot_set_factory,
    detector_result_factory,
    monkeypatch,
    assert_no_event_emitted,
) -> None:
    spoofing_config.analyzer.publish_updates = True
    spoofing_config.analyzer.publish_detected_only = True
    spoofing_config.analyzer.publish_lifecycle_events = False
    spoofing_config.analyzer.publish_score_updates = True
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    key = make_key(analyzer)

    monkeypatch.setattr(
        analyzer,
        "_run_base_detectors",
        lambda **kwargs: [
            detector_result_factory(
                score=0.95,
                confidence=0.90,
                wall_id="strong-result-but-no-signal",
            )
        ],
    )
    monkeypatch.setattr(analyzer, "_run_optional_detectors", lambda **kwargs: [])
    monkeypatch.setattr(analyzer, "_build_signal", lambda **kwargs: None)

    output = await analyzer.process_snapshots(
        snapshots=wall_snapshot_set_factory(),
        key=key,
        current_mid_price=100.0,
        correlation_id="corr-detected-only-no-signal",
    )

    assert output.signal is None
    assert output.detector_results

    assert_no_event_emitted(
        mock_event_bus,
        spoofing_config.analyzer.event_topic_updated,
    )
    assert_no_event_emitted(
        mock_event_bus,
        spoofing_config.analyzer.event_topic_detected,
    )
    assert_no_event_emitted(
        mock_event_bus,
        spoofing_config.analyzer.event_topic_score_updated,
    )


@pytest.mark.asyncio
async def test_publish_flags_suppress_lifecycle_score_and_updates_independently(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    wall_snapshot_set_factory,
    assert_no_event_emitted,
) -> None:
    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.analyzer.publish_updates = False
    spoofing_config.analyzer.publish_lifecycle_events = False
    spoofing_config.analyzer.publish_score_updates = False
    spoofing_config.analyzer.publish_detected_only = False
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    key = make_key(analyzer)

    output = await analyzer.process_snapshots(
        snapshots=wall_snapshot_set_factory(),
        key=key,
        current_mid_price=100.0,
        correlation_id="corr-publish-flags",
    )

    assert output.signal is not None

    assert_no_event_emitted(mock_event_bus, spoofing_config.analyzer.event_topic_updated)
    assert_no_event_emitted(mock_event_bus, spoofing_config.analyzer.event_topic_lifecycle)
    assert_no_event_emitted(mock_event_bus, spoofing_config.analyzer.event_topic_score_updated)

    detected_events = mock_event_bus.events_for_topic(
        spoofing_config.analyzer.event_topic_detected
    )
    assert detected_events


@pytest.mark.asyncio
async def test_eventbus_payload_is_serialized_copy_not_mutable_tracker_reference(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    wall_snapshot_set_factory,
    assert_event_emitted,
) -> None:
    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    key = make_key(analyzer)

    output = await analyzer.process_snapshots(
        snapshots=wall_snapshot_set_factory(),
        key=key,
        current_mid_price=100.0,
    )

    updated_events = assert_event_emitted(
        mock_event_bus,
        spoofing_config.analyzer.event_topic_updated,
    )

    payload_wall = updated_events[-1].payload["tracked_walls"][0]
    wall_id = payload_wall["wall_id"]

    payload_wall["current_size"] = -999_999.0
    payload_wall["state"] = "corrupted"

    internal_wall = analyzer.persistence_tracker.get_wall(wall_id)

    assert internal_wall is not None
    assert internal_wall.current_size >= 0.0
    assert internal_wall.state.value != "corrupted"
    assert output.tracked_walls[0].current_size >= 0.0


# =============================================================================
# Detector failure / result explosion
# =============================================================================


@pytest.mark.asyncio
async def test_optional_detector_exception_is_isolated_and_does_not_hide_base_results(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    wall_snapshot_set_factory,
) -> None:
    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
        fake_liquidity_detector=ExplodingOptionalDetector(),
        flip_pressure_detector=None,
        layering_detector=None,
    )
    key = make_key(analyzer)

    output = await analyzer.process_snapshots(
        snapshots=wall_snapshot_set_factory(),
        key=key,
        current_mid_price=100.0,
        correlation_id="corr-optional-detector-explosion",
    )

    assert output.detector_results
    assert all(isinstance(item, DetectorResult) for item in output.detector_results)
    assert output.signal is not None


@pytest.mark.asyncio
async def test_base_detector_exception_is_not_silently_converted_to_empty_result(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    wall_snapshot_set_factory,
    monkeypatch,
) -> None:
    """
    Base detector failure має падати гучно. Інакше production може отримати
    silent no-signal при зламаній основній detector-логіці.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    key = make_key(analyzer)

    def broken_base_detectors(**kwargs: Any) -> list[DetectorResult]:
        raise RuntimeError("base detector exploded")

    monkeypatch.setattr(analyzer, "_run_base_detectors", broken_base_detectors)

    with pytest.raises(RuntimeError, match="base detector exploded"):
        await analyzer.process_snapshots(
            snapshots=wall_snapshot_set_factory(),
            key=key,
            current_mid_price=100.0,
        )

    assert mock_event_bus.emitted == []


@pytest.mark.asyncio
async def test_detector_result_explosion_is_limited_and_sorted_before_scoring(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    wall_snapshot_set_factory,
    detector_result_factory,
    monkeypatch,
) -> None:
    spoofing_config.analyzer.max_detector_results_per_cycle = 3
    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    key = make_key(analyzer)

    base_results = [
        detector_result_factory(
            detector=SpoofingComponent.ORDERBOOK_WALL_DETECTOR,
            score=0.10 + index * 0.01,
            confidence=0.50,
            wall_id=f"base-wall-{index}",
        )
        for index in range(20)
    ]

    optional_results = [
        detector_result_factory(
            detector=SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
            score=0.90 - index * 0.01,
            confidence=0.80,
            wall_id=f"optional-wall-{index}",
        )
        for index in range(20)
    ]

    monkeypatch.setattr(
        analyzer,
        "_run_base_detectors",
        lambda **kwargs: list(base_results),
    )
    monkeypatch.setattr(
        analyzer,
        "_run_optional_detectors",
        lambda **kwargs: list(optional_results),
    )

    output = await analyzer.process_snapshots(
        snapshots=wall_snapshot_set_factory(),
        key=key,
        current_mid_price=100.0,
        correlation_id="corr-detector-explosion",
    )

    assert len(output.detector_results) == spoofing_config.analyzer.max_detector_results_per_cycle

    scores = [item.score for item in output.detector_results]
    assert scores == sorted(scores, reverse=True)
    assert all(result.score >= 0.88 for result in output.detector_results)


# =============================================================================
# Adversarial market data
# =============================================================================


@pytest.mark.asyncio
async def test_crossed_book_and_inconsistent_best_quotes_do_not_crash_pipeline(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    """
    Реальні orderbook snapshots іноді приходять crossed або з inconsistent best.
    Analyzer не має падати; він має фільтрувати/оцінювати те, що може.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    output = await analyzer.process_orderbook(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
        bids=[
            (100.2, 1.0),
            (100.1, 3_000.0),
            (99.9, 1.0),
        ],
        asks=[
            (100.0, 1.0),
            (100.3, 1.2),
            (100.4, 1.3),
        ],
        best_bid=100.2,
        best_ask=100.0,
        current_mid_price=100.1,
    )

    assert output.key == make_key(analyzer)
    assert output.metadata["input_levels_count"] >= 3
    assert output.metadata["filtered_levels_count"] >= 3


@pytest.mark.asyncio
async def test_nan_inf_and_absurd_numeric_values_do_not_reach_tracker_as_valid_walls(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    output = await analyzer.process_orderbook(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
        bids=[
            (float("nan"), 1000.0),
            (99.9, float("nan")),
            (99.8, float("inf")),
            (99.7, 0.0),
        ],
        asks=[
            (float("inf"), 1000.0),
            (100.1, float("-inf")),
            (100.2, -1.0),
        ],
        best_bid=99.9,
        best_ask=100.1,
        current_mid_price=100.0,
    )

    assert output.signal is None or all(
        wall.current_size >= 0.0 for wall in output.tracked_walls
    )
    assert all(wall.price > 0.0 for wall in output.tracked_walls)
    assert all(wall.current_size >= 0.0 for wall in output.tracked_walls)


# =============================================================================
# Concurrency / state consistency
# =============================================================================


@pytest.mark.asyncio
async def test_concurrent_processing_same_key_does_not_create_duplicate_wall_ids(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    wall_snapshot_set_factory,
) -> None:
    """
    Вразливість: PersistenceTracker мутує dict/set state. Цей тест шукає
    дублікати wall_id / index inconsistency при паралельній обробці одного key.
    """

    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )
    key = make_key(analyzer)

    batches = [
        wall_snapshot_set_factory(
            wall_price=99.7,
            wall_size=2_000.0 + index * 100.0,
            normal_size=1.0,
            mid_price=100.0,
        )
        for index in range(10)
    ]

    outputs = await asyncio.gather(
        *[
            analyzer.process_snapshots(
                snapshots=batch,
                key=key,
                current_mid_price=100.0,
                correlation_id=f"corr-concurrent-{index}",
            )
            for index, batch in enumerate(batches)
        ]
    )

    walls = analyzer.persistence_tracker.get_walls_for_key(key)
    wall_ids = [wall.wall_id for wall in walls]

    assert outputs
    assert len(wall_ids) == len(set(wall_ids))
    assert analyzer.persistence_tracker.snapshot_state(key=key)
    assert analyzer.get_latest_output_by_key(key) in outputs


# =============================================================================
# Cleanup job behavior
# =============================================================================


@pytest.mark.asyncio
async def test_cleanup_job_calls_cleanup_and_returns_none(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    monkeypatch,
) -> None:
    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    calls: list[str] = []

    def fake_cleanup() -> int:
        calls.append("cleanup")
        return 123

    monkeypatch.setattr(analyzer, "cleanup", fake_cleanup)

    result = await analyzer.cleanup_job()

    assert result is None
    assert calls == ["cleanup"]


@pytest.mark.asyncio
async def test_scheduler_registered_cleanup_job_is_callable_and_async_safe(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    analyzer.register()

    job = mock_scheduler.get_job_by_name(
        spoofing_config.analyzer.scheduler_cleanup_job_name
    )

    assert job is not None
    assert job.func == analyzer.cleanup_job

    result = await job.func()

    assert result is None