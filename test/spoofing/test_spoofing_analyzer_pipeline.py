# tests/analytics/spoofing/test_spoofing_analyzer_pipeline.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from analytics.spoofing import (
    AnalyzerOutput,
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
    Event-like object for SpoofingAnalyzer.on_orderbook_event().

    Важливо: analyzer зараз бере correlation_id саме з event.event_id,
    а не з event.correlation_id / metadata. Цей fake навмисно це перевіряє.
    """

    def __init__(
        self,
        payload: Any,
        *,
        event_id: str | None = "event-id-1",
        correlation_id: str | None = "correlation-id-that-should-not-be-used",
    ) -> None:
        self.payload = payload
        self.event_id = event_id
        self.correlation_id = correlation_id
        self.metadata = {"correlation_id": correlation_id}
        self.headers = {"correlation_id": correlation_id}


class FailingEventBus:
    """
    EventBus, який падає на emit().

    Потрібен для перевірки, що analyzer не ковтає помилки publish layer.
    """

    def __init__(self) -> None:
        self.subscriptions: list[tuple[str, Any]] = []
        self.emit_attempts: list[tuple[str, dict[str, Any]]] = []

    def subscribe(self, topic: str, handler: Any, **kwargs: Any) -> None:
        self.subscriptions.append((topic, handler))

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        self.emit_attempts.append((topic, payload))
        raise RuntimeError(f"emit failed for topic={topic}")


# =============================================================================
# Registration / infrastructure contract
# =============================================================================


def test_register_subscribes_once_and_registers_cleanup_job_with_strict_core_contract(
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
    analyzer.register()

    assert len(mock_event_bus.subscriptions) == 1
    subscription = mock_event_bus.subscriptions[0]

    assert subscription.topic == spoofing_config.analyzer.event_topic_orderbook
    assert subscription.handler == analyzer.on_orderbook_event

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


def test_register_is_safe_with_missing_event_bus_and_scheduler(
    spoofing_config,
) -> None:
    analyzer = SpoofingAnalyzer(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
    )

    analyzer.register()

    stats = analyzer.stats()
    assert stats["registered"] is True
    assert stats["cleanup_job_id"] is None


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

    assert len(mock_event_bus.subscriptions) == 1
    assert mock_scheduler.interval_jobs == []
    assert analyzer.stats()["cleanup_job_id"] is None


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

    output = await analyzer.process_snapshots(
        snapshots=wall_snapshot_set_factory(),
        symbol="BTCUSDT",
        exchange="binance",
        metadata={"source": "disabled-test"},
        correlation_id="corr-disabled",
    )

    assert output.symbol == "BTCUSDT"
    assert output.exchange == "binance"
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

    output = await analyzer.process_snapshots(
        snapshots=[],
        symbol="BTCUSDT",
        exchange="binance",
        metadata={"batch_id": "empty-batch"},
        correlation_id="corr-empty",
    )

    assert output.symbol == "BTCUSDT"
    assert output.exchange == "binance"
    assert output.signal is None
    assert output.metadata == {
        "reason": "empty_snapshots",
        "batch_id": "empty-batch",
    }
    assert mock_event_bus.emitted == []


@pytest.mark.asyncio
async def test_process_snapshots_explicit_market_mismatch_does_not_fall_back_to_payload_market(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: якщо caller явно передав symbol/exchange, analyzer не має
    silently fallback-нутися на перший snapshot з іншого ринку.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    snapshots = [
        orderbook_snapshot_factory(
            symbol="ETHUSDT",
            exchange="binance",
            price=1999.0,
            size=1000.0,
        ),
        orderbook_snapshot_factory(
            symbol="ETHUSDT",
            exchange="binance",
            price=1998.0,
            size=1200.0,
        ),
    ]

    output = await analyzer.process_snapshots(
        snapshots=snapshots,
        symbol="BTCUSDT",
        exchange="binance",
        current_mid_price=2000.0,
        correlation_id="corr-market-mismatch",
    )

    assert output.symbol == "BTCUSDT"
    assert output.exchange == "binance"
    assert output.signal is None
    assert output.detector_results == []
    assert output.tracked_walls == []
    assert output.metadata["reason"] == "no_levels_for_resolved_market"

    assert mock_event_bus.emitted == []


@pytest.mark.asyncio
async def test_process_snapshots_filters_poison_levels_unknown_side_disallowed_symbol_and_bad_prices(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: analyzer мусить відсікти всі невалідні рівні до tracker/detectors.

    Тут одночасно перевіряємо:
    - price <= 0;
    - size <= 0;
    - SpoofingSide.UNKNOWN;
    - symbol whitelist.
    """

    spoofing_config.symbols = ["BTCUSDT"]
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    snapshots = [
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            side=SpoofingSide.BID,
            price=0.0,
            size=1000.0,
        ),
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            side=SpoofingSide.ASK,
            price=100.2,
            size=0.0,
        ),
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            side=SpoofingSide.UNKNOWN,
            price=100.0,
            size=1000.0,
        ),
        orderbook_snapshot_factory(
            symbol="ETHUSDT",
            side=SpoofingSide.BID,
            price=99.9,
            size=1000.0,
        ),
    ]

    output = await analyzer.process_snapshots(
        snapshots=snapshots,
        symbol="BTCUSDT",
        exchange="binance",
        current_mid_price=100.0,
    )

    assert output.signal is None
    assert output.detector_results == []
    assert output.tracked_walls == []
    assert output.metadata["reason"] == "no_levels_after_filtering"

    assert analyzer.persistence_tracker.stats()["tracked_walls"] == 0
    assert mock_event_bus.emitted == []


@pytest.mark.asyncio
async def test_process_snapshots_below_min_levels_to_scan_is_not_analyzed(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: надто короткий orderbook slice не має створювати сигнал.

    Якщо цей тест падає, analyzer може генерувати spoofing-сигнали на
    недостатній глибині стакана.
    """

    spoofing_config.wall_detection.min_levels_to_scan = 5
    spoofing_config.wall_detection.max_levels_to_scan = 50
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

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
        symbol="BTCUSDT",
        exchange="binance",
        current_mid_price=100.0,
    )

    assert output.signal is None
    assert output.detector_results == []
    assert output.tracked_walls == []
    assert output.metadata["reason"] == "no_levels_after_filtering"

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

    Тут перевіряємо не зручний isolated unit, а повний шлях:
    snapshots -> filtering -> PersistenceTracker -> wall detector ->
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

    snapshots = wall_snapshot_set_factory(
        symbol="BTCUSDT",
        exchange="binance",
        wall_price=99.9,
        wall_size=2_000.0,
        normal_size=1.0,
        mid_price=100.0,
    )

    output = await analyzer.process_snapshots(
        snapshots=snapshots,
        symbol="BTCUSDT",
        exchange="binance",
        current_mid_price=100.0,
        metadata={"batch_id": "real-pipeline"},
        correlation_id="corr-real-pipeline",
    )

    assert output.symbol == "BTCUSDT"
    assert output.exchange == "binance"
    assert output.tracked_walls
    assert output.lifecycle_events
    assert output.detector_results
    assert output.signal is not None

    assert output.metadata["input_levels_count"] == len(snapshots)
    assert output.metadata["filtered_levels_count"] == len(snapshots)
    assert output.metadata["current_mid_price"] == 100.0
    assert output.metadata["batch_id"] == "real-pipeline"

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
    assert updated_payload["symbol"] == "BTCUSDT"
    assert updated_payload["exchange"] == "binance"
    assert updated_payload["tracked_walls"]
    assert updated_payload["detector_results"]
    assert updated_payload["metadata"]["batch_id"] == "real-pipeline"

    detected_payload = detected_events[-1].payload
    assert detected_payload["symbol"] == "BTCUSDT"
    assert detected_payload["exchange"] == "binance"
    assert detected_payload["score"] == output.signal.score
    assert detected_payload["confidence"] == output.signal.confidence
    assert detected_payload["signal"]["signal_id"] == output.signal.signal_id

    score_payload = score_events[-1].payload
    assert score_payload["signal_id"] == output.signal.signal_id
    assert score_payload["passed"] is True
    assert score_payload["contributions"]


@pytest.mark.asyncio
async def test_process_orderbook_raw_input_ignores_malformed_levels_and_uses_valid_liquidity_only(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    """
    Вразливість: raw bids/asks часто приходять частково битими.
    Analyzer має не падати на смітті всередині book-side, але має впасти,
    якщо відсутні required symbol/exchange.
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
        sequence_id=991,
        current_mid_price=None,
        correlation_id="corr-raw-book",
    )

    assert output.symbol == "BTCUSDT"
    assert output.exchange == "binance"
    assert output.metadata["input_levels_count"] == 5
    assert output.metadata["filtered_levels_count"] == 5
    assert output.metadata["current_mid_price"] == pytest.approx(100.0)

    assert all(wall.symbol == "BTCUSDT" for wall in output.tracked_walls)
    assert all(wall.exchange == "binance" for wall in output.tracked_walls)


@pytest.mark.asyncio
async def test_process_event_payload_accepts_snapshot_dicts_and_discards_invalid_snapshot_items(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    fixed_now,
) -> None:
    """
    Вразливість: payload['snapshots'] може містити суміш dict, dataclass,
    сміття, невалідні side/price/size. Analyzer має нормалізувати лише
    валідну частину і не зламати весь batch.
    """

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
        "sequence_id": 1001,
        "current_mid_price": "100.0",
        "snapshots": [
            {
                "symbol": "BTCUSDT",
                "exchange": "binance",
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

    assert output.symbol == "BTCUSDT"
    assert output.exchange == "binance"
    assert output.metadata["input_levels_count"] == 5
    assert output.metadata["filtered_levels_count"] == 3
    assert output.metadata["current_mid_price"] == 100.0
    assert output.metadata["event_payload"] == {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "sequence_id": 1001,
    }

    assert output.tracked_walls
    assert all(wall.side in {SpoofingSide.BID, SpoofingSide.ASK} for wall in output.tracked_walls)


# =============================================================================
# Error handling / failure propagation
# =============================================================================


@pytest.mark.asyncio
async def test_on_orderbook_event_non_dict_payload_is_ignored_without_error_event(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    await analyzer.on_orderbook_event(
        EventWithId(
            payload=["not", "a", "dict"],
            event_id="event-non-dict",
        )
    )

    assert mock_event_bus.emitted == []


@pytest.mark.asyncio
async def test_on_orderbook_event_missing_required_payload_keys_publishes_error_and_reraises(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    assert_event_emitted,
) -> None:
    """
    Вразливість: bad payload не має тихо зникнути.
    Поточний контракт analyzer-а: publish error event і re-raise.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    with pytest.raises(ValueError, match="Missing required orderbook payload keys"):
        await analyzer.on_orderbook_event(
            EventWithId(
                payload={
                    "bids": [(99.9, 1.0)],
                    "asks": [(100.1, 1.0)],
                },
                event_id="event-bad-payload",
            )
        )

    error_events = assert_event_emitted(
        mock_event_bus,
        spoofing_config.analyzer.event_topic_error,
    )

    error_payload = error_events[-1].payload

    assert error_payload["error_type"] == "ValueError"
    assert "Missing required orderbook payload keys" in error_payload["error"]
    assert error_payload["context"]["handler"] == "on_orderbook_event"
    assert error_payload["payload"]["bids"] == [(99.9, 1.0)]


@pytest.mark.asyncio
async def test_on_orderbook_event_uses_event_id_as_correlation_id_not_metadata_correlation_id(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    raw_orderbook_payload_factory,
) -> None:
    """
    Regression-test на поточну реалізацію _extract_event_correlation_id().

    Якщо в майбутньому захочеш брати metadata['correlation_id'], цей тест
    треба буде змінити свідомо.
    """

    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    await analyzer.on_orderbook_event(
        EventWithId(
            payload=raw_orderbook_payload_factory(
                bids=[
                    (99.9, 1.0),
                    (99.8, 1.1),
                    (99.7, 2_000.0),
                ],
                asks=[
                    (100.1, 1.0),
                    (100.2, 1.1),
                    (100.3, 1.2),
                ],
            ),
            event_id="event-id-used-as-correlation",
            correlation_id="metadata-correlation-ignored",
        )
    )

    assert mock_event_bus.emitted
    assert {event.correlation_id for event in mock_event_bus.emitted} == {
        "event-id-used-as-correlation"
    }


@pytest.mark.asyncio
async def test_publish_failure_from_event_bus_is_not_swallowed(
    spoofing_config,
    mock_scheduler,
    wall_snapshot_set_factory,
) -> None:
    """
    Вразливість: якщо EventBus.emit падає, analyzer не має робити вигляд,
    що pipeline успішний. Інакше signal може загубитися без retry/error path.
    """

    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.analyzer.publish_updates = True
    spoofing_config.analyzer.publish_detected_only = False
    spoofing_config.validate()

    failing_bus = FailingEventBus()

    analyzer = SpoofingAnalyzer(
        event_bus=failing_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    with pytest.raises(RuntimeError, match="emit failed"):
        await analyzer.process_snapshots(
            snapshots=wall_snapshot_set_factory(
                wall_price=99.9,
                wall_size=2_000.0,
                normal_size=1.0,
            ),
            symbol="BTCUSDT",
            exchange="binance",
            current_mid_price=100.0,
            correlation_id="corr-emit-fail",
        )

    assert failing_bus.emit_attempts
    assert failing_bus.emit_attempts[0][0] in {
        spoofing_config.analyzer.event_topic_lifecycle,
        spoofing_config.analyzer.event_topic_updated,
        spoofing_config.analyzer.event_topic_detected,
        spoofing_config.analyzer.event_topic_score_updated,
    }


# =============================================================================
# Detector explosion / publish policy vulnerabilities
# =============================================================================


@pytest.mark.asyncio
async def test_process_snapshots_limits_detector_result_explosion_before_building_signal(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    wall_snapshot_set_factory,
    detector_result_factory,
    monkeypatch,
) -> None:
    """
    Вразливість: detector-и можуть повернути надто багато result-ів.
    Analyzer має обрізати їх через analyzer.max_detector_results_per_cycle
    до побудови signal/output.
    """

    spoofing_config.analyzer.max_detector_results_per_cycle = 3
    spoofing_config.scoring.detection_threshold = 0.01
    spoofing_config.validate()

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

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
        symbol="BTCUSDT",
        exchange="binance",
        current_mid_price=100.0,
        correlation_id="corr-detector-explosion",
    )

    assert len(output.detector_results) == spoofing_config.analyzer.max_detector_results_per_cycle

    scores = [item.score for item in output.detector_results]
    assert scores == sorted(scores, reverse=True)

    assert all(result.score >= 0.88 for result in output.detector_results)


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
    """
    Вразливість: publish_detected_only=True має реально не шуміти update events,
    навіть якщо detector_results є, але score не сформував signal.
    """

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
        symbol="BTCUSDT",
        exchange="binance",
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
async def test_process_snapshots_detector_exception_is_not_silently_converted_to_empty_result(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    wall_snapshot_set_factory,
    monkeypatch,
) -> None:
    """
    Вразливість: якщо base detector падає, analyzer зараз має впасти теж.
    Це краще, ніж silently no-signal, бо інакше production може приховати
    зламаний detector.
    """

    analyzer = SpoofingAnalyzer(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    def broken_base_detectors(**kwargs: Any) -> list[DetectorResult]:
        raise RuntimeError("base detector exploded")

    monkeypatch.setattr(analyzer, "_run_base_detectors", broken_base_detectors)

    with pytest.raises(RuntimeError, match="base detector exploded"):
        await analyzer.process_snapshots(
            snapshots=wall_snapshot_set_factory(),
            symbol="BTCUSDT",
            exchange="binance",
            current_mid_price=100.0,
        )

    assert mock_event_bus.emitted == []


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