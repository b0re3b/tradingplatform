# tests/analytics/liquidations/test_liquidation_pipeline.py

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable

import pytest

from core.event_bus import Event
from core.scheduler import Scheduler

from analytics.liquidations.cascade_detector import CascadeDetector
from analytics.liquidations.config import CascadeDetectorConfig, LiquidationStreamConfig
from analytics.liquidations.enums import LiquidationEventType, LiquidationSide
from analytics.liquidations.liquidation_stream import LiquidationStream
from analytics.liquidations.metrics import LiquidationMetrics
from analytics.liquidations.models import CascadeDetectionResult, LiquidationEvent
from analytics.liquidations.state import LiquidationState


pytestmark = pytest.mark.asyncio


# =============================================================================
# Test doubles
# =============================================================================

class FailingPipelineHistoryStore:
    """
    Навмисно токсичний history store.

    Його задача — перевірити, що storage failure не валить ingestion pipeline,
    не блокує normalized publish і не заважає detector-у побачити cascade.
    """

    def __init__(self, *, fail_append: bool = True, fail_flush: bool = False) -> None:
        self.fail_append = fail_append
        self.fail_flush = fail_flush
        self.appended_events: list[LiquidationEvent] = []
        self.appended_large_events: list[LiquidationEvent] = []
        self.flush_calls = 0

    async def append_event(self, event: LiquidationEvent) -> None:
        if self.fail_append:
            raise RuntimeError("pipeline history append failed")
        self.appended_events.append(event)

    async def append_large_event(self, event: LiquidationEvent) -> None:
        if self.fail_append:
            raise RuntimeError("pipeline history large append failed")
        self.appended_large_events.append(event)

    async def flush(self) -> None:
        self.flush_calls += 1
        if self.fail_flush:
            raise RuntimeError("pipeline history flush failed")


# =============================================================================
# Helpers
# =============================================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _scope_key(
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
) -> str:
    return f"{exchange}:{market_type}:{symbol}:{timeframe}"


def _make_cascade_raw_payloads(
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
    *,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    exchange_symbol: str | None = None,
    market_type: str = "usdm_futures",
    timeframe: str = "realtime",
    side: str = "SELL",
    base_price: Decimal = Decimal("65000"),
    quantity_1: Decimal = Decimal("1"),
    quantity_2: Decimal = Decimal("1.2"),
    quantity_3: Decimal = Decimal("1.5"),
    trade_prefix: str = "cascade",
    correlation_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """
    Створює 3 raw payload-и, які мають пройти CascadeDetector thresholds.

    Timestamps навмисно розставлені нерівномірно:
    - перша подія відносно давніша;
    - друга й третя ближче одна до одної;
    - acceleration має бути вищим за threshold.
    """
    now = _utc_now()

    correlation_prefix = correlation_prefix or trade_prefix

    return [
        make_raw_liquidation_payload(
            exchange=exchange,
            symbol=symbol,
            exchange_symbol=exchange_symbol or symbol,
            market_type=market_type,
            timeframe=timeframe,
            side=side,
            price=str(base_price),
            quantity=str(quantity_1),
            timestamp=now - timedelta(seconds=3),
            trade_id=f"{trade_prefix}-1",
            order_id=f"{trade_prefix}-order-1",
            extra={"pipeline_correlation_hint": f"{correlation_prefix}-1"},
        ),
        make_raw_liquidation_payload(
            exchange=exchange,
            symbol=symbol,
            exchange_symbol=exchange_symbol or symbol,
            market_type=market_type,
            timeframe=timeframe,
            side=side,
            price=str(base_price - Decimal("50")),
            quantity=str(quantity_2),
            timestamp=now - timedelta(seconds=1),
            trade_id=f"{trade_prefix}-2",
            order_id=f"{trade_prefix}-order-2",
            extra={"pipeline_correlation_hint": f"{correlation_prefix}-2"},
        ),
        make_raw_liquidation_payload(
            exchange=exchange,
            symbol=symbol,
            exchange_symbol=exchange_symbol or symbol,
            market_type=market_type,
            timeframe=timeframe,
            side=side,
            price=str(base_price - Decimal("100")),
            quantity=str(quantity_3),
            timestamp=now - timedelta(milliseconds=200),
            trade_id=f"{trade_prefix}-3",
            order_id=f"{trade_prefix}-order-3",
            extra={"pipeline_correlation_hint": f"{correlation_prefix}-3"},
        ),
    ]


async def _emit_raw_payloads(
    event_bus,
    topic: str,
    payloads: list[dict[str, Any]],
    *,
    correlation_id: str = "pipeline-correlation-id",
    source: str = "test.exchange_adapter",
) -> None:
    for index, payload in enumerate(payloads):
        accepted = await event_bus.emit(
            topic,
            payload,
            source=source,
            correlation_id=f"{correlation_id}-{index}",
        )
        assert accepted is True


async def _start_pipeline(
    *,
    event_bus,
    scheduler: Scheduler | None,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    state: LiquidationState,
    metrics: LiquidationMetrics,
    history_store: Any | None = None,
) -> tuple[LiquidationStream, CascadeDetector]:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=state,
        metrics=metrics,
        history_store=history_store,
    )
    detector = CascadeDetector(
        event_bus=event_bus,
        scheduler=scheduler,
        config=cascade_config,
        state=state,
        metrics=metrics,
    )

    # Порядок важливий:
    # detector має слухати normalized topic до того, як stream почне publishing.
    await detector.start()
    await stream.start()

    return stream, detector


# =============================================================================
# E2E happy path, але з повними production-перевірками
# =============================================================================

async def test_pipeline_raw_liquidations_flow_to_cascade_detection_end_to_end(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
    event_collector,
    wait_for_events,
) -> None:
    normalized_events = event_collector(event_bus, stream_config.publish_topic_normalized)
    updated_events = event_collector(event_bus, stream_config.publish_topic_updated)
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)

    stream, detector = await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    raw_payloads = _make_cascade_raw_payloads(
        make_raw_liquidation_payload,
        trade_prefix="e2e-happy",
        correlation_prefix="e2e-happy",
    )

    await _emit_raw_payloads(
        event_bus,
        stream_config.input_topic_raw,
        raw_payloads,
        correlation_id="e2e-happy-correlation",
    )

    await wait_for_events(normalized_events, expected_count=3)
    await wait_for_events(updated_events, expected_count=3)
    await wait_for_events(cascade_events, expected_count=1)

    assert len(normalized_events) == 3
    assert len(updated_events) == 3
    assert len(cascade_events) == 1

    cascade_event = cascade_events[0]
    assert cascade_event.topic == cascade_config.publish_topic_detected
    assert cascade_event.source == detector.service_name
    assert isinstance(cascade_event.payload, CascadeDetectionResult)

    result = cascade_event.payload
    assert result.exchange == "binance"
    assert result.market_type == "usdm_futures"
    assert result.symbol == "BTCUSDT"
    assert result.timeframe == "realtime"
    assert result.key == ("binance", "usdm_futures", "BTCUSDT", "realtime")
    assert result.event_count >= cascade_config.min_events
    assert result.total_notional_usd >= cascade_config.min_total_notional_usd
    assert result.cluster.key == result.key

    assert cascade_event.headers["event_type"] == LiquidationEventType.CASCADE.value
    assert cascade_event.headers["scope"] == _scope_key()
    assert cascade_event.headers["market_type"] == "usdm_futures"
    assert cascade_event.headers["timeframe"] == "realtime"

    symbol_state = liquidation_state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    assert symbol_state is not None
    assert symbol_state.total_buffered_events == 3
    assert symbol_state.cooldown_until is not None

    stream_stats = stream.get_stats()
    detector_stats = detector.get_stats()

    assert stream_stats["processed_messages"] == 3
    assert stream_stats["processed_events"] == 3
    assert stream_stats["published_normalized"] == 3
    assert stream_stats["published_updated"] == 3
    assert stream_stats["tracked_scopes"] == 1

    assert detector_stats["processed_events"] == 3
    assert detector_stats["cascade_signals_emitted"] == 1
    assert detector_stats["latest_signals_buffered"] == 1

    assert liquidation_metrics.total_valid_events == 3
    assert liquidation_metrics.total_cascades_detected == 1
    assert liquidation_metrics.scope_event_counts[_scope_key()] == 3
    assert liquidation_metrics.cascade_by_scope[_scope_key()] == 1


# =============================================================================
# Raw/normalized boundary protection
# =============================================================================

async def test_pipeline_raw_payload_never_reaches_detector_as_valid_analytics_event(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    raw_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)

    _, detector = await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    # Навмисно шлемо raw dict прямо в detector input topic.
    # Detector має відхилити це як invalid payload, а не створити signal.
    accepted = await event_bus.emit(
        cascade_config.input_topic,
        raw_liquidation_payload,
        source="malicious.test",
        correlation_id="raw-to-detector-attack",
    )

    await wait_for_events(cascade_events, timeout=0.1)

    assert accepted is True
    assert cascade_events == []

    detector_stats = detector.get_stats()
    assert detector_stats["invalid_payload_skips"] == 1
    assert detector_stats["processed_events"] == 0
    assert detector_stats["cascade_signals_emitted"] == 0


# =============================================================================
# Scope isolation: market_type / timeframe / exchange
# =============================================================================

async def test_pipeline_does_not_mix_same_symbol_across_market_types_into_false_cascade(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
    event_collector,
    wait_for_events,
) -> None:
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)

    stream, detector = await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    now = _utc_now()

    payloads = [
        make_raw_liquidation_payload(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            side="SELL",
            price="65000",
            quantity="1",
            timestamp=now - timedelta(seconds=3),
            trade_id="scope-usdm-1",
            order_id="scope-usdm-order-1",
        ),
        make_raw_liquidation_payload(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            side="SELL",
            price="64950",
            quantity="1",
            timestamp=now - timedelta(seconds=1),
            trade_id="scope-usdm-2",
            order_id="scope-usdm-order-2",
        ),
        make_raw_liquidation_payload(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSD_PERP",
            market_type="coinm_futures",
            timeframe="realtime",
            side="SELL",
            price="64900",
            quantity="1",
            timestamp=now - timedelta(milliseconds=200),
            trade_id="scope-coinm-1",
            order_id="scope-coinm-order-1",
        ),
    ]

    await _emit_raw_payloads(
        event_bus,
        stream_config.input_topic_raw,
        payloads,
        correlation_id="mixed-market-type",
    )
    await wait_for_events(cascade_events, timeout=0.15)

    assert cascade_events == []
    assert liquidation_state.scopes_count == 2

    usdm_state = liquidation_state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    coinm_state = liquidation_state.get(
        "binance",
        "BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
    )

    assert usdm_state is not None
    assert coinm_state is not None
    assert usdm_state.total_buffered_events == 2
    assert coinm_state.total_buffered_events == 1

    assert stream.get_stats()["processed_events"] == 3
    assert detector.get_stats()["cascade_signals_emitted"] == 0
    assert detector.get_stats()["threshold_skips"] >= 1


async def test_pipeline_does_not_mix_same_symbol_across_timeframes_into_false_cascade(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
    event_collector,
    wait_for_events,
) -> None:
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)

    await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    now = _utc_now()

    payloads = [
        make_raw_liquidation_payload(
            market_type="usdm_futures",
            timeframe="realtime",
            side="SELL",
            quantity="1",
            timestamp=now - timedelta(seconds=3),
            trade_id="tf-realtime-1",
            order_id="tf-realtime-order-1",
        ),
        make_raw_liquidation_payload(
            market_type="usdm_futures",
            timeframe="realtime",
            side="SELL",
            quantity="1",
            timestamp=now - timedelta(seconds=1),
            trade_id="tf-realtime-2",
            order_id="tf-realtime-order-2",
        ),
        make_raw_liquidation_payload(
            market_type="usdm_futures",
            timeframe="1m",
            side="SELL",
            quantity="1",
            timestamp=now - timedelta(milliseconds=200),
            trade_id="tf-1m-1",
            order_id="tf-1m-order-1",
        ),
    ]

    await _emit_raw_payloads(
        event_bus,
        stream_config.input_topic_raw,
        payloads,
        correlation_id="mixed-timeframe",
    )
    await wait_for_events(cascade_events, timeout=0.15)

    assert cascade_events == []
    assert liquidation_state.scopes_count == 2

    realtime_state = liquidation_state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    one_minute_state = liquidation_state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="1m",
    )

    assert realtime_state is not None
    assert one_minute_state is not None
    assert realtime_state.total_buffered_events == 2
    assert one_minute_state.total_buffered_events == 1


async def test_pipeline_interleaved_two_futures_scopes_create_two_independent_cascades(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
    event_collector,
    wait_for_events,
) -> None:
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)

    _, detector = await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    usdm_payloads = _make_cascade_raw_payloads(
        make_raw_liquidation_payload,
        exchange="binance",
        symbol="BTCUSDT",
        exchange_symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        trade_prefix="interleaved-usdm",
    )
    coinm_payloads = _make_cascade_raw_payloads(
        make_raw_liquidation_payload,
        exchange="binance",
        symbol="BTCUSDT",
        exchange_symbol="BTCUSD_PERP",
        market_type="coinm_futures",
        timeframe="realtime",
        trade_prefix="interleaved-coinm",
    )

    # Навмисно interleaved порядок, щоб ловити помилки state/window aggregation.
    interleaved_payloads = [
        usdm_payloads[0],
        coinm_payloads[0],
        usdm_payloads[1],
        coinm_payloads[1],
        usdm_payloads[2],
        coinm_payloads[2],
    ]

    await _emit_raw_payloads(
        event_bus,
        stream_config.input_topic_raw,
        interleaved_payloads,
        correlation_id="interleaved-scopes",
    )

    await wait_for_events(cascade_events, expected_count=2)

    assert len(cascade_events) == 2

    result_scopes = {
        event.payload.key
        for event in cascade_events
        if isinstance(event.payload, CascadeDetectionResult)
    }

    assert result_scopes == {
        ("binance", "usdm_futures", "BTCUSDT", "realtime"),
        ("binance", "coinm_futures", "BTCUSDT", "realtime"),
    }

    usdm_state = liquidation_state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    coinm_state = liquidation_state.get(
        "binance",
        "BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
    )

    assert usdm_state is not None
    assert coinm_state is not None
    assert usdm_state.cooldown_until is not None
    assert coinm_state.cooldown_until is not None

    assert liquidation_metrics.total_cascades_detected == 2
    assert liquidation_metrics.cascade_by_scope[
        _scope_key(market_type="usdm_futures")
    ] == 1
    assert liquidation_metrics.cascade_by_scope[
        _scope_key(market_type="coinm_futures")
    ] == 1

    assert detector.get_stats()["cooldown_skips"] == 0


# =============================================================================
# Deduplication / noise / poisoning
# =============================================================================

async def test_pipeline_duplicate_burst_does_not_create_false_cascade(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    raw_large_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    stream_config.deduplication_enabled = True

    normalized_events = event_collector(event_bus, stream_config.publish_topic_normalized)
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)

    stream, detector = await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    # Три однакові raw payload-и не мають перетворитися на 3 normalized events.
    for index in range(3):
        accepted = await event_bus.emit(
            stream_config.input_topic_raw,
            dict(raw_large_liquidation_payload),
            source="test.exchange_adapter",
            correlation_id=f"duplicate-burst-{index}",
        )
        assert accepted is True

    await wait_for_events(normalized_events, expected_count=1)
    await wait_for_events(cascade_events, timeout=0.15)

    assert len(normalized_events) == 1
    assert cascade_events == []

    symbol_state = liquidation_state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert symbol_state is not None
    assert symbol_state.total_buffered_events == 1

    stream_stats = stream.get_stats()
    detector_stats = detector.get_stats()

    assert stream_stats["processed_messages"] == 3
    assert stream_stats["processed_events"] == 1
    assert stream_stats["dropped_duplicates"] == 2
    assert detector_stats["cascade_signals_emitted"] == 0


async def test_pipeline_invalid_and_stale_noise_do_not_poison_state_or_detector(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    raw_invalid_liquidation_payload: dict[str, Any],
    raw_stale_liquidation_payload: dict[str, Any],
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
    event_collector,
    wait_for_events,
) -> None:
    normalized_events = event_collector(event_bus, stream_config.publish_topic_normalized)
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)

    stream, detector = await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    valid_payload_1 = make_raw_liquidation_payload(
        trade_id="noise-valid-1",
        order_id="noise-valid-order-1",
        quantity="1",
    )
    valid_payload_2 = make_raw_liquidation_payload(
        trade_id="noise-valid-2",
        order_id="noise-valid-order-2",
        quantity="1",
    )

    payloads = [
        raw_invalid_liquidation_payload,
        raw_stale_liquidation_payload,
        valid_payload_1,
        valid_payload_2,
    ]

    await _emit_raw_payloads(
        event_bus,
        stream_config.input_topic_raw,
        payloads,
        correlation_id="invalid-stale-noise",
    )

    await wait_for_events(normalized_events, expected_count=2)
    await wait_for_events(cascade_events, timeout=0.15)

    assert len(normalized_events) == 2
    assert cascade_events == []

    state = liquidation_state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    assert state is not None
    assert state.total_buffered_events == 2

    stream_stats = stream.get_stats()
    detector_stats = detector.get_stats()

    assert stream_stats["processed_messages"] == 4
    assert stream_stats["processed_events"] == 2
    assert stream_stats["dropped_invalid"] == 1
    assert stream_stats["dropped_stale"] == 1
    assert detector_stats["cascade_signals_emitted"] == 0

    assert liquidation_metrics.total_invalid_events == 1
    assert liquidation_metrics.total_stale_events == 1
    assert liquidation_metrics.total_valid_events == 2


async def test_pipeline_out_of_scope_payload_is_filtered_before_detector_sees_it(
    event_bus,
    scheduler: Scheduler,
    strict_btc_usdm_stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    bybit_linear_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    normalized_events = event_collector(
        event_bus,
        strict_btc_usdm_stream_config.publish_topic_normalized,
    )
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)

    stream, detector = await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=strict_btc_usdm_stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    accepted = await event_bus.emit(
        strict_btc_usdm_stream_config.input_topic_raw,
        bybit_linear_liquidation_payload,
        source="test.exchange_adapter",
        correlation_id="out-of-scope-bybit-linear",
    )

    await wait_for_events(normalized_events, timeout=0.1)
    await wait_for_events(cascade_events, timeout=0.1)

    assert accepted is True
    assert normalized_events == []
    assert cascade_events == []
    assert liquidation_state.scopes_count == 0

    assert stream.get_stats()["processed_messages"] == 1
    assert stream.get_stats()["filtered_scope"] == 1
    assert stream.get_stats()["processed_events"] == 0
    assert detector.get_stats()["processed_events"] == 0


# =============================================================================
# Correlation propagation / event metadata
# =============================================================================

async def test_pipeline_preserves_final_trigger_correlation_id_to_cascade_result(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
    event_collector,
    wait_for_events,
) -> None:
    normalized_events = event_collector(event_bus, stream_config.publish_topic_normalized)
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)

    await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    payloads = _make_cascade_raw_payloads(
        make_raw_liquidation_payload,
        trade_prefix="correlation",
    )

    for index, payload in enumerate(payloads):
        accepted = await event_bus.emit(
            stream_config.input_topic_raw,
            payload,
            source="test.exchange_adapter",
            correlation_id=f"raw-correlation-{index}",
        )
        assert accepted is True

    await wait_for_events(normalized_events, expected_count=3)
    await wait_for_events(cascade_events, expected_count=1)

    assert len(normalized_events) == 3
    assert len(cascade_events) == 1

    for index, normalized in enumerate(normalized_events):
        assert isinstance(normalized.payload, LiquidationEvent)
        assert normalized.payload.correlation_id == f"raw-correlation-{index}"
        assert normalized.correlation_id == f"raw-correlation-{index}"
        assert normalized.payload.metadata["source_topic"] == stream_config.input_topic_raw

    cascade_result = cascade_events[0].payload
    assert isinstance(cascade_result, CascadeDetectionResult)

    # Cascade має успадкувати correlation саме trigger-event-а.
    assert cascade_result.correlation_id == "raw-correlation-2"
    assert cascade_events[0].correlation_id == "raw-correlation-2"


# =============================================================================
# Cooldown behavior through full pipeline
# =============================================================================

async def test_pipeline_scope_cooldown_blocks_repeated_cascade_but_not_other_scope(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
    event_collector,
    wait_for_events,
) -> None:
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)

    stream, detector = await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    usdm_payloads = _make_cascade_raw_payloads(
        make_raw_liquidation_payload,
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        trade_prefix="cooldown-usdm",
    )
    repeated_usdm_payload = make_raw_liquidation_payload(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        side="SELL",
        price="64800",
        quantity="2",
        trade_id="cooldown-usdm-repeat",
        order_id="cooldown-usdm-repeat-order",
    )

    coinm_payloads = _make_cascade_raw_payloads(
        make_raw_liquidation_payload,
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        trade_prefix="cooldown-coinm",
    )

    await _emit_raw_payloads(
        event_bus,
        stream_config.input_topic_raw,
        usdm_payloads,
        correlation_id="cooldown-usdm",
    )
    await wait_for_events(cascade_events, expected_count=1)

    accepted = await event_bus.emit(
        stream_config.input_topic_raw,
        repeated_usdm_payload,
        source="test.exchange_adapter",
        correlation_id="cooldown-repeat-same-scope",
    )
    assert accepted is True

    await wait_for_events(cascade_events, expected_count=1, timeout=0.15)

    # Та сама scope у cooldown — новий cascade не має з'явитися.
    assert len(cascade_events) == 1

    await _emit_raw_payloads(
        event_bus,
        stream_config.input_topic_raw,
        coinm_payloads,
        correlation_id="cooldown-coinm",
    )
    await wait_for_events(cascade_events, expected_count=2)

    assert len(cascade_events) == 2

    scopes = {
        event.payload.key
        for event in cascade_events
        if isinstance(event.payload, CascadeDetectionResult)
    }

    assert scopes == {
        ("binance", "usdm_futures", "BTCUSDT", "realtime"),
        ("binance", "coinm_futures", "BTCUSDT", "realtime"),
    }

    assert detector.get_stats()["cascade_signals_emitted"] == 2
    assert detector.get_stats()["cooldown_skips"] >= 1
    assert stream.get_stats()["processed_events"] == 7


# =============================================================================
# Storage failure must not break analytics pipeline
# =============================================================================

async def test_pipeline_history_store_failure_does_not_block_cascade_detection(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
    event_collector,
    wait_for_events,
) -> None:
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)
    history_store = FailingPipelineHistoryStore(fail_append=True)

    stream, detector = await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
        history_store=history_store,
    )

    raw_payloads = _make_cascade_raw_payloads(
        make_raw_liquidation_payload,
        trade_prefix="history-failure",
    )

    await _emit_raw_payloads(
        event_bus,
        stream_config.input_topic_raw,
        raw_payloads,
        correlation_id="history-failure",
    )

    await wait_for_events(cascade_events, expected_count=1)

    assert len(cascade_events) == 1
    assert isinstance(cascade_events[0].payload, CascadeDetectionResult)

    assert stream.get_stats()["processed_events"] == 3
    assert stream.get_stats()["storage_errors"] >= 1
    assert stream.get_stats()["published_normalized"] == 3
    assert detector.get_stats()["cascade_signals_emitted"] == 1
    assert liquidation_metrics.total_cascades_detected == 1


# =============================================================================
# Lifecycle boundary: closed stream must not ingest, detector remains safe
# =============================================================================

async def test_pipeline_closed_stream_does_not_ingest_new_raw_events(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
    event_collector,
    wait_for_events,
) -> None:
    normalized_events = event_collector(event_bus, stream_config.publish_topic_normalized)
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)

    stream, detector = await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    await stream.close()

    assert stream.is_closed is True
    assert stream.is_running is False

    raw_payloads = _make_cascade_raw_payloads(
        make_raw_liquidation_payload,
        trade_prefix="closed-stream",
    )

    await _emit_raw_payloads(
        event_bus,
        stream_config.input_topic_raw,
        raw_payloads,
        correlation_id="closed-stream",
    )

    await wait_for_events(normalized_events, timeout=0.15)
    await wait_for_events(cascade_events, timeout=0.15)

    assert normalized_events == []
    assert cascade_events == []
    assert liquidation_state.scopes_count == 0
    assert stream.get_stats()["processed_messages"] == 0
    assert detector.get_stats()["processed_events"] == 0


# =============================================================================
# Large events must not imply cascade by themselves
# =============================================================================

async def test_pipeline_large_events_are_published_but_do_not_bypass_cascade_thresholds(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    raw_large_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    large_events = event_collector(event_bus, stream_config.publish_topic_large)
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)

    stream, detector = await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    accepted = await event_bus.emit(
        stream_config.input_topic_raw,
        raw_large_liquidation_payload,
        source="test.exchange_adapter",
        correlation_id="single-large-event",
    )
    assert accepted is True

    await wait_for_events(large_events, expected_count=1)
    await wait_for_events(cascade_events, timeout=0.15)

    assert len(large_events) == 1
    assert cascade_events == []

    assert isinstance(large_events[0].payload, LiquidationEvent)
    assert large_events[0].payload.notional_usd >= stream_config.large_liquidation_threshold_usd

    assert stream.get_stats()["published_large"] == 1
    assert detector.get_stats()["cascade_signals_emitted"] == 0
    assert detector.get_stats()["threshold_skips"] >= 1


# =============================================================================
# Backpressure-like burst with valid, duplicate, stale, invalid and scoped noise
# =============================================================================

async def test_pipeline_mixed_burst_produces_exactly_one_clean_cascade(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    cascade_config: CascadeDetectorConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
    raw_invalid_liquidation_payload: dict[str, Any],
    raw_stale_liquidation_payload: dict[str, Any],
    bybit_linear_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    normalized_events = event_collector(event_bus, stream_config.publish_topic_normalized)
    cascade_events = event_collector(event_bus, cascade_config.publish_topic_detected)

    stream, detector = await _start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    valid_cascade_payloads = _make_cascade_raw_payloads(
        make_raw_liquidation_payload,
        trade_prefix="mixed-burst-clean",
    )

    duplicate_payload = dict(valid_cascade_payloads[0])

    out_of_scope_payload = dict(bybit_linear_liquidation_payload)
    out_of_scope_payload["symbol"] = "DOGEUSDT"

    payloads = [
        raw_invalid_liquidation_payload,
        valid_cascade_payloads[0],
        duplicate_payload,
        raw_stale_liquidation_payload,
        valid_cascade_payloads[1],
        out_of_scope_payload,
        valid_cascade_payloads[2],
    ]

    await _emit_raw_payloads(
        event_bus,
        stream_config.input_topic_raw,
        payloads,
        correlation_id="mixed-burst",
    )

    await wait_for_events(normalized_events, expected_count=3)
    await wait_for_events(cascade_events, expected_count=1)

    assert len(normalized_events) == 3
    assert len(cascade_events) == 1

    result = cascade_events[0].payload
    assert isinstance(result, CascadeDetectionResult)
    assert result.key == ("binance", "usdm_futures", "BTCUSDT", "realtime")

    stream_stats = stream.get_stats()
    detector_stats = detector.get_stats()

    assert stream_stats["processed_messages"] == 7
    assert stream_stats["processed_events"] == 3
    assert stream_stats["dropped_invalid"] == 1
    assert stream_stats["dropped_stale"] == 1
    assert stream_stats["dropped_duplicates"] == 1
    assert stream_stats["filtered_scope"] == 1
    assert stream_stats["published_normalized"] == 3

    assert detector_stats["processed_events"] == 3
    assert detector_stats["cascade_signals_emitted"] == 1

    assert liquidation_metrics.total_valid_events == 3
    assert liquidation_metrics.total_invalid_events == 1
    assert liquidation_metrics.total_stale_events == 1
    assert liquidation_metrics.total_cascades_detected == 1