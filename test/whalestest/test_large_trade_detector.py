from __future__ import annotations

import asyncio
import math
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from analytics.whales.config import LargeTradeDetectorConfig
from analytics.whales.large_trade_detector import LargeTradeDetector


pytestmark = pytest.mark.asyncio


# =============================================================================
# Topics
# =============================================================================

TRADES_UPDATED_TOPIC = "market.trades.updated"
RAW_TRADE_TOPIC = "market.trade"
LARGE_TRADE_TOPIC = "analytics.whales.large_trade"


# =============================================================================
# Local helpers
# =============================================================================

def _build_detector(
    *,
    config: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> LargeTradeDetector:
    return LargeTradeDetector(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )


async def _feed_baseline_trades(
    detector: LargeTradeDetector,
    raw_trade_payload_factory,
    *,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    timeframe: str = "realtime",
    count: int = 5,
    notional: float = 10_000.0,
    side: str = "buy",
    start_ts_ms: int | None = None,
) -> None:
    base_ts = start_ts_ms if start_ts_ms is not None else int(time.time() * 1000)

    for index in range(count):
        signal = await detector.process_trade_payload(
            raw_trade_payload_factory(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                price=100.0,
                quantity=notional / 100.0,
                side=side,
                timestamp_ms=base_ts + index,
                trade_id=f"baseline-{exchange}-{market_type}-{symbol}-{index}",
            ),
            source_topic="manual.test.single_trade",
            allow_raw_payload=True,
        )
        assert signal is None


def _assert_no_nan_or_inf(value: float) -> None:
    assert not math.isnan(value)
    assert not math.isinf(value)


def _signal_symbols(signals: list[Any]) -> list[str]:
    return [signal.symbol for signal in signals if signal is not None]


def _signal_trade_ids(signals: list[Any]) -> list[str | None]:
    return [getattr(signal, "trade_id", None) for signal in signals if signal is not None]


def _make_mixed_batch(
    raw_trade_payload_factory,
    *,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    timeframe: str = "realtime",
    base_ts: int | None = None,
) -> list[dict[str, Any]]:
    ts = base_ts if base_ts is not None else int(time.time() * 1000)

    return [
        raw_trade_payload_factory(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            price=100.0,
            quantity=900.0,
            side="buy",
            timestamp_ms=ts,
            trade_id="valid-large-1",
        ),
        raw_trade_payload_factory(
            symbol="",
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            price=100.0,
            quantity=900.0,
            side="buy",
            timestamp_ms=ts + 1,
            trade_id="invalid-empty-symbol",
        ),
        raw_trade_payload_factory(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            price=100.0,
            quantity=0.01,
            side="buy",
            timestamp_ms=ts + 2,
            trade_id="tiny-noise",
        ),
        raw_trade_payload_factory(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            price=100.0,
            quantity=950.0,
            side="sell",
            timestamp_ms=ts + 3,
            trade_id="valid-large-2",
        ),
        raw_trade_payload_factory(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            price="bad",
            quantity=1_000.0,
            side="buy",
            timestamp_ms=ts + 4,
            trade_id="invalid-price",
        ),
    ]


# =============================================================================
# Lifecycle / core architecture
# =============================================================================

async def test_register_is_idempotent_and_subscribes_only_to_production_topic(
    large_trade_detector: LargeTradeDetector,
) -> None:
    await large_trade_detector.register()
    await large_trade_detector.register()
    await large_trade_detector.register()

    assert large_trade_detector.is_registered is True
    assert large_trade_detector.is_started is False

    patterns = {subscription.pattern for subscription in large_trade_detector.subscriptions}

    assert patterns == {TRADES_UPDATED_TOPIC}
    assert RAW_TRADE_TOPIC not in patterns
    assert len(large_trade_detector.subscriptions) == 1


async def test_legacy_register_subscribes_to_production_and_raw_topics(
    large_trade_detector_legacy: LargeTradeDetector,
) -> None:
    await large_trade_detector_legacy.register()
    await large_trade_detector_legacy.register()

    assert large_trade_detector_legacy.is_registered is True

    patterns = {
        subscription.pattern
        for subscription in large_trade_detector_legacy.subscriptions
    }

    assert TRADES_UPDATED_TOPIC in patterns
    assert RAW_TRADE_TOPIC in patterns
    assert len(patterns) == 2


async def test_start_registers_subscription_and_cleanup_job_once(
    large_trade_detector: LargeTradeDetector,
) -> None:
    await large_trade_detector.start()
    await large_trade_detector.start()
    await large_trade_detector.start()

    assert large_trade_detector.is_started is True
    assert large_trade_detector.is_registered is True

    assert len(large_trade_detector.subscriptions) == 1
    assert len(large_trade_detector.scheduler_job_ids) == 1

    health = large_trade_detector.get_healthcheck()
    assert health["started"] is True
    assert health["registered"] is True
    assert health["scheduler_jobs"] == 1
    assert health["scope"] == "exchange:market_type:symbol:timeframe"


async def test_stop_unsubscribes_and_removes_scheduler_jobs(
    large_trade_detector: LargeTradeDetector,
) -> None:
    await large_trade_detector.start()

    assert large_trade_detector.is_started is True
    assert len(large_trade_detector.subscriptions) == 1
    assert len(large_trade_detector.scheduler_job_ids) == 1

    await large_trade_detector.stop()
    await large_trade_detector.stop()

    assert large_trade_detector.is_started is False
    assert large_trade_detector.is_registered is False
    assert len(large_trade_detector.subscriptions) == 0
    assert len(large_trade_detector.scheduler_job_ids) == 0


async def test_disabled_detector_does_not_register_start_process_or_emit(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    trades_updated_payload_factory,
) -> None:
    config = replace(large_trade_detector_config_fast, enabled=False)
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    await detector.register()
    await detector.start()

    assert detector.is_registered is False
    assert detector.is_started is False
    assert len(detector.subscriptions) == 0
    assert len(detector.scheduler_job_ids) == 0

    signals = await detector.process_trades_payload(
        trades_updated_payload_factory(count=2, notional=100_000.0),
        source_topic=TRADES_UPDATED_TOPIC,
    )
    single_signal = await detector.process_trade_payload(
        trades_updated_payload_factory(count=1, notional=100_000.0),
        source_topic=TRADES_UPDATED_TOPIC,
    )

    assert signals == []
    assert single_signal is None
    assert detector.get_all_stats() == {}

    await asyncio.sleep(0.05)
    assert event_collector.by_topic(LARGE_TRADE_TOPIC) == []


# =============================================================================
# Production batch processing
# =============================================================================

async def test_process_trades_payload_processes_all_valid_large_trades_in_batch(
    large_trade_detector: LargeTradeDetector,
    trades_updated_payload_factory,
    raw_trade_payload_factory,
) -> None:
    batch = _make_mixed_batch(raw_trade_payload_factory)

    signals = await large_trade_detector.process_trades_payload(
        trades_updated_payload_factory(
            trades=batch,
            symbol="BTCUSDT",
            batch_id="mixed-batch",
        ),
        source_topic=TRADES_UPDATED_TOPIC,
    )

    assert len(signals) == 2
    assert _signal_trade_ids(signals) == ["valid-large-1", "valid-large-2"]
    assert _signal_symbols(signals) == ["BTCUSDT", "BTCUSDT"]

    stats = large_trade_detector.get_symbol_stats("BTCUSDT")

    assert stats["exists"] is True
    assert stats["trades_processed"] == 3
    assert stats["signals_emitted"] == 2

    # Один tiny trade пройшов normalization/basic state path, але не став signal.
    assert stats["last_notional"] > 0


async def test_process_trades_payload_preserves_batch_order_for_signals(
    large_trade_detector: LargeTradeDetector,
    trades_updated_payload_factory,
    raw_trade_payload_factory,
) -> None:
    trades = [
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=700.0,
            side="buy",
            trade_id="order-1",
        ),
        raw_trade_payload_factory(
            symbol="ETHUSDT",
            price=100.0,
            quantity=300.0,
            side="sell",
            trade_id="order-2",
        ),
        raw_trade_payload_factory(
            symbol="SOLUSDT",
            price=100.0,
            quantity=200.0,
            side="buy",
            trade_id="order-3",
        ),
    ]

    signals = await large_trade_detector.process_trades_payload(
        trades_updated_payload_factory(
            trades=trades,
            symbol="BTCUSDT",
            batch_id="ordered-batch",
        ),
        source_topic=TRADES_UPDATED_TOPIC,
    )

    assert _signal_trade_ids(signals) == ["order-1", "order-2", "order-3"]
    assert _signal_symbols(signals) == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


async def test_process_trades_payload_accepts_nested_data_batch(
    large_trade_detector: LargeTradeDetector,
    trades_updated_payload_factory,
) -> None:
    signals = await large_trade_detector.process_trades_payload(
        trades_updated_payload_factory(
            symbol="BTCUSDT",
            count=2,
            notional=100_000.0,
            nested_data=True,
            batch_id="nested-batch",
        ),
        source_topic=TRADES_UPDATED_TOPIC,
    )

    assert len(signals) == 2
    assert all(signal.symbol == "BTCUSDT" for signal in signals)
    assert all(signal.notional == 100_000.0 for signal in signals)


async def test_process_trades_payload_supports_data_list_shape(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
) -> None:
    payload = {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
        "data": [
            raw_trade_payload_factory(
                symbol="BTCUSDT",
                price=100.0,
                quantity=700.0,
                side="buy",
                trade_id="data-list-1",
            ),
            raw_trade_payload_factory(
                symbol="BTCUSDT",
                price=100.0,
                quantity=800.0,
                side="sell",
                trade_id="data-list-2",
            ),
        ],
    }

    signals = await large_trade_detector.process_trades_payload(
        payload,
        source_topic=TRADES_UPDATED_TOPIC,
    )

    assert len(signals) == 2
    assert _signal_trade_ids(signals) == ["data-list-1", "data-list-2"]


async def test_process_trades_payload_supports_single_trade_shape(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
) -> None:
    payload = {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
        "trade": raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=700.0,
            side="buy",
            trade_id="single-trade-shape",
        ),
    }

    signals = await large_trade_detector.process_trades_payload(
        payload,
        source_topic=TRADES_UPDATED_TOPIC,
    )

    assert len(signals) == 1
    assert signals[0].trade_id == "single-trade-shape"


async def test_process_trade_payload_backward_compatible_returns_first_signal_from_batch(
    large_trade_detector: LargeTradeDetector,
    trades_updated_payload_factory,
    raw_trade_payload_factory,
) -> None:
    trades = [
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=700.0,
            side="buy",
            trade_id="first-signal",
        ),
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=800.0,
            side="sell",
            trade_id="second-signal",
        ),
    ]

    signal = await large_trade_detector.process_trade_payload(
        trades_updated_payload_factory(trades=trades, batch_id="compat-batch"),
        source_topic=TRADES_UPDATED_TOPIC,
    )

    assert signal is not None
    assert signal.trade_id == "first-signal"

    stats = large_trade_detector.get_symbol_stats("BTCUSDT")
    assert stats["trades_processed"] == 2
    assert stats["signals_emitted"] == 2


async def test_batch_with_only_invalid_or_tiny_trades_does_not_create_large_signals(
    large_trade_detector: LargeTradeDetector,
    trades_updated_payload_factory,
    raw_trade_payload_factory,
) -> None:
    trades = [
        raw_trade_payload_factory(
            symbol="",
            price=100.0,
            quantity=1_000.0,
            side="buy",
            trade_id="invalid-symbol",
        ),
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=0.001,
            side="buy",
            trade_id="tiny-1",
        ),
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price="bad",
            quantity=1_000.0,
            side="buy",
            trade_id="bad-price",
        ),
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=1_000.0,
            side="teleport",
            trade_id="bad-side",
        ),
    ]

    signals = await large_trade_detector.process_trades_payload(
        trades_updated_payload_factory(trades=trades, batch_id="invalid-batch"),
        source_topic=TRADES_UPDATED_TOPIC,
    )

    assert signals == []

    stats = large_trade_detector.get_symbol_stats("BTCUSDT")
    assert stats["exists"] is False or stats["signals_emitted"] == 0


# =============================================================================
# EventBus production / legacy topic behavior
# =============================================================================

async def test_eventbus_market_trades_updated_batch_emits_multiple_large_trade_events(
    large_trade_detector: LargeTradeDetector,
    event_bus: EventBus,
    event_collector,
    trades_updated_payload_factory,
) -> None:
    await large_trade_detector.start()

    accepted = await event_bus.emit(
        TRADES_UPDATED_TOPIC,
        trades_updated_payload_factory(
            symbol="BTCUSDT",
            count=3,
            notional=90_000.0,
            batch_id="eventbus-batch",
        ),
        source="tests.trades_cache",
        correlation_id="corr-eventbus-batch",
    )

    assert accepted is True

    await event_collector.wait_for_topic(
        LARGE_TRADE_TOPIC,
        count=3,
        timeout=1.0,
    )

    events = event_collector.by_topic(LARGE_TRADE_TOPIC)

    assert len(events) == 3
    assert all(event.correlation_id == "corr-eventbus-batch" for event in events)
    assert {event.payload["symbol"] for event in events} == {"BTCUSDT"}
    assert [event.payload["trade_id"] for event in events] == [
        "eventbus-batch-0",
        "eventbus-batch-1",
        "eventbus-batch-2",
    ]


async def test_raw_market_trade_is_ignored_when_legacy_topics_disabled(
    large_trade_detector: LargeTradeDetector,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    await large_trade_detector.start()

    accepted = await event_bus.emit(
        RAW_TRADE_TOPIC,
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=1_000.0,
            side="buy",
            trade_id="raw-ignored",
        ),
        source="tests.market_stream",
        correlation_id="corr-raw-disabled",
    )

    assert accepted is True

    await asyncio.sleep(0.05)

    assert event_collector.by_topic(LARGE_TRADE_TOPIC) == []
    assert large_trade_detector.get_all_stats() == {}


async def test_raw_market_trade_is_processed_when_legacy_topics_enabled(
    large_trade_detector_legacy: LargeTradeDetector,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    await large_trade_detector_legacy.start()

    accepted = await event_bus.emit(
        RAW_TRADE_TOPIC,
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=1_000.0,
            side="buy",
            trade_id="raw-accepted",
        ),
        source="tests.market_stream",
        correlation_id="corr-raw-enabled",
    )

    assert accepted is True

    await event_collector.wait_for_topic(
        LARGE_TRADE_TOPIC,
        count=1,
        timeout=1.0,
    )

    events = event_collector.by_topic(LARGE_TRADE_TOPIC)
    assert len(events) == 1
    assert events[0].payload["trade_id"] == "raw-accepted"
    assert events[0].correlation_id == "corr-raw-enabled"


async def test_handle_trade_event_never_raises_on_malformed_event_payloads(
    large_trade_detector: LargeTradeDetector,
) -> None:
    malformed_payloads: list[Any] = [
        None,
        [],
        "not-a-dict",
        {"data": None},
        {"data": []},
        {"data": {"symbol": "", "price": "bad", "quantity": object()}},
        {"trades": None},
        {"trades": []},
        {"trades": [None, [], "bad", {"symbol": ""}]},
    ]

    for payload in malformed_payloads:
        event = Event(
            topic=TRADES_UPDATED_TOPIC,
            payload=payload,
            source="test",
        )
        await large_trade_detector.handle_trade_event(event)

    assert large_trade_detector.get_all_stats() == {}


async def test_handle_raw_trade_event_respects_legacy_guard_even_if_called_directly(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
) -> None:
    event = Event(
        topic=RAW_TRADE_TOPIC,
        payload=raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=1_000.0,
            side="buy",
            trade_id="direct-raw-handler-disabled",
        ),
        source="test",
    )

    await large_trade_detector.handle_raw_trade_event(event)

    assert large_trade_detector.get_all_stats() == {}


# =============================================================================
# Normalization / hostile payloads
# =============================================================================

@pytest.mark.parametrize(
    "payload_patch",
    [
        {"symbol": None},
        {"symbol": ""},
        {"symbol": "   "},
        {"price": None},
        {"price": "not-a-number"},
        {"price": float("nan")},
        {"price": 0},
        {"price": -1},
        {"quantity": None},
        {"quantity": "not-a-number"},
        {"quantity": float("nan")},
        {"quantity": 0},
        {"quantity": -1},
        {"side": "teleport"},
        {"side": None, "m": None},
    ],
)
async def test_invalid_trade_payloads_are_rejected_without_state_mutation(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
    payload_patch: dict[str, Any],
) -> None:
    payload = raw_trade_payload_factory(
        price=100_000.0,
        quantity=2.0,
        side="buy",
        extra=payload_patch,
    )

    signal = await large_trade_detector.process_trade_payload(
        payload,
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert signal is None
    assert large_trade_detector.get_all_stats() == {}


async def test_nested_data_payload_is_accepted_and_normalized(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
) -> None:
    signal = await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="btcusdt",
            price=50_000.0,
            quantity=2.0,
            side="BUY",
            nested=True,
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert signal is not None
    assert signal.symbol == "BTCUSDT"
    assert signal.exchange == "binance"
    assert signal.market_type == "usdm_futures"
    assert signal.timeframe == "realtime"
    assert signal.side == "buy"
    assert signal.notional == 100_000.0
    assert signal.trigger_type in {"absolute", "absolute_and_relative"}


@pytest.mark.parametrize(
    ("maker_flag", "expected_side"),
    [
        (True, "sell"),
        (False, "buy"),
        ("true", "sell"),
        ("false", "buy"),
        ("1", "sell"),
        ("0", "buy"),
    ],
)
async def test_maker_flag_fallback_is_used_when_side_is_unknown(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
    maker_flag: Any,
    expected_side: str,
) -> None:
    signal = await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            price=50_000.0,
            quantity=2.0,
            side="unknown-side-value",
            maker_flag=maker_flag,
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert signal is not None
    assert signal.side == expected_side


@pytest.mark.parametrize(
    "timestamp_value",
    [
        1_700_000_000,
        1_700_000_000_000,
        "1700000000",
        "1700000000000",
        "2024-01-01T00:00:00Z",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        "not-a-real-timestamp",
    ],
)
async def test_timestamp_parser_accepts_common_exchange_formats_without_crashing(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
    timestamp_value: Any,
) -> None:
    signal = await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            price=50_000.0,
            quantity=2.0,
            side="buy",
            extra={"timestamp_ms": timestamp_value},
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert signal is not None
    assert isinstance(signal.timestamp_ms, int)
    assert signal.timestamp_ms > 0


async def test_filtering_happens_before_state_creation_for_tiny_noise_trades(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        min_notional_filter=10_000.0,
        default_abs_notional_threshold=50_000.0,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    signal = await detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=1.0,
            side="buy",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert signal is None
    assert detector.get_symbol_stats("BTCUSDT") == {
        "symbol": "BTCUSDT",
        "exists": False,
    }


async def test_side_filter_rejects_opposite_side_before_state_creation(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        side_filter="buy",
        min_notional_filter=1.0,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    signal = await detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=50_000.0,
            quantity=2.0,
            side="sell",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert signal is None
    assert detector.get_symbol_stats("BTCUSDT")["exists"] is False


# =============================================================================
# Absolute / relative detection
# =============================================================================

async def test_absolute_threshold_emits_signal_with_expected_payload_fields(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
    assert_payload_has_common_signal_fields,
    assert_symbol_payload,
) -> None:
    signal = await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=50_000.0,
            quantity=2.0,
            side="buy",
            trade_id="abs-1",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert signal is not None
    assert signal.symbol == "BTCUSDT"
    assert signal.exchange == "binance"
    assert signal.market_type == "usdm_futures"
    assert signal.timeframe == "realtime"
    assert signal.side == "buy"
    assert signal.price == 50_000.0
    assert signal.quantity == 2.0
    assert signal.notional == 100_000.0
    assert signal.abs_threshold == 50_000.0
    assert signal.trade_id == "abs-1"
    assert signal.trigger_type in {"absolute", "absolute_and_relative"}

    payload = signal.to_payload()
    assert_payload_has_common_signal_fields(payload)
    assert_symbol_payload(payload, "BTCUSDT")
    assert payload["event_type"] == "large_trade"
    assert payload["detector"] == "LargeTradeDetector"


async def test_symbol_specific_threshold_overrides_default_threshold(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
) -> None:
    eth_signal = await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="ETHUSDT",
            price=2_000.0,
            quantity=15.0,
            side="buy",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    btc_signal = await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=2_000.0,
            quantity=15.0,
            side="buy",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert eth_signal is not None
    assert eth_signal.notional == 30_000.0
    assert eth_signal.abs_threshold == 25_000.0

    assert btc_signal is None


async def test_scoped_threshold_overrides_symbol_and_default_threshold(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        default_abs_notional_threshold=1_000_000.0,
        symbol_abs_thresholds={"BTCUSDT": 500_000.0},
        scoped_abs_thresholds={
            "binance:usdm_futures:BTCUSDT:realtime": 25_000.0,
        },
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    signal = await detector.process_trade_payload(
        raw_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="realtime",
            price=100.0,
            quantity=300.0,
            side="buy",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert signal is not None
    assert signal.notional == 30_000.0
    assert signal.abs_threshold == 25_000.0


async def test_no_signal_when_absolute_and_relative_thresholds_are_not_met(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        default_abs_notional_threshold=1_000_000.0,
        use_relative_detection=True,
        min_samples_for_relative_detection=5,
        zscore_threshold=10.0,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    await _feed_baseline_trades(
        detector,
        raw_trade_payload_factory,
        count=5,
        notional=10_000.0,
    )

    signal = await detector.process_trade_payload(
        raw_trade_payload_factory(
            price=100.0,
            quantity=120.0,
            side="buy",
            trade_id="not-large",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert signal is None

    stats = detector.get_symbol_stats("BTCUSDT")
    assert stats["exists"] is True
    assert stats["trades_processed"] == 6
    assert stats["signals_emitted"] == 0


async def test_relative_zscore_detection_emits_without_absolute_trigger(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        default_abs_notional_threshold=10_000_000.0,
        use_relative_detection=True,
        min_samples_for_relative_detection=3,
        zscore_threshold=2.0,
        min_notional_filter=1.0,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    for index, notional in enumerate([10_000.0, 10_200.0, 9_800.0]):
        signal = await detector.process_trade_payload(
            raw_trade_payload_factory(
                price=100.0,
                quantity=notional / 100.0,
                side="buy",
                trade_id=f"relative-baseline-{index}",
            ),
            source_topic="manual.test.single_trade",
            allow_raw_payload=True,
        )
        assert signal is None

    signal = await detector.process_trade_payload(
        raw_trade_payload_factory(
            price=100.0,
            quantity=100_000.0 / 100.0,
            side="buy",
            trade_id="relative-spike",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert signal is not None
    assert signal.trigger_type == "relative"
    assert signal.abs_threshold == 10_000_000.0
    assert signal.notional == 100_000.0
    assert signal.zscore >= config.zscore_threshold
    _assert_no_nan_or_inf(signal.zscore)


async def test_absolute_and_relative_trigger_combined_when_both_conditions_match(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        default_abs_notional_threshold=50_000.0,
        use_relative_detection=True,
        min_samples_for_relative_detection=3,
        zscore_threshold=2.0,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    for index, notional in enumerate([10_000.0, 10_100.0, 9_900.0]):
        signal = await detector.process_trade_payload(
            raw_trade_payload_factory(
                price=100.0,
                quantity=notional / 100.0,
                side="buy",
                trade_id=f"combined-baseline-{index}",
            ),
            source_topic="manual.test.single_trade",
            allow_raw_payload=True,
        )
        assert signal is None

    signal = await detector.process_trade_payload(
        raw_trade_payload_factory(
            price=100.0,
            quantity=120_000.0 / 100.0,
            side="buy",
            trade_id="combined-spike",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert signal is not None
    assert signal.trigger_type == "absolute_and_relative"
    assert signal.notional == 120_000.0
    assert signal.zscore >= config.zscore_threshold


async def test_zero_std_relative_detection_does_not_emit_nan_or_false_positive(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        default_abs_notional_threshold=10_000_000.0,
        use_relative_detection=True,
        min_samples_for_relative_detection=3,
        zscore_threshold=2.0,
        min_notional_filter=1.0,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    for index in range(3):
        signal = await detector.process_trade_payload(
            raw_trade_payload_factory(
                price=100.0,
                quantity=100.0,
                side="buy",
                trade_id=f"zero-std-baseline-{index}",
            ),
            source_topic="manual.test.single_trade",
            allow_raw_payload=True,
        )
        assert signal is None

    signal = await detector.process_trade_payload(
        raw_trade_payload_factory(
            price=100.0,
            quantity=100.0,
            side="buy",
            trade_id="zero-std-same-value",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert signal is None

    stats = detector.get_symbol_stats("BTCUSDT")
    assert _safe_stats_float(stats.get("std_notional")) >= 0.0


def _safe_stats_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0

    if math.isnan(result) or math.isinf(result):
        return 0.0

    return result


# =============================================================================
# Cooldowns / concurrency / scoped isolation
# =============================================================================

async def test_signal_cooldown_blocks_duplicate_signals_but_state_keeps_growing(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        signal_cooldown_sec=60.0,
        symbol_cooldown_sec={},
        scoped_cooldown_sec={},
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    first = await detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=1_000.0,
            side="buy",
            trade_id="cooldown-first",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )
    second = await detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=1_100.0,
            side="buy",
            trade_id="cooldown-second",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert first is not None
    assert second is None

    stats = detector.get_symbol_stats("BTCUSDT")
    assert stats["exists"] is True
    assert stats["trades_processed"] == 2
    assert stats["signals_emitted"] == 1


async def test_scoped_cooldown_only_blocks_matching_exchange_market_symbol_timeframe(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        signal_cooldown_sec=0.0,
        scoped_cooldown_sec={
            "binance:usdm_futures:BTCUSDT:realtime": 60.0,
        },
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    first_binance = await detector.process_trade_payload(
        raw_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="realtime",
            price=100.0,
            quantity=1_000.0,
            side="buy",
            trade_id="binance-first",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )
    second_binance = await detector.process_trade_payload(
        raw_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="realtime",
            price=100.0,
            quantity=1_100.0,
            side="buy",
            trade_id="binance-second",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )
    okx_same_symbol = await detector.process_trade_payload(
        raw_trade_payload_factory(
            exchange="okx",
            market_type="swap",
            symbol="BTCUSDT",
            timeframe="realtime",
            price=100.0,
            quantity=1_200.0,
            side="buy",
            trade_id="okx-first",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert first_binance is not None
    assert second_binance is None
    assert okx_same_symbol is not None


async def test_concurrent_same_scope_inputs_do_not_corrupt_state_or_bypass_cooldown(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        signal_cooldown_sec=60.0,
        symbol_cooldown_sec={},
        scoped_cooldown_sec={},
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    payloads = [
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=(90_000.0 + index) / 100.0,
            side="buy",
            trade_id=f"concurrent-same-scope-{index}",
        )
        for index in range(50)
    ]

    results = await asyncio.gather(
        *[
            detector.process_trade_payload(
                payload,
                source_topic="manual.test.single_trade",
                allow_raw_payload=True,
            )
            for payload in payloads
        ]
    )

    stats = detector.get_symbol_stats("BTCUSDT")

    assert stats["exists"] is True
    assert stats["trades_processed"] == 50
    assert stats["signals_emitted"] <= 1
    assert sum(signal is not None for signal in results) <= 1


async def test_concurrent_different_scopes_do_not_cross_contaminate_state(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
) -> None:
    payloads = [
        raw_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="realtime",
            price=100.0,
            quantity=900.0,
            side="buy",
            trade_id="binance-btc",
        ),
        raw_trade_payload_factory(
            exchange="okx",
            market_type="swap",
            symbol="BTCUSDT",
            timeframe="realtime",
            price=100.0,
            quantity=900.0,
            side="sell",
            trade_id="okx-btc",
        ),
        raw_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="ETHUSDT",
            timeframe="realtime",
            price=100.0,
            quantity=900.0,
            side="buy",
            trade_id="binance-eth",
        ),
        raw_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="1m",
            price=100.0,
            quantity=900.0,
            side="buy",
            trade_id="binance-btc-1m",
        ),
    ]

    signals = await asyncio.gather(
        *[
            large_trade_detector.process_trade_payload(
                payload,
                source_topic="manual.test.single_trade",
                allow_raw_payload=True,
            )
            for payload in payloads
        ]
    )

    assert all(signal is not None for signal in signals)

    all_stats = large_trade_detector.get_all_stats()

    assert len(all_stats) == 4

    btc_aggregate = large_trade_detector.get_symbol_stats("BTCUSDT")
    assert btc_aggregate["symbol"] == "BTCUSDT"
    assert btc_aggregate["exists"] is True

    binance_btc = large_trade_detector.get_symbol_stats(
        "BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    okx_btc = large_trade_detector.get_symbol_stats(
        "BTCUSDT",
        exchange="okx",
        market_type="swap",
        timeframe="realtime",
    )

    assert binance_btc["exists"] is True
    assert okx_btc["exists"] is True
    assert binance_btc["exchange"] == "binance"
    assert okx_btc["exchange"] == "okx"


# =============================================================================
# Cleanup / reset / health
# =============================================================================

async def test_cleanup_removes_stale_scoped_states(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        stats_ttl_sec=0.01,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    await detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=900.0,
            side="buy",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )
    await detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="ETHUSDT",
            price=100.0,
            quantity=900.0,
            side="buy",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert len(detector.get_all_stats()) == 2

    await asyncio.sleep(0.02)
    await detector.cleanup()

    assert detector.get_all_stats() == {}


async def test_reset_key_reset_symbol_and_reset_all_are_scoped(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
) -> None:
    await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            timeframe="realtime",
            price=100.0,
            quantity=900.0,
            side="buy",
            trade_id="reset-btc-binance",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )
    await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            exchange="okx",
            market_type="swap",
            symbol="BTCUSDT",
            timeframe="realtime",
            price=100.0,
            quantity=900.0,
            side="buy",
            trade_id="reset-btc-okx",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )
    await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="ETHUSDT",
            timeframe="realtime",
            price=100.0,
            quantity=900.0,
            side="buy",
            trade_id="reset-eth-binance",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    assert len(large_trade_detector.get_all_stats()) == 3

    await large_trade_detector.reset_symbol(
        "BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert (
        large_trade_detector.get_symbol_stats(
            "BTCUSDT",
            exchange="binance",
            market_type="usdm_futures",
            timeframe="realtime",
        )["exists"]
        is False
    )
    assert (
        large_trade_detector.get_symbol_stats(
            "BTCUSDT",
            exchange="okx",
            market_type="swap",
            timeframe="realtime",
        )["exists"]
        is True
    )
    assert large_trade_detector.get_symbol_stats("ETHUSDT")["exists"] is True

    await large_trade_detector.reset_symbol("BTCUSDT")

    assert large_trade_detector.get_symbol_stats("BTCUSDT")["exists"] is False
    assert large_trade_detector.get_symbol_stats("ETHUSDT")["exists"] is True

    await large_trade_detector.reset_all()

    assert large_trade_detector.get_all_stats() == {}


async def test_invalid_symbol_state_api_is_safe(
    large_trade_detector: LargeTradeDetector,
) -> None:
    assert large_trade_detector.get_symbol_stats("")["error"] == "invalid_symbol"
    assert large_trade_detector.get_symbol_stats("   ")["error"] == "invalid_symbol"


async def test_healthcheck_reports_scoped_runtime_shape(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
) -> None:
    await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=900.0,
            side="buy",
        ),
        source_topic="manual.test.single_trade",
        allow_raw_payload=True,
    )

    health = large_trade_detector.get_healthcheck()

    assert health["component"] == "large_trade_detector"
    assert health["event_bus_available"] is True
    assert health["scheduler_available"] is True
    assert health["scope"] == "exchange:market_type:symbol:timeframe"
    assert health["enabled"] is True
    assert health["tracked_scopes"] >= 1
    assert health["state_locks"] >= 1
    assert health["production_input_topics"] == [TRADES_UPDATED_TOPIC]
    assert RAW_TRADE_TOPIC in health["legacy_raw_input_topics"]


# =============================================================================
# EventBus emission payload quality
# =============================================================================

async def test_direct_processing_publishes_large_trade_event_when_emit_on_bus_enabled(
    large_trade_detector: LargeTradeDetector,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    signal = await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=50_000.0,
            quantity=2.0,
            side="buy",
            trade_id="emit-direct-1",
        ),
        source_topic="manual.test.single_trade",
        correlation_id="corr-direct-large-trade",
        allow_raw_payload=True,
    )

    assert signal is not None

    await event_collector.wait_for_topic(
        LARGE_TRADE_TOPIC,
        count=1,
        timeout=1.0,
    )

    event = event_collector.by_topic(LARGE_TRADE_TOPIC)[0]

    assert event.correlation_id == "corr-direct-large-trade"
    assert event.payload["symbol"] == "BTCUSDT"
    assert event.payload["exchange"] == "binance"
    assert event.payload["market_type"] == "usdm_futures"
    assert event.payload["timeframe"] == "realtime"
    assert event.payload["trade_id"] == "emit-direct-1"
    assert event.payload["event_type"] == "large_trade"
    assert event.payload["detector"] == "LargeTradeDetector"
    assert "scope" in event.payload


async def test_emit_on_bus_false_returns_signals_without_publishing_events(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    trades_updated_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        emit_on_bus=False,
        log_signals=False,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    signals = await detector.process_trades_payload(
        trades_updated_payload_factory(
            symbol="BTCUSDT",
            count=2,
            notional=90_000.0,
            batch_id="no-emit-batch",
        ),
        source_topic=TRADES_UPDATED_TOPIC,
        correlation_id="corr-no-emit",
    )

    assert len(signals) == 2

    await asyncio.sleep(0.05)
    assert event_collector.by_topic(LARGE_TRADE_TOPIC) == []


async def test_correlation_source_event_and_source_topic_are_attached_to_emitted_batch_signals(
    large_trade_detector: LargeTradeDetector,
    event_collector,
    trades_updated_payload_factory,
) -> None:
    signals = await large_trade_detector.process_trades_payload(
        trades_updated_payload_factory(
            symbol="BTCUSDT",
            count=2,
            notional=90_000.0,
            batch_id="metadata-batch",
        ),
        source_topic=TRADES_UPDATED_TOPIC,
        source_event_id="event-123",
        correlation_id="corr-metadata-batch",
    )

    assert len(signals) == 2

    await event_collector.wait_for_topic(
        LARGE_TRADE_TOPIC,
        count=2,
        timeout=1.0,
    )

    for event in event_collector.by_topic(LARGE_TRADE_TOPIC):
        assert event.correlation_id == "corr-metadata-batch"
        assert event.headers["source_event_id"] == "event-123"
        assert event.headers["source_topic"] == TRADES_UPDATED_TOPIC
        assert "scope" in event.headers


# =============================================================================
# Production guard against raw payload source topics
# =============================================================================

async def test_process_trades_payload_rejects_raw_topic_when_legacy_not_allowed(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
) -> None:
    signals = await large_trade_detector.process_trades_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=1_000.0,
            side="buy",
            trade_id="raw-topic-direct-rejected",
        ),
        source_topic=RAW_TRADE_TOPIC,
        allow_raw_payload=False,
    )

    assert signals == []
    assert large_trade_detector.get_all_stats() == {}


async def test_process_trades_payload_accepts_raw_topic_when_explicitly_allowed(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
) -> None:
    signals = await large_trade_detector.process_trades_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=1_000.0,
            side="buy",
            trade_id="raw-topic-direct-allowed",
        ),
        source_topic=RAW_TRADE_TOPIC,
        allow_raw_payload=True,
    )

    assert len(signals) == 1
    assert signals[0].trade_id == "raw-topic-direct-allowed"


# =============================================================================
# Large hostile production batch
# =============================================================================

async def test_large_hostile_batch_does_not_crash_and_keeps_valid_scoped_signals(
    large_trade_detector: LargeTradeDetector,
    trades_updated_payload_factory,
    raw_trade_payload_factory,
) -> None:
    valid_trades = [
        raw_trade_payload_factory(
            exchange="binance",
            market_type="usdm_futures",
            symbol="BTCUSDT",
            price=100.0,
            quantity=900.0 + index,
            side="buy" if index % 2 == 0 else "sell",
            trade_id=f"hostile-valid-{index}",
        )
        for index in range(20)
    ]

    invalid_items: list[Any] = [
        None,
        [],
        "bad",
        {"symbol": ""},
        {"symbol": "BTCUSDT", "price": "bad", "quantity": 100, "side": "buy"},
        {"symbol": "BTCUSDT", "price": 100, "quantity": 0, "side": "buy"},
        {"symbol": "BTCUSDT", "price": 100, "quantity": 100, "side": "teleport"},
    ]

    mixed_batch: list[Any] = []

    for index, trade in enumerate(valid_trades):
        mixed_batch.append(trade)
        if index < len(invalid_items):
            mixed_batch.append(invalid_items[index])

    signals = await large_trade_detector.process_trades_payload(
        trades_updated_payload_factory(
            trades=mixed_batch,
            symbol="BTCUSDT",
            batch_id="large-hostile-batch",
        ),
        source_topic=TRADES_UPDATED_TOPIC,
        correlation_id="corr-large-hostile-batch",
    )

    assert len(signals) == 20
    assert _signal_trade_ids(signals) == [
        f"hostile-valid-{index}"
        for index in range(20)
    ]

    stats = large_trade_detector.get_symbol_stats("BTCUSDT")
    assert stats["exists"] is True
    assert stats["signals_emitted"] == 20
    assert stats["trades_processed"] == 20