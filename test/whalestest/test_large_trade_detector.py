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


MARKET_TRADE_TOPIC = "market.trade"
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
    count: int = 5,
    notional: float = 10_000.0,
    side: str = "buy",
) -> None:
    for index in range(count):
        signal = await detector.process_trade_payload(
            raw_trade_payload_factory(
                symbol=symbol,
                price=100.0,
                quantity=notional / 100.0,
                side=side,
                trade_id=f"baseline-{symbol}-{index}",
            )
        )
        assert signal is None


def _assert_no_nan_or_inf(value: float) -> None:
    assert not math.isnan(value)
    assert not math.isinf(value)


# =============================================================================
# Lifecycle / core architecture
# =============================================================================


async def test_register_is_idempotent_and_subscribes_only_once(
    large_trade_detector: LargeTradeDetector,
) -> None:
    await large_trade_detector.register()
    await large_trade_detector.register()
    await large_trade_detector.register()

    assert large_trade_detector.is_registered is True
    assert len(large_trade_detector.subscriptions) == 1
    assert large_trade_detector.subscriptions[0].pattern == MARKET_TRADE_TOPIC


async def test_start_registers_subscription_and_cleanup_job_once(
    large_trade_detector: LargeTradeDetector,
) -> None:
    await large_trade_detector.start()
    await large_trade_detector.start()

    assert large_trade_detector.is_started is True
    assert large_trade_detector.is_registered is True
    assert len(large_trade_detector.subscriptions) == 1
    assert len(large_trade_detector.scheduler_job_ids) == 1


async def test_stop_unsubscribes_and_removes_scheduler_jobs(
    large_trade_detector: LargeTradeDetector,
) -> None:
    await large_trade_detector.start()

    assert large_trade_detector.is_started is True
    assert len(large_trade_detector.subscriptions) == 1
    assert len(large_trade_detector.scheduler_job_ids) == 1

    await large_trade_detector.stop()

    assert large_trade_detector.is_started is False
    assert large_trade_detector.is_registered is False
    assert len(large_trade_detector.subscriptions) == 0
    assert len(large_trade_detector.scheduler_job_ids) == 0


async def test_disabled_detector_does_not_register_start_or_emit(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    config = replace(large_trade_detector_config_fast, enabled=False)
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    await detector.register()
    await detector.start()

    assert detector.is_registered is False
    assert detector.is_started is False
    assert len(detector.subscriptions) == 0
    assert len(detector.scheduler_job_ids) == 0

    signal = await detector.process_trade_payload(
        raw_trade_payload_factory(price=100_000.0, quantity=10.0)
    )

    assert signal is None
    assert event_collector.by_topic(LARGE_TRADE_TOPIC) == []


async def test_handle_trade_event_never_raises_on_malformed_event_payload(
    large_trade_detector: LargeTradeDetector,
) -> None:
    malformed_payloads: list[Any] = [
        None,
        [],
        "not-a-dict",
        {"data": None},
        {"data": []},
        {"data": {"symbol": "", "price": "bad", "quantity": object()}},
    ]

    for payload in malformed_payloads:
        event = Event(
            topic=MARKET_TRADE_TOPIC,
            payload=payload,
            source="test",
        )
        await large_trade_detector.handle_trade_event(event)

    assert large_trade_detector.get_all_stats() == {}


# =============================================================================
# Normalization / adversarial raw payloads
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

    signal = await large_trade_detector.process_trade_payload(payload)

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
        )
    )

    assert signal is not None
    assert signal.symbol == "BTCUSDT"
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
        )
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
        )
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
        )
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
        )
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
        )
    )

    assert signal is not None
    assert signal.symbol == "BTCUSDT"
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
        )
    )

    btc_signal = await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=2_000.0,
            quantity=15.0,
            side="buy",
        )
    )

    assert eth_signal is not None
    assert eth_signal.notional == 30_000.0
    assert eth_signal.abs_threshold == 25_000.0

    assert btc_signal is None


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
        )
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
            )
        )
        assert signal is None

    signal = await detector.process_trade_payload(
        raw_trade_payload_factory(
            price=100.0,
            quantity=100_000.0 / 100.0,
            side="buy",
            trade_id="relative-spike",
        )
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
            )
        )
        assert signal is None

    signal = await detector.process_trade_payload(
        raw_trade_payload_factory(
            price=100.0,
            quantity=120_000.0 / 100.0,
            side="buy",
            trade_id="combined-spike",
        )
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
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    for index in range(3):
        signal = await detector.process_trade_payload(
            raw_trade_payload_factory(
                price=100.0,
                quantity=100.0,
                side="buy",
                trade_id=f"flat-{index}",
            )
        )
        assert signal is None

    signal = await detector.process_trade_payload(
        raw_trade_payload_factory(
            price=100.0,
            quantity=100.0,
            side="buy",
            trade_id="same-notional",
        )
    )

    assert signal is None

    stats = detector.get_symbol_stats("BTCUSDT")
    assert stats["exists"] is True
    assert stats["std_notional"] == 0.0


# =============================================================================
# Cooldown / state mutation / rolling window behavior
# =============================================================================


async def test_symbol_cooldown_blocks_duplicate_signals_but_still_updates_stats(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        signal_cooldown_sec=60.0,
        default_abs_notional_threshold=50_000.0,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    first = await detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=50_000.0,
            quantity=2.0,
            side="buy",
            trade_id="cooldown-1",
        )
    )
    second = await detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=50_000.0,
            quantity=3.0,
            side="buy",
            trade_id="cooldown-2",
        )
    )

    assert first is not None
    assert second is None

    stats = detector.get_symbol_stats("BTCUSDT")
    assert stats["exists"] is True
    assert stats["trades_processed"] == 2
    assert stats["signals_emitted"] == 1
    assert stats["sample_size"] == 2


async def test_symbol_specific_cooldown_does_not_block_other_symbols(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        signal_cooldown_sec=60.0,
        symbol_cooldown_sec={"BTCUSDT": 60.0, "ETHUSDT": 60.0},
        default_abs_notional_threshold=50_000.0,
        symbol_abs_thresholds={"ETHUSDT": 25_000.0},
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    btc = await detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=50_000.0,
            quantity=2.0,
            trade_id="btc-1",
        )
    )
    eth = await detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="ETHUSDT",
            price=2_000.0,
            quantity=20.0,
            trade_id="eth-1",
        )
    )

    assert btc is not None
    assert eth is not None
    assert btc.symbol == "BTCUSDT"
    assert eth.symbol == "ETHUSDT"

    assert detector.get_symbol_stats("BTCUSDT")["signals_emitted"] == 1
    assert detector.get_symbol_stats("ETHUSDT")["signals_emitted"] == 1


async def test_rolling_window_evicts_old_values_and_recalibrates_running_stats(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        default_abs_notional_threshold=10_000_000.0,
        rolling_window_size=3,
        min_samples_for_relative_detection=3,
        recalibration_interval=2,
        zscore_threshold=100.0,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    notionals = [10_000.0, 20_000.0, 30_000.0, 40_000.0]

    for index, notional in enumerate(notionals):
        signal = await detector.process_trade_payload(
            raw_trade_payload_factory(
                price=100.0,
                quantity=notional / 100.0,
                side="buy",
                trade_id=f"rolling-{index}",
            )
        )
        assert signal is None

    stats = detector.get_symbol_stats("BTCUSDT")

    assert stats["exists"] is True
    assert stats["sample_size"] == 3
    assert stats["trades_processed"] == 4
    assert stats["mean_notional"] == pytest.approx((20_000.0 + 30_000.0 + 40_000.0) / 3)


# =============================================================================
# EventBus emission
# =============================================================================


async def test_direct_processing_emits_large_trade_event_on_event_bus(
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
            trade_id="bus-direct-1",
        )
    )

    assert signal is not None

    await event_collector.wait_for_topic(LARGE_TRADE_TOPIC, count=1)

    payloads = event_collector.payloads_by_topic(LARGE_TRADE_TOPIC)
    assert len(payloads) == 1

    payload = payloads[0]
    assert payload["event_type"] == "large_trade"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["trade_id"] == "bus-direct-1"
    assert payload["notional"] == 100_000.0


async def test_emit_on_bus_false_returns_signal_without_publishing_event(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    config = replace(large_trade_detector_config_fast, emit_on_bus=False)
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    signal = await detector.process_trade_payload(
        raw_trade_payload_factory(
            price=50_000.0,
            quantity=2.0,
            side="buy",
            trade_id="no-bus-1",
        )
    )

    assert signal is not None
    await asyncio.sleep(0.05)
    assert event_collector.by_topic(LARGE_TRADE_TOPIC) == []


async def test_registered_eventbus_handler_processes_market_trade_and_preserves_correlation_id(
    large_trade_detector: LargeTradeDetector,
    event_bus: EventBus,
    event_collector,
    raw_trade_payload_factory,
) -> None:
    await large_trade_detector.register()

    accepted = await event_bus.emit(
        MARKET_TRADE_TOPIC,
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=50_000.0,
            quantity=2.0,
            side="buy",
            trade_id="bus-handler-1",
        ),
        source="tests.market_stream",
        correlation_id="corr-large-trade-1",
    )

    assert accepted is True

    events = await event_collector.wait_for_topic(LARGE_TRADE_TOPIC, count=1)
    emitted = [event for event in events if event.topic == LARGE_TRADE_TOPIC][0]

    assert emitted.correlation_id == "corr-large-trade-1"
    assert emitted.payload["trade_id"] == "bus-handler-1"


# =============================================================================
# Cleanup / reset / healthcheck
# =============================================================================


async def test_cleanup_removes_stale_symbol_stats_and_locks(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        stats_ttl_sec=0.01,
        cleanup_interval_sec=0.01,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    signal = await detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=50_000.0,
            quantity=2.0,
            side="buy",
        )
    )
    assert signal is not None
    assert detector.get_symbol_stats("BTCUSDT")["exists"] is True

    await asyncio.sleep(0.02)
    await detector.cleanup()

    assert detector.get_symbol_stats("BTCUSDT")["exists"] is False
    assert "BTCUSDT" not in detector.get_all_stats()


async def test_reset_symbol_removes_only_requested_symbol(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
) -> None:
    await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=50_000.0,
            quantity=2.0,
            trade_id="reset-btc",
        )
    )
    await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="ETHUSDT",
            price=2_000.0,
            quantity=20.0,
            trade_id="reset-eth",
        )
    )

    assert large_trade_detector.get_symbol_stats("BTCUSDT")["exists"] is True
    assert large_trade_detector.get_symbol_stats("ETHUSDT")["exists"] is True

    await large_trade_detector.reset_symbol("btcusdt")

    assert large_trade_detector.get_symbol_stats("BTCUSDT")["exists"] is False
    assert large_trade_detector.get_symbol_stats("ETHUSDT")["exists"] is True


async def test_reset_all_clears_all_symbol_stats_and_healthcheck_counts(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
) -> None:
    await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=50_000.0,
            quantity=2.0,
        )
    )
    await large_trade_detector.process_trade_payload(
        raw_trade_payload_factory(
            symbol="ETHUSDT",
            price=2_000.0,
            quantity=20.0,
        )
    )

    assert large_trade_detector.get_healthcheck()["tracked_symbols"] == 2

    await large_trade_detector.reset_all()

    assert large_trade_detector.get_all_stats() == {}
    assert large_trade_detector.get_healthcheck()["tracked_symbols"] == 0


async def test_get_symbol_stats_returns_invalid_symbol_error_without_exception(
    large_trade_detector: LargeTradeDetector,
) -> None:
    assert large_trade_detector.get_symbol_stats("") == {
        "symbol": "",
        "exists": False,
        "error": "invalid_symbol",
    }

    assert large_trade_detector.get_symbol_stats("   ") == {
        "symbol": "   ",
        "exists": False,
        "error": "invalid_symbol",
    }


# =============================================================================
# Concurrency / race-condition tests
# =============================================================================


async def test_concurrent_trades_same_symbol_do_not_corrupt_stats_or_locks(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        default_abs_notional_threshold=10_000_000.0,
        use_relative_detection=False,
        rolling_window_size=200,
        min_notional_filter=1.0,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    payloads = [
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=100.0,
            quantity=(10_000.0 + index) / 100.0,
            side="buy" if index % 2 == 0 else "sell",
            trade_id=f"concurrent-{index}",
        )
        for index in range(100)
    ]

    results = await asyncio.gather(
        *(detector.process_trade_payload(payload) for payload in payloads)
    )

    assert all(result is None for result in results)

    stats = detector.get_symbol_stats("BTCUSDT")
    assert stats["exists"] is True
    assert stats["trades_processed"] == 100
    assert stats["sample_size"] == 100
    assert stats["signals_emitted"] == 0

    # Вразливе місце: для одного symbol має існувати тільки один lock.
    assert len(detector._symbol_locks) == 1  # noqa: SLF001
    assert "BTCUSDT" in detector._symbol_locks  # noqa: SLF001


async def test_concurrent_trades_different_symbols_create_isolated_states(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        default_abs_notional_threshold=10_000_000.0,
        use_relative_detection=False,
        rolling_window_size=50,
        min_notional_filter=1.0,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

    payloads = [
        raw_trade_payload_factory(
            symbol=symbol,
            price=100.0,
            quantity=100.0 + index,
            side="buy",
            trade_id=f"{symbol}-{index}",
        )
        for symbol in symbols
        for index in range(10)
    ]

    await asyncio.gather(
        *(detector.process_trade_payload(payload) for payload in payloads)
    )

    all_stats = detector.get_all_stats()

    assert set(all_stats) == set(symbols)

    for symbol in symbols:
        stats = detector.get_symbol_stats(symbol)
        assert stats["exists"] is True
        assert stats["trades_processed"] == 10
        assert stats["sample_size"] == 10


async def test_concurrent_large_trades_with_cooldown_emit_only_one_signal_per_symbol(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
    raw_trade_payload_factory,
) -> None:
    config = replace(
        large_trade_detector_config_fast,
        default_abs_notional_threshold=50_000.0,
        signal_cooldown_sec=60.0,
        use_relative_detection=False,
    )
    detector = _build_detector(config=config, event_bus=event_bus, scheduler=scheduler)

    payloads = [
        raw_trade_payload_factory(
            symbol="BTCUSDT",
            price=50_000.0,
            quantity=2.0 + index,
            side="buy",
            trade_id=f"cooldown-race-{index}",
        )
        for index in range(20)
    ]

    results = await asyncio.gather(
        *(detector.process_trade_payload(payload) for payload in payloads)
    )

    emitted_signals = [result for result in results if result is not None]

    assert len(emitted_signals) == 1

    stats = detector.get_symbol_stats("BTCUSDT")
    assert stats["exists"] is True
    assert stats["trades_processed"] == 20
    assert stats["signals_emitted"] == 1
    assert stats["sample_size"] == 20


# =============================================================================
# Backward compatibility
# =============================================================================


async def test_backward_compatible_process_trade_alias_matches_process_trade_payload(
    large_trade_detector: LargeTradeDetector,
    raw_trade_payload_factory,
) -> None:
    payload = raw_trade_payload_factory(
        symbol="BTCUSDT",
        price=50_000.0,
        quantity=2.0,
        side="buy",
        trade_id="alias-1",
    )

    signal = await large_trade_detector.process_trade(payload)

    assert signal is not None
    assert signal.symbol == "BTCUSDT"
    assert signal.trade_id == "alias-1"