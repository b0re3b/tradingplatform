# tests/analytics/liquidations/test_liquidation_stream.py

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any, Callable

import pytest

from core.event_bus import Event
from core.scheduler import Scheduler

from analytics.liquidations.config import LiquidationStreamConfig
from analytics.liquidations.enums import LiquidationEventType, LiquidationSide
from analytics.liquidations.liquidation_stream import LiquidationStream
from analytics.liquidations.metrics import LiquidationMetrics
from analytics.liquidations.models import LiquidationEvent
from analytics.liquidations.state import LiquidationState
from analytics.liquidations.utils import utc_now


pytestmark = pytest.mark.asyncio


# =============================================================================
# Test doubles
# =============================================================================

class FakeLiquidationHistoryStore:
    def __init__(self, *, fail_append: bool = False, fail_flush: bool = False) -> None:
        self.fail_append = fail_append
        self.fail_flush = fail_flush

        self.events: list[LiquidationEvent] = []
        self.large_events: list[LiquidationEvent] = []
        self.flush_calls = 0

    async def append_event(self, event: LiquidationEvent) -> None:
        if self.fail_append:
            raise RuntimeError("append failed")
        self.events.append(event)

    async def append_large_event(self, event: LiquidationEvent) -> None:
        if self.fail_append:
            raise RuntimeError("append large failed")
        self.large_events.append(event)

    async def flush(self) -> None:
        self.flush_calls += 1
        if self.fail_flush:
            raise RuntimeError("flush failed")


# =============================================================================
# Helpers
# =============================================================================

def _scope_key(
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
) -> str:
    return f"{exchange}:{market_type}:{symbol}:{timeframe}"


def _job_names(scheduler: Scheduler) -> set[str]:
    return {job["name"] for job in scheduler.list_jobs().values()}


def _assert_stream_jobs_registered(
    scheduler: Scheduler,
    config: LiquidationStreamConfig,
    service_name: str = "liquidation_stream",
) -> None:
    job_names = _job_names(scheduler)

    assert f"{config.healthcheck_job_name}:{service_name}" in job_names
    assert f"{config.snapshot_job_name}:{service_name}" in job_names
    assert f"{config.cleanup_job_name}:{service_name}" in job_names


# =============================================================================
# Construction / registration
# =============================================================================

async def test_liquidation_stream_uses_core_dependencies(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
) -> None:
    history_store = FakeLiquidationHistoryStore()

    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
        history_store=history_store,
    )

    assert stream.event_bus is event_bus
    assert stream.scheduler is scheduler
    assert stream.config is stream_config
    assert stream.state is liquidation_state
    assert stream.metrics is liquidation_metrics
    assert stream.history_store is history_store
    assert stream.service_name == "liquidation_stream"
    assert stream.input_topic == "market.liquidation"
    assert stream.publish_topic_updated == stream_config.publish_topic_updated
    assert stream.is_running is False
    assert stream.is_connected is False
    assert stream.is_closed is False


async def test_stream_rejects_raw_topic_when_raw_input_topics_are_disabled() -> None:
    with pytest.raises(
        ValueError,
        match="LiquidationStream is configured with raw input topics",
    ):
        LiquidationStreamConfig(
            enabled=True,
            input_topic_raw="market.liquidation",
            input_topics_raw=("market.liquidation",),
            allow_raw_input_topics=False,
        )


async def test_register_subscribes_to_raw_input_topic_and_adds_scheduler_jobs(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
    )

    stream.register()

    assert stream.is_running is False
    assert stream.is_connected is False

    stats = stream.get_stats()
    assert stats["registered"] is True
    assert stats["running"] is False
    assert stats["input_topics"] == ["market.liquidation"]
    assert stats["scope"] == "exchange:market_type:symbol:timeframe"
    assert stats["healthcheck_job_id"] is not None
    assert stats["snapshot_job_id"] is not None
    assert stats["cleanup_job_id"] is not None

    assert event_bus.stats()["subscriptions"] == 1
    _assert_stream_jobs_registered(scheduler, stream_config)


async def test_register_without_scheduler_only_subscribes_to_event_bus(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=None,
        config=stream_config,
        state=liquidation_state,
    )

    stream.register()

    assert event_bus.stats()["subscriptions"] == 1

    stats = stream.get_stats()
    assert stats["registered"] is True
    assert stats["healthcheck_job_id"] is None
    assert stats["snapshot_job_id"] is None
    assert stats["cleanup_job_id"] is None


async def test_register_is_idempotent(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
    )

    stream.register()

    subscriptions_after_first_register = event_bus.stats()["subscriptions"]
    jobs_after_first_register = scheduler.list_jobs()

    stream.register()

    subscriptions_after_second_register = event_bus.stats()["subscriptions"]
    jobs_after_second_register = scheduler.list_jobs()

    assert subscriptions_after_second_register == subscriptions_after_first_register
    assert len(jobs_after_second_register) == len(jobs_after_first_register)
    assert stream.get_stats()["registered"] is True


async def test_register_is_skipped_when_config_disabled(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream_config.enabled = False

    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
    )

    stream.register()

    assert event_bus.stats()["subscriptions"] == 0
    assert scheduler.list_jobs() == {}
    assert stream.get_stats()["registered"] is False


async def test_register_closed_stream_raises_runtime_error(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.close()

    with pytest.raises(RuntimeError, match="Cannot register closed LiquidationStream"):
        stream.register()


# =============================================================================
# Lifecycle
# =============================================================================

async def test_start_auto_registers_stream(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()

    assert stream.is_running is True
    assert stream.is_connected is True
    assert stream.get_health()["status"] == "starting"
    assert event_bus.stats()["subscriptions"] == 1
    _assert_stream_jobs_registered(scheduler, stream_config)

    await stream.stop()

    assert stream.is_running is False
    assert stream.is_connected is False
    assert stream.get_health()["status"] == "stopped"


async def test_start_is_idempotent(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()
    subscriptions_after_first_start = event_bus.stats()["subscriptions"]
    jobs_after_first_start = scheduler.list_jobs()

    await stream.start()

    assert stream.is_running is True
    assert event_bus.stats()["subscriptions"] == subscriptions_after_first_start
    assert len(scheduler.list_jobs()) == len(jobs_after_first_start)


async def test_start_does_nothing_when_config_disabled(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream_config.enabled = False

    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()

    assert stream.is_running is False
    assert stream.is_connected is False
    assert event_bus.stats()["subscriptions"] == 0
    assert scheduler.list_jobs() == {}


async def test_start_closed_stream_raises_runtime_error(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.close()

    with pytest.raises(RuntimeError, match="Cannot start closed LiquidationStream"):
        await stream.start()


async def test_stop_is_idempotent(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.stop()
    await stream.stop()

    assert stream.is_running is False
    assert stream.get_health()["status"] == "stopped"


async def test_stop_flushes_history_store_when_running(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    history_store = FakeLiquidationHistoryStore()

    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
        history_store=history_store,
    )

    await stream.start()
    await stream.stop()

    assert history_store.flush_calls == 1
    assert stream.get_stats()["storage_errors"] == 0


async def test_stop_records_storage_error_when_history_flush_fails(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    history_store = FakeLiquidationHistoryStore(fail_flush=True)

    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
        history_store=history_store,
    )

    await stream.start()
    await stream.stop()

    stats = stream.get_stats()
    assert history_store.flush_calls == 1
    assert stats["storage_errors"] == 1


async def test_unregister_removes_subscriptions_and_scheduler_jobs(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
    )

    stream.register()

    assert event_bus.stats()["subscriptions"] == 1
    assert scheduler.list_jobs()

    await stream.unregister()

    assert event_bus.stats()["subscriptions"] == 0
    assert scheduler.list_jobs() == {}
    assert stream.get_stats()["registered"] is False


async def test_close_stops_unregisters_and_marks_stream_closed(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()

    assert stream.is_running is True
    assert scheduler.list_jobs()

    await stream.close()

    assert stream.is_running is False
    assert stream.is_connected is False
    assert stream.is_closed is True
    assert event_bus.stats()["subscriptions"] == 0
    assert scheduler.list_jobs() == {}

    stats = stream.get_stats()
    assert stats["closed"] is True
    assert stats["registered"] is False


async def test_close_is_idempotent(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.close()
    await stream.close()

    assert stream.is_closed is True
    assert event_bus.stats()["subscriptions"] == 0
    assert scheduler.list_jobs() == {}


async def test_restart_stops_and_starts_stream_again(
    event_bus,
    scheduler: Scheduler,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()
    assert stream.is_running is True

    await stream.restart()

    assert stream.is_running is True
    assert stream.is_connected is True
    assert stream.get_stats()["registered"] is True

    await stream.stop()

    assert stream.is_running is False


# =============================================================================
# Normalization
# =============================================================================

async def test_normalize_flat_payload_returns_liquidation_event_with_full_scope(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = stream.normalize_event(raw_liquidation_payload)

    assert isinstance(event, LiquidationEvent)
    assert event.exchange == "binance"
    assert event.market_type == "usdm_futures"
    assert event.symbol == "BTCUSDT"
    assert event.timeframe == "realtime"
    assert event.exchange_symbol == "BTCUSDT"
    assert event.side is LiquidationSide.LONG
    assert event.price == Decimal("65000")
    assert event.quantity == Decimal("2")
    assert event.notional_usd == Decimal("130000")
    assert event.event_type is LiquidationEventType.NORMALIZED
    assert event.source == stream.service_name
    assert event.is_valid is True
    assert event.key == ("binance", "usdm_futures", "BTCUSDT", "realtime")
    assert event.metadata["scope"] == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
    }


async def test_normalize_buy_side_payload_returns_short_liquidation(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_buy_side_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = stream.normalize_event(raw_buy_side_liquidation_payload)

    assert isinstance(event, LiquidationEvent)
    assert event.side is LiquidationSide.SHORT
    assert event.direction.value == "up"


async def test_normalize_binance_force_order_payload_returns_liquidation_event(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    binance_force_order_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = stream.normalize_event(binance_force_order_payload)

    assert isinstance(event, LiquidationEvent)
    assert event.exchange == "binance"
    assert event.market_type == "usdm_futures"
    assert event.symbol == "BTCUSDT"
    assert event.timeframe == "realtime"
    assert event.exchange_symbol == "BTCUSDT"
    assert event.side is LiquidationSide.LONG
    assert event.price == Decimal("65000")
    assert event.quantity == Decimal("2")
    assert event.notional_usd == Decimal("130000")
    assert event.is_valid is True


async def test_normalize_bybit_linear_payload_keeps_linear_market_type(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    bybit_linear_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = stream.normalize_event(bybit_linear_liquidation_payload)

    assert isinstance(event, LiquidationEvent)
    assert event.exchange == "bybit"
    assert event.market_type == "linear"
    assert event.symbol == "BTCUSDT"
    assert event.timeframe == "realtime"
    assert event.exchange_symbol == "BTCUSDT"
    assert event.side is LiquidationSide.LONG


async def test_normalize_okx_swap_payload_keeps_native_exchange_symbol(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    okx_swap_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = stream.normalize_event(okx_swap_liquidation_payload)

    assert isinstance(event, LiquidationEvent)
    assert event.exchange == "okx"
    assert event.market_type == "swap"
    assert event.symbol == "BTCUSDT"
    assert event.exchange_symbol == "BTC-USDT-SWAP"
    assert event.key == ("okx", "swap", "BTCUSDT", "realtime")


async def test_normalize_payload_uses_explicit_notional_when_present(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> None:
    payload = make_raw_liquidation_payload(
        price="65000",
        quantity="1",
        extra={"notional_usd": "99999"},
    )

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = stream.normalize_event(payload)

    assert isinstance(event, LiquidationEvent)
    assert event.notional_usd == Decimal("99999")


async def test_normalize_invalid_payload_returns_none(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_invalid_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = stream.normalize_event(raw_invalid_liquidation_payload)

    assert event is None


async def test_normalize_malformed_timestamp_returns_none_and_records_error(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> None:
    payload = make_raw_liquidation_payload(
        extra={"timestamp": "not-a-valid-datetime"},
    )

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = stream.normalize_event(payload)

    assert event is None
    assert stream.get_stats()["last_error"] is not None


# =============================================================================
# Direct raw handling / state / metrics
# =============================================================================

async def test_handle_raw_message_adds_event_to_state_using_full_scope(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = await stream.handle_raw_message(raw_liquidation_payload)

    assert event is not None

    symbol_state = liquidation_state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert symbol_state is not None
    assert symbol_state.total_buffered_events == 1
    assert symbol_state.total_events_seen == 1
    assert liquidation_state.get("binance", "BTCUSDT") is None

    stats = stream.get_stats()
    assert stats["processed_events"] == 1
    assert stats["tracked_scopes"] == 1
    assert stats["state_total_buffered_events"] == 1
    assert stats["last_event_at"] is not None


async def test_handle_raw_message_updates_metrics_with_full_scope(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    raw_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    event = await stream.handle_raw_message(raw_liquidation_payload)

    assert event is not None
    assert liquidation_metrics.total_events_seen == 1
    assert liquidation_metrics.total_valid_events == 1
    assert liquidation_metrics.total_invalid_events == 0
    assert liquidation_metrics.total_long_events == 1
    assert liquidation_metrics.total_long_notional_usd == Decimal("130000")
    assert liquidation_metrics.scope_event_counts[_scope_key()] == 1
    assert liquidation_metrics.market_type_event_counts["usdm_futures"] == 1


async def test_handle_invalid_payload_is_dropped_and_updates_metrics(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    raw_invalid_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    event = await stream.handle_raw_message(raw_invalid_liquidation_payload)

    assert event is None
    assert liquidation_state.scopes_count == 0
    assert liquidation_metrics.total_events_seen == 1
    assert liquidation_metrics.total_invalid_events == 1

    stats = stream.get_stats()
    assert stats["dropped_invalid"] == 1
    assert stats["processed_events"] == 0


async def test_handle_stale_payload_is_dropped_and_not_added_to_state(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
    raw_stale_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    event = await stream.handle_raw_message(raw_stale_liquidation_payload)

    assert event is None
    assert liquidation_state.scopes_count == 0
    assert liquidation_metrics.total_events_seen == 1
    assert liquidation_metrics.total_stale_events == 1

    stats = stream.get_stats()
    assert stats["dropped_stale"] == 1
    assert stats["processed_events"] == 0


async def test_duplicate_payload_is_dropped_after_first_event(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_liquidation_payload: dict[str, Any],
) -> None:
    stream_config.deduplication_enabled = True

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    first_event = await stream.handle_raw_message(raw_liquidation_payload)
    second_event = await stream.handle_raw_message(raw_liquidation_payload)

    assert first_event is not None
    assert second_event is None

    symbol_state = liquidation_state.get(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert symbol_state is not None
    assert symbol_state.total_buffered_events == 1

    stats = stream.get_stats()
    assert stats["processed_events"] == 1
    assert stats["dropped_duplicates"] == 1


async def test_deduplication_can_be_disabled(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_liquidation_payload: dict[str, Any],
) -> None:
    stream_config.deduplication_enabled = False

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    first_event = await stream.handle_raw_message(raw_liquidation_payload)
    second_event = await stream.handle_raw_message(raw_liquidation_payload)

    assert first_event is not None
    assert second_event is not None

    symbol_state = liquidation_state.get_key(first_event.key)
    assert symbol_state is not None
    assert symbol_state.total_buffered_events == 2
    assert stream.get_stats()["dropped_duplicates"] == 0


async def test_different_payloads_with_different_trade_ids_are_not_deduplicated(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> None:
    stream_config.deduplication_enabled = True

    first_payload = make_raw_liquidation_payload(
        trade_id="dedup-trade-1",
        order_id="dedup-order-1",
    )
    second_payload = make_raw_liquidation_payload(
        trade_id="dedup-trade-2",
        order_id="dedup-order-2",
    )

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    first_event = await stream.handle_raw_message(first_payload)
    second_event = await stream.handle_raw_message(second_payload)

    assert first_event is not None
    assert second_event is not None
    assert stream.get_stats()["processed_events"] == 2
    assert stream.get_stats()["dropped_duplicates"] == 0


async def test_scope_filter_drops_payload_before_normalization_side_effects(
    event_bus,
    strict_btc_usdm_stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    bybit_linear_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=strict_btc_usdm_stream_config,
        state=liquidation_state,
    )

    event = await stream.handle_raw_message(bybit_linear_liquidation_payload)

    assert event is None
    assert liquidation_state.scopes_count == 0

    stats = stream.get_stats()
    assert stats["filtered_scope"] == 1
    assert stats["processed_events"] == 0
    assert stats["published_raw"] == 0
    assert stats["published_normalized"] == 0


async def test_same_symbol_different_market_types_create_separate_state_scopes(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    same_symbol_different_market_type_payloads: list[dict[str, Any]],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    events = [
        await stream.handle_raw_message(payload)
        for payload in same_symbol_different_market_type_payloads
    ]

    assert all(event is not None for event in events)
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
    assert usdm_state is not coinm_state
    assert usdm_state.exchange_symbol == "BTCUSDT"
    assert coinm_state.exchange_symbol == "BTCUSD_PERP"


async def test_same_symbol_different_timeframes_create_separate_state_scopes(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    same_symbol_different_timeframe_payloads: list[dict[str, Any]],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    events = [
        await stream.handle_raw_message(payload)
        for payload in same_symbol_different_timeframe_payloads
    ]

    assert all(event is not None for event in events)
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
    assert realtime_state is not one_minute_state


# =============================================================================
# Direct EventBus publishing
# =============================================================================

async def test_handle_raw_message_publishes_raw_event_when_enabled(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    stream_config.emit_raw_events = True
    received = event_collector(event_bus, stream_config.publish_topic_raw)

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = await stream.handle_raw_message(raw_liquidation_payload)
    await wait_for_events(received)

    assert event is not None
    assert len(received) == 1
    assert received[0].topic == stream_config.publish_topic_raw
    assert received[0].source == stream.service_name

    payload = received[0].payload
    assert isinstance(payload, dict)
    assert payload["exchange"] == "binance"
    assert payload["market_type"] == "usdm_futures"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["timeframe"] == "realtime"
    assert payload["exchange_symbol"] == "BTCUSDT"
    assert payload["fingerprint"] == event.raw_payload_hash
    assert payload["payload"]["symbol"] == "BTCUSDT"

    assert received[0].headers["event_type"] == LiquidationEventType.RAW.value
    assert received[0].headers["market_type"] == "usdm_futures"


async def test_handle_raw_message_publishes_normalized_event_with_full_scope_headers(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    received = event_collector(event_bus, stream_config.publish_topic_normalized)

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = await stream.handle_raw_message(raw_liquidation_payload)
    await wait_for_events(received)

    assert event is not None
    assert len(received) == 1
    assert received[0].topic == stream_config.publish_topic_normalized
    assert isinstance(received[0].payload, LiquidationEvent)
    assert received[0].payload.key == event.key
    assert received[0].payload.notional_usd == Decimal("130000")
    assert received[0].source == stream.service_name
    assert received[0].correlation_id == event.correlation_id
    assert received[0].headers == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
        "exchange_symbol": "BTCUSDT",
        "scope": _scope_key(),
        "event_type": LiquidationEventType.NORMALIZED.value,
    }

    assert stream.get_stats()["published_normalized"] == 1


async def test_handle_raw_message_publishes_updated_event(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    received = event_collector(event_bus, stream_config.publish_topic_updated)

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = await stream.handle_raw_message(raw_liquidation_payload)
    await wait_for_events(received)

    assert event is not None
    assert len(received) == 1
    assert received[0].topic == stream_config.publish_topic_updated
    assert received[0].source == stream.service_name
    assert received[0].correlation_id == event.correlation_id
    assert received[0].headers["event_type"] == "updated"
    assert received[0].headers["scope"] == _scope_key()

    payload = received[0].payload
    assert payload["exchange"] == "binance"
    assert payload["market_type"] == "usdm_futures"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["timeframe"] == "realtime"
    assert payload["scope_key"] == _scope_key()
    assert payload["symbol_state_total"] == 1
    assert payload["event"]["notional_usd"] == "130000"

    assert stream.get_stats()["published_updated"] == 1


async def test_handle_large_liquidation_publishes_large_event_and_buffers_it(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_large_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    stream_config.emit_large_events = True
    stream_config.large_liquidation_threshold_usd = Decimal("100000")

    received = event_collector(event_bus, stream_config.publish_topic_large)

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = await stream.handle_raw_message(raw_large_liquidation_payload)
    await wait_for_events(received)

    assert event is not None
    assert event.notional_usd >= stream_config.large_liquidation_threshold_usd
    assert len(received) == 1
    assert received[0].topic == stream_config.publish_topic_large
    assert isinstance(received[0].payload, LiquidationEvent)
    assert received[0].headers["event_type"] == LiquidationEventType.LARGE.value
    assert received[0].headers["scope"] == _scope_key()

    large_events = stream.get_recent_large_events(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        limit=10,
    )

    assert large_events == [event]
    assert stream.get_stats()["published_large"] == 1
    assert stream.get_stats()["large_events_buffered"] == 1


async def test_small_liquidation_does_not_publish_large_event(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_small_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    stream_config.emit_large_events = True
    stream_config.large_liquidation_threshold_usd = Decimal("100000")

    received = event_collector(event_bus, stream_config.publish_topic_large)

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = await stream.handle_raw_message(raw_small_liquidation_payload)
    await wait_for_events(received, timeout=0.05)

    assert event is not None
    assert event.notional_usd < stream_config.large_liquidation_threshold_usd
    assert received == []
    assert stream.get_recent_large_events(limit=10) == []
    assert stream.get_stats()["published_large"] == 0


async def test_large_liquidation_does_not_publish_large_event_when_disabled(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_large_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    stream_config.emit_large_events = False

    received = event_collector(event_bus, stream_config.publish_topic_large)

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = await stream.handle_raw_message(raw_large_liquidation_payload)
    await wait_for_events(received, timeout=0.05)

    assert event is not None
    assert event.notional_usd >= stream_config.large_liquidation_threshold_usd
    assert received == []
    assert stream.get_recent_large_events(limit=10) == []
    assert stream.get_stats()["published_large"] == 0


# =============================================================================
# EventBus input handler
# =============================================================================

async def test_on_raw_liquidation_ignores_event_when_stream_not_running(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.on_raw_liquidation(
        Event(
            topic=stream_config.input_topic_raw,
            payload=raw_liquidation_payload,
        )
    )

    assert liquidation_state.scopes_count == 0
    assert stream.get_stats()["processed_messages"] == 0
    assert stream.get_stats()["processed_events"] == 0


async def test_on_raw_liquidation_drops_non_dict_payload_when_running(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    liquidation_metrics: LiquidationMetrics,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
        metrics=liquidation_metrics,
    )

    await stream.start()

    await stream.on_raw_liquidation(
        Event(
            topic=stream_config.input_topic_raw,
            payload="not-a-dict",
        )
    )

    assert liquidation_state.scopes_count == 0
    assert liquidation_metrics.total_invalid_events == 1

    stats = stream.get_stats()
    assert stats["processed_messages"] == 1
    assert stats["processed_events"] == 0
    assert stats["dropped_invalid"] == 1


async def test_event_bus_raw_liquidation_triggers_stream_pipeline(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    normalized_events = event_collector(event_bus, stream_config.publish_topic_normalized)
    updated_events = event_collector(event_bus, stream_config.publish_topic_updated)

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()

    accepted = await event_bus.emit(
        stream_config.input_topic_raw,
        raw_liquidation_payload,
        source="test.exchange_adapter",
        correlation_id="raw-event-correlation-id",
    )

    await wait_for_events(normalized_events)
    await wait_for_events(updated_events)

    assert accepted is True
    assert len(normalized_events) == 1
    assert len(updated_events) == 1

    normalized_payload = normalized_events[0].payload
    assert isinstance(normalized_payload, LiquidationEvent)
    assert normalized_payload.correlation_id == "raw-event-correlation-id"
    assert normalized_payload.metadata["source_topic"] == stream_config.input_topic_raw
    assert normalized_payload.key == ("binance", "usdm_futures", "BTCUSDT", "realtime")

    assert liquidation_state.scopes_count == 1
    assert stream.get_stats()["processed_messages"] == 1
    assert stream.get_stats()["processed_events"] == 1


async def test_event_bus_scope_filtered_payload_does_not_publish_normalized_event(
    event_bus,
    strict_btc_usdm_stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    bybit_linear_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    received = event_collector(
        event_bus,
        strict_btc_usdm_stream_config.publish_topic_normalized,
    )

    stream = LiquidationStream(
        event_bus=event_bus,
        config=strict_btc_usdm_stream_config,
        state=liquidation_state,
    )

    await stream.start()

    accepted = await event_bus.emit(
        strict_btc_usdm_stream_config.input_topic_raw,
        bybit_linear_liquidation_payload,
        source="test.exchange_adapter",
    )

    await wait_for_events(received, timeout=0.05)

    assert accepted is True
    assert received == []
    assert liquidation_state.scopes_count == 0

    stats = stream.get_stats()
    assert stats["processed_messages"] == 1
    assert stats["filtered_scope"] == 1
    assert stats["processed_events"] == 0


# =============================================================================
# History store
# =============================================================================

async def test_history_store_receives_normalized_and_large_events(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_large_liquidation_payload: dict[str, Any],
) -> None:
    history_store = FakeLiquidationHistoryStore()

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
        history_store=history_store,
    )

    event = await stream.handle_raw_message(raw_large_liquidation_payload)

    assert event is not None
    assert history_store.events == [event]
    assert history_store.large_events == [event]
    assert stream.get_stats()["storage_errors"] == 0


async def test_history_store_receives_normalized_but_not_large_for_small_events(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_small_liquidation_payload: dict[str, Any],
) -> None:
    history_store = FakeLiquidationHistoryStore()

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
        history_store=history_store,
    )

    event = await stream.handle_raw_message(raw_small_liquidation_payload)

    assert event is not None
    assert history_store.events == [event]
    assert history_store.large_events == []


async def test_history_store_append_error_does_not_drop_processed_event(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_large_liquidation_payload: dict[str, Any],
) -> None:
    history_store = FakeLiquidationHistoryStore(fail_append=True)

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
        history_store=history_store,
    )

    event = await stream.handle_raw_message(raw_large_liquidation_payload)

    assert event is not None
    assert liquidation_state.scopes_count == 1

    stats = stream.get_stats()
    assert stats["processed_events"] == 1
    assert stats["storage_errors"] == 1
    assert stats["last_error"] is not None


# =============================================================================
# Read API
# =============================================================================

async def test_get_recent_events_filters_by_full_scope_and_side(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    long_usdm = await stream.handle_raw_message(
        make_raw_liquidation_payload(
            market_type="usdm_futures",
            timeframe="realtime",
            side="SELL",
            trade_id="read-long-usdm",
            order_id="read-long-usdm-order",
        )
    )
    short_usdm = await stream.handle_raw_message(
        make_raw_liquidation_payload(
            market_type="usdm_futures",
            timeframe="realtime",
            side="BUY",
            trade_id="read-short-usdm",
            order_id="read-short-usdm-order",
        )
    )
    long_coinm = await stream.handle_raw_message(
        make_raw_liquidation_payload(
            market_type="coinm_futures",
            timeframe="realtime",
            exchange_symbol="BTCUSD_PERP",
            side="SELL",
            trade_id="read-long-coinm",
            order_id="read-long-coinm-order",
        )
    )

    assert long_usdm is not None
    assert short_usdm is not None
    assert long_coinm is not None

    recent_usdm_longs = stream.get_recent_events(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        side=LiquidationSide.LONG,
        limit=10,
    )
    recent_all = stream.get_recent_events(
        exchange="binance",
        symbol="BTCUSDT",
        limit=10,
    )

    assert recent_usdm_longs == [long_usdm]
    assert recent_all == [long_coinm, short_usdm, long_usdm]


async def test_get_recent_events_for_key_and_symbol_snapshot(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    event = await stream.handle_raw_message(raw_liquidation_payload)

    assert event is not None

    recent_for_key = stream.get_recent_events_for_key(event.key, limit=10)
    key_snapshot = stream.get_key_snapshot(event.key)
    symbol_snapshot = stream.get_symbol_snapshot(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    legacy_snapshot = stream.get_symbol_snapshot(
        exchange="binance",
        symbol="BTCUSDT",
    )

    assert recent_for_key == [event]
    assert key_snapshot is not None
    assert symbol_snapshot is not None
    assert legacy_snapshot is None
    assert key_snapshot.total_buffered_events == 1
    assert symbol_snapshot.market_type == "usdm_futures"


async def test_get_recent_large_events_returns_empty_for_non_positive_limit(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    assert stream.get_recent_large_events(limit=0) == []
    assert stream.get_recent_large_events(limit=-1) == []


# =============================================================================
# Health / snapshots / scheduled jobs
# =============================================================================

async def test_emit_health_publishes_health_event(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    event_collector,
    wait_for_events,
) -> None:
    received = event_collector(event_bus, stream_config.publish_topic_health)

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()

    accepted = await stream.emit_health()
    await wait_for_events(received)

    assert accepted is True
    assert len(received) == 1
    assert received[0].topic == stream_config.publish_topic_health
    assert received[0].source == stream.service_name
    assert received[0].headers["event_type"] == LiquidationEventType.HEALTH.value

    payload = received[0].payload
    assert payload["service"] == stream.service_name
    assert payload["status"] == "starting"
    assert payload["running"] is True
    assert payload["registered"] is True
    assert payload["scope"] == "exchange:market_type:symbol:timeframe"

    assert stream.get_stats()["published_health"] == 1


async def test_emit_runtime_snapshot_publishes_snapshot_event(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_liquidation_payload: dict[str, Any],
    event_collector,
    wait_for_events,
) -> None:
    received = event_collector(event_bus, stream_config.publish_topic_snapshot)

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()
    await stream.handle_raw_message(raw_liquidation_payload)

    accepted = await stream.emit_runtime_snapshot()
    await wait_for_events(received)

    assert accepted is True
    assert len(received) == 1
    assert received[0].topic == stream_config.publish_topic_snapshot
    assert received[0].source == stream.service_name
    assert received[0].headers["event_type"] == LiquidationEventType.SNAPSHOT.value

    payload = received[0].payload
    assert payload["service"] == stream.service_name
    assert payload["input_topics"] == ["market.liquidation"]
    assert payload["scope"] == "exchange:market_type:symbol:timeframe"
    assert payload["stats"]["processed_events"] == 1
    assert payload["metrics"]["total_valid_events"] == 1
    assert len(payload["state"]) == 1

    assert stream.get_stats()["published_snapshots"] == 1


async def test_scheduled_cleanup_prunes_old_events_and_empty_scopes(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> None:
    stream_config.stale_event_threshold_seconds = 1
    stream_config.cleanup_interval_seconds = 1

    old_payload = make_raw_liquidation_payload(
        symbol="BTCUSDT",
        timestamp=utc_now() - timedelta(seconds=5),
        trade_id="cleanup-old",
        order_id="cleanup-old-order",
    )
    fresh_payload = make_raw_liquidation_payload(
        symbol="ETHUSDT",
        timestamp=utc_now(),
        trade_id="cleanup-fresh",
        order_id="cleanup-fresh-order",
    )

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    stream_config.stale_event_threshold_seconds = 60

    old_event = await stream.handle_raw_message(old_payload)
    fresh_event = await stream.handle_raw_message(fresh_payload)

    assert old_event is not None
    assert fresh_event is not None
    assert liquidation_state.scopes_count == 2

    stream_config.stale_event_threshold_seconds = 1
    await stream._scheduled_cleanup()

    assert liquidation_state.scopes_count == 1
    assert stream.get_symbol_snapshot(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    ) is None
    assert stream.get_symbol_snapshot(
        exchange="binance",
        symbol="ETHUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    ) is not None


async def test_scheduled_healthcheck_emits_health(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    event_collector,
    wait_for_events,
) -> None:
    received = event_collector(event_bus, stream_config.publish_topic_health)

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()
    await stream._scheduled_healthcheck()
    await wait_for_events(received)

    assert len(received) == 1
    assert received[0].headers["event_type"] == LiquidationEventType.HEALTH.value
    assert stream.get_stats()["published_health"] == 1


async def test_scheduled_snapshot_emits_runtime_snapshot(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    event_collector,
    wait_for_events,
) -> None:
    received = event_collector(event_bus, stream_config.publish_topic_snapshot)

    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()
    await stream._scheduled_snapshot()
    await wait_for_events(received)

    assert len(received) == 1
    assert received[0].headers["event_type"] == LiquidationEventType.SNAPSHOT.value
    assert stream.get_stats()["published_snapshots"] == 1


async def test_get_health_reports_degraded_when_last_message_is_too_old(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()

    stream._last_message_at = utc_now() - timedelta(
        seconds=stream_config.stale_event_threshold_seconds * 3,
    )

    health = stream.get_health()

    assert health["status"] == "degraded"
    assert health["running"] is True
    assert health["registered"] is True


async def test_stats_expose_expected_runtime_counters(
    event_bus,
    stream_config: LiquidationStreamConfig,
    liquidation_state: LiquidationState,
    raw_large_liquidation_payload: dict[str, Any],
) -> None:
    stream = LiquidationStream(
        event_bus=event_bus,
        config=stream_config,
        state=liquidation_state,
    )

    await stream.start()
    await stream.handle_raw_message(raw_large_liquidation_payload)

    stats = stream.get_stats()

    assert stats["service_name"] == stream.service_name
    assert stats["running"] is True
    assert stats["registered"] is True
    assert stats["closed"] is False
    assert stats["scope"] == "exchange:market_type:symbol:timeframe"
    assert stats["processed_events"] == 1
    assert stats["published_normalized"] == 1
    assert stats["published_updated"] == 1
    assert stats["published_large"] == 1
    assert stats["tracked_scopes"] == 1
    assert stats["tracked_symbols"] == 1
    assert stats["state_total_buffered_events"] == 1
    assert stats["large_events_buffered"] == 1
    assert stats["last_event_at"] is not None