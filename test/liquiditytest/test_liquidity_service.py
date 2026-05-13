# tests/analytics/liquidity/test_liquidity_service.py

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from analytics.liquidity.enums import LiquidityStatus, SweepStatus
from analytics.liquidity.liquidity_service import (
    LiquidityService,
    LiquidityTopics,
)
from analytics.liquidity.models import LiquidityLevel, LiquidityMapSnapshot
from analytics.liquidity.state import LiquidityState


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _event(
    *,
    topic: str,
    payload: dict[str, Any],
    event_id: str = "test-event",
) -> dict[str, Any]:
    """
    LiquidityService._event_payload() підтримує і core.Event, і dict.
    Для тестів простіше використовувати dict.
    """
    return payload


def _published_topics(fake_event_bus) -> list[str]:
    return fake_event_bus.emitted_topics()


def _events_for(fake_event_bus, topic: str):
    return fake_event_bus.events_for(topic)


def _assert_context_ready(
    service: LiquidityService,
    *,
    symbol: str,
    timeframe: str,
) -> None:
    context = service.get_context(symbol, timeframe)

    assert context is not None
    assert context.symbol == symbol
    assert context.timeframe == timeframe
    assert context.current_price is not None
    assert context.current_price > 0


async def _feed_candles(
    service: LiquidityService,
    candles: list[dict[str, Any]],
) -> None:
    for candle in candles:
        await service.on_candle_closed(
            {
                "symbol": candle["symbol"],
                "timeframe": candle["timeframe"],
                "candle": candle,
                "current_price": candle["close"],
                "timestamp": candle["close_time"],
            }
        )


# ---------------------------------------------------------------------
# Lifecycle / registration
# ---------------------------------------------------------------------


class TestLiquidityServiceLifecycle:
    def test_register_subscribes_to_market_topics_and_scheduler_jobs(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        liquidity_service.register()

        assert liquidity_service._registered is True

        assert set(fake_event_bus.topics()) == {
            LiquidityTopics.MARKET_CANDLE_CLOSED,
            LiquidityTopics.MARKET_ORDERBOOK_UPDATED,
            LiquidityTopics.MARKET_PRICE_UPDATED,
        }

        job_names = set(fake_scheduler.job_names())

        assert "analytics_liquidity.cleanup" in job_names
        assert "analytics_liquidity.emit_state_metrics" in job_names
        assert "analytics_liquidity.healthcheck" in job_names

    def test_register_is_idempotent(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        liquidity_service.register()

        subscriptions_count = len(fake_event_bus.subscriptions)
        jobs_count = len(fake_scheduler.jobs)

        liquidity_service.register()

        assert len(fake_event_bus.subscriptions) == subscriptions_count
        assert len(fake_scheduler.jobs) == jobs_count

    @pytest.mark.asyncio
    async def test_start_auto_registers_and_sets_runtime_state(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        await liquidity_service.start()

        assert liquidity_service._registered is True
        assert liquidity_service._running is True

        assert liquidity_service.get_stats().started_at is not None
        assert liquidity_service.get_stats().stopped_at is None

        assert len(fake_event_bus.subscriptions) == 3
        assert len(fake_scheduler.jobs) == 3

    @pytest.mark.asyncio
    async def test_start_is_idempotent(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        await liquidity_service.start()

        started_at = liquidity_service.get_stats().started_at
        subscriptions_count = len(fake_event_bus.subscriptions)
        jobs_count = len(fake_scheduler.jobs)

        await liquidity_service.start()

        assert liquidity_service.get_stats().started_at == started_at
        assert len(fake_event_bus.subscriptions) == subscriptions_count
        assert len(fake_scheduler.jobs) == jobs_count

    @pytest.mark.asyncio
    async def test_stop_unsubscribes_and_removes_scheduler_jobs(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        await liquidity_service.start()

        job_ids = set(fake_scheduler.jobs.keys())

        await liquidity_service.stop()

        assert liquidity_service._running is False
        assert liquidity_service._registered is False
        assert liquidity_service.get_stats().stopped_at is not None

        assert len(fake_event_bus.unsubscribed) == 3
        assert fake_event_bus.subscriptions == []

        assert set(fake_scheduler.removed_job_ids) == job_ids
        assert fake_scheduler.jobs == {}

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        await liquidity_service.start()
        await liquidity_service.stop()

        unsubscribed_count = len(fake_event_bus.unsubscribed)
        removed_jobs_count = len(fake_scheduler.removed_job_ids)

        await liquidity_service.stop()

        assert len(fake_event_bus.unsubscribed) == unsubscribed_count
        assert len(fake_scheduler.removed_job_ids) == removed_jobs_count

    def test_register_without_scheduler_only_subscribes_to_event_bus(
        self,
        liquidity_service_without_scheduler: LiquidityService,
        fake_event_bus,
    ) -> None:
        liquidity_service_without_scheduler.register()

        assert liquidity_service_without_scheduler._registered is True
        assert len(fake_event_bus.subscriptions) == 3
        assert liquidity_service_without_scheduler._scheduler_job_ids == []


# ---------------------------------------------------------------------
# Candle handling / rebuild
# ---------------------------------------------------------------------


class TestLiquidityServiceCandleHandling:
    @pytest.mark.asyncio
    async def test_candle_events_update_context_and_state(
        self,
        liquidity_service: LiquidityService,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        await liquidity_service.start()

        await _feed_candles(
            liquidity_service,
            candles_without_clear_equal_levels[:5],
        )

        context = liquidity_service.get_context(symbol, timeframe)
        state = liquidity_service.get_state().get(symbol, timeframe)
        stats = liquidity_service.get_stats()

        assert context is not None
        assert state is not None

        assert len(context.candles) == 5
        assert context.current_price == pytest.approx(
            candles_without_clear_equal_levels[4]["close"]
        )

        assert state.processed_candles == 5
        assert stats.candle_events_processed == 5

    @pytest.mark.asyncio
    async def test_candle_events_build_snapshot_when_enough_candles(
        self,
        liquidity_service: LiquidityService,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        fake_event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        await liquidity_service.start()

        await _feed_candles(
            liquidity_service,
            candles_without_clear_equal_levels,
        )

        snapshot = liquidity_service.get_last_snapshot(symbol, timeframe)
        context = liquidity_service.get_context(symbol, timeframe)
        state = liquidity_service.get_state().get(symbol, timeframe)
        stats = liquidity_service.get_stats()

        assert snapshot is not None
        assert context is not None
        assert state is not None

        assert context.last_snapshot is snapshot
        assert state.last_snapshot is snapshot

        assert state.snapshots_built >= 1
        assert stats.snapshots_built >= 1

        assert LiquidityTopics.ANALYTICS_LIQUIDITY_MAP_UPDATED in _published_topics(
            fake_event_bus
        )
        assert LiquidityTopics.ANALYTICS_LIQUIDITY_SIGNAL_UPDATED in _published_topics(
            fake_event_bus
        )

    @pytest.mark.asyncio
    async def test_candle_event_does_not_build_snapshot_when_not_enough_candles(
        self,
        liquidity_service: LiquidityService,
        too_few_candles: list[dict[str, Any]],
        fake_event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        await liquidity_service.start()

        await _feed_candles(liquidity_service, too_few_candles)

        snapshot = liquidity_service.get_last_snapshot(symbol, timeframe)

        assert snapshot is None
        assert LiquidityTopics.ANALYTICS_LIQUIDITY_MAP_UPDATED not in _published_topics(
            fake_event_bus
        )

    @pytest.mark.asyncio
    async def test_candle_context_is_trimmed_to_max_candles_per_context(
        self,
        liquidity_service: LiquidityService,
        liquidity_config,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        liquidity_config.max_candles_per_context = 5

        await liquidity_service.start()

        await _feed_candles(
            liquidity_service,
            candles_without_clear_equal_levels,
        )

        context = liquidity_service.get_context(symbol, timeframe)

        assert context is not None
        assert len(context.candles) == 5
        assert context.candles[-1] == candles_without_clear_equal_levels[-1]

    @pytest.mark.asyncio
    async def test_invalid_candle_payload_records_error(
        self,
        liquidity_service: LiquidityService,
    ) -> None:
        await liquidity_service.start()

        await liquidity_service.on_candle_closed({"bad": "payload"})

        stats = liquidity_service.get_stats()

        assert stats.errors_count == 1
        assert stats.last_error is not None
        assert "symbol" in stats.last_error
        assert stats.last_error_at is not None


# ---------------------------------------------------------------------
# Orderbook / price handling
# ---------------------------------------------------------------------


class TestLiquidityServiceMarketUpdates:
    @pytest.mark.asyncio
    async def test_orderbook_event_updates_context_and_state(
        self,
        liquidity_service: LiquidityService,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        balanced_orderbook: dict[str, list[list[float]]],
        symbol: str,
        timeframe: str,
    ) -> None:
        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        await liquidity_service.on_orderbook_updated(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "bids": balanced_orderbook["bids"],
                "asks": balanced_orderbook["asks"],
                "current_price": 100.0,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        context = liquidity_service.get_context(symbol, timeframe)
        state = liquidity_service.get_state().get(symbol, timeframe)
        stats = liquidity_service.get_stats()

        assert context is not None
        assert state is not None

        assert context.orderbook["bids"] == balanced_orderbook["bids"]
        assert context.orderbook["asks"] == balanced_orderbook["asks"]
        assert context.current_price == pytest.approx(100.0)

        assert state.processed_orderbook_updates == 1
        assert stats.orderbook_events_processed == 1

    @pytest.mark.asyncio
    async def test_orderbook_rebuild_can_be_disabled(
        self,
        liquidity_service: LiquidityService,
        liquidity_config,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        balanced_orderbook: dict[str, list[list[float]]],
        fake_event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        liquidity_config.rebuild_on_orderbook_updates = False

        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        before = liquidity_service.get_stats().snapshots_built

        fake_event_bus.published_events.clear()

        await liquidity_service.on_orderbook_updated(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "bids": balanced_orderbook["bids"],
                "asks": balanced_orderbook["asks"],
                "current_price": 100.0,
            }
        )

        after = liquidity_service.get_stats().snapshots_built

        assert after == before
        assert LiquidityTopics.ANALYTICS_LIQUIDITY_MAP_UPDATED not in _published_topics(
            fake_event_bus
        )

    @pytest.mark.asyncio
    async def test_price_event_updates_context_and_state_for_specific_timeframe(
        self,
        liquidity_service: LiquidityService,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        await liquidity_service.on_price_updated(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "price": 101.25,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        context = liquidity_service.get_context(symbol, timeframe)
        state = liquidity_service.get_state().get(symbol, timeframe)
        stats = liquidity_service.get_stats()

        assert context is not None
        assert state is not None

        assert context.current_price == pytest.approx(101.25)
        assert state.processed_price_updates == 1
        assert stats.price_events_processed == 1

    @pytest.mark.asyncio
    async def test_price_event_without_timeframe_updates_all_symbol_contexts(
        self,
        liquidity_service: LiquidityService,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
    ) -> None:
        await liquidity_service.start()

        candles_1m = [
            {**candle, "timeframe": "1m"}
            for candle in candles_without_clear_equal_levels
        ]
        candles_5m = [
            {**candle, "timeframe": "5m"}
            for candle in candles_without_clear_equal_levels
        ]

        await _feed_candles(liquidity_service, candles_1m)
        await _feed_candles(liquidity_service, candles_5m)

        await liquidity_service.on_price_updated(
            {
                "symbol": symbol,
                "price": 102.50,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        context_1m = liquidity_service.get_context(symbol, "1m")
        context_5m = liquidity_service.get_context(symbol, "5m")

        assert context_1m is not None
        assert context_5m is not None

        assert context_1m.current_price == pytest.approx(102.50)
        assert context_5m.current_price == pytest.approx(102.50)

        state_1m = liquidity_service.get_state().get(symbol, "1m")
        state_5m = liquidity_service.get_state().get(symbol, "5m")

        assert state_1m is not None
        assert state_5m is not None

        assert state_1m.processed_price_updates == 1
        assert state_5m.processed_price_updates == 1

    @pytest.mark.asyncio
    async def test_invalid_price_event_records_error(
        self,
        liquidity_service: LiquidityService,
        symbol: str,
        timeframe: str,
    ) -> None:
        await liquidity_service.start()

        await liquidity_service.on_price_updated(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "price": 0.0,
            }
        )

        stats = liquidity_service.get_stats()

        assert stats.price_events_processed == 1
        assert stats.errors_count == 1
        assert stats.last_error is not None
        assert "price" in stats.last_error


# ---------------------------------------------------------------------
# Explicit rebuild / snapshot publishing
# ---------------------------------------------------------------------


class TestLiquidityServiceRebuildAndPublishing:
    @pytest.mark.asyncio
    async def test_explicit_rebuild_returns_none_when_context_missing(
        self,
        liquidity_service: LiquidityService,
        symbol: str,
        timeframe: str,
    ) -> None:
        await liquidity_service.start()

        snapshot = await liquidity_service.rebuild_snapshot(
            symbol=symbol,
            timeframe=timeframe,
            force=True,
        )

        assert snapshot is None

    @pytest.mark.asyncio
    async def test_explicit_rebuild_applies_snapshot_to_context_and_state(
        self,
        liquidity_service: LiquidityService,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        buy_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster,
        fake_event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        fake_event_bus.published_events.clear()

        snapshot = await liquidity_service.rebuild_snapshot(
            symbol=symbol,
            timeframe=timeframe,
            extra_levels=buy_side_levels,
            extra_clusters=[buy_side_stop_cluster],
            force=True,
        )

        assert snapshot is not None

        context = liquidity_service.get_context(symbol, timeframe)
        state = liquidity_service.get_state().get(symbol, timeframe)

        assert context is not None
        assert state is not None

        assert context.last_snapshot is snapshot
        assert state.last_snapshot is snapshot

        assert liquidity_service.get_last_snapshot(symbol, timeframe) is snapshot

        assert LiquidityTopics.ANALYTICS_LIQUIDITY_MAP_UPDATED in _published_topics(
            fake_event_bus
        )
        assert LiquidityTopics.ANALYTICS_LIQUIDITY_SIGNAL_UPDATED in _published_topics(
            fake_event_bus
        )

    @pytest.mark.asyncio
    async def test_publish_events_false_disables_snapshot_event_emits(
        self,
        liquidity_service: LiquidityService,
        liquidity_config,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        fake_event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        liquidity_config.publish_events = False

        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        assert liquidity_service.get_last_snapshot(symbol, timeframe) is not None
        assert fake_event_bus.published_events == []

    @pytest.mark.asyncio
    async def test_emit_flags_disable_specific_event_groups(
        self,
        liquidity_service: LiquidityService,
        liquidity_config,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        fake_event_bus,
    ) -> None:
        liquidity_config.emit_map_updates = False
        liquidity_config.emit_level_events = False
        liquidity_config.emit_cluster_events = False
        liquidity_config.emit_signal_events = False

        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        topics = _published_topics(fake_event_bus)

        assert LiquidityTopics.ANALYTICS_LIQUIDITY_MAP_UPDATED not in topics
        assert LiquidityTopics.ANALYTICS_LIQUIDITY_LEVEL_DETECTED not in topics
        assert LiquidityTopics.ANALYTICS_LIQUIDITY_STOP_CLUSTER_DETECTED not in topics
        assert LiquidityTopics.ANALYTICS_LIQUIDITY_SIGNAL_UPDATED not in topics

    @pytest.mark.asyncio
    async def test_new_levels_and_clusters_emit_detection_events(
        self,
        liquidity_service: LiquidityService,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        buy_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster,
        fake_event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        fake_event_bus.published_events.clear()

        snapshot = await liquidity_service.rebuild_snapshot(
            symbol=symbol,
            timeframe=timeframe,
            extra_levels=buy_side_levels,
            extra_clusters=[buy_side_stop_cluster],
            force=True,
        )

        assert snapshot is not None

        topics = _published_topics(fake_event_bus)

        assert LiquidityTopics.ANALYTICS_LIQUIDITY_LEVEL_DETECTED in topics
        assert LiquidityTopics.ANALYTICS_LIQUIDITY_STOP_CLUSTER_DETECTED in topics

    @pytest.mark.asyncio
    async def test_sweep_change_emits_sweep_event(
        self,
        liquidity_service: LiquidityService,
        complete_snapshot: LiquidityMapSnapshot,
        fake_event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        await liquidity_service.start()

        context = liquidity_service._get_or_create_context(symbol, timeframe)
        context.current_price = complete_snapshot.current_price
        context.candles = [{"close": complete_snapshot.current_price}] * 10

        previous_snapshot = deepcopy(complete_snapshot)
        current_snapshot = deepcopy(complete_snapshot)

        previous_level = deepcopy(previous_snapshot.active_levels[0])
        previous_level.sweep_status = SweepStatus.NOT_SWEPT
        previous_level.status = LiquidityStatus.ACTIVE
        previous_level.swept_at = None

        current_level = deepcopy(previous_level)
        current_level.mark_partially_swept()

        previous_snapshot.active_levels = [previous_level]
        current_snapshot.active_levels = [current_level]

        await liquidity_service._apply_snapshot(
            context=context,
            snapshot=previous_snapshot,
        )

        fake_event_bus.published_events.clear()

        await liquidity_service._apply_snapshot(
            context=context,
            snapshot=current_snapshot,
        )

        assert LiquidityTopics.ANALYTICS_LIQUIDITY_LEVEL_SWEPT in _published_topics(
            fake_event_bus
        )


# ---------------------------------------------------------------------
# Scheduler jobs: cleanup / metrics / healthcheck
# ---------------------------------------------------------------------


class TestLiquidityServiceMaintenance:
    @pytest.mark.asyncio
    async def test_cleanup_removes_empty_contexts(
        self,
        liquidity_service: LiquidityService,
        symbol: str,
        timeframe: str,
    ) -> None:
        await liquidity_service.start()

        liquidity_service._get_or_create_context(symbol, timeframe)

        assert liquidity_service.get_context(symbol, timeframe) is not None

        await liquidity_service._cleanup()

        assert liquidity_service.get_context(symbol, timeframe) is None
        assert liquidity_service.get_stats().cleanup_runs == 1
        assert liquidity_service.get_stats().removed_empty_contexts >= 1

    @pytest.mark.asyncio
    async def test_cleanup_prunes_state_and_removes_inactive_levels(
        self,
        liquidity_service: LiquidityService,
        complete_snapshot: LiquidityMapSnapshot,
        symbol: str,
        timeframe: str,
    ) -> None:
        await liquidity_service.start()

        state = liquidity_service.get_state().apply_snapshot(complete_snapshot)

        inactive_level = deepcopy(complete_snapshot.active_levels[0])
        inactive_level.mark_swept()

        state.active_levels.append(inactive_level)

        before = len(state.active_levels)

        await liquidity_service._cleanup()

        after = len(state.active_levels)

        assert after < before
        assert liquidity_service.get_stats().cleanup_runs == 1
        assert liquidity_service.get_stats().removed_inactive_levels >= 1

    @pytest.mark.asyncio
    async def test_emit_state_metrics_publishes_metrics_payload(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
    ) -> None:
        await liquidity_service.start()

        await liquidity_service._emit_state_metrics()

        events = _events_for(
            fake_event_bus,
            LiquidityTopics.ANALYTICS_LIQUIDITY_STATE_METRICS,
        )

        assert events
        assert liquidity_service.get_stats().emitted_metrics_events == 1

        payload = events[-1].payload

        assert payload["service"] == "analytics_liquidity"
        assert "timestamp" in payload
        assert "stats" in payload
        assert "state" in payload
        assert "contexts_count" in payload

    @pytest.mark.asyncio
    async def test_emit_state_metrics_skips_when_not_running(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
    ) -> None:
        await liquidity_service._emit_state_metrics()

        assert _events_for(
            fake_event_bus,
            LiquidityTopics.ANALYTICS_LIQUIDITY_STATE_METRICS,
        ) == []

    @pytest.mark.asyncio
    async def test_emit_healthcheck_publishes_health_payload(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
    ) -> None:
        await liquidity_service.start()

        await liquidity_service._emit_healthcheck()

        events = _events_for(
            fake_event_bus,
            LiquidityTopics.ANALYTICS_LIQUIDITY_HEALTHCHECK,
        )

        assert events
        assert liquidity_service.get_stats().emitted_healthcheck_events == 1

        payload = events[-1].payload

        assert payload["service"] == "analytics_liquidity"
        assert payload["running"] is True
        assert payload["registered"] is True
        assert "contexts_count" in payload
        assert "states_count" in payload
        assert "subscriptions" in payload
        assert "scheduler_jobs" in payload
        assert "errors_count" in payload

    @pytest.mark.asyncio
    async def test_emit_healthcheck_skips_when_publish_events_false(
        self,
        liquidity_service: LiquidityService,
        liquidity_config,
        fake_event_bus,
    ) -> None:
        liquidity_config.publish_events = False

        await liquidity_service.start()
        await liquidity_service._emit_healthcheck()

        assert _events_for(
            fake_event_bus,
            LiquidityTopics.ANALYTICS_LIQUIDITY_HEALTHCHECK,
        ) == []

    @pytest.mark.asyncio
    async def test_safe_emit_records_error_when_event_bus_emit_fails(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
    ) -> None:
        await liquidity_service.start()

        async def failing_emit(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("emit failed")

        fake_event_bus.emit = failing_emit

        await liquidity_service._safe_emit(
            topic=LiquidityTopics.ANALYTICS_LIQUIDITY_HEALTHCHECK,
            payload={"ok": True},
        )

        stats = liquidity_service.get_stats()

        assert stats.errors_count == 1
        assert stats.last_error == "emit failed"
        assert stats.last_error_at is not None


# ---------------------------------------------------------------------
# Disabled service behavior
# ---------------------------------------------------------------------


class TestLiquidityServiceDisabled:
    @pytest.mark.asyncio
    async def test_handlers_skip_when_service_not_running(
        self,
        liquidity_service: LiquidityService,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        candle = candles_without_clear_equal_levels[0]

        await liquidity_service.on_candle_closed(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "candle": candle,
                "current_price": candle["close"],
            }
        )

        assert liquidity_service.get_context(symbol, timeframe) is None
        assert liquidity_service.get_stats().candle_events_processed == 0

    @pytest.mark.asyncio
    async def test_handlers_skip_when_config_disabled(
        self,
        liquidity_service: LiquidityService,
        liquidity_config,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        liquidity_config.enabled = False

        await liquidity_service.start()

        candle = candles_without_clear_equal_levels[0]

        await liquidity_service.on_candle_closed(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "candle": candle,
                "current_price": candle["close"],
            }
        )

        assert liquidity_service.get_context(symbol, timeframe) is None
        assert liquidity_service.get_stats().candle_events_processed == 0

    @pytest.mark.asyncio
    async def test_rebuild_snapshot_respects_context_but_returns_none_if_not_buildable(
        self,
        liquidity_service: LiquidityService,
        too_few_candles: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        await liquidity_service.start()

        await _feed_candles(liquidity_service, too_few_candles)

        snapshot = await liquidity_service.rebuild_snapshot(
            symbol=symbol,
            timeframe=timeframe,
            force=False,
        )

        assert snapshot is None