# tests/analytics/liquidity/test_liquidity_service.py

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from analytics.liquidity.config import (
    DEFAULT_CANDLE_CLOSED_TOPIC,
    DEFAULT_CANDLES_UPDATED_TOPIC,
    DEFAULT_ORDERBOOK_UPDATED_TOPIC,
    DEFAULT_PRICE_UPDATED_TOPIC,
    DEFAULT_RAW_CANDLE_TOPIC,
    LiquidityConfig,
)
from analytics.liquidity.enums import LiquidityStatus, SweepStatus
from analytics.liquidity.liquidity_map import LiquidityMap
from analytics.liquidity.liquidity_service import LiquidityService
from analytics.liquidity.models import (
    LiquidityLevel,
    LiquidityMapSnapshot,
    liquidity_key_to_dict,
)


# ---------------------------------------------------------------------
# Canonical test scope
# ---------------------------------------------------------------------


TEST_EXCHANGE = "binance"
TEST_MARKET_TYPE = "usdm_futures"
ALT_EXCHANGE = "bybit"
ALT_MARKET_TYPE = "linear"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _patch_fake_scheduler_for_service(fake_scheduler: Any) -> None:
    """
    Поточний LiquidityService очікує Scheduler.get_job_by_name(),
    а старий FakeScheduler із conftest цього методу не має.

    Патчимо тільки тестовий double, не production-код.
    """

    if hasattr(fake_scheduler, "get_job_by_name"):
        return

    def get_job_by_name(name: str) -> Any | None:
        for job_id, job in fake_scheduler.jobs.items():
            if job.name == name:
                return SimpleNamespace(job_id=job_id, id=job_id, name=job.name)
        return None

    fake_scheduler.get_job_by_name = get_job_by_name


def _patch_fake_event_bus_emit_returns_true(fake_event_bus: Any) -> None:
    """
    LiquidityService._safe_emit() трактує return value EventBus.emit()
    як accepted flag. Старий FakeEventBus.emit() додає event, але повертає None.
    Для service tests робимо fake ближчим до production-контракту.
    """

    if getattr(fake_event_bus, "_returns_true_patch_applied", False):
        return

    original_emit = fake_event_bus.emit

    async def emit_and_accept(
        topic: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        await original_emit(topic, payload, **kwargs)
        return True

    fake_event_bus.emit = emit_and_accept
    fake_event_bus._returns_true_patch_applied = True


def _make_emit_fail_on_topic(fake_event_bus: Any, fail_topic: str) -> None:
    """
    Робить EventBus hostile: один topic падає на emit().
    Це перевіряє, що LiquidityService не втрачає snapshot/state,
    якщо downstream event publishing тимчасово зламався.
    """

    original_emit = fake_event_bus.emit

    async def emit_with_failure(
        topic: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        if topic == fail_topic:
            raise RuntimeError(f"synthetic emit failure for {topic}")
        result = await original_emit(topic, payload, **kwargs)
        return bool(result) if result is not None else True

    fake_event_bus.emit = emit_with_failure


def _prepare_service_test_doubles(
    *,
    fake_event_bus: Any | None = None,
    fake_scheduler: Any | None = None,
) -> None:
    if fake_event_bus is not None:
        _patch_fake_event_bus_emit_returns_true(fake_event_bus)

    if fake_scheduler is not None:
        _patch_fake_scheduler_for_service(fake_scheduler)


def _published_topics(fake_event_bus: Any) -> list[str]:
    return fake_event_bus.emitted_topics()


def _events_for(fake_event_bus: Any, topic: str) -> list[Any]:
    return fake_event_bus.events_for(topic)


def _scope(
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
    symbol: str,
    timeframe: str,
) -> dict[str, str]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def _scoped_candle_payload(
    candle: dict[str, Any],
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": candle["symbol"],
        "timeframe": candle["timeframe"],
        "candle": {
            **candle,
            "exchange": exchange,
            "market_type": market_type,
        },
        "current_price": candle["close"],
        "timestamp": candle["close_time"],
    }


async def _feed_candles(
    service: LiquidityService,
    candles: list[dict[str, Any]],
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> None:
    for candle in candles:
        await service.on_candle_closed(
            _scoped_candle_payload(
                candle,
                exchange=exchange,
                market_type=market_type,
            )
        )


def _get_context(
    service: LiquidityService,
    *,
    symbol: str,
    timeframe: str,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
):
    return service.get_context(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


def _get_state(
    service: LiquidityService,
    *,
    symbol: str,
    timeframe: str,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
):
    return service.get_state().get(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


def _get_last_snapshot(
    service: LiquidityService,
    *,
    symbol: str,
    timeframe: str,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> LiquidityMapSnapshot | None:
    return service.get_last_snapshot(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


def _assert_context_ready(
    service: LiquidityService,
    *,
    symbol: str,
    timeframe: str,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> None:
    context = _get_context(
        service,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    assert context is not None
    assert context.exchange == exchange
    assert context.market_type == market_type
    assert context.symbol == symbol
    assert context.timeframe == timeframe
    assert context.current_price is not None
    assert context.current_price > 0


def _scope_extra_levels(
    levels: list[LiquidityLevel],
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
    symbol: str,
    timeframe: str,
) -> list[LiquidityLevel]:
    scoped: list[LiquidityLevel] = []

    for level in levels:
        cloned = deepcopy(level)
        cloned.exchange = exchange
        cloned.market_type = market_type
        cloned.symbol = symbol
        cloned.timeframe = timeframe
        cloned.metadata["scope"] = _scope(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        scoped.append(cloned)

    return scoped


# ---------------------------------------------------------------------
# Lifecycle / registration
# ---------------------------------------------------------------------


class TestLiquidityServiceLifecycle:
    def test_register_subscribes_only_to_configured_production_input_topics(
        self,
        liquidity_service: LiquidityService,
        liquidity_config: LiquidityConfig,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        liquidity_service.register()

        assert liquidity_service.is_registered is True

        assert tuple(fake_event_bus.topics()) == liquidity_config.production_input_topics
        assert set(fake_event_bus.topics()) == {
            DEFAULT_CANDLE_CLOSED_TOPIC,
            DEFAULT_ORDERBOOK_UPDATED_TOPIC,
        }

        assert DEFAULT_PRICE_UPDATED_TOPIC not in fake_event_bus.topics()
        assert DEFAULT_RAW_CANDLE_TOPIC not in fake_event_bus.topics()

        assert set(fake_scheduler.job_names()) == set(
            liquidity_config.scheduler_job_names
        )

    def test_register_adds_price_topic_only_when_explicitly_enabled(
        self,
        fake_event_bus,
        fake_scheduler,
        liquidity_config: LiquidityConfig,
        liquidity_map: LiquidityMap,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        liquidity_config.allow_price_input_topics = True
        liquidity_config.price_input_topics = (DEFAULT_PRICE_UPDATED_TOPIC,)

        service = LiquidityService(
            event_bus=fake_event_bus,  # type: ignore[arg-type]
            scheduler=fake_scheduler,  # type: ignore[arg-type]
            config=liquidity_config,
            liquidity_map=liquidity_map,
        )

        service.register()

        assert DEFAULT_PRICE_UPDATED_TOPIC in fake_event_bus.topics()
        assert tuple(fake_event_bus.topics()) == liquidity_config.production_input_topics

    def test_register_is_idempotent(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

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
        liquidity_config: LiquidityConfig,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()

        assert liquidity_service.is_registered is True
        assert liquidity_service.is_running is True

        assert liquidity_service.get_stats().started_at is not None
        assert liquidity_service.get_stats().stopped_at is None

        assert tuple(fake_event_bus.topics()) == liquidity_config.production_input_topics
        assert set(fake_scheduler.job_names()) == set(
            liquidity_config.scheduler_job_names
        )

    @pytest.mark.asyncio
    async def test_start_is_idempotent(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

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
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()

        job_ids = set(fake_scheduler.jobs.keys())
        subscriptions_count = len(fake_event_bus.subscriptions)

        await liquidity_service.stop()

        assert liquidity_service.is_running is False
        assert liquidity_service.is_registered is False
        assert liquidity_service.get_stats().stopped_at is not None

        assert len(fake_event_bus.unsubscribed) == subscriptions_count
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
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

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
        liquidity_config: LiquidityConfig,
        fake_event_bus,
    ) -> None:
        _prepare_service_test_doubles(fake_event_bus=fake_event_bus)

        liquidity_service_without_scheduler.register()

        assert liquidity_service_without_scheduler.is_registered is True
        assert tuple(fake_event_bus.topics()) == liquidity_config.production_input_topics
        assert liquidity_service_without_scheduler._scheduler_job_ids == []


# ---------------------------------------------------------------------
# Topic guards / scope filters / isolation
# ---------------------------------------------------------------------


class TestLiquidityServiceTopicGuardsAndScope:
    def test_raw_market_topic_is_rejected_by_config_guard(
        self,
        fake_event_bus,
        fake_scheduler,
        liquidity_config: LiquidityConfig,
        liquidity_map: LiquidityMap,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        liquidity_config.candle_input_topics = (DEFAULT_RAW_CANDLE_TOPIC,)

        with pytest.raises(ValueError, match="Raw market topic"):
            LiquidityService(
                event_bus=fake_event_bus,  # type: ignore[arg-type]
                scheduler=fake_scheduler,  # type: ignore[arg-type]
                config=liquidity_config,
                liquidity_map=liquidity_map,
            )

    @pytest.mark.asyncio
    async def test_scope_filter_skips_disallowed_exchange_market_symbol_timeframe(
        self,
        liquidity_service: LiquidityService,
        liquidity_config: LiquidityConfig,
        fake_event_bus,
        fake_scheduler,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        liquidity_config.allowed_exchanges = {TEST_EXCHANGE}
        liquidity_config.allowed_market_types = {TEST_MARKET_TYPE}
        liquidity_config.allowed_symbols = {symbol}
        liquidity_config.allowed_timeframes = {timeframe}

        await liquidity_service.start()

        bad_scope_events = [
            (ALT_EXCHANGE, TEST_MARKET_TYPE, symbol, timeframe),
            (TEST_EXCHANGE, ALT_MARKET_TYPE, symbol, timeframe),
            (TEST_EXCHANGE, TEST_MARKET_TYPE, "ETHUSDT", timeframe),
            (TEST_EXCHANGE, TEST_MARKET_TYPE, symbol, "5m"),
        ]

        for exchange, market_type, event_symbol, event_timeframe in bad_scope_events:
            candle = {
                **candles_without_clear_equal_levels[0],
                "symbol": event_symbol,
                "timeframe": event_timeframe,
            }

            await liquidity_service.on_candle_closed(
                _scoped_candle_payload(
                    candle,
                    exchange=exchange,
                    market_type=market_type,
                )
            )

        assert liquidity_service.get_stats().skipped_by_scope_filter == len(
            bad_scope_events
        )
        assert liquidity_service.get_state().count() == 0
        assert liquidity_service._contexts == {}

    @pytest.mark.asyncio
    async def test_same_symbol_timeframe_isolated_by_exchange_and_market_type(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()

        await _feed_candles(
            liquidity_service,
            candles_without_clear_equal_levels,
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
        )
        await _feed_candles(
            liquidity_service,
            candles_without_clear_equal_levels,
            exchange=ALT_EXCHANGE,
            market_type=ALT_MARKET_TYPE,
        )

        binance_context = _get_context(
            liquidity_service,
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
        )
        bybit_context = _get_context(
            liquidity_service,
            exchange=ALT_EXCHANGE,
            market_type=ALT_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
        )

        assert binance_context is not None
        assert bybit_context is not None
        assert binance_context.key != bybit_context.key
        assert binance_context.scope_key != bybit_context.scope_key

        assert _get_last_snapshot(
            liquidity_service,
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
        ) is not None
        assert _get_last_snapshot(
            liquidity_service,
            exchange=ALT_EXCHANGE,
            market_type=ALT_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
        ) is not None

        assert liquidity_service.get_state().count() == 2


# ---------------------------------------------------------------------
# Candle / candles.updated handling
# ---------------------------------------------------------------------


class TestLiquidityServiceCandleHandling:
    @pytest.mark.asyncio
    async def test_candle_events_update_context_and_state_without_snapshot_when_too_few(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
        too_few_candles: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()

        await _feed_candles(liquidity_service, too_few_candles)

        context = _get_context(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        )
        state = _get_state(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        )
        stats = liquidity_service.get_stats()

        assert context is not None
        assert state is not None

        assert len(context.candles) == len(too_few_candles)
        assert context.current_price == pytest.approx(too_few_candles[-1]["close"])

        assert state.processed_candles == len(too_few_candles)
        assert state.snapshots_built == 0
        assert stats.candle_events_processed == len(too_few_candles)
        assert stats.skipped_not_enough_data >= 1

        assert _get_last_snapshot(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        ) is None
        assert liquidity_config_topic_absent(
            fake_event_bus,
            topic=liquidity_service._config.map_updated_topic,
        )

    @pytest.mark.asyncio
    async def test_candle_events_build_snapshot_when_context_is_ready(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()

        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        snapshot = _get_last_snapshot(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        )
        context = _get_context(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        )
        state = _get_state(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        )
        topics = _published_topics(fake_event_bus)

        assert snapshot is not None
        assert context is not None
        assert state is not None

        assert context.last_snapshot is snapshot
        assert state.last_snapshot is snapshot

        assert state.snapshots_built >= 1
        assert liquidity_service.get_stats().snapshots_built >= 1

        assert liquidity_service._config.map_updated_topic in topics
        assert liquidity_service._config.signal_updated_topic in topics

        payload = _events_for(
            fake_event_bus,
            liquidity_service._config.map_updated_topic,
        )[-1].payload

        assert payload["exchange"] == TEST_EXCHANGE
        assert payload["market_type"] == TEST_MARKET_TYPE
        assert payload["symbol"] == symbol
        assert payload["timeframe"] == timeframe
        assert payload["scope"] == _scope(symbol=symbol, timeframe=timeframe)
        assert payload["scope_key"] == (
            f"{TEST_EXCHANGE}:{TEST_MARKET_TYPE}:{symbol}:{timeframe}"
        )

    @pytest.mark.asyncio
    async def test_candle_context_is_trimmed_to_max_candles_per_context(
        self,
        liquidity_service: LiquidityService,
        liquidity_config: LiquidityConfig,
        fake_event_bus,
        fake_scheduler,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        liquidity_config.max_candles_per_context = 5

        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        context = _get_context(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        )

        assert context is not None
        assert len(context.candles) == 5
        assert context.candles[-1]["close"] == candles_without_clear_equal_levels[-1][
            "close"
        ]

    @pytest.mark.asyncio
    async def test_invalid_candle_payload_records_error_without_crashing_runtime(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()

        await liquidity_service.on_candle_closed(
            {
                "exchange": TEST_EXCHANGE,
                "market_type": TEST_MARKET_TYPE,
                "timeframe": "1m",
                "current_price": 100.0,
            }
        )

        stats = liquidity_service.get_stats()

        assert liquidity_service.is_running is True
        assert stats.errors_count == 1
        assert stats.last_error is not None
        assert "symbol" in stats.last_error
        assert stats.last_error_at is not None

    @pytest.mark.asyncio
    async def test_candles_updated_replaces_context_window_and_builds_snapshot(
        self,
        fake_event_bus,
        fake_scheduler,
        liquidity_config: LiquidityConfig,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        liquidity_config.candle_input_topics = ()
        liquidity_config.candles_updated_input_topics = (
            DEFAULT_CANDLES_UPDATED_TOPIC,
        )

        service = LiquidityService(
            event_bus=fake_event_bus,  # type: ignore[arg-type]
            scheduler=fake_scheduler,  # type: ignore[arg-type]
            config=liquidity_config,
            liquidity_map=liquidity_map,
        )

        await service.start()

        await service.on_candles_updated(
            {
                "exchange": TEST_EXCHANGE,
                "market_type": TEST_MARKET_TYPE,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": candles_without_clear_equal_levels,
                "current_price": candles_without_clear_equal_levels[-1]["close"],
                "timestamp": candles_without_clear_equal_levels[-1]["close_time"],
            }
        )

        context = _get_context(service, symbol=symbol, timeframe=timeframe)
        state = _get_state(service, symbol=symbol, timeframe=timeframe)

        assert context is not None
        assert state is not None

        assert context.candles == candles_without_clear_equal_levels
        assert context.current_price == pytest.approx(
            candles_without_clear_equal_levels[-1]["close"]
        )
        assert state.processed_candles == 1
        assert service.get_stats().candles_updated_events_processed == 1

        assert _get_last_snapshot(service, symbol=symbol, timeframe=timeframe) is not None
        assert service._config.map_updated_topic in _published_topics(fake_event_bus)

    @pytest.mark.asyncio
    async def test_candles_updated_rejects_non_sequence_candles_payload(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()

        await liquidity_service.on_candles_updated(
            {
                "exchange": TEST_EXCHANGE,
                "market_type": TEST_MARKET_TYPE,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": 12345,
                "current_price": 100.0,
            }
        )

        stats = liquidity_service.get_stats()

        assert stats.errors_count == 1
        assert stats.last_error is not None
        assert "candles payload" in stats.last_error


def liquidity_config_topic_absent(fake_event_bus: Any, *, topic: str) -> bool:
    return topic not in _published_topics(fake_event_bus)


# ---------------------------------------------------------------------
# Orderbook / price handling
# ---------------------------------------------------------------------


class TestLiquidityServiceMarketUpdates:
    @pytest.mark.asyncio
    async def test_orderbook_update_without_existing_candle_context_is_skipped(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
        balanced_orderbook: dict[str, list[list[float]]],
        symbol: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()

        await liquidity_service.on_orderbook_updated(
            {
                "exchange": TEST_EXCHANGE,
                "market_type": TEST_MARKET_TYPE,
                "symbol": symbol,
                "bids": balanced_orderbook["bids"],
                "asks": balanced_orderbook["asks"],
                "timestamp": datetime.now(timezone.utc),
            }
        )

        assert liquidity_service.get_stats().orderbook_events_processed == 1
        assert liquidity_service.get_stats().skipped_no_context == 1
        assert liquidity_service.get_state().count() == 0
        assert fake_event_bus.published_events == []

    @pytest.mark.asyncio
    async def test_orderbook_event_updates_existing_contexts_for_same_market(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        balanced_orderbook: dict[str, list[list[float]]],
        symbol: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

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

        before = liquidity_service.get_stats().snapshots_built

        await liquidity_service.on_orderbook_updated(
            {
                "exchange": TEST_EXCHANGE,
                "market_type": TEST_MARKET_TYPE,
                "symbol": symbol,
                "bids": balanced_orderbook["bids"],
                "asks": balanced_orderbook["asks"],
                "current_price": 100.0,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        context_1m = _get_context(liquidity_service, symbol=symbol, timeframe="1m")
        context_5m = _get_context(liquidity_service, symbol=symbol, timeframe="5m")

        assert context_1m is not None
        assert context_5m is not None

        assert context_1m.orderbook["bids"] == balanced_orderbook["bids"]
        assert context_1m.orderbook["asks"] == balanced_orderbook["asks"]
        assert context_5m.orderbook["bids"] == balanced_orderbook["bids"]
        assert context_5m.orderbook["asks"] == balanced_orderbook["asks"]

        assert context_1m.current_price == pytest.approx(100.0)
        assert context_5m.current_price == pytest.approx(100.0)

        assert _get_state(liquidity_service, symbol=symbol, timeframe="1m").processed_orderbook_updates == 1
        assert _get_state(liquidity_service, symbol=symbol, timeframe="5m").processed_orderbook_updates == 1

        assert liquidity_service.get_stats().orderbook_events_processed == 1
        assert liquidity_service.get_stats().snapshots_built >= before

    @pytest.mark.asyncio
    async def test_orderbook_rebuild_can_be_disabled_without_losing_context_update(
        self,
        liquidity_service: LiquidityService,
        liquidity_config: LiquidityConfig,
        fake_event_bus,
        fake_scheduler,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        balanced_orderbook: dict[str, list[list[float]]],
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        liquidity_config.rebuild_on_orderbook_updates = False

        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        before = liquidity_service.get_stats().snapshots_built
        fake_event_bus.published_events.clear()

        await liquidity_service.on_orderbook_updated(
            {
                "exchange": TEST_EXCHANGE,
                "market_type": TEST_MARKET_TYPE,
                "symbol": symbol,
                "bids": balanced_orderbook["bids"],
                "asks": balanced_orderbook["asks"],
                "current_price": 100.0,
            }
        )

        context = _get_context(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        )

        assert context is not None
        assert context.orderbook["bids"] == balanced_orderbook["bids"]
        assert context.orderbook["asks"] == balanced_orderbook["asks"]

        assert liquidity_service.get_stats().snapshots_built == before
        assert liquidity_service._config.map_updated_topic not in _published_topics(
            fake_event_bus
        )

    @pytest.mark.asyncio
    async def test_price_event_is_ignored_when_price_topics_are_disabled(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        context = _get_context(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        )
        assert context is not None

        previous_price = context.current_price

        await liquidity_service.on_price_updated(
            {
                "exchange": TEST_EXCHANGE,
                "market_type": TEST_MARKET_TYPE,
                "symbol": symbol,
                "timeframe": timeframe,
                "price": 999.0,
            }
        )

        assert context.current_price == previous_price
        assert liquidity_service.get_stats().price_events_processed == 0

    @pytest.mark.asyncio
    async def test_price_event_updates_specific_timeframe_when_explicitly_enabled(
        self,
        fake_event_bus,
        fake_scheduler,
        liquidity_config: LiquidityConfig,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        liquidity_config.allow_price_input_topics = True
        liquidity_config.price_input_topics = (DEFAULT_PRICE_UPDATED_TOPIC,)

        service = LiquidityService(
            event_bus=fake_event_bus,  # type: ignore[arg-type]
            scheduler=fake_scheduler,  # type: ignore[arg-type]
            config=liquidity_config,
            liquidity_map=liquidity_map,
        )

        await service.start()
        await _feed_candles(service, candles_without_clear_equal_levels)

        fake_event_bus.published_events.clear()

        await service.on_price_updated(
            {
                "exchange": TEST_EXCHANGE,
                "market_type": TEST_MARKET_TYPE,
                "symbol": symbol,
                "timeframe": timeframe,
                "price": 101.25,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        context = _get_context(service, symbol=symbol, timeframe=timeframe)
        state = _get_state(service, symbol=symbol, timeframe=timeframe)

        assert context is not None
        assert state is not None

        assert context.current_price == pytest.approx(101.25)
        assert state.processed_price_updates == 1
        assert service.get_stats().price_events_processed == 1

    @pytest.mark.asyncio
    async def test_price_event_without_timeframe_updates_all_contexts_for_market_only(
        self,
        fake_event_bus,
        fake_scheduler,
        liquidity_config: LiquidityConfig,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        liquidity_config.allow_price_input_topics = True
        liquidity_config.price_input_topics = (DEFAULT_PRICE_UPDATED_TOPIC,)

        service = LiquidityService(
            event_bus=fake_event_bus,  # type: ignore[arg-type]
            scheduler=fake_scheduler,  # type: ignore[arg-type]
            config=liquidity_config,
            liquidity_map=liquidity_map,
        )

        await service.start()

        candles_1m = [
            {**candle, "timeframe": "1m"}
            for candle in candles_without_clear_equal_levels
        ]
        candles_5m = [
            {**candle, "timeframe": "5m"}
            for candle in candles_without_clear_equal_levels
        ]

        await _feed_candles(service, candles_1m)
        await _feed_candles(service, candles_5m)
        await _feed_candles(
            service,
            candles_1m,
            exchange=ALT_EXCHANGE,
            market_type=ALT_MARKET_TYPE,
        )

        await service.on_price_updated(
            {
                "exchange": TEST_EXCHANGE,
                "market_type": TEST_MARKET_TYPE,
                "symbol": symbol,
                "price": 102.50,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        assert _get_context(service, symbol=symbol, timeframe="1m").current_price == pytest.approx(102.50)
        assert _get_context(service, symbol=symbol, timeframe="5m").current_price == pytest.approx(102.50)

        alt_context = _get_context(
            service,
            exchange=ALT_EXCHANGE,
            market_type=ALT_MARKET_TYPE,
            symbol=symbol,
            timeframe="1m",
        )
        assert alt_context is not None
        assert alt_context.current_price != pytest.approx(102.50)

    @pytest.mark.asyncio
    async def test_invalid_price_event_records_error_before_processing_counter(
        self,
        fake_event_bus,
        fake_scheduler,
        liquidity_config: LiquidityConfig,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        liquidity_config.allow_price_input_topics = True
        liquidity_config.price_input_topics = (DEFAULT_PRICE_UPDATED_TOPIC,)

        service = LiquidityService(
            event_bus=fake_event_bus,  # type: ignore[arg-type]
            scheduler=fake_scheduler,  # type: ignore[arg-type]
            config=liquidity_config,
            liquidity_map=liquidity_map,
        )

        await service.start()
        await _feed_candles(service, candles_without_clear_equal_levels)

        await service.on_price_updated(
            {
                "exchange": TEST_EXCHANGE,
                "market_type": TEST_MARKET_TYPE,
                "symbol": symbol,
                "timeframe": timeframe,
                "price": 0.0,
            }
        )

        stats = service.get_stats()

        assert stats.price_events_processed == 0
        assert stats.errors_count == 1
        assert stats.last_error is not None
        assert "price" in stats.last_error


# ---------------------------------------------------------------------
# Explicit rebuild / publishing
# ---------------------------------------------------------------------


class TestLiquidityServiceRebuildAndPublishing:
    @pytest.mark.asyncio
    async def test_explicit_rebuild_returns_none_when_context_missing(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()

        snapshot = await liquidity_service.rebuild_snapshot(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            force=True,
        )

        assert snapshot is None
        assert liquidity_service.get_stats().skipped_no_context == 1

    @pytest.mark.asyncio
    async def test_explicit_rebuild_applies_snapshot_to_context_and_state(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        buy_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        fake_event_bus.published_events.clear()

        scoped_levels = _scope_extra_levels(
            buy_side_levels,
            symbol=symbol,
            timeframe=timeframe,
        )

        buy_cluster = deepcopy(buy_side_stop_cluster)
        buy_cluster.exchange = TEST_EXCHANGE
        buy_cluster.market_type = TEST_MARKET_TYPE
        buy_cluster.symbol = symbol
        buy_cluster.timeframe = timeframe

        snapshot = await liquidity_service.rebuild_snapshot(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            extra_levels=scoped_levels,
            extra_clusters=[buy_cluster],
            force=True,
        )

        assert snapshot is not None

        context = _get_context(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        )
        state = _get_state(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        )

        assert context is not None
        assert state is not None

        assert context.last_snapshot is snapshot
        assert state.last_snapshot is snapshot

        assert _get_last_snapshot(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        ) is snapshot

        topics = _published_topics(fake_event_bus)

        assert liquidity_service._config.map_updated_topic in topics
        assert liquidity_service._config.signal_updated_topic in topics
        assert liquidity_service._config.level_detected_topic in topics
        assert liquidity_service._config.stop_cluster_detected_topic in topics

    @pytest.mark.asyncio
    async def test_publish_events_false_disables_all_snapshot_event_emits(
        self,
        liquidity_service: LiquidityService,
        liquidity_config: LiquidityConfig,
        fake_event_bus,
        fake_scheduler,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        liquidity_config.publish_events = False

        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        assert _get_last_snapshot(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        ) is not None
        assert fake_event_bus.published_events == []

    @pytest.mark.asyncio
    async def test_emit_flags_disable_specific_snapshot_event_groups(
        self,
        liquidity_service: LiquidityService,
        liquidity_config: LiquidityConfig,
        fake_event_bus,
        fake_scheduler,
        candles_without_clear_equal_levels: list[dict[str, Any]],
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        liquidity_config.emit_map_updates = False
        liquidity_config.emit_level_events = False
        liquidity_config.emit_cluster_events = False
        liquidity_config.emit_signal_events = False

        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        topics = _published_topics(fake_event_bus)

        assert liquidity_service._config.map_updated_topic not in topics
        assert liquidity_service._config.level_detected_topic not in topics
        assert liquidity_service._config.stop_cluster_detected_topic not in topics
        assert liquidity_service._config.signal_updated_topic not in topics

    @pytest.mark.asyncio
    async def test_emit_failure_records_error_but_keeps_snapshot_and_state(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )
        _make_emit_fail_on_topic(
            fake_event_bus,
            liquidity_service._config.map_updated_topic,
        )

        await liquidity_service.start()
        await _feed_candles(liquidity_service, candles_without_clear_equal_levels)

        snapshot = _get_last_snapshot(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        )
        state = _get_state(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        )

        assert snapshot is not None
        assert state is not None
        assert state.last_snapshot is snapshot

        assert liquidity_service.get_stats().errors_count >= 1
        assert liquidity_service.get_stats().last_error is not None
        assert "synthetic emit failure" in liquidity_service.get_stats().last_error

    @pytest.mark.asyncio
    async def test_sweep_change_emits_sweep_event(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
        complete_snapshot: LiquidityMapSnapshot,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()

        context = liquidity_service._get_or_create_context_from_key(
            complete_snapshot.liquidity_key
        )
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

        assert liquidity_service._config.level_swept_topic in _published_topics(
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
        fake_event_bus,
        fake_scheduler,
        symbol: str,
        timeframe: str,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()

        key = liquidity_service.make_key(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
        )
        liquidity_service._get_or_create_context_from_key(key)

        assert _get_context(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        ) is not None

        await liquidity_service._cleanup()

        assert _get_context(
            liquidity_service,
            symbol=symbol,
            timeframe=timeframe,
        ) is None
        assert liquidity_service.get_stats().cleanup_runs == 1
        assert liquidity_service.get_stats().removed_empty_contexts >= 1

    @pytest.mark.asyncio
    async def test_cleanup_prunes_inactive_levels_from_state(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
        complete_snapshot: LiquidityMapSnapshot,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

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
    async def test_emit_state_metrics_publishes_scoped_diagnostics_payload(
        self,
        liquidity_service: LiquidityService,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()

        await liquidity_service._emit_state_metrics()

        events = _events_for(
            fake_event_bus,
            liquidity_service._config.state_metrics_topic,
        )

        assert events
        assert liquidity_service.get_stats().emitted_metrics_events == 1

        payload = events[-1].payload

        assert payload["service"] == "analytics_liquidity"
        assert payload["scope"] == "exchange:market_type:symbol:timeframe"
        assert "timestamp" in payload
        assert "stats" in payload
        assert "state" in payload
        assert "contexts_count" in payload
        assert "context_keys" in payload

    @pytest.mark.asyncio
    async def test_emit_healthcheck_publishes_runtime_contract_payload(
        self,
        liquidity_service: LiquidityService,
        liquidity_config: LiquidityConfig,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        await liquidity_service.start()

        await liquidity_service._emit_healthcheck()

        events = _events_for(
            fake_event_bus,
            liquidity_service._config.healthcheck_topic,
        )

        assert events
        assert liquidity_service.get_stats().emitted_healthcheck_events == 1

        payload = events[-1].payload

        assert payload["service"] == "analytics_liquidity"
        assert payload["running"] is True
        assert payload["registered"] is True
        assert payload["scope"] == "exchange:market_type:symbol:timeframe"
        assert payload["input_topics"] == list(liquidity_config.production_input_topics)
        assert payload["output_topics"] == list(liquidity_config.output_topics)
        assert payload["subscriptions"] == len(liquidity_service._subscriptions)
        assert payload["scheduler_jobs"] == len(liquidity_service._scheduler_job_ids)

    @pytest.mark.asyncio
    async def test_metrics_and_healthcheck_do_not_emit_when_publish_events_disabled(
        self,
        liquidity_service: LiquidityService,
        liquidity_config: LiquidityConfig,
        fake_event_bus,
        fake_scheduler,
    ) -> None:
        _prepare_service_test_doubles(
            fake_event_bus=fake_event_bus,
            fake_scheduler=fake_scheduler,
        )

        liquidity_config.publish_events = False

        await liquidity_service.start()

        await liquidity_service._emit_state_metrics()
        await liquidity_service._emit_healthcheck()

        assert fake_event_bus.published_events == []
        assert liquidity_service.get_stats().emitted_metrics_events == 0
        assert liquidity_service.get_stats().emitted_healthcheck_events == 0