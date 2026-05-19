# tests/strategy/test_strategy_engine_integration.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from core.event_bus import EventPriority

from strategy.config import StrategyConfig, StrategyRuntimeConfig
from strategy.engine import (
    StrategyContextBuilder,
    StrategyEngine,
    StrategyEngineStats,
    StrategyEventHandler,
    StrategyLifecycleManager,
    _event_name_from_event,
    _event_timestamp,
    _payload_from_event,
)
from strategy.enums import (
    FeatureSource,
    MarketRegime,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    Timeframe,
)
from strategy.exceptions import StrategyStateError
from strategy.models import (
    FeatureSnapshot,
    PortfolioSnapshot,
    PriceSnapshot,
    RegimeSnapshot,
    StrategyContext,
    StrategyEvaluation,
    utcnow,
)
from strategy.processor import ProcessedSignalBatch
from strategy.registry import StrategyRegistry
from strategy.state import StrategyRuntimeState

from conftest import DummyStrategy, FailingStrategy, MockEvent


# =============================================================================
# Local helpers / doubles
# =============================================================================


@dataclass(slots=True)
class MinimalEvent:
    payload: Any = None
    topic: str | None = None
    name: str | None = None
    event_name: str | None = None
    type: str | None = None
    timestamp: Any = None
    created_at: Any = None


class ProcessorStub:
    """
    Minimal SignalProcessor double для StrategyEngine orchestration tests.

    Не робить scoring/building/risk-payload logic — тільки повертає готовий batch
    або кидає exception, щоб перевірити engine-level stats/error handling.
    """

    component_name = "ProcessorStub"

    def __init__(
        self,
        *,
        batch: ProcessedSignalBatch | None = None,
        error: Exception | None = None,
    ) -> None:
        self.batch = batch
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self._started = False
        self._registered = False

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_registered(self) -> bool:
        return self._registered

    def register(self) -> None:
        self._registered = True

    async def start(self) -> None:
        self._registered = True
        self._started = True

    async def stop(self) -> None:
        self._started = False
        self._registered = False

    async def process_event(
        self,
        *,
        event_name: str,
        payload: dict[str, Any],
        timestamp: datetime | None = None,
        emit: bool = True,
    ) -> ProcessedSignalBatch:
        self.calls.append(
            {
                "event_name": event_name,
                "payload": dict(payload),
                "timestamp": timestamp,
                "emit": emit,
            }
        )

        if self.error is not None:
            raise self.error

        if self.batch is not None:
            return self.batch

        return ProcessedSignalBatch(
            symbol=payload.get("symbol", "BTCUSDT"),
            timestamp=timestamp or utcnow(),
            accepted=False,
            emitted=False,
            reasons=["stub_default"],
        )


class ComponentStub:
    component_name = "ComponentStub"

    def __init__(
        self,
        *,
        name: str,
        fail_stop: bool = False,
    ) -> None:
        self.component_name = name
        self.fail_stop = fail_stop

        self.register_calls = 0
        self.start_calls = 0
        self.stop_calls = 0

        self._registered = False
        self._started = False

    @property
    def is_registered(self) -> bool:
        return self._registered

    @property
    def is_started(self) -> bool:
        return self._started

    def register(self) -> None:
        self.register_calls += 1
        self._registered = True

    async def start(self) -> None:
        self.start_calls += 1
        self._registered = True
        self._started = True

    async def stop(self) -> None:
        self.stop_calls += 1
        if self.fail_stop:
            raise RuntimeError(f"{self.component_name} stop failed")
        self._started = False
        self._registered = False


def _accepted_batch(
    *,
    symbol: str = "BTCUSDT",
    timestamp: datetime | None = None,
) -> ProcessedSignalBatch:
    return ProcessedSignalBatch(
        symbol=symbol,
        timestamp=timestamp or utcnow(),
        accepted=True,
        emitted=True,
        final_signals=[],
        risk_payloads=[],
        reasons=[],
    )


def _rejected_batch(
    *,
    symbol: str = "BTCUSDT",
    timestamp: datetime | None = None,
    reasons: list[str] | None = None,
) -> ProcessedSignalBatch:
    return ProcessedSignalBatch(
        symbol=symbol,
        timestamp=timestamp or utcnow(),
        accepted=False,
        emitted=False,
        final_signals=[],
        risk_payloads=[],
        reasons=reasons or ["unit_rejected"],
    )


def _make_engine(
    *,
    config: StrategyConfig,
    event_bus=None,
    scheduler=None,
    processor=None,
    registry: StrategyRegistry | None = None,
    state: StrategyRuntimeState | None = None,
) -> StrategyEngine:
    return StrategyEngine(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
        registry=registry,
        state=state,
        processor=processor,
    )


# =============================================================================
# Event extraction helpers
# =============================================================================


class TestEngineEventExtractionHelpers:
    def test_payload_from_event_returns_dict_payload(self) -> None:
        event = MinimalEvent(payload={"symbol": "BTCUSDT"})

        assert _payload_from_event(event) == {"symbol": "BTCUSDT"}

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            [],
            "bad",
            123,
        ],
    )
    def test_payload_from_event_returns_empty_for_non_dict_payload(
        self,
        payload: Any,
    ) -> None:
        event = MinimalEvent(payload=payload)

        assert _payload_from_event(event) == {}

    @pytest.mark.parametrize(
        ("event", "expected"),
        [
            (MinimalEvent(topic="analytics.orderflow.updated"), "analytics.orderflow.updated"),
            (MinimalEvent(name="analytics.name"), "analytics.name"),
            (MinimalEvent(event_name="analytics.event_name"), "analytics.event_name"),
            (MinimalEvent(type="analytics.type"), "analytics.type"),
            (MinimalEvent(topic="  "), "unknown"),
            (MinimalEvent(), "unknown"),
        ],
    )
    def test_event_name_from_event(self, event: MinimalEvent, expected: str) -> None:
        assert _event_name_from_event(event) == expected

    def test_event_name_prefers_topic_over_other_fields(self) -> None:
        event = MinimalEvent(
            topic="analytics.topic",
            name="analytics.name",
            event_name="analytics.event_name",
            type="analytics.type",
        )

        assert _event_name_from_event(event) == "analytics.topic"

    def test_event_timestamp_from_datetime(self) -> None:
        raw = datetime(2026, 5, 20, 12, 0, 0)
        event = MinimalEvent(timestamp=raw)

        result = _event_timestamp(event)

        assert result == raw.replace(tzinfo=timezone.utc)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (
                1_764_156_000,
                datetime.fromtimestamp(1_764_156_000, tz=timezone.utc),
            ),
            (
                1_764_156_000_000,
                datetime.fromtimestamp(1_764_156_000, tz=timezone.utc),
            ),
        ],
    )
    def test_event_timestamp_from_seconds_or_milliseconds(
        self,
        raw: int,
        expected: datetime,
    ) -> None:
        event = MinimalEvent(timestamp=raw)

        assert _event_timestamp(event) == expected

    def test_event_timestamp_uses_created_at_fallback(self) -> None:
        raw = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        event = MinimalEvent(created_at=raw)

        assert _event_timestamp(event) == raw

    @pytest.mark.parametrize(
        "event",
        [
            None,
            MinimalEvent(),
            MinimalEvent(timestamp="bad"),
        ],
    )
    def test_event_timestamp_returns_none_for_missing_or_unsupported(
        self,
        event: Any,
    ) -> None:
        assert _event_timestamp(event) is None


# =============================================================================
# StrategyEngineStats
# =============================================================================


class TestStrategyEngineStats:
    def test_record_start_and_stop(self) -> None:
        stats = StrategyEngineStats()

        stats.record_start()

        assert stats.started_at is not None
        assert stats.stopped_at is None

        stats.record_stop()

        assert stats.stopped_at is not None

    def test_record_event_and_processed_batches(self) -> None:
        stats = StrategyEngineStats()

        stats.record_event_received()
        stats.record_processed(accepted=True)
        stats.record_processed(accepted=False)

        assert stats.events_received == 1
        assert stats.events_processed == 2
        assert stats.batches_accepted == 1
        assert stats.batches_rejected == 1
        assert stats.last_event_at is not None
        assert stats.last_processed_at is not None

    def test_record_error_trims_to_100_errors(self) -> None:
        stats = StrategyEngineStats()

        for index in range(105):
            stats.record_error(f"error-{index}")

        assert stats.events_failed == 105
        assert stats.last_error_at is not None
        assert len(stats.errors) == 100
        assert stats.errors[0] == "error-5"
        assert stats.errors[-1] == "error-104"

    def test_summary_serializes_datetimes_and_recent_errors(self) -> None:
        stats = StrategyEngineStats()

        stats.record_start()
        stats.record_event_received()
        stats.record_processed(accepted=True)

        for index in range(12):
            stats.record_error(f"error-{index}")

        summary = stats.summary()

        assert summary["events_received"] == 1
        assert summary["events_processed"] == 1
        assert summary["events_failed"] == 12
        assert summary["batches_accepted"] == 1
        assert summary["started_at"] is not None
        assert summary["last_event_at"] is not None
        assert summary["last_processed_at"] is not None
        assert summary["recent_errors"] == [f"error-{index}" for index in range(2, 12)]


# =============================================================================
# StrategyContextBuilder
# =============================================================================


class TestStrategyContextBuilder:
    def test_build_context_persists_features_domain_data_and_metadata(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        mock_scheduler,
        make_feature,
    ) -> None:
        state = StrategyRuntimeState()
        builder = StrategyContextBuilder(
            config=strategy_config,
            state=state,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        ts = datetime(2026, 5, 20, 12, 0, 0)
        feature = make_feature(
            name="orderflow_imbalance",
            symbol="BTCUSDT",
            freshness_seconds=30,
        )

        context = builder.build(
            symbol=" BTCUSDT ",
            timestamp=ts,
            timeframe=Timeframe.M5,
            features=[feature],
            domain_data={
                FeatureSource.ORDERFLOW: {
                    "pressure": 0.85,
                }
            },
            metadata={
                "exchange": "binance",
                "market_type": "usdm_futures",
            },
            persist=True,
        )

        assert context.symbol == "BTCUSDT"
        assert context.timestamp == ts.replace(tzinfo=timezone.utc)
        assert context.timeframe is Timeframe.M5
        assert context.get_feature("orderflow_imbalance") == feature.value
        assert context.freshness_map["orderflow_imbalance"] == 30
        assert context.domain_dict(FeatureSource.ORDERFLOW)["pressure"] == 0.85
        assert context.metadata["exchange"] == "binance"
        assert state.contexts.get_context("BTCUSDT") is context
        assert "BTCUSDT" in state.active_symbols

    def test_build_context_can_skip_persist(
        self,
        strategy_config: StrategyConfig,
        make_feature,
    ) -> None:
        state = StrategyRuntimeState()
        builder = StrategyContextBuilder(
            config=strategy_config,
            state=state,
        )

        context = builder.build(
            symbol="BTCUSDT",
            features=[make_feature(name="orderflow_imbalance")],
            persist=False,
        )

        assert context.symbol == "BTCUSDT"
        assert state.contexts.get_context("BTCUSDT") is None
        assert "BTCUSDT" not in state.active_symbols

    def test_build_context_rejects_empty_symbol(
        self,
        strategy_config: StrategyConfig,
    ) -> None:
        builder = StrategyContextBuilder(
            config=strategy_config,
            state=StrategyRuntimeState(),
        )

        with pytest.raises(StrategyStateError, match="symbol cannot be empty"):
            builder.build(symbol=" ")

    def test_get_or_build_returns_existing_and_updates_timestamp_timeframe(
        self,
        strategy_config: StrategyConfig,
        strategy_context: StrategyContext,
    ) -> None:
        state = StrategyRuntimeState()
        state.update_context(strategy_context)

        builder = StrategyContextBuilder(
            config=strategy_config,
            state=state,
        )
        new_ts = utcnow() + timedelta(seconds=10)

        context = builder.get_or_build(
            symbol=strategy_context.symbol,
            timestamp=new_ts,
            timeframe=Timeframe.M15,
        )

        assert context is strategy_context
        assert context.timestamp == new_ts
        assert context.timeframe is Timeframe.M15

    def test_get_or_build_creates_new_context_when_missing(
        self,
        strategy_config: StrategyConfig,
    ) -> None:
        state = StrategyRuntimeState()
        builder = StrategyContextBuilder(
            config=strategy_config,
            state=state,
        )

        context = builder.get_or_build(
            symbol="BTCUSDT",
            timeframe=Timeframe.M5,
        )

        assert context.symbol == "BTCUSDT"
        assert context.timeframe is Timeframe.M5
        assert state.contexts.get_context("BTCUSDT") is context

    def test_update_from_payload_updates_price_regime_and_metadata(
        self,
        strategy_config: StrategyConfig,
    ) -> None:
        state = StrategyRuntimeState()
        builder = StrategyContextBuilder(
            config=strategy_config,
            state=state,
        )

        context = builder.update_from_payload(
            event_name="system.strategy.context_update",
            payload={
                "symbol": "BTCUSDT",
                "timeframe": "5m",
                "timestamp": datetime(2026, 5, 20, 12, 0, 0),
                "price": 100.5,
                "bid": 100.4,
                "ask": 100.6,
                "mark_price": 100.55,
                "index_price": 100.52,
                "spread_bps": 2.0,
                "regime": MarketRegime.TRENDING_UP.value,
                "regime_confidence": 0.8,
            },
        )

        assert context.symbol == "BTCUSDT"
        assert context.timeframe is Timeframe.M5
        assert context.price is not None
        assert context.price.last == 100.5
        assert context.price.bid == 100.4
        assert context.price.ask == 100.6
        assert context.regime is not None
        assert context.regime.regime is MarketRegime.TRENDING_UP
        assert context.regime.confidence == 0.8
        assert context.metadata["source_event"] == "system.strategy.context_update"
        assert context.metadata["updated_by"] == "StrategyContextBuilder"
        assert state.contexts.get_context("BTCUSDT") is context

    def test_update_from_payload_accepts_portfolio_snapshot(
        self,
        strategy_config: StrategyConfig,
    ) -> None:
        state = StrategyRuntimeState()
        builder = StrategyContextBuilder(
            config=strategy_config,
            state=state,
        )
        portfolio = PortfolioSnapshot(
            timestamp=utcnow(),
            equity=10_000.0,
            available_balance=8_000.0,
            total_exposure=1_000.0,
            open_positions=2,
        )
        portfolio.validate()

        context = builder.update_from_payload(
            event_name="system.strategy.context_update",
            payload={
                "symbol": "BTCUSDT",
                "portfolio": portfolio,
            },
        )

        assert context.portfolio is portfolio

    @pytest.mark.parametrize(
        ("event_name", "payload", "match"),
        [
            ("", {"symbol": "BTCUSDT"}, "event_name cannot be empty"),
            ("system.strategy.context_update", ["bad"], "payload must be a dict"),
            ("system.strategy.context_update", {}, "valid symbol"),
            (
                "system.strategy.context_update",
                {"symbol": "BTCUSDT", "timestamp": object()},
                "unsupported timestamp",
            ),
        ],
    )
    def test_update_from_payload_invalid_inputs_raise(
        self,
        strategy_config: StrategyConfig,
        event_name: str,
        payload: Any,
        match: str,
    ) -> None:
        builder = StrategyContextBuilder(
            config=strategy_config,
            state=StrategyRuntimeState(),
        )

        with pytest.raises(StrategyStateError, match=match):
            builder.update_from_payload(
                event_name=event_name,
                payload=payload,
            )

    def test_persist_updates_state(
        self,
        strategy_config: StrategyConfig,
        strategy_context: StrategyContext,
    ) -> None:
        state = StrategyRuntimeState()
        builder = StrategyContextBuilder(
            config=strategy_config,
            state=state,
        )

        builder.persist(strategy_context)

        assert state.contexts.get_context(strategy_context.symbol) is strategy_context


# =============================================================================
# StrategyEventHandler
# =============================================================================


class TestStrategyEventHandler:
    def test_register_subscribes_to_configured_analytics_and_feedback_topics(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        processor = ProcessorStub()
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            processor=processor,  # type: ignore[arg-type]
        )
        handler = StrategyEventHandler(
            config=strategy_config,
            engine=engine,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        handler.register()

        topics = [subscription.topic for subscription in mock_event_bus.subscriptions]

        assert "analytics.orderflow.updated" in topics
        assert "analytics.open_interest.updated" in topics
        assert "analytics.funding.updated" in topics
        assert "analytics.hybrid.updated" in topics

        assert "signal.confirmed" in topics
        assert "risk.position_blocked" in topics
        assert "execution.order_rejected" in topics
        assert "execution.order_failed" in topics
        assert "execution.order_filled" in topics
        assert "execution.order_cancelled" in topics
        assert "system.strategy.context_update" in topics

        assert handler.is_registered
        assert handler.subscriptions_count == len(topics)

    def test_register_without_event_bus_marks_registered_without_subscriptions(
        self,
        strategy_config: StrategyConfig,
    ) -> None:
        engine = _make_engine(
            config=strategy_config,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )
        handler = StrategyEventHandler(
            config=strategy_config,
            engine=engine,
            event_bus=None,
        )

        handler.register()

        assert handler.is_registered
        assert handler.subscriptions_count == 0

    def test_register_is_idempotent(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )
        handler = StrategyEventHandler(
            config=strategy_config,
            engine=engine,
            event_bus=mock_event_bus,
        )

        handler.register()
        first_count = handler.subscriptions_count
        handler.register()

        assert handler.subscriptions_count == first_count
        assert mock_event_bus.subscribe_calls == first_count

    @pytest.mark.asyncio()
    async def test_unregister_unsubscribes_all_and_survives_failures(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )
        handler = StrategyEventHandler(
            config=strategy_config,
            engine=engine,
            event_bus=mock_event_bus,
        )
        handler.register()

        mock_event_bus.fail_unsubscribe = True

        await handler.stop()

        assert not handler.is_registered
        assert handler.subscriptions_count == 0
        assert mock_event_bus.unsubscribe_calls > 0

    @pytest.mark.asyncio()
    async def test_analytics_event_handler_delegates_to_engine(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        processor = ProcessorStub(batch=_accepted_batch())
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            processor=processor,  # type: ignore[arg-type]
        )
        handler = StrategyEventHandler(
            config=strategy_config,
            engine=engine,
            event_bus=mock_event_bus,
        )
        event = MockEvent(
            topic="analytics.orderflow.updated",
            payload={"symbol": "BTCUSDT"},
            source="unit",
        )

        await handler._handle_analytics_event(event)

        assert processor.calls
        assert processor.calls[0]["event_name"] == "analytics.orderflow.updated"
        assert processor.calls[0]["payload"] == {"symbol": "BTCUSDT"}

    @pytest.mark.asyncio()
    async def test_context_update_handler_delegates_to_engine(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )
        handler = StrategyEventHandler(
            config=strategy_config,
            engine=engine,
            event_bus=mock_event_bus,
        )
        event = MockEvent(
            topic="system.strategy.context_update",
            payload={
                "symbol": "BTCUSDT",
                "price": 100.0,
            },
            source="unit",
        )

        await handler._handle_context_update(event)

        context = engine.state.contexts.get_context("BTCUSDT")
        assert context is not None
        assert context.price is not None
        assert context.price.last == 100.0

    @pytest.mark.parametrize(
        ("topic", "status", "reason"),
        [
            ("signal.confirmed", SignalStatus.CONFIRMED, "risk_confirmed"),
            ("risk.position_blocked", SignalStatus.REJECTED, "risk_position_blocked"),
            ("execution.order_rejected", SignalStatus.REJECTED, "execution_order_rejected"),
            ("execution.order_failed", SignalStatus.FAILED, "execution_order_failed"),
            ("execution.order_filled", SignalStatus.EXECUTED, "execution_order_filled"),
            ("execution.order_cancelled", SignalStatus.CANCELLED, "execution_order_cancelled"),
        ],
    )
    @pytest.mark.asyncio()
    async def test_feedback_handlers_update_signal_status(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        risk_ready_strategy_signal,
        topic: str,
        status: SignalStatus,
        reason: str,
    ) -> None:
        state = StrategyRuntimeState()
        state.update_signal(risk_ready_strategy_signal)

        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            state=state,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )
        handler = StrategyEventHandler(
            config=strategy_config,
            engine=engine,
            event_bus=mock_event_bus,
        )
        event = MockEvent(
            topic=topic,
            payload={
                "signal_id": risk_ready_strategy_signal.signal_id,
                "symbol": risk_ready_strategy_signal.symbol,
            },
            source="unit",
        )

        if topic == "signal.confirmed":
            await handler._handle_signal_confirmed(event)
        elif topic == "risk.position_blocked":
            await handler._handle_signal_blocked(event)
        elif topic == "execution.order_rejected":
            await handler._handle_execution_rejected(event)
        elif topic == "execution.order_failed":
            await handler._handle_execution_failed(event)
        elif topic == "execution.order_filled":
            await handler._handle_execution_filled(event)
        elif topic == "execution.order_cancelled":
            await handler._handle_execution_cancelled(event)

        updated = state.signals.get_by_signal_id(risk_ready_strategy_signal.signal_id)

        assert updated is risk_ready_strategy_signal
        assert risk_ready_strategy_signal.status is status
        assert reason in risk_ready_strategy_signal.reasons

        if status.is_active:
            assert risk_ready_strategy_signal in state.signals.get_active()
        else:
            assert risk_ready_strategy_signal not in state.signals.get_active()

    @pytest.mark.asyncio()
    async def test_feedback_handler_uses_payload_reason_when_present(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        risk_ready_strategy_signal,
    ) -> None:
        state = StrategyRuntimeState()
        state.update_signal(risk_ready_strategy_signal)

        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            state=state,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )
        handler = StrategyEventHandler(
            config=strategy_config,
            engine=engine,
            event_bus=mock_event_bus,
        )
        event = MockEvent(
            topic="risk.position_blocked",
            payload={
                "signal_id": risk_ready_strategy_signal.signal_id,
                "reason": "max_drawdown_guard",
            },
        )

        await handler._handle_signal_blocked(event)

        assert risk_ready_strategy_signal.status is SignalStatus.REJECTED
        assert "max_drawdown_guard" in risk_ready_strategy_signal.reasons

    @pytest.mark.asyncio()
    async def test_feedback_handler_ignores_unknown_signal(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        state = StrategyRuntimeState()
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            state=state,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )
        handler = StrategyEventHandler(
            config=strategy_config,
            engine=engine,
            event_bus=mock_event_bus,
        )
        event = MockEvent(
            topic="signal.confirmed",
            payload={
                "signal_id": "missing",
                "symbol": "BTCUSDT",
            },
        )

        await handler._handle_signal_confirmed(event)

        assert state.signals.summary()["history"] == 0


# =============================================================================
# StrategyLifecycleManager
# =============================================================================


class TestStrategyLifecycleManager:
    @pytest.mark.asyncio()
    async def test_lifecycle_start_registers_starts_components_schedules_cleanup_and_emits_event(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        first = ComponentStub(name="first")
        second = ComponentStub(name="second")
        state = StrategyRuntimeState()

        lifecycle = StrategyLifecycleManager(
            config=strategy_config,
            state=state,
            components=[first, second],  # type: ignore[list-item]
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        await lifecycle.start()

        assert lifecycle.is_started
        assert first.register_calls == 1
        assert second.register_calls == 1
        assert first.start_calls == 1
        assert second.start_calls == 1
        assert len(mock_scheduler.added_jobs) == 1
        assert mock_scheduler.added_jobs[0].name == "strategy.runtime.cleanup"
        assert mock_event_bus.topic_emitted("strategy.engine.started")

    @pytest.mark.asyncio()
    async def test_lifecycle_start_is_idempotent(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        component = ComponentStub(name="component")
        lifecycle = StrategyLifecycleManager(
            config=strategy_config,
            state=StrategyRuntimeState(),
            components=[component],  # type: ignore[list-item]
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        await lifecycle.start()
        await lifecycle.start()

        assert component.start_calls == 1
        assert len(mock_scheduler.added_jobs) == 1

    @pytest.mark.asyncio()
    async def test_lifecycle_stop_stops_components_in_reverse_order_and_emits_event(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        first = ComponentStub(name="first")
        second = ComponentStub(name="second")
        lifecycle = StrategyLifecycleManager(
            config=strategy_config,
            state=StrategyRuntimeState(),
            components=[first, second],  # type: ignore[list-item]
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        await lifecycle.start()
        await lifecycle.stop()

        assert not lifecycle.is_started
        assert first.stop_calls == 1
        assert second.stop_calls == 1
        assert mock_event_bus.topic_emitted("strategy.engine.stopped")

    @pytest.mark.asyncio()
    async def test_lifecycle_stop_continues_when_component_stop_fails(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        first = ComponentStub(name="first")
        failing = ComponentStub(name="failing", fail_stop=True)
        lifecycle = StrategyLifecycleManager(
            config=strategy_config,
            state=StrategyRuntimeState(),
            components=[first, failing],  # type: ignore[list-item]
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        await lifecycle.start()
        await lifecycle.stop()

        assert failing.stop_calls == 1
        assert first.stop_calls == 1
        assert not lifecycle.is_started

    @pytest.mark.asyncio()
    async def test_cleanup_state_job_calls_prune(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        mock_scheduler,
        make_signal,
    ) -> None:
        state = StrategyRuntimeState()
        old_signal = make_signal(
            timestamp=utcnow() - timedelta(seconds=999),
        )
        state.update_signal(old_signal)

        lifecycle = StrategyLifecycleManager(
            config=strategy_config,
            state=state,
            components=[],
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        await lifecycle._cleanup_state_job()

        assert state.signals.get_by_signal_id(old_signal.signal_id) is None


# =============================================================================
# StrategyEngine
# =============================================================================


class TestStrategyEngine:
    def test_engine_constructs_default_subcomponents(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        assert isinstance(engine.state, StrategyRuntimeState)
        assert isinstance(engine.registry, StrategyRegistry)
        assert isinstance(engine.context_builder, StrategyContextBuilder)
        assert isinstance(engine.event_handler, StrategyEventHandler)
        assert isinstance(engine.lifecycle, StrategyLifecycleManager)
        assert engine.processor is not None
        assert isinstance(engine.stats, StrategyEngineStats)

    def test_engine_register_delegates_subscriptions_to_event_handler(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )

        engine.register()

        assert engine.is_registered
        assert engine.event_handler.is_registered
        assert mock_event_bus.subscribe_calls > 0

    @pytest.mark.asyncio()
    async def test_engine_start_and_stop_lifecycle(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )

        await engine.start()

        assert engine.is_started
        assert engine.is_registered
        assert engine.stats.started_at is not None
        assert engine.registry.is_started
        assert engine.context_builder.is_started
        assert engine.event_handler.is_started
        assert mock_event_bus.topic_emitted("strategy.engine.started")

        await engine.start()

        assert engine.is_started

        await engine.stop()

        assert not engine.is_started
        assert engine.stats.stopped_at is not None
        assert mock_event_bus.topic_emitted("strategy.engine.stopped")

        await engine.stop()

        assert not engine.is_started

    def test_engine_strategy_management_delegates_to_registry(
        self,
        strategy_config: StrategyConfig,
        dummy_strategy,
        mock_event_bus,
    ) -> None:
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )

        engine.add_strategy(dummy_strategy)

        assert engine.registry.get(dummy_strategy.strategy_name) is dummy_strategy

        removed = engine.remove_strategy(dummy_strategy.strategy_name)

        assert removed is dummy_strategy
        assert engine.registry.get(dummy_strategy.strategy_name) is None

    def test_engine_add_strategies(
        self,
        strategy_config: StrategyConfig,
        dummy_strategy,
        short_dummy_strategy,
        mock_event_bus,
    ) -> None:
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )

        engine.add_strategies([dummy_strategy, short_dummy_strategy])

        assert engine.registry.count() == 2
        assert engine.registry.get(dummy_strategy.strategy_name) is dummy_strategy
        assert engine.registry.get(short_dummy_strategy.strategy_name) is short_dummy_strategy

    @pytest.mark.asyncio()
    async def test_process_analytics_event_records_accepted_batch_and_emits_engine_event(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        batch = _accepted_batch(symbol="BTCUSDT")
        processor = ProcessorStub(batch=batch)
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            processor=processor,  # type: ignore[arg-type]
        )
        event_ts = datetime(2026, 5, 20, 12, 0, 0)
        event = MockEvent(
            topic="analytics.orderflow.updated",
            payload={"symbol": "BTCUSDT"},
            timestamp=event_ts,
        )

        result = await engine.process_analytics_event(
            event_name="analytics.orderflow.updated",
            payload={"symbol": "BTCUSDT"},
            event=event,
        )

        assert result is batch
        assert processor.calls[0]["event_name"] == "analytics.orderflow.updated"
        assert processor.calls[0]["payload"] == {"symbol": "BTCUSDT"}
        assert processor.calls[0]["timestamp"] == event_ts
        assert processor.calls[0]["emit"] is True

        assert engine.stats.events_received == 1
        assert engine.stats.events_processed == 1
        assert engine.stats.batches_accepted == 1
        assert engine.stats.batches_rejected == 0
        assert mock_event_bus.topic_emitted("strategy.engine.batch_processed")

        payload = mock_event_bus.emitted[-1].payload
        assert payload["symbol"] == "BTCUSDT"
        assert payload["accepted"] is True
        assert payload["emitted"] is True

    @pytest.mark.asyncio()
    async def test_process_analytics_event_records_rejected_batch(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        batch = _rejected_batch(symbol="BTCUSDT", reasons=["no_strategies_routed"])
        processor = ProcessorStub(batch=batch)
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            processor=processor,  # type: ignore[arg-type]
        )

        result = await engine.process_analytics_event(
            event_name="analytics.orderflow.updated",
            payload={"symbol": "BTCUSDT"},
        )

        assert result is batch
        assert engine.stats.events_received == 1
        assert engine.stats.events_processed == 1
        assert engine.stats.batches_accepted == 0
        assert engine.stats.batches_rejected == 1

        payload = mock_event_bus.emitted[-1].payload
        assert payload["accepted"] is False
        assert payload["reasons"] == ["no_strategies_routed"]

    @pytest.mark.asyncio()
    async def test_process_analytics_event_records_errors_and_rethrows(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        processor = ProcessorStub(error=RuntimeError("processor failed"))
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            processor=processor,  # type: ignore[arg-type]
        )

        with pytest.raises(RuntimeError, match="processor failed"):
            await engine.process_analytics_event(
                event_name="analytics.orderflow.updated",
                payload={"symbol": "BTCUSDT"},
            )

        assert engine.stats.events_received == 1
        assert engine.stats.events_processed == 0
        assert engine.stats.events_failed == 1
        assert engine.state.metrics.errors_total == 1
        assert mock_event_bus.topic_emitted("strategy.engine.error")

        payload = mock_event_bus.emitted[-1].payload
        assert payload["event_name"] == "analytics.orderflow.updated"
        assert payload["error"] == "processor failed"
        assert payload["payload_symbol"] == "BTCUSDT"

    def test_update_context_from_payload_updates_stats(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )

        context = engine.update_context_from_payload(
            event_name="system.strategy.context_update",
            payload={
                "symbol": "BTCUSDT",
                "price": 100.0,
            },
        )

        assert context.symbol == "BTCUSDT"
        assert context.price is not None
        assert engine.stats.contexts_updated == 1
        assert engine.state.contexts.get_context("BTCUSDT") is context

    def test_build_context_updates_stats(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        make_feature,
    ) -> None:
        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )

        context = engine.build_context(
            symbol="BTCUSDT",
            timeframe=Timeframe.M5,
            features=[make_feature(name="orderflow_imbalance")],
            domain_data={
                FeatureSource.ORDERFLOW: {
                    "pressure": 0.8,
                }
            },
        )

        assert context.symbol == "BTCUSDT"
        assert context.timeframe is Timeframe.M5
        assert engine.stats.contexts_built == 1
        assert engine.state.contexts.get_context("BTCUSDT") is context

    @pytest.mark.asyncio()
    async def test_evaluate_symbol_does_not_emit_signal_generated(
        self,
        strategy_config: StrategyConfig,
        dummy_strategy,
        strategy_context: StrategyContext,
        mock_event_bus,
    ) -> None:
        state = StrategyRuntimeState()
        state.update_context(strategy_context)

        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            state=state,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )
        engine.add_strategy(dummy_strategy)

        evaluations = await engine.evaluate_symbol("BTCUSDT")

        assert len(evaluations) == 1
        assert evaluations[0].strategy_name == dummy_strategy.strategy_name
        assert evaluations[0].passed
        assert state.metrics.evaluations_total == 1
        assert not mock_event_bus.topic_emitted("signal.generated")
        assert not mock_event_bus.nowait_topic_emitted("signal.generated")

    @pytest.mark.asyncio()
    async def test_evaluate_symbol_catches_strategy_errors(
        self,
        strategy_config: StrategyConfig,
        failing_strategy,
        strategy_context: StrategyContext,
        mock_event_bus,
    ) -> None:
        state = StrategyRuntimeState()
        state.update_context(strategy_context)

        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            state=state,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )
        engine.add_strategy(failing_strategy)

        evaluations = await engine.evaluate_symbol("BTCUSDT")

        assert len(evaluations) == 1
        assert not evaluations[0].passed
        assert evaluations[0].reasons[0].startswith("manual_evaluation_error:")
        assert evaluations[0].metadata["error_type"] == "StrategyEvaluationError"
        assert state.metrics.errors_total == 1

    def test_prune_delegates_to_runtime_state(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        make_signal,
    ) -> None:
        state = StrategyRuntimeState()
        old_signal = make_signal(
            timestamp=utcnow() - timedelta(seconds=999),
        )
        state.update_signal(old_signal)

        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            state=state,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )

        removed = engine.prune()

        assert isinstance(removed, dict)
        assert state.signals.get_by_signal_id(old_signal.signal_id) is None

    def test_summary_contains_engine_registry_and_state(
        self,
        strategy_config: StrategyConfig,
        dummy_strategy,
        strategy_context: StrategyContext,
        mock_event_bus,
    ) -> None:
        state = StrategyRuntimeState()
        state.update_context(strategy_context)

        engine = _make_engine(
            config=strategy_config,
            event_bus=mock_event_bus,
            state=state,
            processor=ProcessorStub(),  # type: ignore[arg-type]
        )
        engine.add_strategy(dummy_strategy)
        engine.stats.record_event_received()

        summary = engine.summary()

        assert summary["engine"]["started"] is False
        assert summary["engine"]["registered"] is False
        assert summary["engine"]["stats"]["events_received"] == 1
        assert summary["registry"]["total"] == 1
        assert summary["state"]["contexts"]["contexts"] == 1