# tests/strategy/test_strategy_base_and_lifecycle.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from core.event_bus import EventPriority

from strategy.base import BaseStrategyComponent
from strategy.config import StrategyConfig
from strategy.enums import (
    MarketRegime,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    Timeframe,
)
from strategy.exceptions import (
    StrategyConfigError,
    StrategyEvaluationError,
)
from strategy.models import StrategyContext


# =============================================================================
# Local test components
# =============================================================================


class TrackingComponent(BaseStrategyComponent):
    """
    Concrete component для прямого тестування BaseStrategyComponent.

    register() спеціально робить одну subscription, щоб перевірити:
    - start() викликає register();
    - subscription зберігається;
    - stop()/unregister() її знімає.
    """

    component_namespace = "tests.strategy.tracking_component"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: Any | None = None,
        scheduler: Any | None = None,
    ) -> None:
        self.register_calls = 0
        self.unregister_calls = 0
        self.handled_events: list[Any] = []
        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )

    def register(self) -> None:
        self.register_calls += 1

        if self.event_bus is not None and not self._registered:
            self.subscribe_event("test.topic", self.handle_event)

        self._registered = True

    def unregister(self) -> None:
        self.unregister_calls += 1
        super().unregister()

    async def handle_event(self, event: Any) -> None:
        self.handled_events.append(event)


@dataclass(slots=True)
class ConfigValidationProbe:
    validate_calls: int = 0
    should_fail: bool = False

    def validate(self) -> None:
        self.validate_calls += 1
        if self.should_fail:
            raise StrategyConfigError("probe validation failed")


# =============================================================================
# BaseStrategyComponent lifecycle
# =============================================================================


class TestBaseStrategyComponentLifecycle:
    def test_component_requires_config(self) -> None:
        with pytest.raises(StrategyConfigError):
            TrackingComponent(config=None)  # type: ignore[arg-type]

    def test_component_calls_config_validate(self) -> None:
        probe = ConfigValidationProbe()

        component = TrackingComponent(config=probe)  # type: ignore[arg-type]

        assert component.config is probe
        assert probe.validate_calls == 1
        assert component.component_name == "TrackingComponent"
        assert not component.is_started
        assert not component.is_registered
        assert component.subscriptions_count == 0
        assert component.scheduler_jobs_count == 0

    def test_component_propagates_config_validation_error(self) -> None:
        probe = ConfigValidationProbe(should_fail=True)

        with pytest.raises(StrategyConfigError, match="probe validation failed"):
            TrackingComponent(config=probe)  # type: ignore[arg-type]

    def test_register_marks_component_registered(
        self,
        strategy_config: StrategyConfig,
    ) -> None:
        component = TrackingComponent(config=strategy_config)

        component.register()

        assert component.is_registered
        assert component.register_calls == 1

    @pytest.mark.asyncio()
    async def test_start_registers_component_once(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        await component.start()

        assert component.is_started
        assert component.is_registered
        assert component.register_calls == 1
        assert component.subscriptions_count == 1
        assert mock_event_bus.subscribe_calls == 1

        await component.start()

        assert component.is_started
        assert component.register_calls == 1
        assert component.subscriptions_count == 1
        assert mock_event_bus.subscribe_calls == 1

    @pytest.mark.asyncio()
    async def test_stop_unregisters_subscriptions_and_is_idempotent(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        await component.start()
        subscription = component._subscriptions[0]

        await component.stop()

        assert not component.is_started
        assert not component.is_registered
        assert component.subscriptions_count == 0
        assert component.unregister_calls == 1
        assert mock_event_bus.unsubscribe_calls == 1
        assert subscription in mock_event_bus.unsubscribed

        await component.stop()

        assert component.unregister_calls == 1
        assert mock_event_bus.unsubscribe_calls == 1

    @pytest.mark.asyncio()
    async def test_stop_continues_when_unsubscribe_fails(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        await component.start()

        mock_event_bus.fail_unsubscribe = True

        await component.stop()

        assert not component.is_started
        assert not component.is_registered
        assert component.subscriptions_count == 0
        assert mock_event_bus.unsubscribe_calls == 1

    @pytest.mark.asyncio()
    async def test_subscribed_handler_can_be_dispatched(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        await component.start()
        event = await mock_event_bus.dispatch(
            "test.topic",
            {"symbol": "BTCUSDT"},
        )

        assert component.handled_events == [event]


# =============================================================================
# EventBus helpers
# =============================================================================


class TestBaseStrategyComponentEventBusHelpers:
    def test_ensure_event_bus_raises_without_bus(
        self,
        strategy_config: StrategyConfig,
    ) -> None:
        component = TrackingComponent(config=strategy_config)

        with pytest.raises(RuntimeError, match="event_bus is not configured"):
            component.ensure_event_bus()

    def test_ensure_event_bus_returns_configured_bus(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        assert component.ensure_event_bus() is mock_event_bus

    @pytest.mark.asyncio()
    async def test_emit_event_skips_when_bus_missing(
        self,
        strategy_config: StrategyConfig,
    ) -> None:
        component = TrackingComponent(config=strategy_config)

        await component.emit_event("test.topic", {"ok": True})

        assert component.is_started is False

    @pytest.mark.asyncio()
    async def test_emit_event_validates_topic_and_payload(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        with pytest.raises(ValueError, match="topic cannot be empty"):
            await component.emit_event("", {"ok": True})

        with pytest.raises(ValueError, match="payload must be a dict"):
            await component.emit_event("test.topic", ["bad"])  # type: ignore[arg-type]

        assert mock_event_bus.emit_calls == 0

    @pytest.mark.asyncio()
    async def test_emit_event_uses_event_bus_emit(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        await component.emit_event(
            "strategy.test.event",
            {"symbol": "BTCUSDT"},
            priority=EventPriority.HIGH,
            source="unit-test",
            correlation_id="abc",
        )

        assert mock_event_bus.emit_calls == 1
        assert mock_event_bus.topic_emitted("strategy.test.event")

        event = mock_event_bus.emitted[0]
        assert event.topic == "strategy.test.event"
        assert event.payload == {"symbol": "BTCUSDT"}
        assert event.priority is EventPriority.HIGH
        assert event.source == "unit-test"
        assert event.kwargs["correlation_id"] == "abc"

    @pytest.mark.asyncio()
    async def test_emit_event_propagates_bus_error(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            event_bus=mock_event_bus,
        )
        mock_event_bus.fail_emit = True

        with pytest.raises(RuntimeError, match="mock emit failure"):
            await component.emit_event("strategy.test.event", {"ok": True})

    def test_nowait_emit_skips_when_bus_missing(
        self,
        strategy_config: StrategyConfig,
    ) -> None:
        component = TrackingComponent(config=strategy_config)

        component.emit_event_nowait_best_effort("strategy.test.event", {"ok": True})

        assert component.is_started is False

    def test_nowait_emit_uses_publish_nowait_best_effort(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        component.emit_event_nowait_best_effort(
            "strategy.test.nowait",
            {"symbol": "BTCUSDT"},
            priority=EventPriority.LOW,
            source="unit-test",
        )

        assert mock_event_bus.nowait_calls == 1
        assert mock_event_bus.nowait_topic_emitted("strategy.test.nowait")

        event = mock_event_bus.nowait[0]
        assert event.priority is EventPriority.LOW
        assert event.source == "unit-test"
        assert event.payload == {"symbol": "BTCUSDT"}

    def test_nowait_emit_propagates_bus_error(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            event_bus=mock_event_bus,
        )
        mock_event_bus.fail_nowait = True

        with pytest.raises(RuntimeError, match="mock nowait failure"):
            component.emit_event_nowait_best_effort(
                "strategy.test.nowait",
                {"ok": True},
            )

    def test_subscribe_event_requires_topic(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        with pytest.raises(ValueError, match="topic cannot be empty"):
            component.subscribe_event("", lambda event: None)

    def test_subscribe_event_requires_event_bus(
        self,
        strategy_config: StrategyConfig,
    ) -> None:
        component = TrackingComponent(config=strategy_config)

        with pytest.raises(RuntimeError, match="event_bus is not configured"):
            component.subscribe_event("test.topic", lambda event: None)

    def test_subscribe_event_stores_subscription(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        subscription = component.subscribe_event(
            "test.topic",
            lambda event: None,
            priority=EventPriority.LOW,
        )

        assert component.subscriptions_count == 1
        assert component._subscriptions == [subscription]
        assert mock_event_bus.subscriptions == [subscription]
        assert subscription.kwargs["priority"] is EventPriority.LOW


# =============================================================================
# Scheduler helpers
# =============================================================================


class TestBaseStrategyComponentSchedulerHelpers:
    def test_ensure_scheduler_raises_without_scheduler(
        self,
        strategy_config: StrategyConfig,
    ) -> None:
        component = TrackingComponent(config=strategy_config)

        with pytest.raises(RuntimeError, match="scheduler is not configured"):
            component.ensure_scheduler()

    def test_ensure_scheduler_returns_configured_scheduler(
        self,
        strategy_config: StrategyConfig,
        mock_scheduler,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            scheduler=mock_scheduler,
        )

        assert component.ensure_scheduler() is mock_scheduler

    def test_remember_scheduler_job_tracks_job(
        self,
        strategy_config: StrategyConfig,
        mock_scheduler,
    ) -> None:
        component = TrackingComponent(
            config=strategy_config,
            scheduler=mock_scheduler,
        )
        job = mock_scheduler.add_interval_job(
            name="strategy.test.cleanup",
            callback=lambda: None,
            interval_seconds=10,
        )

        remembered = component.remember_scheduler_job(job)

        assert remembered is job
        assert component.scheduler_jobs_count == 1
        assert component._scheduler_jobs == [job]


# =============================================================================
# Context-aware helpers
# =============================================================================


class TestContextAwareStrategyHelpers:
    def test_validate_context_accepts_strategy_context(
        self,
        dummy_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        dummy_strategy.validate_context(strategy_context)

    @pytest.mark.parametrize("bad_context", [None, object(), {"symbol": "BTCUSDT"}])
    def test_validate_context_rejects_invalid_context(
        self,
        dummy_strategy,
        bad_context: Any,
    ) -> None:
        with pytest.raises(StrategyEvaluationError):
            dummy_strategy.validate_context(bad_context)  # type: ignore[arg-type]

    def test_require_feature_returns_feature_value(
        self,
        dummy_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        value = dummy_strategy.require_feature(
            strategy_context,
            "orderflow_imbalance",
        )

        assert value == strategy_context.get_feature("orderflow_imbalance")

    def test_require_feature_rejects_empty_name(
        self,
        dummy_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        with pytest.raises(StrategyEvaluationError, match="feature_name cannot be empty"):
            dummy_strategy.require_feature(strategy_context, "")

    def test_require_feature_rejects_missing_feature(
        self,
        dummy_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        with pytest.raises(StrategyEvaluationError, match="missing required feature"):
            dummy_strategy.require_feature(strategy_context, "missing_feature")

    def test_optional_feature_returns_default_for_missing_feature(
        self,
        dummy_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        assert (
            dummy_strategy.optional_feature(
                strategy_context,
                "missing_feature",
                default="fallback",
            )
            == "fallback"
        )

    def test_optional_feature_rejects_empty_name(
        self,
        dummy_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        with pytest.raises(StrategyEvaluationError, match="feature_name cannot be empty"):
            dummy_strategy.optional_feature(strategy_context, "")

    def test_has_required_features(
        self,
        dummy_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        assert dummy_strategy.has_required_features(
            strategy_context,
            {"orderflow_imbalance"},
        )
        assert not dummy_strategy.has_required_features(
            strategy_context,
            {"orderflow_imbalance", "missing_feature"},
        )


# =============================================================================
# BaseStrategy config accessors and applicability
# =============================================================================


class TestBaseStrategyConfigAndApplicability:
    def test_strategy_properties_resolve_definition(
        self,
        dummy_strategy,
        definition_config,
    ) -> None:
        assert dummy_strategy.strategy_name == definition_config.name
        assert dummy_strategy.name == definition_config.name
        assert dummy_strategy.category is StrategyCategory.ORDERFLOW
        assert dummy_strategy.priority == definition_config.priority
        assert dummy_strategy.weight == definition_config.weight
        assert dummy_strategy.required_features() == definition_config.required_features
        assert dummy_strategy.supported_timeframes() == set(definition_config.runtime.timeframes)
        assert dummy_strategy.supported_regimes() == set(definition_config.runtime.allowed_regimes)
        assert dummy_strategy.allowed_symbols() == set(definition_config.runtime.symbols)
        assert dummy_strategy.min_confidence() == definition_config.runtime.min_confidence
        assert dummy_strategy.min_score() == definition_config.runtime.min_score
        assert dummy_strategy.cooldown_seconds() == definition_config.runtime.cooldown_seconds
        assert (
            dummy_strategy.max_signal_age_seconds()
            == definition_config.runtime.max_signal_age_seconds
        )

    def test_strategy_supports_symbol_timeframe_and_regime(
        self,
        dummy_strategy,
    ) -> None:
        assert dummy_strategy.supports_symbol("BTCUSDT")
        assert not dummy_strategy.supports_symbol("")
        assert not dummy_strategy.supports_symbol("SOLUSDT")

        assert dummy_strategy.supports_timeframe(Timeframe.M1)
        assert not dummy_strategy.supports_timeframe(Timeframe.H4)

        assert dummy_strategy.supports_regime(MarketRegime.UNKNOWN)
        assert dummy_strategy.supports_regime(MarketRegime.TRENDING_UP)

    def test_strategy_without_symbol_allowlist_supports_any_symbol(
        self,
        make_definition,
        make_config,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        definition = make_definition(
            name="any_symbol_strategy",
            runtime=None,
            required_features=("orderflow_imbalance",),
        )
        definition.runtime.symbols = []
        config = make_config(definitions=[definition])

        from conftest import DummyStrategy

        strategy = DummyStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=definition,
        )

        assert strategy.supports_symbol("BTCUSDT")
        assert strategy.supports_symbol("SOLUSDT")

    def test_validate_context_requirements_accepts_applicable_context(
        self,
        dummy_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        dummy_strategy.validate_context_requirements(strategy_context)

    def test_validate_context_requirements_rejects_unsupported_symbol(
        self,
        dummy_strategy,
        make_context,
        make_feature,
    ) -> None:
        context = make_context(
            symbol="SOLUSDT",
            features=[
                make_feature(symbol="SOLUSDT", name="orderflow_imbalance"),
            ],
        )

        with pytest.raises(StrategyEvaluationError, match="symbol SOLUSDT is not allowed"):
            dummy_strategy.validate_context_requirements(context)

    def test_validate_context_requirements_rejects_unsupported_timeframe(
        self,
        dummy_strategy,
        make_context,
        make_feature,
    ) -> None:
        context = make_context(
            timeframe=Timeframe.H4,
            features=[make_feature(name="orderflow_imbalance")],
        )

        with pytest.raises(StrategyEvaluationError, match="timeframe"):
            dummy_strategy.validate_context_requirements(context)

    def test_validate_context_requirements_rejects_missing_required_feature(
        self,
        dummy_strategy,
        make_context,
        make_feature,
    ) -> None:
        context = make_context(
            features=[make_feature(name="different_feature")],
        )

        with pytest.raises(StrategyEvaluationError, match="missing required features"):
            dummy_strategy.validate_context_requirements(context)

    def test_should_evaluate_returns_true_for_applicable_context(
        self,
        dummy_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        assert dummy_strategy.should_evaluate(strategy_context)

    def test_should_evaluate_returns_false_for_disabled_strategy(
        self,
        make_definition,
        make_config,
        mock_event_bus,
        mock_scheduler,
        strategy_context: StrategyContext,
    ) -> None:
        from conftest import DummyStrategy, make_runtime_config

        runtime = make_runtime_config(enabled=False)
        definition = make_definition(
            name="disabled_strategy",
            runtime=runtime,
            required_features=("orderflow_imbalance",),
        )
        config = make_config(definitions=[definition])

        strategy = DummyStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=definition,
        )

        assert not strategy.should_evaluate(strategy_context)

    def test_should_evaluate_returns_false_for_normal_negative_cases(
        self,
        dummy_strategy,
        make_context,
        make_feature,
    ) -> None:
        context = make_context(
            symbol="SOLUSDT",
            features=[
                make_feature(symbol="SOLUSDT", name="orderflow_imbalance"),
            ],
        )

        assert not dummy_strategy.should_evaluate(context)


# =============================================================================
# BaseStrategy.evaluate()
# =============================================================================


class TestBaseStrategyEvaluation:
    @pytest.mark.asyncio()
    async def test_evaluate_returns_passed_evaluation_with_signal(
        self,
        dummy_strategy,
        strategy_context: StrategyContext,
        mock_event_bus,
    ) -> None:
        evaluation = await dummy_strategy.evaluate(strategy_context)

        evaluation.validate()

        assert evaluation.passed
        assert evaluation.signal is not None
        assert evaluation.strategy_name == dummy_strategy.strategy_name
        assert evaluation.symbol == strategy_context.symbol
        assert evaluation.score == evaluation.signal.score
        assert evaluation.confidence == evaluation.signal.confidence
        assert dummy_strategy.generate_calls == 1

        signal = evaluation.signal
        assert signal.symbol == strategy_context.symbol
        assert signal.strategy_name == dummy_strategy.strategy_name
        assert signal.category is dummy_strategy.category
        assert signal.timeframe is strategy_context.timeframe
        assert signal.regime is MarketRegime.UNKNOWN
        assert signal.metadata["strategy_name"] == dummy_strategy.strategy_name
        assert signal.metadata["category"] == dummy_strategy.category.value
        assert signal.metadata["source"] == "strategy"
        assert signal.metadata["signal_id"] == signal.signal_id

        assert not mock_event_bus.topic_emitted("signal.generated")
        assert not mock_event_bus.nowait_topic_emitted("signal.generated")

    @pytest.mark.asyncio()
    async def test_evaluate_returns_not_applicable_when_strategy_disabled(
        self,
        make_definition,
        make_config,
        mock_event_bus,
        mock_scheduler,
        strategy_context: StrategyContext,
    ) -> None:
        from conftest import DummyStrategy, make_runtime_config

        runtime = make_runtime_config(enabled=False)
        definition = make_definition(
            name="disabled_strategy",
            runtime=runtime,
            required_features=("orderflow_imbalance",),
        )
        config = make_config(definitions=[definition])

        strategy = DummyStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=definition,
        )

        evaluation = await strategy.evaluate(strategy_context)

        assert not evaluation.passed
        assert evaluation.signal is None
        assert evaluation.reasons == ["strategy_not_applicable"]
        assert strategy.generate_calls == 0

    @pytest.mark.asyncio()
    async def test_evaluate_returns_no_signal_generated(
        self,
        no_signal_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        evaluation = await no_signal_strategy.evaluate(strategy_context)

        assert not evaluation.passed
        assert evaluation.signal is None
        assert evaluation.reasons == ["no_signal_generated"]
        assert no_signal_strategy.generate_calls == 1

    @pytest.mark.asyncio()
    async def test_evaluate_rejects_non_directional_signal(
        self,
        flat_dummy_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        evaluation = await flat_dummy_strategy.evaluate(strategy_context)

        assert not evaluation.passed
        assert evaluation.signal is not None
        assert evaluation.signal.status is SignalStatus.REJECTED
        assert "signal_side_is_not_directional" in evaluation.reasons
        assert "signal_side_is_not_directional" in evaluation.signal.reasons

    @pytest.mark.asyncio()
    async def test_evaluate_rejects_low_confidence_signal(
        self,
        make_definition,
        make_config,
        mock_event_bus,
        mock_scheduler,
        strategy_context: StrategyContext,
    ) -> None:
        from conftest import DummyStrategy, make_runtime_config

        runtime = make_runtime_config(
            min_confidence=0.9,
            min_score=0.0,
        )
        definition = make_definition(
            name="low_confidence_strategy",
            runtime=runtime,
            required_features=("orderflow_imbalance",),
        )
        config = make_config(definitions=[definition])

        strategy = DummyStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=definition,
            confidence=0.5,
            score=1.0,
        )

        evaluation = await strategy.evaluate(strategy_context)

        assert not evaluation.passed
        assert evaluation.signal is not None
        assert evaluation.signal.status is SignalStatus.REJECTED
        assert "confidence_below_strategy_minimum" in evaluation.reasons

    @pytest.mark.asyncio()
    async def test_evaluate_rejects_low_score_signal(
        self,
        make_definition,
        make_config,
        mock_event_bus,
        mock_scheduler,
        strategy_context: StrategyContext,
    ) -> None:
        from conftest import DummyStrategy, make_runtime_config

        runtime = make_runtime_config(
            min_confidence=0.1,
            min_score=2.0,
        )
        definition = make_definition(
            name="low_score_strategy",
            runtime=runtime,
            required_features=("orderflow_imbalance",),
        )
        config = make_config(definitions=[definition])

        strategy = DummyStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=definition,
            confidence=1.0,
            score=0.1,
        )

        evaluation = await strategy.evaluate(strategy_context)

        assert not evaluation.passed
        assert evaluation.signal is not None
        assert evaluation.signal.status is SignalStatus.REJECTED
        assert "score_below_strategy_minimum" in evaluation.reasons

    @pytest.mark.asyncio()
    async def test_evaluate_catches_strategy_exception(
        self,
        failing_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        evaluation = await failing_strategy.evaluate(strategy_context)

        assert not evaluation.passed
        assert evaluation.signal is None
        assert evaluation.reasons
        assert evaluation.reasons[0].startswith("evaluation_error:")
        assert evaluation.metadata["error_type"] == "StrategyEvaluationError"
        assert "intentional strategy failure" in evaluation.metadata["error"]

    @pytest.mark.asyncio()
    async def test_evaluate_catches_invalid_context(
        self,
        dummy_strategy,
    ) -> None:
        evaluation = await dummy_strategy.evaluate(None)  # type: ignore[arg-type]

        assert not evaluation.passed
        assert evaluation.signal is None
        assert evaluation.reasons[0].startswith("evaluation_error:")

    @pytest.mark.asyncio()
    async def test_evaluate_does_not_call_generate_when_context_not_applicable(
        self,
        dummy_strategy,
        make_context,
        make_feature,
    ) -> None:
        context = make_context(
            symbol="SOLUSDT",
            features=[
                make_feature(symbol="SOLUSDT", name="orderflow_imbalance"),
            ],
        )

        evaluation = await dummy_strategy.evaluate(context)

        assert not evaluation.passed
        assert evaluation.signal is None
        assert evaluation.reasons == ["strategy_not_applicable"]
        assert dummy_strategy.generate_calls == 0


# =============================================================================
# BaseStrategy.build_signal()
# =============================================================================


class TestBaseStrategyBuildSignal:
    def test_build_signal_creates_enriched_strategy_signal(
        self,
        dummy_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        signal = dummy_strategy.build_signal(
            context=strategy_context,
            side=SignalSide.LONG,
            confidence=0.88,
            score=1.25,
            reasons=["reason"],
            confirmations=["confirmation"],
            source_features=["orderflow_imbalance"],
            metadata={
                "exchange": "binance",
                "market_type": "usdm_futures",
            },
        )

        signal.validate()

        assert signal.symbol == strategy_context.symbol
        assert signal.side is SignalSide.LONG
        assert signal.strategy_name == dummy_strategy.strategy_name
        assert signal.category is StrategyCategory.ORDERFLOW
        assert signal.timeframe is strategy_context.timeframe
        assert signal.setup_type is dummy_strategy.default_setup_type
        assert signal.trigger_type is dummy_strategy.default_trigger_type
        assert signal.confidence == 0.88
        assert signal.score == 1.25
        assert signal.reasons == ["reason"]
        assert signal.confirmations == ["confirmation"]
        assert signal.source_features == ["orderflow_imbalance"]
        assert signal.metadata["source"] == "strategy"
        assert signal.metadata["exchange"] == "binance"
        assert signal.metadata["market_type"] == "usdm_futures"

    def test_build_signal_validates_context_requirements(
        self,
        dummy_strategy,
        make_context,
        make_feature,
    ) -> None:
        context = make_context(
            symbol="SOLUSDT",
            features=[
                make_feature(symbol="SOLUSDT", name="orderflow_imbalance"),
            ],
        )

        with pytest.raises(StrategyEvaluationError, match="symbol SOLUSDT is not allowed"):
            dummy_strategy.build_signal(
                context=context,
                side=SignalSide.LONG,
                confidence=0.8,
                score=1.0,
            )

    def test_build_signal_rejects_flat_signal_when_status_active(
        self,
        dummy_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        with pytest.raises(Exception):
            dummy_strategy.build_signal(
                context=strategy_context,
                side=SignalSide.FLAT,
                confidence=0.8,
                score=1.0,
                status=SignalStatus.PENDING,
            )


# =============================================================================
# Logging helpers smoke tests
# =============================================================================


class TestBaseStrategyLoggingHelpers:
    def test_logging_helpers_do_not_raise(
        self,
        strategy_config: StrategyConfig,
    ) -> None:
        component = TrackingComponent(config=strategy_config)

        component.log_debug("debug message", key="value")
        component.log_info("info message", key="value")
        component.log_warning("warning message", key="value")
        component.log_error("error message", key="value")

        assert component.component_name == "TrackingComponent"