"""
Tests for full-pipeline StrategyTester orchestration.

Covered modules:
- backtesting.strategy_tester
- backtesting.market_replay
- backtesting.execution_simulator
- backtesting.position_simulator
- backtesting.performance_metrics
- backtesting.model_analytics
- backtesting.report_builder

These tests focus on pipeline wiring and lifecycle correctness:
- dependency validation;
- component registration/start/stop order behavior;
- production-like event flow from market replay to strategy/risk/execution/position;
- risk-blocked signals not reaching execution;
- result collection, metrics, analytics and reports;
- guards against live execution dependencies.

The tests use fake strategy/risk components from conftest so they stay fully
offline and deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backtesting.config import BacktestConfig, StrategyTesterConfig
from backtesting.enums import BacktestMode, BacktestStatus, SignalOutcome
from backtesting.exceptions import (
    BacktestDependencyError,
    BacktestLifecycleError,
    StrategyBacktestRunError,
    StrategyRegistryEmptyError,
    StrategySelectionError,
)
from backtesting.market_replay import MarketReplay
from backtesting.models import BacktestDataset, BacktestResult
from backtesting.strategy_tester import (
    InMemoryBacktestEventBus,
    StrategyTester,
    run_backtest,
)


# =============================================================================
# Compatibility helpers
# =============================================================================


def _outcome_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _status_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _copy_strategy_tester_config(
    config: StrategyTesterConfig,
    **changes: Any,
) -> StrategyTesterConfig:
    payload = {
        field: getattr(config, field)
        for field in getattr(config, "__dataclass_fields__", {})
    }
    payload.update(changes)
    return StrategyTesterConfig(**payload)


def _disable_reports(config: BacktestConfig) -> BacktestConfig:
    config.save_report = False
    config.report_builder.enabled = False
    return config


def _assert_completed_result(result: BacktestResult) -> None:
    assert result.status == BacktestStatus.COMPLETED
    assert result.completed_successfully
    assert result.error is None
    assert result.started_at is not None
    assert result.finished_at is not None
    assert result.duration_seconds >= 0.0


class _NamedStrategy:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRegistry:
    def __init__(self, names: list[str]) -> None:
        self.strategies = {name: _NamedStrategy(name) for name in names}


class _EngineWithSignalProcessor:
    """
    Minimal engine that satisfies require_signal_processor via owned processor.
    """

    def __init__(self) -> None:
        self.signal_processor = object()
        self.registered = False
        self.started = False

    def register(self) -> None:
        self.registered = True

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False


class _EngineWithLiveOrderManager(_EngineWithSignalProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.order_manager = object()


class _LifecycleProbe:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls
        self.registered = False
        self.started = False

    def register(self) -> None:
        self.registered = True
        self.calls.append(f"register:{self.name}")

    async def start(self) -> None:
        self.started = True
        self.calls.append(f"start:{self.name}")

    async def stop(self) -> None:
        self.started = False
        self.calls.append(f"stop:{self.name}")


# =============================================================================
# Dependency and environment validation
# =============================================================================


@pytest.mark.asyncio
async def test_strategy_tester_requires_dataset(
    backtest_config: BacktestConfig,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)

    tester = StrategyTester(
        backtest_config,
        event_bus=event_bus,
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    with pytest.raises(BacktestLifecycleError):
        await tester.prepare_environment()


@pytest.mark.asyncio
async def test_strategy_tester_requires_strategy_engine_when_configured(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)
    backtest_config.strategy_tester.require_strategy_engine = True

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        risk_manager=fake_risk_manager_approve,
    )

    with pytest.raises(BacktestDependencyError):
        await tester.prepare_environment()


@pytest.mark.asyncio
async def test_strategy_tester_requires_risk_manager_when_configured(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
) -> None:
    _disable_reports(backtest_config)
    backtest_config.strategy_tester.require_risk_manager = True

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=fake_strategy_engine,
    )

    with pytest.raises(BacktestDependencyError):
        await tester.prepare_environment()


@pytest.mark.asyncio
async def test_strategy_tester_accepts_engine_owned_signal_processor(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)
    backtest_config.strategy_tester.require_signal_processor = True
    backtest_config.strategy_tester.require_strategy_engine = True

    engine = _EngineWithSignalProcessor()

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=engine,
        risk_manager=fake_risk_manager_approve,
    )

    await tester.prepare_environment()

    assert tester.components.signal_processor is None
    assert engine.registered
    assert tester._prepared is True


@pytest.mark.asyncio
async def test_strategy_tester_rejects_live_execution_dependency(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)
    backtest_config.strategy_tester.fail_if_live_execution_detected = True

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=_EngineWithLiveOrderManager(),
        risk_manager=fake_risk_manager_approve,
    )

    with pytest.raises(BacktestDependencyError):
        await tester.prepare_environment()


# =============================================================================
# Strategy selection validation
# =============================================================================


@pytest.mark.asyncio
async def test_strategy_tester_rejects_empty_registry(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_registry=_FakeRegistry([]),
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    with pytest.raises(StrategyRegistryEmptyError):
        await tester.prepare_environment()


@pytest.mark.asyncio
async def test_strategy_tester_rejects_missing_selected_strategy(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)
    backtest_config.test_all_registered_strategies = False
    backtest_config.strategies = ["missing_strategy"]

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_registry=_FakeRegistry(["fake_strategy"]),
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    with pytest.raises(StrategySelectionError):
        await tester.prepare_environment()


@pytest.mark.asyncio
async def test_strategy_tester_accepts_registered_selected_strategy(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)
    backtest_config.test_all_registered_strategies = False
    backtest_config.strategies = ["fake_strategy"]

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_registry=_FakeRegistry(["fake_strategy"]),
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    await tester.prepare_environment()

    assert tester._prepared is True


# =============================================================================
# Lifecycle and component wiring
# =============================================================================


@pytest.mark.asyncio
async def test_prepare_environment_builds_backtesting_components(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    await tester.prepare_environment()

    assert tester.components.cost_model is not None
    assert tester.components.clock is not None
    assert tester.components.market_replay is not None
    assert tester.components.execution_simulator is not None
    assert tester.components.position_simulator is not None
    assert tester.components.performance_metrics is not None
    assert tester.components.model_analytics is not None
    assert tester.components.report_builder is not None
    assert isinstance(tester.components.market_replay, MarketReplay)
    assert fake_strategy_engine.registered
    assert fake_risk_manager_approve.registered


@pytest.mark.asyncio
async def test_strategy_tester_registers_and_starts_components_in_pipeline_order(
    strategy_tester_config: StrategyTesterConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
) -> None:
    config = _copy_strategy_tester_config(
        strategy_tester_config,
        require_analytics=False,
        require_strategy_engine=True,
        require_risk_manager=True,
        require_signal_processor=False,
        cleanup_after_run=False,
        stop_on_first_error=True,
        symbols=["BTCUSDT"],
        timeframes=["1m"],
    )

    calls: list[str] = []
    data_cache = _LifecycleProbe("data", calls)
    analytics = _LifecycleProbe("analytics", calls)

    tester = StrategyTester(
        config,
        dataset=sample_dataset,
        event_bus=event_bus,
        data_caches=[data_cache],
        analytics_components=[analytics],
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    await tester.prepare_environment()
    await tester._start_components()
    await tester._stop_components()

    assert calls[:2] == ["register:data", "register:analytics"]
    assert calls.index("start:data") < calls.index("start:analytics")
    assert calls.index("start:analytics") < calls.index("stop:analytics")
    assert calls.index("stop:analytics") < calls.index("stop:data")


# =============================================================================
# Full pipeline runs
# =============================================================================


@pytest.mark.asyncio
async def test_strategy_tester_full_pipeline_approved_signal_completes(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    result = await tester.run()

    _assert_completed_result(result)
    assert fake_strategy_engine.generated == 1
    assert fake_risk_manager_approve.decisions == 1

    assert len(result.signals) == 1
    assert len(result.risk_decisions) == 1
    assert result.risk_decisions[0].approved is True
    assert result.orders
    assert result.fills
    assert result.positions
    assert result.execution_records
    assert result.position_records

    assert result.initial_balance == pytest.approx(10_000.0)
    assert result.final_balance >= 0.0
    assert result.final_equity >= 0.0
    assert result.portfolio.summary.initial_balance == pytest.approx(10_000.0)
    assert result.analytics is not None

    outcomes = {_outcome_value(signal.outcome) for signal in result.signals}
    assert outcomes & {
        SignalOutcome.CONFIRMED_BY_RISK.value,
        SignalOutcome.ORDER_FILLED.value,
        SignalOutcome.POSITION_OPENED.value,
    }


@pytest.mark.asyncio
async def test_strategy_tester_blocked_signal_does_not_execute(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_block,
) -> None:
    _disable_reports(backtest_config)

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_block,
    )

    result = await tester.run()

    _assert_completed_result(result)
    assert fake_strategy_engine.generated == 1
    assert fake_risk_manager_block.decisions == 1

    assert len(result.signals) == 1
    assert len(result.risk_decisions) == 1
    assert result.risk_decisions[0].blocked is True
    assert result.risk_decisions[0].reason == "test_block"
    assert result.orders == []
    assert result.fills == []
    assert result.positions == []
    assert result.trades == []
    assert _outcome_value(result.signals[0].outcome) == SignalOutcome.BLOCKED_BY_RISK.value


@pytest.mark.asyncio
async def test_strategy_tester_two_signals_are_collected_and_executed(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine_two_signals,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=fake_strategy_engine_two_signals,
        risk_manager=fake_risk_manager_approve,
    )

    result = await tester.run()

    _assert_completed_result(result)
    assert fake_strategy_engine_two_signals.generated == 2
    assert fake_risk_manager_approve.decisions == 2
    assert len(result.signals) == 2
    assert len(result.risk_decisions) == 2
    assert len(result.fills) >= 1


@pytest.mark.asyncio
async def test_strategy_tester_short_signal_pipeline_runs(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_short_strategy_engine,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=fake_short_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    result = await tester.run()

    _assert_completed_result(result)
    assert result.signals
    assert result.signals[0].side == "sell"
    assert result.risk_decisions[0].approved is True
    assert result.fills
    assert result.fills[0].side == "sell"


@pytest.mark.asyncio
async def test_run_backtest_convenience_helper_executes_pipeline(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)

    result = await run_backtest(
        backtest_config,
        sample_dataset,
        event_bus=event_bus,
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    _assert_completed_result(result)
    assert result.signals
    assert result.risk_decisions
    assert result.fills


# =============================================================================
# Mode helpers and reports
# =============================================================================


@pytest.mark.asyncio
async def test_run_single_strategy_sets_strategy_selection(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    result = await tester.run_single_strategy("fake_strategy")

    _assert_completed_result(result)
    assert tester.config.test_all_registered_strategies is False
    assert tester.config.strategies == ["fake_strategy"]


@pytest.mark.asyncio
async def test_run_multi_strategy_sets_mode_and_selection(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine_two_signals,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=fake_strategy_engine_two_signals,
        risk_manager=fake_risk_manager_approve,
    )

    result = await tester.run_multi_strategy(["fake_strategy"])

    _assert_completed_result(result)
    assert result.mode == BacktestMode.MULTI_STRATEGY
    assert tester.config.strategies == ["fake_strategy"]


@pytest.mark.asyncio
async def test_strategy_tester_builds_report_artifacts_when_enabled(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
) -> None:
    backtest_config.save_report = True
    backtest_config.report_builder.enabled = True

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    result = await tester.run()

    _assert_completed_result(result)
    assert result.reports
    assert result.artifacts
    assert any(Path(artifact.path).exists() for artifact in result.artifacts if artifact.path)


# =============================================================================
# Failure behavior
# =============================================================================


@pytest.mark.asyncio
async def test_strategy_tester_raises_when_component_fails_and_stop_on_first_error(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
    fake_failing_component,
) -> None:
    _disable_reports(backtest_config)
    backtest_config.strategy_tester.stop_on_first_error = True

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        data_caches=[fake_failing_component],
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    with pytest.raises(StrategyBacktestRunError):
        await tester.run()

    assert tester.result is not None
    assert tester.result.status == BacktestStatus.FAILED
    assert tester.result.error is not None


@pytest.mark.asyncio
async def test_strategy_tester_returns_failed_result_when_errors_are_non_fatal(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
    fake_failing_component,
) -> None:
    _disable_reports(backtest_config)
    backtest_config.strategy_tester.stop_on_first_error = False

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        data_caches=[fake_failing_component],
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    result = await tester.run()

    assert result.status == BacktestStatus.FAILED
    assert result.error is not None
    assert result.error_details


@pytest.mark.asyncio
async def test_strategy_tester_cleanup_resets_prepared_flag(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
) -> None:
    _disable_reports(backtest_config)
    backtest_config.strategy_tester.cleanup_after_run = True

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )

    result = await tester.run()

    _assert_completed_result(result)
    assert tester._prepared is False
    assert tester._running is False