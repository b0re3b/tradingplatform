"""
Tests for walk-forward testing and parameter optimization.

Covered modules:
- backtesting.optimizer
- backtesting.walk_forward

These tests focus on orchestration and diagnostics rather than on the full
market/strategy/risk/execution pipeline:
- optimizer parameter expansion and dotted-path config application;
- grid/random trial generation and objective ranking;
- failed/skipped trial handling and optimizer stats;
- walk-forward rolling/anchored splits, dataset slicing and window configs;
- train/validation/test execution through tester factories;
- optimizer-selected parameters carried into validation/test windows;
- partial failure behavior and convenience helpers.

The tests use deterministic fake StrategyTester instances from conftest so they
stay fast, offline and independent from live exchange or production services.
"""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

import pytest

from backtesting.config import BacktestConfig, OptimizerConfig, WalkForwardConfig
from backtesting.enums import (
    BacktestDataType,
    BacktestMode,
    BacktestStatus,
    OptimizationDirection,
    OptimizationMethod,
    OptimizationMetric,
    WalkForwardMode,
    WalkForwardWindowType,
)
from backtesting.exceptions import (
    OptimizationConfigurationError,
    OptimizationMetricError,
    OptimizationParameterError,
    OptimizationRunError,
    WalkForwardConfigurationError,
    WalkForwardRunError,
    WalkForwardSplitError,
)
from backtesting.market_replay import build_dataset_from_records
from backtesting.models import (
    BacktestDataset,
    BacktestPeriod,
    BacktestResult,
    HistoricalCandle,
    OptimizationParameter,
    OptimizationResult,
    OptimizationTrialResult,
    PerformanceSummary,
    PortfolioBacktestResult,
    TradeStats,
    WalkForwardIterationResult,
    timestamp_ms,
)
from backtesting.optimizer import ParameterCandidate, StrategyOptimizer, run_optimization
from backtesting.walk_forward import WalkForwardRunner, run_walk_forward


# =============================================================================
# Compatibility helpers
# =============================================================================


def _status_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _copy_dataclass(instance: Any, **changes: Any) -> Any:
    if not is_dataclass(instance):
        raise TypeError(f"Expected dataclass instance, got {type(instance)!r}")

    payload = {field.name: copy.deepcopy(getattr(instance, field.name)) for field in fields(instance)}
    payload.update(changes)
    return instance.__class__(**payload)


def _fast_walk_forward_config(config: WalkForwardConfig, **changes: Any) -> WalkForwardConfig:
    """
    The package fixture intentionally uses minute-sized windows. The production
    config validates split length in days, so tests lower the min-day guards.
    """

    defaults = {
        "enabled": True,
        "mode": WalkForwardMode.ROLLING,
        "train_window": timedelta(minutes=40),
        "validation_window": timedelta(minutes=20),
        "test_window": timedelta(minutes=20),
        "step_size": timedelta(minutes=20),
        "min_train_days": 0.0001,
        "min_test_days": 0.0001,
        "max_iterations": 3,
        "aggregate_results": True,
        "calculate_stability_score": True,
        "calculate_overfitting_score": True,
    }
    defaults.update(changes)
    return _copy_dataclass(config, **defaults)


def _optimizer_config(config: OptimizerConfig, **changes: Any) -> OptimizerConfig:
    defaults = {
        "enabled": True,
        "method": OptimizationMethod.GRID_SEARCH,
        "objective_metric": OptimizationMetric.NET_PROFIT_PCT,
        "direction": OptimizationDirection.MAXIMIZE,
        "parameter_space": {
            "cost_model.fixed_slippage_bps": [1.0, 2.0],
            "position_simulator.default_leverage": [1.0, 2.0],
        },
        "max_trials": 20,
        "parallel_jobs": 1,
        "min_trades_required": 0,
        "stop_on_trial_error": False,
    }
    defaults.update(changes)
    return _copy_dataclass(config, **defaults)


def _make_result(
    config: BacktestConfig,
    *,
    net_profit_pct: float,
    status: BacktestStatus = BacktestStatus.COMPLETED,
    error: str | None = None,
) -> BacktestResult:
    net_profit = config.initial_balance * net_profit_pct / 100.0
    final_equity = config.initial_balance + net_profit

    summary = PerformanceSummary(
        key="system",
        initial_balance=config.initial_balance,
        final_balance=final_equity,
        final_equity=final_equity,
        net_profit=net_profit,
        net_profit_pct=net_profit_pct,
        gross_profit=max(net_profit, 0.0),
        gross_loss=abs(min(net_profit, 0.0)),
        profit_factor=2.0 if net_profit >= 0 else 0.5,
        expectancy=net_profit,
        total_trades=2,
        win_rate=50.0,
        max_drawdown=100.0,
        max_drawdown_pct=1.0,
        recovery_factor=net_profit / 100.0 if net_profit else 0.0,
    )
    portfolio = PortfolioBacktestResult(
        summary=summary,
        trade_stats=TradeStats(
            total_trades=2,
            closed_trades=2,
            winning_trades=1,
            losing_trades=1,
            win_rate=50.0,
            net_profit=net_profit,
        ),
    )
    result = BacktestResult(
        run_name=config.run_name,
        mode=config.mode,
        status=status,
        period=config.period(),
        initial_balance=config.initial_balance,
        final_balance=final_equity,
        final_equity=final_equity,
        portfolio=portfolio,
        error=error,
    )
    if status == BacktestStatus.COMPLETED:
        result.mark_completed()
    return result


def _profit_pct(result: BacktestResult | None) -> float:
    assert result is not None
    return float(result.portfolio.summary.net_profit_pct)


def _build_dense_dataset(period: BacktestPeriod, *, interval_minutes: int = 10) -> BacktestDataset:
    records: list[HistoricalCandle] = []
    current = period.start
    index = 0

    while current <= period.end:
        open_time = current
        close_time = min(current + timedelta(minutes=1), period.end)
        open_time_ms = timestamp_ms(open_time)
        close_time_ms = timestamp_ms(close_time)
        price = 100.0 + index
        records.append(
            HistoricalCandle(
                exchange="binance",
                symbol="BTCUSDT",
                market_type="usdm_futures",
                timeframe="1m",
                timestamp_ms=timestamp_ms(current),
                received_at_ms=timestamp_ms(current),
                open_time_ms=open_time_ms,
                close_time_ms=close_time_ms,
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price + 0.5,
                volume=1_000.0 + index,
                quote_volume=(1_000.0 + index) * (price + 0.5),
                trades_count=100 + index,
                is_closed=True,
                source="test",
                metadata={"index": index},
            )
        )
        current += timedelta(minutes=interval_minutes)
        index += 1

    return build_dataset_from_records(
        {BacktestDataType.CANDLES: records},
        period=period,
        run_id="walk_forward_test_run",
    )


class _RecordingTesterFactory:
    def __init__(self, *, profit_fn: Callable[[BacktestConfig], float] | None = None) -> None:
        self.configs: list[BacktestConfig] = []
        self.datasets: list[BacktestDataset] = []
        self.profit_fn = profit_fn

    def __call__(self, config: BacktestConfig, dataset: BacktestDataset) -> Any:
        self.configs.append(config)
        self.datasets.append(dataset)
        profit_fn = self.profit_fn

        class Tester:
            async def run(self_nonlocal: Any) -> BacktestResult:
                profit_pct = profit_fn(config) if profit_fn is not None else None
                if profit_pct is None:
                    leverage = float(config.position_simulator.default_leverage)
                    slippage = float(config.cost_model.fixed_slippage_bps)
                    profit_pct = leverage * 2.0 - slippage * 0.1
                return _make_result(config, net_profit_pct=float(profit_pct))

        return Tester()


class _RaisingTesterFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, config: BacktestConfig, dataset: BacktestDataset) -> Any:
        self.calls += 1

        class Tester:
            async def run(self_nonlocal: Any) -> BacktestResult:
                raise RuntimeError("synthetic tester failure")

        return Tester()


class _StaticOptimizer:
    def __init__(
        self,
        *,
        config: BacktestConfig,
        parameters: dict[str, Any],
        train_profit_pct: float = 9.0,
    ) -> None:
        self.config = config
        self.parameters = dict(parameters)
        self.train_profit_pct = train_profit_pct
        self.runs = 0

    async def run(self) -> OptimizationResult:
        self.runs += 1
        backtest_result = _make_result(self.config, net_profit_pct=self.train_profit_pct)
        trial = OptimizationTrialResult(
            index=0,
            parameters=dict(self.parameters),
            objective_metric=OptimizationMetric.NET_PROFIT_PCT,
            objective_value=self.train_profit_pct,
            direction=OptimizationDirection.MAXIMIZE,
            backtest_result=backtest_result,
            status=BacktestStatus.COMPLETED,
        )
        return OptimizationResult(
            trials=[trial],
            best_trial=trial,
            objective_metric=OptimizationMetric.NET_PROFIT_PCT,
            direction=OptimizationDirection.MAXIMIZE,
        )


# =============================================================================
# Optimizer parameter helpers
# =============================================================================


def test_parameter_candidate_validation() -> None:
    candidate = ParameterCandidate(name="cost_model.fixed_slippage_bps", values=[1.0, 2.0])
    candidate.validate()

    with pytest.raises(OptimizationParameterError):
        ParameterCandidate(name="", values=[1.0]).validate()

    with pytest.raises(OptimizationParameterError):
        ParameterCandidate(name="x", values=[]).validate()


def test_optimizer_expands_list_range_and_parameter_specs(optimizer_config: OptimizerConfig) -> None:
    config = _optimizer_config(
        optimizer_config,
        parameter_space={
            "cost_model.fixed_slippage_bps": [1.0, 2.0],
            "position_simulator.default_leverage": {"min": 1, "max": 3, "step": 1},
            "execution_simulator.max_volume_participation_pct": {
                "parameter": OptimizationParameter(
                    name="execution_simulator.max_volume_participation_pct",
                    values=[5.0, 10.0],
                )
            },
        },
    )
    optimizer = StrategyOptimizer(config)

    candidates = optimizer.build_parameter_candidates(config.parameter_space)

    values_by_name = {candidate.name: candidate.values for candidate in candidates}
    assert values_by_name["cost_model.fixed_slippage_bps"] == [1.0, 2.0]
    assert values_by_name["position_simulator.default_leverage"] == [1, 2, 3]
    assert values_by_name["execution_simulator.max_volume_participation_pct"] == [5.0, 10.0]


def test_optimizer_builds_grid_trials(optimizer_config: OptimizerConfig) -> None:
    config = _optimizer_config(optimizer_config)
    optimizer = StrategyOptimizer(config)

    candidates = optimizer.build_parameter_candidates(config.parameter_space)
    trials = optimizer.build_trial_parameter_sets(candidates)

    assert len(trials) == 4
    assert {trial["cost_model.fixed_slippage_bps"] for trial in trials} == {1.0, 2.0}
    assert {trial["position_simulator.default_leverage"] for trial in trials} == {1.0, 2.0}


def test_optimizer_random_trials_are_seeded_and_capped(optimizer_config: OptimizerConfig) -> None:
    config = _optimizer_config(
        optimizer_config,
        method=OptimizationMethod.RANDOM_SEARCH,
        max_trials=3,
        random_seed=123,
        parameter_space={
            "cost_model.fixed_slippage_bps": [1.0, 2.0, 3.0],
            "position_simulator.default_leverage": [1.0, 2.0, 3.0],
        },
    )

    optimizer_a = StrategyOptimizer(config)
    optimizer_b = StrategyOptimizer(config)

    candidates_a = optimizer_a.build_parameter_candidates(config.parameter_space)
    candidates_b = optimizer_b.build_parameter_candidates(config.parameter_space)

    assert optimizer_a.build_trial_parameter_sets(candidates_a) == optimizer_b.build_trial_parameter_sets(candidates_b)
    assert len(optimizer_a.build_trial_parameter_sets(candidates_a)) == 3


def test_optimizer_apply_parameters_uses_dotted_paths(
    backtest_config: BacktestConfig,
    optimizer_config: OptimizerConfig,
) -> None:
    config = copy.deepcopy(backtest_config)

    StrategyOptimizer.apply_parameters(
        config,
        {
            "cost_model.fixed_slippage_bps": 0.5,
            "position_simulator.default_leverage": 4.0,
        },
    )

    assert config.cost_model.fixed_slippage_bps == pytest.approx(0.5)
    assert config.position_simulator.default_leverage == pytest.approx(4.0)

    with pytest.raises(OptimizationParameterError):
        StrategyOptimizer.apply_parameter(config, "missing.path.value", 1)


# =============================================================================
# Optimizer run behavior
# =============================================================================


@pytest.mark.asyncio
async def test_optimizer_grid_run_ranks_best_trial_and_updates_stats(
    backtest_config: BacktestConfig,
    optimizer_config: OptimizerConfig,
    sample_dataset: BacktestDataset,
    fake_tester_factory: Any,
) -> None:
    config = _optimizer_config(optimizer_config)
    callbacks: list[OptimizationTrialResult] = []

    async def on_trial_finished(trial: OptimizationTrialResult) -> None:
        callbacks.append(trial)

    optimizer = StrategyOptimizer(
        config=config,
        tester_factory=fake_tester_factory,
        on_trial_finished=on_trial_finished,
    )

    result = await optimizer.run(base_config=backtest_config, dataset=sample_dataset)

    assert len(result.trials) == 4
    assert len(callbacks) == 4
    assert result.best_trial is not None
    assert result.best_trial.status == BacktestStatus.COMPLETED
    assert result.best_trial.parameters["position_simulator.default_leverage"] == 2.0
    assert result.best_trial.parameters["cost_model.fixed_slippage_bps"] == 1.0
    assert result.best_trial.objective_value == pytest.approx(3.9)
    assert result.overfitting_score is not None
    assert result.parameter_importance

    stats = optimizer.stats()
    assert stats["total_trials"] == 4
    assert stats["completed_trials"] == 4
    assert stats["failed_trials"] == 0
    assert stats["best_objective_value"] == pytest.approx(3.9)


@pytest.mark.asyncio
async def test_optimizer_parallel_run_collects_all_trials(
    backtest_config: BacktestConfig,
    optimizer_config: OptimizerConfig,
    sample_dataset: BacktestDataset,
    fake_tester_factory: Any,
) -> None:
    config = _optimizer_config(optimizer_config, parallel_jobs=2)
    optimizer = StrategyOptimizer(config=config, tester_factory=fake_tester_factory)

    result = await optimizer.run(base_config=backtest_config, dataset=sample_dataset)

    assert [trial.index for trial in result.trials] == [0, 1, 2, 3]
    assert all(trial.status == BacktestStatus.COMPLETED for trial in result.trials)
    assert result.best_trial is not None


@pytest.mark.asyncio
async def test_optimizer_constraints_cancel_trials_and_emit_warning(
    backtest_config: BacktestConfig,
    optimizer_config: OptimizerConfig,
    sample_dataset: BacktestDataset,
    fake_tester_factory: Any,
) -> None:
    config = _optimizer_config(
        optimizer_config,
        min_profit_factor=3.0,
    )
    optimizer = StrategyOptimizer(config=config, tester_factory=fake_tester_factory)

    result = await optimizer.run(base_config=backtest_config, dataset=sample_dataset)

    assert result.best_trial is None
    assert all(trial.status == BacktestStatus.CANCELLED for trial in result.trials)
    assert any(warning.code == "optimization_skipped_trials" for warning in result.warnings)
    assert any(warning.code == "optimization_no_successful_trials" for warning in result.warnings)
    assert optimizer.stats()["skipped_trials"] == len(result.trials)


@pytest.mark.asyncio
async def test_optimizer_failed_backtests_do_not_select_best_trial(
    backtest_config: BacktestConfig,
    optimizer_config: OptimizerConfig,
    sample_dataset: BacktestDataset,
    failing_fake_tester_factory: Any,
) -> None:
    config = _optimizer_config(optimizer_config, max_trials=3)
    optimizer = StrategyOptimizer(config=config, tester_factory=failing_fake_tester_factory)

    result = await optimizer.run(base_config=backtest_config, dataset=sample_dataset)

    assert result.best_trial is None
    assert all(trial.status == BacktestStatus.FAILED for trial in result.trials)
    assert any(warning.code == "optimization_failed_trials" for warning in result.warnings)
    assert optimizer.stats()["failed_trials"] == len(result.trials)


@pytest.mark.asyncio
async def test_optimizer_stops_after_failed_backtest_when_configured(
    backtest_config: BacktestConfig,
    optimizer_config: OptimizerConfig,
    sample_dataset: BacktestDataset,
    failing_fake_tester_factory: Any,
) -> None:
    config = _optimizer_config(
        optimizer_config,
        stop_on_trial_error=True,
        max_trials=20,
    )
    optimizer = StrategyOptimizer(config=config, tester_factory=failing_fake_tester_factory)

    result = await optimizer.run(base_config=backtest_config, dataset=sample_dataset)

    assert len(result.trials) == 1
    assert result.trials[0].status == BacktestStatus.FAILED


@pytest.mark.asyncio
async def test_optimizer_raises_on_empty_dataset(
    backtest_config: BacktestConfig,
    optimizer_config: OptimizerConfig,
) -> None:
    optimizer = StrategyOptimizer(config=_optimizer_config(optimizer_config))

    with pytest.raises(OptimizationConfigurationError):
        await optimizer.run(base_config=backtest_config, dataset=BacktestDataset(events=[]))


def test_optimizer_objective_custom_requires_metadata(
    backtest_config: BacktestConfig,
    optimizer_config: OptimizerConfig,
) -> None:
    config = _optimizer_config(optimizer_config, objective_metric=OptimizationMetric.CUSTOM)
    optimizer = StrategyOptimizer(config=config)
    result = _make_result(backtest_config, net_profit_pct=1.0)

    with pytest.raises(OptimizationMetricError):
        optimizer.calculate_objective_value(result)

    result.metadata["custom_objective_value"] = 12.5
    assert optimizer.calculate_objective_value(result) == pytest.approx(12.5)


def test_optimizer_unsupported_method_raises(
    optimizer_config: OptimizerConfig,
) -> None:
    config = _optimizer_config(optimizer_config, method=OptimizationMethod.GENETIC)
    optimizer = StrategyOptimizer(config=config)
    candidates = optimizer.build_parameter_candidates(config.parameter_space)

    with pytest.raises(OptimizationConfigurationError):
        optimizer.build_trial_parameter_sets(candidates)


@pytest.mark.asyncio
async def test_run_optimization_convenience_helper(
    backtest_config: BacktestConfig,
    optimizer_config: OptimizerConfig,
    sample_dataset: BacktestDataset,
    fake_tester_factory: Any,
) -> None:
    result = await run_optimization(
        base_config=backtest_config,
        dataset=sample_dataset,
        config=_optimizer_config(optimizer_config, max_trials=2),
        tester_factory=fake_tester_factory,
    )

    assert len(result.trials) == 2
    assert result.best_trial is not None


# =============================================================================
# Walk-forward split and helpers
# =============================================================================


def test_walk_forward_rolling_split_builds_expected_windows(
    backtest_config: BacktestConfig,
    walk_forward_config: WalkForwardConfig,
) -> None:
    config = _fast_walk_forward_config(walk_forward_config, mode=WalkForwardMode.ROLLING)
    runner = WalkForwardRunner(config=config)

    splits = runner.split_periods(period=backtest_config.period())

    assert len(splits) == 3
    assert splits[0].iteration == 0
    assert splits[0].train_window.window_type == WalkForwardWindowType.TRAIN
    assert splits[0].validation_window is not None
    assert splits[0].validation_window.window_type == WalkForwardWindowType.VALIDATION
    assert splits[0].test_window.window_type == WalkForwardWindowType.TEST
    assert splits[1].train_window.period.start == splits[0].train_window.period.start + config.step_size


def test_walk_forward_anchored_split_keeps_train_anchor(
    backtest_config: BacktestConfig,
    walk_forward_config: WalkForwardConfig,
) -> None:
    config = _fast_walk_forward_config(
        walk_forward_config,
        mode=WalkForwardMode.ANCHORED,
        max_iterations=2,
    )
    runner = WalkForwardRunner(config=config)

    splits = runner.split_periods(period=backtest_config.period())

    assert len(splits) == 2
    assert splits[0].train_window.period.start == backtest_config.period().start
    assert splits[1].train_window.period.start == backtest_config.period().start
    assert splits[1].train_window.period.end > splits[0].train_window.period.end


def test_walk_forward_split_rejects_too_large_windows(
    backtest_config: BacktestConfig,
    walk_forward_config: WalkForwardConfig,
) -> None:
    config = _fast_walk_forward_config(
        walk_forward_config,
        train_window=timedelta(days=10),
        validation_window=None,
        test_window=timedelta(days=1),
    )
    runner = WalkForwardRunner(config=config)

    with pytest.raises(WalkForwardSplitError):
        runner.split_periods(period=backtest_config.period())


def test_walk_forward_slices_dataset_to_window_period(
    backtest_config: BacktestConfig,
    walk_forward_config: WalkForwardConfig,
) -> None:
    config = _fast_walk_forward_config(walk_forward_config)
    runner = WalkForwardRunner(config=config)
    dataset = _build_dense_dataset(backtest_config.period())
    split = runner.split_periods(period=backtest_config.period())[0]

    sliced = runner._slice_dataset(dataset, split.train_window.period)

    assert sliced.events
    assert sliced.metadata["sliced"] is True
    assert sliced.info.total_events == len(sliced.events)
    assert all(split.train_window.period.start_ms <= event.timestamp_ms <= split.train_window.period.end_ms for event in sliced.events)


def test_walk_forward_config_for_window_applies_selected_parameters(
    backtest_config: BacktestConfig,
    walk_forward_config: WalkForwardConfig,
) -> None:
    config = _fast_walk_forward_config(walk_forward_config)
    runner = WalkForwardRunner(config=config)
    split = runner.split_periods(period=backtest_config.period())[0]

    window_config = runner._config_for_window(
        backtest_config,
        split.test_window,
        suffix="test",
        selected_parameters={
            "cost_model.fixed_slippage_bps": 0.0,
            "position_simulator.default_leverage": 3.0,
        },
    )

    assert window_config.run_name.endswith("_test")
    assert window_config.mode == BacktestMode.MULTI_STRATEGY
    assert window_config.walk_forward.enabled is False
    assert window_config.optimizer.enabled is False
    assert window_config.cost_model.fixed_slippage_bps == pytest.approx(0.0)
    assert window_config.position_simulator.default_leverage == pytest.approx(3.0)


# =============================================================================
# Walk-forward run behavior
# =============================================================================


@pytest.mark.asyncio
async def test_walk_forward_run_without_optimization_executes_train_validation_and_test(
    backtest_config: BacktestConfig,
    walk_forward_config: WalkForwardConfig,
) -> None:
    config = _fast_walk_forward_config(
        walk_forward_config,
        optimize_on_train=False,
        validate_before_test=True,
    )
    dataset = _build_dense_dataset(backtest_config.period())
    factory = _RecordingTesterFactory()
    runner = WalkForwardRunner(config=config, tester_factory=factory)

    result = await runner.run(base_config=backtest_config, dataset=dataset)

    assert len(result.iterations) == 3
    assert all(item.train_result is not None for item in result.iterations)
    assert all(item.validation_result is not None for item in result.iterations)
    assert all(item.test_result is not None for item in result.iterations)
    assert len(factory.configs) == 9
    assert result.aggregated_summary.key == "walk_forward_aggregate"
    assert result.aggregated_summary.metadata["iterations"] == 3
    assert result.stability_score is not None
    assert result.overfitting_score is not None

    stats = runner.stats()
    assert stats["completed_iterations"] == 3
    assert stats["train_runs"] == 3
    assert stats["validation_runs"] == 3
    assert stats["test_runs"] == 3


@pytest.mark.asyncio
async def test_walk_forward_run_with_optimizer_carries_best_parameters_to_test(
    backtest_config: BacktestConfig,
    walk_forward_config: WalkForwardConfig,
) -> None:
    selected_parameters = {
        "position_simulator.default_leverage": 3.0,
        "cost_model.fixed_slippage_bps": 0.0,
    }
    config = _fast_walk_forward_config(
        walk_forward_config,
        optimize_on_train=True,
        validate_before_test=True,
        max_iterations=2,
    )
    dataset = _build_dense_dataset(backtest_config.period())
    tester_factory = _RecordingTesterFactory()
    optimizers: list[_StaticOptimizer] = []

    def optimizer_factory(train_config: BacktestConfig, train_dataset: BacktestDataset) -> _StaticOptimizer:
        optimizer = _StaticOptimizer(
            config=train_config,
            parameters=selected_parameters,
            train_profit_pct=9.0,
        )
        optimizers.append(optimizer)
        return optimizer

    runner = WalkForwardRunner(
        config=config,
        tester_factory=tester_factory,
        optimizer_factory=optimizer_factory,
    )

    result = await runner.run(base_config=backtest_config, dataset=dataset)

    assert len(result.iterations) == 2
    assert len(optimizers) == 2
    assert all(item.selected_parameters == selected_parameters for item in result.iterations)
    assert all(_profit_pct(item.train_result) == pytest.approx(9.0) for item in result.iterations)
    assert all(_profit_pct(item.validation_result) == pytest.approx(6.0) for item in result.iterations)
    assert all(_profit_pct(item.test_result) == pytest.approx(6.0) for item in result.iterations)
    assert all(config.position_simulator.default_leverage == pytest.approx(3.0) for config in tester_factory.configs)
    assert all(config.cost_model.fixed_slippage_bps == pytest.approx(0.0) for config in tester_factory.configs)


@pytest.mark.asyncio
async def test_walk_forward_partial_iteration_failures_become_warnings(
    backtest_config: BacktestConfig,
    walk_forward_config: WalkForwardConfig,
) -> None:
    config = _fast_walk_forward_config(
        walk_forward_config,
        optimize_on_train=False,
        validate_before_test=False,
        max_iterations=2,
    )
    dataset = _build_dense_dataset(backtest_config.period())
    factory = _RaisingTesterFactory()
    runner = WalkForwardRunner(config=config, tester_factory=factory)

    result = await runner.run(base_config=backtest_config, dataset=dataset)

    assert len(result.iterations) == 2
    assert all(item.metadata.get("failed") is True for item in result.iterations)
    assert result.aggregated_summary.metadata.get("empty") is True
    assert any(warning.code == "walk_forward_partial_failures" for warning in result.warnings)
    assert runner.stats()["failed_iterations"] == 2


@pytest.mark.asyncio
async def test_walk_forward_fail_fast_raises_run_error(
    backtest_config: BacktestConfig,
    walk_forward_config: WalkForwardConfig,
) -> None:
    config = _fast_walk_forward_config(
        walk_forward_config,
        optimize_on_train=False,
        validate_before_test=False,
        max_iterations=1,
    )
    base_config = copy.deepcopy(backtest_config)
    base_config.fail_fast = True
    dataset = _build_dense_dataset(base_config.period())
    runner = WalkForwardRunner(config=config, tester_factory=_RaisingTesterFactory())

    with pytest.raises(WalkForwardRunError):
        await runner.run(base_config=base_config, dataset=dataset)


@pytest.mark.asyncio
async def test_walk_forward_rejects_empty_dataset(
    backtest_config: BacktestConfig,
    walk_forward_config: WalkForwardConfig,
) -> None:
    runner = WalkForwardRunner(config=_fast_walk_forward_config(walk_forward_config))

    with pytest.raises(WalkForwardConfigurationError):
        await runner.run(base_config=backtest_config, dataset=BacktestDataset(events=[]))


def test_walk_forward_aggregate_and_scores_from_manual_iterations(
    backtest_config: BacktestConfig,
    walk_forward_config: WalkForwardConfig,
) -> None:
    config = _fast_walk_forward_config(walk_forward_config)
    runner = WalkForwardRunner(config=config)

    iterations = [
        WalkForwardIterationResult(
            iteration=0,
            train_result=_make_result(backtest_config, net_profit_pct=5.0),
            test_result=_make_result(backtest_config, net_profit_pct=3.0),
        ),
        WalkForwardIterationResult(
            iteration=1,
            train_result=_make_result(backtest_config, net_profit_pct=4.0),
            test_result=_make_result(backtest_config, net_profit_pct=-1.0),
        ),
    ]

    summary = runner.aggregate_results(iterations)
    stability = runner.calculate_stability_score(iterations)
    overfitting = runner.calculate_overfitting_score(iterations)

    assert summary.net_profit_pct == pytest.approx(2.0)
    assert summary.total_trades == 4
    assert summary.metadata["profitable_iterations"] == 1
    assert summary.metadata["losing_iterations"] == 1
    assert stability is not None and 0.0 <= stability <= 100.0
    assert overfitting is not None and overfitting > 0.0


@pytest.mark.asyncio
async def test_run_walk_forward_convenience_helper(
    backtest_config: BacktestConfig,
    walk_forward_config: WalkForwardConfig,
) -> None:
    config = _fast_walk_forward_config(
        walk_forward_config,
        optimize_on_train=False,
        validate_before_test=False,
        max_iterations=1,
    )
    dataset = _build_dense_dataset(backtest_config.period())
    factory = _RecordingTesterFactory()

    result = await run_walk_forward(
        base_config=backtest_config,
        dataset=dataset,
        config=config,
        tester_factory=factory,
    )

    assert len(result.iterations) == 1
    assert result.iterations[0].test_result is not None
    assert factory.configs