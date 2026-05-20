"""
Walk-forward testing for backtesting.

WalkForwardRunner evaluates strategy/system robustness across rolling,
anchored or expanding train/validation/test windows.

Typical flow:

    full BacktestDataset
        -> split into walk-forward windows
        -> optional train optimization
        -> optional validation
        -> test run
        -> aggregate test results
        -> calculate stability / overfitting diagnostics

Important:
- WalkForwardRunner does not generate signals directly.
- It does not bypass RiskManager.
- It delegates actual pipeline execution to StrategyTester.
- It can optionally use StrategyOptimizer for train windows.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Callable

from backtesting.config import BacktestConfig, WalkForwardConfig
from backtesting.enums import (
    BacktestMode,
    BacktestStatus,
    BacktestWarningLevel,
    WalkForwardMode,
    WalkForwardWindowType,
)
from backtesting.exceptions import (
    WalkForwardConfigurationError,
    WalkForwardRunError,
    WalkForwardSplitError,
)
from backtesting.models import (
    BacktestDataset,
    BacktestEvent,
    BacktestPeriod,
    BacktestResult,
    PerformanceSummary,
    WalkForwardIterationResult,
    WalkForwardResult,
    WalkForwardWindow,
    safe_div,
)
from backtesting.strategy_tester import StrategyTester

try:
    from core.logger import get_logger
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


StrategyTesterFactory = Callable[[BacktestConfig, BacktestDataset], StrategyTester]
OptimizerFactory = Callable[[BacktestConfig, BacktestDataset], Any]


@dataclass(slots=True)
class WalkForwardSplit:
    """
    One walk-forward split.
    """

    iteration: int
    train_window: WalkForwardWindow
    validation_window: WalkForwardWindow | None
    test_window: WalkForwardWindow

    def windows(self) -> list[WalkForwardWindow]:
        result = [self.train_window]
        if self.validation_window is not None:
            result.append(self.validation_window)
        result.append(self.test_window)
        return result


@dataclass(slots=True)
class WalkForwardRunStats:
    """
    Runtime stats for WalkForwardRunner.
    """

    total_iterations: int = 0
    completed_iterations: int = 0
    failed_iterations: int = 0
    skipped_iterations: int = 0

    train_runs: int = 0
    validation_runs: int = 0
    test_runs: int = 0

    best_test_net_profit_pct: float = 0.0
    worst_test_net_profit_pct: float = 0.0
    average_test_net_profit_pct: float = 0.0
    stability_score: float | None = None
    overfitting_score: float | None = None

    errors: list[str] = field(default_factory=list)


class WalkForwardRunner:
    """
    Runs walk-forward evaluation over a dataset.
    """

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
        *,
        tester_factory: StrategyTesterFactory | None = None,
        optimizer_factory: OptimizerFactory | None = None,
        logger_name: str = "backtesting.walk_forward",
    ) -> None:
        self.config = config or WalkForwardConfig()
        self.config.validate()

        self.tester_factory = tester_factory
        self.optimizer_factory = optimizer_factory
        self.logger = get_logger(logger_name)

        self.stats_state = WalkForwardRunStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        base_config: BacktestConfig,
        dataset: BacktestDataset,
        component_kwargs: dict[str, Any] | None = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward evaluation.

        component_kwargs are passed into StrategyTester when default factory is
        used, for example:
            event_bus, scheduler, data_caches, analytics_components,
            strategy_registry, strategy_engine, signal_processor, risk_manager.
        """

        base_config.validate()

        if dataset.is_empty:
            raise WalkForwardConfigurationError("Cannot run walk-forward on empty dataset.")

        splits = self.split_periods(
            period=base_config.period(),
        )

        self.stats_state.total_iterations = len(splits)

        result = WalkForwardResult(
            mode=self.config.mode,
            metadata={
                "base_run_name": base_config.run_name,
                "iterations": len(splits),
            },
        )

        for split in splits:
            try:
                iteration_result = await self._run_iteration(
                    split=split,
                    base_config=base_config,
                    dataset=dataset,
                    component_kwargs=component_kwargs or {},
                )
                result.iterations.append(iteration_result)
                self.stats_state.completed_iterations += 1

            except Exception as exc:
                self.stats_state.failed_iterations += 1
                self.stats_state.errors.append(str(exc))

                if base_config.fail_fast:
                    raise WalkForwardRunError(
                        "Walk-forward iteration failed.",
                        details={
                            "iteration": split.iteration,
                            "error": str(exc),
                            "error_type": exc.__class__.__name__,
                        },
                    ) from exc

                failed_result = WalkForwardIterationResult(
                    iteration=split.iteration,
                    train_window=split.train_window,
                    validation_window=split.validation_window,
                    test_window=split.test_window,
                    metadata={
                        "failed": True,
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    },
                )
                result.iterations.append(failed_result)

        if self.config.aggregate_results:
            result.aggregated_summary = self.aggregate_results(result.iterations)

        if self.config.calculate_stability_score:
            result.stability_score = self.calculate_stability_score(result.iterations)
            self.stats_state.stability_score = result.stability_score

        if self.config.calculate_overfitting_score:
            result.overfitting_score = self.calculate_overfitting_score(result.iterations)
            self.stats_state.overfitting_score = result.overfitting_score

        self._update_stats_from_result(result)

        if self.stats_state.failed_iterations > 0:
            result.warnings.append(
                self._build_warning(
                    message="Some walk-forward iterations failed.",
                    code="walk_forward_partial_failures",
                    details={
                        "failed_iterations": self.stats_state.failed_iterations,
                        "errors": list(self.stats_state.errors),
                    },
                )
            )

        return result

    def split_periods(
        self,
        *,
        period: BacktestPeriod,
    ) -> list[WalkForwardSplit]:
        """
        Split full period into walk-forward windows.
        """

        if self.config.mode == WalkForwardMode.ROLLING:
            return self._split_rolling(period)

        if self.config.mode == WalkForwardMode.ANCHORED:
            return self._split_anchored(period)

        if self.config.mode == WalkForwardMode.EXPANDING:
            return self._split_expanding(period)

        raise WalkForwardSplitError(
            "Unsupported walk-forward mode.",
            details={"mode": self.config.mode.value},
        )

    def aggregate_results(
        self,
        iterations: list[WalkForwardIterationResult],
    ) -> PerformanceSummary:
        """
        Aggregate test-window results into one summary.
        """

        test_results = [
            item.test_result
            for item in iterations
            if item.test_result is not None and item.test_result.completed_successfully
        ]

        if not test_results:
            return PerformanceSummary(
                key="walk_forward_aggregate",
                metadata={"empty": True},
            )

        initial_balance = test_results[0].initial_balance
        final_equity = initial_balance + sum(item.net_profit for item in test_results)

        net_profit = final_equity - initial_balance
        net_profit_pct = safe_div(net_profit, initial_balance) * 100.0

        summaries = [item.portfolio.summary for item in test_results]
        trade_stats = [item.portfolio.trade_stats for item in test_results]

        total_trades = sum(summary.total_trades for summary in summaries)
        gross_profit = sum(summary.gross_profit for summary in summaries)
        gross_loss = sum(summary.gross_loss for summary in summaries)

        weighted_win_rate = self._weighted_average(
            values=[summary.win_rate for summary in summaries],
            weights=[summary.total_trades for summary in summaries],
        )

        max_drawdown = max((summary.max_drawdown for summary in summaries), default=0.0)
        max_drawdown_pct = max((summary.max_drawdown_pct for summary in summaries), default=0.0)

        sharpe_values = [summary.sharpe_ratio for summary in summaries if summary.sharpe_ratio is not None]
        sortino_values = [summary.sortino_ratio for summary in summaries if summary.sortino_ratio is not None]
        calmar_values = [summary.calmar_ratio for summary in summaries if summary.calmar_ratio is not None]

        return PerformanceSummary(
            key="walk_forward_aggregate",
            initial_balance=initial_balance,
            final_balance=final_equity,
            final_equity=final_equity,
            net_profit=net_profit,
            net_profit_pct=net_profit_pct,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=safe_div(gross_profit, gross_loss),
            expectancy=self._weighted_average(
                values=[summary.expectancy for summary in summaries],
                weights=[summary.total_trades for summary in summaries],
            ),
            total_trades=total_trades,
            win_rate=weighted_win_rate,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            average_drawdown=mean([summary.average_drawdown for summary in summaries]) if summaries else 0.0,
            sharpe_ratio=mean(sharpe_values) if sharpe_values else None,
            sortino_ratio=mean(sortino_values) if sortino_values else None,
            calmar_ratio=mean(calmar_values) if calmar_values else None,
            recovery_factor=safe_div(net_profit, max_drawdown) if max_drawdown > 0 else None,
            exposure_time_pct=mean([summary.exposure_time_pct for summary in summaries]) if summaries else 0.0,
            total_fees=sum(summary.total_fees for summary in summaries),
            total_slippage=sum(summary.total_slippage for summary in summaries),
            total_funding=sum(summary.total_funding for summary in summaries),
            metadata={
                "iterations": len(test_results),
                "avg_test_net_profit_pct": mean([summary.net_profit_pct for summary in summaries]),
                "profitable_iterations": len([summary for summary in summaries if summary.net_profit > 0]),
                "losing_iterations": len([summary for summary in summaries if summary.net_profit < 0]),
                "total_closed_trades": sum(stats.closed_trades for stats in trade_stats),
            },
        )

    def calculate_stability_score(
        self,
        iterations: list[WalkForwardIterationResult],
    ) -> float | None:
        """
        Calculate simple stability score in 0..100.

        Higher is better. It rewards:
        - positive average test return;
        - low variation between windows;
        - high percentage of profitable test windows.
        """

        test_summaries = [
            item.test_result.portfolio.summary
            for item in iterations
            if item.test_result is not None and item.test_result.completed_successfully
        ]

        if not test_summaries:
            return None

        returns = [summary.net_profit_pct for summary in test_summaries]
        profitable_ratio = safe_div(len([value for value in returns if value > 0]), len(returns))

        avg_return = mean(returns)
        volatility = pstdev(returns) if len(returns) > 1 else 0.0

        return_score = max(0.0, min(1.0, (avg_return + 25.0) / 50.0))
        volatility_score = 1.0 / (1.0 + max(0.0, volatility) / 10.0)

        score = (
            0.45 * profitable_ratio
            + 0.35 * return_score
            + 0.20 * volatility_score
        ) * 100.0

        return max(0.0, min(100.0, score))

    def calculate_overfitting_score(
        self,
        iterations: list[WalkForwardIterationResult],
    ) -> float | None:
        """
        Calculate simple overfitting score in 0..100.

        Higher means more overfitting risk.

        It compares train/validation performance against test performance.
        """

        pairs: list[tuple[float, float]] = []

        for item in iterations:
            if item.test_result is None or not item.test_result.completed_successfully:
                continue

            test_return = item.test_result.portfolio.summary.net_profit_pct

            reference_result = item.validation_result or item.train_result

            if reference_result is None or not reference_result.completed_successfully:
                continue

            reference_return = reference_result.portfolio.summary.net_profit_pct
            pairs.append((reference_return, test_return))

        if not pairs:
            return None

        gaps: list[float] = []

        for reference_return, test_return in pairs:
            # Positive gap means train/validation was better than test.
            gap = reference_return - test_return
            gaps.append(max(0.0, gap))

        avg_gap = mean(gaps)
        negative_test_ratio = safe_div(
            len([test for _, test in pairs if test < 0]),
            len(pairs),
        )

        score = min(100.0, avg_gap * 2.0 + negative_test_ratio * 50.0)
        return max(0.0, score)

    # ------------------------------------------------------------------
    # Iteration execution
    # ------------------------------------------------------------------

    async def _run_iteration(
        self,
        *,
        split: WalkForwardSplit,
        base_config: BacktestConfig,
        dataset: BacktestDataset,
        component_kwargs: dict[str, Any],
    ) -> WalkForwardIterationResult:
        """
        Run one walk-forward iteration.
        """

        selected_parameters: dict[str, Any] = {}

        train_result: BacktestResult | None = None
        validation_result: BacktestResult | None = None
        test_result: BacktestResult | None = None

        train_config = self._config_for_window(
            base_config,
            split.train_window,
            suffix=f"wf_{split.iteration}_train",
        )
        train_dataset = self._slice_dataset(dataset, split.train_window.period)

        if self.config.optimize_on_train and self.optimizer_factory is not None:
            optimizer = self.optimizer_factory(train_config, train_dataset)
            optimization_result = await self._maybe_await(optimizer.run())

            best_trial = getattr(optimization_result, "best_trial", None)

            if best_trial is not None:
                selected_parameters = dict(getattr(best_trial, "parameters", {}) or {})

            train_result = getattr(best_trial, "backtest_result", None)

        else:
            train_result = await self._run_backtest_window(
                config=train_config,
                dataset=train_dataset,
                component_kwargs=component_kwargs,
            )

        if split.validation_window is not None and self.config.validate_before_test:
            validation_config = self._config_for_window(
                base_config,
                split.validation_window,
                suffix=f"wf_{split.iteration}_validation",
                selected_parameters=selected_parameters,
            )
            validation_dataset = self._slice_dataset(dataset, split.validation_window.period)
            validation_result = await self._run_backtest_window(
                config=validation_config,
                dataset=validation_dataset,
                component_kwargs=component_kwargs,
            )

        test_config = self._config_for_window(
            base_config,
            split.test_window,
            suffix=f"wf_{split.iteration}_test",
            selected_parameters=selected_parameters,
        )
        test_dataset = self._slice_dataset(dataset, split.test_window.period)
        test_result = await self._run_backtest_window(
            config=test_config,
            dataset=test_dataset,
            component_kwargs=component_kwargs,
        )

        self.stats_state.train_runs += 1
        if validation_result is not None:
            self.stats_state.validation_runs += 1
        self.stats_state.test_runs += 1

        return WalkForwardIterationResult(
            iteration=split.iteration,
            train_window=split.train_window,
            validation_window=split.validation_window,
            test_window=split.test_window,
            train_result=train_result,
            validation_result=validation_result,
            test_result=test_result,
            selected_parameters=selected_parameters,
            metadata={
                "train_events": len(train_dataset.events),
                "validation_events": len(self._slice_dataset(dataset, split.validation_window.period).events)
                if split.validation_window is not None
                else 0,
                "test_events": len(test_dataset.events),
            },
        )

    async def _run_backtest_window(
        self,
        *,
        config: BacktestConfig,
        dataset: BacktestDataset,
        component_kwargs: dict[str, Any],
    ) -> BacktestResult:
        if dataset.is_empty:
            result = BacktestResult(
                run_name=config.run_name,
                mode=config.mode,
                status=BacktestStatus.FAILED,
                period=config.period(),
                initial_balance=config.initial_balance,
                final_balance=config.initial_balance,
                final_equity=config.initial_balance,
                error="Walk-forward window dataset is empty.",
            )
            result.add_warning(
                "Walk-forward window dataset is empty.",
                level=BacktestWarningLevel.ERROR,
                code="empty_walk_forward_window",
            )
            return result

        tester = self._build_tester(
            config=config,
            dataset=dataset,
            component_kwargs=component_kwargs,
        )
        return await tester.run()

    def _build_tester(
        self,
        *,
        config: BacktestConfig,
        dataset: BacktestDataset,
        component_kwargs: dict[str, Any],
    ) -> StrategyTester:
        if self.tester_factory is not None:
            return self.tester_factory(config, dataset)

        return StrategyTester(
            config,
            dataset=dataset,
            **component_kwargs,
        )

    # ------------------------------------------------------------------
    # Split logic
    # ------------------------------------------------------------------

    def _split_rolling(self, period: BacktestPeriod) -> list[WalkForwardSplit]:
        splits: list[WalkForwardSplit] = []

        cursor = period.start
        iteration = 0

        while True:
            train_start = cursor
            train_end = train_start + self.config.train_window

            validation_start: datetime | None = None
            validation_end: datetime | None = None

            if self.config.validation_window is not None:
                validation_start = train_end
                validation_end = validation_start + self.config.validation_window
                test_start = validation_end
            else:
                test_start = train_end

            test_end = test_start + self.config.test_window

            if test_end > period.end:
                break

            split = self._build_split(
                iteration=iteration,
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            )
            splits.append(split)

            iteration += 1

            if self.config.max_iterations is not None and iteration >= self.config.max_iterations:
                break

            cursor = cursor + self.config.step_size

        self._validate_splits(splits)
        return splits

    def _split_anchored(self, period: BacktestPeriod) -> list[WalkForwardSplit]:
        splits: list[WalkForwardSplit] = []

        anchor = period.start
        train_end = anchor + self.config.train_window
        iteration = 0

        while True:
            validation_start: datetime | None = None
            validation_end: datetime | None = None

            if self.config.validation_window is not None:
                validation_start = train_end
                validation_end = validation_start + self.config.validation_window
                test_start = validation_end
            else:
                test_start = train_end

            test_end = test_start + self.config.test_window

            if test_end > period.end:
                break

            split = self._build_split(
                iteration=iteration,
                train_start=anchor,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            )
            splits.append(split)

            iteration += 1

            if self.config.max_iterations is not None and iteration >= self.config.max_iterations:
                break

            train_end = train_end + self.config.step_size

        self._validate_splits(splits)
        return splits

    def _split_expanding(self, period: BacktestPeriod) -> list[WalkForwardSplit]:
        # Expanding is close to anchored, but test window moves forward and
        # train window expands from original start.
        return self._split_anchored(period)

    def _build_split(
        self,
        *,
        iteration: int,
        train_start: datetime,
        train_end: datetime,
        validation_start: datetime | None,
        validation_end: datetime | None,
        test_start: datetime,
        test_end: datetime,
    ) -> WalkForwardSplit:
        train_period = BacktestPeriod(
            start=train_start,
            end=train_end,
        )
        test_period = BacktestPeriod(
            start=test_start,
            end=test_end,
        )

        validation_window = None

        if validation_start is not None and validation_end is not None:
            validation_window = WalkForwardWindow(
                window_id=f"wf_{iteration}_validation",
                window_type=WalkForwardWindowType.VALIDATION,
                period=BacktestPeriod(
                    start=validation_start,
                    end=validation_end,
                ),
                index=iteration,
            )

        return WalkForwardSplit(
            iteration=iteration,
            train_window=WalkForwardWindow(
                window_id=f"wf_{iteration}_train",
                window_type=WalkForwardWindowType.TRAIN,
                period=train_period,
                index=iteration,
            ),
            validation_window=validation_window,
            test_window=WalkForwardWindow(
                window_id=f"wf_{iteration}_test",
                window_type=WalkForwardWindowType.TEST,
                period=test_period,
                index=iteration,
            ),
        )

    def _validate_splits(self, splits: list[WalkForwardSplit]) -> None:
        if not splits:
            raise WalkForwardSplitError(
                "Walk-forward split produced no windows. "
                "Check train_window, validation_window, test_window and full period."
            )

        for split in splits:
            train_days = split.train_window.period.duration.total_seconds() / 86400.0
            test_days = split.test_window.period.duration.total_seconds() / 86400.0

            if train_days < self.config.min_train_days:
                raise WalkForwardSplitError(
                    "Train window is shorter than min_train_days.",
                    details={
                        "iteration": split.iteration,
                        "train_days": train_days,
                        "min_train_days": self.config.min_train_days,
                    },
                )

            if test_days < self.config.min_test_days:
                raise WalkForwardSplitError(
                    "Test window is shorter than min_test_days.",
                    details={
                        "iteration": split.iteration,
                        "test_days": test_days,
                        "min_test_days": self.config.min_test_days,
                    },
                )

    # ------------------------------------------------------------------
    # Dataset slicing / config cloning
    # ------------------------------------------------------------------

    def _slice_dataset(
        self,
        dataset: BacktestDataset,
        period: BacktestPeriod,
    ) -> BacktestDataset:
        start_ms = period.start_ms
        end_ms = period.end_ms

        events = [
            self._copy_event_with_period_flag(event, period)
            for event in dataset.events
            if start_ms <= event.timestamp_ms <= end_ms
        ]

        sliced = BacktestDataset(
            events=events,
            ordering=dataset.ordering,
            replay_mode=dataset.replay_mode,
            metadata={
                **dict(dataset.metadata),
                "sliced": True,
                "period": period.to_dict(),
            },
        )
        sliced.info.period = period
        sliced.info.instruments = list(dataset.info.instruments)
        sliced.info.data_sources = list(dataset.info.data_sources)
        sliced.info.data_types = set(dataset.info.data_types)
        sliced.info.total_events = len(events)

        if events:
            sliced.info.first_event_time = events[0].event_time
            sliced.info.last_event_time = events[-1].event_time

        return sliced

    @staticmethod
    def _copy_event_with_period_flag(
            event: BacktestEvent,
            period: BacktestPeriod,
    ) -> BacktestEvent:
        return event.copy_with(
            is_warmup=period.is_warmup(event.timestamp_ms),
        )

    def _config_for_window(
        self,
        base_config: BacktestConfig,
        window: WalkForwardWindow,
        *,
        suffix: str,
        selected_parameters: dict[str, Any] | None = None,
    ) -> BacktestConfig:
        config = copy.deepcopy(base_config)

        config.run_name = f"{base_config.run_name}_{suffix}"
        config.mode = BacktestMode.MULTI_STRATEGY
        config.start_time = window.period.start
        config.end_time = window.period.end
        config.warmup_start_time = window.period.warmup_start

        config.walk_forward.enabled = False
        config.optimizer.enabled = False

        if selected_parameters:
            self._apply_selected_parameters(config, selected_parameters)

        config.validate()
        return config

    @staticmethod
    def _apply_selected_parameters(
        config: BacktestConfig,
        parameters: dict[str, Any],
    ) -> None:
        """
        Apply selected optimizer parameters to config.

        Supports dotted paths, for example:
            "risk_config.max_risk_per_trade"
            "strategy_config.min_confidence"
            "cost_model.fixed_slippage_bps"
            "execution_simulator.max_volume_participation_pct"
        """

        for path, value in parameters.items():
            parts = str(path).split(".")

            if not parts:
                continue

            target: Any = config

            for part in parts[:-1]:
                target = getattr(target, part, None)
                if target is None:
                    break

            if target is None:
                continue

            attr = parts[-1]

            if hasattr(target, attr):
                setattr(target, attr, value)

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def _update_stats_from_result(self, result: WalkForwardResult) -> None:
        test_returns = [
            item.test_result.portfolio.summary.net_profit_pct
            for item in result.iterations
            if item.test_result is not None and item.test_result.completed_successfully
        ]

        if not test_returns:
            return

        self.stats_state.best_test_net_profit_pct = max(test_returns)
        self.stats_state.worst_test_net_profit_pct = min(test_returns)
        self.stats_state.average_test_net_profit_pct = mean(test_returns)

    @staticmethod
    def _weighted_average(
        *,
        values: list[float],
        weights: list[int | float],
    ) -> float:
        if not values:
            return 0.0

        total_weight = sum(weights)

        if total_weight <= 0:
            return mean(values)

        return sum(value * weight for value, weight in zip(values, weights)) / total_weight

    @staticmethod
    def _build_warning(
        *,
        message: str,
        code: str,
        details: dict[str, Any],
    ) -> Any:
        from .models import BacktestWarning

        return BacktestWarning(
            message=message,
            level=BacktestWarningLevel.WARNING,
            code=code,
            details=details,
        )

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if hasattr(value, "__await__"):
            return await value
        return value

    def stats(self) -> dict[str, Any]:
        return {
            "total_iterations": self.stats_state.total_iterations,
            "completed_iterations": self.stats_state.completed_iterations,
            "failed_iterations": self.stats_state.failed_iterations,
            "skipped_iterations": self.stats_state.skipped_iterations,
            "train_runs": self.stats_state.train_runs,
            "validation_runs": self.stats_state.validation_runs,
            "test_runs": self.stats_state.test_runs,
            "best_test_net_profit_pct": self.stats_state.best_test_net_profit_pct,
            "worst_test_net_profit_pct": self.stats_state.worst_test_net_profit_pct,
            "average_test_net_profit_pct": self.stats_state.average_test_net_profit_pct,
            "stability_score": self.stats_state.stability_score,
            "overfitting_score": self.stats_state.overfitting_score,
            "errors": list(self.stats_state.errors),
        }


# =============================================================================
# Convenience functions
# =============================================================================


async def run_walk_forward(
    *,
    base_config: BacktestConfig,
    dataset: BacktestDataset,
    config: WalkForwardConfig | None = None,
    component_kwargs: dict[str, Any] | None = None,
    tester_factory: StrategyTesterFactory | None = None,
    optimizer_factory: OptimizerFactory | None = None,
) -> WalkForwardResult:
    """
    Convenience helper for async walk-forward execution.
    """

    runner = WalkForwardRunner(
        config=config or base_config.walk_forward,
        tester_factory=tester_factory,
        optimizer_factory=optimizer_factory,
    )
    return await runner.run(
        base_config=base_config,
        dataset=dataset,
        component_kwargs=component_kwargs or {},
    )


def run_walk_forward_sync(
    *,
    base_config: BacktestConfig,
    dataset: BacktestDataset,
    config: WalkForwardConfig | None = None,
    component_kwargs: dict[str, Any] | None = None,
    tester_factory: StrategyTesterFactory | None = None,
    optimizer_factory: OptimizerFactory | None = None,
) -> WalkForwardResult:
    """
    Synchronous wrapper for scripts/notebooks.
    """

    import asyncio

    return asyncio.run(
        run_walk_forward(
            base_config=base_config,
            dataset=dataset,
            config=config,
            component_kwargs=component_kwargs,
            tester_factory=tester_factory,
            optimizer_factory=optimizer_factory,
        )
    )


__all__ = [
    "StrategyTesterFactory",
    "OptimizerFactory",
    "WalkForwardSplit",
    "WalkForwardRunStats",
    "WalkForwardRunner",
    "run_walk_forward",
    "run_walk_forward_sync",
]