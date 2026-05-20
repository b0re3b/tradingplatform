"""
Backtest optimizer.

This module runs repeated backtests with different parameter combinations and
ranks trials by an objective metric.

Important:
- Optimizer does not generate signals directly.
- Optimizer does not approve risk.
- Optimizer does not bypass StrategyTester.
- Each trial must run through the same full pipeline:
  market replay -> analytics -> strategy -> risk -> execution simulator -> position simulator.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import random
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Callable

from backtesting.config import BacktestConfig, OptimizerConfig
from backtesting.enums import (
    BacktestStatus,
    BacktestWarningLevel,
    OptimizationDirection,
    OptimizationMethod,
    OptimizationMetric,
)
from backtesting.exceptions import (
    OptimizationConfigurationError,
    OptimizationMetricError,
    OptimizationParameterError,
    OptimizationRunError,
)
from backtesting.models import (
    BacktestDataset,
    BacktestResult,
    BacktestWarning,
    OptimizationParameter,
    OptimizationResult,
    OptimizationTrialResult,
    utcnow,
)
from .strategy_tester import StrategyTester

try:
    from core.logger import get_logger
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


StrategyTesterFactory = Callable[[BacktestConfig, BacktestDataset], StrategyTester]
TrialCallback = Callable[[OptimizationTrialResult], Any]


@dataclass(slots=True)
class ParameterCandidate:
    """
    One normalized optimizer parameter candidate.
    """

    name: str
    values: list[Any]

    def validate(self) -> None:
        if not self.name:
            raise OptimizationParameterError("Parameter name cannot be empty.")

        if not self.values:
            raise OptimizationParameterError(
                "Parameter candidate values cannot be empty.",
                details={"name": self.name},
            )


@dataclass(slots=True)
class OptimizerRunStats:
    """
    Runtime optimizer stats.
    """

    total_trials: int = 0
    completed_trials: int = 0
    failed_trials: int = 0
    skipped_trials: int = 0

    best_objective_value: float | None = None
    worst_objective_value: float | None = None
    average_objective_value: float | None = None

    started_at: datetime | None = None
    finished_at: datetime | None = None
    errors: list[str] = field(default_factory=list)


class StrategyOptimizer:
    """
    Backtest parameter optimizer.

    Supported modes:
    - grid search;
    - random search;
    - manual list of parameter dictionaries.

    The optimizer expects parameters as dotted paths into BacktestConfig, for example:
    - "cost_model.fixed_slippage_bps"
    - "execution_simulator.max_volume_participation_pct"
    - "position_simulator.default_leverage"
    - "strategy_config.min_confidence"
    - "risk_config.max_risk_per_trade"
    """

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        *,
        tester_factory: StrategyTesterFactory | None = None,
        on_trial_finished: TrialCallback | None = None,
        logger_name: str = "backtesting.optimizer",
    ) -> None:
        self.config = config or OptimizerConfig()
        self.config.validate()

        self.tester_factory = tester_factory
        self.on_trial_finished = on_trial_finished
        self.logger = get_logger(logger_name)

        self.stats_state = OptimizerRunStats()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    async def run(
        self,
        *,
        base_config: BacktestConfig,
        dataset: BacktestDataset,
        component_kwargs: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """
        Run optimization trials.
        """

        base_config.validate()

        if dataset.is_empty:
            raise OptimizationConfigurationError("Cannot optimize on empty dataset.")

        candidates = self.build_parameter_candidates(self.config.parameter_space)
        trials_parameters = self.build_trial_parameter_sets(candidates)

        if self.config.max_trials > 0:
            trials_parameters = trials_parameters[: self.config.max_trials]

        self.stats_state = OptimizerRunStats(
            total_trials=len(trials_parameters),
            started_at=utcnow(),
        )

        result = OptimizationResult(
            objective_metric=self.config.objective_metric,
            direction=self.config.direction,
            metadata={
                "method": self.config.method.value,
                "max_trials": self.config.max_trials,
                "parameter_space": copy.deepcopy(self.config.parameter_space),
            },
        )

        if not trials_parameters:
            raise OptimizationParameterError("Optimizer produced no parameter trials.")

        semaphore = asyncio.Semaphore(max(1, self.config.parallel_jobs))

        if self.config.parallel_jobs <= 1:
            for index, parameters in enumerate(trials_parameters):
                trial = await self._run_trial_guarded(
                    index=index,
                    parameters=parameters,
                    base_config=base_config,
                    dataset=dataset,
                    component_kwargs=component_kwargs or {},
                )
                result.trials.append(trial)
                await self._notify_trial_finished(trial)

                if trial.status == BacktestStatus.FAILED and self.config.stop_on_trial_error:
                    break
        else:
            tasks = [
                self._run_trial_with_semaphore(
                    semaphore=semaphore,
                    index=index,
                    parameters=parameters,
                    base_config=base_config,
                    dataset=dataset,
                    component_kwargs=component_kwargs or {},
                )
                for index, parameters in enumerate(trials_parameters)
            ]

            for task in asyncio.as_completed(tasks):
                trial = await task
                result.trials.append(trial)
                await self._notify_trial_finished(trial)

                if trial.status == BacktestStatus.FAILED and self.config.stop_on_trial_error:
                    break

        result.trials.sort(key=lambda item: item.index)
        result.best_trial = self.select_best_trial(result.trials)
        result.overfitting_score = self.calculate_overfitting_score(result.trials)
        result.parameter_importance = self.estimate_parameter_importance(result.trials)
        result.warnings = self._build_warnings(result.trials)

        self.stats_state.finished_at = utcnow()
        self._update_stats(result.trials)

        return result

    def build_parameter_candidates(
        self,
        parameter_space: dict[str, Any],
    ) -> list[ParameterCandidate]:
        """
        Normalize raw parameter_space into ParameterCandidate list.

        Supported shapes:

        {
            "risk_config.max_risk_per_trade": [0.005, 0.01, 0.015],
            "strategy_config.min_confidence": {
                "values": [0.55, 0.6, 0.65]
            },
            "cost_model.fixed_slippage_bps": {
                "min": 1,
                "max": 5,
                "step": 1
            }
        }
        """

        if not parameter_space:
            raise OptimizationParameterError("parameter_space cannot be empty.")

        candidates: list[ParameterCandidate] = []

        for name, raw_spec in parameter_space.items():
            values = self._expand_parameter_values(name, raw_spec)
            candidate = ParameterCandidate(name=name, values=values)
            candidate.validate()
            candidates.append(candidate)

        return candidates

    def build_trial_parameter_sets(
            self,
            candidates: list[ParameterCandidate],
    ) -> list[dict[str, Any]]:
        """
        Build concrete parameter dictionaries for all trials.
        """

        if not candidates:
            raise OptimizationParameterError("No optimization candidates provided.")

        if self.config.method == OptimizationMethod.GRID_SEARCH:
            return self._build_grid_trials(candidates)

        if self.config.method == OptimizationMethod.RANDOM_SEARCH:
            return self._build_random_trials(candidates)

        if self.config.method == OptimizationMethod.BAYESIAN:
            self.logger.warning(
                "Optimization method %s is not implemented yet; falling back to random search.",
                self.config.method.value,
            )
            return self._build_random_trials(candidates)

        raise OptimizationConfigurationError(
            "Unsupported optimization method.",
            details={"method": self.config.method.value},
        )

    def select_best_trial(
        self,
        trials: list[OptimizationTrialResult],
    ) -> OptimizationTrialResult | None:
        """
        Select best successful trial by objective value.
        """

        successful = [
            trial
            for trial in trials
            if trial.status == BacktestStatus.COMPLETED and trial.backtest_result is not None
        ]

        if not successful:
            return None

        reverse = self.config.direction == OptimizationDirection.MAXIMIZE

        return sorted(
            successful,
            key=lambda item: item.objective_value,
            reverse=reverse,
        )[0]

    def calculate_objective_value(
            self,
            result: BacktestResult,
    ) -> float:
        """
        Extract objective metric value from BacktestResult.
        """

        if result is None:
            raise OptimizationMetricError("BacktestResult is required.")

        summary = result.portfolio.summary

        metric = self.config.objective_metric

        if metric == OptimizationMetric.NET_PROFIT:
            return float(summary.net_profit)

        if metric == OptimizationMetric.NET_PROFIT_PCT:
            return float(summary.net_profit_pct)

        if metric == OptimizationMetric.SHARPE_RATIO:
            return float(summary.sharpe_ratio or 0.0)

        if metric == OptimizationMetric.SORTINO_RATIO:
            return float(summary.sortino_ratio or 0.0)

        if metric == OptimizationMetric.CALMAR_RATIO:
            return float(summary.calmar_ratio or 0.0)

        if metric == OptimizationMetric.PROFIT_FACTOR:
            return float(summary.profit_factor)

        if metric == OptimizationMetric.EXPECTANCY:
            return float(summary.expectancy)

        if metric == OptimizationMetric.EXPECTANCY_R:
            r_values = [
                trade.r_multiple
                for trade in result.trades
                if trade.r_multiple is not None
            ]
            return float(mean(r_values)) if r_values else 0.0

        if metric == OptimizationMetric.MAX_DRAWDOWN:
            return float(summary.max_drawdown)

        if metric == OptimizationMetric.MAX_DRAWDOWN_PCT:
            return float(summary.max_drawdown_pct)

        if metric == OptimizationMetric.WIN_RATE:
            return float(summary.win_rate)

        if metric == OptimizationMetric.RECOVERY_FACTOR:
            return float(summary.recovery_factor or 0.0)

        if metric == OptimizationMetric.CUSTOM:
            custom_value = result.metadata.get("custom_objective_value")

            if custom_value is None:
                raise OptimizationMetricError(
                    "OptimizationMetric.CUSTOM requires result.metadata['custom_objective_value'].",
                    details={"metric": metric.value},
                )

            return float(custom_value)

        raise OptimizationMetricError(
            "Unsupported optimization metric.",
            details={"metric": metric.value},
        )

    def passes_constraints(
        self,
        result: BacktestResult,
    ) -> tuple[bool, str | None]:
        """
        Check hard optimizer constraints.
        """

        summary = result.portfolio.summary

        if summary.total_trades < self.config.min_trades_required:
            return (
                False,
                f"total_trades {summary.total_trades} < min_trades_required {self.config.min_trades_required}",
            )

        if self.config.max_drawdown_pct_limit is not None:
            if summary.max_drawdown_pct > self.config.max_drawdown_pct_limit:
                return (
                    False,
                    f"max_drawdown_pct {summary.max_drawdown_pct} > limit {self.config.max_drawdown_pct_limit}",
                )

        if self.config.min_profit_factor is not None:
            if summary.profit_factor < self.config.min_profit_factor:
                return (
                    False,
                    f"profit_factor {summary.profit_factor} < min_profit_factor {self.config.min_profit_factor}",
                )

        if self.config.min_win_rate is not None:
            if summary.win_rate < self.config.min_win_rate:
                return (
                    False,
                    f"win_rate {summary.win_rate} < min_win_rate {self.config.min_win_rate}",
                )

        return True, None

    def calculate_overfitting_score(
        self,
        trials: list[OptimizationTrialResult],
    ) -> float | None:
        """
        Estimate overfitting risk from trial distribution.

        Higher score means more risk.

        This is a simple diagnostic:
        - many trials with one extreme winner increase risk;
        - high objective dispersion increases risk;
        - low trade count on best trial increases risk.
        """

        successful = [
            trial
            for trial in trials
            if trial.status == BacktestStatus.COMPLETED and trial.backtest_result is not None
        ]

        if len(successful) < 2:
            return None

        values = [trial.objective_value for trial in successful]
        best = self.select_best_trial(successful)

        if best is None or best.backtest_result is None:
            return None

        avg_value = mean(values)
        deviation = pstdev(values) if len(values) > 1 else 0.0

        if self.config.direction == OptimizationDirection.MAXIMIZE:
            edge_over_average = best.objective_value - avg_value
        else:
            edge_over_average = avg_value - best.objective_value

        dispersion_score = min(50.0, abs(deviation) * 2.0)
        winner_gap_score = min(35.0, max(0.0, edge_over_average) * 1.5)

        trades = best.backtest_result.portfolio.summary.total_trades
        low_sample_penalty = 0.0

        if trades < self.config.min_trades_required * 2:
            low_sample_penalty = 15.0

        return min(100.0, dispersion_score + winner_gap_score + low_sample_penalty)

    def estimate_parameter_importance(
        self,
        trials: list[OptimizationTrialResult],
    ) -> dict[str, float]:
        """
        Estimate rough parameter importance from grouped objective means.

        This is not a statistical feature importance model. It is a practical
        diagnostic showing which parameters changed objective value the most.
        """

        successful = [
            trial
            for trial in trials
            if trial.status == BacktestStatus.COMPLETED
        ]

        if len(successful) < 2:
            return {}

        parameter_names = sorted({
            name
            for trial in successful
            for name in trial.parameters.keys()
        })

        values = [trial.objective_value for trial in successful]
        global_range = max(values) - min(values)

        if global_range == 0:
            return {name: 0.0 for name in parameter_names}

        importance: dict[str, float] = {}

        for name in parameter_names:
            grouped: dict[str, list[float]] = {}

            for trial in successful:
                value = trial.parameters.get(name)
                key = repr(value)
                grouped.setdefault(key, []).append(trial.objective_value)

            if len(grouped) <= 1:
                importance[name] = 0.0
                continue

            group_means = [mean(items) for items in grouped.values()]
            effect_range = max(group_means) - min(group_means)
            importance[name] = abs(effect_range) / abs(global_range)

        total = sum(importance.values())

        if total > 0:
            importance = {
                name: value / total
                for name, value in importance.items()
            }

        return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))

    # ---------------------------------------------------------------------
    # Trial execution
    # ---------------------------------------------------------------------

    async def _run_trial_with_semaphore(
        self,
        *,
        semaphore: asyncio.Semaphore,
        index: int,
        parameters: dict[str, Any],
        base_config: BacktestConfig,
        dataset: BacktestDataset,
        component_kwargs: dict[str, Any],
    ) -> OptimizationTrialResult:
        async with semaphore:
            return await self._run_trial_guarded(
                index=index,
                parameters=parameters,
                base_config=base_config,
                dataset=dataset,
                component_kwargs=component_kwargs,
            )

    async def _run_trial_guarded(
        self,
        *,
        index: int,
        parameters: dict[str, Any],
        base_config: BacktestConfig,
        dataset: BacktestDataset,
        component_kwargs: dict[str, Any],
    ) -> OptimizationTrialResult:
        trial = OptimizationTrialResult(
            index=index,
            parameters=copy.deepcopy(parameters),
            objective_metric=self.config.objective_metric,
            direction=self.config.direction,
            status=BacktestStatus.RUNNING,
            started_at=utcnow(),
        )

        try:
            backtest_config = self._build_trial_config(
                base_config=base_config,
                parameters=parameters,
                index=index,
            )

            tester = self._build_tester(
                config=backtest_config,
                dataset=dataset,
                component_kwargs=component_kwargs,
            )

            backtest_result = await tester.run()

            trial.backtest_result = backtest_result
            trial.finished_at = utcnow()

            if not backtest_result.completed_successfully:
                trial.status = BacktestStatus.FAILED
                trial.error = backtest_result.error or "Backtest trial failed."
                self.stats_state.failed_trials += 1
                return trial

            passes, reason = self.passes_constraints(backtest_result)

            if not passes:
                trial.status = BacktestStatus.CANCELLED
                trial.error = reason
                trial.objective_value = self.calculate_objective_value(backtest_result)
                self.stats_state.skipped_trials += 1
                return trial

            trial.objective_value = self.calculate_objective_value(backtest_result)
            trial.status = BacktestStatus.COMPLETED
            self.stats_state.completed_trials += 1
            return trial

        except Exception as exc:
            trial.status = BacktestStatus.FAILED
            trial.finished_at = utcnow()
            trial.error = str(exc)
            trial.metadata["error_type"] = exc.__class__.__name__

            self.stats_state.failed_trials += 1
            self.stats_state.errors.append(str(exc))

            if self.config.stop_on_trial_error:
                raise OptimizationRunError(
                    "Optimization trial failed.",
                    details={
                        "index": index,
                        "parameters": parameters,
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    },
                ) from exc

            return trial

    def _build_trial_config(
        self,
        *,
        base_config: BacktestConfig,
        parameters: dict[str, Any],
        index: int,
    ) -> BacktestConfig:
        config = copy.deepcopy(base_config)
        config.run_name = f"{base_config.run_name}_opt_{index:04d}"
        config.optimizer.enabled = False
        config.walk_forward.enabled = False

        self.apply_parameters(config, parameters)
        config.validate()
        return config

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

    async def _notify_trial_finished(self, trial: OptimizationTrialResult) -> None:
        if self.on_trial_finished is None:
            return

        result = self.on_trial_finished(trial)

        if hasattr(result, "__await__"):
            await result

    # ---------------------------------------------------------------------
    # Parameter expansion
    # ---------------------------------------------------------------------

    def _build_grid_trials(
        self,
        candidates: list[ParameterCandidate],
    ) -> list[dict[str, Any]]:
        names = [candidate.name for candidate in candidates]
        value_sets = [candidate.values for candidate in candidates]

        trials = [
            dict(zip(names, values))
            for values in itertools.product(*value_sets)
        ]

        return trials

    def _build_random_trials(
        self,
        candidates: list[ParameterCandidate],
    ) -> list[dict[str, Any]]:
        rng = random.Random(self.config.random_seed)
        max_trials = max(1, self.config.max_trials)

        trials: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()

        attempts = 0
        max_attempts = max_trials * 20

        while len(trials) < max_trials and attempts < max_attempts:
            attempts += 1

            parameters = {
                candidate.name: rng.choice(candidate.values)
                for candidate in candidates
            }

            key = tuple(sorted((name, repr(value)) for name, value in parameters.items()))

            if key in seen:
                continue

            seen.add(key)
            trials.append(parameters)

        return trials

    def _build_manual_trials(
        self,
        candidates: list[ParameterCandidate],
    ) -> list[dict[str, Any]]:
        """
        Manual mode expects parameter_space like:
            {
                "trials": [
                    {"a.b": 1, "x.y": 2},
                    {"a.b": 2, "x.y": 3},
                ]
            }

        If not provided, falls back to grid candidates.
        """

        trials = self.config.parameter_space.get("trials")

        if isinstance(trials, list):
            normalized: list[dict[str, Any]] = []

            for item in trials:
                if not isinstance(item, dict):
                    raise OptimizationParameterError(
                        "Manual trials must be dictionaries.",
                        details={"trial": item},
                    )
                normalized.append(dict(item))

            return normalized

        return self._build_grid_trials(candidates)

    def _expand_parameter_values(
        self,
        name: str,
        raw_spec: Any,
    ) -> list[Any]:
        if name == "trials":
            return [raw_spec]

        if isinstance(raw_spec, list):
            if not raw_spec:
                raise OptimizationParameterError(
                    "Parameter list cannot be empty.",
                    details={"name": name},
                )
            return list(raw_spec)

        if isinstance(raw_spec, tuple):
            if not raw_spec:
                raise OptimizationParameterError(
                    "Parameter tuple cannot be empty.",
                    details={"name": name},
                )
            return list(raw_spec)

        if isinstance(raw_spec, dict):
            if "values" in raw_spec:
                values = raw_spec["values"]

                if not isinstance(values, (list, tuple)):
                    raise OptimizationParameterError(
                        "Parameter 'values' must be list or tuple.",
                        details={"name": name},
                    )

                if not values:
                    raise OptimizationParameterError(
                        "Parameter 'values' cannot be empty.",
                        details={"name": name},
                    )

                return list(values)

            if {"min", "max", "step"}.issubset(raw_spec.keys()):
                return self._range_values(
                    name=name,
                    minimum=raw_spec["min"],
                    maximum=raw_spec["max"],
                    step=raw_spec["step"],
                )

            if {"minimum", "maximum", "step"}.issubset(raw_spec.keys()):
                return self._range_values(
                    name=name,
                    minimum=raw_spec["minimum"],
                    maximum=raw_spec["maximum"],
                    step=raw_spec["step"],
                )

            if "parameter" in raw_spec:
                parameter = self._parameter_from_dict(name, raw_spec)
                return self._values_from_parameter(parameter)

        # Single constant value.
        return [raw_spec]

    @staticmethod
    def _range_values(
        *,
        name: str,
        minimum: Any,
        maximum: Any,
        step: Any,
    ) -> list[Any]:
        minimum_f = float(minimum)
        maximum_f = float(maximum)
        step_f = float(step)

        if step_f <= 0:
            raise OptimizationParameterError(
                "Parameter range step must be positive.",
                details={"name": name, "step": step},
            )

        if maximum_f < minimum_f:
            raise OptimizationParameterError(
                "Parameter range maximum cannot be less than minimum.",
                details={
                    "name": name,
                    "minimum": minimum,
                    "maximum": maximum,
                },
            )

        values: list[Any] = []
        current = minimum_f

        # Include max with small tolerance.
        while current <= maximum_f + step_f * 1e-9:
            value = round(current, 12)

            if float(value).is_integer() and all(
                float(item).is_integer()
                for item in (minimum_f, maximum_f, step_f)
            ):
                values.append(int(value))
            else:
                values.append(value)

            current += step_f

        return values

    @staticmethod
    def _parameter_from_dict(
        fallback_name: str,
        payload: dict[str, Any],
    ) -> OptimizationParameter:
        parameter_payload = payload.get("parameter")

        if isinstance(parameter_payload, OptimizationParameter):
            return parameter_payload

        if isinstance(parameter_payload, dict):
            return OptimizationParameter(
                name=str(parameter_payload.get("name") or fallback_name),
                values=parameter_payload.get("values"),
                minimum=parameter_payload.get("minimum"),
                maximum=parameter_payload.get("maximum"),
                step=parameter_payload.get("step"),
                distribution=parameter_payload.get("distribution"),
                metadata=dict(parameter_payload.get("metadata") or {}),
            )

        raise OptimizationParameterError(
            "Invalid parameter definition.",
            details={"name": fallback_name, "payload": payload},
        )

    @classmethod
    def _values_from_parameter(
        cls,
        parameter: OptimizationParameter,
    ) -> list[Any]:
        if parameter.values:
            return list(parameter.values)

        if parameter.minimum is not None and parameter.maximum is not None and parameter.step is not None:
            return cls._range_values(
                name=parameter.name,
                minimum=parameter.minimum,
                maximum=parameter.maximum,
                step=parameter.step,
            )

        raise OptimizationParameterError(
            "OptimizationParameter must define values or minimum/maximum/step.",
            details={"name": parameter.name},
        )

    # ---------------------------------------------------------------------
    # Parameter application
    # ---------------------------------------------------------------------

    @staticmethod
    def apply_parameters(
        config: BacktestConfig,
        parameters: dict[str, Any],
    ) -> None:
        """
        Apply dotted-path parameters to BacktestConfig.

        Examples:
            "cost_model.fixed_slippage_bps": 3
            "execution_simulator.max_volume_participation_pct": 5
            "strategy_config.min_confidence": 0.6
        """

        for path, value in parameters.items():
            StrategyOptimizer.apply_parameter(config, path, value)

    @staticmethod
    def apply_parameter(
        config: BacktestConfig,
        path: str,
        value: Any,
    ) -> None:
        parts = str(path).split(".")

        if not parts:
            raise OptimizationParameterError(
                "Parameter path cannot be empty.",
                details={"path": path},
            )

        target: Any = config

        for part in parts[:-1]:
            if isinstance(target, dict):
                target = target.get(part)
            else:
                target = getattr(target, part, None)

            if target is None:
                raise OptimizationParameterError(
                    "Parameter path cannot be resolved.",
                    details={"path": path, "missing_part": part},
                )

        attr = parts[-1]

        if isinstance(target, dict):
            target[attr] = value
            return

        if not hasattr(target, attr):
            raise OptimizationParameterError(
                "Parameter target does not have requested attribute.",
                details={
                    "path": path,
                    "target_type": target.__class__.__name__,
                    "attribute": attr,
                },
            )

        setattr(target, attr, value)

    # ---------------------------------------------------------------------
    # Result diagnostics
    # ---------------------------------------------------------------------

    def _build_warnings(
        self,
        trials: list[OptimizationTrialResult],
    ) -> list[BacktestWarning]:
        warnings: list[BacktestWarning] = []

        failed = [trial for trial in trials if trial.status == BacktestStatus.FAILED]
        skipped = [trial for trial in trials if trial.status == BacktestStatus.CANCELLED]
        completed = [trial for trial in trials if trial.status == BacktestStatus.COMPLETED]

        if failed:
            warnings.append(
                BacktestWarning(
                    message="Some optimization trials failed.",
                    level=BacktestWarningLevel.WARNING,
                    code="optimization_failed_trials",
                    details={
                        "failed_trials": len(failed),
                        "errors": [trial.error for trial in failed[:20]],
                    },
                )
            )

        if skipped:
            warnings.append(
                BacktestWarning(
                    message="Some optimization trials were skipped by constraints.",
                    level=BacktestWarningLevel.INFO,
                    code="optimization_skipped_trials",
                    details={
                        "skipped_trials": len(skipped),
                        "reasons": [trial.error for trial in skipped[:20]],
                    },
                )
            )

        if not completed:
            warnings.append(
                BacktestWarning(
                    message="No optimization trials completed successfully.",
                    level=BacktestWarningLevel.ERROR,
                    code="optimization_no_successful_trials",
                    details={},
                )
            )

        return warnings

    def _update_stats(
        self,
        trials: list[OptimizationTrialResult],
    ) -> None:
        successful = [
            trial
            for trial in trials
            if trial.status == BacktestStatus.COMPLETED
        ]

        if not successful:
            return

        values = [trial.objective_value for trial in successful]

        self.stats_state.best_objective_value = (
            max(values)
            if self.config.direction == OptimizationDirection.MAXIMIZE
            else min(values)
        )
        self.stats_state.worst_objective_value = (
            min(values)
            if self.config.direction == OptimizationDirection.MAXIMIZE
            else max(values)
        )
        self.stats_state.average_objective_value = mean(values)

    def stats(self) -> dict[str, Any]:
        return {
            "total_trials": self.stats_state.total_trials,
            "completed_trials": self.stats_state.completed_trials,
            "failed_trials": self.stats_state.failed_trials,
            "skipped_trials": self.stats_state.skipped_trials,
            "best_objective_value": self.stats_state.best_objective_value,
            "worst_objective_value": self.stats_state.worst_objective_value,
            "average_objective_value": self.stats_state.average_objective_value,
            "started_at": self.stats_state.started_at.isoformat() if self.stats_state.started_at else None,
            "finished_at": self.stats_state.finished_at.isoformat() if self.stats_state.finished_at else None,
            "errors": list(self.stats_state.errors),
        }


# =============================================================================
# Convenience helpers
# =============================================================================


async def run_optimization(
    *,
    base_config: BacktestConfig,
    dataset: BacktestDataset,
    config: OptimizerConfig | None = None,
    component_kwargs: dict[str, Any] | None = None,
    tester_factory: StrategyTesterFactory | None = None,
    on_trial_finished: TrialCallback | None = None,
) -> OptimizationResult:
    """
    Convenience async optimization helper.
    """

    optimizer = StrategyOptimizer(
        config=config or base_config.optimizer,
        tester_factory=tester_factory,
        on_trial_finished=on_trial_finished,
    )

    return await optimizer.run(
        base_config=base_config,
        dataset=dataset,
        component_kwargs=component_kwargs or {},
    )


def run_optimization_sync(
    *,
    base_config: BacktestConfig,
    dataset: BacktestDataset,
    config: OptimizerConfig | None = None,
    component_kwargs: dict[str, Any] | None = None,
    tester_factory: StrategyTesterFactory | None = None,
    on_trial_finished: TrialCallback | None = None,
) -> OptimizationResult:
    """
    Synchronous wrapper for scripts/notebooks.
    """

    return asyncio.run(
        run_optimization(
            base_config=base_config,
            dataset=dataset,
            config=config,
            component_kwargs=component_kwargs,
            tester_factory=tester_factory,
            on_trial_finished=on_trial_finished,
        )
    )


__all__ = [
    "StrategyTesterFactory",
    "TrialCallback",
    "ParameterCandidate",
    "OptimizerRunStats",
    "StrategyOptimizer",
    "run_optimization",
    "run_optimization_sync",
]