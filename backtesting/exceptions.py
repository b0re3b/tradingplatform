"""
Backtesting-specific exceptions.

This module defines only exceptions that belong to the offline backtesting,
historical data, market replay, simulation, metrics, reporting, walk-forward
and optimization layers.

Production strategy/risk/execution exceptions should not be duplicated here
unless the failure is specific to backtesting orchestration or simulation.
"""

from __future__ import annotations

from typing import Any


class BacktestError(Exception):
    """
    Base exception for all backtesting package errors.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


# ============================================================================
# Configuration
# ============================================================================


class BacktestConfigurationError(BacktestError):
    """
    Raised when a backtest config is invalid, incomplete or inconsistent.
    """


class BacktestDependencyError(BacktestError):
    """
    Raised when required runtime dependencies are missing.

    Examples:
    - EventBus is not provided;
    - Scheduler is required but missing;
    - RiskManager is missing in full-pipeline mode;
    - StrategyRegistry has no registered strategies.
    """


class BacktestComponentError(BacktestError):
    """
    Raised when a backtesting component cannot be initialized, started,
    registered, stopped or cleaned up.
    """


class BacktestStateError(BacktestError):
    """
    Raised when a backtest component is used in an invalid state.
    """


class BacktestLifecycleError(BacktestError):
    """
    Raised when the backtest lifecycle order is violated.

    Example:
    - calling run before prepare_environment;
    - calling replay before data is loaded;
    - collecting results before the replay is completed.
    """


# ============================================================================
# Historical data / download / load
# ============================================================================


class BacktestDataError(BacktestError):
    """
    Base exception for historical data errors.
    """


class HistoricalDataNotFoundError(BacktestDataError):
    """
    Raised when requested historical data does not exist.
    """


class HistoricalDataDownloadError(BacktestDataError):
    """
    Raised when history_downloader fails to download data.
    """


class HistoricalDataStorageError(BacktestDataError):
    """
    Raised when downloaded or processed historical data cannot be saved,
    read or deleted from storage.
    """


class HistoricalDataFormatError(BacktestDataError):
    """
    Raised when a historical data file has an unsupported or invalid format.
    """


class HistoricalDataSchemaError(BacktestDataError):
    """
    Raised when historical data is missing required fields or contains
    incompatible schema.
    """


class HistoricalDataValidationError(BacktestDataError):
    """
    Raised when historical data fails validation.
    """


class HistoricalDataGapError(BacktestDataError):
    """
    Raised when data gaps violate the configured gap policy.
    """


class HistoricalDataRangeError(BacktestDataError):
    """
    Raised when requested start/end time is invalid or outside available data.
    """


class DataAlignmentError(BacktestDataError):
    """
    Raised when multiple historical streams cannot be aligned consistently.
    """


class DataLoaderError(BacktestDataError):
    """
    Raised when DataLoader fails to load, validate or build a replay dataset.
    """


class EmptyDatasetError(BacktestDataError):
    """
    Raised when a loaded dataset contains no replayable events.
    """


class DatasetIntegrityError(BacktestDataError):
    """
    Raised when a dataset is internally inconsistent.

    Examples:
    - candles are not sorted;
    - duplicated timestamps are invalid for selected stream;
    - candle OHLC values are inconsistent;
    - trade timestamps fall outside the configured period.
    """


class DataLoadError(BacktestDataError):
    """
    Raised when local historical data cannot be loaded from disk.
    """


class DataValidationError(HistoricalDataValidationError):
    """
    Raised when loaded historical data fails validation.
    """


class DataNormalizationError(HistoricalDataValidationError):
    """
    Raised when raw historical rows cannot be normalized into domain records.
    """


class DataGapError(HistoricalDataGapError):
    """
    Raised when historical data has unacceptable timestamp gaps.
    """


class UnsupportedDataTypeError(BacktestDataError):
    """
    Raised when a requested historical data type is not supported.
    """


# ============================================================================
# Backtest clock / time
# ============================================================================


class BacktestTimeError(BacktestError):
    """
    Base exception for backtest clock and simulated time errors.
    """


class BacktestClockError(BacktestTimeError):
    """
    Raised when the backtest clock fails or is used incorrectly.
    """


class BacktestClockNotInitializedError(BacktestTimeError):
    """
    Raised when simulated time is requested before clock initialization.
    """


class BacktestTimeRangeError(BacktestTimeError):
    """
    Raised when simulated time moves outside the configured backtest period.
    """


class BacktestTimeTravelError(BacktestTimeError):
    """
    Raised when the clock attempts to move backwards in a deterministic replay.
    """


# ============================================================================
# Market replay
# ============================================================================


class MarketReplayError(BacktestError):
    """
    Base exception for market replay errors.
    """


class MarketReplayNotPreparedError(MarketReplayError):
    """
    Raised when replay starts before the dataset and dependencies are prepared.
    """


class MarketReplayAlreadyRunningError(MarketReplayError):
    """
    Raised when attempting to start an already running replay.
    """


class MarketReplayStoppedError(MarketReplayError):
    """
    Raised when an operation requires an active replay but replay is stopped.
    """


class MarketReplayPausedError(MarketReplayError):
    """
    Raised when an operation cannot be performed while replay is paused.
    """


class ReplayEventError(MarketReplayError):
    """
    Raised when a historical event cannot be converted to a market.* event.
    """


class ReplayOrderingError(MarketReplayError):
    """
    Raised when replay event ordering is invalid or non-deterministic.
    """


class ReplayEmitError(MarketReplayError):
    """
    Raised when MarketReplay fails to emit an event through EventBus.
    """


class ReplaySeekError(MarketReplayError):
    """
    Raised when seeking to a specific timestamp/event is not possible.
    """


# ============================================================================
# Cost models
# ============================================================================


class CostModelError(BacktestError):
    """
    Base exception for trading cost model errors.
    """


class CommissionCalculationError(CostModelError):
    """
    Raised when commission calculation fails.
    """


class SlippageCalculationError(CostModelError):
    """
    Raised when slippage calculation fails.
    """


class SpreadCostCalculationError(CostModelError):
    """
    Raised when spread cost calculation fails.
    """


class FundingCostCalculationError(CostModelError):
    """
    Raised when funding cost calculation fails.
    """


class BorrowCostCalculationError(CostModelError):
    """
    Raised when borrow cost calculation fails.
    """


class LiquidationPriceCalculationError(CostModelError):
    """
    Raised when liquidation price estimation fails.
    """


class InvalidCostModelInputError(CostModelError):
    """
    Raised when a cost model receives invalid input values.
    """


# ============================================================================
# Execution simulation
# ============================================================================


class ExecutionSimulationError(BacktestError):
    """
    Base exception for simulated execution errors.
    """


class ExecutionSimulatorNotReadyError(ExecutionSimulationError):
    """
    Raised when execution simulator is used before registration/start.
    """


class SimulatedOrderError(ExecutionSimulationError):
    """
    Base exception for simulated order errors.
    """


class SimulatedOrderValidationError(SimulatedOrderError):
    """
    Raised when a simulated order request is invalid.
    """


class SimulatedOrderRejectedError(SimulatedOrderError):
    """
    Raised when a simulated order is rejected by the simulation layer.
    """


class SimulatedOrderNotFoundError(SimulatedOrderError):
    """
    Raised when a simulated order cannot be found.
    """


class SimulatedOrderStateError(SimulatedOrderError):
    """
    Raised when an invalid state transition is attempted for a simulated order.
    """


class SimulatedOrderFillError(SimulatedOrderError):
    """
    Raised when an order cannot be filled or fill calculation fails.
    """


class SimulatedOrderCancelError(SimulatedOrderError):
    """
    Raised when a simulated order cannot be cancelled.
    """


class FillModelError(ExecutionSimulationError):
    """
    Raised when a fill model cannot determine a valid fill.
    """


class LiquiditySimulationError(ExecutionSimulationError):
    """
    Raised when liquidity constraints cannot be evaluated.
    """


class LatencySimulationError(ExecutionSimulationError):
    """
    Raised when latency simulation fails.
    """


class SimulatedExchangeError(ExecutionSimulationError):
    """
    Raised when simulated exchange behavior fails.

    Examples:
    - simulated outage;
    - artificial market halt;
    - unavailable order book;
    - invalid exchange-specific constraints.
    """


# ============================================================================
# Position simulation
# ============================================================================


class PositionSimulationError(BacktestError):
    """
    Base exception for simulated position errors.
    """


class PositionSimulatorNotReadyError(PositionSimulationError):
    """
    Raised when position simulator is used before registration/start.
    """


class SimulatedPositionError(PositionSimulationError):
    """
    Base exception for simulated position errors.
    """


class SimulatedPositionNotFoundError(SimulatedPositionError):
    """
    Raised when a simulated position cannot be found.
    """


class SimulatedPositionValidationError(SimulatedPositionError):
    """
    Raised when a simulated position update is invalid.
    """


class SimulatedPositionStateError(SimulatedPositionError):
    """
    Raised when a simulated position state transition is invalid.
    """


class PositionAccountingError(PositionSimulationError):
    """
    Raised when position accounting fails.
    """


class PnLCalculationError(PositionSimulationError):
    """
    Raised when PnL calculation fails.
    """


class MarginCalculationError(PositionSimulationError):
    """
    Raised when margin calculation fails.
    """


class EquityCalculationError(PositionSimulationError):
    """
    Raised when equity/balance calculation fails.
    """


class LiquidationSimulationError(PositionSimulationError):
    """
    Raised when liquidation check or liquidation event simulation fails.
    """


class ProtectiveOrderSimulationError(PositionSimulationError):
    """
    Raised when simulated stop-loss / take-profit / trailing order handling fails.
    """


# ============================================================================
# Strategy testing / full pipeline
# ============================================================================


class StrategyTestError(BacktestError):
    """
    Base exception for strategy tester errors.
    """


class StrategySelectionError(StrategyTestError):
    """
    Raised when selected strategies cannot be found or resolved.
    """


class StrategyRegistryEmptyError(StrategyTestError):
    """
    Raised when the strategy registry contains no testable strategies.
    """


class StrategyBacktestRunError(StrategyTestError):
    """
    Raised when a strategy test run fails.
    """


class StrategySignalCollectionError(StrategyTestError):
    """
    Raised when generated signals cannot be collected or attributed.
    """


class RiskPipelineError(StrategyTestError):
    """
    Raised when risk processing fails during backtest orchestration.

    This should wrap/augment production risk exceptions instead of replacing
    risk-domain errors.
    """


class ExecutionPipelineError(StrategyTestError):
    """
    Raised when simulated execution pipeline fails during strategy testing.
    """


class PositionPipelineError(StrategyTestError):
    """
    Raised when simulated position pipeline fails during strategy testing.
    """


class BacktestResultCollectionError(StrategyTestError):
    """
    Raised when final results cannot be collected from the pipeline.
    """


# ============================================================================
# Metrics / analytics / reporting
# ============================================================================


class PerformanceCalculationError(BacktestError):
    """
    Base exception for performance metric calculation errors.
    """


class MetricInputError(PerformanceCalculationError):
    """
    Raised when metric calculation receives invalid or incomplete inputs.
    """


class EquityCurveError(PerformanceCalculationError):
    """
    Raised when equity curve construction or validation fails.
    """


class DrawdownCalculationError(PerformanceCalculationError):
    """
    Raised when drawdown calculation fails.
    """


class RatioCalculationError(PerformanceCalculationError):
    """
    Raised when Sharpe, Sortino, Calmar or similar ratio calculation fails.
    """


class TradeStatsCalculationError(PerformanceCalculationError):
    """
    Raised when trade statistics cannot be calculated.
    """


class ModelAnalyticsError(BacktestError):
    """
    Base exception for backtest model analytics errors.
    """


class SignalAttributionError(ModelAnalyticsError):
    """
    Raised when trades or outcomes cannot be attributed to source signals.
    """


class StrategyAttributionError(ModelAnalyticsError):
    """
    Raised when PnL or risk cannot be attributed to strategies.
    """


class RegimeAnalyticsError(ModelAnalyticsError):
    """
    Raised when regime-based performance analytics fail.
    """


class FeatureAnalyticsError(ModelAnalyticsError):
    """
    Raised when feature-level analytics fail.
    """


class ReportBuildError(BacktestError):
    """
    Base exception for report generation errors.
    """


class ReportFormatError(ReportBuildError):
    """
    Raised when requested report format is invalid or unsupported.
    """


class ReportSectionError(ReportBuildError):
    """
    Raised when a report section cannot be built.
    """


class ReportArtifactError(ReportBuildError):
    """
    Raised when report artifacts cannot be saved or exported.
    """


# ============================================================================
# Walk-forward / optimization
# ============================================================================


class WalkForwardError(BacktestError):
    """
    Base exception for walk-forward testing errors.
    """


class WalkForwardConfigurationError(WalkForwardError):
    """
    Raised when walk-forward configuration is invalid.
    """


class WalkForwardSplitError(WalkForwardError):
    """
    Raised when train/validation/test windows cannot be built.
    """


class WalkForwardRunError(WalkForwardError):
    """
    Raised when one walk-forward window run fails.
    """


class WalkForwardAggregationError(WalkForwardError):
    """
    Raised when walk-forward results cannot be aggregated.
    """


class OptimizationError(BacktestError):
    """
    Base exception for strategy/portfolio optimization errors.
    """


class OptimizationConfigurationError(OptimizationError):
    """
    Raised when optimizer configuration is invalid.
    """


class OptimizationParameterError(OptimizationError):
    """
    Raised when an optimization parameter space is invalid.
    """


class OptimizationRunError(OptimizationError):
    """
    Raised when an optimization trial/run fails.
    """


class OptimizationMetricError(OptimizationError):
    """
    Raised when optimization objective metric cannot be calculated.
    """


class OptimizationResultError(OptimizationError):
    """
    Raised when optimization results cannot be ranked, validated or exported.
    """


class OverfittingDetectionError(OptimizationError):
    """
    Raised when overfitting detection fails.
    """


# ============================================================================
# Utility helpers
# ============================================================================


def wrap_backtest_error(
    exc: Exception,
    message: str,
    *,
    code: str | None = None,
    details: dict[str, Any] | None = None,
) -> BacktestError:
    """
    Wrap an arbitrary exception into BacktestError while preserving context.

    This helper is useful at orchestration boundaries where production
    strategy/risk/execution exceptions should be converted into a backtesting
    failure with additional run context.
    """

    if isinstance(exc, BacktestError):
        return exc

    merged_details: dict[str, Any] = {
        "wrapped_error": exc.__class__.__name__,
        "wrapped_message": str(exc),
    }

    if details:
        merged_details.update(details)

    return BacktestError(
        message,
        code=code or "wrapped_backtest_error",
        details=merged_details,
    )


__all__ = [
    # Base
    "BacktestError",
    # Configuration / lifecycle
    "BacktestConfigurationError",
    "BacktestDependencyError",
    "BacktestComponentError",
    "BacktestStateError",
    "BacktestLifecycleError",
    # Historical data
    "BacktestDataError",
    "HistoricalDataNotFoundError",
    "HistoricalDataDownloadError",
    "HistoricalDataStorageError",
    "HistoricalDataFormatError",
    "HistoricalDataSchemaError",
    "HistoricalDataValidationError",
    "HistoricalDataGapError",
    "HistoricalDataRangeError",
    "DataAlignmentError",
    "DataLoaderError",
    "EmptyDatasetError",
    "DatasetIntegrityError",
    "DataLoadError",
    "DataValidationError",
    "DataNormalizationError",
    "DataGapError",
    "UnsupportedDataTypeError",
    # Time
    "BacktestTimeError",
    "BacktestClockError",
    "BacktestClockNotInitializedError",
    "BacktestTimeRangeError",
    "BacktestTimeTravelError",
    # Replay
    "MarketReplayError",
    "MarketReplayNotPreparedError",
    "MarketReplayAlreadyRunningError",
    "MarketReplayStoppedError",
    "MarketReplayPausedError",
    "ReplayEventError",
    "ReplayOrderingError",
    "ReplayEmitError",
    "ReplaySeekError",
    # Cost models
    "CostModelError",
    "CommissionCalculationError",
    "SlippageCalculationError",
    "SpreadCostCalculationError",
    "FundingCostCalculationError",
    "BorrowCostCalculationError",
    "LiquidationPriceCalculationError",
    "InvalidCostModelInputError",
    # Execution simulation
    "ExecutionSimulationError",
    "ExecutionSimulatorNotReadyError",
    "SimulatedOrderError",
    "SimulatedOrderValidationError",
    "SimulatedOrderRejectedError",
    "SimulatedOrderNotFoundError",
    "SimulatedOrderStateError",
    "SimulatedOrderFillError",
    "SimulatedOrderCancelError",
    "FillModelError",
    "LiquiditySimulationError",
    "LatencySimulationError",
    "SimulatedExchangeError",
    # Position simulation
    "PositionSimulationError",
    "PositionSimulatorNotReadyError",
    "SimulatedPositionError",
    "SimulatedPositionNotFoundError",
    "SimulatedPositionValidationError",
    "SimulatedPositionStateError",
    "PositionAccountingError",
    "PnLCalculationError",
    "MarginCalculationError",
    "EquityCalculationError",
    "LiquidationSimulationError",
    "ProtectiveOrderSimulationError",
    # Strategy testing
    "StrategyTestError",
    "StrategySelectionError",
    "StrategyRegistryEmptyError",
    "StrategyBacktestRunError",
    "StrategySignalCollectionError",
    "RiskPipelineError",
    "ExecutionPipelineError",
    "PositionPipelineError",
    "BacktestResultCollectionError",
    # Metrics / analytics / reports
    "PerformanceCalculationError",
    "MetricInputError",
    "EquityCurveError",
    "DrawdownCalculationError",
    "RatioCalculationError",
    "TradeStatsCalculationError",
    "ModelAnalyticsError",
    "SignalAttributionError",
    "StrategyAttributionError",
    "RegimeAnalyticsError",
    "FeatureAnalyticsError",
    "ReportBuildError",
    "ReportFormatError",
    "ReportSectionError",
    "ReportArtifactError",
    # Walk-forward / optimization
    "WalkForwardError",
    "WalkForwardConfigurationError",
    "WalkForwardSplitError",
    "WalkForwardRunError",
    "WalkForwardAggregationError",
    "OptimizationError",
    "OptimizationConfigurationError",
    "OptimizationParameterError",
    "OptimizationRunError",
    "OptimizationMetricError",
    "OptimizationResultError",
    "OverfittingDetectionError",
    # Helpers
    "wrap_backtest_error",
]