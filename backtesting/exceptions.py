# trading_system/backtesting/exceptions.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BacktestErrorContext:
    """
    Structured context attached to backtesting exceptions.

    This is intentionally lightweight and serializable so it can be logged,
    included in reports, emitted in system.backtest.* events, or stored with
    backtest run artifacts.
    """

    exchange: str | None = None
    market_type: str | None = None
    symbol: str | None = None
    timeframe: str | None = None

    run_id: str | None = None
    strategy_name: str | None = None
    model_name: str | None = None

    topic: str | None = None
    source: str | None = None

    timestamp_ms: int | None = None
    start_time_ms: int | None = None
    end_time_ms: int | None = None

    data_type: str | None = None
    data_path: str | None = None

    order_id: str | None = None
    signal_id: str | None = None
    position_id: str | None = None
    trade_id: str | None = None

    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-friendly representation without empty values.
        """

        result: dict[str, Any] = {}

        for key in (
            "exchange",
            "market_type",
            "symbol",
            "timeframe",
            "run_id",
            "strategy_name",
            "model_name",
            "topic",
            "source",
            "timestamp_ms",
            "start_time_ms",
            "end_time_ms",
            "data_type",
            "data_path",
            "order_id",
            "signal_id",
            "position_id",
            "trade_id",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value

        if self.details:
            result["details"] = dict(self.details)

        return result


class BacktestError(Exception):
    """
    Base exception for the backtesting package.

    All package-specific exceptions should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        *,
        context: BacktestErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or BacktestErrorContext()
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """
        Return a structured representation useful for logs, reports and events.
        """

        payload: dict[str, Any] = {
            "error_type": self.__class__.__name__,
            "message": self.message,
        }

        context = self.context.to_dict()
        if context:
            payload["context"] = context

        if self.cause is not None:
            payload["cause_type"] = self.cause.__class__.__name__
            payload["cause_message"] = str(self.cause)

        return payload

    def __str__(self) -> str:
        context = self.context.to_dict()
        if not context:
            return self.message

        return f"{self.message} | context={context}"


# ---------------------------------------------------------------------------
# Config / validation errors
# ---------------------------------------------------------------------------


class BacktestConfigError(BacktestError):
    """
    Raised when backtest configuration is invalid.
    """


class BacktestValidationError(BacktestError):
    """
    Raised when runtime validation fails.
    """


class BacktestUnsupportedModeError(BacktestError):
    """
    Raised when a requested mode is not supported by the current implementation.
    """


class BacktestDependencyError(BacktestError):
    """
    Raised when a required dependency was not provided.

    Example:
    - EventBus missing
    - Scheduler missing
    - StrategyEngine missing
    - RiskManager missing
    - data cache missing
    """


# ---------------------------------------------------------------------------
# History download / ingestion errors
# ---------------------------------------------------------------------------


class HistoryDownloadError(BacktestError):
    """
    Base error for historical data downloads from exchanges or providers.
    """


class HistoryDownloadConfigError(HistoryDownloadError):
    """
    Raised when history download configuration is invalid.
    """


class HistoryRequestError(HistoryDownloadError):
    """
    Raised when an exchange/provider request cannot be built or is invalid.
    """


class HistoryResponseError(HistoryDownloadError):
    """
    Raised when an exchange/provider returns malformed or unexpected data.
    """


class HistoryRateLimitError(HistoryDownloadError):
    """
    Raised when an exchange/provider rate limit is reached.
    """


class HistoryAuthenticationError(HistoryDownloadError):
    """
    Raised when a private or provider API request fails due to authentication.

    Most market-history endpoints should be public, but some providers may
    require API credentials.
    """


class HistoryUnavailableError(HistoryDownloadError):
    """
    Raised when the requested data is unavailable for a symbol, exchange,
    market type or date range.
    """


class HistoryNormalizationError(BacktestError):
    """
    Raised when raw exchange/provider data cannot be normalized into the
    internal market.* payload format.
    """


class HistoryWriteError(BacktestError):
    """
    Raised when normalized history cannot be written to local storage.
    """


class HistoryReadError(BacktestError):
    """
    Raised when local historical data cannot be read.
    """


# ---------------------------------------------------------------------------
# Data loading / data quality errors
# ---------------------------------------------------------------------------


class BacktestDataError(BacktestError):
    """
    Base error for backtest data loading and preparation.
    """


class BacktestDataNotFoundError(BacktestDataError):
    """
    Raised when required local historical data is missing.
    """


class BacktestDataSchemaError(BacktestDataError):
    """
    Raised when local historical data has an invalid schema.
    """


class BacktestDataQualityError(BacktestDataError):
    """
    Raised when historical data has gaps, duplicates, invalid ordering,
    invalid timestamps or other quality problems.
    """


class BacktestDataGapError(BacktestDataQualityError):
    """
    Raised when historical data has unacceptable gaps.
    """


class BacktestDataDuplicateError(BacktestDataQualityError):
    """
    Raised when historical data has unacceptable duplicates.
    """


class BacktestDataOrderError(BacktestDataQualityError):
    """
    Raised when historical data is not sorted chronologically and cannot be
    safely replayed.
    """


class BacktestDataRangeError(BacktestDataError):
    """
    Raised when the requested time range is invalid or unavailable.
    """


# ---------------------------------------------------------------------------
# Replay errors
# ---------------------------------------------------------------------------


class BacktestReplayError(BacktestError):
    """
    Base error for market replay failures.
    """


class BacktestReplayStateError(BacktestReplayError):
    """
    Raised when replay lifecycle state is invalid.

    Example:
    - replay started twice
    - replay stopped before it was started
    - replay completed after failure
    """


class BacktestReplayChronologyError(BacktestReplayError):
    """
    Raised when replay receives events that violate chronological ordering.
    """


class BacktestReplayEventError(BacktestReplayError):
    """
    Raised when a HistoricalMarketEvent cannot be emitted or processed.
    """


class BacktestReplayCancelledError(BacktestReplayError):
    """
    Raised when replay is intentionally cancelled.
    """


# ---------------------------------------------------------------------------
# Simulated time / scheduler errors
# ---------------------------------------------------------------------------


class BacktestTimeError(BacktestError):
    """
    Base error for simulated backtest time.
    """


class BacktestClockError(BacktestTimeError):
    """
    Raised when BacktestClock receives an invalid time transition.
    """


class BacktestSchedulerError(BacktestTimeError):
    """
    Raised when BacktestScheduler fails to schedule or execute a simulated job.
    """


class BacktestScheduledJobError(BacktestSchedulerError):
    """
    Raised when a scheduled backtest job fails.
    """


# ---------------------------------------------------------------------------
# Execution simulation errors
# ---------------------------------------------------------------------------


class BacktestExecutionError(BacktestError):
    """
    Base error for simulated execution.
    """


class BacktestOrderError(BacktestExecutionError):
    """
    Raised when a simulated order is invalid or cannot be processed.
    """


class BacktestOrderRejectedError(BacktestExecutionError):
    """
    Raised when the execution simulator rejects an order.
    """


class BacktestFillError(BacktestExecutionError):
    """
    Raised when a simulated fill cannot be produced or validated.
    """


class BacktestSlippageError(BacktestExecutionError):
    """
    Raised when slippage calculation fails.
    """


class BacktestFeeError(BacktestExecutionError):
    """
    Raised when fee calculation fails.
    """


class BacktestLatencyError(BacktestExecutionError):
    """
    Raised when latency simulation fails.
    """


class BacktestLiquidityError(BacktestExecutionError):
    """
    Raised when available simulated liquidity is insufficient or invalid.
    """


# ---------------------------------------------------------------------------
# Position / portfolio simulation errors
# ---------------------------------------------------------------------------


class BacktestPositionError(BacktestError):
    """
    Base error for simulated positions.
    """


class BacktestPositionStateError(BacktestPositionError):
    """
    Raised when a position transition is invalid.

    Example:
    - closing a non-existing position
    - updating a closed position
    - reducing more quantity than exists
    """


class BacktestMarginError(BacktestPositionError):
    """
    Raised when margin calculation or validation fails.
    """


class BacktestLiquidationError(BacktestPositionError):
    """
    Raised when liquidation simulation fails.
    """


class BacktestFundingError(BacktestPositionError):
    """
    Raised when funding payment calculation fails.
    """


class BacktestPnLError(BacktestPositionError):
    """
    Raised when realized or unrealized PnL calculation fails.
    """


# ---------------------------------------------------------------------------
# Metrics / analytics / reports
# ---------------------------------------------------------------------------


class BacktestMetricsError(BacktestError):
    """
    Base error for performance metrics.
    """


class BacktestMetricCalculationError(BacktestMetricsError):
    """
    Raised when a specific metric cannot be calculated.
    """


class BacktestEquityCurveError(BacktestMetricsError):
    """
    Raised when equity curve calculation fails.
    """


class BacktestDrawdownError(BacktestMetricsError):
    """
    Raised when drawdown calculation fails.
    """


class ModelAnalyticsError(BacktestError):
    """
    Base error for model/signal analytics.
    """


class ModelAnalyticsDataError(ModelAnalyticsError):
    """
    Raised when model analytics has insufficient or inconsistent data.
    """


class ModelCalibrationError(ModelAnalyticsError):
    """
    Raised when confidence calibration analysis fails.
    """


class ModelOutcomeLinkError(ModelAnalyticsError):
    """
    Raised when predictions, signals, trades and outcomes cannot be linked.
    """


class BacktestReportError(BacktestError):
    """
    Base error for report generation.
    """


class BacktestReportWriteError(BacktestReportError):
    """
    Raised when a report cannot be written.
    """


# ---------------------------------------------------------------------------
# Walk-forward / optimization errors
# ---------------------------------------------------------------------------


class WalkForwardError(BacktestError):
    """
    Base error for walk-forward testing.
    """


class WalkForwardWindowError(WalkForwardError):
    """
    Raised when walk-forward windows are invalid or overlapping incorrectly.
    """


class OptimizationError(BacktestError):
    """
    Base error for strategy/model parameter optimization.
    """


class OptimizationConfigError(OptimizationError):
    """
    Raised when optimizer configuration is invalid.
    """


class OptimizationObjectiveError(OptimizationError):
    """
    Raised when an optimization objective cannot be evaluated.
    """


class OptimizationTrialError(OptimizationError):
    """
    Raised when a single optimization trial fails.
    """


# ---------------------------------------------------------------------------
# Helper constructors
# ---------------------------------------------------------------------------


def build_error_context(
    *,
    exchange: str | None = None,
    market_type: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    run_id: str | None = None,
    strategy_name: str | None = None,
    model_name: str | None = None,
    topic: str | None = None,
    source: str | None = None,
    timestamp_ms: int | None = None,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    data_type: str | None = None,
    data_path: str | None = None,
    order_id: str | None = None,
    signal_id: str | None = None,
    position_id: str | None = None,
    trade_id: str | None = None,
    **details: Any,
) -> BacktestErrorContext:
    """
    Convenience helper for creating structured error context.

    Example:
        raise BacktestDataNotFoundError(
            "Missing candle history",
            context=build_error_context(
                exchange="binance",
                market_type="usdm_futures",
                symbol="BTCUSDT",
                timeframe="1m",
                data_type="candles",
                data_path="data/history/...",
            ),
        )
    """

    return BacktestErrorContext(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        run_id=run_id,
        strategy_name=strategy_name,
        model_name=model_name,
        topic=topic,
        source=source,
        timestamp_ms=timestamp_ms,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        data_type=data_type,
        data_path=data_path,
        order_id=order_id,
        signal_id=signal_id,
        position_id=position_id,
        trade_id=trade_id,
        details={key: value for key, value in details.items() if value is not None},
    )