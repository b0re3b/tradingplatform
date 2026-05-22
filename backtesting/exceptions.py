from __future__ import annotations


class BacktestError(Exception):
    """Base exception for the backtesting package."""


class BacktestConfigurationError(BacktestError):
    """Invalid backtest configuration."""


class BacktestSafetyError(BacktestError):
    """Backtest attempted to use live-trading side effects."""


class BacktestDataError(BacktestError):
    """Historical data is missing, malformed, or unavailable."""


class BacktestReplayError(BacktestError):
    """Replay failed or violated causal ordering."""


class BacktestExecutionError(BacktestError):
    """Paper execution simulation failed."""


class BacktestFactoryError(BacktestError):
    """Backtest production wiring failed."""


class BacktestReportError(BacktestError):
    """Report generation failed."""
