from __future__ import annotations

from enum import Enum


class RiskDecisionType(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REDUCE_SIZE = "reduce_size"
    FORCE_CLOSE = "force_close"
    HALT_TRADING = "halt_trading"
    ONLY_REDUCE = "only_reduce"


class RiskViolationType(str, Enum):
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    DAILY_LOSS_EXCEEDED = "daily_loss_exceeded"
    MAX_EXPOSURE_EXCEEDED = "max_exposure_exceeded"
    MAX_SYMBOL_EXPOSURE_EXCEEDED = "max_symbol_exposure_exceeded"
    MAX_SIDE_EXPOSURE_EXCEEDED = "max_side_exposure_exceeded"
    MAX_OPEN_POSITIONS_EXCEEDED = "max_open_positions_exceeded"
    MAX_LEVERAGE_EXCEEDED = "max_leverage_exceeded"
    CORRELATION_LIMIT_EXCEEDED = "correlation_limit_exceeded"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    POSITION_SIZE_INVALID = "position_size_invalid"
    INSUFFICIENT_MARGIN = "insufficient_margin"
    STOP_LOSS_MISSING = "stop_loss_missing"
    STOP_DISTANCE_INVALID = "stop_distance_invalid"
    TRADING_HALTED = "trading_halted"
    SAFE_MODE_ACTIVE = "safe_mode_active"
    INVALID_REQUEST = "invalid_request"


class RiskLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TradingMode(str, Enum):
    NORMAL = "normal"
    SAFE_MODE = "safe_mode"
    REDUCE_ONLY = "reduce_only"
    HALTED = "halted"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is PositionSide.LONG else -1


class MarginMode(str, Enum):
    ISOLATED = "isolated"
    CROSS = "cross"


class CircuitBreakerReason(str, Enum):
    EXTREME_VOLATILITY = "extreme_volatility"
    LOSS_STREAK = "loss_streak"
    SYSTEM_ERROR_RATE = "system_error_rate"
    LIQUIDITY_COLLAPSE = "liquidity_collapse"
    DRAWDOWN_BREACH = "drawdown_breach"
    DAILY_LOSS_BREACH = "daily_loss_breach"
    MANUAL_HALT = "manual_halt"
    EXECUTION_FAILURES = "execution_failures"
    DATA_FEED_FAILURE = "data_feed_failure"