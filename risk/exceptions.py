from __future__ import annotations


class RiskError(Exception):
    """Base exception for risk domain."""


class RiskConfigurationError(RiskError):
    """Raised when risk configuration is invalid."""


class InvalidRiskRequestError(RiskError):
    """Raised when incoming risk request is malformed or inconsistent."""


class InvalidPositionSizeError(RiskError):
    """Raised when calculated or requested position size is invalid."""


class RiskLimitExceededError(RiskError):
    """Raised when a hard risk limit has been exceeded."""


class TradingHaltedError(RiskError):
    """Raised when trading is halted and new actions are not allowed."""


class InsufficientRiskDataError(RiskError):
    """Raised when risk layer does not have enough state/data to decide safely."""