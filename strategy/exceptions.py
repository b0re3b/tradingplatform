class StrategyError(Exception):
    """Base exception for strategy package."""


class StrategyConfigError(StrategyError):
    """Raised when strategy config is invalid."""


class StrategyRegistrationError(StrategyError):
    """Raised when a strategy cannot be registered or deregistered."""


class StrategyEvaluationError(StrategyError):
    """Raised when a strategy fails during evaluation."""


class StrategyStateError(StrategyError):
    """Raised when strategy state is inconsistent or unavailable."""


class FeatureStoreError(StrategyStateError):
    """Raised when feature store operations fail."""


class SignalNormalizationError(StrategyError):
    """Raised when analytics payload cannot be normalized into strategy features."""


class SignalRoutingError(StrategyError):
    """Raised when routing logic fails."""


class ConfluenceError(StrategyError):
    """Raised when confluence/scoring stage fails."""


class FilterExecutionError(StrategyError):
    """Raised when a filter cannot be executed correctly."""


class BuilderError(StrategyError):
    """Raised when trade plan builders fail."""


class PortfolioCoordinationError(StrategyError):
    """Raised when portfolio-level signal coordination fails."""


class UnsupportedStrategyError(StrategyError):
    """Raised when strategy type or strategy name is unsupported."""


class ValidationError(StrategyError):
    """Raised when model validation fails."""