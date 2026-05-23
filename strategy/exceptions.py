import logging
class StrategyError(Exception):
    """Base exception for strategy package."""
    _logger = logging.getLogger(__name__ + ".StrategyError")


class StrategyConfigError(StrategyError):
    """Raised when strategy config is invalid."""
    _logger = logging.getLogger(__name__ + ".StrategyConfigError")


class StrategyRegistrationError(StrategyError):
    """Raised when a strategy cannot be registered or deregistered."""
    _logger = logging.getLogger(__name__ + ".StrategyRegistrationError")


class StrategyEvaluationError(StrategyError):
    """Raised when a strategy fails during evaluation."""
    _logger = logging.getLogger(__name__ + ".StrategyEvaluationError")


class StrategyStateError(StrategyError):
    """Raised when strategy state is inconsistent or unavailable."""
    _logger = logging.getLogger(__name__ + ".StrategyStateError")


class FeatureStoreError(StrategyStateError):
    """Raised when feature store operations fail."""
    _logger = logging.getLogger(__name__ + ".FeatureStoreError")


class SignalNormalizationError(StrategyError):
    """Raised when analytics payload cannot be normalized into strategy features."""
    _logger = logging.getLogger(__name__ + ".SignalNormalizationError")


class SignalRoutingError(StrategyError):
    """Raised when routing logic fails."""
    _logger = logging.getLogger(__name__ + ".SignalRoutingError")


class ConfluenceError(StrategyError):
    """Raised when confluence/scoring stage fails."""
    _logger = logging.getLogger(__name__ + ".ConfluenceError")


class FilterExecutionError(StrategyError):
    """Raised when a filter cannot be executed correctly."""
    _logger = logging.getLogger(__name__ + ".FilterExecutionError")


class BuilderError(StrategyError):
    """Raised when trade plan builders fail."""
    _logger = logging.getLogger(__name__ + ".BuilderError")


class PortfolioCoordinationError(StrategyError):
    """Raised when portfolio-level signal coordination fails."""
    _logger = logging.getLogger(__name__ + ".PortfolioCoordinationError")


class UnsupportedStrategyError(StrategyError):
    """Raised when strategy type or strategy name is unsupported."""
    _logger = logging.getLogger(__name__ + ".UnsupportedStrategyError")


class ValidationError(StrategyError):
    """Raised when model validation fails."""
    _logger = logging.getLogger(__name__ + ".ValidationError")