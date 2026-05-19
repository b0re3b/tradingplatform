from __future__ import annotations


class ExecutionError(Exception):
    """
    Base exception for the execution domain.

    All execution-specific errors should inherit from this class so callers can
    catch one common exception type at service boundaries.
    """


class ExecutionConfigurationError(ExecutionError):
    """
    Raised when execution configuration is invalid or incomplete.
    """


class ExecutionRejectedError(ExecutionError):
    """
    Raised when an execution request is rejected before order submission.

    Typical reasons:
    - kill switch is active;
    - unsupported execution action;
    - missing risk-approved fields;
    - invalid ExecutionIntent;
    - reduce-only validation failed.
    """


class KillSwitchActiveError(ExecutionRejectedError):
    """
    Raised when a new risk-increasing execution is attempted while kill switch
    or emergency execution lock is active.
    """


class ExchangeClientError(ExecutionError):
    """
    Raised when the configured exchange client is missing, unavailable or
    returns an unexpected response.

    Low-level exchange adapters should stay exchange-focused. Execution services
    may wrap their failures into this exception when crossing execution-domain
    boundaries.
    """


class OrderError(ExecutionError):
    """
    Base exception for order lifecycle errors.
    """


class OrderSubmitError(OrderError):
    """
    Raised when OrderManager cannot submit an order to the exchange.
    """


class OrderCancelError(OrderError):
    """
    Raised when OrderManager cannot cancel an existing order.
    """


class OrderReplaceError(OrderError):
    """
    Raised when OrderManager cannot replace an existing order.

    Binance USD-M Futures does not have a universal atomic replace flow for all
    order types, so replacement may be implemented as cancel + submit.
    """


class OrderNotFoundError(OrderError):
    """
    Raised when an expected order is not found locally or on the exchange.
    """


class OrderStateError(OrderError):
    """
    Raised when an order transition is invalid or inconsistent.

    Example:
    - trying to cancel an already terminal order;
    - receiving a fill for an unknown order;
    - receiving a lower executed quantity than previously recorded.
    """


class PositionError(ExecutionError):
    """
    Base exception for position state and reconciliation errors.
    """


class PositionNotFoundError(PositionError):
    """
    Raised when an expected position does not exist locally or on the exchange.
    """


class PositionStateError(PositionError):
    """
    Raised when a position update is internally inconsistent.
    """


class PositionSyncError(PositionError):
    """
    Raised when PositionManager cannot reconcile local state with exchange
    position snapshots.
    """


class SLTPError(ExecutionError):
    """
    Base exception for stop-loss / take-profit / trailing-stop management.
    """


class ProtectiveOrderError(SLTPError):
    """
    Raised when protective order creation, update or cancellation fails.
    """


class ProtectiveOrderStateError(SLTPError):
    """
    Raised when protective order state becomes inconsistent with position state.

    Example:
    - active SL/TP exists for a closed position;
    - protective order size exceeds current position size;
    - missing reduce-only / close-position semantics.
    """


class SmartExecutionError(ExecutionError):
    """
    Base exception for SmartExecution planning errors.
    """


class ExecutionPlanError(SmartExecutionError):
    """
    Raised when SmartExecution cannot build a valid ExecutionPlan.
    """


class ExecutionPlanValidationError(SmartExecutionError):
    """
    Raised when an ExecutionPlan is structurally invalid or violates
    risk-approved execution constraints.
    """


__all__ = [
    "ExecutionError",
    "ExecutionConfigurationError",
    "ExecutionRejectedError",
    "KillSwitchActiveError",
    "ExchangeClientError",
    "OrderError",
    "OrderSubmitError",
    "OrderCancelError",
    "OrderReplaceError",
    "OrderNotFoundError",
    "OrderStateError",
    "PositionError",
    "PositionNotFoundError",
    "PositionStateError",
    "PositionSyncError",
    "SLTPError",
    "ProtectiveOrderError",
    "ProtectiveOrderStateError",
    "SmartExecutionError",
    "ExecutionPlanError",
    "ExecutionPlanValidationError",
]