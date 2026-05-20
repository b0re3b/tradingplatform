from __future__ import annotations

from execution.config import (
    ExecutionConfig,
    OrderManagerConfig,
    PositionManagerConfig,
    SLTPManagerConfig,
    SmartExecutionConfig,
    TradeExecutorConfig,
)
from execution.enums import (
    ExecutionMode,
    ExecutionStatus,
    OrderSide,
    OrderStatus,
    OrderType,
    SLTPType,
    TimeInForce,
    TriggerType,
    WorkingType,
)
from execution.exceptions import (
    ExchangeClientError,
    ExecutionConfigurationError,
    ExecutionError,
    ExecutionPlanError,
    ExecutionPlanValidationError,
    ExecutionRejectedError,
    KillSwitchActiveError,
    OrderCancelError,
    OrderError,
    OrderNotFoundError,
    OrderReplaceError,
    OrderStateError,
    OrderSubmitError,
    PositionError,
    PositionNotFoundError,
    PositionStateError,
    PositionSyncError,
    ProtectiveOrderError,
    ProtectiveOrderStateError,
    SLTPError,
    SmartExecutionError,
)
from execution.models import (
    ExecutionIntent,
    ExecutionLeg,
    ExecutionPlan,
    ExecutionStats,
    OrderFill,
    OrderManagerStats,
    OrderRequest,
    OrderResult,
    OrderState,
    OrderUpdate,
    PositionManagerStats,
    PositionSnapshot,
    PositionState,
    PositionUpdate,
    ProtectiveOrderState,
    SLTPManagerStats,
    SLTPPlan,
    SmartExecutionStats,
)
from execution.order_manager import (
    BinanceOrderClientProtocol,
    OrderManager,
)
from execution.position_manager import (
    BinancePositionClientProtocol,
    PositionManager,
)
from execution.sl_tp_manager import (
    OrderManagerProtocol as SLTPOrderManagerProtocol,
    SLTPManager,
)
from execution.smart_execution import SmartExecution
from execution.trade_executor import (
    MarketContextProvider,
    OrderManagerProtocol,
    PositionManagerProtocol,
    SLTPManagerProtocol,
    TradeExecutor,
)

__all__ = [
    # Config
    "ExecutionConfig",
    "TradeExecutorConfig",
    "OrderManagerConfig",
    "PositionManagerConfig",
    "SLTPManagerConfig",
    "SmartExecutionConfig",

    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "ExecutionStatus",
    "ExecutionMode",
    "TriggerType",
    "WorkingType",
    "SLTPType",

    # Exceptions
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

    # Models
    "ExecutionIntent",
    "ExecutionPlan",
    "ExecutionLeg",
    "OrderRequest",
    "OrderResult",
    "OrderUpdate",
    "OrderFill",
    "OrderState",
    "PositionState",
    "PositionSnapshot",
    "PositionUpdate",
    "SLTPPlan",
    "ProtectiveOrderState",
    "ExecutionStats",
    "OrderManagerStats",
    "PositionManagerStats",
    "SLTPManagerStats",
    "SmartExecutionStats",

    # Services
    "OrderManager",
    "PositionManager",
    "SLTPManager",
    "SmartExecution",
    "TradeExecutor",

    # Protocols / integration contracts
    "BinanceOrderClientProtocol",
    "BinancePositionClientProtocol",
    "OrderManagerProtocol",
    "PositionManagerProtocol",
    "SLTPManagerProtocol",
    "SLTPOrderManagerProtocol",
    "MarketContextProvider",
]