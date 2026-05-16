from __future__ import annotations

from risk.budget import (
    RiskBudgetGuard,
    RiskModeResolver,
    StrategyRiskGuard,
    SymbolRiskGuard,
)
from risk.circuit_breaker import CircuitBreaker, CircuitBreakerStats
from risk.config import (
    CircuitBreakerConfig,
    ExecutionCostConfig,
    ExposureConfig,
    LeveragePolicyConfig,
    PositionSizingConfig,
    RiskBudgetConfig,
    RiskConfig,
    RiskUnitConfig,
    StrategyRiskConfig,
    SymbolRiskConfig,
    TierModelConfig,
    TierRiskConfig,
)
from risk.enums import (
    CircuitBreakerReason,
    ExecutionQuality,
    LiquidityClass,
    MarginMode,
    OrderIntent,
    PositionSide,
    RiskDecisionType,
    RiskLevel,
    RiskMode,
    RiskViolationType,
    StrategyRiskStatus,
    SymbolRiskStatus,
    TradeTier,
    TradingMode,
)
from risk.exceptions import (
    InvalidPositionSizeError,
    InvalidRiskRequestError,
    RiskConfigurationError,
    RiskError,
)
from risk.exposure_control import ExposureControl
from risk.guards import (
    ExecutionCostGuard,
    LeverageGuard,
    RiskRewardGuard,
    TierRiskGuard,
)
from risk.metrics import (
    GroupMetrics,
    MetricStats,
    RiskMetrics,
)
from risk.models import (
    CircuitBreakerState,
    CorrelationSnapshot,
    DrawdownSnapshot,
    ExecutionCostEstimate,
    ExpectedValueSnapshot,
    ExposureSnapshot,
    OpenRiskSnapshot,
    PortfolioPosition,
    PositionSizeRequest,
    PositionSizeResult,
    RiskBudgetSnapshot,
    RiskCheckResult,
    RiskDecision,
    RiskEvaluationRequest,
    RiskStateSnapshot,
    RiskUnitSnapshot,
    RiskViolation,
    StrategyRiskSnapshot,
    SymbolRiskSnapshot,
    TierRiskProfile,
    TierStatsSnapshot,
)
from risk.position_sizing import (
    PositionSizer,
    RiskUnitCalculator,
    SymbolConstraints,
)
from risk.risk_manager import RiskManager
from risk.state import (
    CooldownState,
    RiskState,
    StrategyRiskState,
    SymbolRiskState,
    TierRuntimeStats,
)
from risk.utils import (
    apply_cap,
    apply_confidence_scale,
    apply_volatility_scale,
    calculate_cost_to_reward_ratio,
    calculate_drawdown_pct,
    calculate_expected_value,
    calculate_loss_r,
    calculate_margin_from_notional,
    calculate_margin_required,
    calculate_notional,
    calculate_pct,
    calculate_pnl,
    calculate_position_size_by_risk,
    calculate_reward_distance,
    calculate_r_units,
    calculate_risk_amount_from_size,
    calculate_risk_reward_ratio,
    calculate_side_aware_stop_distance,
    calculate_stop_distance,
    clamp,
    coalesce_float,
    is_finite_number,
    normalize_confidence,
    normalize_probability,
    round_down_to_step,
    safe_div,
)


__all__ = [
    # Manager
    "RiskManager",

    # Config
    "RiskConfig",
    "RiskUnitConfig",
    "TierRiskConfig",
    "TierModelConfig",
    "RiskBudgetConfig",
    "SymbolRiskConfig",
    "StrategyRiskConfig",
    "ExecutionCostConfig",
    "LeveragePolicyConfig",
    "ExposureConfig",
    "PositionSizingConfig",
    "CircuitBreakerConfig",

    # Enums
    "RiskDecisionType",
    "RiskViolationType",
    "RiskLevel",
    "RiskMode",
    "TradingMode",
    "TradeTier",
    "LiquidityClass",
    "ExecutionQuality",
    "OrderIntent",
    "PositionSide",
    "MarginMode",
    "StrategyRiskStatus",
    "SymbolRiskStatus",
    "CircuitBreakerReason",

    # Exceptions
    "RiskError",
    "RiskConfigurationError",
    "InvalidRiskRequestError",
    "InvalidPositionSizeError",

    # Models
    "RiskEvaluationRequest",
    "PositionSizeRequest",
    "PositionSizeResult",
    "RiskDecision",
    "RiskCheckResult",
    "RiskViolation",
    "PortfolioPosition",
    "ExecutionCostEstimate",
    "ExpectedValueSnapshot",
    "TierRiskProfile",
    "RiskUnitSnapshot",
    "OpenRiskSnapshot",
    "ExposureSnapshot",
    "CorrelationSnapshot",
    "DrawdownSnapshot",
    "RiskBudgetSnapshot",
    "SymbolRiskSnapshot",
    "StrategyRiskSnapshot",
    "TierStatsSnapshot",
    "CircuitBreakerState",
    "RiskStateSnapshot",

    # State
    "RiskState",
    "CooldownState",
    "SymbolRiskState",
    "StrategyRiskState",
    "TierRuntimeStats",

    # Metrics
    "RiskMetrics",
    "MetricStats",
    "GroupMetrics",

    # Position sizing
    "RiskUnitCalculator",
    "PositionSizer",
    "SymbolConstraints",

    # Guards
    "TierRiskGuard",
    "RiskRewardGuard",
    "ExecutionCostGuard",
    "LeverageGuard",
    "RiskModeResolver",
    "RiskBudgetGuard",
    "SymbolRiskGuard",
    "StrategyRiskGuard",
    "ExposureControl",
    "CircuitBreaker",
    "CircuitBreakerStats",

    # Utils
    "clamp",
    "safe_div",
    "calculate_pct",
    "calculate_drawdown_pct",
    "calculate_loss_r",
    "calculate_r_units",
    "calculate_stop_distance",
    "calculate_side_aware_stop_distance",
    "calculate_reward_distance",
    "calculate_risk_reward_ratio",
    "calculate_expected_value",
    "calculate_cost_to_reward_ratio",
    "calculate_notional",
    "calculate_margin_required",
    "calculate_margin_from_notional",
    "calculate_position_size_by_risk",
    "calculate_risk_amount_from_size",
    "calculate_pnl",
    "normalize_probability",
    "normalize_confidence",
    "apply_confidence_scale",
    "apply_volatility_scale",
    "apply_cap",
    "round_down_to_step",
    "coalesce_float",
    "is_finite_number",
]