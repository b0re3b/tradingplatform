from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


@dataclass(slots=True)
class ExecutionCostEstimate:
    """
    Estimated execution cost before opening/increasing a position.

    Використовується ExecutionCostGuard для перевірки:
    cost_to_reward <= allowed threshold
    і expected_value_after_cost > 0.
    """

    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    fee_cost: float = 0.0
    funding_cost: float = 0.0
    other_cost: float = 0.0

    spread_pct: float | None = None
    slippage_pct: float | None = None
    quality: ExecutionQuality = ExecutionQuality.ACCEPTABLE

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return (
            self.spread_cost
            + self.slippage_cost
            + self.fee_cost
            + self.funding_cost
            + self.other_cost
        )


@dataclass(slots=True)
class ExpectedValueSnapshot:
    """
    Risk/reward/EV snapshot for one candidate trade.
    """

    expected_reward: float
    expected_loss: float
    expected_cost: float = 0.0
    win_probability: float | None = None

    risk_reward_ratio: float | None = None
    cost_to_reward_ratio: float | None = None
    expected_value: float | None = None
    expected_value_after_cost: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TierRiskProfile:
    """
    Resolved tier profile after TierRiskGuard.

    requested_tier може бути T4, але final_tier може бути T2,
    якщо система у SAFE_MODE або strategy/symbol має reduced status.
    """

    requested_tier: TradeTier
    final_tier: TradeTier

    risk_units: float
    min_rr: float
    min_expected_value: float
    max_cost_to_reward_pct: float

    default_leverage: float
    max_leverage: float

    downgraded: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskUnitSnapshot:
    """
    Snapshot of current R calculation.

    R — базова одиниця ризику, яка масштабується через режим,
    strategy/symbol multipliers та available budgets.
    """

    base_risk_unit: float
    effective_risk_unit: float

    mode: RiskMode = RiskMode.NORMAL
    mode_multiplier: float = 1.0
    strategy_multiplier: float = 1.0
    symbol_multiplier: float = 1.0
    confidence_multiplier: float = 1.0
    volatility_multiplier: float = 1.0

    capped_by_daily_budget: bool = False
    capped_by_open_risk: bool = False
    capped_by_strategy_budget: bool = False
    capped_by_symbol_budget: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskEvaluationRequest:
    """
    Incoming pre-trade request from Strategy/Signal layer.

    Strategy може запропонувати tier/leverage/size, але RiskManager
    має право знизити tier, leverage або повністю відхилити угоду.
    """

    symbol: str
    side: PositionSide
    entry_price: float
    stop_loss: float | None

    take_profit: float | None = None

    signal_id: str | None = None
    strategy_name: str | None = None

    tier: TradeTier | None = None
    order_intent: OrderIntent = OrderIntent.OPEN
    liquidity_class: LiquidityClass = LiquidityClass.NORMAL
    execution_quality: ExecutionQuality = ExecutionQuality.ACCEPTABLE

    confidence: float | None = None
    edge_score: float | None = None
    volatility: float | None = None

    expected_reward: float | None = None
    expected_loss: float | None = None
    expected_win_probability: float | None = None
    expected_cost: float | None = None
    execution_cost: ExecutionCostEstimate | None = None

    requested_size: float | None = None
    requested_margin: float | None = None
    requested_leverage: float | None = None

    reduce_only: bool = False
    margin_mode: MarginMode = MarginMode.ISOLATED

    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PositionSizeRequest:
    """
    Request for PositionSizer.

    На відміну від старої моделі, тут немає risk_percent.
    Risk amount приходить із RiskUnitCalculator + TierRiskProfile.
    """

    symbol: str
    side: PositionSide
    entry_price: float
    stop_loss: float | None

    account_equity: float
    free_balance: float

    risk_amount: float
    risk_unit_snapshot: RiskUnitSnapshot
    tier_profile: TierRiskProfile

    leverage: float
    margin_mode: MarginMode = MarginMode.ISOLATED

    requested_size: float | None = None
    requested_margin: float | None = None

    confidence: float | None = None
    volatility: float | None = None

    min_size: float | None = None
    max_size: float | None = None
    step_size: float | None = None
    min_notional: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PositionSizeResult:
    """
    Result of position sizing.

    final risk is represented explicitly: risk_amount, notional, margin,
    tier and leverage.
    """

    size: float
    notional_value: float
    margin_required: float

    risk_amount: float
    risk_unit_used: float
    risk_units_used: float

    leverage_used: float
    tier: TradeTier

    stop_distance: float | None = None
    capped: bool = False
    rejected_by_min_size: bool = False
    reason: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskViolation:
    """
    Unified risk violation object.
    """

    violation_type: RiskViolationType
    level: RiskLevel
    message: str

    current_value: float | None = None
    limit_value: float | None = None

    symbol: str | None = None
    strategy_name: str | None = None
    tier: TradeTier | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskCheckResult:
    """
    Result of one guard/check.

    Guard-и не зобовʼязані приймати фінальне рішення. Вони можуть
    повернути adjusted tier/leverage/risk/size, а RiskManager збере
    фінальне RiskDecision.
    """

    passed: bool
    decision: RiskDecisionType

    violations: list[RiskViolation] = field(default_factory=list)

    adjusted_tier: TradeTier | None = None
    adjusted_risk_amount: float | None = None
    adjusted_size: float | None = None
    adjusted_margin: float | None = None
    adjusted_leverage: float | None = None

    risk_mode: RiskMode | None = None
    reason: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskDecision:
    """
    Final decision emitted by RiskManager.

    Це payload-основа для:
    signal.confirmed
    risk.position_blocked
    risk.limit_warning
    risk.kill_switch

    Legacy/dashboard topics such as risk.approved/risk.rejected may still
    reuse the same payload.
    """

    allowed: bool
    decision: RiskDecisionType

    final_size: float | None
    final_leverage: float | None

    reason: str | None

    final_tier: TradeTier | None = None
    final_risk_amount: float | None = None
    final_margin: float | None = None
    final_notional: float | None = None

    reservation_id: str | None = None
    reservation_expires_at: float | None = None

    risk_mode: RiskMode = RiskMode.NORMAL

    risk_reward_ratio: float | None = None
    expected_value: float | None = None
    expected_value_after_cost: float | None = None
    expected_cost: float | None = None
    cost_to_reward_ratio: float | None = None

    signal_id: str | None = None
    strategy_name: str | None = None
    symbol: str | None = None
    side: PositionSide | None = None
    order_intent: OrderIntent | None = None

    violations: list[RiskViolation] = field(default_factory=list)
    checks: dict[str, RiskCheckResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioPosition:
    """
    Runtime portfolio position tracked by RiskState.
    """

    symbol: str
    side: PositionSide
    size: float

    entry_price: float
    mark_price: float
    notional_value: float

    leverage: float | None = None
    margin_used: float = 0.0

    risk_amount: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None

    tier: TradeTier | None = None
    strategy_name: str | None = None
    signal_id: str | None = None
    position_id: str | None = None

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    opened_at: float | None = None
    updated_at: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def signed_notional(self) -> float:
        return self.notional_value * self.side.sign

    @property
    def open_risk(self) -> float:
        return max(0.0, self.risk_amount)


@dataclass(slots=True)
class PendingRiskReservation:
    """
    Temporary reservation created after a risk approval and before execution
    confirms/rejects/cancels the order.

    This model lets RiskState count already-approved but not-yet-opened risk
    in exposure, symbol/strategy budgets and dashboard snapshots.
    """

    reservation_id: str
    symbol: str
    side: PositionSide

    signal_id: str | None = None
    strategy_name: str | None = None
    tier: TradeTier | None = None
    position_id: str | None = None

    size: float = 0.0
    open_risk: float = 0.0
    margin: float = 0.0
    notional: float = 0.0

    created_at: float | None = None
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PendingRiskReservationSnapshot:
    """
    Read-only snapshot of one pending risk reservation.
    """

    reservation_id: str
    symbol: str
    side: PositionSide

    signal_id: str | None = None
    strategy_name: str | None = None
    tier: TradeTier | None = None
    position_id: str | None = None

    size: float = 0.0
    open_risk: float = 0.0
    margin: float = 0.0
    notional: float = 0.0

    created_at: float | None = None
    expires_at: float | None = None
    expired: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskReservationSnapshot:
    """
    Aggregate pending-reservation snapshot for RiskState/dashboard/API.
    """

    reservations_count: int = 0
    total_open_risk: float = 0.0
    total_open_risk_r: float = 0.0
    total_margin: float = 0.0
    total_notional: float = 0.0

    symbol_open_risk: dict[str, float] = field(default_factory=dict)
    strategy_open_risk: dict[str, float] = field(default_factory=dict)
    tier_open_risk: dict[str, float] = field(default_factory=dict)

    reservations: dict[str, PendingRiskReservationSnapshot] = field(default_factory=dict)


@dataclass(slots=True)
class OpenRiskSnapshot:
    """
    Open risk / margin snapshot.

    total_open_risk is conservative: actual_open_risk + pending_open_risk.
    This prevents approved-but-not-yet-filled orders from being invisible to
    downstream guards and dashboards.
    """

    total_open_risk: float
    total_open_risk_r: float

    used_margin: float
    used_margin_pct: float

    symbol_open_risk: dict[str, float] = field(default_factory=dict)
    strategy_open_risk: dict[str, float] = field(default_factory=dict)
    tier_open_risk: dict[str, float] = field(default_factory=dict)

    positions_count: int = 0

    actual_open_risk: float = 0.0
    actual_open_risk_r: float = 0.0

    pending_orders_risk: float = 0.0
    pending_orders_risk_r: float = 0.0
    pending_margin: float = 0.0
    pending_notional: float = 0.0
    pending_reservations_count: int = 0

    projected_open_risk: float = 0.0
    projected_open_risk_r: float = 0.0


@dataclass(slots=True)
class ExposureSnapshot:
    """
    Notional exposure snapshot.

    gross_exposure / total_notional may include pending reservations when
    produced by RiskState. The explicit pending_* fields make that accounting
    visible to dashboards and tests.
    """

    total_notional: float
    gross_exposure: float
    net_exposure: float

    symbol_exposure: dict[str, float]
    side_exposure: dict[str, float]

    leverage_weighted_exposure: float | None = None
    margin_used: float = 0.0
    margin_used_pct: float = 0.0

    actual_notional: float = 0.0
    pending_notional: float = 0.0
    pending_margin: float = 0.0
    pending_symbol_exposure: dict[str, float] = field(default_factory=dict)
    pending_side_exposure: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CorrelationSnapshot:
    """
    Correlation group exposure/open-risk snapshot.
    """

    groups: dict[str, list[str]]
    symbol_to_group: dict[str, str]

    group_exposure: dict[str, float]
    group_open_risk: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class DrawdownSnapshot:
    """
    Account drawdown snapshot.
    """

    peak_equity: float
    current_equity: float

    absolute_drawdown: float
    drawdown_percent: float

    daily_pnl: float
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    loss_streak: int = 0


@dataclass(slots=True)
class RiskBudgetSnapshot:
    """
    Global budget state.
    """

    mode: RiskMode

    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float

    daily_loss_r: float
    weekly_loss_r: float
    monthly_loss_r: float

    caution_daily_loss_r: float
    soft_daily_loss_r: float
    hard_daily_loss_r: float
    weekly_hard_loss_r: float
    monthly_review_loss_r: float
    emergency_stop_loss_r: float

    remaining_daily_r: float
    remaining_weekly_r: float
    remaining_monthly_r: float

    manual_review_required: bool = False
    emergency_stop_active: bool = False


@dataclass(slots=True)
class SymbolRiskSnapshot:
    """
    Symbol-level budget/state snapshot.
    """

    symbol: str
    status: SymbolRiskStatus

    daily_pnl: float = 0.0
    daily_loss_r: float = 0.0
    open_risk: float = 0.0
    open_risk_r: float = 0.0

    trades_today: int = 0
    consecutive_losses: int = 0

    cooldown_until: float | None = None
    disabled_until: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyRiskSnapshot:
    """
    Strategy-level budget/expectancy snapshot.
    """

    strategy_name: str
    status: StrategyRiskStatus

    daily_pnl: float = 0.0
    daily_loss_r: float = 0.0
    open_risk: float = 0.0
    open_risk_r: float = 0.0

    trades_today: int = 0
    consecutive_losses: int = 0

    rolling_expectancy: float | None = None
    rolling_trades: int = 0
    risk_multiplier: float = 1.0

    cooldown_until: float | None = None
    disabled_until: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TierStatsSnapshot:
    """
    Runtime stats grouped by TradeTier.
    """

    tier: TradeTier

    trades: int = 0
    approvals: int = 0
    rejections: int = 0

    realized_pnl: float = 0.0
    open_risk: float = 0.0

    avg_rr: float | None = None
    avg_expected_value: float | None = None
    avg_cost_to_reward: float | None = None


@dataclass(slots=True)
class CircuitBreakerState:
    """
    Runtime circuit breaker state.
    """

    active: bool = False
    reason: CircuitBreakerReason | None = None

    triggered_at: float | None = None
    cooldown_until: float | None = None

    message: str | None = None
    manual_release_required: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskStateSnapshot:
    """
    Full RiskState snapshot for stats/dashboard/API.

    Це snapshot, а не mutable state. Mutable runtime state буде в state.py.
    """

    balance: float
    equity: float
    free_balance: float
    used_margin: float

    realized_pnl: float
    unrealized_pnl: float

    peak_equity: float
    daily_start_equity: float

    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float

    loss_streak: int

    risk_mode: RiskMode
    trading_mode: TradingMode
    trading_halted: bool
    halt_reason: str | None

    positions_count: int

    exposure: ExposureSnapshot
    open_risk: OpenRiskSnapshot
    drawdown: DrawdownSnapshot
    budget: RiskBudgetSnapshot

    circuit_breaker: CircuitBreakerState

    weekly_start_equity: float = 0.0
    monthly_start_equity: float = 0.0

    pending_reservations: RiskReservationSnapshot = field(
        default_factory=RiskReservationSnapshot
    )

    symbols: dict[str, SymbolRiskSnapshot] = field(default_factory=dict)
    strategies: dict[str, StrategyRiskSnapshot] = field(default_factory=dict)
    tiers: dict[TradeTier, TierStatsSnapshot] = field(default_factory=dict)


__all__ = [
    "CircuitBreakerState",
    "CorrelationSnapshot",
    "DrawdownSnapshot",
    "ExecutionCostEstimate",
    "ExpectedValueSnapshot",
    "ExposureSnapshot",
    "OpenRiskSnapshot",
    "PendingRiskReservation",
    "PendingRiskReservationSnapshot",
    "PortfolioPosition",
    "PositionSizeRequest",
    "PositionSizeResult",
    "RiskBudgetSnapshot",
    "RiskCheckResult",
    "RiskDecision",
    "RiskEvaluationRequest",
    "RiskReservationSnapshot",
    "RiskStateSnapshot",
    "RiskUnitSnapshot",
    "RiskViolation",
    "StrategyRiskSnapshot",
    "SymbolRiskSnapshot",
    "TierRiskProfile",
    "TierStatsSnapshot",
]