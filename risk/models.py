from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from risk.enums import (
    CircuitBreakerReason,
    MarginMode,
    PositionSide,
    RiskDecisionType,
    RiskLevel,
    RiskViolationType,
    TradingMode,
)


@dataclass(slots=True)
class RiskEvaluationRequest:
    symbol: str
    side: PositionSide
    entry_price: float
    stop_loss: float | None
    take_profit: float | None = None
    signal_id: str | None = None
    strategy_name: str | None = None
    confidence: float | None = None
    requested_size: float | None = None
    requested_leverage: float | None = None
    margin_mode: MarginMode = MarginMode.ISOLATED
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PositionSizeRequest:
    symbol: str
    side: PositionSide
    entry_price: float
    stop_loss: float | None
    account_equity: float
    free_balance: float
    risk_percent: float
    confidence: float | None = None
    volatility: float | None = None
    leverage: float | None = None
    min_size: float | None = None
    max_size: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PositionSizeResult:
    size: float
    notional_value: float
    risk_amount: float
    risk_percent_used: float
    leverage_used: float | None
    capped: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskViolation:
    violation_type: RiskViolationType
    level: RiskLevel
    message: str
    current_value: float | None = None
    limit_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskCheckResult:
    passed: bool
    decision: RiskDecisionType
    violations: list[RiskViolation] = field(default_factory=list)
    adjusted_size: float | None = None
    adjusted_leverage: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskDecision:
    allowed: bool
    decision: RiskDecisionType
    final_size: float | None
    final_leverage: float | None
    reason: str | None
    violations: list[RiskViolation] = field(default_factory=list)
    checks: dict[str, RiskCheckResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioPosition:
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    mark_price: float
    notional_value: float
    leverage: float | None = None
    unrealized_pnl: float = 0.0
    strategy_name: str | None = None
    position_id: str | None = None
    opened_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def signed_notional(self) -> float:
        return self.notional_value * self.side.sign


@dataclass(slots=True)
class ExposureSnapshot:
    total_notional: float
    gross_exposure: float
    net_exposure: float
    symbol_exposure: dict[str, float]
    side_exposure: dict[str, float]
    leverage_weighted_exposure: float | None = None


@dataclass(slots=True)
class DrawdownSnapshot:
    peak_equity: float
    current_equity: float
    absolute_drawdown: float
    drawdown_percent: float
    daily_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    loss_streak: int


@dataclass(slots=True)
class CorrelationSnapshot:
    groups: dict[str, list[str]]
    symbol_to_group: dict[str, str]
    group_exposure: dict[str, float]


@dataclass(slots=True)
class CircuitBreakerState:
    active: bool = False
    reason: CircuitBreakerReason | None = None
    triggered_at: float | None = None
    cooldown_until: float | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskStateSnapshot:
    balance: float
    equity: float
    free_balance: float
    used_margin: float
    realized_pnl: float
    unrealized_pnl: float
    peak_equity: float
    daily_start_equity: float
    daily_pnl: float
    loss_streak: int
    trading_mode: TradingMode
    trading_halted: bool
    halt_reason: str | None
    positions_count: int
    exposure: ExposureSnapshot
    drawdown: DrawdownSnapshot
    circuit_breaker: CircuitBreakerState