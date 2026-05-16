from __future__ import annotations

from typing import Any

from core.logger import get_logger
from risk.config import RiskBudgetConfig, StrategyRiskConfig, SymbolRiskConfig
from risk.enums import (
    RiskDecisionType,
    RiskLevel,
    RiskMode,
    RiskViolationType,
    StrategyRiskStatus,
    SymbolRiskStatus,
)
from risk.models import RiskCheckResult, RiskEvaluationRequest, RiskViolation
from risk.state import RiskState, StrategyRiskState, SymbolRiskState
from risk.utils import is_finite_number, safe_div


class RiskModeResolver:
    """
    Resolves global RiskMode from current RiskState and RiskBudgetConfig.

    Pure domain service: it does not mutate RiskState. RiskManager must apply
    the returned mode through state.set_risk_mode(...) under its orchestration
    lock, so all downstream guards observe one consistent mode.
    """

    def __init__(
        self,
        config: RiskBudgetConfig,
        *,
        service_name: str = "risk.mode_resolver",
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="risk_mode_resolver",
        )

    def resolve(
        self,
        state: RiskState,
        *,
        risk_unit: float,
    ) -> tuple[RiskMode, str | None]:
        if state.emergency_stop_active:
            return RiskMode.EMERGENCY_STOP, "Emergency stop is active"

        if state.trading_halted:
            return RiskMode.HALTED, state.halt_reason or "Trading is halted"

        if not is_finite_number(risk_unit) or risk_unit <= 0:
            return RiskMode.HALTED, "Invalid risk unit"

        daily_loss_r = self._loss_r(state.get_daily_pnl(), risk_unit)
        weekly_loss_r = self._loss_r(state.get_weekly_pnl(), risk_unit)
        monthly_loss_r = self._loss_r(state.get_monthly_pnl(), risk_unit)

        if monthly_loss_r >= self._config.emergency_stop_loss_r:
            return RiskMode.EMERGENCY_STOP, "Emergency stop monthly loss limit reached"

        if monthly_loss_r >= self._config.monthly_review_loss_r:
            return RiskMode.REDUCE_ONLY, "Monthly review loss limit reached"

        if weekly_loss_r >= self._config.weekly_hard_loss_r:
            return RiskMode.HALTED, "Weekly hard loss limit reached"

        if daily_loss_r >= self._config.hard_daily_loss_r:
            return RiskMode.HALTED, "Hard daily loss limit reached"

        if daily_loss_r >= self._config.soft_daily_loss_r:
            if self._config.allow_new_positions_after_soft_daily_loss:
                return RiskMode.SAFE_MODE, "Soft daily loss limit reached"
            return RiskMode.REDUCE_ONLY, "Soft daily loss limit reached"

        if daily_loss_r >= self._config.caution_daily_loss_r:
            return RiskMode.CAUTION, "Caution daily loss threshold reached"

        return RiskMode.NORMAL, None

    @staticmethod
    def _loss_r(pnl: float, risk_unit: float) -> float:
        return safe_div(abs(min(0.0, pnl)), risk_unit)


class RiskBudgetGuard:
    """
    Checks global risk budget.

    This guard resolves the mode and returns a decision, but deliberately does
    not mutate RiskState. RiskManager should call state.set_risk_mode(...) once
    per evaluation after this result is accepted into the pipeline.
    """

    def __init__(
        self,
        config: RiskBudgetConfig,
        *,
        mode_resolver: RiskModeResolver | None = None,
        service_name: str = "risk.budget_guard",
    ) -> None:
        self._config = config
        self._mode_resolver = mode_resolver or RiskModeResolver(
            config,
            service_name=f"{service_name}.mode_resolver",
        )
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="risk_budget_guard",
        )

    def check(
        self,
        request: RiskEvaluationRequest,
        state: RiskState,
        *,
        risk_unit: float,
    ) -> RiskCheckResult:
        mode, reason = self._mode_resolver.resolve(state, risk_unit=risk_unit)
        budget_metadata = self.build_snapshot_metadata(state, risk_unit=risk_unit)

        if request.order_intent.reduces_risk:
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.ALLOW,
                risk_mode=mode,
                reason=reason,
                metadata={
                    "mode": mode.value,
                    "reason": reason,
                    "reduce_order_allowed": True,
                    **budget_metadata,
                },
            )

        if mode is RiskMode.EMERGENCY_STOP:
            return self._deny(
                request,
                mode=mode,
                reason=reason or "Emergency stop is active",
                violation_type=RiskViolationType.EMERGENCY_STOP_TRIGGERED,
                decision=RiskDecisionType.EMERGENCY_STOP,
                metadata=budget_metadata,
            )

        if mode is RiskMode.HALTED:
            return self._deny(
                request,
                mode=mode,
                reason=reason or "Trading is halted",
                violation_type=RiskViolationType.TRADING_HALTED,
                decision=RiskDecisionType.HALT_TRADING,
                metadata=budget_metadata,
            )

        if mode is RiskMode.REDUCE_ONLY:
            return self._deny(
                request,
                mode=mode,
                reason=reason or "Reduce-only mode is active",
                violation_type=RiskViolationType.REDUCE_ONLY_ACTIVE,
                decision=RiskDecisionType.ONLY_REDUCE,
                metadata=budget_metadata,
            )

        violations: list[RiskViolation] = []

        if mode is RiskMode.SAFE_MODE:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.SAFE_MODE_ACTIVE,
                    level=RiskLevel.WARNING,
                    message=reason or "Safe mode is active",
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"risk_mode": mode.value},
                )
            )

        if mode is RiskMode.CAUTION:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.SOFT_DAILY_LOSS_EXCEEDED,
                    level=RiskLevel.WARNING,
                    message=reason or "Caution risk mode is active",
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"risk_mode": mode.value},
                )
            )

        decision = (
            RiskDecisionType.REDUCE_RISK
            if mode in {RiskMode.CAUTION, RiskMode.SAFE_MODE}
            else RiskDecisionType.ALLOW
        )

        return RiskCheckResult(
            passed=True,
            decision=decision,
            violations=violations,
            risk_mode=mode,
            reason=reason,
            metadata={
                "mode": mode.value,
                "daily_pnl": state.get_daily_pnl(),
                "weekly_pnl": state.get_weekly_pnl(),
                "monthly_pnl": state.get_monthly_pnl(),
                **budget_metadata,
            },
        )

    def build_snapshot_metadata(
        self,
        state: RiskState,
        *,
        risk_unit: float,
    ) -> dict[str, object]:
        snapshot = state.get_budget_snapshot(
            risk_unit=risk_unit,
            caution_daily_loss_r=self._config.caution_daily_loss_r,
            soft_daily_loss_r=self._config.soft_daily_loss_r,
            hard_daily_loss_r=self._config.hard_daily_loss_r,
            weekly_hard_loss_r=self._config.weekly_hard_loss_r,
            monthly_review_loss_r=self._config.monthly_review_loss_r,
            emergency_stop_loss_r=self._config.emergency_stop_loss_r,
        )
        return {"budget_snapshot": snapshot}

    def _deny(
        self,
        request: RiskEvaluationRequest,
        *,
        mode: RiskMode,
        reason: str,
        violation_type: RiskViolationType,
        decision: RiskDecisionType,
        metadata: dict[str, object] | None = None,
    ) -> RiskCheckResult:
        self._logger.warning(
            "Global budget check denied request | symbol=%s mode=%s reason=%s",
            request.symbol,
            mode.value,
            reason,
            extra={"symbol": request.symbol, "risk_mode": mode.value},
        )

        return RiskCheckResult(
            passed=False,
            decision=decision,
            violations=[
                RiskViolation(
                    violation_type=violation_type,
                    level=RiskLevel.CRITICAL,
                    message=reason,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"risk_mode": mode.value},
                )
            ],
            risk_mode=mode,
            reason=reason,
            metadata=dict(metadata or {}),
        )


class SymbolRiskGuard:
    """
    Checks symbol-level budget and throttling.

    Pending risk reservations are treated as already committed risk for limits.
    This prevents multiple approved-but-not-yet-opened orders from bypassing
    symbol position/open-risk/trade budgets.
    """

    def __init__(
        self,
        config: SymbolRiskConfig,
        *,
        service_name: str = "risk.symbol_guard",
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="symbol_risk_guard",
        )

    def check(
        self,
        request: RiskEvaluationRequest,
        state: RiskState,
        *,
        risk_unit: float,
        candidate_open_risk: float = 0.0,
    ) -> RiskCheckResult:
        validation_error = self._validate_inputs(
            risk_unit=risk_unit,
            candidate_open_risk=candidate_open_risk,
        )
        if validation_error is not None:
            return validation_error

        symbol_state = state.get_symbol_state(request.symbol)
        symbol_state.refresh_status()

        pending_open_risk = _pending_open_risk(state, symbol=request.symbol)
        pending_trades = _pending_count(state, symbol=request.symbol)
        projected_open_risk = (
            symbol_state.open_risk
            + pending_open_risk
            + max(0.0, candidate_open_risk)
        )
        projected_open_risk_r = safe_div(projected_open_risk, risk_unit)

        metadata: dict[str, Any] = {
            "symbol_snapshot": symbol_state.snapshot(risk_unit=risk_unit),
            "actual_open_risk": symbol_state.open_risk,
            "pending_open_risk": pending_open_risk,
            "candidate_open_risk": max(0.0, candidate_open_risk),
            "projected_open_risk": projected_open_risk,
            "projected_open_risk_r": projected_open_risk_r,
            "pending_reservations_count": pending_trades,
        }

        if request.order_intent.reduces_risk:
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.ALLOW,
                metadata={
                    **metadata,
                    "symbol_status": symbol_state.status.value,
                    "reduce_order_allowed": True,
                },
            )

        violations: list[RiskViolation] = []

        if symbol_state.status is SymbolRiskStatus.DISABLED:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.SYMBOL_DISABLED,
                    level=RiskLevel.CRITICAL,
                    message=symbol_state.disabled_reason or "Symbol is disabled by risk policy",
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"disabled_until": symbol_state.disabled_until},
                )
            )

        if symbol_state.status is SymbolRiskStatus.COOLDOWN and symbol_state.cooldown.is_active():
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.SYMBOL_COOLDOWN_ACTIVE,
                    level=RiskLevel.WARNING,
                    message=symbol_state.cooldown.reason or "Symbol cooldown is active",
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"cooldown_until": symbol_state.cooldown.cooldown_until},
                )
            )

        actual_positions = self._count_symbol_positions(state, request.symbol)
        projected_positions = actual_positions + pending_trades + self._candidate_position_increment(request)
        metadata.update(
            {
                "actual_positions": actual_positions,
                "projected_positions": projected_positions,
            }
        )
        if projected_positions > self._config.max_positions_per_symbol:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.SYMBOL_POSITION_LIMIT_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Maximum positions per symbol would be exceeded",
                    current_value=float(projected_positions),
                    limit_value=float(self._config.max_positions_per_symbol),
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={
                        "actual_positions": actual_positions,
                        "pending_positions": pending_trades,
                    },
                )
            )

        trade_limit = self._resolve_trade_limit(request.symbol)
        projected_trades_today = symbol_state.trades_today + pending_trades + self._candidate_trade_increment(request)
        metadata["projected_trades_today"] = projected_trades_today
        if trade_limit is not None and projected_trades_today > trade_limit:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.SYMBOL_TRADE_LIMIT_EXCEEDED,
                    level=RiskLevel.WARNING,
                    message="Maximum trades per symbol/day would be exceeded",
                    current_value=float(projected_trades_today),
                    limit_value=float(trade_limit),
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"pending_trades": pending_trades},
                )
            )

        symbol_daily_loss_limit_r = self._resolve_daily_loss_limit(request.symbol)
        symbol_daily_loss_r = safe_div(abs(min(0.0, symbol_state.daily_pnl)), risk_unit)
        metadata["symbol_daily_loss_r"] = symbol_daily_loss_r
        if symbol_daily_loss_r >= symbol_daily_loss_limit_r:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.SYMBOL_DAILY_LOSS_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Symbol daily loss budget exceeded",
                    current_value=symbol_daily_loss_r,
                    limit_value=symbol_daily_loss_limit_r,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                )
            )

        symbol_open_risk_limit_r = self._resolve_open_risk_limit(request.symbol)
        metadata["symbol_open_risk_limit_r"] = symbol_open_risk_limit_r
        if projected_open_risk_r > symbol_open_risk_limit_r:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.SYMBOL_OPEN_RISK_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Symbol open risk budget would be exceeded",
                    current_value=projected_open_risk_r,
                    limit_value=symbol_open_risk_limit_r,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"pending_open_risk": pending_open_risk},
                )
            )

        if violations:
            self._logger.warning(
                "Symbol risk check failed | symbol=%s violations=%s pending=%s projected_open_risk_r=%s",
                request.symbol,
                len(violations),
                pending_trades,
                projected_open_risk_r,
                extra={"symbol": request.symbol},
            )
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=violations,
                reason="Symbol risk check failed",
                metadata=metadata,
            )

        decision = (
            RiskDecisionType.REDUCE_RISK
            if symbol_state.status is SymbolRiskStatus.REDUCED
            else RiskDecisionType.ALLOW
        )

        return RiskCheckResult(
            passed=True,
            decision=decision,
            metadata=metadata,
        )

    def should_apply_loss_cooldown(self, symbol_state: SymbolRiskState) -> bool:
        return (
            self._config.cooldown_after_consecutive_losses > 0
            and symbol_state.consecutive_losses >= self._config.cooldown_after_consecutive_losses
        )

    def _resolve_daily_loss_limit(self, symbol: str) -> float:
        return self._config.per_symbol_daily_loss_r.get(
            symbol,
            self._config.max_symbol_daily_loss_r,
        )

    def _resolve_open_risk_limit(self, symbol: str) -> float:
        return self._config.per_symbol_open_risk_r.get(
            symbol,
            self._config.max_symbol_open_risk_r,
        )

    def _resolve_trade_limit(self, symbol: str) -> int | None:
        return self._config.per_symbol_trade_limit.get(
            symbol,
            self._config.max_trades_per_symbol_per_day,
        )

    @staticmethod
    def _count_symbol_positions(state: RiskState, symbol: str) -> int:
        return sum(1 for position in state.positions.values() if position.symbol == symbol)

    @staticmethod
    def _candidate_position_increment(request: RiskEvaluationRequest) -> int:
        # OPEN creates a new position slot. INCREASE normally changes an existing
        # position, while reduce/close are returned early.
        return 1 if request.order_intent.value == "open" else 0

    @staticmethod
    def _candidate_trade_increment(request: RiskEvaluationRequest) -> int:
        return 1 if getattr(request.order_intent, "increases_risk", True) else 0

    @staticmethod
    def _validate_inputs(
        *,
        risk_unit: float,
        candidate_open_risk: float,
    ) -> RiskCheckResult | None:
        if not is_finite_number(risk_unit) or risk_unit <= 0:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.INVALID_REQUEST,
                        level=RiskLevel.CRITICAL,
                        message="risk_unit must be a finite positive number",
                    )
                ],
                reason="Invalid risk unit",
            )

        if not is_finite_number(candidate_open_risk) or candidate_open_risk < 0:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.INVALID_REQUEST,
                        level=RiskLevel.CRITICAL,
                        message="candidate_open_risk must be a finite non-negative number",
                    )
                ],
                reason="Invalid candidate open risk",
            )

        return None


class StrategyRiskGuard:
    """
    Checks strategy-level budget and expectancy.

    Pending reservations are counted as committed strategy risk/trades. The
    guard does not disable/reduce strategies directly; it returns suggested
    actions for RiskManager to apply explicitly.
    """

    def __init__(
        self,
        config: StrategyRiskConfig,
        *,
        service_name: str = "risk.strategy_guard",
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="strategy_risk_guard",
        )

    def check(
        self,
        request: RiskEvaluationRequest,
        state: RiskState,
        *,
        risk_unit: float,
        candidate_open_risk: float = 0.0,
    ) -> RiskCheckResult:
        validation_error = SymbolRiskGuard._validate_inputs(
            risk_unit=risk_unit,
            candidate_open_risk=candidate_open_risk,
        )
        if validation_error is not None:
            return validation_error

        if not request.strategy_name:
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.ALLOW,
                metadata={"strategy_name": None},
            )

        strategy_state = state.get_strategy_state(request.strategy_name)
        strategy_state.refresh_status()

        pending_open_risk = _pending_open_risk(state, strategy_name=request.strategy_name)
        pending_trades = _pending_count(state, strategy_name=request.strategy_name)
        projected_open_risk = (
            strategy_state.open_risk
            + pending_open_risk
            + max(0.0, candidate_open_risk)
        )
        projected_open_risk_r = safe_div(projected_open_risk, risk_unit)

        metadata: dict[str, Any] = {
            "strategy_snapshot": strategy_state.snapshot(risk_unit=risk_unit),
            "actual_open_risk": strategy_state.open_risk,
            "pending_open_risk": pending_open_risk,
            "candidate_open_risk": max(0.0, candidate_open_risk),
            "projected_open_risk": projected_open_risk,
            "projected_open_risk_r": projected_open_risk_r,
            "pending_reservations_count": pending_trades,
        }

        if request.order_intent.reduces_risk:
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.ALLOW,
                metadata={
                    **metadata,
                    "strategy_status": strategy_state.status.value,
                    "reduce_order_allowed": True,
                },
            )

        violations: list[RiskViolation] = []
        suggested_multiplier = strategy_state.risk_multiplier
        suggested_action: str | None = None

        if strategy_state.status is StrategyRiskStatus.DISABLED:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.STRATEGY_DISABLED,
                    level=RiskLevel.CRITICAL,
                    message=strategy_state.disabled_reason or "Strategy is disabled by risk policy",
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"disabled_until": strategy_state.disabled_until},
                )
            )

        if strategy_state.status is StrategyRiskStatus.COOLDOWN and strategy_state.cooldown.is_active():
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.STRATEGY_COOLDOWN_ACTIVE,
                    level=RiskLevel.WARNING,
                    message=strategy_state.cooldown.reason or "Strategy cooldown is active",
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"cooldown_until": strategy_state.cooldown.cooldown_until},
                )
            )

        trade_limit = self._resolve_trade_limit(request.strategy_name)
        projected_trades_today = (
            strategy_state.trades_today
            + pending_trades
            + SymbolRiskGuard._candidate_trade_increment(request)
        )
        metadata["projected_trades_today"] = projected_trades_today
        if trade_limit is not None and projected_trades_today > trade_limit:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.STRATEGY_BUDGET_EXCEEDED,
                    level=RiskLevel.WARNING,
                    message="Maximum trades per strategy/day would be exceeded",
                    current_value=float(projected_trades_today),
                    limit_value=float(trade_limit),
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"pending_trades": pending_trades},
                )
            )

        daily_loss_budget_r = self._resolve_daily_loss_budget(request.strategy_name)
        strategy_daily_loss_r = safe_div(abs(min(0.0, strategy_state.daily_pnl)), risk_unit)
        metadata["strategy_daily_loss_r"] = strategy_daily_loss_r
        if strategy_daily_loss_r >= daily_loss_budget_r:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.STRATEGY_DAILY_LOSS_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Strategy daily loss budget exceeded",
                    current_value=strategy_daily_loss_r,
                    limit_value=daily_loss_budget_r,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                )
            )

        open_risk_budget_r = self._resolve_open_risk_budget(request.strategy_name)
        metadata["strategy_open_risk_budget_r"] = open_risk_budget_r
        if projected_open_risk_r > open_risk_budget_r:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.STRATEGY_OPEN_RISK_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Strategy open risk budget would be exceeded",
                    current_value=projected_open_risk_r,
                    limit_value=open_risk_budget_r,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"pending_open_risk": pending_open_risk},
                )
            )

        if (
            self._config.max_consecutive_losses > 0
            and strategy_state.consecutive_losses >= self._config.max_consecutive_losses
        ):
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.STRATEGY_COOLDOWN_ACTIVE,
                    level=RiskLevel.WARNING,
                    message="Strategy consecutive loss threshold reached",
                    current_value=float(strategy_state.consecutive_losses),
                    limit_value=float(self._config.max_consecutive_losses),
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                )
            )
            suggested_action = "cooldown"

        rolling_trades = len(strategy_state.rolling_pnls)
        metadata["rolling_trades"] = rolling_trades
        expectancy = strategy_state.rolling_expectancy
        has_enough_expectancy_data = rolling_trades >= max(1, self._config.rolling_expectancy_window)
        metadata["has_enough_expectancy_data"] = has_enough_expectancy_data

        if expectancy is not None and has_enough_expectancy_data:
            if (
                self._config.disable_on_negative_expectancy
                and expectancy <= self._config.disable_when_expectancy_below
            ):
                violations.append(
                    RiskViolation(
                        violation_type=RiskViolationType.STRATEGY_EXPECTANCY_NEGATIVE,
                        level=RiskLevel.CRITICAL,
                        message="Strategy rolling expectancy is below disable threshold",
                        current_value=expectancy,
                        limit_value=self._config.disable_when_expectancy_below,
                        symbol=request.symbol,
                        strategy_name=request.strategy_name,
                        tier=request.tier,
                    )
                )
                suggested_action = "disable"

            elif expectancy < self._config.reduce_when_expectancy_below:
                suggested_multiplier = min(
                    suggested_multiplier,
                    self._config.reduced_risk_multiplier,
                )
                suggested_action = "reduce"

        metadata.update(
            {
                "suggested_action": suggested_action,
                "suggested_multiplier": suggested_multiplier,
            }
        )

        if violations:
            deny_level_violations = [
                violation
                for violation in violations
                if violation.level is RiskLevel.CRITICAL
            ]
            decision = RiskDecisionType.DENY if deny_level_violations else RiskDecisionType.REDUCE_RISK
            passed = not deny_level_violations

            self._logger.warning(
                "Strategy risk check %s | strategy=%s violations=%s pending=%s projected_open_risk_r=%s",
                "failed" if not passed else "reduced",
                request.strategy_name,
                len(violations),
                pending_trades,
                projected_open_risk_r,
                extra={"strategy": request.strategy_name},
            )

            return RiskCheckResult(
                passed=passed,
                decision=decision,
                violations=violations,
                reason="Strategy risk check failed" if not passed else "Strategy risk reduced",
                metadata=metadata,
            )

        if strategy_state.status is StrategyRiskStatus.REDUCED or suggested_action == "reduce":
            metadata["suggested_action"] = suggested_action or "reduce"
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.REDUCE_RISK,
                metadata=metadata,
            )

        return RiskCheckResult(
            passed=True,
            decision=RiskDecisionType.ALLOW,
            metadata=metadata,
        )

    def should_apply_loss_cooldown(self, strategy_state: StrategyRiskState) -> bool:
        return (
            self._config.max_consecutive_losses > 0
            and strategy_state.consecutive_losses >= self._config.max_consecutive_losses
        )

    def _resolve_daily_loss_budget(self, strategy_name: str) -> float:
        return self._config.per_strategy_daily_loss_budget_r.get(
            strategy_name,
            self._config.default_daily_loss_budget_r,
        )

    def _resolve_open_risk_budget(self, strategy_name: str) -> float:
        return self._config.per_strategy_open_risk_budget_r.get(
            strategy_name,
            self._config.default_open_risk_budget_r,
        )

    def _resolve_trade_limit(self, strategy_name: str) -> int | None:
        return self._config.per_strategy_trade_limit.get(strategy_name)


def _pending_open_risk(
    state: RiskState,
    *,
    symbol: str | None = None,
    strategy_name: str | None = None,
) -> float:
    getter = getattr(state, "get_pending_open_risk", None)
    if callable(getter):
        kwargs: dict[str, Any] = {}
        if symbol is not None:
            kwargs["symbol"] = symbol
        if strategy_name is not None:
            kwargs["strategy_name"] = strategy_name
        return float(getter(**kwargs))

    return sum(
        float(getattr(reservation, "open_risk", 0.0) or 0.0)
        for reservation in _iter_pending_reservations(
            state,
            symbol=symbol,
            strategy_name=strategy_name,
        )
    )


def _pending_count(
    state: RiskState,
    *,
    symbol: str | None = None,
    strategy_name: str | None = None,
) -> int:
    return sum(
        1
        for _ in _iter_pending_reservations(
            state,
            symbol=symbol,
            strategy_name=strategy_name,
        )
    )


def _iter_pending_reservations(
    state: RiskState,
    *,
    symbol: str | None = None,
    strategy_name: str | None = None,
):
    iterator = getattr(state, "_iter_pending_reservations", None)
    if callable(iterator):
        yield from iterator(symbol=symbol, strategy_name=strategy_name)
        return

    reservations = getattr(state, "pending_reservations", {})
    for reservation in reservations.values():
        if symbol is not None and getattr(reservation, "symbol", None) != symbol:
            continue
        if strategy_name is not None and getattr(reservation, "strategy_name", None) != strategy_name:
            continue
        is_expired = getattr(reservation, "is_expired", None)
        if callable(is_expired) and is_expired():
            continue
        yield reservation


__all__ = [
    "RiskBudgetGuard",
    "RiskModeResolver",
    "StrategyRiskGuard",
    "SymbolRiskGuard",
]
