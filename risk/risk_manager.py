from __future__ import annotations

import time
from typing import Any, Optional

from core.logger import get_logger

from risk.circuit_breaker import CircuitBreaker
from risk.config import RiskConfig
from risk.correlation_guard import CorrelationGuard
from risk.daily_loss_guard import DailyLossGuard
from risk.enums import CircuitBreakerReason, PositionSide, RiskDecisionType, TradingMode
from risk.exposure_control import ExposureControl
from risk.leverage_guard import LeverageGuard
from risk.max_drawdown_guard import MaxDrawdownGuard
from risk.metrics import RiskMetrics
from risk.models import (
    PortfolioPosition,
    PositionSizeRequest,
    RiskCheckResult,
    RiskDecision,
    RiskEvaluationRequest,
)
from risk.position_sizing import PositionSizer
from risk.state import RiskState


class RiskManager:
    """
    Production-oriented risk service.

    Responsibilities:
    - evaluate incoming trade requests
    - maintain local risk state
    - react to account / position / execution events
    - publish risk decisions via EventBus
    - expose lifecycle start/stop
    - register EventBus subscriptions automatically
    """

    def __init__(
        self,
        config: RiskConfig,
        *,
        event_bus: Optional[Any] = None,
        state: Optional[RiskState] = None,
        position_sizer: Optional[PositionSizer] = None,
        exposure_control: Optional[ExposureControl] = None,
        max_drawdown_guard: Optional[MaxDrawdownGuard] = None,
        daily_loss_guard: Optional[DailyLossGuard] = None,
        leverage_guard: Optional[LeverageGuard] = None,
        correlation_guard: Optional[CorrelationGuard] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        metrics: Optional[RiskMetrics] = None,
        auto_subscribe: bool = True,
        service_name: str = "risk_manager",
    ) -> None:
        self._config = config
        self._config.validate()

        self._event_bus = event_bus
        self._state = state or RiskState()
        self._metrics = metrics or RiskMetrics()
        self._service_name = service_name
        self._auto_subscribe = auto_subscribe

        self._position_sizer = position_sizer or PositionSizer(
            config.position_sizing,
            service_name="risk.position_sizer",
        )
        self._exposure_control = exposure_control or ExposureControl(
            config.exposure,
            service_name="risk.exposure_control",
        )
        self._max_drawdown_guard = max_drawdown_guard or MaxDrawdownGuard(
            config.drawdown,
            service_name="risk.max_drawdown_guard",
        )
        self._daily_loss_guard = daily_loss_guard or DailyLossGuard(
            config.daily_loss,
            service_name="risk.daily_loss_guard",
        )
        self._leverage_guard = leverage_guard or LeverageGuard(
            config.leverage,
            service_name="risk.leverage_guard",
        )
        self._correlation_guard = correlation_guard or CorrelationGuard(
            config.correlation,
            service_name="risk.correlation_guard",
        )
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            config.circuit_breaker,
            service_name="risk.circuit_breaker",
        )

        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="risk_manager",
        )

        self._subscriptions: list[Any] = []
        self._running = False
        self._started_at: float | None = None

    @property
    def state(self) -> RiskState:
        return self._state

    @property
    def metrics(self) -> RiskMetrics:
        return self._metrics

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            self._logger.warning("RiskManager already started")
            return

        self._running = True
        self._started_at = time.time()

        if self._event_bus is not None and self._auto_subscribe:
            self.register_subscriptions()

        await self._emit_event(
            "risk.manager.started",
            {
                "service": self._service_name,
                "started_at": self._started_at,
                "auto_subscribe": self._auto_subscribe,
            },
        )

        self._logger.info(
            "RiskManager started | auto_subscribe=%s subscriptions=%s",
            self._auto_subscribe,
            len(self._subscriptions),
        )

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("RiskManager already stopped")
            return

        self.unregister_subscriptions()

        await self._emit_event(
            "risk.manager.stopped",
            {
                "service": self._service_name,
                "stopped_at": time.time(),
            },
        )

        self._running = False
        self._logger.info("RiskManager stopped")

    def register_subscriptions(self) -> None:
        if self._event_bus is None:
            self._logger.warning("Cannot register subscriptions: event_bus is not configured")
            return

        if self._subscriptions:
            self._logger.warning("RiskManager subscriptions already registered")
            return

        self._subscriptions.extend(
            [
                self._event_bus.subscribe("signal.generated", self._handle_signal_generated, name="risk_on_signal"),
                self._event_bus.subscribe("account.*", self._handle_account_event, name="risk_on_account"),
                self._event_bus.subscribe("position.opened", self._handle_position_opened, name="risk_on_position_opened"),
                self._event_bus.subscribe("position.updated", self._handle_position_updated, name="risk_on_position_updated"),
                self._event_bus.subscribe("position.closed", self._handle_position_closed, name="risk_on_position_closed"),
                self._event_bus.subscribe("execution.order_rejected", self._handle_execution_rejected, name="risk_on_execution_rejected"),
                self._event_bus.subscribe("execution.order_failed", self._handle_execution_failed, name="risk_on_execution_failed"),
                self._event_bus.subscribe("execution.order_filled", self._handle_execution_filled, name="risk_on_execution_filled"),
                self._event_bus.subscribe("system.clock.day_rollover", self._handle_day_rollover, name="risk_on_day_rollover"),
                self._event_bus.subscribe("system.scheduler.job_failed", self._handle_scheduler_job_failed, name="risk_on_scheduler_failure"),
            ]
        )

        self._logger.info(
            "RiskManager subscriptions registered | count=%s",
            len(self._subscriptions),
        )

    def unregister_subscriptions(self) -> None:
        if self._event_bus is None:
            self._subscriptions.clear()
            return

        for subscription in self._subscriptions:
            try:
                self._event_bus.unsubscribe(subscription)
            except Exception:
                self._logger.exception("Failed to unsubscribe risk subscription")

        count = len(self._subscriptions)
        self._subscriptions.clear()

        self._logger.info(
            "RiskManager subscriptions unregistered | count=%s",
            count,
        )

    async def evaluate_request(self, request: RiskEvaluationRequest) -> RiskDecision:
        self._circuit_breaker.release_if_ready(self._state)

        checks: dict[str, RiskCheckResult] = {}

        await self._emit_event(
            "risk.request_received",
            {
                "symbol": request.symbol,
                "side": request.side.value,
                "signal_id": request.signal_id,
                "strategy_name": request.strategy_name,
                "requested_size": request.requested_size,
                "requested_leverage": request.requested_leverage,
            },
        )

        cb_result = self._circuit_breaker.check(self._state)
        checks["circuit_breaker"] = cb_result
        if cb_result.decision is RiskDecisionType.HALT_TRADING:
            decision = self._build_terminal_decision(
                check=cb_result,
                checks=checks,
                reason="Circuit breaker is active",
                final_size=None,
                final_leverage=request.requested_leverage,
            )
            await self._finalize_decision(request, decision)
            return decision

        drawdown_result = self._max_drawdown_guard.check(self._state)
        checks["max_drawdown"] = drawdown_result
        if drawdown_result.decision is RiskDecisionType.HALT_TRADING:
            self._state.halt_trading("Maximum drawdown exceeded")
            decision = self._build_terminal_decision(
                check=drawdown_result,
                checks=checks,
                reason="Maximum drawdown exceeded",
                final_size=None,
                final_leverage=request.requested_leverage,
            )
            await self._finalize_decision(request, decision)
            return decision

        daily_loss_result = self._daily_loss_guard.check(self._state)
        checks["daily_loss"] = daily_loss_result
        if daily_loss_result.decision is RiskDecisionType.HALT_TRADING:
            self._state.halt_trading("Daily loss limit exceeded")
            decision = self._build_terminal_decision(
                check=daily_loss_result,
                checks=checks,
                reason="Daily loss limit exceeded",
                final_size=None,
                final_leverage=request.requested_leverage,
            )
            await self._finalize_decision(request, decision)
            return decision

        if drawdown_result.decision is RiskDecisionType.REDUCE_SIZE:
            self._state.enable_safe_mode("Drawdown threshold reached")

        leverage_result = self._leverage_guard.check(
            request,
            self._state,
            candidate_leverage=request.requested_leverage,
        )
        checks["leverage"] = leverage_result
        if not leverage_result.passed:
            decision = self._build_terminal_decision(
                check=leverage_result,
                checks=checks,
                reason="Leverage validation failed",
                final_size=None,
                final_leverage=request.requested_leverage,
            )
            await self._finalize_decision(request, decision)
            return decision

        final_leverage = leverage_result.adjusted_leverage or request.requested_leverage or 1.0

        size_request = PositionSizeRequest(
            symbol=request.symbol,
            side=request.side,
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
            account_equity=self._state.equity,
            free_balance=self._state.free_balance,
            risk_percent=self._config.position_sizing.default_risk_per_trade_pct,
            confidence=request.confidence,
            leverage=final_leverage,
            metadata=dict(request.metadata),
        )

        size_result = self._position_sizer.check(size_request, self._state)
        checks["position_sizing"] = size_result
        if not size_result.passed:
            decision = self._build_terminal_decision(
                check=size_result,
                checks=checks,
                reason="Position sizing failed",
                final_size=None,
                final_leverage=final_leverage,
            )
            await self._finalize_decision(request, decision)
            return decision

        candidate_size = size_result.adjusted_size
        if candidate_size is None or candidate_size <= 0:
            decision = RiskDecision(
                allowed=False,
                decision=RiskDecisionType.DENY,
                final_size=None,
                final_leverage=final_leverage,
                reason="Calculated candidate_size is invalid",
                violations=self._collect_violations(checks),
                checks=checks,
                metadata={"symbol": request.symbol},
            )
            await self._finalize_decision(request, decision)
            return decision

        if request.requested_size is not None and request.requested_size > 0:
            candidate_size = min(candidate_size, request.requested_size)

        exposure_result = self._exposure_control.check(
            request,
            self._state,
            candidate_size=candidate_size,
        )
        checks["exposure"] = exposure_result
        if not exposure_result.passed:
            decision = self._build_terminal_decision(
                check=exposure_result,
                checks=checks,
                reason="Exposure limit exceeded",
                final_size=None,
                final_leverage=final_leverage,
            )
            await self._finalize_decision(request, decision)
            return decision

        correlation_result = self._correlation_guard.check(
            request,
            self._state,
            candidate_size=candidate_size,
        )
        checks["correlation"] = correlation_result
        if not correlation_result.passed:
            decision = self._build_terminal_decision(
                check=correlation_result,
                checks=checks,
                reason="Correlation/group exposure limit exceeded",
                final_size=None,
                final_leverage=final_leverage,
            )
            await self._finalize_decision(request, decision)
            return decision

        decision_type = self._resolve_success_decision(checks)

        decision = RiskDecision(
            allowed=True,
            decision=decision_type,
            final_size=candidate_size,
            final_leverage=final_leverage,
            reason=self._build_success_reason(decision_type),
            violations=self._collect_violations(checks),
            checks=checks,
            metadata={
                "symbol": request.symbol,
                "side": request.side.value,
                "signal_id": request.signal_id,
                "strategy_name": request.strategy_name,
            },
        )

        await self._finalize_decision(request, decision)
        self._circuit_breaker.register_success()
        return decision

    async def on_position_opened(self, position: PortfolioPosition) -> None:
        self._state.add_position(position)
        await self._emit_event(
            "risk.position_registered",
            {
                "symbol": position.symbol,
                "position_id": position.position_id,
                "size": position.size,
                "side": position.side.value,
                "notional_value": position.notional_value,
            },
        )

    async def on_position_updated(
        self,
        symbol: str,
        *,
        position_id: str | None = None,
        size: float | None = None,
        mark_price: float | None = None,
        notional_value: float | None = None,
        leverage: float | None = None,
        unrealized_pnl: float | None = None,
    ) -> None:
        self._state.update_position(
            symbol,
            position_id=position_id,
            size=size,
            mark_price=mark_price,
            notional_value=notional_value,
            leverage=leverage,
            unrealized_pnl=unrealized_pnl,
        )

    async def on_position_closed(
        self,
        symbol: str,
        *,
        realized_pnl: float = 0.0,
        position_id: str | None = None,
    ) -> None:
        self._state.remove_position(symbol, position_id=position_id)
        self._state.register_trade_outcome(realized_pnl)

        await self._emit_event(
            "risk.position_unregistered",
            {
                "symbol": symbol,
                "position_id": position_id,
                "realized_pnl": realized_pnl,
            },
        )

    async def on_account_update(
        self,
        *,
        balance: float | None = None,
        equity: float | None = None,
        free_balance: float | None = None,
        used_margin: float | None = None,
        realized_pnl: float | None = None,
        unrealized_pnl: float | None = None,
    ) -> None:
        self._state.update_account(
            balance=balance,
            equity=equity,
            free_balance=free_balance,
            used_margin=used_margin,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
        )

    async def on_execution_failure(
        self,
        *,
        message: str | None = None,
        reason: CircuitBreakerReason = CircuitBreakerReason.EXECUTION_FAILURES,
    ) -> None:
        was_active = self._state.is_circuit_breaker_active()

        self._circuit_breaker.register_failure(
            self._state,
            reason=reason,
            message=message,
            count_as_execution_failure=True,
        )

        is_active = self._state.is_circuit_breaker_active()
        if not was_active and is_active:
            self._metrics.register_circuit_breaker_trigger()
            await self._emit_event(
                "risk.circuit_breaker_triggered",
                {
                    "reason": self._state.circuit_breaker.reason.value
                    if self._state.circuit_breaker.reason
                    else None,
                    "message": self._state.circuit_breaker.message,
                    "cooldown_until": self._state.circuit_breaker.cooldown_until,
                },
            )

    async def reset_daily_state(self) -> None:
        self._state.reset_daily_state()
        await self._emit_event(
            "risk.daily_state_reset",
            {
                "equity": self._state.equity,
                "daily_start_equity": self._state.daily_start_equity,
            },
        )

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "started_at": self._started_at,
            "subscriptions": len(self._subscriptions),
            "state": self._state.snapshot(),
            "metrics": self._metrics.snapshot(self._state),
            "circuit_breaker": self._circuit_breaker.stats(),
        }

    async def _handle_signal_generated(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}

        symbol = payload.get("symbol")
        side_raw = payload.get("side")
        entry_price = payload.get("entry_price")
        stop_loss = payload.get("stop_loss")

        if not symbol or not side_raw or entry_price is None:
            self._logger.warning("Ignoring malformed signal.generated event")
            return

        try:
            side = PositionSide(side_raw.lower())
        except Exception:
            self._logger.warning("Ignoring signal with invalid side | side=%s", side_raw)
            return

        request = RiskEvaluationRequest(
            symbol=symbol,
            side=side,
            entry_price=float(entry_price),
            stop_loss=float(stop_loss) if stop_loss is not None else None,
            take_profit=float(payload["take_profit"]) if payload.get("take_profit") is not None else None,
            signal_id=payload.get("signal_id"),
            strategy_name=payload.get("strategy_name"),
            confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
            requested_size=float(payload["requested_size"]) if payload.get("requested_size") is not None else None,
            requested_leverage=float(payload["requested_leverage"]) if payload.get("requested_leverage") is not None else None,
            metadata=dict(payload.get("metadata", {})),
        )

        await self.evaluate_request(request)

    async def _handle_account_event(self, event: Any) -> None:
        topic = getattr(event, "topic", "")
        payload = getattr(event, "payload", {}) or {}

        if topic == "account.balance_updated":
            await self.on_account_update(
                balance=self._to_float_or_none(payload.get("balance")),
                free_balance=self._to_float_or_none(payload.get("free_balance")),
            )
            return

        if topic == "account.equity_updated":
            await self.on_account_update(
                equity=self._to_float_or_none(payload.get("equity")),
                unrealized_pnl=self._to_float_or_none(payload.get("unrealized_pnl")),
            )
            return

        if topic == "account.margin_updated":
            await self.on_account_update(
                used_margin=self._to_float_or_none(payload.get("used_margin")),
                free_balance=self._to_float_or_none(payload.get("free_balance")),
            )
            return

        if topic == "account.snapshot":
            await self.on_account_update(
                balance=self._to_float_or_none(payload.get("balance")),
                equity=self._to_float_or_none(payload.get("equity")),
                free_balance=self._to_float_or_none(payload.get("free_balance")),
                used_margin=self._to_float_or_none(payload.get("used_margin")),
                realized_pnl=self._to_float_or_none(payload.get("realized_pnl")),
                unrealized_pnl=self._to_float_or_none(payload.get("unrealized_pnl")),
            )

    async def _handle_position_opened(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}

        symbol = payload.get("symbol")
        side_raw = payload.get("side")
        size = self._to_float_or_none(payload.get("size"))
        entry_price = self._to_float_or_none(payload.get("entry_price"))
        mark_price = self._to_float_or_none(payload.get("mark_price")) or entry_price
        notional_value = self._to_float_or_none(payload.get("notional_value"))
        leverage = self._to_float_or_none(payload.get("leverage"))

        if not symbol or not side_raw or size is None or entry_price is None:
            self._logger.warning("Ignoring malformed position.opened event")
            return

        try:
            side = PositionSide(side_raw.lower())
        except Exception:
            self._logger.warning("Ignoring position.opened with invalid side | side=%s", side_raw)
            return

        if notional_value is None:
            notional_value = abs(size * entry_price)

        position = PortfolioPosition(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            mark_price=mark_price or entry_price,
            notional_value=notional_value,
            leverage=leverage,
            unrealized_pnl=self._to_float_or_none(payload.get("unrealized_pnl")) or 0.0,
            strategy_name=payload.get("strategy_name"),
            position_id=payload.get("position_id"),
            opened_at=self._to_float_or_none(payload.get("opened_at")) or time.time(),
            metadata=dict(payload.get("metadata", {})),
        )

        await self.on_position_opened(position)

    async def _handle_position_updated(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}

        symbol = payload.get("symbol")
        if not symbol:
            return

        await self.on_position_updated(
            symbol,
            position_id=payload.get("position_id"),
            size=self._to_float_or_none(payload.get("size")),
            mark_price=self._to_float_or_none(payload.get("mark_price")),
            notional_value=self._to_float_or_none(payload.get("notional_value")),
            leverage=self._to_float_or_none(payload.get("leverage")),
            unrealized_pnl=self._to_float_or_none(payload.get("unrealized_pnl")),
        )

    async def _handle_position_closed(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}

        symbol = payload.get("symbol")
        if not symbol:
            return

        await self.on_position_closed(
            symbol,
            realized_pnl=self._to_float_or_none(payload.get("realized_pnl")) or 0.0,
            position_id=payload.get("position_id"),
        )

    async def _handle_execution_rejected(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}
        await self.on_execution_failure(
            message=payload.get("reason") or payload.get("message") or "Execution order rejected",
            reason=CircuitBreakerReason.EXECUTION_FAILURES,
        )

    async def _handle_execution_failed(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}
        await self.on_execution_failure(
            message=payload.get("reason") or payload.get("message") or "Execution order failed",
            reason=CircuitBreakerReason.EXECUTION_FAILURES,
        )

    async def _handle_execution_filled(self, event: Any) -> None:
        self._circuit_breaker.register_success()

    async def _handle_day_rollover(self, event: Any) -> None:
        await self.reset_daily_state()

    async def _handle_scheduler_job_failed(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}
        job_name = payload.get("job_name", "unknown")
        message = f"Scheduler job failed: {job_name}"
        await self.on_execution_failure(
            message=message,
            reason=CircuitBreakerReason.SYSTEM_ERROR_RATE,
        )

    async def _finalize_decision(
        self,
        request: RiskEvaluationRequest,
        decision: RiskDecision,
    ) -> None:
        self._metrics.register_decision(decision)

        serialized = self._serialize_decision(request, decision)

        if decision.decision is RiskDecisionType.HALT_TRADING:
            self._state.halt_trading(decision.reason or "Risk halt triggered")
            await self._emit_event("risk.trading_halted", serialized)
        elif not decision.allowed:
            self._state.last_rejected_reason = decision.reason
            await self._emit_event("risk.rejected", serialized)
        elif decision.decision is RiskDecisionType.REDUCE_SIZE:
            await self._emit_event("risk.size_adjusted", serialized)
        else:
            await self._emit_event("risk.approved", serialized)

        self._logger.info(
            "Risk decision completed | symbol=%s decision=%s allowed=%s size=%s leverage=%s",
            request.symbol,
            decision.decision.value,
            decision.allowed,
            decision.final_size,
            decision.final_leverage,
            extra={
                "symbol": request.symbol,
                "strategies": request.strategy_name,
                "signal_id": request.signal_id,
            },
        )

    async def _emit_event(self, topic: str, payload: dict[str, Any]) -> None:
        if self._event_bus is None:
            return

        try:
            await self._event_bus.emit(
                topic,
                payload,
                source="risk_manager",
            )
        except Exception:
            self._logger.exception(
                "Failed to emit risk event | topic=%s",
                topic,
            )

    @staticmethod
    def _collect_violations(checks: dict[str, RiskCheckResult]) -> list[Any]:
        violations = []
        for result in checks.values():
            violations.extend(result.violations)
        return violations

    @staticmethod
    def _build_terminal_decision(
        *,
        check: RiskCheckResult,
        checks: dict[str, RiskCheckResult],
        reason: str,
        final_size: float | None,
        final_leverage: float | None,
    ) -> RiskDecision:
        return RiskDecision(
            allowed=False,
            decision=check.decision,
            final_size=final_size,
            final_leverage=final_leverage,
            reason=reason,
            violations=RiskManager._collect_violations(checks),
            checks=checks,
        )

    @staticmethod
    def _resolve_success_decision(checks: dict[str, RiskCheckResult]) -> RiskDecisionType:
        for result in checks.values():
            if result.decision is RiskDecisionType.REDUCE_SIZE:
                return RiskDecisionType.REDUCE_SIZE
        return RiskDecisionType.ALLOW

    @staticmethod
    def _build_success_reason(decision_type: RiskDecisionType) -> str:
        if decision_type is RiskDecisionType.REDUCE_SIZE:
            return "Request approved with reduced constraints"
        return "Request approved"

    @staticmethod
    def _serialize_decision(
        request: RiskEvaluationRequest,
        decision: RiskDecision,
    ) -> dict[str, Any]:
        return {
            "symbol": request.symbol,
            "side": request.side.value,
            "signal_id": request.signal_id,
            "strategy_name": request.strategy_name,
            "allowed": decision.allowed,
            "decision": decision.decision.value,
            "final_size": decision.final_size,
            "final_leverage": decision.final_leverage,
            "reason": decision.reason,
            "violations": [
                {
                    "type": violation.violation_type.value,
                    "level": violation.level.value,
                    "message": violation.message,
                    "current_value": violation.current_value,
                    "limit_value": violation.limit_value,
                    "metadata": dict(violation.metadata),
                }
                for violation in decision.violations
            ],
            "checks": {
                check_name: {
                    "passed": check_result.passed,
                    "decision": check_result.decision.value,
                    "adjusted_size": check_result.adjusted_size,
                    "adjusted_leverage": check_result.adjusted_leverage,
                    "violations": [
                        {
                            "type": violation.violation_type.value,
                            "level": violation.level.value,
                            "message": violation.message,
                            "current_value": violation.current_value,
                            "limit_value": violation.limit_value,
                            "metadata": dict(violation.metadata),
                        }
                        for violation in check_result.violations
                    ],
                    "metadata": dict(check_result.metadata),
                }
                for check_name, check_result in decision.checks.items()
            },
            "metadata": dict(decision.metadata),
        }

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None