from __future__ import annotations

import asyncio
import dataclasses
import math
import time
from enum import Enum
from typing import Any

from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler

from risk.budget import RiskBudgetGuard, StrategyRiskGuard, SymbolRiskGuard
from risk.circuit_breaker import CircuitBreaker
from risk.config import RiskConfig
from risk.enums import (
    CircuitBreakerReason,
    ExecutionQuality,
    LiquidityClass,
    MarginMode,
    OrderIntent,
    PositionSide,
    RiskDecisionType,
    RiskMode,
    TradeTier,
)
from risk.exposure_control import ExposureControl
from risk.guards import ExecutionCostGuard, LeverageGuard, RiskRewardGuard, TierRiskGuard
from risk.metrics import RiskMetrics
from risk.models import (
    ExecutionCostEstimate,
    ExpectedValueSnapshot,
    PortfolioPosition,
    PositionSizeRequest,
    RiskCheckResult,
    RiskDecision,
    RiskEvaluationRequest,
    TierRiskProfile,
)
from risk.position_sizing import PositionSizer, RiskUnitCalculator
from risk.state import RiskState


class RiskManager:
    """
    Production-grade risk orchestration service.

    Responsibilities:
    - receive signals from EventBus;
    - evaluate trade requests through adaptive tier-based risk pipeline;
    - maintain RiskState under async lock;
    - emit risk decisions;
    - subscribe/unsubscribe from EventBus;
    - use Scheduler for daily/weekly/monthly reset jobs when provided.

    This is the only risk package component that owns EventBus/Scheduler
    integration. Guards, state, metrics and sizing remain local domain layers.
    """

    def __init__(
            self,
            config: RiskConfig,
            *,
            event_bus: EventBus | None = None,
            scheduler: Scheduler | None = None,
            state: RiskState | None = None,
            metrics: RiskMetrics | None = None,
            risk_unit_calculator: RiskUnitCalculator | None = None,
            tier_guard: TierRiskGuard | None = None,
            risk_reward_guard: RiskRewardGuard | None = None,
            execution_cost_guard: ExecutionCostGuard | None = None,
            leverage_guard: LeverageGuard | None = None,
            budget_guard: RiskBudgetGuard | None = None,
            symbol_guard: SymbolRiskGuard | None = None,
            strategy_guard: StrategyRiskGuard | None = None,
            position_sizer: PositionSizer | None = None,
            exposure_control: ExposureControl | None = None,
            circuit_breaker: CircuitBreaker | None = None,
            auto_subscribe: bool = True,
            register_scheduler_jobs: bool = True,
            service_name: str = "risk_manager",
    ) -> None:
        self._config = config
        self._config.validate()

        self._event_bus = event_bus
        self._scheduler = scheduler
        self._state = state or RiskState()
        self._metrics = metrics or RiskMetrics()

        self._service_name = service_name
        self._auto_subscribe = auto_subscribe
        self._register_scheduler_jobs = register_scheduler_jobs

        self._risk_unit_calculator = risk_unit_calculator or RiskUnitCalculator(
            config.risk_unit,
            service_name="risk.risk_unit",
        )
        self._tier_guard = tier_guard or TierRiskGuard(
            config.tiers,
            service_name="risk.tier_guard",
        )
        self._risk_reward_guard = risk_reward_guard or RiskRewardGuard(
            service_name="risk.risk_reward_guard",
        )
        self._execution_cost_guard = execution_cost_guard or ExecutionCostGuard(
            config.execution_cost,
            service_name="risk.execution_cost_guard",
        )
        self._leverage_guard = leverage_guard or LeverageGuard(
            config.leverage,
            service_name="risk.leverage_guard",
        )
        self._budget_guard = budget_guard or RiskBudgetGuard(
            config.budget,
            service_name="risk.budget_guard",
        )
        self._symbol_guard = symbol_guard or SymbolRiskGuard(
            config.symbol,
            service_name="risk.symbol_guard",
        )
        self._strategy_guard = strategy_guard or StrategyRiskGuard(
            config.strategy,
            service_name="risk.strategy_guard",
        )
        self._position_sizer = position_sizer or PositionSizer(
            config.position_sizing,
            service_name="risk.position_sizer",
        )
        self._exposure_control = exposure_control or ExposureControl(
            config.exposure,
            service_name="risk.exposure_control",
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

        self._lock = asyncio.Lock()
        self._subscriptions: list[Any] = []
        self._scheduler_jobs: list[Any] = []
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
            self.register()

        if self._scheduler is not None and self._register_scheduler_jobs:
            self.register_scheduler_jobs()

        await self._emit_event(
            "risk.manager.started",
            {
                "service": self._service_name,
                "started_at": self._started_at,
                "auto_subscribe": self._auto_subscribe,
                "scheduler_jobs": len(self._scheduler_jobs),
            },
        )

        self._logger.info(
            "RiskManager started | auto_subscribe=%s subscriptions=%s scheduler_jobs=%s",
            self._auto_subscribe,
            len(self._subscriptions),
            len(self._scheduler_jobs),
        )

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("RiskManager already stopped")
            return

        self.unregister()

        await self._emit_event(
            "risk.manager.stopped",
            {
                "service": self._service_name,
                "stopped_at": time.time(),
            },
        )

        self._running = False
        self._logger.info("RiskManager stopped")

    def register(self) -> None:
        """
        Register EventBus subscriptions.

        This method follows the project convention: modules expose register()
        for EventBus subscriptions.
        """
        if self._event_bus is None:
            self._logger.warning("Cannot register RiskManager: event_bus is not configured")
            return

        if self._subscriptions:
            self._logger.warning("RiskManager subscriptions already registered")
            return

        self._subscriptions.extend(
            [
                self._event_bus.subscribe(
                    "signal.generated",
                    self._handle_signal_generated,
                    name="risk_on_signal_generated",
                ),
                self._event_bus.subscribe(
                    "account.*",
                    self._handle_account_event,
                    name="risk_on_account_event",
                ),
                self._event_bus.subscribe(
                    "position.opened",
                    self._handle_position_opened,
                    name="risk_on_position_opened",
                ),
                self._event_bus.subscribe(
                    "position.updated",
                    self._handle_position_updated,
                    name="risk_on_position_updated",
                ),
                self._event_bus.subscribe(
                    "position.closed",
                    self._handle_position_closed,
                    name="risk_on_position_closed",
                ),
                self._event_bus.subscribe(
                    "execution.order_rejected",
                    self._handle_execution_rejected,
                    name="risk_on_execution_rejected",
                ),
                self._event_bus.subscribe(
                    "execution.order_failed",
                    self._handle_execution_failed,
                    name="risk_on_execution_failed",
                ),
                self._event_bus.subscribe(
                    "execution.order_cancelled",
                    self._handle_execution_cancelled,
                    name="risk_on_execution_cancelled",
                ),
                self._event_bus.subscribe(
                    "execution.order_filled",
                    self._handle_execution_filled,
                    name="risk_on_execution_filled",
                ),
                self._event_bus.subscribe(
                    "system.clock.day_rollover",
                    self._handle_day_rollover,
                    name="risk_on_day_rollover",
                ),
                self._event_bus.subscribe(
                    "system.clock.week_rollover",
                    self._handle_week_rollover,
                    name="risk_on_week_rollover",
                ),
                self._event_bus.subscribe(
                    "system.clock.month_rollover",
                    self._handle_month_rollover,
                    name="risk_on_month_rollover",
                ),
                self._event_bus.subscribe(
                    "system.scheduler.job_failed",
                    self._handle_scheduler_job_failed,
                    name="risk_on_scheduler_failure",
                ),
                self._event_bus.subscribe(
                    "risk.manual_halt",
                    self._handle_manual_halt,
                    name="risk_on_manual_halt",
                ),
                self._event_bus.subscribe(
                    "risk.manual_resume",
                    self._handle_manual_resume,
                    name="risk_on_manual_resume",
                ),
            ]
        )

        self._logger.info(
            "RiskManager subscriptions registered | count=%s",
            len(self._subscriptions),
        )

    def unregister(self) -> None:
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

    def register_scheduler_jobs(self) -> None:
        """
        Register reset/cleanup jobs in core Scheduler if provided.

        Reset boundaries should preferably be driven by system.clock.* events
        emitted by the platform clock service. These interval jobs are defensive
        fallbacks only; RiskManager never starts unmanaged asyncio loops.
        """
        if self._scheduler is None:
            return

        jobs: list[tuple[str, Any, float]] = [
            ("risk.daily_reset", self.reset_daily_state, 24 * 60 * 60),
            ("risk.weekly_reset", self.reset_weekly_state, 7 * 24 * 60 * 60),
            ("risk.monthly_reset", self.reset_monthly_state, 30 * 24 * 60 * 60),
        ]

        reservation_cfg = getattr(self._config, "reservation", None)
        if reservation_cfg is not None and getattr(reservation_cfg, "enabled", False):
            jobs.append(
                (
                    "risk.reservation_cleanup",
                    self.cleanup_expired_reservations,
                    max(1.0, float(getattr(reservation_cfg, "cleanup_interval_seconds", 10.0))),
                )
            )

        for name, func, interval_seconds in jobs:
            try:
                job_id = self._scheduler.add_interval_job(
                    name=name,
                    func=func,
                    interval=interval_seconds,
                    run_immediately=False,
                )
                self._scheduler_jobs.append(job_id)
            except Exception:
                self._logger.exception("Failed to register risk scheduler job | name=%s", name)

    async def evaluate_request(self, request: RiskEvaluationRequest) -> RiskDecision:
        """
        Evaluate a pre-trade request and emit resulting events outside the state lock.

        The lock protects state reads/mutations and pending reservation creation.
        EventBus emits are intentionally performed after the lock is released to
        avoid re-entrancy/deadlock when downstream handlers call RiskManager again.
        """
        started_at = time.perf_counter()

        await self._emit_event(
            "risk.request_received",
            {
                "symbol": request.symbol,
                "side": request.side.value,
                "signal_id": request.signal_id,
                "strategy_name": request.strategy_name,
                "tier": request.tier.value if request.tier else None,
                "order_intent": request.order_intent.value,
                "requested_size": request.requested_size,
                "requested_leverage": request.requested_leverage,
            },
        )

        async with self._lock:
            decision, events = self._evaluate_request_locked(request, started_at=started_at)

        await self._emit_events(events)
        return decision

    def _evaluate_request_locked(
            self,
            request: RiskEvaluationRequest,
            *,
            started_at: float,
    ) -> tuple[RiskDecision, list[tuple[str, dict[str, Any], EventPriority | None]]]:
        events: list[tuple[str, dict[str, Any], EventPriority | None]] = []

        self._circuit_breaker.release_if_ready(self._state)
        _, expired_events = self._expire_reservations_locked(reason="evaluate_request")
        events.extend(expired_events)

        checks: dict[str, RiskCheckResult] = {}

        def finalize(
                decision: RiskDecision,
        ) -> tuple[RiskDecision, list[tuple[str, dict[str, Any], EventPriority | None]]]:
            events.extend(
                self._finalize_decision_locked(
                    request,
                    decision,
                    latency_ms=self._elapsed_ms(started_at),
                )
            )
            return decision, events

        request_validation = self._finite_request_validation(request)
        checks["request_validation"] = request_validation

        if not request_validation.passed:
            decision = self._build_terminal_decision(
                request=request,
                check=request_validation,
                checks=checks,
                reason=request_validation.reason or "Request validation failed",
                final_size=None,
                final_leverage=request.requested_leverage,
                final_tier=request.tier,
            )
            return finalize(decision)

        self._normalize_circuit_breaker_reason(self._state)

        base_risk_unit_snapshot = self._risk_unit_calculator.calculate(
            self._state,
            mode=self._state.risk_mode,
        )
        risk_unit = base_risk_unit_snapshot.effective_risk_unit

        cb_result = self._circuit_breaker.check(self._state)
        checks["circuit_breaker"] = cb_result

        if not cb_result.passed:
            decision = self._build_terminal_decision(
                request=request,
                check=cb_result,
                checks=checks,
                reason=cb_result.reason or "Circuit breaker is active",
                final_size=None,
                final_leverage=request.requested_leverage,
            )
            return finalize(decision)

        budget_result = self._budget_guard.check(
            request,
            self._state,
            risk_unit=max(risk_unit, 1e-12),
        )
        checks["budget"] = budget_result

        if budget_result.risk_mode is not None:
            self._state.set_risk_mode(
                budget_result.risk_mode,
                reason=budget_result.reason,
            )

        if not budget_result.passed:
            decision = self._build_terminal_decision(
                request=request,
                check=budget_result,
                checks=checks,
                reason=budget_result.reason or "Global risk budget check failed",
                final_size=None,
                final_leverage=request.requested_leverage,
            )
            return finalize(decision)

        tier_result = self._tier_guard.check(
            request,
            self._state,
            mode=self._state.risk_mode,
        )
        checks["tier"] = tier_result

        if not tier_result.passed:
            decision = self._build_terminal_decision(
                request=request,
                check=tier_result,
                checks=checks,
                reason=tier_result.reason or "Tier validation failed",
                final_size=None,
                final_leverage=request.requested_leverage,
            )
            return finalize(decision)

        tier_profile = self._extract_tier_profile(tier_result)

        if tier_profile is None:
            tier_profile = self._tier_guard.resolve_profile(
                request,
                self._state,
                mode=self._state.risk_mode,
            )

        rr_result = self._risk_reward_guard.check(request, tier_profile)
        checks["risk_reward"] = rr_result

        ev_snapshot: ExpectedValueSnapshot | None = None

        try:
            ev_snapshot = self._extract_ev_snapshot(rr_result)
        except ValueError:
            ev_snapshot = None

        if not rr_result.passed:
            if ev_snapshot is not None:
                execution_cost_result = self._execution_cost_guard.check(
                    request,
                    tier_profile,
                    ev_snapshot,
                    mode=self._state.risk_mode,
                )
                checks["execution_cost"] = execution_cost_result

            decision = self._build_terminal_decision(
                request=request,
                check=rr_result,
                checks=checks,
                reason=rr_result.reason or "Risk/reward validation failed",
                final_size=None,
                final_leverage=request.requested_leverage,
                final_tier=tier_profile.final_tier,
                ev_snapshot=ev_snapshot,
            )
            return finalize(decision)

        if ev_snapshot is None:
            decision = RiskDecision(
                allowed=False,
                decision=RiskDecisionType.DENY,
                final_size=None,
                final_leverage=request.requested_leverage,
                final_tier=tier_profile.final_tier,
                risk_mode=self._state.risk_mode,
                reason="Expected value snapshot missing from risk/reward check",
                symbol=request.symbol,
                side=request.side,
                signal_id=request.signal_id,
                strategy_name=request.strategy_name,
                order_intent=request.order_intent,
                violations=self._collect_violations(checks),
                checks=checks,
            )
            return finalize(decision)

        execution_cost_result = self._execution_cost_guard.check(
            request,
            tier_profile,
            ev_snapshot,
            mode=self._state.risk_mode,
        )
        checks["execution_cost"] = execution_cost_result

        if not execution_cost_result.passed:
            decision = self._build_terminal_decision(
                request=request,
                check=execution_cost_result,
                checks=checks,
                reason=execution_cost_result.reason or "Execution cost validation failed",
                final_size=None,
                final_leverage=request.requested_leverage,
                final_tier=tier_profile.final_tier,
                ev_snapshot=ev_snapshot,
            )
            return finalize(decision)

        leverage_result = self._leverage_guard.check(
            request,
            tier_profile,
            mode=self._state.risk_mode,
        )
        checks["leverage"] = leverage_result

        if not leverage_result.passed:
            decision = self._build_terminal_decision(
                request=request,
                check=leverage_result,
                checks=checks,
                reason=leverage_result.reason or "Leverage validation failed",
                final_size=None,
                final_leverage=request.requested_leverage,
                final_tier=tier_profile.final_tier,
                ev_snapshot=ev_snapshot,
            )
            return finalize(decision)

        final_leverage = (
                leverage_result.adjusted_leverage
                or request.requested_leverage
                or tier_profile.default_leverage
                or 1.0
        )

        if (
                request.strategy_name
                and self._order_increases_risk(request.order_intent)
                and self._is_strategy_disabled(request.strategy_name)
        ):
            strategy_result = self._strategy_guard.check(
                request,
                self._state,
                risk_unit=max(risk_unit, 1e-12),
                candidate_open_risk=0.0,
            )
            checks["strategy"] = strategy_result

            if not strategy_result.passed:
                decision = self._build_terminal_decision(
                    request=request,
                    check=strategy_result,
                    checks=checks,
                    reason=strategy_result.reason or "Strategy risk check failed",
                    final_size=None,
                    final_leverage=final_leverage,
                    final_tier=tier_profile.final_tier,
                    ev_snapshot=ev_snapshot,
                )
                return finalize(decision)

        strategy_multiplier = self._resolve_strategy_multiplier(request)
        symbol_multiplier = self._resolve_symbol_multiplier(request)

        risk_unit_snapshot = self._risk_unit_calculator.calculate(
            self._state,
            mode=self._state.risk_mode,
            strategy_multiplier=strategy_multiplier,
            symbol_multiplier=symbol_multiplier,
        )

        risk_amount = risk_unit_snapshot.effective_risk_unit * tier_profile.risk_units

        if risk_amount <= 0 and self._order_increases_risk(request.order_intent):
            decision = RiskDecision(
                allowed=False,
                decision=RiskDecisionType.DENY,
                final_size=None,
                final_leverage=final_leverage,
                final_tier=tier_profile.final_tier,
                final_risk_amount=risk_amount,
                risk_mode=self._state.risk_mode,
                reason="Effective risk amount is zero",
                symbol=request.symbol,
                side=request.side,
                signal_id=request.signal_id,
                strategy_name=request.strategy_name,
                order_intent=request.order_intent,
                violations=self._collect_violations(checks),
                checks=checks,
            )
            return finalize(decision)

        size_request = PositionSizeRequest(
            symbol=request.symbol,
            side=request.side,
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
            account_equity=self._state.equity,
            free_balance=self._state.free_balance,
            risk_amount=risk_amount,
            risk_unit_snapshot=risk_unit_snapshot,
            tier_profile=tier_profile,
            leverage=final_leverage,
            margin_mode=request.margin_mode,
            requested_size=request.requested_size,
            requested_margin=request.requested_margin,
            confidence=request.confidence,
            volatility=request.volatility,
            metadata=dict(request.metadata),
        )

        size_result = self._position_sizer.check(size_request, self._state)
        checks["position_sizing"] = size_result

        if not size_result.passed:
            decision = self._build_terminal_decision(
                request=request,
                check=size_result,
                checks=checks,
                reason=size_result.reason or "Position sizing failed",
                final_size=None,
                final_leverage=final_leverage,
                final_tier=tier_profile.final_tier,
                ev_snapshot=ev_snapshot,
            )
            return finalize(decision)

        candidate_size = size_result.adjusted_size
        candidate_margin = size_result.adjusted_margin
        candidate_open_risk = size_result.adjusted_risk_amount or risk_amount

        if candidate_size is None or candidate_size <= 0:
            decision = RiskDecision(
                allowed=False,
                decision=RiskDecisionType.DENY,
                final_size=None,
                final_leverage=final_leverage,
                final_tier=tier_profile.final_tier,
                final_risk_amount=candidate_open_risk,
                risk_mode=self._state.risk_mode,
                reason="Calculated candidate size is invalid",
                symbol=request.symbol,
                side=request.side,
                signal_id=request.signal_id,
                strategy_name=request.strategy_name,
                order_intent=request.order_intent,
                violations=self._collect_violations(checks),
                checks=checks,
            )
            return finalize(decision)

        strategy_result = self._strategy_guard.check(
            request,
            self._state,
            risk_unit=max(risk_unit_snapshot.effective_risk_unit, 1e-12),
            candidate_open_risk=candidate_open_risk,
        )
        checks["strategy"] = strategy_result

        if not strategy_result.passed:
            decision = self._build_terminal_decision(
                request=request,
                check=strategy_result,
                checks=checks,
                reason=strategy_result.reason or "Strategy risk check failed",
                final_size=None,
                final_leverage=final_leverage,
                final_tier=tier_profile.final_tier,
                ev_snapshot=ev_snapshot,
            )
            return finalize(decision)

        symbol_result = self._symbol_guard.check(
            request,
            self._state,
            risk_unit=max(risk_unit_snapshot.effective_risk_unit, 1e-12),
            candidate_open_risk=candidate_open_risk,
        )
        checks["symbol"] = symbol_result

        if not symbol_result.passed:
            decision = self._build_terminal_decision(
                request=request,
                check=symbol_result,
                checks=checks,
                reason=symbol_result.reason or "Symbol risk check failed",
                final_size=None,
                final_leverage=final_leverage,
                final_tier=tier_profile.final_tier,
                ev_snapshot=ev_snapshot,
            )
            return finalize(decision)

        exposure_result = self._exposure_control.check(
            request,
            self._state,
            candidate_size=candidate_size,
            candidate_open_risk=candidate_open_risk,
            candidate_leverage=final_leverage,
            candidate_margin=candidate_margin,
            risk_unit=max(risk_unit_snapshot.effective_risk_unit, 1e-12),
            mode=self._state.risk_mode,
        )
        checks["exposure"] = exposure_result

        if not exposure_result.passed:
            decision = self._build_terminal_decision(
                request=request,
                check=exposure_result,
                checks=checks,
                reason=exposure_result.reason or "Exposure limit exceeded",
                final_size=None,
                final_leverage=final_leverage,
                final_tier=tier_profile.final_tier,
                ev_snapshot=ev_snapshot,
            )
            return finalize(decision)

        decision_type = self._resolve_success_decision(checks)

        sizing_metadata: dict[str, Any] = dict(size_result.metadata or {})
        exposure_metadata: dict[str, Any] = dict(exposure_result.metadata or {})

        final_notional: float | None = None

        raw_sizing_notional: object = sizing_metadata.get("notional_value")

        final_notional: float | None = None

        raw_sizing_notional = sizing_metadata.get("notional_value")

        if isinstance(raw_sizing_notional, bool):
            raw_sizing_notional = None

        if isinstance(raw_sizing_notional, int | float | str | bytes | bytearray):
            try:
                parsed_notional = float(raw_sizing_notional)
            except (TypeError, ValueError):
                parsed_notional = math.nan

            if math.isfinite(parsed_notional):
                final_notional = parsed_notional

        if final_notional is None:
            raw_exposure_notional = exposure_metadata.get("candidate_notional")

            if isinstance(raw_exposure_notional, bool):
                raw_exposure_notional = None

            if isinstance(raw_exposure_notional, int | float | str | bytes | bytearray):
                try:
                    parsed_notional = float(raw_exposure_notional)
                except (TypeError, ValueError):
                    parsed_notional = math.nan

                if math.isfinite(parsed_notional):
                    final_notional = parsed_notional

        decision = RiskDecision(
            allowed=True,
            decision=decision_type,
            final_size=candidate_size,
            final_leverage=final_leverage,
            final_tier=tier_profile.final_tier,
            final_risk_amount=candidate_open_risk,
            final_margin=candidate_margin,
            final_notional=final_notional,
            risk_mode=self._state.risk_mode,
            risk_reward_ratio=ev_snapshot.risk_reward_ratio,
            expected_value=ev_snapshot.expected_value,
            expected_value_after_cost=ev_snapshot.expected_value_after_cost,
            expected_cost=ev_snapshot.expected_cost,
            cost_to_reward_ratio=ev_snapshot.cost_to_reward_ratio,
            reason=self._build_success_reason(decision_type),
            signal_id=request.signal_id,
            strategy_name=request.strategy_name,
            symbol=request.symbol,
            side=request.side,
            order_intent=request.order_intent,
            violations=self._collect_violations(checks),
            checks=checks,
            metadata={
                "risk_unit_snapshot": risk_unit_snapshot,
                "tier_profile": tier_profile,
                "expected_value_snapshot": ev_snapshot,
                "confidence": request.confidence,
                "edge_score": request.edge_score,
                "liquidity_class": request.liquidity_class.value,
                "execution_quality": request.execution_quality.value,
            },
        )

        self._circuit_breaker.register_success()
        return finalize(decision)

    async def on_position_opened(self, position: PortfolioPosition) -> None:
        events: list[tuple[str, dict[str, Any], EventPriority | None]] = []

        async with self._lock:
            reservation = self._state.confirm_risk_reservation(
                getattr(position, "metadata", {}).get("reservation_id") if position.metadata else None,
                signal_id=position.signal_id,
                symbol=position.symbol,
                position_id=position.position_id,
            )
            if reservation is not None:
                age_ms = self._reservation_age_ms(reservation)
                self._metrics.register_reservation_confirmed(
                    reservation_id=reservation.reservation_id,
                    symbol=reservation.symbol,
                    tier=reservation.tier,
                    strategy_name=reservation.strategy_name,
                    open_risk=reservation.open_risk,
                    margin=reservation.margin,
                    notional=reservation.notional,
                    age_ms=age_ms,
                )
                events.append(
                    (
                        "risk.reservation.confirmed",
                        self._serialize_reservation(reservation, status="confirmed", age_ms=age_ms),
                        EventPriority.NORMAL,
                    )
                )

            self._state.add_position(position)
            self._metrics.register_position_opened(
                symbol=position.symbol,
                tier=position.tier,
                strategy_name=position.strategy_name,
                open_risk=position.open_risk,
            )

        events.append(
            (
                "risk.position_registered",
                {
                    "symbol": position.symbol,
                    "position_id": position.position_id,
                    "size": position.size,
                    "side": position.side.value,
                    "notional_value": position.notional_value,
                    "margin_used": position.margin_used,
                    "risk_amount": position.risk_amount,
                    "tier": position.tier.value if position.tier else None,
                    "strategy_name": position.strategy_name,
                    "signal_id": position.signal_id,
                    "reservation_id": reservation.reservation_id if reservation is not None else None,
                },
                EventPriority.NORMAL,
            )
        )

        await self._emit_events(events)

    async def on_position_updated(
            self,
            symbol: str,
            *,
            position_id: str | None = None,
            size: float | None = None,
            mark_price: float | None = None,
            notional_value: float | None = None,
            leverage: float | None = None,
            margin_used: float | None = None,
            risk_amount: float | None = None,
            stop_loss: float | None = None,
            take_profit: float | None = None,
            unrealized_pnl: float | None = None,
    ) -> None:
        async with self._lock:
            self._state.update_position(
                symbol,
                position_id=position_id,
                size=size,
                mark_price=mark_price,
                notional_value=notional_value,
                leverage=leverage,
                margin_used=margin_used,
                risk_amount=risk_amount,
                stop_loss=stop_loss,
                take_profit=take_profit,
                unrealized_pnl=unrealized_pnl,
            )

    async def on_position_closed(
            self,
            symbol: str,
            *,
            realized_pnl: float = 0.0,
            position_id: str | None = None,
    ) -> None:
        async with self._lock:
            position = self._state.remove_position(
                symbol,
                position_id=position_id,
                realized_pnl=realized_pnl,
            )

            if position is not None:
                self._metrics.register_position_closed(
                    symbol=position.symbol,
                    realized_pnl=realized_pnl,
                    released_risk=position.open_risk,
                    tier=position.tier,
                    strategy_name=position.strategy_name,
                )

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
        async with self._lock:
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
        async with self._lock:
            was_active = self._state.is_circuit_breaker_active()

            triggered = self._circuit_breaker.register_failure(
                self._state,
                reason=reason,
                message=message,
            )

            is_active = self._state.is_circuit_breaker_active()

            if triggered or (not was_active and is_active):
                self._metrics.register_circuit_breaker_trigger()
                payload = {
                    "reason": (
                        self._state.circuit_breaker.reason.value
                        if self._state.circuit_breaker.reason
                        else None
                    ),
                    "message": self._state.circuit_breaker.message,
                    "cooldown_until": self._state.circuit_breaker.cooldown_until,
                    "manual_release_required": self._state.circuit_breaker.manual_release_required,
                }
            else:
                payload = None

        if payload is not None:
            await self._emit_event(
                "risk.circuit_breaker_triggered",
                payload,
                priority=EventPriority.CRITICAL,
            )

    async def reset_daily_state(self) -> None:
        async with self._lock:
            self._state.reset_daily_state()
            payload = {
                "equity": self._state.equity,
                "daily_start_equity": self._state.daily_start_equity,
            }

        await self._emit_event("risk.daily_state_reset", payload)

    async def reset_weekly_state(self) -> None:
        async with self._lock:
            self._state.reset_weekly_state()
            payload = {
                "equity": self._state.equity,
                "weekly_start_equity": self._state.weekly_start_equity,
            }

        await self._emit_event("risk.weekly_state_reset", payload)

    async def reset_monthly_state(self) -> None:
        async with self._lock:
            self._state.reset_monthly_state()
            payload = {
                "equity": self._state.equity,
                "monthly_start_equity": self._state.monthly_start_equity,
            }

        await self._emit_event("risk.monthly_state_reset", payload)

    def stats(self) -> dict[str, Any]:
        base_r_snapshot = self._risk_unit_calculator.calculate(self._state)
        state_snapshot = self._state.snapshot(
            risk_unit=max(base_r_snapshot.effective_risk_unit, 1e-12),
            caution_daily_loss_r=self._config.budget.caution_daily_loss_r,
            soft_daily_loss_r=self._config.budget.soft_daily_loss_r,
            hard_daily_loss_r=self._config.budget.hard_daily_loss_r,
            weekly_hard_loss_r=self._config.budget.weekly_hard_loss_r,
            monthly_review_loss_r=self._config.budget.monthly_review_loss_r,
            emergency_stop_loss_r=self._config.budget.emergency_stop_loss_r,
        )

        return {
            "running": self._running,
            "started_at": self._started_at,
            "subscriptions": len(self._subscriptions),
            "scheduler_jobs": len(self._scheduler_jobs),
            "state": state_snapshot,
            "metrics": self._metrics.snapshot(state_snapshot=state_snapshot),
            "circuit_breaker": self._circuit_breaker.stats(),
        }

    async def _handle_signal_generated(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}

        try:
            request = self._request_from_payload(payload)
        except (TypeError, ValueError) as exc:
            self._logger.warning(
                "Ignoring malformed signal.generated event | reason=%s",
                str(exc),
            )
            await self._emit_event(
                "risk.signal_invalid",
                {
                    "reason": str(exc),
                    "payload_keys": sorted(str(key) for key in payload.keys()),
                },
                priority=EventPriority.HIGH,
            )
            return

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

        if topic in {"account.updated", "account.snapshot"}:
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

        try:
            position = self._position_from_payload(payload)
        except ValueError as exc:
            self._logger.warning(
                "Ignoring malformed position.opened event | reason=%s",
                str(exc),
            )
            return

        await self.on_position_opened(position)

    async def _handle_position_updated(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}

        if not isinstance(payload, dict):
            return

        raw_symbol = payload.get("symbol")

        if not isinstance(raw_symbol, str) or not raw_symbol:
            return

        symbol = raw_symbol

        await self.on_position_updated(
            symbol,
            position_id=payload.get("position_id"),
            size=self._to_float_or_none(
                payload.get("size")
                or payload.get("quantity")
                or payload.get("filled_quantity")
            ),
            mark_price=self._to_float_or_none(
                payload.get("mark_price")
                or payload.get("fill_price")
                or payload.get("average_fill_price")
            ),
            notional_value=self._to_float_or_none(
                payload.get("notional_value")
                or payload.get("notional")
                or payload.get("fill_notional")
            ),
            leverage=self._to_float_or_none(
                payload.get("leverage")
                or payload.get("final_leverage")
            ),
            margin_used=self._to_float_or_none(
                payload.get("margin_used")
                or payload.get("margin")
                or payload.get("final_margin")
            ),
            risk_amount=self._to_float_or_none(
                payload.get("risk_amount")
                or payload.get("final_risk_amount")
                or payload.get("open_risk")
            ),
            stop_loss=self._to_float_or_none(payload.get("stop_loss")),
            take_profit=self._to_float_or_none(payload.get("take_profit")),
            unrealized_pnl=self._to_float_or_none(payload.get("unrealized_pnl")),
        )

    async def _handle_position_closed(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}

        if not isinstance(payload, dict):
            return

        raw_symbol = payload.get("symbol")

        if not isinstance(raw_symbol, str) or not raw_symbol:
            return

        symbol = raw_symbol

        await self.on_position_closed(
            symbol,
            realized_pnl=self._to_float_or_none(payload.get("realized_pnl")) or 0.0,
            position_id=payload.get("position_id"),
        )

    async def _handle_execution_rejected(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}
        await self.on_execution_rejected(
            reservation_id=payload.get("reservation_id"),
            signal_id=payload.get("signal_id"),
            symbol=payload.get("symbol"),
            position_id=payload.get("position_id"),
            reason=payload.get("reason") or payload.get("message") or "Execution order rejected",
        )

    async def _handle_execution_failed(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}
        await self.on_execution_failed(
            reservation_id=payload.get("reservation_id"),
            signal_id=payload.get("signal_id"),
            symbol=payload.get("symbol"),
            position_id=payload.get("position_id"),
            reason=payload.get("reason") or payload.get("message") or "Execution order failed",
        )

    async def _handle_execution_cancelled(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}
        await self.on_execution_cancelled(
            reservation_id=payload.get("reservation_id"),
            signal_id=payload.get("signal_id"),
            symbol=payload.get("symbol"),
            position_id=payload.get("position_id"),
            reason=payload.get("reason") or payload.get("message") or "Execution order cancelled",
        )

    async def _handle_execution_filled(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}
        self._circuit_breaker.register_success()

        # A filled order should never leave an approved risk reservation pending.
        # Prefer the explicit position payload when available; it lets state and
        # metrics confirm the reservation and register the opened position in one
        # path. If the fill event does not contain enough position data, release
        # the reservation fail-safe so projected risk is not stuck forever.
        if payload.get("symbol") and payload.get("side") and (
            payload.get("size") is not None
            or payload.get("quantity") is not None
            or payload.get("filled_quantity") is not None
            or payload.get("fill_quantity") is not None
        ):
            try:
                position = self._position_from_payload(payload)
            except ValueError as exc:
                self._logger.warning(
                    "Malformed execution.order_filled position payload | reason=%s",
                    str(exc),
                )
            else:
                await self.on_position_opened(position)
                return

        events = await self._release_reservation_for_execution_event(
            reservation_id=payload.get("reservation_id"),
            signal_id=payload.get("signal_id"),
            symbol=payload.get("symbol"),
            position_id=payload.get("position_id"),
            status="released",
            reason="Execution order filled without usable position payload",
        )
        await self._emit_events(events)

    async def _handle_day_rollover(self, event: Any) -> None:
        await self.reset_daily_state()

    async def _handle_week_rollover(self, event: Any) -> None:
        await self.reset_weekly_state()

    async def _handle_month_rollover(self, event: Any) -> None:
        await self.reset_monthly_state()

    async def _handle_scheduler_job_failed(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}
        job_name = payload.get("job_name", "unknown")

        await self.on_execution_failure(
            message=f"Scheduler job failed: {job_name}",
            reason=CircuitBreakerReason.SYSTEM_ERROR_RATE,
        )

    async def _handle_manual_halt(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}
        message = payload.get("message") or "Manual risk halt"

        async with self._lock:
            self._circuit_breaker.trigger_manual_halt(self._state, message=message)
            self._state.halt_trading(message)
            self._metrics.register_circuit_breaker_trigger()
            payload = {
                "reason": CircuitBreakerReason.MANUAL_HALT.value,
                "message": message,
                "manual_release_required": True,
            }

        await self._emit_event("risk.kill_switch", payload, priority=EventPriority.CRITICAL)
        await self._emit_event("risk.trading_halted", payload, priority=EventPriority.CRITICAL)
        await self._emit_event("risk.circuit_breaker_triggered", payload, priority=EventPriority.CRITICAL)

    async def _handle_manual_resume(self, event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}
        clear_emergency_stop = bool(payload.get("clear_emergency_stop", False))

        async with self._lock:
            released = self._circuit_breaker.force_release(
                self._state,
                clear_emergency_stop=clear_emergency_stop,
            )
            if clear_emergency_stop:
                self._state.clear_emergency_stop()
            if released and not self._state.emergency_stop_active:
                self._state.resume_trading()

        payload = {
            "released": released,
            "clear_emergency_stop": clear_emergency_stop,
            "emergency_stop_active": self._state.emergency_stop_active,
        }
        await self._emit_event("risk.manual_resume_processed", payload)
        if released and not self._state.emergency_stop_active:
            await self._emit_event("risk.trading_resumed", payload)

    async def _finalize_decision(
            self,
            request: RiskEvaluationRequest,
            decision: RiskDecision,
            *,
            latency_ms: float | None = None,
    ) -> None:
        """
        Backward-compatible async finalizer.

        New code uses _finalize_decision_locked() under the manager lock and emits
        returned events afterwards. This wrapper is kept for older call sites.
        """
        async with self._lock:
            events = self._finalize_decision_locked(request, decision, latency_ms=latency_ms)
        await self._emit_events(events)

    def _finalize_decision_locked(
            self,
            request: RiskEvaluationRequest,
            decision: RiskDecision,
            *,
            latency_ms: float | None = None,
    ) -> list[tuple[str, dict[str, Any], EventPriority | None]]:
        events: list[tuple[str, dict[str, Any], EventPriority | None]] = []

        if decision.allowed and self._should_reserve_for_decision(request, decision):
            try:
                reservation = self._create_reservation_locked(request, decision)
                setattr(decision, "reservation_id", reservation.reservation_id)
                setattr(decision, "reservation_expires_at", reservation.expires_at)
                decision.metadata["reservation_id"] = reservation.reservation_id
                decision.metadata["reservation_expires_at"] = reservation.expires_at
                self._metrics.register_reservation_created(
                    reservation_id=reservation.reservation_id,
                    symbol=reservation.symbol,
                    tier=reservation.tier,
                    strategy_name=reservation.strategy_name,
                    open_risk=reservation.open_risk,
                    margin=reservation.margin,
                    notional=reservation.notional,
                )
                events.append(
                    (
                        "risk.reservation.created",
                        self._serialize_reservation(reservation, status="created"),
                        EventPriority.NORMAL,
                    )
                )
            except Exception as exc:
                self._logger.exception("Failed to create risk reservation")
                self._metrics.register_reservation_failed(
                    reservation_id=None,
                    symbol=request.symbol,
                    tier=decision.final_tier or request.tier,
                    strategy_name=request.strategy_name,
                    open_risk=decision.final_risk_amount or 0.0,
                    margin=decision.final_margin or 0.0,
                    notional=decision.final_notional or 0.0,
                    reason=str(exc),
                )
                if getattr(self._config.reservation, "fail_closed_on_reservation_error", True):
                    decision.allowed = False
                    decision.decision = RiskDecisionType.DENY
                    decision.reason = f"Risk reservation failed: {exc}"

        self._metrics.register_decision(decision, latency_ms=latency_ms)

        if not decision.allowed:
            self._state.register_rejection(
                reason=decision.reason or decision.decision.value,
                symbol=request.symbol,
                strategy_name=request.strategy_name,
                tier=decision.final_tier or request.tier,
            )

        if decision.decision is RiskDecisionType.HALT_TRADING:
            self._state.halt_trading(decision.reason or "Risk halt triggered")

        if decision.decision is RiskDecisionType.EMERGENCY_STOP:
            self._state.emergency_stop(decision.reason or "Emergency stop triggered")

        serialized = self._serialize_decision(request, decision)
        events.extend(self._events_for_decision(decision, serialized))

        self._logger.info(
            "Risk decision completed | symbol=%s decision=%s allowed=%s tier=%s size=%s leverage=%s reservation_id=%s",
            request.symbol,
            decision.decision.value,
            decision.allowed,
            decision.final_tier.value if decision.final_tier else None,
            decision.final_size,
            decision.final_leverage,
            getattr(decision, "reservation_id", None),
            extra={
                "symbol": request.symbol,
                "strategy_name": request.strategy_name,
                "signal_id": request.signal_id,
                "reservation_id": getattr(decision, "reservation_id", None),
            },
        )
        return events

    async def _emit_event(
            self,
            topic: str,
            payload: dict[str, Any],
            *,
            priority: EventPriority | None = None,
    ) -> None:
        if self._event_bus is None:
            return

        try:
            if priority is not None:
                await self._event_bus.emit(
                    topic,
                    payload,
                    source=self._service_name,
                    priority=priority,
                )
            else:
                await self._event_bus.emit(
                    topic,
                    payload,
                    source=self._service_name,
                )
        except TypeError:
            await self._event_bus.emit(
                topic,
                payload,
                source=self._service_name,
            )
        except Exception:
            self._logger.exception(
                "Failed to emit risk event | topic=%s",
                topic,
            )

    async def on_execution_rejected(
            self,
            *,
            reservation_id: str | None = None,
            signal_id: str | None = None,
            symbol: str | None = None,
            position_id: str | None = None,
            reason: str | None = None,
    ) -> None:
        events = await self._release_reservation_for_execution_event(
            reservation_id=reservation_id,
            signal_id=signal_id,
            symbol=symbol,
            position_id=position_id,
            status="released",
            reason=reason or "Execution order rejected",
        )
        await self._emit_events(events)

    async def on_execution_cancelled(
            self,
            *,
            reservation_id: str | None = None,
            signal_id: str | None = None,
            symbol: str | None = None,
            position_id: str | None = None,
            reason: str | None = None,
    ) -> None:
        events = await self._release_reservation_for_execution_event(
            reservation_id=reservation_id,
            signal_id=signal_id,
            symbol=symbol,
            position_id=position_id,
            status="released",
            reason=reason or "Execution order cancelled",
        )
        await self._emit_events(events)

    async def on_execution_failed(
            self,
            *,
            reservation_id: str | None = None,
            signal_id: str | None = None,
            symbol: str | None = None,
            position_id: str | None = None,
            reason: str | None = None,
    ) -> None:
        async with self._lock:
            events = self._release_reservation_locked(
                reservation_id=reservation_id,
                signal_id=signal_id,
                symbol=symbol,
                position_id=position_id,
                status="failed",
                reason=reason or "Execution order failed",
            )

            was_active = self._state.is_circuit_breaker_active()
            triggered = self._circuit_breaker.register_failure(
                self._state,
                reason=CircuitBreakerReason.EXECUTION_FAILURES,
                message=reason,
            )
            is_active = self._state.is_circuit_breaker_active()

            if triggered or (not was_active and is_active):
                self._metrics.register_circuit_breaker_trigger()

                circuit_reason = self._state.circuit_breaker.reason

                if isinstance(circuit_reason, CircuitBreakerReason):
                    reason_value = circuit_reason.value
                elif isinstance(circuit_reason, str):
                    reason_value = circuit_reason
                else:
                    reason_value = None

                events.append(
                    (
                        "risk.kill_switch",
                        {
                            "reason": reason_value,
                            "message": self._state.circuit_breaker.message,
                            "cooldown_until": self._state.circuit_breaker.cooldown_until,
                            "manual_release_required": self._state.circuit_breaker.manual_release_required,
                        },
                        EventPriority.CRITICAL,
                    )
                )

        await self._emit_events(events)

    async def cleanup_expired_reservations(
            self,
            *,
            now_ts: float | None = None,
    ) -> list[Any]:
        async with self._lock:
            expired, events = self._expire_reservations_locked(
                reason="scheduled_cleanup",
                now_ts=now_ts,
            )
        await self._emit_events(events)
        return expired

    async def _release_reservation_for_execution_event(
            self,
            *,
            reservation_id: str | None = None,
            signal_id: str | None = None,
            symbol: str | None = None,
            position_id: str | None = None,
            status: str,
            reason: str | None = None,
    ) -> list[tuple[str, dict[str, Any], EventPriority | None]]:
        async with self._lock:
            return self._release_reservation_locked(
                reservation_id=reservation_id,
                signal_id=signal_id,
                symbol=symbol,
                position_id=position_id,
                status=status,
                reason=reason,
            )

    def _release_reservation_locked(
            self,
            *,
            reservation_id: str | None = None,
            signal_id: str | None = None,
            symbol: str | None = None,
            position_id: str | None = None,
            status: str,
            reason: str | None = None,
    ) -> list[tuple[str, dict[str, Any], EventPriority | None]]:
        reservation = self._state.release_risk_reservation(
            reservation_id,
            signal_id=signal_id,
            symbol=symbol,
            position_id=position_id,
        )
        if reservation is None:
            return []

        age_ms = self._reservation_age_ms(reservation)
        if status == "failed":
            self._metrics.register_reservation_failed(
                reservation_id=reservation.reservation_id,
                symbol=reservation.symbol,
                tier=reservation.tier,
                strategy_name=reservation.strategy_name,
                open_risk=reservation.open_risk,
                margin=reservation.margin,
                notional=reservation.notional,
                reason=reason,
                age_ms=age_ms,
            )
        else:
            self._metrics.register_reservation_released(
                reservation_id=reservation.reservation_id,
                symbol=reservation.symbol,
                tier=reservation.tier,
                strategy_name=reservation.strategy_name,
                open_risk=reservation.open_risk,
                margin=reservation.margin,
                notional=reservation.notional,
                reason=reason,
                age_ms=age_ms,
            )

        return [
            (
                f"risk.reservation.{status}",
                self._serialize_reservation(reservation, status=status, reason=reason, age_ms=age_ms),
                EventPriority.NORMAL,
            )
        ]

    def _expire_reservations_locked(
            self,
            *,
            reason: str,
            now_ts: float | None = None,
    ) -> tuple[list[Any], list[tuple[str, dict[str, Any], EventPriority | None]]]:
        reservation_cfg = getattr(self._config, "reservation", None)
        if reservation_cfg is not None and not getattr(reservation_cfg, "auto_expire_on_evaluate", True):
            return [], []

        expired = self._state.expire_pending_reservations(now_ts=now_ts)
        events: list[tuple[str, dict[str, Any], EventPriority | None]] = []
        for reservation in expired:
            age_ms = self._reservation_age_ms(reservation)
            self._metrics.register_reservation_expired(
                reservation_id=reservation.reservation_id,
                symbol=reservation.symbol,
                tier=reservation.tier,
                strategy_name=reservation.strategy_name,
                open_risk=reservation.open_risk,
                margin=reservation.margin,
                notional=reservation.notional,
                age_ms=age_ms,
            )
            events.append(
                (
                    "risk.reservation.expired",
                    self._serialize_reservation(
                        reservation,
                        status="expired",
                        reason=reason,
                        age_ms=age_ms,
                    ),
                    EventPriority.NORMAL,
                )
            )
        return expired, events

    def _should_reserve_for_decision(
            self,
            request: RiskEvaluationRequest,
            decision: RiskDecision,
    ) -> bool:
        reservation_cfg = getattr(self._config, "reservation", None)
        if reservation_cfg is None:
            return False
        if not getattr(reservation_cfg, "enabled", False):
            return False
        if not getattr(reservation_cfg, "reserve_on_allow", True):
            return False
        if not decision.allowed:
            return False
        if not self._order_increases_risk(request.order_intent):
            return False
        return bool(decision.final_size and decision.final_size > 0)

    def _create_reservation_locked(
            self,
            request: RiskEvaluationRequest,
            decision: RiskDecision,
    ) -> Any:
        reservation_cfg = self._config.reservation
        pending = list(self._state.pending_reservations.values())

        raw_max_pending = getattr(reservation_cfg, "max_pending_reservations", None)
        max_pending = raw_max_pending if isinstance(raw_max_pending, int) else None

        if max_pending is not None and len(pending) >= max_pending:
            raise RuntimeError("maximum pending risk reservations exceeded")

        raw_max_per_symbol = getattr(reservation_cfg, "max_pending_per_symbol", None)
        max_per_symbol = raw_max_per_symbol if isinstance(raw_max_per_symbol, int) else None

        if max_per_symbol is not None:
            symbol_pending = sum(1 for item in pending if item.symbol == request.symbol)

            if symbol_pending >= max_per_symbol:
                raise RuntimeError("maximum pending risk reservations per symbol exceeded")

        raw_max_per_strategy = getattr(reservation_cfg, "max_pending_per_strategy", None)
        max_per_strategy = (
            raw_max_per_strategy if isinstance(raw_max_per_strategy, int) else None
        )

        if max_per_strategy is not None and request.strategy_name:
            strategy_pending = sum(
                1 for item in pending if item.strategy_name == request.strategy_name
            )

            if strategy_pending >= max_per_strategy:
                raise RuntimeError("maximum pending risk reservations per strategy exceeded")

        raw_ttl_seconds = getattr(reservation_cfg, "ttl_seconds", 30)
        ttl_seconds = raw_ttl_seconds if isinstance(raw_ttl_seconds, int) else 30

        return self._state.reserve_risk(
            symbol=request.symbol,
            side=request.side,
            signal_id=request.signal_id,
            strategy_name=request.strategy_name,
            tier=decision.final_tier or request.tier,
            size=decision.final_size or 0.0,
            open_risk=decision.final_risk_amount or 0.0,
            margin=decision.final_margin or 0.0,
            notional=decision.final_notional or 0.0,
            ttl_seconds=ttl_seconds,
            metadata={
                "decision": decision.decision.value,
                "risk_mode": decision.risk_mode.value,
                "source": self._service_name,
            },
        )

    def _events_for_decision(
            self,
            decision: RiskDecision,
            serialized: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any], EventPriority | None]]:
        primary_topic, primary_priority = self._topic_for_decision(decision)
        events: list[tuple[str, dict[str, Any], EventPriority | None]] = [
            (primary_topic, serialized, primary_priority)
        ]

        if decision.decision is RiskDecisionType.EMERGENCY_STOP:
            events.append(("risk.emergency_stop", serialized, EventPriority.CRITICAL))
        elif decision.decision is RiskDecisionType.HALT_TRADING:
            events.append(("risk.trading_halted", serialized, EventPriority.CRITICAL))
        elif not decision.allowed:
            events.append(("risk.rejected", serialized, EventPriority.HIGH))
        elif decision.decision in {
            RiskDecisionType.REDUCE_SIZE,
            RiskDecisionType.REDUCE_RISK,
            RiskDecisionType.DOWNGRADE_TIER,
        }:
            events.append(("risk.limit_warning", serialized, EventPriority.NORMAL))
            events.append(("risk.size_adjusted", serialized, EventPriority.NORMAL))
        else:
            events.append(("risk.approved", serialized, EventPriority.NORMAL))

        # Deduplicate primary/legacy pair when they match.
        deduped: list[tuple[str, dict[str, Any], EventPriority | None]] = []
        seen: set[str] = set()
        for topic, payload, priority in events:
            if topic in seen:
                continue
            seen.add(topic)
            deduped.append((topic, payload, priority))
        return deduped

    async def _emit_events(
            self,
            events: list[tuple[str, dict[str, Any], EventPriority | None]],
    ) -> None:
        for topic, payload, priority in events:
            await self._emit_event(topic, payload, priority=priority)

    @staticmethod
    def _order_increases_risk(order_intent: OrderIntent) -> bool:
        value = getattr(order_intent, "increases_risk", None)
        if isinstance(value, bool):
            return value
        return order_intent in {OrderIntent.OPEN, OrderIntent.INCREASE}

    @staticmethod
    def _reservation_age_ms(reservation: Any) -> float:
        return max(0.0, (time.time() - reservation.created_at) * 1000.0)

    @staticmethod
    def _serialize_reservation(
            reservation: Any,
            *,
            status: str,
            reason: str | None = None,
            age_ms: float | None = None,
    ) -> dict[str, Any]:
        return {
            "reservation_id": reservation.reservation_id,
            "status": status,
            "reason": reason,
            "signal_id": reservation.signal_id,
            "symbol": reservation.symbol,
            "side": reservation.side.value,
            "strategy_name": reservation.strategy_name,
            "tier": reservation.tier.value if reservation.tier else None,
            "position_id": reservation.position_id,
            "size": reservation.size,
            "open_risk": reservation.open_risk,
            "margin": reservation.margin,
            "notional": reservation.notional,
            "created_at": reservation.created_at,
            "expires_at": reservation.expires_at,
            "age_ms": age_ms,
            "metadata": RiskManager._json_safe(reservation.metadata),
        }

    def _is_strategy_disabled(self, strategy_name: str) -> bool:
        strategy_state = self._state.get_strategy_state(strategy_name)
        status = getattr(strategy_state, "status", None)
        return getattr(status, "value", status) == "disabled"

    def _resolve_strategy_multiplier(self, request: RiskEvaluationRequest) -> float:
        if not request.strategy_name:
            return 1.0
        return self._state.get_strategy_state(request.strategy_name).risk_multiplier

    def _resolve_symbol_multiplier(self, request: RiskEvaluationRequest) -> float:
        symbol_state = self._state.get_symbol_state(request.symbol)
        if symbol_state.status.value == "reduced":
            return 0.5
        return 1.0

    @staticmethod
    def _extract_tier_profile(result: RiskCheckResult) -> TierRiskProfile | None:
        profile = result.metadata.get("tier_profile") if result.metadata else None
        return profile if isinstance(profile, TierRiskProfile) else None

    @staticmethod
    def _extract_ev_snapshot(result: RiskCheckResult) -> ExpectedValueSnapshot:
        snapshot = result.metadata.get("expected_value_snapshot") if result.metadata else None
        if isinstance(snapshot, ExpectedValueSnapshot):
            return snapshot
        raise ValueError("ExpectedValueSnapshot missing from risk_reward check")

    @staticmethod
    def _collect_violations(checks: dict[str, RiskCheckResult]) -> list[Any]:
        violations = []
        for result in checks.values():
            violations.extend(result.violations)
        return violations

    @staticmethod
    def _build_terminal_decision(
            *,
            request: RiskEvaluationRequest,
            check: RiskCheckResult,
            checks: dict[str, RiskCheckResult],
            reason: str,
            final_size: float | None,
            final_leverage: float | None,
            final_tier: TradeTier | None = None,
            ev_snapshot: ExpectedValueSnapshot | None = None,
    ) -> RiskDecision:
        return RiskDecision(
            allowed=False,
            decision=check.decision,
            final_size=final_size,
            final_leverage=final_leverage,
            final_tier=final_tier or check.adjusted_tier or request.tier,
            final_risk_amount=check.adjusted_risk_amount,
            final_margin=check.adjusted_margin,
            risk_mode=check.risk_mode or RiskMode.NORMAL,
            risk_reward_ratio=ev_snapshot.risk_reward_ratio if ev_snapshot else None,
            expected_value=ev_snapshot.expected_value if ev_snapshot else None,
            expected_value_after_cost=(
                ev_snapshot.expected_value_after_cost if ev_snapshot else None
            ),
            expected_cost=ev_snapshot.expected_cost if ev_snapshot else None,
            cost_to_reward_ratio=ev_snapshot.cost_to_reward_ratio if ev_snapshot else None,
            reason=reason,
            signal_id=request.signal_id,
            strategy_name=request.strategy_name,
            symbol=request.symbol,
            side=request.side,
            order_intent=request.order_intent,
            violations=RiskManager._collect_violations(checks),
            checks=checks,
            metadata=dict(request.metadata),
        )

    @staticmethod
    def _resolve_success_decision(checks: dict[str, RiskCheckResult]) -> RiskDecisionType:
        priority = [
            RiskDecisionType.REDUCE_SIZE,
            RiskDecisionType.REDUCE_RISK,
            RiskDecisionType.DOWNGRADE_TIER,
        ]

        for decision_type in priority:
            if any(result.decision is decision_type for result in checks.values()):
                return decision_type

        return RiskDecisionType.ALLOW

    @staticmethod
    def _build_success_reason(decision_type: RiskDecisionType) -> str:
        if decision_type is RiskDecisionType.REDUCE_SIZE:
            return "Request approved with reduced size or leverage"
        if decision_type is RiskDecisionType.REDUCE_RISK:
            return "Request approved with reduced risk"
        if decision_type is RiskDecisionType.DOWNGRADE_TIER:
            return "Request approved with downgraded tier"
        return "Request approved"

    @staticmethod
    def _topic_for_decision(decision: RiskDecision) -> tuple[str, EventPriority | None]:
        if decision.decision is RiskDecisionType.EMERGENCY_STOP:
            return "risk.kill_switch", EventPriority.CRITICAL

        if decision.decision is RiskDecisionType.HALT_TRADING:
            return "risk.kill_switch", EventPriority.CRITICAL

        if not decision.allowed:
            return "risk.position_blocked", EventPriority.HIGH

        return "signal.confirmed", EventPriority.NORMAL

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
            "risk_mode": decision.risk_mode.value,
            "entry_price": request.entry_price,
            "stop_loss": request.stop_loss,
            "take_profit": request.take_profit,
            "order_intent": request.order_intent.value,
            "margin_mode": request.margin_mode.value,
            "requested_leverage": request.requested_leverage,
            "requested_size": request.requested_size,
            "requested_margin": request.requested_margin,
            "final_tier": decision.final_tier.value if decision.final_tier else None,
            "tier": decision.final_tier.value if decision.final_tier else None,
            "final_size": decision.final_size,
            "final_leverage": decision.final_leverage,
            "final_risk_amount": decision.final_risk_amount,
            "final_margin": decision.final_margin,
            "final_notional": decision.final_notional,
            "reservation_id": getattr(decision, "reservation_id", None),
            "reservation_expires_at": getattr(decision, "reservation_expires_at", None),
            "risk_reward_ratio": decision.risk_reward_ratio,
            "expected_value": decision.expected_value,
            "expected_value_after_cost": decision.expected_value_after_cost,
            "expected_cost": decision.expected_cost,
            "cost_to_reward_ratio": decision.cost_to_reward_ratio,
            "reason": decision.reason,
            "violations": [
                {
                    "type": violation.violation_type.value,
                    "level": violation.level.value,
                    "message": violation.message,
                    "current_value": violation.current_value,
                    "limit_value": violation.limit_value,
                    "symbol": violation.symbol,
                    "strategy_name": violation.strategy_name,
                    "tier": violation.tier.value if violation.tier else None,
                    "metadata": RiskManager._json_safe(violation.metadata),
                }
                for violation in decision.violations
            ],
            "checks": {
                check_name: {
                    "passed": check_result.passed,
                    "decision": check_result.decision.value,
                    "adjusted_tier": (
                        check_result.adjusted_tier.value
                        if check_result.adjusted_tier
                        else None
                    ),
                    "adjusted_size": check_result.adjusted_size,
                    "adjusted_margin": check_result.adjusted_margin,
                    "adjusted_leverage": check_result.adjusted_leverage,
                    "adjusted_risk_amount": check_result.adjusted_risk_amount,
                    "risk_mode": (
                        check_result.risk_mode.value
                        if check_result.risk_mode
                        else None
                    ),
                    "reason": check_result.reason,
                    "metadata": RiskManager._json_safe(check_result.metadata),
                }
                for check_name, check_result in decision.checks.items()
            },
            "metadata": RiskManager._json_safe(decision.metadata),
        }

    @staticmethod
    def _request_from_payload(payload: dict[str, Any]) -> RiskEvaluationRequest:
        symbol = payload.get("symbol")
        side_raw = payload.get("side")
        entry_price = RiskManager._to_float_or_none(payload.get("entry_price"))

        if not symbol:
            raise ValueError("symbol is required")
        if not side_raw:
            raise ValueError("side is required")
        if entry_price is None:
            raise ValueError("entry_price is required")

        raw_metadata = payload.get("metadata")
        metadata: dict[str, Any] = {}

        if isinstance(raw_metadata, dict):
            metadata = {
                str(key): value
                for key, value in raw_metadata.items()
            }

        return RiskEvaluationRequest(
            symbol=str(symbol),
            side=RiskManager._enum_from_value(PositionSide, side_raw),
            entry_price=entry_price,
            stop_loss=RiskManager._to_float_or_none(payload.get("stop_loss")),
            take_profit=RiskManager._to_float_or_none(payload.get("take_profit")),
            signal_id=payload.get("signal_id"),
            strategy_name=payload.get("strategy_name"),
            tier=RiskManager._optional_enum_from_value(TradeTier, payload.get("tier")),
            order_intent=RiskManager._enum_from_value(
                OrderIntent,
                payload.get("order_intent", OrderIntent.OPEN.value),
            ),
            liquidity_class=RiskManager._enum_from_value(
                LiquidityClass,
                payload.get("liquidity_class", LiquidityClass.NORMAL.value),
            ),
            execution_quality=RiskManager._enum_from_value(
                ExecutionQuality,
                payload.get("execution_quality", ExecutionQuality.ACCEPTABLE.value),
            ),
            confidence=RiskManager._to_float_or_none(payload.get("confidence")),
            edge_score=RiskManager._to_float_or_none(payload.get("edge_score")),
            volatility=RiskManager._to_float_or_none(payload.get("volatility")),
            expected_reward=RiskManager._to_float_or_none(payload.get("expected_reward")),
            expected_loss=RiskManager._to_float_or_none(payload.get("expected_loss")),
            expected_win_probability=RiskManager._to_float_or_none(
                payload.get("expected_win_probability")
            ),
            expected_cost=RiskManager._to_float_or_none(payload.get("expected_cost")),
            execution_cost=RiskManager._execution_cost_from_payload(payload),
            requested_size=RiskManager._to_float_or_none(payload.get("requested_size")),
            requested_margin=RiskManager._to_float_or_none(payload.get("requested_margin")),
            requested_leverage=RiskManager._to_float_or_none(
                payload.get("requested_leverage")
            ),
            reduce_only=bool(payload.get("reduce_only", False)),
            margin_mode=RiskManager._enum_from_value(
                MarginMode,
                payload.get("margin_mode", MarginMode.ISOLATED.value),
            ),
            timestamp=RiskManager._to_float_or_none(payload.get("timestamp")),
            metadata=metadata,
        )

    @staticmethod
    def _position_from_payload(payload: dict[str, Any]) -> PortfolioPosition:
        metadata = dict(payload.get("metadata", {}) or {})

        def first_value(*keys: str) -> Any:
            for key in keys:
                value = payload.get(key)
                if value is not None:
                    return value
            for key in keys:
                value = metadata.get(key)
                if value is not None:
                    return value
            nested_payload = metadata.get("payload")
            if isinstance(nested_payload, dict):
                for key in keys:
                    value = nested_payload.get(key)
                    if value is not None:
                        return value
            return None

        def first_float(*keys: str) -> float | None:
            return RiskManager._to_float_or_none(first_value(*keys))

        symbol = first_value("symbol")
        side_raw = first_value("side", "position_side")
        size = first_float("size", "quantity", "filled_quantity", "fill_quantity", "final_size")
        entry_price = first_float("entry_price", "fill_price", "average_fill_price", "price")

        if not symbol:
            raise ValueError("symbol is required")
        if not side_raw:
            raise ValueError("side is required")
        if size is None:
            raise ValueError("size is required")
        if entry_price is None:
            raise ValueError("entry_price is required")

        mark_price = first_float("mark_price", "fill_price", "average_fill_price", "price") or entry_price
        notional_value = first_float("notional_value", "notional", "fill_notional", "final_notional") or abs(size * entry_price)
        margin_used = first_float("margin_used", "margin", "final_margin") or 0.0
        risk_amount = first_float("risk_amount", "open_risk", "final_risk_amount") or 0.0
        tier = RiskManager._optional_enum_from_value(TradeTier, first_value("tier", "final_tier"))

        reservation_id = first_value("reservation_id")
        if reservation_id is not None:
            metadata["reservation_id"] = reservation_id

        opened_at = first_float("opened_at")
        if opened_at is None:
            opened_at_ms = first_float("opened_at_ms", "timestamp_ms", "filled_at_ms")
            opened_at = (opened_at_ms / 1000.0) if opened_at_ms and opened_at_ms > 10_000_000_000 else opened_at_ms

        updated_at = first_float("updated_at")
        if updated_at is None:
            updated_at_ms = first_float("updated_at_ms", "timestamp_ms", "filled_at_ms")
            updated_at = (updated_at_ms / 1000.0) if updated_at_ms and updated_at_ms > 10_000_000_000 else updated_at_ms

        return PortfolioPosition(
            symbol=str(symbol),
            side=RiskManager._enum_from_value(PositionSide, side_raw),
            size=size,
            entry_price=entry_price,
            mark_price=mark_price,
            notional_value=notional_value,
            leverage=first_float("leverage", "final_leverage"),
            margin_used=margin_used,
            risk_amount=risk_amount,
            stop_loss=first_float("stop_loss"),
            take_profit=first_float("take_profit"),
            tier=tier,
            strategy_name=first_value("strategy_name"),
            signal_id=first_value("signal_id"),
            position_id=first_value("position_id"),
            realized_pnl=first_float("realized_pnl", "net_realized_pnl") or 0.0,
            unrealized_pnl=first_float("unrealized_pnl") or 0.0,
            opened_at=opened_at or time.time(),
            updated_at=updated_at,
            metadata=metadata,
        )

    @staticmethod
    def _execution_cost_from_payload(payload: dict[str, Any]) -> Any:
        raw = payload.get("execution_cost")
        if isinstance(raw, ExecutionCostEstimate):
            return raw

        if not isinstance(raw, dict):
            return None

        return ExecutionCostEstimate(
            spread_cost=RiskManager._to_float_or_none(raw.get("spread_cost")) or 0.0,
            slippage_cost=RiskManager._to_float_or_none(raw.get("slippage_cost")) or 0.0,
            fee_cost=RiskManager._to_float_or_none(raw.get("fee_cost")) or 0.0,
            funding_cost=RiskManager._to_float_or_none(raw.get("funding_cost")) or 0.0,
            other_cost=RiskManager._to_float_or_none(raw.get("other_cost")) or 0.0,
            spread_pct=RiskManager._to_float_or_none(raw.get("spread_pct")),
            slippage_pct=RiskManager._to_float_or_none(raw.get("slippage_pct")),
            quality=RiskManager._enum_from_value(
                ExecutionQuality,
                raw.get("quality", ExecutionQuality.ACCEPTABLE.value),
            ),
            metadata=dict(raw.get("metadata", {})),
        )

    @staticmethod
    def _enum_from_value(enum_cls: type[Enum], value: Any) -> Any:
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, str):
            return enum_cls(value.lower())
        return enum_cls(value)

    @staticmethod
    def _optional_enum_from_value(enum_cls: type[Enum], value: Any) -> Any:
        if value is None:
            return None
        return RiskManager._enum_from_value(enum_cls, value)

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (time.perf_counter() - started_at) * 1000.0

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if dataclasses.is_dataclass(value):
            return {
                key: RiskManager._json_safe(val)
                for key, val in dataclasses.asdict(value).items()
            }

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, dict):
            return {
                str(key): RiskManager._json_safe(val)
                for key, val in value.items()
            }

        if isinstance(value, list):
            return [RiskManager._json_safe(item) for item in value]

        if isinstance(value, tuple):
            return [RiskManager._json_safe(item) for item in value]

        return value

    @staticmethod
    def _normalize_circuit_breaker_reason(state: RiskState) -> None:
        """
        Keep RiskState tolerant to tests/events that set circuit breaker reason
        as a raw string while CircuitBreaker internals expect CircuitBreakerReason.
        """
        reason = getattr(state.circuit_breaker, "reason", None)
        if reason is None or isinstance(reason, CircuitBreakerReason):
            return

        if isinstance(reason, str):
            try:
                state.circuit_breaker.reason = CircuitBreakerReason(reason)
            except ValueError:
                state.circuit_breaker.reason = CircuitBreakerReason.MANUAL_HALT
            return

        value = getattr(reason, "value", None)
        if isinstance(value, str):
            try:
                state.circuit_breaker.reason = CircuitBreakerReason(value)
            except ValueError:
                state.circuit_breaker.reason = CircuitBreakerReason.MANUAL_HALT

    @staticmethod
    def _finite_request_validation(request: RiskEvaluationRequest) -> RiskCheckResult:
        fields = {
            "entry_price": request.entry_price,
            "stop_loss": request.stop_loss,
            "take_profit": request.take_profit,
            "confidence": request.confidence,
            "edge_score": request.edge_score,
            "volatility": request.volatility,
            "expected_reward": request.expected_reward,
            "expected_loss": request.expected_loss,
            "expected_win_probability": request.expected_win_probability,
            "expected_cost": request.expected_cost,
            "requested_size": request.requested_size,
            "requested_margin": request.requested_margin,
            "requested_leverage": request.requested_leverage,
        }

        for name, value in fields.items():
            if value is None:
                continue
            if not math.isfinite(float(value)):
                return RiskCheckResult(
                    passed=False,
                    decision=RiskDecisionType.DENY,
                    reason=f"Invalid non-finite request field: {name}",
                    metadata={"field": name, "value": str(value)},
                )

        execution_cost = request.execution_cost
        if execution_cost is not None:
            cost_fields = {
                "execution_cost.spread_cost": execution_cost.spread_cost,
                "execution_cost.slippage_cost": execution_cost.slippage_cost,
                "execution_cost.fee_cost": execution_cost.fee_cost,
                "execution_cost.funding_cost": execution_cost.funding_cost,
                "execution_cost.other_cost": execution_cost.other_cost,
                "execution_cost.spread_pct": execution_cost.spread_pct,
                "execution_cost.slippage_pct": execution_cost.slippage_pct,
            }

            for name, value in cost_fields.items():
                if value is None:
                    continue
                if not math.isfinite(float(value)):
                    return RiskCheckResult(
                        passed=False,
                        decision=RiskDecisionType.DENY,
                        reason=f"Invalid non-finite request field: {name}",
                        metadata={"field": name, "value": str(value)},
                    )

        return RiskCheckResult(
            passed=True,
            decision=RiskDecisionType.ALLOW,
            metadata={"validated": True},
        )


__all__ = ["RiskManager"]