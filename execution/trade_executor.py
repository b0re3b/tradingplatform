from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from core.event_bus import Event, EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler
from execution.config import TradeExecutorConfig
from execution.enums import ExecutionStatus
from execution.exceptions import (
    ExecutionError,
    ExecutionRejectedError,
    KillSwitchActiveError,
)
from execution.models import (
    ExecutionIntent,
    ExecutionPlan,
    ExecutionStats,
    OrderRequest,
    OrderResult,
    PositionState,
)
from execution.smart_execution import SmartExecution
from execution.utils import (
    merge_metadata,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    now_ms,
    now_ts,
    safe_float,
)
from risk.enums import MarginMode, OrderIntent, PositionSide, RiskMode, TradeTier
from risk.models import RiskDecision


class OrderManagerProtocol(Protocol):
    async def submit_order(self, request: OrderRequest) -> OrderResult:
        ...

    async def cancel_all_orders(
        self,
        *,
        symbol: str,
        exchange: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        ...


class PositionManagerProtocol(Protocol):
    def get_position(
        self,
        *,
        symbol: str,
        side: PositionSide | str | None = None,
    ) -> PositionState | None:
        ...

    def list_positions(
        self,
        *,
        symbol: str | None = None,
        include_closed: bool = False,
    ) -> list[PositionState]:
        ...


class SLTPManagerProtocol(Protocol):
    async def cancel_protective_orders(
        self,
        *,
        symbol: str,
        position_side: PositionSide | str | None = None,
        position_id: str | None = None,
        sltp_type: Any | None = None,
        reason: str | None = None,
    ) -> list[Any]:
        ...


MarketContextProvider = Callable[[ExecutionIntent], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]


class TradeExecutor:
    """
    Final execution orchestrator.

    Responsibilities:
    - listen to risk-approved signal.confirmed events;
    - convert RiskDecision / signal.confirmed payload into ExecutionIntent;
    - coordinate SmartExecution -> ExecutionPlan -> OrderManager.submit_order();
    - handle risk.position_close_requested;
    - handle risk.position_reduce_requested;
    - handle risk.kill_switch;
    - emit execution lifecycle events;
    - never listen to signal.generated;
    - never duplicate risk sizing, guards, budgets or exposure logic.

    Important architecture contract:
    Strategy emits signal.generated.
    RiskManager consumes signal.generated and emits signal.confirmed.
    TradeExecutor consumes only signal.confirmed for new/increase entries.
    """

    def __init__(
        self,
        config: TradeExecutorConfig,
        *,
        order_manager: OrderManagerProtocol,
        position_manager: PositionManagerProtocol | None = None,
        sltp_manager: SLTPManagerProtocol | None = None,
        smart_execution: SmartExecution,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        market_context_provider: MarketContextProvider | None = None,
        service_name: str = "execution.trade_executor",
        auto_subscribe: bool | None = None,
        register_scheduler_jobs: bool | None = None,
    ) -> None:
        self._config = config
        self._config.validate()

        self._order_manager = order_manager
        self._position_manager = position_manager
        self._sltp_manager = sltp_manager
        self._smart_execution = smart_execution

        self._event_bus = event_bus
        self._scheduler = scheduler
        self._market_context_provider = market_context_provider

        self._service_name = service_name
        self._auto_subscribe = (
            self._config.auto_subscribe if auto_subscribe is None else auto_subscribe
        )
        self._register_scheduler_jobs = (
            self._config.register_scheduler_jobs
            if register_scheduler_jobs is None
            else register_scheduler_jobs
        )

        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="trade_executor",
        )

        self._lock = asyncio.Lock()
        self._execution_semaphore = asyncio.Semaphore(self._config.max_concurrent_executions)
        self._symbol_locks: dict[str, asyncio.Lock] = {}

        self._subscriptions: list[Any] = []
        self._scheduler_jobs: list[Any] = []

        self._active_executions: dict[str, ExecutionIntent] = {}
        self._execution_status: dict[str, ExecutionStatus] = {}

        self._stats = ExecutionStats()

        self._running = False
        self._started_at: float | None = None
        self._kill_switch_active = False
        self._kill_switch_reason: str | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    @property
    def stats(self) -> ExecutionStats:
        return self._stats

    async def start(self) -> None:
        if self._running:
            self._logger.warning("TradeExecutor already started")
            return

        self._running = True
        self._started_at = now_ts()

        if self._event_bus is not None and self._auto_subscribe:
            self.register()

        if self._scheduler is not None and self._register_scheduler_jobs:
            self.register_scheduler_jobs()

        await self._emit_event(
            "execution.trade_executor.started",
            {
                "service": self._service_name,
                "started_at": self._started_at,
                "auto_subscribe": self._auto_subscribe,
                "scheduler_jobs": len(self._scheduler_jobs),
            },
            priority=EventPriority.LOW,
        )

        self._logger.info(
            "TradeExecutor started | subscriptions=%s scheduler_jobs=%s",
            len(self._subscriptions),
            len(self._scheduler_jobs),
        )

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("TradeExecutor already stopped")
            return

        self._cancel_scheduler_jobs()
        self.unregister()

        await self._emit_event(
            "execution.trade_executor.stopped",
            {
                "service": self._service_name,
                "stopped_at": now_ts(),
            },
            priority=EventPriority.LOW,
        )

        self._running = False
        self._logger.info("TradeExecutor stopped")

    def _cancel_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            self._scheduler_jobs.clear()
            return
        for job_id in list(self._scheduler_jobs):
            try:
                self._scheduler.remove_job(job_id)
            except Exception:
                pass
        self._scheduler_jobs.clear()

    def register(self) -> None:
        """
        Register EventBus subscriptions.

        TradeExecutor intentionally does not subscribe to:
        - signal.generated
        - analytics.*
        - strategy.*
        """
        if self._event_bus is None:
            self._logger.warning("Cannot register TradeExecutor: event_bus is not configured")
            return

        if self._subscriptions:
            self._logger.warning("TradeExecutor subscriptions already registered")
            return

        self._subscriptions.extend(
            [
                self._event_bus.subscribe(
                    "signal.confirmed",
                    self._handle_signal_confirmed,
                    name="execution_trade_executor_on_signal_confirmed",
                ),
                self._event_bus.subscribe(
                    "risk.position_close_requested",
                    self._handle_position_close_requested,
                    name="execution_trade_executor_on_position_close_requested",
                ),
                self._event_bus.subscribe(
                    "risk.position_reduce_requested",
                    self._handle_position_reduce_requested,
                    name="execution_trade_executor_on_position_reduce_requested",
                ),
                self._event_bus.subscribe(
                    "risk.kill_switch",
                    self._handle_kill_switch,
                    name="execution_trade_executor_on_kill_switch",
                ),
                self._event_bus.subscribe(
                    "risk.manual_resume",
                    self._handle_manual_resume,
                    name="execution_trade_executor_on_manual_resume",
                ),
                self._event_bus.subscribe(
                    "execution.order_filled",
                    self._handle_execution_order_filled,
                    name="execution_trade_executor_on_order_filled",
                ),
                self._event_bus.subscribe(
                    "execution.order_failed",
                    self._handle_execution_order_failed,
                    name="execution_trade_executor_on_order_failed",
                ),
                self._event_bus.subscribe(
                    "execution.order_rejected",
                    self._handle_execution_order_rejected,
                    name="execution_trade_executor_on_order_rejected",
                ),
                self._event_bus.subscribe(
                    "execution.order_cancelled",
                    self._handle_execution_order_cancelled,
                    name="execution_trade_executor_on_order_cancelled",
                ),
            ]
        )

        self._logger.info(
            "TradeExecutor subscriptions registered | count=%s",
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
                self._logger.exception("Failed to unsubscribe TradeExecutor subscription")

        count = len(self._subscriptions)
        self._subscriptions.clear()

        self._logger.info(
            "TradeExecutor subscriptions unregistered | count=%s",
            count,
        )

    def register_scheduler_jobs(self) -> None:
        """
        Register defensive cleanup through Scheduler.

        No unmanaged asyncio loops are used.
        """
        if self._scheduler is None:
            return

        try:
            job = self._scheduler.add_interval_job(
                name="execution.trade_executor.cleanup_stale_executions",
                func=self.cleanup_stale_executions,
                interval=max(5.0, self._config.execution_timeout_seconds),
                run_immediately=False,
            )
            self._scheduler_jobs.append(job)
        except Exception:
            self._logger.exception("Failed to register TradeExecutor scheduler job")

    async def execute_intent(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None = None,
    ) -> ExecutionPlan:
        """
        Execute one risk-approved intent.

        Flow:
        ExecutionIntent -> SmartExecution.build_execution_plan()
        -> OrderManager.submit_order(...) for each leg.
        """
        intent.validate()

        if not self._config.enabled:
            raise ExecutionRejectedError("TradeExecutor is disabled")

        self._validate_execution_allowed(intent)

        symbol_lock = self._get_symbol_lock(intent.symbol)

        async with self._execution_semaphore:
            if self._config.per_symbol_execution_lock:
                async with symbol_lock:
                    return await self._execute_intent_locked(intent, market_context=market_context)

            return await self._execute_intent_locked(intent, market_context=market_context)

    async def execute_plan(self, plan: ExecutionPlan) -> list[OrderResult]:
        """
        Submit every order request from an already built plan.

        This method does not rebuild or re-approve the plan.
        """
        plan.validate()

        results: list[OrderResult] = []

        for request in plan.to_order_requests():
            result = await self._order_manager.submit_order(request)
            results.append(result)

        return results

    async def close_position(
        self,
        *,
        symbol: str,
        side: PositionSide | str,
        size: float | None = None,
        position_id: str | None = None,
        signal_id: str | None = None,
        strategy_name: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionPlan:
        """
        Close an existing position through reduce-only execution.

        RiskManager should normally request this through
        risk.position_close_requested.
        """
        side_n = self._position_side_from_raw(side)
        if side_n is None:
            raise ExecutionRejectedError("position side is required for close_position")

        position = (
            self._position_manager.get_position(symbol=symbol, side=side_n)
            if self._position_manager is not None
            else None
        )

        resolved_size = size
        if resolved_size is None and position is not None:
            resolved_size = position.size

        if resolved_size is None or resolved_size <= 0:
            raise ExecutionRejectedError("close_position requires positive size")

        intent = ExecutionIntent(
            exchange=self._config.default_exchange,
            market_type=self._config.default_market_type,
            symbol=symbol,
            side=side_n,
            order_intent=OrderIntent.CLOSE,
            final_size=resolved_size,
            final_leverage=position.leverage if position and position.leverage else 1.0,
            final_tier=position.tier if position else None,
            final_risk_amount=position.risk_amount if position else 0.0,
            final_margin=position.margin_used if position else 0.0,
            final_notional=position.notional_value if position else 0.0,
            entry_price=position.entry_price if position else None,
            stop_loss=position.stop_loss if position else None,
            take_profit=position.take_profit if position else None,
            signal_id=signal_id or (position.signal_id if position else None),
            strategy_name=strategy_name or (position.strategy_name if position else None),
            reservation_id=position.reservation_id if position else None,
            risk_mode=RiskMode.REDUCE_ONLY,
            margin_mode=MarginMode.ISOLATED,
            reduce_only=True,
            close_position=False,
            metadata=merge_metadata(
                metadata,
                {
                    "manual_reason": reason,
                    "position_id": position_id or (position.position_id if position else None),
                    "source": "close_position",
                },
            ),
        )

        return await self.execute_intent(intent)

    async def reduce_position(
        self,
        *,
        symbol: str,
        side: PositionSide | str,
        reduce_size: float,
        position_id: str | None = None,
        signal_id: str | None = None,
        strategy_name: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionPlan:
        """
        Reduce an existing position through reduce-only execution.
        """
        side_n = self._position_side_from_raw(side)
        if side_n is None:
            raise ExecutionRejectedError("position side is required for reduce_position")

        if reduce_size <= 0:
            raise ExecutionRejectedError("reduce_size must be > 0")

        position = (
            self._position_manager.get_position(symbol=symbol, side=side_n)
            if self._position_manager is not None
            else None
        )

        if position is not None and position.size > 0:
            reduce_size = min(reduce_size, position.size)

        intent = ExecutionIntent(
            exchange=self._config.default_exchange,
            market_type=self._config.default_market_type,
            symbol=symbol,
            side=side_n,
            order_intent=OrderIntent.REDUCE,
            final_size=reduce_size,
            final_leverage=position.leverage if position and position.leverage else 1.0,
            final_tier=position.tier if position else None,
            final_risk_amount=position.risk_amount if position else 0.0,
            final_margin=position.margin_used if position else 0.0,
            final_notional=position.notional_value if position else 0.0,
            entry_price=position.entry_price if position else None,
            stop_loss=position.stop_loss if position else None,
            take_profit=position.take_profit if position else None,
            signal_id=signal_id or (position.signal_id if position else None),
            strategy_name=strategy_name or (position.strategy_name if position else None),
            reservation_id=position.reservation_id if position else None,
            risk_mode=RiskMode.REDUCE_ONLY,
            margin_mode=MarginMode.ISOLATED,
            reduce_only=True,
            close_position=False,
            metadata=merge_metadata(
                metadata,
                {
                    "manual_reason": reason,
                    "position_id": position_id or (position.position_id if position else None),
                    "source": "reduce_position",
                },
            ),
        )

        return await self.execute_intent(intent)

    async def handle_kill_switch(
        self,
        *,
        reason: str | None = None,
        cancel_open_orders: bool = True,
    ) -> None:
        """
        Activate execution kill switch.

        TradeExecutor blocks new risk-increasing entries. It may cancel open
        orders, but actual position closing should be requested explicitly by
        RiskManager through risk.position_close_requested.
        """
        self._kill_switch_active = True
        self._kill_switch_reason = reason or "risk.kill_switch"

        if cancel_open_orders and self._config.kill_switch_cancels_open_orders:
            await self._cancel_open_orders_for_known_symbols(reason=self._kill_switch_reason)

        await self._emit_event(
            "execution.kill_switch_handled",
            {
                "exchange": self._config.default_exchange,
                "market_type": self._config.default_market_type,
                "reason": self._kill_switch_reason,
                "cancel_open_orders": cancel_open_orders,
                "timestamp": now_ms(),
            },
            priority=EventPriority.CRITICAL,
        )

    async def cleanup_stale_executions(self) -> None:
        """
        Defensive cleanup of stale active execution metadata.
        """
        now = now_ts()
        stale_ids: list[str] = []

        async with self._lock:
            for execution_id, intent in self._active_executions.items():
                age = now - intent.created_at
                if age >= self._config.execution_timeout_seconds:
                    stale_ids.append(execution_id)

            for execution_id in stale_ids:
                self._active_executions.pop(execution_id, None)
                self._execution_status[execution_id] = ExecutionStatus.EXPIRED

        for execution_id in stale_ids:
            await self._emit_event(
                "execution.execution_expired",
                {
                    "execution_id": execution_id,
                    "timestamp": now_ms(),
                },
                priority=EventPriority.HIGH,
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "service": self._service_name,
            "running": self._running,
            "started_at": self._started_at,
            "kill_switch_active": self._kill_switch_active,
            "kill_switch_reason": self._kill_switch_reason,
            "active_executions": len(self._active_executions),
            "execution_status": {
                execution_id: status.value
                for execution_id, status in self._execution_status.items()
            },
            "stats": self._stats.snapshot(),
        }

    # ------------------------------------------------------------------
    # Internal execution flow
    # ------------------------------------------------------------------

    async def _execute_intent_locked(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None,
    ) -> ExecutionPlan:
        async with self._lock:
            self._active_executions[intent.execution_id] = intent
            self._execution_status[intent.execution_id] = ExecutionStatus.ACCEPTED
            self._stats.register_started(intent.execution_id)

        await self._emit_event(
            "execution.trade_accepted",
            intent.to_event_payload(),
            priority=EventPriority.HIGH,
        )

        try:
            resolved_market_context = (
                market_context
                if market_context is not None
                else await self._resolve_market_context(intent)
            )

            await self._emit_event(
                "execution.execution_started",
                intent.to_event_payload(),
                priority=EventPriority.HIGH,
            )

            plan = await self._smart_execution.build_execution_plan(
                intent,
                market_context=resolved_market_context,
            )

            async with self._lock:
                self._execution_status[intent.execution_id] = ExecutionStatus.PLANNED

            await self._emit_event(
                "execution.execution_plan_created",
                plan.to_event_payload(),
                priority=EventPriority.HIGH,
            )

            results = await self.execute_plan(plan)

            async with self._lock:
                self._execution_status[intent.execution_id] = ExecutionStatus.SUBMITTED

            await self._emit_event(
                "execution.execution_submitted",
                {
                    **plan.to_event_payload(),
                    "orders_count": len(results),
                    "orders": [result.to_event_payload() for result in results],
                },
                priority=EventPriority.HIGH,
            )

            if all(result.is_filled for result in results):
                async with self._lock:
                    self._execution_status[intent.execution_id] = ExecutionStatus.COMPLETED
                    self._active_executions.pop(intent.execution_id, None)
                    self._stats.register_completed(intent.execution_id)

                await self._emit_event(
                    "execution.execution_completed",
                    {
                        **plan.to_event_payload(),
                        "orders_count": len(results),
                        "orders": [result.to_event_payload() for result in results],
                    },
                    priority=EventPriority.CRITICAL,
                )
            else:
                async with self._lock:
                    self._execution_status[intent.execution_id] = ExecutionStatus.SUBMITTED

            return plan

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with self._lock:
                self._execution_status[intent.execution_id] = ExecutionStatus.FAILED
                self._active_executions.pop(intent.execution_id, None)
                self._stats.register_failed(str(exc), intent.execution_id)

            await self._emit_event(
                "execution.execution_failed",
                {
                    **intent.to_event_payload(),
                    "error": str(exc),
                    "failure_stage": "execute_intent",
                },
                priority=EventPriority.CRITICAL,
            )

            raise ExecutionError(f"Execution failed: {exc}") from exc

    def _validate_execution_allowed(self, intent: ExecutionIntent) -> None:
        if self._kill_switch_active:
            if intent.increases_risk and self._config.kill_switch_blocks_new_entries:
                self._stats.register_rejected(
                    kill_switch=True,
                    error=self._kill_switch_reason,
                )
                raise KillSwitchActiveError(
                    self._kill_switch_reason or "Kill switch is active"
                )

            if intent.reduces_risk and not self._config.kill_switch_allows_reduce_only:
                raise KillSwitchActiveError("Kill switch blocks reduce-only executions")

        if intent.increases_risk and not self._config.allow_new_entries:
            self._stats.register_rejected(error="new entries disabled")
            raise ExecutionRejectedError("New entries are disabled")

        if intent.order_intent is OrderIntent.REDUCE and not self._config.allow_position_reductions:
            self._stats.register_rejected(error="position reductions disabled")
            raise ExecutionRejectedError("Position reductions are disabled")

        if intent.order_intent is OrderIntent.CLOSE and not self._config.allow_position_closes:
            self._stats.register_rejected(error="position closes disabled")
            raise ExecutionRejectedError("Position closes are disabled")

        if (
            self._config.reject_expired_risk_reservations
            and intent.increases_risk
            and intent.reservation_expires_at is not None
        ):
            expires_at = intent.reservation_expires_at + self._config.reservation_grace_seconds
            if now_ts() >= expires_at:
                self._stats.register_rejected(error="risk reservation expired")
                raise ExecutionRejectedError("Risk reservation is expired")

    async def _resolve_market_context(self, intent: ExecutionIntent) -> Mapping[str, Any]:
        if self._market_context_provider is None:
            return {}

        result = self._market_context_provider(intent)

        if inspect.isawaitable(result):
            resolved = await result
        else:
            resolved = result

        return dict(resolved or {})

    def _get_symbol_lock(self, symbol: str) -> asyncio.Lock:
        symbol_n = normalize_symbol(symbol)

        lock = self._symbol_locks.get(symbol_n)
        if lock is None:
            lock = asyncio.Lock()
            self._symbol_locks[symbol_n] = lock

        return lock

    async def _cancel_open_orders_for_known_symbols(self, *, reason: str | None) -> None:
        symbols: set[str] = set()

        if self._position_manager is not None:
            for position in self._position_manager.list_positions(include_closed=False):
                symbols.add(position.symbol)

        for intent in self._active_executions.values():
            symbols.add(intent.symbol)

        for symbol in sorted(symbols):
            try:
                await self._order_manager.cancel_all_orders(
                    symbol=symbol,
                    exchange=self._config.default_exchange,
                    reason=reason or "kill_switch",
                )
            except Exception:
                self._logger.exception(
                    "Failed to cancel open orders during kill switch | symbol=%s",
                    symbol,
                )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_signal_confirmed(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._event_payload(event)

        try:
            intent = self._intent_from_signal_confirmed_payload(payload)

            await self._emit_event(
                "execution.trade_requested",
                intent.to_event_payload(),
                priority=EventPriority.HIGH,
            )

            await self.execute_intent(intent)

        except asyncio.CancelledError:
            raise
        except ExecutionRejectedError as exc:
            self._stats.register_rejected(error=str(exc))
            self._logger.warning(
                "signal.confirmed rejected before execution | signal_id=%s symbol=%s error=%s",
                payload.get("signal_id"),
                payload.get("symbol"),
                str(exc),
            )

            await self._emit_event(
                "execution.trade_rejected",
                {
                    **payload,
                    "error": str(exc),
                    "failure_stage": "signal_confirmed_validation",
                },
                priority=EventPriority.CRITICAL,
            )
        except ExecutionError as exc:
            # execute_intent() already emitted execution.execution_failed and
            # updated execution failure stats. Do not convert order/network
            # failures into trade_rejected, and do not spam a second traceback.
            self._logger.warning(
                "signal.confirmed execution failed | signal_id=%s symbol=%s error=%s",
                payload.get("signal_id"),
                payload.get("symbol"),
                str(exc),
            )
        except Exception as exc:
            self._stats.register_rejected(error=str(exc))
            self._logger.exception("Failed to handle signal.confirmed")

            await self._emit_event(
                "execution.trade_rejected",
                {
                    **payload,
                    "error": str(exc),
                    "failure_stage": "signal_confirmed",
                },
                priority=EventPriority.CRITICAL,
            )

    async def _handle_position_close_requested(
            self,
            event: Event | Mapping[str, Any],
    ) -> None:
        payload = self._event_payload(event)

        try:
            raw_symbol = payload.get("symbol")
            if raw_symbol is None:
                raise ExecutionError("position close request missing symbol")

            raw_side = payload.get("side") or payload.get("position_side")
            if raw_side is None:
                raise ExecutionError("position close request missing side")

            if not isinstance(raw_side, PositionSide | str):
                raise ExecutionError(
                    f"position close request side must be PositionSide | str, "
                    f"got {type(raw_side).__name__}"
                )

            await self.close_position(
                symbol=str(raw_symbol),
                side=raw_side,
                size=safe_float(payload.get("size") or payload.get("quantity")),
                position_id=payload.get("position_id"),
                signal_id=payload.get("signal_id"),
                strategy_name=payload.get("strategy_name"),
                reason=payload.get("reason"),
                metadata=payload.get("metadata"),
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failed(str(exc))
            self._logger.exception("Failed to handle risk.position_close_requested")

            await self._emit_event(
                "execution.execution_failed",
                {
                    **payload,
                    "error": str(exc),
                    "failure_stage": "risk_position_close_requested",
                },
                priority=EventPriority.CRITICAL,
            )

    async def _handle_position_reduce_requested(
            self,
            event: Event | Mapping[str, Any],
    ) -> None:
        payload = self._event_payload(event)

        try:
            raw_symbol = payload.get("symbol")
            if raw_symbol is None:
                raise ExecutionRejectedError("position reduce request missing symbol")

            raw_side = payload.get("side") or payload.get("position_side")
            if raw_side is None:
                raise ExecutionRejectedError("position reduce request missing side")

            if not isinstance(raw_side, PositionSide | str):
                raise ExecutionRejectedError(
                    f"position reduce request side must be PositionSide | str, "
                    f"got {type(raw_side).__name__}"
                )

            reduce_size = safe_float(
                payload.get("reduce_size")
                or payload.get("size")
                or payload.get("quantity")
            )
            if reduce_size is None:
                raise ExecutionRejectedError("reduce_size is required")

            await self.reduce_position(
                symbol=str(raw_symbol),
                side=raw_side,
                reduce_size=reduce_size,
                position_id=payload.get("position_id"),
                signal_id=payload.get("signal_id"),
                strategy_name=payload.get("strategy_name"),
                reason=payload.get("reason"),
                metadata=payload.get("metadata"),
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failed(str(exc))
            self._logger.exception("Failed to handle risk.position_reduce_requested")

            await self._emit_event(
                "execution.execution_failed",
                {
                    **payload,
                    "error": str(exc),
                    "failure_stage": "risk_position_reduce_requested",
                },
                priority=EventPriority.CRITICAL,
            )

    async def _handle_kill_switch(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._event_payload(event)

        await self.handle_kill_switch(
            reason=payload.get("reason") or payload.get("message"),
            cancel_open_orders=bool(payload.get("cancel_open_orders", True)),
        )

    async def _handle_manual_resume(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._event_payload(event)

        self._kill_switch_active = False
        self._kill_switch_reason = None

        await self._emit_event(
            "execution.kill_switch_released",
            {
                "exchange": self._config.default_exchange,
                "market_type": self._config.default_market_type,
                "reason": payload.get("reason") or "risk.manual_resume",
                "timestamp": now_ms(),
            },
            priority=EventPriority.HIGH,
        )

    async def _handle_execution_order_filled(
            self,
            event: Event | Mapping[str, Any],
    ) -> None:
        payload = self._event_payload(event)
        raw_execution_id = payload.get("execution_id")

        if not isinstance(raw_execution_id, str) or not raw_execution_id:
            return

        execution_id = raw_execution_id

        async with self._lock:
            if execution_id in self._active_executions:
                self._execution_status[execution_id] = ExecutionStatus.FILLED

    async def _handle_execution_order_failed(self, event: Event | Mapping[str, Any]) -> None:
        await self._mark_execution_terminal_from_order_event(
            event,
            status=ExecutionStatus.FAILED,
        )

    async def _handle_execution_order_rejected(self, event: Event | Mapping[str, Any]) -> None:
        await self._mark_execution_terminal_from_order_event(
            event,
            status=ExecutionStatus.REJECTED,
        )

    async def _handle_execution_order_cancelled(self, event: Event | Mapping[str, Any]) -> None:
        await self._mark_execution_terminal_from_order_event(
            event,
            status=ExecutionStatus.CANCELLED,
        )

    async def _mark_execution_terminal_from_order_event(
            self,
            event: Event | Mapping[str, Any],
            *,
            status: ExecutionStatus,
    ) -> None:
        payload = self._event_payload(event)
        raw_execution_id = payload.get("execution_id")

        if not isinstance(raw_execution_id, str) or not raw_execution_id:
            return

        execution_id = raw_execution_id

        async with self._lock:
            self._execution_status[execution_id] = status
            self._active_executions.pop(execution_id, None)

    # ------------------------------------------------------------------
    # Payload -> intent helpers
    # ------------------------------------------------------------------

    def _intent_from_signal_confirmed_payload(self, payload: Mapping[str, Any]) -> ExecutionIntent:
        risk_decision = payload.get("risk_decision") or payload.get("decision")

        if isinstance(risk_decision, RiskDecision):
            return ExecutionIntent.from_risk_decision(
                risk_decision,
                exchange=self._config.default_exchange,
                market_type=self._config.default_market_type,
                metadata=payload.get("metadata"),
            )

        # Some EventBus payloads may be the RiskDecision fields directly.
        return self._intent_from_mapping(payload)

    def _intent_from_mapping(self, payload: Mapping[str, Any]) -> ExecutionIntent:
        symbol = payload.get("symbol")
        if not symbol:
            raise ExecutionRejectedError("signal.confirmed payload missing symbol")

        side = self._position_side_from_raw(payload.get("side"))
        if side is None:
            raise ExecutionRejectedError("signal.confirmed payload missing side")

        order_intent = self._order_intent_from_raw(payload.get("order_intent") or OrderIntent.OPEN)

        final_size = safe_float(payload.get("final_size") or payload.get("size"))
        final_leverage = safe_float(payload.get("final_leverage") or payload.get("leverage"))
        final_risk_amount = safe_float(payload.get("final_risk_amount") or payload.get("risk_amount"), 0.0)
        final_margin = safe_float(payload.get("final_margin") or payload.get("margin"), 0.0)
        final_notional = safe_float(payload.get("final_notional") or payload.get("notional"), 0.0)

        if final_size is None:
            raise ExecutionRejectedError("signal.confirmed payload missing final_size")

        if final_leverage is None:
            raise ExecutionRejectedError("signal.confirmed payload missing final_leverage")

        tier = self._trade_tier_from_raw(payload.get("final_tier") or payload.get("tier"))
        risk_mode = self._risk_mode_from_raw(payload.get("risk_mode") or RiskMode.NORMAL)
        margin_mode = self._margin_mode_from_raw(payload.get("margin_mode") or MarginMode.ISOLATED)

        intent = ExecutionIntent(
            exchange=normalize_exchange(payload.get("exchange") or self._config.default_exchange),
            market_type=normalize_market_type(payload.get("market_type") or self._config.default_market_type),
            symbol=normalize_symbol(str(symbol)),
            side=side,
            order_intent=order_intent,
            final_size=final_size,
            final_leverage=final_leverage,
            final_tier=tier,
            final_risk_amount=final_risk_amount or 0.0,
            final_margin=final_margin or 0.0,
            final_notional=final_notional or 0.0,
            entry_price=safe_float(payload.get("entry_price")),
            stop_loss=safe_float(payload.get("stop_loss")),
            take_profit=safe_float(payload.get("take_profit")),
            signal_id=payload.get("signal_id"),
            strategy_name=payload.get("strategy_name"),
            reservation_id=payload.get("reservation_id"),
            reservation_expires_at=safe_float(payload.get("reservation_expires_at")),
            risk_mode=risk_mode,
            margin_mode=margin_mode,
            reduce_only=bool(payload.get("reduce_only", False) or order_intent.reduces_risk),
            close_position=bool(payload.get("close_position", False)),
            metadata=merge_metadata(
                payload.get("metadata"),
                {
                    "source_event": "signal.confirmed",
                    "raw_decision": payload.get("decision"),
                },
            ),
        )
        intent.validate()
        return intent

    @staticmethod
    def _position_side_from_raw(value: Any) -> PositionSide | None:
        if value is None:
            return None

        if isinstance(value, PositionSide):
            return value

        normalized = str(value).strip().lower()

        if normalized in {"long", "buy"}:
            return PositionSide.LONG

        if normalized in {"short", "sell"}:
            return PositionSide.SHORT

        if normalized.upper() == "LONG":
            return PositionSide.LONG

        if normalized.upper() == "SHORT":
            return PositionSide.SHORT

        return None

    @staticmethod
    def _order_intent_from_raw(value: Any) -> OrderIntent:
        if isinstance(value, OrderIntent):
            return value

        normalized = str(value).strip().lower()

        for item in OrderIntent:
            if item.value == normalized or item.name.lower() == normalized:
                return item

        raise ExecutionRejectedError(f"Unsupported order_intent: {value!r}")

    @staticmethod
    def _trade_tier_from_raw(value: Any) -> TradeTier | None:
        if value is None:
            return None

        if isinstance(value, TradeTier):
            return value

        normalized = str(value).strip()

        for item in TradeTier:
            if item.value == normalized or item.name == normalized:
                return item

        return None

    @staticmethod
    def _risk_mode_from_raw(value: Any) -> RiskMode:
        if isinstance(value, RiskMode):
            return value

        normalized = str(value).strip().lower()

        for item in RiskMode:
            if item.value == normalized or item.name.lower() == normalized:
                return item

        return RiskMode.NORMAL

    @staticmethod
    def _margin_mode_from_raw(value: Any) -> MarginMode:
        if isinstance(value, MarginMode):
            return value

        normalized = str(value).strip().lower()

        for item in MarginMode:
            if item.value.lower() == normalized or item.name.lower() == normalized:
                return item

        return MarginMode.ISOLATED

    @staticmethod
    def _event_payload(event: Event | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(event, Mapping):
            return dict(event)

        payload = getattr(event, "payload", None)

        if isinstance(payload, Mapping):
            return dict(payload)

        return {}

    async def _emit_event(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        if self._event_bus is None:
            return

        try:
            maybe_result = self._event_bus.emit(
                topic,
                payload,
                priority=priority,
                source=self._service_name,
            )

            if inspect.isawaitable(maybe_result):
                await maybe_result

        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception(
                "Failed to emit TradeExecutor event | topic=%s",
                topic,
            )


__all__ = [
    "OrderManagerProtocol",
    "PositionManagerProtocol",
    "SLTPManagerProtocol",
    "MarketContextProvider",
    "TradeExecutor",
]