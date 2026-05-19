from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from typing import Any, Protocol

from core.event_bus import Event, EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler
from execution.config import SLTPManagerConfig
from execution.enums import OrderStatus, OrderType, SLTPType, TriggerType
from execution.exceptions import ProtectiveOrderError, SLTPError
from execution.models import (
    OrderRequest,
    OrderResult,
    ProtectiveOrderState,
    SLTPManagerStats,
    SLTPPlan,
)
from execution.utils import (
    base_execution_payload,
    build_client_order_id,
    merge_metadata,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    now_ms,
    now_ts,
    order_side_for_position_close,
    safe_float,
)
from risk.enums import PositionSide


class OrderManagerProtocol(Protocol):
    """
    Minimal OrderManager surface required by SLTPManager.

    Concrete implementation:
    execution.order_manager.OrderManager
    """

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        ...

    async def cancel_order(
        self,
        *,
        symbol: str,
        order_id: str | int | None = None,
        client_order_id: str | None = None,
        exchange: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> OrderResult:
        ...

    async def cancel_all_orders(
        self,
        *,
        symbol: str,
        exchange: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        ...


class SLTPManager:
    """
    Stop-loss / take-profit / trailing-stop manager.

    Responsibilities:
    - listen to position.opened / position.updated / position.closed;
    - place protective reduce-only / close-position orders through OrderManager;
    - react to risk.stop_update_requested / risk.take_profit_update_requested /
      risk.trailing_stop_requested / risk.position_close_requested;
    - keep protective order state consistent with position lifecycle;
    - never call BinanceRestClient directly;
    - never perform risk approval or position sizing.

    Important:
    - Binance USD-M Futures protective orders must be reduce-only or
      close-position according to order type and intent.
    - RiskManager/TradeExecutor decide whether a position should be closed.
      SLTPManager only maintains protective orders.
    """

    def __init__(
        self,
        config: SLTPManagerConfig,
        *,
        order_manager: OrderManagerProtocol,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        service_name: str = "execution.sltp_manager",
        auto_subscribe: bool = True,
        register_scheduler_jobs: bool = True,
    ) -> None:
        self._config = config
        self._config.validate()

        self._order_manager = order_manager
        self._event_bus = event_bus
        self._scheduler = scheduler

        self._service_name = service_name
        self._auto_subscribe = auto_subscribe
        self._register_scheduler_jobs = register_scheduler_jobs

        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="sltp_manager",
        )

        self._lock = asyncio.Lock()
        self._subscriptions: list[Any] = []
        self._scheduler_jobs: list[Any] = []

        # position_id -> protective order states
        self._protective_orders: dict[str, list[ProtectiveOrderState]] = {}

        self._stats = SLTPManagerStats()
        self._running = False
        self._started_at: float | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> SLTPManagerStats:
        return self._stats

    async def start(self) -> None:
        if self._running:
            self._logger.warning("SLTPManager already started")
            return

        self._running = True
        self._started_at = now_ts()

        if self._event_bus is not None and self._auto_subscribe:
            self.register()

        if self._scheduler is not None and self._register_scheduler_jobs:
            self.register_scheduler_jobs()

        await self._emit_event(
            "execution.sltp_manager.started",
            {
                "service": self._service_name,
                "started_at": self._started_at,
                "auto_subscribe": self._auto_subscribe,
                "scheduler_jobs": len(self._scheduler_jobs),
            },
            priority=EventPriority.LOW,
        )

        self._logger.info(
            "SLTPManager started | subscriptions=%s scheduler_jobs=%s",
            len(self._subscriptions),
            len(self._scheduler_jobs),
        )

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("SLTPManager already stopped")
            return

        self.unregister()

        await self._emit_event(
            "execution.sltp_manager.stopped",
            {
                "service": self._service_name,
                "stopped_at": now_ts(),
            },
            priority=EventPriority.LOW,
        )

        self._running = False
        self._logger.info("SLTPManager stopped")

    def register(self) -> None:
        if self._event_bus is None:
            self._logger.warning("Cannot register SLTPManager: event_bus is not configured")
            return

        if self._subscriptions:
            self._logger.warning("SLTPManager subscriptions already registered")
            return

        self._subscriptions.extend(
            [
                self._event_bus.subscribe(
                    "position.opened",
                    self._handle_position_opened,
                    name="execution_sltp_manager_on_position_opened",
                ),
                self._event_bus.subscribe(
                    "position.updated",
                    self._handle_position_updated,
                    name="execution_sltp_manager_on_position_updated",
                ),
                self._event_bus.subscribe(
                    "position.closed",
                    self._handle_position_closed,
                    name="execution_sltp_manager_on_position_closed",
                ),
                self._event_bus.subscribe(
                    "risk.stop_update_requested",
                    self._handle_stop_update_requested,
                    name="execution_sltp_manager_on_stop_update_requested",
                ),
                self._event_bus.subscribe(
                    "risk.take_profit_update_requested",
                    self._handle_take_profit_update_requested,
                    name="execution_sltp_manager_on_take_profit_update_requested",
                ),
                self._event_bus.subscribe(
                    "risk.trailing_stop_requested",
                    self._handle_trailing_stop_requested,
                    name="execution_sltp_manager_on_trailing_stop_requested",
                ),
                self._event_bus.subscribe(
                    "risk.position_close_requested",
                    self._handle_position_close_requested,
                    name="execution_sltp_manager_on_position_close_requested",
                ),
                self._event_bus.subscribe(
                    "execution.order_filled",
                    self._handle_execution_order_filled,
                    name="execution_sltp_manager_on_order_filled",
                ),
                self._event_bus.subscribe(
                    "execution.order_cancelled",
                    self._handle_execution_order_cancelled,
                    name="execution_sltp_manager_on_order_cancelled",
                ),
                self._event_bus.subscribe(
                    "execution.order_rejected",
                    self._handle_execution_order_rejected,
                    name="execution_sltp_manager_on_order_rejected",
                ),
                self._event_bus.subscribe(
                    "execution.order_failed",
                    self._handle_execution_order_failed,
                    name="execution_sltp_manager_on_order_failed",
                ),
            ]
        )

        self._logger.info(
            "SLTPManager subscriptions registered | count=%s",
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
                self._logger.exception("Failed to unsubscribe SLTPManager subscription")

        count = len(self._subscriptions)
        self._subscriptions.clear()

        self._logger.info(
            "SLTPManager subscriptions unregistered | count=%s",
            count,
        )

    def register_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            return

        if not self._config.reconcile_enabled:
            return

        try:
            job = self._scheduler.add_interval_job(
                self.reconcile_protective_orders,
                interval_seconds=self._config.reconcile_interval_seconds,
                name="execution.sltp_manager.reconcile_protective_orders",
                run_immediately=False,
            )
            self._scheduler_jobs.append(job)
        except Exception:
            self._logger.exception("Failed to register SLTPManager scheduler job")

    async def place_protective_orders(self, plan: SLTPPlan) -> list[ProtectiveOrderState]:
        """
        Place SL/TP/trailing orders for one position via OrderManager.
        """
        if not self._config.enabled:
            return []

        plan.validate()

        created: list[ProtectiveOrderState] = []

        if plan.stop_loss is not None:
            created.append(await self._place_stop_loss(plan))

        if plan.take_profit is not None:
            created.append(await self._place_take_profit(plan))

        if plan.trailing_callback_rate is not None:
            created.append(await self._place_trailing_stop(plan))

        async with self._lock:
            key = self._position_key(plan.position_id, plan.symbol, plan.position_side)
            existing = self._protective_orders.setdefault(key, [])
            existing.extend(created)

        await self._emit_event(
            "execution.sltp.place_completed",
            {
                **plan.to_event_payload(),
                "protective_orders_count": len(created),
                "protective_orders": [state.snapshot() for state in created],
            },
            priority=EventPriority.HIGH,
        )

        return created

    async def update_stop_loss(
        self,
        *,
        symbol: str,
        position_side: PositionSide,
        stop_loss: float,
        size: float,
        position_id: str | None = None,
        signal_id: str | None = None,
        strategy_name: str | None = None,
        reservation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProtectiveOrderState:
        """
        Replace existing stop-loss protective order with a new one.
        """
        await self.cancel_protective_orders(
            symbol=symbol,
            position_side=position_side,
            position_id=position_id,
            sltp_type=SLTPType.STOP_LOSS,
            reason="update_stop_loss",
        )

        plan = SLTPPlan(
            exchange=self._config.default_exchange,
            market_type=self._config.default_market_type,
            symbol=symbol,
            position_side=position_side,
            size=size,
            stop_loss=stop_loss,
            working_type=self._config.default_working_type,
            price_protect=self._config.price_protect,
            use_close_position_for_stop=self._config.use_close_position_for_full_stop,
            signal_id=signal_id,
            strategy_name=strategy_name,
            reservation_id=reservation_id,
            position_id=position_id,
            metadata=merge_metadata(metadata, {"update_reason": "stop_loss_update"}),
        )

        state = await self._place_stop_loss(plan)

        async with self._lock:
            key = self._position_key(position_id, symbol, position_side)
            self._protective_orders.setdefault(key, []).append(state)

        await self._emit_event(
            "execution.sltp.update_completed",
            {
                **plan.to_event_payload(),
                "sltp_type": SLTPType.STOP_LOSS.value,
                "protective_order": state.snapshot(),
            },
            priority=EventPriority.HIGH,
        )

        return state

    async def update_take_profit(
        self,
        *,
        symbol: str,
        position_side: PositionSide,
        take_profit: float,
        size: float,
        position_id: str | None = None,
        signal_id: str | None = None,
        strategy_name: str | None = None,
        reservation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProtectiveOrderState:
        """
        Replace existing take-profit protective order with a new one.
        """
        await self.cancel_protective_orders(
            symbol=symbol,
            position_side=position_side,
            position_id=position_id,
            sltp_type=SLTPType.TAKE_PROFIT,
            reason="update_take_profit",
        )

        plan = SLTPPlan(
            exchange=self._config.default_exchange,
            market_type=self._config.default_market_type,
            symbol=symbol,
            position_side=position_side,
            size=size,
            take_profit=take_profit,
            working_type=self._config.default_working_type,
            price_protect=self._config.price_protect,
            use_close_position_for_take_profit=self._config.use_close_position_for_full_take_profit,
            signal_id=signal_id,
            strategy_name=strategy_name,
            reservation_id=reservation_id,
            position_id=position_id,
            metadata=merge_metadata(metadata, {"update_reason": "take_profit_update"}),
        )

        state = await self._place_take_profit(plan)

        async with self._lock:
            key = self._position_key(position_id, symbol, position_side)
            self._protective_orders.setdefault(key, []).append(state)

        await self._emit_event(
            "execution.sltp.update_completed",
            {
                **plan.to_event_payload(),
                "sltp_type": SLTPType.TAKE_PROFIT.value,
                "protective_order": state.snapshot(),
            },
            priority=EventPriority.HIGH,
        )

        return state

    async def update_trailing_stop(
        self,
        *,
        symbol: str,
        position_side: PositionSide,
        callback_rate: float,
        size: float,
        activation_price: float | None = None,
        position_id: str | None = None,
        signal_id: str | None = None,
        strategy_name: str | None = None,
        reservation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProtectiveOrderState:
        """
        Replace existing trailing stop with a new trailing stop order.
        """
        if not self._config.trailing_stop_enabled:
            raise SLTPError("Trailing stop is disabled")

        callback_rate = self._clamp_callback_rate(callback_rate)

        await self.cancel_protective_orders(
            symbol=symbol,
            position_side=position_side,
            position_id=position_id,
            sltp_type=SLTPType.TRAILING_STOP,
            reason="update_trailing_stop",
        )

        plan = SLTPPlan(
            exchange=self._config.default_exchange,
            market_type=self._config.default_market_type,
            symbol=symbol,
            position_side=position_side,
            size=size,
            trailing_activation_price=activation_price,
            trailing_callback_rate=callback_rate,
            working_type=self._config.default_working_type,
            price_protect=self._config.price_protect,
            signal_id=signal_id,
            strategy_name=strategy_name,
            reservation_id=reservation_id,
            position_id=position_id,
            metadata=merge_metadata(metadata, {"update_reason": "trailing_stop_update"}),
        )

        state = await self._place_trailing_stop(plan)

        async with self._lock:
            key = self._position_key(position_id, symbol, position_side)
            self._protective_orders.setdefault(key, []).append(state)

        await self._emit_event(
            "execution.trailing_stop_updated",
            {
                **plan.to_event_payload(),
                "protective_order": state.snapshot(),
            },
            priority=EventPriority.HIGH,
        )

        return state

    async def cancel_protective_orders(
        self,
        *,
        symbol: str,
        position_side: PositionSide | str | None = None,
        position_id: str | None = None,
        sltp_type: SLTPType | None = None,
        reason: str | None = None,
    ) -> list[ProtectiveOrderState]:
        """
        Cancel tracked protective orders for a position/symbol.

        This does not use cancel_all_orders to avoid cancelling entry/exit orders
        owned by TradeExecutor/OrderManager.
        """
        symbol_n = normalize_symbol(symbol)
        side_n = self._position_side_from_raw(position_side)

        async with self._lock:
            states = self._matching_protective_orders(
                symbol=symbol_n,
                position_side=side_n,
                position_id=position_id,
                sltp_type=sltp_type,
                only_active=True,
            )

        cancelled: list[ProtectiveOrderState] = []

        for state in states:
            try:
                if not state.client_order_id and not state.order_id:
                    continue

                result = await self._order_manager.cancel_order(
                    symbol=state.symbol,
                    order_id=state.order_id,
                    client_order_id=state.client_order_id,
                    exchange=state.exchange,
                    reason=reason or "cancel_protective_order",
                    metadata={
                        "sltp_type": state.sltp_type.value,
                        "position_id": state.position_id,
                    },
                )

                state.apply_order_result(result)
                cancelled.append(state)
                self._stats.register_cancelled()

                await self._emit_event(
                    "execution.sltp.cancel_completed",
                    {
                        **base_execution_payload(
                            exchange=state.exchange,
                            market_type=state.market_type,
                            symbol=state.symbol,
                            metadata=state.metadata,
                        ),
                        "sltp_type": state.sltp_type.value,
                        "order_id": state.order_id,
                        "client_order_id": state.client_order_id,
                        "position_id": state.position_id,
                        "reason": reason,
                    },
                    priority=EventPriority.HIGH,
                )

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats.register_failure(str(exc))
                self._logger.exception(
                    "Failed to cancel protective order | symbol=%s order_id=%s client_order_id=%s",
                    state.symbol,
                    state.order_id,
                    state.client_order_id,
                )
                await self._emit_event(
                    "execution.sltp.cancel_failed",
                    {
                        **base_execution_payload(
                            exchange=state.exchange,
                            market_type=state.market_type,
                            symbol=state.symbol,
                            metadata=state.metadata,
                        ),
                        "sltp_type": state.sltp_type.value,
                        "order_id": state.order_id,
                        "client_order_id": state.client_order_id,
                        "position_id": state.position_id,
                        "error": str(exc),
                    },
                    priority=EventPriority.HIGH,
                )

        return cancelled

    async def reconcile_protective_orders(self) -> None:
        """
        Lightweight protective-order reconciliation.

        Compact package rule: keep this inside sl_tp_manager.py.
        Full exchange reconciliation remains OrderManager responsibility.
        """
        async with self._lock:
            active_count = sum(
                1
                for states in self._protective_orders.values()
                for state in states
                if state.is_active
            )

        await self._emit_event(
            "execution.sltp.reconciled",
            {
                "exchange": self._config.default_exchange,
                "market_type": self._config.default_market_type,
                "active_protective_orders": active_count,
                "timestamp": now_ms(),
            },
            priority=EventPriority.LOW,
        )

    def list_protective_orders(
        self,
        *,
        symbol: str | None = None,
        position_id: str | None = None,
        include_terminal: bool = True,
    ) -> list[ProtectiveOrderState]:
        symbol_n = normalize_symbol(symbol) if symbol else None

        states: list[ProtectiveOrderState] = []

        for group in self._protective_orders.values():
            for state in group:
                if symbol_n is not None and state.symbol != symbol_n:
                    continue

                if position_id is not None and state.position_id != position_id:
                    continue

                if not include_terminal and state.is_terminal:
                    continue

                states.append(state)

        return states

    def snapshot(self) -> dict[str, Any]:
        return {
            "service": self._service_name,
            "running": self._running,
            "started_at": self._started_at,
            "positions_tracked": len(self._protective_orders),
            "protective_orders": [
                state.snapshot()
                for state in self.list_protective_orders(include_terminal=True)
            ],
            "stats": self._stats.snapshot(),
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_position_opened(self, event: Event | Mapping[str, Any]) -> None:
        if not self._config.auto_place_on_position_opened:
            return

        payload = self._event_payload(event)

        try:
            plan = self._plan_from_position_payload(payload)
            if plan is None:
                return

            await self.place_protective_orders(plan)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            self._logger.exception("Failed to place SL/TP on position.opened")
            await self._emit_event(
                "execution.sltp.place_failed",
                {
                    **payload,
                    "error": str(exc),
                    "reason": "position_opened",
                },
                priority=EventPriority.HIGH,
            )

    async def _handle_position_updated(self, event: Event | Mapping[str, Any]) -> None:
        if not self._config.auto_resize_on_position_updated:
            return

        payload = self._event_payload(event)

        try:
            size = safe_float(payload.get("size"), 0.0) or 0.0
            if size <= 0:
                return

            symbol = payload.get("symbol")
            side = self._position_side_from_raw(payload.get("side"))
            position_id = payload.get("position_id")

            if not symbol or side is None:
                return

            async with self._lock:
                states = self._matching_protective_orders(
                    symbol=normalize_symbol(str(symbol)),
                    position_side=side,
                    position_id=position_id,
                    sltp_type=None,
                    only_active=True,
                )

                for state in states:
                    state.quantity = size
                    state.updated_at = now_ts()
                    state.metadata["resized_from_position_update"] = True

            await self._emit_event(
                "execution.sltp.resized",
                {
                    **base_execution_payload(
                        exchange=payload.get("exchange") or self._config.default_exchange,
                        market_type=payload.get("market_type") or self._config.default_market_type,
                        symbol=str(symbol),
                        signal_id=payload.get("signal_id"),
                        strategy_name=payload.get("strategy_name"),
                        reservation_id=payload.get("reservation_id"),
                        metadata=payload.get("metadata"),
                    ),
                    "position_id": position_id,
                    "position_side": side.value,
                    "size": size,
                },
                priority=EventPriority.NORMAL,
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            self._logger.exception("Failed to handle position.updated in SLTPManager")

    async def _handle_position_closed(self, event: Event | Mapping[str, Any]) -> None:
        if not self._config.auto_cancel_on_position_closed:
            return

        payload = self._event_payload(event)

        try:
            symbol = str(payload["symbol"])
            side = self._position_side_from_raw(payload.get("side") or payload.get("previous_side"))

            await self.cancel_protective_orders(
                symbol=symbol,
                position_side=side,
                position_id=payload.get("position_id"),
                reason="position_closed",
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            self._logger.exception("Failed to cancel SL/TP on position.closed")

    async def _handle_stop_update_requested(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._event_payload(event)

        try:
            side = self._require_position_side(payload.get("side") or payload.get("position_side"))

            await self.update_stop_loss(
                symbol=str(payload["symbol"]),
                position_side=side,
                stop_loss=self._require_price(payload, "stop_loss"),
                size=self._require_size(payload),
                position_id=payload.get("position_id"),
                signal_id=payload.get("signal_id"),
                strategy_name=payload.get("strategy_name"),
                reservation_id=payload.get("reservation_id"),
                metadata=payload.get("metadata"),
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            self._logger.exception("Failed to handle risk.stop_update_requested")
            await self._emit_sltp_update_failed(payload, exc, update_type="stop_loss")

    async def _handle_take_profit_update_requested(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._event_payload(event)

        try:
            side = self._require_position_side(payload.get("side") or payload.get("position_side"))

            await self.update_take_profit(
                symbol=str(payload["symbol"]),
                position_side=side,
                take_profit=self._require_price(payload, "take_profit"),
                size=self._require_size(payload),
                position_id=payload.get("position_id"),
                signal_id=payload.get("signal_id"),
                strategy_name=payload.get("strategy_name"),
                reservation_id=payload.get("reservation_id"),
                metadata=payload.get("metadata"),
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            self._logger.exception("Failed to handle risk.take_profit_update_requested")
            await self._emit_sltp_update_failed(payload, exc, update_type="take_profit")

    async def _handle_trailing_stop_requested(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._event_payload(event)

        try:
            side = self._require_position_side(payload.get("side") or payload.get("position_side"))

            callback_rate = safe_float(payload.get("callback_rate"))
            if callback_rate is None:
                raise SLTPError("callback_rate is required")

            await self.update_trailing_stop(
                symbol=str(payload["symbol"]),
                position_side=side,
                callback_rate=callback_rate,
                activation_price=safe_float(payload.get("activation_price")),
                size=self._require_size(payload),
                position_id=payload.get("position_id"),
                signal_id=payload.get("signal_id"),
                strategy_name=payload.get("strategy_name"),
                reservation_id=payload.get("reservation_id"),
                metadata=payload.get("metadata"),
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            self._logger.exception("Failed to handle risk.trailing_stop_requested")
            await self._emit_sltp_update_failed(payload, exc, update_type="trailing_stop")

    async def _handle_position_close_requested(self, event: Event | Mapping[str, Any]) -> None:
        """
        Before explicit close flow, cancel take-profit/trailing orders to avoid
        conflicting exits. Stop-loss can be cancelled too if requested.
        """
        payload = self._event_payload(event)

        try:
            symbol = str(payload["symbol"])
            side = self._position_side_from_raw(payload.get("side") or payload.get("position_side"))

            await self.cancel_protective_orders(
                symbol=symbol,
                position_side=side,
                position_id=payload.get("position_id"),
                reason="risk_position_close_requested",
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            self._logger.exception("Failed to handle risk.position_close_requested in SLTPManager")

    async def _handle_execution_order_filled(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._event_payload(event)
        await self._apply_order_lifecycle_to_protective_state(payload, OrderStatus.FILLED)

    async def _handle_execution_order_cancelled(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._event_payload(event)
        await self._apply_order_lifecycle_to_protective_state(payload, OrderStatus.CANCELED)

    async def _handle_execution_order_rejected(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._event_payload(event)
        await self._apply_order_lifecycle_to_protective_state(payload, OrderStatus.REJECTED)

    async def _handle_execution_order_failed(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._event_payload(event)
        status_raw = payload.get("status")
        status = OrderStatus.from_raw(status_raw) if status_raw else OrderStatus.EXPIRED
        await self._apply_order_lifecycle_to_protective_state(payload, status)

    # ------------------------------------------------------------------
    # Placement helpers
    # ------------------------------------------------------------------

    async def _place_stop_loss(self, plan: SLTPPlan) -> ProtectiveOrderState:
        if plan.stop_loss is None:
            raise ProtectiveOrderError("stop_loss is required")

        order_side = order_side_for_position_close(plan.position_side)
        client_order_id = build_client_order_id(
            prefix="sl",
            signal_id=plan.signal_id,
            strategy_name=plan.strategy_name,
            symbol=plan.symbol,
            order_intent=SLTPType.STOP_LOSS.value,
        )

        close_position = bool(plan.use_close_position_for_stop)
        quantity = None if close_position else plan.size

        request = OrderRequest(
            execution_id=f"sltp:{plan.plan_id}",
            exchange=plan.exchange,
            market_type=plan.market_type,
            symbol=plan.symbol,
            side=order_side,
            order_type=OrderType.STOP_MARKET,
            quantity=quantity,
            position_side=plan.position_side,
            reduce_only=not close_position,
            close_position=close_position,
            client_order_id=client_order_id,
            stop_price=plan.stop_loss,
            working_type=plan.working_type,
            price_protect=plan.price_protect,
            trigger_type=TriggerType.STOP_LOSS,
            signal_id=plan.signal_id,
            strategy_name=plan.strategy_name,
            reservation_id=plan.reservation_id,
            metadata=merge_metadata(
                plan.metadata,
                {
                    "sltp_type": SLTPType.STOP_LOSS.value,
                    "position_id": plan.position_id,
                    "plan_id": plan.plan_id,
                },
            ),
        )

        result = await self._submit_protective_order(request)

        state = self._state_from_result(
            result=result,
            sltp_type=SLTPType.STOP_LOSS,
            position_id=plan.position_id,
            position_side=plan.position_side,
            stop_price=plan.stop_loss,
            quantity=quantity,
            close_position=close_position,
        )

        self._stats.register_placed(state)

        await self._emit_event(
            "execution.stop_loss_placed",
            {
                **plan.to_event_payload(),
                "protective_order": state.snapshot(),
            },
            priority=EventPriority.HIGH,
        )

        return state

    async def _place_take_profit(self, plan: SLTPPlan) -> ProtectiveOrderState:
        if plan.take_profit is None:
            raise ProtectiveOrderError("take_profit is required")

        order_side = order_side_for_position_close(plan.position_side)
        client_order_id = build_client_order_id(
            prefix="tp",
            signal_id=plan.signal_id,
            strategy_name=plan.strategy_name,
            symbol=plan.symbol,
            order_intent=SLTPType.TAKE_PROFIT.value,
        )

        close_position = bool(plan.use_close_position_for_take_profit)
        quantity = None if close_position else plan.size

        request = OrderRequest(
            execution_id=f"sltp:{plan.plan_id}",
            exchange=plan.exchange,
            market_type=plan.market_type,
            symbol=plan.symbol,
            side=order_side,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            quantity=quantity,
            position_side=plan.position_side,
            reduce_only=not close_position,
            close_position=close_position,
            client_order_id=client_order_id,
            stop_price=plan.take_profit,
            working_type=plan.working_type,
            price_protect=plan.price_protect,
            trigger_type=TriggerType.TAKE_PROFIT,
            signal_id=plan.signal_id,
            strategy_name=plan.strategy_name,
            reservation_id=plan.reservation_id,
            metadata=merge_metadata(
                plan.metadata,
                {
                    "sltp_type": SLTPType.TAKE_PROFIT.value,
                    "position_id": plan.position_id,
                    "plan_id": plan.plan_id,
                },
            ),
        )

        result = await self._submit_protective_order(request)

        state = self._state_from_result(
            result=result,
            sltp_type=SLTPType.TAKE_PROFIT,
            position_id=plan.position_id,
            position_side=plan.position_side,
            stop_price=plan.take_profit,
            quantity=quantity,
            close_position=close_position,
        )

        self._stats.register_placed(state)

        await self._emit_event(
            "execution.take_profit_placed",
            {
                **plan.to_event_payload(),
                "protective_order": state.snapshot(),
            },
            priority=EventPriority.HIGH,
        )

        return state

    async def _place_trailing_stop(self, plan: SLTPPlan) -> ProtectiveOrderState:
        if plan.trailing_callback_rate is None:
            raise ProtectiveOrderError("trailing_callback_rate is required")

        if not self._config.trailing_stop_enabled:
            raise ProtectiveOrderError("Trailing stop is disabled")

        order_side = order_side_for_position_close(plan.position_side)
        callback_rate = self._clamp_callback_rate(plan.trailing_callback_rate)

        client_order_id = build_client_order_id(
            prefix="ts",
            signal_id=plan.signal_id,
            strategy_name=plan.strategy_name,
            symbol=plan.symbol,
            order_intent=SLTPType.TRAILING_STOP.value,
        )

        request = OrderRequest(
            execution_id=f"sltp:{plan.plan_id}",
            exchange=plan.exchange,
            market_type=plan.market_type,
            symbol=plan.symbol,
            side=order_side,
            order_type=OrderType.TRAILING_STOP_MARKET,
            quantity=plan.size,
            position_side=plan.position_side,
            reduce_only=True,
            close_position=False,
            client_order_id=client_order_id,
            activation_price=plan.trailing_activation_price,
            callback_rate=callback_rate,
            working_type=plan.working_type,
            price_protect=plan.price_protect,
            trigger_type=TriggerType.TRAILING_STOP,
            signal_id=plan.signal_id,
            strategy_name=plan.strategy_name,
            reservation_id=plan.reservation_id,
            metadata=merge_metadata(
                plan.metadata,
                {
                    "sltp_type": SLTPType.TRAILING_STOP.value,
                    "position_id": plan.position_id,
                    "plan_id": plan.plan_id,
                },
            ),
        )

        result = await self._submit_protective_order(request)

        state = self._state_from_result(
            result=result,
            sltp_type=SLTPType.TRAILING_STOP,
            position_id=plan.position_id,
            position_side=plan.position_side,
            stop_price=plan.trailing_activation_price,
            quantity=plan.size,
            close_position=False,
        )

        self._stats.register_placed(state)

        await self._emit_event(
            "execution.trailing_stop_placed",
            {
                **plan.to_event_payload(),
                "protective_order": state.snapshot(),
            },
            priority=EventPriority.HIGH,
        )

        return state

    async def _submit_protective_order(self, request: OrderRequest) -> OrderResult:
        try:
            await self._emit_event(
                "execution.sltp.place_requested",
                request.to_event_payload(),
                priority=EventPriority.HIGH,
            )

            return await self._order_manager.submit_order(request)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            raise ProtectiveOrderError(f"Failed to submit protective order: {exc}") from exc

    @staticmethod
    def _state_from_result(
        *,
        result: OrderResult,
        sltp_type: SLTPType,
        position_id: str | None,
        position_side: PositionSide,
        stop_price: float | None,
        quantity: float | None,
        close_position: bool,
    ) -> ProtectiveOrderState:
        return ProtectiveOrderState(
            exchange=result.exchange,
            market_type=result.market_type,
            symbol=result.symbol,
            sltp_type=sltp_type,
            status=result.status,
            order_id=result.order_id,
            client_order_id=result.client_order_id,
            position_id=position_id,
            position_side=position_side,
            price=result.price,
            stop_price=stop_price,
            quantity=quantity,
            reduce_only=not close_position,
            close_position=close_position,
            metadata=dict(result.metadata),
        )

    # ------------------------------------------------------------------
    # Plan / state helpers
    # ------------------------------------------------------------------

    def _plan_from_position_payload(self, payload: Mapping[str, Any]) -> SLTPPlan | None:
        symbol = payload.get("symbol")
        if not symbol:
            raise SLTPError("position payload missing symbol")

        side = self._position_side_from_raw(payload.get("side"))
        if side is None:
            raise SLTPError("position payload missing side")

        size = safe_float(payload.get("size"), 0.0) or 0.0
        if size <= 0:
            return None

        stop_loss = safe_float(payload.get("stop_loss"))
        take_profit = safe_float(payload.get("take_profit"))

        if stop_loss is None and take_profit is None:
            return None

        return SLTPPlan(
            exchange=normalize_exchange(payload.get("exchange") or self._config.default_exchange),
            market_type=normalize_market_type(payload.get("market_type") or self._config.default_market_type),
            symbol=normalize_symbol(str(symbol)),
            position_side=side,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            working_type=self._config.default_working_type,
            price_protect=self._config.price_protect,
            use_close_position_for_stop=self._config.use_close_position_for_full_stop,
            use_close_position_for_take_profit=self._config.use_close_position_for_full_take_profit,
            signal_id=payload.get("signal_id"),
            strategy_name=payload.get("strategy_name"),
            reservation_id=payload.get("reservation_id"),
            position_id=payload.get("position_id"),
            metadata=merge_metadata(
                payload.get("metadata"),
                {
                    "source_event": "position.opened",
                    "entry_price": payload.get("entry_price"),
                    "mark_price": payload.get("mark_price"),
                    "tier": payload.get("tier"),
                    "risk_amount": payload.get("risk_amount"),
                },
            ),
        )

    async def _apply_order_lifecycle_to_protective_state(
        self,
        payload: Mapping[str, Any],
        status: OrderStatus,
    ) -> None:
        client_order_id = payload.get("client_order_id")
        order_id = payload.get("order_id")

        if not client_order_id and not order_id:
            return

        async with self._lock:
            state = self._find_protective_state(
                order_id=str(order_id) if order_id is not None else None,
                client_order_id=str(client_order_id) if client_order_id is not None else None,
            )

            if state is None:
                return

            state.status = status
            state.updated_at = now_ts()

            if status is OrderStatus.FILLED:
                self._stats.triggered += 1
                if state.sltp_type is SLTPType.STOP_LOSS:
                    topic = "execution.stop_loss_triggered"
                elif state.sltp_type is SLTPType.TAKE_PROFIT:
                    topic = "execution.take_profit_triggered"
                else:
                    topic = "execution.trailing_stop_triggered"

                await self._emit_event(
                    topic,
                    {
                        **base_execution_payload(
                            exchange=state.exchange,
                            market_type=state.market_type,
                            symbol=state.symbol,
                            metadata=merge_metadata(state.metadata, payload.get("metadata")),
                        ),
                        "sltp_type": state.sltp_type.value,
                        "order_id": state.order_id,
                        "client_order_id": state.client_order_id,
                        "position_id": state.position_id,
                        "position_side": state.position_side.value if state.position_side else None,
                    },
                    priority=EventPriority.CRITICAL,
                )

    def _matching_protective_orders(
        self,
        *,
        symbol: str,
        position_side: PositionSide | None,
        position_id: str | None,
        sltp_type: SLTPType | None,
        only_active: bool,
    ) -> list[ProtectiveOrderState]:
        states: list[ProtectiveOrderState] = []

        for group in self._protective_orders.values():
            for state in group:
                if state.symbol != symbol:
                    continue

                if position_id is not None and state.position_id != position_id:
                    continue

                if position_side is not None and state.position_side is not position_side:
                    continue

                if sltp_type is not None and state.sltp_type is not sltp_type:
                    continue

                if only_active and not state.is_active:
                    continue

                states.append(state)

        return states

    def _find_protective_state(
        self,
        *,
        order_id: str | None,
        client_order_id: str | None,
    ) -> ProtectiveOrderState | None:
        for group in self._protective_orders.values():
            for state in group:
                if order_id is not None and state.order_id == order_id:
                    return state

                if client_order_id is not None and state.client_order_id == client_order_id:
                    return state

        return None

    @staticmethod
    def _position_key(
        position_id: str | None,
        symbol: str,
        position_side: PositionSide | None,
    ) -> str:
        symbol_n = normalize_symbol(symbol)
        side_value = position_side.value if position_side else "unknown"

        if position_id:
            return f"position:{position_id}"

        return f"{symbol_n}:{side_value}"

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

    def _require_position_side(self, value: Any) -> PositionSide:
        side = self._position_side_from_raw(value)
        if side is None:
            raise SLTPError("position side is required")
        return side

    @staticmethod
    def _require_price(payload: Mapping[str, Any], key: str) -> float:
        value = safe_float(payload.get(key))
        if value is None or value <= 0:
            raise SLTPError(f"{key} must be > 0")
        return value

    @staticmethod
    def _require_size(payload: Mapping[str, Any]) -> float:
        value = safe_float(payload.get("size") or payload.get("quantity"))
        if value is None or value <= 0:
            raise SLTPError("size must be > 0")
        return value

    def _clamp_callback_rate(self, value: float) -> float:
        return max(
            self._config.min_trailing_callback_rate,
            min(value, self._config.max_trailing_callback_rate),
        )

    async def _emit_sltp_update_failed(
        self,
        payload: Mapping[str, Any],
        exc: Exception,
        *,
        update_type: str,
    ) -> None:
        await self._emit_event(
            "execution.sltp.update_failed",
            {
                **dict(payload),
                "update_type": update_type,
                "error": str(exc),
            },
            priority=EventPriority.HIGH,
        )

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
                "Failed to emit SLTPManager event | topic=%s",
                topic,
            )


__all__ = [
    "OrderManagerProtocol",
    "SLTPManager",
]