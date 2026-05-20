from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from core.event_bus import Event, EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler
from execution.config import OrderManagerConfig
from execution.enums import OrderSide, OrderStatus, OrderType, TriggerType, WorkingType
from execution.exceptions import (
    ExchangeClientError,
    OrderCancelError,
    OrderNotFoundError,
    OrderReplaceError,
    OrderSubmitError,
)
from execution.models import (
    OrderFill,
    OrderManagerStats,
    OrderRequest,
    OrderResult,
    OrderState,
    OrderUpdate,
)
from execution.utils import (
    base_execution_payload,
    build_client_order_id,
    merge_metadata,
    normalize_exchange,
    normalize_market_type,
    normalize_order_side,
    normalize_order_type,
    normalize_symbol,
    normalize_time_in_force,
    now_ms,
    now_ts,
)
from risk.enums import PositionSide


class BinanceOrderClientProtocol(Protocol):
    """
    Minimal Binance USD-M Futures REST methods required by OrderManager.

    The concrete implementation is exchanges.binance.binance_rest.BinanceRestClient.
    """

    async def create_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float | None = None,
        price: float | None = None,
        position_side: str | None = None,
        time_in_force: str | None = None,
        reduce_only: bool | None = None,
        new_client_order_id: str | None = None,
        stop_price: float | None = None,
        close_position: bool | None = None,
        activation_price: float | None = None,
        callback_rate: float | None = None,
        working_type: str | None = None,
        price_protect: bool | None = None,
        new_order_resp_type: str | None = "RESULT",
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        ...

    async def cancel_order(
        self,
        *,
        symbol: str,
        order_id: int | None = None,
        orig_client_order_id: str | None = None,
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        ...

    async def cancel_all_open_orders(
        self,
        *,
        symbol: str,
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        ...

    async def get_order(
        self,
        *,
        symbol: str,
        order_id: int | None = None,
        orig_client_order_id: str | None = None,
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        ...

    async def get_open_orders(
        self,
        *,
        symbol: str | None = None,
        recv_window: int | None = None,
    ) -> list[dict[str, Any]]:
        ...

    async def get_user_trades(
        self,
        *,
        symbol: str,
        limit: int = 500,
        order_id: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        recv_window: int | None = None,
    ) -> list[dict[str, Any]]:
        ...


class OrderManager:
    """
    Binance USD-M Futures order bridge.

    Responsibilities:
    - receive internal OrderRequest objects or execution.order_submit_requested events;
    - map OrderRequest -> BinanceRestClient.create_order(...) params;
    - cancel/fetch/reconcile exchange orders;
    - normalize Binance order responses into OrderResult / OrderState;
    - emit execution.order_* events required by RiskManager;
    - never perform risk sizing, risk approval or strategy decisions.

    Important RiskManager contract:
    - execution.order_rejected
    - execution.order_failed
    - execution.order_cancelled
    - execution.order_filled
    - execution.order_partially_filled

    These events are used by risk pending-reservation lifecycle.
    """

    def __init__(
        self,
        config: OrderManagerConfig,
        *,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        exchange_clients: Mapping[str, BinanceOrderClientProtocol] | None = None,
        service_name: str = "execution.order_manager",
        auto_subscribe: bool = True,
        register_scheduler_jobs: bool = True,
    ) -> None:
        self._config = config
        self._config.validate()

        self._event_bus = event_bus
        self._scheduler = scheduler
        self._exchange_clients = dict(exchange_clients or {})

        self._service_name = service_name
        self._auto_subscribe = auto_subscribe
        self._register_scheduler_jobs = register_scheduler_jobs

        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="order_manager",
        )

        self._lock = asyncio.Lock()
        self._subscriptions: list[Any] = []
        self._scheduler_jobs: list[Any] = []

        self._orders_by_order_id: dict[str, OrderState] = {}
        self._orders_by_client_order_id: dict[str, OrderState] = {}

        self._stats = OrderManagerStats()
        self._running = False
        self._started_at: float | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> OrderManagerStats:
        return self._stats

    @property
    def active_orders(self) -> list[OrderState]:
        return [order for order in self._orders_by_order_id.values() if order.is_open]

    async def start(self) -> None:
        if self._running:
            self._logger.warning("OrderManager already started")
            return

        self._running = True
        self._started_at = now_ts()

        if self._event_bus is not None and self._auto_subscribe:
            self.register()

        if self._scheduler is not None and self._register_scheduler_jobs:
            self.register_scheduler_jobs()

        await self._emit_event(
            "execution.order_manager.started",
            {
                "service": self._service_name,
                "started_at": self._started_at,
                "auto_subscribe": self._auto_subscribe,
                "scheduler_jobs": len(self._scheduler_jobs),
            },
            priority=EventPriority.LOW,
        )

        self._logger.info(
            "OrderManager started | subscriptions=%s scheduler_jobs=%s",
            len(self._subscriptions),
            len(self._scheduler_jobs),
        )

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("OrderManager already stopped")
            return

        self.unregister()

        await self._emit_event(
            "execution.order_manager.stopped",
            {
                "service": self._service_name,
                "stopped_at": now_ts(),
            },
            priority=EventPriority.LOW,
        )

        self._running = False
        self._logger.info("OrderManager stopped")

    def register(self) -> None:
        """
        Register EventBus subscriptions.
        """
        if self._event_bus is None:
            self._logger.warning("Cannot register OrderManager: event_bus is not configured")
            return

        if self._subscriptions:
            self._logger.warning("OrderManager subscriptions already registered")
            return

        self._subscriptions.extend(
            [
                self._event_bus.subscribe(
                    "execution.order_submit_requested",
                    self._handle_order_submit_requested,
                    name="execution_order_manager_on_submit_requested",
                ),
                self._event_bus.subscribe(
                    "execution.order_cancel_requested",
                    self._handle_order_cancel_requested,
                    name="execution_order_manager_on_cancel_requested",
                ),
                self._event_bus.subscribe(
                    "execution.order_replace_requested",
                    self._handle_order_replace_requested,
                    name="execution_order_manager_on_replace_requested",
                ),
                self._event_bus.subscribe(
                    "exchange.order.submitted",
                    self._handle_exchange_order_update,
                    name="execution_order_manager_on_exchange_order_submitted",
                ),
                self._event_bus.subscribe(
                    "exchange.order.cancelled",
                    self._handle_exchange_order_update,
                    name="execution_order_manager_on_exchange_order_cancelled",
                ),
                self._event_bus.subscribe(
                    "exchange.open_orders.snapshot",
                    self._handle_exchange_open_orders_snapshot,
                    name="execution_order_manager_on_open_orders_snapshot",
                ),
                self._event_bus.subscribe(
                    "risk.kill_switch",
                    self._handle_kill_switch,
                    name="execution_order_manager_on_kill_switch",
                ),
            ]
        )

        self._logger.info(
            "OrderManager subscriptions registered | count=%s",
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
                self._logger.exception("Failed to unsubscribe OrderManager subscription")

        count = len(self._subscriptions)
        self._subscriptions.clear()

        self._logger.info(
            "OrderManager subscriptions unregistered | count=%s",
            count,
        )

    def register_scheduler_jobs(self) -> None:
        """
        Register periodic reconciliation through core Scheduler.

        No unmanaged asyncio loops are started here.
        """
        if self._scheduler is None:
            return

        if not self._config.reconcile_enabled:
            return

        jobs: list[tuple[str, Callable[[], Awaitable[None]], float]] = [
            (
                "execution.order_manager.reconcile_orders",
                self.reconcile_orders,
                self._config.reconcile_interval_seconds,
            ),
            (
                "execution.order_manager.sync_open_orders",
                self.sync_open_orders,
                self._config.open_order_sync_interval_seconds,
            ),
        ]

        for name, callback, interval_seconds in jobs:
            try:
                job = self._scheduler.add_interval_job(
                    callback,
                    interval_seconds=interval_seconds,
                    name=name,
                    run_immediately=False,
                )
                self._scheduler_jobs.append(job)
            except Exception:
                self._logger.exception(
                    "Failed to register OrderManager scheduler job | name=%s",
                    name,
                )

    def _register_submit_stats(self, result: OrderResult) -> None:
        """
        Register stats for create_order response.

        Binance REST may immediately return terminal or partial status:
        - FILLED
        - PARTIALLY_FILLED
        - REJECTED
        - EXPIRED

        Therefore submit stats and lifecycle stats must both be updated here.
        """
        self._stats.register_submit(result)

        if result.status is not OrderStatus.NEW:
            self._stats.register_update(result)

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        """
        Submit an order to Binance USD-M Futures.

        This method is the main bridge:
        OrderRequest -> BinanceRestClient.create_order(...) -> OrderResult.
        """
        request.validate()

        if not self._config.enabled:
            raise OrderSubmitError("OrderManager is disabled")

        if not self._running:
            self._logger.warning("Submitting order while OrderManager is not marked running")

        if request.client_order_id is None and self._config.generate_client_order_id:
            request.client_order_id = build_client_order_id(
                prefix=self._config.client_order_id_prefix,
                signal_id=request.signal_id,
                strategy_name=request.strategy_name,
                symbol=request.symbol,
                order_intent=request.trigger_type.value,
                leg_index=None,
                max_length=self._config.max_client_order_id_length,
            )

        client = self._get_exchange_client(request.exchange)

        await self._emit_event(
            "execution.order_submit_started",
            request.to_event_payload(),
            priority=EventPriority.HIGH,
        )

        try:
            payload = await self._with_retries(
                operation=lambda: client.create_order(**request.to_binance_params()),
                retries=self._config.submit_retries,
                retry_delay_seconds=self._config.retry_delay_seconds,
                operation_name="create_order",
            )

            result = OrderResult.from_exchange_order(payload, request=request)

            async with self._lock:
                self._upsert_order_state(result)
                self._register_submit_stats(result)

            await self._emit_order_submitted(result)
            await self._emit_order_lifecycle_event(result)

            return result

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            self._logger.exception(
                "Order submit failed | symbol=%s client_order_id=%s",
                request.symbol,
                request.client_order_id,
            )

            await self._emit_event(
                "execution.order_failed",
                {
                    **request.to_event_payload(),
                    "error": str(exc),
                    "failure_stage": "submit",
                },
                priority=EventPriority.CRITICAL,
            )

            raise OrderSubmitError(f"Failed to submit order: {exc}") from exc

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
        """
        Cancel one Binance order by order_id or client_order_id.
        """
        symbol_n = normalize_symbol(symbol)
        exchange_n = normalize_exchange(exchange or self._config.default_exchange)

        if order_id is None and client_order_id is None:
            raise OrderCancelError("Either order_id or client_order_id is required")

        client = self._get_exchange_client(exchange_n)

        state = self.get_order_state(
            order_id=str(order_id) if order_id is not None else None,
            client_order_id=client_order_id,
        )

        if state is not None and state.is_terminal:
            raise OrderCancelError(
                f"Cannot cancel terminal order | status={state.status.value}"
            )

        try:
            payload = await self._with_retries(
                operation=lambda: client.cancel_order(
                    symbol=symbol_n,
                    order_id=int(order_id) if order_id is not None else None,
                    orig_client_order_id=client_order_id,
                ),
                retries=self._config.cancel_retries,
                retry_delay_seconds=self._config.retry_delay_seconds,
                operation_name="cancel_order",
            )

            dummy_request = self._request_from_state_or_cancel_args(
                state=state,
                symbol=symbol_n,
                exchange=exchange_n,
                order_id=order_id,
                client_order_id=client_order_id,
                metadata=metadata,
            )
            result = OrderResult.from_exchange_order(
                payload,
                request=dummy_request,
                metadata={"cancel_reason": reason} if reason else None,
            )

            async with self._lock:
                self._upsert_order_state(result)
                self._stats.register_update(result)

            await self._emit_order_lifecycle_event(result)

            return result

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            self._logger.exception(
                "Order cancel failed | symbol=%s order_id=%s client_order_id=%s",
                symbol_n,
                order_id,
                client_order_id,
            )

            await self._emit_event(
                "execution.order_failed",
                {
                    **base_execution_payload(
                        exchange=exchange_n,
                        market_type=self._config.default_market_type,
                        symbol=symbol_n,
                        metadata=metadata,
                    ),
                    "order_id": str(order_id) if order_id is not None else None,
                    "client_order_id": client_order_id,
                    "error": str(exc),
                    "failure_stage": "cancel",
                    "reason": reason,
                },
                priority=EventPriority.CRITICAL,
            )

            raise OrderCancelError(f"Failed to cancel order: {exc}") from exc

    async def cancel_all_orders(
        self,
        *,
        symbol: str,
        exchange: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Cancel all open orders for one symbol.

        Binance endpoint works per symbol.
        """
        symbol_n = normalize_symbol(symbol)
        exchange_n = normalize_exchange(exchange or self._config.default_exchange)
        client = self._get_exchange_client(exchange_n)

        try:
            payload = await client.cancel_all_open_orders(symbol=symbol_n)

            await self._emit_event(
                "execution.orders_cancelled",
                {
                    **base_execution_payload(
                        exchange=exchange_n,
                        market_type=self._config.default_market_type,
                        symbol=symbol_n,
                    ),
                    "reason": reason,
                    "exchange_payload": dict(payload),
                },
                priority=EventPriority.CRITICAL,
            )

            async with self._lock:
                for state in self._orders_by_order_id.values():
                    if state.symbol == symbol_n and state.is_open:
                        state.status = OrderStatus.CANCELED
                        state.updated_at = now_ts()

            return dict(payload)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            self._logger.exception(
                "Cancel all orders failed | symbol=%s",
                symbol_n,
            )
            raise OrderCancelError(f"Failed to cancel all orders: {exc}") from exc

    async def replace_order(
        self,
        *,
        existing_symbol: str,
        new_request: OrderRequest,
        existing_order_id: str | int | None = None,
        existing_client_order_id: str | None = None,
        reason: str | None = None,
    ) -> OrderResult:
        """
        Replace order as cancel + submit.

        Binance USD-M Futures does not provide one universal atomic replace
        flow for all order types.
        """
        try:
            await self.cancel_order(
                symbol=existing_symbol,
                order_id=existing_order_id,
                client_order_id=existing_client_order_id,
                exchange=new_request.exchange,
                reason=reason or "replace_order",
            )
            return await self.submit_order(new_request)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            raise OrderReplaceError(f"Failed to replace order: {exc}") from exc

    async def fetch_order(
        self,
        *,
        symbol: str,
        order_id: str | int | None = None,
        client_order_id: str | None = None,
        exchange: str | None = None,
    ) -> OrderResult:
        symbol_n = normalize_symbol(symbol)
        exchange_n = normalize_exchange(exchange or self._config.default_exchange)

        if order_id is None and client_order_id is None:
            raise OrderNotFoundError("Either order_id or client_order_id is required")

        client = self._get_exchange_client(exchange_n)

        try:
            payload = await client.get_order(
                symbol=symbol_n,
                order_id=int(order_id) if order_id is not None else None,
                orig_client_order_id=client_order_id,
            )

            result = OrderResult.from_exchange_order(payload)

            async with self._lock:
                self._upsert_order_state(result)
                self._stats.register_update(result)

            return result

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            raise OrderNotFoundError(f"Failed to fetch order: {exc}") from exc

    async def sync_open_orders(self, symbol: str | None = None) -> None:
        """
        Sync open orders from Binance.

        This is also used by reconciliation jobs.
        """
        if not self._config.reconcile_enabled:
            return

        exchange = self._config.default_exchange
        client = self._get_exchange_client(exchange)

        symbol_n = normalize_symbol(symbol) if symbol else None

        try:
            payloads = await client.get_open_orders(symbol=symbol_n)

            async with self._lock:
                exchange_open_keys: set[str] = set()

                for payload in payloads:
                    result = OrderResult.from_exchange_order(payload)
                    state = self._upsert_order_state(result)

                    if state.order_id:
                        exchange_open_keys.add(f"order:{state.order_id}")
                    if state.client_order_id:
                        exchange_open_keys.add(f"client:{state.client_order_id}")

                self._mark_missing_local_orders_as_sync_required(
                    symbol=symbol_n,
                    exchange_open_keys=exchange_open_keys,
                )

            await self._emit_event(
                "execution.order_sync_completed",
                {
                    **base_execution_payload(
                        exchange=exchange,
                        market_type=self._config.default_market_type,
                        symbol=symbol_n,
                    ),
                    "open_orders_count": len(payloads),
                },
                priority=EventPriority.LOW,
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.reconciliation_failures += 1
            self._stats.register_failure(str(exc))
            self._logger.exception("Open order sync failed")
            await self._emit_event(
                "execution.order_sync_failed",
                {
                    **base_execution_payload(
                        exchange=exchange,
                        market_type=self._config.default_market_type,
                        symbol=symbol_n,
                    ),
                    "error": str(exc),
                },
                priority=EventPriority.HIGH,
            )

    async def reconcile_orders(self) -> None:
        """
        Periodic order reconciliation.

        Compact package rule: reconciliation lives inside order_manager.py.
        """
        self._stats.reconciliation_runs += 1

        try:
            await self.sync_open_orders()
            await self._emit_event(
                "execution.order_reconciled",
                {
                    "exchange": self._config.default_exchange,
                    "market_type": self._config.default_market_type,
                    "active_orders": len(self.active_orders),
                    "timestamp": now_ms(),
                },
                priority=EventPriority.LOW,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.reconciliation_failures += 1
            self._stats.register_failure(str(exc))
            self._logger.exception("Order reconciliation failed")

    async def load_user_trades_as_fills(
        self,
        *,
        symbol: str,
        order_id: str | int | None = None,
        execution_id: str | None = None,
        signal_id: str | None = None,
        strategy_name: str | None = None,
        reservation_id: str | None = None,
        limit: int = 500,
    ) -> list[OrderFill]:
        """
        Fetch Binance user trades and normalize them as OrderFill objects.

        This method is useful for reconciliation and recovery.
        """
        symbol_n = normalize_symbol(symbol)
        client = self._get_exchange_client(self._config.default_exchange)

        trades = await client.get_user_trades(
            symbol=symbol_n,
            order_id=int(order_id) if order_id is not None else None,
            limit=limit,
        )

        fills: list[OrderFill] = []

        for trade in trades:
            try:
                fill = OrderFill.from_user_trade(
                    trade,
                    execution_id=execution_id,
                    signal_id=signal_id,
                    strategy_name=strategy_name,
                    reservation_id=reservation_id,
                )
                fills.append(fill)
            except Exception:
                self._logger.exception(
                    "Failed to normalize user trade as fill | symbol=%s order_id=%s",
                    symbol_n,
                    order_id,
                )

        return fills

    def get_order_state(
        self,
        *,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> OrderState | None:
        if order_id is not None:
            state = self._orders_by_order_id.get(str(order_id))
            if state is not None:
                return state

        if client_order_id is not None:
            return self._orders_by_client_order_id.get(client_order_id)

        return None

    def list_orders(
        self,
        *,
        symbol: str | None = None,
        include_terminal: bool = True,
    ) -> list[OrderState]:
        symbol_n = normalize_symbol(symbol) if symbol else None

        orders = list(self._orders_by_order_id.values())

        if symbol_n is not None:
            orders = [order for order in orders if order.symbol == symbol_n]

        if not include_terminal:
            orders = [order for order in orders if order.is_open]

        return orders

    def snapshot(self) -> dict[str, Any]:
        return {
            "service": self._service_name,
            "running": self._running,
            "started_at": self._started_at,
            "orders_count": len(self._orders_by_order_id),
            "active_orders_count": len(self.active_orders),
            "stats": self._stats.snapshot(),
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_order_submit_requested(self, event: Event | dict[str, Any]) -> None:
        payload = self._event_payload(event)

        try:
            request = self._order_request_from_payload(payload)
            await self.submit_order(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._logger.exception("Failed to handle order submit request")
            await self._emit_event(
                "execution.order_failed",
                {
                    **payload,
                    "error": str(exc),
                    "failure_stage": "event_submit_request",
                },
                priority=EventPriority.CRITICAL,
            )

    async def _handle_order_cancel_requested(self, event: Event | dict[str, Any]) -> None:
        payload = self._event_payload(event)

        try:
            await self.cancel_order(
                symbol=payload["symbol"],
                order_id=payload.get("order_id"),
                client_order_id=payload.get("client_order_id"),
                exchange=payload.get("exchange"),
                reason=payload.get("reason"),
                metadata=payload.get("metadata"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._logger.exception("Failed to handle order cancel request")
            await self._emit_event(
                "execution.order_failed",
                {
                    **payload,
                    "error": str(exc),
                    "failure_stage": "event_cancel_request",
                },
                priority=EventPriority.CRITICAL,
            )

    async def _handle_order_replace_requested(self, event: Event | dict[str, Any]) -> None:
        payload = self._event_payload(event)

        try:
            new_request_payload = payload.get("new_order") or payload.get("request") or payload
            new_request = self._order_request_from_payload(new_request_payload)

            await self.replace_order(
                existing_symbol=payload["symbol"],
                existing_order_id=payload.get("order_id"),
                existing_client_order_id=payload.get("client_order_id"),
                new_request=new_request,
                reason=payload.get("reason"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._logger.exception("Failed to handle order replace request")
            await self._emit_event(
                "execution.order_failed",
                {
                    **payload,
                    "error": str(exc),
                    "failure_stage": "event_replace_request",
                },
                priority=EventPriority.CRITICAL,
            )

    async def _handle_exchange_order_update(self, event: Event | dict[str, Any]) -> None:
        payload = self._event_payload(event)

        try:
            result = OrderResult.from_exchange_order(payload)

            async with self._lock:
                previous_state = self.get_order_state(
                    order_id=result.order_id,
                    client_order_id=result.client_order_id,
                )
                previous_status = previous_state.status if previous_state else None
                self._upsert_order_state(result)
                self._stats.register_update(result)

            update = OrderUpdate(
                result=result,
                previous_status=previous_status,
                update_reason="exchange_update",
            )

            await self._emit_event(
                "execution.order_status_updated",
                update.to_event_payload(),
                priority=EventPriority.HIGH,
            )
            await self._emit_order_lifecycle_event(result)

        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception("Failed to handle exchange order update")

    async def _handle_exchange_open_orders_snapshot(self, event: Event | dict[str, Any]) -> None:
        payload = self._event_payload(event)
        orders = payload.get("orders", [])

        if not isinstance(orders, list):
            return

        async with self._lock:
            for order_payload in orders:
                try:
                    result = OrderResult.from_exchange_order(order_payload)
                    self._upsert_order_state(result)
                except Exception:
                    self._logger.exception("Failed to apply open order snapshot item")

    async def _handle_kill_switch(self, event: Event | dict[str, Any]) -> None:
        payload = self._event_payload(event)

        if not self._config.enabled:
            return

        reason = payload.get("reason") or payload.get("message") or "risk.kill_switch"

        if not payload.get("cancel_open_orders", True):
            return

        symbols = sorted({order.symbol for order in self.active_orders})

        for symbol in symbols:
            try:
                await self.cancel_all_orders(
                    symbol=symbol,
                    reason=str(reason),
                )
            except Exception:
                self._logger.exception(
                    "Failed to cancel orders during kill switch | symbol=%s",
                    symbol,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_exchange_client(self, exchange: str | None) -> BinanceOrderClientProtocol:
        exchange_n = normalize_exchange(exchange or self._config.default_exchange)

        client = self._exchange_clients.get(exchange_n)

        if client is None:
            raise ExchangeClientError(f"Exchange client is not configured: {exchange_n}")

        return client

    async def _with_retries(
        self,
        *,
        operation: Callable[[], Awaitable[dict[str, Any]]],
        retries: int,
        retry_delay_seconds: float,
        operation_name: str,
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                return await operation()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc

                if attempt >= retries:
                    break

                self._logger.warning(
                    "OrderManager operation retry scheduled | operation=%s attempt=%s retries=%s error=%s",
                    operation_name,
                    attempt + 1,
                    retries,
                    str(exc),
                )
                await asyncio.sleep(retry_delay_seconds)

        assert last_error is not None
        raise last_error

    def _upsert_order_state(self, result: OrderResult) -> OrderState:
        state = self.get_order_state(
            order_id=result.order_id,
            client_order_id=result.client_order_id,
        )

        if state is None:
            state = OrderState.from_result(result)
        else:
            state.apply_result(result)

        if state.order_id:
            self._orders_by_order_id[state.order_id] = state

        if state.client_order_id:
            self._orders_by_client_order_id[state.client_order_id] = state

        return state

    def _mark_missing_local_orders_as_sync_required(
        self,
        *,
        symbol: str | None,
        exchange_open_keys: set[str],
    ) -> None:
        """
        Conservative reconciliation helper.

        If a local order is open but absent from exchange open orders, do not
        guess filled/cancelled. Mark it as needing fetch/reconcile.
        """
        for state in self._orders_by_order_id.values():
            if symbol is not None and state.symbol != symbol:
                continue

            if not state.is_open:
                continue

            order_key = f"order:{state.order_id}" if state.order_id else None
            client_key = f"client:{state.client_order_id}" if state.client_order_id else None

            if order_key in exchange_open_keys or client_key in exchange_open_keys:
                continue

            state.metadata["sync_required"] = True
            state.metadata["sync_required_at"] = now_ts()
            state.updated_at = now_ts()

    async def _emit_order_submitted(self, result: OrderResult) -> None:
        await self._emit_event(
            "execution.order_submitted",
            result.to_event_payload(),
            priority=EventPriority.CRITICAL,
        )

        if self._config.emit_acknowledged_events:
            await self._emit_event(
                "execution.order_acknowledged",
                result.to_event_payload(),
                priority=EventPriority.HIGH,
            )

    async def _emit_order_lifecycle_event(self, result: OrderResult) -> None:
        payload = result.to_event_payload()

        if result.status is OrderStatus.PARTIALLY_FILLED:
            if self._config.emit_partially_filled_events:
                await self._emit_event(
                    "execution.order_partially_filled",
                    payload,
                    priority=EventPriority.CRITICAL,
                )

        elif result.status is OrderStatus.FILLED:
            await self._emit_event(
                "execution.order_filled",
                payload,
                priority=EventPriority.CRITICAL,
            )

        elif result.status is OrderStatus.CANCELED:
            await self._emit_event(
                "execution.order_cancelled",
                payload,
                priority=EventPriority.CRITICAL,
            )

        elif result.status is OrderStatus.REJECTED:
            await self._emit_event(
                "execution.order_rejected",
                payload,
                priority=EventPriority.CRITICAL,
            )

        elif result.status in {OrderStatus.EXPIRED, OrderStatus.EXPIRED_IN_MATCH}:
            await self._emit_event(
                "execution.order_failed",
                {
                    **payload,
                    "failure_stage": "expired",
                },
                priority=EventPriority.CRITICAL,
            )

    def _request_from_state_or_cancel_args(
        self,
        *,
        state: OrderState | None,
        symbol: str,
        exchange: str,
        order_id: str | int | None,
        client_order_id: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> OrderRequest:
        if state is not None:
            return OrderRequest(
                execution_id=state.execution_id or "unknown",
                leg_id=state.leg_id,
                exchange=state.exchange,
                market_type=state.market_type,
                symbol=state.symbol,
                side=state.side or OrderSide.BUY,
                order_type=state.order_type or OrderType.MARKET,
                quantity=state.original_quantity if state.original_quantity > 0 else None,
                client_order_id=state.client_order_id,
                signal_id=state.signal_id,
                strategy_name=state.strategy_name,
                reservation_id=state.reservation_id,
                reduce_only=bool(state.reduce_only),
                close_position=bool(state.close_position),
                metadata=merge_metadata(state.metadata, metadata),
            )

        return OrderRequest(
            execution_id="unknown",
            exchange=exchange,
            market_type=self._config.default_market_type,
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=None,
            client_order_id=client_order_id,
            metadata=merge_metadata(
                metadata,
                {
                    "cancel_order_id": str(order_id) if order_id is not None else None,
                    "cancel_client_order_id": client_order_id,
                },
            ),
        )

    def _order_request_from_payload(self, payload: Mapping[str, Any]) -> OrderRequest:
        return OrderRequest(
            execution_id=str(payload.get("execution_id") or "unknown"),
            leg_id=payload.get("leg_id"),
            exchange=normalize_exchange(payload.get("exchange") or self._config.default_exchange),
            market_type=normalize_market_type(payload.get("market_type") or self._config.default_market_type),
            symbol=normalize_symbol(str(payload["symbol"])),
            side=normalize_order_side(payload["side"]),
            order_type=normalize_order_type(payload.get("order_type") or payload.get("type")),
            quantity=payload.get("quantity"),
            price=payload.get("price"),
            position_side=self._position_side_from_raw(payload.get("position_side")),
            time_in_force=normalize_time_in_force(payload.get("time_in_force")),
            reduce_only=bool(payload.get("reduce_only", False)),
            close_position=bool(payload.get("close_position", False)),
            client_order_id=payload.get("client_order_id"),
            stop_price=payload.get("stop_price"),
            activation_price=payload.get("activation_price"),
            callback_rate=payload.get("callback_rate"),
            working_type=self._working_type_from_raw(payload.get("working_type")),
            price_protect=payload.get("price_protect"),
            new_order_resp_type=payload.get("new_order_resp_type") or self._config.new_order_response_type,
            signal_id=payload.get("signal_id"),
            strategy_name=payload.get("strategy_name"),
            reservation_id=payload.get("reservation_id"),
            trigger_type=self._trigger_type_from_raw(payload.get("trigger_type")),
            metadata=dict(payload.get("metadata") or {}),
        )

    @staticmethod
    def _position_side_from_raw(value: Any) -> PositionSide | None:
        if value is None:
            return None

        if isinstance(value, PositionSide):
            return value

        normalized = str(value).strip().lower()

        if normalized == PositionSide.LONG.value:
            return PositionSide.LONG

        if normalized == PositionSide.SHORT.value:
            return PositionSide.SHORT

        if normalized.upper() == "LONG":
            return PositionSide.LONG

        if normalized.upper() == "SHORT":
            return PositionSide.SHORT

        return None

    @staticmethod
    def _working_type_from_raw(value: Any) -> WorkingType | None:
        if value is None:
            return None

        if isinstance(value, WorkingType):
            return value

        normalized = str(value).strip().upper()

        try:
            return WorkingType(normalized)
        except ValueError:
            return None

    @staticmethod
    def _trigger_type_from_raw(value: Any) -> TriggerType:
        if value is None:
            return TriggerType.NONE

        if isinstance(value, TriggerType):
            return value

        normalized = str(value).strip().lower()

        try:
            return TriggerType(normalized)
        except ValueError:
            return TriggerType.NONE

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
                "Failed to emit OrderManager event | topic=%s",
                topic,
            )


__all__ = [
    "BinanceOrderClientProtocol",
    "OrderManager",
]