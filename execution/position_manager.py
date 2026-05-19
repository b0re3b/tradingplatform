from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from typing import Any, Protocol

from core.event_bus import Event, EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler

from execution.config import PositionManagerConfig
from execution.enums import OrderSide
from execution.exceptions import ExchangeClientError, PositionError, PositionSyncError
from execution.models import (
    OrderFill,
    PositionManagerStats,
    PositionSnapshot,
    PositionState,
    PositionUpdate,
)
from execution.utils import (
    base_execution_payload,
    calculate_notional,
    calculate_order_avg_price_from_payload,
    extract_client_order_id,
    extract_executed_quantity,
    extract_order_id,
    merge_metadata,
    normalize_exchange,
    normalize_market_type,
    normalize_order_side,
    normalize_symbol,
    now_ms,
    now_ts,
    safe_float,
)

from risk.enums import PositionSide, TradeTier


class BinancePositionClientProtocol(Protocol):
    """
    Minimal Binance USD-M Futures REST methods required by PositionManager.

    Concrete implementation:
    exchanges.binance.binance_rest.BinanceRestClient
    """

    async def get_positions(
        self,
        *,
        symbol: str | None = None,
        recv_window: int | None = None,
    ) -> list[dict[str, Any]]:
        ...


class PositionManager:
    """
    Execution-side futures position state manager.

    Responsibilities:
    - listen to execution.order_filled / execution.order_partially_filled;
    - maintain local PositionState per symbol/side;
    - reconcile positions from BinanceRestClient.get_positions();
    - emit position.opened / position.updated / position.closed;
    - provide payloads compatible with risk.models.PortfolioPosition.

    Important RiskManager contract:
    RiskManager listens to:
    - position.opened
    - position.updated
    - position.closed

    Therefore these event payloads must contain:
    symbol, side, size, entry_price, mark_price, notional_value, leverage,
    margin_used, risk_amount, stop_loss, take_profit, tier, strategy_name,
    signal_id, position_id, realized_pnl, unrealized_pnl.
    """

    def __init__(
        self,
        config: PositionManagerConfig,
        *,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        exchange_clients: Mapping[str, BinancePositionClientProtocol] | None = None,
        service_name: str = "execution.position_manager",
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
            event_type="position_manager",
        )

        self._lock = asyncio.Lock()
        self._subscriptions: list[Any] = []
        self._scheduler_jobs: list[Any] = []

        self._positions: dict[str, PositionState] = {}

        # Prevent duplicated position application when order lifecycle emits
        # several updates for the same order. We store cumulative applied qty.
        self._applied_order_qty: dict[str, float] = {}

        self._stats = PositionManagerStats()
        self._running = False
        self._started_at: float | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> PositionManagerStats:
        return self._stats

    @property
    def open_positions(self) -> list[PositionState]:
        return [position for position in self._positions.values() if position.is_open]

    async def start(self) -> None:
        if self._running:
            self._logger.warning("PositionManager already started")
            return

        self._running = True
        self._started_at = now_ts()

        if self._event_bus is not None and self._auto_subscribe:
            self.register()

        if self._scheduler is not None and self._register_scheduler_jobs:
            self.register_scheduler_jobs()

        await self._emit_event(
            "execution.position_manager.started",
            {
                "service": self._service_name,
                "started_at": self._started_at,
                "auto_subscribe": self._auto_subscribe,
                "scheduler_jobs": len(self._scheduler_jobs),
            },
            priority=EventPriority.LOW,
        )

        self._logger.info(
            "PositionManager started | subscriptions=%s scheduler_jobs=%s",
            len(self._subscriptions),
            len(self._scheduler_jobs),
        )

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("PositionManager already stopped")
            return

        self.unregister()

        await self._emit_event(
            "execution.position_manager.stopped",
            {
                "service": self._service_name,
                "stopped_at": now_ts(),
            },
            priority=EventPriority.LOW,
        )

        self._running = False
        self._logger.info("PositionManager stopped")

    def register(self) -> None:
        if self._event_bus is None:
            self._logger.warning("Cannot register PositionManager: event_bus is not configured")
            return

        if self._subscriptions:
            self._logger.warning("PositionManager subscriptions already registered")
            return

        self._subscriptions.extend(
            [
                self._event_bus.subscribe(
                    "execution.order_filled",
                    self._handle_order_filled,
                    name="execution_position_manager_on_order_filled",
                ),
                self._event_bus.subscribe(
                    "execution.order_partially_filled",
                    self._handle_order_partially_filled,
                    name="execution_position_manager_on_order_partially_filled",
                ),
                self._event_bus.subscribe(
                    "exchange.positions.snapshot",
                    self._handle_exchange_positions_snapshot,
                    name="execution_position_manager_on_exchange_positions_snapshot",
                ),
                self._event_bus.subscribe(
                    "risk.kill_switch",
                    self._handle_kill_switch,
                    name="execution_position_manager_on_kill_switch",
                ),
            ]
        )

        self._logger.info(
            "PositionManager subscriptions registered | count=%s",
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
                self._logger.exception("Failed to unsubscribe PositionManager subscription")

        count = len(self._subscriptions)
        self._subscriptions.clear()

        self._logger.info(
            "PositionManager subscriptions unregistered | count=%s",
            count,
        )

    def register_scheduler_jobs(self) -> None:
        """
        Register position reconciliation jobs through core Scheduler.

        No unmanaged asyncio loops are used.
        """
        if self._scheduler is None:
            return

        if not self._config.reconcile_enabled:
            return

        jobs: list[tuple[str, Any, float]] = [
            (
                "execution.position_manager.reconcile_positions",
                self.reconcile_positions,
                self._config.reconcile_interval_seconds,
            ),
            (
                "execution.position_manager.sync_positions",
                self.sync_positions,
                self._config.position_sync_interval_seconds,
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
                    "Failed to register PositionManager scheduler job | name=%s",
                    name,
                )

    async def apply_fill(self, fill: OrderFill) -> PositionUpdate | None:
        """
        Apply normalized fill to local position state.

        Returns emitted PositionUpdate or None when fill has zero delta after
        deduplication.
        """
        fill.validate()

        async with self._lock:
            order_key = self._order_key_from_fill(fill)
            if order_key is not None:
                previous_applied_qty = self._applied_order_qty.get(order_key, 0.0)
                delta_qty = fill.quantity - previous_applied_qty

                if delta_qty <= self._config.min_position_size_epsilon:
                    self._logger.debug(
                        "Skipping duplicate/old fill quantity | order_key=%s fill_qty=%s previous_applied=%s",
                        order_key,
                        fill.quantity,
                        previous_applied_qty,
                    )
                    return None

                self._applied_order_qty[order_key] = fill.quantity

                fill = self._copy_fill_with_quantity(fill, delta_qty)

            key = self._position_key(fill.symbol, fill.position_side)
            state = self._positions.get(key)

            if state is None:
                state = self._new_position_state_from_fill(fill)
                self._positions[key] = state

            self._enrich_state_from_fill_metadata(state, fill)

            update = state.apply_fill(fill)

            if update.update_type == "closed" and not state.is_open:
                # Keep the closed state for audit/snapshot. It can be replaced
                # by future open on the same key.
                pass

            self._stats.register_update(update)

        await self._emit_position_update(update)

        return update

    async def apply_position_snapshot(
        self,
        snapshot: PositionSnapshot,
        *,
        emit_unchanged: bool | None = None,
    ) -> PositionUpdate | None:
        """
        Apply one exchange position snapshot.
        """
        snapshot.validate()

        emit_unchanged = (
            self._config.emit_unchanged_snapshots
            if emit_unchanged is None
            else emit_unchanged
        )

        async with self._lock:
            key = self._position_key(snapshot.symbol, snapshot.side)
            state = self._positions.get(key)

            if state is None:
                state = self._new_position_state_from_snapshot(snapshot)
                self._positions[key] = state

            previous_size = state.size
            previous_side = state.side

            update = state.apply_snapshot(snapshot)

            unchanged = (
                abs(previous_size - state.size) <= self._config.min_position_size_epsilon
                and previous_side is state.side
                and update.update_type == "updated"
            )

            if unchanged and not emit_unchanged:
                return None

            self._stats.register_update(update)

        await self._emit_position_update(update)

        return update

    async def sync_positions(self, symbol: str | None = None) -> list[PositionSnapshot]:
        """
        Fetch Binance position snapshots and apply them to local state.
        """
        if not self._config.reconcile_enabled:
            return []

        exchange = self._config.default_exchange
        client = self._get_exchange_client(exchange)
        symbol_n = normalize_symbol(symbol) if symbol else None

        try:
            payloads = await client.get_positions(symbol=symbol_n)

            snapshots: list[PositionSnapshot] = []

            for payload in payloads:
                try:
                    snapshot = PositionSnapshot.from_exchange_position(payload)
                    snapshots.append(snapshot)
                    await self.apply_position_snapshot(snapshot)
                except Exception:
                    self._logger.exception(
                        "Failed to apply position snapshot item | symbol=%s",
                        payload.get("symbol"),
                    )

            await self._emit_event(
                "position.sync_completed",
                {
                    **base_execution_payload(
                        exchange=exchange,
                        market_type=self._config.default_market_type,
                        symbol=symbol_n,
                    ),
                    "positions_count": len(snapshots),
                },
                priority=EventPriority.LOW,
            )

            return snapshots

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.reconciliation_failures += 1
            self._stats.register_failure(str(exc))
            self._logger.exception("Position sync failed")

            await self._emit_event(
                "position.sync_required",
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

            raise PositionSyncError(f"Failed to sync positions: {exc}") from exc

    async def reconcile_positions(self) -> None:
        """
        Periodic position reconciliation.

        Compact package rule: reconciliation lives inside position_manager.py.
        """
        self._stats.reconciliation_runs += 1

        try:
            await self.sync_positions()
            await self._emit_event(
                "position.reconciled",
                {
                    "exchange": self._config.default_exchange,
                    "market_type": self._config.default_market_type,
                    "open_positions": len(self.open_positions),
                    "timestamp": now_ms(),
                },
                priority=EventPriority.LOW,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.reconciliation_failures += 1
            self._stats.register_failure(str(exc))
            self._logger.exception("Position reconciliation failed")

    def get_position(
        self,
        *,
        symbol: str,
        side: PositionSide | str | None = None,
    ) -> PositionState | None:
        symbol_n = normalize_symbol(symbol)
        side_n = self._position_side_from_raw(side)

        if side_n is not None:
            return self._positions.get(self._position_key(symbol_n, side_n))

        # One-way fallback: return any open state for symbol.
        for state in self._positions.values():
            if state.symbol == symbol_n and state.is_open:
                return state

        return self._positions.get(self._position_key(symbol_n, None))

    def list_positions(
        self,
        *,
        symbol: str | None = None,
        include_closed: bool = False,
    ) -> list[PositionState]:
        symbol_n = normalize_symbol(symbol) if symbol else None

        positions = list(self._positions.values())

        if symbol_n is not None:
            positions = [position for position in positions if position.symbol == symbol_n]

        if not include_closed:
            positions = [position for position in positions if position.is_open]

        return positions

    def has_open_position(
        self,
        *,
        symbol: str,
        side: PositionSide | str | None = None,
    ) -> bool:
        position = self.get_position(symbol=symbol, side=side)
        return bool(position and position.is_open)

    def calculate_exposure(self) -> dict[str, Any]:
        """
        Lightweight execution-side exposure snapshot.

        RiskManager remains the source of truth for risk exposure. This is
        execution-local diagnostic data.
        """
        positions = self.open_positions

        gross_notional = sum(position.notional_value for position in positions)
        margin_used = sum(position.margin_used for position in positions)
        unrealized_pnl = sum(position.unrealized_pnl for position in positions)

        by_symbol: dict[str, float] = {}
        by_side: dict[str, float] = {}

        for position in positions:
            by_symbol[position.symbol] = by_symbol.get(position.symbol, 0.0) + position.notional_value

            side_value = position.side.value if position.side else "unknown"
            by_side[side_value] = by_side.get(side_value, 0.0) + position.notional_value

        return {
            "exchange": self._config.default_exchange,
            "market_type": self._config.default_market_type,
            "open_positions": len(positions),
            "gross_notional": gross_notional,
            "margin_used": margin_used,
            "unrealized_pnl": unrealized_pnl,
            "by_symbol": by_symbol,
            "by_side": by_side,
            "timestamp": now_ms(),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "service": self._service_name,
            "running": self._running,
            "started_at": self._started_at,
            "positions_count": len(self._positions),
            "open_positions_count": len(self.open_positions),
            "positions": [
                position.to_portfolio_position_payload()
                for position in self.list_positions(include_closed=True)
            ],
            "exposure": self.calculate_exposure(),
            "stats": self._stats.snapshot(),
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_order_filled(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._event_payload(event)

        try:
            fill = self._fill_from_order_payload(payload)
            if fill is None:
                return

            await self.apply_fill(fill)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            self._logger.exception("Failed to handle execution.order_filled")

            await self._emit_event(
                "position.sync_required",
                {
                    **base_execution_payload(
                        exchange=payload.get("exchange") or self._config.default_exchange,
                        market_type=payload.get("market_type") or self._config.default_market_type,
                        symbol=payload.get("symbol"),
                        signal_id=payload.get("signal_id"),
                        strategy_name=payload.get("strategy_name"),
                        reservation_id=payload.get("reservation_id"),
                    ),
                    "error": str(exc),
                    "reason": "failed_to_apply_order_filled",
                },
                priority=EventPriority.HIGH,
            )

    async def _handle_order_partially_filled(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._event_payload(event)

        try:
            fill = self._fill_from_order_payload(payload)
            if fill is None:
                return

            await self.apply_fill(fill)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            self._logger.exception("Failed to handle execution.order_partially_filled")

            await self._emit_event(
                "position.sync_required",
                {
                    **base_execution_payload(
                        exchange=payload.get("exchange") or self._config.default_exchange,
                        market_type=payload.get("market_type") or self._config.default_market_type,
                        symbol=payload.get("symbol"),
                        signal_id=payload.get("signal_id"),
                        strategy_name=payload.get("strategy_name"),
                        reservation_id=payload.get("reservation_id"),
                    ),
                    "error": str(exc),
                    "reason": "failed_to_apply_order_partially_filled",
                },
                priority=EventPriority.HIGH,
            )

    async def _handle_exchange_positions_snapshot(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._event_payload(event)
        positions = payload.get("positions", [])

        if not isinstance(positions, list):
            return

        for item in positions:
            if not isinstance(item, Mapping):
                continue

            try:
                snapshot = PositionSnapshot.from_exchange_position(item)
                await self.apply_position_snapshot(snapshot)
            except Exception:
                self._logger.exception(
                    "Failed to apply exchange position snapshot | symbol=%s",
                    item.get("symbol"),
                )

    async def _handle_kill_switch(self, event: Event | Mapping[str, Any]) -> None:
        """
        PositionManager does not close positions by itself.

        RiskManager/TradeExecutor should emit explicit close/reduce commands.
        Here we only emit current exposure snapshot for observability.
        """
        payload = self._event_payload(event)

        await self._emit_event(
            "position.kill_switch_snapshot",
            {
                **self.calculate_exposure(),
                "reason": payload.get("reason") or payload.get("message"),
            },
            priority=EventPriority.HIGH,
        )

    # ------------------------------------------------------------------
    # Internal model helpers
    # ------------------------------------------------------------------

    def _fill_from_order_payload(self, payload: Mapping[str, Any]) -> OrderFill | None:
        symbol = payload.get("symbol")
        if not symbol:
            raise PositionError("Order fill payload missing symbol")

        executed_quantity = extract_executed_quantity(payload)
        if executed_quantity <= self._config.min_position_size_epsilon:
            return None

        price = (
            safe_float(payload.get("avg_price"))
            or calculate_order_avg_price_from_payload(payload)
            or safe_float(payload.get("price"))
        )

        if price is None or price <= 0:
            raise PositionError("Order fill payload missing valid fill price")

        side = payload.get("side")
        if side is None:
            raise PositionError("Order fill payload missing order side")

        position_side = self._position_side_from_raw(payload.get("position_side"))

        if position_side is None:
            position_side = self._infer_position_side_from_order_side(
                order_side=normalize_order_side(side),
                reduce_only=bool(payload.get("reduce_only", False)),
                close_position=bool(payload.get("close_position", False)),
            )

        quote_qty = (
            safe_float(payload.get("cumulative_quote_quantity"))
            or safe_float(payload.get("cum_quote"))
            or safe_float(payload.get("quote_quantity"))
        )

        return OrderFill(
            exchange=normalize_exchange(payload.get("exchange") or self._config.default_exchange),
            market_type=normalize_market_type(payload.get("market_type") or self._config.default_market_type),
            symbol=normalize_symbol(str(symbol)),
            side=normalize_order_side(side),
            quantity=executed_quantity,
            price=price,
            order_id=extract_order_id(payload),
            client_order_id=extract_client_order_id(payload),
            trade_id=payload.get("trade_id"),
            position_side=position_side,
            quote_quantity=quote_qty,
            commission=safe_float(payload.get("commission")),
            commission_asset=payload.get("commission_asset"),
            realized_pnl=safe_float(payload.get("realized_pnl")),
            maker=payload.get("maker"),
            execution_id=payload.get("execution_id"),
            signal_id=payload.get("signal_id"),
            strategy_name=payload.get("strategy_name"),
            reservation_id=payload.get("reservation_id"),
            fill_time=payload.get("exchange_time") or payload.get("update_time") or payload.get("timestamp"),
            metadata=merge_metadata(
                payload.get("metadata"),
                {
                    "source_event": "execution.order_filled",
                    "order_status": payload.get("status"),
                    "trigger_type": payload.get("trigger_type"),
                    "tier": payload.get("tier"),
                    "stop_loss": payload.get("stop_loss"),
                    "take_profit": payload.get("take_profit"),
                    "final_risk_amount": payload.get("final_risk_amount"),
                    "final_margin": payload.get("final_margin"),
                    "final_notional": payload.get("final_notional"),
                    "final_leverage": payload.get("final_leverage"),
                },
            ),
            raw=dict(payload),
        )

    def _new_position_state_from_fill(self, fill: OrderFill) -> PositionState:
        leverage = safe_float(fill.metadata.get("final_leverage"))
        margin_used = safe_float(fill.metadata.get("final_margin"), 0.0) or 0.0
        risk_amount = safe_float(fill.metadata.get("final_risk_amount"), 0.0) or 0.0
        notional = safe_float(fill.metadata.get("final_notional"))

        if notional is None:
            notional = calculate_notional(fill.price, fill.quantity)

        tier = self._trade_tier_from_raw(fill.metadata.get("tier"))

        return PositionState(
            exchange=fill.exchange,
            market_type=fill.market_type,
            symbol=fill.symbol,
            side=fill.position_side,
            size=0.0,
            entry_price=None,
            mark_price=fill.price,
            notional_value=0.0,
            leverage=leverage,
            margin_used=margin_used,
            risk_amount=risk_amount,
            stop_loss=safe_float(fill.metadata.get("stop_loss")),
            take_profit=safe_float(fill.metadata.get("take_profit")),
            tier=tier,
            signal_id=fill.signal_id,
            strategy_name=fill.strategy_name,
            reservation_id=fill.reservation_id,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            metadata=merge_metadata(fill.metadata),
        )

    def _new_position_state_from_snapshot(self, snapshot: PositionSnapshot) -> PositionState:
        return PositionState(
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            symbol=snapshot.symbol,
            side=snapshot.side,
            size=0.0,
            entry_price=snapshot.entry_price,
            mark_price=snapshot.mark_price,
            notional_value=0.0,
            leverage=snapshot.leverage,
            margin_used=snapshot.margin_used,
            risk_amount=0.0,
            realized_pnl=0.0,
            unrealized_pnl=snapshot.unrealized_pnl,
            metadata=merge_metadata(snapshot.metadata),
        )

    def _enrich_state_from_fill_metadata(self, state: PositionState, fill: OrderFill) -> None:
        state.signal_id = state.signal_id or fill.signal_id
        state.strategy_name = state.strategy_name or fill.strategy_name
        state.reservation_id = state.reservation_id or fill.reservation_id

        leverage = safe_float(fill.metadata.get("final_leverage"))
        margin_used = safe_float(fill.metadata.get("final_margin"))
        risk_amount = safe_float(fill.metadata.get("final_risk_amount"))
        notional = safe_float(fill.metadata.get("final_notional"))

        if leverage is not None:
            state.leverage = leverage

        if margin_used is not None:
            state.margin_used = margin_used

        if risk_amount is not None:
            state.risk_amount = risk_amount

        if notional is not None:
            state.notional_value = notional

        stop_loss = safe_float(fill.metadata.get("stop_loss"))
        take_profit = safe_float(fill.metadata.get("take_profit"))

        if stop_loss is not None:
            state.stop_loss = stop_loss

        if take_profit is not None:
            state.take_profit = take_profit

        tier = self._trade_tier_from_raw(fill.metadata.get("tier"))
        if tier is not None:
            state.tier = tier

        state.metadata.update(fill.metadata)

    @staticmethod
    def _copy_fill_with_quantity(fill: OrderFill, quantity: float) -> OrderFill:
        """
        Create a delta fill from cumulative order fill.
        """
        ratio = quantity / fill.quantity if fill.quantity > 0 else 0.0

        quote_quantity = (
            fill.quote_quantity * ratio
            if fill.quote_quantity is not None
            else None
        )

        commission = (
            fill.commission * ratio
            if fill.commission is not None
            else None
        )

        realized_pnl = (
            fill.realized_pnl * ratio
            if fill.realized_pnl is not None
            else None
        )

        return OrderFill(
            exchange=fill.exchange,
            market_type=fill.market_type,
            symbol=fill.symbol,
            side=fill.side,
            quantity=quantity,
            price=fill.price,
            order_id=fill.order_id,
            client_order_id=fill.client_order_id,
            trade_id=fill.trade_id,
            position_side=fill.position_side,
            quote_quantity=quote_quantity,
            commission=commission,
            commission_asset=fill.commission_asset,
            realized_pnl=realized_pnl,
            maker=fill.maker,
            execution_id=fill.execution_id,
            signal_id=fill.signal_id,
            strategy_name=fill.strategy_name,
            reservation_id=fill.reservation_id,
            fill_time=fill.fill_time,
            metadata=dict(fill.metadata),
            raw=dict(fill.raw),
        )

    @staticmethod
    def _order_key_from_fill(fill: OrderFill) -> str | None:
        if fill.order_id:
            return f"order:{fill.exchange}:{fill.symbol}:{fill.order_id}"

        if fill.client_order_id:
            return f"client:{fill.exchange}:{fill.symbol}:{fill.client_order_id}"

        if fill.fill_id:
            return f"fill:{fill.exchange}:{fill.symbol}:{fill.fill_id}"

        return None

    def _position_key(self, symbol: str, side: PositionSide | None) -> str:
        symbol_n = normalize_symbol(symbol)
        side_value = side.value if side is not None else "both"
        return f"{self._config.default_exchange}:{self._config.default_market_type}:{symbol_n}:{side_value}"

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
    def _trade_tier_from_raw(value: Any) -> TradeTier | None:
        if value is None:
            return None

        if isinstance(value, TradeTier):
            return value

        normalized = str(value).strip()

        for tier in TradeTier:
            if tier.value == normalized or tier.name == normalized:
                return tier

        return None

    @staticmethod
    def _infer_position_side_from_order_side(
        *,
        order_side: OrderSide,
        reduce_only: bool,
        close_position: bool,
    ) -> PositionSide:
        """
        Best-effort fallback when Binance positionSide is absent.

        In one-way mode, order side alone is ambiguous for reduce-only orders.
        For reduce/close:
        - SELL likely reduces LONG;
        - BUY likely reduces SHORT.

        For risk-increasing/open:
        - BUY opens LONG;
        - SELL opens SHORT.
        """
        if reduce_only or close_position:
            if order_side is OrderSide.SELL:
                return PositionSide.LONG
            return PositionSide.SHORT

        if order_side is OrderSide.BUY:
            return PositionSide.LONG

        return PositionSide.SHORT

    # ------------------------------------------------------------------
    # Event emit helpers
    # ------------------------------------------------------------------

    async def _emit_position_update(self, update: PositionUpdate) -> None:
        if update.update_type == "opened":
            topic = "position.opened"
            priority = EventPriority.CRITICAL
        elif update.update_type == "closed":
            topic = "position.closed"
            priority = EventPriority.CRITICAL
        else:
            topic = "position.updated"
            priority = EventPriority.HIGH

        payload = update.to_event_payload()

        await self._emit_event(topic, payload, priority=priority)

        if self._config.emit_pnl_updates and update.unrealized_pnl != 0:
            await self._emit_event(
                "position.pnl_updated",
                payload,
                priority=EventPriority.NORMAL,
            )

    def _get_exchange_client(self, exchange: str | None) -> BinancePositionClientProtocol:
        exchange_n = normalize_exchange(exchange or self._config.default_exchange)

        client = self._exchange_clients.get(exchange_n)

        if client is None:
            raise ExchangeClientError(f"Exchange client is not configured: {exchange_n}")

        return client

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
                "Failed to emit PositionManager event | topic=%s",
                topic,
            )


__all__ = [
    "BinancePositionClientProtocol",
    "PositionManager",
]