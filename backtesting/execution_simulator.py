"""
Simulated execution layer for backtesting.

ExecutionSimulator is the offline replacement for live execution during
historical backtests.

It listens to risk-approved events and simulates order submission/fills:

- signal.confirmed
- risk.position_close_requested
- risk.position_reduce_requested
- risk.kill_switch

It emits production-compatible execution events:

- execution.order_submitted
- execution.order_rejected
- execution.order_failed
- execution.order_cancelled
- execution.order_filled
- execution.order_partially_filled

Important:
- It must not listen to signal.generated.
- It must not bypass RiskManager.
- It must not call live exchange clients.
- It must not own position accounting; PositionSimulator listens to fills.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from backtesting.backtest_time import BacktestClock
from backtesting.config import ExecutionSimulatorConfig
from backtesting.cost_models import (
    TradingCostModel,
    calculate_slippage_bps,
)
from backtesting.enums import (
    BacktestStatus,
    CandleExecutionPath,
    FillModel,
    LatencyModel,
    LiquidityModel,
    OrderRejectionReason,
    SimulatedOrderStatus,
)
from backtesting.exceptions import (
    ExecutionSimulationError,
    ExecutionSimulatorNotReadyError,
    FillModelError,
    LiquiditySimulationError,
    SimulatedOrderCancelError,
    SimulatedOrderRejectedError,
    SimulatedOrderStateError,
    SimulatedOrderValidationError,
)
from backtesting.models import (
    BacktestExecutionRecord,
    HistoricalCandle,
    HistoricalOrderBookSnapshot,
    SimulatedFill,
    SimulatedOrder,
    SerializableMixin,
    new_id,
    timestamp_ms,
    utcnow,
)


try:
    from core.logger import get_logger
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


@dataclass(slots=True)
class ExecutionMarketState(SerializableMixin):
    """
    Minimal market state required for execution simulation.
    """

    exchange: str = "binance"
    market_type: str = "usdm_futures"
    symbol: str = ""

    last_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None

    last_candle: HistoricalCandle | None = None
    last_orderbook: HistoricalOrderBookSnapshot | None = None

    updated_at_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_from_candle(self, candle: HistoricalCandle) -> None:
        self.exchange = candle.exchange
        self.market_type = candle.market_type
        self.symbol = candle.symbol
        self.last_price = candle.close
        self.last_candle = candle
        self.updated_at_ms = candle.timestamp_ms

    def update_from_orderbook(self, orderbook: HistoricalOrderBookSnapshot) -> None:
        self.exchange = orderbook.exchange
        self.market_type = orderbook.market_type
        self.symbol = orderbook.symbol
        self.last_orderbook = orderbook
        self.bid = orderbook.best_bid
        self.ask = orderbook.best_ask
        self.spread = orderbook.spread
        self.updated_at_ms = orderbook.timestamp_ms

        if self.bid is not None and self.ask is not None:
            self.last_price = (self.bid + self.ask) / 2.0


@dataclass(slots=True)
class ExecutionSimulatorStats(SerializableMixin):
    """
    Runtime stats for ExecutionSimulator.
    """

    status: BacktestStatus = BacktestStatus.CREATED

    orders_created: int = 0
    orders_submitted: int = 0
    orders_accepted: int = 0
    orders_rejected: int = 0
    orders_failed: int = 0
    orders_cancelled: int = 0
    orders_filled: int = 0
    orders_partially_filled: int = 0

    fills_created: int = 0

    signal_confirmed_events: int = 0
    close_requested_events: int = 0
    reduce_requested_events: int = 0
    kill_switch_events: int = 0

    total_fees: float = 0.0
    total_slippage: float = 0.0
    average_slippage_bps: float = 0.0
    average_latency_ms: float = 0.0

    started_at: datetime | None = None
    stopped_at: datetime | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SimulatedOrderRequest(SerializableMixin):
    """
    Internal normalized order request built from risk-confirmed payloads.
    """

    signal_id: str | None
    strategy_name: str | None

    exchange: str
    market_type: str
    symbol: str

    side: str
    order_type: str
    quantity: float

    price: float | None = None
    stop_price: float | None = None

    reduce_only: bool = False
    close_position: bool = False
    time_in_force: str | None = None
    leverage: float | None = None

    source_event: str = "signal.confirmed"
    source_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.symbol:
            raise SimulatedOrderValidationError("SimulatedOrderRequest.symbol is required.")

        if not self.side:
            raise SimulatedOrderValidationError("SimulatedOrderRequest.side is required.")

        if self.quantity <= 0:
            raise SimulatedOrderValidationError(
                "SimulatedOrderRequest.quantity must be positive.",
                details={"quantity": self.quantity},
            )

        if self.order_type.lower() in {"limit", "stop_limit"} and not self.price:
            raise SimulatedOrderValidationError(
                "Limit order simulation requires price.",
                details={"order_type": self.order_type},
            )

        if self.order_type.lower() in {"stop", "stop_market", "stop_limit"} and not self.stop_price:
            raise SimulatedOrderValidationError(
                "Stop order simulation requires stop_price.",
                details={"order_type": self.order_type},
            )


class ExecutionSimulator:
    """
    Offline execution simulator.

    It converts risk-confirmed trading intents into simulated orders/fills and
    emits execution.* events compatible with the rest of the system.
    """

    def __init__(
        self,
        config: ExecutionSimulatorConfig | None = None,
        *,
        event_bus: Any | None = None,
        clock: BacktestClock | None = None,
        cost_model: TradingCostModel | None = None,
        random_seed: int | None = 42,
        logger_name: str = "backtesting.execution_simulator",
    ) -> None:
        self.config = config or ExecutionSimulatorConfig()
        self.config.validate()

        self.event_bus = event_bus
        self.clock = clock
        self.cost_model = cost_model or TradingCostModel()

        self.logger = get_logger(logger_name)
        self.random = random.Random(random_seed)

        self.orders: dict[str, SimulatedOrder] = {}
        self.fills: list[SimulatedFill] = []
        self.records: list[BacktestExecutionRecord] = []
        self.market_state: dict[str, ExecutionMarketState] = {}

        self.stats_state = ExecutionSimulatorStats()

        self._subscriptions: list[Any] = []
        self._running = False
        self._registered = False
        self._kill_switch_active = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Register EventBus subscriptions.
        """

        if self._registered:
            return

        if self.event_bus is None:
            self._registered = True
            return

        self._subscribe(self.config.signal_confirmed_topic, self._on_signal_confirmed)
        self._subscribe(self.config.position_close_requested_topic, self._on_position_close_requested)
        self._subscribe(self.config.position_reduce_requested_topic, self._on_position_reduce_requested)
        self._subscribe(self.config.kill_switch_topic, self._on_kill_switch)

        # Market state updates are optional but useful for realistic fills.
        self._subscribe("market.candle", self._on_market_candle)
        self._subscribe("market.orderbook", self._on_market_orderbook)

        self._registered = True

    async def start(self) -> None:
        async with self._lock:
            self.register()
            self._running = True
            self.stats_state.status = BacktestStatus.RUNNING
            self.stats_state.started_at = utcnow()

        await self._emit_best_effort(
            "system.backtest.execution_simulator.started",
            self.stats(),
        )

    async def stop(self) -> None:
        async with self._lock:
            self._running = False
            self.stats_state.status = BacktestStatus.COMPLETED
            self.stats_state.stopped_at = utcnow()

        await self._emit_best_effort(
            "system.backtest.execution_simulator.stopped",
            self.stats(),
        )

    async def reset(self) -> None:
        async with self._lock:
            self.orders.clear()
            self.fills.clear()
            self.records.clear()
            self.market_state.clear()
            self.stats_state = ExecutionSimulatorStats()
            self._kill_switch_active = False

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_signal_confirmed(self, payload: dict[str, Any]) -> None:
        """
        Handle risk-approved signal.
        """

        if not self.config.listen_signal_confirmed:
            return

        self._ensure_running()
        self.stats_state.signal_confirmed_events += 1

        request = self._build_order_request_from_signal_confirmed(payload)
        await self.submit_order(request)

    async def _on_position_close_requested(self, payload: dict[str, Any]) -> None:
        """
        Handle risk-requested close.
        """

        self._ensure_running()
        self.stats_state.close_requested_events += 1

        request = self._build_order_request_from_close_request(payload)
        await self.submit_order(request)

    async def _on_position_reduce_requested(self, payload: dict[str, Any]) -> None:
        """
        Handle risk-requested reduce.
        """

        self._ensure_running()
        self.stats_state.reduce_requested_events += 1

        request = self._build_order_request_from_reduce_request(payload)
        await self.submit_order(request)

    async def _on_kill_switch(self, payload: dict[str, Any]) -> None:
        """
        Handle risk kill switch.
        """

        self._ensure_running()
        self.stats_state.kill_switch_events += 1
        self._kill_switch_active = True

        reason = str(payload.get("reason") or "risk.kill_switch")

        for order in list(self.orders.values()):
            if order.is_active:
                await self.cancel_order(order.order_id, reason=reason)

    async def _on_market_candle(self, payload: dict[str, Any]) -> None:
        """
        Update local market state from market.candle.
        """

        try:
            candle = self._payload_to_candle(payload)
        except Exception:
            return

        key = self._market_key(candle.exchange, candle.market_type, candle.symbol)
        state = self.market_state.setdefault(
            key,
            ExecutionMarketState(
                exchange=candle.exchange,
                market_type=candle.market_type,
                symbol=candle.symbol,
            ),
        )
        state.update_from_candle(candle)

    async def _on_market_orderbook(self, payload: dict[str, Any]) -> None:
        """
        Update local market state from market.orderbook.
        """

        try:
            orderbook = self._payload_to_orderbook(payload)
        except Exception:
            return

        key = self._market_key(orderbook.exchange, orderbook.market_type, orderbook.symbol)
        state = self.market_state.setdefault(
            key,
            ExecutionMarketState(
                exchange=orderbook.exchange,
                market_type=orderbook.market_type,
                symbol=orderbook.symbol,
            ),
        )
        state.update_from_orderbook(orderbook)

    # ------------------------------------------------------------------
    # Main order API
    # ------------------------------------------------------------------

    async def submit_order(self, request: SimulatedOrderRequest) -> SimulatedOrder:
        """
        Submit simulated order.

        This emits order_submitted and then either rejected/failed/filled events.
        """

        self._ensure_running()
        request.validate()

        async with self._lock:
            order = self._create_order(request)
            self.orders[order.order_id] = order
            self.stats_state.orders_created += 1

        try:
            await self._submit_order(order)
            await self._simulate_accept_or_reject(order)

            if order.status == SimulatedOrderStatus.REJECTED:
                return order

            await self._simulate_fill(order)
            return order

        except Exception as exc:
            self.stats_state.last_error = str(exc)

            if order.status not in {
                SimulatedOrderStatus.REJECTED,
                SimulatedOrderStatus.CANCELLED,
                SimulatedOrderStatus.FILLED,
                SimulatedOrderStatus.PARTIALLY_FILLED,
            }:
                order.status = SimulatedOrderStatus.FAILED
                self.stats_state.orders_failed += 1
                await self._emit_order_event(self.config.order_failed_topic, order)

            raise

    async def cancel_order(
        self,
        order_id: str,
        *,
        reason: str = "cancelled",
    ) -> SimulatedOrder:
        """
        Cancel active simulated order.
        """

        self._ensure_running()

        order = self.orders.get(order_id)

        if order is None:
            raise SimulatedOrderCancelError(
                "Simulated order not found.",
                details={"order_id": order_id},
            )

        if order.is_terminal:
            raise SimulatedOrderStateError(
                "Cannot cancel terminal simulated order.",
                details={
                    "order_id": order_id,
                    "status": order.status.value,
                },
            )

        order.status = SimulatedOrderStatus.CANCELLED
        order.cancelled_at_ms = self._now_ms()
        order.metadata["cancel_reason"] = reason

        self.stats_state.orders_cancelled += 1
        await self._emit_order_event(self.config.order_cancelled_topic, order)
        return order

    async def cancel_all_open_orders(
        self,
        *,
        symbol: str | None = None,
        reason: str = "cancel_all",
    ) -> int:
        """
        Cancel all active orders, optionally filtered by symbol.
        """

        count = 0

        for order in list(self.orders.values()):
            if not order.is_active:
                continue

            if symbol is not None and order.symbol.upper() != symbol.upper():
                continue

            await self.cancel_order(order.order_id, reason=reason)
            count += 1

        return count

    # ------------------------------------------------------------------
    # Order simulation
    # ------------------------------------------------------------------

    def _create_order(self, request: SimulatedOrderRequest) -> SimulatedOrder:
        now_ms = self._now_ms()
        client_order_id = f"bt_{new_id('cid')}"

        return SimulatedOrder(
            client_order_id=client_order_id,
            run_id=request.metadata.get("run_id"),
            signal_id=request.signal_id,
            strategy_name=request.strategy_name,
            exchange=request.exchange,
            symbol=request.symbol,
            market_type=request.market_type,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
            stop_price=request.stop_price,
            reduce_only=request.reduce_only,
            close_position=request.close_position,
            time_in_force=request.time_in_force,
            leverage=request.leverage,
            submitted_at_ms=now_ms,
            source_payload=request.source_payload,
            metadata={
                **request.metadata,
                "source_event": request.source_event,
            },
        )

    async def _submit_order(self, order: SimulatedOrder) -> None:
        order.mark_submitted(self._now_ms())

        self.stats_state.orders_submitted += 1
        await self._emit_order_event(self.config.market_order_topic, order)

    async def _simulate_accept_or_reject(self, order: SimulatedOrder) -> None:
        rejection_reason = self._check_rejection(order)

        if rejection_reason != OrderRejectionReason.NONE:
            order.mark_rejected(
                rejection_reason,
                message=rejection_reason.value,
                timestamp_ms_value=self._now_ms(),
            )

            self.stats_state.orders_rejected += 1
            await self._emit_order_event(self.config.order_rejected_topic, order)

            if self.config.allow_order_rejections:
                return

            raise SimulatedOrderRejectedError(
                "Simulated order rejected.",
                details={
                    "order_id": order.order_id,
                    "reason": rejection_reason.value,
                },
            )

        latency_ms = self._simulate_latency_ms()
        order.latency_ms = latency_ms

        if latency_ms > 0:
            await self._advance_latency(latency_ms)

        order.mark_accepted(self._now_ms())
        self.stats_state.orders_accepted += 1

    async def _simulate_fill(self, order: SimulatedOrder) -> None:
        fill_quantity = self._calculate_fill_quantity(order)

        if fill_quantity <= 0:
            order.mark_rejected(
                OrderRejectionReason.INSUFFICIENT_LIQUIDITY,
                message="No simulated liquidity available.",
                timestamp_ms_value=self._now_ms(),
            )
            self.stats_state.orders_rejected += 1
            await self._emit_order_event(self.config.order_rejected_topic, order)
            return

        intended_price = self._resolve_intended_price(order)
        fill_price = self._resolve_fill_price(order, intended_price, fill_quantity)

        notional = abs(fill_price * fill_quantity)
        costs = self.cost_model.calculate_trade_costs(
            data=self._build_trade_cost_input(
                order=order,
                intended_price=intended_price,
                fill_price=fill_price,
                quantity=fill_quantity,
                notional=notional,
            )
        )

        fill = SimulatedFill(
            order_id=order.order_id,
            run_id=order.run_id,
            signal_id=order.signal_id,
            exchange=order.exchange,
            symbol=order.symbol,
            market_type=order.market_type,
            side=order.side,
            price=fill_price,
            quantity=fill_quantity,
            notional=notional,
            fee=costs.commission,
            fee_asset="USDT",
            slippage=costs.slippage,
            slippage_bps=calculate_slippage_bps(
                intended_price=intended_price,
                fill_price=fill_price,
            ),
            timestamp_ms=self._now_ms(),
            metadata={
                "strategy_name": order.strategy_name,
                "order_type": order.order_type,
                "reduce_only": order.reduce_only,
                "close_position": order.close_position,
                "costs": costs.to_dict(),
            },
        )

        order.apply_fill(fill)
        self.fills.append(fill)

        self.stats_state.fills_created += 1
        self.stats_state.total_fees += fill.fee
        self.stats_state.total_slippage += fill.slippage
        self._update_average_stats(fill, order)

        if order.status == SimulatedOrderStatus.FILLED:
            self.stats_state.orders_filled += 1
            await self._emit_fill_event(self.config.order_filled_topic, order, fill)
        elif order.status == SimulatedOrderStatus.PARTIALLY_FILLED:
            self.stats_state.orders_partially_filled += 1
            await self._emit_fill_event(self.config.order_partially_filled_topic, order, fill)

    def _check_rejection(self, order: SimulatedOrder) -> OrderRejectionReason:
        if self._kill_switch_active:
            return OrderRejectionReason.KILL_SWITCH_ACTIVE

        if not self.config.allow_market_orders and order.order_type.lower() == "market":
            return OrderRejectionReason.INVALID_ORDER

        if not self.config.allow_limit_orders and order.order_type.lower() == "limit":
            return OrderRejectionReason.INVALID_ORDER

        if not self.config.allow_stop_orders and order.order_type.lower().startswith("stop"):
            return OrderRejectionReason.INVALID_ORDER

        if order.reduce_only and not self.config.allow_reduce_only:
            return OrderRejectionReason.INVALID_ORDER

        if self.config.reject_if_no_price:
            try:
                self._resolve_intended_price(order)
            except Exception:
                return OrderRejectionReason.PRICE_OUT_OF_RANGE

        if self.config.reject_if_no_liquidity:
            available_qty = self._estimate_available_quantity(order)
            if available_qty <= 0:
                return OrderRejectionReason.INSUFFICIENT_LIQUIDITY

        return OrderRejectionReason.NONE

    def _calculate_fill_quantity(self, order: SimulatedOrder) -> float:
        available_qty = self._estimate_available_quantity(order)

        if self.config.liquidity_model == LiquidityModel.UNLIMITED:
            available_qty = order.remaining_quantity

        fill_qty = min(order.remaining_quantity, available_qty)

        if fill_qty <= 0:
            return 0.0

        if self.config.allow_partial_fills:
            if self.config.partial_fill_probability > 0:
                if self.random.random() < self.config.partial_fill_probability:
                    ratio = max(self.config.min_fill_ratio, self.random.random())
                    return max(0.0, min(fill_qty, order.remaining_quantity * ratio))

            if fill_qty < order.remaining_quantity:
                return max(fill_qty, order.remaining_quantity * self.config.min_fill_ratio)

        return order.remaining_quantity if fill_qty >= order.remaining_quantity else fill_qty

    def _estimate_available_quantity(self, order: SimulatedOrder) -> float:
        if self.config.liquidity_model == LiquidityModel.UNLIMITED:
            return order.remaining_quantity

        state = self._get_market_state(order)

        if self.config.liquidity_model == LiquidityModel.CANDLE_VOLUME_PERCENT:
            candle = state.last_candle if state else None

            if candle is None:
                if self.config.reject_if_no_liquidity:
                    return 0.0
                return order.remaining_quantity

            return candle.volume * self.config.max_volume_participation_pct / 100.0

        if self.config.liquidity_model == LiquidityModel.ORDERBOOK_DEPTH:
            orderbook = state.last_orderbook if state else None

            if orderbook is None:
                if self.config.reject_if_no_liquidity:
                    return 0.0
                return order.remaining_quantity

            levels = orderbook.asks if self._is_buy(order.side) else orderbook.bids
            return sum(level.quantity for level in levels)

        if self.config.liquidity_model == LiquidityModel.TRADE_VOLUME_PERCENT:
            candle = state.last_candle if state else None

            if candle is None:
                return 0.0 if self.config.reject_if_no_liquidity else order.remaining_quantity

            return candle.volume * self.config.max_volume_participation_pct / 100.0

        if self.config.liquidity_model == LiquidityModel.PROBABILISTIC:
            probability = max(0.0, min(1.0, self.config.max_volume_participation_pct / 100.0))
            if self.random.random() <= probability:
                return order.remaining_quantity
            return 0.0

        raise LiquiditySimulationError(
            "Unsupported liquidity model.",
            details={"model": self.config.liquidity_model.value},
        )

    def _resolve_intended_price(self, order: SimulatedOrder) -> float:
        order_type = order.order_type.lower()
        state = self._get_market_state(order)

        if order_type in {"limit", "stop_limit"} and order.price is not None:
            return order.price

        if order_type in {"stop", "stop_market"} and order.stop_price is not None:
            return order.stop_price

        if state is None or state.last_price is None:
            if order.price is not None:
                return order.price
            if order.stop_price is not None:
                return order.stop_price

            raise FillModelError(
                "Cannot resolve intended price without market state.",
                details={
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                },
            )

        if self.config.fill_model == FillModel.NEXT_CANDLE_OPEN:
            candle = state.last_candle
            if candle is not None:
                return candle.open

        if self.config.fill_model == FillModel.NEXT_CANDLE_CLOSE:
            candle = state.last_candle
            if candle is not None:
                return candle.close

        if self.config.fill_model == FillModel.VWAP:
            candle = state.last_candle
            if candle is not None:
                return self._estimate_candle_vwap(candle)

        if self.config.fill_model == FillModel.ORDERBOOK_DEPTH:
            if state.bid is not None and state.ask is not None:
                return state.ask if self._is_buy(order.side) else state.bid

        if self.config.fill_model == FillModel.OHLC_PATH:
            candle = state.last_candle
            if candle is not None:
                return self._resolve_ohlc_path_price(order, candle)

        # INSTANT / NEXT_TICK / PROBABILISTIC fallback.
        return state.last_price

    def _resolve_fill_price(
        self,
        order: SimulatedOrder,
        intended_price: float,
        quantity: float,
    ) -> float:
        state = self._get_market_state(order)

        if self.config.fill_model == FillModel.INSTANT:
            return self.cost_model.calculate_fill_price(
                side=order.side,
                intended_price=intended_price,
                quantity=quantity,
                candle=state.last_candle if state else None,
                orderbook=state.last_orderbook if state else None,
                spread=state.spread if state else None,
            )

        if self.config.fill_model == FillModel.NEXT_TICK:
            return self.cost_model.calculate_fill_price(
                side=order.side,
                intended_price=intended_price,
                quantity=quantity,
                candle=state.last_candle if state else None,
                orderbook=state.last_orderbook if state else None,
                spread=state.spread if state else None,
            )

        if self.config.fill_model in {
            FillModel.NEXT_CANDLE_OPEN,
            FillModel.NEXT_CANDLE_CLOSE,
            FillModel.VWAP,
            FillModel.OHLC_PATH,
            FillModel.ORDERBOOK_DEPTH,
            FillModel.PROBABILISTIC,
        }:
            return self.cost_model.calculate_fill_price(
                side=order.side,
                intended_price=intended_price,
                quantity=quantity,
                candle=state.last_candle if state else None,
                orderbook=state.last_orderbook if state else None,
                spread=state.spread if state else None,
            )

        raise FillModelError(
            "Unsupported fill model.",
            details={"fill_model": self.config.fill_model.value},
        )

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------

    def _build_order_request_from_signal_confirmed(
        self,
        payload: dict[str, Any],
    ) -> SimulatedOrderRequest:
        """
        Build order request from risk-approved signal payload.

        Supports several payload shapes:
        - risk decision payload with final_size/final_leverage;
        - execution_intent-like payload;
        - signal payload with execution_plan/entry_plan.
        """

        intent = self._extract_nested(payload, ["execution_intent", "intent", "risk_decision"], default=payload)

        symbol = self._first_value(
            intent,
            payload,
            keys=["symbol", "instrument", "ticker"],
            default="",
        )
        side = self._normalize_side(
            self._first_value(intent, payload, keys=["side", "signal_side", "direction"], default="")
        )

        quantity = self._float_first(
            intent,
            payload,
            keys=["final_size", "quantity", "qty", "size", "position_size"],
            default=0.0,
        )

        entry_plan = self._extract_nested(payload, ["entry_plan", "entry", "execution_plan"], default={})

        order_type = str(
            self._first_value(
                intent,
                entry_plan,
                payload,
                keys=["order_type", "entry_type", "type"],
                default="market",
            )
        ).lower()

        price = self._optional_float_first(intent, entry_plan, payload, keys=["price", "entry_price", "limit_price"])
        stop_price = self._optional_float_first(intent, entry_plan, payload, keys=["stop_price", "trigger_price"])

        if order_type in {"market_entry", "market"}:
            order_type = "market"
        elif order_type in {"limit_entry", "limit"}:
            order_type = "limit"
        elif order_type in {"stop_market", "stop"}:
            order_type = "stop_market"

        return SimulatedOrderRequest(
            signal_id=self._first_value(intent, payload, keys=["signal_id", "id"], default=None),
            strategy_name=self._first_value(intent, payload, keys=["strategy_name", "strategy"], default=None),
            exchange=str(
                self._first_value(intent, payload, keys=["exchange"], default=self.config.exchange)
            ),
            market_type=str(
                self._first_value(intent, payload, keys=["market_type"], default=self.config.market_type)
            ),
            symbol=str(symbol),
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            reduce_only=bool(self._first_value(intent, payload, keys=["reduce_only"], default=False)),
            close_position=bool(self._first_value(intent, payload, keys=["close_position"], default=False)),
            time_in_force=self._first_value(intent, entry_plan, payload, keys=["time_in_force"], default=None),
            leverage=self._optional_float_first(intent, payload, keys=["final_leverage", "leverage"]),
            source_event=self.config.signal_confirmed_topic,
            source_payload=payload,
            metadata={
                "run_id": payload.get("run_id"),
                "reservation_id": self._first_value(intent, payload, keys=["reservation_id"], default=None),
                "risk_mode": self._first_value(intent, payload, keys=["risk_mode"], default=None),
            },
        )

    def _build_order_request_from_close_request(
        self,
        payload: dict[str, Any],
    ) -> SimulatedOrderRequest:
        symbol = str(payload.get("symbol") or "")
        side = self._opposite_side(str(payload.get("side") or payload.get("position_side") or ""))
        quantity = float(payload.get("quantity") or payload.get("size") or payload.get("position_size") or 0.0)

        return SimulatedOrderRequest(
            signal_id=payload.get("signal_id"),
            strategy_name=payload.get("strategy_name"),
            exchange=str(payload.get("exchange") or self.config.exchange),
            market_type=str(payload.get("market_type") or self.config.market_type),
            symbol=symbol,
            side=side,
            order_type=str(payload.get("order_type") or "market"),
            quantity=quantity,
            reduce_only=True,
            close_position=True,
            source_event=self.config.position_close_requested_topic,
            source_payload=payload,
            metadata={
                "run_id": payload.get("run_id"),
                "reason": payload.get("reason"),
            },
        )

    def _build_order_request_from_reduce_request(
        self,
        payload: dict[str, Any],
    ) -> SimulatedOrderRequest:
        symbol = str(payload.get("symbol") or "")
        side = self._opposite_side(str(payload.get("side") or payload.get("position_side") or ""))
        quantity = float(payload.get("reduce_quantity") or payload.get("quantity") or payload.get("size") or 0.0)

        return SimulatedOrderRequest(
            signal_id=payload.get("signal_id"),
            strategy_name=payload.get("strategy_name"),
            exchange=str(payload.get("exchange") or self.config.exchange),
            market_type=str(payload.get("market_type") or self.config.market_type),
            symbol=symbol,
            side=side,
            order_type=str(payload.get("order_type") or "market"),
            quantity=quantity,
            reduce_only=True,
            close_position=False,
            source_event=self.config.position_reduce_requested_topic,
            source_payload=payload,
            metadata={
                "run_id": payload.get("run_id"),
                "reason": payload.get("reason"),
            },
        )

    def _build_trade_cost_input(
        self,
        *,
        order: SimulatedOrder,
        intended_price: float,
        fill_price: float,
        quantity: float,
        notional: float,
    ) -> Any:
        from .cost_models import TradeCostInput

        state = self._get_market_state(order)

        return TradeCostInput(
            side=order.side,
            intended_price=intended_price,
            fill_price=fill_price,
            quantity=quantity,
            notional=notional,
            is_maker=order.order_type.lower() == "limit",
            candle=state.last_candle if state else None,
            orderbook=state.last_orderbook if state else None,
            metadata={
                "order_id": order.order_id,
                "signal_id": order.signal_id,
                "strategy_name": order.strategy_name,
            },
        )

    # ------------------------------------------------------------------
    # Event payload emitters
    # ------------------------------------------------------------------

    async def _emit_order_event(self, topic: str, order: SimulatedOrder) -> None:
        payload = self._order_to_event_payload(order)
        await self._emit(topic, payload)

        self._record_execution_event(
            topic=topic,
            order=order,
            fill=None,
            payload=payload,
        )

    async def _emit_fill_event(
        self,
        topic: str,
        order: SimulatedOrder,
        fill: SimulatedFill,
    ) -> None:
        payload = self._fill_to_event_payload(order, fill)
        await self._emit(topic, payload)

        self._record_execution_event(
            topic=topic,
            order=order,
            fill=fill,
            payload=payload,
        )

    def _order_to_event_payload(self, order: SimulatedOrder) -> dict[str, Any]:
        return {
            "order_id": order.order_id,
            "client_order_id": order.client_order_id,
            "run_id": order.run_id,
            "signal_id": order.signal_id,
            "strategy_name": order.strategy_name,
            "exchange": order.exchange,
            "market_type": order.market_type,
            "symbol": order.symbol,
            "side": order.side,
            "order_type": order.order_type,
            "status": order.status.value,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "remaining_quantity": order.remaining_quantity,
            "price": order.price,
            "stop_price": order.stop_price,
            "average_fill_price": order.average_fill_price,
            "reduce_only": order.reduce_only,
            "close_position": order.close_position,
            "time_in_force": order.time_in_force,
            "leverage": order.leverage,
            "submitted_at_ms": order.submitted_at_ms,
            "accepted_at_ms": order.accepted_at_ms,
            "filled_at_ms": order.filled_at_ms,
            "cancelled_at_ms": order.cancelled_at_ms,
            "rejected_at_ms": order.rejected_at_ms,
            "rejection_reason": order.rejection_reason.value,
            "rejection_message": order.rejection_message,
            "fees": order.fees,
            "slippage": order.slippage,
            "latency_ms": order.latency_ms,
            "source": "execution_simulator",
            "metadata": {
                **order.metadata,
                "backtest": True,
            },
        }

    def _fill_to_event_payload(
        self,
        order: SimulatedOrder,
        fill: SimulatedFill,
    ) -> dict[str, Any]:
        payload = self._order_to_event_payload(order)
        payload.update(
            {
                "fill_id": fill.fill_id,
                "fill_price": fill.price,
                "fill_quantity": fill.quantity,
                "fill_notional": fill.notional,
                "fee": fill.fee,
                "fee_asset": fill.fee_asset,
                "fill_slippage": fill.slippage,
                "fill_slippage_bps": fill.slippage_bps,
                "liquidity_type": fill.liquidity_type,
                "timestamp_ms": fill.timestamp_ms,
                "filled_at_ms": fill.timestamp_ms,
                "fill": fill.to_dict(),
            }
        )
        return payload

    async def _emit(self, topic: str, payload: dict[str, Any]) -> None:
        if self.event_bus is None:
            return

        emit = getattr(self.event_bus, "emit", None) or getattr(self.event_bus, "publish", None)

        if emit is None:
            raise ExecutionSimulationError(
                "EventBus does not expose emit() or publish().",
                details={"event_bus_type": self.event_bus.__class__.__name__},
            )

        result = emit(topic, payload)

        if hasattr(result, "__await__"):
            await result

    async def _emit_best_effort(self, topic: str, payload: dict[str, Any]) -> None:
        try:
            await self._emit(topic, payload)
        except Exception as exc:
            self.logger.warning("Failed to emit %s: %s", topic, exc)

    def _record_execution_event(
        self,
        *,
        topic: str,
        order: SimulatedOrder,
        fill: SimulatedFill | None,
        payload: dict[str, Any],
    ) -> None:
        if not self.config.record_orders and fill is None:
            return

        if not self.config.record_fills and fill is not None:
            return

        record = BacktestExecutionRecord(
            run_id=order.run_id,
            timestamp_ms=self._now_ms(),
            topic=topic,
            order_id=order.order_id,
            fill_id=fill.fill_id if fill else None,
            signal_id=order.signal_id,
            strategy_name=order.strategy_name,
            symbol=order.symbol,
            status=order.status,
            payload=payload,
            metadata={"source": "execution_simulator"},
        )
        self.records.append(record)

    # ------------------------------------------------------------------
    # Market state conversion
    # ------------------------------------------------------------------

    def _payload_to_candle(self, payload: dict[str, Any]) -> HistoricalCandle:
        return HistoricalCandle(
            exchange=str(payload.get("exchange") or self.config.exchange),
            symbol=str(payload.get("symbol") or ""),
            market_type=str(payload.get("market_type") or self.config.market_type),
            timeframe=str(payload.get("timeframe") or "1m"),
            timestamp_ms=int(payload.get("timestamp_ms") or payload.get("close_time_ms") or self._now_ms()),
            received_at_ms=int(payload.get("received_at_ms") or payload.get("timestamp_ms") or self._now_ms()),
            open_time_ms=int(payload.get("open_time_ms") or payload.get("timestamp_ms") or self._now_ms()),
            close_time_ms=int(payload.get("close_time_ms") or payload.get("timestamp_ms") or self._now_ms()),
            open=float(payload.get("open") or 0.0),
            high=float(payload.get("high") or 0.0),
            low=float(payload.get("low") or 0.0),
            close=float(payload.get("close") or 0.0),
            volume=float(payload.get("volume") or 0.0),
            quote_volume=float(payload.get("quote_volume") or 0.0),
            trades_count=int(payload.get("trades_count") or 0),
            is_closed=bool(payload.get("is_closed", True)),
            source="market_replay",
            metadata=dict(payload.get("metadata") or {}),
        )

    def _payload_to_orderbook(self, payload: dict[str, Any]) -> HistoricalOrderBookSnapshot:
        from .models import HistoricalOrderBookLevel

        bids = [
            HistoricalOrderBookLevel(price=float(level[0]), quantity=float(level[1]))
            for level in payload.get("bids", [])
        ]
        asks = [
            HistoricalOrderBookLevel(price=float(level[0]), quantity=float(level[1]))
            for level in payload.get("asks", [])
        ]

        return HistoricalOrderBookSnapshot(
            exchange=str(payload.get("exchange") or self.config.exchange),
            symbol=str(payload.get("symbol") or ""),
            market_type=str(payload.get("market_type") or self.config.market_type),
            timestamp_ms=int(payload.get("timestamp_ms") or self._now_ms()),
            received_at_ms=int(payload.get("received_at_ms") or payload.get("timestamp_ms") or self._now_ms()),
            bids=bids,
            asks=asks,
            sequence=payload.get("sequence"),
            depth=int(payload.get("depth") or max(len(bids), len(asks))),
            source="market_replay",
            metadata=dict(payload.get("metadata") or {}),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _subscribe(self, topic: str, handler: Any) -> None:
        if self.event_bus is None:
            return

        subscribe = getattr(self.event_bus, "subscribe", None)

        if subscribe is None:
            return

        result = subscribe(topic, handler)
        self._subscriptions.append(result)

    def _ensure_running(self) -> None:
        if not self._running:
            raise ExecutionSimulatorNotReadyError(
                "ExecutionSimulator is not running. Call start() first."
            )

    def _now_ms(self) -> int:
        if self.clock is not None and self.clock.started:
            return self.clock.timestamp_ms()
        return timestamp_ms(utcnow())

    async def _advance_latency(self, latency_ms: int) -> None:
        if latency_ms <= 0:
            return

        if self.clock is not None and self.clock.started:
            # During deterministic replay we usually do not want latency to move
            # beyond current market event timestamp unless explicitly allowed.
            # Store latency on order, but avoid forcing wall-clock sleeps.
            return

        await asyncio.sleep(latency_ms / 1000.0)

    def _simulate_latency_ms(self) -> int:
        if self.config.latency_model == LatencyModel.NONE:
            return 0

        if self.config.latency_model == LatencyModel.FIXED_MS:
            return self.config.fixed_latency_ms

        if self.config.latency_model == LatencyModel.RANDOM_MS:
            return self.random.randint(
                self.config.random_latency_min_ms,
                self.config.random_latency_max_ms,
            )

        if self.config.latency_model == LatencyModel.DISTRIBUTION:
            low = self.config.random_latency_min_ms
            high = self.config.random_latency_max_ms
            if high <= low:
                return low
            return int(self.random.triangular(low, high, (low + high) / 2))

        return 0

    def _update_average_stats(self, fill: SimulatedFill, order: SimulatedOrder) -> None:
        fills_count = max(1, self.stats_state.fills_created)

        previous_slippage_total = self.stats_state.average_slippage_bps * (fills_count - 1)
        self.stats_state.average_slippage_bps = (
            previous_slippage_total + fill.slippage_bps
        ) / fills_count

        previous_latency_total = self.stats_state.average_latency_ms * (fills_count - 1)
        self.stats_state.average_latency_ms = (
            previous_latency_total + order.latency_ms
        ) / fills_count

    def _get_market_state(self, order: SimulatedOrder) -> ExecutionMarketState | None:
        return self.market_state.get(
            self._market_key(order.exchange, order.market_type, order.symbol)
        )

    @staticmethod
    def _market_key(exchange: str, market_type: str, symbol: str) -> str:
        return f"{exchange.lower()}:{market_type.lower()}:{symbol.upper()}"

    @staticmethod
    def _is_buy(side: str) -> bool:
        return side.lower() in {"buy", "long"}

    @staticmethod
    def _normalize_side(side: str) -> str:
        value = str(side).lower()

        if value in {"buy", "long", "bullish"}:
            return "buy"

        if value in {"sell", "short", "bearish"}:
            return "sell"

        return value

    @staticmethod
    def _opposite_side(side: str) -> str:
        value = side.lower()

        if value in {"buy", "long"}:
            return "sell"

        if value in {"sell", "short"}:
            return "buy"

        # If position side is absent, default close side must be explicitly
        # supplied by caller. Empty side will fail validation.
        return ""

    @staticmethod
    def _estimate_candle_vwap(candle: HistoricalCandle) -> float:
        if candle.volume <= 0:
            return candle.close
        return (candle.open + candle.high + candle.low + candle.close) / 4.0

    def _resolve_ohlc_path_price(self, order: SimulatedOrder, candle: HistoricalCandle) -> float:
        if self.config.candle_execution_path == CandleExecutionPath.OPEN_HIGH_LOW_CLOSE:
            return candle.high if self._is_buy(order.side) else candle.low

        if self.config.candle_execution_path == CandleExecutionPath.OPEN_LOW_HIGH_CLOSE:
            return candle.low if self._is_buy(order.side) else candle.high

        if self.config.candle_execution_path == CandleExecutionPath.OPTIMISTIC:
            return candle.low if self._is_buy(order.side) else candle.high

        if self.config.candle_execution_path == CandleExecutionPath.RANDOMIZED:
            values = [candle.open, candle.high, candle.low, candle.close]
            return self.random.choice(values)

        # Conservative default.
        return candle.high if self._is_buy(order.side) else candle.low

    @staticmethod
    def _extract_nested(
        payload: dict[str, Any],
        keys: list[str],
        *,
        default: Any,
    ) -> Any:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        return default

    @staticmethod
    def _first_value(
        *payloads: dict[str, Any],
        keys: list[str],
        default: Any = None,
    ) -> Any:
        for payload in payloads:
            if not isinstance(payload, dict):
                continue

            for key in keys:
                value = payload.get(key)
                if value is not None:
                    return value

        return default

    @classmethod
    def _float_first(
        cls,
        *payloads: dict[str, Any],
        keys: list[str],
        default: float,
    ) -> float:
        value = cls._first_value(*payloads, keys=keys, default=default)

        try:
            return float(value)
        except Exception:
            return default

    @classmethod
    def _optional_float_first(
        cls,
        *payloads: dict[str, Any],
        keys: list[str],
    ) -> float | None:
        value = cls._first_value(*payloads, keys=keys, default=None)

        if value is None:
            return None

        try:
            return float(value)
        except Exception:
            return None

    def stats(self) -> dict[str, Any]:
        payload = self.stats_state.to_dict()
        payload.update(
            {
                "running": self._running,
                "registered": self._registered,
                "kill_switch_active": self._kill_switch_active,
                "open_orders": len([order for order in self.orders.values() if order.is_active]),
                "total_orders": len(self.orders),
                "total_fills": len(self.fills),
                "market_states": len(self.market_state),
            }
        )
        return payload


__all__ = [
    "ExecutionMarketState",
    "ExecutionSimulatorStats",
    "SimulatedOrderRequest",
    "ExecutionSimulator",
]