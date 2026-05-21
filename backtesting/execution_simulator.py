from __future__ import annotations

import asyncio
import inspect
import random
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from core.event_bus import EventBus, EventPriority
from core.logger import get_logger

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
    LiquiditySimulationError,
    SimulatedOrderCancelError,
    SimulatedOrderRejectedError,
    SimulatedOrderValidationError,
)
from backtesting.models import (
    BacktestExecutionRecord,
    HistoricalCandle,
    HistoricalOrderBookLevel,
    HistoricalOrderBookSnapshot,
    SimulatedFill,
    SimulatedOrder,
    SerializableMixin,
    new_id,
    timestamp_ms,
    utcnow,
)


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

    started_at: Any | None = None
    stopped_at: Any | None = None
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

    component_name = "ExecutionSimulator"

    def __init__(
        self,
        config: ExecutionSimulatorConfig | None = None,
        *,
        event_bus: EventBus | None = None,
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

        This simulator must never subscribe to signal.generated.
        It starts only after RiskManager emits signal.confirmed or explicit
        risk close/reduce events.
        """

        if self._registered:
            return

        if self.event_bus is None:
            self.logger.warning("ExecutionSimulator has no EventBus; no subscriptions created")
            self._registered = True
            return

        if not isinstance(self.event_bus, EventBus):
            raise ExecutionSimulatorNotReadyError(
                "ExecutionSimulator requires core.event_bus.EventBus for full-pipeline backtesting."
            )

        if self.config.listen_signal_confirmed:
            self._subscribe(self.config.signal_confirmed_topic, self._on_signal_confirmed)

        self._subscribe(self.config.position_close_requested_topic, self._on_position_close_requested)
        self._subscribe(self.config.position_reduce_requested_topic, self._on_position_reduce_requested)
        self._subscribe(self.config.kill_switch_topic, self._on_kill_switch)

        # Market state updates are used only for realistic fill prices and liquidity.
        # They do not bypass strategy/risk.
        self._subscribe("market.candle", self._on_market_candle)
        self._subscribe("market.orderbook", self._on_market_orderbook)

        self._registered = True

        self.logger.info(
            "ExecutionSimulator registered",
            extra={
                "subscriptions": len(self._subscriptions),
                "signal_confirmed_topic": self.config.signal_confirmed_topic,
                "close_topic": self.config.position_close_requested_topic,
                "reduce_topic": self.config.position_reduce_requested_topic,
                "kill_switch_topic": self.config.kill_switch_topic,
            },
        )

    def unregister(self) -> None:
        if self.event_bus is None:
            self._subscriptions.clear()
            self._registered = False
            return

        for subscription in list(self._subscriptions):
            try:
                self.event_bus.unsubscribe(subscription)
            except (RuntimeError, ValueError, TypeError) as exc:
                self.logger.exception(
                    "Failed to unsubscribe ExecutionSimulator handler",
                    extra={"subscription": str(subscription)},
                )

        self._subscriptions.clear()
        self._registered = False

    async def start(self) -> None:
        async with self._lock:
            if self._running:
                return

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
            if not self._running:
                return

            self._running = False
            self.stats_state.status = BacktestStatus.COMPLETED
            self.stats_state.stopped_at = utcnow()

        await self._emit_best_effort(
            "system.backtest.execution_simulator.stopped",
            self.stats(),
        )

        self.unregister()

    async def reset(self) -> None:
        async with self._lock:
            self.orders.clear()
            self.fills.clear()
            self.records.clear()
            self.market_state.clear()
            self.stats_state = ExecutionSimulatorStats()
            self._kill_switch_active = False

    def _subscribe(self, topic: str, handler: Any) -> None:
        if self.event_bus is None:
            return

        wrapped = self._wrap_event_handler(handler)

        try:
            subscription = self.event_bus.subscribe(
                topic,
                wrapped,
                name=f"backtest_execution_on_{topic.replace('.', '_')}",
            )
        except TypeError:
            subscription = self.event_bus.subscribe(
                pattern=topic,
                handler=wrapped,
                name=f"backtest_execution_on_{topic.replace('.', '_')}",
            )

        self._subscriptions.append(subscription)

    @staticmethod
    def _wrap_event_handler(handler: Any) -> Any:
        async def _wrapped(event_or_payload: Any) -> None:
            payload = ExecutionSimulator._payload_from_event_or_dict(event_or_payload)
            result = handler(payload)
            if inspect.isawaitable(result):
                await result

        return _wrapped

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_signal_confirmed(self, payload: dict[str, Any]) -> None:
        """
        Handle risk-approved signal.

        This is the only signal entrypoint. ExecutionSimulator must not listen
        to signal.generated.
        """

        if not self.config.listen_signal_confirmed:
            return

        self._ensure_running()
        self.stats_state.signal_confirmed_events += 1

        request = self._build_order_request_from_signal_confirmed(payload)
        await self.submit_order(request)

    async def _on_position_close_requested(self, payload: dict[str, Any]) -> None:
        """
        Handle risk-requested position close.
        """

        self._ensure_running()
        self.stats_state.close_requested_events += 1

        request = self._build_order_request_from_close_request(payload)
        await self.submit_order(request)

    async def _on_position_reduce_requested(self, payload: dict[str, Any]) -> None:
        """
        Handle risk-requested position reduce.
        """

        self._ensure_running()
        self.stats_state.reduce_requested_events += 1

        request = self._build_order_request_from_reduce_request(payload)
        await self.submit_order(request)

    async def _on_kill_switch(self, payload: dict[str, Any]) -> None:
        """
        Activate simulated kill switch and cancel active orders.
        """

        self.stats_state.kill_switch_events += 1
        self._kill_switch_active = True

        reason = str(payload.get("reason") or "risk_kill_switch")

        for order in list(self.orders.values()):
            if not order.is_active:
                continue
            await self.cancel_order(order.order_id, reason=reason)

    async def _on_market_candle(self, payload: dict[str, Any]) -> None:
        """
        Update market state from raw market.candle.
        """

        try:
            candle = self._payload_to_candle(payload)
        except (KeyError, TypeError, ValueError, SimulatedOrderValidationError):
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
        Update market state from raw market.orderbook.
        """

        try:
            orderbook = self._payload_to_orderbook(payload)
        except (KeyError, TypeError, ValueError, SimulatedOrderValidationError):
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
        Create, submit, accept/reject and possibly fill a simulated order.
        """

        self._ensure_running()

        if self._kill_switch_active and not request.reduce_only and not request.close_position:
            order = self._create_order(request)
            order.mark_rejected(
                OrderRejectionReason.KILL_SWITCH_ACTIVE,
                message="Execution blocked by simulated kill switch.",
                timestamp_ms_value=self._now_ms(),
            )
            self.orders[order.order_id] = order
            self.stats_state.orders_created += 1
            self.stats_state.orders_rejected += 1
            await self._emit_order_event(self.config.order_rejected_topic, order)
            return order

        request.validate()
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

        except (ExecutionSimulationError, LiquiditySimulationError, SimulatedOrderValidationError, SimulatedOrderRejectedError, ValueError, TypeError) as exc:
            order.status = SimulatedOrderStatus.FAILED
            order.rejection_message = str(exc)
            self.stats_state.orders_failed += 1
            self.stats_state.last_error = str(exc)

            await self._emit_order_event(self.config.order_failed_topic, order)

            if isinstance(exc, SimulatedOrderRejectedError):
                raise

            raise ExecutionSimulationError(
                "Failed to simulate order execution.",
                details={
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            ) from exc

    async def cancel_order(self, order_id: str, *, reason: str = "cancel_requested") -> SimulatedOrder:
        order = self.orders.get(order_id)

        if order is None:
            raise SimulatedOrderCancelError(
                "Cannot cancel unknown simulated order.",
                details={"order_id": order_id},
            )

        if order.is_terminal:
            return order

        order.status = SimulatedOrderStatus.CANCELLED
        order.cancelled_at_ms = self._now_ms()
        order.metadata["cancel_reason"] = reason

        self.stats_state.orders_cancelled += 1
        await self._emit_order_event(self.config.order_cancelled_topic, order)

        return order

    def _create_order(self, request: SimulatedOrderRequest) -> SimulatedOrder:
        now_ms = self._now_ms()

        return SimulatedOrder(
            order_id=new_id("sim_order"),
            client_order_id=self._client_order_id(request),
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
                "intended_price": intended_price,
                "source_order_status": order.status.value,
            },
        )

        order.apply_fill(fill)

        self.fills.append(fill)
        self.stats_state.fills_created += 1
        self.stats_state.total_fees += fill.fee
        self.stats_state.total_slippage += fill.slippage

        self._update_average_slippage(fill)
        self._update_average_latency(order)

        if order.status == SimulatedOrderStatus.FILLED:
            self.stats_state.orders_filled += 1
            await self._emit_fill_event(self.config.order_filled_topic, order, fill)
        elif order.status == SimulatedOrderStatus.PARTIALLY_FILLED:
            self.stats_state.orders_partially_filled += 1
            await self._emit_fill_event(self.config.order_partially_filled_topic, order, fill)

    # ------------------------------------------------------------------
    # Validation / simulation rules
    # ------------------------------------------------------------------

    def _check_rejection(self, order: SimulatedOrder) -> OrderRejectionReason:
        order_type = order.order_type.lower()

        if self._kill_switch_active and not order.reduce_only and not order.close_position:
            return OrderRejectionReason.KILL_SWITCH_ACTIVE

        if order_type == "market" and not self.config.allow_market_orders:
            return OrderRejectionReason.INVALID_ORDER

        if order_type in {"limit", "stop_limit"} and not self.config.allow_limit_orders:
            return OrderRejectionReason.INVALID_ORDER

        if order_type in {"stop", "stop_market", "stop_limit"} and not self.config.allow_stop_orders:
            return OrderRejectionReason.INVALID_ORDER

        if order.reduce_only and not self.config.allow_reduce_only:
            return OrderRejectionReason.INVALID_ORDER

        state = self._get_market_state(order)

        if self.config.reject_if_no_price and state is None:
            return OrderRejectionReason.INVALID_ORDER

        if self.config.reject_if_no_price and state is not None and state.last_price is None:
            return OrderRejectionReason.INVALID_ORDER

        if self.config.reject_if_no_liquidity and not self._has_liquidity(order):
            return OrderRejectionReason.INSUFFICIENT_LIQUIDITY

        if self.config.reject_if_price_outside_candle and not self._price_is_possible(order):
            return OrderRejectionReason.PRICE_OUT_OF_RANGE

        return OrderRejectionReason.NONE

    def _calculate_fill_quantity(self, order: SimulatedOrder) -> float:
        if self.config.liquidity_model == LiquidityModel.UNLIMITED:
            return order.quantity

        state = self._get_market_state(order)

        if state is None or state.last_candle is None:
            if self.config.reject_if_no_liquidity:
                return 0.0
            return order.quantity

        candle_volume = max(0.0, float(state.last_candle.volume))
        max_participation = candle_volume * (self.config.max_volume_participation_pct / 100.0)

        if max_participation <= 0:
            if self.config.reject_if_no_liquidity:
                return 0.0
            return order.quantity

        fill_quantity = min(order.quantity, max_participation)

        if (
            self.config.allow_partial_fills
            and self.config.partial_fill_probability > 0
            and fill_quantity < order.quantity
            and self.random.random() < self.config.partial_fill_probability
        ):
            fill_quantity = max(
                order.quantity * self.config.min_fill_ratio,
                fill_quantity,
            )

        if not self.config.allow_partial_fills and fill_quantity < order.quantity:
            return 0.0

        return max(0.0, min(order.quantity, fill_quantity))

    def _resolve_intended_price(self, order: SimulatedOrder) -> float:
        if order.price and order.price > 0:
            return float(order.price)

        state = self._get_market_state(order)

        if state is None:
            raise LiquiditySimulationError(
                "Cannot resolve intended price without market state.",
                details={"order_id": order.order_id, "symbol": order.symbol},
            )

        if state.last_price is not None and state.last_price > 0:
            return float(state.last_price)

        if state.bid is not None and state.ask is not None:
            return (state.bid + state.ask) / 2.0

        raise LiquiditySimulationError(
            "Cannot resolve intended price.",
            details={"order_id": order.order_id, "symbol": order.symbol},
        )

    def _resolve_fill_price(
        self,
        order: SimulatedOrder,
        intended_price: float,
        quantity: float,
    ) -> float:
        state = self._get_market_state(order)

        if self.config.fill_model == FillModel.INSTANT:
            return intended_price

        if self.config.fill_model == FillModel.NEXT_CANDLE_OPEN:
            if state is not None and state.last_candle is not None:
                return float(state.last_candle.open)
            return intended_price

        if self.config.fill_model == FillModel.NEXT_CANDLE_CLOSE:
            if state is not None and state.last_candle is not None:
                return float(state.last_candle.close)
            return intended_price

        if self.config.fill_model == FillModel.OHLC_PATH:
            return self._resolve_candle_path_price(order, intended_price)

        return intended_price

    def _resolve_candle_path_price(self, order: SimulatedOrder, intended_price: float) -> float:
        state = self._get_market_state(order)

        if state is None or state.last_candle is None:
            return intended_price

        candle = state.last_candle

        if self.config.candle_execution_path == CandleExecutionPath.OPEN_HIGH_LOW_CLOSE:
            return candle.high if self._is_buy(order.side) else candle.low

        if self.config.candle_execution_path == CandleExecutionPath.OPEN_LOW_HIGH_CLOSE:
            return candle.low if self._is_buy(order.side) else candle.high

        if self.config.candle_execution_path == CandleExecutionPath.OPTIMISTIC:
            return candle.low if self._is_buy(order.side) else candle.high

        if self.config.candle_execution_path == CandleExecutionPath.RANDOMIZED:
            return self.random.choice([candle.open, candle.high, candle.low, candle.close])

        # Conservative default.
        return candle.high if self._is_buy(order.side) else candle.low

    def _has_liquidity(self, order: SimulatedOrder) -> bool:
        if self.config.liquidity_model == LiquidityModel.UNLIMITED:
            return True

        state = self._get_market_state(order)

        if state is None:
            return not self.config.reject_if_no_liquidity

        if state.last_candle is None:
            return not self.config.reject_if_no_liquidity

        return state.last_candle.volume > 0

    def _price_is_possible(self, order: SimulatedOrder) -> bool:
        if order.order_type.lower() == "market":
            return True

        price = order.price or order.stop_price

        if price is None:
            return True

        state = self._get_market_state(order)

        if state is None or state.last_candle is None:
            return not self.config.reject_if_price_outside_candle

        candle = state.last_candle
        return candle.low <= price <= candle.high

    # ------------------------------------------------------------------
    # Builders from risk payloads
    # ------------------------------------------------------------------

    def _build_order_request_from_signal_confirmed(
        self,
        payload: dict[str, Any],
    ) -> SimulatedOrderRequest:
        intent = self._extract_nested(
            payload,
            ["execution_intent", "intent", "execution", "risk_decision", "decision"],
            default={},
        )
        signal = self._extract_nested(
            payload,
            ["signal", "strategy_signal"],
            default={},
        )
        plan = self._extract_nested(
            payload,
            ["execution_plan", "plan", "order"],
            default={},
        )

        symbol = str(
            self._first_value(intent, plan, signal, payload, keys=["symbol"], default="")
        )
        side = str(
            self._first_value(intent, plan, signal, payload, keys=["side", "position_side"], default="")
        )

        order_type = str(
            self._first_value(
                plan,
                intent,
                payload,
                keys=["order_type", "entry_type", "type"],
                default="market",
            )
        )

        quantity = self._float_first(
            intent,
            plan,
            payload,
            keys=[
                "final_size",
                "quantity",
                "qty",
                "size",
                "position_size",
                "base_quantity",
            ],
            default=0.0,
        )

        request = SimulatedOrderRequest(
            signal_id=self._optional_str(self._first_value(signal, intent, payload, keys=["signal_id", "id"], default=None)),
            strategy_name=self._optional_str(self._first_value(signal, intent, payload, keys=["strategy_name", "strategy"], default=None)),
            exchange=str(
                self._first_value(intent, payload, keys=["exchange"], default=self.config.exchange)
            ),
            market_type=str(
                self._first_value(intent, payload, keys=["market_type"], default=self.config.market_type)
            ),
            symbol=symbol,
            side=self._normalize_order_side(side),
            order_type=self._normalize_order_type(order_type),
            quantity=quantity,
            price=self._optional_float_first(plan, intent, payload, keys=["price", "entry_price", "limit_price"]),
            stop_price=self._optional_float_first(plan, intent, payload, keys=["stop_price", "trigger_price"]),
            reduce_only=bool(
                self._first_value(plan, intent, payload, keys=["reduce_only"], default=False)
            ),
            close_position=bool(
                self._first_value(plan, intent, payload, keys=["close_position"], default=False)
            ),
            time_in_force=self._optional_str(self._first_value(plan, intent, payload, keys=["time_in_force", "tif"], default=None)),
            leverage=self._optional_float_first(intent, payload, keys=["final_leverage", "leverage"]),
            source_event=self.config.signal_confirmed_topic,
            source_payload=payload,
            metadata={
                "run_id": payload.get("run_id"),
                "reservation_id": self._first_value(intent, payload, keys=["reservation_id"], default=None),
                "risk_mode": self._first_value(intent, payload, keys=["risk_mode"], default=None),
                "final_risk_amount": self._first_value(intent, payload, keys=["final_risk_amount", "risk_amount"], default=None),
                "final_margin": self._first_value(intent, payload, keys=["final_margin", "margin"], default=None),
                "final_notional": self._first_value(intent, payload, keys=["final_notional", "notional"], default=None),
                "final_tier": self._first_value(intent, payload, keys=["final_tier", "tier"], default=None),
                "stop_loss": self._first_value(plan, intent, signal, payload, keys=["stop_loss"], default=None),
                "take_profit": self._first_value(plan, intent, signal, payload, keys=["take_profit"], default=None),
                "entry_price": self._first_value(plan, intent, signal, payload, keys=["entry_price", "price"], default=None),
                "risk_decision_payload": dict(payload),
            },
        )
        request.validate()
        return request

    def _build_order_request_from_close_request(
        self,
        payload: dict[str, Any],
    ) -> SimulatedOrderRequest:
        symbol = str(payload.get("symbol") or "")
        side = self._opposite_side(str(payload.get("side") or payload.get("position_side") or ""))
        quantity = float(
            payload.get("quantity")
            or payload.get("size")
            or payload.get("position_size")
            or payload.get("final_size")
            or 0.0
        )

        request = SimulatedOrderRequest(
            signal_id=self._optional_str(payload.get("signal_id")),
            strategy_name=self._optional_str(payload.get("strategy_name")),
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
                "position_id": payload.get("position_id"),
            },
        )
        request.validate()
        return request

    def _build_order_request_from_reduce_request(
        self,
        payload: dict[str, Any],
    ) -> SimulatedOrderRequest:
        symbol = str(payload.get("symbol") or "")
        side = self._opposite_side(str(payload.get("side") or payload.get("position_side") or ""))
        quantity = float(
            payload.get("reduce_quantity")
            or payload.get("quantity")
            or payload.get("size")
            or payload.get("position_size")
            or 0.0
        )

        request = SimulatedOrderRequest(
            signal_id=self._optional_str(payload.get("signal_id")),
            strategy_name=self._optional_str(payload.get("strategy_name")),
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
                "position_id": payload.get("position_id"),
            },
        )
        request.validate()
        return request

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

    @staticmethod
    def _order_to_event_payload(order: SimulatedOrder) -> dict[str, Any]:
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
            "reservation_id": order.metadata.get("reservation_id"),
            "final_risk_amount": order.metadata.get("final_risk_amount"),
            "risk_amount": order.metadata.get("final_risk_amount"),
            "final_margin": order.metadata.get("final_margin"),
            "margin_used": order.metadata.get("final_margin"),
            "final_notional": order.metadata.get("final_notional"),
            "notional_value": order.metadata.get("final_notional"),
            "final_tier": order.metadata.get("final_tier"),
            "tier": order.metadata.get("final_tier"),
            "stop_loss": order.metadata.get("stop_loss"),
            "take_profit": order.metadata.get("take_profit"),
            "entry_price": order.metadata.get("entry_price"),
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
                **dict(order.metadata),
                "backtest": True,
                "simulated": True,
            },
        }

    def _fill_to_event_payload(
        self,
        order: SimulatedOrder,
        fill: SimulatedFill,
    ) -> dict[str, Any]:
        return {
            **self._order_to_event_payload(order),
            "fill_id": fill.fill_id,
            "fill_price": fill.price,
            "fill_quantity": fill.quantity,
            "fill_notional": fill.notional,
            "fill_fee": fill.fee,
            "fee": fill.fee,
            "fee_asset": fill.fee_asset,
            "fill_slippage": fill.slippage,
            "slippage_bps": fill.slippage_bps,
            "timestamp_ms": fill.timestamp_ms,
            "filled_at_ms": fill.timestamp_ms,
            "liquidity_type": fill.liquidity_type,
            "metadata": {
                **dict(order.metadata),
                **dict(fill.metadata),
                "backtest": True,
                "simulated": True,
            },
        }

    async def _emit(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        if not self.config.emit_execution_events:
            return

        if self.event_bus is None:
            return

        result = self.event_bus.emit(
            topic,
            payload,
            priority=priority,
            source="ExecutionSimulator",
        )

        if inspect.isawaitable(result):
            await result

    async def _emit_best_effort(self, topic: str, payload: dict[str, Any]) -> None:
        try:
            await self._emit(topic, payload, priority=EventPriority.LOW)
        except (RuntimeError, ValueError, TypeError) as exc:
            self.logger.debug(
                "Best-effort execution simulator event failed",
                extra={"topic": topic},
            )

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

        self.records.append(
            BacktestExecutionRecord(
                run_id=order.run_id,
                timestamp_ms=fill.timestamp_ms if fill is not None else self._now_ms(),
                topic=topic,
                order_id=order.order_id,
                fill_id=fill.fill_id if fill is not None else None,
                signal_id=order.signal_id,
                strategy_name=order.strategy_name,
                symbol=order.symbol,
                status=order.status,
                payload=dict(payload),
                metadata={
                    "source": "execution_simulator",
                    "simulated": True,
                },
            )
        )

    # ------------------------------------------------------------------
    # Market payload conversion
    # ------------------------------------------------------------------

    def _payload_to_candle(self, payload: dict[str, Any]) -> HistoricalCandle:
        now_ms = self._now_ms()

        return HistoricalCandle(
            exchange=str(payload.get("exchange") or self.config.exchange),
            symbol=str(payload.get("symbol") or ""),
            market_type=str(payload.get("market_type") or self.config.market_type),
            timeframe=str(payload.get("timeframe") or "1m"),
            timestamp_ms=int(payload.get("timestamp_ms") or payload.get("close_time_ms") or now_ms),
            received_at_ms=int(payload.get("received_at_ms") or payload.get("timestamp_ms") or now_ms),
            open_time_ms=int(payload.get("open_time_ms") or payload.get("timestamp_ms") or now_ms),
            close_time_ms=int(payload.get("close_time_ms") or payload.get("timestamp_ms") or now_ms),
            open=float(payload.get("open") or payload.get("price") or payload.get("close") or 0.0),
            high=float(payload.get("high") or payload.get("price") or payload.get("close") or 0.0),
            low=float(payload.get("low") or payload.get("price") or payload.get("close") or 0.0),
            close=float(payload.get("close") or payload.get("price") or 0.0),
            volume=float(payload.get("volume") or 0.0),
            quote_volume=float(payload.get("quote_volume") or payload.get("notional") or 0.0),
            trades_count=int(payload.get("trades_count") or 0),
            is_closed=bool(payload.get("is_closed", True)),
            source="market_replay",
            metadata=dict(payload.get("metadata") or {}),
        )

    def _payload_to_orderbook(self, payload: dict[str, Any]) -> HistoricalOrderBookSnapshot:
        now_ms = self._now_ms()
        bids = self._levels_from_payload(payload.get("bids") or [])
        asks = self._levels_from_payload(payload.get("asks") or [])

        return HistoricalOrderBookSnapshot(
            exchange=str(payload.get("exchange") or self.config.exchange),
            symbol=str(payload.get("symbol") or ""),
            market_type=str(payload.get("market_type") or self.config.market_type),
            timeframe=str(payload.get("timeframe") or "1m") if payload.get("timeframe") else None,
            timestamp_ms=int(payload.get("timestamp_ms") or now_ms),
            received_at_ms=int(payload.get("received_at_ms") or payload.get("timestamp_ms") or now_ms),
            bids=bids,
            asks=asks,
            sequence=self._optional_int(payload.get("sequence") or payload.get("last_update_id")),
            depth=int(payload.get("depth") or max(len(bids), len(asks))),
            source="market_replay",
            metadata=dict(payload.get("metadata") or {}),
        )

    @staticmethod
    def _levels_from_payload(value: Any) -> list[HistoricalOrderBookLevel]:
        result: list[HistoricalOrderBookLevel] = []

        if not isinstance(value, list):
            return result

        for item in value:
            if isinstance(item, dict):
                price = item.get("price")
                quantity = item.get("quantity") or item.get("qty")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                price = item[0]
                quantity = item[1]
            else:
                continue

            try:
                result.append(
                    HistoricalOrderBookLevel(
                        price=float(price),
                        quantity=float(quantity),
                    )
                )
            except (TypeError, ValueError, SimulatedOrderValidationError):
                continue

        return result

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _get_market_state(self, order: SimulatedOrder) -> ExecutionMarketState | None:
        return self.market_state.get(
            self._market_key(order.exchange, order.market_type, order.symbol)
        )

    @staticmethod
    def _market_key(exchange: str, market_type: str, symbol: str) -> str:
        return f"{str(exchange).lower()}:{str(market_type).lower()}:{str(symbol).upper()}"

    def _now_ms(self) -> int:
        if self.clock is not None:
            try:
                return self.clock.timestamp_ms_or_wall_clock()
            except (RuntimeError, ValueError, TypeError):
                pass
        return timestamp_ms(utcnow())

    async def _advance_latency(self, latency_ms: int) -> None:
        if latency_ms <= 0:
            return

        if self.clock is not None:
            await self.clock.advance_by_async(timedelta(milliseconds=latency_ms))
            return

        await asyncio.sleep(0)

    def _simulate_latency_ms(self) -> int:
        if self.config.latency_model == LatencyModel.NONE:
            return 0

        if self.config.latency_model == LatencyModel.FIXED_MS:
            return int(self.config.fixed_latency_ms)

        if self.config.latency_model == LatencyModel.RANDOM_MS:
            return self.random.randint(
                int(self.config.random_latency_min_ms),
                int(self.config.random_latency_max_ms),
            )

        return 0

    @staticmethod
    def _payload_from_event_or_dict(event_or_payload: Any) -> dict[str, Any]:
        if isinstance(event_or_payload, dict):
            return dict(event_or_payload)

        payload = getattr(event_or_payload, "payload", None)
        if isinstance(payload, dict):
            return dict(payload)

        return {}

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
        except (KeyError, TypeError, ValueError, SimulatedOrderValidationError):
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
        except (KeyError, TypeError, ValueError, SimulatedOrderValidationError):
            return None

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(float(value))
        except (KeyError, TypeError, ValueError, SimulatedOrderValidationError):
            return None

    @staticmethod
    def _normalize_order_type(value: str) -> str:
        normalized = str(value or "market").strip().lower()

        aliases = {
            "market": "market",
            "limit": "limit",
            "stop": "stop",
            "stop_market": "stop_market",
            "stop_limit": "stop_limit",
        }

        return aliases.get(normalized, "market")

    @staticmethod
    def _normalize_order_side(value: str) -> str:
        normalized = str(value or "").strip().lower()

        if normalized in {"long", "buy", "bid"}:
            return "buy"

        if normalized in {"short", "sell", "ask"}:
            return "sell"

        return normalized

    @staticmethod
    def _opposite_side(value: str) -> str:
        normalized = str(value or "").strip().lower()

        if normalized in {"long", "buy", "bid"}:
            return "sell"

        if normalized in {"short", "sell", "ask"}:
            return "buy"

        return "sell"

    @staticmethod
    def _is_buy(side: str) -> bool:
        return str(side).lower() in {"buy", "long", "bid"}

    @staticmethod
    def _client_order_id(request: SimulatedOrderRequest) -> str:
        signal_part = request.signal_id or "manual"
        return f"bt_{signal_part}_{new_id('coid')}"

    def _update_average_slippage(self, fill: SimulatedFill) -> None:
        count = max(1, self.stats_state.fills_created)
        previous_total = self.stats_state.average_slippage_bps * max(0, count - 1)
        self.stats_state.average_slippage_bps = (previous_total + fill.slippage_bps) / count

    def _update_average_latency(self, order: SimulatedOrder) -> None:
        total_orders = max(1, self.stats_state.orders_accepted)
        previous_total = self.stats_state.average_latency_ms * max(0, total_orders - 1)
        self.stats_state.average_latency_ms = (previous_total + order.latency_ms) / total_orders

    def _ensure_running(self) -> None:
        if not self._running:
            raise ExecutionSimulatorNotReadyError("ExecutionSimulator is not running.")

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
                "event_bus_type": self.event_bus.__class__.__name__ if self.event_bus else None,
                "subscriptions": len(self._subscriptions),
            }
        )
        return payload


__all__ = [
    "ExecutionMarketState",
    "ExecutionSimulatorStats",
    "SimulatedOrderRequest",
    "ExecutionSimulator",
]