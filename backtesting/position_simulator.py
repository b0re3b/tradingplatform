"""
Simulated position accounting for backtesting.

PositionSimulator is the offline replacement for live PositionManager during
historical backtests.

It listens to simulated execution fills:

- execution.order_filled
- execution.order_partially_filled

And emits production-compatible position events:

- position.opened
- position.updated
- position.closed
- position.liquidated

Important:
- It does not generate strategy signals.
- It does not approve risk.
- It does not submit orders.
- It only accounts positions, balance, equity, funding, liquidation and trade lifecycle.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.event_bus import EventBus, EventPriority
from core.logger import get_logger

from backtesting.backtest_time import BacktestClock
from backtesting.config import PositionSimulatorConfig
from backtesting.cost_models import (
    TradingCostModel,
    calculate_return_pct,
    calculate_r_multiple,
)
from backtesting.enums import (
    BacktestStatus,
    PnLAccountingMode,
    PositionAccountingMode,
    SimulatedPositionStatus,
    TradeOutcome,
)
from backtesting.exceptions import (
    EquityCalculationError,
    PositionAccountingError,
    PositionSimulatorNotReadyError,
    SimulatedPositionNotFoundError,
    SimulatedPositionValidationError,
)
from backtesting.models import (
    BacktestPositionRecord,
    HistoricalCandle,
    HistoricalFundingRecord,
    SimulatedBalance,
    SimulatedEquityPoint,
    SimulatedFill,
    SimulatedPosition,
    SimulatedTrade,
    SerializableMixin,
    new_id,
    safe_div,
    timestamp_ms,
    utcnow,
)


@dataclass(slots=True)
class PositionMarketState(SerializableMixin):
    """
    Minimal market state needed for mark-to-market, funding and liquidation.
    """

    exchange: str = "binance"
    market_type: str = "usdm_futures"
    symbol: str = ""

    mark_price: float | None = None
    last_price: float | None = None
    last_candle: HistoricalCandle | None = None
    last_funding: HistoricalFundingRecord | None = None

    updated_at_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_from_candle(self, candle: HistoricalCandle) -> None:
        self.exchange = candle.exchange
        self.market_type = candle.market_type
        self.symbol = candle.symbol
        self.last_price = candle.close
        self.mark_price = candle.close
        self.last_candle = candle
        self.updated_at_ms = candle.timestamp_ms

    def update_from_funding(self, funding: HistoricalFundingRecord) -> None:
        self.exchange = funding.exchange
        self.market_type = funding.market_type
        self.symbol = funding.symbol
        self.last_funding = funding

        if funding.mark_price is not None and funding.mark_price > 0:
            self.mark_price = funding.mark_price

        self.updated_at_ms = funding.timestamp_ms


@dataclass(slots=True)
class PositionSimulatorStats(SerializableMixin):
    """
    Runtime stats for PositionSimulator.
    """

    status: BacktestStatus = BacktestStatus.CREATED

    fills_processed: int = 0

    positions_opened: int = 0
    positions_updated: int = 0
    positions_closed: int = 0
    positions_liquidated: int = 0

    trades_created: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    funding_events_processed: int = 0
    funding_paid: float = 0.0
    funding_received: float = 0.0

    total_fees: float = 0.0
    total_slippage: float = 0.0

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_equity: float = 0.0
    min_equity: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    started_at: datetime | None = None
    stopped_at: datetime | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PositionSimulator:
    """
    Simulated position manager for backtesting.

    It maintains:
    - current positions;
    - closed positions;
    - reconstructed trades;
    - account balance/equity;
    - equity curve;
    - position event records.
    """

    component_name = "PositionSimulator"

    def __init__(
        self,
        config: PositionSimulatorConfig | None = None,
        *,
        event_bus: EventBus | None = None,
        clock: BacktestClock | None = None,
        cost_model: TradingCostModel | None = None,
        logger_name: str = "backtesting.position_simulator",
    ) -> None:
        self.config = config or PositionSimulatorConfig()
        self.config.validate()

        self.event_bus = event_bus
        self.clock = clock
        self.cost_model = cost_model or TradingCostModel()

        self.logger = get_logger(logger_name)

        self.balance = SimulatedBalance(
            currency=self.config.quote_currency,
            initial_balance=self.config.initial_balance,
            cash_balance=self.config.initial_balance,
            available_balance=self.config.initial_balance,
            equity=self.config.initial_balance,
            updated_at_ms=self._now_ms(),
        )

        self.positions: dict[str, SimulatedPosition] = {}
        self.closed_positions: list[SimulatedPosition] = []
        self.trades: list[SimulatedTrade] = []
        self.equity_curve: list[SimulatedEquityPoint] = []
        self.records: list[BacktestPositionRecord] = []
        self.market_state: dict[str, PositionMarketState] = {}

        self.stats_state = PositionSimulatorStats(
            max_equity=self.config.initial_balance,
            min_equity=self.config.initial_balance,
        )

        self._registered = False
        self._running = False
        self._subscriptions: list[Any] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Register EventBus subscriptions.

        PositionSimulator listens to execution fills and raw market state only.
        It does not listen to signal.generated and does not bypass risk/execution.
        """

        if self._registered:
            return

        if self.event_bus is None:
            self.logger.warning("PositionSimulator has no EventBus; no subscriptions created")
            self._registered = True
            return

        if not isinstance(self.event_bus, EventBus):
            raise PositionSimulatorNotReadyError(
                "PositionSimulator requires core.event_bus.EventBus for full-pipeline backtesting."
            )

        if self.config.listen_order_filled:
            self._subscribe(self.config.order_filled_topic, self._on_order_filled)

        if self.config.listen_order_partially_filled:
            self._subscribe(
                self.config.order_partially_filled_topic,
                self._on_order_partially_filled,
            )

        if self.config.listen_position_close_requested:
            self._subscribe("risk.position_close_requested", self._on_position_close_requested)

        if self.config.listen_position_reduce_requested:
            self._subscribe("risk.position_reduce_requested", self._on_position_reduce_requested)

        # Raw market streams are used only for mark-to-market, equity curve,
        # protective checks and funding. They do not open/close positions by
        # themselves except protective fallback checks.
        self._subscribe("market.candle", self._on_market_candle)
        self._subscribe("market.funding", self._on_market_funding)

        self._registered = True

        self.logger.info(
            "PositionSimulator registered",
            extra={
                "subscriptions": len(self._subscriptions),
                "order_filled_topic": self.config.order_filled_topic,
                "order_partially_filled_topic": self.config.order_partially_filled_topic,
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
            except Exception:
                self.logger.exception(
                    "Failed to unsubscribe PositionSimulator handler",
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
            self._record_equity_point(source="position_simulator.start")

        await self._emit_best_effort(
            "system.backtest.position_simulator.started",
            self.stats(),
        )

    async def stop(self) -> None:
        async with self._lock:
            if not self._running:
                return

            self._running = False
            self.stats_state.status = BacktestStatus.COMPLETED
            self.stats_state.stopped_at = utcnow()
            self._recalculate_account_state()
            self._record_equity_point(source="position_simulator.stop")

        await self._emit_best_effort(
            "system.backtest.position_simulator.stopped",
            self.stats(),
        )

        self.unregister()

    async def reset(self) -> None:
        async with self._lock:
            self.balance = SimulatedBalance(
                currency=self.config.quote_currency,
                initial_balance=self.config.initial_balance,
                cash_balance=self.config.initial_balance,
                available_balance=self.config.initial_balance,
                equity=self.config.initial_balance,
                updated_at_ms=self._now_ms(),
            )
            self.positions.clear()
            self.closed_positions.clear()
            self.trades.clear()
            self.equity_curve.clear()
            self.records.clear()
            self.market_state.clear()
            self.stats_state = PositionSimulatorStats(
                max_equity=self.config.initial_balance,
                min_equity=self.config.initial_balance,
            )

    def _subscribe(self, topic: str, handler: Any) -> None:
        if self.event_bus is None:
            return

        wrapped = self._wrap_event_handler(handler)

        try:
            subscription = self.event_bus.subscribe(
                topic,
                wrapped,
                name=f"backtest_position_on_{topic.replace('.', '_')}",
            )
        except TypeError:
            subscription = self.event_bus.subscribe(
                pattern=topic,
                handler=wrapped,
                name=f"backtest_position_on_{topic.replace('.', '_')}",
            )

        self._subscriptions.append(subscription)

    @staticmethod
    def _wrap_event_handler(handler: Any) -> Any:
        async def _wrapped(event_or_payload: Any) -> None:
            payload = PositionSimulator._payload_from_event_or_dict(event_or_payload)
            result = handler(payload)
            if inspect.isawaitable(result):
                await result

        return _wrapped

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_order_filled(self, payload: dict[str, Any]) -> None:
        """
        Handle complete simulated fill.
        """

        self._ensure_running()
        fill = self._payload_to_fill(payload)
        await self.apply_fill(fill, payload=payload)

    async def _on_order_partially_filled(self, payload: dict[str, Any]) -> None:
        """
        Handle partial simulated fill.
        """

        self._ensure_running()
        fill = self._payload_to_fill(payload)
        await self.apply_fill(fill, payload=payload)

    async def _on_market_candle(self, payload: dict[str, Any]) -> None:
        """
        Update market state and mark positions to market.
        """

        if not self._running:
            return

        try:
            candle = self._payload_to_candle(payload)
        except Exception as exc:
            self.logger.debug(
                "PositionSimulator skipped invalid market.candle payload",
                extra={"error": str(exc)},
            )
            return

        key = self._market_key(candle.exchange, candle.market_type, candle.symbol)
        state = self.market_state.setdefault(
            key,
            PositionMarketState(
                exchange=candle.exchange,
                market_type=candle.market_type,
                symbol=candle.symbol,
            ),
        )
        state.update_from_candle(candle)

        await self.mark_to_market(
            exchange=candle.exchange,
            market_type=candle.market_type,
            symbol=candle.symbol,
            mark_price=candle.close,
            timestamp_ms_value=candle.timestamp_ms,
            source="market.candle",
        )

        await self._check_protective_levels_for_symbol(
            exchange=candle.exchange,
            market_type=candle.market_type,
            symbol=candle.symbol,
            candle=candle,
        )

    async def _on_market_funding(self, payload: dict[str, Any]) -> None:
        """
        Apply funding to open positions.
        """

        if not self._running or not self.config.enable_funding_application:
            return

        try:
            funding = self._payload_to_funding(payload)
        except Exception as exc:
            self.logger.debug(
                "PositionSimulator skipped invalid market.funding payload",
                extra={"error": str(exc)},
            )
            return

        key = self._market_key(funding.exchange, funding.market_type, funding.symbol)
        state = self.market_state.setdefault(
            key,
            PositionMarketState(
                exchange=funding.exchange,
                market_type=funding.market_type,
                symbol=funding.symbol,
            ),
        )
        state.update_from_funding(funding)

        await self.apply_funding(funding)

    async def _on_position_close_requested(self, payload: dict[str, Any]) -> None:
        """
        Risk close request is executed by ExecutionSimulator.

        This handler only annotates open positions for diagnostics. Actual
        closing must happen from execution.order_filled.
        """

        if not self._running:
            return

        position_id = payload.get("position_id")
        symbol = payload.get("symbol")

        for position in self._matching_open_positions(position_id=position_id, symbol=symbol):
            position.metadata["close_requested"] = True
            position.metadata["close_requested_at_ms"] = self._now_ms()
            position.metadata["close_request_payload"] = payload

    async def _on_position_reduce_requested(self, payload: dict[str, Any]) -> None:
        """
        Risk reduce request is executed by ExecutionSimulator.

        This handler only annotates open positions for diagnostics.
        """

        if not self._running:
            return

        position_id = payload.get("position_id")
        symbol = payload.get("symbol")

        for position in self._matching_open_positions(position_id=position_id, symbol=symbol):
            position.metadata["reduce_requested"] = True
            position.metadata["reduce_requested_at_ms"] = self._now_ms()
            position.metadata["reduce_request_payload"] = payload

    # ------------------------------------------------------------------
    # Main position API
    # ------------------------------------------------------------------

    async def apply_fill(
        self,
        fill: SimulatedFill,
        *,
        payload: dict[str, Any] | None = None,
    ) -> SimulatedPosition:
        """
        Apply simulated fill to position accounting.
        """

        self._ensure_running()

        async with self._lock:
            self.stats_state.fills_processed += 1
            self.stats_state.total_fees += fill.fee
            self.stats_state.total_slippage += fill.slippage

            existing = self._find_position_for_fill(fill)

            if existing is None:
                position = self._open_position_from_fill(fill, payload=payload or {})
                self.positions[position.position_id] = position
                self.stats_state.positions_opened += 1
                event_topic = self.config.position_opened_topic
            else:
                position, event_topic = self._apply_fill_to_existing_position(
                    existing,
                    fill,
                    payload=payload or {},
                )

            self._apply_fill_costs_to_balance(fill)
            self._recalculate_account_state()
            self._record_equity_point(source="fill")

        await self._emit_position_event(event_topic, position)
        return position

    async def mark_to_market(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        mark_price: float,
        timestamp_ms_value: int,
        source: str = "mark_to_market",
    ) -> None:
        """
        Update unrealized PnL and equity for all open positions on symbol.
        """

        if not self.config.enable_mark_to_market:
            return

        if mark_price <= 0:
            raise SimulatedPositionValidationError(
                "mark_price must be positive.",
                details={"mark_price": mark_price},
            )

        updated_positions: list[SimulatedPosition] = []

        async with self._lock:
            for position in self.positions.values():
                if not position.is_open:
                    continue

                if position.exchange != exchange.lower():
                    continue

                if position.market_type != market_type.lower():
                    continue

                if position.symbol != symbol.upper():
                    continue

                self._update_position_mark_price(
                    position,
                    mark_price=mark_price,
                    timestamp_ms_value=timestamp_ms_value,
                )
                updated_positions.append(position)

            self._recalculate_account_state()
            self._record_equity_point(source=source)

        for position in updated_positions:
            self.stats_state.positions_updated += 1
            await self._emit_position_event(self.config.position_updated_topic, position)

        if self.config.enable_liquidation_check:
            await self._check_liquidation_for_positions(updated_positions)

    async def apply_funding(self, funding: HistoricalFundingRecord) -> None:
        """
        Apply funding payment to matching open positions.

        Positive funding_rate:
        - longs pay;
        - shorts receive.
        """

        if not self.config.enable_funding_application:
            return

        affected: list[SimulatedPosition] = []

        async with self._lock:
            for position in self.positions.values():
                if not position.is_open:
                    continue

                if position.exchange != funding.exchange:
                    continue

                if position.market_type != funding.market_type:
                    continue

                if position.symbol != funding.symbol:
                    continue

                mark_price = funding.mark_price or position.mark_price or position.entry_price
                notional = abs(position.quantity * mark_price)
                funding_amount = notional * funding.funding_rate

                if self._is_long(position.side):
                    cashflow = -funding_amount
                else:
                    cashflow = funding_amount

                if cashflow >= 0:
                    position.funding_received += cashflow
                    self.balance.cash_balance += cashflow
                    self.balance.realized_pnl += cashflow
                    self.balance.total_funding += cashflow
                    self.stats_state.funding_received += cashflow
                else:
                    paid = abs(cashflow)
                    position.funding_paid += paid
                    self.balance.cash_balance -= paid
                    self.balance.realized_pnl -= paid
                    self.balance.total_funding -= paid
                    self.stats_state.funding_paid += paid

                position.updated_at_ms = funding.timestamp_ms
                affected.append(position)

            self.stats_state.funding_events_processed += 1
            self._recalculate_account_state()
            self._record_equity_point(source="funding")

        for position in affected:
            await self._emit_position_event(self.config.position_updated_topic, position)

    async def close_position(
        self,
        position_id: str,
        *,
        exit_price: float,
        reason: str,
        timestamp_ms_value: int | None = None,
        liquidation: bool = False,
    ) -> SimulatedPosition:
        """
        Close an open position directly.

        This is used only for protective fallback checks and liquidation checks.
        Normal position close should happen through execution.order_filled.
        """

        timestamp_value = timestamp_ms_value or self._now_ms()

        async with self._lock:
            position = self.positions.get(position_id)

            if position is None:
                raise SimulatedPositionNotFoundError(
                    "Position not found.",
                    details={"position_id": position_id},
                )

            if not position.is_open:
                return position

            self._close_position(
                position,
                exit_price=exit_price,
                timestamp_ms_value=timestamp_value,
                reason=reason,
                liquidation=liquidation,
            )
            self._finalize_closed_position(position)

            self.positions.pop(position.position_id, None)
            self.closed_positions.append(position)

            if liquidation:
                self.stats_state.positions_liquidated += 1
                event_topic = self.config.position_liquidated_topic
            else:
                self.stats_state.positions_closed += 1
                event_topic = self.config.position_closed_topic

            self._recalculate_account_state()
            self._record_equity_point(source=f"position.{reason}")

        await self._emit_position_event(event_topic, position)
        return position

    # ------------------------------------------------------------------
    # Fill accounting
    # ------------------------------------------------------------------

    def _open_position_from_fill(
        self,
        fill: SimulatedFill,
        *,
        payload: dict[str, Any],
    ) -> SimulatedPosition:
        side = self._position_side_from_fill_side(fill.side)
        leverage = self._float_from_payload(payload, ["leverage", "final_leverage"], self.config.default_leverage)
        leverage = max(1.0, min(float(leverage), float(self.config.max_leverage)))

        notional = abs(fill.price * fill.quantity)
        margin = notional / leverage

        position = SimulatedPosition(
            run_id=fill.run_id,
            signal_id=fill.signal_id,
            strategy_name=payload.get("strategy_name") or fill.metadata.get("strategy_name"),
            exchange=fill.exchange,
            symbol=fill.symbol,
            market_type=fill.market_type,
            side=side,
            status=SimulatedPositionStatus.OPEN,
            quantity=fill.quantity,
            entry_price=fill.price,
            mark_price=fill.price,
            leverage=leverage,
            margin=margin,
            notional=notional,
            fees_paid=fill.fee,
            slippage_paid=fill.slippage,
            opened_at_ms=fill.timestamp_ms,
            updated_at_ms=fill.timestamp_ms,
            source_order_ids=[fill.order_id],
            liquidation_price=self._estimate_liquidation_price(
                side=side,
                entry_price=fill.price,
                leverage=leverage,
            ),
            metadata={
                "source": "execution_fill",
                "fill_id": fill.fill_id,
                "order_id": fill.order_id,
                "payload": payload,
            },
        )

        self.balance.cash_balance -= fill.fee
        self.balance.cash_balance -= fill.slippage

        return position

    def _apply_fill_to_existing_position(
        self,
        position: SimulatedPosition,
        fill: SimulatedFill,
        *,
        payload: dict[str, Any],
    ) -> tuple[SimulatedPosition, str]:
        fill_side = self._position_side_from_fill_side(fill.side)

        if fill_side == position.side:
            self._increase_position(position, fill)
            self.stats_state.positions_updated += 1
            return position, self.config.position_updated_topic

        # Opposite fill: reduce, close, or reverse.
        if fill.quantity < position.quantity:
            self._reduce_position(position, fill)
            self.stats_state.positions_updated += 1
            return position, self.config.position_updated_topic

        if fill.quantity == position.quantity:
            self._close_position_by_fill(position, fill, reason="opposite_fill")
            self._finalize_closed_position(position)
            self.positions.pop(position.position_id, None)
            self.closed_positions.append(position)
            self.stats_state.positions_closed += 1
            return position, self.config.position_closed_topic

        # fill.quantity > position.quantity: close old and optionally reverse.
        if not self.config.allow_position_reversal:
            raise PositionAccountingError(
                "Position reversal is disabled.",
                details={
                    "position_id": position.position_id,
                    "position_quantity": position.quantity,
                    "fill_quantity": fill.quantity,
                },
            )

        remaining_qty = fill.quantity - position.quantity

        self._close_position_by_fill(position, fill, reason="reverse_fill")
        self._finalize_closed_position(position)
        self.positions.pop(position.position_id, None)
        self.closed_positions.append(position)
        self.stats_state.positions_closed += 1

        reversal_fill = SimulatedFill(
            order_id=fill.order_id,
            run_id=fill.run_id,
            signal_id=fill.signal_id,
            exchange=fill.exchange,
            symbol=fill.symbol,
            market_type=fill.market_type,
            side=fill.side,
            price=fill.price,
            quantity=remaining_qty,
            notional=abs(fill.price * remaining_qty),
            fee=0.0,
            fee_asset=fill.fee_asset,
            slippage=0.0,
            slippage_bps=fill.slippage_bps,
            timestamp_ms=fill.timestamp_ms,
            metadata={
                **fill.metadata,
                "reversal": True,
                "original_fill_id": fill.fill_id,
            },
        )

        new_position = self._open_position_from_fill(reversal_fill, payload=payload)
        self.positions[new_position.position_id] = new_position
        self.stats_state.positions_opened += 1

        return new_position, self.config.position_opened_topic

    def _increase_position(
        self,
        position: SimulatedPosition,
        fill: SimulatedFill,
    ) -> None:
        previous_notional = position.entry_price * position.quantity
        added_notional = fill.price * fill.quantity
        new_quantity = position.quantity + fill.quantity

        position.entry_price = (previous_notional + added_notional) / new_quantity
        position.quantity = new_quantity
        position.notional = abs(position.entry_price * position.quantity)
        position.margin = position.notional / position.leverage
        position.mark_price = fill.price
        position.updated_at_ms = fill.timestamp_ms
        position.fees_paid += fill.fee
        position.slippage_paid += fill.slippage
        position.source_order_ids.append(fill.order_id)
        position.liquidation_price = self._estimate_liquidation_price(
            side=position.side,
            entry_price=position.entry_price,
            leverage=position.leverage,
        )

        self.balance.cash_balance -= fill.fee
        self.balance.cash_balance -= fill.slippage

    def _reduce_position(
        self,
        position: SimulatedPosition,
        fill: SimulatedFill,
    ) -> None:
        reduce_qty = min(position.quantity, fill.quantity)
        gross_pnl = self._pnl_for_quantity(
            side=position.side,
            entry_price=position.entry_price,
            exit_price=fill.price,
            quantity=reduce_qty,
        )

        closed_fraction = safe_div(reduce_qty, position.quantity)
        released_margin = position.margin * closed_fraction

        position.quantity -= reduce_qty
        position.notional = abs(position.entry_price * position.quantity)
        position.margin = max(0.0, position.margin - released_margin)
        position.mark_price = fill.price
        position.realized_pnl += gross_pnl
        position.fees_paid += fill.fee
        position.slippage_paid += fill.slippage
        position.updated_at_ms = fill.timestamp_ms
        position.source_order_ids.append(fill.order_id)

        self.balance.cash_balance += gross_pnl
        self.balance.cash_balance -= fill.fee
        self.balance.cash_balance -= fill.slippage

    def _close_position_by_fill(
        self,
        position: SimulatedPosition,
        fill: SimulatedFill,
        *,
        reason: str,
    ) -> None:
        self._close_position(
            position,
            exit_price=fill.price,
            timestamp_ms_value=fill.timestamp_ms,
            reason=reason,
            liquidation=False,
        )

        position.fees_paid += fill.fee
        position.slippage_paid += fill.slippage
        position.source_order_ids.append(fill.order_id)

        self.balance.cash_balance -= fill.fee
        self.balance.cash_balance -= fill.slippage

    def _close_position(
        self,
        position: SimulatedPosition,
        *,
        exit_price: float,
        timestamp_ms_value: int,
        reason: str,
        liquidation: bool,
    ) -> None:
        gross_pnl = self._pnl_for_quantity(
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
        )

        position.exit_price = exit_price
        position.mark_price = exit_price
        position.realized_pnl += gross_pnl
        position.unrealized_pnl = 0.0
        position.closed_at_ms = timestamp_ms_value
        position.updated_at_ms = timestamp_ms_value
        position.close_reason = reason
        position.status = (
            SimulatedPositionStatus.LIQUIDATED
            if liquidation
            else SimulatedPositionStatus.CLOSED
        )

        self.balance.cash_balance += gross_pnl

    def _finalize_closed_position(self, position: SimulatedPosition) -> SimulatedTrade:
        trade = self._build_trade_from_position(position)
        self.trades.append(trade)
        self.stats_state.trades_created += 1
        self.stats_state.realized_pnl += trade.net_pnl

        if trade.outcome == TradeOutcome.WIN:
            self.stats_state.winning_trades += 1
        elif trade.outcome == TradeOutcome.LOSS:
            self.stats_state.losing_trades += 1
        elif trade.outcome == TradeOutcome.BREAKEVEN:
            self.stats_state.breakeven_trades += 1

        return trade

    def _build_trade_from_position(self, position: SimulatedPosition) -> SimulatedTrade:
        gross_pnl = position.realized_pnl
        net_pnl = gross_pnl - position.fees_paid - position.slippage_paid + position.net_funding
        pnl_pct = calculate_return_pct(
            entry_price=position.entry_price,
            exit_price=position.exit_price or position.mark_price,
            side=position.side,
        )
        r_multiple = calculate_r_multiple(
            pnl=net_pnl,
            risk_amount=position.metadata.get("risk_amount"),
        )

        if position.status == SimulatedPositionStatus.LIQUIDATED:
            outcome = TradeOutcome.LIQUIDATED
        elif net_pnl > 0:
            outcome = TradeOutcome.WIN
        elif net_pnl < 0:
            outcome = TradeOutcome.LOSS
        else:
            outcome = TradeOutcome.BREAKEVEN

        return SimulatedTrade(
            run_id=position.run_id,
            position_id=position.position_id,
            signal_id=position.signal_id,
            strategy_name=position.strategy_name,
            exchange=position.exchange,
            symbol=position.symbol,
            market_type=position.market_type,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=position.exit_price or position.mark_price,
            opened_at_ms=position.opened_at_ms,
            closed_at_ms=position.closed_at_ms,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            pnl_pct=pnl_pct,
            r_multiple=r_multiple,
            fees=position.fees_paid,
            slippage=position.slippage_paid,
            funding=position.net_funding,
            outcome=outcome,
            close_reason=position.close_reason,
            metadata={
                "liquidation_price": position.liquidation_price,
                "holding_time_seconds": position.holding_time_seconds,
            },
        )

    # ------------------------------------------------------------------
    # Protective checks
    # ------------------------------------------------------------------

    async def _check_protective_levels_for_symbol(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        candle: HistoricalCandle,
    ) -> None:
        """
        Check SL/TP using candle high/low.

        This is only fallback protective simulation. In full production-like
        pipeline, explicit protective orders should be simulated by
        ExecutionSimulator.
        """

        positions = [
            position
            for position in self.positions.values()
            if position.is_open
            and position.exchange == exchange.lower()
            and position.market_type == market_type.lower()
            and position.symbol == symbol.upper()
        ]

        for position in positions:
            if self.config.enable_stop_loss and position.stop_loss is not None:
                if self._stop_loss_hit(position, candle):
                    await self.close_position(
                        position.position_id,
                        exit_price=position.stop_loss,
                        reason="stop_loss",
                        timestamp_ms_value=candle.timestamp_ms,
                    )
                    continue

            if self.config.enable_take_profit and position.take_profit is not None:
                if self._take_profit_hit(position, candle):
                    await self.close_position(
                        position.position_id,
                        exit_price=position.take_profit,
                        reason="take_profit",
                        timestamp_ms_value=candle.timestamp_ms,
                    )
                    continue

    async def _check_liquidation_for_positions(
        self,
        positions: list[SimulatedPosition],
    ) -> None:
        if not self.config.enable_liquidation_check:
            return

        for position in positions:
            if not position.is_open:
                continue

            if position.liquidation_price is None:
                continue

            liquidated = (
                self._is_long(position.side)
                and position.mark_price <= position.liquidation_price
            ) or (
                self._is_short(position.side)
                and position.mark_price >= position.liquidation_price
            )

            if not liquidated:
                continue

            await self.close_position(
                position.position_id,
                exit_price=position.liquidation_price,
                reason="liquidation",
                timestamp_ms_value=position.updated_at_ms or self._now_ms(),
                liquidation=True,
            )

    @staticmethod
    def _stop_loss_hit(position: SimulatedPosition, candle: HistoricalCandle) -> bool:
        if position.stop_loss is None:
            return False

        if PositionSimulator._is_long(position.side):
            return candle.low <= position.stop_loss

        if PositionSimulator._is_short(position.side):
            return candle.high >= position.stop_loss

        return False

    @staticmethod
    def _take_profit_hit(position: SimulatedPosition, candle: HistoricalCandle) -> bool:
        if position.take_profit is None:
            return False

        if PositionSimulator._is_long(position.side):
            return candle.high >= position.take_profit

        if PositionSimulator._is_short(position.side):
            return candle.low <= position.take_profit

        return False

    # ------------------------------------------------------------------
    # Account / equity
    # ------------------------------------------------------------------

    def _apply_fill_costs_to_balance(self, fill: SimulatedFill) -> None:
        """
        Keep aggregate cost fields synchronized.
        """

        self.balance.total_fees += fill.fee
        self.balance.total_slippage += fill.slippage

    def _recalculate_account_state(self) -> None:
        unrealized = 0.0
        margin_used = 0.0

        for position in self.positions.values():
            if not position.is_open:
                continue

            unrealized += position.unrealized_pnl
            margin_used += position.margin

        self.balance.margin_used = max(0.0, margin_used)
        self.balance.unrealized_pnl = unrealized

        if self.config.pnl_accounting_mode == PnLAccountingMode.REALIZED_ONLY:
            self.balance.equity = self.balance.cash_balance
        elif self.config.pnl_accounting_mode == PnLAccountingMode.MARK_TO_MARKET:
            self.balance.equity = self.balance.cash_balance + unrealized
        else:
            self.balance.equity = self.balance.cash_balance + unrealized

        self.balance.available_balance = self.balance.cash_balance - self.balance.margin_used
        self.balance.updated_at_ms = self._now_ms()

        self.stats_state.unrealized_pnl = unrealized

        if self.balance.equity > self.stats_state.max_equity:
            self.stats_state.max_equity = self.balance.equity

        if self.balance.equity < self.stats_state.min_equity:
            self.stats_state.min_equity = self.balance.equity

        peak = max(self.stats_state.max_equity, self.config.initial_balance)
        drawdown = max(0.0, peak - self.balance.equity)
        drawdown_pct = safe_div(drawdown, peak) * 100.0 if peak > 0 else 0.0

        if drawdown > self.stats_state.max_drawdown:
            self.stats_state.max_drawdown = drawdown

        if drawdown_pct > self.stats_state.max_drawdown_pct:
            self.stats_state.max_drawdown_pct = drawdown_pct

        if self.balance.equity < 0:
            raise EquityCalculationError(
                "Simulated equity became negative.",
                details={
                    "equity": self.balance.equity,
                    "cash_balance": self.balance.cash_balance,
                    "unrealized_pnl": unrealized,
                    "margin_used": margin_used,
                },
            )

    def _record_equity_point(self, *, source: str) -> None:
        if not self.config.record_equity_curve:
            return

        peak = max(self.stats_state.max_equity, self.config.initial_balance)
        drawdown = max(0.0, peak - self.balance.equity)
        drawdown_pct = safe_div(drawdown, peak) * 100.0 if peak > 0 else 0.0

        point = SimulatedEquityPoint(
            timestamp_ms=self._now_ms(),
            equity=self.balance.equity,
            balance=self.balance.cash_balance,
            available_balance=self.balance.available_balance,
            margin_used=self.balance.margin_used,
            unrealized_pnl=self.balance.unrealized_pnl,
            realized_pnl=self.balance.realized_pnl,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct,
            open_positions=len(
                [position for position in self.positions.values() if position.is_open]
            ),
            source=source,
        )
        self.equity_curve.append(point)

    # ------------------------------------------------------------------
    # Event emission / records
    # ------------------------------------------------------------------

    async def _emit_position_event(
        self,
        topic: str,
        position: SimulatedPosition,
    ) -> None:
        payload = self._position_to_event_payload(position)
        await self._emit(topic, payload)

        self._record_position_event(
            topic=topic,
            position=position,
            payload=payload,
        )

    def _position_to_event_payload(self, position: SimulatedPosition) -> dict[str, Any]:
        return {
            "position_id": position.position_id,
            "run_id": position.run_id,
            "signal_id": position.signal_id,
            "strategy_name": position.strategy_name,
            "exchange": position.exchange,
            "market_type": position.market_type,
            "symbol": position.symbol,
            "side": position.side,
            "status": position.status.value,
            "quantity": position.quantity,
            "entry_price": position.entry_price,
            "mark_price": position.mark_price,
            "exit_price": position.exit_price,
            "leverage": position.leverage,
            "margin": position.margin,
            "notional": position.notional,
            "realized_pnl": position.realized_pnl,
            "unrealized_pnl": position.unrealized_pnl,
            "net_realized_pnl": self._net_realized_pnl(position),
            "fees_paid": position.fees_paid,
            "funding_paid": position.funding_paid,
            "funding_received": position.funding_received,
            "slippage_paid": position.slippage_paid,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "liquidation_price": position.liquidation_price,
            "opened_at_ms": position.opened_at_ms,
            "updated_at_ms": position.updated_at_ms,
            "closed_at_ms": position.closed_at_ms,
            "close_reason": position.close_reason,
            "source_order_ids": list(position.source_order_ids),
            "account": self.balance.to_dict(),
            "source": "position_simulator",
            "metadata": {
                **position.metadata,
                "backtest": True,
                "simulated": True,
            },
        }

    def _record_position_event(
        self,
        *,
        topic: str,
        position: SimulatedPosition,
        payload: dict[str, Any],
    ) -> None:
        if not self.config.record_positions:
            return

        record = BacktestPositionRecord(
            run_id=position.run_id,
            timestamp_ms=self._now_ms(),
            topic=topic,
            position_id=position.position_id,
            signal_id=position.signal_id,
            strategy_name=position.strategy_name,
            symbol=position.symbol,
            status=position.status,
            payload=payload,
            metadata={"source": "position_simulator"},
        )
        self.records.append(record)

    async def _emit(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        if not self.config.emit_position_events:
            return

        if self.event_bus is None:
            return

        result = self.event_bus.emit(
            topic,
            payload,
            priority=priority,
            source="PositionSimulator",
        )

        if inspect.isawaitable(result):
            await result

    async def _emit_best_effort(self, topic: str, payload: dict[str, Any]) -> None:
        try:
            await self._emit(topic, payload, priority=EventPriority.LOW)
        except Exception:
            self.logger.debug(
                "Best-effort position simulator event failed",
                extra={"topic": topic},
            )

    # ------------------------------------------------------------------
    # Payload conversion
    # ------------------------------------------------------------------

    def _payload_to_fill(self, payload: dict[str, Any]) -> SimulatedFill:
        timestamp_value = int(
            payload.get("timestamp_ms")
            or payload.get("filled_at_ms")
            or payload.get("updated_at_ms")
            or self._now_ms()
        )

        price = float(
            payload.get("fill_price")
            or payload.get("average_fill_price")
            or payload.get("price")
            or 0.0
        )
        quantity = float(
            payload.get("fill_quantity")
            or payload.get("filled_quantity")
            or payload.get("quantity")
            or 0.0
        )
        notional = float(
            payload.get("fill_notional")
            or payload.get("notional")
            or abs(price * quantity)
        )

        return SimulatedFill(
            order_id=str(payload.get("order_id") or payload.get("client_order_id") or new_id("unknown_order")),
            run_id=payload.get("run_id"),
            signal_id=payload.get("signal_id"),
            exchange=str(payload.get("exchange") or "binance"),
            symbol=str(payload.get("symbol") or ""),
            market_type=str(payload.get("market_type") or "usdm_futures"),
            side=str(payload.get("side") or ""),
            price=price,
            quantity=quantity,
            notional=notional,
            fee=float(payload.get("fill_fee") or payload.get("fee") or 0.0),
            fee_asset=str(payload.get("fee_asset") or self.config.quote_currency),
            slippage=float(payload.get("fill_slippage") or payload.get("slippage") or 0.0),
            slippage_bps=float(payload.get("slippage_bps") or 0.0),
            timestamp_ms=timestamp_value,
            metadata={
                **dict(payload.get("metadata") or {}),
                "strategy_name": payload.get("strategy_name"),
                "fill_id": payload.get("fill_id"),
                "source_payload": payload,
            },
        )

    def _payload_to_candle(self, payload: dict[str, Any]) -> HistoricalCandle:
        now_ms = self._now_ms()

        return HistoricalCandle(
            exchange=str(payload.get("exchange") or "binance"),
            symbol=str(payload.get("symbol") or ""),
            market_type=str(payload.get("market_type") or "usdm_futures"),
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

    def _payload_to_funding(self, payload: dict[str, Any]) -> HistoricalFundingRecord:
        now_ms = self._now_ms()

        return HistoricalFundingRecord(
            exchange=str(payload.get("exchange") or "binance"),
            symbol=str(payload.get("symbol") or ""),
            market_type=str(payload.get("market_type") or "usdm_futures"),
            timestamp_ms=int(payload.get("timestamp_ms") or now_ms),
            received_at_ms=int(payload.get("received_at_ms") or payload.get("timestamp_ms") or now_ms),
            funding_rate=float(payload.get("funding_rate") or payload.get("rate") or 0.0),
            predicted_rate=self._optional_float(payload.get("predicted_rate")),
            mark_price=self._optional_float(payload.get("mark_price")),
            index_price=self._optional_float(payload.get("index_price")),
            next_funding_time_ms=self._optional_int(payload.get("next_funding_time_ms")),
            source="market_replay",
            metadata=dict(payload.get("metadata") or {}),
        )

    # ------------------------------------------------------------------
    # Lookup / helpers
    # ------------------------------------------------------------------

    def all_positions(self) -> list[SimulatedPosition]:
        return [
            *self.positions.values(),
            *self.closed_positions,
        ]

    def get_open_positions(self) -> list[SimulatedPosition]:
        return [
            position
            for position in self.positions.values()
            if position.is_open
        ]

    def get_position(self, position_id: str) -> SimulatedPosition | None:
        if position_id in self.positions:
            return self.positions[position_id]

        for position in self.closed_positions:
            if position.position_id == position_id:
                return position

        return None

    def _find_position_for_fill(self, fill: SimulatedFill) -> SimulatedPosition | None:
        fill_side = self._position_side_from_fill_side(fill.side)

        candidates = [
            position
            for position in self.positions.values()
            if position.is_open
            and position.exchange == fill.exchange
            and position.market_type == fill.market_type
            and position.symbol == fill.symbol
        ]

        if not candidates:
            return None

        if self.config.position_accounting_mode == PositionAccountingMode.HEDGE:
            for position in candidates:
                if position.side == fill_side:
                    return position

            if self.config.close_opposite_position_on_reverse:
                return candidates[0]

            return None

        return candidates[0]

    def _matching_open_positions(
        self,
        *,
        position_id: Any | None = None,
        symbol: Any | None = None,
    ) -> list[SimulatedPosition]:
        result: list[SimulatedPosition] = []

        for position in self.positions.values():
            if not position.is_open:
                continue

            if position_id is not None and position.position_id != str(position_id):
                continue

            if symbol is not None and position.symbol != str(symbol).upper():
                continue

            result.append(position)

        return result

    @staticmethod
    def _position_side_from_fill_side(side: str) -> str:
        normalized = str(side or "").lower()

        if normalized in {"buy", "long"}:
            return "long"

        if normalized in {"sell", "short"}:
            return "short"

        return normalized

    @staticmethod
    def _pnl_for_quantity(
        *,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
    ) -> float:
        if PositionSimulator._is_long(side):
            return (exit_price - entry_price) * quantity

        if PositionSimulator._is_short(side):
            return (entry_price - exit_price) * quantity

        return 0.0

    def _update_position_mark_price(
        self,
        position: SimulatedPosition,
        *,
        mark_price: float,
        timestamp_ms_value: int,
    ) -> None:
        position.mark_price = mark_price
        position.updated_at_ms = timestamp_ms_value

        position.unrealized_pnl = self._pnl_for_quantity(
            side=position.side,
            entry_price=position.entry_price,
            exit_price=mark_price,
            quantity=position.quantity,
        )

    def _estimate_liquidation_price(
        self,
        *,
        side: str,
        entry_price: float,
        leverage: float,
    ) -> float | None:
        if entry_price <= 0 or leverage <= 0:
            return None

        maintenance = self.config.maintenance_margin_rate
        buffer = self.config.liquidation_buffer_bps / 10_000.0

        if self._is_long(side):
            return entry_price * (1.0 - (1.0 / leverage) + maintenance + buffer)

        if self._is_short(side):
            return entry_price * (1.0 + (1.0 / leverage) - maintenance - buffer)

        return None

    @staticmethod
    def _net_realized_pnl(position: SimulatedPosition) -> float:
        return position.realized_pnl - position.fees_paid - position.slippage_paid + position.net_funding

    @staticmethod
    def _is_long(side: str) -> bool:
        return str(side or "").lower() in {"buy", "long"}

    @staticmethod
    def _is_short(side: str) -> bool:
        return str(side or "").lower() in {"sell", "short"}

    @staticmethod
    def _market_key(exchange: str, market_type: str, symbol: str) -> str:
        return f"{str(exchange).lower()}:{str(market_type).lower()}:{str(symbol).upper()}"

    def _now_ms(self) -> int:
        if self.clock is not None:
            try:
                return self.clock.timestamp_ms_or_wall_clock()
            except Exception:
                pass

        return timestamp_ms(utcnow())

    @staticmethod
    def _payload_from_event_or_dict(event_or_payload: Any) -> dict[str, Any]:
        if isinstance(event_or_payload, dict):
            return dict(event_or_payload)

        payload = getattr(event_or_payload, "payload", None)
        if isinstance(payload, dict):
            return dict(payload)

        return {}

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None

        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None

        try:
            return int(float(value))
        except Exception:
            return None

    @staticmethod
    def _float_from_payload(
        payload: dict[str, Any],
        keys: list[str],
        default: float,
    ) -> float:
        for key in keys:
            value = payload.get(key)

            if value is None:
                continue

            try:
                return float(value)
            except Exception:
                continue

        nested_keys = ("metadata", "execution_intent", "risk_decision", "intent")

        for nested_key in nested_keys:
            nested = payload.get(nested_key)
            if not isinstance(nested, dict):
                continue

            for key in keys:
                value = nested.get(key)
                if value is None:
                    continue

                try:
                    return float(value)
                except Exception:
                    continue

        return float(default)

    def _ensure_running(self) -> None:
        if not self._running:
            raise PositionSimulatorNotReadyError("PositionSimulator is not running.")

    def stats(self) -> dict[str, Any]:
        payload = self.stats_state.to_dict()
        payload.update(
            {
                "running": self._running,
                "registered": self._registered,
                "subscriptions": len(self._subscriptions),
                "open_positions": len(self.get_open_positions()),
                "closed_positions": len(self.closed_positions),
                "trades": len(self.trades),
                "equity_points": len(self.equity_curve),
                "records": len(self.records),
                "balance": self.balance.to_dict(),
                "event_bus_type": self.event_bus.__class__.__name__ if self.event_bus else None,
            }
        )
        return payload


__all__ = [
    "PositionMarketState",
    "PositionSimulatorStats",
    "PositionSimulator",
]