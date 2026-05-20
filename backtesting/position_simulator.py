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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

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
    SimulatedPositionStateError,
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

try:
    from core.logger import get_logger
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


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

    def __init__(
        self,
        config: PositionSimulatorConfig | None = None,
        *,
        event_bus: Any | None = None,
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
        """

        if self._registered:
            return

        if self.event_bus is None:
            self._registered = True
            return

        if self.config.listen_order_filled:
            self._subscribe(self.config.order_filled_topic, self._on_order_filled)

        if self.config.listen_order_partially_filled:
            self._subscribe(self.config.order_partially_filled_topic, self._on_order_partially_filled)

        if self.config.listen_position_close_requested:
            self._subscribe("risk.position_close_requested", self._on_position_close_requested)

        if self.config.listen_position_reduce_requested:
            self._subscribe("risk.position_reduce_requested", self._on_position_reduce_requested)

        # Market state streams for MTM, SL/TP, liquidation and funding.
        self._subscribe("market.candle", self._on_market_candle)
        self._subscribe("market.funding", self._on_market_funding)

        self._registered = True

    async def start(self) -> None:
        async with self._lock:
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
            self._running = False
            self.stats_state.status = BacktestStatus.COMPLETED
            self.stats_state.stopped_at = utcnow()
            self._recalculate_account_state()
            self._record_equity_point(source="position_simulator.stop")

        await self._emit_best_effort(
            "system.backtest.position_simulator.stopped",
            self.stats(),
        )

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
        except Exception:
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
        except Exception:
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
        Position close request is normally executed by ExecutionSimulator first.

        This handler is best-effort fallback for bookkeeping diagnostics only.
        """

        if not self._running:
            return

        position_id = payload.get("position_id")
        symbol = payload.get("symbol")

        if position_id and position_id not in self.positions:
            return

        if not position_id and not symbol:
            return

        # Do not close directly here; fills should close. We only mark metadata.
        for position in self._matching_open_positions(position_id=position_id, symbol=symbol):
            position.metadata["close_requested"] = True
            position.metadata["close_requested_at_ms"] = self._now_ms()
            position.metadata["close_request_payload"] = payload

    async def _on_position_reduce_requested(self, payload: dict[str, Any]) -> None:
        """
        Position reduce request is normally executed by ExecutionSimulator first.
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

            key = self._market_key(fill.exchange, fill.market_type, fill.symbol)
            open_position = self._find_position_for_fill(fill)

            if open_position is None:
                position = self._open_position_from_fill(fill, payload=payload or {})
                self.positions[position.position_id] = position
                self.stats_state.positions_opened += 1

                self._apply_fill_costs_to_balance(fill)
                self._recalculate_account_state()
                self._record_equity_point(source="fill.open")
            else:
                position = await self._apply_fill_to_existing_position(
                    open_position,
                    fill,
                    payload=payload or {},
                )
                self._apply_fill_costs_to_balance(fill)
                self._recalculate_account_state()
                self._record_equity_point(source="fill.update")

        if position.status == SimulatedPositionStatus.OPEN:
            await self._emit_position_event(self.config.position_opened_topic, position)

        elif position.status == SimulatedPositionStatus.CLOSED:
            await self._emit_position_event(self.config.position_closed_topic, position)

        elif position.status == SimulatedPositionStatus.LIQUIDATED:
            await self._emit_position_event(self.config.position_liquidated_topic, position)

        else:
            await self._emit_position_event(self.config.position_updated_topic, position)

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

                position.update_mark_price(mark_price, timestamp_ms_value)
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

                notional = abs(position.quantity * (funding.mark_price or position.mark_price or position.entry_price))
                cashflow = self.cost_model.funding.calculate_from_record(
                    side=position.side,
                    notional=notional,
                    funding=funding,
                    holding_seconds=position.holding_time_seconds,
                )

                if cashflow >= 0:
                    position.funding_received += cashflow
                    self.balance.cash_balance += cashflow
                    self.balance.realized_pnl += cashflow
                    self.stats_state.funding_received += cashflow
                else:
                    paid = abs(cashflow)
                    position.funding_paid += paid
                    self.balance.cash_balance -= paid
                    self.balance.realized_pnl -= paid
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
        Close an open simulated position.
        """

        async with self._lock:
            position = self.positions.get(position_id)

            if position is None:
                raise SimulatedPositionNotFoundError(
                    "Position not found.",
                    details={"position_id": position_id},
                )

            if not position.is_open:
                raise SimulatedPositionStateError(
                    "Position is not open.",
                    details={
                        "position_id": position_id,
                        "status": position.status.value,
                    },
                )

            ts = timestamp_ms_value or self._now_ms()
            position.close(
                exit_price=exit_price,
                timestamp_ms_value=ts,
                reason=reason,
            )

            if liquidation:
                position.status = SimulatedPositionStatus.LIQUIDATED
                position.close_reason = reason

            self._finalize_closed_position(position)
            self.positions.pop(position_id, None)
            self.closed_positions.append(position)

            self._recalculate_account_state()
            self._record_equity_point(source="close_position")

        if liquidation:
            await self._emit_position_event(self.config.position_liquidated_topic, position)
        else:
            await self._emit_position_event(self.config.position_closed_topic, position)

        return position

    # ------------------------------------------------------------------
    # Position accounting internals
    # ------------------------------------------------------------------

    def _open_position_from_fill(
        self,
        fill: SimulatedFill,
        *,
        payload: dict[str, Any],
    ) -> SimulatedPosition:
        side = self._position_side_from_fill_side(fill.side)
        leverage = self._extract_float(payload, ["leverage", "final_leverage"], default=self.config.default_leverage)
        leverage = min(max(leverage, 1.0), self.config.max_leverage)

        notional = abs(fill.price * fill.quantity)
        margin = notional / leverage

        stop_loss = self._extract_optional_float(payload, ["stop_loss", "sl", "stop_price"])
        take_profit = self._extract_optional_float(payload, ["take_profit", "tp", "target_price"])

        liquidation_price = self.cost_model.calculate_liquidation_price(
            side=side,
            entry_price=fill.price,
            quantity=fill.quantity,
            leverage=leverage,
            margin=margin,
            maintenance_margin_rate=self.config.maintenance_margin_rate,
            buffer_bps=self.config.liquidation_buffer_bps,
        )

        position = SimulatedPosition(
            run_id=fill.run_id,
            signal_id=fill.signal_id,
            strategy_name=fill.metadata.get("strategy_name"),
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
            stop_loss=stop_loss,
            take_profit=take_profit,
            liquidation_price=liquidation_price,
            opened_at_ms=fill.timestamp_ms,
            updated_at_ms=fill.timestamp_ms,
            source_order_ids=[fill.order_id],
            metadata={
                "entry_fill_id": fill.fill_id,
                "entry_order_id": fill.order_id,
                "source": "position_simulator",
            },
        )

        self.balance.cash_balance -= fill.fee
        self.balance.cash_balance -= fill.slippage
        self.balance.margin_used += margin
        self.balance.available_balance = self.balance.cash_balance - self.balance.margin_used

        return position

    async def _apply_fill_to_existing_position(
        self,
        position: SimulatedPosition,
        fill: SimulatedFill,
        *,
        payload: dict[str, Any],
    ) -> SimulatedPosition:
        fill_side = self._position_side_from_fill_side(fill.side)

        if fill_side == position.side:
            self._increase_position(position, fill)
            self.stats_state.positions_updated += 1
            return position

        # Opposite side: reduce, close or reverse.
        if fill.quantity < position.quantity:
            self._reduce_position(position, fill)
            self.stats_state.positions_updated += 1
            return position

        if fill.quantity == position.quantity:
            self._close_position_by_fill(position, fill, reason="opposite_fill")
            self._finalize_closed_position(position)
            self.positions.pop(position.position_id, None)
            self.closed_positions.append(position)
            self.stats_state.positions_closed += 1
            return position

        # fill.quantity > position.quantity
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
        return new_position

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

        position.liquidation_price = self.cost_model.calculate_liquidation_price(
            side=position.side,
            entry_price=position.entry_price,
            quantity=position.quantity,
            leverage=position.leverage,
            margin=position.margin,
            maintenance_margin_rate=self.config.maintenance_margin_rate,
            buffer_bps=self.config.liquidation_buffer_bps,
        )

    def _reduce_position(
        self,
        position: SimulatedPosition,
        fill: SimulatedFill,
    ) -> None:
        if fill.quantity <= 0 or fill.quantity >= position.quantity:
            raise PositionAccountingError(
                "Invalid reduce fill quantity.",
                details={
                    "position_id": position.position_id,
                    "position_quantity": position.quantity,
                    "fill_quantity": fill.quantity,
                },
            )

        realized = self._calculate_realized_pnl(
            side=position.side,
            entry_price=position.entry_price,
            exit_price=fill.price,
            quantity=fill.quantity,
        )

        reduction_ratio = fill.quantity / position.quantity
        released_margin = position.margin * reduction_ratio

        position.quantity -= fill.quantity
        position.notional = abs(position.entry_price * position.quantity)
        position.margin -= released_margin
        position.realized_pnl += realized
        position.mark_price = fill.price
        position.updated_at_ms = fill.timestamp_ms
        position.fees_paid += fill.fee
        position.slippage_paid += fill.slippage
        position.source_order_ids.append(fill.order_id)

        self.balance.cash_balance += realized
        self.balance.cash_balance -= fill.fee
        self.balance.cash_balance -= fill.slippage
        self.balance.margin_used -= released_margin
        self.balance.realized_pnl += realized - fill.fee - fill.slippage

    def _close_position_by_fill(
        self,
        position: SimulatedPosition,
        fill: SimulatedFill,
        *,
        reason: str,
    ) -> None:
        realized = self._calculate_realized_pnl(
            side=position.side,
            entry_price=position.entry_price,
            exit_price=fill.price,
            quantity=position.quantity,
        )

        position.exit_price = fill.price
        position.mark_price = fill.price
        position.closed_at_ms = fill.timestamp_ms
        position.updated_at_ms = fill.timestamp_ms
        position.close_reason = reason
        position.status = SimulatedPositionStatus.CLOSED
        position.realized_pnl += realized
        position.unrealized_pnl = 0.0
        position.fees_paid += fill.fee
        position.slippage_paid += fill.slippage
        position.source_order_ids.append(fill.order_id)

        self.balance.cash_balance += position.margin
        self.balance.cash_balance += realized
        self.balance.cash_balance -= fill.fee
        self.balance.cash_balance -= fill.slippage
        self.balance.margin_used -= position.margin
        self.balance.realized_pnl += realized - fill.fee - fill.slippage

    def _finalize_closed_position(self, position: SimulatedPosition) -> SimulatedTrade:
        net_pnl = position.net_realized_pnl
        gross_pnl = position.realized_pnl
        pnl_pct = calculate_return_pct(
            side=position.side,
            entry_price=position.entry_price,
            exit_price=position.exit_price or position.mark_price,
        )

        initial_risk = self._estimate_initial_risk(position)
        r_multiple = calculate_r_multiple(pnl=net_pnl, initial_risk=initial_risk)

        if position.status == SimulatedPositionStatus.LIQUIDATED:
            outcome = TradeOutcome.LIQUIDATED
            self.stats_state.positions_liquidated += 1
        elif net_pnl > 0:
            outcome = TradeOutcome.WIN
            self.stats_state.winning_trades += 1
        elif net_pnl < 0:
            outcome = TradeOutcome.LOSS
            self.stats_state.losing_trades += 1
        else:
            outcome = TradeOutcome.BREAKEVEN
            self.stats_state.breakeven_trades += 1

        trade = SimulatedTrade(
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

        self.trades.append(trade)
        self.stats_state.trades_created += 1
        self.stats_state.realized_pnl += net_pnl
        return trade

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

        This is a fallback protective simulation. In a full pipeline, SL/TP
        orders can also be simulated by ExecutionSimulator from explicit
        protective orders.
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

            liquidated = self.cost_model.liquidation.is_liquidated(
                side=position.side,
                mark_price=position.mark_price,
                liquidation_price=position.liquidation_price,
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

        if position.side.lower() in {"buy", "long"}:
            return candle.low <= position.stop_loss

        if position.side.lower() in {"sell", "short"}:
            return candle.high >= position.stop_loss

        return False

    @staticmethod
    def _take_profit_hit(position: SimulatedPosition, candle: HistoricalCandle) -> bool:
        if position.take_profit is None:
            return False

        if position.side.lower() in {"buy", "long"}:
            return candle.high >= position.take_profit

        if position.side.lower() in {"sell", "short"}:
            return candle.low <= position.take_profit

        return False

    # ------------------------------------------------------------------
    # Account / equity
    # ------------------------------------------------------------------

    def _apply_fill_costs_to_balance(self, fill: SimulatedFill) -> None:
        """
        Apply costs from fill.

        Entry costs are applied in open/reduce/close methods too, so this
        method only keeps aggregate fields synchronized.
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

        peak = self.stats_state.max_equity
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

        peak = max(self.stats_state.max_equity, self.balance.equity)
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
            open_positions=len([position for position in self.positions.values() if position.is_open]),
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
            "net_realized_pnl": position.net_realized_pnl,
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

    async def _emit(self, topic: str, payload: dict[str, Any]) -> None:
        if not self.config.emit_position_events:
            return

        if self.event_bus is None:
            return

        emit = getattr(self.event_bus, "emit", None) or getattr(self.event_bus, "publish", None)

        if emit is None:
            return

        result = emit(topic, payload)

        if hasattr(result, "__await__"):
            await result

    async def _emit_best_effort(self, topic: str, payload: dict[str, Any]) -> None:
        try:
            await self._emit(topic, payload)
        except Exception as exc:
            self.logger.warning("Failed to emit %s: %s", topic, exc)

    # ------------------------------------------------------------------
    # Payload conversion
    # ------------------------------------------------------------------

    def _payload_to_fill(self, payload: dict[str, Any]) -> SimulatedFill:
        fill_payload = payload.get("fill")

        if isinstance(fill_payload, dict):
            return SimulatedFill(
                fill_id=str(fill_payload.get("fill_id") or payload.get("fill_id") or new_id("sim_fill")),
                order_id=str(fill_payload.get("order_id") or payload.get("order_id") or ""),
                run_id=fill_payload.get("run_id") or payload.get("run_id"),
                signal_id=fill_payload.get("signal_id") or payload.get("signal_id"),
                position_id=fill_payload.get("position_id") or payload.get("position_id"),
                exchange=str(fill_payload.get("exchange") or payload.get("exchange") or self.config.metadata.get("exchange", "binance")),
                symbol=str(fill_payload.get("symbol") or payload.get("symbol") or ""),
                market_type=str(fill_payload.get("market_type") or payload.get("market_type") or "usdm_futures"),
                side=str(fill_payload.get("side") or payload.get("side") or ""),
                price=float(fill_payload.get("price") or payload.get("fill_price") or 0.0),
                quantity=float(fill_payload.get("quantity") or payload.get("fill_quantity") or 0.0),
                notional=float(fill_payload.get("notional") or payload.get("fill_notional") or 0.0),
                fee=float(fill_payload.get("fee") or payload.get("fee") or 0.0),
                fee_asset=str(fill_payload.get("fee_asset") or payload.get("fee_asset") or self.config.quote_currency),
                slippage=float(fill_payload.get("slippage") or payload.get("fill_slippage") or 0.0),
                slippage_bps=float(fill_payload.get("slippage_bps") or payload.get("fill_slippage_bps") or 0.0),
                liquidity_type=fill_payload.get("liquidity_type") or payload.get("liquidity_type"),
                timestamp_ms=int(fill_payload.get("timestamp_ms") or payload.get("timestamp_ms") or payload.get("filled_at_ms") or self._now_ms()),
                source_event_id=fill_payload.get("source_event_id") or payload.get("replay_event_id"),
                metadata={
                    **dict(fill_payload.get("metadata") or {}),
                    "strategy_name": payload.get("strategy_name") or dict(fill_payload.get("metadata") or {}).get("strategy_name"),
                    "order_type": payload.get("order_type"),
                    "reduce_only": payload.get("reduce_only"),
                    "close_position": payload.get("close_position"),
                },
            )

        return SimulatedFill(
            fill_id=str(payload.get("fill_id") or new_id("sim_fill")),
            order_id=str(payload.get("order_id") or ""),
            run_id=payload.get("run_id"),
            signal_id=payload.get("signal_id"),
            position_id=payload.get("position_id"),
            exchange=str(payload.get("exchange") or "binance"),
            symbol=str(payload.get("symbol") or ""),
            market_type=str(payload.get("market_type") or "usdm_futures"),
            side=str(payload.get("side") or ""),
            price=float(payload.get("fill_price") or payload.get("price") or payload.get("average_fill_price") or 0.0),
            quantity=float(payload.get("fill_quantity") or payload.get("filled_quantity") or 0.0),
            notional=float(payload.get("fill_notional") or 0.0),
            fee=float(payload.get("fee") or 0.0),
            fee_asset=str(payload.get("fee_asset") or self.config.quote_currency),
            slippage=float(payload.get("fill_slippage") or payload.get("slippage") or 0.0),
            slippage_bps=float(payload.get("fill_slippage_bps") or 0.0),
            liquidity_type=payload.get("liquidity_type"),
            timestamp_ms=int(payload.get("timestamp_ms") or payload.get("filled_at_ms") or self._now_ms()),
            source_event_id=payload.get("replay_event_id"),
            metadata={
                "strategy_name": payload.get("strategy_name"),
                "order_type": payload.get("order_type"),
                "reduce_only": payload.get("reduce_only"),
                "close_position": payload.get("close_position"),
            },
        )

    def _payload_to_candle(self, payload: dict[str, Any]) -> HistoricalCandle:
        return HistoricalCandle(
            exchange=str(payload.get("exchange") or "binance"),
            symbol=str(payload.get("symbol") or ""),
            market_type=str(payload.get("market_type") or "usdm_futures"),
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

    def _payload_to_funding(self, payload: dict[str, Any]) -> HistoricalFundingRecord:
        return HistoricalFundingRecord(
            exchange=str(payload.get("exchange") or "binance"),
            symbol=str(payload.get("symbol") or ""),
            market_type=str(payload.get("market_type") or "usdm_futures"),
            timestamp_ms=int(payload.get("timestamp_ms") or self._now_ms()),
            received_at_ms=int(payload.get("received_at_ms") or payload.get("timestamp_ms") or self._now_ms()),
            funding_rate=float(payload.get("funding_rate") or payload.get("rate") or 0.0),
            predicted_rate=self._optional_float(payload.get("predicted_rate")),
            mark_price=self._optional_float(payload.get("mark_price")),
            index_price=self._optional_float(payload.get("index_price")),
            next_funding_time_ms=self._optional_int(payload.get("next_funding_time_ms")),
            source="market_replay",
            metadata=dict(payload.get("metadata") or {}),
        )

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _find_position_for_fill(self, fill: SimulatedFill) -> SimulatedPosition | None:
        """
        Find compatible open position for fill.

        In NETTING mode there is one net position per exchange/market/symbol.
        In HEDGE mode we separate long/short sides.
        """

        fill_position_side = self._position_side_from_fill_side(fill.side)

        for position in self.positions.values():
            if not position.is_open:
                continue

            if position.exchange != fill.exchange:
                continue

            if position.market_type != fill.market_type:
                continue

            if position.symbol != fill.symbol:
                continue

            if self.config.position_accounting_mode == PositionAccountingMode.HEDGE:
                if position.side == fill_position_side:
                    return position
                continue

            return position

        return None

    def _matching_open_positions(
        self,
        *,
        position_id: str | None = None,
        symbol: str | None = None,
    ) -> list[SimulatedPosition]:
        result: list[SimulatedPosition] = []

        for position in self.positions.values():
            if not position.is_open:
                continue

            if position_id is not None and position.position_id != position_id:
                continue

            if symbol is not None and position.symbol != str(symbol).upper():
                continue

            result.append(position)

        return result

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _position_side_from_fill_side(side: str) -> str:
        value = side.lower()

        if value in {"buy", "long"}:
            return "long"

        if value in {"sell", "short"}:
            return "short"

        raise SimulatedPositionValidationError(
            "Unsupported fill side.",
            details={"side": side},
        )

    @staticmethod
    def _calculate_realized_pnl(
        *,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
    ) -> float:
        if entry_price <= 0 or exit_price <= 0:
            raise PositionAccountingError(
                "entry_price and exit_price must be positive.",
                details={
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                },
            )

        if quantity <= 0:
            raise PositionAccountingError(
                "quantity must be positive.",
                details={"quantity": quantity},
            )

        if side.lower() in {"buy", "long"}:
            return (exit_price - entry_price) * quantity

        if side.lower() in {"sell", "short"}:
            return (entry_price - exit_price) * quantity

        raise PositionAccountingError(
            "Unsupported position side.",
            details={"side": side},
        )

    @staticmethod
    def _estimate_initial_risk(position: SimulatedPosition) -> float:
        if position.stop_loss is None:
            return position.margin if position.margin > 0 else 0.0

        if position.side.lower() in {"buy", "long"}:
            risk_per_unit = max(0.0, position.entry_price - position.stop_loss)
        else:
            risk_per_unit = max(0.0, position.stop_loss - position.entry_price)

        return risk_per_unit * position.quantity

    # ------------------------------------------------------------------
    # General utilities
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
            raise PositionSimulatorNotReadyError(
                "PositionSimulator is not running. Call start() first."
            )

    def _now_ms(self) -> int:
        if self.clock is not None and self.clock.started:
            return self.clock.timestamp_ms()
        return timestamp_ms(utcnow())

    @staticmethod
    def _market_key(exchange: str, market_type: str, symbol: str) -> str:
        return f"{exchange.lower()}:{market_type.lower()}:{symbol.upper()}"

    @staticmethod
    def _extract_float(
        payload: dict[str, Any],
        keys: list[str],
        *,
        default: float,
    ) -> float:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    return default
        return default

    @staticmethod
    def _extract_optional_float(
        payload: dict[str, Any],
        keys: list[str],
    ) -> float | None:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    return None

        # Try nested metadata / plans.
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in keys:
                value = metadata.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except Exception:
                        return None

        return None

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

    # ------------------------------------------------------------------
    # Public snapshots
    # ------------------------------------------------------------------

    def open_positions(self) -> list[SimulatedPosition]:
        return [position for position in self.positions.values() if position.is_open]

    def all_positions(self) -> list[SimulatedPosition]:
        return list(self.positions.values()) + list(self.closed_positions)

    def stats(self) -> dict[str, Any]:
        payload = self.stats_state.to_dict()
        payload.update(
            {
                "running": self._running,
                "registered": self._registered,
                "open_positions": len(self.open_positions()),
                "closed_positions": len(self.closed_positions),
                "trades": len(self.trades),
                "equity_points": len(self.equity_curve),
                "records": len(self.records),
                "balance": self.balance.to_dict(),
            }
        )
        return payload


__all__ = [
    "PositionMarketState",
    "PositionSimulatorStats",
    "PositionSimulator",
]