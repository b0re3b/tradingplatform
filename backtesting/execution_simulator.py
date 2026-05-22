from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
from typing import Any
from uuid import uuid4

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger

from backtesting.config import BacktestConfig
from backtesting.exceptions import BacktestExecutionError, BacktestSafetyError
from backtesting.models import ClosedTrade, PaperOrder, PaperPosition
from backtesting.utils import bps_to_fraction, decimal_from, normalize_symbol


class BacktestExecutionSimulator:
    """
    Paper execution service with production-compatible EventBus contract.

    It listens to the same inbound topics as live execution:
    - signal.confirmed
    - risk.position_close_requested
    - risk.position_reduce_requested
    - risk.kill_switch

    It emits the same domain topics expected by RiskManager:
    - execution.order_submitted
    - execution.order_filled / execution.order_rejected / execution.order_cancelled / execution.order_failed
    - position.opened / position.updated / position.closed

    It never receives a live Binance client and never calls live order endpoints.
    """

    def __init__(self, *, config: BacktestConfig, event_bus: EventBus) -> None:
        if config.execution_mode != "paper":
            raise BacktestSafetyError("BacktestExecutionSimulator supports only paper mode.")
        self._config = config
        self._event_bus = event_bus
        self._logger = get_logger(__name__, service="backtesting.execution", event_type="paper_execution")

        self._subscriptions: list[Subscription] = []
        self._latest_price: dict[str, Decimal] = {}
        self._latest_candle: dict[str, dict[str, Any]] = {}
        self._positions: dict[str, PaperPosition] = {}
        self._orders: dict[str, PaperOrder] = {}
        self._closed_trades: list[ClosedTrade] = []

        self.balance: Decimal = config.initial_balance_usd
        self.equity: Decimal = config.initial_balance_usd
        self.total_fees: Decimal = Decimal("0")
        self.total_slippage: Decimal = Decimal("0")
        self.kill_switch_active = False

    @property
    def closed_trades(self) -> list[ClosedTrade]:
        return list(self._closed_trades)

    def register(self) -> None:
        if self._subscriptions:
            return
        self._subscriptions.extend(
            [
                self._event_bus.subscribe("market.candle", self._on_market_candle, name="backtest_execution_on_candle"),
                self._event_bus.subscribe("market.mark_price", self._on_mark_price, name="backtest_execution_on_mark_price"),
                self._event_bus.subscribe("signal.confirmed", self._on_signal_confirmed, name="backtest_execution_on_signal_confirmed"),
                self._event_bus.subscribe("risk.position_close_requested", self._on_close_requested, name="backtest_execution_on_close_requested"),
                self._event_bus.subscribe("risk.position_reduce_requested", self._on_reduce_requested, name="backtest_execution_on_reduce_requested"),
                self._event_bus.subscribe("risk.kill_switch", self._on_kill_switch, name="backtest_execution_on_kill_switch"),
            ]
        )

    def unregister(self) -> None:
        for subscription in self._subscriptions:
            self._event_bus.unsubscribe(subscription)
        self._subscriptions.clear()

    async def start(self) -> None:
        self.register()
        await self._emit("execution.simulator.started", {"mode": "backtest", "balance": float(self.balance)})

    async def stop(self) -> None:
        self.unregister()
        await self._emit("execution.simulator.stopped", {"mode": "backtest", "balance": float(self.balance), "equity": float(self.equity)})

    async def close_all_at_last_price(self, *, reason: str = "backtest_final_mark") -> None:
        for key in list(self._positions.keys()):
            position = self._positions.get(key)
            if position is None:
                continue
            price = self._latest_price.get(position.symbol)
            if price is None:
                continue
            await self._close_position(position, exit_price=price, timestamp_ms=max(self._latest_timestamp_ms(), position.opened_at_ms), reason=reason)

    async def _on_market_candle(self, event: Event) -> None:
        payload = self._payload(event)
        symbol = normalize_symbol(str(payload.get("symbol") or ""))
        if not symbol:
            return
        close = decimal_from(payload.get("close"))
        self._latest_price[symbol] = close
        self._latest_candle[symbol] = payload
        await self._mark_positions(symbol=symbol, price=close, timestamp_ms=int(payload.get("timestamp_ms") or payload.get("close_time_ms") or 0))

    async def _on_mark_price(self, event: Event) -> None:
        payload = self._payload(event)
        symbol = normalize_symbol(str(payload.get("symbol") or ""))
        close = decimal_from(payload.get("close"))
        if symbol and close > 0:
            self._latest_price[symbol] = close
            await self._mark_positions(symbol=symbol, price=close, timestamp_ms=int(payload.get("timestamp_ms") or 0))

    async def _on_signal_confirmed(self, event: Event) -> None:
        if self.kill_switch_active:
            await self._reject(event, reason="kill_switch_active")
            return

        payload = self._payload(event)
        symbol = normalize_symbol(str(payload.get("symbol") or ""))
        if not symbol:
            await self._reject(event, reason="missing_symbol")
            return

        price = self._execution_price(symbol=symbol, side=str(payload.get("side") or "LONG"))
        if price <= 0:
            await self._reject(event, reason="no_market_price_available")
            return

        side = str(payload.get("side") or "LONG").upper()
        size = decimal_from(payload.get("final_size"))
        if size <= 0:
            await self._reject(event, reason="non_positive_final_size")
            return

        timestamp_ms = self._latest_timestamp_ms(symbol)
        order_id = f"bt_order_{uuid4().hex}"
        client_order_id = f"bt_{payload.get('signal_id') or uuid4().hex}"
        fee = self._fee(price=price, size=size)
        slippage = self._slippage(price=price, size=size)
        self.total_fees += fee
        self.total_slippage += slippage

        order = PaperOrder(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side="BUY" if side in {"LONG", "BUY"} else "SELL",
            position_side="LONG" if side in {"LONG", "BUY"} else "SHORT",
            quantity=size,
            status="FILLED",
            submitted_at_ms=timestamp_ms,
            filled_at_ms=timestamp_ms,
            avg_price=price,
            fee=fee,
            signal_id=payload.get("signal_id"),
            strategy_name=payload.get("strategy_name"),
            metadata=dict(payload),
        )
        self._orders[order_id] = order

        await self._emit("execution.order_submitted", self._order_payload(order, payload, status="SUBMITTED"))
        await self._emit("execution.order_filled", self._order_payload(order, payload, status="FILLED"))

        position_key = self._position_key(symbol, order.position_side)
        if position_key in self._positions:
            # Keep simulator conservative: one position per symbol/side. Live pyramiding can be added later.
            await self._reject(event, reason="position_already_open")
            return

        leverage = decimal_from(payload.get("final_leverage"), Decimal("1"))
        margin = decimal_from(payload.get("final_margin"), (price * size) / max(leverage, Decimal("1")))
        position = PaperPosition(
            position_id=f"bt_pos_{uuid4().hex}",
            symbol=symbol,
            side=order.position_side,
            size=size,
            entry_price=price,
            leverage=leverage,
            margin_used=margin,
            opened_at_ms=timestamp_ms,
            signal_id=payload.get("signal_id"),
            strategy_name=payload.get("strategy_name"),
            stop_loss=decimal_from(payload.get("stop_loss")) if payload.get("stop_loss") is not None else None,
            take_profit=decimal_from(payload.get("take_profit")) if payload.get("take_profit") is not None else None,
            last_mark_price=price,
        )
        self._positions[position_key] = position
        await self._emit("position.opened", self._position_payload(position, timestamp_ms=timestamp_ms))

    async def _on_close_requested(self, event: Event) -> None:
        payload = self._payload(event)
        symbol = normalize_symbol(str(payload.get("symbol") or ""))
        side = str(payload.get("side") or payload.get("position_side") or "").upper() or None
        for key, position in list(self._positions.items()):
            if symbol and position.symbol != symbol:
                continue
            if side and position.side != side:
                continue
            price = self._latest_price.get(position.symbol)
            if price is None:
                continue
            await self._close_position(position, exit_price=price, timestamp_ms=self._latest_timestamp_ms(position.symbol), reason="risk_close_requested")

    async def _on_reduce_requested(self, event: Event) -> None:
        # Minimal safe implementation: treat reduce as close if no partial execution model is configured.
        await self._on_close_requested(event)

    async def _on_kill_switch(self, event: Event) -> None:
        self.kill_switch_active = True
        await self.close_all_at_last_price(reason="kill_switch")

    async def _mark_positions(self, *, symbol: str, price: Decimal, timestamp_ms: int) -> None:
        for position in list(self._positions.values()):
            if position.symbol != symbol:
                continue
            position.last_mark_price = price
            position.unrealized_pnl = self._pnl(position=position, exit_price=price)
            await self._emit("position.updated", self._position_payload(position, timestamp_ms=timestamp_ms))

            hit_sl = position.stop_loss is not None and (
                price <= position.stop_loss if position.side == "LONG" else price >= position.stop_loss
            )
            hit_tp = position.take_profit is not None and (
                price >= position.take_profit if position.side == "LONG" else price <= position.take_profit
            )
            if hit_sl or hit_tp:
                await self._close_position(position, exit_price=price, timestamp_ms=timestamp_ms, reason="stop_loss" if hit_sl else "take_profit")

        self.equity = self.balance + sum(p.unrealized_pnl for p in self._positions.values())

    async def _close_position(self, position: PaperPosition, *, exit_price: Decimal, timestamp_ms: int, reason: str) -> None:
        key = self._position_key(position.symbol, position.side)
        if key not in self._positions:
            return
        gross_pnl = self._pnl(position=position, exit_price=exit_price)
        exit_fee = self._fee(price=exit_price, size=position.size)
        slippage = self._slippage(price=exit_price, size=position.size)
        net_pnl = gross_pnl - exit_fee - slippage
        self.total_fees += exit_fee
        self.total_slippage += slippage
        self.balance += net_pnl
        position.realized_pnl += net_pnl
        position.unrealized_pnl = Decimal("0")
        self._positions.pop(key, None)

        trade = ClosedTrade(
            trade_id=f"bt_trade_{uuid4().hex}",
            symbol=position.symbol,
            side=position.side,
            strategy_name=position.strategy_name,
            timeframe=str(position.metadata.get("timeframe")) if hasattr(position, "metadata") else None,
            entry_time_ms=position.opened_at_ms,
            exit_time_ms=timestamp_ms,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            gross_pnl=gross_pnl,
            fees=exit_fee,
            slippage=slippage,
            net_pnl=net_pnl,
            signal_id=position.signal_id,
        )
        self._closed_trades.append(trade)
        await self._emit("position.closed", {**self._position_payload(position, timestamp_ms=timestamp_ms), "exit_price": float(exit_price), "realized_pnl": float(net_pnl), "close_reason": reason})
        self.equity = self.balance + sum(p.unrealized_pnl for p in self._positions.values())

    def _execution_price(self, *, symbol: str, side: str) -> Decimal:
        price = self._latest_price.get(symbol, Decimal("0"))
        slip = bps_to_fraction(self._config.slippage_bps)
        if side.upper() in {"LONG", "BUY"}:
            return price * (Decimal("1") + slip)
        return price * (Decimal("1") - slip)

    def _fee(self, *, price: Decimal, size: Decimal) -> Decimal:
        return price * size * bps_to_fraction(self._config.taker_fee_bps)

    def _slippage(self, *, price: Decimal, size: Decimal) -> Decimal:
        return price * size * bps_to_fraction(self._config.slippage_bps)

    @staticmethod
    def _pnl(*, position: PaperPosition, exit_price: Decimal) -> Decimal:
        if position.side == "LONG":
            return (exit_price - position.entry_price) * position.size
        return (position.entry_price - exit_price) * position.size

    async def _reject(self, event: Event, *, reason: str) -> None:
        payload = self._payload(event)
        await self._emit(
            "execution.order_rejected",
            {
                **payload,
                "status": "REJECTED",
                "reason": reason,
                "mode": "backtest",
                "paper": True,
            },
            priority=EventPriority.HIGH,
        )

    def _order_payload(self, order: PaperOrder, source: dict[str, Any], *, status: str) -> dict[str, Any]:
        return {
            "exchange": self._config.exchange,
            "market_type": self._config.market_type,
            "symbol": order.symbol,
            "order_id": order.order_id,
            "client_order_id": order.client_order_id,
            "side": order.side,
            "position_side": order.position_side,
            "order_type": "MARKET",
            "status": status,
            "quantity": float(order.quantity),
            "executed_qty": float(order.quantity if status == "FILLED" else Decimal("0")),
            "avg_price": float(order.avg_price or Decimal("0")),
            "fee": float(order.fee),
            "signal_id": order.signal_id,
            "strategy_name": order.strategy_name,
            "reservation_id": source.get("reservation_id"),
            "mode": "backtest",
            "paper": True,
            "metadata": {"source_signal": source},
        }

    def _position_payload(self, position: PaperPosition, *, timestamp_ms: int) -> dict[str, Any]:
        mark_price = position.last_mark_price or position.entry_price
        return {
            "exchange": self._config.exchange,
            "market_type": self._config.market_type,
            "symbol": position.symbol,
            "side": position.side,
            "position_side": position.side,
            "size": float(position.size),
            "entry_price": float(position.entry_price),
            "mark_price": float(mark_price),
            "notional_value": float(mark_price * position.size),
            "leverage": float(position.leverage),
            "margin_used": float(position.margin_used),
            "risk_amount": 0.0,
            "stop_loss": float(position.stop_loss) if position.stop_loss is not None else None,
            "take_profit": float(position.take_profit) if position.take_profit is not None else None,
            "tier": None,
            "strategy_name": position.strategy_name,
            "signal_id": position.signal_id,
            "position_id": position.position_id,
            "realized_pnl": float(position.realized_pnl),
            "unrealized_pnl": float(position.unrealized_pnl),
            "timestamp_ms": timestamp_ms,
            "mode": "backtest",
            "paper": True,
        }

    def _latest_timestamp_ms(self, symbol: str | None = None) -> int:
        if symbol and symbol in self._latest_candle:
            return int(self._latest_candle[symbol].get("timestamp_ms") or self._latest_candle[symbol].get("close_time_ms") or 0)
        timestamps = [int(c.get("timestamp_ms") or c.get("close_time_ms") or 0) for c in self._latest_candle.values()]
        return max(timestamps) if timestamps else 0

    @staticmethod
    def _position_key(symbol: str, side: str) -> str:
        return f"{symbol}:{side}"

    @staticmethod
    def _payload(event: Event) -> dict[str, Any]:
        return event.payload if isinstance(event.payload, dict) else {}

    async def _emit(self, topic: str, payload: dict[str, Any], *, priority: EventPriority = EventPriority.NORMAL) -> None:
        await self._event_bus.publish(
            Event(
                topic=topic,
                payload=payload,
                priority=priority,
                source="backtesting.execution_simulator",
                headers={"mode": "backtest", "paper": True},
            )
        )
