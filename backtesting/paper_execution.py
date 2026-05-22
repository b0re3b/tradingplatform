from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any
from uuid import uuid4

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger

from backtesting.config import BacktestConfig
from backtesting.exceptions import BacktestSafetyError
from backtesting.models import ClosedTrade, PaperOrder, PaperPosition
from backtesting.utils import bps_to_fraction, decimal_from, normalize_symbol


class BacktestPaperExecution:
    """
    Paper execution service compatible with production risk/execution event contracts.

    Inbound:
    - market.candle
    - market.mark_price
    - signal.confirmed
    - risk.position_close_requested
    - risk.position_reduce_requested
    - risk.kill_switch

    Outbound:
    - execution.order_submitted
    - execution.order_filled / rejected / cancelled / failed
    - position.opened / updated / closed
    - portfolio.updated
    """

    def __init__(self, *, config: BacktestConfig, event_bus: EventBus) -> None:
        if config.execution_mode != "paper":
            raise BacktestSafetyError("BacktestPaperExecution supports only paper mode.")
        self._config = config
        self._event_bus = event_bus
        self._logger = get_logger(
            __name__,
            service="backtesting.paper_execution",
            event_type="paper_execution",
        )

        self._subscriptions: list[Subscription] = []
        self._latest_price: dict[str, Decimal] = {}
        self._latest_timestamp: dict[str, int] = {}
        self._positions: dict[str, PaperPosition] = {}
        self._orders: dict[str, PaperOrder] = {}
        self._closed_trades: list[ClosedTrade] = []

        self.balance: Decimal = config.initial_balance_usd
        self.equity: Decimal = config.initial_balance_usd
        self.total_fees: Decimal = Decimal("0")
        self.total_slippage: Decimal = Decimal("0")
        self.kill_switch_active: bool = False

    @property
    def closed_trades(self) -> list[ClosedTrade]:
        return list(self._closed_trades)

    def register(self) -> None:
        if self._subscriptions:
            return
        self._subscriptions.extend(
            [
                self._event_bus.subscribe("market.candle", self._on_market_price_event, name="backtest_paper_on_candle"),
                self._event_bus.subscribe("market.mark_price", self._on_market_price_event, name="backtest_paper_on_mark_price"),
                self._event_bus.subscribe("signal.confirmed", self._on_signal_confirmed, name="backtest_paper_on_signal_confirmed"),
                self._event_bus.subscribe("risk.position_close_requested", self._on_close_requested, name="backtest_paper_on_close_requested"),
                self._event_bus.subscribe("risk.position_reduce_requested", self._on_reduce_requested, name="backtest_paper_on_reduce_requested"),
                self._event_bus.subscribe("risk.kill_switch", self._on_kill_switch, name="backtest_paper_on_kill_switch"),
            ]
        )

    def unregister(self) -> None:
        for subscription in self._subscriptions:
            self._event_bus.unsubscribe(subscription)
        self._subscriptions.clear()

    async def start(self) -> None:
        self.register()
        await self._emit_portfolio_update(reason="paper_execution_started")

    async def stop(self) -> None:
        await self._emit_portfolio_update(reason="paper_execution_stopped")
        self.unregister()

    async def close_all_at_last_price(self, *, reason: str = "backtest_end") -> None:
        for position in list(self._positions.values()):
            price = self._latest_price.get(position.symbol)
            if price is None:
                continue
            await self._close_position(
                position,
                exit_price=price,
                timestamp_ms=max(self._latest_timestamp_ms(position.symbol), position.opened_at_ms),
                reason=reason,
            )

    async def _on_market_price_event(self, event: Event) -> None:
        payload = self._payload(event)
        symbol = normalize_symbol(str(payload.get("symbol") or payload.get("exchange_symbol") or ""))
        if not symbol:
            return

        price = decimal_from(payload.get("mark_price") or payload.get("close") or payload.get("price"))
        if price <= 0:
            return

        timestamp_ms = int(payload.get("timestamp_ms") or payload.get("close_time_ms") or 0)
        self._latest_price[symbol] = price
        self._latest_timestamp[symbol] = timestamp_ms

        await self._mark_positions(symbol=symbol, price=price, timestamp_ms=timestamp_ms)

    async def _on_signal_confirmed(self, event: Event) -> None:
        payload = self._payload(event)

        if self.kill_switch_active:
            await self._reject(payload, reason="kill_switch_active")
            return

        symbol = normalize_symbol(str(payload.get("symbol") or payload.get("exchange_symbol") or ""))
        if not symbol:
            await self._reject(payload, reason="missing_symbol")
            return

        raw_side = str(payload.get("side") or payload.get("position_side") or "LONG").upper()
        position_side = "LONG" if raw_side in {"LONG", "BUY", "BULLISH"} else "SHORT"
        position_key = self._position_key(symbol, position_side)

        if position_key in self._positions:
            await self._reject(payload, reason="position_already_open")
            return

        size = decimal_from(
            payload.get("final_size")
            or payload.get("size")
            or payload.get("quantity")
            or payload.get("qty")
        )
        if size <= 0:
            await self._reject(payload, reason="non_positive_size")
            return

        price = self._execution_price(symbol=symbol, position_side=position_side)
        if price <= 0:
            await self._reject(payload, reason="no_market_price_available")
            return

        timestamp_ms = self._latest_timestamp_ms(symbol)
        leverage = decimal_from(payload.get("final_leverage"), Decimal("1"))
        margin = decimal_from(payload.get("final_margin"), (price * size) / max(leverage, Decimal("1")))
        entry_fee = self._fee(price=price, size=size)
        entry_slippage = self._slippage(price=price, size=size)

        self.balance -= entry_fee
        self.total_fees += entry_fee
        self.total_slippage += entry_slippage

        order = PaperOrder(
            order_id=f"bt_order_{uuid4().hex}",
            client_order_id=f"bt_{payload.get('signal_id') or uuid4().hex}",
            symbol=symbol,
            side="BUY" if position_side == "LONG" else "SELL",
            position_side=position_side,
            quantity=size,
            status="FILLED",
            submitted_at_ms=timestamp_ms,
            filled_at_ms=timestamp_ms,
            avg_price=price,
            fee=entry_fee,
            slippage=entry_slippage,
            signal_id=str(payload.get("signal_id")) if payload.get("signal_id") is not None else None,
            strategy_name=str(payload.get("strategy_name")) if payload.get("strategy_name") is not None else None,
            metadata=dict(payload),
        )
        self._orders[order.order_id] = order

        position = PaperPosition(
            position_id=f"bt_pos_{uuid4().hex}",
            symbol=symbol,
            side=position_side,
            size=size,
            entry_price=price,
            leverage=leverage,
            margin_used=margin,
            opened_at_ms=timestamp_ms,
            signal_id=order.signal_id,
            strategy_name=order.strategy_name,
            timeframe=str(payload.get("timeframe")) if payload.get("timeframe") is not None else None,
            stop_loss=decimal_from(payload.get("stop_loss")) if payload.get("stop_loss") is not None else None,
            take_profit=decimal_from(payload.get("take_profit")) if payload.get("take_profit") is not None else None,
            last_mark_price=price,
            entry_fee=entry_fee,
            entry_slippage=entry_slippage,
            metadata=dict(payload),
        )
        self._positions[position_key] = position

        await self._emit("execution.order_submitted", self._order_payload(order, status="SUBMITTED"))
        await self._emit("execution.order_filled", self._order_payload(order, status="FILLED"))
        await self._emit("position.opened", self._position_payload(position, timestamp_ms=timestamp_ms))
        await self._emit_portfolio_update(reason="position_opened", timestamp_ms=timestamp_ms)

    async def _on_close_requested(self, event: Event) -> None:
        payload = self._payload(event)
        symbol = normalize_symbol(str(payload.get("symbol") or payload.get("exchange_symbol") or ""))
        side = str(payload.get("side") or payload.get("position_side") or "").upper()
        side = "LONG" if side in {"LONG", "BUY"} else "SHORT" if side in {"SHORT", "SELL"} else ""

        for position in list(self._positions.values()):
            if symbol and position.symbol != symbol:
                continue
            if side and position.side != side:
                continue
            price = self._latest_price.get(position.symbol)
            if price is not None:
                await self._close_position(
                    position,
                    exit_price=price,
                    timestamp_ms=self._latest_timestamp_ms(position.symbol),
                    reason="risk_close_requested",
                )

    async def _on_reduce_requested(self, event: Event) -> None:
        # Conservative initial implementation: reduce request closes the position.
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
                await self._close_position(
                    position,
                    exit_price=price,
                    timestamp_ms=timestamp_ms,
                    reason="stop_loss" if hit_sl else "take_profit",
                )

        await self._emit_portfolio_update(reason="mark_to_market", timestamp_ms=timestamp_ms)

    async def _close_position(
        self,
        position: PaperPosition,
        *,
        exit_price: Decimal,
        timestamp_ms: int,
        reason: str,
    ) -> None:
        key = self._position_key(position.symbol, position.side)
        if key not in self._positions:
            return

        gross_pnl = self._pnl(position=position, exit_price=exit_price)
        exit_fee = self._fee(price=exit_price, size=position.size)
        exit_slippage = self._slippage(price=exit_price, size=position.size)
        total_fees = position.entry_fee + exit_fee
        total_slippage = position.entry_slippage + exit_slippage
        net_pnl = gross_pnl - exit_fee - exit_slippage

        self.total_fees += exit_fee
        self.total_slippage += exit_slippage
        self.balance += net_pnl

        position.exit_fee = exit_fee
        position.exit_slippage = exit_slippage
        position.realized_pnl = net_pnl
        position.unrealized_pnl = Decimal("0")
        position.last_mark_price = exit_price

        self._positions.pop(key, None)

        trade = ClosedTrade(
            trade_id=f"bt_trade_{uuid4().hex}",
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            fees=total_fees,
            slippage=total_slippage,
            opened_at_ms=position.opened_at_ms,
            closed_at_ms=timestamp_ms,
            holding_ms=max(0, timestamp_ms - position.opened_at_ms),
            signal_id=position.signal_id,
            strategy_name=position.strategy_name,
            timeframe=position.timeframe,
            reason=reason,
            metadata=dict(position.metadata),
        )
        self._closed_trades.append(trade)

        await self._emit("position.closed", trade.to_payload())
        await self._emit_portfolio_update(reason="position_closed", timestamp_ms=timestamp_ms)

    async def _reject(self, payload: dict[str, Any], *, reason: str) -> None:
        await self._emit(
            "execution.order_rejected",
            {
                "reason": reason,
                "symbol": payload.get("symbol"),
                "signal_id": payload.get("signal_id"),
                "strategy_name": payload.get("strategy_name"),
                "timestamp_ms": self._latest_timestamp_ms(),
                "mode": "backtest",
                "payload": dict(payload),
            },
        )

    def _execution_price(self, *, symbol: str, position_side: str) -> Decimal:
        price = self._latest_price.get(symbol, Decimal("0"))
        if price <= 0:
            return Decimal("0")
        slip = bps_to_fraction(self._config.slippage_bps)
        return price * (Decimal("1") + slip if position_side == "LONG" else Decimal("1") - slip)

    def _fee(self, *, price: Decimal, size: Decimal) -> Decimal:
        return abs(price * size) * bps_to_fraction(self._config.taker_fee_bps)

    def _slippage(self, *, price: Decimal, size: Decimal) -> Decimal:
        return abs(price * size) * bps_to_fraction(self._config.slippage_bps)

    @staticmethod
    def _pnl(*, position: PaperPosition, exit_price: Decimal) -> Decimal:
        if position.side == "LONG":
            return (exit_price - position.entry_price) * position.size
        return (position.entry_price - exit_price) * position.size

    def _position_key(self, symbol: str, side: str) -> str:
        return f"{symbol}:{side}"

    def _latest_timestamp_ms(self, symbol: str | None = None) -> int:
        if symbol and symbol in self._latest_timestamp:
            return self._latest_timestamp[symbol]
        if self._latest_timestamp:
            return max(self._latest_timestamp.values())
        return 0

    def _position_payload(self, position: PaperPosition, *, timestamp_ms: int) -> dict[str, Any]:
        return {
            "position_id": position.position_id,
            "symbol": position.symbol,
            "side": position.side,
            "size": float(position.size),
            "entry_price": float(position.entry_price),
            "mark_price": float(position.last_mark_price or position.entry_price),
            "notional_value": float((position.last_mark_price or position.entry_price) * position.size),
            "leverage": float(position.leverage),
            "margin_used": float(position.margin_used),
            "risk_amount": float(position.margin_used),
            "stop_loss": float(position.stop_loss) if position.stop_loss is not None else None,
            "take_profit": float(position.take_profit) if position.take_profit is not None else None,
            "tier": position.metadata.get("final_tier") or position.metadata.get("tier"),
            "strategy_name": position.strategy_name,
            "signal_id": position.signal_id,
            "timeframe": position.timeframe,
            "realized_pnl": float(position.realized_pnl),
            "unrealized_pnl": float(position.unrealized_pnl),
            "timestamp_ms": timestamp_ms,
            "mode": "backtest",
            "balance": float(self.balance),
            "equity": float(self._portfolio_equity()),
        }

    def _order_payload(self, order: PaperOrder, *, status: str) -> dict[str, Any]:
        return {
            "order_id": order.order_id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "position_side": order.position_side,
            "quantity": float(order.quantity),
            "status": status,
            "submitted_at_ms": order.submitted_at_ms,
            "filled_at_ms": order.filled_at_ms,
            "avg_price": float(order.avg_price or 0),
            "fee": float(order.fee),
            "slippage": float(order.slippage),
            "signal_id": order.signal_id,
            "strategy_name": order.strategy_name,
            "timestamp_ms": order.filled_at_ms or order.submitted_at_ms,
            "mode": "backtest",
        }

    async def _emit_portfolio_update(self, *, reason: str, timestamp_ms: int | None = None) -> None:
        await self._emit(
            "portfolio.updated",
            {
                "balance": float(self.balance),
                "equity": float(self._portfolio_equity()),
                "unrealized_pnl": float(sum(p.unrealized_pnl for p in self._positions.values())),
                "open_positions": len(self._positions),
                "closed_trades": len(self._closed_trades),
                "total_fees": float(self.total_fees),
                "total_slippage": float(self.total_slippage),
                "reason": reason,
                "timestamp_ms": timestamp_ms if timestamp_ms is not None else self._latest_timestamp_ms(),
                "mode": "backtest",
            },
        )

    def _portfolio_equity(self) -> Decimal:
        return self.balance + sum(p.unrealized_pnl for p in self._positions.values())

    async def _emit(self, topic: str, payload: dict[str, Any]) -> None:
        result = self._event_bus.publish(
            Event(
                topic=topic,
                payload=payload,
                priority=EventPriority.NORMAL,
                timestamp=(int(payload.get("timestamp_ms") or self._latest_timestamp_ms()) or 0) / 1000.0,
                source="backtesting.paper_execution",
                headers={"mode": "backtest"},
            )
        )
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            await result

    @staticmethod
    def _payload(event: Event) -> dict[str, Any]:
        return event.payload if isinstance(event.payload, dict) else {}
