from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(slots=True)
class HistoricalCandle:
    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    open_time_ms: int
    close_time_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal
    trades_count: int
    is_closed: bool = True

    def to_market_payload(self, *, replay_id: str, backtest_id: str) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "exchange_symbol": self.symbol,
            "timeframe": self.timeframe,
            "open_time_ms": self.open_time_ms,
            "close_time_ms": self.close_time_ms,
            "timestamp_ms": self.close_time_ms,
            "received_at_ms": self.close_time_ms,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
            "quote_volume": float(self.quote_volume),
            "trades_count": self.trades_count,
            "is_closed": self.is_closed,
            "mode": "backtest",
            "replay_id": replay_id,
            "backtest_id": backtest_id,
            "replay_time": True,
        }


@dataclass(slots=True)
class HistoricalTrade:
    exchange: str
    market_type: str
    symbol: str
    trade_id: str
    price: Decimal
    quantity: Decimal
    timestamp_ms: int
    is_buyer_maker: bool

    def to_market_payload(self, *, replay_id: str, backtest_id: str) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "exchange_symbol": self.symbol,
            "trade_id": self.trade_id,
            "price": float(self.price),
            "quantity": float(self.quantity),
            "qty": float(self.quantity),
            "timestamp_ms": self.timestamp_ms,
            "received_at_ms": self.timestamp_ms,
            "is_buyer_maker": self.is_buyer_maker,
            "side": "sell" if self.is_buyer_maker else "buy",
            "mode": "backtest",
            "replay_id": replay_id,
            "backtest_id": backtest_id,
            "replay_time": True,
        }


@dataclass(slots=True)
class HistoricalFundingRate:
    exchange: str
    market_type: str
    symbol: str
    funding_time_ms: int
    funding_rate: Decimal
    mark_price: Decimal | None = None

    def to_market_payload(self, *, replay_id: str, backtest_id: str) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "exchange_symbol": self.symbol,
            "funding_time_ms": self.funding_time_ms,
            "timestamp_ms": self.funding_time_ms,
            "received_at_ms": self.funding_time_ms,
            "funding_rate": float(self.funding_rate),
            "mark_price": float(self.mark_price) if self.mark_price is not None else None,
            "mode": "backtest",
            "replay_id": replay_id,
            "backtest_id": backtest_id,
            "replay_time": True,
        }


@dataclass(slots=True)
class HistoricalOpenInterest:
    exchange: str
    market_type: str
    symbol: str
    timestamp_ms: int
    sum_open_interest: Decimal
    sum_open_interest_value: Decimal | None = None

    def to_market_payload(self, *, replay_id: str, backtest_id: str) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "exchange_symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "received_at_ms": self.timestamp_ms,
            "open_interest": float(self.sum_open_interest),
            "sum_open_interest": float(self.sum_open_interest),
            "sum_open_interest_value": (
                float(self.sum_open_interest_value)
                if self.sum_open_interest_value is not None
                else None
            ),
            "mode": "backtest",
            "replay_id": replay_id,
            "backtest_id": backtest_id,
            "replay_time": True,
        }


@dataclass(slots=True)
class HistoricalMarkPrice:
    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    open_time_ms: int
    close_time_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    def to_market_payload(self, *, replay_id: str, backtest_id: str) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "exchange_symbol": self.symbol,
            "timeframe": self.timeframe,
            "open_time_ms": self.open_time_ms,
            "close_time_ms": self.close_time_ms,
            "timestamp_ms": self.close_time_ms,
            "received_at_ms": self.close_time_ms,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "mark_price": float(self.close),
            "mode": "backtest",
            "replay_id": replay_id,
            "backtest_id": backtest_id,
            "replay_time": True,
        }


@dataclass(slots=True)
class ReplayEvent:
    topic: str
    timestamp_ms: int
    payload: dict[str, Any]
    sequence: int = 0


@dataclass(slots=True)
class PaperOrder:
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    position_side: str
    quantity: Decimal
    status: str
    submitted_at_ms: int
    filled_at_ms: int | None = None
    avg_price: Decimal | None = None
    fee: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    signal_id: str | None = None
    strategy_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PaperPosition:
    position_id: str
    symbol: str
    side: str
    size: Decimal
    entry_price: Decimal
    leverage: Decimal
    margin_used: Decimal
    opened_at_ms: int
    signal_id: str | None = None
    strategy_name: str | None = None
    timeframe: str | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    last_mark_price: Decimal | None = None
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    entry_fee: Decimal = Decimal("0")
    exit_fee: Decimal = Decimal("0")
    entry_slippage: Decimal = Decimal("0")
    exit_slippage: Decimal = Decimal("0")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClosedTrade:
    trade_id: str
    position_id: str
    symbol: str
    side: str
    size: Decimal
    entry_price: Decimal
    exit_price: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    fees: Decimal
    slippage: Decimal
    opened_at_ms: int
    closed_at_ms: int
    holding_ms: int
    signal_id: str | None = None
    strategy_name: str | None = None
    timeframe: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side,
            "size": float(self.size),
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price),
            "gross_pnl": float(self.gross_pnl),
            "realized_pnl": float(self.net_pnl),
            "net_pnl": float(self.net_pnl),
            "fees": float(self.fees),
            "slippage": float(self.slippage),
            "opened_at_ms": self.opened_at_ms,
            "closed_at_ms": self.closed_at_ms,
            "timestamp_ms": self.closed_at_ms,
            "holding_ms": self.holding_ms,
            "signal_id": self.signal_id,
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }
