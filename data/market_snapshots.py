from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data.market_models import MarketScope


@dataclass(frozen=True, slots=True)
class TradeSnapshot:
    price: float
    quantity: float
    side: str
    aggressor_side: str
    timestamp_ms: int
    trade_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side,
            "aggressor_side": self.aggressor_side,
            "timestamp_ms": self.timestamp_ms,
            "trade_id": self.trade_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TradesWindowSnapshot:
    trades: tuple[TradeSnapshot, ...] = ()
    last_price: float | None = None
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    total_volume: float = 0.0
    trade_count: int = 0
    first_timestamp_ms: int | None = None
    last_timestamp_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_price": self.last_price,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "total_volume": self.total_volume,
            "trade_count": self.trade_count,
            "first_timestamp_ms": self.first_timestamp_ms,
            "last_timestamp_ms": self.last_timestamp_ms,
            "trades": [trade.to_dict() for trade in self.trades],
        }


@dataclass(frozen=True, slots=True)
class CandleSnapshot:
    timeframe: str
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool
    timestamp_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "open_time_ms": self.open_time_ms,
            "close_time_ms": self.close_time_ms,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "is_closed": self.is_closed,
            "timestamp_ms": self.timestamp_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CandlesWindowSnapshot:
    timeframe: str
    candles: tuple[CandleSnapshot, ...] = ()
    last_close: float | None = None
    last_closed: CandleSnapshot | None = None
    candle_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "last_close": self.last_close,
            "last_closed": self.last_closed.to_dict() if self.last_closed else None,
            "candle_count": self.candle_count,
            "candles": [candle.to_dict() for candle in self.candles],
        }


@dataclass(frozen=True, slots=True)
class OrderBookLevelSnapshot:
    price: float
    quantity: float

    def to_dict(self) -> dict[str, Any]:
        return {"price": self.price, "quantity": self.quantity}


@dataclass(frozen=True, slots=True)
class OrderBookSnapshotView:
    bids: tuple[OrderBookLevelSnapshot, ...] = ()
    asks: tuple[OrderBookLevelSnapshot, ...] = ()
    best_bid: float | None = None
    best_ask: float | None = None
    mid_price: float | None = None
    spread: float | None = None
    sequence: int | None = None
    snapshot_received: bool = False
    resync_required: bool = False
    last_update_ms: int | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "sequence": self.sequence,
            "snapshot_received": self.snapshot_received,
            "resync_required": self.resync_required,
            "last_update_ms": self.last_update_ms,
            "last_error": self.last_error,
            "bids": [level.to_dict() for level in self.bids],
            "asks": [level.to_dict() for level in self.asks],
        }


@dataclass(frozen=True, slots=True)
class FundingSnapshot:
    funding_rate: float | None = None
    next_funding_time_ms: int | None = None
    mark_price: float | None = None
    index_price: float | None = None
    timestamp_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "funding_rate": self.funding_rate,
            "next_funding_time_ms": self.next_funding_time_ms,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "timestamp_ms": self.timestamp_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class OpenInterestSnapshot:
    open_interest: float | None = None
    open_interest_value: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    timestamp_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_interest": self.open_interest,
            "open_interest_value": self.open_interest_value,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "timestamp_ms": self.timestamp_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class LiquidationSnapshot:
    price: float
    quantity: float
    side: str
    timestamp_ms: int
    order_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side,
            "timestamp_ms": self.timestamp_ms,
            "order_id": self.order_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    scope: MarketScope
    last_price: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    reference_price: float | None = None
    price_source: str | None = None
    trades: TradesWindowSnapshot = field(default_factory=TradesWindowSnapshot)
    candles: dict[str, CandlesWindowSnapshot] = field(default_factory=dict)
    orderbook: OrderBookSnapshotView | None = None
    funding: FundingSnapshot | None = None
    open_interest: OpenInterestSnapshot | None = None
    liquidations: tuple[LiquidationSnapshot, ...] = ()
    dirty_reasons: tuple[str, ...] = ()
    updated_at_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def current_price(self) -> float | None:
        return self.reference_price or self.last_price or self.mark_price or self.index_price

    def to_strategy_context_dict(self) -> dict[str, Any]:
        price = self.current_price
        return {
            **self.scope.to_dict(),
            "current_price": price,
            "last_price": self.last_price or price,
            "price": price,
            "reference_price": price,
            "entry_reference_price": price,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "price_source": self.price_source,
            "updated_at_ms": self.updated_at_ms,
            "dirty_reasons": list(self.dirty_reasons),
            "trades": self.trades.to_dict(),
            "candles": {timeframe: candles.to_dict() for timeframe, candles in self.candles.items()},
            "orderbook": self.orderbook.to_dict() if self.orderbook else None,
            "funding": self.funding.to_dict() if self.funding else None,
            "open_interest": self.open_interest.to_dict() if self.open_interest else None,
            "liquidations": [item.to_dict() for item in self.liquidations],
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_strategy_context_dict()
