from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from .enums import (
    InstrumentType,
    OpportunityStatus,
    PricingSource,
    QuoteValidity,
    SpreadDirection,
    SpreadRegime,
    SpreadSignalType,
    SpreadType,
)


@dataclass(slots=True)
class QuoteSnapshot:
    exchange: str
    symbol: str
    instrument_type: InstrumentType

    bid: Decimal | None = None
    ask: Decimal | None = None
    bid_size: Decimal | None = None
    ask_size: Decimal | None = None

    last_price: Decimal | None = None
    mark_price: Decimal | None = None
    index_price: Decimal | None = None

    timestamp: datetime = field(default_factory=datetime.utcnow)
    received_at: datetime = field(default_factory=datetime.utcnow)

    sequence_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mid_price(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / Decimal("2")

    @property
    def spread(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def is_complete_top_of_book(self) -> bool:
        return self.bid is not None and self.ask is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "instrument_type": self.instrument_type.value,
            "bid": str(self.bid) if self.bid is not None else None,
            "ask": str(self.ask) if self.ask is not None else None,
            "bid_size": str(self.bid_size) if self.bid_size is not None else None,
            "ask_size": str(self.ask_size) if self.ask_size is not None else None,
            "last_price": str(self.last_price) if self.last_price is not None else None,
            "mark_price": str(self.mark_price) if self.mark_price is not None else None,
            "index_price": str(self.index_price) if self.index_price is not None else None,
            "timestamp": self.timestamp.isoformat(),
            "received_at": self.received_at.isoformat(),
            "sequence_id": self.sequence_id,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class FundingSnapshot:
    exchange: str
    symbol: str
    funding_rate: Decimal
    timestamp: datetime = field(default_factory=datetime.utcnow)
    next_funding_time: datetime | None = None
    predicted_rate: Decimal | None = None
    interval_hours: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "funding_rate": str(self.funding_rate),
            "timestamp": self.timestamp.isoformat(),
            "next_funding_time": self.next_funding_time.isoformat()
            if self.next_funding_time
            else None,
            "predicted_rate": str(self.predicted_rate)
            if self.predicted_rate is not None
            else None,
            "interval_hours": self.interval_hours,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RollingStats:
    count: int = 0
    mean: Decimal | None = None
    std: Decimal | None = None
    min_value: Decimal | None = None
    max_value: Decimal | None = None
    ema: Decimal | None = None
    last_value: Decimal | None = None
    zscore: Decimal | None = None
    percentile_rank: Decimal | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": str(self.mean) if self.mean is not None else None,
            "std": str(self.std) if self.std is not None else None,
            "min_value": str(self.min_value) if self.min_value is not None else None,
            "max_value": str(self.max_value) if self.max_value is not None else None,
            "ema": str(self.ema) if self.ema is not None else None,
            "last_value": str(self.last_value) if self.last_value is not None else None,
            "zscore": str(self.zscore) if self.zscore is not None else None,
            "percentile_rank": str(self.percentile_rank)
            if self.percentile_rank is not None
            else None,
        }


@dataclass(slots=True)
class SpreadSnapshot:
    spread_type: SpreadType
    symbol: str

    leg_a_exchange: str
    leg_b_exchange: str
    leg_a_type: InstrumentType
    leg_b_type: InstrumentType

    pricing_source: PricingSource = PricingSource.BID_ASK

    raw_spread: Decimal | None = None
    spread_pct: Decimal | None = None
    spread_bps: Decimal | None = None
    net_spread: Decimal | None = None
    basis: Decimal | None = None
    funding_adjusted_spread: Decimal | None = None

    direction: SpreadDirection = SpreadDirection.FLAT
    regime: SpreadRegime = SpreadRegime.NORMAL

    stats: RollingStats | None = None

    leg_a_bid: Decimal | None = None
    leg_a_ask: Decimal | None = None
    leg_b_bid: Decimal | None = None
    leg_b_ask: Decimal | None = None

    leg_a_mid: Decimal | None = None
    leg_b_mid: Decimal | None = None

    estimated_fees: Decimal | None = None
    estimated_slippage: Decimal | None = None

    quote_validity: QuoteValidity = QuoteValidity.VALID

    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_edge(self) -> bool:
        return self.net_spread is not None and self.net_spread > Decimal("0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "spread_type": self.spread_type.value,
            "symbol": self.symbol,
            "leg_a_exchange": self.leg_a_exchange,
            "leg_b_exchange": self.leg_b_exchange,
            "leg_a_type": self.leg_a_type.value,
            "leg_b_type": self.leg_b_type.value,
            "pricing_source": self.pricing_source.value,
            "raw_spread": str(self.raw_spread) if self.raw_spread is not None else None,
            "spread_pct": str(self.spread_pct) if self.spread_pct is not None else None,
            "spread_bps": str(self.spread_bps) if self.spread_bps is not None else None,
            "net_spread": str(self.net_spread) if self.net_spread is not None else None,
            "basis": str(self.basis) if self.basis is not None else None,
            "funding_adjusted_spread": str(self.funding_adjusted_spread)
            if self.funding_adjusted_spread is not None
            else None,
            "direction": self.direction.value,
            "regime": self.regime.value,
            "stats": self.stats.to_dict() if self.stats is not None else None,
            "leg_a_bid": str(self.leg_a_bid) if self.leg_a_bid is not None else None,
            "leg_a_ask": str(self.leg_a_ask) if self.leg_a_ask is not None else None,
            "leg_b_bid": str(self.leg_b_bid) if self.leg_b_bid is not None else None,
            "leg_b_ask": str(self.leg_b_ask) if self.leg_b_ask is not None else None,
            "leg_a_mid": str(self.leg_a_mid) if self.leg_a_mid is not None else None,
            "leg_b_mid": str(self.leg_b_mid) if self.leg_b_mid is not None else None,
            "estimated_fees": str(self.estimated_fees)
            if self.estimated_fees is not None
            else None,
            "estimated_slippage": str(self.estimated_slippage)
            if self.estimated_slippage is not None
            else None,
            "quote_validity": self.quote_validity.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ArbitrageOpportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_instrument_type: InstrumentType
    sell_instrument_type: InstrumentType

    buy_price: Decimal
    sell_price: Decimal
    gross_edge: Decimal

    estimated_fees: Decimal = Decimal("0")
    estimated_slippage: Decimal = Decimal("0")
    net_edge: Decimal = Decimal("0")

    spread_pct: Decimal | None = None
    spread_bps: Decimal | None = None
    confidence: Decimal | None = None

    status: OpportunityStatus = OpportunityStatus.ACTIVE
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_profitable(self) -> bool:
        return self.net_edge > Decimal("0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "buy_exchange": self.buy_exchange,
            "sell_exchange": self.sell_exchange,
            "buy_instrument_type": self.buy_instrument_type.value,
            "sell_instrument_type": self.sell_instrument_type.value,
            "buy_price": str(self.buy_price),
            "sell_price": str(self.sell_price),
            "gross_edge": str(self.gross_edge),
            "estimated_fees": str(self.estimated_fees),
            "estimated_slippage": str(self.estimated_slippage),
            "net_edge": str(self.net_edge),
            "spread_pct": str(self.spread_pct) if self.spread_pct is not None else None,
            "spread_bps": str(self.spread_bps) if self.spread_bps is not None else None,
            "confidence": str(self.confidence) if self.confidence is not None else None,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SpreadSignal:
    signal_type: SpreadSignalType
    spread_type: SpreadType
    symbol: str
    message: str

    value: Decimal | None = None
    threshold: Decimal | None = None
    confidence: Decimal | None = None

    exchange_a: str | None = None
    exchange_b: str | None = None

    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "spread_type": self.spread_type.value,
            "symbol": self.symbol,
            "message": self.message,
            "value": str(self.value) if self.value is not None else None,
            "threshold": str(self.threshold) if self.threshold is not None else None,
            "confidence": str(self.confidence) if self.confidence is not None else None,
            "exchange_a": self.exchange_a,
            "exchange_b": self.exchange_b,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }