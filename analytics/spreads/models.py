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


# =========================
# Market Data Models
# =========================

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


# =========================
# Analytics Models
# =========================

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


# =========================
# Signals
# =========================

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


# =========================
# Arbitrage
# =========================

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