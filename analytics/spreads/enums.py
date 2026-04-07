from __future__ import annotations

from enum import Enum


class SpreadType(str, Enum):
    SPOT_FUTURES = "spot_futures"
    CROSS_EXCHANGE = "cross_exchange"


class InstrumentType(str, Enum):
    SPOT = "spot"
    PERPETUAL = "perpetual"
    FUTURES = "futures"
    UNKNOWN = "unknown"


class SpreadSignalType(str, Enum):
    WIDENING = "widening"
    NARROWING = "narrowing"
    ANOMALY = "anomaly"
    ARBITRAGE = "arbitrage"
    MEAN_REVERSION = "mean_reversion"
    REGIME_SHIFT = "regime_shift"
    STALE_DATA = "stale_data"
    INVALID_DATA = "invalid_data"


class SpreadDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    FLAT = "flat"


class SpreadRegime(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"
    COMPRESSED = "compressed"
    DISLOCATED = "dislocated"


class OpportunityStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REJECTED = "rejected"
    EXECUTED = "executed"


class QuoteValidity(str, Enum):
    VALID = "valid"
    STALE = "stale"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


class PricingSource(str, Enum):
    BID_ASK = "bid_ask"
    MID = "mid"
    LAST = "last"
    MARK = "mark"
    INDEX = "index"