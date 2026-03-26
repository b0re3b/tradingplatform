from __future__ import annotations

from enum import Enum


class LiquiditySide(str, Enum):
    BUY_SIDE = "buy_side"
    SELL_SIDE = "sell_side"
    BOTH = "both"
    UNKNOWN = "unknown"


class LiquidityLevelType(str, Enum):
    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    STOP_CLUSTER = "stop_cluster"
    RANGE_HIGH = "range_high"
    RANGE_LOW = "range_low"
    ORDERBOOK_WALL = "orderbook_wall"
    LIQUIDATION_ZONE = "liquidation_zone"


class LiquidityStatus(str, Enum):
    ACTIVE = "active"
    SWEPT = "swept"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    WEAK = "weak"


class SweepStatus(str, Enum):
    NOT_SWEPT = "not_swept"
    PARTIALLY_SWEPT = "partially_swept"
    SWEPT = "swept"
    UNKNOWN = "unknown"


class ClusterStrength(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class LiquidityBias(str, Enum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"