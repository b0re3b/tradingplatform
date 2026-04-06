from __future__ import annotations

from enum import Enum


class WhaleTradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class LargeTradeTriggerType(str, Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    ABSOLUTE_AND_RELATIVE = "absolute_and_relative"
    UNKNOWN = "unknown"


class WhaleEventType(str, Enum):
    LARGE_TRADE = "large_trade"
    WHALE_ACTIVITY = "whale_activity"
    WHALE_PRESSURE = "whale_pressure"
    WHALE_LIQUIDATION_CONTEXT = "whale_liquidation_context"
    WHALE_CLUSTER = "whale_cluster"
    WHALE_CLUSTER_UPDATE = "whale_cluster_update"
    WHALE_CLUSTER_EXHAUSTION = "whale_cluster_exhaustion"


class WhaleComponentName(str, Enum):
    LARGE_TRADE_DETECTOR = "large_trade_detector"
    WHALE_TRACKER = "whale_tracker"
    WHALE_CLUSTER_ANALYZER = "whale_cluster_analyzer"
    WHALE_ANALYZER = "analyzer"


class WhaleBias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class WhaleClusterStateType(str, Enum):
    FORMING = "forming"
    ACTIVE = "active"
    EXHAUSTING = "exhausting"
    INACTIVE = "inactive"


class WhalePressureType(str, Enum):
    BUY_PRESSURE = "buy_pressure"
    SELL_PRESSURE = "sell_pressure"
    BALANCED = "balanced"
    UNKNOWN = "unknown"


class WhaleNormalizationStatus(str, Enum):
    NORMALIZED = "normalized"
    INVALID = "invalid"
    FILTERED = "filtered"


class WhaleDataSource(str, Enum):
    MARKET_TRADE = "market.trade"
    MARKET_LIQUIDATION = "market.liquidation"
    ANALYTICS_WHALES_LARGE_TRADE = "analytics.whales.large_trade"
    ANALYTICS_WHALES_ACTIVITY = "analytics.whales.whale_activity"
    ANALYTICS_WHALES_PRESSURE = "analytics.whales.whale_pressure"
    ANALYTICS_WHALES_LIQUIDATION_CONTEXT = "analytics.whales.whale_liquidation_context"
    ANALYTICS_WHALES_CLUSTER = "analytics.whales.whale_cluster"
    ANALYTICS_WHALES_CLUSTER_UPDATE = "analytics.whales.whale_cluster_update"
    ANALYTICS_WHALES_CLUSTER_EXHAUSTION = "analytics.whales.whale_cluster_exhaustion"