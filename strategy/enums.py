from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Base enum that behaves nicely as a string."""

    def __str__(self) -> str:
        return self.value


class SignalSide(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    UNKNOWN = "unknown"


class SignalStatus(StrEnum):
    NEW = "new"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    EXECUTED = "executed"
    FAILED = "failed"


class SetupType(StrEnum):
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    SQUEEZE = "squeeze"
    MOMENTUM = "momentum"
    ABSORPTION = "absorption"
    EXHAUSTION = "exhaustion"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class EntryType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    PASSIVE = "passive"
    PULLBACK = "pullback"
    BREAKOUT_CONFIRMATION = "breakout_confirmation"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"


class ExitType(StrEnum):
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    SIGNAL_FLIP = "signal_flip"
    INVALIDATION = "invalidation"
    MANUAL = "manual"
    PARTIAL = "partial"


class StrategyCategory(StrEnum):
    ORDERFLOW = "orderflow"
    LIQUIDITY = "liquidity"
    PRICE_ACTION = "price_action"
    LIQUIDATIONS = "liquidations"
    WHALES = "whales"
    SPOOFING = "spoofing"
    SPREADS = "spreads"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    HYBRID = "hybrid"


class MarketRegime(StrEnum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    SQUEEZE = "squeeze"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    NEWS_DRIVEN = "news_driven"
    ILLIQUID = "illiquid"
    RISK_OFF = "risk_off"
    UNKNOWN = "unknown"


class ConflictType(StrEnum):
    SIDE_CONFLICT = "side_conflict"
    REGIME_CONFLICT = "regime_conflict"
    FLOW_CONFLICT = "flow_conflict"
    EXECUTION_CONFLICT = "execution_conflict"
    FILTER_CONFLICT = "filter_conflict"
    TIMEFRAME_CONFLICT = "timeframe_conflict"
    PORTFOLIO_CONFLICT = "portfolio_conflict"
    RISK_CONFLICT = "risk_conflict"


class ConfidenceGrade(StrEnum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class Timeframe(StrEnum):
    TICK = "tick"
    S1 = "1s"
    S5 = "5s"
    S15 = "15s"
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class FilterDecision(StrEnum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    SKIP = "skip"


class SignalStrength(StrEnum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"


class TriggerType(StrEnum):
    PRIMARY = "primary"
    CONFIRMATION = "confirmation"
    CONFLUENCE = "confluence"
    DERIVED = "derived"
    FILTERED = "filtered"


class FeatureSource(StrEnum):
    ORDERFLOW = "orderflow"
    LIQUIDITY = "liquidity"
    PRICE_ACTION = "price_action"
    LIQUIDATIONS = "liquidations"
    WHALES = "whales"
    SPOOFING = "spoofing"
    SPREADS = "spreads"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    REGIME = "regime"
    PORTFOLIO = "portfolio"
    SYSTEM = "system"
    EXTERNAL = "external"


class PresetMode(StrEnum):
    SCALPING = "scalping"
    INTRADAY = "intraday"
    SWING = "swing"


class SignalPriority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SignalOrigin(StrEnum):
    SINGLE_STRATEGY = "single_strategy"
    MULTI_STRATEGY = "multi_strategy"
    CONFLUENCE = "confluence"
    HYBRID = "hybrid"


class FreshnessStatus(StrEnum):
    FRESH = "fresh"
    AGING = "aging"
    STALE = "stale"
    EXPIRED = "expired"