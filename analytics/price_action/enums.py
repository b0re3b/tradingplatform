from __future__ import annotations

from enum import Enum
from typing import TypeVar


EnumT = TypeVar("EnumT", bound="PriceActionEnum")


class PriceActionEnum(str, Enum):
    """
    Base string enum for all price action domain enums.

    This class is intentionally infrastructure-free:
    - no EventBus dependency
    - no Scheduler dependency
    - no logger dependency
    - no Config dependency

    Enums are pure domain contracts used by:
    - analytics.price_action.models
    - analytics.price_action analyzers
    - strategy modules consuming analytics.price_action.* events
    """

    def __str__(self) -> str:
        return self.value

    @classmethod
    def values(cls) -> tuple[str, ...]:
        """
        Return all raw string values for validation, schemas and UI filters.
        """
        return tuple(member.value for member in cls)

    @classmethod
    def has_value(cls, value: object) -> bool:
        """
        Check whether a raw value belongs to the enum.
        """
        return any(member.value == value for member in cls)

    @classmethod
    def from_value(cls: type[EnumT], value: str | EnumT, *, default: EnumT | None = None) -> EnumT:
        """
        Safely coerce a raw value into the enum.

        Args:
            value: Raw string value or already-existing enum member.
            default: Optional fallback when coercion fails.

        Raises:
            ValueError: If value is invalid and default is not provided.
        """
        if isinstance(value, cls):
            return value

        try:
            return cls(str(value))
        except ValueError:
            if default is not None:
                return default
            raise


# ---------------------------------------------------------------------------
# Shared analyzer/module enums
# ---------------------------------------------------------------------------

class PriceActionModuleType(PriceActionEnum):
    """
    Logical module identifiers used by the future PriceActionAnalyzer facade.
    """

    MARKET_STRUCTURE = "market_structure"
    SUPPORT_RESISTANCE = "support_resistance"
    LIQUIDITY_LEVELS = "liquidity_levels"
    FAIR_VALUE_GAP = "fair_value_gap"
    TREND = "trend"


class PriceActionSnapshotType(PriceActionEnum):
    """
    Common snapshot/update event types shared by price action modules.

    Concrete EventBus topics should still be namespaced as:
        analytics.price_action.<module>.<event>
    """

    SNAPSHOT = "snapshot"
    UPDATED = "updated"
    RESET = "reset"


# ---------------------------------------------------------------------------
# Shared structure enums
# ---------------------------------------------------------------------------

class SwingType(PriceActionEnum):
    HIGH = "high"
    LOW = "low"


class StructureLayer(PriceActionEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"


class MarketBias(PriceActionEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    UNKNOWN = "unknown"


class StructureEventType(PriceActionEnum):
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    HH = "hh"
    HL = "hl"
    LH = "lh"
    LL = "ll"
    BOS = "bos"
    CHOCH = "choch"
    MSS = "mss"


# ---------------------------------------------------------------------------
# Support / Resistance enums
# ---------------------------------------------------------------------------

class LevelType(PriceActionEnum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    FLIP_SUPPORT = "flip_support"
    FLIP_RESISTANCE = "flip_resistance"


class LevelStatus(PriceActionEnum):
    ACTIVE = "active"
    BROKEN = "broken"
    INACTIVE = "inactive"


class SREventType(PriceActionEnum):
    LEVEL_CREATED = "level_created"
    LEVEL_MERGED = "level_merged"
    LEVEL_TOUCHED = "level_touched"
    LEVEL_REJECTED = "level_rejected"
    LEVEL_BROKEN = "level_broken"
    LEVEL_FLIPPED = "level_flipped"
    LEVEL_RETESTED = "level_retested"


# ---------------------------------------------------------------------------
# Fair Value Gap enums
# ---------------------------------------------------------------------------

class FVGDirection(PriceActionEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class FVGStatus(PriceActionEnum):
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    RESPECTED = "respected"
    INVALIDATED = "invalidated"


class FVGEventType(PriceActionEnum):
    FVG_CREATED = "fvg_created"
    FVG_FILL_STARTED = "fvg_fill_started"
    FVG_PARTIALLY_FILLED = "fvg_partially_filled"
    FVG_FILLED = "fvg_filled"
    FVG_RESPECTED = "fvg_respected"
    FVG_INVALIDATED = "fvg_invalidated"
    FVG_RETESTED = "fvg_retested"
    FVG_MERGED = "fvg_merged"


# ---------------------------------------------------------------------------
# Liquidity enums
# ---------------------------------------------------------------------------

class LiquidityLevelType(PriceActionEnum):
    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"
    BUY_SIDE_LIQUIDITY = "buy_side_liquidity"
    SELL_SIDE_LIQUIDITY = "sell_side_liquidity"
    SWING_HIGH_LIQUIDITY = "swing_high_liquidity"
    SWING_LOW_LIQUIDITY = "swing_low_liquidity"


class LiquidityLevelStatus(PriceActionEnum):
    ACTIVE = "active"
    SWEPT = "swept"
    RECLAIMED = "reclaimed"
    INVALIDATED = "invalidated"


class LiquidityEventType(PriceActionEnum):
    LEVEL_CREATED = "level_created"
    LEVEL_MERGED = "level_merged"
    LIQUIDITY_TOUCHED = "liquidity_touched"
    LIQUIDITY_SWEPT = "liquidity_swept"
    LIQUIDITY_RECLAIMED = "liquidity_reclaimed"
    FAILED_BREAKOUT = "failed_breakout"
    STOP_RUN = "stop_run"
    LIQUIDITY_INVALIDATED = "liquidity_invalidated"


# ---------------------------------------------------------------------------
# Trend enums
# ---------------------------------------------------------------------------

class TrendDirection(PriceActionEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class TrendRegime(PriceActionEnum):
    TRENDING = "trending"
    PULLBACK = "pullback"
    CONSOLIDATING = "consolidating"
    REVERSING = "reversing"
    EXHAUSTED = "exhausted"
    UNKNOWN = "unknown"


class TrendEventType(PriceActionEnum):
    TREND_STARTED = "trend_started"
    TREND_CONTINUATION = "trend_continuation"
    TREND_ACCELERATION = "trend_acceleration"
    TREND_WEAKENING = "trend_weakening"
    PULLBACK_STARTED = "pullback_started"
    PULLBACK_ENDED = "pullback_ended"
    TREND_REVERSAL = "trend_reversal"
    TREND_EXHAUSTION = "trend_exhaustion"
    TREND_ALIGNMENT = "trend_alignment"
    TREND_DISAGREEMENT = "trend_disagreement"


__all__ = [
    "PriceActionEnum",
    "PriceActionModuleType",
    "PriceActionSnapshotType",
    "SwingType",
    "StructureLayer",
    "MarketBias",
    "StructureEventType",
    "LevelType",
    "LevelStatus",
    "SREventType",
    "FVGDirection",
    "FVGStatus",
    "FVGEventType",
    "LiquidityLevelType",
    "LiquidityLevelStatus",
    "LiquidityEventType",
    "TrendDirection",
    "TrendRegime",
    "TrendEventType",
]