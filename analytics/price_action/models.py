from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, NewType

from analytics.price_action.enums import (
    FVGDirection,
    FVGEventType,
    FVGStatus,
    LevelStatus,
    LevelType,
    LiquidityEventType,
    LiquidityLevelStatus,
    LiquidityLevelType,
    MarketBias,
    SREventType,
    StructureEventType,
    StructureLayer,
    SwingType,
    TrendDirection,
    TrendEventType,
    TrendRegime,
)


SignedScore = NewType("SignedScore", float)   # expected range [-1.0, 1.0]
UnitScore = NewType("UnitScore", float)       # expected range [0.0, 1.0]

Metadata = dict[str, Any]


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------

def clamp_unit(value: float) -> float:
    """
    Clamp a numeric value to [0.0, 1.0].
    """
    return max(0.0, min(1.0, float(value)))


def clamp_signed(value: float) -> float:
    """
    Clamp a numeric value to [-1.0, 1.0].
    """
    return max(-1.0, min(1.0, float(value)))


def ensure_non_negative(value: float, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def ensure_bounds(*, upper_bound: float, lower_bound: float) -> None:
    if lower_bound > upper_bound:
        raise ValueError("lower_bound cannot be greater than upper_bound")


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    index: int = 0

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("Invalid candle: low cannot be greater than high")
        if min(self.open, self.high, self.low, self.close) < 0:
            raise ValueError("Invalid candle: OHLC cannot be negative")
        if self.high < max(self.open, self.close):
            raise ValueError("Invalid candle: high must be >= max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError("Invalid candle: low must be <= min(open, close)")
        if self.volume < 0:
            raise ValueError("Invalid candle: volume cannot be negative")
        if self.index < 0:
            raise ValueError("Invalid candle: index cannot be negative")

    @property
    def range_size(self) -> float:
        return max(self.high - self.low, 0.0)

    @property
    def body_high(self) -> float:
        return max(self.open, self.close)

    @property
    def body_low(self) -> float:
        return min(self.open, self.close)

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def body_ratio(self) -> float:
        if self.range_size <= 0:
            return 0.0
        return self.body_size / self.range_size

    @property
    def upper_wick(self) -> float:
        return max(self.high - self.body_high, 0.0)

    @property
    def lower_wick(self) -> float:
        return max(self.body_low - self.low, 0.0)

    @property
    def upper_wick_ratio(self) -> float:
        if self.range_size <= 0:
            return 0.0
        return self.upper_wick / self.range_size

    @property
    def lower_wick_ratio(self) -> float:
        if self.range_size <= 0:
            return 0.0
        return self.lower_wick / self.range_size

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        return self.close == self.open


# ---------------------------------------------------------------------------
# Market structure models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SwingPoint:
    swing_id: str
    timestamp: datetime
    price: float
    swing_type: SwingType
    layer: StructureLayer
    index: int
    candle_open: float
    candle_high: float
    candle_low: float
    candle_close: float
    strength: float
    is_confirmed: bool = True
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.swing_id:
            raise ValueError("swing_id must not be empty")
        ensure_non_negative(self.price, "price")
        if self.index < 0:
            raise ValueError("index must be >= 0")
        self.strength = clamp_unit(self.strength)


@dataclass(slots=True)
class StructureEvent:
    event_id: str
    event_type: StructureEventType
    timestamp: datetime
    price: float
    layer: StructureLayer
    direction: MarketBias | None = None
    swing_id: str | None = None
    reference_price: float | None = None
    reference_swing_id: str | None = None
    confidence: float = 0.0
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        ensure_non_negative(self.price, "price")
        if self.reference_price is not None:
            ensure_non_negative(self.reference_price, "reference_price")
        self.confidence = clamp_unit(self.confidence)


@dataclass(slots=True)
class StructureLayerState:
    layer: StructureLayer
    bias: MarketBias = MarketBias.UNKNOWN
    confidence: float = 0.0
    trend_strength: float = 0.0
    in_breakout: bool = False

    last_swing_high: SwingPoint | None = None
    previous_swing_high: SwingPoint | None = None
    last_swing_low: SwingPoint | None = None
    previous_swing_low: SwingPoint | None = None

    last_hh: StructureEvent | None = None
    last_hl: StructureEvent | None = None
    last_lh: StructureEvent | None = None
    last_ll: StructureEvent | None = None
    last_bos: StructureEvent | None = None
    last_choch: StructureEvent | None = None
    last_mss: StructureEvent | None = None

    swing_count: int = 0
    event_count: int = 0
    sequence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = clamp_unit(self.confidence)
        self.trend_strength = clamp_unit(self.trend_strength)
        if self.swing_count < 0:
            raise ValueError("swing_count must be >= 0")
        if self.event_count < 0:
            raise ValueError("event_count must be >= 0")


@dataclass(slots=True)
class MultiTimeframeAlignment:
    higher_timeframe: str | None = None
    higher_timeframe_bias: MarketBias = MarketBias.UNKNOWN
    higher_timeframe_confidence: float = 0.0

    internal_bias_aligned: bool = False
    external_bias_aligned: bool = False
    internal_with_external_aligned: bool = False

    alignment_score: float = 0.0
    last_updated: datetime | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.higher_timeframe_confidence = clamp_unit(self.higher_timeframe_confidence)
        self.alignment_score = clamp_unit(self.alignment_score)


@dataclass(slots=True)
class MarketStructureState:
    symbol: str
    timeframe: str
    last_price: float | None = None
    last_update: datetime | None = None

    internal: StructureLayerState = field(
        default_factory=lambda: StructureLayerState(layer=StructureLayer.INTERNAL)
    )
    external: StructureLayerState = field(
        default_factory=lambda: StructureLayerState(layer=StructureLayer.EXTERNAL)
    )
    mtf_alignment: MultiTimeframeAlignment = field(default_factory=MultiTimeframeAlignment)

    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.timeframe:
            raise ValueError("timeframe must not be empty")
        if self.last_price is not None:
            ensure_non_negative(self.last_price, "last_price")


# ---------------------------------------------------------------------------
# Support / Resistance models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SupportResistanceLevel:
    level_id: str
    layer: StructureLayer
    level_type: LevelType
    price: float
    upper_bound: float
    lower_bound: float
    strength: float
    status: LevelStatus = LevelStatus.ACTIVE

    created_at: datetime | None = None
    updated_at: datetime | None = None
    broken_at: datetime | None = None
    flipped_at: datetime | None = None
    last_tested_at: datetime | None = None
    last_rejected_at: datetime | None = None
    last_broken_at: datetime | None = None
    last_retested_at: datetime | None = None

    touch_count: int = 0
    rejection_count: int = 0
    break_count: int = 0
    retest_count: int = 0
    source_count: int = 0

    source_swing_ids: list[str] = field(default_factory=list)
    source_prices: list[float] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.level_id:
            raise ValueError("level_id must not be empty")
        ensure_non_negative(self.price, "price")
        ensure_bounds(upper_bound=self.upper_bound, lower_bound=self.lower_bound)
        self.strength = clamp_unit(self.strength)

        for field_name in (
            "touch_count",
            "rejection_count",
            "break_count",
            "retest_count",
            "source_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")


@dataclass(slots=True)
class SupportResistanceEvent:
    event_id: str
    event_type: SREventType
    timestamp: datetime
    symbol: str
    timeframe: str
    layer: StructureLayer
    level_id: str
    level_type: LevelType
    price: float
    confidence: float = 0.0
    reference_price: float | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.timeframe:
            raise ValueError("timeframe must not be empty")
        if not self.level_id:
            raise ValueError("level_id must not be empty")
        ensure_non_negative(self.price, "price")
        if self.reference_price is not None:
            ensure_non_negative(self.reference_price, "reference_price")
        self.confidence = clamp_unit(self.confidence)


@dataclass(slots=True)
class LayerSRState:
    layer: StructureLayer
    total_levels: int = 0
    active_supports: int = 0
    active_resistances: int = 0
    active_flip_supports: int = 0
    active_flip_resistances: int = 0

    strongest_support: SupportResistanceLevel | None = None
    strongest_resistance: SupportResistanceLevel | None = None
    nearest_support: SupportResistanceLevel | None = None
    nearest_resistance: SupportResistanceLevel | None = None

    last_event: SupportResistanceEvent | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "total_levels",
            "active_supports",
            "active_resistances",
            "active_flip_supports",
            "active_flip_resistances",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")


@dataclass(slots=True)
class SupportResistanceState:
    symbol: str
    timeframe: str
    last_price: float | None = None
    last_update: datetime | None = None
    internal: LayerSRState = field(default_factory=lambda: LayerSRState(layer=StructureLayer.INTERNAL))
    external: LayerSRState = field(default_factory=lambda: LayerSRState(layer=StructureLayer.EXTERNAL))
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.timeframe:
            raise ValueError("timeframe must not be empty")
        if self.last_price is not None:
            ensure_non_negative(self.last_price, "last_price")


# ---------------------------------------------------------------------------
# Fair Value Gap models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FairValueGap:
    gap_id: str
    layer: StructureLayer
    direction: FVGDirection

    upper_bound: float
    lower_bound: float
    mid_price: float
    size: float
    size_pct: float
    strength: float

    status: FVGStatus = FVGStatus.ACTIVE
    fill_percentage: float = 0.0
    touch_count: int = 0
    retest_count: int = 0

    created_at: datetime | None = None
    updated_at: datetime | None = None
    first_touch_at: datetime | None = None
    filled_at: datetime | None = None
    respected_at: datetime | None = None
    invalidated_at: datetime | None = None

    created_index: int | None = None
    last_touch_index: int | None = None
    last_fill_index: int | None = None

    source_candle_indices: list[int] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.gap_id:
            raise ValueError("gap_id must not be empty")
        ensure_bounds(upper_bound=self.upper_bound, lower_bound=self.lower_bound)
        ensure_non_negative(self.mid_price, "mid_price")
        ensure_non_negative(self.size, "size")
        ensure_non_negative(self.size_pct, "size_pct")
        self.strength = clamp_unit(self.strength)
        self.fill_percentage = clamp_unit(self.fill_percentage)

        if self.touch_count < 0:
            raise ValueError("touch_count must be >= 0")
        if self.retest_count < 0:
            raise ValueError("retest_count must be >= 0")


@dataclass(slots=True)
class FVGEvent:
    event_id: str
    event_type: FVGEventType
    timestamp: datetime
    symbol: str
    timeframe: str
    layer: StructureLayer
    gap_id: str
    direction: FVGDirection
    upper_bound: float
    lower_bound: float
    fill_percentage: float
    confidence: float = 0.0
    reference_price: float | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.timeframe:
            raise ValueError("timeframe must not be empty")
        if not self.gap_id:
            raise ValueError("gap_id must not be empty")
        ensure_bounds(upper_bound=self.upper_bound, lower_bound=self.lower_bound)
        self.fill_percentage = clamp_unit(self.fill_percentage)
        self.confidence = clamp_unit(self.confidence)
        if self.reference_price is not None:
            ensure_non_negative(self.reference_price, "reference_price")


@dataclass(slots=True)
class LayerFVGState:
    layer: StructureLayer
    total_gaps: int = 0
    active_gaps: int = 0
    partially_filled_gaps: int = 0
    filled_gaps: int = 0
    respected_gaps: int = 0
    invalidated_gaps: int = 0

    nearest_bullish_gap: FairValueGap | None = None
    nearest_bearish_gap: FairValueGap | None = None
    strongest_bullish_gap: FairValueGap | None = None
    strongest_bearish_gap: FairValueGap | None = None

    recent_fill_activity: float = 0.0
    last_event: FVGEvent | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "total_gaps",
            "active_gaps",
            "partially_filled_gaps",
            "filled_gaps",
            "respected_gaps",
            "invalidated_gaps",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")
        self.recent_fill_activity = clamp_unit(self.recent_fill_activity)


@dataclass(slots=True)
class FairValueGapState:
    symbol: str
    timeframe: str
    last_price: float | None = None
    last_update: datetime | None = None
    internal: LayerFVGState = field(default_factory=lambda: LayerFVGState(layer=StructureLayer.INTERNAL))
    external: LayerFVGState = field(default_factory=lambda: LayerFVGState(layer=StructureLayer.EXTERNAL))
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.timeframe:
            raise ValueError("timeframe must not be empty")
        if self.last_price is not None:
            ensure_non_negative(self.last_price, "last_price")


# ---------------------------------------------------------------------------
# Liquidity models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LiquidityLevel:
    level_id: str
    layer: StructureLayer
    level_type: LiquidityLevelType
    price: float
    upper_bound: float
    lower_bound: float
    strength: float

    status: LiquidityLevelStatus = LiquidityLevelStatus.ACTIVE
    touch_count: int = 0
    sweep_count: int = 0
    reclaim_count: int = 0

    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_touched_at: datetime | None = None
    swept_at: datetime | None = None
    reclaimed_at: datetime | None = None
    invalidated_at: datetime | None = None

    last_sweep_side: str | None = None
    last_sweep_price: float | None = None
    last_sweep_index: int | None = None

    source_swing_ids: list[str] = field(default_factory=list)
    source_prices: list[float] = field(default_factory=list)
    source_count: int = 0

    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.level_id:
            raise ValueError("level_id must not be empty")
        ensure_non_negative(self.price, "price")
        ensure_bounds(upper_bound=self.upper_bound, lower_bound=self.lower_bound)
        self.strength = clamp_unit(self.strength)

        for field_name in ("touch_count", "sweep_count", "reclaim_count", "source_count"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")

        if self.last_sweep_price is not None:
            ensure_non_negative(self.last_sweep_price, "last_sweep_price")


@dataclass(slots=True)
class LiquidityEvent:
    event_id: str
    event_type: LiquidityEventType
    timestamp: datetime
    symbol: str
    timeframe: str
    layer: StructureLayer
    level_id: str
    level_type: LiquidityLevelType
    price: float
    confidence: float = 0.0
    reference_price: float | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.timeframe:
            raise ValueError("timeframe must not be empty")
        if not self.level_id:
            raise ValueError("level_id must not be empty")
        ensure_non_negative(self.price, "price")
        if self.reference_price is not None:
            ensure_non_negative(self.reference_price, "reference_price")
        self.confidence = clamp_unit(self.confidence)


@dataclass(slots=True)
class LayerLiquidityState:
    layer: StructureLayer
    total_levels: int = 0
    active_levels: int = 0
    swept_levels: int = 0
    reclaimed_levels: int = 0
    invalidated_levels: int = 0
    
    nearest_buy_side: LiquidityLevel | None = None
    nearest_sell_side: LiquidityLevel | None = None
    strongest_buy_side: LiquidityLevel | None = None
    strongest_sell_side: LiquidityLevel | None = None

    recent_sweep_count: int = 0
    last_event: LiquidityEvent | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "total_levels",
            "active_levels",
            "swept_levels",
            "reclaimed_levels",
            "recent_sweep_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")


@dataclass(slots=True)
class LiquidityState:
    symbol: str
    timeframe: str
    last_price: float | None = None
    last_update: datetime | None = None
    internal: LayerLiquidityState = field(default_factory=lambda: LayerLiquidityState(layer=StructureLayer.INTERNAL))
    external: LayerLiquidityState = field(default_factory=lambda: LayerLiquidityState(layer=StructureLayer.EXTERNAL))
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.timeframe:
            raise ValueError("timeframe must not be empty")
        if self.last_price is not None:
            ensure_non_negative(self.last_price, "last_price")


# ---------------------------------------------------------------------------
# Trend models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrendSignal:
    signal_id: str
    timestamp: datetime
    symbol: str
    timeframe: str
    layer: StructureLayer
    event_type: TrendEventType
    direction: TrendDirection
    strength: float
    confidence: float
    regime: TrendRegime
    price: float | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.signal_id:
            raise ValueError("signal_id must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.timeframe:
            raise ValueError("timeframe must not be empty")
        self.strength = clamp_unit(self.strength)
        self.confidence = clamp_unit(self.confidence)
        if self.price is not None:
            ensure_non_negative(self.price, "price")


@dataclass(slots=True)
class TrendLayerState:
    layer: StructureLayer
    direction: TrendDirection = TrendDirection.UNKNOWN
    regime: TrendRegime = TrendRegime.UNKNOWN

    strength: UnitScore = UnitScore(0.0)
    confidence: UnitScore = UnitScore(0.0)

    momentum_direction_score: SignedScore = SignedScore(0.0)
    slope_direction_score: SignedScore = SignedScore(0.0)

    structure_score: UnitScore = UnitScore(0.0)
    continuation_probability: UnitScore = UnitScore(0.0)
    reversal_risk: UnitScore = UnitScore(0.0)
    exhaustion_score: UnitScore = UnitScore(0.0)
    pullback_depth: UnitScore = UnitScore(0.0)
    consolidation_score: UnitScore = UnitScore(0.0)

    is_accelerating: bool = False
    is_exhausted: bool = False
    in_pullback: bool = False
    is_aligned_with_structure: bool = False

    last_signal: TrendSignal | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.strength = UnitScore(clamp_unit(float(self.strength)))
        self.confidence = UnitScore(clamp_unit(float(self.confidence)))

        self.momentum_direction_score = SignedScore(clamp_signed(float(self.momentum_direction_score)))
        self.slope_direction_score = SignedScore(clamp_signed(float(self.slope_direction_score)))

        self.structure_score = UnitScore(clamp_unit(float(self.structure_score)))
        self.continuation_probability = UnitScore(clamp_unit(float(self.continuation_probability)))
        self.reversal_risk = UnitScore(clamp_unit(float(self.reversal_risk)))
        self.exhaustion_score = UnitScore(clamp_unit(float(self.exhaustion_score)))
        self.pullback_depth = UnitScore(clamp_unit(float(self.pullback_depth)))
        self.consolidation_score = UnitScore(clamp_unit(float(self.consolidation_score)))


@dataclass(slots=True)
class TrendState:
    symbol: str
    timeframe: str
    last_price: float | None = None
    last_update: datetime | None = None

    internal: TrendLayerState = field(default_factory=lambda: TrendLayerState(layer=StructureLayer.INTERNAL))
    external: TrendLayerState = field(default_factory=lambda: TrendLayerState(layer=StructureLayer.EXTERNAL))

    internal_external_alignment: UnitScore = UnitScore(0.0)
    higher_timeframe_alignment: UnitScore = UnitScore(0.0)
    overall_trend_score: UnitScore = UnitScore(0.0)

    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.timeframe:
            raise ValueError("timeframe must not be empty")
        if self.last_price is not None:
            ensure_non_negative(self.last_price, "last_price")

        self.internal_external_alignment = UnitScore(clamp_unit(float(self.internal_external_alignment)))
        self.higher_timeframe_alignment = UnitScore(clamp_unit(float(self.higher_timeframe_alignment)))
        self.overall_trend_score = UnitScore(clamp_unit(float(self.overall_trend_score)))


# ---------------------------------------------------------------------------
# Facade / aggregate models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PriceActionCompositeState:
    """
    Aggregated state for the future PriceActionAnalyzer facade.

    This model does not orchestrate anything by itself.
    It is only a typed state container for consolidated snapshots.
    """

    symbol: str
    timeframe: str
    last_price: float | None = None
    last_update: datetime | None = None

    market_structure: MarketStructureState | None = None
    support_resistance: SupportResistanceState | None = None
    fair_value_gap: FairValueGapState | None = None
    liquidity: LiquidityState | None = None
    trend: TrendState | None = None

    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.timeframe:
            raise ValueError("timeframe must not be empty")
        if self.last_price is not None:
            ensure_non_negative(self.last_price, "last_price")


__all__ = [
    "SignedScore",
    "UnitScore",
    "Metadata",
    "clamp_unit",
    "clamp_signed",
    "ensure_non_negative",
    "ensure_bounds",
    "Candle",
    "SwingPoint",
    "StructureEvent",
    "StructureLayerState",
    "MultiTimeframeAlignment",
    "MarketStructureState",
    "SupportResistanceLevel",
    "SupportResistanceEvent",
    "LayerSRState",
    "SupportResistanceState",
    "FairValueGap",
    "FVGEvent",
    "LayerFVGState",
    "FairValueGapState",
    "LiquidityLevel",
    "LiquidityEvent",
    "LayerLiquidityState",
    "LiquidityState",
    "TrendSignal",
    "TrendLayerState",
    "TrendState",
    "PriceActionCompositeState",
]