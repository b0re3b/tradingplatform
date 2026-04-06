from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, NewType

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
        return self.high - self.body_high

    @property
    def lower_wick(self) -> float:
        return self.body_low - self.low

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
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructureEvent:
    event_id: str
    event_type: StructureEventType
    timestamp: datetime
    price: float
    layer: StructureLayer
    direction: Optional[MarketBias] = None
    swing_id: Optional[str] = None
    reference_price: Optional[float] = None
    reference_swing_id: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructureLayerState:
    layer: StructureLayer
    bias: MarketBias = MarketBias.UNKNOWN
    confidence: float = 0.0
    trend_strength: float = 0.0
    in_breakout: bool = False

    last_swing_high: Optional[SwingPoint] = None
    previous_swing_high: Optional[SwingPoint] = None
    last_swing_low: Optional[SwingPoint] = None
    previous_swing_low: Optional[SwingPoint] = None

    last_hh: Optional[StructureEvent] = None
    last_hl: Optional[StructureEvent] = None
    last_lh: Optional[StructureEvent] = None
    last_ll: Optional[StructureEvent] = None
    last_bos: Optional[StructureEvent] = None
    last_choch: Optional[StructureEvent] = None
    last_mss: Optional[StructureEvent] = None

    swing_count: int = 0
    event_count: int = 0
    sequence: List[str] = field(default_factory=list)


@dataclass(slots=True)
class MultiTimeframeAlignment:
    higher_timeframe: Optional[str] = None
    higher_timeframe_bias: MarketBias = MarketBias.UNKNOWN
    higher_timeframe_confidence: float = 0.0

    internal_bias_aligned: bool = False
    external_bias_aligned: bool = False
    internal_with_external_aligned: bool = False

    alignment_score: float = 0.0
    last_updated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MarketStructureState:
    symbol: str
    timeframe: str
    last_price: Optional[float] = None
    last_update: Optional[datetime] = None

    internal: StructureLayerState = field(
        default_factory=lambda: StructureLayerState(layer=StructureLayer.INTERNAL)
    )
    external: StructureLayerState = field(
        default_factory=lambda: StructureLayerState(layer=StructureLayer.EXTERNAL)
    )
    mtf_alignment: MultiTimeframeAlignment = field(default_factory=MultiTimeframeAlignment)

    metadata: Dict[str, Any] = field(default_factory=dict)


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

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    broken_at: Optional[datetime] = None
    flipped_at: Optional[datetime] = None
    last_tested_at: Optional[datetime] = None
    last_rejected_at: Optional[datetime] = None
    last_broken_at: Optional[datetime] = None
    last_retested_at: Optional[datetime] = None

    touch_count: int = 0
    rejection_count: int = 0
    break_count: int = 0
    retest_count: int = 0
    source_count: int = 0

    source_swing_ids: List[str] = field(default_factory=list)
    source_prices: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    reference_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LayerSRState:
    layer: StructureLayer
    total_levels: int = 0
    active_supports: int = 0
    active_resistances: int = 0
    active_flip_supports: int = 0
    active_flip_resistances: int = 0

    strongest_support: Optional[SupportResistanceLevel] = None
    strongest_resistance: Optional[SupportResistanceLevel] = None
    nearest_support: Optional[SupportResistanceLevel] = None
    nearest_resistance: Optional[SupportResistanceLevel] = None

    last_event: Optional[SupportResistanceEvent] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SupportResistanceState:
    symbol: str
    timeframe: str
    last_price: Optional[float] = None
    last_update: Optional[datetime] = None
    internal: LayerSRState = field(default_factory=lambda: LayerSRState(layer=StructureLayer.INTERNAL))
    external: LayerSRState = field(default_factory=lambda: LayerSRState(layer=StructureLayer.EXTERNAL))
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    first_touch_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    respected_at: Optional[datetime] = None
    invalidated_at: Optional[datetime] = None

    created_index: Optional[int] = None
    last_touch_index: Optional[int] = None
    last_fill_index: Optional[int] = None

    source_candle_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    reference_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LayerFVGState:
    layer: StructureLayer
    total_gaps: int = 0
    active_gaps: int = 0
    partially_filled_gaps: int = 0
    filled_gaps: int = 0
    respected_gaps: int = 0
    invalidated_gaps: int = 0

    nearest_bullish_gap: Optional[FairValueGap] = None
    nearest_bearish_gap: Optional[FairValueGap] = None
    strongest_bullish_gap: Optional[FairValueGap] = None
    strongest_bearish_gap: Optional[FairValueGap] = None

    recent_fill_activity: float = 0.0
    last_event: Optional[FVGEvent] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FairValueGapState:
    symbol: str
    timeframe: str
    last_price: Optional[float] = None
    last_update: Optional[datetime] = None
    internal: LayerFVGState = field(default_factory=lambda: LayerFVGState(layer=StructureLayer.INTERNAL))
    external: LayerFVGState = field(default_factory=lambda: LayerFVGState(layer=StructureLayer.EXTERNAL))
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_touched_at: Optional[datetime] = None
    swept_at: Optional[datetime] = None
    reclaimed_at: Optional[datetime] = None
    invalidated_at: Optional[datetime] = None

    last_sweep_side: Optional[str] = None
    last_sweep_price: Optional[float] = None
    last_sweep_index: Optional[int] = None

    source_swing_ids: List[str] = field(default_factory=list)
    source_prices: List[float] = field(default_factory=list)
    source_count: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)


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
    reference_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LayerLiquidityState:
    layer: StructureLayer
    total_levels: int = 0
    active_levels: int = 0
    swept_levels: int = 0
    reclaimed_levels: int = 0

    nearest_buy_side: Optional[LiquidityLevel] = None
    nearest_sell_side: Optional[LiquidityLevel] = None
    strongest_buy_side: Optional[LiquidityLevel] = None
    strongest_sell_side: Optional[LiquidityLevel] = None

    recent_sweep_count: int = 0
    last_event: Optional[LiquidityEvent] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LiquidityState:
    symbol: str
    timeframe: str
    last_price: Optional[float] = None
    last_update: Optional[datetime] = None
    internal: LayerLiquidityState = field(default_factory=lambda: LayerLiquidityState(layer=StructureLayer.INTERNAL))
    external: LayerLiquidityState = field(default_factory=lambda: LayerLiquidityState(layer=StructureLayer.EXTERNAL))
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    last_signal: Optional[TrendSignal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrendState:
    symbol: str
    timeframe: str
    last_price: Optional[float] = None
    last_update: Optional[datetime] = None

    internal: TrendLayerState = field(default_factory=lambda: TrendLayerState(layer=StructureLayer.INTERNAL))
    external: TrendLayerState = field(default_factory=lambda: TrendLayerState(layer=StructureLayer.EXTERNAL))

    internal_external_alignment: UnitScore = UnitScore(0.0)
    higher_timeframe_alignment: UnitScore = UnitScore(0.0)
    overall_trend_score: UnitScore = UnitScore(0.0)

    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "SignedScore",
    "UnitScore",
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
]