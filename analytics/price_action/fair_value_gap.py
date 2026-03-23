from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from core.logger import get_logger

from analytics.price_action.market_structure import StructureLayer


class FVGDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class FVGStatus(str, Enum):
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    RESPECTED = "respected"
    INVALIDATED = "invalidated"


class FVGEventType(str, Enum):
    FVG_CREATED = "fvg_created"
    FVG_FILL_STARTED = "fvg_fill_started"
    FVG_PARTIALLY_FILLED = "fvg_partially_filled"
    FVG_FILLED = "fvg_filled"
    FVG_RESPECTED = "fvg_respected"
    FVG_INVALIDATED = "fvg_invalidated"
    FVG_RETESTED = "fvg_retested"
    FVG_MERGED = "fvg_merged"


@dataclass(slots=True)
class FairValueGapConfig:
    max_candles: int = 3000
    max_gaps_per_layer: int = 500
    max_events: int = 1000

    min_gap_pct_internal: float = 0.00035
    min_gap_pct_external: float = 0.00080
    merge_distance_pct_internal: float = 0.00025
    merge_distance_pct_external: float = 0.00050

    min_impulse_body_ratio: float = 0.45
    respected_reaction_threshold_pct: float = 0.0012
    invalidation_close_buffer_pct: float = 0.0002
    retest_window_bars: int = 20

    emit_events: bool = True
    event_namespace: str = "price_action.fair_value_gap"
    publish_snapshots: bool = False

    def validate(self) -> None:
        if self.max_candles < 100:
            raise ValueError("max_candles must be >= 100")
        if self.max_gaps_per_layer < 20:
            raise ValueError("max_gaps_per_layer must be >= 20")
        if self.max_events < 50:
            raise ValueError("max_events must be >= 50")
        if self.min_gap_pct_internal < 0:
            raise ValueError("min_gap_pct_internal must be >= 0")
        if self.min_gap_pct_external < 0:
            raise ValueError("min_gap_pct_external must be >= 0")
        if self.merge_distance_pct_internal < 0:
            raise ValueError("merge_distance_pct_internal must be >= 0")
        if self.merge_distance_pct_external < 0:
            raise ValueError("merge_distance_pct_external must be >= 0")
        if self.min_impulse_body_ratio < 0:
            raise ValueError("min_impulse_body_ratio must be >= 0")
        if self.respected_reaction_threshold_pct < 0:
            raise ValueError("respected_reaction_threshold_pct must be >= 0")
        if self.invalidation_close_buffer_pct < 0:
            raise ValueError("invalidation_close_buffer_pct must be >= 0")
        if self.retest_window_bars < 1:
            raise ValueError("retest_window_bars must be >= 1")


@dataclass(slots=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    index: int = 0

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
    def range_size(self) -> float:
        return max(self.high - self.low, 0.0)

    @property
    def body_ratio(self) -> float:
        if self.range_size <= 0:
            return 0.0
        return self.body_size / self.range_size

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


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


class FairValueGapAnalyzer:
    """
    Stateful Fair Value Gap analyzer.

    Features
    --------
    - detects bullish / bearish FVG using 3-candle logic
    - supports internal / external layers
    - tracks partial fills, full fills, respected reactions, invalidations
    - supports merge of nearby same-direction gaps
    - integrates with EventBus
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        event_bus: Optional[Any] = None,
        config: Optional[FairValueGapConfig] = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.event_bus = event_bus
        self.config = config or FairValueGapConfig()
        self.config.validate()

        self.logger = get_logger(__name__, service_name="price_action.fair_value_gap")

        self._candles: Deque[Candle] = deque(maxlen=self.config.max_candles)
        self._internal_gaps: Deque[FairValueGap] = deque(maxlen=self.config.max_gaps_per_layer)
        self._external_gaps: Deque[FairValueGap] = deque(maxlen=self.config.max_gaps_per_layer)
        self._events: Deque[FVGEvent] = deque(maxlen=self.config.max_events)

        self._global_candle_index = 0
        self._last_processed_triplet_end_index = -1

        self._processed_fill_keys: set[Tuple[str, int]] = set()
        self._processed_respect_keys: set[Tuple[str, int]] = set()
        self._processed_invalidation_keys: set[Tuple[str, int]] = set()
        self._processed_retest_keys: set[Tuple[str, int]] = set()

        self._state = FairValueGapState(
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        self.logger.info(
            "Initialized FairValueGapAnalyzer",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "config": asdict(self.config),
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._candles.clear()
        self._internal_gaps.clear()
        self._external_gaps.clear()
        self._events.clear()

        self._global_candle_index = 0
        self._last_processed_triplet_end_index = -1

        self._processed_fill_keys.clear()
        self._processed_respect_keys.clear()
        self._processed_invalidation_keys.clear()
        self._processed_retest_keys.clear()

        self._state = FairValueGapState(
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        self.logger.info(
            "FairValueGapAnalyzer reset",
            extra={"symbol": self.symbol, "timeframe": self.timeframe},
        )

    def update(
        self,
        *,
        candles: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return self.add_candles(candles or [])

    def add_candle(self, candle: Mapping[str, Any]) -> Dict[str, Any]:
        return self.add_candles([candle])

    def add_candles(self, candles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not candles:
            self._refresh_state()
            return {
                "state": self.snapshot(),
                "updated_gaps": [],
                "new_events": [],
            }

        updated_gaps: List[FairValueGap] = []
        new_events: List[FVGEvent] = []

        for raw in candles:
            candle = self._parse_candle(raw)
            self._candles.append(candle)
            self._state.last_price = candle.close
            self._state.last_update = candle.timestamp

            created_gaps, creation_events = self._process_incremental_gap_detection()
            if created_gaps:
                updated_gaps.extend(created_gaps)
            if creation_events:
                new_events.extend(creation_events)

            lifecycle_events = self._process_gap_lifecycle(candle)
            if lifecycle_events:
                new_events.extend(lifecycle_events)

        self._refresh_state()

        if self.config.publish_snapshots:
            self._publish_snapshot()

        self.logger.debug(
            "Fair value gap analyzer updated",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "updated_gaps": len(updated_gaps),
                "new_events": len(new_events),
                "last_price": self._state.last_price,
            },
        )

        return {
            "state": self.snapshot(),
            "updated_gaps": [self._gap_to_dict(gap) for gap in updated_gaps],
            "new_events": [self._event_to_dict(event) for event in new_events],
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "symbol": self._state.symbol,
            "timeframe": self._state.timeframe,
            "last_price": self._state.last_price,
            "last_update": self._state.last_update.isoformat() if self._state.last_update else None,
            "internal": self._layer_state_to_dict(self._state.internal),
            "external": self._layer_state_to_dict(self._state.external),
            "gaps": {
                "internal": [self._gap_to_dict(x) for x in self._internal_gaps],
                "external": [self._gap_to_dict(x) for x in self._external_gaps],
            },
            "metadata": dict(self._state.metadata),
        }

    def get_state(self) -> FairValueGapState:
        return self._state

    def get_internal_gaps(self) -> List[FairValueGap]:
        return list(self._internal_gaps)

    def get_external_gaps(self) -> List[FairValueGap]:
        return list(self._external_gaps)

    def get_events(self) -> List[FVGEvent]:
        return list(self._events)

    # ------------------------------------------------------------------
    # Incremental detection
    # ------------------------------------------------------------------

    def _process_incremental_gap_detection(self) -> Tuple[List[FairValueGap], List[FVGEvent]]:
        if len(self._candles) < 3:
            return [], []

        last_index = len(self._candles) - 1
        if last_index <= self._last_processed_triplet_end_index:
            return [], []

        self._last_processed_triplet_end_index = last_index

        c1 = self._candles[-3]
        c2 = self._candles[-2]
        c3 = self._candles[-1]

        created_gaps: List[FairValueGap] = []
        events: List[FVGEvent] = []

        for layer in (StructureLayer.INTERNAL, StructureLayer.EXTERNAL):
            candidate = self._detect_fvg_from_triplet(c1, c2, c3, layer)
            if candidate is None:
                continue

            merged_or_created, event = self._register_gap(candidate)
            if merged_or_created is not None:
                created_gaps.append(merged_or_created)
            if event is not None:
                events.append(event)

        return created_gaps, events

    def _detect_fvg_from_triplet(
        self,
        c1: Candle,
        c2: Candle,
        c3: Candle,
        layer: StructureLayer,
    ) -> Optional[FairValueGap]:
        min_gap_pct = self._min_gap_pct(layer)

        if c2.body_ratio < self.config.min_impulse_body_ratio:
            return None

        # Bullish FVG: low of candle3 > high of candle1
        if c3.low > c1.high:
            gap_size = c3.low - c1.high
            reference = max(abs(c2.close), 1e-12)
            gap_pct = gap_size / reference

            if gap_pct >= min_gap_pct:
                lower_bound = c1.high
                upper_bound = c3.low
                return self._build_gap(
                    layer=layer,
                    direction=FVGDirection.BULLISH,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    size=gap_size,
                    size_pct=gap_pct,
                    created_at=c3.timestamp,
                    created_index=c3.index,
                    source_indices=[c1.index, c2.index, c3.index],
                    impulse_ratio=c2.body_ratio,
                )

        # Bearish FVG: high of candle3 < low of candle1
        if c3.high < c1.low:
            gap_size = c1.low - c3.high
            reference = max(abs(c2.close), 1e-12)
            gap_pct = gap_size / reference

            if gap_pct >= min_gap_pct:
                lower_bound = c3.high
                upper_bound = c1.low
                return self._build_gap(
                    layer=layer,
                    direction=FVGDirection.BEARISH,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    size=gap_size,
                    size_pct=gap_pct,
                    created_at=c3.timestamp,
                    created_index=c3.index,
                    source_indices=[c1.index, c2.index, c3.index],
                    impulse_ratio=c2.body_ratio,
                )

        return None

    def _build_gap(
        self,
        *,
        layer: StructureLayer,
        direction: FVGDirection,
        lower_bound: float,
        upper_bound: float,
        size: float,
        size_pct: float,
        created_at: datetime,
        created_index: int,
        source_indices: List[int],
        impulse_ratio: float,
    ) -> FairValueGap:
        mid = (lower_bound + upper_bound) / 2.0
        strength = self._calculate_gap_strength(
            size_pct=size_pct,
            impulse_ratio=impulse_ratio,
            layer=layer,
        )

        return FairValueGap(
            gap_id=self._new_id(),
            layer=layer,
            direction=direction,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            mid_price=mid,
            size=size,
            size_pct=size_pct,
            strength=strength,
            created_at=created_at,
            updated_at=created_at,
            created_index=created_index,
            source_candle_indices=source_indices,
            metadata={
                "impulse_ratio": impulse_ratio,
            },
        )

    def _register_gap(
        self,
        gap: FairValueGap,
    ) -> Tuple[Optional[FairValueGap], Optional[FVGEvent]]:
        existing = self._find_merge_candidate(gap)

        if existing is not None:
            self._merge_gap(existing, gap)
            event = self._create_event(
                event_type=FVGEventType.FVG_MERGED,
                timestamp=gap.created_at or self._state.last_update or datetime.now(timezone.utc),
                gap=existing,
                confidence=min(1.0, 0.55 + existing.strength * 0.30),
                reference_price=gap.mid_price,
                metadata={
                    "merged_gap_id": gap.gap_id,
                    "source_count": len(existing.source_candle_indices),
                },
            )
            self._append_event(event)
            return existing, event

        self._gaps_by_layer(gap.layer).append(gap)
        event = self._create_event(
            event_type=FVGEventType.FVG_CREATED,
            timestamp=gap.created_at or self._state.last_update or datetime.now(timezone.utc),
            gap=gap,
            confidence=min(1.0, 0.50 + gap.strength * 0.35),
            reference_price=gap.mid_price,
            metadata={
                "size_pct": gap.size_pct,
                "source_candle_indices": list(gap.source_candle_indices),
            },
        )
        self._append_event(event)
        return gap, event

    # ------------------------------------------------------------------
    # Gap lifecycle
    # ------------------------------------------------------------------

    def _process_gap_lifecycle(self, candle: Candle) -> List[FVGEvent]:
        events: List[FVGEvent] = []

        for layer in (StructureLayer.INTERNAL, StructureLayer.EXTERNAL):
            for gap in list(self._gaps_by_layer(layer)):
                if gap.status == FVGStatus.INVALIDATED:
                    continue

                fill_event = self._process_fill(gap, candle)
                if fill_event is not None:
                    events.append(fill_event)

                retest_event = self._process_retest(gap, candle)
                if retest_event is not None:
                    events.append(retest_event)

                respect_event = self._process_respected_reaction(gap, candle)
                if respect_event is not None:
                    events.append(respect_event)

                invalidation_event = self._process_invalidation(gap, candle)
                if invalidation_event is not None:
                    events.append(invalidation_event)

        return events

    def _process_fill(self, gap: FairValueGap, candle: Candle) -> Optional[FVGEvent]:
        fill_pct = self._calculate_fill_percentage(gap, candle)

        if fill_pct <= gap.fill_percentage:
            return None

        key = (gap.gap_id, candle.index)
        if key in self._processed_fill_keys:
            return None
        self._processed_fill_keys.add(key)

        previous_fill = gap.fill_percentage
        gap.fill_percentage = fill_pct
        gap.updated_at = candle.timestamp
        gap.last_fill_index = candle.index

        if gap.first_touch_at is None and fill_pct > 0:
            gap.first_touch_at = candle.timestamp
            gap.last_touch_index = candle.index
            gap.touch_count += 1

        event_type = FVGEventType.FVG_FILL_STARTED
        if 0 < fill_pct < 1.0:
            gap.status = FVGStatus.PARTIALLY_FILLED
            event_type = FVGEventType.FVG_PARTIALLY_FILLED
        if fill_pct >= 1.0:
            gap.status = FVGStatus.FILLED
            gap.filled_at = candle.timestamp
            event_type = FVGEventType.FVG_FILLED

        event = self._create_event(
            event_type=event_type,
            timestamp=candle.timestamp,
            gap=gap,
            confidence=min(1.0, 0.45 + gap.strength * 0.30 + fill_pct * 0.20),
            reference_price=candle.close,
            metadata={
                "previous_fill_percentage": previous_fill,
                "new_fill_percentage": fill_pct,
                "candle_index": candle.index,
            },
        )
        self._append_event(event)
        return event

    def _process_retest(self, gap: FairValueGap, candle: Candle) -> Optional[FVGEvent]:
        if gap.created_index is None:
            return None

        bars_since_creation = candle.index - gap.created_index
        if bars_since_creation < 1 or bars_since_creation > self.config.retest_window_bars:
            return None

        if not self._candle_intersects_gap(candle, gap):
            return None

        key = (gap.gap_id, candle.index)
        if key in self._processed_retest_keys:
            return None
        self._processed_retest_keys.add(key)

        gap.retest_count += 1
        gap.updated_at = candle.timestamp

        event = self._create_event(
            event_type=FVGEventType.FVG_RETESTED,
            timestamp=candle.timestamp,
            gap=gap,
            confidence=min(1.0, 0.45 + gap.strength * 0.35),
            reference_price=candle.close,
            metadata={
                "retest_count": gap.retest_count,
                "bars_since_creation": bars_since_creation,
                "candle_index": candle.index,
            },
        )
        self._append_event(event)
        return event

    def _process_respected_reaction(self, gap: FairValueGap, candle: Candle) -> Optional[FVGEvent]:
        if gap.status in {FVGStatus.RESPECTED, FVGStatus.INVALIDATED, FVGStatus.FILLED}:
            return None
        if gap.first_touch_at is None:
            return None

        reaction_threshold = max(abs(gap.mid_price) * self.config.respected_reaction_threshold_pct, 1e-12)
        respected = False

        if gap.direction == FVGDirection.BULLISH:
            if candle.close > (gap.upper_bound + reaction_threshold):
                respected = True
        else:
            if candle.close < (gap.lower_bound - reaction_threshold):
                respected = True

        if not respected:
            return None

        key = (gap.gap_id, candle.index)
        if key in self._processed_respect_keys:
            return None
        self._processed_respect_keys.add(key)

        gap.status = FVGStatus.RESPECTED
        gap.respected_at = candle.timestamp
        gap.updated_at = candle.timestamp
        gap.strength = self._recalculate_gap_strength(gap)

        event = self._create_event(
            event_type=FVGEventType.FVG_RESPECTED,
            timestamp=candle.timestamp,
            gap=gap,
            confidence=min(1.0, 0.55 + gap.strength * 0.30),
            reference_price=candle.close,
            metadata={
                "reaction_threshold": reaction_threshold,
                "candle_index": candle.index,
            },
        )
        self._append_event(event)
        return event

    def _process_invalidation(self, gap: FairValueGap, candle: Candle) -> Optional[FVGEvent]:
        if gap.status == FVGStatus.INVALIDATED:
            return None

        buffer_size = max(abs(gap.mid_price) * self.config.invalidation_close_buffer_pct, 1e-12)
        invalidated = False

        if gap.direction == FVGDirection.BULLISH:
            if candle.close < (gap.lower_bound - buffer_size):
                invalidated = True
        else:
            if candle.close > (gap.upper_bound + buffer_size):
                invalidated = True

        if not invalidated:
            return None

        key = (gap.gap_id, candle.index)
        if key in self._processed_invalidation_keys:
            return None
        self._processed_invalidation_keys.add(key)

        gap.status = FVGStatus.INVALIDATED
        gap.invalidated_at = candle.timestamp
        gap.updated_at = candle.timestamp

        event = self._create_event(
            event_type=FVGEventType.FVG_INVALIDATED,
            timestamp=candle.timestamp,
            gap=gap,
            confidence=min(1.0, 0.60 + gap.strength * 0.25),
            reference_price=candle.close,
            metadata={
                "buffer_size": buffer_size,
                "candle_index": candle.index,
            },
        )
        self._append_event(event)
        return event

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _calculate_fill_percentage(self, gap: FairValueGap, candle: Candle) -> float:
        if gap.direction == FVGDirection.BULLISH:
            if candle.low >= gap.upper_bound:
                return gap.fill_percentage
            touched_depth = gap.upper_bound - max(candle.low, gap.lower_bound)
            fill = touched_depth / max(gap.size, 1e-12)
            return max(0.0, min(1.0, fill))

        if candle.high <= gap.lower_bound:
            return gap.fill_percentage
        touched_depth = min(candle.high, gap.upper_bound) - gap.lower_bound
        fill = touched_depth / max(gap.size, 1e-12)
        return max(0.0, min(1.0, fill))

    def _candle_intersects_gap(self, candle: Candle, gap: FairValueGap) -> bool:
        return not (candle.high < gap.lower_bound or candle.low > gap.upper_bound)

    def _find_merge_candidate(self, new_gap: FairValueGap) -> Optional[FairValueGap]:
        candidates = [
            gap for gap in self._gaps_by_layer(new_gap.layer)
            if gap.direction == new_gap.direction and gap.status != FVGStatus.INVALIDATED
        ]
        if not candidates:
            return None

        merge_distance = self._merge_distance_pct(new_gap.layer)
        price_tolerance = max(abs(new_gap.mid_price) * merge_distance, 1e-12)

        for gap in reversed(candidates):
            overlapping = not (new_gap.upper_bound < gap.lower_bound or new_gap.lower_bound > gap.upper_bound)
            near_mid = abs(new_gap.mid_price - gap.mid_price) <= price_tolerance
            if overlapping or near_mid:
                return gap

        return None

    def _merge_gap(self, target: FairValueGap, incoming: FairValueGap) -> None:
        target.lower_bound = min(target.lower_bound, incoming.lower_bound)
        target.upper_bound = max(target.upper_bound, incoming.upper_bound)
        target.mid_price = (target.lower_bound + target.upper_bound) / 2.0
        target.size = target.upper_bound - target.lower_bound
        target.size_pct = max(target.size_pct, incoming.size_pct)
        target.updated_at = incoming.created_at
        target.source_candle_indices.extend(incoming.source_candle_indices)
        target.source_candle_indices = sorted(set(target.source_candle_indices))
        target.strength = self._recalculate_gap_strength(target)

    def _calculate_gap_strength(
        self,
        *,
        size_pct: float,
        impulse_ratio: float,
        layer: StructureLayer,
    ) -> float:
        score = 0.25
        score += min(1.0, size_pct / max(self._min_gap_pct(layer), 1e-12)) * 0.35
        score += min(1.0, impulse_ratio) * 0.30
        if layer == StructureLayer.EXTERNAL:
            score += 0.10
        return max(0.0, min(1.0, score))

    def _recalculate_gap_strength(self, gap: FairValueGap) -> float:
        score = 0.20
        score += min(1.0, gap.size_pct / max(self._min_gap_pct(gap.layer), 1e-12)) * 0.35
        score += min(gap.touch_count, 4) * 0.05
        score += min(gap.retest_count, 4) * 0.06

        if gap.status == FVGStatus.RESPECTED:
            score += 0.15
        elif gap.status == FVGStatus.PARTIALLY_FILLED:
            score += 0.05
        elif gap.status == FVGStatus.FILLED:
            score -= 0.10
        elif gap.status == FVGStatus.INVALIDATED:
            score -= 0.20

        if gap.layer == StructureLayer.EXTERNAL:
            score += 0.08

        return max(0.0, min(1.0, score))

    def _min_gap_pct(self, layer: StructureLayer) -> float:
        return (
            self.config.min_gap_pct_internal
            if layer == StructureLayer.INTERNAL
            else self.config.min_gap_pct_external
        )

    def _merge_distance_pct(self, layer: StructureLayer) -> float:
        return (
            self.config.merge_distance_pct_internal
            if layer == StructureLayer.INTERNAL
            else self.config.merge_distance_pct_external
        )

    def _gaps_by_layer(self, layer: StructureLayer) -> Deque[FairValueGap]:
        return self._internal_gaps if layer == StructureLayer.INTERNAL else self._external_gaps

    # ------------------------------------------------------------------
    # State refresh
    # ------------------------------------------------------------------

    def _refresh_state(self) -> None:
        self._refresh_layer_state(StructureLayer.INTERNAL)
        self._refresh_layer_state(StructureLayer.EXTERNAL)

        self._state.metadata = {
            "internal_gaps_total": len(self._internal_gaps),
            "external_gaps_total": len(self._external_gaps),
            "events_total": len(self._events),
        }

    def _refresh_layer_state(self, layer: StructureLayer) -> None:
        state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        gaps = [g for g in self._gaps_by_layer(layer)]

        active = [g for g in gaps if g.status == FVGStatus.ACTIVE]
        partial = [g for g in gaps if g.status == FVGStatus.PARTIALLY_FILLED]
        filled = [g for g in gaps if g.status == FVGStatus.FILLED]
        respected = [g for g in gaps if g.status == FVGStatus.RESPECTED]
        invalidated = [g for g in gaps if g.status == FVGStatus.INVALIDATED]

        bullish = [g for g in gaps if g.direction == FVGDirection.BULLISH and g.status != FVGStatus.INVALIDATED]
        bearish = [g for g in gaps if g.direction == FVGDirection.BEARISH and g.status != FVGStatus.INVALIDATED]

        state.total_gaps = len(gaps)
        state.active_gaps = len(active)
        state.partially_filled_gaps = len(partial)
        state.filled_gaps = len(filled)
        state.respected_gaps = len(respected)
        state.invalidated_gaps = len(invalidated)

        state.nearest_bullish_gap = self._nearest_gap_below_price(bullish, self._state.last_price)
        state.nearest_bearish_gap = self._nearest_gap_above_price(bearish, self._state.last_price)
        state.strongest_bullish_gap = self._strongest_gap(bullish)
        state.strongest_bearish_gap = self._strongest_gap(bearish)

        recent_fill_events = [
            e for e in self._events
            if e.layer == layer and e.event_type in {
                FVGEventType.FVG_FILL_STARTED,
                FVGEventType.FVG_PARTIALLY_FILLED,
                FVGEventType.FVG_FILLED,
            }
        ]
        state.recent_fill_activity = max(0.0, min(1.0, len(recent_fill_events[-10:]) / 8.0))
        layer_events = [e for e in self._events if e.layer == layer]
        state.last_event = layer_events[-1] if layer_events else None
        state.metadata = {
            "bullish_gaps": len(bullish),
            "bearish_gaps": len(bearish),
        }

    # ------------------------------------------------------------------
    # Strategy helper functions
    # ------------------------------------------------------------------

    def bullish_fvg_score(self, layer: StructureLayer) -> float:
        state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        gap = state.strongest_bullish_gap
        if gap is None:
            return 0.0
        return max(0.0, min(1.0, gap.strength))

    def bearish_fvg_score(self, layer: StructureLayer) -> float:
        state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        gap = state.strongest_bearish_gap
        if gap is None:
            return 0.0
        return max(0.0, min(1.0, gap.strength))

    def fill_pressure_score(self, layer: StructureLayer) -> float:
        state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        return max(0.0, min(1.0, state.recent_fill_activity))

    def inefficiency_pressure_score(self, layer: StructureLayer) -> float:
        """
        Високий score = на ринку багато незакритих inefficiency.
        Корисно для continuation / magnet logic.
        """
        gaps = [
            g for g in self._gaps_by_layer(layer)
            if g.status in {FVGStatus.ACTIVE, FVGStatus.PARTIALLY_FILLED, FVGStatus.RESPECTED}
        ]
        if not gaps:
            return 0.0

        score = 0.0
        for gap in gaps[-10:]:
            score += gap.strength * (1.0 - gap.fill_percentage)

        return max(0.0, min(1.0, score / 5.0))

    def mean_reversion_gap_score(self, layer: StructureLayer) -> float:
        """
        Високий score = є сенс чекати повернення ціни в gap.
        """
        gaps = [
            g for g in self._gaps_by_layer(layer)
            if g.status in {FVGStatus.ACTIVE, FVGStatus.PARTIALLY_FILLED}
        ]
        if not gaps:
            return 0.0

        values = []
        for gap in gaps[-8:]:
            values.append(gap.strength * (1.0 - gap.fill_percentage))

        return max(0.0, min(1.0, self._safe_mean(values)))

    def respect_quality_score(self, gap: FairValueGap) -> float:
        score = 0.0
        score += gap.strength * 0.40
        score += (1.0 - gap.fill_percentage) * 0.20
        score += min(gap.retest_count, 3) * 0.10
        if gap.status == FVGStatus.RESPECTED:
            score += 0.30
        return max(0.0, min(1.0, score))

    def nearest_gap_summary(self, layer: StructureLayer) -> Dict[str, Any]:
        state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        return {
            "nearest_bullish_gap": self._gap_to_dict(state.nearest_bullish_gap),
            "nearest_bearish_gap": self._gap_to_dict(state.nearest_bearish_gap),
            "strongest_bullish_gap": self._gap_to_dict(state.strongest_bullish_gap),
            "strongest_bearish_gap": self._gap_to_dict(state.strongest_bearish_gap),
            "fill_pressure_score": self.fill_pressure_score(layer),
            "inefficiency_pressure_score": self.inefficiency_pressure_score(layer),
            "mean_reversion_gap_score": self.mean_reversion_gap_score(layer),
        }

    # ------------------------------------------------------------------
    # EventBus
    # ------------------------------------------------------------------

    def _append_event(self, event: FVGEvent) -> None:
        self._events.append(event)
        self._emit_event(event)

    def _emit_event(self, event: FVGEvent) -> None:
        if not self.config.emit_events or self.event_bus is None:
            return

        event_name = f"{self.config.event_namespace}.{event.event_type.value}"
        payload = {
            "source": self.config.event_namespace,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "event": self._event_to_dict(event),
            "state": {
                "last_price": self._state.last_price,
            },
        }

        try:
            if hasattr(self.event_bus, "emit"):
                self.event_bus.emit(event_name, payload)
            elif hasattr(self.event_bus, "publish"):
                self.event_bus.publish(event_name, payload)
            elif hasattr(self.event_bus, "dispatch"):
                self.event_bus.dispatch(event_name, payload)
            else:
                self.logger.warning(
                    "EventBus provided but no supported method found",
                    extra={
                        "symbol": self.symbol,
                        "timeframe": self.timeframe,
                        "event_name": event_name,
                    },
                )
                return
        except Exception as exc:
            self.logger.exception(
                "Failed to emit FVG event",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "event_name": event_name,
                    "error": str(exc),
                },
            )

    def _publish_snapshot(self) -> None:
        if self.event_bus is None:
            return

        event_name = f"{self.config.event_namespace}.snapshot"
        payload = {
            "source": self.config.event_namespace,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "snapshot": self.snapshot(),
        }

        try:
            if hasattr(self.event_bus, "emit"):
                self.event_bus.emit(event_name, payload)
            elif hasattr(self.event_bus, "publish"):
                self.event_bus.publish(event_name, payload)
            elif hasattr(self.event_bus, "dispatch"):
                self.event_bus.dispatch(event_name, payload)
        except Exception as exc:
            self.logger.exception(
                "Failed to publish FVG snapshot",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "error": str(exc),
                },
            )

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    def _nearest_gap_below_price(
        self,
        gaps: Sequence[FairValueGap],
        price: Optional[float],
    ) -> Optional[FairValueGap]:
        if price is None:
            return None
        candidates = [g for g in gaps if g.mid_price <= price]
        if not candidates:
            return None
        return min(candidates, key=lambda g: abs(price - g.mid_price))

    def _nearest_gap_above_price(
        self,
        gaps: Sequence[FairValueGap],
        price: Optional[float],
    ) -> Optional[FairValueGap]:
        if price is None:
            return None
        candidates = [g for g in gaps if g.mid_price >= price]
        if not candidates:
            return None
        return min(candidates, key=lambda g: abs(price - g.mid_price))

    def _strongest_gap(self, gaps: Sequence[FairValueGap]) -> Optional[FairValueGap]:
        if not gaps:
            return None
        return max(gaps, key=lambda g: g.strength)

    def _create_event(
        self,
        *,
        event_type: FVGEventType,
        timestamp: datetime,
        gap: FairValueGap,
        confidence: float,
        reference_price: Optional[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FVGEvent:
        return FVGEvent(
            event_id=self._new_id(),
            event_type=event_type,
            timestamp=timestamp,
            symbol=self.symbol,
            timeframe=self.timeframe,
            layer=gap.layer,
            gap_id=gap.gap_id,
            direction=gap.direction,
            upper_bound=gap.upper_bound,
            lower_bound=gap.lower_bound,
            fill_percentage=gap.fill_percentage,
            confidence=max(0.0, min(1.0, confidence)),
            reference_price=reference_price,
            metadata=metadata or {},
        )

    @staticmethod
    def _safe_mean(values: Sequence[float]) -> float:
        return mean(values) if values else 0.0

    @staticmethod
    def _new_id() -> str:
        return uuid4().hex

    # ------------------------------------------------------------------
    # Parsing / normalization
    # ------------------------------------------------------------------

    def _parse_candle(self, raw: Mapping[str, Any]) -> Candle:
        timestamp = raw.get("timestamp", raw.get("time", raw.get("ts")))
        if timestamp is None:
            raise ValueError("Candle is missing timestamp/time/ts")

        open_ = raw.get("open", raw.get("o"))
        high = raw.get("high", raw.get("h"))
        low = raw.get("low", raw.get("l"))
        close = raw.get("close", raw.get("c"))
        volume = raw.get("volume", raw.get("v", 0.0))

        if open_ is None or high is None or low is None or close is None:
            raise ValueError("Candle must contain open/high/low/close")

        dt = self._normalize_timestamp(timestamp)
        candle = Candle(
            timestamp=dt,
            open=float(open_),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=float(volume),
            index=self._global_candle_index,
        )
        self._global_candle_index += 1

        if candle.low > candle.high:
            raise ValueError("Invalid candle: low cannot be greater than high")
        if min(candle.open, candle.high, candle.low, candle.close) < 0:
            raise ValueError("Invalid candle: OHLC cannot be negative")

        return candle

    @staticmethod
    def _normalize_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value

        if isinstance(value, (int, float)):
            if value > 1e12:
                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(value, tz=timezone.utc)

        if isinstance(value, str):
            normalized = value.strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _gap_to_dict(self, gap: Optional[FairValueGap]) -> Optional[Dict[str, Any]]:
        if gap is None:
            return None

        return {
            "gap_id": gap.gap_id,
            "layer": gap.layer.value,
            "direction": gap.direction.value,
            "upper_bound": gap.upper_bound,
            "lower_bound": gap.lower_bound,
            "mid_price": gap.mid_price,
            "size": gap.size,
            "size_pct": gap.size_pct,
            "strength": gap.strength,
            "status": gap.status.value,
            "fill_percentage": gap.fill_percentage,
            "touch_count": gap.touch_count,
            "retest_count": gap.retest_count,
            "created_at": gap.created_at.isoformat() if gap.created_at else None,
            "updated_at": gap.updated_at.isoformat() if gap.updated_at else None,
            "first_touch_at": gap.first_touch_at.isoformat() if gap.first_touch_at else None,
            "filled_at": gap.filled_at.isoformat() if gap.filled_at else None,
            "respected_at": gap.respected_at.isoformat() if gap.respected_at else None,
            "invalidated_at": gap.invalidated_at.isoformat() if gap.invalidated_at else None,
            "created_index": gap.created_index,
            "last_touch_index": gap.last_touch_index,
            "last_fill_index": gap.last_fill_index,
            "source_candle_indices": list(gap.source_candle_indices),
            "metadata": dict(gap.metadata),
        }

    def _event_to_dict(self, event: Optional[FVGEvent]) -> Optional[Dict[str, Any]]:
        if event is None:
            return None

        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "symbol": event.symbol,
            "timeframe": event.timeframe,
            "layer": event.layer.value,
            "gap_id": event.gap_id,
            "direction": event.direction.value,
            "upper_bound": event.upper_bound,
            "lower_bound": event.lower_bound,
            "fill_percentage": event.fill_percentage,
            "confidence": event.confidence,
            "reference_price": event.reference_price,
            "metadata": dict(event.metadata),
        }

    def _layer_state_to_dict(self, state: LayerFVGState) -> Dict[str, Any]:
        return {
            "layer": state.layer.value,
            "total_gaps": state.total_gaps,
            "active_gaps": state.active_gaps,
            "partially_filled_gaps": state.partially_filled_gaps,
            "filled_gaps": state.filled_gaps,
            "respected_gaps": state.respected_gaps,
            "invalidated_gaps": state.invalidated_gaps,
            "nearest_bullish_gap": self._gap_to_dict(state.nearest_bullish_gap),
            "nearest_bearish_gap": self._gap_to_dict(state.nearest_bearish_gap),
            "strongest_bullish_gap": self._gap_to_dict(state.strongest_bullish_gap),
            "strongest_bearish_gap": self._gap_to_dict(state.strongest_bearish_gap),
            "recent_fill_activity": state.recent_fill_activity,
            "last_event": self._event_to_dict(state.last_event),
            "metadata": dict(state.metadata),
        }