from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from core.logger import get_logger


class SwingType(str, Enum):
    HIGH = "high"
    LOW = "low"


class StructureLayer(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"


class StructureEventType(str, Enum):
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    HH = "hh"
    HL = "hl"
    LH = "lh"
    LL = "ll"
    BOS = "bos"
    CHOCH = "choch"
    MSS = "mss"


class MarketBias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class MarketStructureConfig:
    pivot_left: int = 3
    pivot_right: int = 3
    internal_min_swing_distance_pct: float = 0.0008
    external_min_swing_distance_pct: float = 0.0020
    structure_break_threshold_pct: float = 0.0005
    require_close_break: bool = True
    max_candles: int = 4000
    max_internal_swings: int = 800
    max_external_swings: int = 400
    max_events: int = 1000
    emit_events: bool = True
    event_namespace: str = "price_action.market_structure"
    publish_snapshots: bool = False
    alignment_window: int = 5
    external_strength_multiplier: float = 1.35
    min_external_strength: float = 0.30

    def validate(self) -> None:
        if self.pivot_left < 1 or self.pivot_right < 1:
            raise ValueError("pivot_left and pivot_right must be >= 1")
        if self.internal_min_swing_distance_pct < 0:
            raise ValueError("internal_min_swing_distance_pct must be >= 0")
        if self.external_min_swing_distance_pct < 0:
            raise ValueError("external_min_swing_distance_pct must be >= 0")
        if self.structure_break_threshold_pct < 0:
            raise ValueError("structure_break_threshold_pct must be >= 0")
        if self.max_candles < (self.pivot_left + self.pivot_right + 10):
            raise ValueError("max_candles is too small for selected pivot settings")
        if self.alignment_window < 1:
            raise ValueError("alignment_window must be >= 1")


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


class MarketStructureAnalyzer:
    """
    Stateful production-style market structure analyzer.

    Features
    --------
    - incremental pivot-based swing detection
    - internal / external structure separation
    - HH / HL / LH / LL classification
    - BOS / CHoCH / MSS detection
    - EventBus integration
    - optional higher timeframe alignment
    - snapshot-oriented API for strategy layer
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        event_bus: Optional[Any] = None,
        config: Optional[MarketStructureConfig] = None,
        higher_timeframe: Optional[str] = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.event_bus = event_bus
        self.config = config or MarketStructureConfig()
        self.config.validate()
        self.higher_timeframe = higher_timeframe

        self.logger = get_logger(__name__, service_name="price_action.market_structure")

        self._candles: Deque[Candle] = deque(maxlen=self.config.max_candles)
        self._internal_swings: Deque[SwingPoint] = deque(maxlen=self.config.max_internal_swings)
        self._external_swings: Deque[SwingPoint] = deque(maxlen=self.config.max_external_swings)
        self._events: Deque[StructureEvent] = deque(maxlen=self.config.max_events)

        self._processed_pivots: set[Tuple[int, SwingType]] = set()
        self._processed_structure_labels: set[Tuple[str, StructureEventType, StructureLayer]] = set()
        self._processed_breaks: set[Tuple[StructureLayer, str, str, str]] = set()

        self._global_candle_index: int = 0
        self._last_processed_pivot_center_index: int = -1

        self._state = MarketStructureState(
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        self.logger.info(
            "Initialized MarketStructureAnalyzer",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "higher_timeframe": self.higher_timeframe,
                "config": asdict(self.config),
            },
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        self._candles.clear()
        self._internal_swings.clear()
        self._external_swings.clear()
        self._events.clear()

        self._processed_pivots.clear()
        self._processed_structure_labels.clear()
        self._processed_breaks.clear()

        self._global_candle_index = 0
        self._last_processed_pivot_center_index = -1

        self._state = MarketStructureState(
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        self.logger.info(
            "MarketStructureAnalyzer reset",
            extra={"symbol": self.symbol, "timeframe": self.timeframe},
        )

    def update(
        self,
        candles: Sequence[Mapping[str, Any]],
        *,
        higher_timeframe_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.add_candles(candles, higher_timeframe_context=higher_timeframe_context)

    def add_candle(
        self,
        candle: Mapping[str, Any],
        *,
        higher_timeframe_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.add_candles([candle], higher_timeframe_context=higher_timeframe_context)

    def add_candles(
        self,
        candles: Sequence[Mapping[str, Any]],
        *,
        higher_timeframe_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not candles:
            if higher_timeframe_context is not None:
                self.update_higher_timeframe_context(higher_timeframe_context)
            self._refresh_state()
            return {
                "state": self.snapshot(),
                "new_swings": [],
                "new_events": [],
            }

        new_swings: List[SwingPoint] = []
        new_events: List[StructureEvent] = []

        for raw in candles:
            candle = self._parse_candle(raw)
            self._candles.append(candle)
            self._state.last_price = candle.close
            self._state.last_update = candle.timestamp

            swings_from_increment = self._process_incremental_pivot_detection()
            if swings_from_increment:
                new_swings.extend(swings_from_increment)

        if new_swings:
            label_events = self._classify_structure_labels(new_swings)
            if label_events:
                new_events.extend(label_events)

        break_events = self._detect_break_events()
        if break_events:
            new_events.extend(break_events)

        if higher_timeframe_context is not None:
            self.update_higher_timeframe_context(higher_timeframe_context)

        self._refresh_state()

        if self.config.publish_snapshots:
            self._publish_snapshot()

        self.logger.debug(
            "Market structure incrementally updated",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "added_candles": len(candles),
                "new_swings": len(new_swings),
                "new_events": len(new_events),
                "internal_bias": self._state.internal.bias.value,
                "external_bias": self._state.external.bias.value,
                "alignment_score": self._state.mtf_alignment.alignment_score,
            },
        )

        return {
            "state": self.snapshot(),
            "new_swings": [self._swing_to_dict(x) for x in new_swings],
            "new_events": [self._event_to_dict(x) for x in new_events],
        }

    def update_higher_timeframe_context(self, context: Mapping[str, Any]) -> None:
        mtf = self._state.mtf_alignment

        tf = context.get("timeframe") or context.get("higher_timeframe") or self.higher_timeframe
        bias_raw = context.get("bias", context.get("higher_timeframe_bias", MarketBias.UNKNOWN.value))
        conf = float(context.get("confidence", context.get("higher_timeframe_confidence", 0.0)))

        if isinstance(bias_raw, MarketBias):
            bias = bias_raw
        else:
            try:
                bias = MarketBias(str(bias_raw))
            except ValueError:
                bias = MarketBias.UNKNOWN

        mtf.higher_timeframe = tf
        mtf.higher_timeframe_bias = bias
        mtf.higher_timeframe_confidence = max(0.0, min(1.0, conf))
        mtf.last_updated = self._state.last_update or datetime.now(timezone.utc)
        mtf.metadata = {
            "raw_context_keys": list(context.keys()),
        }

        self._compute_alignment()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "symbol": self._state.symbol,
            "timeframe": self._state.timeframe,
            "last_price": self._state.last_price,
            "last_update": self._state.last_update.isoformat() if self._state.last_update else None,
            "internal": self._layer_state_to_dict(self._state.internal),
            "external": self._layer_state_to_dict(self._state.external),
            "mtf_alignment": self._alignment_to_dict(self._state.mtf_alignment),
            "metadata": dict(self._state.metadata),
        }

    def get_state(self) -> MarketStructureState:
        return self._state

    def get_internal_swings(self) -> List[SwingPoint]:
        return list(self._internal_swings)

    def get_external_swings(self) -> List[SwingPoint]:
        return list(self._external_swings)

    def get_events(self) -> List[StructureEvent]:
        return list(self._events)

    # -------------------------------------------------------------------------
    # Incremental processing
    # -------------------------------------------------------------------------

    def _process_incremental_pivot_detection(self) -> List[SwingPoint]:
        """
        True incremental pivot processing:
        on each newly appended candle, only one pivot-center becomes confirmable:
            center = current_index - pivot_right
        """
        if len(self._candles) < (self.config.pivot_left + self.config.pivot_right + 1):
            return []

        last_local_index = len(self._candles) - 1
        center_local_index = last_local_index - self.config.pivot_right

        if center_local_index <= self._last_processed_pivot_center_index:
            return []

        self._last_processed_pivot_center_index = center_local_index
        return self._detect_pivot_at_local_index(center_local_index)

    def _detect_pivot_at_local_index(self, center_local_index: int) -> List[SwingPoint]:
        left = self.config.pivot_left
        right = self.config.pivot_right

        if center_local_index - left < 0:
            return []
        if center_local_index + right >= len(self._candles):
            return []

        candles = list(self._candles)
        center = candles[center_local_index]
        window = candles[center_local_index - left:center_local_index + right + 1]

        is_swing_high = all(
            center.high >= candle.high
            for idx, candle in enumerate(window)
            if idx != left
        )
        is_swing_low = all(
            center.low <= candle.low
            for idx, candle in enumerate(window)
            if idx != left
        )

        detected: List[SwingPoint] = []

        if is_swing_high:
            pivot_key = (center.index, SwingType.HIGH)
            if pivot_key not in self._processed_pivots:
                internal_swing = self._build_swing(center, SwingType.HIGH, StructureLayer.INTERNAL)
                if self._accept_internal_swing(internal_swing):
                    self._internal_swings.append(internal_swing)
                    detected.append(internal_swing)
                    self._processed_pivots.add(pivot_key)
                    self._register_swing_event(internal_swing)

                    external_candidate = self._try_promote_external_swing(internal_swing)
                    if external_candidate is not None:
                        self._external_swings.append(external_candidate)
                        detected.append(external_candidate)
                        self._register_swing_event(external_candidate)

        if is_swing_low:
            pivot_key = (center.index, SwingType.LOW)
            if pivot_key not in self._processed_pivots:
                internal_swing = self._build_swing(center, SwingType.LOW, StructureLayer.INTERNAL)
                if self._accept_internal_swing(internal_swing):
                    self._internal_swings.append(internal_swing)
                    detected.append(internal_swing)
                    self._processed_pivots.add(pivot_key)
                    self._register_swing_event(internal_swing)

                    external_candidate = self._try_promote_external_swing(internal_swing)
                    if external_candidate is not None:
                        self._external_swings.append(external_candidate)
                        detected.append(external_candidate)
                        self._register_swing_event(external_candidate)

        return detected

    # -------------------------------------------------------------------------
    # Swing construction / filtering
    # -------------------------------------------------------------------------

    def _build_swing(
        self,
        candle: Candle,
        swing_type: SwingType,
        layer: StructureLayer,
    ) -> SwingPoint:
        if swing_type == SwingType.HIGH:
            price = candle.high
            wick_component = (candle.high - candle.body_high) / max(candle.high, 1e-12)
        else:
            price = candle.low
            wick_component = (candle.body_low - candle.low) / max(abs(candle.low), 1e-12)

        range_component = candle.range_size / max(abs(candle.close), 1e-12)
        body_component = candle.body_size / max(abs(candle.close), 1e-12)

        base_strength = max(
            0.0,
            min(
                1.0,
                wick_component * 8.0 + range_component * 6.0 + body_component * 2.0,
            ),
        )

        if layer == StructureLayer.EXTERNAL:
            base_strength = min(1.0, base_strength * self.config.external_strength_multiplier)

        return SwingPoint(
            swing_id=self._new_id(),
            timestamp=candle.timestamp,
            price=price,
            swing_type=swing_type,
            layer=layer,
            index=candle.index,
            candle_open=candle.open,
            candle_high=candle.high,
            candle_low=candle.low,
            candle_close=candle.close,
            strength=base_strength,
            is_confirmed=True,
            metadata={
                "range_size": candle.range_size,
                "body_size": candle.body_size,
            },
        )

    def _accept_internal_swing(self, swing: SwingPoint) -> bool:
        same_type = [s for s in self._internal_swings if s.swing_type == swing.swing_type]
        if not same_type:
            return True

        prev = same_type[-1]
        min_distance = max(abs(prev.price) * self.config.internal_min_swing_distance_pct, 1e-12)

        if swing.index <= prev.index:
            return False
        if abs(swing.price - prev.price) < min_distance:
            return False

        return True

    def _try_promote_external_swing(self, internal_swing: SwingPoint) -> Optional[SwingPoint]:
        """
        External structure = more significant pivots.
        Promotion rules:
        - stronger distance from previous same-type external swing
        - minimum strength threshold
        - usually meaningful expansion vs last external same-type swing
        """
        if internal_swing.strength < self.config.min_external_strength:
            return None

        same_type_ext = [s for s in self._external_swings if s.swing_type == internal_swing.swing_type]
        if not same_type_ext:
            return self._clone_swing_for_external(internal_swing)

        prev = same_type_ext[-1]
        min_distance = max(abs(prev.price) * self.config.external_min_swing_distance_pct, 1e-12)

        if internal_swing.index <= prev.index:
            return None
        if abs(internal_swing.price - prev.price) < min_distance:
            return None

        if internal_swing.swing_type == SwingType.HIGH and internal_swing.price <= prev.price:
            # external highs usually matter more when they actually extend or strongly reset structure
            # but still allow lower highs if strength is very strong
            if internal_swing.strength < 0.7:
                return None

        if internal_swing.swing_type == SwingType.LOW and internal_swing.price >= prev.price:
            if internal_swing.strength < 0.7:
                return None

        return self._clone_swing_for_external(internal_swing)

    def _clone_swing_for_external(self, swing: SwingPoint) -> SwingPoint:
        return SwingPoint(
            swing_id=self._new_id(),
            timestamp=swing.timestamp,
            price=swing.price,
            swing_type=swing.swing_type,
            layer=StructureLayer.EXTERNAL,
            index=swing.index,
            candle_open=swing.candle_open,
            candle_high=swing.candle_high,
            candle_low=swing.candle_low,
            candle_close=swing.candle_close,
            strength=min(1.0, swing.strength * self.config.external_strength_multiplier),
            is_confirmed=swing.is_confirmed,
            metadata={**swing.metadata, "promoted_from_internal": swing.swing_id},
        )

    # -------------------------------------------------------------------------
    # Structure classification
    # -------------------------------------------------------------------------

    def _classify_structure_labels(self, new_swings: Sequence[SwingPoint]) -> List[StructureEvent]:
        events: List[StructureEvent] = []

        for swing in new_swings:
            storage = self._swings_by_layer(swing.layer)

            if swing.swing_type == SwingType.HIGH:
                previous_highs = [x for x in storage if x.swing_type == SwingType.HIGH and x.swing_id != swing.swing_id]
                if previous_highs:
                    prev = previous_highs[-1]
                    event_type = StructureEventType.HH if swing.price > prev.price else StructureEventType.LH
                    direction = MarketBias.BULLISH if event_type == StructureEventType.HH else MarketBias.BEARISH
                    event = self._create_structure_label_event(
                        swing=swing,
                        event_type=event_type,
                        direction=direction,
                        reference_swing=prev,
                    )
                    if event is not None:
                        events.append(event)

            elif swing.swing_type == SwingType.LOW:
                previous_lows = [x for x in storage if x.swing_type == SwingType.LOW and x.swing_id != swing.swing_id]
                if previous_lows:
                    prev = previous_lows[-1]
                    event_type = StructureEventType.HL if swing.price > prev.price else StructureEventType.LL
                    direction = MarketBias.BULLISH if event_type == StructureEventType.HL else MarketBias.BEARISH
                    event = self._create_structure_label_event(
                        swing=swing,
                        event_type=event_type,
                        direction=direction,
                        reference_swing=prev,
                    )
                    if event is not None:
                        events.append(event)

        for event in events:
            self._append_event(event)

        return events

    def _create_structure_label_event(
        self,
        *,
        swing: SwingPoint,
        event_type: StructureEventType,
        direction: MarketBias,
        reference_swing: SwingPoint,
    ) -> Optional[StructureEvent]:
        key = (swing.swing_id, event_type, swing.layer)
        if key in self._processed_structure_labels:
            return None

        delta_pct = abs(swing.price - reference_swing.price) / max(abs(reference_swing.price), 1e-12)
        confidence = max(
            0.0,
            min(
                1.0,
                0.45 + delta_pct * 20.0 + swing.strength * 0.25,
            ),
        )

        event = StructureEvent(
            event_id=self._new_id(),
            event_type=event_type,
            timestamp=swing.timestamp,
            price=swing.price,
            layer=swing.layer,
            direction=direction,
            swing_id=swing.swing_id,
            reference_price=reference_swing.price,
            reference_swing_id=reference_swing.swing_id,
            confidence=confidence,
            metadata={
                "delta_pct": delta_pct,
                "reference_index": reference_swing.index,
                "current_index": swing.index,
            },
        )
        self._processed_structure_labels.add(key)
        return event

    # -------------------------------------------------------------------------
    # Break events: BOS / CHoCH / MSS
    # -------------------------------------------------------------------------

    def _detect_break_events(self) -> List[StructureEvent]:
        if not self._candles:
            return []

        current_candle = self._candles[-1]
        events: List[StructureEvent] = []

        for layer in (StructureLayer.INTERNAL, StructureLayer.EXTERNAL):
            swings = self._swings_by_layer(layer)
            if len(swings) < 2:
                continue

            last_high = self._last_swing(layer, SwingType.HIGH)
            last_low = self._last_swing(layer, SwingType.LOW)

            if last_high and self._is_break_above(current_candle, last_high.price):
                bos = self._create_break_event(
                    layer=layer,
                    break_type=StructureEventType.BOS,
                    direction=MarketBias.BULLISH,
                    candle=current_candle,
                    reference_swing=last_high,
                )
                if bos is not None:
                    events.append(bos)

            if last_low and self._is_break_below(current_candle, last_low.price):
                bos = self._create_break_event(
                    layer=layer,
                    break_type=StructureEventType.BOS,
                    direction=MarketBias.BEARISH,
                    candle=current_candle,
                    reference_swing=last_low,
                )
                if bos is not None:
                    events.append(bos)

        derived = self._derive_choch_and_mss(events)
        if derived:
            events.extend(derived)

        for event in events:
            self._append_event(event)

        return events

    def _create_break_event(
        self,
        *,
        layer: StructureLayer,
        break_type: StructureEventType,
        direction: MarketBias,
        candle: Candle,
        reference_swing: SwingPoint,
    ) -> Optional[StructureEvent]:
        break_key = (
            layer,
            direction.value,
            reference_swing.swing_id,
            candle.timestamp.isoformat(),
        )
        if break_key in self._processed_breaks:
            return None

        if direction == MarketBias.BULLISH:
            penetration = (
                candle.close - reference_swing.price
                if self.config.require_close_break
                else candle.high - reference_swing.price
            )
        else:
            penetration = (
                reference_swing.price - candle.close
                if self.config.require_close_break
                else reference_swing.price - candle.low
            )

        confidence = max(
            0.0,
            min(
                1.0,
                0.50
                + (penetration / max(abs(reference_swing.price), 1e-12)) * 45.0
                + (candle.body_size / max(abs(candle.close), 1e-12)) * 5.0,
            ),
        )

        event = StructureEvent(
            event_id=self._new_id(),
            event_type=break_type,
            timestamp=candle.timestamp,
            price=candle.close,
            layer=layer,
            direction=direction,
            swing_id=None,
            reference_price=reference_swing.price,
            reference_swing_id=reference_swing.swing_id,
            confidence=confidence,
            metadata={
                "reference_swing_type": reference_swing.swing_type.value,
                "reference_index": reference_swing.index,
                "penetration": penetration,
                "require_close_break": self.config.require_close_break,
            },
        )

        self._processed_breaks.add(break_key)
        return event

    def _derive_choch_and_mss(self, bos_events: Sequence[StructureEvent]) -> List[StructureEvent]:
        derived: List[StructureEvent] = []

        for bos in bos_events:
            if bos.direction is None:
                continue

            previous_bias = self._infer_layer_bias_before_timestamp(
                layer=bos.layer,
                timestamp=bos.timestamp,
                exclude_event_id=bos.event_id,
            )

            if previous_bias in (MarketBias.BULLISH, MarketBias.BEARISH) and previous_bias != bos.direction:
                choch = StructureEvent(
                    event_id=self._new_id(),
                    event_type=StructureEventType.CHOCH,
                    timestamp=bos.timestamp,
                    price=bos.price,
                    layer=bos.layer,
                    direction=bos.direction,
                    swing_id=None,
                    reference_price=bos.reference_price,
                    reference_swing_id=bos.reference_swing_id,
                    confidence=min(1.0, bos.confidence * 0.96),
                    metadata={
                        "trigger_event_id": bos.event_id,
                        "previous_bias": previous_bias.value,
                    },
                )
                mss = StructureEvent(
                    event_id=self._new_id(),
                    event_type=StructureEventType.MSS,
                    timestamp=bos.timestamp,
                    price=bos.price,
                    layer=bos.layer,
                    direction=bos.direction,
                    swing_id=None,
                    reference_price=bos.reference_price,
                    reference_swing_id=bos.reference_swing_id,
                    confidence=min(1.0, bos.confidence * 0.92),
                    metadata={
                        "trigger_event_id": bos.event_id,
                        "previous_bias": previous_bias.value,
                    },
                )
                derived.extend([choch, mss])

        return derived

    # -------------------------------------------------------------------------
    # State refresh
    # -------------------------------------------------------------------------

    def _refresh_state(self) -> None:
        self._refresh_layer_state(StructureLayer.INTERNAL)
        self._refresh_layer_state(StructureLayer.EXTERNAL)
        self._compute_alignment()

        self._state.metadata = {
            "internal_swings_total": len(self._internal_swings),
            "external_swings_total": len(self._external_swings),
            "events_total": len(self._events),
        }

    def _refresh_layer_state(self, layer: StructureLayer) -> None:
        layer_state = self._layer_state(layer)
        swings = self._swings_by_layer(layer)
        layer_events = [e for e in self._events if e.layer == layer]

        layer_state.last_swing_high = self._last_swing(layer, SwingType.HIGH)
        layer_state.previous_swing_high = self._previous_swing(layer, SwingType.HIGH)
        layer_state.last_swing_low = self._last_swing(layer, SwingType.LOW)
        layer_state.previous_swing_low = self._previous_swing(layer, SwingType.LOW)

        layer_state.last_hh = self._last_event(layer, StructureEventType.HH)
        layer_state.last_hl = self._last_event(layer, StructureEventType.HL)
        layer_state.last_lh = self._last_event(layer, StructureEventType.LH)
        layer_state.last_ll = self._last_event(layer, StructureEventType.LL)
        layer_state.last_bos = self._last_event(layer, StructureEventType.BOS)
        layer_state.last_choch = self._last_event(layer, StructureEventType.CHOCH)
        layer_state.last_mss = self._last_event(layer, StructureEventType.MSS)

        layer_state.swing_count = len(swings)
        layer_state.event_count = len(layer_events)
        layer_state.sequence = [
            event.event_type.value
            for event in layer_events[-8:]
            if event.event_type in {
                StructureEventType.HH,
                StructureEventType.HL,
                StructureEventType.LH,
                StructureEventType.LL,
                StructureEventType.BOS,
                StructureEventType.CHOCH,
                StructureEventType.MSS,
            }
        ]

        bias = self._infer_layer_bias(layer)
        confidence = self._compute_layer_confidence(layer, bias)
        trend_strength = self._compute_layer_trend_strength(layer, bias)
        in_breakout = self._is_layer_in_recent_breakout(layer)

        layer_state.bias = bias
        layer_state.confidence = confidence
        layer_state.trend_strength = trend_strength
        layer_state.in_breakout = in_breakout

    # -------------------------------------------------------------------------
    # Bias / confidence / alignment
    # -------------------------------------------------------------------------

    def _infer_layer_bias(self, layer: StructureLayer) -> MarketBias:
        relevant = [
            e for e in self._events
            if e.layer == layer and e.event_type in {
                StructureEventType.HH,
                StructureEventType.HL,
                StructureEventType.LH,
                StructureEventType.LL,
                StructureEventType.BOS,
                StructureEventType.CHOCH,
            }
        ]

        if not relevant:
            return MarketBias.UNKNOWN

        score = 0.0
        for event in relevant[-8:]:
            if event.event_type in {StructureEventType.HH, StructureEventType.HL}:
                score += 1.0
            elif event.event_type in {StructureEventType.LH, StructureEventType.LL}:
                score -= 1.0
            elif event.event_type == StructureEventType.BOS and event.direction == MarketBias.BULLISH:
                score += 1.6
            elif event.event_type == StructureEventType.BOS and event.direction == MarketBias.BEARISH:
                score -= 1.6
            elif event.event_type == StructureEventType.CHOCH and event.direction == MarketBias.BULLISH:
                score += 1.2
            elif event.event_type == StructureEventType.CHOCH and event.direction == MarketBias.BEARISH:
                score -= 1.2

        if score >= 2.0:
            return MarketBias.BULLISH
        if score <= -2.0:
            return MarketBias.BEARISH
        return MarketBias.RANGING

    def _infer_layer_bias_before_timestamp(
        self,
        *,
        layer: StructureLayer,
        timestamp: datetime,
        exclude_event_id: Optional[str] = None,
    ) -> MarketBias:
        relevant = [
            e for e in self._events
            if e.layer == layer
            and e.timestamp <= timestamp
            and e.event_id != exclude_event_id
            and e.event_type in {
                StructureEventType.HH,
                StructureEventType.HL,
                StructureEventType.LH,
                StructureEventType.LL,
                StructureEventType.BOS,
                StructureEventType.CHOCH,
            }
        ]

        if not relevant:
            return MarketBias.UNKNOWN

        score = 0.0
        for event in relevant[-8:]:
            if event.event_type in {StructureEventType.HH, StructureEventType.HL}:
                score += 1.0
            elif event.event_type in {StructureEventType.LH, StructureEventType.LL}:
                score -= 1.0
            elif event.event_type == StructureEventType.BOS and event.direction == MarketBias.BULLISH:
                score += 1.6
            elif event.event_type == StructureEventType.BOS and event.direction == MarketBias.BEARISH:
                score -= 1.6
            elif event.event_type == StructureEventType.CHOCH and event.direction == MarketBias.BULLISH:
                score += 1.2
            elif event.event_type == StructureEventType.CHOCH and event.direction == MarketBias.BEARISH:
                score -= 1.2

        if score >= 2.0:
            return MarketBias.BULLISH
        if score <= -2.0:
            return MarketBias.BEARISH
        return MarketBias.RANGING

    def _compute_layer_confidence(self, layer: StructureLayer, bias: MarketBias) -> float:
        if bias == MarketBias.UNKNOWN:
            return 0.0

        relevant = [
            e for e in self._events
            if e.layer == layer and e.event_type in {
                StructureEventType.HH,
                StructureEventType.HL,
                StructureEventType.LH,
                StructureEventType.LL,
                StructureEventType.BOS,
                StructureEventType.CHOCH,
                StructureEventType.MSS,
            }
        ][-8:]

        if not relevant:
            return 0.0

        aligned = 0
        for event in relevant:
            if bias == MarketBias.BULLISH:
                if event.event_type in {StructureEventType.HH, StructureEventType.HL}:
                    aligned += 1
                elif event.direction == MarketBias.BULLISH:
                    aligned += 1
            elif bias == MarketBias.BEARISH:
                if event.event_type in {StructureEventType.LH, StructureEventType.LL}:
                    aligned += 1
                elif event.direction == MarketBias.BEARISH:
                    aligned += 1
            elif bias == MarketBias.RANGING:
                if event.event_type in {StructureEventType.CHOCH}:
                    aligned += 1

        return max(0.0, min(1.0, aligned / max(len(relevant), 1)))

    def _compute_layer_trend_strength(self, layer: StructureLayer, bias: MarketBias) -> float:
        if bias in {MarketBias.UNKNOWN, MarketBias.RANGING}:
            return 0.0

        last_bos = self._last_event(layer, StructureEventType.BOS)
        if last_bos is None or last_bos.direction != bias:
            return max(0.0, min(1.0, self._compute_layer_confidence(layer, bias) * 0.7))

        return max(0.0, min(1.0, 0.4 + last_bos.confidence * 0.6))

    def _is_layer_in_recent_breakout(self, layer: StructureLayer) -> bool:
        last_bos = self._last_event(layer, StructureEventType.BOS)
        if last_bos is None or self._state.last_update is None:
            return False

        recent = [
            e for e in self._events
            if e.layer == layer and e.timestamp >= last_bos.timestamp
        ]
        return any(e.event_type == StructureEventType.BOS for e in recent)

    def _compute_alignment(self) -> None:
        mtf = self._state.mtf_alignment
        internal_bias = self._state.internal.bias
        external_bias = self._state.external.bias
        htf_bias = mtf.higher_timeframe_bias

        mtf.internal_with_external_aligned = (
            internal_bias == external_bias
            and internal_bias in {MarketBias.BULLISH, MarketBias.BEARISH}
        )

        mtf.internal_bias_aligned = (
            htf_bias in {MarketBias.BULLISH, MarketBias.BEARISH}
            and internal_bias == htf_bias
        )
        mtf.external_bias_aligned = (
            htf_bias in {MarketBias.BULLISH, MarketBias.BEARISH}
            and external_bias == htf_bias
        )

        score = 0.0

        if mtf.internal_with_external_aligned:
            score += 0.35

        if mtf.internal_bias_aligned:
            score += 0.30

        if mtf.external_bias_aligned:
            score += 0.35

        score *= max(0.5, mtf.higher_timeframe_confidence if htf_bias != MarketBias.UNKNOWN else 1.0)
        mtf.alignment_score = max(0.0, min(1.0, score))

        mtf.metadata = {
            **mtf.metadata,
            "internal_bias": internal_bias.value,
            "external_bias": external_bias.value,
        }

    # -------------------------------------------------------------------------
    # EventBus integration
    # -------------------------------------------------------------------------

    def _register_swing_event(self, swing: SwingPoint) -> None:
        event_type = (
            StructureEventType.SWING_HIGH
            if swing.swing_type == SwingType.HIGH
            else StructureEventType.SWING_LOW
        )

        event = StructureEvent(
            event_id=self._new_id(),
            event_type=event_type,
            timestamp=swing.timestamp,
            price=swing.price,
            layer=swing.layer,
            direction=None,
            swing_id=swing.swing_id,
            reference_price=None,
            reference_swing_id=None,
            confidence=min(1.0, 0.4 + swing.strength),
            metadata={
                "index": swing.index,
                "strength": swing.strength,
            },
        )
        self._append_event(event)

    def _append_event(self, event: StructureEvent) -> None:
        self._events.append(event)
        self._emit_event(event)

    def _emit_event(self, event: StructureEvent) -> None:
        if not self.config.emit_events or self.event_bus is None:
            return

        payload = {
            "source": self.config.event_namespace,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "higher_timeframe": self.higher_timeframe,
            "event": self._event_to_dict(event),
            "state": {
                "internal_bias": self._state.internal.bias.value,
                "external_bias": self._state.external.bias.value,
                "alignment_score": self._state.mtf_alignment.alignment_score,
            },
        }

        event_name = f"{self.config.event_namespace}.{event.event_type.value}"

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

            self.logger.debug(
                "Market structure event emitted",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "event_name": event_name,
                    "layer": event.layer.value,
                },
            )
        except Exception as exc:
            self.logger.exception(
                "Failed to emit market structure event",
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

        payload = {
            "source": self.config.event_namespace,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "snapshot": self.snapshot(),
        }

        event_name = f"{self.config.event_namespace}.snapshot"

        try:
            if hasattr(self.event_bus, "emit"):
                self.event_bus.emit(event_name, payload)
            elif hasattr(self.event_bus, "publish"):
                self.event_bus.publish(event_name, payload)
            elif hasattr(self.event_bus, "dispatch"):
                self.event_bus.dispatch(event_name, payload)
        except Exception as exc:
            self.logger.exception(
                "Failed to publish market structure snapshot",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "event_name": event_name,
                    "error": str(exc),
                },
            )

    # -------------------------------------------------------------------------
    # Parsing / normalization
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _swings_by_layer(self, layer: StructureLayer) -> Deque[SwingPoint]:
        if layer == StructureLayer.INTERNAL:
            return self._internal_swings
        return self._external_swings

    def _layer_state(self, layer: StructureLayer) -> StructureLayerState:
        if layer == StructureLayer.INTERNAL:
            return self._state.internal
        return self._state.external

    def _last_swing(self, layer: StructureLayer, swing_type: SwingType) -> Optional[SwingPoint]:
        swings = self._swings_by_layer(layer)
        for swing in reversed(swings):
            if swing.swing_type == swing_type:
                return swing
        return None

    def _previous_swing(self, layer: StructureLayer, swing_type: SwingType) -> Optional[SwingPoint]:
        swings = self._swings_by_layer(layer)
        found = 0
        for swing in reversed(swings):
            if swing.swing_type == swing_type:
                found += 1
                if found == 2:
                    return swing
        return None

    def _last_event(self, layer: StructureLayer, event_type: StructureEventType) -> Optional[StructureEvent]:
        for event in reversed(self._events):
            if event.layer == layer and event.event_type == event_type:
                return event
        return None

    def _is_break_above(self, candle: Candle, level: float) -> bool:
        threshold = abs(level) * self.config.structure_break_threshold_pct
        if self.config.require_close_break:
            return candle.close > (level + threshold)
        return candle.high > (level + threshold)

    def _is_break_below(self, candle: Candle, level: float) -> bool:
        threshold = abs(level) * self.config.structure_break_threshold_pct
        if self.config.require_close_break:
            return candle.close < (level - threshold)
        return candle.low < (level - threshold)

    @staticmethod
    def _new_id() -> str:
        return uuid4().hex

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------

    def _swing_to_dict(self, swing: Optional[SwingPoint]) -> Optional[Dict[str, Any]]:
        if swing is None:
            return None

        return {
            "swing_id": swing.swing_id,
            "timestamp": swing.timestamp.isoformat(),
            "price": swing.price,
            "swing_type": swing.swing_type.value,
            "layer": swing.layer.value,
            "index": swing.index,
            "candle_open": swing.candle_open,
            "candle_high": swing.candle_high,
            "candle_low": swing.candle_low,
            "candle_close": swing.candle_close,
            "strength": swing.strength,
            "is_confirmed": swing.is_confirmed,
            "metadata": dict(swing.metadata),
        }

    def _event_to_dict(self, event: Optional[StructureEvent]) -> Optional[Dict[str, Any]]:
        if event is None:
            return None

        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "price": event.price,
            "layer": event.layer.value,
            "direction": event.direction.value if event.direction else None,
            "swing_id": event.swing_id,
            "reference_price": event.reference_price,
            "reference_swing_id": event.reference_swing_id,
            "confidence": event.confidence,
            "metadata": dict(event.metadata),
        }

    def _layer_state_to_dict(self, state: StructureLayerState) -> Dict[str, Any]:
        return {
            "layer": state.layer.value,
            "bias": state.bias.value,
            "confidence": state.confidence,
            "trend_strength": state.trend_strength,
            "in_breakout": state.in_breakout,
            "swing_count": state.swing_count,
            "event_count": state.event_count,
            "sequence": list(state.sequence),
            "last_swing_high": self._swing_to_dict(state.last_swing_high),
            "previous_swing_high": self._swing_to_dict(state.previous_swing_high),
            "last_swing_low": self._swing_to_dict(state.last_swing_low),
            "previous_swing_low": self._swing_to_dict(state.previous_swing_low),
            "last_hh": self._event_to_dict(state.last_hh),
            "last_hl": self._event_to_dict(state.last_hl),
            "last_lh": self._event_to_dict(state.last_lh),
            "last_ll": self._event_to_dict(state.last_ll),
            "last_bos": self._event_to_dict(state.last_bos),
            "last_choch": self._event_to_dict(state.last_choch),
            "last_mss": self._event_to_dict(state.last_mss),
        }

    def _alignment_to_dict(self, alignment: MultiTimeframeAlignment) -> Dict[str, Any]:
        return {
            "higher_timeframe": alignment.higher_timeframe,
            "higher_timeframe_bias": alignment.higher_timeframe_bias.value,
            "higher_timeframe_confidence": alignment.higher_timeframe_confidence,
            "internal_bias_aligned": alignment.internal_bias_aligned,
            "external_bias_aligned": alignment.external_bias_aligned,
            "internal_with_external_aligned": alignment.internal_with_external_aligned,
            "alignment_score": alignment.alignment_score,
            "last_updated": alignment.last_updated.isoformat() if alignment.last_updated else None,
            "metadata": dict(alignment.metadata),
        }