from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from core.logger import get_logger

from analytics.price_action.market_structure import StructureLayer, SwingPoint, SwingType


class LevelType(str, Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    FLIP_SUPPORT = "flip_support"
    FLIP_RESISTANCE = "flip_resistance"


class LevelStatus(str, Enum):
    ACTIVE = "active"
    BROKEN = "broken"
    INACTIVE = "inactive"


class SREventType(str, Enum):
    LEVEL_CREATED = "level_created"
    LEVEL_MERGED = "level_merged"
    LEVEL_TOUCHED = "level_touched"
    LEVEL_REJECTED = "level_rejected"
    LEVEL_BROKEN = "level_broken"
    LEVEL_FLIPPED = "level_flipped"
    LEVEL_RETESTED = "level_retested"


@dataclass(slots=True)
class SupportResistanceConfig:
    internal_merge_distance_pct: float = 0.0010
    external_merge_distance_pct: float = 0.0020
    internal_zone_half_width_pct: float = 0.0008
    external_zone_half_width_pct: float = 0.0015

    min_touches_for_validation: int = 2
    breakout_threshold_pct: float = 0.0005
    require_close_break: bool = True
    rejection_wick_ratio_threshold: float = 0.35

    max_candles: int = 3000
    max_levels_per_layer: int = 300
    max_events: int = 1000

    emit_events: bool = True
    event_namespace: str = "price_action.support_resistance"
    publish_snapshots: bool = False

    retest_window_bars: int = 12
    allow_flip_on_break: bool = True
    decay_broken_levels: bool = False

    def validate(self) -> None:
        if self.internal_merge_distance_pct < 0:
            raise ValueError("internal_merge_distance_pct must be >= 0")
        if self.external_merge_distance_pct < 0:
            raise ValueError("external_merge_distance_pct must be >= 0")
        if self.internal_zone_half_width_pct < 0:
            raise ValueError("internal_zone_half_width_pct must be >= 0")
        if self.external_zone_half_width_pct < 0:
            raise ValueError("external_zone_half_width_pct must be >= 0")
        if self.breakout_threshold_pct < 0:
            raise ValueError("breakout_threshold_pct must be >= 0")
        if self.min_touches_for_validation < 1:
            raise ValueError("min_touches_for_validation must be >= 1")
        if self.max_candles < 100:
            raise ValueError("max_candles must be >= 100")
        if self.max_levels_per_layer < 10:
            raise ValueError("max_levels_per_layer must be >= 10")
        if self.max_events < 10:
            raise ValueError("max_events must be >= 10")
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


class SupportResistanceAnalyzer:
    """
    Stateful support / resistance analyzer.

    Features
    --------
    - builds zones from swing highs / lows
    - clusters nearby levels into a shared zone
    - tracks touches / rejections / breakouts / retests / flips
    - separates internal and external structure layers
    - integrates with EventBus
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        event_bus: Optional[Any] = None,
        config: Optional[SupportResistanceConfig] = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.event_bus = event_bus
        self.config = config or SupportResistanceConfig()
        self.config.validate()

        self.logger = get_logger(__name__, service_name="price_action.support_resistance")

        self._candles: Deque[Candle] = deque(maxlen=self.config.max_candles)
        self._internal_levels: Deque[SupportResistanceLevel] = deque(maxlen=self.config.max_levels_per_layer)
        self._external_levels: Deque[SupportResistanceLevel] = deque(maxlen=self.config.max_levels_per_layer)
        self._events: Deque[SupportResistanceEvent] = deque(maxlen=self.config.max_events)

        self._processed_swings: set[str] = set()
        self._processed_touch_keys: set[Tuple[str, int]] = set()
        self._processed_break_keys: set[Tuple[str, int]] = set()
        self._processed_retest_keys: set[Tuple[str, int]] = set()

        self._global_candle_index: int = 0

        self._state = SupportResistanceState(
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        self.logger.info(
            "Initialized SupportResistanceAnalyzer",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "config": asdict(self.config),
            },
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        self._candles.clear()
        self._internal_levels.clear()
        self._external_levels.clear()
        self._events.clear()

        self._processed_swings.clear()
        self._processed_touch_keys.clear()
        self._processed_break_keys.clear()
        self._processed_retest_keys.clear()

        self._global_candle_index = 0

        self._state = SupportResistanceState(
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        self.logger.info(
            "SupportResistanceAnalyzer reset",
            extra={"symbol": self.symbol, "timeframe": self.timeframe},
        )

    def update(
        self,
        *,
        candles: Optional[Sequence[Mapping[str, Any]]] = None,
        swings: Optional[Sequence[SwingPoint | Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return self.add_data(candles=candles, swings=swings)

    def add_data(
        self,
        *,
        candles: Optional[Sequence[Mapping[str, Any]]] = None,
        swings: Optional[Sequence[SwingPoint | Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        new_events: List[SupportResistanceEvent] = []
        updated_levels: List[SupportResistanceLevel] = []

        if swings:
            levels_from_swings, events_from_swings = self._ingest_swings(swings)
            updated_levels.extend(levels_from_swings)
            new_events.extend(events_from_swings)

        if candles:
            events_from_candles = self._ingest_candles(candles)
            new_events.extend(events_from_candles)

        self._refresh_state()

        if self.config.publish_snapshots:
            self._publish_snapshot()

        self.logger.debug(
            "Support/resistance updated",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "updated_levels": len(updated_levels),
                "new_events": len(new_events),
                "last_price": self._state.last_price,
            },
        )

        return {
            "state": self.snapshot(),
            "updated_levels": [self._level_to_dict(level) for level in updated_levels],
            "new_events": [self._event_to_dict(event) for event in new_events],
        }

    def add_candle(self, candle: Mapping[str, Any]) -> Dict[str, Any]:
        return self.add_data(candles=[candle])

    def add_candles(self, candles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return self.add_data(candles=candles)

    def add_swings(self, swings: Sequence[SwingPoint | Mapping[str, Any]]) -> Dict[str, Any]:
        return self.add_data(swings=swings)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "symbol": self._state.symbol,
            "timeframe": self._state.timeframe,
            "last_price": self._state.last_price,
            "last_update": self._state.last_update.isoformat() if self._state.last_update else None,
            "internal": self._layer_state_to_dict(self._state.internal),
            "external": self._layer_state_to_dict(self._state.external),
            "levels": {
                "internal": [self._level_to_dict(x) for x in self._internal_levels],
                "external": [self._level_to_dict(x) for x in self._external_levels],
            },
            "metadata": dict(self._state.metadata),
        }

    def get_state(self) -> SupportResistanceState:
        return self._state

    def get_internal_levels(self) -> List[SupportResistanceLevel]:
        return list(self._internal_levels)

    def get_external_levels(self) -> List[SupportResistanceLevel]:
        return list(self._external_levels)

    def get_events(self) -> List[SupportResistanceEvent]:
        return list(self._events)

    # -------------------------------------------------------------------------
    # Swings -> Levels
    # -------------------------------------------------------------------------

    def _ingest_swings(
        self,
        swings: Sequence[SwingPoint | Mapping[str, Any]],
    ) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceEvent]]:
        updated_levels: List[SupportResistanceLevel] = []
        events: List[SupportResistanceEvent] = []

        for raw in swings:
            swing = self._normalize_swing(raw)
            if swing.swing_id in self._processed_swings:
                continue

            self._processed_swings.add(swing.swing_id)

            level, event = self._register_level_from_swing(swing)
            if level is not None:
                updated_levels.append(level)
            if event is not None:
                events.append(event)

        return updated_levels, events

    def _register_level_from_swing(
        self,
        swing: SwingPoint,
    ) -> Tuple[Optional[SupportResistanceLevel], Optional[SupportResistanceEvent]]:
        level_type = LevelType.RESISTANCE if swing.swing_type == SwingType.HIGH else LevelType.SUPPORT
        existing = self._find_matching_level(
            layer=swing.layer,
            level_type=level_type,
            price=swing.price,
        )

        if existing is None:
            level = self._create_level_from_swing(swing, level_type)
            self._levels_by_layer(swing.layer).append(level)

            event = self._create_event(
                event_type=SREventType.LEVEL_CREATED,
                timestamp=swing.timestamp,
                level=level,
                confidence=min(1.0, 0.45 + swing.strength * 0.5),
                reference_price=swing.price,
                metadata={
                    "source_swing_id": swing.swing_id,
                    "source_swing_type": swing.swing_type.value,
                },
            )
            self._append_event(event)

            return level, event

        self._merge_swing_into_level(existing, swing)
        event = self._create_event(
            event_type=SREventType.LEVEL_MERGED,
            timestamp=swing.timestamp,
            level=existing,
            confidence=min(1.0, 0.50 + existing.strength * 0.4),
            reference_price=swing.price,
            metadata={
                "source_swing_id": swing.swing_id,
                "source_swing_type": swing.swing_type.value,
                "source_count": existing.source_count,
            },
        )
        self._append_event(event)

        return existing, event

    def _create_level_from_swing(
        self,
        swing: SwingPoint,
        level_type: LevelType,
    ) -> SupportResistanceLevel:
        half_width_pct = self._zone_half_width_pct(swing.layer)
        half_width = max(abs(swing.price) * half_width_pct, 1e-12)

        strength = self._base_strength_from_swing(swing)

        return SupportResistanceLevel(
            level_id=self._new_id(),
            layer=swing.layer,
            level_type=level_type,
            price=swing.price,
            upper_bound=swing.price + half_width,
            lower_bound=swing.price - half_width,
            strength=strength,
            created_at=swing.timestamp,
            updated_at=swing.timestamp,
            source_count=1,
            source_swing_ids=[swing.swing_id],
            source_prices=[swing.price],
            metadata={
                "created_from": "swing",
                "source_layer": swing.layer.value,
            },
        )

    def _merge_swing_into_level(self, level: SupportResistanceLevel, swing: SwingPoint) -> None:
        if swing.swing_id in level.source_swing_ids:
            return

        level.source_swing_ids.append(swing.swing_id)
        level.source_prices.append(swing.price)
        level.source_count += 1

        level.price = sum(level.source_prices) / len(level.source_prices)

        half_width_pct = self._zone_half_width_pct(level.layer)
        half_width = max(abs(level.price) * half_width_pct, 1e-12)
        level.upper_bound = level.price + half_width
        level.lower_bound = level.price - half_width

        level.updated_at = swing.timestamp
        level.strength = self._recalculate_level_strength(level)

    # -------------------------------------------------------------------------
    # Candles -> reactions / breaks / retests / flips
    # -------------------------------------------------------------------------

    def _ingest_candles(self, candles: Sequence[Mapping[str, Any]]) -> List[SupportResistanceEvent]:
        events: List[SupportResistanceEvent] = []

        for raw in candles:
            candle = self._parse_candle(raw)
            self._candles.append(candle)
            self._state.last_price = candle.close
            self._state.last_update = candle.timestamp

            for layer in (StructureLayer.INTERNAL, StructureLayer.EXTERNAL):
                for level in list(self._levels_by_layer(layer)):
                    if level.status == LevelStatus.INACTIVE:
                        continue

                    touch_event = self._process_level_touch(level, candle)
                    if touch_event is not None:
                        events.append(touch_event)

                    rejection_event = self._process_level_rejection(level, candle)
                    if rejection_event is not None:
                        events.append(rejection_event)

                    break_event = self._process_level_break(level, candle)
                    if break_event is not None:
                        events.append(break_event)

                    retest_event = self._process_level_retest(level, candle)
                    if retest_event is not None:
                        events.append(retest_event)

        return events

    def _process_level_touch(
        self,
        level: SupportResistanceLevel,
        candle: Candle,
    ) -> Optional[SupportResistanceEvent]:
        if not self._candle_intersects_zone(candle, level):
            return None

        key = (level.level_id, candle.index)
        if key in self._processed_touch_keys:
            return None

        self._processed_touch_keys.add(key)
        level.touch_count += 1
        level.last_tested_at = candle.timestamp
        level.updated_at = candle.timestamp
        level.strength = self._recalculate_level_strength(level)

        event = self._create_event(
            event_type=SREventType.LEVEL_TOUCHED,
            timestamp=candle.timestamp,
            level=level,
            confidence=min(1.0, 0.45 + level.strength * 0.4),
            reference_price=candle.close,
            metadata={
                "candle_index": candle.index,
                "touch_count": level.touch_count,
            },
        )
        self._append_event(event)
        return event

    def _process_level_rejection(
        self,
        level: SupportResistanceLevel,
        candle: Candle,
    ) -> Optional[SupportResistanceEvent]:
        if not self._candle_intersects_zone(candle, level):
            return None

        if candle.range_size <= 0:
            return None

        rejection_detected = False
        direction = None

        upper_wick = candle.high - candle.body_high
        lower_wick = candle.body_low - candle.low
        wick_ratio_upper = upper_wick / max(candle.range_size, 1e-12)
        wick_ratio_lower = lower_wick / max(candle.range_size, 1e-12)

        if level.level_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}:
            if candle.high >= level.lower_bound and candle.close < level.price:
                if wick_ratio_upper >= self.config.rejection_wick_ratio_threshold:
                    rejection_detected = True
                    direction = "bearish_rejection"

        elif level.level_type in {LevelType.SUPPORT, LevelType.FLIP_SUPPORT}:
            if candle.low <= level.upper_bound and candle.close > level.price:
                if wick_ratio_lower >= self.config.rejection_wick_ratio_threshold:
                    rejection_detected = True
                    direction = "bullish_rejection"

        if not rejection_detected:
            return None

        level.rejection_count += 1
        level.last_rejected_at = candle.timestamp
        level.updated_at = candle.timestamp
        level.strength = self._recalculate_level_strength(level)

        event = self._create_event(
            event_type=SREventType.LEVEL_REJECTED,
            timestamp=candle.timestamp,
            level=level,
            confidence=min(1.0, 0.50 + level.strength * 0.45),
            reference_price=candle.close,
            metadata={
                "direction": direction,
                "rejection_count": level.rejection_count,
                "upper_wick_ratio": wick_ratio_upper,
                "lower_wick_ratio": wick_ratio_lower,
                "candle_index": candle.index,
            },
        )
        self._append_event(event)
        return event

    def _process_level_break(
        self,
        level: SupportResistanceLevel,
        candle: Candle,
    ) -> Optional[SupportResistanceEvent]:
        if level.status == LevelStatus.BROKEN and not self.config.allow_flip_on_break:
            return None

        break_detected = False

        threshold = max(abs(level.price) * self.config.breakout_threshold_pct, 1e-12)

        if level.level_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}:
            if self.config.require_close_break:
                break_detected = candle.close > (level.upper_bound + threshold)
            else:
                break_detected = candle.high > (level.upper_bound + threshold)

        elif level.level_type in {LevelType.SUPPORT, LevelType.FLIP_SUPPORT}:
            if self.config.require_close_break:
                break_detected = candle.close < (level.lower_bound - threshold)
            else:
                break_detected = candle.low < (level.lower_bound - threshold)

        if not break_detected:
            return None

        key = (level.level_id, candle.index)
        if key in self._processed_break_keys:
            return None

        self._processed_break_keys.add(key)

        level.break_count += 1
        level.status = LevelStatus.BROKEN
        level.broken_at = candle.timestamp
        level.last_broken_at = candle.timestamp
        level.updated_at = candle.timestamp
        level.strength = self._recalculate_level_strength(level)

        event = self._create_event(
            event_type=SREventType.LEVEL_BROKEN,
            timestamp=candle.timestamp,
            level=level,
            confidence=min(1.0, 0.55 + level.strength * 0.4),
            reference_price=candle.close,
            metadata={
                "break_count": level.break_count,
                "candle_index": candle.index,
                "require_close_break": self.config.require_close_break,
            },
        )
        self._append_event(event)

        if self.config.allow_flip_on_break:
            flip_event = self._flip_level_after_break(level, candle)
            if flip_event is not None:
                return flip_event

        return event

    def _flip_level_after_break(
        self,
        level: SupportResistanceLevel,
        candle: Candle,
    ) -> Optional[SupportResistanceEvent]:
        old_type = level.level_type

        if old_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}:
            level.level_type = LevelType.FLIP_SUPPORT
        elif old_type in {LevelType.SUPPORT, LevelType.FLIP_SUPPORT}:
            level.level_type = LevelType.FLIP_RESISTANCE
        else:
            return None

        level.status = LevelStatus.ACTIVE
        level.flipped_at = candle.timestamp
        level.updated_at = candle.timestamp
        level.strength = self._recalculate_level_strength(level)

        event = self._create_event(
            event_type=SREventType.LEVEL_FLIPPED,
            timestamp=candle.timestamp,
            level=level,
            confidence=min(1.0, 0.55 + level.strength * 0.35),
            reference_price=candle.close,
            metadata={
                "previous_level_type": old_type.value,
                "new_level_type": level.level_type.value,
                "candle_index": candle.index,
            },
        )
        self._append_event(event)
        return event

    def _process_level_retest(
        self,
        level: SupportResistanceLevel,
        candle: Candle,
    ) -> Optional[SupportResistanceEvent]:
        if level.flipped_at is None:
            return None

        if not self._candle_intersects_zone(candle, level):
            return None

        bars_since_flip = candle.index - self._find_candle_index_by_timestamp(level.flipped_at)
        if bars_since_flip < 0 or bars_since_flip > self.config.retest_window_bars:
            return None

        key = (level.level_id, candle.index)
        if key in self._processed_retest_keys:
            return None

        self._processed_retest_keys.add(key)
        level.retest_count += 1
        level.last_retested_at = candle.timestamp
        level.updated_at = candle.timestamp
        level.strength = self._recalculate_level_strength(level)

        event = self._create_event(
            event_type=SREventType.LEVEL_RETESTED,
            timestamp=candle.timestamp,
            level=level,
            confidence=min(1.0, 0.50 + level.strength * 0.4),
            reference_price=candle.close,
            metadata={
                "retest_count": level.retest_count,
                "bars_since_flip": bars_since_flip,
                "candle_index": candle.index,
            },
        )
        self._append_event(event)
        return event

    # -------------------------------------------------------------------------
    # State
    # -------------------------------------------------------------------------

    def _refresh_state(self) -> None:
        self._refresh_layer_state(StructureLayer.INTERNAL)
        self._refresh_layer_state(StructureLayer.EXTERNAL)

        self._state.metadata = {
            "internal_levels_total": len(self._internal_levels),
            "external_levels_total": len(self._external_levels),
            "events_total": len(self._events),
        }

    def _refresh_layer_state(self, layer: StructureLayer) -> None:
        layer_state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        levels = [x for x in self._levels_by_layer(layer) if x.status != LevelStatus.INACTIVE]
        active = [x for x in levels if x.status == LevelStatus.ACTIVE]

        layer_state.total_levels = len(levels)
        layer_state.active_supports = len([x for x in active if x.level_type == LevelType.SUPPORT])
        layer_state.active_resistances = len([x for x in active if x.level_type == LevelType.RESISTANCE])
        layer_state.active_flip_supports = len([x for x in active if x.level_type == LevelType.FLIP_SUPPORT])
        layer_state.active_flip_resistances = len([x for x in active if x.level_type == LevelType.FLIP_RESISTANCE])

        support_candidates = [x for x in active if x.level_type in {LevelType.SUPPORT, LevelType.FLIP_SUPPORT}]
        resistance_candidates = [x for x in active if x.level_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}]

        layer_state.strongest_support = self._strongest_level(support_candidates)
        layer_state.strongest_resistance = self._strongest_level(resistance_candidates)

        layer_state.nearest_support = self._nearest_support(support_candidates, self._state.last_price)
        layer_state.nearest_resistance = self._nearest_resistance(resistance_candidates, self._state.last_price)

        layer_events = [e for e in self._events if e.layer == layer]
        layer_state.last_event = layer_events[-1] if layer_events else None

        layer_state.metadata = {
            "validated_levels": len([x for x in levels if x.touch_count >= self.config.min_touches_for_validation]),
        }

    # -------------------------------------------------------------------------
    # EventBus
    # -------------------------------------------------------------------------

    def _append_event(self, event: SupportResistanceEvent) -> None:
        self._events.append(event)
        self._emit_event(event)

    def _emit_event(self, event: SupportResistanceEvent) -> None:
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

            self.logger.debug(
                "Support/resistance event emitted",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "event_name": event_name,
                    "level_id": event.level_id,
                },
            )
        except Exception as exc:
            self.logger.exception(
                "Failed to emit support/resistance event",
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
                "Failed to publish support/resistance snapshot",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "event_name": event_name,
                    "error": str(exc),
                },
            )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _find_matching_level(
        self,
        *,
        layer: StructureLayer,
        level_type: LevelType,
        price: float,
    ) -> Optional[SupportResistanceLevel]:
        levels = self._levels_by_layer(layer)
        merge_distance = max(abs(price) * self._merge_distance_pct(layer), 1e-12)

        candidates = [
            level for level in levels
            if level.level_type == level_type and level.status != LevelStatus.INACTIVE
        ]

        if not candidates:
            return None

        candidates.sort(key=lambda x: abs(x.price - price))
        best = candidates[0]

        if abs(best.price - price) <= merge_distance:
            return best
        return None

    def _levels_by_layer(self, layer: StructureLayer) -> Deque[SupportResistanceLevel]:
        return self._internal_levels if layer == StructureLayer.INTERNAL else self._external_levels

    def _merge_distance_pct(self, layer: StructureLayer) -> float:
        return (
            self.config.internal_merge_distance_pct
            if layer == StructureLayer.INTERNAL
            else self.config.external_merge_distance_pct
        )

    def _zone_half_width_pct(self, layer: StructureLayer) -> float:
        return (
            self.config.internal_zone_half_width_pct
            if layer == StructureLayer.INTERNAL
            else self.config.external_zone_half_width_pct
        )

    def _base_strength_from_swing(self, swing: SwingPoint) -> float:
        base = 0.30 + swing.strength * 0.55
        if swing.layer == StructureLayer.EXTERNAL:
            base += 0.10
        return max(0.0, min(1.0, base))

    def _recalculate_level_strength(self, level: SupportResistanceLevel) -> float:
        score = 0.25
        score += min(level.source_count, 5) * 0.08
        score += min(level.touch_count, 6) * 0.06
        score += min(level.rejection_count, 6) * 0.08
        score += min(level.retest_count, 4) * 0.05

        if level.layer == StructureLayer.EXTERNAL:
            score += 0.10

        if level.touch_count >= self.config.min_touches_for_validation:
            score += 0.10

        if level.break_count > 0:
            score -= min(level.break_count, 3) * 0.08

        return max(0.0, min(1.0, score))

    def _candle_intersects_zone(self, candle: Candle, level: SupportResistanceLevel) -> bool:
        return not (candle.high < level.lower_bound or candle.low > level.upper_bound)

    def _strongest_level(
        self,
        levels: Sequence[SupportResistanceLevel],
    ) -> Optional[SupportResistanceLevel]:
        if not levels:
            return None
        return max(levels, key=lambda x: x.strength)

    def _nearest_support(
        self,
        levels: Sequence[SupportResistanceLevel],
        price: Optional[float],
    ) -> Optional[SupportResistanceLevel]:
        if price is None:
            return None

        candidates = [x for x in levels if x.price <= price]
        if not candidates:
            return None
        return min(candidates, key=lambda x: abs(price - x.price))

    def _nearest_resistance(
        self,
        levels: Sequence[SupportResistanceLevel],
        price: Optional[float],
    ) -> Optional[SupportResistanceLevel]:
        if price is None:
            return None

        candidates = [x for x in levels if x.price >= price]
        if not candidates:
            return None
        return min(candidates, key=lambda x: abs(price - x.price))

    def _find_candle_index_by_timestamp(self, ts: datetime) -> int:
        for candle in reversed(self._candles):
            if candle.timestamp == ts:
                return candle.index
        return -1

    def _create_event(
        self,
        *,
        event_type: SREventType,
        timestamp: datetime,
        level: SupportResistanceLevel,
        confidence: float,
        reference_price: Optional[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SupportResistanceEvent:
        return SupportResistanceEvent(
            event_id=self._new_id(),
            event_type=event_type,
            timestamp=timestamp,
            symbol=self.symbol,
            timeframe=self.timeframe,
            layer=level.layer,
            level_id=level.level_id,
            level_type=level.level_type,
            price=level.price,
            confidence=max(0.0, min(1.0, confidence)),
            reference_price=reference_price,
            metadata=metadata or {},
        )

    @staticmethod
    def _new_id() -> str:
        return uuid4().hex

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

    def _normalize_swing(self, raw: SwingPoint | Mapping[str, Any]) -> SwingPoint:
        if isinstance(raw, SwingPoint):
            return raw

        timestamp = raw.get("timestamp")
        if timestamp is None:
            raise ValueError("Swing mapping must contain timestamp")

        swing_type_raw = raw.get("swing_type")
        if swing_type_raw is None:
            raise ValueError("Swing mapping must contain swing_type")

        layer_raw = raw.get("layer")
        if layer_raw is None:
            raise ValueError("Swing mapping must contain layer")

        try:
            swing_type = swing_type_raw if isinstance(swing_type_raw, SwingType) else SwingType(str(swing_type_raw))
        except ValueError as exc:
            raise ValueError(f"Invalid swing_type: {swing_type_raw}") from exc

        try:
            layer = layer_raw if isinstance(layer_raw, StructureLayer) else StructureLayer(str(layer_raw))
        except ValueError as exc:
            raise ValueError(f"Invalid layer: {layer_raw}") from exc

        return SwingPoint(
            swing_id=str(raw.get("swing_id", self._new_id())),
            timestamp=self._normalize_timestamp(timestamp),
            price=float(raw["price"]),
            swing_type=swing_type,
            layer=layer,
            index=int(raw.get("index", 0)),
            candle_open=float(raw.get("candle_open", raw.get("price", 0.0))),
            candle_high=float(raw.get("candle_high", raw.get("price", 0.0))),
            candle_low=float(raw.get("candle_low", raw.get("price", 0.0))),
            candle_close=float(raw.get("candle_close", raw.get("price", 0.0))),
            strength=float(raw.get("strength", 0.0)),
            is_confirmed=bool(raw.get("is_confirmed", True)),
            metadata=dict(raw.get("metadata", {})),
        )

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
    # Serialization
    # -------------------------------------------------------------------------

    def _level_to_dict(self, level: Optional[SupportResistanceLevel]) -> Optional[Dict[str, Any]]:
        if level is None:
            return None

        return {
            "level_id": level.level_id,
            "layer": level.layer.value,
            "level_type": level.level_type.value,
            "status": level.status.value,
            "price": level.price,
            "upper_bound": level.upper_bound,
            "lower_bound": level.lower_bound,
            "strength": level.strength,
            "created_at": level.created_at.isoformat() if level.created_at else None,
            "updated_at": level.updated_at.isoformat() if level.updated_at else None,
            "broken_at": level.broken_at.isoformat() if level.broken_at else None,
            "flipped_at": level.flipped_at.isoformat() if level.flipped_at else None,
            "last_tested_at": level.last_tested_at.isoformat() if level.last_tested_at else None,
            "last_rejected_at": level.last_rejected_at.isoformat() if level.last_rejected_at else None,
            "last_broken_at": level.last_broken_at.isoformat() if level.last_broken_at else None,
            "last_retested_at": level.last_retested_at.isoformat() if level.last_retested_at else None,
            "touch_count": level.touch_count,
            "rejection_count": level.rejection_count,
            "break_count": level.break_count,
            "retest_count": level.retest_count,
            "source_count": level.source_count,
            "source_swing_ids": list(level.source_swing_ids),
            "source_prices": list(level.source_prices),
            "metadata": dict(level.metadata),
        }

    def _event_to_dict(self, event: Optional[SupportResistanceEvent]) -> Optional[Dict[str, Any]]:
        if event is None:
            return None

        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "symbol": event.symbol,
            "timeframe": event.timeframe,
            "layer": event.layer.value,
            "level_id": event.level_id,
            "level_type": event.level_type.value,
            "price": event.price,
            "confidence": event.confidence,
            "reference_price": event.reference_price,
            "metadata": dict(event.metadata),
        }

    def _layer_state_to_dict(self, state: LayerSRState) -> Dict[str, Any]:
        return {
            "layer": state.layer.value,
            "total_levels": state.total_levels,
            "active_supports": state.active_supports,
            "active_resistances": state.active_resistances,
            "active_flip_supports": state.active_flip_supports,
            "active_flip_resistances": state.active_flip_resistances,
            "strongest_support": self._level_to_dict(state.strongest_support),
            "strongest_resistance": self._level_to_dict(state.strongest_resistance),
            "nearest_support": self._level_to_dict(state.nearest_support),
            "nearest_resistance": self._level_to_dict(state.nearest_resistance),
            "last_event": self._event_to_dict(state.last_event),
            "metadata": dict(state.metadata),
        }