from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from core.logger import get_logger

from analytics.price_action.market_structure import StructureLayer, SwingPoint, SwingType


class LiquidityLevelType(str, Enum):
    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"
    BUY_SIDE_LIQUIDITY = "buy_side_liquidity"
    SELL_SIDE_LIQUIDITY = "sell_side_liquidity"
    SWING_HIGH_LIQUIDITY = "swing_high_liquidity"
    SWING_LOW_LIQUIDITY = "swing_low_liquidity"


class LiquidityLevelStatus(str, Enum):
    ACTIVE = "active"
    SWEPT = "swept"
    RECLAIMED = "reclaimed"
    INVALIDATED = "invalidated"


class LiquidityEventType(str, Enum):
    LEVEL_CREATED = "level_created"
    LEVEL_MERGED = "level_merged"
    LIQUIDITY_TOUCHED = "liquidity_touched"
    LIQUIDITY_SWEPT = "liquidity_swept"
    LIQUIDITY_RECLAIMED = "liquidity_reclaimed"
    FAILED_BREAKOUT = "failed_breakout"
    STOP_RUN = "stop_run"
    LIQUIDITY_INVALIDATED = "liquidity_invalidated"


@dataclass(slots=True)
class LiquidityLevelsConfig:
    max_candles: int = 3000
    max_levels_per_layer: int = 500
    max_events: int = 1000

    equal_level_tolerance_pct_internal: float = 0.0008
    equal_level_tolerance_pct_external: float = 0.0015
    swing_liquidity_zone_width_pct_internal: float = 0.0007
    swing_liquidity_zone_width_pct_external: float = 0.0012

    min_cluster_size_for_equal_levels: int = 2
    min_sweep_penetration_pct: float = 0.00035
    reclaim_close_buffer_pct: float = 0.00015
    require_close_reclaim: bool = True

    retest_window_bars: int = 8
    stop_run_wick_ratio_threshold: float = 0.45
    failed_breakout_reclaim_window_bars: int = 3

    emit_events: bool = True
    event_namespace: str = "price_action.liquidity_levels"
    publish_snapshots: bool = False

    def validate(self) -> None:
        if self.max_candles < 100:
            raise ValueError("max_candles must be >= 100")
        if self.max_levels_per_layer < 20:
            raise ValueError("max_levels_per_layer must be >= 20")
        if self.max_events < 50:
            raise ValueError("max_events must be >= 50")
        if self.equal_level_tolerance_pct_internal < 0:
            raise ValueError("equal_level_tolerance_pct_internal must be >= 0")
        if self.equal_level_tolerance_pct_external < 0:
            raise ValueError("equal_level_tolerance_pct_external must be >= 0")
        if self.swing_liquidity_zone_width_pct_internal < 0:
            raise ValueError("swing_liquidity_zone_width_pct_internal must be >= 0")
        if self.swing_liquidity_zone_width_pct_external < 0:
            raise ValueError("swing_liquidity_zone_width_pct_external must be >= 0")
        if self.min_cluster_size_for_equal_levels < 2:
            raise ValueError("min_cluster_size_for_equal_levels must be >= 2")
        if self.min_sweep_penetration_pct < 0:
            raise ValueError("min_sweep_penetration_pct must be >= 0")
        if self.reclaim_close_buffer_pct < 0:
            raise ValueError("reclaim_close_buffer_pct must be >= 0")
        if self.retest_window_bars < 1:
            raise ValueError("retest_window_bars must be >= 1")
        if self.failed_breakout_reclaim_window_bars < 1:
            raise ValueError("failed_breakout_reclaim_window_bars must be >= 1")


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


class LiquidityLevelsAnalyzer:
    """
    Stateful liquidity levels analyzer.

    Features
    --------
    - builds liquidity pools from swing highs/lows
    - detects equal highs / equal lows clusters
    - tracks touches, sweeps, reclaims, failed breakouts, stop runs
    - supports internal / external layers
    - integrates with EventBus
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        event_bus: Optional[Any] = None,
        config: Optional[LiquidityLevelsConfig] = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.event_bus = event_bus
        self.config = config or LiquidityLevelsConfig()
        self.config.validate()

        self.logger = get_logger(__name__, service_name="price_action.liquidity_levels")

        self._candles: Deque[Candle] = deque(maxlen=self.config.max_candles)
        self._internal_levels: Deque[LiquidityLevel] = deque(maxlen=self.config.max_levels_per_layer)
        self._external_levels: Deque[LiquidityLevel] = deque(maxlen=self.config.max_levels_per_layer)
        self._events: Deque[LiquidityEvent] = deque(maxlen=self.config.max_events)

        self._processed_swings: set[str] = set()
        self._processed_touch_keys: set[Tuple[str, int]] = set()
        self._processed_sweep_keys: set[Tuple[str, int]] = set()
        self._processed_reclaim_keys: set[Tuple[str, int]] = set()
        self._processed_failed_breakout_keys: set[Tuple[str, int]] = set()

        self._global_candle_index: int = 0

        self._state = LiquidityState(symbol=self.symbol, timeframe=self.timeframe)

        self.logger.info(
            "Initialized LiquidityLevelsAnalyzer",
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
        self._internal_levels.clear()
        self._external_levels.clear()
        self._events.clear()

        self._processed_swings.clear()
        self._processed_touch_keys.clear()
        self._processed_sweep_keys.clear()
        self._processed_reclaim_keys.clear()
        self._processed_failed_breakout_keys.clear()

        self._global_candle_index = 0
        self._state = LiquidityState(symbol=self.symbol, timeframe=self.timeframe)

        self.logger.info(
            "LiquidityLevelsAnalyzer reset",
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
        updated_levels: List[LiquidityLevel] = []
        new_events: List[LiquidityEvent] = []

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
            "Liquidity levels updated",
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

    def get_state(self) -> LiquidityState:
        return self._state

    def get_internal_levels(self) -> List[LiquidityLevel]:
        return list(self._internal_levels)

    def get_external_levels(self) -> List[LiquidityLevel]:
        return list(self._external_levels)

    def get_events(self) -> List[LiquidityEvent]:
        return list(self._events)

    # ------------------------------------------------------------------
    # Swings -> Liquidity levels
    # ------------------------------------------------------------------

    def _ingest_swings(
        self,
        swings: Sequence[SwingPoint | Mapping[str, Any]],
    ) -> Tuple[List[LiquidityLevel], List[LiquidityEvent]]:
        updated_levels: List[LiquidityLevel] = []
        events: List[LiquidityEvent] = []

        for raw in swings:
            swing = self._normalize_swing(raw)
            if swing.swing_id in self._processed_swings:
                continue

            self._processed_swings.add(swing.swing_id)

            levels_for_swing, events_for_swing = self._register_liquidity_from_swing(swing)
            updated_levels.extend(levels_for_swing)
            events.extend(events_for_swing)

        return updated_levels, events

    def _register_liquidity_from_swing(
        self,
        swing: SwingPoint,
    ) -> Tuple[List[LiquidityLevel], List[LiquidityEvent]]:
        levels: List[LiquidityLevel] = []
        events: List[LiquidityEvent] = []

        # 1. Завжди створюємо/апдейтимо swing liquidity
        swing_level_type = (
            LiquidityLevelType.SWING_HIGH_LIQUIDITY
            if swing.swing_type == SwingType.HIGH
            else LiquidityLevelType.SWING_LOW_LIQUIDITY
        )
        swing_side_type = (
            LiquidityLevelType.BUY_SIDE_LIQUIDITY
            if swing.swing_type == SwingType.HIGH
            else LiquidityLevelType.SELL_SIDE_LIQUIDITY
        )

        level_1, event_1 = self._create_or_merge_level(
            swing=swing,
            level_type=swing_level_type,
        )
        if level_1 is not None:
            levels.append(level_1)
        if event_1 is not None:
            events.append(event_1)

        level_2, event_2 = self._create_or_merge_level(
            swing=swing,
            level_type=swing_side_type,
        )
        if level_2 is not None:
            levels.append(level_2)
        if event_2 is not None:
            events.append(event_2)

        # 2. Якщо є кластер однакових high/low - робимо equal highs/lows
        equal_type = (
            LiquidityLevelType.EQUAL_HIGHS
            if swing.swing_type == SwingType.HIGH
            else LiquidityLevelType.EQUAL_LOWS
        )
        equal_candidate = self._find_equal_level_cluster_candidate(swing, equal_type)
        if equal_candidate is not None:
            self._merge_swing_into_level(equal_candidate, swing)
            equal_event = self._create_event(
                event_type=SREventTypeAdapter.level_merged(),
                timestamp=swing.timestamp,
                level=equal_candidate,
                confidence=min(1.0, 0.55 + equal_candidate.strength * 0.35),
                reference_price=swing.price,
                metadata={
                    "cluster_type": equal_type.value,
                    "source_swing_id": swing.swing_id,
                    "cluster_size": equal_candidate.source_count,
                },
            )
            # Перепишемо event_type на liquidity namespace
            equal_event.event_type = LiquidityEventType.LEVEL_MERGED
            self._append_event(equal_event)
            events.append(equal_event)
            levels.append(equal_candidate)

            if equal_candidate.source_count >= self.config.min_cluster_size_for_equal_levels:
                equal_candidate.level_type = equal_type

        else:
            seed = self._create_level_from_swing(
                swing=swing,
                level_type=equal_type,
                seed_only=True,
            )
            self._levels_by_layer(swing.layer).append(seed)
            created_event = self._create_event(
                event_type=LiquidityEventType.LEVEL_CREATED,
                timestamp=swing.timestamp,
                level=seed,
                confidence=min(1.0, 0.40 + swing.strength * 0.35),
                reference_price=swing.price,
                metadata={
                    "cluster_type": equal_type.value,
                    "source_swing_id": swing.swing_id,
                    "seed_only": True,
                },
            )
            self._append_event(created_event)
            events.append(created_event)
            levels.append(seed)

        return levels, events

    def _create_or_merge_level(
        self,
        *,
        swing: SwingPoint,
        level_type: LiquidityLevelType,
    ) -> Tuple[Optional[LiquidityLevel], Optional[LiquidityEvent]]:
        existing = self._find_matching_level(
            layer=swing.layer,
            level_type=level_type,
            price=swing.price,
        )

        if existing is None:
            level = self._create_level_from_swing(
                swing=swing,
                level_type=level_type,
                seed_only=False,
            )
            self._levels_by_layer(swing.layer).append(level)

            event = self._create_event(
                event_type=LiquidityEventType.LEVEL_CREATED,
                timestamp=swing.timestamp,
                level=level,
                confidence=min(1.0, 0.45 + swing.strength * 0.45),
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
            event_type=LiquidityEventType.LEVEL_MERGED,
            timestamp=swing.timestamp,
            level=existing,
            confidence=min(1.0, 0.50 + existing.strength * 0.35),
            reference_price=swing.price,
            metadata={
                "source_swing_id": swing.swing_id,
                "source_count": existing.source_count,
            },
        )
        self._append_event(event)
        return existing, event

    def _create_level_from_swing(
        self,
        *,
        swing: SwingPoint,
        level_type: LiquidityLevelType,
        seed_only: bool,
    ) -> LiquidityLevel:
        zone_width_pct = self._zone_width_pct(swing.layer)
        half_width = max(abs(swing.price) * zone_width_pct, 1e-12)

        base_strength = self._base_strength_from_swing(swing)
        if level_type in {LiquidityLevelType.EQUAL_HIGHS, LiquidityLevelType.EQUAL_LOWS}:
            base_strength *= 0.90

        return LiquidityLevel(
            level_id=self._new_id(),
            layer=swing.layer,
            level_type=level_type,
            price=swing.price,
            upper_bound=swing.price + half_width,
            lower_bound=swing.price - half_width,
            strength=max(0.0, min(1.0, base_strength)),
            created_at=swing.timestamp,
            updated_at=swing.timestamp,
            source_swing_ids=[swing.swing_id],
            source_prices=[swing.price],
            source_count=1,
            metadata={
                "created_from": "swing",
                "source_layer": swing.layer.value,
                "seed_only": seed_only,
            },
        )

    def _merge_swing_into_level(self, level: LiquidityLevel, swing: SwingPoint) -> None:
        if swing.swing_id in level.source_swing_ids:
            return

        level.source_swing_ids.append(swing.swing_id)
        level.source_prices.append(swing.price)
        level.source_count += 1

        level.price = self._safe_mean(level.source_prices)
        half_width = max(abs(level.price) * self._zone_width_pct(level.layer), 1e-12)
        level.upper_bound = level.price + half_width
        level.lower_bound = level.price - half_width
        level.updated_at = swing.timestamp
        level.strength = self._recalculate_level_strength(level)

        if level.level_type == LiquidityLevelType.EQUAL_HIGHS:
            if level.source_count >= self.config.min_cluster_size_for_equal_levels:
                level.metadata["validated_equal_cluster"] = True
        elif level.level_type == LiquidityLevelType.EQUAL_LOWS:
            if level.source_count >= self.config.min_cluster_size_for_equal_levels:
                level.metadata["validated_equal_cluster"] = True

    # ------------------------------------------------------------------
    # Candles -> touch / sweep / reclaim / failed breakout / stop run
    # ------------------------------------------------------------------

    def _ingest_candles(self, candles: Sequence[Mapping[str, Any]]) -> List[LiquidityEvent]:
        events: List[LiquidityEvent] = []

        for raw in candles:
            candle = self._parse_candle(raw)
            self._candles.append(candle)
            self._state.last_price = candle.close
            self._state.last_update = candle.timestamp

            for layer in (StructureLayer.INTERNAL, StructureLayer.EXTERNAL):
                for level in list(self._levels_by_layer(layer)):
                    if level.status == LiquidityLevelStatus.INVALIDATED:
                        continue

                    touch = self._process_touch(level, candle)
                    if touch is not None:
                        events.append(touch)

                    sweep = self._process_sweep(level, candle)
                    if sweep is not None:
                        events.append(sweep)

                    reclaim = self._process_reclaim(level, candle)
                    if reclaim is not None:
                        events.append(reclaim)

                    failed_breakout = self._process_failed_breakout(level, candle)
                    if failed_breakout is not None:
                        events.append(failed_breakout)

        return events

    def _process_touch(self, level: LiquidityLevel, candle: Candle) -> Optional[LiquidityEvent]:
        if not self._candle_intersects_zone(candle, level):
            return None

        key = (level.level_id, candle.index)
        if key in self._processed_touch_keys:
            return None

        self._processed_touch_keys.add(key)

        level.touch_count += 1
        level.last_touched_at = candle.timestamp
        level.updated_at = candle.timestamp
        level.strength = self._recalculate_level_strength(level)

        event = self._create_event(
            event_type=LiquidityEventType.LIQUIDITY_TOUCHED,
            timestamp=candle.timestamp,
            level=level,
            confidence=min(1.0, 0.40 + level.strength * 0.35),
            reference_price=candle.close,
            metadata={
                "touch_count": level.touch_count,
                "candle_index": candle.index,
            },
        )
        self._append_event(event)
        return event

    def _process_sweep(self, level: LiquidityLevel, candle: Candle) -> Optional[LiquidityEvent]:
        if level.status == LiquidityLevelStatus.SWEPT:
            return None

        sweep_side = self._detect_sweep_side(level, candle)
        if sweep_side is None:
            return None

        key = (level.level_id, candle.index)
        if key in self._processed_sweep_keys:
            return None

        self._processed_sweep_keys.add(key)

        level.status = LiquidityLevelStatus.SWEPT
        level.sweep_count += 1
        level.swept_at = candle.timestamp
        level.updated_at = candle.timestamp
        level.last_sweep_side = sweep_side
        level.last_sweep_price = candle.high if sweep_side == "up" else candle.low
        level.last_sweep_index = candle.index
        level.strength = self._recalculate_level_strength(level)

        metadata = {
            "sweep_side": sweep_side,
            "candle_index": candle.index,
            "sweep_price": level.last_sweep_price,
        }

        event_type = LiquidityEventType.LIQUIDITY_SWEPT
        if self._is_stop_run(level, candle, sweep_side):
            event_type = LiquidityEventType.STOP_RUN
            metadata["stop_run"] = True

        event = self._create_event(
            event_type=event_type,
            timestamp=candle.timestamp,
            level=level,
            confidence=min(1.0, 0.55 + level.strength * 0.35),
            reference_price=candle.close,
            metadata=metadata,
        )
        self._append_event(event)
        return event

    def _process_reclaim(self, level: LiquidityLevel, candle: Candle) -> Optional[LiquidityEvent]:
        if level.status != LiquidityLevelStatus.SWEPT:
            return None
        if level.last_sweep_index is None:
            return None

        bars_since_sweep = candle.index - level.last_sweep_index
        if bars_since_sweep < 0 or bars_since_sweep > self.config.retest_window_bars:
            return None

        reclaim_detected = self._is_reclaim(level, candle)
        if not reclaim_detected:
            return None

        key = (level.level_id, candle.index)
        if key in self._processed_reclaim_keys:
            return None

        self._processed_reclaim_keys.add(key)

        level.status = LiquidityLevelStatus.RECLAIMED
        level.reclaim_count += 1
        level.reclaimed_at = candle.timestamp
        level.updated_at = candle.timestamp
        level.strength = self._recalculate_level_strength(level)

        event = self._create_event(
            event_type=LiquidityEventType.LIQUIDITY_RECLAIMED,
            timestamp=candle.timestamp,
            level=level,
            confidence=min(1.0, 0.60 + level.strength * 0.30),
            reference_price=candle.close,
            metadata={
                "bars_since_sweep": bars_since_sweep,
                "last_sweep_side": level.last_sweep_side,
                "candle_index": candle.index,
            },
        )
        self._append_event(event)
        return event

    def _process_failed_breakout(self, level: LiquidityLevel, candle: Candle) -> Optional[LiquidityEvent]:
        if level.last_sweep_index is None:
            return None

        bars_since_sweep = candle.index - level.last_sweep_index
        if bars_since_sweep < 0 or bars_since_sweep > self.config.failed_breakout_reclaim_window_bars:
            return None

        if not self._is_failed_breakout(level, candle):
            return None

        key = (level.level_id, candle.index)
        if key in self._processed_failed_breakout_keys:
            return None

        self._processed_failed_breakout_keys.add(key)

        event = self._create_event(
            event_type=LiquidityEventType.FAILED_BREAKOUT,
            timestamp=candle.timestamp,
            level=level,
            confidence=min(1.0, 0.62 + level.strength * 0.28),
            reference_price=candle.close,
            metadata={
                "bars_since_sweep": bars_since_sweep,
                "last_sweep_side": level.last_sweep_side,
                "candle_index": candle.index,
            },
        )
        self._append_event(event)
        return event

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------

    def _detect_sweep_side(self, level: LiquidityLevel, candle: Candle) -> Optional[str]:
        penetration_threshold = max(abs(level.price) * self.config.min_sweep_penetration_pct, 1e-12)

        if level.level_type in {
            LiquidityLevelType.EQUAL_HIGHS,
            LiquidityLevelType.BUY_SIDE_LIQUIDITY,
            LiquidityLevelType.SWING_HIGH_LIQUIDITY,
        }:
            if candle.high > (level.upper_bound + penetration_threshold):
                return "up"

        if level.level_type in {
            LiquidityLevelType.EQUAL_LOWS,
            LiquidityLevelType.SELL_SIDE_LIQUIDITY,
            LiquidityLevelType.SWING_LOW_LIQUIDITY,
        }:
            if candle.low < (level.lower_bound - penetration_threshold):
                return "down"

        return None

    def _is_reclaim(self, level: LiquidityLevel, candle: Candle) -> bool:
        buffer_size = max(abs(level.price) * self.config.reclaim_close_buffer_pct, 1e-12)

        if level.last_sweep_side == "up":
            reclaim_line = level.upper_bound - buffer_size
            if self.config.require_close_reclaim:
                return candle.close < reclaim_line
            return candle.low < reclaim_line

        if level.last_sweep_side == "down":
            reclaim_line = level.lower_bound + buffer_size
            if self.config.require_close_reclaim:
                return candle.close > reclaim_line
            return candle.high > reclaim_line

        return False

    def _is_failed_breakout(self, level: LiquidityLevel, candle: Candle) -> bool:
        if level.last_sweep_side == "up":
            return candle.close < level.price
        if level.last_sweep_side == "down":
            return candle.close > level.price
        return False

    def _is_stop_run(self, level: LiquidityLevel, candle: Candle, sweep_side: str) -> bool:
        if candle.range_size <= 0:
            return False

        if sweep_side == "up":
            return (
                candle.upper_wick_ratio >= self.config.stop_run_wick_ratio_threshold
                and candle.close < candle.high
            )

        if sweep_side == "down":
            return (
                candle.lower_wick_ratio >= self.config.stop_run_wick_ratio_threshold
                and candle.close > candle.low
            )

        return False

    # ------------------------------------------------------------------
    # State refresh
    # ------------------------------------------------------------------

    def _refresh_state(self) -> None:
        self._refresh_layer_state(StructureLayer.INTERNAL)
        self._refresh_layer_state(StructureLayer.EXTERNAL)

        self._state.metadata = {
            "internal_levels_total": len(self._internal_levels),
            "external_levels_total": len(self._external_levels),
            "events_total": len(self._events),
        }

    def _refresh_layer_state(self, layer: StructureLayer) -> None:
        state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        levels = [x for x in self._levels_by_layer(layer) if x.status != LiquidityLevelStatus.INVALIDATED]
        active = [x for x in levels if x.status == LiquidityLevelStatus.ACTIVE]
        swept = [x for x in levels if x.status == LiquidityLevelStatus.SWEPT]
        reclaimed = [x for x in levels if x.status == LiquidityLevelStatus.RECLAIMED]

        buy_side = [
            x for x in levels
            if x.level_type in {
                LiquidityLevelType.BUY_SIDE_LIQUIDITY,
                LiquidityLevelType.EQUAL_HIGHS,
                LiquidityLevelType.SWING_HIGH_LIQUIDITY,
            }
        ]
        sell_side = [
            x for x in levels
            if x.level_type in {
                LiquidityLevelType.SELL_SIDE_LIQUIDITY,
                LiquidityLevelType.EQUAL_LOWS,
                LiquidityLevelType.SWING_LOW_LIQUIDITY,
            }
        ]

        state.total_levels = len(levels)
        state.active_levels = len(active)
        state.swept_levels = len(swept)
        state.reclaimed_levels = len(reclaimed)
        state.nearest_buy_side = self._nearest_above(buy_side, self._state.last_price)
        state.nearest_sell_side = self._nearest_below(sell_side, self._state.last_price)
        state.strongest_buy_side = self._strongest_level(buy_side)
        state.strongest_sell_side = self._strongest_level(sell_side)

        recent_events = [
            e for e in self._events
            if e.layer == layer and e.event_type in {
                LiquidityEventType.LIQUIDITY_SWEPT,
                LiquidityEventType.STOP_RUN,
            }
        ]
        state.recent_sweep_count = len(recent_events[-10:])
        layer_events = [e for e in self._events if e.layer == layer]
        state.last_event = layer_events[-1] if layer_events else None
        state.metadata = {
            "equal_high_clusters": len([x for x in levels if x.level_type == LiquidityLevelType.EQUAL_HIGHS]),
            "equal_low_clusters": len([x for x in levels if x.level_type == LiquidityLevelType.EQUAL_LOWS]),
        }

    # ------------------------------------------------------------------
    # Helper functions for strategy layer
    # ------------------------------------------------------------------

    def buy_side_liquidity_score(self, layer: StructureLayer) -> float:
        state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        level = state.strongest_buy_side
        if level is None:
            return 0.0
        return max(0.0, min(1.0, level.strength))

    def sell_side_liquidity_score(self, layer: StructureLayer) -> float:
        state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        level = state.strongest_sell_side
        if level is None:
            return 0.0
        return max(0.0, min(1.0, level.strength))

    def sweep_pressure_score(self, layer: StructureLayer) -> float:
        """
        Високий score = багато sweep activity останнім часом.
        Це корисно для mean-reversion / trap / reversal logic.
        """
        state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        return max(0.0, min(1.0, state.recent_sweep_count / 8.0))

    def reclaim_quality_score(self, level: LiquidityLevel) -> float:
        """
        Наскільки якісно після sweep ринок повернувся назад.
        """
        score = 0.0
        score += min(level.reclaim_count, 3) * 0.30
        score += level.strength * 0.40
        if level.status == LiquidityLevelStatus.RECLAIMED:
            score += 0.20
        if level.last_sweep_side is not None:
            score += 0.10
        return max(0.0, min(1.0, score))

    def liquidity_trap_score(self, layer: StructureLayer) -> float:
        """
        Високий score = висока ймовірність trap environment:
        stop-runs, failed breakouts, reclaim-и.
        """
        events = [e for e in self._events if e.layer == layer]
        recent = events[-12:]

        trap_points = 0.0
        for event in recent:
            if event.event_type == LiquidityEventType.STOP_RUN:
                trap_points += 1.0
            elif event.event_type == LiquidityEventType.FAILED_BREAKOUT:
                trap_points += 0.9
            elif event.event_type == LiquidityEventType.LIQUIDITY_RECLAIMED:
                trap_points += 0.7

        return max(0.0, min(1.0, trap_points / 6.0))

    def nearest_liquidity_summary(self, layer: StructureLayer) -> Dict[str, Any]:
        state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        return {
            "nearest_buy_side": self._level_to_dict(state.nearest_buy_side),
            "nearest_sell_side": self._level_to_dict(state.nearest_sell_side),
            "strongest_buy_side": self._level_to_dict(state.strongest_buy_side),
            "strongest_sell_side": self._level_to_dict(state.strongest_sell_side),
            "sweep_pressure_score": self.sweep_pressure_score(layer),
            "liquidity_trap_score": self.liquidity_trap_score(layer),
        }

    # ------------------------------------------------------------------
    # EventBus
    # ------------------------------------------------------------------

    def _append_event(self, event: LiquidityEvent) -> None:
        self._events.append(event)
        self._emit_event(event)

    def _emit_event(self, event: LiquidityEvent) -> None:
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
                "Failed to emit liquidity event",
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
                "Failed to publish liquidity snapshot",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "error": str(exc),
                },
            )

    # ------------------------------------------------------------------
    # Level matching helpers
    # ------------------------------------------------------------------

    def _find_matching_level(
        self,
        *,
        layer: StructureLayer,
        level_type: LiquidityLevelType,
        price: float,
    ) -> Optional[LiquidityLevel]:
        levels = self._levels_by_layer(layer)
        tolerance = max(abs(price) * self._equal_tolerance_pct(layer), 1e-12)

        candidates = [
            level for level in levels
            if level.level_type == level_type and level.status != LiquidityLevelStatus.INVALIDATED
        ]
        if not candidates:
            return None

        candidates.sort(key=lambda x: abs(x.price - price))
        best = candidates[0]
        if abs(best.price - price) <= tolerance:
            return best
        return None

    def _find_equal_level_cluster_candidate(
        self,
        swing: SwingPoint,
        equal_type: LiquidityLevelType,
    ) -> Optional[LiquidityLevel]:
        levels = self._levels_by_layer(swing.layer)
        tolerance = max(abs(swing.price) * self._equal_tolerance_pct(swing.layer), 1e-12)

        candidates = [
            level for level in levels
            if level.level_type == equal_type and level.status != LiquidityLevelStatus.INVALIDATED
        ]
        if not candidates:
            return None

        candidates.sort(key=lambda x: abs(x.price - swing.price))
        best = candidates[0]
        if abs(best.price - swing.price) <= tolerance:
            return best
        return None

    def _levels_by_layer(self, layer: StructureLayer) -> Deque[LiquidityLevel]:
        return self._internal_levels if layer == StructureLayer.INTERNAL else self._external_levels

    def _equal_tolerance_pct(self, layer: StructureLayer) -> float:
        return (
            self.config.equal_level_tolerance_pct_internal
            if layer == StructureLayer.INTERNAL
            else self.config.equal_level_tolerance_pct_external
        )

    def _zone_width_pct(self, layer: StructureLayer) -> float:
        return (
            self.config.swing_liquidity_zone_width_pct_internal
            if layer == StructureLayer.INTERNAL
            else self.config.swing_liquidity_zone_width_pct_external
        )

    def _base_strength_from_swing(self, swing: SwingPoint) -> float:
        score = 0.25 + swing.strength * 0.55
        if swing.layer == StructureLayer.EXTERNAL:
            score += 0.10
        return max(0.0, min(1.0, score))

    def _recalculate_level_strength(self, level: LiquidityLevel) -> float:
        score = 0.20
        score += min(level.source_count, 5) * 0.08
        score += min(level.touch_count, 6) * 0.05
        score += min(level.sweep_count, 4) * 0.12
        score += min(level.reclaim_count, 4) * 0.10

        if level.layer == StructureLayer.EXTERNAL:
            score += 0.08

        if level.level_type in {LiquidityLevelType.EQUAL_HIGHS, LiquidityLevelType.EQUAL_LOWS}:
            score += min(level.source_count, 4) * 0.05

        if level.status == LiquidityLevelStatus.RECLAIMED:
            score += 0.08

        return max(0.0, min(1.0, score))

    def _candle_intersects_zone(self, candle: Candle, level: LiquidityLevel) -> bool:
        return not (candle.high < level.lower_bound or candle.low > level.upper_bound)

    def _nearest_above(
        self,
        levels: Sequence[LiquidityLevel],
        price: Optional[float],
    ) -> Optional[LiquidityLevel]:
        if price is None:
            return None
        candidates = [x for x in levels if x.price >= price]
        if not candidates:
            return None
        return min(candidates, key=lambda x: abs(x.price - price))

    def _nearest_below(
        self,
        levels: Sequence[LiquidityLevel],
        price: Optional[float],
    ) -> Optional[LiquidityLevel]:
        if price is None:
            return None
        candidates = [x for x in levels if x.price <= price]
        if not candidates:
            return None
        return min(candidates, key=lambda x: abs(x.price - price))

    def _strongest_level(
        self,
        levels: Sequence[LiquidityLevel],
    ) -> Optional[LiquidityLevel]:
        if not levels:
            return None
        return max(levels, key=lambda x: x.strength)

    # ------------------------------------------------------------------
    # Event creation
    # ------------------------------------------------------------------

    def _create_event(
        self,
        *,
        event_type: LiquidityEventType,
        timestamp: datetime,
        level: LiquidityLevel,
        confidence: float,
        reference_price: Optional[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LiquidityEvent:
        return LiquidityEvent(
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

    @staticmethod
    def _safe_mean(values: Sequence[float]) -> float:
        return mean(values) if values else 0.0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _level_to_dict(self, level: Optional[LiquidityLevel]) -> Optional[Dict[str, Any]]:
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
            "touch_count": level.touch_count,
            "sweep_count": level.sweep_count,
            "reclaim_count": level.reclaim_count,
            "created_at": level.created_at.isoformat() if level.created_at else None,
            "updated_at": level.updated_at.isoformat() if level.updated_at else None,
            "last_touched_at": level.last_touched_at.isoformat() if level.last_touched_at else None,
            "swept_at": level.swept_at.isoformat() if level.swept_at else None,
            "reclaimed_at": level.reclaimed_at.isoformat() if level.reclaimed_at else None,
            "invalidated_at": level.invalidated_at.isoformat() if level.invalidated_at else None,
            "last_sweep_side": level.last_sweep_side,
            "last_sweep_price": level.last_sweep_price,
            "last_sweep_index": level.last_sweep_index,
            "source_swing_ids": list(level.source_swing_ids),
            "source_prices": list(level.source_prices),
            "source_count": level.source_count,
            "metadata": dict(level.metadata),
        }

    def _event_to_dict(self, event: Optional[LiquidityEvent]) -> Optional[Dict[str, Any]]:
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

    def _layer_state_to_dict(self, state: LayerLiquidityState) -> Dict[str, Any]:
        return {
            "layer": state.layer.value,
            "total_levels": state.total_levels,
            "active_levels": state.active_levels,
            "swept_levels": state.swept_levels,
            "reclaimed_levels": state.reclaimed_levels,
            "nearest_buy_side": self._level_to_dict(state.nearest_buy_side),
            "nearest_sell_side": self._level_to_dict(state.nearest_sell_side),
            "strongest_buy_side": self._level_to_dict(state.strongest_buy_side),
            "strongest_sell_side": self._level_to_dict(state.strongest_sell_side),
            "recent_sweep_count": state.recent_sweep_count,
            "last_event": self._event_to_dict(state.last_event),
            "metadata": dict(state.metadata),
        }


class SREventTypeAdapter:
    """
    Технічний адаптер, щоб не дублювати шматки логіки при формуванні merge-подій.
    """
    @staticmethod
    def level_merged() -> LiquidityEventType:
        return LiquidityEventType.LEVEL_MERGED