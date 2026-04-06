from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from analytics.price_action.base import BasePriceActionConfig, BasePriceActionModule
from analytics.price_action.enums import (
    LiquidityEventType,
    LiquidityLevelStatus,
    LiquidityLevelType,
    StructureLayer,
    SwingType,
)
from analytics.price_action.models import (
    Candle,
    LayerLiquidityState,
    LiquidityEvent,
    LiquidityLevel,
    LiquidityState,
    SwingPoint,
)


@dataclass(slots=True)
class LiquidityLevelsConfig(BasePriceActionConfig):
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
        super().validate()

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
        if self.stop_run_wick_ratio_threshold < 0:
            raise ValueError("stop_run_wick_ratio_threshold must be >= 0")


class LiquidityLevelsAnalyzer(BasePriceActionModule[LiquidityState]):
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
        resolved_config = config or LiquidityLevelsConfig()
        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            event_bus=event_bus,
            config=resolved_config,
            service_name="price_action.liquidity_levels",
        )
        self.config: LiquidityLevelsConfig = resolved_config

        self._candles: Deque[Candle] = deque(maxlen=self.config.max_candles)
        self._internal_levels: Deque[LiquidityLevel] = deque(maxlen=self.config.max_levels_per_layer)
        self._external_levels: Deque[LiquidityLevel] = deque(maxlen=self.config.max_levels_per_layer)
        self._events: Deque[LiquidityEvent] = deque(maxlen=self.config.max_events)

        self._processed_swings: set[str] = set()
        self._processed_touch_keys: set[Tuple[str, int]] = set()
        self._processed_sweep_keys: set[Tuple[str, int]] = set()
        self._processed_reclaim_keys: set[Tuple[str, int]] = set()
        self._processed_failed_breakout_keys: set[Tuple[str, int]] = set()
        self._processed_stop_run_keys: set[Tuple[str, int]] = set()

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
        self._processed_sweep_keys.clear()
        self._processed_reclaim_keys.clear()
        self._processed_failed_breakout_keys.clear()
        self._processed_stop_run_keys.clear()

        self._global_candle_index = 0
        self._state = LiquidityState(symbol=self.symbol, timeframe=self.timeframe)

        self.logger.info(
            "LiquidityLevelsAnalyzer reset",
            extra={"symbol": self.symbol, "timeframe": self.timeframe},
        )

    def get_state(self) -> LiquidityState:
        return self._state

    def get_levels(self, layer: Optional[StructureLayer] = None) -> List[LiquidityLevel]:
        if layer == StructureLayer.INTERNAL:
            return list(self._internal_levels)
        if layer == StructureLayer.EXTERNAL:
            return list(self._external_levels)
        return [*self._internal_levels, *self._external_levels]

    def get_events(self) -> List[LiquidityEvent]:
        return list(self._events)

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
        return self._snapshot_envelope(
            state=self._state,
            metadata={
                "total_candles": len(self._candles),
                "internal_levels": len(self._internal_levels),
                "external_levels": len(self._external_levels),
                "events": len(self._events),
                "global_candle_index": self._global_candle_index,
                "config": self._serialize_config(),
            },
        )

    # -------------------------------------------------------------------------
    # Swings ingestion
    # -------------------------------------------------------------------------

    def _ingest_swings(
        self,
        swings: Sequence[SwingPoint | Mapping[str, Any]],
    ) -> Tuple[List[LiquidityLevel], List[LiquidityEvent]]:
        updated_levels: List[LiquidityLevel] = []
        new_events: List[LiquidityEvent] = []

        for raw in swings:
            swing = self._parse_swing(raw)

            if swing.swing_id in self._processed_swings:
                continue

            self._processed_swings.add(swing.swing_id)

            base_level_type = self._base_liquidity_type_from_swing(swing)
            side_level_type = self._side_liquidity_type_from_swing(swing)

            updated_base, events_base = self._upsert_liquidity_level_from_swing(
                swing=swing,
                level_type=base_level_type,
            )
            updated_side, events_side = self._upsert_liquidity_level_from_swing(
                swing=swing,
                level_type=side_level_type,
            )

            if updated_base is not None:
                updated_levels.append(updated_base)
            if updated_side is not None:
                updated_levels.append(updated_side)

            new_events.extend(events_base)
            new_events.extend(events_side)

        return updated_levels, new_events

    def _parse_swing(self, raw: SwingPoint | Mapping[str, Any]) -> SwingPoint:
        if isinstance(raw, SwingPoint):
            return raw

        swing_type_raw = raw["swing_type"]
        layer_raw = raw["layer"]

        return SwingPoint(
            swing_id=str(raw["swing_id"]),
            timestamp=self._ensure_utc_datetime(raw["timestamp"]),
            price=float(raw["price"]),
            swing_type=swing_type_raw if isinstance(swing_type_raw, SwingType) else SwingType(str(swing_type_raw)),
            layer=layer_raw if isinstance(layer_raw, StructureLayer) else StructureLayer(str(layer_raw)),
            index=int(raw["index"]),
            candle_open=float(raw.get("candle_open", raw.get("open", 0.0))),
            candle_high=float(raw.get("candle_high", raw.get("high", 0.0))),
            candle_low=float(raw.get("candle_low", raw.get("low", 0.0))),
            candle_close=float(raw.get("candle_close", raw.get("close", 0.0))),
            strength=float(raw.get("strength", 0.0)),
            is_confirmed=bool(raw.get("is_confirmed", True)),
            metadata=dict(raw.get("metadata", {})),
        )

    def _upsert_liquidity_level_from_swing(
        self,
        *,
        swing: SwingPoint,
        level_type: LiquidityLevelType,
    ) -> Tuple[Optional[LiquidityLevel], List[LiquidityEvent]]:
        events: List[LiquidityEvent] = []

        upper_bound, lower_bound = self._build_zone_bounds(price=swing.price, layer=swing.layer)
        existing = self._find_merge_candidate(
            layer=swing.layer,
            level_type=level_type,
            price=swing.price,
        )

        if existing is not None:
            was_equal_before = self._is_equal_level(existing)

            self._merge_swing_into_level(
                level=existing,
                swing=swing,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
            )

            events.append(
                self._create_event(
                    event_type=LiquidityEventType.LEVEL_MERGED,
                    level=existing,
                    timestamp=swing.timestamp,
                    reference_price=swing.price,
                    confidence=min(1.0, existing.strength),
                    metadata={
                        "merged_swing_id": swing.swing_id,
                        "source_count": existing.source_count,
                    },
                )
            )

            is_equal_now = self._is_equal_level(existing)
            if not was_equal_before and is_equal_now:
                existing.metadata["equal_cluster_confirmed"] = True

            return existing, events

        level = LiquidityLevel(
            level_id=uuid4().hex,
            layer=swing.layer,
            level_type=level_type,
            price=swing.price,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            strength=max(0.0, min(1.0, swing.strength)),
            status=LiquidityLevelStatus.ACTIVE,
            touch_count=1,
            sweep_count=0,
            reclaim_count=0,
            created_at=swing.timestamp,
            updated_at=swing.timestamp,
            source_swing_ids=[swing.swing_id],
            source_prices=[swing.price],
            source_count=1,
            metadata={
                "origin_swing_type": swing.swing_type.value,
                "origin_swing_index": swing.index,
                "equal_cluster_confirmed": False,
            },
        )

        self._levels_for_layer(swing.layer).append(level)

        events.append(
            self._create_event(
                event_type=LiquidityEventType.LEVEL_CREATED,
                level=level,
                timestamp=swing.timestamp,
                reference_price=swing.price,
                confidence=level.strength,
                metadata={
                    "source_swing_id": swing.swing_id,
                    "source_swing_type": swing.swing_type.value,
                },
            )
        )

        return level, events

    def _base_liquidity_type_from_swing(self, swing: SwingPoint) -> LiquidityLevelType:
        return (
            LiquidityLevelType.SWING_HIGH_LIQUIDITY
            if swing.swing_type == SwingType.HIGH
            else LiquidityLevelType.SWING_LOW_LIQUIDITY
        )

    def _side_liquidity_type_from_swing(self, swing: SwingPoint) -> LiquidityLevelType:
        return (
            LiquidityLevelType.BUY_SIDE_LIQUIDITY
            if swing.swing_type == SwingType.HIGH
            else LiquidityLevelType.SELL_SIDE_LIQUIDITY
        )

    def _merge_swing_into_level(
        self,
        level: LiquidityLevel,
        swing: SwingPoint,
        *,
        upper_bound: float,
        lower_bound: float,
    ) -> None:
        level.source_swing_ids.append(swing.swing_id)
        level.source_prices.append(swing.price)
        level.source_count += 1
        level.touch_count += 1

        level.price = sum(level.source_prices) / len(level.source_prices)
        level.upper_bound = max(level.upper_bound, upper_bound)
        level.lower_bound = min(level.lower_bound, lower_bound)
        level.updated_at = swing.timestamp

        avg_strength = (level.strength + max(0.0, min(1.0, swing.strength))) / 2.0
        source_bonus = min(0.25, 0.03 * max(0, level.source_count - 1))
        equal_cluster_bonus = 0.10 if level.source_count >= self.config.min_cluster_size_for_equal_levels else 0.0
        level.strength = max(0.0, min(1.0, avg_strength + source_bonus + equal_cluster_bonus))

        if self._can_promote_to_equal_level(level):
            if level.level_type == LiquidityLevelType.SWING_HIGH_LIQUIDITY:
                level.level_type = LiquidityLevelType.EQUAL_HIGHS
            elif level.level_type == LiquidityLevelType.SWING_LOW_LIQUIDITY:
                level.level_type = LiquidityLevelType.EQUAL_LOWS
            level.metadata["equal_cluster_confirmed"] = True

    # -------------------------------------------------------------------------
    # Candles ingestion
    # -------------------------------------------------------------------------

    def _ingest_candles(self, candles: Sequence[Mapping[str, Any]]) -> List[LiquidityEvent]:
        new_events: List[LiquidityEvent] = []

        for raw in candles:
            candle = self._parse_candle(raw, index=self._global_candle_index)
            self._global_candle_index += 1

            self._candles.append(candle)
            self._state.last_price = candle.close
            self._state.last_update = candle.timestamp

            for layer in (StructureLayer.INTERNAL, StructureLayer.EXTERNAL):
                layer_levels = list(self._levels_for_layer(layer))
                for level in layer_levels:
                    events = self._process_level_against_candle(level, candle)
                    if events:
                        new_events.extend(events)

        return new_events

    def _process_level_against_candle(
        self,
        level: LiquidityLevel,
        candle: Candle,
    ) -> List[LiquidityEvent]:
        events: List[LiquidityEvent] = []

        if level.status == LiquidityLevelStatus.INVALIDATED:
            return events

        touched = self._is_level_touched(level, candle)
        if touched:
            touch_key = (level.level_id, candle.index)
            if touch_key not in self._processed_touch_keys:
                self._processed_touch_keys.add(touch_key)
                level.touch_count += 1
                level.last_touched_at = candle.timestamp
                level.updated_at = candle.timestamp

                events.append(
                    self._create_event(
                        event_type=LiquidityEventType.LIQUIDITY_TOUCHED,
                        level=level,
                        timestamp=candle.timestamp,
                        reference_price=candle.close,
                        confidence=min(1.0, level.strength),
                        metadata={"candle_index": candle.index},
                    )
                )

        swept = self._is_swept(level, candle)
        if swept:
            sweep_key = (level.level_id, candle.index)
            if sweep_key not in self._processed_sweep_keys:
                self._processed_sweep_keys.add(sweep_key)

                sweep_side = self._sweep_side(level)
                sweep_price = candle.high if sweep_side == "up" else candle.low

                level.status = LiquidityLevelStatus.SWEPT
                level.sweep_count += 1
                level.swept_at = candle.timestamp
                level.updated_at = candle.timestamp
                level.last_sweep_side = sweep_side
                level.last_sweep_price = sweep_price
                level.last_sweep_index = candle.index

                events.append(
                    self._create_event(
                        event_type=LiquidityEventType.LIQUIDITY_SWEPT,
                        level=level,
                        timestamp=candle.timestamp,
                        reference_price=candle.close,
                        confidence=self._sweep_confidence(level, candle),
                        metadata={
                            "candle_index": candle.index,
                            "sweep_side": sweep_side,
                            "sweep_price": sweep_price,
                        },
                    )
                )

                if self._is_stop_run(level, candle):
                    stop_run_key = (level.level_id, candle.index)
                    if stop_run_key not in self._processed_stop_run_keys:
                        self._processed_stop_run_keys.add(stop_run_key)
                        events.append(
                            self._create_event(
                                event_type=LiquidityEventType.STOP_RUN,
                                level=level,
                                timestamp=candle.timestamp,
                                reference_price=candle.close,
                                confidence=self._stop_run_confidence(level, candle),
                                metadata={
                                    "candle_index": candle.index,
                                    "sweep_side": sweep_side,
                                },
                            )
                        )

        reclaimed = self._is_reclaimed(level, candle)
        if reclaimed:
            reclaim_key = (level.level_id, candle.index)
            if reclaim_key not in self._processed_reclaim_keys:
                self._processed_reclaim_keys.add(reclaim_key)

                level.status = LiquidityLevelStatus.RECLAIMED
                level.reclaim_count += 1
                level.reclaimed_at = candle.timestamp
                level.updated_at = candle.timestamp

                events.append(
                    self._create_event(
                        event_type=LiquidityEventType.LIQUIDITY_RECLAIMED,
                        level=level,
                        timestamp=candle.timestamp,
                        reference_price=candle.close,
                        confidence=self._reclaim_confidence(level, candle),
                        metadata={
                            "candle_index": candle.index,
                            "last_sweep_side": level.last_sweep_side,
                        },
                    )
                )

                if self._is_failed_breakout(level, candle):
                    failed_key = (level.level_id, candle.index)
                    if failed_key not in self._processed_failed_breakout_keys:
                        self._processed_failed_breakout_keys.add(failed_key)
                        events.append(
                            self._create_event(
                                event_type=LiquidityEventType.FAILED_BREAKOUT,
                                level=level,
                                timestamp=candle.timestamp,
                                reference_price=candle.close,
                                confidence=self._failed_breakout_confidence(level, candle),
                                metadata={
                                    "candle_index": candle.index,
                                    "last_sweep_side": level.last_sweep_side,
                                },
                            )
                        )

        self._maybe_invalidate_level(level, candle)

        return events

    # -------------------------------------------------------------------------
    # Detection rules
    # -------------------------------------------------------------------------

    def _is_level_touched(self, level: LiquidityLevel, candle: Candle) -> bool:
        return candle.high >= level.lower_bound and candle.low <= level.upper_bound

    def _is_swept(self, level: LiquidityLevel, candle: Candle) -> bool:
        penetration = self.config.min_sweep_penetration_pct

        if self._is_upper_side_liquidity(level):
            threshold = level.upper_bound * (1.0 + penetration)
            return candle.high > threshold

        threshold = level.lower_bound * (1.0 - penetration)
        return candle.low < threshold

    def _is_reclaimed(self, level: LiquidityLevel, candle: Candle) -> bool:
        if level.status != LiquidityLevelStatus.SWEPT:
            return False
        if level.last_sweep_index is None:
            return False

        if self._bars_since_index(level.last_sweep_index) is None:
            return False

        buffer_pct = self.config.reclaim_close_buffer_pct

        if self._is_upper_side_liquidity(level):
            reclaim_price = level.upper_bound * (1.0 - buffer_pct)
            if self.config.require_close_reclaim:
                return candle.close < reclaim_price
            return candle.low < reclaim_price or candle.close < reclaim_price

        reclaim_price = level.lower_bound * (1.0 + buffer_pct)
        if self.config.require_close_reclaim:
            return candle.close > reclaim_price
        return candle.high > reclaim_price or candle.close > reclaim_price

    def _is_failed_breakout(self, level: LiquidityLevel, candle: Candle) -> bool:
        if level.status != LiquidityLevelStatus.RECLAIMED:
            return False
        if level.last_sweep_index is None:
            return False

        bars_since_sweep = self._bars_since_index(level.last_sweep_index)
        if bars_since_sweep is None:
            return False

        return bars_since_sweep <= self.config.failed_breakout_reclaim_window_bars

    def _is_stop_run(self, level: LiquidityLevel, candle: Candle) -> bool:
        if self._is_upper_side_liquidity(level):
            return candle.upper_wick_ratio >= self.config.stop_run_wick_ratio_threshold and candle.close < level.price

        return candle.lower_wick_ratio >= self.config.stop_run_wick_ratio_threshold and candle.close > level.price

    def _maybe_invalidate_level(self, level: LiquidityLevel, candle: Candle) -> None:
        if level.status == LiquidityLevelStatus.INVALIDATED:
            return
        if level.status != LiquidityLevelStatus.RECLAIMED:
            return
        if level.last_sweep_index is None:
            return

        bars_since_sweep = self._bars_since_index(level.last_sweep_index)
        if bars_since_sweep is None:
            return

        if bars_since_sweep > self.config.retest_window_bars * 2 and level.reclaim_count == 0:
            level.status = LiquidityLevelStatus.INVALIDATED
            level.invalidated_at = candle.timestamp
            level.updated_at = candle.timestamp

    # -------------------------------------------------------------------------
    # State refresh
    # -------------------------------------------------------------------------

    def _refresh_state(self) -> None:
        self._refresh_layer_state(StructureLayer.INTERNAL)
        self._refresh_layer_state(StructureLayer.EXTERNAL)

    def _refresh_layer_state(self, layer: StructureLayer) -> None:
        state = self._layer_state(layer)
        levels = list(self._levels_for_layer(layer))

        active_levels = [x for x in levels if x.status == LiquidityLevelStatus.ACTIVE]
        swept_levels = [x for x in levels if x.status == LiquidityLevelStatus.SWEPT]
        reclaimed_levels = [x for x in levels if x.status == LiquidityLevelStatus.RECLAIMED]

        state.total_levels = len(levels)
        state.active_levels = len(active_levels)
        state.swept_levels = len(swept_levels)
        state.reclaimed_levels = len(reclaimed_levels)

        current_price = self._state.last_price
        state.nearest_buy_side = self._nearest_level(
            levels,
            current_price=current_price,
            upper_side=True,
        )
        state.nearest_sell_side = self._nearest_level(
            levels,
            current_price=current_price,
            upper_side=False,
        )
        state.strongest_buy_side = self._strongest_level(levels, upper_side=True)
        state.strongest_sell_side = self._strongest_level(levels, upper_side=False)

        layer_events = [x for x in self._events if x.layer == layer]
        state.last_event = layer_events[-1] if layer_events else None
        state.recent_sweep_count = len(
            [
                x for x in levels
                if x.last_sweep_index is not None
                and self._bars_since_index(x.last_sweep_index) is not None
                and self._bars_since_index(x.last_sweep_index) <= self.config.retest_window_bars
            ]
        )
        state.metadata = {
            "equal_highs": len([x for x in levels if x.level_type == LiquidityLevelType.EQUAL_HIGHS]),
            "equal_lows": len([x for x in levels if x.level_type == LiquidityLevelType.EQUAL_LOWS]),
            "buy_side_levels": len([x for x in levels if self._is_upper_side_liquidity(x)]),
            "sell_side_levels": len([x for x in levels if not self._is_upper_side_liquidity(x)]),
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _levels_for_layer(self, layer: StructureLayer) -> Deque[LiquidityLevel]:
        return self._internal_levels if layer == StructureLayer.INTERNAL else self._external_levels

    def _layer_state(self, layer: StructureLayer) -> LayerLiquidityState:
        return self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

    def _build_zone_bounds(self, *, price: float, layer: StructureLayer) -> Tuple[float, float]:
        width_pct = (
            self.config.swing_liquidity_zone_width_pct_internal
            if layer == StructureLayer.INTERNAL
            else self.config.swing_liquidity_zone_width_pct_external
        )
        half_width = price * width_pct
        return price + half_width, price - half_width

    def _equal_level_tolerance_pct(self, layer: StructureLayer) -> float:
        return (
            self.config.equal_level_tolerance_pct_internal
            if layer == StructureLayer.INTERNAL
            else self.config.equal_level_tolerance_pct_external
        )

    def _find_merge_candidate(
        self,
        *,
        layer: StructureLayer,
        level_type: LiquidityLevelType,
        price: float,
    ) -> Optional[LiquidityLevel]:
        candidates = [
            x for x in self._levels_for_layer(layer)
            if x.level_type == level_type and x.status != LiquidityLevelStatus.INVALIDATED
        ]
        if not candidates:
            return None

        threshold_pct = self._equal_level_tolerance_pct(layer)
        best: Optional[LiquidityLevel] = None
        best_distance = float("inf")

        for level in candidates:
            if level.price <= 0:
                continue
            distance_pct = abs(price - level.price) / level.price
            if distance_pct <= threshold_pct and distance_pct < best_distance:
                best = level
                best_distance = distance_pct

        return best

    def _can_promote_to_equal_level(self, level: LiquidityLevel) -> bool:
        if level.source_count < self.config.min_cluster_size_for_equal_levels:
            return False
        return level.level_type in {
            LiquidityLevelType.SWING_HIGH_LIQUIDITY,
            LiquidityLevelType.SWING_LOW_LIQUIDITY,
        }

    def _is_equal_level(self, level: LiquidityLevel) -> bool:
        return level.level_type in {
            LiquidityLevelType.EQUAL_HIGHS,
            LiquidityLevelType.EQUAL_LOWS,
        }

    def _is_upper_side_liquidity(self, level: LiquidityLevel) -> bool:
        return level.level_type in {
            LiquidityLevelType.EQUAL_HIGHS,
            LiquidityLevelType.BUY_SIDE_LIQUIDITY,
            LiquidityLevelType.SWING_HIGH_LIQUIDITY,
        }

    def _sweep_side(self, level: LiquidityLevel) -> str:
        return "up" if self._is_upper_side_liquidity(level) else "down"

    def _bars_since_index(self, index: int) -> Optional[int]:
        if not self._candles:
            return None
        return max(0, self._candles[-1].index - index)

    def _nearest_level(
        self,
        levels: Sequence[LiquidityLevel],
        *,
        current_price: Optional[float],
        upper_side: bool,
    ) -> Optional[LiquidityLevel]:
        if current_price is None:
            return None

        candidates = [
            x for x in levels
            if x.status != LiquidityLevelStatus.INVALIDATED
            and self._is_upper_side_liquidity(x) == upper_side
        ]
        if upper_side:
            candidates = [x for x in candidates if x.price >= current_price]
        else:
            candidates = [x for x in candidates if x.price <= current_price]

        if not candidates:
            return None

        return min(candidates, key=lambda x: abs(x.price - current_price))

    def _strongest_level(
        self,
        levels: Sequence[LiquidityLevel],
        *,
        upper_side: bool,
    ) -> Optional[LiquidityLevel]:
        candidates = [
            x for x in levels
            if x.status != LiquidityLevelStatus.INVALIDATED
            and self._is_upper_side_liquidity(x) == upper_side
        ]
        if not candidates:
            return None

        return max(
            candidates,
            key=lambda x: (x.strength, x.source_count, x.touch_count, x.sweep_count),
        )

    def _sweep_confidence(self, level: LiquidityLevel, candle: Candle) -> float:
        penetration = (
            max(candle.high - level.upper_bound, 0.0)
            if self._is_upper_side_liquidity(level)
            else max(level.lower_bound - candle.low, 0.0)
        ) / max(level.price, 1e-9)

        raw = (level.strength + candle.body_ratio + min(1.0, penetration * 150.0)) / 3.0
        return max(0.0, min(1.0, raw))

    def _reclaim_confidence(self, level: LiquidityLevel, candle: Candle) -> float:
        wick_ratio = candle.upper_wick_ratio if self._is_upper_side_liquidity(level) else candle.lower_wick_ratio
        raw = (level.strength + min(1.0, wick_ratio) + candle.body_ratio) / 3.0
        return max(0.0, min(1.0, raw))

    def _failed_breakout_confidence(self, level: LiquidityLevel, candle: Candle) -> float:
        raw = (self._reclaim_confidence(level, candle) + min(1.0, level.strength)) / 2.0
        return max(0.0, min(1.0, raw))

    def _stop_run_confidence(self, level: LiquidityLevel, candle: Candle) -> float:
        wick_ratio = candle.upper_wick_ratio if self._is_upper_side_liquidity(level) else candle.lower_wick_ratio
        raw = (level.strength + min(1.0, wick_ratio)) / 2.0
        return max(0.0, min(1.0, raw))

    def _create_event(
        self,
        *,
        event_type: LiquidityEventType,
        level: LiquidityLevel,
        timestamp: Any,
        reference_price: Optional[float],
        confidence: float,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> LiquidityEvent:
        event = LiquidityEvent(
            event_id=uuid4().hex,
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
            metadata=dict(metadata or {}),
        )
        self._events.append(event)
        self._emit_event(
            self._build_event_name(event.event_type.value),
            self._event_to_dict(event),
            source="liquidity_levels_analyzer",
        )
        return event

    def _level_to_dict(self, level: LiquidityLevel) -> Dict[str, Any]:
        return self._safe_serialize(level)

    def _event_to_dict(self, event: LiquidityEvent) -> Dict[str, Any]:
        return self._safe_serialize(event)