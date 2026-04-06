from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from analytics.price_action.base import BasePriceActionConfig, BasePriceActionModule
from analytics.price_action.enums import MarketBias, StructureEventType, StructureLayer, SwingType
from analytics.price_action.models import (
    Candle,
    MarketStructureState,
    MultiTimeframeAlignment,
    StructureEvent,
    StructureLayerState,
    SwingPoint,
)


@dataclass(slots=True)
class MarketStructureConfig(BasePriceActionConfig):
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

    alignment_window: int = 5
    external_strength_multiplier: float = 1.35
    min_external_strength: float = 0.30

    emit_events: bool = True
    event_namespace: str = "price_action.market_structure"
    publish_snapshots: bool = False

    def validate(self) -> None:
        super().validate()

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

        if self.max_internal_swings < 10:
            raise ValueError("max_internal_swings must be >= 10")

        if self.max_external_swings < 10:
            raise ValueError("max_external_swings must be >= 10")

        if self.max_events < 10:
            raise ValueError("max_events must be >= 10")

        if self.alignment_window < 1:
            raise ValueError("alignment_window must be >= 1")

        if self.external_strength_multiplier <= 0:
            raise ValueError("external_strength_multiplier must be > 0")

        if not 0.0 <= self.min_external_strength <= 1.0:
            raise ValueError("min_external_strength must be in [0.0, 1.0]")


class MarketStructureAnalyzer(BasePriceActionModule[MarketStructureState]):
    """
    Stateful production-style market structure analyzer.

    Features
    --------
    - incremental pivot-based swing detection
    - internal / external structure separation
    - HH / HL / LH / LL classification
    - BOS / CHOCH / MSS detection
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
        resolved_config = config or MarketStructureConfig()
        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            event_bus=event_bus,
            config=resolved_config,
            service_name="price_action.market_structure",
        )
        self.config: MarketStructureConfig = resolved_config
        self.higher_timeframe = higher_timeframe

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

    def get_state(self) -> MarketStructureState:
        return self._state

    def get_swings(self, layer: Optional[StructureLayer] = None) -> List[SwingPoint]:
        if layer == StructureLayer.INTERNAL:
            return list(self._internal_swings)
        if layer == StructureLayer.EXTERNAL:
            return list(self._external_swings)
        return [*self._internal_swings, *self._external_swings]

    def get_events(self) -> List[StructureEvent]:
        return list(self._events)

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
            candle = self._parse_candle(raw, index=self._global_candle_index)
            self._global_candle_index += 1

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
        bias_raw = context.get("bias") or context.get("higher_timeframe_bias") or MarketBias.UNKNOWN
        confidence = float(context.get("confidence", context.get("higher_timeframe_confidence", 0.0)))

        try:
            higher_bias = bias_raw if isinstance(bias_raw, MarketBias) else MarketBias(str(bias_raw))
        except ValueError:
            higher_bias = MarketBias.UNKNOWN

        mtf.higher_timeframe = tf
        mtf.higher_timeframe_bias = higher_bias
        mtf.higher_timeframe_confidence = max(0.0, min(1.0, confidence))
        mtf.last_updated = self._state.last_update
        mtf.metadata = dict(context.get("metadata", {}))

        self._refresh_alignment_state()

    def snapshot(self) -> Dict[str, Any]:
        return self._snapshot_envelope(
            state=self._state,
            metadata={
                "total_candles": len(self._candles),
                "internal_swings": len(self._internal_swings),
                "external_swings": len(self._external_swings),
                "events": len(self._events),
                "higher_timeframe": self.higher_timeframe,
                "last_processed_pivot_center_index": self._last_processed_pivot_center_index,
                "global_candle_index": self._global_candle_index,
                "config": self._serialize_config(),
            },
        )

    # -------------------------------------------------------------------------
    # Incremental pivot detection
    # -------------------------------------------------------------------------

    def _process_incremental_pivot_detection(self) -> List[SwingPoint]:
        candles = list(self._candles)
        needed = self.config.pivot_left + self.config.pivot_right + 1
        if len(candles) < needed:
            return []

        center_pos = len(candles) - self.config.pivot_right - 1
        center_candle = candles[center_pos]

        if center_candle.index <= self._last_processed_pivot_center_index:
            return []

        left_slice = candles[center_pos - self.config.pivot_left:center_pos]
        right_slice = candles[center_pos + 1:center_pos + 1 + self.config.pivot_right]

        is_swing_high = all(center_candle.high > x.high for x in left_slice) and all(
            center_candle.high >= x.high for x in right_slice
        )
        is_swing_low = all(center_candle.low < x.low for x in left_slice) and all(
            center_candle.low <= x.low for x in right_slice
        )

        self._last_processed_pivot_center_index = center_candle.index

        created_swings: List[SwingPoint] = []

        if is_swing_high:
            created_swings.extend(
                self._register_pivot(center_candle=center_candle, swing_type=SwingType.HIGH)
            )

        if is_swing_low:
            created_swings.extend(
                self._register_pivot(center_candle=center_candle, swing_type=SwingType.LOW)
            )

        return created_swings

    def _register_pivot(self, *, center_candle: Candle, swing_type: SwingType) -> List[SwingPoint]:
        pivot_key = (center_candle.index, swing_type)
        if pivot_key in self._processed_pivots:
            return []

        self._processed_pivots.add(pivot_key)

        created: List[SwingPoint] = []

        internal_swing = self._maybe_create_swing(
            center_candle=center_candle,
            swing_type=swing_type,
            layer=StructureLayer.INTERNAL,
        )
        if internal_swing is not None:
            self._internal_swings.append(internal_swing)
            created.append(internal_swing)
            self._emit_structure_event_for_swing(internal_swing)

        external_swing = self._maybe_create_swing(
            center_candle=center_candle,
            swing_type=swing_type,
            layer=StructureLayer.EXTERNAL,
        )
        if external_swing is not None:
            self._external_swings.append(external_swing)
            created.append(external_swing)
            self._emit_structure_event_for_swing(external_swing)

        return created

    def _maybe_create_swing(
        self,
        *,
        center_candle: Candle,
        swing_type: SwingType,
        layer: StructureLayer,
    ) -> Optional[SwingPoint]:
        price = center_candle.high if swing_type == SwingType.HIGH else center_candle.low
        min_distance_pct = self._layer_min_distance_pct(layer)
        strength = self._calculate_swing_strength(center_candle, swing_type, layer)

        if layer == StructureLayer.EXTERNAL and strength < self.config.min_external_strength:
            return None

        existing_swings = self._swings_for_layer(layer)
        previous_same_type = self._last_swing_of_type(existing_swings, swing_type)

        if previous_same_type is not None:
            if previous_same_type.price <= 0:
                return None

            distance_pct = abs(price - previous_same_type.price) / previous_same_type.price
            if distance_pct < min_distance_pct:
                return None

        return SwingPoint(
            swing_id=uuid4().hex,
            timestamp=center_candle.timestamp,
            price=price,
            swing_type=swing_type,
            layer=layer,
            index=center_candle.index,
            candle_open=center_candle.open,
            candle_high=center_candle.high,
            candle_low=center_candle.low,
            candle_close=center_candle.close,
            strength=max(0.0, min(1.0, strength)),
            is_confirmed=True,
            metadata={
                "body_ratio": center_candle.body_ratio,
                "range_size": center_candle.range_size,
            },
        )

    def _calculate_swing_strength(
        self,
        center_candle: Candle,
        swing_type: SwingType,
        layer: StructureLayer,
    ) -> float:
        candles = list(self._candles)
        center_pos = None
        for idx, candle in enumerate(candles):
            if candle.index == center_candle.index:
                center_pos = idx
                break

        if center_pos is None:
            return 0.0

        left_slice = candles[max(0, center_pos - self.config.pivot_left):center_pos]
        right_slice = candles[center_pos + 1:center_pos + 1 + self.config.pivot_right]
        neighbors = [*left_slice, *right_slice]
        if not neighbors:
            return 0.0

        if swing_type == SwingType.HIGH:
            pivot_distance = mean_safe([max(center_candle.high - x.high, 0.0) for x in neighbors])
            normalizer = center_candle.high if center_candle.high > 0 else 1.0
        else:
            pivot_distance = mean_safe([max(x.low - center_candle.low, 0.0) for x in neighbors])
            normalizer = center_candle.low if center_candle.low > 0 else 1.0

        distance_score = pivot_distance / normalizer
        candle_quality = min(1.0, center_candle.body_ratio + 0.25)
        range_score = min(1.0, center_candle.range_size / max(center_candle.close, 1e-9))

        score = (distance_score * 8.0 + candle_quality + range_score) / 3.0

        if layer == StructureLayer.EXTERNAL:
            score *= self.config.external_strength_multiplier

        return max(0.0, min(1.0, score))

    # -------------------------------------------------------------------------
    # Structure label classification
    # -------------------------------------------------------------------------

    def _classify_structure_labels(self, swings: Sequence[SwingPoint]) -> List[StructureEvent]:
        created_events: List[StructureEvent] = []

        grouped: Dict[StructureLayer, List[SwingPoint]] = {
            StructureLayer.INTERNAL: [],
            StructureLayer.EXTERNAL: [],
        }
        for swing in swings:
            grouped[swing.layer].append(swing)

        for layer, layer_swings in grouped.items():
            if not layer_swings:
                continue

            all_swings = self._sorted_swings_for_layer(layer)
            for swing in layer_swings:
                event = self._classify_single_swing(all_swings, swing)
                if event is not None:
                    created_events.append(event)

        return created_events

    def _classify_single_swing(
        self,
        all_swings: Sequence[SwingPoint],
        swing: SwingPoint,
    ) -> Optional[StructureEvent]:
        same_type_swings = [x for x in all_swings if x.swing_type == swing.swing_type and x.index < swing.index]
        if not same_type_swings:
            return None

        previous = same_type_swings[-1]

        if swing.swing_type == SwingType.HIGH:
            event_type = StructureEventType.HH if swing.price > previous.price else StructureEventType.LH
            direction = MarketBias.BULLISH if event_type == StructureEventType.HH else MarketBias.BEARISH
        else:
            event_type = StructureEventType.HL if swing.price > previous.price else StructureEventType.LL
            direction = MarketBias.BULLISH if event_type == StructureEventType.HL else MarketBias.BEARISH

        dedup_key = (swing.swing_id, event_type, swing.layer)
        if dedup_key in self._processed_structure_labels:
            return None

        self._processed_structure_labels.add(dedup_key)

        event = StructureEvent(
            event_id=uuid4().hex,
            event_type=event_type,
            timestamp=swing.timestamp,
            price=swing.price,
            layer=swing.layer,
            direction=direction,
            swing_id=swing.swing_id,
            reference_price=previous.price,
            reference_swing_id=previous.swing_id,
            confidence=self._label_confidence(swing, previous),
            metadata={
                "previous_price": previous.price,
                "previous_index": previous.index,
                "swing_strength": swing.strength,
            },
        )

        self._events.append(event)
        self._emit_event(
            self._build_event_name(event.event_type.value),
            self._event_to_dict(event),
            source="market_structure_analyzer",
        )

        return event

    def _label_confidence(self, current: SwingPoint, previous: SwingPoint) -> float:
        if previous.price <= 0:
            return max(0.0, min(1.0, current.strength))

        move_pct = abs(current.price - previous.price) / previous.price
        raw = (current.strength + min(1.0, move_pct * 100.0)) / 2.0
        return max(0.0, min(1.0, raw))

    # -------------------------------------------------------------------------
    # Break detection
    # -------------------------------------------------------------------------

    def _detect_break_events(self) -> List[StructureEvent]:
        if not self._candles:
            return []

        current_candle = self._candles[-1]
        created_events: List[StructureEvent] = []

        for layer in (StructureLayer.INTERNAL, StructureLayer.EXTERNAL):
            swings = self._sorted_swings_for_layer(layer)
            last_high = self._last_swing_of_type(swings, SwingType.HIGH)
            last_low = self._last_swing_of_type(swings, SwingType.LOW)

            if last_high is not None:
                high_break = self._maybe_break_event(
                    layer=layer,
                    current_candle=current_candle,
                    swing=last_high,
                    broken_side="high",
                )
                if high_break is not None:
                    created_events.append(high_break)

            if last_low is not None:
                low_break = self._maybe_break_event(
                    layer=layer,
                    current_candle=current_candle,
                    swing=last_low,
                    broken_side="low",
                )
                if low_break is not None:
                    created_events.append(low_break)

        return created_events

    def _maybe_break_event(
        self,
        *,
        layer: StructureLayer,
        current_candle: Candle,
        swing: SwingPoint,
        broken_side: str,
    ) -> Optional[StructureEvent]:
        threshold = self.config.structure_break_threshold_pct
        reference_price = swing.price

        if broken_side == "high":
            required_price = reference_price * (1.0 + threshold)
            broken = current_candle.close > required_price if self.config.require_close_break else current_candle.high > required_price
            direction = MarketBias.BULLISH
        else:
            required_price = reference_price * (1.0 - threshold)
            broken = current_candle.close < required_price if self.config.require_close_break else current_candle.low < required_price
            direction = MarketBias.BEARISH

        if not broken:
            return None

        prev_bias = self._layer_state(layer).bias
        event_type = self._resolve_break_event_type(prev_bias=prev_bias, direction=direction)

        dedup_key = (layer, swing.swing_id, str(current_candle.index), event_type.value)
        if dedup_key in self._processed_breaks:
            return None

        self._processed_breaks.add(dedup_key)

        confidence = self._break_confidence(current_candle=current_candle, swing=swing, direction=direction)

        event = StructureEvent(
            event_id=uuid4().hex,
            event_type=event_type,
            timestamp=current_candle.timestamp,
            price=current_candle.close,
            layer=layer,
            direction=direction,
            swing_id=None,
            reference_price=swing.price,
            reference_swing_id=swing.swing_id,
            confidence=confidence,
            metadata={
                "broken_side": broken_side,
                "trigger_candle_index": current_candle.index,
                "trigger_close": current_candle.close,
                "trigger_high": current_candle.high,
                "trigger_low": current_candle.low,
                "threshold_pct": threshold,
            },
        )

        self._events.append(event)
        self._emit_event(
            self._build_event_name(event.event_type.value),
            self._event_to_dict(event),
            source="market_structure_analyzer",
        )

        return event

    def _resolve_break_event_type(
        self,
        *,
        prev_bias: MarketBias,
        direction: MarketBias,
    ) -> StructureEventType:
        if prev_bias == MarketBias.UNKNOWN or prev_bias == MarketBias.RANGING:
            return StructureEventType.BOS

        if prev_bias == direction:
            return StructureEventType.BOS

        # Якщо був розворот проти попереднього bias
        return StructureEventType.CHOCH

    def _break_confidence(
        self,
        *,
        current_candle: Candle,
        swing: SwingPoint,
        direction: MarketBias,
    ) -> float:
        reference = swing.price if swing.price > 0 else 1.0
        if direction == MarketBias.BULLISH:
            move_pct = max(current_candle.close - swing.price, 0.0) / reference
        else:
            move_pct = max(swing.price - current_candle.close, 0.0) / reference

        raw = (swing.strength + current_candle.body_ratio + min(1.0, move_pct * 100.0)) / 3.0
        return max(0.0, min(1.0, raw))

    # -------------------------------------------------------------------------
    # State refresh
    # -------------------------------------------------------------------------

    def _refresh_state(self) -> None:
        self._refresh_layer_state(StructureLayer.INTERNAL)
        self._refresh_layer_state(StructureLayer.EXTERNAL)
        self._refresh_alignment_state()

    def _refresh_layer_state(self, layer: StructureLayer) -> None:
        state = self._layer_state(layer)
        swings = self._sorted_swings_for_layer(layer)
        events = [x for x in self._events if x.layer == layer]

        highs = [x for x in swings if x.swing_type == SwingType.HIGH]
        lows = [x for x in swings if x.swing_type == SwingType.LOW]

        state.last_swing_high = highs[-1] if highs else None
        state.previous_swing_high = highs[-2] if len(highs) >= 2 else None
        state.last_swing_low = lows[-1] if lows else None
        state.previous_swing_low = lows[-2] if len(lows) >= 2 else None

        state.last_hh = self._last_event_of_type(events, StructureEventType.HH)
        state.last_hl = self._last_event_of_type(events, StructureEventType.HL)
        state.last_lh = self._last_event_of_type(events, StructureEventType.LH)
        state.last_ll = self._last_event_of_type(events, StructureEventType.LL)
        state.last_bos = self._last_event_of_type(events, StructureEventType.BOS)
        state.last_choch = self._last_event_of_type(events, StructureEventType.CHOCH)
        state.last_mss = self._last_event_of_type(events, StructureEventType.MSS)

        state.swing_count = len(swings)
        state.event_count = len(events)
        state.sequence = [x.event_type.value for x in events[-10:]]

        state.bias = self._infer_bias(layer)
        state.trend_strength = self._infer_trend_strength(layer)
        state.confidence = self._infer_layer_confidence(layer)
        state.in_breakout = bool(
            state.last_bos is not None
            and state.last_bos.timestamp == self._state.last_update
        )

    def _refresh_alignment_state(self) -> None:
        mtf = self._state.mtf_alignment
        internal_bias = self._state.internal.bias
        external_bias = self._state.external.bias

        mtf.internal_with_external_aligned = (
            internal_bias == external_bias
            and internal_bias not in {MarketBias.UNKNOWN, MarketBias.RANGING}
        )

        if mtf.higher_timeframe_bias not in {MarketBias.UNKNOWN, MarketBias.RANGING}:
            mtf.internal_bias_aligned = internal_bias == mtf.higher_timeframe_bias
            mtf.external_bias_aligned = external_bias == mtf.higher_timeframe_bias
        else:
            mtf.internal_bias_aligned = False
            mtf.external_bias_aligned = False

        score = 0.0
        if mtf.internal_with_external_aligned:
            score += 0.4
        if mtf.internal_bias_aligned:
            score += 0.3
        if mtf.external_bias_aligned:
            score += 0.3

        mtf.alignment_score = max(0.0, min(1.0, score))

    def _infer_bias(self, layer: StructureLayer) -> MarketBias:
        state = self._layer_state(layer)

        if state.last_hh and state.last_hl:
            latest_bullish_ts = max(state.last_hh.timestamp, state.last_hl.timestamp)
        else:
            latest_bullish_ts = None

        if state.last_lh and state.last_ll:
            latest_bearish_ts = max(state.last_lh.timestamp, state.last_ll.timestamp)
        else:
            latest_bearish_ts = None

        last_break = None
        if state.last_bos and state.last_choch:
            last_break = max([state.last_bos, state.last_choch], key=lambda x: x.timestamp)
        else:
            last_break = state.last_bos or state.last_choch

        if last_break is not None and last_break.direction is not None:
            return last_break.direction

        if latest_bullish_ts and latest_bearish_ts:
            if latest_bullish_ts > latest_bearish_ts:
                return MarketBias.BULLISH
            if latest_bearish_ts > latest_bullish_ts:
                return MarketBias.BEARISH

        if latest_bullish_ts:
            return MarketBias.BULLISH
        if latest_bearish_ts:
            return MarketBias.BEARISH

        return MarketBias.UNKNOWN

    def _infer_trend_strength(self, layer: StructureLayer) -> float:
        swings = self._sorted_swings_for_layer(layer)
        if len(swings) < 2:
            return 0.0

        recent = swings[-self.config.alignment_window:]
        avg_strength = mean_safe([x.strength for x in recent])

        prices = [x.price for x in recent if x.price > 0]
        if len(prices) >= 2:
            price_dispersion = abs(prices[-1] - prices[0]) / prices[0]
        else:
            price_dispersion = 0.0

        raw = (avg_strength + min(1.0, price_dispersion * 100.0)) / 2.0
        return max(0.0, min(1.0, raw))

    def _infer_layer_confidence(self, layer: StructureLayer) -> float:
        state = self._layer_state(layer)

        components: List[float] = [state.trend_strength]

        if state.last_bos:
            components.append(state.last_bos.confidence)
        if state.last_choch:
            components.append(state.last_choch.confidence)

        if state.last_swing_high:
            components.append(state.last_swing_high.strength)
        if state.last_swing_low:
            components.append(state.last_swing_low.strength)

        if not components:
            return 0.0

        return max(0.0, min(1.0, sum(components) / len(components)))

    # -------------------------------------------------------------------------
    # Event helpers
    # -------------------------------------------------------------------------

    def _emit_structure_event_for_swing(self, swing: SwingPoint) -> None:
        event_type = (
            StructureEventType.SWING_HIGH if swing.swing_type == SwingType.HIGH else StructureEventType.SWING_LOW
        )

        event = StructureEvent(
            event_id=uuid4().hex,
            event_type=event_type,
            timestamp=swing.timestamp,
            price=swing.price,
            layer=swing.layer,
            direction=None,
            swing_id=swing.swing_id,
            reference_price=None,
            reference_swing_id=None,
            confidence=swing.strength,
            metadata={
                "index": swing.index,
                "strength": swing.strength,
            },
        )

        self._events.append(event)
        self._emit_event(
            self._build_event_name(event.event_type.value),
            self._event_to_dict(event),
            source="market_structure_analyzer",
        )

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

    def _layer_min_distance_pct(self, layer: StructureLayer) -> float:
        return (
            self.config.internal_min_swing_distance_pct
            if layer == StructureLayer.INTERNAL
            else self.config.external_min_swing_distance_pct
        )

    def _swings_for_layer(self, layer: StructureLayer) -> Deque[SwingPoint]:
        return self._internal_swings if layer == StructureLayer.INTERNAL else self._external_swings

    def _sorted_swings_for_layer(self, layer: StructureLayer) -> List[SwingPoint]:
        return sorted(self._swings_for_layer(layer), key=lambda x: x.index)

    def _layer_state(self, layer: StructureLayer) -> StructureLayerState:
        return self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

    def _last_swing_of_type(
        self,
        swings: Sequence[SwingPoint],
        swing_type: SwingType,
    ) -> Optional[SwingPoint]:
        filtered = [x for x in swings if x.swing_type == swing_type]
        return filtered[-1] if filtered else None

    def _last_event_of_type(
        self,
        events: Sequence[StructureEvent],
        event_type: StructureEventType,
    ) -> Optional[StructureEvent]:
        filtered = [x for x in events if x.event_type == event_type]
        return filtered[-1] if filtered else None

    def _swing_to_dict(self, swing: SwingPoint) -> Dict[str, Any]:
        return self._safe_serialize(swing)

    def _event_to_dict(self, event: StructureEvent) -> Dict[str, Any]:
        return self._safe_serialize(event)


def mean_safe(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)