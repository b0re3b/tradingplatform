from __future__ import annotations

import asyncio
import inspect
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, NewType
from uuid import uuid4

from core.logger import get_logger
from analytics.price_action.market_structure import MarketBias, StructureLayer


SignedScore = NewType("SignedScore", float)   # expected range [-1.0, 1.0]
UnitScore = NewType("UnitScore", float)       # expected range [0.0, 1.0]


class TrendDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class TrendRegime(str, Enum):
    TRENDING = "trending"
    PULLBACK = "pullback"
    CONSOLIDATING = "consolidating"
    REVERSING = "reversing"
    EXHAUSTED = "exhausted"
    UNKNOWN = "unknown"


class TrendEventType(str, Enum):
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


@dataclass(slots=True)
class TrendDetectionConfig:
    max_candles: int = 500
    max_events: int = 500

    short_window: int = 10
    medium_window: int = 20
    long_window: int = 50
    atr_window: int = 14

    trend_strength_threshold: float = 0.55
    acceleration_threshold: float = 0.70
    exhaustion_threshold: float = 0.72
    reversal_risk_threshold: float = 0.68

    pullback_depth_threshold: float = 0.0035
    momentum_slope_threshold: float = 0.0015
    consolidation_range_threshold: float = 0.0045

    direction_positive_threshold: float = 0.22
    direction_negative_threshold: float = -0.22
    structure_bias_weight: float = 0.15

    emit_events: bool = True
    event_namespace: str = "price_action.trend_detection"
    publish_snapshots: bool = False
    log_missing_mtf_once: bool = True

    def validate(self) -> None:
        if self.max_candles < 100:
            raise ValueError("max_candles must be >= 100")
        if self.max_events < 50:
            raise ValueError("max_events must be >= 50")

        if self.short_window < 2:
            raise ValueError("short_window must be >= 2")
        if self.medium_window <= self.short_window:
            raise ValueError("medium_window must be > short_window")
        if self.long_window <= self.medium_window:
            raise ValueError("long_window must be > medium_window")
        if self.atr_window < 2:
            raise ValueError("atr_window must be >= 2")

        bounded_unit_fields = (
            "trend_strength_threshold",
            "acceleration_threshold",
            "exhaustion_threshold",
            "reversal_risk_threshold",
        )
        for field_name in bounded_unit_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0.0, 1.0]")

        if self.exhaustion_threshold < self.acceleration_threshold:
            raise ValueError("exhaustion_threshold must be >= acceleration_threshold")

        if self.pullback_depth_threshold <= 0.0:
            raise ValueError("pullback_depth_threshold must be > 0")

        if self.momentum_slope_threshold < 0.0:
            raise ValueError("momentum_slope_threshold must be >= 0")

        if self.consolidation_range_threshold <= 0.0:
            raise ValueError("consolidation_range_threshold must be > 0")

        if self.direction_negative_threshold >= self.direction_positive_threshold:
            raise ValueError("direction_negative_threshold must be < direction_positive_threshold")

        if not 0.0 <= self.structure_bias_weight <= 0.30:
            raise ValueError("structure_bias_weight must be in [0.0, 0.30]")


@dataclass(slots=True)
class Candle:
    """
    Candle.index is a detector-local monotonic stream index.

    Contract:
    - index is monotonically increasing for the lifetime of a TrendDetector instance
    - index is NOT reset by TrendDetector.reset()
    - index is suitable for ordering inside a single detector stream
    - index should not be interpreted as exchange-native sequence id
    """
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
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


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


class TrendDetector:
    """
    Production-style trend detector with:
    - per-layer trend state
    - cross-layer alignment/disagreement events
    - async/sync EventBus compatibility
    - conflict-resolved signal emission
    - point-in-time snapshots with state versioning

    Public API contract notes:
    - reset() clears detector state/history/signals but preserves candle stream monotonic index
    - snapshot() returns a point-in-time immutable-by-convention dict representation
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        event_bus: Optional[Any] = None,
        config: Optional[TrendDetectionConfig] = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.event_bus = event_bus
        self.config = config or TrendDetectionConfig()
        self.config.validate()

        self.logger = get_logger(__name__, service_name="price_action.trend_detection")

        self._candles: Deque[Candle] = deque(maxlen=self.config.max_candles)
        self._signals: Deque[TrendSignal] = deque(maxlen=self.config.max_events)
        self._pending_tasks: Set[asyncio.Task[Any]] = set()

        self._global_candle_index = 0
        self._state_version = 0

        self._latest_market_structure: Dict[str, Any] = {}
        self._latest_support_resistance: Dict[str, Any] = {}

        self._state = TrendState(symbol=self.symbol, timeframe=self.timeframe)
        self._missing_mtf_logged = False

        self.logger.info(
            "Initialized TrendDetector",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "config": self._serialize_config(),
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset detector rolling state, cached input context and generated signals.

        Important contract:
        - _global_candle_index is intentionally preserved
        - newly ingested candles will continue with strictly increasing Candle.index
        - downstream consumers may rely on Candle.index monotonicity across reset()
        """
        self._candles.clear()
        self._signals.clear()
        self._latest_market_structure = {}
        self._latest_support_resistance = {}
        self._state = TrendState(symbol=self.symbol, timeframe=self.timeframe)
        self._missing_mtf_logged = False
        self._state_version += 1

        self.logger.info(
            "TrendDetector reset",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "global_candle_index_preserved": self._global_candle_index,
                "state_version": self._state_version,
            },
        )

    async def shutdown(self) -> None:
        """
        Gracefully wait for pending async EventBus tasks.
        """
        if not self._pending_tasks:
            return

        pending = list(self._pending_tasks)
        self.logger.info(
            "Shutting down TrendDetector with pending event tasks",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "pending_tasks": len(pending),
            },
        )
        await asyncio.gather(*pending, return_exceptions=True)

    def update(
        self,
        *,
        candles: Optional[Sequence[Mapping[str, Any]]] = None,
        market_structure: Optional[Mapping[str, Any]] = None,
        support_resistance: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.add_data(
            candles=candles,
            market_structure=market_structure,
            support_resistance=support_resistance,
        )

    def add_data(
        self,
        *,
        candles: Optional[Sequence[Mapping[str, Any]]] = None,
        market_structure: Optional[Mapping[str, Any]] = None,
        support_resistance: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if candles:
            for raw in candles:
                candle = self._parse_candle(raw)
                self._candles.append(candle)
                self._state.last_price = candle.close
                self._state.last_update = candle.timestamp

        if market_structure is not None:
            self._latest_market_structure = dict(market_structure)

        if support_resistance is not None:
            self._latest_support_resistance = dict(support_resistance)

        self._state_version += 1

        new_signals: List[TrendSignal] = []
        new_signals.extend(self._refresh_layer(StructureLayer.INTERNAL))
        new_signals.extend(self._refresh_layer(StructureLayer.EXTERNAL))
        self._refresh_global_state()
        new_signals.extend(self._detect_cross_layer_events())

        if self.config.publish_snapshots:
            self._publish_snapshot()

        return {
            "state": self.snapshot(),
            "new_signals": [self._signal_to_dict(signal) for signal in new_signals],
        }

    def get_state(self) -> TrendState:
        return self._state

    def get_signals(self) -> List[TrendSignal]:
        return list(self._signals)

    def snapshot(self) -> Dict[str, Any]:
        snapshot_ts = datetime.now(timezone.utc).isoformat()
        return {
            "symbol": self._state.symbol,
            "timeframe": self._state.timeframe,
            "last_price": self._state.last_price,
            "last_update": self._state.last_update.isoformat() if self._state.last_update else None,
            "internal": self._layer_state_to_dict(self._state.internal),
            "external": self._layer_state_to_dict(self._state.external),
            "internal_external_alignment": float(self._state.internal_external_alignment),
            "higher_timeframe_alignment": float(self._state.higher_timeframe_alignment),
            "overall_trend_score": float(self._state.overall_trend_score),
            "metadata": {
                **dict(self._state.metadata),
                "snapshot_ts": snapshot_ts,
                "state_version": self._state_version,
            },
        }

    # ------------------------------------------------------------------
    # Core refresh
    # ------------------------------------------------------------------

    def _refresh_layer(self, layer: StructureLayer) -> List[TrendSignal]:
        layer_state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

        if not self._candles:
            self._clear_layer_state_no_candles(layer_state)
            return []

        previous_direction = layer_state.direction
        previous_regime = layer_state.regime
        previous_strength = float(layer_state.strength)
        previous_pullback = layer_state.in_pullback

        closes = [c.close for c in self._candles]
        highs = [c.high for c in self._candles]
        lows = [c.low for c in self._candles]

        momentum_direction_score = self._calculate_momentum_direction_score(closes)
        slope_direction_score = self._calculate_slope_direction_score(
            closes,
            self.config.short_window,
            self.config.medium_window,
        )
        structure_score, structure_bias = self._extract_structure_score(layer)
        sr_score = self._extract_sr_context_score(layer)
        pullback_depth = self._calculate_pullback_depth(closes)
        consolidation_score = self._calculate_consolidation_score(highs, lows, closes)
        exhaustion_score = self._calculate_exhaustion_score()

        direction = self._derive_direction(
            momentum_direction_score=momentum_direction_score,
            slope_direction_score=slope_direction_score,
            structure_bias=structure_bias,
        )

        continuation_probability = self._calculate_continuation_probability(
            direction=direction,
            momentum_direction_score=momentum_direction_score,
            slope_direction_score=slope_direction_score,
            structure_score=structure_score,
            sr_score=sr_score,
            exhaustion_score=exhaustion_score,
            consolidation_score=consolidation_score,
        )

        reversal_risk = self._calculate_reversal_risk(
            structure_score=structure_score,
            momentum_direction_score=momentum_direction_score,
            exhaustion_score=exhaustion_score,
            pullback_depth=pullback_depth,
            consolidation_score=consolidation_score,
        )

        strength = self._calculate_strength(
            momentum_direction_score=momentum_direction_score,
            slope_direction_score=slope_direction_score,
            structure_score=structure_score,
            sr_score=sr_score,
        )

        confidence = self._calculate_confidence(
            direction=direction,
            structure_bias=structure_bias,
            structure_score=structure_score,
            momentum_direction_score=momentum_direction_score,
            slope_direction_score=slope_direction_score,
        )

        regime = self._derive_regime(
            direction=direction,
            strength=strength,
            pullback_depth=pullback_depth,
            consolidation_score=consolidation_score,
            exhaustion_score=exhaustion_score,
            reversal_risk=reversal_risk,
        )

        layer_state.direction = direction
        layer_state.regime = regime
        layer_state.strength = strength
        layer_state.confidence = confidence
        layer_state.momentum_direction_score = momentum_direction_score
        layer_state.slope_direction_score = slope_direction_score
        layer_state.structure_score = structure_score
        layer_state.continuation_probability = continuation_probability
        layer_state.reversal_risk = reversal_risk
        layer_state.exhaustion_score = exhaustion_score
        layer_state.pullback_depth = pullback_depth
        layer_state.consolidation_score = consolidation_score
        layer_state.is_accelerating = (
            float(strength) >= self.config.acceleration_threshold
            and self._directional_component(direction, momentum_direction_score) >= 0.65
        )
        layer_state.is_exhausted = float(exhaustion_score) >= self.config.exhaustion_threshold
        layer_state.in_pullback = regime == TrendRegime.PULLBACK
        layer_state.is_aligned_with_structure = self._is_direction_aligned_with_bias(direction, structure_bias)
        layer_state.metadata = {
            "structure_bias": structure_bias.value,
            "sr_score": float(sr_score),
            "close_count": len(closes),
            "direction_signed_score": (
                float(momentum_direction_score) * 0.45
                + float(slope_direction_score) * 0.35
                + self._structure_bias_signed_component(structure_bias)
            ),
        }

        signals = self._detect_layer_events(
            layer=layer,
            previous_direction=previous_direction,
            previous_regime=previous_regime,
            previous_strength=previous_strength,
            previous_pullback=previous_pullback,
            current_state=layer_state,
        )
        return signals

    def _refresh_global_state(self) -> None:
        internal = self._state.internal
        external = self._state.external

        self._state.internal_external_alignment = self._calculate_internal_external_alignment()
        mtf_alignment, mtf_available = self._calculate_higher_timeframe_alignment()
        self._state.higher_timeframe_alignment = mtf_alignment

        weighted_components: List[Tuple[float, float]] = [
            (float(internal.strength), 0.35),
            (float(external.strength), 0.40),
            (float(self._state.internal_external_alignment), 0.15),
        ]
        if mtf_available:
            weighted_components.append((float(mtf_alignment), 0.10))

        weight_sum = sum(weight for _, weight in weighted_components)
        total = sum(value * weight for value, weight in weighted_components)

        overall = total / weight_sum if weight_sum > 0 else 0.0
        self._state.overall_trend_score = self._normalize_unit_score(overall)

        self._state.metadata = {
            "signal_count": len(self._signals),
            "market_structure_available": bool(self._latest_market_structure),
            "support_resistance_available": bool(self._latest_support_resistance),
            "mtf_alignment_available": mtf_available,
            "global_candle_index": self._global_candle_index,
            "state_version": self._state_version,
        }

    # ------------------------------------------------------------------
    # Signal detection and conflict resolution
    # ------------------------------------------------------------------

    def _detect_layer_events(
        self,
        *,
        layer: StructureLayer,
        previous_direction: TrendDirection,
        previous_regime: TrendRegime,
        previous_strength: float,
        previous_pullback: bool,
        current_state: TrendLayerState,
    ) -> List[TrendSignal]:
        candidates: List[TrendSignal] = []

        current_direction = current_state.direction
        current_regime = current_state.regime
        current_strength = float(current_state.strength)

        if previous_direction in {TrendDirection.UNKNOWN, TrendDirection.NEUTRAL} and current_direction in {
            TrendDirection.BULLISH,
            TrendDirection.BEARISH,
        }:
            candidates.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_STARTED,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=float(current_state.confidence),
                    regime=current_regime,
                    metadata={"from_direction": previous_direction.value},
                )
            )
        elif previous_direction in {TrendDirection.BULLISH, TrendDirection.BEARISH} and current_direction != previous_direction:
            if current_direction in {TrendDirection.BULLISH, TrendDirection.BEARISH}:
                candidates.append(
                    self._create_signal(
                        layer=layer,
                        event_type=TrendEventType.TREND_REVERSAL,
                        direction=current_direction,
                        strength=current_strength,
                        confidence=float(current_state.confidence),
                        regime=current_regime,
                        metadata={"from_direction": previous_direction.value},
                    )
                )
        elif current_direction in {TrendDirection.BULLISH, TrendDirection.BEARISH} and current_strength > previous_strength + 0.08:
            candidates.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_CONTINUATION,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=float(current_state.confidence),
                    regime=current_regime,
                    metadata={"previous_strength": previous_strength},
                )
            )

        if not previous_pullback and current_state.in_pullback:
            candidates.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.PULLBACK_STARTED,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=float(current_state.confidence),
                    regime=current_regime,
                    metadata={"pullback_depth": float(current_state.pullback_depth)},
                )
            )
        elif previous_pullback and not current_state.in_pullback:
            candidates.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.PULLBACK_ENDED,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=float(current_state.confidence),
                    regime=current_regime,
                    metadata={"pullback_depth": float(current_state.pullback_depth)},
                )
            )

        if current_state.is_accelerating and current_strength > previous_strength:
            candidates.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_ACCELERATION,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=float(current_state.confidence),
                    regime=current_regime,
                    metadata={"momentum_direction_score": float(current_state.momentum_direction_score)},
                )
            )

        if current_strength < previous_strength - 0.10 and current_direction in {TrendDirection.BULLISH, TrendDirection.BEARISH}:
            candidates.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_WEAKENING,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=float(current_state.confidence),
                    regime=current_regime,
                    metadata={"previous_strength": previous_strength},
                )
            )

        if current_state.is_exhausted:
            candidates.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_EXHAUSTION,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=float(current_state.confidence),
                    regime=current_regime,
                    metadata={"exhaustion_score": float(current_state.exhaustion_score)},
                )
            )

        resolved = self._resolve_signal_conflicts(candidates)
        for signal in resolved:
            self._append_signal(signal)

        if resolved:
            current_state.last_signal = resolved[-1]

        return resolved

    def _resolve_signal_conflicts(self, signals: List[TrendSignal]) -> List[TrendSignal]:
        if not signals:
            return signals

        by_type: Dict[TrendEventType, TrendSignal] = {signal.event_type: signal for signal in signals}

        highest_priority_singletons = (
            TrendEventType.TREND_REVERSAL,
            TrendEventType.TREND_STARTED,
        )
        for event_type in highest_priority_singletons:
            if event_type in by_type:
                base = [by_type[event_type]]
                if TrendEventType.PULLBACK_STARTED in by_type:
                    base.append(by_type[TrendEventType.PULLBACK_STARTED])
                elif TrendEventType.PULLBACK_ENDED in by_type:
                    base.append(by_type[TrendEventType.PULLBACK_ENDED])

                if TrendEventType.TREND_EXHAUSTION in by_type and event_type != TrendEventType.TREND_STARTED:
                    base.append(by_type[TrendEventType.TREND_EXHAUSTION])

                return self._deduplicate_preserve_order(base)

        filtered = list(signals)

        if TrendEventType.TREND_CONTINUATION in by_type and TrendEventType.PULLBACK_STARTED in by_type:
            filtered = [s for s in filtered if s.event_type != TrendEventType.TREND_CONTINUATION]

        if TrendEventType.TREND_ACCELERATION in by_type and TrendEventType.TREND_WEAKENING in by_type:
            weakening = by_type[TrendEventType.TREND_WEAKENING]
            acceleration = by_type[TrendEventType.TREND_ACCELERATION]
            if acceleration.strength >= weakening.strength:
                filtered = [s for s in filtered if s.event_type != TrendEventType.TREND_WEAKENING]
            else:
                filtered = [s for s in filtered if s.event_type != TrendEventType.TREND_ACCELERATION]

        if TrendEventType.TREND_ACCELERATION in by_type and TrendEventType.TREND_EXHAUSTION in by_type:
            filtered = [s for s in filtered if s.event_type != TrendEventType.TREND_ACCELERATION]

        return self._deduplicate_preserve_order(filtered)

    def _detect_cross_layer_events(self) -> List[TrendSignal]:
        signals: List[TrendSignal] = []

        internal = self._state.internal
        external = self._state.external

        if (
            internal.direction in {TrendDirection.BULLISH, TrendDirection.BEARISH}
            and external.direction in {TrendDirection.BULLISH, TrendDirection.BEARISH}
        ):
            if internal.direction == external.direction:
                current_status = "aligned"
            else:
                current_status = "disagreement"
        else:
            current_status = "neutral"

        previous_status = str(self._state.metadata.get("cross_layer_status", "unknown"))
        self._state.metadata["cross_layer_status"] = current_status

        if current_status == previous_status:
            return signals

        if current_status == "aligned":
            signal = self._create_signal(
                layer=StructureLayer.EXTERNAL,
                event_type=TrendEventType.TREND_ALIGNMENT,
                direction=external.direction,
                strength=max(float(internal.strength), float(external.strength)),
                confidence=(float(internal.confidence) + float(external.confidence)) / 2.0,
                regime=external.regime,
                metadata={
                    "internal_direction": internal.direction.value,
                    "external_direction": external.direction.value,
                    "alignment_score": float(self._state.internal_external_alignment),
                    "previous_status": previous_status,
                },
            )
            self._append_signal(signal)
            signals.append(signal)

        elif current_status == "disagreement":
            signal = self._create_signal(
                layer=StructureLayer.EXTERNAL,
                event_type=TrendEventType.TREND_DISAGREEMENT,
                direction=external.direction,
                strength=float(external.strength),
                confidence=float(external.confidence),
                regime=external.regime,
                metadata={
                    "internal_direction": internal.direction.value,
                    "external_direction": external.direction.value,
                    "alignment_score": float(self._state.internal_external_alignment),
                    "previous_status": previous_status,
                },
            )
            self._append_signal(signal)
            signals.append(signal)

        return signals

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    def _extract_structure_score(self, layer: StructureLayer) -> Tuple[UnitScore, MarketBias]:
        layer_key = layer.value
        if not self._latest_market_structure:
            return UnitScore(0.0), MarketBias.UNKNOWN

        layer_data = self._latest_market_structure.get(layer_key, {})
        bias_raw = layer_data.get("bias", MarketBias.UNKNOWN.value)

        try:
            bias = MarketBias(bias_raw)
        except ValueError:
            bias = MarketBias.UNKNOWN

        confidence = float(layer_data.get("confidence", 0.0))
        trend_strength = float(layer_data.get("trend_strength", 0.0))
        in_breakout = bool(layer_data.get("in_breakout", False))

        score = confidence * 0.55 + trend_strength * 0.35 + (0.10 if in_breakout else 0.0)
        return self._normalize_unit_score(score), bias

    def _extract_sr_context_score(self, layer: StructureLayer) -> UnitScore:
        if not self._latest_support_resistance:
            return UnitScore(0.0)

        layer_data = self._latest_support_resistance.get(layer.value, {})
        strongest_support = layer_data.get("strongest_support")
        strongest_resistance = layer_data.get("strongest_resistance")
        nearest_support = layer_data.get("nearest_support")
        nearest_resistance = layer_data.get("nearest_resistance")

        score = 0.0
        if strongest_support:
            score += float(strongest_support.get("strength", 0.0)) * 0.25
        if strongest_resistance:
            score += float(strongest_resistance.get("strength", 0.0)) * 0.25
        if nearest_support:
            score += float(nearest_support.get("strength", 0.0)) * 0.20
        if nearest_resistance:
            score += float(nearest_resistance.get("strength", 0.0)) * 0.20

        return self._normalize_unit_score(score)

    def _calculate_momentum_direction_score(self, closes: Sequence[float]) -> SignedScore:
        if len(closes) < self.config.medium_window:
            return SignedScore(0.0)

        short_ma = self._safe_mean(closes[-self.config.short_window:])
        medium_ma = self._safe_mean(closes[-self.config.medium_window:])
        long_window = min(self.config.long_window, len(closes))
        long_ma = self._safe_mean(closes[-long_window:])

        if long_ma == 0:
            return SignedScore(0.0)

        separation_1 = (short_ma - medium_ma) / abs(long_ma)
        separation_2 = (medium_ma - long_ma) / abs(long_ma)
        raw = (separation_1 * 80.0) + (separation_2 * 50.0)

        return self._normalize_signed_score(raw)

    def _calculate_slope_direction_score(
        self,
        closes: Sequence[float],
        short_window: int,
        medium_window: int,
    ) -> SignedScore:
        if len(closes) < medium_window:
            return SignedScore(0.0)

        short_slope = self._linear_slope(closes[-short_window:])
        medium_slope = self._linear_slope(closes[-medium_window:])
        combined = short_slope * 0.65 + medium_slope * 0.35
        return self._normalize_signed_score(combined * 200.0)

    def _calculate_pullback_depth(self, closes: Sequence[float]) -> UnitScore:
        if len(closes) < self.config.medium_window:
            return UnitScore(0.0)

        recent = closes[-self.config.medium_window:]
        max_close = max(recent)
        min_close = min(recent)
        last = recent[-1]

        if max_close <= 0 or min_close <= 0:
            return UnitScore(0.0)

        bullish_pullback = (max_close - last) / max_close
        bearish_pullback = (last - min_close) / min_close
        return self._normalize_unit_score(max(bullish_pullback, bearish_pullback))

    def _calculate_consolidation_score(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
    ) -> UnitScore:
        if len(closes) < self.config.short_window:
            return UnitScore(0.0)

        window = min(self.config.short_window, len(closes))
        local_high = max(highs[-window:])
        local_low = min(lows[-window:])
        local_mean = self._safe_mean(closes[-window:])

        if local_mean == 0:
            return UnitScore(0.0)

        compression = (local_high - local_low) / abs(local_mean)
        raw = 1.0 - min(1.0, compression / self.config.consolidation_range_threshold)
        return self._normalize_unit_score(raw)

    def _calculate_exhaustion_score(self) -> UnitScore:
        if len(self._candles) < max(self.config.atr_window + 2, 10):
            return UnitScore(0.0)

        recent = list(self._candles)[-min(12, len(self._candles)):]
        atr = self._calculate_atr(self.config.atr_window)
        if atr <= 0:
            return UnitScore(0.0)

        impulse_stretch = self._impulse_stretch_score(recent, atr)
        wick_pressure = self._wick_pressure_score(recent)
        deceleration = self._deceleration_score(recent)

        score = float(impulse_stretch) * 0.45 + float(wick_pressure) * 0.30 + float(deceleration) * 0.25
        return self._normalize_unit_score(score)

    def _calculate_continuation_probability(
        self,
        *,
        direction: TrendDirection,
        momentum_direction_score: SignedScore,
        slope_direction_score: SignedScore,
        structure_score: UnitScore,
        sr_score: UnitScore,
        exhaustion_score: UnitScore,
        consolidation_score: UnitScore,
    ) -> UnitScore:
        directional_momentum = self._directional_component(direction, momentum_direction_score)
        directional_slope = self._directional_component(direction, slope_direction_score)

        score = 0.0
        score += directional_momentum * 0.25
        score += directional_slope * 0.25
        score += float(structure_score) * 0.25
        score += (1.0 - min(1.0, float(exhaustion_score))) * 0.15
        score += (1.0 - float(consolidation_score)) * 0.10

        if float(sr_score) > 0.6:
            score += 0.05

        return self._normalize_unit_score(score)

    def _calculate_reversal_risk(
        self,
        *,
        structure_score: UnitScore,
        momentum_direction_score: SignedScore,
        exhaustion_score: UnitScore,
        pullback_depth: UnitScore,
        consolidation_score: UnitScore,
    ) -> UnitScore:
        score = 0.0
        score += float(exhaustion_score) * 0.35
        score += min(1.0, float(pullback_depth) / self.config.pullback_depth_threshold) * 0.25
        score += float(consolidation_score) * 0.15
        score += (1.0 - float(structure_score)) * 0.15
        score += (1.0 - abs(float(momentum_direction_score))) * 0.10
        return self._normalize_unit_score(score)

    def _calculate_strength(
        self,
        *,
        momentum_direction_score: SignedScore,
        slope_direction_score: SignedScore,
        structure_score: UnitScore,
        sr_score: UnitScore,
    ) -> UnitScore:
        score = 0.0
        score += abs(float(momentum_direction_score)) * 0.30
        score += abs(float(slope_direction_score)) * 0.25
        score += float(structure_score) * 0.30
        score += float(sr_score) * 0.15
        return self._normalize_unit_score(score)

    def _calculate_confidence(
        self,
        *,
        direction: TrendDirection,
        structure_bias: MarketBias,
        structure_score: UnitScore,
        momentum_direction_score: SignedScore,
        slope_direction_score: SignedScore,
    ) -> UnitScore:
        if direction in {TrendDirection.UNKNOWN, TrendDirection.NEUTRAL}:
            return UnitScore(0.0)

        confidence = 0.30
        confidence += float(structure_score) * 0.35
        confidence += abs(float(momentum_direction_score)) * 0.20
        confidence += abs(float(slope_direction_score)) * 0.15

        if self._is_direction_aligned_with_bias(direction, structure_bias):
            confidence += 0.10

        return self._normalize_unit_score(confidence)

    def _derive_direction(
        self,
        *,
        momentum_direction_score: SignedScore,
        slope_direction_score: SignedScore,
        structure_bias: MarketBias,
    ) -> TrendDirection:
        signed_score = float(momentum_direction_score) * 0.45 + float(slope_direction_score) * 0.35
        signed_score += self._structure_bias_signed_component(structure_bias)

        if signed_score >= self.config.direction_positive_threshold:
            return TrendDirection.BULLISH
        if signed_score <= self.config.direction_negative_threshold:
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL

    def _derive_regime(
        self,
        *,
        direction: TrendDirection,
        strength: UnitScore,
        pullback_depth: UnitScore,
        consolidation_score: UnitScore,
        exhaustion_score: UnitScore,
        reversal_risk: UnitScore,
    ) -> TrendRegime:
        if direction in {TrendDirection.UNKNOWN, TrendDirection.NEUTRAL}:
            if float(consolidation_score) >= 0.55:
                return TrendRegime.CONSOLIDATING
            return TrendRegime.UNKNOWN

        if float(reversal_risk) >= self.config.reversal_risk_threshold:
            return TrendRegime.REVERSING
        if float(exhaustion_score) >= self.config.exhaustion_threshold:
            return TrendRegime.EXHAUSTED
        if float(pullback_depth) >= self.config.pullback_depth_threshold:
            return TrendRegime.PULLBACK
        if float(consolidation_score) >= 0.60:
            return TrendRegime.CONSOLIDATING
        if float(strength) >= self.config.trend_strength_threshold:
            return TrendRegime.TRENDING
        return TrendRegime.UNKNOWN

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def _calculate_internal_external_alignment(self) -> UnitScore:
        internal = self._state.internal
        external = self._state.external

        score = 0.0
        if internal.direction == external.direction and internal.direction in {
            TrendDirection.BULLISH,
            TrendDirection.BEARISH,
        }:
            score += 0.55

        if internal.regime == external.regime:
            score += 0.15

        score += min(float(internal.confidence), float(external.confidence)) * 0.30
        return self._normalize_unit_score(score)

    def _calculate_higher_timeframe_alignment(self) -> Tuple[UnitScore, bool]:
        if not self._latest_market_structure:
            self._log_missing_mtf()
            return UnitScore(0.0), False

        mtf = self._latest_market_structure.get("mtf_alignment")
        if not isinstance(mtf, Mapping):
            self._log_missing_mtf()
            return UnitScore(0.0), False

        alignment_score = float(mtf.get("alignment_score", 0.0))
        self._missing_mtf_logged = False
        return self._normalize_unit_score(alignment_score), True

    def _log_missing_mtf(self) -> None:
        if not self.config.log_missing_mtf_once:
            self.logger.debug(
                "Higher timeframe alignment is unavailable",
                extra={"symbol": self.symbol, "timeframe": self.timeframe},
            )
            return

        if not self._missing_mtf_logged:
            self.logger.debug(
                "Higher timeframe alignment is unavailable",
                extra={"symbol": self.symbol, "timeframe": self.timeframe},
            )
            self._missing_mtf_logged = True

    # ------------------------------------------------------------------
    # Extra derived scores
    # ------------------------------------------------------------------

    def trend_quality_score(self, layer: StructureLayer) -> UnitScore:
        layer_state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

        score = 0.0
        score += float(layer_state.strength) * 0.35
        score += float(layer_state.confidence) * 0.25
        score += float(layer_state.continuation_probability) * 0.20
        score += (1.0 - float(layer_state.exhaustion_score)) * 0.10
        score += (1.0 - float(layer_state.reversal_risk)) * 0.10
        return self._normalize_unit_score(score)

    def trend_tradeability_score(self, layer: StructureLayer) -> UnitScore:
        layer_state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

        score = 0.0
        score += float(layer_state.strength) * 0.30
        score += float(layer_state.confidence) * 0.20
        score += float(layer_state.continuation_probability) * 0.20
        score += float(self._state.internal_external_alignment) * 0.15
        score += float(self._state.higher_timeframe_alignment) * 0.15
        score -= float(layer_state.exhaustion_score) * 0.15
        score -= float(layer_state.reversal_risk) * 0.15

        return self._normalize_unit_score(score)

    def pullback_opportunity_score(self, layer: StructureLayer) -> UnitScore:
        layer_state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

        if not layer_state.in_pullback:
            return UnitScore(0.0)

        score = 0.0
        score += float(layer_state.confidence) * 0.25
        score += float(layer_state.structure_score) * 0.25
        score += float(layer_state.continuation_probability) * 0.20
        score += (1.0 - float(layer_state.reversal_risk)) * 0.20
        score += (1.0 - float(layer_state.exhaustion_score)) * 0.10

        return self._normalize_unit_score(score)

    def reversal_setup_score(self, layer: StructureLayer) -> UnitScore:
        layer_state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

        score = 0.0
        score += float(layer_state.reversal_risk) * 0.40
        score += float(layer_state.exhaustion_score) * 0.30
        score += float(layer_state.consolidation_score) * 0.10
        score += (1.0 - float(layer_state.continuation_probability)) * 0.20

        return self._normalize_unit_score(score)

    def regime_summary(self, layer: StructureLayer) -> Dict[str, Any]:
        layer_state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        return {
            "direction": layer_state.direction.value,
            "regime": layer_state.regime.value,
            "strength": float(layer_state.strength),
            "confidence": float(layer_state.confidence),
            "trend_quality_score": float(self.trend_quality_score(layer)),
            "trend_tradeability_score": float(self.trend_tradeability_score(layer)),
            "pullback_opportunity_score": float(self.pullback_opportunity_score(layer)),
            "reversal_setup_score": float(self.reversal_setup_score(layer)),
            "continuation_probability": float(layer_state.continuation_probability),
            "reversal_risk": float(layer_state.reversal_risk),
            "exhaustion_score": float(layer_state.exhaustion_score),
        }

    # ------------------------------------------------------------------
    # EventBus
    # ------------------------------------------------------------------

    def _append_signal(self, signal: TrendSignal) -> None:
        self._signals.append(signal)
        self._emit_signal(signal)

    def _emit_signal(self, signal: TrendSignal) -> None:
        if not self.config.emit_events or self.event_bus is None:
            return

        event_name = f"{self.config.event_namespace}.{signal.event_type.value}"
        payload = {
            "source": self.config.event_namespace,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "signal": self._signal_to_dict(signal),
            "state": self.snapshot(),
        }
        self._dispatch_eventbus_event(event_name, payload)

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
        self._dispatch_eventbus_event(event_name, payload)

    def _dispatch_eventbus_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        method = None
        method_name = None

        for candidate in ("emit", "publish", "dispatch"):
            if hasattr(self.event_bus, candidate):
                method = getattr(self.event_bus, candidate)
                method_name = candidate
                break

        if method is None:
            self.logger.warning(
                "EventBus provided but no supported method found",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "event_name": event_name,
                },
            )
            return

        try:
            result = method(event_name, payload)

            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    self.logger.warning(
                        "EventBus returned awaitable but no running loop is available",
                        extra={
                            "symbol": self.symbol,
                            "timeframe": self.timeframe,
                            "event_name": event_name,
                            "method_name": method_name,
                        },
                    )
                    return

                task = loop.create_task(result)
                self._track_task(task)

        except Exception as exc:
            self.logger.exception(
                "Failed to dispatch event bus event",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "event_name": event_name,
                    "method_name": method_name,
                    "error": str(exc),
                },
            )

    def _track_task(self, task: asyncio.Task[Any]) -> None:
        self._pending_tasks.add(task)

        def _done_callback(done_task: asyncio.Task[Any]) -> None:
            self._pending_tasks.discard(done_task)
            self._handle_eventbus_task_result(done_task)

        task.add_done_callback(_done_callback)

    def _handle_eventbus_task_result(self, task: asyncio.Task[Any]) -> None:
        try:
            task.result()
        except Exception as exc:
            self.logger.exception(
                "Async EventBus task failed",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "error": str(exc),
                },
            )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _clear_layer_state_no_candles(self, layer_state: TrendLayerState) -> None:
        layer_state.direction = TrendDirection.UNKNOWN
        layer_state.regime = TrendRegime.UNKNOWN
        layer_state.strength = UnitScore(0.0)
        layer_state.confidence = UnitScore(0.0)
        layer_state.momentum_direction_score = SignedScore(0.0)
        layer_state.slope_direction_score = SignedScore(0.0)
        layer_state.structure_score = UnitScore(0.0)
        layer_state.continuation_probability = UnitScore(0.0)
        layer_state.reversal_risk = UnitScore(0.0)
        layer_state.exhaustion_score = UnitScore(0.0)
        layer_state.pullback_depth = UnitScore(0.0)
        layer_state.consolidation_score = UnitScore(0.0)
        layer_state.is_accelerating = False
        layer_state.is_exhausted = False
        layer_state.in_pullback = False
        layer_state.is_aligned_with_structure = False
        layer_state.metadata = {
            "reason": "no_candles",
            "close_count": 0,
        }

    def _create_signal(
        self,
        *,
        layer: StructureLayer,
        event_type: TrendEventType,
        direction: TrendDirection,
        strength: float,
        confidence: float,
        regime: TrendRegime,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrendSignal:
        return TrendSignal(
            signal_id=self._new_id(),
            timestamp=self._state.last_update or datetime.now(timezone.utc),
            symbol=self.symbol,
            timeframe=self.timeframe,
            layer=layer,
            event_type=event_type,
            direction=direction,
            strength=float(self._normalize_unit_score(strength)),
            confidence=float(self._normalize_unit_score(confidence)),
            regime=regime,
            price=self._state.last_price,
            metadata=metadata or {},
        )

    def _is_direction_aligned_with_bias(self, direction: TrendDirection, bias: MarketBias) -> bool:
        return (
            (direction == TrendDirection.BULLISH and bias == MarketBias.BULLISH)
            or (direction == TrendDirection.BEARISH and bias == MarketBias.BEARISH)
        )

    def _structure_bias_signed_component(self, bias: MarketBias) -> float:
        if bias == MarketBias.BULLISH:
            return self.config.structure_bias_weight
        if bias == MarketBias.BEARISH:
            return -self.config.structure_bias_weight
        return 0.0

    def _directional_component(self, direction: TrendDirection, score: SignedScore) -> float:
        if direction == TrendDirection.BULLISH:
            return max(0.0, float(score))
        if direction == TrendDirection.BEARISH:
            return max(0.0, -float(score))
        return 0.0

    def _calculate_atr(self, window: int) -> float:
        candles = list(self._candles)
        if len(candles) < window + 1:
            return 0.0

        true_ranges: List[float] = []
        for i in range(1, min(len(candles), window + 1)):
            current = candles[-i]
            previous = candles[-i - 1]
            tr = max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
            true_ranges.append(tr)

        return self._safe_mean(true_ranges) if true_ranges else 0.0

    def _linear_slope(self, values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = self._safe_mean(values)

        numerator = 0.0
        denominator = 0.0
        for i, y in enumerate(values):
            dx = i - x_mean
            dy = y - y_mean
            numerator += dx * dy
            denominator += dx * dx

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        base = abs(y_mean) if y_mean != 0 else 1.0
        return slope / base

    def _impulse_stretch_score(self, candles: Sequence[Candle], atr: float) -> UnitScore:
        if len(candles) < 3 or atr <= 0:
            return UnitScore(0.0)

        bodies = [c.body_size for c in candles[-3:]]
        avg_body = self._safe_mean(bodies)
        stretch = avg_body / atr
        return self._normalize_unit_score(stretch / 2.5)

    def _wick_pressure_score(self, candles: Sequence[Candle]) -> UnitScore:
        if not candles:
            return UnitScore(0.0)

        pressures: List[float] = []
        for candle in candles[-5:]:
            if candle.range_size <= 0:
                continue
            upper_wick = candle.high - candle.body_high
            lower_wick = candle.body_low - candle.low
            wick_ratio = (upper_wick + lower_wick) / candle.range_size
            pressures.append(min(1.0, wick_ratio))

        return self._normalize_unit_score(self._safe_mean(pressures) if pressures else 0.0)

    def _deceleration_score(self, candles: Sequence[Candle]) -> UnitScore:
        if len(candles) < 4:
            return UnitScore(0.0)

        recent = candles[-4:]
        body_sizes = [c.body_size for c in recent]
        range_sizes = [c.range_size for c in recent]

        first_body = max(body_sizes[0], 1e-12)
        first_range = max(range_sizes[0], 1e-12)

        body_ratio = body_sizes[-1] / first_body
        range_ratio = range_sizes[-1] / first_range

        body_deceleration = max(0.0, min(1.0, 1.0 - body_ratio))
        range_deceleration = max(0.0, min(1.0, 1.0 - range_ratio))

        body_monotonicity = self._decreasing_sequence_score(body_sizes)
        range_monotonicity = self._decreasing_sequence_score(range_sizes)

        score = (
            body_deceleration * 0.35
            + range_deceleration * 0.35
            + body_monotonicity * 0.15
            + range_monotonicity * 0.15
        )
        return self._normalize_unit_score(score)

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
        open_f = float(open_)
        high_f = float(high)
        low_f = float(low)
        close_f = float(close)
        volume_f = float(volume)

        candle = Candle(
            timestamp=dt,
            open=open_f,
            high=high_f,
            low=low_f,
            close=close_f,
            volume=volume_f,
            index=self._global_candle_index,
        )
        self._global_candle_index += 1
        return candle

    @staticmethod
    def _normalize_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
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

    @staticmethod
    def _normalize_signed_score(value: float) -> SignedScore:
        if value >= 0:
            return SignedScore(min(1.0, value))
        return SignedScore(max(-1.0, value))

    @staticmethod
    def _normalize_unit_score(value: float) -> UnitScore:
        return UnitScore(max(0.0, min(1.0, value)))

    @staticmethod
    def _decreasing_sequence_score(values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0

        decreases = 0
        total = len(values) - 1
        for i in range(1, len(values)):
            if values[i] <= values[i - 1]:
                decreases += 1
        return decreases / total if total > 0 else 0.0

    @staticmethod
    def _deduplicate_preserve_order(signals: Iterable[TrendSignal]) -> List[TrendSignal]:
        seen: Set[TrendEventType] = set()
        result: List[TrendSignal] = []
        for signal in signals:
            if signal.event_type in seen:
                continue
            seen.add(signal.event_type)
            result.append(signal)
        return result

    @staticmethod
    def _new_id() -> str:
        return uuid4().hex

    def _serialize_config(self) -> Dict[str, Any]:
        return {
            "max_candles": self.config.max_candles,
            "max_events": self.config.max_events,
            "short_window": self.config.short_window,
            "medium_window": self.config.medium_window,
            "long_window": self.config.long_window,
            "atr_window": self.config.atr_window,
            "trend_strength_threshold": self.config.trend_strength_threshold,
            "acceleration_threshold": self.config.acceleration_threshold,
            "exhaustion_threshold": self.config.exhaustion_threshold,
            "reversal_risk_threshold": self.config.reversal_risk_threshold,
            "pullback_depth_threshold": self.config.pullback_depth_threshold,
            "momentum_slope_threshold": self.config.momentum_slope_threshold,
            "consolidation_range_threshold": self.config.consolidation_range_threshold,
            "direction_positive_threshold": self.config.direction_positive_threshold,
            "direction_negative_threshold": self.config.direction_negative_threshold,
            "structure_bias_weight": self.config.structure_bias_weight,
            "emit_events": self.config.emit_events,
            "event_namespace": self.config.event_namespace,
            "publish_snapshots": self.config.publish_snapshots,
            "log_missing_mtf_once": self.config.log_missing_mtf_once,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _signal_to_dict(self, signal: Optional[TrendSignal]) -> Optional[Dict[str, Any]]:
        if signal is None:
            return None

        return {
            "signal_id": signal.signal_id,
            "timestamp": signal.timestamp.isoformat(),
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "layer": signal.layer.value,
            "event_type": signal.event_type.value,
            "direction": signal.direction.value,
            "strength": signal.strength,
            "confidence": signal.confidence,
            "regime": signal.regime.value,
            "price": signal.price,
            "metadata": dict(signal.metadata),
        }

    def _layer_state_to_dict(self, state: TrendLayerState) -> Dict[str, Any]:
        return {
            "layer": state.layer.value,
            "direction": state.direction.value,
            "regime": state.regime.value,
            "strength": float(state.strength),
            "confidence": float(state.confidence),
            "momentum_direction_score": float(state.momentum_direction_score),
            "slope_direction_score": float(state.slope_direction_score),
            "structure_score": float(state.structure_score),
            "continuation_probability": float(state.continuation_probability),
            "reversal_risk": float(state.reversal_risk),
            "exhaustion_score": float(state.exhaustion_score),
            "pullback_depth": float(state.pullback_depth),
            "consolidation_score": float(state.consolidation_score),
            "is_accelerating": state.is_accelerating,
            "is_exhausted": state.is_exhausted,
            "in_pullback": state.in_pullback,
            "is_aligned_with_structure": state.is_aligned_with_structure,
            "last_signal": self._signal_to_dict(state.last_signal),
            "metadata": dict(state.metadata),
        }