from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from core.logger import get_logger

from analytics.price_action.market_structure import MarketBias, StructureLayer


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

    emit_events: bool = True
    event_namespace: str = "price_action.trend_detection"
    publish_snapshots: bool = False

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

    strength: float = 0.0
    confidence: float = 0.0
    momentum_score: float = 0.0
    slope_score: float = 0.0
    structure_score: float = 0.0
    continuation_probability: float = 0.0
    reversal_risk: float = 0.0
    exhaustion_score: float = 0.0
    pullback_depth: float = 0.0
    consolidation_score: float = 0.0

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

    internal_external_alignment: float = 0.0
    higher_timeframe_alignment: float = 0.0
    overall_trend_score: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)


class TrendDetector:
    """
    Stateful trend detector.

    Inputs:
    - candles
    - market_structure snapshot/context
    - support_resistance snapshot/context

    Outputs:
    - internal trend state
    - external trend state
    - alignment scores
    - trend event signals
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

        self._global_candle_index = 0

        self._latest_market_structure: Dict[str, Any] = {}
        self._latest_support_resistance: Dict[str, Any] = {}

        self._state = TrendState(
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        self.logger.info(
            "Initialized TrendDetector",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "config": asdict(self.config),
            },
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def reset(self) -> None:
        self._candles.clear()
        self._signals.clear()
        self._global_candle_index = 0
        self._latest_market_structure = {}
        self._latest_support_resistance = {}
        self._state = TrendState(symbol=self.symbol, timeframe=self.timeframe)

        self.logger.info(
            "TrendDetector reset",
            extra={"symbol": self.symbol, "timeframe": self.timeframe},
        )

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

        new_signals: List[TrendSignal] = []
        new_signals.extend(self._refresh_layer(StructureLayer.INTERNAL))
        new_signals.extend(self._refresh_layer(StructureLayer.EXTERNAL))
        self._refresh_global_state()
        new_signals.extend(self._detect_cross_layer_events())

        if self.config.publish_snapshots:
            self._publish_snapshot()

        self.logger.debug(
            "Trend detection updated",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "candles_count": len(self._candles),
                "signals_generated": len(new_signals),
                "overall_trend_score": self._state.overall_trend_score,
            },
        )

        return {
            "state": self.snapshot(),
            "new_signals": [self._signal_to_dict(signal) for signal in new_signals],
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "symbol": self._state.symbol,
            "timeframe": self._state.timeframe,
            "last_price": self._state.last_price,
            "last_update": self._state.last_update.isoformat() if self._state.last_update else None,
            "internal": self._layer_state_to_dict(self._state.internal),
            "external": self._layer_state_to_dict(self._state.external),
            "internal_external_alignment": self._state.internal_external_alignment,
            "higher_timeframe_alignment": self._state.higher_timeframe_alignment,
            "overall_trend_score": self._state.overall_trend_score,
            "metadata": dict(self._state.metadata),
        }

    def get_state(self) -> TrendState:
        return self._state

    def get_signals(self) -> List[TrendSignal]:
        return list(self._signals)

    # ---------------------------------------------------------------------
    # Core layer refresh
    # ---------------------------------------------------------------------

    def _refresh_layer(self, layer: StructureLayer) -> List[TrendSignal]:
        layer_state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        previous_direction = layer_state.direction
        previous_regime = layer_state.regime
        previous_strength = layer_state.strength
        previous_pullback = layer_state.in_pullback

        closes = [c.close for c in self._candles]
        highs = [c.high for c in self._candles]
        lows = [c.low for c in self._candles]

        momentum_score = self._calculate_momentum_score(closes)
        slope_score = self._calculate_slope_score(closes, self.config.short_window, self.config.medium_window)
        structure_score, structure_bias = self._extract_structure_score(layer)
        sr_score = self._extract_sr_context_score(layer)
        pullback_depth = self._calculate_pullback_depth(closes)
        consolidation_score = self._calculate_consolidation_score(highs, lows, closes)
        exhaustion_score = self._calculate_exhaustion_score()
        continuation_probability = self._calculate_continuation_probability(
            momentum_score=momentum_score,
            slope_score=slope_score,
            structure_score=structure_score,
            sr_score=sr_score,
            exhaustion_score=exhaustion_score,
            consolidation_score=consolidation_score,
        )
        reversal_risk = self._calculate_reversal_risk(
            structure_score=structure_score,
            momentum_score=momentum_score,
            exhaustion_score=exhaustion_score,
            pullback_depth=pullback_depth,
            consolidation_score=consolidation_score,
        )

        direction = self._derive_direction(
            momentum_score=momentum_score,
            slope_score=slope_score,
            structure_bias=structure_bias,
        )
        strength = self._calculate_strength(
            momentum_score=momentum_score,
            slope_score=slope_score,
            structure_score=structure_score,
            sr_score=sr_score,
        )
        confidence = self._calculate_confidence(
            direction=direction,
            structure_bias=structure_bias,
            structure_score=structure_score,
            momentum_score=momentum_score,
            slope_score=slope_score,
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
        layer_state.momentum_score = momentum_score
        layer_state.slope_score = slope_score
        layer_state.structure_score = structure_score
        layer_state.continuation_probability = continuation_probability
        layer_state.reversal_risk = reversal_risk
        layer_state.exhaustion_score = exhaustion_score
        layer_state.pullback_depth = pullback_depth
        layer_state.consolidation_score = consolidation_score
        layer_state.is_accelerating = strength >= self.config.acceleration_threshold and momentum_score >= 0.65
        layer_state.is_exhausted = exhaustion_score >= self.config.exhaustion_threshold
        layer_state.in_pullback = regime == TrendRegime.PULLBACK
        layer_state.is_aligned_with_structure = self._is_direction_aligned_with_bias(direction, structure_bias)
        layer_state.metadata = {
            "structure_bias": structure_bias.value,
            "sr_score": sr_score,
            "close_count": len(closes),
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
        self._state.higher_timeframe_alignment = self._calculate_higher_timeframe_alignment()
        self._state.overall_trend_score = max(
            0.0,
            min(
                1.0,
                (
                    internal.strength * 0.35
                    + external.strength * 0.40
                    + self._state.internal_external_alignment * 0.15
                    + self._state.higher_timeframe_alignment * 0.10
                ),
            ),
        )

        self._state.metadata = {
            "signal_count": len(self._signals),
            "market_structure_available": bool(self._latest_market_structure),
            "support_resistance_available": bool(self._latest_support_resistance),
        }

    # ---------------------------------------------------------------------
    # Event detection
    # ---------------------------------------------------------------------

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
        signals: List[TrendSignal] = []

        current_direction = current_state.direction
        current_regime = current_state.regime
        current_strength = current_state.strength

        if previous_direction in {TrendDirection.UNKNOWN, TrendDirection.NEUTRAL} and current_direction in {
            TrendDirection.BULLISH,
            TrendDirection.BEARISH,
        }:
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_STARTED,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=current_state.confidence,
                    regime=current_regime,
                    metadata={"from_direction": previous_direction.value},
                )
            )

        elif previous_direction in {TrendDirection.BULLISH, TrendDirection.BEARISH} and current_direction != previous_direction:
            if current_direction in {TrendDirection.BULLISH, TrendDirection.BEARISH}:
                signals.append(
                    self._create_signal(
                        layer=layer,
                        event_type=TrendEventType.TREND_REVERSAL,
                        direction=current_direction,
                        strength=current_strength,
                        confidence=current_state.confidence,
                        regime=current_regime,
                        metadata={"from_direction": previous_direction.value},
                    )
                )

        elif current_direction in {TrendDirection.BULLISH, TrendDirection.BEARISH} and current_strength > previous_strength + 0.08:
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_CONTINUATION,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=current_state.confidence,
                    regime=current_regime,
                    metadata={"previous_strength": previous_strength},
                )
            )

        if current_state.is_accelerating and current_strength > previous_strength:
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_ACCELERATION,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=current_state.confidence,
                    regime=current_regime,
                    metadata={"momentum_score": current_state.momentum_score},
                )
            )

        if current_strength < previous_strength - 0.10 and current_direction in {TrendDirection.BULLISH, TrendDirection.BEARISH}:
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_WEAKENING,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=current_state.confidence,
                    regime=current_regime,
                    metadata={"previous_strength": previous_strength},
                )
            )

        if not previous_pullback and current_state.in_pullback:
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.PULLBACK_STARTED,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=current_state.confidence,
                    regime=current_regime,
                    metadata={"pullback_depth": current_state.pullback_depth},
                )
            )
        elif previous_pullback and not current_state.in_pullback:
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.PULLBACK_ENDED,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=current_state.confidence,
                    regime=current_regime,
                    metadata={"pullback_depth": current_state.pullback_depth},
                )
            )

        if current_state.is_exhausted:
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_EXHAUSTION,
                    direction=current_direction,
                    strength=current_strength,
                    confidence=current_state.confidence,
                    regime=current_regime,
                    metadata={"exhaustion_score": current_state.exhaustion_score},
                )
            )

        for signal in signals:
            self._append_signal(signal)

        current_state.last_signal = signals[-1] if signals else current_state.last_signal
        return signals

    def _detect_cross_layer_events(self) -> List[TrendSignal]:
        signals: List[TrendSignal] = []

        internal = self._state.internal
        external = self._state.external

        if internal.direction in {TrendDirection.BULLISH, TrendDirection.BEARISH} and internal.direction == external.direction:
            signal = self._create_signal(
                layer=StructureLayer.EXTERNAL,
                event_type=TrendEventType.TREND_ALIGNMENT,
                direction=external.direction,
                strength=max(internal.strength, external.strength),
                confidence=(internal.confidence + external.confidence) / 2.0,
                regime=external.regime,
                metadata={
                    "internal_direction": internal.direction.value,
                    "external_direction": external.direction.value,
                    "alignment_score": self._state.internal_external_alignment,
                },
            )
            self._append_signal(signal)
            signals.append(signal)

        elif (
            internal.direction in {TrendDirection.BULLISH, TrendDirection.BEARISH}
            and external.direction in {TrendDirection.BULLISH, TrendDirection.BEARISH}
            and internal.direction != external.direction
        ):
            signal = self._create_signal(
                layer=StructureLayer.EXTERNAL,
                event_type=TrendEventType.TREND_DISAGREEMENT,
                direction=external.direction,
                strength=external.strength,
                confidence=external.confidence,
                regime=external.regime,
                metadata={
                    "internal_direction": internal.direction.value,
                    "external_direction": external.direction.value,
                    "alignment_score": self._state.internal_external_alignment,
                },
            )
            self._append_signal(signal)
            signals.append(signal)

        return signals

    # ---------------------------------------------------------------------
    # Scores
    # ---------------------------------------------------------------------

    def _extract_structure_score(self, layer: StructureLayer) -> Tuple[float, MarketBias]:
        layer_key = layer.value
        if not self._latest_market_structure:
            return 0.0, MarketBias.UNKNOWN

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
        return max(0.0, min(1.0, score)), bias

    def _extract_sr_context_score(self, layer: StructureLayer) -> float:
        if not self._latest_support_resistance:
            return 0.0

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

        return max(0.0, min(1.0, score))

    def _calculate_momentum_score(self, closes: Sequence[float]) -> float:
        if len(closes) < self.config.medium_window:
            return 0.0

        short_ma = self._safe_mean(closes[-self.config.short_window:])
        medium_ma = self._safe_mean(closes[-self.config.medium_window:])
        long_window = min(self.config.long_window, len(closes))
        long_ma = self._safe_mean(closes[-long_window:])

        if long_ma == 0:
            return 0.0

        separation_1 = (short_ma - medium_ma) / abs(long_ma)
        separation_2 = (medium_ma - long_ma) / abs(long_ma)
        raw = (separation_1 * 80.0) + (separation_2 * 50.0)

        return self._normalize_signed_score(raw)

    def _calculate_slope_score(self, closes: Sequence[float], short_window: int, medium_window: int) -> float:
        if len(closes) < medium_window:
            return 0.0

        short_slope = self._linear_slope(closes[-short_window:])
        medium_slope = self._linear_slope(closes[-medium_window:])
        combined = short_slope * 0.65 + medium_slope * 0.35

        return self._normalize_signed_score(combined * 200.0)

    def _calculate_pullback_depth(self, closes: Sequence[float]) -> float:
        if len(closes) < self.config.medium_window:
            return 0.0

        recent = closes[-self.config.medium_window:]
        max_close = max(recent)
        min_close = min(recent)
        last = recent[-1]

        if max_close <= 0 or min_close <= 0:
            return 0.0

        bullish_pullback = (max_close - last) / max_close
        bearish_pullback = (last - min_close) / min_close
        return max(bullish_pullback, bearish_pullback)

    def _calculate_consolidation_score(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
    ) -> float:
        if len(closes) < self.config.short_window:
            return 0.0

        window = min(self.config.short_window, len(closes))
        local_high = max(highs[-window:])
        local_low = min(lows[-window:])
        local_mean = self._safe_mean(closes[-window:])

        if local_mean == 0:
            return 0.0

        compression = (local_high - local_low) / abs(local_mean)
        raw = 1.0 - min(1.0, compression / self.config.consolidation_range_threshold)
        return max(0.0, min(1.0, raw))

    def _calculate_exhaustion_score(self) -> float:
        if len(self._candles) < max(self.config.atr_window + 2, 10):
            return 0.0

        recent = list(self._candles)[-min(12, len(self._candles)):]
        atr = self._calculate_atr(self.config.atr_window)
        if atr <= 0:
            return 0.0

        impulse_stretch = self.impulse_stretch_score(recent, atr)
        wick_pressure = self.wick_pressure_score(recent)
        deceleration = self.deceleration_score(recent)

        score = impulse_stretch * 0.45 + wick_pressure * 0.30 + deceleration * 0.25
        return max(0.0, min(1.0, score))

    def _calculate_continuation_probability(
        self,
        *,
        momentum_score: float,
        slope_score: float,
        structure_score: float,
        sr_score: float,
        exhaustion_score: float,
        consolidation_score: float,
    ) -> float:
        score = 0.0
        score += abs(momentum_score) * 0.25
        score += abs(slope_score) * 0.25
        score += structure_score * 0.25
        score += (1.0 - min(1.0, exhaustion_score)) * 0.15
        score += (1.0 - consolidation_score) * 0.10

        if sr_score > 0.6:
            score += 0.05

        return max(0.0, min(1.0, score))

    def _calculate_reversal_risk(
        self,
        *,
        structure_score: float,
        momentum_score: float,
        exhaustion_score: float,
        pullback_depth: float,
        consolidation_score: float,
    ) -> float:
        score = 0.0
        score += exhaustion_score * 0.35
        score += min(1.0, pullback_depth / max(self.config.pullback_depth_threshold, 1e-12)) * 0.25
        score += consolidation_score * 0.15
        score += (1.0 - structure_score) * 0.15
        score += (1.0 - abs(momentum_score)) * 0.10
        return max(0.0, min(1.0, score))

    def _calculate_strength(
        self,
        *,
        momentum_score: float,
        slope_score: float,
        structure_score: float,
        sr_score: float,
    ) -> float:
        score = 0.0
        score += abs(momentum_score) * 0.30
        score += abs(slope_score) * 0.25
        score += structure_score * 0.30
        score += sr_score * 0.15
        return max(0.0, min(1.0, score))

    def _calculate_confidence(
        self,
        *,
        direction: TrendDirection,
        structure_bias: MarketBias,
        structure_score: float,
        momentum_score: float,
        slope_score: float,
    ) -> float:
        if direction in {TrendDirection.UNKNOWN, TrendDirection.NEUTRAL}:
            return 0.0

        confidence = 0.30
        confidence += structure_score * 0.35
        confidence += abs(momentum_score) * 0.20
        confidence += abs(slope_score) * 0.15

        if self._is_direction_aligned_with_bias(direction, structure_bias):
            confidence += 0.10

        return max(0.0, min(1.0, confidence))

    def _derive_direction(
        self,
        *,
        momentum_score: float,
        slope_score: float,
        structure_bias: MarketBias,
    ) -> TrendDirection:
        signed_score = momentum_score * 0.45 + slope_score * 0.35

        if structure_bias == MarketBias.BULLISH:
            signed_score += 0.20
        elif structure_bias == MarketBias.BEARISH:
            signed_score -= 0.20

        if signed_score >= 0.22:
            return TrendDirection.BULLISH
        if signed_score <= -0.22:
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL

    def _derive_regime(
        self,
        *,
        direction: TrendDirection,
        strength: float,
        pullback_depth: float,
        consolidation_score: float,
        exhaustion_score: float,
        reversal_risk: float,
    ) -> TrendRegime:
        if direction in {TrendDirection.UNKNOWN, TrendDirection.NEUTRAL}:
            if consolidation_score >= 0.55:
                return TrendRegime.CONSOLIDATING
            return TrendRegime.UNKNOWN

        if reversal_risk >= self.config.reversal_risk_threshold:
            return TrendRegime.REVERSING

        if exhaustion_score >= self.config.exhaustion_threshold:
            return TrendRegime.EXHAUSTED

        if pullback_depth >= self.config.pullback_depth_threshold:
            return TrendRegime.PULLBACK

        if consolidation_score >= 0.60:
            return TrendRegime.CONSOLIDATING

        if strength >= self.config.trend_strength_threshold:
            return TrendRegime.TRENDING

        return TrendRegime.UNKNOWN

    # ---------------------------------------------------------------------
    # Alignment
    # ---------------------------------------------------------------------

    def _calculate_internal_external_alignment(self) -> float:
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

        score += min(internal.confidence, external.confidence) * 0.30
        return max(0.0, min(1.0, score))

    def _calculate_higher_timeframe_alignment(self) -> float:
        if not self._latest_market_structure:
            return 0.0

        mtf = self._latest_market_structure.get("mtf_alignment", {})
        return float(mtf.get("alignment_score", 0.0))

    # ---------------------------------------------------------------------
    # Fancy helper functions
    # ---------------------------------------------------------------------

    def impulse_stretch_score(self, candles: Sequence[Candle], atr: float) -> float:
        """
        Оцінює, наскільки останній імпульс уже 'розтягнувся'
        відносно ATR. Корисно для exhaustion / late-entry filter.
        """
        if len(candles) < 3 or atr <= 0:
            return 0.0

        bodies = [c.body_size for c in candles[-3:]]
        avg_body = self._safe_mean(bodies)
        stretch = avg_body / atr
        return max(0.0, min(1.0, stretch / 2.5))

    def wick_pressure_score(self, candles: Sequence[Candle]) -> float:
        """
        Високий score означає, що в останніх свічках багато wick-pressure,
        тобто ринок починає втрачати чистий directional impulse.
        """
        if not candles:
            return 0.0

        pressures: List[float] = []
        for candle in candles[-5:]:
            if candle.range_size <= 0:
                continue
            upper_wick = candle.high - candle.body_high
            lower_wick = candle.body_low - candle.low
            wick_ratio = (upper_wick + lower_wick) / candle.range_size
            pressures.append(min(1.0, wick_ratio))

        return self._safe_mean(pressures) if pressures else 0.0

    def deceleration_score(self, candles: Sequence[Candle]) -> float:
        """
        Виявляє ослаблення імпульсу:
        якщо body size та range size падають у кількох останніх свічках.
        """
        if len(candles) < 4:
            return 0.0

        recent = candles[-4:]
        body_sizes = [c.body_size for c in recent]
        range_sizes = [c.range_size for c in recent]

        body_drop = 1.0 if body_sizes[-1] < body_sizes[0] else 0.0
        range_drop = 1.0 if range_sizes[-1] < range_sizes[0] else 0.0

        body_ratio = body_sizes[-1] / max(body_sizes[0], 1e-12)
        range_ratio = range_sizes[-1] / max(range_sizes[0], 1e-12)

        score = (
            body_drop * (1.0 - min(1.0, body_ratio))
            + range_drop * (1.0 - min(1.0, range_ratio))
        ) / 2.0

        return max(0.0, min(1.0, score))

    def trend_quality_score(self, layer: StructureLayer) -> float:
        """
        Загальна якість тренду для strategy layer.
        Високий score = хороший, чистий, підтверджений тренд.
        """
        layer_state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

        score = 0.0
        score += layer_state.strength * 0.35
        score += layer_state.confidence * 0.25
        score += layer_state.continuation_probability * 0.20
        score += (1.0 - layer_state.exhaustion_score) * 0.10
        score += (1.0 - layer_state.reversal_risk) * 0.10

        return max(0.0, min(1.0, score))

    def trend_tradeability_score(self, layer: StructureLayer) -> float:
        """
        Оцінка 'чи варто взагалі лізти в угоду' по тренду.
        Це більш практичний score, ніж просто strength.
        """
        layer_state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

        score = 0.0
        score += layer_state.strength * 0.30
        score += layer_state.confidence * 0.20
        score += layer_state.continuation_probability * 0.20
        score += self._state.internal_external_alignment * 0.15
        score += self._state.higher_timeframe_alignment * 0.15

        score -= layer_state.exhaustion_score * 0.15
        score -= layer_state.reversal_risk * 0.15

        return max(0.0, min(1.0, score))

    def pullback_opportunity_score(self, layer: StructureLayer) -> float:
        """
        Високий score = pullback може бути здоровим continuation-entry,
        а не ознакою зламу тренду.
        """
        layer_state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

        if not layer_state.in_pullback:
            return 0.0

        score = 0.0
        score += layer_state.confidence * 0.25
        score += layer_state.structure_score * 0.25
        score += layer_state.continuation_probability * 0.20
        score += (1.0 - layer_state.reversal_risk) * 0.20
        score += (1.0 - layer_state.exhaustion_score) * 0.10

        return max(0.0, min(1.0, score))

    def reversal_setup_score(self, layer: StructureLayer) -> float:
        """
        Високий score = гарний кандидат на reversal setup.
        """
        layer_state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

        score = 0.0
        score += layer_state.reversal_risk * 0.40
        score += layer_state.exhaustion_score * 0.30
        score += layer_state.consolidation_score * 0.10
        score += (1.0 - layer_state.continuation_probability) * 0.20

        return max(0.0, min(1.0, score))

    def regime_summary(self, layer: StructureLayer) -> Dict[str, Any]:
        """
        Компактний summary для strategy/execution layer.
        """
        layer_state = self._state.internal if layer == StructureLayer.INTERNAL else self._state.external
        return {
            "direction": layer_state.direction.value,
            "regime": layer_state.regime.value,
            "strength": layer_state.strength,
            "confidence": layer_state.confidence,
            "trend_quality_score": self.trend_quality_score(layer),
            "trend_tradeability_score": self.trend_tradeability_score(layer),
            "pullback_opportunity_score": self.pullback_opportunity_score(layer),
            "reversal_setup_score": self.reversal_setup_score(layer),
            "continuation_probability": layer_state.continuation_probability,
            "reversal_risk": layer_state.reversal_risk,
            "exhaustion_score": layer_state.exhaustion_score,
        }

    # ---------------------------------------------------------------------
    # EventBus
    # ---------------------------------------------------------------------

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
                "Failed to emit trend signal",
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
                "Failed to publish trend snapshot",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "error": str(exc),
                },
            )

    # ---------------------------------------------------------------------
    # Utility logic
    # ---------------------------------------------------------------------

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
            strength=max(0.0, min(1.0, strength)),
            confidence=max(0.0, min(1.0, confidence)),
            regime=regime,
            price=self._state.last_price,
            metadata=metadata or {},
        )

    def _is_direction_aligned_with_bias(self, direction: TrendDirection, bias: MarketBias) -> bool:
        return (
            (direction == TrendDirection.BULLISH and bias == MarketBias.BULLISH)
            or (direction == TrendDirection.BEARISH and bias == MarketBias.BEARISH)
        )

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

    @staticmethod
    def _safe_mean(values: Sequence[float]) -> float:
        return mean(values) if values else 0.0

    @staticmethod
    def _normalize_signed_score(value: float) -> float:
        """
        Повертає score в межах [-1, 1]
        """
        if value >= 0:
            return min(1.0, value)
        return max(-1.0, value)

    @staticmethod
    def _new_id() -> str:
        return uuid4().hex

    # ---------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------

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
            "strength": state.strength,
            "confidence": state.confidence,
            "momentum_score": state.momentum_score,
            "slope_score": state.slope_score,
            "structure_score": state.structure_score,
            "continuation_probability": state.continuation_probability,
            "reversal_risk": state.reversal_risk,
            "exhaustion_score": state.exhaustion_score,
            "pullback_depth": state.pullback_depth,
            "consolidation_score": state.consolidation_score,
            "is_accelerating": state.is_accelerating,
            "is_exhausted": state.is_exhausted,
            "in_pullback": state.in_pullback,
            "is_aligned_with_structure": state.is_aligned_with_structure,
            "last_signal": self._signal_to_dict(state.last_signal),
            "metadata": dict(state.metadata),
        }