from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Mapping, Sequence
from uuid import uuid4

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from analytics.price_action.base import BasePriceActionConfig, BasePriceActionModule
from analytics.price_action.enums import (
    MarketBias,
    StructureLayer,
    TrendDirection,
    TrendEventType,
    TrendRegime,
)
from analytics.price_action.models import (
    Candle,
    SignedScore,
    TrendLayerState,
    TrendSignal,
    TrendState,
    UnitScore,
)


@dataclass(slots=True)
class TrendConfig(BasePriceActionConfig):
    max_candles: int = 500
    max_signals: int = 500

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
    support_resistance_weight: float = 0.10

    emit_events: bool = True
    event_namespace: str = "analytics.price_action.trend"
    publish_snapshots: bool = False
    log_missing_mtf_once: bool = True

    subscribe_market_structure: bool = True
    market_structure_updated_topic: str = "analytics.price_action.market_structure.updated"

    subscribe_support_resistance: bool = True
    support_resistance_updated_topic: str = "analytics.price_action.support_resistance.updated"

    def validate(self) -> None:
        super().validate()

        if self.max_candles < 100:
            raise ValueError("max_candles must be >= 100")
        if self.max_signals < 50:
            raise ValueError("max_signals must be >= 50")

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
        if not 0.0 <= self.support_resistance_weight <= 0.30:
            raise ValueError("support_resistance_weight must be in [0.0, 0.30]")

        if self.subscribe_market_structure and not self.market_structure_updated_topic:
            raise ValueError("market_structure_updated_topic must not be empty")

        if self.subscribe_support_resistance and not self.support_resistance_updated_topic:
            raise ValueError("support_resistance_updated_topic must not be empty")


class TrendAnalyzer(BasePriceActionModule[TrendState]):
    """
    Event-driven trend analyzer.

    Responsibilities:
    - listen to market.candle / market.candles;
    - optionally consume analytics.price_action.market_structure.updated;
    - optionally consume analytics.price_action.support_resistance.updated;
    - calculate internal/external trend state;
    - emit analytics.price_action.trend.* events;
    - expose snapshots for strategy/dashboard/storage layers.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        config: TrendConfig | None = None,
    ) -> None:
        resolved_config = config or TrendConfig()

        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            event_bus=event_bus,
            scheduler=scheduler,
            config=resolved_config,
            service_name="analytics.price_action.trend",
        )

        self.config: TrendConfig = resolved_config

        self._candles: Deque[Candle] = deque(maxlen=self.config.max_candles)
        self._signals: Deque[TrendSignal] = deque(maxlen=self.config.max_signals)

        self._global_candle_index = 0
        self._state_version = 0

        self._latest_market_structure: dict[str, Any] = {}
        self._latest_support_resistance: dict[str, Any] = {}

        self._state = TrendState(
            symbol=self.symbol,
            timeframe=self.timeframe,
        )
        self._missing_mtf_logged = False

        self.logger.info(
            "Initialized TrendAnalyzer",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "config": asdict(self.config),
            },
        )

    # ------------------------------------------------------------------
    # Registration / EventBus handlers
    # ------------------------------------------------------------------

    def register(self) -> None:
        super().register()

        if self.config.subscribe_market_structure:
            self._subscribe(
                self.config.market_structure_updated_topic,
                self.on_market_structure_event,
                name=f"{self.module_name}.on_market_structure_event",
            )

        if self.config.subscribe_support_resistance:
            self._subscribe(
                self.config.support_resistance_updated_topic,
                self.on_support_resistance_event,
                name=f"{self.module_name}.on_support_resistance_event",
            )

    async def on_candle_event(self, event: Event) -> None:
        candles = self._extract_candles_payload(event)
        if not candles:
            self.logger.warning(
                "TrendAnalyzer received empty candle payload",
                extra={"topic": event.topic, "event_id": event.event_id},
            )
            return

        result = self.add_data(candles=candles)
        await self._publish_update_result(result, correlation_id=event.correlation_id)

    async def on_candles_event(self, event: Event) -> None:
        candles = self._extract_candles_payload(event)
        if not candles:
            self.logger.warning(
                "TrendAnalyzer received empty candles payload",
                extra={"topic": event.topic, "event_id": event.event_id},
            )
            return

        result = self.add_data(candles=candles)
        await self._publish_update_result(result, correlation_id=event.correlation_id)

    async def on_market_structure_event(self, event: Event) -> None:
        if not isinstance(event.payload, Mapping):
            self.logger.warning(
                "TrendAnalyzer received invalid market structure payload",
                extra={"topic": event.topic, "event_id": event.event_id},
            )
            return

        result = self.add_data(market_structure=event.payload)
        await self._publish_update_result(result, correlation_id=event.correlation_id)

    async def on_support_resistance_event(self, event: Event) -> None:
        if not isinstance(event.payload, Mapping):
            self.logger.warning(
                "TrendAnalyzer received invalid support/resistance payload",
                extra={"topic": event.topic, "event_id": event.event_id},
            )
            return

        result = self.add_data(support_resistance=event.payload)
        await self._publish_update_result(result, correlation_id=event.correlation_id)

    async def _publish_update_result(
        self,
        result: Mapping[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        for signal_payload in result.get("new_signals", []):
            if not isinstance(signal_payload, Mapping):
                continue

            event_type = signal_payload.get("event_type")
            if not event_type:
                continue

            await self._emit_event(
                self._build_event_name(str(event_type)),
                signal_payload,
                source=self.module_name,
                correlation_id=correlation_id,
            )

        await self._emit_event(
            self._build_event_name("updated"),
            {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "state": result.get("state"),
                "new_signals_count": len(result.get("new_signals", [])),
            },
            source=self.module_name,
            correlation_id=correlation_id,
        )

        if self.config.publish_snapshots:
            await self.publish_snapshot(correlation_id=correlation_id)

    # ------------------------------------------------------------------
    # Public sync domain API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._candles.clear()
        self._signals.clear()
        self._latest_market_structure.clear()
        self._latest_support_resistance.clear()

        self._global_candle_index = 0
        self._state_version += 1
        self._missing_mtf_logged = False

        self._state = TrendState(
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        self.logger.info(
            "TrendAnalyzer reset",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "state_version": self._state_version,
            },
        )

    def get_state(self) -> TrendState:
        return self._state

    def get_signals(self) -> list[TrendSignal]:
        return list(self._signals)

    def update(
        self,
        *,
        candles: Sequence[Mapping[str, Any]] | None = None,
        market_structure: Mapping[str, Any] | None = None,
        support_resistance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.add_data(
            candles=candles,
            market_structure=market_structure,
            support_resistance=support_resistance,
        )

    def add_candle(self, candle: Mapping[str, Any]) -> dict[str, Any]:
        return self.add_data(candles=[candle])

    def add_candles(self, candles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return self.add_data(candles=candles)

    def add_data(
        self,
        *,
        candles: Sequence[Mapping[str, Any]] | None = None,
        market_structure: Mapping[str, Any] | None = None,
        support_resistance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if candles:
            for raw in candles:
                candle = self._parse_candle(raw, index=self._global_candle_index)
                self._global_candle_index += 1

                self._candles.append(candle)
                self._state.last_price = candle.close
                self._state.last_update = candle.timestamp

        if market_structure is not None:
            self._latest_market_structure = dict(market_structure)

        if support_resistance is not None:
            self._latest_support_resistance = dict(support_resistance)

        self._state_version += 1

        new_signals: list[TrendSignal] = []

        new_signals.extend(self._refresh_layer(StructureLayer.INTERNAL))
        new_signals.extend(self._refresh_layer(StructureLayer.EXTERNAL))

        self._refresh_global_state()

        new_signals.extend(self._detect_cross_layer_events())

        self.logger.debug(
            "TrendAnalyzer updated",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "new_signals": len(new_signals),
                "last_price": self._state.last_price,
                "state_version": self._state_version,
            },
        )

        return {
            "state": self.snapshot(),
            "new_signals": [self._signal_to_dict(signal) for signal in new_signals],
        }

    def snapshot(self) -> dict[str, Any]:
        return self._snapshot_envelope(
            state=self._state,
            metadata={
                "total_candles": len(self._candles),
                "signals": len(self._signals),
                "state_version": self._state_version,
                "global_candle_index": self._global_candle_index,
                "has_market_structure": bool(self._latest_market_structure),
                "has_support_resistance": bool(self._latest_support_resistance),
                "config": self._serialize_config(),
            },
        )

    # ------------------------------------------------------------------
    # Layer refresh
    # ------------------------------------------------------------------

    def _refresh_layer(self, layer: StructureLayer) -> list[TrendSignal]:
        layer_state = self._layer_state(layer)

        previous_direction = layer_state.direction
        previous_regime = layer_state.regime
        previous_strength = float(layer_state.strength)
        previous_acceleration = layer_state.is_accelerating
        previous_pullback = layer_state.in_pullback
        previous_exhausted = layer_state.is_exhausted

        closes = [candle.close for candle in self._candles]
        if len(closes) < self.config.long_window:
            self._reset_layer_state(layer_state)
            return []

        short_ma = self._sma(closes, self.config.short_window)
        medium_ma = self._sma(closes, self.config.medium_window)
        long_ma = self._sma(closes, self.config.long_window)

        momentum_score = self._momentum_direction_score(closes)
        slope_score = self._slope_direction_score(closes, short_ma, medium_ma, long_ma)
        structure_score = self._structure_score(layer)
        sr_score = self._support_resistance_score(layer)

        combined_direction_score = (
            momentum_score * 0.35
            + slope_score * 0.40
            + structure_score * self.config.structure_bias_weight
            + sr_score * self.config.support_resistance_weight
        )

        direction = self._resolve_direction(combined_direction_score)
        strength = self._resolve_strength(
            momentum_score=momentum_score,
            slope_score=slope_score,
            structure_score=structure_score,
            sr_score=sr_score,
            direction=direction,
        )
        reversal_risk = self._reversal_risk(closes, direction, slope_score)
        exhaustion_score = self._exhaustion_score(closes, direction)
        consolidation_score = self._consolidation_score(closes)
        pullback_depth = self._pullback_depth(closes, direction)
        continuation_probability = self._continuation_probability(
            strength=strength,
            reversal_risk=reversal_risk,
            exhaustion_score=exhaustion_score,
            consolidation_score=consolidation_score,
        )

        regime = self._resolve_regime(
            direction=direction,
            strength=strength,
            reversal_risk=reversal_risk,
            exhaustion_score=exhaustion_score,
            pullback_depth=pullback_depth,
            consolidation_score=consolidation_score,
        )

        confidence = self._resolve_confidence(
            direction=direction,
            strength=strength,
            structure_score=structure_score,
            sr_score=sr_score,
            reversal_risk=reversal_risk,
        )

        layer_state.direction = direction
        layer_state.regime = regime
        layer_state.strength = self._unit_score(strength)
        layer_state.confidence = self._unit_score(confidence)

        layer_state.momentum_direction_score = self._signed_score(momentum_score)
        layer_state.slope_direction_score = self._signed_score(slope_score)

        layer_state.structure_score = self._unit_score(abs(structure_score))
        layer_state.continuation_probability = self._unit_score(continuation_probability)
        layer_state.reversal_risk = self._unit_score(reversal_risk)
        layer_state.exhaustion_score = self._unit_score(exhaustion_score)
        layer_state.pullback_depth = self._unit_score(pullback_depth)
        layer_state.consolidation_score = self._unit_score(consolidation_score)

        layer_state.is_accelerating = (
            strength >= self.config.acceleration_threshold
            and abs(slope_score) > 0.45
        )
        layer_state.is_exhausted = exhaustion_score >= self.config.exhaustion_threshold
        layer_state.in_pullback = pullback_depth >= self.config.pullback_depth_threshold
        layer_state.is_aligned_with_structure = self._is_direction_aligned_with_structure(
            direction,
            layer,
        )

        layer_state.metadata = {
            "short_ma": short_ma,
            "medium_ma": medium_ma,
            "long_ma": long_ma,
            "combined_direction_score": combined_direction_score,
            "support_resistance_score": sr_score,
            "previous_regime": previous_regime.value,
        }

        signals: list[TrendSignal] = []

        if previous_direction != direction and direction in {
            TrendDirection.BULLISH,
            TrendDirection.BEARISH,
        }:
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_STARTED,
                    direction=direction,
                    strength=strength,
                    confidence=confidence,
                    regime=regime,
                    metadata={"previous_direction": previous_direction.value},
                )
            )

        if previous_direction == direction and direction in {
            TrendDirection.BULLISH,
            TrendDirection.BEARISH,
        }:
            if strength >= self.config.trend_strength_threshold:
                signals.append(
                    self._create_signal(
                        layer=layer,
                        event_type=TrendEventType.TREND_CONTINUATION,
                        direction=direction,
                        strength=strength,
                        confidence=confidence,
                        regime=regime,
                        metadata={"previous_strength": previous_strength},
                    )
                )

        if not previous_acceleration and layer_state.is_accelerating:
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_ACCELERATION,
                    direction=direction,
                    strength=strength,
                    confidence=confidence,
                    regime=regime,
                )
            )

        if (
            previous_strength >= self.config.trend_strength_threshold
            and strength < self.config.trend_strength_threshold
        ):
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_WEAKENING,
                    direction=direction,
                    strength=strength,
                    confidence=confidence,
                    regime=regime,
                )
            )

        if not previous_pullback and layer_state.in_pullback:
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.PULLBACK_STARTED,
                    direction=direction,
                    strength=strength,
                    confidence=confidence,
                    regime=regime,
                )
            )

        if previous_pullback and not layer_state.in_pullback:
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.PULLBACK_ENDED,
                    direction=direction,
                    strength=strength,
                    confidence=confidence,
                    regime=regime,
                )
            )

        if (
            previous_direction in {TrendDirection.BULLISH, TrendDirection.BEARISH}
            and direction in {TrendDirection.BULLISH, TrendDirection.BEARISH}
            and previous_direction != direction
        ):
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_REVERSAL,
                    direction=direction,
                    strength=strength,
                    confidence=confidence,
                    regime=regime,
                    metadata={"previous_direction": previous_direction.value},
                )
            )

        if not previous_exhausted and layer_state.is_exhausted:
            signals.append(
                self._create_signal(
                    layer=layer,
                    event_type=TrendEventType.TREND_EXHAUSTION,
                    direction=direction,
                    strength=strength,
                    confidence=confidence,
                    regime=regime,
                )
            )

        return signals

    def _refresh_global_state(self) -> None:
        internal = self._state.internal
        external = self._state.external

        alignment = 0.0
        if internal.direction == external.direction and internal.direction in {
            TrendDirection.BULLISH,
            TrendDirection.BEARISH,
        }:
            alignment = 1.0
        elif internal.direction == TrendDirection.UNKNOWN or external.direction == TrendDirection.UNKNOWN:
            alignment = 0.0
        else:
            alignment = 0.2

        self._state.internal_external_alignment = self._unit_score(alignment)
        self._state.higher_timeframe_alignment = self._unit_score(
            self._higher_timeframe_alignment()
        )
        self._state.overall_trend_score = self._unit_score(
            (
                float(internal.strength)
                + float(external.strength)
                + float(self._state.internal_external_alignment)
                + float(self._state.higher_timeframe_alignment)
            )
            / 4.0
        )

        self._state.metadata = {
            "internal_direction": internal.direction.value,
            "external_direction": external.direction.value,
        }

    def _detect_cross_layer_events(self) -> list[TrendSignal]:
        internal = self._state.internal
        external = self._state.external

        if internal.direction == TrendDirection.UNKNOWN or external.direction == TrendDirection.UNKNOWN:
            return []

        if internal.direction == TrendDirection.NEUTRAL and external.direction == TrendDirection.NEUTRAL:
            return []

        if internal.direction == external.direction:
            return [
                self._create_signal(
                    layer=StructureLayer.EXTERNAL,
                    event_type=TrendEventType.TREND_ALIGNMENT,
                    direction=external.direction,
                    strength=float(self._state.overall_trend_score),
                    confidence=(float(internal.confidence) + float(external.confidence)) / 2.0,
                    regime=external.regime,
                    metadata={
                        "internal_direction": internal.direction.value,
                        "external_direction": external.direction.value,
                    },
                )
            ]

        return [
            self._create_signal(
                layer=StructureLayer.EXTERNAL,
                event_type=TrendEventType.TREND_DISAGREEMENT,
                direction=external.direction,
                strength=float(self._state.overall_trend_score),
                confidence=(float(internal.confidence) + float(external.confidence)) / 2.0,
                regime=external.regime,
                metadata={
                    "internal_direction": internal.direction.value,
                    "external_direction": external.direction.value,
                },
            )
        ]

    # ------------------------------------------------------------------
    # Core scoring logic
    # ------------------------------------------------------------------

    def _momentum_direction_score(self, closes: Sequence[float]) -> float:
        if len(closes) < self.config.short_window + 1:
            return 0.0

        lookback = closes[-self.config.short_window :]
        start = lookback[0]
        end = lookback[-1]

        if start <= 0:
            return 0.0

        pct = (end - start) / start
        scaled = pct / max(self.config.momentum_slope_threshold, 1e-9)
        return self._clamp_signed(scaled)

    def _slope_direction_score(
        self,
        closes: Sequence[float],
        short_ma: float,
        medium_ma: float,
        long_ma: float,
    ) -> float:
        current_price = closes[-1]
        if current_price <= 0:
            return 0.0

        ma_stack = 0.0
        if short_ma > medium_ma > long_ma:
            ma_stack = 1.0
        elif short_ma < medium_ma < long_ma:
            ma_stack = -1.0

        short_dist = (current_price - short_ma) / current_price
        medium_dist = (short_ma - medium_ma) / current_price
        long_dist = (medium_ma - long_ma) / current_price

        composite = ma_stack * 0.5 + (short_dist + medium_dist + long_dist) * 40.0 * 0.5
        return self._clamp_signed(composite)

    def _structure_score(self, layer: StructureLayer) -> float:
        if not self._latest_market_structure:
            return 0.0

        layer_key = "internal" if layer == StructureLayer.INTERNAL else "external"

        layer_ctx = self._latest_market_structure.get("state", {}).get(layer_key)
        if layer_ctx is None:
            layer_ctx = self._latest_market_structure.get(layer_key, {})

        if not isinstance(layer_ctx, Mapping):
            return 0.0

        raw_bias = layer_ctx.get("bias", MarketBias.UNKNOWN)
        bias = self._coerce_market_bias(raw_bias)

        confidence = self._coerce_float(layer_ctx.get("confidence", 0.0))
        trend_strength = self._coerce_float(layer_ctx.get("trend_strength", 0.0))

        signed = 0.0
        if bias == MarketBias.BULLISH:
            signed = 1.0
        elif bias == MarketBias.BEARISH:
            signed = -1.0

        return self._clamp_signed(signed * ((confidence + trend_strength) / 2.0))

    def _support_resistance_score(self, layer: StructureLayer) -> float:
        if not self._latest_support_resistance:
            return 0.0

        current_price = self._state.last_price
        if current_price is None or current_price <= 0:
            return 0.0

        layer_key = "internal" if layer == StructureLayer.INTERNAL else "external"

        layer_ctx = self._latest_support_resistance.get("state", {}).get(layer_key)
        if layer_ctx is None:
            layer_ctx = self._latest_support_resistance.get(layer_key, {})

        if not isinstance(layer_ctx, Mapping):
            return 0.0

        nearest_support = layer_ctx.get("nearest_support")
        nearest_resistance = layer_ctx.get("nearest_resistance")

        support_distance = self._extract_distance_pct(nearest_support, current_price)
        resistance_distance = self._extract_distance_pct(nearest_resistance, current_price)

        if support_distance is None and resistance_distance is None:
            return 0.0
        if support_distance is None:
            return -0.35
        if resistance_distance is None:
            return 0.35

        if support_distance < resistance_distance:
            return self._clamp_signed((resistance_distance - support_distance) * 40.0)

        return self._clamp_signed(-((support_distance - resistance_distance) * 40.0))

    def _resolve_direction(self, score: float) -> TrendDirection:
        if score >= self.config.direction_positive_threshold:
            return TrendDirection.BULLISH
        if score <= self.config.direction_negative_threshold:
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL

    def _resolve_strength(
        self,
        *,
        momentum_score: float,
        slope_score: float,
        structure_score: float,
        sr_score: float,
        direction: TrendDirection,
    ) -> float:
        if direction == TrendDirection.NEUTRAL:
            return 0.0

        strength = (
            abs(momentum_score) * 0.30
            + abs(slope_score) * 0.35
            + abs(structure_score) * 0.25
            + abs(sr_score) * 0.10
        )
        return self._clamp_unit(strength)

    def _reversal_risk(
        self,
        closes: Sequence[float],
        direction: TrendDirection,
        slope_score: float,
    ) -> float:
        if len(closes) < self.config.medium_window:
            return 0.0
        if direction == TrendDirection.NEUTRAL:
            return 0.0

        short_base = max(closes[-self.config.short_window], 1e-9)
        medium_base = max(closes[-self.config.medium_window], 1e-9)

        short_return = abs((closes[-1] - closes[-self.config.short_window]) / short_base)
        medium_return = abs((closes[-1] - closes[-self.config.medium_window]) / medium_base)

        overstretch = min(1.0, (short_return + medium_return) * 25.0)
        slope_fade = 1.0 - min(1.0, abs(slope_score))
        raw = overstretch * 0.7 + slope_fade * 0.3
        return self._clamp_unit(raw)

    def _exhaustion_score(
        self,
        closes: Sequence[float],
        direction: TrendDirection,
    ) -> float:
        if len(closes) < self.config.short_window:
            return 0.0
        if direction == TrendDirection.NEUTRAL:
            return 0.0

        candles = list(self._candles)[-self.config.short_window :]
        if not candles:
            return 0.0

        wick_ratios: list[float] = []
        for candle in candles:
            if direction == TrendDirection.BULLISH:
                wick_ratios.append(candle.upper_wick_ratio)
            else:
                wick_ratios.append(candle.lower_wick_ratio)

        avg_wick = sum(wick_ratios) / len(wick_ratios)
        return self._clamp_unit(avg_wick)

    def _consolidation_score(self, closes: Sequence[float]) -> float:
        if len(closes) < self.config.medium_window:
            return 0.0

        recent = closes[-self.config.medium_window :]
        high_ = max(recent)
        low_ = min(recent)
        mid = sum(recent) / len(recent)

        if mid <= 0:
            return 0.0

        range_pct = (high_ - low_) / mid
        score = 1.0 - min(1.0, range_pct / self.config.consolidation_range_threshold)
        return self._clamp_unit(score)

    def _pullback_depth(
        self,
        closes: Sequence[float],
        direction: TrendDirection,
    ) -> float:
        if len(closes) < self.config.short_window:
            return 0.0
        if direction == TrendDirection.NEUTRAL:
            return 0.0

        recent = closes[-self.config.short_window :]
        current = recent[-1]

        if direction == TrendDirection.BULLISH:
            peak = max(recent)
            if peak <= 0:
                return 0.0
            return self._clamp_unit((peak - current) / peak)

        trough = min(recent)
        if current <= 0:
            return 0.0

        return self._clamp_unit((current - trough) / current)

    def _continuation_probability(
        self,
        *,
        strength: float,
        reversal_risk: float,
        exhaustion_score: float,
        consolidation_score: float,
    ) -> float:
        raw = (
            strength * 0.5
            + (1.0 - reversal_risk) * 0.25
            + (1.0 - exhaustion_score) * 0.15
            + (1.0 - consolidation_score) * 0.10
        )
        return self._clamp_unit(raw)

    def _resolve_regime(
        self,
        *,
        direction: TrendDirection,
        strength: float,
        reversal_risk: float,
        exhaustion_score: float,
        pullback_depth: float,
        consolidation_score: float,
    ) -> TrendRegime:
        if direction == TrendDirection.NEUTRAL:
            return (
                TrendRegime.CONSOLIDATING
                if consolidation_score >= 0.55
                else TrendRegime.UNKNOWN
            )

        if exhaustion_score >= self.config.exhaustion_threshold:
            return TrendRegime.EXHAUSTED

        if reversal_risk >= self.config.reversal_risk_threshold:
            return TrendRegime.REVERSING

        if pullback_depth >= self.config.pullback_depth_threshold:
            return TrendRegime.PULLBACK

        if consolidation_score >= 0.60:
            return TrendRegime.CONSOLIDATING

        if strength >= self.config.trend_strength_threshold:
            return TrendRegime.TRENDING

        return TrendRegime.UNKNOWN

    def _resolve_confidence(
        self,
        *,
        direction: TrendDirection,
        strength: float,
        structure_score: float,
        sr_score: float,
        reversal_risk: float,
    ) -> float:
        if direction == TrendDirection.UNKNOWN:
            return 0.0

        base = strength * 0.6 + abs(structure_score) * 0.25 + abs(sr_score) * 0.15
        confidence = base * (1.0 - reversal_risk * 0.5)
        return self._clamp_unit(confidence)

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _higher_timeframe_alignment(self) -> float:
        if not self._latest_market_structure:
            if self.config.log_missing_mtf_once and not self._missing_mtf_logged:
                self._missing_mtf_logged = True
                self.logger.debug(
                    "TrendAnalyzer has no market structure context yet",
                    extra={"symbol": self.symbol, "timeframe": self.timeframe},
                )
            return 0.0

        mtf = self._latest_market_structure.get("state", {}).get("mtf_alignment")
        if mtf is None:
            mtf = self._latest_market_structure.get("mtf_alignment", {})

        if not isinstance(mtf, Mapping):
            return 0.0

        return self._clamp_unit(self._coerce_float(mtf.get("alignment_score", 0.0)))

    def _is_direction_aligned_with_structure(
        self,
        direction: TrendDirection,
        layer: StructureLayer,
    ) -> bool:
        structure_score = self._structure_score(layer)

        if direction == TrendDirection.BULLISH:
            return structure_score > 0
        if direction == TrendDirection.BEARISH:
            return structure_score < 0

        return False

    def _extract_distance_pct(
        self,
        level_ctx: Any,
        current_price: float,
    ) -> float | None:
        if not isinstance(level_ctx, Mapping):
            return None

        price = level_ctx.get("price")
        if price is None:
            return None

        price_float = self._coerce_float(price)
        if price_float <= 0:
            return None

        return abs(price_float - current_price) / current_price

    def _coerce_market_bias(self, value: Any) -> MarketBias:
        if isinstance(value, MarketBias):
            return value

        try:
            return MarketBias(str(value))
        except Exception:
            return MarketBias.UNKNOWN

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    def _create_signal(
        self,
        *,
        layer: StructureLayer,
        event_type: TrendEventType,
        direction: TrendDirection,
        strength: float,
        confidence: float,
        regime: TrendRegime,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrendSignal:
        signal = TrendSignal(
            signal_id=uuid4().hex,
            timestamp=self._state.last_update or self._now_utc(),
            symbol=self.symbol,
            timeframe=self.timeframe,
            layer=layer,
            event_type=event_type,
            direction=direction,
            strength=self._unit_score(strength),
            confidence=self._unit_score(confidence),
            regime=regime,
            price=self._state.last_price,
            metadata=dict(metadata or {}),
        )

        self._signals.append(signal)
        self._layer_state(layer).last_signal = signal

        return signal

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _layer_state(self, layer: StructureLayer) -> TrendLayerState:
        return self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

    def _reset_layer_state(self, layer_state: TrendLayerState) -> None:
        layer = layer_state.layer

        layer_state.direction = TrendDirection.UNKNOWN
        layer_state.regime = TrendRegime.UNKNOWN
        layer_state.strength = self._unit_score(0.0)
        layer_state.confidence = self._unit_score(0.0)
        layer_state.momentum_direction_score = self._signed_score(0.0)
        layer_state.slope_direction_score = self._signed_score(0.0)
        layer_state.structure_score = self._unit_score(0.0)
        layer_state.continuation_probability = self._unit_score(0.0)
        layer_state.reversal_risk = self._unit_score(0.0)
        layer_state.exhaustion_score = self._unit_score(0.0)
        layer_state.pullback_depth = self._unit_score(0.0)
        layer_state.consolidation_score = self._unit_score(0.0)
        layer_state.is_accelerating = False
        layer_state.is_exhausted = False
        layer_state.in_pullback = False
        layer_state.is_aligned_with_structure = False
        layer_state.metadata = {"layer": layer.value}

    def _sma(self, values: Sequence[float], window: int) -> float:
        if len(values) < window:
            return sum(values) / len(values) if values else 0.0

        slice_ = values[-window:]
        return sum(slice_) / len(slice_)

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _clamp_unit(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _clamp_signed(self, value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    def _unit_score(self, value: float) -> UnitScore:
        return UnitScore(self._clamp_unit(value))

    def _signed_score(self, value: float) -> SignedScore:
        return SignedScore(self._clamp_signed(value))

    def _signal_to_dict(self, signal: TrendSignal) -> dict[str, Any]:
        serialized = self._safe_serialize(signal)
        return serialized if isinstance(serialized, dict) else {"value": serialized}