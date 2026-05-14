from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Mapping, Sequence
from uuid import uuid4
from copy import deepcopy
from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from analytics.price_action.base import BasePriceActionConfig, BasePriceActionModule
from analytics.price_action.enums import (
    LevelStatus,
    LevelType,
    SREventType,
    StructureLayer,
    SwingType,
)
from analytics.price_action.models import (
    Candle,
    LayerSRState,
    SupportResistanceEvent,
    SupportResistanceLevel,
    SupportResistanceState,
    SwingPoint,
    clamp_unit,
)


@dataclass(slots=True)
class SupportResistanceConfig(BasePriceActionConfig):
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

    retest_window_bars: int = 12
    allow_flip_on_break: bool = True
    decay_broken_levels: bool = False

    emit_events: bool = True
    event_namespace: str = "analytics.price_action.support_resistance"
    publish_snapshots: bool = False

    subscribe_market_structure_swings: bool = True
    swing_high_topic: str = "analytics.price_action.market_structure.swing_high"
    swing_low_topic: str = "analytics.price_action.market_structure.swing_low"

    def validate(self) -> None:
        BasePriceActionConfig.validate(self)

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
        if self.rejection_wick_ratio_threshold < 0:
            raise ValueError("rejection_wick_ratio_threshold must be >= 0")

        if self.subscribe_market_structure_swings:
            if not self.swing_high_topic:
                raise ValueError("swing_high_topic must not be empty")
            if not self.swing_low_topic:
                raise ValueError("swing_low_topic must not be empty")


class SupportResistanceAnalyzer(BasePriceActionModule[SupportResistanceState]):
    """
    Event-driven support / resistance analyzer.

    Responsibilities:
    - listen to market.candle / market.candles
    - listen to analytics.price_action.market_structure.swing_high
    - listen to analytics.price_action.market_structure.swing_low
    - build support/resistance zones from swing points
    - track touches, rejections, breakouts, retests and flips
    - publish analytics.price_action.support_resistance.* events
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        config: SupportResistanceConfig | None = None,
    ) -> None:
        resolved_config = config or SupportResistanceConfig()

        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            event_bus=event_bus,
            scheduler=scheduler,
            config=resolved_config,
            service_name="analytics.price_action.support_resistance",
        )

        self.config: SupportResistanceConfig = resolved_config

        self._candles: Deque[Candle] = deque(maxlen=self.config.max_candles)
        self._internal_levels: Deque[SupportResistanceLevel] = deque(maxlen=self.config.max_levels_per_layer)
        self._external_levels: Deque[SupportResistanceLevel] = deque(maxlen=self.config.max_levels_per_layer)
        self._events: Deque[SupportResistanceEvent] = deque(maxlen=self.config.max_events)

        self._processed_swings: set[str] = set()
        self._processed_touch_keys: set[tuple[str, int]] = set()
        self._processed_break_keys: set[tuple[str, int]] = set()
        self._processed_retest_keys: set[tuple[str, int]] = set()
        self._processed_rejection_keys: set[tuple[str, int]] = set()
        self._processed_flip_keys: set[tuple[str, int]] = set()

        self._global_candle_index = 0

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
    # Registration / EventBus handlers
    # -------------------------------------------------------------------------

    def register(self) -> None:
        super().register()

        if self.config.subscribe_market_structure_swings:
            self._subscribe(
                self.config.swing_high_topic,
                self.on_swing_event,
                name=f"{self.module_name}.on_swing_high_event",
            )
            self._subscribe(
                self.config.swing_low_topic,
                self.on_swing_event,
                name=f"{self.module_name}.on_swing_low_event",
            )

    async def on_candle_event(self, event: Event) -> None:
        candles = self._extract_candles_payload(event)
        if not candles:
            self.logger.warning(
                "SupportResistanceAnalyzer received empty candle payload",
                extra={"topic": event.topic, "event_id": event.event_id},
            )
            return

        result = self.add_candles(candles)
        await self._publish_update_result(result, correlation_id=event.correlation_id)

    async def on_candles_event(self, event: Event) -> None:
        candles = self._extract_candles_payload(event)
        if not candles:
            self.logger.warning(
                "SupportResistanceAnalyzer received empty candles payload",
                extra={"topic": event.topic, "event_id": event.event_id},
            )
            return

        result = self.add_candles(candles)
        await self._publish_update_result(result, correlation_id=event.correlation_id)

    async def on_swing_event(self, event: Event) -> None:
        if not isinstance(event.payload, Mapping):
            self.logger.warning(
                "SupportResistanceAnalyzer received invalid swing payload",
                extra={"topic": event.topic, "event_id": event.event_id},
            )
            return

        result = self.add_swings([event.payload])
        await self._publish_update_result(result, correlation_id=event.correlation_id)

    async def _publish_update_result(
        self,
        result: Mapping[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        for event_payload in result.get("new_events", []):
            if not isinstance(event_payload, Mapping):
                continue

            event_type = event_payload.get("event_type")
            if not event_type:
                continue

            await self._emit_event(
                self._build_event_name(str(event_type)),
                event_payload,
                source=self.module_name,
                correlation_id=correlation_id,
            )

        await self._emit_event(
            self._build_event_name("updated"),
            {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "state": result.get("state"),
                "updated_levels_count": len(result.get("updated_levels", [])),
                "new_events_count": len(result.get("new_events", [])),
            },
            source=self.module_name,
            correlation_id=correlation_id,
        )

        if self.config.publish_snapshots:
            await self.publish_snapshot(correlation_id=correlation_id)

    # -------------------------------------------------------------------------
    # Public sync domain API
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
        self._processed_rejection_keys.clear()
        self._processed_flip_keys.clear()

        self._global_candle_index = 0
        self._state = SupportResistanceState(
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        self.logger.info(
            "SupportResistanceAnalyzer reset",
            extra={"symbol": self.symbol, "timeframe": self.timeframe},
        )

    def get_state(self) -> SupportResistanceState:
        return self._state

    def get_levels(self, layer: StructureLayer | None = None) -> list[SupportResistanceLevel]:
        if layer == StructureLayer.INTERNAL:
            return list(self._internal_levels)
        if layer == StructureLayer.EXTERNAL:
            return list(self._external_levels)
        return [*self._internal_levels, *self._external_levels]

    def get_events(self) -> list[SupportResistanceEvent]:
        return list(self._events)

    def update(
        self,
        *,
        candles: Sequence[Mapping[str, Any]] | None = None,
        swings: Sequence[SwingPoint | Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self.add_data(candles=candles, swings=swings)

    def add_data(
            self,
            *,
            candles: Sequence[Mapping[str, Any]] | None = None,
            swings: Sequence[SwingPoint | Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Atomically ingest support/resistance inputs.

        Important:
        - If any candle or swing in the provided batch is invalid, no partial state
          mutation is kept.
        - This protects rolling levels/events/processed keys/global index from
          half-committed updates.
        - On success, behavior remains the same as before.
        """
        candles_batch = list(candles or [])
        swings_batch = list(swings or [])

        # Fast no-op path.
        if not candles_batch and not swings_batch:
            self._refresh_state()

            self.logger.debug(
                "Support/resistance updated",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "updated_levels": 0,
                    "new_events": 0,
                    "last_price": self._state.last_price,
                },
            )

            return {
                "state": self.snapshot(),
                "updated_levels": [],
                "new_events": [],
            }

        # Transaction snapshot. Deep copy is intentional because candle processing
        # mutates existing level objects in-place: touch_count, status, flipped_at,
        # metadata, etc.
        rollback_internal_levels = deepcopy(self._internal_levels)
        rollback_external_levels = deepcopy(self._external_levels)
        rollback_events = deepcopy(self._events)
        rollback_candles = deepcopy(self._candles)

        rollback_processed_swings = set(self._processed_swings)
        rollback_processed_touch_keys = set(self._processed_touch_keys)
        rollback_processed_break_keys = set(self._processed_break_keys)
        rollback_processed_retest_keys = set(self._processed_retest_keys)
        rollback_processed_rejection_keys = set(self._processed_rejection_keys)
        rollback_processed_flip_keys = set(self._processed_flip_keys)

        rollback_global_candle_index = self._global_candle_index
        rollback_state = deepcopy(self._state)

        new_events: list[SupportResistanceEvent] = []
        updated_levels: list[SupportResistanceLevel] = []

        try:
            if swings_batch:
                levels_from_swings, events_from_swings = self._ingest_swings(swings_batch)
                updated_levels.extend(levels_from_swings)
                new_events.extend(events_from_swings)

            if candles_batch:
                events_from_candles = self._ingest_candles(candles_batch)
                new_events.extend(events_from_candles)

            self._refresh_state()

        except Exception:
            self._internal_levels = rollback_internal_levels
            self._external_levels = rollback_external_levels
            self._events = rollback_events
            self._candles = rollback_candles

            self._processed_swings = rollback_processed_swings
            self._processed_touch_keys = rollback_processed_touch_keys
            self._processed_break_keys = rollback_processed_break_keys
            self._processed_retest_keys = rollback_processed_retest_keys
            self._processed_rejection_keys = rollback_processed_rejection_keys
            self._processed_flip_keys = rollback_processed_flip_keys

            self._global_candle_index = rollback_global_candle_index
            self._state = rollback_state

            self.logger.exception(
                "Support/resistance batch ingestion failed and was rolled back",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "candles_count": len(candles_batch),
                    "swings_count": len(swings_batch),
                    "rollback_global_candle_index": rollback_global_candle_index,
                },
            )
            raise

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

        return {
            "state": self.snapshot(),
            "updated_levels": [self._level_to_dict(level) for level in updated_levels],
            "new_events": [self._event_to_dict(event) for event in new_events],
        }

    def add_candle(self, candle: Mapping[str, Any]) -> dict[str, Any]:
        return self.add_data(candles=[candle])

    def add_candles(self, candles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return self.add_data(candles=candles)

    def add_swings(self, swings: Sequence[SwingPoint | Mapping[str, Any]]) -> dict[str, Any]:
        return self.add_data(swings=swings)

    def snapshot(self) -> dict[str, Any]:
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
    ) -> tuple[list[SupportResistanceLevel], list[SupportResistanceEvent]]:
        updated_levels: list[SupportResistanceLevel] = []
        new_events: list[SupportResistanceEvent] = []

        for raw in swings:
            swing = self._parse_swing(raw)

            if swing.swing_id in self._processed_swings:
                continue

            self._processed_swings.add(swing.swing_id)

            level_type = self._level_type_from_swing(swing)
            level_price = swing.price
            upper_bound, lower_bound = self._build_zone_bounds(
                price=level_price,
                layer=swing.layer,
            )

            existing = self._find_merge_candidate(
                layer=swing.layer,
                level_type=level_type,
                price=level_price,
            )

            if existing is not None:
                self._merge_swing_into_level(
                    existing,
                    swing,
                    upper_bound=upper_bound,
                    lower_bound=lower_bound,
                )
                updated_levels.append(existing)

                event = self._create_event(
                    event_type=SREventType.LEVEL_MERGED,
                    level=existing,
                    timestamp=swing.timestamp,
                    reference_price=swing.price,
                    confidence=existing.strength,
                    metadata={
                        "merged_swing_id": swing.swing_id,
                        "source_count": existing.source_count,
                    },
                )
                new_events.append(event)
                continue

            level = SupportResistanceLevel(
                level_id=uuid4().hex,
                layer=swing.layer,
                level_type=level_type,
                price=level_price,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
                strength=clamp_unit(swing.strength),
                status=LevelStatus.ACTIVE,
                created_at=swing.timestamp,
                updated_at=swing.timestamp,
                touch_count=1,
                source_count=1,
                source_swing_ids=[swing.swing_id],
                source_prices=[swing.price],
                metadata={
                    "origin_swing_type": swing.swing_type.value,
                    "origin_swing_index": swing.index,
                    "validated": 1 >= self.config.min_touches_for_validation,
                },
            )

            self._levels_for_layer(swing.layer).append(level)
            updated_levels.append(level)

            event = self._create_event(
                event_type=SREventType.LEVEL_CREATED,
                level=level,
                timestamp=swing.timestamp,
                reference_price=swing.price,
                confidence=level.strength,
                metadata={
                    "source_swing_id": swing.swing_id,
                    "source_swing_type": swing.swing_type.value,
                },
            )
            new_events.append(event)

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
            strength=clamp_unit(float(raw.get("strength", 0.0))),
            is_confirmed=bool(raw.get("is_confirmed", True)),
            metadata=dict(raw.get("metadata", {})),
        )

    def _level_type_from_swing(self, swing: SwingPoint) -> LevelType:
        return LevelType.RESISTANCE if swing.swing_type == SwingType.HIGH else LevelType.SUPPORT

    def _merge_swing_into_level(
        self,
        level: SupportResistanceLevel,
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

        avg_strength = (level.strength + clamp_unit(swing.strength)) / 2.0
        source_bonus = min(0.25, 0.03 * max(0, level.source_count - 1))
        level.strength = clamp_unit(avg_strength + source_bonus)
        level.metadata["validated"] = level.touch_count >= self.config.min_touches_for_validation

    # -------------------------------------------------------------------------
    # Candles ingestion
    # -------------------------------------------------------------------------

    def _ingest_candles(self, candles: Sequence[Mapping[str, Any]]) -> list[SupportResistanceEvent]:
        new_events: list[SupportResistanceEvent] = []

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
        level: SupportResistanceLevel,
        candle: Candle,
    ) -> list[SupportResistanceEvent]:
        events: list[SupportResistanceEvent] = []

        if level.status == LevelStatus.INACTIVE:
            return events

        if self._is_level_touched(level, candle):
            touch_key = (level.level_id, candle.index)
            if touch_key not in self._processed_touch_keys:
                self._processed_touch_keys.add(touch_key)

                level.touch_count += 1
                level.last_tested_at = candle.timestamp
                level.updated_at = candle.timestamp
                level.metadata["validated"] = level.touch_count >= self.config.min_touches_for_validation

                events.append(
                    self._create_event(
                        event_type=SREventType.LEVEL_TOUCHED,
                        level=level,
                        timestamp=candle.timestamp,
                        reference_price=candle.close,
                        confidence=level.strength,
                        metadata={"candle_index": candle.index},
                    )
                )

        if self._is_rejection(level, candle):
            reject_key = (level.level_id, candle.index)
            if reject_key not in self._processed_rejection_keys:
                self._processed_rejection_keys.add(reject_key)

                level.rejection_count += 1
                level.last_rejected_at = candle.timestamp
                level.updated_at = candle.timestamp

                events.append(
                    self._create_event(
                        event_type=SREventType.LEVEL_REJECTED,
                        level=level,
                        timestamp=candle.timestamp,
                        reference_price=candle.close,
                        confidence=self._rejection_confidence(level, candle),
                        metadata={"candle_index": candle.index},
                    )
                )

        if self._is_broken(level, candle):
            break_key = (level.level_id, candle.index)
            if break_key not in self._processed_break_keys:
                self._processed_break_keys.add(break_key)

                old_type = level.level_type
                level.status = LevelStatus.BROKEN
                level.break_count += 1
                level.broken_at = candle.timestamp
                level.last_broken_at = candle.timestamp
                level.updated_at = candle.timestamp

                events.append(
                    self._create_event(
                        event_type=SREventType.LEVEL_BROKEN,
                        level=level,
                        timestamp=candle.timestamp,
                        reference_price=candle.close,
                        confidence=self._break_confidence(level, candle),
                        metadata={
                            "candle_index": candle.index,
                            "old_level_type": old_type.value,
                        },
                    )
                )

                if self.config.allow_flip_on_break:
                    flipped = self._flip_level(level, timestamp=candle.timestamp)
                    flip_key = (level.level_id, candle.index)

                    if flipped and flip_key not in self._processed_flip_keys:
                        self._processed_flip_keys.add(flip_key)
                        events.append(
                            self._create_event(
                                event_type=SREventType.LEVEL_FLIPPED,
                                level=level,
                                timestamp=candle.timestamp,
                                reference_price=candle.close,
                                confidence=level.strength,
                                metadata={
                                    "candle_index": candle.index,
                                    "new_level_type": level.level_type.value,
                                },
                            )
                        )

        if self._is_retested_after_break(level, candle):
            retest_key = (level.level_id, candle.index)
            if retest_key not in self._processed_retest_keys:
                self._processed_retest_keys.add(retest_key)

                level.retest_count += 1
                level.last_retested_at = candle.timestamp
                level.updated_at = candle.timestamp

                events.append(
                    self._create_event(
                        event_type=SREventType.LEVEL_RETESTED,
                        level=level,
                        timestamp=candle.timestamp,
                        reference_price=candle.close,
                        confidence=level.strength,
                        metadata={"candle_index": candle.index},
                    )
                )

        if self.config.decay_broken_levels:
            self._maybe_decay_broken_level(level, candle)

        return events

    # -------------------------------------------------------------------------
    # Detection rules
    # -------------------------------------------------------------------------

    def _is_level_touched(self, level: SupportResistanceLevel, candle: Candle) -> bool:
        return candle.high >= level.lower_bound and candle.low <= level.upper_bound

    def _is_rejection(self, level: SupportResistanceLevel, candle: Candle) -> bool:
        if not self._is_level_touched(level, candle):
            return False

        if level.level_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}:
            return (
                candle.high >= level.lower_bound
                and candle.close < level.price
                and candle.upper_wick_ratio >= self.config.rejection_wick_ratio_threshold
            )

        return (
            candle.low <= level.upper_bound
            and candle.close > level.price
            and candle.lower_wick_ratio >= self.config.rejection_wick_ratio_threshold
        )

    def _is_broken(self, level: SupportResistanceLevel, candle: Candle) -> bool:
        threshold = self.config.breakout_threshold_pct

        if level.level_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}:
            breakout_price = level.upper_bound * (1.0 + threshold)
            return (
                candle.close > breakout_price
                if self.config.require_close_break
                else candle.high > breakout_price
            )

        breakout_price = level.lower_bound * (1.0 - threshold)
        return (
            candle.close < breakout_price
            if self.config.require_close_break
            else candle.low < breakout_price
        )

    def _is_retested_after_break(self, level: SupportResistanceLevel, candle: Candle) -> bool:
        if level.status != LevelStatus.BROKEN:
            return False
        if level.last_broken_at is None:
            return False

        bars_since_break = self._bars_since_timestamp(level.last_broken_at)
        if bars_since_break is None or bars_since_break > self.config.retest_window_bars:
            return False

        return self._is_level_touched(level, candle)

    def _flip_level(self, level: SupportResistanceLevel, *, timestamp: Any) -> bool:
        if level.level_type == LevelType.SUPPORT:
            level.level_type = LevelType.FLIP_RESISTANCE
        elif level.level_type == LevelType.RESISTANCE:
            level.level_type = LevelType.FLIP_SUPPORT
        elif level.level_type == LevelType.FLIP_SUPPORT:
            level.level_type = LevelType.FLIP_RESISTANCE
        elif level.level_type == LevelType.FLIP_RESISTANCE:
            level.level_type = LevelType.FLIP_SUPPORT
        else:
            return False

        level.flipped_at = timestamp
        level.updated_at = timestamp
        return True

    def _maybe_decay_broken_level(self, level: SupportResistanceLevel, candle: Candle) -> None:
        if level.status != LevelStatus.BROKEN or level.last_broken_at is None:
            return

        bars_since_break = self._bars_since_timestamp(level.last_broken_at)
        if bars_since_break is None:
            return

        if bars_since_break > self.config.retest_window_bars * 2:
            level.status = LevelStatus.INACTIVE
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
        active_levels = [x for x in levels if x.status == LevelStatus.ACTIVE]
        broken_levels = [x for x in levels if x.status == LevelStatus.BROKEN]

        state.total_levels = len(levels)
        state.active_supports = len(
            [x for x in active_levels if x.level_type in {LevelType.SUPPORT, LevelType.FLIP_SUPPORT}]
        )
        state.active_resistances = len(
            [x for x in active_levels if x.level_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}]
        )
        state.active_flip_supports = len([x for x in active_levels if x.level_type == LevelType.FLIP_SUPPORT])
        state.active_flip_resistances = len([x for x in active_levels if x.level_type == LevelType.FLIP_RESISTANCE])

        current_price = self._state.last_price

        state.strongest_support = self._strongest_level(
            levels,
            types={LevelType.SUPPORT, LevelType.FLIP_SUPPORT},
        )
        state.strongest_resistance = self._strongest_level(
            levels,
            types={LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE},
        )
        state.nearest_support = self._nearest_level(
            levels,
            current_price=current_price,
            below_or_equal=True,
            types={LevelType.SUPPORT, LevelType.FLIP_SUPPORT},
        )
        state.nearest_resistance = self._nearest_level(
            levels,
            current_price=current_price,
            below_or_equal=False,
            types={LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE},
        )

        layer_events = [x for x in self._events if x.layer == layer]
        state.last_event = layer_events[-1] if layer_events else None
        state.metadata = {
            "active_levels": len(active_levels),
            "broken_levels": len(broken_levels),
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_zone_bounds(self, *, price: float, layer: StructureLayer) -> tuple[float, float]:
        half_width_pct = (
            self.config.internal_zone_half_width_pct
            if layer == StructureLayer.INTERNAL
            else self.config.external_zone_half_width_pct
        )
        half_width = price * half_width_pct
        return price + half_width, price - half_width

    def _merge_distance_pct(self, layer: StructureLayer) -> float:
        return (
            self.config.internal_merge_distance_pct
            if layer == StructureLayer.INTERNAL
            else self.config.external_merge_distance_pct
        )

    def _levels_for_layer(self, layer: StructureLayer) -> Deque[SupportResistanceLevel]:
        return self._internal_levels if layer == StructureLayer.INTERNAL else self._external_levels

    def _layer_state(self, layer: StructureLayer) -> LayerSRState:
        return self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

    def _find_merge_candidate(
        self,
        *,
        layer: StructureLayer,
        level_type: LevelType,
        price: float,
    ) -> SupportResistanceLevel | None:
        candidates = [
            x for x in self._levels_for_layer(layer)
            if x.level_type == level_type and x.status != LevelStatus.INACTIVE
        ]

        if not candidates:
            return None

        threshold_pct = self._merge_distance_pct(layer)

        best: SupportResistanceLevel | None = None
        best_distance = float("inf")

        for level in candidates:
            if level.price <= 0:
                continue

            distance_pct = abs(price - level.price) / level.price
            if distance_pct <= threshold_pct and distance_pct < best_distance:
                best = level
                best_distance = distance_pct

        return best

    def _bars_since_timestamp(self, timestamp: Any) -> int | None:
        candles = list(self._candles)
        if not candles:
            return None

        last_index = candles[-1].index
        matching = None

        for candle in reversed(candles):
            if candle.timestamp == timestamp:
                matching = candle.index
                break

        if matching is None:
            return None

        return max(0, last_index - matching)

    def _strongest_level(
        self,
        levels: Sequence[SupportResistanceLevel],
        *,
        types: set[LevelType],
    ) -> SupportResistanceLevel | None:
        candidates = [
            x for x in levels
            if x.level_type in types and x.status != LevelStatus.INACTIVE
        ]

        if not candidates:
            return None

        return max(candidates, key=lambda x: (x.strength, x.touch_count, x.source_count))

    def _nearest_level(
        self,
        levels: Sequence[SupportResistanceLevel],
        *,
        current_price: float | None,
        below_or_equal: bool,
        types: set[LevelType],
    ) -> SupportResistanceLevel | None:
        if current_price is None:
            return None

        candidates = [
            x for x in levels
            if x.level_type in types and x.status != LevelStatus.INACTIVE
        ]

        if below_or_equal:
            candidates = [x for x in candidates if x.price <= current_price]
        else:
            candidates = [x for x in candidates if x.price >= current_price]

        if not candidates:
            return None

        return min(candidates, key=lambda x: abs(x.price - current_price))

    def _rejection_confidence(self, level: SupportResistanceLevel, candle: Candle) -> float:
        wick_ratio = (
            candle.upper_wick_ratio
            if level.level_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}
            else candle.lower_wick_ratio
        )
        raw = (level.strength + min(1.0, wick_ratio)) / 2.0
        return clamp_unit(raw)

    def _break_confidence(self, level: SupportResistanceLevel, candle: Candle) -> float:
        move_pct = abs(candle.close - level.price) / max(level.price, 1e-9)
        raw = (level.strength + candle.body_ratio + min(1.0, move_pct * 100.0)) / 3.0
        return clamp_unit(raw)

    def _create_event(
        self,
        *,
        event_type: SREventType,
        level: SupportResistanceLevel,
        timestamp: Any,
        reference_price: float | None,
        confidence: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> SupportResistanceEvent:
        event = SupportResistanceEvent(
            event_id=uuid4().hex,
            event_type=event_type,
            timestamp=timestamp,
            symbol=self.symbol,
            timeframe=self.timeframe,
            layer=level.layer,
            level_id=level.level_id,
            level_type=level.level_type,
            price=level.price,
            confidence=clamp_unit(confidence),
            reference_price=reference_price,
            metadata=dict(metadata or {}),
        )
        self._events.append(event)
        return event

    def _level_to_dict(self, level: SupportResistanceLevel) -> dict[str, Any]:
        return self._safe_serialize(level)

    def _event_to_dict(self, event: SupportResistanceEvent) -> dict[str, Any]:
        return self._safe_serialize(event)


__all__ = [
    "SupportResistanceConfig",
    "SupportResistanceAnalyzer",
]