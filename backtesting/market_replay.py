"""
Historical market replay for backtesting.

MarketReplay is the bridge between historical data and the production-style
event-driven trading pipeline.

It takes replay-ready BacktestDataset/BacktestEvent objects and emits the same
raw market.* topics that exchange adapters would emit in live trading:

- market.candle
- market.trade
- market.orderbook
- market.funding
- market.open_interest
- market.liquidation

The rest of the system should not care whether events came from live exchange
adapters or from historical replay.

Important:
- MarketReplay does not emit market.*.updated.
- MarketReplay does not run analytics directly.
- MarketReplay does not run strategies directly.
- MarketReplay does not call RiskManager directly.
- MarketReplay does not simulate execution.
- MarketReplay only advances simulated time and emits historical raw market events.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.event_bus import EventBus, EventPriority
from core.logger import get_logger

from backtesting.backtest_time import BacktestClock
from backtesting.config import MarketReplayConfig
from backtesting.enums import (
    BacktestDataType,
    BacktestEventType,
    BacktestStatus,
    ReplayEventPriority,
    ReplayMode,
    WarmupPolicy,
)
from backtesting.exceptions import (
    MarketReplayAlreadyRunningError,
    MarketReplayError,
    MarketReplayNotPreparedError,
    MarketReplayPausedError,
    MarketReplayStoppedError,
    ReplayEmitError,
    ReplayEventError,
    ReplayOrderingError,
    ReplaySeekError,
)
from backtesting.models import (
    BacktestDataset,
    BacktestEvent,
    BacktestPeriod,
    ReplayEventBatch,
    SerializableMixin,
    datetime_from_ms,
    timestamp_ms,
    utcnow,
)


@dataclass(slots=True)
class MarketReplayStats(SerializableMixin):
    """
    Runtime replay stats.
    """

    status: BacktestStatus = BacktestStatus.CREATED
    total_events: int = 0
    processed_events: int = 0
    emitted_events: int = 0
    skipped_events: int = 0
    failed_events: int = 0

    market_candles: int = 0
    market_trades: int = 0
    market_orderbooks: int = 0
    market_funding: int = 0
    market_open_interest: int = 0
    market_liquidations: int = 0

    warmup_events: int = 0
    trading_events: int = 0

    current_index: int = 0
    current_timestamp_ms: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def progress_pct(self) -> float:
        if self.total_events <= 0:
            return 0.0
        return min(100.0, max(0.0, self.processed_events / self.total_events * 100.0))


@dataclass(slots=True)
class MarketReplayCheckpoint(SerializableMixin):
    """
    Replay checkpoint for pause/resume/seek.
    """

    index: int
    timestamp_ms: int
    processed_events: int
    emitted_events: int
    skipped_events: int
    failed_events: int
    created_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class MarketReplay:
    """
    Deterministic historical market replay.

    Event flow:

        BacktestDataset.events
            -> BacktestClock.advance_to(event.timestamp_ms)
            -> core.EventBus.emit("market.*", payload)
            -> production data caches
            -> market.*.updated
            -> analytics
            -> strategy
            -> risk
            -> execution simulator
            -> position simulator
    """

    component_name = "MarketReplay"

    def __init__(
        self,
        config: MarketReplayConfig | None = None,
        *,
        event_bus: EventBus | None = None,
        clock: BacktestClock | None = None,
        logger_name: str = "backtesting.market_replay",
    ) -> None:
        self.config = config or MarketReplayConfig()
        self.config.validate()

        self.event_bus = event_bus
        self.clock = clock
        self.logger = get_logger(logger_name)

        self.dataset: BacktestDataset | None = None
        self.stats_state = MarketReplayStats()
        self._checkpoints: list[MarketReplayCheckpoint] = []

        self._running = False
        self._paused = False
        self._stopped = False
        self._prepared = False
        self._registered = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Lifecycle-compatible no-op.

        MarketReplay does not subscribe to production topics. It only emits raw
        market.* replay events when replay() / replay_step() is called.
        """

        self._registered = True

    def prepare(
        self,
        dataset: BacktestDataset,
        *,
        clock: BacktestClock | None = None,
    ) -> None:
        """
        Prepare replay with dataset and optional clock.
        """

        if dataset.is_empty:
            raise MarketReplayNotPreparedError(
                "Cannot prepare MarketReplay with an empty dataset."
            )

        if self.event_bus is None:
            raise MarketReplayNotPreparedError(
                "MarketReplay requires core EventBus before prepare()."
            )

        if not isinstance(self.event_bus, EventBus):
            raise MarketReplayNotPreparedError(
                "MarketReplay requires core.event_bus.EventBus for full-pipeline replay."
            )

        self.dataset = dataset

        if clock is not None:
            self.clock = clock

        if self.clock is None:
            period = dataset.info.period

            if period is None:
                first_time = dataset.events[0].event_time
                last_time = dataset.events[-1].event_time
                period = BacktestPeriod(start=first_time, end=last_time)

            self.clock = BacktestClock(period)

        self._validate_dataset_ordering(dataset)

        self.stats_state = MarketReplayStats(
            status=BacktestStatus.CREATED,
            total_events=len(dataset.events),
            current_index=0,
            metadata={
                "event_bus_type": self.event_bus.__class__.__name__,
                "raw_market_topics": self.raw_market_topics(),
            },
        )

        self._prepared = True
        self._running = False
        self._paused = False
        self._stopped = False
        self._checkpoints.clear()

    async def start(self) -> None:
        """
        Start replay lifecycle.

        Does not start analytics/strategy/risk/execution. StrategyTester owns
        component lifecycle. MarketReplay only starts its own replay state and
        simulated clock if the clock is not already started.
        """

        async with self._lock:
            self._ensure_prepared()

            if self._running:
                raise MarketReplayAlreadyRunningError("MarketReplay is already running.")

            assert self.clock is not None
            assert self.dataset is not None

            if not self.clock.started:
                self.clock.start(total_events=len(self.dataset.events))

            self._running = True
            self._paused = False
            self._stopped = False

            self.stats_state.status = BacktestStatus.RUNNING
            self.stats_state.started_at = utcnow()

        if self.config.emit_replay_lifecycle_events:
            await self._emit_lifecycle(
                self.config.replay_started_topic,
                {
                    "total_events": self.stats_state.total_events,
                    "replay_mode": self.config.replay_mode.value,
                    "replay_speed": self.config.replay_speed.value,
                    "raw_market_topics": self.raw_market_topics(),
                },
            )

    async def stop(self) -> None:
        """
        Stop replay lifecycle.

        MarketReplay may stop its simulated clock, but it does not stop the
        production EventBus or any injected production components.
        """

        async with self._lock:
            if not self._prepared:
                return

            already_stopped = self._stopped

            self._running = False
            self._paused = False
            self._stopped = True

            self.stats_state.status = (
                BacktestStatus.COMPLETED
                if self.stats_state.failed_events == 0
                else BacktestStatus.FAILED
            )
            self.stats_state.finished_at = utcnow()

            if self.clock is not None and self.clock.started and not self.clock.stopped:
                self.clock.stop()

        if already_stopped:
            return

        if self.config.emit_replay_lifecycle_events:
            await self._emit_lifecycle(
                self.config.replay_finished_topic,
                self.stats(),
            )

    async def pause(self) -> MarketReplayCheckpoint:
        """
        Pause replay and return checkpoint.
        """

        async with self._lock:
            self._ensure_running()

            self._paused = True
            self.stats_state.status = BacktestStatus.PAUSED

            checkpoint = self.create_checkpoint()
            self._checkpoints.append(checkpoint)

            return checkpoint

    async def resume(self) -> None:
        """
        Resume replay from pause.
        """

        async with self._lock:
            self._ensure_prepared()

            if self._stopped:
                raise MarketReplayStoppedError("Cannot resume stopped MarketReplay.")

            if not self._paused:
                return

            self._paused = False
            self._running = True
            self.stats_state.status = BacktestStatus.RUNNING

    # ------------------------------------------------------------------
    # Main replay methods
    # ------------------------------------------------------------------

    async def replay(self) -> MarketReplayStats:
        """
        Replay all remaining events.
        """

        self._ensure_prepared()

        if not self._running:
            await self.start()

        if self.config.replay_mode == ReplayMode.STEP_BY_STEP:
            raise MarketReplayError(
                "MarketReplay.replay() cannot be used in STEP_BY_STEP mode. "
                "Use replay_step()."
            )

        if self.config.batch_events_by_timestamp or self.config.replay_mode == ReplayMode.BATCHED:
            return await self._replay_batches()

        return await self._replay_events()

    async def replay_step(self) -> BacktestEvent | None:
        """
        Replay one event.

        Returns the replayed event or None if finished.
        """

        self._ensure_prepared()

        if not self._running:
            await self.start()

        if self._paused:
            raise MarketReplayPausedError("Cannot replay step while replay is paused.")

        assert self.dataset is not None

        if self.stats_state.current_index >= len(self.dataset.events):
            await self.stop()
            return None

        event = self.dataset.events[self.stats_state.current_index]
        await self._process_event(event, index=self.stats_state.current_index)

        self.stats_state.current_index += 1

        if self.stats_state.current_index >= len(self.dataset.events):
            await self.stop()

        return event

    async def _replay_events(self) -> MarketReplayStats:
        """
        Replay event-by-event.
        """

        assert self.dataset is not None

        while self.stats_state.current_index < len(self.dataset.events):
            if self._paused:
                break

            if self._stopped:
                raise MarketReplayStoppedError("MarketReplay stopped during replay.")

            event = self.dataset.events[self.stats_state.current_index]
            await self._process_event(event, index=self.stats_state.current_index)

            self.stats_state.current_index += 1

            if self.config.yield_every_events > 0:
                if self.stats_state.processed_events % self.config.yield_every_events == 0:
                    await asyncio.sleep(0)

            await self._maybe_emit_progress()

        if self.stats_state.current_index >= len(self.dataset.events):
            await self.stop()

        return self.stats_state

    async def _replay_batches(self) -> MarketReplayStats:
        """
        Replay events grouped by timestamp.
        """

        assert self.dataset is not None

        batches = self.dataset.batches_by_timestamp()
        event_index = self.stats_state.current_index
        batch_start_index = 0

        for batch in batches:
            if self._paused:
                break

            if self._stopped:
                raise MarketReplayStoppedError("MarketReplay stopped during batch replay.")

            batch_end_index = batch_start_index + batch.size

            if batch_end_index <= event_index:
                batch_start_index = batch_end_index
                continue

            await self._process_batch(
                batch,
                start_index=batch_start_index,
                skip_before_index=event_index,
            )

            self.stats_state.current_index = batch_end_index
            batch_start_index = batch_end_index

            if self.config.yield_every_events > 0:
                if self.stats_state.processed_events % self.config.yield_every_events == 0:
                    await asyncio.sleep(0)

            await self._maybe_emit_progress()

        if self.stats_state.current_index >= len(self.dataset.events):
            await self.stop()

        return self.stats_state

    async def _process_batch(
        self,
        batch: ReplayEventBatch,
        *,
        start_index: int,
        skip_before_index: int = 0,
    ) -> None:
        """
        Process one timestamp batch.
        """

        assert self.clock is not None

        await self.clock.advance_to_async(
            batch.timestamp_ms,
            events_processed_increment=0,
            allow_equal=True,
            run_due_jobs=True,
        )

        for offset, event in enumerate(batch.events):
            index = start_index + offset

            if index < skip_before_index:
                continue

            emitted = await self._emit_replay_event(event)
            self._update_stats_after_event(event, emitted=emitted)
            self.stats_state.current_index = index

    async def _process_event(
        self,
        event: BacktestEvent,
        *,
        index: int,
    ) -> None:
        """
        Advance clock and emit one event.
        """

        assert self.clock is not None

        await self.clock.advance_to_async(
            event.timestamp_ms,
            events_processed_increment=0,
            allow_equal=True,
            run_due_jobs=True,
        )

        try:
            emitted = await self._emit_replay_event(event)
            self._update_stats_after_event(event, emitted=emitted)
            self.stats_state.current_index = index

        except Exception:
            self._update_stats_after_event(event, emitted=False)
            raise

    # ------------------------------------------------------------------
    # Emit logic
    # ------------------------------------------------------------------

    async def _emit_replay_event(self, event: BacktestEvent) -> bool:
        """
        Emit one replay event through the production EventBus.

        Returns:
            True  - event was emitted;
            False - event was intentionally skipped.
        """

        if event.event_type != BacktestEventType.MARKET:
            return False

        if not self._should_emit_event(event):
            return False

        topic = self._resolve_topic(event)
        payload = self._build_payload(event)

        if not topic:
            raise ReplayEventError(
                "Replay event has no topic.",
                details={"event_id": event.event_id},
            )

        if not topic.startswith("market."):
            raise ReplayEventError(
                "MarketReplay is allowed to emit only raw market.* topics.",
                details={
                    "event_id": event.event_id,
                    "topic": topic,
                },
            )

        if topic.endswith(".updated"):
            raise ReplayEventError(
                "MarketReplay must not emit market.*.updated topics. "
                "Updated events are produced by production data caches.",
                details={
                    "event_id": event.event_id,
                    "topic": topic,
                },
            )

        if self.event_bus is None:
            raise ReplayEmitError("MarketReplay requires EventBus to emit market events.")

        try:
            await self._emit(
                topic,
                payload,
                priority=self._event_priority(event),
                source="MarketReplay",
                headers=self._event_headers(event, topic),
            )
            return True

        except Exception as exc:
            self.stats_state.failed_events += 1
            self.stats_state.last_error = str(exc)

            if self.config.fail_on_emit_error:
                raise ReplayEmitError(
                    "Failed to emit replay market event.",
                    details={
                        "event_id": event.event_id,
                        "topic": topic,
                        "timestamp_ms": event.timestamp_ms,
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    },
                ) from exc

            self.logger.warning(
                "Failed to emit replay event",
                extra={
                    "event_id": event.event_id,
                    "topic": topic,
                    "timestamp_ms": event.timestamp_ms,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            )
            return False

    async def _emit(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str = "MarketReplay",
        headers: dict[str, Any] | None = None,
    ) -> None:
        """
        Emit through production core.EventBus.
        """

        if self.event_bus is None:
            raise ReplayEmitError("EventBus is not configured.")

        emit = getattr(self.event_bus, "emit", None)

        if not callable(emit):
            raise ReplayEmitError(
                "EventBus does not expose emit().",
                details={"event_bus_type": self.event_bus.__class__.__name__},
            )

        result = emit(
            topic,
            payload,
            priority=priority,
            source=source,
            headers=headers or {},
        )

        if inspect.isawaitable(result):
            await result

    async def _emit_lifecycle(self, topic: str, payload: dict[str, Any]) -> None:
        """
        Best-effort lifecycle event emission.
        """

        if self.event_bus is None:
            return

        try:
            await self._emit(
                topic,
                {
                    **payload,
                    "source": "market_replay",
                    "timestamp_ms": self.clock.timestamp_ms_or_wall_clock()
                    if self.clock is not None
                    else timestamp_ms(utcnow()),
                },
                priority=EventPriority.LOW,
                source="MarketReplay",
                headers={
                    "component": "backtesting.market_replay",
                    "event_type": "backtest_lifecycle",
                },
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to emit replay lifecycle event",
                extra={
                    "topic": topic,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            )

    async def _maybe_emit_progress(self) -> None:
        """
        Emit replay progress periodically.
        """

        if not self.config.emit_replay_lifecycle_events:
            return

        interval = self.config.progress_interval_events

        if interval <= 0:
            return

        processed = self.stats_state.processed_events

        if processed <= 0 or processed % interval != 0:
            return

        await self._emit_lifecycle(
            self.config.replay_progress_topic,
            {
                "processed_events": processed,
                "total_events": self.stats_state.total_events,
                "progress_pct": self.stats_state.progress_pct,
                "current_index": self.stats_state.current_index,
                "current_timestamp_ms": self.stats_state.current_timestamp_ms,
            },
        )

    # ------------------------------------------------------------------
    # Event resolution / filters
    # ------------------------------------------------------------------

    def _should_emit_event(self, event: BacktestEvent) -> bool:
        """
        Return whether event should be emitted.
        """

        if event.is_warmup:
            if self.config.warmup_policy == WarmupPolicy.NONE:
                return False

            if not self.config.emit_warmup_events:
                return False

        topic = self._resolve_topic(event)

        if topic == self.config.market_candle_topic:
            return self.config.emit_market_candles

        if topic == self.config.market_trade_topic:
            return self.config.emit_market_trades

        if topic == self.config.market_orderbook_topic:
            return self.config.emit_market_orderbook

        if topic == self.config.market_funding_topic:
            return self.config.emit_market_funding

        if topic == self.config.market_open_interest_topic:
            return self.config.emit_market_open_interest

        if topic == self.config.market_liquidation_topic:
            return self.config.emit_market_liquidations

        # Unknown market.* raw topics are allowed for future extensions.
        return topic.startswith("market.") and not topic.endswith(".updated")

    def _resolve_topic(self, event: BacktestEvent) -> str:
        """
        Resolve production raw market topic.

        BacktestEvent should already carry a topic, but this method provides
        fallback based on metadata["data_type"].
        """

        if event.topic:
            return event.topic

        data_type_value = event.metadata.get("data_type")

        try:
            data_type = BacktestDataType(data_type_value)
        except Exception:
            return ""

        return market_topic_for_data_type(data_type, config=self.config)

    def _build_payload(self, event: BacktestEvent) -> dict[str, Any]:
        """
        Add replay metadata to market payload without overwriting domain fields.
        """

        payload = dict(event.payload)

        payload.setdefault("timestamp_ms", event.timestamp_ms)
        payload.setdefault("received_at_ms", event.timestamp_ms)
        payload.setdefault("source", "market_replay")
        payload.setdefault("replay_event_id", event.event_id)
        payload.setdefault("replay_sequence", event.sequence)
        payload.setdefault("run_id", event.run_id)

        if self.config.mark_warmup_payloads:
            payload.setdefault("is_warmup", event.is_warmup)

        data_type = event.metadata.get("data_type")
        if data_type is not None:
            payload.setdefault("data_type", data_type)

        metadata = dict(payload.get("metadata") or {})
        metadata.update(
            {
                "backtest": True,
                "replay_event_id": event.event_id,
                "replay_source": event.source,
                "replay_sequence": event.sequence,
                "replay_is_warmup": event.is_warmup,
                "replay_timestamp_ms": event.timestamp_ms,
                "data_type": data_type,
                "record_type": event.metadata.get("record_type"),
                "instrument_key": event.metadata.get("instrument_key"),
            }
        )
        payload["metadata"] = metadata

        return payload

    def _event_priority(self, event: BacktestEvent) -> EventPriority:
        """
        Map replay priority to core EventBus priority.

        Market replay events generally use NORMAL priority. Higher priority is
        reserved for lifecycle/errors; deterministic ordering is controlled by
        BacktestDataset sorting, not EventBus priority.
        """

        if event.priority in {
            ReplayEventPriority.LIQUIDATION,
            ReplayEventPriority.ORDERBOOK,
        }:
            return EventPriority.NORMAL

        if event.priority in {
            ReplayEventPriority.FUNDING,
            ReplayEventPriority.OPEN_INTEREST,
            ReplayEventPriority.MARK_PRICE,
            ReplayEventPriority.INDEX_PRICE,
        }:
            return EventPriority.NORMAL

        return EventPriority.NORMAL

    @staticmethod
    def _event_headers(event: BacktestEvent, topic: str) -> dict[str, Any]:
        return {
            "component": "backtesting.market_replay",
            "topic": topic,
            "replay_event_id": event.event_id,
            "replay_sequence": event.sequence,
            "is_warmup": event.is_warmup,
            "data_type": event.metadata.get("data_type"),
            "record_type": event.metadata.get("record_type"),
        }

    # ------------------------------------------------------------------
    # Seeking / checkpoints
    # ------------------------------------------------------------------

    def create_checkpoint(self) -> MarketReplayCheckpoint:
        """
        Create replay checkpoint.
        """

        return MarketReplayCheckpoint(
            index=self.stats_state.current_index,
            timestamp_ms=self.stats_state.current_timestamp_ms or 0,
            processed_events=self.stats_state.processed_events,
            emitted_events=self.stats_state.emitted_events,
            skipped_events=self.stats_state.skipped_events,
            failed_events=self.stats_state.failed_events,
            metadata={
                "status": self.stats_state.status.value,
                "progress_pct": self.stats_state.progress_pct,
            },
        )

    def checkpoints(self) -> list[MarketReplayCheckpoint]:
        """
        Return saved checkpoints.
        """

        return list(self._checkpoints)

    async def seek_to_index(self, index: int) -> MarketReplayCheckpoint:
        """
        Seek replay to event index.

        Intended for debugging. For deterministic full backtests, prefer running
        from the beginning.
        """

        async with self._lock:
            self._ensure_prepared()

            assert self.dataset is not None
            assert self.clock is not None

            if index < 0 or index >= len(self.dataset.events):
                raise ReplaySeekError(
                    "Replay index out of range.",
                    details={
                        "index": index,
                        "total_events": len(self.dataset.events),
                    },
                )

            event = self.dataset.events[index]

            self.stats_state.current_index = index
            self.stats_state.current_timestamp_ms = event.timestamp_ms
            self.stats_state.processed_events = index

            if event.timestamp_ms >= self.clock.timestamp_ms_or_wall_clock():
                self.clock.advance_to(event.timestamp_ms, allow_equal=True)
            elif self.clock.config.allow_time_travel_backwards:
                self.clock.advance_to(event.timestamp_ms, allow_equal=True)

            checkpoint = self.create_checkpoint()
            self._checkpoints.append(checkpoint)
            return checkpoint

    async def seek_to_timestamp(self, value: datetime | int | float) -> MarketReplayCheckpoint:
        """
        Seek to first event at or after timestamp.
        """

        self._ensure_prepared()

        assert self.dataset is not None

        target_ms = timestamp_ms(value)

        for index, event in enumerate(self.dataset.events):
            if event.timestamp_ms >= target_ms:
                return await self.seek_to_index(index)

        raise ReplaySeekError(
            "No replay event found at or after timestamp.",
            details={
                "target_timestamp_ms": target_ms,
                "target_time": datetime_from_ms(target_ms).isoformat(),
            },
        )

    # ------------------------------------------------------------------
    # Stats / diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """
        Return replay stats.
        """

        payload = self.stats_state.to_dict()

        if self.clock is not None and self.clock.started and not self.clock.stopped:
            payload["clock"] = self.clock.stats()

        payload.update(
            {
                "prepared": self._prepared,
                "registered": self._registered,
                "running": self._running,
                "paused": self._paused,
                "stopped": self._stopped,
                "checkpoints": len(self._checkpoints),
                "event_bus_type": self.event_bus.__class__.__name__
                if self.event_bus is not None
                else None,
                "raw_market_topics": self.raw_market_topics(),
            }
        )

        return payload

    def raw_market_topics(self) -> list[str]:
        return [
            self.config.market_candle_topic,
            self.config.market_trade_topic,
            self.config.market_orderbook_topic,
            self.config.market_funding_topic,
            self.config.market_open_interest_topic,
            self.config.market_liquidation_topic,
        ]

    # ------------------------------------------------------------------
    # Internal stats
    # ------------------------------------------------------------------

    def _update_stats_after_event(
        self,
        event: BacktestEvent,
        *,
        emitted: bool,
    ) -> None:
        self.stats_state.processed_events += 1
        self.stats_state.current_timestamp_ms = event.timestamp_ms

        if emitted:
            self.stats_state.emitted_events += 1
        else:
            self.stats_state.skipped_events += 1

        if event.is_warmup:
            self.stats_state.warmup_events += 1
        else:
            self.stats_state.trading_events += 1

        if not emitted:
            return

        topic = self._resolve_topic(event)

        if topic == self.config.market_candle_topic:
            self.stats_state.market_candles += 1
        elif topic == self.config.market_trade_topic:
            self.stats_state.market_trades += 1
        elif topic == self.config.market_orderbook_topic:
            self.stats_state.market_orderbooks += 1
        elif topic == self.config.market_funding_topic:
            self.stats_state.market_funding += 1
        elif topic == self.config.market_open_interest_topic:
            self.stats_state.market_open_interest += 1
        elif topic == self.config.market_liquidation_topic:
            self.stats_state.market_liquidations += 1

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _ensure_prepared(self) -> None:
        if not self._prepared or self.dataset is None:
            raise MarketReplayNotPreparedError(
                "MarketReplay is not prepared. Call prepare(dataset) first."
            )

        if self.clock is None:
            raise MarketReplayNotPreparedError("MarketReplay clock is missing.")

        if self.event_bus is None:
            raise MarketReplayNotPreparedError("MarketReplay EventBus is missing.")

    def _ensure_running(self) -> None:
        self._ensure_prepared()

        if not self._running:
            raise MarketReplayStoppedError("MarketReplay is not running.")

        if self._stopped:
            raise MarketReplayStoppedError("MarketReplay is stopped.")

    @staticmethod
    def _validate_dataset_ordering(dataset: BacktestDataset) -> None:
        """
        Validate monotonic event ordering.
        """

        previous_ts: int | None = None

        for event in dataset.events:
            if previous_ts is not None and event.timestamp_ms < previous_ts:
                raise ReplayOrderingError(
                    "BacktestDataset events must be sorted by timestamp.",
                    details={
                        "previous_timestamp_ms": previous_ts,
                        "current_timestamp_ms": event.timestamp_ms,
                        "event_id": event.event_id,
                    },
                )

            previous_ts = event.timestamp_ms


# ============================================================================
# Dataset builders
# ============================================================================


def market_topic_for_data_type(
    data_type: BacktestDataType,
    config: MarketReplayConfig | None = None,
) -> str:
    """
    Resolve default raw market topic for a historical data type.
    """

    replay_config = config or MarketReplayConfig()

    if data_type == BacktestDataType.CANDLES:
        return replay_config.market_candle_topic

    if data_type == BacktestDataType.TRADES:
        return replay_config.market_trade_topic

    if data_type in {BacktestDataType.ORDERBOOK, BacktestDataType.ORDERBOOK_SNAPSHOT}:
        return replay_config.market_orderbook_topic

    if data_type == BacktestDataType.FUNDING:
        return replay_config.market_funding_topic

    if data_type == BacktestDataType.OPEN_INTEREST:
        return replay_config.market_open_interest_topic

    if data_type == BacktestDataType.LIQUIDATIONS:
        return replay_config.market_liquidation_topic

    raise ReplayEventError(
        "Unsupported data type for market replay topic.",
        details={"data_type": data_type.value},
    )


def replay_priority_for_data_type(data_type: BacktestDataType) -> ReplayEventPriority | None:
    """
    Resolve deterministic replay priority for a historical data type.

    This priority is used by BacktestDataset sorting/replay ordering, not as a
    trading decision priority.
    """

    if data_type in {BacktestDataType.ORDERBOOK, BacktestDataType.ORDERBOOK_SNAPSHOT}:
        return ReplayEventPriority.ORDERBOOK

    if data_type == BacktestDataType.TRADES:
        return ReplayEventPriority.TRADE

    if data_type == BacktestDataType.CANDLES:
        return ReplayEventPriority.CANDLE

    if data_type == BacktestDataType.FUNDING:
        return ReplayEventPriority.FUNDING

    if data_type == BacktestDataType.OPEN_INTEREST:
        return ReplayEventPriority.OPEN_INTEREST

    if data_type == BacktestDataType.LIQUIDATIONS:
        return ReplayEventPriority.LIQUIDATION

    if data_type == BacktestDataType.MARK_PRICE:
        return ReplayEventPriority.MARK_PRICE

    if data_type == BacktestDataType.INDEX_PRICE:
        return ReplayEventPriority.INDEX_PRICE

    return None


def build_replay_event_from_record(
    record: Any,
    *,
    data_type: BacktestDataType,
    period: BacktestPeriod | None = None,
    run_id: str | None = None,
    sequence: int | None = None,
    config: MarketReplayConfig | None = None,
) -> BacktestEvent:
    """
    Build BacktestEvent from a Historical* record.

    The record must expose to_market_event_payload().
    """

    if not hasattr(record, "to_market_event_payload"):
        raise ReplayEventError(
            "Historical record cannot be converted to replay event.",
            details={
                "record_type": record.__class__.__name__,
                "data_type": data_type.value,
            },
        )

    record_timestamp_ms = getattr(record, "timestamp_ms", None)

    if record_timestamp_ms is None:
        record_timestamp_ms = getattr(record, "close_time_ms", None)

    if record_timestamp_ms is None:
        record_timestamp_ms = getattr(record, "open_time_ms", None)

    if record_timestamp_ms is None:
        raise ReplayEventError(
            "Historical record has no timestamp field.",
            details={
                "record_type": record.__class__.__name__,
                "data_type": data_type.value,
            },
        )

    topic = market_topic_for_data_type(data_type, config=config)
    priority = replay_priority_for_data_type(data_type)

    is_warmup = False

    if period is not None:
        is_warmup = period.is_warmup(record_timestamp_ms)

    payload = record.to_market_event_payload()

    return BacktestEvent(
        run_id=run_id,
        event_type=BacktestEventType.MARKET,
        topic=topic,
        timestamp_ms=int(record_timestamp_ms),
        payload=payload,
        source="market_replay",
        sequence=sequence,
        priority=priority,
        is_warmup=is_warmup,
        metadata={
            "data_type": data_type.value,
            "instrument_key": getattr(record, "instrument_key", None),
            "record_type": record.__class__.__name__,
        },
    )


def build_dataset_from_records(
    records_by_type: dict[BacktestDataType, list[Any]],
    *,
    period: BacktestPeriod | None = None,
    run_id: str | None = None,
    config: MarketReplayConfig | None = None,
) -> BacktestDataset:
    """
    Convenience builder for small tests or custom loaders.

    For production use, data_loader.py should build BacktestDataset.
    """

    events: list[BacktestEvent] = []
    sequence = 0

    for data_type, records in records_by_type.items():
        for record in records:
            events.append(
                build_replay_event_from_record(
                    record,
                    data_type=data_type,
                    period=period,
                    run_id=run_id,
                    sequence=sequence,
                    config=config,
                )
            )
            sequence += 1

    dataset = BacktestDataset(events=events)
    dataset.info.period = period
    dataset.info.total_events = len(events)
    dataset.info.data_types = set(records_by_type.keys())
    dataset.sort_events()

    return dataset


__all__ = [
    "MarketReplayStats",
    "MarketReplayCheckpoint",
    "MarketReplay",
    "market_topic_for_data_type",
    "replay_priority_for_data_type",
    "build_replay_event_from_record",
    "build_dataset_from_records",
]