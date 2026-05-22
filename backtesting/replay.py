from __future__ import annotations

import asyncio
from typing import Iterable
from uuid import uuid4

from core.event_bus import Event, EventBus, EventPriority
from core.logger import get_logger

from backtesting.binance_history import HistoricalDataset
from backtesting.config import BacktestConfig
from backtesting.exceptions import BacktestReplayError
from backtesting.models import ReplayEvent
from backtesting.utils import drain_event_bus, sort_events_causally


class HistoricalMarketReplay:
    """
    Convert historical rows into production-compatible EventBus market.* events.

    Batch drains are non-fatal. During heavy production analytics replay, a queue.join()
    can time out because of slow in-flight handlers or scheduler-derived work. Backtest
    should continue and record the queue state instead of crashing the run.
    """

    def __init__(self, *, config: BacktestConfig, event_bus: EventBus) -> None:
        self._config = config
        self._event_bus = event_bus
        self._replay_id = f"replay_{uuid4().hex}"
        self._logger = get_logger(
            __name__,
            service="backtesting.replay",
            event_type="historical_replay",
        )

    @property
    def replay_id(self) -> str:
        return self._replay_id

    def build_events(self, dataset: HistoricalDataset, *, backtest_id: str) -> list[ReplayEvent]:
        events: list[ReplayEvent] = []
        sequence = 0

        def append(topic: str, timestamp_ms: int, payload: dict) -> None:
            nonlocal sequence
            events.append(
                ReplayEvent(
                    topic=topic,
                    timestamp_ms=timestamp_ms,
                    payload=payload,
                    sequence=sequence,
                )
            )
            sequence += 1

        for candle in dataset.candles:
            append(
                "market.candle",
                candle.close_time_ms,
                candle.to_market_payload(replay_id=self._replay_id, backtest_id=backtest_id),
            )

        for funding in dataset.funding_rates:
            append(
                "market.funding",
                funding.funding_time_ms,
                funding.to_market_payload(replay_id=self._replay_id, backtest_id=backtest_id),
            )

        for oi in dataset.open_interest:
            append(
                "market.open_interest",
                oi.timestamp_ms,
                oi.to_market_payload(replay_id=self._replay_id, backtest_id=backtest_id),
            )

        for trade in dataset.trades:
            append(
                "market.trade",
                trade.timestamp_ms,
                trade.to_market_payload(replay_id=self._replay_id, backtest_id=backtest_id),
            )

        for mark in dataset.mark_prices:
            append(
                "market.mark_price",
                mark.close_time_ms,
                mark.to_market_payload(replay_id=self._replay_id, backtest_id=backtest_id),
            )

        sorted_events = sort_events_causally(events)
        deduped_events = self._dedupe_events(sorted_events)

        self._logger.info(
            "Historical replay events built | events=%s removed_duplicates=%s replay_id=%s",
            len(deduped_events),
            len(sorted_events) - len(deduped_events),
            self._replay_id,
        )
        return deduped_events

    async def replay(self, events: Iterable[ReplayEvent]) -> int:
        events_list = list(events)
        last_timestamp_ms: int | None = None
        count = 0
        drain_every = max(1, int(self._config.replay_drain_every_events))

        for replay_event in events_list:
            if last_timestamp_ms is not None and replay_event.timestamp_ms < last_timestamp_ms:
                raise BacktestReplayError("Replay events are not sorted causally.")

            last_timestamp_ms = replay_event.timestamp_ms

            await self._publish(replay_event)
            count += 1

            if count % drain_every == 0:
                await self._drain_batch(
                    replayed_events=count,
                    total_events=len(events_list),
                )

            if self._config.replay_speed == "realistic" and self._config.accelerated_delay_multiplier > 0:
                await asyncio.sleep(self._config.accelerated_delay_multiplier)

        await self._drain_final()

        self._logger.info(
            "Historical replay completed | events=%s queue_size=%s replay_id=%s",
            count,
            self._queue_size(),
            self._replay_id,
        )
        return count

    async def _publish(self, replay_event: ReplayEvent) -> None:
        result = self._event_bus.publish(
            Event(
                topic=replay_event.topic,
                payload=replay_event.payload,
                priority=EventPriority.NORMAL,
                timestamp=replay_event.timestamp_ms / 1000.0,
                source="backtesting.historical_replay",
                headers={
                    "mode": "backtest",
                    "replay_id": self._replay_id,
                    "backtest_id": replay_event.payload.get("backtest_id"),
                    "replay_time": True,
                },
            )
        )

        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            result = await result

        # Some EventBus implementations return None on success.
        # Treat only explicit False as rejection.
        if result is False:
            raise BacktestReplayError(
                f"EventBus rejected replay event topic={replay_event.topic}"
            )

    async def _drain_batch(self, *, replayed_events: int, total_events: int) -> None:
        drained = await drain_event_bus(
            self._event_bus,
            require_public_join=self._config.require_event_bus_join,
            timeout=float(self._config.replay_batch_drain_timeout_seconds),
            raise_on_timeout=False,
        )

        queue_size = self._queue_size()

        if drained:
            self._logger.info(
                "Replay batch drained | replayed_events=%s total_events=%s queue_size=%s replay_id=%s",
                replayed_events,
                total_events,
                queue_size,
                self._replay_id,
            )
            return

        if queue_size is not None and queue_size <= self._config.low_queue_size_threshold:
            self._logger.info(
                "Replay batch drain timed out with low queue size; continuing | "
                "replayed_events=%s total_events=%s queue_size=%s replay_id=%s",
                replayed_events,
                total_events,
                queue_size,
                self._replay_id,
            )
            return

        self._logger.warning(
            "Replay batch drain timed out; continuing because pipeline is still active | "
            "replayed_events=%s total_events=%s queue_size=%s replay_id=%s",
            replayed_events,
            total_events,
            queue_size,
            self._replay_id,
        )

    async def _drain_final(self) -> None:
        drained = await drain_event_bus(
            self._event_bus,
            require_public_join=self._config.require_event_bus_join,
            timeout=float(self._config.replay_final_drain_timeout_seconds),
            raise_on_timeout=False,
        )

        if drained:
            self._logger.info(
                "Final replay drain completed | queue_size=%s replay_id=%s",
                self._queue_size(),
                self._replay_id,
            )
            return

        self._logger.warning(
            "Final replay drain timed out; report may miss late async events | "
            "queue_size=%s replay_id=%s",
            self._queue_size(),
            self._replay_id,
        )

    def _dedupe_events(self, events: list[ReplayEvent]) -> list[ReplayEvent]:
        seen: set[tuple[str, str, str, str, str, int]] = set()
        result: list[ReplayEvent] = []

        for event in events:
            payload = event.payload
            exchange = str(payload.get("exchange") or "")
            market_type = str(payload.get("market_type") or "")
            symbol = str(payload.get("symbol") or payload.get("exchange_symbol") or "")
            timeframe = str(payload.get("timeframe") or "")
            timestamp_ms = int(
                payload.get("timestamp_ms")
                or payload.get("close_time_ms")
                or payload.get("funding_time_ms")
                or payload.get("open_time_ms")
                or event.timestamp_ms
                or 0
            )

            key = (event.topic, exchange, market_type, symbol, timeframe, timestamp_ms)
            if key in seen:
                continue

            seen.add(key)
            result.append(event)

        return result

    def _queue_size(self) -> int | None:
        queue = getattr(self._event_bus, "_queue", None)
        if queue is None:
            return None

        qsize = getattr(queue, "qsize", None)
        if not callable(qsize):
            return None

        try:
            return int(qsize())
        except Exception:
            return None
