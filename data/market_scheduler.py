from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler
from data.market_snapshots import MarketSnapshot
from data.market_state import MarketStateStore


SnapshotCallback = Callable[[MarketSnapshot], Awaitable[None] | None]
BatchSnapshotCallback = Callable[[list[MarketSnapshot]], Awaitable[None] | None]


@dataclass(slots=True)
class MarketSchedulerConfig:
    enabled: bool = True
    interval_seconds: float = 1.0
    batch_size: int = 100
    snapshot_depth: int = 50
    run_immediately: bool = False
    job_timeout_seconds: float = 30.0
    emit_snapshot_ready_events: bool = False
    emit_empty_ticks: bool = False
    service_name: str = "market_scheduler"
    # Skip evaluation when EventBus queue utilization exceeds this threshold.
    # Prevents analytics from generating new events that cannot be enqueued,
    # which would cause evaluators to hang on safe_emit().
    eventbus_skip_utilization_threshold: float = 0.80

    def validate(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.snapshot_depth <= 0:
            raise ValueError("snapshot_depth must be > 0")


@dataclass(slots=True)
class MarketSchedulerStats:
    ticks: int = 0
    empty_ticks: int = 0
    snapshots_evaluated: int = 0
    callback_errors: int = 0
    skipped_overlap: int = 0
    skipped_eventbus_saturated: int = 0
    last_tick_snapshot_count: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


@dataclass(slots=True)
class RegisteredEvaluator:
    name: str
    callback: SnapshotCallback | None = None
    batch_callback: BatchSnapshotCallback | None = None
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class MarketScheduler:
    """
    Controlled snapshot evaluator for state-driven analytics.

    It drains dirty scopes from MarketStateStore at a bounded cadence, builds
    copy-on-read snapshots, and invokes registered analytics evaluators. It does
    not transport raw market data through EventBus.
    """

    def __init__(
        self,
        *,
        state_store: MarketStateStore,
        scheduler: Scheduler,
        event_bus: EventBus | None = None,
        config: MarketSchedulerConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        self.state_store = state_store
        self.scheduler = scheduler
        self.event_bus = event_bus
        self.config = config or MarketSchedulerConfig()
        self.config.validate()
        self._service_name = service_name or self.config.service_name
        self._logger = get_logger(__name__, service=self._service_name, event_type="market_scheduler")
        self._evaluators: dict[str, RegisteredEvaluator] = {}
        self._job_id: str | None = None
        self._running = False
        self._tick_lock = asyncio.Lock()
        self._stats = MarketSchedulerStats()

    def register_evaluator(
        self,
        name: str,
        callback: SnapshotCallback | None = None,
        *,
        batch_callback: BatchSnapshotCallback | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if callback is None and batch_callback is None:
            raise ValueError("callback or batch_callback is required")
        self._evaluators[name] = RegisteredEvaluator(
            name=name,
            callback=callback,
            batch_callback=batch_callback,
            enabled=enabled,
            metadata=dict(metadata or {}),
        )
        self._logger.info("Market snapshot evaluator registered | name=%s enabled=%s", name, enabled)

    def unregister_evaluator(self, name: str) -> None:
        self._evaluators.pop(name, None)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self.config.enabled:
            self._job_id = self.scheduler.add_interval_job(
                name="market-snapshot-evaluation",
                func=self.evaluate_dirty_once,
                interval=self.config.interval_seconds,
                run_immediately=self.config.run_immediately,
                max_retries=0,
                retry_delay=1.0,
                timeout=self.config.job_timeout_seconds,
                allow_overlap=False,
                enabled=True,
            )
        self._logger.info(
            "MarketScheduler started | interval=%s batch_size=%s evaluators=%s job_id=%s",
            self.config.interval_seconds,
            self.config.batch_size,
            len(self._evaluators),
            self._job_id,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._job_id is not None:
            self.scheduler.remove_job(self._job_id)
            self._job_id = None
        self._logger.info("MarketScheduler stopped | stats=%s", self._stats.to_dict())

    async def evaluate_dirty_once(self) -> dict[str, Any]:
        if self._tick_lock.locked():
            self._stats.skipped_overlap += 1
            return {"skipped": True, "reason": "previous_tick_running"}

        # Guard: skip evaluation when EventBus is saturated to avoid evaluators
        # hanging on safe_emit() calls and making queue congestion worse.
        if self.event_bus is not None:
            utilization = self.event_bus.stats().get("queue_utilization", 0.0)
            threshold = self.config.eventbus_skip_utilization_threshold
            if utilization > threshold:
                self._stats.skipped_eventbus_saturated += 1
                self._logger.warning(
                    "Skipping evaluation tick — EventBus saturated | utilization=%.2f threshold=%.2f skipped_total=%s",
                    utilization,
                    threshold,
                    self._stats.skipped_eventbus_saturated,
                )
                return {"skipped": True, "reason": "eventbus_saturated", "utilization": utilization}

        async with self._tick_lock:
            self._stats.ticks += 1
            snapshots = await self.state_store.snapshots_for_dirty(
                limit=self.config.batch_size,
                depth=self.config.snapshot_depth,
            )
            self._stats.last_tick_snapshot_count = len(snapshots)
            if not snapshots:
                self._stats.empty_ticks += 1
                if self.config.emit_empty_ticks:
                    await self._emit_snapshot_ready([])
                return {"snapshots": 0, "evaluators": len(self._evaluators)}

            enabled = [item for item in self._evaluators.values() if item.enabled]
            for evaluator in enabled:
                try:
                    if evaluator.batch_callback is not None:
                        result = evaluator.batch_callback(snapshots)
                        if inspect.isawaitable(result):
                            await result
                    if evaluator.callback is not None:
                        for snapshot in snapshots:
                            result = evaluator.callback(snapshot)
                            if inspect.isawaitable(result):
                                await result
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._stats.callback_errors += 1
                    self._stats.last_error = str(exc)
                    self._logger.exception(
                        "Market snapshot evaluator failed | evaluator=%s snapshots=%s",
                        evaluator.name,
                        len(snapshots),
                    )

            self._stats.snapshots_evaluated += len(snapshots)
            await self._emit_snapshot_ready(snapshots)
            return {
                "snapshots": len(snapshots),
                "evaluators": len(enabled),
                "callback_errors": self._stats.callback_errors,
            }

    async def _emit_snapshot_ready(self, snapshots: list[MarketSnapshot]) -> None:
        if not self.config.emit_snapshot_ready_events or self.event_bus is None:
            return
        await self.event_bus.emit(
            "market.state.snapshot_ready",
            {
                "snapshot_count": len(snapshots),
                "symbols": [snapshot.scope.symbol for snapshot in snapshots[:50]],
                "scopes": [snapshot.scope.to_dict() for snapshot in snapshots[:50]],
                "stats": self._stats.to_dict(),
            },
            priority=EventPriority.LOW,
            source=self._service_name,
        )

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats.to_dict(),
            "running": self._running,
            "job_id": self._job_id,
            "evaluators": list(self._evaluators),
        }