"""
Backtest time and simulated clock utilities.

This module provides a deterministic simulated clock for the offline
backtesting pipeline. Backtest components should use this clock instead of
directly reading wall-clock UTC time when running historical replay.

Responsibilities:
- keep current simulated time;
- move time forward during market replay;
- protect against accidental time travel;
- expose timestamp helpers;
- provide warmup/progress state;
- optionally drive scheduler-like due jobs during replay.

This module does not own trading logic and does not emit market/strategy/risk
events by itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from inspect import isawaitable
from typing import Any

from backtesting.config import BacktestTimeConfig
from backtesting.exceptions import (
    BacktestClockError,
    BacktestClockNotInitializedError,
    BacktestTimeRangeError,
    BacktestTimeTravelError,
)
from backtesting.models import (
    BacktestClockState,
    BacktestPeriod,
    SerializableMixin,
    ensure_aware_utc,
    timestamp_ms,
    utcnow,
)


SchedulerJobCallback = Callable[[], Any | Awaitable[Any]]


@dataclass(slots=True)
class BacktestScheduledJob(SerializableMixin):
    """
    Scheduler-like job controlled by simulated backtest time.

    This is intentionally lightweight. It is not a replacement for the core
    Scheduler. It exists to make interval jobs deterministic in backtests.
    """

    job_id: str
    name: str
    callback: SchedulerJobCallback
    interval: timedelta
    next_run_at: datetime
    enabled: bool = True
    run_immediately: bool = False
    max_runs: int | None = None
    runs: int = 0
    last_run_at: datetime | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.next_run_at = ensure_aware_utc(self.next_run_at)

        if self.interval.total_seconds() <= 0:
            raise BacktestClockError(
                "BacktestScheduledJob.interval must be positive.",
                details={
                    "job_id": self.job_id,
                    "interval_seconds": self.interval.total_seconds(),
                },
            )

    @property
    def is_exhausted(self) -> bool:
        return self.max_runs is not None and self.runs >= self.max_runs

    def is_due(self, current_time: datetime) -> bool:
        if not self.enabled or self.is_exhausted:
            return False
        return ensure_aware_utc(current_time) >= self.next_run_at

    def advance_next_run(self) -> None:
        self.last_run_at = self.next_run_at
        self.runs += 1
        self.next_run_at = self.next_run_at + self.interval


@dataclass(slots=True)
class BacktestTimeSnapshot(SerializableMixin):
    """
    User-facing clock snapshot.
    """

    current_time: datetime
    current_timestamp_ms: int
    start_time: datetime
    end_time: datetime
    warmup_start_time: datetime | None
    is_warmup: bool
    progress_pct: float
    events_processed: int
    total_events: int
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BacktestClock:
    """
    Deterministic simulated clock for backtesting.

    The clock is monotonic by default: it can only move forward. This protects
    market replay, cooldown logic, risk guards and metrics from accidental
    non-deterministic ordering.

    Typical usage:

        period = BacktestPeriod(start=..., end=..., warmup_start=...)
        clock = BacktestClock(period)
        clock.start(total_events=len(dataset))
        clock.advance_to(event.timestamp_ms)
        current_time = clock.now()
    """

    def __init__(
        self,
        period: BacktestPeriod,
        config: BacktestTimeConfig | None = None,
    ) -> None:
        self.period = period
        self.config = config or BacktestTimeConfig()
        self.config.validate()

        initial_time = period.warmup_start or period.start

        self.state = BacktestClockState(
            period=period,
            current_time=initial_time,
            current_timestamp_ms=timestamp_ms(initial_time),
        )

        self._started = False
        self._stopped = False
        self._lock = asyncio.Lock()
        self._jobs: dict[str, BacktestScheduledJob] = {}

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def start(self, *, total_events: int = 0) -> None:
        """
        Start the simulated clock.
        """

        if total_events < 0:
            raise BacktestClockError(
                "total_events cannot be negative.",
                details={"total_events": total_events},
            )

        initial_time = self.period.warmup_start or self.period.start

        self.state.current_time = initial_time
        self.state.current_timestamp_ms = timestamp_ms(initial_time)
        self.state.started_at = utcnow()
        self.state.finished_at = None
        self.state.events_processed = 0
        self.state.total_events = total_events

        self._started = True
        self._stopped = False

    async def start_async(self, *, total_events: int = 0) -> None:
        """
        Async-safe start wrapper.
        """

        async with self._lock:
            self.start(total_events=total_events)

    def stop(self) -> None:
        """
        Stop the simulated clock.
        """

        self._ensure_started()

        self.state.finished_at = utcnow()
        self._stopped = True

    async def stop_async(self) -> None:
        """
        Async-safe stop wrapper.
        """

        async with self._lock:
            self.stop()

    def reset(self, *, total_events: int = 0) -> None:
        """
        Reset the clock to the initial warmup/start time.
        """

        initial_time = self.period.warmup_start or self.period.start

        self.state.current_time = initial_time
        self.state.current_timestamp_ms = timestamp_ms(initial_time)
        self.state.started_at = None
        self.state.finished_at = None
        self.state.events_processed = 0
        self.state.total_events = total_events

        self._started = False
        self._stopped = False

        for job in self._jobs.values():
            job.runs = 0
            job.last_run_at = None
            job.last_error = None

    async def reset_async(self, *, total_events: int = 0) -> None:
        """
        Async-safe reset wrapper.
        """

        async with self._lock:
            self.reset(total_events=total_events)

    # ---------------------------------------------------------------------
    # Time access
    # ---------------------------------------------------------------------

    def now(self) -> datetime:
        """
        Return current simulated datetime.
        """

        self._ensure_started()
        return self.state.now

    def timestamp_ms(self) -> int:
        """
        Return current simulated timestamp in milliseconds.
        """

        self._ensure_started()

        if self.state.current_timestamp_ms is None:
            raise BacktestClockNotInitializedError(
                "Backtest clock timestamp is not initialized."
            )

        return self.state.current_timestamp_ms

    def now_or_wall_clock(self) -> datetime:
        """
        Return simulated time if started, otherwise wall-clock UTC.

        Useful for logging/auditing components that may be reused outside
        actual replay.
        """

        if not self._started:
            return utcnow()
        return self.state.now

    def timestamp_ms_or_wall_clock(self) -> int:
        """
        Return simulated timestamp if started, otherwise wall-clock timestamp.
        """

        return timestamp_ms(self.now_or_wall_clock())

    @property
    def started(self) -> bool:
        return self._started

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def is_warmup(self) -> bool:
        self._ensure_started()
        return self.period.is_warmup(self.state.now)

    @property
    def progress_pct(self) -> float:
        return self.state.progress_pct

    # ---------------------------------------------------------------------
    # Time advancement
    # ---------------------------------------------------------------------

    def advance_to(
        self,
        value: datetime | int | float,
        *,
        events_processed_increment: int = 0,
        allow_equal: bool = True,
    ) -> datetime:
        """
        Move simulated time to a specific datetime or timestamp.

        By default, moving to the same timestamp is allowed because multiple
        replay events may share the same event time.
        """

        self._ensure_started()
        self._ensure_not_stopped()

        next_timestamp_ms = timestamp_ms(value)
        next_time = datetime.fromtimestamp(next_timestamp_ms / 1000.0, tz=timezone.utc)

        self._validate_advance(
            next_timestamp_ms,
            allow_equal=allow_equal,
        )

        self.state.current_time = next_time
        self.state.current_timestamp_ms = next_timestamp_ms

        if events_processed_increment:
            if events_processed_increment < 0:
                raise BacktestClockError(
                    "events_processed_increment cannot be negative.",
                    details={"events_processed_increment": events_processed_increment},
                )
            self.state.events_processed += events_processed_increment

        return next_time

    async def advance_to_async(
        self,
        value: datetime | int | float,
        *,
        events_processed_increment: int = 0,
        allow_equal: bool = True,
        run_due_jobs: bool = True,
    ) -> datetime:
        """
        Async-safe time advancement.

        If configured, due jobs can be executed after the advancement.
        """

        async with self._lock:
            result = self.advance_to(
                value,
                events_processed_increment=events_processed_increment,
                allow_equal=allow_equal,
            )

        if run_due_jobs and self.config.run_due_jobs_after_each_event:
            await self.run_due_jobs()

        return result

    def advance_by(
        self,
        delta: timedelta,
        *,
        events_processed_increment: int = 0,
    ) -> datetime:
        """
        Move simulated time forward by a timedelta.
        """

        self._ensure_started()

        if delta.total_seconds() < 0 and not self.config.allow_time_travel_backwards:
            raise BacktestTimeTravelError(
                "Cannot advance backtest clock backwards.",
                details={"delta_seconds": delta.total_seconds()},
            )

        return self.advance_to(
            self.state.now + delta,
            events_processed_increment=events_processed_increment,
            allow_equal=True,
        )

    async def advance_by_async(
        self,
        delta: timedelta,
        *,
        events_processed_increment: int = 0,
        run_due_jobs: bool = True,
    ) -> datetime:
        """
        Async-safe advance_by wrapper.
        """

        async with self._lock:
            result = self.advance_by(
                delta,
                events_processed_increment=events_processed_increment,
            )

        if run_due_jobs and self.config.run_due_jobs_after_each_event:
            await self.run_due_jobs()

        return result

    def mark_event_processed(self, count: int = 1) -> None:
        """
        Increment processed event counter without changing time.
        """

        self._ensure_started()

        if count < 0:
            raise BacktestClockError(
                "Processed event count increment cannot be negative.",
                details={"count": count},
            )

        self.state.events_processed += count

    async def mark_event_processed_async(self, count: int = 1) -> None:
        """
        Async-safe event counter update.
        """

        async with self._lock:
            self.mark_event_processed(count)

    # ---------------------------------------------------------------------
    # Period helpers
    # ---------------------------------------------------------------------

    def contains(
        self,
        value: datetime | int | float,
        *,
        include_warmup: bool = False,
    ) -> bool:
        """
        Return whether value is inside the configured backtest period.
        """

        return self.period.contains(value, include_warmup=include_warmup)

    def is_warmup_time(self, value: datetime | int | float) -> bool:
        """
        Return whether value belongs to warmup period.
        """

        return self.period.is_warmup(value)

    def is_trading_time(self, value: datetime | int | float) -> bool:
        """
        Return whether value belongs to actual trading/test period.
        """

        value_ms = timestamp_ms(value)
        return self.period.start_ms <= value_ms <= self.period.end_ms

    def seconds_since_start(self) -> float:
        """
        Seconds elapsed since actual backtest start, excluding warmup.
        """

        self._ensure_started()
        return max(0.0, (self.timestamp_ms() - self.period.start_ms) / 1000.0)

    def seconds_since_warmup_start(self) -> float:
        """
        Seconds elapsed since warmup start or start time.
        """

        self._ensure_started()
        return max(0.0, (self.timestamp_ms() - self.period.warmup_start_ms) / 1000.0)

    def seconds_until_end(self) -> float:
        """
        Seconds left until backtest period end.
        """

        self._ensure_started()
        return max(0.0, (self.period.end_ms - self.timestamp_ms()) / 1000.0)

    # ---------------------------------------------------------------------
    # Scheduler-like jobs
    # ---------------------------------------------------------------------

    def add_interval_job(
        self,
        *,
        job_id: str,
        name: str,
        callback: SchedulerJobCallback,
        interval: timedelta,
        start_at: datetime | None = None,
        run_immediately: bool = False,
        max_runs: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BacktestScheduledJob:
        """
        Register a deterministic interval job.

        Components can use this for cleanup/metrics jobs when running under
        a backtesting clock. For production runtime, core Scheduler should be
        used instead.
        """

        if not job_id:
            raise BacktestClockError("Backtest scheduled job_id is required.")

        if job_id in self._jobs:
            raise BacktestClockError(
                "Backtest scheduled job already exists.",
                details={"job_id": job_id},
            )

        current_time = self.state.now
        next_run_at = ensure_aware_utc(start_at or current_time)

        if run_immediately:
            next_run_at = current_time

        job = BacktestScheduledJob(
            job_id=job_id,
            name=name or job_id,
            callback=callback,
            interval=interval,
            next_run_at=next_run_at,
            run_immediately=run_immediately,
            max_runs=max_runs,
            metadata=metadata or {},
        )
        self._jobs[job_id] = job
        return job

    def remove_job(self, job_id: str) -> bool:
        """
        Remove scheduled job.
        """

        return self._jobs.pop(job_id, None) is not None

    def enable_job(self, job_id: str) -> None:
        """
        Enable scheduled job.
        """

        job = self._get_job(job_id)
        job.enabled = True

    def disable_job(self, job_id: str) -> None:
        """
        Disable scheduled job.
        """

        job = self._get_job(job_id)
        job.enabled = False

    def clear_jobs(self) -> None:
        """
        Remove all scheduled jobs.
        """

        self._jobs.clear()

    def due_jobs(self) -> list[BacktestScheduledJob]:
        """
        Return jobs due at current simulated time.
        """

        self._ensure_started()
        current_time = self.state.now
        return [job for job in self._jobs.values() if job.is_due(current_time)]

    async def run_due_jobs(self) -> int:
        """
        Run all due scheduled jobs.

        Returns number of executed jobs. If a job raises, the error is stored
        on the job and then re-raised as BacktestClockError.
        """

        if not self.config.run_scheduler_jobs:
            return 0

        if not self.config.run_interval_jobs_during_replay:
            return 0

        due = self.due_jobs()
        executed = 0

        for job in due:
            try:
                result = job.callback()

                if isawaitable(result):
                    await result

                job.advance_next_run()
                job.last_error = None
                executed += 1

            except Exception as exc:
                job.last_error = str(exc)
                raise BacktestClockError(
                    "Backtest scheduled job failed.",
                    details={
                        "job_id": job.job_id,
                        "name": job.name,
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    },
                ) from exc

        return executed

    # ---------------------------------------------------------------------
    # Snapshots / diagnostics
    # ---------------------------------------------------------------------

    def snapshot(self) -> BacktestTimeSnapshot:
        """
        Return current clock snapshot.
        """

        self._ensure_started()

        return BacktestTimeSnapshot(
            current_time=self.state.now,
            current_timestamp_ms=self.timestamp_ms(),
            start_time=self.period.start,
            end_time=self.period.end,
            warmup_start_time=self.period.warmup_start,
            is_warmup=self.period.is_warmup(self.state.now),
            progress_pct=self.state.progress_pct,
            events_processed=self.state.events_processed,
            total_events=self.state.total_events,
            started_at=self.state.started_at,
            finished_at=self.state.finished_at,
            metadata={
                "started": self._started,
                "stopped": self._stopped,
                "jobs": len(self._jobs),
                "due_jobs": len(self.due_jobs()) if self._started and not self._stopped else 0,
            },
        )

    def stats(self) -> dict[str, Any]:
        """
        Return diagnostic stats.
        """

        snapshot = self.snapshot()

        return {
            "current_time": snapshot.current_time.isoformat(),
            "current_timestamp_ms": snapshot.current_timestamp_ms,
            "start_time": snapshot.start_time.isoformat(),
            "end_time": snapshot.end_time.isoformat(),
            "warmup_start_time": (
                snapshot.warmup_start_time.isoformat()
                if snapshot.warmup_start_time is not None
                else None
            ),
            "is_warmup": snapshot.is_warmup,
            "progress_pct": snapshot.progress_pct,
            "events_processed": snapshot.events_processed,
            "total_events": snapshot.total_events,
            "started": self._started,
            "stopped": self._stopped,
            "jobs": {
                job_id: {
                    "name": job.name,
                    "enabled": job.enabled,
                    "runs": job.runs,
                    "max_runs": job.max_runs,
                    "next_run_at": job.next_run_at.isoformat(),
                    "last_run_at": job.last_run_at.isoformat() if job.last_run_at else None,
                    "last_error": job.last_error,
                }
                for job_id, job in self._jobs.items()
            },
        }

    # ---------------------------------------------------------------------
    # Internal guards
    # ---------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if not self._started:
            raise BacktestClockNotInitializedError(
                "Backtest clock has not been started."
            )

    def _ensure_not_stopped(self) -> None:
        if self._stopped:
            raise BacktestClockError("Backtest clock is already stopped.")

    def _validate_advance(
        self,
        next_timestamp_ms: int,
        *,
        allow_equal: bool,
    ) -> None:
        current_timestamp_ms = self.state.current_timestamp_ms

        if current_timestamp_ms is None:
            raise BacktestClockNotInitializedError(
                "Backtest clock current timestamp is not initialized."
            )

        if next_timestamp_ms < current_timestamp_ms and not self.config.allow_time_travel_backwards:
            raise BacktestTimeTravelError(
                "Backtest clock cannot move backwards.",
                details={
                    "current_timestamp_ms": current_timestamp_ms,
                    "next_timestamp_ms": next_timestamp_ms,
                },
            )

        if next_timestamp_ms == current_timestamp_ms and not allow_equal:
            raise BacktestTimeTravelError(
                "Backtest clock cannot stay on the same timestamp when allow_equal=False.",
                details={
                    "current_timestamp_ms": current_timestamp_ms,
                    "next_timestamp_ms": next_timestamp_ms,
                },
            )

        lower_bound_ms = self.period.warmup_start_ms
        upper_bound_ms = self.period.end_ms

        if self.config.fail_on_time_out_of_range:
            if next_timestamp_ms < lower_bound_ms:
                raise BacktestTimeRangeError(
                    "Backtest clock cannot move before warmup/start time.",
                    details={
                        "next_timestamp_ms": next_timestamp_ms,
                        "warmup_start_ms": lower_bound_ms,
                    },
                )

            if next_timestamp_ms > upper_bound_ms:
                raise BacktestTimeRangeError(
                    "Backtest clock cannot move after end time.",
                    details={
                        "next_timestamp_ms": next_timestamp_ms,
                        "end_ms": upper_bound_ms,
                    },
                )

    def _get_job(self, job_id: str) -> BacktestScheduledJob:
        job = self._jobs.get(job_id)

        if job is None:
            raise BacktestClockError(
                "Backtest scheduled job not found.",
                details={"job_id": job_id},
            )

        return job


class BacktestTimeProvider:
    """
    Small adapter that can be injected into components needing current time.

    This avoids passing the full BacktestClock where only now()/timestamp_ms()
    is needed.
    """

    def __init__(self, clock: BacktestClock) -> None:
        self.clock = clock

    def now(self) -> datetime:
        return self.clock.now()

    def timestamp_ms(self) -> int:
        return self.clock.timestamp_ms()

    def is_warmup(self) -> bool:
        return self.clock.is_warmup

    def progress_pct(self) -> float:
        return self.clock.progress_pct


def build_backtest_clock(
    *,
    start: datetime,
    end: datetime,
    warmup_start: datetime | None = None,
    config: BacktestTimeConfig | None = None,
) -> BacktestClock:
    """
    Convenience factory for BacktestClock.
    """

    period = BacktestPeriod(
        start=ensure_aware_utc(start),
        end=ensure_aware_utc(end),
        warmup_start=ensure_aware_utc(warmup_start) if warmup_start else None,
    )
    return BacktestClock(period=period, config=config)


__all__ = [
    "SchedulerJobCallback",
    "BacktestScheduledJob",
    "BacktestTimeSnapshot",
    "BacktestClock",
    "BacktestTimeProvider",
    "build_backtest_clock",
]