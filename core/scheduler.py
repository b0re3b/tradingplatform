from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Union

from core.logger import get_logger


JobCallable = Callable[..., Union[None, Awaitable[None]]]


class JobStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


class ScheduleMode(str, Enum):
    """
    How periodic interval jobs calculate their next run.

    FIXED_RATE keeps the cadence anchored to the planned schedule. If a job
    takes longer than expected, the scheduler skips missed slots and schedules
    the next future slot instead of drifting forever.

    FIXED_DELAY preserves the old behaviour: wait `interval` seconds after a
    job finishes before running it again. Use this only for maintenance jobs
    where exact cadence does not matter.
    """

    FIXED_RATE = "fixed_rate"
    FIXED_DELAY = "fixed_delay"


@dataclass(slots=True)
class ScheduledJob:
    job_id: str
    name: str
    func: JobCallable
    interval: Optional[float] = None
    delay: float = 0.0
    args: tuple = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    run_immediately: bool = False
    max_retries: int = 0
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    allow_overlap: bool = False
    schedule_mode: ScheduleMode = ScheduleMode.FIXED_RATE
    next_run_at: Optional[float] = None
    last_run_at: Optional[float] = None
    last_scheduled_run_at: Optional[float] = None
    last_finish_at: Optional[float] = None
    last_error: Optional[str] = None
    status: JobStatus = JobStatus.IDLE
    total_runs: int = 0
    total_failures: int = 0
    total_success: int = 0
    total_skipped: int = 0
    total_missed_slots: int = 0
    one_shot: bool = False
    task: Optional[asyncio.Task] = None

    @property
    def is_periodic(self) -> bool:
        return self.interval is not None and not self.one_shot


class Scheduler:
    """
    Async scheduler для трейдинг-системи.

    Призначення:
    - periodic jobs
    - delayed one-shot jobs
    - retries
    - timeout
    - overlap control
    - graceful shutdown
    - інтеграція з EventBus

    Важливо для live:
    - periodic jobs за замовчуванням працюють у fixed-rate режимі;
    - fixed-rate не накопичує drift після довгих виконань;
    - якщо overlap заборонений і попередній запуск ще працює, scheduler
      пропускає прострочений слот і переходить до наступного майбутнього слоту.
    """

    def __init__(
        self,
        *,
        event_bus: Optional[Any] = None,
        tick_interval: float = 0.2,
        default_schedule_mode: ScheduleMode | str = ScheduleMode.FIXED_RATE,
        service_name: str = "scheduler",
    ) -> None:
        self._event_bus = event_bus
        self._tick_interval = tick_interval
        self._default_schedule_mode = self._coerce_schedule_mode(default_schedule_mode)
        self._service_name = service_name

        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="scheduler",
        )

        self._jobs: dict[str, ScheduledJob] = {}
        self._scheduler_task: Optional[asyncio.Task] = None

        self._running = False
        self._stopping = False

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            self._logger.warning("Scheduler already started")
            return

        self._running = True
        self._stopping = False
        self._scheduler_task = asyncio.create_task(
            self._run_loop(),
            name="scheduler-main-loop",
        )

        self._logger.info(
            "Scheduler started | tick_interval=%s jobs=%s default_schedule_mode=%s",
            self._tick_interval,
            len(self._jobs),
            self._default_schedule_mode.value,
        )

    async def stop(self, *, wait_running_jobs: bool = True, timeout: float = 10.0) -> None:
        if not self._running:
            self._logger.warning("Scheduler already stopped")
            return

        self._stopping = True
        self._running = False

        self._logger.info(
            "Stopping Scheduler | wait_running_jobs=%s timeout=%s",
            wait_running_jobs,
            timeout,
        )

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

        running_tasks = [
            job.task for job in self._jobs.values()
            if job.task is not None and not job.task.done()
        ]

        if running_tasks and wait_running_jobs:
            _, pending = await asyncio.wait(running_tasks, timeout=timeout)
            if pending:
                self._logger.warning(
                    "Scheduler stop timeout reached, cancelling running jobs | count=%s",
                    len(pending),
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        elif running_tasks and not wait_running_jobs:
            self._logger.warning(
                "Cancelling running jobs immediately | count=%s",
                len(running_tasks),
            )
            for task in running_tasks:
                task.cancel()
            await asyncio.gather(*running_tasks, return_exceptions=True)

        self._logger.info("Scheduler stopped")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def add_interval_job(
        self,
        name: str,
        func: JobCallable,
        *,
        interval: float,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
        run_immediately: bool = False,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
        allow_overlap: bool = False,
        schedule_mode: ScheduleMode | str | None = None,
        enabled: bool = True,
    ) -> str:
        if interval <= 0:
            raise ValueError("interval must be > 0")

        job_id = str(uuid.uuid4())
        now = time.time()
        mode = self._coerce_schedule_mode(schedule_mode or self._default_schedule_mode)

        job = ScheduledJob(
            job_id=job_id,
            name=name,
            func=func,
            interval=interval,
            args=args,
            kwargs=kwargs or {},
            enabled=enabled,
            run_immediately=run_immediately,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            allow_overlap=allow_overlap,
            schedule_mode=mode,
            next_run_at=now if run_immediately else now + interval,
            one_shot=False,
        )

        self._jobs[job_id] = job

        self._job_logger(job).info(
            "Interval job added | interval=%s run_immediately=%s enabled=%s allow_overlap=%s schedule_mode=%s",
            interval,
            run_immediately,
            enabled,
            allow_overlap,
            mode.value,
        )
        return job_id

    def add_delayed_job(
        self,
        name: str,
        func: JobCallable,
        *,
        delay: float,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
        enabled: bool = True,
    ) -> str:
        if delay < 0:
            raise ValueError("delay must be >= 0")

        job_id = str(uuid.uuid4())
        now = time.time()

        job = ScheduledJob(
            job_id=job_id,
            name=name,
            func=func,
            delay=delay,
            args=args,
            kwargs=kwargs or {},
            enabled=enabled,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            allow_overlap=False,
            schedule_mode=ScheduleMode.FIXED_DELAY,
            next_run_at=now + delay,
            one_shot=True,
        )

        self._jobs[job_id] = job

        self._job_logger(job).info(
            "Delayed job added | delay=%s enabled=%s",
            delay,
            enabled,
        )
        return job_id

    async def run_job_now(self, job_id: str) -> None:
        job = self._get_job_or_raise(job_id)
        log = self._job_logger(job)

        if not job.enabled:
            log.warning("Manual run ignored: job is disabled")
            return

        if job.task is not None and not job.task.done() and not job.allow_overlap:
            log.warning("Manual run skipped: job is already running and overlap is disabled")
            return

        log.info("Manual run requested")
        await self._launch_job(job, scheduled_run_at=time.time(), reserve_next=False)

    def remove_job(self, job_id: str) -> None:
        job = self._get_job_or_raise(job_id)
        log = self._job_logger(job)

        if job.task is not None and not job.task.done():
            log.warning("Removing running job")

        del self._jobs[job_id]
        log.info("Job removed")

    def enable_job(self, job_id: str) -> None:
        job = self._get_job_or_raise(job_id)
        job.enabled = True

        if job.next_run_at is None:
            now = time.time()
            base_delay = job.interval if job.interval is not None else job.delay
            job.next_run_at = now if job.run_immediately else now + base_delay

        self._job_logger(job).info("Job enabled")

    def disable_job(self, job_id: str) -> None:
        job = self._get_job_or_raise(job_id)
        job.enabled = False
        self._job_logger(job).info("Job disabled")

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        return self._jobs.get(job_id)

    def get_job_by_name(self, name: str) -> Optional[ScheduledJob]:
        for job in self._jobs.values():
            if job.name == name:
                return job
        return None

    def list_jobs(self) -> dict[str, dict[str, Any]]:
        return {
            job_id: {
                "job_id": job.job_id,
                "name": job.name,
                "enabled": job.enabled,
                "status": job.status.value,
                "interval": job.interval,
                "delay": job.delay,
                "schedule_mode": job.schedule_mode.value,
                "next_run_at": job.next_run_at,
                "last_run_at": job.last_run_at,
                "last_scheduled_run_at": job.last_scheduled_run_at,
                "last_finish_at": job.last_finish_at,
                "last_error": job.last_error,
                "total_runs": job.total_runs,
                "total_success": job.total_success,
                "total_failures": job.total_failures,
                "total_skipped": job.total_skipped,
                "total_missed_slots": job.total_missed_slots,
                "one_shot": job.one_shot,
                "allow_overlap": job.allow_overlap,
            }
            for job_id, job in self._jobs.items()
        }

    def stats(self) -> dict[str, Any]:
        running_jobs = sum(
            1 for job in self._jobs.values()
            if job.task is not None and not job.task.done()
        )
        enabled_jobs = sum(1 for job in self._jobs.values() if job.enabled)

        return {
            "running": self._running,
            "stopping": self._stopping,
            "jobs_total": len(self._jobs),
            "jobs_enabled": enabled_jobs,
            "jobs_running": running_jobs,
            "tick_interval": self._tick_interval,
            "default_schedule_mode": self._default_schedule_mode.value,
        }

    # ---------------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------------

    async def _run_loop(self) -> None:
        self._logger.info("Scheduler loop started")

        try:
            while self._running and not self._stopping:
                now = time.time()

                for job in list(self._jobs.values()):
                    if not job.enabled or job.next_run_at is None:
                        continue

                    if now < job.next_run_at:
                        continue

                    scheduled_run_at = job.next_run_at

                    if job.task is not None and not job.task.done() and not job.allow_overlap:
                        job.total_skipped += 1
                        self._job_logger(job).warning(
                            "Job skipped due to overlap protection | scheduled_run_at=%s",
                            scheduled_run_at,
                        )
                        await self._emit_scheduler_event(
                            "system.scheduler.job_skipped",
                            {
                                "job_id": job.job_id,
                                "job_name": job.name,
                                "reason": "overlap_blocked",
                                "scheduled_run_at": scheduled_run_at,
                                "schedule_mode": job.schedule_mode.value,
                            },
                        )

                        if job.is_periodic:
                            self._schedule_next_run(job, completed_at=now, previous_scheduled_run_at=scheduled_run_at)
                        continue

                    await self._launch_job(job, scheduled_run_at=scheduled_run_at, reserve_next=True)

                await asyncio.sleep(self._tick_interval)

        except asyncio.CancelledError:
            self._logger.info("Scheduler loop cancelled")
            raise
        except Exception:
            self._logger.exception("Scheduler loop crashed")
            raise
        finally:
            self._logger.info("Scheduler loop finished")

    async def _launch_job(
        self,
        job: ScheduledJob,
        *,
        scheduled_run_at: Optional[float],
        reserve_next: bool,
    ) -> None:
        if self._stopping:
            return

        if reserve_next and job.is_periodic and job.schedule_mode == ScheduleMode.FIXED_RATE:
            self._schedule_next_run(
                job,
                completed_at=time.time(),
                previous_scheduled_run_at=scheduled_run_at,
            )

        if job.one_shot:
            job.enabled = False
            job.next_run_at = None

        job.task = asyncio.create_task(
            self._execute_job(job, scheduled_run_at=scheduled_run_at),
            name=f"scheduler-job-{job.name}-{job.job_id[:8]}",
        )

    async def _execute_job(self, job: ScheduledJob, *, scheduled_run_at: Optional[float]) -> None:
        log = self._job_logger(job)

        job.status = JobStatus.RUNNING
        job.last_run_at = time.time()
        job.last_scheduled_run_at = scheduled_run_at
        job.total_runs += 1

        lag = max(0.0, job.last_run_at - scheduled_run_at) if scheduled_run_at is not None else 0.0

        log.info(
            "Job started | run=%s one_shot=%s periodic=%s schedule_mode=%s lag=%.3fs",
            job.total_runs,
            job.one_shot,
            job.is_periodic,
            job.schedule_mode.value,
            lag,
        )

        await self._emit_scheduler_event(
            "system.scheduler.job_started",
            {
                "job_id": job.job_id,
                "job_name": job.name,
                "run_number": job.total_runs,
                "scheduled_run_at": scheduled_run_at,
                "actual_run_at": job.last_run_at,
                "lag_seconds": lag,
                "schedule_mode": job.schedule_mode.value,
            },
        )

        attempt = 0

        while True:
            try:
                await self._run_job_callable(job)

                job.status = JobStatus.IDLE
                job.last_finish_at = time.time()
                job.last_error = None
                job.total_success += 1

                log.info(
                    "Job completed | success=%s failures=%s duration=%.3fs",
                    job.total_success,
                    job.total_failures,
                    job.last_finish_at - job.last_run_at,
                )

                await self._emit_scheduler_event(
                    "system.scheduler.job_completed",
                    {
                        "job_id": job.job_id,
                        "job_name": job.name,
                        "run_number": job.total_runs,
                        "duration_seconds": job.last_finish_at - job.last_run_at,
                    },
                )

                if job.one_shot:
                    job.status = JobStatus.STOPPED
                    self._job_logger(job).info("One-shot job finished and disabled")
                elif job.schedule_mode == ScheduleMode.FIXED_DELAY:
                    self._schedule_next_run(job, completed_at=job.last_finish_at, previous_scheduled_run_at=scheduled_run_at)
                return

            except asyncio.CancelledError:
                job.status = JobStatus.STOPPED
                job.last_finish_at = time.time()
                job.last_error = "cancelled"

                log.warning("Job cancelled")
                raise

            except Exception as exc:
                attempt += 1
                job.total_failures += 1
                job.status = JobStatus.FAILED
                job.last_error = str(exc)
                job.last_finish_at = time.time()

                log.exception(
                    "Job failed | attempt=%s max_attempts=%s",
                    attempt,
                    job.max_retries + 1,
                )

                await self._emit_scheduler_event(
                    "system.scheduler.job_failed",
                    {
                        "job_id": job.job_id,
                        "job_name": job.name,
                        "attempt": attempt,
                        "max_retries": job.max_retries,
                        "error": str(exc),
                    },
                )

                if attempt > job.max_retries:
                    if job.one_shot:
                        job.enabled = False
                        job.next_run_at = None
                        job.status = JobStatus.STOPPED
                    elif job.schedule_mode == ScheduleMode.FIXED_DELAY:
                        self._schedule_next_run(job, completed_at=job.last_finish_at, previous_scheduled_run_at=scheduled_run_at)
                    else:
                        job.status = JobStatus.IDLE
                    return

                if job.retry_delay > 0:
                    await asyncio.sleep(job.retry_delay)

    async def _run_job_callable(self, job: ScheduledJob) -> None:
        if inspect.iscoroutinefunction(job.func):
            coro = job.func(*job.args, **job.kwargs)
            if job.timeout is not None:
                await asyncio.wait_for(coro, timeout=job.timeout)
            else:
                await coro
            return

        thread_call = asyncio.to_thread(job.func, *job.args, **job.kwargs)
        if job.timeout is not None:
            await asyncio.wait_for(thread_call, timeout=job.timeout)
        else:
            await thread_call

    def _schedule_next_run(
        self,
        job: ScheduledJob,
        *,
        completed_at: float,
        previous_scheduled_run_at: Optional[float],
    ) -> None:
        if job.one_shot:
            job.enabled = False
            job.next_run_at = None
            job.status = JobStatus.STOPPED
            self._job_logger(job).info("One-shot job finished and disabled")
            return

        if job.interval is None:
            job.next_run_at = None
            job.status = JobStatus.STOPPED
            self._job_logger(job).warning("Job has no interval, disabling scheduling")
            return

        if job.schedule_mode == ScheduleMode.FIXED_DELAY:
            job.next_run_at = completed_at + job.interval
        else:
            anchor = previous_scheduled_run_at if previous_scheduled_run_at is not None else completed_at
            next_run_at = anchor + job.interval
            missed = 0
            # Skip missed historical slots. This prevents a long job from causing
            # a burst of immediate catch-up runs and keeps live cadence stable.
            while next_run_at <= completed_at:
                next_run_at += job.interval
                missed += 1

            if missed:
                job.total_missed_slots += missed
                self._job_logger(job).warning(
                    "Fixed-rate job missed slots | missed=%s next_run_at=%s",
                    missed,
                    next_run_at,
                )

            job.next_run_at = next_run_at

        job.status = JobStatus.IDLE

        self._job_logger(job).debug(
            "Next run scheduled | next_run_at=%s interval=%s schedule_mode=%s",
            job.next_run_at,
            job.interval,
            job.schedule_mode.value,
        )

    async def _emit_scheduler_event(self, topic: str, payload: dict[str, Any]) -> None:
        if self._event_bus is None:
            return

        try:
            await self._event_bus.emit(
                topic,
                payload,
                source="scheduler",
            )
        except Exception:
            self._logger.exception(
                "Failed to emit scheduler event | topic=%s",
                topic,
            )

    def _get_job_or_raise(self, job_id: str) -> ScheduledJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")
        return job

    @staticmethod
    def _coerce_schedule_mode(value: ScheduleMode | str) -> ScheduleMode:
        if isinstance(value, ScheduleMode):
            return value
        try:
            return ScheduleMode(str(value).strip().lower())
        except Exception as exc:
            raise ValueError(f"Invalid schedule mode: {value!r}") from exc

    def _job_logger(self, job: ScheduledJob):
        return get_logger(
            __name__,
            service=self._service_name,
            event_type="scheduler_job",
            job_name=job.name,
            job_id=job.job_id,
        )
