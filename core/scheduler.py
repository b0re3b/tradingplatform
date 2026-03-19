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
    next_run_at: Optional[float] = None
    last_run_at: Optional[float] = None
    last_finish_at: Optional[float] = None
    last_error: Optional[str] = None
    status: JobStatus = JobStatus.IDLE
    total_runs: int = 0
    total_failures: int = 0
    total_success: int = 0
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
    """

    def __init__(
        self,
        *,
        event_bus: Optional[Any] = None,
        tick_interval: float = 0.2,
        service_name: str = "scheduler",
    ) -> None:
        self._event_bus = event_bus
        self._tick_interval = tick_interval
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
            "Scheduler started | tick_interval=%s jobs=%s",
            self._tick_interval,
            len(self._jobs),
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
            done, pending = await asyncio.wait(running_tasks, timeout=timeout)
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
        enabled: bool = True,
    ) -> str:
        if interval <= 0:
            raise ValueError("interval must be > 0")

        job_id = str(uuid.uuid4())
        now = time.time()

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
            next_run_at=now if run_immediately else now + interval,
            one_shot=False,
        )

        self._jobs[job_id] = job

        self._job_logger(job).info(
            "Interval job added | interval=%s run_immediately=%s enabled=%s allow_overlap=%s",
            interval,
            run_immediately,
            enabled,
            allow_overlap,
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
        await self._launch_job(job)

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
                "next_run_at": job.next_run_at,
                "last_run_at": job.last_run_at,
                "last_finish_at": job.last_finish_at,
                "last_error": job.last_error,
                "total_runs": job.total_runs,
                "total_success": job.total_success,
                "total_failures": job.total_failures,
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

                    if job.task is not None and not job.task.done() and not job.allow_overlap:
                        self._job_logger(job).warning(
                            "Job skipped due to overlap protection"
                        )
                        await self._emit_scheduler_event(
                            "system.scheduler.job_skipped",
                            {
                                "job_id": job.job_id,
                                "job_name": job.name,
                                "reason": "overlap_blocked",
                            },
                        )

                        if job.is_periodic:
                            job.next_run_at = now + (job.interval or 0.0)
                        continue

                    await self._launch_job(job)

                await asyncio.sleep(self._tick_interval)

        except asyncio.CancelledError:
            self._logger.info("Scheduler loop cancelled")
            raise
        except Exception:
            self._logger.exception("Scheduler loop crashed")
            raise
        finally:
            self._logger.info("Scheduler loop finished")

    async def _launch_job(self, job: ScheduledJob) -> None:
        if self._stopping:
            return

        job.task = asyncio.create_task(
            self._execute_job(job),
            name=f"scheduler-job-{job.name}-{job.job_id[:8]}",
        )

    async def _execute_job(self, job: ScheduledJob) -> None:
        log = self._job_logger(job)

        job.status = JobStatus.RUNNING
        job.last_run_at = time.time()
        job.total_runs += 1

        log.info(
            "Job started | run=%s one_shot=%s periodic=%s",
            job.total_runs,
            job.one_shot,
            job.is_periodic,
        )

        await self._emit_scheduler_event(
            "system.scheduler.job_started",
            {
                "job_id": job.job_id,
                "job_name": job.name,
                "run_number": job.total_runs,
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
                    "Job completed | success=%s failures=%s",
                    job.total_success,
                    job.total_failures,
                )

                await self._emit_scheduler_event(
                    "system.scheduler.job_completed",
                    {
                        "job_id": job.job_id,
                        "job_name": job.name,
                        "run_number": job.total_runs,
                    },
                )

                self._schedule_next_run(job)
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
                    self._schedule_next_run(job)
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

    def _schedule_next_run(self, job: ScheduledJob) -> None:
        now = time.time()

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

        job.next_run_at = now + job.interval
        job.status = JobStatus.IDLE

        self._job_logger(job).debug(
            "Next run scheduled | next_run_at=%s interval=%s",
            job.next_run_at,
            job.interval,
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

    def _job_logger(self, job: ScheduledJob):
        return get_logger(
            __name__,
            service=self._service_name,
            event_type="scheduled_job",
            job_id=job.job_id,
            job_name=job.name,
        )