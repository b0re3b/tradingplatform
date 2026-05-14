from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler


EventHandler = Callable[[Event], None | Awaitable[None]]
JobCallable = Callable[..., None | Awaitable[None]]


class BaseWhaleComponent(ABC):
    """
    Базовий клас для всіх analytics.whales runtime-компонентів.

    Відповідає тільки за core-інтеграцію:
    - logger через core.logger.get_logger;
    - EventBus dependency injection;
    - Scheduler dependency injection;
    - register() / EventBus.subscribe();
    - централізоване збереження Subscription;
    - EventBus.emit();
    - Scheduler.add_interval_job();
    - lifecycle state.

    Не створює власних background asyncio loops.
    """

    def __init__(
        self,
        *,
        component_name: str,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> None:
        self.component_name = component_name
        self.event_bus = event_bus
        self.scheduler = scheduler

        self.logger = get_logger(
            __name__,
            service_name="analytics.whales",
            component=component_name,
            event_type="whale_component",
        )

        self._started = False
        self._registered = False
        self._subscriptions: list[Subscription] = []
        self._scheduler_job_ids: list[str] = []

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_registered(self) -> bool:
        return self._registered

    @property
    def subscriptions(self) -> tuple[Subscription, ...]:
        return tuple(self._subscriptions)

    @property
    def scheduler_job_ids(self) -> tuple[str, ...]:
        return tuple(self._scheduler_job_ids)

    # =========================================================================
    # Lifecycle contract
    # =========================================================================

    @abstractmethod
    async def register(self) -> None:
        """
        Зареєструвати EventBus subscriptions.

        Дочірній клас має викликати self._subscribe(...).
        Метод має бути idempotent.
        """
        raise NotImplementedError

    @abstractmethod
    async def start(self) -> None:
        """
        Запуск runtime-компонента.

        Типовий порядок у дочірньому класі:
        1. перевірити config.enabled;
        2. await self.register();
        3. self._add_interval_job(... cleanup ...);
        4. self._started = True.
        """
        raise NotImplementedError

    async def stop(self) -> None:
        """
        Базова зупинка компонента.

        Scheduler jobs у поточному core Scheduler не мають bulk remove API за name/prefix,
        тому тут ми прибираємо тільки EventBus subscriptions.
        Якщо дочірній клас хоче remove_job(job_id), він може зробити це перед super().stop().
        """
        if not self._started and not self._registered:
            return

        self._unsubscribe_all()
        self._registered = False
        self._started = False

        self.logger.info(
            "Whale component stopped",
            extra={"component": self.component_name},
        )

    # =========================================================================
    # Health / diagnostics
    # =========================================================================

    def get_healthcheck(self) -> dict[str, Any]:
        return {
            "component": self.component_name,
            "started": self._started,
            "registered": self._registered,
            "subscriptions": len(self._subscriptions),
            "scheduler_jobs": len(self._scheduler_job_ids),
            "event_bus_available": self.event_bus is not None,
            "scheduler_available": self.scheduler is not None,
        }

    # =========================================================================
    # EventBus helpers
    # =========================================================================

    def _subscribe(
        self,
        topic: str,
        handler: EventHandler,
        *,
        name: str | None = None,
    ) -> Subscription:
        """
        Зареєструвати handler у core EventBus.

        core.EventBus.subscribe() є sync-методом і повертає Subscription.
        """
        if not topic or not topic.strip():
            raise ValueError("EventBus topic must be a non-empty string")

        handler_name = name or f"{self.component_name}.{getattr(handler, '__name__', 'handler')}"

        subscription = self.event_bus.subscribe(
            topic,
            handler,
            name=handler_name,
        )
        self._subscriptions.append(subscription)

        self.logger.info(
            "Subscribed to EventBus topic",
            extra={
                "component": self.component_name,
                "topic": topic,
                "handler": handler_name,
            },
        )
        return subscription

    def _unsubscribe_all(self) -> None:
        """
        Відписати всі subscriptions, які створив цей компонент.
        """
        while self._subscriptions:
            subscription = self._subscriptions.pop()
            try:
                self.event_bus.unsubscribe(subscription)
                self.logger.info(
                    "Unsubscribed from EventBus topic",
                    extra={
                        "component": self.component_name,
                        "topic": subscription.pattern,
                        "handler": subscription.name,
                    },
                )
            except Exception:
                self.logger.exception(
                    "Failed to unsubscribe from EventBus topic",
                    extra={
                        "component": self.component_name,
                        "topic": subscription.pattern,
                        "handler": subscription.name,
                    },
                )

    async def _emit(
        self,
        topic: str,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str | None = None,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        """
        Опублікувати payload через core EventBus.emit().
        """
        if not topic or not topic.strip():
            raise ValueError("EventBus topic must be a non-empty string")

        try:
            accepted = await self.event_bus.emit(
                topic,
                dict(payload),
                priority=priority,
                source=source or self.component_name,
                correlation_id=correlation_id,
                headers=headers,
            )

            if not accepted:
                self.logger.warning(
                    "EventBus rejected whale event",
                    extra={
                        "component": self.component_name,
                        "topic": topic,
                        "source": source or self.component_name,
                    },
                )

            return accepted

        except Exception:
            self.logger.exception(
                "Failed to emit whale event",
                extra={
                    "component": self.component_name,
                    "topic": topic,
                    "source": source or self.component_name,
                },
            )
            raise

    @staticmethod
    def _payload_from_event(event: Event) -> dict[str, Any]:
        """
        Дістати dict payload із core Event.

        Handler-и мають приймати core.event_bus.Event, а бізнес-методи можуть
        працювати з dict payload.
        """
        if isinstance(event.payload, dict):
            return event.payload

        if isinstance(event.payload, Mapping):
            return dict(event.payload)

        raise TypeError(
            f"Expected Event.payload to be mapping, got {type(event.payload).__name__}"
        )

    # =========================================================================
    # Scheduler helpers
    # =========================================================================

    def _add_interval_job(
        self,
        *,
        name: str,
        func: JobCallable,
        interval: float,
        run_immediately: bool = False,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        timeout: float | None = None,
        allow_overlap: bool = False,
        enabled: bool = True,
    ) -> str:
        """
        Зареєструвати periodic job через core Scheduler.add_interval_job().
        """
        if not name or not name.strip():
            raise ValueError("Scheduler job name must be a non-empty string")
        if interval <= 0:
            raise ValueError("Scheduler interval must be > 0")

        job_id = self.scheduler.add_interval_job(
            name=name,
            func=func,
            interval=interval,
            run_immediately=run_immediately,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            allow_overlap=allow_overlap,
            enabled=enabled,
        )
        self._scheduler_job_ids.append(job_id)

        self.logger.info(
            "Registered scheduler interval job",
            extra={
                "component": self.component_name,
                "job_name": name,
                "job_id": job_id,
                "interval": interval,
                "run_immediately": run_immediately,
                "allow_overlap": allow_overlap,
            },
        )
        return job_id

    def _remove_scheduler_jobs(self) -> None:
        """
        Видалити jobs, створені цим компонентом.

        Викликати з дочірнього stop(), якщо потрібно повністю прибрати job-и.
        """
        while self._scheduler_job_ids:
            job_id = self._scheduler_job_ids.pop()
            try:
                self.scheduler.remove_job(job_id)
                self.logger.info(
                    "Removed scheduler job",
                    extra={
                        "component": self.component_name,
                        "job_id": job_id,
                    },
                )
            except KeyError:
                self.logger.warning(
                    "Scheduler job already removed",
                    extra={
                        "component": self.component_name,
                        "job_id": job_id,
                    },
                )
            except Exception:
                self.logger.exception(
                    "Failed to remove scheduler job",
                    extra={
                        "component": self.component_name,
                        "job_id": job_id,
                    },
                )

    # =========================================================================
    # Shared helpers
    # =========================================================================

    @staticmethod
    def _passes_cooldown(last_ts_monotonic: float, cooldown_sec: float) -> bool:
        if cooldown_sec <= 0:
            return True
        return (time.monotonic() - last_ts_monotonic) >= cooldown_sec

    @staticmethod
    def _clamp_0_1(value: float) -> float:
        return max(0.0, min(1.0, value))


class BaseWhaleAnalyzerComponent(BaseWhaleComponent):
    """
    Семантичний базовий клас для фасадного analyzer-рівня.

    Фасад може не мати власних EventBus subscriptions, але все одно отримує
    event_bus/scheduler через dependency injection і керує дочірніми компонентами.
    """

    def __init__(
        self,
        *,
        component_name: str = "analyzer",
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> None:
        super().__init__(
            component_name=component_name,
            event_bus=event_bus,
            scheduler=scheduler,
        )

    async def register(self) -> None:
        """
        Analyzer facade за замовчуванням не підписується на EventBus напряму.
        Підписки виконують дочірні runtime-компоненти.
        """
        if self._registered:
            return

        self._registered = True
        self.logger.info(
            "Whale analyzer facade registered",
            extra={"component": self.component_name},
        )


__all__ = [
    "BaseWhaleComponent",
    "BaseWhaleAnalyzerComponent",
    "EventHandler",
    "JobCallable",
]