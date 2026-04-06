from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from core.logger import get_logger


class BaseWhaleComponent(ABC):
    """
    Базовий клас для всіх whale-компонентів.

    Дає:
    - уніфіковану ініціалізацію logger
    - event_bus / scheduler dependency slots
    - started lifecycle flag
    - базові helper-утиліти
    """

    def __init__(
        self,
        component_name: str,
        event_bus: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        self.component_name = component_name
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.logger = get_logger(
            __name__,
            service_name=f"analytics.whales.{component_name}",
        )
        self._started = False

    @property
    def is_started(self) -> bool:
        return self._started

    @abstractmethod
    async def start(self) -> None:
        """
        Запуск компонента:
        - subscribe to EventBus
        - register scheduler jobs
        - init background tasks
        """
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """
        Коректна зупинка компонента.
        """
        raise NotImplementedError

    def get_healthcheck(self) -> Dict[str, Any]:
        return {
            "component": self.component_name,
            "started": self._started,
            "has_event_bus": self.event_bus is not None,
            "has_scheduler": self.scheduler is not None,
        }

    async def _safe_emit(
        self,
        event_name: str,
        payload: Dict[str, Any],
        *,
        source: Optional[str] = None,
        priority: Optional[Any] = None,
    ) -> None:
        """
        Безпечний emit події в EventBus.

        Підтримує bus.emit(...) у різних варіантах сигнатури.
        """
        if self.event_bus is None:
            return

        try:
            kwargs: Dict[str, Any] = {}
            if source is not None:
                kwargs["source"] = source
            if priority is not None:
                kwargs["priority"] = priority

            await self.event_bus.emit(event_name, payload, **kwargs)
        except TypeError:
            try:
                await self.event_bus.emit(event_name, payload)
            except Exception:
                self.logger.exception(
                    "Failed to emit event to EventBus",
                    extra={
                        "event_name": event_name,
                        "source": source,
                    },
                )
        except Exception:
            self.logger.exception(
                "Failed to emit event to EventBus",
                extra={
                    "event_name": event_name,
                    "source": source,
                },
            )

    async def _safe_subscribe(
        self,
        event_name: str,
        handler: Any,
    ) -> None:
        if self.event_bus is None:
            return

        try:
            await self.event_bus.subscribe(event_name, handler)
            self.logger.info(
                "Subscribed to EventBus event",
                extra={
                    "event_name": event_name,
                    "handler": getattr(handler, "__name__", str(handler)),
                },
            )
        except Exception:
            self.logger.exception(
                "Failed to subscribe to EventBus event",
                extra={
                    "event_name": event_name,
                    "handler": getattr(handler, "__name__", str(handler)),
                },
            )
            raise

    async def _safe_unsubscribe(
        self,
        event_name: str,
        handler: Any,
    ) -> None:
        if self.event_bus is None:
            return

        try:
            await self.event_bus.unsubscribe(event_name, handler)
            self.logger.info(
                "Unsubscribed from EventBus event",
                extra={
                    "event_name": event_name,
                    "handler": getattr(handler, "__name__", str(handler)),
                },
            )
        except Exception:
            self.logger.exception(
                "Failed to unsubscribe from EventBus event",
                extra={
                    "event_name": event_name,
                    "handler": getattr(handler, "__name__", str(handler)),
                },
            )

    async def _register_interval_job(
        self,
        *,
        name: str,
        interval_seconds: int,
        coro: Any,
        replace_existing: bool = True,
    ) -> None:
        if self.scheduler is None:
            return

        try:
            await self.scheduler.add_interval_job(
                name=name,
                interval_seconds=interval_seconds,
                coro=coro,
                replace_existing=replace_existing,
            )
            self.logger.info(
                "Registered scheduler interval job",
                extra={
                    "job_name": name,
                    "interval_seconds": interval_seconds,
                },
            )
        except Exception:
            self.logger.exception(
                "Failed to register scheduler interval job",
                extra={
                    "job_name": name,
                    "interval_seconds": interval_seconds,
                },
            )
            raise

    async def _cancel_task(self, task: Optional[asyncio.Task[Any]]) -> None:
        if task is None:
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception(
                "Unhandled exception while cancelling background task",
                extra={"component": self.component_name},
            )