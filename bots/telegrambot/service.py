"""
Telegram bot package service.

Головний facade/lifecycle layer для Telegram notification service.

Цей модуль:
- інтегрується з core.EventBus;
- інтегрується з core.Scheduler для healthcheck;
- створює client/router/formatter/handlers/state;
- реєструє EventBus subscriptions;
- не містить trading/business logic;
- не читає market data напряму;
- не викликає analytics/strategy/risk/execution напряму.

Pipeline:
    EventBus events
        -> TelegramEventHandlers
        -> TelegramRouter
        -> TelegramFormatter
        -> TelegramBotClient
        -> Telegram forum topic
"""

from __future__ import annotations

from typing import Any

from core.event_bus import EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from .client import TelegramBotClient
from .config import TelegramBotConfig
from .enums import TelegramBotStatus, TelegramDeliveryStatus, TelegramTopic
from .exceptions import (
    TelegramConfigError,
    TelegramDependencyError,
    TelegramDisabledError,
    TelegramServiceError,
    TelegramStateError,
)
from .formatter import TelegramFormatter
from .handlers import TelegramEventHandlers
from .models import TelegramHealthStatus
from .router import TelegramRouter, TelegramRoutingRule
from .state import TelegramBotState


class TelegramBotService:
    """
    Telegram notification service.

    Відповідальність:
    - lifecycle: register/start/stop;
    - EventBus subscriptions;
    - Scheduler healthcheck job;
    - dependency wiring;
    - safe stats/debug;
    - lifecycle/system events.

    Не відповідає за:
    - trading decisions;
    - signal validation;
    - risk checks;
    - order execution;
    - market data reading.
    """

    HEALTHCHECK_JOB_NAME = "telegram_bot_healthcheck"

    def __init__(
        self,
        *,
        config: TelegramBotConfig,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        client: TelegramBotClient | None = None,
        router: TelegramRouter | None = None,
        formatter: TelegramFormatter | None = None,
        routing_rules: list[TelegramRoutingRule] | None = None,
    ) -> None:
        if event_bus is None:
            raise TelegramDependencyError("TelegramBotService requires EventBus.")

        self.config = config
        self.event_bus = event_bus
        self.scheduler = scheduler

        self.state = TelegramBotState(enabled=config.enabled)
        self.state.rate_limit.enabled = config.rate_limit.enabled

        self.client = client or TelegramBotClient(config)
        self.router = router or TelegramRouter(config, rules=routing_rules)
        self.formatter = formatter or TelegramFormatter(config)

        self.handlers = TelegramEventHandlers(
            config=self.config,
            event_bus=self.event_bus,
            router=self.router,
            formatter=self.formatter,
            client=self.client,
            state=self.state,
        )

        self._subscriptions: list[Subscription] = []
        self._healthcheck_job_id: str | None = None
        self._registered: bool = False
        self._logger = get_logger(__name__)

        self._validate_config()

    async def register(self) -> None:
        """
        Реєструє EventBus subscriptions.

        Метод idempotent: повторний виклик не дублює підписки.
        """

        if self._registered:
            return

        if not self.config.enabled:
            self.state.mark_disabled()
            self._registered = True
            return

        try:
            self.state.initialize_topics(
                topic_ids=self.config.topic_ids,
                enabled_topics={
                    topic
                    for topic, thread_id in self.config.topic_ids.items()
                    if thread_id and thread_id > 0
                },
            )

            await self._subscribe_events()

            self._registered = True
            self.state.mark_registered()

            await self._emit_lifecycle_event(
                "system.telegram_bot.registered",
                {
                    "service": "telegram_bot",
                    "status": self.state.status.value,
                    "subscriptions": len(self._subscriptions),
                    "topics": {
                        topic.value: thread_id
                        for topic, thread_id in self.config.topic_ids.items()
                    },
                },
                priority=EventPriority.LOW,
            )

        except Exception as exc:
            self.state.mark_error(error=str(exc))
            raise TelegramServiceError(
                "Failed to register TelegramBotService.",
                details={"registered_subscriptions": len(self._subscriptions)},
                cause=exc,
            ) from exc

    async def start(self) -> None:
        """
        Стартує TelegramBotService.

        - register(), якщо ще не зареєстровано;
        - стартує TelegramBotClient;
        - запускає healthcheck через Scheduler, якщо scheduler переданий;
        - публікує lifecycle event.
        """

        if not self.config.enabled:
            self.state.mark_disabled()
            raise TelegramDisabledError("TelegramBotService is disabled by config.")

        if self.state.started:
            raise TelegramStateError("TelegramBotService is already started.")

        self.state.mark_starting()

        try:
            if not self._registered:
                await self.register()

            await self.client.start()

            health = await self.client.health_check()
            self.state.update_health(health)

            if not health.ok:
                self._logger.warning(
                    "Telegram bot healthcheck failed during start.",
                    extra={"error": health.error},
                )

            await self._start_healthcheck_job()

            self.state.mark_started()

            await self._emit_lifecycle_event(
                "system.telegram_bot.started",
                {
                    "service": "telegram_bot",
                    "status": self.state.status.value,
                    "health": health.to_dict(),
                    "topics_count": len(self.state.topics),
                    "healthcheck_job_id": self._healthcheck_job_id,
                },
                priority=EventPriority.LOW,
            )

        except Exception as exc:
            self.state.mark_error(error=str(exc))
            raise TelegramServiceError(
                "Failed to start TelegramBotService.",
                cause=exc,
            ) from exc

    async def stop(self) -> None:
        """
        Зупиняє TelegramBotService.

        - зупиняє healthcheck job;
        - закриває TelegramBotClient;
        - lifecycle event публікує до закриття client не обовʼязково;
        - EventBus subscriptions зазвичай залишаємо, бо EventBus сам керує lifecycle.
        """

        if not self.state.started and self.state.status == TelegramBotStatus.STOPPED:
            return

        self.state.mark_stopping()

        try:
            await self._stop_healthcheck_job()

            await self._emit_lifecycle_event(
                "system.telegram_bot.stopped",
                {
                    "service": "telegram_bot",
                    "status": TelegramBotStatus.STOPPED.value,
                    "stats": self.state.stats.to_dict(),
                },
                priority=EventPriority.LOW,
            )

            await self.client.close()
            self.state.mark_stopped()

        except Exception as exc:
            self.state.mark_error(error=str(exc))
            raise TelegramServiceError(
                "Failed to stop TelegramBotService.",
                cause=exc,
            ) from exc

    async def health_check(self) -> TelegramHealthStatus:
        """
        Виконує healthcheck Telegram API і оновлює state.
        """

        if not self.config.enabled:
            health = TelegramHealthStatus(
                ok=False,
                status="disabled",
                error="TelegramBotService is disabled.",
            )
            self.state.update_health(health)
            return health

        try:
            health = await self.client.health_check()
            self.state.update_health(health)

            await self._emit_lifecycle_event(
                "system.telegram_bot.healthcheck",
                {
                    "service": "telegram_bot",
                    "status": health.status,
                    "ok": health.ok,
                    "latency_ms": health.latency_ms,
                    "bot_username": health.bot_username,
                    "error": health.error,
                    "sent_messages": self.state.stats.sent_messages,
                    "failed_messages": self.state.stats.failed_messages,
                    "success_rate": self.state.stats.success_rate,
                },
                priority=EventPriority.LOW,
            )

            return health

        except Exception as exc:
            health = TelegramHealthStatus(
                ok=False,
                status="error",
                error=str(exc),
            )
            self.state.update_health(health)

            await self._emit_lifecycle_event(
                "system.telegram_bot.healthcheck_failed",
                {
                    "service": "telegram_bot",
                    "status": "error",
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                priority=EventPriority.NORMAL,
            )

            return health

    async def send_test_message(
        self,
        *,
        message: str = "Telegram bot test message.",
        topic: TelegramTopic = TelegramTopic.SYSTEM,
    ) -> bool:
        """
        Відправляє тестове повідомлення в Telegram topic.

        Корисно для ручної перевірки після start().
        """

        if not self.config.enabled:
            raise TelegramDisabledError("TelegramBotService is disabled by config.")

        result = await self.handlers.publish_test_message(
            message=message,
            topic=topic,
        )

        return result.ok and result.status == TelegramDeliveryStatus.SENT

    def stats(self, *, include_history: bool = False) -> dict[str, Any]:
        """
        Safe stats без bot_token/secrets.
        """

        return {
            "service": "telegram_bot",
            "registered": self._registered,
            "subscriptions": len(self._subscriptions),
            "healthcheck_job_id": self._healthcheck_job_id,
            "config": self.config.to_safe_dict(),
            "state": self.state.to_dict(include_history=include_history),
            "client": self.client.stats(),
            "router_rules": self.router.list_rules(),
        }

    def is_running(self) -> bool:
        return self.state.started and self.state.status == TelegramBotStatus.RUNNING

    async def _subscribe_events(self) -> None:
        """
        Реєструє всі потрібні EventBus subscriptions.

        Патерни відповідають нашій event-driven архітектурі:
        - analytics.* -> analytics topics;
        - news.* / ai.news.* -> news topic;
        - signal.* -> signals/open trades;
        - execution.* -> open trades/risk;
        - position.* -> open/closed trades;
        - risk.* -> risk topic;
        - system.* -> system topic.

        Щоб уникнути нескінченного циклу, system.telegram_bot.* теж слухається,
        але handlers мають захист від повторного emit error loops.
        """

        subscriptions: list[Subscription] = []

        subscriptions.append(
             self.event_bus.subscribe(
                "analytics.*",
                self.handlers.handle_analytics_event,
            )
        )

        subscriptions.append(
            self.event_bus.subscribe(
                "news.*",
                self.handlers.handle_news_event,
            )
        )

        subscriptions.append(
            self.event_bus.subscribe(
                "ai.*",
                self.handlers.handle_ai_event,
            )
        )

        subscriptions.append(
            self.event_bus.subscribe(
                "signal.*",
                self.handlers.handle_signal_event,
            )
        )

        subscriptions.append(
            self.event_bus.subscribe(
                "execution.*",
                self.handlers.handle_execution_event,
            )
        )

        subscriptions.append(
            self.event_bus.subscribe(
                "position.*",
                self.handlers.handle_position_event,
            )
        )

        subscriptions.append(
            self.event_bus.subscribe(
                "risk.*",
                self.handlers.handle_risk_event,
            )
        )

        subscriptions.append(
            self.event_bus.subscribe(
                "system.*",
                self.handlers.handle_system_event,
            )
        )

        self._subscriptions.extend(subscriptions)

    async def _start_healthcheck_job(self) -> None:
        """
        Додає periodic healthcheck job через core.Scheduler.

        Власні uncontrolled asyncio loops не створюємо.
        """

        if self.scheduler is None:
            return

        if self._healthcheck_job_id is not None:
            return

        job = self.scheduler.add_interval_job(
            name=self.HEALTHCHECK_JOB_NAME,
            func=self.health_check,
            interval=self.config.healthcheck_interval_sec,
            run_immediately=False,
            max_retries=1,
            retry_delay=self.config.retry.retry_delay_sec,
            timeout=self.config.request_timeout_sec + 5.0,
            allow_overlap=False,
        )

        self._healthcheck_job_id = getattr(job, "id", None) or getattr(job, "job_id", None)

    async def _stop_healthcheck_job(self) -> None:
        """
        Вимикає або видаляє healthcheck job.

        Підлаштовуємось під core.Scheduler API:
        - якщо є remove_job — видаляємо;
        - інакше пробуємо disable_job.
        """

        if self.scheduler is None:
            return

        if self._healthcheck_job_id is None:
            return

        try:
            if hasattr(self.scheduler, "remove_job"):
                self.scheduler.remove_job(self._healthcheck_job_id)
            elif hasattr(self.scheduler, "disable_job"):
               self.scheduler.disable_job(self._healthcheck_job_id)
        finally:
            self._healthcheck_job_id = None

    async def _emit_lifecycle_event(
        self,
        event_name: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.LOW,
    ) -> None:
        if not self.config.emit_lifecycle_events:
            return

        try:
            await self.event_bus.emit(
                event_name,
                payload,
                source="telegram_bot",
                priority=priority,
            )
        except Exception:
            self._logger.exception(
                "Failed to emit TelegramBotService lifecycle event.",
                extra={"event_name": event_name},
            )

    def _validate_config(self) -> None:
        """
        Локальна валідація service dependencies/config.
        """

        if self.config is None:
            raise TelegramConfigError("TelegramBotConfig is required.")

        if self.event_bus is None:
            raise TelegramDependencyError("EventBus is required.")

        self.config.validate()