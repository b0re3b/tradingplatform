from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import time
from typing import Any

from core.event_bus import Event, EventBus, EventPriority
from core.logger import get_logger
from .client import TelegramBotClient
from .config import TelegramBotConfig
from .enums import (
    TelegramDeliveryStatus,
    TelegramMessageType,
    TelegramTopic,
)
from .exceptions import (
    TelegramFormattingError,
    TelegramHandlerError,
    TelegramRoutingError,
    TelegramTopicNotConfiguredError,
)
from .formatter import TelegramFormatter
from .models import (
    TelegramEventMetadata,
    TelegramEventPayload,
    TelegramResolvedMessage,
    TelegramSendRequest,
    TelegramSendResult,
    TelegramTopicRoute,
)
from .router import TelegramRouter
from .state import TelegramBotState


@dataclass(slots=True, frozen=True)
class TelegramHandlerResult:
    """
    Результат обробки однієї EventBus-події.
    """

    ok: bool
    event_name: str
    status: TelegramDeliveryStatus
    message_type: TelegramMessageType | None = None
    topic: TelegramTopic | None = None
    sent_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "event_name": self.event_name,
            "status": self.status.value,
            "message_type": self.message_type.value if self.message_type else None,
            "topic": self.topic.value if self.topic else None,
            "sent_count": self.sent_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "error": self.error,
        }


@dataclass(slots=True, frozen=True)
class TelegramQueuedEvent:
    """EventBus event queued for asynchronous Telegram delivery."""

    sequence: int
    event: Event
    event_name: str
    enqueued_at_ms: int


class TelegramEventHandlers:
    """
    Набір EventBus handlers для TelegramBotService.

    Service створює цей клас і реєструє його methods через EventBus.subscribe().
    """

    def __init__(
        self,
        *,
        config: TelegramBotConfig,
        event_bus: EventBus,
        router: TelegramRouter,
        formatter: TelegramFormatter,
        client: TelegramBotClient,
        state: TelegramBotState,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.router = router
        self.formatter = formatter
        self.client = client
        self.state = state
        self._logger = get_logger(__name__)

        self._queue: asyncio.Queue[TelegramQueuedEvent | None] | None = (
            asyncio.Queue(maxsize=config.queue.max_size)
            if config.queue.enabled
            else None
        )
        self._queue_workers: list[asyncio.Task[None]] = []
        self._queue_running: bool = False
        self._queue_sequence: int = 0
        self.state.queue.configure(
            enabled=config.queue.enabled,
            max_size=config.queue.max_size,
            worker_count=config.queue.worker_count,
        )

    async def handle_analytics_event(self, event: Event) -> None:
        await self.handle_event(event)

    async def handle_news_event(self, event: Event) -> None:
        await self.handle_event(event)

    async def handle_ai_event(self, event: Event) -> None:
        await self.handle_event(event)

    async def handle_signal_event(self, event: Event) -> None:
        await self.handle_event(event)

    async def handle_execution_event(self, event: Event) -> None:
        await self.handle_event(event)

    async def handle_position_event(self, event: Event) -> None:
        await self.handle_event(event)

    async def handle_risk_event(self, event: Event) -> None:
        await self.handle_event(event)

    async def handle_system_event(self, event: Event) -> None:
        await self.handle_event(event)

    async def start_queue_workers(self) -> None:
        """Start controlled async workers used for Telegram delivery."""

        if not self.config.queue.enabled or self._queue is None:
            return

        if self._queue_workers:
            return

        self._queue_running = True
        self._queue_workers = [
            asyncio.create_task(
                self._queue_worker(worker_id=worker_id),
                name=f"telegram_delivery_worker_{worker_id}",
            )
            for worker_id in range(self.config.queue.worker_count)
        ]
        self.state.queue.mark_started(active_workers=len(self._queue_workers))

    async def stop_queue_workers(self) -> None:
        """Drain or cancel Telegram delivery workers during service shutdown."""

        if not self._queue_workers:
            self.state.queue.mark_stopped()
            return

        queue = self._queue
        self._queue_running = False

        if queue is not None and self.config.queue.drain_on_stop:
            try:
                await asyncio.wait_for(
                    queue.join(),
                    timeout=self.config.queue.shutdown_timeout_sec,
                )
            except asyncio.TimeoutError:
                self.state.queue.mark_error(
                    error="telegram queue drain timed out during shutdown",
                )

        if queue is not None and not self.config.queue.drain_on_stop:
            while True:
                try:
                    dropped = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if dropped is not None:
                    self.state.queue.mark_dropped(
                        reason="telegram queue dropped during shutdown",
                        queue_size=queue.qsize(),
                    )
                queue.task_done()

        if queue is not None:
            for _ in self._queue_workers:
                await queue.put(None)

        await asyncio.gather(*self._queue_workers, return_exceptions=True)
        self._queue_workers.clear()
        self.state.queue.mark_stopped()

    async def handle_event(self, event: Event) -> TelegramHandlerResult:
        """
        Універсальний EventBus handler.

        In async-queue mode this method does not call Telegram HTTP and does not
        sleep for rate limits. It only enqueues the event and returns quickly,
        keeping EventBus/analytics/risk/execution workers unblocked.
        """

        event_name = self._event_name(event)
        self.state.mark_event_received()

        if self.config.queue.enabled and self._queue is not None:
            return await self._enqueue_event(event=event, event_name=event_name)

        return await self._process_event_now(event)

    async def _process_event_now(self, event: Event) -> TelegramHandlerResult:
        """Route, format and deliver one EventBus event to Telegram."""

        event_name = self._event_name(event)

        if not self.config.enabled:
            return self._record_skipped(
                event_name=event_name,
                topic=TelegramTopic.SYSTEM,
                message_type=TelegramMessageType.SYSTEM_INFO,
                reason="Telegram bot is disabled.",
            )

        try:
            event_payload = self._to_telegram_event_payload(event)
            route = self.router.resolve(event_payload)

            if not route.is_routable:
                return self._record_skipped(
                    event_name=event_name,
                    topic=route.topic,
                    message_type=route.message_type,
                    reason=route.reason or "Telegram route is not routable.",
                    route=route,
                )

            rate_limit_decision = self.state.check_rate_limit(
                topic=route.topic,
                max_messages_per_second=self.config.rate_limit.max_messages_per_second,
                max_messages_per_topic_per_minute=(
                    self.config.rate_limit.max_messages_per_topic_per_minute
                ),
                min_interval_per_topic_sec=(
                    self.config.rate_limit.min_interval_per_topic_sec
                ),
            )

            if not rate_limit_decision.allowed:
                if self.config.rate_limit.drop_when_limited:
                    return self._record_rate_limited(
                        event_name=event_name,
                        topic=route.topic,
                        message_type=route.message_type,
                        reason=rate_limit_decision.reason
                        or "Telegram local rate limit exceeded.",
                    )

                if rate_limit_decision.retry_after_sec:
                    # Handler не створює background task. Він лише робить коротку
                    # контрольовану паузу в межах поточної обробки події.
                    await self._sleep_for_rate_limit(rate_limit_decision.retry_after_sec)

            formatted = self.formatter.format(route=route, event=event_payload)

            resolved = TelegramResolvedMessage(
                route=route,
                formatted=formatted,
                event=event_payload,
            )

            send_requests = self._build_send_requests(resolved)
            results = await self.client.send_messages(
                send_requests,
                max_length=self.config.max_message_length,
            )

            self.state.mark_rate_limit_allowed()
            return await self._record_send_results(
                event_name=event_name,
                route=route,
                results=results,
            )

        except TelegramTopicNotConfiguredError as exc:
            return await self._handle_processing_error(
                event=event,
                error=exc,
                message_type=TelegramMessageType.SYSTEM_WARNING,
                topic=TelegramTopic.SYSTEM,
                status=TelegramDeliveryStatus.SKIPPED,
            )

        except (TelegramRoutingError, TelegramFormattingError) as exc:
            return await self._handle_processing_error(
                event=event,
                error=exc,
                message_type=TelegramMessageType.SYSTEM_ERROR,
                topic=TelegramTopic.SYSTEM,
                status=TelegramDeliveryStatus.FAILED,
            )

        except Exception as exc:
            self._logger.exception(
                "Unexpected Telegram event handler error.",
                extra={"event_name": event_name},
            )

            wrapped = TelegramHandlerError(
                "Unexpected Telegram event handler error.",
                details={"event_name": event_name},
                cause=exc,
            )

            return await self._handle_processing_error(
                event=event,
                error=wrapped,
                message_type=TelegramMessageType.SYSTEM_ERROR,
                topic=TelegramTopic.SYSTEM,
                status=TelegramDeliveryStatus.FAILED,
            )

    async def _enqueue_event(
        self,
        *,
        event: Event,
        event_name: str,
    ) -> TelegramHandlerResult:
        queue = self._queue
        if queue is None:
            return await self._process_event_now(event)

        self._queue_sequence += 1
        item = TelegramQueuedEvent(
            sequence=self._queue_sequence,
            event=event,
            event_name=event_name,
            enqueued_at_ms=int(time() * 1000),
        )

        try:
            if not queue.full():
                queue.put_nowait(item)
                self.state.queue.mark_enqueued(queue_size=queue.qsize())
                return self._record_queued(event_name=event_name)

            policy = self.config.queue.full_policy

            if policy == "drop_newest":
                self.state.queue.mark_dropped(
                    reason="telegram queue is full; newest event dropped",
                    queue_size=queue.qsize(),
                )
                return TelegramHandlerResult(
                    ok=False,
                    event_name=event_name,
                    status=TelegramDeliveryStatus.RATE_LIMITED,
                    message_type=TelegramMessageType.SYSTEM_WARNING,
                    topic=TelegramTopic.SYSTEM,
                    failed_count=1,
                    error="telegram queue is full; newest event dropped",
                )

            if policy == "drop_oldest":
                try:
                    dropped = queue.get_nowait()
                    if dropped is not None:
                        self.state.queue.mark_dropped(
                            reason="telegram queue is full; oldest event dropped",
                            dropped_oldest=True,
                            queue_size=queue.qsize(),
                        )
                    queue.task_done()
                except asyncio.QueueEmpty:
                    pass

                queue.put_nowait(item)
                self.state.queue.mark_enqueued(queue_size=queue.qsize())
                return self._record_queued(event_name=event_name)

            try:
                await asyncio.wait_for(
                    queue.put(item),
                    timeout=self.config.queue.enqueue_timeout_sec,
                )
                self.state.queue.mark_enqueued(queue_size=queue.qsize())
                return self._record_queued(event_name=event_name)
            except asyncio.TimeoutError:
                self.state.queue.mark_dropped(
                    reason="telegram queue enqueue timed out",
                    timeout=True,
                    queue_size=queue.qsize(),
                )
                return TelegramHandlerResult(
                    ok=False,
                    event_name=event_name,
                    status=TelegramDeliveryStatus.RATE_LIMITED,
                    message_type=TelegramMessageType.SYSTEM_WARNING,
                    topic=TelegramTopic.SYSTEM,
                    failed_count=1,
                    error="telegram queue enqueue timed out",
                )

        except Exception as exc:
            self.state.queue.mark_error(error=str(exc))
            self._logger.exception(
                "Failed to enqueue Telegram event.",
                extra={"event_name": event_name},
            )
            return TelegramHandlerResult(
                ok=False,
                event_name=event_name,
                status=TelegramDeliveryStatus.FAILED,
                message_type=TelegramMessageType.SYSTEM_ERROR,
                topic=TelegramTopic.SYSTEM,
                failed_count=1,
                error=str(exc),
            )

    def _record_queued(self, *, event_name: str) -> TelegramHandlerResult:
        return TelegramHandlerResult(
            ok=True,
            event_name=event_name,
            status=TelegramDeliveryStatus.PENDING,
        )

    async def _queue_worker(self, *, worker_id: int) -> None:
        queue = self._queue
        if queue is None:
            return

        while True:
            item = await queue.get()
            try:
                if item is None:
                    return

                self.state.queue.mark_dequeued(queue_size=queue.qsize())
                result = await self._process_event_now(item.event)
                self.state.queue.mark_processed(ok=result.ok)

            except Exception as exc:
                self.state.queue.mark_processed(ok=False)
                self.state.queue.mark_error(error=str(exc))
                self._logger.exception(
                    "Unexpected Telegram queue worker error.",
                    extra={"worker_id": worker_id},
                )
            finally:
                queue.task_done()


    async def publish_test_message(
        self,
        *,
        message: str = "Telegram bot test message.",
        topic: TelegramTopic = TelegramTopic.SYSTEM,
    ) -> TelegramSendResult:
        """
        Допоміжний метод для service health/test.

        Не використовується trading logic. Просто перевіряє, що client може
        відправити повідомлення в задану гілку.
        """

        thread_id = self.config.get_topic_id(topic)

        if thread_id is None:
            return TelegramSendResult.skipped(
                reason=f"Telegram topic {topic.value} is not configured.",
                chat_id=self.config.chat_id,
                message_thread_id=None,
            )

        request = TelegramSendRequest(
            chat_id=self._require_chat_id(),
            text=message,
            message_thread_id=thread_id,
            parse_mode=self.config.parse_mode,
            disable_web_page_preview=True,
            disable_notification=True,
            protect_content=self.config.protect_content,
            metadata={
                "source": "telegram_bot.handlers.publish_test_message",
                "topic": topic.value,
            },
        )

        result = await self.client.send_message(
            request,
            max_length=self.config.max_message_length,
        )

        self.state.record_delivery(
            topic=topic,
            message_type=TelegramMessageType.SYSTEM_INFO,
            status=result.status,
            event_name="system.telegram_bot.test_message",
            message_id=result.message_id,
            thread_id=thread_id,
            error=result.error,
        )

        return result

    def _to_telegram_event_payload(self, event: Event) -> TelegramEventPayload:
        event_name = self._event_name(event)
        payload = self._event_payload(event)

        metadata = TelegramEventMetadata(
            event_name=event_name,
            event_id=self._optional_str_attr(event, "event_id", "id"),
            source=self._optional_str_attr(event, "source"),
            correlation_id=self._optional_str_attr(
                event,
                "correlation_id",
                "correlation",
                "trace_correlation_id",
            ),
            trace_id=self._optional_str_attr(event, "trace_id"),
            timestamp_ms=self._event_timestamp_ms(event),
        )

        category = self.router.category_for_event_name(event_name)

        return TelegramEventPayload(
            metadata=metadata,
            category=category,
            payload=payload,
        )

    def _build_send_requests(
        self,
        resolved: TelegramResolvedMessage,
    ) -> list[TelegramSendRequest]:
        """
        Формує TelegramSendRequest list.

        Якщо formatter повернув повідомлення довше max_message_length,
        розбиваємо його на chunks.
        """

        chat_id = self._require_chat_id()
        chunks = self.formatter.split_message(
            resolved.formatted,
            max_length=self.config.max_message_length,
        )

        requests: list[TelegramSendRequest] = []

        for chunk in chunks:
            requests.append(
                TelegramSendRequest(
                    chat_id=chat_id,
                    text=chunk.text,
                    message_thread_id=resolved.route.thread_id,
                    parse_mode=resolved.formatted.parse_mode,
                    disable_web_page_preview=(
                        resolved.formatted.disable_web_page_preview
                    ),
                    disable_notification=resolved.formatted.disable_notification,
                    protect_content=resolved.formatted.protect_content,
                    metadata={
                        "event": resolved.event.metadata.to_dict(),
                        "route": resolved.route.to_dict(),
                        "message_type": resolved.formatted.message_type.value,
                        "topic": resolved.formatted.topic.value,
                        "chunk": chunk.to_dict(),
                    },
                )
            )

        return requests

    async def _record_send_results(
        self,
        *,
        event_name: str,
        route: TelegramTopicRoute,
        results: list[TelegramSendResult],
    ) -> TelegramHandlerResult:
        sent_count = 0
        failed_count = 0
        skipped_count = 0
        last_error: str | None = None
        final_status = TelegramDeliveryStatus.SENT

        for result in results:
            if result.status == TelegramDeliveryStatus.SENT:
                sent_count += 1
            elif result.status == TelegramDeliveryStatus.SKIPPED:
                skipped_count += 1
                last_error = result.error
                final_status = TelegramDeliveryStatus.SKIPPED
            else:
                failed_count += 1
                last_error = result.error
                final_status = result.status

            self.state.record_delivery(
                topic=route.topic,
                message_type=route.message_type,
                status=result.status,
                event_name=event_name,
                message_id=result.message_id,
                thread_id=route.thread_id,
                error=result.error,
            )

        ok = failed_count == 0 and sent_count > 0

        if failed_count > 0:
            await self._emit_delivery_failed(
                event_name=event_name,
                route=route,
                error=last_error or "Telegram delivery failed.",
            )

        return TelegramHandlerResult(
            ok=ok,
            event_name=event_name,
            status=final_status,
            message_type=route.message_type,
            topic=route.topic,
            sent_count=sent_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            error=last_error,
        )

    def _record_skipped(
        self,
        *,
        event_name: str,
        topic: TelegramTopic,
        message_type: TelegramMessageType,
        reason: str,
        route: TelegramTopicRoute | None = None,
    ) -> TelegramHandlerResult:
        thread_id = route.thread_id if route else self.config.get_topic_id(topic)

        self.state.record_delivery(
            topic=topic,
            message_type=message_type,
            status=TelegramDeliveryStatus.SKIPPED,
            event_name=event_name,
            thread_id=thread_id,
            error=reason,
        )

        return TelegramHandlerResult(
            ok=False,
            event_name=event_name,
            status=TelegramDeliveryStatus.SKIPPED,
            message_type=message_type,
            topic=topic,
            skipped_count=1,
            error=reason,
        )

    def _record_rate_limited(
        self,
        *,
        event_name: str,
        topic: TelegramTopic,
        message_type: TelegramMessageType,
        reason: str,
    ) -> TelegramHandlerResult:
        thread_id = self.config.get_topic_id(topic)

        self.state.record_delivery(
            topic=topic,
            message_type=message_type,
            status=TelegramDeliveryStatus.RATE_LIMITED,
            event_name=event_name,
            thread_id=thread_id,
            error=reason,
        )

        return TelegramHandlerResult(
            ok=False,
            event_name=event_name,
            status=TelegramDeliveryStatus.RATE_LIMITED,
            message_type=message_type,
            topic=topic,
            failed_count=1,
            error=reason,
        )

    async def _handle_processing_error(
        self,
        *,
        event: Event,
        error: BaseException,
        message_type: TelegramMessageType,
        topic: TelegramTopic,
        status: TelegramDeliveryStatus,
    ) -> TelegramHandlerResult:
        event_name = self._event_name(event)
        error_message = str(error)

        self.state.record_delivery(
            topic=topic,
            message_type=message_type,
            status=status,
            event_name=event_name,
            thread_id=self.config.get_topic_id(topic),
            error=error_message,
        )

        self.state.mark_error(error=error_message)

        await self._emit_handler_error(
            event_name=event_name,
            error=error,
            status=status,
        )

        return TelegramHandlerResult(
            ok=False,
            event_name=event_name,
            status=status,
            message_type=message_type,
            topic=topic,
            failed_count=1 if status == TelegramDeliveryStatus.FAILED else 0,
            skipped_count=1 if status == TelegramDeliveryStatus.SKIPPED else 0,
            error=error_message,
        )

    async def _emit_delivery_failed(
        self,
        *,
        event_name: str,
        route: TelegramTopicRoute,
        error: str,
    ) -> None:
        """
        Публікує system event про failed delivery.

        Щоб не створити нескінченний цикл, не емiтимо помилку для
        system.telegram_bot.* подій.
        """

        if event_name.startswith("system.telegram_bot."):
            return

        await self._safe_emit(
            "system.telegram_bot.delivery_failed",
            {
                "service": "telegram_bot",
                "source_event": event_name,
                "topic": route.topic.value,
                "message_type": route.message_type.value,
                "thread_id": route.thread_id,
                "error": error,
            },
            priority=EventPriority.NORMAL,
        )

    async def _emit_handler_error(
        self,
        *,
        event_name: str,
        error: BaseException,
        status: TelegramDeliveryStatus,
    ) -> None:
        if event_name.startswith("system.telegram_bot."):
            return

        await self._safe_emit(
            "system.telegram_bot.handler_error",
            {
                "service": "telegram_bot",
                "source_event": event_name,
                "status": status.value,
                "error": str(error),
                "error_type": type(error).__name__,
            },
            priority=EventPriority.NORMAL,
        )

    async def _safe_emit(
        self,
        event_name: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        try:
            await self.event_bus.emit(
                event_name,
                payload,
                source="telegram_bot",
                priority=priority,
            )
        except Exception:
            self._logger.exception(
                "Failed to emit Telegram bot system event.",
                extra={"event_name": event_name},
            )

    async def _sleep_for_rate_limit(self, retry_after_sec: float) -> None:
        """
        Контрольована коротка пауза при локальному rate-limit.

        Щоб handler не блокувався надовго, обмежуємо sleep зверху.
        """

        safe_sleep = max(0.0, min(float(retry_after_sec), 3.0))
        if safe_sleep > 0:
            import asyncio

            await asyncio.sleep(safe_sleep)

    def _event_name(self, event: Event) -> str:
        for attr in ("name", "event_type", "type", "topic"):
            value = getattr(event, attr, None)
            if value:
                return str(value)

        return "unknown"

    def _event_payload(self, event: Event) -> dict[str, Any]:
        payload = getattr(event, "payload", None)

        if payload is None:
            data = getattr(event, "data", None)
            if isinstance(data, dict):
                return dict(data)
            return {}

        if isinstance(payload, dict):
            return dict(payload)

        return {"value": payload}

    def _event_timestamp_ms(self, event: Event) -> int | None:
        for attr in ("timestamp_ms", "created_at_ms", "ts_ms"):
            value = getattr(event, attr, None)
            if value is not None:
                return self._to_int_or_none(value)

        for attr in ("timestamp", "created_at", "ts"):
            value = getattr(event, attr, None)
            if value is None:
                continue

            number = self._to_float_or_none(value)
            if number is None:
                continue

            if number < 10_000_000_000:
                number *= 1000

            return int(number)

        return int(time() * 1000)

    def _optional_str_attr(self, event: Event, *attrs: str) -> str | None:
        for attr in attrs:
            value = getattr(event, attr, None)
            if value is not None and value != "":
                return str(value)
        return None

    def _to_int_or_none(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _to_float_or_none(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _require_chat_id(self) -> int | str:
        if self.config.chat_id is None or self.config.chat_id == "":
            raise TelegramHandlerError("Telegram chat_id is required.")
        return self.config.chat_id

    def stats(self) -> dict[str, Any]:
        """
        Safe stats для service.py.
        """

        return {
            "config_enabled": self.config.enabled,
            "state": self.state.to_dict(include_history=False),
            "router_rules": self.router.list_rules(),
            "client": self.client.stats(),
        }