"""
Telegram bot package models.

DTO-моделі для Telegram notification layer.

Цей модуль:
- не викликає Telegram API;
- не містить EventBus-підписок;
- не містить торгової бізнес-логіки;
- описує тільки структури даних, які використовують formatter/router/client/handlers/service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any

from .enums import (
    TelegramDeliveryStatus,
    TelegramEventCategory,
    TelegramMessageType,
    TelegramNotificationLevel,
    TelegramParseMode,
    TelegramPriority,
    TelegramTopic,
)


@dataclass(slots=True, frozen=True)
class TelegramEventMetadata:
    """
    Metadata з EventBus-події, потрібна Telegram notification layer.

    Це lightweight DTO, щоб handlers не тягнули повний Event у formatter/client.
    """

    event_name: str
    event_id: str | None = None
    source: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    timestamp_ms: int | None = None

    @property
    def namespace(self) -> str:
        """
        Повертає першу частину event_name.

        Наприклад:
        - analytics.orderflow.absorption -> analytics
        - position.closed -> position
        """

        if not self.event_name:
            return ""
        return self.event_name.split(".", maxsplit=1)[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_name": self.event_name,
            "event_id": self.event_id,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "timestamp_ms": self.timestamp_ms,
            "namespace": self.namespace,
        }


@dataclass(slots=True, frozen=True)
class TelegramEventPayload:
    """
    Нормалізований payload для formatter/router.

    payload зберігається як dict[str, Any], бо різні EventBus-події
    матимуть різну структуру: analytics, signal, risk, position, news.
    """

    metadata: TelegramEventMetadata
    category: TelegramEventCategory
    payload: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.payload:
            raise KeyError(f"Telegram event payload missing required key: {key}")
        return self.payload[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "category": self.category.value,
            "payload": dict(self.payload),
        }


@dataclass(slots=True, frozen=True)
class TelegramTopicRoute:
    """
    Результат routing: EventBus event -> Telegram topic/thread.

    topic — логічна гілка.
    thread_id — Telegram message_thread_id.
    message_type — тип повідомлення для formatter-а.
    """

    topic: TelegramTopic
    message_type: TelegramMessageType
    thread_id: int | None = None
    category: TelegramEventCategory = TelegramEventCategory.UNKNOWN
    priority: TelegramPriority = TelegramPriority.NORMAL
    level: TelegramNotificationLevel = TelegramNotificationLevel.INFO
    reason: str | None = None

    @property
    def is_routable(self) -> bool:
        return self.thread_id is not None and self.thread_id > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic.value,
            "message_type": self.message_type.value,
            "thread_id": self.thread_id,
            "category": self.category.value,
            "priority": self.priority.value,
            "level": self.level.value,
            "reason": self.reason,
            "is_routable": self.is_routable,
        }


@dataclass(slots=True, frozen=True)
class TelegramFormattedMessage:
    """
    Повідомлення після formatter-а, але ще до Telegram client-а.

    Тут уже є готовий text, parse_mode і metadata для подальшої доставки.
    """

    text: str
    topic: TelegramTopic
    message_type: TelegramMessageType
    parse_mode: TelegramParseMode = TelegramParseMode.HTML
    level: TelegramNotificationLevel = TelegramNotificationLevel.INFO
    priority: TelegramPriority = TelegramPriority.NORMAL
    title: str | None = None
    disable_web_page_preview: bool = True
    disable_notification: bool = False
    protect_content: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, max_length: int = 4096) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("Telegram formatted message text is empty.")

        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        if len(self.text) > max_length:
            raise ValueError(
                f"Telegram formatted message text is too long: "
                f"{len(self.text)} > {max_length}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "topic": self.topic.value,
            "message_type": self.message_type.value,
            "parse_mode": self.parse_mode.value,
            "level": self.level.value,
            "priority": self.priority.value,
            "title": self.title,
            "disable_web_page_preview": self.disable_web_page_preview,
            "disable_notification": self.disable_notification,
            "protect_content": self.protect_content,
            "extra": dict(self.extra),
        }


@dataclass(slots=True, frozen=True)
class TelegramSendRequest:
    """
    Запит на відправку повідомлення в Telegram.

    Це модель, яку отримує TelegramBotClient.
    """

    chat_id: int | str
    text: str
    message_thread_id: int | None = None
    parse_mode: TelegramParseMode = TelegramParseMode.HTML
    disable_web_page_preview: bool = True
    disable_notification: bool = False
    protect_content: bool = False
    reply_to_message_id: int | None = None
    allow_sending_without_reply: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, max_length: int = 4096) -> None:
        if self.chat_id is None or self.chat_id == "":
            raise ValueError("Telegram send request chat_id is required.")

        if not self.text or not self.text.strip():
            raise ValueError("Telegram send request text is empty.")

        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        if len(self.text) > max_length:
            raise ValueError(
                f"Telegram send request text is too long: "
                f"{len(self.text)} > {max_length}"
            )

        if self.message_thread_id is not None and self.message_thread_id <= 0:
            raise ValueError("Telegram message_thread_id must be positive.")

        if self.reply_to_message_id is not None and self.reply_to_message_id <= 0:
            raise ValueError("Telegram reply_to_message_id must be positive.")

    def to_api_payload(self) -> dict[str, Any]:
        """
        Формує payload для Telegram sendMessage.

        None-значення не включаються.
        """

        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": self.text,
            "disable_web_page_preview": self.disable_web_page_preview,
            "disable_notification": self.disable_notification,
            "protect_content": self.protect_content,
            "allow_sending_without_reply": self.allow_sending_without_reply,
        }

        if self.parse_mode != TelegramParseMode.PLAIN:
            payload["parse_mode"] = self.parse_mode.value

        if self.message_thread_id is not None:
            payload["message_thread_id"] = self.message_thread_id

        if self.reply_to_message_id is not None:
            payload["reply_to_message_id"] = self.reply_to_message_id

        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "text": self.text,
            "message_thread_id": self.message_thread_id,
            "parse_mode": self.parse_mode.value,
            "disable_web_page_preview": self.disable_web_page_preview,
            "disable_notification": self.disable_notification,
            "protect_content": self.protect_content,
            "reply_to_message_id": self.reply_to_message_id,
            "allow_sending_without_reply": self.allow_sending_without_reply,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class TelegramSendResult:
    """
    Результат відправки Telegram-повідомлення.

    Не кидаємо exception на кожну failed delivery вгору без потреби:
    client може повертати цей DTO, а service/handlers вирішують,
    чи retry/skip/log/emit system event.
    """

    status: TelegramDeliveryStatus
    ok: bool
    message_id: int | None = None
    chat_id: int | str | None = None
    message_thread_id: int | None = None
    error: str | None = None
    error_code: int | None = None
    retry_after_sec: float | None = None
    attempt: int = 1
    sent_at_ms: int | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def sent(
        cls,
        *,
        message_id: int | None,
        chat_id: int | str | None,
        message_thread_id: int | None,
        attempt: int = 1,
        raw_response: dict[str, Any] | None = None,
    ) -> TelegramSendResult:
        return cls(
            status=TelegramDeliveryStatus.SENT,
            ok=True,
            message_id=message_id,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            attempt=attempt,
            sent_at_ms=_now_ms(),
            raw_response=raw_response or {},
        )

    @classmethod
    def failed(
        cls,
        *,
        error: str,
        chat_id: int | str | None = None,
        message_thread_id: int | None = None,
        error_code: int | None = None,
        retry_after_sec: float | None = None,
        attempt: int = 1,
        raw_response: dict[str, Any] | None = None,
    ) -> TelegramSendResult:
        status = (
            TelegramDeliveryStatus.RATE_LIMITED
            if retry_after_sec is not None
            else TelegramDeliveryStatus.FAILED
        )

        return cls(
            status=status,
            ok=False,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            error=error,
            error_code=error_code,
            retry_after_sec=retry_after_sec,
            attempt=attempt,
            raw_response=raw_response or {},
        )

    @classmethod
    def skipped(
        cls,
        *,
        reason: str,
        chat_id: int | str | None = None,
        message_thread_id: int | None = None,
    ) -> TelegramSendResult:
        return cls(
            status=TelegramDeliveryStatus.SKIPPED,
            ok=False,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            error=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "ok": self.ok,
            "message_id": self.message_id,
            "chat_id": self.chat_id,
            "message_thread_id": self.message_thread_id,
            "error": self.error,
            "error_code": self.error_code,
            "retry_after_sec": self.retry_after_sec,
            "attempt": self.attempt,
            "sent_at_ms": self.sent_at_ms,
            "raw_response": dict(self.raw_response),
        }


@dataclass(slots=True, frozen=True)
class TelegramDeliveryRecord:
    """
    Lightweight record для state/history.

    Не зберігаємо повний text повідомлення, щоб не роздувати памʼять
    і випадково не тримати чутливі дані.
    """

    event_name: str | None
    topic: TelegramTopic
    message_type: TelegramMessageType
    status: TelegramDeliveryStatus
    message_id: int | None = None
    thread_id: int | None = None
    error: str | None = None
    created_at_ms: int = field(default_factory=lambda: _now_ms())

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_name": self.event_name,
            "topic": self.topic.value,
            "message_type": self.message_type.value,
            "status": self.status.value,
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "error": self.error,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(slots=True, frozen=True)
class TelegramRateLimitDecision:
    """
    Результат локальної rate-limit перевірки.
    """

    allowed: bool
    reason: str | None = None
    retry_after_sec: float | None = None
    topic: TelegramTopic | None = None

    @classmethod
    def allow(cls, *, topic: TelegramTopic | None = None) -> TelegramRateLimitDecision:
        return cls(allowed=True, topic=topic)

    @classmethod
    def deny(
        cls,
        *,
        reason: str,
        retry_after_sec: float | None = None,
        topic: TelegramTopic | None = None,
    ) -> TelegramRateLimitDecision:
        return cls(
            allowed=False,
            reason=reason,
            retry_after_sec=retry_after_sec,
            topic=topic,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "retry_after_sec": self.retry_after_sec,
            "topic": self.topic.value if self.topic else None,
        }


@dataclass(slots=True, frozen=True)
class TelegramHealthStatus:
    """
    Healthcheck результат TelegramBotService/client.
    """

    ok: bool
    status: str
    checked_at_ms: int = field(default_factory=lambda: _now_ms())
    latency_ms: float | None = None
    bot_username: str | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "checked_at_ms": self.checked_at_ms,
            "latency_ms": self.latency_ms,
            "bot_username": self.bot_username,
            "error": self.error,
            "details": dict(self.details),
        }


@dataclass(slots=True, frozen=True)
class TelegramMessageChunk:
    """
    Частина довгого повідомлення після split.

    Telegram sendMessage має обмеження довжини тексту, тому formatter/client
    можуть розбивати довгі повідомлення на chunks.
    """

    index: int
    total: int
    text: str

    @property
    def is_first(self) -> bool:
        return self.index == 1

    @property
    def is_last(self) -> bool:
        return self.index == self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "total": self.total,
            "text": self.text,
            "is_first": self.is_first,
            "is_last": self.is_last,
        }


@dataclass(slots=True, frozen=True)
class TelegramResolvedMessage:
    """
    Повністю готове повідомлення після route + format.

    Це зручна модель для handlers:
    Event -> route -> format -> resolved -> client send request.
    """

    route: TelegramTopicRoute
    formatted: TelegramFormattedMessage
    event: TelegramEventPayload

    def to_send_request(
        self,
        *,
        chat_id: int | str,
        max_length: int = 4096,
    ) -> TelegramSendRequest:
        self.formatted.validate(max_length=max_length)

        return TelegramSendRequest(
            chat_id=chat_id,
            text=self.formatted.text,
            message_thread_id=self.route.thread_id,
            parse_mode=self.formatted.parse_mode,
            disable_web_page_preview=self.formatted.disable_web_page_preview,
            disable_notification=self.formatted.disable_notification,
            protect_content=self.formatted.protect_content,
            metadata={
                "event": self.event.metadata.to_dict(),
                "route": self.route.to_dict(),
                "message_type": self.formatted.message_type.value,
                "topic": self.formatted.topic.value,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route.to_dict(),
            "formatted": self.formatted.to_dict(),
            "event": self.event.to_dict(),
        }


def _now_ms() -> int:
    return int(time() * 1000)