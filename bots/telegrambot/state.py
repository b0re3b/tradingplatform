"""
Telegram bot package state.

Runtime state для Telegram notification layer.

Цей модуль:
- не викликає Telegram API;
- не підписується на EventBus;
- не містить торгової логіки;
- зберігає тільки runtime counters, delivery history, health status і rate-limit state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import monotonic, time
from typing import Any

from .enums import (
    TelegramBotStatus,
    TelegramDeliveryStatus,
    TelegramMessageType,
    TelegramTopic,
)
from .models import (
    TelegramDeliveryRecord,
    TelegramHealthStatus,
    TelegramRateLimitDecision,
)


@dataclass(slots=True)
class TelegramTopicState:
    """
    Runtime state окремої Telegram topic-гілки.

    Використовується для:
    - статистики по topic;
    - локального rate-limit;
    - debug/stats без зберігання повного тексту повідомлень.
    """

    topic: TelegramTopic
    thread_id: int | None = None
    enabled: bool = True

    sent_messages: int = 0
    failed_messages: int = 0
    skipped_messages: int = 0
    rate_limited_messages: int = 0

    last_message_id: int | None = None
    last_sent_at_ms: int | None = None
    last_failed_at_ms: int | None = None
    last_error: str | None = None

    # Monotonic timestamps для rate-limit.
    last_sent_monotonic: float | None = None
    sent_timestamps_monotonic: deque[float] = field(default_factory=deque)

    def mark_sent(self, *, message_id: int | None = None) -> None:
        now_ms = _now_ms()
        now_mono = monotonic()

        self.sent_messages += 1
        self.last_message_id = message_id
        self.last_sent_at_ms = now_ms
        self.last_sent_monotonic = now_mono
        self.last_error = None
        self.sent_timestamps_monotonic.append(now_mono)

    def mark_failed(self, *, error: str | None = None) -> None:
        self.failed_messages += 1
        self.last_failed_at_ms = _now_ms()
        self.last_error = error

    def mark_skipped(self, *, reason: str | None = None) -> None:
        self.skipped_messages += 1
        self.last_error = reason

    def mark_rate_limited(self, *, reason: str | None = None) -> None:
        self.rate_limited_messages += 1
        self.last_error = reason

    def cleanup_rate_limit_window(self, *, window_sec: float = 60.0) -> None:
        """
        Видаляє старі monotonic timestamps поза rate-limit window.
        """

        if window_sec <= 0:
            self.sent_timestamps_monotonic.clear()
            return

        threshold = monotonic() - window_sec

        while (
            self.sent_timestamps_monotonic
            and self.sent_timestamps_monotonic[0] < threshold
        ):
            self.sent_timestamps_monotonic.popleft()

    def messages_in_window(self, *, window_sec: float = 60.0) -> int:
        self.cleanup_rate_limit_window(window_sec=window_sec)
        return len(self.sent_timestamps_monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic.value,
            "thread_id": self.thread_id,
            "enabled": self.enabled,
            "sent_messages": self.sent_messages,
            "failed_messages": self.failed_messages,
            "skipped_messages": self.skipped_messages,
            "rate_limited_messages": self.rate_limited_messages,
            "last_message_id": self.last_message_id,
            "last_sent_at_ms": self.last_sent_at_ms,
            "last_failed_at_ms": self.last_failed_at_ms,
            "last_error": self.last_error,
            "messages_in_last_minute": self.messages_in_window(window_sec=60.0),
        }


@dataclass(slots=True)
class TelegramDeliveryStats:
    """
    Aggregated delivery counters для всього TelegramBotService.
    """

    total_messages: int = 0
    sent_messages: int = 0
    failed_messages: int = 0
    skipped_messages: int = 0
    rate_limited_messages: int = 0
    retried_messages: int = 0

    messages_by_type: dict[TelegramMessageType, int] = field(default_factory=dict)
    messages_by_topic: dict[TelegramTopic, int] = field(default_factory=dict)
    failures_by_topic: dict[TelegramTopic, int] = field(default_factory=dict)

    first_sent_at_ms: int | None = None
    last_sent_at_ms: int | None = None
    last_failed_at_ms: int | None = None
    last_error: str | None = None

    def record(
        self,
        *,
        topic: TelegramTopic,
        message_type: TelegramMessageType,
        status: TelegramDeliveryStatus,
        error: str | None = None,
    ) -> None:
        self.total_messages += 1

        self.messages_by_type[message_type] = (
            self.messages_by_type.get(message_type, 0) + 1
        )
        self.messages_by_topic[topic] = self.messages_by_topic.get(topic, 0) + 1

        if status == TelegramDeliveryStatus.SENT:
            now_ms = _now_ms()
            self.sent_messages += 1
            self.last_sent_at_ms = now_ms
            if self.first_sent_at_ms is None:
                self.first_sent_at_ms = now_ms
            self.last_error = None
            return

        if status == TelegramDeliveryStatus.FAILED:
            self.failed_messages += 1
            self.failures_by_topic[topic] = self.failures_by_topic.get(topic, 0) + 1
            self.last_failed_at_ms = _now_ms()
            self.last_error = error
            return

        if status == TelegramDeliveryStatus.SKIPPED:
            self.skipped_messages += 1
            self.last_error = error
            return

        if status == TelegramDeliveryStatus.RATE_LIMITED:
            self.rate_limited_messages += 1
            self.last_error = error
            return

        if status == TelegramDeliveryStatus.RETRYING:
            self.retried_messages += 1
            self.last_error = error

    @property
    def success_rate(self) -> float:
        if self.total_messages <= 0:
            return 0.0
        return self.sent_messages / self.total_messages

    @property
    def failure_rate(self) -> float:
        if self.total_messages <= 0:
            return 0.0
        return self.failed_messages / self.total_messages

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_messages": self.total_messages,
            "sent_messages": self.sent_messages,
            "failed_messages": self.failed_messages,
            "skipped_messages": self.skipped_messages,
            "rate_limited_messages": self.rate_limited_messages,
            "retried_messages": self.retried_messages,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "messages_by_type": {
                message_type.value: count
                for message_type, count in self.messages_by_type.items()
            },
            "messages_by_topic": {
                topic.value: count for topic, count in self.messages_by_topic.items()
            },
            "failures_by_topic": {
                topic.value: count for topic, count in self.failures_by_topic.items()
            },
            "first_sent_at_ms": self.first_sent_at_ms,
            "last_sent_at_ms": self.last_sent_at_ms,
            "last_failed_at_ms": self.last_failed_at_ms,
            "last_error": self.last_error,
        }


@dataclass(slots=True)
class TelegramRateLimitState:
    """
    Локальний rate-limit state.

    Реальна перевірка Telegram API rate limits буде в client.py,
    але цей state потрібен, щоб не створювати burst у конкретні topic-гілки.
    """

    enabled: bool = True

    # Monotonic timestamps усіх успішно дозволених повідомлень.
    global_sent_timestamps: deque[float] = field(default_factory=deque)

    last_limited_at_ms: int | None = None
    last_limited_topic: TelegramTopic | None = None
    last_limited_reason: str | None = None

    def check_global_limit(
        self,
        *,
        max_messages_per_second: float,
    ) -> TelegramRateLimitDecision:
        if not self.enabled:
            return TelegramRateLimitDecision.allow()

        if max_messages_per_second <= 0:
            return TelegramRateLimitDecision.deny(
                reason="global rate limit config is invalid",
                retry_after_sec=1.0,
            )

        now = monotonic()
        window_sec = 1.0
        threshold = now - window_sec

        while self.global_sent_timestamps and self.global_sent_timestamps[0] < threshold:
            self.global_sent_timestamps.popleft()

        if len(self.global_sent_timestamps) >= max_messages_per_second:
            retry_after = max(0.01, window_sec - (now - self.global_sent_timestamps[0]))
            self.mark_limited(
                topic=None,
                reason="global telegram rate limit exceeded",
            )
            return TelegramRateLimitDecision.deny(
                reason="global telegram rate limit exceeded",
                retry_after_sec=retry_after,
            )

        return TelegramRateLimitDecision.allow()

    def check_topic_limit(
        self,
        *,
        topic_state: TelegramTopicState,
        max_messages_per_topic_per_minute: int,
        min_interval_per_topic_sec: float,
    ) -> TelegramRateLimitDecision:
        if not self.enabled:
            return TelegramRateLimitDecision.allow(topic=topic_state.topic)

        if max_messages_per_topic_per_minute <= 0:
            return TelegramRateLimitDecision.deny(
                topic=topic_state.topic,
                reason="topic rate limit config is invalid",
                retry_after_sec=60.0,
            )

        now = monotonic()

        if (
            topic_state.last_sent_monotonic is not None
            and min_interval_per_topic_sec > 0
        ):
            elapsed = now - topic_state.last_sent_monotonic
            if elapsed < min_interval_per_topic_sec:
                retry_after = min_interval_per_topic_sec - elapsed
                reason = "telegram topic min interval exceeded"
                self.mark_limited(topic=topic_state.topic, reason=reason)
                return TelegramRateLimitDecision.deny(
                    topic=topic_state.topic,
                    reason=reason,
                    retry_after_sec=retry_after,
                )

        topic_state.cleanup_rate_limit_window(window_sec=60.0)
        if (
            len(topic_state.sent_timestamps_monotonic)
            >= max_messages_per_topic_per_minute
        ):
            oldest = topic_state.sent_timestamps_monotonic[0]
            retry_after = max(0.01, 60.0 - (now - oldest))
            reason = "telegram topic messages per minute exceeded"
            self.mark_limited(topic=topic_state.topic, reason=reason)
            return TelegramRateLimitDecision.deny(
                topic=topic_state.topic,
                reason=reason,
                retry_after_sec=retry_after,
            )

        return TelegramRateLimitDecision.allow(topic=topic_state.topic)

    def mark_allowed(self) -> None:
        if not self.enabled:
            return

        self.global_sent_timestamps.append(monotonic())

    def mark_limited(
        self,
        *,
        topic: TelegramTopic | None,
        reason: str,
    ) -> None:
        self.last_limited_at_ms = _now_ms()
        self.last_limited_topic = topic
        self.last_limited_reason = reason

    def cleanup(self, *, global_window_sec: float = 1.0) -> None:
        if global_window_sec <= 0:
            self.global_sent_timestamps.clear()
            return

        threshold = monotonic() - global_window_sec

        while self.global_sent_timestamps and self.global_sent_timestamps[0] < threshold:
            self.global_sent_timestamps.popleft()

    def to_dict(self) -> dict[str, Any]:
        self.cleanup(global_window_sec=1.0)

        return {
            "enabled": self.enabled,
            "global_messages_in_last_second": len(self.global_sent_timestamps),
            "last_limited_at_ms": self.last_limited_at_ms,
            "last_limited_topic": (
                self.last_limited_topic.value if self.last_limited_topic else None
            ),
            "last_limited_reason": self.last_limited_reason,
        }


@dataclass(slots=True)
class TelegramBotState:
    """
    Центральний runtime state TelegramBotService.

    Один екземпляр цього класу має жити всередині TelegramBotService.
    """

    status: TelegramBotStatus = TelegramBotStatus.CREATED
    registered: bool = False
    started: bool = False
    enabled: bool = True

    created_at_ms: int = field(default_factory=lambda: _now_ms())
    registered_at_ms: int | None = None
    started_at_ms: int | None = None
    stopped_at_ms: int | None = None
    last_event_at_ms: int | None = None

    last_error: str | None = None
    last_error_at_ms: int | None = None

    health: TelegramHealthStatus | None = None
    stats: TelegramDeliveryStats = field(default_factory=TelegramDeliveryStats)
    rate_limit: TelegramRateLimitState = field(default_factory=TelegramRateLimitState)

    topics: dict[TelegramTopic, TelegramTopicState] = field(default_factory=dict)

    # Обмежена історія delivery без тексту повідомлень.
    delivery_history: deque[TelegramDeliveryRecord] = field(
        default_factory=lambda: deque(maxlen=500)
    )

    def initialize_topics(
        self,
        *,
        topic_ids: dict[TelegramTopic, int],
        enabled_topics: set[TelegramTopic] | None = None,
    ) -> None:
        """
        Ініціалізує state для topic-гілок із config.topic_ids.
        """

        enabled_topics = enabled_topics or set(topic_ids.keys())

        for topic, thread_id in topic_ids.items():
            self.topics[topic] = TelegramTopicState(
                topic=topic,
                thread_id=thread_id if thread_id and thread_id > 0 else None,
                enabled=topic in enabled_topics and bool(thread_id and thread_id > 0),
            )

    def get_or_create_topic_state(
        self,
        topic: TelegramTopic,
        *,
        thread_id: int | None = None,
    ) -> TelegramTopicState:
        topic_state = self.topics.get(topic)
        if topic_state is not None:
            if thread_id is not None and topic_state.thread_id is None:
                topic_state.thread_id = thread_id
            return topic_state

        topic_state = TelegramTopicState(
            topic=topic,
            thread_id=thread_id,
            enabled=bool(thread_id and thread_id > 0),
        )
        self.topics[topic] = topic_state
        return topic_state

    def mark_registered(self) -> None:
        self.registered = True
        self.registered_at_ms = _now_ms()
        self.status = TelegramBotStatus.REGISTERED

    def mark_starting(self) -> None:
        self.status = TelegramBotStatus.STARTING

    def mark_started(self) -> None:
        now_ms = _now_ms()
        self.started = True
        self.started_at_ms = now_ms
        self.stopped_at_ms = None
        self.status = TelegramBotStatus.RUNNING
        self.last_error = None

    def mark_stopping(self) -> None:
        self.status = TelegramBotStatus.STOPPING

    def mark_stopped(self) -> None:
        self.started = False
        self.stopped_at_ms = _now_ms()
        self.status = TelegramBotStatus.STOPPED

    def mark_disabled(self) -> None:
        self.enabled = False
        self.started = False
        self.status = TelegramBotStatus.DISABLED

    def mark_error(self, *, error: str) -> None:
        self.last_error = error
        self.last_error_at_ms = _now_ms()
        self.status = TelegramBotStatus.ERROR

    def mark_event_received(self) -> None:
        self.last_event_at_ms = _now_ms()

    def update_health(self, health: TelegramHealthStatus) -> None:
        self.health = health

        if not health.ok:
            self.last_error = health.error
            self.last_error_at_ms = health.checked_at_ms

    def record_delivery(
        self,
        *,
        topic: TelegramTopic,
        message_type: TelegramMessageType,
        status: TelegramDeliveryStatus,
        event_name: str | None = None,
        message_id: int | None = None,
        thread_id: int | None = None,
        error: str | None = None,
    ) -> None:
        """
        Оновлює global stats, topic stats і delivery history.
        """

        self.stats.record(
            topic=topic,
            message_type=message_type,
            status=status,
            error=error,
        )

        topic_state = self.get_or_create_topic_state(topic, thread_id=thread_id)

        if status == TelegramDeliveryStatus.SENT:
            topic_state.mark_sent(message_id=message_id)
        elif status == TelegramDeliveryStatus.FAILED:
            topic_state.mark_failed(error=error)
        elif status == TelegramDeliveryStatus.SKIPPED:
            topic_state.mark_skipped(reason=error)
        elif status == TelegramDeliveryStatus.RATE_LIMITED:
            topic_state.mark_rate_limited(reason=error)

        self.delivery_history.append(
            TelegramDeliveryRecord(
                event_name=event_name,
                topic=topic,
                message_type=message_type,
                status=status,
                message_id=message_id,
                thread_id=thread_id,
                error=error,
            )
        )

    def check_rate_limit(
        self,
        *,
        topic: TelegramTopic,
        max_messages_per_second: float,
        max_messages_per_topic_per_minute: int,
        min_interval_per_topic_sec: float,
    ) -> TelegramRateLimitDecision:
        """
        Перевіряє global і topic-level rate limit.
        """

        global_decision = self.rate_limit.check_global_limit(
            max_messages_per_second=max_messages_per_second,
        )
        if not global_decision.allowed:
            return global_decision

        topic_state = self.get_or_create_topic_state(topic)
        topic_decision = self.rate_limit.check_topic_limit(
            topic_state=topic_state,
            max_messages_per_topic_per_minute=max_messages_per_topic_per_minute,
            min_interval_per_topic_sec=min_interval_per_topic_sec,
        )
        if not topic_decision.allowed:
            return topic_decision

        return TelegramRateLimitDecision.allow(topic=topic)

    def mark_rate_limit_allowed(self) -> None:
        """
        Викликається після успішного проходження rate-limit перед send.
        """

        self.rate_limit.mark_allowed()

    def uptime_sec(self) -> float:
        if not self.started_at_ms:
            return 0.0

        end_ms = self.stopped_at_ms or _now_ms()
        return max(0.0, (end_ms - self.started_at_ms) / 1000.0)

    def recent_deliveries(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        records = list(self.delivery_history)[-limit:]
        return [record.to_dict() for record in records]

    def to_dict(self, *, include_history: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status.value,
            "registered": self.registered,
            "started": self.started,
            "enabled": self.enabled,
            "created_at_ms": self.created_at_ms,
            "registered_at_ms": self.registered_at_ms,
            "started_at_ms": self.started_at_ms,
            "stopped_at_ms": self.stopped_at_ms,
            "last_event_at_ms": self.last_event_at_ms,
            "last_error": self.last_error,
            "last_error_at_ms": self.last_error_at_ms,
            "uptime_sec": self.uptime_sec(),
            "health": self.health.to_dict() if self.health else None,
            "stats": self.stats.to_dict(),
            "rate_limit": self.rate_limit.to_dict(),
            "topics": {
                topic.value: topic_state.to_dict()
                for topic, topic_state in self.topics.items()
            },
        }

        if include_history:
            data["delivery_history"] = self.recent_deliveries(limit=100)

        return data


def _now_ms() -> int:
    return int(time() * 1000)