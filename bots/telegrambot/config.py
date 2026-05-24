from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .enums import (
    TelegramParseMode,
    TelegramRoutePolicy,
    TelegramTopic,
)
from .exceptions import TelegramConfigError


@dataclass(slots=True)
class TelegramRetryConfig:
    """
    Retry policy для Telegram API запитів.
    """

    max_retries: int = 3
    retry_delay_sec: float = 1.0
    retry_backoff_multiplier: float = 2.0
    max_retry_delay_sec: float = 30.0
    retry_on_rate_limit: bool = True
    retry_on_timeout: bool = True
    retry_on_network_error: bool = True
    retry_on_5xx: bool = True

    def validate(self) -> None:
        if self.max_retries < 0:
            raise TelegramConfigError(
                "Telegram retry max_retries must be >= 0.",
                details={"max_retries": self.max_retries},
            )

        if self.retry_delay_sec < 0:
            raise TelegramConfigError(
                "Telegram retry_delay_sec must be >= 0.",
                details={"retry_delay_sec": self.retry_delay_sec},
            )

        if self.retry_backoff_multiplier < 1:
            raise TelegramConfigError(
                "Telegram retry_backoff_multiplier must be >= 1.",
                details={"retry_backoff_multiplier": self.retry_backoff_multiplier},
            )

        if self.max_retry_delay_sec < self.retry_delay_sec:
            raise TelegramConfigError(
                "Telegram max_retry_delay_sec must be >= retry_delay_sec.",
                details={
                    "retry_delay_sec": self.retry_delay_sec,
                    "max_retry_delay_sec": self.max_retry_delay_sec,
                },
            )


@dataclass(slots=True)
class TelegramRateLimitConfig:
    """
    Локальний rate-limit notification layer.

    Це не замінює Telegram API limits, але допомагає не спамити гілки
    при burst-подіях з analytics/execution/risk.
    """

    enabled: bool = True
    max_messages_per_second: float = 20.0
    max_messages_per_topic_per_minute: int = 60
    drop_when_limited: bool = False
    min_interval_per_topic_sec: float = 0.25

    def validate(self) -> None:
        if self.max_messages_per_second <= 0:
            raise TelegramConfigError(
                "Telegram max_messages_per_second must be > 0.",
                details={"max_messages_per_second": self.max_messages_per_second},
            )

        if self.max_messages_per_topic_per_minute <= 0:
            raise TelegramConfigError(
                "Telegram max_messages_per_topic_per_minute must be > 0.",
                details={
                    "max_messages_per_topic_per_minute": (
                        self.max_messages_per_topic_per_minute
                    )
                },
            )

        if self.min_interval_per_topic_sec < 0:
            raise TelegramConfigError(
                "Telegram min_interval_per_topic_sec must be >= 0.",
                details={"min_interval_per_topic_sec": self.min_interval_per_topic_sec},
            )


@dataclass(slots=True)
class TelegramQueueConfig:
    """
    Async delivery queue config for Telegram notification layer.

    EventBus handlers must not wait for Telegram HTTP, retries, or local
    rate-limit sleeps. They enqueue work into this bounded queue and return
    quickly; dedicated workers perform formatting and delivery.
    """

    enabled: bool = True
    max_size: int = 1000
    worker_count: int = 1
    enqueue_timeout_sec: float = 0.05
    shutdown_timeout_sec: float = 10.0
    drain_on_stop: bool = True
    full_policy: str = "drop_oldest"  # drop_oldest | drop_newest | block

    def validate(self) -> None:
        if self.max_size <= 0:
            raise TelegramConfigError(
                "Telegram queue max_size must be > 0.",
                details={"max_size": self.max_size},
            )

        if self.worker_count <= 0:
            raise TelegramConfigError(
                "Telegram queue worker_count must be > 0.",
                details={"worker_count": self.worker_count},
            )

        if self.enqueue_timeout_sec < 0:
            raise TelegramConfigError(
                "Telegram queue enqueue_timeout_sec must be >= 0.",
                details={"enqueue_timeout_sec": self.enqueue_timeout_sec},
            )

        if self.shutdown_timeout_sec < 0:
            raise TelegramConfigError(
                "Telegram queue shutdown_timeout_sec must be >= 0.",
                details={"shutdown_timeout_sec": self.shutdown_timeout_sec},
            )

        if self.full_policy not in {"drop_oldest", "drop_newest", "block"}:
            raise TelegramConfigError(
                "Telegram queue full_policy must be one of: drop_oldest, drop_newest, block.",
                details={"full_policy": self.full_policy},
            )


@dataclass(slots=True)
class TelegramTopicConfig:
    """
    Конфіг однієї Telegram topic-гілки.

    thread_id — це message_thread_id Telegram forum topic.
    """

    topic: TelegramTopic
    thread_id: int | None
    enabled: bool = True
    title: str | None = None
    description: str | None = None

    def validate(self) -> None:
        if not isinstance(self.topic, TelegramTopic):
            raise TelegramConfigError(
                "Telegram topic config has invalid topic.",
                details={"topic": str(self.topic)},
            )

        if self.enabled and self.thread_id is not None and self.thread_id <= 0:
            raise TelegramConfigError(
                "Telegram topic thread_id must be positive.",
                details={
                    "topic": self.topic.value,
                    "thread_id": self.thread_id,
                },
            )


@dataclass(slots=True)
class TelegramBotConfig:
    """
    Основний config для TelegramBotService.

    Очікувана Telegram структура:
    - Telegram supergroup/forum;
    - кожна analytics domain має власну topic-гілку;
    - news має окрему topic-гілку;
    - open trades має окрему topic-гілку;
    - closed trades/results має окрему topic-гілку.
    """

    enabled: bool = True

    # Sensitive: не логувати.
    bot_token: str | None = None

    # Telegram chat_id supergroup/forum.
    chat_id: int | str | None = None

    # Якщо default_topic_id заданий, router може відправляти unknown/default events туди.
    default_topic_id: int | None = None

    # Поведінка, якщо route/topic не знайдено.
    route_policy: TelegramRoutePolicy = TelegramRoutePolicy.SEND_TO_SYSTEM

    # Telegram formatting.
    parse_mode: TelegramParseMode = TelegramParseMode.HTML
    disable_web_page_preview: bool = True
    disable_notification: bool = False
    protect_content: bool = False

    # HTTP/API.
    api_base_url: str = "https://api.telegram.org"
    request_timeout_sec: float = 10.0
    connect_timeout_sec: float = 5.0

    # Message limits.
    max_message_length: int = 4096
    split_long_messages: bool = True

    # Lifecycle.
    healthcheck_interval_sec: float = 60.0
    emit_lifecycle_events: bool = True

    # Feature flags.
    enable_analytics_alerts: bool = True
    enable_news_alerts: bool = True
    enable_signal_alerts: bool = True
    enable_trade_updates: bool = True
    enable_risk_alerts: bool = True
    enable_system_alerts: bool = True

    # Analytics alert hygiene. Telegram should not publish context/noise updates
    # that carry no actionable signal strength. These thresholds are intentionally
    # Telegram-only: analytics still publishes/stores its own events, but the
    # notification layer filters low-value alerts.
    filter_non_actionable_liquidity_alerts: bool = True
    liquidity_min_signal_confidence_to_alert: float = 0.20
    liquidity_min_signal_score_to_alert: float = 0.20
    liquidity_min_sweep_risk_to_alert: float = 0.20
    liquidity_min_magnet_score_to_alert: float = 0.20
    liquidity_allow_neutral_bias_alerts: bool = False

    # Sub-configs.
    retry: TelegramRetryConfig = field(default_factory=TelegramRetryConfig)
    rate_limit: TelegramRateLimitConfig = field(default_factory=TelegramRateLimitConfig)
    queue: TelegramQueueConfig = field(default_factory=TelegramQueueConfig)

    # Мапінг topic -> message_thread_id.
    topic_ids: dict[TelegramTopic, int] = field(default_factory=dict)

    # Optional topic metadata.
    topics: dict[TelegramTopic, TelegramTopicConfig] = field(default_factory=dict)

    @classmethod
    def default_topic_ids(cls) -> dict[TelegramTopic, int]:
        """
        Повертає mapping з реальними message_thread_id для кожної topic-гілки.

        Значення задаються напряму в коді після створення Telegram forum topics.
        """

        return {
            TelegramTopic.ORDERFLOW:     9,
            TelegramTopic.LIQUIDITY:     21,
            TelegramTopic.PRICE_ACTION:  12,
            TelegramTopic.LIQUIDATIONS:  10,
            TelegramTopic.WHALES:        24,
            TelegramTopic.SPOOFING:      22,
            TelegramTopic.SPREADS:       23,
            TelegramTopic.FUNDING:       19,
            TelegramTopic.OPEN_INTEREST: 11,
            TelegramTopic.NEWS:          2,
            TelegramTopic.SIGNALS:       3,
            TelegramTopic.OPEN_TRADES:   4,
            TelegramTopic.CLOSED_TRADES: 25,

        }

    @classmethod
    def from_env(cls, prefix: str = "TELEGRAM_BOT_") -> TelegramBotConfig:
        """
        Створює TelegramBotConfig з environment variables.

        topic_ids беруться з default_topic_ids() — тобто задані напряму в коді.

        Приклади env:
        - TELEGRAM_BOT_ENABLED=true
        - TELEGRAM_BOT_TOKEN=...
        - TELEGRAM_BOT_CHAT_ID=-1001234567890
        """

        config = cls(
            enabled=_to_bool(os.getenv(f"{prefix}ENABLED"), default=True),
            bot_token=_to_optional_str(os.getenv(f"{prefix}TOKEN")),
            chat_id=_to_chat_id(os.getenv(f"{prefix}CHAT_ID")),
            default_topic_id=_to_optional_int(os.getenv(f"{prefix}DEFAULT_TOPIC_ID")),
            route_policy=_to_route_policy(
                os.getenv(f"{prefix}ROUTE_POLICY"),
                default=TelegramRoutePolicy.SEND_TO_SYSTEM,
            ),
            parse_mode=_to_parse_mode(
                os.getenv(f"{prefix}PARSE_MODE"),
                default=TelegramParseMode.HTML,
            ),
            disable_web_page_preview=_to_bool(
                os.getenv(f"{prefix}DISABLE_WEB_PAGE_PREVIEW"),
                default=True,
            ),
            disable_notification=_to_bool(
                os.getenv(f"{prefix}DISABLE_NOTIFICATION"),
                default=False,
            ),
            protect_content=_to_bool(
                os.getenv(f"{prefix}PROTECT_CONTENT"),
                default=False,
            ),
            api_base_url=os.getenv(f"{prefix}API_BASE_URL", "https://api.telegram.org"),
            request_timeout_sec=_to_float(
                os.getenv(f"{prefix}REQUEST_TIMEOUT_SEC"),
                default=10.0,
            ),
            connect_timeout_sec=_to_float(
                os.getenv(f"{prefix}CONNECT_TIMEOUT_SEC"),
                default=5.0,
            ),
            max_message_length=_to_int(
                os.getenv(f"{prefix}MAX_MESSAGE_LENGTH"),
                default=4096,
            ),
            split_long_messages=_to_bool(
                os.getenv(f"{prefix}SPLIT_LONG_MESSAGES"),
                default=True,
            ),
            healthcheck_interval_sec=_to_float(
                os.getenv(f"{prefix}HEALTHCHECK_INTERVAL_SEC"),
                default=60.0,
            ),
            emit_lifecycle_events=_to_bool(
                os.getenv(f"{prefix}EMIT_LIFECYCLE_EVENTS"),
                default=True,
            ),
            enable_analytics_alerts=_to_bool(
                os.getenv(f"{prefix}ENABLE_ANALYTICS_ALERTS"),
                default=True,
            ),
            enable_news_alerts=_to_bool(
                os.getenv(f"{prefix}ENABLE_NEWS_ALERTS"),
                default=True,
            ),
            enable_signal_alerts=_to_bool(
                os.getenv(f"{prefix}ENABLE_SIGNAL_ALERTS"),
                default=True,
            ),
            enable_trade_updates=_to_bool(
                os.getenv(f"{prefix}ENABLE_TRADE_UPDATES"),
                default=True,
            ),
            enable_risk_alerts=_to_bool(
                os.getenv(f"{prefix}ENABLE_RISK_ALERTS"),
                default=True,
            ),
            enable_system_alerts=_to_bool(
                os.getenv(f"{prefix}ENABLE_SYSTEM_ALERTS"),
                default=True,
            ),
            filter_non_actionable_liquidity_alerts=_to_bool(
                os.getenv(f"{prefix}FILTER_NON_ACTIONABLE_LIQUIDITY_ALERTS"),
                default=True,
            ),
            liquidity_min_signal_confidence_to_alert=_to_float(
                os.getenv(f"{prefix}LIQUIDITY_MIN_SIGNAL_CONFIDENCE_TO_ALERT"),
                default=0.20,
            ),
            liquidity_min_signal_score_to_alert=_to_float(
                os.getenv(f"{prefix}LIQUIDITY_MIN_SIGNAL_SCORE_TO_ALERT"),
                default=0.20,
            ),
            liquidity_min_sweep_risk_to_alert=_to_float(
                os.getenv(f"{prefix}LIQUIDITY_MIN_SWEEP_RISK_TO_ALERT"),
                default=0.20,
            ),
            liquidity_min_magnet_score_to_alert=_to_float(
                os.getenv(f"{prefix}LIQUIDITY_MIN_MAGNET_SCORE_TO_ALERT"),
                default=0.20,
            ),
            liquidity_allow_neutral_bias_alerts=_to_bool(
                os.getenv(f"{prefix}LIQUIDITY_ALLOW_NEUTRAL_BIAS_ALERTS"),
                default=False,
            ),
            retry=TelegramRetryConfig(
                max_retries=_to_int(
                    os.getenv(f"{prefix}MAX_RETRIES"),
                    default=3,
                ),
                retry_delay_sec=_to_float(
                    os.getenv(f"{prefix}RETRY_DELAY_SEC"),
                    default=1.0,
                ),
                retry_backoff_multiplier=_to_float(
                    os.getenv(f"{prefix}RETRY_BACKOFF_MULTIPLIER"),
                    default=2.0,
                ),
                max_retry_delay_sec=_to_float(
                    os.getenv(f"{prefix}MAX_RETRY_DELAY_SEC"),
                    default=30.0,
                ),
                retry_on_rate_limit=_to_bool(
                    os.getenv(f"{prefix}RETRY_ON_RATE_LIMIT"),
                    default=True,
                ),
                retry_on_timeout=_to_bool(
                    os.getenv(f"{prefix}RETRY_ON_TIMEOUT"),
                    default=True,
                ),
                retry_on_network_error=_to_bool(
                    os.getenv(f"{prefix}RETRY_ON_NETWORK_ERROR"),
                    default=True,
                ),
                retry_on_5xx=_to_bool(
                    os.getenv(f"{prefix}RETRY_ON_5XX"),
                    default=True,
                ),
            ),
            rate_limit=TelegramRateLimitConfig(
                enabled=_to_bool(
                    os.getenv(f"{prefix}RATE_LIMIT_ENABLED"),
                    default=True,
                ),
                max_messages_per_second=_to_float(
                    os.getenv(f"{prefix}MAX_MESSAGES_PER_SECOND"),
                    default=20.0,
                ),
                max_messages_per_topic_per_minute=_to_int(
                    os.getenv(f"{prefix}MAX_MESSAGES_PER_TOPIC_PER_MINUTE"),
                    default=60,
                ),
                drop_when_limited=_to_bool(
                    os.getenv(f"{prefix}DROP_WHEN_LIMITED"),
                    default=False,
                ),
                min_interval_per_topic_sec=_to_float(
                    os.getenv(f"{prefix}MIN_INTERVAL_PER_TOPIC_SEC"),
                    default=0.25,
                ),
            ),
            queue=TelegramQueueConfig(
                enabled=_to_bool(
                    os.getenv(f"{prefix}QUEUE_ENABLED"),
                    default=True,
                ),
                max_size=_to_int(
                    os.getenv(f"{prefix}QUEUE_MAX_SIZE"),
                    default=1000,
                ),
                worker_count=_to_int(
                    os.getenv(f"{prefix}QUEUE_WORKER_COUNT"),
                    default=1,
                ),
                enqueue_timeout_sec=_to_float(
                    os.getenv(f"{prefix}QUEUE_ENQUEUE_TIMEOUT_SEC"),
                    default=0.05,
                ),
                shutdown_timeout_sec=_to_float(
                    os.getenv(f"{prefix}QUEUE_SHUTDOWN_TIMEOUT_SEC"),
                    default=10.0,
                ),
                drain_on_stop=_to_bool(
                    os.getenv(f"{prefix}QUEUE_DRAIN_ON_STOP"),
                    default=True,
                ),
                full_policy=os.getenv(f"{prefix}QUEUE_FULL_POLICY", "drop_oldest"),
            ),
            topic_ids=cls.default_topic_ids(),
        )

        config.topics = config.build_topic_configs()
        config.validate()
        return config

    def build_topic_configs(self) -> dict[TelegramTopic, TelegramTopicConfig]:
        """
        Створює TelegramTopicConfig для всіх відомих topic_ids.
        """

        return {
            topic: TelegramTopicConfig(
                topic=topic,
                thread_id=thread_id if thread_id and thread_id > 0 else None,
                enabled=bool(thread_id and thread_id > 0),
                title=topic.value,
            )
            for topic, thread_id in self.topic_ids.items()
        }

    def validate(self) -> None:
        """
        Валідує config.

        Якщо bot disabled, не вимагаємо token/chat_id/topic_ids,
        щоб можна було запускати систему без Telegram notification layer.
        """

        if not self.enabled:
            return

        if not self.bot_token:
            raise TelegramConfigError("Telegram bot_token is required.")

        if self.chat_id is None or self.chat_id == "":
            raise TelegramConfigError("Telegram chat_id is required.")

        if self.request_timeout_sec <= 0:
            raise TelegramConfigError(
                "Telegram request_timeout_sec must be > 0.",
                details={"request_timeout_sec": self.request_timeout_sec},
            )

        if self.connect_timeout_sec <= 0:
            raise TelegramConfigError(
                "Telegram connect_timeout_sec must be > 0.",
                details={"connect_timeout_sec": self.connect_timeout_sec},
            )

        if self.connect_timeout_sec > self.request_timeout_sec:
            raise TelegramConfigError(
                "Telegram connect_timeout_sec must be <= request_timeout_sec.",
                details={
                    "connect_timeout_sec": self.connect_timeout_sec,
                    "request_timeout_sec": self.request_timeout_sec,
                },
            )

        if self.max_message_length <= 0 or self.max_message_length > 4096:
            raise TelegramConfigError(
                "Telegram max_message_length must be in range 1..4096.",
                details={"max_message_length": self.max_message_length},
            )

        if self.healthcheck_interval_sec <= 0:
            raise TelegramConfigError(
                "Telegram healthcheck_interval_sec must be > 0.",
                details={"healthcheck_interval_sec": self.healthcheck_interval_sec},
            )

        if not isinstance(self.parse_mode, TelegramParseMode):
            raise TelegramConfigError(
                "Telegram parse_mode is invalid.",
                details={"parse_mode": str(self.parse_mode)},
            )

        if not isinstance(self.route_policy, TelegramRoutePolicy):
            raise TelegramConfigError(
                "Telegram route_policy is invalid.",
                details={"route_policy": str(self.route_policy)},
            )

        threshold_values = {
            "liquidity_min_signal_confidence_to_alert": self.liquidity_min_signal_confidence_to_alert,
            "liquidity_min_signal_score_to_alert": self.liquidity_min_signal_score_to_alert,
            "liquidity_min_sweep_risk_to_alert": self.liquidity_min_sweep_risk_to_alert,
            "liquidity_min_magnet_score_to_alert": self.liquidity_min_magnet_score_to_alert,
        }
        for name, value in threshold_values.items():
            if value < 0:
                raise TelegramConfigError(
                    "Telegram liquidity alert thresholds must be >= 0.",
                    details={name: value},
                )

        self.retry.validate()
        self.rate_limit.validate()
        self.queue.validate()

        for topic, thread_id in self.topic_ids.items():
            if not isinstance(topic, TelegramTopic):
                raise TelegramConfigError(
                    "Telegram topic_ids contains invalid topic key.",
                    details={"topic": str(topic)},
                )

            if thread_id is not None and thread_id < 0:
                raise TelegramConfigError(
                    "Telegram topic thread_id must be >= 0.",
                    details={"topic": topic.value, "thread_id": thread_id},
                )

        for topic_config in self.topics.values():
            topic_config.validate()

        self._validate_required_topics()

    def get_topic_id(self, topic: TelegramTopic) -> int | None:
        """
        Повертає message_thread_id для TelegramTopic.
        """

        thread_id = self.topic_ids.get(topic)
        if thread_id and thread_id > 0:
            return thread_id

        topic_config = self.topics.get(topic)
        if topic_config and topic_config.enabled and topic_config.thread_id:
            return topic_config.thread_id

        return None

    def is_topic_enabled(self, topic: TelegramTopic) -> bool:
        """
        Перевіряє, чи topic має валідний message_thread_id.
        """

        return self.get_topic_id(topic) is not None

    def to_safe_dict(self) -> dict[str, Any]:
        """
        Safe representation для stats/debug без bot_token.
        """

        return {
            "enabled": self.enabled,
            "chat_id": self.chat_id,
            "default_topic_id": self.default_topic_id,
            "route_policy": self.route_policy.value,
            "parse_mode": self.parse_mode.value,
            "disable_web_page_preview": self.disable_web_page_preview,
            "disable_notification": self.disable_notification,
            "protect_content": self.protect_content,
            "api_base_url": self.api_base_url,
            "request_timeout_sec": self.request_timeout_sec,
            "connect_timeout_sec": self.connect_timeout_sec,
            "max_message_length": self.max_message_length,
            "split_long_messages": self.split_long_messages,
            "healthcheck_interval_sec": self.healthcheck_interval_sec,
            "emit_lifecycle_events": self.emit_lifecycle_events,
            "enable_analytics_alerts": self.enable_analytics_alerts,
            "enable_news_alerts": self.enable_news_alerts,
            "enable_signal_alerts": self.enable_signal_alerts,
            "enable_trade_updates": self.enable_trade_updates,
            "enable_risk_alerts": self.enable_risk_alerts,
            "enable_system_alerts": self.enable_system_alerts,
            "filter_non_actionable_liquidity_alerts": self.filter_non_actionable_liquidity_alerts,
            "liquidity_min_signal_confidence_to_alert": self.liquidity_min_signal_confidence_to_alert,
            "liquidity_min_signal_score_to_alert": self.liquidity_min_signal_score_to_alert,
            "liquidity_min_sweep_risk_to_alert": self.liquidity_min_sweep_risk_to_alert,
            "liquidity_min_magnet_score_to_alert": self.liquidity_min_magnet_score_to_alert,
            "liquidity_allow_neutral_bias_alerts": self.liquidity_allow_neutral_bias_alerts,
            "topic_ids": {
                topic.value: thread_id
                for topic, thread_id in self.topic_ids.items()
            },
            "retry": {
                "max_retries": self.retry.max_retries,
                "retry_delay_sec": self.retry.retry_delay_sec,
                "retry_backoff_multiplier": self.retry.retry_backoff_multiplier,
                "max_retry_delay_sec": self.retry.max_retry_delay_sec,
                "retry_on_rate_limit": self.retry.retry_on_rate_limit,
                "retry_on_timeout": self.retry.retry_on_timeout,
                "retry_on_network_error": self.retry.retry_on_network_error,
                "retry_on_5xx": self.retry.retry_on_5xx,
            },
            "rate_limit": {
                "enabled": self.rate_limit.enabled,
                "max_messages_per_second": self.rate_limit.max_messages_per_second,
                "max_messages_per_topic_per_minute": (
                    self.rate_limit.max_messages_per_topic_per_minute
                ),
                "drop_when_limited": self.rate_limit.drop_when_limited,
                "min_interval_per_topic_sec": (
                    self.rate_limit.min_interval_per_topic_sec
                ),
            },
            "queue": {
                "enabled": self.queue.enabled,
                "max_size": self.queue.max_size,
                "worker_count": self.queue.worker_count,
                "enqueue_timeout_sec": self.queue.enqueue_timeout_sec,
                "shutdown_timeout_sec": self.queue.shutdown_timeout_sec,
                "drain_on_stop": self.queue.drain_on_stop,
                "full_policy": self.queue.full_policy,
            },
        }

    def _validate_required_topics(self) -> None:
        """
        Перевіряє мінімальний набір topic-гілок для поточно увімкнених features.
        """

        required: list[TelegramTopic] = []

        if self.enable_news_alerts:
            required.append(TelegramTopic.NEWS)

        if self.enable_trade_updates:
            required.extend(
                [
                    TelegramTopic.OPEN_TRADES,
                    TelegramTopic.CLOSED_TRADES,
                ]
            )

        missing = [topic.value for topic in required if not self.is_topic_enabled(topic)]

        if missing:
            raise TelegramConfigError(
                "Telegram required topic ids are missing.",
                details={"missing_topics": missing},
            )


def _to_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise TelegramConfigError(
        "Invalid boolean value.",
        details={"value": value},
    )


def _to_int(value: str | None, *, default: int = 0) -> int:
    if value is None or value.strip() == "":
        return default

    try:
        return int(value.strip())
    except ValueError as exc:
        raise TelegramConfigError(
            "Invalid integer value.",
            details={"value": value},
            cause=exc,
        ) from exc


def _to_optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None

    return _to_int(value, default=0)


def _to_float(value: str | None, *, default: float = 0.0) -> float:
    if value is None or value.strip() == "":
        return default

    try:
        return float(value.strip())
    except ValueError as exc:
        raise TelegramConfigError(
            "Invalid float value.",
            details={"value": value},
            cause=exc,
        ) from exc


def _to_optional_str(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def _to_chat_id(value: str | None) -> int | str | None:
    if value is None or value.strip() == "":
        return None

    stripped = value.strip()

    try:
        return int(stripped)
    except ValueError:
        return stripped


def _to_parse_mode(
    value: str | None,
    *,
    default: TelegramParseMode,
) -> TelegramParseMode:
    if value is None or value.strip() == "":
        return default

    normalized = value.strip()

    for mode in TelegramParseMode:
        if normalized == mode.value or normalized.upper() == mode.value.upper():
            return mode

    raise TelegramConfigError(
        "Invalid Telegram parse_mode.",
        details={
            "value": value,
            "allowed": [mode.value for mode in TelegramParseMode],
        },
    )


def _to_route_policy(
    value: str | None,
    *,
    default: TelegramRoutePolicy,
) -> TelegramRoutePolicy:
    if value is None or value.strip() == "":
        return default

    normalized = value.strip().lower()

    for policy in TelegramRoutePolicy:
        if normalized == policy.value:
            return policy

    raise TelegramConfigError(
        "Invalid Telegram route_policy.",
        details={
            "value": value,
            "allowed": [policy.value for policy in TelegramRoutePolicy],
        },
    )