"""
Telegram bot package config.

Конфіг Telegram notification layer.

Правила:
- bot_token не hardcode-иться і не логиться;
- усі параметри typed;
- dataclass(slots=True);
- Telegram bot не містить торгової логіки;
- routing у topic-гілки задається через topic_ids;
- service/client/handlers отримують цей config через constructor dependency injection.
"""

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

    # Глобальний soft-limit для всього бота.
    max_messages_per_second: float = 20.0

    # Soft-limit на окрему topic-гілку.
    max_messages_per_topic_per_minute: int = 60

    # Якщо true — замість exception повідомлення можуть бути пропущені/відкладені.
    drop_when_limited: bool = False

    # Мінімальна пауза між повідомленнями в одну й ту саму гілку.
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

    # Sub-configs.
    retry: TelegramRetryConfig = field(default_factory=TelegramRetryConfig)
    rate_limit: TelegramRateLimitConfig = field(default_factory=TelegramRateLimitConfig)

    # Мапінг topic -> message_thread_id.
    topic_ids: dict[TelegramTopic, int] = field(default_factory=dict)

    # Optional topic metadata.
    topics: dict[TelegramTopic, TelegramTopicConfig] = field(default_factory=dict)

    @classmethod
    def default_topic_ids(cls) -> dict[TelegramTopic, int]:
        """
        Повертає порожній mapping з усіма очікуваними topic keys.

        Реальні message_thread_id треба задати з env/config після створення
        Telegram forum topics.
        """

        return {
            TelegramTopic.ORDERFLOW: 0,
            TelegramTopic.LIQUIDITY: 0,
            TelegramTopic.PRICE_ACTION: 0,
            TelegramTopic.LIQUIDATIONS: 0,
            TelegramTopic.WHALES: 0,
            TelegramTopic.SPOOFING: 0,
            TelegramTopic.SPREADS: 0,
            TelegramTopic.FUNDING: 0,
            TelegramTopic.OPEN_INTEREST: 0,
            TelegramTopic.NEWS: 0,
            TelegramTopic.SIGNALS: 0,
            TelegramTopic.OPEN_TRADES: 0,
            TelegramTopic.CLOSED_TRADES: 0,
            TelegramTopic.RISK: 0,
            TelegramTopic.SYSTEM: 0,
        }

    @classmethod
    def from_env(cls, prefix: str = "TELEGRAM_BOT_") -> TelegramBotConfig:
        """
        Створює TelegramBotConfig з environment variables.

        Приклади env:
        - TELEGRAM_BOT_ENABLED=true
        - TELEGRAM_BOT_TOKEN=...
        - TELEGRAM_BOT_CHAT_ID=-1001234567890
        - TELEGRAM_BOT_TOPIC_ORDERFLOW=111
        - TELEGRAM_BOT_TOPIC_NEWS=222
        - TELEGRAM_BOT_TOPIC_OPEN_TRADES=333
        - TELEGRAM_BOT_TOPIC_CLOSED_TRADES=444

        Token не валідується на формат тут, тільки на наявність.
        """

        topic_ids = cls._topic_ids_from_env(prefix=prefix)

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
            topic_ids=topic_ids,
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

        self.retry.validate()
        self.rate_limit.validate()

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
        }

    def _validate_required_topics(self) -> None:
        """
        Перевіряє мінімальний набір topic-гілок для поточно увімкнених features.

        Не вимагаємо всі analytics topics обовʼязково, бо можна поступово
        вмикати гілки. Але ключові гілки для trading lifecycle краще задати.
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

        if self.enable_risk_alerts:
            required.append(TelegramTopic.RISK)

        if self.enable_system_alerts and self.route_policy == TelegramRoutePolicy.SEND_TO_SYSTEM:
            required.append(TelegramTopic.SYSTEM)

        missing = [topic.value for topic in required if not self.is_topic_enabled(topic)]

        if missing:
            raise TelegramConfigError(
                "Telegram required topic ids are missing.",
                details={"missing_topics": missing},
            )

    @staticmethod
    def _topic_ids_from_env(prefix: str) -> dict[TelegramTopic, int]:
        topic_ids = TelegramBotConfig.default_topic_ids()

        env_mapping: dict[TelegramTopic, str] = {
            TelegramTopic.ORDERFLOW: "TOPIC_ORDERFLOW",
            TelegramTopic.LIQUIDITY: "TOPIC_LIQUIDITY",
            TelegramTopic.PRICE_ACTION: "TOPIC_PRICE_ACTION",
            TelegramTopic.LIQUIDATIONS: "TOPIC_LIQUIDATIONS",
            TelegramTopic.WHALES: "TOPIC_WHALES",
            TelegramTopic.SPOOFING: "TOPIC_SPOOFING",
            TelegramTopic.SPREADS: "TOPIC_SPREADS",
            TelegramTopic.FUNDING: "TOPIC_FUNDING",
            TelegramTopic.OPEN_INTEREST: "TOPIC_OPEN_INTEREST",
            TelegramTopic.NEWS: "TOPIC_NEWS",
            TelegramTopic.SIGNALS: "TOPIC_SIGNALS",
            TelegramTopic.OPEN_TRADES: "TOPIC_OPEN_TRADES",
            TelegramTopic.CLOSED_TRADES: "TOPIC_CLOSED_TRADES",
            TelegramTopic.RISK: "TOPIC_RISK",
            TelegramTopic.SYSTEM: "TOPIC_SYSTEM",
        }

        for topic, env_name in env_mapping.items():
            topic_ids[topic] = _to_int(
                os.getenv(f"{prefix}{env_name}"),
                default=0,
            )

        return topic_ids


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
    """
    Telegram chat_id може бути:
    - int, наприклад -1001234567890;
    - str username/channel id, якщо використовується @name.
    """

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