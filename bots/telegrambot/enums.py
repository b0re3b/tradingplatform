"""
Telegram bot package enums.

Цей модуль містить тільки enum-контракти для Telegram notification layer.
Тут немає бізнес-логіки, EventBus-підписок або Telegram API викликів.
"""

from __future__ import annotations

from enum import Enum


class TelegramTopic(str, Enum):
    """
    Логічні Telegram-гілки/forum topics.

    Значення enum використовуються всередині пакету для routing:
    EventBus event -> TelegramTopic -> message_thread_id.
    """

    # Analytics topics
    ORDERFLOW = "orderflow"
    LIQUIDITY = "liquidity"
    PRICE_ACTION = "price_action"
    LIQUIDATIONS = "liquidations"
    WHALES = "whales"
    SPOOFING = "spoofing"
    SPREADS = "spreads"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"

    # AI / news
    NEWS = "news"

    # Trading lifecycle
    SIGNALS = "signals"
    OPEN_TRADES = "open_trades"
    CLOSED_TRADES = "closed_trades"

    # Risk / system
    RISK = "risk"
    SYSTEM = "system"


class TelegramMessageType(str, Enum):
    """
    Тип повідомлення, яке буде форматуватись і відправлятись у Telegram.
    """

    ANALYTICS_ALERT = "analytics_alert"
    NEWS_ALERT = "news_alert"

    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_REJECTED = "signal_rejected"
    SIGNAL_CONFIRMED = "signal_confirmed"
    SIGNAL_UPDATED = "signal_updated"

    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"

    POSITION_OPENED = "position_opened"
    POSITION_UPDATED = "position_updated"
    POSITION_CLOSED = "position_closed"

    RISK_WARNING = "risk_warning"
    RISK_BLOCKED = "risk_blocked"
    RISK_KILL_SWITCH = "risk_kill_switch"

    SYSTEM_INFO = "system_info"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_ERROR = "system_error"
    HEALTHCHECK = "healthcheck"


class TelegramEventCategory(str, Enum):
    """
    Високорівнева категорія EventBus-події.

    Використовується router-ом, щоб грубо класифікувати event.name.
    """

    ANALYTICS = "analytics"
    NEWS = "news"
    AI = "ai"
    SIGNAL = "signal"
    RISK = "risk"
    EXECUTION = "execution"
    POSITION = "position"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class TelegramParseMode(str, Enum):
    """
    Telegram parse_mode.

    Для production краще використовувати HTML, бо його простіше безпечно
    екранувати перед відправкою.
    """

    HTML = "HTML"
    MARKDOWN = "Markdown"
    MARKDOWN_V2 = "MarkdownV2"
    PLAIN = "plain"


class TelegramBotStatus(str, Enum):
    """
    Runtime status TelegramBotService.
    """

    CREATED = "created"
    REGISTERED = "registered"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DISABLED = "disabled"
    ERROR = "error"


class TelegramDeliveryStatus(str, Enum):
    """
    Результат доставки повідомлення в Telegram.
    """

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    SKIPPED = "skipped"
    RATE_LIMITED = "rate_limited"
    RETRYING = "retrying"


class TelegramPriority(str, Enum):
    """
    Пріоритет Telegram-повідомлення.

    Це локальний пріоритет notification layer, не заміна core.EventPriority.
    EventBus все одно використовує core.event_bus.EventPriority.
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TelegramNotificationLevel(str, Enum):
    """
    Візуальний/смисловий рівень повідомлення.
    """

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TelegramTradeResult(str, Enum):
    """
    Результат закритої угоди для форматування у CLOSED_TRADES topic.
    """

    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class TelegramRoutePolicy(str, Enum):
    """
    Поведінка router-а, якщо для події не знайдено точну гілку.
    """

    SEND_TO_SYSTEM = "send_to_system"
    SEND_TO_DEFAULT = "send_to_default"
    SKIP = "skip"
    RAISE = "raise"


class TelegramRateLimitScope(str, Enum):
    """
    Scope для rate-limit контролю.
    """

    GLOBAL = "global"
    BY_TOPIC = "by_topic"
    BY_MESSAGE_TYPE = "by_message_type"