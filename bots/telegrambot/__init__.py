"""
Telegram bot notification package.

Пакет відповідає тільки за notification/presentation layer:
- слухає EventBus-події;
- маршрутизує їх у Telegram forum topics;
- форматує повідомлення;
- відправляє їх через Telegram Bot API.

Пакет не містить торгової логіки й не викликає напряму:
- analytics;
- strategy;
- risk;
- execution;
- data;
- exchanges.

Основний entrypoint:
    TelegramBotService
"""

from .client import TelegramBotClient
from .config import (
    TelegramBotConfig,
    TelegramRateLimitConfig,
    TelegramRetryConfig,
    TelegramTopicConfig,
)
from .enums import (
    TelegramBotStatus,
    TelegramDeliveryStatus,
    TelegramEventCategory,
    TelegramMessageType,
    TelegramNotificationLevel,
    TelegramParseMode,
    TelegramPriority,
    TelegramRateLimitScope,
    TelegramRoutePolicy,
    TelegramTopic,
    TelegramTradeResult,
)
from .exceptions import (
    TelegramAPIError,
    TelegramBotError,
    TelegramClientError,
    TelegramConfigError,
    TelegramDependencyError,
    TelegramDisabledError,
    TelegramFormattingError,
    TelegramHandlerError,
    TelegramMessageTooLongError,
    TelegramNetworkError,
    TelegramPayloadError,
    TelegramRateLimitError,
    TelegramRoutingError,
    TelegramServiceError,
    TelegramStateError,
    TelegramTemplateError,
    TelegramTimeoutError,
    TelegramTopicNotConfiguredError,
)
from .formatter import TelegramFormatter
from .handlers import TelegramEventHandlers, TelegramHandlerResult
from .models import (
    TelegramDeliveryRecord,
    TelegramEventMetadata,
    TelegramEventPayload,
    TelegramFormattedMessage,
    TelegramHealthStatus,
    TelegramMessageChunk,
    TelegramRateLimitDecision,
    TelegramResolvedMessage,
    TelegramSendRequest,
    TelegramSendResult,
    TelegramTopicRoute,
)
from .router import TelegramRouter, TelegramRoutingRule
from .service import TelegramBotService
from .state import (
    TelegramBotState,
    TelegramDeliveryStats,
    TelegramRateLimitState,
    TelegramTopicState,
)


__all__ = [
    # Service / facade
    "TelegramBotService",

    # Client
    "TelegramBotClient",

    # Config
    "TelegramBotConfig",
    "TelegramRetryConfig",
    "TelegramRateLimitConfig",
    "TelegramTopicConfig",

    # Enums
    "TelegramTopic",
    "TelegramMessageType",
    "TelegramEventCategory",
    "TelegramParseMode",
    "TelegramBotStatus",
    "TelegramDeliveryStatus",
    "TelegramPriority",
    "TelegramNotificationLevel",
    "TelegramTradeResult",
    "TelegramRoutePolicy",
    "TelegramRateLimitScope",

    # Exceptions
    "TelegramBotError",
    "TelegramConfigError",
    "TelegramClientError",
    "TelegramAPIError",
    "TelegramNetworkError",
    "TelegramTimeoutError",
    "TelegramRateLimitError",
    "TelegramRoutingError",
    "TelegramTopicNotConfiguredError",
    "TelegramFormattingError",
    "TelegramTemplateError",
    "TelegramPayloadError",
    "TelegramMessageTooLongError",
    "TelegramHandlerError",
    "TelegramServiceError",
    "TelegramDisabledError",
    "TelegramStateError",
    "TelegramDependencyError",

    # Router / formatter / handlers
    "TelegramRouter",
    "TelegramRoutingRule",
    "TelegramFormatter",
    "TelegramEventHandlers",
    "TelegramHandlerResult",

    # Models
    "TelegramEventMetadata",
    "TelegramEventPayload",
    "TelegramTopicRoute",
    "TelegramFormattedMessage",
    "TelegramSendRequest",
    "TelegramSendResult",
    "TelegramDeliveryRecord",
    "TelegramRateLimitDecision",
    "TelegramHealthStatus",
    "TelegramMessageChunk",
    "TelegramResolvedMessage",

    # State
    "TelegramBotState",
    "TelegramTopicState",
    "TelegramDeliveryStats",
    "TelegramRateLimitState",
]