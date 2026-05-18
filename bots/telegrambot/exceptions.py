"""
Telegram bot package exceptions.

У цьому модулі зібрані всі доменні винятки notification layer.
Вони не залежать від Telegram API бібліотек напряму й можуть безпечно
використовуватись у config/router/formatter/client/service.
"""

from __future__ import annotations

from typing import Any


class TelegramBotError(Exception):
    """
    Базовий виняток пакету telegram_bot.

    Усі інші винятки цього пакету мають наслідуватися від нього,
    щоб верхній service/handler міг централізовано обробляти помилки.
    """

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        if not self.details:
            return self.message
        return f"{self.message} | details={self.details}"


class TelegramConfigError(TelegramBotError):
    """
    Помилка конфігурації Telegram-бота.

    Приклади:
    - відсутній bot_token;
    - відсутній chat_id;
    - topic id не заданий;
    - некоректний parse_mode;
    - невалідні retry/rate-limit параметри.
    """


class TelegramClientError(TelegramBotError):
    """
    Помилка Telegram HTTP/API client layer.

    Використовується для помилок запиту, відповіді Telegram API,
    timeout, network failure або некоректної відповіді.
    """


class TelegramAPIError(TelegramClientError):
    """
    Telegram API повернув ok=false або іншу помилкову відповідь.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: int | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        merged_details = details or {}
        if status_code is not None:
            merged_details["status_code"] = status_code
        if error_code is not None:
            merged_details["error_code"] = error_code
        if description is not None:
            merged_details["description"] = description
        if parameters is not None:
            merged_details["parameters"] = parameters

        super().__init__(message, details=merged_details, cause=cause)
        self.status_code = status_code
        self.error_code = error_code
        self.description = description
        self.parameters = parameters or {}


class TelegramNetworkError(TelegramClientError):
    """
    Network-level помилка при зверненні до Telegram API.

    Наприклад:
    - DNS/connect error;
    - connection reset;
    - SSL error;
    - недоступний Telegram API endpoint.
    """


class TelegramTimeoutError(TelegramClientError):
    """
    Timeout при зверненні до Telegram API.
    """


class TelegramRateLimitError(TelegramClientError):
    """
    Rate limit помилка.

    Може бути локальною помилкою notification layer або відповіддю Telegram API
    з retry_after.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_sec: float | None = None,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        merged_details = details or {}
        if retry_after_sec is not None:
            merged_details["retry_after_sec"] = retry_after_sec

        super().__init__(message, details=merged_details, cause=cause)
        self.retry_after_sec = retry_after_sec


class TelegramRoutingError(TelegramBotError):
    """
    Помилка routing layer.

    Наприклад:
    - EventBus event не вдалося зіставити з TelegramTopic;
    - route policy = RAISE;
    - некоректна routing rule.
    """


class TelegramTopicNotConfiguredError(TelegramRoutingError):
    """
    Для TelegramTopic не задано message_thread_id.

    Це окремий тип, бо такі помилки часто треба або відправляти в SYSTEM topic,
    або пропускати залежно від TelegramRoutePolicy.
    """

    def __init__(
        self,
        message: str,
        *,
        topic: str | None = None,
        event_name: str | None = None,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        merged_details = details or {}
        if topic is not None:
            merged_details["topic"] = topic
        if event_name is not None:
            merged_details["event_name"] = event_name

        super().__init__(message, details=merged_details, cause=cause)
        self.topic = topic
        self.event_name = event_name


class TelegramFormattingError(TelegramBotError):
    """
    Помилка formatter layer.

    Наприклад:
    - Event payload не містить потрібних полів;
    - шаблон не може бути заповнений;
    - текст повідомлення некоректний або порожній.
    """


class TelegramTemplateError(TelegramFormattingError):
    """
    Помилка шаблону повідомлення.

    Наприклад:
    - відсутній placeholder;
    - невалідний HTML/Markdown формат;
    - шаблон не відповідає типу повідомлення.
    """


class TelegramPayloadError(TelegramFormattingError):
    """
    Event payload має некоректну структуру для форматування.

    Наприклад:
    - немає symbol/exchange/side;
    - price не можна привести до числа;
    - closed trade не містить pnl.
    """


class TelegramMessageTooLongError(TelegramFormattingError):
    """
    Сформоване повідомлення перевищує допустиму довжину Telegram.
    """


class TelegramHandlerError(TelegramBotError):
    """
    Помилка EventBus handler layer.

    Використовується, коли handler не зміг обробити Event:
    routing/formatting/sending завершились помилкою.
    """


class TelegramServiceError(TelegramBotError):
    """
    Помилка lifecycle/facade layer.

    Наприклад:
    - start/stop/register failure;
    - healthcheck failure;
    - scheduler job setup failure.
    """


class TelegramDisabledError(TelegramServiceError):
    """
    Операцію неможливо виконати, бо TelegramBotService вимкнений у config.
    """


class TelegramStateError(TelegramServiceError):
    """
    Некоректний runtime state.

    Наприклад:
    - повторний start;
    - stop до start;
    - send до register/start, якщо це заборонено.
    """


class TelegramDependencyError(TelegramServiceError):
    """
    Відсутня або некоректна dependency.

    Наприклад:
    - event_bus не передано;
    - scheduler потрібен для healthcheck, але не переданий;
    - client/formatter/router не ініціалізовані.
    """