from __future__ import annotations

import asyncio
from time import monotonic
from typing import Any

import aiohttp

from core.logger import get_logger
from .config import TelegramBotConfig
from .exceptions import (
    TelegramAPIError,
    TelegramClientError,
    TelegramConfigError,
    TelegramNetworkError,
    TelegramRateLimitError,
    TelegramTimeoutError,
)
from .models import (
    TelegramHealthStatus,
    TelegramSendRequest,
    TelegramSendResult,
)


class TelegramBotClient:
    """
    Async Telegram Bot API client.

    Відповідальність:
    - sendMessage;
    - getMe healthcheck;
    - retry/backoff;
    - timeout handling;
    - Telegram API error parsing.

    Не відповідає за:
    - EventBus subscriptions;
    - routing;
    - formatting;
    - trading decisions.
    """

    def __init__(
        self,
        config: TelegramBotConfig,
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.config = config
        self._external_session = session is not None
        self._session: aiohttp.ClientSession | None = session
        self._closed: bool = False
        self._logger = get_logger(__name__)

        self._validate_config()

    async def __aenter__(self) -> TelegramBotClient:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def start(self) -> None:
        """
        Ініціалізує HTTP session, якщо вона не передана зовні.
        """

        if self._closed:
            raise TelegramClientError("TelegramBotClient is already closed.")

        if self._session is not None and not self._session.closed:
            return

        timeout = aiohttp.ClientTimeout(
            total=self.config.request_timeout_sec,
            connect=self.config.connect_timeout_sec,
        )

        self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        """
        Закриває HTTP session, якщо client створив її сам.
        """

        self._closed = True

        if self._session is None:
            return

        if not self._external_session and not self._session.closed:
            await self._session.close()

    async def send_message(
        self,
        request: TelegramSendRequest,
        *,
        max_length: int | None = None,
    ) -> TelegramSendResult:
        """
        Відправляє одне повідомлення через Telegram sendMessage.

        Не кидає exception для звичайних delivery failures.
        Повертає TelegramSendResult.
        """

        max_len = max_length or self.config.max_message_length

        try:
            request.validate(max_length=max_len)
        except Exception as exc:
            return TelegramSendResult.failed(
                error=f"Invalid Telegram send request: {exc}",
                chat_id=request.chat_id,
                message_thread_id=request.message_thread_id,
            )

        if not self.config.enabled:
            return TelegramSendResult.skipped(
                reason="Telegram bot is disabled.",
                chat_id=request.chat_id,
                message_thread_id=request.message_thread_id,
            )

        if self._closed:
            return TelegramSendResult.failed(
                error="TelegramBotClient is closed.",
                chat_id=request.chat_id,
                message_thread_id=request.message_thread_id,
            )

        await self.start()

        payload = request.to_api_payload()
        attempt = 1
        delay = self.config.retry.retry_delay_sec

        while True:
            try:
                response = await self._request_json(
                    method="sendMessage",
                    payload=payload,
                )

                result = response.get("result") if isinstance(response, dict) else None
                message_id = None

                if isinstance(result, dict):
                    raw_message_id = result.get("message_id")
                    if isinstance(raw_message_id, int):
                        message_id = raw_message_id

                return TelegramSendResult.sent(
                    message_id=message_id,
                    chat_id=request.chat_id,
                    message_thread_id=request.message_thread_id,
                    attempt=attempt,
                    raw_response=response,
                )

            except TelegramRateLimitError as exc:
                if not self._should_retry_rate_limit(attempt):
                    return TelegramSendResult.failed(
                        error=exc.message,
                        chat_id=request.chat_id,
                        message_thread_id=request.message_thread_id,
                        error_code=None,
                        retry_after_sec=exc.retry_after_sec,
                        attempt=attempt,
                    )

                sleep_for = exc.retry_after_sec or delay
                await self._sleep_before_retry(
                    sleep_for=sleep_for,
                    attempt=attempt,
                    reason="telegram rate limit",
                )

            except TelegramTimeoutError as exc:
                if not self._should_retry_timeout(attempt):
                    return TelegramSendResult.failed(
                        error=exc.message,
                        chat_id=request.chat_id,
                        message_thread_id=request.message_thread_id,
                        attempt=attempt,
                    )

                await self._sleep_before_retry(
                    sleep_for=delay,
                    attempt=attempt,
                    reason="telegram timeout",
                )

            except TelegramNetworkError as exc:
                if not self._should_retry_network(attempt):
                    return TelegramSendResult.failed(
                        error=exc.message,
                        chat_id=request.chat_id,
                        message_thread_id=request.message_thread_id,
                        attempt=attempt,
                    )

                await self._sleep_before_retry(
                    sleep_for=delay,
                    attempt=attempt,
                    reason="telegram network error",
                )

            except TelegramAPIError as exc:
                if not self._should_retry_api_error(exc, attempt):
                    return TelegramSendResult.failed(
                        error=exc.description or exc.message,
                        chat_id=request.chat_id,
                        message_thread_id=request.message_thread_id,
                        error_code=exc.error_code,
                        attempt=attempt,
                        raw_response=exc.details,
                    )

                await self._sleep_before_retry(
                    sleep_for=delay,
                    attempt=attempt,
                    reason="telegram api error",
                )

            except TelegramClientError as exc:
                return TelegramSendResult.failed(
                    error=exc.message,
                    chat_id=request.chat_id,
                    message_thread_id=request.message_thread_id,
                    attempt=attempt,
                )

            except Exception as exc:
                self._logger.exception(
                    "Unexpected Telegram client error while sending message.",
                    extra={
                        "chat_id": request.chat_id,
                        "message_thread_id": request.message_thread_id,
                        "attempt": attempt,
                    },
                )
                return TelegramSendResult.failed(
                    error=f"Unexpected Telegram client error: {exc}",
                    chat_id=request.chat_id,
                    message_thread_id=request.message_thread_id,
                    attempt=attempt,
                )

            attempt += 1
            delay = self._next_delay(delay)

    async def send_messages(
        self,
        requests: list[TelegramSendRequest],
        *,
        max_length: int | None = None,
    ) -> list[TelegramSendResult]:
        """
        Послідовно відправляє список повідомлень.

        Навмисно не робимо gather(), щоб не створювати burst у Telegram API.
        Rate-limit на рівні service/state буде додатково контролювати частоту.
        """

        results: list[TelegramSendResult] = []

        for request in requests:
            result = await self.send_message(request, max_length=max_length)
            results.append(result)

        return results

    async def health_check(self) -> TelegramHealthStatus:
        """
        Перевіряє доступність Telegram Bot API через getMe.
        """

        if not self.config.enabled:
            return TelegramHealthStatus(
                ok=False,
                status="disabled",
                error="Telegram bot is disabled.",
            )

        if self._closed:
            return TelegramHealthStatus(
                ok=False,
                status="closed",
                error="TelegramBotClient is closed.",
            )

        started = monotonic()

        try:
            await self.start()
            response = await self._request_json(method="getMe", payload=None)
            latency_ms = (monotonic() - started) * 1000

            result = response.get("result") if isinstance(response, dict) else None
            username = None

            if isinstance(result, dict):
                raw_username = result.get("username")
                if raw_username is not None:
                    username = str(raw_username)

            return TelegramHealthStatus(
                ok=True,
                status="ok",
                latency_ms=latency_ms,
                bot_username=username,
                details={
                    "can_join_groups": result.get("can_join_groups")
                    if isinstance(result, dict)
                    else None,
                    "can_read_all_group_messages": result.get(
                        "can_read_all_group_messages"
                    )
                    if isinstance(result, dict)
                    else None,
                    "supports_inline_queries": result.get("supports_inline_queries")
                    if isinstance(result, dict)
                    else None,
                },
            )

        except TelegramClientError as exc:
            return TelegramHealthStatus(
                ok=False,
                status="error",
                latency_ms=(monotonic() - started) * 1000,
                error=exc.message,
                details=exc.details,
            )

        except Exception as exc:
            self._logger.exception("Unexpected Telegram healthcheck error.")
            return TelegramHealthStatus(
                ok=False,
                status="error",
                latency_ms=(monotonic() - started) * 1000,
                error=f"Unexpected Telegram healthcheck error: {exc}",
            )

    async def _request_json(
        self,
        *,
        method: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Виконує Telegram Bot API request і повертає JSON.

        Не логить URL, бо URL містить bot_token.
        """

        session = self._require_session()
        url = self._method_url(method)

        try:
            if payload is None:
                async with session.get(url) as response:
                    return await self._parse_response(response=response)

            async with session.post(url, json=payload) as response:
                return await self._parse_response(response=response)

        except asyncio.TimeoutError as exc:
            raise TelegramTimeoutError(
                "Telegram API request timeout.",
                details={"method": method},
                cause=exc,
            ) from exc

        except aiohttp.ClientError as exc:
            raise TelegramNetworkError(
                "Telegram API network error.",
                details={"method": method},
                cause=exc,
            ) from exc

    async def _parse_response(
        self,
        *,
        response: aiohttp.ClientResponse,
    ) -> dict[str, Any]:
        status_code = response.status

        try:
            data = await response.json(content_type=None)
        except Exception as exc:
            text = await response.text()
            raise TelegramAPIError(
                "Telegram API returned non-JSON response.",
                status_code=status_code,
                description=text[:500],
                cause=exc,
            ) from exc

        if not isinstance(data, dict):
            raise TelegramAPIError(
                "Telegram API returned invalid JSON structure.",
                status_code=status_code,
                details={"response_type": type(data).__name__},
            )

        ok = data.get("ok")

        if status_code == 429:
            retry_after = self._extract_retry_after(data)
            raise TelegramRateLimitError(
                "Telegram API rate limit exceeded.",
                retry_after_sec=retry_after,
                details={
                    "status_code": status_code,
                    "error_code": data.get("error_code"),
                    "description": data.get("description"),
                },
            )

        if status_code >= 500:
            raise TelegramAPIError(
                "Telegram API server error.",
                status_code=status_code,
                error_code=self._as_int(data.get("error_code")),
                description=self._as_str(data.get("description")),
                parameters=self._as_dict(data.get("parameters")),
                details=data,
            )

        if status_code >= 400:
            raise TelegramAPIError(
                "Telegram API request failed.",
                status_code=status_code,
                error_code=self._as_int(data.get("error_code")),
                description=self._as_str(data.get("description")),
                parameters=self._as_dict(data.get("parameters")),
                details=data,
            )

        if ok is not True:
            parameters = self._as_dict(data.get("parameters"))
            retry_after = self._extract_retry_after(data)

            if retry_after is not None:
                raise TelegramRateLimitError(
                    "Telegram API rate limit exceeded.",
                    retry_after_sec=retry_after,
                    details=data,
                )

            raise TelegramAPIError(
                "Telegram API returned ok=false.",
                status_code=status_code,
                error_code=self._as_int(data.get("error_code")),
                description=self._as_str(data.get("description")),
                parameters=parameters,
                details=data,
            )

        return data

    def _method_url(self, method: str) -> str:
        token = self.config.bot_token
        if not token:
            raise TelegramConfigError("Telegram bot_token is required.")

        base_url = self.config.api_base_url.rstrip("/")
        clean_method = method.strip().lstrip("/")
        return f"{base_url}/bot{token}/{clean_method}"

    def _require_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            raise TelegramClientError("Telegram HTTP session is not initialized.")

        return self._session

    def _validate_config(self) -> None:
        if not self.config.enabled:
            return

        if not self.config.bot_token:
            raise TelegramConfigError("Telegram bot_token is required.")

        if self.config.request_timeout_sec <= 0:
            raise TelegramConfigError(
                "Telegram request_timeout_sec must be > 0.",
                details={"request_timeout_sec": self.config.request_timeout_sec},
            )

        if self.config.connect_timeout_sec <= 0:
            raise TelegramConfigError(
                "Telegram connect_timeout_sec must be > 0.",
                details={"connect_timeout_sec": self.config.connect_timeout_sec},
            )

        if self.config.max_message_length <= 0:
            raise TelegramConfigError(
                "Telegram max_message_length must be > 0.",
                details={"max_message_length": self.config.max_message_length},
            )

    def _should_retry_rate_limit(self, attempt: int) -> bool:
        return (
            self.config.retry.retry_on_rate_limit
            and attempt <= self.config.retry.max_retries
        )

    def _should_retry_timeout(self, attempt: int) -> bool:
        return (
            self.config.retry.retry_on_timeout
            and attempt <= self.config.retry.max_retries
        )

    def _should_retry_network(self, attempt: int) -> bool:
        return (
            self.config.retry.retry_on_network_error
            and attempt <= self.config.retry.max_retries
        )

    def _should_retry_api_error(
        self,
        exc: TelegramAPIError,
        attempt: int,
    ) -> bool:
        if not self.config.retry.retry_on_5xx:
            return False

        if attempt > self.config.retry.max_retries:
            return False

        status_code = exc.status_code
        return status_code is not None and status_code >= 500

    async def _sleep_before_retry(
        self,
        *,
        sleep_for: float,
        attempt: int,
        reason: str,
    ) -> None:
        safe_sleep_for = max(0.0, min(sleep_for, self.config.retry.max_retry_delay_sec))

        self._logger.warning(
            "Retrying Telegram API request.",
            extra={
                "attempt": attempt,
                "retry_after_sec": safe_sleep_for,
                "reason": reason,
            },
        )

        if safe_sleep_for > 0:
            await asyncio.sleep(safe_sleep_for)

    def _next_delay(self, current_delay: float) -> float:
        next_delay = current_delay * self.config.retry.retry_backoff_multiplier
        return min(next_delay, self.config.retry.max_retry_delay_sec)

    def _extract_retry_after(self, data: dict[str, Any]) -> float | None:
        parameters = data.get("parameters")

        if not isinstance(parameters, dict):
            return None

        retry_after = parameters.get("retry_after")
        if retry_after is None:
            return None

        try:
            return float(retry_after)
        except (TypeError, ValueError):
            return None

    def _as_int(self, value: Any) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _as_str(self, value: Any) -> str | None:
        if value is None:
            return None

        return str(value)

    def _as_dict(self, value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value

        return None

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def is_started(self) -> bool:
        return self._session is not None and not self._session.closed

    def stats(self) -> dict[str, Any]:
        """
        Safe client stats без bot_token.
        """

        return {
            "started": self.is_started,
            "closed": self.is_closed,
            "external_session": self._external_session,
            "api_base_url": self.config.api_base_url,
            "request_timeout_sec": self.config.request_timeout_sec,
            "connect_timeout_sec": self.config.connect_timeout_sec,
            "max_retries": self.config.retry.max_retries,
            "retry_delay_sec": self.config.retry.retry_delay_sec,
            "retry_backoff_multiplier": self.config.retry.retry_backoff_multiplier,
            "max_retry_delay_sec": self.config.retry.max_retry_delay_sec,
        }