from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .enums import NewsFailureReason, NewsProcessingStage


@dataclass(slots=True, frozen=True)
class NewsErrorContext:
    """
    Structured context attached to AI/news exceptions.

    This keeps error handling production-friendly without logging sensitive
    values such as API keys, tokens, or raw provider credentials.
    """

    stage: NewsProcessingStage | None = None
    reason: NewsFailureReason = NewsFailureReason.UNKNOWN
    source_name: str | None = None
    source_type: str | None = None
    url: str | None = None
    news_id: str | None = None
    symbol: str | None = None
    details: dict[str, Any] | None = None

    def safe_details(self) -> dict[str, Any]:
        """
        Return details safe enough for logs/events.

        The centralized core logger should still handle redaction, but this
        method avoids accidentally exposing common sensitive fields.
        """

        if not self.details:
            return {}

        sensitive_keys = {
            "api_key",
            "apikey",
            "token",
            "access_token",
            "refresh_token",
            "secret",
            "password",
            "authorization",
            "cookie",
            "set_cookie",
            "x_api_key",
            "x-api-key",
        }

        sanitized: dict[str, Any] = {}
        for key, value in self.details.items():
            normalized_key = str(key).lower().replace("-", "_")
            if normalized_key in sensitive_keys:
                sanitized[str(key)] = "***REDACTED***"
            else:
                sanitized[str(key)] = value

        return sanitized

    def as_dict(self) -> dict[str, Any]:
        """
        Convert context to a serializable dict for EventBus payloads/logs.
        """

        return {
            "stage": str(self.stage) if self.stage else None,
            "reason": str(self.reason),
            "source_name": self.source_name,
            "source_type": self.source_type,
            "url": self.url,
            "news_id": self.news_id,
            "symbol": self.symbol,
            "details": self.safe_details(),
        }


class NewsAIError(Exception):
    """
    Base exception for the AI/news package.

    All custom exceptions in the package should inherit from this class so the
    service layer can catch package-level failures consistently.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or NewsErrorContext()
        self.cause = cause

    def __str__(self) -> str:
        context = self.context.as_dict()
        compact_context = {
            key: value
            for key, value in context.items()
            if value not in (None, {}, "")
        }

        if not compact_context:
            return self.message

        return f"{self.message} | context={compact_context}"

    def as_event_payload(self) -> dict[str, Any]:
        """
        Convert exception to a safe EventBus payload.
        """

        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context.as_dict(),
            "cause_type": self.cause.__class__.__name__ if self.cause else None,
        }


class NewsConfigError(NewsAIError):
    """
    Raised when AI/news configuration is invalid.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        context = context or NewsErrorContext(
            stage=NewsProcessingStage.COLLECT,
            reason=NewsFailureReason.INVALID_CONFIG,
        )
        super().__init__(message, context=context, cause=cause)


class NewsSourceError(NewsAIError):
    """
    Base exception for source adapter errors.
    """


class NewsFetchError(NewsSourceError):
    """
    Raised when a source fails to fetch raw news items.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        context = context or NewsErrorContext(
            stage=NewsProcessingStage.FETCH,
            reason=NewsFailureReason.NETWORK_ERROR,
        )
        super().__init__(message, context=context, cause=cause)


class NewsRateLimitError(NewsFetchError):
    """
    Raised when a news source returns a rate-limit response.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        context = context or NewsErrorContext(
            stage=NewsProcessingStage.FETCH,
            reason=NewsFailureReason.RATE_LIMITED,
        )
        super().__init__(message, context=context, cause=cause)


class NewsTimeoutError(NewsFetchError):
    """
    Raised when a news source request times out.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        context = context or NewsErrorContext(
            stage=NewsProcessingStage.FETCH,
            reason=NewsFailureReason.TIMEOUT,
        )
        super().__init__(message, context=context, cause=cause)


class NewsInvalidResponseError(NewsFetchError):
    """
    Raised when a source returns malformed or unsupported data.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        context = context or NewsErrorContext(
            stage=NewsProcessingStage.FETCH,
            reason=NewsFailureReason.INVALID_RESPONSE,
        )
        super().__init__(message, context=context, cause=cause)


class NewsProcessingError(NewsAIError):
    """
    Raised when raw news normalization or processing fails.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        context = context or NewsErrorContext(
            stage=NewsProcessingStage.PROCESS,
            reason=NewsFailureReason.PARSE_ERROR,
        )
        super().__init__(message, context=context, cause=cause)


class NewsValidationError(NewsProcessingError):
    """
    Raised when a news item does not satisfy required model constraints.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        context = context or NewsErrorContext(
            stage=NewsProcessingStage.PROCESS,
            reason=NewsFailureReason.VALIDATION_ERROR,
        )
        super().__init__(message, context=context, cause=cause)


class NewsDeduplicationError(NewsAIError):
    """
    Raised when deduplication state or logic fails unexpectedly.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        context = context or NewsErrorContext(
            stage=NewsProcessingStage.DEDUPLICATE,
            reason=NewsFailureReason.UNKNOWN,
        )
        super().__init__(message, context=context, cause=cause)


class NewsScoringError(NewsAIError):
    """
    Raised when rule-based or hybrid news scoring fails.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        context = context or NewsErrorContext(
            stage=NewsProcessingStage.SCORE,
            reason=NewsFailureReason.UNKNOWN,
        )
        super().__init__(message, context=context, cause=cause)


class NewsFeatureExtractionError(NewsAIError):
    """
    Raised when feature extraction fails.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        context = context or NewsErrorContext(
            stage=NewsProcessingStage.EXTRACT_FEATURES,
            reason=NewsFailureReason.UNKNOWN,
        )
        super().__init__(message, context=context, cause=cause)


class NewsLLMError(NewsAIError):
    """
    Base exception for optional LLM-based news analysis.
    """


class NewsLLMUnavailableError(NewsLLMError):
    """
    Raised when LLM scoring is enabled but the provider is unavailable.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        context = context or NewsErrorContext(
            stage=NewsProcessingStage.LLM_ANALYZE,
            reason=NewsFailureReason.LLM_UNAVAILABLE,
        )
        super().__init__(message, context=context, cause=cause)


class NewsLLMInvalidOutputError(NewsLLMError):
    """
    Raised when LLM output cannot be parsed or validated.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        context = context or NewsErrorContext(
            stage=NewsProcessingStage.LLM_ANALYZE,
            reason=NewsFailureReason.LLM_INVALID_OUTPUT,
        )
        super().__init__(message, context=context, cause=cause)


class NewsPublishError(NewsAIError):
    """
    Raised when publishing news events fails.
    """

    def __init__(
        self,
        message: str,
        *,
        context: NewsErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        context = context or NewsErrorContext(
            stage=NewsProcessingStage.PUBLISH,
            reason=NewsFailureReason.UNKNOWN,
        )
        super().__init__(message, context=context, cause=cause)


__all__ = [
    "NewsErrorContext",
    "NewsAIError",
    "NewsConfigError",
    "NewsSourceError",
    "NewsFetchError",
    "NewsRateLimitError",
    "NewsTimeoutError",
    "NewsInvalidResponseError",
    "NewsProcessingError",
    "NewsValidationError",
    "NewsDeduplicationError",
    "NewsScoringError",
    "NewsFeatureExtractionError",
    "NewsLLMError",
    "NewsLLMUnavailableError",
    "NewsLLMInvalidOutputError",
    "NewsPublishError",
]