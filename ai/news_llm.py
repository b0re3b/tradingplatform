from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import aiohttp

from core.logger import get_logger

from .config import NewsLLMConfig
from .enums import (
    LLMOutputStatus,
    LLMProvider,
    NewsCategory,
    NewsFailureReason,
    NewsMarketBias,
    NewsProcessingStage,
    NewsSentiment,
    NewsTimeHorizon,
)
from .exceptions import (
    NewsErrorContext,
    NewsLLMError,
    NewsLLMInvalidOutputError,
    NewsLLMUnavailableError,
)
from .models import NewsFeatures, NewsLLMResult, NormalizedNewsItem


TEnum = TypeVar("TEnum", bound=Enum)

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)


@dataclass(slots=True, frozen=True)
class NewsLLMRequest:
    """
    Internal request object for LLM news analysis.
    """

    item: NormalizedNewsItem
    features: NewsFeatures


class NewsLLMClient:
    """
    Optional LLM client for structured news analysis.

    The rest of the AI/news package must work without this component. If LLM is
    disabled or unavailable, analyze() returns a non-success NewsLLMResult
    instead of breaking the full news pipeline.
    """

    def __init__(self, config: NewsLLMConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    async def analyze(
        self,
        item: NormalizedNewsItem,
        features: NewsFeatures,
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> NewsLLMResult:
        """
        Analyze a normalized news item with optional LLM support.
        """

        if not self.config.enabled:
            return NewsLLMResult(
                status=LLMOutputStatus.DISABLED,
                provider=self.config.provider,
                model=self.config.model,
                error="LLM analysis is disabled",
            )

        if self.config.provider == LLMProvider.DISABLED:
            return NewsLLMResult(
                status=LLMOutputStatus.DISABLED,
                provider=self.config.provider,
                model=self.config.model,
                error="LLM provider is disabled",
            )

        try:
            request = NewsLLMRequest(item=item, features=features)
            prompt_payload = self._build_prompt_payload(request)

            if self.config.provider == LLMProvider.OPENAI:
                return await self._analyze_openai_compatible(
                    prompt_payload,
                    session=session,
                )

            if self.config.provider == LLMProvider.LOCAL:
                return await self._analyze_openai_compatible(
                    prompt_payload,
                    session=session,
                )

            raise NewsLLMUnavailableError(
                f"Unsupported LLM provider: {self.config.provider}",
                context=NewsErrorContext(
                    stage=NewsProcessingStage.LLM_ANALYZE,
                    reason=NewsFailureReason.LLM_UNAVAILABLE,
                    news_id=item.news_id,
                    source_name=item.source_name,
                    source_type=str(item.source_type),
                    url=item.url,
                    details={
                        "provider": str(self.config.provider),
                        "model": self.config.model,
                    },
                ),
            )

        except NewsLLMError:
            if self.config.fallback_to_rule_based:
                return NewsLLMResult(
                    status=LLMOutputStatus.FALLBACK_USED,
                    provider=self.config.provider,
                    model=self.config.model,
                    error="LLM failed; rule-based fallback should be used",
                )
            raise

        except Exception as exc:
            if self.config.fallback_to_rule_based:
                self.logger.warning(
                    "LLM news analysis failed, falling back to rule-based scoring",
                    extra={
                        "news_id": item.news_id,
                        "source_name": item.source_name,
                        "provider": str(self.config.provider),
                        "model": self.config.model,
                        "error": str(exc),
                    },
                )
                return NewsLLMResult(
                    status=LLMOutputStatus.FALLBACK_USED,
                    provider=self.config.provider,
                    model=self.config.model,
                    error="Unexpected LLM failure; rule-based fallback should be used",
                )

            raise NewsLLMUnavailableError(
                "Unexpected LLM failure",
                context=NewsErrorContext(
                    stage=NewsProcessingStage.LLM_ANALYZE,
                    reason=NewsFailureReason.LLM_UNAVAILABLE,
                    news_id=item.news_id,
                    source_name=item.source_name,
                    source_type=str(item.source_type),
                    url=item.url,
                    details={
                        "provider": str(self.config.provider),
                        "model": self.config.model,
                    },
                ),
                cause=exc,
            ) from exc

    async def _analyze_openai_compatible(
        self,
        prompt_payload: dict[str, Any],
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> NewsLLMResult:
        """
        Analyze news through an OpenAI-compatible chat-completions endpoint.

        For local models, set NewsLLMConfig.base_url to the compatible server.
        For remote providers, keep API keys in environment variables only.
        """

        if not self.config.model:
            raise NewsLLMUnavailableError(
                "LLM model is not configured",
                context=NewsErrorContext(
                    stage=NewsProcessingStage.LLM_ANALYZE,
                    reason=NewsFailureReason.INVALID_CONFIG,
                    details={"provider": str(self.config.provider)},
                ),
            )

        base_url = self.config.base_url or "https://api.openai.com/v1"
        endpoint = base_url.rstrip("/") + "/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key_env:
            api_key = os.getenv(self.config.api_key_env)
            if not api_key:
                raise NewsLLMUnavailableError(
                    "LLM API key environment variable is not set",
                    context=NewsErrorContext(
                        stage=NewsProcessingStage.LLM_ANALYZE,
                        reason=NewsFailureReason.INVALID_CONFIG,
                        details={
                            "api_key_env": self.config.api_key_env,
                            "provider": str(self.config.provider),
                        },
                    ),
                )
            headers["Authorization"] = f"Bearer {api_key}"

        request_body = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        prompt_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
        }

        if self.config.require_json_output:
            request_body["response_format"] = {"type": "json_object"}

        raw_response = await self._post_json_with_retries(
            endpoint=endpoint,
            headers=headers,
            body=request_body,
            session=session,
        )

        content = self._extract_message_content(raw_response)
        parsed = self._parse_json_content(content)

        return self._result_from_payload(parsed, raw_response=raw_response)

    async def _post_json_with_retries(
        self,
        *,
        endpoint: str,
        headers: dict[str, str],
        body: dict[str, Any],
        session: aiohttp.ClientSession | None = None,
    ) -> dict[str, Any]:
        close_session = False
        active_session = session

        if active_session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
            active_session = aiohttp.ClientSession(timeout=timeout)
            close_session = True

        try:
            last_error: BaseException | None = None

            for attempt in range(self.config.max_retries + 1):
                try:
                    async with active_session.post(
                        endpoint,
                        headers=headers,
                        json=body,
                        timeout=aiohttp.ClientTimeout(
                            total=self.config.request_timeout_seconds
                        ),
                    ) as response:
                        response_text = await response.text()

                        if response.status == 429:
                            last_error = NewsLLMUnavailableError(
                                "LLM provider rate limited the request",
                                context=NewsErrorContext(
                                    stage=NewsProcessingStage.LLM_ANALYZE,
                                    reason=NewsFailureReason.RATE_LIMITED,
                                    details={"status": response.status},
                                ),
                            )
                            await self._sleep_before_retry(attempt)
                            continue

                        if response.status >= 500:
                            last_error = NewsLLMUnavailableError(
                                "LLM provider returned a server error",
                                context=NewsErrorContext(
                                    stage=NewsProcessingStage.LLM_ANALYZE,
                                    reason=NewsFailureReason.LLM_UNAVAILABLE,
                                    details={"status": response.status},
                                ),
                            )
                            await self._sleep_before_retry(attempt)
                            continue

                        if response.status >= 400:
                            raise NewsLLMUnavailableError(
                                "LLM provider rejected the request",
                                context=NewsErrorContext(
                                    stage=NewsProcessingStage.LLM_ANALYZE,
                                    reason=NewsFailureReason.INVALID_RESPONSE,
                                    details={
                                        "status": response.status,
                                        "body_preview": response_text[:500],
                                    },
                                ),
                            )

                        try:
                            payload = json.loads(response_text)
                        except json.JSONDecodeError as exc:
                            raise NewsLLMInvalidOutputError(
                                "LLM provider returned non-JSON HTTP response",
                                context=NewsErrorContext(
                                    stage=NewsProcessingStage.LLM_ANALYZE,
                                    reason=NewsFailureReason.LLM_INVALID_OUTPUT,
                                    details={"body_preview": response_text[:500]},
                                ),
                                cause=exc,
                            ) from exc

                        if not isinstance(payload, dict):
                            raise NewsLLMInvalidOutputError(
                                "LLM provider returned unsupported response root",
                                context=NewsErrorContext(
                                    stage=NewsProcessingStage.LLM_ANALYZE,
                                    reason=NewsFailureReason.LLM_INVALID_OUTPUT,
                                    details={"root_type": type(payload).__name__},
                                ),
                            )

                        return payload

                except asyncio.TimeoutError as exc:
                    last_error = NewsLLMUnavailableError(
                        "LLM request timed out",
                        context=NewsErrorContext(
                            stage=NewsProcessingStage.LLM_ANALYZE,
                            reason=NewsFailureReason.TIMEOUT,
                        ),
                        cause=exc,
                    )
                    await self._sleep_before_retry(attempt)

            if isinstance(last_error, NewsLLMError):
                raise last_error

            raise NewsLLMUnavailableError(
                "LLM request failed after retries",
                context=NewsErrorContext(
                    stage=NewsProcessingStage.LLM_ANALYZE,
                    reason=NewsFailureReason.LLM_UNAVAILABLE,
                ),
                cause=last_error,
            )

        finally:
            if close_session and active_session is not None:
                await active_session.close()

    async def _sleep_before_retry(self, attempt: int) -> None:
        if attempt >= self.config.max_retries:
            return

        delay = self.config.retry_delay_seconds * (attempt + 1)
        if delay > 0:
            await asyncio.sleep(delay)

    def _system_prompt(self) -> str:
        """
        System prompt for strict structured news analysis.
        """

        return (
            "You are a crypto market news analyst. "
            "Analyze only the provided news item and deterministic features. "
            "Do not invent facts. "
            "Return only valid JSON. "
            "The analysis is for manual trading review only and must not be treated "
            "as an instruction to open, close, or size a position."
        )

    def _build_prompt_payload(self, request: NewsLLMRequest) -> dict[str, Any]:
        item = request.item
        features = request.features

        text = item.text[: self.config.max_input_chars]

        return {
            "task": "analyze_crypto_news_for_manual_trading_review",
            "required_output_schema": {
                "sentiment_score": "float between -1.0 and 1.0",
                "impact_score": "float between 0.0 and 1.0",
                "confidence_score": "float between 0.0 and 1.0",
                "urgency_score": "float between 0.0 and 1.0",
                "relevance_score": "float between 0.0 and 1.0",
                "sentiment": "very_bearish|bearish|slightly_bearish|neutral|slightly_bullish|bullish|very_bullish|mixed|unknown",
                "market_bias": "bullish|bearish|neutral|mixed|risk_off|risk_on|unknown",
                "time_horizon": "immediate|scalp|intraday|swing|macro|unknown",
                "categories": "array of known category strings",
                "summary": "short factual summary",
                "explanation": "why the scores were assigned",
                "trading_notes": "manual review notes, not financial advice",
            },
            "news": {
                "news_id": item.news_id,
                "source_name": item.source_name,
                "source_type": str(item.source_type),
                "title": item.title,
                "text": text,
                "url": item.url,
                "published_at": item.published_at.isoformat()
                if item.published_at
                else None,
                "language": str(item.language),
                "symbols": list(item.symbols),
                "entities": [entity.to_dict() for entity in item.entities],
                "categories": [str(category) for category in item.categories],
            },
            "deterministic_features": features.to_dict(),
            "constraints": {
                "manual_review_only": True,
                "do_not_recommend_position_size": True,
                "do_not_recommend_leverage": True,
                "do_not_claim_certainty": True,
                "prefer_unknown_when_uncertain": True,
            },
        }

    def _extract_message_content(self, response: dict[str, Any]) -> str:
        """
        Extract assistant message content from OpenAI-compatible response.
        """

        try:
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise KeyError("choices")

            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                raise KeyError("choices[0]")

            message = first_choice.get("message")
            if not isinstance(message, dict):
                raise KeyError("message")

            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise KeyError("content")

            return content.strip()

        except KeyError as exc:
            raise NewsLLMInvalidOutputError(
                "LLM response does not contain message content",
                context=NewsErrorContext(
                    stage=NewsProcessingStage.LLM_ANALYZE,
                    reason=NewsFailureReason.LLM_INVALID_OUTPUT,
                    details={"response_keys": list(response.keys())[:20]},
                ),
                cause=exc,
            ) from exc

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        """
        Parse strict or embedded JSON object from LLM output.
        """

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = _JSON_OBJECT_PATTERN.search(content)
            if not match:
                raise NewsLLMInvalidOutputError(
                    "LLM output is not valid JSON",
                    context=NewsErrorContext(
                        stage=NewsProcessingStage.LLM_ANALYZE,
                        reason=NewsFailureReason.LLM_INVALID_OUTPUT,
                        details={"content_preview": content[:500]},
                    ),
                )

            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise NewsLLMInvalidOutputError(
                    "LLM embedded JSON object is invalid",
                    context=NewsErrorContext(
                        stage=NewsProcessingStage.LLM_ANALYZE,
                        reason=NewsFailureReason.LLM_INVALID_OUTPUT,
                        details={"content_preview": content[:500]},
                    ),
                    cause=exc,
                ) from exc

        if not isinstance(parsed, dict):
            raise NewsLLMInvalidOutputError(
                "LLM JSON output root must be an object",
                context=NewsErrorContext(
                    stage=NewsProcessingStage.LLM_ANALYZE,
                    reason=NewsFailureReason.LLM_INVALID_OUTPUT,
                    details={"root_type": type(parsed).__name__},
                ),
            )

        return parsed

    def _result_from_payload(
        self,
        payload: dict[str, Any],
        *,
        raw_response: dict[str, Any],
    ) -> NewsLLMResult:
        """
        Convert parsed LLM JSON into NewsLLMResult.
        """

        sentiment_score = self._optional_float(
            payload.get("sentiment_score"),
            min_value=-1.0,
            max_value=1.0,
        )
        impact_score = self._optional_float(payload.get("impact_score"))
        confidence_score = self._optional_float(payload.get("confidence_score"))
        urgency_score = self._optional_float(payload.get("urgency_score"))
        relevance_score = self._optional_float(payload.get("relevance_score"))

        sentiment = self._parse_enum(
            payload.get("sentiment"),
            NewsSentiment,
            NewsSentiment.UNKNOWN,
        )
        market_bias = self._parse_enum(
            payload.get("market_bias"),
            NewsMarketBias,
            NewsMarketBias.UNKNOWN,
        )
        time_horizon = self._parse_enum(
            payload.get("time_horizon"),
            NewsTimeHorizon,
            NewsTimeHorizon.UNKNOWN,
        )

        categories = self._parse_categories(payload.get("categories"))

        return NewsLLMResult(
            status=LLMOutputStatus.SUCCESS,
            provider=self.config.provider,
            model=self.config.model,
            sentiment_score=sentiment_score,
            impact_score=impact_score,
            confidence_score=confidence_score,
            urgency_score=urgency_score,
            relevance_score=relevance_score,
            sentiment=sentiment,
            market_bias=market_bias,
            time_horizon=time_horizon,
            categories=categories,
            summary=self._optional_str(payload.get("summary"), max_length=600),
            explanation=self._optional_str(payload.get("explanation"), max_length=1_200),
            trading_notes=self._optional_str(payload.get("trading_notes"), max_length=1_000),
            raw_response=raw_response,
            error=None,
        )

    def _optional_float(
        self,
        value: Any,
        *,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> float | None:
        if value is None:
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        return max(min_value, min(max_value, number))

    def _optional_str(self, value: Any, *, max_length: int) -> str | None:
        if value is None:
            return None

        text = str(value).strip()
        if not text:
            return None

        return text[:max_length]

    def _parse_enum(
        self,
        value: Any,
        enum_cls: type[TEnum],
        default: TEnum,
    ) -> TEnum:
        if value is None:
            return default

        normalized = str(value).strip().lower()

        for member in enum_cls:
            if (
                str(member).lower() == normalized
                or str(member.value).lower() == normalized
            ):
                return member

        return default

    def _parse_categories(self, value: Any) -> tuple[NewsCategory, ...]:
        if value is None:
            return ()

        raw_values: list[Any]
        if isinstance(value, list):
            raw_values = value
        else:
            raw_values = [value]

        categories: list[NewsCategory] = []

        for raw_value in raw_values:
            category = self._parse_enum(
                raw_value,
                NewsCategory,
                NewsCategory.UNKNOWN,
            )
            if category != NewsCategory.UNKNOWN:
                categories.append(category)

        seen: set[NewsCategory] = set()
        deduped: list[NewsCategory] = []

        for category in categories:
            if category in seen:
                continue
            seen.add(category)
            deduped.append(category)

        return tuple(deduped)


__all__ = [
    "NewsLLMRequest",
    "NewsLLMClient",
]