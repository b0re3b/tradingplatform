"""
AI package.

This package contains an independent news intelligence layer for the trading
system.

The AI package is currently focused only on news collection, normalization,
feature extraction, optional LLM analysis, and scoring for manual trading
review.

Important:
    - AI/news does not directly affect risk management.
    - AI/news does not submit, cancel, or modify orders.
    - AI/news does not change position sizing or leverage.
    - High-impact news is published as events for dashboard/bots/manual review.
"""

from __future__ import annotations

from .config import (
    NewsAIConfig,
    NewsDeduplicationConfig,
    NewsFeatureConfig,
    NewsLLMConfig,
    NewsScoringConfig,
    NewsSourceConfig,
    build_default_news_ai_config,
)
from .enums import (
    LLMOutputStatus,
    LLMProvider,
    NewsAlertType,
    NewsCategory,
    NewsDeduplicationReason,
    NewsEntityType,
    NewsFailureReason,
    NewsFetchStatus,
    NewsImpactLevel,
    NewsLanguage,
    NewsMarketBias,
    NewsProcessingStage,
    NewsRelevanceLevel,
    NewsSentiment,
    NewsSourceStatus,
    NewsSourceType,
    NewsTimeHorizon,
    NewsUrgencyLevel,
)
from .exceptions import (
    NewsAIError,
    NewsConfigError,
    NewsDeduplicationError,
    NewsErrorContext,
    NewsFeatureExtractionError,
    NewsFetchError,
    NewsInvalidResponseError,
    NewsLLMError,
    NewsLLMInvalidOutputError,
    NewsLLMUnavailableError,
    NewsProcessingError,
    NewsPublishError,
    NewsRateLimitError,
    NewsScoringError,
    NewsSourceError,
    NewsTimeoutError,
    NewsValidationError,
)
from .models import (
    DeduplicationDecision,
    NewsBatch,
    NewsEntity,
    NewsFeatures,
    NewsLLMResult,
    NewsProcessingResult,
    NewsScore,
    NewsSourceHealth,
    NormalizedNewsItem,
    RawNewsItem,
    ScoredNewsItem,
    new_id,
    utc_now,
)
from .news_collector import NewsCollectionResult, NewsCollector
from .news_deduplicator import NewsDeduplicator
from .news_features import NewsFeatureExtractor
from .news_llm import NewsLLMClient, NewsLLMRequest
from .news_processor import NewsProcessor, ProcessedNewsBatch
from .news_scorer import NewsScorer
from .news_service import NewsAIService, NewsServiceRunResult
from .news_sources import (
    APINewsSource,
    BaseNewsSource,
    ExchangeAnnouncementSource,
    RSSNewsSource,
    StaticHTMLNewsSource,
    build_news_source,
)


__all__ = [
    # Config
    "NewsAIConfig",
    "NewsSourceConfig",
    "NewsDeduplicationConfig",
    "NewsFeatureConfig",
    "NewsScoringConfig",
    "NewsLLMConfig",
    "build_default_news_ai_config",

    # Enums
    "NewsSourceType",
    "NewsSourceStatus",
    "NewsFetchStatus",
    "NewsLanguage",
    "NewsCategory",
    "NewsEntityType",
    "NewsSentiment",
    "NewsMarketBias",
    "NewsImpactLevel",
    "NewsTimeHorizon",
    "NewsUrgencyLevel",
    "NewsRelevanceLevel",
    "NewsProcessingStage",
    "NewsFailureReason",
    "NewsDeduplicationReason",
    "NewsAlertType",
    "LLMProvider",
    "LLMOutputStatus",

    # Exceptions
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

    # Models
    "utc_now",
    "new_id",
    "NewsEntity",
    "RawNewsItem",
    "NormalizedNewsItem",
    "NewsFeatures",
    "NewsLLMResult",
    "NewsScore",
    "ScoredNewsItem",
    "NewsSourceHealth",
    "NewsProcessingResult",
    "DeduplicationDecision",
    "NewsBatch",

    # Sources
    "BaseNewsSource",
    "RSSNewsSource",
    "APINewsSource",
    "ExchangeAnnouncementSource",
    "StaticHTMLNewsSource",
    "build_news_source",

    # Pipeline components
    "NewsCollector",
    "NewsCollectionResult",
    "NewsDeduplicator",
    "NewsProcessor",
    "ProcessedNewsBatch",
    "NewsFeatureExtractor",
    "NewsLLMClient",
    "NewsLLMRequest",
    "NewsScorer",

    # Service
    "NewsAIService",
    "NewsServiceRunResult",
]