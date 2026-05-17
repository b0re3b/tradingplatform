from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .enums import (
    LLMOutputStatus,
    LLMProvider,
    NewsAlertType,
    NewsCategory,
    NewsDeduplicationReason,
    NewsEntityType,
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


def utc_now() -> datetime:
    """
    Return timezone-aware UTC datetime.
    """

    return datetime.now(UTC)


def new_id(prefix: str = "news") -> str:
    """
    Generate a simple stable-looking runtime identifier.

    Persistent identifiers should later be derived from source item id,
    canonical URL, title hash, or content hash in NewsProcessor/Deduplicator.
    """

    return f"{prefix}_{uuid4().hex}"


@dataclass(slots=True, frozen=True)
class NewsEntity:
    """
    Entity extracted from a news item.

    Examples:
        BTC as SYMBOL
        Binance as EXCHANGE
        SEC as REGULATOR
        Ethereum as PROJECT
    """

    name: str
    entity_type: NewsEntityType = NewsEntityType.UNKNOWN
    symbol: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("NewsEntity.name must not be empty")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("NewsEntity.confidence must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": str(self.entity_type),
            "symbol": self.symbol,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class RawNewsItem:
    """
    Raw news item returned by a source adapter.

    This model should stay close to the source response and should not contain
    advanced scoring or processed trading interpretation.
    """

    source_name: str
    source_type: NewsSourceType
    title: str
    url: str | None = None
    body: str | None = None
    summary: str | None = None
    author: str | None = None
    source_item_id: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime = field(default_factory=utc_now)
    language: NewsLanguage = NewsLanguage.UNKNOWN
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_name.strip():
            raise ValueError("RawNewsItem.source_name must not be empty")

        if not self.title.strip():
            raise ValueError("RawNewsItem.title must not be empty")

    @property
    def text(self) -> str:
        """
        Combined raw text useful for normalization and feature extraction.
        """

        parts = [self.title, self.summary or "", self.body or ""]
        return "\n".join(part.strip() for part in parts if part and part.strip())

    def to_dict(self, *, include_raw_payload: bool = False) -> dict[str, Any]:
        payload = {
            "source_name": self.source_name,
            "source_type": str(self.source_type),
            "title": self.title,
            "url": self.url,
            "body": self.body,
            "summary": self.summary,
            "author": self.author,
            "source_item_id": self.source_item_id,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "fetched_at": self.fetched_at.isoformat(),
            "language": str(self.language),
        }

        if include_raw_payload:
            payload["raw_payload"] = self.raw_payload

        return payload


@dataclass(slots=True, frozen=True)
class NormalizedNewsItem:
    """
    Cleaned and normalized news item used by scoring components.
    """

    news_id: str
    source_name: str
    source_type: NewsSourceType
    title: str
    text: str
    url: str | None = None
    canonical_url: str | None = None
    summary: str | None = None
    author: str | None = None
    source_item_id: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime = field(default_factory=utc_now)
    processed_at: datetime = field(default_factory=utc_now)

    language: NewsLanguage = NewsLanguage.UNKNOWN
    categories: tuple[NewsCategory, ...] = (NewsCategory.UNKNOWN,)
    entities: tuple[NewsEntity, ...] = ()
    symbols: tuple[str, ...] = ()

    title_hash: str | None = None
    content_hash: str | None = None
    source_reputation_score: float = 0.5

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.news_id.strip():
            raise ValueError("NormalizedNewsItem.news_id must not be empty")

        if not self.source_name.strip():
            raise ValueError("NormalizedNewsItem.source_name must not be empty")

        if not self.title.strip():
            raise ValueError("NormalizedNewsItem.title must not be empty")

        if not self.text.strip():
            raise ValueError("NormalizedNewsItem.text must not be empty")

        if not 0.0 <= self.source_reputation_score <= 1.0:
            raise ValueError(
                "NormalizedNewsItem.source_reputation_score must be between 0.0 and 1.0"
            )

    @property
    def age_seconds(self) -> float | None:
        """
        Age of the news item based on published_at, if available.
        """

        if self.published_at is None:
            return None
        return max(0.0, (utc_now() - self.published_at).total_seconds())

    @property
    def primary_symbol(self) -> str | None:
        return self.symbols[0] if self.symbols else None

    @property
    def primary_category(self) -> NewsCategory:
        return self.categories[0] if self.categories else NewsCategory.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "news_id": self.news_id,
            "source_name": self.source_name,
            "source_type": str(self.source_type),
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "summary": self.summary,
            "author": self.author,
            "source_item_id": self.source_item_id,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "fetched_at": self.fetched_at.isoformat(),
            "processed_at": self.processed_at.isoformat(),
            "language": str(self.language),
            "categories": [str(category) for category in self.categories],
            "entities": [entity.to_dict() for entity in self.entities],
            "symbols": list(self.symbols),
            "title_hash": self.title_hash,
            "content_hash": self.content_hash,
            "source_reputation_score": self.source_reputation_score,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class NewsFeatures:
    """
    Extracted deterministic features used by NewsScorer.

    These features are intentionally rule-based and fast. LLM analysis is
    optional and lives separately in NewsLLMResult.
    """

    news_id: str

    source_reputation_score: float = 0.5
    title_strength_score: float = 0.0
    text_length_score: float = 0.0

    symbol_count: int = 0
    entity_count: int = 0
    category_count: int = 0

    has_urgent_keywords: bool = False
    has_regulatory_keywords: bool = False
    has_macro_keywords: bool = False
    has_hack_keywords: bool = False
    has_exploit_keywords: bool = False
    has_listing_keywords: bool = False
    has_delisting_keywords: bool = False
    has_etf_keywords: bool = False
    has_lawsuit_keywords: bool = False
    has_bankruptcy_keywords: bool = False
    has_partnership_keywords: bool = False
    has_token_unlock_keywords: bool = False
    has_airdrop_keywords: bool = False
    has_rumor_keywords: bool = False

    is_official_source: bool = False
    is_exchange_source: bool = False
    is_breaking_news: bool = False
    is_low_quality_source: bool = False

    matched_keywords: tuple[str, ...] = ()
    matched_negative_keywords: tuple[str, ...] = ()
    matched_positive_keywords: tuple[str, ...] = ()

    raw_feature_values: dict[str, float | int | bool | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "source_reputation_score",
            "title_strength_score",
            "text_length_score",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"NewsFeatures.{field_name} must be between 0.0 and 1.0")

        if self.symbol_count < 0:
            raise ValueError("NewsFeatures.symbol_count must be non-negative")
        if self.entity_count < 0:
            raise ValueError("NewsFeatures.entity_count must be non-negative")
        if self.category_count < 0:
            raise ValueError("NewsFeatures.category_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "news_id": self.news_id,
            "source_reputation_score": self.source_reputation_score,
            "title_strength_score": self.title_strength_score,
            "text_length_score": self.text_length_score,
            "symbol_count": self.symbol_count,
            "entity_count": self.entity_count,
            "category_count": self.category_count,
            "has_urgent_keywords": self.has_urgent_keywords,
            "has_regulatory_keywords": self.has_regulatory_keywords,
            "has_macro_keywords": self.has_macro_keywords,
            "has_hack_keywords": self.has_hack_keywords,
            "has_exploit_keywords": self.has_exploit_keywords,
            "has_listing_keywords": self.has_listing_keywords,
            "has_delisting_keywords": self.has_delisting_keywords,
            "has_etf_keywords": self.has_etf_keywords,
            "has_lawsuit_keywords": self.has_lawsuit_keywords,
            "has_bankruptcy_keywords": self.has_bankruptcy_keywords,
            "has_partnership_keywords": self.has_partnership_keywords,
            "has_token_unlock_keywords": self.has_token_unlock_keywords,
            "has_airdrop_keywords": self.has_airdrop_keywords,
            "has_rumor_keywords": self.has_rumor_keywords,
            "is_official_source": self.is_official_source,
            "is_exchange_source": self.is_exchange_source,
            "is_breaking_news": self.is_breaking_news,
            "is_low_quality_source": self.is_low_quality_source,
            "matched_keywords": list(self.matched_keywords),
            "matched_negative_keywords": list(self.matched_negative_keywords),
            "matched_positive_keywords": list(self.matched_positive_keywords),
            "raw_feature_values": self.raw_feature_values,
        }


@dataclass(slots=True, frozen=True)
class NewsLLMResult:
    """
    Optional structured result from LLM-based news analysis.
    """

    status: LLMOutputStatus
    provider: LLMProvider = LLMProvider.DISABLED
    model: str | None = None

    sentiment_score: float | None = None
    impact_score: float | None = None
    confidence_score: float | None = None
    urgency_score: float | None = None
    relevance_score: float | None = None

    sentiment: NewsSentiment = NewsSentiment.UNKNOWN
    market_bias: NewsMarketBias = NewsMarketBias.UNKNOWN
    time_horizon: NewsTimeHorizon = NewsTimeHorizon.UNKNOWN
    categories: tuple[NewsCategory, ...] = ()

    summary: str | None = None
    explanation: str | None = None
    trading_notes: str | None = None

    raw_response: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for field_name in (
            "sentiment_score",
            "impact_score",
            "confidence_score",
            "urgency_score",
            "relevance_score",
        ):
            value = getattr(self, field_name)
            if value is not None and not -1.0 <= value <= 1.0:
                if field_name == "sentiment_score":
                    raise ValueError(
                        f"NewsLLMResult.{field_name} must be between -1.0 and 1.0"
                    )
                raise ValueError(f"NewsLLMResult.{field_name} must be between 0.0 and 1.0")

    @property
    def is_successful(self) -> bool:
        return self.status == LLMOutputStatus.SUCCESS

    def to_dict(self, *, include_raw_response: bool = False) -> dict[str, Any]:
        payload = {
            "status": str(self.status),
            "provider": str(self.provider),
            "model": self.model,
            "sentiment_score": self.sentiment_score,
            "impact_score": self.impact_score,
            "confidence_score": self.confidence_score,
            "urgency_score": self.urgency_score,
            "relevance_score": self.relevance_score,
            "sentiment": str(self.sentiment),
            "market_bias": str(self.market_bias),
            "time_horizon": str(self.time_horizon),
            "categories": [str(category) for category in self.categories],
            "summary": self.summary,
            "explanation": self.explanation,
            "trading_notes": self.trading_notes,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
        }

        if include_raw_response:
            payload["raw_response"] = self.raw_response

        return payload


@dataclass(slots=True, frozen=True)
class NewsScore:
    """
    Final scoring result for a normalized news item.

    The score is intended for manual review, dashboard display, and alerts.
    It must not directly trigger order execution.
    """

    news_id: str

    sentiment_score: float = 0.0
    impact_score: float = 0.0
    confidence_score: float = 0.0
    urgency_score: float = 0.0
    novelty_score: float = 0.0
    relevance_score: float = 0.0
    source_reputation_score: float = 0.5

    sentiment: NewsSentiment = NewsSentiment.UNKNOWN
    market_bias: NewsMarketBias = NewsMarketBias.UNKNOWN
    impact_level: NewsImpactLevel = NewsImpactLevel.UNKNOWN
    urgency_level: NewsUrgencyLevel = NewsUrgencyLevel.UNKNOWN
    relevance_level: NewsRelevanceLevel = NewsRelevanceLevel.UNKNOWN
    time_horizon: NewsTimeHorizon = NewsTimeHorizon.UNKNOWN

    categories: tuple[NewsCategory, ...] = ()
    alert_types: tuple[NewsAlertType, ...] = ()

    summary: str | None = None
    explanation: str | None = None
    trading_notes: str | None = None

    rule_score_weight: float = 1.0
    llm_score_weight: float = 0.0
    llm_status: LLMOutputStatus = LLMOutputStatus.DISABLED

    scored_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.news_id.strip():
            raise ValueError("NewsScore.news_id must not be empty")

        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError("NewsScore.sentiment_score must be between -1.0 and 1.0")

        for field_name in (
            "impact_score",
            "confidence_score",
            "urgency_score",
            "novelty_score",
            "relevance_score",
            "source_reputation_score",
            "rule_score_weight",
            "llm_score_weight",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"NewsScore.{field_name} must be between 0.0 and 1.0")

    @property
    def is_high_impact(self) -> bool:
        return self.impact_level in {
            NewsImpactLevel.HIGH,
            NewsImpactLevel.CRITICAL,
        }

    @property
    def is_actionable_for_manual_review(self) -> bool:
        """
        True when item is relevant enough to show prominently to the user.

        This is not an auto-trading permission.
        """

        return (
            self.relevance_score >= 0.5
            and self.confidence_score >= 0.45
            and self.impact_score >= 0.45
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "news_id": self.news_id,
            "sentiment_score": self.sentiment_score,
            "impact_score": self.impact_score,
            "confidence_score": self.confidence_score,
            "urgency_score": self.urgency_score,
            "novelty_score": self.novelty_score,
            "relevance_score": self.relevance_score,
            "source_reputation_score": self.source_reputation_score,
            "sentiment": str(self.sentiment),
            "market_bias": str(self.market_bias),
            "impact_level": str(self.impact_level),
            "urgency_level": str(self.urgency_level),
            "relevance_level": str(self.relevance_level),
            "time_horizon": str(self.time_horizon),
            "categories": [str(category) for category in self.categories],
            "alert_types": [str(alert_type) for alert_type in self.alert_types],
            "summary": self.summary,
            "explanation": self.explanation,
            "trading_notes": self.trading_notes,
            "rule_score_weight": self.rule_score_weight,
            "llm_score_weight": self.llm_score_weight,
            "llm_status": str(self.llm_status),
            "scored_at": self.scored_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class ScoredNewsItem:
    """
    Final enriched news item used by NewsAIService for publishing.
    """

    item: NormalizedNewsItem
    features: NewsFeatures
    score: NewsScore
    llm_result: NewsLLMResult | None = None

    def __post_init__(self) -> None:
        if self.item.news_id != self.score.news_id:
            raise ValueError("ScoredNewsItem.item.news_id must match score.news_id")

        if self.item.news_id != self.features.news_id:
            raise ValueError("ScoredNewsItem.item.news_id must match features.news_id")

        if self.llm_result is not None and self.llm_result.status == LLMOutputStatus.SUCCESS:
            # No hard validation against news_id because LLM result is provider-level output.
            pass

    @property
    def news_id(self) -> str:
        return self.item.news_id

    @property
    def is_high_impact(self) -> bool:
        return self.score.is_high_impact

    def to_event_payload(self) -> dict[str, Any]:
        """
        Payload safe for EventBus publishing.
        """

        return {
            "news_id": self.news_id,
            "item": self.item.to_dict(),
            "features": self.features.to_dict(),
            "score": self.score.to_dict(),
            "llm_result": self.llm_result.to_dict() if self.llm_result else None,
        }


@dataclass(slots=True, frozen=True)
class NewsSourceHealth:
    """
    Runtime health snapshot for a single source adapter.
    """

    source_name: str
    source_type: NewsSourceType
    status: NewsSourceStatus = NewsSourceStatus.UNKNOWN
    last_fetch_status: NewsFetchStatus = NewsFetchStatus.EMPTY
    last_fetch_at: datetime | None = None
    last_success_at: datetime | None = None
    last_error: str | None = None

    total_fetches: int = 0
    successful_fetches: int = 0
    failed_fetches: int = 0
    total_items_fetched: int = 0

    average_latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_name.strip():
            raise ValueError("NewsSourceHealth.source_name must not be empty")

        for field_name in (
            "total_fetches",
            "successful_fetches",
            "failed_fetches",
            "total_items_fetched",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"NewsSourceHealth.{field_name} must be non-negative")

        if self.average_latency_ms is not None and self.average_latency_ms < 0:
            raise ValueError("NewsSourceHealth.average_latency_ms must be non-negative")

    @property
    def success_rate(self) -> float:
        if self.total_fetches <= 0:
            return 0.0
        return self.successful_fetches / self.total_fetches

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_type": str(self.source_type),
            "status": str(self.status),
            "last_fetch_status": str(self.last_fetch_status),
            "last_fetch_at": self.last_fetch_at.isoformat() if self.last_fetch_at else None,
            "last_success_at": self.last_success_at.isoformat()
            if self.last_success_at
            else None,
            "last_error": self.last_error,
            "total_fetches": self.total_fetches,
            "successful_fetches": self.successful_fetches,
            "failed_fetches": self.failed_fetches,
            "total_items_fetched": self.total_items_fetched,
            "success_rate": self.success_rate,
            "average_latency_ms": self.average_latency_ms,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class NewsProcessingResult:
    """
    Result object for processing a batch of raw news items.
    """

    stage: NewsProcessingStage
    started_at: datetime
    finished_at: datetime
    raw_count: int = 0
    processed_count: int = 0
    duplicate_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    errors: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        counters = (
            "raw_count",
            "processed_count",
            "duplicate_count",
            "failed_count",
            "skipped_count",
        )
        for counter in counters:
            value = getattr(self, counter)
            if value < 0:
                raise ValueError(f"NewsProcessingResult.{counter} must be non-negative")

        if self.finished_at < self.started_at:
            raise ValueError("NewsProcessingResult.finished_at must be >= started_at")

    @property
    def duration_ms(self) -> float:
        return (self.finished_at - self.started_at).total_seconds() * 1000.0

    @property
    def success_count(self) -> int:
        return self.processed_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": str(self.stage),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_ms": self.duration_ms,
            "raw_count": self.raw_count,
            "processed_count": self.processed_count,
            "duplicate_count": self.duplicate_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "errors": list(self.errors),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class DeduplicationDecision:
    """
    Deduplication result for a single news item.
    """

    is_duplicate: bool
    reason: NewsDeduplicationReason = NewsDeduplicationReason.UNKNOWN
    existing_news_id: str | None = None
    similarity_score: float | None = None
    checked_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.similarity_score is not None and not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(
                "DeduplicationDecision.similarity_score must be between 0.0 and 1.0"
            )

    @classmethod
    def unique(cls) -> "DeduplicationDecision":
        return cls(is_duplicate=False)

    @classmethod
    def duplicate(
        cls,
        *,
        reason: NewsDeduplicationReason,
        existing_news_id: str | None = None,
        similarity_score: float | None = None,
    ) -> "DeduplicationDecision":
        return cls(
            is_duplicate=True,
            reason=reason,
            existing_news_id=existing_news_id,
            similarity_score=similarity_score,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_duplicate": self.is_duplicate,
            "reason": str(self.reason),
            "existing_news_id": self.existing_news_id,
            "similarity_score": self.similarity_score,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass(slots=True, frozen=True)
class NewsBatch:
    """
    Generic batch container used between collector/service layers.
    """

    batch_id: str = field(default_factory=lambda: new_id("news_batch"))
    items: tuple[RawNewsItem, ...] = ()
    source_health: tuple[NewsSourceHealth, ...] = ()
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "count": self.count,
            "items": [item.to_dict() for item in self.items],
            "source_health": [health.to_dict() for health in self.source_health],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


__all__ = [
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
]