"""
AI/news package configuration.

This module contains typed dataclass configs for the independent news
intelligence layer.

The AI package is intentionally focused only on news collection and scoring.
It does not directly affect risk management, execution, position sizing, or
automatic order placement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import LLMProvider, NewsCategory, NewsLanguage, NewsSourceType


@dataclass(slots=True, frozen=True)
class NewsSourceConfig:
    """
    Configuration for a single news source adapter.

    Examples:
        - RSS feed
        - REST API endpoint
        - exchange announcement endpoint
        - static HTML page fallback
    """

    name: str
    source_type: NewsSourceType
    enabled: bool = True

    url: str | None = None
    api_url: str | None = None
    api_key_env: str | None = None

    request_timeout_seconds: float = 10.0
    max_items_per_fetch: int = 50
    min_fetch_interval_seconds: float = 30.0

    default_language: NewsLanguage = NewsLanguage.UNKNOWN
    default_categories: tuple[NewsCategory, ...] = (NewsCategory.UNKNOWN,)

    source_reputation_score: float = 0.5
    is_official_source: bool = False
    is_exchange_source: bool = False

    headers: dict[str, str] = field(default_factory=dict)
    query_params: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("NewsSourceConfig.name must not be empty")

        if self.source_type in {
            NewsSourceType.RSS,
            NewsSourceType.STATIC_HTML,
            NewsSourceType.EXCHANGE_ANNOUNCEMENT,
        } and not self.url:
            raise ValueError(
                f"NewsSourceConfig.url is required for source_type={self.source_type}"
            )

        if self.source_type == NewsSourceType.API and not self.api_url:
            raise ValueError("NewsSourceConfig.api_url is required for API sources")

        if self.request_timeout_seconds <= 0:
            raise ValueError("NewsSourceConfig.request_timeout_seconds must be > 0")

        if self.max_items_per_fetch <= 0:
            raise ValueError("NewsSourceConfig.max_items_per_fetch must be > 0")

        if self.min_fetch_interval_seconds < 0:
            raise ValueError("NewsSourceConfig.min_fetch_interval_seconds must be >= 0")

        if not 0.0 <= self.source_reputation_score <= 1.0:
            raise ValueError(
                "NewsSourceConfig.source_reputation_score must be between 0.0 and 1.0"
            )

    @property
    def requires_api_key(self) -> bool:
        return bool(self.api_key_env)

    def safe_dict(self) -> dict[str, Any]:
        """
        Return source config without exposing secrets.

        api_key_env is allowed because it is only an environment variable name,
        not the secret value itself.
        """

        return {
            "name": self.name,
            "source_type": str(self.source_type),
            "enabled": self.enabled,
            "url": self.url,
            "api_url": self.api_url,
            "api_key_env": self.api_key_env,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_items_per_fetch": self.max_items_per_fetch,
            "min_fetch_interval_seconds": self.min_fetch_interval_seconds,
            "default_language": str(self.default_language),
            "default_categories": [str(category) for category in self.default_categories],
            "source_reputation_score": self.source_reputation_score,
            "is_official_source": self.is_official_source,
            "is_exchange_source": self.is_exchange_source,
            "headers": {
                key: "***REDACTED***"
                if key.lower() in {"authorization", "cookie", "x-api-key", "x_api_key"}
                else value
                for key, value in self.headers.items()
            },
            "query_params": {
                key: "***REDACTED***"
                if key.lower() in {"api_key", "apikey", "token", "secret"}
                else value
                for key, value in self.query_params.items()
            },
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class NewsDeduplicationConfig:
    """
    Configuration for in-memory/news-history deduplication.
    """

    enabled: bool = True

    ttl_seconds: int = 86_400
    max_seen_items: int = 50_000

    dedup_by_url: bool = True
    dedup_by_canonical_url: bool = True
    dedup_by_source_item_id: bool = True
    dedup_by_title_hash: bool = True
    dedup_by_content_hash: bool = True

    enable_near_duplicate_detection: bool = False
    title_similarity_threshold: float = 0.92
    content_similarity_threshold: float = 0.88

    def __post_init__(self) -> None:
        if self.ttl_seconds <= 0:
            raise ValueError("NewsDeduplicationConfig.ttl_seconds must be > 0")

        if self.max_seen_items <= 0:
            raise ValueError("NewsDeduplicationConfig.max_seen_items must be > 0")

        if not 0.0 <= self.title_similarity_threshold <= 1.0:
            raise ValueError(
                "NewsDeduplicationConfig.title_similarity_threshold must be between 0.0 and 1.0"
            )

        if not 0.0 <= self.content_similarity_threshold <= 1.0:
            raise ValueError(
                "NewsDeduplicationConfig.content_similarity_threshold must be between 0.0 and 1.0"
            )


@dataclass(slots=True, frozen=True)
class NewsFeatureConfig:
    """
    Configuration for deterministic feature extraction.
    """

    min_symbol_length: int = 2
    max_symbol_length: int = 12
    max_entities_per_item: int = 30
    max_symbols_per_item: int = 15

    official_source_reputation_boost: float = 0.15
    exchange_source_reputation_boost: float = 0.08
    low_quality_source_penalty: float = 0.20

    urgent_keywords: tuple[str, ...] = (
        "breaking",
        "urgent",
        "just in",
        "alert",
        "immediately",
    )

    regulatory_keywords: tuple[str, ...] = (
        "sec",
        "cftc",
        "lawsuit",
        "regulator",
        "regulation",
        "investigation",
        "charged",
        "settlement",
        "court",
        "ban",
    )

    macro_keywords: tuple[str, ...] = (
        "fed",
        "fomc",
        "interest rate",
        "cpi",
        "inflation",
        "jobs report",
        "unemployment",
        "rate cut",
        "rate hike",
        "treasury",
        "dollar",
    )

    hack_keywords: tuple[str, ...] = (
        "hack",
        "hacked",
        "exploit",
        "exploited",
        "breach",
        "stolen",
        "drained",
        "security incident",
    )

    listing_keywords: tuple[str, ...] = (
        "will list",
        "listing",
        "listed on",
        "adds support",
        "trading opens",
        "launchpool",
        "launchpad",
    )

    delisting_keywords: tuple[str, ...] = (
        "delist",
        "delisting",
        "remove trading",
        "suspend trading",
        "trading suspension",
    )

    etf_keywords: tuple[str, ...] = (
        "etf",
        "spot etf",
        "approval",
        "approved",
        "delayed decision",
        "filing",
    )

    positive_keywords: tuple[str, ...] = (
        "approval",
        "approved",
        "partnership",
        "adoption",
        "integration",
        "launch",
        "record inflows",
        "upgrade",
        "mainnet",
    )

    negative_keywords: tuple[str, ...] = (
        "hack",
        "exploit",
        "lawsuit",
        "ban",
        "delist",
        "bankruptcy",
        "outflow",
        "probe",
        "investigation",
        "halt",
        "insolvent",
    )

    rumor_keywords: tuple[str, ...] = (
        "rumor",
        "reportedly",
        "sources say",
        "unconfirmed",
        "allegedly",
    )

    def __post_init__(self) -> None:
        if self.min_symbol_length <= 0:
            raise ValueError("NewsFeatureConfig.min_symbol_length must be > 0")

        if self.max_symbol_length < self.min_symbol_length:
            raise ValueError(
                "NewsFeatureConfig.max_symbol_length must be >= min_symbol_length"
            )

        if self.max_entities_per_item <= 0:
            raise ValueError("NewsFeatureConfig.max_entities_per_item must be > 0")

        if self.max_symbols_per_item <= 0:
            raise ValueError("NewsFeatureConfig.max_symbols_per_item must be > 0")

        for field_name in (
            "official_source_reputation_boost",
            "exchange_source_reputation_boost",
            "low_quality_source_penalty",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"NewsFeatureConfig.{field_name} must be between 0.0 and 1.0")


@dataclass(slots=True, frozen=True)
class NewsScoringConfig:
    """
    Configuration for rule-based and hybrid news scoring.
    """

    high_impact_threshold: float = 0.75
    critical_impact_threshold: float = 0.90

    high_urgency_threshold: float = 0.75
    min_relevance_score: float = 0.35
    min_confidence_score: float = 0.35

    rule_score_weight: float = 0.70
    llm_score_weight: float = 0.30

    source_reputation_weight: float = 0.20
    keyword_impact_weight: float = 0.30
    urgency_weight: float = 0.15
    category_weight: float = 0.20
    novelty_weight: float = 0.15

    stale_news_after_seconds: int = 21_600
    fresh_news_window_seconds: int = 1_800

    default_novelty_score: float = 0.50
    default_confidence_score: float = 0.50
    default_relevance_score: float = 0.50

    def __post_init__(self) -> None:
        threshold_fields = (
            "high_impact_threshold",
            "critical_impact_threshold",
            "high_urgency_threshold",
            "min_relevance_score",
            "min_confidence_score",
            "rule_score_weight",
            "llm_score_weight",
            "source_reputation_weight",
            "keyword_impact_weight",
            "urgency_weight",
            "category_weight",
            "novelty_weight",
            "default_novelty_score",
            "default_confidence_score",
            "default_relevance_score",
        )

        for field_name in threshold_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"NewsScoringConfig.{field_name} must be between 0.0 and 1.0")

        if self.critical_impact_threshold < self.high_impact_threshold:
            raise ValueError(
                "NewsScoringConfig.critical_impact_threshold must be >= high_impact_threshold"
            )

        if self.stale_news_after_seconds <= 0:
            raise ValueError("NewsScoringConfig.stale_news_after_seconds must be > 0")

        if self.fresh_news_window_seconds <= 0:
            raise ValueError("NewsScoringConfig.fresh_news_window_seconds must be > 0")

    @property
    def total_component_weight(self) -> float:
        return (
            self.source_reputation_weight
            + self.keyword_impact_weight
            + self.urgency_weight
            + self.category_weight
            + self.novelty_weight
        )


@dataclass(slots=True, frozen=True)
class NewsLLMConfig:
    """
    Configuration for optional LLM-based news analysis.

    LLM must be optional. The system should still work with deterministic
    rule-based scoring when LLM is disabled or unavailable.
    """

    enabled: bool = False
    provider: LLMProvider = LLMProvider.DISABLED

    model: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None

    request_timeout_seconds: float = 20.0
    max_retries: int = 2
    retry_delay_seconds: float = 1.5

    max_input_chars: int = 6_000
    max_output_tokens: int = 800
    temperature: float = 0.0

    require_json_output: bool = True
    fallback_to_rule_based: bool = True

    def __post_init__(self) -> None:
        if self.enabled and self.provider == LLMProvider.DISABLED:
            raise ValueError(
                "NewsLLMConfig.provider must not be DISABLED when LLM is enabled"
            )

        if self.enabled and not self.model:
            raise ValueError("NewsLLMConfig.model is required when LLM is enabled")

        if self.request_timeout_seconds <= 0:
            raise ValueError("NewsLLMConfig.request_timeout_seconds must be > 0")

        if self.max_retries < 0:
            raise ValueError("NewsLLMConfig.max_retries must be >= 0")

        if self.retry_delay_seconds < 0:
            raise ValueError("NewsLLMConfig.retry_delay_seconds must be >= 0")

        if self.max_input_chars <= 0:
            raise ValueError("NewsLLMConfig.max_input_chars must be > 0")

        if self.max_output_tokens <= 0:
            raise ValueError("NewsLLMConfig.max_output_tokens must be > 0")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("NewsLLMConfig.temperature must be between 0.0 and 2.0")

    @property
    def requires_api_key(self) -> bool:
        return self.enabled and bool(self.api_key_env)

    def safe_dict(self) -> dict[str, Any]:
        """
        Return config safe for logs/events.

        api_key_env is safe because it is only the environment variable name,
        not the secret value.
        """

        return {
            "enabled": self.enabled,
            "provider": str(self.provider),
            "model": self.model,
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "max_input_chars": self.max_input_chars,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "require_json_output": self.require_json_output,
            "fallback_to_rule_based": self.fallback_to_rule_based,
        }


@dataclass(slots=True, frozen=True)
class NewsAIConfig:
    """
    Top-level config for the AI news intelligence service.

    This config is passed into NewsAIService together with EventBus and
    Scheduler through constructor dependency injection.
    """

    enabled: bool = True

    collect_interval_seconds: float = 60.0
    startup_collect_enabled: bool = True
    publish_raw_fetched_event: bool = False
    publish_scored_event: bool = True
    publish_high_impact_event: bool = True
    publish_failed_events: bool = True

    max_items_per_cycle: int = 200
    max_concurrent_sources: int = 5

    default_language: NewsLanguage = NewsLanguage.EN
    tracked_symbols: tuple[str, ...] = (
        "BTC",
        "ETH",
        "SOL",
        "BNB",
        "XRP",
        "DOGE",
        "ADA",
        "AVAX",
        "LINK",
        "TON",
    )

    source_configs: tuple[NewsSourceConfig, ...] = ()
    deduplication: NewsDeduplicationConfig = field(default_factory=NewsDeduplicationConfig)
    features: NewsFeatureConfig = field(default_factory=NewsFeatureConfig)
    scoring: NewsScoringConfig = field(default_factory=NewsScoringConfig)
    llm: NewsLLMConfig = field(default_factory=NewsLLMConfig)

    service_name: str = "news_ai_service"

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.collect_interval_seconds <= 0:
            raise ValueError("NewsAIConfig.collect_interval_seconds must be > 0")

        if self.max_items_per_cycle <= 0:
            raise ValueError("NewsAIConfig.max_items_per_cycle must be > 0")

        if self.max_concurrent_sources <= 0:
            raise ValueError("NewsAIConfig.max_concurrent_sources must be > 0")

        if not self.service_name.strip():
            raise ValueError("NewsAIConfig.service_name must not be empty")

        normalized_symbols = tuple(
            symbol.strip().upper() for symbol in self.tracked_symbols if symbol.strip()
        )
        if len(normalized_symbols) != len(set(normalized_symbols)):
            raise ValueError("NewsAIConfig.tracked_symbols must not contain duplicates")

        source_names = tuple(source.name for source in self.source_configs)
        if len(source_names) != len(set(source_names)):
            raise ValueError("NewsAIConfig.source_configs must not contain duplicate names")

    @property
    def enabled_sources(self) -> tuple[NewsSourceConfig, ...]:
        return tuple(source for source in self.source_configs if source.enabled)

    @property
    def has_sources(self) -> bool:
        return bool(self.enabled_sources)

    @property
    def normalized_tracked_symbols(self) -> tuple[str, ...]:
        return tuple(symbol.strip().upper() for symbol in self.tracked_symbols if symbol.strip())

    def safe_dict(self) -> dict[str, Any]:
        """
        Return top-level config safe for logs and diagnostics.
        """

        return {
            "enabled": self.enabled,
            "collect_interval_seconds": self.collect_interval_seconds,
            "startup_collect_enabled": self.startup_collect_enabled,
            "publish_raw_fetched_event": self.publish_raw_fetched_event,
            "publish_scored_event": self.publish_scored_event,
            "publish_high_impact_event": self.publish_high_impact_event,
            "publish_failed_events": self.publish_failed_events,
            "max_items_per_cycle": self.max_items_per_cycle,
            "max_concurrent_sources": self.max_concurrent_sources,
            "default_language": str(self.default_language),
            "tracked_symbols": list(self.normalized_tracked_symbols),
            "service_name": self.service_name,
            "source_configs": [source.safe_dict() for source in self.source_configs],
            "deduplication": {
                "enabled": self.deduplication.enabled,
                "ttl_seconds": self.deduplication.ttl_seconds,
                "max_seen_items": self.deduplication.max_seen_items,
                "dedup_by_url": self.deduplication.dedup_by_url,
                "dedup_by_canonical_url": self.deduplication.dedup_by_canonical_url,
                "dedup_by_source_item_id": self.deduplication.dedup_by_source_item_id,
                "dedup_by_title_hash": self.deduplication.dedup_by_title_hash,
                "dedup_by_content_hash": self.deduplication.dedup_by_content_hash,
                "enable_near_duplicate_detection": (
                    self.deduplication.enable_near_duplicate_detection
                ),
                "title_similarity_threshold": self.deduplication.title_similarity_threshold,
                "content_similarity_threshold": (
                    self.deduplication.content_similarity_threshold
                ),
            },
            "features": {
                "min_symbol_length": self.features.min_symbol_length,
                "max_symbol_length": self.features.max_symbol_length,
                "max_entities_per_item": self.features.max_entities_per_item,
                "max_symbols_per_item": self.features.max_symbols_per_item,
                "official_source_reputation_boost": (
                    self.features.official_source_reputation_boost
                ),
                "exchange_source_reputation_boost": (
                    self.features.exchange_source_reputation_boost
                ),
                "low_quality_source_penalty": self.features.low_quality_source_penalty,
            },
            "scoring": {
                "high_impact_threshold": self.scoring.high_impact_threshold,
                "critical_impact_threshold": self.scoring.critical_impact_threshold,
                "high_urgency_threshold": self.scoring.high_urgency_threshold,
                "min_relevance_score": self.scoring.min_relevance_score,
                "min_confidence_score": self.scoring.min_confidence_score,
                "rule_score_weight": self.scoring.rule_score_weight,
                "llm_score_weight": self.scoring.llm_score_weight,
                "stale_news_after_seconds": self.scoring.stale_news_after_seconds,
                "fresh_news_window_seconds": self.scoring.fresh_news_window_seconds,
            },
            "llm": self.llm.safe_dict(),
            "metadata": self.metadata,
        }


def build_default_news_ai_config() -> NewsAIConfig:
    """
    Build a conservative default config.

    The default config contains no active external source because real source
    URLs/API keys should be configured explicitly by the application layer.
    """

    return NewsAIConfig(
        enabled=True,
        collect_interval_seconds=60.0,
        startup_collect_enabled=True,
        source_configs=(),
        llm=NewsLLMConfig(enabled=False, provider=LLMProvider.DISABLED),
    )


__all__ = [
    "NewsSourceConfig",
    "NewsDeduplicationConfig",
    "NewsFeatureConfig",
    "NewsScoringConfig",
    "NewsLLMConfig",
    "NewsAIConfig",
    "build_default_news_ai_config",
]