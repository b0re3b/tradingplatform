"""
AI/news package enums.

This module defines stable enum contracts for the news intelligence layer.
The AI package is intentionally focused on independent news analysis and does
not directly affect risk management, execution, or position sizing.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """
    Local string enum base.

    Python 3.11 has enum.StrEnum, but this local implementation keeps the
    package compatible with Python 3.10+ without adding extra dependencies.
    """

    def __str__(self) -> str:
        return self.value


class NewsSourceType(StrEnum):
    """
    Supported news source adapter types.
    """

    RSS = "rss"
    API = "api"
    EXCHANGE_ANNOUNCEMENT = "exchange_announcement"
    STATIC_HTML = "static_html"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class NewsSourceStatus(StrEnum):
    """
    Runtime health/status of a news source.
    """

    ENABLED = "enabled"
    DISABLED = "disabled"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    UNKNOWN = "unknown"


class NewsFetchStatus(StrEnum):
    """
    Result status for a fetch attempt.
    """

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    EMPTY = "empty"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    INVALID_RESPONSE = "invalid_response"


class NewsLanguage(StrEnum):
    """
    Basic language classification for news text.

    The first version should mostly operate with English sources, but keeping
    language explicit helps later filtering, routing, and LLM prompt selection.
    """

    EN = "en"
    UK = "uk"
    RU = "ru"
    ES = "es"
    DE = "de"
    FR = "fr"
    ZH = "zh"
    JA = "ja"
    KO = "ko"
    UNKNOWN = "unknown"


class NewsCategory(StrEnum):
    """
    High-level market category of a news item.
    """

    REGULATION = "regulation"
    MACRO = "macro"
    ETF = "etf"
    EXCHANGE = "exchange"
    LISTING = "listing"
    DELISTING = "delisting"
    HACK = "hack"
    EXPLOIT = "exploit"
    SECURITY = "security"
    STABLECOIN = "stablecoin"
    DEFI = "defi"
    NFT = "nft"
    LAYER_1 = "layer_1"
    LAYER_2 = "layer_2"
    PARTNERSHIP = "partnership"
    FUNDING = "funding"
    TOKEN_UNLOCK = "token_unlock"
    AIRDROP = "airdrop"
    GOVERNANCE = "governance"
    LEGAL = "legal"
    BANKRUPTCY = "bankruptcy"
    MARKET_MOVING = "market_moving"
    RUMOR = "rumor"
    GENERAL = "general"
    UNKNOWN = "unknown"


class NewsEntityType(StrEnum):
    """
    Type of extracted entity from a news item.
    """

    SYMBOL = "symbol"
    PROJECT = "project"
    EXCHANGE = "exchange"
    PERSON = "person"
    COMPANY = "company"
    REGULATOR = "regulator"
    COUNTRY = "country"
    PROTOCOL = "protocol"
    BLOCKCHAIN = "blockchain"
    TOKEN = "token"
    UNKNOWN = "unknown"


class NewsSentiment(StrEnum):
    """
    Discrete sentiment label.

    Numeric sentiment_score will live in models.NewsScore.
    """

    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    SLIGHTLY_BEARISH = "slightly_bearish"
    NEUTRAL = "neutral"
    SLIGHTLY_BULLISH = "slightly_bullish"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class NewsMarketBias(StrEnum):
    """
    Trading-oriented directional interpretation.

    This is not a trading signal and must not directly trigger execution.
    It is only used for manual review, dashboard alerts, and bot notifications.
    """

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    RISK_OFF = "risk_off"
    RISK_ON = "risk_on"
    UNKNOWN = "unknown"


class NewsImpactLevel(StrEnum):
    """
    Discrete market impact bucket.

    Numeric impact_score will live in models.NewsScore.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class NewsTimeHorizon(StrEnum):
    """
    Expected time horizon of the news impact.
    """

    IMMEDIATE = "immediate"
    SCALP = "scalp"
    INTRADAY = "intraday"
    SWING = "swing"
    MACRO = "macro"
    UNKNOWN = "unknown"


class NewsUrgencyLevel(StrEnum):
    """
    Urgency bucket used for alerts and prioritization.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class NewsRelevanceLevel(StrEnum):
    """
    How relevant a news item is for the trading universe.
    """

    IRRELEVANT = "irrelevant"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNKNOWN = "unknown"


class NewsProcessingStage(StrEnum):
    """
    Internal pipeline stage names.

    Useful for logs, metrics, errors, and emitted event payloads.
    """

    FETCH = "fetch"
    COLLECT = "collect"
    DEDUPLICATE = "deduplicate"
    NORMALIZE = "normalize"
    PROCESS = "process"
    EXTRACT_FEATURES = "extract_features"
    LLM_ANALYZE = "llm_analyze"
    SCORE = "score"
    PUBLISH = "publish"


class NewsFailureReason(StrEnum):
    """
    Common normalized failure reasons.
    """

    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    INVALID_CONFIG = "invalid_config"
    INVALID_RESPONSE = "invalid_response"
    EMPTY_RESPONSE = "empty_response"
    PARSE_ERROR = "parse_error"
    VALIDATION_ERROR = "validation_error"
    DUPLICATE = "duplicate"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_INVALID_OUTPUT = "llm_invalid_output"
    UNKNOWN = "unknown"


class NewsDeduplicationReason(StrEnum):
    """
    Why a news item was considered duplicate.
    """

    URL = "url"
    CANONICAL_URL = "canonical_url"
    TITLE_HASH = "title_hash"
    CONTENT_HASH = "content_hash"
    TITLE_SIMILARITY = "title_similarity"
    CONTENT_SIMILARITY = "content_similarity"
    SOURCE_ITEM_ID = "source_item_id"
    UNKNOWN = "unknown"


class NewsAlertType(StrEnum):
    """
    Alert types emitted by the news service.
    """

    HIGH_IMPACT = "high_impact"
    BREAKING_NEWS = "breaking_news"
    REGULATORY_RISK = "regulatory_risk"
    SECURITY_INCIDENT = "security_incident"
    EXCHANGE_INCIDENT = "exchange_incident"
    LISTING_EVENT = "listing_event"
    DELISTING_EVENT = "delisting_event"
    MACRO_EVENT = "macro_event"
    RUMOR = "rumor"
    GENERAL = "general"


class LLMProvider(StrEnum):
    """
    Supported LLM providers for optional news analysis.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class LLMOutputStatus(StrEnum):
    """
    Result status of an LLM scoring/explanation call.
    """

    SUCCESS = "success"
    DISABLED = "disabled"
    SKIPPED = "skipped"
    FAILED = "failed"
    INVALID_JSON = "invalid_json"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    FALLBACK_USED = "fallback_used"


__all__ = [
    "StrEnum",
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
]