# tests/ai/news/conftest.py

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from ai import (
    DeduplicationDecision,
    LLMOutputStatus,
    LLMProvider,
    NewsAIConfig,
    NewsAlertType,
    NewsBatch,
    NewsCategory,
    NewsCollectionResult,
    NewsDeduplicationReason,
    NewsEntity,
    NewsEntityType,
    NewsFeatures,
    NewsImpactLevel,
    NewsLanguage,
    NewsLLMResult,
    NewsMarketBias,
    NewsProcessingResult,
    NewsProcessingStage,
    NewsRelevanceLevel,
    NewsScore,
    NewsSentiment,
    NewsSourceConfig,
    NewsSourceHealth,
    NewsSourceStatus,
    NewsSourceType,
    NewsTimeHorizon,
    NewsUrgencyLevel,
    NormalizedNewsItem,
    RawNewsItem,
    build_default_news_ai_config,
    utc_now,
)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def news_source_config() -> NewsSourceConfig:
    return NewsSourceConfig(
        name="test_source",
        source_type=NewsSourceType.RSS,
        url="https://example.com/rss.xml",
        request_timeout_seconds=1.0,
        max_items_per_fetch=10,
        min_fetch_interval_seconds=0.0,
        default_language=NewsLanguage.EN,
        default_categories=(NewsCategory.GENERAL,),
        source_reputation_score=0.75,
        is_official_source=False,
        is_exchange_source=False,
    )


@pytest.fixture
def official_news_source_config() -> NewsSourceConfig:
    return NewsSourceConfig(
        name="sec_press_releases",
        source_type=NewsSourceType.RSS,
        url="https://www.sec.gov/news/pressreleases.rss",
        request_timeout_seconds=1.0,
        max_items_per_fetch=10,
        min_fetch_interval_seconds=0.0,
        default_language=NewsLanguage.EN,
        default_categories=(NewsCategory.REGULATION,),
        source_reputation_score=0.96,
        is_official_source=True,
        is_exchange_source=False,
    )


@pytest.fixture
def exchange_news_source_config() -> NewsSourceConfig:
    return NewsSourceConfig(
        name="binance_announcements",
        source_type=NewsSourceType.EXCHANGE_ANNOUNCEMENT,
        url="https://www.binance.com/en/support/announcement",
        request_timeout_seconds=1.0,
        max_items_per_fetch=10,
        min_fetch_interval_seconds=0.0,
        default_language=NewsLanguage.EN,
        default_categories=(NewsCategory.EXCHANGE,),
        source_reputation_score=0.90,
        is_official_source=True,
        is_exchange_source=True,
    )


@pytest.fixture
def news_config(
    news_source_config: NewsSourceConfig,
    official_news_source_config: NewsSourceConfig,
    exchange_news_source_config: NewsSourceConfig,
) -> NewsAIConfig:
    """
    Conservative test config.

    LLM is disabled by default so tests stay deterministic and do not require
    network/API keys.
    """

    base = build_default_news_ai_config()

    return replace(
        base,
        enabled=True,
        collect_interval_seconds=10.0,
        startup_collect_enabled=False,
        publish_raw_fetched_event=True,
        publish_scored_event=True,
        publish_high_impact_event=True,
        publish_failed_events=True,
        max_items_per_cycle=50,
        max_concurrent_sources=2,
        tracked_symbols=("BTC", "ETH", "SOL", "BNB"),
        source_configs=(
            news_source_config,
            official_news_source_config,
            exchange_news_source_config,
        ),
        service_name="test_news_ai_service",
        metadata={"env": "test"},
    )


@pytest.fixture
def disabled_news_config(news_config: NewsAIConfig) -> NewsAIConfig:
    return replace(news_config, enabled=False)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_regulation_news() -> RawNewsItem:
    return RawNewsItem(
        source_name="sec_press_releases",
        source_type=NewsSourceType.RSS,
        title="SEC sues Binance over alleged securities violations",
        url="https://example.com/sec-binance?utm_source=x&fbclid=y&id=123",
        summary="Bitcoin and Ethereum markets reacted to the lawsuit.",
        body="<p>The regulator announced a new enforcement action.</p>",
        author="SEC",
        source_item_id="sec-001",
        language=NewsLanguage.EN,
        raw_payload={"fixture": "raw_regulation_news"},
    )


@pytest.fixture
def raw_hack_news() -> RawNewsItem:
    return RawNewsItem(
        source_name="test_source",
        source_type=NewsSourceType.RSS,
        title="Breaking: Ethereum DeFi protocol hacked for $120M",
        url="https://example.com/eth-defi-hack",
        summary="Exploit drained funds from a major protocol.",
        body="ETH volatility increased after the security incident.",
        source_item_id="hack-001",
        language=NewsLanguage.EN,
        raw_payload={"fixture": "raw_hack_news"},
    )


@pytest.fixture
def raw_listing_news() -> RawNewsItem:
    return RawNewsItem(
        source_name="binance_announcements",
        source_type=NewsSourceType.EXCHANGE_ANNOUNCEMENT,
        title="Binance will list TEST token, trading opens today",
        url="https://example.com/binance-listing",
        summary="The exchange adds support for TEST/USDT spot trading.",
        source_item_id="listing-001",
        language=NewsLanguage.EN,
        raw_payload={"fixture": "raw_listing_news"},
    )


@pytest.fixture
def normalized_hack_news() -> NormalizedNewsItem:
    return NormalizedNewsItem(
        news_id="news_hack_001",
        source_name="test_source",
        source_type=NewsSourceType.RSS,
        title="Breaking: Ethereum DeFi protocol hacked for $120M",
        text=(
            "Breaking: Ethereum DeFi protocol hacked for $120M\n"
            "Exploit drained funds from a major protocol. ETH volatility increased."
        ),
        url="https://example.com/eth-defi-hack",
        canonical_url="https://example.com/eth-defi-hack",
        summary="Exploit drained funds from a major protocol.",
        source_item_id="hack-001",
        language=NewsLanguage.EN,
        categories=(NewsCategory.HACK, NewsCategory.EXPLOIT, NewsCategory.SECURITY),
        entities=(
            NewsEntity(
                name="Ethereum",
                entity_type=NewsEntityType.PROJECT,
                symbol="ETH",
                confidence=0.95,
            ),
        ),
        symbols=("ETH",),
        title_hash="title_hash_hack_001",
        content_hash="content_hash_hack_001",
        source_reputation_score=0.75,
        metadata={"fixture": "normalized_hack_news"},
    )


@pytest.fixture
def normalized_regulation_news() -> NormalizedNewsItem:
    return NormalizedNewsItem(
        news_id="news_regulation_001",
        source_name="sec_press_releases",
        source_type=NewsSourceType.RSS,
        title="SEC sues Binance over alleged securities violations",
        text=(
            "SEC sues Binance over alleged securities violations\n"
            "Bitcoin and Ethereum markets reacted to the lawsuit."
        ),
        url="https://example.com/sec-binance?id=123",
        canonical_url="https://example.com/sec-binance?id=123",
        summary="Bitcoin and Ethereum markets reacted to the lawsuit.",
        source_item_id="sec-001",
        language=NewsLanguage.EN,
        categories=(NewsCategory.REGULATION, NewsCategory.LEGAL),
        entities=(
            NewsEntity(
                name="SEC",
                entity_type=NewsEntityType.REGULATOR,
                confidence=0.98,
            ),
            NewsEntity(
                name="Binance",
                entity_type=NewsEntityType.EXCHANGE,
                symbol="BNB",
                confidence=0.95,
            ),
        ),
        symbols=("BTC", "ETH", "BNB"),
        title_hash="title_hash_regulation_001",
        content_hash="content_hash_regulation_001",
        source_reputation_score=0.96,
        metadata={"fixture": "normalized_regulation_news"},
    )


@pytest.fixture
def hack_features(normalized_hack_news: NormalizedNewsItem) -> NewsFeatures:
    return NewsFeatures(
        news_id=normalized_hack_news.news_id,
        source_reputation_score=0.75,
        title_strength_score=0.90,
        text_length_score=0.65,
        symbol_count=1,
        entity_count=1,
        category_count=3,
        has_urgent_keywords=True,
        has_hack_keywords=True,
        has_exploit_keywords=True,
        is_breaking_news=True,
        matched_keywords=("breaking", "hack", "exploit", "drained"),
        matched_negative_keywords=("hack", "exploit"),
        matched_positive_keywords=(),
        raw_feature_values={
            "money_amount_detected": True,
            "large_amount_usd": 120_000_000,
        },
    )


@pytest.fixture
def regulation_features(normalized_regulation_news: NormalizedNewsItem) -> NewsFeatures:
    return NewsFeatures(
        news_id=normalized_regulation_news.news_id,
        source_reputation_score=0.96,
        title_strength_score=0.80,
        text_length_score=0.55,
        symbol_count=3,
        entity_count=2,
        category_count=2,
        has_regulatory_keywords=True,
        has_lawsuit_keywords=True,
        is_official_source=True,
        matched_keywords=("sec", "sues", "securities", "lawsuit"),
        matched_negative_keywords=("lawsuit",),
        matched_positive_keywords=(),
        raw_feature_values={"official_regulator_source": True},
    )


@pytest.fixture
def disabled_llm_result() -> NewsLLMResult:
    return NewsLLMResult(
        status=LLMOutputStatus.DISABLED,
        provider=LLMProvider.DISABLED,
        model=None,
        error="LLM disabled for deterministic tests",
    )


@pytest.fixture
def successful_hack_llm_result() -> NewsLLMResult:
    return NewsLLMResult(
        status=LLMOutputStatus.SUCCESS,
        provider=LLMProvider.LOCAL,
        model="test-model",
        sentiment_score=-0.85,
        impact_score=0.95,
        confidence_score=0.90,
        urgency_score=0.92,
        relevance_score=0.88,
        sentiment=NewsSentiment.VERY_BEARISH,
        market_bias=NewsMarketBias.BEARISH,
        time_horizon=NewsTimeHorizon.IMMEDIATE,
        categories=(NewsCategory.HACK, NewsCategory.EXPLOIT, NewsCategory.SECURITY),
        summary="Major DeFi security incident affecting ETH sentiment.",
        explanation="A large exploit can trigger immediate volatility and risk-off flows.",
        trading_notes="Monitor ETH volatility, liquidity, and exchange inflows.",
    )


@pytest.fixture
def high_impact_hack_score(normalized_hack_news: NormalizedNewsItem) -> NewsScore:
    return NewsScore(
        news_id=normalized_hack_news.news_id,
        sentiment_score=-0.85,
        impact_score=0.95,
        confidence_score=0.90,
        urgency_score=0.92,
        novelty_score=1.0,
        relevance_score=0.88,
        source_reputation_score=0.75,
        sentiment=NewsSentiment.VERY_BEARISH,
        market_bias=NewsMarketBias.BEARISH,
        impact_level=NewsImpactLevel.HIGH,
        urgency_level=NewsUrgencyLevel.CRITICAL,
        relevance_level=NewsRelevanceLevel.HIGH,
        time_horizon=NewsTimeHorizon.IMMEDIATE,
        categories=(NewsCategory.HACK, NewsCategory.EXPLOIT, NewsCategory.SECURITY),
        alert_types=(
            NewsAlertType.HIGH_IMPACT,
            NewsAlertType.BREAKING_NEWS,
            NewsAlertType.SECURITY_INCIDENT,
        ),
        summary="Ethereum DeFi protocol hacked for $120M.",
        explanation="Security exploit with high impact and urgent market relevance.",
        trading_notes="Watch ETH volatility and liquidity conditions.",
        rule_score_weight=1.0,
        llm_score_weight=0.0,
        llm_status=LLMOutputStatus.DISABLED,
        metadata={"fixture": "high_impact_hack_score"},
    )


@pytest.fixture
def medium_impact_regulation_score(
    normalized_regulation_news: NormalizedNewsItem,
) -> NewsScore:
    return NewsScore(
        news_id=normalized_regulation_news.news_id,
        sentiment_score=-0.45,
        impact_score=0.70,
        confidence_score=0.82,
        urgency_score=0.65,
        novelty_score=0.80,
        relevance_score=0.78,
        source_reputation_score=0.96,
        sentiment=NewsSentiment.BEARISH,
        market_bias=NewsMarketBias.RISK_OFF,
        impact_level=NewsImpactLevel.MEDIUM,
        urgency_level=NewsUrgencyLevel.HIGH,
        relevance_level=NewsRelevanceLevel.HIGH,
        time_horizon=NewsTimeHorizon.INTRADAY,
        categories=(NewsCategory.REGULATION, NewsCategory.LEGAL),
        alert_types=(NewsAlertType.REGULATORY_RISK,),
        summary="SEC action against Binance may pressure crypto sentiment.",
        explanation="Official regulatory action can affect exchange tokens and broad sentiment.",
        trading_notes="Monitor BTC, ETH, and BNB reaction.",
        rule_score_weight=1.0,
        llm_score_weight=0.0,
        llm_status=LLMOutputStatus.DISABLED,
        metadata={"fixture": "medium_impact_regulation_score"},
    )


# ---------------------------------------------------------------------------
# Fake core infrastructure
# ---------------------------------------------------------------------------


class FakeEventBus:
    """
    Minimal async EventBus replacement for service integration tests.

    Captures emitted events without requiring the real core EventBus lifecycle.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.fail_on_emit: bool = False

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: Any = None,
        source: str | None = None,
        **kwargs: Any,
    ) -> None:
        if self.fail_on_emit:
            raise RuntimeError("FakeEventBus configured to fail on emit")

        self.events.append(
            {
                "topic": topic,
                "payload": payload or {},
                "priority": priority,
                "source": source,
                "kwargs": kwargs,
            }
        )

    def topics(self) -> list[str]:
        return [event["topic"] for event in self.events]

    def events_by_topic(self, topic: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["topic"] == topic]

    def last_event(self, topic: str | None = None) -> dict[str, Any] | None:
        if topic is None:
            return self.events[-1] if self.events else None

        matching = self.events_by_topic(topic)
        return matching[-1] if matching else None


class FakeScheduler:
    """
    Minimal Scheduler replacement for register() tests.
    """

    def __init__(self) -> None:
        self.interval_jobs: list[dict[str, Any]] = []

    def add_interval_job(self, **kwargs: Any) -> None:
        self.interval_jobs.append(kwargs)

    @property
    def job_count(self) -> int:
        return len(self.interval_jobs)

    def last_job(self) -> dict[str, Any] | None:
        return self.interval_jobs[-1] if self.interval_jobs else None


@pytest.fixture
def fake_event_bus() -> FakeEventBus:
    return FakeEventBus()


@pytest.fixture
def fake_scheduler() -> FakeScheduler:
    return FakeScheduler()


# ---------------------------------------------------------------------------
# Fake service dependencies
# ---------------------------------------------------------------------------


class FakeNewsCollector:
    def __init__(
        self,
        *,
        items: list[RawNewsItem] | tuple[RawNewsItem, ...] = (),
        errors: tuple[str, ...] = (),
        delay_seconds: float = 0.0,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._items = tuple(items)
        self._errors = errors
        self._delay_seconds = delay_seconds
        self._raise_exc = raise_exc
        self.collect_calls = 0
        self.source_count = 1
        self.enabled_source_count = 1

    async def collect(self) -> NewsCollectionResult:
        self.collect_calls += 1

        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)

        if self._raise_exc is not None:
            raise self._raise_exc

        started_at = utc_now()
        finished_at = utc_now()

        batch = NewsBatch(
            items=self._items,
            source_health=(
                NewsSourceHealth(
                    source_name="fake_source",
                    source_type=NewsSourceType.RSS,
                    status=NewsSourceStatus.HEALTHY,
                ),
            ),
            metadata={"collector": "fake"},
        )

        processing_result = NewsProcessingResult(
            stage=NewsProcessingStage.COLLECT,
            started_at=started_at,
            finished_at=finished_at,
            raw_count=len(self._items),
            processed_count=len(self._items),
            failed_count=len(self._errors),
            errors=self._errors,
            metadata={"component": "FakeNewsCollector"},
        )

        return NewsCollectionResult(
            batch=batch,
            processing_result=processing_result,
            source_health=batch.source_health,
            errors=self._errors,
        )


class FakeNewsDeduplicator:
    """
    Deduplicator fake compatible with NewsAIService.

    raw_duplicates and normalized_duplicates are sets of source_item_id/news_id.
    """

    def __init__(
        self,
        *,
        raw_duplicates: set[str] | None = None,
        normalized_duplicates: set[str] | None = None,
    ) -> None:
        self.raw_duplicates = raw_duplicates or set()
        self.normalized_duplicates = normalized_duplicates or set()

        self.raw_checked: list[RawNewsItem] = []
        self.raw_remembered: list[RawNewsItem] = []
        self.normalized_checked: list[NormalizedNewsItem] = []
        self.normalized_remembered: list[NormalizedNewsItem] = []

    def check_raw(self, item: RawNewsItem) -> DeduplicationDecision:
        self.raw_checked.append(item)

        key = item.source_item_id or item.url or item.title
        if key in self.raw_duplicates:
            return DeduplicationDecision.duplicate(
                reason=NewsDeduplicationReason.SOURCE_ITEM_ID,
                existing_news_id=f"existing_{key}",
                similarity_score=1.0,
            )

        return DeduplicationDecision.unique()

    def check_normalized(self, item: NormalizedNewsItem) -> DeduplicationDecision:
        self.normalized_checked.append(item)

        if item.news_id in self.normalized_duplicates:
            return DeduplicationDecision.duplicate(
                reason=NewsDeduplicationReason.TITLE_HASH,
                existing_news_id=f"existing_{item.news_id}",
                similarity_score=1.0,
            )

        return DeduplicationDecision.unique()

    def remember_raw(self, item: RawNewsItem) -> None:
        self.raw_remembered.append(item)

    def remember_normalized(self, item: NormalizedNewsItem) -> None:
        self.normalized_remembered.append(item)

    def filter_new_raw(self, items: list[RawNewsItem]) -> list[RawNewsItem]:
        unique: list[RawNewsItem] = []

        for item in items:
            decision = self.check_raw(item)
            if decision.is_duplicate:
                continue

            unique.append(item)
            self.remember_raw(item)

        return unique

    def filter_new_normalized(
        self,
        items: list[NormalizedNewsItem],
    ) -> list[NormalizedNewsItem]:
        unique: list[NormalizedNewsItem] = []

        for item in items:
            decision = self.check_normalized(item)
            if decision.is_duplicate:
                continue

            unique.append(item)
            self.remember_normalized(item)

        return unique


class FakeProcessedNewsBatch:
    """
    Minimal ProcessedNewsBatch-compatible object.

    NewsAIService only needs: items, processed_count, failed_count, errors.
    """

    def __init__(
        self,
        *,
        items: list[NormalizedNewsItem] | tuple[NormalizedNewsItem, ...],
        raw_count: int,
        failed_count: int = 0,
        errors: tuple[str, ...] = (),
    ) -> None:
        self.items = tuple(items)
        self.raw_count = raw_count
        self.processed_count = len(self.items)
        self.failed_count = failed_count
        self.errors = errors


class FakeNewsProcessor:
    def __init__(
        self,
        *,
        normalized_items: list[NormalizedNewsItem] | tuple[NormalizedNewsItem, ...],
        failed_count: int = 0,
        errors: tuple[str, ...] = (),
        raise_exc: BaseException | None = None,
    ) -> None:
        self._normalized_items = tuple(normalized_items)
        self._failed_count = failed_count
        self._errors = errors
        self._raise_exc = raise_exc
        self.process_many_calls = 0
        self.received_raw_items: list[RawNewsItem] = []

    def process_many(self, items: list[RawNewsItem]) -> FakeProcessedNewsBatch:
        self.process_many_calls += 1
        self.received_raw_items.extend(items)

        if self._raise_exc is not None:
            raise self._raise_exc

        return FakeProcessedNewsBatch(
            items=self._normalized_items[: len(items)],
            raw_count=len(items),
            failed_count=self._failed_count,
            errors=self._errors,
        )


class FakeNewsFeatureExtractor:
    def __init__(
        self,
        *,
        features_by_news_id: dict[str, NewsFeatures],
        raise_for_news_ids: set[str] | None = None,
    ) -> None:
        self.features_by_news_id = features_by_news_id
        self.raise_for_news_ids = raise_for_news_ids or set()
        self.extract_calls: list[NormalizedNewsItem] = []

    def extract(self, item: NormalizedNewsItem) -> NewsFeatures:
        self.extract_calls.append(item)

        if item.news_id in self.raise_for_news_ids:
            raise RuntimeError(f"Feature extraction failed for {item.news_id}")

        return self.features_by_news_id[item.news_id]


class FakeNewsLLMClient:
    def __init__(
        self,
        *,
        result: NewsLLMResult | None = None,
        results_by_news_id: dict[str, NewsLLMResult] | None = None,
        raise_for_news_ids: set[str] | None = None,
    ) -> None:
        self.result = result or NewsLLMResult(
            status=LLMOutputStatus.DISABLED,
            provider=LLMProvider.DISABLED,
            model=None,
            error="Fake LLM disabled",
        )
        self.results_by_news_id = results_by_news_id or {}
        self.raise_for_news_ids = raise_for_news_ids or set()
        self.analyze_calls: list[tuple[NormalizedNewsItem, NewsFeatures]] = []

    async def analyze(
        self,
        item: NormalizedNewsItem,
        features: NewsFeatures,
        *,
        session: Any = None,
    ) -> NewsLLMResult:
        self.analyze_calls.append((item, features))

        if item.news_id in self.raise_for_news_ids:
            raise RuntimeError(f"LLM failed for {item.news_id}")

        return self.results_by_news_id.get(item.news_id, self.result)


class FakeNewsScorer:
    def __init__(
        self,
        *,
        scores_by_news_id: dict[str, NewsScore],
        raise_for_news_ids: set[str] | None = None,
    ) -> None:
        self.scores_by_news_id = scores_by_news_id
        self.raise_for_news_ids = raise_for_news_ids or set()
        self.score_calls: list[dict[str, Any]] = []

    def score(
        self,
        item: NormalizedNewsItem,
        features: NewsFeatures,
        llm_result: NewsLLMResult | None = None,
    ) -> NewsScore:
        self.score_calls.append(
            {
                "item": item,
                "features": features,
                "llm_result": llm_result,
            }
        )

        if item.news_id in self.raise_for_news_ids:
            raise RuntimeError(f"Scoring failed for {item.news_id}")

        return self.scores_by_news_id[item.news_id]


# ---------------------------------------------------------------------------
# Fake dependency fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_collector(raw_hack_news: RawNewsItem) -> FakeNewsCollector:
    return FakeNewsCollector(items=(raw_hack_news,))


@pytest.fixture
def fake_deduplicator() -> FakeNewsDeduplicator:
    return FakeNewsDeduplicator()


@pytest.fixture
def fake_processor(
    normalized_hack_news: NormalizedNewsItem,
) -> FakeNewsProcessor:
    return FakeNewsProcessor(normalized_items=(normalized_hack_news,))


@pytest.fixture
def fake_feature_extractor(
    normalized_hack_news: NormalizedNewsItem,
    hack_features: NewsFeatures,
) -> FakeNewsFeatureExtractor:
    return FakeNewsFeatureExtractor(
        features_by_news_id={
            normalized_hack_news.news_id: hack_features,
        }
    )


@pytest.fixture
def fake_llm_client(disabled_llm_result: NewsLLMResult) -> FakeNewsLLMClient:
    return FakeNewsLLMClient(result=disabled_llm_result)


@pytest.fixture
def fake_scorer(
    normalized_hack_news: NormalizedNewsItem,
    high_impact_hack_score: NewsScore,
) -> FakeNewsScorer:
    return FakeNewsScorer(
        scores_by_news_id={
            normalized_hack_news.news_id: high_impact_hack_score,
        }
    )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


@pytest.fixture
def make_raw_news_item():
    def _make_raw_news_item(
        *,
        source_name: str = "test_source",
        source_type: NewsSourceType = NewsSourceType.RSS,
        title: str = "Breaking: Ethereum DeFi protocol hacked for $120M",
        url: str | None = "https://example.com/news",
        summary: str | None = "Exploit drained funds from a major protocol.",
        body: str | None = None,
        source_item_id: str | None = "raw-test-id",
        language: NewsLanguage = NewsLanguage.EN,
        raw_payload: dict[str, Any] | None = None,
    ) -> RawNewsItem:
        return RawNewsItem(
            source_name=source_name,
            source_type=source_type,
            title=title,
            url=url,
            summary=summary,
            body=body,
            source_item_id=source_item_id,
            language=language,
            raw_payload=raw_payload or {},
        )

    return _make_raw_news_item


@pytest.fixture
def make_normalized_news_item():
    def _make_normalized_news_item(
        *,
        news_id: str = "news_test_001",
        source_name: str = "test_source",
        source_type: NewsSourceType = NewsSourceType.RSS,
        title: str = "Breaking: Ethereum DeFi protocol hacked for $120M",
        text: str = "Exploit drained funds from a major Ethereum protocol.",
        url: str | None = "https://example.com/news",
        canonical_url: str | None = "https://example.com/news",
        source_item_id: str | None = "raw-test-id",
        language: NewsLanguage = NewsLanguage.EN,
        categories: tuple[NewsCategory, ...] = (NewsCategory.HACK,),
        symbols: tuple[str, ...] = ("ETH",),
        entities: tuple[NewsEntity, ...] = (),
        source_reputation_score: float = 0.75,
        title_hash: str | None = None,
        content_hash: str | None = None,
    ) -> NormalizedNewsItem:
        return NormalizedNewsItem(
            news_id=news_id,
            source_name=source_name,
            source_type=source_type,
            title=title,
            text=text,
            url=url,
            canonical_url=canonical_url,
            source_item_id=source_item_id,
            language=language,
            categories=categories,
            symbols=symbols,
            entities=entities,
            title_hash=title_hash or f"title_hash_{news_id}",
            content_hash=content_hash or f"content_hash_{news_id}",
            source_reputation_score=source_reputation_score,
        )

    return _make_normalized_news_item


@pytest.fixture
def make_features():
    def _make_features(
        *,
        news_id: str = "news_test_001",
        source_reputation_score: float = 0.75,
        has_urgent_keywords: bool = True,
        has_hack_keywords: bool = True,
        has_exploit_keywords: bool = False,
        has_regulatory_keywords: bool = False,
        has_macro_keywords: bool = False,
        has_listing_keywords: bool = False,
        is_official_source: bool = False,
        is_exchange_source: bool = False,
        matched_keywords: tuple[str, ...] = ("breaking", "hack"),
        matched_negative_keywords: tuple[str, ...] = ("hack",),
        matched_positive_keywords: tuple[str, ...] = (),
    ) -> NewsFeatures:
        return NewsFeatures(
            news_id=news_id,
            source_reputation_score=source_reputation_score,
            title_strength_score=0.80,
            text_length_score=0.60,
            symbol_count=1,
            entity_count=1,
            category_count=1,
            has_urgent_keywords=has_urgent_keywords,
            has_hack_keywords=has_hack_keywords,
            has_exploit_keywords=has_exploit_keywords,
            has_regulatory_keywords=has_regulatory_keywords,
            has_macro_keywords=has_macro_keywords,
            has_listing_keywords=has_listing_keywords,
            is_official_source=is_official_source,
            is_exchange_source=is_exchange_source,
            is_breaking_news=has_urgent_keywords,
            matched_keywords=matched_keywords,
            matched_negative_keywords=matched_negative_keywords,
            matched_positive_keywords=matched_positive_keywords,
        )

    return _make_features


@pytest.fixture
def make_score():
    def _make_score(
        *,
        news_id: str = "news_test_001",
        sentiment_score: float = -0.75,
        impact_score: float = 0.90,
        confidence_score: float = 0.85,
        urgency_score: float = 0.85,
        novelty_score: float = 1.0,
        relevance_score: float = 0.85,
        source_reputation_score: float = 0.75,
        sentiment: NewsSentiment = NewsSentiment.BEARISH,
        market_bias: NewsMarketBias = NewsMarketBias.BEARISH,
        impact_level: NewsImpactLevel = NewsImpactLevel.HIGH,
        urgency_level: NewsUrgencyLevel = NewsUrgencyLevel.HIGH,
        relevance_level: NewsRelevanceLevel = NewsRelevanceLevel.HIGH,
        time_horizon: NewsTimeHorizon = NewsTimeHorizon.IMMEDIATE,
        categories: tuple[NewsCategory, ...] = (NewsCategory.HACK,),
        alert_types: tuple[NewsAlertType, ...] = (NewsAlertType.HIGH_IMPACT,),
        llm_status: LLMOutputStatus = LLMOutputStatus.DISABLED,
    ) -> NewsScore:
        return NewsScore(
            news_id=news_id,
            sentiment_score=sentiment_score,
            impact_score=impact_score,
            confidence_score=confidence_score,
            urgency_score=urgency_score,
            novelty_score=novelty_score,
            relevance_score=relevance_score,
            source_reputation_score=source_reputation_score,
            sentiment=sentiment,
            market_bias=market_bias,
            impact_level=impact_level,
            urgency_level=urgency_level,
            relevance_level=relevance_level,
            time_horizon=time_horizon,
            categories=categories,
            alert_types=alert_types,
            summary="Test score summary.",
            explanation="Test score explanation.",
            trading_notes="Test trading notes for manual review.",
            llm_status=llm_status,
        )

    return _make_score


@pytest.fixture
def make_collection_result():
    def _make_collection_result(
        *,
        items: list[RawNewsItem] | tuple[RawNewsItem, ...],
        errors: tuple[str, ...] = (),
        source_health: tuple[NewsSourceHealth, ...] = (),
    ) -> NewsCollectionResult:
        started_at = utc_now()
        finished_at = utc_now()

        batch = NewsBatch(
            items=tuple(items),
            source_health=source_health,
            metadata={"factory": "make_collection_result"},
        )

        processing_result = NewsProcessingResult(
            stage=NewsProcessingStage.COLLECT,
            started_at=started_at,
            finished_at=finished_at,
            raw_count=len(items),
            processed_count=len(items),
            failed_count=len(errors),
            errors=errors,
            metadata={"factory": "make_collection_result"},
        )

        return NewsCollectionResult(
            batch=batch,
            processing_result=processing_result,
            source_health=source_health,
            errors=errors,
        )

    return _make_collection_result