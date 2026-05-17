# test/newstest/test_news_service_integration.py

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
    NewsAIError,
    NewsBatch,
    NewsCategory,
    NewsCollectionResult,
    NewsDeduplicationReason,
    NewsErrorContext,
    NewsFailureReason,
    NewsFetchStatus,
    NewsImpactLevel,
    NewsLanguage,
    NewsLLMConfig,
    NewsLLMResult,
    NewsMarketBias,
    NewsProcessingResult,
    NewsProcessingStage,
    NewsPublishError,
    NewsScoringError,
    NewsServiceRunResult,
    NewsAIService,
    NewsSourceHealth,
    NewsSourceStatus,
    NewsSourceType,
    NormalizedNewsItem,
    RawNewsItem,
    NewsScore,
    utc_now,
)


# ---------------------------------------------------------------------------
# Strict local test doubles
# ---------------------------------------------------------------------------


class SpyEventBus:
    """
    Strict EventBus spy.

    It records all emitted events and can be configured to fail on selected
    topics to test NewsAIService publish-failure behavior.
    """

    def __init__(
        self,
        *,
        fail_topics: set[str] | None = None,
        fail_all: bool = False,
    ) -> None:
        self.events: list[dict[str, Any]] = []
        self.fail_topics = fail_topics or set()
        self.fail_all = fail_all

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: Any = None,
        source: str | None = None,
        **kwargs: Any,
    ) -> None:
        if self.fail_all or topic in self.fail_topics:
            raise RuntimeError(f"EventBus refused topic {topic}")

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

    def last_event(self, topic: str) -> dict[str, Any]:
        matching = self.events_by_topic(topic)
        assert matching, f"No event with topic={topic!r}. Existing topics: {self.topics()}"
        return matching[-1]


class SpyScheduler:
    """
    Strict Scheduler spy for register() contract tests.
    """

    def __init__(self) -> None:
        self.interval_jobs: list[dict[str, Any]] = []

    def add_interval_job(self, **kwargs: Any) -> None:
        self.interval_jobs.append(kwargs)

    @property
    def job_count(self) -> int:
        return len(self.interval_jobs)

    def last_job(self) -> dict[str, Any]:
        assert self.interval_jobs, "No scheduler jobs were registered"
        return self.interval_jobs[-1]


class CollectorDouble:
    def __init__(
        self,
        *,
        items: list[RawNewsItem] | tuple[RawNewsItem, ...] = (),
        errors: tuple[str, ...] = (),
        delay_seconds: float = 0.0,
        raise_exc: BaseException | None = None,
        source_count: int = 1,
        enabled_source_count: int = 1,
    ) -> None:
        self.items = tuple(items)
        self.errors = errors
        self.delay_seconds = delay_seconds
        self.raise_exc = raise_exc
        self.source_count = source_count
        self.enabled_source_count = enabled_source_count
        self.collect_calls = 0

    async def collect(self) -> NewsCollectionResult:
        self.collect_calls += 1

        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)

        if self.raise_exc is not None:
            raise self.raise_exc

        started_at = utc_now()
        finished_at = utc_now()

        health = (
            NewsSourceHealth(
                source_name="collector_double_source",
                source_type=NewsSourceType.RSS,
                status=NewsSourceStatus.HEALTHY,
                last_fetch_status=NewsFetchStatus.SUCCESS if self.items else NewsFetchStatus.EMPTY,
                total_fetches=1,
                successful_fetches=1,
                failed_fetches=0,
                total_items_fetched=len(self.items),
            ),
        )

        batch = NewsBatch(
            items=self.items,
            source_health=health,
            metadata={"collector": "CollectorDouble"},
        )

        processing_result = NewsProcessingResult(
            stage=NewsProcessingStage.COLLECT,
            started_at=started_at,
            finished_at=finished_at,
            raw_count=len(self.items),
            processed_count=len(self.items),
            failed_count=len(self.errors),
            errors=self.errors,
            metadata={
                "component": "CollectorDouble",
                "source_count": self.source_count,
                "enabled_source_count": self.enabled_source_count,
                "failed_source_count": len(self.errors),
            },
        )

        return NewsCollectionResult(
            batch=batch,
            processing_result=processing_result,
            source_health=health,
            errors=self.errors,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "source_count": self.source_count,
            "enabled_source_count": self.enabled_source_count,
            "collect_calls": self.collect_calls,
            "configured_item_count": len(self.items),
            "configured_error_count": len(self.errors),
        }


class DeduplicatorDouble:
    """
    Dedup gate spy.

    raw_duplicate_keys are matched against source_item_id/url/title.
    normalized_duplicate_ids are matched against news_id.
    """

    def __init__(
        self,
        *,
        raw_duplicate_keys: set[str] | None = None,
        normalized_duplicate_ids: set[str] | None = None,
    ) -> None:
        self.raw_duplicate_keys = raw_duplicate_keys or set()
        self.normalized_duplicate_ids = normalized_duplicate_ids or set()

        self.filter_new_raw_calls: list[list[RawNewsItem]] = []
        self.filter_new_normalized_calls: list[list[NormalizedNewsItem]] = []
        self.remember_normalized_calls: list[NormalizedNewsItem] = []

        self.raw_duplicate_count = 0
        self.normalized_duplicate_count = 0

    def check_raw(self, item: RawNewsItem) -> DeduplicationDecision:
        key = item.source_item_id or item.url or item.title

        if key in self.raw_duplicate_keys:
            self.raw_duplicate_count += 1
            return DeduplicationDecision.duplicate(
                reason=NewsDeduplicationReason.SOURCE_ITEM_ID,
                existing_news_id=f"existing_raw_{key}",
                similarity_score=1.0,
            )

        return DeduplicationDecision.unique()

    def check_normalized(self, item: NormalizedNewsItem) -> DeduplicationDecision:
        if item.news_id in self.normalized_duplicate_ids:
            self.normalized_duplicate_count += 1
            return DeduplicationDecision.duplicate(
                reason=NewsDeduplicationReason.TITLE_HASH,
                existing_news_id=f"existing_normalized_{item.news_id}",
                similarity_score=1.0,
            )

        return DeduplicationDecision.unique()

    def filter_new_raw(self, items: list[RawNewsItem]) -> list[RawNewsItem]:
        self.filter_new_raw_calls.append(list(items))

        unique: list[RawNewsItem] = []
        for item in items:
            if self.check_raw(item).is_duplicate:
                continue
            unique.append(item)

        return unique

    def filter_new_normalized(
        self,
        items: list[NormalizedNewsItem],
    ) -> list[NormalizedNewsItem]:
        self.filter_new_normalized_calls.append(list(items))

        unique: list[NormalizedNewsItem] = []
        for item in items:
            if self.check_normalized(item).is_duplicate:
                continue
            unique.append(item)

        return unique

    def remember_normalized(self, item: NormalizedNewsItem) -> None:
        self.remember_normalized_calls.append(item)

    def stats(self) -> dict[str, Any]:
        return {
            "raw_filter_calls": len(self.filter_new_raw_calls),
            "normalized_filter_calls": len(self.filter_new_normalized_calls),
            "remember_normalized_calls": len(self.remember_normalized_calls),
            "raw_duplicate_count": self.raw_duplicate_count,
            "normalized_duplicate_count": self.normalized_duplicate_count,
        }


class ProcessedBatchDouble:
    def __init__(
        self,
        *,
        items: list[NormalizedNewsItem] | tuple[NormalizedNewsItem, ...],
        processed_count: int | None = None,
        failed_count: int = 0,
        errors: tuple[str, ...] = (),
    ) -> None:
        self.items = tuple(items)
        self.processed_count = len(self.items) if processed_count is None else processed_count
        self.failed_count = failed_count
        self.errors = errors


class ProcessorDouble:
    def __init__(
        self,
        *,
        output_items: list[NormalizedNewsItem] | tuple[NormalizedNewsItem, ...],
        failed_count: int = 0,
        errors: tuple[str, ...] = (),
        raise_exc: BaseException | None = None,
    ) -> None:
        self.output_items = tuple(output_items)
        self.failed_count = failed_count
        self.errors = errors
        self.raise_exc = raise_exc
        self.process_many_calls: list[list[RawNewsItem]] = []

    def process_many(self, items: list[RawNewsItem]) -> ProcessedBatchDouble:
        self.process_many_calls.append(list(items))

        if self.raise_exc is not None:
            raise self.raise_exc

        return ProcessedBatchDouble(
            items=self.output_items[: len(items)],
            failed_count=self.failed_count,
            errors=self.errors,
        )


class FeatureExtractorDouble:
    def __init__(
        self,
        *,
        features_by_news_id: dict[str, Any],
        raise_for_news_ids: set[str] | None = None,
    ) -> None:
        self.features_by_news_id = features_by_news_id
        self.raise_for_news_ids = raise_for_news_ids or set()
        self.extract_calls: list[NormalizedNewsItem] = []

    def extract(self, item: NormalizedNewsItem):
        self.extract_calls.append(item)

        if item.news_id in self.raise_for_news_ids:
            raise RuntimeError(f"Feature extraction exploded for {item.news_id}")

        return self.features_by_news_id[item.news_id]


class LLMClientDouble:
    def __init__(
        self,
        *,
        default_result: NewsLLMResult | None = None,
        results_by_news_id: dict[str, NewsLLMResult] | None = None,
        raise_for_news_ids: set[str] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.default_result = default_result or NewsLLMResult(
            status=LLMOutputStatus.DISABLED,
            provider=LLMProvider.DISABLED,
            model=None,
            error="LLM disabled in test double",
        )
        self.results_by_news_id = results_by_news_id or {}
        self.raise_for_news_ids = raise_for_news_ids or set()
        self.delay_seconds = delay_seconds
        self.analyze_calls: list[dict[str, Any]] = []

    async def analyze(self, *, item: NormalizedNewsItem, features: Any) -> NewsLLMResult:
        self.analyze_calls.append({"item": item, "features": features})

        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)

        if item.news_id in self.raise_for_news_ids:
            raise RuntimeError(f"LLM exploded for {item.news_id}")

        return self.results_by_news_id.get(item.news_id, self.default_result)


class ScorerDouble:
    def __init__(
        self,
        *,
        scores_by_news_id: dict[str, NewsScore],
        raise_for_news_ids: set[str] | None = None,
        raise_news_ai_for_news_ids: set[str] | None = None,
    ) -> None:
        self.scores_by_news_id = scores_by_news_id
        self.raise_for_news_ids = raise_for_news_ids or set()
        self.raise_news_ai_for_news_ids = raise_news_ai_for_news_ids or set()
        self.score_calls: list[dict[str, Any]] = []

    def score(
        self,
        *,
        item: NormalizedNewsItem,
        features: Any,
        llm_result: NewsLLMResult | None = None,
    ) -> NewsScore:
        self.score_calls.append(
            {
                "item": item,
                "features": features,
                "llm_result": llm_result,
            }
        )

        if item.news_id in self.raise_news_ai_for_news_ids:
            raise NewsScoringError(
                f"Structured scoring failure for {item.news_id}",
                context=NewsErrorContext(
                    stage=NewsProcessingStage.SCORE,
                    reason=NewsFailureReason.UNKNOWN,
                    news_id=item.news_id,
                    source_name=item.source_name,
                    url=item.url,
                ),
            )

        if item.news_id in self.raise_for_news_ids:
            raise RuntimeError(f"Unexpected scorer crash for {item.news_id}")

        return self.scores_by_news_id[item.news_id]


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_service(
    *,
    config: NewsAIConfig,
    event_bus: SpyEventBus | None = None,
    scheduler: SpyScheduler | None = None,
    collector: CollectorDouble,
    deduplicator: DeduplicatorDouble,
    processor: ProcessorDouble,
    feature_extractor: FeatureExtractorDouble,
    llm_client: LLMClientDouble,
    scorer: ScorerDouble,
) -> tuple[NewsAIService, SpyEventBus, SpyScheduler]:
    bus = event_bus or SpyEventBus()
    sched = scheduler or SpyScheduler()

    service = NewsAIService(
        event_bus=bus,
        scheduler=sched,
        config=config,
        collector=collector,
        deduplicator=deduplicator,
        processor=processor,
        feature_extractor=feature_extractor,
        llm_client=llm_client,
        scorer=scorer,
    )

    return service, bus, sched


def high_impact_topics() -> set[str]:
    return {
        "news.scored",
        "news.high_impact",
        "dashboard.news_update",
        "bot.news_alert",
    }


# ---------------------------------------------------------------------------
# Register / disabled behavior
# ---------------------------------------------------------------------------


def test_service_register_adds_one_non_overlapping_scheduler_job(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    service, _, scheduler = make_service(
        config=news_config,
        collector=CollectorDouble(items=[raw_hack_news]),
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=ScorerDouble(
            scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
        ),
    )

    service.register()
    service.register()

    assert service.is_registered is True
    assert scheduler.job_count == 1

    job = scheduler.last_job()
    assert job["name"] == f"{news_config.service_name}.collect"
    assert job["interval_seconds"] == news_config.collect_interval_seconds
    assert job["func"] == service.collect_once
    assert job["run_immediately"] == news_config.startup_collect_enabled
    assert job["allow_overlap"] is False
    assert job["timeout"] >= 30.0
    assert job["max_retries"] == 1


def test_service_register_disabled_service_marks_registered_without_scheduler_job(
    disabled_news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    service, _, scheduler = make_service(
        config=disabled_news_config,
        collector=CollectorDouble(items=[raw_hack_news]),
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=ScorerDouble(
            scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
        ),
    )

    service.register()

    assert service.is_registered is True
    assert scheduler.job_count == 0


@pytest.mark.asyncio
async def test_service_collect_once_disabled_returns_without_touching_pipeline(
    disabled_news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    collector = CollectorDouble(items=[raw_hack_news])
    processor = ProcessorDouble(output_items=[normalized_hack_news])
    feature_extractor = FeatureExtractorDouble(
        features_by_news_id={normalized_hack_news.news_id: hack_features}
    )
    llm_client = LLMClientDouble(default_result=disabled_llm_result)
    scorer = ScorerDouble(
        scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
    )

    service, event_bus, _ = make_service(
        config=disabled_news_config,
        collector=collector,
        deduplicator=DeduplicatorDouble(),
        processor=processor,
        feature_extractor=feature_extractor,
        llm_client=llm_client,
        scorer=scorer,
    )

    result = await service.collect_once()

    assert result.collected_count == 0
    assert result.scored_count == 0
    assert result.high_impact_count == 0
    assert result.failed_count == 0
    assert result.errors == ("News AI service is disabled",)
    assert result.metadata == {"enabled": False}

    assert collector.collect_calls == 0
    assert processor.process_many_calls == []
    assert feature_extractor.extract_calls == []
    assert llm_client.analyze_calls == []
    assert scorer.score_calls == []
    assert event_bus.events == []

    stats = service.stats()
    assert stats["enabled"] is False
    assert stats["total_runs"] == 0
    assert stats["last_result"]["metadata"] == {"enabled": False}


# ---------------------------------------------------------------------------
# Happy path / event contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_collect_once_high_impact_pipeline_publishes_all_review_events(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    collector = CollectorDouble(items=[raw_hack_news])
    deduplicator = DeduplicatorDouble()
    processor = ProcessorDouble(output_items=[normalized_hack_news])
    feature_extractor = FeatureExtractorDouble(
        features_by_news_id={normalized_hack_news.news_id: hack_features}
    )
    llm_client = LLMClientDouble(default_result=disabled_llm_result)
    scorer = ScorerDouble(
        scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
    )

    service, event_bus, _ = make_service(
        config=news_config,
        collector=collector,
        deduplicator=deduplicator,
        processor=processor,
        feature_extractor=feature_extractor,
        llm_client=llm_client,
        scorer=scorer,
    )

    result = await service.collect_once()

    assert result.is_successful is True
    assert result.collected_count == 1
    assert result.raw_unique_count == 1
    assert result.processed_count == 1
    assert result.normalized_unique_count == 1
    assert result.scored_count == 1
    assert result.high_impact_count == 1
    assert result.failed_count == 0
    assert result.errors == ()

    assert collector.collect_calls == 1
    assert len(deduplicator.filter_new_raw_calls) == 1
    assert len(deduplicator.filter_new_normalized_calls) == 1
    assert len(deduplicator.remember_normalized_calls) == 1
    assert processor.process_many_calls == [[raw_hack_news]]
    assert feature_extractor.extract_calls == [normalized_hack_news]
    assert llm_client.analyze_calls == []
    assert len(scorer.score_calls) == 1
    assert scorer.score_calls[0]["llm_result"].status == LLMOutputStatus.DISABLED

    topics = event_bus.topics()
    assert topics.count("news.raw_fetched") == 1
    assert topics.count("news.scored") == 1
    assert topics.count("news.high_impact") == 1
    assert topics.count("dashboard.news_update") == 1
    assert topics.count("bot.news_alert") == 1

    assert set(topics) == {
        "news.raw_fetched",
        "news.scored",
        "news.high_impact",
        "dashboard.news_update",
        "bot.news_alert",
    }

    for event in event_bus.events:
        assert event["source"] == news_config.service_name
        assert event["payload"], f"Empty payload for {event['topic']}"

    scored_payload = event_bus.last_event("news.scored")["payload"]
    assert scored_payload["news_id"] == normalized_hack_news.news_id
    assert scored_payload["item"]["news_id"] == normalized_hack_news.news_id
    assert scored_payload["score"]["news_id"] == normalized_hack_news.news_id
    assert scored_payload["features"]["news_id"] == normalized_hack_news.news_id

    high_payload = event_bus.last_event("news.high_impact")["payload"]
    assert high_payload["news_id"] == normalized_hack_news.news_id
    assert high_payload["alert"]["type"] == "high_impact_news"
    assert high_payload["alert"]["manual_review_only"] is True
    assert normalized_hack_news.title in high_payload["alert"]["message"]

    dashboard_payload = event_bus.last_event("dashboard.news_update")["payload"]
    bot_payload = event_bus.last_event("bot.news_alert")["payload"]
    assert dashboard_payload == high_payload
    assert bot_payload == high_payload

    stats = service.stats()
    assert stats["total_runs"] == 1
    assert stats["successful_runs"] == 1
    assert stats["failed_runs"] == 0
    assert stats["total_collected"] == 1
    assert stats["total_processed"] == 1
    assert stats["total_scored"] == 1
    assert stats["total_high_impact"] == 1
    assert stats["last_error"] is None
    assert stats["last_result"]["scored_count"] == 1


@pytest.mark.asyncio
async def test_service_collect_once_medium_impact_publishes_scored_but_not_alert_routes(
    news_config,
    raw_regulation_news,
    normalized_regulation_news,
    regulation_features,
    medium_impact_regulation_score,
    disabled_llm_result,
):
    collector = CollectorDouble(items=[raw_regulation_news])
    processor = ProcessorDouble(output_items=[normalized_regulation_news])

    service, event_bus, _ = make_service(
        config=news_config,
        collector=collector,
        deduplicator=DeduplicatorDouble(),
        processor=processor,
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_regulation_news.news_id: regulation_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=ScorerDouble(
            scores_by_news_id={
                normalized_regulation_news.news_id: medium_impact_regulation_score
            }
        ),
    )

    result = await service.collect_once()

    assert result.scored_count == 1
    assert result.high_impact_count == 0
    assert result.failed_count == 0

    topics = event_bus.topics()
    assert "news.raw_fetched" in topics
    assert "news.scored" in topics
    assert "news.high_impact" not in topics
    assert "dashboard.news_update" not in topics
    assert "bot.news_alert" not in topics


@pytest.mark.asyncio
async def test_service_respects_publish_flags_and_can_run_silent_review_pipeline(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    silent_config = replace(
        news_config,
        publish_raw_fetched_event=False,
        publish_scored_event=False,
        publish_high_impact_event=False,
        publish_failed_events=True,
    )

    service, event_bus, _ = make_service(
        config=silent_config,
        collector=CollectorDouble(items=[raw_hack_news]),
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=ScorerDouble(
            scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
        ),
    )

    result = await service.collect_once()

    assert result.scored_count == 1
    assert result.high_impact_count == 1
    assert result.failed_count == 0
    assert event_bus.events == []


# ---------------------------------------------------------------------------
# Dedup gates / expensive work protection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_raw_dedup_skip_prevents_processor_feature_llm_scorer_and_events(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    deduplicator = DeduplicatorDouble(
        raw_duplicate_keys={raw_hack_news.source_item_id}
    )
    processor = ProcessorDouble(output_items=[normalized_hack_news])
    feature_extractor = FeatureExtractorDouble(
        features_by_news_id={normalized_hack_news.news_id: hack_features}
    )
    llm_client = LLMClientDouble(default_result=disabled_llm_result)
    scorer = ScorerDouble(
        scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
    )

    service, event_bus, _ = make_service(
        config=news_config,
        collector=CollectorDouble(items=[raw_hack_news]),
        deduplicator=deduplicator,
        processor=processor,
        feature_extractor=feature_extractor,
        llm_client=llm_client,
        scorer=scorer,
    )

    result = await service.collect_once()

    assert result.collected_count == 1
    assert result.raw_unique_count == 0
    assert result.processed_count == 0
    assert result.normalized_unique_count == 0
    assert result.scored_count == 0
    assert result.high_impact_count == 0
    assert result.failed_count == 0

    assert len(deduplicator.filter_new_raw_calls) == 1
    assert deduplicator.raw_duplicate_count == 1
    assert processor.process_many_calls == [[]]
    assert deduplicator.filter_new_normalized_calls == [[]]
    assert feature_extractor.extract_calls == []
    assert llm_client.analyze_calls == []
    assert scorer.score_calls == []
    assert deduplicator.remember_normalized_calls == []

    assert event_bus.topics() == ["news.raw_fetched"]


@pytest.mark.asyncio
async def test_service_normalized_dedup_skip_prevents_feature_llm_scorer_and_publish(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    deduplicator = DeduplicatorDouble(
        normalized_duplicate_ids={normalized_hack_news.news_id}
    )
    processor = ProcessorDouble(output_items=[normalized_hack_news])
    feature_extractor = FeatureExtractorDouble(
        features_by_news_id={normalized_hack_news.news_id: hack_features}
    )
    llm_client = LLMClientDouble(default_result=disabled_llm_result)
    scorer = ScorerDouble(
        scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
    )

    service, event_bus, _ = make_service(
        config=news_config,
        collector=CollectorDouble(items=[raw_hack_news]),
        deduplicator=deduplicator,
        processor=processor,
        feature_extractor=feature_extractor,
        llm_client=llm_client,
        scorer=scorer,
    )

    result = await service.collect_once()

    assert result.collected_count == 1
    assert result.raw_unique_count == 1
    assert result.processed_count == 1
    assert result.normalized_unique_count == 0
    assert result.scored_count == 0
    assert result.high_impact_count == 0
    assert result.failed_count == 0

    assert processor.process_many_calls == [[raw_hack_news]]
    assert feature_extractor.extract_calls == []
    assert llm_client.analyze_calls == []
    assert scorer.score_calls == []
    assert deduplicator.remember_normalized_calls == []

    assert event_bus.topics() == ["news.raw_fetched"]


# ---------------------------------------------------------------------------
# LLM branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_when_llm_enabled_calls_llm_and_passes_result_to_scorer(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    successful_hack_llm_result,
    high_impact_hack_score,
):
    llm_config = NewsLLMConfig(
        enabled=True,
        provider=LLMProvider.LOCAL,
        model="test-local-model",
        base_url="http://localhost:11434/v1",
        fallback_to_rule_based=True,
    )
    config = replace(news_config, llm=llm_config)

    llm_client = LLMClientDouble(default_result=successful_hack_llm_result)
    scorer = ScorerDouble(
        scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
    )

    service, _, _ = make_service(
        config=config,
        collector=CollectorDouble(items=[raw_hack_news]),
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=llm_client,
        scorer=scorer,
    )

    result = await service.collect_once()

    assert result.scored_count == 1
    assert result.failed_count == 0
    assert len(llm_client.analyze_calls) == 1
    assert llm_client.analyze_calls[0]["item"] == normalized_hack_news
    assert llm_client.analyze_calls[0]["features"] == hack_features

    assert len(scorer.score_calls) == 1
    assert scorer.score_calls[0]["llm_result"] == successful_hack_llm_result
    assert result.metadata["llm_enabled"] is True


@pytest.mark.asyncio
async def test_service_llm_unexpected_failure_isolated_as_scoring_failed_event(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
):
    llm_config = NewsLLMConfig(
        enabled=True,
        provider=LLMProvider.LOCAL,
        model="test-local-model",
        fallback_to_rule_based=True,
    )
    config = replace(news_config, llm=llm_config)

    llm_client = LLMClientDouble(
        raise_for_news_ids={normalized_hack_news.news_id}
    )
    scorer = ScorerDouble(
        scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
    )

    service, event_bus, _ = make_service(
        config=config,
        collector=CollectorDouble(items=[raw_hack_news]),
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=llm_client,
        scorer=scorer,
    )

    result = await service.collect_once()

    assert result.scored_count == 0
    assert result.high_impact_count == 0
    assert result.failed_count == 1
    assert any("Unexpected scoring pipeline failure" in error for error in result.errors)

    assert len(llm_client.analyze_calls) == 1
    assert scorer.score_calls == []
    assert event_bus.topics() == ["news.raw_fetched", "news.scoring_failed"]

    payload = event_bus.last_event("news.scoring_failed")["payload"]
    assert payload["error_type"] == "RuntimeError"
    assert normalized_hack_news.news_id == payload["context"]["news_id"]


# ---------------------------------------------------------------------------
# Failure isolation / pipeline failure behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_processor_partial_failure_marks_run_failed_but_scores_valid_items(
    news_config,
    raw_hack_news,
    raw_regulation_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    processor = ProcessorDouble(
        output_items=[normalized_hack_news],
        failed_count=1,
        errors=("Invalid raw item skipped by processor",),
    )

    service, event_bus, _ = make_service(
        config=news_config,
        collector=CollectorDouble(items=[raw_hack_news, raw_regulation_news]),
        deduplicator=DeduplicatorDouble(),
        processor=processor,
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=ScorerDouble(
            scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
        ),
    )

    result = await service.collect_once()

    assert result.collected_count == 2
    assert result.raw_unique_count == 2
    assert result.processed_count == 1
    assert result.normalized_unique_count == 1
    assert result.scored_count == 1
    assert result.high_impact_count == 1
    assert result.failed_count == 1
    assert "Invalid raw item skipped by processor" in result.errors

    topics = event_bus.topics()
    assert "news.scored" in topics
    assert "news.high_impact" in topics

    stats = service.stats()
    assert stats["successful_runs"] == 0
    assert stats["failed_runs"] == 1
    assert stats["last_error"]


@pytest.mark.asyncio
async def test_service_feature_failure_isolated_per_item_and_does_not_stop_second_item(
    news_config,
    raw_hack_news,
    raw_regulation_news,
    normalized_hack_news,
    normalized_regulation_news,
    hack_features,
    regulation_features,
    high_impact_hack_score,
    medium_impact_regulation_score,
    disabled_llm_result,
):
    raw_items = [raw_hack_news, raw_regulation_news]
    normalized_items = [normalized_hack_news, normalized_regulation_news]

    feature_extractor = FeatureExtractorDouble(
        features_by_news_id={
            normalized_hack_news.news_id: hack_features,
            normalized_regulation_news.news_id: regulation_features,
        },
        raise_for_news_ids={normalized_hack_news.news_id},
    )

    scorer = ScorerDouble(
        scores_by_news_id={
            normalized_hack_news.news_id: high_impact_hack_score,
            normalized_regulation_news.news_id: medium_impact_regulation_score,
        }
    )

    service, event_bus, _ = make_service(
        config=news_config,
        collector=CollectorDouble(items=raw_items),
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=normalized_items),
        feature_extractor=feature_extractor,
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=scorer,
    )

    result = await service.collect_once()

    assert result.collected_count == 2
    assert result.raw_unique_count == 2
    assert result.processed_count == 2
    assert result.normalized_unique_count == 2
    assert result.scored_count == 1
    assert result.high_impact_count == 0
    assert result.failed_count == 1

    assert any("Feature extraction exploded" in error for error in result.errors)
    assert len(feature_extractor.extract_calls) == 2
    assert len(scorer.score_calls) == 1
    assert scorer.score_calls[0]["item"] == normalized_regulation_news

    topics = event_bus.topics()
    assert topics.count("news.scoring_failed") == 1
    assert topics.count("news.scored") == 1
    assert "news.high_impact" not in topics


@pytest.mark.asyncio
async def test_service_structured_news_ai_scoring_error_publishes_safe_failure_payload(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    scorer = ScorerDouble(
        scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score},
        raise_news_ai_for_news_ids={normalized_hack_news.news_id},
    )

    service, event_bus, _ = make_service(
        config=news_config,
        collector=CollectorDouble(items=[raw_hack_news]),
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=scorer,
    )

    result = await service.collect_once()

    assert result.scored_count == 0
    assert result.high_impact_count == 0
    assert result.failed_count == 1
    assert event_bus.topics() == ["news.raw_fetched", "news.scoring_failed"]

    payload = event_bus.last_event("news.scoring_failed")["payload"]
    assert payload["error_type"] == "NewsScoringError"
    assert payload["context"]["news_id"] == normalized_hack_news.news_id
    assert payload["fallback_context"]["news_id"] == normalized_hack_news.news_id
    assert "api_key" not in str(payload).lower()
    assert "token" not in str(payload).lower()


@pytest.mark.asyncio
async def test_service_collector_news_ai_error_becomes_pipeline_failed_event(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    collector_error = NewsAIError(
        "Collector failed before returning a batch",
        context=NewsErrorContext(
            stage=NewsProcessingStage.COLLECT,
            reason=NewsFailureReason.NETWORK_ERROR,
            source_name="collector_double",
        ),
    )

    service, event_bus, _ = make_service(
        config=news_config,
        collector=CollectorDouble(raise_exc=collector_error),
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=ScorerDouble(
            scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
        ),
    )

    result = await service.collect_once()

    assert result.collected_count == 0
    assert result.scored_count == 0
    assert result.high_impact_count == 0
    assert result.failed_count == 1
    assert result.metadata["pipeline_failed"] is True
    assert "Collector failed before returning a batch" in result.errors[0]

    assert event_bus.topics() == ["news.pipeline_failed"]
    payload = event_bus.last_event("news.pipeline_failed")["payload"]
    assert payload["error_type"] == "NewsAIError"
    assert payload["context"]["stage"] == str(NewsProcessingStage.COLLECT)
    assert payload["fallback_context"]["service_name"] == news_config.service_name

    stats = service.stats()
    assert stats["failed_runs"] == 1
    assert stats["successful_runs"] == 0
    assert stats["last_error"]


@pytest.mark.asyncio
async def test_service_collector_unexpected_error_becomes_unexpected_pipeline_failed_event(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    service, event_bus, _ = make_service(
        config=news_config,
        collector=CollectorDouble(raise_exc=RuntimeError("database exploded")),
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=ScorerDouble(
            scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
        ),
    )

    result = await service.collect_once()

    assert result.failed_count == 1
    assert result.metadata["unexpected_pipeline_failed"] is True
    assert "database exploded" in result.errors[0]

    assert event_bus.topics() == ["news.pipeline_failed"]
    payload = event_bus.last_event("news.pipeline_failed")["payload"]
    assert payload["error_type"] == "RuntimeError"
    assert payload["message"] == "database exploded"
    assert payload["context"]["service_name"] == news_config.service_name


@pytest.mark.asyncio
async def test_service_publish_failure_turns_successful_scoring_into_pipeline_failure(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    """
    This test is intentionally unpleasant.

    A single EventBus failure on news.scored currently aborts the whole pipeline
    and is then reported through news.pipeline_failed. This documents that
    publishing is not best-effort at service level.
    """

    event_bus = SpyEventBus(fail_topics={"news.scored"})

    service, event_bus, _ = make_service(
        config=news_config,
        event_bus=event_bus,
        collector=CollectorDouble(items=[raw_hack_news]),
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=ScorerDouble(
            scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
        ),
    )

    result = await service.collect_once()

    assert result.failed_count == 1
    assert result.scored_count == 1
    assert result.high_impact_count == 0

    # Current behavior: publish failure during per-item scoring is counted as an
    # item-level failure, not as a top-level pipeline_failed metadata flag.
    assert "pipeline_failed" not in result.metadata
    assert any("Failed to publish news event 'news.scored'" in error for error in result.errors)

    # news.raw_fetched was emitted before news.scored failed.
    # news.pipeline_failed is emitted after the publish failure is caught.
    assert event_bus.topics() == ["news.raw_fetched", "news.pipeline_failed"]

    payload = event_bus.last_event("news.pipeline_failed")["payload"]
    assert payload["error_type"] == "NewsPublishError"
    assert payload["context"]["details"]["topic"] == "news.scored"


@pytest.mark.asyncio
async def test_service_publish_failed_events_flag_can_hide_pipeline_failure_event(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    config = replace(news_config, publish_failed_events=False)

    service, event_bus, _ = make_service(
        config=config,
        collector=CollectorDouble(raise_exc=RuntimeError("fatal collector crash")),
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=ScorerDouble(
            scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
        ),
    )

    result = await service.collect_once()

    assert result.failed_count == 1
    assert result.metadata["unexpected_pipeline_failed"] is True
    assert event_bus.events == []


# ---------------------------------------------------------------------------
# Locking / run_now / stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_collect_once_run_lock_skips_overlapping_second_run(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    collector = CollectorDouble(items=[raw_hack_news], delay_seconds=0.05)

    service, event_bus, _ = make_service(
        config=news_config,
        collector=collector,
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=ScorerDouble(
            scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
        ),
    )

    first_task = asyncio.create_task(service.collect_once())

    await asyncio.sleep(0.01)
    assert service.is_running is True

    second_result = await service.collect_once()
    first_result = await first_task

    assert second_result.metadata == {"skipped": True}
    assert second_result.errors == (
        "News AI service run skipped because previous run is active",
    )
    assert second_result.failed_count == 0

    assert first_result.scored_count == 1
    assert first_result.high_impact_count == 1
    assert collector.collect_calls == 1

    stats = service.stats()
    assert stats["running"] is False
    assert stats["total_runs"] == 1

    # The skipped result temporarily becomes _last_result, but the first active
    # run finishes later and overwrites it. This is acceptable but worth locking
    # down because it affects observability.
    assert stats["last_result"]["scored_count"] == 1
    assert "news.high_impact" in event_bus.topics()


@pytest.mark.asyncio
async def test_service_run_now_is_exact_alias_for_collect_once(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    service, _, _ = make_service(
        config=news_config,
        collector=CollectorDouble(items=[raw_hack_news]),
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=ScorerDouble(
            scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
        ),
    )

    result = await service.run_now()

    assert isinstance(result, NewsServiceRunResult)
    assert result.scored_count == 1
    assert result.high_impact_count == 1
    assert service.stats()["total_runs"] == 1


@pytest.mark.asyncio
async def test_service_stats_after_mixed_success_and_failure_runs_do_not_hide_last_error(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    collector = CollectorDouble(items=[raw_hack_news])
    scorer = ScorerDouble(
        scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
    )

    service, _, _ = make_service(
        config=news_config,
        collector=collector,
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=scorer,
    )

    success = await service.collect_once()
    assert success.failed_count == 0

    scorer.raise_for_news_ids.add(normalized_hack_news.news_id)

    failure = await service.collect_once()
    assert failure.failed_count == 1

    stats = service.stats()
    assert stats["total_runs"] == 2
    assert stats["successful_runs"] == 1
    assert stats["failed_runs"] == 1
    assert stats["total_collected"] == 2
    assert stats["total_processed"] == 2
    assert stats["total_scored"] == 1
    assert stats["total_high_impact"] == 1
    assert stats["last_error"]
    assert "Unexpected scoring pipeline failure" in stats["last_error"]
    assert stats["last_result"]["failed_count"] == 1


# ---------------------------------------------------------------------------
# Strict architectural guardrails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_never_publishes_trading_signal_risk_or_execution_topics(
    news_config,
    raw_hack_news,
    normalized_hack_news,
    hack_features,
    high_impact_hack_score,
    disabled_llm_result,
):
    service, event_bus, _ = make_service(
        config=news_config,
        collector=CollectorDouble(items=[raw_hack_news]),
        deduplicator=DeduplicatorDouble(),
        processor=ProcessorDouble(output_items=[normalized_hack_news]),
        feature_extractor=FeatureExtractorDouble(
            features_by_news_id={normalized_hack_news.news_id: hack_features}
        ),
        llm_client=LLMClientDouble(default_result=disabled_llm_result),
        scorer=ScorerDouble(
            scores_by_news_id={normalized_hack_news.news_id: high_impact_hack_score}
        ),
    )

    await service.collect_once()

    forbidden_prefixes = (
        "signal.",
        "risk.",
        "execution.",
        "position.",
    )

    for topic in event_bus.topics():
        assert not topic.startswith(forbidden_prefixes), (
            "AI/news package must stay manual-review only and must not publish "
            f"trading-control topic {topic!r}"
        )

    assert set(event_bus.topics()).issubset(
        {
            "news.raw_fetched",
            "news.scored",
            "news.high_impact",
            "dashboard.news_update",
            "bot.news_alert",
            "news.scoring_failed",
            "news.pipeline_failed",
        }
    )