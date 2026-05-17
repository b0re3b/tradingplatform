from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import NewsAIConfig
from .enums import (
    LLMOutputStatus,
    NewsFailureReason,
    NewsImpactLevel,
    NewsProcessingStage,
)
from .exceptions import NewsAIError, NewsErrorContext, NewsPublishError
from .models import (
    NewsLLMResult,
    NewsProcessingResult,
    ScoredNewsItem,
    utc_now,
)
from .news_collector import NewsCollector
from .news_deduplicator import NewsDeduplicator
from .news_features import NewsFeatureExtractor
from .news_llm import NewsLLMClient
from .news_processor import NewsProcessor
from .news_scorer import NewsScorer


@dataclass(slots=True, frozen=True)
class NewsServiceRunResult:
    """
    Result of one full NewsAIService pipeline run.
    """

    started_at: Any
    finished_at: Any
    collected_count: int = 0
    raw_unique_count: int = 0
    processed_count: int = 0
    normalized_unique_count: int = 0
    scored_count: int = 0
    high_impact_count: int = 0
    failed_count: int = 0
    errors: tuple[str, ...] = ()
    processing_results: tuple[NewsProcessingResult, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.finished_at - self.started_at).total_seconds() * 1000.0

    @property
    def is_successful(self) -> bool:
        return self.failed_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_ms": self.duration_ms,
            "collected_count": self.collected_count,
            "raw_unique_count": self.raw_unique_count,
            "processed_count": self.processed_count,
            "normalized_unique_count": self.normalized_unique_count,
            "scored_count": self.scored_count,
            "high_impact_count": self.high_impact_count,
            "failed_count": self.failed_count,
            "errors": list(self.errors),
            "processing_results": [
                result.to_dict() for result in self.processing_results
            ],
            "metadata": self.metadata,
        }


class NewsAIService:
    """
    Main facade for the AI/news intelligence pipeline.

    This is the only AI/news class that should directly use EventBus and
    Scheduler. Other classes stay focused on their local responsibilities.
    """

    def __init__(
        self,
        event_bus: EventBus,
        scheduler: Scheduler,
        config: NewsAIConfig,
        *,
        collector: NewsCollector | None = None,
        deduplicator: NewsDeduplicator | None = None,
        processor: NewsProcessor | None = None,
        feature_extractor: NewsFeatureExtractor | None = None,
        llm_client: NewsLLMClient | None = None,
        scorer: NewsScorer | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.config = config
        self.logger = get_logger(__name__)

        self.collector = collector or NewsCollector(config)
        self.deduplicator = deduplicator or NewsDeduplicator(config.deduplication)
        self.processor = processor or NewsProcessor(config)
        self.feature_extractor = feature_extractor or NewsFeatureExtractor(config)
        self.llm_client = llm_client or NewsLLMClient(config.llm)
        self.scorer = scorer or NewsScorer(config)

        self._registered = False
        self._running = False
        self._run_lock = asyncio.Lock()

        self._total_runs = 0
        self._successful_runs = 0
        self._failed_runs = 0
        self._total_collected = 0
        self._total_processed = 0
        self._total_scored = 0
        self._total_high_impact = 0
        self._last_run_at = None
        self._last_success_at = None
        self._last_error: str | None = None
        self._last_result: NewsServiceRunResult | None = None

    @property
    def is_registered(self) -> bool:
        return self._registered

    @property
    def is_running(self) -> bool:
        return self._running

    def register(self) -> None:
        """
        Register periodic news collection job in Scheduler.

        The service does not create uncontrolled asyncio loops. All periodic
        execution goes through core Scheduler.
        """

        if self._registered:
            return

        if not self.config.enabled:
            self.logger.info(
                "News AI service is disabled; scheduler job was not registered",
                extra={"service_name": self.config.service_name},
            )
            self._registered = True
            return

        self.scheduler.add_interval_job(
            name=f"{self.config.service_name}.collect",
            interval_seconds=self.config.collect_interval_seconds,
            func=self.collect_once,
            run_immediately=self.config.startup_collect_enabled,
            max_retries=1,
            retry_delay=2.0,
            timeout=max(30.0, self.config.collect_interval_seconds),
            allow_overlap=False,
        )

        self._registered = True

        self.logger.info(
            "News AI service registered",
            extra={
                "service_name": self.config.service_name,
                "collect_interval_seconds": self.config.collect_interval_seconds,
                "source_count": self.collector.source_count,
                "enabled_source_count": self.collector.enabled_source_count,
            },
        )

    async def collect_once(self) -> NewsServiceRunResult:
        """
        Run one full news intelligence pipeline cycle.

        This method is safe to call manually and is also used by Scheduler.
        """

        if not self.config.enabled:
            started_at = utc_now()
            result = NewsServiceRunResult(
                started_at=started_at,
                finished_at=utc_now(),
                errors=("News AI service is disabled",),
                metadata={"enabled": False},
            )
            self._last_result = result
            return result

        if self._run_lock.locked():
            started_at = utc_now()
            result = NewsServiceRunResult(
                started_at=started_at,
                finished_at=utc_now(),
                errors=("News AI service run skipped because previous run is active",),
                metadata={"skipped": True},
            )
            self._last_result = result
            return result

        async with self._run_lock:
            self._running = True
            started_at = utc_now()
            self._last_run_at = started_at
            self._total_runs += 1

            errors: list[str] = []
            processing_results: list[NewsProcessingResult] = []

            collected_count = 0
            raw_unique_count = 0
            processed_count = 0
            normalized_unique_count = 0
            scored_count = 0
            high_impact_count = 0
            failed_count = 0

            try:
                collection_result = await self.collector.collect()
                processing_results.append(collection_result.processing_result)

                raw_items = list(collection_result.items)
                collected_count = len(raw_items)
                self._total_collected += collected_count

                if collection_result.errors:
                    errors.extend(collection_result.errors)

                if self.config.publish_raw_fetched_event:
                    await self._emit(
                        "news.raw_fetched",
                        {
                            "batch": collection_result.batch.to_dict(),
                            "collector": collection_result.processing_result.to_dict(),
                        },
                        priority=EventPriority.NORMAL,
                    )

                raw_unique_items = self.deduplicator.filter_new_raw(raw_items)
                raw_unique_count = len(raw_unique_items)

                processed_batch = self.processor.process_many(raw_unique_items)
                normalized_items = list(processed_batch.items)
                processed_count = processed_batch.processed_count
                failed_count += processed_batch.failed_count
                errors.extend(processed_batch.errors)

                processing_results.append(
                    NewsProcessingResult(
                        stage=NewsProcessingStage.PROCESS,
                        started_at=started_at,
                        finished_at=utc_now(),
                        raw_count=raw_unique_count,
                        processed_count=processed_batch.processed_count,
                        duplicate_count=raw_unique_count - processed_batch.processed_count,
                        failed_count=processed_batch.failed_count,
                        skipped_count=0,
                        errors=processed_batch.errors,
                        metadata={
                            "component": "NewsProcessor",
                        },
                    )
                )

                normalized_unique_items = self.deduplicator.filter_new_normalized(
                    normalized_items
                )
                normalized_unique_count = len(normalized_unique_items)

                scored_items: list[ScoredNewsItem] = []

                for item in normalized_unique_items:
                    try:
                        features = self.feature_extractor.extract(item)
                        llm_result = await self._analyze_with_llm(item, features)
                        score = self.scorer.score(
                            item=item,
                            features=features,
                            llm_result=llm_result,
                        )

                        scored_item = ScoredNewsItem(
                            item=item,
                            features=features,
                            score=score,
                            llm_result=llm_result,
                        )
                        scored_items.append(scored_item)
                        scored_count += 1
                        self._total_scored += 1

                        self.deduplicator.remember_normalized(item)

                        if self.config.publish_scored_event:
                            await self._publish_scored(scored_item)

                        if self._is_high_impact(scored_item):
                            high_impact_count += 1
                            self._total_high_impact += 1

                            if self.config.publish_high_impact_event:
                                await self._publish_high_impact(scored_item)

                    except NewsAIError as exc:
                        failed_count += 1
                        errors.append(str(exc))
                        await self._publish_failure(
                            topic="news.scoring_failed",
                            exc=exc,
                            fallback_context={
                                "news_id": item.news_id,
                                "source_name": item.source_name,
                                "url": item.url,
                                "title": item.title,
                            },
                        )

                    except Exception as exc:
                        failed_count += 1
                        message = f"Unexpected scoring pipeline failure: {exc}"
                        errors.append(message)

                        await self._publish_failure_payload(
                            topic="news.scoring_failed",
                            payload={
                                "error_type": exc.__class__.__name__,
                                "message": str(exc),
                                "context": {
                                    "news_id": item.news_id,
                                    "source_name": item.source_name,
                                    "url": item.url,
                                    "title": item.title,
                                },
                            },
                        )

                self._total_processed += processed_count

                finished_at = utc_now()
                result = NewsServiceRunResult(
                    started_at=started_at,
                    finished_at=finished_at,
                    collected_count=collected_count,
                    raw_unique_count=raw_unique_count,
                    processed_count=processed_count,
                    normalized_unique_count=normalized_unique_count,
                    scored_count=scored_count,
                    high_impact_count=high_impact_count,
                    failed_count=failed_count,
                    errors=tuple(errors),
                    processing_results=tuple(processing_results),
                    metadata={
                        "service_name": self.config.service_name,
                        "collector_stats": self.collector.stats(),
                        "deduplicator_stats": self.deduplicator.stats(),
                        "llm_enabled": self.config.llm.enabled,
                    },
                )

                self._last_result = result

                if failed_count > 0:
                    self._failed_runs += 1
                    self._last_error = "; ".join(errors[-5:]) if errors else None
                else:
                    self._successful_runs += 1
                    self._last_error = None
                    self._last_success_at = finished_at

                return result

            except NewsAIError as exc:
                self._failed_runs += 1
                self._last_error = str(exc)
                failed_count += 1
                errors.append(str(exc))

                await self._publish_failure(
                    topic="news.pipeline_failed",
                    exc=exc,
                    fallback_context={
                        "service_name": self.config.service_name,
                    },
                )

                result = NewsServiceRunResult(
                    started_at=started_at,
                    finished_at=utc_now(),
                    collected_count=collected_count,
                    raw_unique_count=raw_unique_count,
                    processed_count=processed_count,
                    normalized_unique_count=normalized_unique_count,
                    scored_count=scored_count,
                    high_impact_count=high_impact_count,
                    failed_count=failed_count,
                    errors=tuple(errors),
                    processing_results=tuple(processing_results),
                    metadata={
                        "service_name": self.config.service_name,
                        "pipeline_failed": True,
                    },
                )
                self._last_result = result
                return result

            except Exception as exc:
                self._failed_runs += 1
                self._last_error = str(exc)
                failed_count += 1
                errors.append(str(exc))

                await self._publish_failure_payload(
                    topic="news.pipeline_failed",
                    payload={
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                        "context": {
                            "service_name": self.config.service_name,
                        },
                    },
                )

                result = NewsServiceRunResult(
                    started_at=started_at,
                    finished_at=utc_now(),
                    collected_count=collected_count,
                    raw_unique_count=raw_unique_count,
                    processed_count=processed_count,
                    normalized_unique_count=normalized_unique_count,
                    scored_count=scored_count,
                    high_impact_count=high_impact_count,
                    failed_count=failed_count,
                    errors=tuple(errors),
                    processing_results=tuple(processing_results),
                    metadata={
                        "service_name": self.config.service_name,
                        "unexpected_pipeline_failed": True,
                    },
                )
                self._last_result = result
                return result

            finally:
                self._running = False

    async def run_now(self) -> NewsServiceRunResult:
        """
        Manually trigger one collection/scoring cycle.
        """

        return await self.collect_once()

    def stats(self) -> dict[str, Any]:
        """
        Return service runtime stats.
        """

        return {
            "service_name": self.config.service_name,
            "enabled": self.config.enabled,
            "registered": self._registered,
            "running": self._running,
            "total_runs": self._total_runs,
            "successful_runs": self._successful_runs,
            "failed_runs": self._failed_runs,
            "total_collected": self._total_collected,
            "total_processed": self._total_processed,
            "total_scored": self._total_scored,
            "total_high_impact": self._total_high_impact,
            "last_run_at": self._last_run_at.isoformat() if self._last_run_at else None,
            "last_success_at": (
                self._last_success_at.isoformat()
                if self._last_success_at
                else None
            ),
            "last_error": self._last_error,
            "collector": self.collector.stats(),
            "deduplicator": self.deduplicator.stats(),
            "last_result": self._last_result.to_dict() if self._last_result else None,
        }

    async def _analyze_with_llm(
        self,
        item,
        features,
    ) -> NewsLLMResult:
        if not self.config.llm.enabled:
            return NewsLLMResult(
                status=LLMOutputStatus.DISABLED,
                provider=self.config.llm.provider,
                model=self.config.llm.model,
                error="LLM disabled by config",
            )

        return await self.llm_client.analyze(item=item, features=features)

    async def _publish_scored(self, scored_item: ScoredNewsItem) -> None:
        await self._emit(
            "news.scored",
            scored_item.to_event_payload(),
            priority=EventPriority.NORMAL,
        )

    async def _publish_high_impact(self, scored_item: ScoredNewsItem) -> None:
        priority = EventPriority.HIGH

        if scored_item.score.impact_level == NewsImpactLevel.CRITICAL:
            priority = EventPriority.CRITICAL

        payload = scored_item.to_event_payload()
        payload["alert"] = {
            "type": "high_impact_news",
            "manual_review_only": True,
            "message": self._high_impact_message(scored_item),
        }

        await self._emit(
            "news.high_impact",
            payload,
            priority=priority,
        )

        await self._emit(
            "dashboard.news_update",
            payload,
            priority=priority,
        )

        await self._emit(
            "bot.news_alert",
            payload,
            priority=priority,
        )

    async def _publish_failure(
        self,
        *,
        topic: str,
        exc: NewsAIError,
        fallback_context: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.publish_failed_events:
            return

        payload = exc.as_event_payload()

        if fallback_context:
            payload.setdefault("fallback_context", fallback_context)

        await self._emit(
            topic,
            payload,
            priority=EventPriority.HIGH,
        )

    async def _publish_failure_payload(
        self,
        *,
        topic: str,
        payload: dict[str, Any],
    ) -> None:
        if not self.config.publish_failed_events:
            return

        await self._emit(
            topic,
            payload,
            priority=EventPriority.HIGH,
        )

    async def _emit(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        try:
            await self.event_bus.emit(
                topic,
                payload,
                priority=priority,
                source=self.config.service_name,
            )
        except Exception as exc:
            raise NewsPublishError(
                f"Failed to publish news event '{topic}'",
                context=NewsErrorContext(
                    stage=NewsProcessingStage.PUBLISH,
                    reason=NewsFailureReason.UNKNOWN,
                    source_name=self.config.service_name,
                    details={
                        "topic": topic,
                    },
                ),
                cause=exc,
            ) from exc

    def _is_high_impact(self, scored_item: ScoredNewsItem) -> bool:
        score = scored_item.score

        if score.impact_level in {
            NewsImpactLevel.HIGH,
            NewsImpactLevel.CRITICAL,
        }:
            return True

        return (
            score.impact_score >= self.config.scoring.high_impact_threshold
            and score.relevance_score >= self.config.scoring.min_relevance_score
            and score.confidence_score >= self.config.scoring.min_confidence_score
        )

    def _high_impact_message(self, scored_item: ScoredNewsItem) -> str:
        item = scored_item.item
        score = scored_item.score

        symbols = ", ".join(item.symbols[:5]) if item.symbols else "no tracked symbol"

        return (
            f"[{score.impact_level}] {symbols} | "
            f"bias={score.market_bias} | "
            f"impact={score.impact_score:.2f} | "
            f"confidence={score.confidence_score:.2f} | "
            f"{item.title}"
        )


__all__ = [
    "NewsServiceRunResult",
    "NewsAIService",
]