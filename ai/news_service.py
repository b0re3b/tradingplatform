from __future__ import annotations

import asyncio
import hashlib
from collections import deque
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
    NormalizedNewsItem,
    RawNewsItem,
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

    Important deduplication rule:
        The service never commits raw items to the global deduplication memory
        before normalization/scoring. Raw-stage dedup is only a pre-check against
        previously accepted news plus an in-run batch filter.

        The final global dedup commit is done only after a normalized item was
        successfully scored. This prevents the same item from being accepted as
        raw and then immediately rejected as a normalized duplicate in the same
        pipeline run.
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
        self._collect_job_id: str | None = None
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

        self._publish_history: deque[tuple[Any, str]] = deque()
        self._current_cycle_publish_counts: dict[str, int] | None = None
        self._suppressed_publish_counts: dict[str, int] = {}
        self._last_bot_alert_at = None
        self._total_suppressed_publications = 0

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

        self._collect_job_id = self.scheduler.add_interval_job(
            name=f"{self.config.service_name}.collect",
            interval=self.config.collect_interval_seconds,
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

    async def start(self) -> None:
        """
        Start NewsAIService by registering its scheduled collection job.

        app.runtime.start_component() calls start(), not register(), for most
        runtime services. Without this method the news module is constructed
        but never schedules collection, so no news.* events reach Telegram.
        """

        self.register()

        await self._emit(
            "system.news_ai_service.started",
            {
                "service_name": self.config.service_name,
                "enabled": self.config.enabled,
                "collect_interval_seconds": self.config.collect_interval_seconds,
                "startup_collect_enabled": self.config.startup_collect_enabled,
                "source_count": self.collector.source_count,
                "enabled_source_count": self.collector.enabled_source_count,
                "collect_job_id": self._collect_job_id,
            },
            priority=EventPriority.LOW,
        )

    async def stop(self) -> None:
        """Stop NewsAIService and remove its scheduler job."""

        if self._collect_job_id is not None:
            try:
                self.scheduler.remove_job(self._collect_job_id)
            except KeyError:
                pass
            self._collect_job_id = None

        self._registered = False
        self._running = False

        await self._emit(
            "system.news_ai_service.stopped",
            {
                "service_name": self.config.service_name,
                "enabled": self.config.enabled,
                "total_runs": self._total_runs,
                "successful_runs": self._successful_runs,
                "failed_runs": self._failed_runs,
                "total_collected": self._total_collected,
                "total_scored": self._total_scored,
                "total_high_impact": self._total_high_impact,
            },
            priority=EventPriority.LOW,
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
            self._current_cycle_publish_counts = {}
            cycle_suppressed_before = self._total_suppressed_publications

            collected_count = 0
            raw_unique_count = 0
            processed_count = 0
            normalized_unique_count = 0
            scored_count = 0
            high_impact_count = 0
            failed_count = 0
            raw_duplicate_count = 0
            normalized_duplicate_count = 0

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

                raw_unique_items, raw_duplicate_count = self._filter_new_raw_candidates(
                    raw_items
                )
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
                        duplicate_count=raw_duplicate_count,
                        failed_count=processed_batch.failed_count,
                        skipped_count=0,
                        errors=processed_batch.errors,
                        metadata={
                            "component": "NewsProcessor",
                            "raw_duplicate_count": raw_duplicate_count,
                            "dedup_commit_stage": "after_successful_scoring",
                        },
                    )
                )

                (
                    normalized_unique_items,
                    normalized_duplicate_count,
                ) = self._filter_new_normalized_candidates(normalized_items)
                normalized_unique_count = len(normalized_unique_items)

                scoring_work_skipped_count = 0
                if len(normalized_unique_items) > self.config.max_items_to_score_per_cycle:
                    scoring_work_skipped_count = (
                        len(normalized_unique_items) - self.config.max_items_to_score_per_cycle
                    )
                    normalized_unique_items = normalized_unique_items[
                        : self.config.max_items_to_score_per_cycle
                    ]

                    processing_results.append(
                        NewsProcessingResult(
                            stage=NewsProcessingStage.SCORE,
                            started_at=started_at,
                            finished_at=utc_now(),
                            raw_count=normalized_unique_count,
                            processed_count=len(normalized_unique_items),
                            duplicate_count=0,
                            failed_count=0,
                            skipped_count=scoring_work_skipped_count,
                            errors=(),
                            metadata={
                                "component": "NewsAIService",
                                "reason": "max_items_to_score_per_cycle",
                                "max_items_to_score_per_cycle": (
                                    self.config.max_items_to_score_per_cycle
                                ),
                                "dedup_commit": False,
                            },
                        )
                    )

                if normalized_duplicate_count:
                    processing_results.append(
                        NewsProcessingResult(
                            stage=NewsProcessingStage.DEDUPLICATE,
                            started_at=started_at,
                            finished_at=utc_now(),
                            raw_count=len(normalized_items),
                            processed_count=normalized_unique_count,
                            duplicate_count=normalized_duplicate_count,
                            failed_count=0,
                            skipped_count=normalized_duplicate_count,
                            errors=(),
                            metadata={
                                "component": "NewsDeduplicator",
                                "item_type": "NormalizedNewsItem",
                                "commit": False,
                            },
                        )
                    )

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

                        # Final dedup commit happens only after the item passed
                        # normalization, feature extraction, optional LLM analysis,
                        # scoring, and ScoredNewsItem construction.
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
                        "raw_duplicate_count": raw_duplicate_count,
                        "normalized_duplicate_count": normalized_duplicate_count,
                        "scoring_work_skipped_count": scoring_work_skipped_count,
                        "max_items_to_score_per_cycle": self.config.max_items_to_score_per_cycle,
                        "llm_enabled": self.config.llm.enabled,
                        "dedup_commit_stage": "after_successful_scoring",
                        "scored_item_count": len(scored_items),
                        "publication_counts": dict(self._current_cycle_publish_counts or {}),
                        "suppressed_publication_count": (
                            self._total_suppressed_publications - cycle_suppressed_before
                        ),
                        "suppressed_publications_by_topic": dict(self._suppressed_publish_counts),
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
                        "raw_duplicate_count": raw_duplicate_count,
                        "normalized_duplicate_count": normalized_duplicate_count,
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
                        "raw_duplicate_count": raw_duplicate_count,
                        "normalized_duplicate_count": normalized_duplicate_count,
                    },
                )
                self._last_result = result
                return result

            finally:
                if self.config.publish_suppressed_summary_event:
                    try:
                        await self._emit_suppressed_summary(
                            cycle_suppressed_before=cycle_suppressed_before,
                        )
                    except Exception as exc:
                        self.logger.warning(
                            "Failed to publish news publication-limit summary",
                            extra={
                                "service_name": self.config.service_name,
                                "error_type": exc.__class__.__name__,
                                "error": str(exc),
                            },
                        )
                self._current_cycle_publish_counts = None
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
            "collect_job_id": self._collect_job_id,
            "total_runs": self._total_runs,
            "successful_runs": self._successful_runs,
            "failed_runs": self._failed_runs,
            "total_collected": self._total_collected,
            "total_processed": self._total_processed,
            "total_scored": self._total_scored,
            "total_high_impact": self._total_high_impact,
            "total_suppressed_publications": self._total_suppressed_publications,
            "suppressed_publications_by_topic": dict(self._suppressed_publish_counts),
            "publish_window_size": len(self._publish_history),
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

    def _filter_new_raw_candidates(
        self,
        items: list[RawNewsItem],
    ) -> tuple[list[RawNewsItem], int]:
        """
        Pre-filter raw items without committing them to global dedup memory.

        Why not use NewsDeduplicator.filter_new_raw() here?
            filter_new_raw() remembers accepted raw items immediately. The same
            item then appears as NormalizedNewsItem and gets rejected as a
            duplicate before scoring. This method only checks against previous
            dedup state and uses an in-run fingerprint set for same-batch
            duplicates.
        """

        unique: list[RawNewsItem] = []
        seen_in_run: set[str] = set()
        duplicate_count = 0

        for item in items:
            fingerprint = self._raw_item_fingerprint(item)
            if fingerprint in seen_in_run:
                duplicate_count += 1
                continue

            decision = self.deduplicator.check_raw(item)
            if decision.is_duplicate:
                duplicate_count += 1
                continue

            seen_in_run.add(fingerprint)
            unique.append(item)

        return unique, duplicate_count

    def _filter_new_normalized_candidates(
        self,
        items: list[NormalizedNewsItem],
    ) -> tuple[list[NormalizedNewsItem], int]:
        """
        Filter normalized items without committing them before scoring.

        The final commit is done by remember_normalized() only after successful
        scoring. This prevents failed scoring attempts from permanently hiding a
        news item from future runs.
        """

        unique: list[NormalizedNewsItem] = []
        seen_in_run: set[str] = set()
        duplicate_count = 0

        for item in items:
            fingerprint = self._normalized_item_fingerprint(item)
            if fingerprint in seen_in_run:
                duplicate_count += 1
                continue

            decision = self.deduplicator.check_normalized(item)
            if decision.is_duplicate:
                duplicate_count += 1
                continue

            seen_in_run.add(fingerprint)
            unique.append(item)

        return unique, duplicate_count

    def _raw_item_fingerprint(self, item: RawNewsItem) -> str:
        """
        Build an in-run raw item fingerprint.

        This is intentionally local to the service and is not a persistent
        identifier. Persistent deduplication remains NewsDeduplicator's job.
        """

        candidates = (
            f"source_item:{item.source_name}:{item.source_item_id}"
            if item.source_item_id
            else None,
            f"url:{item.url}" if item.url else None,
            f"title:{item.title}",
            f"text:{item.text}",
        )
        return self._stable_fingerprint(candidates)

    def _normalized_item_fingerprint(self, item: NormalizedNewsItem) -> str:
        """
        Build an in-run normalized item fingerprint.

        Prefer the stable news_id generated by NewsProcessor, then canonical
        identifiers. This catches duplicates within the same run before the
        item is committed to global dedup memory.
        """

        candidates = (
            f"news_id:{item.news_id}" if item.news_id else None,
            f"source_item:{item.source_name}:{item.source_item_id}"
            if item.source_item_id
            else None,
            f"canonical_url:{item.canonical_url}" if item.canonical_url else None,
            f"url:{item.url}" if item.url else None,
            f"title_hash:{item.title_hash}" if item.title_hash else None,
            f"content_hash:{item.content_hash}" if item.content_hash else None,
            f"title:{item.title}",
        )
        return self._stable_fingerprint(candidates)

    def _stable_fingerprint(self, candidates: tuple[str | None, ...]) -> str:
        for candidate in candidates:
            if candidate and candidate.strip():
                normalized = " ".join(candidate.strip().lower().split())
                return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

        return hashlib.sha1(str(utc_now().timestamp()).encode("utf-8")).hexdigest()

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
        apply_publication_limits: bool = True,
    ) -> None:
        if apply_publication_limits and not self._allow_publication(topic):
            self._record_suppressed_publication(topic)
            return

        try:
            await self.event_bus.emit(
                topic,
                payload,
                priority=priority,
                source=self.config.service_name,
            )
            self._record_published_event(topic)
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

    async def _emit_suppressed_summary(
        self,
        *,
        cycle_suppressed_before: int,
    ) -> None:
        suppressed_count = self._total_suppressed_publications - cycle_suppressed_before
        if suppressed_count <= 0:
            return

        await self._emit(
            "system.news_ai_service.publication_limited",
            {
                "service_name": self.config.service_name,
                "suppressed_count": suppressed_count,
                "suppressed_by_topic_total": dict(self._suppressed_publish_counts),
                "cycle_publication_counts": dict(self._current_cycle_publish_counts or {}),
                "limits": {
                    "max_published_events_per_cycle": self.config.max_published_events_per_cycle,
                    "max_published_events_per_hour": self.config.max_published_events_per_hour,
                    "max_scored_events_per_cycle": self.config.max_scored_events_per_cycle,
                    "max_scored_events_per_hour": self.config.max_scored_events_per_hour,
                    "max_high_impact_events_per_cycle": self.config.max_high_impact_events_per_cycle,
                    "max_high_impact_events_per_hour": self.config.max_high_impact_events_per_hour,
                    "max_bot_alerts_per_cycle": self.config.max_bot_alerts_per_cycle,
                    "max_bot_alerts_per_hour": self.config.max_bot_alerts_per_hour,
                    "min_seconds_between_bot_alerts": self.config.min_seconds_between_bot_alerts,
                },
            },
            priority=EventPriority.LOW,
            apply_publication_limits=False,
        )

    def _allow_publication(self, topic: str) -> bool:
        if not self._is_limited_topic(topic):
            return True

        self._prune_publish_history()

        if self._cycle_count("__all__") >= self.config.max_published_events_per_cycle:
            return False

        if self._hour_count("__all__") >= self.config.max_published_events_per_hour:
            return False

        if topic == "news.scored":
            if self._cycle_count(topic) >= self.config.max_scored_events_per_cycle:
                return False
            if self._hour_count(topic) >= self.config.max_scored_events_per_hour:
                return False

        high_impact_topics = {"news.high_impact", "dashboard.news_update", "bot.news_alert"}
        if topic in high_impact_topics:
            if self._cycle_count(topic) >= self.config.max_high_impact_events_per_cycle:
                return False
            if self._hour_count(topic) >= self.config.max_high_impact_events_per_hour:
                return False

        if topic == "bot.news_alert":
            if self._cycle_count(topic) >= self.config.max_bot_alerts_per_cycle:
                return False
            if self._hour_count(topic) >= self.config.max_bot_alerts_per_hour:
                return False
            if self._last_bot_alert_at is not None:
                elapsed = (utc_now() - self._last_bot_alert_at).total_seconds()
                if elapsed < self.config.min_seconds_between_bot_alerts:
                    return False

        return True

    def _is_limited_topic(self, topic: str) -> bool:
        return topic.startswith("news.") or topic in {
            "dashboard.news_update",
            "bot.news_alert",
        }

    def _record_published_event(self, topic: str) -> None:
        if not self._is_limited_topic(topic):
            return

        now = utc_now()
        self._publish_history.append((now, topic))

        if self._current_cycle_publish_counts is not None:
            self._current_cycle_publish_counts["__all__"] = (
                self._current_cycle_publish_counts.get("__all__", 0) + 1
            )
            self._current_cycle_publish_counts[topic] = (
                self._current_cycle_publish_counts.get(topic, 0) + 1
            )

        if topic == "bot.news_alert":
            self._last_bot_alert_at = now

        self._prune_publish_history(now=now)

    def _record_suppressed_publication(self, topic: str) -> None:
        self._total_suppressed_publications += 1
        self._suppressed_publish_counts[topic] = (
            self._suppressed_publish_counts.get(topic, 0) + 1
        )

    def _cycle_count(self, topic: str) -> int:
        if self._current_cycle_publish_counts is None:
            return 0
        return self._current_cycle_publish_counts.get(topic, 0)

    def _hour_count(self, topic: str) -> int:
        self._prune_publish_history()

        if topic == "__all__":
            return len(self._publish_history)

        return sum(1 for _, published_topic in self._publish_history if published_topic == topic)

    def _prune_publish_history(self, *, now: Any | None = None) -> None:
        current_time = now or utc_now()
        cutoff_seconds = 3_600.0
        while self._publish_history:
            published_at, _ = self._publish_history[0]
            if (current_time - published_at).total_seconds() <= cutoff_seconds:
                break
            self._publish_history.popleft()

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