from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import aiohttp

from core.logger import get_logger
from .config import NewsAIConfig, NewsSourceConfig
from .enums import (
    NewsFailureReason,
    NewsProcessingStage,
    NewsSourceStatus,
)
from .exceptions import (
    NewsAIError,
    NewsConfigError,
    NewsErrorContext,
    NewsFetchError,
)
from .models import (
    NewsBatch,
    NewsProcessingResult,
    NewsSourceHealth,
    RawNewsItem,
    utc_now,
)
from .news_sources import BaseNewsSource, build_news_source


@dataclass(slots=True, frozen=True)
class NewsCollectionResult:
    """
    Result of one collection cycle.
    """

    batch: NewsBatch
    processing_result: NewsProcessingResult
    source_health: tuple[NewsSourceHealth, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def items(self) -> tuple[RawNewsItem, ...]:
        return self.batch.items

    @property
    def item_count(self) -> int:
        return self.batch.count

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch.to_dict(),
            "processing_result": self.processing_result.to_dict(),
            "source_health": [health.to_dict() for health in self.source_health],
            "errors": list(self.errors),
        }


class NewsCollector:
    """
    Collects raw news from all configured source adapters.

    The collector is source-oriented only. It does not process, deduplicate,
    score, publish, or schedule anything.
    """

    def __init__(
        self,
        config: NewsAIConfig,
        *,
        sources: list[BaseNewsSource] | None = None,
    ) -> None:
        self.config = config
        self.logger = get_logger(__name__)

        self._sources: list[BaseNewsSource] = (
            list(sources) if sources is not None else self._build_sources(config.source_configs)
        )

        self._total_cycles = 0
        self._total_items_collected = 0
        self._total_failed_sources = 0
        self._last_collection_at = None
        self._last_error: str | None = None

        self._validate_sources()

    @property
    def sources(self) -> tuple[BaseNewsSource, ...]:
        return tuple(self._sources)

    @property
    def enabled_sources(self) -> tuple[BaseNewsSource, ...]:
        return tuple(
            source
            for source in self._sources
            if bool(getattr(source, "available", source.enabled))
        )

    @property
    def disabled_source_names(self) -> tuple[str, ...]:
        return tuple(
            source.name
            for source in self._sources
            if bool(getattr(source, "runtime_disabled", False))
        )

    @property
    def source_count(self) -> int:
        return len(self._sources)

    @property
    def enabled_source_count(self) -> int:
        return len(self.enabled_sources)

    async def collect(self) -> NewsCollectionResult:
        """
        Collect raw news from all enabled sources.

        A failure in one source does not fail the whole collection cycle.
        """

        started_at = utc_now()
        self._total_cycles += 1
        self._last_collection_at = started_at

        if not self.config.enabled:
            result = self._empty_result(
                started_at=started_at,
                errors=("News AI collection is disabled",),
            )
            self._last_error = "News AI collection is disabled"
            return result

        enabled_sources = self.enabled_sources
        if not enabled_sources:
            result = self._empty_result(
                started_at=started_at,
                errors=("No enabled news sources configured",),
            )
            self._last_error = "No enabled news sources configured"
            return result

        items: list[RawNewsItem] = []
        errors: list[str] = []

        connector = aiohttp.TCPConnector(limit_per_host=self.config.max_concurrent_sources)
        timeout = aiohttp.ClientTimeout(
            total=max(
                source.config.request_timeout_seconds
                for source in enabled_sources
            )
            + 5.0
        )

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        ) as session:
            semaphore = asyncio.Semaphore(self.config.max_concurrent_sources)

            tasks = [
                self._fetch_source_safely(
                    source=source,
                    session=session,
                    semaphore=semaphore,
                )
                for source in enabled_sources
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

        for source, result in zip(enabled_sources, results):
            if isinstance(result, asyncio.CancelledError):
                raise result

            if isinstance(result, BaseException):
                error = f"News source '{source.name}' failed: {result!r}"
                errors.append(error)
                self.logger.warning(
                    "News source task failed during collection",
                    extra={
                        "source_name": source.name,
                        "source_type": str(source.source_type),
                        "error_type": result.__class__.__name__,
                        "error": repr(result),
                    },
                )
                continue

            source_items, source_error = result
            if source_error:
                errors.append(source_error)
                continue

            if source_items:
                items.extend(source_items)

        if len(items) > self.config.max_items_per_cycle:
            items = items[: self.config.max_items_per_cycle]

        source_health = self.health()
        failed_sources = sum(
            1
            for health in source_health
            if health.status
            in {
                NewsSourceStatus.FAILED,
                NewsSourceStatus.RATE_LIMITED,
                NewsSourceStatus.DEGRADED,
            }
        )

        self._total_failed_sources += failed_sources
        self._total_items_collected += len(items)
        self._last_error = "; ".join(errors) if errors else None

        finished_at = utc_now()
        processing_result = NewsProcessingResult(
            stage=NewsProcessingStage.COLLECT,
            started_at=started_at,
            finished_at=finished_at,
            raw_count=len(items),
            processed_count=len(items),
            duplicate_count=0,
            failed_count=len(errors),
            skipped_count=0,
            errors=tuple(errors),
            metadata={
                "source_count": self.source_count,
                "enabled_source_count": self.enabled_source_count,
                "failed_source_count": failed_sources,
                "max_items_per_cycle": self.config.max_items_per_cycle,
            },
        )

        batch = NewsBatch(
            items=tuple(items),
            source_health=source_health,
            created_at=finished_at,
            metadata={
                "cycle": self._total_cycles,
                "errors": errors,
            },
        )

        return NewsCollectionResult(
            batch=batch,
            processing_result=processing_result,
            source_health=source_health,
            errors=tuple(errors),
        )

    async def collect_from_source(self, source_name: str) -> NewsCollectionResult:
        """
        Collect raw news from a single source by name.

        Useful for tests, manual diagnostics, and targeted refreshes.
        """

        started_at = utc_now()

        source = self.get_source(source_name)
        if source is None:
            raise NewsConfigError(
                f"News source '{source_name}' is not configured",
                context=NewsErrorContext(
                    stage=NewsProcessingStage.COLLECT,
                    reason=NewsFailureReason.INVALID_CONFIG,
                    source_name=source_name,
                ),
            )

        if not source.enabled:
            return self._empty_result(
                started_at=started_at,
                errors=(f"News source '{source_name}' is disabled",),
            )

        if bool(getattr(source, "runtime_disabled", False)):
            reason = getattr(source, "disabled_after_failure_reason", None)
            return self._empty_result(
                started_at=started_at,
                errors=(
                    f"News source '{source_name}' is disabled after previous failure"
                    + (f": {reason}" if reason else ""),
                ),
            )

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=source.config.request_timeout_seconds + 5.0
            )
        ) as session:
            items, error = await self._fetch_source_safely(
                source=source,
                session=session,
                semaphore=asyncio.Semaphore(1),
            )

        errors = (error,) if error else ()
        finished_at = utc_now()
        source_health = (source.health(),)

        processing_result = NewsProcessingResult(
            stage=NewsProcessingStage.COLLECT,
            started_at=started_at,
            finished_at=finished_at,
            raw_count=len(items),
            processed_count=len(items),
            duplicate_count=0,
            failed_count=1 if error else 0,
            skipped_count=0,
            errors=errors,
            metadata={
                "source_name": source_name,
                "targeted_collection": True,
            },
        )

        batch = NewsBatch(
            items=tuple(items),
            source_health=source_health,
            created_at=finished_at,
            metadata={
                "targeted_collection": True,
                "source_name": source_name,
                "errors": list(errors),
            },
        )

        return NewsCollectionResult(
            batch=batch,
            processing_result=processing_result,
            source_health=source_health,
            errors=errors,
        )

    def add_source(self, source: BaseNewsSource) -> None:
        """
        Add a runtime source adapter.

        Source names must be unique.
        """

        if self.get_source(source.name) is not None:
            raise NewsConfigError(
                f"Duplicate news source name: {source.name}",
                context=NewsErrorContext(
                    stage=NewsProcessingStage.COLLECT,
                    reason=NewsFailureReason.INVALID_CONFIG,
                    source_name=source.name,
                    source_type=str(source.source_type),
                ),
            )

        self._sources.append(source)

    def remove_source(self, source_name: str) -> bool:
        """
        Remove a source adapter by name.

        Returns True if a source was removed.
        """

        before = len(self._sources)
        self._sources = [
            source for source in self._sources if source.name != source_name
        ]
        return len(self._sources) < before

    def get_source(self, source_name: str) -> BaseNewsSource | None:
        """
        Return source adapter by name.
        """

        for source in self._sources:
            if source.name == source_name:
                return source
        return None

    def health(self) -> tuple[NewsSourceHealth, ...]:
        """
        Return health snapshots for all sources.
        """

        return tuple(source.health() for source in self._sources)

    def stats(self) -> dict[str, Any]:
        """
        Return collector runtime stats.
        """

        health = self.health()

        return {
            "enabled": self.config.enabled,
            "source_count": self.source_count,
            "enabled_source_count": self.enabled_source_count,
            "total_cycles": self._total_cycles,
            "total_items_collected": self._total_items_collected,
            "total_failed_sources": self._total_failed_sources,
            "last_collection_at": (
                self._last_collection_at.isoformat()
                if self._last_collection_at
                else None
            ),
            "last_error": self._last_error,
            "disabled_source_names": list(self.disabled_source_names),
            "sources": [source_health.to_dict() for source_health in health],
        }

    async def _fetch_source_safely(
        self,
        *,
        source: BaseNewsSource,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
    ) -> tuple[list[RawNewsItem], str | None]:
        """
        Fetch one source without allowing its failure to break the whole cycle.
        """

        async with semaphore:
            if bool(getattr(source, "runtime_disabled", False)):
                reason = getattr(source, "disabled_after_failure_reason", None)
                message = (
                    f"News source '{source.name}' skipped because it is disabled after previous failure"
                    + (f": {reason}" if reason else "")
                )
                return [], message

            try:
                items = await source.fetch(session=session)
                return items, None

            except asyncio.CancelledError:
                raise

            except NewsAIError as exc:
                error_message = str(exc)
                self.logger.warning(
                    "News source fetch failed",
                    extra={
                        "source_name": source.name,
                        "source_type": str(source.source_type),
                        "error_type": exc.__class__.__name__,
                        "error": error_message,
                    },
                )
                return [], error_message

            except Exception as exc:
                wrapped = NewsFetchError(
                    f"Unexpected failure while fetching source '{source.name}'",
                    context=NewsErrorContext(
                        stage=NewsProcessingStage.FETCH,
                        reason=NewsFailureReason.UNKNOWN,
                        source_name=source.name,
                        source_type=str(source.source_type),
                        url=source.config.url or source.config.api_url,
                    ),
                    cause=exc,
                )
                error_message = str(wrapped)
                self.logger.warning(
                    "Unexpected news source fetch failure",
                    extra={
                        "source_name": source.name,
                        "source_type": str(source.source_type),
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
                return [], error_message

    def _build_sources(
        self,
        source_configs: tuple[NewsSourceConfig, ...],
    ) -> list[BaseNewsSource]:
        sources: list[BaseNewsSource] = []

        for source_config in source_configs:
            if not source_config.enabled:
                continue

            try:
                sources.append(build_news_source(source_config))
            except NewsAIError:
                raise
            except Exception as exc:
                raise NewsConfigError(
                    f"Failed to build news source '{source_config.name}'",
                    context=NewsErrorContext(
                        stage=NewsProcessingStage.COLLECT,
                        reason=NewsFailureReason.INVALID_CONFIG,
                        source_name=source_config.name,
                        source_type=str(source_config.source_type),
                        url=source_config.url or source_config.api_url,
                    ),
                    cause=exc,
                ) from exc

        return sources

    def _validate_sources(self) -> None:
        names = [source.name for source in self._sources]
        duplicates = sorted({name for name in names if names.count(name) > 1})

        if duplicates:
            raise NewsConfigError(
                "Duplicate news source names are not allowed",
                context=NewsErrorContext(
                    stage=NewsProcessingStage.COLLECT,
                    reason=NewsFailureReason.INVALID_CONFIG,
                    details={"duplicates": duplicates},
                ),
            )

    def _empty_result(
        self,
        *,
        started_at,
        errors: tuple[str, ...] = (),
    ) -> NewsCollectionResult:
        finished_at = utc_now()

        processing_result = NewsProcessingResult(
            stage=NewsProcessingStage.COLLECT,
            started_at=started_at,
            finished_at=finished_at,
            raw_count=0,
            processed_count=0,
            duplicate_count=0,
            failed_count=len(errors),
            skipped_count=0,
            errors=errors,
            metadata={
                "source_count": self.source_count,
                "enabled_source_count": self.enabled_source_count,
            },
        )

        batch = NewsBatch(
            items=(),
            source_health=self.health(),
            created_at=finished_at,
            metadata={
                "errors": list(errors),
            },
        )

        return NewsCollectionResult(
            batch=batch,
            processing_result=processing_result,
            source_health=self.health(),
            errors=errors,
        )


__all__ = [
    "NewsCollectionResult",
    "NewsCollector",
]