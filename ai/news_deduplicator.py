from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from core.logger import get_logger

from .config import NewsDeduplicationConfig
from .enums import NewsDeduplicationReason
from .exceptions import (
    NewsDeduplicationError,
    NewsErrorContext,
)
from .models import (
    DeduplicationDecision,
    NormalizedNewsItem,
    RawNewsItem,
    utc_now,
)


_TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "fbclid",
    "gclid",
    "yclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "cmpid",
}


@dataclass(slots=True, frozen=True)
class _SeenRecord:
    """
    Internal deduplication record.
    """

    news_id: str
    key: str
    reason: NewsDeduplicationReason
    title: str | None
    text: str | None
    created_at: datetime
    metadata: dict[str, Any]


def _safe_lower(value: str | None) -> str:
    return value.strip().lower() if value else ""


def _normalize_text_for_hash(value: str | None) -> str:
    """
    Normalize text before hashing.

    This intentionally keeps the logic simple and deterministic.
    """

    if not value:
        return ""

    text = value.strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s$.-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _hash_text(value: str | None) -> str | None:
    normalized = _normalize_text_for_hash(value)
    if not normalized:
        return None

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_url(value: str | None) -> str | None:
    """
    Normalize URL for deduplication.

    Removes fragments and common tracking query params.
    """

    if not value:
        return None

    raw = value.strip()
    if not raw:
        return None

    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw.lower()

    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()

    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = re.sub(r"/+$", "", parts.path or "/")

    query_items = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=False)
        if key.lower() not in _TRACKING_QUERY_PARAMS
    ]

    query = urlencode(sorted(query_items), doseq=True)

    return urlunsplit((scheme, netloc, path, query, ""))


class NewsDeduplicator:
    """
    Bounded in-memory news deduplicator.

    This component is intentionally storage-agnostic. Later, it can be extended
    with Redis/Postgres-backed state, but for the first production-ready version
    this class is enough for runtime deduplication.
    """

    def __init__(self, config: NewsDeduplicationConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

        self._seen_by_url: OrderedDict[str, _SeenRecord] = OrderedDict()
        self._seen_by_canonical_url: OrderedDict[str, _SeenRecord] = OrderedDict()
        self._seen_by_source_item_id: OrderedDict[str, _SeenRecord] = OrderedDict()
        self._seen_by_title_hash: OrderedDict[str, _SeenRecord] = OrderedDict()
        self._seen_by_content_hash: OrderedDict[str, _SeenRecord] = OrderedDict()

        # Used only when enable_near_duplicate_detection=True.
        self._recent_titles: OrderedDict[str, _SeenRecord] = OrderedDict()

        self._checked_count = 0
        self._duplicate_count = 0
        self._unique_count = 0
        self._evicted_count = 0

    def check_raw(self, item: RawNewsItem) -> DeduplicationDecision:
        """
        Check whether a raw news item is already known.

        This is useful immediately after fetching, before expensive processing.
        """

        if not self.config.enabled:
            return DeduplicationDecision.unique()

        try:
            self._cleanup_expired()

            news_id = self._raw_runtime_id(item)
            title = item.title
            text = item.text
            url = item.url
            source_item_id = item.source_item_id

            decision = self._check_common(
                news_id=news_id,
                source_name=item.source_name,
                url=url,
                canonical_url=url,
                source_item_id=source_item_id,
                title=title,
                text=text,
            )

            self._record_check(decision)
            return decision

        except Exception as exc:
            raise NewsDeduplicationError(
                "Failed to check raw news item for duplicates",
                context=NewsErrorContext(
                    news_id=item.source_item_id,
                    source_name=item.source_name,
                    url=item.url,
                    details={"title": item.title[:160]},
                ),
                cause=exc,
            ) from exc

    def check_normalized(self, item: NormalizedNewsItem) -> DeduplicationDecision:
        """
        Check whether a normalized news item is already known.
        """

        if not self.config.enabled:
            return DeduplicationDecision.unique()

        try:
            self._cleanup_expired()

            decision = self._check_common(
                news_id=item.news_id,
                source_name=item.source_name,
                url=item.url,
                canonical_url=item.canonical_url,
                source_item_id=item.source_item_id,
                title=item.title,
                text=item.text,
                title_hash=item.title_hash,
                content_hash=item.content_hash,
            )

            self._record_check(decision)
            return decision

        except Exception as exc:
            raise NewsDeduplicationError(
                "Failed to check normalized news item for duplicates",
                context=NewsErrorContext(
                    news_id=item.news_id,
                    source_name=item.source_name,
                    url=item.url,
                    details={"title": item.title[:160]},
                ),
                cause=exc,
            ) from exc

    def remember_raw(self, item: RawNewsItem) -> None:
        """
        Add raw item to deduplication memory.

        Usually called after a raw item is accepted as unique.
        """

        if not self.config.enabled:
            return

        try:
            news_id = self._raw_runtime_id(item)
            self._remember_common(
                news_id=news_id,
                source_name=item.source_name,
                url=item.url,
                canonical_url=item.url,
                source_item_id=item.source_item_id,
                title=item.title,
                text=item.text,
            )
        except Exception as exc:
            raise NewsDeduplicationError(
                "Failed to remember raw news item",
                context=NewsErrorContext(
                    news_id=item.source_item_id,
                    source_name=item.source_name,
                    url=item.url,
                    details={"title": item.title[:160]},
                ),
                cause=exc,
            ) from exc

    def remember_normalized(self, item: NormalizedNewsItem) -> None:
        """
        Add normalized item to deduplication memory.

        Usually called after the normalized item was successfully scored or
        accepted by the pipeline.
        """

        if not self.config.enabled:
            return

        try:
            self._remember_common(
                news_id=item.news_id,
                source_name=item.source_name,
                url=item.url,
                canonical_url=item.canonical_url,
                source_item_id=item.source_item_id,
                title=item.title,
                text=item.text,
                title_hash=item.title_hash,
                content_hash=item.content_hash,
            )
        except Exception as exc:
            raise NewsDeduplicationError(
                "Failed to remember normalized news item",
                context=NewsErrorContext(
                    news_id=item.news_id,
                    source_name=item.source_name,
                    url=item.url,
                    details={"title": item.title[:160]},
                ),
                cause=exc,
            ) from exc

    def filter_new_raw(self, items: list[RawNewsItem]) -> list[RawNewsItem]:
        """
        Filter raw items and remember accepted ones immediately.

        This prevents duplicates inside the same fetch cycle.
        """

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
        """
        Filter normalized items and remember accepted ones immediately.

        This prevents duplicates inside the same processing cycle.
        """

        unique: list[NormalizedNewsItem] = []

        for item in items:
            decision = self.check_normalized(item)
            if decision.is_duplicate:
                continue

            unique.append(item)
            self.remember_normalized(item)

        return unique

    def clear(self) -> None:
        """
        Clear all in-memory deduplication state.
        """

        self._seen_by_url.clear()
        self._seen_by_canonical_url.clear()
        self._seen_by_source_item_id.clear()
        self._seen_by_title_hash.clear()
        self._seen_by_content_hash.clear()
        self._recent_titles.clear()

        self._checked_count = 0
        self._duplicate_count = 0
        self._unique_count = 0
        self._evicted_count = 0

    def stats(self) -> dict[str, Any]:
        """
        Return deduplication runtime stats.
        """

        self._cleanup_expired()

        return {
            "enabled": self.config.enabled,
            "checked_count": self._checked_count,
            "duplicate_count": self._duplicate_count,
            "unique_count": self._unique_count,
            "evicted_count": self._evicted_count,
            "seen_by_url": len(self._seen_by_url),
            "seen_by_canonical_url": len(self._seen_by_canonical_url),
            "seen_by_source_item_id": len(self._seen_by_source_item_id),
            "seen_by_title_hash": len(self._seen_by_title_hash),
            "seen_by_content_hash": len(self._seen_by_content_hash),
            "recent_titles": len(self._recent_titles),
            "ttl_seconds": self.config.ttl_seconds,
            "max_seen_items": self.config.max_seen_items,
            "near_duplicate_detection": self.config.enable_near_duplicate_detection,
        }

    def _check_common(
        self,
        *,
        news_id: str,
        source_name: str,
        url: str | None,
        canonical_url: str | None,
        source_item_id: str | None,
        title: str | None,
        text: str | None,
        title_hash: str | None = None,
        content_hash: str | None = None,
    ) -> DeduplicationDecision:
        normalized_url = _normalize_url(url)
        normalized_canonical_url = _normalize_url(canonical_url)

        if self.config.dedup_by_url and normalized_url:
            record = self._seen_by_url.get(normalized_url)
            if record:
                self._touch(self._seen_by_url, normalized_url)
                return DeduplicationDecision.duplicate(
                    reason=NewsDeduplicationReason.URL,
                    existing_news_id=record.news_id,
                    similarity_score=1.0,
                )

        if self.config.dedup_by_canonical_url and normalized_canonical_url:
            record = self._seen_by_canonical_url.get(normalized_canonical_url)
            if record:
                self._touch(self._seen_by_canonical_url, normalized_canonical_url)
                return DeduplicationDecision.duplicate(
                    reason=NewsDeduplicationReason.CANONICAL_URL,
                    existing_news_id=record.news_id,
                    similarity_score=1.0,
                )

        source_item_key = self._source_item_key(source_name, source_item_id)
        if self.config.dedup_by_source_item_id and source_item_key:
            record = self._seen_by_source_item_id.get(source_item_key)
            if record:
                self._touch(self._seen_by_source_item_id, source_item_key)
                return DeduplicationDecision.duplicate(
                    reason=NewsDeduplicationReason.SOURCE_ITEM_ID,
                    existing_news_id=record.news_id,
                    similarity_score=1.0,
                )

        effective_title_hash = title_hash or _hash_text(title)
        if self.config.dedup_by_title_hash and effective_title_hash:
            record = self._seen_by_title_hash.get(effective_title_hash)
            if record:
                self._touch(self._seen_by_title_hash, effective_title_hash)
                return DeduplicationDecision.duplicate(
                    reason=NewsDeduplicationReason.TITLE_HASH,
                    existing_news_id=record.news_id,
                    similarity_score=1.0,
                )

        effective_content_hash = content_hash or _hash_text(text)
        if self.config.dedup_by_content_hash and effective_content_hash:
            record = self._seen_by_content_hash.get(effective_content_hash)
            if record:
                self._touch(self._seen_by_content_hash, effective_content_hash)
                return DeduplicationDecision.duplicate(
                    reason=NewsDeduplicationReason.CONTENT_HASH,
                    existing_news_id=record.news_id,
                    similarity_score=1.0,
                )

        if self.config.enable_near_duplicate_detection and title:
            decision = self._check_near_duplicate_title(title)
            if decision.is_duplicate:
                return decision

        return DeduplicationDecision.unique()

    def _remember_common(
        self,
        *,
        news_id: str,
        source_name: str,
        url: str | None,
        canonical_url: str | None,
        source_item_id: str | None,
        title: str | None,
        text: str | None,
        title_hash: str | None = None,
        content_hash: str | None = None,
    ) -> None:
        now = utc_now()

        normalized_url = _normalize_url(url)
        normalized_canonical_url = _normalize_url(canonical_url)
        source_item_key = self._source_item_key(source_name, source_item_id)
        effective_title_hash = title_hash or _hash_text(title)
        effective_content_hash = content_hash or _hash_text(text)

        metadata = {
            "source_name": source_name,
            "url": url,
            "canonical_url": canonical_url,
            "source_item_id": source_item_id,
        }

        if self.config.dedup_by_url and normalized_url:
            self._put(
                self._seen_by_url,
                normalized_url,
                _SeenRecord(
                    news_id=news_id,
                    key=normalized_url,
                    reason=NewsDeduplicationReason.URL,
                    title=title,
                    text=text,
                    created_at=now,
                    metadata=metadata,
                ),
            )

        if self.config.dedup_by_canonical_url and normalized_canonical_url:
            self._put(
                self._seen_by_canonical_url,
                normalized_canonical_url,
                _SeenRecord(
                    news_id=news_id,
                    key=normalized_canonical_url,
                    reason=NewsDeduplicationReason.CANONICAL_URL,
                    title=title,
                    text=text,
                    created_at=now,
                    metadata=metadata,
                ),
            )

        if self.config.dedup_by_source_item_id and source_item_key:
            self._put(
                self._seen_by_source_item_id,
                source_item_key,
                _SeenRecord(
                    news_id=news_id,
                    key=source_item_key,
                    reason=NewsDeduplicationReason.SOURCE_ITEM_ID,
                    title=title,
                    text=text,
                    created_at=now,
                    metadata=metadata,
                ),
            )

        if self.config.dedup_by_title_hash and effective_title_hash:
            self._put(
                self._seen_by_title_hash,
                effective_title_hash,
                _SeenRecord(
                    news_id=news_id,
                    key=effective_title_hash,
                    reason=NewsDeduplicationReason.TITLE_HASH,
                    title=title,
                    text=text,
                    created_at=now,
                    metadata=metadata,
                ),
            )

        if self.config.dedup_by_content_hash and effective_content_hash:
            self._put(
                self._seen_by_content_hash,
                effective_content_hash,
                _SeenRecord(
                    news_id=news_id,
                    key=effective_content_hash,
                    reason=NewsDeduplicationReason.CONTENT_HASH,
                    title=title,
                    text=text,
                    created_at=now,
                    metadata=metadata,
                ),
            )

        if self.config.enable_near_duplicate_detection and title:
            normalized_title = _normalize_text_for_hash(title)
            if normalized_title:
                self._put(
                    self._recent_titles,
                    normalized_title,
                    _SeenRecord(
                        news_id=news_id,
                        key=normalized_title,
                        reason=NewsDeduplicationReason.TITLE_SIMILARITY,
                        title=title,
                        text=text,
                        created_at=now,
                        metadata=metadata,
                    ),
                )

        self._enforce_size_limits()

    def _check_near_duplicate_title(self, title: str) -> DeduplicationDecision:
        normalized_title = _normalize_text_for_hash(title)
        if not normalized_title:
            return DeduplicationDecision.unique()

        best_record: _SeenRecord | None = None
        best_score = 0.0

        for seen_title, record in self._recent_titles.items():
            score = SequenceMatcher(None, normalized_title, seen_title).ratio()
            if score > best_score:
                best_score = score
                best_record = record

        if (
            best_record is not None
            and best_score >= self.config.title_similarity_threshold
        ):
            return DeduplicationDecision.duplicate(
                reason=NewsDeduplicationReason.TITLE_SIMILARITY,
                existing_news_id=best_record.news_id,
                similarity_score=best_score,
            )

        return DeduplicationDecision.unique()

    def _cleanup_expired(self) -> None:
        if not self.config.enabled:
            return

        now = utc_now()

        stores = (
            self._seen_by_url,
            self._seen_by_canonical_url,
            self._seen_by_source_item_id,
            self._seen_by_title_hash,
            self._seen_by_content_hash,
            self._recent_titles,
        )

        for store in stores:
            expired_keys = [
                key
                for key, record in store.items()
                if (now - record.created_at).total_seconds() > self.config.ttl_seconds
            ]

            for key in expired_keys:
                store.pop(key, None)
                self._evicted_count += 1

    def _enforce_size_limits(self) -> None:
        stores = (
            self._seen_by_url,
            self._seen_by_canonical_url,
            self._seen_by_source_item_id,
            self._seen_by_title_hash,
            self._seen_by_content_hash,
            self._recent_titles,
        )

        for store in stores:
            while len(store) > self.config.max_seen_items:
                store.popitem(last=False)
                self._evicted_count += 1

    def _put(
        self,
        store: OrderedDict[str, _SeenRecord],
        key: str,
        record: _SeenRecord,
    ) -> None:
        if key in store:
            store.pop(key, None)
        store[key] = record

    def _touch(
        self,
        store: OrderedDict[str, _SeenRecord],
        key: str,
    ) -> None:
        record = store.pop(key, None)
        if record is not None:
            store[key] = record

    def _record_check(self, decision: DeduplicationDecision) -> None:
        self._checked_count += 1
        if decision.is_duplicate:
            self._duplicate_count += 1
        else:
            self._unique_count += 1

    def _source_item_key(
        self,
        source_name: str,
        source_item_id: str | None,
    ) -> str | None:
        if not source_item_id:
            return None

        normalized_source = _safe_lower(source_name)
        normalized_id = _safe_lower(source_item_id)

        if not normalized_source or not normalized_id:
            return None

        return f"{normalized_source}:{normalized_id}"

    def _raw_runtime_id(self, item: RawNewsItem) -> str:
        """
        Build deterministic-ish runtime id for raw dedup state.

        NormalizedNewsItem will later have a proper news_id from NewsProcessor.
        """

        candidates = [
            item.source_item_id,
            _normalize_url(item.url),
            _hash_text(item.title),
            _hash_text(item.text),
        ]

        for candidate in candidates:
            if candidate:
                return f"raw_{hashlib.sha1(str(candidate).encode('utf-8')).hexdigest()}"

        fallback = f"{item.source_name}:{item.title}:{item.fetched_at.isoformat()}"
        return f"raw_{hashlib.sha1(fallback.encode('utf-8')).hexdigest()}"


__all__ = [
    "NewsDeduplicator",
]