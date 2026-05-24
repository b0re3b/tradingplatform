from __future__ import annotations

import asyncio
import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape as html_unescape
from typing import Any
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

import aiohttp

from core.logger import get_logger

from .config import NewsSourceConfig
from .enums import (
    NewsFailureReason,
    NewsFetchStatus,
    NewsLanguage,
    NewsProcessingStage,
    NewsSourceStatus,
    NewsSourceType,
)
from .exceptions import (
    NewsErrorContext,
    NewsFetchError,
    NewsInvalidResponseError,
    NewsRateLimitError,
    NewsSourceError,
    NewsTimeoutError,
)
from .models import NewsSourceHealth, RawNewsItem, utc_now

UTC = timezone.utc
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; TradingSystemNewsBot/1.0; +https://local.trading-system)"
)


def _clean_text(value: Any) -> str:
    """
    Clean primitive text value from RSS/API/HTML payloads.
    """

    if value is None:
        return ""

    text = html_unescape(str(value))
    text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _first_non_empty(*values: Any) -> str | None:
    """
    Return the first non-empty string from values.
    """

    for value in values:
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return None


def _parse_datetime(value: Any) -> datetime | None:
    """
    Parse common RSS/API datetime values into timezone-aware UTC datetime.

    Supports:
        - datetime objects
        - unix timestamps
        - ISO strings
        - RSS/HTTP date strings
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None

    raw = str(value).strip()
    if not raw:
        return None

    if raw.isdigit():
        try:
            timestamp = float(raw)
            if timestamp > 10_000_000_000:
                timestamp /= 1000.0
            return datetime.fromtimestamp(timestamp, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None

    normalized = raw.replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except ValueError:
        pass

    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _get_nested_value(payload: Mapping[str, Any], path: str | None) -> Any:
    """
    Read nested value from dict using dotted path.

    Example:
        path="data.articles"
    """

    if not path:
        return payload

    current: Any = payload
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, Sequence) and not isinstance(current, str):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None

        if current is None:
            return None

    return current


def _looks_like_json_response(content_type: str | None) -> bool:
    if not content_type:
        return False
    return "json" in content_type.lower()


class BaseNewsSource(ABC):
    """
    Base contract for all news source adapters.

    Source adapters return RawNewsItem only. They do not process, score,
    deduplicate, publish, or schedule anything.
    """

    def __init__(self, config: NewsSourceConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self._last_fetch_started_at: float | None = None
        self._last_fetch_at: datetime | None = None
        self._last_success_at: datetime | None = None
        self._last_error: str | None = None
        self._last_status: NewsFetchStatus = NewsFetchStatus.EMPTY
        self._disabled_after_failure = False
        self._disabled_after_failure_at: datetime | None = None
        self._disabled_after_failure_reason: str | None = None
        self._total_fetches = 0
        self._successful_fetches = 0
        self._failed_fetches = 0
        self._total_items_fetched = 0
        self._average_latency_ms: float | None = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def source_type(self) -> NewsSourceType:
        return self.config.source_type

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def runtime_disabled(self) -> bool:
        return self._disabled_after_failure

    @property
    def available(self) -> bool:
        return self.config.enabled and not self._disabled_after_failure

    @property
    def disabled_after_failure_reason(self) -> str | None:
        return self._disabled_after_failure_reason

    @abstractmethod
    async def fetch(
        self,
        session: aiohttp.ClientSession | None = None,
    ) -> list[RawNewsItem]:
        """
        Fetch raw news items from the source.
        """

    def health(self) -> NewsSourceHealth:
        """
        Return current source health snapshot.
        """

        status = NewsSourceStatus.DISABLED
        if self.config.enabled:
            if self._disabled_after_failure:
                status = NewsSourceStatus.FAILED
            elif self._last_status in {
                NewsFetchStatus.SUCCESS,
                NewsFetchStatus.PARTIAL_SUCCESS,
            }:
                status = NewsSourceStatus.HEALTHY
            elif self._last_status == NewsFetchStatus.RATE_LIMITED:
                status = NewsSourceStatus.RATE_LIMITED
            elif self._last_status in {
                NewsFetchStatus.FAILED,
                NewsFetchStatus.TIMEOUT,
                NewsFetchStatus.INVALID_RESPONSE,
            }:
                status = NewsSourceStatus.FAILED
            else:
                status = NewsSourceStatus.UNKNOWN

        return NewsSourceHealth(
            source_name=self.name,
            source_type=self.source_type,
            status=status,
            last_fetch_status=self._last_status,
            last_fetch_at=self._last_fetch_at,
            last_success_at=self._last_success_at,
            last_error=self._disabled_after_failure_reason or self._last_error,
            total_fetches=self._total_fetches,
            successful_fetches=self._successful_fetches,
            failed_fetches=self._failed_fetches,
            total_items_fetched=self._total_items_fetched,
            average_latency_ms=self._average_latency_ms,
        )

    async def _run_fetch(
        self,
        fetch_func: Any,
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> list[RawNewsItem]:
        """
        Wrap concrete fetch logic with metrics and normalized error handling.
        """

        if not self.config.enabled:
            self._last_status = NewsFetchStatus.EMPTY
            return []

        if self._disabled_after_failure:
            self._last_status = NewsFetchStatus.FAILED
            self._last_error = self._disabled_after_failure_reason
            return []

        self._respect_min_fetch_interval()

        started_monotonic = time.monotonic()
        self._last_fetch_started_at = started_monotonic
        self._last_fetch_at = utc_now()
        self._total_fetches += 1

        try:
            items = await fetch_func(session=session)
            limited_items = items[: self.config.max_items_per_fetch]

            self._last_status = (
                NewsFetchStatus.SUCCESS if limited_items else NewsFetchStatus.EMPTY
            )
            self._last_error = None
            self._last_success_at = utc_now()
            self._successful_fetches += 1
            self._total_items_fetched += len(limited_items)
            self._update_latency(started_monotonic)

            return limited_items

        except asyncio.CancelledError:
            self._update_latency(started_monotonic)
            raise

        except NewsSourceError as exc:
            self._failed_fetches += 1
            self._last_error = exc.message
            self._last_status = self._status_from_exception(exc)
            self._update_latency(started_monotonic)
            self._disable_after_failure(exc.message)
            raise

        except TimeoutError as exc:
            self._failed_fetches += 1
            self._last_status = NewsFetchStatus.TIMEOUT
            self._last_error = str(exc)
            self._update_latency(started_monotonic)
            error = NewsTimeoutError(
                f"News source '{self.name}' timed out",
                context=self._context(reason=NewsFailureReason.TIMEOUT),
                cause=exc,
            )
            self._disable_after_failure(error.message)
            raise error from exc

        except Exception as exc:
            self._failed_fetches += 1
            self._last_status = NewsFetchStatus.FAILED
            self._last_error = str(exc)
            self._update_latency(started_monotonic)
            error = NewsFetchError(
                f"News source '{self.name}' fetch failed",
                context=self._context(reason=NewsFailureReason.NETWORK_ERROR),
                cause=exc,
            )
            self._disable_after_failure(error.message)
            raise error from exc

    def _disable_after_failure(self, reason: str) -> None:
        """
        Runtime-quarantine this source after a failed fetch.

        News feeds are optional for trading. A broken source must not keep
        being queried on every scheduler tick and must not be able to cascade
        into global risk halts through repeated scheduler failures.
        """

        if not self.config.disable_after_failure or self._disabled_after_failure:
            return

        self._disabled_after_failure = True
        self._disabled_after_failure_at = utc_now()
        self._disabled_after_failure_reason = reason
        self._last_status = NewsFetchStatus.FAILED
        self._last_error = reason
        self.logger.warning(
            "News source disabled after failed fetch",
            extra={
                "source_name": self.name,
                "source_type": str(self.source_type),
                "reason": reason,
            },
        )

    def _respect_min_fetch_interval(self) -> None:
        """
        Avoid too frequent fetches from the same source.

        The collector/service will also control cadence, but source-level
        protection prevents accidental rapid polling.
        """

        if self._last_fetch_started_at is None:
            return

        elapsed = time.monotonic() - self._last_fetch_started_at
        min_interval = self.config.min_fetch_interval_seconds
        if elapsed < min_interval:
            raise NewsFetchError(
                f"News source '{self.name}' fetched too frequently",
                context=self._context(
                    reason=NewsFailureReason.RATE_LIMITED,
                    details={
                        "elapsed_seconds": elapsed,
                        "min_interval_seconds": min_interval,
                    },
                ),
            )

    def _update_latency(self, started_monotonic: float) -> None:
        latency_ms = (time.monotonic() - started_monotonic) * 1000.0
        if self._average_latency_ms is None:
            self._average_latency_ms = latency_ms
        else:
            self._average_latency_ms = (self._average_latency_ms * 0.8) + (
                latency_ms * 0.2
            )

    def _status_from_exception(self, exc: NewsSourceError) -> NewsFetchStatus:
        if isinstance(exc, NewsRateLimitError):
            return NewsFetchStatus.RATE_LIMITED
        if isinstance(exc, NewsTimeoutError):
            return NewsFetchStatus.TIMEOUT
        if isinstance(exc, NewsInvalidResponseError):
            return NewsFetchStatus.INVALID_RESPONSE
        return NewsFetchStatus.FAILED

    def _context(
        self,
        *,
        reason: NewsFailureReason,
        url: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> NewsErrorContext:
        return NewsErrorContext(
            stage=NewsProcessingStage.FETCH,
            reason=reason,
            source_name=self.name,
            source_type=str(self.source_type),
            url=url or self.config.url or self.config.api_url,
            details=details,
        )

    def _headers(self) -> dict[str, str]:
        """
        Build safe request headers.

        API key values are read from env only inside the source adapter and are
        never returned by config.safe_dict().
        """

        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json,text/xml,application/xml,text/html;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            **self.config.headers,
        }

        if self.config.api_key_env:
            api_key = os.getenv(self.config.api_key_env)
            if api_key:
                # Default convention. Individual API sources may override through config.headers.
                headers.setdefault("Authorization", f"Bearer {api_key}")

        return headers

    def _request_timeout(self) -> aiohttp.ClientTimeout:
        total = max(1.0, float(self.config.request_timeout_seconds))
        connect = min(5.0, total)
        sock_read = min(10.0, total)
        return aiohttp.ClientTimeout(
            total=total,
            connect=connect,
            sock_connect=connect,
            sock_read=sock_read,
        )

    async def _request_text(
        self,
        url: str,
        *,
        session: aiohttp.ClientSession | None = None,
        params: Mapping[str, str] | None = None,
    ) -> str:
        close_session = False
        active_session = session

        if active_session is None:
            timeout = self._request_timeout()
            active_session = aiohttp.ClientSession(timeout=timeout)
            close_session = True

        try:
            async with active_session.get(
                url,
                headers=self._headers(),
                params=params or self.config.query_params,
                timeout=self._request_timeout(),
            ) as response:
                if response.status == 429:
                    raise NewsRateLimitError(
                        f"News source '{self.name}' rate limited",
                        context=self._context(
                            reason=NewsFailureReason.RATE_LIMITED,
                            url=str(response.url),
                            details={"status": response.status},
                        ),
                    )

                if response.status >= 400:
                    raise NewsFetchError(
                        f"News source '{self.name}' returned HTTP {response.status}",
                        context=self._context(
                            reason=NewsFailureReason.INVALID_RESPONSE,
                            url=str(response.url),
                            details={"status": response.status},
                        ),
                    )

                return await response.text()

        except asyncio.TimeoutError as exc:
            raise NewsTimeoutError(
                f"News source '{self.name}' request timed out",
                context=self._context(reason=NewsFailureReason.TIMEOUT, url=url),
                cause=exc,
            ) from exc

        finally:
            if close_session and active_session is not None:
                await active_session.close()

    async def _request_json(
        self,
        url: str,
        *,
        session: aiohttp.ClientSession | None = None,
        params: Mapping[str, str] | None = None,
    ) -> Any:
        close_session = False
        active_session = session

        if active_session is None:
            timeout = self._request_timeout()
            active_session = aiohttp.ClientSession(timeout=timeout)
            close_session = True

        try:
            async with active_session.get(
                url,
                headers=self._headers(),
                params=params or self.config.query_params,
                timeout=self._request_timeout(),
            ) as response:
                if response.status == 429:
                    raise NewsRateLimitError(
                        f"News source '{self.name}' rate limited",
                        context=self._context(
                            reason=NewsFailureReason.RATE_LIMITED,
                            url=str(response.url),
                            details={"status": response.status},
                        ),
                    )

                if response.status >= 400:
                    raise NewsFetchError(
                        f"News source '{self.name}' returned HTTP {response.status}",
                        context=self._context(
                            reason=NewsFailureReason.INVALID_RESPONSE,
                            url=str(response.url),
                            details={"status": response.status},
                        ),
                    )

                content_type = response.headers.get("Content-Type")
                if not _looks_like_json_response(content_type):
                    text = await response.text()
                    raise NewsInvalidResponseError(
                        f"News source '{self.name}' did not return JSON",
                        context=self._context(
                            reason=NewsFailureReason.INVALID_RESPONSE,
                            url=str(response.url),
                            details={
                                "content_type": content_type,
                                "body_preview": text[:300],
                            },
                        ),
                    )

                return await response.json()

        except asyncio.TimeoutError as exc:
            raise NewsTimeoutError(
                f"News source '{self.name}' request timed out",
                context=self._context(reason=NewsFailureReason.TIMEOUT, url=url),
                cause=exc,
            ) from exc

        finally:
            if close_session and active_session is not None:
                await active_session.close()

    def _build_raw_item(
        self,
        *,
        title: str,
        url: str | None = None,
        body: str | None = None,
        summary: str | None = None,
        author: str | None = None,
        source_item_id: str | None = None,
        published_at: datetime | None = None,
        language: NewsLanguage | None = None,
        raw_payload: dict[str, Any] | None = None,
    ) -> RawNewsItem | None:
        title_clean = _clean_text(title)
        if not title_clean:
            return None

        return RawNewsItem(
            source_name=self.name,
            source_type=self.source_type,
            title=title_clean,
            url=url,
            body=_clean_text(body) or None,
            summary=_clean_text(summary) or None,
            author=_clean_text(author) or None,
            source_item_id=source_item_id,
            published_at=published_at,
            fetched_at=utc_now(),
            language=language or self.config.default_language,
            raw_payload=raw_payload or {},
        )


class RSSNewsSource(BaseNewsSource):
    """
    RSS/Atom news source adapter.
    """

    async def fetch(
        self,
        session: aiohttp.ClientSession | None = None,
    ) -> list[RawNewsItem]:
        return await self._run_fetch(self._fetch_rss, session=session)

    async def _fetch_rss(
        self,
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> list[RawNewsItem]:
        if not self.config.url:
            raise NewsInvalidResponseError(
                f"RSS source '{self.name}' has no URL",
                context=self._context(reason=NewsFailureReason.INVALID_CONFIG),
            )

        xml_text = await self._request_text(self.config.url, session=session)

        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError as exc:
            raise NewsInvalidResponseError(
                f"RSS source '{self.name}' returned invalid XML",
                context=self._context(
                    reason=NewsFailureReason.PARSE_ERROR,
                    details={"body_preview": xml_text[:300]},
                ),
                cause=exc,
            ) from exc

        items = self._parse_rss_items(root)
        if not items:
            items = self._parse_atom_items(root)

        return items[: self.config.max_items_per_fetch]

    def _parse_rss_items(self, root: ElementTree.Element) -> list[RawNewsItem]:
        parsed: list[RawNewsItem] = []

        for item in root.findall(".//item"):
            title = _first_non_empty(
                item.findtext("title"),
                item.findtext("{http://purl.org/rss/1.0/}title"),
            )
            if not title:
                continue

            link = _first_non_empty(
                item.findtext("link"),
                item.findtext("{http://purl.org/rss/1.0/}link"),
            )

            description = _first_non_empty(
                item.findtext("description"),
                item.findtext("{http://purl.org/rss/1.0/}description"),
                item.findtext("{http://purl.org/dc/elements/1.1/}description"),
            )

            author = _first_non_empty(
                item.findtext("author"),
                item.findtext("{http://purl.org/dc/elements/1.1/}creator"),
            )

            guid = _first_non_empty(item.findtext("guid"))
            published_at = _parse_datetime(
                _first_non_empty(
                    item.findtext("pubDate"),
                    item.findtext("{http://purl.org/dc/elements/1.1/}date"),
                )
            )

            raw_item = self._build_raw_item(
                title=title,
                url=link,
                summary=description,
                author=author,
                source_item_id=guid or link,
                published_at=published_at,
                raw_payload={
                    "rss_guid": guid,
                    "rss_link": link,
                },
            )

            if raw_item:
                parsed.append(raw_item)

        return parsed

    def _parse_atom_items(self, root: ElementTree.Element) -> list[RawNewsItem]:
        parsed: list[RawNewsItem] = []
        namespace = "{http://www.w3.org/2005/Atom}"

        for entry in root.findall(f".//{namespace}entry"):
            title = _first_non_empty(entry.findtext(f"{namespace}title"))
            if not title:
                continue

            link = None
            for link_el in entry.findall(f"{namespace}link"):
                href = link_el.attrib.get("href")
                rel = link_el.attrib.get("rel")
                if href and rel in (None, "alternate"):
                    link = href
                    break

            summary = _first_non_empty(
                entry.findtext(f"{namespace}summary"),
                entry.findtext(f"{namespace}content"),
            )

            author = None
            author_el = entry.find(f"{namespace}author")
            if author_el is not None:
                author = _first_non_empty(author_el.findtext(f"{namespace}name"))

            source_item_id = _first_non_empty(entry.findtext(f"{namespace}id"))
            published_at = _parse_datetime(
                _first_non_empty(
                    entry.findtext(f"{namespace}published"),
                    entry.findtext(f"{namespace}updated"),
                )
            )

            raw_item = self._build_raw_item(
                title=title,
                url=link,
                summary=summary,
                author=author,
                source_item_id=source_item_id or link,
                published_at=published_at,
                raw_payload={
                    "atom_id": source_item_id,
                    "atom_link": link,
                },
            )

            if raw_item:
                parsed.append(raw_item)

        return parsed


class APINewsSource(BaseNewsSource):
    """
    Generic JSON API news source.

    Mapping is configured through NewsSourceConfig.metadata.

    Supported metadata keys:
        items_path: dotted path to list of articles, default: None/root
        title_field: default "title"
        url_field: default "url"
        body_field: default "body"
        summary_field: default "summary"
        author_field: default "author"
        id_field: default "id"
        published_at_field: default "published_at"
        language_field: default "language"
    """

    async def fetch(
        self,
        session: aiohttp.ClientSession | None = None,
    ) -> list[RawNewsItem]:
        return await self._run_fetch(self._fetch_api, session=session)

    async def _fetch_api(
        self,
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> list[RawNewsItem]:
        if not self.config.api_url:
            raise NewsInvalidResponseError(
                f"API source '{self.name}' has no api_url",
                context=self._context(reason=NewsFailureReason.INVALID_CONFIG),
            )

        payload = await self._request_json(self.config.api_url, session=session)

        items = self._extract_items_from_payload(payload)
        parsed: list[RawNewsItem] = []

        for item in items:
            if not isinstance(item, Mapping):
                continue

            raw_item = self._raw_item_from_mapping(item)
            if raw_item:
                parsed.append(raw_item)

        return parsed[: self.config.max_items_per_fetch]

    def _extract_items_from_payload(self, payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload

        if not isinstance(payload, Mapping):
            raise NewsInvalidResponseError(
                f"API source '{self.name}' returned unsupported JSON root",
                context=self._context(
                    reason=NewsFailureReason.INVALID_RESPONSE,
                    details={"root_type": type(payload).__name__},
                ),
            )

        items_path = self.config.metadata.get("items_path")
        items = _get_nested_value(payload, str(items_path)) if items_path else payload

        if isinstance(items, Mapping):
            # Common API shapes.
            for candidate_key in ("articles", "items", "data", "results", "news"):
                candidate = items.get(candidate_key)
                if isinstance(candidate, list):
                    return candidate

        if isinstance(items, list):
            return items

        raise NewsInvalidResponseError(
            f"API source '{self.name}' could not find article list",
            context=self._context(
                reason=NewsFailureReason.INVALID_RESPONSE,
                details={
                    "items_path": items_path,
                    "payload_keys": list(payload.keys())[:20],
                },
            ),
        )

    def _raw_item_from_mapping(self, item: Mapping[str, Any]) -> RawNewsItem | None:
        metadata = self.config.metadata

        title_field = str(metadata.get("title_field", "title"))
        url_field = str(metadata.get("url_field", "url"))
        body_field = str(metadata.get("body_field", "body"))
        summary_field = str(metadata.get("summary_field", "summary"))
        author_field = str(metadata.get("author_field", "author"))
        id_field = str(metadata.get("id_field", "id"))
        published_at_field = str(metadata.get("published_at_field", "published_at"))
        language_field = str(metadata.get("language_field", "language"))

        title = _get_nested_value(item, title_field)
        url = _get_nested_value(item, url_field)
        body = _get_nested_value(item, body_field)
        summary = _get_nested_value(item, summary_field)
        author = _get_nested_value(item, author_field)
        source_item_id = _get_nested_value(item, id_field)
        published_at = _parse_datetime(_get_nested_value(item, published_at_field))

        language = self._parse_language(_get_nested_value(item, language_field))

        return self._build_raw_item(
            title=str(title or ""),
            url=str(url) if url else None,
            body=str(body) if body else None,
            summary=str(summary) if summary else None,
            author=str(author) if author else None,
            source_item_id=str(source_item_id) if source_item_id else None,
            published_at=published_at,
            language=language,
            raw_payload=dict(item),
        )

    def _parse_language(self, value: Any) -> NewsLanguage:
        if not value:
            return self.config.default_language

        normalized = str(value).strip().lower()
        for language in NewsLanguage:
            if normalized == language.value:
                return language

        return self.config.default_language


class StaticHTMLNewsSource(BaseNewsSource):
    """
    Static HTML / embedded-data news source adapter.

    The adapter remains dependency-free and does not use browser automation, but
    it handles the common shapes used by modern news/exchange pages:
        1. metadata-driven custom regex
        2. JSON-LD structured data
        3. Next.js ``__NEXT_DATA__`` / application JSON script blocks
        4. conservative anchor scan fallback

    Useful metadata keys:
        article_link_regex: regex with named groups title/url/summary/published_at/id
        article_link_regexes: list/tuple of such regexes
        article_url_allow_patterns: substrings or regexes that article URLs must match
        article_url_block_patterns: substrings or regexes to reject
        title_exclude_patterns: substrings or regexes to reject titles
        min_title_length: minimum title length, default 12
        max_embedded_json_nodes: recursion cap for embedded JSON extraction
    """

    async def fetch(
        self,
        session: aiohttp.ClientSession | None = None,
    ) -> list[RawNewsItem]:
        return await self._run_fetch(self._fetch_html, session=session)

    async def _fetch_html(
        self,
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> list[RawNewsItem]:
        if not self.config.url:
            raise NewsInvalidResponseError(
                f"HTML source '{self.name}' has no URL",
                context=self._context(reason=NewsFailureReason.INVALID_CONFIG),
            )

        html = await self._request_text(self.config.url, session=session)
        items = self._extract_items_from_html(html, base_url=self.config.url)

        if not items and self._looks_blocked_or_dynamic(html):
            raise NewsInvalidResponseError(
                f"HTML source '{self.name}' did not expose parseable static article data",
                context=self._context(
                    reason=NewsFailureReason.INVALID_RESPONSE,
                    details={
                        "parser": "static_html",
                        "hint": "blocked_or_javascript_rendered",
                        "body_preview": _clean_text(html[:500]),
                    },
                ),
            )

        return items[: self.config.max_items_per_fetch]

    def _extract_items_from_html(
        self,
        html: str,
        *,
        base_url: str,
    ) -> list[RawNewsItem]:
        parsed: list[RawNewsItem] = []

        parsed.extend(self._extract_by_custom_regexes(html, base_url=base_url))
        parsed.extend(self._extract_json_ld_items(html, base_url=base_url))
        parsed.extend(self._extract_next_data_items(html, base_url=base_url))
        parsed.extend(self._extract_application_json_items(html, base_url=base_url))
        parsed.extend(self._extract_by_basic_anchor_scan(html, base_url=base_url))

        return self._deduplicate_items(parsed)[: self.config.max_items_per_fetch]

    def _extract_by_custom_regexes(
        self,
        html: str,
        *,
        base_url: str,
    ) -> list[RawNewsItem]:
        metadata = self.config.metadata
        patterns: list[str] = []

        single_pattern = metadata.get("article_link_regex")
        if single_pattern:
            patterns.append(str(single_pattern))

        multiple_patterns = metadata.get("article_link_regexes")
        if isinstance(multiple_patterns, Sequence) and not isinstance(multiple_patterns, str):
            patterns.extend(str(pattern) for pattern in multiple_patterns if pattern)

        parsed: list[RawNewsItem] = []
        for pattern in patterns:
            parsed.extend(
                self._extract_by_custom_regex(
                    html,
                    base_url=base_url,
                    pattern=pattern,
                )
            )

        return parsed

    def _extract_by_custom_regex(
        self,
        html: str,
        *,
        base_url: str,
        pattern: str,
    ) -> list[RawNewsItem]:
        parsed: list[RawNewsItem] = []

        try:
            compiled = re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
        except re.error as exc:
            raise NewsInvalidResponseError(
                f"HTML source '{self.name}' has invalid article_link_regex",
                context=self._context(
                    reason=NewsFailureReason.INVALID_CONFIG,
                    details={"regex_error": str(exc)},
                ),
                cause=exc,
            ) from exc

        for match in compiled.finditer(html):
            group_dict = match.groupdict()
            title = group_dict.get("title") or group_dict.get("headline") or match.group(0)
            url = group_dict.get("url") or group_dict.get("href")
            summary = group_dict.get("summary") or group_dict.get("description")
            source_item_id = group_dict.get("id") or group_dict.get("source_item_id")
            published_at = _parse_datetime(
                group_dict.get("published_at")
                or group_dict.get("date")
                or group_dict.get("published")
            )

            raw_item = self._item_from_candidate(
                title=title,
                url=url,
                base_url=base_url,
                summary=summary,
                source_item_id=source_item_id,
                published_at=published_at,
                raw_payload={"parser": "custom_regex", "match": group_dict},
            )
            if raw_item:
                parsed.append(raw_item)

        return parsed

    def _extract_json_ld_items(
        self,
        html: str,
        *,
        base_url: str,
    ) -> list[RawNewsItem]:
        pattern = re.compile(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(?P<payload>.*?)</script>',
            flags=re.IGNORECASE | re.DOTALL,
        )

        parsed: list[RawNewsItem] = []
        for match in pattern.finditer(html):
            payload_text = html_unescape(match.group("payload")).strip()
            if not payload_text:
                continue

            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue

            for candidate in self._iter_json_article_candidates(payload):
                raw_item = self._raw_item_from_json_candidate(
                    candidate,
                    base_url=base_url,
                    parser="json_ld",
                )
                if raw_item:
                    parsed.append(raw_item)

        return parsed

    def _extract_next_data_items(
        self,
        html: str,
        *,
        base_url: str,
    ) -> list[RawNewsItem]:
        pattern = re.compile(
            r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(?P<payload>.*?)</script>',
            flags=re.IGNORECASE | re.DOTALL,
        )

        parsed: list[RawNewsItem] = []
        for match in pattern.finditer(html):
            payload_text = html_unescape(match.group("payload")).strip()
            if not payload_text:
                continue

            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue

            for candidate in self._iter_json_article_candidates(payload):
                raw_item = self._raw_item_from_json_candidate(
                    candidate,
                    base_url=base_url,
                    parser="next_data",
                )
                if raw_item:
                    parsed.append(raw_item)

        return parsed

    def _extract_application_json_items(
        self,
        html: str,
        *,
        base_url: str,
    ) -> list[RawNewsItem]:
        pattern = re.compile(
            r'<script[^>]+type=["\']application/json["\'][^>]*>(?P<payload>.*?)</script>',
            flags=re.IGNORECASE | re.DOTALL,
        )

        parsed: list[RawNewsItem] = []
        max_scripts = int(self.config.metadata.get("max_application_json_scripts", 5))

        for index, match in enumerate(pattern.finditer(html)):
            if index >= max_scripts:
                break

            payload_text = html_unescape(match.group("payload")).strip()
            if not payload_text or len(payload_text) > 3_000_000:
                continue

            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue

            for candidate in self._iter_json_article_candidates(payload):
                raw_item = self._raw_item_from_json_candidate(
                    candidate,
                    base_url=base_url,
                    parser="application_json",
                )
                if raw_item:
                    parsed.append(raw_item)

        return parsed

    def _iter_json_article_candidates(self, payload: Any) -> list[Mapping[str, Any]]:
        max_nodes = int(self.config.metadata.get("max_embedded_json_nodes", 5000))
        candidates: list[Mapping[str, Any]] = []
        visited = 0

        def walk(value: Any) -> None:
            nonlocal visited
            if visited >= max_nodes:
                return
            visited += 1

            if isinstance(value, Mapping):
                if self._mapping_looks_like_article(value):
                    candidates.append(value)

                # JSON-LD ItemList entries often wrap the real object.
                for key in ("item", "article", "news", "post", "data"):
                    nested = value.get(key)
                    if nested is not None:
                        walk(nested)

                for nested in value.values():
                    if isinstance(nested, Mapping | list | tuple):
                        walk(nested)

            elif isinstance(value, list | tuple):
                for nested in value:
                    walk(nested)

        walk(payload)
        return candidates

    def _mapping_looks_like_article(self, item: Mapping[str, Any]) -> bool:
        title = self._first_mapping_value(
            item,
            "headline",
            "title",
            "name",
            "announcementTitle",
            "articleTitle",
        )
        url = self._first_mapping_value(
            item,
            "url",
            "link",
            "href",
            "path",
            "slug",
            "webUrl",
        )

        if not title or not url:
            return False

        title_clean = _clean_text(title)
        if not self._is_allowed_title(title_clean):
            return False

        return True

    def _raw_item_from_json_candidate(
        self,
        item: Mapping[str, Any],
        *,
        base_url: str,
        parser: str,
    ) -> RawNewsItem | None:
        title = self._first_mapping_value(
            item,
            "headline",
            "title",
            "name",
            "announcementTitle",
            "articleTitle",
        )
        url = self._first_mapping_value(
            item,
            "url",
            "link",
            "href",
            "path",
            "slug",
            "webUrl",
        )
        summary = self._first_mapping_value(
            item,
            "description",
            "summary",
            "excerpt",
            "subtitle",
            "brief",
        )
        source_item_id = self._first_mapping_value(
            item,
            "id",
            "guid",
            "uuid",
            "code",
            "articleId",
            "announcementId",
        )
        published_at = _parse_datetime(
            self._first_mapping_value(
                item,
                "datePublished",
                "dateModified",
                "published_at",
                "publishedAt",
                "publishDate",
                "releaseDate",
                "createdAt",
            )
        )

        return self._item_from_candidate(
            title=title,
            url=url,
            base_url=base_url,
            summary=summary,
            source_item_id=str(source_item_id) if source_item_id else None,
            published_at=published_at,
            raw_payload={"parser": parser},
        )

    def _extract_by_basic_anchor_scan(
        self,
        html: str,
        *,
        base_url: str,
    ) -> list[RawNewsItem]:
        anchor_pattern = re.compile(
            r'<a[^>]+href=["\'](?P<url>[^"\']+)["\'][^>]*>(?P<title>.*?)</a>',
            flags=re.IGNORECASE | re.DOTALL,
        )

        parsed: list[RawNewsItem] = []
        for match in anchor_pattern.finditer(html):
            raw_item = self._item_from_candidate(
                title=match.group("title"),
                url=match.group("url"),
                base_url=base_url,
                raw_payload={"parser": "anchor_scan"},
            )
            if raw_item:
                parsed.append(raw_item)

            if len(parsed) >= self.config.max_items_per_fetch * 3:
                break

        return parsed

    def _item_from_candidate(
        self,
        *,
        title: Any,
        url: Any,
        base_url: str,
        summary: Any = None,
        source_item_id: str | None = None,
        published_at: datetime | None = None,
        raw_payload: dict[str, Any] | None = None,
    ) -> RawNewsItem | None:
        title_clean = _clean_text(title)
        if not self._is_allowed_title(title_clean):
            return None

        absolute_url = self._normalize_candidate_url(url, base_url=base_url)
        if absolute_url and not self._is_allowed_article_url(absolute_url):
            return None

        return self._build_raw_item(
            title=title_clean,
            url=absolute_url,
            summary=str(summary) if summary else None,
            source_item_id=source_item_id or absolute_url or title_clean,
            published_at=published_at,
            raw_payload=raw_payload or {},
        )

    def _normalize_candidate_url(self, value: Any, *, base_url: str) -> str | None:
        if value is None:
            return None

        raw = str(value).strip()
        if not raw:
            return None

        if raw.startswith("//"):
            scheme = urlsplit(base_url).scheme or "https"
            raw = f"{scheme}:{raw}"

        return urljoin(base_url, raw)

    def _is_allowed_title(self, title: str) -> bool:
        if not title:
            return False

        min_length = int(self.config.metadata.get("min_title_length", 12))
        if len(title) < min_length:
            return False

        lowered = title.lower()
        blocked_defaults = (
            "log in",
            "sign up",
            "privacy policy",
            "terms of use",
            "cookie",
            "subscribe",
            "help center",
            "download app",
        )
        if any(fragment in lowered for fragment in blocked_defaults):
            return False

        blocked_patterns = self.config.metadata.get("title_exclude_patterns", ())
        return not self._matches_any(title, blocked_patterns)

    def _is_allowed_article_url(self, url: str) -> bool:
        lowered = url.lower()
        if self._is_likely_non_article_url(lowered):
            return False

        allow_patterns = self.config.metadata.get("article_url_allow_patterns", ())
        if allow_patterns and not self._matches_any(url, allow_patterns):
            return False

        block_patterns = self.config.metadata.get("article_url_block_patterns", ())
        if block_patterns and self._matches_any(url, block_patterns):
            return False

        return True

    def _is_likely_non_article_url(self, url: str) -> bool:
        lowered = url.lower()

        blocked_fragments = (
            "#",
            "javascript:",
            "mailto:",
            "/login",
            "/signup",
            "/register",
            "/privacy",
            "/terms",
            "/about",
            "/contact",
            "/advertise",
            "/careers",
            "/settings",
            "/account",
            "/download",
        )

        return any(fragment in lowered for fragment in blocked_fragments)

    def _matches_any(self, value: str, patterns: Any) -> bool:
        if not patterns:
            return False

        if isinstance(patterns, str):
            patterns = (patterns,)

        lowered = value.lower()
        for pattern in patterns:
            raw_pattern = str(pattern)
            if not raw_pattern:
                continue

            # Treat simple strings as case-insensitive substrings; regex still works.
            if raw_pattern.lower() in lowered:
                return True

            try:
                if re.search(raw_pattern, value, flags=re.IGNORECASE):
                    return True
            except re.error:
                continue

        return False

    def _deduplicate_items(self, items: list[RawNewsItem]) -> list[RawNewsItem]:
        deduped: list[RawNewsItem] = []
        seen: set[str] = set()

        for item in items:
            key = (item.url or item.source_item_id or item.title).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        return deduped

    def _first_mapping_value(self, item: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            value = item.get(key)
            if value not in (None, ""):
                return value
        return None

    def _looks_blocked_or_dynamic(self, html: str) -> bool:
        lowered = html[:20_000].lower()
        markers = (
            "cloudflare",
            "enable javascript",
            "please enable js",
            "access denied",
            "request blocked",
            "captcha",
            "cf-browser-verification",
            "__cf_chl",
        )
        return any(marker in lowered for marker in markers)


class ExchangeAnnouncementSource(APINewsSource):
    """
    Exchange announcement source router.

    The old implementation treated every non-API exchange URL as RSS. Real
    exchange announcement pages often return static/Next.js HTML, so this
    adapter now routes by actual payload shape:
        - api_url -> JSON API parser
        - XML/RSS/Atom body -> RSS/Atom parser
        - JSON body from url -> generic API payload parser
        - HTML body -> StaticHTMLNewsSource embedded/anchor parser
    """

    async def fetch(
        self,
        session: aiohttp.ClientSession | None = None,
    ) -> list[RawNewsItem]:
        if self.config.source_type != NewsSourceType.EXCHANGE_ANNOUNCEMENT:
            raise NewsInvalidResponseError(
                f"Exchange source '{self.name}' must use EXCHANGE_ANNOUNCEMENT source type",
                context=self._context(reason=NewsFailureReason.INVALID_CONFIG),
            )

        return await self._run_fetch(self._fetch_exchange_announcements, session=session)

    async def _fetch_exchange_announcements(
        self,
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> list[RawNewsItem]:
        if self.config.api_url:
            return await self._fetch_api(session=session)

        if not self.config.url:
            raise NewsInvalidResponseError(
                f"Exchange announcement source '{self.name}' has neither api_url nor url",
                context=self._context(reason=NewsFailureReason.INVALID_CONFIG),
            )

        payload_text = await self._request_text(self.config.url, session=session)
        stripped = payload_text.lstrip()

        if self._looks_like_json(stripped):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise NewsInvalidResponseError(
                    f"Exchange source '{self.name}' returned invalid JSON",
                    context=self._context(
                        reason=NewsFailureReason.PARSE_ERROR,
                        details={"body_preview": stripped[:300]},
                    ),
                    cause=exc,
                ) from exc

            parsed: list[RawNewsItem] = []
            for item in self._extract_items_from_payload(payload):
                if isinstance(item, Mapping):
                    raw_item = self._raw_item_from_mapping(item)
                    if raw_item:
                        parsed.append(raw_item)
            return parsed[: self.config.max_items_per_fetch]

        if self._looks_like_xml(stripped):
            return self._parse_xml_payload(stripped)

        html_adapter = StaticHTMLNewsSource(self.config)
        return html_adapter._extract_items_from_html(payload_text, base_url=self.config.url)

    def _looks_like_json(self, text: str) -> bool:
        return text.startswith("{") or text.startswith("[")

    def _looks_like_xml(self, text: str) -> bool:
        lowered = text[:500].lower()
        return (
            lowered.startswith("<?xml")
            or lowered.startswith("<rss")
            or lowered.startswith("<feed")
            or "<rss" in lowered
            or "<feed" in lowered
        )

    def _parse_xml_payload(self, xml_text: str) -> list[RawNewsItem]:
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError as exc:
            raise NewsInvalidResponseError(
                f"Exchange source '{self.name}' returned invalid XML",
                context=self._context(
                    reason=NewsFailureReason.PARSE_ERROR,
                    details={"body_preview": xml_text[:300]},
                ),
                cause=exc,
            ) from exc

        rss_adapter = RSSNewsSource(self.config)
        items = rss_adapter._parse_rss_items(root)
        if not items:
            items = rss_adapter._parse_atom_items(root)

        return items[: self.config.max_items_per_fetch]

def build_news_source(config: NewsSourceConfig) -> BaseNewsSource:
    """
    Factory for source adapters.
    """

    if config.source_type == NewsSourceType.RSS:
        return RSSNewsSource(config)

    if config.source_type == NewsSourceType.API:
        return APINewsSource(config)

    if config.source_type == NewsSourceType.EXCHANGE_ANNOUNCEMENT:
        return ExchangeAnnouncementSource(config)

    if config.source_type == NewsSourceType.STATIC_HTML:
        return StaticHTMLNewsSource(config)

    raise NewsInvalidResponseError(
        f"Unsupported news source type: {config.source_type}",
        context=NewsErrorContext(
            stage=NewsProcessingStage.FETCH,
            reason=NewsFailureReason.INVALID_CONFIG,
            source_name=config.name,
            source_type=str(config.source_type),
            url=config.url or config.api_url,
        ),
    )


__all__ = [
    "BaseNewsSource",
    "RSSNewsSource",
    "APINewsSource",
    "ExchangeAnnouncementSource",
    "StaticHTMLNewsSource",
    "build_news_source",
]