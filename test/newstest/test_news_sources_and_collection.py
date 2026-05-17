# test/newstest/test_news_sources_and_collection.py

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from ai import (
    APINewsSource,
    ExchangeAnnouncementSource,
    NewsAIError,
    NewsConfigError,
    NewsFetchError,
    NewsFetchStatus,
    NewsInvalidResponseError,
    NewsLanguage,
    NewsRateLimitError,
    NewsSourceConfig,
    NewsSourceHealth,
    NewsSourceStatus,
    NewsSourceType,
    NewsTimeoutError,
    RawNewsItem,
    RSSNewsSource,
    StaticHTMLNewsSource,
    NewsCollector,
)


# ---------------------------------------------------------------------------
# Fake HTTP infrastructure
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        url: str = "https://example.com/feed",
        text_data: str = "",
        json_data: Any = None,
        headers: dict[str, str] | None = None,
        enter_exception: BaseException | None = None,
        json_exception: BaseException | None = None,
    ) -> None:
        self.status = status
        self.url = url
        self._text_data = text_data
        self._json_data = json_data
        self.headers = headers or {}
        self.enter_exception = enter_exception
        self.json_exception = json_exception

    async def __aenter__(self) -> "FakeResponse":
        if self.enter_exception is not None:
            raise self.enter_exception
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def text(self) -> str:
        return self._text_data

    async def json(self) -> Any:
        if self.json_exception is not None:
            raise self.json_exception
        return self._json_data


class FakeSession:
    """
    Minimal aiohttp.ClientSession-compatible fake.

    It intentionally records every GET call so tests can assert headers,
    params, timeout usage and URL selection.
    """

    def __init__(self, responses: list[FakeResponse] | tuple[FakeResponse, ...]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})

        if not self.responses:
            raise AssertionError(f"Unexpected GET call to {url}; no fake responses left")

        return self.responses.pop(0)

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Local config helpers
# ---------------------------------------------------------------------------


def rss_config(
    *,
    name: str = "rss_test",
    url: str = "https://example.com/rss.xml",
    max_items_per_fetch: int = 10,
    min_fetch_interval_seconds: float = 0.0,
) -> NewsSourceConfig:
    return NewsSourceConfig(
        name=name,
        source_type=NewsSourceType.RSS,
        url=url,
        request_timeout_seconds=1.0,
        max_items_per_fetch=max_items_per_fetch,
        min_fetch_interval_seconds=min_fetch_interval_seconds,
        default_language=NewsLanguage.EN,
        source_reputation_score=0.75,
    )


def api_config(
    *,
    name: str = "api_test",
    api_url: str = "https://api.example.com/news",
    metadata: dict[str, Any] | None = None,
    max_items_per_fetch: int = 10,
    min_fetch_interval_seconds: float = 0.0,
    api_key_env: str | None = None,
    headers: dict[str, str] | None = None,
    query_params: dict[str, str] | None = None,
) -> NewsSourceConfig:
    return NewsSourceConfig(
        name=name,
        source_type=NewsSourceType.API,
        api_url=api_url,
        api_key_env=api_key_env,
        request_timeout_seconds=1.0,
        max_items_per_fetch=max_items_per_fetch,
        min_fetch_interval_seconds=min_fetch_interval_seconds,
        default_language=NewsLanguage.EN,
        source_reputation_score=0.70,
        headers=headers or {},
        query_params=query_params or {},
        metadata=metadata or {},
    )


def html_config(
    *,
    name: str = "html_test",
    url: str = "https://example.com/news/",
    metadata: dict[str, Any] | None = None,
    max_items_per_fetch: int = 10,
    min_fetch_interval_seconds: float = 0.0,
) -> NewsSourceConfig:
    return NewsSourceConfig(
        name=name,
        source_type=NewsSourceType.STATIC_HTML,
        url=url,
        request_timeout_seconds=1.0,
        max_items_per_fetch=max_items_per_fetch,
        min_fetch_interval_seconds=min_fetch_interval_seconds,
        default_language=NewsLanguage.EN,
        source_reputation_score=0.65,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# RSS source tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rss_source_parses_valid_rss_cleans_cdata_html_and_limits_items():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Crypto Feed</title>

        <item>
          <title><![CDATA[Breaking: Ethereum protocol hacked for $120M]]></title>
          <link>https://example.com/hack</link>
          <description><![CDATA[<p>Exploit drained funds from DeFi protocol.</p>]]></description>
          <author>Security Desk</author>
          <guid>hack-guid-1</guid>
          <pubDate>Sun, 17 May 2026 12:00:00 GMT</pubDate>
        </item>

        <item>
          <title>Binance will list TEST token</title>
          <link>https://example.com/listing</link>
          <description>Trading opens today.</description>
          <guid>listing-guid-1</guid>
          <pubDate>2026-05-17T13:30:00Z</pubDate>
        </item>

        <item>
          <title>Should be cut by max_items_per_fetch</title>
          <link>https://example.com/cut</link>
          <guid>cut-guid</guid>
        </item>
      </channel>
    </rss>
    """

    source = RSSNewsSource(rss_config(max_items_per_fetch=2))
    session = FakeSession(
        [
            FakeResponse(
                text_data=xml,
                headers={"Content-Type": "application/rss+xml"},
            )
        ]
    )

    items = await source.fetch(session=session)

    assert len(items) == 2

    first = items[0]
    assert first.source_name == "rss_test"
    assert first.source_type == NewsSourceType.RSS
    assert first.title == "Breaking: Ethereum protocol hacked for $120M"
    assert first.url == "https://example.com/hack"
    assert first.summary == "Exploit drained funds from DeFi protocol."
    assert first.author == "Security Desk"
    assert first.source_item_id == "hack-guid-1"
    assert first.published_at is not None
    assert first.published_at.tzinfo is not None
    assert first.language == NewsLanguage.EN

    health = source.health()
    assert health.status == NewsSourceStatus.HEALTHY
    assert health.last_fetch_status == NewsFetchStatus.SUCCESS
    assert health.successful_fetches == 1
    assert health.failed_fetches == 0
    assert health.total_items_fetched == 2

    assert session.calls[0]["url"] == "https://example.com/rss.xml"
    assert "User-Agent" in session.calls[0]["headers"]


@pytest.mark.asyncio
async def test_rss_source_parses_atom_feed_with_alternate_link_and_author():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Atom Crypto Feed</title>
      <entry>
        <title>SEC delays spot ETF decision</title>
        <link href="https://example.com/etf-delay" rel="alternate"/>
        <summary>Regulatory decision delayed again.</summary>
        <author><name>Reg Desk</name></author>
        <id>atom-etf-1</id>
        <updated>2026-05-17T14:00:00Z</updated>
      </entry>
    </feed>
    """

    source = RSSNewsSource(rss_config())
    session = FakeSession([FakeResponse(text_data=xml)])

    items = await source.fetch(session=session)

    assert len(items) == 1
    assert items[0].title == "SEC delays spot ETF decision"
    assert items[0].url == "https://example.com/etf-delay"
    assert items[0].summary == "Regulatory decision delayed again."
    assert items[0].author == "Reg Desk"
    assert items[0].source_item_id == "atom-etf-1"
    assert source.health().last_fetch_status == NewsFetchStatus.SUCCESS


@pytest.mark.asyncio
async def test_rss_source_rejects_invalid_xml_and_records_invalid_response_health():
    source = RSSNewsSource(rss_config())
    session = FakeSession(
        [
            FakeResponse(
                text_data="<rss><channel><item><title>Broken XML",
                headers={"Content-Type": "application/rss+xml"},
            )
        ]
    )

    with pytest.raises(NewsInvalidResponseError) as exc_info:
        await source.fetch(session=session)

    assert "invalid XML" in str(exc_info.value)

    health = source.health()
    assert health.status == NewsSourceStatus.FAILED
    assert health.last_fetch_status == NewsFetchStatus.INVALID_RESPONSE
    assert health.failed_fetches == 1
    assert health.successful_fetches == 0
    assert health.last_error


@pytest.mark.asyncio
async def test_rss_source_empty_valid_feed_is_not_success_and_stays_observable():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>Empty Feed</title></channel></rss>
    """

    source = RSSNewsSource(rss_config())
    session = FakeSession([FakeResponse(text_data=xml)])

    items = await source.fetch(session=session)

    assert items == []

    health = source.health()
    assert health.status == NewsSourceStatus.UNKNOWN
    assert health.last_fetch_status == NewsFetchStatus.EMPTY
    assert health.successful_fetches == 1
    assert health.total_items_fetched == 0


@pytest.mark.asyncio
async def test_source_http_429_is_rate_limited_and_does_not_look_like_generic_failure():
    source = RSSNewsSource(rss_config())
    session = FakeSession(
        [
            FakeResponse(
                status=429,
                url="https://example.com/rss.xml",
                text_data="Too many requests",
            )
        ]
    )

    with pytest.raises(NewsRateLimitError):
        await source.fetch(session=session)

    health = source.health()
    assert health.status == NewsSourceStatus.RATE_LIMITED
    assert health.last_fetch_status == NewsFetchStatus.RATE_LIMITED
    assert health.failed_fetches == 1


@pytest.mark.asyncio
async def test_source_timeout_is_normalized_to_news_timeout_error():
    source = RSSNewsSource(rss_config())
    session = FakeSession(
        [
            FakeResponse(
                enter_exception=asyncio.TimeoutError(),
            )
        ]
    )

    with pytest.raises(NewsTimeoutError):
        await source.fetch(session=session)

    health = source.health()
    assert health.status == NewsSourceStatus.FAILED
    assert health.last_fetch_status == NewsFetchStatus.TIMEOUT
    assert health.failed_fetches == 1


@pytest.mark.asyncio
async def test_source_min_fetch_interval_blocks_accidental_rapid_refetch():
    source = RSSNewsSource(
        rss_config(
            min_fetch_interval_seconds=999.0,
        )
    )

    good_xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Ethereum protocol upgrade completed</title>
          <link>https://example.com/upgrade</link>
        </item>
      </channel>
    </rss>
    """

    first_session = FakeSession([FakeResponse(text_data=good_xml)])
    second_session = FakeSession([FakeResponse(text_data=good_xml)])

    first_items = await source.fetch(session=first_session)
    assert len(first_items) == 1

    with pytest.raises(NewsFetchError) as exc_info:
        await source.fetch(session=second_session)

    assert "too frequently" in str(exc_info.value)

    # Current implementation raises NewsFetchError with RATE_LIMITED context,
    # but maps it to FAILED status. This assertion documents the current
    # vulnerability: source-level anti-hammering is not reflected as RATE_LIMITED.
    health = source.health()
    assert health.last_fetch_status == NewsFetchStatus.FAILED
    assert health.failed_fetches == 1


# ---------------------------------------------------------------------------
# API source tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_source_parses_nested_payload_with_custom_mapping_and_language():
    source = APINewsSource(
        api_config(
            metadata={
                "items_path": "data.articles",
                "title_field": "headline",
                "url_field": "links.canonical",
                "summary_field": "description",
                "body_field": "content.text",
                "author_field": "byline.name",
                "id_field": "uuid",
                "published_at_field": "timestamps.published",
                "language_field": "lang",
            }
        )
    )

    payload = {
        "data": {
            "articles": [
                {
                    "headline": "CFTC charges crypto exchange with violations",
                    "links": {"canonical": "https://example.com/cftc-charge"},
                    "description": "Official enforcement action.",
                    "content": {"text": "Detailed body text."},
                    "byline": {"name": "Reporter A"},
                    "uuid": "api-001",
                    "timestamps": {"published": "2026-05-17T10:15:00Z"},
                    "lang": "en",
                },
                {
                    "headline": "",
                    "links": {"canonical": "https://example.com/no-title"},
                    "uuid": "api-empty-title",
                },
                "garbage-item-that-must-be-skipped",
            ]
        }
    }

    session = FakeSession(
        [
            FakeResponse(
                json_data=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
        ]
    )

    items = await source.fetch(session=session)

    assert len(items) == 1

    item = items[0]
    assert item.title == "CFTC charges crypto exchange with violations"
    assert item.url == "https://example.com/cftc-charge"
    assert item.summary == "Official enforcement action."
    assert item.body == "Detailed body text."
    assert item.author == "Reporter A"
    assert item.source_item_id == "api-001"
    assert item.published_at is not None
    assert item.language == NewsLanguage.EN
    assert item.raw_payload["uuid"] == "api-001"

    health = source.health()
    assert health.status == NewsSourceStatus.HEALTHY
    assert health.total_items_fetched == 1


@pytest.mark.asyncio
async def test_api_source_rejects_json_content_type_spoofing_with_html_body():
    source = APINewsSource(api_config())
    session = FakeSession(
        [
            FakeResponse(
                status=200,
                text_data="<html>not json</html>",
                headers={"Content-Type": "text/html; charset=utf-8"},
            )
        ]
    )

    with pytest.raises(NewsInvalidResponseError) as exc_info:
        await source.fetch(session=session)

    error_text = str(exc_info.value)
    assert "did not return JSON" in error_text
    assert "body_preview" in error_text
    assert "not json" in error_text

    assert source.health().last_fetch_status == NewsFetchStatus.INVALID_RESPONSE


@pytest.mark.asyncio
async def test_api_source_rejects_unsupported_json_root():
    source = APINewsSource(api_config())
    session = FakeSession(
        [
            FakeResponse(
                json_data="not-a-list-or-dict",
                headers={"Content-Type": "application/json"},
            )
        ]
    )

    with pytest.raises(NewsInvalidResponseError) as exc_info:
        await source.fetch(session=session)

    assert "unsupported JSON root" in str(exc_info.value)
    assert source.health().last_fetch_status == NewsFetchStatus.INVALID_RESPONSE


@pytest.mark.asyncio
async def test_api_source_rejects_dict_payload_without_article_list():
    source = APINewsSource(api_config(metadata={"items_path": "data.missing"}))
    session = FakeSession(
        [
            FakeResponse(
                json_data={"data": {"not_articles": []}, "meta": {"ok": True}},
                headers={"Content-Type": "application/json"},
            )
        ]
    )

    with pytest.raises(NewsInvalidResponseError) as exc_info:
        await source.fetch(session=session)

    assert "could not find article list" in str(exc_info.value)
    assert source.health().last_fetch_status == NewsFetchStatus.INVALID_RESPONSE


@pytest.mark.asyncio
async def test_api_source_uses_headers_query_params_and_env_api_key_without_exposing_secret(
    monkeypatch,
):
    monkeypatch.setenv("NEWS_API_TEST_KEY", "super-secret-token")

    source = APINewsSource(
        api_config(
            api_key_env="NEWS_API_TEST_KEY",
            headers={"X-Custom": "yes"},
            query_params={"category": "crypto", "api_key": "should-not-be-logged-here"},
        )
    )

    session = FakeSession(
        [
            FakeResponse(
                json_data=[
                    {
                        "title": "Bitcoin ETF sees record inflows",
                        "url": "https://example.com/etf-inflows",
                        "id": "api-key-test",
                    }
                ],
                headers={"Content-Type": "application/json"},
            )
        ]
    )

    items = await source.fetch(session=session)

    assert len(items) == 1

    call = session.calls[0]
    assert call["headers"]["Authorization"] == "Bearer super-secret-token"
    assert call["headers"]["X-Custom"] == "yes"
    assert call["params"]["category"] == "crypto"

    safe = source.config.safe_dict()
    assert safe["api_key_env"] == "NEWS_API_TEST_KEY"
    assert safe["query_params"]["api_key"] == "***REDACTED***"
    assert "super-secret-token" not in str(safe)


# ---------------------------------------------------------------------------
# Exchange and static HTML source tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exchange_announcement_source_uses_api_when_api_url_is_configured():
    config = NewsSourceConfig(
        name="exchange_api",
        source_type=NewsSourceType.EXCHANGE_ANNOUNCEMENT,
        url="https://exchange.example.com/announcements",
        api_url="https://exchange.example.com/api/announcements",
        request_timeout_seconds=1.0,
        max_items_per_fetch=5,
        min_fetch_interval_seconds=0.0,
        default_language=NewsLanguage.EN,
        is_exchange_source=True,
        metadata={
            "items_path": "data",
            "title_field": "title",
            "url_field": "url",
            "id_field": "id",
        },
    )

    source = ExchangeAnnouncementSource(config)
    session = FakeSession(
        [
            FakeResponse(
                json_data={
                    "data": [
                        {
                            "title": "Exchange will list ABC token",
                            "url": "https://exchange.example.com/a",
                            "id": "ann-1",
                        }
                    ]
                },
                headers={"Content-Type": "application/json"},
            )
        ]
    )

    items = await source.fetch(session=session)

    assert len(items) == 1
    assert items[0].source_type == NewsSourceType.EXCHANGE_ANNOUNCEMENT
    assert items[0].title == "Exchange will list ABC token"
    assert session.calls[0]["url"] == "https://exchange.example.com/api/announcements"


@pytest.mark.asyncio
async def test_exchange_announcement_source_falls_back_to_rss_when_no_api_url():
    config = NewsSourceConfig(
        name="exchange_rss",
        source_type=NewsSourceType.EXCHANGE_ANNOUNCEMENT,
        url="https://exchange.example.com/rss.xml",
        request_timeout_seconds=1.0,
        max_items_per_fetch=5,
        min_fetch_interval_seconds=0.0,
        default_language=NewsLanguage.EN,
        is_exchange_source=True,
    )

    xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Exchange adds support for SOL futures</title>
          <link>https://exchange.example.com/sol-futures</link>
          <guid>sol-futures</guid>
        </item>
      </channel>
    </rss>
    """

    source = ExchangeAnnouncementSource(config)
    session = FakeSession([FakeResponse(text_data=xml)])

    items = await source.fetch(session=session)

    assert len(items) == 1
    assert items[0].source_type == NewsSourceType.EXCHANGE_ANNOUNCEMENT
    assert items[0].title == "Exchange adds support for SOL futures"


def test_static_html_source_basic_anchor_scan_ignores_navigation_duplicates_and_short_titles():
    source = StaticHTMLNewsSource(html_config(max_items_per_fetch=3))

    html = """
    <html>
      <body>
        <a href="/login">Login to account</a>
        <a href="/privacy">Privacy policy page</a>
        <a href="/article-1">Breaking: Bitcoin ETF approval delayed by regulator</a>
        <a href="/article-1">Breaking: Bitcoin ETF approval delayed by regulator</a>
        <a href="javascript:void(0)">This should never be a news item</a>
        <a href="/x">Too short</a>
        <a href="https://external.example.com/article-2">Ethereum DeFi protocol hacked for $120M</a>
      </body>
    </html>
    """

    items = source._extract_items_from_html(
        html,
        base_url="https://example.com/news/",
    )

    assert len(items) == 2

    urls = {item.url for item in items}
    assert "https://example.com/article-1" in urls
    assert "https://external.example.com/article-2" in urls

    titles = {item.title for item in items}
    assert "Login to account" not in titles
    assert "Privacy policy page" not in titles
    assert "Too short" not in titles
    assert "This should never be a news item" not in titles


def test_static_html_source_custom_regex_extracts_named_title_and_relative_url():
    source = StaticHTMLNewsSource(
        html_config(
            metadata={
                "article_link_regex": (
                    r'<article>\s*'
                    r'<a href="(?P<url>[^"]+)">\s*'
                    r'<h2>(?P<title>.*?)</h2>\s*'
                    r"</a>\s*</article>"
                )
            }
        )
    )

    html = """
    <article>
      <a href="/regulation/sec-action">
        <h2>SEC announces crypto enforcement action</h2>
      </a>
    </article>
    """

    items = source._extract_items_from_html(
        html,
        base_url="https://example.com/news/",
    )

    assert len(items) == 1
    assert items[0].title == "SEC announces crypto enforcement action"
    assert items[0].url == "https://example.com/regulation/sec-action"
    assert items[0].raw_payload["match"]["url"] == "/regulation/sec-action"


def test_static_html_source_invalid_custom_regex_is_reported_as_invalid_config():
    source = StaticHTMLNewsSource(
        html_config(
            metadata={
                "article_link_regex": r"(?P<title>broken-regex",
            }
        )
    )

    with pytest.raises(NewsInvalidResponseError) as exc_info:
        source._extract_items_from_html(
            "<html></html>",
            base_url="https://example.com/news/",
        )

    assert "invalid article_link_regex" in str(exc_info.value)
    assert "regex_error" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Collector fake sources
# ---------------------------------------------------------------------------


class FakeCollectorSource:
    def __init__(
        self,
        *,
        name: str,
        enabled: bool = True,
        items: list[RawNewsItem] | tuple[RawNewsItem, ...] = (),
        fetch_exception: BaseException | None = None,
        delay_seconds: float = 0.0,
        source_type: NewsSourceType = NewsSourceType.RSS,
    ) -> None:
        self.config = NewsSourceConfig(
            name=name,
            source_type=source_type,
            url=f"https://example.com/{name}.xml",
            request_timeout_seconds=1.0,
            max_items_per_fetch=100,
            min_fetch_interval_seconds=0.0,
            default_language=NewsLanguage.EN,
            enabled=enabled,
        )
        self._items = list(items)
        self._fetch_exception = fetch_exception
        self._delay_seconds = delay_seconds

        self.fetch_calls = 0
        self.seen_sessions: list[Any] = []
        self._last_status = NewsFetchStatus.EMPTY
        self._failed_fetches = 0
        self._successful_fetches = 0
        self._total_items_fetched = 0
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def source_type(self) -> NewsSourceType:
        return self.config.source_type

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    async def fetch(self, session=None) -> list[RawNewsItem]:
        self.fetch_calls += 1
        self.seen_sessions.append(session)

        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)

        if self._fetch_exception is not None:
            self._failed_fetches += 1
            self._last_error = str(self._fetch_exception)
            if isinstance(self._fetch_exception, NewsRateLimitError):
                self._last_status = NewsFetchStatus.RATE_LIMITED
            elif isinstance(self._fetch_exception, NewsTimeoutError):
                self._last_status = NewsFetchStatus.TIMEOUT
            elif isinstance(self._fetch_exception, NewsInvalidResponseError):
                self._last_status = NewsFetchStatus.INVALID_RESPONSE
            else:
                self._last_status = NewsFetchStatus.FAILED
            raise self._fetch_exception

        self._successful_fetches += 1
        self._total_items_fetched += len(self._items)
        self._last_status = NewsFetchStatus.SUCCESS if self._items else NewsFetchStatus.EMPTY
        self._last_error = None
        return list(self._items)

    def health(self) -> NewsSourceHealth:
        if not self.enabled:
            status = NewsSourceStatus.DISABLED
        elif self._last_status == NewsFetchStatus.SUCCESS:
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
            last_error=self._last_error,
            total_fetches=self.fetch_calls,
            successful_fetches=self._successful_fetches,
            failed_fetches=self._failed_fetches,
            total_items_fetched=self._total_items_fetched,
        )


# ---------------------------------------------------------------------------
# Collector tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collector_continues_when_one_source_fails_and_preserves_successful_items(
    news_config,
    make_raw_news_item,
):
    ok_item = make_raw_news_item(
        title="Bitcoin ETF sees record inflows",
        url="https://example.com/ok",
        source_item_id="ok-1",
    )

    ok_source = FakeCollectorSource(name="ok_source", items=[ok_item])
    bad_source = FakeCollectorSource(
        name="bad_source",
        fetch_exception=NewsInvalidResponseError("Malformed feed from bad_source"),
    )

    collector = NewsCollector(news_config, sources=[ok_source, bad_source])

    result = await collector.collect()

    assert result.item_count == 1
    assert result.items[0] == ok_item
    assert result.has_errors is True
    assert len(result.errors) == 1
    assert "bad_source" in result.errors[0] or "Malformed feed" in result.errors[0]

    assert result.processing_result.raw_count == 1
    assert result.processing_result.processed_count == 1
    assert result.processing_result.failed_count == 1
    assert result.processing_result.metadata["source_count"] == 2
    assert result.processing_result.metadata["enabled_source_count"] == 2
    assert result.processing_result.metadata["failed_source_count"] == 1

    stats = collector.stats()
    assert stats["total_cycles"] == 1
    assert stats["total_items_collected"] == 1
    assert stats["total_failed_sources"] == 1
    assert stats["last_error"]

    assert ok_source.fetch_calls == 1
    assert bad_source.fetch_calls == 1


@pytest.mark.asyncio
async def test_collector_limits_items_by_max_items_per_cycle_without_mutating_sources(
    news_config,
    make_raw_news_item,
):
    limited_config = replace(news_config, max_items_per_cycle=3)

    source_items = [
        make_raw_news_item(
            title=f"Valid crypto news item number {index}",
            url=f"https://example.com/{index}",
            source_item_id=f"id-{index}",
        )
        for index in range(10)
    ]

    source = FakeCollectorSource(name="large_source", items=source_items)
    collector = NewsCollector(limited_config, sources=[source])

    result = await collector.collect()

    assert result.item_count == 3
    assert [item.source_item_id for item in result.items] == ["id-0", "id-1", "id-2"]
    assert source._items == source_items

    assert result.processing_result.raw_count == 3
    assert result.processing_result.metadata["max_items_per_cycle"] == 3

    stats = collector.stats()
    assert stats["total_items_collected"] == 3


@pytest.mark.asyncio
async def test_collector_returns_empty_result_when_package_is_disabled(
    disabled_news_config,
    make_raw_news_item,
):
    source = FakeCollectorSource(
        name="should_not_be_called",
        items=[
            make_raw_news_item(
                title="This item must not be fetched",
                source_item_id="disabled-1",
            )
        ],
    )

    collector = NewsCollector(disabled_news_config, sources=[source])

    result = await collector.collect()

    assert result.item_count == 0
    assert result.has_errors is True
    assert result.errors == ("News AI collection is disabled",)
    assert result.processing_result.failed_count == 1
    assert source.fetch_calls == 0

    stats = collector.stats()
    assert stats["enabled"] is False
    assert stats["total_cycles"] == 1
    assert stats["total_items_collected"] == 0
    assert stats["last_error"] == "News AI collection is disabled"


@pytest.mark.asyncio
async def test_collector_returns_error_when_no_enabled_sources(news_config):
    disabled_source = FakeCollectorSource(name="disabled_source", enabled=False)
    collector = NewsCollector(news_config, sources=[disabled_source])

    result = await collector.collect()

    assert result.item_count == 0
    assert result.has_errors is True
    assert result.errors == ("No enabled news sources configured",)
    assert result.processing_result.failed_count == 1
    assert result.processing_result.metadata["enabled_source_count"] == 0
    assert disabled_source.fetch_calls == 0

    stats = collector.stats()
    assert stats["source_count"] == 1
    assert stats["enabled_source_count"] == 0
    assert stats["last_error"] == "No enabled news sources configured"


@pytest.mark.asyncio
async def test_collector_wraps_unexpected_source_exception_without_breaking_cycle(
    news_config,
    make_raw_news_item,
):
    ok_item = make_raw_news_item(
        title="Solana network upgrade completed",
        url="https://example.com/sol",
        source_item_id="sol-1",
    )

    ok_source = FakeCollectorSource(name="ok_source", items=[ok_item])
    exploding_source = FakeCollectorSource(
        name="exploding_source",
        fetch_exception=RuntimeError("raw untyped crash"),
    )

    collector = NewsCollector(news_config, sources=[ok_source, exploding_source])

    result = await collector.collect()

    assert result.item_count == 1
    assert result.has_errors is True
    assert len(result.errors) == 1
    assert "Unexpected failure while fetching source" in result.errors[0]
    assert "exploding_source" in result.errors[0]
    assert result.processing_result.failed_count == 1


@pytest.mark.asyncio
async def test_collector_collect_from_source_targets_only_requested_source(
    news_config,
    make_raw_news_item,
):
    target_item = make_raw_news_item(
        title="Targeted source item",
        url="https://example.com/target",
        source_item_id="target-1",
    )

    target = FakeCollectorSource(name="target", items=[target_item])
    other = FakeCollectorSource(
        name="other",
        items=[
            make_raw_news_item(
                title="Other source item",
                url="https://example.com/other",
                source_item_id="other-1",
            )
        ],
    )

    collector = NewsCollector(news_config, sources=[target, other])

    result = await collector.collect_from_source("target")

    assert result.item_count == 1
    assert result.items[0] == target_item
    assert result.processing_result.metadata["targeted_collection"] is True
    assert result.processing_result.metadata["source_name"] == "target"

    assert target.fetch_calls == 1
    assert other.fetch_calls == 0


@pytest.mark.asyncio
async def test_collector_collect_from_unknown_source_raises_config_error(news_config):
    collector = NewsCollector(news_config, sources=[])

    with pytest.raises(NewsConfigError) as exc_info:
        await collector.collect_from_source("missing_source")

    assert "missing_source" in str(exc_info.value)


def test_collector_rejects_duplicate_runtime_source_names(news_config):
    first = FakeCollectorSource(name="duplicate")
    second = FakeCollectorSource(name="duplicate")

    with pytest.raises(NewsConfigError) as exc_info:
        NewsCollector(news_config, sources=[first, second])

    assert "Duplicate news source names" in str(exc_info.value)
    assert "duplicate" in str(exc_info.value)


def test_collector_add_source_rejects_duplicate_name(news_config):
    collector = NewsCollector(news_config, sources=[FakeCollectorSource(name="a")])

    with pytest.raises(NewsConfigError) as exc_info:
        collector.add_source(FakeCollectorSource(name="a"))

    assert "Duplicate news source name" in str(exc_info.value)


def test_collector_remove_source_is_idempotent_and_does_not_corrupt_remaining_sources(
    news_config,
):
    collector = NewsCollector(
        news_config,
        sources=[
            FakeCollectorSource(name="a"),
            FakeCollectorSource(name="b"),
        ],
    )

    assert collector.source_count == 2
    assert collector.remove_source("a") is True
    assert collector.source_count == 1
    assert collector.get_source("a") is None
    assert collector.get_source("b") is not None

    assert collector.remove_source("a") is False
    assert collector.source_count == 1


@pytest.mark.asyncio
async def test_collector_uses_concurrency_limit_not_unbounded_fanout(
    news_config,
    make_raw_news_item,
):
    """
    This test is deliberately timing-sensitive but bounded.

    With max_concurrent_sources=1 and two slow sources, total runtime should be
    at least roughly the sum of delays. If collector accidentally ignores the
    semaphore, this test will become suspiciously fast.
    """

    limited_config = replace(news_config, max_concurrent_sources=1)

    source_a = FakeCollectorSource(
        name="slow_a",
        delay_seconds=0.03,
        items=[
            make_raw_news_item(
                title="Slow source A item",
                url="https://example.com/a",
                source_item_id="slow-a",
            )
        ],
    )
    source_b = FakeCollectorSource(
        name="slow_b",
        delay_seconds=0.03,
        items=[
            make_raw_news_item(
                title="Slow source B item",
                url="https://example.com/b",
                source_item_id="slow-b",
            )
        ],
    )

    collector = NewsCollector(limited_config, sources=[source_a, source_b])

    started = asyncio.get_running_loop().time()
    result = await collector.collect()
    elapsed = asyncio.get_running_loop().time() - started

    assert result.item_count == 2
    assert elapsed >= 0.055


@pytest.mark.asyncio
async def test_collector_with_rate_limited_source_reports_rate_limited_health(
    news_config,
    make_raw_news_item,
):
    ok_source = FakeCollectorSource(
        name="ok",
        items=[
            make_raw_news_item(
                title="Bitcoin ETF flow update",
                url="https://example.com/ok",
                source_item_id="ok",
            )
        ],
    )

    limited_source = FakeCollectorSource(
        name="limited",
        fetch_exception=NewsRateLimitError("Source returned HTTP 429"),
    )

    collector = NewsCollector(news_config, sources=[ok_source, limited_source])

    result = await collector.collect()

    assert result.item_count == 1
    assert result.has_errors is True

    health_by_name = {health.source_name: health for health in result.source_health}

    assert health_by_name["ok"].status == NewsSourceStatus.HEALTHY
    assert health_by_name["limited"].status == NewsSourceStatus.RATE_LIMITED
    assert health_by_name["limited"].last_fetch_status == NewsFetchStatus.RATE_LIMITED

    assert result.processing_result.metadata["failed_source_count"] == 1