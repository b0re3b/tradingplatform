from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from html import unescape
from typing import TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from core.logger import get_logger
from .config import NewsAIConfig, NewsFeatureConfig, NewsSourceConfig
from .enums import (
    NewsCategory,
    NewsEntityType,
    NewsFailureReason,
    NewsLanguage,
)
from .exceptions import (
    NewsErrorContext,
    NewsProcessingError,
    NewsValidationError,
)
from .models import (
    NewsEntity,
    NormalizedNewsItem,
    RawNewsItem,
    utc_now,
)

T = TypeVar("T")


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


_KNOWN_EXCHANGES = {
    "binance": "Binance",
    "bybit": "Bybit",
    "okx": "OKX",
    "coinbase": "Coinbase",
    "kraken": "Kraken",
    "mexc": "MEXC",
    "gate": "Gate",
    "kucoin": "KuCoin",
    "bitget": "Bitget",
    "bitmex": "BitMEX",
    "deribit": "Deribit",
}


_KNOWN_REGULATORS = {
    "sec": "SEC",
    "cftc": "CFTC",
    "federal reserve": "Federal Reserve",
    "fed": "Federal Reserve",
    "treasury": "US Treasury",
    "fca": "FCA",
    "esma": "ESMA",
    "finma": "FINMA",
}


_KNOWN_PROJECTS = {
    "bitcoin": ("Bitcoin", "BTC"),
    "ethereum": ("Ethereum", "ETH"),
    "solana": ("Solana", "SOL"),
    "bnb chain": ("BNB Chain", "BNB"),
    "binance coin": ("BNB", "BNB"),
    "xrp": ("XRP", "XRP"),
    "ripple": ("Ripple", "XRP"),
    "cardano": ("Cardano", "ADA"),
    "dogecoin": ("Dogecoin", "DOGE"),
    "avalanche": ("Avalanche", "AVAX"),
    "chainlink": ("Chainlink", "LINK"),
    "toncoin": ("Toncoin", "TON"),
    "the open network": ("The Open Network", "TON"),
    "polygon": ("Polygon", "POL"),
    "matic": ("Polygon", "MATIC"),
    "arbitrum": ("Arbitrum", "ARB"),
    "optimism": ("Optimism", "OP"),
    "sui": ("Sui", "SUI"),
    "aptos": ("Aptos", "APT"),
    "near protocol": ("NEAR Protocol", "NEAR"),
    "cosmos": ("Cosmos", "ATOM"),
    "polkadot": ("Polkadot", "DOT"),
    "litecoin": ("Litecoin", "LTC"),
    "tron": ("TRON", "TRX"),
    "tether": ("Tether", "USDT"),
    "circle": ("Circle", "USDC"),
    "usd coin": ("USD Coin", "USDC"),
}


_CATEGORY_KEYWORDS: dict[NewsCategory, tuple[str, ...]] = {
    NewsCategory.REGULATION: (
        "sec",
        "cftc",
        "regulator",
        "regulation",
        "lawsuit",
        "court",
        "charged",
        "settlement",
        "investigation",
        "probe",
        "ban",
        "compliance",
    ),
    NewsCategory.MACRO: (
        "fed",
        "fomc",
        "interest rate",
        "rate cut",
        "rate hike",
        "inflation",
        "cpi",
        "jobs report",
        "unemployment",
        "treasury",
        "dollar",
    ),
    NewsCategory.ETF: (
        "etf",
        "spot etf",
        "etf approval",
        "etf filing",
        "blackrock",
        "fidelity",
        "grayscale",
    ),
    NewsCategory.EXCHANGE: (
        "exchange",
        "binance",
        "bybit",
        "okx",
        "coinbase",
        "kraken",
        "mexc",
        "kucoin",
        "gate",
    ),
    NewsCategory.LISTING: (
        "will list",
        "listing",
        "listed on",
        "adds support",
        "trading opens",
        "launchpool",
        "launchpad",
    ),
    NewsCategory.DELISTING: (
        "delist",
        "delisting",
        "remove trading",
        "suspend trading",
        "trading suspension",
    ),
    NewsCategory.HACK: (
        "hack",
        "hacked",
        "security breach",
        "stolen funds",
        "drained",
    ),
    NewsCategory.EXPLOIT: (
        "exploit",
        "exploited",
        "vulnerability",
        "attack",
    ),
    NewsCategory.SECURITY: (
        "security incident",
        "breach",
        "phishing",
        "malware",
        "compromised",
    ),
    NewsCategory.STABLECOIN: (
        "stablecoin",
        "usdt",
        "usdc",
        "tether",
        "circle",
        "depeg",
        "peg",
    ),
    NewsCategory.DEFI: (
        "defi",
        "dex",
        "yield",
        "lending protocol",
        "liquidity pool",
        "amm",
        "aave",
        "compound",
        "uniswap",
        "curve",
    ),
    NewsCategory.NFT: (
        "nft",
        "opensea",
        "blur",
        "ordinals",
        "collection",
    ),
    NewsCategory.LAYER_1: (
        "layer 1",
        "l1",
        "mainnet",
        "validator",
        "consensus",
    ),
    NewsCategory.LAYER_2: (
        "layer 2",
        "l2",
        "rollup",
        "zk rollup",
        "optimistic rollup",
    ),
    NewsCategory.PARTNERSHIP: (
        "partnership",
        "partners with",
        "collaboration",
        "integrates with",
        "integration",
    ),
    NewsCategory.FUNDING: (
        "raises",
        "funding round",
        "investment",
        "venture",
        "series a",
        "series b",
    ),
    NewsCategory.TOKEN_UNLOCK: (
        "token unlock",
        "unlock",
        "vesting",
        "supply release",
    ),
    NewsCategory.AIRDROP: (
        "airdrop",
        "claim",
        "eligibility",
        "snapshot",
    ),
    NewsCategory.GOVERNANCE: (
        "governance",
        "proposal",
        "vote",
        "dao",
    ),
    NewsCategory.LEGAL: (
        "legal",
        "lawsuit",
        "court",
        "judge",
        "settlement",
    ),
    NewsCategory.BANKRUPTCY: (
        "bankruptcy",
        "insolvent",
        "restructuring",
        "chapter 11",
    ),
    NewsCategory.RUMOR: (
        "rumor",
        "reportedly",
        "sources say",
        "unconfirmed",
        "allegedly",
    ),
}


_SYMBOL_PATTERN = re.compile(r"(?<![A-Z0-9])\$?([A-Z]{2,12})(?![A-Z0-9])")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_NON_CONTENT_CHARS_PATTERN = re.compile(r"[^\w\s$./:%#@+-]")


@dataclass(slots=True, frozen=True)
class ProcessedNewsBatch:
    """
    Processing result for a batch of raw news items.
    """

    items: tuple[NormalizedNewsItem, ...]
    failed_count: int
    errors: tuple[str, ...]

    @property
    def processed_count(self) -> int:
        return len(self.items)

    @property
    def total_count(self) -> int:
        return self.processed_count + self.failed_count


class NewsProcessor:
    """
    Converts RawNewsItem into NormalizedNewsItem.

    This component is deterministic and does not rely on LLM calls.
    """

    def __init__(
        self,
        config: NewsAIConfig,
        *,
        source_configs: Iterable[NewsSourceConfig] | None = None,
    ) -> None:
        self.config = config
        self.feature_config: NewsFeatureConfig = config.features
        self.logger = get_logger(__name__)

        source_configs_tuple = tuple(source_configs or config.source_configs)
        self._source_config_by_name = {
            source.name: source for source in source_configs_tuple
        }
        self._tracked_symbols = set(config.normalized_tracked_symbols)

    def process(self, item: RawNewsItem) -> NormalizedNewsItem:
        """
        Process a single raw news item.
        """

        try:
            title = self._clean_text(item.title)
            text = self._build_text(item)

            if not title:
                raise NewsValidationError(
                    "Raw news item title is empty after normalization",
                    context=NewsErrorContext(
                        reason=NewsFailureReason.VALIDATION_ERROR,
                        source_name=item.source_name,
                        source_type=str(item.source_type),
                        url=item.url,
                    ),
                )

            if not text:
                raise NewsValidationError(
                    "Raw news item text is empty after normalization",
                    context=NewsErrorContext(
                        reason=NewsFailureReason.VALIDATION_ERROR,
                        source_name=item.source_name,
                        source_type=str(item.source_type),
                        url=item.url,
                        details={"title": title[:160]},
                    ),
                )

            canonical_url = self._canonicalize_url(item.url)
            title_hash = self._hash_text(title)
            content_hash = self._hash_text(text)

            language = self._detect_language(item, text)
            categories = self._classify_categories(title=title, text=text, raw=item)
            entities = self._extract_entities(title=title, text=text, raw=item)
            symbols = self._extract_symbols(title=title, text=text, entities=entities)

            source_config = self._source_config_by_name.get(item.source_name)
            source_reputation_score = (
                source_config.source_reputation_score
                if source_config is not None
                else 0.5
            )

            news_id = self._generate_news_id(
                source_name=item.source_name,
                source_item_id=item.source_item_id,
                canonical_url=canonical_url,
                title_hash=title_hash,
                content_hash=content_hash,
            )

            return NormalizedNewsItem(
                news_id=news_id,
                source_name=item.source_name,
                source_type=item.source_type,
                title=title,
                text=text,
                url=item.url,
                canonical_url=canonical_url,
                summary=self._clean_text(item.summary) or None,
                author=self._clean_text(item.author) or None,
                source_item_id=item.source_item_id,
                published_at=item.published_at,
                fetched_at=item.fetched_at,
                processed_at=utc_now(),
                language=language,
                categories=categories,
                entities=entities,
                symbols=symbols,
                title_hash=title_hash,
                content_hash=content_hash,
                source_reputation_score=source_reputation_score,
                metadata={
                    "raw_language": str(item.language),
                    "has_body": bool(item.body),
                    "has_summary": bool(item.summary),
                },
            )

        except NewsProcessingError:
            raise

        except Exception as exc:
            raise NewsProcessingError(
                "Failed to process raw news item",
                context=NewsErrorContext(
                    reason=NewsFailureReason.PARSE_ERROR,
                    source_name=item.source_name,
                    source_type=str(item.source_type),
                    url=item.url,
                    details={"title": item.title[:160]},
                ),
                cause=exc,
            ) from exc

    def process_many(self, items: Iterable[RawNewsItem]) -> ProcessedNewsBatch:
        """
        Process a batch of raw news items.

        Invalid items are skipped and returned as errors instead of failing the
        whole batch.
        """

        processed: list[NormalizedNewsItem] = []
        errors: list[str] = []
        failed_count = 0

        for item in items:
            try:
                processed.append(self.process(item))
            except NewsProcessingError as exc:
                failed_count += 1
                errors.append(str(exc))
                self.logger.warning(
                    "Failed to process news item",
                    extra={
                        "source_name": item.source_name,
                        "url": item.url,
                        "error": str(exc),
                    },
                )

        return ProcessedNewsBatch(
            items=tuple(processed),
            failed_count=failed_count,
            errors=tuple(errors),
        )

    def _build_text(self, item: RawNewsItem) -> str:
        parts = [
            item.title,
            item.summary or "",
            item.body or "",
        ]

        cleaned_parts = [self._clean_text(part) for part in parts if part]
        cleaned_parts = [part for part in cleaned_parts if part]

        return "\n".join(dict.fromkeys(cleaned_parts))

    def _clean_text(self, value: str | None) -> str:
        if not value:
            return ""

        text = unescape(str(value))
        text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.DOTALL)
        text = _HTML_TAG_PATTERN.sub(" ", text)
        text = text.replace("\u200b", "")
        text = text.replace("\xa0", " ")
        text = _WHITESPACE_PATTERN.sub(" ", text)
        return text.strip()

    def _normalize_for_hash(self, value: str | None) -> str:
        if not value:
            return ""

        text = self._clean_text(value).lower()
        text = _NON_CONTENT_CHARS_PATTERN.sub(" ", text)
        text = _WHITESPACE_PATTERN.sub(" ", text)
        return text.strip()

    def _hash_text(self, value: str | None) -> str | None:
        normalized = self._normalize_for_hash(value)
        if not normalized:
            return None

        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _canonicalize_url(self, value: str | None) -> str | None:
        if not value:
            return None

        raw = value.strip()
        if not raw:
            return None

        try:
            parts = urlsplit(raw)
        except ValueError:
            return raw.lower()

        scheme = parts.scheme.lower() or "https"
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

    def _generate_news_id(
        self,
        *,
        source_name: str,
        source_item_id: str | None,
        canonical_url: str | None,
        title_hash: str | None,
        content_hash: str | None,
    ) -> str:
        """
        Generate deterministic news_id from the most stable available fields.
        """

        candidates = [
            f"source_item:{source_name}:{source_item_id}" if source_item_id else None,
            f"url:{canonical_url}" if canonical_url else None,
            f"title:{title_hash}" if title_hash else None,
            f"content:{content_hash}" if content_hash else None,
        ]

        for candidate in candidates:
            if candidate:
                digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()
                return f"news_{digest}"

        fallback = f"{source_name}:{utc_now().isoformat()}"
        digest = hashlib.sha1(fallback.encode("utf-8")).hexdigest()
        return f"news_{digest}"

    def _detect_language(self, item: RawNewsItem, text: str) -> NewsLanguage:
        """
        Basic language detection.

        This is intentionally lightweight. Later it can be replaced with a
        dedicated language detector if needed.
        """

        if item.language != NewsLanguage.UNKNOWN:
            return item.language

        lowered = text.lower()

        cyrillic_chars = len(re.findall(r"[а-яіїєґё]", lowered))
        latin_chars = len(re.findall(r"[a-z]", lowered))

        if cyrillic_chars > latin_chars and cyrillic_chars > 20:
            if any(token in lowered for token in ("що", "для", "від", "буде", "ринок")):
                return NewsLanguage.UK
            return NewsLanguage.RU

        if latin_chars > 20:
            return NewsLanguage.EN

        return self.config.default_language

    def _classify_categories(
        self,
        *,
        title: str,
        text: str,
        raw: RawNewsItem,
    ) -> tuple[NewsCategory, ...]:
        lowered = f"{title}\n{text}".lower()
        matched: list[NewsCategory] = []

        source_config = self._source_config_by_name.get(raw.source_name)
        if source_config:
            for category in source_config.default_categories:
                if category != NewsCategory.UNKNOWN:
                    matched.append(category)

        for category, keywords in _CATEGORY_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                matched.append(category)

        deduped = self._deduplicate_preserve_order(matched)

        if not deduped:
            return (NewsCategory.GENERAL,)

        return tuple(deduped)

    def _extract_entities(
        self,
        *,
        title: str,
        text: str,
        raw: RawNewsItem,
    ) -> tuple[NewsEntity, ...]:
        combined = f"{title}\n{text}"
        lowered = combined.lower()

        entities: list[NewsEntity] = []

        for keyword, display_name in _KNOWN_EXCHANGES.items():
            if self._contains_phrase(lowered, keyword):
                entities.append(
                    NewsEntity(
                        name=display_name,
                        entity_type=NewsEntityType.EXCHANGE,
                        confidence=0.90,
                    )
                )

        for keyword, display_name in _KNOWN_REGULATORS.items():
            if self._contains_phrase(lowered, keyword):
                entities.append(
                    NewsEntity(
                        name=display_name,
                        entity_type=NewsEntityType.REGULATOR,
                        confidence=0.90,
                    )
                )

        for keyword, project_data in _KNOWN_PROJECTS.items():
            display_name, symbol = project_data
            if self._contains_phrase(lowered, keyword):
                entities.append(
                    NewsEntity(
                        name=display_name,
                        entity_type=NewsEntityType.PROJECT,
                        symbol=symbol,
                        confidence=0.88,
                    )
                )

        symbol_entities = self._extract_symbol_entities(combined)
        entities.extend(symbol_entities)

        deduped = self._deduplicate_entities(entities)

        return tuple(deduped[: self.feature_config.max_entities_per_item])

    def _extract_symbol_entities(self, text: str) -> list[NewsEntity]:
        symbols: list[str] = []

        for match in _SYMBOL_PATTERN.finditer(text):
            symbol = match.group(1).upper()

            if not self._is_valid_symbol(symbol):
                continue

            if self._tracked_symbols and symbol not in self._tracked_symbols:
                raw_token = match.group(0)
                if not raw_token.startswith("$"):
                    continue

            symbols.append(symbol)

        deduped_symbols = self._deduplicate_preserve_order(symbols)

        return [
            NewsEntity(
                name=symbol,
                entity_type=NewsEntityType.SYMBOL,
                symbol=symbol,
                confidence=0.80,
            )
            for symbol in deduped_symbols[: self.feature_config.max_symbols_per_item]
        ]

    def _extract_symbols(
        self,
        *,
        title: str,
        text: str,
        entities: tuple[NewsEntity, ...],
    ) -> tuple[str, ...]:
        symbols: list[str] = []

        for entity in entities:
            if entity.symbol:
                symbol = entity.symbol.upper()
                if self._is_valid_symbol(symbol):
                    symbols.append(symbol)

        for match in _SYMBOL_PATTERN.finditer(f"{title}\n{text}"):
            symbol = match.group(1).upper()
            if self._is_valid_symbol(symbol):
                if not self._tracked_symbols or symbol in self._tracked_symbols:
                    symbols.append(symbol)

        deduped = self._deduplicate_preserve_order(symbols)
        return tuple(deduped[: self.feature_config.max_symbols_per_item])

    def _is_valid_symbol(self, symbol: str) -> bool:
        if not symbol:
            return False

        if len(symbol) < self.feature_config.min_symbol_length:
            return False

        if len(symbol) > self.feature_config.max_symbol_length:
            return False

        common_false_positives = {
            "THE",
            "AND",
            "FOR",
            "WITH",
            "FROM",
            "THIS",
            "THAT",
            "WILL",
            "JUST",
            "NEWS",
            "USD",
            "EUR",
            "CEO",
            "ETF",
            "SEC",
            "CPI",
            "FOMC",
            "API",
            "NFT",
            "TVL",
            "ATH",
            "ATL",
            "US",
            "EU",
            "UK",
        }

        return symbol not in common_false_positives

    def _contains_phrase(self, lowered_text: str, phrase: str) -> bool:
        phrase = phrase.lower().strip()
        if not phrase:
            return False

        if " " in phrase:
            return phrase in lowered_text

        return (
            re.search(
                rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
                lowered_text,
            )
            is not None
        )

    def _deduplicate_entities(
        self,
        entities: Iterable[NewsEntity],
    ) -> list[NewsEntity]:
        seen: set[tuple[str, str, str | None]] = set()
        result: list[NewsEntity] = []

        for entity in entities:
            key = (
                entity.name.lower(),
                str(entity.entity_type),
                entity.symbol.upper() if entity.symbol else None,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(entity)

        return result

    def _deduplicate_preserve_order(self, items: Iterable[T]) -> list[T]:
        seen: set[T] = set()
        result: list[T] = []

        for item in items:
            if item in seen:
                continue

            seen.add(item)
            result.append(item)

        return result


__all__ = [
    "ProcessedNewsBatch",
    "NewsProcessor",
]