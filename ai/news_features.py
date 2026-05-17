from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TypeVar

from core.logger import get_logger

from .config import NewsAIConfig, NewsFeatureConfig, NewsSourceConfig
from .enums import NewsCategory
from .exceptions import (
    NewsErrorContext,
    NewsFeatureExtractionError,
)
from .models import NewsFeatures, NormalizedNewsItem


T = TypeVar("T")


_WORD_PATTERN = re.compile(r"[a-zA-Z0-9$._-]+")
_NUMBER_PATTERN = re.compile(r"(?<!\w)(?:\d+(?:\.\d+)?%?|\$\d+(?:\.\d+)?[mbkMBK]?)(?!\w)")
_ALL_CAPS_WORD_PATTERN = re.compile(r"\b[A-Z]{2,12}\b")


_EXTRA_HIGH_IMPACT_KEYWORDS: tuple[str, ...] = (
    "breaking",
    "urgent",
    "just in",
    "approval",
    "approved",
    "rejected",
    "delay",
    "delayed",
    "hack",
    "exploit",
    "lawsuit",
    "ban",
    "settlement",
    "bankruptcy",
    "insolvent",
    "halt",
    "suspend",
    "delist",
    "listing",
    "mainnet",
    "airdrop",
    "token unlock",
    "etf",
    "rate cut",
    "rate hike",
)


_EXTRA_LOW_QUALITY_KEYWORDS: tuple[str, ...] = (
    "price prediction",
    "could explode",
    "next 100x",
    "moon",
    "gem",
    "you won't believe",
    "shocking",
    "secret",
)


_CATEGORY_FEATURE_KEYWORDS: dict[NewsCategory, tuple[str, ...]] = {
    NewsCategory.REGULATION: (
        "sec",
        "cftc",
        "regulator",
        "regulation",
        "lawsuit",
        "court",
        "settlement",
        "charged",
        "probe",
        "investigation",
        "ban",
    ),
    NewsCategory.MACRO: (
        "fed",
        "fomc",
        "cpi",
        "inflation",
        "rate cut",
        "rate hike",
        "interest rate",
        "treasury",
        "dollar",
        "jobs report",
    ),
    NewsCategory.ETF: (
        "etf",
        "spot etf",
        "approval",
        "approved",
        "filing",
        "blackrock",
        "fidelity",
        "grayscale",
    ),
    NewsCategory.HACK: (
        "hack",
        "hacked",
        "breach",
        "stolen",
        "drained",
        "security incident",
    ),
    NewsCategory.EXPLOIT: (
        "exploit",
        "exploited",
        "vulnerability",
        "attack",
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
    NewsCategory.BANKRUPTCY: (
        "bankruptcy",
        "insolvent",
        "restructuring",
        "chapter 11",
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
    NewsCategory.PARTNERSHIP: (
        "partnership",
        "partners with",
        "collaboration",
        "integrates with",
        "integration",
    ),
    NewsCategory.RUMOR: (
        "rumor",
        "reportedly",
        "sources say",
        "unconfirmed",
        "allegedly",
    ),
}


class NewsFeatureExtractor:
    """
    Extracts deterministic features from normalized news.

    NewsFeatureExtractor does not decide final sentiment/impact. It only
    prepares structured signals for NewsScorer.
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

    def extract(self, item: NormalizedNewsItem) -> NewsFeatures:
        """
        Extract features from a single normalized news item.
        """

        try:
            title = item.title or ""
            text = item.text or ""
            combined_text = f"{title}\n{text}"
            lowered_text = combined_text.lower()
            lowered_title = title.lower()

            source_config = self._source_config_by_name.get(item.source_name)

            source_reputation_score = self._source_reputation_score(
                item=item,
                source_config=source_config,
                lowered_text=lowered_text,
            )

            matched_urgent_keywords = self._matched_keywords(
                lowered_text,
                self.feature_config.urgent_keywords,
            )
            matched_regulatory_keywords = self._matched_keywords(
                lowered_text,
                self.feature_config.regulatory_keywords,
            )
            matched_macro_keywords = self._matched_keywords(
                lowered_text,
                self.feature_config.macro_keywords,
            )
            matched_hack_keywords = self._matched_keywords(
                lowered_text,
                self.feature_config.hack_keywords,
            )
            matched_listing_keywords = self._matched_keywords(
                lowered_text,
                self.feature_config.listing_keywords,
            )
            matched_delisting_keywords = self._matched_keywords(
                lowered_text,
                self.feature_config.delisting_keywords,
            )
            matched_etf_keywords = self._matched_keywords(
                lowered_text,
                self.feature_config.etf_keywords,
            )
            matched_positive_keywords = self._matched_keywords(
                lowered_text,
                self.feature_config.positive_keywords,
            )
            matched_negative_keywords = self._matched_keywords(
                lowered_text,
                self.feature_config.negative_keywords,
            )
            matched_rumor_keywords = self._matched_keywords(
                lowered_text,
                self.feature_config.rumor_keywords,
            )

            matched_category_keywords = self._matched_category_keywords(lowered_text)
            matched_high_impact_keywords = self._matched_keywords(
                lowered_text,
                _EXTRA_HIGH_IMPACT_KEYWORDS,
            )
            matched_low_quality_keywords = self._matched_keywords(
                lowered_text,
                _EXTRA_LOW_QUALITY_KEYWORDS,
            )

            matched_keywords = self._deduplicate_preserve_order(
                (
                    *matched_urgent_keywords,
                    *matched_regulatory_keywords,
                    *matched_macro_keywords,
                    *matched_hack_keywords,
                    *matched_listing_keywords,
                    *matched_delisting_keywords,
                    *matched_etf_keywords,
                    *matched_positive_keywords,
                    *matched_negative_keywords,
                    *matched_rumor_keywords,
                    *matched_category_keywords,
                    *matched_high_impact_keywords,
                    *matched_low_quality_keywords,
                )
            )

            has_exploit_keywords = self._has_any_phrase(
                lowered_text,
                ("exploit", "exploited", "vulnerability", "attack"),
            )
            has_lawsuit_keywords = self._has_any_phrase(
                lowered_text,
                ("lawsuit", "sued", "court", "settlement", "charged", "judge"),
            )
            has_bankruptcy_keywords = self._has_any_phrase(
                lowered_text,
                ("bankruptcy", "insolvent", "restructuring", "chapter 11"),
            )
            has_partnership_keywords = self._has_any_phrase(
                lowered_text,
                ("partnership", "partners with", "collaboration", "integration"),
            )
            has_token_unlock_keywords = self._has_any_phrase(
                lowered_text,
                ("token unlock", "unlock", "vesting", "supply release"),
            )
            has_airdrop_keywords = self._has_any_phrase(
                lowered_text,
                ("airdrop", "claim", "eligibility", "snapshot"),
            )

            is_official_source = self._is_official_source(
                item=item,
                source_config=source_config,
            )
            is_exchange_source = self._is_exchange_source(
                item=item,
                source_config=source_config,
            )

            title_strength_score = self._title_strength_score(
                title=title,
                lowered_title=lowered_title,
                matched_high_impact_keywords=matched_high_impact_keywords,
            )

            text_length_score = self._text_length_score(text)
            is_breaking_news = self._is_breaking_news(
                lowered_title=lowered_title,
                matched_urgent_keywords=matched_urgent_keywords,
            )
            is_low_quality_source = self._is_low_quality_source(
                item=item,
                lowered_text=lowered_text,
                matched_low_quality_keywords=matched_low_quality_keywords,
            )

            return NewsFeatures(
                news_id=item.news_id,
                source_reputation_score=source_reputation_score,
                title_strength_score=title_strength_score,
                text_length_score=text_length_score,
                symbol_count=len(item.symbols),
                entity_count=len(item.entities),
                category_count=len(item.categories),
                has_urgent_keywords=bool(matched_urgent_keywords),
                has_regulatory_keywords=bool(matched_regulatory_keywords),
                has_macro_keywords=bool(matched_macro_keywords),
                has_hack_keywords=bool(matched_hack_keywords),
                has_exploit_keywords=has_exploit_keywords,
                has_listing_keywords=bool(matched_listing_keywords),
                has_delisting_keywords=bool(matched_delisting_keywords),
                has_etf_keywords=bool(matched_etf_keywords),
                has_lawsuit_keywords=has_lawsuit_keywords,
                has_bankruptcy_keywords=has_bankruptcy_keywords,
                has_partnership_keywords=has_partnership_keywords,
                has_token_unlock_keywords=has_token_unlock_keywords,
                has_airdrop_keywords=has_airdrop_keywords,
                has_rumor_keywords=bool(matched_rumor_keywords),
                is_official_source=is_official_source,
                is_exchange_source=is_exchange_source,
                is_breaking_news=is_breaking_news,
                is_low_quality_source=is_low_quality_source,
                matched_keywords=tuple(matched_keywords),
                matched_negative_keywords=tuple(matched_negative_keywords),
                matched_positive_keywords=tuple(matched_positive_keywords),
                raw_feature_values={
                    "title_word_count": self._word_count(title),
                    "text_word_count": self._word_count(text),
                    "title_char_count": len(title),
                    "text_char_count": len(text),
                    "number_count": self._number_count(combined_text),
                    "all_caps_word_count": self._all_caps_word_count(title),
                    "matched_urgent_keyword_count": len(matched_urgent_keywords),
                    "matched_regulatory_keyword_count": len(matched_regulatory_keywords),
                    "matched_macro_keyword_count": len(matched_macro_keywords),
                    "matched_hack_keyword_count": len(matched_hack_keywords),
                    "matched_listing_keyword_count": len(matched_listing_keywords),
                    "matched_delisting_keyword_count": len(matched_delisting_keywords),
                    "matched_etf_keyword_count": len(matched_etf_keywords),
                    "matched_positive_keyword_count": len(matched_positive_keywords),
                    "matched_negative_keyword_count": len(matched_negative_keywords),
                    "matched_rumor_keyword_count": len(matched_rumor_keywords),
                    "matched_category_keyword_count": len(matched_category_keywords),
                    "matched_high_impact_keyword_count": len(matched_high_impact_keywords),
                    "matched_low_quality_keyword_count": len(matched_low_quality_keywords),
                    "has_numbers": self._has_numbers(combined_text),
                    "has_question_mark": "?" in title,
                    "has_exclamation_mark": "!" in title,
                    "category_names": ",".join(str(category) for category in item.categories),
                    "primary_category": str(item.primary_category),
                    "primary_symbol": item.primary_symbol or "",
                },
            )

        except NewsFeatureExtractionError:
            raise

        except Exception as exc:
            raise NewsFeatureExtractionError(
                "Failed to extract news features",
                context=NewsErrorContext(
                    news_id=item.news_id,
                    source_name=item.source_name,
                    source_type=str(item.source_type),
                    url=item.url,
                    details={
                        "title": item.title[:160],
                        "symbols": list(item.symbols),
                        "categories": [str(category) for category in item.categories],
                    },
                ),
                cause=exc,
            ) from exc

    def extract_many(self, items: Iterable[NormalizedNewsItem]) -> dict[str, NewsFeatures]:
        """
        Extract features for multiple normalized news items.

        Returns:
            Mapping news_id -> NewsFeatures.
        """

        features_by_id: dict[str, NewsFeatures] = {}

        for item in items:
            features_by_id[item.news_id] = self.extract(item)

        return features_by_id

    def _source_reputation_score(
        self,
        *,
        item: NormalizedNewsItem,
        source_config: NewsSourceConfig | None,
        lowered_text: str,
    ) -> float:
        score = item.source_reputation_score

        if source_config is not None:
            score = source_config.source_reputation_score

            if source_config.is_official_source:
                score += self.feature_config.official_source_reputation_boost

            if source_config.is_exchange_source:
                score += self.feature_config.exchange_source_reputation_boost

        if self._is_low_quality_text(lowered_text):
            score -= self.feature_config.low_quality_source_penalty

        return self._clamp01(score)

    def _title_strength_score(
        self,
        *,
        title: str,
        lowered_title: str,
        matched_high_impact_keywords: tuple[str, ...],
    ) -> float:
        if not title:
            return 0.0

        score = 0.15

        word_count = self._word_count(title)
        char_count = len(title)

        if 6 <= word_count <= 18:
            score += 0.20
        elif 3 <= word_count <= 28:
            score += 0.10

        if 40 <= char_count <= 140:
            score += 0.20
        elif 20 <= char_count <= 180:
            score += 0.10

        if matched_high_impact_keywords:
            score += min(0.25, 0.06 * len(matched_high_impact_keywords))

        if self._has_numbers(title):
            score += 0.08

        if ":" in title or "-" in title:
            score += 0.04

        if "?" in title:
            score -= 0.10

        if any(phrase in lowered_title for phrase in _EXTRA_LOW_QUALITY_KEYWORDS):
            score -= 0.25

        if self._all_caps_word_count(title) >= 4:
            score -= 0.08

        return self._clamp01(score)

    def _text_length_score(self, text: str) -> float:
        """
        Estimate whether text length is useful enough for scoring.

        Too short: weak context.
        Medium: best.
        Too long: still useful but may contain noise.
        """

        word_count = self._word_count(text)

        if word_count <= 0:
            return 0.0

        if word_count < 12:
            return 0.25

        if word_count < 35:
            return 0.50

        if word_count < 180:
            return 0.90

        if word_count < 600:
            return 1.0

        if word_count < 1200:
            return 0.85

        return 0.70

    def _is_official_source(
        self,
        *,
        item: NormalizedNewsItem,
        source_config: NewsSourceConfig | None,
    ) -> bool:
        if source_config is not None and source_config.is_official_source:
            return True

        source_name = item.source_name.lower()
        url = (item.canonical_url or item.url or "").lower()

        official_markers = (
            "official",
            "announcements",
            "blog",
            "press",
            "foundation",
        )

        official_domains = (
            "binance.com",
            "bybit.com",
            "okx.com",
            "coinbase.com",
            "kraken.com",
            "ethereum.org",
            "solana.com",
            "chain.link",
            "circle.com",
            "tether.to",
            "sec.gov",
            "cftc.gov",
            "federalreserve.gov",
        )

        return any(marker in source_name for marker in official_markers) or any(
            domain in url for domain in official_domains
        )

    def _is_exchange_source(
        self,
        *,
        item: NormalizedNewsItem,
        source_config: NewsSourceConfig | None,
    ) -> bool:
        if source_config is not None and source_config.is_exchange_source:
            return True

        source_name = item.source_name.lower()
        url = (item.canonical_url or item.url or "").lower()

        exchange_markers = (
            "binance",
            "bybit",
            "okx",
            "coinbase",
            "kraken",
            "mexc",
            "kucoin",
            "gate",
            "bitget",
        )

        return any(marker in source_name for marker in exchange_markers) or any(
            marker in url for marker in exchange_markers
        )

    def _is_breaking_news(
        self,
        *,
        lowered_title: str,
        matched_urgent_keywords: tuple[str, ...],
    ) -> bool:
        if matched_urgent_keywords:
            return True

        breaking_patterns = (
            "breaking:",
            "just in:",
            "urgent:",
            "[breaking]",
            "[urgent]",
        )

        return any(pattern in lowered_title for pattern in breaking_patterns)

    def _is_low_quality_source(
        self,
        *,
        item: NormalizedNewsItem,
        lowered_text: str,
        matched_low_quality_keywords: tuple[str, ...],
    ) -> bool:
        if matched_low_quality_keywords:
            return True

        source_name = item.source_name.lower()

        low_quality_source_markers = (
            "prediction",
            "moon",
            "pump",
            "100x",
            "gem",
        )

        return any(marker in source_name for marker in low_quality_source_markers) or (
            self._is_low_quality_text(lowered_text)
        )

    def _is_low_quality_text(self, lowered_text: str) -> bool:
        return any(phrase in lowered_text for phrase in _EXTRA_LOW_QUALITY_KEYWORDS)

    def _matched_category_keywords(self, lowered_text: str) -> tuple[str, ...]:
        matched: list[str] = []

        for keywords in _CATEGORY_FEATURE_KEYWORDS.values():
            matched.extend(self._matched_keywords(lowered_text, keywords))

        return tuple(self._deduplicate_preserve_order(matched))

    def _matched_keywords(
        self,
        lowered_text: str,
        keywords: Iterable[str],
    ) -> tuple[str, ...]:
        matched: list[str] = []

        for keyword in keywords:
            normalized_keyword = keyword.strip().lower()
            if not normalized_keyword:
                continue

            if self._contains_phrase(lowered_text, normalized_keyword):
                matched.append(normalized_keyword)

        return tuple(self._deduplicate_preserve_order(matched))

    def _has_any_phrase(
        self,
        lowered_text: str,
        phrases: Iterable[str],
    ) -> bool:
        return any(
            self._contains_phrase(lowered_text, phrase.strip().lower())
            for phrase in phrases
            if phrase.strip()
        )

    def _contains_phrase(self, lowered_text: str, phrase: str) -> bool:
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

    def _word_count(self, text: str) -> int:
        return len(_WORD_PATTERN.findall(text or ""))

    def _number_count(self, text: str) -> int:
        return len(_NUMBER_PATTERN.findall(text or ""))

    def _has_numbers(self, text: str) -> bool:
        return bool(_NUMBER_PATTERN.search(text or ""))

    def _all_caps_word_count(self, text: str) -> int:
        return len(_ALL_CAPS_WORD_PATTERN.findall(text or ""))

    def _deduplicate_preserve_order(self, items: Iterable[T]) -> list[T]:
        seen: set[T] = set()
        result: list[T] = []

        for item in items:
            if item in seen:
                continue

            seen.add(item)
            result.append(item)

        return result

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))


__all__ = [
    "NewsFeatureExtractor",
]