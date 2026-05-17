from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from core.logger import get_logger
from .config import NewsAIConfig, NewsScoringConfig
from .enums import (
    LLMOutputStatus,
    NewsAlertType,
    NewsCategory,
    NewsImpactLevel,
    NewsMarketBias,
    NewsRelevanceLevel,
    NewsSentiment,
    NewsTimeHorizon,
    NewsUrgencyLevel,
)
from .exceptions import NewsErrorContext, NewsScoringError
from .models import (
    NewsFeatures,
    NewsLLMResult,
    NewsScore,
    NormalizedNewsItem,
    utc_now,
)

_HIGH_IMPACT_CATEGORIES: set[NewsCategory] = {
    NewsCategory.REGULATION,
    NewsCategory.MACRO,
    NewsCategory.ETF,
    NewsCategory.EXCHANGE,
    NewsCategory.LISTING,
    NewsCategory.DELISTING,
    NewsCategory.HACK,
    NewsCategory.EXPLOIT,
    NewsCategory.SECURITY,
    NewsCategory.STABLECOIN,
    NewsCategory.BANKRUPTCY,
    NewsCategory.TOKEN_UNLOCK,
}

_NEGATIVE_CATEGORIES: set[NewsCategory] = {
    NewsCategory.REGULATION,
    NewsCategory.DELISTING,
    NewsCategory.HACK,
    NewsCategory.EXPLOIT,
    NewsCategory.SECURITY,
    NewsCategory.LEGAL,
    NewsCategory.BANKRUPTCY,
}

_POSITIVE_CATEGORIES: set[NewsCategory] = {
    NewsCategory.LISTING,
    NewsCategory.PARTNERSHIP,
    NewsCategory.FUNDING,
    NewsCategory.AIRDROP,
}

_URGENT_CATEGORIES: set[NewsCategory] = {
    NewsCategory.HACK,
    NewsCategory.EXPLOIT,
    NewsCategory.SECURITY,
    NewsCategory.EXCHANGE,
    NewsCategory.REGULATION,
    NewsCategory.MACRO,
    NewsCategory.ETF,
    NewsCategory.BANKRUPTCY,
}


class NewsScorer:
    """
    Final scoring component for normalized news.

    The scorer is deterministic-first. Optional LLM output can improve final
    interpretation, but the package remains usable without LLM.
    """

    def __init__(self, config: NewsAIConfig) -> None:
        self.config = config
        self.scoring_config: NewsScoringConfig = config.scoring
        self.logger = get_logger(__name__)

    def score(
        self,
        item: NormalizedNewsItem,
        features: NewsFeatures,
        llm_result: NewsLLMResult | None = None,
    ) -> NewsScore:
        """
        Calculate final score for a single normalized news item.
        """

        try:
            rule_sentiment_score = self._rule_sentiment_score(item, features)
            rule_impact_score = self._rule_impact_score(item, features)
            rule_urgency_score = self._rule_urgency_score(item, features)
            rule_relevance_score = self._rule_relevance_score(item, features)
            rule_confidence_score = self._rule_confidence_score(item, features)
            novelty_score = self._novelty_score(item)

            llm_weight = self._effective_llm_weight(llm_result)
            rule_weight = 1.0 - llm_weight

            sentiment_score = self._blend_signed(
                rule_value=rule_sentiment_score,
                llm_value=llm_result.sentiment_score if llm_result else None,
                rule_weight=rule_weight,
                llm_weight=llm_weight,
            )
            impact_score = self._blend01(
                rule_value=rule_impact_score,
                llm_value=llm_result.impact_score if llm_result else None,
                rule_weight=rule_weight,
                llm_weight=llm_weight,
            )
            urgency_score = self._blend01(
                rule_value=rule_urgency_score,
                llm_value=llm_result.urgency_score if llm_result else None,
                rule_weight=rule_weight,
                llm_weight=llm_weight,
            )
            relevance_score = self._blend01(
                rule_value=rule_relevance_score,
                llm_value=llm_result.relevance_score if llm_result else None,
                rule_weight=rule_weight,
                llm_weight=llm_weight,
            )
            confidence_score = self._blend01(
                rule_value=rule_confidence_score,
                llm_value=llm_result.confidence_score if llm_result else None,
                rule_weight=rule_weight,
                llm_weight=llm_weight,
            )

            # Новина без релевантності до нашого trading universe не має бути
            # надмірно high-impact у фінальному результаті.
            impact_score = self._clamp01(
                impact_score * (0.65 + 0.35 * relevance_score)
            )

            categories = self._merge_categories(item, llm_result)
            sentiment = self._sentiment_label(sentiment_score, llm_result)
            market_bias = self._market_bias(
                sentiment_score=sentiment_score,
                item=item,
                features=features,
                llm_result=llm_result,
            )
            impact_level = self._impact_level(impact_score)
            urgency_level = self._urgency_level(urgency_score)
            relevance_level = self._relevance_level(relevance_score)
            time_horizon = self._time_horizon(
                item=item,
                features=features,
                urgency_score=urgency_score,
                llm_result=llm_result,
            )
            alert_types = self._alert_types(
                item=item,
                features=features,
                impact_score=impact_score,
                urgency_score=urgency_score,
                impact_level=impact_level,
            )

            summary = self._summary(item=item, llm_result=llm_result)
            explanation = self._explanation(
                item=item,
                features=features,
                llm_result=llm_result,
                sentiment_score=sentiment_score,
                impact_score=impact_score,
                urgency_score=urgency_score,
                relevance_score=relevance_score,
                confidence_score=confidence_score,
            )
            trading_notes = self._trading_notes(
                item=item,
                market_bias=market_bias,
                impact_level=impact_level,
                urgency_level=urgency_level,
                time_horizon=time_horizon,
                llm_result=llm_result,
            )

            return NewsScore(
                news_id=item.news_id,
                sentiment_score=sentiment_score,
                impact_score=impact_score,
                confidence_score=confidence_score,
                urgency_score=urgency_score,
                novelty_score=novelty_score,
                relevance_score=relevance_score,
                source_reputation_score=features.source_reputation_score,
                sentiment=sentiment,
                market_bias=market_bias,
                impact_level=impact_level,
                urgency_level=urgency_level,
                relevance_level=relevance_level,
                time_horizon=time_horizon,
                categories=categories,
                alert_types=alert_types,
                summary=summary,
                explanation=explanation,
                trading_notes=trading_notes,
                rule_score_weight=rule_weight,
                llm_score_weight=llm_weight,
                llm_status=llm_result.status if llm_result else LLMOutputStatus.DISABLED,
                scored_at=utc_now(),
                metadata={
                    "rule_sentiment_score": rule_sentiment_score,
                    "rule_impact_score": rule_impact_score,
                    "rule_urgency_score": rule_urgency_score,
                    "rule_relevance_score": rule_relevance_score,
                    "rule_confidence_score": rule_confidence_score,
                    "llm_used": bool(
                        llm_result and llm_result.status == LLMOutputStatus.SUCCESS
                    ),
                    "primary_symbol": item.primary_symbol,
                    "primary_category": str(item.primary_category),
                    "matched_keywords": list(features.matched_keywords),
                    "matched_positive_keywords": list(features.matched_positive_keywords),
                    "matched_negative_keywords": list(features.matched_negative_keywords),
                },
            )

        except NewsScoringError:
            raise

        except Exception as exc:
            raise NewsScoringError(
                "Failed to score news item",
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

    def score_many(
        self,
        items: Iterable[NormalizedNewsItem],
        features_by_news_id: dict[str, NewsFeatures],
        llm_results_by_news_id: dict[str, NewsLLMResult] | None = None,
    ) -> dict[str, NewsScore]:
        """
        Score multiple news items.

        Returns:
            Mapping news_id -> NewsScore.
        """

        llm_results_by_news_id = llm_results_by_news_id or {}
        scores: dict[str, NewsScore] = {}

        for item in items:
            features = features_by_news_id.get(item.news_id)
            if features is None:
                raise NewsScoringError(
                    "Missing features for news item",
                    context=NewsErrorContext(
                        news_id=item.news_id,
                        source_name=item.source_name,
                        source_type=str(item.source_type),
                        url=item.url,
                    ),
                )

            scores[item.news_id] = self.score(
                item=item,
                features=features,
                llm_result=llm_results_by_news_id.get(item.news_id),
            )

        return scores

    def _rule_sentiment_score(
        self,
        item: NormalizedNewsItem,
        features: NewsFeatures,
    ) -> float:
        score = 0.0

        positive_count = len(features.matched_positive_keywords)
        negative_count = len(features.matched_negative_keywords)

        score += min(0.45, positive_count * 0.12)
        score -= min(0.55, negative_count * 0.14)

        categories = set(item.categories)

        if categories & _POSITIVE_CATEGORIES:
            score += 0.18

        if categories & _NEGATIVE_CATEGORIES:
            score -= 0.25

        if features.has_listing_keywords:
            score += 0.30

        if features.has_partnership_keywords:
            score += 0.12

        if features.has_airdrop_keywords:
            score += 0.10

        if features.has_delisting_keywords:
            score -= 0.40

        if features.has_hack_keywords or features.has_exploit_keywords:
            score -= 0.55

        if features.has_lawsuit_keywords:
            score -= 0.30

        if features.has_bankruptcy_keywords:
            score -= 0.60

        if features.has_token_unlock_keywords:
            score -= 0.18

        if features.has_rumor_keywords:
            score *= 0.55

        # Regulation може бути mixed: ETF approval bullish, lawsuit/ban bearish.
        if NewsCategory.ETF in categories and features.has_etf_keywords:
            if any(
                keyword in item.text.lower()
                for keyword in ("approved", "approval", "greenlight")
            ):
                score += 0.35
            elif any(
                keyword in item.text.lower()
                for keyword in ("delay", "delayed", "reject", "rejected")
            ):
                score -= 0.25

        return self._clamp_signed(score)

    def _rule_impact_score(
        self,
        item: NormalizedNewsItem,
        features: NewsFeatures,
    ) -> float:
        cfg = self.scoring_config
        categories = set(item.categories)

        score = 0.0

        score += features.source_reputation_score * cfg.source_reputation_weight

        keyword_signal = 0.0
        if features.has_urgent_keywords:
            keyword_signal += 0.20
        if features.has_regulatory_keywords:
            keyword_signal += 0.20
        if features.has_macro_keywords:
            keyword_signal += 0.18
        if features.has_hack_keywords:
            keyword_signal += 0.30
        if features.has_exploit_keywords:
            keyword_signal += 0.28
        if features.has_listing_keywords:
            keyword_signal += 0.20
        if features.has_delisting_keywords:
            keyword_signal += 0.28
        if features.has_etf_keywords:
            keyword_signal += 0.24
        if features.has_lawsuit_keywords:
            keyword_signal += 0.20
        if features.has_bankruptcy_keywords:
            keyword_signal += 0.35
        if features.has_token_unlock_keywords:
            keyword_signal += 0.16

        keyword_signal = self._clamp01(keyword_signal)
        score += keyword_signal * cfg.keyword_impact_weight

        urgency_signal = self._rule_urgency_score(item, features)
        score += urgency_signal * cfg.urgency_weight

        category_signal = 0.25
        if categories & _HIGH_IMPACT_CATEGORIES:
            category_signal = 0.75
        if NewsCategory.HACK in categories or NewsCategory.EXPLOIT in categories:
            category_signal = 0.90
        if NewsCategory.BANKRUPTCY in categories:
            category_signal = 0.95
        if NewsCategory.MACRO in categories or NewsCategory.ETF in categories:
            category_signal = max(category_signal, 0.82)

        score += category_signal * cfg.category_weight

        novelty_signal = self._novelty_score(item)
        score += novelty_signal * cfg.novelty_weight

        if features.symbol_count > 0:
            score += 0.08

        if features.is_official_source:
            score += 0.08

        if features.is_exchange_source and (
            features.has_listing_keywords
            or features.has_delisting_keywords
            or NewsCategory.EXCHANGE in categories
        ):
            score += 0.08

        if features.is_low_quality_source:
            score -= 0.20

        if features.has_rumor_keywords:
            score -= 0.12

        return self._clamp01(score)

    def _rule_urgency_score(
        self,
        item: NormalizedNewsItem,
        features: NewsFeatures,
    ) -> float:
        categories = set(item.categories)
        score = 0.0

        if features.is_breaking_news:
            score += 0.35

        if features.has_urgent_keywords:
            score += 0.25

        if categories & _URGENT_CATEGORIES:
            score += 0.25

        if features.has_hack_keywords or features.has_exploit_keywords:
            score += 0.35

        if features.has_delisting_keywords:
            score += 0.25

        if features.has_macro_keywords:
            score += 0.20

        if features.has_etf_keywords:
            score += 0.15

        age_seconds = item.age_seconds
        if age_seconds is None:
            score += 0.10
        elif age_seconds <= self.scoring_config.fresh_news_window_seconds:
            score += 0.25
        elif age_seconds <= self.scoring_config.stale_news_after_seconds:
            score += 0.10
        else:
            score -= 0.25

        return self._clamp01(score)

    def _rule_relevance_score(
        self,
        item: NormalizedNewsItem,
        features: NewsFeatures,
    ) -> float:
        score = self.scoring_config.default_relevance_score

        if item.symbols:
            tracked_symbols = set(self.config.normalized_tracked_symbols)
            matched_tracked = set(item.symbols) & tracked_symbols

            if matched_tracked:
                score += 0.30
            else:
                score += 0.10

        if features.entity_count > 0:
            score += min(0.15, features.entity_count * 0.03)

        if features.category_count > 0:
            score += min(0.12, features.category_count * 0.03)

        if item.primary_category in _HIGH_IMPACT_CATEGORIES:
            score += 0.12

        if features.is_low_quality_source:
            score -= 0.25

        if features.text_length_score < 0.35:
            score -= 0.12

        return self._clamp01(score)

    def _rule_confidence_score(
        self,
        item: NormalizedNewsItem,
        features: NewsFeatures,
    ) -> float:
        score = self.scoring_config.default_confidence_score

        score += features.source_reputation_score * 0.25
        score += features.text_length_score * 0.15
        score += features.title_strength_score * 0.10

        if item.url or item.canonical_url:
            score += 0.05

        if item.published_at:
            score += 0.06

        if features.is_official_source:
            score += 0.10

        if features.has_rumor_keywords:
            score -= 0.25

        if features.is_low_quality_source:
            score -= 0.25

        if not item.symbols and item.primary_category == NewsCategory.GENERAL:
            score -= 0.10

        return self._clamp01(score)

    def _novelty_score(self, item: NormalizedNewsItem) -> float:
        """
        Estimate novelty/freshness.

        True cross-source novelty is handled by NewsDeduplicator. Here we only
        estimate time freshness.
        """

        age_seconds = item.age_seconds

        if age_seconds is None:
            return self.scoring_config.default_novelty_score

        if age_seconds <= 300:
            return 1.0

        if age_seconds <= self.scoring_config.fresh_news_window_seconds:
            return 0.85

        if age_seconds <= 7_200:
            return 0.65

        if age_seconds <= self.scoring_config.stale_news_after_seconds:
            return 0.40

        return 0.15

    def _effective_llm_weight(self, llm_result: NewsLLMResult | None) -> float:
        if llm_result is None:
            return 0.0

        if llm_result.status != LLMOutputStatus.SUCCESS:
            return 0.0

        if not self.config.llm.enabled:
            return 0.0

        return self._clamp01(self.scoring_config.llm_score_weight)

    def _blend01(
        self,
        *,
        rule_value: float,
        llm_value: float | None,
        rule_weight: float,
        llm_weight: float,
    ) -> float:
        rule_value = self._clamp01(rule_value)

        if llm_value is None or llm_weight <= 0:
            return rule_value

        llm_value = self._clamp01(llm_value)
        return self._clamp01((rule_value * rule_weight) + (llm_value * llm_weight))

    def _blend_signed(
        self,
        *,
        rule_value: float,
        llm_value: float | None,
        rule_weight: float,
        llm_weight: float,
    ) -> float:
        rule_value = self._clamp_signed(rule_value)

        if llm_value is None or llm_weight <= 0:
            return rule_value

        llm_value = self._clamp_signed(llm_value)
        return self._clamp_signed((rule_value * rule_weight) + (llm_value * llm_weight))

    def _sentiment_label(
        self,
        sentiment_score: float,
        llm_result: NewsLLMResult | None,
    ) -> NewsSentiment:
        if (
            llm_result is not None
            and llm_result.status == LLMOutputStatus.SUCCESS
            and llm_result.sentiment != NewsSentiment.UNKNOWN
            and abs(sentiment_score) < 0.20
        ):
            return llm_result.sentiment

        if sentiment_score <= -0.75:
            return NewsSentiment.VERY_BEARISH
        if sentiment_score <= -0.35:
            return NewsSentiment.BEARISH
        if sentiment_score <= -0.12:
            return NewsSentiment.SLIGHTLY_BEARISH
        if sentiment_score < 0.12:
            return NewsSentiment.NEUTRAL
        if sentiment_score < 0.35:
            return NewsSentiment.SLIGHTLY_BULLISH
        if sentiment_score < 0.75:
            return NewsSentiment.BULLISH
        return NewsSentiment.VERY_BULLISH

    def _market_bias(
        self,
        *,
        sentiment_score: float,
        item: NormalizedNewsItem,
        features: NewsFeatures,
        llm_result: NewsLLMResult | None,
    ) -> NewsMarketBias:
        if (
            llm_result is not None
            and llm_result.status == LLMOutputStatus.SUCCESS
            and llm_result.market_bias != NewsMarketBias.UNKNOWN
        ):
            return llm_result.market_bias

        if features.has_macro_keywords:
            if sentiment_score <= -0.15:
                return NewsMarketBias.RISK_OFF
            if sentiment_score >= 0.15:
                return NewsMarketBias.RISK_ON

        if features.has_hack_keywords or features.has_exploit_keywords:
            return NewsMarketBias.BEARISH

        if features.has_bankruptcy_keywords or features.has_delisting_keywords:
            return NewsMarketBias.BEARISH

        if features.has_listing_keywords:
            return NewsMarketBias.BULLISH

        if features.has_rumor_keywords and abs(sentiment_score) < 0.25:
            return NewsMarketBias.MIXED

        if sentiment_score <= -0.18:
            return NewsMarketBias.BEARISH
        if sentiment_score >= 0.18:
            return NewsMarketBias.BULLISH

        if item.primary_category in {NewsCategory.REGULATION, NewsCategory.ETF}:
            return NewsMarketBias.MIXED

        return NewsMarketBias.NEUTRAL

    def _impact_level(self, impact_score: float) -> NewsImpactLevel:
        if impact_score >= self.scoring_config.critical_impact_threshold:
            return NewsImpactLevel.CRITICAL

        if impact_score >= self.scoring_config.high_impact_threshold:
            return NewsImpactLevel.HIGH

        if impact_score >= 0.45:
            return NewsImpactLevel.MEDIUM

        if impact_score >= 0.15:
            return NewsImpactLevel.LOW

        return NewsImpactLevel.NONE

    def _urgency_level(self, urgency_score: float) -> NewsUrgencyLevel:
        if urgency_score >= 0.90:
            return NewsUrgencyLevel.CRITICAL

        if urgency_score >= self.scoring_config.high_urgency_threshold:
            return NewsUrgencyLevel.HIGH

        if urgency_score >= 0.45:
            return NewsUrgencyLevel.MEDIUM

        if urgency_score >= 0.15:
            return NewsUrgencyLevel.LOW

        return NewsUrgencyLevel.NONE

    def _relevance_level(self, relevance_score: float) -> NewsRelevanceLevel:
        if relevance_score >= 0.85:
            return NewsRelevanceLevel.VERY_HIGH

        if relevance_score >= 0.65:
            return NewsRelevanceLevel.HIGH

        if relevance_score >= 0.40:
            return NewsRelevanceLevel.MEDIUM

        if relevance_score >= 0.15:
            return NewsRelevanceLevel.LOW

        return NewsRelevanceLevel.IRRELEVANT

    def _time_horizon(
        self,
        *,
        item: NormalizedNewsItem,
        features: NewsFeatures,
        urgency_score: float,
        llm_result: NewsLLMResult | None,
    ) -> NewsTimeHorizon:
        if (
            llm_result is not None
            and llm_result.status == LLMOutputStatus.SUCCESS
            and llm_result.time_horizon != NewsTimeHorizon.UNKNOWN
        ):
            return llm_result.time_horizon

        categories = set(item.categories)

        if urgency_score >= 0.85:
            return NewsTimeHorizon.IMMEDIATE

        if (
            features.has_hack_keywords
            or features.has_exploit_keywords
            or features.has_listing_keywords
            or features.has_delisting_keywords
        ):
            return NewsTimeHorizon.SCALP

        if categories & {NewsCategory.EXCHANGE, NewsCategory.ETF, NewsCategory.REGULATION}:
            return NewsTimeHorizon.INTRADAY

        if categories & {NewsCategory.MACRO, NewsCategory.TOKEN_UNLOCK}:
            return NewsTimeHorizon.SWING

        if NewsCategory.MACRO in categories:
            return NewsTimeHorizon.MACRO

        return NewsTimeHorizon.INTRADAY

    def _alert_types(
        self,
        *,
        item: NormalizedNewsItem,
        features: NewsFeatures,
        impact_score: float,
        urgency_score: float,
        impact_level: NewsImpactLevel,
    ) -> tuple[NewsAlertType, ...]:
        alerts: list[NewsAlertType] = []

        if impact_level in {NewsImpactLevel.HIGH, NewsImpactLevel.CRITICAL}:
            alerts.append(NewsAlertType.HIGH_IMPACT)

        if features.is_breaking_news or urgency_score >= 0.80:
            alerts.append(NewsAlertType.BREAKING_NEWS)

        if features.has_regulatory_keywords or NewsCategory.REGULATION in item.categories:
            alerts.append(NewsAlertType.REGULATORY_RISK)

        if (
            features.has_hack_keywords
            or features.has_exploit_keywords
            or NewsCategory.SECURITY in item.categories
        ):
            alerts.append(NewsAlertType.SECURITY_INCIDENT)

        if NewsCategory.EXCHANGE in item.categories and impact_score >= 0.45:
            alerts.append(NewsAlertType.EXCHANGE_INCIDENT)

        if features.has_listing_keywords:
            alerts.append(NewsAlertType.LISTING_EVENT)

        if features.has_delisting_keywords:
            alerts.append(NewsAlertType.DELISTING_EVENT)

        if features.has_macro_keywords:
            alerts.append(NewsAlertType.MACRO_EVENT)

        if features.has_rumor_keywords:
            alerts.append(NewsAlertType.RUMOR)

        if not alerts:
            alerts.append(NewsAlertType.GENERAL)

        return tuple(self._deduplicate_preserve_order(alerts))

    def _merge_categories(
        self,
        item: NormalizedNewsItem,
        llm_result: NewsLLMResult | None,
    ) -> tuple[NewsCategory, ...]:
        categories: list[NewsCategory] = list(item.categories)

        if llm_result and llm_result.status == LLMOutputStatus.SUCCESS:
            categories.extend(llm_result.categories)

        categories = [
            category for category in categories if category != NewsCategory.UNKNOWN
        ]

        if not categories:
            return (NewsCategory.GENERAL,)

        return tuple(self._deduplicate_preserve_order(categories))

    def _summary(
        self,
        *,
        item: NormalizedNewsItem,
        llm_result: NewsLLMResult | None,
    ) -> str:
        if (
            llm_result is not None
            and llm_result.status == LLMOutputStatus.SUCCESS
            and llm_result.summary
        ):
            return llm_result.summary

        if item.summary:
            return item.summary[:500]

        return item.title[:500]

    def _explanation(
        self,
        *,
        item: NormalizedNewsItem,
        features: NewsFeatures,
        llm_result: NewsLLMResult | None,
        sentiment_score: float,
        impact_score: float,
        urgency_score: float,
        relevance_score: float,
        confidence_score: float,
    ) -> str:
        if (
            llm_result is not None
            and llm_result.status == LLMOutputStatus.SUCCESS
            and llm_result.explanation
        ):
            return llm_result.explanation

        reasons: list[str] = []

        if features.is_official_source:
            reasons.append("official/high-reputation source")

        if item.symbols:
            reasons.append(f"mentions symbols: {', '.join(item.symbols[:6])}")

        if features.matched_keywords:
            reasons.append(
                "matched keywords: "
                + ", ".join(features.matched_keywords[:10])
            )

        if item.categories:
            reasons.append(
                "categories: "
                + ", ".join(str(category) for category in item.categories[:6])
            )

        if features.has_rumor_keywords:
            reasons.append("rumor/unconfirmed wording reduces confidence")

        if features.is_low_quality_source:
            reasons.append("low-quality wording/source markers reduce confidence")

        score_summary = (
            f"scores sentiment={sentiment_score:.2f}, "
            f"impact={impact_score:.2f}, urgency={urgency_score:.2f}, "
            f"relevance={relevance_score:.2f}, confidence={confidence_score:.2f}"
        )

        if reasons:
            return f"Rule-based analysis: {'; '.join(reasons)}; {score_summary}."

        return f"Rule-based analysis: {score_summary}."

    def _trading_notes(
        self,
        *,
        item: NormalizedNewsItem,
        market_bias: NewsMarketBias,
        impact_level: NewsImpactLevel,
        urgency_level: NewsUrgencyLevel,
        time_horizon: NewsTimeHorizon,
        llm_result: NewsLLMResult | None,
    ) -> str:
        if (
            llm_result is not None
            and llm_result.status == LLMOutputStatus.SUCCESS
            and llm_result.trading_notes
        ):
            return llm_result.trading_notes

        symbol_part = (
            f"Watch {', '.join(item.symbols[:4])}."
            if item.symbols
            else "No directly tracked symbol detected."
        )

        return (
            f"Manual review only. Bias={market_bias}, "
            f"impact={impact_level}, urgency={urgency_level}, "
            f"horizon={time_horizon}. {symbol_part} "
            "Confirm with price action, order flow, liquidity, and market reaction "
            "before making any trading decision."
        )

    def _deduplicate_preserve_order(self, items: Iterable[Any]) -> list[Any]:
        seen: set[Any] = set()
        result: list[Any] = []

        for item in items:
            if item in seen:
                continue

            seen.add(item)
            result.append(item)

        return result

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _clamp_signed(self, value: float) -> float:
        return max(-1.0, min(1.0, value))


__all__ = [
    "NewsScorer",
]