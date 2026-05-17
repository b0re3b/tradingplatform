# tests/ai/news/test_news_processing_pipeline.py

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from ai import (
    LLMOutputStatus,
    LLMProvider,
    NewsCategory,
    NewsDeduplicationConfig,
    NewsDeduplicator,
    NewsFeatureExtractor,
    NewsImpactLevel,
    NewsLanguage,
    NewsLLMResult,
    NewsMarketBias,
    NewsProcessingError,
    NewsRelevanceLevel,
    NewsScorer,
    NewsScoringError,
    NewsSentiment,
    NewsSourceType,
    NewsTimeHorizon,
    NewsUrgencyLevel,
    NewsProcessor,
    RawNewsItem,
    utc_now,
)


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def processor(news_config):
    return NewsProcessor(news_config)


@pytest.fixture
def deduplicator(news_config):
    return NewsDeduplicator(news_config.deduplication)


@pytest.fixture
def feature_extractor(news_config):
    return NewsFeatureExtractor(news_config)


@pytest.fixture
def scorer(news_config):
    return NewsScorer(news_config)


# ---------------------------------------------------------------------------
# NewsProcessor: hostile normalization, canonicalization, ids, categories
# ---------------------------------------------------------------------------


def test_processor_aggressively_cleans_html_entities_tracking_params_and_noise(
    processor,
    raw_regulation_news,
):
    item = processor.process(raw_regulation_news)

    assert item.title == "SEC sues Binance over alleged securities violations"
    assert "<" not in item.text
    assert ">" not in item.text
    assert "\u200b" not in item.text
    assert "\xa0" not in item.text

    assert item.canonical_url == "https://example.com/sec-binance?id=123"
    assert "utm_source" not in item.canonical_url
    assert "fbclid" not in item.canonical_url

    assert item.title_hash
    assert item.content_hash
    assert item.news_id
    assert item.news_id.startswith("news_")

    assert item.source_name == raw_regulation_news.source_name
    assert item.source_type == raw_regulation_news.source_type
    assert item.source_item_id == raw_regulation_news.source_item_id
    assert item.language == NewsLanguage.EN

    assert NewsCategory.REGULATION in item.categories
    assert NewsCategory.LEGAL in item.categories

    entity_names = {entity.name for entity in item.entities}
    assert "SEC" in entity_names
    assert "Binance" in entity_names

    assert "BTC" in item.symbols
    assert "ETH" in item.symbols
    assert "SEC" not in item.symbols
    assert "ETF" not in item.symbols
    assert "USD" not in item.symbols

    assert item.source_reputation_score >= 0.9
    assert item.metadata["raw_language"] == str(raw_regulation_news.language)
    assert item.metadata["has_body"] is True
    assert item.metadata["has_summary"] is True


def test_processor_generates_same_id_for_tracking_url_variants_when_source_item_id_same(
    processor,
    make_raw_news_item,
):
    first = make_raw_news_item(
        source_name="test_source",
        title="Ethereum protocol hacked for $120M",
        url="https://www.example.com/news/eth-hack/?utm_source=x&fbclid=abc&id=42",
        summary="Exploit drained protocol funds.",
        source_item_id="same-provider-id",
    )
    second = make_raw_news_item(
        source_name="test_source",
        title="Ethereum protocol hacked for $120M",
        url="https://example.com/news/eth-hack?id=42&utm_campaign=y&gclid=z",
        summary="Exploit drained protocol funds.",
        source_item_id="same-provider-id",
    )

    first_item = processor.process(first)
    second_item = processor.process(second)

    assert first_item.canonical_url == second_item.canonical_url
    assert first_item.title_hash == second_item.title_hash
    assert first_item.content_hash == second_item.content_hash
    assert first_item.news_id == second_item.news_id


def test_processor_process_many_skips_invalid_items_instead_of_poisoning_batch(
    processor,
    raw_hack_news,
    make_raw_news_item,
):
    invalid_after_cleaning = make_raw_news_item(
        title="<p></p><script></script>",
        summary="",
        body="",
        url="https://example.com/empty",
        source_item_id="invalid-empty-after-clean",
    )

    batch = processor.process_many([raw_hack_news, invalid_after_cleaning])

    assert batch.processed_count == 1
    assert batch.failed_count == 1
    assert batch.total_count == 2
    assert len(batch.errors) == 1
    assert "empty after normalization" in batch.errors[0].lower()
    assert batch.items[0].source_item_id == raw_hack_news.source_item_id


def test_processor_filters_common_false_positive_symbols_and_non_tracked_symbols(
    processor,
    make_raw_news_item,
):
    raw = make_raw_news_item(
        title=(
            "SEC FOMC ETF TVL US EU CEO says BTC ETH SOL XRP DOGE NOTINUNIVERSE "
            "react after Ethereum and Bitcoin update"
        ),
        summary=(
            "BTC, ETH and SOL are in the configured trading universe. "
            "XRP and DOGE should be ignored when not tracked."
        ),
        source_item_id="symbol-filter-case",
    )

    item = processor.process(raw)

    assert "BTC" in item.symbols
    assert "ETH" in item.symbols
    assert "SOL" in item.symbols

    assert "SEC" not in item.symbols
    assert "FOMC" not in item.symbols
    assert "ETF" not in item.symbols
    assert "TVL" not in item.symbols
    assert "US" not in item.symbols
    assert "EU" not in item.symbols
    assert "CEO" not in item.symbols

    assert "XRP" not in item.symbols
    assert "DOGE" not in item.symbols
    assert "NOTINUNIVERSE" not in item.symbols


def test_processor_limits_symbol_extraction_to_configured_maximum(
    processor,
    news_config,
    make_raw_news_item,
):
    raw = make_raw_news_item(
        title="BTC ETH SOL BNB BTC ETH SOL BNB BTC ETH SOL BNB",
        summary="BTC ETH SOL BNB repeated aggressively to test dedup and cap.",
        source_item_id="symbol-limit-case",
    )

    item = processor.process(raw)

    assert len(item.symbols) <= news_config.features.max_symbols_per_item
    assert item.symbols == tuple(dict.fromkeys(item.symbols))


# ---------------------------------------------------------------------------
# NewsDeduplicator: duplicate suppression, cycle-level protection, TTL, disabled
# ---------------------------------------------------------------------------


def test_deduplicator_filter_new_raw_suppresses_tracking_url_duplicates(
    deduplicator,
    make_raw_news_item,
):
    first = make_raw_news_item(
        title="Binance will list TEST token",
        url="https://example.com/listing?id=1&utm_source=x&fbclid=y",
        source_item_id=None,
    )
    duplicate = make_raw_news_item(
        title="Binance will list TEST token",
        url="https://www.example.com/listing?id=1&utm_campaign=z&gclid=abc",
        source_item_id=None,
    )

    unique = deduplicator.filter_new_raw([first, duplicate])

    assert unique == [first]

    stats = deduplicator.stats()
    assert stats["checked_count"] == 2
    assert stats["unique_count"] == 1
    assert stats["duplicate_count"] == 1


def test_deduplicator_filter_new_normalized_suppresses_same_cycle_title_hash_duplicates(
    deduplicator,
    make_normalized_news_item,
):
    first = make_normalized_news_item(
        news_id="news_a",
        title="Breaking: Ethereum protocol hacked for $120M",
        text="Exploit drained funds from protocol.",
        title_hash="same-title-hash",
        content_hash="same-content-hash",
    )
    duplicate = make_normalized_news_item(
        news_id="news_b",
        title="BREAKING: Ethereum protocol hacked for $120M",
        text="Exploit drained funds from protocol.",
        title_hash="same-title-hash",
        content_hash="same-content-hash",
    )

    unique = deduplicator.filter_new_normalized([first, duplicate])

    assert unique == [first]

    duplicate_decision = deduplicator.check_normalized(duplicate)
    assert duplicate_decision.is_duplicate is True
    assert duplicate_decision.existing_news_id is not None


def test_deduplicator_disabled_always_returns_unique_even_for_identical_items(
    make_raw_news_item,
):
    deduplicator = NewsDeduplicator(
        NewsDeduplicationConfig(enabled=False)
    )

    first = make_raw_news_item(
        title="Same title",
        url="https://example.com/same",
        source_item_id="same-id",
    )
    second = make_raw_news_item(
        title="Same title",
        url="https://example.com/same",
        source_item_id="same-id",
    )

    unique = deduplicator.filter_new_raw([first, second])

    assert unique == [first, second]
    assert deduplicator.check_raw(first).is_duplicate is False
    assert deduplicator.stats()["enabled"] is False


def test_deduplicator_evicts_old_records_when_capacity_is_exceeded(
    news_config,
    make_raw_news_item,
):
    config = replace(
        news_config.deduplication,
        max_seen_items=2,
        ttl_seconds=86_400,
    )
    deduplicator = NewsDeduplicator(config)

    items = [
        make_raw_news_item(
            title=f"Unique news item {index}",
            url=f"https://example.com/{index}",
            source_item_id=f"id-{index}",
        )
        for index in range(5)
    ]

    unique = deduplicator.filter_new_raw(items)

    assert len(unique) == 5

    stats = deduplicator.stats()
    assert stats["evicted_count"] > 0
    assert stats["seen_by_source_item_id"] <= config.max_seen_items
    assert stats["seen_by_url"] <= config.max_seen_items


def test_deduplicator_ttl_expiry_allows_old_duplicate_to_be_seen_as_unique_again(
    news_config,
    make_raw_news_item,
):
    config = replace(
        news_config.deduplication,
        ttl_seconds=1,
        max_seen_items=100,
    )
    deduplicator = NewsDeduplicator(config)

    raw = make_raw_news_item(
        title="Ethereum protocol hacked",
        url="https://example.com/ttl-case",
        source_item_id="ttl-case",
    )

    assert deduplicator.filter_new_raw([raw]) == [raw]
    assert deduplicator.check_raw(raw).is_duplicate is True

    for storage_name in (
        "_seen_by_url",
        "_seen_by_canonical_url",
        "_seen_by_source_item_id",
        "_seen_by_title_hash",
        "_seen_by_content_hash",
        "_recent_titles",
    ):
        storage = getattr(deduplicator, storage_name)
        for key, record in list(storage.items()):
            storage[key] = replace(
                record,
                created_at=utc_now() - timedelta(seconds=config.ttl_seconds + 5),
            )

    assert deduplicator.check_raw(raw).is_duplicate is False


def test_deduplicator_near_duplicate_detection_catches_rewritten_headlines(
    news_config,
    make_raw_news_item,
):
    config = replace(
        news_config.deduplication,
        enable_near_duplicate_detection=True,
        title_similarity_threshold=0.82,
        content_similarity_threshold=0.82,
    )
    deduplicator = NewsDeduplicator(config)

    first = make_raw_news_item(
        title="Breaking: Ethereum DeFi protocol hacked for $120M",
        summary="Exploit drained funds from a major protocol.",
        url="https://example.com/original",
        source_item_id="near-1",
    )
    rewritten = make_raw_news_item(
        title="Ethereum DeFi protocol hacked, $120 million drained",
        summary="A major exploit drained funds from the protocol.",
        url="https://mirror.example.com/rewrite",
        source_item_id="near-2",
    )

    unique = deduplicator.filter_new_raw([first, rewritten])

    assert unique == [first]


# ---------------------------------------------------------------------------
# NewsFeatureExtractor: hostile keyword, source quality, counts
# ---------------------------------------------------------------------------


def test_feature_extractor_detects_hack_exploit_urgency_numbers_and_negative_bias(
    feature_extractor,
    normalized_hack_news,
):
    features = feature_extractor.extract(normalized_hack_news)

    assert features.news_id == normalized_hack_news.news_id
    assert features.has_urgent_keywords is True
    assert features.has_hack_keywords is True
    assert features.has_exploit_keywords is True
    assert features.is_breaking_news is True

    assert "hack" in features.matched_keywords
    assert "exploit" in features.matched_keywords
    assert features.matched_negative_keywords

    assert features.symbol_count == len(normalized_hack_news.symbols)
    assert features.entity_count == len(normalized_hack_news.entities)
    assert features.category_count == len(normalized_hack_news.categories)

    assert features.raw_feature_values["number_count"] >= 1
    assert features.raw_feature_values["matched_hack_keyword_count"] >= 1
    assert features.raw_feature_values["matched_high_impact_keyword_count"] >= 1

    assert 0.0 <= features.source_reputation_score <= 1.0
    assert 0.0 <= features.title_strength_score <= 1.0
    assert 0.0 <= features.text_length_score <= 1.0


def test_feature_extractor_boosts_official_exchange_source_and_listing_features(
    feature_extractor,
    processor,
    raw_listing_news,
):
    item = processor.process(raw_listing_news)
    features = feature_extractor.extract(item)

    assert NewsCategory.LISTING in item.categories
    assert NewsCategory.EXCHANGE in item.categories

    assert features.is_official_source is True
    assert features.is_exchange_source is True
    assert features.has_listing_keywords is True
    assert features.matched_positive_keywords
    assert features.source_reputation_score >= item.source_reputation_score

    assert "listing" in features.matched_keywords or "will list" in features.matched_keywords
    assert features.raw_feature_values["matched_listing_keyword_count"] >= 1


def test_feature_extractor_penalizes_clickbait_low_quality_language(
    feature_extractor,
    make_normalized_news_item,
):
    item = make_normalized_news_item(
        news_id="news_clickbait_001",
        source_name="unknown_clickbait_source",
        title="SHOCKING secret coin could explode next 100x, you won't believe it",
        text=(
            "Price prediction says this hidden gem will moon. "
            "No official source, no verifiable data, just hype."
        ),
        categories=(NewsCategory.GENERAL,),
        symbols=(),
        source_reputation_score=0.5,
    )

    features = feature_extractor.extract(item)

    assert features.is_low_quality_source is True
    assert features.source_reputation_score < item.source_reputation_score
    assert features.raw_feature_values["matched_low_quality_keyword_count"] >= 2
    assert "price prediction" in features.matched_keywords
    assert "next 100x" in features.matched_keywords


def test_feature_extractor_does_not_crash_on_minimal_unknown_source_item(
    feature_extractor,
    make_normalized_news_item,
):
    item = make_normalized_news_item(
        news_id="news_minimal_unknown",
        source_name="not_configured_source",
        title="Market update",
        text="General crypto market update with no specific symbol.",
        url=None,
        canonical_url=None,
        categories=(NewsCategory.UNKNOWN,),
        symbols=(),
        entities=(),
        source_reputation_score=0.5,
    )

    features = feature_extractor.extract(item)

    assert features.news_id == item.news_id
    assert features.source_reputation_score == pytest.approx(0.5)
    assert features.symbol_count == 0
    assert features.entity_count == 0
    assert features.category_count == 1
    assert features.matched_keywords == ()


# ---------------------------------------------------------------------------
# NewsScorer: hard logic checks, score bounds, LLM blending, failures
# ---------------------------------------------------------------------------


def test_scorer_marks_major_hack_as_bearish_high_impact_actionable(
    scorer,
    normalized_hack_news,
    feature_extractor,
):
    features = feature_extractor.extract(normalized_hack_news)
    score = scorer.score(normalized_hack_news, features)

    assert score.news_id == normalized_hack_news.news_id
    assert score.sentiment_score < 0
    assert score.market_bias in {
        NewsMarketBias.BEARISH,
        NewsMarketBias.STRONGLY_BEARISH,
        NewsMarketBias.RISK_OFF,
    }

    assert score.impact_score >= 0.60
    assert score.urgency_score >= 0.60
    assert score.relevance_score >= 0.50
    assert score.confidence_score >= 0.45

    assert score.impact_level in {
        NewsImpactLevel.MEDIUM,
        NewsImpactLevel.HIGH,
        NewsImpactLevel.CRITICAL,
    }
    assert score.urgency_level in {
        NewsUrgencyLevel.HIGH,
        NewsUrgencyLevel.URGENT,
        NewsUrgencyLevel.CRITICAL,
    }
    assert score.relevance_level in {
        NewsRelevanceLevel.MEDIUM,
        NewsRelevanceLevel.HIGH,
    }

    assert NewsCategory.HACK in score.categories
    assert score.summary
    assert score.explanation
    assert score.trading_notes
    assert score.is_actionable_for_manual_review is True

    assert score.metadata["llm_used"] is False
    assert score.metadata["primary_symbol"] == normalized_hack_news.primary_symbol
    assert "rule_impact_score" in score.metadata


def test_scorer_marks_listing_as_positive_without_turning_it_into_security_alert(
    scorer,
    processor,
    feature_extractor,
    raw_listing_news,
):
    item = processor.process(raw_listing_news)
    features = feature_extractor.extract(item)
    score = scorer.score(item, features)

    assert NewsCategory.LISTING in score.categories
    assert score.sentiment_score > 0
    assert score.market_bias in {
        NewsMarketBias.BULLISH,
        NewsMarketBias.SLIGHTLY_BULLISH,
        NewsMarketBias.RISK_ON,
    }

    assert score.impact_score >= 0.45
    assert score.relevance_score >= 0.45
    assert score.is_actionable_for_manual_review is True

    assert NewsCategory.HACK not in score.categories
    assert NewsCategory.EXPLOIT not in score.categories
    assert "hack" not in score.metadata["matched_negative_keywords"]


def test_scorer_does_not_allow_low_relevance_clickbait_to_become_high_impact(
    scorer,
    feature_extractor,
    make_normalized_news_item,
):
    item = make_normalized_news_item(
        news_id="news_clickbait_low_relevance",
        source_name="unknown_clickbait_source",
        title="SHOCKING secret coin could explode next 100x",
        text="Price prediction with no official confirmation and no tracked symbols.",
        categories=(NewsCategory.GENERAL, NewsCategory.RUMOR),
        symbols=(),
        entities=(),
        source_reputation_score=0.3,
    )

    features = feature_extractor.extract(item)
    score = scorer.score(item, features)

    assert features.is_low_quality_source is True
    assert score.relevance_score < 0.50
    assert score.is_actionable_for_manual_review is False
    assert score.impact_level not in {
        NewsImpactLevel.HIGH,
        NewsImpactLevel.CRITICAL,
    }


def test_scorer_blends_successful_llm_result_but_keeps_scores_bounded(
    scorer,
    normalized_hack_news,
    feature_extractor,
):
    features = feature_extractor.extract(normalized_hack_news)

    llm_result = NewsLLMResult(
        status=LLMOutputStatus.SUCCESS,
        provider=LLMProvider.LOCAL,
        model="test-local-model",
        sentiment_score=-1.0,
        impact_score=1.0,
        urgency_score=1.0,
        relevance_score=1.0,
        confidence_score=1.0,
        sentiment=NewsSentiment.VERY_BEARISH,
        market_bias=NewsMarketBias.STRONGLY_BEARISH,
        time_horizon=NewsTimeHorizon.IMMEDIATE,
        categories=(NewsCategory.HACK, NewsCategory.EXPLOIT, NewsCategory.SECURITY),
        summary="LLM summary should be accepted.",
        explanation="LLM explanation should be accepted.",
        trading_notes="LLM trading notes should be accepted.",
    )

    score = scorer.score(normalized_hack_news, features, llm_result=llm_result)

    assert score.llm_status == LLMOutputStatus.SUCCESS
    assert score.metadata["llm_used"] is True
    assert score.llm_score_weight > 0
    assert score.rule_score_weight < 1

    assert -1.0 <= score.sentiment_score <= 1.0
    assert 0.0 <= score.impact_score <= 1.0
    assert 0.0 <= score.urgency_score <= 1.0
    assert 0.0 <= score.relevance_score <= 1.0
    assert 0.0 <= score.confidence_score <= 1.0

    assert score.summary == llm_result.summary
    assert score.explanation
    assert score.trading_notes


def test_scorer_ignores_failed_llm_for_weighting_and_uses_rule_based_result(
    scorer,
    normalized_hack_news,
    feature_extractor,
):
    features = feature_extractor.extract(normalized_hack_news)

    failed_llm_result = NewsLLMResult(
        status=LLMOutputStatus.FALLBACK_USED,
        provider=LLMProvider.LOCAL,
        model="test-local-model",
        sentiment_score=1.0,
        impact_score=0.0,
        urgency_score=0.0,
        relevance_score=0.0,
        confidence_score=0.0,
        error="Provider failed; rule-based fallback should be used.",
    )

    score = scorer.score(
        normalized_hack_news,
        features,
        llm_result=failed_llm_result,
    )

    assert score.llm_status == LLMOutputStatus.FALLBACK_USED
    assert score.metadata["llm_used"] is False
    assert score.llm_score_weight == 0.0
    assert score.rule_score_weight == 1.0

    assert score.sentiment_score < 0
    assert score.impact_score > 0
    assert score.urgency_score > 0


def test_scorer_score_many_requires_features_for_every_item(
    scorer,
    normalized_hack_news,
    normalized_regulation_news,
    hack_features,
):
    with pytest.raises(NewsScoringError) as exc_info:
        scorer.score_many(
            items=[normalized_hack_news, normalized_regulation_news],
            features_by_news_id={
                normalized_hack_news.news_id: hack_features,
            },
        )

    assert "Missing features" in str(exc_info.value)


def test_full_processor_feature_scorer_pipeline_handles_regulation_case(
    processor,
    feature_extractor,
    scorer,
    raw_regulation_news,
):
    item = processor.process(raw_regulation_news)
    features = feature_extractor.extract(item)
    score = scorer.score(item, features)

    assert item.news_id == features.news_id == score.news_id
    assert NewsCategory.REGULATION in item.categories
    assert NewsCategory.REGULATION in score.categories

    assert features.has_regulatory_keywords is True
    assert features.has_lawsuit_keywords is True
    assert features.is_official_source is True

    assert score.sentiment_score < 0
    assert score.market_bias in {
        NewsMarketBias.BEARISH,
        NewsMarketBias.STRONGLY_BEARISH,
        NewsMarketBias.RISK_OFF,
    }
    assert score.impact_score >= 0.45
    assert score.relevance_score >= 0.50
    assert score.confidence_score >= 0.45
    assert score.is_actionable_for_manual_review is True


def test_full_pipeline_duplicate_protection_before_expensive_feature_and_scoring(
    processor,
    deduplicator,
    feature_extractor,
    scorer,
    make_raw_news_item,
):
    first = make_raw_news_item(
        title="Breaking: Ethereum DeFi protocol hacked for $120M",
        summary="Exploit drained funds from a major protocol.",
        url="https://example.com/hack?id=9&utm_source=x",
        source_item_id="same-raw-id",
    )
    duplicate = make_raw_news_item(
        title="Breaking: Ethereum DeFi protocol hacked for $120M",
        summary="Exploit drained funds from a major protocol.",
        url="https://www.example.com/hack?id=9&fbclid=y",
        source_item_id="same-raw-id",
    )

    raw_unique = deduplicator.filter_new_raw([first, duplicate])
    assert raw_unique == [first]

    processed = processor.process_many(raw_unique)
    normalized_unique = deduplicator.filter_new_normalized(list(processed.items))

    assert processed.processed_count == 1
    assert processed.failed_count == 0
    assert len(normalized_unique) == 1

    item = normalized_unique[0]
    features = feature_extractor.extract(item)
    score = scorer.score(item, features)

    assert item.news_id == score.news_id
    assert score.sentiment_score < 0
    assert score.impact_score >= 0.50


def test_processor_raises_news_processing_error_for_text_that_becomes_empty(
    processor,
    make_raw_news_item,
):
    raw = make_raw_news_item(
        title="<div></div>",
        summary="<span></span>",
        body="<script></script>",
        url="https://example.com/empty-after-cleaning",
        source_item_id="empty-after-cleaning",
    )

    with pytest.raises(NewsProcessingError) as exc_info:
        processor.process(raw)

    assert "empty after normalization" in str(exc_info.value).lower()