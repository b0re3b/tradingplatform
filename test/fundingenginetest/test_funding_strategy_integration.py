# tests/strategy/funding/test_funding_strategy_integration.py

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from strategy.config import (
    StrategyConfig,
    StrategyDefinitionConfig,
    StrategyRuntimeConfig,
)
from strategy.enums import (
    FeatureSource,
    SignalSide,
    StrategyCategory,
    Timeframe,
)
from strategy.models import StrategyContext
from strategy.processor import SignalNormalizer, SignalRouter
from strategy.registry import StrategyRegistry

from strategy.strategies.funding.funding_divergence_strategy import (
    FundingDivergenceStrategy,
    FundingDivergenceStrategyConfig,
)
from strategy.strategies.funding.funding_extreme_reversal_strategy import (
    FundingExtremeReversalStrategy,
    FundingExtremeReversalStrategyConfig,
)


pytestmark = pytest.mark.asyncio


def _strategy_config() -> StrategyConfig:
    """
    Minimal StrategyConfig for funding strategy integration tests.

    Важливо:
    - required_features deliberately empty here, because generic SignalNormalizer
      may create implicit features from top-level keys only.
    - funding concrete strategies still validate funding domain data through
      FundingTradingStrategy / generate_signal().
    """
    config = StrategyConfig()

    config.routing.event_to_categories = {
        "analytics.funding": [StrategyCategory.FUNDING],
    }

    config.runtime = StrategyRuntimeConfig(
        enabled=True,
        symbols=["BTCUSDT"],
        timeframes=[Timeframe.H1],
        cooldown_seconds=0,
        max_signal_age_seconds=300,
        min_confidence=0.0,
        min_score=0.0,
    )

    config.upsert_strategy(
        StrategyDefinitionConfig(
            name="funding_divergence",
            category=StrategyCategory.FUNDING,
            runtime=StrategyRuntimeConfig(
                enabled=True,
                symbols=["BTCUSDT"],
                timeframes=[Timeframe.H1],
                cooldown_seconds=0,
                max_signal_age_seconds=300,
                min_confidence=0.0,
                min_score=0.0,
            ),
            required_features=set(),
            priority=10,
            weight=1.0,
            tags=["funding", "divergence"],
        )
    )

    config.upsert_strategy(
        StrategyDefinitionConfig(
            name="funding_extreme_reversal",
            category=StrategyCategory.FUNDING,
            runtime=StrategyRuntimeConfig(
                enabled=True,
                symbols=["BTCUSDT"],
                timeframes=[Timeframe.H1],
                cooldown_seconds=0,
                max_signal_age_seconds=300,
                min_confidence=0.0,
                min_score=0.0,
            ),
            required_features=set(),
            priority=20,
            weight=1.0,
            tags=["funding", "extreme_reversal"],
        )
    )

    config.validate()
    return config


def _funding_payload(now: datetime) -> dict:
    """
    Payload shape intentionally matches generic SignalNormalizer behavior:

    - source='funding' lets _resolve_source() return FeatureSource.FUNDING.
    - symbol/timestamp are top-level, as SignalNormalizer expects.
    - domain_data contains normalized StrategyContext funding domain keys.
    - features is optional but gives StrategyRegistry/SignalRouter explicit
      changed features.
    """
    return {
        "source": "funding",
        "symbol": "BTCUSDT",
        "timestamp": now,
        "exchange": "binance",
        "market_type": "usdm_futures",
        "timeframe": "1h",
        "domain_data": {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "timeframe": "1h",
            "exchange_symbol": "BTCUSDT",
            "divergence": {
                "symbol": "BTCUSDT",
                "divergence_type": "bullish_divergence",
                "side": "long",
                "score": 0.82,
                "confidence": 0.78,
                "tags": ["price_divergence", "cvd_divergence"],
                "timestamp": now.isoformat(),
            },
            "extreme": {
                "symbol": "BTCUSDT",
                "extreme_type": "negative_extreme",
                "severity": 0.84,
                "confidence": 0.76,
                "mean_reversion_probability": 0.72,
                "squeeze_probability": 0.61,
                "reversal_risk": True,
                "tags": ["global_extreme", "percentile_extreme"],
                "timestamp": now.isoformat(),
            },
            "pressure": {
                "symbol": "BTCUSDT",
                "direction": "long",
                "pressure_score": 0.68,
                "confidence": 0.72,
                "level": "high",
                "timestamp": now.isoformat(),
            },
            "regime": {
                "symbol": "BTCUSDT",
                "bias": "long",
                "regime": "positive",
                "confidence": 0.64,
                "timestamp": now.isoformat(),
            },
            "flip": {
                "symbol": "BTCUSDT",
                "flip_type": "negative_to_positive",
                "confidence": 0.69,
                "timestamp": now.isoformat(),
            },
            "signal": {
                "symbol": "BTCUSDT",
                "bias": "long",
                "score": 0.71,
                "confidence": 0.70,
                "origin": "divergence",
                "timestamp": now.isoformat(),
            },
        },
        "features": [
            {
                "name": "funding.divergence",
                "value": True,
                "confidence": 0.78,
                "normalized_value": 0.82,
                "freshness_seconds": 300,
            },
            {
                "name": "funding.extreme",
                "value": True,
                "confidence": 0.76,
                "normalized_value": 0.84,
                "freshness_seconds": 300,
            },
            {
                "name": "funding.pressure.score",
                "value": 0.68,
                "confidence": 0.72,
                "normalized_value": 0.68,
                "freshness_seconds": 300,
            },
            {
                "name": "funding.regime.confidence",
                "value": 0.64,
                "confidence": 0.64,
                "normalized_value": 0.64,
                "freshness_seconds": 300,
            },
            {
                "name": "funding.signal.score",
                "value": 0.71,
                "confidence": 0.70,
                "normalized_value": 0.71,
                "freshness_seconds": 300,
            },
        ],
    }


async def test_funding_analytics_payload_reaches_funding_strategies() -> None:
    config = _strategy_config()
    now = datetime.now(timezone.utc)

    normalizer = SignalNormalizer(config=config)
    registry = StrategyRegistry(config=config)
    router = SignalRouter(config=config, registry=registry)

    divergence_strategy = FundingDivergenceStrategy(
        config=config,
        funding_config=FundingDivergenceStrategyConfig(
            min_signal_confidence=0.0,
            min_signal_score=0.0,
            stale_feature_max_age_seconds=300,
            require_fresh_divergence=True,
            require_non_neutral_regime=True,
            require_pressure_alignment=False,
        ),
    )

    extreme_strategy = FundingExtremeReversalStrategy(
        config=config,
        funding_config=FundingExtremeReversalStrategyConfig(
            min_signal_confidence=0.0,
            min_signal_score=0.0,
            stale_feature_max_age_seconds=300,
            require_fresh_extreme=True,
            require_high_pressure_level=True,
            require_reversal_risk=True,
            require_squeeze_risk_or_reversion_probability=True,
        ),
    )

    registry.register_strategy(divergence_strategy, emit_event=False)
    registry.register_strategy(extreme_strategy, emit_event=False)

    payload = _funding_payload(now)

    normalized = normalizer.normalize_event(
        event_name="analytics.funding.updated",
        payload=payload,
        timestamp=now,
    )

    assert normalized.source is FeatureSource.FUNDING
    assert normalized.symbol == "BTCUSDT"
    assert normalized.domain_data["divergence"]["side"] == "long"
    assert normalized.domain_data["extreme"]["extreme_type"] == "negative_extreme"

    context = StrategyContext(
        symbol=normalized.symbol,
        timestamp=normalized.timestamp,
        timeframe=Timeframe.H1,
        metadata={
            "exchange": "binance",
            "market_type": "usdm_futures",
            "timeframe": "1h",
        },
    )

    normalizer.apply_to_context(context, normalized)

    assert context.domain_dict(FeatureSource.FUNDING)["divergence"]["side"] == "long"
    assert context.domain_dict(FeatureSource.FUNDING)["extreme"]["severity"] == 0.84
    assert context.has_feature("funding.divergence")
    assert context.has_feature("funding.extreme")

    route = router.route(
        event_name="analytics.funding.updated",
        context=context,
        source=normalized.source,
        changed_features=[feature.name for feature in normalized.features],
        metadata=normalized.metadata,
    )

    routed_names = set(route.selected_names)

    assert "funding_divergence" in routed_names
    assert "funding_extreme_reversal" in routed_names

    divergence_evaluation = await divergence_strategy.evaluate(context)
    extreme_evaluation = await extreme_strategy.evaluate(context)

    assert divergence_evaluation.passed is True
    assert divergence_evaluation.signal is not None
    assert divergence_evaluation.signal.strategy_name == "funding_divergence"
    assert divergence_evaluation.signal.side is SignalSide.LONG
    assert divergence_evaluation.signal.category is StrategyCategory.FUNDING
    assert "funding_divergence" in divergence_evaluation.signal.metadata[
        "funding_setup_family"
    ]

    assert extreme_evaluation.passed is True
    assert extreme_evaluation.signal is not None
    assert extreme_evaluation.signal.strategy_name == "funding_extreme_reversal"
    assert extreme_evaluation.signal.side is SignalSide.LONG
    assert extreme_evaluation.signal.category is StrategyCategory.FUNDING
    assert "funding_extreme_reversal" in extreme_evaluation.signal.metadata[
        "funding_setup_family"
    ]


async def test_funding_strategy_returns_no_signal_for_missing_domain_data() -> None:
    config = _strategy_config()
    now = datetime.now(timezone.utc)

    strategy = FundingDivergenceStrategy(
        config=config,
        funding_config=FundingDivergenceStrategyConfig(
            min_signal_confidence=0.0,
            min_signal_score=0.0,
        ),
    )

    context = StrategyContext(
        symbol="BTCUSDT",
        timestamp=now,
        timeframe=Timeframe.H1,
    )

    evaluation = await strategy.evaluate(context)

    assert evaluation.passed is False
    assert evaluation.signal is None