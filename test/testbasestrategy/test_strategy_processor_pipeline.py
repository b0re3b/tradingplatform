# tests/strategy/test_strategy_processor_pipeline.py

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from conftest import DummyStrategy, make_runtime_config
from core.event_bus import EventPriority
from strategy.config import (
    BuilderConfig,
    ConfluenceConfig,
    FeatureFreshnessConfig,
    FilterConfig,
    PortfolioCoordinatorConfig,
    RoutingConfig,
    StrategyConfig,
    StrategyRuntimeConfig,
)
from strategy.enums import (
    EntryType,
    FeatureSource,
    FilterDecision,
    MarketRegime,
    SignalOrigin,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    StrategyExecutionQuality,
    StrategyLiquidityClass,
    StrategyMarketType,
    StrategyOrderIntent,
    StrategyTradeTier,
    Timeframe,
)
from strategy.exceptions import (
    BuilderError,
    ConfluenceError,
    SignalNormalizationError,
    SignalRoutingError,
)
from strategy.models import (
    FeatureSnapshot,
    RiskReadySignalPayload,
    StrategyContext,
    StrategySignal,
    utcnow,
)
from strategy.processor import (
    BuildEvaluation,
    ConfluenceEngine,
    CoordinationDecision,
    FilterEvaluation,
    NormalizedPayload,
    PortfolioCoordinator,
    ProcessedSignalBatch,
    RouteDecision,
    SignalBuilder,
    SignalFilterChain,
    SignalNormalizer,
    SignalProcessor,
    SignalRouter,
    SignalScorer,
    WeightedSignal,
)
from strategy.registry import StrategyRegistry
from strategy.state import StrategyRuntimeState


# =============================================================================
# Local helpers
# =============================================================================


def _processor_config(
    *,
    runtime: StrategyRuntimeConfig | None = None,
    confluence_enabled: bool = True,
    min_agreement_count: int = 1,
    min_confidence: float = 0.50,
    min_score: float = 0.0,
    repeated_signal_suppression_seconds: int = 0,
    deduplicate_by_side: bool = True,
    max_signals_per_symbol: int = 3,
    side_cooldown_seconds: int = 0,
    symbol_cooldown_seconds: int = 0,
) -> StrategyConfig:
    config = StrategyConfig(
        runtime=runtime
        or StrategyRuntimeConfig(
            symbols=[],
            timeframes=[Timeframe.M1, Timeframe.M5, Timeframe.M15],
            allowed_regimes=[MarketRegime.UNKNOWN],
            min_confidence=0.50,
            min_score=0.0,
            max_signal_age_seconds=60,
        ),
        routing=RoutingConfig(
            reevaluate_on_any_update=False,
            route_hybrid_on_domain_signal=True,
            allow_partial_context=True,
            stale_feature_threshold_seconds=60,
            event_to_categories={
                "analytics.orderflow.updated": [StrategyCategory.ORDERFLOW],
                "analytics.open_interest.updated": [StrategyCategory.OPEN_INTEREST],
                "analytics.funding.updated": [StrategyCategory.FUNDING],
            },
        ),
        confluence=ConfluenceConfig(
            enabled=confluence_enabled,
            min_agreement_count=min_agreement_count,
            min_confidence=min_confidence,
            min_score=min_score,
            conflict_penalty=0.15,
            confirmation_bonus=0.10,
            max_strategies_per_side=10,
        ),
        filters=FilterConfig(
            max_spread_bps=20.0,
            min_liquidity_score=0.30,
            max_volatility_zscore=3.0,
            min_funding_alignment=-1.0,
        ),
        builders=BuilderConfig(
            default_entry_type=EntryType.LIMIT,
            default_rr_ratio=2.0,
            enable_partial_take_profit=True,
            default_partial_tp_levels=[0.5, 0.5],
            require_invalidation=True,
        ),
        freshness=FeatureFreshnessConfig(
            default_ttl_seconds=60,
            per_feature_ttl_seconds={
                "orderflow_imbalance": 60,
                "liquidity_score": 60,
                "spread_bps": 30,
                "volatility_zscore": 60,
                "funding_alignment": 120,
                "open_interest": 120,
            },
        ),
        portfolio=PortfolioCoordinatorConfig(
            enabled=True,
            max_signals_per_symbol=max_signals_per_symbol,
            deduplicate_by_side=deduplicate_by_side,
            merge_similar_signals=True,
            repeated_signal_suppression_seconds=repeated_signal_suppression_seconds,
            side_cooldown_seconds=side_cooldown_seconds,
            symbol_cooldown_seconds=symbol_cooldown_seconds,
        ),
    )
    config.validate()
    return config


def _make_runtime_signal(
    make_signal,
    *,
    symbol: str = "BTCUSDT",
    strategy_name: str = "dummy_strategy",
    side: SignalSide = SignalSide.LONG,
    confidence: float = 0.85,
    score: float = 0.80,
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    with_plan: bool = False,
) -> StrategySignal:
    signal = make_signal(
        symbol=symbol,
        side=side,
        strategy_name=strategy_name,
        confidence=confidence,
        score=score,
        timestamp=timestamp or utcnow(),
        metadata={
            "exchange": "binance",
            "market_type": "usdm_futures",
            "entry_price": 100.0,
            "stop_loss": 99.0,
            "rr": 2.0,
            "requested_leverage": 2.0,
            **dict(metadata or {}),
        },
        with_execution_plan=with_plan,
    )
    signal.validate()
    return signal


def _context_with_market_features(make_context, make_feature) -> StrategyContext:
    return make_context(
        symbol="BTCUSDT",
        timeframe=Timeframe.M1,
        features=[
            make_feature(name="orderflow_imbalance", value=0.72, normalized_value=0.72),
            make_feature(name="liquidity_score", value=0.85, normalized_value=0.85),
            make_feature(name="spread_bps", value=4.0, normalized_value=None),
            make_feature(name="volatility_zscore", value=1.2, normalized_value=None),
            make_feature(name="funding_alignment", value=0.1, normalized_value=0.1),
        ],
        metadata={
            "exchange": "binance",
            "market_type": "usdm_futures",
        },
    )


def _make_registered_processor(
    *,
    config: StrategyConfig,
    event_bus,
    scheduler,
    strategy: DummyStrategy,
) -> tuple[SignalProcessor, StrategyRegistry, StrategyRuntimeState]:
    state = StrategyRuntimeState()
    registry = StrategyRegistry(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )
    registry.register_strategy(strategy, emit_event=False)

    processor = SignalProcessor(
        config=config,
        registry=registry,
        state=state,
        event_bus=event_bus,
        scheduler=scheduler,
    )
    return processor, registry, state


# =============================================================================
# Pipeline DTOs
# =============================================================================


class TestProcessorDTOs:
    def test_normalized_payload_timestamp_is_aware(self) -> None:
        payload = NormalizedPayload(
            source=FeatureSource.ORDERFLOW,
            symbol="BTCUSDT",
            timestamp=datetime(2026, 5, 20, 12, 0, 0),
        )

        assert payload.timestamp.tzinfo is timezone.utc

    def test_route_decision_properties(self, dummy_strategy) -> None:
        route = RouteDecision(
            event_name="analytics.orderflow.updated",
            symbol="BTCUSDT",
            selected=[dummy_strategy],
        )

        assert route.selected_names == [dummy_strategy.strategy_name]
        assert route.total_selected == 1
        assert not route.is_empty

    def test_weighted_signal_rejects_negative_weights(self, strategy_signal) -> None:
        weighted = WeightedSignal(
            signal=strategy_signal,
            category_weight=-1.0,
            regime_weight=1.0,
            strategy_weight=1.0,
            final_weight=1.0,
            weighted_score=0.5,
            weighted_confidence=0.5,
        )

        with pytest.raises(ConfluenceError):
            weighted.validate()

    def test_filter_evaluation_add_result_tracks_block_and_warn(
        self,
        strategy_signal,
    ) -> None:
        evaluation = FilterEvaluation(
            signal=strategy_signal,
            context_symbol=strategy_signal.symbol,
        )

        from strategy.models import FilterResult

        evaluation.add_result(
            FilterResult(
                name="spread",
                decision=FilterDecision.WARN,
                reason="spread_warning",
            )
        )
        evaluation.add_result(
            FilterResult(
                name="liquidity",
                decision=FilterDecision.BLOCK,
                reason="liquidity_block",
            )
        )

        assert not evaluation.accepted
        assert evaluation.has_warnings
        assert evaluation.has_blocks
        assert evaluation.warning_filters == ["spread"]
        assert evaluation.blocking_filters == ["liquidity"]
        assert "liquidity_block" in evaluation.reasons

    def test_build_evaluation_reject_is_idempotent(self, strategy_signal) -> None:
        evaluation = BuildEvaluation(
            signal=strategy_signal,
            context_symbol=strategy_signal.symbol,
        )

        evaluation.reject("missing_entry")
        evaluation.reject("missing_entry")

        assert not evaluation.accepted
        assert evaluation.reasons == ["missing_entry"]

    def test_coordination_decision_final_signals_prefers_merged(
        self,
        make_signal,
    ) -> None:
        accepted = make_signal(strategy_name="accepted")
        merged = make_signal(strategy_name="merged")

        decision = CoordinationDecision(
            symbol="BTCUSDT",
            timestamp=utcnow(),
            accepted_signals=[accepted],
            merged_signals=[merged],
        )

        assert decision.final_signals == [merged]
        assert decision.selected_names == ["merged"]

    def test_processed_signal_batch_timestamp_is_aware(self) -> None:
        batch = ProcessedSignalBatch(
            symbol="BTCUSDT",
            timestamp=datetime(2026, 5, 20, 12, 0, 0),
        )

        assert batch.timestamp.tzinfo is timezone.utc
        assert not batch.accepted
        assert not batch.emitted


# =============================================================================
# SignalNormalizer
# =============================================================================


class TestSignalNormalizer:
    def test_normalize_event_with_explicit_features(
        self,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        config = _processor_config()
        normalizer = SignalNormalizer(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        ts = datetime(2026, 5, 20, 12, 0, 0)

        normalized = normalizer.normalize_event(
            event_name="analytics.orderflow.updated",
            payload={
                "symbol": " BTCUSDT ",
                "timestamp": ts,
                "confidence": 0.9,
                "features": [
                    {
                        "name": "orderflow_imbalance",
                        "value": 0.72,
                        "normalized_value": 2.0,
                        "confidence": 2.0,
                        "freshness_seconds": 30,
                    }
                ],
                "metadata": {"ignored_for_domain": True},
                "raw_context": {"delta": 123},
            },
        )

        assert normalized.source is FeatureSource.ORDERFLOW
        assert normalized.symbol == "BTCUSDT"
        assert normalized.timestamp.tzinfo is timezone.utc
        assert normalized.domain_data == {"raw_context": {"delta": 123}}
        assert len(normalized.features) == 1

        feature = normalized.features[0]
        assert feature.name == "orderflow_imbalance"
        assert feature.confidence == 1.0
        assert feature.normalized_value == 1.0
        assert feature.freshness_seconds == 30

    @pytest.mark.parametrize(
        ("event_name", "source"),
        [
            ("analytics.orderflow.updated", FeatureSource.ORDERFLOW),
            ("analytics.cvd.signal", FeatureSource.ORDERFLOW),
            ("analytics.open_interest.updated", FeatureSource.OPEN_INTEREST),
            ("analytics.funding.updated", FeatureSource.FUNDING),
            ("analytics.spread.updated", FeatureSource.SPREADS),
            ("analytics.whale.updated", FeatureSource.WHALES),
            ("analytics.spoofing.updated", FeatureSource.SPOOFING),
            ("analytics.liquidation.updated", FeatureSource.LIQUIDATIONS),
            ("analytics.price_action.updated", FeatureSource.PRICE_ACTION),
        ],
    )
    def test_resolve_source_from_event_name(
        self,
        mock_event_bus,
        mock_scheduler,
        event_name: str,
        source: FeatureSource,
    ) -> None:
        normalizer = SignalNormalizer(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        normalized = normalizer.normalize_event(
            event_name=event_name,
            payload={
                "symbol": "BTCUSDT",
                "timestamp": utcnow(),
                "value": 1.0,
            },
        )

        assert normalized.source is source

    def test_explicit_source_overrides_event_name(
        self,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        normalizer = SignalNormalizer(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        normalized = normalizer.normalize_event(
            event_name="analytics.unknown.updated",
            payload={
                "source": "orderflow",
                "symbol": "BTCUSDT",
                "orderflow_imbalance": 0.7,
            },
        )

        assert normalized.source is FeatureSource.ORDERFLOW

    @pytest.mark.parametrize(
        ("payload_ts", "expected"),
        [
            (1_764_156_000, datetime.fromtimestamp(1_764_156_000, tz=timezone.utc)),
            (
                1_764_156_000_000,
                datetime.fromtimestamp(1_764_156_000, tz=timezone.utc),
            ),
        ],
    )
    def test_timestamp_seconds_and_milliseconds(
        self,
        mock_event_bus,
        mock_scheduler,
        payload_ts: int,
        expected: datetime,
    ) -> None:
        normalizer = SignalNormalizer(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        normalized = normalizer.normalize_event(
            event_name="analytics.orderflow.updated",
            payload={
                "symbol": "BTCUSDT",
                "timestamp": payload_ts,
                "orderflow_imbalance": 0.7,
            },
        )

        assert normalized.timestamp == expected

    def test_implicit_features_ignore_metadata_and_private_keys(
        self,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        normalizer = SignalNormalizer(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        normalized = normalizer.normalize_event(
            event_name="analytics.orderflow.updated",
            payload={
                "symbol": "BTCUSDT",
                "timestamp": utcnow(),
                "confidence": 0.8,
                "orderflow_imbalance": 0.7,
                "spread_bps": 3.5,
                "is_valid": True,
                "_private": 1,
                "metadata": {"ignored": True},
            },
        )

        names = {feature.name for feature in normalized.features}

        assert {"confidence", "orderflow_imbalance", "spread_bps", "is_valid"} <= names
        assert "_private" not in names
        assert "metadata" not in names

    @pytest.mark.parametrize(
        ("event_name", "payload", "match"),
        [
            ("", {"symbol": "BTCUSDT"}, "event_name cannot be empty"),
            (
                "analytics.orderflow.updated",
                ["bad"],
                "payload must be a dict",
            ),
            (
                "analytics.unknown.updated",
                {"symbol": "BTCUSDT"},
                "unable to resolve FeatureSource",
            ),
            (
                "analytics.orderflow.updated",
                {"timestamp": utcnow()},
                "valid symbol",
            ),
            (
                "analytics.orderflow.updated",
                {"symbol": "BTCUSDT", "timestamp": object()},
                "unsupported timestamp",
            ),
            (
                "analytics.orderflow.updated",
                {"symbol": "BTCUSDT", "features": "bad"},
                "features",
            ),
            (
                "analytics.orderflow.updated",
                {"symbol": "BTCUSDT", "features": [{}]},
                "feature item",
            ),
            (
                "analytics.orderflow.updated",
                {
                    "symbol": "BTCUSDT",
                    "features": [
                        {
                            "name": "x",
                            "value": 1,
                            "freshness_seconds": 0,
                        }
                    ],
                },
                "freshness_seconds",
            ),
        ],
    )
    def test_normalize_event_invalid_payloads_raise(
        self,
        mock_event_bus,
        mock_scheduler,
        event_name: str,
        payload: Any,
        match: str,
    ) -> None:
        normalizer = SignalNormalizer(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        with pytest.raises(SignalNormalizationError, match=match):
            normalizer.normalize_event(
                event_name=event_name,
                payload=payload,
            )

    def test_apply_to_context_merges_domain_and_features(
        self,
        mock_event_bus,
        mock_scheduler,
        make_context,
        make_feature,
    ) -> None:
        config = _processor_config()
        normalizer = SignalNormalizer(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = make_context(symbol="BTCUSDT", features=[])

        normalized = NormalizedPayload(
            source=FeatureSource.ORDERFLOW,
            symbol="BTCUSDT",
            timestamp=utcnow(),
            domain_data={"pressure": 0.8},
            features=[
                make_feature(
                    name="orderflow_imbalance",
                    source=FeatureSource.ORDERFLOW,
                    symbol="BTCUSDT",
                    value=0.7,
                )
            ],
            metadata={"event_name": "analytics.orderflow.updated"},
        )

        updated = normalizer.apply_to_context(context, normalized)

        assert updated is context
        assert context.get_feature("orderflow_imbalance") == 0.7
        assert context.domain_dict(FeatureSource.ORDERFLOW)["pressure"] == 0.8
        assert context.metadata["last_source"] == "orderflow"
        assert context.metadata["last_event_name"] == "analytics.orderflow.updated"

    def test_apply_to_context_rejects_symbol_mismatch(
        self,
        mock_event_bus,
        mock_scheduler,
        make_context,
    ) -> None:
        normalizer = SignalNormalizer(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = make_context(symbol="BTCUSDT")

        normalized = NormalizedPayload(
            source=FeatureSource.ORDERFLOW,
            symbol="ETHUSDT",
            timestamp=utcnow(),
        )

        with pytest.raises(SignalNormalizationError, match="context symbol"):
            normalizer.apply_to_context(context, normalized)


# =============================================================================
# SignalRouter
# =============================================================================


class TestSignalRouter:
    def test_route_selects_applicable_strategies_by_event_category(
        self,
        mock_event_bus,
        mock_scheduler,
        make_definition,
        make_context,
        make_feature,
    ) -> None:
        config = _processor_config()
        definition = make_definition(
            name="orderflow_strategy",
            category=StrategyCategory.ORDERFLOW,
            runtime=make_runtime_config(symbols=["BTCUSDT"]),
            required_features=("orderflow_imbalance",),
            priority=10,
        )
        config.upsert_strategy(definition)

        strategy = DummyStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=definition,
        )

        registry = StrategyRegistry(config=config, event_bus=mock_event_bus)
        registry.register_strategy(strategy, emit_event=False)

        router = SignalRouter(
            config=config,
            registry=registry,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)

        route = router.route(
            event_name="analytics.orderflow.updated",
            context=context,
            source=FeatureSource.ORDERFLOW,
            changed_features=["orderflow_imbalance"],
        )

        assert route.selected == [strategy]
        assert route.selected_names == ["orderflow_strategy"]
        assert route.categories_used == [StrategyCategory.ORDERFLOW]
        assert route.matched_features == ["orderflow_imbalance"]
        assert not route.skipped

    def test_route_skips_not_applicable_strategy(
        self,
        mock_event_bus,
        mock_scheduler,
        make_definition,
        make_context,
        make_feature,
    ) -> None:
        config = _processor_config()
        definition = make_definition(
            name="eth_only_strategy",
            category=StrategyCategory.ORDERFLOW,
            runtime=make_runtime_config(symbols=["ETHUSDT"]),
            required_features=("orderflow_imbalance",),
        )
        config.upsert_strategy(definition)

        strategy = DummyStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=definition,
        )
        registry = StrategyRegistry(config=config, event_bus=mock_event_bus)
        registry.register_strategy(strategy, emit_event=False)

        router = SignalRouter(
            config=config,
            registry=registry,
            event_bus=mock_event_bus,
        )
        context = _context_with_market_features(make_context, make_feature)

        route = router.route(
            event_name="analytics.orderflow.updated",
            context=context,
            source=FeatureSource.ORDERFLOW,
            changed_features=["orderflow_imbalance"],
        )

        assert route.selected == []
        assert route.skipped[strategy.strategy_name] == "strategy_not_applicable"

    def test_route_requires_event_name(
        self,
        strategy_config,
        mock_event_bus,
        strategy_context,
    ) -> None:
        registry = StrategyRegistry(config=strategy_config)
        router = SignalRouter(
            config=strategy_config,
            registry=registry,
            event_bus=mock_event_bus,
        )

        with pytest.raises(SignalRoutingError, match="event_name cannot be empty"):
            router.route(event_name="", context=strategy_context)

    @pytest.mark.asyncio()
    async def test_emit_signal_generated_uses_event_priority(
        self,
        strategy_config,
        mock_event_bus,
        mock_scheduler,
        risk_ready_strategy_signal,
        strategy_context,
    ) -> None:
        registry = StrategyRegistry(config=strategy_config)
        router = SignalRouter(
            config=strategy_config,
            registry=registry,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        processor = SignalProcessor(
            config=strategy_config,
            registry=registry,
            state=StrategyRuntimeState(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        risk_ready_strategy_signal.metadata["priority_score"] = 0.90

        payload = processor.to_risk_payload(
            signal=risk_ready_strategy_signal,
            context=strategy_context,
        )

        await router.emit_signal_generated(payload=payload)

        assert mock_event_bus.topic_emitted("signal.generated")
        event = mock_event_bus.emitted[-1]
        assert event.priority is EventPriority.HIGH
        assert event.payload["signal_id"] == risk_ready_strategy_signal.signal_id

    @pytest.mark.asyncio()
    async def test_emit_signal_rejected(
        self,
        strategy_config,
        mock_event_bus,
        strategy_signal,
    ) -> None:
        router = SignalRouter(
            config=strategy_config,
            registry=StrategyRegistry(config=strategy_config),
            event_bus=mock_event_bus,
        )

        await router.emit_signal_rejected(
            signal=strategy_signal,
            symbol=strategy_signal.symbol,
            reason="unit_reject",
            metadata={"test": True},
        )

        assert mock_event_bus.topic_emitted("signal.rejected")
        payload = mock_event_bus.emitted[-1].payload
        assert payload["signal_id"] == strategy_signal.signal_id
        assert payload["reason"] == "unit_reject"
        assert payload["metadata"] == {"test": True}


# =============================================================================
# SignalScorer and ConfluenceEngine
# =============================================================================


class TestSignalScorerAndConfluence:
    def test_score_signal_enriches_priority_metadata(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
    ) -> None:
        config = _processor_config()
        scorer = SignalScorer(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        signal = _make_runtime_signal(
            make_signal,
            confidence=0.86,
            score=0.70,
            metadata={
                "priority_components": {
                    "setup_quality": 0.8,
                    "confluence_score": 0.7,
                    "liquidity_score": 0.9,
                    "risk_reward_score": 0.8,
                    "execution_quality_score": 0.9,
                    "regime_alignment_score": 0.8,
                    "freshness_score": 1.0,
                }
            },
        )

        scored = scorer.score_signal(signal=signal, context=context)

        assert scored is signal
        assert 0.0 <= signal.metadata["priority_score"] <= 1.0
        assert signal.metadata["tier"] in StrategyTradeTier.values()
        assert signal.metadata["liquidity_class"] in StrategyLiquidityClass.values()
        assert signal.metadata["execution_quality"] in StrategyExecutionQuality.values()
        assert signal.score >= 0.70

    def test_score_signals_requires_non_empty_same_symbol(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
    ) -> None:
        scorer = SignalScorer(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        with pytest.raises(ConfluenceError, match="signals cannot be empty"):
            scorer.score_signals(signals=[])

        btc = _make_runtime_signal(make_signal, symbol="BTCUSDT")
        eth = _make_runtime_signal(make_signal, symbol="ETHUSDT")

        with pytest.raises(ConfluenceError, match="same symbol"):
            scorer.score_signals(signals=[btc, eth])

    def test_confluence_accepts_single_signal_when_config_allows_it(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
    ) -> None:
        config = _processor_config(
            confluence_enabled=True,
            min_agreement_count=1,
            min_confidence=0.5,
            min_score=0.0,
        )
        engine = ConfluenceEngine(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        signal = _make_runtime_signal(
            make_signal,
            strategy_name="single_strategy",
            side=SignalSide.LONG,
            confidence=0.9,
            score=0.8,
        )

        evaluation = engine.evaluate(signals=[signal], context=context)

        assert evaluation.accepted
        assert evaluation.result is not None
        assert evaluation.result.accepted
        assert evaluation.result.side is SignalSide.LONG
        assert evaluation.accepted_signals == [signal]
        assert evaluation.merged_signal is signal
        assert signal.origin is SignalOrigin.CONFLUENCE

    def test_confluence_disabled_passes_signals_through(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
    ) -> None:
        config = _processor_config(confluence_enabled=False)
        engine = ConfluenceEngine(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        signal = _make_runtime_signal(make_signal)

        evaluation = engine.evaluate(signals=[signal], context=context)

        assert evaluation.accepted
        assert evaluation.accepted_signals == [signal]
        assert evaluation.merged_signal is None
        assert evaluation.result is None

    def test_confluence_rejects_insufficient_agreement(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
    ) -> None:
        config = _processor_config(
            confluence_enabled=True,
            min_agreement_count=2,
            min_confidence=0.5,
            min_score=0.0,
        )
        engine = ConfluenceEngine(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        signal = _make_runtime_signal(make_signal)

        evaluation = engine.evaluate(signals=[signal], context=context)

        assert not evaluation.accepted
        assert evaluation.result is not None
        assert not evaluation.result.accepted
        assert "insufficient_agreement_count" in evaluation.reasons

    def test_confluence_side_conflict_records_conflict(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
    ) -> None:
        config = _processor_config(
            confluence_enabled=True,
            min_agreement_count=1,
            min_confidence=0.1,
            min_score=0.0,
        )
        engine = ConfluenceEngine(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        long_signal = _make_runtime_signal(
            make_signal,
            strategy_name="long_a",
            side=SignalSide.LONG,
            confidence=0.9,
            score=0.8,
        )
        short_signal = _make_runtime_signal(
            make_signal,
            strategy_name="short_b",
            side=SignalSide.SHORT,
            confidence=0.7,
            score=0.6,
        )

        evaluation = engine.evaluate(
            signals=[long_signal, short_signal],
            context=context,
        )

        assert evaluation.result is not None
        assert evaluation.result.side is SignalSide.LONG
        assert evaluation.result.conflicts
        assert evaluation.accepted_signals == [long_signal]
        assert evaluation.merged_signal is long_signal


# =============================================================================
# SignalFilterChain
# =============================================================================


class TestSignalFilterChain:
    def test_filter_accepts_healthy_signal(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
    ) -> None:
        filters = SignalFilterChain(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        signal = _make_runtime_signal(make_signal, confidence=0.9, score=0.8)

        evaluation = filters.evaluate_signal(signal=signal, context=context)

        assert evaluation.accepted
        assert not evaluation.has_blocks

    @pytest.mark.parametrize(
        ("mutator", "expected_filter", "expected_reason"),
        [
            (
                lambda signal, context: setattr(signal, "symbol", "ETHUSDT"),
                "symbol_match",
                "signal_symbol_does_not_match_context",
            ),
            (
                lambda signal, context: setattr(signal, "side", SignalSide.FLAT),
                "directional_signal",
                "signal_side_is_not_directional",
            ),
            (
                lambda signal, context: setattr(signal, "confidence", 0.1),
                "min_confidence",
                "confidence_below_runtime_threshold",
            ),
            (
                lambda signal, context: setattr(signal, "score", 0.0),
                "min_score",
                "score_below_runtime_threshold",
            ),
            (
                lambda signal, context: setattr(
                    signal,
                    "timestamp",
                    utcnow() - timedelta(seconds=999),
                ),
                "signal_age",
                "signal_too_old",
            ),
            (
                lambda signal, context: context.feature_map.__setitem__(
                    "orderflow_imbalance",
                    FeatureSnapshot(
                        name="orderflow_imbalance",
                        value=0.1,
                        source=FeatureSource.ORDERFLOW,
                        symbol=context.symbol,
                        timestamp=utcnow() - timedelta(seconds=999),
                        freshness_seconds=10,
                    ),
                ),
                "feature_freshness",
                "context_contains_stale_features",
            ),
        ],
    )
    def test_filter_blocks_bad_signals(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
        mutator,
        expected_filter: str,
        expected_reason: str,
    ) -> None:
        runtime = StrategyRuntimeConfig(
            symbols=[],
            timeframes=[Timeframe.M1],
            allowed_regimes=[MarketRegime.UNKNOWN],
            min_confidence=0.5,
            min_score=0.2,
            max_signal_age_seconds=60,
        )
        filters = SignalFilterChain(
            config=_processor_config(runtime=runtime),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        signal = _make_runtime_signal(make_signal, confidence=0.9, score=0.8)

        mutator(signal, context)

        evaluation = filters.evaluate_signal(signal=signal, context=context)

        assert not evaluation.accepted
        assert expected_filter in evaluation.blocking_filters
        assert expected_reason in evaluation.reasons

    def test_filter_blocks_blocked_execution_quality(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
    ) -> None:
        filters = SignalFilterChain(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        signal = _make_runtime_signal(
            make_signal,
            metadata={"execution_quality": StrategyExecutionQuality.BLOCKED.value},
        )

        evaluation = filters.evaluate_signal(signal=signal, context=context)

        assert not evaluation.accepted
        assert "execution_quality" in evaluation.blocking_filters

    def test_apply_marks_rejected_signal_and_adds_filter_results(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
    ) -> None:
        runtime = StrategyRuntimeConfig(
            symbols=[],
            timeframes=[Timeframe.M1],
            allowed_regimes=[MarketRegime.UNKNOWN],
            min_confidence=0.95,
            min_score=0.0,
            max_signal_age_seconds=60,
        )
        filters = SignalFilterChain(
            config=_processor_config(runtime=runtime),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        signal = _make_runtime_signal(make_signal, confidence=0.5, score=0.8)

        accepted = filters.apply(signals=[signal], context=context)

        assert accepted == []
        assert signal.status is SignalStatus.REJECTED
        assert signal.filter_results
        assert "confidence_below_runtime_threshold" in signal.reasons


# =============================================================================
# SignalBuilder
# =============================================================================


class TestSignalBuilder:
    def test_build_signal_creates_execution_plan(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
    ) -> None:
        builder = SignalBuilder(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        signal = _make_runtime_signal(
            make_signal,
            side=SignalSide.LONG,
            metadata={
                "entry_price": 100.0,
                "stop_loss": 99.0,
                "rr": 2.0,
                "requested_leverage": 3.0,
            },
        )

        evaluation = builder.build_signal(signal=signal, context=context)

        assert evaluation.accepted
        assert evaluation.execution_plan is signal.execution_plan
        assert signal.entry_plan is not None
        assert signal.invalidation_plan is not None
        assert signal.exit_plan is not None
        assert signal.execution_plan is not None
        assert signal.primary_entry_price == 100.0
        assert signal.primary_stop_loss == 99.0
        assert signal.primary_take_profit == 102.0
        assert signal.execution_plan.leverage == 3.0

    def test_build_signal_short_take_profit_is_below_entry(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
    ) -> None:
        builder = SignalBuilder(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        signal = _make_runtime_signal(
            make_signal,
            side=SignalSide.SHORT,
            metadata={
                "entry_price": 100.0,
                "stop_loss": 101.0,
                "rr": 2.0,
            },
        )

        evaluation = builder.build_signal(signal=signal, context=context)

        assert evaluation.accepted
        assert signal.primary_take_profit == 98.0

    @pytest.mark.parametrize(
        ("metadata", "reason"),
        [
            ({"entry_price": 0, "stop_loss": 99.0}, "entry_plan_failed"),
            ({"entry_price": 100.0}, "invalidation_plan_failed"),
            ({"entry_price": 100.0, "stop_loss": 100.0}, "target_plan_failed"),
        ],
    )
    def test_build_signal_rejects_invalid_trade_plan_inputs(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
        metadata: dict[str, Any],
        reason: str,
    ) -> None:
        builder = SignalBuilder(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        signal = _make_runtime_signal(
            make_signal,
            metadata=metadata,
        )

        evaluation = builder.build_signal(signal=signal, context=context)

        assert not evaluation.accepted
        assert reason in evaluation.reasons

    def test_build_many_splits_accepted_and_rejected(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
    ) -> None:
        builder = SignalBuilder(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        valid = _make_runtime_signal(
            make_signal,
            strategy_name="valid",
            metadata={"entry_price": 100.0, "stop_loss": 99.0},
        )
        invalid = _make_runtime_signal(
            make_signal,
            strategy_name="invalid",
            metadata={"entry_price": 100.0},
        )

        built, rejected = builder.build_many(
            signals=[valid, invalid],
            context=context,
        )

        assert built == [valid]
        assert rejected["invalid"]

    def test_assert_risk_ready_requires_execution_plan(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
    ) -> None:
        builder = SignalBuilder(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        signal = _make_runtime_signal(make_signal, with_plan=False)

        with pytest.raises(BuilderError):
            builder.assert_risk_ready(signal)

    def test_assert_risk_ready_accepts_built_signal(
        self,
        mock_event_bus,
        mock_scheduler,
        risk_ready_strategy_signal,
    ) -> None:
        builder = SignalBuilder(
            config=_processor_config(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        builder.assert_risk_ready(risk_ready_strategy_signal)


# =============================================================================
# PortfolioCoordinator
# =============================================================================


class TestPortfolioCoordinator:
    def test_coordinate_accepts_and_updates_cooldowns(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
        attach_plan,
    ) -> None:
        config = _processor_config(
            side_cooldown_seconds=10,
            symbol_cooldown_seconds=10,
        )
        state = StrategyRuntimeState()
        coordinator = PortfolioCoordinator(
            config=config,
            state=state,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        signal = attach_plan(
            _make_runtime_signal(make_signal, strategy_name="accepted"),
        )

        decision = coordinator.coordinate(signals=[signal], context=context)

        assert decision.accepted
        assert decision.final_signals == [signal]
        assert state.cooldowns.is_strategy_blocked(
            symbol="BTCUSDT",
            strategy_name="accepted",
        )
        assert state.cooldowns.is_side_blocked(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
        )

    def test_coordinate_rejects_empty_signal_list(
        self,
        mock_event_bus,
        mock_scheduler,
        make_context,
        make_feature,
    ) -> None:
        coordinator = PortfolioCoordinator(
            config=_processor_config(),
            state=StrategyRuntimeState(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)

        decision = coordinator.coordinate(signals=[], context=context)

        assert not decision.accepted
        assert decision.reasons == ["no_signals_to_coordinate"]

    def test_coordinate_suppresses_repeating_signal(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
        attach_plan,
    ) -> None:
        config = _processor_config(
            repeated_signal_suppression_seconds=120,
            deduplicate_by_side=False,
        )
        state = StrategyRuntimeState()
        previous = attach_plan(
            _make_runtime_signal(
                make_signal,
                strategy_name="repeat",
                timestamp=utcnow(),
            )
        )
        state.update_signal(previous, active=True)

        coordinator = PortfolioCoordinator(
            config=config,
            state=state,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)
        current = attach_plan(
            _make_runtime_signal(
                make_signal,
                strategy_name="repeat",
                timestamp=utcnow(),
            )
        )

        decision = coordinator.coordinate(signals=[current], context=context)

        assert not decision.accepted
        assert decision.rejected_signals["repeat"] == "repeating_signal_suppressed"

    def test_coordinate_deduplicates_by_side_keeps_best_signal(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
        attach_plan,
    ) -> None:
        config = _processor_config(
            repeated_signal_suppression_seconds=0,
            deduplicate_by_side=True,
        )
        coordinator = PortfolioCoordinator(
            config=config,
            state=StrategyRuntimeState(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)

        weak = attach_plan(
            _make_runtime_signal(
                make_signal,
                strategy_name="weak",
                score=0.5,
                confidence=0.6,
            )
        )
        strong = attach_plan(
            _make_runtime_signal(
                make_signal,
                strategy_name="strong",
                score=0.9,
                confidence=0.9,
            )
        )

        decision = coordinator.coordinate(signals=[weak, strong], context=context)

        assert decision.accepted
        assert decision.final_signals == [strong]
        assert decision.rejected_signals["weak"] == "deduplicated_by_side"

    def test_coordinate_applies_symbol_limit(
        self,
        mock_event_bus,
        mock_scheduler,
        make_signal,
        make_context,
        make_feature,
        attach_plan,
    ) -> None:
        config = _processor_config(
            repeated_signal_suppression_seconds=0,
            deduplicate_by_side=False,
            max_signals_per_symbol=1,
        )
        coordinator = PortfolioCoordinator(
            config=config,
            state=StrategyRuntimeState(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)

        first = attach_plan(
            _make_runtime_signal(
                make_signal,
                strategy_name="first",
                score=0.7,
                confidence=0.7,
            )
        )
        second = attach_plan(
            _make_runtime_signal(
                make_signal,
                strategy_name="second",
                side=SignalSide.SHORT,
                score=0.9,
                confidence=0.9,
            )
        )

        decision = coordinator.coordinate(signals=[first, second], context=context)

        assert decision.accepted
        assert len(decision.final_signals) == 1
        assert decision.final_signals[0] is second
        assert decision.rejected_signals["first"] == "symbol_signal_limit_exceeded"


# =============================================================================
# SignalProcessor facade
# =============================================================================


class TestSignalProcessorFacade:
    @pytest.mark.asyncio()
    async def test_process_event_full_pipeline_accepts_and_emits_signal_generated(
        self,
        mock_event_bus,
        mock_scheduler,
        make_definition,
    ) -> None:
        config = _processor_config(
            confluence_enabled=True,
            min_agreement_count=1,
            min_confidence=0.5,
            min_score=0.0,
        )
        definition = make_definition(
            name="dummy_strategy",
            category=StrategyCategory.ORDERFLOW,
            runtime=make_runtime_config(
                symbols=["BTCUSDT"],
                timeframes=[Timeframe.M1],
                min_confidence=0.5,
                min_score=0.0,
            ),
            required_features=("orderflow_imbalance",),
            priority=10,
        )
        config.upsert_strategy(definition)

        strategy = DummyStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=definition,
            confidence=0.9,
            score=0.85,
        )
        processor, registry, state = _make_registered_processor(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            strategy=strategy,
        )

        batch = await processor.process_event(
            event_name="analytics.orderflow.updated",
            payload={
                "symbol": "BTCUSDT",
                "timestamp": utcnow(),
                "confidence": 0.9,
                "orderflow_imbalance": 0.72,
                "liquidity_score": 0.85,
                "spread_bps": 4.0,
                "volatility_zscore": 1.0,
                "funding_alignment": 0.1,
                "entry_price": 100.0,
                "stop_loss": 99.0,
                "rr": 2.0,
                "requested_leverage": 2.0,
                "exchange": "binance",
                "market_type": "usdm_futures",
            },
            emit=True,
        )

        assert batch.accepted
        assert batch.emitted
        assert batch.normalized is not None
        assert batch.context is not None
        assert batch.route is not None
        assert batch.evaluations
        assert batch.raw_signals
        assert batch.filtered_signals
        assert batch.confluence is not None
        assert batch.coordinated is not None
        assert batch.final_signals
        assert batch.risk_payloads
        assert mock_event_bus.topic_emitted("signal.generated")

        final_signal = batch.final_signals[0]
        assert final_signal.status is SignalStatus.PENDING
        assert state.get_signal_by_id(final_signal.signal_id) is final_signal
        assert state.metrics.signals_generated >= 1

        emitted_payload = mock_event_bus.emitted[-1].payload
        assert emitted_payload["signal_id"] == final_signal.signal_id
        assert emitted_payload["symbol"] == "BTCUSDT"
        assert emitted_payload["side"] == "long"
        assert emitted_payload["market_type"] == "usdm_futures"

    @pytest.mark.asyncio()
    async def test_process_event_with_emit_false_builds_payload_but_does_not_emit(
        self,
        mock_event_bus,
        mock_scheduler,
        make_definition,
    ) -> None:
        config = _processor_config(confluence_enabled=False)
        definition = make_definition(
            name="dummy_strategy",
            category=StrategyCategory.ORDERFLOW,
            runtime=make_runtime_config(
                symbols=["BTCUSDT"],
                timeframes=[Timeframe.M1],
            ),
            required_features=("orderflow_imbalance",),
        )
        config.upsert_strategy(definition)

        strategy = DummyStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=definition,
            confidence=0.9,
            score=0.85,
        )
        processor, _, _ = _make_registered_processor(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            strategy=strategy,
        )

        batch = await processor.process_event(
            event_name="analytics.orderflow.updated",
            payload={
                "symbol": "BTCUSDT",
                "orderflow_imbalance": 0.72,
                "liquidity_score": 0.85,
                "entry_price": 100.0,
                "stop_loss": 99.0,
                "market_type": "usdm_futures",
            },
            emit=False,
        )

        assert batch.accepted
        assert not batch.emitted
        assert batch.risk_payloads
        assert not mock_event_bus.topic_emitted("signal.generated")

    @pytest.mark.asyncio()
    async def test_process_event_no_routed_strategies_returns_rejected_batch_without_emit(
        self,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        config = _processor_config()
        state = StrategyRuntimeState()
        registry = StrategyRegistry(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        processor = SignalProcessor(
            config=config,
            registry=registry,
            state=state,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        batch = await processor.process_event(
            event_name="analytics.orderflow.updated",
            payload={
                "symbol": "BTCUSDT",
                "orderflow_imbalance": 0.72,
            },
        )

        assert not batch.accepted
        assert batch.reasons == ["no_strategies_routed"]
        assert state.metrics.applicability_skipped == 1
        assert not mock_event_bus.topic_emitted("signal.generated")

    @pytest.mark.asyncio()
    async def test_process_event_no_passed_signals_emits_rejected_batch(
        self,
        mock_event_bus,
        mock_scheduler,
        make_definition,
    ) -> None:
        from conftest import NoSignalStrategy

        config = _processor_config()
        definition = make_definition(
            name="no_signal_strategy",
            category=StrategyCategory.ORDERFLOW,
            runtime=make_runtime_config(
                symbols=["BTCUSDT"],
                timeframes=[Timeframe.M1],
            ),
            required_features=("orderflow_imbalance",),
        )
        config.upsert_strategy(definition)

        strategy = NoSignalStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=definition,
        )
        processor, _, _ = _make_registered_processor(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            strategy=strategy,
        )

        batch = await processor.process_event(
            event_name="analytics.orderflow.updated",
            payload={
                "symbol": "BTCUSDT",
                "orderflow_imbalance": 0.72,
            },
        )

        assert not batch.accepted
        assert "no_passed_strategy_signals" in batch.reasons
        assert mock_event_bus.topic_emitted("signal.rejected")
        assert not mock_event_bus.topic_emitted("signal.generated")

    @pytest.mark.asyncio()
    async def test_process_event_strategy_exception_is_captured_as_failed_evaluation(
        self,
        mock_event_bus,
        mock_scheduler,
        make_definition,
    ) -> None:
        from conftest import FailingStrategy

        config = _processor_config()
        definition = make_definition(
            name="failing_strategy",
            category=StrategyCategory.ORDERFLOW,
            runtime=make_runtime_config(
                symbols=["BTCUSDT"],
                timeframes=[Timeframe.M1],
            ),
            required_features=("orderflow_imbalance",),
        )
        config.upsert_strategy(definition)

        strategy = FailingStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=definition,
        )
        processor, _, state = _make_registered_processor(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            strategy=strategy,
        )

        batch = await processor.process_event(
            event_name="analytics.orderflow.updated",
            payload={
                "symbol": "BTCUSDT",
                "orderflow_imbalance": 0.72,
            },
        )

        assert not batch.accepted
        assert batch.evaluations
        assert not batch.evaluations[0].passed
        assert batch.evaluations[0].metadata["error_type"] == "StrategyEvaluationError"
        assert "no_passed_strategy_signals" in batch.reasons
        assert state.metrics.errors_total == 1
        assert mock_event_bus.topic_emitted("signal.rejected")

    @pytest.mark.asyncio()
    async def test_process_event_confluence_rejected_emits_rejected_batch(
        self,
        mock_event_bus,
        mock_scheduler,
        make_definition,
    ) -> None:
        config = _processor_config(
            confluence_enabled=True,
            min_agreement_count=2,
            min_confidence=0.5,
            min_score=0.0,
        )
        definition = make_definition(
            name="dummy_strategy",
            category=StrategyCategory.ORDERFLOW,
            runtime=make_runtime_config(
                symbols=["BTCUSDT"],
                timeframes=[Timeframe.M1],
            ),
            required_features=("orderflow_imbalance",),
        )
        config.upsert_strategy(definition)

        strategy = DummyStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=definition,
            confidence=0.9,
            score=0.85,
        )
        processor, _, _ = _make_registered_processor(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            strategy=strategy,
        )

        batch = await processor.process_event(
            event_name="analytics.orderflow.updated",
            payload={
                "symbol": "BTCUSDT",
                "orderflow_imbalance": 0.72,
                "liquidity_score": 0.85,
                "entry_price": 100.0,
                "stop_loss": 99.0,
                "market_type": "usdm_futures",
            },
        )

        assert not batch.accepted
        assert "confluence_rejected" in batch.reasons
        assert mock_event_bus.topic_emitted("signal.rejected")
        assert not mock_event_bus.topic_emitted("signal.generated")

    @pytest.mark.asyncio()
    async def test_evaluate_strategies_mixes_passed_and_failed_results(
        self,
        mock_event_bus,
        mock_scheduler,
        make_definition,
        make_context,
        make_feature,
    ) -> None:
        from conftest import FailingStrategy

        config = _processor_config()
        ok_definition = make_definition(
            name="ok_strategy",
            category=StrategyCategory.ORDERFLOW,
            runtime=make_runtime_config(symbols=["BTCUSDT"], timeframes=[Timeframe.M1]),
            required_features=("orderflow_imbalance",),
        )
        bad_definition = make_definition(
            name="bad_strategy",
            category=StrategyCategory.ORDERFLOW,
            runtime=make_runtime_config(symbols=["BTCUSDT"], timeframes=[Timeframe.M1]),
            required_features=("orderflow_imbalance",),
        )
        config.upsert_strategy(ok_definition)
        config.upsert_strategy(bad_definition)

        ok = DummyStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=ok_definition,
        )
        bad = FailingStrategy(
            config=config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            definition=bad_definition,
        )

        processor = SignalProcessor(
            config=config,
            registry=StrategyRegistry(config=config),
            state=StrategyRuntimeState(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )
        context = _context_with_market_features(make_context, make_feature)

        evaluations = await processor.evaluate_strategies(
            strategies=[ok, bad],
            context=context,
        )

        assert len(evaluations) == 2
        assert evaluations[0].passed
        assert not evaluations[1].passed
        assert evaluations[1].metadata["error_type"] == "StrategyEvaluationError"

    def test_to_risk_payload_contains_strategy_boundary_fields(
        self,
        strategy_config,
        mock_event_bus,
        mock_scheduler,
        risk_ready_strategy_signal,
        strategy_context,
    ) -> None:
        processor = SignalProcessor(
            config=strategy_config,
            registry=StrategyRegistry(config=strategy_config),
            state=StrategyRuntimeState(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        risk_ready_strategy_signal.metadata.update(
            {
                "exchange": "binance",
                "market_type": "usdm_futures",
                "entry_intent": "limit",
                "order_intent": "open_position",
                "requested_leverage": 3,
                "execution_cost": {
                    "spread_cost": 0.1,
                    "slippage_cost": 0.2,
                    "fee_cost": 0.03,
                    "quality": "good",
                },
            }
        )

        payload = processor.to_risk_payload(
            signal=risk_ready_strategy_signal,
            context=strategy_context,
        )

        payload.validate()

        assert isinstance(payload, RiskReadySignalPayload)
        assert payload.signal_id == risk_ready_strategy_signal.signal_id
        assert payload.symbol == risk_ready_strategy_signal.symbol
        assert payload.side is risk_ready_strategy_signal.side
        assert payload.market_type is StrategyMarketType.USDM_FUTURES
        assert payload.order_intent is StrategyOrderIntent.OPEN_POSITION
        assert payload.entry_price == risk_ready_strategy_signal.primary_entry_price
        assert payload.stop_loss == risk_ready_strategy_signal.primary_stop_loss
        assert payload.requested_leverage == 3
        assert payload.execution_cost is not None
        assert payload.execution_cost.quality is StrategyExecutionQuality.GOOD
        assert payload.metadata["processor"] == processor.component_name

    def test_to_risk_payload_requires_risk_ready_signal(
        self,
        strategy_config,
        mock_event_bus,
        mock_scheduler,
        strategy_signal,
        strategy_context,
    ) -> None:
        processor = SignalProcessor(
            config=strategy_config,
            registry=StrategyRegistry(config=strategy_config),
            state=StrategyRuntimeState(),
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        with pytest.raises((BuilderError, SignalRoutingError)):
            processor.to_risk_payload(
                signal=strategy_signal,
                context=strategy_context,
            )

    def test_record_final_signals_marks_pending_and_updates_state(
        self,
        strategy_config,
        mock_event_bus,
        mock_scheduler,
        risk_ready_strategy_signal,
    ) -> None:
        state = StrategyRuntimeState()
        processor = SignalProcessor(
            config=strategy_config,
            registry=StrategyRegistry(config=strategy_config),
            state=state,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        processor._record_final_signals([risk_ready_strategy_signal])

        assert risk_ready_strategy_signal.status is SignalStatus.PENDING
        assert state.get_signal_by_id(risk_ready_strategy_signal.signal_id) is risk_ready_strategy_signal
        assert state.metrics.signals_generated == 1

    @pytest.mark.asyncio()
    async def test_emit_rejected_batch_without_event_bus_is_safe(
        self,
        strategy_config,
    ) -> None:
        processor = SignalProcessor(
            config=strategy_config,
            registry=StrategyRegistry(config=strategy_config),
            state=StrategyRuntimeState(),
            event_bus=None,
        )
        batch = ProcessedSignalBatch(
            symbol="BTCUSDT",
            timestamp=utcnow(),
            reasons=["unit"],
        )

        await processor._emit_rejected_batch(batch, reason="unit_reject")

        assert not batch.accepted