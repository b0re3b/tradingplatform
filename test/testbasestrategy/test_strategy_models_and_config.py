# tests/strategy/test_strategy_models_and_config.py

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from strategy.config import (
    BuilderConfig,
    ConfidenceConfig,
    ConflictConfig,
    ConfluenceConfig,
    FeatureFreshnessConfig,
    FilterConfig,
    PortfolioCoordinatorConfig,
    PresetConfig,
    RoutingConfig,
    StrategyConfig,
    StrategyDefinitionConfig,
    StrategyRuntimeConfig,
    VotingConfig,
    WeightingConfig,
)
from strategy.enums import (
    ConflictType,
    EntryType,
    ExitType,
    FeatureSource,
    FilterDecision,
    FreshnessStatus,
    MarketRegime,
    PresetMode,
    SetupType,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    Timeframe,
)
from strategy.exceptions import StrategyConfigError, ValidationError
from strategy.models import (
    ConflictRecord,
    EntryPlan,
    ExecutionPlanDraft,
    ExitPlan,
    FeatureSnapshot,
    FilterResult,
    InvalidationPlan,
    StrategyMetadata,
    StrategySignal,
    TargetPlan,
    TradeIdea,
    clamp,
    ensure_aware_utc,
    utcnow,
)


# =============================================================================
# Enum behavior
# =============================================================================


class TestStrategyEnums:
    def test_signal_side_is_stable_string_enum(self) -> None:
        assert str(SignalSide.LONG) == "long"
        assert SignalSide.LONG.value == "long"
        assert "long" in SignalSide.values()
        assert SignalSide.has_value("short")
        assert not SignalSide.has_value("invalid")

    @pytest.mark.parametrize(
        ("side", "expected_directional", "expected_sign"),
        [
            (SignalSide.LONG, True, 1),
            (SignalSide.SHORT, True, -1),
            (SignalSide.FLAT, False, 0),
            (SignalSide.UNKNOWN, False, 0),
        ],
    )
    def test_signal_side_directional_flags(
        self,
        side: SignalSide,
        expected_directional: bool,
        expected_sign: int,
    ) -> None:
        assert side.is_directional is expected_directional
        assert side.sign == expected_sign

    def test_signal_side_safe_parse(self) -> None:
        assert SignalSide.safe_parse(SignalSide.LONG, SignalSide.UNKNOWN) is SignalSide.LONG
        assert SignalSide.safe_parse("long", SignalSide.UNKNOWN) is SignalSide.LONG
        assert SignalSide.safe_parse("bad-side", SignalSide.UNKNOWN) is SignalSide.UNKNOWN
        assert SignalSide.safe_parse(None, SignalSide.UNKNOWN) is SignalSide.UNKNOWN

    @pytest.mark.parametrize(
        "status",
        [
            SignalStatus.NEW,
            SignalStatus.PENDING,
            SignalStatus.CONFIRMED,
        ],
    )
    def test_signal_status_active(self, status: SignalStatus) -> None:
        assert status.is_active
        assert not status.is_terminal

    @pytest.mark.parametrize(
        "status",
        [
            SignalStatus.REJECTED,
            SignalStatus.CANCELLED,
            SignalStatus.EXPIRED,
            SignalStatus.EXECUTED,
            SignalStatus.FAILED,
        ],
    )
    def test_signal_status_terminal(self, status: SignalStatus) -> None:
        assert status.is_terminal
        assert not status.is_active


# =============================================================================
# Config validation
# =============================================================================


class TestStrategyRuntimeConfig:
    def test_valid_runtime_config(self) -> None:
        config = StrategyRuntimeConfig(
            enabled=True,
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframes=[Timeframe.M1, Timeframe.M5],
            allowed_regimes=[MarketRegime.UNKNOWN],
            cooldown_seconds=0,
            max_signal_age_seconds=30,
            min_confidence=0.5,
            min_score=0.0,
        )

        config.validate()

        assert config.allows_symbol("BTCUSDT")
        assert not config.allows_symbol("SOLUSDT")
        assert config.allows_timeframe(Timeframe.M1)
        assert not config.allows_timeframe(Timeframe.H1)
        assert config.allows_regime(MarketRegime.TRENDING_UP)

    def test_empty_symbols_means_all_symbols_allowed(self) -> None:
        config = StrategyRuntimeConfig(symbols=[])
        config.validate()

        assert config.allows_symbol("BTCUSDT")
        assert config.allows_symbol("ANYTHING")

    @pytest.mark.parametrize(
        "config",
        [
            StrategyRuntimeConfig(min_confidence=-0.01),
            StrategyRuntimeConfig(min_confidence=1.01),
            StrategyRuntimeConfig(min_score=-0.01),
            StrategyRuntimeConfig(cooldown_seconds=-1),
            StrategyRuntimeConfig(max_signal_age_seconds=0),
            StrategyRuntimeConfig(symbols=["BTCUSDT", " "]),
            StrategyRuntimeConfig(timeframes=[]),
            StrategyRuntimeConfig(allowed_regimes=[]),
        ],
    )
    def test_invalid_runtime_config_raises(self, config: StrategyRuntimeConfig) -> None:
        with pytest.raises(StrategyConfigError):
            config.validate()

    def test_allowed_regimes_without_unknown_are_strict(self) -> None:
        config = StrategyRuntimeConfig(
            allowed_regimes=[MarketRegime.TRENDING_UP],
        )
        config.validate()

        assert config.allows_regime(MarketRegime.TRENDING_UP)
        assert not config.allows_regime(MarketRegime.TRENDING_DOWN)


class TestStrategyDefinitionConfig:
    def test_valid_definition_config(self) -> None:
        definition = StrategyDefinitionConfig(
            name="cvd_divergence",
            category=StrategyCategory.ORDERFLOW,
            runtime=StrategyRuntimeConfig(),
            required_features={"cvd_delta", "orderflow_imbalance"},
            weight=1.2,
            priority=10,
            tags=["orderflow", "futures"],
        )

        definition.validate()

        assert definition.name == "cvd_divergence"
        assert definition.category is StrategyCategory.ORDERFLOW

    @pytest.mark.parametrize(
        "definition",
        [
            StrategyDefinitionConfig(name="", category=StrategyCategory.ORDERFLOW),
            StrategyDefinitionConfig(name="x", category=StrategyCategory.ORDERFLOW, weight=-0.1),
            StrategyDefinitionConfig(name="x", category=StrategyCategory.ORDERFLOW, priority=-1),
            StrategyDefinitionConfig(
                name="x",
                category=StrategyCategory.ORDERFLOW,
                required_features={"valid", " "},
            ),
            StrategyDefinitionConfig(
                name="x",
                category=StrategyCategory.ORDERFLOW,
                tags=["valid", ""],
            ),
            StrategyDefinitionConfig(
                name="x",
                category=StrategyCategory.ORDERFLOW,
                runtime=StrategyRuntimeConfig(min_confidence=2.0),
            ),
        ],
    )
    def test_invalid_definition_config_raises(
        self,
        definition: StrategyDefinitionConfig,
    ) -> None:
        with pytest.raises(StrategyConfigError):
            definition.validate()


class TestProcessorSubConfigs:
    @pytest.mark.parametrize(
        "config",
        [
            RoutingConfig(stale_feature_threshold_seconds=0),
            RoutingConfig(event_to_categories={"": [StrategyCategory.ORDERFLOW]}),
            RoutingConfig(event_to_categories={"analytics.orderflow.updated": []}),
        ],
    )
    def test_invalid_routing_config_raises(self, config: RoutingConfig) -> None:
        with pytest.raises(StrategyConfigError):
            config.validate()

    def test_routing_categories_for_event(self) -> None:
        config = RoutingConfig(
            event_to_categories={
                "analytics.orderflow.updated": [StrategyCategory.ORDERFLOW],
            }
        )
        config.validate()

        assert config.categories_for_event("analytics.orderflow.updated") == [
            StrategyCategory.ORDERFLOW
        ]
        assert config.categories_for_event("analytics.unknown") == []

    @pytest.mark.parametrize(
        "config",
        [
            ConfluenceConfig(min_agreement_count=0),
            ConfluenceConfig(min_confidence=-0.01),
            ConfluenceConfig(min_confidence=1.01),
            ConfluenceConfig(min_score=-0.01),
            ConfluenceConfig(conflict_penalty=-0.01),
            ConfluenceConfig(confirmation_bonus=-0.01),
            ConfluenceConfig(max_strategies_per_side=0),
        ],
    )
    def test_invalid_confluence_config_raises(self, config: ConfluenceConfig) -> None:
        with pytest.raises(StrategyConfigError):
            config.validate()

    @pytest.mark.parametrize(
        "config",
        [
            ConfidenceConfig(very_low_threshold=-0.1),
            ConfidenceConfig(high_threshold=1.1),
            ConfidenceConfig(
                very_low_threshold=0.4,
                low_threshold=0.3,
                medium_threshold=0.75,
                high_threshold=0.9,
            ),
        ],
    )
    def test_invalid_confidence_config_raises(self, config: ConfidenceConfig) -> None:
        with pytest.raises(StrategyConfigError):
            config.validate()

    def test_confidence_config_grade_bounds(self) -> None:
        config = ConfidenceConfig(
            very_low_threshold=0.2,
            low_threshold=0.4,
            medium_threshold=0.7,
            high_threshold=0.9,
        )
        config.validate()

        assert config.grade_bounds() == (0.2, 0.4, 0.7, 0.9)

    @pytest.mark.parametrize(
        "config",
        [
            WeightingConfig(category_weights={StrategyCategory.ORDERFLOW: -0.1}),
            WeightingConfig(regime_adjustments={MarketRegime.TRENDING_UP: -0.1}),
        ],
    )
    def test_invalid_weighting_config_raises(self, config: WeightingConfig) -> None:
        with pytest.raises(StrategyConfigError):
            config.validate()

    @pytest.mark.parametrize(
        "config",
        [
            VotingConfig(min_confirmations=-1),
            VotingConfig(min_total_votes=0),
            VotingConfig(min_confirmations=2, min_total_votes=1),
        ],
    )
    def test_invalid_voting_config_raises(self, config: VotingConfig) -> None:
        with pytest.raises(StrategyConfigError):
            config.validate()

    @pytest.mark.parametrize(
        "config",
        [
            ConflictConfig(max_total_penalty=-0.01),
            ConflictConfig(max_total_penalty=10.01),
        ],
    )
    def test_invalid_conflict_config_raises(self, config: ConflictConfig) -> None:
        with pytest.raises(StrategyConfigError):
            config.validate()

    @pytest.mark.parametrize(
        "config",
        [
            FilterConfig(max_spread_bps=-0.01),
            FilterConfig(min_liquidity_score=-0.01),
            FilterConfig(min_liquidity_score=1.01),
            FilterConfig(max_volatility_zscore=-0.01),
            FilterConfig(min_funding_alignment=-1.01),
            FilterConfig(min_funding_alignment=1.01),
        ],
    )
    def test_invalid_filter_config_raises(self, config: FilterConfig) -> None:
        with pytest.raises(StrategyConfigError):
            config.validate()

    @pytest.mark.parametrize(
        "config",
        [
            BuilderConfig(default_rr_ratio=0),
            BuilderConfig(default_partial_tp_levels=[]),
            BuilderConfig(default_partial_tp_levels=[0.5, 0.0]),
            BuilderConfig(default_partial_tp_levels=[0.7, 0.6]),
        ],
    )
    def test_invalid_builder_config_raises(self, config: BuilderConfig) -> None:
        with pytest.raises(StrategyConfigError):
            config.validate()

    @pytest.mark.parametrize(
        "config",
        [
            PortfolioCoordinatorConfig(max_signals_per_symbol=0),
            PortfolioCoordinatorConfig(symbol_cooldown_seconds=-1),
            PortfolioCoordinatorConfig(side_cooldown_seconds=-1),
            PortfolioCoordinatorConfig(repeated_signal_suppression_seconds=-1),
            PortfolioCoordinatorConfig(volatility_throttle_threshold=-0.1),
            PortfolioCoordinatorConfig(high_volatility_max_signals_per_symbol=0),
            PortfolioCoordinatorConfig(
                max_signals_per_category={StrategyCategory.ORDERFLOW: 0}
            ),
            PortfolioCoordinatorConfig(priority_overrides={"": 10}),
            PortfolioCoordinatorConfig(priority_overrides={"cvd": -1}),
            PortfolioCoordinatorConfig(exposure_bucket_limits={"": 1}),
            PortfolioCoordinatorConfig(exposure_bucket_limits={"directional": 0}),
        ],
    )
    def test_invalid_portfolio_config_raises(
        self,
        config: PortfolioCoordinatorConfig,
    ) -> None:
        with pytest.raises(StrategyConfigError):
            config.validate()

    @pytest.mark.parametrize(
        "config",
        [
            FeatureFreshnessConfig(default_ttl_seconds=0),
            FeatureFreshnessConfig(per_feature_ttl_seconds={"": 30}),
            FeatureFreshnessConfig(per_feature_ttl_seconds={"cvd_delta": 0}),
        ],
    )
    def test_invalid_freshness_config_raises(
        self,
        config: FeatureFreshnessConfig,
    ) -> None:
        with pytest.raises(StrategyConfigError):
            config.validate()

    def test_freshness_config_get_ttl(self) -> None:
        config = FeatureFreshnessConfig(
            default_ttl_seconds=30,
            per_feature_ttl_seconds={"cvd_delta": 10},
        )
        config.validate()

        assert config.get_ttl("cvd_delta") == 10
        assert config.get_ttl("unknown") == 30

    def test_preset_config_allowed_strategy_names(self) -> None:
        unrestricted = PresetConfig()
        unrestricted.validate()

        restricted = PresetConfig(
            mode=PresetMode.INTRADAY,
            enabled_strategy_names=["cvd_divergence"],
        )
        restricted.validate()

        assert unrestricted.is_strategy_allowed("anything")
        assert restricted.is_strategy_allowed("cvd_divergence")
        assert not restricted.is_strategy_allowed("orderflow_reversal")

    def test_invalid_preset_config_raises(self) -> None:
        config = PresetConfig(enabled_strategy_names=["cvd_divergence", " "])

        with pytest.raises(StrategyConfigError):
            config.validate()


class TestStrategyConfig:
    def test_strategy_config_accessors_and_mutators(
        self,
        make_definition,
    ) -> None:
        definition = make_definition(
            name="cvd_divergence",
            category=StrategyCategory.ORDERFLOW,
            required_features=("cvd_delta", "orderflow_imbalance"),
            weight=1.25,
            priority=7,
        )
        config = StrategyConfig(
            preset=PresetConfig(enabled_strategy_names=["cvd_divergence"]),
            strategies={"cvd_divergence": definition},
        )

        config.validate()

        assert config.get_strategy("cvd_divergence") is definition
        assert config.require_strategy("cvd_divergence") is definition
        assert config.is_strategy_configured("cvd_divergence")
        assert config.is_strategy_enabled("cvd_divergence")
        assert config.is_strategy_allowed_by_preset("cvd_divergence")
        assert config.get_strategy_runtime("cvd_divergence") is definition.runtime
        assert config.get_strategy_weight("cvd_divergence") == 1.25
        assert config.get_strategy_priority("cvd_divergence") == 7
        assert config.get_strategy_required_features("cvd_divergence") == {
            "cvd_delta",
            "orderflow_imbalance",
        }

        assert config.get_strategy("missing") is None
        assert config.get_strategy_runtime("missing") is config.runtime
        assert config.get_strategy_weight("missing", default=9.9) == 9.9
        assert config.get_strategy_priority("missing", default=99) == 99
        assert config.get_strategy_required_features("missing") == set()

        assert config.enabled_strategy_names() == ["cvd_divergence"]
        assert config.strategies_by_category(StrategyCategory.ORDERFLOW) == [definition]

    def test_strategy_config_require_missing_strategy_raises(self) -> None:
        config = StrategyConfig()

        with pytest.raises(StrategyConfigError):
            config.require_strategy("missing")

    def test_strategy_config_rejects_key_name_mismatch(
        self,
        make_definition,
    ) -> None:
        definition = make_definition(name="embedded_name")

        config = StrategyConfig(strategies={"different_key": definition})

        with pytest.raises(StrategyConfigError):
            config.validate()

    def test_strategy_config_rejects_empty_strategy_key(
        self,
        make_definition,
    ) -> None:
        definition = make_definition(name="valid")

        config = StrategyConfig(strategies={"": definition})

        with pytest.raises(StrategyConfigError):
            config.validate()

    def test_strategy_config_add_upsert_remove_strategy(
        self,
        make_definition,
    ) -> None:
        config = StrategyConfig()

        first = make_definition(name="first")
        second = make_definition(name="second")

        config.add_strategy(first)
        assert config.get_strategy("first") is first

        with pytest.raises(StrategyConfigError):
            config.add_strategy(first)

        config.upsert_strategy(second)
        assert config.get_strategy("second") is second

        replacement = make_definition(name="second", priority=1)
        config.upsert_strategy(replacement)
        assert config.get_strategy("second") is replacement
        assert config.get_strategy_priority("second") == 1

        removed = config.remove_strategy("second")
        assert removed is replacement
        assert config.get_strategy("second") is None

    def test_strategies_by_category_sorts_by_priority(
        self,
        make_definition,
    ) -> None:
        slow = make_definition(
            name="slow",
            category=StrategyCategory.ORDERFLOW,
            priority=50,
        )
        fast = make_definition(
            name="fast",
            category=StrategyCategory.ORDERFLOW,
            priority=10,
        )
        other = make_definition(
            name="other",
            category=StrategyCategory.OPEN_INTEREST,
            priority=1,
        )

        config = StrategyConfig(
            strategies={
                slow.name: slow,
                fast.name: fast,
                other.name: other,
            }
        )
        config.validate()

        assert config.strategies_by_category(StrategyCategory.ORDERFLOW) == [fast, slow]


# =============================================================================
# Model helpers and validation
# =============================================================================


class TestTimeHelpers:
    def test_utcnow_is_timezone_aware_utc(self) -> None:
        value = utcnow()

        assert value.tzinfo is not None
        assert value.utcoffset() == timedelta(0)

    def test_ensure_aware_utc_treats_naive_datetime_as_utc(self) -> None:
        naive = datetime(2026, 5, 20, 12, 0, 0)

        result = ensure_aware_utc(naive)

        assert result.tzinfo is timezone.utc
        assert result.hour == 12

    def test_ensure_aware_utc_converts_non_utc_datetime(self) -> None:
        plus_two = timezone(timedelta(hours=2))
        value = datetime(2026, 5, 20, 14, 0, 0, tzinfo=plus_two)

        result = ensure_aware_utc(value)

        assert result.tzinfo is timezone.utc
        assert result.hour == 12

    def test_clamp(self) -> None:
        assert clamp(-1.0, 0.0, 1.0) == 0.0
        assert clamp(0.5, 0.0, 1.0) == 0.5
        assert clamp(2.0, 0.0, 1.0) == 1.0


class TestFeatureSnapshot:
    def test_valid_feature_snapshot(self) -> None:
        feature = FeatureSnapshot(
            name="cvd_delta",
            value=123.0,
            source=FeatureSource.ORDERFLOW,
            symbol="BTCUSDT",
            timestamp=datetime(2026, 5, 20, 12, 0, 0),
            confidence=0.8,
            normalized_value=0.5,
            freshness_seconds=60,
        )

        feature.validate()

        assert feature.timestamp.tzinfo is timezone.utc
        assert feature.age_seconds(feature.timestamp) == 0.0
        assert not feature.is_stale(feature.timestamp)
        assert not feature.is_expired(feature.timestamp)

    @pytest.mark.parametrize(
        "feature",
        [
            FeatureSnapshot(
                name="",
                value=1,
                source=FeatureSource.ORDERFLOW,
                symbol="BTCUSDT",
                timestamp=utcnow(),
            ),
            FeatureSnapshot(
                name="x",
                value=1,
                source=FeatureSource.ORDERFLOW,
                symbol=" ",
                timestamp=utcnow(),
            ),
            FeatureSnapshot(
                name="x",
                value=1,
                source=FeatureSource.ORDERFLOW,
                symbol="BTCUSDT",
                timestamp=utcnow(),
                confidence=-0.1,
            ),
            FeatureSnapshot(
                name="x",
                value=1,
                source=FeatureSource.ORDERFLOW,
                symbol="BTCUSDT",
                timestamp=utcnow(),
                confidence=1.1,
            ),
            FeatureSnapshot(
                name="x",
                value=1,
                source=FeatureSource.ORDERFLOW,
                symbol="BTCUSDT",
                timestamp=utcnow(),
                normalized_value=-1.1,
            ),
            FeatureSnapshot(
                name="x",
                value=1,
                source=FeatureSource.ORDERFLOW,
                symbol="BTCUSDT",
                timestamp=utcnow(),
                freshness_seconds=0,
            ),
        ],
    )
    def test_invalid_feature_snapshot_raises(self, feature: FeatureSnapshot) -> None:
        with pytest.raises(ValidationError):
            feature.validate()

    @pytest.mark.parametrize(
        ("age_seconds", "expected"),
        [
            (10, FreshnessStatus.FRESH),
            (50, FreshnessStatus.AGING),
            (90, FreshnessStatus.STALE),
            (150, FreshnessStatus.EXPIRED),
        ],
    )
    def test_freshness_status_boundaries(
        self,
        age_seconds: int,
        expected: FreshnessStatus,
    ) -> None:
        now = utcnow()
        feature = FeatureSnapshot(
            name="cvd_delta",
            value=1.0,
            source=FeatureSource.ORDERFLOW,
            symbol="BTCUSDT",
            timestamp=now - timedelta(seconds=age_seconds),
            confidence=1.0,
            freshness_seconds=60,
        )
        feature.validate()

        assert feature.freshness_status(now) is expected

    def test_feature_without_ttl_is_always_fresh(self) -> None:
        now = utcnow()
        feature = FeatureSnapshot(
            name="cvd_delta",
            value=1.0,
            source=FeatureSource.ORDERFLOW,
            symbol="BTCUSDT",
            timestamp=now - timedelta(days=365),
            freshness_seconds=None,
        )
        feature.validate()

        assert feature.freshness_status(now) is FreshnessStatus.FRESH


class TestTradePlanModels:
    @pytest.mark.parametrize(
        "entry",
        [
            EntryPlan(entry_type=EntryType.LIMIT, price=0),
            EntryPlan(entry_type=EntryType.LIMIT, timeout_seconds=0),
            EntryPlan(entry_type=EntryType.LIMIT, max_slippage_bps=-0.1),
        ],
    )
    def test_invalid_entry_plan_raises(self, entry: EntryPlan) -> None:
        with pytest.raises(ValidationError):
            entry.validate()

    @pytest.mark.parametrize(
        "target",
        [
            TargetPlan(price=0),
            TargetPlan(price=100, size_fraction=0),
            TargetPlan(price=100, size_fraction=1.1),
            TargetPlan(price=100, rr=0),
        ],
    )
    def test_invalid_target_plan_raises(self, target: TargetPlan) -> None:
        with pytest.raises(ValidationError):
            target.validate()

    @pytest.mark.parametrize(
        "invalidation",
        [
            InvalidationPlan(price=0),
            InvalidationPlan(timeout_seconds=0),
        ],
    )
    def test_invalid_invalidation_plan_raises(
        self,
        invalidation: InvalidationPlan,
    ) -> None:
        with pytest.raises(ValidationError):
            invalidation.validate()

    @pytest.mark.parametrize(
        "exit_plan",
        [
            ExitPlan(stop_loss=0),
            ExitPlan(trailing_distance=0),
            ExitPlan(max_holding_seconds=0),
            ExitPlan(take_profit_levels=[TargetPlan(price=0)]),
        ],
    )
    def test_invalid_exit_plan_raises(self, exit_plan: ExitPlan) -> None:
        with pytest.raises(ValidationError):
            exit_plan.validate()

    def test_valid_execution_plan_draft(self) -> None:
        draft = ExecutionPlanDraft(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            entry=EntryPlan(entry_type=EntryType.LIMIT, price=100.0),
            exit=ExitPlan(
                exit_types=[ExitType.STOP_LOSS, ExitType.TAKE_PROFIT],
                stop_loss=99.0,
                take_profit_levels=[TargetPlan(price=102.0)],
            ),
            invalidation=InvalidationPlan(price=99.0),
            leverage=2.0,
            expected_holding_seconds=300,
        )

        draft.validate()

        assert draft.symbol == "BTCUSDT"
        assert draft.side is SignalSide.LONG

    @pytest.mark.parametrize(
        "draft",
        [
            ExecutionPlanDraft(
                symbol="",
                side=SignalSide.LONG,
                entry=EntryPlan(entry_type=EntryType.MARKET),
                exit=ExitPlan(),
                invalidation=InvalidationPlan(),
            ),
            ExecutionPlanDraft(
                symbol="BTCUSDT",
                side=SignalSide.FLAT,
                entry=EntryPlan(entry_type=EntryType.MARKET),
                exit=ExitPlan(),
                invalidation=InvalidationPlan(),
            ),
            ExecutionPlanDraft(
                symbol="BTCUSDT",
                side=SignalSide.LONG,
                entry=EntryPlan(entry_type=EntryType.MARKET),
                exit=ExitPlan(),
                invalidation=InvalidationPlan(),
                leverage=0,
            ),
            ExecutionPlanDraft(
                symbol="BTCUSDT",
                side=SignalSide.LONG,
                entry=EntryPlan(entry_type=EntryType.MARKET),
                exit=ExitPlan(),
                invalidation=InvalidationPlan(),
                expected_holding_seconds=0,
            ),
            ExecutionPlanDraft(
                symbol="BTCUSDT",
                side=SignalSide.LONG,
                entry=EntryPlan(entry_type=EntryType.LIMIT, price=0),
                exit=ExitPlan(),
                invalidation=InvalidationPlan(),
            ),
        ],
    )
    def test_invalid_execution_plan_draft_raises(
        self,
        draft: ExecutionPlanDraft,
    ) -> None:
        with pytest.raises(ValidationError):
            draft.validate()


class TestFilterAndConflictModels:
    def test_filter_result_passed_and_blocked_flags(self) -> None:
        passed = FilterResult(name="confidence", decision=FilterDecision.PASS)
        warned = FilterResult(name="spread", decision=FilterDecision.WARN)
        blocked = FilterResult(name="liquidity", decision=FilterDecision.BLOCK)

        passed.validate()
        warned.validate()
        blocked.validate()

        assert passed.passed
        assert warned.passed
        assert not blocked.passed
        assert blocked.blocked

    def test_filter_result_empty_name_raises(self) -> None:
        result = FilterResult(name="", decision=FilterDecision.PASS)

        with pytest.raises(ValidationError):
            result.validate()

    @pytest.mark.parametrize(
        "conflict",
        [
            ConflictRecord(
                conflict_type=ConflictType.SIDE_CONFLICT,
                source="",
                message="conflict",
            ),
            ConflictRecord(
                conflict_type=ConflictType.SIDE_CONFLICT,
                source="strategy",
                message="",
            ),
            ConflictRecord(
                conflict_type=ConflictType.SIDE_CONFLICT,
                source="strategy",
                message="conflict",
                penalty=-0.1,
            ),
        ],
    )
    def test_invalid_conflict_record_raises(
        self,
        conflict: ConflictRecord,
    ) -> None:
        with pytest.raises(ValidationError):
            conflict.validate()


class TestStrategyMetadata:
    def test_valid_strategy_metadata(self) -> None:
        metadata = StrategyMetadata(
            strategy_name="cvd_divergence",
            category=StrategyCategory.ORDERFLOW,
            timeframe=Timeframe.M1,
            tags=["futures"],
            version="1.2.3",
        )

        metadata.validate()

        assert metadata.strategy_name == "cvd_divergence"

    @pytest.mark.parametrize(
        "metadata",
        [
            StrategyMetadata(
                strategy_name="",
                category=StrategyCategory.ORDERFLOW,
                timeframe=Timeframe.M1,
            ),
            StrategyMetadata(
                strategy_name="x",
                category=StrategyCategory.ORDERFLOW,
                timeframe=Timeframe.M1,
                version="",
            ),
        ],
    )
    def test_invalid_strategy_metadata_raises(
        self,
        metadata: StrategyMetadata,
    ) -> None:
        with pytest.raises(ValidationError):
            metadata.validate()


class TestStrategySignal:
    def test_valid_strategy_signal_defaults_and_to_dict(self) -> None:
        raw_ts = datetime(2026, 5, 20, 12, 0, 0)
        signal = StrategySignal(
            symbol=" BTCUSDT ",
            side=SignalSide.LONG,
            strategy_name=" cvd_divergence ",
            category=StrategyCategory.ORDERFLOW,
            timeframe=Timeframe.M1,
            setup_type=SetupType.CVD_DIVERGENCE,
            timestamp=raw_ts,
            confidence=0.82,
            score=1.5,
        )

        signal.validate()

        assert signal.symbol == "BTCUSDT"
        assert signal.strategy_name == "cvd_divergence"
        assert signal.timestamp.tzinfo is timezone.utc
        assert signal.metadata["signal_id"] == signal.signal_id
        assert signal.is_long
        assert signal.is_directional
        assert signal.is_active

        data = signal.to_dict()

        assert data["signal_id"] == signal.signal_id
        assert data["symbol"] == "BTCUSDT"
        assert data["side"] == "long"
        assert data["category"] == "orderflow"
        assert data["timeframe"] == "1m"

    def test_signal_post_init_clamps_confidence_and_score(self) -> None:
        signal = StrategySignal(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            strategy_name="cvd_divergence",
            category=StrategyCategory.ORDERFLOW,
            timeframe=Timeframe.M1,
            setup_type=SetupType.CVD_DIVERGENCE,
            timestamp=utcnow(),
            confidence=5.0,
            score=-10.0,
        )

        signal.validate()

        assert signal.confidence == 1.0
        assert signal.score == 0.0

    @pytest.mark.parametrize(
        ("side", "expected_long", "expected_short", "expected_flat", "directional"),
        [
            (SignalSide.LONG, True, False, False, True),
            (SignalSide.SHORT, False, True, False, True),
            (SignalSide.FLAT, False, False, True, False),
            (SignalSide.UNKNOWN, False, False, False, False),
        ],
    )
    def test_signal_side_properties(
        self,
        side: SignalSide,
        expected_long: bool,
        expected_short: bool,
        expected_flat: bool,
        directional: bool,
    ) -> None:
        signal = StrategySignal(
            symbol="BTCUSDT",
            side=side,
            strategy_name="test_strategy",
            category=StrategyCategory.ORDERFLOW,
            timeframe=Timeframe.M1,
            setup_type=SetupType.CVD_DIVERGENCE,
            timestamp=utcnow(),
        )

        signal.validate()

        assert signal.is_long is expected_long
        assert signal.is_short is expected_short
        assert signal.is_flat is expected_flat
        assert signal.is_directional is directional

    @pytest.mark.parametrize(
        "signal",
        [
            StrategySignal(
                symbol="",
                side=SignalSide.LONG,
                strategy_name="strategy",
                category=StrategyCategory.ORDERFLOW,
                timeframe=Timeframe.M1,
                setup_type=SetupType.CVD_DIVERGENCE,
                timestamp=utcnow(),
            ),
            StrategySignal(
                symbol="BTCUSDT",
                side=SignalSide.LONG,
                strategy_name="",
                category=StrategyCategory.ORDERFLOW,
                timeframe=Timeframe.M1,
                setup_type=SetupType.CVD_DIVERGENCE,
                timestamp=utcnow(),
            ),
            StrategySignal(
                symbol="BTCUSDT",
                side=SignalSide.FLAT,
                strategy_name="strategy",
                category=StrategyCategory.ORDERFLOW,
                timeframe=Timeframe.M1,
                setup_type=SetupType.CVD_DIVERGENCE,
                timestamp=utcnow(),
                status=SignalStatus.PENDING,
            ),
            StrategySignal(
                symbol="BTCUSDT",
                side=SignalSide.UNKNOWN,
                strategy_name="strategy",
                category=StrategyCategory.ORDERFLOW,
                timeframe=Timeframe.M1,
                setup_type=SetupType.CVD_DIVERGENCE,
                timestamp=utcnow(),
                status=SignalStatus.CONFIRMED,
            ),
        ],
    )
    def test_invalid_strategy_signal_raises(self, signal: StrategySignal) -> None:
        with pytest.raises(ValidationError):
            signal.validate()

    def test_signal_adders_are_idempotent(self) -> None:
        signal = StrategySignal(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            strategy_name="cvd_divergence",
            category=StrategyCategory.ORDERFLOW,
            timeframe=Timeframe.M1,
            setup_type=SetupType.CVD_DIVERGENCE,
            timestamp=utcnow(),
        )

        signal.add_reason("r1")
        signal.add_reason("r1")
        signal.add_confirmation("c1")
        signal.add_confirmation("c1")
        signal.add_source_feature("cvd_delta")
        signal.add_source_feature("cvd_delta")

        assert signal.reasons == ["r1"]
        assert signal.confirmations == ["c1"]
        assert signal.source_features == ["cvd_delta"]

    def test_signal_status_mutators(self) -> None:
        signal = StrategySignal(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            strategy_name="cvd_divergence",
            category=StrategyCategory.ORDERFLOW,
            timeframe=Timeframe.M1,
            setup_type=SetupType.CVD_DIVERGENCE,
            timestamp=utcnow(),
        )

        signal.to_pending()
        assert signal.status is SignalStatus.PENDING

        signal.to_confirmed()
        assert signal.status is SignalStatus.CONFIRMED

        signal.to_rejected()
        assert signal.status is SignalStatus.REJECTED

        signal.to_cancelled()
        assert signal.status is SignalStatus.CANCELLED

        signal.to_expired()
        assert signal.status is SignalStatus.EXPIRED

        signal.to_executed()
        assert signal.status is SignalStatus.EXECUTED

        signal.to_failed()
        assert signal.status is SignalStatus.FAILED

    def test_signal_primary_prices_from_direct_plans(self) -> None:
        signal = StrategySignal(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            strategy_name="cvd_divergence",
            category=StrategyCategory.ORDERFLOW,
            timeframe=Timeframe.M1,
            setup_type=SetupType.CVD_DIVERGENCE,
            timestamp=utcnow(),
            entry_plan=EntryPlan(entry_type=EntryType.LIMIT, price=100.0),
            invalidation_plan=InvalidationPlan(price=99.0),
            exit_plan=ExitPlan(
                stop_loss=98.5,
                take_profit_levels=[TargetPlan(price=102.0)],
            ),
        )

        signal.validate()

        assert signal.primary_entry_price == 100.0
        assert signal.primary_stop_loss == 98.5
        assert signal.primary_take_profit == 102.0

    def test_signal_primary_prices_from_execution_plan_take_precedence(self) -> None:
        entry = EntryPlan(entry_type=EntryType.LIMIT, price=100.0)
        invalidation = InvalidationPlan(price=99.0)
        exit_plan = ExitPlan(
            stop_loss=98.0,
            take_profit_levels=[TargetPlan(price=104.0)],
        )
        execution_plan = ExecutionPlanDraft(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            entry=entry,
            exit=exit_plan,
            invalidation=invalidation,
            leverage=2.0,
        )

        signal = StrategySignal(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            strategy_name="cvd_divergence",
            category=StrategyCategory.ORDERFLOW,
            timeframe=Timeframe.M1,
            setup_type=SetupType.CVD_DIVERGENCE,
            timestamp=utcnow(),
            entry_plan=EntryPlan(entry_type=EntryType.MARKET, price=101.0),
            invalidation_plan=InvalidationPlan(price=97.0),
            exit_plan=ExitPlan(take_profit_levels=[TargetPlan(price=103.0)]),
            execution_plan=execution_plan,
        )

        signal.validate()

        assert signal.primary_entry_price == 100.0
        assert signal.primary_stop_loss == 98.0
        assert signal.primary_take_profit == 104.0

    def test_signal_rejects_invalid_nested_filter_or_conflict(self) -> None:
        signal = StrategySignal(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            strategy_name="cvd_divergence",
            category=StrategyCategory.ORDERFLOW,
            timeframe=Timeframe.M1,
            setup_type=SetupType.CVD_DIVERGENCE,
            timestamp=utcnow(),
        )
        signal.filter_results.append(
            FilterResult(name="", decision=FilterDecision.PASS)
        )

        with pytest.raises(ValidationError):
            signal.validate()

        signal.filter_results.clear()
        signal.conflicts.append(
            ConflictRecord(
                conflict_type=ConflictType.SIDE_CONFLICT,
                source="",
                message="bad",
            )
        )

        with pytest.raises(ValidationError):
            signal.validate()


class TestTradeIdea:
    def test_trade_idea_validation_and_expiration(
        self,
        risk_ready_strategy_signal,
    ) -> None:
        signal = risk_ready_strategy_signal
        assert signal.execution_plan is not None

        now = utcnow()
        idea = TradeIdea(
            signal=signal,
            execution_plan=signal.execution_plan,
            created_at=now,
            expires_at=now + timedelta(seconds=30),
        )

        idea.validate()

        assert not idea.is_expired(now)
        assert idea.is_expired(now + timedelta(seconds=31))

    def test_trade_idea_rejects_expiration_before_creation(
        self,
        risk_ready_strategy_signal,
    ) -> None:
        signal = risk_ready_strategy_signal
        assert signal.execution_plan is not None

        now = utcnow()
        idea = TradeIdea(
            signal=signal,
            execution_plan=signal.execution_plan,
            created_at=now,
            expires_at=now,
        )

        with pytest.raises(ValidationError):
            idea.validate()