# tests/strategy/strategies/liquidity/test_base_liquidity_strategy.py

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from strategy.enums import FilterDecision, SignalSide, SignalStatus
from strategy.models import StrategySignal
from strategy.strategies.liquidity.base_liquidity_strategy import BaseLiquidityStrategy


class DummyLiquidityStrategy(BaseLiquidityStrategy):
    """
    Minimal concrete strategy for testing BaseLiquidityStrategy behavior.

    We intentionally do not implement any real setup logic here because this
    file tests only the base liquidity-strategy contract:
    - snapshot extraction;
    - context/snapshot validation;
    - common filters;
    - freshness;
    - current-price resolution;
    - signal emission/cooldown.
    """

    @property
    def strategy_name(self) -> str:
        return "dummy_liquidity_strategy"

    def evaluate(self, context: Any) -> StrategySignal | None:
        return None


class AttrWrapper:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class ExplodingFeatureStore:
    def get_feature(self, key: str) -> Any:
        raise RuntimeError(f"feature backend failed for {key}")

    def get_feature_snapshot(self, key: str) -> Any:
        raise RuntimeError(f"feature snapshot backend failed for {key}")


class FeatureSnapshot:
    def __init__(self, stale: bool | Exception) -> None:
        self._stale = stale
        self.calls: list[datetime] = []

    def is_stale(self, timestamp: datetime) -> bool:
        self.calls.append(timestamp)

        if isinstance(self._stale, Exception):
            raise self._stale

        return self._stale


class ContextDouble:
    """
    Lightweight StrategyContext-compatible double.

    It intentionally supports:
    - direct liquidity domain access;
    - feature access through get_feature(...);
    - feature freshness access through get_feature_snapshot(...);
    - optional exploding backend behavior.

    This lets us test BaseLiquidityStrategy's resilience without depending on
    the full StrategyContext constructor shape.
    """

    def __init__(
        self,
        *,
        symbol: str = "BTCUSDT",
        timeframe: Any = "1m",
        timestamp: datetime | None = None,
        liquidity: Any = None,
        price: Any = None,
        regime: Any = None,
        portfolio: Any = None,
        features: dict[str, Any] | None = None,
        feature_snapshot: Any = None,
        metadata: Any = None,
        explode_features: bool = False,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.timestamp = timestamp or datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        self.liquidity = liquidity
        self.price = price
        self.regime = regime
        self.portfolio = portfolio
        self.metadata = metadata
        self._features = features or {}
        self._feature_snapshot = feature_snapshot
        self._exploding_store = ExplodingFeatureStore() if explode_features else None

    def get_feature(self, key: str) -> Any:
        if self._exploding_store is not None:
            return self._exploding_store.get_feature(key)

        if key not in self._features:
            raise KeyError(key)

        return self._features[key]

    def get_feature_snapshot(self, key: str) -> Any:
        if self._exploding_store is not None:
            return self._exploding_store.get_feature_snapshot(key)

        return self._feature_snapshot


@pytest.fixture
def strategy(strategy_config: Any) -> DummyLiquidityStrategy:
    """
    Expects a real StrategyConfig fixture from conftest.py.

    The fixture should return a StrategyConfig with:
    - runtime.enabled = True
    - runtime.min_confidence / min_score defined
    - runtime.emit_cooldown_seconds configurable
    - filters.enable_spread_filter
    - filters.max_spread_bps
    - filters.enable_liquidity_filter
    - filters.min_liquidity_score
    - freshness.get_ttl(...)
    """
    return DummyLiquidityStrategy(config=strategy_config)


def _patch_runtime(
    strategy: DummyLiquidityStrategy,
    *,
    enabled: bool = True,
    symbols: set[str] | None = None,
    timeframes: set[Any] | None = None,
    allowed_regimes: set[Any] | None = None,
    emit_cooldown_seconds: float = 0.0,
) -> None:
    strategy.config.runtime.enabled = enabled
    strategy.config.runtime.symbols = symbols or set()
    strategy.config.runtime.timeframes = timeframes or set()
    strategy.config.runtime.allowed_regimes = allowed_regimes
    strategy.config.runtime.emit_cooldown_seconds = emit_cooldown_seconds


def _patch_filters(
    strategy: DummyLiquidityStrategy,
    *,
    enable_spread_filter: bool = False,
    max_spread_bps: float = 10.0,
    enable_liquidity_filter: bool = False,
    min_liquidity_score: float = 0.0,
) -> None:
    strategy.config.filters.enable_spread_filter = enable_spread_filter
    strategy.config.filters.max_spread_bps = max_spread_bps
    strategy.config.filters.enable_liquidity_filter = enable_liquidity_filter
    strategy.config.filters.min_liquidity_score = min_liquidity_score


def _patch_freshness_ttl(strategy: DummyLiquidityStrategy, ttl_seconds: float | None) -> None:
    strategy.config.freshness.get_ttl = lambda feature_name: ttl_seconds


def _filter_by_name(results: list[Any], name: str) -> Any:
    matches = [item for item in results if item.name == name]
    assert len(matches) == 1, f"Expected exactly one filter named {name}, got {matches!r}"
    return matches[0]


class TestRequiredFeatures:
    def test_required_features_contains_liquidity_snapshot_when_no_strategy_override(
        self,
        strategy: DummyLiquidityStrategy,
    ) -> None:
        strategy.config.get_strategy = lambda name: None

        assert strategy.required_features() == {strategy.SNAPSHOT_FEATURE_NAME}

    def test_required_features_preserves_configured_features_and_forces_snapshot(
        self,
        strategy: DummyLiquidityStrategy,
    ) -> None:
        strategy.config.get_strategy = lambda name: SimpleNamespace(
            required_features={"orderflow.delta", "price.volatility"}
        )

        required = strategy.required_features()

        assert required == {
            "orderflow.delta",
            "price.volatility",
            strategy.SNAPSHOT_FEATURE_NAME,
        }


class TestSnapshotExtraction:
    def test_extract_snapshot_from_liquidity_mapping_direct_snapshot(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        snapshot = liquidity_snapshot_factory(symbol="BTCUSDT", timeframe="1m")
        context = ContextDouble(liquidity={"snapshot": snapshot})

        assert strategy._extract_snapshot(context) is snapshot

    def test_extract_snapshot_from_liquidity_mapping_nested_payload(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        snapshot = liquidity_snapshot_factory(symbol="BTCUSDT", timeframe="1m")
        context = ContextDouble(
            liquidity={
                "snapshot": {"payload": snapshot},
                "liquidity_map_snapshot": object(),
            }
        )

        assert strategy._extract_snapshot(context) is snapshot

    def test_extract_snapshot_from_liquidity_attr_wrapper_value(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        snapshot = liquidity_snapshot_factory(symbol="BTCUSDT", timeframe="1m")
        context = ContextDouble(
            liquidity=AttrWrapper(
                snapshot=None,
                liquidity_map_snapshot=AttrWrapper(value=snapshot),
            )
        )

        assert strategy._extract_snapshot(context) is snapshot

    def test_extract_snapshot_from_feature_after_domain_candidates_are_invalid(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        snapshot = liquidity_snapshot_factory(symbol="BTCUSDT", timeframe="1m")
        context = ContextDouble(
            liquidity={
                "snapshot": {"payload": object()},
                "liquidity_map_snapshot": None,
                "map_snapshot": {"data": "not-a-snapshot"},
                "last_snapshot": object(),
            },
            features={
                "liquidity.snapshot": {"data": snapshot},
            },
        )

        assert strategy._extract_snapshot(context) is snapshot

    def test_extract_snapshot_prefers_domain_snapshot_over_feature_snapshot(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        domain_snapshot = liquidity_snapshot_factory(symbol="BTCUSDT", timeframe="1m")
        feature_snapshot = liquidity_snapshot_factory(symbol="ETHUSDT", timeframe="1m")

        context = ContextDouble(
            liquidity={"snapshot": domain_snapshot},
            features={"liquidity_map_snapshot": feature_snapshot},
        )

        assert strategy._extract_snapshot(context) is domain_snapshot

    def test_extract_snapshot_survives_feature_backend_exceptions_and_returns_none(
        self,
        strategy: DummyLiquidityStrategy,
    ) -> None:
        context = ContextDouble(liquidity=None, explode_features=True)

        assert strategy._extract_snapshot(context) is None

    def test_extract_snapshot_rejects_snapshot_like_object_that_is_not_liquidity_snapshot(
        self,
        strategy: DummyLiquidityStrategy,
    ) -> None:
        fake_snapshot = SimpleNamespace(symbol="BTCUSDT", timeframe="1m")
        context = ContextDouble(
            liquidity={"snapshot": {"value": fake_snapshot}},
            features={"liquidity_map_snapshot": fake_snapshot},
        )

        assert strategy._extract_snapshot(context) is None


class TestCurrentPriceResolution:
    def test_resolve_current_price_prefers_mid_over_last_and_snapshot(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        snapshot = liquidity_snapshot_factory(current_price=99.0)
        context = ContextDouble(
            price=SimpleNamespace(mid_price=101.25, last_price=100.75),
        )

        assert strategy._resolve_current_price(context, snapshot) == 101.25

    def test_resolve_current_price_falls_back_to_last_when_mid_invalid(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        snapshot = liquidity_snapshot_factory(current_price=99.0)
        context = ContextDouble(
            price=SimpleNamespace(mid_price=float("nan"), last_price=100.75),
        )

        assert strategy._resolve_current_price(context, snapshot) == 100.75

    def test_resolve_current_price_falls_back_to_snapshot_when_context_price_invalid(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        snapshot = liquidity_snapshot_factory(current_price=99.0)
        context = ContextDouble(
            price=SimpleNamespace(mid_price=0.0, last_price=-1.0),
        )

        assert strategy._resolve_current_price(context, snapshot) == 99.0

    @pytest.mark.parametrize("raw_price", [None, 0.0, -100.0, float("nan"), float("inf")])
    def test_resolve_current_price_returns_none_for_non_positive_or_non_finite_prices(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
        raw_price: Any,
    ) -> None:
        snapshot = liquidity_snapshot_factory(current_price=raw_price)
        context = ContextDouble(
            price=SimpleNamespace(mid_price=raw_price, last_price=raw_price),
        )

        assert strategy._resolve_current_price(context, snapshot) is None


class TestContextValidation:
    def test_base_context_valid_when_snapshot_symbol_timeframe_and_freshness_match(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_runtime(strategy)
        _patch_freshness_ttl(strategy, ttl_seconds=60.0)

        now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        snapshot = liquidity_snapshot_factory(
            symbol="BTCUSDT",
            timeframe="1m",
            timestamp=now - timedelta(seconds=10),
        )
        context = ContextDouble(symbol="BTCUSDT", timeframe="1m", timestamp=now)

        assert strategy._base_context_is_valid(context, snapshot) is True

    def test_base_context_invalid_when_snapshot_symbol_mismatch(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_freshness_ttl(strategy, ttl_seconds=60.0)

        snapshot = liquidity_snapshot_factory(symbol="ETHUSDT", timeframe="1m")
        context = ContextDouble(symbol="BTCUSDT", timeframe="1m")

        assert strategy._base_context_is_valid(context, snapshot) is False

    def test_base_context_invalid_when_snapshot_timeframe_mismatch_even_if_symbol_matches(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_freshness_ttl(strategy, ttl_seconds=60.0)

        snapshot = liquidity_snapshot_factory(symbol="BTCUSDT", timeframe="5m")
        context = ContextDouble(symbol="BTCUSDT", timeframe="1m")

        assert strategy._base_context_is_valid(context, snapshot) is False

    def test_base_context_invalid_when_symbol_is_not_allowed_by_runtime(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_runtime(strategy, symbols={"ETHUSDT"})
        _patch_freshness_ttl(strategy, ttl_seconds=60.0)

        snapshot = liquidity_snapshot_factory(symbol="BTCUSDT", timeframe="1m")
        context = ContextDouble(symbol="BTCUSDT", timeframe="1m")

        assert strategy._base_context_is_valid(context, snapshot) is False

    def test_base_context_invalid_when_timeframe_is_not_allowed_by_runtime(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_runtime(strategy, timeframes={"5m"})
        _patch_freshness_ttl(strategy, ttl_seconds=60.0)

        snapshot = liquidity_snapshot_factory(symbol="BTCUSDT", timeframe="1m")
        context = ContextDouble(symbol="BTCUSDT", timeframe="1m")

        assert strategy._base_context_is_valid(context, snapshot) is False

    def test_base_context_invalid_when_regime_is_not_allowed(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_runtime(strategy, allowed_regimes={"trend"})
        _patch_freshness_ttl(strategy, ttl_seconds=60.0)

        snapshot = liquidity_snapshot_factory(symbol="BTCUSDT", timeframe="1m")
        context = ContextDouble(
            symbol="BTCUSDT",
            timeframe="1m",
            regime=SimpleNamespace(regime="range"),
        )

        assert strategy._base_context_is_valid(context, snapshot) is False

    def test_base_context_valid_when_regime_missing_even_if_allowed_regimes_configured(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_runtime(strategy, allowed_regimes={"trend"})
        _patch_freshness_ttl(strategy, ttl_seconds=60.0)

        snapshot = liquidity_snapshot_factory(symbol="BTCUSDT", timeframe="1m")
        context = ContextDouble(symbol="BTCUSDT", timeframe="1m", regime=None)

        assert strategy._base_context_is_valid(context, snapshot) is True


class TestSnapshotFreshness:
    def test_snapshot_not_stale_when_ttl_disabled(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_freshness_ttl(strategy, ttl_seconds=None)

        now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        snapshot = liquidity_snapshot_factory(timestamp=now - timedelta(days=365))
        context = ContextDouble(timestamp=now)

        assert strategy._snapshot_is_stale(context, snapshot) is False

    def test_snapshot_stale_when_age_exceeds_ttl(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_freshness_ttl(strategy, ttl_seconds=30.0)

        now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        snapshot = liquidity_snapshot_factory(timestamp=now - timedelta(seconds=31))
        context = ContextDouble(timestamp=now)

        assert strategy._snapshot_is_stale(context, snapshot) is True

    def test_snapshot_not_stale_when_age_is_exactly_ttl_boundary(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_freshness_ttl(strategy, ttl_seconds=30.0)

        now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        snapshot = liquidity_snapshot_factory(timestamp=now - timedelta(seconds=30))
        context = ContextDouble(timestamp=now)

        assert strategy._snapshot_is_stale(context, snapshot) is False

    def test_snapshot_stale_when_snapshot_timestamp_missing_and_ttl_enabled(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_freshness_ttl(strategy, ttl_seconds=30.0)

        snapshot = liquidity_snapshot_factory(timestamp=None)
        context = ContextDouble(timestamp=datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc))

        assert strategy._snapshot_is_stale(context, snapshot) is True

    def test_snapshot_uses_feature_is_stale_before_ttl(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_freshness_ttl(strategy, ttl_seconds=9999.0)

        now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        feature_snapshot = FeatureSnapshot(stale=True)
        snapshot = liquidity_snapshot_factory(timestamp=now)
        context = ContextDouble(timestamp=now, feature_snapshot=feature_snapshot)

        assert strategy._snapshot_is_stale(context, snapshot) is True
        assert feature_snapshot.calls == [now]

    def test_snapshot_falls_back_to_ttl_when_feature_staleness_check_raises(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_freshness_ttl(strategy, ttl_seconds=10.0)

        now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        feature_snapshot = FeatureSnapshot(stale=RuntimeError("stale backend down"))
        snapshot = liquidity_snapshot_factory(timestamp=now - timedelta(seconds=11))
        context = ContextDouble(timestamp=now, feature_snapshot=feature_snapshot)

        assert strategy._snapshot_is_stale(context, snapshot) is True


class TestCommonPreFilters:
    def test_common_pre_filters_include_snapshot_presence_and_price_validation(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
        liquidity_level_factory: Any,
    ) -> None:
        _patch_filters(strategy, enable_spread_filter=False, enable_liquidity_filter=False)

        snapshot = liquidity_snapshot_factory(
            active_levels=[liquidity_level_factory(price=101.0)],
            stop_clusters=[],
            above_liquidity_score=0.5,
            below_liquidity_score=0.2,
        )
        context = ContextDouble(price=SimpleNamespace(mid_price=100.0))

        results = strategy._run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=100.0,
        )

        assert _filter_by_name(results, "liquidity_snapshot_presence").decision is FilterDecision.PASS
        assert _filter_by_name(results, "price_validation").decision is FilterDecision.PASS

    def test_common_pre_filters_block_empty_liquidity_snapshot(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
    ) -> None:
        _patch_filters(strategy, enable_spread_filter=False, enable_liquidity_filter=False)

        snapshot = liquidity_snapshot_factory(active_levels=[], stop_clusters=[])
        context = ContextDouble(price=SimpleNamespace(mid_price=100.0))

        results = strategy._run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=100.0,
        )

        presence = _filter_by_name(results, "liquidity_snapshot_presence")
        assert presence.decision is FilterDecision.BLOCK
        assert presence.blocked is True

    @pytest.mark.parametrize("bad_price", [0.0, -1.0, float("nan"), float("inf")])
    def test_common_pre_filters_block_invalid_current_price(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
        liquidity_level_factory: Any,
        bad_price: float,
    ) -> None:
        _patch_filters(strategy, enable_spread_filter=False, enable_liquidity_filter=False)

        snapshot = liquidity_snapshot_factory(
            active_levels=[liquidity_level_factory(price=101.0)],
            stop_clusters=[],
        )
        context = ContextDouble(price=SimpleNamespace(mid_price=bad_price))

        results = strategy._run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=bad_price,
        )

        price_filter = _filter_by_name(results, "price_validation")
        assert price_filter.decision is FilterDecision.BLOCK
        assert price_filter.blocked is True

    def test_common_pre_filters_block_portfolio_blocked_symbol(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
        liquidity_level_factory: Any,
    ) -> None:
        _patch_filters(strategy, enable_spread_filter=False, enable_liquidity_filter=False)

        snapshot = liquidity_snapshot_factory(
            active_levels=[liquidity_level_factory(price=101.0)],
            stop_clusters=[],
        )
        context = ContextDouble(
            symbol="BTCUSDT",
            portfolio=SimpleNamespace(blocked_symbols={"BTCUSDT", "ETHUSDT"}),
        )

        results = strategy._run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=100.0,
        )

        portfolio_filter = _filter_by_name(results, "portfolio_blocked_symbol")
        assert portfolio_filter.decision is FilterDecision.BLOCK
        assert portfolio_filter.blocked is True

    def test_common_pre_filters_pass_portfolio_filter_when_symbol_not_blocked(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
        liquidity_level_factory: Any,
    ) -> None:
        _patch_filters(strategy, enable_spread_filter=False, enable_liquidity_filter=False)

        snapshot = liquidity_snapshot_factory(
            active_levels=[liquidity_level_factory(price=101.0)],
            stop_clusters=[],
        )
        context = ContextDouble(
            symbol="BTCUSDT",
            portfolio=SimpleNamespace(blocked_symbols={"ETHUSDT"}),
        )

        results = strategy._run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=100.0,
        )

        portfolio_filter = _filter_by_name(results, "portfolio_blocked_symbol")
        assert portfolio_filter.decision is FilterDecision.PASS
        assert portfolio_filter.blocked is False

    def test_common_pre_filters_block_spread_above_threshold(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
        liquidity_level_factory: Any,
    ) -> None:
        _patch_filters(
            strategy,
            enable_spread_filter=True,
            max_spread_bps=5.0,
            enable_liquidity_filter=False,
        )

        snapshot = liquidity_snapshot_factory(
            active_levels=[liquidity_level_factory(price=101.0)],
            stop_clusters=[],
        )
        context = ContextDouble(price=SimpleNamespace(spread_bps=8.5))

        results = strategy._run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=100.0,
        )

        spread_filter = _filter_by_name(results, "spread_filter")
        assert spread_filter.decision is FilterDecision.BLOCK
        assert spread_filter.blocked is True

    def test_common_pre_filters_skip_spread_as_pass_when_price_context_missing(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
        liquidity_level_factory: Any,
    ) -> None:
        _patch_filters(
            strategy,
            enable_spread_filter=True,
            max_spread_bps=5.0,
            enable_liquidity_filter=False,
        )

        snapshot = liquidity_snapshot_factory(
            active_levels=[liquidity_level_factory(price=101.0)],
            stop_clusters=[],
        )
        context = ContextDouble(price=None)

        results = strategy._run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=100.0,
        )

        spread_filter = _filter_by_name(results, "spread_filter")
        assert spread_filter.decision is FilterDecision.PASS
        assert "skipped" in spread_filter.reason.lower()

    def test_common_pre_filters_block_weak_liquidity_when_liquidity_filter_enabled(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
        liquidity_level_factory: Any,
    ) -> None:
        _patch_filters(
            strategy,
            enable_spread_filter=False,
            enable_liquidity_filter=True,
            min_liquidity_score=0.75,
        )

        snapshot = liquidity_snapshot_factory(
            active_levels=[liquidity_level_factory(price=101.0)],
            stop_clusters=[],
            above_liquidity_score=0.3,
            below_liquidity_score=0.4,
        )
        context = ContextDouble(price=SimpleNamespace(mid_price=100.0))

        results = strategy._run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=100.0,
        )

        liquidity_filter = _filter_by_name(results, "liquidity_strength_filter")
        assert liquidity_filter.decision is FilterDecision.BLOCK
        assert liquidity_filter.blocked is True

    def test_common_pre_filters_pass_when_strongest_liquidity_meets_threshold(
        self,
        strategy: DummyLiquidityStrategy,
        liquidity_snapshot_factory: Any,
        liquidity_level_factory: Any,
    ) -> None:
        _patch_filters(
            strategy,
            enable_spread_filter=False,
            enable_liquidity_filter=True,
            min_liquidity_score=0.75,
        )

        snapshot = liquidity_snapshot_factory(
            active_levels=[liquidity_level_factory(price=101.0)],
            stop_clusters=[],
            above_liquidity_score=0.74,
            below_liquidity_score=0.76,
        )
        context = ContextDouble(price=SimpleNamespace(mid_price=100.0))

        results = strategy._run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=100.0,
        )

        liquidity_filter = _filter_by_name(results, "liquidity_strength_filter")
        assert liquidity_filter.decision is FilterDecision.PASS
        assert liquidity_filter.blocked is False


class TestEmitCooldown:
    def test_emit_cooldown_key_is_strategy_symbol_timeframe_scoped(
        self,
        strategy: DummyLiquidityStrategy,
    ) -> None:
        assert (
            strategy._cooldown_key("BTCUSDT", "1m")
            == "dummy_liquidity_strategy:BTCUSDT:1m"
        )

    def test_is_on_emit_cooldown_false_when_runtime_cooldown_disabled(
        self,
        strategy: DummyLiquidityStrategy,
    ) -> None:
        _patch_runtime(strategy, emit_cooldown_seconds=0.0)

        now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        strategy._last_emitted_at[strategy._cooldown_key("BTCUSDT", "1m")] = now

        assert strategy._is_on_emit_cooldown("BTCUSDT", "1m", now) is False

    def test_is_on_emit_cooldown_true_within_window(
        self,
        strategy: DummyLiquidityStrategy,
    ) -> None:
        _patch_runtime(strategy, emit_cooldown_seconds=60.0)

        now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        strategy._last_emitted_at[strategy._cooldown_key("BTCUSDT", "1m")] = (
            now - timedelta(seconds=59)
        )

        assert strategy._is_on_emit_cooldown("BTCUSDT", "1m", now) is True

    def test_is_on_emit_cooldown_false_after_window_expires(
        self,
        strategy: DummyLiquidityStrategy,
    ) -> None:
        _patch_runtime(strategy, emit_cooldown_seconds=60.0)

        now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        strategy._last_emitted_at[strategy._cooldown_key("BTCUSDT", "1m")] = (
            now - timedelta(seconds=60)
        )

        assert strategy._is_on_emit_cooldown("BTCUSDT", "1m", now) is False


class TestSignalEmission:
    @pytest.mark.asyncio
    async def test_emit_signal_publishes_signal_generated_payload(
        self,
        strategy: DummyLiquidityStrategy,
        strategy_signal_factory: Any,
    ) -> None:
        _patch_runtime(strategy, emit_cooldown_seconds=0.0)

        timestamp = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        signal = strategy_signal_factory(
            symbol="BTCUSDT",
            timeframe="1m",
            strategy_name=strategy.strategy_name,
            side=SignalSide.LONG,
            status=SignalStatus.NEW,
            timestamp=timestamp,
            confidence=0.81,
            score=1.45,
        )
        context = ContextDouble(symbol="BTCUSDT", timeframe="1m", timestamp=timestamp)

        strategy.emit_event = AsyncMock()
        strategy._to_payload = lambda signal: {"safe": True, "symbol": signal.symbol}

        result = await strategy.emit_signal(signal=signal, context=context)

        assert result is signal
        strategy.emit_event.assert_awaited_once()

        topic, payload = strategy.emit_event.await_args.args
        kwargs = strategy.emit_event.await_args.kwargs

        assert topic == strategy.SIGNAL_TOPIC
        assert kwargs["source"] == strategy.strategy_name

        assert payload["symbol"] == "BTCUSDT"
        assert payload["strategy_name"] == strategy.strategy_name
        assert payload["side"] == "long"
        assert payload["status"] == "new"
        assert payload["score"] == 1.45
        assert payload["confidence"] == 0.81
        assert payload["source"] == strategy.strategy_name
        assert payload["signal"] is signal
        assert payload["signal_payload"] == {"safe": True, "symbol": "BTCUSDT"}

        cooldown_key = strategy._cooldown_key("BTCUSDT", "1m")
        assert strategy._last_emitted_at[cooldown_key] == timestamp

    @pytest.mark.asyncio
    async def test_emit_signal_suppresses_duplicate_within_cooldown_without_event_emit(
        self,
        strategy: DummyLiquidityStrategy,
        strategy_signal_factory: Any,
    ) -> None:
        _patch_runtime(strategy, emit_cooldown_seconds=60.0)

        first_ts = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        second_ts = first_ts + timedelta(seconds=30)

        signal = strategy_signal_factory(
            symbol="BTCUSDT",
            timeframe="1m",
            strategy_name=strategy.strategy_name,
            side=SignalSide.LONG,
            timestamp=second_ts,
        )
        context = ContextDouble(symbol="BTCUSDT", timeframe="1m", timestamp=second_ts)

        strategy._last_emitted_at[strategy._cooldown_key("BTCUSDT", "1m")] = first_ts
        strategy.emit_event = AsyncMock()

        result = await strategy.emit_signal(signal=signal, context=context)

        assert result is None
        strategy.emit_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_emit_signal_allows_same_symbol_after_cooldown_expired(
        self,
        strategy: DummyLiquidityStrategy,
        strategy_signal_factory: Any,
    ) -> None:
        _patch_runtime(strategy, emit_cooldown_seconds=60.0)

        first_ts = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        second_ts = first_ts + timedelta(seconds=61)

        signal = strategy_signal_factory(
            symbol="BTCUSDT",
            timeframe="1m",
            strategy_name=strategy.strategy_name,
            side=SignalSide.LONG,
            timestamp=second_ts,
        )
        context = ContextDouble(symbol="BTCUSDT", timeframe="1m", timestamp=second_ts)

        strategy._last_emitted_at[strategy._cooldown_key("BTCUSDT", "1m")] = first_ts
        strategy.emit_event = AsyncMock()
        strategy._to_payload = lambda signal: {"safe": True}

        result = await strategy.emit_signal(signal=signal, context=context)

        assert result is signal
        strategy.emit_event.assert_awaited_once()
        assert strategy._last_emitted_at[strategy._cooldown_key("BTCUSDT", "1m")] == second_ts

    @pytest.mark.asyncio
    async def test_emit_signal_cooldown_is_scoped_by_timeframe(
        self,
        strategy: DummyLiquidityStrategy,
        strategy_signal_factory: Any,
    ) -> None:
        _patch_runtime(strategy, emit_cooldown_seconds=60.0)

        last_ts = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        current_ts = last_ts + timedelta(seconds=10)

        strategy._last_emitted_at[strategy._cooldown_key("BTCUSDT", "1m")] = last_ts

        signal = strategy_signal_factory(
            symbol="BTCUSDT",
            timeframe="5m",
            strategy_name=strategy.strategy_name,
            side=SignalSide.LONG,
            timestamp=current_ts,
        )
        context = ContextDouble(symbol="BTCUSDT", timeframe="5m", timestamp=current_ts)

        strategy.emit_event = AsyncMock()
        strategy._to_payload = lambda signal: {"safe": True}

        result = await strategy.emit_signal(signal=signal, context=context)

        assert result is signal
        strategy.emit_event.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_emit_signal_rejects_symbol_mismatch_before_emitting(
        self,
        strategy: DummyLiquidityStrategy,
        strategy_signal_factory: Any,
    ) -> None:
        signal = strategy_signal_factory(
            symbol="ETHUSDT",
            timeframe="1m",
            strategy_name=strategy.strategy_name,
            side=SignalSide.LONG,
        )
        context = ContextDouble(symbol="BTCUSDT", timeframe="1m")

        strategy.emit_event = AsyncMock()

        with pytest.raises(ValueError, match="symbol mismatch"):
            await strategy.emit_signal(signal=signal, context=context)

        strategy.emit_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_emit_signal_rejects_timeframe_mismatch_before_emitting(
        self,
        strategy: DummyLiquidityStrategy,
        strategy_signal_factory: Any,
    ) -> None:
        signal = strategy_signal_factory(
            symbol="BTCUSDT",
            timeframe="5m",
            strategy_name=strategy.strategy_name,
            side=SignalSide.LONG,
        )
        context = ContextDouble(symbol="BTCUSDT", timeframe="1m")

        strategy.emit_event = AsyncMock()

        with pytest.raises(ValueError, match="timeframe mismatch"):
            await strategy.emit_signal(signal=signal, context=context)

        strategy.emit_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_emit_signal_rejects_strategy_name_mismatch_before_emitting(
        self,
        strategy: DummyLiquidityStrategy,
        strategy_signal_factory: Any,
    ) -> None:
        signal = strategy_signal_factory(
            symbol="BTCUSDT",
            timeframe="1m",
            strategy_name="other_strategy",
            side=SignalSide.LONG,
        )
        context = ContextDouble(symbol="BTCUSDT", timeframe="1m")

        strategy.emit_event = AsyncMock()

        with pytest.raises(ValueError, match="Signal strategy mismatch"):
            await strategy.emit_signal(signal=signal, context=context)

        strategy.emit_event.assert_not_awaited()