# tests/strategy/strategies/hybrid/test_hybrid_strategies.py

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from strategy.config import (
    StrategyConfig,
    StrategyDefinitionConfig,
    StrategyRuntimeConfig,
)
from strategy.enums import (
    FeatureSource,
    MarketRegime,
    SetupType,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    Timeframe,
)
from strategy.exceptions import StrategyConfigError
from strategy.models import StrategyContext, StrategySignal, utcnow

from strategy.strategies.hybrid.base import (
    HYBRID_FEATURES,
    HybridCompositeSnapshot,
    HybridStrategyConfig,
    HybridStrategyScope,
)
from strategy.strategies.hybrid.confluence_strategy import (
    ConfluenceStrategy,
    ConfluenceStrategyConfig,
)
from strategy.strategies.hybrid.trend_stack_strategy import (
    TrendStackStrategy,
    TrendStackStrategyConfig,
)
from strategy.strategies.hybrid.mean_reversion_stack_strategy import (
    MeanReversionStackStrategy,
    MeanReversionStackStrategyConfig,
)
from strategy.strategies.hybrid.liquidation_whale_strategy import (
    LiquidationWhaleStrategy,
    LiquidationWhaleStrategyConfig,
)
from strategy.strategies.hybrid.liquidity_orderflow_reversal_strategy import (
    LiquidityOrderflowReversalStrategy,
    LiquidityOrderflowReversalStrategyConfig,
)
from strategy.strategies.hybrid.oi_funding_squeeze_strategy import (
    OIFundingSqueezeStrategy,
    OIFundingSqueezeStrategyConfig,
)
from strategy.strategies.hybrid.whale_orderflow_breakout_strategy import (
    WhaleOrderflowBreakoutStrategy,
    WhaleOrderflowBreakoutStrategyConfig,
)


# =============================================================================
# Shared helpers
# =============================================================================


HYBRID_REQUIRED_FEATURES: tuple[str, ...] = (
    HYBRID_FEATURES.DOMINANT_SIDE,
    HYBRID_FEATURES.ALIGNMENT_SCORE,
    HYBRID_FEATURES.CONFLUENCE_SCORE,
)


def _runtime_config() -> StrategyRuntimeConfig:
    runtime = StrategyRuntimeConfig(
        enabled=True,
        symbols=["BTCUSDT"],
        timeframes=[Timeframe.M1, Timeframe.M5],
        allowed_regimes=[MarketRegime.UNKNOWN],
        cooldown_seconds=0,
        max_signal_age_seconds=120,
        min_confidence=0.0,
        min_score=0.0,
    )
    runtime.validate()
    return runtime


def _definition(
    *,
    name: str,
    required_features: set[str] | None = None,
) -> StrategyDefinitionConfig:
    definition = StrategyDefinitionConfig(
        name=name,
        category=StrategyCategory.HYBRID,
        runtime=_runtime_config(),
        required_features=required_features or set(),
        weight=1.0,
        priority=10,
        tags=["hybrid", "unit", "futures"],
        metadata={"source": "tests"},
    )
    definition.validate()
    return definition


def _strategy_config(definition: StrategyDefinitionConfig) -> StrategyConfig:
    config = StrategyConfig(
        runtime=StrategyRuntimeConfig(
            enabled=True,
            symbols=[],
            timeframes=[Timeframe.M1, Timeframe.M5],
            allowed_regimes=[MarketRegime.UNKNOWN],
            min_confidence=0.0,
            min_score=0.0,
            max_signal_age_seconds=120,
        ),
        strategies={definition.name: definition},
    )
    config.validate()
    return config


def _base_domain(
    *,
    side: str = "long",
    score: float = 0.88,
    confidence: float = 0.86,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "market_type": "usdm_futures",
        "timeframe": Timeframe.M1.value,
        "side": side,
        "score": score,
        "confidence": confidence,
        "timestamp": utcnow(),
        **extra,
    }


def _hybrid_features(
    make_feature: Callable[..., Any],
    *,
    side: str = "long",
    alignment: float = 0.86,
    confluence: float = 0.84,
    confidence: float = 0.86,
) -> list[Any]:
    return [
        make_feature(
            name=HYBRID_FEATURES.DOMINANT_SIDE,
            value=side,
            source=FeatureSource.SYSTEM,
            symbol="BTCUSDT",
            confidence=confidence,
            normalized_value=None,
        ),
        make_feature(
            name=HYBRID_FEATURES.ALIGNMENT_SCORE,
            value=alignment,
            source=FeatureSource.SYSTEM,
            symbol="BTCUSDT",
            confidence=confidence,
            normalized_value=alignment,
        ),
        make_feature(
            name=HYBRID_FEATURES.CONFLUENCE_SCORE,
            value=confluence,
            source=FeatureSource.SYSTEM,
            symbol="BTCUSDT",
            confidence=confidence,
            normalized_value=confluence,
        ),
        make_feature(
            name=HYBRID_FEATURES.CONFLICT_SCORE,
            value=0.05,
            source=FeatureSource.SYSTEM,
            symbol="BTCUSDT",
            confidence=confidence,
            normalized_value=0.05,
        ),
        make_feature(
            name=HYBRID_FEATURES.CONFIDENCE,
            value=confidence,
            source=FeatureSource.SYSTEM,
            symbol="BTCUSDT",
            confidence=confidence,
            normalized_value=confidence,
        ),
    ]


def _context(
    make_context: Callable[..., StrategyContext],
    make_feature: Callable[..., Any],
    *,
    domain_data: dict[FeatureSource, dict[str, Any]],
    side: str = "long",
) -> StrategyContext:
    return make_context(
        symbol="BTCUSDT",
        timeframe=Timeframe.M1,
        features=_hybrid_features(make_feature, side=side),
        domain_data=domain_data,
        metadata={
            "exchange": "binance",
            "market_type": "usdm_futures",
            "source": "tests",
        },
    )


def _permissive_confluence_config() -> ConfluenceStrategyConfig:
    config = ConfluenceStrategyConfig(
        min_score=0.05,
        min_confidence=0.05,
        min_alignment_score=0.05,
        min_confluence_score=0.05,
        max_conflict_score=1.0,
        min_confluence_strategy_score=0.05,
        min_confluence_strategy_confidence=0.05,
        min_required_domains=2,
        min_aligned_domains=2,
        min_strong_votes=0,
        min_vote_score=0.05,
        min_vote_confidence=0.05,
        strong_vote_min_score=0.05,
        strong_vote_min_confidence=0.05,
        reject_direct_conflicts=False,
        required_hybrid_features=HYBRID_REQUIRED_FEATURES,
    )
    config.validate()
    return config


def _permissive_trend_config() -> TrendStackStrategyConfig:
    config = TrendStackStrategyConfig(
        min_score=0.05,
        min_confidence=0.05,
        min_alignment_score=0.05,
        min_confluence_score=0.05,
        max_conflict_score=1.0,
        min_trend_stack_score=0.05,
        min_trend_stack_confidence=0.05,
        min_price_action_score=0.05,
        min_orderflow_score=0.05,
        min_oi_score=0.05,
        min_whale_score=0.05,
        require_open_interest=False,
        require_whales=False,
        require_funding=False,
        reject_high_conflict=False,
        block_extreme_opposite_funding=False,
        block_funding_same_side_overcrowding=False,
        min_aligned_confirmations=1,
        required_hybrid_features=HYBRID_REQUIRED_FEATURES,
    )
    config.validate()
    return config


def _permissive_reversion_config() -> MeanReversionStackStrategyConfig:
    config = MeanReversionStackStrategyConfig(
        min_score=0.05,
        min_confidence=0.05,
        min_alignment_score=0.05,
        min_confluence_score=0.05,
        max_conflict_score=1.0,
        min_reversion_score=0.05,
        min_reversion_confidence=0.05,
        min_liquidity_sweep_score=0.05,
        min_orderflow_exhaustion_score=0.05,
        min_price_rejection_score=0.05,
        require_liquidations=False,
        require_whales=False,
        reject_same_side_momentum=False,
        reject_high_conflict=False,
        min_aligned_confirmations=1,
        required_hybrid_features=HYBRID_REQUIRED_FEATURES,
    )
    config.validate()
    return config


def _permissive_liquidation_whale_config() -> LiquidationWhaleStrategyConfig:
    config = LiquidationWhaleStrategyConfig(
        min_score=0.05,
        min_confidence=0.05,
        min_alignment_score=0.05,
        min_confluence_score=0.05,
        max_conflict_score=1.0,
        min_liquidation_whale_score=0.05,
        min_liquidation_whale_confidence=0.05,
        min_liquidation_score=0.05,
        min_whale_score=0.05,
        min_absorption_score=0.05,
        min_exhaustion_score=0.0,
        reject_same_side_whale_pressure=False,
        reject_high_conflict=False,
        require_exhaustion_same_as_liquidation=False,
        required_hybrid_features=HYBRID_REQUIRED_FEATURES,
    )
    config.validate()
    return config


def _permissive_liquidity_orderflow_config() -> LiquidityOrderflowReversalStrategyConfig:
    config = LiquidityOrderflowReversalStrategyConfig(
        min_score=0.05,
        min_confidence=0.05,
        min_alignment_score=0.05,
        min_confluence_score=0.05,
        max_conflict_score=1.0,
        min_liquidity_orderflow_score=0.05,
        min_liquidity_orderflow_confidence=0.05,
        min_liquidity_sweep_score=0.05,
        min_orderflow_exhaustion_score=0.05,
        min_absorption_score=0.0,
        min_rejection_score=0.0,
        reject_same_side_momentum=False,
        reject_high_conflict=False,
        require_absorption_same_as_reversal=False,
        require_rejection_same_as_reversal=False,
        required_hybrid_features=HYBRID_REQUIRED_FEATURES,
    )
    config.validate()
    return config


def _permissive_oi_funding_config() -> OIFundingSqueezeStrategyConfig:
    config = OIFundingSqueezeStrategyConfig(
        min_score=0.05,
        min_confidence=0.05,
        min_alignment_score=0.05,
        min_confluence_score=0.05,
        max_conflict_score=1.0,
        min_squeeze_score=0.05,
        min_squeeze_confidence=0.05,
        min_oi_score=0.05,
        min_funding_extreme_score=0.05,
        min_squeeze_probability=0.05,
        require_price_action_confirmation=False,
        reject_same_side_crowding_momentum=False,
        reject_high_conflict=False,
        required_hybrid_features=HYBRID_REQUIRED_FEATURES,
    )
    config.validate()
    return config


def _permissive_whale_orderflow_config() -> WhaleOrderflowBreakoutStrategyConfig:
    config = WhaleOrderflowBreakoutStrategyConfig(
        min_score=0.05,
        min_confidence=0.05,
        min_alignment_score=0.05,
        min_confluence_score=0.05,
        max_conflict_score=1.0,
        min_whale_orderflow_score=0.05,
        min_whale_orderflow_confidence=0.05,
        min_whale_score=0.05,
        min_orderflow_score=0.05,
        min_breakout_score=0.05,
        min_large_trade_score=0.0,
        min_pressure_score=0.0,
        min_price_action_score=0.0,
        require_price_action_confirmation=False,
        reject_high_conflict=False,
        reject_opposite_price_action=False,
        required_hybrid_features=HYBRID_REQUIRED_FEATURES,
    )
    config.validate()
    return config


def _build_strategy(
    *,
    strategy_cls: type,
    strategy_name: str,
    hybrid_config: HybridStrategyConfig,
    mock_event_bus,
    mock_scheduler,
) -> Any:
    definition = _definition(name=strategy_name)
    config = _strategy_config(definition)

    return strategy_cls(
        config=config,
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        definition=definition,
        hybrid_config=hybrid_config,
    )


# =============================================================================
# Base hybrid DTOs/config
# =============================================================================


class TestHybridBaseObjects:
    def test_hybrid_feature_names_are_unique_and_non_empty(self) -> None:
        names = HYBRID_FEATURES.all()

        assert HYBRID_FEATURES.DOMINANT_SIDE in names
        assert HYBRID_FEATURES.ALIGNMENT_SCORE in names
        assert HYBRID_FEATURES.CONFLUENCE_SCORE in names
        assert len(names) == len(set(names))
        assert all(name.strip() for name in names)

    def test_hybrid_scope_normalizes_values(self) -> None:
        scope = HybridStrategyScope(
            exchange=" Binance ",
            market_type=" USDM_FUTURES ",
            symbol=" btcusdt ",
            timeframe=" 1M ",
            exchange_symbol=None,
        )

        assert scope.exchange == "binance"
        assert scope.market_type == "usdm_futures"
        assert scope.symbol == "BTCUSDT"
        assert scope.timeframe == "1m"
        assert scope.exchange_symbol == "BTCUSDT"
        assert scope.key == "binance:usdm_futures:BTCUSDT:1m"
        assert scope.legacy_key == "binance:BTCUSDT"

    def test_hybrid_scope_rejects_empty_symbol(self) -> None:
        with pytest.raises(Exception):
            HybridStrategyScope(
                exchange="binance",
                market_type="usdm_futures",
                symbol=" ",
            )

    def test_hybrid_strategy_config_rejects_invalid_values(self) -> None:
        config = HybridStrategyConfig(min_score=1.01)

        with pytest.raises(StrategyConfigError):
            config.validate()

        config = HybridStrategyConfig(stale_feature_max_age_seconds=0)

        with pytest.raises(StrategyConfigError):
            config.validate()

        config = HybridStrategyConfig(min_required_domains=0)

        with pytest.raises(StrategyConfigError):
            config.validate()

    def test_hybrid_composite_snapshot_properties(self) -> None:
        snapshot = HybridCompositeSnapshot(
            symbol="btcusdt",
            exchange="binance",
            market_type="usdm_futures",
            timeframe="1m",
            orderflow=_base_domain(side="long"),
            liquidity=_base_domain(side="long"),
            dominant_side=SignalSide.LONG,
            alignment_score=0.9,
            conflict_score=0.05,
            confluence_score=0.88,
            confidence=0.86,
            timestamp=utcnow(),
        )

        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.directional
        assert snapshot.domain_count >= 2
        assert snapshot.has_domain(FeatureSource.ORDERFLOW)
        assert snapshot.has_domain(FeatureSource.LIQUIDITY)
        assert snapshot.has_minimum_data()
        assert snapshot.scope_key() == "binance:usdm_futures:BTCUSDT:1m"

        payload = snapshot.to_signal_payload()
        assert payload["symbol"] == "BTCUSDT"
        assert payload["dominant_side"] == "long"
        assert payload["market_type"] == "usdm_futures"


# =============================================================================
# Config validation for concrete hybrid strategies
# =============================================================================


@pytest.mark.parametrize(
    "config",
    [
        ConfluenceStrategyConfig(min_confluence_strategy_score=-0.1),
        TrendStackStrategyConfig(min_trend_stack_score=-0.1),
        MeanReversionStackStrategyConfig(min_reversion_score=-0.1),
        LiquidationWhaleStrategyConfig(min_liquidation_whale_score=-0.1),
        LiquidityOrderflowReversalStrategyConfig(min_liquidity_orderflow_score=-0.1),
        OIFundingSqueezeStrategyConfig(min_squeeze_score=-0.1),
        WhaleOrderflowBreakoutStrategyConfig(min_whale_orderflow_score=-0.1),
    ],
)
def test_concrete_hybrid_configs_reject_invalid_unit_scores(
    config: HybridStrategyConfig,
) -> None:
    with pytest.raises(StrategyConfigError):
        config.validate()


@pytest.mark.parametrize(
    "config",
    [
        ConfluenceStrategyConfig(),
        TrendStackStrategyConfig(),
        MeanReversionStackStrategyConfig(),
        LiquidationWhaleStrategyConfig(),
        LiquidityOrderflowReversalStrategyConfig(),
        OIFundingSqueezeStrategyConfig(),
        WhaleOrderflowBreakoutStrategyConfig(),
    ],
)
def test_concrete_hybrid_configs_validate_defaults(
    config: HybridStrategyConfig,
) -> None:
    config.validate()


# =============================================================================
# Strategy construction / metadata / required features
# =============================================================================


@pytest.mark.parametrize(
    ("strategy_cls", "strategy_name", "hybrid_config", "setup_type"),
    [
        (
            ConfluenceStrategy,
            "hybrid_confluence",
            _permissive_confluence_config(),
            SetupType.HYBRID,
        ),
        (
            TrendStackStrategy,
            "trend_stack",
            _permissive_trend_config(),
            SetupType.CONTINUATION,
        ),
        (
            MeanReversionStackStrategy,
            "mean_reversion_stack",
            _permissive_reversion_config(),
            SetupType.MEAN_REVERSION,
        ),
        (
            LiquidationWhaleStrategy,
            "liquidation_whale",
            _permissive_liquidation_whale_config(),
            SetupType.REVERSAL,
        ),
        (
            LiquidityOrderflowReversalStrategy,
            "liquidity_orderflow_reversal",
            _permissive_liquidity_orderflow_config(),
            SetupType.REVERSAL,
        ),
        (
            OIFundingSqueezeStrategy,
            "oi_funding_squeeze",
            _permissive_oi_funding_config(),
            SetupType.SQUEEZE,
        ),
        (
            WhaleOrderflowBreakoutStrategy,
            "whale_orderflow_breakout",
            _permissive_whale_orderflow_config(),
            SetupType.BREAKOUT,
        ),
    ],
)
def test_hybrid_strategy_metadata_and_required_features(
    strategy_cls: type,
    strategy_name: str,
    hybrid_config: HybridStrategyConfig,
    setup_type: SetupType,
    mock_event_bus,
    mock_scheduler,
) -> None:
    strategy = _build_strategy(
        strategy_cls=strategy_cls,
        strategy_name=strategy_name,
        hybrid_config=hybrid_config,
        mock_event_bus=mock_event_bus,
        mock_scheduler=mock_scheduler,
    )

    assert strategy.strategy_name == strategy_name
    assert strategy.category is StrategyCategory.HYBRID
    assert strategy.default_setup_type is setup_type

    metadata = strategy.metadata()
    metadata.validate()

    assert metadata.strategy_name == strategy_name
    assert metadata.category is StrategyCategory.HYBRID

    required = strategy.required_features()

    assert HYBRID_FEATURES.DOMINANT_SIDE in required
    assert HYBRID_FEATURES.ALIGNMENT_SCORE in required
    assert HYBRID_FEATURES.CONFLUENCE_SCORE in required


# =============================================================================
# Happy-path generate_signal()
# =============================================================================


def _confluence_context(make_context, make_feature) -> StrategyContext:
    return _context(
        make_context,
        make_feature,
        side="long",
        domain_data={
            FeatureSource.ORDERFLOW: _base_domain(
                side="long",
                pressure_side="long",
                score=0.88,
                confidence=0.86,
            ),
            FeatureSource.LIQUIDITY: _base_domain(
                side="long",
                sweep_side="long",
                score=0.84,
                confidence=0.82,
            ),
            FeatureSource.PRICE_ACTION: _base_domain(
                side="long",
                trend_side="long",
                score=0.90,
                confidence=0.88,
            ),
        },
    )


def _trend_context(make_context, make_feature) -> StrategyContext:
    return _context(
        make_context,
        make_feature,
        side="long",
        domain_data={
            FeatureSource.PRICE_ACTION: _base_domain(
                side="long",
                trend_side="long",
                breakout_side="long",
                score=0.90,
                confidence=0.88,
            ),
            FeatureSource.ORDERFLOW: _base_domain(
                side="long",
                continuation_side="long",
                pressure_side="long",
                score=0.86,
                confidence=0.84,
            ),
            FeatureSource.OPEN_INTEREST: _base_domain(
                side="long",
                oi_side="long",
                score=0.82,
                confidence=0.80,
            ),
            FeatureSource.WHALES: _base_domain(
                side="long",
                whale_side="long",
                score=0.82,
                confidence=0.80,
            ),
        },
    )


def _mean_reversion_context(make_context, make_feature) -> StrategyContext:
    return _context(
        make_context,
        make_feature,
        side="long",
        domain_data={
            FeatureSource.LIQUIDITY: _base_domain(
                side="short",
                sweep_side="short",
                stop_hunt_side="short",
                score=0.90,
                confidence=0.88,
            ),
            FeatureSource.ORDERFLOW: _base_domain(
                side="short",
                exhaustion_side="short",
                absorption_side="long",
                score=0.86,
                confidence=0.84,
            ),
            FeatureSource.PRICE_ACTION: _base_domain(
                side="long",
                rejection_side="long",
                reversal_side="long",
                score=0.86,
                confidence=0.84,
            ),
            FeatureSource.LIQUIDATIONS: _base_domain(
                side="short",
                liquidation_side="short",
                score=0.78,
                confidence=0.76,
            ),
            FeatureSource.WHALES: _base_domain(
                side="long",
                whale_side="long",
                absorption_side="long",
                score=0.78,
                confidence=0.76,
            ),
        },
    )


def _liquidation_whale_context(make_context, make_feature) -> StrategyContext:
    return _context(
        make_context,
        make_feature,
        side="long",
        domain_data={
            FeatureSource.LIQUIDATIONS: _base_domain(
                side="short",
                liquidation_side="short",
                exhausted_side="short",
                exhaustion_score=0.82,
                score=0.90,
                confidence=0.88,
            ),
            FeatureSource.WHALES: _base_domain(
                side="long",
                whale_side="long",
                absorption_side="long",
                absorption_score=0.88,
                score=0.88,
                confidence=0.86,
            ),
        },
    )


def _liquidity_orderflow_context(make_context, make_feature) -> StrategyContext:
    return _context(
        make_context,
        make_feature,
        side="long",
        domain_data={
            FeatureSource.LIQUIDITY: _base_domain(
                side="short",
                sweep_side="short",
                stop_hunt_side="short",
                score=0.90,
                confidence=0.88,
            ),
            FeatureSource.ORDERFLOW: _base_domain(
                side="short",
                exhaustion_side="short",
                absorption_side="long",
                absorption_score=0.86,
                score=0.88,
                confidence=0.86,
            ),
            FeatureSource.PRICE_ACTION: _base_domain(
                side="long",
                rejection_side="long",
                reversal_side="long",
                score=0.82,
                confidence=0.80,
            ),
        },
    )


def _oi_funding_context(make_context, make_feature) -> StrategyContext:
    return _context(
        make_context,
        make_feature,
        side="long",
        domain_data={
            FeatureSource.OPEN_INTEREST: _base_domain(
                side="short",
                oi_side="short",
                crowded_side="short",
                squeeze_side="long",
                squeeze_probability=0.88,
                score=0.88,
                confidence=0.86,
            ),
            FeatureSource.FUNDING: _base_domain(
                side="short",
                funding_side="short",
                crowded_side="short",
                extreme=True,
                funding_extreme="negative_extreme",
                squeeze_probability=0.84,
                score=0.86,
                confidence=0.84,
            ),
            FeatureSource.PRICE_ACTION: _base_domain(
                side="long",
                signal_side="long",
                score=0.78,
                confidence=0.76,
            ),
        },
    )


def _whale_orderflow_context(make_context, make_feature) -> StrategyContext:
    return _context(
        make_context,
        make_feature,
        side="long",
        domain_data={
            FeatureSource.WHALES: _base_domain(
                side="long",
                whale_side="long",
                pressure_side="long",
                large_trade_score=0.86,
                pressure_score=0.84,
                score=0.90,
                confidence=0.88,
            ),
            FeatureSource.ORDERFLOW: _base_domain(
                side="long",
                continuation_side="long",
                pressure_side="long",
                score=0.88,
                confidence=0.86,
            ),
            FeatureSource.PRICE_ACTION: _base_domain(
                side="long",
                breakout_side="long",
                score=0.82,
                confidence=0.80,
            ),
        },
    )


@pytest.mark.parametrize(
    ("strategy_cls", "strategy_name", "hybrid_config", "context_factory", "expected_side"),
    [
        (
            ConfluenceStrategy,
            "hybrid_confluence",
            _permissive_confluence_config(),
            _confluence_context,
            SignalSide.LONG,
        ),
        (
            TrendStackStrategy,
            "trend_stack",
            _permissive_trend_config(),
            _trend_context,
            SignalSide.LONG,
        ),
        (
            MeanReversionStackStrategy,
            "mean_reversion_stack",
            _permissive_reversion_config(),
            _mean_reversion_context,
            SignalSide.LONG,
        ),
        (
            LiquidationWhaleStrategy,
            "liquidation_whale",
            _permissive_liquidation_whale_config(),
            _liquidation_whale_context,
            SignalSide.LONG,
        ),
        (
            LiquidityOrderflowReversalStrategy,
            "liquidity_orderflow_reversal",
            _permissive_liquidity_orderflow_config(),
            _liquidity_orderflow_context,
            SignalSide.LONG,
        ),
        (
            OIFundingSqueezeStrategy,
            "oi_funding_squeeze",
            _permissive_oi_funding_config(),
            _oi_funding_context,
            SignalSide.LONG,
        ),
        (
            WhaleOrderflowBreakoutStrategy,
            "whale_orderflow_breakout",
            _permissive_whale_orderflow_config(),
            _whale_orderflow_context,
            SignalSide.LONG,
        ),
    ],
)
@pytest.mark.asyncio()
async def test_hybrid_strategy_generate_signal_happy_path(
    strategy_cls: type,
    strategy_name: str,
    hybrid_config: HybridStrategyConfig,
    context_factory: Callable[..., StrategyContext],
    expected_side: SignalSide,
    make_context,
    make_feature,
    mock_event_bus,
    mock_scheduler,
) -> None:
    strategy = _build_strategy(
        strategy_cls=strategy_cls,
        strategy_name=strategy_name,
        hybrid_config=hybrid_config,
        mock_event_bus=mock_event_bus,
        mock_scheduler=mock_scheduler,
    )
    context = context_factory(make_context, make_feature)

    signal = await strategy.generate_signal(context)

    assert signal is not None
    assert isinstance(signal, StrategySignal)
    assert signal.strategy_name == strategy_name
    assert signal.category is StrategyCategory.HYBRID
    assert signal.symbol == "BTCUSDT"
    assert signal.side is expected_side
    assert signal.is_directional
    assert signal.status is SignalStatus.NEW
    assert signal.score >= 0.0
    assert signal.confidence >= 0.0
    assert signal.source_features
    assert signal.reasons
    assert signal.confirmations

    assert signal.entry_plan is None
    assert signal.exit_plan is None
    assert signal.invalidation_plan is None
    assert signal.execution_plan is None

    assert "hybrid" in signal.metadata.get("tags", [])
    assert signal.metadata["market_type"] == "usdm_futures"
    assert signal.metadata["exchange"] == "binance"

    assert not mock_event_bus.topic_emitted("signal.generated")
    assert not mock_event_bus.nowait_topic_emitted("signal.generated")


# =============================================================================
# Negative paths
# =============================================================================


@pytest.mark.parametrize(
    ("strategy_cls", "strategy_name", "hybrid_config"),
    [
        (ConfluenceStrategy, "hybrid_confluence", _permissive_confluence_config()),
        (TrendStackStrategy, "trend_stack", _permissive_trend_config()),
        (MeanReversionStackStrategy, "mean_reversion_stack", _permissive_reversion_config()),
        (LiquidationWhaleStrategy, "liquidation_whale", _permissive_liquidation_whale_config()),
        (
            LiquidityOrderflowReversalStrategy,
            "liquidity_orderflow_reversal",
            _permissive_liquidity_orderflow_config(),
        ),
        (OIFundingSqueezeStrategy, "oi_funding_squeeze", _permissive_oi_funding_config()),
        (
            WhaleOrderflowBreakoutStrategy,
            "whale_orderflow_breakout",
            _permissive_whale_orderflow_config(),
        ),
    ],
)
@pytest.mark.asyncio()
async def test_hybrid_strategy_returns_none_when_required_domains_missing(
    strategy_cls: type,
    strategy_name: str,
    hybrid_config: HybridStrategyConfig,
    make_context,
    make_feature,
    mock_event_bus,
    mock_scheduler,
) -> None:
    strategy = _build_strategy(
        strategy_cls=strategy_cls,
        strategy_name=strategy_name,
        hybrid_config=hybrid_config,
        mock_event_bus=mock_event_bus,
        mock_scheduler=mock_scheduler,
    )
    context = _context(
        make_context,
        make_feature,
        side="long",
        domain_data={},
    )

    signal = await strategy.generate_signal(context)

    assert signal is None
    assert not mock_event_bus.topic_emitted("signal.generated")
    assert not mock_event_bus.nowait_topic_emitted("signal.generated")


@pytest.mark.asyncio()
async def test_confluence_strategy_rejects_direct_conflicts_when_configured(
    make_context,
    make_feature,
    mock_event_bus,
    mock_scheduler,
) -> None:
    hybrid_config = ConfluenceStrategyConfig(
        min_score=0.05,
        min_confidence=0.05,
        min_alignment_score=0.05,
        min_confluence_score=0.05,
        max_conflict_score=1.0,
        min_confluence_strategy_score=0.05,
        min_confluence_strategy_confidence=0.05,
        min_required_domains=2,
        min_aligned_domains=1,
        min_strong_votes=0,
        min_vote_score=0.05,
        min_vote_confidence=0.05,
        reject_direct_conflicts=True,
        allow_single_conflict=False,
        required_hybrid_features=HYBRID_REQUIRED_FEATURES,
    )
    hybrid_config.validate()

    strategy = _build_strategy(
        strategy_cls=ConfluenceStrategy,
        strategy_name="hybrid_confluence",
        hybrid_config=hybrid_config,
        mock_event_bus=mock_event_bus,
        mock_scheduler=mock_scheduler,
    )
    context = _context(
        make_context,
        make_feature,
        side="long",
        domain_data={
            FeatureSource.ORDERFLOW: _base_domain(side="long", score=0.9, confidence=0.9),
            FeatureSource.LIQUIDITY: _base_domain(side="short", score=0.9, confidence=0.9),
            FeatureSource.PRICE_ACTION: _base_domain(side="long", score=0.9, confidence=0.9),
        },
    )

    signal = await strategy.generate_signal(context)

    assert signal is None


@pytest.mark.asyncio()
async def test_trend_stack_blocks_opposite_orderflow_when_same_side_required(
    make_context,
    make_feature,
    mock_event_bus,
    mock_scheduler,
) -> None:
    hybrid_config = _permissive_trend_config()
    hybrid_config.require_orderflow_same_side = True
    hybrid_config.reject_high_conflict = False
    hybrid_config.validate()

    strategy = _build_strategy(
        strategy_cls=TrendStackStrategy,
        strategy_name="trend_stack",
        hybrid_config=hybrid_config,
        mock_event_bus=mock_event_bus,
        mock_scheduler=mock_scheduler,
    )
    context = _context(
        make_context,
        make_feature,
        side="long",
        domain_data={
            FeatureSource.PRICE_ACTION: _base_domain(
                side="long",
                trend_side="long",
                score=0.90,
                confidence=0.88,
            ),
            FeatureSource.ORDERFLOW: _base_domain(
                side="short",
                continuation_side="short",
                score=0.90,
                confidence=0.88,
            ),
        },
    )

    signal = await strategy.generate_signal(context)

    assert signal is None


@pytest.mark.asyncio()
async def test_oi_funding_squeeze_blocks_missing_crowded_side_when_required(
    make_context,
    make_feature,
    mock_event_bus,
    mock_scheduler,
) -> None:
    hybrid_config = _permissive_oi_funding_config()
    hybrid_config.require_crowded_side = True
    hybrid_config.validate()

    strategy = _build_strategy(
        strategy_cls=OIFundingSqueezeStrategy,
        strategy_name="oi_funding_squeeze",
        hybrid_config=hybrid_config,
        mock_event_bus=mock_event_bus,
        mock_scheduler=mock_scheduler,
    )
    context = _context(
        make_context,
        make_feature,
        side="long",
        domain_data={
            FeatureSource.OPEN_INTEREST: _base_domain(
                side="long",
                oi_side="long",
                score=0.90,
                confidence=0.88,
            ),
            FeatureSource.FUNDING: _base_domain(
                side="long",
                funding_side="long",
                extreme=True,
                score=0.90,
                confidence=0.88,
            ),
        },
    )

    signal = await strategy.generate_signal(context)

    assert signal is None


# =============================================================================
# Base evaluate() integration boundary
# =============================================================================


@pytest.mark.parametrize(
    ("strategy_cls", "strategy_name", "hybrid_config", "context_factory"),
    [
        (
            ConfluenceStrategy,
            "hybrid_confluence",
            _permissive_confluence_config(),
            _confluence_context,
        ),
        (
            TrendStackStrategy,
            "trend_stack",
            _permissive_trend_config(),
            _trend_context,
        ),
        (
            MeanReversionStackStrategy,
            "mean_reversion_stack",
            _permissive_reversion_config(),
            _mean_reversion_context,
        ),
        (
            LiquidationWhaleStrategy,
            "liquidation_whale",
            _permissive_liquidation_whale_config(),
            _liquidation_whale_context,
        ),
        (
            LiquidityOrderflowReversalStrategy,
            "liquidity_orderflow_reversal",
            _permissive_liquidity_orderflow_config(),
            _liquidity_orderflow_context,
        ),
        (
            OIFundingSqueezeStrategy,
            "oi_funding_squeeze",
            _permissive_oi_funding_config(),
            _oi_funding_context,
        ),
        (
            WhaleOrderflowBreakoutStrategy,
            "whale_orderflow_breakout",
            _permissive_whale_orderflow_config(),
            _whale_orderflow_context,
        ),
    ],
)
@pytest.mark.asyncio()
async def test_hybrid_strategy_evaluate_wraps_signal_into_evaluation(
    strategy_cls: type,
    strategy_name: str,
    hybrid_config: HybridStrategyConfig,
    context_factory: Callable[..., StrategyContext],
    make_context,
    make_feature,
    mock_event_bus,
    mock_scheduler,
) -> None:
    strategy = _build_strategy(
        strategy_cls=strategy_cls,
        strategy_name=strategy_name,
        hybrid_config=hybrid_config,
        mock_event_bus=mock_event_bus,
        mock_scheduler=mock_scheduler,
    )
    context = context_factory(make_context, make_feature)

    evaluation = await strategy.evaluate(context)

    evaluation.validate()

    assert evaluation.strategy_name == strategy_name
    assert evaluation.symbol == "BTCUSDT"
    assert evaluation.signal is not None
    assert evaluation.passed
    assert evaluation.signal.status is not SignalStatus.REJECTED

    assert not mock_event_bus.topic_emitted("signal.generated")
    assert not mock_event_bus.nowait_topic_emitted("signal.generated")