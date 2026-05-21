# trading_system/strategy/presets.py

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypeAlias, cast

from core.event_bus import EventBus
from core.scheduler import Scheduler

from strategy.base import BaseStrategy
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
    EntryType,
    MarketRegime,
    PresetMode,
    StrategyCategory,
    Timeframe,
)
from strategy.exceptions import StrategyConfigError, StrategyRegistrationError
from strategy.registry import StrategyRegistry


Timeframes: TypeAlias = tuple[Timeframe, ...]
StrategyNames: TypeAlias = tuple[str, ...]

OverrideValue: TypeAlias = (
    bool
    | int
    | float
    | str
    | list[str]
    | list[Timeframe]
    | list[MarketRegime]
    | dict[str, object]
)
StrategyOverride: TypeAlias = dict[str, OverrideValue]
StrategyFactory: TypeAlias = Callable[..., BaseStrategy]


# =============================================================================
# Timeframe helpers
# =============================================================================


def _tf(*items: Timeframe) -> Timeframes:
    return tuple(items)


def _tf_join(*groups: Timeframe | Sequence[Timeframe]) -> Timeframes:
    result: list[Timeframe] = []
    seen: set[Timeframe] = set()

    for group in groups:
        if isinstance(group, Timeframe):
            items: Sequence[Timeframe] = (group,)
        else:
            items = group

        for timeframe in items:
            if not isinstance(timeframe, Timeframe):
                raise StrategyConfigError(
                    f"timeframe must be Timeframe, got {type(timeframe)!r}"
                )

            if timeframe in seen:
                continue

            seen.add(timeframe)
            result.append(timeframe)

    if not result:
        raise StrategyConfigError("timeframe group cannot be empty")

    return tuple(result)


TF_SCALP: Timeframes = _tf(Timeframe.M1, Timeframe.M3, Timeframe.M5)
TF_SCALP_INTRADAY: Timeframes = _tf(
    Timeframe.M1,
    Timeframe.M3,
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.M30,
    Timeframe.H1,
)
TF_INTRADAY: Timeframes = _tf(
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.M30,
    Timeframe.H1,
)
TF_SWING: Timeframes = _tf(Timeframe.H1, Timeframe.H4, Timeframe.D1)
TF_FAST: Timeframes = _tf(Timeframe.M1, Timeframe.M3)
TF_MID: Timeframes = _tf(Timeframe.M5, Timeframe.M15)
TF_FUNDING: Timeframes = _tf(Timeframe.H1, Timeframe.H4, Timeframe.D1)
TF_SPREADS: Timeframes = _tf(
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.M30,
    Timeframe.H1,
)


# =============================================================================
# Strategy catalog
# =============================================================================


@dataclass(frozen=True, slots=True)
class StrategyCatalogEntry:
    """
    Static preset/catalog metadata for one concrete strategy.

    This class is configuration-only:
    - no EventBus subscriptions;
    - no strategy evaluation;
    - no signal.generated emit;
    - no risk/execution calls.
    """

    name: str
    category: StrategyCategory
    default_timeframes: Timeframes
    weight: float = 1.0
    priority: int = 100
    tags: StrategyNames = ()
    feature_hints: StrategyNames = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise StrategyConfigError("StrategyCatalogEntry.name cannot be empty")

        if not self.default_timeframes:
            raise StrategyConfigError(
                f"StrategyCatalogEntry.default_timeframes for '{self.name}' cannot be empty"
            )

        if self.weight < 0:
            raise StrategyConfigError(
                f"StrategyCatalogEntry.weight for '{self.name}' must be >= 0"
            )

        if self.priority < 0:
            raise StrategyConfigError(
                f"StrategyCatalogEntry.priority for '{self.name}' must be >= 0"
            )

        if any(not tag.strip() for tag in self.tags):
            raise StrategyConfigError(
                f"StrategyCatalogEntry.tags for '{self.name}' cannot contain empty tags"
            )

        if any(not feature.strip() for feature in self.feature_hints):
            raise StrategyConfigError(
                f"StrategyCatalogEntry.feature_hints for '{self.name}' cannot contain empty names"
            )


STRATEGY_CATALOG: dict[str, StrategyCatalogEntry] = {
    # -------------------------------------------------------------------------
    # orderflow
    # -------------------------------------------------------------------------
    "cvd_divergence": StrategyCatalogEntry(
        name="cvd_divergence",
        category=StrategyCategory.ORDERFLOW,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=1.10,
        priority=20,
        tags=("orderflow", "cvd", "divergence", "reversal", "futures"),
        feature_hints=(
            "cvd",
            "cvd_delta",
            "cvd_divergence",
            "volume_delta",
            "orderflow_imbalance",
        ),
    ),
    "orderflow_continuation": StrategyCatalogEntry(
        name="orderflow_continuation",
        category=StrategyCategory.ORDERFLOW,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=1.05,
        priority=35,
        tags=("orderflow", "continuation", "momentum", "futures"),
        feature_hints=(
            "orderflow_trend",
            "volume_delta",
            "aggressor_imbalance",
            "buy_sell_pressure",
        ),
    ),
    "orderflow_reversal": StrategyCatalogEntry(
        name="orderflow_reversal",
        category=StrategyCategory.ORDERFLOW,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=1.08,
        priority=25,
        tags=("orderflow", "reversal", "absorption", "futures"),
        feature_hints=(
            "orderflow_reversal",
            "absorption",
            "delta_exhaustion",
            "volume_delta",
        ),
    ),

    # -------------------------------------------------------------------------
    # price_action
    # -------------------------------------------------------------------------
    "market_structure": StrategyCatalogEntry(
        name="market_structure",
        category=StrategyCategory.PRICE_ACTION,
        default_timeframes=_tf_join(TF_SCALP_INTRADAY, Timeframe.H4),
        weight=0.95,
        priority=45,
        tags=("price_action", "market_structure", "trend", "futures"),
        feature_hints=(
            "market_structure",
            "bos",
            "choch",
            "swing_high",
            "swing_low",
        ),
    ),
    "fvg_reaction": StrategyCatalogEntry(
        name="fvg_reaction",
        category=StrategyCategory.PRICE_ACTION,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=0.90,
        priority=50,
        tags=("price_action", "fvg", "reaction", "futures"),
        feature_hints=(
            "fvg",
            "fair_value_gap",
            "fvg_reaction",
            "imbalance_zone",
        ),
    ),
    "support_resistance_reaction": StrategyCatalogEntry(
        name="support_resistance_reaction",
        category=StrategyCategory.PRICE_ACTION,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=0.92,
        priority=48,
        tags=("price_action", "support_resistance", "reaction", "futures"),
        feature_hints=(
            "support",
            "resistance",
            "sr_level",
            "level_reaction",
            "rejection",
        ),
    ),
    "trend_continuation": StrategyCatalogEntry(
        name="trend_continuation",
        category=StrategyCategory.PRICE_ACTION,
        default_timeframes=_tf_join(TF_SCALP_INTRADAY, Timeframe.H4),
        weight=1.00,
        priority=40,
        tags=("price_action", "trend", "continuation", "futures"),
        feature_hints=(
            "trend",
            "trend_strength",
            "pullback",
            "market_structure",
        ),
    ),

    # -------------------------------------------------------------------------
    # open_interest
    # -------------------------------------------------------------------------
    "oi_divergence": StrategyCatalogEntry(
        name="oi_divergence",
        category=StrategyCategory.OPEN_INTEREST,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=1.00,
        priority=36,
        tags=("open_interest", "oi", "divergence", "reversal", "futures"),
        feature_hints=(
            "open_interest",
            "oi_delta",
            "oi_divergence",
            "price_oi_divergence",
        ),
    ),
    "oi_breakout_confirmation": StrategyCatalogEntry(
        name="oi_breakout_confirmation",
        category=StrategyCategory.OPEN_INTEREST,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=1.02,
        priority=38,
        tags=("open_interest", "oi", "breakout", "confirmation", "futures"),
        feature_hints=(
            "open_interest",
            "oi_expansion",
            "breakout_confirmation",
            "price_oi_confirmation",
        ),
    ),
    "oi_anomaly": StrategyCatalogEntry(
        name="oi_anomaly",
        category=StrategyCategory.OPEN_INTEREST,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=0.88,
        priority=58,
        tags=("open_interest", "oi", "anomaly", "futures"),
        feature_hints=(
            "open_interest",
            "oi_anomaly",
            "oi_zscore",
            "oi_spike",
        ),
    ),
    "oi_capitulation": StrategyCatalogEntry(
        name="oi_capitulation",
        category=StrategyCategory.OPEN_INTEREST,
        default_timeframes=_tf_join(TF_MID, Timeframe.M30, Timeframe.H1),
        weight=1.00,
        priority=34,
        tags=("open_interest", "oi", "capitulation", "reversal", "futures"),
        feature_hints=(
            "open_interest",
            "oi_flush",
            "oi_capitulation",
            "long_short_flush",
        ),
    ),


    # -------------------------------------------------------------------------
    # liquidity
    # -------------------------------------------------------------------------
    "liquidity_sweep": StrategyCatalogEntry(
        name="liquidity_sweep",
        category=StrategyCategory.LIQUIDITY,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=1.02,
        priority=24,
        tags=("liquidity", "sweep", "reversal", "futures"),
        feature_hints=(
            "liquidity_sweep",
            "sweep_score",
            "stop_run",
            "liquidity_pool",
        ),
    ),
    "stop_hunt_reversal": StrategyCatalogEntry(
        name="stop_hunt_reversal",
        category=StrategyCategory.LIQUIDITY,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=1.04,
        priority=22,
        tags=("liquidity", "stop_hunt", "reversal", "futures"),
        feature_hints=(
            "stop_hunt",
            "liquidity_sweep",
            "reversal_confirmation",
            "false_breakout",
        ),
    ),
    "equal_high_low": StrategyCatalogEntry(
        name="equal_high_low",
        category=StrategyCategory.LIQUIDITY,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=0.92,
        priority=48,
        tags=("liquidity", "equal_high_low", "pool", "futures"),
        feature_hints=(
            "equal_high",
            "equal_low",
            "liquidity_pool",
            "resting_liquidity",
        ),
    ),

    # -------------------------------------------------------------------------
    # liquidations
    # -------------------------------------------------------------------------
    "liquidation_cascade": StrategyCatalogEntry(
        name="liquidation_cascade",
        category=StrategyCategory.LIQUIDATIONS,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=1.06,
        priority=21,
        tags=("liquidations", "cascade", "momentum", "futures"),
        feature_hints=(
            "liquidation_cascade",
            "forced_flow",
            "liquidation_volume",
            "cascade_score",
        ),
    ),
    "squeeze_reversal": StrategyCatalogEntry(
        name="squeeze_reversal",
        category=StrategyCategory.LIQUIDATIONS,
        default_timeframes=_tf_join(TF_MID, Timeframe.M30, Timeframe.H1),
        weight=1.03,
        priority=25,
        tags=("liquidations", "squeeze", "reversal", "futures"),
        feature_hints=(
            "squeeze_reversal",
            "liquidation_squeeze",
            "forced_flow_exhaustion",
            "reversal_pressure",
        ),
    ),

    # -------------------------------------------------------------------------
    # funding
    # -------------------------------------------------------------------------
    "funding_extreme_reversal": StrategyCatalogEntry(
        name="funding_extreme_reversal",
        category=StrategyCategory.FUNDING,
        default_timeframes=TF_FUNDING,
        weight=0.96,
        priority=40,
        tags=("funding", "extreme", "reversal", "futures"),
        feature_hints=(
            "funding_rate",
            "funding_extreme",
            "funding_zscore",
            "funding_pressure",
        ),
    ),
    "funding_divergence": StrategyCatalogEntry(
        name="funding_divergence",
        category=StrategyCategory.FUNDING,
        default_timeframes=TF_FUNDING,
        weight=0.94,
        priority=42,
        tags=("funding", "divergence", "reversal", "futures"),
        feature_hints=(
            "funding_rate",
            "funding_divergence",
            "funding_bias",
            "price_funding_divergence",
        ),
    ),

    # -------------------------------------------------------------------------
    # spoofing
    # -------------------------------------------------------------------------
    "spoofing_reversal": StrategyCatalogEntry(
        name="spoofing_reversal",
        category=StrategyCategory.SPOOFING,
        default_timeframes=_tf_join(TF_FAST, Timeframe.M5),
        weight=0.92,
        priority=30,
        tags=("spoofing", "reversal", "orderbook", "futures"),
        feature_hints=(
            "spoofing",
            "spoofing_score",
            "fake_liquidity",
            "orderbook_imbalance",
        ),
    ),
    "fake_liquidity_trap": StrategyCatalogEntry(
        name="fake_liquidity_trap",
        category=StrategyCategory.SPOOFING,
        default_timeframes=_tf_join(TF_FAST, Timeframe.M5),
        weight=0.95,
        priority=28,
        tags=("spoofing", "fake_liquidity", "trap", "futures"),
        feature_hints=(
            "fake_liquidity",
            "liquidity_pull",
            "spoofing_trap",
            "orderbook_depth_shift",
        ),
    ),
    "order_pull_reversal": StrategyCatalogEntry(
        name="order_pull_reversal",
        category=StrategyCategory.SPOOFING,
        default_timeframes=_tf_join(TF_FAST, Timeframe.M5),
        weight=0.94,
        priority=29,
        tags=("spoofing", "order_pull", "reversal", "futures"),
        feature_hints=(
            "order_pull",
            "pulling_orders",
            "orderbook_liquidity_drop",
            "spoofing",
        ),
    ),
    "pressure_bluff_reversal": StrategyCatalogEntry(
        name="pressure_bluff_reversal",
        category=StrategyCategory.SPOOFING,
        default_timeframes=_tf_join(TF_FAST, Timeframe.M5),
        weight=0.90,
        priority=33,
        tags=("spoofing", "pressure_bluff", "reversal", "futures"),
        feature_hints=(
            "pressure_bluff",
            "fake_pressure",
            "spoofing_pressure",
            "orderbook_pressure",
        ),
    ),
    "layering_trap": StrategyCatalogEntry(
        name="layering_trap",
        category=StrategyCategory.SPOOFING,
        default_timeframes=_tf_join(TF_FAST, Timeframe.M5),
        weight=0.88,
        priority=42,
        tags=("spoofing", "layering", "trap", "futures"),
        feature_hints=(
            "layering",
            "layering_score",
            "spoofing_layers",
            "orderbook_layers",
        ),
    ),
    "spoofing_absorption_reversal": StrategyCatalogEntry(
        name="spoofing_absorption_reversal",
        category=StrategyCategory.SPOOFING,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=0.98,
        priority=24,
        tags=("spoofing", "absorption", "reversal", "futures"),
        feature_hints=(
            "spoofing",
            "absorption",
            "fake_liquidity_absorption",
            "orderflow_absorption",
        ),
    ),
    "composite_spoofing": StrategyCatalogEntry(
        name="composite_spoofing",
        category=StrategyCategory.SPOOFING,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=1.00,
        priority=22,
        tags=("spoofing", "composite", "orderbook", "futures"),
        feature_hints=(
            "spoofing_score",
            "fake_liquidity",
            "layering",
            "pressure_bluff",
            "order_pull",
        ),
    ),

    # -------------------------------------------------------------------------
    # spreads
    # -------------------------------------------------------------------------
    "spot_futures_basis": StrategyCatalogEntry(
        name="spot_futures_basis",
        category=StrategyCategory.SPREADS,
        default_timeframes=TF_SPREADS,
        weight=0.82,
        priority=70,
        tags=("spreads", "basis", "spot_futures", "futures"),
        feature_hints=(
            "basis",
            "spot_futures_basis",
            "basis_zscore",
            "spread_bps",
        ),
        metadata={
            "note": (
                "Kept intentionally for spread/basis workflows even though the "
                "global trading system is futures-first."
            )
        },
    ),
    "cross_exchange_arb": StrategyCatalogEntry(
        name="cross_exchange_arb",
        category=StrategyCategory.SPREADS,
        default_timeframes=TF_SPREADS,
        weight=0.78,
        priority=75,
        tags=("spreads", "cross_exchange", "arbitrage", "futures"),
        feature_hints=(
            "cross_exchange_spread",
            "arb_opportunity",
            "spread_bps",
            "execution_cost",
        ),
    ),
    "funding_adjusted_basis": StrategyCatalogEntry(
        name="funding_adjusted_basis",
        category=StrategyCategory.SPREADS,
        default_timeframes=TF_SPREADS,
        weight=0.86,
        priority=66,
        tags=("spreads", "basis", "funding_adjusted", "futures"),
        feature_hints=(
            "funding_adjusted_basis",
            "basis",
            "funding_rate",
            "expected_funding",
        ),
    ),
    "spread_mean_reversion": StrategyCatalogEntry(
        name="spread_mean_reversion",
        category=StrategyCategory.SPREADS,
        default_timeframes=TF_SPREADS,
        weight=0.88,
        priority=64,
        tags=("spreads", "mean_reversion", "basis", "futures"),
        feature_hints=(
            "spread_zscore",
            "basis_zscore",
            "mean_reversion_score",
            "spread_bps",
        ),
    ),
    "spread_momentum": StrategyCatalogEntry(
        name="spread_momentum",
        category=StrategyCategory.SPREADS,
        default_timeframes=TF_SPREADS,
        weight=0.84,
        priority=68,
        tags=("spreads", "momentum", "basis", "futures"),
        feature_hints=(
            "spread_momentum",
            "basis_momentum",
            "spread_trend",
            "spread_bps",
        ),
    ),

    # -------------------------------------------------------------------------
    # whales
    # -------------------------------------------------------------------------
    "whale_absorption": StrategyCatalogEntry(
        name="whale_absorption",
        category=StrategyCategory.WHALES,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=1.00,
        priority=26,
        tags=("whales", "absorption", "reversal", "futures"),
        feature_hints=(
            "whale_absorption",
            "large_trade_absorption",
            "large_trade",
            "absorption",
        ),
    ),
    "whale_breakout": StrategyCatalogEntry(
        name="whale_breakout",
        category=StrategyCategory.WHALES,
        default_timeframes=_tf_join(TF_MID, Timeframe.M30, Timeframe.H1),
        weight=1.00,
        priority=32,
        tags=("whales", "breakout", "momentum", "futures"),
        feature_hints=(
            "whale_breakout",
            "large_trade_breakout",
            "large_trade",
            "breakout",
        ),
    ),
    "whale_accumulation": StrategyCatalogEntry(
        name="whale_accumulation",
        category=StrategyCategory.WHALES,
        default_timeframes=_tf_join(TF_INTRADAY, Timeframe.H4),
        weight=0.96,
        priority=44,
        tags=("whales", "accumulation", "trend", "futures"),
        feature_hints=(
            "whale_accumulation",
            "large_trade_accumulation",
            "accumulation",
            "smart_money_flow",
        ),
    ),
    "whale_distribution": StrategyCatalogEntry(
        name="whale_distribution",
        category=StrategyCategory.WHALES,
        default_timeframes=_tf_join(TF_INTRADAY, Timeframe.H4),
        weight=0.96,
        priority=44,
        tags=("whales", "distribution", "trend", "futures"),
        feature_hints=(
            "whale_distribution",
            "large_trade_distribution",
            "distribution",
            "smart_money_flow",
        ),
    ),
    "whale_liquidation_reversal": StrategyCatalogEntry(
        name="whale_liquidation_reversal",
        category=StrategyCategory.WHALES,
        default_timeframes=_tf_join(TF_MID, Timeframe.M30),
        weight=1.04,
        priority=23,
        tags=("whales", "liquidation", "reversal", "futures"),
        feature_hints=(
            "whale_liquidation",
            "liquidation_reversal",
            "large_trade",
            "forced_flow",
        ),
    ),

    # -------------------------------------------------------------------------
    # hybrid
    # -------------------------------------------------------------------------
    "confluence": StrategyCatalogEntry(
        name="confluence",
        category=StrategyCategory.HYBRID,
        default_timeframes=TF_INTRADAY,
        weight=1.20,
        priority=10,
        tags=("hybrid", "confluence", "multi_factor", "futures"),
        feature_hints=(
            "confluence_score",
            "multi_factor_score",
            "strategy_agreement",
        ),
    ),
    "mean_reversion_stack": StrategyCatalogEntry(
        name="mean_reversion_stack",
        category=StrategyCategory.HYBRID,
        default_timeframes=_tf_join(TF_MID, Timeframe.M30, Timeframe.H1),
        weight=1.16,
        priority=14,
        tags=("hybrid", "mean_reversion", "stack", "futures"),
        feature_hints=(
            "mean_reversion_score",
            "oi_divergence",
            "spread_zscore",
            "absorption",
        ),
    ),
    "trend_stack": StrategyCatalogEntry(
        name="trend_stack",
        category=StrategyCategory.HYBRID,
        default_timeframes=TF_INTRADAY,
        weight=1.16,
        priority=14,
        tags=("hybrid", "trend", "stack", "futures"),
        feature_hints=(
            "trend_strength",
            "market_structure",
            "oi_breakout_confirmation",
            "orderflow_trend",
        ),
    ),
    "liquidation_whale": StrategyCatalogEntry(
        name="liquidation_whale",
        category=StrategyCategory.HYBRID,
        default_timeframes=_tf_join(TF_MID, Timeframe.M30),
        weight=1.14,
        priority=16,
        tags=("hybrid", "liquidation", "whale", "reversal", "futures"),
        feature_hints=(
            "liquidation",
            "whale_liquidation",
            "large_trade",
            "forced_flow",
        ),
    ),
    "liquidity_orderflow_reversal": StrategyCatalogEntry(
        name="liquidity_orderflow_reversal",
        category=StrategyCategory.HYBRID,
        default_timeframes=TF_SCALP_INTRADAY,
        weight=1.18,
        priority=12,
        tags=("hybrid", "liquidity", "orderflow", "reversal", "futures"),
        feature_hints=(
            "liquidity_sweep",
            "orderflow_reversal",
            "cvd_divergence",
            "absorption",
        ),
    ),
    "oi_funding_squeeze": StrategyCatalogEntry(
        name="oi_funding_squeeze",
        category=StrategyCategory.HYBRID,
        default_timeframes=_tf_join(TF_MID, Timeframe.M30, Timeframe.H1),
        weight=1.12,
        priority=18,
        tags=("hybrid", "open_interest", "funding", "squeeze", "futures"),
        feature_hints=(
            "open_interest",
            "funding_rate",
            "squeeze_score",
            "oi_expansion",
        ),
    ),
    "whale_orderflow_breakout": StrategyCatalogEntry(
        name="whale_orderflow_breakout",
        category=StrategyCategory.HYBRID,
        default_timeframes=_tf_join(TF_MID, Timeframe.M30, Timeframe.H1),
        weight=1.14,
        priority=15,
        tags=("hybrid", "whale", "orderflow", "breakout", "futures"),
        feature_hints=(
            "whale_breakout",
            "orderflow_continuation",
            "large_trade",
            "volume_delta",
        ),
    ),
}


ORDERFLOW_STRATEGIES: StrategyNames = (
    "cvd_divergence",
    "orderflow_continuation",
    "orderflow_reversal",
)

PRICE_ACTION_STRATEGIES: StrategyNames = (
    "market_structure",
    "fvg_reaction",
    "support_resistance_reaction",
    "trend_continuation",
)

OPEN_INTEREST_STRATEGIES: StrategyNames = (
    "oi_divergence",
    "oi_breakout_confirmation",
    "oi_anomaly",
    "oi_capitulation",
)

LIQUIDITY_STRATEGIES: StrategyNames = (
    "liquidity_sweep",
    "stop_hunt_reversal",
    "equal_high_low",
)

LIQUIDATIONS_STRATEGIES: StrategyNames = (
    "liquidation_cascade",
    "squeeze_reversal",
)

FUNDING_STRATEGIES: StrategyNames = (
    "funding_extreme_reversal",
    "funding_divergence",
)

SPOOFING_STRATEGIES: StrategyNames = (
    "spoofing_reversal",
    "fake_liquidity_trap",
    "order_pull_reversal",
    "pressure_bluff_reversal",
    "layering_trap",
    "spoofing_absorption_reversal",
    "composite_spoofing",
)

SPREADS_STRATEGIES: StrategyNames = (
    "spot_futures_basis",
    "cross_exchange_arb",
    "funding_adjusted_basis",
    "spread_mean_reversion",
    "spread_momentum",
)

WHALES_STRATEGIES: StrategyNames = (
    "whale_absorption",
    "whale_breakout",
    "whale_accumulation",
    "whale_distribution",
    "whale_liquidation_reversal",
)

HYBRID_STRATEGIES: StrategyNames = (
    "confluence",
    "mean_reversion_stack",
    "trend_stack",
    "liquidation_whale",
    "liquidity_orderflow_reversal",
    "oi_funding_squeeze",
    "whale_orderflow_breakout",
)

ALL_STRATEGY_NAMES: StrategyNames = tuple(STRATEGY_CATALOG.keys())
DEFAULT_STRATEGIES: StrategyNames = ALL_STRATEGY_NAMES


SCALPING_STRATEGIES: StrategyNames = (
    "cvd_divergence",
    "orderflow_reversal",
    "orderflow_continuation",
    "liquidity_sweep",
    "stop_hunt_reversal",
    "liquidation_cascade",
    "spoofing_reversal",
    "fake_liquidity_trap",
    "order_pull_reversal",
    "pressure_bluff_reversal",
    "layering_trap",
    "spoofing_absorption_reversal",
    "composite_spoofing",
    "whale_absorption",
    "whale_breakout",
    "liquidity_orderflow_reversal",
)

INTRADAY_STRATEGIES: StrategyNames = (
    "market_structure",
    "fvg_reaction",
    "support_resistance_reaction",
    "trend_continuation",
    "cvd_divergence",
    "orderflow_continuation",
    "orderflow_reversal",
    "oi_divergence",
    "oi_breakout_confirmation",
    "oi_anomaly",
    "oi_capitulation",
    "liquidity_sweep",
    "stop_hunt_reversal",
    "equal_high_low",
    "liquidation_cascade",
    "squeeze_reversal",
    "funding_extreme_reversal",
    "funding_divergence",
    "whale_accumulation",
    "whale_distribution",
    "whale_breakout",
    "spread_momentum",
    "funding_adjusted_basis",
    "confluence",
    "trend_stack",
    "mean_reversion_stack",
    "whale_orderflow_breakout",
)

SWING_STRATEGIES: StrategyNames = (
    "market_structure",
    "trend_continuation",
    "support_resistance_reaction",
    "oi_breakout_confirmation",
    "oi_divergence",
    "oi_anomaly",
    "oi_capitulation",
    "equal_high_low",
    "squeeze_reversal",
    "funding_extreme_reversal",
    "funding_divergence",
    "whale_accumulation",
    "whale_distribution",
    "funding_adjusted_basis",
    "spread_mean_reversion",
    "spread_momentum",
    "confluence",
    "trend_stack",
    "mean_reversion_stack",
)

LIQUIDITY_REVERSAL_STRATEGIES: StrategyNames = (
    "orderflow_reversal",
    "cvd_divergence",
    "spoofing_reversal",
    "fake_liquidity_trap",
    "order_pull_reversal",
    "pressure_bluff_reversal",
    "spoofing_absorption_reversal",
    "composite_spoofing",
    "whale_absorption",
    "whale_liquidation_reversal",
    "liquidity_sweep",
    "stop_hunt_reversal",
    "equal_high_low",
    "liquidation_cascade",
    "squeeze_reversal",
    "oi_divergence",
    "oi_capitulation",
    "liquidity_orderflow_reversal",
    "liquidation_whale",
)

MEAN_REVERSION_STRATEGIES: StrategyNames = (
    "oi_divergence",
    "oi_capitulation",
    "funding_extreme_reversal",
    "funding_divergence",
    "squeeze_reversal",
    "liquidity_sweep",
    "stop_hunt_reversal",
    "spread_mean_reversion",
    "funding_adjusted_basis",
    "whale_absorption",
    "whale_liquidation_reversal",
    "spoofing_absorption_reversal",
    "support_resistance_reaction",
    "mean_reversion_stack",
    "oi_funding_squeeze",
)

TREND_STACK_STRATEGIES: StrategyNames = (
    "trend_continuation",
    "market_structure",
    "orderflow_continuation",
    "oi_breakout_confirmation",
    "liquidation_cascade",
    "whale_breakout",
    "spread_momentum",
    "trend_stack",
    "whale_orderflow_breakout",
    "confluence",
)

SPREADS_PRESET_STRATEGIES: StrategyNames = SPREADS_STRATEGIES


# =============================================================================
# Pure helpers
# =============================================================================


def _dedupe_str(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for item in items:
        value = item.strip()
        if not value or value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def _clean_symbols(symbols: Sequence[str] | None) -> list[str]:
    return _dedupe_str(symbols or [])


def _normalize_names(names: Iterable[str]) -> list[str]:
    normalized = _dedupe_str(names)

    unknown = [name for name in normalized if name not in STRATEGY_CATALOG]
    if unknown:
        raise StrategyConfigError(
            f"Unknown strategy names in preset: {', '.join(sorted(unknown))}"
        )

    return normalized


def _normalize_timeframes(
    timeframes: Sequence[Timeframe] | None,
    fallback: Sequence[Timeframe],
) -> list[Timeframe]:
    source = list(timeframes or fallback)

    result: list[Timeframe] = []
    seen: set[Timeframe] = set()

    for timeframe in source:
        if not isinstance(timeframe, Timeframe):
            raise StrategyConfigError(
                f"timeframe must be Timeframe, got {type(timeframe)!r}"
            )

        if timeframe in seen:
            continue

        seen.add(timeframe)
        result.append(timeframe)

    if not result:
        raise StrategyConfigError("preset timeframes cannot be empty")

    return result


def _normalize_regimes(
    regimes: Sequence[MarketRegime] | None,
) -> list[MarketRegime]:
    source = list(regimes or [MarketRegime.UNKNOWN])

    result: list[MarketRegime] = []
    seen: set[MarketRegime] = set()

    for regime in source:
        if not isinstance(regime, MarketRegime):
            raise StrategyConfigError(
                f"regime must be MarketRegime, got {type(regime)!r}"
            )

        if regime in seen:
            continue

        seen.add(regime)
        result.append(regime)

    if not result:
        raise StrategyConfigError("preset regimes cannot be empty")

    return result


def _runtime(
    *,
    symbols: Sequence[str] | None,
    timeframes: Sequence[Timeframe],
    allowed_regimes: Sequence[MarketRegime] | None = None,
    enabled: bool = True,
    cooldown_seconds: int,
    max_signal_age_seconds: int,
    min_confidence: float,
    min_score: float,
) -> StrategyRuntimeConfig:
    runtime = StrategyRuntimeConfig(
        enabled=enabled,
        symbols=_clean_symbols(symbols),
        timeframes=_normalize_timeframes(timeframes, fallback=timeframes),
        allowed_regimes=_normalize_regimes(allowed_regimes),
        cooldown_seconds=cooldown_seconds,
        max_signal_age_seconds=max_signal_age_seconds,
        min_confidence=min_confidence,
        min_score=min_score,
    )
    runtime.validate()
    return runtime


def _routing_config(
    *,
    stale_feature_threshold_seconds: int,
    reevaluate_on_any_update: bool = False,
    route_hybrid_on_domain_signal: bool = True,
    allow_partial_context: bool = True,
) -> RoutingConfig:
    routing = RoutingConfig(
        reevaluate_on_any_update=reevaluate_on_any_update,
        route_hybrid_on_domain_signal=route_hybrid_on_domain_signal,
        allow_partial_context=allow_partial_context,
        stale_feature_threshold_seconds=stale_feature_threshold_seconds,
        event_to_categories={
            "analytics.orderflow": [StrategyCategory.ORDERFLOW],
            "analytics.price_action": [StrategyCategory.PRICE_ACTION],
            "analytics.open_interest": [StrategyCategory.OPEN_INTEREST],
            "analytics.oi": [StrategyCategory.OPEN_INTEREST],
            "analytics.spoofing": [StrategyCategory.SPOOFING],
            "analytics.spreads": [StrategyCategory.SPREADS],
            "analytics.spread": [StrategyCategory.SPREADS],
            "analytics.whales": [StrategyCategory.WHALES],
            "analytics.whale": [StrategyCategory.WHALES],
            "analytics.hybrid": [StrategyCategory.HYBRID],
            "analytics.confluence": [StrategyCategory.HYBRID],
        },
    )
    routing.validate()
    return routing


def _category_weights(
    overrides: Mapping[StrategyCategory, float] | None = None,
) -> dict[StrategyCategory, float]:
    values: dict[StrategyCategory, float] = {
        StrategyCategory.ORDERFLOW: 1.00,
        StrategyCategory.LIQUIDITY: 1.00,
        StrategyCategory.PRICE_ACTION: 0.90,
        StrategyCategory.LIQUIDATIONS: 0.95,
        StrategyCategory.WHALES: 0.95,
        StrategyCategory.SPOOFING: 0.90,
        StrategyCategory.SPREADS: 0.80,
        StrategyCategory.FUNDING: 0.80,
        StrategyCategory.OPEN_INTEREST: 0.90,
        StrategyCategory.HYBRID: 1.20,
    }

    if overrides:
        values.update(dict(overrides))

    return values


def _regime_adjustments(
    overrides: Mapping[MarketRegime, float] | None = None,
) -> dict[MarketRegime, float]:
    values: dict[MarketRegime, float] = {
        MarketRegime.TRENDING_UP: 1.00,
        MarketRegime.TRENDING_DOWN: 1.00,
        MarketRegime.RANGING: 0.95,
        MarketRegime.BREAKOUT: 1.10,
        MarketRegime.SQUEEZE: 1.05,
        MarketRegime.HIGH_VOLATILITY: 0.85,
        MarketRegime.LOW_VOLATILITY: 0.90,
        MarketRegime.NEWS_DRIVEN: 0.65,
        MarketRegime.ILLIQUID: 0.50,
        MarketRegime.RISK_OFF: 0.60,
        MarketRegime.UNKNOWN: 1.00,
    }

    if overrides:
        values.update(dict(overrides))

    return values


# =============================================================================
# Typed override helpers
# =============================================================================


def _override_bool(
    override: Mapping[str, OverrideValue],
    key: str,
    default: bool,
) -> bool:
    value = override.get(key)
    if isinstance(value, bool):
        return value
    return default


def _override_int(
    override: Mapping[str, OverrideValue],
    key: str,
    default: int,
) -> int:
    value = override.get(key)

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default

    return default


def _override_float(
    override: Mapping[str, OverrideValue],
    key: str,
    default: float,
) -> float:
    value = override.get(key)

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default

    return default


def _override_timeframes(
    override: Mapping[str, OverrideValue],
    key: str,
    default: Sequence[Timeframe] | None,
) -> Sequence[Timeframe] | None:
    value = override.get(key)

    if value is None:
        return default

    if isinstance(value, list) and all(isinstance(item, Timeframe) for item in value):
        return cast(list[Timeframe], value)

    return default


def _override_regimes(
    override: Mapping[str, OverrideValue],
    key: str,
    default: Sequence[MarketRegime] | None,
) -> Sequence[MarketRegime] | None:
    value = override.get(key)

    if value is None:
        return default

    if isinstance(value, list) and all(isinstance(item, MarketRegime) for item in value):
        return cast(list[MarketRegime], value)

    return default


def _override_tags(
    override: Mapping[str, OverrideValue],
    key: str,
) -> Sequence[str] | None:
    value = override.get(key)

    if value is None:
        return None

    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return cast(list[str], value)

    return None


def _override_metadata(
    override: Mapping[str, OverrideValue],
    key: str,
) -> Mapping[str, object] | None:
    value = override.get(key)

    if isinstance(value, dict):
        return cast(dict[str, object], value)

    return None


def _definition_from_catalog(
    entry: StrategyCatalogEntry,
    *,
    symbols: Sequence[str] | None,
    timeframes: Sequence[Timeframe] | None,
    allowed_regimes: Sequence[MarketRegime] | None,
    enabled: bool,
    cooldown_seconds: int,
    max_signal_age_seconds: int,
    min_confidence: float,
    min_score: float,
    weight_multiplier: float,
    priority_offset: int,
    use_required_features: bool,
    extra_tags: Sequence[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> StrategyDefinitionConfig:
    entry.validate()

    resolved_timeframes = _normalize_timeframes(
        timeframes=timeframes,
        fallback=entry.default_timeframes,
    )

    runtime = _runtime(
        symbols=symbols,
        timeframes=resolved_timeframes,
        allowed_regimes=allowed_regimes,
        enabled=enabled,
        cooldown_seconds=cooldown_seconds,
        max_signal_age_seconds=max_signal_age_seconds,
        min_confidence=min_confidence,
        min_score=min_score,
    )

    tags = _dedupe_str([*entry.tags, *(extra_tags or [])])

    definition_metadata: dict[str, object] = {
        "preset_catalog": True,
        "feature_hints": list(entry.feature_hints),
        "default_timeframes": [
            timeframe.value for timeframe in entry.default_timeframes
        ],
        **dict(entry.metadata),
        **dict(metadata or {}),
    }

    definition = StrategyDefinitionConfig(
        name=entry.name,
        category=entry.category,
        runtime=runtime,
        required_features=set(entry.feature_hints) if use_required_features else set(),
        weight=entry.weight * weight_multiplier,
        priority=max(0, entry.priority + priority_offset),
        tags=tags,
        metadata=definition_metadata,
    )
    definition.validate()
    return definition


# =============================================================================
# Preset DTO + Builder
# =============================================================================


@dataclass(slots=True)
class StrategyPreset:
    """
    Wrapper around StrategyConfig.

    This object is safe to use as preset template. Use to_config(copy=True)
    before passing config into StrategyEngine if runtime mutation is possible.
    """

    name: str
    mode: PresetMode
    description: str
    config: StrategyConfig
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def enabled_strategy_names(self) -> list[str]:
        return list(self.config.preset.enabled_strategy_names)

    @property
    def strategy_definitions(self) -> dict[str, StrategyDefinitionConfig]:
        return dict(self.config.strategies)

    def validate(self) -> None:
        if not self.name.strip():
            raise StrategyConfigError("StrategyPreset.name cannot be empty")

        if not self.description.strip():
            raise StrategyConfigError("StrategyPreset.description cannot be empty")

        if not isinstance(self.mode, PresetMode):
            raise StrategyConfigError("StrategyPreset.mode must be PresetMode")

        if self.config.preset.mode is not self.mode:
            raise StrategyConfigError(
                f"StrategyPreset.mode '{self.mode.value}' does not match "
                f"StrategyConfig.preset.mode '{self.config.preset.mode.value}'"
            )

        self.config.validate()

    def to_config(self, *, copy: bool = True) -> StrategyConfig:
        return deepcopy(self.config) if copy else self.config

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mode": self.mode.value,
            "description": self.description,
            "strategies": list(self.enabled_strategy_names),
            "total_strategies": len(self.enabled_strategy_names),
            "metadata": dict(self.metadata),
        }


class StrategyPresetBuilder:
    """
    Builder for StrategyConfig presets.

    This builder does not instantiate strategies, does not subscribe to EventBus,
    and does not emit strategy/risk/execution events.
    """

    def __init__(
        self,
        *,
        name: str,
        mode: PresetMode,
        description: str,
        symbols: Sequence[str] | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        allowed_regimes: Sequence[MarketRegime] | None = None,
        strategy_names: Sequence[str] | None = None,
        use_required_features: bool = False,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not name.strip():
            raise StrategyConfigError("preset name cannot be empty")

        if not description.strip():
            raise StrategyConfigError("preset description cannot be empty")

        self.name = name.strip()
        self.mode = mode
        self.description = description.strip()
        self.symbols = _clean_symbols(symbols)
        self.timeframes = _normalize_timeframes(timeframes, fallback=()) if timeframes else []
        self.allowed_regimes = _normalize_regimes(allowed_regimes)
        self.strategy_names = _normalize_names(strategy_names or [])
        self.use_required_features = bool(use_required_features)
        self.metadata: dict[str, object] = dict(metadata or {})

        self._runtime_defaults: dict[str, int | float] = {
            "cooldown_seconds": 30,
            "max_signal_age_seconds": 45,
            "min_confidence": 0.5,
            "min_score": 0.20,
        }

        self._strategy_overrides: dict[str, StrategyOverride] = {}
        self._category_weight_overrides: dict[StrategyCategory, float] = {}
        self._regime_adjustment_overrides: dict[MarketRegime, float] = {}

        self._routing: RoutingConfig | None = None
        self._confluence: ConfluenceConfig | None = None
        self._confidence: ConfidenceConfig | None = None
        self._voting: VotingConfig | None = None
        self._conflict: ConflictConfig | None = None
        self._filters: FilterConfig | None = None
        self._builders: BuilderConfig | None = None
        self._freshness: FeatureFreshnessConfig | None = None
        self._portfolio: PortfolioCoordinatorConfig | None = None

    def with_runtime_defaults(
        self,
        *,
        cooldown_seconds: int | None = None,
        max_signal_age_seconds: int | None = None,
        min_confidence: float | None = None,
        min_score: float | None = None,
    ) -> StrategyPresetBuilder:
        if cooldown_seconds is not None:
            self._runtime_defaults["cooldown_seconds"] = cooldown_seconds

        if max_signal_age_seconds is not None:
            self._runtime_defaults["max_signal_age_seconds"] = max_signal_age_seconds

        if min_confidence is not None:
            self._runtime_defaults["min_confidence"] = min_confidence

        if min_score is not None:
            self._runtime_defaults["min_score"] = min_score

        return self

    def with_strategy_override(
        self,
        strategy_name: str,
        *,
        enabled: bool | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        allowed_regimes: Sequence[MarketRegime] | None = None,
        cooldown_seconds: int | None = None,
        max_signal_age_seconds: int | None = None,
        min_confidence: float | None = None,
        min_score: float | None = None,
        weight_multiplier: float | None = None,
        priority_offset: int | None = None,
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> StrategyPresetBuilder:
        name = strategy_name.strip()
        if not name:
            raise StrategyConfigError("strategy_name cannot be empty")

        if name not in STRATEGY_CATALOG:
            raise StrategyConfigError(f"Unknown strategy '{name}'")

        override = self._strategy_overrides.setdefault(name, {})

        if enabled is not None:
            override["enabled"] = enabled

        if timeframes is not None:
            override["timeframes"] = list(timeframes)

        if allowed_regimes is not None:
            override["allowed_regimes"] = list(allowed_regimes)

        if cooldown_seconds is not None:
            override["cooldown_seconds"] = cooldown_seconds

        if max_signal_age_seconds is not None:
            override["max_signal_age_seconds"] = max_signal_age_seconds

        if min_confidence is not None:
            override["min_confidence"] = min_confidence

        if min_score is not None:
            override["min_score"] = min_score

        if weight_multiplier is not None:
            override["weight_multiplier"] = weight_multiplier

        if priority_offset is not None:
            override["priority_offset"] = priority_offset

        if tags is not None:
            override["tags"] = list(tags)

        if metadata is not None:
            override["metadata"] = dict(metadata)

        return self

    def with_routing(self, routing: RoutingConfig) -> StrategyPresetBuilder:
        routing.validate()
        self._routing = routing
        return self

    def with_confluence(self, confluence: ConfluenceConfig) -> StrategyPresetBuilder:
        confluence.validate()
        self._confluence = confluence
        return self

    def with_confidence(self, confidence: ConfidenceConfig) -> StrategyPresetBuilder:
        confidence.validate()
        self._confidence = confidence
        return self

    def with_voting(self, voting: VotingConfig) -> StrategyPresetBuilder:
        voting.validate()
        self._voting = voting
        return self

    def with_conflict(self, conflict: ConflictConfig) -> StrategyPresetBuilder:
        conflict.validate()
        self._conflict = conflict
        return self

    def with_filters(self, filters: FilterConfig) -> StrategyPresetBuilder:
        filters.validate()
        self._filters = filters
        return self

    def with_builders(self, builders: BuilderConfig) -> StrategyPresetBuilder:
        builders.validate()
        self._builders = builders
        return self

    def with_portfolio(
        self,
        portfolio: PortfolioCoordinatorConfig,
    ) -> StrategyPresetBuilder:
        portfolio.validate()
        self._portfolio = portfolio
        return self

    def with_freshness(
        self,
        freshness: FeatureFreshnessConfig,
    ) -> StrategyPresetBuilder:
        freshness.validate()
        self._freshness = freshness
        return self

    def with_category_weights(
        self,
        values: Mapping[StrategyCategory, float],
    ) -> StrategyPresetBuilder:
        self._category_weight_overrides.update(dict(values))
        return self

    def with_regime_adjustments(
        self,
        values: Mapping[MarketRegime, float],
    ) -> StrategyPresetBuilder:
        self._regime_adjustment_overrides.update(dict(values))
        return self

    def build(self) -> StrategyPreset:
        strategy_names = self.strategy_names or list(ALL_STRATEGY_NAMES)

        default_cooldown = int(self._runtime_defaults["cooldown_seconds"])
        default_max_age = int(self._runtime_defaults["max_signal_age_seconds"])
        default_min_confidence = float(self._runtime_defaults["min_confidence"])
        default_min_score = float(self._runtime_defaults["min_score"])

        strategies: dict[str, StrategyDefinitionConfig] = {}

        for name in strategy_names:
            entry = STRATEGY_CATALOG[name]
            override = self._strategy_overrides.get(name, {})

            definition = _definition_from_catalog(
                entry,
                symbols=self.symbols,
                timeframes=_override_timeframes(
                    override,
                    "timeframes",
                    self.timeframes or None,
                ),
                allowed_regimes=_override_regimes(
                    override,
                    "allowed_regimes",
                    self.allowed_regimes,
                ),
                enabled=_override_bool(override, "enabled", True),
                cooldown_seconds=_override_int(
                    override,
                    "cooldown_seconds",
                    default_cooldown,
                ),
                max_signal_age_seconds=_override_int(
                    override,
                    "max_signal_age_seconds",
                    default_max_age,
                ),
                min_confidence=_override_float(
                    override,
                    "min_confidence",
                    default_min_confidence,
                ),
                min_score=_override_float(
                    override,
                    "min_score",
                    default_min_score,
                ),
                weight_multiplier=_override_float(
                    override,
                    "weight_multiplier",
                    1.0,
                ),
                priority_offset=_override_int(
                    override,
                    "priority_offset",
                    0,
                ),
                use_required_features=self.use_required_features,
                extra_tags=_override_tags(override, "tags"),
                metadata=_override_metadata(override, "metadata"),
            )
            strategies[definition.name] = definition

        preset = PresetConfig(
            mode=self.mode,
            enabled_strategy_names=list(strategy_names),
            metadata={
                "name": self.name,
                "description": self.description,
                "use_required_features": self.use_required_features,
                **self.metadata,
            },
        )
        preset.validate()

        runtime_timeframes = _normalize_timeframes(
            self.timeframes or None,
            fallback=self._default_runtime_timeframes(),
        )

        runtime = _runtime(
            symbols=self.symbols,
            timeframes=runtime_timeframes,
            allowed_regimes=self.allowed_regimes,
            cooldown_seconds=default_cooldown,
            max_signal_age_seconds=default_max_age,
            min_confidence=default_min_confidence,
            min_score=default_min_score,
        )

        config = StrategyConfig(
            runtime=runtime,
            routing=self._routing or _routing_config(
                stale_feature_threshold_seconds=max(5, default_max_age),
                reevaluate_on_any_update=False,
                route_hybrid_on_domain_signal=True,
                allow_partial_context=True,
            ),
            confluence=self._confluence or ConfluenceConfig(),
            confidence=self._confidence or ConfidenceConfig(),
            weighting=WeightingConfig(
                category_weights=_category_weights(self._category_weight_overrides),
                regime_adjustments=_regime_adjustments(self._regime_adjustment_overrides),
            ),
            voting=self._voting or VotingConfig(),
            conflict=self._conflict or ConflictConfig(),
            filters=self._filters or FilterConfig(),
            builders=self._builders or BuilderConfig(),
            freshness=self._freshness or FeatureFreshnessConfig(
                default_ttl_seconds=max(5, default_max_age)
            ),
            portfolio=self._portfolio or PortfolioCoordinatorConfig(),
            preset=preset,
            strategies=strategies,
        )
        config.validate()

        return StrategyPreset(
            name=self.name,
            mode=self.mode,
            description=self.description,
            config=config,
            metadata=dict(self.metadata),
        )

    def _default_runtime_timeframes(self) -> Timeframes:
        if self.mode is PresetMode.SCALPING:
            return TF_SCALP

        if self.mode is PresetMode.SWING:
            return TF_SWING

        return TF_INTRADAY


# =============================================================================
# Preset factories
# =============================================================================


def build_scalping_preset(
    *,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[Timeframe] | None = None,
    strategy_names: Sequence[str] | None = None,
    use_required_features: bool = False,
) -> StrategyPreset:
    builder = StrategyPresetBuilder(
        name="scalping",
        mode=PresetMode.SCALPING,
        description=(
            "Fast futures scalping preset focused on orderflow, spoofing, "
            "whale flow and hybrid reversals."
        ),
        symbols=symbols,
        timeframes=timeframes,
        strategy_names=strategy_names or SCALPING_STRATEGIES,
        use_required_features=use_required_features,
        metadata={"profile": "scalping", "market_scope": "futures"},
    )

    return (
        builder
        .with_runtime_defaults(
            cooldown_seconds=10,
            max_signal_age_seconds=20,
            min_confidence=0.5,
            min_score=0.20,
        )
        .with_confluence(
            ConfluenceConfig(
                enabled=True,
                min_agreement_count=1,
                min_confidence=0.5,
                min_score=0.20,
                conflict_penalty=0.12,
                confirmation_bonus=0.08,
                max_strategies_per_side=8,
            )
        )
        .with_filters(
            FilterConfig(
                enable_regime_filter=True,
                enable_volatility_filter=True,
                enable_liquidity_filter=True,
                enable_spread_filter=True,
                enable_funding_filter=False,
                enable_session_filter=False,
                enable_news_filter=False,
                max_spread_bps=12.0,
                min_liquidity_score=0.35,
                max_volatility_zscore=4.0,
                min_funding_alignment=-1.0,
            )
        )
        .with_builders(
            BuilderConfig(
                default_entry_type=EntryType.LIMIT,
                default_rr_ratio=1.6,
                enable_partial_take_profit=True,
                default_partial_tp_levels=[0.6, 0.4],
                require_invalidation=True,
            )
        )
        .with_portfolio(
            PortfolioCoordinatorConfig(
                enabled=True,
                max_signals_per_symbol=2,
                deduplicate_by_side=True,
                merge_similar_signals=True,
                correlation_guard_enabled=True,
                symbol_cooldown_seconds=10,
                side_cooldown_seconds=8,
                repeated_signal_suppression_seconds=18,
                volatility_throttle_enabled=True,
                volatility_throttle_threshold=4.0,
                high_volatility_max_signals_per_symbol=1,
                max_signals_per_category={
                    StrategyCategory.ORDERFLOW: 2,
                    StrategyCategory.SPOOFING: 2,
                    StrategyCategory.WHALES: 2,
                    StrategyCategory.HYBRID: 2,
                },
                priority_overrides={
                    "liquidity_orderflow_reversal": 10,
                    "composite_spoofing": 15,
                    "cvd_divergence": 18,
                },
                exposure_bucket_limits={"fast_reversal": 2, "orderbook": 2},
                enable_correlation_direction_conflict=True,
            )
        )
        .with_freshness(
            FeatureFreshnessConfig(
                default_ttl_seconds=20,
                per_feature_ttl_seconds={
                    "orderbook_imbalance": 8,
                    "spoofing_score": 8,
                    "fake_liquidity": 8,
                    "volume_delta": 15,
                    "cvd": 20,
                    "large_trade": 20,
                },
            )
        )
        .with_category_weights(
            {
                StrategyCategory.ORDERFLOW: 1.12,
                StrategyCategory.SPOOFING: 1.08,
                StrategyCategory.WHALES: 1.02,
                StrategyCategory.HYBRID: 1.20,
                StrategyCategory.PRICE_ACTION: 0.80,
                StrategyCategory.SPREADS: 0.55,
                StrategyCategory.OPEN_INTEREST: 0.75,
            }
        )
        .build()
    )


def build_intraday_preset(
    *,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[Timeframe] | None = None,
    strategy_names: Sequence[str] | None = None,
    use_required_features: bool = False,
) -> StrategyPreset:
    builder = StrategyPresetBuilder(
        name="intraday",
        mode=PresetMode.INTRADAY,
        description=(
            "Balanced futures intraday preset using price action, OI, "
            "orderflow, whales, spreads and hybrid stacks."
        ),
        symbols=symbols,
        timeframes=timeframes,
        strategy_names=strategy_names or INTRADAY_STRATEGIES,
        use_required_features=use_required_features,
        metadata={"profile": "intraday", "market_scope": "futures"},
    )

    return (
        builder
        .with_runtime_defaults(
            cooldown_seconds=35,
            max_signal_age_seconds=60,
            min_confidence=0.50,
            min_score=0.30,
        )
        .with_confluence(
            ConfluenceConfig(
                enabled=True,
                min_agreement_count=2,
                min_confidence=0.50,
                min_score=0.35,
                conflict_penalty=0.15,
                confirmation_bonus=0.10,
                max_strategies_per_side=10,
            )
        )
        .with_filters(
            FilterConfig(
                enable_regime_filter=True,
                enable_volatility_filter=True,
                enable_liquidity_filter=True,
                enable_spread_filter=True,
                enable_funding_filter=True,
                enable_session_filter=False,
                enable_news_filter=False,
                max_spread_bps=18.0,
                min_liquidity_score=0.32,
                max_volatility_zscore=3.2,
                min_funding_alignment=-0.75,
            )
        )
        .with_builders(
            BuilderConfig(
                default_entry_type=EntryType.LIMIT,
                default_rr_ratio=2.0,
                enable_partial_take_profit=True,
                default_partial_tp_levels=[0.5, 0.5],
                require_invalidation=True,
            )
        )
        .with_portfolio(
            PortfolioCoordinatorConfig(
                enabled=True,
                max_signals_per_symbol=3,
                deduplicate_by_side=True,
                merge_similar_signals=True,
                correlation_guard_enabled=True,
                symbol_cooldown_seconds=25,
                side_cooldown_seconds=20,
                repeated_signal_suppression_seconds=45,
                volatility_throttle_enabled=True,
                volatility_throttle_threshold=3.2,
                high_volatility_max_signals_per_symbol=1,
                max_signals_per_category={
                    StrategyCategory.PRICE_ACTION: 2,
                    StrategyCategory.ORDERFLOW: 2,
                    StrategyCategory.OPEN_INTEREST: 2,
                    StrategyCategory.WHALES: 2,
                    StrategyCategory.SPREADS: 1,
                    StrategyCategory.HYBRID: 2,
                },
                priority_overrides={
                    "confluence": 8,
                    "trend_stack": 12,
                    "mean_reversion_stack": 13,
                    "whale_orderflow_breakout": 14,
                },
                exposure_bucket_limits={"directional": 3, "hybrid": 2},
                enable_correlation_direction_conflict=True,
            )
        )
        .with_freshness(
            FeatureFreshnessConfig(
                default_ttl_seconds=60,
                per_feature_ttl_seconds={
                    "market_structure": 120,
                    "trend_strength": 90,
                    "open_interest": 90,
                    "oi_delta": 90,
                    "large_trade": 60,
                    "spread_bps": 45,
                    "funding_rate": 300,
                },
            )
        )
        .with_category_weights(
            {
                StrategyCategory.PRICE_ACTION: 1.00,
                StrategyCategory.ORDERFLOW: 1.02,
                StrategyCategory.OPEN_INTEREST: 1.00,
                StrategyCategory.WHALES: 0.98,
                StrategyCategory.SPREADS: 0.82,
                StrategyCategory.HYBRID: 1.20,
                StrategyCategory.SPOOFING: 0.82,
            }
        )
        .build()
    )


def build_swing_preset(
    *,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[Timeframe] | None = None,
    strategy_names: Sequence[str] | None = None,
    use_required_features: bool = False,
) -> StrategyPreset:
    builder = StrategyPresetBuilder(
        name="swing",
        mode=PresetMode.SWING,
        description=(
            "Higher-timeframe futures preset focused on structure, OI, "
            "whale accumulation/distribution and spread regime."
        ),
        symbols=symbols,
        timeframes=timeframes,
        strategy_names=strategy_names or SWING_STRATEGIES,
        use_required_features=use_required_features,
        metadata={"profile": "swing", "market_scope": "futures"},
    )

    return (
        builder
        .with_runtime_defaults(
            cooldown_seconds=300,
            max_signal_age_seconds=600,
            min_confidence=0.50,
            min_score=0.40,
        )
        .with_confluence(
            ConfluenceConfig(
                enabled=True,
                min_agreement_count=2,
                min_confidence=0.50,
                min_score=0.45,
                conflict_penalty=0.18,
                confirmation_bonus=0.12,
                max_strategies_per_side=8,
            )
        )
        .with_filters(
            FilterConfig(
                enable_regime_filter=True,
                enable_volatility_filter=True,
                enable_liquidity_filter=True,
                enable_spread_filter=True,
                enable_funding_filter=True,
                enable_session_filter=False,
                enable_news_filter=True,
                max_spread_bps=25.0,
                min_liquidity_score=0.35,
                max_volatility_zscore=2.8,
                min_funding_alignment=-0.50,
            )
        )
        .with_builders(
            BuilderConfig(
                default_entry_type=EntryType.LIMIT,
                default_rr_ratio=2.5,
                enable_partial_take_profit=True,
                default_partial_tp_levels=[0.4, 0.3, 0.3],
                require_invalidation=True,
            )
        )
        .with_portfolio(
            PortfolioCoordinatorConfig(
                enabled=True,
                max_signals_per_symbol=2,
                deduplicate_by_side=True,
                merge_similar_signals=True,
                correlation_guard_enabled=True,
                symbol_cooldown_seconds=180,
                side_cooldown_seconds=180,
                repeated_signal_suppression_seconds=600,
                volatility_throttle_enabled=True,
                volatility_throttle_threshold=2.8,
                high_volatility_max_signals_per_symbol=1,
                max_signals_per_category={
                    StrategyCategory.PRICE_ACTION: 2,
                    StrategyCategory.OPEN_INTEREST: 2,
                    StrategyCategory.WHALES: 2,
                    StrategyCategory.SPREADS: 1,
                    StrategyCategory.HYBRID: 2,
                },
                priority_overrides={
                    "trend_stack": 10,
                    "confluence": 12,
                    "mean_reversion_stack": 16,
                },
                exposure_bucket_limits={"swing_directional": 2, "macro_basis": 1},
                enable_correlation_direction_conflict=True,
            )
        )
        .with_freshness(
            FeatureFreshnessConfig(
                default_ttl_seconds=600,
                per_feature_ttl_seconds={
                    "market_structure": 900,
                    "trend_strength": 900,
                    "open_interest": 900,
                    "funding_rate": 1800,
                    "basis": 900,
                    "whale_accumulation": 900,
                    "whale_distribution": 900,
                },
            )
        )
        .with_category_weights(
            {
                StrategyCategory.PRICE_ACTION: 1.05,
                StrategyCategory.OPEN_INTEREST: 1.05,
                StrategyCategory.WHALES: 1.00,
                StrategyCategory.SPREADS: 0.90,
                StrategyCategory.HYBRID: 1.18,
                StrategyCategory.ORDERFLOW: 0.70,
                StrategyCategory.SPOOFING: 0.55,
            }
        )
        .build()
    )


def build_liquidity_reversal_preset(
    *,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[Timeframe] | None = None,
    strategy_names: Sequence[str] | None = None,
    use_required_features: bool = False,
) -> StrategyPreset:
    builder = StrategyPresetBuilder(
        name="liquidity_reversal",
        mode=PresetMode.INTRADAY,
        description=(
            "Specialized futures reversal preset for traps, spoofing, absorption, "
            "OI capitulation and whale/liquidation reversals."
        ),
        symbols=symbols,
        timeframes=timeframes,
        strategy_names=strategy_names or LIQUIDITY_REVERSAL_STRATEGIES,
        use_required_features=use_required_features,
        metadata={"profile": "liquidity_reversal", "market_scope": "futures"},
    )

    return (
        builder
        .with_runtime_defaults(
            cooldown_seconds=20,
            max_signal_age_seconds=35,
            min_confidence=0.50,
            min_score=0.25,
        )
        .with_confluence(
            ConfluenceConfig(
                enabled=True,
                min_agreement_count=1,
                min_confidence=0.50,
                min_score=0.25,
                conflict_penalty=0.12,
                confirmation_bonus=0.12,
                max_strategies_per_side=9,
            )
        )
        .with_filters(
            FilterConfig(
                enable_regime_filter=True,
                enable_volatility_filter=True,
                enable_liquidity_filter=True,
                enable_spread_filter=True,
                enable_funding_filter=False,
                enable_session_filter=False,
                enable_news_filter=False,
                max_spread_bps=16.0,
                min_liquidity_score=0.30,
                max_volatility_zscore=4.2,
                min_funding_alignment=-1.0,
            )
        )
        .with_builders(
            BuilderConfig(
                default_entry_type=EntryType.LIMIT,
                default_rr_ratio=1.9,
                enable_partial_take_profit=True,
                default_partial_tp_levels=[0.55, 0.45],
                require_invalidation=True,
            )
        )
        .with_portfolio(
            PortfolioCoordinatorConfig(
                enabled=True,
                max_signals_per_symbol=2,
                deduplicate_by_side=True,
                merge_similar_signals=True,
                correlation_guard_enabled=True,
                symbol_cooldown_seconds=18,
                side_cooldown_seconds=15,
                repeated_signal_suppression_seconds=30,
                volatility_throttle_enabled=True,
                volatility_throttle_threshold=4.2,
                high_volatility_max_signals_per_symbol=1,
                max_signals_per_category={
                    StrategyCategory.ORDERFLOW: 2,
                    StrategyCategory.SPOOFING: 2,
                    StrategyCategory.WHALES: 2,
                    StrategyCategory.OPEN_INTEREST: 2,
                    StrategyCategory.HYBRID: 2,
                },
                priority_overrides={
                    "liquidity_orderflow_reversal": 8,
                    "whale_liquidation_reversal": 12,
                    "spoofing_absorption_reversal": 13,
                    "oi_capitulation": 14,
                    "cvd_divergence": 15,
                },
                exposure_bucket_limits={"liquidity_reversal": 2, "trap": 2},
                enable_correlation_direction_conflict=True,
            )
        )
        .with_freshness(
            FeatureFreshnessConfig(
                default_ttl_seconds=35,
                per_feature_ttl_seconds={
                    "fake_liquidity": 10,
                    "spoofing_score": 10,
                    "order_pull": 10,
                    "absorption": 20,
                    "cvd_divergence": 35,
                    "oi_capitulation": 60,
                    "whale_liquidation": 45,
                },
            )
        )
        .with_category_weights(
            {
                StrategyCategory.ORDERFLOW: 1.10,
                StrategyCategory.SPOOFING: 1.12,
                StrategyCategory.WHALES: 1.05,
                StrategyCategory.OPEN_INTEREST: 1.00,
                StrategyCategory.HYBRID: 1.25,
                StrategyCategory.PRICE_ACTION: 0.75,
                StrategyCategory.SPREADS: 0.55,
            }
        )
        .build()
    )


def build_mean_reversion_preset(
    *,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[Timeframe] | None = None,
    strategy_names: Sequence[str] | None = None,
    use_required_features: bool = False,
) -> StrategyPreset:
    builder = StrategyPresetBuilder(
        name="mean_reversion",
        mode=PresetMode.INTRADAY,
        description=(
            "Futures mean-reversion preset using OI divergence/capitulation, "
            "spread reversion, absorption and hybrid stacks."
        ),
        symbols=symbols,
        timeframes=timeframes,
        strategy_names=strategy_names or MEAN_REVERSION_STRATEGIES,
        use_required_features=use_required_features,
        metadata={"profile": "mean_reversion", "market_scope": "futures"},
    )

    return (
        builder
        .with_runtime_defaults(
            cooldown_seconds=45,
            max_signal_age_seconds=90,
            min_confidence=0.50,
            min_score=0.32,
        )
        .with_confluence(
            ConfluenceConfig(
                enabled=True,
                min_agreement_count=2,
                min_confidence=0.50,
                min_score=0.35,
                conflict_penalty=0.14,
                confirmation_bonus=0.12,
                max_strategies_per_side=8,
            )
        )
        .with_filters(
            FilterConfig(
                enable_regime_filter=True,
                enable_volatility_filter=True,
                enable_liquidity_filter=True,
                enable_spread_filter=True,
                enable_funding_filter=True,
                enable_session_filter=False,
                enable_news_filter=False,
                max_spread_bps=20.0,
                min_liquidity_score=0.32,
                max_volatility_zscore=3.5,
                min_funding_alignment=-0.85,
            )
        )
        .with_builders(
            BuilderConfig(
                default_entry_type=EntryType.LIMIT,
                default_rr_ratio=2.1,
                enable_partial_take_profit=True,
                default_partial_tp_levels=[0.5, 0.5],
                require_invalidation=True,
            )
        )
        .with_category_weights(
            {
                StrategyCategory.OPEN_INTEREST: 1.08,
                StrategyCategory.SPREADS: 1.02,
                StrategyCategory.WHALES: 1.00,
                StrategyCategory.SPOOFING: 0.95,
                StrategyCategory.PRICE_ACTION: 0.95,
                StrategyCategory.HYBRID: 1.22,
            }
        )
        .build()
    )


def build_trend_stack_preset(
    *,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[Timeframe] | None = None,
    strategy_names: Sequence[str] | None = None,
    use_required_features: bool = False,
) -> StrategyPreset:
    builder = StrategyPresetBuilder(
        name="trend_stack",
        mode=PresetMode.INTRADAY,
        description=(
            "Futures trend-following preset using market structure, orderflow "
            "continuation, OI confirmation, whales and hybrid trend stacks."
        ),
        symbols=symbols,
        timeframes=timeframes,
        strategy_names=strategy_names or TREND_STACK_STRATEGIES,
        use_required_features=use_required_features,
        metadata={"profile": "trend_stack", "market_scope": "futures"},
    )

    return (
        builder
        .with_runtime_defaults(
            cooldown_seconds=45,
            max_signal_age_seconds=90,
            min_confidence=0.50,
            min_score=0.35,
        )
        .with_confluence(
            ConfluenceConfig(
                enabled=True,
                min_agreement_count=2,
                min_confidence=0.50,
                min_score=0.40,
                conflict_penalty=0.16,
                confirmation_bonus=0.12,
                max_strategies_per_side=8,
            )
        )
        .with_filters(
            FilterConfig(
                enable_regime_filter=True,
                enable_volatility_filter=True,
                enable_liquidity_filter=True,
                enable_spread_filter=True,
                enable_funding_filter=True,
                enable_session_filter=False,
                enable_news_filter=False,
                max_spread_bps=18.0,
                min_liquidity_score=0.34,
                max_volatility_zscore=3.3,
                min_funding_alignment=-0.65,
            )
        )
        .with_builders(
            BuilderConfig(
                default_entry_type=EntryType.LIMIT,
                default_rr_ratio=2.2,
                enable_partial_take_profit=True,
                default_partial_tp_levels=[0.45, 0.35, 0.20],
                require_invalidation=True,
            )
        )
        .with_category_weights(
            {
                StrategyCategory.PRICE_ACTION: 1.10,
                StrategyCategory.ORDERFLOW: 1.05,
                StrategyCategory.OPEN_INTEREST: 1.05,
                StrategyCategory.WHALES: 1.02,
                StrategyCategory.SPREADS: 0.82,
                StrategyCategory.HYBRID: 1.22,
                StrategyCategory.SPOOFING: 0.55,
            }
        )
        .build()
    )


def build_spreads_preset(
    *,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[Timeframe] | None = None,
    strategy_names: Sequence[str] | None = None,
    use_required_features: bool = False,
) -> StrategyPreset:
    builder = StrategyPresetBuilder(
        name="spreads",
        mode=PresetMode.INTRADAY,
        description=(
            "Futures spreads/basis preset for cross-exchange, basis, "
            "funding-adjusted basis, spread reversion and spread momentum."
        ),
        symbols=symbols,
        timeframes=timeframes,
        strategy_names=strategy_names or SPREADS_PRESET_STRATEGIES,
        use_required_features=use_required_features,
        metadata={"profile": "spreads", "market_scope": "futures"},
    )

    return (
        builder
        .with_runtime_defaults(
            cooldown_seconds=60,
            max_signal_age_seconds=90,
            min_confidence=0.50,
            min_score=0.30,
        )
        .with_confluence(
            ConfluenceConfig(
                enabled=False,
                min_agreement_count=1,
                min_confidence=0.50,
                min_score=0.30,
                conflict_penalty=0.12,
                confirmation_bonus=0.08,
                max_strategies_per_side=5,
            )
        )
        .with_filters(
            FilterConfig(
                enable_regime_filter=False,
                enable_volatility_filter=True,
                enable_liquidity_filter=True,
                enable_spread_filter=False,
                enable_funding_filter=True,
                enable_session_filter=False,
                enable_news_filter=False,
                max_spread_bps=50.0,
                min_liquidity_score=0.40,
                max_volatility_zscore=3.0,
                min_funding_alignment=-1.0,
            )
        )
        .with_builders(
            BuilderConfig(
                default_entry_type=EntryType.LIMIT,
                default_rr_ratio=1.5,
                enable_partial_take_profit=True,
                default_partial_tp_levels=[0.5, 0.5],
                require_invalidation=True,
            )
        )
        .with_portfolio(
            PortfolioCoordinatorConfig(
                enabled=True,
                max_signals_per_symbol=2,
                deduplicate_by_side=False,
                merge_similar_signals=True,
                correlation_guard_enabled=True,
                symbol_cooldown_seconds=60,
                side_cooldown_seconds=45,
                repeated_signal_suppression_seconds=120,
                volatility_throttle_enabled=True,
                volatility_throttle_threshold=3.0,
                high_volatility_max_signals_per_symbol=1,
                max_signals_per_category={StrategyCategory.SPREADS: 3},
                priority_overrides={
                    "funding_adjusted_basis": 20,
                    "spread_mean_reversion": 25,
                    "cross_exchange_arb": 30,
                },
                exposure_bucket_limits={"spreads": 3, "basis": 2},
                enable_correlation_direction_conflict=False,
            )
        )
        .with_category_weights(
            {
                StrategyCategory.SPREADS: 1.20,
                StrategyCategory.FUNDING: 0.90,
                StrategyCategory.OPEN_INTEREST: 0.75,
                StrategyCategory.HYBRID: 0.75,
                StrategyCategory.ORDERFLOW: 0.45,
                StrategyCategory.SPOOFING: 0.35,
            }
        )
        .build()
    )


def build_default_preset(
    *,
    symbols: Sequence[str] | None = None,
    use_required_features: bool = False,
) -> StrategyPreset:
    """
    Build the production default preset.

    Unlike profile-specific presets such as scalping/intraday/swing, the default
    preset intentionally configures every strategy present in STRATEGY_CATALOG.
    This guarantees that build_default_strategy_config() does not silently skip
    a strategy just because its factory was added before a profile list was
    updated.
    """
    builder = StrategyPresetBuilder(
        name="default",
        mode=PresetMode.INTRADAY,
        description=(
            "Full futures strategy preset enabling every catalog strategy. "
            "Use profile-specific presets only when an explicit subset is wanted."
        ),
        symbols=symbols,
        timeframes=None,
        strategy_names=DEFAULT_STRATEGIES,
        use_required_features=use_required_features,
        metadata={
            "profile": "default",
            "market_scope": "futures",
            "enabled_scope": "all_catalog_strategies",
        },
    )

    return (
        builder
        .with_runtime_defaults(
            cooldown_seconds=35,
            max_signal_age_seconds=90,
            min_confidence=0.50,
            min_score=0.18,
        )
        .with_routing(
            _routing_config(
                stale_feature_threshold_seconds=90,
                reevaluate_on_any_update=False,
                route_hybrid_on_domain_signal=True,
                allow_partial_context=True,
            )
        )
        .with_confluence(
            ConfluenceConfig(
                enabled=True,
                min_agreement_count=1,
                min_confidence=0.50,
                min_score=0.24,
                conflict_penalty=0.15,
                confirmation_bonus=0.10,
                max_strategies_per_side=10,
            )
        )
        .with_confidence(
            ConfidenceConfig(
                very_low_threshold=0.35,
                low_threshold=0.40,
                medium_threshold=0.56,
                high_threshold=0.72,
            )
        )
        .with_voting(
            VotingConfig(
                min_confirmations=1,
                min_total_votes=1,
                require_primary_trigger=True,
                allow_single_strategy_confirmation=True,
            )
        )
        .with_conflict(
            ConflictConfig(
                reject_on_side_conflict=False,
                reject_on_regime_conflict=False,
                max_total_penalty=0.45,
            )
        )
        .with_filters(
            FilterConfig(
                enabled=True,
                min_signal_confidence=0.56,
                min_signal_score=0.18,
                min_risk_reward=1.20,
                enable_cooldown_filter=True,
                enable_regime_filter=True,
                enable_freshness_filter=True,
                enable_portfolio_filter=True,
                enable_funding_filter=True,
                min_funding_alignment=-0.85,
            )
        )
        .with_builders(
            BuilderConfig(
                default_entry_type=EntryType.MARKET,
                default_rr_ratio=1.20,
                enable_partial_take_profit=True,
                default_partial_tp_levels=[0.5, 0.3, 0.2],
                require_invalidation=True,
            )
        )
        .with_portfolio(
            PortfolioCoordinatorConfig(
                enabled=True,
                max_signals_per_symbol=4,
                deduplicate_by_side=True,
                merge_similar_signals=True,
                correlation_guard_enabled=True,
                symbol_cooldown_seconds=15,
                side_cooldown_seconds=10,
                repeated_signal_suppression_seconds=30,
                volatility_throttle_enabled=True,
                volatility_throttle_threshold=3.0,
                high_volatility_max_signals_per_symbol=2,
            )
        )
        .with_freshness(
            FeatureFreshnessConfig(
                default_ttl_seconds=90,
                per_feature_ttl_seconds={
                    "orderflow": 30,
                    "liquidity": 45,
                    "liquidations": 45,
                    "spoofing": 30,
                    "whales": 60,
                    "open_interest": 180,
                    "funding": 300,
                    "price_action": 120,
                    "spreads": 120,
                    "hybrid": 90,
                },
            )
        )
        .with_category_weights(
            {
                StrategyCategory.ORDERFLOW: 1.05,
                StrategyCategory.LIQUIDITY: 1.05,
                StrategyCategory.LIQUIDATIONS: 1.02,
                StrategyCategory.OPEN_INTEREST: 1.00,
                StrategyCategory.FUNDING: 0.92,
                StrategyCategory.WHALES: 1.00,
                StrategyCategory.SPOOFING: 0.92,
                StrategyCategory.SPREADS: 0.88,
                StrategyCategory.PRICE_ACTION: 0.96,
                StrategyCategory.HYBRID: 1.20,
            }
        )
        .build()
    )

def build_preset_by_name(
    name: str,
    *,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[Timeframe] | None = None,
    strategy_names: Sequence[str] | None = None,
    use_required_features: bool = False,
) -> StrategyPreset:
    normalized = name.strip().lower()

    if normalized == "default":
        return build_default_preset(
            symbols=symbols,
            use_required_features=use_required_features,
        )

    if normalized == "scalping":
        return build_scalping_preset(
            symbols=symbols,
            timeframes=timeframes,
            strategy_names=strategy_names,
            use_required_features=use_required_features,
        )

    if normalized == "intraday":
        return build_intraday_preset(
            symbols=symbols,
            timeframes=timeframes,
            strategy_names=strategy_names,
            use_required_features=use_required_features,
        )

    if normalized == "swing":
        return build_swing_preset(
            symbols=symbols,
            timeframes=timeframes,
            strategy_names=strategy_names,
            use_required_features=use_required_features,
        )

    if normalized == "liquidity_reversal":
        return build_liquidity_reversal_preset(
            symbols=symbols,
            timeframes=timeframes,
            strategy_names=strategy_names,
            use_required_features=use_required_features,
        )

    if normalized == "mean_reversion":
        return build_mean_reversion_preset(
            symbols=symbols,
            timeframes=timeframes,
            strategy_names=strategy_names,
            use_required_features=use_required_features,
        )

    if normalized == "trend_stack":
        return build_trend_stack_preset(
            symbols=symbols,
            timeframes=timeframes,
            strategy_names=strategy_names,
            use_required_features=use_required_features,
        )

    if normalized == "spreads":
        return build_spreads_preset(
            symbols=symbols,
            timeframes=timeframes,
            strategy_names=strategy_names,
            use_required_features=use_required_features,
        )

    available = (
        "default",
        "scalping",
        "intraday",
        "swing",
        "liquidity_reversal",
        "mean_reversion",
        "trend_stack",
        "spreads",
    )
    raise StrategyConfigError(
        f"Unknown preset '{name}'. Available presets: {', '.join(available)}"
    )


def build_default_strategy_config(
    *,
    symbols: Sequence[str] | None = None,
    preset_name: str = "default",
    use_required_features: bool = False,
) -> StrategyConfig:
    return build_preset_by_name(
        preset_name,
        symbols=symbols,
        use_required_features=use_required_features,
    ).to_config()


# =============================================================================
# Registry bootstrap
# =============================================================================


def build_default_strategy_registry(
    *,
    config: StrategyConfig,
    event_bus: EventBus | None = None,
    scheduler: Scheduler | None = None,
    strategies: Iterable[BaseStrategy] | None = None,
    strategy_factories: Mapping[str, StrategyFactory] | None = None,
    replace: bool = False,
    emit_events: bool = False,
    strict: bool = False,
) -> StrategyRegistry:
    """
    Build and optionally populate StrategyRegistry.

    This function does not evaluate strategies and does not emit signal.generated.
    """

    config.validate()

    registry = StrategyRegistry(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    configured_names = set(config.strategies.keys())
    allowed_names = set(config.preset.enabled_strategy_names or list(configured_names))

    if strategy_factories:
        for name in sorted(allowed_names):
            if not config.is_strategy_configured(name):
                if strict:
                    raise StrategyRegistrationError(
                        f"Strategy '{name}' is allowed by preset but not configured"
                    )
                continue

            factory = strategy_factories.get(name)
            if factory is None:
                if strict:
                    raise StrategyRegistrationError(
                        f"No strategy factory provided for '{name}'"
                    )
                continue

            definition = config.require_strategy(name)
            strategy = _instantiate_strategy(
                factory=factory,
                config=config,
                event_bus=event_bus,
                scheduler=scheduler,
                definition=definition,
            )
            registry.register_strategy(
                strategy,
                replace=replace,
                emit_event=emit_events,
            )

    if strategies:
        for strategy in strategies:
            strategy_name = _safe_strategy_name(strategy)

            if allowed_names and strategy_name not in allowed_names:
                if strict:
                    raise StrategyRegistrationError(
                        f"Strategy instance '{strategy_name}' is not allowed by preset"
                    )
                continue

            registry.register_strategy(
                strategy,
                replace=replace,
                emit_event=emit_events,
            )

    registry.register()
    return registry


def _instantiate_strategy(
    *,
    factory: StrategyFactory,
    config: StrategyConfig,
    event_bus: EventBus | None,
    scheduler: Scheduler | None,
    definition: StrategyDefinitionConfig,
) -> BaseStrategy:
    attempts: tuple[dict[str, object], ...] = (
        {
            "config": config,
            "event_bus": event_bus,
            "scheduler": scheduler,
            "definition": definition,
        },
        {
            "config": config,
            "event_bus": event_bus,
            "scheduler": scheduler,
        },
    )

    last_error: Exception | None = None

    for kwargs in attempts:
        try:
            strategy = factory(**kwargs)
        except TypeError as exc:
            last_error = exc
            continue

        if not isinstance(strategy, BaseStrategy):
            raise StrategyRegistrationError(
                f"Factory for '{definition.name}' returned "
                f"{type(strategy)!r}, expected BaseStrategy"
            )

        return strategy

    try:
        strategy = factory(config, event_bus, scheduler, definition)
    except TypeError as exc:
        last_error = exc
    else:
        if not isinstance(strategy, BaseStrategy):
            raise StrategyRegistrationError(
                f"Factory for '{definition.name}' returned "
                f"{type(strategy)!r}, expected BaseStrategy"
            )
        return strategy

    raise StrategyRegistrationError(
        f"Could not instantiate strategy '{definition.name}' from factory: {last_error}"
    )


def _safe_strategy_name(strategy: BaseStrategy) -> str:
    name = strategy.strategy_name

    if not isinstance(name, str) or not name.strip():
        raise StrategyRegistrationError(
            f"Strategy instance {strategy!r} has empty strategy_name"
        )

    return name.strip()


def configured_strategy_names(config: StrategyConfig) -> list[str]:
    config.validate()
    return sorted(config.strategies.keys())


def enabled_strategy_names(config: StrategyConfig) -> list[str]:
    config.validate()

    names = config.preset.enabled_strategy_names
    if names:
        return list(names)

    return configured_strategy_names(config)


def catalog_by_category(
    category: StrategyCategory,
) -> dict[str, StrategyCatalogEntry]:
    return {
        name: entry
        for name, entry in STRATEGY_CATALOG.items()
        if entry.category is category
    }


def validate_strategy_catalog() -> None:
    for name, entry in STRATEGY_CATALOG.items():
        if name != entry.name:
            raise StrategyConfigError(
                f"Strategy catalog key '{name}' does not match entry.name '{entry.name}'"
            )
        entry.validate()


validate_strategy_catalog()


__all__ = [
    "Timeframes",
    "StrategyNames",
    "StrategyFactory",
    "StrategyCatalogEntry",
    "StrategyPreset",
    "StrategyPresetBuilder",
    "STRATEGY_CATALOG",
    "TF_SCALP",
    "TF_SCALP_INTRADAY",
    "TF_INTRADAY",
    "TF_SWING",
    "TF_FAST",
    "TF_MID",
    "TF_FUNDING",
    "TF_SPREADS",
    "ORDERFLOW_STRATEGIES",
    "PRICE_ACTION_STRATEGIES",
    "OPEN_INTEREST_STRATEGIES",
    "LIQUIDITY_STRATEGIES",
    "LIQUIDATIONS_STRATEGIES",
    "FUNDING_STRATEGIES",
    "SPOOFING_STRATEGIES",
    "SPREADS_STRATEGIES",
    "WHALES_STRATEGIES",
    "HYBRID_STRATEGIES",
    "ALL_STRATEGY_NAMES",
    "DEFAULT_STRATEGIES",
    "SCALPING_STRATEGIES",
    "INTRADAY_STRATEGIES",
    "SWING_STRATEGIES",
    "LIQUIDITY_REVERSAL_STRATEGIES",
    "MEAN_REVERSION_STRATEGIES",
    "TREND_STACK_STRATEGIES",
    "SPREADS_PRESET_STRATEGIES",
    "build_scalping_preset",
    "build_intraday_preset",
    "build_swing_preset",
    "build_liquidity_reversal_preset",
    "build_mean_reversion_preset",
    "build_trend_stack_preset",
    "build_spreads_preset",
    "build_default_preset",
    "build_preset_by_name",
    "build_default_strategy_config",
    "build_default_strategy_registry",
    "configured_strategy_names",
    "enabled_strategy_names",
    "catalog_by_category",
    "validate_strategy_catalog",
]