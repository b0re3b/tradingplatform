# strategy/strategies/__init__.py

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any


_EXPORTS: dict[str, str] = {
    # Base strategy layer
    "TradingStrategy": ".base_strategy",
    "StrategySignalMixin": ".base_strategy",
    "StrategyValidationMixin": ".base_strategy",
    "StrategyRiskRewardMixin": ".base_strategy",

    # Orderflow
    "CvdDivergenceStrategy": ".orderflow",
    "OrderflowContinuationStrategy": ".orderflow",
    "OrderflowReversalStrategy": ".orderflow",

    # Price action
    "MarketStructureStrategy": ".price_action",
    "FVGReactionStrategy": ".price_action",
    "SupportResistanceReactionStrategy": ".price_action",
    "TrendContinuationStrategy": ".price_action",

    # Open interest
    "OIAnomalyStrategy": ".open_interest",
    "OIBreakoutConfirmationStrategy": ".open_interest",
    "OICapitulationStrategy": ".open_interest",
    "OIDivergenceStrategy": ".open_interest",

    # Funding
    "FundingDivergenceStrategy": ".funding",
    "FundingExtremeReversalStrategy": ".funding",

    # Liquidity
    "EqualHighLowStrategy": ".liquidity",
    "LiquidityMapBiasStrategy": ".liquidity",
    "LiquiditySweepStrategy": ".liquidity",
    "StopHuntReversalStrategy": ".liquidity",

    # Liquidations
    "LiquidationCascadeStrategy": ".liquidations",
    "SqueezeReversalStrategy": ".liquidations",

    # Spoofing
    "CompositeSpoofingStrategy": ".spoofing",
    "FakeLiquidityTrapStrategy": ".spoofing",
    "LayeringTrapStrategy": ".spoofing",
    "OrderPullReversalStrategy": ".spoofing",
    "PressureBluffReversalStrategy": ".spoofing",
    "SpoofingAbsorptionReversalStrategy": ".spoofing",
    "SpoofingReversalStrategy": ".spoofing",

    # Spreads
    "CrossExchangeArbStrategy": ".spreads",
    "FundingAdjustedBasisStrategy": ".spreads",
    "SpreadMeanReversionStrategy": ".spreads",
    "SpreadMomentumStrategy": ".spreads",
    "SpotFuturesBasisStrategy": ".spreads",

    # Whales
    "WhaleAbsorptionStrategy": ".whales",
    "WhaleAccumulationStrategy": ".whales",
    "WhaleBreakoutStrategy": ".whales",
    "WhaleDistributionStrategy": ".whales",
    "WhaleLiquidationReversalStrategy": ".whales",

    # Hybrid
    "ConfluenceStrategy": ".hybrid",
    "LiquidationWhaleStrategy": ".hybrid",
    "LiquidityOrderflowReversalStrategy": ".hybrid",
    "MeanReversionStackStrategy": ".hybrid",
    "OIFundingSqueezeStrategy": ".hybrid",
    "TrendStackStrategy": ".hybrid",
    "WhaleOrderflowBreakoutStrategy": ".hybrid",
}


__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """
    Lazily expose strategy classes from domain packages.

    This keeps strategy.strategies import lightweight and avoids circular imports
    during registry/preset bootstrap.
    """
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module = import_module(module_name, package=__name__)
    value = getattr(module, name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .base_strategy import (
        StrategyRiskRewardMixin,
        StrategySignalMixin,
        StrategyValidationMixin,
        TradingStrategy,
    )

    from .funding import (
        FundingDivergenceStrategy,
        FundingExtremeReversalStrategy,
    )

    from .hybrid import (
        ConfluenceStrategy,
        LiquidationWhaleStrategy,
        LiquidityOrderflowReversalStrategy,
        MeanReversionStackStrategy,
        OIFundingSqueezeStrategy,
        TrendStackStrategy,
        WhaleOrderflowBreakoutStrategy,
    )

    from .liquidations import (
        LiquidationCascadeStrategy,
        SqueezeReversalStrategy,
    )

    from .liquidity import (
        EqualHighLowStrategy,
        LiquidityMapBiasStrategy,
        LiquiditySweepStrategy,
        StopHuntReversalStrategy,
    )

    from .open_interest import (
        OIAnomalyStrategy,
        OIBreakoutConfirmationStrategy,
        OICapitulationStrategy,
        OIDivergenceStrategy,
    )

    from .orderflow import (
        CvdDivergenceStrategy,
        OrderflowContinuationStrategy,
        OrderflowReversalStrategy,
    )

    from .price_action import (
        FVGReactionStrategy,
        MarketStructureStrategy,
        SupportResistanceReactionStrategy,
        TrendContinuationStrategy,
    )

    from .spreads import (
        CrossExchangeArbStrategy,
        FundingAdjustedBasisStrategy,
        SpreadMeanReversionStrategy,
        SpreadMomentumStrategy,
        SpotFuturesBasisStrategy,
    )

    from .spoofing import (
        CompositeSpoofingStrategy,
        FakeLiquidityTrapStrategy,
        LayeringTrapStrategy,
        OrderPullReversalStrategy,
        PressureBluffReversalStrategy,
        SpoofingAbsorptionReversalStrategy,
        SpoofingReversalStrategy,
    )

    from .whales import (
        WhaleAbsorptionStrategy,
        WhaleAccumulationStrategy,
        WhaleBreakoutStrategy,
        WhaleDistributionStrategy,
        WhaleLiquidationReversalStrategy,
    )