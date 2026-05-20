from __future__ import annotations

from .base import (
    SPREADS_FEATURES,
    SpreadCompositeSnapshot,
    SpreadsFeatureNames,
    SpreadsStrategyConfig,
    SpreadsStrategyScope,
    SpreadsTradingStrategy,
)
from .cross_exchange_arb_strategy import (
    CrossExchangeArbPayload,
    CrossExchangeArbStrategy,
    CrossExchangeArbStrategyConfig,
)
from .funding_adjusted_basis_strategy import (
    FundingAdjustedBasisPayload,
    FundingAdjustedBasisStrategy,
    FundingAdjustedBasisStrategyConfig,
)
from .spot_futures_basis_strategy import (
    SpotFuturesBasisPayload,
    SpotFuturesBasisStrategy,
    SpotFuturesBasisStrategyConfig,
)
from .spread_mean_reversion_strategy import (
    SpreadMeanReversionPayload,
    SpreadMeanReversionStrategy,
    SpreadMeanReversionStrategyConfig,
)
from .spread_momentum_strategy import (
    SpreadMomentumPayload,
    SpreadMomentumStrategy,
    SpreadMomentumStrategyConfig,
)

__all__ = [
    # Feature contract
    "SPREADS_FEATURES",
    "SpreadsFeatureNames",

    # Base
    "SpreadCompositeSnapshot",
    "SpreadsStrategyConfig",
    "SpreadsStrategyScope",
    "SpreadsTradingStrategy",

    # Spot/futures basis
    "SpotFuturesBasisPayload",
    "SpotFuturesBasisStrategy",
    "SpotFuturesBasisStrategyConfig",

    # Cross-exchange arbitrage
    "CrossExchangeArbPayload",
    "CrossExchangeArbStrategy",
    "CrossExchangeArbStrategyConfig",

    # Funding-adjusted basis
    "FundingAdjustedBasisPayload",
    "FundingAdjustedBasisStrategy",
    "FundingAdjustedBasisStrategyConfig",

    # Generic spread mean reversion
    "SpreadMeanReversionPayload",
    "SpreadMeanReversionStrategy",
    "SpreadMeanReversionStrategyConfig",

    # Generic spread momentum
    "SpreadMomentumPayload",
    "SpreadMomentumStrategy",
    "SpreadMomentumStrategyConfig",
]