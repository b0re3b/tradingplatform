from __future__ import annotations

from .base import (
    WHALES_FEATURES,
    WhaleCompositeSnapshot,
    WhalesFeatureNames,
    WhalesStrategyConfig,
    WhalesStrategyScope,
    WhalesTradingStrategy,
)
from .whale_absorption_strategy import (
    WhaleAbsorptionPayload,
    WhaleAbsorptionStrategy,
    WhaleAbsorptionStrategyConfig,
)
from .whale_accumulation_strategy import (
    WhaleAccumulationPayload,
    WhaleAccumulationStrategy,
    WhaleAccumulationStrategyConfig,
)
from .whale_breakout_strategy import (
    WhaleBreakoutPayload,
    WhaleBreakoutStrategy,
    WhaleBreakoutStrategyConfig,
)
from .whale_distribution_strategy import (
    WhaleDistributionPayload,
    WhaleDistributionStrategy,
    WhaleDistributionStrategyConfig,
)

from .whale_large_trade_strategy import (
    WhaleLargeTradePayload,
    WhaleLargeTradeStrategy,
    WhaleLargeTradeStrategyConfig,
)

from .whale_liquidation_reversal_strategy import (
    WhaleLiquidationReversalPayload,
    WhaleLiquidationReversalStrategy,
    WhaleLiquidationReversalStrategyConfig,
)

__all__ = [
    # Feature contract
    "WHALES_FEATURES",
    "WhalesFeatureNames",

    # Base
    "WhaleCompositeSnapshot",
    "WhalesStrategyConfig",
    "WhalesStrategyScope",
    "WhalesTradingStrategy",

    # Absorption
    "WhaleAbsorptionPayload",
    "WhaleAbsorptionStrategy",
    "WhaleAbsorptionStrategyConfig",

    # Accumulation
    "WhaleAccumulationPayload",
    "WhaleAccumulationStrategy",
    "WhaleAccumulationStrategyConfig",

    # Breakout
    "WhaleBreakoutPayload",
    "WhaleBreakoutStrategy",
    "WhaleBreakoutStrategyConfig",

    # Distribution
    "WhaleDistributionPayload",
    "WhaleDistributionStrategy",
    "WhaleDistributionStrategyConfig",

    # Large trade
    "WhaleLargeTradePayload",
    "WhaleLargeTradeStrategy",
    "WhaleLargeTradeStrategyConfig",

    # Liquidation reversal
    "WhaleLiquidationReversalPayload",
    "WhaleLiquidationReversalStrategy",
    "WhaleLiquidationReversalStrategyConfig",
]