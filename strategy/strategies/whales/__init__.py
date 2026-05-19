# strategy/strategies/whales/__init__.py

from __future__ import annotations

# =============================================================================
# Base / shared whale strategy contracts
# =============================================================================

from strategy.strategies.whales.base import (
    DEFAULT_WHALE_CONTEXT_TOPIC,
    DEFAULT_WHALE_FEATURE_MAX_AGE_MS,
    FUTURES_MARKET_TYPES,
    LoggerLike,
    WhaleFeaturePayload,
    WhalePayloadValidation,
    WhaleStrategyBase,
    WhaleStrategyEventConfig,
    WhaleStrategyInputSnapshot,
)


# =============================================================================
# Concrete whale strategies
# =============================================================================

from strategy.strategies.whales.whale_absorption_strategy import (
    WhaleAbsorptionStrategy,
)
from strategy.strategies.whales.whale_breakout_strategy import (
    WhaleBreakoutStrategy,
)


__all__ = [
    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    "DEFAULT_WHALE_CONTEXT_TOPIC",
    "DEFAULT_WHALE_FEATURE_MAX_AGE_MS",
    "FUTURES_MARKET_TYPES",

    # -------------------------------------------------------------------------
    # Base / typing
    # -------------------------------------------------------------------------
    "LoggerLike",
    "WhaleStrategyBase",
    "WhaleStrategyEventConfig",

    # -------------------------------------------------------------------------
    # Normalized whale input contracts
    # -------------------------------------------------------------------------
    "WhalePayloadValidation",
    "WhaleFeaturePayload",
    "WhaleStrategyInputSnapshot",

    # -------------------------------------------------------------------------
    # Concrete strategies
    # -------------------------------------------------------------------------
    "WhaleAbsorptionStrategy",
    "WhaleBreakoutStrategy",
]