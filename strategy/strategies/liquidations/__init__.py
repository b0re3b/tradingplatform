from __future__ import annotations

from .base import (
    AnalyticsResultProtocol,
    AnalyticsStrategyConfigProtocol,
    BaseAnalyticsStrategy,
    BaseStrategyStats,
    BaseSymbolStrategyState,
    FilterResult,
    StrategyRejection,
    clamp_float,
    ensure_utc,
    make_strategy_scope_key,
    normalize_exchange,
    normalize_exchange_symbol,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    result_scope,
    scoped_key_to_string,
    serialize_value,
    signal_scope,
    utc_now,
)
from .liquidation_cascade_strategy import (
    LiquidationCascadeSignal,
    LiquidationCascadeStrategy,
    LiquidationCascadeStrategyConfig,
    LiquidationCascadeStrategyStats,
    SymbolCascadeStrategyState,
)
from .squeeze_reversal_strategy import (
    PendingReversalCandidate,
    SqueezeReversalSignal,
    SqueezeReversalStrategy,
    SqueezeReversalStrategyConfig,
    SqueezeReversalStrategyStats,
    SymbolSqueezeStrategyState,
)


# Backward-compatible aliases.
CascadeStrategyRejection = StrategyRejection
SqueezeStrategyRejection = StrategyRejection


__all__ = [
    # Base protocols
    "AnalyticsResultProtocol",
    "AnalyticsStrategyConfigProtocol",

    # Base strategy infrastructure
    "BaseAnalyticsStrategy",
    "BaseStrategyStats",
    "BaseSymbolStrategyState",
    "FilterResult",
    "StrategyRejection",

    # Base helpers
    "utc_now",
    "ensure_utc",
    "normalize_exchange",
    "normalize_symbol",
    "normalize_market_type",
    "normalize_timeframe",
    "normalize_exchange_symbol",
    "make_strategy_scope_key",
    "scoped_key_to_string",
    "result_scope",
    "signal_scope",
    "clamp_float",
    "serialize_value",

    # Liquidation cascade continuation strategy
    "LiquidationCascadeSignal",
    "LiquidationCascadeStrategy",
    "LiquidationCascadeStrategyConfig",
    "LiquidationCascadeStrategyStats",
    "SymbolCascadeStrategyState",

    # Squeeze reversal strategy
    "PendingReversalCandidate",
    "SqueezeReversalSignal",
    "SqueezeReversalStrategy",
    "SqueezeReversalStrategyConfig",
    "SqueezeReversalStrategyStats",
    "SymbolSqueezeStrategyState",

    # Backward-compatible aliases
    "CascadeStrategyRejection",
    "SqueezeStrategyRejection",
]