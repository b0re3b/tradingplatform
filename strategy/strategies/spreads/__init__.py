from __future__ import annotations

# ============================================================
# Base contracts
# ============================================================

from .base_spread_strategy import (
    # Input analytics.spreads events
    SPOT_FUTURES_SNAPSHOT_EVENT,
    CROSS_EXCHANGE_SNAPSHOT_EVENT,
    SPREAD_SIGNAL_EVENT,
    ARBITRAGE_OPPORTUNITY_EVENT,

    # Output strategy events
    STRATEGY_SIGNAL_GENERATED_EVENT,
    STRATEGY_SIGNAL_UPDATED_EVENT,
    STRATEGY_SIGNAL_REJECTED_EVENT,
    STRATEGY_SIGNAL_CANCELLED_EVENT,
    STRATEGY_SIGNAL_CLOSED_EVENT,
    STRATEGY_STARTED_EVENT,
    STRATEGY_STOPPED_EVENT,
    STRATEGY_HEARTBEAT_EVENT,

    # State constants
    STATE_IDLE,
    STATE_PENDING,
    STATE_OPEN,
    STATE_BLOCKED,
    STATE_CLOSING,
    STATE_CLOSED,
    STATE_CANCELLED,
    STATE_REJECTED,
    ACTIVE_STATES,
    CLOSED_STATES,

    # Types / base classes
    PayloadHandler,
    EventHandler,
    BaseSpreadStrategyConfig,
    SpreadStrategyState,
    BaseSpreadStrategy,
)


# ============================================================
# Concrete spread strategies
# ============================================================

from .spot_futures_basis_strategy import (
    SpotFuturesBasisStrategy,
    SpotFuturesBasisStrategyConfig,
)

from .cross_exchange_arb_strategy import (
    CrossExchangeArbStrategy,
    CrossExchangeArbStrategyConfig,
)


__all__ = [
    # ========================================================
    # Input analytics.spreads events
    # ========================================================
    "SPOT_FUTURES_SNAPSHOT_EVENT",
    "CROSS_EXCHANGE_SNAPSHOT_EVENT",
    "SPREAD_SIGNAL_EVENT",
    "ARBITRAGE_OPPORTUNITY_EVENT",

    # ========================================================
    # Output strategy events
    # ========================================================
    "STRATEGY_SIGNAL_GENERATED_EVENT",
    "STRATEGY_SIGNAL_UPDATED_EVENT",
    "STRATEGY_SIGNAL_REJECTED_EVENT",
    "STRATEGY_SIGNAL_CANCELLED_EVENT",
    "STRATEGY_SIGNAL_CLOSED_EVENT",
    "STRATEGY_STARTED_EVENT",
    "STRATEGY_STOPPED_EVENT",
    "STRATEGY_HEARTBEAT_EVENT",

    # ========================================================
    # State constants
    # ========================================================
    "STATE_IDLE",
    "STATE_PENDING",
    "STATE_OPEN",
    "STATE_BLOCKED",
    "STATE_CLOSING",
    "STATE_CLOSED",
    "STATE_CANCELLED",
    "STATE_REJECTED",
    "ACTIVE_STATES",
    "CLOSED_STATES",

    # ========================================================
    # Base contracts
    # ========================================================
    "PayloadHandler",
    "EventHandler",
    "BaseSpreadStrategyConfig",
    "SpreadStrategyState",
    "BaseSpreadStrategy",

    # ========================================================
    # Spot/Futures basis strategy
    # ========================================================
    "SpotFuturesBasisStrategy",
    "SpotFuturesBasisStrategyConfig",

    # ========================================================
    # Cross-exchange arbitrage strategy
    # ========================================================
    "CrossExchangeArbStrategy",
    "CrossExchangeArbStrategyConfig",
]