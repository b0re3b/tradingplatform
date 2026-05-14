from __future__ import annotations

# =============================================================================
# Facade
# =============================================================================

from analytics.whales.analyzer import WhaleAnalyzer


# =============================================================================
# Base
# =============================================================================

from analytics.whales.base import (
    BaseWhaleAnalyzerComponent,
    BaseWhaleComponent,
    EventHandler,
    JobCallable,
)


# =============================================================================
# Configs
# =============================================================================

from analytics.whales.config import (
    LargeTradeDetectorConfig,
    WhaleClusterAnalyzerConfig,
    WhaleTrackerConfig,
    WhalesConfig,
)


# =============================================================================
# Enums
# =============================================================================

from analytics.whales.enums import (
    LargeTradeTriggerType,
    WhaleBias,
    WhaleClusterStateType,
    WhaleComponentName,
    WhaleDataSource,
    WhaleEventTopic,
    WhaleEventType,
    WhaleNormalizationStatus,
    WhalePressureType,
    WhaleTradeSide,
)


# =============================================================================
# Runtime components
# =============================================================================

from analytics.whales.large_trade_detector import LargeTradeDetector
from analytics.whales.whale_cluster_analyzer import WhaleClusterAnalyzer
from analytics.whales.whale_tracker import WhaleTracker


# =============================================================================
# Models
# =============================================================================

from analytics.whales.models import (
    LargeTradeSignal,
    LiquidationRecord,
    SymbolClusterState,
    SymbolStats,
    SymbolTrackerState,
    TradeRecord,
    WhaleActivityRecord,
    WhaleActivitySignal,
    WhaleBaseEventModel,
    WhaleBaseSignalModel,
    WhaleClusterAnalysisResult,
    WhaleClusterExhaustionSignal,
    WhaleClusterSignal,
    WhaleClusterUpdateSignal,
    WhaleLiquidationContextRecord,
    WhaleLiquidationContextSignal,
    WhalePressureRecord,
    WhalePressureSignal,
    WhaleTradeRecord,
    WhaleTrackerResult,
    make_symbol_cluster_state,
    make_symbol_stats,
    make_symbol_tracker_state,
    utc_now_ms,
)


__all__ = [
    # Facade
    "WhaleAnalyzer",

    # Base
    "BaseWhaleComponent",
    "BaseWhaleAnalyzerComponent",
    "EventHandler",
    "JobCallable",

    # Configs
    "LargeTradeDetectorConfig",
    "WhaleTrackerConfig",
    "WhaleClusterAnalyzerConfig",
    "WhalesConfig",

    # Enums
    "WhaleTradeSide",
    "LargeTradeTriggerType",
    "WhaleEventType",
    "WhaleEventTopic",
    "WhaleComponentName",
    "WhaleBias",
    "WhaleClusterStateType",
    "WhalePressureType",
    "WhaleNormalizationStatus",
    "WhaleDataSource",

    # Runtime components
    "LargeTradeDetector",
    "WhaleTracker",
    "WhaleClusterAnalyzer",

    # Base models
    "WhaleBaseSignalModel",
    "WhaleBaseEventModel",

    # Normalized records
    "TradeRecord",
    "WhaleTradeRecord",
    "LiquidationRecord",
    "WhaleActivityRecord",
    "WhalePressureRecord",
    "WhaleLiquidationContextRecord",

    # Signals
    "LargeTradeSignal",
    "WhaleActivitySignal",
    "WhalePressureSignal",
    "WhaleLiquidationContextSignal",
    "WhaleClusterSignal",
    "WhaleClusterUpdateSignal",
    "WhaleClusterExhaustionSignal",

    # States
    "SymbolStats",
    "SymbolTrackerState",
    "SymbolClusterState",

    # Result models
    "WhaleTrackerResult",
    "WhaleClusterAnalysisResult",

    # Factories
    "make_symbol_stats",
    "make_symbol_tracker_state",
    "make_symbol_cluster_state",

    # Helpers
    "utc_now_ms",
]