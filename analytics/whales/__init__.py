from __future__ import annotations

from analytics.whales.analyzer import WhaleAnalyzer
from analytics.whales.base import BaseWhaleAnalyzerComponent, BaseWhaleComponent
from analytics.whales.config import (
    LargeTradeDetectorConfig,
    WhaleClusterAnalyzerConfig,
    WhaleTrackerConfig,
    WhalesConfig,
)
from analytics.whales.enums import (
    LargeTradeTriggerType,
    WhaleBias,
    WhaleClusterStateType,
    WhaleComponentName,
    WhaleDataSource,
    WhaleEventType,
    WhaleNormalizationStatus,
    WhalePressureType,
    WhaleTradeSide,
)
from analytics.whales.large_trade_detector import LargeTradeDetector
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
)
from analytics.whales.whale_cluster_analyzer import WhaleClusterAnalyzer
from analytics.whales.whale_tracker import WhaleTracker

__all__ = [
    # facade
    "WhaleAnalyzer",

    # base
    "BaseWhaleComponent",
    "BaseWhaleAnalyzerComponent",

    # configs
    "LargeTradeDetectorConfig",
    "WhaleTrackerConfig",
    "WhaleClusterAnalyzerConfig",
    "WhalesConfig",

    # enums
    "WhaleTradeSide",
    "LargeTradeTriggerType",
    "WhaleEventType",
    "WhaleComponentName",
    "WhaleBias",
    "WhaleClusterStateType",
    "WhalePressureType",
    "WhaleNormalizationStatus",
    "WhaleDataSource",

    # detectors / analyzers
    "LargeTradeDetector",
    "WhaleTracker",
    "WhaleClusterAnalyzer",

    # base model
    "WhaleBaseEventModel",

    # normalized records
    "TradeRecord",
    "WhaleTradeRecord",
    "LiquidationRecord",
    "WhaleActivityRecord",
    "WhalePressureRecord",
    "WhaleLiquidationContextRecord",

    # signals
    "LargeTradeSignal",
    "WhaleActivitySignal",
    "WhalePressureSignal",
    "WhaleLiquidationContextSignal",
    "WhaleClusterSignal",
    "WhaleClusterUpdateSignal",
    "WhaleClusterExhaustionSignal",

    # states
    "SymbolStats",
    "SymbolTrackerState",
    "SymbolClusterState",

    # result models
    "WhaleTrackerResult",
    "WhaleClusterAnalysisResult",

    # factories
    "make_symbol_stats",
    "make_symbol_tracker_state",
    "make_symbol_cluster_state",
]