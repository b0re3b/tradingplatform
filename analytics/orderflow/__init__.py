from __future__ import annotations

# ---------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------

from .analyzer import OrderFlowAnalyzer

# ---------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------

from .base import BaseOrderFlowAnalyzer

# ---------------------------------------------------------------------
# Concrete analyzers
# ---------------------------------------------------------------------

from .aggressive_trades import AggressiveTradesAnalyzer
from .cvd import CvdAnalyzer
from .orderbook_imbalance import OrderbookImbalanceAnalyzer
from .volume_delta import VolumeDeltaAnalyzer

# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------

from .config import (
    AggressiveTradesConfig,
    BaseOrderFlowSubConfig,
    CvdConfig,
    OrderFlowConfig,
    OrderbookImbalanceConfig,
    VolumeDeltaConfig,
)

# ---------------------------------------------------------------------
# Enums / topics
# ---------------------------------------------------------------------

from .enums import (
    ORDERBOOK_INPUT_TOPICS,
    TRADE_INPUT_TOPICS,
    METRIC_SIGNAL_TOPICS,
    METRIC_UPDATE_TOPICS,
    OrderFlowEventTopic,
    OrderFlowInputTopic,
    OrderFlowMetricType,
    OrderFlowSide,
    OrderFlowSignalType,
    OrderFlowSourceType,
    get_signal_topic,
    get_update_topic,
)

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

from .models import (
    AggressiveTradesStats,
    BaseOrderFlowStats,
    CvdPoint,
    CvdStats,
    NormalizedTrade,
    OrderFlowSignal,
    OrderFlowUpdate,
    OrderbookImbalanceStats,
    OrderbookLevel,
    OrderbookSnapshot,
    VolumeDeltaStats,
    model_to_dict,
    orderbook_snapshot_to_dict,
    signal_to_dict,
    stats_to_dict,
    trade_to_dict,
    update_to_dict,
)


__all__ = [
    # -----------------------------------------------------------------
    # Facade
    # -----------------------------------------------------------------
    "OrderFlowAnalyzer",

    # -----------------------------------------------------------------
    # Base
    # -----------------------------------------------------------------
    "BaseOrderFlowAnalyzer",

    # -----------------------------------------------------------------
    # Concrete analyzers
    # -----------------------------------------------------------------
    "CvdAnalyzer",
    "VolumeDeltaAnalyzer",
    "AggressiveTradesAnalyzer",
    "OrderbookImbalanceAnalyzer",

    # -----------------------------------------------------------------
    # Configs
    # -----------------------------------------------------------------
    "BaseOrderFlowSubConfig",
    "OrderFlowConfig",
    "CvdConfig",
    "VolumeDeltaConfig",
    "AggressiveTradesConfig",
    "OrderbookImbalanceConfig",

    # -----------------------------------------------------------------
    # Enums / topics
    # -----------------------------------------------------------------
    "OrderFlowSide",
    "OrderFlowMetricType",
    "OrderFlowSignalType",
    "OrderFlowSourceType",
    "OrderFlowEventTopic",
    "OrderFlowInputTopic",
    "TRADE_INPUT_TOPICS",
    "ORDERBOOK_INPUT_TOPICS",
    "METRIC_UPDATE_TOPICS",
    "METRIC_SIGNAL_TOPICS",
    "get_update_topic",
    "get_signal_topic",

    # -----------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------
    "BaseOrderFlowStats",
    "NormalizedTrade",
    "OrderbookLevel",
    "OrderbookSnapshot",
    "OrderFlowUpdate",
    "OrderFlowSignal",
    "CvdPoint",
    "CvdStats",
    "VolumeDeltaStats",
    "AggressiveTradesStats",
    "OrderbookImbalanceStats",

    # -----------------------------------------------------------------
    # Serialization helpers
    # -----------------------------------------------------------------
    "model_to_dict",
    "stats_to_dict",
    "signal_to_dict",
    "update_to_dict",
    "orderbook_snapshot_to_dict",
    "trade_to_dict",
]