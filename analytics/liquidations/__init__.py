from __future__ import annotations

from typing import Any

from .cascade_detector import CascadeDetector
from .config import (
    CascadeDetectorConfig,
    LiquidationMetricsConfig,
    LiquidationStreamConfig,
    LiquidationsConfig,
)
from .enums import (
    CascadeDirection,
    CascadeSeverity,
    LiquidationEventType,
    LiquidationSide,
    LiquidationStatus,
)
from .liquidation_stream import (
    LiquidationHistoryStoreProtocol,
    LiquidationStream,
)
from .metrics import (
    LatencyHistogram,
    LiquidationMetrics,
    LiquidationMetricsSnapshot,
)
from .models import (
    DEFAULT_LARGE_LIQUIDATION_THRESHOLD_USD,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    DECIMAL_ZERO,
    CascadeDetectionResult,
    LiquidationBufferSnapshot,
    LiquidationCluster,
    LiquidationEvent,
    LiquidationKey,
    LiquidationScopedModel,
    LiquidationWindowStats,
    liquidation_key_to_dict,
    make_liquidation_key,
    normalize_exchange,
    normalize_exchange_symbol,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    scoped_metadata,
)
from .state import (
    LiquidationState,
    SymbolLiquidationState,
    get_shared_liquidation_state,
    reset_shared_liquidation_state,
)
from .utils import (
    build_cluster_from_events,
    build_cluster_from_events_for_key,
    build_key_from_event,
    build_key_from_payload,
    build_liquidation_key,
    build_symbol_key,
    clamp_float,
    compute_acceleration_ratio,
    compute_weighted_average_price,
    compute_window_stats,
    compute_window_stats_for_key,
    ensure_same_scope,
    ensure_utc,
    filter_events_by_key,
    filter_events_by_scope,
    filter_events_by_side,
    filter_valid_events,
    infer_scope_from_events,
    infer_severity,
    is_stale_event,
    key_to_scope,
    normalize_score,
    prune_events_by_window,
    prune_events_older_than,
    safe_decimal,
    safe_float,
    scoped_key_to_string,
    side_to_direction,
    sort_events_by_timestamp,
    split_events_in_halves,
    sum_notional,
    sum_quantity,
    utc_now,
)


def build_liquidation_stream_and_cascade_detector(
    *,
    event_bus: Any,
    scheduler: Any | None = None,
    stream_config: LiquidationStreamConfig | None = None,
    cascade_config: CascadeDetectorConfig | None = None,
    state: LiquidationState | None = None,
    metrics: LiquidationMetrics | None = None,
    history_store: LiquidationHistoryStoreProtocol | None = None,
    market_state_store: Any | None = None,
    market_scheduler: Any | None = None,
) -> tuple[LiquidationStream, CascadeDetector]:
    """Build LiquidationStream and CascadeDetector with one shared state.

    Kept directly in package __init__ to avoid adding a separate pipeline.py
    while still giving app/factories.py a safe production helper.
    """
    resolved_stream_config = stream_config or LiquidationStreamConfig()
    resolved_state = state or get_shared_liquidation_state(
        max_events_per_symbol=resolved_stream_config.max_buffer_size_per_symbol,
    )
    resolved_metrics = metrics or LiquidationMetrics()

    stream = LiquidationStream(
        event_bus=event_bus,
        scheduler=scheduler,
        config=resolved_stream_config,
        state=resolved_state,
        metrics=resolved_metrics,
        history_store=history_store,
        market_state_store=market_state_store,
        market_scheduler=market_scheduler,
    )
    detector = CascadeDetector(
        event_bus=event_bus,
        scheduler=scheduler,
        config=cascade_config or CascadeDetectorConfig(),
        state=resolved_state,
        metrics=resolved_metrics,
        market_state_store=market_state_store,
    )
    return stream, detector


def build_liquidations_components(
    *,
    event_bus: Any,
    scheduler: Any | None = None,
    stream_config: LiquidationStreamConfig | None = None,
    cascade_config: CascadeDetectorConfig | None = None,
    state: LiquidationState | None = None,
    metrics: LiquidationMetrics | None = None,
    history_store: LiquidationHistoryStoreProtocol | None = None,
    market_state_store: Any | None = None,
    market_scheduler: Any | None = None,
) -> list[Any]:
    """Return liquidation analytics components using one shared state."""
    stream, detector = build_liquidation_stream_and_cascade_detector(
        event_bus=event_bus,
        scheduler=scheduler,
        stream_config=stream_config,
        cascade_config=cascade_config,
        state=state,
        metrics=metrics,
        history_store=history_store,
        market_state_store=market_state_store,
        market_scheduler=market_scheduler,
    )
    return [stream, detector]


__all__ = [
    # Runtime classes
    "LiquidationStream",
    "LiquidationHistoryStoreProtocol",
    "CascadeDetector",
    "build_liquidation_stream_and_cascade_detector",
    "build_liquidations_components",

    # Configs
    "LiquidationsConfig",
    "LiquidationStreamConfig",
    "CascadeDetectorConfig",
    "LiquidationMetricsConfig",

    # Enums
    "LiquidationSide",
    "LiquidationEventType",
    "CascadeDirection",
    "CascadeSeverity",
    "LiquidationStatus",

    # Models
    "LiquidationScopedModel",
    "LiquidationEvent",
    "LiquidationCluster",
    "CascadeDetectionResult",
    "LiquidationWindowStats",
    "LiquidationBufferSnapshot",
    "LiquidationKey",

    # State
    "LiquidationState",
    "SymbolLiquidationState",
    "get_shared_liquidation_state",
    "reset_shared_liquidation_state",

    # Metrics
    "LatencyHistogram",
    "LiquidationMetrics",
    "LiquidationMetricsSnapshot",

    # Constants
    "DECIMAL_ZERO",
    "DEFAULT_MARKET_TYPE",
    "DEFAULT_TIMEFRAME",
    "DEFAULT_LARGE_LIQUIDATION_THRESHOLD_USD",

    # Scope / normalization helpers
    "make_liquidation_key",
    "liquidation_key_to_dict",
    "scoped_metadata",
    "normalize_exchange",
    "normalize_symbol",
    "normalize_exchange_symbol",
    "normalize_market_type",
    "normalize_timeframe",
    "build_liquidation_key",
    "build_key_from_event",
    "build_key_from_payload",
    "key_to_scope",
    "scoped_key_to_string",
    "ensure_same_scope",
    "infer_scope_from_events",

    # Backward-compatible legacy helper
    "build_symbol_key",

    # Time / parsing helpers
    "utc_now",
    "ensure_utc",
    "safe_decimal",
    "safe_float",

    # Filtering helpers
    "filter_events_by_side",
    "filter_events_by_key",
    "filter_events_by_scope",
    "filter_valid_events",
    "prune_events_older_than",
    "prune_events_by_window",
    "sort_events_by_timestamp",

    # Aggregation / scoring helpers
    "side_to_direction",
    "sum_notional",
    "sum_quantity",
    "compute_weighted_average_price",
    "compute_window_stats",
    "compute_window_stats_for_key",
    "split_events_in_halves",
    "compute_acceleration_ratio",
    "clamp_float",
    "normalize_score",
    "infer_severity",
    "is_stale_event",
    "build_cluster_from_events",
    "build_cluster_from_events_for_key",
]