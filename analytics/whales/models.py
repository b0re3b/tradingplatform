from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from analytics.whales.enums import (
    LargeTradeTriggerType,
    WhaleBias,
    WhaleClusterStateType,
    WhaleEventType,
    WhalePressureType,
    WhaleTradeSide,
)


# =============================================================================
# Base models
# =============================================================================


@dataclass(slots=True)
class WhaleBaseEventModel:
    """
    Базова модель для всіх whale-сигналів.
    """

    detector_name: str
    event_type: str
    schema_version: int = 1
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_event(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "detector": self.detector_name,
            "created_at_ms": self.created_at_ms,
        }


# =============================================================================
# Raw / normalized records
# =============================================================================


@dataclass(slots=True)
class TradeRecord:
    """
    Нормалізований raw trade record.
    """

    symbol: str
    price: float
    quantity: float
    side: str
    timestamp_ms: int
    trade_id: Optional[str] = None
    exchange: Optional[str] = None
    raw_event: Optional[Dict[str, Any]] = None

    @property
    def notional(self) -> float:
        return self.price * self.quantity


@dataclass(slots=True)
class WhaleTradeRecord:
    """
    Нормалізований record для вже виявленого великого трейду.
    """

    symbol: str
    side: str
    notional: float
    price: float
    quantity: float
    timestamp_ms: int
    zscore: float = 0.0
    trigger_type: str = LargeTradeTriggerType.UNKNOWN.value
    trade_id: Optional[str] = None
    exchange: Optional[str] = None
    raw_event: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class LiquidationRecord:
    """
    Нормалізований liquidation record.
    """

    symbol: str
    side: str
    notional: float
    price: float
    quantity: float
    timestamp_ms: int
    liquidation_id: Optional[str] = None
    exchange: Optional[str] = None
    raw_event: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class WhaleActivityRecord:
    symbol: str
    side: str
    trade_count: int
    total_notional: float
    avg_notional: float
    max_notional: float
    window_sec: int
    timestamp_ms: int
    raw_event: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class WhalePressureRecord:
    symbol: str
    dominant_side: str
    buy_trade_count: int
    sell_trade_count: int
    buy_notional: float
    sell_notional: float
    total_notional: float
    imbalance_ratio: float
    net_flow_notional: float
    window_sec: int
    timestamp_ms: int
    raw_event: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class WhaleLiquidationContextRecord:
    symbol: str
    whale_side: str
    whale_total_notional: float
    whale_trade_count: int
    liquidation_side: str
    liquidation_total_notional: float
    liquidation_count: int
    context_strength: float
    timestamp_ms: int
    raw_event: Optional[Dict[str, Any]] = None


# =============================================================================
# Signal models
# =============================================================================


@dataclass(slots=True)
class LargeTradeSignal(WhaleBaseEventModel):
    symbol: str = ""
    side: str = WhaleTradeSide.UNKNOWN.value
    price: float = 0.0
    quantity: float = 0.0
    notional: float = 0.0
    timestamp_ms: int = 0

    abs_threshold: float = 0.0
    mean_notional: float = 0.0
    std_notional: float = 0.0
    zscore: float = 0.0

    trigger_type: str = LargeTradeTriggerType.UNKNOWN.value
    trade_id: Optional[str] = None
    exchange: Optional[str] = None

    detector_name: str = "LargeTradeDetector"
    event_type: str = WhaleEventType.LARGE_TRADE.value

    def to_event(self) -> Dict[str, Any]:
        base = super().to_event()
        base.update(
            {
                "symbol": self.symbol,
                "side": self.side,
                "price": self.price,
                "quantity": self.quantity,
                "notional": self.notional,
                "timestamp_ms": self.timestamp_ms,
                "abs_threshold": self.abs_threshold,
                "mean_notional": self.mean_notional,
                "std_notional": self.std_notional,
                "zscore": self.zscore,
                "trigger_type": self.trigger_type,
                "trade_id": self.trade_id,
                "exchange": self.exchange,
            }
        )
        return base


@dataclass(slots=True)
class WhaleActivitySignal(WhaleBaseEventModel):
    symbol: str = ""
    side: str = WhaleTradeSide.UNKNOWN.value
    trade_count: int = 0
    total_notional: float = 0.0
    avg_notional: float = 0.0
    max_notional: float = 0.0
    window_sec: int = 0
    timestamp_ms: int = 0

    detector_name: str = "WhaleTracker"
    event_type: str = WhaleEventType.WHALE_ACTIVITY.value

    def to_event(self) -> Dict[str, Any]:
        base = super().to_event()
        base.update(
            {
                "symbol": self.symbol,
                "side": self.side,
                "trade_count": self.trade_count,
                "total_notional": self.total_notional,
                "avg_notional": self.avg_notional,
                "max_notional": self.max_notional,
                "window_sec": self.window_sec,
                "timestamp_ms": self.timestamp_ms,
            }
        )
        return base


@dataclass(slots=True)
class WhalePressureSignal(WhaleBaseEventModel):
    symbol: str = ""
    dominant_side: str = WhaleTradeSide.UNKNOWN.value
    buy_trade_count: int = 0
    sell_trade_count: int = 0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    total_notional: float = 0.0
    imbalance_ratio: float = 0.0
    net_flow_notional: float = 0.0
    window_sec: int = 0
    timestamp_ms: int = 0

    detector_name: str = "WhaleTracker"
    event_type: str = WhaleEventType.WHALE_PRESSURE.value

    @property
    def pressure_type(self) -> str:
        if self.buy_notional > self.sell_notional:
            return WhalePressureType.BUY_PRESSURE.value
        if self.sell_notional > self.buy_notional:
            return WhalePressureType.SELL_PRESSURE.value
        return WhalePressureType.BALANCED.value

    def to_event(self) -> Dict[str, Any]:
        base = super().to_event()
        base.update(
            {
                "symbol": self.symbol,
                "dominant_side": self.dominant_side,
                "buy_trade_count": self.buy_trade_count,
                "sell_trade_count": self.sell_trade_count,
                "buy_notional": self.buy_notional,
                "sell_notional": self.sell_notional,
                "total_notional": self.total_notional,
                "imbalance_ratio": self.imbalance_ratio,
                "net_flow_notional": self.net_flow_notional,
                "pressure_type": self.pressure_type,
                "window_sec": self.window_sec,
                "timestamp_ms": self.timestamp_ms,
            }
        )
        return base


@dataclass(slots=True)
class WhaleLiquidationContextSignal(WhaleBaseEventModel):
    symbol: str = ""
    whale_side: str = WhaleTradeSide.UNKNOWN.value
    whale_total_notional: float = 0.0
    whale_trade_count: int = 0
    liquidation_side: str = WhaleTradeSide.UNKNOWN.value
    liquidation_total_notional: float = 0.0
    liquidation_count: int = 0
    context_strength: float = 0.0
    timestamp_ms: int = 0

    detector_name: str = "WhaleTracker"
    event_type: str = WhaleEventType.WHALE_LIQUIDATION_CONTEXT.value

    def to_event(self) -> Dict[str, Any]:
        base = super().to_event()
        base.update(
            {
                "symbol": self.symbol,
                "whale_side": self.whale_side,
                "whale_total_notional": self.whale_total_notional,
                "whale_trade_count": self.whale_trade_count,
                "liquidation_side": self.liquidation_side,
                "liquidation_total_notional": self.liquidation_total_notional,
                "liquidation_count": self.liquidation_count,
                "context_strength": self.context_strength,
                "timestamp_ms": self.timestamp_ms,
            }
        )
        return base


@dataclass(slots=True)
class WhaleClusterSignal(WhaleBaseEventModel):
    symbol: str = ""
    cluster_side: str = WhaleTradeSide.UNKNOWN.value
    cluster_score: float = 0.0
    persistence_score: float = 0.0
    directional_bias: float = 0.0
    continuation_probability: float = 0.0
    exhaustion_probability: float = 0.0

    activity_signal_count: int = 0
    pressure_signal_count: int = 0
    liquidation_context_count: int = 0

    total_activity_notional: float = 0.0
    total_pressure_notional: float = 0.0
    total_liquidation_context_notional: float = 0.0

    first_seen_ts_ms: int = 0
    last_seen_ts_ms: int = 0
    timestamp_ms: int = 0

    detector_name: str = "WhaleClusterAnalyzer"
    event_type: str = WhaleEventType.WHALE_CLUSTER.value

    @property
    def bias(self) -> str:
        if self.cluster_side == WhaleTradeSide.BUY.value:
            return WhaleBias.BULLISH.value
        if self.cluster_side == WhaleTradeSide.SELL.value:
            return WhaleBias.BEARISH.value
        return WhaleBias.UNKNOWN.value

    @property
    def state(self) -> str:
        if self.exhaustion_probability >= 0.7:
            return WhaleClusterStateType.EXHAUSTING.value
        if self.cluster_score >= 0.6:
            return WhaleClusterStateType.ACTIVE.value
        if self.cluster_score > 0.0:
            return WhaleClusterStateType.FORMING.value
        return WhaleClusterStateType.INACTIVE.value

    def to_event(self) -> Dict[str, Any]:
        base = super().to_event()
        base.update(
            {
                "symbol": self.symbol,
                "cluster_side": self.cluster_side,
                "cluster_score": self.cluster_score,
                "persistence_score": self.persistence_score,
                "directional_bias": self.directional_bias,
                "continuation_probability": self.continuation_probability,
                "exhaustion_probability": self.exhaustion_probability,
                "activity_signal_count": self.activity_signal_count,
                "pressure_signal_count": self.pressure_signal_count,
                "liquidation_context_count": self.liquidation_context_count,
                "total_activity_notional": self.total_activity_notional,
                "total_pressure_notional": self.total_pressure_notional,
                "total_liquidation_context_notional": self.total_liquidation_context_notional,
                "first_seen_ts_ms": self.first_seen_ts_ms,
                "last_seen_ts_ms": self.last_seen_ts_ms,
                "timestamp_ms": self.timestamp_ms,
                "bias": self.bias,
                "cluster_state": self.state,
            }
        )
        return base


@dataclass(slots=True)
class WhaleClusterUpdateSignal(WhaleBaseEventModel):
    symbol: str = ""
    cluster_side: str = WhaleTradeSide.UNKNOWN.value
    cluster_score: float = 0.0
    persistence_score: float = 0.0
    continuation_probability: float = 0.0
    exhaustion_probability: float = 0.0
    activity_signal_count: int = 0
    pressure_signal_count: int = 0
    liquidation_context_count: int = 0
    timestamp_ms: int = 0

    detector_name: str = "WhaleClusterAnalyzer"
    event_type: str = WhaleEventType.WHALE_CLUSTER_UPDATE.value

    def to_event(self) -> Dict[str, Any]:
        base = super().to_event()
        base.update(
            {
                "symbol": self.symbol,
                "cluster_side": self.cluster_side,
                "cluster_score": self.cluster_score,
                "persistence_score": self.persistence_score,
                "continuation_probability": self.continuation_probability,
                "exhaustion_probability": self.exhaustion_probability,
                "activity_signal_count": self.activity_signal_count,
                "pressure_signal_count": self.pressure_signal_count,
                "liquidation_context_count": self.liquidation_context_count,
                "timestamp_ms": self.timestamp_ms,
            }
        )
        return base


@dataclass(slots=True)
class WhaleClusterExhaustionSignal(WhaleBaseEventModel):
    symbol: str = ""
    cluster_side: str = WhaleTradeSide.UNKNOWN.value
    cluster_score: float = 0.0
    exhaustion_probability: float = 0.0
    reversal_risk: float = 0.0
    timestamp_ms: int = 0

    detector_name: str = "WhaleClusterAnalyzer"
    event_type: str = WhaleEventType.WHALE_CLUSTER_EXHAUSTION.value

    def to_event(self) -> Dict[str, Any]:
        base = super().to_event()
        base.update(
            {
                "symbol": self.symbol,
                "cluster_side": self.cluster_side,
                "cluster_score": self.cluster_score,
                "exhaustion_probability": self.exhaustion_probability,
                "reversal_risk": self.reversal_risk,
                "timestamp_ms": self.timestamp_ms,
            }
        )
        return base


# =============================================================================
# Internal rolling / symbol states
# =============================================================================


@dataclass(slots=True)
class SymbolStats:
    """
    Rolling-статистика для LargeTradeDetector.
    """

    notionals: Deque[float]
    trades_processed: int = 0
    signals_emitted: int = 0
    last_signal_ts_monotonic: float = 0.0
    last_update_ts_monotonic: float = field(default_factory=time.monotonic)

    running_sum: float = 0.0
    running_sum_sq: float = 0.0
    updates_since_recalibration: int = 0

    def touch(self) -> None:
        self.last_update_ts_monotonic = time.monotonic()

    @property
    def sample_size(self) -> int:
        return len(self.notionals)

    def add(self, value: float, recalibration_interval: int) -> None:
        if self.notionals.maxlen is not None and len(self.notionals) == self.notionals.maxlen:
            evicted = self.notionals[0]
            self.running_sum -= evicted
            self.running_sum_sq -= evicted * evicted

        self.notionals.append(value)
        self.running_sum += value
        self.running_sum_sq += value * value
        self.updates_since_recalibration += 1
        self.touch()

        if recalibration_interval > 0 and self.updates_since_recalibration >= recalibration_interval:
            self.recalibrate()

    def recalibrate(self) -> None:
        values = list(self.notionals)
        self.running_sum = math.fsum(values)
        self.running_sum_sq = math.fsum(x * x for x in values)
        self.updates_since_recalibration = 0

    def mean(self) -> float:
        n = len(self.notionals)
        if n == 0:
            return 0.0
        return self.running_sum / n

    def std(self) -> float:
        n = len(self.notionals)
        if n < 2:
            return 0.0

        mean_value = self.running_sum / n
        numerator = self.running_sum_sq - (n * mean_value * mean_value)
        numerator = max(numerator, 0.0)
        variance = numerator / (n - 1)
        return math.sqrt(max(variance, 0.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "trades_processed": self.trades_processed,
            "signals_emitted": self.signals_emitted,
            "mean_notional": self.mean(),
            "std_notional": self.std(),
            "last_signal_ts_monotonic": self.last_signal_ts_monotonic,
            "last_update_ts_monotonic": self.last_update_ts_monotonic,
        }


@dataclass(slots=True)
class SymbolTrackerState:
    """
    Rolling-state для WhaleTracker.
    """

    large_trades: Deque[WhaleTradeRecord]
    liquidations: Deque[LiquidationRecord]

    total_large_trades_seen: int = 0
    total_liquidations_seen: int = 0

    whale_activity_signals_emitted: int = 0
    whale_pressure_signals_emitted: int = 0
    whale_liquidation_context_signals_emitted: int = 0

    last_whale_activity_signal_ts_monotonic: float = 0.0
    last_whale_pressure_signal_ts_monotonic: float = 0.0
    last_whale_liquidation_context_signal_ts_monotonic: float = 0.0

    last_update_ts_monotonic: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_update_ts_monotonic = time.monotonic()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "large_trades_buffer_size": len(self.large_trades),
            "liquidations_buffer_size": len(self.liquidations),
            "total_large_trades_seen": self.total_large_trades_seen,
            "total_liquidations_seen": self.total_liquidations_seen,
            "whale_activity_signals_emitted": self.whale_activity_signals_emitted,
            "whale_pressure_signals_emitted": self.whale_pressure_signals_emitted,
            "whale_liquidation_context_signals_emitted": self.whale_liquidation_context_signals_emitted,
            "last_update_ts_monotonic": self.last_update_ts_monotonic,
        }


@dataclass(slots=True)
class SymbolClusterState:
    """
    Rolling-state для WhaleClusterAnalyzer.
    """

    activity_records: Deque[WhaleActivityRecord]
    pressure_records: Deque[WhalePressureRecord]
    liquidation_context_records: Deque[WhaleLiquidationContextRecord]

    total_events_seen: int = 0
    total_clusters_emitted: int = 0
    total_cluster_updates_emitted: int = 0
    total_cluster_exhaustions_emitted: int = 0

    cluster_first_seen_ts_ms: Optional[int] = None
    cluster_last_seen_ts_ms: Optional[int] = None

    last_cluster_emit_ts_monotonic: float = 0.0
    last_cluster_update_emit_ts_monotonic: float = 0.0
    last_cluster_exhaustion_emit_ts_monotonic: float = 0.0

    last_update_ts_monotonic: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_update_ts_monotonic = time.monotonic()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity_records_size": len(self.activity_records),
            "pressure_records_size": len(self.pressure_records),
            "liquidation_context_records_size": len(self.liquidation_context_records),
            "total_events_seen": self.total_events_seen,
            "total_clusters_emitted": self.total_clusters_emitted,
            "total_cluster_updates_emitted": self.total_cluster_updates_emitted,
            "total_cluster_exhaustions_emitted": self.total_cluster_exhaustions_emitted,
            "cluster_first_seen_ts_ms": self.cluster_first_seen_ts_ms,
            "cluster_last_seen_ts_ms": self.cluster_last_seen_ts_ms,
            "last_update_ts_monotonic": self.last_update_ts_monotonic,
        }


# =============================================================================
# Aggregate result models
# =============================================================================


@dataclass(slots=True)
class WhaleTrackerResult:
    whale_activity_signal: Optional[WhaleActivitySignal] = None
    whale_pressure_signal: Optional[WhalePressureSignal] = None
    whale_liquidation_context_signal: Optional[WhaleLiquidationContextSignal] = None

    def to_dict(self) -> Dict[str, Optional[Dict[str, Any]]]:
        return {
            "whale_activity_signal": (
                self.whale_activity_signal.to_event()
                if self.whale_activity_signal is not None
                else None
            ),
            "whale_pressure_signal": (
                self.whale_pressure_signal.to_event()
                if self.whale_pressure_signal is not None
                else None
            ),
            "whale_liquidation_context_signal": (
                self.whale_liquidation_context_signal.to_event()
                if self.whale_liquidation_context_signal is not None
                else None
            ),
        }


@dataclass(slots=True)
class WhaleClusterAnalysisResult:
    whale_cluster_signal: Optional[WhaleClusterSignal] = None
    whale_cluster_update_signal: Optional[WhaleClusterUpdateSignal] = None
    whale_cluster_exhaustion_signal: Optional[WhaleClusterExhaustionSignal] = None

    def to_dict(self) -> Dict[str, Optional[Dict[str, Any]]]:
        return {
            "whale_cluster_signal": (
                self.whale_cluster_signal.to_event()
                if self.whale_cluster_signal is not None
                else None
            ),
            "whale_cluster_update_signal": (
                self.whale_cluster_update_signal.to_event()
                if self.whale_cluster_update_signal is not None
                else None
            ),
            "whale_cluster_exhaustion_signal": (
                self.whale_cluster_exhaustion_signal.to_event()
                if self.whale_cluster_exhaustion_signal is not None
                else None
            ),
        }


# =============================================================================
# Factory helpers
# =============================================================================


def make_symbol_stats(window_size: int) -> SymbolStats:
    return SymbolStats(notionals=deque(maxlen=window_size))


def make_symbol_tracker_state(
    large_trade_window_size: int,
    liquidation_window_size: int,
) -> SymbolTrackerState:
    return SymbolTrackerState(
        large_trades=deque(maxlen=large_trade_window_size),
        liquidations=deque(maxlen=liquidation_window_size),
    )


def make_symbol_cluster_state(
    activity_window_size: int,
    pressure_window_size: int,
    liquidation_context_window_size: int,
) -> SymbolClusterState:
    return SymbolClusterState(
        activity_records=deque(maxlen=activity_window_size),
        pressure_records=deque(maxlen=pressure_window_size),
        liquidation_context_records=deque(maxlen=liquidation_context_window_size),
    )