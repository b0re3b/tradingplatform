from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from analytics.whales.enums import (
    LargeTradeTriggerType,
    WhaleBias,
    WhaleClusterStateType,
    WhaleEventType,
    WhalePressureType,
    WhaleTradeSide,
)


# =============================================================================
# Common helpers
# =============================================================================


def utc_now_ms() -> int:
    return int(time.time() * 1000)


def _safe_non_negative(value: float) -> float:
    return max(0.0, value)


def _clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, value))


# =============================================================================
# Base signal model
# =============================================================================


@dataclass(slots=True)
class WhaleBaseSignalModel:
    """
    Базова модель для всіх whale-сигналів.

    Важливо:
    - це не core.event_bus.Event;
    - to_payload() повертає dict payload для EventBus.emit(...);
    - runtime-компонент сам обгортає payload у EventBus.emit().
    """

    detector_name: str
    event_type: str
    schema_version: int = 1
    created_at_ms: int = field(default_factory=utc_now_ms)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "detector": self.detector_name,
            "created_at_ms": self.created_at_ms,
        }

    def to_event(self) -> dict[str, Any]:
        """
        Backward-compatible alias.

        Новий runtime-код має використовувати to_payload().
        """
        return self.to_payload()


# Backward-compatible alias for old imports.
WhaleBaseEventModel = WhaleBaseSignalModel


# =============================================================================
# Raw / normalized records
# =============================================================================


@dataclass(slots=True)
class TradeRecord:
    """
    Нормалізований raw trade record після прийому market.trade payload.
    """

    symbol: str
    price: float
    quantity: float
    side: str
    timestamp_ms: int
    trade_id: str | None = None
    exchange: str | None = None
    raw_event: dict[str, Any] | None = None

    @property
    def notional(self) -> float:
        return self.price * self.quantity

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side,
            "timestamp_ms": self.timestamp_ms,
            "trade_id": self.trade_id,
            "exchange": self.exchange,
            "notional": self.notional,
        }
        if include_raw:
            payload["raw_event"] = self.raw_event
        return payload


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
    trade_id: str | None = None
    exchange: str | None = None
    raw_event: dict[str, Any] | None = None

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "side": self.side,
            "notional": self.notional,
            "price": self.price,
            "quantity": self.quantity,
            "timestamp_ms": self.timestamp_ms,
            "zscore": self.zscore,
            "trigger_type": self.trigger_type,
            "trade_id": self.trade_id,
            "exchange": self.exchange,
        }
        if include_raw:
            payload["raw_event"] = self.raw_event
        return payload


@dataclass(slots=True)
class LiquidationRecord:
    """
    Нормалізований liquidation record після прийому market.liquidation payload.
    """

    symbol: str
    side: str
    notional: float
    price: float
    quantity: float
    timestamp_ms: int
    liquidation_id: str | None = None
    exchange: str | None = None
    raw_event: dict[str, Any] | None = None

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "side": self.side,
            "notional": self.notional,
            "price": self.price,
            "quantity": self.quantity,
            "timestamp_ms": self.timestamp_ms,
            "liquidation_id": self.liquidation_id,
            "exchange": self.exchange,
        }
        if include_raw:
            payload["raw_event"] = self.raw_event
        return payload


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
    raw_event: dict[str, Any] | None = None

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "side": self.side,
            "trade_count": self.trade_count,
            "total_notional": self.total_notional,
            "avg_notional": self.avg_notional,
            "max_notional": self.max_notional,
            "window_sec": self.window_sec,
            "timestamp_ms": self.timestamp_ms,
        }
        if include_raw:
            payload["raw_event"] = self.raw_event
        return payload


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
    raw_event: dict[str, Any] | None = None

    @property
    def pressure_type(self) -> str:
        return WhalePressureType.from_notional(
            buy_notional=self.buy_notional,
            sell_notional=self.sell_notional,
        ).value

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        if include_raw:
            payload["raw_event"] = self.raw_event
        return payload


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
    raw_event: dict[str, Any] | None = None

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        if include_raw:
            payload["raw_event"] = self.raw_event
        return payload


# =============================================================================
# Signal models
# =============================================================================


@dataclass(slots=True)
class LargeTradeSignal(WhaleBaseSignalModel):
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
    trade_id: str | None = None
    exchange: str | None = None

    detector_name: str = "LargeTradeDetector"
    event_type: str = WhaleEventType.LARGE_TRADE.value

    def to_payload(self) -> dict[str, Any]:
        payload = WhaleBaseSignalModel.to_payload(self)
        payload.update(
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
        return payload

    @classmethod
    def from_trade(
        cls,
        *,
        trade: TradeRecord,
        abs_threshold: float,
        mean_notional: float,
        std_notional: float,
        zscore: float,
        absolute_triggered: bool,
        relative_triggered: bool,
    ) -> LargeTradeSignal:
        return cls(
            symbol=trade.symbol,
            side=trade.side,
            price=trade.price,
            quantity=trade.quantity,
            notional=trade.notional,
            timestamp_ms=trade.timestamp_ms,
            abs_threshold=abs_threshold,
            mean_notional=mean_notional,
            std_notional=std_notional,
            zscore=zscore,
            trigger_type=LargeTradeTriggerType.from_flags(
                absolute_triggered=absolute_triggered,
                relative_triggered=relative_triggered,
            ).value,
            trade_id=trade.trade_id,
            exchange=trade.exchange,
        )


@dataclass(slots=True)
class WhaleActivitySignal(WhaleBaseSignalModel):
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

    def to_payload(self) -> dict[str, Any]:
        payload = WhaleBaseSignalModel.to_payload(self)
        payload.update(
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
        return payload


@dataclass(slots=True)
class WhalePressureSignal(WhaleBaseSignalModel):
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
        return WhalePressureType.from_notional(
            buy_notional=self.buy_notional,
            sell_notional=self.sell_notional,
        ).value

    def to_payload(self) -> dict[str, Any]:
        payload = WhaleBaseSignalModel.to_payload(self)
        payload.update(
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
        return payload


@dataclass(slots=True)
class WhaleLiquidationContextSignal(WhaleBaseSignalModel):
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

    def to_payload(self) -> dict[str, Any]:
        payload = WhaleBaseSignalModel.to_payload(self)
        payload.update(
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
        return payload


@dataclass(slots=True)
class WhaleClusterSignal(WhaleBaseSignalModel):
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
        return WhaleBias.from_side(self.cluster_side).value

    @property
    def state(self) -> str:
        return WhaleClusterStateType.from_scores(
            cluster_score=self.cluster_score,
            exhaustion_probability=self.exhaustion_probability,
        ).value

    def to_payload(self) -> dict[str, Any]:
        payload = WhaleBaseSignalModel.to_payload(self)
        payload.update(
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
        return payload


@dataclass(slots=True)
class WhaleClusterUpdateSignal(WhaleBaseSignalModel):
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

    def to_payload(self) -> dict[str, Any]:
        payload = WhaleBaseSignalModel.to_payload(self)
        payload.update(
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
        return payload


@dataclass(slots=True)
class WhaleClusterExhaustionSignal(WhaleBaseSignalModel):
    symbol: str = ""
    cluster_side: str = WhaleTradeSide.UNKNOWN.value
    cluster_score: float = 0.0
    exhaustion_probability: float = 0.0
    reversal_risk: float = 0.0
    timestamp_ms: int = 0

    detector_name: str = "WhaleClusterAnalyzer"
    event_type: str = WhaleEventType.WHALE_CLUSTER_EXHAUSTION.value

    def to_payload(self) -> dict[str, Any]:
        payload = WhaleBaseSignalModel.to_payload(self)
        payload.update(
            {
                "symbol": self.symbol,
                "cluster_side": self.cluster_side,
                "cluster_score": self.cluster_score,
                "exhaustion_probability": self.exhaustion_probability,
                "reversal_risk": self.reversal_risk,
                "timestamp_ms": self.timestamp_ms,
            }
        )
        return payload


# =============================================================================
# Internal rolling / symbol states
# =============================================================================


@dataclass(slots=True)
class SymbolStats:
    """
    Rolling-статистика для LargeTradeDetector.

    Це внутрішній state, не EventBus payload.
    """

    notionals: deque[float]
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
        value = _safe_non_negative(value)

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
        self.running_sum_sq = math.fsum(value * value for value in values)
        self.updates_since_recalibration = 0

    def mean(self) -> float:
        sample_size = len(self.notionals)
        if sample_size == 0:
            return 0.0
        return self.running_sum / sample_size

    def std(self) -> float:
        sample_size = len(self.notionals)
        if sample_size < 2:
            return 0.0

        mean_value = self.running_sum / sample_size
        numerator = self.running_sum_sq - sample_size * mean_value * mean_value
        numerator = max(numerator, 0.0)

        variance = numerator / (sample_size - 1)
        return math.sqrt(max(variance, 0.0))

    def to_dict(self) -> dict[str, Any]:
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

    Це внутрішній state, не EventBus payload.
    """

    large_trades: deque[WhaleTradeRecord]
    liquidations: deque[LiquidationRecord]

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "large_trades_buffer_size": len(self.large_trades),
            "liquidations_buffer_size": len(self.liquidations),
            "total_large_trades_seen": self.total_large_trades_seen,
            "total_liquidations_seen": self.total_liquidations_seen,
            "whale_activity_signals_emitted": self.whale_activity_signals_emitted,
            "whale_pressure_signals_emitted": self.whale_pressure_signals_emitted,
            "whale_liquidation_context_signals_emitted": (
                self.whale_liquidation_context_signals_emitted
            ),
            "last_update_ts_monotonic": self.last_update_ts_monotonic,
        }


@dataclass(slots=True)
class SymbolClusterState:
    """
    Rolling-state для WhaleClusterAnalyzer.

    Це внутрішній state, не EventBus payload.
    """

    activity_records: deque[WhaleActivityRecord]
    pressure_records: deque[WhalePressureRecord]
    liquidation_context_records: deque[WhaleLiquidationContextRecord]

    total_events_seen: int = 0
    total_clusters_emitted: int = 0
    total_cluster_updates_emitted: int = 0
    total_cluster_exhaustions_emitted: int = 0

    cluster_first_seen_ts_ms: int | None = None
    cluster_last_seen_ts_ms: int | None = None

    last_cluster_emit_ts_monotonic: float = 0.0
    last_cluster_update_emit_ts_monotonic: float = 0.0
    last_cluster_exhaustion_emit_ts_monotonic: float = 0.0

    last_update_ts_monotonic: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_update_ts_monotonic = time.monotonic()

    def to_dict(self) -> dict[str, Any]:
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
    whale_activity_signal: WhaleActivitySignal | None = None
    whale_pressure_signal: WhalePressureSignal | None = None
    whale_liquidation_context_signal: WhaleLiquidationContextSignal | None = None

    @property
    def has_signals(self) -> bool:
        return any(
            signal is not None
            for signal in (
                self.whale_activity_signal,
                self.whale_pressure_signal,
                self.whale_liquidation_context_signal,
            )
        )

    def iter_signals(self) -> tuple[WhaleBaseSignalModel, ...]:
        return tuple(
            signal
            for signal in (
                self.whale_activity_signal,
                self.whale_pressure_signal,
                self.whale_liquidation_context_signal,
            )
            if signal is not None
        )

    def to_dict(self) -> dict[str, dict[str, Any] | None]:
        return {
            "whale_activity_signal": (
                self.whale_activity_signal.to_payload()
                if self.whale_activity_signal is not None
                else None
            ),
            "whale_pressure_signal": (
                self.whale_pressure_signal.to_payload()
                if self.whale_pressure_signal is not None
                else None
            ),
            "whale_liquidation_context_signal": (
                self.whale_liquidation_context_signal.to_payload()
                if self.whale_liquidation_context_signal is not None
                else None
            ),
        }


@dataclass(slots=True)
class WhaleClusterAnalysisResult:
    whale_cluster_signal: WhaleClusterSignal | None = None
    whale_cluster_update_signal: WhaleClusterUpdateSignal | None = None
    whale_cluster_exhaustion_signal: WhaleClusterExhaustionSignal | None = None

    @property
    def has_signals(self) -> bool:
        return any(
            signal is not None
            for signal in (
                self.whale_cluster_signal,
                self.whale_cluster_update_signal,
                self.whale_cluster_exhaustion_signal,
            )
        )

    def iter_signals(self) -> tuple[WhaleBaseSignalModel, ...]:
        return tuple(
            signal
            for signal in (
                self.whale_cluster_signal,
                self.whale_cluster_update_signal,
                self.whale_cluster_exhaustion_signal,
            )
            if signal is not None
        )

    def to_dict(self) -> dict[str, dict[str, Any] | None]:
        return {
            "whale_cluster_signal": (
                self.whale_cluster_signal.to_payload()
                if self.whale_cluster_signal is not None
                else None
            ),
            "whale_cluster_update_signal": (
                self.whale_cluster_update_signal.to_payload()
                if self.whale_cluster_update_signal is not None
                else None
            ),
            "whale_cluster_exhaustion_signal": (
                self.whale_cluster_exhaustion_signal.to_payload()
                if self.whale_cluster_exhaustion_signal is not None
                else None
            ),
        }


# =============================================================================
# Factory helpers
# =============================================================================


def make_symbol_stats(window_size: int) -> SymbolStats:
    if window_size <= 1:
        raise ValueError("window_size must be > 1")
    return SymbolStats(notionals=deque(maxlen=window_size))


def make_symbol_tracker_state(
    large_trade_window_size: int,
    liquidation_window_size: int,
) -> SymbolTrackerState:
    if large_trade_window_size <= 0:
        raise ValueError("large_trade_window_size must be > 0")
    if liquidation_window_size <= 0:
        raise ValueError("liquidation_window_size must be > 0")

    return SymbolTrackerState(
        large_trades=deque(maxlen=large_trade_window_size),
        liquidations=deque(maxlen=liquidation_window_size),
    )


def make_symbol_cluster_state(
    activity_window_size: int,
    pressure_window_size: int,
    liquidation_context_window_size: int,
) -> SymbolClusterState:
    if activity_window_size <= 0:
        raise ValueError("activity_window_size must be > 0")
    if pressure_window_size <= 0:
        raise ValueError("pressure_window_size must be > 0")
    if liquidation_context_window_size <= 0:
        raise ValueError("liquidation_context_window_size must be > 0")

    return SymbolClusterState(
        activity_records=deque(maxlen=activity_window_size),
        pressure_records=deque(maxlen=pressure_window_size),
        liquidation_context_records=deque(maxlen=liquidation_context_window_size),
    )


__all__ = [
    # base
    "WhaleBaseSignalModel",
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

    # results
    "WhaleTrackerResult",
    "WhaleClusterAnalysisResult",

    # factories
    "make_symbol_stats",
    "make_symbol_tracker_state",
    "make_symbol_cluster_state",

    # helpers
    "utc_now_ms",
]