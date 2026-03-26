from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from core.logger import get_logger


@dataclass(slots=True)
class WhaleClusterAnalyzerConfig:
    """
    Конфігурація аналізатора whale cluster-ів.
    """

    enabled: bool = True

    # Вхідні події
    whale_activity_event_name: str = "analytics.whales.whale_activity"
    whale_pressure_event_name: str = "analytics.whales.whale_pressure"
    whale_liquidation_context_event_name: str = "analytics.whales.whale_liquidation_context"

    # Вихідні події
    whale_cluster_event_name: str = "analytics.whales.whale_cluster"
    whale_cluster_update_event_name: str = "analytics.whales.whale_cluster_update"
    whale_cluster_exhaustion_event_name: str = "analytics.whales.whale_cluster_exhaustion"

    # Вікна аналізу
    analysis_window_sec: int = 180
    cluster_ttl_sec: int = 300

    # Мінімальні умови формування cluster
    min_activity_signals: int = 2
    min_total_activity_notional: float = 500_000.0

    # Скоринг
    activity_weight: float = 0.35
    pressure_weight: float = 0.35
    liquidation_context_weight: float = 0.20
    persistence_weight: float = 0.10

    # Cluster thresholds
    min_cluster_score_to_emit: float = 0.55
    min_continuation_probability_to_emit: float = 0.60
    min_exhaustion_probability_to_emit: float = 0.65

    # Cooldowns
    cluster_emit_cooldown_sec: float = 5.0
    cluster_update_cooldown_sec: float = 5.0
    cluster_exhaustion_cooldown_sec: float = 5.0

    # Cleanup
    cleanup_interval_sec: int = 60
    stats_ttl_sec: int = 60 * 60

    # Поведінка
    emit_on_bus: bool = True
    log_signals: bool = True


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


@dataclass(slots=True)
class WhaleClusterSignal:
    symbol: str
    cluster_side: str
    cluster_score: float
    persistence_score: float
    directional_bias: float
    continuation_probability: float
    exhaustion_probability: float

    activity_signal_count: int
    pressure_signal_count: int
    liquidation_context_count: int

    total_activity_notional: float
    total_pressure_notional: float
    total_liquidation_context_notional: float

    first_seen_ts_ms: int
    last_seen_ts_ms: int
    timestamp_ms: int

    detector_name: str = "WhaleClusterAnalyzer"
    event_type: str = "whale_cluster"
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_event(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "detector": self.detector_name,
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
            "created_at_ms": self.created_at_ms,
        }


@dataclass(slots=True)
class WhaleClusterUpdateSignal:
    symbol: str
    cluster_side: str
    cluster_score: float
    persistence_score: float
    continuation_probability: float
    exhaustion_probability: float
    activity_signal_count: int
    pressure_signal_count: int
    liquidation_context_count: int
    timestamp_ms: int

    detector_name: str = "WhaleClusterAnalyzer"
    event_type: str = "whale_cluster_update"
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_event(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "detector": self.detector_name,
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
            "created_at_ms": self.created_at_ms,
        }


@dataclass(slots=True)
class WhaleClusterExhaustionSignal:
    symbol: str
    cluster_side: str
    cluster_score: float
    exhaustion_probability: float
    reversal_risk: float
    timestamp_ms: int

    detector_name: str = "WhaleClusterAnalyzer"
    event_type: str = "whale_cluster_exhaustion"
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_event(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "detector": self.detector_name,
            "symbol": self.symbol,
            "cluster_side": self.cluster_side,
            "cluster_score": self.cluster_score,
            "exhaustion_probability": self.exhaustion_probability,
            "reversal_risk": self.reversal_risk,
            "timestamp_ms": self.timestamp_ms,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(slots=True)
class SymbolClusterState:
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


class WhaleClusterAnalyzer:
    """
    Третій шар whale-аналітики.

    Працює поверх сигналів:
        - whale_activity
        - whale_pressure
        - whale_liquidation_context

    Формує:
        - whale_cluster
        - whale_cluster_update
        - whale_cluster_exhaustion
    """

    def __init__(
        self,
        config: Optional[WhaleClusterAnalyzerConfig] = None,
        event_bus: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        self.config = config or WhaleClusterAnalyzerConfig()
        self.event_bus = event_bus
        self.scheduler = scheduler

        self.logger = get_logger(
            __name__,
            service_name="analytics.whales.whale_cluster_analyzer",
        )

        self._states: Dict[str, SymbolClusterState] = {}
        self._started = False
        self._cleanup_task: Optional[asyncio.Task[Any]] = None
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            self.logger.warning("WhaleClusterAnalyzer already started")
            return

        if not self.config.enabled:
            self.logger.info("WhaleClusterAnalyzer is disabled by config")
            return

        self._started = True

        if self.event_bus is not None:
            await self._safe_subscribe()

        if self.scheduler is not None:
            await self._register_scheduler_jobs()
        else:
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(),
                name="whale_cluster_analyzer_cleanup_loop",
            )

        self.logger.info(
            "WhaleClusterAnalyzer started",
            extra={
                "analysis_window_sec": self.config.analysis_window_sec,
                "cluster_ttl_sec": self.config.cluster_ttl_sec,
                "min_cluster_score_to_emit": self.config.min_cluster_score_to_emit,
            },
        )

    async def stop(self) -> None:
        if not self._started:
            return

        if self.event_bus is not None:
            await self._safe_unsubscribe()

        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

        self._started = False
        self.logger.info("WhaleClusterAnalyzer stopped")

    async def _safe_subscribe(self) -> None:
        try:
            await self.event_bus.subscribe(
                self.config.whale_activity_event_name,
                self.handle_whale_activity_event,
            )
            await self.event_bus.subscribe(
                self.config.whale_pressure_event_name,
                self.handle_whale_pressure_event,
            )
            await self.event_bus.subscribe(
                self.config.whale_liquidation_context_event_name,
                self.handle_whale_liquidation_context_event,
            )

            self.logger.info(
                "Subscribed WhaleClusterAnalyzer to EventBus",
                extra={
                    "activity_event": self.config.whale_activity_event_name,
                    "pressure_event": self.config.whale_pressure_event_name,
                    "liquidation_context_event": self.config.whale_liquidation_context_event_name,
                },
            )
        except Exception:
            self.logger.exception("Failed to subscribe WhaleClusterAnalyzer to EventBus")
            raise

    async def _safe_unsubscribe(self) -> None:
        for event_name, handler in (
            (self.config.whale_activity_event_name, self.handle_whale_activity_event),
            (self.config.whale_pressure_event_name, self.handle_whale_pressure_event),
            (self.config.whale_liquidation_context_event_name, self.handle_whale_liquidation_context_event),
        ):
            try:
                await self.event_bus.unsubscribe(event_name, handler)
            except Exception:
                self.logger.exception(
                    "Failed to unsubscribe WhaleClusterAnalyzer from EventBus",
                    extra={"event_name": event_name},
                )

    async def _register_scheduler_jobs(self) -> None:
        try:
            await self.scheduler.add_interval_job(
                name="whale_cluster_analyzer_cleanup",
                interval_seconds=self.config.cleanup_interval_sec,
                coro=self.cleanup,
                replace_existing=True,
            )
            self.logger.info(
                "Cleanup job registered in Scheduler",
                extra={"interval_sec": self.config.cleanup_interval_sec},
            )
        except Exception:
            self.logger.exception("Failed to register WhaleClusterAnalyzer cleanup job")
            raise

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    async def handle_whale_activity_event(self, event: Dict[str, Any]) -> None:
        try:
            await self.process_whale_activity_event(event)
        except Exception:
            self.logger.exception("Unhandled error while processing whale activity event")

    async def handle_whale_pressure_event(self, event: Dict[str, Any]) -> None:
        try:
            await self.process_whale_pressure_event(event)
        except Exception:
            self.logger.exception("Unhandled error while processing whale pressure event")

    async def handle_whale_liquidation_context_event(self, event: Dict[str, Any]) -> None:
        try:
            await self.process_whale_liquidation_context_event(event)
        except Exception:
            self.logger.exception("Unhandled error while processing whale liquidation context event")

    # -------------------------------------------------------------------------
    # Public processing methods
    # -------------------------------------------------------------------------

    async def process_whale_activity_event(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Optional[object]]:
        record = self._normalize_whale_activity_event(event)
        if record is None:
            return self._empty_result()

        async with self._lock:
            state = self._get_or_create_state(record.symbol)
            state.activity_records.append(record)
            state.total_events_seen += 1
            state.touch()

            self._update_cluster_seen_range(state, record.timestamp_ms)
            self._prune_symbol_state(state, record.timestamp_ms)

            result = self._analyze_symbol(record.symbol, state, record.timestamp_ms)

        await self._emit_analysis_result(result)
        return result

    async def process_whale_pressure_event(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Optional[object]]:
        record = self._normalize_whale_pressure_event(event)
        if record is None:
            return self._empty_result()

        async with self._lock:
            state = self._get_or_create_state(record.symbol)
            state.pressure_records.append(record)
            state.total_events_seen += 1
            state.touch()

            self._update_cluster_seen_range(state, record.timestamp_ms)
            self._prune_symbol_state(state, record.timestamp_ms)

            result = self._analyze_symbol(record.symbol, state, record.timestamp_ms)

        await self._emit_analysis_result(result)
        return result

    async def process_whale_liquidation_context_event(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Optional[object]]:
        record = self._normalize_whale_liquidation_context_event(event)
        if record is None:
            return self._empty_result()

        async with self._lock:
            state = self._get_or_create_state(record.symbol)
            state.liquidation_context_records.append(record)
            state.total_events_seen += 1
            state.touch()

            self._update_cluster_seen_range(state, record.timestamp_ms)
            self._prune_symbol_state(state, record.timestamp_ms)

            result = self._analyze_symbol(record.symbol, state, record.timestamp_ms)

        await self._emit_analysis_result(result)
        return result

    # -------------------------------------------------------------------------
    # Core analysis
    # -------------------------------------------------------------------------

    def _analyze_symbol(
        self,
        symbol: str,
        state: SymbolClusterState,
        current_ts_ms: int,
    ) -> Dict[str, Optional[object]]:
        activity_count = len(state.activity_records)
        total_activity_notional = sum(r.total_notional for r in state.activity_records)

        if activity_count < self.config.min_activity_signals:
            return self._empty_result()

        if total_activity_notional < self.config.min_total_activity_notional:
            return self._empty_result()

        cluster_side = self._determine_cluster_side(state)
        directional_bias = self._calculate_directional_bias(state, cluster_side)
        persistence_score = self._calculate_persistence_score(state, current_ts_ms)
        activity_score = self._calculate_activity_score(state)
        pressure_score = self._calculate_pressure_score(state, cluster_side)
        liquidation_context_score = self._calculate_liquidation_context_score(state, cluster_side)

        cluster_score = self._clamp_0_1(
            activity_score * self.config.activity_weight
            + pressure_score * self.config.pressure_weight
            + liquidation_context_score * self.config.liquidation_context_weight
            + persistence_score * self.config.persistence_weight
        )

        continuation_probability = self._clamp_0_1(
            0.45 * cluster_score
            + 0.35 * directional_bias
            + 0.20 * persistence_score
        )

        exhaustion_probability = self._clamp_0_1(
            0.50 * (1.0 - directional_bias)
            + 0.30 * liquidation_context_score
            + 0.20 * (1.0 - pressure_score)
        )

        cluster_signal: Optional[WhaleClusterSignal] = None
        cluster_update_signal: Optional[WhaleClusterUpdateSignal] = None
        cluster_exhaustion_signal: Optional[WhaleClusterExhaustionSignal] = None

        if (
            cluster_score >= self.config.min_cluster_score_to_emit
            and continuation_probability >= self.config.min_continuation_probability_to_emit
            and self._passes_cooldown(
                state.last_cluster_emit_ts_monotonic,
                self.config.cluster_emit_cooldown_sec,
            )
        ):
            cluster_signal = WhaleClusterSignal(
                symbol=symbol,
                cluster_side=cluster_side,
                cluster_score=cluster_score,
                persistence_score=persistence_score,
                directional_bias=directional_bias,
                continuation_probability=continuation_probability,
                exhaustion_probability=exhaustion_probability,
                activity_signal_count=len(state.activity_records),
                pressure_signal_count=len(state.pressure_records),
                liquidation_context_count=len(state.liquidation_context_records),
                total_activity_notional=total_activity_notional,
                total_pressure_notional=sum(r.total_notional for r in state.pressure_records),
                total_liquidation_context_notional=sum(
                    r.liquidation_total_notional for r in state.liquidation_context_records
                ),
                first_seen_ts_ms=state.cluster_first_seen_ts_ms or current_ts_ms,
                last_seen_ts_ms=state.cluster_last_seen_ts_ms or current_ts_ms,
                timestamp_ms=current_ts_ms,
            )
            state.total_clusters_emitted += 1
            state.last_cluster_emit_ts_monotonic = time.monotonic()

        if (
            cluster_score >= self.config.min_cluster_score_to_emit
            and self._passes_cooldown(
                state.last_cluster_update_emit_ts_monotonic,
                self.config.cluster_update_cooldown_sec,
            )
        ):
            cluster_update_signal = WhaleClusterUpdateSignal(
                symbol=symbol,
                cluster_side=cluster_side,
                cluster_score=cluster_score,
                persistence_score=persistence_score,
                continuation_probability=continuation_probability,
                exhaustion_probability=exhaustion_probability,
                activity_signal_count=len(state.activity_records),
                pressure_signal_count=len(state.pressure_records),
                liquidation_context_count=len(state.liquidation_context_records),
                timestamp_ms=current_ts_ms,
            )
            state.total_cluster_updates_emitted += 1
            state.last_cluster_update_emit_ts_monotonic = time.monotonic()

        if (
            exhaustion_probability >= self.config.min_exhaustion_probability_to_emit
            and self._passes_cooldown(
                state.last_cluster_exhaustion_emit_ts_monotonic,
                self.config.cluster_exhaustion_cooldown_sec,
            )
        ):
            cluster_exhaustion_signal = WhaleClusterExhaustionSignal(
                symbol=symbol,
                cluster_side=cluster_side,
                cluster_score=cluster_score,
                exhaustion_probability=exhaustion_probability,
                reversal_risk=exhaustion_probability,
                timestamp_ms=current_ts_ms,
            )
            state.total_cluster_exhaustions_emitted += 1
            state.last_cluster_exhaustion_emit_ts_monotonic = time.monotonic()

        return {
            "whale_cluster_signal": cluster_signal,
            "whale_cluster_update_signal": cluster_update_signal,
            "whale_cluster_exhaustion_signal": cluster_exhaustion_signal,
        }

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def _determine_cluster_side(self, state: SymbolClusterState) -> str:
        buy_activity = sum(r.total_notional for r in state.activity_records if r.side == "buy")
        sell_activity = sum(r.total_notional for r in state.activity_records if r.side == "sell")

        buy_pressure = sum(
            r.buy_notional for r in state.pressure_records if r.dominant_side == "buy"
        ) + sum(
            r.total_notional * 0.5 for r in state.pressure_records if r.dominant_side != "buy"
        )
        sell_pressure = sum(
            r.sell_notional for r in state.pressure_records if r.dominant_side == "sell"
        ) + sum(
            r.total_notional * 0.5 for r in state.pressure_records if r.dominant_side != "sell"
        )

        buy_score = buy_activity + buy_pressure
        sell_score = sell_activity + sell_pressure

        return "buy" if buy_score >= sell_score else "sell"

    def _calculate_directional_bias(self, state: SymbolClusterState, cluster_side: str) -> float:
        total_activity = sum(r.total_notional for r in state.activity_records)
        total_pressure = sum(r.total_notional for r in state.pressure_records)

        if total_activity <= 0 and total_pressure <= 0:
            return 0.0

        activity_same_side = sum(
            r.total_notional for r in state.activity_records if r.side == cluster_side
        )
        pressure_same_side = sum(
            r.total_notional for r in state.pressure_records if r.dominant_side == cluster_side
        )

        score = (activity_same_side + pressure_same_side) / max(total_activity + total_pressure, 1.0)
        return self._clamp_0_1(score)

    def _calculate_persistence_score(self, state: SymbolClusterState, current_ts_ms: int) -> float:
        first_seen = state.cluster_first_seen_ts_ms
        last_seen = state.cluster_last_seen_ts_ms

        if first_seen is None or last_seen is None:
            return 0.0

        active_duration_sec = max((last_seen - first_seen) / 1000.0, 0.0)
        score = active_duration_sec / max(self.config.analysis_window_sec, 1)
        return self._clamp_0_1(score)

    def _calculate_activity_score(self, state: SymbolClusterState) -> float:
        if not state.activity_records:
            return 0.0

        count_score = min(
            len(state.activity_records) / max(self.config.min_activity_signals * 2, 1),
            1.0,
        )
        notional_score = min(
            sum(r.total_notional for r in state.activity_records)
            / max(self.config.min_total_activity_notional * 2, 1.0),
            1.0,
        )

        return self._clamp_0_1(0.45 * count_score + 0.55 * notional_score)

    def _calculate_pressure_score(self, state: SymbolClusterState, cluster_side: str) -> float:
        if not state.pressure_records:
            return 0.0

        aligned_pressures = [r for r in state.pressure_records if r.dominant_side == cluster_side]
        if not aligned_pressures:
            return 0.0

        avg_imbalance = sum(r.imbalance_ratio for r in aligned_pressures) / len(aligned_pressures)
        aligned_notional = sum(r.total_notional for r in aligned_pressures)
        total_pressure_notional = sum(r.total_notional for r in state.pressure_records)

        alignment_score = aligned_notional / max(total_pressure_notional, 1.0)
        return self._clamp_0_1(0.60 * avg_imbalance + 0.40 * alignment_score)

    def _calculate_liquidation_context_score(
        self,
        state: SymbolClusterState,
        cluster_side: str,
    ) -> float:
        if not state.liquidation_context_records:
            return 0.0

        supportive_contexts = [
            r for r in state.liquidation_context_records if r.whale_side == cluster_side
        ]
        if not supportive_contexts:
            return 0.0

        avg_strength = sum(r.context_strength for r in supportive_contexts) / len(supportive_contexts)
        notional_factor = min(
            sum(r.liquidation_total_notional for r in supportive_contexts)
            / max(self.config.min_total_activity_notional, 1.0),
            1.0,
        )

        return self._clamp_0_1(0.65 * avg_strength + 0.35 * notional_factor)

    # -------------------------------------------------------------------------
    # Emission
    # -------------------------------------------------------------------------

    async def _emit_analysis_result(self, result: Dict[str, Optional[object]]) -> None:
        cluster_signal = result.get("whale_cluster_signal")
        cluster_update_signal = result.get("whale_cluster_update_signal")
        cluster_exhaustion_signal = result.get("whale_cluster_exhaustion_signal")

        if cluster_signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Whale cluster detected",
                    extra={
                        "symbol": cluster_signal.symbol,
                        "cluster_side": cluster_signal.cluster_side,
                        "cluster_score": cluster_signal.cluster_score,
                        "continuation_probability": cluster_signal.continuation_probability,
                        "exhaustion_probability": cluster_signal.exhaustion_probability,
                    },
                )
            await self._emit_signal(
                self.config.whale_cluster_event_name,
                cluster_signal.to_event(),
            )

        if cluster_update_signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Whale cluster updated",
                    extra={
                        "symbol": cluster_update_signal.symbol,
                        "cluster_side": cluster_update_signal.cluster_side,
                        "cluster_score": cluster_update_signal.cluster_score,
                    },
                )
            await self._emit_signal(
                self.config.whale_cluster_update_event_name,
                cluster_update_signal.to_event(),
            )

        if cluster_exhaustion_signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Whale cluster exhaustion detected",
                    extra={
                        "symbol": cluster_exhaustion_signal.symbol,
                        "cluster_side": cluster_exhaustion_signal.cluster_side,
                        "exhaustion_probability": cluster_exhaustion_signal.exhaustion_probability,
                    },
                )
            await self._emit_signal(
                self.config.whale_cluster_exhaustion_event_name,
                cluster_exhaustion_signal.to_event(),
            )

    async def _emit_signal(self, event_name: str, payload: Dict[str, Any]) -> None:
        if not self.config.emit_on_bus or self.event_bus is None:
            return

        try:
            await self.event_bus.emit(event_name, payload)
        except Exception:
            self.logger.exception(
                "Failed to emit WhaleClusterAnalyzer signal",
                extra={"event_name": event_name},
            )

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------

    def _normalize_whale_activity_event(
        self,
        event: Dict[str, Any],
    ) -> Optional[WhaleActivityRecord]:
        try:
            payload = event.get("data", event)

            symbol = self._normalize_symbol(payload.get("symbol"))
            side = self._normalize_side(payload.get("side"))
            trade_count = self._safe_int(payload.get("trade_count"))
            total_notional = self._safe_float(payload.get("total_notional"))
            avg_notional = self._safe_float(payload.get("avg_notional"))
            max_notional = self._safe_float(payload.get("max_notional"))
            window_sec = self._safe_int(payload.get("window_sec"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if not symbol or not side or trade_count <= 0 or total_notional <= 0:
                return None

            return WhaleActivityRecord(
                symbol=symbol,
                side=side,
                trade_count=trade_count,
                total_notional=total_notional,
                avg_notional=avg_notional,
                max_notional=max_notional,
                window_sec=window_sec,
                timestamp_ms=timestamp_ms,
                raw_event=event,
            )
        except Exception:
            self.logger.exception("Failed to normalize whale activity event")
            return None

    def _normalize_whale_pressure_event(
        self,
        event: Dict[str, Any],
    ) -> Optional[WhalePressureRecord]:
        try:
            payload = event.get("data", event)

            symbol = self._normalize_symbol(payload.get("symbol"))
            dominant_side = self._normalize_side(payload.get("dominant_side"))
            buy_trade_count = self._safe_int(payload.get("buy_trade_count"))
            sell_trade_count = self._safe_int(payload.get("sell_trade_count"))
            buy_notional = self._safe_float(payload.get("buy_notional"))
            sell_notional = self._safe_float(payload.get("sell_notional"))
            total_notional = self._safe_float(payload.get("total_notional"))
            imbalance_ratio = self._safe_float(payload.get("imbalance_ratio"))
            net_flow_notional = self._safe_float(payload.get("net_flow_notional"))
            window_sec = self._safe_int(payload.get("window_sec"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if not symbol or not dominant_side or total_notional <= 0:
                return None

            return WhalePressureRecord(
                symbol=symbol,
                dominant_side=dominant_side,
                buy_trade_count=buy_trade_count,
                sell_trade_count=sell_trade_count,
                buy_notional=buy_notional,
                sell_notional=sell_notional,
                total_notional=total_notional,
                imbalance_ratio=imbalance_ratio,
                net_flow_notional=net_flow_notional,
                window_sec=window_sec,
                timestamp_ms=timestamp_ms,
                raw_event=event,
            )
        except Exception:
            self.logger.exception("Failed to normalize whale pressure event")
            return None

    def _normalize_whale_liquidation_context_event(
        self,
        event: Dict[str, Any],
    ) -> Optional[WhaleLiquidationContextRecord]:
        try:
            payload = event.get("data", event)

            symbol = self._normalize_symbol(payload.get("symbol"))
            whale_side = self._normalize_side(payload.get("whale_side"))
            whale_total_notional = self._safe_float(payload.get("whale_total_notional"))
            whale_trade_count = self._safe_int(payload.get("whale_trade_count"))
            liquidation_side = self._normalize_side(payload.get("liquidation_side"))
            liquidation_total_notional = self._safe_float(payload.get("liquidation_total_notional"))
            liquidation_count = self._safe_int(payload.get("liquidation_count"))
            context_strength = self._safe_float(payload.get("context_strength"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if not symbol or not whale_side or not liquidation_side:
                return None

            return WhaleLiquidationContextRecord(
                symbol=symbol,
                whale_side=whale_side,
                whale_total_notional=whale_total_notional,
                whale_trade_count=whale_trade_count,
                liquidation_side=liquidation_side,
                liquidation_total_notional=liquidation_total_notional,
                liquidation_count=liquidation_count,
                context_strength=context_strength,
                timestamp_ms=timestamp_ms,
                raw_event=event,
            )
        except Exception:
            self.logger.exception("Failed to normalize whale liquidation context event")
            return None

    # -------------------------------------------------------------------------
    # State / housekeeping
    # -------------------------------------------------------------------------

    def _get_or_create_state(self, symbol: str) -> SymbolClusterState:
        state = self._states.get(symbol)
        if state is None:
            state = SymbolClusterState(
                activity_records=deque(),
                pressure_records=deque(),
                liquidation_context_records=deque(),
            )
            self._states[symbol] = state
        return state

    def _update_cluster_seen_range(self, state: SymbolClusterState, timestamp_ms: int) -> None:
        if state.cluster_first_seen_ts_ms is None:
            state.cluster_first_seen_ts_ms = timestamp_ms
        state.cluster_last_seen_ts_ms = timestamp_ms

    def _prune_symbol_state(self, state: SymbolClusterState, current_ts_ms: int) -> None:
        analysis_from_ms = current_ts_ms - self.config.analysis_window_sec * 1000

        while state.activity_records and state.activity_records[0].timestamp_ms < analysis_from_ms:
            state.activity_records.popleft()

        while state.pressure_records and state.pressure_records[0].timestamp_ms < analysis_from_ms:
            state.pressure_records.popleft()

        while (
            state.liquidation_context_records
            and state.liquidation_context_records[0].timestamp_ms < analysis_from_ms
        ):
            state.liquidation_context_records.popleft()

        if not state.activity_records and not state.pressure_records and not state.liquidation_context_records:
            state.cluster_first_seen_ts_ms = None
            state.cluster_last_seen_ts_ms = None

    async def cleanup(self) -> None:
        ttl = self.config.stats_ttl_sec
        now = time.monotonic()

        removed_symbols: List[str] = []

        async with self._lock:
            for symbol, state in list(self._states.items()):
                if (now - state.last_update_ts_monotonic) > ttl:
                    del self._states[symbol]
                    removed_symbols.append(symbol)

        if removed_symbols:
            self.logger.info(
                "Cleaned stale WhaleClusterAnalyzer states",
                extra={
                    "removed_count": len(removed_symbols),
                    "symbols": removed_symbols,
                },
            )

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.config.cleanup_interval_sec)
                await self.cleanup()
        except asyncio.CancelledError:
            self.logger.debug("WhaleClusterAnalyzer cleanup loop cancelled")
            raise
        except Exception:
            self.logger.exception("Unexpected error in WhaleClusterAnalyzer cleanup loop")

    # -------------------------------------------------------------------------
    # Public stats
    # -------------------------------------------------------------------------

    def get_symbol_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        state = self._states.get(symbol)
        if state is None:
            return None

        return {
            "symbol": symbol,
            "activity_records": len(state.activity_records),
            "pressure_records": len(state.pressure_records),
            "liquidation_context_records": len(state.liquidation_context_records),
            "total_events_seen": state.total_events_seen,
            "total_clusters_emitted": state.total_clusters_emitted,
            "total_cluster_updates_emitted": state.total_cluster_updates_emitted,
            "total_cluster_exhaustions_emitted": state.total_cluster_exhaustions_emitted,
            "cluster_first_seen_ts_ms": state.cluster_first_seen_ts_ms,
            "cluster_last_seen_ts_ms": state.cluster_last_seen_ts_ms,
        }

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        return {
            symbol: self.get_symbol_state(symbol) or {}
            for symbol in self._states
        }

    async def reset_symbol(self, symbol: str) -> None:
        async with self._lock:
            self._states.pop(symbol, None)

        self.logger.info(
            "Reset symbol state in WhaleClusterAnalyzer",
            extra={"symbol": symbol},
        )

    async def reset_all(self) -> None:
        async with self._lock:
            self._states.clear()

        self.logger.info("Reset all WhaleClusterAnalyzer states")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _passes_cooldown(self, last_ts: float, cooldown_sec: float) -> bool:
        now = time.monotonic()
        return (now - last_ts) >= cooldown_sec

    def _empty_result(self) -> Dict[str, Optional[object]]:
        return {
            "whale_cluster_signal": None,
            "whale_cluster_update_signal": None,
            "whale_cluster_exhaustion_signal": None,
        }

    def _clamp_0_1(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _normalize_symbol(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        result = str(value).strip().upper()
        return result or None

    def _normalize_side(self, value: Any) -> Optional[str]:
        if value is None:
            return None

        side = str(value).strip().lower()
        mapping = {
            "buy": "buy",
            "bid": "buy",
            "b": "buy",
            "sell": "sell",
            "ask": "sell",
            "s": "sell",
        }
        return mapping.get(side)

    def _safe_float(self, value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _safe_int(self, value: Any) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _extract_timestamp_ms(self, payload: Dict[str, Any]) -> int:
        raw_ts = (
            payload.get("timestamp_ms")
            or payload.get("ts")
            or payload.get("timestamp")
            or payload.get("T")
            or payload.get("time")
        )

        if raw_ts is None:
            return int(time.time() * 1000)

        if isinstance(raw_ts, datetime):
            if raw_ts.tzinfo is None:
                raw_ts = raw_ts.replace(tzinfo=timezone.utc)
            return int(raw_ts.timestamp() * 1000)

        if isinstance(raw_ts, str):
            try:
                dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except ValueError:
                pass

        try:
            ts = float(raw_ts)
        except (TypeError, ValueError):
            return int(time.time() * 1000)

        if ts < 10_000_000_000:
            ts *= 1000.0

        return int(ts)