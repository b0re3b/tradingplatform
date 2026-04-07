from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from analytics.whales.base import BaseWhaleComponent
from analytics.whales.config import WhaleClusterAnalyzerConfig
from analytics.whales.enums import WhaleTradeSide
from analytics.whales.models import (
    SymbolClusterState,
    WhaleActivityRecord,
    WhaleClusterAnalysisResult,
    WhaleClusterExhaustionSignal,
    WhaleClusterSignal,
    WhaleClusterUpdateSignal,
    WhaleLiquidationContextRecord,
    WhalePressureRecord,
    make_symbol_cluster_state,
)


class WhaleClusterAnalyzer(BaseWhaleComponent):
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
        super().__init__(
            component_name="whale_cluster_analyzer",
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.config = config or WhaleClusterAnalyzerConfig()

        self._states: Dict[str, SymbolClusterState] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task[Any]] = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        if self._started:
            self.logger.warning("WhaleClusterAnalyzer already started")
            return

        if not self.config.enabled:
            self.logger.info("WhaleClusterAnalyzer is disabled by config")
            return

        if self.event_bus is not None:
            await self._safe_subscribe(
                self.config.whale_activity_event_name,
                self.handle_whale_activity_event,
            )
            await self._safe_subscribe(
                self.config.whale_pressure_event_name,
                self.handle_whale_pressure_event,
            )
            await self._safe_subscribe(
                self.config.whale_liquidation_context_event_name,
                self.handle_whale_liquidation_context_event,
            )

        if self.scheduler is not None:
            await self._register_interval_job(
                name="whales_whale_cluster_analyzer_cleanup",
                interval_seconds=self.config.cleanup_interval_sec,
                coro=self.cleanup,
                replace_existing=True,
            )
        else:
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(),
                name="whales_whale_cluster_analyzer_cleanup_loop",
            )

        self._started = True

        self.logger.info(
            "WhaleClusterAnalyzer started",
            extra={
                "activity_event_name": self.config.whale_activity_event_name,
                "pressure_event_name": self.config.whale_pressure_event_name,
                "liquidation_context_event_name": self.config.whale_liquidation_context_event_name,
                "analysis_window_sec": self.config.analysis_window_sec,
                "cluster_ttl_sec": self.config.cluster_ttl_sec,
                "min_cluster_score_to_emit": self.config.min_cluster_score_to_emit,
            },
        )

    async def stop(self) -> None:
        if not self._started:
            return

        if self.event_bus is not None:
            await self._safe_unsubscribe(
                self.config.whale_activity_event_name,
                self.handle_whale_activity_event,
            )
            await self._safe_unsubscribe(
                self.config.whale_pressure_event_name,
                self.handle_whale_pressure_event,
            )
            await self._safe_unsubscribe(
                self.config.whale_liquidation_context_event_name,
                self.handle_whale_liquidation_context_event,
            )

        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

        self._started = False
        self.logger.info("WhaleClusterAnalyzer stopped")

    # =========================================================================
    # Event handlers
    # =========================================================================

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

    # =========================================================================
    # Public processing API
    # =========================================================================

    async def process_whale_activity_event(
        self,
        event: Dict[str, Any],
    ) -> WhaleClusterAnalysisResult:
        if not self.config.enabled:
            return WhaleClusterAnalysisResult()

        record = self._normalize_whale_activity_event(event)
        if record is None:
            return WhaleClusterAnalysisResult()

        async with self._lock:
            state = self._get_or_create_state(record.symbol)
            state.activity_records.append(record)
            state.total_events_seen += 1
            state.touch()

            self._update_cluster_seen_range(state, record.timestamp_ms)
            self._prune_symbol_state(state, record.timestamp_ms)

            result = self._analyze_symbol(
                symbol=record.symbol,
                state=state,
                current_ts_ms=record.timestamp_ms,
            )

        await self._emit_analysis_result(result)
        return result

    async def process_whale_pressure_event(
        self,
        event: Dict[str, Any],
    ) -> WhaleClusterAnalysisResult:
        if not self.config.enabled:
            return WhaleClusterAnalysisResult()

        record = self._normalize_whale_pressure_event(event)
        if record is None:
            return WhaleClusterAnalysisResult()

        async with self._lock:
            state = self._get_or_create_state(record.symbol)
            state.pressure_records.append(record)
            state.total_events_seen += 1
            state.touch()

            self._update_cluster_seen_range(state, record.timestamp_ms)
            self._prune_symbol_state(state, record.timestamp_ms)

            result = self._analyze_symbol(
                symbol=record.symbol,
                state=state,
                current_ts_ms=record.timestamp_ms,
            )

        await self._emit_analysis_result(result)
        return result

    async def process_whale_liquidation_context_event(
        self,
        event: Dict[str, Any],
    ) -> WhaleClusterAnalysisResult:
        if not self.config.enabled:
            return WhaleClusterAnalysisResult()

        record = self._normalize_whale_liquidation_context_event(event)
        if record is None:
            return WhaleClusterAnalysisResult()

        async with self._lock:
            state = self._get_or_create_state(record.symbol)
            state.liquidation_context_records.append(record)
            state.total_events_seen += 1
            state.touch()

            self._update_cluster_seen_range(state, record.timestamp_ms)
            self._prune_symbol_state(state, record.timestamp_ms)

            result = self._analyze_symbol(
                symbol=record.symbol,
                state=state,
                current_ts_ms=record.timestamp_ms,
            )

        await self._emit_analysis_result(result)
        return result

    # =========================================================================
    # Core analysis
    # =========================================================================

    def _analyze_symbol(
        self,
        *,
        symbol: str,
        state: SymbolClusterState,
        current_ts_ms: int,
    ) -> WhaleClusterAnalysisResult:
        activity_count = len(state.activity_records)
        total_activity_notional = sum(record.total_notional for record in state.activity_records)

        if activity_count < self.config.min_activity_signals:
            return WhaleClusterAnalysisResult()

        if total_activity_notional < self.config.min_total_activity_notional:
            return WhaleClusterAnalysisResult()

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
                total_pressure_notional=sum(record.total_notional for record in state.pressure_records),
                total_liquidation_context_notional=sum(
                    record.liquidation_total_notional
                    for record in state.liquidation_context_records
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

        return WhaleClusterAnalysisResult(
            whale_cluster_signal=cluster_signal,
            whale_cluster_update_signal=cluster_update_signal,
            whale_cluster_exhaustion_signal=cluster_exhaustion_signal,
        )

    # =========================================================================
    # Scoring
    # =========================================================================

    def _determine_cluster_side(self, state: SymbolClusterState) -> str:
        buy_activity = sum(
            record.total_notional
            for record in state.activity_records
            if record.side == WhaleTradeSide.BUY.value
        )
        sell_activity = sum(
            record.total_notional
            for record in state.activity_records
            if record.side == WhaleTradeSide.SELL.value
        )

        buy_pressure = sum(
            record.buy_notional
            for record in state.pressure_records
            if record.dominant_side == WhaleTradeSide.BUY.value
        )
        sell_pressure = sum(
            record.sell_notional
            for record in state.pressure_records
            if record.dominant_side == WhaleTradeSide.SELL.value
        )

        buy_liq_context = sum(
            record.whale_total_notional
            for record in state.liquidation_context_records
            if record.whale_side == WhaleTradeSide.BUY.value
        )
        sell_liq_context = sum(
            record.whale_total_notional
            for record in state.liquidation_context_records
            if record.whale_side == WhaleTradeSide.SELL.value
        )

        buy_score = buy_activity + buy_pressure + buy_liq_context
        sell_score = sell_activity + sell_pressure + sell_liq_context

        if buy_score >= sell_score:
            return WhaleTradeSide.BUY.value
        return WhaleTradeSide.SELL.value

    def _calculate_directional_bias(
        self,
        state: SymbolClusterState,
        cluster_side: str,
    ) -> float:
        total_side_notional = 0.0
        total_other_notional = 0.0

        for record in state.activity_records:
            if record.side == cluster_side:
                total_side_notional += record.total_notional
            else:
                total_other_notional += record.total_notional

        for record in state.pressure_records:
            dominant_notional = max(record.buy_notional, record.sell_notional)
            non_dominant_notional = min(record.buy_notional, record.sell_notional)

            if record.dominant_side == cluster_side:
                total_side_notional += dominant_notional
                total_other_notional += non_dominant_notional
            else:
                total_side_notional += non_dominant_notional
                total_other_notional += dominant_notional

        for record in state.liquidation_context_records:
            if record.whale_side == cluster_side:
                total_side_notional += record.whale_total_notional
            else:
                total_other_notional += record.whale_total_notional

        total = total_side_notional + total_other_notional
        if total <= 0:
            return 0.0

        return self._clamp_0_1(total_side_notional / total)

    def _calculate_persistence_score(
        self,
        state: SymbolClusterState,
        current_ts_ms: int,
    ) -> float:
        first_seen = state.cluster_first_seen_ts_ms
        last_seen = state.cluster_last_seen_ts_ms

        if first_seen is None or last_seen is None:
            return 0.0

        duration_ms = max(0, last_seen - first_seen)
        duration_sec = duration_ms / 1000.0

        if duration_sec <= 0:
            return 0.0

        normalized = duration_sec / max(1.0, float(self.config.analysis_window_sec))
        freshness_penalty = 1.0

        lag_sec = max(0.0, (current_ts_ms - last_seen) / 1000.0)
        if lag_sec > 0:
            freshness_penalty = max(
                0.25,
                1.0 - lag_sec / max(1.0, float(self.config.analysis_window_sec)),
            )

        return self._clamp_0_1(normalized * freshness_penalty)

    def _calculate_activity_score(self, state: SymbolClusterState) -> float:
        if not state.activity_records:
            return 0.0

        signal_factor = len(state.activity_records) / max(1, self.config.min_activity_signals)
        signal_factor = min(1.0, signal_factor)

        total_notional = sum(record.total_notional for record in state.activity_records)
        notional_factor = total_notional / max(1.0, self.config.min_total_activity_notional)
        notional_factor = min(1.0, notional_factor)

        return self._clamp_0_1(0.45 * signal_factor + 0.55 * notional_factor)

    def _calculate_pressure_score(
        self,
        state: SymbolClusterState,
        cluster_side: str,
    ) -> float:
        if not state.pressure_records:
            return 0.0

        aligned_records = [
            record
            for record in state.pressure_records
            if record.dominant_side == cluster_side
        ]
        if not aligned_records:
            return 0.0

        avg_imbalance = sum(record.imbalance_ratio for record in aligned_records) / len(aligned_records)
        alignment_ratio = len(aligned_records) / len(state.pressure_records)

        return self._clamp_0_1(0.60 * avg_imbalance + 0.40 * alignment_ratio)

    def _calculate_liquidation_context_score(
        self,
        state: SymbolClusterState,
        cluster_side: str,
    ) -> float:
        if not state.liquidation_context_records:
            return 0.0

        aligned_records = [
            record
            for record in state.liquidation_context_records
            if record.whale_side == cluster_side
        ]
        if not aligned_records:
            return 0.0

        avg_context_strength = (
            sum(record.context_strength for record in aligned_records) / len(aligned_records)
        )
        alignment_ratio = len(aligned_records) / len(state.liquidation_context_records)

        return self._clamp_0_1(0.70 * avg_context_strength + 0.30 * alignment_ratio)

    # =========================================================================
    # Emission
    # =========================================================================

    async def _emit_analysis_result(self, result: WhaleClusterAnalysisResult) -> None:
        if not self.config.emit_on_bus:
            return

        if result.whale_cluster_signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Whale cluster detected",
                    extra={
                        "symbol": result.whale_cluster_signal.symbol,
                        "cluster_side": result.whale_cluster_signal.cluster_side,
                        "cluster_score": result.whale_cluster_signal.cluster_score,
                        "continuation_probability": result.whale_cluster_signal.continuation_probability,
                    },
                )

            await self._safe_emit(
                self.config.whale_cluster_event_name,
                result.whale_cluster_signal.to_event(),
                source="analytics.whales.whale_cluster_analyzer",
            )

        if result.whale_cluster_update_signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Whale cluster update emitted",
                    extra={
                        "symbol": result.whale_cluster_update_signal.symbol,
                        "cluster_side": result.whale_cluster_update_signal.cluster_side,
                        "cluster_score": result.whale_cluster_update_signal.cluster_score,
                    },
                )

            await self._safe_emit(
                self.config.whale_cluster_update_event_name,
                result.whale_cluster_update_signal.to_event(),
                source="analytics.whales.whale_cluster_analyzer",
            )

        if result.whale_cluster_exhaustion_signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Whale cluster exhaustion emitted",
                    extra={
                        "symbol": result.whale_cluster_exhaustion_signal.symbol,
                        "cluster_side": result.whale_cluster_exhaustion_signal.cluster_side,
                        "exhaustion_probability": result.whale_cluster_exhaustion_signal.exhaustion_probability,
                    },
                )

            await self._safe_emit(
                self.config.whale_cluster_exhaustion_event_name,
                result.whale_cluster_exhaustion_signal.to_event(),
                source="analytics.whales.whale_cluster_analyzer",
            )

    # =========================================================================
    # State management
    # =========================================================================

    def _get_or_create_state(self, symbol: str) -> SymbolClusterState:
        state = self._states.get(symbol)
        if state is not None:
            return state

        state = make_symbol_cluster_state(
            activity_window_size=self.config.activity_buffer_size,
            pressure_window_size=self.config.pressure_buffer_size,
            liquidation_context_window_size=self.config.liquidation_context_buffer_size,
        )
        self._states[symbol] = state
        return state

    def _update_cluster_seen_range(
        self,
        state: SymbolClusterState,
        timestamp_ms: int,
    ) -> None:
        if state.cluster_first_seen_ts_ms is None:
            state.cluster_first_seen_ts_ms = timestamp_ms
        else:
            state.cluster_first_seen_ts_ms = min(state.cluster_first_seen_ts_ms, timestamp_ms)

        if state.cluster_last_seen_ts_ms is None:
            state.cluster_last_seen_ts_ms = timestamp_ms
        else:
            state.cluster_last_seen_ts_ms = max(state.cluster_last_seen_ts_ms, timestamp_ms)

    def _prune_symbol_state(
        self,
        state: SymbolClusterState,
        current_ts_ms: int,
    ) -> None:
        cutoff_ms = current_ts_ms - self.config.analysis_window_sec * 1000

        while state.activity_records and state.activity_records[0].timestamp_ms < cutoff_ms:
            state.activity_records.popleft()

        while state.pressure_records and state.pressure_records[0].timestamp_ms < cutoff_ms:
            state.pressure_records.popleft()

        while (
            state.liquidation_context_records
            and state.liquidation_context_records[0].timestamp_ms < cutoff_ms
        ):
            state.liquidation_context_records.popleft()

        remaining_timestamps = []

        if state.activity_records:
            remaining_timestamps.append(state.activity_records[0].timestamp_ms)
        if state.pressure_records:
            remaining_timestamps.append(state.pressure_records[0].timestamp_ms)
        if state.liquidation_context_records:
            remaining_timestamps.append(state.liquidation_context_records[0].timestamp_ms)

        if remaining_timestamps:
            state.cluster_first_seen_ts_ms = min(remaining_timestamps)
            state.cluster_last_seen_ts_ms = max(
                [
                    *(record.timestamp_ms for record in state.activity_records),
                    *(record.timestamp_ms for record in state.pressure_records),
                    *(record.timestamp_ms for record in state.liquidation_context_records),
                ]
            )
        else:
            state.cluster_first_seen_ts_ms = None
            state.cluster_last_seen_ts_ms = None

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup(self) -> None:
        ttl = self.config.stats_ttl_sec
        if ttl <= 0:
            return

        now_mono = time.monotonic()

        async with self._lock:
            stale_symbols = [
                symbol
                for symbol, state in self._states.items()
                if (now_mono - state.last_update_ts_monotonic) >= ttl
            ]

            for symbol in stale_symbols:
                self._states.pop(symbol, None)

        if stale_symbols:
            self.logger.info(
                "Cleaned stale WhaleClusterAnalyzer symbol states",
                extra={
                    "removed_symbols_count": len(stale_symbols),
                    "symbols": stale_symbols,
                },
            )

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.config.cleanup_interval_sec)
                await self.cleanup()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("Unhandled error in WhaleClusterAnalyzer cleanup loop")

    # =========================================================================
    # Public state / stats API
    # =========================================================================

    def get_symbol_state(self, symbol: str) -> Dict[str, Any]:
        state = self._states.get(symbol)
        if state is None:
            return {
                "symbol": symbol,
                "exists": False,
            }

        return {
            "symbol": symbol,
            "exists": True,
            **state.to_dict(),
        }

    def get_all_states(self) -> Dict[str, Any]:
        return {
            symbol: state.to_dict()
            for symbol, state in self._states.items()
        }

    async def reset_symbol(self, symbol: str) -> None:
        async with self._lock:
            self._states.pop(symbol, None)

        self.logger.info(
            "Reset WhaleClusterAnalyzer symbol state",
            extra={"symbol": symbol},
        )

    async def reset_all(self) -> None:
        async with self._lock:
            self._states.clear()

        self.logger.info("Reset all WhaleClusterAnalyzer states")

    # =========================================================================
    # Normalization helpers
    # =========================================================================

    def _normalize_whale_activity_event(
        self,
        event: Dict[str, Any],
    ) -> Optional[WhaleActivityRecord]:
        try:
            payload = event.get("data", event)

            symbol = self._normalize_symbol(
                payload.get("symbol")
                or payload.get("s")
                or payload.get("instrument")
            )
            side = self._normalize_side(
                payload.get("side")
                or payload.get("S")
            )
            trade_count = self._safe_int(payload.get("trade_count"))
            total_notional = self._safe_float(payload.get("total_notional"))
            avg_notional = self._safe_float(payload.get("avg_notional"))
            max_notional = self._safe_float(payload.get("max_notional"))
            window_sec = self._safe_int(payload.get("window_sec"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if symbol is None or side == WhaleTradeSide.UNKNOWN.value:
                return None
            if trade_count is None or trade_count <= 0:
                return None
            if total_notional is None or total_notional <= 0:
                return None
            if avg_notional is None or avg_notional <= 0:
                return None
            if max_notional is None or max_notional <= 0:
                return None
            if window_sec is None or window_sec <= 0:
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
            self.logger.exception(
                "Failed to normalize whale activity event",
                extra={"event": event},
            )
            return None

    def _normalize_whale_pressure_event(
        self,
        event: Dict[str, Any],
    ) -> Optional[WhalePressureRecord]:
        try:
            payload = event.get("data", event)

            symbol = self._normalize_symbol(
                payload.get("symbol")
                or payload.get("s")
                or payload.get("instrument")
            )
            dominant_side = self._normalize_side(
                payload.get("dominant_side")
                or payload.get("side")
                or payload.get("S")
            )
            buy_trade_count = self._safe_int(payload.get("buy_trade_count"))
            sell_trade_count = self._safe_int(payload.get("sell_trade_count"))
            buy_notional = self._safe_float(payload.get("buy_notional"))
            sell_notional = self._safe_float(payload.get("sell_notional"))
            total_notional = self._safe_float(payload.get("total_notional"))
            imbalance_ratio = self._safe_float(payload.get("imbalance_ratio"))
            net_flow_notional = self._safe_float(payload.get("net_flow_notional"))
            window_sec = self._safe_int(payload.get("window_sec"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if symbol is None or dominant_side == WhaleTradeSide.UNKNOWN.value:
                return None
            if buy_trade_count is None or buy_trade_count < 0:
                return None
            if sell_trade_count is None or sell_trade_count < 0:
                return None
            if buy_notional is None or buy_notional < 0:
                return None
            if sell_notional is None or sell_notional < 0:
                return None
            if total_notional is None or total_notional <= 0:
                return None
            if imbalance_ratio is None or imbalance_ratio < 0:
                return None
            if net_flow_notional is None:
                return None
            if window_sec is None or window_sec <= 0:
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
            self.logger.exception(
                "Failed to normalize whale pressure event",
                extra={"event": event},
            )
            return None

    def _normalize_whale_liquidation_context_event(
        self,
        event: Dict[str, Any],
    ) -> Optional[WhaleLiquidationContextRecord]:
        try:
            payload = event.get("data", event)

            symbol = self._normalize_symbol(
                payload.get("symbol")
                or payload.get("s")
                or payload.get("instrument")
            )
            whale_side = self._normalize_side(payload.get("whale_side"))
            liquidation_side = self._normalize_side(payload.get("liquidation_side"))
            whale_total_notional = self._safe_float(payload.get("whale_total_notional"))
            whale_trade_count = self._safe_int(payload.get("whale_trade_count"))
            liquidation_total_notional = self._safe_float(payload.get("liquidation_total_notional"))
            liquidation_count = self._safe_int(payload.get("liquidation_count"))
            context_strength = self._safe_float(payload.get("context_strength"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if symbol is None:
                return None
            if whale_side == WhaleTradeSide.UNKNOWN.value:
                return None
            if liquidation_side == WhaleTradeSide.UNKNOWN.value:
                return None
            if whale_total_notional is None or whale_total_notional <= 0:
                return None
            if whale_trade_count is None or whale_trade_count <= 0:
                return None
            if liquidation_total_notional is None or liquidation_total_notional <= 0:
                return None
            if liquidation_count is None or liquidation_count <= 0:
                return None
            if context_strength is None or context_strength < 0:
                return None

            return WhaleLiquidationContextRecord(
                symbol=symbol,
                whale_side=whale_side,
                whale_total_notional=whale_total_notional,
                whale_trade_count=whale_trade_count,
                liquidation_side=liquidation_side,
                liquidation_total_notional=liquidation_total_notional,
                liquidation_count=liquidation_count,
                context_strength=self._clamp_0_1(context_strength),
                timestamp_ms=timestamp_ms,
                raw_event=event,
            )
        except Exception:
            self.logger.exception(
                "Failed to normalize whale liquidation context event",
                extra={"event": event},
            )
            return None

    def _normalize_symbol(self, value: Any) -> Optional[str]:
        text = self._safe_str(value)
        if text is None:
            return None
        return text.upper()

    def _normalize_side(self, value: Any) -> str:
        if isinstance(value, str):
            side = value.strip().lower()

            if side in {"buy", "bid", "long"}:
                return WhaleTradeSide.BUY.value
            if side in {"sell", "ask", "short"}:
                return WhaleTradeSide.SELL.value

        return WhaleTradeSide.UNKNOWN.value

    def _extract_timestamp_ms(self, payload: Dict[str, Any]) -> int:
        raw_ts = (
            payload.get("timestamp_ms")
            or payload.get("timestamp")
            or payload.get("ts")
            or payload.get("T")
            or payload.get("E")
        )

        if raw_ts is None:
            return int(time.time() * 1000)

        if isinstance(raw_ts, datetime):
            if raw_ts.tzinfo is None:
                raw_ts = raw_ts.replace(tzinfo=timezone.utc)
            return int(raw_ts.timestamp() * 1000)

        if isinstance(raw_ts, (int, float)):
            if raw_ts < 10_000_000_000:
                return int(raw_ts * 1000)
            return int(raw_ts)

        if isinstance(raw_ts, str):
            raw_ts = raw_ts.strip()

            try:
                dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except Exception:
                pass

            try:
                numeric = float(raw_ts)
                if numeric < 10_000_000_000:
                    return int(numeric * 1000)
                return int(numeric)
            except Exception:
                pass

        return int(time.time() * 1000)

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            result = float(value)
            if result != result:
                return None
            return result
        except (TypeError, ValueError):
            return None

    def _safe_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_str(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None