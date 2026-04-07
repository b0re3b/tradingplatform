from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from analytics.whales.base import BaseWhaleComponent
from analytics.whales.config import WhaleTrackerConfig
from analytics.whales.enums import WhaleTradeSide
from analytics.whales.models import (
    LiquidationRecord,
    WhaleActivitySignal,
    WhaleLiquidationContextSignal,
    WhalePressureSignal,
    WhaleTradeRecord,
    WhaleTrackerResult,
    SymbolTrackerState,
    make_symbol_tracker_state,
)


class WhaleTracker(BaseWhaleComponent):
    """
    High-level tracker whale activity.

    Вхід:
        - large trade signals від LargeTradeDetector
        - liquidation events (опційно)

    Вихід:
        - whale_activity
        - whale_pressure
        - whale_liquidation_context
    """

    def __init__(
        self,
        config: Optional[WhaleTrackerConfig] = None,
        event_bus: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        super().__init__(
            component_name="whale_tracker",
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.config = config or WhaleTrackerConfig()

        self._states: Dict[str, SymbolTrackerState] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task[Any]] = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        if self._started:
            self.logger.warning("WhaleTracker already started")
            return

        if not self.config.enabled:
            self.logger.info("WhaleTracker is disabled by config")
            return

        if self.event_bus is not None:
            await self._safe_subscribe(
                self.config.large_trade_event_name,
                self.handle_large_trade_event,
            )

            if self.config.subscribe_liquidations:
                await self._safe_subscribe(
                    self.config.liquidation_event_name,
                    self.handle_liquidation_event,
                )

        if self.scheduler is not None:
            await self._register_interval_job(
                name="whales_whale_tracker_cleanup",
                interval_seconds=self.config.cleanup_interval_sec,
                coro=self.cleanup,
                replace_existing=True,
            )
        else:
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(),
                name="whales_whale_tracker_cleanup_loop",
            )

        self._started = True

        self.logger.info(
            "WhaleTracker started",
            extra={
                "large_trade_event_name": self.config.large_trade_event_name,
                "liquidation_event_name": self.config.liquidation_event_name,
                "cluster_window_sec": self.config.cluster_window_sec,
                "pressure_window_sec": self.config.pressure_window_sec,
                "liquidation_window_sec": self.config.liquidation_window_sec,
                "subscribe_liquidations": self.config.subscribe_liquidations,
            },
        )

    async def stop(self) -> None:
        if not self._started:
            return

        if self.event_bus is not None:
            await self._safe_unsubscribe(
                self.config.large_trade_event_name,
                self.handle_large_trade_event,
            )

            if self.config.subscribe_liquidations:
                await self._safe_unsubscribe(
                    self.config.liquidation_event_name,
                    self.handle_liquidation_event,
                )

        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

        self._started = False
        self.logger.info("WhaleTracker stopped")

    # =========================================================================
    # Event handlers
    # =========================================================================

    async def handle_large_trade_event(self, event: Dict[str, Any]) -> None:
        try:
            await self.process_large_trade_event(event)
        except Exception:
            self.logger.exception("Unhandled error while processing large trade event")

    async def handle_liquidation_event(self, event: Dict[str, Any]) -> None:
        try:
            await self.process_liquidation_event(event)
        except Exception:
            self.logger.exception("Unhandled error while processing liquidation event")

    # =========================================================================
    # Public processing API
    # =========================================================================

    async def process_large_trade_event(
        self,
        event: Dict[str, Any],
    ) -> WhaleTrackerResult:
        if not self.config.enabled:
            return WhaleTrackerResult()

        record = self._normalize_large_trade_event(event)
        if record is None:
            return WhaleTrackerResult()

        async with self._lock:
            state = self._get_or_create_state(record.symbol)
            state.large_trades.append(record)
            state.total_large_trades_seen += 1
            state.touch()

            self._prune_symbol_state(state, record.timestamp_ms)

            whale_activity_signal = self._detect_whale_activity(
                symbol=record.symbol,
                state=state,
                current_ts_ms=record.timestamp_ms,
            )
            whale_pressure_signal = self._detect_whale_pressure(
                symbol=record.symbol,
                state=state,
                current_ts_ms=record.timestamp_ms,
            )
            whale_liquidation_context_signal = self._detect_whale_liquidation_context(
                symbol=record.symbol,
                state=state,
                current_ts_ms=record.timestamp_ms,
            )

        result = WhaleTrackerResult(
            whale_activity_signal=whale_activity_signal,
            whale_pressure_signal=whale_pressure_signal,
            whale_liquidation_context_signal=whale_liquidation_context_signal,
        )

        await self._emit_detected_signals(result)
        return result

    async def process_liquidation_event(
        self,
        event: Dict[str, Any],
    ) -> Optional[WhaleLiquidationContextSignal]:
        if not self.config.enabled or not self.config.subscribe_liquidations:
            return None

        record = self._normalize_liquidation_event(event)
        if record is None:
            return None

        async with self._lock:
            state = self._get_or_create_state(record.symbol)
            state.liquidations.append(record)
            state.total_liquidations_seen += 1
            state.touch()

            self._prune_symbol_state(state, record.timestamp_ms)

            whale_liquidation_context_signal = self._detect_whale_liquidation_context(
                symbol=record.symbol,
                state=state,
                current_ts_ms=record.timestamp_ms,
            )

        if whale_liquidation_context_signal is not None:
            await self._emit_detected_signals(
                WhaleTrackerResult(
                    whale_liquidation_context_signal=whale_liquidation_context_signal
                )
            )

        return whale_liquidation_context_signal

    # =========================================================================
    # Detection logic
    # =========================================================================

    def _detect_whale_activity(
        self,
        *,
        symbol: str,
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> Optional[WhaleActivitySignal]:
        cluster_start_ms = current_ts_ms - self.config.cluster_window_sec * 1000

        buys = [
            trade
            for trade in state.large_trades
            if trade.timestamp_ms >= cluster_start_ms and trade.side == WhaleTradeSide.BUY.value
        ]
        sells = [
            trade
            for trade in state.large_trades
            if trade.timestamp_ms >= cluster_start_ms and trade.side == WhaleTradeSide.SELL.value
        ]

        buy_signal = self._build_whale_activity_signal_if_triggered(
            symbol=symbol,
            side=WhaleTradeSide.BUY.value,
            trades=buys,
            state=state,
            current_ts_ms=current_ts_ms,
        )
        if buy_signal is not None:
            return buy_signal

        sell_signal = self._build_whale_activity_signal_if_triggered(
            symbol=symbol,
            side=WhaleTradeSide.SELL.value,
            trades=sells,
            state=state,
            current_ts_ms=current_ts_ms,
        )
        return sell_signal

    def _build_whale_activity_signal_if_triggered(
        self,
        *,
        symbol: str,
        side: str,
        trades: List[WhaleTradeRecord],
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> Optional[WhaleActivitySignal]:
        if len(trades) < self.config.cluster_min_trades:
            return None

        total_notional = sum(trade.notional for trade in trades)
        if total_notional < self.config.cluster_min_total_notional:
            return None

        if not self._passes_cooldown(
            state.last_whale_activity_signal_ts_monotonic,
            self.config.whale_activity_cooldown_sec,
        ):
            return None

        max_notional = max(trade.notional for trade in trades)
        avg_notional = total_notional / len(trades)

        signal = WhaleActivitySignal(
            symbol=symbol,
            side=side,
            trade_count=len(trades),
            total_notional=total_notional,
            avg_notional=avg_notional,
            max_notional=max_notional,
            window_sec=self.config.cluster_window_sec,
            timestamp_ms=current_ts_ms,
        )

        state.whale_activity_signals_emitted += 1
        state.last_whale_activity_signal_ts_monotonic = time.monotonic()
        return signal

    def _detect_whale_pressure(
        self,
        *,
        symbol: str,
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> Optional[WhalePressureSignal]:
        pressure_start_ms = current_ts_ms - self.config.pressure_window_sec * 1000
        trades = [
            trade
            for trade in state.large_trades
            if trade.timestamp_ms >= pressure_start_ms
        ]

        if len(trades) < self.config.pressure_min_trades:
            return None

        buy_trades = [trade for trade in trades if trade.side == WhaleTradeSide.BUY.value]
        sell_trades = [trade for trade in trades if trade.side == WhaleTradeSide.SELL.value]

        buy_notional = sum(trade.notional for trade in buy_trades)
        sell_notional = sum(trade.notional for trade in sell_trades)
        total_notional = buy_notional + sell_notional

        if total_notional < self.config.pressure_min_total_notional:
            return None

        dominant_side = (
            WhaleTradeSide.BUY.value
            if buy_notional >= sell_notional
            else WhaleTradeSide.SELL.value
        )
        dominant_notional = max(buy_notional, sell_notional)
        imbalance_ratio = dominant_notional / total_notional if total_notional > 0 else 0.0

        if imbalance_ratio < self.config.pressure_imbalance_ratio_threshold:
            return None

        if not self._passes_cooldown(
            state.last_whale_pressure_signal_ts_monotonic,
            self.config.whale_pressure_cooldown_sec,
        ):
            return None

        signal = WhalePressureSignal(
            symbol=symbol,
            dominant_side=dominant_side,
            buy_trade_count=len(buy_trades),
            sell_trade_count=len(sell_trades),
            buy_notional=buy_notional,
            sell_notional=sell_notional,
            total_notional=total_notional,
            imbalance_ratio=imbalance_ratio,
            net_flow_notional=buy_notional - sell_notional,
            window_sec=self.config.pressure_window_sec,
            timestamp_ms=current_ts_ms,
        )

        state.whale_pressure_signals_emitted += 1
        state.last_whale_pressure_signal_ts_monotonic = time.monotonic()
        return signal

    def _detect_whale_liquidation_context(
        self,
        *,
        symbol: str,
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> Optional[WhaleLiquidationContextSignal]:
        whale_window_start_ms = current_ts_ms - self.config.cluster_window_sec * 1000
        liquidation_window_start_ms = current_ts_ms - self.config.liquidation_window_sec * 1000

        recent_whale_trades = [
            trade
            for trade in state.large_trades
            if trade.timestamp_ms >= whale_window_start_ms
        ]
        recent_liquidations = [
            liquidation
            for liquidation in state.liquidations
            if liquidation.timestamp_ms >= liquidation_window_start_ms
        ]

        if not recent_whale_trades or not recent_liquidations:
            return None

        buy_whale_trades = [
            trade for trade in recent_whale_trades
            if trade.side == WhaleTradeSide.BUY.value
        ]
        sell_whale_trades = [
            trade for trade in recent_whale_trades
            if trade.side == WhaleTradeSide.SELL.value
        ]

        buy_whale_notional = sum(trade.notional for trade in buy_whale_trades)
        sell_whale_notional = sum(trade.notional for trade in sell_whale_trades)

        if max(buy_whale_notional, sell_whale_notional) < self.config.liquidation_context_min_notional:
            return None

        whale_side = (
            WhaleTradeSide.BUY.value
            if buy_whale_notional >= sell_whale_notional
            else WhaleTradeSide.SELL.value
        )

        if whale_side == WhaleTradeSide.BUY.value:
            whale_trades = buy_whale_trades
        else:
            whale_trades = sell_whale_trades

        opposite_liquidation_side = (
            WhaleTradeSide.SELL.value
            if whale_side == WhaleTradeSide.BUY.value
            else WhaleTradeSide.BUY.value
        )

        related_liquidations = [
            liquidation
            for liquidation in recent_liquidations
            if liquidation.side == opposite_liquidation_side
        ]

        if not related_liquidations:
            return None

        whale_total_notional = sum(trade.notional for trade in whale_trades)
        liquidation_total_notional = sum(liquidation.notional for liquidation in related_liquidations)

        if whale_total_notional < self.config.liquidation_context_min_notional:
            return None

        if not self._passes_cooldown(
            state.last_whale_liquidation_context_signal_ts_monotonic,
            self.config.whale_liquidation_context_cooldown_sec,
        ):
            return None

        context_strength = self._calculate_context_strength(
            whale_total_notional=whale_total_notional,
            liquidation_total_notional=liquidation_total_notional,
            whale_trade_count=len(whale_trades),
            liquidation_count=len(related_liquidations),
        )

        signal = WhaleLiquidationContextSignal(
            symbol=symbol,
            whale_side=whale_side,
            whale_total_notional=whale_total_notional,
            whale_trade_count=len(whale_trades),
            liquidation_side=opposite_liquidation_side,
            liquidation_total_notional=liquidation_total_notional,
            liquidation_count=len(related_liquidations),
            context_strength=context_strength,
            timestamp_ms=current_ts_ms,
        )

        state.whale_liquidation_context_signals_emitted += 1
        state.last_whale_liquidation_context_signal_ts_monotonic = time.monotonic()
        return signal

    def _calculate_context_strength(
        self,
        *,
        whale_total_notional: float,
        liquidation_total_notional: float,
        whale_trade_count: int,
        liquidation_count: int,
    ) -> float:
        if whale_total_notional <= 0:
            return 0.0

        liquidation_ratio = liquidation_total_notional / whale_total_notional
        trade_factor = min(1.0, whale_trade_count / max(1, self.config.cluster_min_trades))
        liquidation_factor = min(1.0, liquidation_count / 3.0)

        raw_score = 0.6 * min(1.0, liquidation_ratio) + 0.2 * trade_factor + 0.2 * liquidation_factor
        return self._clamp_0_1(raw_score)

    # =========================================================================
    # Emission
    # =========================================================================

    async def _emit_detected_signals(self, result: WhaleTrackerResult) -> None:
        if not self.config.emit_on_bus:
            return

        if result.whale_activity_signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Whale activity detected",
                    extra={
                        "symbol": result.whale_activity_signal.symbol,
                        "side": result.whale_activity_signal.side,
                        "trade_count": result.whale_activity_signal.trade_count,
                        "total_notional": result.whale_activity_signal.total_notional,
                    },
                )

            await self._safe_emit(
                self.config.whale_activity_event_name,
                result.whale_activity_signal.to_event(),
                source="analytics.whales.whale_tracker",
            )

        if result.whale_pressure_signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Whale pressure detected",
                    extra={
                        "symbol": result.whale_pressure_signal.symbol,
                        "dominant_side": result.whale_pressure_signal.dominant_side,
                        "imbalance_ratio": result.whale_pressure_signal.imbalance_ratio,
                        "total_notional": result.whale_pressure_signal.total_notional,
                    },
                )

            await self._safe_emit(
                self.config.whale_pressure_event_name,
                result.whale_pressure_signal.to_event(),
                source="analytics.whales.whale_tracker",
            )

        if result.whale_liquidation_context_signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Whale liquidation context detected",
                    extra={
                        "symbol": result.whale_liquidation_context_signal.symbol,
                        "whale_side": result.whale_liquidation_context_signal.whale_side,
                        "liquidation_side": result.whale_liquidation_context_signal.liquidation_side,
                        "context_strength": result.whale_liquidation_context_signal.context_strength,
                    },
                )

            await self._safe_emit(
                self.config.whale_liquidation_context_event_name,
                result.whale_liquidation_context_signal.to_event(),
                source="analytics.whales.whale_tracker",
            )

    # =========================================================================
    # State management
    # =========================================================================

    def _get_or_create_state(self, symbol: str) -> SymbolTrackerState:
        state = self._states.get(symbol)
        if state is not None:
            return state

        state = make_symbol_tracker_state(
            large_trade_window_size=self.config.large_trade_buffer_size,
            liquidation_window_size=self.config.liquidation_buffer_size,
        )
        self._states[symbol] = state
        return state

    def _prune_symbol_state(
        self,
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> None:
        cluster_cutoff_ms = current_ts_ms - self.config.cluster_window_sec * 1000
        pressure_cutoff_ms = current_ts_ms - self.config.pressure_window_sec * 1000
        liquidation_cutoff_ms = current_ts_ms - self.config.liquidation_window_sec * 1000

        trade_cutoff_ms = min(cluster_cutoff_ms, pressure_cutoff_ms)

        while state.large_trades and state.large_trades[0].timestamp_ms < trade_cutoff_ms:
            state.large_trades.popleft()

        while state.liquidations and state.liquidations[0].timestamp_ms < liquidation_cutoff_ms:
            state.liquidations.popleft()

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
                "Cleaned stale WhaleTracker symbol states",
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
            self.logger.exception("Unhandled error in WhaleTracker cleanup loop")

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
            "Reset WhaleTracker symbol state",
            extra={"symbol": symbol},
        )

    async def reset_all(self) -> None:
        async with self._lock:
            self._states.clear()

        self.logger.info("Reset all WhaleTracker states")

    # =========================================================================
    # Normalization helpers
    # =========================================================================

    def _normalize_large_trade_event(self, event: Dict[str, Any]) -> Optional[WhaleTradeRecord]:
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
                or payload.get("dominant_side")
            )
            price = self._safe_float(
                payload.get("price")
                or payload.get("p")
            )
            quantity = self._safe_float(
                payload.get("quantity")
                or payload.get("qty")
                or payload.get("q")
                or payload.get("size")
            )
            notional = self._safe_float(payload.get("notional"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if symbol is None or side == WhaleTradeSide.UNKNOWN.value:
                return None

            if price is None or price <= 0:
                return None

            if quantity is None or quantity <= 0:
                return None

            if notional is None:
                notional = price * quantity

            if notional <= 0:
                return None

            return WhaleTradeRecord(
                symbol=symbol,
                side=side,
                notional=notional,
                price=price,
                quantity=quantity,
                timestamp_ms=timestamp_ms,
                zscore=self._safe_float(payload.get("zscore")) or 0.0,
                trigger_type=str(payload.get("trigger_type") or "unknown"),
                trade_id=self._safe_str(
                    payload.get("trade_id")
                    or payload.get("id")
                    or payload.get("t")
                ),
                exchange=self._safe_str(
                    payload.get("exchange")
                    or event.get("exchange")
                ),
                raw_event=event,
            )
        except Exception:
            self.logger.exception(
                "Failed to normalize large trade event",
                extra={"event": event},
            )
            return None

    def _normalize_liquidation_event(self, event: Dict[str, Any]) -> Optional[LiquidationRecord]:
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
                or payload.get("direction")
            )
            price = self._safe_float(
                payload.get("price")
                or payload.get("p")
            )
            quantity = self._safe_float(
                payload.get("quantity")
                or payload.get("qty")
                or payload.get("q")
                or payload.get("size")
            )
            notional = self._safe_float(payload.get("notional"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if symbol is None or side == WhaleTradeSide.UNKNOWN.value:
                return None

            if price is None or price <= 0:
                return None

            if quantity is None or quantity <= 0:
                return None

            if notional is None:
                notional = price * quantity

            if notional <= 0:
                return None

            return LiquidationRecord(
                symbol=symbol,
                side=side,
                notional=notional,
                price=price,
                quantity=quantity,
                timestamp_ms=timestamp_ms,
                liquidation_id=self._safe_str(
                    payload.get("liquidation_id")
                    or payload.get("id")
                    or payload.get("t")
                ),
                exchange=self._safe_str(
                    payload.get("exchange")
                    or event.get("exchange")
                ),
                raw_event=event,
            )
        except Exception:
            self.logger.exception(
                "Failed to normalize liquidation event",
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

    def _safe_str(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None