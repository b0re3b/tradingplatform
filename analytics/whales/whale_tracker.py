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
class WhaleTrackerConfig:
    """
    Конфігурація WhaleTracker.

    WhaleTracker не визначає large trade самостійно.
    Він працює поверх already detected large trades та liquidation events.
    """

    enabled: bool = True

    # EventBus event names
    large_trade_event_name: str = "analytics.whales.large_trade"
    liquidation_event_name: str = "market.liquidation"

    whale_activity_event_name: str = "analytics.whales.whale_activity"
    whale_pressure_event_name: str = "analytics.whales.whale_pressure"
    whale_liquidation_context_event_name: str = "analytics.whales.whale_liquidation_context"

    # Rolling windows
    cluster_window_sec: int = 30
    pressure_window_sec: int = 60
    liquidation_window_sec: int = 60

    # Thresholds
    cluster_min_trades: int = 3
    cluster_min_total_notional: float = 300_000.0

    pressure_min_trades: int = 4
    pressure_min_total_notional: float = 500_000.0
    pressure_imbalance_ratio_threshold: float = 0.65

    liquidation_context_min_notional: float = 100_000.0

    # Signal cooldowns per symbol
    whale_activity_cooldown_sec: float = 5.0
    whale_pressure_cooldown_sec: float = 5.0
    whale_liquidation_context_cooldown_sec: float = 5.0

    # Cleanup
    cleanup_interval_sec: int = 60
    stats_ttl_sec: int = 60 * 60

    # Logging / emissions
    emit_on_bus: bool = True
    log_signals: bool = True

    # Optional behavior
    subscribe_liquidations: bool = True


@dataclass(slots=True)
class WhaleTradeRecord:
    symbol: str
    side: str
    notional: float
    price: float
    quantity: float
    timestamp_ms: int
    zscore: float = 0.0
    trigger_type: str = "unknown"
    trade_id: Optional[str] = None
    exchange: Optional[str] = None
    raw_event: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class LiquidationRecord:
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
class WhaleActivitySignal:
    symbol: str
    side: str
    trade_count: int
    total_notional: float
    avg_notional: float
    max_notional: float
    window_sec: int
    timestamp_ms: int

    detector_name: str = "WhaleTracker"
    event_type: str = "whale_activity"
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_event(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "detector": self.detector_name,
            "symbol": self.symbol,
            "side": self.side,
            "trade_count": self.trade_count,
            "total_notional": self.total_notional,
            "avg_notional": self.avg_notional,
            "max_notional": self.max_notional,
            "window_sec": self.window_sec,
            "timestamp_ms": self.timestamp_ms,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(slots=True)
class WhalePressureSignal:
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

    detector_name: str = "WhaleTracker"
    event_type: str = "whale_pressure"
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_event(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "detector": self.detector_name,
            "symbol": self.symbol,
            "dominant_side": self.dominant_side,
            "buy_trade_count": self.buy_trade_count,
            "sell_trade_count": self.sell_trade_count,
            "buy_notional": self.buy_notional,
            "sell_notional": self.sell_notional,
            "total_notional": self.total_notional,
            "imbalance_ratio": self.imbalance_ratio,
            "net_flow_notional": self.net_flow_notional,
            "window_sec": self.window_sec,
            "timestamp_ms": self.timestamp_ms,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(slots=True)
class WhaleLiquidationContextSignal:
    symbol: str
    whale_side: str
    whale_total_notional: float
    whale_trade_count: int
    liquidation_side: str
    liquidation_total_notional: float
    liquidation_count: int
    context_strength: float
    timestamp_ms: int

    detector_name: str = "WhaleTracker"
    event_type: str = "whale_liquidation_context"
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_event(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "detector": self.detector_name,
            "symbol": self.symbol,
            "whale_side": self.whale_side,
            "whale_total_notional": self.whale_total_notional,
            "whale_trade_count": self.whale_trade_count,
            "liquidation_side": self.liquidation_side,
            "liquidation_total_notional": self.liquidation_total_notional,
            "liquidation_count": self.liquidation_count,
            "context_strength": self.context_strength,
            "timestamp_ms": self.timestamp_ms,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(slots=True)
class SymbolTrackerState:
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


class WhaleTracker:
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
        self.config = config or WhaleTrackerConfig()
        self.event_bus = event_bus
        self.scheduler = scheduler

        self.logger = get_logger(
            __name__,
            service_name="analytics.whales.whale_tracker",
        )

        self._states: Dict[str, SymbolTrackerState] = {}
        self._started = False
        self._cleanup_task: Optional[asyncio.Task[Any]] = None
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            self.logger.warning("WhaleTracker already started")
            return

        if not self.config.enabled:
            self.logger.info("WhaleTracker is disabled by config")
            return

        self._started = True

        if self.event_bus is not None:
            await self._safe_subscribe()

        if self.scheduler is not None:
            await self._register_scheduler_jobs()
        else:
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(),
                name="whale_tracker_cleanup_loop",
            )

        self.logger.info(
            "WhaleTracker started",
            extra={
                "large_trade_event_name": self.config.large_trade_event_name,
                "liquidation_event_name": self.config.liquidation_event_name,
                "cluster_window_sec": self.config.cluster_window_sec,
                "pressure_window_sec": self.config.pressure_window_sec,
                "liquidation_window_sec": self.config.liquidation_window_sec,
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
        self.logger.info("WhaleTracker stopped")

    async def _safe_subscribe(self) -> None:
        try:
            await self.event_bus.subscribe(
                self.config.large_trade_event_name,
                self.handle_large_trade_event,
            )
            self.logger.info(
                "Subscribed to large trade events",
                extra={"event_name": self.config.large_trade_event_name},
            )

            if self.config.subscribe_liquidations:
                await self.event_bus.subscribe(
                    self.config.liquidation_event_name,
                    self.handle_liquidation_event,
                )
                self.logger.info(
                    "Subscribed to liquidation events",
                    extra={"event_name": self.config.liquidation_event_name},
                )
        except Exception:
            self.logger.exception("Failed to subscribe WhaleTracker to EventBus")
            raise

    async def _safe_unsubscribe(self) -> None:
        try:
            await self.event_bus.unsubscribe(
                self.config.large_trade_event_name,
                self.handle_large_trade_event,
            )
        except Exception:
            self.logger.exception(
                "Failed to unsubscribe from large trade events",
                extra={"event_name": self.config.large_trade_event_name},
            )

        if self.config.subscribe_liquidations:
            try:
                await self.event_bus.unsubscribe(
                    self.config.liquidation_event_name,
                    self.handle_liquidation_event,
                )
            except Exception:
                self.logger.exception(
                    "Failed to unsubscribe from liquidation events",
                    extra={"event_name": self.config.liquidation_event_name},
                )

    async def _register_scheduler_jobs(self) -> None:
        try:
            await self.scheduler.add_interval_job(
                name="whale_tracker_cleanup",
                interval_seconds=self.config.cleanup_interval_sec,
                coro=self.cleanup,
                replace_existing=True,
            )
            self.logger.info(
                "Cleanup job registered in Scheduler",
                extra={"interval_sec": self.config.cleanup_interval_sec},
            )
        except Exception:
            self.logger.exception("Failed to register WhaleTracker cleanup job")
            raise

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

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

    async def process_large_trade_event(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Optional[object]]:
        if not self.config.enabled:
            return {
                "whale_activity_signal": None,
                "whale_pressure_signal": None,
                "whale_liquidation_context_signal": None,
            }

        record = self._normalize_large_trade_event(event)
        if record is None:
            return {
                "whale_activity_signal": None,
                "whale_pressure_signal": None,
                "whale_liquidation_context_signal": None,
            }

        async with self._lock:
            state = self._get_or_create_state(record.symbol)
            state.large_trades.append(record)
            state.total_large_trades_seen += 1
            state.touch()

            self._prune_symbol_state(state, record.timestamp_ms)

            whale_activity_signal = self._detect_whale_activity(record.symbol, state, record.timestamp_ms)
            whale_pressure_signal = self._detect_whale_pressure(record.symbol, state, record.timestamp_ms)
            whale_liquidation_context_signal = self._detect_whale_liquidation_context(
                record.symbol,
                state,
                record.timestamp_ms,
            )

        await self._emit_detected_signals(
            whale_activity_signal=whale_activity_signal,
            whale_pressure_signal=whale_pressure_signal,
            whale_liquidation_context_signal=whale_liquidation_context_signal,
        )

        return {
            "whale_activity_signal": whale_activity_signal,
            "whale_pressure_signal": whale_pressure_signal,
            "whale_liquidation_context_signal": whale_liquidation_context_signal,
        }

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
                record.symbol,
                state,
                record.timestamp_ms,
            )

        await self._emit_detected_signals(
            whale_activity_signal=None,
            whale_pressure_signal=None,
            whale_liquidation_context_signal=whale_liquidation_context_signal,
        )

        return whale_liquidation_context_signal

    # -------------------------------------------------------------------------
    # Detection logic
    # -------------------------------------------------------------------------

    def _detect_whale_activity(
        self,
        symbol: str,
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> Optional[WhaleActivitySignal]:
        cluster_start_ms = current_ts_ms - self.config.cluster_window_sec * 1000

        buys = [t for t in state.large_trades if t.timestamp_ms >= cluster_start_ms and t.side == "buy"]
        sells = [t for t in state.large_trades if t.timestamp_ms >= cluster_start_ms and t.side == "sell"]

        buy_signal = self._build_whale_activity_signal_if_triggered(
            symbol=symbol,
            side="buy",
            trades=buys,
            state=state,
            current_ts_ms=current_ts_ms,
        )
        if buy_signal is not None:
            return buy_signal

        sell_signal = self._build_whale_activity_signal_if_triggered(
            symbol=symbol,
            side="sell",
            trades=sells,
            state=state,
            current_ts_ms=current_ts_ms,
        )
        return sell_signal

    def _build_whale_activity_signal_if_triggered(
        self,
        symbol: str,
        side: str,
        trades: List[WhaleTradeRecord],
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> Optional[WhaleActivitySignal]:
        if len(trades) < self.config.cluster_min_trades:
            return None

        total_notional = sum(t.notional for t in trades)
        if total_notional < self.config.cluster_min_total_notional:
            return None

        if not self._passes_cooldown(
            last_ts=state.last_whale_activity_signal_ts_monotonic,
            cooldown_sec=self.config.whale_activity_cooldown_sec,
        ):
            return None

        max_notional = max(t.notional for t in trades)
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
        symbol: str,
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> Optional[WhalePressureSignal]:
        pressure_start_ms = current_ts_ms - self.config.pressure_window_sec * 1000
        trades = [t for t in state.large_trades if t.timestamp_ms >= pressure_start_ms]

        if len(trades) < self.config.pressure_min_trades:
            return None

        buy_trades = [t for t in trades if t.side == "buy"]
        sell_trades = [t for t in trades if t.side == "sell"]

        buy_notional = sum(t.notional for t in buy_trades)
        sell_notional = sum(t.notional for t in sell_trades)
        total_notional = buy_notional + sell_notional

        if total_notional < self.config.pressure_min_total_notional:
            return None

        dominant_side = "buy" if buy_notional >= sell_notional else "sell"
        dominant_notional = max(buy_notional, sell_notional)
        imbalance_ratio = dominant_notional / total_notional if total_notional > 0 else 0.0

        if imbalance_ratio < self.config.pressure_imbalance_ratio_threshold:
            return None

        if not self._passes_cooldown(
            last_ts=state.last_whale_pressure_signal_ts_monotonic,
            cooldown_sec=self.config.whale_pressure_cooldown_sec,
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
        symbol: str,
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> Optional[WhaleLiquidationContextSignal]:
        if not self.config.subscribe_liquidations:
            return None

        pressure_start_ms = current_ts_ms - self.config.pressure_window_sec * 1000
        liquidation_start_ms = current_ts_ms - self.config.liquidation_window_sec * 1000

        recent_whales = [t for t in state.large_trades if t.timestamp_ms >= pressure_start_ms]
        recent_liquidations = [l for l in state.liquidations if l.timestamp_ms >= liquidation_start_ms]

        if not recent_whales or not recent_liquidations:
            return None

        whale_buy = [t for t in recent_whales if t.side == "buy"]
        whale_sell = [t for t in recent_whales if t.side == "sell"]

        whale_buy_notional = sum(t.notional for t in whale_buy)
        whale_sell_notional = sum(t.notional for t in whale_sell)

        whale_side = "buy" if whale_buy_notional >= whale_sell_notional else "sell"
        whale_total_notional = max(whale_buy_notional, whale_sell_notional)
        whale_trade_count = len(whale_buy) if whale_side == "buy" else len(whale_sell)

        liq_buy = [l for l in recent_liquidations if l.side == "buy"]
        liq_sell = [l for l in recent_liquidations if l.side == "sell"]

        liq_buy_notional = sum(l.notional for l in liq_buy)
        liq_sell_notional = sum(l.notional for l in liq_sell)

        liquidation_side = "buy" if liq_buy_notional >= liq_sell_notional else "sell"
        liquidation_total_notional = max(liq_buy_notional, liq_sell_notional)
        liquidation_count = len(liq_buy) if liquidation_side == "buy" else len(liq_sell)

        if liquidation_total_notional < self.config.liquidation_context_min_notional:
            return None

        # Контекст цікавий тоді, коли великі трейди тиснуть в один бік,
        # а ліквідації переважають у протилежному.
        if whale_side == liquidation_side:
            return None

        if whale_trade_count <= 0 or liquidation_count <= 0:
            return None

        if not self._passes_cooldown(
            last_ts=state.last_whale_liquidation_context_signal_ts_monotonic,
            cooldown_sec=self.config.whale_liquidation_context_cooldown_sec,
        ):
            return None

        context_strength = min(
            1.0,
            (whale_total_notional + liquidation_total_notional)
            / max(
                self.config.pressure_min_total_notional + self.config.liquidation_context_min_notional,
                1.0,
            ),
        )

        signal = WhaleLiquidationContextSignal(
            symbol=symbol,
            whale_side=whale_side,
            whale_total_notional=whale_total_notional,
            whale_trade_count=whale_trade_count,
            liquidation_side=liquidation_side,
            liquidation_total_notional=liquidation_total_notional,
            liquidation_count=liquidation_count,
            context_strength=context_strength,
            timestamp_ms=current_ts_ms,
        )

        state.whale_liquidation_context_signals_emitted += 1
        state.last_whale_liquidation_context_signal_ts_monotonic = time.monotonic()

        return signal

    # -------------------------------------------------------------------------
    # Emission
    # -------------------------------------------------------------------------

    async def _emit_detected_signals(
        self,
        whale_activity_signal: Optional[WhaleActivitySignal],
        whale_pressure_signal: Optional[WhalePressureSignal],
        whale_liquidation_context_signal: Optional[WhaleLiquidationContextSignal],
    ) -> None:
        if whale_activity_signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Whale activity detected",
                    extra={
                        "symbol": whale_activity_signal.symbol,
                        "side": whale_activity_signal.side,
                        "trade_count": whale_activity_signal.trade_count,
                        "total_notional": whale_activity_signal.total_notional,
                    },
                )
            await self._emit_signal(
                event_name=self.config.whale_activity_event_name,
                payload=whale_activity_signal.to_event(),
            )

        if whale_pressure_signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Whale pressure detected",
                    extra={
                        "symbol": whale_pressure_signal.symbol,
                        "dominant_side": whale_pressure_signal.dominant_side,
                        "imbalance_ratio": whale_pressure_signal.imbalance_ratio,
                        "total_notional": whale_pressure_signal.total_notional,
                    },
                )
            await self._emit_signal(
                event_name=self.config.whale_pressure_event_name,
                payload=whale_pressure_signal.to_event(),
            )

        if whale_liquidation_context_signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Whale liquidation context detected",
                    extra={
                        "symbol": whale_liquidation_context_signal.symbol,
                        "whale_side": whale_liquidation_context_signal.whale_side,
                        "liquidation_side": whale_liquidation_context_signal.liquidation_side,
                        "context_strength": whale_liquidation_context_signal.context_strength,
                    },
                )
            await self._emit_signal(
                event_name=self.config.whale_liquidation_context_event_name,
                payload=whale_liquidation_context_signal.to_event(),
            )

    async def _emit_signal(self, event_name: str, payload: Dict[str, Any]) -> None:
        if not self.config.emit_on_bus or self.event_bus is None:
            return

        try:
            await self.event_bus.emit(event_name, payload)
        except Exception:
            self.logger.exception(
                "Failed to emit WhaleTracker signal",
                extra={"event_name": event_name},
            )

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------

    def _normalize_large_trade_event(self, event: Dict[str, Any]) -> Optional[WhaleTradeRecord]:
        try:
            payload = event.get("data", event)

            symbol = self._normalize_symbol(payload.get("symbol") or payload.get("s"))
            side = self._normalize_side(payload.get("side"))
            notional = self._safe_float(payload.get("notional"))
            price = self._safe_float(payload.get("price"))
            quantity = self._safe_float(payload.get("quantity"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if not symbol or not side or notional <= 0:
                return None

            return WhaleTradeRecord(
                symbol=symbol,
                side=side,
                notional=notional,
                price=price,
                quantity=quantity,
                timestamp_ms=timestamp_ms,
                zscore=self._safe_float(payload.get("zscore")),
                trigger_type=str(payload.get("trigger_type", "unknown")),
                trade_id=self._safe_str(payload.get("trade_id")),
                exchange=self._safe_str(payload.get("exchange")),
                raw_event=event,
            )
        except Exception:
            self.logger.exception("Failed to normalize large trade event")
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
            price = self._safe_float(payload.get("price") or payload.get("p"))
            quantity = self._safe_float(
                payload.get("quantity")
                or payload.get("qty")
                or payload.get("q")
                or payload.get("size")
            )
            notional = self._safe_float(payload.get("notional"))
            if notional <= 0 and price > 0 and quantity > 0:
                notional = price * quantity

            timestamp_ms = self._extract_timestamp_ms(payload)

            if not symbol or not side or notional <= 0:
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
                ),
                exchange=self._safe_str(payload.get("exchange") or event.get("exchange")),
                raw_event=event,
            )
        except Exception:
            self.logger.exception("Failed to normalize liquidation event")
            return None

    # -------------------------------------------------------------------------
    # State / housekeeping
    # -------------------------------------------------------------------------

    def _get_or_create_state(self, symbol: str) -> SymbolTrackerState:
        state = self._states.get(symbol)
        if state is None:
            state = SymbolTrackerState(
                large_trades=deque(),
                liquidations=deque(),
            )
            self._states[symbol] = state
        return state

    def _prune_symbol_state(self, state: SymbolTrackerState, current_ts_ms: int) -> None:
        keep_from_ms = current_ts_ms - max(
            self.config.cluster_window_sec,
            self.config.pressure_window_sec,
            self.config.liquidation_window_sec,
        ) * 1000

        while state.large_trades and state.large_trades[0].timestamp_ms < keep_from_ms:
            state.large_trades.popleft()

        while state.liquidations and state.liquidations[0].timestamp_ms < keep_from_ms:
            state.liquidations.popleft()

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
                "Cleaned stale WhaleTracker states",
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
            self.logger.debug("WhaleTracker cleanup loop cancelled")
            raise
        except Exception:
            self.logger.exception("Unexpected error in WhaleTracker cleanup loop")

    # -------------------------------------------------------------------------
    # Public stats
    # -------------------------------------------------------------------------

    def get_symbol_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        state = self._states.get(symbol)
        if state is None:
            return None

        buy_notional = sum(t.notional for t in state.large_trades if t.side == "buy")
        sell_notional = sum(t.notional for t in state.large_trades if t.side == "sell")

        return {
            "symbol": symbol,
            "large_trades_buffer_size": len(state.large_trades),
            "liquidations_buffer_size": len(state.liquidations),
            "total_large_trades_seen": state.total_large_trades_seen,
            "total_liquidations_seen": state.total_liquidations_seen,
            "current_buy_notional": buy_notional,
            "current_sell_notional": sell_notional,
            "whale_activity_signals_emitted": state.whale_activity_signals_emitted,
            "whale_pressure_signals_emitted": state.whale_pressure_signals_emitted,
            "whale_liquidation_context_signals_emitted": state.whale_liquidation_context_signals_emitted,
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
            "Reset symbol state in WhaleTracker",
            extra={"symbol": symbol},
        )

    async def reset_all(self) -> None:
        async with self._lock:
            self._states.clear()

        self.logger.info("Reset all WhaleTracker states")

    # -------------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------------

    def _passes_cooldown(self, last_ts: float, cooldown_sec: float) -> bool:
        now = time.monotonic()
        return (now - last_ts) >= cooldown_sec

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

    def _safe_str(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

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