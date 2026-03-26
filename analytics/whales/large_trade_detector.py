from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev
from typing import Any, Deque, Dict, Optional

from core.logger import get_logger


@dataclass(slots=True)
class NormalizedTrade:
    """
    Нормалізований трейд для внутрішньої обробки.
    """
    symbol: str
    price: float
    quantity: float
    side: str
    timestamp: float
    notional: float
    aggressor: Optional[str] = None
    exchange: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LargeTradeSignal:
    """
    Сигнал про великий трейд.
    """
    symbol: str
    side: str
    price: float
    quantity: float
    notional: float
    timestamp: float

    event_type: str = "large_trade_detected"
    trigger_type: str = "absolute"
    score: float = 0.0
    zscore: float = 0.0
    abs_threshold: float = 0.0
    relative_threshold_value: float = 0.0
    rolling_mean_notional: float = 0.0
    rolling_std_notional: float = 0.0

    aggressor: Optional[str] = None
    exchange: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LargeTradeDetectorStats:
    """
    Статистика детектора по символу.
    """
    symbol: str
    processed_trades: int = 0
    detected_large_trades: int = 0
    last_trade_ts: float = 0.0
    last_signal_ts: float = 0.0
    rolling_mean_notional: float = 0.0
    rolling_std_notional: float = 0.0
    current_abs_threshold: float = 0.0


class LargeTradeDetector:
    """
    Детектор одиничних великих трейдів.

    Логіка:
    1. Приймає market trade events із EventBus
    2. Нормалізує подію
    3. Обчислює notional = price * quantity
    4. Перевіряє абсолютний поріг
    5. Перевіряє відносну аномалію через z-score
    6. Якщо умова спрацювала — публікує сигнал у EventBus

    Підтримує:
    - symbol-specific thresholds
    - rolling baseline
    - cooldown між сигналами
    - мінімальний notional filter
    """

    def __init__(
        self,
        event_bus: Any,
        *,
        service_name: str = "large_trade_detector",
        enabled: bool = True,
        subscribe_event_name: str = "market.trade",
        publish_event_name: str = "analytics.whales.large_trade",
        default_abs_notional_threshold: float = 100_000.0,
        symbol_abs_thresholds: Optional[Dict[str, float]] = None,
        rolling_window_size: int = 300,
        min_samples_for_relative_detection: int = 30,
        zscore_threshold: float = 3.0,
        min_notional_filter: float = 10_000.0,
        signal_cooldown_sec: float = 0.0,
        side_filter: Optional[str] = None,
        history_retention_sec: float = 300.0,
        cleanup_interval_sec: float = 60.0,
    ) -> None:
        self._event_bus = event_bus
        self._enabled = enabled

        self._subscribe_event_name = subscribe_event_name
        self._publish_event_name = publish_event_name

        self._default_abs_notional_threshold = float(default_abs_notional_threshold)
        self._symbol_abs_thresholds = symbol_abs_thresholds or {}

        self._rolling_window_size = int(rolling_window_size)
        self._min_samples_for_relative_detection = int(min_samples_for_relative_detection)
        self._zscore_threshold = float(zscore_threshold)
        self._min_notional_filter = float(min_notional_filter)
        self._signal_cooldown_sec = float(signal_cooldown_sec)
        self._side_filter = side_filter.lower() if side_filter else None

        self._history_retention_sec = float(history_retention_sec)
        self._cleanup_interval_sec = float(cleanup_interval_sec)

        self._logger = get_logger(__name__, service_name=service_name)

        self._lock = asyncio.Lock()
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

        self._notional_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._rolling_window_size)
        )
        self._stats: Dict[str, LargeTradeDetectorStats] = {}
        self._last_signal_ts_by_symbol: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            self._logger.warning("LargeTradeDetector already running")
            return

        self._running = True

        if hasattr(self._event_bus, "subscribe"):
            await self._event_bus.subscribe(self._subscribe_event_name, self.handle_trade_event)

        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="large_trade_detector_cleanup",
        )

        self._logger.info(
            "LargeTradeDetector started",
            extra={
                "subscribe_event_name": self._subscribe_event_name,
                "publish_event_name": self._publish_event_name,
                "default_abs_notional_threshold": self._default_abs_notional_threshold,
                "rolling_window_size": self._rolling_window_size,
                "zscore_threshold": self._zscore_threshold,
                "min_notional_filter": self._min_notional_filter,
                "signal_cooldown_sec": self._signal_cooldown_sec,
                "side_filter": self._side_filter,
            },
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._logger.info("LargeTradeDetector stopped")

    # -------------------------------------------------------------------------
    # public handlers
    # -------------------------------------------------------------------------

    async def handle_trade_event(self, event: Dict[str, Any]) -> None:
        if not self._enabled:
            return

        try:
            trade = self._normalize_trade(event)
            if trade is None:
                return

            signal = await self.process_trade(trade)
            if signal is not None:
                await self._publish_signal(signal)

        except Exception as exc:
            self._logger.exception(
                "Failed to handle trade event in LargeTradeDetector",
                extra={"error": str(exc), "event": event},
            )

    async def process_trade(self, trade: NormalizedTrade) -> Optional[LargeTradeSignal]:
        """
        Обробляє один нормалізований трейд і повертає сигнал, якщо знайдена велика угода.
        """
        if trade.notional < self._min_notional_filter:
            return None

        if self._side_filter and trade.side != self._side_filter:
            return None

        async with self._lock:
            stats = self._stats.setdefault(
                trade.symbol,
                LargeTradeDetectorStats(symbol=trade.symbol),
            )
            stats.processed_trades += 1
            stats.last_trade_ts = trade.timestamp

            history = self._notional_history[trade.symbol]

            abs_threshold = self._get_abs_threshold(trade.symbol)
            rolling_mean, rolling_std = self._compute_rolling_stats(history)
            zscore = self._compute_zscore(
                trade.notional,
                rolling_mean,
                rolling_std,
            )

            absolute_trigger = trade.notional >= abs_threshold
            relative_trigger = (
                len(history) >= self._min_samples_for_relative_detection
                and rolling_std > 0
                and zscore >= self._zscore_threshold
            )

            stats.rolling_mean_notional = rolling_mean
            stats.rolling_std_notional = rolling_std
            stats.current_abs_threshold = abs_threshold

            # додаємо трейд після обчислення baseline
            history.append(trade.notional)

            if not (absolute_trigger or relative_trigger):
                return None

            if not self._passes_cooldown(trade.symbol, trade.timestamp):
                self._logger.debug(
                    "Large trade signal skipped by cooldown",
                    extra={"symbol": trade.symbol, "timestamp": trade.timestamp},
                )
                return None

            trigger_type = self._resolve_trigger_type(absolute_trigger, relative_trigger)
            relative_threshold_value = (
                rolling_mean + self._zscore_threshold * rolling_std
                if rolling_std > 0
                else 0.0
            )

            score = self._compute_score(
                notional=trade.notional,
                abs_threshold=abs_threshold,
                zscore=zscore,
            )

            signal = LargeTradeSignal(
                symbol=trade.symbol,
                side=trade.side,
                price=trade.price,
                quantity=trade.quantity,
                notional=trade.notional,
                timestamp=trade.timestamp,
                event_type="large_trade_detected",
                trigger_type=trigger_type,
                score=score,
                zscore=zscore,
                abs_threshold=abs_threshold,
                relative_threshold_value=relative_threshold_value,
                rolling_mean_notional=rolling_mean,
                rolling_std_notional=rolling_std,
                aggressor=trade.aggressor,
                exchange=trade.exchange,
                metadata={
                    "absolute_trigger": absolute_trigger,
                    "relative_trigger": relative_trigger,
                },
            )

            stats.detected_large_trades += 1
            stats.last_signal_ts = trade.timestamp
            self._last_signal_ts_by_symbol[trade.symbol] = trade.timestamp

            self._logger.info(
                "Large trade detected",
                extra={
                    "symbol": signal.symbol,
                    "side": signal.side,
                    "price": signal.price,
                    "quantity": signal.quantity,
                    "notional": signal.notional,
                    "score": signal.score,
                    "zscore": signal.zscore,
                    "trigger_type": signal.trigger_type,
                },
            )

            return signal

    # -------------------------------------------------------------------------
    # normalization
    # -------------------------------------------------------------------------

    def _normalize_trade(self, event: Dict[str, Any]) -> Optional[NormalizedTrade]:
        symbol = event.get("symbol")
        if not symbol:
            return None

        price = self._safe_float(event.get("price"))
        quantity = self._safe_float(
            event.get("quantity", event.get("qty", event.get("size", event.get("amount"))))
        )

        if price <= 0 or quantity <= 0:
            self._logger.debug(
                "Trade skipped due to invalid price/quantity",
                extra={"symbol": symbol, "price": price, "quantity": quantity},
            )
            return None

        side = str(event.get("side", "unknown")).lower()
        timestamp = self._extract_timestamp(event)
        symbol = str(symbol).upper()

        return NormalizedTrade(
            symbol=symbol,
            price=price,
            quantity=quantity,
            side=side,
            timestamp=timestamp,
            notional=price * quantity,
            aggressor=event.get("aggressor"),
            exchange=event.get("exchange"),
            raw=event,
        )

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------

    def _get_abs_threshold(self, symbol: str) -> float:
        return float(
            self._symbol_abs_thresholds.get(symbol.upper(), self._default_abs_notional_threshold)
        )

    def _compute_rolling_stats(self, history: Deque[float]) -> tuple[float, float]:
        if not history:
            return 0.0, 0.0
        if len(history) == 1:
            return history[0], 0.0
        values = list(history)
        return mean(values), pstdev(values)

    @staticmethod
    def _compute_zscore(value: float, mean_value: float, std_value: float) -> float:
        if std_value <= 0:
            return 0.0
        return (value - mean_value) / std_value

    def _resolve_trigger_type(self, absolute_trigger: bool, relative_trigger: bool) -> str:
        if absolute_trigger and relative_trigger:
            return "absolute_and_relative"
        if absolute_trigger:
            return "absolute"
        return "relative"

    def _passes_cooldown(self, symbol: str, timestamp: float) -> bool:
        if self._signal_cooldown_sec <= 0:
            return True

        last_ts = self._last_signal_ts_by_symbol.get(symbol)
        if last_ts is None:
            return True

        return (timestamp - last_ts) >= self._signal_cooldown_sec

    def _compute_score(
        self,
        *,
        notional: float,
        abs_threshold: float,
        zscore: float,
    ) -> float:
        """
        Простий комбінований score:
        - основа: наскільки трейд перевищив абсолютний поріг
        - бонус: наскільки він аномальний відносно baseline
        """
        abs_component = (notional / abs_threshold) if abs_threshold > 0 else 0.0
        rel_component = max(zscore, 0.0) / 5.0

        score = abs_component * 0.75 + rel_component * 0.25
        return round(score, 6)

    def _extract_timestamp(self, event: Dict[str, Any]) -> float:
        raw_ts = (
            event.get("timestamp")
            or event.get("ts")
            or event.get("event_time")
            or event.get("trade_time")
            or time.time()
        )

        ts = self._safe_float(raw_ts)
        if ts <= 0:
            return time.time()

        if ts > 10_000_000_000:
            ts /= 1000.0

        return ts

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    # -------------------------------------------------------------------------
    # publishing
    # -------------------------------------------------------------------------

    async def _publish_signal(self, signal: LargeTradeSignal) -> None:
        payload = asdict(signal)

        if hasattr(self._event_bus, "publish"):
            await self._event_bus.publish(self._publish_event_name, payload)
            return

        if hasattr(self._event_bus, "emit"):
            await self._event_bus.emit(self._publish_event_name, payload)
            return

        self._logger.warning(
            "EventBus has no publish/emit method; large trade signal not dispatched",
            extra={"publish_event_name": self._publish_event_name, "payload": payload},
        )

    # -------------------------------------------------------------------------
    # monitoring / stats
    # -------------------------------------------------------------------------

    def get_symbol_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        stats = self._stats.get(symbol.upper())
        if stats is None:
            return None
        return asdict(stats)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        return {symbol: asdict(stats) for symbol, stats in self._stats.items()}

    def reset_symbol(self, symbol: str) -> None:
        symbol = symbol.upper()
        self._notional_history.pop(symbol, None)
        self._stats.pop(symbol, None)
        self._last_signal_ts_by_symbol.pop(symbol, None)

        self._logger.info(
            "LargeTradeDetector symbol state reset",
            extra={"symbol": symbol},
        )

    def reset_all(self) -> None:
        self._notional_history.clear()
        self._stats.clear()
        self._last_signal_ts_by_symbol.clear()

        self._logger.info("LargeTradeDetector state reset")

    # -------------------------------------------------------------------------
    # cleanup
    # -------------------------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self._cleanup_interval_sec)

                async with self._lock:
                    now_ts = time.time()
                    stale_symbols = []

                    for symbol, stats in self._stats.items():
                        if stats.last_trade_ts <= 0:
                            continue
                        if (now_ts - stats.last_trade_ts) > self._history_retention_sec:
                            stale_symbols.append(symbol)

                    for symbol in stale_symbols:
                        self._notional_history.pop(symbol, None)
                        self._last_signal_ts_by_symbol.pop(symbol, None)

                self._logger.debug("LargeTradeDetector cleanup completed")

        except asyncio.CancelledError:
            self._logger.debug("LargeTradeDetector cleanup task cancelled")
            raise
        except Exception as exc:
            self._logger.exception(
                "LargeTradeDetector cleanup loop failed",
                extra={"error": str(exc)},
            )