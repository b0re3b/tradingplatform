from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from analytics.whales.base import BaseWhaleComponent
from analytics.whales.config import LargeTradeDetectorConfig
from analytics.whales.enums import LargeTradeTriggerType, WhaleTradeSide
from analytics.whales.models import (
    LargeTradeSignal,
    SymbolStats,
    TradeRecord,
    make_symbol_stats,
)


class LargeTradeDetector(BaseWhaleComponent):
    """
    Low-level detector для аномально великих трейдів.

    Призначення:
        - приймати raw trade events
        - нормалізувати їх у TradeRecord
        - підтримувати rolling статистику notional по символу
        - виявляти large trade через:
            1) absolute threshold
            2) relative z-score threshold
        - публікувати сигнал у EventBus

    Підтримує два режими використання:
        1. Event-driven:
           start() -> subscribe на market.trade -> auto processing
        2. Direct:
           await process_trade(event)

    Зауваження:
        stop() відписує detector від EventBus та завершує cleanup loop,
        але не гарантує explicit drain усіх in-flight process_trade().
    """

    def __init__(
        self,
        config: Optional[LargeTradeDetectorConfig] = None,
        event_bus: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        super().__init__(
            component_name="large_trade_detector",
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.config = config or LargeTradeDetectorConfig()

        self._stats: Dict[str, SymbolStats] = {}
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task[Any]] = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        if self._started:
            self.logger.warning("LargeTradeDetector already started")
            return

        if not self.config.enabled:
            self.logger.info("LargeTradeDetector is disabled by config")
            return

        if self.event_bus is not None:
            await self._safe_subscribe(
                self.config.input_event_name,
                self.handle_event,
            )

        if self.scheduler is not None:
            await self._register_interval_job(
                name="whales_large_trade_detector_cleanup",
                interval_seconds=self.config.cleanup_interval_sec,
                coro=self.cleanup,
                replace_existing=True,
            )
        else:
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(),
                name="whales_large_trade_detector_cleanup_loop",
            )

        self._started = True

        self.logger.info(
            "LargeTradeDetector started",
            extra={
                "input_event_name": self.config.input_event_name,
                "output_event_name": self.config.output_event_name,
                "rolling_window_size": self.config.rolling_window_size,
                "zscore_threshold": self.config.zscore_threshold,
                "default_abs_notional_threshold": self.config.default_abs_notional_threshold,
                "recalibration_interval": self.config.recalibration_interval,
            },
        )

    async def stop(self) -> None:
        if not self._started:
            return

        if self.event_bus is not None:
            await self._safe_unsubscribe(
                self.config.input_event_name,
                self.handle_event,
            )

        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

        self._started = False
        self.logger.info("LargeTradeDetector stopped")

    # =========================================================================
    # Public event handling
    # =========================================================================

    async def handle_event(self, event: Dict[str, Any]) -> None:
        try:
            await self.process_trade(event)
        except Exception:
            self.logger.exception("Unhandled error while processing trade event")

    async def process_trade(self, event: Dict[str, Any]) -> Optional[LargeTradeSignal]:
        """
        Основний публічний метод обробки raw trade event.

        Використовує per-symbol lock, щоб не блокувати паралельну обробку
        різних символів одним глобальним lock.
        """
        if not self.config.enabled:
            return None

        trade = self._normalize_trade(event)
        if trade is None:
            return None

        if not self._passes_basic_filters(trade):
            return None

        stats, symbol_lock = await self._get_or_create_symbol_state(trade.symbol)

        signal: Optional[LargeTradeSignal] = None

        async with symbol_lock:
            mean_before = stats.mean()
            std_before = stats.std()

            abs_threshold = self._get_abs_threshold(trade.symbol)
            zscore = self._calculate_zscore(
                value=trade.notional,
                mean=mean_before,
                std=std_before,
            )

            abs_trigger = trade.notional >= abs_threshold
            rel_trigger = self._is_relative_trigger(
                zscore=zscore,
                sample_size=stats.sample_size,
            )

            if abs_trigger or rel_trigger:
                if self._passes_cooldown(stats, trade.symbol):
                    signal = self._build_signal(
                        trade=trade,
                        abs_threshold=abs_threshold,
                        mean_notional=mean_before,
                        std_notional=std_before,
                        zscore=zscore,
                        abs_trigger=abs_trigger,
                        rel_trigger=rel_trigger,
                    )
                    stats.signals_emitted += 1
                    stats.last_signal_ts_monotonic = time.monotonic()

            stats.add(
                trade.notional,
                recalibration_interval=self.config.recalibration_interval,
            )
            stats.trades_processed += 1

        if signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Large trade detected",
                    extra={
                        "symbol": signal.symbol,
                        "side": signal.side,
                        "notional": signal.notional,
                        "zscore": signal.zscore,
                        "trigger_type": signal.trigger_type,
                        "trade_id": signal.trade_id,
                        "exchange": signal.exchange,
                    },
                )

            await self._emit_signal(signal)

        return signal

    # =========================================================================
    # Core detection logic
    # =========================================================================

    def _normalize_trade(self, event: Dict[str, Any]) -> Optional[TradeRecord]:
        """
        Нормалізація raw event payload у TradeRecord.

        Підтримує різні поширені схеми payload:
            - event["data"] / plain event
            - symbol / s / instrument
            - price / p
            - quantity / qty / q / size
            - side / S / maker_side / direction / m
            - timestamp / ts / T / E
        """
        try:
            payload = event.get("data", event)

            symbol = self._normalize_symbol(
                payload.get("symbol")
                or payload.get("s")
                or payload.get("instrument")
            )
            if not symbol:
                self.logger.debug(
                    "Trade event dropped: missing symbol",
                    extra={"event": event},
                )
                return None

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
            side = self._normalize_side(
                payload.get("side")
                or payload.get("S")
                or payload.get("maker_side")
                or payload.get("direction"),
                maker_flag=payload.get("m"),
            )
            timestamp_ms = self._extract_timestamp_ms(payload)

            if price is None or price <= 0:
                self.logger.debug(
                    "Trade event dropped: invalid price",
                    extra={"price": price, "event": event},
                )
                return None

            if quantity is None or quantity <= 0:
                self.logger.debug(
                    "Trade event dropped: invalid quantity",
                    extra={"quantity": quantity, "event": event},
                )
                return None

            if side == WhaleTradeSide.UNKNOWN.value:
                self.logger.debug(
                    "Trade event dropped: invalid side",
                    extra={"event": event},
                )
                return None

            trade_id = self._safe_str(
                payload.get("trade_id")
                or payload.get("id")
                or payload.get("t")
            )
            exchange = self._safe_str(
                payload.get("exchange")
                or event.get("exchange")
            )

            return TradeRecord(
                symbol=symbol,
                price=price,
                quantity=quantity,
                side=side,
                timestamp_ms=timestamp_ms,
                trade_id=trade_id,
                exchange=exchange,
                raw_event=event,
            )
        except Exception:
            self.logger.exception(
                "Failed to normalize trade event",
                extra={"event": event},
            )
            return None

    def _passes_basic_filters(self, trade: TradeRecord) -> bool:
        if trade.notional < self.config.min_notional_filter:
            return False

        if self.config.side_filter is not None and trade.side != self.config.side_filter:
            return False

        return True

    def _calculate_zscore(
        self,
        *,
        value: float,
        mean: float,
        std: float,
    ) -> float:
        if std <= 0:
            return 0.0
        return (value - mean) / std

    def _is_relative_trigger(
        self,
        *,
        zscore: float,
        sample_size: int,
    ) -> bool:
        if not self.config.use_relative_detection:
            return False

        if sample_size < self.config.min_samples_for_relative_detection:
            return False

        return zscore >= self.config.zscore_threshold

    def _get_abs_threshold(self, symbol: str) -> float:
        return self.config.get_symbol_abs_threshold(symbol)

    def _passes_cooldown(self, stats: SymbolStats, symbol: str) -> bool:
        cooldown_sec = self.config.get_symbol_cooldown(symbol)
        if cooldown_sec <= 0:
            return True

        elapsed = time.monotonic() - stats.last_signal_ts_monotonic
        return elapsed >= cooldown_sec

    def _build_signal(
        self,
        *,
        trade: TradeRecord,
        abs_threshold: float,
        mean_notional: float,
        std_notional: float,
        zscore: float,
        abs_trigger: bool,
        rel_trigger: bool,
    ) -> LargeTradeSignal:
        if abs_trigger and rel_trigger:
            trigger_type = LargeTradeTriggerType.ABSOLUTE_AND_RELATIVE.value
        elif abs_trigger:
            trigger_type = LargeTradeTriggerType.ABSOLUTE.value
        elif rel_trigger:
            trigger_type = LargeTradeTriggerType.RELATIVE.value
        else:
            trigger_type = LargeTradeTriggerType.UNKNOWN.value

        return LargeTradeSignal(
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
            trigger_type=trigger_type,
            trade_id=trade.trade_id,
            exchange=trade.exchange,
        )

    async def _emit_signal(self, signal: LargeTradeSignal) -> None:
        if not self.config.emit_on_bus:
            return

        await self._safe_emit(
            self.config.output_event_name,
            signal.to_event(),
            source="analytics.whales.large_trade_detector",
        )

    # =========================================================================
    # Symbol state management
    # =========================================================================

    async def _get_or_create_symbol_state(
        self,
        symbol: str,
    ) -> Tuple[SymbolStats, asyncio.Lock]:
        stats = self._stats.get(symbol)
        lock = self._symbol_locks.get(symbol)

        if stats is not None and lock is not None:
            return stats, lock

        async with self._registry_lock:
            stats = self._stats.get(symbol)
            if stats is None:
                stats = make_symbol_stats(self.config.rolling_window_size)
                self._stats[symbol] = stats

            lock = self._symbol_locks.get(symbol)
            if lock is None:
                lock = asyncio.Lock()
                self._symbol_locks[symbol] = lock

            return stats, lock

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup(self) -> None:
        """
        Видаляє неактивні symbol states.
        """
        now_mono = time.monotonic()
        ttl = self.config.stats_ttl_sec

        if ttl <= 0:
            return

        async with self._registry_lock:
            stale_symbols = [
                symbol
                for symbol, stats in self._stats.items()
                if (now_mono - stats.last_update_ts_monotonic) >= ttl
            ]

            for symbol in stale_symbols:
                self._stats.pop(symbol, None)
                self._symbol_locks.pop(symbol, None)

        if stale_symbols:
            self.logger.info(
                "Cleaned stale LargeTradeDetector symbol states",
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
            self.logger.exception("Unhandled error in LargeTradeDetector cleanup loop")

    # =========================================================================
    # Public state / stats API
    # =========================================================================

    def get_symbol_stats(self, symbol: str) -> Dict[str, Any]:
        stats = self._stats.get(symbol)
        if stats is None:
            return {
                "symbol": symbol,
                "exists": False,
            }

        return {
            "symbol": symbol,
            "exists": True,
            **stats.to_dict(),
        }

    def get_all_stats(self) -> Dict[str, Any]:
        return {
            symbol: stats.to_dict()
            for symbol, stats in self._stats.items()
        }

    async def reset_symbol(self, symbol: str) -> None:
        async with self._registry_lock:
            self._stats.pop(symbol, None)
            self._symbol_locks.pop(symbol, None)

        self.logger.info(
            "Reset LargeTradeDetector symbol state",
            extra={"symbol": symbol},
        )

    async def reset_all(self) -> None:
        async with self._registry_lock:
            self._stats.clear()
            self._symbol_locks.clear()

        self.logger.info("Reset all LargeTradeDetector states")

    # =========================================================================
    # Parsing / normalization helpers
    # =========================================================================

    def _normalize_symbol(self, value: Any) -> Optional[str]:
        symbol = self._safe_str(value)
        if symbol is None:
            return None
        return symbol.upper()

    def _normalize_side(
        self,
        value: Any,
        maker_flag: Any = None,
    ) -> str:
        """
        Нормалізація side.

        Підтримка:
            - "buy"/"sell"
            - "bid"/"ask"
            - "long"/"short" -> buy/sell
            - Binance aggTrade поле `m`:
                m == False -> buy aggressor
                m == True  -> sell aggressor
        """
        if isinstance(value, str):
            side = value.strip().lower()

            if side in {"buy", "bid", "long"}:
                return WhaleTradeSide.BUY.value
            if side in {"sell", "ask", "short"}:
                return WhaleTradeSide.SELL.value

        if maker_flag is not None:
            if maker_flag is False:
                return WhaleTradeSide.BUY.value
            if maker_flag is True:
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
            # якщо seconds, а не ms
            if raw_ts < 10_000_000_000:
                return int(raw_ts * 1000)
            return int(raw_ts)

        if isinstance(raw_ts, str):
            raw_ts = raw_ts.strip()

            # ISO datetime
            try:
                dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except Exception:
                pass

            # numeric string
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
            if result != result:  # nan
                return None
            return result
        except (TypeError, ValueError):
            return None

    def _safe_str(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None