from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler

from analytics.whales.base import BaseWhaleComponent
from analytics.whales.config import LargeTradeDetectorConfig
from analytics.whales.enums import WhaleComponentName, WhaleTradeSide
from analytics.whales.models import (
    LargeTradeSignal,
    SymbolStats,
    TradeRecord,
    make_symbol_stats,
)


class LargeTradeDetector(BaseWhaleComponent):
    """
    Low-level detector для аномально великих трейдів.

    Event-driven режим:
        EventBus topic:
            market.trade
        handler:
            handle_trade_event(event: Event)
        output:
            analytics.whales.large_trade

    Direct режим для тестів/backtesting/replay:
        await process_trade_payload(payload)

    Важливо:
    - EventBus/Scheduler передаються через constructor dependency injection;
    - підписки виконуються через register() / EventBus.subscribe();
    - cleanup запускається тільки через Scheduler.add_interval_job();
    - власних uncontrolled asyncio cleanup loops немає.
    """

    def __init__(
        self,
        *,
        config: LargeTradeDetectorConfig,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> None:
        super().__init__(
            component_name=WhaleComponentName.LARGE_TRADE_DETECTOR.value,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        self.config = config
        self.config.validate()

        self._stats: dict[str, SymbolStats] = {}
        self._symbol_locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def register(self) -> None:
        """
        Зареєструвати EventBus subscriptions.

        Idempotent: повторний виклик не створює дублікати підписок.
        """
        if self._registered:
            return

        if not self.config.enabled:
            self.logger.info(
                "LargeTradeDetector registration skipped: disabled by config",
                extra={"component": self.component_name},
            )
            return

        self._subscribe(
            self.config.input_event_name,
            self.handle_trade_event,
            name="analytics.whales.large_trade_detector.handle_trade_event",
        )

        self._registered = True

    async def start(self) -> None:
        if self._started:
            self.logger.warning("LargeTradeDetector already started")
            return

        if not self.config.enabled:
            self.logger.info("LargeTradeDetector is disabled by config")
            return

        await self.register()

        self._add_interval_job(
            name="analytics.whales.large_trade_detector.cleanup",
            func=self.cleanup,
            interval=self.config.cleanup_interval_sec,
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=min(30.0, max(1.0, self.config.cleanup_interval_sec)),
            allow_overlap=False,
            enabled=True,
        )

        self._started = True

        self.logger.info(
            "LargeTradeDetector started",
            extra={
                "component": self.component_name,
                "input_event_name": self.config.input_event_name,
                "output_event_name": self.config.output_event_name,
                "rolling_window_size": self.config.rolling_window_size,
                "zscore_threshold": self.config.zscore_threshold,
                "default_abs_notional_threshold": (
                    self.config.default_abs_notional_threshold
                ),
                "recalibration_interval": self.config.recalibration_interval,
                "cleanup_interval_sec": self.config.cleanup_interval_sec,
            },
        )

    async def stop(self) -> None:
        if not self._started and not self._registered:
            return

        self._remove_scheduler_jobs()
        await super().stop()

        self.logger.info(
            "LargeTradeDetector stopped",
            extra={"component": self.component_name},
        )

    # =========================================================================
    # EventBus handlers
    # =========================================================================

    async def handle_trade_event(self, event: Event) -> None:
        """
        EventBus handler.

        Core EventBus передає core.event_bus.Event, а бізнес-логіка нижче
        працює з dict payload.
        """
        try:
            payload = self._payload_from_event(event)

            await self.process_trade_payload(
                payload,
                correlation_id=event.correlation_id,
                source_event_id=event.event_id,
                source_topic=event.topic,
            )

        except Exception:
            self.logger.exception(
                "Unhandled error while processing trade event",
                extra={
                    "component": self.component_name,
                    "topic": event.topic,
                    "event_id": event.event_id,
                    "source": event.source,
                    "correlation_id": event.correlation_id,
                },
            )

    # =========================================================================
    # Public processing API
    # =========================================================================

    async def process_trade_payload(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
        source_topic: str | None = None,
    ) -> LargeTradeSignal | None:
        """
        Основний метод обробки raw trade payload.

        Використовується:
        - EventBus handler-ом;
        - тестами;
        - backtesting/replay.
        """
        if not self.config.enabled:
            return None

        trade = self._normalize_trade_payload(payload)
        if trade is None:
            return None

        if not self._passes_basic_filters(trade):
            return None

        stats, symbol_lock = await self._get_or_create_symbol_state(trade.symbol)

        signal: LargeTradeSignal | None = None

        async with symbol_lock:
            mean_before = stats.mean()
            std_before = stats.std()

            abs_threshold = self._get_abs_threshold(trade.symbol)
            zscore = self._calculate_zscore(
                value=trade.notional,
                mean=mean_before,
                std=std_before,
            )

            absolute_triggered = trade.notional >= abs_threshold
            relative_triggered = self._is_relative_trigger(
                zscore=zscore,
                sample_size=stats.sample_size,
            )

            if absolute_triggered or relative_triggered:
                if self._passes_symbol_cooldown(stats, trade.symbol):
                    signal = LargeTradeSignal.from_trade(
                        trade=trade,
                        abs_threshold=abs_threshold,
                        mean_notional=mean_before,
                        std_notional=std_before,
                        zscore=zscore,
                        absolute_triggered=absolute_triggered,
                        relative_triggered=relative_triggered,
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
                        "component": self.component_name,
                        "symbol": signal.symbol,
                        "side": signal.side,
                        "notional": signal.notional,
                        "zscore": signal.zscore,
                        "trigger_type": signal.trigger_type,
                        "trade_id": signal.trade_id,
                        "exchange": signal.exchange,
                        "source_topic": source_topic,
                        "source_event_id": source_event_id,
                    },
                )

            await self._emit_signal(
                signal,
                correlation_id=correlation_id,
                source_event_id=source_event_id,
            )

        return signal

    async def process_trade(
        self,
        event: Mapping[str, Any] | dict[str, Any],
    ) -> LargeTradeSignal | None:
        """
        Backward-compatible alias для старого direct API.

        Новий код має використовувати process_trade_payload().
        """
        return await self.process_trade_payload(event)

    # =========================================================================
    # Core detection logic
    # =========================================================================

    def _normalize_trade_payload(
        self,
        event_payload: Mapping[str, Any] | dict[str, Any],
    ) -> TradeRecord | None:
        """
        Нормалізація raw market.trade payload у TradeRecord.

        Підтримує схеми:
        - payload["data"] / plain payload;
        - symbol / s / instrument;
        - price / p;
        - quantity / qty / q / size;
        - side / S / maker_side / direction / m;
        - timestamp_ms / timestamp / ts / T / E.
        """
        try:
            event = dict(event_payload)
            raw_payload = event.get("data", event)

            if not isinstance(raw_payload, Mapping):
                self.logger.debug(
                    "Trade event dropped: payload data is not mapping",
                    extra={
                        "component": self.component_name,
                        "payload_type": type(raw_payload).__name__,
                    },
                )
                return None

            payload = dict(raw_payload)

            symbol = self._normalize_symbol(
                payload.get("symbol")
                or payload.get("s")
                or payload.get("instrument")
            )
            if not symbol:
                self.logger.debug(
                    "Trade event dropped: missing symbol",
                    extra={"component": self.component_name},
                )
                return None

            price = self._safe_float(payload.get("price") or payload.get("p"))
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
                    extra={
                        "component": self.component_name,
                        "symbol": symbol,
                        "price": price,
                    },
                )
                return None

            if quantity is None or quantity <= 0:
                self.logger.debug(
                    "Trade event dropped: invalid quantity",
                    extra={
                        "component": self.component_name,
                        "symbol": symbol,
                        "quantity": quantity,
                    },
                )
                return None

            if side == WhaleTradeSide.UNKNOWN.value:
                self.logger.debug(
                    "Trade event dropped: invalid side",
                    extra={
                        "component": self.component_name,
                        "symbol": symbol,
                    },
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
                "Failed to normalize trade payload",
                extra={"component": self.component_name},
            )
            return None

    def _passes_basic_filters(self, trade: TradeRecord) -> bool:
        if trade.notional < self.config.min_notional_filter:
            return False

        if self.config.side_filter is not None and trade.side != self.config.side_filter:
            return False

        return True

    @staticmethod
    def _calculate_zscore(
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

    def _passes_symbol_cooldown(self, stats: SymbolStats, symbol: str) -> bool:
        return self._passes_cooldown(
            stats.last_signal_ts_monotonic,
            self.config.get_symbol_cooldown(symbol),
        )

    async def _emit_signal(
        self,
        signal: LargeTradeSignal,
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
    ) -> None:
        if not self.config.emit_on_bus:
            return

        headers: dict[str, Any] = {}
        if source_event_id is not None:
            headers["source_event_id"] = source_event_id

        await self._emit(
            self.config.output_event_name,
            signal.to_payload(),
            priority=EventPriority.NORMAL,
            source=self.component_name,
            correlation_id=correlation_id,
            headers=headers or None,
        )

    # =========================================================================
    # Symbol state management
    # =========================================================================

    async def _get_or_create_symbol_state(
        self,
        symbol: str,
    ) -> tuple[SymbolStats, asyncio.Lock]:
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

        Запускається через core Scheduler.add_interval_job().
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
                    "component": self.component_name,
                    "removed_symbols_count": len(stale_symbols),
                },
            )

    # =========================================================================
    # Public state / stats API
    # =========================================================================

    def get_symbol_stats(self, symbol: str) -> dict[str, Any]:
        normalized_symbol = self._normalize_symbol(symbol)
        if normalized_symbol is None:
            return {
                "symbol": symbol,
                "exists": False,
                "error": "invalid_symbol",
            }

        stats = self._stats.get(normalized_symbol)
        if stats is None:
            return {
                "symbol": normalized_symbol,
                "exists": False,
            }

        return {
            "symbol": normalized_symbol,
            "exists": True,
            **stats.to_dict(),
        }

    def get_all_stats(self) -> dict[str, Any]:
        return {
            symbol: stats.to_dict()
            for symbol, stats in self._stats.items()
        }

    async def reset_symbol(self, symbol: str) -> None:
        normalized_symbol = self._normalize_symbol(symbol)
        if normalized_symbol is None:
            return

        async with self._registry_lock:
            self._stats.pop(normalized_symbol, None)
            self._symbol_locks.pop(normalized_symbol, None)

        self.logger.info(
            "Reset LargeTradeDetector symbol state",
            extra={
                "component": self.component_name,
                "symbol": normalized_symbol,
            },
        )

    async def reset_all(self) -> None:
        async with self._registry_lock:
            self._stats.clear()
            self._symbol_locks.clear()

        self.logger.info(
            "Reset all LargeTradeDetector states",
            extra={"component": self.component_name},
        )

    def get_healthcheck(self) -> dict[str, Any]:
        health = super().get_healthcheck()
        health.update(
            {
                "enabled": self.config.enabled,
                "tracked_symbols": len(self._stats),
                "input_event_name": self.config.input_event_name,
                "output_event_name": self.config.output_event_name,
            }
        )
        return health

    # =========================================================================
    # Parsing / normalization helpers
    # =========================================================================

    def _normalize_symbol(self, value: Any) -> str | None:
        symbol = self._safe_str(value)
        if symbol is None:
            return None
        return symbol.upper()

    @staticmethod
    def _normalize_side(
        value: Any,
        maker_flag: Any = None,
    ) -> str:
        side = WhaleTradeSide.normalize(value)

        if side is not WhaleTradeSide.UNKNOWN:
            return side.value

        return WhaleTradeSide.from_maker_flag(maker_flag).value

    @staticmethod
    def _extract_timestamp_ms(payload: Mapping[str, Any]) -> int:
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
            # seconds, not milliseconds
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

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None

        try:
            result = float(value)
        except (TypeError, ValueError):
            return None

        if result != result:  # NaN
            return None

        return result

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        if value is None:
            return None

        text = str(value).strip()
        return text or None


__all__ = [
    "LargeTradeDetector",
]