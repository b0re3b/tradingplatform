from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional

from core.logger import get_logger


@dataclass(slots=True)
class LargeTradeDetectorConfig:
    """
    Конфігурація детектора великих трейдів.

    Absolute trigger:
        Трейд вважається великим, якщо його notional >= absolute threshold.

    Relative trigger:
        Трейд також може бути визначений як великий відносно rolling
        distribution notional-значень по символу.

    Детекція спрацьовує, якщо:
        1) notional >= absolute threshold
        або
        2) z-score >= zscore threshold

    Примітка:
        Relative detection тут навмисно орієнтована лише на аномально ВЕЛИКІ
        трейди, тому перевіряється лише позитивний z-score.
        Якщо в майбутньому знадобиться виявлення аномально малих трейдів,
        цю логіку треба буде розширити окремо.
    """

    enabled: bool = True

    # Базові пороги
    default_abs_notional_threshold: float = 100_000.0
    symbol_abs_thresholds: Dict[str, float] = field(default_factory=dict)

    # Relative / statistical detection
    use_relative_detection: bool = True
    rolling_window_size: int = 300
    min_samples_for_relative_detection: int = 30
    zscore_threshold: float = 3.0

    # Грубі фільтри
    min_notional_filter: float = 10_000.0
    side_filter: Optional[str] = None  # "buy", "sell", None

    # Cooldown між сигналами по одному символу
    signal_cooldown_sec: float = 2.0
    symbol_cooldown_sec: Dict[str, float] = field(default_factory=dict)

    # Housekeeping
    cleanup_interval_sec: int = 60
    stats_ttl_sec: int = 60 * 60

    # Recalibration для боротьби з накопиченням floating-point drift
    recalibration_interval: int = 2_000

    # Event names
    input_event_name: str = "market.trade"
    output_event_name: str = "analytics.whales.large_trade"

    # Logging / metrics
    log_signals: bool = True
    emit_on_bus: bool = True


@dataclass(slots=True)
class TradeRecord:
    """
    Нормалізована модель трейду.
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
class LargeTradeSignal:
    """
    Сигнал про виявлення великого трейду.
    """

    symbol: str
    side: str
    price: float
    quantity: float
    notional: float
    timestamp_ms: int

    abs_threshold: float
    mean_notional: float
    std_notional: float
    zscore: float

    trigger_type: str  # absolute / relative / absolute_and_relative
    trade_id: Optional[str] = None
    exchange: Optional[str] = None

    detector_name: str = "LargeTradeDetector"
    event_type: str = "large_trade"
    schema_version: int = 1
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_event(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "detector": self.detector_name,
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
            "created_at_ms": self.created_at_ms,
        }


@dataclass(slots=True)
class SymbolStats:
    """
    Rolling-статистика по символу.

    Реалізація використовує:
        - deque для rolling window
        - running_sum / running_sum_sq для O(1) mean/std
        - періодичний _recalibrate() для боротьби з floating-point drift
          при довгій роботі та постійних eviction у deque

    Примітка:
        Тут використовується sample std (n - 1), а не population std.
        Для rolling z-score це зазвичай більш коректно, особливо на відносно
        невеликих вибірках.
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
        """
        Повний перерахунок running_sum / running_sum_sq з deque.

        Це дешевий спосіб прибрати накопичений floating-point drift
        без суттєвого ускладнення sliding-window логіки.
        """
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
        """
        Sample std (n - 1).
        """
        n = len(self.notionals)
        if n < 2:
            return 0.0

        mean_value = self.running_sum / n
        numerator = self.running_sum_sq - (n * mean_value * mean_value)
        numerator = max(numerator, 0.0)

        variance = numerator / (n - 1)
        return math.sqrt(max(variance, 0.0))


class LargeTradeDetector:
    """
    Low-level detector для аномально великих трейдів.

    Призначення:
        - приймати сирі trade events
        - нормалізувати їх
        - рахувати rolling статистику notional
        - виявляти великі трейди через:
            * absolute threshold
            * relative z-score threshold
        - публікувати сигнал у EventBus

    Очікування по EventBus:
        Потрібно, щоб bus мав методи на кшталт:
            await event_bus.subscribe(event_name, handler)
            await event_bus.unsubscribe(event_name, handler)
            await event_bus.emit(event_name, payload)

    Примітка щодо stop():
        stop() відписує detector від EventBus і завершує cleanup loop,
        але навмисно не робить explicit drain усіх in-flight process_trade().
        Тобто корутини, які вже зайшли всередину handle_event/process_trade,
        можуть завершитися після виклику stop(). Це архітектурне обмеження
        типового async EventBus без окремого drain/shutdown protocol.
    """

    def __init__(
        self,
        config: Optional[LargeTradeDetectorConfig] = None,
        event_bus: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        self.config = config or LargeTradeDetectorConfig()
        self.event_bus = event_bus
        self.scheduler = scheduler

        self.logger = get_logger(
            __name__,
            service_name="analytics.whales.large_trade_detector",
        )

        self._stats: Dict[str, SymbolStats] = {}
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

        self._started = False
        self._cleanup_task: Optional[asyncio.Task[Any]] = None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            self.logger.warning("LargeTradeDetector already started")
            return

        if not self.config.enabled:
            self.logger.info("LargeTradeDetector is disabled by config")
            return

        self._started = True

        if self.event_bus is not None:
            await self._safe_subscribe()

        if self.scheduler is not None:
            await self._register_scheduler_jobs()
        else:
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(),
                name="large_trade_detector_cleanup_loop",
            )

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
            await self._safe_unsubscribe()

        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

        self._started = False
        self.logger.info("LargeTradeDetector stopped")

    async def _safe_subscribe(self) -> None:
        try:
            await self.event_bus.subscribe(
                self.config.input_event_name,
                self.handle_event,
            )
            self.logger.info(
                "Subscribed to EventBus",
                extra={"event_name": self.config.input_event_name},
            )
        except Exception:
            self.logger.exception(
                "Failed to subscribe LargeTradeDetector to EventBus",
                extra={"event_name": self.config.input_event_name},
            )
            raise

    async def _safe_unsubscribe(self) -> None:
        try:
            await self.event_bus.unsubscribe(
                self.config.input_event_name,
                self.handle_event,
            )
            self.logger.info(
                "Unsubscribed from EventBus",
                extra={"event_name": self.config.input_event_name},
            )
        except Exception:
            self.logger.exception(
                "Failed to unsubscribe LargeTradeDetector from EventBus",
                extra={"event_name": self.config.input_event_name},
            )

    async def _register_scheduler_jobs(self) -> None:
        try:
            await self.scheduler.add_interval_job(
                name="large_trade_detector_cleanup",
                interval_seconds=self.config.cleanup_interval_sec,
                coro=self.cleanup,
                replace_existing=True,
            )
            self.logger.info(
                "Cleanup job registered in Scheduler",
                extra={"interval_sec": self.config.cleanup_interval_sec},
            )
        except Exception:
            self.logger.exception("Failed to register cleanup job in Scheduler")
            raise

    # -------------------------------------------------------------------------
    # Event handling
    # -------------------------------------------------------------------------

    async def handle_event(self, event: Dict[str, Any]) -> None:
        try:
            await self.process_trade(event)
        except Exception:
            self.logger.exception("Unhandled error while processing trade event")

    async def process_trade(self, event: Dict[str, Any]) -> Optional[LargeTradeSignal]:
        """
        Основний вхід для обробки трейду.

        Реалізація використовує per-symbol lock замість одного глобального,
        щоб уникати зайвої contention між різними символами.
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

    # -------------------------------------------------------------------------
    # Core detection logic
    # -------------------------------------------------------------------------

    def _normalize_trade(self, event: Dict[str, Any]) -> Optional[TradeRecord]:
        """
        Нормалізація різних форм event payload у TradeRecord.

        Якщо payload відкидається, це логуються на debug-рівні з причиною,
        щоб уникнути мовчазних дропів через parsing / data quality issues.
        """
        try:
            payload = event.get("data", event)

            symbol = self._normalize_symbol(
                payload.get("symbol")
                or payload.get("s")
                or payload.get("instrument")
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
            side = self._normalize_side(
                payload.get("side")
                or payload.get("S")
                or payload.get("maker_side")
                or payload.get("direction")
            )
            timestamp_ms = self._extract_timestamp_ms(payload)
            trade_id = self._safe_str(
                payload.get("trade_id")
                or payload.get("id")
                or payload.get("t")
            )
            exchange = self._safe_str(
                payload.get("exchange")
                or event.get("exchange")
            )

            if not symbol:
                self.logger.debug(
                    "Dropping trade event: missing symbol",
                    extra={"payload": payload},
                )
                return None

            if price <= 0:
                self.logger.debug(
                    "Dropping trade event: invalid price",
                    extra={"symbol": symbol, "price": price, "payload": payload},
                )
                return None

            if quantity <= 0:
                self.logger.debug(
                    "Dropping trade event: invalid quantity",
                    extra={"symbol": symbol, "quantity": quantity, "payload": payload},
                )
                return None

            if not side:
                self.logger.debug(
                    "Dropping trade event: invalid side",
                    extra={"symbol": symbol, "payload": payload},
                )
                return None

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
            self.logger.exception("Failed to normalize trade event")
            return None

    def _passes_basic_filters(self, trade: TradeRecord) -> bool:
        if trade.notional < self.config.min_notional_filter:
            return False

        if self.config.side_filter is not None and trade.side != self.config.side_filter:
            return False

        return True

    async def _get_or_create_symbol_state(self, symbol: str) -> tuple[SymbolStats, asyncio.Lock]:
        """
        Повертає пару (stats, symbol_lock) для символу.

        Використовується double-checked creation:
            - об'єкти алокуються поза registry lock
            - під lock лише перевірка/вставка
        """
        existing_stats = self._stats.get(symbol)
        existing_lock = self._symbol_locks.get(symbol)
        if existing_stats is not None and existing_lock is not None:
            return existing_stats, existing_lock

        new_stats = SymbolStats(
            notionals=deque(maxlen=self.config.rolling_window_size),
        )
        new_lock = asyncio.Lock()

        async with self._registry_lock:
            stats = self._stats.get(symbol)
            if stats is None:
                stats = new_stats
                self._stats[symbol] = stats

            symbol_lock = self._symbol_locks.get(symbol)
            if symbol_lock is None:
                symbol_lock = new_lock
                self._symbol_locks[symbol] = symbol_lock

            return stats, symbol_lock

    def _get_abs_threshold(self, symbol: str) -> float:
        return self.config.symbol_abs_thresholds.get(
            symbol,
            self.config.default_abs_notional_threshold,
        )

    def _get_cooldown_sec(self, symbol: str) -> float:
        return self.config.symbol_cooldown_sec.get(
            symbol,
            self.config.signal_cooldown_sec,
        )

    def _calculate_zscore(self, value: float, mean: float, std: float) -> float:
        if std <= 0:
            return 0.0
        return (value - mean) / std

    def _is_relative_trigger(self, zscore: float, sample_size: int) -> bool:
        """
        Навмисно перевіряється лише позитивний z-score, бо detector
        орієнтований на unusually large trades, а не на unusually small trades.
        """
        if not self.config.use_relative_detection:
            return False
        if sample_size < self.config.min_samples_for_relative_detection:
            return False
        return zscore >= self.config.zscore_threshold

    def _passes_cooldown(self, stats: SymbolStats, symbol: str) -> bool:
        now = time.monotonic()
        cooldown = self._get_cooldown_sec(symbol)
        return (now - stats.last_signal_ts_monotonic) >= cooldown

    def _build_signal(
        self,
        trade: TradeRecord,
        abs_threshold: float,
        mean_notional: float,
        std_notional: float,
        zscore: float,
        abs_trigger: bool,
        rel_trigger: bool,
    ) -> LargeTradeSignal:
        if abs_trigger and rel_trigger:
            trigger_type = "absolute_and_relative"
        elif abs_trigger:
            trigger_type = "absolute"
        else:
            trigger_type = "relative"

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

    # -------------------------------------------------------------------------
    # Emission
    # -------------------------------------------------------------------------

    async def _emit_signal(self, signal: LargeTradeSignal) -> None:
        if not self.config.emit_on_bus or self.event_bus is None:
            return

        payload = signal.to_event()

        try:
            await self.event_bus.emit(self.config.output_event_name, payload)
        except Exception:
            self.logger.exception(
                "Failed to emit large trade signal",
                extra={
                    "event_name": self.config.output_event_name,
                    "symbol": signal.symbol,
                    "trade_id": signal.trade_id,
                },
            )

    # -------------------------------------------------------------------------
    # Housekeeping
    # -------------------------------------------------------------------------

    async def cleanup(self) -> None:
        """
        Видалення протухлих symbol stats.

        Примітка:
            `now` спеціально береться ДО входу в lock. Це нормально, бо нам
            потрібен лише консистентний monotonic snapshot для TTL-перевірки,
            а не ідеально свіжий час на кожному symbol.
        """
        ttl = self.config.stats_ttl_sec
        now = time.monotonic()
        removed_symbols: list[str] = []

        async with self._registry_lock:
            items = [(symbol, stats, self._symbol_locks.get(symbol)) for symbol, stats in self._stats.items()]

        for symbol, stats, symbol_lock in items:
            if symbol_lock is None:
                continue

            async with symbol_lock:
                if (now - stats.last_update_ts_monotonic) <= ttl:
                    continue

                async with self._registry_lock:
                    if self._stats.get(symbol) is stats:
                        del self._stats[symbol]
                        self._symbol_locks.pop(symbol, None)
                        removed_symbols.append(symbol)

        if removed_symbols:
            self.logger.info(
                "Cleaned stale LargeTradeDetector symbol stats",
                extra={
                    "removed_count": len(removed_symbols),
                    "symbols": removed_symbols,
                },
            )

    async def _cleanup_loop(self) -> None:
        """
        Fallback cleanup loop, якщо Scheduler не переданий.
        """
        try:
            while True:
                await asyncio.sleep(self.config.cleanup_interval_sec)
                await self.cleanup()
        except asyncio.CancelledError:
            self.logger.info("Cleanup loop cancelled")
            raise
        except Exception:
            self.logger.exception("Unexpected error in cleanup loop")

    # -------------------------------------------------------------------------
    # Public helpers / metrics
    # -------------------------------------------------------------------------

    def get_symbol_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Повертає best-effort snapshot без лока.

        Важливо:
            Це sync-метод, тому він навмисно не await-ить asyncio lock.
            Значення можуть бути зчитані під час конкурентного оновлення і
            тому не гарантують ідеально атомарний snapshot.
            Для metrics/observability це зазвичай прийнятно.
        """
        stats = self._stats.get(symbol)
        if stats is None:
            return None

        return {
            "symbol": symbol,
            "samples": stats.sample_size,
            "mean_notional": stats.mean(),
            "std_notional": stats.std(),
            "trades_processed": stats.trades_processed,
            "signals_emitted": stats.signals_emitted,
            "last_signal_ts_monotonic": stats.last_signal_ts_monotonic,
            "last_update_ts_monotonic": stats.last_update_ts_monotonic,
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Повертає best-effort snapshots без лока.

        Це навмисно не "strongly consistent" view, а легкий observability helper.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for symbol in list(self._stats.keys()):
            snapshot = self.get_symbol_stats(symbol)
            if snapshot is not None:
                result[symbol] = snapshot
        return result

    async def reset_symbol(self, symbol: str) -> None:
        async with self._registry_lock:
            self._stats.pop(symbol, None)
            self._symbol_locks.pop(symbol, None)

        self.logger.info(
            "Reset symbol stats in LargeTradeDetector",
            extra={"symbol": symbol},
        )

    async def reset_all(self) -> None:
        async with self._registry_lock:
            self._stats.clear()
            self._symbol_locks.clear()

        self.logger.info("Reset all LargeTradeDetector stats")

    # -------------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------------

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

        # Евристика: секунди чи мілісекунди
        if ts < 10_000_000_000:
            ts *= 1000.0

        return int(ts)