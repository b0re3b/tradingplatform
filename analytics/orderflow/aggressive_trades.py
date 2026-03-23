from __future__ import annotations

import asyncio
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from core.event_bus import Event, EventBus, EventPriority
from core.logger import get_logger


@dataclass(slots=True)
class AggressiveTrade:
    symbol: str
    side: str
    price: float
    quantity: float
    notional: float
    timestamp: float
    trade_id: Optional[str] = None
    exchange: Optional[str] = None
    is_aggressive: bool = True


@dataclass(slots=True)
class AggressiveTradesWindowStats:
    symbol: str
    window_seconds: float
    trades_count: int
    aggressive_buy_count: int
    aggressive_sell_count: int
    aggressive_buy_volume: float
    aggressive_sell_volume: float
    aggressive_buy_notional: float
    aggressive_sell_notional: float
    net_volume_delta: float
    net_notional_delta: float
    buy_ratio: float
    sell_ratio: float
    burst_score: float
    large_buy_trades: int
    large_sell_trades: int
    avg_trade_size: float
    avg_trade_notional: float
    last_price: Optional[float]
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class AggressiveTradesSignal:
    symbol: str
    side: str
    strength: float
    signal_type: str
    reason: str
    trades_count: int
    net_volume_delta: float
    net_notional_delta: float
    buy_ratio: float
    sell_ratio: float
    burst_score: float
    last_price: Optional[float]
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class AggressiveTradesConfig:
    enabled: bool = True
    window_seconds: float = 8.0
    max_trades_per_symbol: int = 5000

    min_signal_interval_sec: float = 0.50
    min_trades_in_window: int = 8

    bullish_buy_ratio_threshold: float = 0.68
    bearish_sell_ratio_threshold: float = 0.68

    bullish_delta_threshold: float = 0.0
    bearish_delta_threshold: float = 0.0

    large_trade_notional_threshold: float = 25000.0
    min_large_trades_for_signal: int = 1

    burst_trades_threshold: int = 12
    burst_volume_threshold: float = 0.0
    burst_score_threshold: float = 1.15

    emit_updates: bool = True
    emit_signals: bool = True

    health_log_interval_sec: float = 30.0
    cleanup_interval_sec: float = 15.0
    scheduler_job_timeout_sec: float = 10.0
    scheduler_job_retry_delay_sec: float = 1.0
    scheduler_job_max_retries: int = 1

    symbol_allowlist: Optional[set[str]] = None
    publish_priority: EventPriority = EventPriority.NORMAL

    update_topic: str = "analytics.trades.aggressive.updated"
    signal_topic: str = "analytics.trades.aggressive.signal"

    source_name: str = "aggressive_trades"

    @classmethod
    def from_app_config(cls, app_config: Any) -> "AggressiveTradesConfig":
        analytics_cfg = getattr(app_config, "analytics", None)
        orderflow_cfg = getattr(analytics_cfg, "orderflow", None) if analytics_cfg else None
        aggressive_cfg = getattr(orderflow_cfg, "aggressive_trades", None) if orderflow_cfg else None

        if aggressive_cfg is None:
            return cls()

        return cls(
            enabled=getattr(aggressive_cfg, "enabled", True),
            window_seconds=getattr(aggressive_cfg, "window_seconds", 8.0),
            max_trades_per_symbol=getattr(aggressive_cfg, "max_trades_per_symbol", 5000),
            min_signal_interval_sec=getattr(aggressive_cfg, "min_signal_interval_sec", 0.50),
            min_trades_in_window=getattr(aggressive_cfg, "min_trades_in_window", 8),
            bullish_buy_ratio_threshold=getattr(aggressive_cfg, "bullish_buy_ratio_threshold", 0.68),
            bearish_sell_ratio_threshold=getattr(aggressive_cfg, "bearish_sell_ratio_threshold", 0.68),
            bullish_delta_threshold=getattr(aggressive_cfg, "bullish_delta_threshold", 0.0),
            bearish_delta_threshold=getattr(aggressive_cfg, "bearish_delta_threshold", 0.0),
            large_trade_notional_threshold=getattr(
                aggressive_cfg,
                "large_trade_notional_threshold",
                25000.0,
            ),
            min_large_trades_for_signal=getattr(
                aggressive_cfg,
                "min_large_trades_for_signal",
                1,
            ),
            burst_trades_threshold=getattr(aggressive_cfg, "burst_trades_threshold", 12),
            burst_volume_threshold=getattr(aggressive_cfg, "burst_volume_threshold", 0.0),
            burst_score_threshold=getattr(aggressive_cfg, "burst_score_threshold", 1.15),
            emit_updates=getattr(aggressive_cfg, "emit_updates", True),
            emit_signals=getattr(aggressive_cfg, "emit_signals", True),
            health_log_interval_sec=getattr(aggressive_cfg, "health_log_interval_sec", 30.0),
            cleanup_interval_sec=getattr(aggressive_cfg, "cleanup_interval_sec", 15.0),
            scheduler_job_timeout_sec=getattr(
                aggressive_cfg,
                "scheduler_job_timeout_sec",
                10.0,
            ),
            scheduler_job_retry_delay_sec=getattr(
                aggressive_cfg,
                "scheduler_job_retry_delay_sec",
                1.0,
            ),
            scheduler_job_max_retries=getattr(
                aggressive_cfg,
                "scheduler_job_max_retries",
                1,
            ),
            symbol_allowlist=set(getattr(aggressive_cfg, "symbol_allowlist", []) or []),
            publish_priority=getattr(
                aggressive_cfg,
                "publish_priority",
                EventPriority.NORMAL,
            ),
            update_topic=getattr(
                aggressive_cfg,
                "update_topic",
                "analytics.trades.aggressive.updated",
            ),
            signal_topic=getattr(
                aggressive_cfg,
                "signal_topic",
                "analytics.trades.aggressive.signal",
            ),
            source_name=getattr(
                aggressive_cfg,
                "source_name",
                "aggressive_trades",
            ),
        )


class AggressiveTrades:
    """
    Analytics-модуль для аналізу aggressive trades.

    Основні задачі:
    - приймає trade events через EventBus
    - підтягує останні трейди з trades_cache
    - нормалізує aggressive buy/sell потік
    - рахує window stats
    - визначає burst / large aggressive activity
    - публікує analytics update events та signal events
    - використовує Scheduler тільки для допоміжних periodic jobs
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        trades_cache: Any,
        config: Optional[AggressiveTradesConfig] = None,
        app_config: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        source_topic_patterns: Optional[list[str]] = None,
    ) -> None:
        self._event_bus = event_bus
        self._trades_cache = trades_cache
        self._scheduler = scheduler
        self._config = config or (
            AggressiveTradesConfig.from_app_config(app_config)
            if app_config is not None
            else AggressiveTradesConfig()
        )

        self._source_topic_patterns = source_topic_patterns or [
            "market.trade",
            "market.trade.*",
            "market.trades.updated",
            "trades.*",
        ]

        self._logger = get_logger(
            __name__,
            service_name=self._config.source_name,
            component="analytics",
            module="orderflow",
        )

        self._subscriptions: list[Any] = []
        self._running = False
        self._lock = asyncio.Lock()

        self._trades_by_symbol: dict[str, deque[AggressiveTrade]] = {}
        self._last_stats_by_symbol: dict[str, AggressiveTradesWindowStats] = {}
        self._last_signal_ts_by_symbol: dict[str, float] = {}

        self._health_job_id: Optional[str] = None
        self._cleanup_job_id: Optional[str] = None

        self._metrics: dict[str, Any] = {
            "processed_events": 0,
            "processed_trades": 0,
            "signals_emitted": 0,
            "updates_emitted": 0,
            "skipped": 0,
            "errors": 0,
            "symbols": {},
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            self._logger.warning("AggressiveTrades already started")
            return

        if not self._config.enabled:
            self._logger.warning("AggressiveTrades is disabled by config")
            return

        for pattern in self._source_topic_patterns:
            subscription = self._event_bus.subscribe(
                pattern=pattern,
                handler=self._handle_trade_event,
                name=f"{self.__class__.__name__}:{pattern}",
            )
            self._subscriptions.append(subscription)

        self._register_scheduler_jobs()

        self._running = True
        self._logger.info(
            "AggressiveTrades started | window_seconds=%s min_trades=%s large_trade_notional_threshold=%s",
            self._config.window_seconds,
            self._config.min_trades_in_window,
            self._config.large_trade_notional_threshold,
        )

    def stop(self) -> None:
        if not self._running:
            self._logger.warning("AggressiveTrades already stopped")
            return

        for sub in self._subscriptions:
            try:
                self._event_bus.unsubscribe(sub)
            except Exception:
                self._logger.exception("Failed to unsubscribe AggressiveTrades handler")

        self._subscriptions.clear()
        self._disable_scheduler_jobs()

        self._running = False
        self._logger.info("AggressiveTrades stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_symbol(self, symbol: str) -> Optional[AggressiveTradesWindowStats]:
        if not self._should_process_symbol(symbol):
            self._inc_metric("skipped", symbol)
            return None

        async with self._lock:
            try:
                trades = await self._get_recent_trades(symbol)
                if not trades:
                    self._inc_metric("skipped", symbol)
                    return None

                normalized = self._normalize_trades(symbol, trades)
                if not normalized:
                    self._inc_metric("skipped", symbol)
                    return None

                store = self._trades_by_symbol.setdefault(
                    symbol,
                    deque(maxlen=self._config.max_trades_per_symbol),
                )

                added_count = 0
                for trade in normalized:
                    store.append(trade)
                    added_count += 1

                self._prune_old_trades(symbol)
                stats = self._calculate_window_stats(symbol)
                if stats is None:
                    self._inc_metric("skipped", symbol)
                    return None

                self._last_stats_by_symbol[symbol] = stats
                self._inc_metric("processed_events", symbol)
                self._inc_metric("processed_trades", symbol, amount=added_count)

                if self._config.emit_updates:
                    await self._emit_update(stats)

                if self._config.emit_signals:
                    signal = self._build_signal(stats)
                    if signal is not None:
                        await self._emit_signal(signal)

                return stats

            except Exception:
                self._inc_metric("errors", symbol)
                self._logger.exception(
                    "Failed to process aggressive trades | symbol=%s",
                    symbol,
                )
                return None

    def get_latest_stats(self, symbol: str) -> Optional[AggressiveTradesWindowStats]:
        return self._last_stats_by_symbol.get(symbol)

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "config": {
                "enabled": self._config.enabled,
                "window_seconds": self._config.window_seconds,
                "min_trades_in_window": self._config.min_trades_in_window,
                "large_trade_notional_threshold": self._config.large_trade_notional_threshold,
                "burst_score_threshold": self._config.burst_score_threshold,
                "health_log_interval_sec": self._config.health_log_interval_sec,
                "cleanup_interval_sec": self._config.cleanup_interval_sec,
            },
            "tracked_symbols": len(self._trades_by_symbol),
            "processed_events": self._metrics["processed_events"],
            "processed_trades": self._metrics["processed_trades"],
            "signals_emitted": self._metrics["signals_emitted"],
            "updates_emitted": self._metrics["updates_emitted"],
            "skipped": self._metrics["skipped"],
            "errors": self._metrics["errors"],
            "health_job_id": self._health_job_id,
            "cleanup_job_id": self._cleanup_job_id,
            "symbols": dict(self._metrics["symbols"]),
        }

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _handle_trade_event(self, event: Event) -> None:
        symbol = self._extract_symbol_from_event(event)
        if not symbol:
            self._logger.debug(
                "Trade event without symbol skipped | topic=%s event_id=%s",
                event.topic,
                event.event_id,
            )
            self._inc_metric("skipped")
            return

        await self.process_symbol(symbol)

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def _calculate_window_stats(self, symbol: str) -> Optional[AggressiveTradesWindowStats]:
        trades = self._trades_by_symbol.get(symbol)
        if not trades:
            return None

        recent = list(trades)
        if len(recent) < self._config.min_trades_in_window:
            return None

        buy_trades = [t for t in recent if t.side == "buy" and t.is_aggressive]
        sell_trades = [t for t in recent if t.side == "sell" and t.is_aggressive]

        buy_count = len(buy_trades)
        sell_count = len(sell_trades)
        trades_count = buy_count + sell_count

        if trades_count <= 0:
            return None

        buy_volume = sum(t.quantity for t in buy_trades)
        sell_volume = sum(t.quantity for t in sell_trades)

        buy_notional = sum(t.notional for t in buy_trades)
        sell_notional = sum(t.notional for t in sell_trades)

        total_volume = buy_volume + sell_volume
        total_notional = buy_notional + sell_notional

        if total_volume <= 0 or total_notional <= 0:
            return None

        buy_ratio = buy_volume / total_volume
        sell_ratio = sell_volume / total_volume

        net_volume_delta = buy_volume - sell_volume
        net_notional_delta = buy_notional - sell_notional

        large_buy_trades = sum(
            1 for t in buy_trades if t.notional >= self._config.large_trade_notional_threshold
        )
        large_sell_trades = sum(
            1 for t in sell_trades if t.notional >= self._config.large_trade_notional_threshold
        )

        avg_trade_size = statistics.fmean(t.quantity for t in recent) if recent else 0.0
        avg_trade_notional = statistics.fmean(t.notional for t in recent) if recent else 0.0

        burst_score = self._calculate_burst_score(
            trades_count=trades_count,
            total_volume=total_volume,
            avg_trade_size=avg_trade_size,
        )

        last_price = recent[-1].price if recent else None

        return AggressiveTradesWindowStats(
            symbol=symbol,
            window_seconds=self._config.window_seconds,
            trades_count=trades_count,
            aggressive_buy_count=buy_count,
            aggressive_sell_count=sell_count,
            aggressive_buy_volume=buy_volume,
            aggressive_sell_volume=sell_volume,
            aggressive_buy_notional=buy_notional,
            aggressive_sell_notional=sell_notional,
            net_volume_delta=net_volume_delta,
            net_notional_delta=net_notional_delta,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
            burst_score=burst_score,
            large_buy_trades=large_buy_trades,
            large_sell_trades=large_sell_trades,
            avg_trade_size=avg_trade_size,
            avg_trade_notional=avg_trade_notional,
            last_price=last_price,
        )

    def _build_signal(
        self,
        stats: AggressiveTradesWindowStats,
    ) -> Optional[AggressiveTradesSignal]:
        now = time.time()
        last_signal_ts = self._last_signal_ts_by_symbol.get(stats.symbol, 0.0)

        if now - last_signal_ts < self._config.min_signal_interval_sec:
            return None

        bullish_large_ok = stats.large_buy_trades >= self._config.min_large_trades_for_signal
        bearish_large_ok = stats.large_sell_trades >= self._config.min_large_trades_for_signal

        bullish_burst_ok = (
            stats.trades_count >= self._config.burst_trades_threshold
            and stats.burst_score >= self._config.burst_score_threshold
            and stats.aggressive_buy_volume >= self._config.burst_volume_threshold
        )
        bearish_burst_ok = (
            stats.trades_count >= self._config.burst_trades_threshold
            and stats.burst_score >= self._config.burst_score_threshold
            and stats.aggressive_sell_volume >= self._config.burst_volume_threshold
        )

        bullish_flow_ok = (
            stats.buy_ratio >= self._config.bullish_buy_ratio_threshold
            and stats.net_volume_delta > self._config.bullish_delta_threshold
        )
        bearish_flow_ok = (
            stats.sell_ratio >= self._config.bearish_sell_ratio_threshold
            and stats.net_volume_delta < -self._config.bearish_delta_threshold
        )

        if bullish_flow_ok and (bullish_large_ok or bullish_burst_ok):
            self._last_signal_ts_by_symbol[stats.symbol] = now
            return AggressiveTradesSignal(
                symbol=stats.symbol,
                side="bullish",
                strength=max(abs(stats.net_volume_delta), abs(stats.net_notional_delta)),
                signal_type="aggressive_buy_flow",
                reason=self._build_reason(
                    bullish=True,
                    large_ok=bullish_large_ok,
                    burst_ok=bullish_burst_ok,
                ),
                trades_count=stats.trades_count,
                net_volume_delta=stats.net_volume_delta,
                net_notional_delta=stats.net_notional_delta,
                buy_ratio=stats.buy_ratio,
                sell_ratio=stats.sell_ratio,
                burst_score=stats.burst_score,
                last_price=stats.last_price,
            )

        if bearish_flow_ok and (bearish_large_ok or bearish_burst_ok):
            self._last_signal_ts_by_symbol[stats.symbol] = now
            return AggressiveTradesSignal(
                symbol=stats.symbol,
                side="bearish",
                strength=max(abs(stats.net_volume_delta), abs(stats.net_notional_delta)),
                signal_type="aggressive_sell_flow",
                reason=self._build_reason(
                    bullish=False,
                    large_ok=bearish_large_ok,
                    burst_ok=bearish_burst_ok,
                ),
                trades_count=stats.trades_count,
                net_volume_delta=stats.net_volume_delta,
                net_notional_delta=stats.net_notional_delta,
                buy_ratio=stats.buy_ratio,
                sell_ratio=stats.sell_ratio,
                burst_score=stats.burst_score,
                last_price=stats.last_price,
            )

        return None

    def _calculate_burst_score(
        self,
        *,
        trades_count: int,
        total_volume: float,
        avg_trade_size: float,
    ) -> float:
        if trades_count <= 0:
            return 0.0

        burst_from_count = trades_count / max(1.0, float(self._config.min_trades_in_window))

        if self._config.burst_volume_threshold > 0:
            burst_from_volume = total_volume / self._config.burst_volume_threshold
        else:
            burst_from_volume = 1.0 if total_volume > 0 else 0.0

        size_factor = max(1.0, avg_trade_size) if avg_trade_size > 0 else 1.0
        raw_score = (burst_from_count * 0.7) + (burst_from_volume * 0.3)

        return raw_score * math.log(size_factor + 1.0)

    def _build_reason(self, *, bullish: bool, large_ok: bool, burst_ok: bool) -> str:
        parts: list[str] = []

        if bullish:
            parts.append("buy_pressure_dominates")
        else:
            parts.append("sell_pressure_dominates")

        if large_ok:
            parts.append("large_aggressive_trades_detected")

        if burst_ok:
            parts.append("burst_activity_detected")

        return "|".join(parts)

    # ------------------------------------------------------------------
    # Emitters
    # ------------------------------------------------------------------

    async def _emit_update(self, stats: AggressiveTradesWindowStats) -> None:
        payload = {
            "symbol": stats.symbol,
            "window_seconds": stats.window_seconds,
            "trades_count": stats.trades_count,
            "aggressive_buy_count": stats.aggressive_buy_count,
            "aggressive_sell_count": stats.aggressive_sell_count,
            "aggressive_buy_volume": stats.aggressive_buy_volume,
            "aggressive_sell_volume": stats.aggressive_sell_volume,
            "aggressive_buy_notional": stats.aggressive_buy_notional,
            "aggressive_sell_notional": stats.aggressive_sell_notional,
            "net_volume_delta": stats.net_volume_delta,
            "net_notional_delta": stats.net_notional_delta,
            "buy_ratio": stats.buy_ratio,
            "sell_ratio": stats.sell_ratio,
            "burst_score": stats.burst_score,
            "large_buy_trades": stats.large_buy_trades,
            "large_sell_trades": stats.large_sell_trades,
            "avg_trade_size": stats.avg_trade_size,
            "avg_trade_notional": stats.avg_trade_notional,
            "last_price": stats.last_price,
            "timestamp": stats.timestamp,
        }

        accepted = await self._event_bus.emit(
            topic=self._config.update_topic,
            payload=payload,
            priority=self._config.publish_priority,
            source=self._config.source_name,
            headers={
                "symbol": stats.symbol,
                "analytics_type": "aggressive_trades",
            },
        )

        if accepted:
            self._inc_metric("updates_emitted", stats.symbol)

    async def _emit_signal(self, signal: AggressiveTradesSignal) -> None:
        payload = {
            "symbol": signal.symbol,
            "side": signal.side,
            "strength": signal.strength,
            "signal_type": signal.signal_type,
            "reason": signal.reason,
            "trades_count": signal.trades_count,
            "net_volume_delta": signal.net_volume_delta,
            "net_notional_delta": signal.net_notional_delta,
            "buy_ratio": signal.buy_ratio,
            "sell_ratio": signal.sell_ratio,
            "burst_score": signal.burst_score,
            "last_price": signal.last_price,
            "timestamp": signal.timestamp,
        }

        accepted = await self._event_bus.emit(
            topic=self._config.signal_topic,
            payload=payload,
            priority=EventPriority.HIGH,
            source=self._config.source_name,
            headers={
                "symbol": signal.symbol,
                "signal_type": "aggressive_trades",
                "side": signal.side,
            },
        )

        if accepted:
            self._inc_metric("signals_emitted", signal.symbol)
            self._logger.info(
                "Aggressive trades signal emitted | symbol=%s side=%s type=%s strength=%.4f burst=%.4f",
                signal.symbol,
                signal.side,
                signal.signal_type,
                signal.strength,
                signal.burst_score,
            )

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            return

        try:
            if hasattr(self._scheduler, "get_job_by_name"):
                existing_health = self._scheduler.get_job_by_name("aggressive_trades_health")
                if existing_health is not None:
                    self._health_job_id = existing_health.job_id

                existing_cleanup = self._scheduler.get_job_by_name("aggressive_trades_cleanup")
                if existing_cleanup is not None:
                    self._cleanup_job_id = existing_cleanup.job_id

            if self._health_job_id is None and hasattr(self._scheduler, "add_interval_job"):
                self._health_job_id = self._scheduler.add_interval_job(
                    name="aggressive_trades_health",
                    func=self._log_health_snapshot,
                    interval=self._config.health_log_interval_sec,
                    run_immediately=False,
                    max_retries=self._config.scheduler_job_max_retries,
                    retry_delay=self._config.scheduler_job_retry_delay_sec,
                    timeout=self._config.scheduler_job_timeout_sec,
                    allow_overlap=False,
                    enabled=True,
                )

                self._logger.info(
                    "AggressiveTrades health scheduler job registered | job_id=%s",
                    self._health_job_id,
                )

            if self._cleanup_job_id is None and hasattr(self._scheduler, "add_interval_job"):
                self._cleanup_job_id = self._scheduler.add_interval_job(
                    name="aggressive_trades_cleanup",
                    func=self._cleanup_all_symbols,
                    interval=self._config.cleanup_interval_sec,
                    run_immediately=False,
                    max_retries=self._config.scheduler_job_max_retries,
                    retry_delay=self._config.scheduler_job_retry_delay_sec,
                    timeout=self._config.scheduler_job_timeout_sec,
                    allow_overlap=False,
                    enabled=True,
                )

                self._logger.info(
                    "AggressiveTrades cleanup scheduler job registered | job_id=%s",
                    self._cleanup_job_id,
                )

        except Exception:
            self._logger.exception(
                "Failed to register scheduler jobs for AggressiveTrades"
            )

    def _disable_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            return

        for job_id, job_name in (
            (self._health_job_id, "aggressive_trades_health"),
            (self._cleanup_job_id, "aggressive_trades_cleanup"),
        ):
            if job_id is None:
                continue

            try:
                if hasattr(self._scheduler, "disable_job"):
                    self._scheduler.disable_job(job_id)
                    self._logger.info(
                        "AggressiveTrades scheduler job disabled | name=%s job_id=%s",
                        job_name,
                        job_id,
                    )
            except Exception:
                self._logger.exception(
                    "Failed to disable AggressiveTrades scheduler job | name=%s job_id=%s",
                    job_name,
                    job_id,
                )

    async def _log_health_snapshot(self) -> None:
        self._logger.info(
            "AggressiveTrades health | running=%s processed_events=%s processed_trades=%s signals=%s errors=%s tracked_symbols=%s",
            self._running,
            self._metrics["processed_events"],
            self._metrics["processed_trades"],
            self._metrics["signals_emitted"],
            self._metrics["errors"],
            len(self._trades_by_symbol),
        )

    async def _cleanup_all_symbols(self) -> None:
        async with self._lock:
            removed = 0
            for symbol in list(self._trades_by_symbol.keys()):
                removed += self._prune_old_trades(symbol)

            self._logger.debug(
                "AggressiveTrades cleanup finished | removed=%s tracked_symbols=%s",
                removed,
                len(self._trades_by_symbol),
            )

    # ------------------------------------------------------------------
    # Cache access / normalization
    # ------------------------------------------------------------------

    async def _get_recent_trades(self, symbol: str) -> list[Any]:
        cache = self._trades_cache

        for method_name in (
            "get_recent_trades",
            "get_trades",
            "get",
            "get_snapshot",
        ):
            method = getattr(cache, method_name, None)
            if method is None:
                continue

            try:
                result = method(symbol)
                if asyncio.iscoroutine(result):
                    result = await result

                if result is None:
                    continue

                if isinstance(result, list):
                    return result

                if isinstance(result, tuple):
                    return list(result)

                if isinstance(result, deque):
                    return list(result)
            except Exception:
                self._logger.exception(
                    "Failed to read trades from cache | symbol=%s method=%s",
                    symbol,
                    method_name,
                )

        if isinstance(cache, dict):
            raw = cache.get(symbol)
            if raw is None:
                return []
            if isinstance(raw, list):
                return raw
            if isinstance(raw, tuple):
                return list(raw)
            if isinstance(raw, deque):
                return list(raw)

        return []

    def _normalize_trades(self, symbol: str, raw_trades: list[Any]) -> list[AggressiveTrade]:
        normalized: list[AggressiveTrade] = []

        for raw in raw_trades:
            trade = self._parse_trade(symbol, raw)
            if trade is None:
                continue
            normalized.append(trade)

        return normalized

    def _parse_trade(self, symbol: str, raw: Any) -> Optional[AggressiveTrade]:
        try:
            if isinstance(raw, AggressiveTrade):
                return raw

            if isinstance(raw, dict):
                price = self._safe_float(
                    raw.get("price", raw.get("p"))
                )
                quantity = self._safe_float(
                    raw.get("quantity", raw.get("qty", raw.get("size", raw.get("q"))))
                )
                timestamp = self._safe_float(
                    raw.get("timestamp", raw.get("ts", raw.get("time", time.time())))
                )

                side = self._extract_side_from_dict(raw)
                if side is None:
                    return None

                if price is None or quantity is None or timestamp is None:
                    return None

                if price <= 0 or quantity <= 0:
                    return None

                trade_id = raw.get("trade_id", raw.get("id"))
                exchange = raw.get("exchange")

                return AggressiveTrade(
                    symbol=str(raw.get("symbol", symbol)),
                    side=side,
                    price=price,
                    quantity=quantity,
                    notional=price * quantity,
                    timestamp=timestamp,
                    trade_id=str(trade_id) if trade_id is not None else None,
                    exchange=str(exchange) if exchange is not None else None,
                    is_aggressive=True,
                )

            price = self._safe_float(getattr(raw, "price", None))
            quantity = self._safe_float(
                getattr(raw, "quantity", getattr(raw, "qty", getattr(raw, "size", None)))
            )
            timestamp = self._safe_float(
                getattr(raw, "timestamp", getattr(raw, "time", time.time()))
            )
            side = self._extract_side_from_object(raw)

            if side is None or price is None or quantity is None or timestamp is None:
                return None

            if price <= 0 or quantity <= 0:
                return None

            trade_id = getattr(raw, "trade_id", getattr(raw, "id", None))
            exchange = getattr(raw, "exchange", None)
            symbol_value = getattr(raw, "symbol", symbol)

            return AggressiveTrade(
                symbol=str(symbol_value),
                side=side,
                price=price,
                quantity=quantity,
                notional=price * quantity,
                timestamp=timestamp,
                trade_id=str(trade_id) if trade_id is not None else None,
                exchange=str(exchange) if exchange is not None else None,
                is_aggressive=True,
            )
        except Exception:
            return None

    def _extract_side_from_dict(self, raw: dict[str, Any]) -> Optional[str]:
        side = raw.get("side")
        if isinstance(side, str):
            side_lower = side.lower()
            if side_lower in {"buy", "bid", "b"}:
                return "buy"
            if side_lower in {"sell", "ask", "s"}:
                return "sell"

        is_buyer_maker = raw.get("is_buyer_maker", raw.get("m"))
        if isinstance(is_buyer_maker, bool):
            return "sell" if is_buyer_maker else "buy"

        aggressor_side = raw.get("aggressor_side")
        if isinstance(aggressor_side, str):
            aggressor_side = aggressor_side.lower()
            if aggressor_side in {"buy", "sell"}:
                return aggressor_side

        return None

    def _extract_side_from_object(self, raw: Any) -> Optional[str]:
        side = getattr(raw, "side", None)
        if isinstance(side, str):
            side_lower = side.lower()
            if side_lower in {"buy", "bid", "b"}:
                return "buy"
            if side_lower in {"sell", "ask", "s"}:
                return "sell"

        is_buyer_maker = getattr(raw, "is_buyer_maker", getattr(raw, "m", None))
        if isinstance(is_buyer_maker, bool):
            return "sell" if is_buyer_maker else "buy"

        aggressor_side = getattr(raw, "aggressor_side", None)
        if isinstance(aggressor_side, str):
            aggressor_side = aggressor_side.lower()
            if aggressor_side in {"buy", "sell"}:
                return aggressor_side

        return None

    # ------------------------------------------------------------------
    # Cleanup / pruning
    # ------------------------------------------------------------------

    def _prune_old_trades(self, symbol: str) -> int:
        trades = self._trades_by_symbol.get(symbol)
        if not trades:
            return 0

        min_ts = time.time() - self._config.window_seconds
        removed = 0

        while trades and trades[0].timestamp < min_ts:
            trades.popleft()
            removed += 1

        if not trades:
            self._trades_by_symbol.pop(symbol, None)

        return removed

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    def _extract_symbol_from_event(self, event: Event) -> Optional[str]:
        payload = event.payload

        if isinstance(payload, dict):
            symbol = payload.get("symbol") or payload.get("instrument") or payload.get("pair")
            if symbol:
                return str(symbol)

        if event.headers:
            header_symbol = event.headers.get("symbol")
            if header_symbol:
                return str(header_symbol)

        return None

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _should_process_symbol(self, symbol: str) -> bool:
        allowlist = self._config.symbol_allowlist
        if not allowlist:
            return True
        return symbol in allowlist

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            converted = float(value)
            if not math.isfinite(converted):
                return None
            return converted
        except Exception:
            return None

    def _inc_metric(self, key: str, symbol: Optional[str] = None, amount: int = 1) -> None:
        self._metrics[key] = self._metrics.get(key, 0) + amount

        if symbol:
            symbols = self._metrics.setdefault("symbols", {})
            symbol_stats = symbols.setdefault(
                symbol,
                {
                    "processed_events": 0,
                    "processed_trades": 0,
                    "signals_emitted": 0,
                    "updates_emitted": 0,
                    "skipped": 0,
                    "errors": 0,
                },
            )
            if key in symbol_stats:
                symbol_stats[key] += amount