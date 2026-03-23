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
class CvdTrade:
    symbol: str
    side: str
    price: float
    quantity: float
    notional: float
    timestamp: float
    trade_id: Optional[str] = None
    exchange: Optional[str] = None


@dataclass(slots=True)
class CvdPoint:
    timestamp: float
    value: float
    price: Optional[float] = None


@dataclass(slots=True)
class CvdStats:
    symbol: str
    window_seconds: float
    trades_count: int
    buy_volume: float
    sell_volume: float
    volume_delta: float
    notional_delta: float
    cvd_value: float
    cvd_open: float
    cvd_high: float
    cvd_low: float
    cvd_close: float
    cvd_change: float
    cvd_change_pct: float
    cvd_slope: float
    delta_ratio: float
    buy_ratio: float
    sell_ratio: float
    avg_trade_size: float
    avg_trade_notional: float
    last_price: Optional[float]
    price_change: Optional[float]
    price_change_pct: Optional[float]
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class CvdSignal:
    symbol: str
    side: str
    strength: float
    signal_type: str
    reason: str
    cvd_value: float
    cvd_change: float
    cvd_change_pct: float
    cvd_slope: float
    delta_ratio: float
    trades_count: int
    last_price: Optional[float]
    price_change: Optional[float]
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class CvdConfig:
    enabled: bool = True
    window_seconds: float = 20.0
    max_trades_per_symbol: int = 8000
    max_cvd_points_per_symbol: int = 5000

    min_signal_interval_sec: float = 0.75
    min_trades_in_window: int = 12
    min_total_volume: float = 0.0

    bullish_delta_ratio_threshold: float = 0.15
    bearish_delta_ratio_threshold: float = -0.15

    bullish_cvd_change_threshold: float = 0.0
    bearish_cvd_change_threshold: float = 0.0

    bullish_cvd_slope_threshold: float = 0.0
    bearish_cvd_slope_threshold: float = 0.0

    bullish_impulse_threshold_pct: float = 0.0
    bearish_impulse_threshold_pct: float = 0.0

    require_delta_confirmation: bool = True
    require_slope_confirmation: bool = True

    emit_updates: bool = True
    emit_signals: bool = True

    health_log_interval_sec: float = 30.0
    cleanup_interval_sec: float = 15.0
    scheduler_job_timeout_sec: float = 10.0
    scheduler_job_retry_delay_sec: float = 1.0
    scheduler_job_max_retries: int = 1

    symbol_allowlist: Optional[set[str]] = None
    publish_priority: EventPriority = EventPriority.NORMAL

    update_topic: str = "analytics.trades.cvd.updated"
    signal_topic: str = "analytics.trades.cvd.signal"

    source_name: str = "cvd"

    @classmethod
    def from_app_config(cls, app_config: Any) -> "CvdConfig":
        analytics_cfg = getattr(app_config, "analytics", None)
        orderflow_cfg = getattr(analytics_cfg, "orderflow", None) if analytics_cfg else None
        cvd_cfg = getattr(orderflow_cfg, "cvd", None) if orderflow_cfg else None

        if cvd_cfg is None:
            return cls()

        return cls(
            enabled=getattr(cvd_cfg, "enabled", True),
            window_seconds=getattr(cvd_cfg, "window_seconds", 20.0),
            max_trades_per_symbol=getattr(cvd_cfg, "max_trades_per_symbol", 8000),
            max_cvd_points_per_symbol=getattr(cvd_cfg, "max_cvd_points_per_symbol", 5000),
            min_signal_interval_sec=getattr(cvd_cfg, "min_signal_interval_sec", 0.75),
            min_trades_in_window=getattr(cvd_cfg, "min_trades_in_window", 12),
            min_total_volume=getattr(cvd_cfg, "min_total_volume", 0.0),
            bullish_delta_ratio_threshold=getattr(
                cvd_cfg,
                "bullish_delta_ratio_threshold",
                0.15,
            ),
            bearish_delta_ratio_threshold=getattr(
                cvd_cfg,
                "bearish_delta_ratio_threshold",
                -0.15,
            ),
            bullish_cvd_change_threshold=getattr(
                cvd_cfg,
                "bullish_cvd_change_threshold",
                0.0,
            ),
            bearish_cvd_change_threshold=getattr(
                cvd_cfg,
                "bearish_cvd_change_threshold",
                0.0,
            ),
            bullish_cvd_slope_threshold=getattr(
                cvd_cfg,
                "bullish_cvd_slope_threshold",
                0.0,
            ),
            bearish_cvd_slope_threshold=getattr(
                cvd_cfg,
                "bearish_cvd_slope_threshold",
                0.0,
            ),
            bullish_impulse_threshold_pct=getattr(
                cvd_cfg,
                "bullish_impulse_threshold_pct",
                0.0,
            ),
            bearish_impulse_threshold_pct=getattr(
                cvd_cfg,
                "bearish_impulse_threshold_pct",
                0.0,
            ),
            require_delta_confirmation=getattr(
                cvd_cfg,
                "require_delta_confirmation",
                True,
            ),
            require_slope_confirmation=getattr(
                cvd_cfg,
                "require_slope_confirmation",
                True,
            ),
            emit_updates=getattr(cvd_cfg, "emit_updates", True),
            emit_signals=getattr(cvd_cfg, "emit_signals", True),
            health_log_interval_sec=getattr(cvd_cfg, "health_log_interval_sec", 30.0),
            cleanup_interval_sec=getattr(cvd_cfg, "cleanup_interval_sec", 15.0),
            scheduler_job_timeout_sec=getattr(
                cvd_cfg,
                "scheduler_job_timeout_sec",
                10.0,
            ),
            scheduler_job_retry_delay_sec=getattr(
                cvd_cfg,
                "scheduler_job_retry_delay_sec",
                1.0,
            ),
            scheduler_job_max_retries=getattr(
                cvd_cfg,
                "scheduler_job_max_retries",
                1,
            ),
            symbol_allowlist=set(getattr(cvd_cfg, "symbol_allowlist", []) or []),
            publish_priority=getattr(
                cvd_cfg,
                "publish_priority",
                EventPriority.NORMAL,
            ),
            update_topic=getattr(
                cvd_cfg,
                "update_topic",
                "analytics.trades.cvd.updated",
            ),
            signal_topic=getattr(
                cvd_cfg,
                "signal_topic",
                "analytics.trades.cvd.signal",
            ),
            source_name=getattr(
                cvd_cfg,
                "source_name",
                "cvd",
            ),
        )


class CVD:
    """
    Analytics-модуль для розрахунку CVD (Cumulative Volume Delta).

    Основні задачі:
    - приймає trade events через EventBus
    - читає останні трейди з trades_cache
    - нормалізує buy/sell trades
    - веде cumulative delta per symbol
    - формує window-based CVD stats
    - публікує analytics update events та signal events
    - використовує Scheduler для health/cleanup jobs
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        trades_cache: Any,
        config: Optional[CvdConfig] = None,
        app_config: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        source_topic_patterns: Optional[list[str]] = None,
    ) -> None:
        self._event_bus = event_bus
        self._trades_cache = trades_cache
        self._scheduler = scheduler
        self._config = config or (
            CvdConfig.from_app_config(app_config)
            if app_config is not None
            else CvdConfig()
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

        self._trades_by_symbol: dict[str, deque[CvdTrade]] = {}
        self._cvd_points_by_symbol: dict[str, deque[CvdPoint]] = {}
        self._last_stats_by_symbol: dict[str, CvdStats] = {}
        self._last_signal_ts_by_symbol: dict[str, float] = {}

        self._last_seen_trade_key_by_symbol: dict[str, str] = {}
        self._cumulative_cvd_by_symbol: dict[str, float] = {}

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
            self._logger.warning("CVD already started")
            return

        if not self._config.enabled:
            self._logger.warning("CVD is disabled by config")
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
            "CVD started | window_seconds=%s min_trades=%s bullish_delta_ratio_threshold=%s bearish_delta_ratio_threshold=%s",
            self._config.window_seconds,
            self._config.min_trades_in_window,
            self._config.bullish_delta_ratio_threshold,
            self._config.bearish_delta_ratio_threshold,
        )

    def stop(self) -> None:
        if not self._running:
            self._logger.warning("CVD already stopped")
            return

        for sub in self._subscriptions:
            try:
                self._event_bus.unsubscribe(sub)
            except Exception:
                self._logger.exception("Failed to unsubscribe CVD handler")

        self._subscriptions.clear()
        self._disable_scheduler_jobs()

        self._running = False
        self._logger.info("CVD stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_symbol(self, symbol: str) -> Optional[CvdStats]:
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

                new_trades = self._filter_new_trades(symbol, normalized)
                if not new_trades and symbol not in self._trades_by_symbol:
                    self._inc_metric("skipped", symbol)
                    return None

                trade_store = self._trades_by_symbol.setdefault(
                    symbol,
                    deque(maxlen=self._config.max_trades_per_symbol),
                )
                cvd_store = self._cvd_points_by_symbol.setdefault(
                    symbol,
                    deque(maxlen=self._config.max_cvd_points_per_symbol),
                )

                added_count = 0
                for trade in new_trades:
                    trade_store.append(trade)

                    prev_cvd = self._cumulative_cvd_by_symbol.get(symbol, 0.0)
                    signed_volume = trade.quantity if trade.side == "buy" else -trade.quantity
                    new_cvd_value = prev_cvd + signed_volume
                    self._cumulative_cvd_by_symbol[symbol] = new_cvd_value

                    cvd_store.append(
                        CvdPoint(
                            timestamp=trade.timestamp,
                            value=new_cvd_value,
                            price=trade.price,
                        )
                    )
                    added_count += 1

                self._prune_old_trades(symbol)
                self._prune_old_cvd_points(symbol)

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
                    "Failed to process CVD | symbol=%s",
                    symbol,
                )
                return None

    def get_latest_stats(self, symbol: str) -> Optional[CvdStats]:
        return self._last_stats_by_symbol.get(symbol)

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "config": {
                "enabled": self._config.enabled,
                "window_seconds": self._config.window_seconds,
                "min_trades_in_window": self._config.min_trades_in_window,
                "bullish_delta_ratio_threshold": self._config.bullish_delta_ratio_threshold,
                "bearish_delta_ratio_threshold": self._config.bearish_delta_ratio_threshold,
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

    def _calculate_window_stats(self, symbol: str) -> Optional[CvdStats]:
        trades = self._trades_by_symbol.get(symbol)
        cvd_points = self._cvd_points_by_symbol.get(symbol)

        if not trades or not cvd_points:
            return None

        recent_trades = list(trades)
        if len(recent_trades) < self._config.min_trades_in_window:
            return None

        buy_trades = [t for t in recent_trades if t.side == "buy"]
        sell_trades = [t for t in recent_trades if t.side == "sell"]

        buy_volume = sum(t.quantity for t in buy_trades)
        sell_volume = sum(t.quantity for t in sell_trades)
        total_volume = buy_volume + sell_volume

        if total_volume <= 0:
            return None

        if total_volume < self._config.min_total_volume:
            return None

        buy_notional = sum(t.notional for t in buy_trades)
        sell_notional = sum(t.notional for t in sell_trades)

        volume_delta = buy_volume - sell_volume
        notional_delta = buy_notional - sell_notional
        delta_ratio = volume_delta / total_volume

        buy_ratio = buy_volume / total_volume
        sell_ratio = sell_volume / total_volume

        avg_trade_size = total_volume / len(recent_trades)
        total_notional = buy_notional + sell_notional
        avg_trade_notional = total_notional / len(recent_trades) if total_notional > 0 else 0.0

        recent_cvd = list(cvd_points)
        cvd_values = [point.value for point in recent_cvd]

        cvd_open = cvd_values[0]
        cvd_high = max(cvd_values)
        cvd_low = min(cvd_values)
        cvd_close = cvd_values[-1]
        cvd_value = cvd_close
        cvd_change = cvd_close - cvd_open

        cvd_change_pct = 0.0
        if cvd_open != 0:
            cvd_change_pct = (cvd_change / abs(cvd_open)) * 100.0

        cvd_slope = self._calculate_cvd_slope(recent_cvd)

        first_price = recent_trades[0].price if recent_trades else None
        last_price = recent_trades[-1].price if recent_trades else None

        price_change = None
        price_change_pct = None
        if first_price is not None and last_price is not None:
            price_change = last_price - first_price
            if first_price != 0:
                price_change_pct = (price_change / first_price) * 100.0
            else:
                price_change_pct = 0.0

        return CvdStats(
            symbol=symbol,
            window_seconds=self._config.window_seconds,
            trades_count=len(recent_trades),
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            volume_delta=volume_delta,
            notional_delta=notional_delta,
            cvd_value=cvd_value,
            cvd_open=cvd_open,
            cvd_high=cvd_high,
            cvd_low=cvd_low,
            cvd_close=cvd_close,
            cvd_change=cvd_change,
            cvd_change_pct=cvd_change_pct,
            cvd_slope=cvd_slope,
            delta_ratio=delta_ratio,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
            avg_trade_size=avg_trade_size,
            avg_trade_notional=avg_trade_notional,
            last_price=last_price,
            price_change=price_change,
            price_change_pct=price_change_pct,
        )

    def _build_signal(self, stats: CvdStats) -> Optional[CvdSignal]:
        now = time.time()
        last_signal_ts = self._last_signal_ts_by_symbol.get(stats.symbol, 0.0)

        if now - last_signal_ts < self._config.min_signal_interval_sec:
            return None

        bullish_ratio_ok = stats.delta_ratio >= self._config.bullish_delta_ratio_threshold
        bearish_ratio_ok = stats.delta_ratio <= self._config.bearish_delta_ratio_threshold

        bullish_cvd_change_ok = stats.cvd_change > self._config.bullish_cvd_change_threshold
        bearish_cvd_change_ok = stats.cvd_change < -self._config.bearish_cvd_change_threshold

        bullish_slope_ok = stats.cvd_slope > self._config.bullish_cvd_slope_threshold
        bearish_slope_ok = stats.cvd_slope < -self._config.bearish_cvd_slope_threshold

        bullish_impulse_ok = stats.cvd_change_pct >= self._config.bullish_impulse_threshold_pct
        bearish_impulse_ok = stats.cvd_change_pct <= -self._config.bearish_impulse_threshold_pct

        bullish_ok = bullish_cvd_change_ok and bullish_impulse_ok
        bearish_ok = bearish_cvd_change_ok and bearish_impulse_ok

        if self._config.require_delta_confirmation:
            bullish_ok = bullish_ok and bullish_ratio_ok
            bearish_ok = bearish_ok and bearish_ratio_ok

        if self._config.require_slope_confirmation:
            bullish_ok = bullish_ok and bullish_slope_ok
            bearish_ok = bearish_ok and bearish_slope_ok

        if bullish_ok:
            self._last_signal_ts_by_symbol[stats.symbol] = now
            return CvdSignal(
                symbol=stats.symbol,
                side="bullish",
                strength=max(
                    abs(stats.cvd_change),
                    abs(stats.cvd_slope),
                    abs(stats.volume_delta),
                ),
                signal_type="bullish_cvd_impulse",
                reason=self._build_reason(
                    side="bullish",
                    ratio_ok=bullish_ratio_ok,
                    cvd_change_ok=bullish_cvd_change_ok,
                    slope_ok=bullish_slope_ok,
                    impulse_ok=bullish_impulse_ok,
                ),
                cvd_value=stats.cvd_value,
                cvd_change=stats.cvd_change,
                cvd_change_pct=stats.cvd_change_pct,
                cvd_slope=stats.cvd_slope,
                delta_ratio=stats.delta_ratio,
                trades_count=stats.trades_count,
                last_price=stats.last_price,
                price_change=stats.price_change,
            )

        if bearish_ok:
            self._last_signal_ts_by_symbol[stats.symbol] = now
            return CvdSignal(
                symbol=stats.symbol,
                side="bearish",
                strength=max(
                    abs(stats.cvd_change),
                    abs(stats.cvd_slope),
                    abs(stats.volume_delta),
                ),
                signal_type="bearish_cvd_impulse",
                reason=self._build_reason(
                    side="bearish",
                    ratio_ok=bearish_ratio_ok,
                    cvd_change_ok=bearish_cvd_change_ok,
                    slope_ok=bearish_slope_ok,
                    impulse_ok=bearish_impulse_ok,
                ),
                cvd_value=stats.cvd_value,
                cvd_change=stats.cvd_change,
                cvd_change_pct=stats.cvd_change_pct,
                cvd_slope=stats.cvd_slope,
                delta_ratio=stats.delta_ratio,
                trades_count=stats.trades_count,
                last_price=stats.last_price,
                price_change=stats.price_change,
            )

        return None

    def _calculate_cvd_slope(self, points: list[CvdPoint]) -> float:
        if len(points) < 2:
            return 0.0

        first = points[0]
        last = points[-1]

        dt = last.timestamp - first.timestamp
        if dt <= 0:
            return 0.0

        return (last.value - first.value) / dt

    def _build_reason(
        self,
        *,
        side: str,
        ratio_ok: bool,
        cvd_change_ok: bool,
        slope_ok: bool,
        impulse_ok: bool,
    ) -> str:
        parts: list[str] = []

        if side == "bullish":
            parts.append("cvd_buy_pressure")
        else:
            parts.append("cvd_sell_pressure")

        if ratio_ok:
            parts.append("delta_ratio_confirmed")

        if cvd_change_ok:
            parts.append("cvd_change_confirmed")

        if slope_ok:
            parts.append("cvd_slope_confirmed")

        if impulse_ok:
            parts.append("cvd_impulse_confirmed")

        return "|".join(parts)

    # ------------------------------------------------------------------
    # Emitters
    # ------------------------------------------------------------------

    async def _emit_update(self, stats: CvdStats) -> None:
        payload = {
            "symbol": stats.symbol,
            "window_seconds": stats.window_seconds,
            "trades_count": stats.trades_count,
            "buy_volume": stats.buy_volume,
            "sell_volume": stats.sell_volume,
            "volume_delta": stats.volume_delta,
            "notional_delta": stats.notional_delta,
            "cvd_value": stats.cvd_value,
            "cvd_open": stats.cvd_open,
            "cvd_high": stats.cvd_high,
            "cvd_low": stats.cvd_low,
            "cvd_close": stats.cvd_close,
            "cvd_change": stats.cvd_change,
            "cvd_change_pct": stats.cvd_change_pct,
            "cvd_slope": stats.cvd_slope,
            "delta_ratio": stats.delta_ratio,
            "buy_ratio": stats.buy_ratio,
            "sell_ratio": stats.sell_ratio,
            "avg_trade_size": stats.avg_trade_size,
            "avg_trade_notional": stats.avg_trade_notional,
            "last_price": stats.last_price,
            "price_change": stats.price_change,
            "price_change_pct": stats.price_change_pct,
            "timestamp": stats.timestamp,
        }

        accepted = await self._event_bus.emit(
            topic=self._config.update_topic,
            payload=payload,
            priority=self._config.publish_priority,
            source=self._config.source_name,
            headers={
                "symbol": stats.symbol,
                "analytics_type": "cvd",
            },
        )

        if accepted:
            self._inc_metric("updates_emitted", stats.symbol)

    async def _emit_signal(self, signal: CvdSignal) -> None:
        payload = {
            "symbol": signal.symbol,
            "side": signal.side,
            "strength": signal.strength,
            "signal_type": signal.signal_type,
            "reason": signal.reason,
            "cvd_value": signal.cvd_value,
            "cvd_change": signal.cvd_change,
            "cvd_change_pct": signal.cvd_change_pct,
            "cvd_slope": signal.cvd_slope,
            "delta_ratio": signal.delta_ratio,
            "trades_count": signal.trades_count,
            "last_price": signal.last_price,
            "price_change": signal.price_change,
            "timestamp": signal.timestamp,
        }

        accepted = await self._event_bus.emit(
            topic=self._config.signal_topic,
            payload=payload,
            priority=EventPriority.HIGH,
            source=self._config.source_name,
            headers={
                "symbol": signal.symbol,
                "signal_type": "cvd",
                "side": signal.side,
            },
        )

        if accepted:
            self._inc_metric("signals_emitted", signal.symbol)
            self._logger.info(
                "CVD signal emitted | symbol=%s side=%s type=%s strength=%.4f cvd_change=%.4f slope=%.4f",
                signal.symbol,
                signal.side,
                signal.signal_type,
                signal.strength,
                signal.cvd_change,
                signal.cvd_slope,
            )

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            return

        try:
            if hasattr(self._scheduler, "get_job_by_name"):
                existing_health = self._scheduler.get_job_by_name("cvd_health")
                if existing_health is not None:
                    self._health_job_id = existing_health.job_id

                existing_cleanup = self._scheduler.get_job_by_name("cvd_cleanup")
                if existing_cleanup is not None:
                    self._cleanup_job_id = existing_cleanup.job_id

            if self._health_job_id is None and hasattr(self._scheduler, "add_interval_job"):
                self._health_job_id = self._scheduler.add_interval_job(
                    name="cvd_health",
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
                    "CVD health scheduler job registered | job_id=%s",
                    self._health_job_id,
                )

            if self._cleanup_job_id is None and hasattr(self._scheduler, "add_interval_job"):
                self._cleanup_job_id = self._scheduler.add_interval_job(
                    name="cvd_cleanup",
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
                    "CVD cleanup scheduler job registered | job_id=%s",
                    self._cleanup_job_id,
                )

        except Exception:
            self._logger.exception("Failed to register scheduler jobs for CVD")

    def _disable_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            return

        for job_id, job_name in (
            (self._health_job_id, "cvd_health"),
            (self._cleanup_job_id, "cvd_cleanup"),
        ):
            if job_id is None:
                continue

            try:
                if hasattr(self._scheduler, "disable_job"):
                    self._scheduler.disable_job(job_id)
                    self._logger.info(
                        "CVD scheduler job disabled | name=%s job_id=%s",
                        job_name,
                        job_id,
                    )
            except Exception:
                self._logger.exception(
                    "Failed to disable CVD scheduler job | name=%s job_id=%s",
                    job_name,
                    job_id,
                )

    async def _log_health_snapshot(self) -> None:
        self._logger.info(
            "CVD health | running=%s processed_events=%s processed_trades=%s signals=%s errors=%s tracked_symbols=%s",
            self._running,
            self._metrics["processed_events"],
            self._metrics["processed_trades"],
            self._metrics["signals_emitted"],
            self._metrics["errors"],
            len(self._trades_by_symbol),
        )

    async def _cleanup_all_symbols(self) -> None:
        async with self._lock:
            removed_trades = 0
            removed_points = 0

            for symbol in list(self._trades_by_symbol.keys()):
                removed_trades += self._prune_old_trades(symbol)

            for symbol in list(self._cvd_points_by_symbol.keys()):
                removed_points += self._prune_old_cvd_points(symbol)

            self._logger.debug(
                "CVD cleanup finished | removed_trades=%s removed_points=%s tracked_symbols=%s",
                removed_trades,
                removed_points,
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

    def _normalize_trades(self, symbol: str, raw_trades: list[Any]) -> list[CvdTrade]:
        normalized: list[CvdTrade] = []

        for raw in raw_trades:
            trade = self._parse_trade(symbol, raw)
            if trade is None:
                continue
            normalized.append(trade)

        normalized.sort(key=lambda t: (t.timestamp, t.trade_id or ""))

        return normalized

    def _parse_trade(self, symbol: str, raw: Any) -> Optional[CvdTrade]:
        try:
            if isinstance(raw, CvdTrade):
                return raw

            if isinstance(raw, dict):
                price = self._safe_float(raw.get("price", raw.get("p")))
                quantity = self._safe_float(
                    raw.get("quantity", raw.get("qty", raw.get("size", raw.get("q"))))
                )
                timestamp = self._safe_float(
                    raw.get("timestamp", raw.get("ts", raw.get("time", time.time())))
                )
                side = self._extract_side_from_dict(raw)

                if side is None or price is None or quantity is None or timestamp is None:
                    return None

                if price <= 0 or quantity <= 0:
                    return None

                trade_id = raw.get("trade_id", raw.get("id"))
                exchange = raw.get("exchange")

                return CvdTrade(
                    symbol=str(raw.get("symbol", symbol)),
                    side=side,
                    price=price,
                    quantity=quantity,
                    notional=price * quantity,
                    timestamp=timestamp,
                    trade_id=str(trade_id) if trade_id is not None else None,
                    exchange=str(exchange) if exchange is not None else None,
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

            return CvdTrade(
                symbol=str(symbol_value),
                side=side,
                price=price,
                quantity=quantity,
                notional=price * quantity,
                timestamp=timestamp,
                trade_id=str(trade_id) if trade_id is not None else None,
                exchange=str(exchange) if exchange is not None else None,
            )
        except Exception:
            return None

    def _filter_new_trades(self, symbol: str, trades: list[CvdTrade]) -> list[CvdTrade]:
        if not trades:
            return []

        last_seen_key = self._last_seen_trade_key_by_symbol.get(symbol)
        result: list[CvdTrade] = []

        for trade in trades:
            key = self._build_trade_key(trade)
            if last_seen_key is not None and key <= last_seen_key:
                continue
            result.append(trade)

        if result:
            self._last_seen_trade_key_by_symbol[symbol] = self._build_trade_key(result[-1])

        return result

    def _build_trade_key(self, trade: CvdTrade) -> str:
        trade_id = trade.trade_id or ""
        return f"{trade.timestamp:.9f}:{trade_id}:{trade.price:.12f}:{trade.quantity:.12f}:{trade.side}"

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

    def _prune_old_cvd_points(self, symbol: str) -> int:
        points = self._cvd_points_by_symbol.get(symbol)
        if not points:
            return 0

        min_ts = time.time() - self._config.window_seconds
        removed = 0

        while points and points[0].timestamp < min_ts:
            points.popleft()
            removed += 1

        if not points:
            self._cvd_points_by_symbol.pop(symbol, None)

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