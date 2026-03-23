from __future__ import annotations

import asyncio
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.event_bus import Event, EventBus, EventPriority
from core.logger import get_logger


@dataclass(slots=True)
class ImbalanceLevelStats:
    bid_volume: float
    ask_volume: float
    imbalance_ratio: float
    imbalance_diff: float
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    mid_price: Optional[float]
    depth_levels_used: int
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class OrderbookImbalanceSignal:
    symbol: str
    side: str
    strength: float
    imbalance_ratio: float
    imbalance_diff: float
    spread: Optional[float]
    mid_price: Optional[float]
    best_bid: Optional[float]
    best_ask: Optional[float]
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class OrderbookImbalanceConfig:
    enabled: bool = True
    depth_levels: int = 10
    min_total_volume: float = 0.0
    bullish_ratio_threshold: float = 0.60
    bearish_ratio_threshold: float = 0.40
    min_signal_interval_sec: float = 0.30
    emit_updates: bool = True
    emit_signals: bool = True
    normalize_ratio_to_minus_one_one: bool = False
    smooth_window: int = 5
    symbol_allowlist: Optional[set[str]] = None
    publish_priority: EventPriority = EventPriority.NORMAL

    update_topic: str = "analytics.orderbook.imbalance.updated"
    signal_topic: str = "analytics.orderbook.imbalance.signal"

    source_name: str = "orderbook_imbalance"

    @classmethod
    def from_app_config(cls, app_config: Any) -> "OrderbookImbalanceConfig":
        """
        Адаптаційний конструктор під твій Config.
        Тут спеціально м’яка логіка, щоб клас не ламався,
        навіть якщо структура конфігів трохи відрізняється.
        """
        analytics_cfg = getattr(app_config, "analytics", None)
        orderflow_cfg = getattr(analytics_cfg, "orderflow", None) if analytics_cfg else None
        imbalance_cfg = getattr(orderflow_cfg, "orderbook_imbalance", None) if orderflow_cfg else None

        if imbalance_cfg is None:
            return cls()

        return cls(
            enabled=getattr(imbalance_cfg, "enabled", True),
            depth_levels=getattr(imbalance_cfg, "depth_levels", 10),
            min_total_volume=getattr(imbalance_cfg, "min_total_volume", 0.0),
            bullish_ratio_threshold=getattr(imbalance_cfg, "bullish_ratio_threshold", 0.60),
            bearish_ratio_threshold=getattr(imbalance_cfg, "bearish_ratio_threshold", 0.40),
            min_signal_interval_sec=getattr(imbalance_cfg, "min_signal_interval_sec", 0.30),
            emit_updates=getattr(imbalance_cfg, "emit_updates", True),
            emit_signals=getattr(imbalance_cfg, "emit_signals", True),
            normalize_ratio_to_minus_one_one=getattr(
                imbalance_cfg,
                "normalize_ratio_to_minus_one_one",
                False,
            ),
            smooth_window=getattr(imbalance_cfg, "smooth_window", 5),
            symbol_allowlist=set(getattr(imbalance_cfg, "symbol_allowlist", []) or []),
            publish_priority=getattr(
                imbalance_cfg,
                "publish_priority",
                EventPriority.NORMAL,
            ),
            update_topic=getattr(
                imbalance_cfg,
                "update_topic",
                "analytics.orderbook.imbalance.updated",
            ),
            signal_topic=getattr(
                imbalance_cfg,
                "signal_topic",
                "analytics.orderbook.imbalance.signal",
            ),
            source_name=getattr(
                imbalance_cfg,
                "source_name",
                "orderbook_imbalance",
            ),
        )


class OrderbookImbalance:
    """
    Orderbook imbalance analytics module.

    Призначення:
    - бере snapshot зі стакану
    - рахує bid/ask imbalance
    - за потреби згладжує значення
    - публікує update та signal події в EventBus

    Очікувана інтеграція:
    - source events:
        market.orderbook.updated
        market.orderbook.snapshot
    - emitted events:
        analytics.orderbook.imbalance.updated
        analytics.orderbook.imbalance.signal
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        orderbook_cache: Any,
        config: Optional[OrderbookImbalanceConfig] = None,
        app_config: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        source_topic_patterns: Optional[list[str]] = None,
    ) -> None:
        self._event_bus = event_bus
        self._orderbook_cache = orderbook_cache
        self._scheduler = scheduler
        self._config = config or (
            OrderbookImbalanceConfig.from_app_config(app_config)
            if app_config is not None
            else OrderbookImbalanceConfig()
        )

        self._source_topic_patterns = source_topic_patterns or [
            "market.orderbook.updated",
            "market.orderbook.snapshot",
            "orderbook.updated",
            "orderbook.*",
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

        self._last_stats_by_symbol: dict[str, ImbalanceLevelStats] = {}
        self._last_signal_ts_by_symbol: dict[str, float] = {}
        self._ratio_history_by_symbol: dict[str, list[float]] = {}

        self._metrics: dict[str, Any] = {
            "processed": 0,
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
            self._logger.warning("OrderbookImbalance already started")
            return

        if not self._config.enabled:
            self._logger.warning("OrderbookImbalance is disabled by config")
            return

        for pattern in self._source_topic_patterns:
            subscription = self._event_bus.subscribe(
                pattern=pattern,
                handler=self._handle_orderbook_event,
                name=f"{self.__class__.__name__}:{pattern}",
            )
            self._subscriptions.append(subscription)

        self._register_scheduler_jobs()

        self._running = True
        self._logger.info(
            "OrderbookImbalance started | depth_levels=%s bullish_threshold=%.4f bearish_threshold=%.4f",
            self._config.depth_levels,
            self._config.bullish_ratio_threshold,
            self._config.bearish_ratio_threshold,
        )

    def stop(self) -> None:
        if not self._running:
            self._logger.warning("OrderbookImbalance already stopped")
            return

        for sub in self._subscriptions:
            try:
                self._event_bus.unsubscribe(sub)
            except Exception:
                self._logger.exception("Failed to unsubscribe OrderbookImbalance handler")

        self._subscriptions.clear()
        self._running = False

        self._logger.info("OrderbookImbalance stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_symbol(self, symbol: str) -> Optional[ImbalanceLevelStats]:
        if not self._should_process_symbol(symbol):
            self._inc_metric("skipped", symbol)
            return None

        async with self._lock:
            try:
                snapshot = await self._get_orderbook_snapshot(symbol)
                if snapshot is None:
                    self._logger.debug("No orderbook snapshot available | symbol=%s", symbol)
                    self._inc_metric("skipped", symbol)
                    return None

                stats = self._calculate_imbalance(snapshot)
                if stats is None:
                    self._inc_metric("skipped", symbol)
                    return None

                stats = self._apply_smoothing(symbol, stats)
                self._last_stats_by_symbol[symbol] = stats
                self._inc_metric("processed", symbol)

                if self._config.emit_updates:
                    await self._emit_update(symbol, stats)

                if self._config.emit_signals:
                    signal = self._build_signal(symbol, stats)
                    if signal is not None:
                        await self._emit_signal(signal)

                return stats

            except Exception:
                self._inc_metric("errors", symbol)
                self._logger.exception(
                    "Failed to process orderbook imbalance | symbol=%s",
                    symbol,
                )
                return None

    def get_latest_stats(self, symbol: str) -> Optional[ImbalanceLevelStats]:
        return self._last_stats_by_symbol.get(symbol)

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "config": {
                "enabled": self._config.enabled,
                "depth_levels": self._config.depth_levels,
                "bullish_ratio_threshold": self._config.bullish_ratio_threshold,
                "bearish_ratio_threshold": self._config.bearish_ratio_threshold,
                "smooth_window": self._config.smooth_window,
            },
            "symbols_tracked": len(self._last_stats_by_symbol),
            "processed": self._metrics["processed"],
            "signals_emitted": self._metrics["signals_emitted"],
            "updates_emitted": self._metrics["updates_emitted"],
            "skipped": self._metrics["skipped"],
            "errors": self._metrics["errors"],
            "symbols": dict(self._metrics["symbols"]),
        }

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _handle_orderbook_event(self, event: Event) -> None:
        symbol = self._extract_symbol_from_event(event)
        if not symbol:
            self._logger.debug(
                "Orderbook event without symbol skipped | topic=%s event_id=%s",
                event.topic,
                event.event_id,
            )
            self._inc_metric("skipped")
            return

        await self.process_symbol(symbol)

    # ------------------------------------------------------------------
    # Core calculation
    # ------------------------------------------------------------------

    def _calculate_imbalance(self, snapshot: Any) -> Optional[ImbalanceLevelStats]:
        bids = self._extract_side(snapshot, "bids")
        asks = self._extract_side(snapshot, "asks")

        if not bids or not asks:
            return None

        bid_levels = bids[: self._config.depth_levels]
        ask_levels = asks[: self._config.depth_levels]

        bid_volume = sum(size for _, size in bid_levels if size > 0)
        ask_volume = sum(size for _, size in ask_levels if size > 0)
        total_volume = bid_volume + ask_volume

        if total_volume <= 0:
            return None

        if total_volume < self._config.min_total_volume:
            return None

        best_bid = bid_levels[0][0] if bid_levels else None
        best_ask = ask_levels[0][0] if ask_levels else None

        spread = None
        mid_price = None
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2.0

        imbalance_ratio = bid_volume / total_volume
        imbalance_diff = (bid_volume - ask_volume) / total_volume

        if self._config.normalize_ratio_to_minus_one_one:
            imbalance_ratio = (imbalance_ratio * 2.0) - 1.0

        return ImbalanceLevelStats(
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance_ratio=imbalance_ratio,
            imbalance_diff=imbalance_diff,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            mid_price=mid_price,
            depth_levels_used=min(
                self._config.depth_levels,
                len(bid_levels),
                len(ask_levels),
            ),
        )

    def _build_signal(
        self,
        symbol: str,
        stats: ImbalanceLevelStats,
    ) -> Optional[OrderbookImbalanceSignal]:
        now = time.time()
        last_signal_ts = self._last_signal_ts_by_symbol.get(symbol, 0.0)

        if now - last_signal_ts < self._config.min_signal_interval_sec:
            return None

        ratio = stats.imbalance_ratio
        strength = abs(stats.imbalance_diff)

        if ratio >= self._config.bullish_ratio_threshold:
            self._last_signal_ts_by_symbol[symbol] = now
            return OrderbookImbalanceSignal(
                symbol=symbol,
                side="bullish",
                strength=strength,
                imbalance_ratio=stats.imbalance_ratio,
                imbalance_diff=stats.imbalance_diff,
                spread=stats.spread,
                mid_price=stats.mid_price,
                best_bid=stats.best_bid,
                best_ask=stats.best_ask,
                reason="bid_pressure_dominates",
            )

        if ratio <= self._config.bearish_ratio_threshold:
            self._last_signal_ts_by_symbol[symbol] = now
            return OrderbookImbalanceSignal(
                symbol=symbol,
                side="bearish",
                strength=strength,
                imbalance_ratio=stats.imbalance_ratio,
                imbalance_diff=stats.imbalance_diff,
                spread=stats.spread,
                mid_price=stats.mid_price,
                best_bid=stats.best_bid,
                best_ask=stats.best_ask,
                reason="ask_pressure_dominates",
            )

        return None

    # ------------------------------------------------------------------
    # Emitters
    # ------------------------------------------------------------------

    async def _emit_update(self, symbol: str, stats: ImbalanceLevelStats) -> None:
        payload = {
            "symbol": symbol,
            "bid_volume": stats.bid_volume,
            "ask_volume": stats.ask_volume,
            "imbalance_ratio": stats.imbalance_ratio,
            "imbalance_diff": stats.imbalance_diff,
            "best_bid": stats.best_bid,
            "best_ask": stats.best_ask,
            "spread": stats.spread,
            "mid_price": stats.mid_price,
            "depth_levels_used": stats.depth_levels_used,
            "timestamp": stats.timestamp,
        }

        accepted = await self._event_bus.emit(
            topic=self._config.update_topic,
            payload=payload,
            priority=self._config.publish_priority,
            source=self._config.source_name,
            headers={"symbol": symbol, "analytics_type": "orderbook_imbalance"},
        )

        if accepted:
            self._inc_metric("updates_emitted", symbol)

    async def _emit_signal(self, signal: OrderbookImbalanceSignal) -> None:
        payload = {
            "symbol": signal.symbol,
            "side": signal.side,
            "strength": signal.strength,
            "imbalance_ratio": signal.imbalance_ratio,
            "imbalance_diff": signal.imbalance_diff,
            "spread": signal.spread,
            "mid_price": signal.mid_price,
            "best_bid": signal.best_bid,
            "best_ask": signal.best_ask,
            "reason": signal.reason,
            "timestamp": signal.timestamp,
        }

        accepted = await self._event_bus.emit(
            topic=self._config.signal_topic,
            payload=payload,
            priority=EventPriority.HIGH,
            source=self._config.source_name,
            headers={
                "symbol": signal.symbol,
                "signal_type": "orderbook_imbalance",
                "side": signal.side,
            },
        )

        if accepted:
            self._inc_metric("signals_emitted", signal.symbol)
            self._logger.info(
                "Orderbook imbalance signal emitted | symbol=%s side=%s strength=%.4f ratio=%.4f",
                signal.symbol,
                signal.side,
                signal.strength,
                signal.imbalance_ratio,
            )

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            return

        try:
            if hasattr(self._scheduler, "add_interval_job"):
                self._scheduler.add_interval_job(
                    name="orderbook_imbalance_health",
                    interval_seconds=30,
                    func=self._log_health_snapshot,
                    run_immediately=False,
                )
                self._logger.info("Scheduler job registered for OrderbookImbalance")
        except Exception:
            self._logger.exception("Failed to register scheduler jobs for OrderbookImbalance")

    async def _log_health_snapshot(self) -> None:
        self._logger.info(
            "OrderbookImbalance health | running=%s processed=%s signals=%s errors=%s tracked_symbols=%s",
            self._running,
            self._metrics["processed"],
            self._metrics["signals_emitted"],
            self._metrics["errors"],
            len(self._last_stats_by_symbol),
        )

    # ------------------------------------------------------------------
    # Snapshot access
    # ------------------------------------------------------------------

    async def _get_orderbook_snapshot(self, symbol: str) -> Optional[Any]:
        """
        Адаптація під можливі API твого orderbook_cache.

        Підтримані варіанти:
        - async get_snapshot(symbol)
        - sync get_snapshot(symbol)
        - async get(symbol)
        - sync get(symbol)
        - direct dict-like storage
        """
        cache = self._orderbook_cache

        for method_name in ("get_snapshot", "get", "get_orderbook", "get_book"):
            method = getattr(cache, method_name, None)
            if method is None:
                continue

            result = method(symbol)
            if asyncio.iscoroutine(result):
                return await result
            return result

        if isinstance(cache, dict):
            return cache.get(symbol)

        return None

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _extract_symbol_from_event(self, event: Event) -> Optional[str]:
        payload = event.payload

        if isinstance(payload, dict):
            symbol = payload.get("symbol") or payload.get("instrument") or payload.get("pair")
            if symbol:
                return str(symbol)

        symbol_from_headers = event.headers.get("symbol") if event.headers else None
        if symbol_from_headers:
            return str(symbol_from_headers)

        return None

    def _extract_side(self, snapshot: Any, side: str) -> list[tuple[float, float]]:
        raw_levels: Any = None

        if isinstance(snapshot, dict):
            raw_levels = snapshot.get(side)
        else:
            raw_levels = getattr(snapshot, side, None)

        if raw_levels is None:
            return []

        parsed: list[tuple[float, float]] = []

        for level in raw_levels:
            price, size = self._parse_level(level)
            if price is None or size is None:
                continue
            if not math.isfinite(price) or not math.isfinite(size):
                continue
            if size <= 0:
                continue
            parsed.append((price, size))

        if side == "bids":
            parsed.sort(key=lambda x: x[0], reverse=True)
        else:
            parsed.sort(key=lambda x: x[0])

        return parsed

    def _parse_level(self, level: Any) -> tuple[Optional[float], Optional[float]]:
        """
        Підтримує:
        - [price, size]
        - (price, size)
        - {"price": ..., "size": ...}
        - {"price": ..., "quantity": ...}
        - object.price / object.size
        """
        try:
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                return float(level[0]), float(level[1])

            if isinstance(level, dict):
                price = level.get("price")
                size = level.get("size", level.get("qty", level.get("quantity")))
                if price is None or size is None:
                    return None, None
                return float(price), float(size)

            price = getattr(level, "price", None)
            size = getattr(level, "size", getattr(level, "qty", getattr(level, "quantity", None)))
            if price is None or size is None:
                return None, None
            return float(price), float(size)
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def _apply_smoothing(
        self,
        symbol: str,
        stats: ImbalanceLevelStats,
    ) -> ImbalanceLevelStats:
        window = max(1, self._config.smooth_window)
        if window <= 1:
            return stats

        history = self._ratio_history_by_symbol.setdefault(symbol, [])
        history.append(stats.imbalance_ratio)

        if len(history) > window:
            del history[0 : len(history) - window]

        if len(history) < 2:
            return stats

        smoothed_ratio = statistics.fmean(history)

        # imbalance_diff теж синхронно згладжуємо через перетворення,
        # але не чіпаємо volume/spread/raw fields.
        if self._config.normalize_ratio_to_minus_one_one:
            smoothed_diff = smoothed_ratio
        else:
            smoothed_diff = (smoothed_ratio * 2.0) - 1.0

        return ImbalanceLevelStats(
            bid_volume=stats.bid_volume,
            ask_volume=stats.ask_volume,
            imbalance_ratio=smoothed_ratio,
            imbalance_diff=smoothed_diff,
            best_bid=stats.best_bid,
            best_ask=stats.best_ask,
            spread=stats.spread,
            mid_price=stats.mid_price,
            depth_levels_used=stats.depth_levels_used,
            timestamp=stats.timestamp,
        )

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _should_process_symbol(self, symbol: str) -> bool:
        allowlist = self._config.symbol_allowlist
        if not allowlist:
            return True
        return symbol in allowlist

    def _inc_metric(self, key: str, symbol: Optional[str] = None, amount: int = 1) -> None:
        self._metrics[key] = self._metrics.get(key, 0) + amount

        if symbol:
            symbols = self._metrics.setdefault("symbols", {})
            symbol_stats = symbols.setdefault(
                symbol,
                {
                    "processed": 0,
                    "signals_emitted": 0,
                    "updates_emitted": 0,
                    "skipped": 0,
                    "errors": 0,
                },
            )
            if key in symbol_stats:
                symbol_stats[key] += amount