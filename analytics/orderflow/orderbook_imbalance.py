from __future__ import annotations

import asyncio
import math
import statistics
import time
from typing import Any, Optional

from core.event_bus import Event, EventBus

from .base import BaseOrderFlowAnalyzer
from .config import OrderbookImbalanceConfig
from .enums import (
    OrderFlowMetricType,
    OrderFlowSide,
    OrderFlowSignalType,
    OrderFlowSourceType,
)
from .models import (
    BaseOrderFlowStats,
    OrderbookImbalanceStats,
    OrderbookLevel,
    OrderbookSnapshot,
)


class OrderbookImbalanceAnalyzer(BaseOrderFlowAnalyzer):
    """
    Analyzer для оцінки orderbook imbalance.

    Основні задачі:
    - приймає orderbook events через EventBus
    - читає snapshot зі стакану з orderbook_cache
    - нормалізує snapshot
    - рахує bid/ask imbalance на заданій глибині
    - за потреби згладжує imbalance ratio
    - публікує update/signal події
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
        self._orderbook_cache = orderbook_cache
        self._config = config or (
            OrderbookImbalanceConfig.from_app_config(app_config)
            if app_config is not None
            else OrderbookImbalanceConfig()
        )

        super().__init__(
            event_bus=event_bus,
            config=self._config,
            metric_type=OrderFlowMetricType.ORDERBOOK_IMBALANCE,
            source_type=OrderFlowSourceType.ORDERBOOK,
            scheduler=scheduler,
            source_topic_patterns=source_topic_patterns or [
                "market.orderbook.updated",
                "market.orderbook.snapshot",
                "orderbook.updated",
                "orderbook.*",
            ],
            component_module="orderflow",
        )

        self._state_lock = asyncio.Lock()

        self._last_stats_by_symbol: dict[str, OrderbookImbalanceStats] = {}
        self._ratio_history_by_symbol: dict[str, list[float]] = {}
        self._last_snapshot_ts_by_symbol: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_symbol(self, symbol: str) -> Optional[OrderbookImbalanceStats]:
        symbol = str(symbol).upper()

        if not self.should_process_symbol(symbol):
            self._inc_metric("skipped", symbol)
            return None

        async with self._state_lock:
            try:
                snapshot = await self._get_orderbook_snapshot(symbol)
                if snapshot is None:
                    self._inc_metric("skipped", symbol)
                    return None

                stats = self._calculate_imbalance(snapshot)
                if stats is None:
                    self._inc_metric("skipped", symbol)
                    return None

                stats = self._apply_smoothing(symbol, stats)
                self._last_stats_by_symbol[symbol] = stats
                self._last_snapshot_ts_by_symbol[symbol] = snapshot.timestamp
                self._inc_metric("processed", symbol)

                await self.emit_update(stats)

                signal = self._build_signal(stats)
                if signal is not None:
                    await self.emit_signal(signal)

                return stats

            except Exception:
                self._inc_metric("errors", symbol)
                self._logger.exception(
                    "Failed to process orderbook imbalance | symbol=%s",
                    symbol,
                )
                return None

    def get_latest_stats(self, symbol: str) -> Optional[BaseOrderFlowStats]:
        return self._last_stats_by_symbol.get(str(symbol).upper())

    def stats(self) -> dict[str, Any]:
        base_stats = super().stats()
        base_stats["config"].update(
            {
                "depth_levels": self._config.depth_levels,
                "min_total_volume": self._config.min_total_volume,
                "bullish_ratio_threshold": self._config.bullish_ratio_threshold,
                "bearish_ratio_threshold": self._config.bearish_ratio_threshold,
                "normalize_ratio_to_minus_one_one": self._config.normalize_ratio_to_minus_one_one,
                "smooth_window": self._config.smooth_window,
            }
        )
        base_stats["tracked_symbols"] = len(self._last_stats_by_symbol)
        return base_stats

    async def cleanup(self) -> None:
        now = time.time()
        max_age = max(
            float(self._config.cleanup_interval_sec) * 3.0,
            60.0,
        )

        for symbol, ts in list(self._last_snapshot_ts_by_symbol.items()):
            if (now - ts) > max_age:
                self._last_stats_by_symbol.pop(symbol, None)
                self._ratio_history_by_symbol.pop(symbol, None)
                self._last_snapshot_ts_by_symbol.pop(symbol, None)
                self._last_signal_ts_by_symbol.pop(symbol, None)

                self._logger.debug(
                    "Removed stale orderbook imbalance state | symbol=%s max_age=%s",
                    symbol,
                    max_age,
                )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, event: Event) -> None:
        symbol = self.extract_symbol_from_event(event)
        if not symbol:
            self._logger.debug(
                "Orderbook event without symbol skipped | topic=%s event_id=%s",
                getattr(event, "topic", None),
                getattr(event, "event_id", None),
            )
            self._inc_metric("skipped")
            return

        await self.process_symbol(symbol)

    # ------------------------------------------------------------------
    # Internal data loading
    # ------------------------------------------------------------------

    async def _get_orderbook_snapshot(self, symbol: str) -> Optional[OrderbookSnapshot]:
        cache = self._orderbook_cache
        if cache is None:
            return None

        candidates = [
            ("get_snapshot", {"symbol": symbol}),
            ("get_orderbook", {"symbol": symbol}),
            ("get", {"symbol": symbol}),
        ]

        for method_name, kwargs in candidates:
            method = getattr(cache, method_name, None)
            if method is None:
                continue

            try:
                result = method(**kwargs)
                if asyncio.iscoroutine(result):
                    result = await result

                snapshot = self._extract_snapshot_from_result(result, symbol)
                if snapshot is not None:
                    return snapshot

            except TypeError:
                try:
                    result = method(symbol)
                    if asyncio.iscoroutine(result):
                        result = await result

                    snapshot = self._extract_snapshot_from_result(result, symbol)
                    if snapshot is not None:
                        return snapshot
                except Exception:
                    self._logger.exception(
                        "Failed to fetch orderbook snapshot | method=%s symbol=%s",
                        method_name,
                        symbol,
                    )
            except Exception:
                self._logger.exception(
                    "Failed to fetch orderbook snapshot | method=%s symbol=%s",
                    method_name,
                    symbol,
                )

        return None

    def _extract_snapshot_from_result(
        self,
        result: Any,
        symbol: str,
    ) -> Optional[OrderbookSnapshot]:
        if result is None:
            return None

        if isinstance(result, OrderbookSnapshot):
            return result

        if isinstance(result, dict):
            if isinstance(result.get("data"), dict):
                snapshot = self.normalize_orderbook_snapshot(
                    result["data"],
                    default_symbol=symbol,
                )
                if snapshot is not None:
                    return snapshot

            snapshot = self.normalize_orderbook_snapshot(
                result,
                default_symbol=symbol,
            )
            if snapshot is not None:
                return snapshot

        return None

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def _calculate_imbalance(
        self,
        snapshot: OrderbookSnapshot,
    ) -> Optional[OrderbookImbalanceStats]:
        bids = self._prepare_side(snapshot.bids, reverse=True)
        asks = self._prepare_side(snapshot.asks, reverse=False)

        if not bids or not asks:
            return None

        bid_levels = bids[: self._config.depth_levels]
        ask_levels = asks[: self._config.depth_levels]

        bid_volume = sum(level.size for level in bid_levels if level.size > 0)
        ask_volume = sum(level.size for level in ask_levels if level.size > 0)
        total_volume = bid_volume + ask_volume

        if total_volume <= 0:
            return None

        if total_volume < self._config.min_total_volume:
            return None

        best_bid = bid_levels[0].price if bid_levels else None
        best_ask = ask_levels[0].price if ask_levels else None

        spread: Optional[float] = None
        mid_price: Optional[float] = None

        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2.0

        imbalance_ratio = bid_volume / total_volume
        imbalance_diff = (bid_volume - ask_volume) / total_volume

        if self._config.normalize_ratio_to_minus_one_one:
            imbalance_ratio = (imbalance_ratio * 2.0) - 1.0

        return OrderbookImbalanceStats(
            symbol=snapshot.symbol,
            metric=OrderFlowMetricType.ORDERBOOK_IMBALANCE,
            source_type=OrderFlowSourceType.ORDERBOOK,
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

    def _prepare_side(
        self,
        levels: list[OrderbookLevel],
        *,
        reverse: bool,
    ) -> list[OrderbookLevel]:
        filtered = [level for level in levels if level.size > 0 and level.price > 0]
        filtered.sort(key=lambda item: item.price, reverse=reverse)
        return filtered

    def _apply_smoothing(
        self,
        symbol: str,
        stats: OrderbookImbalanceStats,
    ) -> OrderbookImbalanceStats:
        window = max(int(self._config.smooth_window), 1)
        if window <= 1:
            return stats

        history = self._ratio_history_by_symbol.setdefault(symbol, [])
        history.append(stats.imbalance_ratio)

        if len(history) > window:
            del history[:-window]

        smoothed_ratio = statistics.fmean(history) if history else stats.imbalance_ratio

        if self._config.normalize_ratio_to_minus_one_one:
            smoothed_diff = smoothed_ratio
        else:
            smoothed_diff = (smoothed_ratio * 2.0) - 1.0

        return OrderbookImbalanceStats(
            symbol=stats.symbol,
            metric=stats.metric,
            source_type=stats.source_type,
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
    # Signal logic
    # ------------------------------------------------------------------

    def _build_signal(self, stats: OrderbookImbalanceStats):
        ratio_for_signal = self._denormalize_ratio_if_needed(stats.imbalance_ratio)

        bullish_ok = ratio_for_signal >= self._config.bullish_ratio_threshold
        bearish_ok = ratio_for_signal <= self._config.bearish_ratio_threshold

        if bullish_ok and not bearish_ok:
            strength = self._calculate_bullish_strength(ratio_for_signal, stats)
            return self.build_signal(
                symbol=stats.symbol,
                signal_type=OrderFlowSignalType.BULLISH,
                side=OrderFlowSide.BUY,
                strength=strength,
                reason="orderbook_bid_imbalance",
                context={
                    "imbalance_ratio": stats.imbalance_ratio,
                    "imbalance_diff": stats.imbalance_diff,
                    "bid_volume": stats.bid_volume,
                    "ask_volume": stats.ask_volume,
                    "spread": stats.spread,
                    "mid_price": stats.mid_price,
                    "best_bid": stats.best_bid,
                    "best_ask": stats.best_ask,
                    "depth_levels_used": stats.depth_levels_used,
                },
            )

        if bearish_ok and not bullish_ok:
            strength = self._calculate_bearish_strength(ratio_for_signal, stats)
            return self.build_signal(
                symbol=stats.symbol,
                signal_type=OrderFlowSignalType.BEARISH,
                side=OrderFlowSide.SELL,
                strength=strength,
                reason="orderbook_ask_imbalance",
                context={
                    "imbalance_ratio": stats.imbalance_ratio,
                    "imbalance_diff": stats.imbalance_diff,
                    "bid_volume": stats.bid_volume,
                    "ask_volume": stats.ask_volume,
                    "spread": stats.spread,
                    "mid_price": stats.mid_price,
                    "best_bid": stats.best_bid,
                    "best_ask": stats.best_ask,
                    "depth_levels_used": stats.depth_levels_used,
                },
            )

        return None

    def _calculate_bullish_strength(
        self,
        ratio_for_signal: float,
        stats: OrderbookImbalanceStats,
    ) -> float:
        components: list[float] = []

        if self._config.bullish_ratio_threshold > 0:
            components.append(
                self._safe_ratio(
                    ratio_for_signal,
                    self._config.bullish_ratio_threshold,
                )
            )
        else:
            components.append(self._normalize_signed_magnitude(ratio_for_signal))

        components.append(self._normalize_signed_magnitude(max(stats.imbalance_diff, 0.0)))

        if stats.spread is not None and stats.mid_price and stats.mid_price > 0:
            spread_pct = stats.spread / stats.mid_price
            components.append(max(0.0, 1.0 - min(spread_pct * 1000.0, 1.0)))
        else:
            components.append(0.5)

        return self._normalize_strength(components)

    def _calculate_bearish_strength(
        self,
        ratio_for_signal: float,
        stats: OrderbookImbalanceStats,
    ) -> float:
        components: list[float] = []

        bearish_distance = 1.0 - ratio_for_signal
        bearish_threshold = 1.0 - self._config.bearish_ratio_threshold

        if bearish_threshold > 0:
            components.append(
                self._safe_ratio(
                    bearish_distance,
                    bearish_threshold,
                )
            )
        else:
            components.append(self._normalize_signed_magnitude(bearish_distance))

        components.append(self._normalize_signed_magnitude(abs(min(stats.imbalance_diff, 0.0))))

        if stats.spread is not None and stats.mid_price and stats.mid_price > 0:
            spread_pct = stats.spread / stats.mid_price
            components.append(max(0.0, 1.0 - min(spread_pct * 1000.0, 1.0)))
        else:
            components.append(0.5)

        return self._normalize_strength(components)

    def _denormalize_ratio_if_needed(self, value: float) -> float:
        if self._config.normalize_ratio_to_minus_one_one:
            return (value + 1.0) / 2.0
        return value

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    def _safe_ratio(self, value: float, threshold: float) -> float:
        if threshold == 0:
            return self._normalize_signed_magnitude(value)
        return max(0.0, value / threshold)

    def _normalize_signed_magnitude(self, value: float) -> float:
        return math.log1p(abs(value))

    def _normalize_strength(self, values: list[float]) -> float:
        if not values:
            return 0.0

        raw = sum(max(0.0, value) for value in values) / len(values)
        return max(0.0, min(raw / (1.0 + raw), 1.0))