from __future__ import annotations

import asyncio
import math
import statistics
import time
from typing import Any

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from .base import BaseOrderFlowAnalyzer
from .config import OrderbookImbalanceConfig
from .enums import (
    ORDERBOOK_INPUT_TOPICS,
    OrderFlowMetricType,
    OrderFlowSide,
    OrderFlowSignalType,
    OrderFlowSourceType,
)
from .models import (
    BaseOrderFlowStats,
    OrderFlowSignal,
    OrderbookImbalanceStats,
    OrderbookLevel,
    OrderbookSnapshot,
)


class OrderbookImbalanceAnalyzer(BaseOrderFlowAnalyzer):
    """
    Orderbook imbalance analyzer for analytics.orderflow.

    Responsibilities:
    - consume orderbook events through EventBus subscriptions registered in base
    - read snapshots from orderbook_cache
    - normalize snapshots through BaseOrderFlowAnalyzer helpers
    - calculate bid/ask imbalance at configured depth
    - optionally smooth imbalance ratio
    - emit analytics.orderflow.orderbook_imbalance.updated and
      analytics.orderflow.orderbook_imbalance.signal
    - use Scheduler-injected cleanup/health jobs from BaseOrderFlowAnalyzer
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        orderbook_cache: Any,
        config: OrderbookImbalanceConfig | None = None,
        scheduler: Scheduler | None = None,
        source_topic_patterns: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._orderbook_cache = orderbook_cache
        self._config = config or OrderbookImbalanceConfig()

        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=self._config,
            metric_type=OrderFlowMetricType.ORDERBOOK_IMBALANCE,
            source_type=OrderFlowSourceType.ORDERBOOK,
            source_topic_patterns=source_topic_patterns or ORDERBOOK_INPUT_TOPICS,
            component_module="orderflow",
        )

        self._state_lock = asyncio.Lock()

        self._last_stats_by_symbol: dict[str, OrderbookImbalanceStats] = {}
        self._ratio_history_by_symbol: dict[str, list[float]] = {}
        self._last_snapshot_ts_by_symbol: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_symbol(self, symbol: str) -> OrderbookImbalanceStats | None:
        normalized_symbol = str(symbol).strip().upper()

        if not self.should_process_symbol(normalized_symbol):
            self._inc_metric("skipped", normalized_symbol)
            return None

        async with self._state_lock:
            try:
                snapshot = await self._get_orderbook_snapshot(normalized_symbol)
                if snapshot is None:
                    self._inc_metric("skipped", normalized_symbol)
                    return None

                stats = self._calculate_imbalance(snapshot)
                if stats is None:
                    self._inc_metric("skipped", normalized_symbol)
                    return None

                stats = self._apply_smoothing(normalized_symbol, stats)

                self._last_stats_by_symbol[normalized_symbol] = stats
                self._last_snapshot_ts_by_symbol[normalized_symbol] = snapshot.timestamp
                self._inc_metric("processed", normalized_symbol)

                await self.emit_update(stats)

                signal = self._build_signal(stats)
                if signal is not None:
                    await self.emit_signal(signal)

                return stats

            except Exception:
                self._inc_metric("errors", normalized_symbol)
                self._logger.exception(
                    "Failed to process orderbook imbalance | symbol=%s",
                    normalized_symbol,
                )
                return None

    def get_latest_stats(self, symbol: str) -> BaseOrderFlowStats | None:
        return self._last_stats_by_symbol.get(str(symbol).strip().upper())

    def stats(self) -> dict[str, Any]:
        base_stats = super().stats()
        base_stats["config"].update(
            {
                "depth_levels": self._config.depth_levels,
                "min_total_volume": self._config.min_total_volume,
                "bullish_ratio_threshold": self._config.bullish_ratio_threshold,
                "bearish_ratio_threshold": self._config.bearish_ratio_threshold,
                "normalize_ratio_to_minus_one_one": (
                    self._config.normalize_ratio_to_minus_one_one
                ),
                "smooth_window": self._config.smooth_window,
            }
        )
        base_stats["tracked_symbols"] = len(self._last_stats_by_symbol)
        return base_stats

    async def cleanup(self) -> None:
        now = time.time()
        max_age = max(float(self._config.cleanup_interval_sec) * 3.0, 60.0)

        for symbol, snapshot_ts in list(self._last_snapshot_ts_by_symbol.items()):
            if (now - snapshot_ts) <= max_age:
                continue

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
                event.topic,
                event.event_id,
            )
            self._inc_metric("skipped")
            return

        await self.process_symbol(symbol)

    # ------------------------------------------------------------------
    # Internal data loading
    # ------------------------------------------------------------------

    async def _get_orderbook_snapshot(self, symbol: str) -> OrderbookSnapshot | None:
        if self._orderbook_cache is None:
            return None

        candidates: tuple[tuple[str, dict[str, Any]], ...] = (
            ("get_snapshot", {"symbol": symbol}),
            ("get_orderbook", {"symbol": symbol}),
            ("get", {"symbol": symbol}),
        )

        for method_name, kwargs in candidates:
            method = getattr(self._orderbook_cache, method_name, None)
            if method is None:
                continue

            result = await self._call_cache_method(
                method=method,
                method_name=method_name,
                symbol=symbol,
                kwargs=kwargs,
            )
            snapshot = self._extract_snapshot_from_cache_result(result, symbol)
            if snapshot is not None:
                return snapshot

        return None

    async def _call_cache_method(
        self,
        *,
        method: Any,
        method_name: str,
        symbol: str,
        kwargs: dict[str, Any],
    ) -> Any:
        try:
            result = method(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        except TypeError:
            try:
                result = method(symbol)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except Exception:
                self._logger.exception(
                    "Failed to fetch orderbook snapshot | method=%s symbol=%s",
                    method_name,
                    symbol,
                )
                return None

        except Exception:
            self._logger.exception(
                "Failed to fetch orderbook snapshot | method=%s symbol=%s",
                method_name,
                symbol,
            )
            return None

    def _extract_snapshot_from_cache_result(
            self,
            result: Any,
            symbol: str,
    ) -> OrderbookSnapshot | None:
        """
        Extract and normalize OrderbookSnapshot from cache result.

        Supported cache result shapes:
        - OrderbookSnapshot
        - {"data": OrderbookSnapshot}
        - raw orderbook dict
        - {"data": raw orderbook dict}

        The analyzer must accept already-normalized models from the data/cache
        layer, but still enforce symbol matching and valid bid/ask levels.
        """
        if result is None:
            return None

        normalized_symbol = str(symbol).strip().upper()

        if isinstance(result, OrderbookSnapshot):
            return self._normalize_snapshot_model(
                result,
                expected_symbol=normalized_symbol,
            )

        if not isinstance(result, dict):
            return None

        data = result.get("data")

        if isinstance(data, OrderbookSnapshot):
            snapshot = self._normalize_snapshot_model(
                data,
                expected_symbol=normalized_symbol,
            )
            if snapshot is not None:
                return snapshot

        if isinstance(data, dict):
            snapshot = self.normalize_orderbook_snapshot(
                data,
                default_symbol=normalized_symbol,
            )
            if snapshot is not None:
                return self._normalize_snapshot_model(
                    snapshot,
                    expected_symbol=normalized_symbol,
                )

        snapshot = self.normalize_orderbook_snapshot(
            result,
            default_symbol=normalized_symbol,
        )
        if snapshot is not None:
            return self._normalize_snapshot_model(
                snapshot,
                expected_symbol=normalized_symbol,
            )

        return None

    def _normalize_snapshot_model(
            self,
            snapshot: OrderbookSnapshot,
            *,
            expected_symbol: str,
    ) -> OrderbookSnapshot | None:
        """
        Validate and normalize an OrderbookSnapshot model returned by cache.

        This protects the analyzer from:
        - lower-case symbols;
        - snapshots for another symbol;
        - invalid or unsorted levels;
        - empty bid/ask sides after filtering.
        """
        snapshot_symbol = str(snapshot.symbol).strip().upper()
        if not snapshot_symbol:
            return None

        if snapshot_symbol != expected_symbol:
            self._logger.debug(
                "Orderbook snapshot symbol mismatch skipped | expected=%s actual=%s",
                expected_symbol,
                snapshot_symbol,
            )
            return None

        if snapshot.timestamp <= 0:
            return None

        valid_bids = [
            level
            for level in snapshot.bids
            if isinstance(level, OrderbookLevel) and level.is_valid
        ]
        valid_asks = [
            level
            for level in snapshot.asks
            if isinstance(level, OrderbookLevel) and level.is_valid
        ]

        if not valid_bids or not valid_asks:
            return None

        valid_bids.sort(key=lambda item: item.price, reverse=True)
        valid_asks.sort(key=lambda item: item.price)

        return OrderbookSnapshot(
            symbol=snapshot_symbol,
            bids=valid_bids,
            asks=valid_asks,
            timestamp=float(snapshot.timestamp),
            exchange=snapshot.exchange,
            sequence_id=snapshot.sequence_id,
            raw=snapshot.raw,
        )

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def _calculate_imbalance(
        self,
        snapshot: OrderbookSnapshot,
    ) -> OrderbookImbalanceStats | None:
        if not snapshot.is_valid:
            return None

        bids = self._prepare_side(snapshot.bids, reverse=True)
        asks = self._prepare_side(snapshot.asks, reverse=False)

        if not bids or not asks:
            return None

        bid_levels = bids[: self._config.depth_levels]
        ask_levels = asks[: self._config.depth_levels]

        bid_volume = sum(level.size for level in bid_levels)
        ask_volume = sum(level.size for level in ask_levels)
        total_volume = bid_volume + ask_volume

        if total_volume <= 0:
            return None

        if total_volume < self._config.min_total_volume:
            return None

        best_bid = bid_levels[0].price if bid_levels else None
        best_ask = ask_levels[0].price if ask_levels else None

        spread: float | None = None
        mid_price: float | None = None

        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2.0

        raw_ratio = bid_volume / total_volume
        imbalance_diff = (bid_volume - ask_volume) / total_volume
        imbalance_ratio = self._normalize_ratio(raw_ratio)

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
        filtered = [
            level for level in levels
            if level.is_valid
        ]
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

        smoothed_ratio = (
            statistics.fmean(history)
            if history
            else stats.imbalance_ratio
        )

        return OrderbookImbalanceStats(
            symbol=stats.symbol,
            metric=stats.metric,
            source_type=stats.source_type,
            bid_volume=stats.bid_volume,
            ask_volume=stats.ask_volume,
            imbalance_ratio=smoothed_ratio,
            imbalance_diff=self._ratio_to_diff(smoothed_ratio),
            best_bid=stats.best_bid,
            best_ask=stats.best_ask,
            spread=stats.spread,
            mid_price=stats.mid_price,
            depth_levels_used=stats.depth_levels_used,
            timestamp=stats.timestamp,
        )

    def _normalize_ratio(self, raw_ratio: float) -> float:
        if self._config.normalize_ratio_to_minus_one_one:
            return (raw_ratio * 2.0) - 1.0

        return raw_ratio

    def _denormalize_ratio_if_needed(self, value: float) -> float:
        if self._config.normalize_ratio_to_minus_one_one:
            return (value + 1.0) / 2.0

        return value

    def _ratio_to_diff(self, ratio: float) -> float:
        if self._config.normalize_ratio_to_minus_one_one:
            return ratio

        return (ratio * 2.0) - 1.0

    # ------------------------------------------------------------------
    # Signal logic
    # ------------------------------------------------------------------

    def _build_signal(self, stats: OrderbookImbalanceStats) -> OrderFlowSignal | None:
        ratio_for_signal = self._denormalize_ratio_if_needed(stats.imbalance_ratio)

        bullish_ok = ratio_for_signal >= self._config.bullish_ratio_threshold
        bearish_ok = ratio_for_signal <= self._config.bearish_ratio_threshold

        if bullish_ok and not bearish_ok:
            return self.build_signal(
                symbol=stats.symbol,
                signal_type=OrderFlowSignalType.BULLISH,
                side=OrderFlowSide.BUY,
                strength=self._calculate_bullish_strength(ratio_for_signal, stats),
                reason="orderbook_bid_imbalance",
                context=self._build_signal_context(stats),
            )

        if bearish_ok and not bullish_ok:
            return self.build_signal(
                symbol=stats.symbol,
                signal_type=OrderFlowSignalType.BEARISH,
                side=OrderFlowSide.SELL,
                strength=self._calculate_bearish_strength(ratio_for_signal, stats),
                reason="orderbook_ask_imbalance",
                context=self._build_signal_context(stats),
            )

        return None

    def _build_signal_context(self, stats: OrderbookImbalanceStats) -> dict[str, Any]:
        return {
            "imbalance_ratio": stats.imbalance_ratio,
            "imbalance_diff": stats.imbalance_diff,
            "bid_volume": stats.bid_volume,
            "ask_volume": stats.ask_volume,
            "spread": stats.spread,
            "mid_price": stats.mid_price,
            "best_bid": stats.best_bid,
            "best_ask": stats.best_ask,
            "depth_levels_used": stats.depth_levels_used,
        }

    def _calculate_bullish_strength(
        self,
        ratio_for_signal: float,
        stats: OrderbookImbalanceStats,
    ) -> float:
        components: list[float] = [
            self._safe_ratio(
                ratio_for_signal,
                self._config.bullish_ratio_threshold,
            ),
            self._normalize_magnitude(max(stats.imbalance_diff, 0.0)),
            self._spread_quality_component(stats),
        ]

        return self._normalize_strength(components)

    def _calculate_bearish_strength(
        self,
        ratio_for_signal: float,
        stats: OrderbookImbalanceStats,
    ) -> float:
        bearish_distance = 1.0 - ratio_for_signal
        bearish_threshold = 1.0 - self._config.bearish_ratio_threshold

        components: list[float] = [
            self._safe_ratio(
                bearish_distance,
                bearish_threshold,
            ),
            self._normalize_magnitude(abs(min(stats.imbalance_diff, 0.0))),
            self._spread_quality_component(stats),
        ]

        return self._normalize_strength(components)

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    def _spread_quality_component(self, stats: OrderbookImbalanceStats) -> float:
        if stats.spread is None or not stats.mid_price or stats.mid_price <= 0:
            return 0.5

        spread_pct = stats.spread / stats.mid_price
        return max(0.0, 1.0 - min(spread_pct * 1000.0, 1.0))

    def _safe_ratio(self, value: float, threshold: float) -> float:
        if threshold == 0:
            return self._normalize_magnitude(value)

        return max(0.0, value / threshold)

    def _normalize_magnitude(self, value: float) -> float:
        return math.log1p(abs(value))

    def _normalize_strength(self, values: list[float]) -> float:
        if not values:
            return 0.0

        raw = sum(max(0.0, value) for value in values) / len(values)
        return max(0.0, min(raw / (1.0 + raw), 1.0))