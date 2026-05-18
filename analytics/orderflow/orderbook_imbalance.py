from __future__ import annotations

import asyncio
import math
import statistics
import time
from typing import Any, Mapping

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
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    OrderFlowKey,
    OrderFlowSignal,
    OrderbookImbalanceStats,
    OrderbookLevel,
    OrderbookSnapshot,
    orderflow_key_to_dict,
)


class OrderbookImbalanceAnalyzer(BaseOrderFlowAnalyzer):
    """
    Orderbook imbalance analyzer for analytics.orderflow.

    Responsibilities:
    - consume normalized data-layer orderbook updates from OrderbookCache;
    - read orderbook snapshots from orderbook_cache using scoped futures key;
    - normalize snapshots through BaseOrderFlowAnalyzer helpers;
    - calculate bid/ask imbalance at configured depth;
    - optionally smooth imbalance ratio per exchange + market_type + symbol + timeframe;
    - emit analytics.orderflow.orderbook_imbalance.updated;
    - emit analytics.orderflow.orderbook_imbalance.signal;
    - use Scheduler-injected cleanup/health jobs from BaseOrderFlowAnalyzer.

    Correct input flow:
        exchange adapters
            -> market.orderbook
            -> OrderbookCache
            -> market.orderbook.updated
            -> OrderbookImbalanceAnalyzer
            -> analytics.orderflow.orderbook_imbalance.*

    Scope:
        exchange + market_type + symbol + timeframe
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        orderbook_cache: Any,
        config: OrderbookImbalanceConfig | None = None,
        scheduler: Scheduler | None = None,
        source_topic_patterns: list[str] | tuple[str, ...] | None = None,
        default_exchange: str | None = None,
        default_market_type: str = DEFAULT_MARKET_TYPE,
        default_timeframe: str = DEFAULT_TIMEFRAME,
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
            default_exchange=default_exchange,
            default_market_type=default_market_type,
            default_timeframe=default_timeframe,
        )

        self._state_lock = asyncio.Lock()

        self._last_stats_by_key: dict[OrderFlowKey, OrderbookImbalanceStats] = {}
        self._ratio_history_by_key: dict[OrderFlowKey, list[float]] = {}
        self._last_snapshot_ts_by_key: dict[OrderFlowKey, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_key(self, key: OrderFlowKey) -> OrderbookImbalanceStats | None:
        """
        Process one scoped futures market.

        key:
            exchange + market_type + symbol + timeframe
        """
        if not self.should_process_key(key):
            self._inc_metric("skipped", key)
            return None

        async with self._state_lock:
            try:
                snapshot = await self._get_orderbook_snapshot(key)
                if snapshot is None:
                    self._inc_metric("skipped", key)
                    return None

                stats = self._calculate_imbalance(snapshot)
                if stats is None:
                    self._inc_metric("skipped", key)
                    return None

                stats = self._apply_smoothing(key, stats)

                self._last_stats_by_key[key] = stats
                self._last_snapshot_ts_by_key[key] = snapshot.timestamp
                self._inc_metric("processed", key)

                await self.emit_update(stats)

                signal = self._build_signal(stats)
                if signal is not None:
                    await self.emit_signal(signal)

                return stats

            except Exception:
                self._inc_metric("errors", key)
                self._logger.exception(
                    "Failed to process orderbook imbalance",
                    extra=orderflow_key_to_dict(key),
                )
                return None

    def get_latest_stats_by_key(self, key: OrderFlowKey) -> BaseOrderFlowStats | None:
        return self._last_stats_by_key.get(key)

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
        base_stats["tracked_keys"] = len(self._last_stats_by_key)
        base_stats["tracked_markets"] = [
            {
                **orderflow_key_to_dict(key),
                "has_stats": key in self._last_stats_by_key,
                "ratio_history_size": len(self._ratio_history_by_key.get(key, [])),
                "last_snapshot_ts": self._last_snapshot_ts_by_key.get(key),
            }
            for key in sorted(self._tracked_keys())
        ]
        return base_stats

    async def cleanup(self) -> None:
        now = time.time()
        max_age = max(float(self._config.cleanup_interval_sec) * 3.0, 60.0)

        for key, snapshot_ts in list(self._last_snapshot_ts_by_key.items()):
            if (now - snapshot_ts) <= max_age:
                continue

            self._last_stats_by_key.pop(key, None)
            self._ratio_history_by_key.pop(key, None)
            self._last_snapshot_ts_by_key.pop(key, None)
            self._last_signal_ts_by_key.pop(key, None)

            self._logger.debug(
                "Removed stale orderbook imbalance state",
                extra={
                    **orderflow_key_to_dict(key),
                    "max_age": max_age,
                },
            )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, event: Event) -> None:
        key = self.extract_key_from_event(event)
        if key is None:
            self._logger.debug(
                "Orderbook event without scoped key skipped | topic=%s event_id=%s",
                event.topic,
                event.event_id,
            )
            self._inc_metric("skipped")
            return

        await self.process_key(key)

    # ------------------------------------------------------------------
    # Internal data loading
    # ------------------------------------------------------------------

    async def _get_orderbook_snapshot(
        self,
        key: OrderFlowKey,
    ) -> OrderbookSnapshot | None:
        if self._orderbook_cache is None:
            return None

        exchange, market_type, symbol, timeframe = key

        candidates: tuple[tuple[str, dict[str, Any]], ...] = (
            (
                "get_snapshot",
                {
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                    "timeframe": timeframe,
                },
            ),
            (
                "get_snapshot",
                {
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                },
            ),
            (
                "get_orderbook",
                {
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                    "timeframe": timeframe,
                },
            ),
            (
                "get_orderbook",
                {
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                },
            ),
            (
                "get_book",
                {
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                },
            ),
            (
                "get",
                {
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                },
            ),
        )

        for method_name, kwargs in candidates:
            method = getattr(self._orderbook_cache, method_name, None)
            if method is None:
                continue

            result = await self._call_cache_method(
                method=method,
                method_name=method_name,
                key=key,
                kwargs=kwargs,
            )
            snapshot = self._extract_snapshot_from_cache_result(
                result=result,
                key=key,
            )
            if snapshot is not None:
                return snapshot

        return None

    async def _call_cache_method(
        self,
        *,
        method: Any,
        method_name: str,
        key: OrderFlowKey,
        kwargs: dict[str, Any],
    ) -> Any:
        exchange, market_type, symbol, _timeframe = key

        try:
            result = method(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        except TypeError:
            # Compatibility fallback for older cache APIs.
            fallback_calls: tuple[tuple[Any, ...], ...] = (
                (exchange, market_type, symbol),
                (exchange, symbol),
                (symbol,),
            )

            for args in fallback_calls:
                try:
                    result = method(*args)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return result
                except TypeError:
                    continue
                except Exception:
                    self._logger.exception(
                        "Failed to fetch orderbook snapshot",
                        extra={
                            **orderflow_key_to_dict(key),
                            "method": method_name,
                            "args": args,
                        },
                    )
                    return None

            self._logger.exception(
                "Failed to fetch orderbook snapshot: incompatible cache method signature",
                extra={
                    **orderflow_key_to_dict(key),
                    "method": method_name,
                },
            )
            return None

        except Exception:
            self._logger.exception(
                "Failed to fetch orderbook snapshot",
                extra={
                    **orderflow_key_to_dict(key),
                    "method": method_name,
                },
            )
            return None

    def _extract_snapshot_from_cache_result(
        self,
        *,
        result: Any,
        key: OrderFlowKey,
    ) -> OrderbookSnapshot | None:
        """
        Extract and normalize OrderbookSnapshot from cache result.

        Supported cache result shapes:
        - OrderbookSnapshot
        - {"data": OrderbookSnapshot}
        - raw orderbook dict
        - {"data": raw orderbook dict}

        The analyzer accepts already-normalized models from the data/cache layer,
        but still enforces full futures scope and valid bid/ask levels.
        """
        if result is None:
            return None

        if isinstance(result, OrderbookSnapshot):
            return self._normalize_snapshot_model(
                result,
                expected_key=key,
            )

        if not isinstance(result, Mapping):
            return None

        data = result.get("data")

        if isinstance(data, OrderbookSnapshot):
            snapshot = self._normalize_snapshot_model(
                data,
                expected_key=key,
            )
            if snapshot is not None:
                return snapshot

        if isinstance(data, Mapping):
            snapshot = self.normalize_orderbook_snapshot(
                data,
                default_exchange=key[0],
                default_market_type=key[1],
                default_symbol=key[2],
                default_timeframe=key[3],
            )
            if snapshot is not None:
                return self._normalize_snapshot_model(
                    snapshot,
                    expected_key=key,
                )

        snapshot = self.normalize_orderbook_snapshot(
            result,
            default_exchange=key[0],
            default_market_type=key[1],
            default_symbol=key[2],
            default_timeframe=key[3],
        )
        if snapshot is not None:
            return self._normalize_snapshot_model(
                snapshot,
                expected_key=key,
            )

        return None

    def _normalize_snapshot_model(
        self,
        snapshot: OrderbookSnapshot,
        *,
        expected_key: OrderFlowKey,
    ) -> OrderbookSnapshot | None:
        """
        Validate and normalize an OrderbookSnapshot model returned by cache.

        Protects the analyzer from:
        - snapshots for another exchange;
        - snapshots for another market_type;
        - snapshots for another symbol;
        - snapshots for another timeframe;
        - invalid or unsorted levels;
        - empty bid/ask sides after filtering.
        """
        if snapshot.key != expected_key:
            self._logger.debug(
                "Orderbook snapshot scope mismatch skipped",
                extra={
                    "expected": orderflow_key_to_dict(expected_key),
                    "actual": orderflow_key_to_dict(snapshot.key),
                },
            )
            return None

        if not self._is_finite_positive(snapshot.timestamp):
            self._logger.debug(
                "Orderbook snapshot with invalid timestamp skipped",
                extra={
                    **orderflow_key_to_dict(expected_key),
                    "timestamp": snapshot.timestamp,
                },
            )
            return None

        valid_bids = [
            level
            for level in snapshot.bids
            if self._is_valid_level(level)
        ]
        valid_asks = [
            level
            for level in snapshot.asks
            if self._is_valid_level(level)
        ]

        if not valid_bids or not valid_asks:
            self._logger.debug(
                "Orderbook snapshot without valid bid/ask levels skipped",
                extra={
                    **orderflow_key_to_dict(expected_key),
                    "bid_levels": len(snapshot.bids),
                    "ask_levels": len(snapshot.asks),
                    "valid_bid_levels": len(valid_bids),
                    "valid_ask_levels": len(valid_asks),
                },
            )
            return None

        valid_bids.sort(key=lambda item: item.price, reverse=True)
        valid_asks.sort(key=lambda item: item.price)

        return OrderbookSnapshot(
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            symbol=snapshot.symbol,
            exchange_symbol=snapshot.exchange_symbol,
            timeframe=snapshot.timeframe,
            bids=valid_bids,
            asks=valid_asks,
            timestamp=float(snapshot.timestamp),
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

        depth_levels = max(int(self._config.depth_levels), 1)
        bid_levels = bids[:depth_levels]
        ask_levels = asks[:depth_levels]

        if not bid_levels or not ask_levels:
            return None

        bid_volume = math.fsum(level.size for level in bid_levels)
        ask_volume = math.fsum(level.size for level in ask_levels)
        total_volume = bid_volume + ask_volume

        if (
            not math.isfinite(bid_volume)
            or not math.isfinite(ask_volume)
            or not math.isfinite(total_volume)
            or total_volume <= 0.0
        ):
            self._logger.debug(
                "Orderbook snapshot with non-finite or non-positive volume skipped",
                extra={
                    **orderflow_key_to_dict(snapshot.key),
                    "bid_volume": bid_volume,
                    "ask_volume": ask_volume,
                    "total_volume": total_volume,
                },
            )
            return None

        if total_volume < self._config.min_total_volume:
            return None

        best_bid = bid_levels[0].price
        best_ask = ask_levels[0].price

        if not self._is_finite_positive(best_bid) or not self._is_finite_positive(best_ask):
            return None

        spread = best_ask - best_bid

        if not math.isfinite(spread) or spread <= 0.0:
            self._logger.warning(
                "Crossed or locked orderbook snapshot rejected",
                extra={
                    **orderflow_key_to_dict(snapshot.key),
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "sequence_id": snapshot.sequence_id,
                    "snapshot_ts": snapshot.timestamp,
                },
            )
            return None

        mid_price = (best_bid + best_ask) / 2.0
        if not self._is_finite_positive(mid_price):
            return None

        raw_ratio = bid_volume / total_volume
        imbalance_diff = (bid_volume - ask_volume) / total_volume
        imbalance_ratio = self._normalize_ratio(raw_ratio)

        if (
            not math.isfinite(raw_ratio)
            or not math.isfinite(imbalance_diff)
            or not math.isfinite(imbalance_ratio)
        ):
            self._logger.debug(
                "Orderbook snapshot produced non-finite imbalance values",
                extra={
                    **orderflow_key_to_dict(snapshot.key),
                    "raw_ratio": raw_ratio,
                    "imbalance_diff": imbalance_diff,
                    "imbalance_ratio": imbalance_ratio,
                },
            )
            return None

        return OrderbookImbalanceStats(
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            symbol=snapshot.symbol,
            exchange_symbol=snapshot.exchange_symbol,
            timeframe=snapshot.timeframe,
            metric=OrderFlowMetricType.ORDERBOOK_IMBALANCE,
            source_type=OrderFlowSourceType.ORDERBOOK,
            timestamp=time.time(),
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance_ratio=imbalance_ratio,
            imbalance_diff=imbalance_diff,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            mid_price=mid_price,
            depth_levels_used=min(
                depth_levels,
                len(bid_levels),
                len(ask_levels),
            ),
            metadata={
                "snapshot_ts": snapshot.timestamp,
                "sequence_id": snapshot.sequence_id,
                "depth_levels_configured": self._config.depth_levels,
                "bid_levels_available": len(bids),
                "ask_levels_available": len(asks),
                "scope": "exchange:market_type:symbol:timeframe",
            },
        )

    def _prepare_side(
        self,
        levels: list[OrderbookLevel],
        *,
        reverse: bool,
    ) -> list[OrderbookLevel]:
        filtered = [
            level
            for level in levels
            if self._is_valid_level(level)
        ]
        filtered.sort(key=lambda item: item.price, reverse=reverse)
        return filtered

    def _is_valid_level(self, level: Any) -> bool:
        return (
            isinstance(level, OrderbookLevel)
            and level.is_valid
            and self._is_finite_positive(level.price)
            and self._is_finite_positive(level.size)
        )

    @staticmethod
    def _is_finite_positive(value: Any) -> bool:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return False

        return math.isfinite(parsed) and parsed > 0.0

    def _apply_smoothing(
        self,
        key: OrderFlowKey,
        stats: OrderbookImbalanceStats,
    ) -> OrderbookImbalanceStats:
        window = max(int(self._config.smooth_window), 1)
        if window <= 1:
            return stats

        if not math.isfinite(stats.imbalance_ratio):
            return stats

        history = self._ratio_history_by_key.setdefault(key, [])
        history.append(stats.imbalance_ratio)

        if len(history) > window:
            del history[:-window]

        smoothed_ratio = (
            statistics.fmean(history)
            if history
            else stats.imbalance_ratio
        )
        if not math.isfinite(smoothed_ratio):
            return stats

        return OrderbookImbalanceStats(
            exchange=stats.exchange,
            market_type=stats.market_type,
            symbol=stats.symbol,
            exchange_symbol=stats.exchange_symbol,
            timeframe=stats.timeframe,
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
            metadata={
                **dict(stats.metadata),
                "smoothing_window": window,
                "smoothing_points": len(history),
                "raw_imbalance_ratio": stats.imbalance_ratio,
            },
        )

    def _normalize_ratio(self, raw_ratio: float) -> float:
        if not math.isfinite(raw_ratio):
            return float("nan")

        if self._config.normalize_ratio_to_minus_one_one:
            return (raw_ratio * 2.0) - 1.0

        return raw_ratio

    def _denormalize_ratio_if_needed(self, value: float) -> float:
        if not math.isfinite(value):
            return float("nan")

        if self._config.normalize_ratio_to_minus_one_one:
            return (value + 1.0) / 2.0

        return value

    def _ratio_to_diff(self, ratio: float) -> float:
        if not math.isfinite(ratio):
            return float("nan")

        if self._config.normalize_ratio_to_minus_one_one:
            return ratio

        return (ratio * 2.0) - 1.0

    # ------------------------------------------------------------------
    # Signal logic
    # ------------------------------------------------------------------

    def _build_signal(self, stats: OrderbookImbalanceStats) -> OrderFlowSignal | None:
        ratio_for_signal = self._denormalize_ratio_if_needed(stats.imbalance_ratio)
        if not math.isfinite(ratio_for_signal):
            return None

        bullish_ok = ratio_for_signal >= self._config.bullish_ratio_threshold
        bearish_ok = ratio_for_signal <= self._config.bearish_ratio_threshold

        if bullish_ok and not bearish_ok:
            return self.build_signal_from_stats(
                stats=stats,
                signal_type=OrderFlowSignalType.BULLISH,
                side=OrderFlowSide.BUY,
                strength=self._calculate_bullish_strength(ratio_for_signal, stats),
                reason="orderbook_bid_imbalance",
                context=self._build_signal_context(stats),
            )

        if bearish_ok and not bullish_ok:
            return self.build_signal_from_stats(
                stats=stats,
                signal_type=OrderFlowSignalType.BEARISH,
                side=OrderFlowSide.SELL,
                strength=self._calculate_bearish_strength(ratio_for_signal, stats),
                reason="orderbook_ask_imbalance",
                context=self._build_signal_context(stats),
            )

        return None

    def _build_signal_context(self, stats: OrderbookImbalanceStats) -> dict[str, Any]:
        return {
            "exchange": stats.exchange,
            "market_type": stats.market_type,
            "symbol": stats.symbol,
            "exchange_symbol": stats.exchange_symbol,
            "timeframe": stats.timeframe,
            "key": list(stats.key),
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
    # Misc helpers
    # ------------------------------------------------------------------

    def _tracked_keys(self) -> set[OrderFlowKey]:
        return (
            set(self._last_stats_by_key)
            | set(self._ratio_history_by_key)
            | set(self._last_snapshot_ts_by_key)
        )

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    def _spread_quality_component(self, stats: OrderbookImbalanceStats) -> float:
        if (
            stats.spread is None
            or stats.mid_price is None
            or not math.isfinite(stats.spread)
            or not math.isfinite(stats.mid_price)
            or stats.spread <= 0.0
            or stats.mid_price <= 0.0
        ):
            return 0.0

        spread_pct = stats.spread / stats.mid_price
        if not math.isfinite(spread_pct) or spread_pct < 0.0:
            return 0.0

        return max(0.0, 1.0 - min(spread_pct * 1000.0, 1.0))

    def _safe_ratio(self, value: float, threshold: float) -> float:
        if not math.isfinite(value) or not math.isfinite(threshold):
            return 0.0

        if threshold == 0:
            return self._normalize_magnitude(value)

        return max(0.0, value / threshold)

    def _normalize_magnitude(self, value: float) -> float:
        if not math.isfinite(value):
            return 0.0

        return math.log1p(abs(value))

    def _normalize_strength(self, values: list[float]) -> float:
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            return 0.0

        raw = sum(max(0.0, value) for value in finite_values) / len(finite_values)
        if not math.isfinite(raw):
            return 0.0

        return max(0.0, min(raw / (1.0 + raw), 1.0))
