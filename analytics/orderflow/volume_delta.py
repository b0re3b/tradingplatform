from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import Any

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from .base import BaseOrderFlowAnalyzer
from .config import VolumeDeltaConfig
from .enums import (
    TRADE_INPUT_TOPICS,
    OrderFlowMetricType,
    OrderFlowSide,
    OrderFlowSignalType,
    OrderFlowSourceType,
)
from .models import (
    BaseOrderFlowStats,
    NormalizedTrade,
    OrderFlowSignal,
    VolumeDeltaStats,
)


class VolumeDeltaAnalyzer(BaseOrderFlowAnalyzer):
    """
    Volume delta analyzer for analytics.orderflow.

    Responsibilities:
    - consume trade events through EventBus subscriptions registered in base
    - read recent trades from trades_cache
    - normalize trades through BaseOrderFlowAnalyzer helpers
    - maintain sliding trade windows per symbol
    - calculate volume delta, notional delta and cumulative delta stats
    - emit analytics.orderflow.volume_delta.updated and
      analytics.orderflow.volume_delta.signal
    - use Scheduler-injected cleanup/health jobs from BaseOrderFlowAnalyzer
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        trades_cache: Any,
        config: VolumeDeltaConfig | None = None,
        scheduler: Scheduler | None = None,
        source_topic_patterns: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._trades_cache = trades_cache
        self._config = config or VolumeDeltaConfig()

        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=self._config,
            metric_type=OrderFlowMetricType.VOLUME_DELTA,
            source_type=OrderFlowSourceType.TRADES,
            source_topic_patterns=source_topic_patterns or TRADE_INPUT_TOPICS,
            component_module="orderflow",
        )

        self._state_lock = asyncio.Lock()

        self._trades_by_symbol: dict[str, deque[NormalizedTrade]] = {}
        self._last_stats_by_symbol: dict[str, VolumeDeltaStats] = {}
        self._last_seen_trade_key_by_symbol: dict[str, str] = {}

        self._metrics.setdefault("processed_trades", 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_symbol(self, symbol: str) -> VolumeDeltaStats | None:
        normalized_symbol = str(symbol).strip().upper()

        if not self.should_process_symbol(normalized_symbol):
            self._inc_metric("skipped", normalized_symbol)
            return None

        async with self._state_lock:
            try:
                raw_trades = await self._get_recent_trades(normalized_symbol)
                if not raw_trades:
                    self._inc_metric("skipped", normalized_symbol)
                    return None

                normalized_trades = self._normalize_trades(
                    normalized_symbol,
                    raw_trades,
                )
                if not normalized_trades:
                    self._inc_metric("skipped", normalized_symbol)
                    return None

                new_trades = self._filter_new_trades(
                    normalized_symbol,
                    normalized_trades,
                )
                if not new_trades and normalized_symbol not in self._trades_by_symbol:
                    self._inc_metric("skipped", normalized_symbol)
                    return None

                added_count = self._append_new_trades(
                    normalized_symbol,
                    new_trades,
                )

                self._prune_old_trades(normalized_symbol)

                stats = self._calculate_window_stats(normalized_symbol)
                if stats is None:
                    self._inc_metric("skipped", normalized_symbol)
                    return None

                self._last_stats_by_symbol[normalized_symbol] = stats
                self._inc_metric("processed", normalized_symbol)

                if added_count > 0:
                    self._metrics["processed_trades"] += added_count

                await self.emit_update(stats)

                signal = self._build_signal(stats)
                if signal is not None:
                    await self.emit_signal(signal)

                return stats

            except Exception:
                self._inc_metric("errors", normalized_symbol)
                self._logger.exception(
                    "Failed to process volume delta | symbol=%s",
                    normalized_symbol,
                )
                return None

    def get_latest_stats(self, symbol: str) -> BaseOrderFlowStats | None:
        return self._last_stats_by_symbol.get(str(symbol).strip().upper())

    def stats(self) -> dict[str, Any]:
        base_stats = super().stats()
        base_stats["config"].update(
            {
                "window_seconds": self._config.window_seconds,
                "max_trades_per_symbol": self._config.max_trades_per_symbol,
                "min_trades_in_window": self._config.min_trades_in_window,
                "min_total_volume": self._config.min_total_volume,
                "bullish_delta_ratio_threshold": self._config.bullish_delta_ratio_threshold,
                "bearish_delta_ratio_threshold": self._config.bearish_delta_ratio_threshold,
                "bullish_volume_delta_threshold": self._config.bullish_volume_delta_threshold,
                "bearish_volume_delta_threshold": self._config.bearish_volume_delta_threshold,
                "bullish_cumulative_delta_threshold": (
                    self._config.bullish_cumulative_delta_threshold
                ),
                "bearish_cumulative_delta_threshold": (
                    self._config.bearish_cumulative_delta_threshold
                ),
                "require_ratio_and_absolute_confirmation": (
                    self._config.require_ratio_and_absolute_confirmation
                ),
            }
        )
        base_stats["tracked_symbols"] = len(self._trades_by_symbol)
        base_stats["metrics"]["processed_trades"] = self._metrics.get(
            "processed_trades",
            0,
        )
        return base_stats

    async def cleanup(self) -> None:
        now = time.time()
        max_age = max(float(self._config.window_seconds) * 3.0, 60.0)

        for symbol, trades in list(self._trades_by_symbol.items()):
            if not trades:
                continue

            latest_ts = trades[-1].timestamp
            if (now - latest_ts) <= max_age:
                continue

            self._trades_by_symbol.pop(symbol, None)
            self._last_stats_by_symbol.pop(symbol, None)
            self._last_seen_trade_key_by_symbol.pop(symbol, None)
            self._last_signal_ts_by_symbol.pop(symbol, None)

            self._logger.debug(
                "Removed stale volume delta state | symbol=%s max_age=%s",
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
                "Trade event without symbol skipped | topic=%s event_id=%s",
                event.topic,
                event.event_id,
            )
            self._inc_metric("skipped")
            return

        await self.process_symbol(symbol)

    # ------------------------------------------------------------------
    # Internal data loading
    # ------------------------------------------------------------------

    async def _get_recent_trades(self, symbol: str) -> list[Any]:
        if self._trades_cache is None:
            return []

        candidates: tuple[tuple[str, dict[str, Any]], ...] = (
            ("get_recent_trades", {"symbol": symbol}),
            ("get_trades", {"symbol": symbol}),
            ("get", {"symbol": symbol}),
        )

        for method_name, kwargs in candidates:
            method = getattr(self._trades_cache, method_name, None)
            if method is None:
                continue

            result = await self._call_cache_method(
                method=method,
                method_name=method_name,
                symbol=symbol,
                kwargs=kwargs,
            )
            trades = self._extract_trades_from_cache_result(result)
            if trades:
                return trades

        return []

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
                    "Failed to fetch recent trades | method=%s symbol=%s",
                    method_name,
                    symbol,
                )
                return None

        except Exception:
            self._logger.exception(
                "Failed to fetch recent trades | method=%s symbol=%s",
                method_name,
                symbol,
            )
            return None

    def _extract_trades_from_cache_result(self, result: Any) -> list[Any]:
        if result is None:
            return []

        if isinstance(result, list):
            return result

        if isinstance(result, tuple):
            return list(result)

        if isinstance(result, dict):
            trades = result.get("trades")
            if isinstance(trades, list):
                return trades

            data = result.get("data")
            if isinstance(data, list):
                return data

            if isinstance(data, dict):
                nested_trades = data.get("trades")
                if isinstance(nested_trades, list):
                    return nested_trades

        return []

    def _normalize_trades(
        self,
        symbol: str,
        raw_trades: list[Any],
    ) -> list[NormalizedTrade]:
        normalized: list[NormalizedTrade] = []

        for raw_trade in raw_trades:
            trade = self.normalize_trade(raw_trade, default_symbol=symbol)
            if trade is None:
                continue

            if not trade.side.is_known:
                continue

            if not trade.is_valid:
                continue

            normalized.append(trade)

        normalized.sort(key=lambda item: (item.timestamp, item.trade_id or ""))
        return normalized

    # ------------------------------------------------------------------
    # State mutation
    # ------------------------------------------------------------------

    def _append_new_trades(
        self,
        symbol: str,
        trades: list[NormalizedTrade],
    ) -> int:
        if not trades:
            return 0

        store = self._trades_by_symbol.setdefault(
            symbol,
            deque(maxlen=self._config.max_trades_per_symbol),
        )

        for trade in trades:
            store.append(trade)

        return len(trades)

    def _filter_new_trades(
        self,
        symbol: str,
        trades: list[NormalizedTrade],
    ) -> list[NormalizedTrade]:
        if not trades:
            return []

        last_seen_key = self._last_seen_trade_key_by_symbol.get(symbol)

        if not last_seen_key:
            new_trades = trades
        else:
            new_trades = self._collect_trades_after_last_seen(
                trades=trades,
                last_seen_key=last_seen_key,
            )

            if not new_trades and all(
                self.make_trade_key(trade) != last_seen_key for trade in trades
            ):
                new_trades = self._filter_new_trades_fallback(symbol, trades)

        self._last_seen_trade_key_by_symbol[symbol] = self.make_trade_key(trades[-1])
        return new_trades

    def _collect_trades_after_last_seen(
        self,
        *,
        trades: list[NormalizedTrade],
        last_seen_key: str,
    ) -> list[NormalizedTrade]:
        new_trades: list[NormalizedTrade] = []
        seen_last = False

        for trade in trades:
            trade_key = self.make_trade_key(trade)

            if seen_last:
                new_trades.append(trade)
                continue

            if trade_key == last_seen_key:
                seen_last = True

        return new_trades

    def _filter_new_trades_fallback(
        self,
        symbol: str,
        trades: list[NormalizedTrade],
    ) -> list[NormalizedTrade]:
        existing = self._trades_by_symbol.get(symbol)
        if not existing:
            return trades

        existing_keys = {self.make_trade_key(trade) for trade in existing}
        return [
            trade
            for trade in trades
            if self.make_trade_key(trade) not in existing_keys
        ]

    # ------------------------------------------------------------------
    # Window maintenance
    # ------------------------------------------------------------------

    def _prune_old_trades(self, symbol: str) -> None:
        store = self._trades_by_symbol.get(symbol)
        if not store:
            return

        cutoff_ts = time.time() - float(self._config.window_seconds)

        while store and store[0].timestamp < cutoff_ts:
            store.popleft()

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def _calculate_window_stats(self, symbol: str) -> VolumeDeltaStats | None:
        trades = self._trades_by_symbol.get(symbol)
        if not trades:
            return None

        recent_trades = list(trades)

        if len(recent_trades) < self._config.min_trades_in_window:
            return None

        buy_trades = [
            trade for trade in recent_trades
            if trade.side == OrderFlowSide.BUY
        ]
        sell_trades = [
            trade for trade in recent_trades
            if trade.side == OrderFlowSide.SELL
        ]

        buy_volume = sum(trade.quantity for trade in buy_trades)
        sell_volume = sum(trade.quantity for trade in sell_trades)
        total_volume = buy_volume + sell_volume

        if total_volume <= 0:
            return None

        if total_volume < self._config.min_total_volume:
            return None

        buy_notional = sum(trade.notional for trade in buy_trades)
        sell_notional = sum(trade.notional for trade in sell_trades)
        total_notional = buy_notional + sell_notional

        volume_delta = buy_volume - sell_volume
        notional_delta = buy_notional - sell_notional
        delta_ratio = volume_delta / total_volume

        cumulative_volume_delta = sum(trade.signed_volume for trade in recent_trades)
        cumulative_notional_delta = sum(trade.signed_notional for trade in recent_trades)

        return VolumeDeltaStats(
            symbol=symbol,
            metric=OrderFlowMetricType.VOLUME_DELTA,
            source_type=OrderFlowSourceType.TRADES,
            window_seconds=float(self._config.window_seconds),
            trades_count=len(recent_trades),
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            buy_notional=buy_notional,
            sell_notional=sell_notional,
            volume_delta=volume_delta,
            notional_delta=notional_delta,
            delta_ratio=delta_ratio,
            cumulative_volume_delta=cumulative_volume_delta,
            cumulative_notional_delta=cumulative_notional_delta,
            buy_ratio=buy_volume / total_volume,
            sell_ratio=sell_volume / total_volume,
            avg_trade_size=total_volume / len(recent_trades),
            avg_trade_notional=(
                total_notional / len(recent_trades)
                if total_notional > 0
                else 0.0
            ),
            last_price=recent_trades[-1].price,
        )

    # ------------------------------------------------------------------
    # Signal logic
    # ------------------------------------------------------------------

    def _build_signal(self, stats: VolumeDeltaStats) -> OrderFlowSignal | None:
        bullish_ratio_ok = (
            stats.delta_ratio >= self._config.bullish_delta_ratio_threshold
        )
        bearish_ratio_ok = (
            stats.delta_ratio <= self._config.bearish_delta_ratio_threshold
        )

        bullish_absolute_ok = (
            stats.volume_delta >= self._config.bullish_volume_delta_threshold
            and stats.cumulative_volume_delta
            >= self._config.bullish_cumulative_delta_threshold
        )
        bearish_absolute_ok = (
            stats.volume_delta <= self._config.bearish_volume_delta_threshold
            and stats.cumulative_volume_delta
            <= self._config.bearish_cumulative_delta_threshold
        )

        if self._config.require_ratio_and_absolute_confirmation:
            bullish_ok = bullish_ratio_ok and bullish_absolute_ok
            bearish_ok = bearish_ratio_ok and bearish_absolute_ok
        else:
            bullish_ok = bullish_ratio_ok or bullish_absolute_ok
            bearish_ok = bearish_ratio_ok or bearish_absolute_ok

        if bullish_ok and not bearish_ok:
            return self.build_signal(
                symbol=stats.symbol,
                signal_type=OrderFlowSignalType.BULLISH,
                side=OrderFlowSide.BUY,
                strength=self._calculate_bullish_strength(stats),
                reason="volume_delta_bullish_confirmation",
                context=self._build_signal_context(stats),
            )

        if bearish_ok and not bullish_ok:
            return self.build_signal(
                symbol=stats.symbol,
                signal_type=OrderFlowSignalType.BEARISH,
                side=OrderFlowSide.SELL,
                strength=self._calculate_bearish_strength(stats),
                reason="volume_delta_bearish_confirmation",
                context=self._build_signal_context(stats),
            )

        return None

    def _build_signal_context(self, stats: VolumeDeltaStats) -> dict[str, Any]:
        return {
            "volume_delta": stats.volume_delta,
            "notional_delta": stats.notional_delta,
            "delta_ratio": stats.delta_ratio,
            "cumulative_volume_delta": stats.cumulative_volume_delta,
            "cumulative_notional_delta": stats.cumulative_notional_delta,
            "buy_ratio": stats.buy_ratio,
            "sell_ratio": stats.sell_ratio,
            "trades_count": stats.trades_count,
            "last_price": stats.last_price,
        }

    def _calculate_bullish_strength(self, stats: VolumeDeltaStats) -> float:
        components: list[float] = [
            self._safe_ratio(
                stats.delta_ratio,
                self._config.bullish_delta_ratio_threshold,
            )
        ]

        if self._config.bullish_volume_delta_threshold > 0:
            components.append(
                self._safe_ratio(
                    stats.volume_delta,
                    self._config.bullish_volume_delta_threshold,
                )
            )
        else:
            components.append(self._normalize_magnitude(stats.volume_delta))

        if self._config.bullish_cumulative_delta_threshold > 0:
            components.append(
                self._safe_ratio(
                    stats.cumulative_volume_delta,
                    self._config.bullish_cumulative_delta_threshold,
                )
            )
        else:
            components.append(
                self._normalize_magnitude(stats.cumulative_volume_delta)
            )

        return self._normalize_strength(components)

    def _calculate_bearish_strength(self, stats: VolumeDeltaStats) -> float:
        components: list[float] = [
            self._safe_ratio(
                abs(stats.delta_ratio),
                abs(self._config.bearish_delta_ratio_threshold),
            )
        ]

        if self._config.bearish_volume_delta_threshold < 0:
            components.append(
                self._safe_ratio(
                    abs(stats.volume_delta),
                    abs(self._config.bearish_volume_delta_threshold),
                )
            )
        else:
            components.append(self._normalize_magnitude(stats.volume_delta))

        if self._config.bearish_cumulative_delta_threshold < 0:
            components.append(
                self._safe_ratio(
                    abs(stats.cumulative_volume_delta),
                    abs(self._config.bearish_cumulative_delta_threshold),
                )
            )
        else:
            components.append(
                self._normalize_magnitude(stats.cumulative_volume_delta)
            )

        return self._normalize_strength(components)

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

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