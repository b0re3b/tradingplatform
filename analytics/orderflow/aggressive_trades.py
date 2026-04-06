from __future__ import annotations

import asyncio
import math
import statistics
import time
from collections import deque
from typing import Any, Optional

from core.event_bus import Event, EventBus

from .base import BaseOrderFlowAnalyzer
from .config import AggressiveTradesConfig
from .enums import (
    OrderFlowMetricType,
    OrderFlowSide,
    OrderFlowSignalType,
    OrderFlowSourceType,
)
from .models import (
    AggressiveTradesStats,
    BaseOrderFlowStats,
    NormalizedTrade,
)


class AggressiveTradesAnalyzer(BaseOrderFlowAnalyzer):
    """
    Analyzer для аналізу aggressive trades.

    Основні задачі:
    - приймає trade events через EventBus
    - читає останні трейди з trades_cache
    - нормалізує трейди
    - фільтрує / інтерпретує aggressive buy/sell flow
    - рахує window-based stats
    - виявляє burst / large aggressive activity
    - публікує update/signal події
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
        self._trades_cache = trades_cache
        self._config = config or (
            AggressiveTradesConfig.from_app_config(app_config)
            if app_config is not None
            else AggressiveTradesConfig()
        )

        super().__init__(
            event_bus=event_bus,
            config=self._config,
            metric_type=OrderFlowMetricType.AGGRESSIVE_TRADES,
            source_type=OrderFlowSourceType.TRADES,
            scheduler=scheduler,
            source_topic_patterns=source_topic_patterns or [
                "market.trade",
                "market.trade.*",
                "market.trades.updated",
                "trades.*",
            ],
            component_module="orderflow",
        )

        self._state_lock = asyncio.Lock()

        self._trades_by_symbol: dict[str, deque[NormalizedTrade]] = {}
        self._last_stats_by_symbol: dict[str, AggressiveTradesStats] = {}
        self._last_seen_trade_key_by_symbol: dict[str, str] = {}

        self._metrics.setdefault("processed_trades", 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_symbol(self, symbol: str) -> Optional[AggressiveTradesStats]:
        symbol = str(symbol).upper()

        if not self.should_process_symbol(symbol):
            self._inc_metric("skipped", symbol)
            return None

        async with self._state_lock:
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

                store = self._trades_by_symbol.setdefault(
                    symbol,
                    deque(maxlen=self._config.max_trades_per_symbol),
                )

                added_count = 0
                for trade in new_trades:
                    store.append(trade)
                    added_count += 1

                self._prune_old_trades(symbol)

                stats = self._calculate_window_stats(symbol)
                if stats is None:
                    self._inc_metric("skipped", symbol)
                    return None

                self._last_stats_by_symbol[symbol] = stats
                self._inc_metric("processed", symbol)

                if added_count > 0:
                    self._metrics["processed_trades"] += added_count

                await self.emit_update(stats)

                signal = self._build_signal(stats)
                if signal is not None:
                    await self.emit_signal(signal)

                return stats

            except Exception:
                self._inc_metric("errors", symbol)
                self._logger.exception(
                    "Failed to process aggressive trades | symbol=%s",
                    symbol,
                )
                return None

    def get_latest_stats(self, symbol: str) -> Optional[BaseOrderFlowStats]:
        return self._last_stats_by_symbol.get(str(symbol).upper())

    def stats(self) -> dict[str, Any]:
        base_stats = super().stats()
        base_stats["config"].update(
            {
                "window_seconds": self._config.window_seconds,
                "max_trades_per_symbol": self._config.max_trades_per_symbol,
                "min_trades_in_window": self._config.min_trades_in_window,
                "bullish_buy_ratio_threshold": self._config.bullish_buy_ratio_threshold,
                "bearish_sell_ratio_threshold": self._config.bearish_sell_ratio_threshold,
                "bullish_delta_threshold": self._config.bullish_delta_threshold,
                "bearish_delta_threshold": self._config.bearish_delta_threshold,
                "large_trade_notional_threshold": self._config.large_trade_notional_threshold,
                "min_large_trades_for_signal": self._config.min_large_trades_for_signal,
                "burst_trades_threshold": self._config.burst_trades_threshold,
                "burst_volume_threshold": self._config.burst_volume_threshold,
                "burst_score_threshold": self._config.burst_score_threshold,
            }
        )
        base_stats["tracked_symbols"] = len(self._trades_by_symbol)
        base_stats["metrics"]["processed_trades"] = self._metrics.get("processed_trades", 0)
        return base_stats

    async def cleanup(self) -> None:
        now = time.time()
        max_age = float(self._config.window_seconds) * 3.0

        for symbol, trades in list(self._trades_by_symbol.items()):
            if not trades:
                continue

            latest_ts = trades[-1].timestamp
            if (now - latest_ts) > max_age:
                self._trades_by_symbol.pop(symbol, None)
                self._last_stats_by_symbol.pop(symbol, None)
                self._last_seen_trade_key_by_symbol.pop(symbol, None)
                self._last_signal_ts_by_symbol.pop(symbol, None)

                self._logger.debug(
                    "Removed stale aggressive trades state | symbol=%s max_age=%s",
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
                getattr(event, "topic", None),
                getattr(event, "event_id", None),
            )
            self._inc_metric("skipped")
            return

        await self.process_symbol(symbol)

    # ------------------------------------------------------------------
    # Internal data loading
    # ------------------------------------------------------------------

    async def _get_recent_trades(self, symbol: str) -> list[Any]:
        cache = self._trades_cache
        if cache is None:
            return []

        candidates = [
            ("get_recent_trades", {"symbol": symbol}),
            ("get_trades", {"symbol": symbol}),
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

                if result is None:
                    continue

                if isinstance(result, list):
                    return result

                if isinstance(result, tuple):
                    return list(result)

                if isinstance(result, dict):
                    if isinstance(result.get("trades"), list):
                        return result["trades"]
                    if isinstance(result.get("data"), list):
                        return result["data"]

            except TypeError:
                try:
                    result = method(symbol)
                    if asyncio.iscoroutine(result):
                        result = await result

                    if isinstance(result, list):
                        return result

                    if isinstance(result, tuple):
                        return list(result)
                except Exception:
                    self._logger.exception(
                        "Failed to fetch recent trades | method=%s symbol=%s",
                        method_name,
                        symbol,
                    )
            except Exception:
                self._logger.exception(
                    "Failed to fetch recent trades | method=%s symbol=%s",
                    method_name,
                    symbol,
                )

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

            if trade.side == OrderFlowSide.UNKNOWN:
                continue

            if trade.quantity <= 0 or trade.price <= 0:
                continue

            trade.is_aggressive = self._resolve_is_aggressive(trade)
            normalized.append(trade)

        normalized.sort(key=lambda item: (item.timestamp, item.trade_id or ""))
        return normalized

    def _resolve_is_aggressive(self, trade: NormalizedTrade) -> bool:
        if trade.is_aggressive:
            return True

        raw = trade.raw or {}

        if "is_aggressive" in raw:
            return bool(raw["is_aggressive"])

        if "aggressive" in raw:
            return bool(raw["aggressive"])

        # Для більшості trade stream-ів side вже інтерпретований як aggressor side
        # Тому якщо side відомий, вважаємо трейд aggressive.
        return trade.side in {OrderFlowSide.BUY, OrderFlowSide.SELL}

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
            new_trades = []
            seen_last = False

            for trade in trades:
                trade_key = self.make_trade_key(trade)

                if seen_last:
                    new_trades.append(trade)
                    continue

                if trade_key == last_seen_key:
                    seen_last = True

            if not seen_last:
                new_trades = self._filter_new_trades_fallback(symbol, trades)

        self._last_seen_trade_key_by_symbol[symbol] = self.make_trade_key(trades[-1])
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
        return [trade for trade in trades if self.make_trade_key(trade) not in existing_keys]

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

    def _calculate_window_stats(self, symbol: str) -> Optional[AggressiveTradesStats]:
        trades = self._trades_by_symbol.get(symbol)
        if not trades:
            return None

        recent = list(trades)
        if len(recent) < self._config.min_trades_in_window:
            return None

        aggressive_buys = [
            trade for trade in recent
            if trade.side == OrderFlowSide.BUY and trade.is_aggressive
        ]
        aggressive_sells = [
            trade for trade in recent
            if trade.side == OrderFlowSide.SELL and trade.is_aggressive
        ]

        buy_count = len(aggressive_buys)
        sell_count = len(aggressive_sells)
        trades_count = buy_count + sell_count

        if trades_count <= 0:
            return None

        buy_volume = sum(trade.quantity for trade in aggressive_buys)
        sell_volume = sum(trade.quantity for trade in aggressive_sells)

        buy_notional = sum(trade.notional for trade in aggressive_buys)
        sell_notional = sum(trade.notional for trade in aggressive_sells)

        total_volume = buy_volume + sell_volume
        total_notional = buy_notional + sell_notional

        if total_volume <= 0 or total_notional <= 0:
            return None

        buy_ratio = buy_volume / total_volume
        sell_ratio = sell_volume / total_volume

        net_volume_delta = buy_volume - sell_volume
        net_notional_delta = buy_notional - sell_notional

        large_buy_trades = sum(
            1
            for trade in aggressive_buys
            if trade.notional >= self._config.large_trade_notional_threshold
        )
        large_sell_trades = sum(
            1
            for trade in aggressive_sells
            if trade.notional >= self._config.large_trade_notional_threshold
        )

        avg_trade_size = statistics.fmean(trade.quantity for trade in recent) if recent else 0.0
        avg_trade_notional = (
            statistics.fmean(trade.notional for trade in recent) if recent else 0.0
        )

        burst_score = self._calculate_burst_score(
            trades_count=trades_count,
            total_volume=total_volume,
            avg_trade_size=avg_trade_size,
        )

        last_price = recent[-1].price if recent else None

        return AggressiveTradesStats(
            symbol=symbol,
            metric=OrderFlowMetricType.AGGRESSIVE_TRADES,
            source_type=OrderFlowSourceType.TRADES,
            window_seconds=float(self._config.window_seconds),
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

    def _calculate_burst_score(
        self,
        *,
        trades_count: int,
        total_volume: float,
        avg_trade_size: float,
    ) -> float:
        trade_component = 0.0
        if self._config.burst_trades_threshold > 0:
            trade_component = trades_count / float(self._config.burst_trades_threshold)
        else:
            trade_component = math.log1p(max(trades_count, 0))

        volume_component = 0.0
        if self._config.burst_volume_threshold > 0:
            volume_component = total_volume / float(self._config.burst_volume_threshold)
        else:
            volume_component = math.log1p(max(total_volume, 0.0))

        size_component = math.log1p(max(avg_trade_size, 0.0))

        raw_score = (trade_component + volume_component + size_component) / 3.0
        return max(0.0, raw_score)

    # ------------------------------------------------------------------
    # Signal logic
    # ------------------------------------------------------------------

    def _build_signal(self, stats: AggressiveTradesStats):
        bullish_ratio_ok = (
            stats.buy_ratio >= self._config.bullish_buy_ratio_threshold
        )
        bearish_ratio_ok = (
            stats.sell_ratio >= self._config.bearish_sell_ratio_threshold
        )

        bullish_delta_ok = (
            stats.net_volume_delta >= self._config.bullish_delta_threshold
        )
        bearish_delta_ok = (
            stats.net_volume_delta <= self._config.bearish_delta_threshold
        )

        bullish_large_ok = (
            stats.large_buy_trades >= self._config.min_large_trades_for_signal
        )
        bearish_large_ok = (
            stats.large_sell_trades >= self._config.min_large_trades_for_signal
        )

        burst_ok = stats.burst_score >= self._config.burst_score_threshold

        bullish_ok = bullish_ratio_ok and bullish_delta_ok and bullish_large_ok and burst_ok
        bearish_ok = bearish_ratio_ok and bearish_delta_ok and bearish_large_ok and burst_ok

        if bullish_ok and not bearish_ok:
            strength = self._calculate_bullish_strength(stats)
            return self.build_signal(
                symbol=stats.symbol,
                signal_type=OrderFlowSignalType.BULLISH,
                side=OrderFlowSide.BUY,
                strength=strength,
                reason="aggressive_buying_pressure",
                context={
                    "trades_count": stats.trades_count,
                    "aggressive_buy_count": stats.aggressive_buy_count,
                    "aggressive_sell_count": stats.aggressive_sell_count,
                    "net_volume_delta": stats.net_volume_delta,
                    "net_notional_delta": stats.net_notional_delta,
                    "buy_ratio": stats.buy_ratio,
                    "sell_ratio": stats.sell_ratio,
                    "burst_score": stats.burst_score,
                    "large_buy_trades": stats.large_buy_trades,
                    "large_sell_trades": stats.large_sell_trades,
                    "last_price": stats.last_price,
                },
            )

        if bearish_ok and not bullish_ok:
            strength = self._calculate_bearish_strength(stats)
            return self.build_signal(
                symbol=stats.symbol,
                signal_type=OrderFlowSignalType.BEARISH,
                side=OrderFlowSide.SELL,
                strength=strength,
                reason="aggressive_selling_pressure",
                context={
                    "trades_count": stats.trades_count,
                    "aggressive_buy_count": stats.aggressive_buy_count,
                    "aggressive_sell_count": stats.aggressive_sell_count,
                    "net_volume_delta": stats.net_volume_delta,
                    "net_notional_delta": stats.net_notional_delta,
                    "buy_ratio": stats.buy_ratio,
                    "sell_ratio": stats.sell_ratio,
                    "burst_score": stats.burst_score,
                    "large_buy_trades": stats.large_buy_trades,
                    "large_sell_trades": stats.large_sell_trades,
                    "last_price": stats.last_price,
                },
            )

        return None

    def _calculate_bullish_strength(self, stats: AggressiveTradesStats) -> float:
        components: list[float] = []

        if self._config.bullish_buy_ratio_threshold > 0:
            components.append(
                self._safe_ratio(
                    stats.buy_ratio,
                    self._config.bullish_buy_ratio_threshold,
                )
            )
        else:
            components.append(self._normalize_signed_magnitude(stats.buy_ratio))

        if self._config.bullish_delta_threshold > 0:
            components.append(
                self._safe_ratio(
                    stats.net_volume_delta,
                    self._config.bullish_delta_threshold,
                )
            )
        else:
            components.append(self._normalize_signed_magnitude(stats.net_volume_delta))

        components.append(
            self._safe_ratio(
                stats.burst_score,
                self._config.burst_score_threshold if self._config.burst_score_threshold > 0 else 1.0,
            )
        )

        if self._config.min_large_trades_for_signal > 0:
            components.append(
                self._safe_ratio(
                    float(stats.large_buy_trades),
                    float(self._config.min_large_trades_for_signal),
                )
            )
        else:
            components.append(self._normalize_signed_magnitude(float(stats.large_buy_trades)))

        return self._normalize_strength(components)

    def _calculate_bearish_strength(self, stats: AggressiveTradesStats) -> float:
        components: list[float] = []

        if self._config.bearish_sell_ratio_threshold > 0:
            components.append(
                self._safe_ratio(
                    stats.sell_ratio,
                    self._config.bearish_sell_ratio_threshold,
                )
            )
        else:
            components.append(self._normalize_signed_magnitude(stats.sell_ratio))

        if self._config.bearish_delta_threshold < 0:
            components.append(
                self._safe_ratio(
                    abs(stats.net_volume_delta),
                    abs(self._config.bearish_delta_threshold),
                )
            )
        else:
            components.append(
                self._normalize_signed_magnitude(abs(stats.net_volume_delta))
            )

        components.append(
            self._safe_ratio(
                stats.burst_score,
                self._config.burst_score_threshold if self._config.burst_score_threshold > 0 else 1.0,
            )
        )

        if self._config.min_large_trades_for_signal > 0:
            components.append(
                self._safe_ratio(
                    float(stats.large_sell_trades),
                    float(self._config.min_large_trades_for_signal),
                )
            )
        else:
            components.append(
                self._normalize_signed_magnitude(float(stats.large_sell_trades))
            )

        return self._normalize_strength(components)

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