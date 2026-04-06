from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import Any, Optional

from core.event_bus import Event, EventBus

from .base import BaseOrderFlowAnalyzer
from .config import CvdConfig
from .enums import (
    OrderFlowMetricType,
    OrderFlowSide,
    OrderFlowSignalType,
    OrderFlowSourceType,
)
from .models import (
    BaseOrderFlowStats,
    CvdPoint,
    CvdStats,
    NormalizedTrade,
)


class CvdAnalyzer(BaseOrderFlowAnalyzer):
    """
    Analyzer для розрахунку CVD (Cumulative Volume Delta).

    Основні задачі:
    - приймає trade events через EventBus
    - читає останні трейди з trades_cache
    - нормалізує трейди
    - підтримує cumulative delta per symbol
    - будує window-based CVD stats
    - публікує update/signal події
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
        self._trades_cache = trades_cache
        self._config = config or (
            CvdConfig.from_app_config(app_config)
            if app_config is not None
            else CvdConfig()
        )

        super().__init__(
            event_bus=event_bus,
            config=self._config,
            metric_type=OrderFlowMetricType.CVD,
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
        self._cvd_points_by_symbol: dict[str, deque[CvdPoint]] = {}
        self._last_stats_by_symbol: dict[str, CvdStats] = {}

        self._last_seen_trade_key_by_symbol: dict[str, str] = {}
        self._cumulative_cvd_by_symbol: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_symbol(self, symbol: str) -> Optional[CvdStats]:
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
                    signed_volume = (
                        trade.quantity
                        if trade.side == OrderFlowSide.BUY
                        else -trade.quantity
                    )
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
                self._inc_metric("processed", symbol)
                if added_count > 0:
                    self._inc_metric("processed", symbol, amount=0)  # no-op, explicit
                if added_count > 0:
                    self._metrics.setdefault("processed_trades", 0)
                    self._metrics["processed_trades"] += added_count

                await self.emit_update(stats)

                signal = self._build_signal(stats)
                if signal is not None:
                    await self.emit_signal(signal)

                return stats

            except Exception:
                self._inc_metric("errors", symbol)
                self._logger.exception(
                    "Failed to process CVD | symbol=%s",
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
                "max_cvd_points_per_symbol": self._config.max_cvd_points_per_symbol,
                "min_trades_in_window": self._config.min_trades_in_window,
                "min_total_volume": self._config.min_total_volume,
                "bullish_delta_ratio_threshold": self._config.bullish_delta_ratio_threshold,
                "bearish_delta_ratio_threshold": self._config.bearish_delta_ratio_threshold,
                "bullish_cvd_change_threshold": self._config.bullish_cvd_change_threshold,
                "bearish_cvd_change_threshold": self._config.bearish_cvd_change_threshold,
                "bullish_cvd_slope_threshold": self._config.bullish_cvd_slope_threshold,
                "bearish_cvd_slope_threshold": self._config.bearish_cvd_slope_threshold,
                "bullish_impulse_threshold_pct": self._config.bullish_impulse_threshold_pct,
                "bearish_impulse_threshold_pct": self._config.bearish_impulse_threshold_pct,
                "require_delta_confirmation": self._config.require_delta_confirmation,
                "require_slope_confirmation": self._config.require_slope_confirmation,
            }
        )
        base_stats["tracked_symbols"] = len(self._trades_by_symbol)
        base_stats["metrics"]["processed_trades"] = self._metrics.get("processed_trades", 0)
        return base_stats

    async def cleanup(self) -> None:
        now = time.time()
        max_age = float(self._config.window_seconds) * 3.0

        symbols = set(self._trades_by_symbol.keys()) | set(self._cvd_points_by_symbol.keys())
        for symbol in symbols:
            trades = self._trades_by_symbol.get(symbol)
            points = self._cvd_points_by_symbol.get(symbol)

            latest_ts = 0.0
            if trades:
                latest_ts = max(latest_ts, trades[-1].timestamp)
            if points:
                latest_ts = max(latest_ts, points[-1].timestamp)

            if latest_ts <= 0:
                continue

            if (now - latest_ts) > max_age:
                self._trades_by_symbol.pop(symbol, None)
                self._cvd_points_by_symbol.pop(symbol, None)
                self._last_stats_by_symbol.pop(symbol, None)
                self._last_seen_trade_key_by_symbol.pop(symbol, None)
                self._cumulative_cvd_by_symbol.pop(symbol, None)
                self._last_signal_ts_by_symbol.pop(symbol, None)

                self._logger.debug(
                    "Removed stale CVD state | symbol=%s max_age=%s",
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

        # async API variants
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

            normalized.append(trade)

        normalized.sort(key=lambda item: (item.timestamp, item.trade_id or ""))
        return normalized

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
                # cache міг віддати інше вікно або пропустити останній ключ
                # у такому випадку перебираємо більш безпечно по timestamp/order
                new_trades = self._filter_new_trades_fallback(symbol, trades)

        if trades:
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

    def _prune_old_cvd_points(self, symbol: str) -> None:
        store = self._cvd_points_by_symbol.get(symbol)
        if not store:
            return

        cutoff_ts = time.time() - float(self._config.window_seconds)
        while store and store[0].timestamp < cutoff_ts:
            store.popleft()

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def _calculate_window_stats(self, symbol: str) -> Optional[CvdStats]:
        trades = self._trades_by_symbol.get(symbol)
        cvd_points = self._cvd_points_by_symbol.get(symbol)

        if not trades or not cvd_points:
            return None

        recent_trades = list(trades)
        recent_points = list(cvd_points)

        if len(recent_trades) < self._config.min_trades_in_window:
            return None

        buy_trades = [trade for trade in recent_trades if trade.side == OrderFlowSide.BUY]
        sell_trades = [trade for trade in recent_trades if trade.side == OrderFlowSide.SELL]

        buy_volume = sum(trade.quantity for trade in buy_trades)
        sell_volume = sum(trade.quantity for trade in sell_trades)
        total_volume = buy_volume + sell_volume

        if total_volume <= 0:
            return None

        if total_volume < self._config.min_total_volume:
            return None

        buy_notional = sum(trade.notional for trade in buy_trades)
        sell_notional = sum(trade.notional for trade in sell_trades)

        volume_delta = buy_volume - sell_volume
        notional_delta = buy_notional - sell_notional
        delta_ratio = volume_delta / total_volume

        buy_ratio = buy_volume / total_volume
        sell_ratio = sell_volume / total_volume

        total_notional = buy_notional + sell_notional
        avg_trade_size = total_volume / len(recent_trades)
        avg_trade_notional = total_notional / len(recent_trades) if total_notional > 0 else 0.0

        cvd_values = [point.value for point in recent_points]
        cvd_open = cvd_values[0]
        cvd_high = max(cvd_values)
        cvd_low = min(cvd_values)
        cvd_close = cvd_values[-1]
        cvd_value = cvd_close
        cvd_change = cvd_close - cvd_open

        cvd_change_pct = 0.0
        if cvd_open != 0:
            cvd_change_pct = (cvd_change / abs(cvd_open)) * 100.0

        cvd_slope = self._calculate_cvd_slope(recent_points)

        last_price = recent_trades[-1].price if recent_trades else None
        first_price = recent_trades[0].price if recent_trades else None

        price_change: Optional[float] = None
        price_change_pct: Optional[float] = None

        if first_price is not None and last_price is not None:
            price_change = last_price - first_price
            if first_price != 0:
                price_change_pct = (price_change / abs(first_price)) * 100.0
            else:
                price_change_pct = 0.0

        return CvdStats(
            symbol=symbol,
            metric=OrderFlowMetricType.CVD,
            source_type=OrderFlowSourceType.TRADES,
            window_seconds=float(self._config.window_seconds),
            trades_count=len(recent_trades),
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            volume_delta=volume_delta,
            buy_notional=buy_notional,
            sell_notional=sell_notional,
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

    def _calculate_cvd_slope(self, points: list[CvdPoint]) -> float:
        if len(points) < 2:
            return 0.0

        first = points[0]
        last = points[-1]

        delta_t = last.timestamp - first.timestamp
        if delta_t <= 0:
            return 0.0

        return (last.value - first.value) / delta_t

    # ------------------------------------------------------------------
    # Signal logic
    # ------------------------------------------------------------------

    def _build_signal(self, stats: CvdStats):
        bullish_checks = []
        bearish_checks = []

        bullish_checks.append(
            stats.delta_ratio >= self._config.bullish_delta_ratio_threshold
        )
        bearish_checks.append(
            stats.delta_ratio <= self._config.bearish_delta_ratio_threshold
        )

        bullish_checks.append(
            stats.cvd_change >= self._config.bullish_cvd_change_threshold
        )
        bearish_checks.append(
            stats.cvd_change <= self._config.bearish_cvd_change_threshold
        )

        if self._config.require_slope_confirmation:
            bullish_checks.append(
                stats.cvd_slope >= self._config.bullish_cvd_slope_threshold
            )
            bearish_checks.append(
                stats.cvd_slope <= self._config.bearish_cvd_slope_threshold
            )

        bullish_impulse_ok = True
        bearish_impulse_ok = True

        if self._config.bullish_impulse_threshold_pct > 0:
            bullish_impulse_ok = (
                stats.cvd_change_pct >= self._config.bullish_impulse_threshold_pct
            )
        if self._config.bearish_impulse_threshold_pct > 0:
            bearish_impulse_ok = (
                stats.cvd_change_pct <= -abs(self._config.bearish_impulse_threshold_pct)
            )

        if self._config.require_delta_confirmation:
            bullish_ok = all(bullish_checks) and bullish_impulse_ok
            bearish_ok = all(bearish_checks) and bearish_impulse_ok
        else:
            bullish_ok = any(bullish_checks) and bullish_impulse_ok
            bearish_ok = any(bearish_checks) and bearish_impulse_ok

        if bullish_ok and not bearish_ok:
            strength = self._calculate_bullish_strength(stats)
            return self.build_signal(
                symbol=stats.symbol,
                signal_type=OrderFlowSignalType.BULLISH,
                side=OrderFlowSide.BUY,
                strength=strength,
                reason="cvd_bullish_confirmation",
                context={
                    "cvd_value": stats.cvd_value,
                    "cvd_change": stats.cvd_change,
                    "cvd_change_pct": stats.cvd_change_pct,
                    "cvd_slope": stats.cvd_slope,
                    "delta_ratio": stats.delta_ratio,
                    "trades_count": stats.trades_count,
                    "last_price": stats.last_price,
                    "price_change": stats.price_change,
                    "price_change_pct": stats.price_change_pct,
                },
            )

        if bearish_ok and not bullish_ok:
            strength = self._calculate_bearish_strength(stats)
            return self.build_signal(
                symbol=stats.symbol,
                signal_type=OrderFlowSignalType.BEARISH,
                side=OrderFlowSide.SELL,
                strength=strength,
                reason="cvd_bearish_confirmation",
                context={
                    "cvd_value": stats.cvd_value,
                    "cvd_change": stats.cvd_change,
                    "cvd_change_pct": stats.cvd_change_pct,
                    "cvd_slope": stats.cvd_slope,
                    "delta_ratio": stats.delta_ratio,
                    "trades_count": stats.trades_count,
                    "last_price": stats.last_price,
                    "price_change": stats.price_change,
                    "price_change_pct": stats.price_change_pct,
                },
            )

        return None

    def _calculate_bullish_strength(self, stats: CvdStats) -> float:
        components: list[float] = []

        if self._config.bullish_delta_ratio_threshold != 0:
            components.append(
                self._safe_ratio(
                    stats.delta_ratio,
                    self._config.bullish_delta_ratio_threshold,
                )
            )

        if self._config.bullish_cvd_change_threshold > 0:
            components.append(
                self._safe_ratio(
                    stats.cvd_change,
                    self._config.bullish_cvd_change_threshold,
                )
            )
        else:
            components.append(self._normalize_signed_magnitude(stats.cvd_change))

        if self._config.bullish_cvd_slope_threshold > 0:
            components.append(
                self._safe_ratio(
                    stats.cvd_slope,
                    self._config.bullish_cvd_slope_threshold,
                )
            )
        else:
            components.append(self._normalize_signed_magnitude(stats.cvd_slope))

        return self._normalize_strength(components)

    def _calculate_bearish_strength(self, stats: CvdStats) -> float:
        components: list[float] = []

        if self._config.bearish_delta_ratio_threshold != 0:
            components.append(
                self._safe_ratio(
                    abs(stats.delta_ratio),
                    abs(self._config.bearish_delta_ratio_threshold),
                )
            )

        if self._config.bearish_cvd_change_threshold < 0:
            components.append(
                self._safe_ratio(
                    abs(stats.cvd_change),
                    abs(self._config.bearish_cvd_change_threshold),
                )
            )
        else:
            components.append(self._normalize_signed_magnitude(abs(stats.cvd_change)))

        if self._config.bearish_cvd_slope_threshold < 0:
            components.append(
                self._safe_ratio(
                    abs(stats.cvd_slope),
                    abs(self._config.bearish_cvd_slope_threshold),
                )
            )
        else:
            components.append(self._normalize_signed_magnitude(abs(stats.cvd_slope)))

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

        raw = sum(max(0.0, v) for v in values) / len(values)
        # м’яке стискання до [0, 1]
        return max(0.0, min(raw / (1.0 + raw), 1.0))