from __future__ import annotations
from core.logger import get_logger

import asyncio
import math
import time
from collections import deque
from typing import Any, Mapping

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
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    NormalizedTrade,
    OrderFlowKey,
    OrderFlowSignal,
    VolumeDeltaStats,
    orderflow_key_to_dict,
)


class VolumeDeltaAnalyzer(BaseOrderFlowAnalyzer):
    """
    Volume delta analyzer for analytics.orderflow.

    Responsibilities:
    - consume normalized data-layer trade updates from TradesCache;
    - read recent trades from trades_cache using scoped futures key;
    - normalize trades through BaseOrderFlowAnalyzer helpers;
    - maintain sliding trade windows per exchange + market_type + symbol + timeframe;
    - calculate volume delta, notional delta and cumulative delta stats;
    - emit analytics.orderflow.volume_delta.updated;
    - emit analytics.orderflow.volume_delta.signal;
    - use Scheduler-injected cleanup/health jobs from BaseOrderFlowAnalyzer.

    Correct input flow:
        exchange adapters
            -> market.trade
            -> TradesCache
            -> market.trades.updated
            -> VolumeDeltaAnalyzer
            -> analytics.orderflow.volume_delta.*

    Scope:
        exchange + market_type + symbol + timeframe
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        trades_cache: Any,
        config: VolumeDeltaConfig | None = None,
        scheduler: Scheduler | None = None,
        source_topic_patterns: list[str] | tuple[str, ...] | None = None,
        default_exchange: str | None = None,
        default_market_type: str = DEFAULT_MARKET_TYPE,
        default_timeframe: str = DEFAULT_TIMEFRAME,
    ) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__init__", _analytics_args)
        except Exception:
            pass
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
            default_exchange=default_exchange,
            default_market_type=default_market_type,
            default_timeframe=default_timeframe,
        )

        self._state_lock = asyncio.Lock()

        self._trades_by_key: dict[OrderFlowKey, deque[NormalizedTrade]] = {}
        self._last_stats_by_key: dict[OrderFlowKey, VolumeDeltaStats] = {}
        self._last_seen_trade_key_by_key: dict[OrderFlowKey, str] = {}

        self._metrics.setdefault("processed_trades", 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_key(self, key: OrderFlowKey) -> VolumeDeltaStats | None:
        """
        Process one scoped futures market.

        key:
            exchange + market_type + symbol + timeframe
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_key", _analytics_args)
        except Exception:
            pass
        if not self.should_process_key(key):
            self._inc_metric("skipped", key)
            return None

        async with self._state_lock:
            try:
                raw_trades = await self._get_recent_trades(key)
                if not raw_trades:
                    self._inc_metric("skipped", key)
                    return None

                normalized_trades = self._normalize_trades(
                    key=key,
                    raw_trades=raw_trades,
                )
                if not normalized_trades:
                    self._inc_metric("skipped", key)
                    return None

                new_trades = self._filter_new_trades(
                    key=key,
                    trades=normalized_trades,
                )
                if not new_trades and key not in self._trades_by_key:
                    self._inc_metric("skipped", key)
                    return None

                added_count = self._append_new_trades(
                    key=key,
                    trades=new_trades,
                )

                self._prune_old_trades(key)

                stats = self._calculate_window_stats(key)
                if stats is None:
                    self._inc_metric("skipped", key)
                    return None

                self._last_stats_by_key[key] = stats
                self._inc_metric("processed", key)

                if added_count > 0:
                    self._metrics["processed_trades"] += added_count

                await self.emit_update(stats)

                signal = self._build_signal(stats)
                if signal is not None:
                    await self.emit_signal(signal)

                return stats

            except Exception:
                self._inc_metric("errors", key)
                self._logger.exception(
                    "Failed to process volume delta",
                    extra=orderflow_key_to_dict(key),
                )
                return None

    def get_latest_stats_by_key(self, key: OrderFlowKey) -> BaseOrderFlowStats | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_latest_stats_by_key", _analytics_args)
        except Exception:
            pass
        return self._last_stats_by_key.get(key)

    def stats(self) -> dict[str, Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "stats", _analytics_args)
        except Exception:
            pass
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
        base_stats["tracked_keys"] = len(self._trades_by_key)
        base_stats["tracked_markets"] = [
            {
                **orderflow_key_to_dict(key),
                "trades": len(trades),
                "has_stats": key in self._last_stats_by_key,
            }
            for key, trades in self._trades_by_key.items()
        ]
        base_stats["metrics"]["processed_trades"] = self._metrics.get(
            "processed_trades",
            0,
        )
        return base_stats

    async def cleanup(self) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "cleanup", _analytics_args)
        except Exception:
            pass
        now = time.time()
        max_age = max(float(self._config.window_seconds) * 3.0, 60.0)

        for key, trades in list(self._trades_by_key.items()):
            if not trades:
                continue

            latest_ts = trades[-1].timestamp
            if (now - latest_ts) <= max_age:
                continue

            self._trades_by_key.pop(key, None)
            self._last_stats_by_key.pop(key, None)
            self._last_seen_trade_key_by_key.pop(key, None)
            self._last_signal_ts_by_key.pop(key, None)

            self._logger.debug(
                "Removed stale volume delta state",
                extra={
                    **orderflow_key_to_dict(key),
                    "max_age": max_age,
                },
            )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, event: Event) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_handle_event", _analytics_args)
        except Exception:
            pass
        key = self.extract_key_from_event(event)
        if key is None:
            self._logger.debug(
                "Trade event without scoped key skipped | topic=%s event_id=%s",
                event.topic,
                event.event_id,
            )
            self._inc_metric("skipped")
            return

        await self.process_key(key)

    # ------------------------------------------------------------------
    # Internal data loading
    # ------------------------------------------------------------------

    async def _get_recent_trades(self, key: OrderFlowKey) -> list[Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_get_recent_trades", _analytics_args)
        except Exception:
            pass
        if self._trades_cache is None:
            return []

        exchange, market_type, symbol, timeframe = key

        candidates: tuple[tuple[str, dict[str, Any]], ...] = (
            (
                "get_recent_trades",
                {
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "limit": self._config.max_trades_per_symbol,
                },
            ),
            (
                "get_recent_trades",
                {
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                    "limit": self._config.max_trades_per_symbol,
                },
            ),
            (
                "get_trades_since",
                {
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                    "since_ts": time.time() - float(self._config.window_seconds),
                },
            ),
            (
                "get_trades",
                {
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                    "limit": self._config.max_trades_per_symbol,
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
            method = getattr(self._trades_cache, method_name, None)
            if method is None:
                continue

            result = await self._call_cache_method(
                method=method,
                method_name=method_name,
                key=key,
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
        key: OrderFlowKey,
        kwargs: dict[str, Any],
    ) -> Any:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_call_cache_method", _analytics_args)
        except Exception:
            pass
        exchange, market_type, symbol, _timeframe = key

        # First try the full, canonical kwargs contract used by the new analytics layer.
        try:
            result = method(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except TypeError as exc:
            # Backward compatibility: some cache implementations, especially
            # TradesCache.get_recent_trades(), do not accept all canonical kwargs
            # such as timeframe or market_type. Retry with progressively narrower
            # keyword contracts before falling back to positional signatures.
            retry_kwargs_candidates: tuple[dict[str, Any], ...] = (
                {k: v for k, v in kwargs.items() if k != "timeframe"},
                {k: v for k, v in kwargs.items() if k not in {"timeframe", "market_type"}},
                {k: v for k, v in kwargs.items() if k in {"exchange", "symbol", "limit", "since_ts"}},
                {k: v for k, v in kwargs.items() if k in {"symbol", "limit", "since_ts"}},
            )

            seen: set[tuple[str, ...]] = set()
            for retry_kwargs in retry_kwargs_candidates:
                signature = tuple(sorted(retry_kwargs))
                if signature in seen or retry_kwargs == kwargs:
                    continue
                seen.add(signature)

                try:
                    result = method(**retry_kwargs)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return result
                except TypeError:
                    continue
                except Exception:
                    self._logger.exception(
                        "Failed to fetch recent trades",
                        extra={
                            **orderflow_key_to_dict(key),
                            "method": method_name,
                            "kwargs": retry_kwargs,
                        },
                    )
                    return None

            # Compatibility fallback for older positional cache APIs.
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
                        "Failed to fetch recent trades",
                        extra={
                            **orderflow_key_to_dict(key),
                            "method": method_name,
                            "args": args,
                        },
                    )
                    return None

            self._logger.debug(
                "Skipped incompatible cache method signature",
                extra={
                    **orderflow_key_to_dict(key),
                    "method": method_name,
                    "error": str(exc),
                },
            )
            return None

        except Exception:
            self._logger.exception(
                "Failed to fetch recent trades",
                extra={
                    **orderflow_key_to_dict(key),
                    "method": method_name,
                },
            )
            return None

    def _extract_trades_from_cache_result(self, result: Any) -> list[Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_trades_from_cache_result", _analytics_args)
        except Exception:
            pass
        if result is None:
            return []

        if isinstance(result, list):
            return result

        if isinstance(result, tuple):
            return list(result)

        if isinstance(result, deque):
            return list(result)

        if isinstance(result, Mapping):
            trades = result.get("trades")
            if isinstance(trades, list):
                return trades

            data = result.get("data")
            if isinstance(data, list):
                return data

            if isinstance(data, Mapping):
                nested_trades = data.get("trades")
                if isinstance(nested_trades, list):
                    return nested_trades

        return []

    def _normalize_trades(
        self,
        *,
        key: OrderFlowKey,
        raw_trades: list[Any],
    ) -> list[NormalizedTrade]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_trades", _analytics_args)
        except Exception:
            pass
        exchange, market_type, symbol, timeframe = key
        normalized: list[NormalizedTrade] = []

        for raw_trade in raw_trades:
            trade = self.normalize_trade(
                raw_trade,
                default_exchange=exchange,
                default_market_type=market_type,
                default_symbol=symbol,
                default_timeframe=timeframe,
            )
            if trade is None:
                continue

            if trade.key != key:
                # Safety guard: cache result must not leak another market into this key.
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
        *,
        key: OrderFlowKey,
        trades: list[NormalizedTrade],
    ) -> int:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_append_new_trades", _analytics_args)
        except Exception:
            pass
        if not trades:
            return 0

        store = self._trades_by_key.setdefault(
            key,
            deque(maxlen=self._config.max_trades_per_symbol),
        )

        for trade in trades:
            store.append(trade)

        return len(trades)

    def _filter_new_trades(
        self,
        *,
        key: OrderFlowKey,
        trades: list[NormalizedTrade],
    ) -> list[NormalizedTrade]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_filter_new_trades", _analytics_args)
        except Exception:
            pass
        if not trades:
            return []

        last_seen_key = self._last_seen_trade_key_by_key.get(key)

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
                new_trades = self._filter_new_trades_fallback(key, trades)

        self._last_seen_trade_key_by_key[key] = self.make_trade_key(trades[-1])
        return new_trades

    def _collect_trades_after_last_seen(
        self,
        *,
        trades: list[NormalizedTrade],
        last_seen_key: str,
    ) -> list[NormalizedTrade]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_collect_trades_after_last_seen", _analytics_args)
        except Exception:
            pass
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
        key: OrderFlowKey,
        trades: list[NormalizedTrade],
    ) -> list[NormalizedTrade]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_filter_new_trades_fallback", _analytics_args)
        except Exception:
            pass
        existing = self._trades_by_key.get(key)
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

    def _prune_old_trades(self, key: OrderFlowKey) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_prune_old_trades", _analytics_args)
        except Exception:
            pass
        store = self._trades_by_key.get(key)
        if not store:
            return

        cutoff_ts = time.time() - float(self._config.window_seconds)

        while store and store[0].timestamp < cutoff_ts:
            store.popleft()

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def _calculate_window_stats(self, key: OrderFlowKey) -> VolumeDeltaStats | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calculate_window_stats", _analytics_args)
        except Exception:
            pass
        trades = self._trades_by_key.get(key)
        if not trades:
            return None

        recent_trades = list(trades)

        if len(recent_trades) < self._config.min_trades_in_window:
            return None

        buy_trades = [
            trade
            for trade in recent_trades
            if trade.side == OrderFlowSide.BUY
        ]
        sell_trades = [
            trade
            for trade in recent_trades
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

        last_trade = recent_trades[-1]

        return VolumeDeltaStats(
            exchange=last_trade.exchange,
            market_type=last_trade.market_type,
            symbol=last_trade.symbol,
            exchange_symbol=last_trade.exchange_symbol,
            timeframe=last_trade.timeframe,
            metric=OrderFlowMetricType.VOLUME_DELTA,
            source_type=OrderFlowSourceType.TRADES,
            timestamp=time.time(),
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
            last_price=last_trade.price,
            metadata={
                "trades_window_start_ts": recent_trades[0].timestamp,
                "trades_window_end_ts": recent_trades[-1].timestamp,
                "trades_in_store": len(recent_trades),
                "scope": "exchange:market_type:symbol:timeframe",
            },
        )

    # ------------------------------------------------------------------
    # Signal logic
    # ------------------------------------------------------------------

    def _build_signal(self, stats: VolumeDeltaStats) -> OrderFlowSignal | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_signal", _analytics_args)
        except Exception:
            pass
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
            return self.build_signal_from_stats(
                stats=stats,
                signal_type=OrderFlowSignalType.BULLISH,
                side=OrderFlowSide.BUY,
                strength=self._calculate_bullish_strength(stats),
                reason="volume_delta_bullish_confirmation",
                context=self._build_signal_context(stats),
            )

        if bearish_ok and not bullish_ok:
            return self.build_signal_from_stats(
                stats=stats,
                signal_type=OrderFlowSignalType.BEARISH,
                side=OrderFlowSide.SELL,
                strength=self._calculate_bearish_strength(stats),
                reason="volume_delta_bearish_confirmation",
                context=self._build_signal_context(stats),
            )

        return None

    def _build_signal_context(self, stats: VolumeDeltaStats) -> dict[str, Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_signal_context", _analytics_args)
        except Exception:
            pass
        return {
            "exchange": stats.exchange,
            "market_type": stats.market_type,
            "symbol": stats.symbol,
            "exchange_symbol": stats.exchange_symbol,
            "timeframe": stats.timeframe,
            "key": list(stats.key),
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
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calculate_bullish_strength", _analytics_args)
        except Exception:
            pass
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
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calculate_bearish_strength", _analytics_args)
        except Exception:
            pass
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
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_safe_ratio", _analytics_args)
        except Exception:
            pass
        if threshold == 0:
            return self._normalize_magnitude(value)

        return max(0.0, value / threshold)

    def _normalize_magnitude(self, value: float) -> float:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_magnitude", _analytics_args)
        except Exception:
            pass
        return math.log1p(abs(value))

    def _normalize_strength(self, values: list[float]) -> float:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_strength", _analytics_args)
        except Exception:
            pass
        if not values:
            return 0.0

        raw = sum(max(0.0, value) for value in values) / len(values)
        return max(0.0, min(raw / (1.0 + raw), 1.0))