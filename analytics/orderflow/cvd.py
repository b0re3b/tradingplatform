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
from .config import CvdConfig
from .enums import (
    TRADE_INPUT_TOPICS,
    OrderFlowMetricType,
    OrderFlowSide,
    OrderFlowSignalType,
    OrderFlowSourceType,
)
from .models import (
    BaseOrderFlowStats,
    CvdPoint,
    CvdStats,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    NormalizedTrade,
    OrderFlowKey,
    OrderFlowSignal,
    orderflow_key_to_dict,
)


class CvdAnalyzer(BaseOrderFlowAnalyzer):
    """
    CVD analyzer for analytics.orderflow.

    Responsibilities:
    - consume normalized data-layer trade updates from TradesCache;
    - read recent trades from trades_cache using scoped futures key;
    - normalize trades through BaseOrderFlowAnalyzer helpers;
    - maintain cumulative volume delta per exchange + market_type + symbol + timeframe;
    - calculate window-based CVD stats;
    - emit analytics.orderflow.cvd.updated;
    - emit analytics.orderflow.cvd.signal;
    - use Scheduler-injected cleanup/health jobs from BaseOrderFlowAnalyzer.

    Correct input flow:
        exchange adapters
            -> market.trade
            -> TradesCache
            -> market.trades.updated
            -> CvdAnalyzer
            -> analytics.orderflow.cvd.*

    Scope:
        exchange + market_type + symbol + timeframe
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        trades_cache: Any,
        config: CvdConfig | None = None,
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
        self._config = config or CvdConfig()

        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=self._config,
            metric_type=OrderFlowMetricType.CVD,
            source_type=OrderFlowSourceType.TRADES,
            source_topic_patterns=source_topic_patterns or TRADE_INPUT_TOPICS,
            component_module="orderflow",
            default_exchange=default_exchange,
            default_market_type=default_market_type,
            default_timeframe=default_timeframe,
        )

        self._state_lock = asyncio.Lock()

        self._trades_by_key: dict[OrderFlowKey, deque[NormalizedTrade]] = {}
        self._cvd_points_by_key: dict[OrderFlowKey, deque[CvdPoint]] = {}
        self._last_stats_by_key: dict[OrderFlowKey, CvdStats] = {}
        self._last_seen_trade_key_by_key: dict[OrderFlowKey, str] = {}
        self._cumulative_cvd_by_key: dict[OrderFlowKey, float] = {}

        self._metrics.setdefault("processed_trades", 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_key(self, key: OrderFlowKey) -> CvdStats | None:
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
                self._prune_old_cvd_points(key)

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
                    "Failed to process CVD",
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
        base_stats["tracked_keys"] = len(self._trades_by_key)
        base_stats["tracked_markets"] = [
            {
                **orderflow_key_to_dict(key),
                "trades": len(self._trades_by_key.get(key, ())),
                "cvd_points": len(self._cvd_points_by_key.get(key, ())),
                "has_stats": key in self._last_stats_by_key,
                "cumulative_cvd": self._cumulative_cvd_by_key.get(key, 0.0),
            }
            for key in sorted(self._tracked_keys())
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

        for key in list(self._tracked_keys()):
            latest_ts = self._latest_key_timestamp(key)
            if latest_ts <= 0:
                continue

            if (now - latest_ts) <= max_age:
                continue

            self._trades_by_key.pop(key, None)
            self._cvd_points_by_key.pop(key, None)
            self._last_stats_by_key.pop(key, None)
            self._last_seen_trade_key_by_key.pop(key, None)
            self._cumulative_cvd_by_key.pop(key, None)
            self._last_signal_ts_by_key.pop(key, None)

            self._logger.debug(
                "Removed stale CVD state",
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

        trade_store = self._trades_by_key.setdefault(
            key,
            deque(maxlen=self._config.max_trades_per_symbol),
        )
        cvd_store = self._cvd_points_by_key.setdefault(
            key,
            deque(maxlen=self._config.max_cvd_points_per_symbol),
        )

        added_count = 0

        for trade in trades:
            trade_store.append(trade)

            previous_cvd = self._cumulative_cvd_by_key.get(key, 0.0)
            new_cvd_value = previous_cvd + trade.signed_volume
            self._cumulative_cvd_by_key[key] = new_cvd_value

            cvd_store.append(
                CvdPoint(
                    exchange=trade.exchange,
                    market_type=trade.market_type,
                    symbol=trade.symbol,
                    exchange_symbol=trade.exchange_symbol,
                    timeframe=trade.timeframe,
                    timestamp=trade.timestamp,
                    value=new_cvd_value,
                    price=trade.price,
                )
            )
            added_count += 1

        return added_count

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

    def _prune_old_cvd_points(self, key: OrderFlowKey) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_prune_old_cvd_points", _analytics_args)
        except Exception:
            pass
        store = self._cvd_points_by_key.get(key)
        if not store:
            return

        cutoff_ts = time.time() - float(self._config.window_seconds)

        while store and store[0].timestamp < cutoff_ts:
            store.popleft()

    def _latest_key_timestamp(self, key: OrderFlowKey) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_latest_key_timestamp", _analytics_args)
        except Exception:
            pass
        latest_ts = 0.0

        trades = self._trades_by_key.get(key)
        if trades:
            latest_ts = max(latest_ts, trades[-1].timestamp)

        points = self._cvd_points_by_key.get(key)
        if points:
            latest_ts = max(latest_ts, points[-1].timestamp)

        return latest_ts

    def _tracked_keys(self) -> set[OrderFlowKey]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_tracked_keys", _analytics_args)
        except Exception:
            pass
        return (
            set(self._trades_by_key)
            | set(self._cvd_points_by_key)
            | set(self._last_stats_by_key)
            | set(self._cumulative_cvd_by_key)
        )

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def _calculate_window_stats(self, key: OrderFlowKey) -> CvdStats | None:
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
        cvd_points = self._cvd_points_by_key.get(key)

        if not trades or not cvd_points:
            return None

        recent_trades = list(trades)
        recent_points = list(cvd_points)

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

        cvd_values = [point.value for point in recent_points]
        cvd_open = cvd_values[0]
        cvd_close = cvd_values[-1]
        cvd_change = cvd_close - cvd_open

        first_price = recent_trades[0].price
        last_price = recent_trades[-1].price

        price_change = last_price - first_price
        price_change_pct = (
            (price_change / abs(first_price)) * 100.0
            if first_price != 0
            else 0.0
        )

        last_trade = recent_trades[-1]

        return CvdStats(
            exchange=last_trade.exchange,
            market_type=last_trade.market_type,
            symbol=last_trade.symbol,
            exchange_symbol=last_trade.exchange_symbol,
            timeframe=last_trade.timeframe,
            metric=OrderFlowMetricType.CVD,
            source_type=OrderFlowSourceType.TRADES,
            timestamp=time.time(),
            window_seconds=float(self._config.window_seconds),
            trades_count=len(recent_trades),
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            volume_delta=volume_delta,
            buy_notional=buy_notional,
            sell_notional=sell_notional,
            notional_delta=notional_delta,
            cvd_value=cvd_close,
            cvd_open=cvd_open,
            cvd_high=max(cvd_values),
            cvd_low=min(cvd_values),
            cvd_close=cvd_close,
            cvd_change=cvd_change,
            cvd_change_pct=self._calculate_percent_change(
                current=cvd_close,
                previous=cvd_open,
            ),
            cvd_slope=self._calculate_cvd_slope(recent_points),
            delta_ratio=delta_ratio,
            buy_ratio=buy_volume / total_volume,
            sell_ratio=sell_volume / total_volume,
            avg_trade_size=total_volume / len(recent_trades),
            avg_trade_notional=(
                total_notional / len(recent_trades)
                if total_notional > 0
                else 0.0
            ),
            last_price=last_price,
            price_change=price_change,
            price_change_pct=price_change_pct,
            metadata={
                "trades_window_start_ts": recent_trades[0].timestamp,
                "trades_window_end_ts": recent_trades[-1].timestamp,
                "cvd_points_in_window": len(recent_points),
                "trades_in_store": len(recent_trades),
                "scope": "exchange:market_type:symbol:timeframe",
            },
        )

    def _calculate_cvd_slope(self, points: list[CvdPoint]) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calculate_cvd_slope", _analytics_args)
        except Exception:
            pass
        if len(points) < 2:
            return 0.0

        first = points[0]
        last = points[-1]

        delta_t = last.timestamp - first.timestamp
        if delta_t <= 0:
            return 0.0

        return (last.value - first.value) / delta_t

    def _calculate_percent_change(self, *, current: float, previous: float) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calculate_percent_change", _analytics_args)
        except Exception:
            pass
        if previous == 0:
            return 0.0

        return ((current - previous) / abs(previous)) * 100.0

    # ------------------------------------------------------------------
    # Signal logic
    # ------------------------------------------------------------------

    def _build_signal(self, stats: CvdStats) -> OrderFlowSignal | None:
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
        bullish_checks = self._build_bullish_checks(stats)
        bearish_checks = self._build_bearish_checks(stats)

        bullish_impulse_ok = self._is_bullish_impulse_ok(stats)
        bearish_impulse_ok = self._is_bearish_impulse_ok(stats)

        if self._config.require_delta_confirmation:
            bullish_ok = all(bullish_checks) and bullish_impulse_ok
            bearish_ok = all(bearish_checks) and bearish_impulse_ok
        else:
            bullish_ok = any(bullish_checks) and bullish_impulse_ok
            bearish_ok = any(bearish_checks) and bearish_impulse_ok

        if bullish_ok and not bearish_ok:
            return self.build_signal_from_stats(
                stats=stats,
                signal_type=OrderFlowSignalType.BULLISH,
                side=OrderFlowSide.BUY,
                strength=self._calculate_bullish_strength(stats),
                reason="cvd_bullish_confirmation",
                context=self._build_signal_context(stats),
            )

        if bearish_ok and not bullish_ok:
            return self.build_signal_from_stats(
                stats=stats,
                signal_type=OrderFlowSignalType.BEARISH,
                side=OrderFlowSide.SELL,
                strength=self._calculate_bearish_strength(stats),
                reason="cvd_bearish_confirmation",
                context=self._build_signal_context(stats),
            )

        return None

    def _build_bullish_checks(self, stats: CvdStats) -> list[bool]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_bullish_checks", _analytics_args)
        except Exception:
            pass
        checks = [
            stats.delta_ratio >= self._config.bullish_delta_ratio_threshold,
            stats.cvd_change >= self._config.bullish_cvd_change_threshold,
        ]

        if self._config.require_slope_confirmation:
            checks.append(stats.cvd_slope >= self._config.bullish_cvd_slope_threshold)

        return checks

    def _build_bearish_checks(self, stats: CvdStats) -> list[bool]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_bearish_checks", _analytics_args)
        except Exception:
            pass
        checks = [
            stats.delta_ratio <= self._config.bearish_delta_ratio_threshold,
            stats.cvd_change <= self._config.bearish_cvd_change_threshold,
        ]

        if self._config.require_slope_confirmation:
            checks.append(stats.cvd_slope <= self._config.bearish_cvd_slope_threshold)

        return checks

    def _is_bullish_impulse_ok(self, stats: CvdStats) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_bullish_impulse_ok", _analytics_args)
        except Exception:
            pass
        if self._config.bullish_impulse_threshold_pct <= 0:
            return True

        return stats.cvd_change_pct >= self._config.bullish_impulse_threshold_pct

    def _is_bearish_impulse_ok(self, stats: CvdStats) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_bearish_impulse_ok", _analytics_args)
        except Exception:
            pass
        if self._config.bearish_impulse_threshold_pct <= 0:
            return True

        return stats.cvd_change_pct <= -abs(self._config.bearish_impulse_threshold_pct)

    def _build_signal_context(self, stats: CvdStats) -> dict[str, Any]:
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
            "cvd_value": stats.cvd_value,
            "cvd_change": stats.cvd_change,
            "cvd_change_pct": stats.cvd_change_pct,
            "cvd_slope": stats.cvd_slope,
            "delta_ratio": stats.delta_ratio,
            "buy_ratio": stats.buy_ratio,
            "sell_ratio": stats.sell_ratio,
            "trades_count": stats.trades_count,
            "last_price": stats.last_price,
            "price_change": stats.price_change,
            "price_change_pct": stats.price_change_pct,
        }

    def _calculate_bullish_strength(self, stats: CvdStats) -> float:
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

        if self._config.bullish_cvd_change_threshold > 0:
            components.append(
                self._safe_ratio(
                    stats.cvd_change,
                    self._config.bullish_cvd_change_threshold,
                )
            )
        else:
            components.append(self._normalize_magnitude(stats.cvd_change))

        if self._config.bullish_cvd_slope_threshold > 0:
            components.append(
                self._safe_ratio(
                    stats.cvd_slope,
                    self._config.bullish_cvd_slope_threshold,
                )
            )
        else:
            components.append(self._normalize_magnitude(stats.cvd_slope))

        return self._normalize_strength(components)

    def _calculate_bearish_strength(self, stats: CvdStats) -> float:
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

        if self._config.bearish_cvd_change_threshold < 0:
            components.append(
                self._safe_ratio(
                    abs(stats.cvd_change),
                    abs(self._config.bearish_cvd_change_threshold),
                )
            )
        else:
            components.append(self._normalize_magnitude(stats.cvd_change))

        if self._config.bearish_cvd_slope_threshold < 0:
            components.append(
                self._safe_ratio(
                    abs(stats.cvd_slope),
                    abs(self._config.bearish_cvd_slope_threshold),
                )
            )
        else:
            components.append(self._normalize_magnitude(stats.cvd_slope))

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