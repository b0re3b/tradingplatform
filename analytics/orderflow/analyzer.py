from __future__ import annotations

import asyncio
from typing import Any, TypeAlias, Union

from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler

from .aggressive_trades import AggressiveTradesAnalyzer
from .config import OrderFlowConfig
from .cvd import CvdAnalyzer
from .enums import OrderFlowEventTopic
from .models import (
    BaseOrderFlowStats,
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    OrderFlowKey,
    make_orderflow_key,
    normalize_exchange,
    normalize_market_type,
    normalize_timeframe,
    orderflow_key_to_dict,
    orderflow_key_to_string,
)
from .orderbook_imbalance import OrderbookImbalanceAnalyzer
from .volume_delta import VolumeDeltaAnalyzer


OrderFlowModule: TypeAlias = Union[
    CvdAnalyzer,
    VolumeDeltaAnalyzer,
    AggressiveTradesAnalyzer,
    OrderbookImbalanceAnalyzer,
]


class OrderFlowAnalyzer:
    """
    Facade for analytics.orderflow package.

    Responsibilities:
    - own and wire all order-flow analyzers;
    - inject EventBus / Scheduler / Config / data caches into concrete modules;
    - centrally register/stop all submodules;
    - expose aggregated stats and latest scoped futures state;
    - optionally run all sub-analyzers manually for one futures scope.

    This class is an orchestration layer only.
    It must not contain trading logic or strategy decisions.

    Correct input flow:
        exchange adapters
            -> market.trade / market.orderbook
            -> TradesCache / OrderbookCache
            -> market.trades.updated / market.orderbook.updated
            -> analytics.orderflow
            -> analytics.orderflow.*

    Scope:
        exchange + market_type + symbol + timeframe

    Futures examples:
        ("binance", "usdm_futures", "BTCUSDT", "1m")
        ("bybit", "linear", "BTCUSDT", "1m")
        ("okx", "swap", "BTCUSDT", "1m")
        ("mexc", "usdm_futures", "BTCUSDT", "1m")
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        trades_cache: Any,
        orderbook_cache: Any,
        config: OrderFlowConfig | None = None,
        scheduler: Scheduler | None = None,
        trades_topic_patterns: list[str] | tuple[str, ...] | None = None,
        orderbook_topic_patterns: list[str] | tuple[str, ...] | None = None,
        default_exchange: str | None = None,
        default_market_type: str | None = None,
        default_timeframe: str | None = None,
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
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._trades_cache = trades_cache
        self._orderbook_cache = orderbook_cache
        self._config = config or OrderFlowConfig()

        self._config.validate()
        self._config.assert_production_topics_allowed()

        self._default_exchange = normalize_exchange(
            default_exchange or self._config.default_exchange or DEFAULT_EXCHANGE
        )
        self._default_market_type = normalize_market_type(
            default_market_type or self._config.default_market_type or DEFAULT_MARKET_TYPE
        )
        self._default_timeframe = normalize_timeframe(
            default_timeframe or self._config.default_timeframe or DEFAULT_TIMEFRAME
        )

        self._trades_topic_patterns = self._normalize_topics(
            trades_topic_patterns
            if trades_topic_patterns is not None
            else self._config.trades_topics
        )
        self._orderbook_topic_patterns = self._normalize_topics(
            orderbook_topic_patterns
            if orderbook_topic_patterns is not None
            else self._config.orderbook_topics
        )

        for topic in (*self._trades_topic_patterns, *self._orderbook_topic_patterns):
            self._config.assert_input_topic_allowed(topic)

        self._logger = get_logger(
            __name__,
            service_name="orderflow",
            component="analytics",
            component_module="orderflow",
            event_type="orderflow_facade",
        )

        self.cvd = CvdAnalyzer(
            event_bus=self._event_bus,
            scheduler=self._scheduler,
            trades_cache=self._trades_cache,
            config=self._config.cvd,
            source_topic_patterns=self._trades_topic_patterns,
            default_exchange=self._default_exchange,
            default_market_type=self._default_market_type,
            default_timeframe=self._default_timeframe,
        )

        self.volume_delta = VolumeDeltaAnalyzer(
            event_bus=self._event_bus,
            scheduler=self._scheduler,
            trades_cache=self._trades_cache,
            config=self._config.volume_delta,
            source_topic_patterns=self._trades_topic_patterns,
            default_exchange=self._default_exchange,
            default_market_type=self._default_market_type,
            default_timeframe=self._default_timeframe,
        )

        self.aggressive_trades = AggressiveTradesAnalyzer(
            event_bus=self._event_bus,
            scheduler=self._scheduler,
            trades_cache=self._trades_cache,
            config=self._config.aggressive_trades,
            source_topic_patterns=self._trades_topic_patterns,
            default_exchange=self._default_exchange,
            default_market_type=self._default_market_type,
            default_timeframe=self._default_timeframe,
        )

        self.orderbook_imbalance = OrderbookImbalanceAnalyzer(
            event_bus=self._event_bus,
            scheduler=self._scheduler,
            orderbook_cache=self._orderbook_cache,
            config=self._config.orderbook_imbalance,
            source_topic_patterns=self._orderbook_topic_patterns,
            default_exchange=self._default_exchange,
            default_market_type=self._default_market_type,
            default_timeframe=self._default_timeframe,
        )

        self._modules: dict[str, OrderFlowModule] = {
            "cvd": self.cvd,
            "volume_delta": self.volume_delta,
            "aggressive_trades": self.aggressive_trades,
            "orderbook_imbalance": self.orderbook_imbalance,
        }

        self._running = False
        self._lifecycle_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def register(self) -> None:
        """
        Register all enabled order-flow submodules.

        This is the standard project lifecycle entrypoint.
        start() is kept as a compatibility alias.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "register", _analytics_args)
        except Exception:
            pass
        async with self._lifecycle_lock:
            if self._running:
                self._logger.warning("OrderFlowAnalyzer already registered")
                return

            self._config.validate()
            self._config.assert_production_topics_allowed()

            if not self._config.enabled:
                self._logger.warning("OrderFlowAnalyzer is disabled by config")
                await self._emit_lifecycle_event(
                    OrderFlowEventTopic.STOPPED.value,
                    {
                        "reason": "disabled_by_config",
                        "enabled": False,
                        "scope": "exchange:market_type:symbol:timeframe",
                        "defaults": self._defaults_payload(),
                        "input_topics": list(self._config.production_input_topics),
                        "output_topics": list(self._config.output_topics),
                    },
                    priority=EventPriority.LOW,
                    event_type="orderflow_disabled",
                )
                return

            registered_modules: list[str] = []

            for name, module in self._modules.items():
                if not self._is_module_enabled(name):
                    self._logger.info(
                        "OrderFlow module skipped because it is disabled",
                        extra={"module_name": name},
                    )
                    continue

                try:
                    module.register()
                    registered_modules.append(name)
                except Exception:
                    self._logger.exception(
                        "Failed to register order-flow module",
                        extra={"module_name": name},
                    )
                    raise

            self._running = True

            self._logger.info(
                "OrderFlowAnalyzer registered",
                extra={
                    "modules": registered_modules,
                    "trades_topic_patterns": list(self._trades_topic_patterns),
                    "orderbook_topic_patterns": list(self._orderbook_topic_patterns),
                    "scope": "exchange:market_type:symbol:timeframe",
                    "defaults": self._defaults_payload(),
                    "input_topics": list(self._config.production_input_topics),
                    "output_topics": list(self._config.output_topics),
                    "enabled_modules": list(self.enabled_modules()),
                },
            )

            await self._emit_lifecycle_event(
                OrderFlowEventTopic.STARTED.value,
                {
                    "enabled": True,
                    "modules": registered_modules,
                    "enabled_modules": list(self.enabled_modules()),
                    "trades_topic_patterns": list(self._trades_topic_patterns),
                    "orderbook_topic_patterns": list(self._orderbook_topic_patterns),
                    "input_topics": list(self._config.production_input_topics),
                    "output_topics": list(self._config.output_topics),
                    "scope": "exchange:market_type:symbol:timeframe",
                    "defaults": self._defaults_payload(),
                    "config": self._config.to_dict(),
                },
                priority=EventPriority.NORMAL,
                event_type="orderflow_started",
            )

    async def start(self) -> None:
        """
        Backward-compatible alias.

        New code should call register().
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "start", _analytics_args)
        except Exception:
            pass
        await self.register()

    async def stop(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "stop", _analytics_args)
        except Exception:
            pass
        async with self._lifecycle_lock:
            if not self._running:
                self._logger.warning("OrderFlowAnalyzer already stopped")
                return

            stopped_modules: list[str] = []

            for name, module in reversed(list(self._modules.items())):
                try:
                    module.stop()
                    stopped_modules.append(name)
                except Exception:
                    self._logger.exception(
                        "Failed to stop order-flow module",
                        extra={"module_name": name},
                    )

            self._running = False

            self._logger.info(
                "OrderFlowAnalyzer stopped",
                extra={
                    "modules": stopped_modules,
                    "scope": "exchange:market_type:symbol:timeframe",
                },
            )

            await self._emit_lifecycle_event(
                OrderFlowEventTopic.STOPPED.value,
                {
                    "enabled": self._config.enabled,
                    "modules": stopped_modules,
                    "scope": "exchange:market_type:symbol:timeframe",
                    "defaults": self._defaults_payload(),
                    "input_topics": list(self._config.production_input_topics),
                    "output_topics": list(self._config.output_topics),
                },
                priority=EventPriority.NORMAL,
                event_type="orderflow_stopped",
            )

    @property
    def is_running(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_running", _analytics_args)
        except Exception:
            pass
        return self._running

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_module(self, name: str) -> OrderFlowModule | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_module", _analytics_args)
        except Exception:
            pass
        return self._modules.get(str(name).strip().lower())

    def list_modules(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "list_modules", _analytics_args)
        except Exception:
            pass
        return tuple(self._modules.keys())

    def enabled_modules(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "enabled_modules", _analytics_args)
        except Exception:
            pass
        return self._config.enabled_modules()

    def get_latest_stats_by_key(
        self,
        key: OrderFlowKey,
    ) -> dict[str, BaseOrderFlowStats | None | dict[str, str] | list[str]]:
        """
        Return latest stats for one scoped futures market.

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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_latest_stats_by_key", _analytics_args)
        except Exception:
            pass
        normalized_key = self._normalize_key_from_tuple(key)

        if not self._config.should_process_key(normalized_key):
            self._logger.debug(
                "Latest order-flow stats skipped by scope filter",
                extra=self._key_payload(normalized_key),
            )
            return self._empty_key_result(normalized_key)

        return {
            **self._key_result_base(normalized_key),
            "cvd": self.cvd.get_latest_stats_by_key(normalized_key),
            "volume_delta": self.volume_delta.get_latest_stats_by_key(normalized_key),
            "aggressive_trades": self.aggressive_trades.get_latest_stats_by_key(
                normalized_key
            ),
            "orderbook_imbalance": self.orderbook_imbalance.get_latest_stats_by_key(
                normalized_key
            ),
        }

    def get_latest_stats(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> dict[str, BaseOrderFlowStats | None | dict[str, str] | list[str]]:
        """
        Scoped latest-stats API.

        Use this instead of the old symbol-only form.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_latest_stats", _analytics_args)
        except Exception:
            pass
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        return self.get_latest_stats_by_key(key)

    async def process_key(
        self,
        key: OrderFlowKey,
    ) -> dict[str, BaseOrderFlowStats | None | dict[str, str] | list[str]]:
        """
        Manually process one scoped futures market across all enabled sub-analyzers.

        Normal live operation should happen through EventBus subscriptions.
        This method is useful for tests, backtests, warmup, admin actions,
        or one-shot recalculation.
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
        normalized_key = self._normalize_key_from_tuple(key)

        if not self._config.enabled:
            self._logger.warning(
                "Manual order-flow processing skipped: facade disabled",
                extra=self._key_payload(normalized_key),
            )
            return self._empty_key_result(normalized_key)

        if not self._config.should_process_key(normalized_key):
            self._logger.debug(
                "Manual order-flow processing skipped by scope filter",
                extra=self._key_payload(normalized_key),
            )
            return self._empty_key_result(normalized_key)

        results: dict[str, BaseOrderFlowStats | None | dict[str, str] | list[str]] = (
            self._empty_key_result(normalized_key)
        )

        for name, module in self._modules.items():
            if not self._is_module_enabled(name):
                continue

            try:
                results[name] = await module.process_key(normalized_key)
            except Exception as exc:
                self._logger.exception(
                    "Manual order-flow module processing failed",
                    extra={
                        **self._key_payload(normalized_key),
                        "module_name": name,
                        "error": repr(exc),
                    },
                )
                await self._emit_lifecycle_event(
                    OrderFlowEventTopic.ERROR.value,
                    {
                        **self._key_payload(normalized_key),
                        "module": name,
                        "reason": "manual_process_failed",
                        "error": repr(exc),
                    },
                    priority=EventPriority.HIGH,
                    key=normalized_key,
                    event_type="orderflow_manual_process_failed",
                )

        return results

    async def process_market(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> dict[str, BaseOrderFlowStats | None | dict[str, str] | list[str]]:
        """
        Scoped convenience wrapper for manual processing.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_market", _analytics_args)
        except Exception:
            pass
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        return await self.process_key(key)

    async def process_symbol(
        self,
        symbol: str,
    ) -> dict[str, BaseOrderFlowStats | None | dict[str, str] | list[str]]:
        """
        Backward-compatible wrapper.

        Symbol-only processing uses explicit facade defaults.
        Prefer process_market(...) or process_key(...).
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_symbol", _analytics_args)
        except Exception:
            pass
        key = self.make_key(
            exchange=self._default_exchange,
            market_type=self._default_market_type,
            symbol=symbol,
            timeframe=self._default_timeframe,
        )
        return await self.process_key(key)

    async def cleanup(self) -> None:
        """
        Run cleanup hooks for all modules.

        Normally cleanup is scheduled per module by BaseOrderFlowAnalyzer using
        core.scheduler.Scheduler. This method is useful for manual maintenance.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "cleanup", _analytics_args)
        except Exception:
            pass
        for name, module in self._modules.items():
            try:
                result = module.cleanup()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                self._logger.exception(
                    "Order-flow cleanup failed",
                    extra={"module_name": name, "error": repr(exc)},
                )
                await self._emit_lifecycle_event(
                    OrderFlowEventTopic.ERROR.value,
                    {
                        "module": name,
                        "reason": "cleanup_failed",
                        "error": repr(exc),
                        "scope": "exchange:market_type:symbol:timeframe",
                    },
                    priority=EventPriority.HIGH,
                    event_type="orderflow_cleanup_failed",
                )

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
        return {
            "running": self._running,
            "enabled": self._config.enabled,
            "enabled_modules": self.enabled_modules(),
            "scope": "exchange:market_type:symbol:timeframe",
            "defaults": self._defaults_payload(),
            "trades_topic_patterns": list(self._trades_topic_patterns),
            "orderbook_topic_patterns": list(self._orderbook_topic_patterns),
            "input_topics": list(self._config.production_input_topics),
            "output_topics": list(self._config.output_topics),
            "scheduler_job_names": list(self._config.scheduler_job_names),
            "scheduler_attached": self._scheduler is not None,
            "config": self._config.to_dict(),
            "modules": {
                "cvd": self.cvd.stats(),
                "volume_delta": self.volume_delta.stats(),
                "aggressive_trades": self.aggressive_trades.stats(),
                "orderbook_imbalance": self.orderbook_imbalance.stats(),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_module_enabled(self, name: str) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_module_enabled", _analytics_args)
        except Exception:
            pass
        normalized = str(name).strip().lower()

        if normalized == "cvd":
            return self._config.cvd.enabled

        if normalized == "volume_delta":
            return self._config.volume_delta.enabled

        if normalized == "aggressive_trades":
            return self._config.aggressive_trades.enabled

        if normalized == "orderbook_imbalance":
            return self._config.orderbook_imbalance.enabled

        return False

    def _empty_key_result(
        self,
        key: OrderFlowKey,
    ) -> dict[str, BaseOrderFlowStats | None | dict[str, str] | list[str]]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_empty_key_result", _analytics_args)
        except Exception:
            pass
        return {
            **self._key_result_base(key),
            "cvd": None,
            "volume_delta": None,
            "aggressive_trades": None,
            "orderbook_imbalance": None,
        }

    def _key_result_base(
        self,
        key: OrderFlowKey,
    ) -> dict[str, dict[str, str] | list[str] | str]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_key_result_base", _analytics_args)
        except Exception:
            pass
        return {
            "key": list(key),
            "orderflow_key": list(key),
            "scope": orderflow_key_to_dict(key),
            "scope_key": orderflow_key_to_string(key),
        }

    @staticmethod
    def make_key(
        *,
        exchange: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> OrderFlowKey:
        try:
            _analytics_class_name = "OrderFlowAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "make_key", _analytics_args)
        except Exception:
            pass
        return make_orderflow_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    def _normalize_key_from_tuple(self, key: OrderFlowKey) -> OrderFlowKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_key_from_tuple", _analytics_args)
        except Exception:
            pass
        if len(key) != 4:
            raise ValueError(
                "OrderFlowKey must be a 4-tuple: "
                "(exchange, market_type, symbol, timeframe)"
            )

        exchange, market_type, symbol, timeframe = key
        return self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    @staticmethod
    def _normalize_topics(
        values: list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        try:
            _analytics_class_name = "OrderFlowAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_topics", _analytics_args)
        except Exception:
            pass
        return list(
            dict.fromkeys(
                str(value).strip()
                for value in (values or ())
                if str(value).strip()
            )
        )

    def _defaults_payload(self) -> dict[str, str]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_defaults_payload", _analytics_args)
        except Exception:
            pass
        return {
            "exchange": self._default_exchange,
            "market_type": self._default_market_type,
            "timeframe": self._default_timeframe,
        }

    @staticmethod
    def _key_payload(key: OrderFlowKey) -> dict[str, Any]:
        try:
            _analytics_class_name = "OrderFlowAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_key_payload", _analytics_args)
        except Exception:
            pass
        scope = orderflow_key_to_dict(key)
        scope_key = orderflow_key_to_string(key)

        return {
            **scope,
            "scope": scope,
            "scope_key": scope_key,
            "orderflow_key": key,
            "key": list(key),
        }

    async def _emit_lifecycle_event(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority,
        key: OrderFlowKey | None = None,
        event_type: str | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_lifecycle_event", _analytics_args)
        except Exception:
            pass
        try:
            headers: dict[str, str] = {
                "component": "analytics",
                "component_module": "orderflow",
            }

            if event_type:
                headers["event_type"] = event_type

            if key is not None:
                scope = orderflow_key_to_dict(key)
                headers.update(
                    {
                        "exchange": scope["exchange"],
                        "market_type": scope["market_type"],
                        "symbol": scope["symbol"],
                        "timeframe": scope["timeframe"],
                        "scope_key": orderflow_key_to_string(key),
                    }
                )

                payload.setdefault("scope", scope)
                payload.setdefault("scope_key", orderflow_key_to_string(key))
                payload.setdefault("orderflow_key", key)
                payload.setdefault("key", list(key))

            payload.setdefault("source", "orderflow_analyzer")
            payload.setdefault("scope_model", "exchange:market_type:symbol:timeframe")

            await self._event_bus.emit(
                topic,
                payload,
                priority=priority,
                source="orderflow_analyzer",
                headers=headers,
            )
        except Exception:
            self._logger.exception(
                "Failed to emit order-flow lifecycle event",
                extra={
                    "topic": topic,
                    "event_type": event_type,
                    "scope": orderflow_key_to_dict(key) if key is not None else None,
                    "scope_key": orderflow_key_to_string(key) if key is not None else None,
                },
            )