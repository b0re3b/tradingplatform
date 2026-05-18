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
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    OrderFlowKey,
    make_orderflow_key,
    orderflow_key_to_dict,
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
        default_market_type: str = DEFAULT_MARKET_TYPE,
        default_timeframe: str = DEFAULT_TIMEFRAME,
    ) -> None:
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._trades_cache = trades_cache
        self._orderbook_cache = orderbook_cache
        self._config = config or OrderFlowConfig()

        self._config.validate()

        self._default_exchange = (
            str(default_exchange).strip().lower()
            if default_exchange
            else None
        )
        self._default_market_type = self._normalize_market_type(default_market_type)
        self._default_timeframe = self._normalize_timeframe(default_timeframe)

        self._trades_topic_patterns = list(
            trades_topic_patterns
            if trades_topic_patterns is not None
            else self._config.source_topic_patterns_trades
        )
        self._orderbook_topic_patterns = list(
            orderbook_topic_patterns
            if orderbook_topic_patterns is not None
            else self._config.source_topic_patterns_orderbook
        )

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
        async with self._lifecycle_lock:
            if self._running:
                self._logger.warning("OrderFlowAnalyzer already registered")
                return

            if not self._config.enabled:
                self._logger.warning("OrderFlowAnalyzer is disabled by config")
                await self._emit_lifecycle_event(
                    OrderFlowEventTopic.STOPPED.value,
                    {
                        "reason": "disabled_by_config",
                        "enabled": False,
                        "scope": "exchange:market_type:symbol:timeframe",
                    },
                    priority=EventPriority.LOW,
                )
                return

            registered_modules: list[str] = []

            for name, module in self._modules.items():
                if not self._is_module_enabled(name):
                    self._logger.info(
                        "OrderFlow module skipped because it is disabled | module=%s",
                        name,
                    )
                    continue

                try:
                    module.register()
                    registered_modules.append(name)
                except Exception:
                    self._logger.exception(
                        "Failed to register order-flow module | module=%s",
                        name,
                    )
                    raise

            self._running = True

            self._logger.info(
                (
                    "OrderFlowAnalyzer registered | modules=%s "
                    "trades_patterns=%s orderbook_patterns=%s scope=%s"
                ),
                registered_modules,
                self._trades_topic_patterns,
                self._orderbook_topic_patterns,
                "exchange:market_type:symbol:timeframe",
            )

            await self._emit_lifecycle_event(
                OrderFlowEventTopic.STARTED.value,
                {
                    "enabled": True,
                    "modules": registered_modules,
                    "trades_topic_patterns": list(self._trades_topic_patterns),
                    "orderbook_topic_patterns": list(self._orderbook_topic_patterns),
                    "scope": "exchange:market_type:symbol:timeframe",
                    "defaults": {
                        "exchange": self._default_exchange,
                        "market_type": self._default_market_type,
                        "timeframe": self._default_timeframe,
                    },
                },
                priority=EventPriority.NORMAL,
            )

    async def start(self) -> None:
        """
        Backward-compatible alias.

        New code should call register().
        """
        await self.register()

    async def stop(self) -> None:
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
                        "Failed to stop order-flow module | module=%s",
                        name,
                    )

            self._running = False

            self._logger.info(
                "OrderFlowAnalyzer stopped | modules=%s",
                stopped_modules,
            )

            await self._emit_lifecycle_event(
                OrderFlowEventTopic.STOPPED.value,
                {
                    "enabled": self._config.enabled,
                    "modules": stopped_modules,
                    "scope": "exchange:market_type:symbol:timeframe",
                },
                priority=EventPriority.NORMAL,
            )

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_module(self, name: str) -> OrderFlowModule | None:
        return self._modules.get(str(name).strip().lower())

    def list_modules(self) -> tuple[str, ...]:
        return tuple(self._modules.keys())

    def enabled_modules(self) -> tuple[str, ...]:
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
        normalized_key = self._normalize_key_from_tuple(key)

        return {
            "key": list(normalized_key),
            "scope": orderflow_key_to_dict(normalized_key),
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
        normalized_key = self._normalize_key_from_tuple(key)

        if not self._config.enabled:
            self._logger.warning(
                "Manual order-flow processing skipped: facade disabled",
                extra=orderflow_key_to_dict(normalized_key),
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
            except Exception:
                self._logger.exception(
                    "Manual order-flow module processing failed",
                    extra={
                        **orderflow_key_to_dict(normalized_key),
                        "module": name,
                    },
                )
                await self._emit_lifecycle_event(
                    OrderFlowEventTopic.ERROR.value,
                    {
                        **orderflow_key_to_dict(normalized_key),
                        "key": list(normalized_key),
                        "module": name,
                        "reason": "manual_process_failed",
                    },
                    priority=EventPriority.HIGH,
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

        Symbol-only processing is unsafe in multi-exchange futures mode.
        This method is allowed only when default_exchange was explicitly
        configured for this facade instance.
        """
        if not self._default_exchange:
            raise ValueError(
                "process_symbol(symbol) requires default_exchange. "
                "Use process_market(exchange=..., market_type=..., symbol=..., timeframe=...) "
                "or process_key(key) instead."
            )

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
        for name, module in self._modules.items():
            try:
                await module.cleanup()
            except Exception:
                self._logger.exception(
                    "Order-flow cleanup failed | module=%s",
                    name,
                )
                await self._emit_lifecycle_event(
                    OrderFlowEventTopic.ERROR.value,
                    {
                        "module": name,
                        "reason": "cleanup_failed",
                    },
                    priority=EventPriority.HIGH,
                )

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "enabled": self._config.enabled,
            "enabled_modules": self.enabled_modules(),
            "scope": "exchange:market_type:symbol:timeframe",
            "defaults": {
                "exchange": self._default_exchange,
                "market_type": self._default_market_type,
                "timeframe": self._default_timeframe,
            },
            "trades_topic_patterns": list(self._trades_topic_patterns),
            "orderbook_topic_patterns": list(self._orderbook_topic_patterns),
            "scheduler_attached": self._scheduler is not None,
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
        return {
            "key": list(key),
            "scope": orderflow_key_to_dict(key),
            "cvd": None,
            "volume_delta": None,
            "aggressive_trades": None,
            "orderbook_imbalance": None,
        }

    @staticmethod
    def make_key(
        *,
        exchange: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> OrderFlowKey:
        return make_orderflow_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    def _normalize_key_from_tuple(self, key: OrderFlowKey) -> OrderFlowKey:
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
    def _normalize_market_type(value: Any) -> str:
        normalized = str(value or DEFAULT_MARKET_TYPE).strip().lower()
        return normalized if normalized else DEFAULT_MARKET_TYPE

    @staticmethod
    def _normalize_timeframe(value: Any) -> str:
        normalized = str(value or DEFAULT_TIMEFRAME).strip()
        return normalized if normalized else DEFAULT_TIMEFRAME

    async def _emit_lifecycle_event(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority,
    ) -> None:
        try:
            await self._event_bus.emit(
                topic,
                payload,
                priority=priority,
                source="orderflow_analyzer",
            )
        except Exception:
            self._logger.exception(
                "Failed to emit order-flow lifecycle event | topic=%s",
                topic,
            )