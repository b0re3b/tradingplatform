from __future__ import annotations
import math

import asyncio
import inspect
import time
from abc import ABC, abstractmethod
from typing import Any, Mapping

from core.event_bus import Event, EventBus, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import BaseOrderFlowSubConfig
from .enums import (
    OrderFlowMetricType,
    OrderFlowSide,
    OrderFlowSignalType,
    OrderFlowSourceType,
)
from .models import (
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    BaseOrderFlowStats,
    NormalizedTrade,
    OrderFlowKey,
    OrderFlowSignal,
    OrderFlowUpdate,
    OrderbookSnapshot,
    make_orderflow_key,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    orderflow_key_to_dict,
    orderflow_key_to_string,
    signal_to_dict,
    update_to_dict,
)


class BaseOrderFlowAnalyzer(ABC):
    """
    Base class for all analytics.orderflow analyzers.

    Responsibilities:
    - EventBus subscription lifecycle via register()/stop();
    - Scheduler-based health and cleanup jobs;
    - shared metrics;
    - signal throttling;
    - update/signal EventBus publishing;
    - common extraction and normalization helpers;
    - futures scope handling.

    Correct scope:
        exchange + market_type + symbol + timeframe

    Correct input flow:
        exchange adapters
            -> data caches
            -> market.trades.updated / market.orderbook.updated
            -> analytics.orderflow analyzer
            -> analytics.orderflow.*

    Concrete analyzers should implement:
    - process_key()
    - get_latest_stats_by_key()
    - _handle_event()

    Backward-compatible methods process_symbol() / get_latest_stats() are kept
    as wrappers for gradual migration, but new code should use scoped methods.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: BaseOrderFlowSubConfig,
        metric_type: OrderFlowMetricType,
        source_type: OrderFlowSourceType,
        scheduler: Scheduler | None = None,
        source_topic_patterns: list[str] | tuple[str, ...] | None = None,
        component_module: str = "orderflow",
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
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._config = config
        self._metric_type = metric_type
        self._source_type = source_type
        self._source_topic_patterns = list(
            dict.fromkeys(
                str(pattern).strip()
                for pattern in (source_topic_patterns or ())
                if str(pattern).strip()
            )
        )

        self._default_exchange = (
            normalize_exchange(default_exchange)
            if default_exchange is not None
            else DEFAULT_EXCHANGE
        )
        self._default_market_type = normalize_market_type(default_market_type)
        self._default_timeframe = normalize_timeframe(default_timeframe)

        self._logger = get_logger(
            __name__,
            service_name=self._config.source_name,
            component="analytics",
            component_module=component_module,
            metric=self._metric_type.value,
            source_type=self._source_type.value,
        )

        self._subscriptions: list[Subscription] = []
        self._running = False
        self._lock = asyncio.Lock()

        self._health_job_id: str | None = None
        self._cleanup_job_id: str | None = None

        self._last_signal_ts_by_key: dict[OrderFlowKey, float] = {}
        self._last_update_ts_by_key: dict[OrderFlowKey, float] = {}
        self._metrics: dict[str, Any] = self._build_initial_metrics()

        self._validate_config()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Register analyzer subscriptions and scheduler jobs.

        This is the standard lifecycle entrypoint used by the project.
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
        if self._running:
            self._logger.warning("%s already registered", self.__class__.__name__)
            return

        if not self._config.enabled:
            self._logger.warning("%s is disabled by config", self.__class__.__name__)
            return

        if not self._source_topic_patterns:
            self._logger.warning(
                "%s has no source topic patterns; no EventBus subscriptions created",
                self.__class__.__name__,
            )

        for pattern in self._source_topic_patterns:
            self._assert_input_topic_allowed(pattern)

            subscription = self._event_bus.subscribe(
                pattern=pattern,
                handler=self._handle_event,
                name=f"{self.__class__.__name__}:{pattern}",
            )
            self._subscriptions.append(subscription)

        self._register_scheduler_jobs()
        self._running = True

        self._logger.info(
            "%s registered",
            extra={
                "analyzer": self.__class__.__name__,
                "metric": self._metric_type.value,
                "source_type": self._source_type.value,
                "patterns": list(self._source_topic_patterns),
                "scope": "exchange:market_type:symbol:timeframe",
                "defaults": self._defaults_payload(),
                "output_topics": list(self._config.output_topics),
                "scheduler_job_names": list(self._config.scheduler_job_names),
            },
        )

    def start(self) -> None:
        """
        Backward-compatible alias.

        New modules should call register().
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
        self.register()

    def stop(self) -> None:
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
        if not self._running:
            self._logger.warning("%s already stopped", self.__class__.__name__)
            return

        for subscription in list(self._subscriptions):
            try:
                self._event_bus.unsubscribe(subscription)
            except Exception:
                self._logger.exception(
                    "Failed to unsubscribe handler",
                    extra={
                        "analyzer": self.__class__.__name__,
                        "pattern": getattr(subscription, "pattern", None),
                    },
                )

        self._subscriptions.clear()
        self._remove_scheduler_jobs()
        self._running = False

        self._logger.info(
            "%s stopped",
            extra={
                "analyzer": self.__class__.__name__,
                "metric": self._metric_type.value,
                "source_type": self._source_type.value,
            },
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
    # Abstract API
    # ------------------------------------------------------------------

    @abstractmethod
    async def process_key(self, key: OrderFlowKey) -> BaseOrderFlowStats | None:
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
        raise NotImplementedError

    @abstractmethod
    def get_latest_stats_by_key(self, key: OrderFlowKey) -> BaseOrderFlowStats | None:
        """
        Return latest stats for one scoped futures market.
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    async def process_symbol(self, symbol: str) -> BaseOrderFlowStats | None:
        """
        Backward-compatible wrapper.

        Symbol-only processing is unsafe in multi-exchange mode, so this uses
        explicit configured defaults.
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

    def get_latest_stats(self, symbol: str) -> BaseOrderFlowStats | None:
        """
        Backward-compatible wrapper.

        New code should call get_latest_stats_by_key().
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
            exchange=self._default_exchange,
            market_type=self._default_market_type,
            symbol=symbol,
            timeframe=self._default_timeframe,
        )
        return self.get_latest_stats_by_key(key)

    # ------------------------------------------------------------------
    # Public stats
    # ------------------------------------------------------------------

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
        config_payload = (
            self._config.to_dict()
            if hasattr(self._config, "to_dict")
            else {
                "enabled": self._config.enabled,
                "emit_updates": self._config.emit_updates,
                "emit_signals": self._config.emit_signals,
                "source_name": self._config.source_name,
                "update_topic": self._config.update_topic,
                "signal_topic": self._config.signal_topic,
            }
        )

        return {
            "running": self._running,
            "metric": self._metric_type.value,
            "source_type": self._source_type.value,
            "source_topic_patterns": list(self._source_topic_patterns),
            "scope": "exchange:market_type:symbol:timeframe",
            "defaults": self._defaults_payload(),
            "config": config_payload,
            "subscriptions": len(self._subscriptions),
            "health_job_id": self._health_job_id,
            "cleanup_job_id": self._cleanup_job_id,
            "metrics": {
                "processed": self._metrics["processed"],
                "signals_emitted": self._metrics["signals_emitted"],
                "updates_emitted": self._metrics["updates_emitted"],
                "skipped": self._metrics["skipped"],
                "errors": self._metrics["errors"],
                "emit_errors": self._metrics["emit_errors"],
                "keys": dict(self._metrics["keys"]),
            },
        }

    def log_health(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "log_health", _analytics_args)
        except Exception:
            pass
        snapshot = self.stats()
        self._logger.info(
            "%s health",
            extra={
                "analyzer": self.__class__.__name__,
                "running": snapshot["running"],
                "subscriptions": snapshot["subscriptions"],
                "metrics": snapshot["metrics"],
                "source_topic_patterns": snapshot["source_topic_patterns"],
                "scope": snapshot["scope"],
            },
        )

    async def cleanup(self) -> None:
        """
        Hook for stale state cleanup.

        Concrete analyzers can override this method. It is scheduled through
        core.scheduler.Scheduler.add_interval_job().
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
        return None

    # ------------------------------------------------------------------
    # Shared emitters
    # ------------------------------------------------------------------

    async def emit_update(self, stats: BaseOrderFlowStats) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "emit_update", _analytics_args)
        except Exception:
            pass
        if not self._config.emit_updates:
            return

        if not self.should_process_key(stats.key):
            self._inc_metric("skipped", stats.key)
            return

        if not self._can_emit_update(stats.key):
            self._inc_metric("updates_throttled", stats.key)
            return

        update = OrderFlowUpdate.from_stats(stats)
        payload = update_to_dict(update)

        emitted = await self._safe_emit(
            topic=self._config.update_topic,
            payload=payload,
            source=self._config.source_name,
            key=stats.key,
            event_type="orderflow_update",
        )

        if emitted:
            self._last_update_ts_by_key[stats.key] = time.time()
            self._inc_metric("updates_emitted", stats.key)

    async def emit_signal(self, signal: OrderFlowSignal) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "emit_signal", _analytics_args)
        except Exception:
            pass
        if not self._config.emit_signals:
            return

        if not self.should_process_key(signal.key):
            self._inc_metric("skipped", signal.key)
            return

        if not self._can_emit_signal(signal.key):
            self._inc_metric("skipped", signal.key)
            return

        payload = signal_to_dict(signal)

        emitted = await self._safe_emit(
            topic=self._config.signal_topic,
            payload=payload,
            source=self._config.source_name,
            key=signal.key,
            event_type="orderflow_signal",
        )

        if emitted:
            self._last_signal_ts_by_key[signal.key] = time.time()
            self._inc_metric("signals_emitted", signal.key)

    def build_signal(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
        signal_type: OrderFlowSignalType,
        side: OrderFlowSide,
        strength: float,
        reason: str,
        exchange_symbol: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> OrderFlowSignal:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_signal", _analytics_args)
        except Exception:
            pass
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        scope = orderflow_key_to_dict(key)

        merged_context = {
            "scope": scope,
            "scope_key": orderflow_key_to_string(key),
            "metric": self._metric_type.value,
            "source_type": self._source_type.value,
            **dict(context or {}),
        }

        return OrderFlowSignal(
            exchange=scope["exchange"],
            market_type=scope["market_type"],
            symbol=scope["symbol"],
            exchange_symbol=exchange_symbol,
            timeframe=scope["timeframe"],
            metric=self._metric_type,
            source_type=self._source_type,
            signal_type=signal_type,
            side=side,
            strength=max(0.0, min(1.0, float(strength))),
            reason=reason,
            context=merged_context,
        )

    def build_signal_from_stats(
        self,
        *,
        stats: BaseOrderFlowStats,
        signal_type: OrderFlowSignalType,
        side: OrderFlowSide,
        strength: float,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> OrderFlowSignal:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_signal_from_stats", _analytics_args)
        except Exception:
            pass
        merged_context = {
            "stats": stats.to_dict(),
            "scope": stats.scope,
            "scope_key": stats.scope_key,
            **dict(context or {}),
        }

        return self.build_signal(
            exchange=stats.exchange,
            market_type=stats.market_type,
            symbol=stats.symbol,
            exchange_symbol=stats.exchange_symbol,
            timeframe=stats.timeframe,
            signal_type=signal_type,
            side=side,
            strength=strength,
            reason=reason,
            context=merged_context,
        )

    # ------------------------------------------------------------------
    # Shared event helpers
    # ------------------------------------------------------------------

    def should_process_key(self, key: OrderFlowKey | None) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "should_process_key", _analytics_args)
        except Exception:
            pass
        if key is None:
            return False

        exchange, market_type, symbol, timeframe = key
        if not exchange or not market_type or not symbol or not timeframe:
            return False

        return self._config.should_process_key(key)

    def should_process_symbol(self, symbol: str | None) -> bool:
        """
        Backward-compatible symbol allowlist check.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "should_process_symbol", _analytics_args)
        except Exception:
            pass
        if not symbol:
            return False

        return self._config.should_process_symbol(symbol)

    def extract_key_from_event(self, event: Event) -> OrderFlowKey | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "extract_key_from_event", _analytics_args)
        except Exception:
            pass
        payload = getattr(event, "payload", None)
        key = self._extract_key_from_payload(payload)
        if key is not None:
            return key

        headers = getattr(event, "headers", None)
        if isinstance(headers, Mapping):
            key = self._extract_key_from_mapping(headers)
            if key is not None:
                return key

        symbol = getattr(event, "symbol", None)
        exchange = getattr(event, "exchange", None) or self._default_exchange
        market_type = getattr(event, "market_type", None) or self._default_market_type
        timeframe = getattr(event, "timeframe", None) or self._default_timeframe

        if symbol and exchange:
            return self.make_key(
                exchange=str(exchange),
                market_type=str(market_type),
                symbol=str(symbol),
                timeframe=str(timeframe),
            )

        return None

    def extract_symbol_from_event(self, event: Event) -> str | None:
        """
        Backward-compatible helper.

        Prefer extract_key_from_event().
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "extract_symbol_from_event", _analytics_args)
        except Exception:
            pass
        key = self.extract_key_from_event(event)
        return key[2] if key is not None else None

    def extract_exchange_from_event(self, event: Event) -> str | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "extract_exchange_from_event", _analytics_args)
        except Exception:
            pass
        key = self.extract_key_from_event(event)
        return key[0] if key is not None else None

    def extract_market_type_from_event(self, event: Event) -> str | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "extract_market_type_from_event", _analytics_args)
        except Exception:
            pass
        key = self.extract_key_from_event(event)
        return key[1] if key is not None else None

    def extract_timeframe_from_event(self, event: Event) -> str | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "extract_timeframe_from_event", _analytics_args)
        except Exception:
            pass
        key = self.extract_key_from_event(event)
        return key[3] if key is not None else None

    def extract_payload_data(self, event: Event) -> Any:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "extract_payload_data", _analytics_args)
        except Exception:
            pass
        payload = getattr(event, "payload", None)

        if isinstance(payload, Mapping) and "data" in payload:
            return payload["data"]

        return payload

    def normalize_trade(
        self,
        raw_trade: Any,
        *,
        default_symbol: str | None = None,
        default_exchange: str | None = None,
        default_market_type: str | None = None,
        default_timeframe: str | None = None,
        default_exchange_symbol: str | None = None,
    ) -> NormalizedTrade | None:
        """
        Normalize one trade payload into a finite, scoped NormalizedTrade.

        This method is intentionally strict because every trade-flow analyzer
        depends on it. Invalid numeric values such as NaN/inf must be rejected
        here before CVD, volume delta or aggressive-trades calculations can
        produce corrupted stats.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalize_trade", _analytics_args)
        except Exception:
            pass
        if raw_trade is None:
            return None

        if isinstance(raw_trade, NormalizedTrade):
            if not self._is_valid_normalized_trade(raw_trade):
                return None
            return raw_trade if self.should_process_key(raw_trade.key) else None

        if not isinstance(raw_trade, Mapping):
            return None

        symbol = (
            raw_trade.get("symbol")
            or raw_trade.get("s")
            or raw_trade.get("instrument")
            or default_symbol
        )
        exchange = (
            raw_trade.get("exchange")
            or raw_trade.get("venue")
            or raw_trade.get("source_exchange")
            or default_exchange
            or self._default_exchange
        )
        market_type = (
            raw_trade.get("market_type")
            or raw_trade.get("category")
            or raw_trade.get("inst_type")
            or raw_trade.get("instrument_type")
            or default_market_type
            or self._default_market_type
        )
        timeframe = (
            raw_trade.get("timeframe")
            or raw_trade.get("tf")
            or raw_trade.get("interval")
            or default_timeframe
            or self._default_timeframe
        )
        exchange_symbol = (
            raw_trade.get("exchange_symbol")
            or raw_trade.get("raw_symbol")
            or raw_trade.get("exchangeSymbol")
            or default_exchange_symbol
        )

        if not symbol or not exchange:
            return None

        raw_price = self._first_present(raw_trade, "price", "p")
        raw_quantity = self._first_present(
            raw_trade,
            "quantity",
            "qty",
            "q",
            "size",
            "amount",
            "volume",
        )
        raw_timestamp = self._first_present(
            raw_trade,
            "timestamp",
            "timestamp_ms",
            "ts",
            "T",
            "time",
        )

        if raw_timestamp is None:
            raw_timestamp = time.time()

        price = self._parse_finite_float(raw_price)
        quantity = self._parse_finite_float(raw_quantity)
        timestamp_raw = self._parse_finite_float(raw_timestamp)

        if price is None or quantity is None or timestamp_raw is None:
            return None

        if price <= 0.0 or quantity <= 0.0 or timestamp_raw <= 0.0:
            return None

        timestamp = self._normalize_timestamp(timestamp_raw)
        if not math.isfinite(timestamp) or timestamp <= 0.0:
            return None

        side = self._extract_trade_side(dict(raw_trade))
        if not side.is_known:
            return None

        raw_notional = self._first_present(
            raw_trade,
            "notional",
            "quote_qty",
            "quote_quantity",
            "quote_volume",
            "quoteVolume",
        )

        notional = self._parse_finite_float(raw_notional)
        if notional is None:
            notional = price * quantity

        if not math.isfinite(notional) or notional <= 0.0:
            return None

        trade_id = raw_trade.get("trade_id", raw_trade.get("id"))
        is_aggressive = self._parse_bool(
            raw_trade.get(
                "is_aggressive",
                raw_trade.get("aggressive", False),
            ),
            default=False,
        )

        try:
            trade = NormalizedTrade.create(
                exchange=str(exchange),
                market_type=str(market_type),
                symbol=str(symbol),
                exchange_symbol=(
                    str(exchange_symbol) if exchange_symbol is not None else None
                ),
                timeframe=str(timeframe),
                side=side,
                price=price,
                quantity=quantity,
                notional=notional,
                timestamp=timestamp,
                trade_id=str(trade_id) if trade_id is not None else None,
                is_aggressive=is_aggressive,
                raw=dict(raw_trade),
            )
        except (TypeError, ValueError, OverflowError):
            return None

        if not self._is_valid_normalized_trade(trade):
            return None

        return trade if self.should_process_key(trade.key) else None

    @staticmethod
    def _first_present(data: Mapping[str, Any], *keys: str) -> Any:
        """
        Return the first key that exists and is not None.

        Unlike `or` chains, this does not accidentally skip valid falsy values
        such as 0. Those values are parsed and rejected explicitly later.
        """
        try:
            _analytics_class_name = "BaseOrderFlowAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_first_present", _analytics_args)
        except Exception:
            pass
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return None

    @staticmethod
    def _parse_finite_float(value: Any) -> float | None:
        try:
            _analytics_class_name = "BaseOrderFlowAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_parse_finite_float", _analytics_args)
        except Exception:
            pass
        if value is None:
            return None

        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return None

        if not math.isfinite(parsed):
            return None

        return parsed

    @staticmethod
    def _parse_bool(value: Any, *, default: bool = False) -> bool:
        try:
            _analytics_class_name = "BaseOrderFlowAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_parse_bool", _analytics_args)
        except Exception:
            pass
        if value is None:
            return default

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                return default
            return bool(value)

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False

        return default

    def _is_valid_normalized_trade(self, trade: NormalizedTrade) -> bool:
        """
        Strong validation for NormalizedTrade models.

        NormalizedTrade.is_valid may verify positive values, but this method
        additionally guarantees that no NaN/inf can reach downstream analyzers.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_valid_normalized_trade", _analytics_args)
        except Exception:
            pass
        if not trade.is_valid:
            return False

        numeric_values = (
            trade.price,
            trade.quantity,
            trade.notional,
            trade.timestamp,
        )

        for value in numeric_values:
            if not isinstance(value, (int, float)):
                return False
            if not math.isfinite(float(value)):
                return False

        if trade.price <= 0.0:
            return False

        if trade.quantity <= 0.0:
            return False

        if trade.notional <= 0.0:
            return False

        if trade.timestamp <= 0.0:
            return False

        if not trade.side.is_known:
            return False

        return True

    def normalize_orderbook_snapshot(
        self,
        raw_snapshot: Any,
        *,
        default_symbol: str | None = None,
        default_exchange: str | None = None,
        default_market_type: str | None = None,
        default_timeframe: str | None = None,
        default_exchange_symbol: str | None = None,
    ) -> OrderbookSnapshot | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalize_orderbook_snapshot", _analytics_args)
        except Exception:
            pass
        if raw_snapshot is None:
            return None

        if isinstance(raw_snapshot, OrderbookSnapshot):
            if not raw_snapshot.is_valid:
                return None
            return raw_snapshot if self.should_process_key(raw_snapshot.key) else None

        if not isinstance(raw_snapshot, Mapping):
            return None

        symbol = (
            raw_snapshot.get("symbol")
            or raw_snapshot.get("s")
            or raw_snapshot.get("instrument")
            or default_symbol
        )
        exchange = (
            raw_snapshot.get("exchange")
            or raw_snapshot.get("venue")
            or raw_snapshot.get("source_exchange")
            or default_exchange
            or self._default_exchange
        )
        market_type = (
            raw_snapshot.get("market_type")
            or raw_snapshot.get("category")
            or raw_snapshot.get("inst_type")
            or raw_snapshot.get("instrument_type")
            or default_market_type
            or self._default_market_type
        )
        timeframe = (
            raw_snapshot.get("timeframe")
            or raw_snapshot.get("tf")
            or raw_snapshot.get("interval")
            or default_timeframe
            or self._default_timeframe
        )
        exchange_symbol = (
            raw_snapshot.get("exchange_symbol")
            or raw_snapshot.get("raw_symbol")
            or raw_snapshot.get("exchangeSymbol")
            or default_exchange_symbol
        )

        if not symbol or not exchange:
            return None

        bids_raw = raw_snapshot.get("bids", raw_snapshot.get("b", []))
        asks_raw = raw_snapshot.get("asks", raw_snapshot.get("a", []))

        if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
            return None

        raw_timestamp = raw_snapshot.get("timestamp")
        if raw_timestamp is None:
            raw_timestamp = raw_snapshot.get("timestamp_ms")
        if raw_timestamp is None:
            raw_timestamp = raw_snapshot.get("last_update_ts_ms")
        if raw_timestamp is None:
            raw_timestamp = raw_snapshot.get("ts")
        if raw_timestamp is None:
            raw_timestamp = raw_snapshot.get("T")
        if raw_timestamp is None:
            raw_timestamp = time.time()

        sequence_id = (
            raw_snapshot.get("sequence_id")
            or raw_snapshot.get("sequence")
            or raw_snapshot.get("u")
        )

        try:
            snapshot = OrderbookSnapshot.create(
                exchange=str(exchange),
                market_type=str(market_type),
                symbol=str(symbol),
                exchange_symbol=(
                    str(exchange_symbol) if exchange_symbol is not None else None
                ),
                timeframe=str(timeframe),
                bids=bids_raw,
                asks=asks_raw,
                timestamp=self._normalize_timestamp(float(raw_timestamp)),
                sequence_id=str(sequence_id) if sequence_id is not None else None,
                raw=dict(raw_snapshot),
            )
        except (TypeError, ValueError):
            return None

        if not snapshot.is_valid:
            return None

        return snapshot if self.should_process_key(snapshot.key) else None

    def make_trade_key(self, trade: NormalizedTrade) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "make_trade_key", _analytics_args)
        except Exception:
            pass
        if trade.trade_id:
            return (
                f"{trade.exchange}:{trade.market_type}:{trade.symbol}:"
                f"{trade.timeframe}:{trade.trade_id}"
            )

        return (
            f"{trade.exchange}:{trade.market_type}:{trade.symbol}:{trade.timeframe}:"
            f"{trade.timestamp:.9f}:{trade.price:.12f}:"
            f"{trade.quantity:.12f}:{trade.side.value}"
        )

    @staticmethod
    def make_key(
        *,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> OrderFlowKey:
        try:
            _analytics_class_name = "BaseOrderFlowAnalyzer"
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

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_config", _analytics_args)
        except Exception:
            pass
        try:
            self._config.validate()
        except Exception:
            self._logger.exception(
                "Invalid analyzer config",
                extra={"analyzer": self.__class__.__name__},
            )
            raise

    def _assert_input_topic_allowed(self, topic: str) -> None:
        """
        Best-effort topic guard.

        Top-level OrderFlowConfig owns canonical raw-topic guards. Sub-config
        can also expose assert_input_topic_allowed() in newer versions.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_assert_input_topic_allowed", _analytics_args)
        except Exception:
            pass
        guard = getattr(self._config, "assert_input_topic_allowed", None)
        if callable(guard):
            guard(topic)
            return

        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("Orderflow input topic must not be empty")

        if " " in topic:
            raise ValueError("Orderflow input topic must not contain spaces")

        if topic in {"market.trade", "market.orderbook"}:
            raise ValueError(
                f"Raw market topic {topic!r} is not allowed for orderflow analyzer. "
                "Use data/cache-layer topics such as market.trades.updated "
                "or market.orderbook.updated."
            )

    def _extract_key_from_payload(self, payload: Any) -> OrderFlowKey | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_key_from_payload", _analytics_args)
        except Exception:
            pass
        if not isinstance(payload, Mapping):
            return None

        key = self._extract_key_from_mapping(payload)
        if key is not None:
            return key

        data = payload.get("data")
        if isinstance(data, Mapping):
            return self._extract_key_from_mapping(data)

        return None

    def _extract_key_from_mapping(self, data: Mapping[str, Any]) -> OrderFlowKey | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_key_from_mapping", _analytics_args)
        except Exception:
            pass
        scope = data.get("scope")
        if isinstance(scope, Mapping):
            scoped_key = self._extract_key_from_mapping(scope)
            if scoped_key is not None:
                return scoped_key

        raw_key = data.get("orderflow_key") or data.get("key")
        if isinstance(raw_key, (list, tuple)) and len(raw_key) == 4:
            try:
                return self.make_key(
                    exchange=str(raw_key[0]),
                    market_type=str(raw_key[1]),
                    symbol=str(raw_key[2]),
                    timeframe=str(raw_key[3]),
                )
            except ValueError:
                return None

        exchange = (
            data.get("exchange")
            or data.get("venue")
            or data.get("source_exchange")
            or self._default_exchange
        )
        market_type = (
            data.get("market_type")
            or data.get("category")
            or data.get("inst_type")
            or data.get("instrument_type")
            or self._default_market_type
        )
        symbol = data.get("symbol") or data.get("s") or data.get("instrument")
        timeframe = (
            data.get("timeframe")
            or data.get("tf")
            or data.get("interval")
            or self._default_timeframe
        )

        if not exchange or not symbol:
            return None

        try:
            return self.make_key(
                exchange=str(exchange),
                market_type=str(market_type),
                symbol=str(symbol),
                timeframe=str(timeframe),
            )
        except ValueError:
            return None

    def _extract_symbol_from_payload(self, payload: Any) -> str | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_symbol_from_payload", _analytics_args)
        except Exception:
            pass
        key = self._extract_key_from_payload(payload)
        return key[2] if key is not None else None

    def _extract_trade_side(self, raw_trade: dict[str, Any]) -> OrderFlowSide:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_trade_side", _analytics_args)
        except Exception:
            pass
        side = raw_trade.get("side")
        if side is not None:
            side_enum = OrderFlowSide.from_value(side)
            if side_enum.is_known:
                return side_enum

        side = raw_trade.get("aggressor_side")
        if side is not None:
            side_enum = OrderFlowSide.from_value(side)
            if side_enum.is_known:
                return side_enum

        side = raw_trade.get("taker_side")
        if side is not None:
            side_enum = OrderFlowSide.from_value(side)
            if side_enum.is_known:
                return side_enum

        # Binance-style maker flag:
        # m=False => buyer aggressive => buy
        # m=True  => seller aggressive => sell
        maker_flag = raw_trade.get("m")
        if maker_flag is not None:
            return OrderFlowSide.SELL if bool(maker_flag) else OrderFlowSide.BUY

        is_buyer_maker = raw_trade.get("is_buyer_maker")
        if is_buyer_maker is not None:
            return OrderFlowSide.SELL if bool(is_buyer_maker) else OrderFlowSide.BUY

        return OrderFlowSide.UNKNOWN

    def _can_emit_update(self, key: OrderFlowKey) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_can_emit_update", _analytics_args)
        except Exception:
            pass
        interval = float(getattr(self._config, "min_update_interval_sec", 0.0) or 0.0)
        if interval <= 0:
            return True

        now = time.time()
        last_ts = self._last_update_ts_by_key.get(key, 0.0)
        return (now - last_ts) >= interval

    def _can_emit_signal(self, key: OrderFlowKey) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_can_emit_signal", _analytics_args)
        except Exception:
            pass
        now = time.time()
        last_ts = self._last_signal_ts_by_key.get(key, 0.0)
        return (now - last_ts) >= float(self._config.min_signal_interval_sec)

    async def _safe_emit(
        self,
        *,
        topic: str,
        payload: dict[str, Any],
        source: str,
        key: OrderFlowKey | None = None,
        event_type: str | None = None,
    ) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_safe_emit", _analytics_args)
        except Exception:
            pass
        if not topic:
            self._logger.warning(
                "Emit skipped because topic is empty",
                extra={"analyzer": self.__class__.__name__},
            )
            self._inc_metric("emit_errors", key)
            return False

        headers: dict[str, str] = {}

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

        if event_type:
            headers["event_type"] = event_type

        headers.setdefault("metric", self._metric_type.value)
        headers.setdefault("source_type", self._source_type.value)

        try:
            return await self._event_bus.emit(
                topic,
                payload,
                priority=self._config.publish_priority,
                source=source,
                headers=headers,
            )
        except Exception:
            self._inc_metric("emit_errors", key)
            self._logger.exception(
                "Failed to emit EventBus event",
                extra={
                    "analyzer": self.__class__.__name__,
                    "topic": topic,
                    "scope": orderflow_key_to_dict(key) if key is not None else None,
                    "scope_key": orderflow_key_to_string(key) if key is not None else None,
                },
            )
            return False

    def _inc_metric(
        self,
        name: str,
        key: OrderFlowKey | None = None,
        amount: int = 1,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_inc_metric", _analytics_args)
        except Exception:
            pass
        if name not in self._metrics:
            self._metrics[name] = 0

        self._metrics[name] += amount

        if key is None:
            return

        key_payload = orderflow_key_to_dict(key)
        key_label = orderflow_key_to_string(key)

        key_metrics = self._metrics["keys"].setdefault(
            key_label,
            {
                **key_payload,
                "scope": key_payload,
                "scope_key": key_label,
                "orderflow_key": key,
                "key": list(key),
                "processed": 0,
                "signals_emitted": 0,
                "updates_emitted": 0,
                "skipped": 0,
                "errors": 0,
                "emit_errors": 0,
            },
        )

        if name in key_metrics:
            key_metrics[name] += amount

    def _build_initial_metrics(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_initial_metrics", _analytics_args)
        except Exception:
            pass
        return {
            "processed": 0,
            "signals_emitted": 0,
            "updates_emitted": 0,
            "skipped": 0,
            "errors": 0,
            "emit_errors": 0,
            "keys": {},
        }

    @staticmethod
    def _normalize_timestamp(value: float) -> float:
        """
        Normalize timestamp to seconds.

        Data caches may use timestamp_ms. Analytics models use seconds.
        """
        try:
            _analytics_class_name = "BaseOrderFlowAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_timestamp", _analytics_args)
        except Exception:
            pass
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            return timestamp / 1000.0
        return timestamp

    @staticmethod
    def _normalize_market_type(value: Any) -> str:
        try:
            _analytics_class_name = "BaseOrderFlowAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_market_type", _analytics_args)
        except Exception:
            pass
        return normalize_market_type(value)

    @staticmethod
    def _normalize_timeframe(value: Any) -> str:
        try:
            _analytics_class_name = "BaseOrderFlowAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_timeframe", _analytics_args)
        except Exception:
            pass
        return normalize_timeframe(value)

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

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_register_scheduler_jobs", _analytics_args)
        except Exception:
            pass
        if self._scheduler is None:
            return

        job_names = list(getattr(self._config, "scheduler_job_names", ()) or ())
        health_name = (
            job_names[0]
            if len(job_names) >= 1
            else f"analytics.orderflow.{self._config.source_name}.health"
        )
        cleanup_name = (
            job_names[1]
            if len(job_names) >= 2
            else f"analytics.orderflow.{self._config.source_name}.cleanup"
        )

        self._health_job_id = self._add_interval_job_once(
            name=health_name,
            func=self._safe_health_job,
            interval=float(self._config.health_log_interval_sec),
        )

        self._cleanup_job_id = self._add_interval_job_once(
            name=cleanup_name,
            func=self._safe_cleanup_job,
            interval=float(self._config.cleanup_interval_sec),
        )

        self._logger.info(
            "Scheduler jobs registered",
            extra={
                "analyzer": self.__class__.__name__,
                "health_job_id": self._health_job_id,
                "cleanup_job_id": self._cleanup_job_id,
                "health_job_name": health_name,
                "cleanup_job_name": cleanup_name,
            },
        )

    def _add_interval_job_once(
        self,
        *,
        name: str,
        func: Any,
        interval: float,
    ) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_add_interval_job_once", _analytics_args)
        except Exception:
            pass
        assert self._scheduler is not None

        existing = self._scheduler.get_job_by_name(name)
        if existing is not None:
            return existing.job_id

        return self._scheduler.add_interval_job(
            name=name,
            func=func,
            interval=interval,
            max_retries=int(self._config.scheduler_job_max_retries),
            retry_delay=float(self._config.scheduler_job_retry_delay_sec),
            timeout=float(self._config.scheduler_job_timeout_sec),
            allow_overlap=False,
            enabled=True,
        )

    def _remove_scheduler_jobs(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_remove_scheduler_jobs", _analytics_args)
        except Exception:
            pass
        if self._scheduler is None:
            self._health_job_id = None
            self._cleanup_job_id = None
            return

        for job_id in (self._health_job_id, self._cleanup_job_id):
            if job_id is None:
                continue

            try:
                remove_job = getattr(self._scheduler, "remove_job", None)
                if callable(remove_job):
                    result = remove_job(job_id)
                    if inspect.isawaitable(result):
                        self._logger.warning(
                            "Scheduler.remove_job returned awaitable; "
                            "job may require explicit async cleanup by caller",
                            extra={
                                "analyzer": self.__class__.__name__,
                                "job_id": job_id,
                            },
                        )
                else:
                    self._scheduler.disable_job(job_id)
            except KeyError:
                self._logger.debug(
                    "Scheduler job already removed",
                    extra={
                        "analyzer": self.__class__.__name__,
                        "job_id": job_id,
                    },
                )
            except Exception:
                self._logger.exception(
                    "Failed to cleanup scheduler job",
                    extra={
                        "analyzer": self.__class__.__name__,
                        "job_id": job_id,
                    },
                )

        self._health_job_id = None
        self._cleanup_job_id = None

    # Backward-compatible alias.
    def _disable_scheduler_jobs(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_disable_scheduler_jobs", _analytics_args)
        except Exception:
            pass
        self._remove_scheduler_jobs()

    async def _safe_cleanup_job(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_safe_cleanup_job", _analytics_args)
        except Exception:
            pass
        try:
            result = self.cleanup()
            if inspect.isawaitable(result):
                await result
        except Exception:
            self._inc_metric("errors")
            self._logger.exception(
                "Cleanup job failed",
                extra={"analyzer": self.__class__.__name__},
            )

    async def _safe_health_job(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_safe_health_job", _analytics_args)
        except Exception:
            pass
        try:
            result = self.log_health()
            if inspect.isawaitable(result):
                await result
        except Exception:
            self._inc_metric("errors")
            self._logger.exception(
                "Health job failed",
                extra={"analyzer": self.__class__.__name__},
            )