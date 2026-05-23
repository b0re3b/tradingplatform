from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Final, Mapping

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import BaseSpreadConfig
from .models import (
    DEFAULT_TIMEFRAME,
    FundingSnapshot,
    QuoteSnapshot,
    SpreadKey,
    SpreadSignal,
    SpreadSnapshot,
    make_spread_key,
    model_to_payload,
    quote_snapshot_from_orderbook_payload,
    spread_key_to_dict,
)
from .spread_regime_detector import SpreadRegimeDetector
from .spread_signal_engine import SpreadSignalEngine


class BaseSpreadAnalyzer(ABC):
    """
    Production-grade базовий клас для analytics.spreads analyzer-компонентів.

    Відповідальність:
    - constructor dependency injection через core.EventBus / core.Scheduler / BaseSpreadConfig;
    - register()/unregister() для EventBus subscriptions;
    - lifecycle start()/stop();
    - EventBus.emit() helpers;
    - Scheduler job registration helpers;
    - cooldown/throttling для сигналів і snapshot emit;
    - shared scoped multi-market helpers;
    - dict/dataclass payload normalization для data-layer events;
    - спільні stats;
    - production-grade logging через core.logger.get_logger;
    - shared SpreadRegimeDetector / SpreadSignalEngine.

    Correct production input flow:
        exchange adapters
            -> market.orderbook / market.funding
            -> OrderBookCache / FundingCache
            -> market.orderbook.updated / market.funding.updated
            -> analytics.spreads.*

    Важливо:
    - analyzer-и не читають біржові WS/REST adapters напряму;
    - production price input іде з OrderBookCache через market.orderbook.updated;
    - QuoteSnapshot залишається внутрішньою normalized top-of-book моделлю;
    - raw market.orderbook / market.quote / market.funding дозволені тільки
      якщо config.allow_legacy_raw_topics=True;
    - market.quote.updated можна лишити тільки як legacy-сумісність;
    - цей base class не містить spread business logic.
    """

    DEFAULT_SNAPSHOT_PRIORITY: Final[EventPriority] = EventPriority.NORMAL
    DEFAULT_SIGNAL_PRIORITY: Final[EventPriority] = EventPriority.HIGH

    def __init__(
        self,
        *,
        config: BaseSpreadConfig,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        service_name: str = "spread_analyzer",
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
        self._config = config
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._service_name = service_name

        self._logger = get_logger(
            __name__,
            service_name=service_name,
            event_type="spread_analyzer",
        )

        self._regime_detector = SpreadRegimeDetector(config)
        self._signal_engine = SpreadSignalEngine(
            config=config,
            regime_detector=self._regime_detector,
        )

        self._running = False
        self._registered = False
        self._lock = asyncio.Lock()

        self._subscriptions: list[Subscription] = []
        self._scheduler_job_ids: list[str] = []

        self._last_signal_times: dict[str, datetime] = {}
        self._last_emit_times: dict[tuple[Any, ...], datetime] = {}

        self._stats: dict[str, int] = self._build_base_stats()

    # ------------------------------------------------------------------
    # Required subclass API
    # ------------------------------------------------------------------

    @abstractmethod
    def register(self) -> None:
        """
        Конкретний analyzer має підписатись на потрібні EventBus topics.

        Production subscriptions мають використовувати:
            self._subscribe_orderbook_updates(...)
            self._subscribe_funding_updates(...)

        Backward-compatible:
            self._subscribe_quote_updates(...)

        `_subscribe_quote_updates(...)` лишається alias і теж підписує
        analyzer на market.orderbook.updated, якщо config уже оновлений.
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
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Конкретний analyzer має повернути розширену статистику.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_stats", _analytics_args)
        except Exception:
            pass
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Вмикає analyzer.

        Важливо:
        - start() не робить EventBus.subscribe() напряму;
        - підписки мають створюватись через register();
        - periodic jobs запускаються через Scheduler, якщо він переданий.
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
        if self._running:
            self._logger.warning(
                "Analyzer already started | analyzer=%s",
                self.__class__.__name__,
            )
            return

        if not self._config.enabled:
            self._logger.warning(
                "Analyzer start skipped because config.enabled=False | analyzer=%s",
                self.__class__.__name__,
            )
            return

        if not self._registered:
            self.register()

        self._running = True
        self._register_scheduler_jobs()

        self._logger.info(
            "Analyzer started | analyzer=%s",
            self.__class__.__name__,
            extra=self._build_start_log_extra(),
        )

        await self._emit_lifecycle_event(
            self._config.analyzer_started_event_topic,
            {
                "analyzer": self.__class__.__name__,
                "service_name": self._service_name,
                "production_input_topics": list(self._production_input_topics()),
                "production_price_input_topics": list(self._production_price_input_topics()),
                "scope": "exchange:market_type:symbol:timeframe",
            },
        )

    async def stop(self) -> None:
        """
        Вимикає analyzer.

        За замовчуванням stop() не видаляє EventBus subscriptions, а лише
        переводить analyzer у неактивний стан. Для повної відписки викликати unregister().

        Scheduler jobs прибираються/disable-яться best-effort, щоб повторний
        start() не створював дублікати cleanup/heartbeat jobs.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "stop", _analytics_args)
        except Exception:
            pass
        if not self._running:
            self._logger.warning(
                "Analyzer already stopped | analyzer=%s",
                self.__class__.__name__,
            )
            return

        self._running = False
        self._clear_runtime_state_on_stop()
        self._remove_scheduler_jobs()

        self._logger.info(
            "Analyzer stopped | analyzer=%s",
            self.__class__.__name__,
            extra=self._build_stop_log_extra(),
        )

        await self._emit_lifecycle_event(
            self._config.analyzer_stopped_event_topic,
            {
                "analyzer": self.__class__.__name__,
                "service_name": self._service_name,
                "stats": self._stats.copy(),
            },
        )

    def unregister(self) -> None:
        """
        Повністю відписує analyzer від EventBus.

        Використовувати під час shutdown/reconfigure, коли об'єкт більше
        не має отримувати події.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "unregister", _analytics_args)
        except Exception:
            pass
        if not self._subscriptions:
            self._registered = False
            return

        for subscription in list(self._subscriptions):
            try:
                self._event_bus.unsubscribe(subscription)
            except Exception as exc:
                self._mark_exception(
                    "Failed to unsubscribe EventBus subscription",
                    exc,
                    pattern=getattr(subscription, "pattern", None),
                    handler=getattr(subscription, "name", None),
                )

        self._subscriptions.clear()
        self._registered = False

        self._logger.info(
            "Analyzer unregistered | analyzer=%s",
            self.__class__.__name__,
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

    @property
    def is_registered(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_registered", _analytics_args)
        except Exception:
            pass
        return self._registered

    # ------------------------------------------------------------------
    # Scoped key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_exchange(exchange: object) -> str:
        try:
            _analytics_class_name = "BaseSpreadAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalize_exchange", _analytics_args)
        except Exception:
            pass
        normalized = str(exchange or "").strip().lower()
        if not normalized:
            raise ValueError("exchange must not be empty")
        return normalized

    @staticmethod
    def normalize_symbol(symbol: object) -> str:
        try:
            _analytics_class_name = "BaseSpreadAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalize_symbol", _analytics_args)
        except Exception:
            pass
        normalized = (
            str(symbol or "")
            .replace("-", "")
            .replace("/", "")
            .replace("_", "")
            .upper()
            .strip()
        )
        if not normalized:
            raise ValueError("symbol must not be empty")
        return normalized

    @staticmethod
    def normalize_market_type(market_type: object) -> str:
        try:
            _analytics_class_name = "BaseSpreadAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalize_market_type", _analytics_args)
        except Exception:
            pass
        normalized = str(market_type or "").strip().lower()
        if not normalized:
            raise ValueError("market_type must not be empty")
        return normalized

    @staticmethod
    def normalize_timeframe(timeframe: object = DEFAULT_TIMEFRAME) -> str:
        try:
            _analytics_class_name = "BaseSpreadAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalize_timeframe", _analytics_args)
        except Exception:
            pass
        normalized = str(timeframe or DEFAULT_TIMEFRAME).strip()
        return normalized if normalized else DEFAULT_TIMEFRAME

    def make_key(
        self,
        *,
        exchange: object,
        market_type: object,
        symbol: object,
        timeframe: object | None = None,
    ) -> SpreadKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "make_key", _analytics_args)
        except Exception:
            pass
        return make_spread_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe or self._config.default_timeframe,
        )

    def key_to_dict(self, key: SpreadKey) -> dict[str, str]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "key_to_dict", _analytics_args)
        except Exception:
            pass
        return spread_key_to_dict(key)

    def should_process_key(self, key: SpreadKey) -> bool:
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
        return self._config.should_process_key(key)

    def should_process_scope(
        self,
        *,
        symbol: object,
        market_type: object,
        timeframe: object | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "should_process_scope", _analytics_args)
        except Exception:
            pass
        return self._config.should_process_scope(
            symbol=self.normalize_symbol(symbol),
            market_type=self.normalize_market_type(market_type),
            timeframe=self.normalize_timeframe(timeframe or self._config.default_timeframe),
        )

    def extract_key_from_quote(self, quote: QuoteSnapshot) -> SpreadKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "extract_key_from_quote", _analytics_args)
        except Exception:
            pass
        return quote.key

    def extract_key_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        default_timeframe: str | None = None,
    ) -> SpreadKey | None:
        """
        Витягує SpreadKey із data-layer payload.

        Мінімальні поля:
            exchange, market_type, symbol

        timeframe optional, default = config.default_timeframe.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "extract_key_from_payload", _analytics_args)
        except Exception:
            pass
        exchange = payload.get("exchange")
        market_type = payload.get("market_type")
        symbol = payload.get("symbol")
        timeframe = payload.get("timeframe") or default_timeframe or self._config.default_timeframe

        if not exchange or not market_type or not symbol:
            return None

        try:
            return self.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        except ValueError:
            return None

    def extract_key_from_event(self, event: Event) -> SpreadKey | None:
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

        if isinstance(payload, QuoteSnapshot):
            return payload.key

        if isinstance(payload, FundingSnapshot):
            return payload.key

        if isinstance(payload, Mapping):
            return self.extract_key_from_payload(payload)

        return None

    # ------------------------------------------------------------------
    # Payload normalization helpers
    # ------------------------------------------------------------------

    def normalize_orderbook_payload(
        self,
        payload: QuoteSnapshot | Mapping[str, Any],
    ) -> QuoteSnapshot | None:
        """
        Нормалізує production price payload у QuoteSnapshot.

        Production-compatible payload styles:
        - QuoteSnapshot;
        - dict payload від OrderBookCache / market.orderbook.updated.

        QuoteSnapshot тут є внутрішньою normalized top-of-book моделлю.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalize_orderbook_payload", _analytics_args)
        except Exception:
            pass
        try:
            if isinstance(payload, QuoteSnapshot):
                return payload

            if isinstance(payload, Mapping):
                return quote_snapshot_from_orderbook_payload(payload)

            return None

        except Exception as exc:
            self._mark_exception(
                "Failed to normalize orderbook payload to quote snapshot",
                exc,
                payload_type=type(payload).__name__,
            )
            return None

    def normalize_quote_payload(
        self,
        payload: QuoteSnapshot | Mapping[str, Any],
    ) -> QuoteSnapshot | None:
        """
        Backward-compatible alias.

        Старі analyzer-и ще можуть викликати normalize_quote_payload(),
        але production input уже приходить із OrderBookCache як
        market.orderbook.updated.

        Підтримує:
        - QuoteSnapshot;
        - dict payload із top-of-book orderbook полями;
        - legacy quote-like dict payload, якщо QuoteSnapshot.from_payload()
          ще використовується у тестах або старих інтеграціях.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalize_quote_payload", _analytics_args)
        except Exception:
            pass
        try:
            if isinstance(payload, QuoteSnapshot):
                return payload

            if not isinstance(payload, Mapping):
                return None

            if self._looks_like_orderbook_payload(payload):
                return quote_snapshot_from_orderbook_payload(payload)

            return QuoteSnapshot.from_payload(payload)

        except Exception as exc:
            self._mark_exception(
                "Failed to normalize quote/orderbook payload",
                exc,
                payload_type=type(payload).__name__,
            )
            return None

    def normalize_funding_payload(
        self,
        payload: FundingSnapshot | Mapping[str, Any],
    ) -> FundingSnapshot | None:
        """
        Підтримує обидва production-compatible payload styles:
        - FundingSnapshot;
        - dict payload від FundingCache / data-layer.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalize_funding_payload", _analytics_args)
        except Exception:
            pass
        try:
            if isinstance(payload, FundingSnapshot):
                return payload

            if isinstance(payload, Mapping):
                return FundingSnapshot.from_payload(payload)

            return None

        except Exception as exc:
            self._mark_exception(
                "Failed to normalize funding payload",
                exc,
                payload_type=type(payload).__name__,
            )
            return None

    def normalize_orderbook_event(self, event: Event) -> QuoteSnapshot | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalize_orderbook_event", _analytics_args)
        except Exception:
            pass
        return self.normalize_orderbook_payload(getattr(event, "payload", None))

    def normalize_quote_event(self, event: Event) -> QuoteSnapshot | None:
        """
        Backward-compatible event normalizer.

        Поточні analyzer-и ще можуть мати handler-и on_quote_update(),
        але якщо вони підписані через _subscribe_quote_updates(), фактично
        отримуватимуть market.orderbook.updated і будуть нормалізовані тут.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalize_quote_event", _analytics_args)
        except Exception:
            pass
        return self.normalize_quote_payload(getattr(event, "payload", None))

    def normalize_funding_event(self, event: Event) -> FundingSnapshot | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalize_funding_event", _analytics_args)
        except Exception:
            pass
        return self.normalize_funding_payload(getattr(event, "payload", None))

    @staticmethod
    def _looks_like_orderbook_payload(payload: Mapping[str, Any]) -> bool:
        """
        Визначає, чи dict схожий на payload від OrderBookCache.

        Підтримує різні top-of-book формати:
        - bids/asks;
        - best_bid/best_ask;
        - best_bid_price/best_ask_price;
        - bid_price/ask_price;
        - bid/ask.
        """
        try:
            _analytics_class_name = "BaseSpreadAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_looks_like_orderbook_payload", _analytics_args)
        except Exception:
            pass
        orderbook_keys = {
            "bids",
            "asks",
            "best_bid",
            "best_ask",
            "best_bid_price",
            "best_ask_price",
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
        }
        return any(key in payload for key in orderbook_keys)

    # ------------------------------------------------------------------
    # EventBus helpers
    # ------------------------------------------------------------------

    def _subscribe_orderbook_updates(
        self,
        handler: Any,
        *,
        name: str | None = None,
    ) -> list[Subscription]:
        """
        Підписка на production top-of-book/orderbook updates із data-layer.

        Production topic:
            market.orderbook.updated
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_subscribe_orderbook_updates", _analytics_args)
        except Exception:
            pass
        return [
            self._subscribe(
                topic,
                handler,
                name=name or f"{self.__class__.__name__}.on_orderbook_update",
            )
            for topic in self._orderbook_event_topic_patterns()
        ]

    def _subscribe_quote_updates(
        self,
        handler: Any,
        *,
        name: str | None = None,
    ) -> list[Subscription]:
        """
        Backward-compatible alias для старих analyzer-ів.

        Раніше цей метод підписував на market.quote.updated.
        Тепер, після оновлення config.py, quote_event_topic_patterns мають
        вказувати на market.orderbook.updated.

        Це дозволяє поточним on_quote_update handler-ам працювати з
        orderbook payload без негайного перейменування методів у дочірніх класах.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_subscribe_quote_updates", _analytics_args)
        except Exception:
            pass
        return [
            self._subscribe(
                topic,
                handler,
                name=name or f"{self.__class__.__name__}.on_quote_update",
            )
            for topic in self._quote_event_topic_patterns()
        ]

    def _subscribe_funding_updates(
        self,
        handler: Any,
        *,
        name: str | None = None,
    ) -> list[Subscription]:
        """
        Підписка на production funding updates із data-layer.

        Production topic:
            market.funding.updated
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_subscribe_funding_updates", _analytics_args)
        except Exception:
            pass
        return [
            self._subscribe(
                topic,
                handler,
                name=name or f"{self.__class__.__name__}.on_funding_update",
            )
            for topic in self._funding_event_topic_patterns()
        ]

    def _subscribe_legacy_raw_inputs(
        self,
        orderbook_handler: Any | None = None,
        quote_handler: Any | None = None,
        funding_handler: Any | None = None,
    ) -> list[Subscription]:
        """
        Legacy/raw subscriptions.

        Не використовувати в production, якщо allow_legacy_raw_topics=False.

        Підтримується для dev/testing:
        - raw market.orderbook;
        - raw market.quote;
        - raw market.funding.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_subscribe_legacy_raw_inputs", _analytics_args)
        except Exception:
            pass
        if not getattr(self._config, "allow_legacy_raw_topics", False):
            self._logger.warning(
                "Legacy raw input subscription skipped because allow_legacy_raw_topics=False | analyzer=%s",
                self.__class__.__name__,
            )
            return []

        subscriptions: list[Subscription] = []

        if orderbook_handler is not None:
            subscriptions.append(
                self._subscribe(
                    getattr(self._config, "raw_orderbook_event_topic", "market.orderbook"),
                    orderbook_handler,
                    name=f"{self.__class__.__name__}.on_raw_orderbook",
                    allow_raw=True,
                )
            )

        if quote_handler is not None:
            subscriptions.append(
                self._subscribe(
                    getattr(self._config, "raw_quote_event_topic", "market.quote"),
                    quote_handler,
                    name=f"{self.__class__.__name__}.on_raw_quote",
                    allow_raw=True,
                )
            )

        if funding_handler is not None:
            subscriptions.append(
                self._subscribe(
                    getattr(self._config, "raw_funding_event_topic", "market.funding"),
                    funding_handler,
                    name=f"{self.__class__.__name__}.on_raw_funding",
                    allow_raw=True,
                )
            )

        return subscriptions

    def _subscribe(
        self,
        topic_pattern: str,
        handler: Any,
        *,
        name: str | None = None,
        allow_raw: bool = False,
    ) -> Subscription:
        """
        Реєструє EventBus subscription і зберігає Subscription для unregister().

        За замовчуванням блокує raw topics:
            market.orderbook
            market.quote
            market.funding

        Production topics:
            market.orderbook.updated
            market.funding.updated
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_subscribe", _analytics_args)
        except Exception:
            pass
        self._validate_subscription_topic(topic_pattern, allow_raw=allow_raw)

        subscription = self._event_bus.subscribe(
            topic_pattern,
            handler,
            name=name or f"{self.__class__.__name__}.{getattr(handler, '__name__', 'handler')}",
        )
        self._subscriptions.append(subscription)

        self._logger.info(
            "Analyzer subscribed | analyzer=%s topic_pattern=%s handler=%s",
            self.__class__.__name__,
            topic_pattern,
            subscription.name,
        )

        return subscription

    def _validate_subscription_topic(
        self,
        topic_pattern: str,
        *,
        allow_raw: bool = False,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_subscription_topic", _analytics_args)
        except Exception:
            pass
        raw_topics = set(self._legacy_raw_input_topics())

        if topic_pattern in raw_topics and not allow_raw:
            raise ValueError(
                f"{self.__class__.__name__} tried to subscribe to raw topic "
                f"{topic_pattern!r}. Use data-layer updated topics instead."
            )

        if topic_pattern in raw_topics and allow_raw and not getattr(
            self._config,
            "allow_legacy_raw_topics",
            False,
        ):
            raise ValueError(
                f"{self.__class__.__name__} tried to subscribe to raw topic "
                f"{topic_pattern!r}, but config.allow_legacy_raw_topics=False."
            )

        legacy_quote_topics = set(self._legacy_quote_input_topics())
        allow_legacy_quote = getattr(self._config, "allow_legacy_quote_topics", False)

        if topic_pattern in legacy_quote_topics and not allow_legacy_quote:
            raise ValueError(
                f"{self.__class__.__name__} tried to subscribe to legacy quote topic "
                f"{topic_pattern!r}. Use market.orderbook.updated from OrderBookCache instead."
            )

    async def _emit(
        self,
        topic: str,
        payload: Any,
        *,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        """
        Єдиний helper для EventBus.emit().

        Payload серіалізується в dict, якщо модель має to_payload().
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit", _analytics_args)
        except Exception:
            pass
        try:
            accepted = await self._event_bus.emit(
                topic,
                self._payload_for_eventbus(payload),
                priority=priority,
                source=self._service_name,
                correlation_id=correlation_id,
                headers=headers or {},
            )
            if not accepted:
                self._stats["events_rejected"] += 1
                self._logger.warning(
                    "Event rejected by EventBus | topic=%s analyzer=%s",
                    topic,
                    self.__class__.__name__,
                )
            return accepted

        except Exception as exc:
            self._stats["events_failed"] += 1
            self._mark_exception(
                "Failed to emit EventBus event",
                exc,
                topic=topic,
                analyzer=self.__class__.__name__,
            )
            return False

    async def _emit_lifecycle_event(
        self,
        topic: str,
        payload: dict[str, Any],
    ) -> None:
        """
        Best-effort lifecycle event.

        Не піднімає exception назовні, щоб lifecycle analyzer-а не ламався
        через telemetry/event issue.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_lifecycle_event", _analytics_args)
        except Exception:
            pass
        try:
            await self._event_bus.emit(
                topic,
                self._payload_for_eventbus(payload),
                priority=EventPriority.LOW,
                source=self._service_name,
            )
        except Exception:
            self._logger.exception(
                "Failed to emit analyzer lifecycle event | topic=%s analyzer=%s",
                topic,
                self.__class__.__name__,
            )

    # ------------------------------------------------------------------
    # Scheduler helpers
    # ------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        """
        Hook для дочірніх analyzer-ів.

        Базовий клас не додає jobs сам, бо не знає конкретної state-cleanup
        логіки. Дочірній клас може override-нути цей метод і викликати
        _add_interval_job().
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_register_scheduler_jobs", _analytics_args)
        except Exception:
            pass
        return

    def _add_interval_job(
        self,
        *,
        name: str,
        func: Any,
        interval: float,
        run_immediately: bool = False,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        timeout: float | None = None,
        allow_overlap: bool = False,
        enabled: bool = True,
    ) -> str | None:
        """
        Безпечний helper для Scheduler.add_interval_job().
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_add_interval_job", _analytics_args)
        except Exception:
            pass
        if self._scheduler is None:
            self._logger.debug(
                "Scheduler is not configured; interval job skipped | job=%s analyzer=%s",
                name,
                self.__class__.__name__,
            )
            return None

        existing_job = self._scheduler.get_job_by_name(name)
        if existing_job is not None:
            self._logger.debug(
                "Scheduler job already exists | job=%s analyzer=%s",
                name,
                self.__class__.__name__,
            )
            return existing_job.job_id

        try:
            job_id = self._scheduler.add_interval_job(
                name=name,
                func=func,
                interval=interval,
                run_immediately=run_immediately,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout,
                allow_overlap=allow_overlap,
                enabled=enabled,
            )
            self._scheduler_job_ids.append(job_id)

            self._logger.info(
                "Scheduler interval job added | job=%s interval=%s analyzer=%s",
                name,
                interval,
                self.__class__.__name__,
            )
            return job_id

        except TypeError:
            try:
                job_id = self._scheduler.add_interval_job(
                    name=name,
                    callback=func,
                    interval_seconds=interval,
                    run_immediately=run_immediately,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    timeout=timeout,
                    allow_overlap=allow_overlap,
                    enabled=enabled,
                )
                self._scheduler_job_ids.append(job_id)
                return job_id
            except Exception as exc:
                self._mark_exception(
                    "Failed to add scheduler interval job with fallback signature",
                    exc,
                    job_name=name,
                    interval=interval,
                    analyzer=self.__class__.__name__,
                )
                return None

        except Exception as exc:
            self._mark_exception(
                "Failed to add scheduler interval job",
                exc,
                job_name=name,
                interval=interval,
                analyzer=self.__class__.__name__,
            )
            return None

    def _remove_scheduler_jobs(self) -> None:
        """
        Best-effort cleanup для Scheduler jobs, створених analyzer-ом.

        Потрібно для безпечного stop/start без дублікатів cleanup/heartbeat jobs.
        Підтримує Scheduler API:
        - remove_job(job_id)
        - disable_job(job_id)
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_remove_scheduler_jobs", _analytics_args)
        except Exception:
            pass
        if self._scheduler is None or not self._scheduler_job_ids:
            return

        for job_id in list(self._scheduler_job_ids):
            try:
                remove_job = getattr(self._scheduler, "remove_job", None)
                if callable(remove_job):
                    remove_job(job_id)
                    continue

                disable_job = getattr(self._scheduler, "disable_job", None)
                if callable(disable_job):
                    disable_job(job_id)
                    continue

                self._logger.debug(
                    "Scheduler has no remove_job/disable_job method | analyzer=%s job_id=%s",
                    self.__class__.__name__,
                    job_id,
                )

            except Exception as exc:
                self._mark_exception(
                    "Failed to remove scheduler job",
                    exc,
                    job_id=job_id,
                    analyzer=self.__class__.__name__,
                )

        self._scheduler_job_ids.clear()

    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------

    async def _publish_snapshot(
        self,
        topic: str,
        snapshot: SpreadSnapshot,
        *,
        priority: EventPriority = DEFAULT_SNAPSHOT_PRIORITY,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_snapshot", _analytics_args)
        except Exception:
            pass
        accepted = await self._emit(
            topic,
            snapshot,
            priority=priority,
            correlation_id=correlation_id,
            headers=headers,
        )

        if not accepted:
            return False

        self._stats["snapshots_published"] += 1

        self._logger.debug(
            "Spread snapshot published | topic=%s symbol=%s spread_type=%s",
            topic,
            snapshot.symbol,
            snapshot.spread_type.value,
            extra={
                "symbol": snapshot.symbol,
                "event_type": topic,
                "spread_type": snapshot.spread_type.value,
                "exchange_a": snapshot.leg_a_exchange,
                "exchange_b": snapshot.leg_b_exchange,
                "market_type_a": snapshot.leg_a_market_type,
                "market_type_b": snapshot.leg_b_market_type,
                "timeframe": snapshot.timeframe,
                "spread_bps": self._to_str(snapshot.spread_bps),
                "net_spread": self._to_str(snapshot.net_spread),
                "regime": snapshot.regime.value,
            },
        )
        return True

    async def _publish_signal(
        self,
        topic: str,
        signal: SpreadSignal,
        *,
        priority: EventPriority = DEFAULT_SIGNAL_PRIORITY,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_signal", _analytics_args)
        except Exception:
            pass
        if self._should_skip_signal(signal):
            self._stats["cooldown_skips"] += 1
            return False

        accepted = await self._emit(
            topic,
            signal,
            priority=priority,
            correlation_id=correlation_id,
            headers=headers,
        )

        if not accepted:
            return False

        self._stats["signals_published"] += 1

        self._logger.debug(
            "Spread signal published | topic=%s signal_type=%s symbol=%s",
            topic,
            signal.signal_type.value,
            signal.symbol,
            extra={
                "symbol": signal.symbol,
                "event_type": topic,
                "signal_type": signal.signal_type.value,
                "spread_type": signal.spread_type.value,
                "exchange_a": signal.exchange_a,
                "exchange_b": signal.exchange_b,
                "market_type_a": signal.market_type_a,
                "market_type_b": signal.market_type_b,
                "timeframe": signal.timeframe,
                "value": self._to_str(signal.value),
                "threshold": self._to_str(signal.threshold),
                "confidence": self._to_str(signal.confidence),
            },
        )
        return True

    async def _publish_signals(
        self,
        topic: str,
        signals: list[SpreadSignal],
        *,
        priority: EventPriority = DEFAULT_SIGNAL_PRIORITY,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_signals", _analytics_args)
        except Exception:
            pass
        published_count = 0

        for signal in signals:
            published = await self._publish_signal(
                topic,
                signal,
                priority=priority,
                correlation_id=correlation_id,
                headers=headers,
            )
            if published:
                published_count += 1

        return published_count

    # ------------------------------------------------------------------
    # Signal / throttling helpers
    # ------------------------------------------------------------------

    def _evaluate_snapshot_signals(
        self,
        *,
        snapshot: SpreadSnapshot,
        previous_snapshot: SpreadSnapshot | None = None,
        opportunity: Any | None = None,
    ) -> list[SpreadSignal]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_evaluate_snapshot_signals", _analytics_args)
        except Exception:
            pass
        result = self._signal_engine.evaluate_snapshot(
            snapshot=snapshot,
            previous_snapshot=previous_snapshot,
            opportunity=opportunity,
        )
        return result.signals

    def _should_skip_emit(
        self,
        key: tuple[Any, ...],
        timestamp: datetime,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_should_skip_emit", _analytics_args)
        except Exception:
            pass
        last_emit_at = self._last_emit_times.get(key)
        if last_emit_at is None:
            self._last_emit_times[key] = timestamp
            return False

        min_interval = timedelta(milliseconds=self._config.min_emit_interval_ms)
        if (timestamp - last_emit_at) < min_interval:
            return True

        self._last_emit_times[key] = timestamp
        return False

    def _should_skip_snapshot_emit(
        self,
        snapshot: SpreadSnapshot,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_should_skip_snapshot_emit", _analytics_args)
        except Exception:
            pass
        return self._should_skip_emit(
            key=(
                snapshot.spread_type.value,
                snapshot.symbol,
                snapshot.leg_a_exchange,
                snapshot.leg_a_market_type,
                snapshot.leg_b_exchange,
                snapshot.leg_b_market_type,
                snapshot.timeframe,
            ),
            timestamp=snapshot.timestamp,
        )

    def _should_skip_signal(
        self,
        signal: SpreadSignal,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_should_skip_signal", _analytics_args)
        except Exception:
            pass
        signal_key = self._build_signal_key(signal)
        now = signal.timestamp

        last_signal_at = self._last_signal_times.get(signal_key)
        if last_signal_at is None:
            self._last_signal_times[signal_key] = now
            return False

        cooldown = timedelta(seconds=self._config.cooldown_seconds)
        if (now - last_signal_at) < cooldown:
            return True

        self._last_signal_times[signal_key] = now
        return False

    def _build_signal_key(self, signal: SpreadSignal) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_signal_key", _analytics_args)
        except Exception:
            pass
        exchange_a = signal.exchange_a or "na"
        exchange_b = signal.exchange_b or "na"
        market_type_a = signal.market_type_a or "na"
        market_type_b = signal.market_type_b or "na"

        return (
            f"{signal.signal_type.value}|"
            f"{signal.spread_type.value}|"
            f"{signal.symbol}|"
            f"{exchange_a}|{market_type_a}|"
            f"{exchange_b}|{market_type_b}|"
            f"{signal.timeframe}"
        )

    # ------------------------------------------------------------------
    # Config compatibility helpers
    # ------------------------------------------------------------------

    def _orderbook_event_topic_patterns(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_orderbook_event_topic_patterns", _analytics_args)
        except Exception:
            pass
        value = getattr(self._config, "orderbook_event_topic_patterns", None)
        if value:
            return tuple(value)

        value = getattr(self._config, "quote_event_topic_patterns", None)
        if value:
            return tuple(value)

        topic = getattr(self._config, "orderbook_event_topic", None)
        if topic:
            return (topic,)

        topic = getattr(self._config, "quote_event_topic", None)
        if topic:
            return (topic,)

        return ("market.orderbook.updated",)

    def _quote_event_topic_patterns(self) -> tuple[str, ...]:
        """
        Backward-compatible topic resolver.

        Після оновлення config.py quote_event_topic_patterns уже має вказувати
        на market.orderbook.updated. Якщо в config є окремі orderbook patterns,
        вони мають пріоритет.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_quote_event_topic_patterns", _analytics_args)
        except Exception:
            pass
        value = getattr(self._config, "orderbook_event_topic_patterns", None)
        if value:
            return tuple(value)

        value = getattr(self._config, "quote_event_topic_patterns", None)
        if value:
            return tuple(value)

        topic = getattr(self._config, "orderbook_event_topic", None)
        if topic:
            return (topic,)

        topic = getattr(self._config, "quote_event_topic", None)
        if topic:
            return (topic,)

        return ("market.orderbook.updated",)

    def _funding_event_topic_patterns(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_funding_event_topic_patterns", _analytics_args)
        except Exception:
            pass
        value = getattr(self._config, "funding_event_topic_patterns", None)
        if value:
            return tuple(value)

        topic = getattr(self._config, "funding_event_topic", None)
        if topic:
            return (topic,)

        return ("market.funding.updated",)

    def _production_price_input_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_production_price_input_topics", _analytics_args)
        except Exception:
            pass
        value = getattr(self._config, "production_price_input_topics", None)
        if value:
            return tuple(value)

        return self._orderbook_event_topic_patterns()

    def _production_input_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_production_input_topics", _analytics_args)
        except Exception:
            pass
        value = getattr(self._config, "production_input_topics", None)
        if value:
            return tuple(value)

        return (
            *self._orderbook_event_topic_patterns(),
            *self._funding_event_topic_patterns(),
        )

    def _legacy_raw_input_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_legacy_raw_input_topics", _analytics_args)
        except Exception:
            pass
        value = getattr(self._config, "legacy_raw_input_topics", None)
        if value:
            return tuple(value)

        return (
            getattr(self._config, "raw_orderbook_event_topic", "market.orderbook"),
            getattr(self._config, "raw_quote_event_topic", "market.quote"),
            getattr(self._config, "raw_funding_event_topic", "market.funding"),
        )

    def _legacy_quote_input_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_legacy_quote_input_topics", _analytics_args)
        except Exception:
            pass
        value = getattr(self._config, "legacy_quote_input_topics", None)
        if value:
            return tuple(value)

        topic = getattr(self._config, "legacy_quote_event_topic", "market.quote.updated")
        return (topic,)

    # ------------------------------------------------------------------
    # Stats / logging helpers
    # ------------------------------------------------------------------

    def _build_base_stats(self) -> dict[str, int]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_base_stats", _analytics_args)
        except Exception:
            pass
        return {
            "calculations_total": 0,
            "snapshots_published": 0,
            "signals_published": 0,
            "cooldown_skips": 0,
            "emit_skips": 0,
            "events_rejected": 0,
            "events_failed": 0,
            "invalid_payloads": 0,
            "events_skipped_not_running": 0,
            "events_skipped_scope": 0,
            "legacy_raw_events": 0,
            "exceptions": 0,
        }

    def _build_start_log_extra(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_start_log_extra", _analytics_args)
        except Exception:
            pass
        return {
            "enabled": self._config.enabled,
            "max_quote_age_ms": self._config.max_quote_age_ms,
            "max_quote_skew_ms": self._config.max_quote_skew_ms,
            "rolling_window_size": self._config.rolling_window_size,
            "min_emit_interval_ms": self._config.min_emit_interval_ms,
            "cooldown_seconds": self._config.cooldown_seconds,
            "subscriptions": len(self._subscriptions),
            "scheduler_jobs": len(self._scheduler_job_ids),
            "production_input_topics": list(self._production_input_topics()),
            "production_price_input_topics": list(self._production_price_input_topics()),
            "allow_legacy_raw_topics": getattr(self._config, "allow_legacy_raw_topics", False),
            "allow_legacy_quote_topics": getattr(self._config, "allow_legacy_quote_topics", False),
            "scope": "exchange:market_type:symbol:timeframe",
        }

    def _build_stop_log_extra(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_stop_log_extra", _analytics_args)
        except Exception:
            pass
        return {
            "stats": self._stats.copy(),
            "subscriptions": len(self._subscriptions),
            "scheduler_jobs": len(self._scheduler_job_ids),
        }

    def _mark_exception(
        self,
        message: str,
        exc: Exception,
        **extra: Any,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_mark_exception", _analytics_args)
        except Exception:
            pass
        self._stats["exceptions"] += 1
        self._logger.exception(
            message,
            extra={
                "error": str(exc),
                "analyzer": self.__class__.__name__,
                **extra,
            },
        )

    def _mark_invalid_payload(
        self,
        reason: str,
        *,
        payload_type: str | None = None,
        topic: str | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_mark_invalid_payload", _analytics_args)
        except Exception:
            pass
        self._stats["invalid_payloads"] += 1
        self._logger.warning(
            "Invalid spread analyzer payload | analyzer=%s reason=%s payload_type=%s topic=%s",
            self.__class__.__name__,
            reason,
            payload_type,
            topic,
        )

    def _clear_runtime_state_on_stop(self) -> None:
        """
        Очищає runtime-only throttling state.

        Не очищає market caches дочірніх analyzer-ів — це їхня відповідальність.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_clear_runtime_state_on_stop", _analytics_args)
        except Exception:
            pass
        self._last_signal_times.clear()
        self._last_emit_times.clear()

    @staticmethod
    def _to_str(value: Decimal | None) -> str | None:
        try:
            _analytics_class_name = "BaseSpreadAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_to_str", _analytics_args)
        except Exception:
            pass
        return str(value) if value is not None else None

    @staticmethod
    def _payload_for_eventbus(payload: Any) -> Any:
        try:
            _analytics_class_name = "BaseSpreadAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_payload_for_eventbus", _analytics_args)
        except Exception:
            pass
        if hasattr(payload, "to_payload") and callable(payload.to_payload):
            return payload.to_payload()

        if isinstance(payload, Mapping):
            return {
                str(key): BaseSpreadAnalyzer._payload_for_eventbus(value)
                for key, value in payload.items()
            }

        if isinstance(payload, list):
            return [BaseSpreadAnalyzer._payload_for_eventbus(item) for item in payload]

        if isinstance(payload, tuple):
            return [BaseSpreadAnalyzer._payload_for_eventbus(item) for item in payload]

        if isinstance(payload, set):
            return sorted(BaseSpreadAnalyzer._payload_for_eventbus(item) for item in payload)

        try:
            return model_to_payload(payload)
        except Exception:
            return payload