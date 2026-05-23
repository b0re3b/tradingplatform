from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from analytics.whales.models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    UNKNOWN_EXCHANGE,
    WhaleBaseSignalModel,
    WhaleKey,
    make_whale_key,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    whale_key_to_dict,
)


EventHandler = Callable[[Event], None | Awaitable[None]]
JobCallable = Callable[..., None | Awaitable[None]]


RAW_WHALE_MARKET_TOPICS = {
    "market.trade",
    "market.liquidation",
}

PRODUCTION_WHALE_MARKET_TOPICS = {
    "market.trades.updated",
    "market.liquidations.updated",
}


class BaseWhaleComponent(ABC):
    """
    Базовий клас для всіх analytics.whales runtime-компонентів.

    Відповідає тільки за core-інтеграцію:
    - logger через core.logger.get_logger;
    - EventBus dependency injection;
    - optional Scheduler dependency injection;
    - register() / EventBus.subscribe();
    - централізоване збереження Subscription;
    - EventBus.emit();
    - Scheduler.add_interval_job();
    - lifecycle state;
    - scoped multi-exchange helpers;
    - production data-layer topic guard.

    Correct production input flow:
        exchange adapters
            -> market.trade / market.liquidation
            -> data cache / analytics liquidation layer
            -> market.trades.updated / market.liquidations.updated
            -> analytics.whales.*

    Важливо:
    - не створює власних background asyncio loops;
    - не читає exchange adapters напряму;
    - не має підписуватись на raw market.trade / market.liquidation,
      якщо legacy raw mode явно не дозволений дочірнім компонентом.
    """

    def __init__(
        self,
        *,
        component_name: str,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        default_exchange: str = UNKNOWN_EXCHANGE,
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
        self.component_name = component_name
        self.event_bus = event_bus
        self.scheduler = scheduler

        self.default_exchange = normalize_exchange(default_exchange)
        self.default_market_type = normalize_market_type(default_market_type)
        self.default_timeframe = normalize_timeframe(default_timeframe)

        self.logger = get_logger(
            __name__,
            service_name="analytics.whales",
            component=component_name,
            event_type="whale_component",
        )

        self._started = False
        self._registered = False
        self._subscriptions: list[Subscription] = []
        self._scheduler_job_ids: list[str] = []

        self._last_emit_ts_by_key: dict[tuple[Any, ...], float] = {}

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_started(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_started", _analytics_args)
        except Exception:
            pass
        return self._started

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

    @property
    def subscriptions(self) -> tuple[Subscription, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "subscriptions", _analytics_args)
        except Exception:
            pass
        return tuple(self._subscriptions)

    @property
    def scheduler_job_ids(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "scheduler_job_ids", _analytics_args)
        except Exception:
            pass
        return tuple(self._scheduler_job_ids)

    # =========================================================================
    # Lifecycle contract
    # =========================================================================

    @abstractmethod
    async def register(self) -> None:
        """
        Зареєструвати EventBus subscriptions.

        Дочірній клас має викликати:
            self._subscribe_production(...)
        або:
            self._subscribe_legacy_raw(...)

        Метод має бути idempotent.
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
    async def start(self) -> None:
        """
        Запуск runtime-компонента.

        Типовий порядок у дочірньому класі:
        1. перевірити config.enabled;
        2. await self.register();
        3. self._add_interval_job(... cleanup ...);
        4. self._started = True.
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
        raise NotImplementedError

    async def stop(self) -> None:
        """
        Базова зупинка компонента.

        Видаляє:
        - EventBus subscriptions;
        - Scheduler jobs, створені через _add_interval_job();
        - runtime cooldown state.
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
        if not self._started and not self._registered:
            return

        self._remove_scheduler_jobs()
        self._unsubscribe_all()
        self._last_emit_ts_by_key.clear()

        self._registered = False
        self._started = False

        self.logger.info(
            "Whale component stopped",
            extra={"component": self.component_name},
        )

    # =========================================================================
    # Health / diagnostics
    # =========================================================================

    def get_healthcheck(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_healthcheck", _analytics_args)
        except Exception:
            pass
        return {
            "component": self.component_name,
            "started": self._started,
            "registered": self._registered,
            "subscriptions": len(self._subscriptions),
            "scheduler_jobs": len(self._scheduler_job_ids),
            "event_bus_available": self.event_bus is not None,
            "scheduler_available": self.scheduler is not None,
            "default_scope": {
                "exchange": self.default_exchange,
                "market_type": self.default_market_type,
                "timeframe": self.default_timeframe,
            },
            "scope": "exchange:market_type:symbol:timeframe",
        }

    # =========================================================================
    # Scoped key helpers
    # =========================================================================

    def make_key(
        self,
        *,
        exchange: object | None = None,
        market_type: object | None = None,
        symbol: object,
        timeframe: object | None = None,
    ) -> WhaleKey:
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
        return make_whale_key(
            exchange=exchange or self.default_exchange,
            market_type=market_type or self.default_market_type,
            symbol=symbol,
            timeframe=timeframe or self.default_timeframe,
        )

    @staticmethod
    def key_to_dict(key: WhaleKey) -> dict[str, str]:
        try:
            _analytics_class_name = "BaseWhaleComponent"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "key_to_dict", _analytics_args)
        except Exception:
            pass
        return whale_key_to_dict(key)

    def scoped_mapping_key(self, key: WhaleKey) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "scoped_mapping_key", _analytics_args)
        except Exception:
            pass
        scope = whale_key_to_dict(key)
        return (
            f"{scope['exchange']}:"
            f"{scope['market_type']}:"
            f"{scope['symbol']}:"
            f"{scope['timeframe']}"
        )

    def extract_key_from_payload(
        self,
        payload: Mapping[str, Any],
    ) -> WhaleKey | None:
        """
        Витягує WhaleKey із payload.

        Мінімально потрібен symbol.
        exchange/market_type/timeframe можуть братися з defaults.
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
        symbol = payload.get("symbol")
        if not symbol:
            return None

        try:
            return self.make_key(
                exchange=payload.get("exchange") or self.default_exchange,
                market_type=payload.get("market_type") or self.default_market_type,
                symbol=symbol,
                timeframe=payload.get("timeframe") or self.default_timeframe,
            )
        except Exception:
            return None

    def extract_key_from_event(self, event: Event) -> WhaleKey | None:
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
        payload = self._payload_from_event(event)
        return self.extract_key_from_payload(payload)

    def should_process_key(self, key: WhaleKey, config: Any | None = None) -> bool:
        """
        Делегує scoped filtering у config, якщо config має should_process_key().
        Якщо ні — дозволяє key.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "should_process_key", _analytics_args)
        except Exception:
            pass
        if config is not None:
            checker = getattr(config, "should_process_key", None)
            if callable(checker):
                return bool(checker(key))
        return True

    def should_process_payload(
        self,
        payload: Mapping[str, Any],
        *,
        config: Any | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "should_process_payload", _analytics_args)
        except Exception:
            pass
        key = self.extract_key_from_payload(payload)
        if key is None:
            return False
        return self.should_process_key(key, config=config)

    # =========================================================================
    # EventBus helpers
    # =========================================================================

    def _subscribe_production(
        self,
        topic: str,
        handler: EventHandler,
        *,
        name: str | None = None,
    ) -> Subscription:
        """
        Підписка на production topic.

        Для market-data це мають бути data-layer updated topics:
            market.trades.updated
            market.liquidations.updated
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_subscribe_production", _analytics_args)
        except Exception:
            pass
        return self._subscribe(
            topic,
            handler,
            name=name,
            allow_raw=False,
        )

    def _subscribe_production_many(
        self,
        topics: tuple[str, ...] | list[str],
        handler: EventHandler,
        *,
        name: str | None = None,
    ) -> list[Subscription]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_subscribe_production_many", _analytics_args)
        except Exception:
            pass
        return [
            self._subscribe_production(
                topic,
                handler,
                name=name,
            )
            for topic in topics
        ]

    def _subscribe_legacy_raw(
        self,
        topic: str,
        handler: EventHandler,
        *,
        name: str | None = None,
        allow_legacy_raw_topics: bool,
    ) -> Subscription:
        """
        Підписка на raw market topic.

        Дозволено тільки для migration/test/manual mode.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_subscribe_legacy_raw", _analytics_args)
        except Exception:
            pass
        if not allow_legacy_raw_topics:
            raise ValueError(
                f"{self.component_name} tried to subscribe to raw topic {topic!r}, "
                "but allow_legacy_raw_topics=False"
            )

        return self._subscribe(
            topic,
            handler,
            name=name,
            allow_raw=True,
        )

    def _subscribe(
        self,
        topic: str,
        handler: EventHandler,
        *,
        name: str | None = None,
        allow_raw: bool = False,
    ) -> Subscription:
        """
        Зареєструвати handler у core EventBus.

        core.EventBus.subscribe() є sync-методом і повертає Subscription.
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
        self._validate_subscription_topic(topic, allow_raw=allow_raw)

        handler_name = name or f"{self.component_name}.{getattr(handler, '__name__', 'handler')}"

        subscription = self.event_bus.subscribe(
            topic,
            handler,
            name=handler_name,
        )
        self._subscriptions.append(subscription)

        self.logger.info(
            "Subscribed to EventBus topic",
            extra={
                "component": self.component_name,
                "topic": topic,
                "handler": handler_name,
                "allow_raw": allow_raw,
            },
        )
        return subscription

    def _validate_subscription_topic(
        self,
        topic: str,
        *,
        allow_raw: bool,
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
        if not topic or not topic.strip():
            raise ValueError("EventBus topic must be a non-empty string")

        normalized = topic.strip()

        if normalized in RAW_WHALE_MARKET_TOPICS and not allow_raw:
            raise ValueError(
                f"{self.component_name} attempted to subscribe to raw market topic "
                f"{normalized!r}. Use data-layer updated topic instead."
            )

    def _unsubscribe_all(self) -> None:
        """
        Відписати всі subscriptions, які створив цей компонент.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_unsubscribe_all", _analytics_args)
        except Exception:
            pass
        while self._subscriptions:
            subscription = self._subscriptions.pop()
            try:
                self.event_bus.unsubscribe(subscription)
                self.logger.info(
                    "Unsubscribed from EventBus topic",
                    extra={
                        "component": self.component_name,
                        "topic": subscription.pattern,
                        "handler": subscription.name,
                    },
                )
            except Exception:
                self.logger.exception(
                    "Failed to unsubscribe from EventBus topic",
                    extra={
                        "component": self.component_name,
                        "topic": getattr(subscription, "pattern", None),
                        "handler": getattr(subscription, "name", None),
                    },
                )

    async def _emit(
        self,
        topic: str,
        payload: Mapping[str, Any] | dict[str, Any] | WhaleBaseSignalModel,
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str | None = None,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        """
        Опублікувати payload через core EventBus.emit().
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
        if not topic or not topic.strip():
            raise ValueError("EventBus topic must be a non-empty string")

        event_payload = self._payload_for_eventbus(payload)

        try:
            accepted = await self.event_bus.emit(
                topic,
                event_payload,
                priority=priority,
                source=source or self.component_name,
                correlation_id=correlation_id,
                headers=headers or {},
            )

            if not accepted:
                self.logger.warning(
                    "EventBus rejected whale event",
                    extra={
                        "component": self.component_name,
                        "topic": topic,
                        "source": source or self.component_name,
                    },
                )

            return accepted

        except Exception:
            self.logger.exception(
                "Failed to emit whale event",
                extra={
                    "component": self.component_name,
                    "topic": topic,
                    "source": source or self.component_name,
                },
            )
            raise

    @staticmethod
    def _payload_for_eventbus(payload: Any) -> dict[str, Any]:
        """
        Перетворює dataclass/signal/mapping у EventBus-safe dict.
        """
        try:
            _analytics_class_name = "BaseWhaleComponent"
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
        if isinstance(payload, WhaleBaseSignalModel):
            return payload.to_payload()

        to_payload = getattr(payload, "to_payload", None)
        if callable(to_payload):
            value = to_payload()
            if isinstance(value, Mapping):
                return dict(value)

        if isinstance(payload, Mapping):
            return dict(payload)

        if is_dataclass(payload):
            return asdict(payload)

        raise TypeError(
            f"Expected mapping/dataclass/signal payload, got {type(payload).__name__}"
        )

    @staticmethod
    def _payload_from_event(event: Event) -> dict[str, Any]:
        """
        Дістати dict payload із core Event.

        Handler-и мають приймати core.event_bus.Event, а бізнес-методи можуть
        працювати з dict payload.
        """
        try:
            _analytics_class_name = "BaseWhaleComponent"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_payload_from_event", _analytics_args)
        except Exception:
            pass
        payload = event.payload

        if isinstance(payload, Mapping):
            return dict(payload)

        to_payload = getattr(payload, "to_payload", None)
        if callable(to_payload):
            value = to_payload()
            if isinstance(value, Mapping):
                return dict(value)

        if is_dataclass(payload):
            return asdict(payload)

        raise TypeError(
            f"Expected Event.payload to be mapping/dataclass, got {type(payload).__name__}"
        )

    @staticmethod
    def _event_correlation_id(event: Event) -> str | None:
        try:
            _analytics_class_name = "BaseWhaleComponent"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_event_correlation_id", _analytics_args)
        except Exception:
            pass
        correlation_id = getattr(event, "correlation_id", None)
        if isinstance(correlation_id, str) and correlation_id:
            return correlation_id

        event_id = getattr(event, "event_id", None)
        if isinstance(event_id, str) and event_id:
            return event_id

        return None

    # =========================================================================
    # Scheduler helpers
    # =========================================================================

    def _add_interval_job(
        self,
        *,
        name: str,
        func: JobCallable,
        interval: float,
        run_immediately: bool = False,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        timeout: float | None = None,
        allow_overlap: bool = False,
        enabled: bool = True,
    ) -> str | None:
        """
        Зареєструвати periodic job через core Scheduler.add_interval_job().
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
        if not name or not name.strip():
            raise ValueError("Scheduler job name must be a non-empty string")
        if interval <= 0:
            raise ValueError("Scheduler interval must be > 0")

        if self.scheduler is None:
            self.logger.warning(
                "Scheduler is not configured; interval job skipped",
                extra={
                    "component": self.component_name,
                    "job_name": name,
                    "interval": interval,
                },
            )
            return None

        existing_job = self.scheduler.get_job_by_name(name)
        if existing_job is not None:
            self._scheduler_job_ids.append(existing_job.job_id)
            return existing_job.job_id

        job_id = self.scheduler.add_interval_job(
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

        self.logger.info(
            "Registered scheduler interval job",
            extra={
                "component": self.component_name,
                "job_name": name,
                "job_id": job_id,
                "interval": interval,
                "run_immediately": run_immediately,
                "allow_overlap": allow_overlap,
            },
        )
        return job_id

    def _remove_scheduler_jobs(self) -> None:
        """
        Видалити jobs, створені цим компонентом.
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
        if self.scheduler is None:
            self._scheduler_job_ids.clear()
            return

        while self._scheduler_job_ids:
            job_id = self._scheduler_job_ids.pop()
            try:
                self.scheduler.remove_job(job_id)
                self.logger.info(
                    "Removed scheduler job",
                    extra={
                        "component": self.component_name,
                        "job_id": job_id,
                    },
                )
            except KeyError:
                self.logger.warning(
                    "Scheduler job already removed",
                    extra={
                        "component": self.component_name,
                        "job_id": job_id,
                    },
                )
            except Exception:
                self.logger.exception(
                    "Failed to remove scheduler job",
                    extra={
                        "component": self.component_name,
                        "job_id": job_id,
                    },
                )

    # =========================================================================
    # Shared helpers
    # =========================================================================

    @staticmethod
    def _passes_cooldown(last_ts_monotonic: float, cooldown_sec: float) -> bool:
        try:
            _analytics_class_name = "BaseWhaleComponent"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_passes_cooldown", _analytics_args)
        except Exception:
            pass
        if cooldown_sec <= 0:
            return True
        return (time.monotonic() - last_ts_monotonic) >= cooldown_sec

    def _passes_key_cooldown(
        self,
        cooldown_key: tuple[Any, ...],
        *,
        cooldown_sec: float,
    ) -> bool:
        """
        Key-aware cooldown helper для компонентів, які не зберігають cooldown
        у власному state.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_passes_key_cooldown", _analytics_args)
        except Exception:
            pass
        if cooldown_sec <= 0:
            self._last_emit_ts_by_key[cooldown_key] = time.monotonic()
            return True

        now = time.monotonic()
        last_ts = self._last_emit_ts_by_key.get(cooldown_key, 0.0)

        if (now - last_ts) < cooldown_sec:
            return False

        self._last_emit_ts_by_key[cooldown_key] = now
        return True

    @staticmethod
    def _clamp_0_1(value: float) -> float:
        try:
            _analytics_class_name = "BaseWhaleComponent"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_clamp_0_1", _analytics_args)
        except Exception:
            pass
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _safe_float(value: Any, *, default: float = 0.0) -> float:
        try:
            _analytics_class_name = "BaseWhaleComponent"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_safe_float", _analytics_args)
        except Exception:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, *, default: int = 0) -> int:
        try:
            _analytics_class_name = "BaseWhaleComponent"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_safe_int", _analytics_args)
        except Exception:
            pass
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


class BaseWhaleAnalyzerComponent(BaseWhaleComponent):
    """
    Семантичний базовий клас для фасадного analyzer-рівня.

    Фасад може не мати власних EventBus subscriptions, але все одно отримує
    event_bus/scheduler через dependency injection і керує дочірніми компонентами.
    """

    def __init__(
        self,
        *,
        component_name: str = "analyzer",
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        default_exchange: str = UNKNOWN_EXCHANGE,
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
        super().__init__(
            component_name=component_name,
            event_bus=event_bus,
            scheduler=scheduler,
            default_exchange=default_exchange,
            default_market_type=default_market_type,
            default_timeframe=default_timeframe,
        )

    async def register(self) -> None:
        """
        Analyzer facade за замовчуванням не підписується на EventBus напряму.
        Підписки виконують дочірні runtime-компоненти.
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
        if self._registered:
            return

        self._registered = True
        self.logger.info(
            "Whale analyzer facade registered",
            extra={"component": self.component_name},
        )


__all__ = [
    "BaseWhaleComponent",
    "BaseWhaleAnalyzerComponent",
    "EventHandler",
    "JobCallable",
    "RAW_WHALE_MARKET_TOPICS",
    "PRODUCTION_WHALE_MARKET_TOPICS",
]