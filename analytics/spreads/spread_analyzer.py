from __future__ import annotations

from typing import Any

from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import CrossExchangeSpreadConfig, SpotFuturesSpreadConfig
from .cross_exchange_analyzer import CrossExchangeSpreadAnalyzer
from .enums import InstrumentType
from .models import (
    ArbitrageOpportunity,
    SpreadKey,
    SpreadSnapshot,
)
from .spot_futures_analyzer import SpotFuturesSpreadAnalyzer


class SpreadAnalyzer:
    """
    Production-grade facade для analytics.spreads.

    Відповідальність:
    - агрегує SpotFuturesSpreadAnalyzer і CrossExchangeSpreadAnalyzer;
    - передає EventBus / Scheduler через constructor dependency injection;
    - керує register/start/stop/shutdown lifecycle;
    - дозволяє вмикати analyzer-и окремо;
    - надає read-only facade API для latest snapshots/opportunities.

    Correct production input flow:
        exchange adapters / REST warmup / parquet restore
            -> MarketIngestionService
            -> MarketStateStore dirty scopes
            -> MarketScheduler
            -> SpreadAnalyzer.process_market_snapshot() або child analyzer evaluator-и
            -> analytics.spreads.*

    Price/funding data:
        MarketStateStore snapshot
            -> QuoteSnapshot/FundingSnapshot як внутрішні normalized моделі
            -> SpotFuturesSpreadAnalyzer / CrossExchangeSpreadAnalyzer

    Важливо:
    - facade не отримує market data напряму;
    - facade не створює exchange adapters;
    - facade не містить spread business logic;
    - facade не викликає strategy/risk/execution напряму;
    - QuoteCache не використовується і не потрібен;
    - SpotFuturesSpreadAnalyzer залишається production-компонентом;
    - CrossExchangeSpreadAnalyzer залишається production-компонентом.
    """

    PRICE_INPUT_SOURCE = "market_state_snapshot"
    FUNDING_INPUT_SOURCE = "market_state_snapshot"

    def __init__(
        self,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        market_scheduler: Any | None = None,
        spot_futures_config: SpotFuturesSpreadConfig | None = None,
        cross_exchange_config: CrossExchangeSpreadConfig | None = None,
        enable_spot_futures: bool = True,
        enable_cross_exchange: bool = True,
        auto_register: bool = False,
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
        self._market_scheduler = market_scheduler

        self._spot_futures_config = spot_futures_config or SpotFuturesSpreadConfig()
        self._cross_exchange_config = cross_exchange_config or CrossExchangeSpreadConfig()

        self._enable_spot_futures = bool(enable_spot_futures)
        self._enable_cross_exchange = bool(enable_cross_exchange)

        self._logger = get_logger(
            __name__,
            service_name="spread_analyzer",
            event_type="spreads_facade",
        )

        self._spot_futures_analyzer: SpotFuturesSpreadAnalyzer | None = None
        self._cross_exchange_analyzer: CrossExchangeSpreadAnalyzer | None = None

        if self._enable_spot_futures:
            self._spot_futures_analyzer = SpotFuturesSpreadAnalyzer(
                config=self._spot_futures_config,
                event_bus=self._event_bus,
                scheduler=self._scheduler,
                market_scheduler=self._market_scheduler,
            )

        if self._enable_cross_exchange:
            self._cross_exchange_analyzer = CrossExchangeSpreadAnalyzer(
                config=self._cross_exchange_config,
                event_bus=self._event_bus,
                scheduler=self._scheduler,
                market_scheduler=self._market_scheduler,
            )

        self._running = False
        self._registered = False

        if auto_register:
            self.register()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    @property
    def spot_futures_enabled(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "spot_futures_enabled", _analytics_args)
        except Exception:
            pass
        return self._spot_futures_analyzer is not None

    @property
    def cross_exchange_enabled(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "cross_exchange_enabled", _analytics_args)
        except Exception:
            pass
        return self._cross_exchange_analyzer is not None

    @property
    def spot_futures(self) -> SpotFuturesSpreadAnalyzer:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "spot_futures", _analytics_args)
        except Exception:
            pass
        if self._spot_futures_analyzer is None:
            raise RuntimeError("SpotFuturesSpreadAnalyzer is disabled")
        return self._spot_futures_analyzer

    @property
    def cross_exchange(self) -> CrossExchangeSpreadAnalyzer:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "cross_exchange", _analytics_args)
        except Exception:
            pass
        if self._cross_exchange_analyzer is None:
            raise RuntimeError("CrossExchangeSpreadAnalyzer is disabled")
        return self._cross_exchange_analyzer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Реєструє enabled analyzer-и в EventBus.

        Production subscriptions створюються всередині analyzer-ів:
        - SpotFuturesSpreadAnalyzer:
            market.orderbook.updated / market.funding.updated
        - CrossExchangeSpreadAnalyzer:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "register", _analytics_args)
        except Exception:
            pass
        if self._registered:
            self._logger.warning("SpreadAnalyzer already registered")
            return

        registered_components: list[str] = []

        if self._spot_futures_analyzer is not None:
            self._spot_futures_analyzer.register()
            registered_components.append("spot_futures")

        if self._cross_exchange_analyzer is not None:
            self._cross_exchange_analyzer.register()
            registered_components.append("cross_exchange")

        self._registered = True

        self._logger.info(
            "SpreadAnalyzer registered | components=%s "
            "spot_futures_registered=%s cross_exchange_registered=%s",
            registered_components,
            (
                self._spot_futures_analyzer.is_registered
                if self._spot_futures_analyzer is not None
                else False
            ),
            (
                self._cross_exchange_analyzer.is_registered
                if self._cross_exchange_analyzer is not None
                else False
            ),
            extra={
                "components": registered_components,
                "scope": "exchange:market_type:symbol:timeframe",
                "spot_futures_enabled": self._spot_futures_analyzer is not None,
                "cross_exchange_enabled": self._cross_exchange_analyzer is not None,
                "price_input_source": self.PRICE_INPUT_SOURCE,
                "funding_input_source": self.FUNDING_INPUT_SOURCE,
                "uses_quote_cache": False,
            },
        )

    def unregister(self) -> None:
        """
        Відписує enabled analyzer-и від EventBus.

        Рекомендований порядок:
            await stop()
            unregister()
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
        if self._running:
            self._logger.warning(
                "SpreadAnalyzer unregister requested while running; "
                "stop() should be called first"
            )

        if not self._registered:
            self._logger.warning("SpreadAnalyzer already unregistered")
            return

        if self._cross_exchange_analyzer is not None:
            self._cross_exchange_analyzer.unregister()

        if self._spot_futures_analyzer is not None:
            self._spot_futures_analyzer.unregister()

        self._registered = False

        self._logger.info(
            "SpreadAnalyzer unregistered",
            extra={
                "price_input_source": self.PRICE_INPUT_SOURCE,
                "funding_input_source": self.FUNDING_INPUT_SOURCE,
                "uses_quote_cache": False,
            },
        )

    async def start(self) -> None:
        """
        Запускає enabled analyzer-и.

        Якщо register() ще не викликаний — викликає його автоматично.
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
            self._logger.warning("SpreadAnalyzer already started")
            return

        if not self._registered:
            self.register()

        started_components: list[str] = []

        if self._spot_futures_analyzer is not None:
            await self._spot_futures_analyzer.start()
            started_components.append("spot_futures")

        if self._cross_exchange_analyzer is not None:
            await self._cross_exchange_analyzer.start()
            started_components.append("cross_exchange")

        self._running = True

        self._logger.info(
            "SpreadAnalyzer started | components=%s "
            "spot_futures_enabled=%s cross_exchange_enabled=%s",
            started_components,
            (
                self._spot_futures_config.enabled
                if self._spot_futures_analyzer is not None
                else False
            ),
            (
                self._cross_exchange_config.enabled
                if self._cross_exchange_analyzer is not None
                else False
            ),
            extra={
                "components": started_components,
                "scope": "exchange:market_type:symbol:timeframe",
                "price_input_source": self.PRICE_INPUT_SOURCE,
                "funding_input_source": self.FUNDING_INPUT_SOURCE,
                "uses_quote_cache": False,
                "production_flow": (
                    "OrderBookCache/FundingCache -> "
                    "market.orderbook.updated/market.funding.updated -> "
                    "analytics.spreads"
                ),
            },
        )

    async def stop(self) -> None:
        """
        Зупиняє enabled analyzer-и.

        Порядок зупинки зворотний до старту.
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
            self._logger.warning("SpreadAnalyzer already stopped")
            return

        if self._cross_exchange_analyzer is not None:
            await self._cross_exchange_analyzer.stop()

        if self._spot_futures_analyzer is not None:
            await self._spot_futures_analyzer.stop()

        self._running = False

        self._logger.info(
            "SpreadAnalyzer stopped",
            extra={
                "price_input_source": self.PRICE_INPUT_SOURCE,
                "funding_input_source": self.FUNDING_INPUT_SOURCE,
                "uses_quote_cache": False,
            },
        )

    async def shutdown(self) -> None:
        """
        Повний shutdown:
        - stop(), якщо running;
        - unregister(), якщо registered.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "shutdown", _analytics_args)
        except Exception:
            pass
        if self._running:
            await self.stop()

        if self._registered:
            self.unregister()

        self._logger.info(
            "SpreadAnalyzer shutdown completed",
            extra={
                "price_input_source": self.PRICE_INPUT_SOURCE,
                "funding_input_source": self.FUNDING_INPUT_SOURCE,
                "uses_quote_cache": False,
            },
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------


    async def process_market_snapshot(self, snapshot: Any) -> dict[str, Any]:
        """
        State-driven MarketScheduler entrypoint for the spreads facade.

        This method must be compatible with both the current facade attributes
        (``_spot_futures_analyzer`` / ``_cross_exchange_analyzer``) and older
        hotfix attribute names.  It deliberately uses ``getattr`` so a disabled
        child analyzer does not crash the whole MarketScheduler evaluation.
        """
        result: dict[str, Any] = {
            "processed": False,
            "spot_futures": None,
            "cross_exchange": None,
            "skipped": {},
        }

        async def _call_child(name: str, child: Any) -> dict[str, Any]:
            if child is None:
                return {"processed": False, "reason": f"{name}_child_not_configured"}

            callback = getattr(child, "process_market_snapshot", None)
            if not callable(callback):
                return {"processed": False, "reason": f"{name}_child_missing_process_market_snapshot"}

            child_result = await callback(snapshot)
            if isinstance(child_result, dict):
                return child_result
            return {"processed": bool(child_result), "result": child_result}

        spot_futures = (
            getattr(self, "_spot_futures_analyzer", None)
            or getattr(self, "_spot_futures", None)
        )
        cross_exchange = (
            getattr(self, "_cross_exchange_analyzer", None)
            or getattr(self, "_cross_exchange", None)
        )

        result["spot_futures"] = await _call_child("spot_futures", spot_futures)
        if result["spot_futures"].get("processed"):
            result["processed"] = True
        else:
            result["skipped"]["spot_futures"] = result["spot_futures"].get("reason")

        result["cross_exchange"] = await _call_child("cross_exchange", cross_exchange)
        if result["cross_exchange"].get("processed"):
            result["processed"] = True
        else:
            result["skipped"]["cross_exchange"] = result["cross_exchange"].get("reason")

        if not result["processed"]:
            result["reason"] = "no_child_analyzer_processed_snapshot"

        return result

    def get_stats(self) -> dict[str, Any]:
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
        return {
            "running": self._running,
            "registered": self._registered,
            "scope": "exchange:market_type:symbol:timeframe",
            "price_input_source": self.PRICE_INPUT_SOURCE,
            "funding_input_source": self.FUNDING_INPUT_SOURCE,
            "uses_quote_cache": False,
            "production_flow": {
                "price": (
                    "exchange adapters -> market.orderbook -> "
                    "OrderBookCache -> market.orderbook.updated -> spreads"
                ),
                "funding": (
                    "exchange adapters -> market.funding -> "
                    "FundingCache -> market.funding.updated -> spreads"
                ),
            },
            "enabled_components": {
                "spot_futures": self._spot_futures_analyzer is not None,
                "cross_exchange": self._cross_exchange_analyzer is not None,
            },
            "configs": {
                "spot_futures_enabled": (
                    self._spot_futures_config.enabled
                    if self._spot_futures_analyzer is not None
                    else False
                ),
                "cross_exchange_enabled": (
                    self._cross_exchange_config.enabled
                    if self._cross_exchange_analyzer is not None
                    else False
                ),
                "spot_futures_topics": (
                    list(self._spot_futures_config.production_input_topics)
                    if self._spot_futures_analyzer is not None
                    else []
                ),
                "spot_futures_price_topics": (
                    list(self._spot_futures_config.production_price_input_topics)
                    if self._spot_futures_analyzer is not None
                    else []
                ),
                "cross_exchange_topics": (
                    list(self._cross_exchange_config.production_input_topics)
                    if self._cross_exchange_analyzer is not None
                    else []
                ),
                "cross_exchange_price_topics": (
                    list(self._cross_exchange_config.production_price_input_topics)
                    if self._cross_exchange_analyzer is not None
                    else []
                ),
            },
            "spot_futures": (
                self._spot_futures_analyzer.get_stats()
                if self._spot_futures_analyzer is not None
                else None
            ),
            "cross_exchange": (
                self._cross_exchange_analyzer.get_stats()
                if self._cross_exchange_analyzer is not None
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Read API: spot/futures
    # ------------------------------------------------------------------

    def get_latest_spot_futures_snapshot(
        self,
        symbol: str,
        spot_exchange: str,
        futures_exchange: str,
        *,
        spot_market_type: str | None = None,
        futures_market_type: str | None = None,
        timeframe: str | None = None,
    ) -> SpreadSnapshot | None:
        """
        Повертає latest spot/futures snapshot для одного scoped pair.

        За замовчуванням market types беруться з SpotFuturesSpreadConfig:
        - default_spot_market_type
        - default_futures_market_type
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_latest_spot_futures_snapshot", _analytics_args)
        except Exception:
            pass
        if self._spot_futures_analyzer is None:
            return None

        return self._spot_futures_analyzer.get_latest_snapshot(
            symbol=symbol,
            spot_exchange=spot_exchange,
            futures_exchange=futures_exchange,
            spot_market_type=spot_market_type,
            futures_market_type=futures_market_type,
            timeframe=timeframe,
        )

    def get_latest_spot_futures_snapshot_by_keys(
        self,
        *,
        spot_key: SpreadKey,
        futures_key: SpreadKey,
    ) -> SpreadSnapshot | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_latest_spot_futures_snapshot_by_keys", _analytics_args)
        except Exception:
            pass
        if self._spot_futures_analyzer is None:
            return None

        return self._spot_futures_analyzer.get_latest_snapshot_by_keys(
            spot_key=spot_key,
            futures_key=futures_key,
        )

    # ------------------------------------------------------------------
    # Read API: cross-exchange
    # ------------------------------------------------------------------

    def get_latest_cross_exchange_snapshot(
        self,
        symbol: str,
        exchange_a: str,
        exchange_b: str,
        instrument_type: InstrumentType,
        *,
        market_type: str,
        timeframe: str | None = None,
    ) -> SpreadSnapshot | None:
        """
        Повертає latest cross-exchange snapshot для одного scoped market.

        Важливо:
        cross-exchange analyzer порівнює однаковий
        symbol + instrument_type + market_type + timeframe між різними біржами.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_latest_cross_exchange_snapshot", _analytics_args)
        except Exception:
            pass
        if self._cross_exchange_analyzer is None:
            return None

        return self._cross_exchange_analyzer.get_latest_snapshot(
            symbol=symbol,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            instrument_type=instrument_type,
            market_type=market_type,
            timeframe=timeframe,
        )

    def get_latest_cross_exchange_snapshot_by_keys(
        self,
        *,
        quote_a_key: SpreadKey,
        quote_b_key: SpreadKey,
    ) -> SpreadSnapshot | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_latest_cross_exchange_snapshot_by_keys", _analytics_args)
        except Exception:
            pass
        if self._cross_exchange_analyzer is None:
            return None

        return self._cross_exchange_analyzer.get_latest_snapshot_by_keys(
            quote_a_key=quote_a_key,
            quote_b_key=quote_b_key,
        )

    def get_best_cross_exchange_opportunities(
        self,
        symbol: str | None = None,
        instrument_type: InstrumentType | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        profitable_only: bool = True,
        active_only: bool = True,
        limit: int | None = None,
    ) -> list[ArbitrageOpportunity]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_best_cross_exchange_opportunities", _analytics_args)
        except Exception:
            pass
        if self._cross_exchange_analyzer is None:
            return []

        return self._cross_exchange_analyzer.get_best_opportunities(
            symbol=symbol,
            instrument_type=instrument_type,
            market_type=market_type,
            timeframe=timeframe,
            profitable_only=profitable_only,
            active_only=active_only,
            limit=limit,
        )