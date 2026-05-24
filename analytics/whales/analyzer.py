from __future__ import annotations
from core.logger import get_logger

from collections.abc import Iterable, Mapping
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler

from analytics.whales.base import BaseWhaleAnalyzerComponent
from analytics.whales.config import WhalesConfig
from analytics.whales.enums import WhaleComponentName
from analytics.whales.large_trade_detector import LargeTradeDetector
from analytics.whales.models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    UNKNOWN_EXCHANGE,
    WhaleKey,
    normalize_symbol,
    whale_key_to_dict,
)
from analytics.whales.whale_cluster_analyzer import WhaleClusterAnalyzer
from analytics.whales.whale_tracker import WhaleTracker


class WhaleAnalyzer(BaseWhaleAnalyzerComponent):
    """
    Facade всього пакета analytics.whales.

    Відповідає тільки за orchestration між:
        - LargeTradeDetector
        - WhaleTracker
        - WhaleClusterAnalyzer

    Correct production flow:
        exchange adapters
            -> market.trade
            -> TradesCache
            -> market.trades.updated
            -> LargeTradeDetector
            -> analytics.whales.large_trade
            -> WhaleTracker
            -> analytics.whales.whale_activity
            -> analytics.whales.whale_pressure
            -> analytics.whales.whale_liquidation_context
            -> WhaleClusterAnalyzer
            -> analytics.whales.whale_cluster
            -> analytics.whales.whale_cluster_update
            -> analytics.whales.whale_cluster_exhaustion

    Liquidation production flow:
        liquidation stream/cache
            -> market.liquidation
            -> LiquidationCache / liquidation analytics layer
            -> market.liquidations.updated або analytics.liquidations.*
            -> WhaleTracker
            -> analytics.whales.whale_liquidation_context
            -> WhaleClusterAnalyzer

    Scope:
        exchange + market_type + symbol + timeframe

    Core-правила:
        - EventBus/Scheduler передаються через constructor dependency injection;
        - Scheduler optional, але periodic cleanup працює тільки якщо він переданий;
        - facade не містить власної торгової/аналітичної логіки;
        - facade не дублює EventBus/Scheduler logic;
        - дочірні компоненти самі виконують register(), emit() і scheduler jobs;
        - facade не підписується напряму на market.* topics;
        - direct API залишений тільки для tests/backtesting/replay/manual path.

    Lifecycle semantics:
        - register() завжди реєструє EventBus subscriptions дочірніх компонентів;
        - start() з auto_start_components=True запускає дочірні компоненти повністю;
        - start() з auto_start_components=False тільки реєструє дочірні компоненти,
          але не стартує їхні scheduler jobs;
        - facade не переходить у стан started, якщо pipeline не зареєстрований.
    """

    def __init__(
        self,
        *,
        config: WhalesConfig,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
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
        config.validate()

        super().__init__(
            component_name=WhaleComponentName.WHALE_ANALYZER.value,
            event_bus=event_bus,
            scheduler=scheduler,
            default_exchange=(
                config.large_trade_detector.default_exchange
                if config.large_trade_detector.default_exchange
                else UNKNOWN_EXCHANGE
            ),
            default_market_type=(
                config.large_trade_detector.default_market_type
                if config.large_trade_detector.default_market_type
                else DEFAULT_MARKET_TYPE
            ),
            default_timeframe=(
                config.large_trade_detector.default_timeframe
                if config.large_trade_detector.default_timeframe
                else DEFAULT_TIMEFRAME
            ),
        )

        self.config = config

        self.large_trade_detector = LargeTradeDetector(
            config=self.config.large_trade_detector,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
        )
        self.whale_tracker = WhaleTracker(
            config=self.config.whale_tracker,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
        )
        self.whale_cluster_analyzer = WhaleClusterAnalyzer(
            config=self.config.whale_cluster_analyzer,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
        )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    @property
    def _child_components(self) -> tuple[Any, Any, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_child_components", _analytics_args)
        except Exception:
            pass
        return (
            self.large_trade_detector,
            self.whale_tracker,
            self.whale_cluster_analyzer,
        )

    @property
    def _child_components_stop_order(self) -> tuple[Any, Any, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_child_components_stop_order", _analytics_args)
        except Exception:
            pass
        return (
            self.whale_cluster_analyzer,
            self.whale_tracker,
            self.large_trade_detector,
        )

    def _children_registered(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_children_registered", _analytics_args)
        except Exception:
            pass
        return all(
            bool(getattr(component, "is_registered", False))
            for component in self._child_components
        )

    def _children_started(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_children_started", _analytics_args)
        except Exception:
            pass
        return all(
            bool(getattr(component, "is_started", False))
            for component in self._child_components
        )

    def _has_child_runtime_state(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_child_runtime_state", _analytics_args)
        except Exception:
            pass
        return any(
            bool(getattr(component, "is_started", False))
            or bool(getattr(component, "is_registered", False))
            for component in self._child_components
        )

    def _direct_api_enabled(self) -> bool:
        """
        Backward-compatible guard для manual/test/backtesting/replay API.

        Поточний WhalesConfig ще не має allow_direct_raw_api.
        Якщо поле буде додано в config — production може вимкнути direct raw path.
        За замовчуванням True, щоб не зламати існуючі тести/backtesting.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_direct_api_enabled", _analytics_args)
        except Exception:
            pass
        return bool(getattr(self.config, "allow_direct_raw_api", True))

    async def _register_child_components(self) -> None:
        """
        Зареєструвати EventBus subscriptions дочірніх компонентів.

        Важливо:
        - це не стартує scheduler jobs;
        - це тільки підписує pipeline на EventBus;
        - порядок відповідає data-flow pipeline.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_register_child_components", _analytics_args)
        except Exception:
            pass
        for component in self._child_components:
            await component.register()

    async def _start_child_components(self) -> list[Any]:
        """
        Повністю стартує дочірні компоненти.

        Повертає список реально стартованих компонентів для rollback.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_start_child_components", _analytics_args)
        except Exception:
            pass
        started_components: list[Any] = []

        for component in self._child_components:
            await component.start()
            started_components.append(component)

        return started_components

    async def _rollback_started_components(
        self,
        started_components: list[Any],
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_rollback_started_components", _analytics_args)
        except Exception:
            pass
        for component in reversed(started_components):
            try:
                await component.stop()
            except Exception:
                self.logger.exception(
                    "Failed to rollback whale component during startup failure",
                    extra={
                        "component": self.component_name,
                        "child_component": getattr(
                            component,
                            "component_name",
                            "unknown",
                        ),
                    },
                )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def register(self) -> None:
        """
        Зареєструвати EventBus subscriptions дочірніх компонентів.

        Facade сам напряму не підписується на market/analytics topics.

        Виправлена semantics:
        - register() більше не залежить від auto_start_components;
        - auto_start_components контролює start(), а не EventBus registration;
        - після register() pipeline уже слухає production topics.
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
        if self._registered and self._children_registered():
            return

        if not self.config.enabled:
            self.logger.info(
                "WhaleAnalyzer registration skipped: disabled by config",
                extra={"component": self.component_name},
            )
            return

        registered_components: list[Any] = []

        try:
            for component in self._child_components:
                if not getattr(component, "is_registered", False):
                    await component.register()
                registered_components.append(component)

            if not self._children_registered():
                raise RuntimeError(
                    "WhaleAnalyzer registration failed: not all child components "
                    "became registered"
                )

            self._registered = True

            self.logger.info(
                "WhaleAnalyzer registered",
                extra={
                    "component": self.component_name,
                    "auto_start_components": self.config.auto_start_components,
                    "children_registered": self._children_registered(),
                    "children_started": self._children_started(),
                    "production_input_topics": self.config.production_input_topics,
                    "legacy_raw_input_topics": self.config.legacy_raw_input_topics,
                    "scope": "exchange:market_type:symbol:timeframe",
                },
            )

        except Exception:
            self.logger.exception(
                "WhaleAnalyzer registration failed; rolling back registered components",
                extra={"component": self.component_name},
            )

            for component in reversed(registered_components):
                try:
                    if getattr(component, "is_registered", False):
                        await component.stop()
                except Exception:
                    self.logger.exception(
                        "Failed to rollback whale component during registration failure",
                        extra={
                            "component": self.component_name,
                            "child_component": getattr(
                                component,
                                "component_name",
                                "unknown",
                            ),
                        },
                    )

            self._registered = False
            self._started = False
            raise

    async def start(self) -> None:
        """
        Запустити WhaleAnalyzer.

        Якщо config.auto_start_components=True:
            - стартує всі дочірні компоненти;
            - дочірні компоненти самі роблять register();
            - scheduler cleanup jobs також запускаються.

        Якщо config.auto_start_components=False:
            - тільки реєструє дочірні компоненти через register();
            - EventBus pipeline працює;
            - scheduler jobs дочірніх компонентів не стартують;
            - це корисно для bootstrap/container режиму, де lifecycle керується зовні.
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
        if self._started:
            self.logger.warning("WhaleAnalyzer already started")
            return

        if not self.config.enabled:
            self.logger.info("WhaleAnalyzer is disabled by config")
            return

        started_components: list[Any] = []

        try:
            if self.config.auto_start_components:
                started_components = await self._start_child_components()

                if not self._children_started():
                    raise RuntimeError(
                        "WhaleAnalyzer startup failed: not all child components "
                        "became started"
                    )

                self._registered = self._children_registered()
                lifecycle_mode = "auto_started_components"

            else:
                await self.register()

                if not self._children_registered():
                    raise RuntimeError(
                        "WhaleAnalyzer startup failed: child components are not "
                        "registered while auto_start_components=False"
                    )

                lifecycle_mode = "registered_components_only"

                self.logger.warning(
                    "WhaleAnalyzer started in registration-only mode; "
                    "child Scheduler jobs are not started because "
                    "auto_start_components=False",
                    extra={
                        "component": self.component_name,
                        "auto_start_components": self.config.auto_start_components,
                        "children_registered": self._children_registered(),
                        "children_started": self._children_started(),
                    },
                )

            self._started = True

            self.logger.info(
                "WhaleAnalyzer started",
                extra={
                    "component": self.component_name,
                    "lifecycle_mode": lifecycle_mode,
                    "auto_start_components": self.config.auto_start_components,
                    "children_registered": self._children_registered(),
                    "children_started": self._children_started(),
                    "production_input_topics": self.config.production_input_topics,
                    "legacy_raw_input_topics": self.config.legacy_raw_input_topics,
                    "large_trade_input_topics": list(
                        self.config.large_trade_detector.production_input_topics
                    ),
                    "large_trade_legacy_raw_topics": list(
                        self.config.large_trade_detector.legacy_raw_input_topics
                    ),
                    "large_trade_output_event": (
                        self.config.large_trade_detector.output_event_name
                    ),
                    "whale_tracker_input_topics": list(
                        self.config.whale_tracker.production_input_topics
                    ),
                    "whale_tracker_legacy_raw_topics": list(
                        self.config.whale_tracker.legacy_raw_input_topics
                    ),
                    "whale_activity_event": (
                        self.config.whale_tracker.whale_activity_event_name
                    ),
                    "whale_pressure_event": (
                        self.config.whale_tracker.whale_pressure_event_name
                    ),
                    "whale_liquidation_context_event": (
                        self.config.whale_tracker.whale_liquidation_context_event_name
                    ),
                    "whale_cluster_input_topics": list(
                        self.config.whale_cluster_analyzer.production_input_topics
                    ),
                    "whale_cluster_event": (
                        self.config.whale_cluster_analyzer.whale_cluster_event_name
                    ),
                    "whale_cluster_update_event": (
                        self.config.whale_cluster_analyzer.whale_cluster_update_event_name
                    ),
                    "whale_cluster_exhaustion_event": (
                        self.config.whale_cluster_analyzer.whale_cluster_exhaustion_event_name
                    ),
                    "scope": "exchange:market_type:symbol:timeframe",
                },
            )

        except Exception:
            self.logger.exception(
                "WhaleAnalyzer startup failed; rolling back started components",
                extra={"component": self.component_name},
            )

            if started_components:
                await self._rollback_started_components(started_components)
            else:
                for component in reversed(self._child_components):
                    try:
                        if getattr(component, "is_registered", False):
                            await component.stop()
                    except Exception:
                        self.logger.exception(
                            "Failed to rollback whale component during startup failure",
                            extra={
                                "component": self.component_name,
                                "child_component": getattr(
                                    component,
                                    "component_name",
                                    "unknown",
                                ),
                            },
                        )

            self._started = False
            self._registered = False
            raise

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
        children = self._child_components_stop_order

        if not self._started and not self._registered and not self._has_child_runtime_state():
            return

        for child in children:
            try:
                await child.stop()
            except Exception:
                self.logger.exception(
                    "Failed to stop whale child component",
                    extra={
                        "component": self.component_name,
                        "child_component": getattr(
                            child,
                            "component_name",
                            "unknown",
                        ),
                    },
                )

        await super().stop()

        self._registered = False
        self._started = False

        self.logger.info(
            "WhaleAnalyzer stopped",
            extra={
                "component": self.component_name,
                "children_registered": self._children_registered(),
                "children_started": self._children_started(),
            },
        )

    async def shutdown(self) -> None:
        """
        Backward-compatible повний shutdown alias.
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
        await self.stop()

    # =========================================================================
    # Direct input API
    # =========================================================================

    async def process_market_snapshot(self, snapshot: Any) -> dict[str, Any]:
        """MarketScheduler-compatible whale pipeline input.

        Reads already-ingested trades/liquidations from MarketSnapshot and feeds
        the existing detector/tracker direct processing APIs.  This keeps raw WS
        data off EventBus while still allowing whales to run in the state-driven
        scheduler path.
        """
        scope = getattr(snapshot, "scope", None)
        exchange = getattr(scope, "exchange", None) or getattr(snapshot, "exchange", None) or self.default_exchange
        market_type = getattr(scope, "market_type", None) or getattr(snapshot, "market_type", None) or self.default_market_type
        symbol = getattr(scope, "symbol", None) or getattr(snapshot, "symbol", None)
        timeframe = getattr(scope, "timeframe", None) or getattr(snapshot, "timeframe", None) or self.default_timeframe
        exchange_symbol = getattr(scope, "exchange_symbol", None) or symbol

        trades = self._market_snapshot_items(snapshot, "trades", fallback_attr="recent_trades")
        liquidations = self._market_snapshot_items(snapshot, "liquidations")
        result: dict[str, Any] = {
            "trades_seen": len(trades),
            "liquidations_seen": len(liquidations),
            "large_trade_signals": 0,
            "liquidation_context_signals": 0,
        }

        if trades:
            trade_payload = {
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "exchange_symbol": exchange_symbol,
                "trades": [self._plain_market_item(item) for item in trades],
                "source_topic": "market_state.trades",
                "source": "market_state_store",
            }
            signals = await self.large_trade_detector.process_trades_payload(
                trade_payload,
                source_topic="market_state.trades",
                allow_raw_payload=True,
            )
            result["large_trade_signals"] = len(signals or [])
            # Feed generated large-trade signals into tracker directly as a
            # fallback for setups where EventBus loopback is disabled/throttled.
            for signal in signals or []:
                payload = self._plain_market_item(signal)
                if isinstance(payload, dict):
                    await self.whale_tracker.process_large_trade_payload(
                        payload,
                        source_topic="market_state.large_trade_detector",
                    )

        for item in liquidations:
            liq_payload = self._plain_market_item(item)
            if not isinstance(liq_payload, dict):
                continue
            metadata = dict(liq_payload.get("metadata") or {})
            liq_payload.setdefault("exchange", metadata.get("exchange") or exchange)
            liq_payload.setdefault("market_type", metadata.get("market_type") or market_type)
            liq_payload.setdefault("symbol", metadata.get("symbol") or symbol)
            liq_payload.setdefault("timeframe", metadata.get("timeframe") or timeframe)
            liq_payload.setdefault("exchange_symbol", metadata.get("exchange_symbol") or exchange_symbol)
            liq_payload.setdefault("source_topic", "market_state.liquidations")
            signal = await self.whale_tracker.process_liquidation_payload(
                liq_payload,
                source_topic="market_state.liquidations",
                allow_raw_payload=True,
            )
            if signal is not None:
                result["liquidation_context_signals"] += 1

        return result


    @staticmethod
    def _market_snapshot_items(snapshot: Any, attr: str, *, fallback_attr: str | None = None) -> list[Any]:
        """Return event items from MarketSnapshot-compatible containers.

        MarketSnapshot.trades is a TradesWindowSnapshot, not an iterable.  The
        actual iterable lives under `.trades`.  Other state/cache snapshots can
        expose `.items`, `.events`, `.liquidations`, `.data`, or a `to_dict()`
        payload.  This helper keeps the whale state-driven path compatible with
        both dataclass snapshots and plain dict payloads.
        """
        candidates: list[Any] = []
        if isinstance(snapshot, Mapping):
            candidates.append(snapshot.get(attr))
            if fallback_attr:
                candidates.append(snapshot.get(fallback_attr))
        else:
            candidates.append(getattr(snapshot, attr, None))
            if fallback_attr:
                candidates.append(getattr(snapshot, fallback_attr, None))

        for candidate in candidates:
            items = WhaleAnalyzer._coerce_market_items(candidate, preferred_attr=attr)
            if items:
                return items
        return []

    @staticmethod
    def _coerce_market_items(value: Any, *, preferred_attr: str | None = None) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            for key in (preferred_attr, "trades", "recent_trades", "liquidations", "items", "events", "data", "records"):
                if not key:
                    continue
                if key in value:
                    items = WhaleAnalyzer._coerce_market_items(value.get(key), preferred_attr=preferred_attr)
                    if items:
                        return items
            return [dict(value)]

        for key in (preferred_attr, "trades", "recent_trades", "liquidations", "items", "events", "data", "records"):
            if not key:
                continue
            nested = getattr(value, key, None)
            if nested is value:
                continue
            if nested is not None:
                items = WhaleAnalyzer._coerce_market_items(nested, preferred_attr=preferred_attr)
                if items:
                    return items

        if isinstance(value, (str, bytes, bytearray)):
            return []
        if isinstance(value, Iterable):
            return list(value)

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                mapped = to_dict()
            except Exception:
                mapped = None
            if mapped is not None and mapped is not value:
                items = WhaleAnalyzer._coerce_market_items(mapped, preferred_attr=preferred_attr)
                if items:
                    return items

        return [value]

    @staticmethod
    def _plain_market_item(item: Any) -> Any:
        if isinstance(item, Mapping):
            return dict(item)
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            try:
                value = to_dict()
                if isinstance(value, Mapping):
                    return dict(value)
                return value
            except Exception:
                pass
        return {
            "price": getattr(item, "price", None),
            "quantity": getattr(item, "quantity", getattr(item, "qty", None)),
            "qty": getattr(item, "qty", getattr(item, "quantity", None)),
            "side": getattr(item, "side", None),
            "timestamp_ms": getattr(item, "timestamp_ms", None),
            "trade_id": getattr(item, "trade_id", None),
            "order_id": getattr(item, "order_id", None),
            "metadata": dict(getattr(item, "metadata", {}) or {}),
        }

    async def process_trade(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> Any:
        """
        Прямий manual/test/backtesting/replay вхід trade payload.

        У production trade payload має йти так:
            market.trade -> TradesCache -> market.trades.updated -> LargeTradeDetector

        Цей direct API обходить EventBus source topic guard, тому має бути
        дозволений тільки для tests/backtesting/replay/manual path.

        Якщо у WhalesConfig буде додано allow_direct_raw_api=False,
        цей метод почне блокувати прямий raw path у production.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_trade", _analytics_args)
        except Exception:
            pass
        if not self._direct_api_enabled():
            raise RuntimeError(
                "WhaleAnalyzer direct trade API is disabled by config. "
                "Use production EventBus flow: market.trade -> TradesCache "
                "-> market.trades.updated -> LargeTradeDetector."
            )

        process_trades_payload = getattr(
            self.large_trade_detector,
            "process_trades_payload",
            None,
        )

        if callable(process_trades_payload):
            return await process_trades_payload(
                payload,
                source_topic="manual.direct.trade",
                allow_raw_payload=True,
            )

        return await self.large_trade_detector.process_trade_payload(
            payload,
            source_topic="manual.direct.trade",
            allow_raw_payload=True,
        )

    async def process_liquidation(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> Any:
        """
        Прямий manual/test/backtesting/replay вхід liquidation payload.

        У production liquidation payload має йти через:
            market.liquidation -> liquidation cache/layer
            -> market.liquidations.updated / analytics.liquidations.*
            -> WhaleTracker
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_liquidation", _analytics_args)
        except Exception:
            pass
        if not self._direct_api_enabled():
            raise RuntimeError(
                "WhaleAnalyzer direct liquidation API is disabled by config. "
                "Use production EventBus flow: market.liquidation -> "
                "market.liquidations.updated / analytics.liquidations.* "
                "-> WhaleTracker."
            )

        return await self.whale_tracker.process_liquidation_payload(
            payload,
            source_topic="manual.direct.liquidation",
            allow_raw_payload=True,
        )

    async def process_large_trade_signal(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> Any:
        """
        Прямий вхід уже готового large_trade signal payload у WhaleTracker.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_large_trade_signal", _analytics_args)
        except Exception:
            pass
        return await self.whale_tracker.process_large_trade_payload(
            payload,
            source_topic="manual.direct.large_trade_signal",
        )

    async def process_whale_activity_signal(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> Any:
        """
        Прямий вхід whale_activity signal payload у WhaleClusterAnalyzer.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_whale_activity_signal", _analytics_args)
        except Exception:
            pass
        return await self.whale_cluster_analyzer.process_whale_activity_payload(
            payload,
            source_topic="manual.direct.whale_activity_signal",
        )

    async def process_whale_pressure_signal(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> Any:
        """
        Прямий вхід whale_pressure signal payload у WhaleClusterAnalyzer.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_whale_pressure_signal", _analytics_args)
        except Exception:
            pass
        return await self.whale_cluster_analyzer.process_whale_pressure_payload(
            payload,
            source_topic="manual.direct.whale_pressure_signal",
        )

    async def process_whale_liquidation_context_signal(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> Any:
        """
        Прямий вхід whale_liquidation_context signal payload у WhaleClusterAnalyzer.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_whale_liquidation_context_signal", _analytics_args)
        except Exception:
            pass
        return await self.whale_cluster_analyzer.process_whale_liquidation_context_payload(
            payload,
            source_topic="manual.direct.whale_liquidation_context_signal",
        )

    # =========================================================================
    # Health / stats / state
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
        base_health = super().get_healthcheck()

        base_health.update(
            {
                "enabled": self.config.enabled,
                "auto_start_components": self.config.auto_start_components,
                "direct_api_enabled": self._direct_api_enabled(),
                "children_registered": self._children_registered(),
                "children_started": self._children_started(),
                "lifecycle_mode": (
                    "auto_started_components"
                    if self.config.auto_start_components
                    else "registered_components_only"
                ),
                "scope": "exchange:market_type:symbol:timeframe",
                "production_input_topics": self.config.production_input_topics,
                "legacy_raw_input_topics": self.config.legacy_raw_input_topics,
                "pipeline_topics": {
                    "large_trade_input": list(
                        self.config.large_trade_detector.production_input_topics
                    ),
                    "large_trade_output": (
                        self.config.large_trade_detector.output_event_name
                    ),
                    "whale_tracker_input": list(
                        self.config.whale_tracker.production_input_topics
                    ),
                    "whale_activity_output": (
                        self.config.whale_tracker.whale_activity_event_name
                    ),
                    "whale_pressure_output": (
                        self.config.whale_tracker.whale_pressure_event_name
                    ),
                    "whale_liquidation_context_output": (
                        self.config.whale_tracker.whale_liquidation_context_event_name
                    ),
                    "whale_cluster_input": list(
                        self.config.whale_cluster_analyzer.production_input_topics
                    ),
                    "whale_cluster_output": (
                        self.config.whale_cluster_analyzer.whale_cluster_event_name
                    ),
                    "whale_cluster_update_output": (
                        self.config.whale_cluster_analyzer.whale_cluster_update_event_name
                    ),
                    "whale_cluster_exhaustion_output": (
                        self.config.whale_cluster_analyzer.whale_cluster_exhaustion_event_name
                    ),
                },
                "components": {
                    "large_trade_detector": (
                        self.large_trade_detector.get_healthcheck()
                    ),
                    "whale_tracker": self.whale_tracker.get_healthcheck(),
                    "whale_cluster_analyzer": (
                        self.whale_cluster_analyzer.get_healthcheck()
                    ),
                },
            }
        )
        return base_health

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
            "analyzer_started": self._started,
            "analyzer_registered": self._registered,
            "enabled": self.config.enabled,
            "auto_start_components": self.config.auto_start_components,
            "direct_api_enabled": self._direct_api_enabled(),
            "children_registered": self._children_registered(),
            "children_started": self._children_started(),
            "scope": "exchange:market_type:symbol:timeframe",
            "production_input_topics": self.config.production_input_topics,
            "legacy_raw_input_topics": self.config.legacy_raw_input_topics,
            "large_trade_detector": self.large_trade_detector.get_all_stats(),
            "whale_tracker": self.whale_tracker.get_all_states(),
            "whale_cluster_analyzer": self.whale_cluster_analyzer.get_all_states(),
        }

    def get_key_stats(self, key: WhaleKey) -> dict[str, Any]:
        """
        Scoped read API для одного WhaleKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_key_stats", _analytics_args)
        except Exception:
            pass
        scope = whale_key_to_dict(key)

        return {
            **scope,
            "scope": scope,
            "large_trade_detector": self.large_trade_detector.get_key_stats(key),
            "whale_tracker": self.whale_tracker.get_key_state(key),
            "whale_cluster_analyzer": self.whale_cluster_analyzer.get_key_state(key),
        }

    def get_symbol_stats(
        self,
        symbol: str,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> dict[str, Any]:
        """
        Backward-compatible read API.

        Якщо exchange/market_type/timeframe передані — повертає scoped stats.
        Якщо ні — повертає всі scopes для symbol з кожного дочірнього компонента.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_symbol_stats", _analytics_args)
        except Exception:
            pass
        try:
            normalized_symbol = normalize_symbol(symbol)
        except ValueError:
            return {
                "symbol": symbol,
                "exists": False,
                "error": "invalid_symbol",
            }

        if exchange is not None or market_type is not None or timeframe is not None:
            key = self.make_key(
                exchange=exchange or self.default_exchange,
                market_type=market_type or self.default_market_type,
                symbol=normalized_symbol,
                timeframe=timeframe or self.default_timeframe,
            )
            return self.get_key_stats(key)

        return {
            "symbol": normalized_symbol,
            "scope": "symbol-only aggregate",
            "large_trade_detector": (
                self.large_trade_detector.get_symbol_stats(normalized_symbol)
            ),
            "whale_tracker": self.whale_tracker.get_symbol_state(normalized_symbol),
            "whale_cluster_analyzer": (
                self.whale_cluster_analyzer.get_symbol_state(normalized_symbol)
            ),
        }

    # =========================================================================
    # Reset API
    # =========================================================================

    async def reset_key(self, key: WhaleKey) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "reset_key", _analytics_args)
        except Exception:
            pass
        await self.large_trade_detector.reset_key(key)
        await self.whale_tracker.reset_key(key)
        await self.whale_cluster_analyzer.reset_key(key)

        self.logger.info(
            "Reset scoped state in WhaleAnalyzer",
            extra={
                "component": self.component_name,
                "scope": whale_key_to_dict(key),
            },
        )

    async def reset_symbol(
        self,
        symbol: str,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> None:
        """
        Backward-compatible reset API.

        Якщо exchange/market_type/timeframe передані — reset одного WhaleKey.
        Якщо ні — reset усіх scope-ів для symbol.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "reset_symbol", _analytics_args)
        except Exception:
            pass
        try:
            normalized_symbol = normalize_symbol(symbol)
        except ValueError:
            return

        if exchange is not None or market_type is not None or timeframe is not None:
            key = self.make_key(
                exchange=exchange or self.default_exchange,
                market_type=market_type or self.default_market_type,
                symbol=normalized_symbol,
                timeframe=timeframe or self.default_timeframe,
            )
            await self.reset_key(key)
            return

        await self.large_trade_detector.reset_symbol(normalized_symbol)
        await self.whale_tracker.reset_symbol(normalized_symbol)
        await self.whale_cluster_analyzer.reset_symbol(normalized_symbol)

        self.logger.info(
            "Reset symbol state in WhaleAnalyzer",
            extra={
                "component": self.component_name,
                "symbol": normalized_symbol,
            },
        )

    async def reset_all(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "reset_all", _analytics_args)
        except Exception:
            pass
        await self.large_trade_detector.reset_all()
        await self.whale_tracker.reset_all()
        await self.whale_cluster_analyzer.reset_all()

        self.logger.info(
            "Reset all WhaleAnalyzer states",
            extra={"component": self.component_name},
        )

    # =========================================================================
    # Component accessors
    # =========================================================================

    def get_components(self) -> dict[str, Any]:
        """
        Повертає прямий доступ до компонентів whale pipeline.

        Корисно для bootstrap/container інтеграції та тестів.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_components", _analytics_args)
        except Exception:
            pass
        return {
            "large_trade_detector": self.large_trade_detector,
            "whale_tracker": self.whale_tracker,
            "whale_cluster_analyzer": self.whale_cluster_analyzer,
        }

    # =========================================================================
    # Configuration helpers
    # =========================================================================

    @classmethod
    def from_config(
        cls,
        config: WhalesConfig,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
    ) -> WhaleAnalyzer:
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "WhaleAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_config", _analytics_args)
        except Exception:
            pass
        return cls(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )


__all__ = [
    "WhaleAnalyzer",
]