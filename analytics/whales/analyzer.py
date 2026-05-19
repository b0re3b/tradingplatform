from __future__ import annotations

from collections.abc import Mapping
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
        return (
            self.large_trade_detector,
            self.whale_tracker,
            self.whale_cluster_analyzer,
        )

    @property
    def _child_components_stop_order(self) -> tuple[Any, Any, Any]:
        return (
            self.whale_cluster_analyzer,
            self.whale_tracker,
            self.large_trade_detector,
        )

    def _children_registered(self) -> bool:
        return all(
            bool(getattr(component, "is_registered", False))
            for component in self._child_components
        )

    def _children_started(self) -> bool:
        return all(
            bool(getattr(component, "is_started", False))
            for component in self._child_components
        )

    def _has_child_runtime_state(self) -> bool:
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
        return bool(getattr(self.config, "allow_direct_raw_api", True))

    async def _register_child_components(self) -> None:
        """
        Зареєструвати EventBus subscriptions дочірніх компонентів.

        Важливо:
        - це не стартує scheduler jobs;
        - це тільки підписує pipeline на EventBus;
        - порядок відповідає data-flow pipeline.
        """
        for component in self._child_components:
            await component.register()

    async def _start_child_components(self) -> list[Any]:
        """
        Повністю стартує дочірні компоненти.

        Повертає список реально стартованих компонентів для rollback.
        """
        started_components: list[Any] = []

        for component in self._child_components:
            await component.start()
            started_components.append(component)

        return started_components

    async def _rollback_started_components(
        self,
        started_components: list[Any],
    ) -> None:
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
        await self.stop()

    # =========================================================================
    # Direct input API
    # =========================================================================

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
        return await self.whale_cluster_analyzer.process_whale_liquidation_context_payload(
            payload,
            source_topic="manual.direct.whale_liquidation_context_signal",
        )

    # =========================================================================
    # Health / stats / state
    # =========================================================================

    def get_healthcheck(self) -> dict[str, Any]:
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
        return cls(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )


__all__ = [
    "WhaleAnalyzer",
]