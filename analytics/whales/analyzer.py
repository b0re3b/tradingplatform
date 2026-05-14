from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler

from analytics.whales.base import BaseWhaleAnalyzerComponent
from analytics.whales.config import WhalesConfig
from analytics.whales.enums import WhaleComponentName
from analytics.whales.large_trade_detector import LargeTradeDetector
from analytics.whales.whale_cluster_analyzer import WhaleClusterAnalyzer
from analytics.whales.whale_tracker import WhaleTracker


class WhaleAnalyzer(BaseWhaleAnalyzerComponent):
    """
    Фасад усього пакета analytics.whales.

    Відповідає тільки за orchestration між:
        - LargeTradeDetector
        - WhaleTracker
        - WhaleClusterAnalyzer

    Event-driven production flow:
        market.trade
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

    Додатковий liquidation flow:
        market.liquidation
            -> WhaleTracker
            -> analytics.whales.whale_liquidation_context
            -> WhaleClusterAnalyzer

    Core-правила:
        - EventBus/Scheduler передаються через constructor dependency injection;
        - фасад не містить власної торгової/аналітичної логіки;
        - фасад не дублює EventBus/Scheduler logic;
        - дочірні компоненти самі виконують register(), emit() і scheduler jobs;
        - direct API залишений для tests/backtesting/replay.
    """

    def __init__(
        self,
        *,
        config: WhalesConfig,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> None:
        super().__init__(
            component_name=WhaleComponentName.WHALE_ANALYZER.value,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        self.config = config
        self.config.validate()

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
    # Lifecycle
    # =========================================================================

    async def register(self) -> None:
        """
        Зареєструвати EventBus subscriptions дочірніх компонентів.

        Facade сам напряму не підписується на market/analytics topics.
        """
        if self._registered:
            return

        if not self.config.enabled:
            self.logger.info(
                "WhaleAnalyzer registration skipped: disabled by config",
                extra={"component": self.component_name},
            )
            return

        if self.config.auto_start_components:
            await self.large_trade_detector.register()
            await self.whale_tracker.register()
            await self.whale_cluster_analyzer.register()

        self._registered = True

        self.logger.info(
            "WhaleAnalyzer registered",
            extra={
                "component": self.component_name,
                "auto_start_components": self.config.auto_start_components,
            },
        )

    async def start(self) -> None:
        if self._started:
            self.logger.warning("WhaleAnalyzer already started")
            return

        if not self.config.enabled:
            self.logger.info("WhaleAnalyzer is disabled by config")
            return

        started_components: list[Any] = []

        try:
            if self.config.auto_start_components:
                await self.large_trade_detector.start()
                started_components.append(self.large_trade_detector)

                await self.whale_tracker.start()
                started_components.append(self.whale_tracker)

                await self.whale_cluster_analyzer.start()
                started_components.append(self.whale_cluster_analyzer)
            else:
                await self.register()

            self._started = True

            self.logger.info(
                "WhaleAnalyzer started",
                extra={
                    "component": self.component_name,
                    "auto_start_components": self.config.auto_start_components,
                    "large_trade_input_event": (
                        self.config.large_trade_detector.input_event_name
                    ),
                    "large_trade_output_event": (
                        self.config.large_trade_detector.output_event_name
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
                    "whale_cluster_event": (
                        self.config.whale_cluster_analyzer.whale_cluster_event_name
                    ),
                    "whale_cluster_update_event": (
                        self.config.whale_cluster_analyzer.whale_cluster_update_event_name
                    ),
                    "whale_cluster_exhaustion_event": (
                        self.config.whale_cluster_analyzer.whale_cluster_exhaustion_event_name
                    ),
                },
            )

        except Exception:
            self.logger.exception(
                "WhaleAnalyzer startup failed; rolling back started components",
                extra={"component": self.component_name},
            )

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

            self._started = False
            self._registered = False
            raise

    async def stop(self) -> None:
        children = (
            self.whale_cluster_analyzer,
            self.whale_tracker,
            self.large_trade_detector,
        )

        has_child_runtime_state = any(
            child.is_started or child.is_registered
            for child in children
        )

        if not self._started and not self._registered and not has_child_runtime_state:
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

        self.logger.info(
            "WhaleAnalyzer stopped",
            extra={"component": self.component_name},
        )

    # =========================================================================
    # Direct input API
    # =========================================================================

    async def process_trade(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> Any:
        """
        Прямий вхід raw trade payload у whale pipeline.

        У production market.trade має приходити через EventBus.
        Цей метод корисний для:
            - tests;
            - backtesting;
            - replay;
            - ручного прогону.
        """
        return await self.large_trade_detector.process_trade_payload(payload)

    async def process_liquidation(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> Any:
        """
        Прямий вхід raw liquidation payload у whale pipeline.
        """
        return await self.whale_tracker.process_liquidation_payload(payload)

    async def process_large_trade_signal(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> Any:
        """
        Прямий вхід уже готового large_trade signal payload у WhaleTracker.
        """
        return await self.whale_tracker.process_large_trade_payload(payload)

    async def process_whale_activity_signal(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> Any:
        """
        Прямий вхід whale_activity signal payload у WhaleClusterAnalyzer.
        """
        return await self.whale_cluster_analyzer.process_whale_activity_payload(payload)

    async def process_whale_pressure_signal(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> Any:
        """
        Прямий вхід whale_pressure signal payload у WhaleClusterAnalyzer.
        """
        return await self.whale_cluster_analyzer.process_whale_pressure_payload(payload)

    async def process_whale_liquidation_context_signal(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> Any:
        """
        Прямий вхід whale_liquidation_context signal payload у WhaleClusterAnalyzer.
        """
        return await self.whale_cluster_analyzer.process_whale_liquidation_context_payload(
            payload
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
            "large_trade_detector": self.large_trade_detector.get_all_stats(),
            "whale_tracker": self.whale_tracker.get_all_states(),
            "whale_cluster_analyzer": self.whale_cluster_analyzer.get_all_states(),
        }

    def get_symbol_stats(self, symbol: str) -> dict[str, Any]:
        normalized_symbol = self._normalize_symbol(symbol)

        if normalized_symbol is None:
            return {
                "symbol": symbol,
                "exists": False,
                "error": "invalid_symbol",
            }

        return {
            "symbol": normalized_symbol,
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

    async def reset_symbol(self, symbol: str) -> None:
        normalized_symbol = self._normalize_symbol(symbol)
        if normalized_symbol is None:
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
        scheduler: Scheduler,
    ) -> WhaleAnalyzer:
        return cls(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    @staticmethod
    def _normalize_symbol(value: Any) -> str | None:
        if value is None:
            return None

        symbol = str(value).strip()
        if not symbol:
            return None

        return symbol.upper()


__all__ = [
    "WhaleAnalyzer",
]