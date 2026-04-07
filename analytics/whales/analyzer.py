from __future__ import annotations

from typing import Any, Dict, Optional

from analytics.whales.base import BaseWhaleAnalyzerComponent
from analytics.whales.config import WhalesConfig
from analytics.whales.large_trade_detector import LargeTradeDetector
from analytics.whales.whale_cluster_analyzer import WhaleClusterAnalyzer
from analytics.whales.whale_tracker import WhaleTracker


class WhaleAnalyzer(BaseWhaleAnalyzerComponent):
    """
    Фасад усього пакета analytics.whales.

    Відповідає за orchestration між:
        - LargeTradeDetector
        - WhaleTracker
        - WhaleClusterAnalyzer

    Базовий потік даних:
        market.trade
            -> LargeTradeDetector
            -> analytics.whales.large_trade
            -> WhaleTracker
            -> analytics.whales.whale_activity / whale_pressure / whale_liquidation_context
            -> WhaleClusterAnalyzer
            -> analytics.whales.whale_cluster / whale_cluster_update / whale_cluster_exhaustion

    Додатково:
        market.liquidation
            -> WhaleTracker
            -> analytics.whales.whale_liquidation_context
            -> WhaleClusterAnalyzer
    """

    def __init__(
        self,
        config: Optional[WhalesConfig] = None,
        event_bus: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        super().__init__(
            component_name="analyzer",
            event_bus=event_bus,
            scheduler=scheduler,
        )

        self.config = config or WhalesConfig()
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

    async def start(self) -> None:
        if self._started:
            self.logger.warning("WhaleAnalyzer already started")
            return

        if not self.config.enabled:
            self.logger.info("WhaleAnalyzer is disabled by config")
            return

        if self.config.auto_start_components:
            await self.large_trade_detector.start()
            await self.whale_tracker.start()
            await self.whale_cluster_analyzer.start()

        self._started = True

        self.logger.info(
            "WhaleAnalyzer started",
            extra={
                "auto_start_components": self.config.auto_start_components,
                "large_trade_input_event": self.config.large_trade_detector.input_event_name,
                "large_trade_output_event": self.config.large_trade_detector.output_event_name,
                "whale_activity_event": self.config.whale_tracker.whale_activity_event_name,
                "whale_pressure_event": self.config.whale_tracker.whale_pressure_event_name,
                "whale_liquidation_context_event": self.config.whale_tracker.whale_liquidation_context_event_name,
                "whale_cluster_event": self.config.whale_cluster_analyzer.whale_cluster_event_name,
                "whale_cluster_update_event": self.config.whale_cluster_analyzer.whale_cluster_update_event_name,
                "whale_cluster_exhaustion_event": self.config.whale_cluster_analyzer.whale_cluster_exhaustion_event_name,
            },
        )

    async def stop(self) -> None:
        if not self._started:
            return

        await self.whale_cluster_analyzer.stop()
        await self.whale_tracker.stop()
        await self.large_trade_detector.stop()

        self._started = False
        self.logger.info("WhaleAnalyzer stopped")

    # =========================================================================
    # Direct input API
    # =========================================================================

    async def process_trade(self, event: Dict[str, Any]) -> Optional[Any]:
        """
        Прямий вхід сирого trade event у whale pipeline.

        Зазвичай у production raw trade буде приходити через EventBus
        у LargeTradeDetector, але цей метод корисний для:
            - тестів
            - backtesting
            - replay
            - ручного прогону
        """
        return await self.large_trade_detector.process_trade(event)

    async def process_liquidation(self, event: Dict[str, Any]) -> Optional[Any]:
        """
        Прямий вхід raw liquidation event у whale pipeline.
        """
        return await self.whale_tracker.process_liquidation_event(event)

    async def process_large_trade_signal(self, event: Dict[str, Any]) -> Any:
        """
        Прямий вхід уже готового large_trade signal у WhaleTracker.
        Корисно для тестів або replay вже нормалізованих подій.
        """
        return await self.whale_tracker.process_large_trade_event(event)

    async def process_whale_activity_signal(self, event: Dict[str, Any]) -> Any:
        """
        Прямий вхід whale_activity signal у WhaleClusterAnalyzer.
        """
        return await self.whale_cluster_analyzer.process_whale_activity_event(event)

    async def process_whale_pressure_signal(self, event: Dict[str, Any]) -> Any:
        """
        Прямий вхід whale_pressure signal у WhaleClusterAnalyzer.
        """
        return await self.whale_cluster_analyzer.process_whale_pressure_event(event)

    async def process_whale_liquidation_context_signal(self, event: Dict[str, Any]) -> Any:
        """
        Прямий вхід whale_liquidation_context signal у WhaleClusterAnalyzer.
        """
        return await self.whale_cluster_analyzer.process_whale_liquidation_context_event(event)

    # =========================================================================
    # Health / stats / state
    # =========================================================================

    def get_healthcheck(self) -> Dict[str, Any]:
        base_health = super().get_healthcheck()

        base_health.update(
            {
                "components": {
                    "large_trade_detector": self.large_trade_detector.get_healthcheck(),
                    "whale_tracker": self.whale_tracker.get_healthcheck(),
                    "whale_cluster_analyzer": self.whale_cluster_analyzer.get_healthcheck(),
                }
            }
        )
        return base_health

    def get_stats(self) -> Dict[str, Any]:
        return {
            "analyzer_started": self._started,
            "large_trade_detector": self.large_trade_detector.get_all_stats(),
            "whale_tracker": self.whale_tracker.get_all_states(),
            "whale_cluster_analyzer": self.whale_cluster_analyzer.get_all_states(),
        }

    def get_symbol_stats(self, symbol: str) -> Dict[str, Any]:
        normalized_symbol = symbol.upper()

        return {
            "symbol": normalized_symbol,
            "large_trade_detector": self.large_trade_detector.get_symbol_stats(normalized_symbol),
            "whale_tracker": self.whale_tracker.get_symbol_state(normalized_symbol),
            "whale_cluster_analyzer": self.whale_cluster_analyzer.get_symbol_state(normalized_symbol),
        }

    # =========================================================================
    # Reset API
    # =========================================================================

    async def reset_symbol(self, symbol: str) -> None:
        normalized_symbol = symbol.upper()

        await self.large_trade_detector.reset_symbol(normalized_symbol)
        await self.whale_tracker.reset_symbol(normalized_symbol)
        await self.whale_cluster_analyzer.reset_symbol(normalized_symbol)

        self.logger.info(
            "Reset symbol state in WhaleAnalyzer",
            extra={"symbol": normalized_symbol},
        )

    async def reset_all(self) -> None:
        await self.large_trade_detector.reset_all()
        await self.whale_tracker.reset_all()
        await self.whale_cluster_analyzer.reset_all()

        self.logger.info("Reset all WhaleAnalyzer states")

    # =========================================================================
    # Component accessors
    # =========================================================================

    def get_components(self) -> Dict[str, Any]:
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
        event_bus: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> "WhaleAnalyzer":
        return cls(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )