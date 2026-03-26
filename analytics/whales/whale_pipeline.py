from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.logger import get_logger

from analytics.whales.large_trade_detector import (
    LargeTradeDetector,
    LargeTradeDetectorConfig,
)
from analytics.whales.whale_tracker import (
    WhaleTracker,
    WhaleTrackerConfig,
)
from analytics.whales.whale_cluster_analyzer import (
    WhaleClusterAnalyzer,
    WhaleClusterAnalyzerConfig,
)


@dataclass(slots=True)
class WhalePipelineConfig:
    """
    Конфігурація всього whale pipeline.

    Дає змогу централізовано керувати підмодулями:
        - LargeTradeDetector
        - WhaleTracker
        - WhaleClusterAnalyzer
    """

    enabled: bool = True
    auto_start_components: bool = True

    large_trade_detector: LargeTradeDetectorConfig = field(
        default_factory=LargeTradeDetectorConfig
    )
    whale_tracker: WhaleTrackerConfig = field(
        default_factory=WhaleTrackerConfig
    )
    whale_cluster_analyzer: WhaleClusterAnalyzerConfig = field(
        default_factory=WhaleClusterAnalyzerConfig
    )


class WhalePipeline:
    """
    Єдиний orchestration layer для whale analytics.

    Потік даних:
        raw market.trade
            -> LargeTradeDetector
            -> analytics.whales.large_trade
            -> WhaleTracker
            -> analytics.whales.whale_activity / whale_pressure / whale_liquidation_context
            -> WhaleClusterAnalyzer
            -> analytics.whales.whale_cluster / whale_cluster_update / whale_cluster_exhaustion

    Також:
        raw market.liquidation
            -> WhaleTracker
            -> analytics.whales.whale_liquidation_context
            -> WhaleClusterAnalyzer
    """

    def __init__(
        self,
        config: Optional[WhalePipelineConfig] = None,
        event_bus: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        self.config = config or WhalePipelineConfig()
        self.event_bus = event_bus
        self.scheduler = scheduler

        self.logger = get_logger(
            __name__,
            service_name="analytics.whales.whale_pipeline",
        )

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

        self._started = False

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            self.logger.warning("WhalePipeline already started")
            return

        if not self.config.enabled:
            self.logger.info("WhalePipeline is disabled by config")
            return

        if self.config.auto_start_components:
            await self.large_trade_detector.start()
            await self.whale_tracker.start()
            await self.whale_cluster_analyzer.start()

        self._started = True

        self.logger.info(
            "WhalePipeline started",
            extra={
                "large_trade_input_event": self.config.large_trade_detector.input_event_name,
                "large_trade_output_event": self.config.large_trade_detector.output_event_name,
                "whale_activity_event": self.config.whale_tracker.whale_activity_event_name,
                "whale_pressure_event": self.config.whale_tracker.whale_pressure_event_name,
                "whale_cluster_event": self.config.whale_cluster_analyzer.whale_cluster_event_name,
            },
        )

    async def stop(self) -> None:
        if not self._started:
            return

        await self.whale_cluster_analyzer.stop()
        await self.whale_tracker.stop()
        await self.large_trade_detector.stop()

        self._started = False
        self.logger.info("WhalePipeline stopped")

    # -------------------------------------------------------------------------
    # Direct input methods
    # -------------------------------------------------------------------------

    async def process_trade(self, event: Dict[str, Any]) -> Optional[Any]:
        """
        Прямий вхід raw trade event у pipeline.

        Зазвичай raw trade приходить через EventBus у LargeTradeDetector,
        але цей метод корисний для:
            - тестів
            - backtesting
            - ручного прогону
        """
        return await self.large_trade_detector.process_trade(event)

    async def process_liquidation(self, event: Dict[str, Any]) -> Optional[Any]:
        """
        Прямий вхід raw liquidation event у pipeline.
        """
        return await self.whale_tracker.process_liquidation_event(event)

    # -------------------------------------------------------------------------
    # State / stats
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "pipeline_started": self._started,
            "large_trade_detector": self.large_trade_detector.get_all_stats(),
            "whale_tracker": self.whale_tracker.get_all_states(),
            "whale_cluster_analyzer": self.whale_cluster_analyzer.get_all_states(),
        }

    def get_symbol_stats(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "large_trade_detector": self.large_trade_detector.get_symbol_stats(symbol),
            "whale_tracker": self.whale_tracker.get_symbol_state(symbol),
            "whale_cluster_analyzer": self.whale_cluster_analyzer.get_symbol_state(symbol),
        }

    async def reset_symbol(self, symbol: str) -> None:
        await self.large_trade_detector.reset_symbol(symbol)
        await self.whale_tracker.reset_symbol(symbol)
        await self.whale_cluster_analyzer.reset_symbol(symbol)

        self.logger.info(
            "Reset symbol state in WhalePipeline",
            extra={"symbol": symbol},
        )

    async def reset_all(self) -> None:
        await self.large_trade_detector.reset_all()
        await self.whale_tracker.reset_all()
        await self.whale_cluster_analyzer.reset_all()

        self.logger.info("Reset all WhalePipeline states")