from __future__ import annotations

from typing import Any

from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import CrossExchangeSpreadConfig, SpotFuturesSpreadConfig
from .cross_exchange_analyzer import CrossExchangeSpreadAnalyzer
from .enums import InstrumentType
from .models import ArbitrageOpportunity, SpreadSnapshot
from .spot_futures_analyzer import SpotFuturesSpreadAnalyzer


class SpreadAnalyzer:
    """
    Production-grade facade для analytics/spreads.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        spot_futures_config: SpotFuturesSpreadConfig | None = None,
        cross_exchange_config: CrossExchangeSpreadConfig | None = None,
        auto_register: bool = False,
    ) -> None:
        self._event_bus = event_bus
        self._scheduler = scheduler

        self._spot_futures_config = spot_futures_config or SpotFuturesSpreadConfig()
        self._cross_exchange_config = cross_exchange_config or CrossExchangeSpreadConfig()

        self._logger = get_logger(
            __name__,
            service_name="spread_analyzer",
            event_type="spreads_facade",
        )

        self._spot_futures_analyzer = SpotFuturesSpreadAnalyzer(
            config=self._spot_futures_config,
            event_bus=self._event_bus,
            scheduler=self._scheduler,
        )

        self._cross_exchange_analyzer = CrossExchangeSpreadAnalyzer(
            config=self._cross_exchange_config,
            event_bus=self._event_bus,
            scheduler=self._scheduler,
        )

        self._running = False
        self._registered = False

        if auto_register:
            self.register()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_registered(self) -> bool:
        return self._registered

    @property
    def spot_futures(self) -> SpotFuturesSpreadAnalyzer:
        return self._spot_futures_analyzer

    @property
    def cross_exchange(self) -> CrossExchangeSpreadAnalyzer:
        return self._cross_exchange_analyzer

    def register(self) -> None:
        if self._registered:
            self._logger.warning("SpreadAnalyzer already registered")
            return

        self._spot_futures_analyzer.register()
        self._cross_exchange_analyzer.register()

        self._registered = True

        self._logger.info(
            "SpreadAnalyzer registered | spot_futures=%s cross_exchange=%s",
            self._spot_futures_analyzer.is_registered,
            self._cross_exchange_analyzer.is_registered,
        )

    def unregister(self) -> None:
        if self._running:
            self._logger.warning(
                "SpreadAnalyzer unregister requested while running; stop() should be called first"
            )

        if not self._registered:
            self._logger.warning("SpreadAnalyzer already unregistered")
            return

        self._cross_exchange_analyzer.unregister()
        self._spot_futures_analyzer.unregister()

        self._registered = False

        self._logger.info("SpreadAnalyzer unregistered")

    async def start(self) -> None:
        if self._running:
            self._logger.warning("SpreadAnalyzer already started")
            return

        if not self._registered:
            self.register()

        await self._spot_futures_analyzer.start()
        await self._cross_exchange_analyzer.start()

        self._running = True

        self._logger.info(
            "SpreadAnalyzer started | spot_futures_enabled=%s cross_exchange_enabled=%s",
            self._spot_futures_config.enabled,
            self._cross_exchange_config.enabled,
        )

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("SpreadAnalyzer already stopped")
            return

        await self._cross_exchange_analyzer.stop()
        await self._spot_futures_analyzer.stop()

        self._running = False

        self._logger.info("SpreadAnalyzer stopped")

    async def shutdown(self) -> None:
        if self._running:
            await self.stop()

        if self._registered:
            self.unregister()

        self._logger.info("SpreadAnalyzer shutdown completed")

    def get_stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "registered": self._registered,
            "spot_futures": self._spot_futures_analyzer.get_stats(),
            "cross_exchange": self._cross_exchange_analyzer.get_stats(),
        }

    def get_latest_spot_futures_snapshot(
        self,
        symbol: str,
        spot_exchange: str,
        futures_exchange: str,
    ) -> SpreadSnapshot | None:
        return self._spot_futures_analyzer.get_latest_snapshot(
            symbol=symbol,
            spot_exchange=spot_exchange,
            futures_exchange=futures_exchange,
        )

    def get_latest_cross_exchange_snapshot(
        self,
        symbol: str,
        exchange_a: str,
        exchange_b: str,
        instrument_type: InstrumentType,
    ) -> SpreadSnapshot | None:
        return self._cross_exchange_analyzer.get_latest_snapshot(
            symbol=symbol,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            instrument_type=instrument_type,
        )

    def get_best_cross_exchange_opportunities(
        self,
        symbol: str | None = None,
        instrument_type: InstrumentType | None = None,
        profitable_only: bool = True,
        active_only: bool = True,
        limit: int | None = None,
    ) -> list[ArbitrageOpportunity]:
        return self._cross_exchange_analyzer.get_best_opportunities(
            symbol=symbol,
            instrument_type=instrument_type,
            profitable_only=profitable_only,
            active_only=active_only,
            limit=limit,
        )