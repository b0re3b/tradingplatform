from __future__ import annotations

from typing import Any

from .config import CrossExchangeSpreadConfig, SpotFuturesSpreadConfig
from .cross_exchange_analyzer import CrossExchangeSpreadAnalyzer
from .models import ArbitrageOpportunity, SpreadSnapshot
from .spot_futures_analyzer import SpotFuturesSpreadAnalyzer


class SpreadAnalyzer:
    """
    Верхньорівневий фасад для пакета spreads.

    Відповідальність:
    - створювати й координувати піданалізатори
    - запускати / зупиняти їх разом
    - надавати єдину точку доступу до stats
    - віддавати latest snapshots / opportunities

    Не відповідає за:
    - власну обробку quote/funding подій
    - побудову snapshot-ів
    - signal generation
    - arbitrage detection
    """

    def __init__(
        self,
        *,
        event_bus: Any,
        scheduler: Any | None = None,
        spot_futures_config: SpotFuturesSpreadConfig | None = None,
        cross_exchange_config: CrossExchangeSpreadConfig | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._scheduler = scheduler

        self._spot_futures_config = spot_futures_config or SpotFuturesSpreadConfig()
        self._cross_exchange_config = cross_exchange_config or CrossExchangeSpreadConfig()

        self._spot_futures_analyzer = SpotFuturesSpreadAnalyzer(
            config=self._spot_futures_config,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        self._cross_exchange_analyzer = CrossExchangeSpreadAnalyzer(
            config=self._cross_exchange_config,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def spot_futures(self) -> SpotFuturesSpreadAnalyzer:
        return self._spot_futures_analyzer

    @property
    def cross_exchange(self) -> CrossExchangeSpreadAnalyzer:
        return self._cross_exchange_analyzer

    async def start(self) -> None:
        if self._running:
            return

        await self._spot_futures_analyzer.start()
        await self._cross_exchange_analyzer.start()

        self._running = True

    async def stop(self) -> None:
        if not self._running:
            return

        await self._cross_exchange_analyzer.stop()
        await self._spot_futures_analyzer.stop()

        self._running = False

    def get_stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
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
        instrument_type: Any,
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
        instrument_type: Any | None = None,
        profitable_only: bool = True,
        active_only: bool = True,
    ) -> list[ArbitrageOpportunity]:
        return self._cross_exchange_analyzer.get_best_opportunities(
            symbol=symbol,
            instrument_type=instrument_type,
            profitable_only=profitable_only,
            active_only=active_only,
        )