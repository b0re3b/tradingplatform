from __future__ import annotations

from typing import Any

from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import CrossExchangeSpreadConfig, SpotFuturesSpreadConfig
from .cross_exchange_analyzer import CrossExchangeSpreadAnalyzer
from .enums import InstrumentType
from .models import (
    DEFAULT_TIMEFRAME,
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
        exchange adapters
            -> market.quote / market.funding
            -> QuoteCache / FundingCache
            -> market.quote.updated / market.funding.updated
            -> analytics.spreads analyzers
            -> analytics.spreads.*

    Важливо:
    - facade не отримує market data напряму;
    - не створює exchange adapters;
    - не містить spread business logic;
    - не викликає strategy/risk/execution напряму;
    - SpotFuturesSpreadAnalyzer є дозволеним spot+futures компонентом;
    - CrossExchangeSpreadAnalyzer працює за своїм config, зокрема може бути
      spot/perp/futures або futures-only залежно від allowed_instrument_types.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        spot_futures_config: SpotFuturesSpreadConfig | None = None,
        cross_exchange_config: CrossExchangeSpreadConfig | None = None,
        enable_spot_futures: bool = True,
        enable_cross_exchange: bool = True,
        auto_register: bool = False,
    ) -> None:
        self._event_bus = event_bus
        self._scheduler = scheduler

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
            )

        if self._enable_cross_exchange:
            self._cross_exchange_analyzer = CrossExchangeSpreadAnalyzer(
                config=self._cross_exchange_config,
                event_bus=self._event_bus,
                scheduler=self._scheduler,
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
        return self._running

    @property
    def is_registered(self) -> bool:
        return self._registered

    @property
    def spot_futures_enabled(self) -> bool:
        return self._spot_futures_analyzer is not None

    @property
    def cross_exchange_enabled(self) -> bool:
        return self._cross_exchange_analyzer is not None

    @property
    def spot_futures(self) -> SpotFuturesSpreadAnalyzer:
        if self._spot_futures_analyzer is None:
            raise RuntimeError("SpotFuturesSpreadAnalyzer is disabled")
        return self._spot_futures_analyzer

    @property
    def cross_exchange(self) -> CrossExchangeSpreadAnalyzer:
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
        - SpotFuturesSpreadAnalyzer -> market.quote.updated / market.funding.updated
        - CrossExchangeSpreadAnalyzer -> market.quote.updated
        """
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
            "SpreadAnalyzer registered | components=%s spot_futures_registered=%s cross_exchange_registered=%s",
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
            },
        )

    def unregister(self) -> None:
        """
        Відписує enabled analyzer-и від EventBus.

        Рекомендований порядок:
            await stop()
            unregister()
        """
        if self._running:
            self._logger.warning(
                "SpreadAnalyzer unregister requested while running; stop() should be called first"
            )

        if not self._registered:
            self._logger.warning("SpreadAnalyzer already unregistered")
            return

        if self._cross_exchange_analyzer is not None:
            self._cross_exchange_analyzer.unregister()

        if self._spot_futures_analyzer is not None:
            self._spot_futures_analyzer.unregister()

        self._registered = False

        self._logger.info("SpreadAnalyzer unregistered")

    async def start(self) -> None:
        """
        Запускає enabled analyzer-и.

        Якщо register() ще не викликаний — викликає його автоматично.
        """
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
            "SpreadAnalyzer started | components=%s spot_futures_enabled=%s cross_exchange_enabled=%s",
            started_components,
            self._spot_futures_config.enabled if self._spot_futures_analyzer is not None else False,
            self._cross_exchange_config.enabled if self._cross_exchange_analyzer is not None else False,
            extra={
                "components": started_components,
                "scope": "exchange:market_type:symbol:timeframe",
            },
        )

    async def stop(self) -> None:
        """
        Зупиняє enabled analyzer-и.

        Порядок зупинки зворотний до старту.
        """
        if not self._running:
            self._logger.warning("SpreadAnalyzer already stopped")
            return

        if self._cross_exchange_analyzer is not None:
            await self._cross_exchange_analyzer.stop()

        if self._spot_futures_analyzer is not None:
            await self._spot_futures_analyzer.stop()

        self._running = False

        self._logger.info("SpreadAnalyzer stopped")

    async def shutdown(self) -> None:
        """
        Повний shutdown:
        - stop(), якщо running;
        - unregister(), якщо registered.
        """
        if self._running:
            await self.stop()

        if self._registered:
            self.unregister()

        self._logger.info("SpreadAnalyzer shutdown completed")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "registered": self._registered,
            "scope": "exchange:market_type:symbol:timeframe",
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
                "cross_exchange_topics": (
                    list(self._cross_exchange_config.production_input_topics)
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