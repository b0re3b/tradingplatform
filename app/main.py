from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from core.config import Config
from core.logger import get_logger

from app.factories import (
    build_analytics_components,
    build_data_caches,
    build_exchange_ws_clients,
    build_execution_components,
    build_market_stream,
    build_news_service,
    build_rest_clients,
    build_risk_manager,
    build_strategy_engine,
    build_telegram_service,
    send_telegram_startup_message,
)
from app.runtime import (
    RuntimeSettings,
    build_event_bus,
    build_scheduler,
    install_signal_handlers,
    register_component,
    start_component,
    stop_component,
)
from app.universe import discover_exchange_universe


logger = get_logger(__name__, service="app.main", event_type="bootstrap")


# ----------------------------------------------------------------------
# Environment diagnostics
# ----------------------------------------------------------------------


def _mask_env_value(value: str | None) -> str:
    if value is None:
        return "missing"

    if value == "":
        return "empty"

    if len(value) <= 6:
        return "***"

    return f"{value[:3]}***{value[-3:]}"


def _env_status(key: str) -> str:
    return _mask_env_value(os.environ.get(key))


def _resolve_env_file() -> Path:
    """
    Resolve .env path explicitly from project root.

    app/main.py -> project root is parent of app/.
    This avoids depending on IDE / terminal working directory.
    """
    return Path(__file__).resolve().parent.parent / ".env"


def _log_env_diagnostics(env_file: Path) -> None:
    logger.info(
        "Environment diagnostics | cwd=%s env_file=%s exists=%s",
        str(Path.cwd()),
        str(env_file),
        env_file.exists(),
    )

    # Do NOT print secrets. Only masked values / presence checks.
    watched_keys = [
        "APP_ENV",
        "EXCHANGE_NAME",
        "EXCHANGE_TESTNET",
        "MARKET_DATA_EXCHANGES",
        "MARKET_DATA_DISCOVER_ALL_SYMBOLS",
        "MARKET_DATA_SYMBOL_ALLOWLIST",
        "EXECUTION_EXCHANGE",
        "EXECUTION_MODE",
        "EXECUTION_LIVE_TRADING_ENABLED",
        "TELEGRAM_BOT_ENABLED",
        "TELEGRAM_ENABLED",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_TOKEN",
        "TELEGRAM_CHAT_ID",
        "TELEGRAM_DEFAULT_CHAT_ID",
        "NEWS_AI_ENABLED",
    ]

    logger.info(
        "Environment variables loaded | %s",
        " ".join(f"{key}={_env_status(key)}" for key in watched_keys),
    )


# ----------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------


class TradingSystemRuntime:
    """
    Production bootstrap for the whole project.

    Order:
    1. core EventBus/Scheduler
    2. REST clients and symbol discovery
    3. sharded WS clients and data caches
    4. analytics
    5. strategy
    6. risk
    7. execution
    8. Telegram observer
    9. independent news branch
    """

    def __init__(self, config: Config, settings: RuntimeSettings) -> None:
        self.config = config
        self.settings = settings
        self.settings.validate()

        self.event_bus = build_event_bus(config)
        self.scheduler = build_scheduler(config, self.event_bus)

        self.rest_clients: dict[str, Any] = {}
        self.ws_clients: dict[str, Any] = {}
        self.caches: dict[str, Any] = {}
        self.components: list[Any] = []
        self.telegram: Any | None = None
        self.news_service: Any | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return

        await self.event_bus.start()
        await self.scheduler.start()

        # REST clients first: needed for discovery and Binance execution.
        self.rest_clients = build_rest_clients(self.config, self.event_bus, self.scheduler)
        for rest in self.rest_clients.values():
            await register_component(rest)
            await start_component(rest)

        universe = await discover_exchange_universe(self.rest_clients, self.settings)
        logger.info(
            "Universe discovered | binance=%s bybit=%s okx=%s mexc=%s canonical=%s",
            len(universe.binance),
            len(universe.bybit),
            len(universe.okx),
            len(universe.mexc),
            len(universe.all_canonical_symbols()),
        )

        if self.settings.enable_market_data:
            self.ws_clients = build_exchange_ws_clients(
                config=self.config,
                event_bus=self.event_bus,
                scheduler=self.scheduler,
                universe=universe,
                settings=self.settings,
            )
            self.caches = build_data_caches(self.config, self.event_bus, self.scheduler)
            market_stream = build_market_stream(
                config=self.config,
                event_bus=self.event_bus,
                scheduler=self.scheduler,
                exchange_clients=self.ws_clients,
                caches=self.caches,
            )
            # MarketStream.start() guarantees its own registration.
            # Calling register_component() here and then start_component() causes
            # cache subscriptions to be registered twice unless every child cache is
            # perfectly idempotent.
            self.components.append(market_stream)
            await start_component(market_stream)

        if self.settings.enable_analytics:
            analytics_components = build_analytics_components(
                config=self.config,
                event_bus=self.event_bus,
                scheduler=self.scheduler,
                caches=self.caches,
                universe=universe,
            )
            for component in analytics_components:
                # Analytics components commonly call register() from start().
                # Avoid explicit register+start here to prevent subscribe/unsubscribe
                # churn and duplicate scheduler jobs.
                self.components.append(component)
                await start_component(component)

        if self.settings.enable_strategy:
            strategy_engine = build_strategy_engine(
                event_bus=self.event_bus,
                scheduler=self.scheduler,
                universe=universe,
            )
            self.components.append(strategy_engine)
            await start_component(strategy_engine)

        if self.settings.enable_risk:
            risk_manager = build_risk_manager(self.event_bus, self.scheduler)
            self.components.append(risk_manager)
            await start_component(risk_manager)

        if self.settings.enable_execution:
            execution_components = build_execution_components(
                event_bus=self.event_bus,
                scheduler=self.scheduler,
                binance_rest=self.rest_clients["binance"],
                settings=self.settings,
            )
            for component in execution_components:
                self.components.append(component)
                await start_component(component)

        if self.settings.enable_telegram:
            self.telegram = build_telegram_service(self.event_bus, self.scheduler)
            if self.telegram is not None:
                self.components.append(self.telegram)
                await start_component(self.telegram)
                await send_telegram_startup_message(
                    self.telegram,
                    (
                        "✅ Trading system started. Telegram is running as EventBus observer. "
                        "News is isolated from trading pipeline."
                    ),
                )

        if self.settings.enable_news:
            self.news_service = build_news_service(self.event_bus, self.scheduler)
            self.components.append(self.news_service)
            await start_component(self.news_service)

        await self.event_bus.emit(
            "system.runtime.started",
            {
                "market_data_exchanges": self.settings.market_data_exchanges,
                "execution_exchange": self.settings.execution_exchange,
                "execution_mode": self.settings.execution_mode,
                "live_trading_enabled": self.settings.live_trading_enabled,
                "news_isolated": True,
            },
            source="app.main",
        )
        self._started = True

    async def stop(self) -> None:
        # Stop in reverse order. Telegram/news/execution/risk/strategy/analytics/market_stream.
        for component in reversed(self.components):
            try:
                await stop_component(component)
            except Exception:
                logger.exception(
                    "Failed to stop component | component=%s",
                    component.__class__.__name__,
                )

        # Stop WS clients explicitly before REST/Scheduler/EventBus shutdown.
        # This prevents exchange adapters from emitting market.* events while
        # EventBus is already stopping.
        for ws in reversed(list(self.ws_clients.values())):
            try:
                await stop_component(ws)
            except Exception:
                logger.exception(
                    "Failed to stop WS client | component=%s",
                    ws.__class__.__name__,
                )

        for rest in reversed(list(self.rest_clients.values())):
            try:
                await stop_component(rest)
            except Exception:
                logger.exception(
                    "Failed to stop REST client | component=%s",
                    rest.__class__.__name__,
                )

        try:
            await self.scheduler.stop(
                wait_running_jobs=self.config.scheduler.wait_running_jobs_on_shutdown,
                timeout=self.config.scheduler.graceful_shutdown_timeout,
            )
        except Exception:
            logger.exception("Failed to stop scheduler")

        try:
            await self.event_bus.stop(drain=True, timeout=10.0)
        except Exception:
            logger.exception("Failed to stop event bus")

        self._started = False


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------


async def amain() -> None:
    logging.basicConfig(level=logging.INFO)

    env_file = _resolve_env_file()

    # Config.from_env() calls the project's internal .env loader.
    # Passing an absolute path removes ambiguity from IDE / terminal working directory.
    config = Config.from_env(env_file=env_file)

    # RuntimeSettings reads from os.environ, so it must be called after Config.from_env().
    settings = RuntimeSettings.from_env()

    _log_env_diagnostics(env_file)

    runtime = TradingSystemRuntime(config=config, settings=settings)
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)

    try:
        await runtime.start()
        await stop_event.wait()
    finally:
        await runtime.stop()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()