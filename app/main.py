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


async def _start_analytics_component(component: Any) -> None:
    """
    Універсальний запуск аналітичного компонента.

    Деякі компоненти мають тільки register() без start() (OIAnalyzer,
    PriceActionAnalyzer, SpoofingAnalyzer). start_component() шукає
    атрибут .start і якщо його нема — мовчки нічого не робить, через
    що компонент ніколи не реєструється і не слухає жодних подій.

    Порядок:
    1. Якщо є start() — викликаємо його. Компоненти що мають обидва методи
       (FundingAnalyzer, WhaleAnalyzer, SpreadAnalyzer, LiquidationStream,
       LiquidityService, OrderFlowAnalyzer) самі викликають register()
       всередині start(), тому окремий виклик register() їм не потрібен.
    2. Якщо start() відсутній але є register() — викликаємо register()
       напряму, щоб компонент підписався на EventBus і запустив scheduler
       jobs.
    """
    has_start = callable(getattr(component, "start", None))
    has_register = callable(getattr(component, "register", None))

    if has_start:
        await start_component(component)
    elif has_register:
        await register_component(component)
        logger.debug(
            "Analytics component started via register() (no start() method) | component=%s",
            component.__class__.__name__,
        )
    else:
        logger.warning(
            "Analytics component has neither start() nor register() | component=%s",
            component.__class__.__name__,
        )


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
        self._derivative_snapshot_poll_job_id: str | None = None
        self._derivative_snapshot_disabled_symbols: set[str] = set()
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
                # Components that have start() call register() internally (e.g.
                # FundingAnalyzer, WhaleAnalyzer, SpreadAnalyzer, LiquidationStream,
                # LiquidityService, OrderFlowAnalyzer).
                # Components that only have register() (e.g. OIAnalyzer,
                # PriceActionAnalyzer, SpoofingAnalyzer) are handled by
                # _start_analytics_component() which falls back to register()
                # so they actually subscribe to EventBus topics and start
                # scheduler jobs instead of silently doing nothing.
                self.components.append(component)
                await _start_analytics_component(component)

            # Orderflow is WebSocket-driven, but Binance USD-M open interest and
            # funding are REST snapshot endpoints. Start the poller only after
            # caches and analytics have subscribed to market.*.snapshot events so
            # the first run_immediately tick is not lost.
            self._start_derivative_snapshot_polling(universe)

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

    def _derivative_snapshot_poll_interval_seconds(self) -> float:
        value = os.getenv("DERIVATIVE_SNAPSHOT_POLL_INTERVAL_SECONDS")
        if value is None or not value.strip():
            return 60.0

        try:
            interval = float(value)
        except ValueError:
            logger.warning(
                "Invalid DERIVATIVE_SNAPSHOT_POLL_INTERVAL_SECONDS=%r; using default 60s",
                value,
            )
            return 60.0

        if interval < 10.0:
            logger.warning(
                "DERIVATIVE_SNAPSHOT_POLL_INTERVAL_SECONDS=%s is too low; clamping to 10s",
                interval,
            )
            return 10.0

        return interval

    @staticmethod
    def _is_derivative_symbol_unavailable_error(exc: BaseException) -> bool:
        """
        Return True for Binance USD-M symbols that should not be polled anymore.

        Binance can keep some symbols in discovery/universe or user allowlists while
        individual derivative endpoints reject them as delivering, delivered, settling,
        closed, or pre-trading. Retrying those symbols every scheduler tick only
        creates log noise and blocks useful REST budget for active contracts.
        """
        text = str(exc).lower()
        return (
            "code=-4108" in text
            or "code=-1121" in text
            or "delivering" in text
            or "delivered" in text
            or "settling" in text
            or "closed" in text
            or "pre-trading" in text
            or "invalid symbol" in text
        )

    def _disable_derivative_snapshot_symbol(self, symbol: str, exc: BaseException) -> None:
        normalized = str(symbol).upper()
        if normalized in self._derivative_snapshot_disabled_symbols:
            return

        self._derivative_snapshot_disabled_symbols.add(normalized)
        logger.warning(
            "Derivative snapshot polling disabled for unavailable Binance symbol | symbol=%s reason=%s",
            normalized,
            exc,
        )

    def _derivative_snapshot_symbols(self, universe: Any) -> list[str]:
        symbols = [
            str(symbol).upper()
            for symbol in getattr(universe, "binance", [])
            if str(symbol).strip()
        ]

        allowlist = [symbol.upper() for symbol in self.settings.symbol_allowlist]
        if allowlist:
            allowed = set(allowlist)
            symbols = [symbol for symbol in symbols if symbol in allowed]

        blocklist = {symbol.upper() for symbol in self.settings.symbol_blocklist}
        if blocklist:
            symbols = [symbol for symbol in symbols if symbol not in blocklist]

        # Preserve order while removing duplicates.
        seen: set[str] = set()
        unique_symbols: list[str] = []
        for symbol in symbols:
            if symbol in seen:
                continue
            seen.add(symbol)
            unique_symbols.append(symbol)

        return unique_symbols

    def _start_derivative_snapshot_polling(self, universe: Any) -> None:
        """
        Start REST polling for derivative-only market-data snapshots.

        Orderflow is fed by WebSocket trades/orderbook data. Binance USD-M open
        interest and funding are REST endpoints in this project, so caches and
        analytics stay silent unless the REST client is called periodically.
        """
        if self._derivative_snapshot_poll_job_id is not None:
            return

        if not self.settings.enable_market_data or not self.settings.enable_analytics:
            return

        if "binance" not in self.settings.market_data_exchanges:
            return

        binance = self.rest_clients.get("binance")
        if binance is None:
            logger.warning("Derivative snapshot polling disabled: Binance REST client is missing")
            return

        symbols = self._derivative_snapshot_symbols(universe)
        if not symbols:
            logger.warning("Derivative snapshot polling disabled: Binance universe is empty")
            return

        interval = self._derivative_snapshot_poll_interval_seconds()

        async def poll_binance_derivative_snapshots() -> None:
            active_symbols = [
                symbol
                for symbol in symbols
                if symbol not in self._derivative_snapshot_disabled_symbols
            ]

            if not active_symbols:
                logger.warning(
                    "Derivative snapshot polling skipped: all Binance symbols are disabled | original_symbols=%s",
                    len(symbols),
                )
                return

            for symbol in active_symbols:
                try:
                    await binance.get_open_interest(symbol=symbol)
                except Exception as exc:
                    if self._is_derivative_symbol_unavailable_error(exc):
                        self._disable_derivative_snapshot_symbol(symbol, exc)
                        continue

                    logger.exception(
                        "Failed to poll Binance open interest snapshot | symbol=%s",
                        symbol,
                    )

                try:
                    await binance.get_funding_rate(symbol=symbol, limit=1)
                except Exception as exc:
                    if self._is_derivative_symbol_unavailable_error(exc):
                        self._disable_derivative_snapshot_symbol(symbol, exc)
                        continue

                    logger.exception(
                        "Failed to poll Binance funding snapshot | symbol=%s",
                        symbol,
                    )

        self._derivative_snapshot_poll_job_id = self.scheduler.add_interval_job(
            name="binance-derivative-snapshots-poll",
            func=poll_binance_derivative_snapshots,
            interval=interval,
            run_immediately=True,
            max_retries=0,
            timeout=max(30.0, min(interval * 0.8, 120.0)),
            allow_overlap=False,
        )

        logger.info(
            "Derivative snapshot polling started | exchange=binance symbols=%s interval=%s job_id=%s",
            len(symbols),
            interval,
            self._derivative_snapshot_poll_job_id,
        )

    def _stop_derivative_snapshot_polling(self) -> None:
        job_id = self._derivative_snapshot_poll_job_id
        if job_id is None:
            return

        self._derivative_snapshot_poll_job_id = None
        try:
            self.scheduler.remove_job(job_id)
        except Exception:
            logger.exception("Failed to remove derivative snapshot polling job | job_id=%s", job_id)
        else:
            logger.info("Derivative snapshot polling stopped | job_id=%s", job_id)

    async def stop(self) -> None:
        self._stop_derivative_snapshot_polling()

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