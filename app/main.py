from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
from pathlib import Path
from typing import Any

from core.config import Config
from core.logger import get_logger
from core.event_flow_monitor import EventFlowMonitor, EventFlowMonitorConfig
from app.factories import (
    build_analytics_components,
    build_market_ingestion_service,
    build_market_scheduler,
    build_market_state_store,
    build_data_caches,
    build_exchange_ws_clients,
    build_execution_components,
    build_market_stream,
    build_news_service,
    build_parquet_storage,
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
        "MARKET_DATA_SYMBOL_BLOCKLIST",
        "MARKET_DATA_QUOTE_ASSET",
        "MARKET_DATA_TIMEFRAMES",
        "ANALYTICS_EXCHANGE",
        "ANALYTICS_MARKET_TYPE",
        "ANALYTICS_SYMBOLS",
        "PRICE_ACTION_EXCHANGE",
        "PRICE_ACTION_MARKET_TYPE",
        "PRICE_ACTION_SYMBOLS",
        "PRICE_ACTION_TIMEFRAMES",
        "ORDERFLOW_DEFAULT_EXCHANGE",
        "ORDERFLOW_DEFAULT_MARKET_TYPE",
        "ORDERFLOW_DEFAULT_TIMEFRAME",
        "LIQUIDITY_CANDLES_UPDATED_TOPICS",
        "LIQUIDITY_MIN_CANDLES_FOR_SNAPSHOT",
        "BINANCE_WS_STREAMS",
        "BINANCE_LIQUIDATION_STREAM_NAME",
        "BINANCE_WS_KLINE_INTERVAL",
        "BYBIT_WS_STREAMS",
        "BYBIT_LIQUIDATION_STREAM_NAME",
        "BYBIT_CATEGORY",
        "OKX_WS_STREAMS",
        "OKX_CANDLE_CHANNEL",
        "MEXC_WS_STREAMS",
        "MEXC_KLINE_INTERVAL",
        "STARTUP_WARMUP_EXCHANGE",
        "DERIVATIVE_SNAPSHOT_EXCHANGE",
        "EXECUTION_EXCHANGE",
        "EXECUTION_MARKET_TYPE",
        "EXECUTION_MODE",
        "EXECUTION_LIVE_TRADING_ENABLED",
        "TELEGRAM_BOT_ENABLED",
        "TELEGRAM_ENABLED",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_TOKEN",
        "TELEGRAM_CHAT_ID",
        "TELEGRAM_DEFAULT_CHAT_ID",
        "NEWS_AI_ENABLED",
        "STARTUP_WARMUP_ENABLED",
        "STARTUP_WARMUP_REQUIRED",
        "STARTUP_WARMUP_TIMEFRAMES",
        "STARTUP_WARMUP_KLINE_LIMIT",
        "STARTUP_WARMUP_FUNDING_LIMIT",
        "STARTUP_WARMUP_CONCURRENCY",
        "STARTUP_WARMUP_BATCH_SIZE",
        "STARTUP_WARMUP_PERSIST_ENABLED",
        "STARTUP_WARMUP_PERSIST_REQUIRED",
        "STARTUP_WARMUP_FLUSH_STORAGE_BEFORE_TRADING",
        "STORAGE_PARQUET_ENABLED",
        "STORAGE_PARQUET_DIR",
        "STORAGE_FLUSH_INTERVAL_SECONDS",
        "STORAGE_BATCH_SIZE",
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
    3. sharded WS clients and data caches registration
    4. analytics
    5. startup warmup
    6. strategy
    7. risk
    8. execution
    9. Telegram observer
    10. live market stream
    11. derivative snapshot polling
    12. independent news branch
    """

    def __init__(self, config: Config, settings: RuntimeSettings) -> None:
        self.config = config
        self.settings = settings
        self.settings.validate()

        self.event_bus = build_event_bus(config)
        self.scheduler = build_scheduler(config, self.event_bus)
        self._event_flow_monitor: EventFlowMonitor | None = None
        self.market_state_store: Any | None = None
        self.market_ingestion: Any | None = None
        self.market_scheduler: Any | None = None
        self.rest_clients: dict[str, Any] = {}
        self.ws_clients: dict[str, Any] = {}
        self.caches: dict[str, Any] = {}
        self.components: list[Any] = []
        self.parquet_storage: Any | None = None
        self.telegram: Any | None = None
        self.news_service: Any | None = None
        self._market_stream: Any | None = None
        self._derivative_snapshot_poll_job_id: str | None = None
        self._derivative_snapshot_poll_task: asyncio.Task[None] | None = None
        self._derivative_snapshot_poll_stop: asyncio.Event | None = None
        self._derivative_snapshot_disabled_symbols: set[str] = set()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return

        await self.event_bus.start()
        self._event_flow_monitor = EventFlowMonitor(
            self.event_bus,
            EventFlowMonitorConfig.from_env(),
        )

        await self._event_flow_monitor.start()
        await self.scheduler.start()

        # State-driven market-data foundation. REST/WS adapters write through
        # MarketIngestionService instead of emitting high-frequency raw market.*
        # events to EventBus. Analytics reads snapshots via MarketScheduler.
        self.market_state_store = build_market_state_store(
            self.config,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            settings=self.settings,
        )
        self.market_ingestion = build_market_ingestion_service(
            config=self.config,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            market_state_store=self.market_state_store,
            settings=self.settings,
        )
        self.market_scheduler = build_market_scheduler(
            config=self.config,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            market_state_store=self.market_state_store,
            settings=self.settings,
        )

        # REST clients first: needed for discovery, warmup and Binance execution.
        self.rest_clients = build_rest_clients(
            self.config,
            self.event_bus,
            self.scheduler,
            self.settings,
            market_ingestion=self.market_ingestion,
        )
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
                market_ingestion=self.market_ingestion,
            )
            self.caches = build_data_caches(
                self.config,
                self.event_bus,
                self.scheduler,
                market_state_store=self.market_state_store,
                market_ingestion=self.market_ingestion,
            )

            if self.settings.startup_warmup_persist_enabled:
                self.parquet_storage = build_parquet_storage(
                    self.config,
                    self.event_bus,
                    self.scheduler,
                )
                # Append storage before market_stream so shutdown happens in reverse:
                # analytics/execution -> market_stream -> parquet final flush.
                self.components.append(self.parquet_storage)
                await register_component(self.parquet_storage)
                await start_component(self.parquet_storage)
                self._validate_startup_storage_ready()
            elif self.settings.startup_warmup_persist_required and self._startup_warmup_enabled():
                raise RuntimeError(
                    "STARTUP_WARMUP_PERSIST_REQUIRED=true but STARTUP_WARMUP_PERSIST_ENABLED=false"
                )

            market_stream = build_market_stream(
                config=self.config,
                event_bus=self.event_bus,
                scheduler=self.scheduler,
                exchange_clients=self.ws_clients,
                caches=self.caches,
                market_state_store=self.market_state_store,
                market_ingestion=self.market_ingestion,
                market_scheduler=self.market_scheduler,
            )
            self._market_stream = market_stream
            self.components.append(market_stream)

            # Register caches now, but intentionally do not start WS clients yet.
            # Historical REST warmup must populate caches/analytics before live
            # market-data events and before strategy/risk/execution are started.
            # ParquetStorage is already registered at this point, so every closed
            # warmup candle emitted by CandlesCache is persisted.
            await register_component(market_stream)

        if self.settings.enable_analytics:
            analytics_components = build_analytics_components(
                config=self.config,
                event_bus=self.event_bus,
                scheduler=self.scheduler,
                caches=self.caches,
                universe=universe,
                settings=self.settings,
                market_state_store=self.market_state_store,
                market_scheduler=self.market_scheduler,
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

            # Start snapshot evaluation before warmup so REST warmup data written
            # into MarketStateStore can immediately feed analytics, while
            # strategy/risk/execution are still not running.
            if self.market_scheduler is not None:
                await start_component(self.market_scheduler)

            await self._run_startup_warmup(universe)

            # Live market publishers are intentionally started later, after
            # strategy/risk/execution/Telegram consumers are subscribed. Warmup
            # can safely run here because it uses REST and cache/analytics state,
            # while live WS and derivative polling must not race ahead of the
            # trading pipeline.

        if self.settings.enable_strategy:
            strategy_engine = build_strategy_engine(
                event_bus=self.event_bus,
                scheduler=self.scheduler,
                universe=universe,
                settings=self.settings,
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

        # Start live market-data publishers only after the consumers that depend
        # on them are already active: strategy, risk, execution and Telegram.
        # This prevents live analytics events from being emitted before the
        # trading pipeline can observe/reject/build them.
        if self._market_stream is not None:
            await start_component(self._market_stream)
            self._verify_liquidation_ws_capability()

        # Binance USD-M open interest/funding are REST snapshot endpoints. Start
        # polling after live consumers are subscribed, but before independent
        # news, so derivative analytics can feed strategy immediately.
        if self.settings.enable_analytics:
            self._start_derivative_snapshot_polling(universe)

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

    def _startup_warmup_enabled(self) -> bool:
        return (
            bool(self.settings.startup_warmup_enabled)
            and self.settings.enable_market_data
            and self.settings.enable_analytics
            and self.settings.startup_warmup_exchange in self.settings.market_data_exchanges
        )

    def _startup_warmup_timeframes(self) -> list[str]:
        configured = [str(tf).strip() for tf in self.settings.startup_warmup_timeframes if str(tf).strip()]
        if not configured:
            configured = [str(tf).strip() for tf in self.settings.timeframes if str(tf).strip()]
        # Preserve order and remove duplicates.
        seen: set[str] = set()
        result: list[str] = []
        for timeframe in configured:
            if timeframe in seen:
                continue
            seen.add(timeframe)
            result.append(timeframe)
        return result

    def _validate_startup_storage_ready(self) -> None:
        if not self.settings.startup_warmup_persist_enabled:
            return

        if self.parquet_storage is None:
            if self.settings.startup_warmup_persist_required and self._startup_warmup_enabled():
                raise RuntimeError("Startup warmup persistence is required but ParquetStorage was not created")
            return

        stats_method = getattr(self.parquet_storage, "stats", None)
        stats = stats_method() if callable(stats_method) else {}
        enabled = bool(stats.get("enabled", True))
        started = bool(stats.get("started", False))
        registered = bool(stats.get("registered", False))

        if self.settings.startup_warmup_persist_required and self._startup_warmup_enabled():
            if not enabled:
                raise RuntimeError("Startup warmup persistence is required but ParquetStorage is disabled")
            if not registered or not started:
                raise RuntimeError(
                    f"Startup warmup persistence is required but ParquetStorage is not ready: {stats}"
                )

        logger.info(
            "Startup warmup persistence ready | enabled=%s registered=%s started=%s root_dir=%s",
            enabled,
            registered,
            started,
            stats.get("root_dir"),
        )

    async def _flush_startup_warmup_storage(self) -> None:
        if not (
            self._startup_warmup_enabled()
            and self.settings.startup_warmup_persist_enabled
            and self.settings.startup_warmup_flush_storage_before_trading
        ):
            return

        if self.parquet_storage is None:
            if self.settings.startup_warmup_persist_required:
                raise RuntimeError("Startup warmup persistence is required but ParquetStorage is missing")
            logger.warning("Startup warmup Parquet flush skipped: ParquetStorage is missing")
            return

        flush = getattr(self.parquet_storage, "flush", None)
        if not callable(flush):
            if self.settings.startup_warmup_persist_required:
                raise RuntimeError("ParquetStorage has no flush() method")
            logger.warning("Startup warmup Parquet flush skipped: flush() method is missing")
            return

        result = flush()
        if inspect.isawaitable(result):
            result = await result

        stats_method = getattr(self.parquet_storage, "stats", None)
        stats = stats_method() if callable(stats_method) else {}

        logger.info(
            "Startup warmup Parquet flush completed | result=%s stats=%s",
            result,
            stats,
        )

        if not self.settings.startup_warmup_persist_required:
            return

        records_written = int(stats.get("records_written") or 0)
        files_written = int(stats.get("files_written") or 0)
        buffered_total = int(stats.get("buffered_total") or 0)
        if records_written <= 0 or files_written <= 0:
            raise RuntimeError(
                "Startup warmup persistence failed: no Parquet records/files were written "
                f"before trading startup | result={result} stats={stats}"
            )
        if buffered_total > 0:
            raise RuntimeError(
                "Startup warmup persistence failed: Parquet buffer is not empty after forced flush "
                f"| result={result} stats={stats}"
            )

    async def _wait_event_bus_idle(self, *, timeout: float | None = None) -> None:
        """
        Best-effort wait until REST warmup events have propagated through caches
        and analytics subscriptions.

        EventBus intentionally does not expose a public drain API while running.
        Using the internal queue size here is restricted to application bootstrap;
        it prevents strategy/risk/execution from starting while warmup events are
        still waiting in the queue.
        """
        queue = getattr(self.event_bus, "_queue", None)
        if queue is None or not hasattr(queue, "qsize"):
            await asyncio.sleep(max(0.0, self.settings.startup_warmup_settle_seconds))
            return

        deadline = asyncio.get_running_loop().time() + (
            self.settings.startup_warmup_eventbus_idle_timeout if timeout is None else timeout
        )
        stable_empty_ticks = 0

        while True:
            try:
                qsize = int(queue.qsize())
            except Exception:
                qsize = 0

            if qsize <= 0:
                stable_empty_ticks += 1
                if stable_empty_ticks >= 2:
                    break
            else:
                stable_empty_ticks = 0

            if asyncio.get_running_loop().time() >= deadline:
                logger.warning(
                    "Startup warmup EventBus idle wait timed out | remaining_queue=%s",
                    qsize,
                )
                break

            await asyncio.sleep(0.05)

        settle = max(0.0, self.settings.startup_warmup_settle_seconds)
        if settle:
            await asyncio.sleep(settle)

    async def _evaluate_market_state_once(self, *, reason: str) -> None:
        """Run one bounded state-snapshot analytics tick if the state scheduler exists."""
        if self.market_scheduler is None:
            return
        evaluate = getattr(self.market_scheduler, "evaluate_dirty_once", None)
        if not callable(evaluate):
            return
        try:
            result = evaluate()
            if inspect.isawaitable(result):
                result = await result
            logger.debug("Market state evaluated | reason=%s result=%s", reason, result)
        except Exception:
            logger.exception("Market state evaluation failed | reason=%s", reason)

    async def _run_startup_warmup(self, universe: Any) -> None:
        """
        Load the minimum historical context required by analytics before the
        trading pipeline starts.

        Startup order is intentional:
        - EventBus/Scheduler/REST are running.
        - Caches are registered.
        - Analytics are registered/started.
        - Strategy/risk/execution/Telegram/news are NOT started yet.

        This lets price_action/liquidity/funding build initial state without
        producing live strategy/risk/execution side effects.
        """
        if not self._startup_warmup_enabled():
            logger.info(
                "Startup warmup skipped | enabled=%s market_data=%s analytics=%s exchanges=%s",
                self.settings.startup_warmup_enabled,
                self.settings.enable_market_data,
                self.settings.enable_analytics,
                self.settings.market_data_exchanges,
            )
            return

        warmup_exchange = self.settings.startup_warmup_exchange
        rest = self.rest_clients.get(warmup_exchange)
        if rest is None:
            message = f"Startup warmup cannot run: {warmup_exchange} REST client is missing"
            if self.settings.startup_warmup_required:
                raise RuntimeError(message)
            logger.warning(message)
            return

        symbols = self._derivative_snapshot_symbols(universe, exchange=warmup_exchange)
        if not symbols:
            message = f"Startup warmup cannot run: {warmup_exchange} universe is empty"
            if self.settings.startup_warmup_required:
                raise RuntimeError(message)
            logger.warning(message)
            return

        timeframes = self._startup_warmup_timeframes()
        kline_limit = max(1, int(self.settings.startup_warmup_kline_limit))
        funding_limit = max(1, int(self.settings.startup_warmup_funding_limit))
        concurrency = max(1, int(self.settings.startup_warmup_concurrency))
        batch_size = max(1, int(self.settings.startup_warmup_batch_size))

        logger.info(
            "Startup warmup started | exchange=%s symbols=%s timeframes=%s kline_limit=%s funding_limit=%s concurrency=%s batch_size=%s",
            warmup_exchange,
            len(symbols),
            timeframes,
            kline_limit,
            funding_limit,
            concurrency,
            batch_size,
        )

        kline_success = 0
        kline_failed: list[str] = []
        funding_success = 0
        funding_failed: list[str] = []
        sem = asyncio.Semaphore(concurrency)

        async def warmup_klines(symbol: str, timeframe: str) -> tuple[str, str, int, str | None]:
            async with sem:
                try:
                    candles = await rest.get_klines(
                        symbol=symbol,
                        interval=timeframe,
                        limit=kline_limit,
                    )
                    return symbol, timeframe, len(candles or []), None
                except Exception as exc:
                    if self._is_derivative_symbol_unavailable_error(exc):
                        self._disable_derivative_snapshot_symbol(symbol, exc)
                    logger.exception(
                        "Startup candle warmup failed | exchange=%s symbol=%s timeframe=%s",
                        warmup_exchange,
                        symbol,
                        timeframe,
                    )
                    return symbol, timeframe, 0, str(exc)

        async def warmup_funding(symbol: str) -> tuple[str, int, str | None]:
            async with sem:
                try:
                    items = await rest.get_funding_rate(
                        symbol=symbol,
                        limit=funding_limit,
                    )
                    return symbol, len(items or []), None
                except Exception as exc:
                    if self._is_derivative_symbol_unavailable_error(exc):
                        self._disable_derivative_snapshot_symbol(symbol, exc)
                    logger.exception(
                        "Startup funding warmup failed | exchange=%s symbol=%s",
                        warmup_exchange,
                        symbol,
                    )
                    return symbol, 0, str(exc)

        # Candle warmup is shared by price_action and liquidity. REST emits
        # market.candles.snapshot; CandlesCache turns it into market.candles.updated;
        # price_action/liquidity analytics consume those updates before strategy starts.
        kline_jobs = [(symbol, timeframe) for symbol in symbols for timeframe in timeframes]
        for start in range(0, len(kline_jobs), batch_size):
            batch = kline_jobs[start : start + batch_size]
            results = await asyncio.gather(
                *(warmup_klines(symbol, timeframe) for symbol, timeframe in batch),
                return_exceptions=False,
            )
            for symbol, timeframe, count, error in results:
                if count > 0:
                    kline_success += 1
                else:
                    kline_failed.append(f"{symbol}:{timeframe}:{error or 'empty'}")
            await self._evaluate_market_state_once(reason="startup_warmup_klines_batch")
            await self._wait_event_bus_idle(timeout=self.settings.startup_warmup_eventbus_idle_timeout)

        # Funding warmup gives FundingAnalyzer enough samples for statistics/regime.
        for start in range(0, len(symbols), batch_size):
            batch = symbols[start : start + batch_size]
            results = await asyncio.gather(
                *(warmup_funding(symbol) for symbol in batch),
                return_exceptions=False,
            )
            for symbol, count, error in results:
                if count > 0:
                    funding_success += 1
                else:
                    funding_failed.append(f"{symbol}:{error or 'empty'}")
            await self._evaluate_market_state_once(reason="startup_warmup_funding_batch")
            await self._wait_event_bus_idle(timeout=self.settings.startup_warmup_eventbus_idle_timeout)

        await self._evaluate_market_state_once(reason="startup_warmup_complete")
        await self._wait_event_bus_idle(timeout=self.settings.startup_warmup_eventbus_idle_timeout)

        if self.settings.startup_warmup_required:
            if kline_jobs and kline_success <= 0:
                raise RuntimeError(
                    "Startup warmup failed: no historical candles were loaded for price_action/liquidity"
                )

            if symbols and funding_success <= 0:
                logger.warning(
                    "Startup warmup loaded no historical funding records; continuing because funding warmup is optional | symbols=%s failed_examples=%s",
                    len(symbols),
                    funding_failed[:20],
                )

        logger.info(
            "Startup warmup completed | kline_success=%s kline_failed=%s funding_success=%s funding_failed=%s disabled_symbols=%s",
            kline_success,
            len(kline_failed),
            funding_success,
            len(funding_failed),
            len(self._derivative_snapshot_disabled_symbols),
        )

        if kline_failed:
            logger.warning(
                "Startup candle warmup had partial failures | failed_count=%s examples=%s",
                len(kline_failed),
                kline_failed[:20],
            )
        if funding_failed:
            logger.warning(
                "Startup funding warmup had partial failures | failed_count=%s examples=%s",
                len(funding_failed),
                funding_failed[:20],
            )

    def _verify_liquidation_ws_capability(self) -> None:
        if not self.settings.enable_market_data or not self.settings.enable_analytics:
            return

        configured: list[tuple[str, list[str]]] = []
        missing: list[str] = []

        for name, client in self.ws_clients.items():
            ws_config = getattr(client, "_ws_config", None) or getattr(client, "ws_config", None) or getattr(client, "config", None)
            streams = list(getattr(ws_config, "streams", []) or [])
            configured.append((name, [str(stream) for stream in streams]))

            lowered = {str(stream).lower() for stream in streams}
            if (
                name.startswith("binance")
                and self.settings.binance_liquidation_stream_name.lower() not in lowered
            ):
                missing.append(name)
            if (
                name.startswith("bybit")
                and self.settings.bybit_liquidation_stream_name.lower() not in lowered
            ):
                missing.append(name)

        logger.info(
            "Liquidation WS capability checked | clients=%s missing_liquidation_stream=%s",
            configured,
            missing,
        )

        if missing:
            logger.warning(
                "Some WS clients are missing liquidation streams; liquidation analytics will be silent for those shards | clients=%s",
                missing,
            )

    def _derivative_snapshot_poll_interval_seconds(self) -> float:
        interval = float(self.settings.derivative_snapshot_poll_interval_seconds)
        minimum = float(self.settings.derivative_snapshot_min_interval_seconds)
        if interval < minimum:
            logger.warning(
                "DERIVATIVE_SNAPSHOT_POLL_INTERVAL_SECONDS=%s is too low; clamping to %s",
                interval,
                minimum,
            )
            return minimum
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

    def _derivative_snapshot_symbols(self, universe: Any, exchange: str | None = None) -> list[str]:
        source_exchange = exchange or self.settings.derivative_snapshot_exchange
        symbols = [
            str(symbol).upper()
            for symbol in getattr(universe, source_exchange, [])
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

    def _derivative_snapshot_poll_concurrency(self) -> int:
        return max(1, int(self.settings.derivative_snapshot_poll_concurrency))

    def _derivative_snapshot_poll_batch_size(self) -> int:
        return max(1, int(self.settings.derivative_snapshot_poll_batch_size))

    def _start_derivative_snapshot_polling(self, universe: Any) -> None:
        """
        Start live fixed-rate REST polling for derivative-only market snapshots.

        This is intentionally not a Scheduler job. Open interest/funding snapshots
        are market-data inputs, and they must not drift behind maintenance jobs or
        wait for fixed-delay scheduler semantics.
        """
        if self._derivative_snapshot_poll_task is not None and not self._derivative_snapshot_poll_task.done():
            return

        if not self.settings.enable_market_data or not self.settings.enable_analytics:
            return

        snapshot_exchange = self.settings.derivative_snapshot_exchange
        if snapshot_exchange not in self.settings.market_data_exchanges:
            return

        rest = self.rest_clients.get(snapshot_exchange)
        if rest is None:
            logger.warning(
                "Derivative snapshot polling disabled: %s REST client is missing",
                snapshot_exchange,
            )
            return

        symbols = self._derivative_snapshot_symbols(universe, exchange=snapshot_exchange)
        if not symbols:
            logger.warning("Derivative snapshot polling disabled: %s universe is empty", snapshot_exchange)
            return

        interval = self._derivative_snapshot_poll_interval_seconds()
        concurrency = self._derivative_snapshot_poll_concurrency()
        batch_size = self._derivative_snapshot_poll_batch_size()
        stop_event = asyncio.Event()
        self._derivative_snapshot_poll_stop = stop_event

        async def poll_symbol(symbol: str, sem: asyncio.Semaphore) -> None:
            async with sem:
                try:
                    await rest.get_open_interest(symbol=symbol)
                except Exception as exc:
                    if self._is_derivative_symbol_unavailable_error(exc):
                        self._disable_derivative_snapshot_symbol(symbol, exc)
                        return
                    logger.exception(
                        "Failed to poll derivative open interest snapshot | exchange=%s symbol=%s",
                        snapshot_exchange,
                        symbol,
                    )

                try:
                    await rest.get_funding_rate(symbol=symbol, limit=self.settings.derivative_snapshot_funding_limit)
                except Exception as exc:
                    if self._is_derivative_symbol_unavailable_error(exc):
                        self._disable_derivative_snapshot_symbol(symbol, exc)
                        return
                    logger.exception(
                        "Failed to poll derivative funding snapshot | exchange=%s symbol=%s",
                        snapshot_exchange,
                        symbol,
                    )

        async def poll_once() -> None:
            active_symbols = [
                symbol
                for symbol in symbols
                if symbol not in self._derivative_snapshot_disabled_symbols
            ]

            if not active_symbols:
                logger.warning(
                    "Derivative snapshot polling skipped: all derivative snapshot symbols are disabled | original_symbols=%s",
                    len(symbols),
                )
                return

            sem = asyncio.Semaphore(concurrency)
            for start in range(0, len(active_symbols), batch_size):
                if stop_event.is_set():
                    return
                batch = active_symbols[start : start + batch_size]
                await asyncio.gather(*(poll_symbol(symbol, sem) for symbol in batch))
                await self._evaluate_market_state_once(reason="derivative_snapshot_poll_batch")

        async def fixed_rate_loop() -> None:
            next_run = asyncio.get_running_loop().time()
            logger.info(
                "Derivative snapshot polling loop started | exchange=%s symbols=%s interval=%s concurrency=%s batch_size=%s",
                snapshot_exchange,
                len(symbols),
                interval,
                concurrency,
                batch_size,
            )

            while not stop_event.is_set():
                now = asyncio.get_running_loop().time()
                sleep_for = next_run - now
                if sleep_for > 0:
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=sleep_for)
                        break
                    except asyncio.TimeoutError:
                        pass

                started_at = asyncio.get_running_loop().time()
                try:
                    await poll_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Derivative snapshot polling tick failed")

                # Fixed-rate schedule: advance from the previous planned tick, not
                # from completion time. If we are behind, skip missed ticks instead
                # of accumulating backlog.
                next_run += interval
                now = asyncio.get_running_loop().time()
                while next_run <= now:
                    next_run += interval

                duration = now - started_at
                if duration > interval:
                    logger.warning(
                        "Derivative snapshot polling tick exceeded interval | duration=%.3f interval=%.3f",
                        duration,
                        interval,
                    )

            logger.info("Derivative snapshot polling loop stopped")

        self._derivative_snapshot_poll_task = asyncio.create_task(
            fixed_rate_loop(),
            name=f"{snapshot_exchange}-derivative-snapshots-poll",
        )
        self._derivative_snapshot_poll_job_id = f"{snapshot_exchange}-derivative-snapshots-poll"

    async def _stop_derivative_snapshot_polling(self) -> None:
        task = self._derivative_snapshot_poll_task
        stop_event = self._derivative_snapshot_poll_stop

        self._derivative_snapshot_poll_job_id = None
        self._derivative_snapshot_poll_task = None
        self._derivative_snapshot_poll_stop = None

        if stop_event is not None:
            stop_event.set()

        if task is None:
            return

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        logger.info("Derivative snapshot polling stopped")

    async def stop(self) -> None:
        await self._stop_derivative_snapshot_polling()

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

        if self._event_flow_monitor is not None:
            try:
                await self._event_flow_monitor.stop()
            except Exception:
                logger.exception("Failed to stop event flow monitor")
            finally:
                self._event_flow_monitor = None

        if self.market_scheduler is not None:
            try:
                await stop_component(self.market_scheduler)
            except Exception:
                logger.exception("Failed to stop market scheduler")

        try:
            await self.scheduler.stop(
                wait_running_jobs=self.config.scheduler.wait_running_jobs_on_shutdown,
                timeout=self.config.scheduler.graceful_shutdown_timeout,
            )
        except Exception:
            logger.exception("Failed to stop scheduler")

        if self._event_flow_monitor is not None:
            try:
                await self._event_flow_monitor.stop()
            except Exception:
                logger.exception("Failed to stop event flow monitor")
            self._event_flow_monitor = None

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