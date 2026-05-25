from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import time
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
from data.market_restore import MarketRestoreConfig, MarketStateRestorer
from data.market_models import MarketScope


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
        "STARTUP_WARMUP_FORCE_INGEST_PERSIST",
        "STARTUP_MARKET_STATE_EVALUATE_MAX_PASSES",
        "STARTUP_PARQUET_LOAD_ENABLED",
        "STARTUP_PARQUET_LOAD_REQUIRED",
        "STARTUP_PARQUET_LOAD_DAYS",
        "STARTUP_PARQUET_LOAD_CANDLES",
        "STARTUP_PARQUET_LOAD_TRADES",
        "STARTUP_PARQUET_LOAD_ORDERBOOK_SNAPSHOTS",
        "STARTUP_PARQUET_LOAD_FUNDING",
        "STARTUP_PARQUET_LOAD_OPEN_INTEREST",
        "STARTUP_PARQUET_LOAD_LIQUIDATIONS",
        "STARTUP_PARQUET_LOAD_BATCH_SIZE",
        "MARKET_INGESTION_EMIT_PERSISTABLE_EVENTS",
        "MARKET_INGESTION_PERSIST_TRADES",
        "MARKET_INGESTION_PERSIST_ORDERBOOK_SNAPSHOTS",
        "MARKET_INGESTION_SUPPRESS_BATCH_CANDLE_EVENTS",
        "TELEGRAM_BOT_QUEUE_ENABLED",
        "TELEGRAM_BOT_QUEUE_MAX_SIZE",
        "TELEGRAM_BOT_QUEUE_WORKER_COUNT",
        "TELEGRAM_BOT_QUEUE_FULL_POLICY",
        "STARTUP_PARQUET_LOAD_EVALUATE_AFTER_LOAD",
        "MARKET_STATE_MAX_CANDLES_PER_SCOPE",
        "MARKET_SCHEDULER_INTERVAL_SECONDS",
        "MARKET_SCHEDULER_BATCH_SIZE",
        "MARKET_SCHEDULER_SNAPSHOT_DEPTH",
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


def _component_flag(component: Any, *names: str) -> bool:
    """Best-effort lifecycle flag reader for mixed legacy/new components."""
    for name in names:
        value = getattr(component, name, None)
        if isinstance(value, bool):
            return value
        if callable(value):
            try:
                result = value()
            except TypeError:
                continue
            if isinstance(result, bool):
                return result
    return False


async def _start_analytics_component(component: Any) -> None:
    """
    Start analytics components safely under the state-driven runtime.

    Old analytics classes are not consistent: some implement only register(),
    some implement start() but do not call register(), and newer services call
    register() inside start().  The app bootstrap must not assume any one style.

    Correct lifecycle for analytics is always:
        register() once, then start() once when available.

    This guarantees EventBus subscriptions and Scheduler jobs are installed for
    Spoofing, Spreads, Funding/OI, Liquidations, Liquidity,
    Orderflow and Whales without putting domain logic into app.
    """
    has_register = callable(getattr(component, "register", None))
    has_start = callable(getattr(component, "start", None))

    if has_register and not _component_flag(component, "is_registered", "registered", "_registered"):
        await register_component(component)
        logger.debug(
            "Analytics component registered | component=%s",
            component.__class__.__name__,
        )

    if has_start and not _component_flag(component, "is_started", "started", "_started"):
        await start_component(component)
        logger.debug(
            "Analytics component started | component=%s",
            component.__class__.__name__,
        )
        return

    if has_register:
        logger.debug(
            "Analytics component has no start() or is already started; register() completed | component=%s",
            component.__class__.__name__,
        )
        return

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

        # ------------------------------------------------------------------
        # State-driven market-data foundation
        # ------------------------------------------------------------------
        # Use local non-null variables so type-checkers know these objects exist.
        # Instance attributes stay Optional because they are lifecycle-managed.
        market_state_store = build_market_state_store(
            self.config,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            settings=self.settings,
        )
        self.market_state_store = market_state_store

        market_ingestion = build_market_ingestion_service(
            config=self.config,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            market_state_store=market_state_store,
            settings=self.settings,
        )
        self.market_ingestion = market_ingestion

        market_scheduler = build_market_scheduler(
            config=self.config,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            market_state_store=market_state_store,
            settings=self.settings,
        )
        self.market_scheduler = market_scheduler

        # ------------------------------------------------------------------
        # REST clients
        # ------------------------------------------------------------------
        # Needed for:
        # - production market-data discovery/warmup/derivative snapshots;
        # - separated Binance execution/testnet client.
        self.rest_clients = build_rest_clients(
            self.config,
            self.event_bus,
            self.scheduler,
            self.settings,
            market_ingestion=market_ingestion,
        )

        for rest in self._unique_components(self.rest_clients.values()):
            await register_component(rest)
            await start_component(rest)

        # Optional but strongly recommended diagnostic after responsibility split.
        # Add this helper to TradingSystemRuntime, or comment this call if not added.
        log_rest_roles = getattr(self, "_log_rest_client_roles", None)
        if callable(log_rest_roles):
            log_rest_roles()

        universe = await discover_exchange_universe(
            self._market_data_rest_clients(),
            self.settings,
        )
        logger.info(
            "Universe discovered | binance=%s bybit=%s okx=%s mexc=%s canonical=%s",
            len(universe.binance),
            len(universe.bybit),
            len(universe.okx),
            len(universe.mexc),
            len(universe.all_canonical_symbols()),
        )

        # ------------------------------------------------------------------
        # Market data infrastructure
        # ------------------------------------------------------------------
        if self.settings.enable_market_data:
            self.ws_clients = build_exchange_ws_clients(
                config=self.config,
                event_bus=self.event_bus,
                scheduler=self.scheduler,
                universe=universe,
                settings=self.settings,
                market_ingestion=market_ingestion,
            )

            self.caches = build_data_caches(
                self.config,
                self.event_bus,
                self.scheduler,
                market_state_store=market_state_store,
                market_ingestion=market_ingestion,
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
                market_state_store=market_state_store,
                market_ingestion=market_ingestion,
                market_scheduler=market_scheduler,
            )
            self._market_stream = market_stream
            self.components.append(market_stream)

            # Register market stream now, but intentionally do not start WS clients yet.
            # Live market starts only after analytics/strategy/risk/execution are ready.
            await register_component(market_stream)

        # ------------------------------------------------------------------
        # Analytics
        # ------------------------------------------------------------------
        if self.settings.enable_analytics:
            analytics_components = build_analytics_components(
                config=self.config,
                event_bus=self.event_bus,
                scheduler=self.scheduler,
                caches=self.caches,
                universe=universe,
                settings=self.settings,
                market_state_store=market_state_store,
                market_scheduler=market_scheduler,
            )

            for component in analytics_components:
                self.components.append(component)
                await _start_analytics_component(component)

            # Start snapshot evaluation before live stream start.
            await start_component(market_scheduler)

        # ------------------------------------------------------------------
        # Strategy
        # ------------------------------------------------------------------
        if self.settings.enable_strategy:
            strategy_engine = build_strategy_engine(
                event_bus=self.event_bus,
                scheduler=self.scheduler,
                universe=universe,
                settings=self.settings,
            )
            self.components.append(strategy_engine)
            await start_component(strategy_engine)

        # ------------------------------------------------------------------
        # Risk
        # ------------------------------------------------------------------
        if self.settings.enable_risk:
            risk_manager = build_risk_manager(self.event_bus, self.scheduler)
            self.components.append(risk_manager)
            await start_component(risk_manager)

        # ------------------------------------------------------------------
        # Execution
        # ------------------------------------------------------------------
        if self.settings.enable_execution:
            # Must return binance_execution only. Do not allow fallback to market-data REST.
            execution_rest = self._execution_rest_client("binance")

            execution_components = build_execution_components(
                event_bus=self.event_bus,
                scheduler=self.scheduler,
                binance_rest=execution_rest,
                settings=self.settings,
            )

            for component in execution_components:
                self.components.append(component)
                await start_component(component)

        # ------------------------------------------------------------------
        # Telegram observer
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Live market data
        # ------------------------------------------------------------------
        # Start live market-data publishers only after consumers are active.
        if self._market_stream is not None:
            await start_component(self._market_stream)
            self._verify_liquidation_ws_capability()

        # ------------------------------------------------------------------
        # Derivative snapshots
        # ------------------------------------------------------------------
        # Binance USD-M open interest/funding are REST snapshot endpoints.
        if self.settings.enable_analytics:
            self._start_derivative_snapshot_polling(universe)

        # ------------------------------------------------------------------
        # News
        # ------------------------------------------------------------------
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
        """Return timeframes used during startup history hydration."""
        configured: list[str] = []
        configured.extend(str(tf).strip() for tf in self.settings.startup_warmup_timeframes if str(tf).strip())

        if not configured:
            configured = [str(tf).strip() for tf in self.settings.timeframes if str(tf).strip()]

        seen: set[str] = set()
        result: list[str] = []
        for timeframe in configured:
            if timeframe in seen:
                continue
            seen.add(timeframe)
            result.append(timeframe)
        return result


    @staticmethod
    def _unique_components(components: Any) -> list[Any]:
        """Return unique object instances while preserving insertion order."""
        unique: list[Any] = []
        seen_ids: set[int] = set()
        for component in components:
            ident = id(component)
            if ident in seen_ids:
                continue
            seen_ids.add(ident)
            unique.append(component)
        return unique

    def _market_data_rest_client(self, exchange: str) -> Any | None:
        """Return the REST client that is allowed to fetch production market data."""
        exchange = str(exchange).lower()
        if exchange == "binance":
            return self.rest_clients.get("binance_market_data") or self.rest_clients.get("binance")
        return self.rest_clients.get(exchange)

    def _execution_rest_client(self, exchange: str) -> Any:
        """Return the REST client that is allowed to touch execution/account endpoints."""
        exchange = str(exchange).lower()

        if exchange == "binance":
            client = self.rest_clients.get("binance_execution")
            if client is None:
                available = sorted(self.rest_clients.keys())
                raise RuntimeError(
                    "Binance execution REST client is missing. "
                    "Execution must not fall back to market-data REST client. "
                    f"Available REST clients: {available}"
                )
            return client

        client = self.rest_clients.get(exchange)
        if client is None:
            available = sorted(self.rest_clients.keys())
            raise RuntimeError(
                f"Execution REST client is missing | exchange={exchange} available={available}"
            )
        return client

    def _market_data_rest_clients(self) -> dict[str, Any]:
        """Mapping expected by universe discovery: exchange name -> market-data REST client."""
        result: dict[str, Any] = {}
        for exchange in self.settings.market_data_exchanges:
            client = self._market_data_rest_client(exchange)
            if client is not None:
                result[str(exchange).lower()] = client
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
        """Drain bounded state-snapshot analytics ticks if the state scheduler exists.

        A single MarketScheduler tick is intentionally bounded by batch_size.
        During startup warmup/restores a REST batch can mark more scopes dirty
        than that limit, so one tick can leave analytics scopes
        unprocessed.  Drain until the scheduler reports no snapshots or a small
        safety limit is reached.
        """
        if self.market_scheduler is None:
            return
        evaluate = getattr(self.market_scheduler, "evaluate_dirty_once", None)
        if not callable(evaluate):
            return
        max_passes = max(1, int(getattr(self.settings, "startup_market_state_evaluate_max_passes", 25)))
        try:
            for attempt in range(max_passes):
                result = evaluate()
                if inspect.isawaitable(result):
                    result = await result
                logger.debug(
                    "Market state evaluated | reason=%s pass=%s/%s result=%s",
                    reason,
                    attempt + 1,
                    max_passes,
                    result,
                )
                if not isinstance(result, dict):
                    break
                if result.get("skipped"):
                    break
                if int(result.get("snapshots") or 0) <= 0:
                    break
        except Exception:
            logger.exception("Market state evaluation failed | reason=%s", reason)

    async def _warmup_analytics_from_market_state(self) -> None:
        """Best-effort explicit warmup hook for analyzers that can rebuild from MarketStateStore.

        State-driven analyzers normally receive snapshots through MarketScheduler.
        Some domain analyzers may also expose an
        explicit warmup_from_market_state() method to rebuild child-module state
        from restored candles before strategy/risk/execution are started.
        """
        warmed = 0
        skipped = 0

        for component in self.components:
            warmup = getattr(component, "warmup_from_market_state", None)
            if not callable(warmup):
                continue

            try:
                result = warmup()
                if inspect.isawaitable(result):
                    result = await result

                warmed += 1
                logger.info(
                    "Analytics component warmed from MarketStateStore | component=%s result=%s",
                    component.__class__.__name__,
                    result,
                )
            except Exception:
                skipped += 1
                logger.exception(
                    "Analytics component warmup_from_market_state failed | component=%s",
                    component.__class__.__name__,
                )

        logger.info(
            "Analytics MarketState explicit warmup completed | warmed=%s failed=%s",
            warmed,
            skipped,
        )

    @staticmethod
    def _first_present(payload: Any, *keys: str) -> Any:
        if not isinstance(payload, dict):
            return None
        for key in keys:
            if key in payload and payload.get(key) is not None:
                return payload.get(key)
        return None

    def _normalize_startup_candle_payload(
        self,
        candle: Any,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
    ) -> dict[str, Any] | None:
        """Normalize REST kline rows before they are forced through ingestion.

        Binance adapters in this project have existed in both forms: normalized
        dicts and raw list/tuple klines.  Startup must not depend on the adapter
        having a hidden MarketIngestion side effect, so this helper converts both
        shapes into the CandleUpdate.from_payload contract.
        """
        if isinstance(candle, dict):
            open_time = self._first_present(candle, "open_time_ms", "open_time", "timestamp_ms", "timestamp", "start", "t")
            close_time = self._first_present(candle, "close_time_ms", "close_time", "end", "T")
            open_price = self._first_present(candle, "open", "o")
            high = self._first_present(candle, "high", "h")
            low = self._first_present(candle, "low", "l")
            close = self._first_present(candle, "close", "c")
            volume = self._first_present(candle, "volume", "v")
            is_closed = self._first_present(candle, "is_closed", "closed", "x")
            timestamp = self._first_present(candle, "timestamp_ms", "event_time", "timestamp") or close_time or open_time
            exchange_symbol = self._first_present(candle, "exchange_symbol", "raw_symbol", "symbol") or symbol
            source = self._first_present(candle, "source") or "startup_rest_warmup"
        elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
            # Binance /fapi/v1/klines raw shape:
            # [open_time, open, high, low, close, volume, close_time, ...]
            open_time = candle[0]
            open_price = candle[1]
            high = candle[2]
            low = candle[3]
            close = candle[4]
            volume = candle[5]
            close_time = candle[6] if len(candle) > 6 else None
            is_closed = True
            timestamp = close_time or open_time
            exchange_symbol = symbol
            source = "startup_rest_warmup"
        else:
            return None

        try:
            open_ms = int(float(open_time))
        except (TypeError, ValueError):
            return None

        timeframe_ms = self._timeframe_to_ms(timeframe)
        try:
            close_ms = int(float(close_time)) if close_time is not None else open_ms + timeframe_ms - 1
        except (TypeError, ValueError):
            close_ms = open_ms + timeframe_ms - 1

        normalized = {
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "exchange_symbol": exchange_symbol,
            "timeframe": timeframe,
            "open_time_ms": open_ms,
            "close_time_ms": close_ms,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume if volume is not None else 0.0,
            "is_closed": bool(is_closed if is_closed is not None else True),
            "timestamp_ms": timestamp,
            "source": source,
            "metadata": {
                "source": "startup_rest_warmup",
                "startup_warmup": True,
            },
        }
        return normalized

    async def _ingest_startup_candles(
        self,
        candles: Any,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
        suppress_persistable_events: bool | None = None,
    ) -> int:
        """Force REST warmup candles into the shared MarketStateStore.

        Some exchange REST adapters already ingest klines as a side effect, while
        others only return rows. Re-ingesting normalized candles is safe for
        MarketStateStore because candles are keyed by open_time_ms.
        """
        if self.market_ingestion is None or not candles:
            return 0

        items: list[Any]
        if isinstance(candles, dict):
            raw_items = candles.get("candles") or candles.get("items") or candles.get("data") or []
            items = list(raw_items) if isinstance(raw_items, (list, tuple)) else []
        elif isinstance(candles, (list, tuple)):
            items = list(candles)
        else:
            return 0

        normalized: list[dict[str, Any]] = []
        for candle in items:
            payload = self._normalize_startup_candle_payload(
                candle,
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
            if payload is not None:
                normalized.append(payload)

        if not normalized:
            return 0

        ingest = getattr(self.market_ingestion, "ingest_candles_batch", None)
        if not callable(ingest):
            logger.warning("Startup candle ingestion skipped: MarketIngestionService has no ingest_candles_batch")
            return 0

        # Default to emitting persistable events when startup persistence is enabled.
        # Operators can turn it off with STARTUP_WARMUP_FORCE_INGEST_PERSIST=false.
        if suppress_persistable_events is None:
            suppress_persistable_events = not bool(getattr(self.settings, "startup_warmup_force_ingest_persist", True))

        result = ingest(
            normalized,
            source="startup_rest_warmup",
            suppress_persistable_events=suppress_persistable_events,
        )
        if inspect.isawaitable(result):
            result = await result

        try:
            return int(result or 0)
        except (TypeError, ValueError):
            return len(normalized)

    async def _market_state_candle_count(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
    ) -> dict[str, Any]:
        """Read candle availability from MarketStateStore for diagnostics/strict gates."""
        if self.market_state_store is None:
            return {"ok": False, "reason": "market_state_store_missing", "candle_count": 0, "closed_candles": 0}

        try:
            snapshot = await self.market_state_store.snapshot(
                MarketScope(exchange=exchange, market_type=market_type, symbol=symbol, timeframe=timeframe),
                depth=getattr(self.settings, "market_scheduler_snapshot_depth", None),
            )
        except Exception as exc:
            logger.exception(
                "MarketState candle diagnostics failed | exchange=%s market_type=%s symbol=%s timeframe=%s",
                exchange,
                market_type,
                symbol,
                timeframe,
            )
            return {"ok": False, "reason": str(exc), "candle_count": 0, "closed_candles": 0}

        window = getattr(snapshot, "candles", {}).get(timeframe)
        candles = tuple(getattr(window, "candles", ()) or ()) if window is not None else ()
        closed = [item for item in candles if bool(getattr(item, "is_closed", False))]
        return {
            "ok": bool(closed),
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "candle_count": int(getattr(window, "candle_count", len(candles)) or 0) if window is not None else 0,
            "closed_candles": len(closed),
            "first_open_time_ms": getattr(candles[0], "open_time_ms", None) if candles else None,
            "last_open_time_ms": getattr(candles[-1], "open_time_ms", None) if candles else None,
            "last_close_time_ms": getattr(candles[-1], "close_time_ms", None) if candles else None,
        }

    @staticmethod
    def _timeframe_to_ms(timeframe: str) -> int:
        value = str(timeframe or "1m").strip().lower()
        aliases = {
            "1": "1m",
            "3": "3m",
            "5": "5m",
            "15": "15m",
            "30": "30m",
            "60": "1h",
            "min1": "1m",
            "min5": "5m",
            "min15": "15m",
            "hour1": "1h",
            "day1": "1d",
        }
        value = aliases.get(value, value)
        if len(value) < 2:
            return 60_000

        unit = value[-1]
        try:
            amount = int(value[:-1])
        except ValueError:
            return 60_000

        if unit == "m":
            return amount * 60_000
        if unit == "h":
            return amount * 60 * 60_000
        if unit == "d":
            return amount * 24 * 60 * 60_000
        if unit == "w":
            return amount * 7 * 24 * 60 * 60_000
        return 60_000


    def _startup_required_candles_for_timeframe(self, timeframe: str) -> int:
        days = float(getattr(self.settings, "startup_warmup_days", 0.0) or 0.0)
        if days <= 0:
            return max(1, int(getattr(self.settings, "startup_warmup_kline_limit", 1500)))
        timeframe_ms = max(1, self._timeframe_to_ms(timeframe))
        window_ms = int(days * 24 * 60 * 60 * 1000)
        # +2 gives a small edge buffer for inclusive/exclusive REST ranges.
        return max(1, (window_ms + timeframe_ms - 1) // timeframe_ms + 2)

    def _startup_min_acceptable_candles(self, timeframe: str) -> int:
        required = self._startup_required_candles_for_timeframe(timeframe)
        return max(1, int(required * 0.90))
    def _startup_warmup_start_ms(self) -> int | None:
        days = float(getattr(self.settings, "startup_warmup_days", 0.0) or 0.0)
        if days <= 0:
            return None
        return int(time.time() * 1000) - int(days * 24 * 60 * 60 * 1000)

    def _startup_parquet_load_enabled(self) -> bool:
        return (
            bool(getattr(self.settings, "startup_parquet_load_enabled", True))
            and self.settings.enable_market_data
            and self.settings.enable_analytics
            and self.market_ingestion is not None
        )

    def _parquet_root_dir(self) -> str:
        storage = getattr(self.config, "storage", None)
        root_dir = getattr(storage, "parquet_dir", None)
        if root_dir is None:
            root_dir = getattr(storage, "data_dir", None)
        if root_dir is None:
            app = getattr(self.config, "app", None)
            data_dir = getattr(app, "data_dir", None)
            if data_dir is not None:
                root_dir = str(Path(data_dir) / "parquet")
        return str(root_dir or "data/parquet")

    async def _load_startup_parquet_history(self, universe: Any) -> None:
        """Load persisted candles/funding back into MarketStateStore before REST/live data."""
        if not self._startup_parquet_load_enabled():
            logger.info(
                "Startup Parquet load skipped | enabled=%s market_data=%s analytics=%s ingestion=%s",
                getattr(self.settings, "startup_parquet_load_enabled", None),
                self.settings.enable_market_data,
                self.settings.enable_analytics,
                self.market_ingestion is not None,
            )
            return

        symbols = self._derivative_snapshot_symbols(
            universe,
            exchange=self.settings.startup_warmup_exchange,
        )
        if not symbols:
            message = "Startup Parquet load skipped: universe is empty"
            if getattr(self.settings, "startup_parquet_load_required", False):
                raise RuntimeError(message)
            logger.warning(message)
            return

        days = float(getattr(self.settings, "startup_parquet_load_days", 7.0) or 0.0)
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - int(days * 24 * 60 * 60 * 1000) if days > 0 else None
        batch_size = max(1, int(getattr(self.settings, "startup_parquet_load_batch_size", 1000)))

        restorer = MarketStateRestorer(
            market_ingestion=self.market_ingestion,
            market_scheduler=self.market_scheduler,
            event_bus=self.event_bus,
            config=MarketRestoreConfig(
                root_dir=self._parquet_root_dir(),
                default_exchange=self.settings.startup_warmup_exchange,
                default_market_type=self.settings.analytics_market_type,
                batch_size=batch_size,
                evaluate_after_restore=bool(getattr(self.settings, "startup_parquet_load_evaluate_after_load", True)),
                restore_candles=bool(getattr(self.settings, "startup_parquet_load_candles", True)),
                restore_trades=bool(getattr(self.settings, "startup_parquet_load_trades", False)),
                restore_orderbook_snapshots=bool(getattr(self.settings, "startup_parquet_load_orderbook_snapshots", True)),
                restore_funding=bool(getattr(self.settings, "startup_parquet_load_funding", True)),
                restore_open_interest=bool(getattr(self.settings, "startup_parquet_load_open_interest", True)),
                restore_liquidations=bool(getattr(self.settings, "startup_parquet_load_liquidations", True)),
                suppress_persistable_on_replay=True,
            ),
        )

        logger.info(
            "Startup Parquet restore started | root_dir=%s exchange=%s symbols=%s timeframes=%s days=%s start_ms=%s end_ms=%s candles=%s trades=%s orderbook_snapshots=%s funding=%s open_interest=%s liquidations=%s batch_size=%s",
            self._parquet_root_dir(),
            self.settings.startup_warmup_exchange,
            len(symbols),
            self._startup_warmup_timeframes(),
            days,
            start_ms,
            end_ms,
            getattr(self.settings, "startup_parquet_load_candles", True),
            getattr(self.settings, "startup_parquet_load_trades", False),
            getattr(self.settings, "startup_parquet_load_orderbook_snapshots", True),
            getattr(self.settings, "startup_parquet_load_funding", True),
            getattr(self.settings, "startup_parquet_load_open_interest", True),
            getattr(self.settings, "startup_parquet_load_liquidations", True),
            batch_size,
        )

        try:
            result = await restorer.restore(
                exchange=self.settings.startup_warmup_exchange,
                market_type=self.settings.analytics_market_type,
                symbols=symbols,
                timeframes=self._startup_warmup_timeframes(),
                start_ms=start_ms,
                end_ms=end_ms,
            )

            result_dict = result.to_dict() if hasattr(result, "to_dict") else result
            logger.info("Startup Parquet restore completed | result=%s", result_dict)

            rows_loaded = int(getattr(result, "rows_loaded", 0) or 0)
            candle_rows = 0
            if hasattr(result, "datasets") and isinstance(result.datasets, dict):
                candles = result.datasets.get("candles", {})
                if isinstance(candles, dict):
                    candle_rows = int(candles.get("rows_loaded") or 0)

            if getattr(self.settings, "startup_parquet_load_required", False) and candle_rows <= 0:
                raise RuntimeError(f"Startup Parquet restore required but no candles were loaded | result={result_dict}")

            if rows_loaded > 0:
                await self._wait_event_bus_idle(timeout=self.settings.startup_warmup_eventbus_idle_timeout)

        except Exception:
            if getattr(self.settings, "startup_parquet_load_required", False):
                raise
            logger.exception("Startup Parquet load failed; continuing with REST warmup")

    async def _run_startup_warmup(self, universe: Any) -> None:
        """
        Load the minimum historical context required by analytics before the
        trading pipeline starts.

        Startup order is intentional:
        - EventBus/Scheduler/REST are running.
        - Caches are registered.
        - Analytics are registered/started.
        - Strategy/risk/execution/Telegram/news are NOT started yet.

        This lets analytics build initial state without
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
        rest = self._market_data_rest_client(warmup_exchange)
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
        configured_kline_limit = max(1, int(self.settings.startup_warmup_kline_limit))
        required_kline_limits = {tf: self._startup_required_candles_for_timeframe(tf) for tf in timeframes}
        effective_kline_limit = max([configured_kline_limit, *required_kline_limits.values()]) if required_kline_limits else configured_kline_limit
        klines_per_request = max(1, min(1500, int(getattr(self.settings, "startup_warmup_klines_per_request", effective_kline_limit))))
        warmup_start_ms = self._startup_warmup_start_ms()
        warmup_end_ms = int(time.time() * 1000)
        funding_limit = max(1, int(self.settings.startup_warmup_funding_limit))
        concurrency = max(1, int(self.settings.startup_warmup_concurrency))
        batch_size = max(1, int(self.settings.startup_warmup_batch_size))

        logger.info(
            "Startup warmup started | exchange=%s symbols=%s timeframes=%s days=%s start_ms=%s end_ms=%s configured_kline_limit=%s effective_kline_limit=%s required_by_timeframe=%s klines_per_request=%s funding_limit=%s concurrency=%s batch_size=%s",
            warmup_exchange,
            len(symbols),
            timeframes,
            getattr(self.settings, "startup_warmup_days", None),
            warmup_start_ms,
            warmup_end_ms,
            configured_kline_limit,
            effective_kline_limit,
            required_kline_limits,
            klines_per_request,
            funding_limit,
            concurrency,
            batch_size,
        )

        orderbook_success = 0
        orderbook_failed: list[str] = []
        kline_success = 0
        kline_failed: list[str] = []
        funding_success = 0
        funding_failed: list[str] = []
        sem = asyncio.Semaphore(concurrency)

        async def warmup_orderbook(symbol: str) -> tuple[str, bool, str | None]:
            async with sem:
                try:
                    limit = max(5, int(getattr(self.settings, "startup_warmup_orderbook_limit", 100)))
                    await rest.get_orderbook_snapshot(symbol=symbol, limit=limit)
                    return symbol, True, None
                except Exception as exc:
                    if self._is_derivative_symbol_unavailable_error(exc):
                        self._disable_derivative_snapshot_symbol(symbol, exc)
                    logger.exception(
                        "Startup orderbook warmup failed | exchange=%s symbol=%s",
                        warmup_exchange,
                        symbol,
                    )
                    return symbol, False, str(exc)

        async def warmup_klines(symbol: str, timeframe: str) -> tuple[str, str, int, str | None]:
            async with sem:
                try:
                    total = 0
                    seen_open_times: set[int] = set()

                    # Backward-compatible path: one request when a historical
                    # horizon is not configured. Otherwise page through the
                    # requested warmup window because Binance returns max 1500
                    # klines per request.
                    if warmup_start_ms is None:
                        candles = await rest.get_klines(
                            symbol=symbol,
                            interval=timeframe,
                            limit=effective_kline_limit,
                        )
                        loaded = len(candles or [])
                        ingested = await self._ingest_startup_candles(
                            candles,
                            exchange=warmup_exchange,
                            market_type=self.settings.analytics_market_type,
                            symbol=symbol,
                            timeframe=timeframe,
                        )
                        state_info = await self._market_state_candle_count(
                            exchange=warmup_exchange,
                            market_type=self.settings.analytics_market_type,
                            symbol=symbol,
                            timeframe=timeframe,
                        )
                        return symbol, timeframe, max(loaded, ingested, int(state_info.get("closed_candles") or 0)), None

                    timeframe_ms = self._timeframe_to_ms(timeframe)
                    cursor_ms = warmup_start_ms
                    target_limit = max(configured_kline_limit, self._startup_required_candles_for_timeframe(timeframe))
                    max_pages = max(1, (target_limit + klines_per_request - 1) // klines_per_request)

                    for _ in range(max_pages):
                        if cursor_ms >= warmup_end_ms:
                            break

                        candles = await rest.get_klines(
                            symbol=symbol,
                            interval=timeframe,
                            limit=klines_per_request,
                            start_time=cursor_ms,
                            end_time=warmup_end_ms,
                        )
                        if not candles:
                            break

                        await self._ingest_startup_candles(
                            candles,
                            exchange=warmup_exchange,
                            market_type=self.settings.analytics_market_type,
                            symbol=symbol,
                            timeframe=timeframe,
                        )

                        page_new = 0
                        last_open_ms: int | None = None
                        for candle in candles:
                            normalized_candle = self._normalize_startup_candle_payload(
                                candle,
                                exchange=warmup_exchange,
                                market_type=self.settings.analytics_market_type,
                                symbol=symbol,
                                timeframe=timeframe,
                            )
                            if normalized_candle is None:
                                continue
                            open_time = normalized_candle.get("open_time_ms")
                            try:
                                open_ms = int(float(open_time))
                            except (TypeError, ValueError):
                                continue
                            last_open_ms = open_ms
                            if open_ms not in seen_open_times:
                                seen_open_times.add(open_ms)
                                page_new += 1

                        total += page_new
                        if last_open_ms is None or page_new <= 0:
                            break

                        cursor_ms = last_open_ms + timeframe_ms
                        if len(candles) < klines_per_request:
                            break

                    state_info = await self._market_state_candle_count(
                        exchange=warmup_exchange,
                        market_type=self.settings.analytics_market_type,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                    return symbol, timeframe, max(total, int(state_info.get("closed_candles") or 0)), None
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

        # Orderbook warmup writes REST depth snapshots into MarketStateStore before
        # live WS deltas arrive. This gives spoofing/spreads valid book depth and
        # prevents delta-before-snapshot resync-only startup states.
        if bool(getattr(self.settings, "startup_warmup_orderbook_enabled", True)):
            for start in range(0, len(symbols), batch_size):
                batch = symbols[start : start + batch_size]
                results = await asyncio.gather(
                    *(warmup_orderbook(symbol) for symbol in batch),
                    return_exceptions=False,
                )
                for symbol, ok, error in results:
                    if ok:
                        orderbook_success += 1
                    else:
                        orderbook_failed.append(f"{symbol}:{error or 'empty'}")
                await self._evaluate_market_state_once(reason="startup_warmup_orderbook_batch")
                await self._wait_event_bus_idle(timeout=self.settings.startup_warmup_eventbus_idle_timeout)

        # Candle warmup writes directly into MarketStateStore through
        # MarketIngestionService. Price-action/liquidity receive restored scopes
        # through MarketScheduler snapshots before strategy/risk/execution start.
        kline_jobs = [(symbol, timeframe) for symbol in symbols for timeframe in timeframes]
        for start in range(0, len(kline_jobs), batch_size):
            batch = kline_jobs[start : start + batch_size]
            results = await asyncio.gather(
                *(warmup_klines(symbol, timeframe) for symbol, timeframe in batch),
                return_exceptions=False,
            )
            for symbol, timeframe, count, error in results:
                min_required = self._startup_min_acceptable_candles(timeframe)
                if count >= min_required:
                    kline_success += 1
                else:
                    reason = error or f"insufficient_history loaded={count} min_required={min_required}"
                    kline_failed.append(f"{symbol}:{timeframe}:{reason}")
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
        await self._flush_startup_warmup_storage()

        if self.settings.startup_warmup_required:
            if kline_jobs and kline_success <= 0:
                raise RuntimeError(
                    "Startup warmup failed: no historical candles were loaded for analytics"
                )
            if kline_failed:
                raise RuntimeError(
                    "Startup warmup failed: some candle scopes do not have enough history "
                    f"for analytics | failed_count={len(kline_failed)} examples={kline_failed[:20]}"
                )

            if symbols and funding_success <= 0:
                logger.warning(
                    "Startup warmup loaded no historical funding records; continuing because funding warmup is optional | symbols=%s failed_examples=%s",
                    len(symbols),
                    funding_failed[:20],
                )

        logger.info(
            "Startup warmup completed | orderbook_success=%s orderbook_failed=%s kline_success=%s kline_failed=%s funding_success=%s funding_failed=%s disabled_symbols=%s",
            orderbook_success,
            len(orderbook_failed),
            kline_success,
            len(kline_failed),
            funding_success,
            len(funding_failed),
            len(self._derivative_snapshot_disabled_symbols),
        )

        if orderbook_failed:
            logger.warning(
                "Startup orderbook warmup had partial failures | failed_count=%s examples=%s",
                len(orderbook_failed),
                orderbook_failed[:20],
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

        rest = self._market_data_rest_client(snapshot_exchange)
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

        for rest in reversed(self._unique_components(self.rest_clients.values())):
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
