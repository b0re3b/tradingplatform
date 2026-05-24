from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from dataclasses import dataclass
from typing import Any

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler
from data.market_ingestion import MarketIngestionService
from data.market_scheduler import MarketScheduler
from data.market_state import MarketStateStore


@dataclass(slots=True)
class MarketDataSubscription:
    """Declarative market-data subscription target for exchange adapters."""

    exchange: str
    symbol: str
    channels: tuple[str, ...] = ("trade", "orderbook", "candle")
    market_type: str = "usdm_futures"
    timeframe: str | None = None
    depth: int | None = None
    enabled: bool = True


@dataclass(slots=True)
class ExchangeRuntimeState:
    exchange: str
    is_started: bool = False
    started_at: float | None = None
    stopped_at: float | None = None
    last_error: str | None = None
    restarts: int = 0


@dataclass(slots=True)
class MarketStreamConfig:
    """Local config for the state-driven market-data orchestration service."""

    healthcheck_interval_seconds: float = 30.0
    startup_timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 15.0
    start_clients_on_start: bool = True
    start_market_scheduler_on_start: bool = True
    emit_lifecycle_events: bool = True
    wire_ingestion_into_clients: bool = True


class MarketStream:
    """
    State-driven market-data orchestrator.

    Responsibilities:
    - start/stop exchange WS clients;
    - provide/wire MarketIngestionService and MarketStateStore into clients;
    - start optional MarketScheduler for coalesced snapshot evaluation;
    - publish only low-frequency system.market_stream.* lifecycle/health events.

    It intentionally does NOT register caches as EventBus subscribers and it does
    NOT move raw market.orderbook.batch / market.trades.batch through EventBus.
    """

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        exchange_clients: dict[str, Any],
        scheduler: Scheduler | None = None,
        stream_config: MarketStreamConfig | None = None,
        caches: list[Any] | None = None,
        market_state: MarketStateStore | None = None,
        state_store: MarketStateStore | None = None,
        ingestion: MarketIngestionService | None = None,
        market_scheduler: MarketScheduler | None = None,
        service_name: str = "market_stream",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.exchange_clients = dict(exchange_clients)
        self.scheduler = scheduler
        self.stream_config = stream_config or MarketStreamConfig()
        self.caches = list(caches or [])
        self.market_state = market_state or state_store or (ingestion.state_store if ingestion is not None else MarketStateStore())
        self.ingestion = ingestion or MarketIngestionService(state_store=self.market_state, event_bus=event_bus)
        self.market_scheduler = market_scheduler
        self._service_name = service_name

        self._logger = get_logger(__name__, service=service_name, event_type="market_stream")
        self._running = False
        self._registered = False
        self._healthcheck_job_id: str | None = None
        self._states: dict[str, ExchangeRuntimeState] = {
            exchange: ExchangeRuntimeState(exchange=exchange)
            for exchange in self.exchange_clients
        }
        self._subscriptions: list[MarketDataSubscription] = []
        self._metrics: dict[str, int | float] = {
            "started_at": 0.0,
            "clients_started": 0,
            "clients_failed": 0,
            "clients_stopped": 0,
            "healthcheck_runs": 0,
            "cache_components": len(self.caches),
            "raw_eventbus_cache_subscriptions": 0,
            "state_driven": 1,
        }

    def register(self) -> None:
        """
        Register state-driven dependencies only.

        Cache registration no longer subscribes to raw market.* topics. If cache
        objects are provided, their register() methods are intentionally NOT called
        here because old cache register() implementations may subscribe to EventBus.
        The new direct-apply caches can be started separately if needed.
        """
        if self._registered:
            self._logger.debug("MarketStream already registered")
            return
        self._registered = True
        self._wire_clients()
        self._logger.info(
            "MarketStream registered in state-driven mode | exchanges=%s caches_attached=%s raw_cache_subscriptions=0",
            list(self.exchange_clients),
            len(self.caches),
        )

    async def start(self, subscriptions: list[MarketDataSubscription] | None = None) -> None:
        if self._running:
            self._logger.warning("MarketStream already started")
            return
        if not self._registered:
            self.register()

        self._running = True
        self._metrics["started_at"] = time.time()
        self._subscriptions = [sub for sub in subscriptions or [] if sub.enabled]

        if self.market_scheduler is not None and self.stream_config.start_market_scheduler_on_start:
            await self.market_scheduler.start()

        if self.stream_config.start_clients_on_start:
            await self._start_exchange_clients()

        await self._start_healthcheck()
        await self._emit_event(
            "system.market_stream.started",
            {
                "service": self._service_name,
                "mode": "state_driven",
                "exchanges": list(self.exchange_clients),
                "subscriptions": [self._serialize_subscription(s) for s in self._subscriptions],
                "caches": [cache.__class__.__name__ for cache in self.caches],
                "raw_market_eventbus_topics_disabled": True,
            },
            priority=EventPriority.NORMAL,
        )

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("MarketStream already stopped")
            return
        self._running = False
        await self._stop_healthcheck()
        await self._stop_exchange_clients()
        if self.market_scheduler is not None and self.stream_config.start_market_scheduler_on_start:
            await self.market_scheduler.stop()
        await self._emit_event(
            "system.market_stream.stopped",
            {"service": self._service_name, "mode": "state_driven", "exchanges": list(self.exchange_clients)},
            priority=EventPriority.NORMAL,
        )

    def _wire_clients(self) -> None:
        if not self.stream_config.wire_ingestion_into_clients:
            return
        for exchange, client in self.exchange_clients.items():
            for attr, value in (
                ("market_ingestion", self.ingestion),
                ("ingestion", self.ingestion),
                ("market_state", self.market_state),
                ("market_state_store", self.market_state),
                ("state_store", self.market_state),
            ):
                if hasattr(client, attr):
                    with contextlib.suppress(Exception):
                        setattr(client, attr, value)
            setter = getattr(client, "set_market_ingestion", None)
            if callable(setter):
                result = setter(self.ingestion)
                if inspect.isawaitable(result):
                    raise RuntimeError(f"set_market_ingestion must be sync for {exchange}")
            state_setter = getattr(client, "set_market_state", None)
            if callable(state_setter):
                result = state_setter(self.market_state)
                if inspect.isawaitable(result):
                    raise RuntimeError(f"set_market_state must be sync for {exchange}")

    async def _start_exchange_clients(self) -> None:
        for exchange, client in self.exchange_clients.items():
            state = self._states.setdefault(exchange, ExchangeRuntimeState(exchange=exchange))
            try:
                register = getattr(client, "register", None)
                if register is not None:
                    result = register()
                    if inspect.isawaitable(result):
                        await result

                start = getattr(client, "start", None)
                if start is None:
                    self._logger.warning("Exchange client has no start() | exchange=%s", exchange)
                    continue

                result = start()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, timeout=self.stream_config.startup_timeout_seconds)

                state.is_started = True
                state.started_at = time.time()
                state.last_error = None
                self._metrics["clients_started"] += 1
                await self._emit_event(
                    "system.market_stream.exchange_started",
                    {"exchange": exchange, "mode": "state_driven"},
                    priority=EventPriority.LOW,
                )
            except Exception as exc:
                state.is_started = False
                state.last_error = str(exc)
                self._metrics["clients_failed"] += 1
                self._logger.exception("Failed to start exchange client | exchange=%s", exchange)
                await self._emit_event(
                    "system.market_stream.exchange_start_failed",
                    {"exchange": exchange, "error": str(exc), "mode": "state_driven"},
                    priority=EventPriority.HIGH,
                )

    async def _stop_exchange_clients(self) -> None:
        for exchange, client in reversed(list(self.exchange_clients.items())):
            state = self._states.setdefault(exchange, ExchangeRuntimeState(exchange=exchange))
            try:
                stop = getattr(client, "stop", None) or getattr(client, "close", None)
                if stop is None:
                    continue
                result = stop()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, timeout=self.stream_config.shutdown_timeout_seconds)
                state.is_started = False
                state.stopped_at = time.time()
                state.last_error = None
                self._metrics["clients_stopped"] += 1
            except Exception as exc:
                state.last_error = str(exc)
                self._logger.exception("Failed to stop exchange client | exchange=%s", exchange)

    async def restart_exchange(self, exchange: str) -> None:
        client = self.exchange_clients.get(exchange)
        if client is None:
            raise KeyError(f"Unknown exchange client: {exchange}")
        stop = getattr(client, "stop", None) or getattr(client, "close", None)
        if stop is not None:
            result = stop()
            if inspect.isawaitable(result):
                await result
        self._wire_clients()
        start = getattr(client, "start", None)
        if start is None:
            raise RuntimeError(f"Exchange client has no start(): {exchange}")
        result = start()
        if inspect.isawaitable(result):
            await result
        state = self._states.setdefault(exchange, ExchangeRuntimeState(exchange=exchange))
        state.is_started = True
        state.started_at = time.time()
        state.last_error = None
        state.restarts += 1
        await self._emit_event("system.market_stream.exchange_restarted", {"exchange": exchange, "restarts": state.restarts, "mode": "state_driven"})

    async def health_check(self) -> dict[str, Any]:
        market_state_stats = await self.market_state.stats()
        return {
            "running": self._running,
            "mode": "state_driven",
            "stats": self.stats(),
            "market_state": market_state_stats,
            "ingestion": self.ingestion.stats(),
            "exchanges": {
                exchange: {
                    "is_started": state.is_started,
                    "started_at": state.started_at,
                    "stopped_at": state.stopped_at,
                    "last_error": state.last_error,
                    "restarts": state.restarts,
                }
                for exchange, state in self._states.items()
            },
            "caches": [
                cache.stats() if hasattr(cache, "stats") else {"component": cache.__class__.__name__}
                for cache in self.caches
            ],
        }

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "mode": "state_driven",
            "exchanges_total": len(self.exchange_clients),
            "exchanges_started": sum(1 for s in self._states.values() if s.is_started),
            "subscriptions_total": len(self._subscriptions),
            **self._metrics,
        }

    async def _start_healthcheck(self) -> None:
        if self.scheduler is None or self._healthcheck_job_id is not None:
            return
        self._healthcheck_job_id = self.scheduler.add_interval_job(
            name="market-stream-state-healthcheck",
            func=self._scheduled_healthcheck,
            interval=self.stream_config.healthcheck_interval_seconds,
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=10.0,
            allow_overlap=False,
            enabled=True,
        )

    async def _stop_healthcheck(self) -> None:
        if self.scheduler is None or self._healthcheck_job_id is None:
            return
        with contextlib.suppress(Exception):
            self.scheduler.remove_job(self._healthcheck_job_id)
        self._healthcheck_job_id = None

    async def _scheduled_healthcheck(self) -> None:
        self._metrics["healthcheck_runs"] += 1
        report = await self.health_check()
        unhealthy = [name for name, s in report["exchanges"].items() if not s["is_started"] or s["last_error"]]
        if unhealthy:
            await self._emit_event(
                "system.market_stream.healthcheck_warning",
                {"unhealthy_exchanges": unhealthy, "report": report},
                priority=EventPriority.NORMAL,
            )

    async def _emit_event(self, topic: str, payload: dict[str, Any], *, priority: EventPriority = EventPriority.LOW) -> None:
        if not self.stream_config.emit_lifecycle_events:
            return
        try:
            await self.event_bus.emit(topic, payload, source=self._service_name, priority=priority)
        except Exception:
            self._logger.exception("Failed to emit MarketStream event | topic=%s", topic)

    @staticmethod
    def _serialize_subscription(subscription: MarketDataSubscription) -> dict[str, Any]:
        return {
            "exchange": subscription.exchange,
            "symbol": subscription.symbol,
            "channels": list(subscription.channels),
            "market_type": subscription.market_type,
            "timeframe": subscription.timeframe,
            "depth": subscription.depth,
            "enabled": subscription.enabled,
        }


StreamSubscription = MarketDataSubscription
