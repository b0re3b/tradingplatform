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
    """Local config for the data-layer orchestration service."""

    healthcheck_interval_seconds: float = 30.0
    startup_timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 15.0
    start_clients_on_start: bool = True
    register_caches_on_start: bool = True
    emit_lifecycle_events: bool = True


class MarketStream:
    """
    EventBus-first orchestration layer for market data.

    Responsibilities:
    - start/stop exchange adapters for all configured exchanges;
    - register data caches so they listen to market.* / market.*.snapshot topics;
    - run lightweight health checks through Scheduler;
    - publish system.market_stream.* lifecycle events;
    - never normalize raw exchange payloads and never update caches directly.

    Data flow stays:
        exchanges/* -> EventBus market.* -> data/*_cache.py -> market.*.updated -> analytics.
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
        service_name: str = "market_stream",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.exchange_clients = dict(exchange_clients)
        self.scheduler = scheduler
        self.stream_config = stream_config or MarketStreamConfig()
        self.caches = list(caches or [])
        self._service_name = service_name

        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="market_stream",
        )

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
        }

    def register(self) -> None:
        """
        Register all cache components against EventBus.

        This method is intentionally idempotent. The app bootstrap may call
        register_component(market_stream) before start_component(market_stream),
        while start() also guarantees registration for direct-start use cases.
        Without this guard, cache handlers are subscribed twice and every
        market.* event is processed twice.
        """
        if self._registered:
            self._logger.debug("MarketStream already registered")
            return

        if not self.stream_config.register_caches_on_start:
            self._registered = True
            return

        registered_count = 0

        for cache in self.caches:
            register = getattr(cache, "register", None)
            if register is None:
                continue

            result = register()
            if inspect.isawaitable(result):
                raise RuntimeError(
                    f"Cache register() must be synchronous for {cache.__class__.__name__}"
                )

            registered_count += 1

        self._registered = True

        self._logger.info(
            "MarketStream caches registered | caches=%s",
            registered_count,
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

        if self.stream_config.start_clients_on_start:
            await self._start_exchange_clients()

        await self._start_healthcheck()

        await self._emit_event(
            "system.market_stream.started",
            {
                "service": self._service_name,
                "exchanges": list(self.exchange_clients),
                "subscriptions": [self._serialize_subscription(s) for s in self._subscriptions],
                "caches": [cache.__class__.__name__ for cache in self.caches],
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

        await self._emit_event(
            "system.market_stream.stopped",
            {
                "service": self._service_name,
                "exchanges": list(self.exchange_clients),
            },
            priority=EventPriority.NORMAL,
        )

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
                    {"exchange": exchange},
                    priority=EventPriority.LOW,
                )
            except Exception as exc:
                state.is_started = False
                state.last_error = str(exc)
                self._metrics["clients_failed"] += 1
                self._logger.exception("Failed to start exchange client | exchange=%s", exchange)
                await self._emit_event(
                    "system.market_stream.exchange_start_failed",
                    {"exchange": exchange, "error": str(exc)},
                    priority=EventPriority.HIGH,
                )

    async def _stop_exchange_clients(self) -> None:
        for exchange, client in self.exchange_clients.items():
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

        state = self._states.setdefault(exchange, ExchangeRuntimeState(exchange=exchange))
        stop = getattr(client, "stop", None) or getattr(client, "close", None)
        if stop is not None:
            result = stop()
            if inspect.isawaitable(result):
                await result

        start = getattr(client, "start", None)
        if start is None:
            raise RuntimeError(f"Exchange client has no start(): {exchange}")
        result = start()
        if inspect.isawaitable(result):
            await result

        state.is_started = True
        state.started_at = time.time()
        state.last_error = None
        state.restarts += 1

        await self._emit_event(
            "system.market_stream.exchange_restarted",
            {"exchange": exchange, "restarts": state.restarts},
            priority=EventPriority.NORMAL,
        )

    async def health_check(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "stats": self.stats(),
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
            "exchanges_total": len(self.exchange_clients),
            "exchanges_started": sum(1 for s in self._states.values() if s.is_started),
            "subscriptions_total": len(self._subscriptions),
            **self._metrics,
        }

    async def _start_healthcheck(self) -> None:
        if self.scheduler is None or self._healthcheck_job_id is not None:
            return
        self._healthcheck_job_id = self.scheduler.add_interval_job(
            name="market-stream-healthcheck",
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

    async def _emit_event(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.LOW,
    ) -> None:
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


# Backward-compatible alias for older imports.
StreamSubscription = MarketDataSubscription