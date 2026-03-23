from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from core.config import Config
from core.logger import get_logger
from core.scheduler import Scheduler


@dataclass(slots=True)
class StreamSubscription:
    exchange: str
    symbol: str
    channel: str
    market_type: str = "perpetual"
    timeframe: str | None = None
    depth: int | None = None
    enabled: bool = True


@dataclass(slots=True)
class StreamRuntimeState:
    subscription: StreamSubscription
    stream_id: str
    is_running: bool = False
    reconnect_count: int = 0
    last_message_at: float | None = None
    last_error: str | None = None
    queue_dropped: int = 0
    stream_task: asyncio.Task | None = None
    consumer_task: asyncio.Task | None = None


class MarketStream:
    """
    Центральний orchestration layer для market data.

    Відповідальність:
    - керувати підписками на ринкові стріми бірж
    - приймати сирі повідомлення
    - нормалізувати market events
    - оновлювати відповідні кеші
    - публікувати події в EventBus
    - робити healthcheck / reconnect / monitoring
    """

    def __init__(
        self,
        *,
        config: Config,
        event_bus: Any,
        exchange_clients: dict[str, Any],
        scheduler: Scheduler | None = None,
        orderbook_cache: Any | None = None,
        trades_cache: Any | None = None,
        candles_cache: Any | None = None,
        funding_cache: Any | None = None,
        open_interest_cache: Any | None = None,
        liquidation_cache: Any | None = None,
        service_name: str = "market_stream",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.exchange_clients = exchange_clients
        self.scheduler = scheduler

        self.orderbook_cache = orderbook_cache
        self.trades_cache = trades_cache
        self.candles_cache = candles_cache
        self.funding_cache = funding_cache
        self.open_interest_cache = open_interest_cache
        self.liquidation_cache = liquidation_cache

        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="market_stream",
        )

        self._service_name = service_name
        self._running = False
        self._stopping = False

        self._subscriptions: dict[str, StreamSubscription] = {}
        self._states: dict[str, StreamRuntimeState] = {}
        self._queues: dict[str, asyncio.Queue] = {}

        self._healthcheck_job_id: str | None = None

        self._metrics: dict[str, int | float] = {
            "messages_total": 0,
            "published_events": 0,
            "parse_errors": 0,
            "reconnects": 0,
            "queue_dropped": 0,
            "started_at": 0.0,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, subscriptions: list[StreamSubscription]) -> None:
        if self._running:
            self._logger.warning("MarketStream already started")
            return

        self._running = True
        self._stopping = False
        self._metrics["started_at"] = time.time()

        for subscription in subscriptions:
            if subscription.enabled:
                await self.subscribe(subscription)

        await self._start_healthcheck()

        self._logger.info(
            "MarketStream started | subscriptions=%s exchanges=%s",
            len(self._subscriptions),
            list(self.exchange_clients.keys()),
        )

        await self._emit_system_event(
            "system.market_stream.started",
            {
                "subscriptions": len(self._subscriptions),
                "service": self._service_name,
            },
        )

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("MarketStream already stopped")
            return

        self._stopping = True
        self._running = False

        self._logger.info("Stopping MarketStream")

        await self._stop_healthcheck()

        tasks: list[asyncio.Task] = []
        for state in self._states.values():
            if state.stream_task is not None:
                state.stream_task.cancel()
                tasks.append(state.stream_task)
            if state.consumer_task is not None:
                state.consumer_task.cancel()
                tasks.append(state.consumer_task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        for exchange_name, client in self.exchange_clients.items():
            try:
                if hasattr(client, "close"):
                    result = client.close()
                    if inspect.isawaitable(result):
                        await result
            except Exception:
                self._logger.exception(
                    "Failed to close exchange client | exchange=%s",
                    exchange_name,
                )

        self._logger.info("MarketStream stopped")

        await self._emit_system_event(
            "system.market_stream.stopped",
            {
                "service": self._service_name,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def subscribe(self, subscription: StreamSubscription) -> str:
        stream_id = self._build_stream_id(subscription)

        if stream_id in self._subscriptions:
            self._logger.warning(
                "Subscription already exists | stream_id=%s",
                stream_id,
            )
            return stream_id

        queue = asyncio.Queue(maxsize=self.config.event_bus.max_queue_size)

        state = StreamRuntimeState(
            subscription=subscription,
            stream_id=stream_id,
        )

        self._subscriptions[stream_id] = subscription
        self._states[stream_id] = state
        self._queues[stream_id] = queue

        state.consumer_task = asyncio.create_task(
            self._consume_queue(stream_id),
            name=f"market-stream-consumer-{stream_id}",
        )
        state.stream_task = asyncio.create_task(
            self._run_stream(stream_id),
            name=f"market-stream-source-{stream_id}",
        )

        self._logger.info(
            "Subscription added | stream_id=%s exchange=%s symbol=%s channel=%s",
            stream_id,
            subscription.exchange,
            subscription.symbol,
            subscription.channel,
        )

        return stream_id

    async def unsubscribe(self, stream_id: str) -> None:
        state = self._states.get(stream_id)
        if state is None:
            self._logger.warning("Unsubscribe ignored: stream not found | stream_id=%s", stream_id)
            return

        if state.stream_task is not None:
            state.stream_task.cancel()
        if state.consumer_task is not None:
            state.consumer_task.cancel()

        tasks = [task for task in (state.stream_task, state.consumer_task) if task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._subscriptions.pop(stream_id, None)
        self._states.pop(stream_id, None)
        self._queues.pop(stream_id, None)

        self._logger.info("Subscription removed | stream_id=%s", stream_id)

    async def health_check(self) -> dict[str, Any]:
        now = time.time()

        streams: dict[str, Any] = {}
        for stream_id, state in self._states.items():
            lag_ms: int | None = None
            if state.last_message_at is not None:
                lag_ms = int((now - state.last_message_at) * 1000)

            streams[stream_id] = {
                "exchange": state.subscription.exchange,
                "symbol": state.subscription.symbol,
                "channel": state.subscription.channel,
                "is_running": state.is_running,
                "reconnect_count": state.reconnect_count,
                "last_message_at": state.last_message_at,
                "lag_ms": lag_ms,
                "last_error": state.last_error,
                "queue_size": self._queues[stream_id].qsize() if stream_id in self._queues else 0,
                "queue_dropped": state.queue_dropped,
            }

        return {
            "running": self._running,
            "stopping": self._stopping,
            "stats": self.stats(),
            "streams": streams,
        }

    def stats(self) -> dict[str, Any]:
        running_streams = sum(1 for state in self._states.values() if state.is_running)

        return {
            "running": self._running,
            "stopping": self._stopping,
            "subscriptions_total": len(self._subscriptions),
            "streams_running": running_streams,
            "messages_total": self._metrics["messages_total"],
            "published_events": self._metrics["published_events"],
            "parse_errors": self._metrics["parse_errors"],
            "reconnects": self._metrics["reconnects"],
            "queue_dropped": self._metrics["queue_dropped"],
            "started_at": self._metrics["started_at"],
        }

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    async def _start_healthcheck(self) -> None:
        if self.scheduler is None:
            self._logger.info("Scheduler is not provided, healthcheck job skipped")
            return

        existing = self.scheduler.get_job_by_name("market-stream-healthcheck")
        if existing is not None:
            self._healthcheck_job_id = existing.job_id
            self._logger.warning(
                "Healthcheck job already exists | job_id=%s",
                existing.job_id,
            )
            return

        self._healthcheck_job_id = self.scheduler.add_interval_job(
            name="market-stream-healthcheck",
            func=self._scheduled_healthcheck,
            interval=max(self.config.scheduler.tick_interval * 10, 1.0),
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=10.0,
            allow_overlap=False,
            enabled=True,
        )

        self._logger.info(
            "MarketStream healthcheck job registered | job_id=%s",
            self._healthcheck_job_id,
        )

    async def _stop_healthcheck(self) -> None:
        if self.scheduler is None or self._healthcheck_job_id is None:
            return

        with contextlib.suppress(Exception):
            self.scheduler.remove_job(self._healthcheck_job_id)

        self._logger.info(
            "MarketStream healthcheck job removed | job_id=%s",
            self._healthcheck_job_id,
        )
        self._healthcheck_job_id = None

    async def _scheduled_healthcheck(self) -> None:
        report = await self.health_check()
        lag_threshold_ms = int(self.config.exchange.timeout_seconds * 1000)

        for stream_id, stream_data in report["streams"].items():
            lag_ms = stream_data.get("lag_ms")
            if lag_ms is not None and lag_ms > lag_threshold_ms:
                self._logger.warning(
                    "Stream lag detected | stream_id=%s lag_ms=%s threshold_ms=%s",
                    stream_id,
                    lag_ms,
                    lag_threshold_ms,
                )

                await self._emit_system_event(
                    "system.market_stream.lag_detected",
                    {
                        "stream_id": stream_id,
                        "lag_ms": lag_ms,
                        "threshold_ms": lag_threshold_ms,
                    },
                )

    # ------------------------------------------------------------------
    # Stream runtime
    # ------------------------------------------------------------------

    async def _run_stream(self, stream_id: str) -> None:
        state = self._states[stream_id]
        subscription = state.subscription

        while self._running and not self._stopping and stream_id in self._subscriptions:
            try:
                state.is_running = True
                state.last_error = None

                self._logger.info(
                    "Stream connected | stream_id=%s exchange=%s symbol=%s channel=%s",
                    stream_id,
                    subscription.exchange,
                    subscription.symbol,
                    subscription.channel,
                )

                await self._emit_system_event(
                    "system.market_stream.connected",
                    {
                        "stream_id": stream_id,
                        "exchange": subscription.exchange,
                        "symbol": subscription.symbol,
                        "channel": subscription.channel,
                    },
                )

                client = self._get_exchange_client(subscription.exchange)

                async for raw_message in self._iterate_exchange_stream(client, subscription):
                    if not self._running or self._stopping:
                        break

                    state.last_message_at = time.time()
                    self._metrics["messages_total"] += 1

                    await self._enqueue_message(stream_id, raw_message)

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                state.is_running = False
                state.last_error = str(exc)
                state.reconnect_count += 1
                self._metrics["reconnects"] += 1

                self._logger.exception(
                    "Stream failed, reconnect scheduled | stream_id=%s reconnect_count=%s",
                    stream_id,
                    state.reconnect_count,
                )

                await self._emit_system_event(
                    "system.market_stream.error",
                    {
                        "stream_id": stream_id,
                        "exchange": subscription.exchange,
                        "symbol": subscription.symbol,
                        "channel": subscription.channel,
                        "error": str(exc),
                        "reconnect_count": state.reconnect_count,
                    },
                )

                await asyncio.sleep(self.config.exchange.reconnect_delay)

            finally:
                state.is_running = False

                if self._running and not self._stopping:
                    await self._emit_system_event(
                        "system.market_stream.disconnected",
                        {
                            "stream_id": stream_id,
                            "exchange": subscription.exchange,
                            "symbol": subscription.symbol,
                            "channel": subscription.channel,
                            "error": state.last_error,
                        },
                    )

    async def _consume_queue(self, stream_id: str) -> None:
        queue = self._queues[stream_id]
        subscription = self._subscriptions[stream_id]

        while self._running and not self._stopping and stream_id in self._subscriptions:
            try:
                raw_message = await queue.get()
                await self._dispatch_message(subscription, raw_message)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._metrics["parse_errors"] += 1
                self._logger.exception(
                    "Failed to process market data message | stream_id=%s channel=%s",
                    stream_id,
                    subscription.channel,
                )

    async def _enqueue_message(self, stream_id: str, raw_message: Any) -> None:
        queue = self._queues[stream_id]
        state = self._states[stream_id]

        if queue.full():
            state.queue_dropped += 1
            self._metrics["queue_dropped"] += 1

            self._logger.warning(
                "Queue full, dropping message | stream_id=%s queue_size=%s",
                stream_id,
                queue.qsize(),
            )
            return

        await queue.put(raw_message)

    async def _dispatch_message(self, subscription: StreamSubscription, raw_message: dict[str, Any]) -> None:
        channel = subscription.channel.lower()

        if channel == "trades":
            await self._handle_trade_message(subscription, raw_message)
            return

        if channel == "orderbook":
            await self._handle_orderbook_message(subscription, raw_message)
            return

        if channel == "candles":
            await self._handle_candle_message(subscription, raw_message)
            return

        if channel == "funding":
            await self._handle_funding_message(subscription, raw_message)
            return

        if channel == "open_interest":
            await self._handle_open_interest_message(subscription, raw_message)
            return

        if channel == "liquidations":
            await self._handle_liquidation_message(subscription, raw_message)
            return

        self._logger.warning(
            "Unknown channel received | exchange=%s symbol=%s channel=%s",
            subscription.exchange,
            subscription.symbol,
            subscription.channel,
        )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_trade_message(self, sub: StreamSubscription, raw: dict[str, Any]) -> None:
        event = self._normalize_trade_event(sub, raw)

        if self.trades_cache is not None:
            result = self.trades_cache.update(event)
            if inspect.isawaitable(result):
                await result

        await self._publish_market_event("market.trade", event)

    async def _handle_orderbook_message(self, sub: StreamSubscription, raw: dict[str, Any]) -> None:
        is_snapshot = bool(raw.get("is_snapshot", False))

        if is_snapshot:
            event = self._normalize_orderbook_snapshot_event(sub, raw)

            if self.orderbook_cache is not None:
                result = self.orderbook_cache.apply_snapshot(event)
                if inspect.isawaitable(result):
                    await result

            await self._publish_market_event("market.orderbook.snapshot", event)
            return

        event = self._normalize_orderbook_update_event(sub, raw)

        if self.orderbook_cache is not None:
            result = self.orderbook_cache.apply_delta(event)
            if inspect.isawaitable(result):
                await result

        await self._publish_market_event("market.orderbook.update", event)

    async def _handle_candle_message(self, sub: StreamSubscription, raw: dict[str, Any]) -> None:
        event = self._normalize_candle_event(sub, raw)

        if self.candles_cache is not None:
            result = self.candles_cache.update(event)
            if inspect.isawaitable(result):
                await result

        topic = "market.candle.closed" if event["is_closed"] else "market.candle.updated"
        await self._publish_market_event(topic, event)

    async def _handle_funding_message(self, sub: StreamSubscription, raw: dict[str, Any]) -> None:
        event = self._normalize_funding_event(sub, raw)

        if self.funding_cache is not None:
            result = self.funding_cache.update(event)
            if inspect.isawaitable(result):
                await result

        await self._publish_market_event("market.funding.updated", event)

    async def _handle_open_interest_message(self, sub: StreamSubscription, raw: dict[str, Any]) -> None:
        event = self._normalize_open_interest_event(sub, raw)

        if self.open_interest_cache is not None:
            result = self.open_interest_cache.update(event)
            if inspect.isawaitable(result):
                await result

        await self._publish_market_event("market.open_interest.updated", event)

    async def _handle_liquidation_message(self, sub: StreamSubscription, raw: dict[str, Any]) -> None:
        event = self._normalize_liquidation_event(sub, raw)

        if self.liquidation_cache is not None:
            result = self.liquidation_cache.update(event)
            if inspect.isawaitable(result):
                await result

        await self._publish_market_event("market.liquidation", event)

    # ------------------------------------------------------------------
    # Event publishing
    # ------------------------------------------------------------------

    async def _publish_market_event(self, topic: str, payload: dict[str, Any]) -> None:
        await self.event_bus.emit(
            topic,
            payload,
            source="market_stream",
        )
        self._metrics["published_events"] += 1

    async def _emit_system_event(self, topic: str, payload: dict[str, Any]) -> None:
        try:
            await self.event_bus.emit(
                topic,
                payload,
                source="market_stream",
            )
        except Exception:
            self._logger.exception(
                "Failed to emit system event | topic=%s",
                topic,
            )

    # ------------------------------------------------------------------
    # Exchange integration
    # ------------------------------------------------------------------

    def _get_exchange_client(self, exchange: str) -> Any:
        client = self.exchange_clients.get(exchange)
        if client is None:
            raise KeyError(f"Exchange client not found: {exchange}")
        return client

    async def _iterate_exchange_stream(
        self,
        client: Any,
        subscription: StreamSubscription,
    ) -> AsyncIterator[dict[str, Any]]:
        if hasattr(client, "stream"):
            async for item in client.stream(subscription):
                yield item
            return

        if hasattr(client, "subscribe"):
            async for item in client.subscribe(
                symbol=subscription.symbol,
                channel=subscription.channel,
                market_type=subscription.market_type,
                timeframe=subscription.timeframe,
                depth=subscription.depth,
            ):
                yield item
            return

        raise AttributeError(
            f"Exchange client for {subscription.exchange} must provide "
            "stream(subscription) or subscribe(...)"
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_trade_event(self, sub: StreamSubscription, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_type": "trade",
            "exchange": sub.exchange,
            "symbol": sub.symbol,
            "market_type": sub.market_type,
            "timestamp_ms": self._extract_timestamp_ms(raw),
            "received_at_ms": self._now_ms(),
            "trade_id": self._safe_str(raw.get("trade_id", raw.get("id"))),
            "price": float(raw.get("price", raw.get("p", 0.0))),
            "quantity": float(raw.get("quantity", raw.get("qty", raw.get("q", 0.0)))),
            "side": self._normalize_side(raw.get("side")),
            "aggressor_side": self._normalize_side(
                raw.get("aggressor_side", raw.get("taker_side", raw.get("side")))
            ),
        }

    def _normalize_orderbook_snapshot_event(self, sub: StreamSubscription, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_type": "orderbook_snapshot",
            "exchange": sub.exchange,
            "symbol": sub.symbol,
            "market_type": sub.market_type,
            "timestamp_ms": self._extract_timestamp_ms(raw),
            "received_at_ms": self._now_ms(),
            "bids": self._normalize_book_side(raw.get("bids", raw.get("b", []))),
            "asks": self._normalize_book_side(raw.get("asks", raw.get("a", []))),
            "sequence": self._safe_int(raw.get("sequence", raw.get("u"))),
        }

    def _normalize_orderbook_update_event(self, sub: StreamSubscription, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_type": "orderbook_update",
            "exchange": sub.exchange,
            "symbol": sub.symbol,
            "market_type": sub.market_type,
            "timestamp_ms": self._extract_timestamp_ms(raw),
            "received_at_ms": self._now_ms(),
            "bids": self._normalize_book_side(raw.get("bids", raw.get("b", []))),
            "asks": self._normalize_book_side(raw.get("asks", raw.get("a", []))),
            "sequence": self._safe_int(raw.get("sequence", raw.get("u"))),
            "prev_sequence": self._safe_int(raw.get("prev_sequence", raw.get("pu"))),
        }

    def _normalize_candle_event(self, sub: StreamSubscription, raw: dict[str, Any]) -> dict[str, Any]:
        is_closed = bool(raw.get("is_closed", raw.get("x", False)))

        return {
            "event_type": "candle",
            "exchange": sub.exchange,
            "symbol": sub.symbol,
            "market_type": sub.market_type,
            "timestamp_ms": self._extract_timestamp_ms(raw),
            "received_at_ms": self._now_ms(),
            "timeframe": raw.get("timeframe", sub.timeframe or "1m"),
            "open": float(raw.get("open", raw.get("o", 0.0))),
            "high": float(raw.get("high", raw.get("h", 0.0))),
            "low": float(raw.get("low", raw.get("l", 0.0))),
            "close": float(raw.get("close", raw.get("c", 0.0))),
            "volume": float(raw.get("volume", raw.get("v", 0.0))),
            "is_closed": is_closed,
        }

    def _normalize_funding_event(self, sub: StreamSubscription, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_type": "funding",
            "exchange": sub.exchange,
            "symbol": sub.symbol,
            "market_type": sub.market_type,
            "timestamp_ms": self._extract_timestamp_ms(raw),
            "received_at_ms": self._now_ms(),
            "funding_rate": float(raw.get("funding_rate", raw.get("rate", 0.0))),
            "next_funding_time_ms": self._safe_int(raw.get("next_funding_time_ms")),
        }

    def _normalize_open_interest_event(self, sub: StreamSubscription, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_type": "open_interest",
            "exchange": sub.exchange,
            "symbol": sub.symbol,
            "market_type": sub.market_type,
            "timestamp_ms": self._extract_timestamp_ms(raw),
            "received_at_ms": self._now_ms(),
            "open_interest": float(raw.get("open_interest", raw.get("oi", 0.0))),
        }

    def _normalize_liquidation_event(self, sub: StreamSubscription, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_type": "liquidation",
            "exchange": sub.exchange,
            "symbol": sub.symbol,
            "market_type": sub.market_type,
            "timestamp_ms": self._extract_timestamp_ms(raw),
            "received_at_ms": self._now_ms(),
            "price": float(raw.get("price", raw.get("p", 0.0))),
            "quantity": float(raw.get("quantity", raw.get("qty", raw.get("q", 0.0)))),
            "side": self._normalize_side(raw.get("side")),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_stream_id(subscription: StreamSubscription) -> str:
        timeframe = f":{subscription.timeframe}" if subscription.timeframe else ""
        depth = f":depth={subscription.depth}" if subscription.depth is not None else ""
        return (
            f"{subscription.exchange}:"
            f"{subscription.market_type}:"
            f"{subscription.symbol}:"
            f"{subscription.channel}"
            f"{timeframe}{depth}"
        )

    @staticmethod
    def _extract_timestamp_ms(raw: dict[str, Any]) -> int:
        value = raw.get("timestamp_ms", raw.get("ts", raw.get("T", raw.get("E"))))
        if value is None:
            return int(time.time() * 1000)
        return int(value)

    @staticmethod
    def _normalize_side(value: Any) -> str:
        if value is None:
            return "unknown"

        normalized = str(value).strip().lower()
        if normalized in {"buy", "bid", "b"}:
            return "buy"
        if normalized in {"sell", "ask", "s"}:
            return "sell"
        return "unknown"

    @staticmethod
    def _normalize_book_side(levels: Any) -> list[list[float]]:
        normalized: list[list[float]] = []

        for level in levels or []:
            if not isinstance(level, (list, tuple)) or len(level) < 2:
                continue

            price = float(level[0])
            quantity = float(level[1])
            normalized.append([price, quantity])

        return normalized

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)