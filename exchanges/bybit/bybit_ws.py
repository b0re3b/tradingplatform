from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger


@dataclass(slots=True)
class BybitWebSocketClientConfig:
    """
    Local Bybit WS adapter config.

    Bybit is currently used mainly as a market-data source.
    Private stream is optional and must publish only exchange.* events,
    not execution.* domain events.
    """

    category: str = "linear"

    public_ws_url: str | None = None
    private_ws_url: str = "wss://stream.bybit.com/v5/private"

    timeout_seconds: float = 10.0
    reconnect_delay_seconds: float = 5.0
    max_reconnect_attempts: int = 20

    ping_interval_seconds: float = 20.0
    recv_window_ms: int = 5_000

    symbols: list[str] = field(default_factory=list)
    streams: list[str] = field(default_factory=lambda: ["trade", "orderbook", "kline"])

    orderbook_depth: int = 50
    kline_interval: str = "1"
    orderbook_emit_min_interval_ms: int = 250
    orderbook_batch_max_size: int = 500
    trade_emit_min_interval_ms: int = 250
    trade_batch_max_size: int = 1000

    enable_private_stream: bool = False

    @classmethod
    def from_core_config(
        cls,
        *,
        config: Config,
        symbols: list[str],
        streams: list[str] | None = None,
        category: str = "linear",
        orderbook_depth: int = 50,
        kline_interval: str = "1",
        enable_private_stream: bool = False,
        ping_interval_seconds: float = 20.0,
        recv_window_ms: int = 5_000,
        orderbook_emit_min_interval_ms: int = 250,
        orderbook_batch_max_size: int = 500,
        trade_emit_min_interval_ms: int = 250,
        trade_batch_max_size: int = 1000,
    ) -> "BybitWebSocketClientConfig":
        return cls(
            category=category,
            public_ws_url=config.exchange.ws_url,
            timeout_seconds=config.exchange.timeout_seconds,
            reconnect_delay_seconds=config.exchange.reconnect_delay,
            max_reconnect_attempts=config.exchange.max_reconnect_attempts,
            ping_interval_seconds=ping_interval_seconds,
            recv_window_ms=recv_window_ms,
            symbols=symbols,
            streams=streams or ["trade", "orderbook", "kline"],
            orderbook_depth=orderbook_depth,
            kline_interval=kline_interval,
            enable_private_stream=enable_private_stream,
            orderbook_emit_min_interval_ms=orderbook_emit_min_interval_ms,
            orderbook_batch_max_size=orderbook_batch_max_size,
            trade_emit_min_interval_ms=trade_emit_min_interval_ms,
            trade_batch_max_size=trade_batch_max_size,
        )


class BybitWebSocketClient:
    """
    Bybit WebSocket exchange adapter.

    Responsibilities:
    - connect to Bybit public market streams;
    - normalize raw Bybit payloads into internal market events;
    - optionally connect to private stream and publish exchange.* updates;
    - publish all events through EventBus;
    - never call analytics, strategy, risk, or execution directly;
    - never contain trading decision logic.

    Public market events:
    - market.trade
    - market.orderbook
    - market.candle
    - market.liquidation

    Optional private exchange events:
    - exchange.order.updated
    - exchange.position.updated
    - exchange.account.wallet_updated

    System events:
    - system.exchange.ws.started
    - system.exchange.ws.stopped
    - system.exchange.ws.connected
    - system.exchange.ws.disconnected
    - system.exchange.ws.error
    - system.exchange.ws.authenticated
    - system.exchange.ws.subscribed
    """

    EXCHANGE = "bybit"
    SOURCE = "bybit_ws"

    SUPPORTED_CATEGORIES = {"linear", "inverse", "spot", "option"}
    SUPPORTED_STREAMS = {"trade", "orderbook", "kline", "liquidation"}
    SUPPORTED_ORDERBOOK_DEPTHS = {1, 50, 200, 500}

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        ws_config: BybitWebSocketClientConfig | None = None,
        symbols: list[str] | None = None,
        streams: list[str] | None = None,
        category: str = "linear",
        orderbook_depth: int = 50,
        kline_interval: str = "1",
        enable_private_stream: bool = False,
        ping_interval: float = 20.0,
        recv_window_ms: int = 5_000,
        orderbook_emit_min_interval_ms: int = 250,
        orderbook_batch_max_size: int = 500,
        trade_emit_min_interval_ms: int = 250,
        trade_batch_max_size: int = 1000,
    ) -> None:
        resolved_config = ws_config or BybitWebSocketClientConfig.from_core_config(
            config=config,
            symbols=symbols or [],
            streams=streams,
            category=category,
            orderbook_depth=orderbook_depth,
            kline_interval=kline_interval,
            enable_private_stream=enable_private_stream,
            ping_interval_seconds=ping_interval,
            recv_window_ms=recv_window_ms,
            orderbook_emit_min_interval_ms=orderbook_emit_min_interval_ms,
            orderbook_batch_max_size=orderbook_batch_max_size,
            trade_emit_min_interval_ms=trade_emit_min_interval_ms,
            trade_batch_max_size=trade_batch_max_size,
        )

        self._config = config
        self._event_bus = event_bus
        self._ws_config = resolved_config

        self._category = self._ws_config.category.lower()
        self._symbols = [symbol.upper() for symbol in self._ws_config.symbols]
        self._streams = self._normalize_streams(self._ws_config.streams)

        self._api_key = config.exchange.credentials.api_key
        self._api_secret = config.exchange.credentials.api_secret

        self._logger = get_logger(
            __name__,
            exchange=self.EXCHANGE,
            event_type="exchange_ws",
        )

        self._public_ws_url = (
            self._ws_config.public_ws_url
            or self._resolve_public_url(self._category)
        )
        self._private_ws_url = self._ws_config.private_ws_url

        self._session: aiohttp.ClientSession | None = None

        self._public_ws: aiohttp.ClientWebSocketResponse | None = None
        self._private_ws: aiohttp.ClientWebSocketResponse | None = None

        self._public_task: asyncio.Task | None = None
        self._private_task: asyncio.Task | None = None
        self._public_ping_task: asyncio.Task | None = None
        self._private_ping_task: asyncio.Task | None = None

        self._running = False
        self._started = False

        self._last_orderbook_emit_ms: dict[str, int] = {}
        self._pending_orderbook_payloads: dict[str, list[dict[str, Any]]] = {}
        self._orderbook_flush_tasks: dict[str, asyncio.Task] = {}
        self._orderbook_throttled_updates: dict[str, int] = {}

        self._last_trade_batch_emit_ms: dict[str, int] = {}
        self._pending_trade_payloads: dict[str, list[dict[str, Any]]] = {}
        self._trade_flush_tasks: dict[str, asyncio.Task] = {}
        self._trade_throttled_updates: dict[str, int] = {}

        self._validate_config()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        WS adapter currently does not subscribe to EventBus topics.

        Kept for project-wide consistency with modules that expose register().
        """
        self._logger.debug("Bybit WS register called | subscriptions=0")

    async def start(self) -> None:
        if self._running:
            self._logger.warning("Bybit WS client already started")
            return

        self._running = True
        self._started = True

        await self._ensure_session()

        self._logger.info(
            "Starting Bybit WS client | category=%s symbols=%s streams=%s private_stream=%s",
            self._category,
            self._symbols,
            self._streams,
            self._ws_config.enable_private_stream,
        )

        await self._emit_event(
            "system.exchange.ws.started",
            {
                "exchange": self.EXCHANGE,
                "category": self._category,
                "symbols": self._symbols,
                "streams": self._streams,
                "private_stream": self._ws_config.enable_private_stream,
            },
            priority=EventPriority.NORMAL,
        )

        self._public_task = asyncio.create_task(
            self._run_public_loop(),
            name="bybit-public-ws-loop",
        )

        if self._ws_config.enable_private_stream:
            self._require_private_credentials()

            self._private_task = asyncio.create_task(
                self._run_private_loop(),
                name="bybit-private-ws-loop",
            )

    async def stop(self) -> None:
        if not self._running and not self._started:
            self._logger.warning("Bybit WS client already stopped")
            return

        self._logger.info("Stopping Bybit WS client")

        self._running = False
        self._started = False

        tasks = [
            task
            for task in (
                self._public_task,
                self._private_task,
                self._public_ping_task,
                self._private_ping_task,
            )
            if task is not None
        ]

        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        await self._cancel_orderbook_flush_tasks()
        await self._cancel_trade_flush_tasks()

        self._public_task = None
        self._private_task = None
        self._public_ping_task = None
        self._private_ping_task = None

        await self._close_ws(self._public_ws, channel="public")
        await self._close_ws(self._private_ws, channel="private")

        self._public_ws = None
        self._private_ws = None

        await self._close_session()

        self._logger.info("Bybit WS client stopped")

        await self._emit_event(
            "system.exchange.ws.stopped",
            {
                "exchange": self.EXCHANGE,
            },
            priority=EventPriority.NORMAL,
        )

    # ------------------------------------------------------------------
    # Public WS loop
    # ------------------------------------------------------------------

    async def _run_public_loop(self) -> None:
        reconnect_attempt = 0

        while self._running:
            try:
                await self._ensure_session()

                self._logger.info(
                    "Connecting to Bybit public WS | url=%s",
                    self._public_ws_url,
                )

                assert self._session is not None
                self._public_ws = await self._session.ws_connect(
                    self._public_ws_url,
                    heartbeat=None,
                    autoping=False,
                )

                reconnect_attempt = 0

                self._logger.info("Connected to Bybit public WS")

                await self._emit_event(
                    "system.exchange.ws.connected",
                    {
                        "exchange": self.EXCHANGE,
                        "channel": "public",
                        "category": self._category,
                        "symbols": self._symbols,
                    },
                    priority=EventPriority.HIGH,
                )

                await self._subscribe_public_topics()

                self._public_ping_task = asyncio.create_task(
                    self._ping_loop(self._public_ws, "public"),
                    name="bybit-public-ws-ping",
                )

                await self._consume_public_messages()

            except asyncio.CancelledError:
                self._logger.info("Bybit public WS loop cancelled")
                raise
            except Exception as exc:
                reconnect_attempt += 1

                await self._handle_ws_loop_error(
                    channel="public",
                    exc=exc,
                    reconnect_attempt=reconnect_attempt,
                )

                if self._should_stop_reconnecting(reconnect_attempt):
                    self._logger.error("Bybit public WS max reconnect attempts reached")
                    break

                await asyncio.sleep(self._ws_config.reconnect_delay_seconds)

            finally:
                await self._cancel_ping_task(channel="public")
                await self._close_ws(self._public_ws, channel="public")
                self._public_ws = None

                if self._running:
                    await self._emit_event(
                        "system.exchange.ws.disconnected",
                        {
                            "exchange": self.EXCHANGE,
                            "channel": "public",
                        },
                        priority=EventPriority.HIGH,
                    )

    async def _subscribe_public_topics(self) -> None:
        if self._public_ws is None:
            raise RuntimeError("Bybit public WebSocket is not connected")

        topics: list[str] = []

        for symbol in self._symbols:
            if "trade" in self._streams:
                topics.append(f"publicTrade.{symbol}")

            if "orderbook" in self._streams:
                topics.append(
                    f"orderbook.{self._ws_config.orderbook_depth}.{symbol}"
                )

            if "kline" in self._streams:
                topics.append(
                    f"kline.{self._ws_config.kline_interval}.{symbol}"
                )

            if "liquidation" in self._streams:
                topics.append(f"liquidation.{symbol}")

        if not topics:
            raise RuntimeError("No Bybit public topics to subscribe")

        payload = {
            "op": "subscribe",
            "args": topics,
        }

        await self._public_ws.send_json(payload)

        self._logger.info(
            "Subscribed to Bybit public topics | count=%s",
            len(topics),
        )

        await self._emit_event(
            "system.exchange.ws.subscribed",
            {
                "exchange": self.EXCHANGE,
                "channel": "public",
                "category": self._category,
                "topics": topics,
            },
            priority=EventPriority.LOW,
        )

    async def _consume_public_messages(self) -> None:
        assert self._public_ws is not None

        async for msg in self._public_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_public_message(msg.data)
                continue

            if msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("Bybit public WebSocket error")

            if msg.type in {
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            }:
                self._logger.warning("Bybit public WS closed by server")
                break

    async def _handle_public_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode Bybit public WS message")
            return

        if message.get("op") == "pong":
            self._logger.debug("Received Bybit public pong")
            return

        if message.get("op") == "subscribe":
            success = bool(message.get("success"))

            self._logger.info(
                "Bybit public subscribe response | success=%s",
                success,
            )

            if not success:
                await self._emit_event(
                    "system.exchange.ws.error",
                    {
                        "exchange": self.EXCHANGE,
                        "channel": "public",
                        "operation": "subscribe",
                        "message": message.get("ret_msg") or message.get("msg"),
                    },
                    priority=EventPriority.HIGH,
                )
            return

        topic = message.get("topic")
        data = message.get("data")

        if not topic or data is None:
            self._logger.debug("Received empty or unhandled Bybit public payload")
            return

        if topic.startswith("publicTrade."):
            await self._publish_trade_events(data)
            return

        if topic.startswith("orderbook."):
            await self._publish_orderbook_event(topic, data, message.get("ts"))
            return

        if topic.startswith("kline."):
            await self._publish_kline_events(data)
            return

        if topic.startswith("liquidation."):
            await self._publish_liquidation_events(data)
            return

        self._logger.debug("Unhandled Bybit public topic | topic=%s", topic)

    # ------------------------------------------------------------------
    # Private WS loop
    # ------------------------------------------------------------------

    async def _run_private_loop(self) -> None:
        reconnect_attempt = 0

        while self._running:
            try:
                await self._ensure_session()
                self._require_private_credentials()

                self._logger.info("Connecting to Bybit private WS")

                assert self._session is not None
                self._private_ws = await self._session.ws_connect(
                    self._private_ws_url,
                    heartbeat=None,
                    autoping=False,
                )

                await self._authenticate_private_ws()
                await self._subscribe_private_topics()

                reconnect_attempt = 0

                self._logger.info("Connected to Bybit private WS")

                await self._emit_event(
                    "system.exchange.ws.connected",
                    {
                        "exchange": self.EXCHANGE,
                        "channel": "private",
                    },
                    priority=EventPriority.HIGH,
                )

                self._private_ping_task = asyncio.create_task(
                    self._ping_loop(self._private_ws, "private"),
                    name="bybit-private-ws-ping",
                )

                await self._consume_private_messages()

            except asyncio.CancelledError:
                self._logger.info("Bybit private WS loop cancelled")
                raise
            except Exception as exc:
                reconnect_attempt += 1

                await self._handle_ws_loop_error(
                    channel="private",
                    exc=exc,
                    reconnect_attempt=reconnect_attempt,
                )

                if self._should_stop_reconnecting(reconnect_attempt):
                    self._logger.error("Bybit private WS max reconnect attempts reached")
                    break

                await asyncio.sleep(self._ws_config.reconnect_delay_seconds)

            finally:
                await self._cancel_ping_task(channel="private")
                await self._close_ws(self._private_ws, channel="private")
                self._private_ws = None

                if self._running:
                    await self._emit_event(
                        "system.exchange.ws.disconnected",
                        {
                            "exchange": self.EXCHANGE,
                            "channel": "private",
                        },
                        priority=EventPriority.HIGH,
                    )

    async def _authenticate_private_ws(self) -> None:
        if self._private_ws is None:
            raise RuntimeError("Bybit private WebSocket is not connected")

        self._require_private_credentials()

        assert self._api_key is not None

        expires = str(self._current_timestamp_ms() + self._ws_config.recv_window_ms)
        signature = self._sign_ws_auth(expires)

        payload = {
            "op": "auth",
            "args": [self._api_key, expires, signature],
        }

        await self._private_ws.send_json(payload)
        self._logger.info("Bybit private WS auth request sent")

    async def _subscribe_private_topics(self) -> None:
        if self._private_ws is None:
            raise RuntimeError("Bybit private WebSocket is not connected")

        topics = ["order", "position", "wallet"]

        payload = {
            "op": "subscribe",
            "args": topics,
        }

        await self._private_ws.send_json(payload)

        self._logger.info(
            "Subscribed to Bybit private topics | count=%s",
            len(topics),
        )

    async def _consume_private_messages(self) -> None:
        assert self._private_ws is not None

        async for msg in self._private_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_private_message(msg.data)
                continue

            if msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("Bybit private WebSocket error")

            if msg.type in {
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            }:
                self._logger.warning("Bybit private WS closed by server")
                break

    async def _handle_private_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode Bybit private WS message")
            return

        if message.get("op") == "pong":
            self._logger.debug("Received Bybit private pong")
            return

        if message.get("op") == "auth":
            success = bool(message.get("success"))

            if not success:
                raise RuntimeError(
                    f"Bybit private auth failed | msg={message.get('ret_msg') or message.get('msg')}"
                )

            self._logger.info("Bybit private auth succeeded")

            await self._emit_event(
                "system.exchange.ws.authenticated",
                {
                    "exchange": self.EXCHANGE,
                    "channel": "private",
                },
                priority=EventPriority.HIGH,
            )
            return

        if message.get("op") == "subscribe":
            success = bool(message.get("success"))

            self._logger.info(
                "Bybit private subscribe response | success=%s",
                success,
            )

            await self._emit_event(
                "system.exchange.ws.subscribed",
                {
                    "exchange": self.EXCHANGE,
                    "channel": "private",
                    "success": success,
                },
                priority=EventPriority.LOW if success else EventPriority.HIGH,
            )
            return

        topic = message.get("topic")
        data = message.get("data")

        if not topic or data is None:
            self._logger.debug("Received empty or unhandled Bybit private payload")
            return

        if topic == "order":
            await self._publish_private_order_events(data)
            return

        if topic == "position":
            await self._publish_private_position_events(data)
            return

        if topic == "wallet":
            await self._publish_private_wallet_events(data)
            return

        self._logger.debug("Unhandled Bybit private topic | topic=%s", topic)

    # ------------------------------------------------------------------
    # Ping
    # ------------------------------------------------------------------

    async def _ping_loop(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        channel: str,
    ) -> None:
        try:
            while self._running and not ws.closed:
                await asyncio.sleep(self._ws_config.ping_interval_seconds)

                if not self._running or ws.closed:
                    break

                await ws.send_json({"op": "ping"})
                self._logger.debug("Sent Bybit ping | channel=%s", channel)

        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception(
                "Bybit ping loop failed | channel=%s",
                channel,
            )
            raise

    async def _cancel_orderbook_flush_tasks(self) -> None:
        tasks = [task for task in self._orderbook_flush_tasks.values() if task is not None and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._orderbook_flush_tasks.clear()
        self._pending_orderbook_payloads.clear()

    async def _emit_orderbook_event_coalesced(
        self,
        *,
        key: str,
        payload: dict[str, Any],
    ) -> None:
        """
        Coalesces high-frequency orderbook updates into market.orderbook.batch.

        Important: orderbook updates may be exchange deltas, so this method keeps
        every pending update in order instead of replacing them with the latest
        payload. OrderBookCache applies the batch sequentially and performs the
        usual sequence validation.
        """
        interval_ms = max(
            0,
            int(getattr(self._ws_config, "orderbook_emit_min_interval_ms", 0) or 0),
        )

        if interval_ms <= 0:
            await self._emit_event("market.orderbook", payload, priority=EventPriority.LOW)
            return

        normalized_key = key or str(payload.get("symbol") or "unknown").upper()
        now_ms = int(time.time() * 1000)

        batch = self._pending_orderbook_payloads.setdefault(normalized_key, [])
        batch.append(payload)

        max_batch_size = max(
            1,
            int(getattr(self._ws_config, "orderbook_batch_max_size", 500) or 500),
        )

        last_emit_ms = self._last_orderbook_emit_ms.get(normalized_key, 0)
        elapsed_ms = now_ms - last_emit_ms

        should_flush_now = (
            last_emit_ms <= 0
            or elapsed_ms >= interval_ms
            or len(batch) >= max_batch_size
        )

        if should_flush_now:
            await self._flush_pending_orderbook_batch(normalized_key, delay_seconds=0.0)
            return

        self._orderbook_throttled_updates[normalized_key] = (
            self._orderbook_throttled_updates.get(normalized_key, 0) + 1
        )

        existing_task = self._orderbook_flush_tasks.get(normalized_key)
        if existing_task is not None and not existing_task.done():
            return

        delay_seconds = max((interval_ms - elapsed_ms) / 1000.0, 0.0)
        self._orderbook_flush_tasks[normalized_key] = asyncio.create_task(
            self._flush_pending_orderbook_batch(normalized_key, delay_seconds)
        )

    async def _flush_pending_orderbook_batch(self, key: str, delay_seconds: float) -> None:
        try:
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

            updates = self._pending_orderbook_payloads.pop(key, None)
            if not updates:
                return

            self._last_orderbook_emit_ms[key] = int(time.time() * 1000)

            first = updates[0]
            last = updates[-1]
            batch_payload = {
                "exchange": self.EXCHANGE,
                "symbol": first.get("symbol") or last.get("symbol"),
                "market_type": first.get("market_type") or last.get("market_type") or "usdm_futures",
                "source": self.SOURCE,
                "count": len(updates),
                "first_event_time": first.get("event_time") or first.get("timestamp_ms") or first.get("timestamp"),
                "last_event_time": last.get("event_time") or last.get("timestamp_ms") or last.get("timestamp"),
                "updates": updates,
            }

            await self._emit_event(
                "market.orderbook.batch",
                batch_payload,
                priority=EventPriority.LOW,
            )

        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception(
                "Failed to flush coalesced orderbook batch | exchange=%s key=%s",
                self.EXCHANGE,
                key,
            )
        finally:
            task = self._orderbook_flush_tasks.get(key)
            if task is asyncio.current_task():
                self._orderbook_flush_tasks.pop(key, None)

    async def _cancel_trade_flush_tasks(self) -> None:
        tasks = [task for task in self._trade_flush_tasks.values() if task is not None and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._trade_flush_tasks.clear()
        self._pending_trade_payloads.clear()

    async def _emit_trade_event_coalesced(
        self,
        *,
        key: str,
        payload: dict[str, Any],
    ) -> None:
        """
        Coalesces high-frequency raw trades into market.trades.batch.

        TradesCache can already ingest a list from payload["trades"], so this
        reduces EventBus pressure without losing individual trade records.
        """
        interval_ms = max(
            0,
            int(getattr(self._ws_config, "trade_emit_min_interval_ms", 0) or 0),
        )

        if interval_ms <= 0:
            await self._emit_event("market.trade", payload, priority=EventPriority.LOW)
            return

        normalized_key = key or str(payload.get("symbol") or "unknown").upper()
        now_ms = int(time.time() * 1000)

        batch = self._pending_trade_payloads.setdefault(normalized_key, [])
        batch.append(payload)

        max_batch_size = max(
            1,
            int(getattr(self._ws_config, "trade_batch_max_size", 1000) or 1000),
        )

        last_emit_ms = self._last_trade_batch_emit_ms.get(normalized_key, 0)
        elapsed_ms = now_ms - last_emit_ms

        should_flush_now = (
            last_emit_ms <= 0
            or elapsed_ms >= interval_ms
            or len(batch) >= max_batch_size
        )

        if should_flush_now:
            await self._flush_pending_trade_batch(normalized_key, delay_seconds=0.0)
            return

        self._trade_throttled_updates[normalized_key] = (
            self._trade_throttled_updates.get(normalized_key, 0) + 1
        )

        existing_task = self._trade_flush_tasks.get(normalized_key)
        if existing_task is not None and not existing_task.done():
            return

        delay_seconds = max((interval_ms - elapsed_ms) / 1000.0, 0.0)
        self._trade_flush_tasks[normalized_key] = asyncio.create_task(
            self._flush_pending_trade_batch(normalized_key, delay_seconds)
        )

    async def _flush_pending_trade_batch(self, key: str, delay_seconds: float) -> None:
        try:
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

            trades = self._pending_trade_payloads.pop(key, None)
            if not trades:
                return

            self._last_trade_batch_emit_ms[key] = int(time.time() * 1000)

            first = trades[0]
            last = trades[-1]
            batch_payload = {
                "exchange": self.EXCHANGE,
                "symbol": first.get("symbol") or last.get("symbol"),
                "market_type": first.get("market_type") or last.get("market_type") or "usdm_futures",
                "source": self.SOURCE,
                "count": len(trades),
                "first_trade_time": first.get("trade_time") or first.get("event_time") or first.get("timestamp_ms"),
                "last_trade_time": last.get("trade_time") or last.get("event_time") or last.get("timestamp_ms"),
                "trades": trades,
            }

            await self._emit_event(
                "market.trades.batch",
                batch_payload,
                priority=EventPriority.LOW,
            )

        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception(
                "Failed to flush coalesced trade batch | exchange=%s key=%s",
                self.EXCHANGE,
                key,
            )
        finally:
            task = self._trade_flush_tasks.get(key)
            if task is asyncio.current_task():
                self._trade_flush_tasks.pop(key, None)


    # ------------------------------------------------------------------
    # Event publishers: public market data
    # ------------------------------------------------------------------

    async def _publish_trade_events(self, trades: list[dict[str, Any]]) -> None:
        if not isinstance(trades, list):
            return

        for trade in trades:
            side = (trade.get("S") or "").lower()

            payload = {
                "exchange": self.EXCHANGE,
                "market_type": self._category,
                "symbol": trade.get("s"),
                "trade_id": trade.get("i"),
                "price": self._safe_float(trade.get("p")),
                "qty": self._safe_float(trade.get("v")),
                "side": side,
                "trade_time": self._safe_int(trade.get("T")),
                "is_block_trade": trade.get("BT"),
            }

            await self._emit_trade_event_coalesced(
                key=str(payload.get("symbol") or "unknown").upper(),
                payload=payload,
            )

    async def _publish_orderbook_event(
        self,
        topic: str,
        data: dict[str, Any],
        event_time: int | None,
    ) -> None:
        if not isinstance(data, dict):
            return

        payload = {
            "exchange": self.EXCHANGE,
            "market_type": self._category,
            "symbol": data.get("s"),
            "type": data.get("type"),
            "update_id": data.get("u"),
            "sequence": data.get("seq"),
            "bids": [
                [self._safe_float(price), self._safe_float(qty)]
                for price, qty in data.get("b", [])
            ],
            "asks": [
                [self._safe_float(price), self._safe_float(qty)]
                for price, qty in data.get("a", [])
            ],
            "event_time": self._safe_int(event_time),
            "topic": topic,
        }

        await self._emit_orderbook_event_coalesced(
            key=str(payload.get("symbol") or topic or "unknown").upper(),
            payload=payload,
        )

    async def _publish_kline_events(self, klines: list[dict[str, Any]]) -> None:
        if not isinstance(klines, list):
            return

        for item in klines:
            payload = {
                "exchange": self.EXCHANGE,
                "market_type": self._category,
                "symbol": item.get("symbol"),
                "timeframe": str(item.get("interval")) if item.get("interval") is not None else None,
                "open_time": self._safe_int(item.get("start")),
                "close_time": self._safe_int(item.get("end")),
                "open": self._safe_float(item.get("open")),
                "high": self._safe_float(item.get("high")),
                "low": self._safe_float(item.get("low")),
                "close": self._safe_float(item.get("close")),
                "volume": self._safe_float(item.get("volume")),
                "quote_volume": self._safe_float(item.get("turnover")),
                "trades_count": None,
                "is_closed": bool(item.get("confirm")),
                "confirm": item.get("confirm"),
                "timestamp": self._safe_int(item.get("timestamp")),
            }

            await self._emit_event(
                "market.candle",
                payload,
                priority=EventPriority.HIGH,
            )

    async def _publish_liquidation_events(self, data: Any) -> None:
        items = data if isinstance(data, list) else [data]

        for item in items:
            if not isinstance(item, dict):
                continue

            payload = {
                "exchange": self.EXCHANGE,
                "symbol": item.get("symbol"),
                "side": (item.get("side") or "").lower(),
                "price": self._safe_float(item.get("price")),
                "qty": self._safe_float(item.get("size")),
                "updated_time": self._safe_int(item.get("updatedTime")),
            }

            await self._emit_event(
                "market.liquidation",
                payload,
                priority=EventPriority.HIGH,
            )

    # ------------------------------------------------------------------
    # Event publishers: optional private exchange updates
    # ------------------------------------------------------------------

    async def _publish_private_order_events(self, orders: list[dict[str, Any]]) -> None:
        if not isinstance(orders, list):
            return

        for order in orders:
            payload = {
                "exchange": self.EXCHANGE,
                "symbol": order.get("symbol"),
                "order_id": order.get("orderId"),
                "client_order_id": order.get("orderLinkId"),
                "side": order.get("side"),
                "order_type": order.get("orderType"),
                "order_status": order.get("orderStatus"),
                "price": self._safe_float(order.get("price")),
                "qty": self._safe_float(order.get("qty")),
                "cum_exec_qty": self._safe_float(order.get("cumExecQty")),
                "cum_exec_value": self._safe_float(order.get("cumExecValue")),
                "cum_exec_fee": self._safe_float(order.get("cumExecFee")),
                "avg_price": self._safe_float(order.get("avgPrice")),
                "trigger_price": self._safe_float(order.get("triggerPrice")),
                "created_time": self._safe_int(order.get("createdTime")),
                "updated_time": self._safe_int(order.get("updatedTime")),
            }

            await self._emit_event(
                "exchange.order.updated",
                payload,
                priority=EventPriority.HIGH,
            )

    async def _publish_private_position_events(self, positions: list[dict[str, Any]]) -> None:
        if not isinstance(positions, list):
            return

        for position in positions:
            payload = {
                "exchange": self.EXCHANGE,
                "symbol": position.get("symbol"),
                "side": position.get("side"),
                "size": self._safe_float(position.get("size")),
                "entry_price": self._safe_float(position.get("avgPrice")),
                "mark_price": self._safe_float(position.get("markPrice")),
                "liq_price": self._safe_float(position.get("liqPrice")),
                "unrealised_pnl": self._safe_float(position.get("unrealisedPnl")),
                "position_value": self._safe_float(position.get("positionValue")),
                "leverage": self._safe_float(position.get("leverage")),
                "updated_time": self._safe_int(position.get("updatedTime")),
            }

            await self._emit_event(
                "exchange.position.updated",
                payload,
                priority=EventPriority.HIGH,
            )

    async def _publish_private_wallet_events(self, wallets: list[dict[str, Any]]) -> None:
        if not isinstance(wallets, list):
            return

        for wallet in wallets:
            payload = {
                "exchange": self.EXCHANGE,
                "account_type": wallet.get("accountType"),
                "total_equity": self._safe_float(wallet.get("totalEquity")),
                "total_wallet_balance": self._safe_float(wallet.get("totalWalletBalance")),
                "total_margin_balance": self._safe_float(wallet.get("totalMarginBalance")),
                "total_available_balance": self._safe_float(wallet.get("totalAvailableBalance")),
                "coin": wallet.get("coin", []),
            }

            await self._emit_event(
                "exchange.account.wallet_updated",
                payload,
                priority=EventPriority.HIGH,
            )

    # ------------------------------------------------------------------
    # Shared event/error helpers
    # ------------------------------------------------------------------

    async def _emit_event(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority,
    ) -> None:
        try:
            await self._event_bus.emit(
                topic,
                payload,
                priority=priority,
                source=self.SOURCE,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception(
                "Failed to emit Bybit WS event | topic=%s",
                topic,
            )

    async def _handle_ws_loop_error(
        self,
        *,
        channel: str,
        exc: Exception,
        reconnect_attempt: int,
    ) -> None:
        self._logger.exception(
            "Bybit WS loop error | channel=%s attempt=%s max_attempts=%s",
            channel,
            reconnect_attempt,
            self._ws_config.max_reconnect_attempts,
        )

        await self._emit_event(
            "system.exchange.ws.error",
            {
                "exchange": self.EXCHANGE,
                "channel": channel,
                "error": str(exc),
                "attempt": reconnect_attempt,
                "max_attempts": self._ws_config.max_reconnect_attempts,
            },
            priority=EventPriority.HIGH,
        )

    # ------------------------------------------------------------------
    # Session / websocket helpers
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> None:
        if self._session is not None and not self._session.closed:
            return

        timeout = aiohttp.ClientTimeout(total=self._ws_config.timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def _close_session(self) -> None:
        if self._session is None:
            return

        try:
            await self._session.close()
        finally:
            self._session = None

    async def _close_ws(
        self,
        ws: aiohttp.ClientWebSocketResponse | None,
        *,
        channel: str,
    ) -> None:
        if ws is None:
            return

        try:
            await ws.close()
        except Exception:
            self._logger.exception(
                "Failed to close Bybit websocket cleanly | channel=%s",
                channel,
            )

    async def _cancel_ping_task(self, *, channel: str) -> None:
        task: asyncio.Task | None

        if channel == "public":
            task = self._public_ping_task
            self._public_ping_task = None
        elif channel == "private":
            task = self._private_ping_task
            self._private_ping_task = None
        else:
            return

        if task is None:
            return

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    # ------------------------------------------------------------------
    # Auth / signing helpers
    # ------------------------------------------------------------------

    def _require_private_credentials(self) -> None:
        if not self._api_key or not self._api_secret:
            raise RuntimeError(
                "Bybit API key and API secret are required for private WebSocket stream"
            )

    def _sign_ws_auth(self, expires: str) -> str:
        self._require_private_credentials()

        assert self._api_secret is not None

        payload = f"GET/realtime{expires}"

        return hmac.new(
            self._api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    # ------------------------------------------------------------------
    # Validation / normalization helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        if self._category not in self.SUPPORTED_CATEGORIES:
            raise ValueError(f"Unsupported Bybit category: {self._category}")

        if not self._symbols:
            raise ValueError("At least one Bybit symbol must be configured")

        unsupported_streams = set(self._streams) - self.SUPPORTED_STREAMS
        if unsupported_streams:
            raise ValueError(f"Unsupported Bybit streams: {sorted(unsupported_streams)}")

        if (
            "orderbook" in self._streams
            and self._ws_config.orderbook_depth not in self.SUPPORTED_ORDERBOOK_DEPTHS
        ):
            raise ValueError(
                f"Unsupported Bybit orderbook depth: {self._ws_config.orderbook_depth}"
            )

        if self._ws_config.enable_private_stream:
            self._require_private_credentials()

    def _should_stop_reconnecting(self, reconnect_attempt: int) -> bool:
        return (
            self._ws_config.max_reconnect_attempts > 0
            and reconnect_attempt >= self._ws_config.max_reconnect_attempts
        )

    @classmethod
    def _normalize_streams(cls, streams: list[str]) -> list[str]:
        normalized = [stream.strip().lower() for stream in streams if stream.strip()]
        return list(dict.fromkeys(normalized))

    @classmethod
    def _resolve_public_url(cls, category: str) -> str:
        category_normalized = category.lower()

        if category_normalized == "linear":
            return "wss://stream.bybit.com/v5/public/linear"

        if category_normalized == "inverse":
            return "wss://stream.bybit.com/v5/public/inverse"

        if category_normalized == "spot":
            return "wss://stream.bybit.com/v5/public/spot"

        if category_normalized == "option":
            return "wss://stream.bybit.com/v5/public/option"

        raise ValueError(f"Unsupported Bybit category: {category}")

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None or value == "":
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None or value == "":
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _current_timestamp_ms() -> int:
        return int(time.time() * 1000)