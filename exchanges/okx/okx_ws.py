from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger


@dataclass(slots=True)
class OkxWebSocketClientConfig:
    """
    Local OKX WS adapter config.

    Global credentials and exchange-level timeouts still come from core.config.Config.exchange.
    This config contains only OKX WebSocket adapter-specific settings.
    """

    public_ws_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    private_ws_url: str = "wss://ws.okx.com:8443/ws/v5/private"

    timeout_seconds: float = 10.0
    reconnect_delay_seconds: float = 5.0
    max_reconnect_attempts: int = 20

    ping_interval_seconds: float = 20.0
    use_demo: bool = False

    inst_ids: list[str] = field(default_factory=list)
    streams: list[str] = field(default_factory=lambda: ["trades", "books", "candle"])

    orderbook_channel: str = "books5"
    candle_channel: str = "candle1m"
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
        inst_ids: list[str],
        streams: list[str] | None = None,
        orderbook_channel: str = "books5",
        candle_channel: str = "candle1m",
        enable_private_stream: bool = False,
        ping_interval_seconds: float = 20.0,
        use_demo: bool = False,
        orderbook_emit_min_interval_ms: int = 250,
        orderbook_batch_max_size: int = 500,
        trade_emit_min_interval_ms: int = 250,
        trade_batch_max_size: int = 1000,
    ) -> "OkxWebSocketClientConfig":
        return cls(
            timeout_seconds=config.exchange.timeout_seconds,
            reconnect_delay_seconds=config.exchange.reconnect_delay,
            max_reconnect_attempts=config.exchange.max_reconnect_attempts,
            ping_interval_seconds=ping_interval_seconds,
            use_demo=use_demo or config.exchange.credentials.testnet,
            inst_ids=inst_ids,
            streams=streams or ["trades", "books", "candle"],
            orderbook_channel=orderbook_channel,
            candle_channel=candle_channel,
            enable_private_stream=enable_private_stream,
            orderbook_emit_min_interval_ms=orderbook_emit_min_interval_ms,
            orderbook_batch_max_size=orderbook_batch_max_size,
            trade_emit_min_interval_ms=trade_emit_min_interval_ms,
            trade_batch_max_size=trade_batch_max_size,
        )


class OkxWebSocketClient:
    """
    OKX WebSocket exchange adapter.

    Responsibilities:
    - connect to OKX public WebSocket channels;
    - optionally connect to OKX private WebSocket channels;
    - normalize raw OKX payloads into internal market/exchange events;
    - write high-frequency market data into MarketIngestionService/MarketStateStore;
    - publish only lifecycle/private exchange events through EventBus;
    - never call analytics, strategy, risk, or execution directly;
    - never contain trading decision logic.

    Public market events:
    - market.trade
    - market.orderbook
    - market.candle

    Private exchange events:
    - exchange.account.updated
    - exchange.position.updated
    - exchange.order.updated

    System events:
    - system.exchange.ws.started
    - system.exchange.ws.stopped
    - system.exchange.ws.connected
    - system.exchange.ws.disconnected
    - system.exchange.ws.error
    - system.exchange.ws.authenticated
    - system.exchange.ws.subscribed
    """

    EXCHANGE = "okx"
    SOURCE = "okx_ws"

    SUPPORTED_STREAMS = {"trades", "books", "candle"}
    SUPPORTED_PUBLIC_BOOK_CHANNELS = {"books", "books5", "books-l2-tbt", "books50-l2-tbt"}

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        ws_config: OkxWebSocketClientConfig | None = None,
        inst_ids: list[str] | None = None,
        streams: list[str] | None = None,
        orderbook_channel: str = "books5",
        candle_channel: str = "candle1m",
        enable_private_stream: bool = False,
        ping_interval: float = 20.0,
        use_demo: bool = False,
        orderbook_emit_min_interval_ms: int = 250,
        orderbook_batch_max_size: int = 500,
        trade_emit_min_interval_ms: int = 250,
        trade_batch_max_size: int = 1000,
        market_ingestion: MarketIngestionService | None = None,
    ) -> None:
        resolved_config = ws_config or OkxWebSocketClientConfig.from_core_config(
            config=config,
            inst_ids=inst_ids or [],
            streams=streams,
            orderbook_channel=orderbook_channel,
            candle_channel=candle_channel,
            enable_private_stream=enable_private_stream,
            ping_interval_seconds=ping_interval,
            use_demo=use_demo,
            orderbook_emit_min_interval_ms=orderbook_emit_min_interval_ms,
            orderbook_batch_max_size=orderbook_batch_max_size,
            trade_emit_min_interval_ms=trade_emit_min_interval_ms,
            trade_batch_max_size=trade_batch_max_size,
        )

        self._config = config
        self._event_bus = event_bus
        self._ws_config = resolved_config

        self._inst_ids = [inst_id.upper() for inst_id in self._ws_config.inst_ids]
        self._streams = self._normalize_streams(self._ws_config.streams)

        self._api_key = config.exchange.credentials.api_key
        self._api_secret = config.exchange.credentials.api_secret
        self._passphrase = config.exchange.credentials.passphrase

        self._logger = get_logger(
            __name__,
            exchange=self.EXCHANGE,
            event_type="exchange_ws",
        )

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
        self._logger.debug("OKX WS register called | subscriptions=0")

    async def start(self) -> None:
        if self._running:
            self._logger.warning("OKX WS client already started")
            return

        self._running = True
        self._started = True

        await self._ensure_session()

        self._logger.info(
            "Starting OKX WS client | inst_ids=%s streams=%s private_stream=%s demo=%s",
            self._inst_ids,
            self._streams,
            self._ws_config.enable_private_stream,
            self._ws_config.use_demo,
        )

        await self._emit_event(
            "system.exchange.ws.started",
            {
                "exchange": self.EXCHANGE,
                "inst_ids": self._inst_ids,
                "streams": self._streams,
                "private_stream": self._ws_config.enable_private_stream,
                "demo": self._ws_config.use_demo,
            },
            priority=EventPriority.NORMAL,
        )

        self._public_task = asyncio.create_task(
            self._run_public_loop(),
            name="okx-public-ws-loop",
        )

        if self._ws_config.enable_private_stream:
            self._require_private_credentials()

            self._private_task = asyncio.create_task(
                self._run_private_loop(),
                name="okx-private-ws-loop",
            )

    async def stop(self) -> None:
        if not self._running and not self._started:
            self._logger.warning("OKX WS client already stopped")
            return

        self._logger.info("Stopping OKX WS client")

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

        self._logger.info("OKX WS client stopped")

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
                    "Connecting to OKX public WS | url=%s",
                    self._ws_config.public_ws_url,
                )

                assert self._session is not None
                self._public_ws = await self._session.ws_connect(
                    self._ws_config.public_ws_url,
                    heartbeat=None,
                    autoping=False,
                    headers=self._build_common_headers(),
                )

                reconnect_attempt = 0

                self._logger.info("Connected to OKX public WS")

                await self._emit_event(
                    "system.exchange.ws.connected",
                    {
                        "exchange": self.EXCHANGE,
                        "channel": "public",
                        "inst_ids": self._inst_ids,
                    },
                    priority=EventPriority.HIGH,
                )

                await self._subscribe_public_topics()

                self._public_ping_task = asyncio.create_task(
                    self._ping_loop(self._public_ws, "public"),
                    name="okx-public-ws-ping",
                )

                await self._consume_public_messages()

            except asyncio.CancelledError:
                self._logger.info("OKX public WS loop cancelled")
                raise
            except Exception as exc:
                reconnect_attempt += 1

                await self._handle_ws_loop_error(
                    channel="public",
                    exc=exc,
                    reconnect_attempt=reconnect_attempt,
                )

                if self._should_stop_reconnecting(reconnect_attempt):
                    self._logger.error("OKX public WS max reconnect attempts reached")
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
            raise RuntimeError("OKX public WebSocket is not connected")

        args: list[dict[str, Any]] = []

        for inst_id in self._inst_ids:
            if "trades" in self._streams:
                args.append({"channel": "trades", "instId": inst_id})

            if "books" in self._streams:
                args.append(
                    {
                        "channel": self._ws_config.orderbook_channel,
                        "instId": inst_id,
                    }
                )

            if "candle" in self._streams:
                args.append(
                    {
                        "channel": self._ws_config.candle_channel,
                        "instId": inst_id,
                    }
                )

        if not args:
            raise RuntimeError("No OKX public topics to subscribe")

        payload = {
            "id": self._next_request_id(),
            "op": "subscribe",
            "args": args,
        }

        await self._public_ws.send_json(payload)

        self._logger.info(
            "Subscribed to OKX public topics | count=%s",
            len(args),
        )

    async def _consume_public_messages(self) -> None:
        assert self._public_ws is not None

        async for msg in self._public_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_public_message(msg.data)
                continue

            if msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("OKX public WebSocket error")

            if msg.type in {
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            }:
                self._logger.warning("OKX public WS closed by server")
                break

    async def _handle_public_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode OKX public WS message")
            return

        if self._is_pong(message):
            self._logger.debug("Received OKX public pong")
            return

        if "event" in message:
            await self._handle_ws_event_message(message, "public")
            return

        arg = message.get("arg")
        data = message.get("data")

        if not isinstance(arg, dict) or not isinstance(data, list):
            self._logger.debug("Received malformed OKX public payload")
            return

        channel = arg.get("channel")

        if channel == "trades":
            await self._publish_trade_events(data)
            return

        if channel in self.SUPPORTED_PUBLIC_BOOK_CHANNELS:
            await self._publish_orderbook_event(arg, data)
            return

        if isinstance(channel, str) and channel.startswith("candle"):
            await self._publish_candle_events(arg, data)
            return

        self._logger.debug(
            "Unhandled OKX public channel | channel=%s",
            channel,
        )

    # ------------------------------------------------------------------
    # Private WS loop
    # ------------------------------------------------------------------

    async def _run_private_loop(self) -> None:
        reconnect_attempt = 0

        while self._running:
            try:
                await self._ensure_session()
                self._require_private_credentials()

                self._logger.info(
                    "Connecting to OKX private WS | url=%s",
                    self._ws_config.private_ws_url,
                )

                assert self._session is not None
                self._private_ws = await self._session.ws_connect(
                    self._ws_config.private_ws_url,
                    heartbeat=None,
                    autoping=False,
                    headers=self._build_common_headers(),
                )

                await self._login_private_ws()
                await self._subscribe_private_topics()

                reconnect_attempt = 0

                self._logger.info("Connected to OKX private WS")

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
                    name="okx-private-ws-ping",
                )

                await self._consume_private_messages()

            except asyncio.CancelledError:
                self._logger.info("OKX private WS loop cancelled")
                raise
            except Exception as exc:
                reconnect_attempt += 1

                await self._handle_ws_loop_error(
                    channel="private",
                    exc=exc,
                    reconnect_attempt=reconnect_attempt,
                )

                if self._should_stop_reconnecting(reconnect_attempt):
                    self._logger.error("OKX private WS max reconnect attempts reached")
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

    async def _login_private_ws(self) -> None:
        if self._private_ws is None:
            raise RuntimeError("OKX private WebSocket is not connected")

        self._require_private_credentials()

        assert self._api_key is not None
        assert self._passphrase is not None

        timestamp = str(int(time.time()))
        sign = self._sign_login(timestamp)

        payload = {
            "id": self._next_request_id(),
            "op": "login",
            "args": [
                {
                    "apiKey": self._api_key,
                    "passphrase": self._passphrase,
                    "timestamp": timestamp,
                    "sign": sign,
                }
            ],
        }

        await self._private_ws.send_json(payload)
        self._logger.info("OKX private WS login request sent")

    async def _subscribe_private_topics(self) -> None:
        if self._private_ws is None:
            raise RuntimeError("OKX private WebSocket is not connected")

        args = [
            {"channel": "account"},
            {"channel": "positions"},
            {"channel": "orders"},
        ]

        payload = {
            "id": self._next_request_id(),
            "op": "subscribe",
            "args": args,
        }

        await self._private_ws.send_json(payload)

        self._logger.info(
            "Subscribed to OKX private topics | count=%s",
            len(args),
        )

    async def _consume_private_messages(self) -> None:
        assert self._private_ws is not None

        async for msg in self._private_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_private_message(msg.data)
                continue

            if msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("OKX private WebSocket error")

            if msg.type in {
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            }:
                self._logger.warning("OKX private WS closed by server")
                break

    async def _handle_private_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode OKX private WS message")
            return

        if self._is_pong(message):
            self._logger.debug("Received OKX private pong")
            return

        if "event" in message:
            await self._handle_ws_event_message(message, "private")
            return

        arg = message.get("arg")
        data = message.get("data")

        if not isinstance(arg, dict) or not isinstance(data, list):
            self._logger.debug("Received malformed OKX private payload")
            return

        channel = arg.get("channel")

        if channel == "account":
            await self._publish_account_events(data)
            return

        if channel == "positions":
            await self._publish_position_events(data)
            return

        if channel == "orders":
            await self._publish_order_events(data)
            return

        self._logger.debug(
            "Unhandled OKX private channel | channel=%s",
            channel,
        )

    # ------------------------------------------------------------------
    # Shared WS event handlers
    # ------------------------------------------------------------------

    async def _handle_ws_event_message(
        self,
        message: dict[str, Any],
        channel_type: str,
    ) -> None:
        event = message.get("event")

        if event == "error":
            code = str(message.get("code") or "")
            msg = str(message.get("msg") or "")
            arg = message.get("arg")

            await self._emit_event(
                "system.exchange.ws.error",
                {
                    "exchange": self.EXCHANGE,
                    "channel": channel_type,
                    "code": code,
                    "message": msg,
                    "arg": arg,
                },
                priority=EventPriority.HIGH,
            )

            # OKX code 60018 means a single requested channel/instId pair is invalid.
            # Example: candle1m for a specific SWAP instrument can be rejected while
            # trades/books subscriptions in the same shard are still useful.
            #
            # Do not restart the entire WebSocket loop for one bad subscription.
            # Keep the connection alive and let valid subscriptions continue streaming.
            if code == "60018":
                self._logger.warning(
                    "OKX subscription rejected; keeping WS connection alive | "
                    "channel_type=%s code=%s msg=%s arg=%s",
                    channel_type,
                    code,
                    msg,
                    arg,
                )
                return

            raise RuntimeError(
                f"OKX WebSocket error | channel={channel_type} code={code} msg={msg}"
            )

        if event == "login":
            code = message.get("code")
            if str(code) not in {"0", ""}:
                raise RuntimeError(
                    f"OKX private login failed | code={code} msg={message.get('msg')}"
                )

            self._logger.info("OKX private WS auth succeeded")

            await self._emit_event(
                "system.exchange.ws.authenticated",
                {
                    "exchange": self.EXCHANGE,
                    "channel": "private",
                },
                priority=EventPriority.HIGH,
            )

            return

        if event == "subscribe":
            self._logger.info(
                "OKX subscribe response | channel_type=%s arg=%s",
                channel_type,
                message.get("arg"),
            )

            await self._emit_event(
                "system.exchange.ws.subscribed",
                {
                    "exchange": self.EXCHANGE,
                    "channel_type": channel_type,
                    "arg": message.get("arg"),
                },
                priority=EventPriority.LOW,
            )

            return

        if event == "unsubscribe":
            self._logger.info(
                "OKX unsubscribe response | channel_type=%s arg=%s",
                channel_type,
                message.get("arg"),
            )
            return

        self._logger.debug(
            "Unhandled OKX event message | channel_type=%s event=%s",
            channel_type,
            event,
        )

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

                await ws.send_str("ping")
                self._logger.debug("Sent OKX ping | channel=%s", channel)

        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception(
                "OKX ping loop failed | channel=%s",
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
            if self._market_ingestion is not None:
                await self._market_ingestion.ingest_orderbook_delta(payload)
            else:
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

            if self._market_ingestion is not None:
                # Preserve delta order while avoiding EventBus raw market-data flood.
                for update in updates:
                    await self._market_ingestion.ingest_orderbook_delta(update)
            else:
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
            if self._market_ingestion is not None:
                await self._market_ingestion.ingest_trade(payload)
            else:
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

            if self._market_ingestion is not None:
                await self._market_ingestion.ingest_trades_batch(batch_payload)
            else:
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

    async def _publish_trade_events(self, data: list[dict[str, Any]]) -> None:
        for trade in data:
            inst_id = trade.get("instId")

            payload = {
                "exchange": self.EXCHANGE,
                "market_type": "swap",
                "symbol": inst_id,
                "inst_id": inst_id,
                "trade_id": trade.get("tradeId"),
                "price": self._safe_float(trade.get("px")),
                "qty": self._safe_float(trade.get("sz")),
                "side": (trade.get("side") or "").lower(),
                "trade_time": self._safe_int(trade.get("ts")),
            }

            await self._emit_trade_event_coalesced(
                key=str(payload.get("symbol") or payload.get("inst_id") or "unknown").upper(),
                payload=payload,
            )

    async def _publish_orderbook_event(
        self,
        arg: dict[str, Any],
        data: list[dict[str, Any]],
    ) -> None:
        if not data:
            return

        snapshot = data[0]
        inst_id = arg.get("instId")

        payload = {
            "exchange": self.EXCHANGE,
            "market_type": "swap",
            "symbol": inst_id,
            "inst_id": inst_id,
            "channel": arg.get("channel"),
            "asks": [
                self._normalize_book_level(level)
                for level in snapshot.get("asks", [])
            ],
            "bids": [
                self._normalize_book_level(level)
                for level in snapshot.get("bids", [])
            ],
            "checksum": snapshot.get("checksum"),
            "seq_id": snapshot.get("seqId"),
            "prev_seq_id": snapshot.get("prevSeqId"),
            "timestamp": self._safe_int(snapshot.get("ts")),
        }

        await self._emit_orderbook_event_coalesced(
            key=str(payload.get("symbol") or payload.get("inst_id") or "unknown").upper(),
            payload=payload,
        )

    async def _publish_candle_events(
        self,
        arg: dict[str, Any],
        data: list[list[Any]],
    ) -> None:
        inst_id = arg.get("instId")
        channel = arg.get("channel")
        timeframe = self._timeframe_from_candle_channel(channel)

        for candle in data:
            if len(candle) < 9:
                continue

            payload = {
                "exchange": self.EXCHANGE,
                "market_type": "swap",
                "symbol": inst_id,
                "inst_id": inst_id,
                "timeframe": timeframe,
                "channel": channel,
                "open_time": self._safe_int(candle[0]),
                "close_time": None,
                "open": self._safe_float(candle[1]),
                "high": self._safe_float(candle[2]),
                "low": self._safe_float(candle[3]),
                "close": self._safe_float(candle[4]),
                "volume": self._safe_float(candle[5]),
                "volume_ccy": self._safe_float(candle[6]),
                "quote_volume": self._safe_float(candle[7]),
                "trades_count": None,
                "is_closed": str(candle[8]) == "1",
                "confirm": candle[8],
            }

            await self._emit_event(
                "market.candle",
                payload,
                priority=EventPriority.HIGH,
            )

    # ------------------------------------------------------------------
    # Event publishers: private exchange updates
    # ------------------------------------------------------------------

    async def _publish_account_events(self, data: list[dict[str, Any]]) -> None:
        for item in data:
            payload = {
                "exchange": self.EXCHANGE,
                "u_time": self._safe_int(item.get("uTime")),
                "total_eq": self._safe_float(item.get("totalEq")),
                "iso_eq": self._safe_float(item.get("isoEq")),
                "adj_eq": self._safe_float(item.get("adjEq")),
                "imr": self._safe_float(item.get("imr")),
                "mmr": self._safe_float(item.get("mmr")),
                "mgn_ratio": self._safe_float(item.get("mgnRatio")),
                "details": item.get("details", []),
            }

            await self._emit_event(
                "exchange.account.updated",
                payload,
                priority=EventPriority.HIGH,
            )

    async def _publish_position_events(self, data: list[dict[str, Any]]) -> None:
        for item in data:
            inst_id = item.get("instId")

            payload = {
                "exchange": self.EXCHANGE,
                "symbol": inst_id,
                "inst_id": inst_id,
                "pos_id": item.get("posId"),
                "pos_side": item.get("posSide"),
                "position": self._safe_float(item.get("pos")),
                "avg_px": self._safe_float(item.get("avgPx")),
                "mark_px": self._safe_float(item.get("markPx")),
                "liq_px": self._safe_float(item.get("liqPx")),
                "upl": self._safe_float(item.get("upl")),
                "upl_ratio": self._safe_float(item.get("uplRatio")),
                "lever": self._safe_float(item.get("lever")),
                "margin": self._safe_float(item.get("margin")),
                "mgn_mode": item.get("mgnMode"),
                "u_time": self._safe_int(item.get("uTime")),
                "c_time": self._safe_int(item.get("cTime")),
            }

            await self._emit_event(
                "exchange.position.updated",
                payload,
                priority=EventPriority.HIGH,
            )

    async def _publish_order_events(self, data: list[dict[str, Any]]) -> None:
        for item in data:
            inst_id = item.get("instId")

            payload = {
                "exchange": self.EXCHANGE,
                "symbol": inst_id,
                "inst_id": inst_id,
                "ord_id": item.get("ordId"),
                "cl_ord_id": item.get("clOrdId"),
                "tag": item.get("tag"),
                "side": item.get("side"),
                "pos_side": item.get("posSide"),
                "td_mode": item.get("tdMode"),
                "ord_type": item.get("ordType"),
                "state": item.get("state"),
                "px": self._safe_float(item.get("px")),
                "sz": self._safe_float(item.get("sz")),
                "acc_fill_sz": self._safe_float(item.get("accFillSz")),
                "avg_px": self._safe_float(item.get("avgPx")),
                "fill_px": self._safe_float(item.get("fillPx")),
                "fill_sz": self._safe_float(item.get("fillSz")),
                "fill_time": self._safe_int(item.get("fillTime")),
                "fee": self._safe_float(item.get("fee")),
                "fee_ccy": item.get("feeCcy"),
                "rebate": self._safe_float(item.get("rebate")),
                "rebate_ccy": item.get("rebateCcy"),
                "pnl": self._safe_float(item.get("pnl")),
                "source": item.get("source"),
                "category": item.get("category"),
                "u_time": self._safe_int(item.get("uTime")),
                "c_time": self._safe_int(item.get("cTime")),
            }

            await self._emit_event(
                "exchange.order.updated",
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
                "Failed to emit OKX WS event | topic=%s",
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
            "OKX WS loop error | channel=%s attempt=%s max_attempts=%s",
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
                "Failed to close OKX websocket cleanly | channel=%s",
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

    def _build_common_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}

        if self._ws_config.use_demo:
            headers["x-simulated-trading"] = "1"

        return headers

    def _require_private_credentials(self) -> None:
        if not self._api_key or not self._api_secret or not self._passphrase:
            raise RuntimeError(
                "OKX API key, API secret and passphrase are required for private WebSocket stream"
            )

    def _sign_login(self, timestamp: str) -> str:
        """
        OKX WS login sign:
        Base64(HMAC_SHA256(secret, f"{timestamp}GET/users/self/verify"))
        """
        self._require_private_credentials()

        assert self._api_secret is not None

        payload = f"{timestamp}GET/users/self/verify"
        digest = hmac.new(
            self._api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        return base64.b64encode(digest).decode("utf-8")

    # ------------------------------------------------------------------
    # Validation / normalization helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        if not self._inst_ids:
            raise ValueError("At least one OKX instrument id must be configured")

        unsupported = set(self._streams) - self.SUPPORTED_STREAMS
        if unsupported:
            raise ValueError(f"Unsupported OKX streams: {sorted(unsupported)}")

        if (
            "books" in self._streams
            and self._ws_config.orderbook_channel not in self.SUPPORTED_PUBLIC_BOOK_CHANNELS
        ):
            raise ValueError(
                f"Unsupported OKX orderbook channel: {self._ws_config.orderbook_channel}"
            )

        if "candle" in self._streams and not self._ws_config.candle_channel.startswith("candle"):
            raise ValueError(
                f"Unsupported OKX candle channel: {self._ws_config.candle_channel}"
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

    @staticmethod
    def _timeframe_from_candle_channel(channel: Any) -> str | None:
        if not isinstance(channel, str):
            return None

        if not channel.startswith("candle"):
            return None

        return channel.replace("candle", "", 1) or None

    @staticmethod
    def _is_pong(message: dict[str, Any]) -> bool:
        return message.get("event") == "pong" or message.get("op") == "pong"

    @staticmethod
    def _next_request_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _normalize_book_level(level: list[Any]) -> list[float | int | None]:
        """
        OKX book level format is usually:
        [price, size, liquidated_orders, order_count]
        """
        normalized: list[float | int | None] = []

        for index, item in enumerate(level):
            if index <= 1:
                normalized.append(OkxWebSocketClient._safe_float(item))
            else:
                normalized.append(OkxWebSocketClient._safe_int(item))

        return normalized

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