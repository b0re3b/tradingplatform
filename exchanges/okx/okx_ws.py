from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Optional

import aiohttp

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger


class OkxWebSocketClient:
    """
    OKX WebSocket client (API v5).

    Public channels:
    - trades
    - books / books5
    - candle1m / candleXm

    Private channels:
    - account
    - positions
    - orders

    EventBus topics:
    - market.trade
    - market.orderbook
    - market.candle
    - account.updated
    - position.updated
    - execution.order_updated
    - system.ws.connected
    - system.ws.disconnected
    - system.ws.error
    """

    DEFAULT_PUBLIC_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
    DEFAULT_PRIVATE_WS_URL = "wss://ws.okx.com:8443/ws/v5/private"

    API_KEY_PLACEHOLDER = "OKX_API_KEY_PLACEHOLDER"
    API_SECRET_PLACEHOLDER = "OKX_API_SECRET_PLACEHOLDER"
    PASSPHRASE_PLACEHOLDER = "OKX_PASSPHRASE_PLACEHOLDER"

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        inst_ids: list[str],
        streams: Optional[list[str]] = None,
        orderbook_channel: str = "books5",
        candle_channel: str = "candle1m",
        enable_private_stream: bool = False,
        ping_interval: float = 20.0,
        use_demo: bool = False,
    ) -> None:
        self._config = config
        self._event_bus = event_bus

        self._inst_ids = [inst_id.upper() for inst_id in inst_ids]
        self._streams = streams or ["trades", "books", "candle"]
        self._orderbook_channel = orderbook_channel
        self._candle_channel = candle_channel
        self._enable_private_stream = enable_private_stream
        self._ping_interval = ping_interval
        self._use_demo = use_demo

        self._logger = get_logger(
            __name__,
            exchange="okx",
            event_type="okx_ws",
        )

        self._public_ws_url = self.DEFAULT_PUBLIC_WS_URL
        self._private_ws_url = self.DEFAULT_PRIVATE_WS_URL

        self._api_key = (
            config.exchange.credentials.api_key
            or self.API_KEY_PLACEHOLDER
        )
        self._api_secret = (
            config.exchange.credentials.api_secret
            or self.API_SECRET_PLACEHOLDER
        )
        self._passphrase = (
            config.exchange.credentials.passphrase
            or self.PASSPHRASE_PLACEHOLDER
        )

        self._timeout_seconds = config.exchange.timeout_seconds
        self._reconnect_delay = config.exchange.reconnect_delay
        self._max_reconnect_attempts = config.exchange.max_reconnect_attempts

        self._session: Optional[aiohttp.ClientSession] = None
        self._public_ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._private_ws: Optional[aiohttp.ClientWebSocketResponse] = None

        self._public_task: Optional[asyncio.Task] = None
        self._private_task: Optional[asyncio.Task] = None
        self._public_ping_task: Optional[asyncio.Task] = None
        self._private_ping_task: Optional[asyncio.Task] = None

        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            self._logger.warning("OKX WS client already started")
            return

        self._running = True

        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

        self._logger.info(
            "Starting OKX WS client | inst_ids=%s streams=%s private_stream=%s demo=%s",
            self._inst_ids,
            self._streams,
            self._enable_private_stream,
            self._use_demo,
        )

        self._public_task = asyncio.create_task(
            self._run_public_loop(),
            name="okx-public-ws-loop",
        )

        if self._enable_private_stream:
            self._private_task = asyncio.create_task(
                self._run_private_loop(),
                name="okx-private-ws-loop",
            )

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("OKX WS client already stopped")
            return

        self._running = False
        self._logger.info("Stopping OKX WS client")

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

        self._public_task = None
        self._private_task = None
        self._public_ping_task = None
        self._private_ping_task = None

        await self._close_ws(self._public_ws)
        await self._close_ws(self._private_ws)

        self._public_ws = None
        self._private_ws = None

        if self._session is not None:
            await self._session.close()
            self._session = None

        self._logger.info("OKX WS client stopped")

    # ------------------------------------------------------------------
    # Public WS
    # ------------------------------------------------------------------

    async def _run_public_loop(self) -> None:
        reconnect_attempt = 0

        while self._running:
            try:
                self._logger.info(
                    "Connecting to OKX public WS | url=%s",
                    self._public_ws_url,
                )

                assert self._session is not None
                self._public_ws = await self._session.ws_connect(
                    self._public_ws_url,
                    heartbeat=None,
                    autoping=False,
                    headers=self._build_common_headers(),
                )

                reconnect_attempt = 0

                self._logger.info("Connected to OKX public WS")
                await self._event_bus.emit(
                    "system.ws.connected",
                    {
                        "exchange": "okx",
                        "channel": "public",
                        "inst_ids": self._inst_ids,
                    },
                    priority=EventPriority.HIGH,
                    source="okx_ws",
                )

                await self._subscribe_public_topics()

                self._public_ping_task = asyncio.create_task(
                    self._ping_loop(self._public_ws, "public"),
                    name="okx-public-ws-ping",
                )

                await self._consume_public_messages()

            except asyncio.CancelledError:
                self._logger.info("Public WS loop cancelled")
                raise
            except Exception as exc:
                reconnect_attempt += 1
                self._logger.exception(
                    "Public WS loop error | attempt=%s max_attempts=%s",
                    reconnect_attempt,
                    self._max_reconnect_attempts,
                )

                await self._event_bus.emit(
                    "system.ws.error",
                    {
                        "exchange": "okx",
                        "channel": "public",
                        "error": str(exc),
                        "attempt": reconnect_attempt,
                    },
                    priority=EventPriority.HIGH,
                    source="okx_ws",
                )

                if (
                    self._max_reconnect_attempts > 0
                    and reconnect_attempt >= self._max_reconnect_attempts
                ):
                    self._logger.error("Public WS max reconnect attempts reached")
                    break

                await asyncio.sleep(self._reconnect_delay)

            finally:
                if self._public_ping_task is not None:
                    self._public_ping_task.cancel()
                    await asyncio.gather(self._public_ping_task, return_exceptions=True)
                    self._public_ping_task = None

                await self._close_ws(self._public_ws)
                self._public_ws = None

                if self._running:
                    await self._event_bus.emit(
                        "system.ws.disconnected",
                        {
                            "exchange": "okx",
                            "channel": "public",
                        },
                        priority=EventPriority.HIGH,
                        source="okx_ws",
                    )

    async def _subscribe_public_topics(self) -> None:
        if self._public_ws is None:
            raise RuntimeError("Public websocket is not connected")

        args: list[dict[str, Any]] = []

        for inst_id in self._inst_ids:
            if "trades" in self._streams:
                args.append({"channel": "trades", "instId": inst_id})

            if "books" in self._streams:
                args.append({"channel": self._orderbook_channel, "instId": inst_id})

            if "candle" in self._streams:
                args.append({"channel": self._candle_channel, "instId": inst_id})

        if not args:
            self._logger.warning("No public OKX topics to subscribe")
            return

        payload = {
            "id": self._next_request_id(),
            "op": "subscribe",
            "args": args,
        }

        await self._public_ws.send_json(payload)
        self._logger.info("Subscribed to OKX public topics | args=%s", args)

    async def _consume_public_messages(self) -> None:
        assert self._public_ws is not None

        async for msg in self._public_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_public_message(msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("OKX public websocket error")
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                self._logger.warning("OKX public WS closed by server")
                break

    async def _handle_public_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode OKX public message")
            return

        if self._is_pong(message):
            self._logger.debug("Received OKX public pong")
            return

        if "event" in message:
            await self._handle_ws_event_message(message, "public")
            return

        arg = message.get("arg", {})
        channel = arg.get("channel")
        data = message.get("data", [])

        if not channel:
            self._logger.debug("Received empty or unhandled public payload")
            return

        if channel == "trades":
            await self._publish_trade_events(data)
            return

        if channel in {"books", "books5"}:
            await self._publish_orderbook_event(arg, data)
            return

        if channel.startswith("candle"):
            await self._publish_candle_event(arg, data)
            return

        self._logger.debug("Unhandled public channel | channel=%s", channel)

    # ------------------------------------------------------------------
    # Private WS
    # ------------------------------------------------------------------

    async def _run_private_loop(self) -> None:
        reconnect_attempt = 0

        while self._running:
            try:
                self._logger.info(
                    "Connecting to OKX private WS | url=%s",
                    self._private_ws_url,
                )

                assert self._session is not None
                self._private_ws = await self._session.ws_connect(
                    self._private_ws_url,
                    heartbeat=None,
                    autoping=False,
                    headers=self._build_common_headers(),
                )

                await self._login_private_ws()
                await self._subscribe_private_topics()

                reconnect_attempt = 0

                self._logger.info("Connected to OKX private WS")
                await self._event_bus.emit(
                    "system.ws.connected",
                    {
                        "exchange": "okx",
                        "channel": "private",
                    },
                    priority=EventPriority.HIGH,
                    source="okx_ws",
                )

                self._private_ping_task = asyncio.create_task(
                    self._ping_loop(self._private_ws, "private"),
                    name="okx-private-ws-ping",
                )

                await self._consume_private_messages()

            except asyncio.CancelledError:
                self._logger.info("Private WS loop cancelled")
                raise
            except Exception as exc:
                reconnect_attempt += 1
                self._logger.exception(
                    "Private WS loop error | attempt=%s max_attempts=%s",
                    reconnect_attempt,
                    self._max_reconnect_attempts,
                )

                await self._event_bus.emit(
                    "system.ws.error",
                    {
                        "exchange": "okx",
                        "channel": "private",
                        "error": str(exc),
                        "attempt": reconnect_attempt,
                    },
                    priority=EventPriority.HIGH,
                    source="okx_ws",
                )

                if (
                    self._max_reconnect_attempts > 0
                    and reconnect_attempt >= self._max_reconnect_attempts
                ):
                    self._logger.error("Private WS max reconnect attempts reached")
                    break

                await asyncio.sleep(self._reconnect_delay)

            finally:
                if self._private_ping_task is not None:
                    self._private_ping_task.cancel()
                    await asyncio.gather(self._private_ping_task, return_exceptions=True)
                    self._private_ping_task = None

                await self._close_ws(self._private_ws)
                self._private_ws = None

                if self._running:
                    await self._event_bus.emit(
                        "system.ws.disconnected",
                        {
                            "exchange": "okx",
                            "channel": "private",
                        },
                        priority=EventPriority.HIGH,
                        source="okx_ws",
                    )

    async def _login_private_ws(self) -> None:
        if self._private_ws is None:
            raise RuntimeError("Private websocket is not connected")

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
            raise RuntimeError("Private websocket is not connected")

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
        self._logger.info("Subscribed to OKX private topics | args=%s", args)

    async def _consume_private_messages(self) -> None:
        assert self._private_ws is not None

        async for msg in self._private_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_private_message(msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("OKX private websocket error")
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                self._logger.warning("OKX private WS closed by server")
                break

    async def _handle_private_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode OKX private message")
            return

        if self._is_pong(message):
            self._logger.debug("Received OKX private pong")
            return

        if "event" in message:
            await self._handle_ws_event_message(message, "private")
            return

        arg = message.get("arg", {})
        channel = arg.get("channel")
        data = message.get("data", [])

        if not channel:
            self._logger.debug("Received empty or unhandled private payload")
            return

        if channel == "account":
            await self._publish_account_event(data)
            return

        if channel == "positions":
            await self._publish_position_event(data)
            return

        if channel == "orders":
            await self._publish_order_event(data)
            return

        self._logger.debug("Unhandled private channel | channel=%s", channel)

    # ------------------------------------------------------------------
    # Shared handlers
    # ------------------------------------------------------------------

    async def _handle_ws_event_message(self, message: dict[str, Any], channel_type: str) -> None:
        event = message.get("event")

        if event == "error":
            raise RuntimeError(f"OKX websocket error ({channel_type}): {message}")

        if event == "login":
            code = message.get("code")
            if str(code) not in {"0", ""}:
                raise RuntimeError(f"OKX login failed: {message}")

            self._logger.info("OKX private auth succeeded")
            return

        if event == "subscribe":
            self._logger.info(
                "OKX subscribe response | channel_type=%s arg=%s",
                channel_type,
                message.get("arg"),
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
                await asyncio.sleep(self._ping_interval)
                await ws.send_str("ping")
                self._logger.debug("Sent OKX ping | channel=%s", channel)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception("OKX ping loop failed | channel=%s", channel)
            raise

    # ------------------------------------------------------------------
    # Event publishers - public
    # ------------------------------------------------------------------

    async def _publish_trade_events(self, data: list[dict[str, Any]]) -> None:
        for trade in data:
            payload = {
                "exchange": "okx",
                "inst_id": trade.get("instId"),
                "trade_id": trade.get("tradeId"),
                "price": self._safe_float(trade.get("px")),
                "qty": self._safe_float(trade.get("sz")),
                "side": (trade.get("side") or "").lower(),
                "trade_time": self._safe_int(trade.get("ts")),
            }

            await self._event_bus.emit(
                "market.trade",
                payload,
                priority=EventPriority.NORMAL,
                source="okx_ws",
            )

    async def _publish_orderbook_event(
        self,
        arg: dict[str, Any],
        data: list[dict[str, Any]],
    ) -> None:
        if not data:
            return

        snapshot = data[0]
        payload = {
            "exchange": "okx",
            "inst_id": arg.get("instId"),
            "channel": arg.get("channel"),
            "asks": [self._normalize_book_level(level) for level in snapshot.get("asks", [])],
            "bids": [self._normalize_book_level(level) for level in snapshot.get("bids", [])],
            "checksum": snapshot.get("checksum"),
            "seq_id": snapshot.get("seqId"),
            "prev_seq_id": snapshot.get("prevSeqId"),
            "timestamp": self._safe_int(snapshot.get("ts")),
        }

        await self._event_bus.emit(
            "market.orderbook",
            payload,
            priority=EventPriority.LOW,
            source="okx_ws",
        )

    async def _publish_candle_event(
        self,
        arg: dict[str, Any],
        data: list[list[Any]],
    ) -> None:
        inst_id = arg.get("instId")
        channel = arg.get("channel")

        for candle in data:
            if len(candle) < 9:
                continue

            payload = {
                "exchange": "okx",
                "inst_id": inst_id,
                "channel": channel,
                "open_time": self._safe_int(candle[0]),
                "open": self._safe_float(candle[1]),
                "high": self._safe_float(candle[2]),
                "low": self._safe_float(candle[3]),
                "close": self._safe_float(candle[4]),
                "volume": self._safe_float(candle[5]),
                "volume_ccy": self._safe_float(candle[6]),
                "volume_ccy_quote": self._safe_float(candle[7]),
                "confirm": candle[8],
            }

            await self._event_bus.emit(
                "market.candle",
                payload,
                priority=EventPriority.NORMAL,
                source="okx_ws",
            )

    # ------------------------------------------------------------------
    # Event publishers - private
    # ------------------------------------------------------------------

    async def _publish_account_event(self, data: list[dict[str, Any]]) -> None:
        for item in data:
            payload = {
                "exchange": "okx",
                "u_time": self._safe_int(item.get("uTime")),
                "total_eq": self._safe_float(item.get("totalEq")),
                "iso_eq": self._safe_float(item.get("isoEq")),
                "adj_eq": self._safe_float(item.get("adjEq")),
                "imr": self._safe_float(item.get("imr")),
                "mmr": self._safe_float(item.get("mmr")),
                "mgn_ratio": self._safe_float(item.get("mgnRatio")),
                "details": item.get("details", []),
            }

            await self._event_bus.emit(
                "account.updated",
                payload,
                priority=EventPriority.HIGH,
                source="okx_ws",
            )

    async def _publish_position_event(self, data: list[dict[str, Any]]) -> None:
        for item in data:
            payload = {
                "exchange": "okx",
                "inst_id": item.get("instId"),
                "pos_side": item.get("posSide"),
                "position": self._safe_float(item.get("pos")),
                "avg_px": self._safe_float(item.get("avgPx")),
                "mark_px": self._safe_float(item.get("markPx")),
                "liq_px": self._safe_float(item.get("liqPx")),
                "upl": self._safe_float(item.get("upl")),
                "lever": self._safe_float(item.get("lever")),
                "margin": self._safe_float(item.get("margin")),
                "u_time": self._safe_int(item.get("uTime")),
            }

            await self._event_bus.emit(
                "position.updated",
                payload,
                priority=EventPriority.HIGH,
                source="okx_ws",
            )

    async def _publish_order_event(self, data: list[dict[str, Any]]) -> None:
        for item in data:
            payload = {
                "exchange": "okx",
                "inst_id": item.get("instId"),
                "ord_id": item.get("ordId"),
                "cl_ord_id": item.get("clOrdId"),
                "side": item.get("side"),
                "ord_type": item.get("ordType"),
                "state": item.get("state"),
                "px": self._safe_float(item.get("px")),
                "sz": self._safe_float(item.get("sz")),
                "acc_fill_sz": self._safe_float(item.get("accFillSz")),
                "avg_px": self._safe_float(item.get("avgPx")),
                "fill_px": self._safe_float(item.get("fillPx")),
                "fill_sz": self._safe_float(item.get("fillSz")),
                "fee": self._safe_float(item.get("fee")),
                "u_time": self._safe_int(item.get("uTime")),
                "c_time": self._safe_int(item.get("cTime")),
            }

            await self._event_bus.emit(
                "execution.order_updated",
                payload,
                priority=EventPriority.HIGH,
                source="okx_ws",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_common_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._use_demo:
            headers["x-simulated-trading"] = "1"
        return headers

    def _sign_login(self, timestamp: str) -> str:
        """
        OKX WS login sign:
        Base64(HMAC_SHA256(secret, f"{timestamp}GET/users/self/verify"))
        """
        payload = f"{timestamp}GET/users/self/verify"
        digest = hmac.new(
            self._api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode("utf-8")

    @staticmethod
    def _is_pong(message: dict[str, Any]) -> bool:
        return message.get("event") == "pong" or message.get("op") == "pong"

    @staticmethod
    def _next_request_id() -> str:
        return uuid.uuid4().hex[:12]

    async def _close_ws(self, ws: Optional[aiohttp.ClientWebSocketResponse]) -> None:
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            self._logger.exception("Failed to close websocket cleanly")

    @staticmethod
    def _normalize_book_level(level: list[Any]) -> list[Optional[float]]:
        normalized: list[Optional[float]] = []
        for item in level:
            try:
                normalized.append(float(item))
            except (TypeError, ValueError):
                normalized.append(None)
        return normalized

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None