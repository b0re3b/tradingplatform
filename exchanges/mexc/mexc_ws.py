from __future__ import annotations

import asyncio
import gzip
import hashlib
import hmac
import json
import time
from typing import Any, Optional

import aiohttp

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger


class MexcWebSocketClient:
    """
    MEXC Futures WebSocket client.

    Public channels:
    - sub.deal
    - sub.depth
    - sub.kline

    Private channels:
    - login
    - personal.filter (order / order.deal / position / asset ...)

    EventBus topics:
    - market.trade
    - market.orderbook
    - market.candle
    - execution.order_updated
    - execution.deal_updated
    - position.updated
    - account.asset_updated
    - system.ws.connected
    - system.ws.disconnected
    - system.ws.error
    """

    DEFAULT_WS_URL = "wss://contract.mexc.com/edge"

    API_KEY_PLACEHOLDER = "MEXC_API_KEY_PLACEHOLDER"
    API_SECRET_PLACEHOLDER = "MEXC_API_SECRET_PLACEHOLDER"

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        symbols: list[str],
        streams: Optional[list[str]] = None,
        kline_interval: str = "Min1",
        enable_private_stream: bool = False,
        ping_interval: float = 15.0,
        recv_window_ms: int = 10000,
        disable_default_private_pushes: bool = False,
    ) -> None:
        self._config = config
        self._event_bus = event_bus

        self._symbols = [self._normalize_symbol(symbol) for symbol in symbols]
        self._streams = streams or ["deal", "depth", "kline"]
        self._kline_interval = kline_interval
        self._enable_private_stream = enable_private_stream
        self._ping_interval = ping_interval
        self._recv_window_ms = recv_window_ms
        self._disable_default_private_pushes = disable_default_private_pushes

        self._logger = get_logger(
            __name__,
            exchange="mexc",
            event_type="mexc_ws",
        )

        self._ws_url = config.exchange.ws_url or self.DEFAULT_WS_URL
        self._api_key = config.exchange.credentials.api_key or self.API_KEY_PLACEHOLDER
        self._api_secret = (
            config.exchange.credentials.api_secret or self.API_SECRET_PLACEHOLDER
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
            self._logger.warning("MEXC WS client already started")
            return

        self._running = True

        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

        self._logger.info(
            "Starting MEXC WS client | symbols=%s streams=%s private_stream=%s",
            self._symbols,
            self._streams,
            self._enable_private_stream,
        )

        self._public_task = asyncio.create_task(
            self._run_public_loop(),
            name="mexc-public-ws-loop",
        )

        if self._enable_private_stream:
            self._private_task = asyncio.create_task(
                self._run_private_loop(),
                name="mexc-private-ws-loop",
            )

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("MEXC WS client already stopped")
            return

        self._running = False
        self._logger.info("Stopping MEXC WS client")

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

        self._logger.info("MEXC WS client stopped")

    # ------------------------------------------------------------------
    # Public WS
    # ------------------------------------------------------------------

    async def _run_public_loop(self) -> None:
        reconnect_attempt = 0

        while self._running:
            try:
                self._logger.info("Connecting to MEXC public WS | url=%s", self._ws_url)

                assert self._session is not None
                self._public_ws = await self._session.ws_connect(
                    self._ws_url,
                    heartbeat=None,
                    autoping=False,
                )

                reconnect_attempt = 0

                self._logger.info("Connected to MEXC public WS")
                await self._event_bus.emit(
                    "system.ws.connected",
                    {
                        "exchange": "mexc",
                        "channel": "public",
                        "symbols": self._symbols,
                    },
                    priority=EventPriority.HIGH,
                    source="mexc_ws",
                )

                await self._subscribe_public_topics()

                self._public_ping_task = asyncio.create_task(
                    self._ping_loop(self._public_ws, "public"),
                    name="mexc-public-ws-ping",
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
                        "exchange": "mexc",
                        "channel": "public",
                        "error": str(exc),
                        "attempt": reconnect_attempt,
                    },
                    priority=EventPriority.HIGH,
                    source="mexc_ws",
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
                            "exchange": "mexc",
                            "channel": "public",
                        },
                        priority=EventPriority.HIGH,
                        source="mexc_ws",
                    )

    async def _subscribe_public_topics(self) -> None:
        if self._public_ws is None:
            raise RuntimeError("Public websocket is not connected")

        for symbol in self._symbols:
            if "deal" in self._streams:
                await self._send_ws_json(
                    self._public_ws,
                    {
                        "method": "sub.deal",
                        "param": {"symbol": symbol},
                    },
                )

            if "depth" in self._streams:
                await self._send_ws_json(
                    self._public_ws,
                    {
                        "method": "sub.depth",
                        "param": {"symbol": symbol},
                    },
                )

            if "kline" in self._streams:
                await self._send_ws_json(
                    self._public_ws,
                    {
                        "method": "sub.kline",
                        "param": {
                            "symbol": symbol,
                            "interval": self._kline_interval,
                        },
                        "gzip": False,
                    },
                )

        self._logger.info(
            "Subscribed to MEXC public topics | symbols=%s streams=%s",
            self._symbols,
            self._streams,
        )

    async def _consume_public_messages(self) -> None:
        assert self._public_ws is not None

        async for msg in self._public_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_public_message(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await self._handle_public_binary_message(msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("MEXC public websocket error")
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                self._logger.warning("MEXC public WS closed by server")
                break

    async def _handle_public_binary_message(self, raw_message: bytes) -> None:
        decoded = self._decode_mexc_binary_message(raw_message)
        if decoded is not None:
            await self._handle_public_payload(decoded)

    async def _handle_public_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode public WS text message")
            return

        await self._handle_public_payload(message)

    async def _handle_public_payload(self, message: dict[str, Any]) -> None:
        channel = message.get("channel")

        if channel == "pong":
            self._logger.debug("Received MEXC public pong")
            return

        if channel == "rs.error":
            raise RuntimeError(f"MEXC public WS error: {message}")

        # subscription ack
        if "code" in message and "msg" in message and "channel" not in message:
            self._logger.info(
                "MEXC public subscribe response | code=%s msg=%s",
                message.get("code"),
                message.get("msg"),
            )
            return

        if channel == "push.deal":
            await self._publish_trade_events(message)
            return

        if channel in {"push.depth", "push.depth.step"}:
            await self._publish_orderbook_event(message)
            return

        if channel == "push.kline":
            await self._publish_kline_event(message)
            return

        self._logger.debug("Unhandled MEXC public payload | channel=%s", channel)

    # ------------------------------------------------------------------
    # Private WS
    # ------------------------------------------------------------------

    async def _run_private_loop(self) -> None:
        reconnect_attempt = 0

        while self._running:
            try:
                self._logger.info("Connecting to MEXC private WS | url=%s", self._ws_url)

                assert self._session is not None
                self._private_ws = await self._session.ws_connect(
                    self._ws_url,
                    heartbeat=None,
                    autoping=False,
                )

                await self._login_private_ws()

                if not self._disable_default_private_pushes:
                    await self._send_private_filter()

                reconnect_attempt = 0

                self._logger.info("Connected to MEXC private WS")
                await self._event_bus.emit(
                    "system.ws.connected",
                    {
                        "exchange": "mexc",
                        "channel": "private",
                    },
                    priority=EventPriority.HIGH,
                    source="mexc_ws",
                )

                self._private_ping_task = asyncio.create_task(
                    self._ping_loop(self._private_ws, "private"),
                    name="mexc-private-ws-ping",
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
                        "exchange": "mexc",
                        "channel": "private",
                        "error": str(exc),
                        "attempt": reconnect_attempt,
                    },
                    priority=EventPriority.HIGH,
                    source="mexc_ws",
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
                            "exchange": "mexc",
                            "channel": "private",
                        },
                        priority=EventPriority.HIGH,
                        source="mexc_ws",
                    )

    async def _login_private_ws(self) -> None:
        if self._private_ws is None:
            raise RuntimeError("Private websocket is not connected")

        req_time = str(self._current_timestamp_ms())
        signature = self._sign_private_login(req_time)

        payload = {
            "method": "login",
            "param": {
                "apiKey": self._api_key,
                "reqTime": req_time,
                "signature": signature,
            },
            "subscribe": not self._disable_default_private_pushes,
        }

        await self._send_ws_json(self._private_ws, payload)
        self._logger.info("MEXC private WS login request sent")

    async def _send_private_filter(self) -> None:
        if self._private_ws is None:
            raise RuntimeError("Private websocket is not connected")

        payload = {
            "method": "personal.filter",
            "param": {
                "filters": [
                    {"filter": "order"},
                    {"filter": "order.deal"},
                    {"filter": "position"},
                    {"filter": "asset"},
                ]
            },
        }

        await self._send_ws_json(self._private_ws, payload)
        self._logger.info("MEXC private personal.filter sent")

    async def _consume_private_messages(self) -> None:
        assert self._private_ws is not None

        async for msg in self._private_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_private_message(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await self._handle_private_binary_message(msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("MEXC private websocket error")
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                self._logger.warning("MEXC private WS closed by server")
                break

    async def _handle_private_binary_message(self, raw_message: bytes) -> None:
        decoded = self._decode_mexc_binary_message(raw_message)
        if decoded is not None:
            await self._handle_private_payload(decoded)

    async def _handle_private_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode private WS text message")
            return

        await self._handle_private_payload(message)

    async def _handle_private_payload(self, message: dict[str, Any]) -> None:
        channel = message.get("channel")

        if channel == "pong":
            self._logger.debug("Received MEXC private pong")
            return

        if channel == "rs.error":
            raise RuntimeError(f"MEXC private WS error: {message}")

        if channel == "rs.login":
            self._logger.info("MEXC private auth succeeded")
            return

        if channel == "push.personal.order":
            await self._publish_private_order_event(message)
            return

        if channel == "push.personal.order.deal":
            await self._publish_private_order_deal_event(message)
            return

        if channel == "push.personal.position":
            await self._publish_private_position_event(message)
            return

        if channel == "push.personal.asset":
            await self._publish_private_asset_event(message)
            return

        self._logger.debug("Unhandled MEXC private payload | channel=%s", channel)

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
                await asyncio.sleep(self._ping_interval)
                await self._send_ws_json(ws, {"method": "ping"})
                self._logger.debug("Sent MEXC ping | channel=%s", channel)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception("MEXC ping loop failed | channel=%s", channel)
            raise

    # ------------------------------------------------------------------
    # Event publishers - public
    # ------------------------------------------------------------------

    async def _publish_trade_events(self, message: dict[str, Any]) -> None:
        symbol = message.get("symbol")
        trades = message.get("data", [])
        ts = message.get("ts")

        for trade in trades:
            side_code = trade.get("T")
            payload = {
                "exchange": "mexc",
                "symbol": symbol,
                "trade_id": trade.get("i"),
                "price": self._safe_float(trade.get("p")),
                "qty": self._safe_float(trade.get("v")),
                "side": "buy" if side_code == 1 else "sell" if side_code == 2 else None,
                "open_close_flag": trade.get("O"),
                "self_trade_flag": trade.get("M"),
                "trade_time": trade.get("t"),
                "event_time": ts,
            }

            await self._event_bus.emit(
                "market.trade",
                payload,
                priority=EventPriority.NORMAL,
                source="mexc_ws",
            )

    async def _publish_orderbook_event(self, message: dict[str, Any]) -> None:
        symbol = message.get("symbol")
        data = message.get("data", {})

        payload = {
            "exchange": "mexc",
            "symbol": symbol,
            "channel": message.get("channel"),
            "version": data.get("version"),
            "asks": [self._normalize_depth_level(level) for level in data.get("asks", [])],
            "bids": [self._normalize_depth_level(level) for level in data.get("bids", [])],
            "event_time": message.get("ts"),
        }

        await self._event_bus.emit(
            "market.orderbook",
            payload,
            priority=EventPriority.LOW,
            source="mexc_ws",
        )

    async def _publish_kline_event(self, message: dict[str, Any]) -> None:
        data = message.get("data", {})
        symbol = message.get("symbol") or data.get("symbol")

        payload = {
            "exchange": "mexc",
            "symbol": symbol,
            "interval": data.get("interval"),
            "open_time": self._normalize_kline_timestamp(data.get("t")),
            "open": self._safe_float(data.get("o")),
            "high": self._safe_float(data.get("h")),
            "low": self._safe_float(data.get("l")),
            "close": self._safe_float(data.get("c")),
            "amount": self._safe_float(data.get("a")),
            "volume": self._safe_float(data.get("q") or data.get("v")),
            "real_open": self._safe_float(data.get("ro")),
        }

        await self._event_bus.emit(
            "market.candle",
            payload,
            priority=EventPriority.NORMAL,
            source="mexc_ws",
        )

    # ------------------------------------------------------------------
    # Event publishers - private
    # ------------------------------------------------------------------

    async def _publish_private_order_event(self, message: dict[str, Any]) -> None:
        data = message.get("data", {})

        payload = {
            "exchange": "mexc",
            "symbol": data.get("symbol"),
            "order_id": data.get("orderId"),
            "position_id": data.get("positionId"),
            "price": self._safe_float(data.get("price")),
            "vol": self._safe_float(data.get("vol")),
            "leverage": self._safe_float(data.get("leverage")),
            "side": data.get("side"),
            "category": data.get("category"),
            "order_type": data.get("orderType"),
            "deal_avg_price": self._safe_float(data.get("dealAvgPrice")),
            "deal_vol": self._safe_float(data.get("dealVol")),
            "order_margin": self._safe_float(data.get("orderMargin")),
            "used_margin": self._safe_float(data.get("usedMargin")),
            "create_time": data.get("createTime"),
            "update_time": data.get("updateTime"),
            "state": data.get("state"),
        }

        await self._event_bus.emit(
            "execution.order_updated",
            payload,
            priority=EventPriority.HIGH,
            source="mexc_ws",
        )

    async def _publish_private_order_deal_event(self, message: dict[str, Any]) -> None:
        data = message.get("data", {})

        payload = {
            "exchange": "mexc",
            "symbol": data.get("symbol"),
            "order_id": data.get("orderId"),
            "deal_id": data.get("id") or data.get("dealId"),
            "trade_price": self._safe_float(data.get("price")),
            "trade_volume": self._safe_float(data.get("vol")),
            "trade_fee": self._safe_float(data.get("fee")),
            "side": data.get("side"),
            "category": data.get("category"),
            "create_time": data.get("createTime"),
        }

        await self._event_bus.emit(
            "execution.deal_updated",
            payload,
            priority=EventPriority.HIGH,
            source="mexc_ws",
        )

    async def _publish_private_position_event(self, message: dict[str, Any]) -> None:
        data = message.get("data", {})

        payload = {
            "exchange": "mexc",
            "symbol": data.get("symbol"),
            "position_id": data.get("positionId"),
            "hold_vol": self._safe_float(data.get("holdVol")),
            "position_type": data.get("positionType"),
            "open_type": data.get("openType"),
            "frozen_vol": self._safe_float(data.get("frozenVol")),
            "close_vol": self._safe_float(data.get("closeVol")),
            "hold_avg_price": self._safe_float(data.get("holdAvgPrice")),
            "liquidate_price": self._safe_float(data.get("liquidatePrice")),
            "oim": self._safe_float(data.get("oim")),
            "adl_level": data.get("adlLevel"),
            "leverage": self._safe_float(data.get("leverage")),
            "create_time": data.get("createTime"),
            "update_time": data.get("updateTime"),
        }

        await self._event_bus.emit(
            "position.updated",
            payload,
            priority=EventPriority.HIGH,
            source="mexc_ws",
        )

    async def _publish_private_asset_event(self, message: dict[str, Any]) -> None:
        data = message.get("data", {})

        payload = {
            "exchange": "mexc",
            "currency": data.get("currency"),
            "position_margin": self._safe_float(data.get("positionMargin")),
            "available_balance": self._safe_float(data.get("availableBalance")),
            "cash_balance": self._safe_float(data.get("cashBalance")),
            "frozen_balance": self._safe_float(data.get("frozenBalance")),
            "equity": self._safe_float(data.get("equity")),
            "unrealized": self._safe_float(data.get("unrealized")),
            "bonus": self._safe_float(data.get("bonus")),
            "update_time": data.get("updateTime"),
        }

        await self._event_bus.emit(
            "account.asset_updated",
            payload,
            priority=EventPriority.HIGH,
            source="mexc_ws",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send_ws_json(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        payload: dict[str, Any],
    ) -> None:
        await ws.send_json(payload)

    async def _close_ws(self, ws: Optional[aiohttp.ClientWebSocketResponse]) -> None:
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            self._logger.exception("Failed to close websocket cleanly")

    def _sign_private_login(self, req_time: str) -> str:
        """
        Для MEXC futures open-api login signature:
        HMAC_SHA256(secret, apiKey + reqTime + parameterString)

        Для WS login бізнес-параметрів немає, тому parameterString = "".
        """
        payload = f"{self._api_key}{req_time}"
        return hmac.new(
            self._api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _decode_mexc_binary_message(raw_message: bytes) -> Optional[dict[str, Any]]:
        try:
            decompressed = gzip.decompress(raw_message).decode("utf-8")
            return json.loads(decompressed)
        except Exception:
            # Інколи сервер може надіслати не gzip-бінарний кадр
            try:
                return json.loads(raw_message.decode("utf-8"))
            except Exception:
                return None

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        symbol = symbol.upper().replace("-", "_").replace("/", "_")
        if "_" in symbol:
            return symbol

        quote_candidates = ("USDT", "USDC", "USD")
        for quote in quote_candidates:
            if symbol.endswith(quote):
                base = symbol[: -len(quote)]
                return f"{base}_{quote}"

        return symbol

    @staticmethod
    def _normalize_depth_level(level: list[Any]) -> list[Optional[float]]:
        normalized: list[Optional[float]] = []
        for item in level:
            try:
                normalized.append(float(item))
            except (TypeError, ValueError):
                normalized.append(None)
        return normalized

    @staticmethod
    def _normalize_kline_timestamp(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            ts = int(value)
            # У документації приклад для kline t виглядає як секунди epoch
            if ts < 10_000_000_000:
                return ts * 1000
            return ts
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _current_timestamp_ms() -> int:
        return int(time.time() * 1000)