from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from typing import Any, Optional

import aiohttp

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger


class BybitWebSocketClient:
    """
    Bybit WebSocket client.

    Public topics:
    - publicTrade.{symbol}
    - orderbook.{depth}.{symbol}
    - kline.{interval}.{symbol}
    - liquidation.{symbol}

    Private topics:
    - order
    - position
    - wallet

    EventBus topics:
    - market.trade
    - market.orderbook
    - market.candle
    - market.liquidation
    - execution.order_updated
    - position.updated
    - account.wallet_updated
    - system.ws.connected
    - system.ws.disconnected
    - system.ws.error
    """

    DEFAULT_PUBLIC_WS_URL = "wss://stream.bybit.com/v5/public/linear"
    DEFAULT_PRIVATE_WS_URL = "wss://stream.bybit.com/v5/private"

    API_KEY_PLACEHOLDER = "BYBIT_API_KEY_PLACEHOLDER"
    API_SECRET_PLACEHOLDER = "BYBIT_API_SECRET_PLACEHOLDER"

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        symbols: list[str],
        streams: Optional[list[str]] = None,
        category: str = "linear",
        orderbook_depth: int = 50,
        kline_interval: str = "1",
        enable_private_stream: bool = False,
        ping_interval: float = 20.0,
        recv_window_ms: int = 5000,
    ) -> None:
        self._config = config
        self._event_bus = event_bus

        self._symbols = [symbol.upper() for symbol in symbols]
        self._streams = streams or ["trade", "orderbook", "kline"]
        self._category = category
        self._orderbook_depth = orderbook_depth
        self._kline_interval = kline_interval
        self._enable_private_stream = enable_private_stream
        self._ping_interval = ping_interval
        self._recv_window_ms = recv_window_ms

        self._logger = get_logger(
            __name__,
            exchange="bybit",
            event_type="bybit_ws",
        )

        self._public_ws_url = self._resolve_public_url(category)
        self._private_ws_url = self.DEFAULT_PRIVATE_WS_URL

        self._api_key = (
            config.exchange.credentials.api_key
            or self.API_KEY_PLACEHOLDER
        )
        self._api_secret = (
            config.exchange.credentials.api_secret
            or self.API_SECRET_PLACEHOLDER
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
            self._logger.warning("Bybit WS client already started")
            return

        self._running = True

        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

        self._logger.info(
            "Starting Bybit WS client | category=%s symbols=%s streams=%s private_stream=%s",
            self._category,
            self._symbols,
            self._streams,
            self._enable_private_stream,
        )

        self._public_task = asyncio.create_task(
            self._run_public_loop(),
            name="bybit-public-ws-loop",
        )

        if self._enable_private_stream:
            self._private_task = asyncio.create_task(
                self._run_private_loop(),
                name="bybit-private-ws-loop",
            )

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("Bybit WS client already stopped")
            return

        self._running = False
        self._logger.info("Stopping Bybit WS client")

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

        self._logger.info("Bybit WS client stopped")

    # ------------------------------------------------------------------
    # Public WS
    # ------------------------------------------------------------------

    async def _run_public_loop(self) -> None:
        reconnect_attempt = 0

        while self._running:
            try:
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
                await self._event_bus.emit(
                    "system.ws.connected",
                    {
                        "exchange": "bybit",
                        "channel": "public",
                        "category": self._category,
                        "symbols": self._symbols,
                    },
                    priority=EventPriority.HIGH,
                    source="bybit_ws",
                )

                await self._subscribe_public_topics()
                self._public_ping_task = asyncio.create_task(
                    self._ping_loop(self._public_ws, "public"),
                    name="bybit-public-ws-ping",
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
                        "exchange": "bybit",
                        "channel": "public",
                        "error": str(exc),
                        "attempt": reconnect_attempt,
                    },
                    priority=EventPriority.HIGH,
                    source="bybit_ws",
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
                            "exchange": "bybit",
                            "channel": "public",
                        },
                        priority=EventPriority.HIGH,
                        source="bybit_ws",
                    )

    async def _subscribe_public_topics(self) -> None:
        if self._public_ws is None:
            raise RuntimeError("Public websocket is not connected")

        topics: list[str] = []

        for symbol in self._symbols:
            if "trade" in self._streams:
                topics.append(f"publicTrade.{symbol}")

            if "orderbook" in self._streams:
                topics.append(f"orderbook.{self._orderbook_depth}.{symbol}")

            if "kline" in self._streams:
                topics.append(f"kline.{self._kline_interval}.{symbol}")

            if "liquidation" in self._streams:
                topics.append(f"liquidation.{symbol}")

        if not topics:
            self._logger.warning("No public topics to subscribe")
            return

        payload = {
            "op": "subscribe",
            "args": topics,
        }

        await self._public_ws.send_json(payload)
        self._logger.info("Subscribed to Bybit public topics | topics=%s", topics)

    async def _consume_public_messages(self) -> None:
        assert self._public_ws is not None

        async for msg in self._public_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_public_message(msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("Bybit public websocket error")
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                self._logger.warning("Bybit public WS closed by server")
                break

    async def _handle_public_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode public WS message")
            return

        if message.get("op") == "pong":
            self._logger.debug("Received Bybit public pong")
            return

        if message.get("op") == "subscribe":
            success = message.get("success")
            self._logger.info("Bybit public subscribe response | success=%s", success)
            return

        topic = message.get("topic")
        data = message.get("data")

        if not topic or data is None:
            self._logger.debug("Received empty or unhandled public payload")
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
            await self._publish_liquidation_event(data)
            return

        self._logger.debug("Unhandled public topic | topic=%s", topic)

    # ------------------------------------------------------------------
    # Private WS
    # ------------------------------------------------------------------

    async def _run_private_loop(self) -> None:
        reconnect_attempt = 0

        while self._running:
            try:
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
                await self._event_bus.emit(
                    "system.ws.connected",
                    {
                        "exchange": "bybit",
                        "channel": "private",
                    },
                    priority=EventPriority.HIGH,
                    source="bybit_ws",
                )

                self._private_ping_task = asyncio.create_task(
                    self._ping_loop(self._private_ws, "private"),
                    name="bybit-private-ws-ping",
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
                        "exchange": "bybit",
                        "channel": "private",
                        "error": str(exc),
                        "attempt": reconnect_attempt,
                    },
                    priority=EventPriority.HIGH,
                    source="bybit_ws",
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
                            "exchange": "bybit",
                            "channel": "private",
                        },
                        priority=EventPriority.HIGH,
                        source="bybit_ws",
                    )

    async def _authenticate_private_ws(self) -> None:
        if self._private_ws is None:
            raise RuntimeError("Private websocket is not connected")

        expires = str(int(time.time() * 1000) + self._recv_window_ms)
        signature = self._sign_ws_auth(expires)

        payload = {
            "op": "auth",
            "args": [self._api_key, expires, signature],
        }

        await self._private_ws.send_json(payload)
        self._logger.info("Bybit private WS auth request sent")

    async def _subscribe_private_topics(self) -> None:
        if self._private_ws is None:
            raise RuntimeError("Private websocket is not connected")

        topics = ["order", "position", "wallet"]

        payload = {
            "op": "subscribe",
            "args": topics,
        }

        await self._private_ws.send_json(payload)
        self._logger.info("Subscribed to Bybit private topics | topics=%s", topics)

    async def _consume_private_messages(self) -> None:
        assert self._private_ws is not None

        async for msg in self._private_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_private_message(msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("Bybit private websocket error")
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                self._logger.warning("Bybit private WS closed by server")
                break

    async def _handle_private_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode private WS message")
            return

        if message.get("op") == "pong":
            self._logger.debug("Received Bybit private pong")
            return

        if message.get("op") == "auth":
            success = message.get("success")
            if not success:
                raise RuntimeError(f"Bybit private auth failed: {message}")
            self._logger.info("Bybit private auth succeeded")
            return

        if message.get("op") == "subscribe":
            success = message.get("success")
            self._logger.info("Bybit private subscribe response | success=%s", success)
            return

        topic = message.get("topic")
        data = message.get("data")

        if not topic or data is None:
            self._logger.debug("Received empty or unhandled private payload")
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

        self._logger.debug("Unhandled private topic | topic=%s", topic)

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
                await ws.send_json({"op": "ping"})
                self._logger.debug("Sent Bybit ping | channel=%s", channel)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception("Bybit ping loop failed | channel=%s", channel)
            raise

    # ------------------------------------------------------------------
    # Event publishers - public
    # ------------------------------------------------------------------

    async def _publish_trade_events(self, trades: list[dict[str, Any]]) -> None:
        for trade in trades:
            side = (trade.get("S") or "").lower()

            payload = {
                "exchange": "bybit",
                "symbol": trade.get("s"),
                "trade_id": trade.get("i"),
                "price": self._safe_float(trade.get("p")),
                "qty": self._safe_float(trade.get("v")),
                "side": side,
                "trade_time": trade.get("T"),
                "is_block_trade": trade.get("BT"),
            }

            await self._event_bus.emit(
                "market.trade",
                payload,
                priority=EventPriority.NORMAL,
                source="bybit_ws",
            )

    async def _publish_orderbook_event(
        self,
        topic: str,
        data: dict[str, Any],
        ts: Optional[int],
    ) -> None:
        payload = {
            "exchange": "bybit",
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
            "event_time": ts,
            "topic": topic,
        }

        await self._event_bus.emit(
            "market.orderbook",
            payload,
            priority=EventPriority.LOW,
            source="bybit_ws",
        )

    async def _publish_kline_events(self, klines: list[dict[str, Any]]) -> None:
        for item in klines:
            payload = {
                "exchange": "bybit",
                "symbol": item.get("symbol"),
                "interval": item.get("interval"),
                "start": item.get("start"),
                "end": item.get("end"),
                "open": self._safe_float(item.get("open")),
                "high": self._safe_float(item.get("high")),
                "low": self._safe_float(item.get("low")),
                "close": self._safe_float(item.get("close")),
                "volume": self._safe_float(item.get("volume")),
                "turnover": self._safe_float(item.get("turnover")),
                "confirm": item.get("confirm"),
                "timestamp": item.get("timestamp"),
            }

            await self._event_bus.emit(
                "market.candle",
                payload,
                priority=EventPriority.NORMAL,
                source="bybit_ws",
            )

    async def _publish_liquidation_event(self, data: dict[str, Any]) -> None:
        payload = {
            "exchange": "bybit",
            "symbol": data.get("symbol"),
            "side": (data.get("side") or "").lower(),
            "price": self._safe_float(data.get("price")),
            "qty": self._safe_float(data.get("size")),
            "updated_time": data.get("updatedTime"),
        }

        await self._event_bus.emit(
            "market.liquidation",
            payload,
            priority=EventPriority.HIGH,
            source="bybit_ws",
        )

    # ------------------------------------------------------------------
    # Event publishers - private
    # ------------------------------------------------------------------

    async def _publish_private_order_events(self, orders: list[dict[str, Any]]) -> None:
        for order in orders:
            payload = {
                "exchange": "bybit",
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
                "created_time": order.get("createdTime"),
                "updated_time": order.get("updatedTime"),
            }

            await self._event_bus.emit(
                "execution.order_updated",
                payload,
                priority=EventPriority.HIGH,
                source="bybit_ws",
            )

    async def _publish_private_position_events(self, positions: list[dict[str, Any]]) -> None:
        for position in positions:
            payload = {
                "exchange": "bybit",
                "symbol": position.get("symbol"),
                "side": position.get("side"),
                "size": self._safe_float(position.get("size")),
                "entry_price": self._safe_float(position.get("avgPrice")),
                "mark_price": self._safe_float(position.get("markPrice")),
                "liq_price": self._safe_float(position.get("liqPrice")),
                "unrealised_pnl": self._safe_float(position.get("unrealisedPnl")),
                "position_value": self._safe_float(position.get("positionValue")),
                "leverage": self._safe_float(position.get("leverage")),
                "updated_time": position.get("updatedTime"),
            }

            await self._event_bus.emit(
                "position.updated",
                payload,
                priority=EventPriority.HIGH,
                source="bybit_ws",
            )

    async def _publish_private_wallet_events(self, wallets: list[dict[str, Any]]) -> None:
        for wallet in wallets:
            payload = {
                "exchange": "bybit",
                "account_type": wallet.get("accountType"),
                "total_equity": self._safe_float(wallet.get("totalEquity")),
                "total_wallet_balance": self._safe_float(wallet.get("totalWalletBalance")),
                "total_margin_balance": self._safe_float(wallet.get("totalMarginBalance")),
                "total_available_balance": self._safe_float(wallet.get("totalAvailableBalance")),
                "coin": wallet.get("coin", []),
            }

            await self._event_bus.emit(
                "account.wallet_updated",
                payload,
                priority=EventPriority.HIGH,
                source="bybit_ws",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_public_url(self, category: str) -> str:
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

    def _sign_ws_auth(self, expires: str) -> str:
        payload = f"GET/realtime{expires}"
        return hmac.new(
            self._api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def _close_ws(self, ws: Optional[aiohttp.ClientWebSocketResponse]) -> None:
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            self._logger.exception("Failed to close websocket cleanly")

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None