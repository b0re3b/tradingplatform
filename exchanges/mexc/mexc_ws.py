from __future__ import annotations

import asyncio
import gzip
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
class MexcWebSocketClientConfig:
    """
    Local MEXC Futures WS adapter config.

    For our architecture MEXC is primarily a market-data exchange.
    Private stream is optional and should publish only exchange.* events,
    not execution.* domain events.
    """

    ws_url: str = "wss://contract.mexc.com/edge"

    timeout_seconds: float = 10.0
    reconnect_delay_seconds: float = 5.0
    max_reconnect_attempts: int = 20

    ping_interval_seconds: float = 15.0
    recv_window_ms: int = 10_000

    symbols: list[str] = field(default_factory=list)
    streams: list[str] = field(default_factory=lambda: ["deal", "depth", "kline"])
    kline_interval: str = "Min1"

    enable_private_stream: bool = False
    disable_default_private_pushes: bool = False

    @classmethod
    def from_core_config(
        cls,
        *,
        config: Config,
        symbols: list[str],
        streams: list[str] | None = None,
        kline_interval: str = "Min1",
        enable_private_stream: bool = False,
        ping_interval_seconds: float = 15.0,
        recv_window_ms: int = 10_000,
        disable_default_private_pushes: bool = False,
    ) -> "MexcWebSocketClientConfig":
        return cls(
            ws_url=config.exchange.ws_url or cls.ws_url,
            timeout_seconds=config.exchange.timeout_seconds,
            reconnect_delay_seconds=config.exchange.reconnect_delay,
            max_reconnect_attempts=config.exchange.max_reconnect_attempts,
            ping_interval_seconds=ping_interval_seconds,
            recv_window_ms=recv_window_ms,
            symbols=symbols,
            streams=streams or ["deal", "depth", "kline"],
            kline_interval=kline_interval,
            enable_private_stream=enable_private_stream,
            disable_default_private_pushes=disable_default_private_pushes,
        )


class MexcWebSocketClient:
    """
    MEXC Futures WebSocket exchange adapter.

    Responsibilities:
    - connect to MEXC public market streams;
    - normalize raw MEXC payloads into internal market events;
    - optionally connect to private stream and publish exchange.* updates;
    - publish everything through EventBus;
    - never call analytics, strategy, risk, or execution directly;
    - never contain trading decision logic.

    Public market events:
    - market.trade
    - market.orderbook
    - market.candle

    Optional private exchange events:
    - exchange.order.updated
    - exchange.deal.updated
    - exchange.position.updated
    - exchange.account.asset_updated

    System events:
    - system.exchange.ws.started
    - system.exchange.ws.stopped
    - system.exchange.ws.connected
    - system.exchange.ws.disconnected
    - system.exchange.ws.error
    - system.exchange.ws.authenticated
    - system.exchange.ws.subscribed
    """

    EXCHANGE = "mexc"
    SOURCE = "mexc_ws"

    SUPPORTED_STREAMS = {"deal", "depth", "kline"}

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        ws_config: MexcWebSocketClientConfig | None = None,
        symbols: list[str] | None = None,
        streams: list[str] | None = None,
        kline_interval: str = "Min1",
        enable_private_stream: bool = False,
        ping_interval: float = 15.0,
        recv_window_ms: int = 10_000,
        disable_default_private_pushes: bool = False,
    ) -> None:
        resolved_config = ws_config or MexcWebSocketClientConfig.from_core_config(
            config=config,
            symbols=symbols or [],
            streams=streams,
            kline_interval=kline_interval,
            enable_private_stream=enable_private_stream,
            ping_interval_seconds=ping_interval,
            recv_window_ms=recv_window_ms,
            disable_default_private_pushes=disable_default_private_pushes,
        )

        self._config = config
        self._event_bus = event_bus
        self._ws_config = resolved_config

        self._symbols = [
            self._normalize_symbol(symbol)
            for symbol in self._ws_config.symbols
        ]
        self._streams = self._normalize_streams(self._ws_config.streams)

        self._api_key = config.exchange.credentials.api_key
        self._api_secret = config.exchange.credentials.api_secret

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

        self._validate_config()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        WS adapter currently does not subscribe to EventBus topics.

        Kept for project-wide consistency with modules that expose register().
        """
        self._logger.debug("MEXC WS register called | subscriptions=0")

    async def start(self) -> None:
        if self._running:
            self._logger.warning("MEXC WS client already started")
            return

        self._running = True
        self._started = True

        await self._ensure_session()

        self._logger.info(
            "Starting MEXC WS client | symbols=%s streams=%s private_stream=%s",
            self._symbols,
            self._streams,
            self._ws_config.enable_private_stream,
        )

        await self._emit_event(
            "system.exchange.ws.started",
            {
                "exchange": self.EXCHANGE,
                "symbols": self._symbols,
                "streams": self._streams,
                "private_stream": self._ws_config.enable_private_stream,
            },
            priority=EventPriority.NORMAL,
        )

        self._public_task = asyncio.create_task(
            self._run_public_loop(),
            name="mexc-public-ws-loop",
        )

        if self._ws_config.enable_private_stream:
            self._require_private_credentials()

            self._private_task = asyncio.create_task(
                self._run_private_loop(),
                name="mexc-private-ws-loop",
            )

    async def stop(self) -> None:
        if not self._running and not self._started:
            self._logger.warning("MEXC WS client already stopped")
            return

        self._logger.info("Stopping MEXC WS client")

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

        self._public_task = None
        self._private_task = None
        self._public_ping_task = None
        self._private_ping_task = None

        await self._close_ws(self._public_ws, channel="public")
        await self._close_ws(self._private_ws, channel="private")

        self._public_ws = None
        self._private_ws = None

        await self._close_session()

        self._logger.info("MEXC WS client stopped")

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
                    "Connecting to MEXC public WS | url=%s",
                    self._ws_config.ws_url,
                )

                assert self._session is not None
                self._public_ws = await self._session.ws_connect(
                    self._ws_config.ws_url,
                    heartbeat=None,
                    autoping=False,
                )

                reconnect_attempt = 0

                self._logger.info("Connected to MEXC public WS")

                await self._emit_event(
                    "system.exchange.ws.connected",
                    {
                        "exchange": self.EXCHANGE,
                        "channel": "public",
                        "symbols": self._symbols,
                    },
                    priority=EventPriority.HIGH,
                )

                await self._subscribe_public_topics()

                self._public_ping_task = asyncio.create_task(
                    self._ping_loop(self._public_ws, "public"),
                    name="mexc-public-ws-ping",
                )

                await self._consume_public_messages()

            except asyncio.CancelledError:
                self._logger.info("MEXC public WS loop cancelled")
                raise
            except Exception as exc:
                reconnect_attempt += 1

                await self._handle_ws_loop_error(
                    channel="public",
                    exc=exc,
                    reconnect_attempt=reconnect_attempt,
                )

                if self._should_stop_reconnecting(reconnect_attempt):
                    self._logger.error("MEXC public WS max reconnect attempts reached")
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
            raise RuntimeError("MEXC public WebSocket is not connected")

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
                            "interval": self._ws_config.kline_interval,
                        },
                        "gzip": False,
                    },
                )

        self._logger.info(
            "Subscribed to MEXC public topics | symbols=%s streams=%s",
            self._symbols,
            self._streams,
        )

        await self._emit_event(
            "system.exchange.ws.subscribed",
            {
                "exchange": self.EXCHANGE,
                "channel": "public",
                "symbols": self._symbols,
                "streams": self._streams,
            },
            priority=EventPriority.LOW,
        )

    async def _consume_public_messages(self) -> None:
        assert self._public_ws is not None

        async for msg in self._public_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_public_message(msg.data)
                continue

            if msg.type == aiohttp.WSMsgType.BINARY:
                await self._handle_public_binary_message(msg.data)
                continue

            if msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("MEXC public WebSocket error")

            if msg.type in {
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            }:
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
            self._logger.warning("Failed to decode MEXC public WS text message")
            return

        await self._handle_public_payload(message)

    async def _handle_public_payload(self, message: dict[str, Any]) -> None:
        channel = message.get("channel")

        if channel == "pong":
            self._logger.debug("Received MEXC public pong")
            return

        if channel == "rs.error":
            await self._emit_event(
                "system.exchange.ws.error",
                {
                    "exchange": self.EXCHANGE,
                    "channel": "public",
                    "message": message.get("msg"),
                    "code": message.get("code"),
                },
                priority=EventPriority.HIGH,
            )
            raise RuntimeError(f"MEXC public WS error | code={message.get('code')}")

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

        self._logger.debug(
            "Unhandled MEXC public payload | channel=%s",
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
                    "Connecting to MEXC private WS | url=%s",
                    self._ws_config.ws_url,
                )

                assert self._session is not None
                self._private_ws = await self._session.ws_connect(
                    self._ws_config.ws_url,
                    heartbeat=None,
                    autoping=False,
                )

                await self._login_private_ws()

                if not self._ws_config.disable_default_private_pushes:
                    await self._send_private_filter()

                reconnect_attempt = 0

                self._logger.info("Connected to MEXC private WS")

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
                    name="mexc-private-ws-ping",
                )

                await self._consume_private_messages()

            except asyncio.CancelledError:
                self._logger.info("MEXC private WS loop cancelled")
                raise
            except Exception as exc:
                reconnect_attempt += 1

                await self._handle_ws_loop_error(
                    channel="private",
                    exc=exc,
                    reconnect_attempt=reconnect_attempt,
                )

                if self._should_stop_reconnecting(reconnect_attempt):
                    self._logger.error("MEXC private WS max reconnect attempts reached")
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
            raise RuntimeError("MEXC private WebSocket is not connected")

        self._require_private_credentials()

        assert self._api_key is not None

        req_time = str(self._current_timestamp_ms())
        signature = self._sign_private_login(req_time)

        payload = {
            "method": "login",
            "param": {
                "apiKey": self._api_key,
                "reqTime": req_time,
                "signature": signature,
            },
            "subscribe": not self._ws_config.disable_default_private_pushes,
        }

        await self._send_ws_json(self._private_ws, payload)
        self._logger.info("MEXC private WS login request sent")

    async def _send_private_filter(self) -> None:
        if self._private_ws is None:
            raise RuntimeError("MEXC private WebSocket is not connected")

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
                continue

            if msg.type == aiohttp.WSMsgType.BINARY:
                await self._handle_private_binary_message(msg.data)
                continue

            if msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("MEXC private WebSocket error")

            if msg.type in {
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            }:
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
            self._logger.warning("Failed to decode MEXC private WS text message")
            return

        await self._handle_private_payload(message)

    async def _handle_private_payload(self, message: dict[str, Any]) -> None:
        channel = message.get("channel")

        if channel == "pong":
            self._logger.debug("Received MEXC private pong")
            return

        if channel == "rs.error":
            await self._emit_event(
                "system.exchange.ws.error",
                {
                    "exchange": self.EXCHANGE,
                    "channel": "private",
                    "message": message.get("msg"),
                    "code": message.get("code"),
                },
                priority=EventPriority.HIGH,
            )
            raise RuntimeError(f"MEXC private WS error | code={message.get('code')}")

        if channel == "rs.login":
            self._logger.info("MEXC private auth succeeded")

            await self._emit_event(
                "system.exchange.ws.authenticated",
                {
                    "exchange": self.EXCHANGE,
                    "channel": "private",
                },
                priority=EventPriority.HIGH,
            )
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

        self._logger.debug(
            "Unhandled MEXC private payload | channel=%s",
            channel,
        )

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

                await self._send_ws_json(ws, {"method": "ping"})
                self._logger.debug("Sent MEXC ping | channel=%s", channel)

        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception(
                "MEXC ping loop failed | channel=%s",
                channel,
            )
            raise

    # ------------------------------------------------------------------
    # Event publishers: public market data
    # ------------------------------------------------------------------

    async def _publish_trade_events(self, message: dict[str, Any]) -> None:
        symbol = message.get("symbol")
        trades = message.get("data", [])
        event_time = message.get("ts")

        if not isinstance(trades, list):
            return

        for trade in trades:
            side_code = trade.get("T")

            payload = {
                "exchange": self.EXCHANGE,
                "symbol": symbol,
                "trade_id": trade.get("i"),
                "price": self._safe_float(trade.get("p")),
                "qty": self._safe_float(trade.get("v")),
                "side": self._normalize_trade_side(side_code),
                "open_close_flag": trade.get("O"),
                "self_trade_flag": trade.get("M"),
                "trade_time": trade.get("t"),
                "event_time": event_time,
            }

            await self._emit_event(
                "market.trade",
                payload,
                priority=EventPriority.NORMAL,
            )

    async def _publish_orderbook_event(self, message: dict[str, Any]) -> None:
        symbol = message.get("symbol")
        data = message.get("data", {})

        if not isinstance(data, dict):
            return

        payload = {
            "exchange": self.EXCHANGE,
            "symbol": symbol,
            "channel": message.get("channel"),
            "version": data.get("version"),
            "asks": [
                self._normalize_depth_level(level)
                for level in data.get("asks", [])
            ],
            "bids": [
                self._normalize_depth_level(level)
                for level in data.get("bids", [])
            ],
            "event_time": message.get("ts"),
        }

        await self._emit_event(
            "market.orderbook",
            payload,
            priority=EventPriority.LOW,
        )

    async def _publish_kline_event(self, message: dict[str, Any]) -> None:
        data = message.get("data", {})

        if not isinstance(data, dict):
            return

        symbol = message.get("symbol") or data.get("symbol")
        timeframe = data.get("interval") or self._ws_config.kline_interval

        payload = {
            "exchange": self.EXCHANGE,
            "symbol": symbol,
            "timeframe": timeframe,
            "open_time": self._normalize_kline_timestamp(data.get("t")),
            "close_time": None,
            "open": self._safe_float(data.get("o")),
            "high": self._safe_float(data.get("h")),
            "low": self._safe_float(data.get("l")),
            "close": self._safe_float(data.get("c")),
            "volume": self._safe_float(data.get("q") or data.get("v")),
            "quote_volume": self._safe_float(data.get("a")),
            "trades_count": None,
            "is_closed": self._infer_kline_is_closed(data),
            "amount": self._safe_float(data.get("a")),
            "real_open": self._safe_float(data.get("ro")),
            "event_time": message.get("ts"),
        }

        await self._emit_event(
            "market.candle",
            payload,
            priority=EventPriority.NORMAL,
        )

    # ------------------------------------------------------------------
    # Event publishers: optional private exchange updates
    # ------------------------------------------------------------------

    async def _publish_private_order_event(self, message: dict[str, Any]) -> None:
        data = message.get("data", {})

        if not isinstance(data, dict):
            return

        payload = {
            "exchange": self.EXCHANGE,
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

        await self._emit_event(
            "exchange.order.updated",
            payload,
            priority=EventPriority.HIGH,
        )

    async def _publish_private_order_deal_event(self, message: dict[str, Any]) -> None:
        data = message.get("data", {})

        if not isinstance(data, dict):
            return

        payload = {
            "exchange": self.EXCHANGE,
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

        await self._emit_event(
            "exchange.deal.updated",
            payload,
            priority=EventPriority.HIGH,
        )

    async def _publish_private_position_event(self, message: dict[str, Any]) -> None:
        data = message.get("data", {})

        if not isinstance(data, dict):
            return

        payload = {
            "exchange": self.EXCHANGE,
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

        await self._emit_event(
            "exchange.position.updated",
            payload,
            priority=EventPriority.HIGH,
        )

    async def _publish_private_asset_event(self, message: dict[str, Any]) -> None:
        data = message.get("data", {})

        if not isinstance(data, dict):
            return

        payload = {
            "exchange": self.EXCHANGE,
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

        await self._emit_event(
            "exchange.account.asset_updated",
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
                "Failed to emit MEXC WS event | topic=%s",
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
            "MEXC WS loop error | channel=%s attempt=%s max_attempts=%s",
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

    async def _send_ws_json(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        payload: dict[str, Any],
    ) -> None:
        await ws.send_json(payload)

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
                "Failed to close MEXC websocket cleanly | channel=%s",
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
                "MEXC API key and API secret are required for private WebSocket stream"
            )

    def _sign_private_login(self, req_time: str) -> str:
        """
        MEXC Futures WS login signature:
        HMAC_SHA256(secret, apiKey + reqTime)

        No parameterString is used for WS login here.
        """
        self._require_private_credentials()

        assert self._api_key is not None
        assert self._api_secret is not None

        payload = f"{self._api_key}{req_time}"

        return hmac.new(
            self._api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    # ------------------------------------------------------------------
    # Validation / normalization helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        if not self._symbols:
            raise ValueError("At least one MEXC symbol must be configured")

        unsupported = set(self._streams) - self.SUPPORTED_STREAMS
        if unsupported:
            raise ValueError(f"Unsupported MEXC streams: {sorted(unsupported)}")

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
    def _decode_mexc_binary_message(raw_message: bytes) -> dict[str, Any] | None:
        try:
            decompressed = gzip.decompress(raw_message).decode("utf-8")
            return json.loads(decompressed)
        except Exception:
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
    def _normalize_trade_side(side_code: Any) -> str | None:
        if side_code == 1:
            return "buy"

        if side_code == 2:
            return "sell"

        return None

    @staticmethod
    def _normalize_depth_level(level: list[Any]) -> list[float | None]:
        normalized: list[float | None] = []

        for item in level:
            try:
                normalized.append(float(item))
            except (TypeError, ValueError):
                normalized.append(None)

        return normalized

    @staticmethod
    def _normalize_kline_timestamp(value: Any) -> int | None:
        if value is None:
            return None

        try:
            ts = int(value)

            # MEXC kline examples often use epoch seconds.
            if ts < 10_000_000_000:
                return ts * 1000

            return ts
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _infer_kline_is_closed(data: dict[str, Any]) -> bool:
        """
        MEXC push.kline does not always provide a direct closed flag.

        If exchange sends explicit flags in the future, support them here.
        Until then this returns False for live updates.
        CandlesCache can still update the rolling current candle.
        """
        for key in ("is_closed", "closed", "confirm", "x"):
            if key in data:
                value = data.get(key)

                if isinstance(value, bool):
                    return value

                return str(value).lower() in {"1", "true", "yes"}

        return False

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None or value == "":
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _current_timestamp_ms() -> int:
        return int(time.time() * 1000)