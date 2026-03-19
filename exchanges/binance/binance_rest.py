from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import aiohttp

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler


class BinanceWebSocketClient:
    """
    Binance WebSocket client for public market streams + optional private user stream.

    Public streams:
    - trade
    - depth
    - kline

    Private streams:
    - outboundAccountPosition
    - balanceUpdate
    - executionReport

    EventBus topics:
    - market.trade
    - market.orderbook
    - market.candle
    - execution.order_updated
    - account.position_updated
    - account.balance_updated
    - system.ws.connected
    - system.ws.disconnected
    - system.ws.error
    """

    DEFAULT_PUBLIC_WS_URL = "wss://stream.binance.com:9443/stream"
    DEFAULT_REST_URL = "https://api.binance.com"

    API_KEY_PLACEHOLDER = "BINANCE_API_KEY_PLACEHOLDER"
    API_SECRET_PLACEHOLDER = "BINANCE_API_SECRET_PLACEHOLDER"

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        scheduler: Optional[Scheduler] = None,
        symbols: list[str],
        streams: Optional[list[str]] = None,
        depth_level: str = "20",
        kline_interval: str = "1m",
        enable_private_stream: bool = False,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._scheduler = scheduler

        self._symbols = [symbol.lower() for symbol in symbols]
        self._streams = streams or ["trade", "depth", "kline"]

        self._depth_level = depth_level
        self._kline_interval = kline_interval
        self._enable_private_stream = enable_private_stream

        self._logger = get_logger(
            __name__,
            exchange="binance",
            event_type="binance_ws",
        )

        self._public_ws_url = config.exchange.ws_url or self.DEFAULT_PUBLIC_WS_URL
        self._rest_url = config.exchange.rest_url or self.DEFAULT_REST_URL

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

        self._running = False
        self._listen_key: Optional[str] = None
        self._listen_key_job_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            self._logger.warning("Binance WS client already started")
            return

        self._running = True

        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

        self._logger.info(
            "Starting Binance WS client | symbols=%s streams=%s private_stream=%s",
            self._symbols,
            self._streams,
            self._enable_private_stream,
        )

        self._public_task = asyncio.create_task(
            self._run_public_loop(),
            name="binance-public-ws-loop",
        )

        if self._enable_private_stream:
            self._private_task = asyncio.create_task(
                self._run_private_loop(),
                name="binance-private-ws-loop",
            )

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("Binance WS client already stopped")
            return

        self._running = False
        self._logger.info("Stopping Binance WS client")

        if self._listen_key_job_id and self._scheduler is not None:
            try:
                self._scheduler.remove_job(self._listen_key_job_id)
            except Exception:
                self._logger.exception("Failed to remove listen key keepalive job")
            finally:
                self._listen_key_job_id = None

        tasks = [task for task in (self._public_task, self._private_task) if task is not None]
        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._public_task = None
        self._private_task = None

        await self._close_ws(self._public_ws)
        await self._close_ws(self._private_ws)

        self._public_ws = None
        self._private_ws = None

        if self._session is not None:
            await self._session.close()
            self._session = None

        self._logger.info("Binance WS client stopped")

    # ------------------------------------------------------------------
    # Public WS
    # ------------------------------------------------------------------

    async def _run_public_loop(self) -> None:
        reconnect_attempt = 0

        while self._running:
            try:
                stream_url = self._build_public_stream_url()

                self._logger.info(
                    "Connecting to Binance public WS | url=%s",
                    stream_url,
                )

                assert self._session is not None
                self._public_ws = await self._session.ws_connect(
                    stream_url,
                    heartbeat=20,
                    autoping=True,
                )

                reconnect_attempt = 0

                self._logger.info("Connected to Binance public WS")
                await self._event_bus.emit(
                    "system.ws.connected",
                    {
                        "exchange": "binance",
                        "channel": "public",
                        "symbols": self._symbols,
                    },
                    priority=EventPriority.HIGH,
                    source="binance_ws",
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
                        "exchange": "binance",
                        "channel": "public",
                        "error": str(exc),
                        "attempt": reconnect_attempt,
                    },
                    priority=EventPriority.HIGH,
                    source="binance_ws",
                )

                if (
                    self._max_reconnect_attempts > 0
                    and reconnect_attempt >= self._max_reconnect_attempts
                ):
                    self._logger.error("Public WS max reconnect attempts reached")
                    break

                await asyncio.sleep(self._reconnect_delay)
            finally:
                await self._close_ws(self._public_ws)
                self._public_ws = None

                if self._running:
                    await self._event_bus.emit(
                        "system.ws.disconnected",
                        {
                            "exchange": "binance",
                            "channel": "public",
                        },
                        priority=EventPriority.HIGH,
                        source="binance_ws",
                    )

    async def _consume_public_messages(self) -> None:
        assert self._public_ws is not None

        async for msg in self._public_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_public_message(msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("Binance public websocket error")
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                self._logger.warning("Binance public WS closed by server")
                break

    async def _handle_public_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode public WS message")
            return

        stream_name = message.get("stream")
        data = message.get("data", {})

        if not stream_name or not data:
            self._logger.debug("Received empty or malformed public WS payload")
            return

        if "@trade" in stream_name:
            await self._publish_trade_event(data)
            return

        if "@depth" in stream_name:
            await self._publish_orderbook_event(data)
            return

        if "@kline_" in stream_name:
            await self._publish_kline_event(data)
            return

        self._logger.debug("Unhandled public stream message | stream=%s", stream_name)

    # ------------------------------------------------------------------
    # Private WS
    # ------------------------------------------------------------------

    async def _run_private_loop(self) -> None:
        reconnect_attempt = 0

        while self._running:
            try:
                await self._ensure_listen_key()

                if not self._listen_key:
                    raise RuntimeError("listenKey is not available")

                private_ws_url = f"wss://stream.binance.com:9443/ws/{self._listen_key}"

                self._logger.info("Connecting to Binance private WS")

                assert self._session is not None
                self._private_ws = await self._session.ws_connect(
                    private_ws_url,
                    heartbeat=20,
                    autoping=True,
                )

                reconnect_attempt = 0

                self._logger.info("Connected to Binance private WS")
                await self._event_bus.emit(
                    "system.ws.connected",
                    {
                        "exchange": "binance",
                        "channel": "private",
                    },
                    priority=EventPriority.HIGH,
                    source="binance_ws",
                )

                if self._scheduler is not None and self._listen_key_job_id is None:
                    self._listen_key_job_id = self._scheduler.add_interval_job(
                        name="binance_listen_key_keepalive",
                        func=self._keepalive_listen_key,
                        interval=30 * 60,
                        run_immediately=False,
                        max_retries=2,
                        retry_delay=2.0,
                        timeout=10.0,
                        allow_overlap=False,
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
                        "exchange": "binance",
                        "channel": "private",
                        "error": str(exc),
                        "attempt": reconnect_attempt,
                    },
                    priority=EventPriority.HIGH,
                    source="binance_ws",
                )

                if (
                    self._max_reconnect_attempts > 0
                    and reconnect_attempt >= self._max_reconnect_attempts
                ):
                    self._logger.error("Private WS max reconnect attempts reached")
                    break

                await asyncio.sleep(self._reconnect_delay)
            finally:
                await self._close_ws(self._private_ws)
                self._private_ws = None

                if self._running:
                    await self._event_bus.emit(
                        "system.ws.disconnected",
                        {
                            "exchange": "binance",
                            "channel": "private",
                        },
                        priority=EventPriority.HIGH,
                        source="binance_ws",
                    )

    async def _consume_private_messages(self) -> None:
        assert self._private_ws is not None

        async for msg in self._private_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_private_message(msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("Binance private websocket error")
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                self._logger.warning("Binance private WS closed by server")
                break

    async def _handle_private_message(self, raw_message: str) -> None:
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode private WS message")
            return

        event_type = data.get("e")

        if event_type == "executionReport":
            await self._publish_execution_report_event(data)
            return

        if event_type == "outboundAccountPosition":
            await self._publish_account_position_event(data)
            return

        if event_type == "balanceUpdate":
            await self._publish_balance_update_event(data)
            return

        self._logger.debug("Unhandled private event | event_type=%s", event_type)

    # ------------------------------------------------------------------
    # REST for listenKey
    # ------------------------------------------------------------------

    async def _ensure_listen_key(self) -> None:
        if self._listen_key:
            return
        self._listen_key = await self._create_listen_key()

    async def _create_listen_key(self) -> str:
        """
        Для Binance user data stream достатньо API key.
        Secret тут напряму не використовується.
        """

        headers = {
            "X-MBX-APIKEY": self._api_key,
        }

        url = f"{self._rest_url}/api/v3/userDataStream"

        self._logger.info("Creating Binance listen key")

        assert self._session is not None
        async with self._session.post(url, headers=headers) as response:
            text = await response.text()

            if response.status >= 400:
                raise RuntimeError(
                    f"Failed to create listen key | status={response.status} body={text}"
                )

            payload = json.loads(text)
            listen_key = payload.get("listenKey")

            if not listen_key:
                raise RuntimeError("listenKey missing in Binance response")

            self._logger.info("Binance listen key created")
            return listen_key

    async def _keepalive_listen_key(self) -> None:
        if not self._listen_key:
            self._logger.warning("Listen key keepalive skipped: listen key is empty")
            return

        headers = {
            "X-MBX-APIKEY": self._api_key,
        }
        params = {
            "listenKey": self._listen_key,
        }
        url = f"{self._rest_url}/api/v3/userDataStream"

        self._logger.info("Refreshing Binance listen key")

        assert self._session is not None
        async with self._session.put(url, headers=headers, params=params) as response:
            text = await response.text()

            if response.status >= 400:
                raise RuntimeError(
                    f"Failed to refresh listen key | status={response.status} body={text}"
                )

            self._logger.info("Binance listen key refreshed")

    # ------------------------------------------------------------------
    # Event publishers
    # ------------------------------------------------------------------

    async def _publish_trade_event(self, data: dict[str, Any]) -> None:
        payload = {
            "exchange": "binance",
            "symbol": data.get("s"),
            "trade_id": data.get("t"),
            "price": self._safe_float(data.get("p")),
            "qty": self._safe_float(data.get("q")),
            "buyer_is_maker": data.get("m"),
            "side": "sell" if data.get("m") else "buy",
            "event_time": data.get("E"),
            "trade_time": data.get("T"),
        }

        await self._event_bus.emit(
            "market.trade",
            payload,
            priority=EventPriority.NORMAL,
            source="binance_ws",
        )

    async def _publish_orderbook_event(self, data: dict[str, Any]) -> None:
        payload = {
            "exchange": "binance",
            "symbol": data.get("s"),
            "first_update_id": data.get("U"),
            "final_update_id": data.get("u"),
            "bids": [
                [self._safe_float(price), self._safe_float(qty)]
                for price, qty in data.get("b", [])
            ],
            "asks": [
                [self._safe_float(price), self._safe_float(qty)]
                for price, qty in data.get("a", [])
            ],
            "event_time": data.get("E"),
        }

        await self._event_bus.emit(
            "market.orderbook",
            payload,
            priority=EventPriority.LOW,
            source="binance_ws",
        )

    async def _publish_kline_event(self, data: dict[str, Any]) -> None:
        kline = data.get("k", {})

        payload = {
            "exchange": "binance",
            "symbol": data.get("s"),
            "interval": kline.get("i"),
            "open_time": kline.get("t"),
            "close_time": kline.get("T"),
            "open": self._safe_float(kline.get("o")),
            "high": self._safe_float(kline.get("h")),
            "low": self._safe_float(kline.get("l")),
            "close": self._safe_float(kline.get("c")),
            "volume": self._safe_float(kline.get("v")),
            "trades": kline.get("n"),
            "is_closed": kline.get("x"),
            "quote_volume": self._safe_float(kline.get("q")),
        }

        await self._event_bus.emit(
            "market.candle",
            payload,
            priority=EventPriority.NORMAL,
            source="binance_ws",
        )

    async def _publish_execution_report_event(self, data: dict[str, Any]) -> None:
        payload = {
            "exchange": "binance",
            "symbol": data.get("s"),
            "side": data.get("S"),
            "order_type": data.get("o"),
            "time_in_force": data.get("f"),
            "order_qty": self._safe_float(data.get("q")),
            "order_price": self._safe_float(data.get("p")),
            "stop_price": self._safe_float(data.get("P")),
            "execution_type": data.get("x"),
            "order_status": data.get("X"),
            "order_id": data.get("i"),
            "last_executed_qty": self._safe_float(data.get("l")),
            "cumulative_filled_qty": self._safe_float(data.get("z")),
            "last_executed_price": self._safe_float(data.get("L")),
            "commission": self._safe_float(data.get("n")),
            "commission_asset": data.get("N"),
            "transaction_time": data.get("T"),
            "event_time": data.get("E"),
            "client_order_id": data.get("c"),
        }

        await self._event_bus.emit(
            "execution.order_updated",
            payload,
            priority=EventPriority.HIGH,
            source="binance_ws",
        )

    async def _publish_account_position_event(self, data: dict[str, Any]) -> None:
        balances = data.get("B", [])

        payload = {
            "exchange": "binance",
            "event_time": data.get("E"),
            "last_account_update_time": data.get("u"),
            "balances": [
                {
                    "asset": balance.get("a"),
                    "free": self._safe_float(balance.get("f")),
                    "locked": self._safe_float(balance.get("l")),
                }
                for balance in balances
            ],
        }

        await self._event_bus.emit(
            "account.position_updated",
            payload,
            priority=EventPriority.HIGH,
            source="binance_ws",
        )

    async def _publish_balance_update_event(self, data: dict[str, Any]) -> None:
        payload = {
            "exchange": "binance",
            "asset": data.get("a"),
            "balance_delta": self._safe_float(data.get("d")),
            "clear_time": data.get("T"),
            "event_time": data.get("E"),
        }

        await self._event_bus.emit(
            "account.balance_updated",
            payload,
            priority=EventPriority.HIGH,
            source="binance_ws",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_public_stream_url(self) -> str:
        stream_names: list[str] = []

        for symbol in self._symbols:
            if "trade" in self._streams:
                stream_names.append(f"{symbol}@trade")

            if "depth" in self._streams:
                stream_names.append(f"{symbol}@depth{self._depth_level}@100ms")

            if "kline" in self._streams:
                stream_names.append(f"{symbol}@kline_{self._kline_interval}")

        streams = "/".join(stream_names)
        return f"{self._public_ws_url}?streams={streams}"

    async def _close_ws(self, ws: Optional[aiohttp.ClientWebSocketResponse]) -> None:
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            self._logger.exception("Failed to close websocket cleanly")

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None