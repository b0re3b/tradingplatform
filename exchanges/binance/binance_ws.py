from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler


@dataclass(slots=True)
class BinanceWebSocketClientConfig:
    """
    Binance WS adapter local config.

    Global exchange-level values still come from core.config.Config.exchange.
    This dataclass contains only Binance WS adapter-specific settings.
    """

    public_ws_url: str = "wss://stream.binance.com:9443/stream"
    private_ws_base_url: str = "wss://stream.binance.com:9443/ws"
    rest_url: str = "https://api.binance.com"

    timeout_seconds: float = 10.0
    heartbeat_seconds: float = 20.0
    reconnect_delay_seconds: float = 5.0
    max_reconnect_attempts: int = 20

    listen_key_keepalive_interval_seconds: float = 30 * 60
    listen_key_keepalive_timeout_seconds: float = 10.0

    symbols: list[str] = field(default_factory=list)
    streams: list[str] = field(default_factory=lambda: ["trade", "depth", "kline"])

    depth_level: str = "20"
    depth_speed: str = "100ms"
    kline_interval: str = "1m"

    enable_private_stream: bool = False
    emit_raw_private_account_events: bool = True

    @classmethod
    def from_core_config(
        cls,
        *,
        config: Config,
        symbols: list[str],
        streams: list[str] | None = None,
        depth_level: str = "20",
        kline_interval: str = "1m",
        enable_private_stream: bool = False,
    ) -> "BinanceWebSocketClientConfig":
        return cls(
            public_ws_url=config.exchange.ws_url or cls.public_ws_url,
            rest_url=config.exchange.rest_url or cls.rest_url,
            timeout_seconds=config.exchange.timeout_seconds,
            reconnect_delay_seconds=config.exchange.reconnect_delay,
            max_reconnect_attempts=config.exchange.max_reconnect_attempts,
            symbols=symbols,
            streams=streams or ["trade", "depth", "kline"],
            depth_level=depth_level,
            kline_interval=kline_interval,
            enable_private_stream=enable_private_stream,
        )


class BinanceWebSocketClient:
    """
    Binance WebSocket exchange adapter.

    Responsibilities:
    - connect to Binance public market streams;
    - optionally connect to Binance private user stream;
    - normalize raw Binance messages into internal payloads;
    - publish events through EventBus;
    - use Scheduler for listenKey keepalive;
    - never call analytics, strategy, risk, or execution directly.

    Public events:
    - market.trade
    - market.orderbook
    - market.candle

    Private exchange events:
    - exchange.order.updated
    - exchange.account.position_updated
    - exchange.account.balance_updated

    System events:
    - system.exchange.ws.started
    - system.exchange.ws.stopped
    - system.exchange.ws.connected
    - system.exchange.ws.disconnected
    - system.exchange.ws.error
    - system.exchange.listen_key.created
    - system.exchange.listen_key.refreshed
    """

    EXCHANGE = "binance"
    SOURCE = "binance_ws"

    SUPPORTED_STREAMS = {"trade", "depth", "kline"}

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        ws_config: BinanceWebSocketClientConfig | None = None,
        symbols: list[str] | None = None,
        streams: list[str] | None = None,
        depth_level: str = "20",
        kline_interval: str = "1m",
        enable_private_stream: bool = False,
    ) -> None:
        resolved_config = ws_config or BinanceWebSocketClientConfig.from_core_config(
            config=config,
            symbols=symbols or [],
            streams=streams,
            depth_level=depth_level,
            kline_interval=kline_interval,
            enable_private_stream=enable_private_stream,
        )

        self._config = config
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._ws_config = resolved_config

        self._symbols = [symbol.lower() for symbol in self._ws_config.symbols]
        self._streams = self._normalize_streams(self._ws_config.streams)

        self._api_key = config.exchange.credentials.api_key

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

        self._listen_key: str | None = None
        self._listen_key_job_id: str | None = None

        self._running = False
        self._started = False

        self._validate_config()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        WS adapter currently does not subscribe to EventBus topics.

        Kept for project-wide consistency: every module that participates
        in EventBus wiring exposes register().
        """
        self._logger.debug("Binance WS register called | subscriptions=0")

    async def start(self) -> None:
        if self._running:
            self._logger.warning("Binance WS client already started")
            return

        self._running = True
        self._started = True

        await self._ensure_session()

        self._logger.info(
            "Starting Binance WS client | symbols=%s streams=%s private_stream=%s",
            self._symbols,
            self._streams,
            self._ws_config.enable_private_stream,
        )

        await self._emit_event(
            "system.exchange.ws.started",
            {
                "exchange": self.EXCHANGE,
                "symbols": [symbol.upper() for symbol in self._symbols],
                "streams": self._streams,
                "private_stream": self._ws_config.enable_private_stream,
            },
            priority=EventPriority.NORMAL,
        )

        self._public_task = asyncio.create_task(
            self._run_public_loop(),
            name="binance-public-ws-loop",
        )

        if self._ws_config.enable_private_stream:
            self._require_private_stream_dependencies()

            self._private_task = asyncio.create_task(
                self._run_private_loop(),
                name="binance-private-ws-loop",
            )

    async def stop(self) -> None:
        if not self._running and not self._started:
            self._logger.warning("Binance WS client already stopped")
            return

        self._logger.info("Stopping Binance WS client")

        self._running = False
        self._started = False

        await self._remove_listen_key_keepalive_job()

        tasks = [
            task
            for task in (self._public_task, self._private_task)
            if task is not None
        ]

        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._public_task = None
        self._private_task = None

        await self._close_ws(self._public_ws, channel="public")
        await self._close_ws(self._private_ws, channel="private")

        self._public_ws = None
        self._private_ws = None

        await self._close_session()

        self._logger.info("Binance WS client stopped")

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
                stream_url = self._build_public_stream_url()

                self._logger.info(
                    "Connecting to Binance public WS | symbols=%s streams=%s",
                    self._symbols,
                    self._streams,
                )

                assert self._session is not None
                self._public_ws = await self._session.ws_connect(
                    stream_url,
                    heartbeat=self._ws_config.heartbeat_seconds,
                    autoping=True,
                )

                reconnect_attempt = 0

                self._logger.info("Connected to Binance public WS")

                await self._emit_event(
                    "system.exchange.ws.connected",
                    {
                        "exchange": self.EXCHANGE,
                        "channel": "public",
                        "symbols": [symbol.upper() for symbol in self._symbols],
                        "streams": self._streams,
                    },
                    priority=EventPriority.HIGH,
                )

                await self._consume_public_messages()

            except asyncio.CancelledError:
                self._logger.info("Binance public WS loop cancelled")
                raise
            except Exception as exc:
                reconnect_attempt += 1

                await self._handle_ws_loop_error(
                    channel="public",
                    exc=exc,
                    reconnect_attempt=reconnect_attempt,
                )

                if self._should_stop_reconnecting(reconnect_attempt):
                    self._logger.error("Public WS max reconnect attempts reached")
                    break

                await asyncio.sleep(self._ws_config.reconnect_delay_seconds)
            finally:
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

    async def _consume_public_messages(self) -> None:
        assert self._public_ws is not None

        async for msg in self._public_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_public_message(msg.data)
                continue

            if msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("Binance public WebSocket error")

            if msg.type in {
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            }:
                self._logger.warning("Binance public WS closed by server")
                break

    async def _handle_public_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode Binance public WS message")
            return

        stream_name = message.get("stream")
        data = message.get("data")

        if not stream_name or not isinstance(data, dict):
            self._logger.debug("Received malformed Binance public WS payload")
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

        self._logger.debug(
            "Unhandled Binance public stream message | stream=%s",
            stream_name,
        )

    # ------------------------------------------------------------------
    # Private WS loop
    # ------------------------------------------------------------------

    async def _run_private_loop(self) -> None:
        reconnect_attempt = 0

        while self._running:
            try:
                await self._ensure_session()
                await self._ensure_listen_key()

                if not self._listen_key:
                    raise RuntimeError("Binance listenKey is not available")

                private_ws_url = (
                    f"{self._ws_config.private_ws_base_url}/{self._listen_key}"
                )

                self._logger.info("Connecting to Binance private WS")

                assert self._session is not None
                self._private_ws = await self._session.ws_connect(
                    private_ws_url,
                    heartbeat=self._ws_config.heartbeat_seconds,
                    autoping=True,
                )

                reconnect_attempt = 0

                self._logger.info("Connected to Binance private WS")

                await self._emit_event(
                    "system.exchange.ws.connected",
                    {
                        "exchange": self.EXCHANGE,
                        "channel": "private",
                    },
                    priority=EventPriority.HIGH,
                )

                self._ensure_listen_key_keepalive_job()

                await self._consume_private_messages()

            except asyncio.CancelledError:
                self._logger.info("Binance private WS loop cancelled")
                raise
            except Exception as exc:
                reconnect_attempt += 1

                await self._handle_ws_loop_error(
                    channel="private",
                    exc=exc,
                    reconnect_attempt=reconnect_attempt,
                )

                if self._should_stop_reconnecting(reconnect_attempt):
                    self._logger.error("Private WS max reconnect attempts reached")
                    break

                await asyncio.sleep(self._ws_config.reconnect_delay_seconds)
            finally:
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

    async def _consume_private_messages(self) -> None:
        assert self._private_ws is not None

        async for msg in self._private_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_private_message(msg.data)
                continue

            if msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("Binance private WebSocket error")

            if msg.type in {
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            }:
                self._logger.warning("Binance private WS closed by server")
                break

    async def _handle_private_message(self, raw_message: str) -> None:
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode Binance private WS message")
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

        self._logger.debug(
            "Unhandled Binance private event | event_type=%s",
            event_type,
        )

    # ------------------------------------------------------------------
    # ListenKey REST helpers
    # ------------------------------------------------------------------

    async def _ensure_listen_key(self) -> None:
        if self._listen_key:
            return

        self._listen_key = await self._create_listen_key()

    async def _create_listen_key(self) -> str:
        self._require_api_key()

        assert self._api_key is not None
        assert self._session is not None

        headers = {
            "X-MBX-APIKEY": self._api_key,
        }

        url = f"{self._ws_config.rest_url}/api/v3/userDataStream"

        self._logger.info("Creating Binance listen key")

        async with self._session.post(url, headers=headers) as response:
            response_text = await response.text()

            if response.status >= 400:
                await self._emit_event(
                    "system.exchange.listen_key.error",
                    {
                        "exchange": self.EXCHANGE,
                        "status": response.status,
                        "operation": "create",
                    },
                    priority=EventPriority.HIGH,
                )

                raise RuntimeError(
                    f"Failed to create Binance listenKey | status={response.status}"
                )

            payload = json.loads(response_text)
            listen_key = payload.get("listenKey")

            if not listen_key:
                raise RuntimeError("listenKey missing in Binance response")

            self._logger.info("Binance listen key created")

            await self._emit_event(
                "system.exchange.listen_key.created",
                {
                    "exchange": self.EXCHANGE,
                },
                priority=EventPriority.NORMAL,
            )

            return listen_key

    async def _keepalive_listen_key(self) -> None:
        if not self._listen_key:
            self._logger.warning("ListenKey keepalive skipped: listenKey is empty")
            return

        self._require_api_key()
        await self._ensure_session()

        assert self._api_key is not None
        assert self._session is not None

        headers = {
            "X-MBX-APIKEY": self._api_key,
        }

        params = {
            "listenKey": self._listen_key,
        }

        url = f"{self._ws_config.rest_url}/api/v3/userDataStream"

        self._logger.info("Refreshing Binance listen key")

        async with self._session.put(url, headers=headers, params=params) as response:
            if response.status >= 400:
                await self._emit_event(
                    "system.exchange.listen_key.error",
                    {
                        "exchange": self.EXCHANGE,
                        "status": response.status,
                        "operation": "keepalive",
                    },
                    priority=EventPriority.HIGH,
                )

                raise RuntimeError(
                    f"Failed to refresh Binance listenKey | status={response.status}"
                )

            self._logger.info("Binance listen key refreshed")

            await self._emit_event(
                "system.exchange.listen_key.refreshed",
                {
                    "exchange": self.EXCHANGE,
                },
                priority=EventPriority.LOW,
            )

    def _ensure_listen_key_keepalive_job(self) -> None:
        if self._scheduler is None:
            raise RuntimeError(
                "Scheduler is required when Binance private stream is enabled"
            )

        if self._listen_key_job_id is not None:
            return

        self._listen_key_job_id = self._scheduler.add_interval_job(
            name="binance_listen_key_keepalive",
            func=self._keepalive_listen_key,
            interval=self._ws_config.listen_key_keepalive_interval_seconds,
            run_immediately=False,
            max_retries=2,
            retry_delay=2.0,
            timeout=self._ws_config.listen_key_keepalive_timeout_seconds,
            allow_overlap=False,
        )

        self._logger.info(
            "Binance listenKey keepalive job registered | job_id=%s",
            self._listen_key_job_id,
        )

    async def _remove_listen_key_keepalive_job(self) -> None:
        if self._listen_key_job_id is None or self._scheduler is None:
            self._listen_key_job_id = None
            return

        try:
            self._scheduler.remove_job(self._listen_key_job_id)
        except Exception:
            self._logger.exception("Failed to remove Binance listenKey keepalive job")
        finally:
            self._listen_key_job_id = None

    # ------------------------------------------------------------------
    # Event publishers: public market data
    # ------------------------------------------------------------------

    async def _publish_trade_event(self, data: dict[str, Any]) -> None:
        payload = {
            "exchange": self.EXCHANGE,
            "symbol": data.get("s"),
            "trade_id": data.get("t"),
            "price": self._safe_float(data.get("p")),
            "qty": self._safe_float(data.get("q")),
            "buyer_is_maker": data.get("m"),
            "side": "sell" if data.get("m") else "buy",
            "event_time": data.get("E"),
            "trade_time": data.get("T"),
        }

        await self._emit_event(
            "market.trade",
            payload,
            priority=EventPriority.NORMAL,
        )

    async def _publish_orderbook_event(self, data: dict[str, Any]) -> None:
        payload = {
            "exchange": self.EXCHANGE,
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

        await self._emit_event(
            "market.orderbook",
            payload,
            priority=EventPriority.LOW,
        )

    async def _publish_kline_event(self, data: dict[str, Any]) -> None:
        kline = data.get("k")

        if not isinstance(kline, dict):
            self._logger.debug("Malformed Binance kline payload")
            return

        payload = {
            "exchange": self.EXCHANGE,
            "symbol": data.get("s"),
            "timeframe": kline.get("i"),
            "open_time": kline.get("t"),
            "close_time": kline.get("T"),
            "open": self._safe_float(kline.get("o")),
            "high": self._safe_float(kline.get("h")),
            "low": self._safe_float(kline.get("l")),
            "close": self._safe_float(kline.get("c")),
            "volume": self._safe_float(kline.get("v")),
            "quote_volume": self._safe_float(kline.get("q")),
            "trades_count": kline.get("n"),
            "taker_buy_base_volume": self._safe_float(kline.get("V")),
            "taker_buy_quote_volume": self._safe_float(kline.get("Q")),
            "is_closed": bool(kline.get("x")),
            "event_time": data.get("E"),
        }

        await self._emit_event(
            "market.candle",
            payload,
            priority=EventPriority.NORMAL,
        )

    # ------------------------------------------------------------------
    # Event publishers: private exchange updates
    # ------------------------------------------------------------------

    async def _publish_execution_report_event(self, data: dict[str, Any]) -> None:
        """
        Publish exchange-level order update.

        Execution/OrderManager layer should listen to exchange.order.updated
        and decide how to transform it into execution.order_* domain events.
        """

        payload = {
            "exchange": self.EXCHANGE,
            "symbol": data.get("s"),
            "side": data.get("S"),
            "order_type": data.get("o"),
            "time_in_force": data.get("f"),
            "order_qty": self._safe_float(data.get("q")),
            "order_price": self._safe_float(data.get("p")),
            "stop_price": self._safe_float(data.get("P")),
            "execution_type": data.get("x"),
            "order_status": data.get("X"),
            "order_reject_reason": data.get("r"),
            "order_id": data.get("i"),
            "last_executed_qty": self._safe_float(data.get("l")),
            "cumulative_filled_qty": self._safe_float(data.get("z")),
            "last_executed_price": self._safe_float(data.get("L")),
            "commission": self._safe_float(data.get("n")),
            "commission_asset": data.get("N"),
            "transaction_time": data.get("T"),
            "event_time": data.get("E"),
            "client_order_id": data.get("c"),
            "trade_id": data.get("t"),
            "is_working": data.get("w"),
            "is_maker": data.get("m"),
        }

        await self._emit_event(
            "exchange.order.updated",
            payload,
            priority=EventPriority.HIGH,
        )

    async def _publish_account_position_event(self, data: dict[str, Any]) -> None:
        balances = data.get("B", [])

        payload = {
            "exchange": self.EXCHANGE,
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

        await self._emit_event(
            "exchange.account.position_updated",
            payload,
            priority=EventPriority.HIGH,
        )

    async def _publish_balance_update_event(self, data: dict[str, Any]) -> None:
        payload = {
            "exchange": self.EXCHANGE,
            "asset": data.get("a"),
            "balance_delta": self._safe_float(data.get("d")),
            "clear_time": data.get("T"),
            "event_time": data.get("E"),
        }

        await self._emit_event(
            "exchange.account.balance_updated",
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
                "Failed to emit Binance WS event | topic=%s",
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
            "Binance WS loop error | channel=%s attempt=%s max_attempts=%s",
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
    # URL/session helpers
    # ------------------------------------------------------------------

    def _build_public_stream_url(self) -> str:
        stream_names: list[str] = []

        for symbol in self._symbols:
            if "trade" in self._streams:
                stream_names.append(f"{symbol}@trade")

            if "depth" in self._streams:
                stream_names.append(
                    f"{symbol}@depth{self._ws_config.depth_level}@{self._ws_config.depth_speed}"
                )

            if "kline" in self._streams:
                stream_names.append(
                    f"{symbol}@kline_{self._ws_config.kline_interval}"
                )

        if not stream_names:
            raise RuntimeError("No Binance public streams configured")

        streams = "/".join(stream_names)
        return f"{self._ws_config.public_ws_url}?streams={streams}"

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
                "Failed to close Binance websocket cleanly | channel=%s",
                channel,
            )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        if not self._symbols:
            raise ValueError("At least one Binance symbol must be configured")

        unsupported = set(self._streams) - self.SUPPORTED_STREAMS
        if unsupported:
            raise ValueError(f"Unsupported Binance streams: {sorted(unsupported)}")

        if self._ws_config.enable_private_stream:
            self._require_private_stream_dependencies()

    def _require_private_stream_dependencies(self) -> None:
        self._require_api_key()

        if self._scheduler is None:
            raise RuntimeError(
                "Scheduler is required when Binance private stream is enabled"
            )

    def _require_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "Binance API key is required for private user data stream"
            )

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
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None