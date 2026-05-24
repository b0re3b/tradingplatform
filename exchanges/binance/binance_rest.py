from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from data.market_ingestion import MarketIngestionService


class BinanceSymbolUnavailableError(Exception):
    """
    Raised when Binance returns a non-retryable symbol-level error code such as
    -4108 (symbol on delivering/settling/closed/pre-trading) or -1121 (invalid
    symbol).  Callers that poll derivative data should catch this and disable
    the symbol without retrying.
    """

    def __init__(self, code: int, message: str, symbol: str | None = None) -> None:
        self.code = code
        self.message = message
        self.symbol = symbol
        super().__init__(f"Binance symbol unavailable | code={code} symbol={symbol} message={message}")


@dataclass(slots=True)
class BinanceFuturesRestClientConfig:
    """
    Binance USD-M Futures REST adapter config.

    This adapter is the execution-capable Binance futures client.
    It uses /fapi/* endpoints, not spot /api/v3/* endpoints.
    """

    rest_url: str = "https://testnet.binancefuture.com"
    timeout_seconds: float = 10.0

    recv_window: int = 10000
    max_recv_window: int = 60000
    time_sync_on_start: bool = True
    resync_time_error_codes: tuple[int, ...] = (-1021,)
    request_retries: int = 2
    retry_delay_seconds: float = 0.25

    emit_success_events: bool = False
    emit_error_events: bool = True

    # In paper/dev mode the execution layer may still ask for read-only private
    # snapshots such as open orders or positions. Without API credentials Binance
    # cannot serve those endpoints. Returning an empty snapshot prevents noisy
    # sync failures while keeping all trading/write endpoints strictly protected.
    allow_private_read_without_credentials: bool = True

    # ---------------------------------------------------------------------------
    # Derivative snapshot polling throttle
    # ---------------------------------------------------------------------------
    # How many symbols to poll concurrently for open-interest / derivative data.
    # Binance rate-limits /fapi/v1/openInterest aggressively; keep this at 1-2.
    derivative_snapshot_poll_concurrency: int = 1
    # How many symbols per batch tick.
    derivative_snapshot_poll_batch_size: int = 2
    # Minimum seconds between derivative snapshot poll ticks.
    derivative_snapshot_poll_interval_seconds: float = 180.0

    # ---------------------------------------------------------------------------
    # Symbol availability management
    # ---------------------------------------------------------------------------
    # Binance error codes that mean the symbol is permanently unavailable for
    # the current operation (delivering, settling, pre-trading, invalid).
    # These are NOT retried — the symbol is disabled immediately.
    # -4108: symbol on delivering/delivered/settling/closed/pre-trading
    # -1121: invalid symbol
    symbol_unavailable_error_codes: tuple[int, ...] = (-4108, -1121)

    # Pre-configured symbol blocklist.  Any symbol in this set is silently
    # skipped for derivative polling without attempting a request.
    derivative_symbol_blocklist: tuple[str, ...] = ()

    @classmethod
    def from_core_config(cls, config: Config) -> "BinanceFuturesRestClientConfig":
        defaults = cls()
        return cls(
            rest_url=config.exchange.rest_url or defaults.rest_url,
            timeout_seconds=config.exchange.timeout_seconds,
            recv_window=defaults.recv_window,
            max_recv_window=defaults.max_recv_window,
            time_sync_on_start=defaults.time_sync_on_start,
            resync_time_error_codes=defaults.resync_time_error_codes,
        )


class BinanceRestClient:
    """
    Binance USD-M Futures REST exchange adapter.

    Responsibilities:
    - perform Binance Futures REST HTTP requests;
    - normalize futures market/account/order/position payloads;
    - publish market.*, exchange.*, system.exchange.* events through EventBus;
    - never call analytics, strategy, risk, or execution directly;
    - never contain trading decision logic.

    Execution layer may call:
    - create_order()
    - cancel_order()
    - cancel_all_open_orders()
    - change_leverage()
    - change_margin_type()
    - change_position_mode()

    This adapter emits exchange.* events.
    OrderManager/TradeExecutor must transform those into execution.* domain events.
    """

    EXCHANGE = "binance"
    SOURCE = "binance_futures_rest"

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        rest_config: BinanceFuturesRestClientConfig | None = None,
        market_ingestion: MarketIngestionService | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._market_ingestion = market_ingestion
        self._rest_config = rest_config or BinanceFuturesRestClientConfig.from_core_config(config)

        self._logger = get_logger(
            __name__,
            exchange=self.EXCHANGE,
            event_type="exchange_futures_rest",
        )

        self._api_key = config.exchange.credentials.api_key
        self._api_secret = config.exchange.credentials.api_secret

        self._session: aiohttp.ClientSession | None = None
        self._time_offset_ms: int = 0
        self._started = False

        # Runtime symbol blocklist — populated dynamically when Binance returns
        # a symbol-unavailable error code (-4108 / -1121).  Pre-seeded from
        # the static config blocklist so callers can also set it via env/config.
        self._derivative_symbol_blocklist: set[str] = {
            s.upper() for s in self._rest_config.derivative_symbol_blocklist
        }

        # Semaphore that limits concurrent derivative snapshot REST requests.
        self._derivative_poll_semaphore = asyncio.Semaphore(
            max(1, self._rest_config.derivative_snapshot_poll_concurrency)
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        REST adapter currently does not subscribe to EventBus topics.

        Kept for project-wide consistency with modules that expose register().
        """
        self._logger.debug("Binance Futures REST register called | subscriptions=0")

    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            self._logger.warning("Binance Futures REST client already started")
            self._started = True
            return

        timeout = aiohttp.ClientTimeout(total=self._rest_config.timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._started = True

        self._logger.info(
            "Binance Futures REST client started | base_url=%s timeout=%s recv_window=%s",
            self._rest_config.rest_url,
            self._rest_config.timeout_seconds,
            self._rest_config.recv_window,
        )

        if self._rest_config.time_sync_on_start:
            try:
                await self.sync_time()
            except Exception:
                self._logger.warning(
                    "Initial Binance Futures server time sync failed; will retry before signed requests",
                    exc_info=True,
                )

        await self._emit_event(
            "system.exchange.rest.started",
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "rest_url": self._rest_config.rest_url,
            },
            priority=EventPriority.LOW,
        )

    async def stop(self) -> None:
        if self._session is None:
            self._logger.warning("Binance Futures REST client already stopped")
            self._started = False
            return

        await self._session.close()
        self._session = None
        self._started = False

        self._logger.info("Binance Futures REST client stopped")

        await self._emit_event(
            "system.exchange.rest.stopped",
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
            },
            priority=EventPriority.LOW,
        )

    # ------------------------------------------------------------------
    # Public market endpoints
    # ------------------------------------------------------------------

    async def ping(self) -> dict[str, Any]:
        return await self._request(
            method="GET",
            path="/fapi/v1/ping",
        )

    async def get_server_time(self) -> dict[str, Any]:
        return await self._request(
            method="GET",
            path="/fapi/v1/time",
        )

    async def sync_time(self) -> int:
        local_before = self._current_timestamp_ms()
        payload = await self.get_server_time()
        local_after = self._current_timestamp_ms()

        server_time = int(payload["serverTime"])
        local_estimated = (local_before + local_after) // 2
        self._time_offset_ms = server_time - local_estimated

        self._logger.info(
            "Binance Futures server time synced | server_time=%s offset_ms=%s",
            server_time,
            self._time_offset_ms,
        )

        await self._emit_event(
            "system.exchange.time_synced",
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "server_time": server_time,
                "offset_ms": self._time_offset_ms,
            },
            priority=EventPriority.LOW,
        )

        return self._time_offset_ms

    async def get_exchange_info(self, symbol: str | None = None) -> dict[str, Any]:
        payload = await self._request(
            method="GET",
            path="/fapi/v1/exchangeInfo",
        )

        if symbol is None:
            return payload

        symbol = symbol.upper()
        symbols = payload.get("symbols", [])

        filtered = [
            item
            for item in symbols
            if isinstance(item, dict) and item.get("symbol") == symbol
        ]

        return {
            **payload,
            "symbols": filtered,
        }

    async def get_orderbook_snapshot(
        self,
        *,
        symbol: str,
        limit: int = 100,
    ) -> dict[str, Any]:
        symbol = symbol.upper()

        payload = await self._request(
            method="GET",
            path="/fapi/v1/depth",
            params={
                "symbol": symbol,
                "limit": limit,
            },
        )

        normalized = {
            "exchange": self.EXCHANGE,
            "market_type": "usdm_futures",
            "symbol": symbol,
            "last_update_id": payload.get("lastUpdateId"),
            "message_output_time": payload.get("E"),
            "transaction_time": payload.get("T"),
            "bids": [
                [self._safe_float(price), self._safe_float(qty)]
                for price, qty in payload.get("bids", [])
            ],
            "asks": [
                [self._safe_float(price), self._safe_float(qty)]
                for price, qty in payload.get("asks", [])
            ],
            "snapshot_time": self._current_timestamp_ms(),
        }

        if self._market_ingestion is not None:
            await self._market_ingestion.ingest_orderbook_snapshot(normalized)
        else:
            await self._emit_event(
                "market.orderbook.snapshot",
                normalized,
                priority=EventPriority.NORMAL,
            )

        return normalized

    async def get_recent_trades(
        self,
        *,
        symbol: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        symbol = symbol.upper()

        payload = await self._request(
            method="GET",
            path="/fapi/v1/trades",
            params={
                "symbol": symbol,
                "limit": limit,
            },
        )

        normalized = [
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "symbol": symbol,
                "trade_id": trade.get("id"),
                "price": self._safe_float(trade.get("price")),
                "qty": self._safe_float(trade.get("qty")),
                "quote_qty": self._mul_safe(
                    self._safe_float(trade.get("price")),
                    self._safe_float(trade.get("qty")),
                ),
                "trade_time": trade.get("time"),
                "buyer_is_maker": trade.get("isBuyerMaker"),
                "side": "sell" if trade.get("isBuyerMaker") else "buy",
            }
            for trade in payload
            if isinstance(trade, dict)
        ]

        trades_snapshot_payload = {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "symbol": symbol,
                "count": len(normalized),
                "trades": normalized,
                "snapshot_time": self._current_timestamp_ms(),
            }
        if self._market_ingestion is not None:
            await self._market_ingestion.ingest_trades_batch(trades_snapshot_payload)
        else:
            await self._emit_event(
                "market.trades.snapshot",
                trades_snapshot_payload,
                priority=EventPriority.LOW,
            )

        return normalized

    async def get_klines(
        self,
        *,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        symbol = symbol.upper()

        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        payload = await self._request(
            method="GET",
            path="/fapi/v1/klines",
            params=params,
        )

        normalized = [
            self._normalize_kline(
                item=item,
                symbol=symbol,
                timeframe=interval,
                is_closed=True,
            )
            for item in payload
            if isinstance(item, list) and len(item) >= 11
        ]

        candles_snapshot_payload = {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "symbol": symbol,
                "timeframe": interval,
                "count": len(normalized),
                "candles": normalized,
                "snapshot_time": self._current_timestamp_ms(),
            }
        if self._market_ingestion is not None:
            await self._market_ingestion.ingest_candles_batch(candles_snapshot_payload)
        else:
            await self._emit_event(
                "market.candles.snapshot",
                candles_snapshot_payload,
                priority=EventPriority.LOW,
            )

        return normalized

    async def get_premium_index(
        self,
        *,
        symbol: str | None = None,
    ) -> Any:
        params: dict[str, Any] = {}

        if symbol is not None:
            params["symbol"] = symbol.upper()

        payload = await self._request(
            method="GET",
            path="/fapi/v1/premiumIndex",
            params=params,
        )

        return payload

    async def get_funding_rate(
        self,
        *,
        symbol: str,
        limit: int = 100,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        symbol = symbol.upper()

        params: dict[str, Any] = {
            "symbol": symbol,
            "limit": limit,
        }

        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        payload = await self._request(
            method="GET",
            path="/fapi/v1/fundingRate",
            params=params,
        )

        normalized = [
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "symbol": item.get("symbol") or symbol,
                "funding_rate": self._safe_float(item.get("fundingRate")),
                "funding_time": item.get("fundingTime"),
            }
            for item in payload
            if isinstance(item, dict)
        ]

        funding_snapshot_payload = {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "symbol": symbol,
                "count": len(normalized),
                "items": normalized,
                "snapshot_time": self._current_timestamp_ms(),
            }
        if self._market_ingestion is not None:
            for item in normalized:
                await self._market_ingestion.ingest_funding(item)
        else:
            await self._emit_event(
                "market.funding.snapshot",
                funding_snapshot_payload,
                priority=EventPriority.NORMAL,
            )

        return normalized

    async def get_open_interest(
        self,
        *,
        symbol: str,
    ) -> dict[str, Any]:
        symbol = symbol.upper()

        if symbol in self._derivative_symbol_blocklist:
            self._logger.debug(
                "Skipping open interest poll — symbol in blocklist | symbol=%s",
                symbol,
            )
            return {"exchange": self.EXCHANGE, "market_type": "usdm_futures", "symbol": symbol, "skipped": True, "skip_reason": "blocklisted"}

        async with self._derivative_poll_semaphore:
            try:
                payload = await self._request(
                    method="GET",
                    path="/fapi/v1/openInterest",
                    params={"symbol": symbol},
                )
            except asyncio.TimeoutError as exc:
                self._logger.warning(
                    "Open interest poll timed out — will retry next tick | symbol=%s timeout_seconds=%s",
                    symbol,
                    self._rest_config.timeout_seconds,
                )
                return {"exchange": self.EXCHANGE, "market_type": "usdm_futures", "symbol": symbol, "skipped": True, "skip_reason": "timeout"}
            except BinanceSymbolUnavailableError as exc:
                self._derivative_symbol_blocklist.add(symbol)
                self._logger.warning(
                    "Symbol disabled for derivative polling — permanently unavailable | symbol=%s code=%s",
                    symbol,
                    exc.code,
                )
                return {"exchange": self.EXCHANGE, "market_type": "usdm_futures", "symbol": symbol, "skipped": True, "skip_reason": "symbol_unavailable", "code": exc.code}

        normalized = {
            "exchange": self.EXCHANGE,
            "market_type": "usdm_futures",
            "symbol": payload.get("symbol") or symbol,
            "open_interest": self._safe_float(payload.get("openInterest")),
            "time": payload.get("time"),
            "snapshot_time": self._current_timestamp_ms(),
        }

        if self._market_ingestion is not None:
            await self._market_ingestion.ingest_open_interest(normalized)
        else:
            await self._emit_event(
                "market.open_interest.snapshot",
                normalized,
                priority=EventPriority.NORMAL,
            )

        return normalized

    async def get_ticker_24h(
        self,
        *,
        symbol: str | None = None,
    ) -> Any:
        params: dict[str, Any] = {}

        if symbol is not None:
            params["symbol"] = symbol.upper()

        payload = await self._request(
            method="GET",
            path="/fapi/v1/ticker/24hr",
            params=params,
        )

        await self._emit_event(
            "market.tickers.snapshot",
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "symbol": symbol.upper() if symbol else None,
                "snapshot_time": self._current_timestamp_ms(),
            },
            priority=EventPriority.LOW,
        )

        return payload

    # ------------------------------------------------------------------
    # Listen key endpoints for Binance Futures user data stream
    # ------------------------------------------------------------------

    async def create_listen_key(self) -> str:
        payload = await self._request(
            method="POST",
            path="/fapi/v1/listenKey",
            auth_required=True,
        )

        listen_key = payload.get("listenKey")
        if not listen_key:
            raise RuntimeError("Binance Futures listenKey missing in response")

        self._logger.info("Binance Futures listenKey created")

        await self._emit_event(
            "system.exchange.listen_key.created",
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
            },
            priority=EventPriority.NORMAL,
        )

        return listen_key

    async def keepalive_listen_key(self, listen_key: str) -> None:
        await self._request(
            method="PUT",
            path="/fapi/v1/listenKey",
            params={"listenKey": listen_key},
            auth_required=True,
        )

        self._logger.info("Binance Futures listenKey refreshed")

        await self._emit_event(
            "system.exchange.listen_key.refreshed",
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
            },
            priority=EventPriority.LOW,
        )

    async def close_listen_key(self, listen_key: str) -> None:
        await self._request(
            method="DELETE",
            path="/fapi/v1/listenKey",
            params={"listenKey": listen_key},
            auth_required=True,
        )

        self._logger.info("Binance Futures listenKey closed")

        await self._emit_event(
            "system.exchange.listen_key.closed",
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
            },
            priority=EventPriority.LOW,
        )

    # ------------------------------------------------------------------
    # Private account / position endpoints
    # ------------------------------------------------------------------

    async def get_balance(self, *, recv_window: int | None = None) -> list[dict[str, Any]]:
        if self._should_skip_private_read_without_credentials("get_balance"):
            await self._emit_event(
                "exchange.account.balance.snapshot",
                {
                    "exchange": self.EXCHANGE,
                    "market_type": "usdm_futures",
                    "balances": [],
                    "count": 0,
                    "snapshot_time": self._current_timestamp_ms(),
                    "skipped": True,
                    "skip_reason": "missing_credentials",
                },
                priority=EventPriority.LOW,
            )
            return []

        payload = await self._request(
            method="GET",
            path="/fapi/v2/balance",
            params={"recvWindow": recv_window or self._rest_config.recv_window},
            signed=True,
            auth_required=True,
        )

        normalized = [
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "account_alias": item.get("accountAlias"),
                "asset": item.get("asset"),
                "balance": self._safe_float(item.get("balance")),
                "cross_wallet_balance": self._safe_float(item.get("crossWalletBalance")),
                "cross_unrealized_pnl": self._safe_float(item.get("crossUnPnl")),
                "available_balance": self._safe_float(item.get("availableBalance")),
                "max_withdraw_amount": self._safe_float(item.get("maxWithdrawAmount")),
                "margin_available": item.get("marginAvailable"),
                "update_time": item.get("updateTime"),
            }
            for item in payload
            if isinstance(item, dict)
        ]

        await self._emit_event(
            "exchange.account.balance.snapshot",
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "balances": normalized,
                "count": len(normalized),
                "snapshot_time": self._current_timestamp_ms(),
            },
            priority=EventPriority.HIGH,
        )

        return normalized

    async def get_account_info(self, *, recv_window: int | None = None) -> dict[str, Any]:
        if self._should_skip_private_read_without_credentials("get_account_info"):
            normalized = {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "fee_tier": None,
                "can_trade": False,
                "can_deposit": None,
                "can_withdraw": None,
                "update_time": None,
                "total_initial_margin": None,
                "total_maint_margin": None,
                "total_wallet_balance": None,
                "total_unrealized_profit": None,
                "total_margin_balance": None,
                "total_position_initial_margin": None,
                "total_open_order_initial_margin": None,
                "total_cross_wallet_balance": None,
                "total_cross_unrealized_pnl": None,
                "available_balance": None,
                "max_withdraw_amount": None,
                "assets": [],
                "positions": [],
                "snapshot_time": self._current_timestamp_ms(),
                "skipped": True,
                "skip_reason": "missing_credentials",
            }

            await self._emit_event(
                "exchange.account.updated",
                normalized,
                priority=EventPriority.LOW,
            )

            return normalized

        payload = await self._request(
            method="GET",
            path="/fapi/v2/account",
            params={"recvWindow": recv_window or self._rest_config.recv_window},
            signed=True,
            auth_required=True,
        )

        normalized = {
            "exchange": self.EXCHANGE,
            "market_type": "usdm_futures",
            "fee_tier": payload.get("feeTier"),
            "can_trade": payload.get("canTrade"),
            "can_deposit": payload.get("canDeposit"),
            "can_withdraw": payload.get("canWithdraw"),
            "update_time": payload.get("updateTime"),
            "total_initial_margin": self._safe_float(payload.get("totalInitialMargin")),
            "total_maint_margin": self._safe_float(payload.get("totalMaintMargin")),
            "total_wallet_balance": self._safe_float(payload.get("totalWalletBalance")),
            "total_unrealized_profit": self._safe_float(payload.get("totalUnrealizedProfit")),
            "total_margin_balance": self._safe_float(payload.get("totalMarginBalance")),
            "total_position_initial_margin": self._safe_float(payload.get("totalPositionInitialMargin")),
            "total_open_order_initial_margin": self._safe_float(payload.get("totalOpenOrderInitialMargin")),
            "total_cross_wallet_balance": self._safe_float(payload.get("totalCrossWalletBalance")),
            "total_cross_unrealized_pnl": self._safe_float(payload.get("totalCrossUnPnl")),
            "available_balance": self._safe_float(payload.get("availableBalance")),
            "max_withdraw_amount": self._safe_float(payload.get("maxWithdrawAmount")),
            "assets": payload.get("assets", []),
            "positions": payload.get("positions", []),
            "snapshot_time": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.account.updated",
            normalized,
            priority=EventPriority.HIGH,
        )

        return normalized

    async def get_positions(
        self,
        *,
        symbol: str | None = None,
        recv_window: int | None = None,
    ) -> list[dict[str, Any]]:
        if self._should_skip_private_read_without_credentials("get_positions"):
            await self._emit_event(
                "exchange.positions.snapshot",
                {
                    "exchange": self.EXCHANGE,
                    "market_type": "usdm_futures",
                    "symbol": symbol.upper() if symbol else None,
                    "positions": [],
                    "count": 0,
                    "snapshot_time": self._current_timestamp_ms(),
                    "skipped": True,
                    "skip_reason": "missing_credentials",
                },
                priority=EventPriority.LOW,
            )
            return []

        params: dict[str, Any] = {
            "recvWindow": recv_window or self._rest_config.recv_window,
        }

        if symbol is not None:
            params["symbol"] = symbol.upper()

        payload = await self._request(
            method="GET",
            path="/fapi/v2/positionRisk",
            params=params,
            signed=True,
            auth_required=True,
        )

        normalized = [
            self._normalize_position(item)
            for item in payload
            if isinstance(item, dict)
        ]

        await self._emit_event(
            "exchange.positions.snapshot",
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "symbol": symbol.upper() if symbol else None,
                "positions": normalized,
                "count": len(normalized),
                "snapshot_time": self._current_timestamp_ms(),
            },
            priority=EventPriority.HIGH,
        )

        return normalized

    # ------------------------------------------------------------------
    # Private order / trade endpoints
    # ------------------------------------------------------------------

    async def get_open_orders(
        self,
        *,
        symbol: str | None = None,
        recv_window: int | None = None,
    ) -> list[dict[str, Any]]:
        if self._should_skip_private_read_without_credentials("get_open_orders"):
            await self._emit_event(
                "exchange.open_orders.snapshot",
                {
                    "exchange": self.EXCHANGE,
                    "market_type": "usdm_futures",
                    "symbol": symbol.upper() if symbol else None,
                    "orders": [],
                    "count": 0,
                    "snapshot_time": self._current_timestamp_ms(),
                    "skipped": True,
                    "skip_reason": "missing_credentials",
                },
                priority=EventPriority.LOW,
            )
            return []

        params: dict[str, Any] = {
            "recvWindow": recv_window or self._rest_config.recv_window,
        }

        if symbol is not None:
            params["symbol"] = symbol.upper()

        try:
            payload = await self._request(
                method="GET",
                path="/fapi/v1/openOrders",
                params=params,
                signed=True,
                auth_required=True,
            )
        except asyncio.TimeoutError:
            self._logger.warning(
                "get_open_orders timed out — returning empty snapshot | symbol=%s timeout_seconds=%s",
                symbol,
                self._rest_config.timeout_seconds,
            )
            return []

        normalized = [
            self._normalize_order(order)
            for order in payload
            if isinstance(order, dict)
        ]

        await self._emit_event(
            "exchange.open_orders.snapshot",
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "symbol": symbol.upper() if symbol else None,
                "orders": normalized,
                "count": len(normalized),
                "snapshot_time": self._current_timestamp_ms(),
            },
            priority=EventPriority.HIGH,
        )

        return normalized

    async def get_order(
        self,
        *,
        symbol: str,
        order_id: int | None = None,
        orig_client_order_id: str | None = None,
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        if order_id is None and orig_client_order_id is None:
            raise ValueError("Either order_id or orig_client_order_id must be provided")

        symbol = symbol.upper()

        params: dict[str, Any] = {
            "symbol": symbol,
            "recvWindow": recv_window or self._rest_config.recv_window,
        }

        if order_id is not None:
            params["orderId"] = order_id
        if orig_client_order_id is not None:
            params["origClientOrderId"] = orig_client_order_id

        payload = await self._request(
            method="GET",
            path="/fapi/v1/order",
            params=params,
            signed=True,
            auth_required=True,
        )

        normalized = self._normalize_order(payload)

        await self._emit_event(
            "exchange.order.fetched",
            normalized,
            priority=EventPriority.HIGH,
        )

        return normalized

    async def get_user_trades(
        self,
        *,
        symbol: str,
        limit: int = 500,
        order_id: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        recv_window: int | None = None,
    ) -> list[dict[str, Any]]:
        symbol = symbol.upper()

        if self._should_skip_private_read_without_credentials("get_user_trades"):
            await self._emit_event(
                "exchange.trades.snapshot",
                {
                    "exchange": self.EXCHANGE,
                    "market_type": "usdm_futures",
                    "symbol": symbol,
                    "trades": [],
                    "count": 0,
                    "snapshot_time": self._current_timestamp_ms(),
                    "skipped": True,
                    "skip_reason": "missing_credentials",
                },
                priority=EventPriority.LOW,
            )
            return []

        params: dict[str, Any] = {
            "symbol": symbol,
            "limit": limit,
            "recvWindow": recv_window or self._rest_config.recv_window,
        }

        if order_id is not None:
            params["orderId"] = order_id
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        payload = await self._request(
            method="GET",
            path="/fapi/v1/userTrades",
            params=params,
            signed=True,
            auth_required=True,
        )

        normalized = [
            self._normalize_user_trade(trade, symbol=symbol)
            for trade in payload
            if isinstance(trade, dict)
        ]

        await self._emit_event(
            "exchange.trades.snapshot",
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "symbol": symbol,
                "trades": normalized,
                "count": len(normalized),
                "snapshot_time": self._current_timestamp_ms(),
            },
            priority=EventPriority.NORMAL,
        )

        return normalized

    # ------------------------------------------------------------------
    # Futures trading endpoints
    # ------------------------------------------------------------------

    async def create_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float | None = None,
        price: float | None = None,
        position_side: str | None = None,
        time_in_force: str | None = None,
        reduce_only: bool | None = None,
        new_client_order_id: str | None = None,
        stop_price: float | None = None,
        close_position: bool | None = None,
        activation_price: float | None = None,
        callback_rate: float | None = None,
        working_type: str | None = None,
        price_protect: bool | None = None,
        new_order_resp_type: str | None = "RESULT",
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        """
        Low-level Binance USD-M Futures order endpoint.

        This method must be called by execution/order_manager layer.
        It does not decide whether an order should be opened.

        Common examples:
        - MARKET open long: side=BUY, order_type=MARKET, position_side=LONG
        - MARKET open short: side=SELL, order_type=MARKET, position_side=SHORT
        - LIMIT: provide price + time_in_force
        - STOP_MARKET close: provide stop_price + close_position=True
        """

        symbol = symbol.upper()

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "recvWindow": recv_window or self._rest_config.recv_window,
        }

        if quantity is not None:
            params["quantity"] = self._format_number(quantity)

        if price is not None:
            params["price"] = self._format_number(price)

        if position_side is not None:
            params["positionSide"] = position_side.upper()

        if time_in_force is not None:
            params["timeInForce"] = time_in_force.upper()

        if reduce_only is not None:
            params["reduceOnly"] = self._bool_str(reduce_only)

        if new_client_order_id is not None:
            params["newClientOrderId"] = new_client_order_id

        if stop_price is not None:
            params["stopPrice"] = self._format_number(stop_price)

        if close_position is not None:
            params["closePosition"] = self._bool_str(close_position)

        if activation_price is not None:
            params["activationPrice"] = self._format_number(activation_price)

        if callback_rate is not None:
            params["callbackRate"] = self._format_number(callback_rate)

        if working_type is not None:
            params["workingType"] = working_type.upper()

        if price_protect is not None:
            params["priceProtect"] = self._bool_str(price_protect)

        if new_order_resp_type is not None:
            params["newOrderRespType"] = new_order_resp_type.upper()

        payload = await self._request(
            method="POST",
            path="/fapi/v1/order",
            params=params,
            signed=True,
            auth_required=True,
        )

        normalized = self._normalize_order(payload)

        await self._emit_event(
            "exchange.order.submitted",
            normalized,
            priority=EventPriority.CRITICAL,
        )

        return normalized

    async def cancel_order(
        self,
        *,
        symbol: str,
        order_id: int | None = None,
        orig_client_order_id: str | None = None,
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        if order_id is None and orig_client_order_id is None:
            raise ValueError("Either order_id or orig_client_order_id must be provided")

        symbol = symbol.upper()

        params: dict[str, Any] = {
            "symbol": symbol,
            "recvWindow": recv_window or self._rest_config.recv_window,
        }

        if order_id is not None:
            params["orderId"] = order_id
        if orig_client_order_id is not None:
            params["origClientOrderId"] = orig_client_order_id

        payload = await self._request(
            method="DELETE",
            path="/fapi/v1/order",
            params=params,
            signed=True,
            auth_required=True,
        )

        normalized = self._normalize_order(payload)

        await self._emit_event(
            "exchange.order.cancelled",
            normalized,
            priority=EventPriority.CRITICAL,
        )

        return normalized

    async def cancel_all_open_orders(
        self,
        *,
        symbol: str,
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        symbol = symbol.upper()

        payload = await self._request(
            method="DELETE",
            path="/fapi/v1/allOpenOrders",
            params={
                "symbol": symbol,
                "recvWindow": recv_window or self._rest_config.recv_window,
            },
            signed=True,
            auth_required=True,
        )

        normalized = {
            "exchange": self.EXCHANGE,
            "market_type": "usdm_futures",
            "symbol": symbol,
            "code": payload.get("code"),
            "message": payload.get("msg"),
            "timestamp": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.orders.cancelled",
            normalized,
            priority=EventPriority.CRITICAL,
        )

        return normalized

    async def change_leverage(
        self,
        *,
        symbol: str,
        leverage: int,
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        if leverage < 1 or leverage > 125:
            raise ValueError("Binance Futures leverage must be between 1 and 125")

        symbol = symbol.upper()

        payload = await self._request(
            method="POST",
            path="/fapi/v1/leverage",
            params={
                "symbol": symbol,
                "leverage": leverage,
                "recvWindow": recv_window or self._rest_config.recv_window,
            },
            signed=True,
            auth_required=True,
        )

        normalized = {
            "exchange": self.EXCHANGE,
            "market_type": "usdm_futures",
            "symbol": payload.get("symbol") or symbol,
            "leverage": self._safe_int(payload.get("leverage")),
            "max_notional_value": self._safe_float(payload.get("maxNotionalValue")),
            "timestamp": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.leverage.changed",
            normalized,
            priority=EventPriority.HIGH,
        )

        return normalized

    async def change_margin_type(
        self,
        *,
        symbol: str,
        margin_type: str,
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        symbol = symbol.upper()
        margin_type = margin_type.upper()

        if margin_type not in {"ISOLATED", "CROSSED"}:
            raise ValueError("margin_type must be ISOLATED or CROSSED")

        payload = await self._request(
            method="POST",
            path="/fapi/v1/marginType",
            params={
                "symbol": symbol,
                "marginType": margin_type,
                "recvWindow": recv_window or self._rest_config.recv_window,
            },
            signed=True,
            auth_required=True,
        )

        normalized = {
            "exchange": self.EXCHANGE,
            "market_type": "usdm_futures",
            "symbol": symbol,
            "margin_type": margin_type,
            "code": payload.get("code"),
            "message": payload.get("msg"),
            "timestamp": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.margin_type.changed",
            normalized,
            priority=EventPriority.HIGH,
        )

        return normalized

    async def change_position_mode(
        self,
        *,
        dual_side_position: bool,
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        """
        dual_side_position=False -> One-way Mode
        dual_side_position=True  -> Hedge Mode
        """

        payload = await self._request(
            method="POST",
            path="/fapi/v1/positionSide/dual",
            params={
                "dualSidePosition": self._bool_str(dual_side_position),
                "recvWindow": recv_window or self._rest_config.recv_window,
            },
            signed=True,
            auth_required=True,
        )

        normalized = {
            "exchange": self.EXCHANGE,
            "market_type": "usdm_futures",
            "dual_side_position": dual_side_position,
            "code": payload.get("code"),
            "message": payload.get("msg"),
            "timestamp": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.position_mode.changed",
            normalized,
            priority=EventPriority.HIGH,
        )

        return normalized

    # ------------------------------------------------------------------
    # Core HTTP request logic
    # ------------------------------------------------------------------

    async def _request(
        self,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        auth_required: bool = False,
    ) -> Any:
        await self._ensure_session()

        assert self._session is not None

        method = method.upper()
        base_params = dict(params or {})
        headers: dict[str, str] = {}

        if auth_required:
            self._require_credentials()
            assert self._api_key is not None
            headers["X-MBX-APIKEY"] = self._api_key

        if signed:
            self._require_credentials()
            assert self._api_secret is not None

            # Keep signed requests tolerant to local clock drift and request latency.
            # Binance returns -1021 when timestamp is outside recvWindow; on that
            # error we resync server time, rebuild timestamp/signature, and retry.
            if self._time_offset_ms == 0:
                await self.sync_time()

            recv_window = self._safe_int(base_params.get("recvWindow")) or self._rest_config.recv_window
            recv_window = min(max(recv_window, self._rest_config.recv_window), self._rest_config.max_recv_window)
            base_params["recvWindow"] = recv_window

        url = f"{self._rest_config.rest_url}{path}"

        last_error: Exception | None = None

        for attempt in range(self._rest_config.request_retries + 1):
            params_for_attempt = dict(base_params)

            if signed:
                assert self._api_secret is not None
                params_for_attempt["timestamp"] = self._current_timestamp_ms() + self._time_offset_ms
                query_string = self._build_query_string(params_for_attempt)
                params_for_attempt["signature"] = self._sign(query_string)

            try:
                self._logger.debug(
                    "Sending Binance Futures REST request | method=%s path=%s signed=%s auth_required=%s attempt=%s time_offset_ms=%s",
                    method,
                    path,
                    signed,
                    auth_required,
                    attempt + 1,
                    self._time_offset_ms,
                )

                async with self._session.request(
                    method=method,
                    url=url,
                    params=params_for_attempt,
                    headers=headers,
                ) as response:
                    response_text = await response.text()

                    if response.status >= 400:
                        error_payload = self._parse_error_payload(response_text)
                        error_code = self._safe_int(error_payload.get("code"))

                        # Symbol-unavailable errors are non-retryable. Raise
                        # immediately so the caller can disable the symbol.
                        if (
                            isinstance(error_code, int)
                            and error_code in self._rest_config.symbol_unavailable_error_codes
                        ):
                            symbol_hint = self._safe_symbol_from_params(base_params)
                            self._logger.warning(
                                "Binance symbol unavailable (non-retryable) | method=%s path=%s code=%s symbol=%s message=%s",
                                method,
                                path,
                                error_code,
                                symbol_hint,
                                error_payload.get("message"),
                            )
                            raise BinanceSymbolUnavailableError(
                                code=error_code,
                                message=str(error_payload.get("message") or ""),
                                symbol=symbol_hint,
                            )

                        await self._handle_http_error(
                            method=method,
                            path=path,
                            status=response.status,
                            error_payload=error_payload,
                        )

                        if (
                            signed
                            and error_code in self._rest_config.resync_time_error_codes
                            and attempt < self._rest_config.request_retries
                        ):
                            self._logger.warning(
                                "Binance timestamp rejected; resyncing server time before retry | method=%s path=%s code=%s",
                                method,
                                path,
                                error_code,
                            )
                            await self.sync_time()
                            await asyncio.sleep(self._rest_config.retry_delay_seconds)
                            continue

                        raise RuntimeError(
                            f"Binance Futures REST error | method={method} path={path} "
                            f"status={response.status} code={error_payload.get('code')} "
                            f"message={error_payload.get('message')}"
                        )

                    payload = await response.json()

                    # Binance Futures may return {"code":..., "msg":...} for some business errors
                    # even with HTTP 200.
                    code = payload.get("code") if isinstance(payload, dict) else None
                    code_i = self._safe_int(code)
                    if isinstance(code_i, int) and code_i < 0:
                        # Symbol-unavailable business errors — no retry.
                        if code_i in self._rest_config.symbol_unavailable_error_codes:
                            symbol_hint = self._safe_symbol_from_params(base_params)
                            self._logger.warning(
                                "Binance symbol unavailable (business error, non-retryable) | method=%s path=%s code=%s symbol=%s msg=%s",
                                method,
                                path,
                                code_i,
                                symbol_hint,
                                payload.get("msg"),
                            )
                            raise BinanceSymbolUnavailableError(
                                code=code_i,
                                message=str(payload.get("msg") or ""),
                                symbol=symbol_hint,
                            )

                        await self._handle_business_error(
                            method=method,
                            path=path,
                            payload=payload,
                        )

                        if (
                            signed
                            and code_i in self._rest_config.resync_time_error_codes
                            and attempt < self._rest_config.request_retries
                        ):
                            self._logger.warning(
                                "Binance business timestamp error; resyncing server time before retry | method=%s path=%s code=%s",
                                method,
                                path,
                                code_i,
                            )
                            await self.sync_time()
                            await asyncio.sleep(self._rest_config.retry_delay_seconds)
                            continue

                        raise RuntimeError(
                            f"Binance Futures business error | method={method} path={path} "
                            f"code={code_i} msg={payload.get('msg')}"
                        )

                    if self._rest_config.emit_success_events:
                        await self._emit_event(
                            "system.exchange.rest.success",
                            {
                                "exchange": self.EXCHANGE,
                                "market_type": "usdm_futures",
                                "method": method,
                                "path": path,
                                "status": response.status,
                            },
                            priority=EventPriority.LOW,
                        )

                    return payload

            except asyncio.CancelledError:
                raise
            except BinanceSymbolUnavailableError:
                # Never retry — propagate immediately so callers can disable the symbol.
                raise
            except asyncio.TimeoutError as exc:
                last_error = exc
                self._logger.warning(
                    "Binance Futures REST timeout | method=%s path=%s attempt=%s/%s timeout_seconds=%s",
                    method,
                    path,
                    attempt + 1,
                    self._rest_config.request_retries + 1,
                    self._rest_config.timeout_seconds,
                )
                if attempt >= self._rest_config.request_retries:
                    raise
                await asyncio.sleep(self._rest_config.retry_delay_seconds)
            except Exception as exc:
                last_error = exc

                if attempt >= self._rest_config.request_retries:
                    self._logger.exception(
                        "Binance Futures REST request failed | method=%s path=%s attempts=%s",
                        method,
                        path,
                        attempt + 1,
                    )
                    raise

                self._logger.warning(
                    "Binance Futures REST retry scheduled | method=%s path=%s attempt=%s",
                    method,
                    path,
                    attempt + 1,
                )

                await asyncio.sleep(self._rest_config.retry_delay_seconds)

        if last_error is not None:
            raise last_error

        raise RuntimeError(f"Binance Futures REST request failed unexpectedly | method={method} path={path}")

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            await self.start()

    # ------------------------------------------------------------------
    # Derivative symbol blocklist management
    # ------------------------------------------------------------------

    def block_derivative_symbol(self, symbol: str) -> None:
        """
        Add a symbol to the runtime derivative polling blocklist.

        The symbol will be skipped on all future ``get_open_interest()`` calls
        without any REST request being made.  This is called automatically when
        Binance returns a symbol-unavailable error (-4108 / -1121), but callers
        can also pre-populate the blocklist via config or by calling this method
        directly.
        """
        self._derivative_symbol_blocklist.add(symbol.upper())
        self._logger.info(
            "Symbol added to derivative polling blocklist | symbol=%s total_blocked=%s",
            symbol.upper(),
            len(self._derivative_symbol_blocklist),
        )

    def unblock_derivative_symbol(self, symbol: str) -> None:
        """Remove a symbol from the runtime derivative polling blocklist."""
        self._derivative_symbol_blocklist.discard(symbol.upper())

    def derivative_symbol_blocklist(self) -> frozenset[str]:
        """Return a snapshot of the current derivative polling blocklist."""
        return frozenset(self._derivative_symbol_blocklist)

    @staticmethod
    def _safe_symbol_from_params(params: dict[str, Any]) -> str | None:
        """Best-effort extraction of the symbol from request params for error logging."""
        for key in ("symbol", "Symbol", "SYMBOL"):
            val = params.get(key)
            if val:
                return str(val)
        return None

    async def _handle_http_error(
        self,
        *,
        method: str,
        path: str,
        status: int,
        error_payload: dict[str, Any],
    ) -> None:
        self._logger.error(
            "Binance Futures REST HTTP error | method=%s path=%s status=%s code=%s",
            method,
            path,
            status,
            error_payload.get("code"),
        )

        if not self._rest_config.emit_error_events:
            return

        await self._emit_event(
            "system.exchange.rest.error",
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "method": method,
                "path": path,
                "status": status,
                "code": error_payload.get("code"),
                "message": error_payload.get("message"),
            },
            priority=EventPriority.HIGH,
        )

    async def _handle_business_error(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any],
    ) -> None:
        self._logger.error(
            "Binance Futures REST business error | method=%s path=%s code=%s",
            method,
            path,
            payload.get("code"),
        )

        if not self._rest_config.emit_error_events:
            return

        await self._emit_event(
            "system.exchange.rest.error",
            {
                "exchange": self.EXCHANGE,
                "market_type": "usdm_futures",
                "method": method,
                "path": path,
                "code": payload.get("code"),
                "message": payload.get("msg") or payload.get("message"),
            },
            priority=EventPriority.HIGH,
        )

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
                "Failed to emit Binance Futures REST event | topic=%s",
                topic,
            )

    # ------------------------------------------------------------------
    # Auth / signing helpers
    # ------------------------------------------------------------------

    def has_credentials(self) -> bool:
        return bool(self._api_key and self._api_secret)

    def _should_skip_private_read_without_credentials(self, operation: str) -> bool:
        if self.has_credentials():
            return False

        if not self._rest_config.allow_private_read_without_credentials:
            return False

        self._logger.warning(
            "Skipping Binance Futures private read without credentials | operation=%s",
            operation,
        )
        return True

    def _require_credentials(self) -> None:
        if not self.has_credentials():
            raise RuntimeError(
                "Binance Futures API credentials are required for this endpoint"
            )

    def _sign(self, query_string: str) -> str:
        self._require_credentials()

        assert self._api_secret is not None

        return hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    # ------------------------------------------------------------------
    # Normalizers
    # ------------------------------------------------------------------

    def _normalize_kline(
        self,
        *,
        item: list[Any],
        symbol: str,
        timeframe: str,
        is_closed: bool,
    ) -> dict[str, Any]:
        return {
            "exchange": self.EXCHANGE,
            "market_type": "usdm_futures",
            "symbol": symbol,
            "timeframe": timeframe,
            "open_time": self._safe_int(item[0]),
            "open": self._safe_float(item[1]),
            "high": self._safe_float(item[2]),
            "low": self._safe_float(item[3]),
            "close": self._safe_float(item[4]),
            "volume": self._safe_float(item[5]),
            "close_time": self._safe_int(item[6]),
            "quote_volume": self._safe_float(item[7]),
            "trades_count": self._safe_int(item[8]),
            "taker_buy_base_volume": self._safe_float(item[9]),
            "taker_buy_quote_volume": self._safe_float(item[10]),
            "is_closed": is_closed,
        }

    def _normalize_position(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "exchange": self.EXCHANGE,
            "market_type": "usdm_futures",
            "symbol": payload.get("symbol"),
            "position_side": payload.get("positionSide"),
            "position_amt": self._safe_float(payload.get("positionAmt")),
            "entry_price": self._safe_float(payload.get("entryPrice")),
            "break_even_price": self._safe_float(payload.get("breakEvenPrice")),
            "mark_price": self._safe_float(payload.get("markPrice")),
            "unrealized_profit": self._safe_float(payload.get("unRealizedProfit")),
            "liquidation_price": self._safe_float(payload.get("liquidationPrice")),
            "leverage": self._safe_int(payload.get("leverage")),
            "max_notional_value": self._safe_float(payload.get("maxNotionalValue")),
            "margin_type": payload.get("marginType"),
            "isolated_margin": self._safe_float(payload.get("isolatedMargin")),
            "is_auto_add_margin": payload.get("isAutoAddMargin"),
            "update_time": payload.get("updateTime"),
            "notional": self._safe_float(payload.get("notional")),
            "isolated_wallet": self._safe_float(payload.get("isolatedWallet")),
        }

    def _normalize_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "exchange": self.EXCHANGE,
            "market_type": "usdm_futures",
            "symbol": payload.get("symbol"),
            "order_id": payload.get("orderId"),
            "client_order_id": payload.get("clientOrderId"),
            "price": self._safe_float(payload.get("price")),
            "avg_price": self._safe_float(payload.get("avgPrice")),
            "orig_qty": self._safe_float(payload.get("origQty")),
            "executed_qty": self._safe_float(payload.get("executedQty")),
            "cum_qty": self._safe_float(payload.get("cumQty")),
            "cum_quote": self._safe_float(payload.get("cumQuote")),
            "cumulative_quote_qty": self._safe_float(payload.get("cumQuote")),
            "status": payload.get("status"),
            "time_in_force": payload.get("timeInForce"),
            "type": payload.get("type"),
            "orig_type": payload.get("origType"),
            "side": payload.get("side"),
            "position_side": payload.get("positionSide"),
            "reduce_only": payload.get("reduceOnly"),
            "close_position": payload.get("closePosition"),
            "stop_price": self._safe_float(payload.get("stopPrice")),
            "working_type": payload.get("workingType"),
            "price_protect": payload.get("priceProtect"),
            "update_time": payload.get("updateTime"),
            "time": payload.get("time"),
        }

    def _normalize_user_trade(
        self,
        payload: dict[str, Any],
        *,
        symbol: str,
    ) -> dict[str, Any]:
        return {
            "exchange": self.EXCHANGE,
            "market_type": "usdm_futures",
            "symbol": payload.get("symbol") or symbol,
            "id": payload.get("id"),
            "order_id": payload.get("orderId"),
            "side": payload.get("side"),
            "position_side": payload.get("positionSide"),
            "price": self._safe_float(payload.get("price")),
            "qty": self._safe_float(payload.get("qty")),
            "quote_qty": self._safe_float(payload.get("quoteQty")),
            "realized_pnl": self._safe_float(payload.get("realizedPnl")),
            "margin_asset": payload.get("marginAsset"),
            "commission": self._safe_float(payload.get("commission")),
            "commission_asset": payload.get("commissionAsset"),
            "time": payload.get("time"),
            "buyer": payload.get("buyer"),
            "maker": payload.get("maker"),
        }

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query_string(params: dict[str, Any]) -> str:
        filtered = {
            key: value
            for key, value in params.items()
            if value is not None
        }

        return urlencode(filtered, doseq=True)

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
    def _format_number(value: float) -> str:
        formatted = format(value, "f")
        return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted

    @staticmethod
    def _bool_str(value: bool) -> str:
        return "true" if value else "false"

    @staticmethod
    def _mul_safe(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None

        return left * right

    @staticmethod
    def _parse_error_payload(response_text: str) -> dict[str, Any]:
        try:
            import json

            raw = json.loads(response_text)

            return {
                "code": raw.get("code"),
                "message": raw.get("msg") or raw.get("message"),
            }
        except Exception:
            return {
                "code": None,
                "message": "Unable to parse Binance Futures error response",
            }

    @staticmethod
    def _current_timestamp_ms() -> int:
        return int(time.time() * 1000)