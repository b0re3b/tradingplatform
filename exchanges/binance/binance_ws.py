from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from typing import Any, Optional
from urllib.parse import urlencode

import aiohttp

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger


class BinanceRestClient:
    """
    Binance REST client for public and private endpoints.

    Підходить для:
    - snapshot orderbook
    - candles / trades / exchange info
    - account info
    - open orders
    - order placement / cancel
    - user data stream listenKey management
    """

    DEFAULT_REST_URL = "https://api.binance.com"

    API_KEY_PLACEHOLDER = "BINANCE_API_KEY_PLACEHOLDER"
    API_SECRET_PLACEHOLDER = "BINANCE_API_SECRET_PLACEHOLDER"

    def __init__(
        self,
        *,
        config: Config,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus

        self._logger = get_logger(
            __name__,
            exchange="binance",
            event_type="binance_rest",
        )

        self._rest_url = config.exchange.rest_url or self.DEFAULT_REST_URL
        self._timeout_seconds = config.exchange.timeout_seconds

        self._api_key = (
            config.exchange.credentials.api_key
            or self.API_KEY_PLACEHOLDER
        )
        self._api_secret = (
            config.exchange.credentials.api_secret
            or self.API_SECRET_PLACEHOLDER
        )

        self._session: Optional[aiohttp.ClientSession] = None
        self._time_offset_ms: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            self._logger.warning("Binance REST client already started")
            return

        timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)

        self._logger.info("Binance REST client started | base_url=%s", self._rest_url)

    async def stop(self) -> None:
        if self._session is None:
            self._logger.warning("Binance REST client already stopped")
            return

        await self._session.close()
        self._session = None

        self._logger.info("Binance REST client stopped")

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    async def ping(self) -> dict[str, Any]:
        return await self._request(
            method="GET",
            path="/api/v3/ping",
            signed=False,
            auth_required=False,
        )

    async def get_server_time(self) -> dict[str, Any]:
        return await self._request(
            method="GET",
            path="/api/v3/time",
            signed=False,
            auth_required=False,
        )

    async def sync_time(self) -> int:
        """
        Синхронізація локального часу з часом сервера Binance.
        Це важливо для signed endpoint'ів.
        """
        local_before = int(time.time() * 1000)
        payload = await self.get_server_time()
        local_after = int(time.time() * 1000)

        server_time = int(payload["serverTime"])
        local_estimated = (local_before + local_after) // 2
        self._time_offset_ms = server_time - local_estimated

        self._logger.info(
            "Binance time synced | server_time=%s time_offset_ms=%s",
            server_time,
            self._time_offset_ms,
        )
        return self._time_offset_ms

    async def get_exchange_info(self, symbol: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()

        return await self._request(
            method="GET",
            path="/api/v3/exchangeInfo",
            params=params,
            signed=False,
            auth_required=False,
        )

    async def get_orderbook_snapshot(
        self,
        *,
        symbol: str,
        limit: int = 100,
    ) -> dict[str, Any]:
        payload = await self._request(
            method="GET",
            path="/api/v3/depth",
            params={
                "symbol": symbol.upper(),
                "limit": limit,
            },
            signed=False,
            auth_required=False,
        )

        normalized = {
            "exchange": "binance",
            "symbol": symbol.upper(),
            "last_update_id": payload.get("lastUpdateId"),
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
        payload = await self._request(
            method="GET",
            path="/api/v3/trades",
            params={
                "symbol": symbol.upper(),
                "limit": limit,
            },
            signed=False,
            auth_required=False,
        )

        normalized = [
            {
                "exchange": "binance",
                "symbol": symbol.upper(),
                "trade_id": trade.get("id"),
                "price": self._safe_float(trade.get("price")),
                "qty": self._safe_float(trade.get("qty")),
                "quote_qty": self._safe_float(trade.get("quoteQty")),
                "time": trade.get("time"),
                "buyer_is_maker": trade.get("isBuyerMaker"),
                "side": "sell" if trade.get("isBuyerMaker") else "buy",
            }
            for trade in payload
        ]

        await self._emit_event(
            "market.trades.snapshot",
            {
                "exchange": "binance",
                "symbol": symbol.upper(),
                "count": len(normalized),
                "trades": normalized,
            },
            priority=EventPriority.LOW,
        )
        return normalized

    async def get_klines(
        self,
        *,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        payload = await self._request(
            method="GET",
            path="/api/v3/klines",
            params=params,
            signed=False,
            auth_required=False,
        )

        normalized = [
            {
                "exchange": "binance",
                "symbol": symbol.upper(),
                "interval": interval,
                "open_time": item[0],
                "open": self._safe_float(item[1]),
                "high": self._safe_float(item[2]),
                "low": self._safe_float(item[3]),
                "close": self._safe_float(item[4]),
                "volume": self._safe_float(item[5]),
                "close_time": item[6],
                "quote_asset_volume": self._safe_float(item[7]),
                "number_of_trades": item[8],
                "taker_buy_base_volume": self._safe_float(item[9]),
                "taker_buy_quote_volume": self._safe_float(item[10]),
            }
            for item in payload
        ]

        await self._emit_event(
            "market.klines.snapshot",
            {
                "exchange": "binance",
                "symbol": symbol.upper(),
                "interval": interval,
                "count": len(normalized),
            },
            priority=EventPriority.LOW,
        )
        return normalized

    # ------------------------------------------------------------------
    # Listen key endpoints
    # ------------------------------------------------------------------

    async def create_listen_key(self) -> str:
        payload = await self._request(
            method="POST",
            path="/api/v3/userDataStream",
            signed=False,
            auth_required=True,
        )

        listen_key = payload.get("listenKey")
        if not listen_key:
            raise RuntimeError("listenKey missing in Binance response")

        self._logger.info("Binance listen key created")
        return listen_key

    async def keepalive_listen_key(self, listen_key: str) -> None:
        await self._request(
            method="PUT",
            path="/api/v3/userDataStream",
            params={"listenKey": listen_key},
            signed=False,
            auth_required=True,
        )
        self._logger.info("Binance listen key refreshed")

    async def close_listen_key(self, listen_key: str) -> None:
        await self._request(
            method="DELETE",
            path="/api/v3/userDataStream",
            params={"listenKey": listen_key},
            signed=False,
            auth_required=True,
        )
        self._logger.info("Binance listen key closed")

    # ------------------------------------------------------------------
    # Private/account endpoints
    # ------------------------------------------------------------------

    async def get_account_info(self, *, recv_window: int = 5000) -> dict[str, Any]:
        payload = await self._request(
            method="GET",
            path="/api/v3/account",
            params={"recvWindow": recv_window},
            signed=True,
            auth_required=True,
        )

        normalized = {
            "exchange": "binance",
            "maker_commission": payload.get("makerCommission"),
            "taker_commission": payload.get("takerCommission"),
            "buyer_commission": payload.get("buyerCommission"),
            "seller_commission": payload.get("sellerCommission"),
            "can_trade": payload.get("canTrade"),
            "can_withdraw": payload.get("canWithdraw"),
            "can_deposit": payload.get("canDeposit"),
            "account_type": payload.get("accountType"),
            "balances": [
                {
                    "asset": item.get("asset"),
                    "free": self._safe_float(item.get("free")),
                    "locked": self._safe_float(item.get("locked")),
                }
                for item in payload.get("balances", [])
            ],
            "permissions": payload.get("permissions", []),
            "update_time": payload.get("updateTime"),
        }

        await self._emit_event(
            "account.info.updated",
            normalized,
            priority=EventPriority.HIGH,
        )
        return normalized

    async def get_open_orders(
        self,
        *,
        symbol: Optional[str] = None,
        recv_window: int = 5000,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "recvWindow": recv_window,
        }
        if symbol:
            params["symbol"] = symbol.upper()

        payload = await self._request(
            method="GET",
            path="/api/v3/openOrders",
            params=params,
            signed=True,
            auth_required=True,
        )

        normalized = [self._normalize_order(order) for order in payload]

        await self._emit_event(
            "execution.open_orders.snapshot",
            {
                "exchange": "binance",
                "symbol": symbol.upper() if symbol else None,
                "count": len(normalized),
            },
            priority=EventPriority.HIGH,
        )
        return normalized

    async def get_order(
        self,
        *,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None,
        recv_window: int = 5000,
    ) -> dict[str, Any]:
        if order_id is None and orig_client_order_id is None:
            raise ValueError("Either order_id or orig_client_order_id must be provided")

        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "recvWindow": recv_window,
        }

        if order_id is not None:
            params["orderId"] = order_id
        if orig_client_order_id is not None:
            params["origClientOrderId"] = orig_client_order_id

        payload = await self._request(
            method="GET",
            path="/api/v3/order",
            params=params,
            signed=True,
            auth_required=True,
        )

        normalized = self._normalize_order(payload)

        await self._emit_event(
            "execution.order.fetched",
            normalized,
            priority=EventPriority.HIGH,
        )
        return normalized

    async def get_my_trades(
        self,
        *,
        symbol: str,
        limit: int = 500,
        order_id: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        recv_window: int = 5000,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "limit": limit,
            "recvWindow": recv_window,
        }

        if order_id is not None:
            params["orderId"] = order_id
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        payload = await self._request(
            method="GET",
            path="/api/v3/myTrades",
            params=params,
            signed=True,
            auth_required=True,
        )

        normalized = [
            {
                "exchange": "binance",
                "symbol": symbol.upper(),
                "id": trade.get("id"),
                "order_id": trade.get("orderId"),
                "price": self._safe_float(trade.get("price")),
                "qty": self._safe_float(trade.get("qty")),
                "quote_qty": self._safe_float(trade.get("quoteQty")),
                "commission": self._safe_float(trade.get("commission")),
                "commission_asset": trade.get("commissionAsset"),
                "time": trade.get("time"),
                "is_buyer": trade.get("isBuyer"),
                "is_maker": trade.get("isMaker"),
                "is_best_match": trade.get("isBestMatch"),
                "side": "buy" if trade.get("isBuyer") else "sell",
            }
            for trade in payload
        ]

        await self._emit_event(
            "execution.trades.snapshot",
            {
                "exchange": "binance",
                "symbol": symbol.upper(),
                "count": len(normalized),
            },
            priority=EventPriority.NORMAL,
        )
        return normalized

    # ------------------------------------------------------------------
    # Trading endpoints
    # ------------------------------------------------------------------

    async def create_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Optional[float] = None,
        quote_order_qty: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[str] = None,
        new_client_order_id: Optional[str] = None,
        stop_price: Optional[float] = None,
        recv_window: int = 5000,
    ) -> dict[str, Any]:
        """
        Приклади:
        - MARKET BUY/SELL
        - LIMIT BUY/SELL
        - STOP / STOP_LIMIT можна доробити окремо
        """
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "recvWindow": recv_window,
        }

        if quantity is not None:
            params["quantity"] = self._format_number(quantity)

        if quote_order_qty is not None:
            params["quoteOrderQty"] = self._format_number(quote_order_qty)

        if price is not None:
            params["price"] = self._format_number(price)

        if stop_price is not None:
            params["stopPrice"] = self._format_number(stop_price)

        if time_in_force is not None:
            params["timeInForce"] = time_in_force.upper()

        if new_client_order_id is not None:
            params["newClientOrderId"] = new_client_order_id

        payload = await self._request(
            method="POST",
            path="/api/v3/order",
            params=params,
            signed=True,
            auth_required=True,
        )

        normalized = self._normalize_order(payload)

        await self._emit_event(
            "execution.order_submitted",
            normalized,
            priority=EventPriority.CRITICAL,
        )
        return normalized

    async def cancel_order(
        self,
        *,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None,
        recv_window: int = 5000,
    ) -> dict[str, Any]:
        if order_id is None and orig_client_order_id is None:
            raise ValueError("Either order_id or orig_client_order_id must be provided")

        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "recvWindow": recv_window,
        }

        if order_id is not None:
            params["orderId"] = order_id
        if orig_client_order_id is not None:
            params["origClientOrderId"] = orig_client_order_id

        payload = await self._request(
            method="DELETE",
            path="/api/v3/order",
            params=params,
            signed=True,
            auth_required=True,
        )

        normalized = self._normalize_order(payload)

        await self._emit_event(
            "execution.order_cancelled",
            normalized,
            priority=EventPriority.CRITICAL,
        )
        return normalized

    # ------------------------------------------------------------------
    # Core request logic
    # ------------------------------------------------------------------

    async def _request(
        self,
        *,
        method: str,
        path: str,
        params: Optional[dict[str, Any]] = None,
        signed: bool = False,
        auth_required: bool = False,
    ) -> Any:
        if self._session is None or self._session.closed:
            await self.start()

        assert self._session is not None

        method = method.upper()
        params = dict(params or {})
        headers: dict[str, str] = {}

        if auth_required:
            headers["X-MBX-APIKEY"] = self._api_key

        if signed:
            if self._time_offset_ms == 0:
                try:
                    await self.sync_time()
                except Exception:
                    self._logger.exception("Failed to sync Binance server time before signed request")
                    raise

            params["timestamp"] = self._current_timestamp_ms() + self._time_offset_ms
            query_string = self._build_query_string(params)
            signature = self._sign(query_string)
            params["signature"] = signature

        url = f"{self._rest_url}{path}"

        self._logger.debug(
            "Sending Binance REST request | method=%s path=%s signed=%s auth_required=%s",
            method,
            path,
            signed,
            auth_required,
        )

        try:
            async with self._session.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
            ) as response:
                response_text = await response.text()

                if response.status >= 400:
                    self._logger.error(
                        "Binance REST request failed | method=%s path=%s status=%s",
                        method,
                        path,
                        response.status,
                    )

                    await self._emit_event(
                        "system.rest.error",
                        {
                            "exchange": "binance",
                            "method": method,
                            "path": path,
                            "status": response.status,
                            "response": response_text,
                        },
                        priority=EventPriority.HIGH,
                    )

                    raise RuntimeError(
                        f"Binance REST error | method={method} path={path} "
                        f"status={response.status} response={response_text}"
                    )

                payload = await response.json()

                await self._emit_event(
                    "system.rest.success",
                    {
                        "exchange": "binance",
                        "method": method,
                        "path": path,
                        "status": response.status,
                    },
                    priority=EventPriority.LOW,
                )

                return payload

        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception(
                "Binance REST request exception | method=%s path=%s",
                method,
                path,
            )
            raise

    async def _emit_event(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority,
    ) -> None:
        if self._event_bus is None:
            return

        try:
            await self._event_bus.emit(
                topic,
                payload,
                priority=priority,
                source="binance_rest",
            )
        except Exception:
            self._logger.exception(
                "Failed to emit REST event | topic=%s",
                topic,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sign(self, query_string: str) -> str:
        return hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _build_query_string(params: dict[str, Any]) -> str:
        filtered = {
            key: value
            for key, value in params.items()
            if value is not None
        }
        return urlencode(filtered, doseq=True)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_number(value: float) -> str:
        """
        Форматування без scientific notation.
        """
        return format(value, "f").rstrip("0").rstrip(".") if "." in format(value, "f") else format(value, "f")

    @staticmethod
    def _current_timestamp_ms() -> int:
        return int(time.time() * 1000)

    def _normalize_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "exchange": "binance",
            "symbol": payload.get("symbol"),
            "order_id": payload.get("orderId"),
            "client_order_id": payload.get("clientOrderId"),
            "price": self._safe_float(payload.get("price")),
            "orig_qty": self._safe_float(payload.get("origQty")),
            "executed_qty": self._safe_float(payload.get("executedQty")),
            "cummulative_quote_qty": self._safe_float(payload.get("cummulativeQuoteQty")),
            "status": payload.get("status"),
            "time_in_force": payload.get("timeInForce"),
            "type": payload.get("type"),
            "side": payload.get("side"),
            "stop_price": self._safe_float(payload.get("stopPrice")),
            "iceberg_qty": self._safe_float(payload.get("icebergQty")),
            "time": payload.get("time"),
            "update_time": payload.get("updateTime"),
            "is_working": payload.get("isWorking"),
            "working_time": payload.get("workingTime"),
            "orig_quote_order_qty": self._safe_float(payload.get("origQuoteOrderQty")),
            "self_trade_prevention_mode": payload.get("selfTradePreventionMode"),
            "transact_time": payload.get("transactTime"),
        }