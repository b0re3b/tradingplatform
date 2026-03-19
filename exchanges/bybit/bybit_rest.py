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


class BybitRestClient:
    """
    Bybit REST client for public and private endpoints.

    Підходить для:
    - market data snapshots
    - instrument metadata
    - wallet / position / orders
    - order placement / amend / cancel
    """

    DEFAULT_REST_URL = "https://api.bybit.com"

    API_KEY_PLACEHOLDER = "BYBIT_API_KEY_PLACEHOLDER"
    API_SECRET_PLACEHOLDER = "BYBIT_API_SECRET_PLACEHOLDER"

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
            exchange="bybit",
            event_type="bybit_rest",
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

        self._recv_window = 5000
        self._session: Optional[aiohttp.ClientSession] = None
        self._time_offset_ms: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            self._logger.warning("Bybit REST client already started")
            return

        timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)

        self._logger.info("Bybit REST client started | base_url=%s", self._rest_url)

    async def stop(self) -> None:
        if self._session is None:
            self._logger.warning("Bybit REST client already stopped")
            return

        await self._session.close()
        self._session = None

        self._logger.info("Bybit REST client stopped")

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    async def get_server_time(self) -> dict[str, Any]:
        payload = await self._request(
            method="GET",
            path="/v5/market/time",
            signed=False,
        )
        return payload

    async def sync_time(self) -> int:
        local_before = int(time.time() * 1000)
        payload = await self.get_server_time()
        local_after = int(time.time() * 1000)

        result = payload.get("result", {})
        time_second = result.get("timeSecond")
        time_nano = result.get("timeNano")

        if time_nano is not None:
            server_time = int(int(time_nano) / 1_000_000)
        elif time_second is not None:
            server_time = int(time_second) * 1000
        else:
            raise RuntimeError("Bybit server time missing in response")

        local_estimated = (local_before + local_after) // 2
        self._time_offset_ms = server_time - local_estimated

        self._logger.info(
            "Bybit time synced | server_time=%s time_offset_ms=%s",
            server_time,
            self._time_offset_ms,
        )
        return self._time_offset_ms

    async def get_instruments_info(
        self,
        *,
        category: str = "linear",
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "category": category,
            "limit": limit,
        }
        if symbol:
            params["symbol"] = symbol.upper()
        if base_coin:
            params["baseCoin"] = base_coin.upper()

        payload = await self._request(
            method="GET",
            path="/v5/market/instruments-info",
            params=params,
            signed=False,
        )

        await self._emit_event(
            "market.instruments.info",
            {
                "exchange": "bybit",
                "category": category,
                "symbol": symbol.upper() if symbol else None,
            },
            priority=EventPriority.LOW,
        )
        return payload

    async def get_orderbook(
        self,
        *,
        category: str = "linear",
        symbol: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = await self._request(
            method="GET",
            path="/v5/market/orderbook",
            params={
                "category": category,
                "symbol": symbol.upper(),
                "limit": limit,
            },
            signed=False,
        )

        result = payload.get("result", {})

        normalized = {
            "exchange": "bybit",
            "category": category,
            "symbol": result.get("s") or symbol.upper(),
            "update_id": result.get("u"),
            "sequence": result.get("seq"),
            "timestamp": result.get("ts"),
            "bids": [
                [self._safe_float(price), self._safe_float(qty)]
                for price, qty in result.get("b", [])
            ],
            "asks": [
                [self._safe_float(price), self._safe_float(qty)]
                for price, qty in result.get("a", [])
            ],
        }

        await self._emit_event(
            "market.orderbook.snapshot",
            normalized,
            priority=EventPriority.NORMAL,
        )
        return normalized

    async def get_public_trades(
        self,
        *,
        category: str = "linear",
        symbol: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        payload = await self._request(
            method="GET",
            path="/v5/market/recent-trade",
            params={
                "category": category,
                "symbol": symbol.upper(),
                "limit": limit,
            },
            signed=False,
        )

        result = payload.get("result", {})
        trades = result.get("list", [])

        normalized = [
            {
                "exchange": "bybit",
                "category": category,
                "symbol": item.get("symbol") or symbol.upper(),
                "trade_id": item.get("execId"),
                "price": self._safe_float(item.get("price")),
                "qty": self._safe_float(item.get("size")),
                "side": (item.get("side") or "").lower(),
                "time": self._safe_int(item.get("time")),
                "is_block_trade": item.get("isBlockTrade"),
            }
            for item in trades
        ]

        await self._emit_event(
            "market.trades.snapshot",
            {
                "exchange": "bybit",
                "category": category,
                "symbol": symbol.upper(),
                "count": len(normalized),
            },
            priority=EventPriority.LOW,
        )
        return normalized

    async def get_klines(
        self,
        *,
        category: str = "linear",
        symbol: str,
        interval: str = "1",
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end

        payload = await self._request(
            method="GET",
            path="/v5/market/kline",
            params=params,
            signed=False,
        )

        result = payload.get("result", {})
        raw_klines = result.get("list", [])

        normalized = [
            {
                "exchange": "bybit",
                "category": category,
                "symbol": symbol.upper(),
                "interval": interval,
                "start_time": self._safe_int(item[0]),
                "open": self._safe_float(item[1]),
                "high": self._safe_float(item[2]),
                "low": self._safe_float(item[3]),
                "close": self._safe_float(item[4]),
                "volume": self._safe_float(item[5]),
                "turnover": self._safe_float(item[6]),
            }
            for item in raw_klines
        ]

        await self._emit_event(
            "market.klines.snapshot",
            {
                "exchange": "bybit",
                "category": category,
                "symbol": symbol.upper(),
                "interval": interval,
                "count": len(normalized),
            },
            priority=EventPriority.LOW,
        )
        return normalized

    async def get_tickers(
        self,
        *,
        category: str = "linear",
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "category": category,
        }
        if symbol:
            params["symbol"] = symbol.upper()
        if base_coin:
            params["baseCoin"] = base_coin.upper()

        payload = await self._request(
            method="GET",
            path="/v5/market/tickers",
            params=params,
            signed=False,
        )

        result = payload.get("result", {})
        raw_items = result.get("list", [])

        normalized = [
            {
                "exchange": "bybit",
                "category": category,
                "symbol": item.get("symbol"),
                "last_price": self._safe_float(item.get("lastPrice")),
                "index_price": self._safe_float(item.get("indexPrice")),
                "mark_price": self._safe_float(item.get("markPrice")),
                "open_interest": self._safe_float(item.get("openInterest")),
                "open_interest_value": self._safe_float(item.get("openInterestValue")),
                "turnover_24h": self._safe_float(item.get("turnover24h")),
                "volume_24h": self._safe_float(item.get("volume24h")),
                "funding_rate": self._safe_float(item.get("fundingRate")),
                "bid1_price": self._safe_float(item.get("bid1Price")),
                "ask1_price": self._safe_float(item.get("ask1Price")),
                "next_funding_time": self._safe_int(item.get("nextFundingTime")),
            }
            for item in raw_items
        ]

        await self._emit_event(
            "market.tickers.snapshot",
            {
                "exchange": "bybit",
                "category": category,
                "symbol": symbol.upper() if symbol else None,
                "count": len(normalized),
            },
            priority=EventPriority.LOW,
        )
        return normalized

    # ------------------------------------------------------------------
    # Private endpoints
    # ------------------------------------------------------------------

    async def get_wallet_balance(
        self,
        *,
        account_type: str = "UNIFIED",
        coin: Optional[str] = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "accountType": account_type,
        }
        if coin:
            params["coin"] = coin.upper()

        payload = await self._request(
            method="GET",
            path="/v5/account/wallet-balance",
            params=params,
            signed=True,
        )

        await self._emit_event(
            "account.wallet.snapshot",
            {
                "exchange": "bybit",
                "account_type": account_type,
                "coin": coin.upper() if coin else None,
            },
            priority=EventPriority.HIGH,
        )
        return payload

    async def get_positions(
        self,
        *,
        category: str = "linear",
        symbol: Optional[str] = None,
        settle_coin: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "category": category,
            "limit": limit,
        }
        if symbol:
            params["symbol"] = symbol.upper()
        if settle_coin:
            params["settleCoin"] = settle_coin.upper()

        payload = await self._request(
            method="GET",
            path="/v5/position/list",
            params=params,
            signed=True,
        )

        result = payload.get("result", {})
        items = result.get("list", [])

        normalized = [
            {
                "exchange": "bybit",
                "category": category,
                "symbol": item.get("symbol"),
                "side": item.get("side"),
                "size": self._safe_float(item.get("size")),
                "avg_price": self._safe_float(item.get("avgPrice")),
                "position_value": self._safe_float(item.get("positionValue")),
                "mark_price": self._safe_float(item.get("markPrice")),
                "liq_price": self._safe_float(item.get("liqPrice")),
                "leverage": self._safe_float(item.get("leverage")),
                "unrealised_pnl": self._safe_float(item.get("unrealisedPnl")),
                "take_profit": self._safe_float(item.get("takeProfit")),
                "stop_loss": self._safe_float(item.get("stopLoss")),
                "trailing_stop": self._safe_float(item.get("trailingStop")),
                "updated_time": self._safe_int(item.get("updatedTime")),
            }
            for item in items
        ]

        await self._emit_event(
            "position.snapshot",
            {
                "exchange": "bybit",
                "category": category,
                "symbol": symbol.upper() if symbol else None,
                "count": len(normalized),
            },
            priority=EventPriority.HIGH,
        )
        return normalized

    async def get_open_orders(
        self,
        *,
        category: str = "linear",
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        settle_coin: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "category": category,
            "limit": limit,
        }
        if symbol:
            params["symbol"] = symbol.upper()
        if base_coin:
            params["baseCoin"] = base_coin.upper()
        if settle_coin:
            params["settleCoin"] = settle_coin.upper()

        payload = await self._request(
            method="GET",
            path="/v5/order/realtime",
            params=params,
            signed=True,
        )

        result = payload.get("result", {})
        items = result.get("list", [])

        normalized = [self._normalize_order(item, category=category) for item in items]

        await self._emit_event(
            "execution.open_orders.snapshot",
            {
                "exchange": "bybit",
                "category": category,
                "symbol": symbol.upper() if symbol else None,
                "count": len(normalized),
            },
            priority=EventPriority.HIGH,
        )
        return normalized

    async def get_order_history(
        self,
        *,
        category: str = "linear",
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "category": category,
            "limit": limit,
        }
        if symbol:
            params["symbol"] = symbol.upper()
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id

        payload = await self._request(
            method="GET",
            path="/v5/order/history",
            params=params,
            signed=True,
        )

        result = payload.get("result", {})
        items = result.get("list", [])

        normalized = [self._normalize_order(item, category=category) for item in items]

        await self._emit_event(
            "execution.order_history.snapshot",
            {
                "exchange": "bybit",
                "category": category,
                "symbol": symbol.upper() if symbol else None,
                "count": len(normalized),
            },
            priority=EventPriority.NORMAL,
        )
        return normalized

    async def get_execution_history(
        self,
        *,
        category: str = "linear",
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "category": category,
            "limit": limit,
        }
        if symbol:
            params["symbol"] = symbol.upper()
        if order_id:
            params["orderId"] = order_id

        payload = await self._request(
            method="GET",
            path="/v5/execution/list",
            params=params,
            signed=True,
        )

        result = payload.get("result", {})
        items = result.get("list", [])

        normalized = [
            {
                "exchange": "bybit",
                "category": category,
                "symbol": item.get("symbol"),
                "order_id": item.get("orderId"),
                "exec_id": item.get("execId"),
                "side": item.get("side"),
                "price": self._safe_float(item.get("execPrice")),
                "qty": self._safe_float(item.get("execQty")),
                "value": self._safe_float(item.get("execValue")),
                "fee": self._safe_float(item.get("execFee")),
                "fee_rate": self._safe_float(item.get("feeRate")),
                "exec_time": self._safe_int(item.get("execTime")),
                "is_maker": item.get("isMaker"),
                "order_type": item.get("orderType"),
            }
            for item in items
        ]

        await self._emit_event(
            "execution.history.snapshot",
            {
                "exchange": "bybit",
                "category": category,
                "symbol": symbol.upper() if symbol else None,
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
        category: str = "linear",
        symbol: str,
        side: str,
        order_type: str,
        qty: float,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        time_in_force: Optional[str] = None,
        order_link_id: Optional[str] = None,
        reduce_only: Optional[bool] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        position_idx: Optional[int] = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol.upper(),
            "side": side.capitalize(),
            "orderType": order_type.capitalize(),
            "qty": self._format_number(qty),
        }

        if price is not None:
            params["price"] = self._format_number(price)
        if trigger_price is not None:
            params["triggerPrice"] = self._format_number(trigger_price)
        if time_in_force is not None:
            params["timeInForce"] = time_in_force
        if order_link_id is not None:
            params["orderLinkId"] = order_link_id
        if reduce_only is not None:
            params["reduceOnly"] = reduce_only
        if take_profit is not None:
            params["takeProfit"] = self._format_number(take_profit)
        if stop_loss is not None:
            params["stopLoss"] = self._format_number(stop_loss)
        if position_idx is not None:
            params["positionIdx"] = position_idx

        payload = await self._request(
            method="POST",
            path="/v5/order/create",
            body=params,
            signed=True,
        )

        result = payload.get("result", {})
        normalized = {
            "exchange": "bybit",
            "category": category,
            "symbol": symbol.upper(),
            "order_id": result.get("orderId"),
            "order_link_id": result.get("orderLinkId"),
            "side": side.capitalize(),
            "order_type": order_type.capitalize(),
            "qty": qty,
            "price": price,
        }

        await self._emit_event(
            "execution.order_submitted",
            normalized,
            priority=EventPriority.CRITICAL,
        )
        return normalized

    async def amend_order(
        self,
        *,
        category: str = "linear",
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        qty: Optional[float] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> dict[str, Any]:
        if order_id is None and order_link_id is None:
            raise ValueError("Either order_id or order_link_id must be provided")

        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol.upper(),
        }

        if order_id is not None:
            params["orderId"] = order_id
        if order_link_id is not None:
            params["orderLinkId"] = order_link_id
        if qty is not None:
            params["qty"] = self._format_number(qty)
        if price is not None:
            params["price"] = self._format_number(price)
        if trigger_price is not None:
            params["triggerPrice"] = self._format_number(trigger_price)
        if take_profit is not None:
            params["takeProfit"] = self._format_number(take_profit)
        if stop_loss is not None:
            params["stopLoss"] = self._format_number(stop_loss)

        payload = await self._request(
            method="POST",
            path="/v5/order/amend",
            body=params,
            signed=True,
        )

        result = payload.get("result", {})
        normalized = {
            "exchange": "bybit",
            "category": category,
            "symbol": symbol.upper(),
            "order_id": result.get("orderId"),
            "order_link_id": result.get("orderLinkId"),
        }

        await self._emit_event(
            "execution.order_amended",
            normalized,
            priority=EventPriority.CRITICAL,
        )
        return normalized

    async def cancel_order(
        self,
        *,
        category: str = "linear",
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if order_id is None and order_link_id is None:
            raise ValueError("Either order_id or order_link_id must be provided")

        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol.upper(),
        }

        if order_id is not None:
            params["orderId"] = order_id
        if order_link_id is not None:
            params["orderLinkId"] = order_link_id

        payload = await self._request(
            method="POST",
            path="/v5/order/cancel",
            body=params,
            signed=True,
        )

        result = payload.get("result", {})
        normalized = {
            "exchange": "bybit",
            "category": category,
            "symbol": symbol.upper(),
            "order_id": result.get("orderId"),
            "order_link_id": result.get("orderLinkId"),
        }

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
        body: Optional[dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        if self._session is None or self._session.closed:
            await self.start()

        assert self._session is not None

        method = method.upper()
        params = dict(params or {})
        body = dict(body or {})

        url = f"{self._rest_url}{path}"
        headers: dict[str, str] = {}

        request_kwargs: dict[str, Any] = {}

        if signed:
            if self._time_offset_ms == 0:
                await self.sync_time()

            timestamp = str(self._current_timestamp_ms() + self._time_offset_ms)

            if method == "GET":
                query_string = self._build_query_string(params)
                signature_payload = f"{timestamp}{self._api_key}{self._recv_window}{query_string}"
            else:
                body_json = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
                signature_payload = f"{timestamp}{self._api_key}{self._recv_window}{body_json}"

            signature = self._sign(signature_payload)

            headers.update(
                {
                    "X-BAPI-API-KEY": self._api_key,
                    "X-BAPI-TIMESTAMP": timestamp,
                    "X-BAPI-RECV-WINDOW": str(self._recv_window),
                    "X-BAPI-SIGN": signature,
                }
            )

        if method == "GET":
            request_kwargs["params"] = params
        else:
            headers["Content-Type"] = "application/json"
            request_kwargs["json"] = body

        self._logger.debug(
            "Sending Bybit REST request | method=%s path=%s signed=%s",
            method,
            path,
            signed,
        )

        try:
            async with self._session.request(
                method=method,
                url=url,
                headers=headers,
                **request_kwargs,
            ) as response:
                response_text = await response.text()

                if response.status >= 400:
                    self._logger.error(
                        "Bybit REST request failed | method=%s path=%s status=%s",
                        method,
                        path,
                        response.status,
                    )

                    await self._emit_event(
                        "system.rest.error",
                        {
                            "exchange": "bybit",
                            "method": method,
                            "path": path,
                            "status": response.status,
                            "response": response_text,
                        },
                        priority=EventPriority.HIGH,
                    )

                    raise RuntimeError(
                        f"Bybit REST HTTP error | method={method} path={path} "
                        f"status={response.status} response={response_text}"
                    )

                payload = await response.json()

                ret_code = payload.get("retCode")
                if ret_code != 0:
                    ret_msg = payload.get("retMsg")

                    self._logger.error(
                        "Bybit REST business error | method=%s path=%s ret_code=%s",
                        method,
                        path,
                        ret_code,
                    )

                    await self._emit_event(
                        "system.rest.error",
                        {
                            "exchange": "bybit",
                            "method": method,
                            "path": path,
                            "ret_code": ret_code,
                            "ret_msg": ret_msg,
                        },
                        priority=EventPriority.HIGH,
                    )

                    raise RuntimeError(
                        f"Bybit REST business error | method={method} path={path} "
                        f"retCode={ret_code} retMsg={ret_msg}"
                    )

                await self._emit_event(
                    "system.rest.success",
                    {
                        "exchange": "bybit",
                        "method": method,
                        "path": path,
                        "ret_code": ret_code,
                    },
                    priority=EventPriority.LOW,
                )

                return payload

        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception(
                "Bybit REST request exception | method=%s path=%s",
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
                source="bybit_rest",
            )
        except Exception:
            self._logger.exception("Failed to emit REST event | topic=%s", topic)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sign(self, payload: str) -> str:
        return hmac.new(
            self._api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _build_query_string(params: dict[str, Any]) -> str:
        filtered = {
            key: value
            for key, value in params.items()
            if value is not None
        }
        parts: list[str] = []
        for key in sorted(filtered.keys()):
            parts.append(f"{key}={filtered[key]}")
        return "&".join(parts)

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

    @staticmethod
    def _format_number(value: float) -> str:
        formatted = format(value, "f")
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted

    @staticmethod
    def _current_timestamp_ms() -> int:
        return int(time.time() * 1000)

    def _normalize_order(self, payload: dict[str, Any], *, category: str) -> dict[str, Any]:
        return {
            "exchange": "bybit",
            "category": category,
            "symbol": payload.get("symbol"),
            "order_id": payload.get("orderId"),
            "order_link_id": payload.get("orderLinkId"),
            "side": payload.get("side"),
            "order_type": payload.get("orderType"),
            "order_status": payload.get("orderStatus"),
            "time_in_force": payload.get("timeInForce"),
            "price": self._safe_float(payload.get("price")),
            "qty": self._safe_float(payload.get("qty")),
            "leaves_qty": self._safe_float(payload.get("leavesQty")),
            "cum_exec_qty": self._safe_float(payload.get("cumExecQty")),
            "cum_exec_value": self._safe_float(payload.get("cumExecValue")),
            "avg_price": self._safe_float(payload.get("avgPrice")),
            "trigger_price": self._safe_float(payload.get("triggerPrice")),
            "take_profit": self._safe_float(payload.get("takeProfit")),
            "stop_loss": self._safe_float(payload.get("stopLoss")),
            "reduce_only": payload.get("reduceOnly"),
            "created_time": self._safe_int(payload.get("createdTime")),
            "updated_time": self._safe_int(payload.get("updatedTime")),
        }