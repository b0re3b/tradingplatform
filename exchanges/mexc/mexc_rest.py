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


class MexcRestClient:
    """
    MEXC Futures REST client.

    Підтримує:
    - public market endpoints
    - private account / position / order endpoints
    - HMAC signing
    - інтеграцію з EventBus
    """

    DEFAULT_REST_URL = "https://contract.mexc.com"

    API_KEY_PLACEHOLDER = "MEXC_API_KEY_PLACEHOLDER"
    API_SECRET_PLACEHOLDER = "MEXC_API_SECRET_PLACEHOLDER"

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
            exchange="mexc",
            event_type="mexc_rest",
        )

        configured_rest_url = config.exchange.rest_url
        self._rest_url = configured_rest_url or self.DEFAULT_REST_URL
        self._timeout_seconds = config.exchange.timeout_seconds

        self._api_key = (
            config.exchange.credentials.api_key
            or self.API_KEY_PLACEHOLDER
        )
        self._api_secret = (
            config.exchange.credentials.api_secret
            or self.API_SECRET_PLACEHOLDER
        )

        self._recv_window_ms = 10_000

        self._session: Optional[aiohttp.ClientSession] = None
        self._time_offset_ms: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            self._logger.warning("MEXC REST client already started")
            return

        timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)

        self._logger.info(
            "MEXC REST client started | base_url=%s",
            self._rest_url,
        )

    async def stop(self) -> None:
        if self._session is None:
            self._logger.warning("MEXC REST client already stopped")
            return

        await self._session.close()
        self._session = None

        self._logger.info("MEXC REST client stopped")

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    async def get_server_time(self) -> dict[str, Any]:
        return await self._request(
            method="GET",
            path="/api/v1/contract/ping",
            signed=False,
            auth_required=False,
        )

    async def sync_time(self) -> int:
        local_before = int(time.time() * 1000)
        payload = await self.get_server_time()
        local_after = int(time.time() * 1000)

        data = payload.get("data")
        if data is None:
            raise RuntimeError("MEXC server time missing in response")

        server_time = int(data)
        local_estimated = (local_before + local_after) // 2
        self._time_offset_ms = server_time - local_estimated

        self._logger.info(
            "MEXC time synced | server_time=%s time_offset_ms=%s",
            server_time,
            self._time_offset_ms,
        )
        return self._time_offset_ms

    async def get_contract_info(
        self,
        *,
        symbol: Optional[str] = None,
    ) -> Any:
        path = "/api/v1/contract/detail"
        if symbol:
            path = f"/api/v1/contract/detail/{self._normalize_symbol(symbol)}"

        payload = await self._request(
            method="GET",
            path=path,
            signed=False,
            auth_required=False,
        )

        await self._emit_event(
            "market.instruments.info",
            {
                "exchange": "mexc",
                "symbol": self._normalize_symbol(symbol) if symbol else None,
            },
            priority=EventPriority.LOW,
        )
        return payload

    async def get_orderbook(
        self,
        *,
        symbol: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        normalized_symbol = self._normalize_symbol(symbol)

        payload = await self._request(
            method="GET",
            path=f"/api/v1/contract/depth/{normalized_symbol}",
            params={"limit": limit},
            signed=False,
            auth_required=False,
        )

        data = payload.get("data", {})

        normalized = {
            "exchange": "mexc",
            "symbol": normalized_symbol,
            "version": data.get("version"),
            "timestamp": data.get("timestamp"),
            "asks": [self._normalize_depth_level(level) for level in data.get("asks", [])],
            "bids": [self._normalize_depth_level(level) for level in data.get("bids", [])],
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
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        normalized_symbol = self._normalize_symbol(symbol)

        payload = await self._request(
            method="GET",
            path=f"/api/v1/contract/deals/{normalized_symbol}",
            params={"limit": limit},
            signed=False,
            auth_required=False,
        )

        raw_items = payload.get("data", [])

        normalized = [
            {
                "exchange": "mexc",
                "symbol": normalized_symbol,
                "trade_id": item.get("id"),
                "price": self._safe_float(item.get("price")),
                "qty": self._safe_float(item.get("vol")),
                "side": self._normalize_side_from_int(item.get("side")),
                "time": item.get("time"),
            }
            for item in raw_items
        ]

        await self._emit_event(
            "market.trades.snapshot",
            {
                "exchange": "mexc",
                "symbol": normalized_symbol,
                "count": len(normalized),
            },
            priority=EventPriority.LOW,
        )
        return normalized

    async def get_klines(
        self,
        *,
        symbol: str,
        interval: str = "Min1",
        limit: int = 500,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        normalized_symbol = self._normalize_symbol(symbol)

        params: dict[str, Any] = {
            "interval": interval,
            "limit": limit,
        }
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end

        payload = await self._request(
            method="GET",
            path=f"/api/v1/contract/kline/{normalized_symbol}",
            params=params,
            signed=False,
            auth_required=False,
        )

        data = payload.get("data", {})

        times = data.get("time", [])
        opens = data.get("open", [])
        highs = data.get("high", [])
        lows = data.get("low", [])
        closes = data.get("close", [])
        vols = data.get("vol", [])
        amounts = data.get("amount", [])

        size = min(
            len(times),
            len(opens),
            len(highs),
            len(lows),
            len(closes),
            len(vols),
            len(amounts),
        )

        normalized: list[dict[str, Any]] = []
        for i in range(size):
            normalized.append(
                {
                    "exchange": "mexc",
                    "symbol": normalized_symbol,
                    "interval": interval,
                    "open_time": self._normalize_kline_timestamp(times[i]),
                    "open": self._safe_float(opens[i]),
                    "high": self._safe_float(highs[i]),
                    "low": self._safe_float(lows[i]),
                    "close": self._safe_float(closes[i]),
                    "volume": self._safe_float(vols[i]),
                    "amount": self._safe_float(amounts[i]),
                }
            )

        await self._emit_event(
            "market.klines.snapshot",
            {
                "exchange": "mexc",
                "symbol": normalized_symbol,
                "interval": interval,
                "count": len(normalized),
            },
            priority=EventPriority.LOW,
        )
        return normalized

    async def get_ticker(
        self,
        *,
        symbol: Optional[str] = None,
    ) -> Any:
        if symbol:
            path = f"/api/v1/contract/ticker/{self._normalize_symbol(symbol)}"
        else:
            path = "/api/v1/contract/ticker"

        payload = await self._request(
            method="GET",
            path=path,
            signed=False,
            auth_required=False,
        )

        await self._emit_event(
            "market.tickers.snapshot",
            {
                "exchange": "mexc",
                "symbol": self._normalize_symbol(symbol) if symbol else None,
            },
            priority=EventPriority.LOW,
        )
        return payload

    async def get_funding_rate(
        self,
        *,
        symbol: str,
    ) -> Any:
        normalized_symbol = self._normalize_symbol(symbol)

        payload = await self._request(
            method="GET",
            path=f"/api/v1/contract/funding_rate/{normalized_symbol}",
            signed=False,
            auth_required=False,
        )

        await self._emit_event(
            "market.funding.snapshot",
            {
                "exchange": "mexc",
                "symbol": normalized_symbol,
            },
            priority=EventPriority.NORMAL,
        )
        return payload

    # ------------------------------------------------------------------
    # Private/account endpoints
    # ------------------------------------------------------------------

    async def get_assets(self) -> Any:
        payload = await self._request(
            method="GET",
            path="/api/v1/private/account/assets",
            signed=True,
            auth_required=True,
        )

        await self._emit_event(
            "account.asset.snapshot",
            {"exchange": "mexc"},
            priority=EventPriority.HIGH,
        )
        return payload

    async def get_asset(self, *, currency: str) -> Any:
        payload = await self._request(
            method="GET",
            path=f"/api/v1/private/account/asset/{currency.upper()}",
            signed=True,
            auth_required=True,
        )

        await self._emit_event(
            "account.asset.snapshot",
            {
                "exchange": "mexc",
                "currency": currency.upper(),
            },
            priority=EventPriority.HIGH,
        )
        return payload

    async def get_positions(
        self,
        *,
        symbol: Optional[str] = None,
    ) -> Any:
        if symbol:
            path = f"/api/v1/private/position/open_positions/{self._normalize_symbol(symbol)}"
        else:
            path = "/api/v1/private/position/open_positions"

        payload = await self._request(
            method="GET",
            path=path,
            signed=True,
            auth_required=True,
        )

        await self._emit_event(
            "position.snapshot",
            {
                "exchange": "mexc",
                "symbol": self._normalize_symbol(symbol) if symbol else None,
            },
            priority=EventPriority.HIGH,
        )
        return payload

    async def get_open_orders(
        self,
        *,
        symbol: Optional[str] = None,
        page_num: int = 1,
        page_size: int = 50,
    ) -> Any:
        params: dict[str, Any] = {
            "page_num": page_num,
            "page_size": page_size,
        }
        if symbol:
            params["symbol"] = self._normalize_symbol(symbol)

        payload = await self._request(
            method="GET",
            path="/api/v1/private/order/list/open_orders",
            params=params,
            signed=True,
            auth_required=True,
        )

        await self._emit_event(
            "execution.open_orders.snapshot",
            {
                "exchange": "mexc",
                "symbol": self._normalize_symbol(symbol) if symbol else None,
                "page_num": page_num,
                "page_size": page_size,
            },
            priority=EventPriority.HIGH,
        )
        return payload

    async def get_history_orders(
        self,
        *,
        symbol: Optional[str] = None,
        page_num: int = 1,
        page_size: int = 50,
    ) -> Any:
        params: dict[str, Any] = {
            "page_num": page_num,
            "page_size": page_size,
        }
        if symbol:
            params["symbol"] = self._normalize_symbol(symbol)

        payload = await self._request(
            method="GET",
            path="/api/v1/private/order/list/history_orders",
            params=params,
            signed=True,
            auth_required=True,
        )

        await self._emit_event(
            "execution.order_history.snapshot",
            {
                "exchange": "mexc",
                "symbol": self._normalize_symbol(symbol) if symbol else None,
                "page_num": page_num,
                "page_size": page_size,
            },
            priority=EventPriority.NORMAL,
        )
        return payload

    async def get_order(
        self,
        *,
        order_id: str,
    ) -> Any:
        payload = await self._request(
            method="GET",
            path=f"/api/v1/private/order/get/{order_id}",
            signed=True,
            auth_required=True,
        )

        await self._emit_event(
            "execution.order.fetched",
            {
                "exchange": "mexc",
                "order_id": order_id,
            },
            priority=EventPriority.HIGH,
        )
        return payload

    # ------------------------------------------------------------------
    # Trading endpoints
    # ------------------------------------------------------------------

    async def create_order(
        self,
        *,
        symbol: str,
        price: float,
        vol: float,
        side: int,
        order_type: int,
        open_type: int,
        leverage: Optional[int] = None,
        external_oid: Optional[str] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        position_id: Optional[int] = None,
        reduce_only: Optional[bool] = None,
    ) -> Any:
        """
        side:
            1 open long
            2 close short
            3 open short
            4 close long

        order_type:
            1 limit
            5 market
            інші типи за потреби можна розширити

        open_type:
            1 isolated
            2 cross
        """
        normalized_symbol = self._normalize_symbol(symbol)

        body: dict[str, Any] = {
            "symbol": normalized_symbol,
            "price": self._format_number(price),
            "vol": self._format_number(vol),
            "side": side,
            "type": order_type,
            "openType": open_type,
        }

        if leverage is not None:
            body["leverage"] = leverage
        if external_oid is not None:
            body["externalOid"] = external_oid
        if stop_loss_price is not None:
            body["stopLossPrice"] = self._format_number(stop_loss_price)
        if take_profit_price is not None:
            body["takeProfitPrice"] = self._format_number(take_profit_price)
        if position_id is not None:
            body["positionId"] = position_id
        if reduce_only is not None:
            body["reduceOnly"] = reduce_only

        payload = await self._request(
            method="POST",
            path="/api/v1/private/order/submit",
            body=body,
            signed=True,
            auth_required=True,
        )

        await self._emit_event(
            "execution.order_submitted",
            {
                "exchange": "mexc",
                "symbol": normalized_symbol,
                "side": side,
                "order_type": order_type,
                "price": price,
                "vol": vol,
            },
            priority=EventPriority.CRITICAL,
        )
        return payload

    async def cancel_order(
        self,
        *,
        order_id: str,
    ) -> Any:
        payload = await self._request(
            method="POST",
            path="/api/v1/private/order/cancel",
            body={"orderId": order_id},
            signed=True,
            auth_required=True,
        )

        await self._emit_event(
            "execution.order_cancelled",
            {
                "exchange": "mexc",
                "order_id": order_id,
            },
            priority=EventPriority.CRITICAL,
        )
        return payload

    async def cancel_orders(
        self,
        *,
        symbol: str,
    ) -> Any:
        normalized_symbol = self._normalize_symbol(symbol)

        payload = await self._request(
            method="POST",
            path="/api/v1/private/order/cancel_all",
            body={"symbol": normalized_symbol},
            signed=True,
            auth_required=True,
        )

        await self._emit_event(
            "execution.orders_cancelled",
            {
                "exchange": "mexc",
                "symbol": normalized_symbol,
            },
            priority=EventPriority.CRITICAL,
        )
        return payload

    async def change_leverage(
        self,
        *,
        symbol: str,
        leverage: int,
        position_type: int = 1,
    ) -> Any:
        normalized_symbol = self._normalize_symbol(symbol)

        payload = await self._request(
            method="POST",
            path="/api/v1/private/position/change_leverage",
            body={
                "symbol": normalized_symbol,
                "leverage": leverage,
                "positionType": position_type,
            },
            signed=True,
            auth_required=True,
        )

        await self._emit_event(
            "position.leverage_changed",
            {
                "exchange": "mexc",
                "symbol": normalized_symbol,
                "leverage": leverage,
                "position_type": position_type,
            },
            priority=EventPriority.HIGH,
        )
        return payload

    async def change_position_mode(
        self,
        *,
        symbol: str,
        position_mode: int,
    ) -> Any:
        normalized_symbol = self._normalize_symbol(symbol)

        payload = await self._request(
            method="POST",
            path="/api/v1/private/position/change_position_mode",
            body={
                "symbol": normalized_symbol,
                "positionMode": position_mode,
            },
            signed=True,
            auth_required=True,
        )

        await self._emit_event(
            "position.mode_changed",
            {
                "exchange": "mexc",
                "symbol": normalized_symbol,
                "position_mode": position_mode,
            },
            priority=EventPriority.HIGH,
        )
        return payload

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
        auth_required: bool = False,
    ) -> Any:
        if self._session is None or self._session.closed:
            await self.start()

        assert self._session is not None

        method = method.upper()
        params = dict(params or {})
        body = dict(body or {})
        headers: dict[str, str] = {}

        url = f"{self._rest_url}{path}"
        request_kwargs: dict[str, Any] = {}

        if signed or auth_required:
            if self._time_offset_ms == 0:
                await self.sync_time()

            req_time = str(self._current_timestamp_ms() + self._time_offset_ms)

            if method == "GET":
                param_string = self._build_query_string(params)
                request_kwargs["params"] = params
            else:
                param_string = self._build_query_string(body)
                headers["Content-Type"] = "application/json"
                request_kwargs["json"] = body

            signature = self._sign(
                api_key=self._api_key,
                req_time=req_time,
                param_string=param_string,
            )

            headers.update(
                {
                    "ApiKey": self._api_key,
                    "Request-Time": req_time,
                    "Signature": signature,
                    "Recv-Window": str(self._recv_window_ms),
                }
            )
        else:
            if method == "GET":
                request_kwargs["params"] = params
            else:
                headers["Content-Type"] = "application/json"
                request_kwargs["json"] = body

        self._logger.debug(
            "Sending MEXC REST request | method=%s path=%s signed=%s auth_required=%s",
            method,
            path,
            signed,
            auth_required,
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
                        "MEXC REST HTTP error | method=%s path=%s status=%s",
                        method,
                        path,
                        response.status,
                    )

                    await self._emit_event(
                        "system.rest.error",
                        {
                            "exchange": "mexc",
                            "method": method,
                            "path": path,
                            "status": response.status,
                            "response": response_text,
                        },
                        priority=EventPriority.HIGH,
                    )

                    raise RuntimeError(
                        f"MEXC REST HTTP error | method={method} path={path} "
                        f"status={response.status} response={response_text}"
                    )

                payload = await response.json()

                success = payload.get("success")
                code = payload.get("code")

                if success is False:
                    self._logger.error(
                        "MEXC REST business error | method=%s path=%s code=%s",
                        method,
                        path,
                        code,
                    )

                    await self._emit_event(
                        "system.rest.error",
                        {
                            "exchange": "mexc",
                            "method": method,
                            "path": path,
                            "code": code,
                            "message": payload.get("message"),
                        },
                        priority=EventPriority.HIGH,
                    )

                    raise RuntimeError(
                        f"MEXC REST business error | method={method} path={path} "
                        f"code={code} message={payload.get('message')}"
                    )

                await self._emit_event(
                    "system.rest.success",
                    {
                        "exchange": "mexc",
                        "method": method,
                        "path": path,
                        "code": code,
                    },
                    priority=EventPriority.LOW,
                )

                return payload

        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception(
                "MEXC REST request exception | method=%s path=%s",
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
                source="mexc_rest",
            )
        except Exception:
            self._logger.exception(
                "Failed to emit REST event | topic=%s",
                topic,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _sign(
        self,
        *,
        api_key: str,
        req_time: str,
        param_string: str,
    ) -> str:
        payload = f"{api_key}{req_time}{param_string}"
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
        return urlencode(filtered, doseq=True)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_number(value: float) -> str:
        formatted = format(value, "f")
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_side_from_int(value: Any) -> Optional[str]:
        try:
            side = int(value)
        except (TypeError, ValueError):
            return None

        if side == 1:
            return "buy"
        if side == 2:
            return "sell"
        return None

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
            if ts < 10_000_000_000:
                return ts * 1000
            return ts
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _current_timestamp_ms() -> int:
        return int(time.time() * 1000)