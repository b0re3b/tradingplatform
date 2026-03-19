from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from typing import Any, Optional
from urllib.parse import urlencode

import aiohttp

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger


class OkxRestClient:
    """
    OKX REST client (API v5).

    Підтримує:
    - public market endpoints
    - private account / position / order endpoints
    - HMAC signing
    - demo trading mode
    - інтеграцію з EventBus
    """

    DEFAULT_REST_URL = "https://www.okx.com"

    API_KEY_PLACEHOLDER = "OKX_API_KEY_PLACEHOLDER"
    API_SECRET_PLACEHOLDER = "OKX_API_SECRET_PLACEHOLDER"
    PASSPHRASE_PLACEHOLDER = "OKX_PASSPHRASE_PLACEHOLDER"

    def __init__(
        self,
        *,
        config: Config,
        event_bus: Optional[EventBus] = None,
        use_demo: bool = False,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._use_demo = use_demo

        self._logger = get_logger(
            __name__,
            exchange="okx",
            event_type="okx_rest",
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
        self._passphrase = (
            config.exchange.credentials.passphrase
            or self.PASSPHRASE_PLACEHOLDER
        )

        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            self._logger.warning("OKX REST client already started")
            return

        timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)

        self._logger.info(
            "OKX REST client started | base_url=%s demo=%s",
            self._rest_url,
            self._use_demo,
        )

    async def stop(self) -> None:
        if self._session is None:
            self._logger.warning("OKX REST client already stopped")
            return

        await self._session.close()
        self._session = None

        self._logger.info("OKX REST client stopped")

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    async def get_server_time(self) -> dict[str, Any]:
        return await self._request(
            method="GET",
            path="/api/v5/public/time",
            signed=False,
        )

    async def get_instruments(
        self,
        *,
        inst_type: str,
        uly: Optional[str] = None,
        inst_family: Optional[str] = None,
        inst_id: Optional[str] = None,
    ) -> Any:
        params: dict[str, Any] = {
            "instType": inst_type,
        }
        if uly is not None:
            params["uly"] = uly
        if inst_family is not None:
            params["instFamily"] = inst_family
        if inst_id is not None:
            params["instId"] = inst_id

        payload = await self._request(
            method="GET",
            path="/api/v5/public/instruments",
            params=params,
            signed=False,
        )

        await self._emit_event(
            "market.instruments.info",
            {
                "exchange": "okx",
                "inst_type": inst_type,
                "inst_id": inst_id,
            },
            priority=EventPriority.LOW,
        )
        return payload

    async def get_orderbook(
        self,
        *,
        inst_id: str,
        depth: Optional[int] = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "instId": inst_id.upper(),
        }
        if depth is not None:
            params["sz"] = depth

        payload = await self._request(
            method="GET",
            path="/api/v5/market/books",
            params=params,
            signed=False,
        )

        data = (payload.get("data") or [{}])[0]

        normalized = {
            "exchange": "okx",
            "inst_id": inst_id.upper(),
            "asks": [self._normalize_book_level(level) for level in data.get("asks", [])],
            "bids": [self._normalize_book_level(level) for level in data.get("bids", [])],
            "timestamp": self._safe_int(data.get("ts")),
        }

        await self._emit_event(
            "market.orderbook.snapshot",
            normalized,
            priority=EventPriority.NORMAL,
        )
        return normalized

    async def get_trades(
        self,
        *,
        inst_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        payload = await self._request(
            method="GET",
            path="/api/v5/market/trades",
            params={
                "instId": inst_id.upper(),
                "limit": limit,
            },
            signed=False,
        )

        items = payload.get("data", [])
        normalized = [
            {
                "exchange": "okx",
                "inst_id": item.get("instId"),
                "trade_id": item.get("tradeId"),
                "price": self._safe_float(item.get("px")),
                "qty": self._safe_float(item.get("sz")),
                "side": (item.get("side") or "").lower(),
                "trade_time": self._safe_int(item.get("ts")),
            }
            for item in items
        ]

        await self._emit_event(
            "market.trades.snapshot",
            {
                "exchange": "okx",
                "inst_id": inst_id.upper(),
                "count": len(normalized),
            },
            priority=EventPriority.LOW,
        )
        return normalized

    async def get_candles(
        self,
        *,
        inst_id: str,
        bar: str = "1m",
        limit: int = 100,
        before: Optional[str] = None,
        after: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "instId": inst_id.upper(),
            "bar": bar,
            "limit": limit,
        }
        if before is not None:
            params["before"] = before
        if after is not None:
            params["after"] = after

        payload = await self._request(
            method="GET",
            path="/api/v5/market/candles",
            params=params,
            signed=False,
        )

        items = payload.get("data", [])
        normalized = []

        for candle in items:
            if len(candle) < 9:
                continue

            normalized.append(
                {
                    "exchange": "okx",
                    "inst_id": inst_id.upper(),
                    "bar": bar,
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
            )

        await self._emit_event(
            "market.candles.snapshot",
            {
                "exchange": "okx",
                "inst_id": inst_id.upper(),
                "bar": bar,
                "count": len(normalized),
            },
            priority=EventPriority.LOW,
        )
        return normalized

    async def get_tickers(
        self,
        *,
        inst_type: str,
        uly: Optional[str] = None,
        inst_family: Optional[str] = None,
    ) -> Any:
        params: dict[str, Any] = {
            "instType": inst_type,
        }
        if uly is not None:
            params["uly"] = uly
        if inst_family is not None:
            params["instFamily"] = inst_family

        payload = await self._request(
            method="GET",
            path="/api/v5/market/tickers",
            params=params,
            signed=False,
        )

        await self._emit_event(
            "market.tickers.snapshot",
            {
                "exchange": "okx",
                "inst_type": inst_type,
            },
            priority=EventPriority.LOW,
        )
        return payload

    async def get_funding_rate(
        self,
        *,
        inst_id: str,
    ) -> Any:
        payload = await self._request(
            method="GET",
            path="/api/v5/public/funding-rate",
            params={"instId": inst_id.upper()},
            signed=False,
        )

        await self._emit_event(
            "market.funding.snapshot",
            {
                "exchange": "okx",
                "inst_id": inst_id.upper(),
            },
            priority=EventPriority.NORMAL,
        )
        return payload

    # ------------------------------------------------------------------
    # Private account endpoints
    # ------------------------------------------------------------------

    async def get_balance(
        self,
        *,
        ccy: Optional[str] = None,
    ) -> Any:
        params: dict[str, Any] = {}
        if ccy is not None:
            params["ccy"] = ccy.upper()

        payload = await self._request(
            method="GET",
            path="/api/v5/account/balance",
            params=params,
            signed=True,
        )

        await self._emit_event(
            "account.balance.snapshot",
            {
                "exchange": "okx",
                "ccy": ccy.upper() if ccy else None,
            },
            priority=EventPriority.HIGH,
        )
        return payload

    async def get_positions(
        self,
        *,
        inst_type: Optional[str] = None,
        inst_id: Optional[str] = None,
        pos_id: Optional[str] = None,
    ) -> Any:
        params: dict[str, Any] = {}
        if inst_type is not None:
            params["instType"] = inst_type
        if inst_id is not None:
            params["instId"] = inst_id.upper()
        if pos_id is not None:
            params["posId"] = pos_id

        payload = await self._request(
            method="GET",
            path="/api/v5/account/positions",
            params=params,
            signed=True,
        )

        await self._emit_event(
            "position.snapshot",
            {
                "exchange": "okx",
                "inst_id": inst_id.upper() if inst_id else None,
            },
            priority=EventPriority.HIGH,
        )
        return payload

    # ------------------------------------------------------------------
    # Private trading endpoints
    # ------------------------------------------------------------------

    async def get_open_orders(
        self,
        *,
        inst_type: Optional[str] = None,
        uly: Optional[str] = None,
        inst_family: Optional[str] = None,
        inst_id: Optional[str] = None,
        ord_type: Optional[str] = None,
        state: Optional[str] = None,
    ) -> Any:
        params: dict[str, Any] = {}
        if inst_type is not None:
            params["instType"] = inst_type
        if uly is not None:
            params["uly"] = uly
        if inst_family is not None:
            params["instFamily"] = inst_family
        if inst_id is not None:
            params["instId"] = inst_id.upper()
        if ord_type is not None:
            params["ordType"] = ord_type
        if state is not None:
            params["state"] = state

        payload = await self._request(
            method="GET",
            path="/api/v5/trade/orders-pending",
            params=params,
            signed=True,
        )

        await self._emit_event(
            "execution.open_orders.snapshot",
            {
                "exchange": "okx",
                "inst_id": inst_id.upper() if inst_id else None,
            },
            priority=EventPriority.HIGH,
        )
        return payload

    async def get_order_history(
        self,
        *,
        inst_type: str,
        uly: Optional[str] = None,
        inst_family: Optional[str] = None,
        inst_id: Optional[str] = None,
        state: Optional[str] = None,
        before: Optional[str] = None,
        after: Optional[str] = None,
        limit: int = 100,
    ) -> Any:
        params: dict[str, Any] = {
            "instType": inst_type,
            "limit": limit,
        }
        if uly is not None:
            params["uly"] = uly
        if inst_family is not None:
            params["instFamily"] = inst_family
        if inst_id is not None:
            params["instId"] = inst_id.upper()
        if state is not None:
            params["state"] = state
        if before is not None:
            params["before"] = before
        if after is not None:
            params["after"] = after

        payload = await self._request(
            method="GET",
            path="/api/v5/trade/orders-history",
            params=params,
            signed=True,
        )

        await self._emit_event(
            "execution.order_history.snapshot",
            {
                "exchange": "okx",
                "inst_id": inst_id.upper() if inst_id else None,
            },
            priority=EventPriority.NORMAL,
        )
        return payload

    async def get_order(
        self,
        *,
        inst_id: str,
        ord_id: Optional[str] = None,
        cl_ord_id: Optional[str] = None,
    ) -> Any:
        if ord_id is None and cl_ord_id is None:
            raise ValueError("Either ord_id or cl_ord_id must be provided")

        params: dict[str, Any] = {
            "instId": inst_id.upper(),
        }
        if ord_id is not None:
            params["ordId"] = ord_id
        if cl_ord_id is not None:
            params["clOrdId"] = cl_ord_id

        payload = await self._request(
            method="GET",
            path="/api/v5/trade/order",
            params=params,
            signed=True,
        )

        await self._emit_event(
            "execution.order.fetched",
            {
                "exchange": "okx",
                "inst_id": inst_id.upper(),
                "ord_id": ord_id,
                "cl_ord_id": cl_ord_id,
            },
            priority=EventPriority.HIGH,
        )
        return payload

    async def place_order(
        self,
        *,
        inst_id: str,
        td_mode: str,
        side: str,
        ord_type: str,
        sz: str,
        ccy: Optional[str] = None,
        cl_ord_id: Optional[str] = None,
        tag: Optional[str] = None,
        pos_side: Optional[str] = None,
        px: Optional[str] = None,
        reduce_only: Optional[bool] = None,
    ) -> Any:
        body: dict[str, Any] = {
            "instId": inst_id.upper(),
            "tdMode": td_mode,
            "side": side,
            "ordType": ord_type,
            "sz": sz,
        }
        if ccy is not None:
            body["ccy"] = ccy
        if cl_ord_id is not None:
            body["clOrdId"] = cl_ord_id
        if tag is not None:
            body["tag"] = tag
        if pos_side is not None:
            body["posSide"] = pos_side
        if px is not None:
            body["px"] = px
        if reduce_only is not None:
            body["reduceOnly"] = reduce_only

        payload = await self._request(
            method="POST",
            path="/api/v5/trade/order",
            body=body,
            signed=True,
        )

        await self._emit_event(
            "execution.order_submitted",
            {
                "exchange": "okx",
                "inst_id": inst_id.upper(),
                "side": side,
                "ord_type": ord_type,
                "sz": sz,
                "px": px,
            },
            priority=EventPriority.CRITICAL,
        )
        return payload

    async def cancel_order(
        self,
        *,
        inst_id: str,
        ord_id: Optional[str] = None,
        cl_ord_id: Optional[str] = None,
    ) -> Any:
        if ord_id is None and cl_ord_id is None:
            raise ValueError("Either ord_id or cl_ord_id must be provided")

        body: dict[str, Any] = {
            "instId": inst_id.upper(),
        }
        if ord_id is not None:
            body["ordId"] = ord_id
        if cl_ord_id is not None:
            body["clOrdId"] = cl_ord_id

        payload = await self._request(
            method="POST",
            path="/api/v5/trade/cancel-order",
            body=body,
            signed=True,
        )

        await self._emit_event(
            "execution.order_cancelled",
            {
                "exchange": "okx",
                "inst_id": inst_id.upper(),
                "ord_id": ord_id,
                "cl_ord_id": cl_ord_id,
            },
            priority=EventPriority.CRITICAL,
        )
        return payload

    async def amend_order(
        self,
        *,
        inst_id: str,
        ord_id: Optional[str] = None,
        cl_ord_id: Optional[str] = None,
        new_sz: Optional[str] = None,
        new_px: Optional[str] = None,
        cxl_on_fail: Optional[bool] = None,
    ) -> Any:
        if ord_id is None and cl_ord_id is None:
            raise ValueError("Either ord_id or cl_ord_id must be provided")

        body: dict[str, Any] = {
            "instId": inst_id.upper(),
        }
        if ord_id is not None:
            body["ordId"] = ord_id
        if cl_ord_id is not None:
            body["clOrdId"] = cl_ord_id
        if new_sz is not None:
            body["newSz"] = new_sz
        if new_px is not None:
            body["newPx"] = new_px
        if cxl_on_fail is not None:
            body["cxlOnFail"] = cxl_on_fail

        payload = await self._request(
            method="POST",
            path="/api/v5/trade/amend-order",
            body=body,
            signed=True,
        )

        await self._emit_event(
            "execution.order_amended",
            {
                "exchange": "okx",
                "inst_id": inst_id.upper(),
                "ord_id": ord_id,
                "cl_ord_id": cl_ord_id,
            },
            priority=EventPriority.CRITICAL,
        )
        return payload

    async def get_fills(
        self,
        *,
        inst_type: Optional[str] = None,
        uly: Optional[str] = None,
        inst_family: Optional[str] = None,
        inst_id: Optional[str] = None,
        ord_id: Optional[str] = None,
        before: Optional[str] = None,
        after: Optional[str] = None,
        limit: int = 100,
    ) -> Any:
        params: dict[str, Any] = {
            "limit": limit,
        }
        if inst_type is not None:
            params["instType"] = inst_type
        if uly is not None:
            params["uly"] = uly
        if inst_family is not None:
            params["instFamily"] = inst_family
        if inst_id is not None:
            params["instId"] = inst_id.upper()
        if ord_id is not None:
            params["ordId"] = ord_id
        if before is not None:
            params["before"] = before
        if after is not None:
            params["after"] = after

        payload = await self._request(
            method="GET",
            path="/api/v5/trade/fills",
            params=params,
            signed=True,
        )

        await self._emit_event(
            "execution.fills.snapshot",
            {
                "exchange": "okx",
                "inst_id": inst_id.upper() if inst_id else None,
            },
            priority=EventPriority.NORMAL,
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
    ) -> Any:
        if self._session is None or self._session.closed:
            await self.start()

        assert self._session is not None

        method = method.upper()
        params = dict(params or {})
        body = dict(body or {})
        headers: dict[str, str] = {}

        query_string = self._build_query_string(params)
        request_path = path
        if query_string:
            request_path = f"{path}?{query_string}"

        url = f"{self._rest_url}{request_path}"

        if method == "GET":
            request_kwargs: dict[str, Any] = {}
        else:
            request_kwargs = {"json": body}
            headers["Content-Type"] = "application/json"

        if signed:
            timestamp = self._timestamp_iso()
            body_text = "" if method == "GET" else json.dumps(body, separators=(",", ":"))
            sign = self._sign(
                timestamp=timestamp,
                method=method,
                request_path=request_path,
                body_text=body_text,
            )

            headers.update(
                {
                    "OK-ACCESS-KEY": self._api_key,
                    "OK-ACCESS-SIGN": sign,
                    "OK-ACCESS-TIMESTAMP": timestamp,
                    "OK-ACCESS-PASSPHRASE": self._passphrase,
                }
            )

            if self._use_demo:
                headers["x-simulated-trading"] = "1"

        self._logger.debug(
            "Sending OKX REST request | method=%s path=%s signed=%s demo=%s",
            method,
            path,
            signed,
            self._use_demo,
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
                        "OKX REST HTTP error | method=%s path=%s status=%s",
                        method,
                        path,
                        response.status,
                    )

                    await self._emit_event(
                        "system.rest.error",
                        {
                            "exchange": "okx",
                            "method": method,
                            "path": path,
                            "status": response.status,
                            "response": response_text,
                        },
                        priority=EventPriority.HIGH,
                    )

                    raise RuntimeError(
                        f"OKX REST HTTP error | method={method} path={path} "
                        f"status={response.status} response={response_text}"
                    )

                payload = await response.json()

                code = payload.get("code")
                if code not in (None, "0", 0):
                    self._logger.error(
                        "OKX REST business error | method=%s path=%s code=%s msg=%s",
                        method,
                        path,
                        code,
                        payload.get("msg"),
                    )

                    await self._emit_event(
                        "system.rest.error",
                        {
                            "exchange": "okx",
                            "method": method,
                            "path": path,
                            "code": code,
                            "msg": payload.get("msg"),
                        },
                        priority=EventPriority.HIGH,
                    )

                    raise RuntimeError(
                        f"OKX REST business error | method={method} path={path} "
                        f"code={code} msg={payload.get('msg')}"
                    )

                await self._emit_event(
                    "system.rest.success",
                    {
                        "exchange": "okx",
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
                "OKX REST request exception | method=%s path=%s",
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
                source="okx_rest",
            )
        except Exception:
            self._logger.exception("Failed to emit REST event | topic=%s", topic)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sign(
        self,
        *,
        timestamp: str,
        method: str,
        request_path: str,
        body_text: str,
    ) -> str:
        payload = f"{timestamp}{method.upper()}{request_path}{body_text}"
        digest = hmac.new(
            self._api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode("utf-8")

    @staticmethod
    def _timestamp_iso() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

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
    def _safe_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_book_level(level: list[Any]) -> list[Optional[float]]:
        normalized: list[Optional[float]] = []
        for item in level:
            try:
                normalized.append(float(item))
            except (TypeError, ValueError):
                normalized.append(None)
        return normalized