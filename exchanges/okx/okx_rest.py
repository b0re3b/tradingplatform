from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from data.market_ingestion import MarketIngestionService


@dataclass(slots=True)
class OkxRestClientConfig:
    """
    Local OKX REST adapter config.

    Global exchange credentials/settings still come from core.config.Config.exchange.
    This dataclass contains only OKX REST adapter-specific knobs.
    """

    rest_url: str = "https://www.okx.com"
    timeout_seconds: float = 10.0
    use_demo: bool = False

    request_retries: int = 2
    retry_delay_seconds: float = 0.25

    emit_success_events: bool = False
    emit_error_events: bool = True

    @classmethod
    def from_core_config(
            cls,
            *,
            config: Config,
            use_demo: bool = False,
    ) -> "OkxRestClientConfig":
        defaults = cls()
        return cls(
            rest_url=defaults.rest_url,
            timeout_seconds=config.exchange.timeout_seconds,
            use_demo=use_demo or config.exchange.credentials.testnet,
        )


class OkxRestClient:
    """
    OKX REST exchange adapter.

    Responsibilities:
    - perform OKX REST API v5 requests;
    - normalize OKX payloads into internal market/exchange formats;
    - publish events through EventBus;
    - never call analytics, strategy, risk, or execution directly;
    - never contain trading decision logic.

    Public market events:
    - market.orderbook.snapshot
    - market.trades.snapshot
    - market.candles.snapshot
    - market.tickers.snapshot
    - market.funding.snapshot

    Private exchange events:
    - exchange.account.balance.snapshot
    - exchange.positions.snapshot
    - exchange.open_orders.snapshot
    - exchange.order.fetched
    - exchange.order.submitted
    - exchange.order.cancelled
    - exchange.order.amended
    - exchange.fills.snapshot

    Execution layer should listen to exchange.* order events and publish
    execution.* domain events itself.
    """

    EXCHANGE = "okx"
    SOURCE = "okx_rest"

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        rest_config: OkxRestClientConfig | None = None,
        use_demo: bool = False,
        market_ingestion: MarketIngestionService | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._market_ingestion = market_ingestion
        self._rest_config = rest_config or OkxRestClientConfig.from_core_config(
            config=config,
            use_demo=use_demo,
        )

        self._logger = get_logger(
            __name__,
            exchange=self.EXCHANGE,
            event_type="exchange_rest",
        )

        self._api_key = config.exchange.credentials.api_key
        self._api_secret = config.exchange.credentials.api_secret
        self._passphrase = config.exchange.credentials.passphrase

        self._session: aiohttp.ClientSession | None = None
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            self._logger.warning("OKX REST client already started")
            self._started = True
            return

        timeout = aiohttp.ClientTimeout(total=self._rest_config.timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._started = True

        self._logger.info(
            "OKX REST client started | base_url=%s demo=%s timeout=%s",
            self._rest_config.rest_url,
            self._rest_config.use_demo,
            self._rest_config.timeout_seconds,
        )

        await self._emit_event(
            "system.exchange.rest.started",
            {
                "exchange": self.EXCHANGE,
                "rest_url": self._rest_config.rest_url,
                "demo": self._rest_config.use_demo,
            },
            priority=EventPriority.LOW,
        )

    async def stop(self) -> None:
        if self._session is None:
            self._logger.warning("OKX REST client already stopped")
            self._started = False
            return

        await self._session.close()
        self._session = None
        self._started = False

        self._logger.info("OKX REST client stopped")

        await self._emit_event(
            "system.exchange.rest.stopped",
            {
                "exchange": self.EXCHANGE,
            },
            priority=EventPriority.LOW,
        )

    def register(self) -> None:
        """
        REST adapter currently does not subscribe to EventBus topics.

        Kept for project-wide consistency with modules that expose register().
        """
        self._logger.debug("OKX REST client register called | subscriptions=0")

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
        uly: str | None = None,
        inst_family: str | None = None,
        inst_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "instType": inst_type,
        }

        if uly is not None:
            params["uly"] = uly
        if inst_family is not None:
            params["instFamily"] = inst_family
        if inst_id is not None:
            params["instId"] = inst_id.upper()

        payload = await self._request(
            method="GET",
            path="/api/v5/public/instruments",
            params=params,
            signed=False,
        )

        await self._emit_event(
            "market.instruments.info",
            {
                "exchange": self.EXCHANGE,
                "inst_type": inst_type,
                "inst_id": inst_id.upper() if inst_id else None,
                "count": len(payload.get("data", [])),
            },
            priority=EventPriority.LOW,
        )

        return payload

    async def get_orderbook(
        self,
        *,
        inst_id: str,
        depth: int | None = None,
    ) -> dict[str, Any]:
        inst_id = inst_id.upper()

        params: dict[str, Any] = {
            "instId": inst_id,
        }

        if depth is not None:
            params["sz"] = depth

        payload = await self._request(
            method="GET",
            path="/api/v5/market/books",
            params=params,
            signed=False,
        )

        data = self._first_data_item(payload)

        normalized = {
            "exchange": self.EXCHANGE,
            "symbol": inst_id,
            "inst_id": inst_id,
            "asks": [
                self._normalize_book_level(level)
                for level in data.get("asks", [])
            ],
            "bids": [
                self._normalize_book_level(level)
                for level in data.get("bids", [])
            ],
            "timestamp": self._safe_int(data.get("ts")),
            "snapshot_time": self._current_timestamp_ms(),
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
        inst_id = inst_id.upper()

        payload = await self._request(
            method="GET",
            path="/api/v5/market/trades",
            params={
                "instId": inst_id,
                "limit": limit,
            },
            signed=False,
        )

        normalized = [
            self._normalize_trade(item)
            for item in payload.get("data", [])
        ]

        await self._emit_event(
            "market.trades.snapshot",
            {
                "exchange": self.EXCHANGE,
                "symbol": inst_id,
                "inst_id": inst_id,
                "count": len(normalized),
                "trades": normalized,
                "snapshot_time": self._current_timestamp_ms(),
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
        before: str | None = None,
        after: str | None = None,
    ) -> list[dict[str, Any]]:
        inst_id = inst_id.upper()

        params: dict[str, Any] = {
            "instId": inst_id,
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

        normalized = [
            self._normalize_candle(
                item=candle,
                inst_id=inst_id,
                timeframe=bar,
            )
            for candle in payload.get("data", [])
            if len(candle) >= 9
        ]

        await self._emit_event(
            "market.candles.snapshot",
            {
                "exchange": self.EXCHANGE,
                "symbol": inst_id,
                "inst_id": inst_id,
                "timeframe": bar,
                "count": len(normalized),
                "candles": normalized,
                "snapshot_time": self._current_timestamp_ms(),
            },
            priority=EventPriority.LOW,
        )

        return normalized

    async def get_tickers(
        self,
        *,
        inst_type: str,
        uly: str | None = None,
        inst_family: str | None = None,
    ) -> dict[str, Any]:
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
                "exchange": self.EXCHANGE,
                "inst_type": inst_type,
                "count": len(payload.get("data", [])),
                "snapshot_time": self._current_timestamp_ms(),
            },
            priority=EventPriority.LOW,
        )

        return payload

    async def get_funding_rate(
        self,
        *,
        inst_id: str,
    ) -> dict[str, Any]:
        inst_id = inst_id.upper()

        payload = await self._request(
            method="GET",
            path="/api/v5/public/funding-rate",
            params={"instId": inst_id},
            signed=False,
        )

        data = self._first_data_item(payload)

        normalized = {
            "exchange": self.EXCHANGE,
            "symbol": inst_id,
            "inst_id": inst_id,
            "funding_rate": self._safe_float(data.get("fundingRate")),
            "next_funding_rate": self._safe_float(data.get("nextFundingRate")),
            "funding_time": self._safe_int(data.get("fundingTime")),
            "next_funding_time": self._safe_int(data.get("nextFundingTime")),
            "method": data.get("method"),
            "max_funding_rate": self._safe_float(data.get("maxFundingRate")),
            "min_funding_rate": self._safe_float(data.get("minFundingRate")),
            "sett_state": data.get("settState"),
            "timestamp": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "market.funding.snapshot",
            normalized,
            priority=EventPriority.NORMAL,
        )

        return normalized

    # ------------------------------------------------------------------
    # Private account endpoints
    # ------------------------------------------------------------------

    async def get_balance(
        self,
        *,
        ccy: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}

        if ccy is not None:
            params["ccy"] = ccy.upper()

        payload = await self._request(
            method="GET",
            path="/api/v5/account/balance",
            params=params,
            signed=True,
        )

        normalized = {
            "exchange": self.EXCHANGE,
            "ccy": ccy.upper() if ccy else None,
            "data": payload.get("data", []),
            "snapshot_time": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.account.balance.snapshot",
            normalized,
            priority=EventPriority.HIGH,
        )

        return normalized

    async def get_positions(
        self,
        *,
        inst_type: str | None = None,
        inst_id: str | None = None,
        pos_id: str | None = None,
    ) -> dict[str, Any]:
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

        normalized = {
            "exchange": self.EXCHANGE,
            "inst_type": inst_type,
            "symbol": inst_id.upper() if inst_id else None,
            "inst_id": inst_id.upper() if inst_id else None,
            "pos_id": pos_id,
            "positions": payload.get("data", []),
            "snapshot_time": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.positions.snapshot",
            normalized,
            priority=EventPriority.HIGH,
        )

        return normalized

    # ------------------------------------------------------------------
    # Private trading endpoints
    # ------------------------------------------------------------------

    async def get_open_orders(
        self,
        *,
        inst_type: str | None = None,
        uly: str | None = None,
        inst_family: str | None = None,
        inst_id: str | None = None,
        ord_type: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
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

        normalized = {
            "exchange": self.EXCHANGE,
            "inst_type": inst_type,
            "symbol": inst_id.upper() if inst_id else None,
            "inst_id": inst_id.upper() if inst_id else None,
            "orders": payload.get("data", []),
            "count": len(payload.get("data", [])),
            "snapshot_time": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.open_orders.snapshot",
            normalized,
            priority=EventPriority.HIGH,
        )

        return normalized

    async def get_order_history(
        self,
        *,
        inst_type: str,
        uly: str | None = None,
        inst_family: str | None = None,
        inst_id: str | None = None,
        state: str | None = None,
        before: str | None = None,
        after: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
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

        normalized = {
            "exchange": self.EXCHANGE,
            "inst_type": inst_type,
            "symbol": inst_id.upper() if inst_id else None,
            "inst_id": inst_id.upper() if inst_id else None,
            "orders": payload.get("data", []),
            "count": len(payload.get("data", [])),
            "snapshot_time": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.order_history.snapshot",
            normalized,
            priority=EventPriority.NORMAL,
        )

        return normalized

    async def get_order(
        self,
        *,
        inst_id: str,
        ord_id: str | None = None,
        cl_ord_id: str | None = None,
    ) -> dict[str, Any]:
        if ord_id is None and cl_ord_id is None:
            raise ValueError("Either ord_id or cl_ord_id must be provided")

        inst_id = inst_id.upper()

        params: dict[str, Any] = {
            "instId": inst_id,
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

        normalized = {
            "exchange": self.EXCHANGE,
            "symbol": inst_id,
            "inst_id": inst_id,
            "ord_id": ord_id,
            "cl_ord_id": cl_ord_id,
            "orders": payload.get("data", []),
            "snapshot_time": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.order.fetched",
            normalized,
            priority=EventPriority.HIGH,
        )

        return normalized

    async def place_order(
        self,
        *,
        inst_id: str,
        td_mode: str,
        side: str,
        ord_type: str,
        sz: str,
        ccy: str | None = None,
        cl_ord_id: str | None = None,
        tag: str | None = None,
        pos_side: str | None = None,
        px: str | None = None,
        reduce_only: bool | None = None,
    ) -> dict[str, Any]:
        """
        Low-level OKX order endpoint.

        This method must be called by execution/order_manager layer.
        It does not decide whether an order should be opened.
        """

        inst_id = inst_id.upper()

        body: dict[str, Any] = {
            "instId": inst_id,
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

        normalized = {
            "exchange": self.EXCHANGE,
            "symbol": inst_id,
            "inst_id": inst_id,
            "side": side,
            "ord_type": ord_type,
            "td_mode": td_mode,
            "sz": sz,
            "px": px,
            "cl_ord_id": cl_ord_id,
            "response": payload.get("data", []),
            "timestamp": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.order.submitted",
            normalized,
            priority=EventPriority.CRITICAL,
        )

        return normalized

    async def cancel_order(
        self,
        *,
        inst_id: str,
        ord_id: str | None = None,
        cl_ord_id: str | None = None,
    ) -> dict[str, Any]:
        if ord_id is None and cl_ord_id is None:
            raise ValueError("Either ord_id or cl_ord_id must be provided")

        inst_id = inst_id.upper()

        body: dict[str, Any] = {
            "instId": inst_id,
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

        normalized = {
            "exchange": self.EXCHANGE,
            "symbol": inst_id,
            "inst_id": inst_id,
            "ord_id": ord_id,
            "cl_ord_id": cl_ord_id,
            "response": payload.get("data", []),
            "timestamp": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.order.cancelled",
            normalized,
            priority=EventPriority.CRITICAL,
        )

        return normalized

    async def amend_order(
        self,
        *,
        inst_id: str,
        ord_id: str | None = None,
        cl_ord_id: str | None = None,
        new_sz: str | None = None,
        new_px: str | None = None,
        cxl_on_fail: bool | None = None,
    ) -> dict[str, Any]:
        if ord_id is None and cl_ord_id is None:
            raise ValueError("Either ord_id or cl_ord_id must be provided")

        inst_id = inst_id.upper()

        body: dict[str, Any] = {
            "instId": inst_id,
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

        normalized = {
            "exchange": self.EXCHANGE,
            "symbol": inst_id,
            "inst_id": inst_id,
            "ord_id": ord_id,
            "cl_ord_id": cl_ord_id,
            "new_sz": new_sz,
            "new_px": new_px,
            "response": payload.get("data", []),
            "timestamp": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.order.amended",
            normalized,
            priority=EventPriority.CRITICAL,
        )

        return normalized

    async def get_fills(
        self,
        *,
        inst_type: str | None = None,
        uly: str | None = None,
        inst_family: str | None = None,
        inst_id: str | None = None,
        ord_id: str | None = None,
        before: str | None = None,
        after: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
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

        normalized = {
            "exchange": self.EXCHANGE,
            "inst_type": inst_type,
            "symbol": inst_id.upper() if inst_id else None,
            "inst_id": inst_id.upper() if inst_id else None,
            "ord_id": ord_id,
            "fills": payload.get("data", []),
            "count": len(payload.get("data", [])),
            "snapshot_time": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "exchange.fills.snapshot",
            normalized,
            priority=EventPriority.NORMAL,
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
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> dict[str, Any]:
        await self._ensure_session()

        assert self._session is not None

        method = method.upper()
        params = dict(params or {})
        body = dict(body or {})

        query_string = self._build_query_string(params)
        request_path = f"{path}?{query_string}" if query_string else path
        url = f"{self._rest_config.rest_url}{request_path}"

        last_error: Exception | None = None

        for attempt in range(self._rest_config.request_retries + 1):
            try:
                headers, request_kwargs = self._build_request_context(
                    method=method,
                    request_path=request_path,
                    body=body,
                    signed=signed,
                )

                self._logger.debug(
                    "Sending OKX REST request | method=%s path=%s signed=%s demo=%s attempt=%s",
                    method,
                    path,
                    signed,
                    self._rest_config.use_demo,
                    attempt + 1,
                )

                async with self._session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    **request_kwargs,
                ) as response:
                    response_text = await response.text()

                    if response.status >= 400:
                        error_payload = self._parse_error_payload(response_text)

                        await self._handle_http_error(
                            method=method,
                            path=path,
                            status=response.status,
                            error_payload=error_payload,
                        )

                        raise RuntimeError(
                            f"OKX REST HTTP error | method={method} path={path} "
                            f"status={response.status} code={error_payload.get('code')}"
                        )

                    payload = await response.json()

                    code = payload.get("code")
                    if code not in (None, "0", 0):
                        await self._handle_business_error(
                            method=method,
                            path=path,
                            payload=payload,
                        )

                        raise RuntimeError(
                            f"OKX REST business error | method={method} path={path} "
                            f"code={code} msg={payload.get('msg')}"
                        )

                    if self._rest_config.emit_success_events:
                        await self._emit_event(
                            "system.exchange.rest.success",
                            {
                                "exchange": self.EXCHANGE,
                                "method": method,
                                "path": path,
                                "code": code,
                            },
                            priority=EventPriority.LOW,
                        )

                    return payload

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc

                if attempt >= self._rest_config.request_retries:
                    self._logger.exception(
                        "OKX REST request failed | method=%s path=%s attempts=%s",
                        method,
                        path,
                        attempt + 1,
                    )
                    raise

                self._logger.warning(
                    "OKX REST request retry scheduled | method=%s path=%s attempt=%s",
                    method,
                    path,
                    attempt + 1,
                )

                await asyncio.sleep(self._rest_config.retry_delay_seconds)

        if last_error is not None:
            raise last_error

        raise RuntimeError(f"OKX REST request failed unexpectedly | method={method} path={path}")

    def _build_request_context(
        self,
        *,
        method: str,
        request_path: str,
        body: dict[str, Any],
        signed: bool,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        headers: dict[str, str] = {}
        request_kwargs: dict[str, Any] = {}

        if method != "GET":
            request_kwargs["json"] = body
            headers["Content-Type"] = "application/json"

        if signed:
            self._require_credentials()

            assert self._api_key is not None
            assert self._api_secret is not None
            assert self._passphrase is not None

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

            if self._rest_config.use_demo:
                headers["x-simulated-trading"] = "1"

        return headers, request_kwargs

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            await self.start()

    async def _handle_http_error(
        self,
        *,
        method: str,
        path: str,
        status: int,
        error_payload: dict[str, Any],
    ) -> None:
        self._logger.error(
            "OKX REST HTTP error | method=%s path=%s status=%s code=%s",
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
            "OKX REST business error | method=%s path=%s code=%s msg=%s",
            method,
            path,
            payload.get("code"),
            payload.get("msg"),
        )

        if not self._rest_config.emit_error_events:
            return

        await self._emit_event(
            "system.exchange.rest.error",
            {
                "exchange": self.EXCHANGE,
                "method": method,
                "path": path,
                "code": payload.get("code"),
                "message": payload.get("msg"),
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
                "Failed to emit OKX REST event | topic=%s",
                topic,
            )

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _require_credentials(self) -> None:
        if not self._api_key or not self._api_secret or not self._passphrase:
            raise RuntimeError(
                "OKX API key, API secret and passphrase are required for this endpoint"
            )

    def _sign(
        self,
        *,
        timestamp: str,
        method: str,
        request_path: str,
        body_text: str,
    ) -> str:
        self._require_credentials()

        assert self._api_secret is not None

        payload = f"{timestamp}{method.upper()}{request_path}{body_text}"
        digest = hmac.new(
            self._api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        return base64.b64encode(digest).decode("utf-8")

    # ------------------------------------------------------------------
    # Normalizers
    # ------------------------------------------------------------------

    def _normalize_trade(self, item: dict[str, Any]) -> dict[str, Any]:
        inst_id = item.get("instId")

        return {
            "exchange": self.EXCHANGE,
            "symbol": inst_id,
            "inst_id": inst_id,
            "trade_id": item.get("tradeId"),
            "price": self._safe_float(item.get("px")),
            "qty": self._safe_float(item.get("sz")),
            "side": (item.get("side") or "").lower(),
            "trade_time": self._safe_int(item.get("ts")),
        }

    def _normalize_candle(
        self,
        *,
        item: list[Any],
        inst_id: str,
        timeframe: str,
    ) -> dict[str, Any]:
        confirm = str(item[8]) == "1"

        return {
            "exchange": self.EXCHANGE,
            "symbol": inst_id,
            "inst_id": inst_id,
            "timeframe": timeframe,
            "open_time": self._safe_int(item[0]),
            "close_time": None,
            "open": self._safe_float(item[1]),
            "high": self._safe_float(item[2]),
            "low": self._safe_float(item[3]),
            "close": self._safe_float(item[4]),
            "volume": self._safe_float(item[5]),
            "quote_volume": self._safe_float(item[7]),
            "volume_ccy": self._safe_float(item[6]),
            "trades_count": None,
            "is_closed": confirm,
            "confirm": item[8],
        }

    @staticmethod
    def _normalize_book_level(level: list[Any]) -> list[float | int | None]:
        """
        OKX book level format is usually:
        [price, size, liquidated_orders, order_count]
        """
        normalized: list[float | int | None] = []

        for index, item in enumerate(level):
            if index <= 1:
                normalized.append(OkxRestClient._safe_float(item))
            else:
                normalized.append(OkxRestClient._safe_int(item))

        return normalized

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_data_item(payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data")

        if isinstance(data, list) and data:
            item = data[0]
            return item if isinstance(item, dict) else {}

        return {}

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
    def _parse_error_payload(response_text: str) -> dict[str, Any]:
        try:
            raw = json.loads(response_text)

            return {
                "code": raw.get("code"),
                "message": raw.get("msg") or raw.get("message"),
            }
        except Exception:
            return {
                "code": None,
                "message": "Unable to parse OKX error response",
            }

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
    def _current_timestamp_ms() -> int:
        return int(time.time() * 1000)