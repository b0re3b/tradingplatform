from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger


@dataclass(slots=True)
class BybitMarketRestClientConfig:
    """
    Local Bybit public REST market-data adapter config.

    Bybit is currently used as a market-data source in our architecture.
    Trading/private endpoints should live in a separate future adapter only
    if Bybit becomes an execution exchange.
    """

    rest_url: str = "https://api.bybit.com"
    timeout_seconds: float = 10.0

    request_retries: int = 2
    retry_delay_seconds: float = 0.25

    emit_success_events: bool = False
    emit_error_events: bool = True

    @classmethod
    def from_core_config(cls, config: Config) -> "BybitMarketRestClientConfig":
        return cls(
            rest_url=config.exchange.rest_url or cls.rest_url,
            timeout_seconds=config.exchange.timeout_seconds,
        )


class BybitRestClient:
    """
    Bybit public REST market-data adapter.

    Responsibilities:
    - perform public Bybit REST API v5 requests;
    - normalize market data into internal payload contracts;
    - publish market.* and system.exchange.rest.* events through EventBus;
    - never call analytics, strategy, risk, or execution directly;
    - never contain trading decision logic.

    Public market events:
    - market.instruments.info
    - market.orderbook.snapshot
    - market.trades.snapshot
    - market.candles.snapshot
    - market.tickers.snapshot
    - market.funding.snapshot
    - market.open_interest.snapshot

    Note:
    - This class intentionally does not implement private/trading endpoints.
    - Binance is the execution exchange for now.
    """

    EXCHANGE = "bybit"
    SOURCE = "bybit_rest"

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        rest_config: BybitMarketRestClientConfig | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._rest_config = rest_config or BybitMarketRestClientConfig.from_core_config(config)

        self._logger = get_logger(
            __name__,
            exchange=self.EXCHANGE,
            event_type="exchange_market_rest",
        )

        self._session: aiohttp.ClientSession | None = None
        self._started = False
        self._time_offset_ms: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        REST market-data adapter currently does not subscribe to EventBus topics.

        Kept for project-wide consistency with modules that expose register().
        """
        self._logger.debug("Bybit market REST register called | subscriptions=0")

    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            self._logger.warning("Bybit market REST client already started")
            self._started = True
            return

        timeout = aiohttp.ClientTimeout(total=self._rest_config.timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._started = True

        self._logger.info(
            "Bybit market REST client started | base_url=%s timeout=%s",
            self._rest_config.rest_url,
            self._rest_config.timeout_seconds,
        )

        await self._emit_event(
            "system.exchange.rest.started",
            {
                "exchange": self.EXCHANGE,
                "rest_url": self._rest_config.rest_url,
                "mode": "market_data",
            },
            priority=EventPriority.LOW,
        )

    async def stop(self) -> None:
        if self._session is None:
            self._logger.warning("Bybit market REST client already stopped")
            self._started = False
            return

        await self._session.close()
        self._session = None
        self._started = False

        self._logger.info("Bybit market REST client stopped")

        await self._emit_event(
            "system.exchange.rest.stopped",
            {
                "exchange": self.EXCHANGE,
                "mode": "market_data",
            },
            priority=EventPriority.LOW,
        )

    # ------------------------------------------------------------------
    # Public market endpoints
    # ------------------------------------------------------------------

    async def get_server_time(self) -> dict[str, Any]:
        return await self._request(
            method="GET",
            path="/v5/market/time",
        )

    async def sync_time(self) -> int:
        local_before = self._current_timestamp_ms()
        payload = await self.get_server_time()
        local_after = self._current_timestamp_ms()

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
            "Bybit server time synced | server_time=%s offset_ms=%s",
            server_time,
            self._time_offset_ms,
        )

        await self._emit_event(
            "system.exchange.time_synced",
            {
                "exchange": self.EXCHANGE,
                "server_time": server_time,
                "offset_ms": self._time_offset_ms,
            },
            priority=EventPriority.LOW,
        )

        return self._time_offset_ms

    async def get_instruments_info(
        self,
        *,
        category: str = "linear",
        symbol: str | None = None,
        base_coin: str | None = None,
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
        )

        result = payload.get("result", {})
        items = result.get("list", [])

        await self._emit_event(
            "market.instruments.info",
            {
                "exchange": self.EXCHANGE,
                "category": category,
                "symbol": symbol.upper() if symbol else None,
                "base_coin": base_coin.upper() if base_coin else None,
                "count": len(items) if isinstance(items, list) else 0,
                "snapshot_time": self._current_timestamp_ms(),
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
        symbol = symbol.upper()

        payload = await self._request(
            method="GET",
            path="/v5/market/orderbook",
            params={
                "category": category,
                "symbol": symbol,
                "limit": limit,
            },
        )

        result = payload.get("result", {})
        if not isinstance(result, dict):
            result = {}

        normalized = {
            "exchange": self.EXCHANGE,
            "category": category,
            "symbol": result.get("s") or symbol,
            "update_id": result.get("u"),
            "sequence": result.get("seq"),
            "timestamp": self._safe_int(result.get("ts")),
            "bids": [
                [self._safe_float(price), self._safe_float(qty)]
                for price, qty in result.get("b", [])
            ],
            "asks": [
                [self._safe_float(price), self._safe_float(qty)]
                for price, qty in result.get("a", [])
            ],
            "snapshot_time": self._current_timestamp_ms(),
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
        symbol = symbol.upper()

        payload = await self._request(
            method="GET",
            path="/v5/market/recent-trade",
            params={
                "category": category,
                "symbol": symbol,
                "limit": limit,
            },
        )

        result = payload.get("result", {})
        trades = result.get("list", [])

        if not isinstance(trades, list):
            trades = []

        normalized = [
            self._normalize_trade(item, category=category, fallback_symbol=symbol)
            for item in trades
            if isinstance(item, dict)
        ]

        await self._emit_event(
            "market.trades.snapshot",
            {
                "exchange": self.EXCHANGE,
                "category": category,
                "symbol": symbol,
                "count": len(normalized),
                "trades": normalized,
                "snapshot_time": self._current_timestamp_ms(),
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
        start: int | None = None,
        end: int | None = None,
    ) -> list[dict[str, Any]]:
        symbol = symbol.upper()

        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol,
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
        )

        result = payload.get("result", {})
        raw_klines = result.get("list", [])

        if not isinstance(raw_klines, list):
            raw_klines = []

        normalized = [
            self._normalize_kline(
                item=item,
                category=category,
                symbol=symbol,
                timeframe=interval,
            )
            for item in raw_klines
            if isinstance(item, list) and len(item) >= 7
        ]

        await self._emit_event(
            "market.candles.snapshot",
            {
                "exchange": self.EXCHANGE,
                "category": category,
                "symbol": symbol,
                "timeframe": interval,
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
        category: str = "linear",
        symbol: str | None = None,
        base_coin: str | None = None,
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
        )

        result = payload.get("result", {})
        raw_items = result.get("list", [])

        if not isinstance(raw_items, list):
            raw_items = []

        normalized = [
            self._normalize_ticker(item, category=category)
            for item in raw_items
            if isinstance(item, dict)
        ]

        await self._emit_event(
            "market.tickers.snapshot",
            {
                "exchange": self.EXCHANGE,
                "category": category,
                "symbol": symbol.upper() if symbol else None,
                "base_coin": base_coin.upper() if base_coin else None,
                "count": len(normalized),
                "tickers": normalized,
                "snapshot_time": self._current_timestamp_ms(),
            },
            priority=EventPriority.LOW,
        )

        return normalized

    async def get_funding_rate(
        self,
        *,
        category: str = "linear",
        symbol: str,
    ) -> dict[str, Any]:
        """
        Bybit funding rate can be derived from ticker payload for linear/inverse contracts.
        This method wraps get_tickers(symbol=...) and emits a dedicated funding snapshot.
        """

        symbol = symbol.upper()

        tickers = await self.get_tickers(
            category=category,
            symbol=symbol,
        )

        item = tickers[0] if tickers else {}

        normalized = {
            "exchange": self.EXCHANGE,
            "category": category,
            "symbol": symbol,
            "funding_rate": item.get("funding_rate"),
            "next_funding_time": item.get("next_funding_time"),
            "mark_price": item.get("mark_price"),
            "index_price": item.get("index_price"),
            "timestamp": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "market.funding.snapshot",
            normalized,
            priority=EventPriority.NORMAL,
        )

        return normalized

    async def get_open_interest(
        self,
        *,
        category: str = "linear",
        symbol: str,
        interval_time: str = "5min",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        symbol = symbol.upper()

        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "intervalTime": interval_time,
            "limit": limit,
        }

        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        payload = await self._request(
            method="GET",
            path="/v5/market/open-interest",
            params=params,
        )

        result = payload.get("result", {})
        raw_items = result.get("list", [])

        if not isinstance(raw_items, list):
            raw_items = []

        normalized = [
            {
                "exchange": self.EXCHANGE,
                "category": category,
                "symbol": symbol,
                "open_interest": self._safe_float(item.get("openInterest")),
                "timestamp": self._safe_int(item.get("timestamp")),
            }
            for item in raw_items
            if isinstance(item, dict)
        ]

        await self._emit_event(
            "market.open_interest.snapshot",
            {
                "exchange": self.EXCHANGE,
                "category": category,
                "symbol": symbol,
                "interval_time": interval_time,
                "count": len(normalized),
                "items": normalized,
                "snapshot_time": self._current_timestamp_ms(),
            },
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
    ) -> dict[str, Any]:
        await self._ensure_session()

        assert self._session is not None

        method = method.upper()
        params = dict(params or {})
        url = f"{self._rest_config.rest_url}{path}"

        last_error: Exception | None = None

        for attempt in range(self._rest_config.request_retries + 1):
            try:
                self._logger.debug(
                    "Sending Bybit market REST request | method=%s path=%s attempt=%s",
                    method,
                    path,
                    attempt + 1,
                )

                async with self._session.request(
                    method=method,
                    url=url,
                    params=params if method == "GET" else None,
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
                            f"Bybit REST HTTP error | method={method} path={path} "
                            f"status={response.status} code={error_payload.get('code')}"
                        )

                    payload = await response.json()

                    ret_code = payload.get("retCode")
                    if ret_code != 0:
                        await self._handle_business_error(
                            method=method,
                            path=path,
                            payload=payload,
                        )

                        raise RuntimeError(
                            f"Bybit REST business error | method={method} path={path} "
                            f"retCode={ret_code} retMsg={payload.get('retMsg')}"
                        )

                    if self._rest_config.emit_success_events:
                        await self._emit_event(
                            "system.exchange.rest.success",
                            {
                                "exchange": self.EXCHANGE,
                                "method": method,
                                "path": path,
                                "ret_code": ret_code,
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
                        "Bybit market REST request failed | method=%s path=%s attempts=%s",
                        method,
                        path,
                        attempt + 1,
                    )
                    raise

                self._logger.warning(
                    "Bybit market REST request retry scheduled | method=%s path=%s attempt=%s",
                    method,
                    path,
                    attempt + 1,
                )

                await asyncio.sleep(self._rest_config.retry_delay_seconds)

        if last_error is not None:
            raise last_error

        raise RuntimeError(f"Bybit REST request failed unexpectedly | method={method} path={path}")

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
            "Bybit REST HTTP error | method=%s path=%s status=%s code=%s",
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
            "Bybit REST business error | method=%s path=%s ret_code=%s ret_msg=%s",
            method,
            path,
            payload.get("retCode"),
            payload.get("retMsg"),
        )

        if not self._rest_config.emit_error_events:
            return

        await self._emit_event(
            "system.exchange.rest.error",
            {
                "exchange": self.EXCHANGE,
                "method": method,
                "path": path,
                "ret_code": payload.get("retCode"),
                "message": payload.get("retMsg"),
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
                "Failed to emit Bybit market REST event | topic=%s",
                topic,
            )

    # ------------------------------------------------------------------
    # Normalizers
    # ------------------------------------------------------------------

    def _normalize_trade(
        self,
        item: dict[str, Any],
        *,
        category: str,
        fallback_symbol: str,
    ) -> dict[str, Any]:
        return {
            "exchange": self.EXCHANGE,
            "category": category,
            "symbol": item.get("symbol") or fallback_symbol,
            "trade_id": item.get("execId"),
            "price": self._safe_float(item.get("price")),
            "qty": self._safe_float(item.get("size")),
            "side": (item.get("side") or "").lower(),
            "trade_time": self._safe_int(item.get("time")),
            "is_block_trade": item.get("isBlockTrade"),
        }

    def _normalize_kline(
        self,
        *,
        item: list[Any],
        category: str,
        symbol: str,
        timeframe: str,
    ) -> dict[str, Any]:
        return {
            "exchange": self.EXCHANGE,
            "category": category,
            "symbol": symbol,
            "timeframe": timeframe,
            "open_time": self._safe_int(item[0]),
            "close_time": None,
            "open": self._safe_float(item[1]),
            "high": self._safe_float(item[2]),
            "low": self._safe_float(item[3]),
            "close": self._safe_float(item[4]),
            "volume": self._safe_float(item[5]),
            "quote_volume": self._safe_float(item[6]),
            "trades_count": None,
            "is_closed": True,
        }

    def _normalize_ticker(
        self,
        item: dict[str, Any],
        *,
        category: str,
    ) -> dict[str, Any]:
        return {
            "exchange": self.EXCHANGE,
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

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_error_payload(response_text: str) -> dict[str, Any]:
        try:
            import json

            raw = json.loads(response_text)

            return {
                "code": raw.get("retCode") or raw.get("code"),
                "message": raw.get("retMsg") or raw.get("message") or raw.get("msg"),
            }
        except Exception:
            return {
                "code": None,
                "message": "Unable to parse Bybit error response",
            }

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
    def _current_timestamp_ms() -> int:
        return int(time.time() * 1000)