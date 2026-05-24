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
from data.market_ingestion import MarketIngestionService


@dataclass(slots=True)
class MexcMarketRestClientConfig:
    """
    Local MEXC Futures public REST market-data adapter config.

    MEXC is currently used only as a market-data source in our architecture.
    Trading/private endpoints should live in a separate future adapter only
    if MEXC becomes an execution exchange.
    """

    rest_url: str = "https://contract.mexc.com"
    timeout_seconds: float = 10.0

    request_retries: int = 2
    retry_delay_seconds: float = 0.25

    emit_success_events: bool = False
    emit_error_events: bool = True

    @classmethod
    def from_core_config(cls, config: Config) -> "MexcMarketRestClientConfig":
        defaults = cls()
        return cls(
            rest_url=defaults.rest_url,
            timeout_seconds=config.exchange.timeout_seconds,
        )


class MexcRestClient:
    """
    MEXC Futures public REST market-data adapter.

    Responsibilities:
    - perform public MEXC Futures REST requests;
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

    Note:
    - This class intentionally does not implement private/trading endpoints.
    - Binance is the execution exchange for now.
    """

    EXCHANGE = "mexc"
    SOURCE = "mexc_rest"

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        rest_config: MexcMarketRestClientConfig | None = None,
        market_ingestion: MarketIngestionService | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._market_ingestion = market_ingestion
        self._rest_config = rest_config or MexcMarketRestClientConfig.from_core_config(config)

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
        self._logger.debug("MEXC market REST register called | subscriptions=0")

    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            self._logger.warning("MEXC market REST client already started")
            self._started = True
            return

        timeout = aiohttp.ClientTimeout(total=self._rest_config.timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._started = True

        self._logger.info(
            "MEXC market REST client started | base_url=%s timeout=%s",
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
            self._logger.warning("MEXC market REST client already stopped")
            self._started = False
            return

        await self._session.close()
        self._session = None
        self._started = False

        self._logger.info("MEXC market REST client stopped")

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
            path="/api/v1/contract/ping",
        )

    async def sync_time(self) -> int:
        local_before = self._current_timestamp_ms()
        payload = await self.get_server_time()
        local_after = self._current_timestamp_ms()

        data = payload.get("data")
        if data is None:
            raise RuntimeError("MEXC server time missing in response")

        server_time = int(data)
        local_estimated = (local_before + local_after) // 2
        self._time_offset_ms = server_time - local_estimated

        self._logger.info(
            "MEXC server time synced | server_time=%s offset_ms=%s",
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

    async def get_contract_info(
        self,
        *,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        normalized_symbol = self._normalize_symbol(symbol) if symbol else None

        path = "/api/v1/contract/detail"
        if normalized_symbol:
            path = f"/api/v1/contract/detail/{normalized_symbol}"

        payload = await self._request(
            method="GET",
            path=path,
        )

        data = payload.get("data", [])
        count = len(data) if isinstance(data, list) else 1 if data else 0

        await self._emit_event(
            "market.instruments.info",
            {
                "exchange": self.EXCHANGE,
                "symbol": normalized_symbol,
                "count": count,
                "snapshot_time": self._current_timestamp_ms(),
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
        )

        data = payload.get("data", {})
        if not isinstance(data, dict):
            data = {}

        normalized = {
            "exchange": self.EXCHANGE,
            "symbol": normalized_symbol,
            "version": data.get("version"),
            "timestamp": self._safe_int(data.get("timestamp")),
            "asks": [
                self._normalize_depth_level(level)
                for level in data.get("asks", [])
            ],
            "bids": [
                self._normalize_depth_level(level)
                for level in data.get("bids", [])
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
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        normalized_symbol = self._normalize_symbol(symbol)

        payload = await self._request(
            method="GET",
            path=f"/api/v1/contract/deals/{normalized_symbol}",
            params={"limit": limit},
        )

        raw_items = payload.get("data", [])
        if not isinstance(raw_items, list):
            raw_items = []

        normalized = [
            self._normalize_trade(item, symbol=normalized_symbol)
            for item in raw_items
            if isinstance(item, dict)
        ]

        await self._emit_event(
            "market.trades.snapshot",
            {
                "exchange": self.EXCHANGE,
                "symbol": normalized_symbol,
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
        symbol: str,
        interval: str = "Min1",
        limit: int = 500,
        start: int | None = None,
        end: int | None = None,
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
        )

        data = payload.get("data", {})
        if not isinstance(data, dict):
            data = {}

        normalized = self._normalize_klines_payload(
            data=data,
            symbol=normalized_symbol,
            timeframe=interval,
        )

        await self._emit_event(
            "market.candles.snapshot",
            {
                "exchange": self.EXCHANGE,
                "symbol": normalized_symbol,
                "timeframe": interval,
                "count": len(normalized),
                "candles": normalized,
                "snapshot_time": self._current_timestamp_ms(),
            },
            priority=EventPriority.LOW,
        )

        return normalized

    async def get_ticker(
        self,
        *,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        normalized_symbol = self._normalize_symbol(symbol) if symbol else None

        if normalized_symbol:
            path = f"/api/v1/contract/ticker/{normalized_symbol}"
        else:
            path = "/api/v1/contract/ticker"

        payload = await self._request(
            method="GET",
            path=path,
        )

        data = payload.get("data", [])
        count = len(data) if isinstance(data, list) else 1 if data else 0

        await self._emit_event(
            "market.tickers.snapshot",
            {
                "exchange": self.EXCHANGE,
                "symbol": normalized_symbol,
                "count": count,
                "snapshot_time": self._current_timestamp_ms(),
            },
            priority=EventPriority.LOW,
        )

        return payload

    async def get_funding_rate(
        self,
        *,
        symbol: str,
    ) -> dict[str, Any]:
        normalized_symbol = self._normalize_symbol(symbol)

        payload = await self._request(
            method="GET",
            path=f"/api/v1/contract/funding_rate/{normalized_symbol}",
        )

        data = payload.get("data", {})
        if not isinstance(data, dict):
            data = {}

        normalized = {
            "exchange": self.EXCHANGE,
            "symbol": normalized_symbol,
            "funding_rate": self._safe_float(data.get("fundingRate")),
            "max_funding_rate": self._safe_float(data.get("maxFundingRate")),
            "min_funding_rate": self._safe_float(data.get("minFundingRate")),
            "collect_cycle": data.get("collectCycle"),
            "next_settle_time": self._safe_int(data.get("nextSettleTime")),
            "timestamp": self._current_timestamp_ms(),
        }

        await self._emit_event(
            "market.funding.snapshot",
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
                    "Sending MEXC market REST request | method=%s path=%s attempt=%s",
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
                            f"MEXC REST HTTP error | method={method} path={path} "
                            f"status={response.status} code={error_payload.get('code')}"
                        )

                    payload = await response.json()

                    success = payload.get("success")
                    code = payload.get("code")

                    if success is False:
                        await self._handle_business_error(
                            method=method,
                            path=path,
                            payload=payload,
                        )

                        raise RuntimeError(
                            f"MEXC REST business error | method={method} path={path} "
                            f"code={code} message={payload.get('message')}"
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
                        "MEXC market REST request failed | method=%s path=%s attempts=%s",
                        method,
                        path,
                        attempt + 1,
                    )
                    raise

                self._logger.warning(
                    "MEXC market REST request retry scheduled | method=%s path=%s attempt=%s",
                    method,
                    path,
                    attempt + 1,
                )

                await asyncio.sleep(self._rest_config.retry_delay_seconds)

        if last_error is not None:
            raise last_error

        raise RuntimeError(f"MEXC REST request failed unexpectedly | method={method} path={path}")

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
            "MEXC REST HTTP error | method=%s path=%s status=%s code=%s",
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
            "MEXC REST business error | method=%s path=%s code=%s message=%s",
            method,
            path,
            payload.get("code"),
            payload.get("message"),
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
                "message": payload.get("message"),
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
                "Failed to emit MEXC market REST event | topic=%s",
                topic,
            )

    # ------------------------------------------------------------------
    # Normalizers
    # ------------------------------------------------------------------

    def _normalize_trade(
        self,
        item: dict[str, Any],
        *,
        symbol: str,
    ) -> dict[str, Any]:
        return {
            "exchange": self.EXCHANGE,
            "symbol": symbol,
            "trade_id": item.get("id"),
            "price": self._safe_float(item.get("price")),
            "qty": self._safe_float(item.get("vol")),
            "side": self._normalize_side_from_int(item.get("side")),
            "trade_time": self._safe_int(item.get("time")),
        }

    def _normalize_klines_payload(
        self,
        *,
        data: dict[str, Any],
        symbol: str,
        timeframe: str,
    ) -> list[dict[str, Any]]:
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

        candles: list[dict[str, Any]] = []

        for index in range(size):
            candles.append(
                {
                    "exchange": self.EXCHANGE,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "open_time": self._normalize_kline_timestamp(times[index]),
                    "close_time": None,
                    "open": self._safe_float(opens[index]),
                    "high": self._safe_float(highs[index]),
                    "low": self._safe_float(lows[index]),
                    "close": self._safe_float(closes[index]),
                    "volume": self._safe_float(vols[index]),
                    "quote_volume": self._safe_float(amounts[index]),
                    "trades_count": None,
                    "is_closed": True,
                    "amount": self._safe_float(amounts[index]),
                }
            )

        return candles

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
    def _normalize_side_from_int(value: Any) -> str | None:
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

            if ts < 10_000_000_000:
                return ts * 1000

            return ts
        except (TypeError, ValueError):
            return None

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
    def _parse_error_payload(response_text: str) -> dict[str, Any]:
        try:
            import json

            raw = json.loads(response_text)

            return {
                "code": raw.get("code"),
                "message": raw.get("message") or raw.get("msg"),
            }
        except Exception:
            return {
                "code": None,
                "message": "Unable to parse MEXC error response",
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