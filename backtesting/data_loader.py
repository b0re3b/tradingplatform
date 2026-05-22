from __future__ import annotations

import asyncio
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from core.logger import get_logger

from backtesting.config import BacktestConfig
from backtesting.exceptions import BacktestDataError
from backtesting.models import (
    HistoricalCandle,
    HistoricalFundingRate,
    HistoricalMarkPrice,
    HistoricalOpenInterest,
    HistoricalTrade,
)
from backtesting.utils import decimal_from, ensure_dir, normalize_symbol, timeframe_to_ms


@dataclass(slots=True)
class HistoricalDataset:
    candles: list[HistoricalCandle]
    funding_rates: list[HistoricalFundingRate]
    open_interest: list[HistoricalOpenInterest]
    trades: list[HistoricalTrade]
    mark_prices: list[HistoricalMarkPrice]

    def is_empty(self) -> bool:
        return not any(
            (
                self.candles,
                self.funding_rates,
                self.open_interest,
                self.trades,
                self.mark_prices,
            )
        )


class BinanceHistoricalDataLoader:
    """
    Read-only Binance USD-M Futures historical loader.

    It uses public /fapi endpoints only. It never calls private account or trading endpoints.
    Local CSV cache is used to avoid repeated downloads.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._config.validate()
        self._logger = get_logger(
            __name__,
            service="backtesting.data_loader",
            event_type="historical_loader",
        )
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "BinanceHistoricalDataLoader":
        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def load(self, *, start_time_ms: int, end_time_ms: int) -> HistoricalDataset:
        candles: list[HistoricalCandle] = []
        funding: list[HistoricalFundingRate] = []
        open_interest: list[HistoricalOpenInterest] = []
        trades: list[HistoricalTrade] = []
        mark_prices: list[HistoricalMarkPrice] = []

        for symbol in self._config.symbols:
            normalized_symbol = normalize_symbol(symbol)

            for timeframe in self._config.timeframes:
                candles.extend(
                    await self.get_historical_klines(
                        symbol=normalized_symbol,
                        interval=timeframe,
                        start_time_ms=start_time_ms,
                        end_time_ms=end_time_ms,
                    )
                )

                mark_prices.extend(
                    await self.get_mark_price_klines(
                        symbol=normalized_symbol,
                        interval=timeframe,
                        start_time_ms=start_time_ms,
                        end_time_ms=end_time_ms,
                    )
                )

            if self._config.enable_funding:
                funding.extend(
                    await self.get_funding_rate_history(
                        symbol=normalized_symbol,
                        start_time_ms=start_time_ms,
                        end_time_ms=end_time_ms,
                    )
                )

            if self._config.enable_open_interest:
                # Binance open interest history has coarse periods only:
                # 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d.
                # For a 1m strategy context this must be treated as latest-known OI,
                # never as a synthetic 1m series.
                oi_period = "15m" if "15m" in self._config.timeframes else "5m"
                open_interest.extend(
                    await self.get_open_interest_history(
                        symbol=normalized_symbol,
                        period=oi_period,
                        start_time_ms=start_time_ms,
                        end_time_ms=end_time_ms,
                    )
                )

            if self._config.enable_orderflow and self._config.enable_agg_trades:
                try:
                    trades.extend(
                        await self.get_agg_trades(
                            symbol=normalized_symbol,
                            start_time_ms=start_time_ms,
                            end_time_ms=end_time_ms,
                        )
                    )
                except BacktestDataError as exc:
                    # Do not synthesize orderflow data. Keep the real pipeline running
                    # with candles/funding/OI/mark-price data and make missing trades explicit.
                    self._logger.warning(
                        "Agg trades history unavailable; trade replay will be incomplete | symbol=%s error=%s",
                        normalized_symbol,
                        exc,
                    )
            elif self._config.enable_orderflow:
                self._logger.info(
                    "Agg trades download disabled; orderflow trade replay will be incomplete | symbol=%s",
                    normalized_symbol,
                )

        dataset = HistoricalDataset(
            candles=candles,
            funding_rates=funding,
            open_interest=open_interest,
            trades=trades,
            mark_prices=mark_prices,
        )

        if dataset.is_empty():
            raise BacktestDataError("No historical Binance data was loaded.")

        return dataset

    async def get_historical_klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[HistoricalCandle]:
        cache = self._cache_file(
            symbol=symbol,
            category="klines",
            timeframe=interval,
        )
        cached = self._read_cached_rows(cache, start_time_ms, end_time_ms)

        if self._covers_range(
            cached,
            start_time_ms,
            end_time_ms,
            timestamp_key="close_time_ms",
        ):
            return [self._row_to_candle(row) for row in cached]

        rows: list[dict[str, Any]] = []
        cursor = start_time_ms
        interval_ms = timeframe_to_ms(interval)

        while cursor < end_time_ms:
            data = await self._request(
                "/fapi/v1/klines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_time_ms,
                    "limit": self._config.max_klines_per_request,
                },
            )
            if not data:
                break

            for item in data:
                close_time_ms = int(item[6])
                if close_time_ms > end_time_ms:
                    continue

                rows.append(
                    {
                        "exchange": self._config.exchange,
                        "market_type": self._config.market_type,
                        "symbol": symbol,
                        "timeframe": interval,
                        "open_time_ms": int(item[0]),
                        "close_time_ms": close_time_ms,
                        "open": str(item[1]),
                        "high": str(item[2]),
                        "low": str(item[3]),
                        "close": str(item[4]),
                        "volume": str(item[5]),
                        "quote_volume": str(item[7]),
                        "trades_count": int(item[8]),
                    }
                )

            last_open = int(data[-1][0])
            next_cursor = last_open + interval_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor

        merged = self._merge_rows(cached, rows, unique_key="open_time_ms")
        self._write_cached_rows(cache, merged)

        return [
            self._row_to_candle(row)
            for row in merged
            if start_time_ms <= int(row["close_time_ms"]) <= end_time_ms
        ]

    async def get_mark_price_klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[HistoricalMarkPrice]:
        cache = self._cache_file(
            symbol=symbol,
            category="mark_price_klines",
            timeframe=interval,
        )
        cached = self._read_cached_rows(cache, start_time_ms, end_time_ms)

        if self._covers_range(
            cached,
            start_time_ms,
            end_time_ms,
            timestamp_key="close_time_ms",
        ):
            return [self._row_to_mark_price(row) for row in cached]

        rows: list[dict[str, Any]] = []
        cursor = start_time_ms
        interval_ms = timeframe_to_ms(interval)

        while cursor < end_time_ms:
            data = await self._request(
                "/fapi/v1/markPriceKlines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_time_ms,
                    "limit": self._config.max_klines_per_request,
                },
            )
            if not data:
                break

            for item in data:
                rows.append(
                    {
                        "exchange": self._config.exchange,
                        "market_type": self._config.market_type,
                        "symbol": symbol,
                        "timeframe": interval,
                        "open_time_ms": int(item[0]),
                        "close_time_ms": int(item[6]),
                        "open": str(item[1]),
                        "high": str(item[2]),
                        "low": str(item[3]),
                        "close": str(item[4]),
                    }
                )

            last_open = int(data[-1][0])
            next_cursor = last_open + interval_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor

        merged = self._merge_rows(cached, rows, unique_key="open_time_ms")
        self._write_cached_rows(cache, merged)

        return [
            self._row_to_mark_price(row)
            for row in merged
            if start_time_ms <= int(row["close_time_ms"]) <= end_time_ms
        ]

    async def get_funding_rate_history(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[HistoricalFundingRate]:
        cache = self._cache_file(symbol=symbol, category="funding", timeframe="8h")
        cached = self._read_cached_rows(
            cache,
            start_time_ms,
            end_time_ms,
            timestamp_key="funding_time_ms",
        )

        if self._covers_range(
            cached,
            start_time_ms,
            end_time_ms,
            timestamp_key="funding_time_ms",
        ):
            return [self._row_to_funding(row) for row in cached]

        rows: list[dict[str, Any]] = []
        cursor = start_time_ms

        while cursor < end_time_ms:
            data = await self._request(
                "/fapi/v1/fundingRate",
                {
                    "symbol": symbol,
                    "startTime": cursor,
                    "endTime": end_time_ms,
                    "limit": self._config.max_funding_points_per_request,
                },
            )
            if not data:
                break

            for item in data:
                funding_time = int(item["fundingTime"])
                rows.append(
                    {
                        "exchange": self._config.exchange,
                        "market_type": self._config.market_type,
                        "symbol": symbol,
                        "funding_time_ms": funding_time,
                        "funding_rate": str(item.get("fundingRate", "0")),
                        "mark_price": (
                            str(item.get("markPrice"))
                            if item.get("markPrice") is not None
                            else ""
                        ),
                    }
                )

            next_cursor = int(data[-1]["fundingTime"]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor

        merged = self._merge_rows(cached, rows, unique_key="funding_time_ms")
        self._write_cached_rows(cache, merged)

        return [
            self._row_to_funding(row)
            for row in merged
            if start_time_ms <= int(row["funding_time_ms"]) <= end_time_ms
        ]

    async def get_open_interest_history(
        self,
        *,
        symbol: str,
        period: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[HistoricalOpenInterest]:
        cache = self._cache_file(
            symbol=symbol,
            category="open_interest",
            timeframe=period,
        )
        cached = self._read_cached_rows(
            cache,
            start_time_ms,
            end_time_ms,
            timestamp_key="timestamp_ms",
        )

        if self._covers_range(
            cached,
            start_time_ms,
            end_time_ms,
            timestamp_key="timestamp_ms",
        ):
            return [self._row_to_open_interest(row) for row in cached]

        rows: list[dict[str, Any]] = []
        cursor = start_time_ms

        while cursor < end_time_ms:
            data = await self._request(
                "/futures/data/openInterestHist",
                {
                    "symbol": symbol,
                    "period": period,
                    "startTime": cursor,
                    "endTime": end_time_ms,
                    "limit": self._config.max_open_interest_points_per_request,
                },
            )
            if not data:
                break

            for item in data:
                ts = int(item["timestamp"])
                rows.append(
                    {
                        "exchange": self._config.exchange,
                        "market_type": self._config.market_type,
                        "symbol": symbol,
                        "timestamp_ms": ts,
                        "sum_open_interest": str(item.get("sumOpenInterest", "0")),
                        "sum_open_interest_value": (
                            str(item.get("sumOpenInterestValue"))
                            if item.get("sumOpenInterestValue") is not None
                            else ""
                        ),
                    }
                )

            next_cursor = int(data[-1]["timestamp"]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor

        merged = self._merge_rows(cached, rows, unique_key="timestamp_ms")
        self._write_cached_rows(cache, merged)

        return [
            self._row_to_open_interest(row)
            for row in merged
            if start_time_ms <= int(row["timestamp_ms"]) <= end_time_ms
        ]

    async def get_agg_trades(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[HistoricalTrade]:
        cache = self._cache_file(symbol=symbol, category="agg_trades", timeframe="tick")
        cached = self._read_cached_rows(
            cache,
            start_time_ms,
            end_time_ms,
            timestamp_key="timestamp_ms",
        )

        if self._covers_range(
            cached,
            start_time_ms,
            end_time_ms,
            timestamp_key="timestamp_ms",
        ):
            return [self._row_to_trade(row) for row in cached]

        rows: list[dict[str, Any]] = []
        cursor = start_time_ms

        while cursor < end_time_ms:
            data = await self._request(
                "/fapi/v1/aggTrades",
                {
                    "symbol": symbol,
                    "startTime": cursor,
                    "endTime": end_time_ms,
                    "limit": self._config.max_agg_trades_per_request,
                },
            )
            if not data:
                break

            for item in data:
                ts = int(item["T"])
                rows.append(
                    {
                        "exchange": self._config.exchange,
                        "market_type": self._config.market_type,
                        "symbol": symbol,
                        "trade_id": str(item["a"]),
                        "price": str(item["p"]),
                        "quantity": str(item["q"]),
                        "timestamp_ms": ts,
                        "is_buyer_maker": bool(item["m"]),
                    }
                )

            next_cursor = int(data[-1]["T"]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor

        merged = self._merge_rows(cached, rows, unique_key="trade_id")
        self._write_cached_rows(cache, merged)

        return [
            self._row_to_trade(row)
            for row in merged
            if start_time_ms <= int(row["timestamp_ms"]) <= end_time_ms
        ]

    async def _request(self, path: str, params: dict[str, Any]) -> Any:
        if self._session is None:
            raise BacktestDataError(
                "BinanceHistoricalDataLoader must be used as an async context manager."
            )

        url = f"{self._config.binance_public_base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(1, self._config.request_retries + 1):
            if self._config.request_delay_seconds > 0:
                await asyncio.sleep(self._config.request_delay_seconds)

            try:
                async with self._session.get(url, params=params) as response:
                    body = await response.text()

                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        delay = self._retry_delay(
                            attempt=attempt,
                            retry_after=retry_after,
                        )
                        last_error = BacktestDataError(
                            "Binance historical request rate-limited "
                            f"status=429 path={path} body={body[:300]}"
                        )
                        self._logger.warning(
                            "Binance rate limit hit; backing off | path=%s attempt=%s delay=%s",
                            path,
                            attempt,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if response.status >= 400:
                        raise BacktestDataError(
                            "Binance historical request failed "
                            f"status={response.status} path={path} body={body[:300]}"
                        )

                    if not body:
                        return None

                    return json.loads(body)

            except BacktestDataError as exc:
                last_error = exc
                if attempt >= self._config.request_retries:
                    break

                delay = self._config.request_retry_delay_seconds * attempt
                self._logger.warning(
                    "Historical request retry | path=%s attempt=%s delay=%s error=%s",
                    path,
                    attempt,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

            except Exception as exc:
                last_error = exc
                if attempt >= self._config.request_retries:
                    break

                delay = self._config.request_retry_delay_seconds * attempt
                self._logger.warning(
                    "Historical request retry | path=%s attempt=%s delay=%s error=%s",
                    path,
                    attempt,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        raise BacktestDataError(
            f"Binance historical request failed after retries | path={path} error={last_error}"
        )

    def _retry_delay(self, *, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return max(
                    float(retry_after),
                    self._config.request_retry_delay_seconds,
                )
            except ValueError:
                pass

        return self._config.request_retry_delay_seconds * attempt

    def _cache_file(self, *, symbol: str, category: str, timeframe: str) -> Path:
        return (
            ensure_dir(self._config.historical_cache_dir / symbol / timeframe)
            / f"{category}.csv"
        )

    def _read_cached_rows(
        self,
        path: Path,
        start_time_ms: int,
        end_time_ms: int,
        *,
        timestamp_key: str = "close_time_ms",
    ) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return [
                row
                for row in reader
                if row.get(timestamp_key)
                and start_time_ms <= int(row[timestamp_key]) <= end_time_ms
            ]

    def _write_cached_rows(self, path: Path, rows: list[dict[str, Any]]) -> None:
        ensure_dir(path.parent)
        if not rows:
            return

        fieldnames: list[str] = list(
            dict.fromkeys(k for row in rows for k in row.keys())
        )

        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _covers_range(
        rows: list[dict[str, Any]],
        start_time_ms: int,
        end_time_ms: int,
        *,
        timestamp_key: str,
    ) -> bool:
        if not rows:
            return False

        values = sorted(
            int(row[timestamp_key])
            for row in rows
            if row.get(timestamp_key)
        )
        if not values:
            return False

        return values[0] <= start_time_ms and values[-1] >= end_time_ms

    @staticmethod
    def _merge_rows(
        old: list[dict[str, Any]],
        new: list[dict[str, Any]],
        *,
        unique_key: str,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        for row in old + new:
            if row.get(unique_key) is not None:
                merged[str(row[unique_key])] = row

        return sorted(merged.values(), key=lambda row: int(row.get(unique_key, 0)))

    @staticmethod
    def _row_to_candle(row: dict[str, Any]) -> HistoricalCandle:
        return HistoricalCandle(
            exchange=row["exchange"],
            market_type=row["market_type"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            open_time_ms=int(row["open_time_ms"]),
            close_time_ms=int(row["close_time_ms"]),
            open=decimal_from(row["open"]),
            high=decimal_from(row["high"]),
            low=decimal_from(row["low"]),
            close=decimal_from(row["close"]),
            volume=decimal_from(row["volume"]),
            quote_volume=decimal_from(row.get("quote_volume")),
            trades_count=int(row.get("trades_count") or 0),
        )

    @staticmethod
    def _row_to_mark_price(row: dict[str, Any]) -> HistoricalMarkPrice:
        return HistoricalMarkPrice(
            exchange=row["exchange"],
            market_type=row["market_type"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            open_time_ms=int(row["open_time_ms"]),
            close_time_ms=int(row["close_time_ms"]),
            open=decimal_from(row["open"]),
            high=decimal_from(row["high"]),
            low=decimal_from(row["low"]),
            close=decimal_from(row["close"]),
        )

    @staticmethod
    def _row_to_funding(row: dict[str, Any]) -> HistoricalFundingRate:
        mark_price = row.get("mark_price") or None
        return HistoricalFundingRate(
            exchange=row["exchange"],
            market_type=row["market_type"],
            symbol=row["symbol"],
            funding_time_ms=int(row["funding_time_ms"]),
            funding_rate=decimal_from(row["funding_rate"]),
            mark_price=decimal_from(mark_price) if mark_price else None,
        )

    @staticmethod
    def _row_to_open_interest(row: dict[str, Any]) -> HistoricalOpenInterest:
        value = row.get("sum_open_interest_value") or None
        return HistoricalOpenInterest(
            exchange=row["exchange"],
            market_type=row["market_type"],
            symbol=row["symbol"],
            timestamp_ms=int(row["timestamp_ms"]),
            sum_open_interest=decimal_from(row["sum_open_interest"]),
            sum_open_interest_value=decimal_from(value) if value else None,
        )

    @staticmethod
    def _row_to_trade(row: dict[str, Any]) -> HistoricalTrade:
        raw_bool = row.get("is_buyer_maker")
        is_buyer_maker = (
            raw_bool
            if isinstance(raw_bool, bool)
            else str(raw_bool).lower() in {"1", "true", "yes"}
        )

        return HistoricalTrade(
            exchange=row["exchange"],
            market_type=row["market_type"],
            symbol=row["symbol"],
            trade_id=str(row["trade_id"]),
            price=decimal_from(row["price"]),
            quantity=decimal_from(row["quantity"]),
            timestamp_ms=int(row["timestamp_ms"]),
            is_buyer_maker=is_buyer_maker,
        )