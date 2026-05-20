"""
Historical market data downloader for backtesting.

This module downloads historical market data and stores it locally for offline
backtesting. It is Binance-first, but the implementation accepts an injected
REST client so it can be reused with other exchange adapters.

Important:
- This downloader is allowed to call exchange REST APIs.
- Market replay itself must not call live exchanges.
- This module does not run strategies, risk, execution or analytics.
- It only downloads, normalizes and stores historical data.
"""

from __future__ import annotations

import asyncio
import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
from typing import Callable, TypeVar

from backtesting.config import HistoryDownloaderConfig
from backtesting.enums import BacktestDataType, HistoricalDataFormat
from backtesting.exceptions import (
    HistoricalDataDownloadError,
    HistoricalDataFormatError,
    HistoricalDataStorageError,
    HistoricalDataValidationError,
)
from backtesting.models import (
    HistoricalCandle,
    HistoricalFundingRecord,
    HistoricalLiquidationRecord,
    HistoricalOpenInterestRecord,
    HistoricalOrderBookLevel,
    HistoricalOrderBookSnapshot,
    HistoricalTrade,
    SerializableMixin,
    ensure_aware_utc,
    timestamp_ms,
)

T = TypeVar("T")
try:
    from core.logger import get_logger
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


class HistoryDownloader:
    """
    Historical data downloader.

    The downloader expects an injected REST client with Binance-like methods.
    It uses method discovery to stay compatible with slightly different client
    method names.

    Expected possible REST client methods:
    - get_klines / get_candles / fetch_klines
    - get_agg_trades / get_trades / fetch_trades
    - get_funding_rate_history / get_funding_history
    - get_open_interest_history / get_open_interest
    - get_orderbook / get_order_book / depth
    - get_force_orders / get_liquidations
    """

    def __init__(
        self,
        config: HistoryDownloaderConfig | None = None,
        *,
        rest_client: Any | None = None,
        event_bus: Any | None = None,
        logger_name: str = "backtesting.history_downloader",
    ) -> None:
        self.config = config or HistoryDownloaderConfig()
        self.config.validate()

        self.rest_client = rest_client
        self.event_bus = event_bus
        self.logger = get_logger(logger_name)

        self._running = False
        self._lock = asyncio.Lock()

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def register(self) -> None:
        """
        Placeholder for consistency with the rest of the project.

        Downloader does not need EventBus subscriptions by default.
        """

    async def start(self) -> None:
        """
        Mark downloader as active.
        """

        async with self._lock:
            self._running = True
            await self._emit(
                "system.backtest.history_downloader.started",
                {
                    "exchange": self.config.exchange,
                    "market_type": self.config.market_type,
                    "symbols": self.config.symbols,
                    "timeframes": self.config.timeframes,
                },
            )

    async def stop(self) -> None:
        """
        Mark downloader as inactive.
        """

        async with self._lock:
            self._running = False
            await self._emit(
                "system.backtest.history_downloader.stopped",
                {
                    "exchange": self.config.exchange,
                    "market_type": self.config.market_type,
                },
            )

    # ---------------------------------------------------------------------
    # High-level API
    # ---------------------------------------------------------------------

    async def download_all(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        symbols: Sequence[str] | None = None,
        timeframes: Sequence[str] | None = None,
        data_types: set[BacktestDataType] | None = None,
    ) -> dict[BacktestDataType, dict[str, int]]:
        """
        Download all configured data types.

        Returns counts grouped by data type and symbol/timeframe key.
        """

        self._ensure_client()

        start = ensure_aware_utc(start_time)
        end = ensure_aware_utc(end_time)

        if end <= start:
            raise HistoricalDataValidationError(
                "end_time must be greater than start_time.",
                details={
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                },
            )

        selected_symbols = [symbol.upper() for symbol in (symbols or self.config.symbols)]
        selected_timeframes = list(timeframes or self.config.timeframes)
        selected_data_types = data_types or self.config.data_types

        results: dict[BacktestDataType, dict[str, int]] = {}

        await self._emit(
            "system.backtest.history_download.started",
            {
                "exchange": self.config.exchange,
                "market_type": self.config.market_type,
                "symbols": selected_symbols,
                "timeframes": selected_timeframes,
                "data_types": [item.value for item in selected_data_types],
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
            },
        )

        try:
            if BacktestDataType.CANDLES in selected_data_types:
                results[BacktestDataType.CANDLES] = {}
                for symbol in selected_symbols:
                    for timeframe in selected_timeframes:
                        records = await self.download_candles(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_time=start,
                            end_time=end,
                        )
                        results[BacktestDataType.CANDLES][f"{symbol}:{timeframe}"] = len(records)

            if BacktestDataType.TRADES in selected_data_types:
                results[BacktestDataType.TRADES] = {}
                for symbol in selected_symbols:
                    records = await self.download_trades(
                        symbol=symbol,
                        start_time=start,
                        end_time=end,
                    )
                    results[BacktestDataType.TRADES][symbol] = len(records)

            if BacktestDataType.FUNDING in selected_data_types:
                results[BacktestDataType.FUNDING] = {}
                for symbol in selected_symbols:
                    records = await self.download_funding(
                        symbol=symbol,
                        start_time=start,
                        end_time=end,
                    )
                    results[BacktestDataType.FUNDING][symbol] = len(records)

            if BacktestDataType.OPEN_INTEREST in selected_data_types:
                results[BacktestDataType.OPEN_INTEREST] = {}
                for symbol in selected_symbols:
                    records = await self.download_open_interest(
                        symbol=symbol,
                        start_time=start,
                        end_time=end,
                    )
                    results[BacktestDataType.OPEN_INTEREST][symbol] = len(records)

            if BacktestDataType.ORDERBOOK in selected_data_types or BacktestDataType.ORDERBOOK_SNAPSHOT in selected_data_types:
                results[BacktestDataType.ORDERBOOK_SNAPSHOT] = {}
                for symbol in selected_symbols:
                    records = await self.download_orderbook_snapshots(
                        symbol=symbol,
                        start_time=start,
                        end_time=end,
                    )
                    results[BacktestDataType.ORDERBOOK_SNAPSHOT][symbol] = len(records)

            if BacktestDataType.LIQUIDATIONS in selected_data_types:
                results[BacktestDataType.LIQUIDATIONS] = {}
                for symbol in selected_symbols:
                    records = await self.download_liquidations(
                        symbol=symbol,
                        start_time=start,
                        end_time=end,
                    )
                    results[BacktestDataType.LIQUIDATIONS][symbol] = len(records)

            await self._emit(
                "system.backtest.history_download.finished",
                {
                    "results": {
                        data_type.value: counts
                        for data_type, counts in results.items()
                    }
                },
            )

            return results

        except Exception as exc:
            await self._emit(
                "system.backtest.history_download.failed",
                {
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            )
            raise

    async def download_candles(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        save: bool = True,
    ) -> list[HistoricalCandle]:
        """
        Download OHLCV candles.
        """

        self._ensure_client()

        symbol = symbol.upper()
        start_ms = timestamp_ms(start_time)
        end_ms = timestamp_ms(end_time)

        output_path = self._build_path(
            data_type=BacktestDataType.CANDLES,
            symbol=symbol,
            timeframe=timeframe,
        )

        if self._should_skip_existing(output_path):
            return self._load_existing_stub(output_path, HistoricalCandle)

        raw_rows: list[Any] = []
        cursor = start_ms

        while cursor < end_ms:
            rows = await self._call_with_retries(
                [
                    "get_klines",
                    "get_candles",
                    "fetch_klines",
                    "fetch_candles",
                    "klines",
                ],
                symbol=symbol,
                interval=timeframe,
                timeframe=timeframe,
                start_time=cursor,
                startTime=cursor,
                end_time=end_ms,
                endTime=end_ms,
                limit=self.config.candle_limit_per_request,
                market_type=self.config.market_type,
            )

            rows = self._normalize_raw_rows(rows)

            if not rows:
                break

            raw_rows.extend(rows)

            last_open_time = self._extract_candle_open_time_ms(rows[-1])
            next_cursor = last_open_time + self._timeframe_to_milliseconds(timeframe)

            if next_cursor <= cursor:
                break

            cursor = next_cursor
            await self._sleep_rate_limit()

            if len(rows) < self.config.candle_limit_per_request:
                break

        records = [
            self._normalize_candle(row, symbol=symbol, timeframe=timeframe)
            for row in raw_rows
        ]
        records = self._dedupe_by_key(records, key=lambda item: item.open_time_ms)
        records = [
            item
            for item in records
            if start_ms <= item.open_time_ms <= end_ms
        ]

        if save:
            self.save_records(records, output_path)

        return records

    async def download_trades(
        self,
        *,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        save: bool = True,
    ) -> list[HistoricalTrade]:
        """
        Download historical trades or aggregate trades.
        """

        self._ensure_client()

        symbol = symbol.upper()
        start_ms = timestamp_ms(start_time)
        end_ms = timestamp_ms(end_time)

        output_path = self._build_path(
            data_type=BacktestDataType.TRADES,
            symbol=symbol,
        )

        if self._should_skip_existing(output_path):
            return self._load_existing_stub(output_path, HistoricalTrade)

        raw_rows: list[Any] = []
        cursor = start_ms

        while cursor < end_ms:
            rows = await self._call_with_retries(
                [
                    "get_agg_trades",
                    "get_historical_trades",
                    "get_trades",
                    "fetch_trades",
                    "agg_trades",
                ],
                symbol=symbol,
                start_time=cursor,
                startTime=cursor,
                end_time=end_ms,
                endTime=end_ms,
                limit=self.config.trade_limit_per_request,
                market_type=self.config.market_type,
            )

            rows = self._normalize_raw_rows(rows)

            if not rows:
                break

            raw_rows.extend(rows)

            last_ts = self._extract_trade_timestamp_ms(rows[-1])
            next_cursor = last_ts + 1

            if next_cursor <= cursor:
                break

            cursor = next_cursor
            await self._sleep_rate_limit()

            if len(rows) < self.config.trade_limit_per_request:
                break

        records = [
            self._normalize_trade(row, symbol=symbol)
            for row in raw_rows
        ]
        records = self._dedupe_by_key(records, key=lambda item: str(item.trade_id or item.timestamp_ms))
        records = [
            item
            for item in records
            if start_ms <= item.timestamp_ms <= end_ms
        ]

        if save:
            self.save_records(records, output_path)

        return records

    async def download_funding(
        self,
        *,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        save: bool = True,
    ) -> list[HistoricalFundingRecord]:
        """
        Download funding rate history.
        """

        self._ensure_client()

        symbol = symbol.upper()
        start_ms = timestamp_ms(start_time)
        end_ms = timestamp_ms(end_time)

        output_path = self._build_path(
            data_type=BacktestDataType.FUNDING,
            symbol=symbol,
        )

        if self._should_skip_existing(output_path):
            return self._load_existing_stub(output_path, HistoricalFundingRecord)

        raw_rows: list[Any] = []
        cursor = start_ms

        while cursor < end_ms:
            rows = await self._call_with_retries(
                [
                    "get_funding_rate_history",
                    "get_funding_history",
                    "fetch_funding_rate_history",
                    "funding_rate_history",
                ],
                symbol=symbol,
                start_time=cursor,
                startTime=cursor,
                end_time=end_ms,
                endTime=end_ms,
                limit=self.config.funding_limit_per_request,
                market_type=self.config.market_type,
            )

            rows = self._normalize_raw_rows(rows)

            if not rows:
                break

            raw_rows.extend(rows)

            last_ts = self._extract_timestamp_ms(rows[-1], ["fundingTime", "funding_time", "timestamp", "time"])
            next_cursor = last_ts + 1

            if next_cursor <= cursor:
                break

            cursor = next_cursor
            await self._sleep_rate_limit()

            if len(rows) < self.config.funding_limit_per_request:
                break

        records = [
            self._normalize_funding(row, symbol=symbol)
            for row in raw_rows
        ]
        records = self._dedupe_by_key(records, key=lambda item: item.timestamp_ms)
        records = [
            item
            for item in records
            if start_ms <= item.timestamp_ms <= end_ms
        ]

        if save:
            self.save_records(records, output_path)

        return records

    async def download_open_interest(
        self,
        *,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        period: str = "5m",
        save: bool = True,
    ) -> list[HistoricalOpenInterestRecord]:
        """
        Download open interest history.
        """

        self._ensure_client()

        symbol = symbol.upper()
        start_ms = timestamp_ms(start_time)
        end_ms = timestamp_ms(end_time)

        output_path = self._build_path(
            data_type=BacktestDataType.OPEN_INTEREST,
            symbol=symbol,
            timeframe=period,
        )

        if self._should_skip_existing(output_path):
            return self._load_existing_stub(output_path, HistoricalOpenInterestRecord)

        raw_rows: list[Any] = []
        cursor = start_ms

        while cursor < end_ms:
            rows = await self._call_with_retries(
                [
                    "get_open_interest_history",
                    "get_open_interest",
                    "fetch_open_interest_history",
                    "open_interest_history",
                ],
                symbol=symbol,
                period=period,
                start_time=cursor,
                startTime=cursor,
                end_time=end_ms,
                endTime=end_ms,
                limit=self.config.open_interest_limit_per_request,
                market_type=self.config.market_type,
            )

            rows = self._normalize_raw_rows(rows)

            if not rows:
                break

            raw_rows.extend(rows)

            last_ts = self._extract_timestamp_ms(rows[-1], ["timestamp", "time", "sumOpenInterestTime"])
            next_cursor = last_ts + 1

            if next_cursor <= cursor:
                break

            cursor = next_cursor
            await self._sleep_rate_limit()

            if len(rows) < self.config.open_interest_limit_per_request:
                break

        records = [
            self._normalize_open_interest(row, symbol=symbol)
            for row in raw_rows
        ]
        records = self._dedupe_by_key(records, key=lambda item: item.timestamp_ms)
        records = [
            item
            for item in records
            if start_ms <= item.timestamp_ms <= end_ms
        ]

        if save:
            self.save_records(records, output_path)

        return records

    async def download_orderbook_snapshots(
        self,
        *,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        depth: int = 100,
        interval: timedelta = timedelta(minutes=5),
        save: bool = True,
    ) -> list[HistoricalOrderBookSnapshot]:
        """
        Download periodic order book snapshots.

        Most exchanges do not provide deep historical order book snapshots via
        simple REST. This method is useful when the injected client supports it.
        Otherwise it can snapshot the current order book only when called live,
        which is not enough for historical replay.
        """

        self._ensure_client()

        symbol = symbol.upper()
        start = ensure_aware_utc(start_time)
        end = ensure_aware_utc(end_time)

        output_path = self._build_path(
            data_type=BacktestDataType.ORDERBOOK_SNAPSHOT,
            symbol=symbol,
        )

        if self._should_skip_existing(output_path):
            return self._load_existing_stub(output_path, HistoricalOrderBookSnapshot)

        records: list[HistoricalOrderBookSnapshot] = []
        current = start

        while current <= end:
            current_ms = timestamp_ms(current)

            try:
                row = await self._call_with_retries(
                    [
                        "get_orderbook_snapshot",
                        "get_order_book_snapshot",
                        "get_orderbook",
                        "get_order_book",
                        "depth",
                    ],
                    symbol=symbol,
                    timestamp=current_ms,
                    time=current_ms,
                    limit=depth,
                    depth=depth,
                    market_type=self.config.market_type,
                )
            except HistoricalDataDownloadError:
                raise

            if row:
                records.append(
                    self._normalize_orderbook_snapshot(
                        row,
                        symbol=symbol,
                        current_timestamp_ms=current_ms,
                    )
                )

            current = current + interval
            await self._sleep_rate_limit()

        if save:
            self.save_records(records, output_path)

        return records

    async def download_liquidations(
        self,
        *,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        save: bool = True,
    ) -> list[HistoricalLiquidationRecord]:
        """
        Download liquidation / force-order history if supported by the client.
        """

        self._ensure_client()

        symbol = symbol.upper()
        start_ms = timestamp_ms(start_time)
        end_ms = timestamp_ms(end_time)

        output_path = self._build_path(
            data_type=BacktestDataType.LIQUIDATIONS,
            symbol=symbol,
        )

        if self._should_skip_existing(output_path):
            return self._load_existing_stub(output_path, HistoricalLiquidationRecord)

        raw_rows: list[Any] = []
        cursor = start_ms

        while cursor < end_ms:
            rows = await self._call_with_retries(
                [
                    "get_force_orders",
                    "get_liquidations",
                    "fetch_liquidations",
                    "force_orders",
                ],
                symbol=symbol,
                start_time=cursor,
                startTime=cursor,
                end_time=end_ms,
                endTime=end_ms,
                limit=self.config.request_limit,
                market_type=self.config.market_type,
            )

            rows = self._normalize_raw_rows(rows)

            if not rows:
                break

            raw_rows.extend(rows)

            last_ts = self._extract_timestamp_ms(rows[-1], ["time", "timestamp", "updateTime"])
            next_cursor = last_ts + 1

            if next_cursor <= cursor:
                break

            cursor = next_cursor
            await self._sleep_rate_limit()

            if len(rows) < self.config.request_limit:
                break

        records = [
            self._normalize_liquidation(row, symbol=symbol)
            for row in raw_rows
        ]
        records = self._dedupe_by_key(
            records,
            key=lambda item: str(item.liquidation_id or item.timestamp_ms),
        )
        records = [
            item
            for item in records
            if start_ms <= item.timestamp_ms <= end_ms
        ]

        if save:
            self.save_records(records, output_path)

        return records

    # ---------------------------------------------------------------------
    # Storage
    # ---------------------------------------------------------------------

    def save_records(
        self,
        records: Sequence[SerializableMixin],
        path: str | Path,
    ) -> None:
        """
        Save records to configured storage format.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not self.config.overwrite_existing and not self.config.skip_existing:
            raise HistoricalDataStorageError(
                "Historical data file already exists and overwrite_existing=False.",
                details={"path": str(path)},
            )

        rows = [self._record_to_dict(record) for record in records]

        try:
            if self.config.output_format == HistoricalDataFormat.PARQUET:
                self._save_parquet(rows, path)
                return

            if self.config.output_format == HistoricalDataFormat.CSV:
                self._save_csv(rows, path)
                return

            if self.config.output_format == HistoricalDataFormat.JSON:
                self._save_json(rows, path)
                return

            if self.config.output_format == HistoricalDataFormat.JSONL:
                self._save_jsonl(rows, path)
                return

            raise HistoricalDataFormatError(
                "Unsupported historical data output format.",
                details={
                    "format": self.config.output_format.value,
                    "path": str(path),
                },
            )

        except HistoricalDataFormatError:
            raise
        except Exception as exc:
            raise HistoricalDataStorageError(
                "Failed to save historical records.",
                details={
                    "path": str(path),
                    "records": len(records),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            ) from exc

    def validate_downloaded_data(
        self,
        records: Sequence[SerializableMixin],
        *,
        expected_data_type: BacktestDataType,
    ) -> None:
        """
        Basic validation for downloaded records.
        """

        if not records:
            raise HistoricalDataValidationError(
                "Downloaded data is empty.",
                details={"data_type": expected_data_type.value},
            )

        timestamps: list[int] = []

        for record in records:
            value = getattr(record, "timestamp_ms", None)
            if value is None:
                value = getattr(record, "open_time_ms", None)

            if value is None:
                raise HistoricalDataValidationError(
                    "Downloaded record has no timestamp field.",
                    details={
                        "data_type": expected_data_type.value,
                        "record": self._record_to_dict(record),
                    },
                )

            timestamps.append(int(value))

        if timestamps != sorted(timestamps):
            raise HistoricalDataValidationError(
                "Downloaded records are not sorted by timestamp.",
                details={"data_type": expected_data_type.value},
            )

    # ---------------------------------------------------------------------
    # Normalizers
    # ---------------------------------------------------------------------

    def _normalize_candle(
        self,
        row: Any,
        *,
        symbol: str,
        timeframe: str,
    ) -> HistoricalCandle:
        item = self._as_mapping_or_sequence(row)

        if isinstance(item, dict):
            open_time_ms = self._int_from_keys(item, ["open_time_ms", "openTime", "open_time", "t", "timestamp"])
            close_time_ms = self._int_from_keys(
                item,
                ["close_time_ms", "closeTime", "close_time", "T"],
                default=open_time_ms,
            )
            open_price = self._float_from_keys(item, ["open", "o"])
            high = self._float_from_keys(item, ["high", "h"])
            low = self._float_from_keys(item, ["low", "l"])
            close = self._float_from_keys(item, ["close", "c"])
            volume = self._float_from_keys(item, ["volume", "v"], default=0.0)
            quote_volume = self._float_from_keys(item, ["quote_volume", "quoteVolume", "q"], default=0.0)
            trades_count = self._int_from_keys(item, ["trades_count", "numberOfTrades", "n"], default=0)
            is_closed = bool(item.get("is_closed", item.get("x", True)))
        else:
            # Binance kline REST array:
            # [open_time, open, high, low, close, volume, close_time,
            #  quote_asset_volume, number_of_trades, ...]
            values = list(item)
            open_time_ms = int(values[0])
            open_price = float(values[1])
            high = float(values[2])
            low = float(values[3])
            close = float(values[4])
            volume = float(values[5])
            close_time_ms = int(values[6]) if len(values) > 6 else open_time_ms
            quote_volume = float(values[7]) if len(values) > 7 else 0.0
            trades_count = int(values[8]) if len(values) > 8 else 0
            is_closed = True

        return HistoricalCandle(
            exchange=self.config.exchange,
            symbol=symbol,
            market_type=self.config.market_type,
            timeframe=timeframe,
            timestamp_ms=close_time_ms,
            received_at_ms=close_time_ms,
            open_time_ms=open_time_ms,
            close_time_ms=close_time_ms,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            quote_volume=quote_volume,
            trades_count=trades_count,
            is_closed=is_closed,
            source="history_downloader",
        )

    def _normalize_trade(
        self,
        row: Any,
        *,
        symbol: str,
    ) -> HistoricalTrade:
        item = self._as_mapping_or_sequence(row)

        if isinstance(item, dict):
            trade_id = item.get("trade_id", item.get("id", item.get("a")))
            price = self._float_from_keys(item, ["price", "p"])
            quantity = self._float_from_keys(item, ["quantity", "qty", "q"])
            timestamp = self._int_from_keys(item, ["timestamp_ms", "timestamp", "time", "T"])
            buyer_maker = item.get("buyer_maker", item.get("m"))
            side = item.get("side")
            aggressor_side = item.get("aggressor_side")
        else:
            values = list(item)
            trade_id = values[0] if len(values) > 0 else None
            price = float(values[1])
            quantity = float(values[2])
            timestamp = int(values[5]) if len(values) > 5 else int(values[3])
            buyer_maker = bool(values[6]) if len(values) > 6 else None
            side = None
            aggressor_side = None

        if aggressor_side is None and buyer_maker is not None:
            aggressor_side = "sell" if buyer_maker else "buy"

        return HistoricalTrade(
            exchange=self.config.exchange,
            symbol=symbol,
            market_type=self.config.market_type,
            timestamp_ms=timestamp,
            received_at_ms=timestamp,
            trade_id=trade_id,
            price=price,
            quantity=quantity,
            side=side,
            aggressor_side=aggressor_side,
            buyer_maker=buyer_maker,
            source="history_downloader",
        )

    def _normalize_funding(
        self,
        row: Any,
        *,
        symbol: str,
    ) -> HistoricalFundingRecord:
        item = self._as_dict(row)

        funding_time = self._int_from_keys(item, ["fundingTime", "funding_time", "timestamp", "time"])
        funding_rate = self._float_from_keys(item, ["fundingRate", "funding_rate", "rate"], default=0.0)

        return HistoricalFundingRecord(
            exchange=self.config.exchange,
            symbol=symbol,
            market_type=self.config.market_type,
            timestamp_ms=funding_time,
            received_at_ms=funding_time,
            funding_rate=funding_rate,
            predicted_rate=self._optional_float_from_keys(item, ["predictedRate", "predicted_rate"]),
            mark_price=self._optional_float_from_keys(item, ["markPrice", "mark_price"]),
            index_price=self._optional_float_from_keys(item, ["indexPrice", "index_price"]),
            next_funding_time_ms=self._optional_int_from_keys(item, ["nextFundingTime", "next_funding_time_ms"]),
            source="history_downloader",
        )

    def _normalize_open_interest(
        self,
        row: Any,
        *,
        symbol: str,
    ) -> HistoricalOpenInterestRecord:
        item = self._as_dict(row)

        ts = self._int_from_keys(item, ["timestamp", "time", "sumOpenInterestTime"])
        open_interest = self._float_from_keys(
            item,
            ["openInterest", "open_interest", "sumOpenInterest"],
            default=0.0,
        )
        open_interest_value = self._optional_float_from_keys(
            item,
            ["open_interest_value", "sumOpenInterestValue", "openInterestValue"],
        )

        return HistoricalOpenInterestRecord(
            exchange=self.config.exchange,
            symbol=symbol,
            market_type=self.config.market_type,
            timestamp_ms=ts,
            received_at_ms=ts,
            open_interest=open_interest,
            open_interest_value=open_interest_value,
            mark_price=self._optional_float_from_keys(item, ["markPrice", "mark_price"]),
            source="history_downloader",
        )

    def _normalize_orderbook_snapshot(
        self,
        row: Any,
        *,
        symbol: str,
        current_timestamp_ms: int,
    ) -> HistoricalOrderBookSnapshot:
        item = self._as_dict(row)

        bids = [
            HistoricalOrderBookLevel(price=float(level[0]), quantity=float(level[1]))
            for level in item.get("bids", [])
        ]
        asks = [
            HistoricalOrderBookLevel(price=float(level[0]), quantity=float(level[1]))
            for level in item.get("asks", [])
        ]

        ts = self._int_from_keys(
            item,
            ["timestamp_ms", "timestamp", "time", "lastUpdateTime"],
            default=current_timestamp_ms,
        )

        return HistoricalOrderBookSnapshot(
            exchange=self.config.exchange,
            symbol=symbol,
            market_type=self.config.market_type,
            timestamp_ms=ts,
            received_at_ms=ts,
            bids=bids,
            asks=asks,
            sequence=self._optional_int_from_keys(item, ["lastUpdateId", "sequence", "u"]),
            depth=max(len(bids), len(asks)),
            source="history_downloader",
        )

    def _normalize_liquidation(
        self,
        row: Any,
        *,
        symbol: str,
    ) -> HistoricalLiquidationRecord:
        item = self._as_dict(row)

        # Binance force order records may wrap order details in "o".
        if "o" in item and isinstance(item["o"], dict):
            order = item["o"]
        else:
            order = item

        ts = self._int_from_keys(order, ["time", "timestamp", "updateTime", "T"])
        price = self._float_from_keys(order, ["price", "p", "avgPrice", "ap"])
        quantity = self._float_from_keys(order, ["quantity", "qty", "q", "origQty"])
        side = str(order.get("side", order.get("S", ""))).lower()

        return HistoricalLiquidationRecord(
            exchange=self.config.exchange,
            symbol=symbol,
            market_type=self.config.market_type,
            timestamp_ms=ts,
            received_at_ms=ts,
            liquidation_id=order.get("orderId", order.get("id")),
            side=side,
            price=price,
            quantity=quantity,
            notional=price * quantity,
            source="history_downloader",
            metadata={"raw_side": order.get("side", order.get("S"))},
        )

    # ---------------------------------------------------------------------
    # REST client helpers
    # ---------------------------------------------------------------------

    async def _call_with_retries(
        self,
        method_names: Sequence[str],
        **kwargs: Any,
    ) -> Any:
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._call_client(method_names, **kwargs)

            except TypeError as exc:
                # Retry with cleaned kwargs when client does not accept aliases.
                last_error = exc
                cleaned_kwargs = self._clean_alias_kwargs(kwargs)

                try:
                    return await self._call_client(method_names, **cleaned_kwargs)
                except Exception as inner_exc:
                    last_error = inner_exc

            except Exception as exc:
                last_error = exc

            if attempt < self.config.max_retries:
                await asyncio.sleep(self.config.retry_delay_seconds)

        raise HistoricalDataDownloadError(
            "REST client call failed after retries.",
            details={
                "method_names": list(method_names),
                "kwargs": self._safe_kwargs(kwargs),
                "max_retries": self.config.max_retries,
                "error": str(last_error),
                "error_type": last_error.__class__.__name__ if last_error else None,
            },
        ) from last_error

    async def _call_client(
        self,
        method_names: Sequence[str],
        **kwargs: Any,
    ) -> Any:
        self._ensure_client()

        method = None
        method_name = None

        for candidate in method_names:
            if hasattr(self.rest_client, candidate):
                method = getattr(self.rest_client, candidate)
                method_name = candidate
                break

        if method is None:
            raise HistoricalDataDownloadError(
                "REST client does not expose any required historical data method.",
                details={
                    "method_names": list(method_names),
                    "client_type": self.rest_client.__class__.__name__,
                },
            )

        try:
            result = method(**kwargs)
        except TypeError:
            raise
        except Exception as exc:
            raise HistoricalDataDownloadError(
                "REST client method call failed.",
                details={
                    "method_name": method_name,
                    "kwargs": self._safe_kwargs(kwargs),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            ) from exc

        if hasattr(result, "__await__"):
            return await result

        return result

    def _ensure_client(self) -> None:
        if self.rest_client is None:
            raise HistoricalDataDownloadError(
                "HistoryDownloader requires an injected REST client."
            )

    @staticmethod
    def _clean_alias_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Remove common duplicate aliases after a TypeError.

        This is a compatibility fallback for clients that accept either
        snake_case or Binance-style camelCase parameters, but not both.
        """

        cleaned = dict(kwargs)

        alias_pairs = [
            ("start_time", "startTime"),
            ("end_time", "endTime"),
            ("timeframe", "interval"),
        ]

        for snake, camel in alias_pairs:
            if snake in cleaned and camel in cleaned:
                cleaned.pop(camel, None)

        return cleaned

    @staticmethod
    def _safe_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Remove potentially noisy values from error details.
        """

        return {
            key: value
            for key, value in kwargs.items()
            if key.lower() not in {"api_key", "secret", "signature", "password", "token"}
        }

    # ---------------------------------------------------------------------
    # Path / file helpers
    # ---------------------------------------------------------------------

    def _build_path(
        self,
        *,
        data_type: BacktestDataType,
        symbol: str,
        timeframe: str | None = None,
    ) -> Path:
        extension = self._format_extension(self.config.output_format)

        parts = [
            Path(self.config.output_dir),
            self.config.exchange,
            self.config.market_type,
            data_type.value,
            symbol.upper(),
        ]

        if timeframe:
            parts.append(timeframe)

        directory = Path(*parts)
        filename = f"{symbol.upper()}"

        if timeframe:
            filename += f"_{timeframe}"

        filename += f".{extension}"

        return directory / filename

    @staticmethod
    def _format_extension(value: HistoricalDataFormat) -> str:
        if value == HistoricalDataFormat.PARQUET:
            return "parquet"
        if value == HistoricalDataFormat.CSV:
            return "csv"
        if value == HistoricalDataFormat.JSON:
            return "json"
        if value == HistoricalDataFormat.JSONL:
            return "jsonl"

        raise HistoricalDataFormatError(
            "Unsupported output format.",
            details={"format": value.value},
        )

    def _should_skip_existing(self, path: Path) -> bool:
        return path.exists() and self.config.skip_existing and not self.config.overwrite_existing

    def _load_existing_stub(
        self,
        path: Path,
        record_type: type[Any],
    ) -> list[Any]:
        """
        Return an empty list when skipping existing files.

        DataLoader is responsible for reading historical files. The downloader
        only avoids re-downloading if skip_existing=True.
        """

        self.logger.info(
            "Historical data already exists, skipping download: %s (%s)",
            path,
            record_type.__name__,
        )
        return []

    @staticmethod
    def _save_parquet(rows: list[dict[str, Any]], path: Path) -> None:
        try:
            import pandas as pd
        except Exception as exc:
            raise HistoricalDataFormatError(
                "Saving parquet requires pandas and a parquet engine.",
                details={"path": str(path)},
            ) from exc

        dataframe = pd.DataFrame(rows)
        dataframe.to_parquet(path, index=False)

    @staticmethod
    def _save_csv(rows: list[dict[str, Any]], path: Path) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _save_json(rows: list[dict[str, Any]], path: Path) -> None:
        path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str))
                handle.write("\n")

    # ---------------------------------------------------------------------
    # Generic normalization helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _normalize_raw_rows(result: Any) -> list[Any]:
        if result is None:
            return []

        if isinstance(result, dict):
            for key in ("data", "rows", "items", "result", "list"):
                value = result.get(key)
                if isinstance(value, list):
                    return value

            # Some clients return {"symbol": ..., "bids": ..., "asks": ...}
            return [result]

        if isinstance(result, list):
            return result

        if isinstance(result, tuple):
            return list(result)

        return [result]

    @staticmethod
    def _as_dict(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return row

        if is_dataclass(row):
            return asdict(row)

        if hasattr(row, "to_dict"):
            return row.to_dict()

        if hasattr(row, "__dict__"):
            return dict(row.__dict__)

        raise HistoricalDataValidationError(
            "Cannot convert row to dict.",
            details={"row_type": row.__class__.__name__},
        )

    @staticmethod
    def _as_mapping_or_sequence(row: Any) -> dict[str, Any] | Sequence[Any]:
        if isinstance(row, dict):
            return row

        if is_dataclass(row):
            return asdict(row)

        if hasattr(row, "to_dict"):
            return row.to_dict()

        if isinstance(row, (list, tuple)):
            return row

        if hasattr(row, "__dict__"):
            return dict(row.__dict__)

        raise HistoricalDataValidationError(
            "Cannot normalize row.",
            details={"row_type": row.__class__.__name__},
        )

    @staticmethod
    def _record_to_dict(record: Any) -> dict[str, Any]:
        if hasattr(record, "to_dict"):
            return record.to_dict()

        if is_dataclass(record):
            return asdict(record)

        if isinstance(record, dict):
            return record

        if hasattr(record, "__dict__"):
            return dict(record.__dict__)

        raise HistoricalDataStorageError(
            "Cannot serialize historical record.",
            details={"record_type": record.__class__.__name__},
        )

    @staticmethod
    def _dedupe_by_key(
            records: Iterable[T],
            *,
            key: Callable[[T], Any],
    ) -> list[T]:
        seen: set[Any] = set()
        result: list[T] = []

        for record in records:
            value = key(record)
            if value in seen:
                continue

            seen.add(value)
            result.append(record)

        return result

    @staticmethod
    def _float_from_keys(
        row: dict[str, Any],
        keys: Sequence[str],
        *,
        default: float | None = None,
    ) -> float:
        for key in keys:
            value = row.get(key)
            if value is not None and value != "":
                return float(value)

        if default is not None:
            return default

        raise HistoricalDataValidationError(
            "Missing required float field.",
            details={"keys": list(keys), "row": row},
        )

    @staticmethod
    def _optional_float_from_keys(
        row: dict[str, Any],
        keys: Sequence[str],
    ) -> float | None:
        for key in keys:
            value = row.get(key)
            if value is not None and value != "":
                return float(value)
        return None

    @staticmethod
    def _int_from_keys(
        row: dict[str, Any],
        keys: Sequence[str],
        *,
        default: int | None = None,
    ) -> int:
        for key in keys:
            value = row.get(key)
            if value is not None and value != "":
                return int(float(value))

        if default is not None:
            return default

        raise HistoricalDataValidationError(
            "Missing required int field.",
            details={"keys": list(keys), "row": row},
        )

    @staticmethod
    def _optional_int_from_keys(
        row: dict[str, Any],
        keys: Sequence[str],
    ) -> int | None:
        for key in keys:
            value = row.get(key)
            if value is not None and value != "":
                return int(float(value))
        return None

    def _extract_timestamp_ms(
        self,
        row: Any,
        keys: Sequence[str],
    ) -> int:
        item = self._as_mapping_or_sequence(row)

        if isinstance(item, dict):
            return self._int_from_keys(item, keys)

        values = list(item)
        if values:
            return int(float(values[0]))

        raise HistoricalDataValidationError(
            "Cannot extract timestamp from row.",
            details={"row": row},
        )

    def _extract_candle_open_time_ms(self, row: Any) -> int:
        item = self._as_mapping_or_sequence(row)

        if isinstance(item, dict):
            return self._int_from_keys(item, ["open_time_ms", "openTime", "open_time", "t", "timestamp"])

        values = list(item)
        if values:
            return int(float(values[0]))

        raise HistoricalDataValidationError(
            "Cannot extract candle open time.",
            details={"row": row},
        )

    def _extract_trade_timestamp_ms(self, row: Any) -> int:
        item = self._as_mapping_or_sequence(row)

        if isinstance(item, dict):
            return self._int_from_keys(item, ["timestamp_ms", "timestamp", "time", "T"])

        values = list(item)
        if len(values) > 5:
            return int(float(values[5]))
        if len(values) > 3:
            return int(float(values[3]))

        raise HistoricalDataValidationError(
            "Cannot extract trade timestamp.",
            details={"row": row},
        )

    @staticmethod
    def _timeframe_to_milliseconds(timeframe: str) -> int:
        units = {
            "m": 60_000,
            "h": 60 * 60_000,
            "d": 24 * 60 * 60_000,
            "w": 7 * 24 * 60 * 60_000,
        }

        value = timeframe.strip().lower()

        if len(value) < 2:
            raise HistoricalDataValidationError(
                "Invalid timeframe.",
                details={"timeframe": timeframe},
            )

        amount = int(value[:-1])
        unit = value[-1]

        if unit not in units:
            raise HistoricalDataValidationError(
                "Unsupported timeframe unit.",
                details={"timeframe": timeframe},
            )

        return amount * units[unit]

    async def _sleep_rate_limit(self) -> None:
        if self.config.rate_limit_sleep_seconds > 0:
            await asyncio.sleep(self.config.rate_limit_sleep_seconds)

    async def _emit(self, topic: str, payload: dict[str, Any]) -> None:
        if self.event_bus is None:
            return

        emit = getattr(self.event_bus, "emit", None) or getattr(self.event_bus, "publish", None)

        if emit is None:
            return

        try:
            result = emit(topic, payload)

            if hasattr(result, "__await__"):
                await result

        except Exception as exc:
            self.logger.warning(
                "Failed to emit history downloader event %s: %s",
                topic,
                exc,
            )

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "exchange": self.config.exchange,
            "market_type": self.config.market_type,
            "symbols": self.config.symbols,
            "timeframes": self.config.timeframes,
            "data_types": [item.value for item in self.config.data_types],
            "output_dir": str(self.config.output_dir),
            "output_format": self.config.output_format.value,
            "has_rest_client": self.rest_client is not None,
        }


__all__ = [
    "HistoryDownloader",
]