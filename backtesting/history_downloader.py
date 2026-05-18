# trading_system/backtesting/history_downloader.py

from __future__ import annotations

import asyncio
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import aiohttp
import pandas as pd

from core.logger import get_logger

from .config import HistoryDownloadConfig
from .enums import (
    ExchangeName,
    HistoryDataType,
    HistoryDownloadStatus,
    MarketType,
    StorageFormat,
)
from .exceptions import (
    HistoryDownloadConfigError,
    HistoryDownloadError,
    HistoryNormalizationError,
    HistoryRequestError,
    HistoryResponseError,
    HistoryUnavailableError,
    HistoryWriteError,
    build_error_context,
)
from .models import (
    HistoryDownloadProgress,
    HistoryDownloadRequest,
    HistoryDownloadResult,
    NormalizedHistoryBatch,
    RawHistoryBatch,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


BINANCE_USDM_BASE_URL = "https://fapi.binance.com"
BINANCE_COINM_BASE_URL = "https://dapi.binance.com"

BYBIT_BASE_URL = "https://api.bybit.com"
OKX_BASE_URL = "https://www.okx.com"
MEXC_FUTURES_BASE_URL = "https://contract.mexc.com"

MS_IN_SECOND = 1_000
MS_IN_MINUTE = 60 * MS_IN_SECOND
MS_IN_HOUR = 60 * MS_IN_MINUTE
MS_IN_DAY = 24 * MS_IN_HOUR


TIMEFRAME_TO_MS: dict[str, int] = {
    "1m": MS_IN_MINUTE,
    "3m": 3 * MS_IN_MINUTE,
    "5m": 5 * MS_IN_MINUTE,
    "15m": 15 * MS_IN_MINUTE,
    "30m": 30 * MS_IN_MINUTE,
    "1h": MS_IN_HOUR,
    "2h": 2 * MS_IN_HOUR,
    "4h": 4 * MS_IN_HOUR,
    "6h": 6 * MS_IN_HOUR,
    "8h": 8 * MS_IN_HOUR,
    "12h": 12 * MS_IN_HOUR,
    "1d": MS_IN_DAY,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _month_partition(timestamp_ms: int) -> str:
    return time.strftime("%Y-%m", time.gmtime(timestamp_ms / 1000))


def _day_partition(timestamp_ms: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(timestamp_ms / 1000))


def _year_partition(timestamp_ms: int) -> str:
    return time.strftime("%Y", time.gmtime(timestamp_ms / 1000))


def _timeframe_ms(timeframe: str) -> int:
    if timeframe not in TIMEFRAME_TO_MS:
        raise HistoryDownloadConfigError(
            f"Unsupported timeframe: {timeframe}",
            context=build_error_context(timeframe=timeframe),
        )
    return TIMEFRAME_TO_MS[timeframe]


def _dedupe_rows(rows: Iterable[dict[str, Any]], key: str = "timestamp_ms") -> list[dict[str, Any]]:
    """
    Keep the last row for a timestamp key.
    """

    deduped: dict[Any, dict[str, Any]] = {}
    for row in rows:
        deduped[row.get(key)] = row
    return [deduped[k] for k in sorted(deduped) if k is not None]


def _chunk_time_range(
    *,
    start_time_ms: int,
    end_time_ms: int,
    chunk_ms: int,
) -> list[tuple[int, int]]:
    if start_time_ms >= end_time_ms:
        return []

    chunks: list[tuple[int, int]] = []
    current = start_time_ms

    while current < end_time_ms:
        chunk_end = min(current + chunk_ms, end_time_ms)
        chunks.append((current, chunk_end))
        current = chunk_end + 1

    return chunks


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ParquetHistoryWriter:
    """
    Writes normalized futures history into local partitioned Parquet files.

    Layout:
        data/history/{exchange}/{market_type}/{symbol}/{data_type}/...
    """

    output_dir: str = "data/history"
    compression: str = "snappy"
    logger_name: str = __name__

    _logger: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = get_logger(self.logger_name)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    async def write_batch(
        self,
        batch: NormalizedHistoryBatch,
        *,
        overwrite: bool = False,
    ) -> list[str]:
        if not batch.rows:
            return []

        try:
            grouped = self._group_rows_by_partition(batch)
            written_files: list[str] = []

            for partition, rows in grouped.items():
                path = self._build_output_path(batch, partition)
                path.parent.mkdir(parents=True, exist_ok=True)

                rows = _dedupe_rows(rows, key="timestamp_ms")
                df_new = pd.DataFrame(rows)

                if path.exists() and not overwrite:
                    try:
                        df_old = pd.read_parquet(path)
                        df_all = pd.concat([df_old, df_new], ignore_index=True)
                        if "timestamp_ms" in df_all.columns:
                            df_all = (
                                df_all.drop_duplicates(subset=["timestamp_ms"], keep="last")
                                .sort_values("timestamp_ms")
                                .reset_index(drop=True)
                            )
                        df_all.to_parquet(path, index=False, compression=self.compression)
                    except Exception as exc:
                        raise HistoryWriteError(
                            "Failed to merge existing Parquet history",
                            context=build_error_context(
                                exchange=batch.exchange,
                                market_type=batch.market_type,
                                symbol=batch.symbol,
                                timeframe=batch.timeframe,
                                data_type=batch.data_type,
                                data_path=str(path),
                            ),
                            cause=exc,
                        ) from exc
                else:
                    if "timestamp_ms" in df_new.columns:
                        df_new = df_new.sort_values("timestamp_ms").reset_index(drop=True)
                    df_new.to_parquet(path, index=False, compression=self.compression)

                written_files.append(str(path))

            return written_files

        except HistoryWriteError:
            raise
        except Exception as exc:
            raise HistoryWriteError(
                "Failed to write normalized history batch",
                context=build_error_context(
                    exchange=batch.exchange,
                    market_type=batch.market_type,
                    symbol=batch.symbol,
                    timeframe=batch.timeframe,
                    data_type=batch.data_type,
                ),
                cause=exc,
            ) from exc

    def _group_rows_by_partition(
        self,
        batch: NormalizedHistoryBatch,
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}

        for row in batch.rows:
            timestamp_ms = _safe_int(row.get("timestamp_ms"))

            if batch.data_type in {
                HistoryDataType.TRADES.value,
                HistoryDataType.AGG_TRADES.value,
                HistoryDataType.ORDERBOOK_SNAPSHOTS.value,
                HistoryDataType.ORDERBOOK_DELTAS.value,
                HistoryDataType.LIQUIDATIONS.value,
            }:
                partition = _day_partition(timestamp_ms)
            elif batch.data_type == HistoryDataType.FUNDING.value:
                partition = _year_partition(timestamp_ms)
            else:
                partition = _month_partition(timestamp_ms)

            grouped.setdefault(partition, []).append(row)

        return grouped

    def _build_output_path(
        self,
        batch: NormalizedHistoryBatch,
        partition: str,
    ) -> Path:
        base = (
            Path(self.output_dir)
            / batch.exchange
            / batch.market_type
            / batch.symbol
            / batch.data_type
        )

        if batch.data_type == HistoryDataType.CANDLES.value:
            if not batch.timeframe:
                raise HistoryWriteError(
                    "Candle batch requires timeframe",
                    context=build_error_context(
                        exchange=batch.exchange,
                        market_type=batch.market_type,
                        symbol=batch.symbol,
                        data_type=batch.data_type,
                    ),
                )
            return base / batch.timeframe / f"{partition}.parquet"

        return base / f"{partition}.parquet"


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


class HistoryDataNormalizer:
    """
    Normalizes raw exchange/provider data into internal market.* compatible rows.

    These rows are intentionally close to live exchange-adapter payloads.
    """

    def normalize_batch(self, batch: RawHistoryBatch) -> NormalizedHistoryBatch:
        exchange = batch.exchange
        data_type = batch.data_type

        if exchange == ExchangeName.BINANCE.value:
            rows = self._normalize_binance_batch(batch)
        elif exchange == ExchangeName.BYBIT.value:
            rows = self._normalize_bybit_batch(batch)
        elif exchange == ExchangeName.OKX.value:
            rows = self._normalize_okx_batch(batch)
        elif exchange == ExchangeName.MEXC.value:
            rows = self._normalize_mexc_batch(batch)
        else:
            raise HistoryNormalizationError(
                f"Unsupported exchange for normalization: {exchange}",
                context=build_error_context(
                    exchange=exchange,
                    market_type=batch.market_type,
                    symbol=batch.symbol,
                    data_type=data_type,
                    timeframe=batch.timeframe,
                ),
            )

        return NormalizedHistoryBatch(
            exchange=batch.exchange,
            market_type=batch.market_type,
            symbol=batch.symbol,
            data_type=batch.data_type,
            timeframe=batch.timeframe,
            start_time_ms=batch.start_time_ms,
            end_time_ms=batch.end_time_ms,
            rows=rows,
            metadata=dict(batch.metadata),
        )

    # -----------------------------
    # Binance
    # -----------------------------

    def _normalize_binance_batch(self, batch: RawHistoryBatch) -> list[dict[str, Any]]:
        if batch.data_type == HistoryDataType.CANDLES.value:
            return [self._normalize_binance_kline(row, batch) for row in batch.rows]

        if batch.data_type == HistoryDataType.AGG_TRADES.value:
            return [self._normalize_binance_agg_trade(row, batch) for row in batch.rows]

        if batch.data_type == HistoryDataType.FUNDING.value:
            return [self._normalize_binance_funding(row, batch) for row in batch.rows]

        if batch.data_type == HistoryDataType.OPEN_INTEREST.value:
            return [self._normalize_binance_open_interest(row, batch) for row in batch.rows]

        raise HistoryNormalizationError(
            f"Unsupported Binance data type: {batch.data_type}",
            context=build_error_context(
                exchange=batch.exchange,
                market_type=batch.market_type,
                symbol=batch.symbol,
                data_type=batch.data_type,
            ),
        )

    def _normalize_binance_kline(
        self,
        row: list[Any],
        batch: RawHistoryBatch,
    ) -> dict[str, Any]:
        try:
            open_time_ms = _safe_int(row[0])
            close_time_ms = _safe_int(row[6])

            return {
                "exchange": batch.exchange,
                "symbol": batch.symbol,
                "market_type": batch.market_type,
                "timeframe": batch.timeframe,
                "open_time_ms": open_time_ms,
                "close_time_ms": close_time_ms,
                "open": _safe_float(row[1]),
                "high": _safe_float(row[2]),
                "low": _safe_float(row[3]),
                "close": _safe_float(row[4]),
                "volume": _safe_float(row[5]),
                "quote_volume": _safe_float(row[7]),
                "trades_count": _safe_int(row[8]),
                "taker_buy_base_volume": _safe_float(row[9]),
                "taker_buy_quote_volume": _safe_float(row[10]),
                "is_closed": True,
                "timestamp_ms": close_time_ms,
                "received_at_ms": close_time_ms,
            }
        except Exception as exc:
            raise HistoryNormalizationError(
                "Failed to normalize Binance kline",
                context=build_error_context(
                    exchange=batch.exchange,
                    market_type=batch.market_type,
                    symbol=batch.symbol,
                    timeframe=batch.timeframe,
                    data_type=batch.data_type,
                    raw_row=row,
                ),
                cause=exc,
            ) from exc

    def _normalize_binance_agg_trade(
        self,
        row: dict[str, Any],
        batch: RawHistoryBatch,
    ) -> dict[str, Any]:
        try:
            ts = _safe_int(row.get("T"))
            buyer_is_maker = bool(row.get("m"))
            aggressor_side = "sell" if buyer_is_maker else "buy"

            return {
                "exchange": batch.exchange,
                "symbol": batch.symbol,
                "market_type": batch.market_type,
                "trade_id": str(row.get("a")),
                "first_trade_id": str(row.get("f")),
                "last_trade_id": str(row.get("l")),
                "price": _safe_float(row.get("p")),
                "quantity": _safe_float(row.get("q")),
                "side": aggressor_side,
                "aggressor_side": aggressor_side,
                "buyer_is_maker": buyer_is_maker,
                "timestamp_ms": ts,
                "received_at_ms": ts,
            }
        except Exception as exc:
            raise HistoryNormalizationError(
                "Failed to normalize Binance aggregate trade",
                context=build_error_context(
                    exchange=batch.exchange,
                    market_type=batch.market_type,
                    symbol=batch.symbol,
                    data_type=batch.data_type,
                    raw_row=row,
                ),
                cause=exc,
            ) from exc

    def _normalize_binance_funding(
        self,
        row: dict[str, Any],
        batch: RawHistoryBatch,
    ) -> dict[str, Any]:
        try:
            ts = _safe_int(row.get("fundingTime"))

            return {
                "exchange": batch.exchange,
                "symbol": batch.symbol,
                "market_type": batch.market_type,
                "funding_rate": _safe_float(row.get("fundingRate")),
                "mark_price": _safe_float(row.get("markPrice")),
                "timestamp_ms": ts,
                "received_at_ms": ts,
            }
        except Exception as exc:
            raise HistoryNormalizationError(
                "Failed to normalize Binance funding row",
                context=build_error_context(
                    exchange=batch.exchange,
                    market_type=batch.market_type,
                    symbol=batch.symbol,
                    data_type=batch.data_type,
                    raw_row=row,
                ),
                cause=exc,
            ) from exc

    def _normalize_binance_open_interest(
        self,
        row: dict[str, Any],
        batch: RawHistoryBatch,
    ) -> dict[str, Any]:
        try:
            ts = _safe_int(row.get("timestamp"))

            return {
                "exchange": batch.exchange,
                "symbol": batch.symbol,
                "market_type": batch.market_type,
                "open_interest": _safe_float(row.get("sumOpenInterest")),
                "open_interest_value": _safe_float(row.get("sumOpenInterestValue")),
                "timestamp_ms": ts,
                "received_at_ms": ts,
            }
        except Exception as exc:
            raise HistoryNormalizationError(
                "Failed to normalize Binance open interest row",
                context=build_error_context(
                    exchange=batch.exchange,
                    market_type=batch.market_type,
                    symbol=batch.symbol,
                    data_type=batch.data_type,
                    raw_row=row,
                ),
                cause=exc,
            ) from exc

    # -----------------------------
    # Bybit placeholders with real normalized shape
    # -----------------------------

    def _normalize_bybit_batch(self, batch: RawHistoryBatch) -> list[dict[str, Any]]:
        if batch.data_type == HistoryDataType.CANDLES.value:
            return [self._normalize_bybit_kline(row, batch) for row in batch.rows]

        if batch.data_type == HistoryDataType.FUNDING.value:
            return [self._normalize_bybit_funding(row, batch) for row in batch.rows]

        if batch.data_type == HistoryDataType.OPEN_INTEREST.value:
            return [self._normalize_bybit_open_interest(row, batch) for row in batch.rows]

        raise HistoryNormalizationError(
            f"Unsupported Bybit data type: {batch.data_type}",
            context=build_error_context(
                exchange=batch.exchange,
                market_type=batch.market_type,
                symbol=batch.symbol,
                data_type=batch.data_type,
            ),
        )

    def _normalize_bybit_kline(self, row: list[Any] | dict[str, Any], batch: RawHistoryBatch) -> dict[str, Any]:
        if isinstance(row, dict):
            start = _safe_int(row.get("start") or row.get("startTime"))
            open_ = row.get("open")
            high = row.get("high")
            low = row.get("low")
            close = row.get("close")
            volume = row.get("volume")
            turnover = row.get("turnover")
        else:
            start = _safe_int(row[0])
            open_, high, low, close, volume, turnover = row[1:7]

        tf_ms = _timeframe_ms(batch.timeframe or "1m")
        close_time_ms = start + tf_ms - 1

        return {
            "exchange": batch.exchange,
            "symbol": batch.symbol,
            "market_type": batch.market_type,
            "timeframe": batch.timeframe,
            "open_time_ms": start,
            "close_time_ms": close_time_ms,
            "open": _safe_float(open_),
            "high": _safe_float(high),
            "low": _safe_float(low),
            "close": _safe_float(close),
            "volume": _safe_float(volume),
            "quote_volume": _safe_float(turnover),
            "trades_count": 0,
            "is_closed": True,
            "timestamp_ms": close_time_ms,
            "received_at_ms": close_time_ms,
        }

    def _normalize_bybit_funding(self, row: dict[str, Any], batch: RawHistoryBatch) -> dict[str, Any]:
        ts = _safe_int(row.get("fundingRateTimestamp") or row.get("timestamp"))
        return {
            "exchange": batch.exchange,
            "symbol": batch.symbol,
            "market_type": batch.market_type,
            "funding_rate": _safe_float(row.get("fundingRate")),
            "timestamp_ms": ts,
            "received_at_ms": ts,
        }

    def _normalize_bybit_open_interest(self, row: dict[str, Any], batch: RawHistoryBatch) -> dict[str, Any]:
        ts = _safe_int(row.get("timestamp"))
        return {
            "exchange": batch.exchange,
            "symbol": batch.symbol,
            "market_type": batch.market_type,
            "open_interest": _safe_float(row.get("openInterest")),
            "open_interest_value": _safe_float(row.get("openInterestValue")),
            "timestamp_ms": ts,
            "received_at_ms": ts,
        }

    # -----------------------------
    # OKX placeholders with real normalized shape
    # -----------------------------

    def _normalize_okx_batch(self, batch: RawHistoryBatch) -> list[dict[str, Any]]:
        if batch.data_type == HistoryDataType.CANDLES.value:
            return [self._normalize_okx_kline(row, batch) for row in batch.rows]

        if batch.data_type == HistoryDataType.FUNDING.value:
            return [self._normalize_okx_funding(row, batch) for row in batch.rows]

        if batch.data_type == HistoryDataType.OPEN_INTEREST.value:
            return [self._normalize_okx_open_interest(row, batch) for row in batch.rows]

        raise HistoryNormalizationError(
            f"Unsupported OKX data type: {batch.data_type}",
            context=build_error_context(
                exchange=batch.exchange,
                market_type=batch.market_type,
                symbol=batch.symbol,
                data_type=batch.data_type,
            ),
        )

    def _normalize_okx_kline(self, row: list[Any], batch: RawHistoryBatch) -> dict[str, Any]:
        ts = _safe_int(row[0])
        tf_ms = _timeframe_ms(batch.timeframe or "1m")
        close_time_ms = ts + tf_ms - 1

        return {
            "exchange": batch.exchange,
            "symbol": batch.symbol,
            "market_type": batch.market_type,
            "timeframe": batch.timeframe,
            "open_time_ms": ts,
            "close_time_ms": close_time_ms,
            "open": _safe_float(row[1]),
            "high": _safe_float(row[2]),
            "low": _safe_float(row[3]),
            "close": _safe_float(row[4]),
            "volume": _safe_float(row[5]),
            "quote_volume": _safe_float(row[7]) if len(row) > 7 else 0.0,
            "trades_count": 0,
            "is_closed": True,
            "timestamp_ms": close_time_ms,
            "received_at_ms": close_time_ms,
        }

    def _normalize_okx_funding(self, row: dict[str, Any], batch: RawHistoryBatch) -> dict[str, Any]:
        ts = _safe_int(row.get("fundingTime") or row.get("ts"))
        return {
            "exchange": batch.exchange,
            "symbol": batch.symbol,
            "market_type": batch.market_type,
            "funding_rate": _safe_float(row.get("fundingRate")),
            "timestamp_ms": ts,
            "received_at_ms": ts,
        }

    def _normalize_okx_open_interest(self, row: dict[str, Any], batch: RawHistoryBatch) -> dict[str, Any]:
        ts = _safe_int(row.get("ts"))
        return {
            "exchange": batch.exchange,
            "symbol": batch.symbol,
            "market_type": batch.market_type,
            "open_interest": _safe_float(row.get("oi")),
            "open_interest_value": _safe_float(row.get("oiCcy")),
            "timestamp_ms": ts,
            "received_at_ms": ts,
        }

    # -----------------------------
    # MEXC placeholders with real normalized shape
    # -----------------------------

    def _normalize_mexc_batch(self, batch: RawHistoryBatch) -> list[dict[str, Any]]:
        if batch.data_type == HistoryDataType.CANDLES.value:
            return [self._normalize_mexc_kline(row, batch) for row in batch.rows]

        raise HistoryNormalizationError(
            f"Unsupported MEXC data type: {batch.data_type}",
            context=build_error_context(
                exchange=batch.exchange,
                market_type=batch.market_type,
                symbol=batch.symbol,
                data_type=batch.data_type,
            ),
        )

    def _normalize_mexc_kline(self, row: dict[str, Any], batch: RawHistoryBatch) -> dict[str, Any]:
        open_time_ms = _safe_int(row.get("time")) * 1000
        tf_ms = _timeframe_ms(batch.timeframe or "1m")
        close_time_ms = open_time_ms + tf_ms - 1

        return {
            "exchange": batch.exchange,
            "symbol": batch.symbol,
            "market_type": batch.market_type,
            "timeframe": batch.timeframe,
            "open_time_ms": open_time_ms,
            "close_time_ms": close_time_ms,
            "open": _safe_float(row.get("open")),
            "high": _safe_float(row.get("high")),
            "low": _safe_float(row.get("low")),
            "close": _safe_float(row.get("close")),
            "volume": _safe_float(row.get("vol")),
            "quote_volume": _safe_float(row.get("amount")),
            "trades_count": 0,
            "is_closed": True,
            "timestamp_ms": close_time_ms,
            "received_at_ms": close_time_ms,
        }


# ---------------------------------------------------------------------------
# Base downloader
# ---------------------------------------------------------------------------


class BaseFuturesHistoryDownloader(ABC):
    """
    Base class for futures history downloaders.

    Downloader responsibility:
    - call exchange/provider REST endpoints;
    - return RawHistoryBatch;
    - do not use EventBus;
    - do not call strategy/risk/execution modules.
    """

    exchange: ExchangeName

    def __init__(
        self,
        *,
        config: HistoryDownloadConfig,
        session: aiohttp.ClientSession,
    ) -> None:
        self.config = config
        self.session = session
        self.logger = get_logger(self.__class__.__module__)

    @abstractmethod
    async def download_candles(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        raise NotImplementedError

    async def download_agg_trades(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        raise HistoryUnavailableError(
            f"{self.exchange.value} aggregate trades download is not implemented",
            context=build_error_context(
                exchange=self.exchange.value,
                market_type=_enum_value(self.config.market_type),
                symbol=symbol,
                data_type=HistoryDataType.AGG_TRADES.value,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
            ),
        )

    async def download_funding(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        raise HistoryUnavailableError(
            f"{self.exchange.value} funding download is not implemented",
            context=build_error_context(
                exchange=self.exchange.value,
                market_type=_enum_value(self.config.market_type),
                symbol=symbol,
                data_type=HistoryDataType.FUNDING.value,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
            ),
        )

    async def download_open_interest(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        raise HistoryUnavailableError(
            f"{self.exchange.value} open interest download is not implemented",
            context=build_error_context(
                exchange=self.exchange.value,
                market_type=_enum_value(self.config.market_type),
                symbol=symbol,
                data_type=HistoryDataType.OPEN_INTEREST.value,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
            ),
        )

    async def download_liquidations(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        raise HistoryUnavailableError(
            f"{self.exchange.value} liquidations download is not implemented",
            context=build_error_context(
                exchange=self.exchange.value,
                market_type=_enum_value(self.config.market_type),
                symbol=symbol,
                data_type=HistoryDataType.LIQUIDATIONS.value,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
            ),
        )

    async def _request_json(
        self,
        *,
        url: str,
        params: dict[str, Any],
    ) -> Any:
        last_exc: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                await asyncio.sleep(self.config.rate_limit_delay_sec)

                query = urlencode({k: v for k, v in params.items() if v is not None})
                full_url = f"{url}?{query}" if query else url

                async with self.session.get(
                    full_url,
                    timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_sec),
                    headers={
                        "User-Agent": self.config.user_agent,
                        **self.config.extra_headers,
                    },
                ) as response:
                    text = await response.text()

                    if response.status in {418, 429}:
                        raise HistoryResponseError(
                            "Exchange rate limit reached",
                            context=build_error_context(
                                exchange=self.exchange.value,
                                market_type=_enum_value(self.config.market_type),
                                url=url,
                                status=response.status,
                            ),
                        )

                    if response.status >= 400:
                        raise HistoryResponseError(
                            "Exchange returned HTTP error",
                            context=build_error_context(
                                exchange=self.exchange.value,
                                market_type=_enum_value(self.config.market_type),
                                url=url,
                                status=response.status,
                                response=text[:500],
                            ),
                        )

                    try:
                        return await response.json()
                    except Exception as exc:
                        raise HistoryResponseError(
                            "Failed to decode exchange JSON response",
                            context=build_error_context(
                                exchange=self.exchange.value,
                                market_type=_enum_value(self.config.market_type),
                                url=url,
                                response=text[:500],
                            ),
                            cause=exc,
                        ) from exc

            except Exception as exc:
                last_exc = exc
                if attempt >= self.config.max_retries:
                    break

                delay = self.config.retry_delay_sec * attempt
                self.logger.warning(
                    "History request failed, retrying",
                    extra={
                        "exchange": self.exchange.value,
                        "attempt": attempt,
                        "delay_sec": delay,
                        "url": url,
                    },
                )
                await asyncio.sleep(delay)

        raise HistoryRequestError(
            "History request failed after retries",
            context=build_error_context(
                exchange=self.exchange.value,
                market_type=_enum_value(self.config.market_type),
                url=url,
                params=params,
            ),
            cause=last_exc,
        )


# ---------------------------------------------------------------------------
# Binance downloader
# ---------------------------------------------------------------------------


class BinanceFuturesHistoryDownloader(BaseFuturesHistoryDownloader):
    exchange = ExchangeName.BINANCE

    @property
    def _base_url(self) -> str:
        market_type = _enum_value(self.config.market_type)
        if market_type == MarketType.COINM_FUTURES.value:
            return BINANCE_COINM_BASE_URL
        return BINANCE_USDM_BASE_URL

    async def download_candles(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        rows: list[Any] = []
        tf_ms = _timeframe_ms(timeframe)
        limit = min(self.config.max_rows_per_request, 1500)
        step_ms = tf_ms * limit

        for chunk_start, chunk_end in _chunk_time_range(
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            chunk_ms=step_ms,
        ):
            data = await self._request_json(
                url=f"{self._base_url}/fapi/v1/klines",
                params={
                    "symbol": symbol,
                    "interval": timeframe,
                    "startTime": chunk_start,
                    "endTime": chunk_end,
                    "limit": limit,
                },
            )

            if not isinstance(data, list):
                raise HistoryResponseError(
                    "Unexpected Binance klines response",
                    context=build_error_context(
                        exchange=self.exchange.value,
                        market_type=_enum_value(self.config.market_type),
                        symbol=symbol,
                        timeframe=timeframe,
                        data_type=HistoryDataType.CANDLES.value,
                    ),
                )

            rows.extend(data)

            if len(data) < limit:
                continue

        return RawHistoryBatch(
            exchange=self.exchange.value,
            market_type=_enum_value(self.config.market_type),
            symbol=symbol,
            data_type=HistoryDataType.CANDLES.value,
            timeframe=timeframe,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            rows=rows,
            source="binance_futures_rest",
        )

    async def download_agg_trades(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        rows: list[Any] = []
        limit = min(self.config.max_rows_per_request, 1000)
        current = start_time_ms

        while current < end_time_ms:
            data = await self._request_json(
                url=f"{self._base_url}/fapi/v1/aggTrades",
                params={
                    "symbol": symbol,
                    "startTime": current,
                    "endTime": end_time_ms,
                    "limit": limit,
                },
            )

            if not isinstance(data, list):
                raise HistoryResponseError(
                    "Unexpected Binance aggTrades response",
                    context=build_error_context(
                        exchange=self.exchange.value,
                        market_type=_enum_value(self.config.market_type),
                        symbol=symbol,
                        data_type=HistoryDataType.AGG_TRADES.value,
                    ),
                )

            if not data:
                break

            rows.extend(data)

            last_ts = _safe_int(data[-1].get("T"))
            if last_ts <= current:
                break
            current = last_ts + 1

            if len(data) < limit:
                break

        return RawHistoryBatch(
            exchange=self.exchange.value,
            market_type=_enum_value(self.config.market_type),
            symbol=symbol,
            data_type=HistoryDataType.AGG_TRADES.value,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            rows=rows,
            source="binance_futures_rest",
        )

    async def download_funding(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        rows: list[Any] = []
        limit = min(self.config.max_rows_per_request, 1000)
        current = start_time_ms

        while current < end_time_ms:
            data = await self._request_json(
                url=f"{self._base_url}/fapi/v1/fundingRate",
                params={
                    "symbol": symbol,
                    "startTime": current,
                    "endTime": end_time_ms,
                    "limit": limit,
                },
            )

            if not isinstance(data, list):
                raise HistoryResponseError(
                    "Unexpected Binance funding response",
                    context=build_error_context(
                        exchange=self.exchange.value,
                        market_type=_enum_value(self.config.market_type),
                        symbol=symbol,
                        data_type=HistoryDataType.FUNDING.value,
                    ),
                )

            if not data:
                break

            rows.extend(data)

            last_ts = _safe_int(data[-1].get("fundingTime"))
            if last_ts <= current:
                break
            current = last_ts + 1

            if len(data) < limit:
                break

        return RawHistoryBatch(
            exchange=self.exchange.value,
            market_type=_enum_value(self.config.market_type),
            symbol=symbol,
            data_type=HistoryDataType.FUNDING.value,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            rows=rows,
            source="binance_futures_rest",
        )

    async def download_open_interest(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        """
        Binance historical open interest statistics.

        Uses a 5m period by default because it is suitable for backtesting and
        avoids pretending that current-only OI is historical data.
        """

        rows: list[Any] = []
        limit = min(self.config.max_rows_per_request, 500)
        period = "5m"
        step_ms = 5 * MS_IN_MINUTE * limit

        for chunk_start, chunk_end in _chunk_time_range(
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            chunk_ms=step_ms,
        ):
            data = await self._request_json(
                url=f"{self._base_url}/futures/data/openInterestHist",
                params={
                    "symbol": symbol,
                    "period": period,
                    "startTime": chunk_start,
                    "endTime": chunk_end,
                    "limit": limit,
                },
            )

            if not isinstance(data, list):
                raise HistoryResponseError(
                    "Unexpected Binance open interest response",
                    context=build_error_context(
                        exchange=self.exchange.value,
                        market_type=_enum_value(self.config.market_type),
                        symbol=symbol,
                        data_type=HistoryDataType.OPEN_INTEREST.value,
                    ),
                )

            rows.extend(data)

        return RawHistoryBatch(
            exchange=self.exchange.value,
            market_type=_enum_value(self.config.market_type),
            symbol=symbol,
            data_type=HistoryDataType.OPEN_INTEREST.value,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            rows=rows,
            source="binance_futures_rest",
            metadata={"period": period},
        )


# ---------------------------------------------------------------------------
# Other exchange downloaders
# ---------------------------------------------------------------------------


class BybitFuturesHistoryDownloader(BaseFuturesHistoryDownloader):
    """
    Compact Bybit downloader skeleton.

    Candle/funding/OI methods are intentionally structured and ready for
    implementation, but Binance should be treated as the first MVP source.
    """

    exchange = ExchangeName.BYBIT

    async def download_candles(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        rows: list[Any] = []
        limit = min(self.config.max_rows_per_request, 1000)
        category = self._category()

        interval = self._to_bybit_interval(timeframe)
        current = start_time_ms

        while current < end_time_ms:
            data = await self._request_json(
                url=f"{BYBIT_BASE_URL}/v5/market/kline",
                params={
                    "category": category,
                    "symbol": symbol,
                    "interval": interval,
                    "start": current,
                    "end": end_time_ms,
                    "limit": limit,
                },
            )

            result = data.get("result", {}) if isinstance(data, dict) else {}
            batch_rows = result.get("list", [])

            if not isinstance(batch_rows, list):
                raise HistoryResponseError(
                    "Unexpected Bybit kline response",
                    context=build_error_context(
                        exchange=self.exchange.value,
                        market_type=_enum_value(self.config.market_type),
                        symbol=symbol,
                        timeframe=timeframe,
                        data_type=HistoryDataType.CANDLES.value,
                    ),
                )

            if not batch_rows:
                break

            rows.extend(batch_rows)
            last_ts = max(_safe_int(item[0]) for item in batch_rows)
            next_ts = last_ts + _timeframe_ms(timeframe)

            if next_ts <= current:
                break
            current = next_ts

            if len(batch_rows) < limit:
                break

        return RawHistoryBatch(
            exchange=self.exchange.value,
            market_type=_enum_value(self.config.market_type),
            symbol=symbol,
            data_type=HistoryDataType.CANDLES.value,
            timeframe=timeframe,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            rows=rows,
            source="bybit_v5_rest",
        )

    async def download_funding(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        data = await self._request_json(
            url=f"{BYBIT_BASE_URL}/v5/market/funding/history",
            params={
                "category": self._category(),
                "symbol": symbol,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": min(self.config.max_rows_per_request, 200),
            },
        )
        rows = data.get("result", {}).get("list", []) if isinstance(data, dict) else []

        return RawHistoryBatch(
            exchange=self.exchange.value,
            market_type=_enum_value(self.config.market_type),
            symbol=symbol,
            data_type=HistoryDataType.FUNDING.value,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            rows=rows,
            source="bybit_v5_rest",
        )

    async def download_open_interest(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        data = await self._request_json(
            url=f"{BYBIT_BASE_URL}/v5/market/open-interest",
            params={
                "category": self._category(),
                "symbol": symbol,
                "intervalTime": "5min",
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": min(self.config.max_rows_per_request, 200),
            },
        )
        rows = data.get("result", {}).get("list", []) if isinstance(data, dict) else []

        return RawHistoryBatch(
            exchange=self.exchange.value,
            market_type=_enum_value(self.config.market_type),
            symbol=symbol,
            data_type=HistoryDataType.OPEN_INTEREST.value,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            rows=rows,
            source="bybit_v5_rest",
        )

    def _category(self) -> str:
        market_type = _enum_value(self.config.market_type)
        if market_type == MarketType.INVERSE.value:
            return "inverse"
        return "linear"

    def _to_bybit_interval(self, timeframe: str) -> str:
        mapping = {
            "1m": "1",
            "3m": "3",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "6h": "360",
            "12h": "720",
            "1d": "D",
        }
        if timeframe not in mapping:
            raise HistoryDownloadConfigError(
                f"Unsupported Bybit timeframe: {timeframe}",
                context=build_error_context(timeframe=timeframe),
            )
        return mapping[timeframe]


class OKXFuturesHistoryDownloader(BaseFuturesHistoryDownloader):
    exchange = ExchangeName.OKX

    async def download_candles(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        rows: list[Any] = []
        bar = self._to_okx_bar(timeframe)
        limit = min(self.config.max_rows_per_request, 100)
        current_after = end_time_ms

        while current_after > start_time_ms:
            data = await self._request_json(
                url=f"{OKX_BASE_URL}/api/v5/market/history-candles",
                params={
                    "instId": symbol,
                    "bar": bar,
                    "after": current_after,
                    "limit": limit,
                },
            )

            batch_rows = data.get("data", []) if isinstance(data, dict) else []

            if not batch_rows:
                break

            normalized_rows = [
                row for row in batch_rows if start_time_ms <= _safe_int(row[0]) <= end_time_ms
            ]
            rows.extend(normalized_rows)

            oldest_ts = min(_safe_int(row[0]) for row in batch_rows)
            if oldest_ts >= current_after:
                break

            current_after = oldest_ts - 1

            if len(batch_rows) < limit:
                break

        return RawHistoryBatch(
            exchange=self.exchange.value,
            market_type=_enum_value(self.config.market_type),
            symbol=symbol,
            data_type=HistoryDataType.CANDLES.value,
            timeframe=timeframe,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            rows=rows,
            source="okx_rest",
        )

    def _to_okx_bar(self, timeframe: str) -> str:
        mapping = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1H",
            "2h": "2H",
            "4h": "4H",
            "6h": "6H",
            "12h": "12H",
            "1d": "1D",
        }
        if timeframe not in mapping:
            raise HistoryDownloadConfigError(
                f"Unsupported OKX timeframe: {timeframe}",
                context=build_error_context(timeframe=timeframe),
            )
        return mapping[timeframe]


class MEXCFuturesHistoryDownloader(BaseFuturesHistoryDownloader):
    exchange = ExchangeName.MEXC

    async def download_candles(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> RawHistoryBatch:
        rows: list[Any] = []
        interval = self._to_mexc_interval(timeframe)

        chunks = _chunk_time_range(
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            chunk_ms=30 * MS_IN_DAY,
        )

        for chunk_start, chunk_end in chunks:
            data = await self._request_json(
                url=f"{MEXC_FUTURES_BASE_URL}/api/v1/contract/kline/{symbol}",
                params={
                    "interval": interval,
                    "start": math.floor(chunk_start / 1000),
                    "end": math.floor(chunk_end / 1000),
                },
            )

            raw = data.get("data", {}) if isinstance(data, dict) else {}
            times = raw.get("time", [])
            opens = raw.get("open", [])
            highs = raw.get("high", [])
            lows = raw.get("low", [])
            closes = raw.get("close", [])
            vols = raw.get("vol", [])
            amounts = raw.get("amount", [])

            for idx, ts in enumerate(times):
                rows.append(
                    {
                        "time": ts,
                        "open": opens[idx] if idx < len(opens) else None,
                        "high": highs[idx] if idx < len(highs) else None,
                        "low": lows[idx] if idx < len(lows) else None,
                        "close": closes[idx] if idx < len(closes) else None,
                        "vol": vols[idx] if idx < len(vols) else None,
                        "amount": amounts[idx] if idx < len(amounts) else None,
                    }
                )

        return RawHistoryBatch(
            exchange=self.exchange.value,
            market_type=_enum_value(self.config.market_type),
            symbol=symbol,
            data_type=HistoryDataType.CANDLES.value,
            timeframe=timeframe,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            rows=rows,
            source="mexc_futures_rest",
        )

    def _to_mexc_interval(self, timeframe: str) -> str:
        mapping = {
            "1m": "Min1",
            "5m": "Min5",
            "15m": "Min15",
            "30m": "Min30",
            "1h": "Min60",
            "4h": "Hour4",
            "1d": "Day1",
        }
        if timeframe not in mapping:
            raise HistoryDownloadConfigError(
                f"Unsupported MEXC timeframe: {timeframe}",
                context=build_error_context(timeframe=timeframe),
            )
        return mapping[timeframe]


# ---------------------------------------------------------------------------
# Download manager
# ---------------------------------------------------------------------------


class HistoryDownloadManager:
    """
    Orchestrates historical futures data ingestion.

    Responsibilities:
    - choose exchange downloader;
    - download raw historical batches;
    - normalize to internal payload-compatible rows;
    - write local Parquet files;
    - return structured HistoryDownloadResult.

    It must not:
    - publish EventBus events;
    - call analytics/strategy/risk/execution modules;
    - run backtests.
    """

    def __init__(
        self,
        *,
        config: HistoryDownloadConfig,
        normalizer: HistoryDataNormalizer | None = None,
        writer: ParquetHistoryWriter | None = None,
    ) -> None:
        self.config = config
        self.normalizer = normalizer or HistoryDataNormalizer()
        self.writer = writer or ParquetHistoryWriter(output_dir=config.output_dir)
        self.logger = get_logger(__name__)

    async def download(self) -> HistoryDownloadResult:
        self.config.validate()

        request = HistoryDownloadRequest(
            exchange=self.config.exchange,
            market_type=self.config.market_type,
            symbols=list(self.config.symbols),
            start_time_ms=self.config.start_time_ms,
            end_time_ms=self.config.end_time_ms,
            data_types=list(self.config.data_types),
            timeframes=list(self.config.timeframes),
            source_type=self.config.source_type,
            storage_format=self.config.storage_format,
            output_dir=self.config.output_dir,
            overwrite_existing=self.config.overwrite_existing,
            validate_after_download=self.config.validate_after_download,
        )

        started_at_ms = _now_ms()
        files_written: list[str] = []
        downloaded_rows = 0
        written_rows = 0
        failed_rows = 0
        errors: list[dict[str, Any]] = []

        status: HistoryDownloadStatus | str = HistoryDownloadStatus.RUNNING

        self.logger.info(
            "Starting history download",
            extra={
                "exchange": _enum_value(self.config.exchange),
                "market_type": _enum_value(self.config.market_type),
                "symbols": self.config.symbols,
                "data_types": [_enum_value(item) for item in self.config.data_types],
                "start_time_ms": self.config.start_time_ms,
                "end_time_ms": self.config.end_time_ms,
            },
        )

        try:
            connector = aiohttp.TCPConnector(limit_per_host=max(1, self.config.max_concurrent_symbols))
            async with aiohttp.ClientSession(connector=connector) as session:
                downloader = self._build_downloader(session)

                for symbol in self.config.symbols:
                    for data_type_raw in self.config.data_types:
                        data_type = _enum_value(data_type_raw)

                        try:
                            if data_type == HistoryDataType.CANDLES.value:
                                for timeframe in self.config.timeframes:
                                    result = await self._download_normalize_write(
                                        downloader=downloader,
                                        symbol=symbol,
                                        data_type=data_type,
                                        timeframe=timeframe,
                                    )
                                    files_written.extend(result.files_written)
                                    downloaded_rows += result.downloaded_rows
                                    written_rows += result.written_rows
                                    failed_rows += result.failed_rows

                            else:
                                result = await self._download_normalize_write(
                                    downloader=downloader,
                                    symbol=symbol,
                                    data_type=data_type,
                                    timeframe=None,
                                )
                                files_written.extend(result.files_written)
                                downloaded_rows += result.downloaded_rows
                                written_rows += result.written_rows
                                failed_rows += result.failed_rows

                        except Exception as exc:
                            failed_rows += 1
                            err = (
                                exc.to_dict()
                                if hasattr(exc, "to_dict")
                                else {
                                    "error_type": exc.__class__.__name__,
                                    "message": str(exc),
                                }
                            )
                            errors.append(err)

                            self.logger.exception(
                                "History download task failed",
                                extra={
                                    "exchange": _enum_value(self.config.exchange),
                                    "market_type": _enum_value(self.config.market_type),
                                    "symbol": symbol,
                                    "data_type": data_type,
                                },
                            )

                            if not self.config.validate_after_download:
                                continue

            status = HistoryDownloadStatus.COMPLETED if not errors else HistoryDownloadStatus.PARTIALLY_COMPLETED

        except Exception as exc:
            status = HistoryDownloadStatus.FAILED
            err = (
                exc.to_dict()
                if hasattr(exc, "to_dict")
                else {
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                }
            )
            errors.append(err)

        completed_at_ms = _now_ms()

        return HistoryDownloadResult(
            request_id=request.request_id,
            exchange=_enum_value(self.config.exchange),
            market_type=_enum_value(self.config.market_type),
            status=status,
            symbols=list(self.config.symbols),
            data_types=[_enum_value(item) for item in self.config.data_types],
            timeframes=list(self.config.timeframes),
            start_time_ms=self.config.start_time_ms,
            end_time_ms=self.config.end_time_ms,
            output_dir=self.config.output_dir,
            files_written=files_written,
            downloaded_rows=downloaded_rows,
            written_rows=written_rows,
            failed_rows=failed_rows,
            started_at_ms=started_at_ms,
            completed_at_ms=completed_at_ms,
            errors=errors,
            metadata={
                "storage_format": _enum_value(self.config.storage_format),
                "source_type": _enum_value(self.config.source_type),
            },
        )

    async def _download_normalize_write(
        self,
        *,
        downloader: BaseFuturesHistoryDownloader,
        symbol: str,
        data_type: str,
        timeframe: str | None,
    ) -> HistoryDownloadProgress:
        progress = HistoryDownloadProgress(
            request_id="runtime",
            exchange=_enum_value(self.config.exchange),
            market_type=_enum_value(self.config.market_type),
            symbol=symbol,
            data_type=data_type,
            timeframe=timeframe,
            status=HistoryDownloadStatus.RUNNING,
            started_at_ms=_now_ms(),
            current_start_time_ms=self.config.start_time_ms,
            current_end_time_ms=self.config.end_time_ms,
        )

        raw_batch = await self._download_raw_batch(
            downloader=downloader,
            symbol=symbol,
            data_type=data_type,
            timeframe=timeframe,
        )

        progress.downloaded_rows = len(raw_batch.rows)

        normalized_batch = self.normalizer.normalize_batch(raw_batch)
        written_files = await self.writer.write_batch(
            normalized_batch,
            overwrite=self.config.overwrite_existing,
        )

        progress.written_rows = len(normalized_batch.rows)
        progress.completed_at_ms = _now_ms()
        progress.status = HistoryDownloadStatus.COMPLETED
        progress.progress_pct = 100.0
        progress.message = f"Written files: {len(written_files)}"
        progress.metadata = {"files_written": written_files} if hasattr(progress, "metadata") else {}

        return progress

    async def _download_raw_batch(
        self,
        *,
        downloader: BaseFuturesHistoryDownloader,
        symbol: str,
        data_type: str,
        timeframe: str | None,
    ) -> RawHistoryBatch:
        if data_type == HistoryDataType.CANDLES.value:
            if not timeframe:
                raise HistoryDownloadConfigError(
                    "timeframe is required for candle history",
                    context=build_error_context(
                        exchange=_enum_value(self.config.exchange),
                        market_type=_enum_value(self.config.market_type),
                        symbol=symbol,
                        data_type=data_type,
                    ),
                )

            return await downloader.download_candles(
                symbol=symbol,
                timeframe=timeframe,
                start_time_ms=self.config.start_time_ms,
                end_time_ms=self.config.end_time_ms,
            )

        if data_type == HistoryDataType.AGG_TRADES.value:
            return await downloader.download_agg_trades(
                symbol=symbol,
                start_time_ms=self.config.start_time_ms,
                end_time_ms=self.config.end_time_ms,
            )

        if data_type == HistoryDataType.TRADES.value:
            return await downloader.download_agg_trades(
                symbol=symbol,
                start_time_ms=self.config.start_time_ms,
                end_time_ms=self.config.end_time_ms,
            )

        if data_type == HistoryDataType.FUNDING.value:
            return await downloader.download_funding(
                symbol=symbol,
                start_time_ms=self.config.start_time_ms,
                end_time_ms=self.config.end_time_ms,
            )

        if data_type == HistoryDataType.OPEN_INTEREST.value:
            return await downloader.download_open_interest(
                symbol=symbol,
                start_time_ms=self.config.start_time_ms,
                end_time_ms=self.config.end_time_ms,
            )

        if data_type == HistoryDataType.LIQUIDATIONS.value:
            return await downloader.download_liquidations(
                symbol=symbol,
                start_time_ms=self.config.start_time_ms,
                end_time_ms=self.config.end_time_ms,
            )

        raise HistoryUnavailableError(
            f"Unsupported data type for history download: {data_type}",
            context=build_error_context(
                exchange=_enum_value(self.config.exchange),
                market_type=_enum_value(self.config.market_type),
                symbol=symbol,
                data_type=data_type,
            ),
        )

    def _build_downloader(
        self,
        session: aiohttp.ClientSession,
    ) -> BaseFuturesHistoryDownloader:
        exchange = _enum_value(self.config.exchange)

        if exchange == ExchangeName.BINANCE.value:
            return BinanceFuturesHistoryDownloader(config=self.config, session=session)

        if exchange == ExchangeName.BYBIT.value:
            return BybitFuturesHistoryDownloader(config=self.config, session=session)

        if exchange == ExchangeName.OKX.value:
            return OKXFuturesHistoryDownloader(config=self.config, session=session)

        if exchange == ExchangeName.MEXC.value:
            return MEXCFuturesHistoryDownloader(config=self.config, session=session)

        raise HistoryDownloadConfigError(
            f"Unsupported exchange: {exchange}",
            context=build_error_context(exchange=exchange),
        )


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


async def download_history(config: HistoryDownloadConfig) -> HistoryDownloadResult:
    """
    Convenience function for CLI/tests.

    Example:
        result = await download_history(config)
    """

    manager = HistoryDownloadManager(config=config)
    return await manager.download()