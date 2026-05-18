# trading_system/backtesting/data_loader.py

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import pandas as pd

from core.logger import get_logger

from .config import BacktestDataConfig
from .enums import (
    DataQualityStatus,
    DuplicateHandlingPolicy,
    GapHandlingPolicy,
    HistoricalEventTopic,
    HistoryDataType,
    StorageFormat,
)
from .exceptions import (
    BacktestDataDuplicateError,
    BacktestDataGapError,
    BacktestDataNotFoundError,
    BacktestDataOrderError,
    BacktestDataQualityError,
    BacktestDataRangeError,
    BacktestDataSchemaError,
    HistoryReadError,
    build_error_context,
)
from .models import DataQualityReport, HistoricalMarketEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _month_range(start_time_ms: int, end_time_ms: int) -> list[str]:
    """
    Return YYYY-MM partitions touched by the requested time range.
    """

    start = pd.Timestamp(start_time_ms, unit="ms", tz="UTC").to_period("M")
    end = pd.Timestamp(end_time_ms, unit="ms", tz="UTC").to_period("M")

    periods = pd.period_range(start=start, end=end, freq="M")
    return [str(period) for period in periods]


def _day_range(start_time_ms: int, end_time_ms: int) -> list[str]:
    """
    Return YYYY-MM-DD partitions touched by the requested time range.
    """

    start = pd.Timestamp(start_time_ms, unit="ms", tz="UTC").normalize()
    end = pd.Timestamp(end_time_ms, unit="ms", tz="UTC").normalize()

    days = pd.date_range(start=start, end=end, freq="D", tz="UTC")
    return [day.strftime("%Y-%m-%d") for day in days]


def _year_range(start_time_ms: int, end_time_ms: int) -> list[str]:
    """
    Return YYYY partitions touched by the requested time range.
    """

    start_year = pd.Timestamp(start_time_ms, unit="ms", tz="UTC").year
    end_year = pd.Timestamp(end_time_ms, unit="ms", tz="UTC").year

    return [str(year) for year in range(start_year, end_year + 1)]


def _timeframe_to_ms(timeframe: str) -> int:
    mapping = {
        "1m": 60_000,
        "3m": 3 * 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 60 * 60_000,
        "2h": 2 * 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "6h": 6 * 60 * 60_000,
        "8h": 8 * 60 * 60_000,
        "12h": 12 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
    }

    if timeframe not in mapping:
        raise BacktestDataSchemaError(
            f"Unsupported timeframe: {timeframe}",
            context=build_error_context(timeframe=timeframe),
        )

    return mapping[timeframe]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HistoryFileRef:
    """
    Reference to one local history file.
    """

    path: Path
    exchange: str
    market_type: str
    symbol: str
    data_type: str
    timeframe: str | None = None
    partition: str | None = None


class HistoryFileResolver:
    """
    Resolves expected local history file paths.

    Layout must match history_downloader.py:

        data/history/{exchange}/{market_type}/{symbol}/{data_type}/...

    Candles:
        candles/{timeframe}/{YYYY-MM}.parquet

    High-frequency event data:
        agg_trades/{YYYY-MM-DD}.parquet
        trades/{YYYY-MM-DD}.parquet
        liquidations/{YYYY-MM-DD}.parquet
        orderbook_snapshots/{YYYY-MM-DD}.parquet
        orderbook_deltas/{YYYY-MM-DD}.parquet

    Funding:
        funding/{YYYY}.parquet

    Other medium-frequency data:
        open_interest/{YYYY-MM}.parquet
        mark_price/{YYYY-MM}.parquet
        index_price/{YYYY-MM}.parquet
    """

    def __init__(self, config: BacktestDataConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def resolve(self) -> list[HistoryFileRef]:
        refs: list[HistoryFileRef] = []

        exchange = _enum_value(self.config.exchange)
        market_type = _enum_value(self.config.market_type)

        for symbol in self.config.symbols:
            for data_type_raw in self.config.data_types:
                data_type = _enum_value(data_type_raw)

                if data_type == HistoryDataType.CANDLES.value:
                    for timeframe in self.config.timeframes:
                        refs.extend(
                            self._resolve_candle_files(
                                exchange=exchange,
                                market_type=market_type,
                                symbol=symbol,
                                timeframe=timeframe,
                            )
                        )
                else:
                    refs.extend(
                        self._resolve_non_candle_files(
                            exchange=exchange,
                            market_type=market_type,
                            symbol=symbol,
                            data_type=data_type,
                        )
                    )

        return refs

    def _resolve_candle_files(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
    ) -> list[HistoryFileRef]:
        refs: list[HistoryFileRef] = []

        base = (
            Path(self.config.data_dir)
            / exchange
            / market_type
            / symbol
            / HistoryDataType.CANDLES.value
            / timeframe
        )

        for partition in _month_range(self.config.start_time_ms, self.config.end_time_ms):
            path = base / f"{partition}.{_enum_value(self.config.storage_format)}"
            refs.append(
                HistoryFileRef(
                    path=path,
                    exchange=exchange,
                    market_type=market_type,
                    symbol=symbol,
                    data_type=HistoryDataType.CANDLES.value,
                    timeframe=timeframe,
                    partition=partition,
                )
            )

        return refs

    def _resolve_non_candle_files(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        data_type: str,
    ) -> list[HistoryFileRef]:
        refs: list[HistoryFileRef] = []

        base = Path(self.config.data_dir) / exchange / market_type / symbol / data_type

        if data_type in {
            HistoryDataType.TRADES.value,
            HistoryDataType.AGG_TRADES.value,
            HistoryDataType.ORDERBOOK_SNAPSHOTS.value,
            HistoryDataType.ORDERBOOK_DELTAS.value,
            HistoryDataType.LIQUIDATIONS.value,
        }:
            partitions = _day_range(self.config.start_time_ms, self.config.end_time_ms)

        elif data_type == HistoryDataType.FUNDING.value:
            partitions = _year_range(self.config.start_time_ms, self.config.end_time_ms)

        else:
            partitions = _month_range(self.config.start_time_ms, self.config.end_time_ms)

        for partition in partitions:
            path = base / f"{partition}.{_enum_value(self.config.storage_format)}"
            refs.append(
                HistoryFileRef(
                    path=path,
                    exchange=exchange,
                    market_type=market_type,
                    symbol=symbol,
                    data_type=data_type,
                    timeframe=None,
                    partition=partition,
                )
            )

        return refs


# ---------------------------------------------------------------------------
# Local readers
# ---------------------------------------------------------------------------


class ParquetBacktestDataReader:
    """
    Reads local Parquet files and returns filtered pandas DataFrames.

    This class does not convert rows into EventBus events.
    """

    def __init__(self, config: BacktestDataConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    async def read_file(self, ref: HistoryFileRef) -> pd.DataFrame:
        if not ref.path.exists():
            raise BacktestDataNotFoundError(
                "History file not found",
                context=build_error_context(
                    exchange=ref.exchange,
                    market_type=ref.market_type,
                    symbol=ref.symbol,
                    timeframe=ref.timeframe,
                    data_type=ref.data_type,
                    data_path=str(ref.path),
                ),
            )

        try:
            df = await asyncio.to_thread(pd.read_parquet, ref.path)
        except Exception as exc:
            raise HistoryReadError(
                "Failed to read local history file",
                context=build_error_context(
                    exchange=ref.exchange,
                    market_type=ref.market_type,
                    symbol=ref.symbol,
                    timeframe=ref.timeframe,
                    data_type=ref.data_type,
                    data_path=str(ref.path),
                ),
                cause=exc,
            ) from exc

        if df.empty:
            return df

        if "timestamp_ms" not in df.columns:
            raise BacktestDataSchemaError(
                "History file is missing required timestamp_ms column",
                context=build_error_context(
                    exchange=ref.exchange,
                    market_type=ref.market_type,
                    symbol=ref.symbol,
                    timeframe=ref.timeframe,
                    data_type=ref.data_type,
                    data_path=str(ref.path),
                ),
            )

        df = df[
            (df["timestamp_ms"] >= self.config.start_time_ms)
            & (df["timestamp_ms"] <= self.config.end_time_ms)
        ].copy()

        if df.empty:
            return df

        if self.config.sort_events:
            df = df.sort_values("timestamp_ms").reset_index(drop=True)

        return df


# ---------------------------------------------------------------------------
# Schema / quality validation
# ---------------------------------------------------------------------------


class BacktestDataValidator:
    """
    Performs lightweight schema and quality validation before replay.

    Heavy data cleaning should happen in ingestion/downloader layer.
    """

    REQUIRED_COLUMNS_BY_TYPE: dict[str, set[str]] = {
        HistoryDataType.CANDLES.value: {
            "exchange",
            "symbol",
            "market_type",
            "timeframe",
            "open_time_ms",
            "close_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "is_closed",
            "timestamp_ms",
            "received_at_ms",
        },
        HistoryDataType.TRADES.value: {
            "exchange",
            "symbol",
            "market_type",
            "trade_id",
            "price",
            "quantity",
            "side",
            "timestamp_ms",
            "received_at_ms",
        },
        HistoryDataType.AGG_TRADES.value: {
            "exchange",
            "symbol",
            "market_type",
            "trade_id",
            "price",
            "quantity",
            "side",
            "timestamp_ms",
            "received_at_ms",
        },
        HistoryDataType.FUNDING.value: {
            "exchange",
            "symbol",
            "market_type",
            "funding_rate",
            "timestamp_ms",
            "received_at_ms",
        },
        HistoryDataType.OPEN_INTEREST.value: {
            "exchange",
            "symbol",
            "market_type",
            "open_interest",
            "timestamp_ms",
            "received_at_ms",
        },
        HistoryDataType.LIQUIDATIONS.value: {
            "exchange",
            "symbol",
            "market_type",
            "price",
            "quantity",
            "side",
            "timestamp_ms",
            "received_at_ms",
        },
        HistoryDataType.MARK_PRICE.value: {
            "exchange",
            "symbol",
            "market_type",
            "mark_price",
            "timestamp_ms",
            "received_at_ms",
        },
        HistoryDataType.INDEX_PRICE.value: {
            "exchange",
            "symbol",
            "market_type",
            "index_price",
            "timestamp_ms",
            "received_at_ms",
        },
    }

    def __init__(self, config: BacktestDataConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def validate_dataframe(
        self,
        *,
        ref: HistoryFileRef,
        df: pd.DataFrame,
    ) -> DataQualityReport:
        if df.empty:
            return DataQualityReport(
                exchange=ref.exchange,
                market_type=ref.market_type,
                symbol=ref.symbol,
                timeframe=ref.timeframe,
                data_type=ref.data_type,
                status=DataQualityStatus.EMPTY,
                start_time_ms=self.config.start_time_ms,
                end_time_ms=self.config.end_time_ms,
                rows=0,
            )

        if self.config.validate_schema:
            self._validate_schema(ref=ref, df=df)

        duplicate_rows = int(df.duplicated(subset=["timestamp_ms"]).sum())
        out_of_order_rows = self._count_out_of_order_rows(df)

        gap_count = 0
        if ref.data_type == HistoryDataType.CANDLES.value and ref.timeframe:
            gap_count = self._count_candle_gaps(df=df, timeframe=ref.timeframe)

        status = DataQualityStatus.VALID

        if duplicate_rows > 0:
            status = DataQualityStatus.HAS_DUPLICATES

        if gap_count > 0:
            status = DataQualityStatus.HAS_GAPS

        if out_of_order_rows > 0:
            status = DataQualityStatus.OUT_OF_ORDER

        report = DataQualityReport(
            exchange=ref.exchange,
            market_type=ref.market_type,
            symbol=ref.symbol,
            timeframe=ref.timeframe,
            data_type=ref.data_type,
            status=status,
            start_time_ms=self.config.start_time_ms,
            end_time_ms=self.config.end_time_ms,
            rows=len(df),
            duplicate_rows=duplicate_rows,
            gap_count=gap_count,
            out_of_order_rows=out_of_order_rows,
            gap_policy=self.config.gap_policy,
            duplicate_policy=_enum_value(self.config.duplicate_policy),
        )

        if self.config.validate_quality:
            self._apply_quality_policy(report)

        return report

    def _validate_schema(self, *, ref: HistoryFileRef, df: pd.DataFrame) -> None:
        required = self.REQUIRED_COLUMNS_BY_TYPE.get(ref.data_type)

        if not required:
            # Unknown/advanced data types may be supported later by custom loaders.
            return

        missing = required.difference(set(df.columns))
        if missing:
            raise BacktestDataSchemaError(
                "History file has invalid schema",
                context=build_error_context(
                    exchange=ref.exchange,
                    market_type=ref.market_type,
                    symbol=ref.symbol,
                    timeframe=ref.timeframe,
                    data_type=ref.data_type,
                    data_path=str(ref.path),
                    missing_columns=sorted(missing),
                ),
            )

    def _count_out_of_order_rows(self, df: pd.DataFrame) -> int:
        timestamps = df["timestamp_ms"].tolist()
        count = 0

        previous: int | None = None
        for value in timestamps:
            ts = _safe_int(value)
            if previous is not None and ts < previous:
                count += 1
            previous = ts

        return count

    def _count_candle_gaps(self, *, df: pd.DataFrame, timeframe: str) -> int:
        if len(df) <= 1:
            return 0

        expected_step = _timeframe_to_ms(timeframe)

        timestamps = sorted(_safe_int(ts) for ts in df["timestamp_ms"].tolist())

        gaps = 0
        for prev, current in zip(timestamps, timestamps[1:]):
            if current - prev > expected_step:
                gaps += 1

        return gaps

    def _apply_quality_policy(self, report: DataQualityReport) -> None:
        gap_policy = _enum_value(self.config.gap_policy)
        duplicate_policy = _enum_value(self.config.duplicate_policy)

        if report.gap_count > 0 and gap_policy == GapHandlingPolicy.FAIL.value:
            raise BacktestDataGapError(
                "Historical data contains gaps",
                context=build_error_context(
                    exchange=report.exchange,
                    market_type=report.market_type,
                    symbol=report.symbol,
                    timeframe=report.timeframe,
                    data_type=report.data_type,
                    start_time_ms=report.start_time_ms,
                    end_time_ms=report.end_time_ms,
                    gap_count=report.gap_count,
                ),
            )

        if report.duplicate_rows > 0 and duplicate_policy == DuplicateHandlingPolicy.FAIL.value:
            raise BacktestDataDuplicateError(
                "Historical data contains duplicate rows",
                context=build_error_context(
                    exchange=report.exchange,
                    market_type=report.market_type,
                    symbol=report.symbol,
                    timeframe=report.timeframe,
                    data_type=report.data_type,
                    duplicate_rows=report.duplicate_rows,
                ),
            )

        if report.out_of_order_rows > 0 and self.config.enforce_chronological_order:
            raise BacktestDataOrderError(
                "Historical data is out of chronological order",
                context=build_error_context(
                    exchange=report.exchange,
                    market_type=report.market_type,
                    symbol=report.symbol,
                    timeframe=report.timeframe,
                    data_type=report.data_type,
                    out_of_order_rows=report.out_of_order_rows,
                ),
            )


# ---------------------------------------------------------------------------
# Event builder
# ---------------------------------------------------------------------------


class HistoricalEventBuilder:
    """
    Converts normalized local history rows into HistoricalMarketEvent objects.
    """

    TOPIC_BY_DATA_TYPE: dict[str, str] = {
        HistoryDataType.CANDLES.value: HistoricalEventTopic.MARKET_CANDLE.value,
        HistoryDataType.TRADES.value: HistoricalEventTopic.MARKET_TRADE.value,
        HistoryDataType.AGG_TRADES.value: HistoricalEventTopic.MARKET_TRADE.value,
        HistoryDataType.FUNDING.value: HistoricalEventTopic.MARKET_FUNDING.value,
        HistoryDataType.OPEN_INTEREST.value: HistoricalEventTopic.MARKET_OPEN_INTEREST.value,
        HistoryDataType.LIQUIDATIONS.value: HistoricalEventTopic.MARKET_LIQUIDATION.value,
        HistoryDataType.MARK_PRICE.value: HistoricalEventTopic.MARKET_MARK_PRICE.value,
        HistoryDataType.INDEX_PRICE.value: HistoricalEventTopic.MARKET_INDEX_PRICE.value,
        HistoryDataType.ORDERBOOK_SNAPSHOTS.value: HistoricalEventTopic.MARKET_ORDERBOOK_SNAPSHOT.value,
        HistoryDataType.ORDERBOOK_DELTAS.value: HistoricalEventTopic.MARKET_ORDERBOOK.value,
    }

    def build_events_from_dataframe(
        self,
        *,
        ref: HistoryFileRef,
        df: pd.DataFrame,
    ) -> list[HistoricalMarketEvent]:
        if df.empty:
            return []

        topic = self.TOPIC_BY_DATA_TYPE.get(ref.data_type)
        if not topic:
            raise BacktestDataSchemaError(
                "Unsupported data type for event building",
                context=build_error_context(
                    exchange=ref.exchange,
                    market_type=ref.market_type,
                    symbol=ref.symbol,
                    timeframe=ref.timeframe,
                    data_type=ref.data_type,
                    data_path=str(ref.path),
                ),
            )

        events: list[HistoricalMarketEvent] = []

        for idx, row in enumerate(df.to_dict(orient="records")):
            timestamp_ms = _safe_int(row.get("timestamp_ms"))

            payload = self._clean_payload(row)

            event = HistoricalMarketEvent(
                topic=topic,
                timestamp_ms=timestamp_ms,
                payload=payload,
                source="backtest.data_loader",
                sequence=idx,
                exchange=_safe_str(row.get("exchange"), ref.exchange),
                market_type=_safe_str(row.get("market_type"), ref.market_type),
                symbol=_safe_str(row.get("symbol"), ref.symbol),
                timeframe=_safe_str(row.get("timeframe"), ref.timeframe or "") or None,
                data_type=ref.data_type,
            )
            events.append(event)

        return events

    def _clean_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Convert pandas/numpy nulls to None and keep the normalized shape.
        """

        payload: dict[str, Any] = {}

        for key, value in row.items():
            if pd.isna(value):
                payload[key] = None
            else:
                payload[key] = value

        return payload


# ---------------------------------------------------------------------------
# Stream merging
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EventBuffer:
    """
    Small in-memory event buffer.

    For MVP and medium-sized backtests this is enough. Later, if needed,
    this can be replaced by an external k-way merge over file streams.
    """

    events: list[HistoricalMarketEvent] = field(default_factory=list)

    def add(self, new_events: Iterable[HistoricalMarketEvent]) -> None:
        self.events.extend(new_events)

    def sort(self) -> None:
        self.events.sort(
            key=lambda event: (
                event.timestamp_ms,
                event.topic,
                event.exchange or "",
                event.symbol or "",
                event.timeframe or "",
                event.sequence if event.sequence is not None else 0,
            )
        )

    def trim(self, max_events: int | None) -> None:
        if max_events is not None and len(self.events) > max_events:
            self.events = self.events[:max_events]


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


class BacktestDataLoader:
    """
    Reads local futures historical data and yields HistoricalMarketEvent objects.

    Responsibility:
    - read local files;
    - validate schema and data quality;
    - convert rows to HistoricalMarketEvent;
    - emit no EventBus events;
    - call no strategy/risk/execution code.

    MarketReplay is responsible for publishing these events into EventBus.
    """

    def __init__(
        self,
        *,
        config: BacktestDataConfig,
        file_resolver: HistoryFileResolver | None = None,
        reader: ParquetBacktestDataReader | None = None,
        validator: BacktestDataValidator | None = None,
        event_builder: HistoricalEventBuilder | None = None,
    ) -> None:
        self.config = config
        self.file_resolver = file_resolver or HistoryFileResolver(config)
        self.reader = reader or ParquetBacktestDataReader(config)
        self.validator = validator or BacktestDataValidator(config)
        self.event_builder = event_builder or HistoricalEventBuilder()

        self.logger = get_logger(__name__)
        self.quality_reports: list[DataQualityReport] = []

    async def iter_events(self) -> AsyncIterator[HistoricalMarketEvent]:
        """
        Yield HistoricalMarketEvent stream in chronological order.

        For now this uses an in-memory buffer. This is simple and reliable for
        MVP candle/trade-level backtests. Later we can replace it with a true
        streaming k-way merge if datasets become too large.
        """

        self.config.validate()

        if _enum_value(self.config.storage_format) != StorageFormat.PARQUET.value:
            raise BacktestDataSchemaError(
                "Only Parquet storage is currently supported by BacktestDataLoader",
                context=build_error_context(
                    data_path=self.config.data_dir,
                    storage_format=_enum_value(self.config.storage_format),
                ),
            )

        buffer = EventBuffer()

        refs = self.file_resolver.resolve()

        if not refs:
            raise BacktestDataNotFoundError(
                "No history file references resolved",
                context=build_error_context(
                    exchange=_enum_value(self.config.exchange),
                    market_type=_enum_value(self.config.market_type),
                    start_time_ms=self.config.start_time_ms,
                    end_time_ms=self.config.end_time_ms,
                    data_types=[_enum_value(item) for item in self.config.data_types],
                ),
            )

        missing_required_files: list[str] = []

        for ref in refs:
            try:
                df = await self.reader.read_file(ref)
            except BacktestDataNotFoundError:
                missing_required_files.append(str(ref.path))
                continue

            if df.empty:
                self.quality_reports.append(
                    DataQualityReport(
                        exchange=ref.exchange,
                        market_type=ref.market_type,
                        symbol=ref.symbol,
                        timeframe=ref.timeframe,
                        data_type=ref.data_type,
                        status=DataQualityStatus.EMPTY,
                        start_time_ms=self.config.start_time_ms,
                        end_time_ms=self.config.end_time_ms,
                        rows=0,
                    )
                )
                continue

            report = self.validator.validate_dataframe(ref=ref, df=df)
            self.quality_reports.append(report)

            df = self._apply_duplicate_policy(ref=ref, df=df)

            events = self.event_builder.build_events_from_dataframe(ref=ref, df=df)
            buffer.add(events)

        if missing_required_files:
            self._handle_missing_files(missing_required_files)

        if self.config.sort_events:
            buffer.sort()

        if self.config.enforce_chronological_order:
            self._validate_event_chronology(buffer.events)

        buffer.trim(self.config.max_events)

        self.logger.info(
            "Backtest historical events loaded",
            extra={
                "events": len(buffer.events),
                "exchange": _enum_value(self.config.exchange),
                "market_type": _enum_value(self.config.market_type),
                "symbols": self.config.symbols,
                "data_types": [_enum_value(item) for item in self.config.data_types],
                "start_time_ms": self.config.start_time_ms,
                "end_time_ms": self.config.end_time_ms,
            },
        )

        for event in buffer.events:
            yield event

    async def load_all(self) -> list[HistoricalMarketEvent]:
        """
        Convenience method for tests and small backtests.
        """

        return [event async for event in self.iter_events()]

    def get_quality_reports(self) -> list[DataQualityReport]:
        return list(self.quality_reports)

    def _apply_duplicate_policy(
        self,
        *,
        ref: HistoryFileRef,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        duplicate_policy = _enum_value(self.config.duplicate_policy)

        if duplicate_policy == DuplicateHandlingPolicy.FAIL.value:
            return df

        if not df.duplicated(subset=["timestamp_ms"]).any():
            return df

        if duplicate_policy == DuplicateHandlingPolicy.KEEP_FIRST.value:
            return (
                df.drop_duplicates(subset=["timestamp_ms"], keep="first")
                .sort_values("timestamp_ms")
                .reset_index(drop=True)
            )

        if duplicate_policy == DuplicateHandlingPolicy.KEEP_LAST.value:
            return (
                df.drop_duplicates(subset=["timestamp_ms"], keep="last")
                .sort_values("timestamp_ms")
                .reset_index(drop=True)
            )

        if duplicate_policy == DuplicateHandlingPolicy.MERGE.value:
            # For now merge means keep last. Advanced merge can be implemented
            # per data type later.
            return (
                df.drop_duplicates(subset=["timestamp_ms"], keep="last")
                .sort_values("timestamp_ms")
                .reset_index(drop=True)
            )

        raise BacktestDataQualityError(
            "Unsupported duplicate handling policy",
            context=build_error_context(
                exchange=ref.exchange,
                market_type=ref.market_type,
                symbol=ref.symbol,
                timeframe=ref.timeframe,
                data_type=ref.data_type,
                duplicate_policy=duplicate_policy,
            ),
        )

    def _handle_missing_files(self, missing_files: list[str]) -> None:
        """
        Missing files are treated depending on gap_policy.

        FAIL -> raise
        WARN/SKIP/FORWARD_FILL/BACK_FILL -> log and continue

        Actual filling is intentionally not done here. Filling historical market
        data can introduce bias, so this loader only allows the run to continue.
        """

        gap_policy = _enum_value(self.config.gap_policy)

        if gap_policy == GapHandlingPolicy.FAIL.value:
            raise BacktestDataNotFoundError(
                "Required history files are missing",
                context=build_error_context(
                    exchange=_enum_value(self.config.exchange),
                    market_type=_enum_value(self.config.market_type),
                    data_path=self.config.data_dir,
                    missing_files=missing_files[:50],
                    missing_files_count=len(missing_files),
                ),
            )

        self.logger.warning(
            "Some history files are missing",
            extra={
                "missing_files_count": len(missing_files),
                "gap_policy": gap_policy,
                "first_missing_files": missing_files[:10],
            },
        )

    def _validate_event_chronology(self, events: list[HistoricalMarketEvent]) -> None:
        previous_ts: int | None = None

        for event in events:
            if previous_ts is not None and event.timestamp_ms < previous_ts:
                raise BacktestDataOrderError(
                    "Historical events are not sorted chronologically",
                    context=build_error_context(
                        exchange=event.exchange,
                        market_type=event.market_type,
                        symbol=event.symbol,
                        timeframe=event.timeframe,
                        topic=event.topic,
                        timestamp_ms=event.timestamp_ms,
                        previous_timestamp_ms=previous_ts,
                    ),
                )

            previous_ts = event.timestamp_ms


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


async def iter_backtest_events(
    config: BacktestDataConfig,
) -> AsyncIterator[HistoricalMarketEvent]:
    """
    Convenience async generator for tests/CLI.
    """

    loader = BacktestDataLoader(config=config)

    async for event in loader.iter_events():
        yield event


async def load_backtest_events(config: BacktestDataConfig) -> list[HistoricalMarketEvent]:
    """
    Convenience function for small datasets.
    """

    loader = BacktestDataLoader(config=config)
    return await loader.load_all()