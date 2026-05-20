"""
Historical data loader for backtesting.

DataLoader reads already downloaded historical files from local storage and
builds replay-ready BacktestDataset objects.

Main responsibilities:
- discover local historical files;
- read parquet/csv/json/jsonl;
- normalize rows into Historical* records;
- validate timestamps/order/gaps;
- convert records into BacktestEvent objects;
- build BacktestDataset for MarketReplay.

Important:
- No live exchange calls here.
- No strategy/risk/execution logic here.
- No EventBus emission here.
- MarketReplay is responsible for emitting market.* events.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from backtesting.config import DataLoaderConfig
from backtesting.enums import (
    BacktestDataType,
    BacktestEventType,
    DataGapPolicy,
    DataValidationLevel,
    HistoricalDataFormat,
    ReplayOrdering,
    ReplayMode,
)
from backtesting.exceptions import (
    DataGapError,
    DataLoadError,
    DataNormalizationError,
    DataValidationError,
    HistoricalDataFormatError,
)
from backtesting.market_replay import market_topic_for_data_type, replay_priority_for_data_type
from backtesting.models import (
    BacktestDataset,
    BacktestDatasetInfo,
    BacktestDataSource,
    BacktestEvent,
    BacktestInstrument,
    BacktestPeriod,
    HistoricalCandle,
    HistoricalFundingRecord,
    HistoricalLiquidationRecord,
    HistoricalOpenInterestRecord,
    HistoricalOrderBookLevel,
    HistoricalOrderBookSnapshot,
    HistoricalTrade,
)

try:
    from core.logger import get_logger
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


@dataclass(slots=True)
class DataFileRef:
    """
    Reference to a local historical data file.
    """

    path: Path
    data_type: BacktestDataType
    exchange: str
    market_type: str
    symbol: str
    timeframe: str | None = None
    format: HistoricalDataFormat = HistoricalDataFormat.PARQUET
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoadedDataBundle:
    """
    Loaded normalized historical records.
    """

    candles: list[HistoricalCandle] = field(default_factory=list)
    trades: list[HistoricalTrade] = field(default_factory=list)
    orderbooks: list[HistoricalOrderBookSnapshot] = field(default_factory=list)
    funding: list[HistoricalFundingRecord] = field(default_factory=list)
    open_interest: list[HistoricalOpenInterestRecord] = field(default_factory=list)
    liquidations: list[HistoricalLiquidationRecord] = field(default_factory=list)

    files: list[DataFileRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_records(self) -> int:
        return (
            len(self.candles)
            + len(self.trades)
            + len(self.orderbooks)
            + len(self.funding)
            + len(self.open_interest)
            + len(self.liquidations)
        )

    @property
    def is_empty(self) -> bool:
        return self.total_records == 0

    def records_by_type(self) -> dict[BacktestDataType, list[Any]]:
        return {
            BacktestDataType.CANDLES: list(self.candles),
            BacktestDataType.TRADES: list(self.trades),
            BacktestDataType.ORDERBOOK_SNAPSHOT: list(self.orderbooks),
            BacktestDataType.FUNDING: list(self.funding),
            BacktestDataType.OPEN_INTEREST: list(self.open_interest),
            BacktestDataType.LIQUIDATIONS: list(self.liquidations),
        }


class DataLoader:
    """
    Local historical data loader.

    Expected default file layout from HistoryDownloader:

        data/history/
            binance/
                usdm_futures/
                    candles/
                        BTCUSDT/
                            1m/
                                BTCUSDT_1m.parquet
                    funding/
                        BTCUSDT/
                            BTCUSDT.parquet
                    open_interest/
                        BTCUSDT/
                            5m/
                                BTCUSDT_5m.parquet
    """

    def __init__(
        self,
        config: DataLoaderConfig | None = None,
        *,
        logger_name: str = "backtesting.data_loader",
    ) -> None:
        self.config = config or DataLoaderConfig()
        self.config.validate()
        self.logger = get_logger(logger_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        *,
        period: BacktestPeriod | None = None,
        run_id: str | None = None,
    ) -> BacktestDataset:
        """
        Load configured historical files and build BacktestDataset.
        """

        bundle = self.load_bundle(period=period)
        return self.build_dataset(bundle, period=period, run_id=run_id)

    def load_bundle(
        self,
        *,
        period: BacktestPeriod | None = None,
    ) -> LoadedDataBundle:
        """
        Load all configured historical files into normalized records.
        """

        files = self.discover_files()

        if not files:
            raise DataLoadError(
                "No historical data files found.",
                details={
                    "data_dir": str(self.config.data_dir),
                    "exchange": self.config.exchange,
                    "market_type": self.config.market_type,
                    "symbols": self.config.symbols,
                    "timeframes": self.config.timeframes,
                    "data_types": [item.value for item in self.config.data_types],
                },
            )

        bundle = LoadedDataBundle(files=files)

        for file_ref in files:
            try:
                rows = self.read_rows(file_ref.path, file_ref.format)
                records = self.normalize_rows(
                    rows,
                    data_type=file_ref.data_type,
                    symbol=file_ref.symbol,
                    timeframe=file_ref.timeframe,
                    source_path=file_ref.path,
                )

                records = self._filter_records_by_period(records, period)

                if file_ref.data_type == BacktestDataType.CANDLES:
                    bundle.candles.extend(records)
                elif file_ref.data_type == BacktestDataType.TRADES:
                    bundle.trades.extend(records)
                elif file_ref.data_type in {BacktestDataType.ORDERBOOK, BacktestDataType.ORDERBOOK_SNAPSHOT}:
                    bundle.orderbooks.extend(records)
                elif file_ref.data_type == BacktestDataType.FUNDING:
                    bundle.funding.extend(records)
                elif file_ref.data_type == BacktestDataType.OPEN_INTEREST:
                    bundle.open_interest.extend(records)
                elif file_ref.data_type == BacktestDataType.LIQUIDATIONS:
                    bundle.liquidations.extend(records)

            except Exception as exc:
                message = f"Failed to load {file_ref.path}: {exc}"

                if self.config.allow_empty_optional_streams and not self._is_required(file_ref.data_type):
                    bundle.warnings.append(message)
                    self.logger.warning(message)
                    continue

                raise DataLoadError(
                    "Failed to load historical data file.",
                    details={
                        "path": str(file_ref.path),
                        "data_type": file_ref.data_type.value,
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    },
                ) from exc

        self.validate_bundle(bundle, period=period)
        self._sort_bundle(bundle)
        self._dedupe_bundle(bundle)

        return bundle

    def build_dataset(
        self,
        bundle: LoadedDataBundle,
        *,
        period: BacktestPeriod | None = None,
        run_id: str | None = None,
    ) -> BacktestDataset:
        """
        Convert normalized records into BacktestDataset.
        """

        if bundle.is_empty:
            raise DataLoadError("Cannot build BacktestDataset from empty LoadedDataBundle.")

        events: list[BacktestEvent] = []
        sequence = 0

        for data_type, records in bundle.records_by_type().items():
            for record in records:
                event = self.build_event_from_record(
                    record,
                    data_type=data_type,
                    period=period,
                    run_id=run_id,
                    sequence=sequence,
                )
                events.append(event)
                sequence += 1

        dataset = BacktestDataset(
            events=events,
            ordering=ReplayOrdering.TIMESTAMP_THEN_PRIORITY,
            replay_mode=ReplayMode.FULL_RUN,
            metadata={
                "source": "data_loader",
                "warnings": list(bundle.warnings),
                "files": [str(file_ref.path) for file_ref in bundle.files],
            },
        )

        dataset.sort_events()

        dataset.info = self._build_dataset_info(
            dataset=dataset,
            bundle=bundle,
            period=period,
        )

        if self.config.max_events is not None:
            dataset.events = dataset.events[: self.config.max_events]
            dataset.info.total_events = len(dataset.events)

        return dataset

    def build_event_from_record(
        self,
        record: Any,
        *,
        data_type: BacktestDataType,
        period: BacktestPeriod | None = None,
        run_id: str | None = None,
        sequence: int | None = None,
    ) -> BacktestEvent:
        """
        Convert one Historical* record into BacktestEvent.
        """

        if not hasattr(record, "to_market_event_payload"):
            raise DataNormalizationError(
                "Historical record does not expose to_market_event_payload().",
                details={
                    "record_type": record.__class__.__name__,
                    "data_type": data_type.value,
                },
            )

        event_timestamp_ms = self._record_timestamp_ms(record)

        topic = market_topic_for_data_type(data_type)
        priority = replay_priority_for_data_type(data_type)

        is_warmup = period.is_warmup(event_timestamp_ms) if period is not None else False

        payload = record.to_market_event_payload()

        return BacktestEvent(
            run_id=run_id,
            event_type=BacktestEventType.MARKET,
            topic=topic,
            timestamp_ms=event_timestamp_ms,
            payload=payload,
            source="data_loader",
            sequence=sequence,
            priority=priority,
            is_warmup=is_warmup,
            metadata={
                "data_type": data_type.value,
                "record_type": record.__class__.__name__,
                "instrument_key": getattr(record, "instrument_key", None),
            },
        )

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def discover_files(self) -> list[DataFileRef]:
        """
        Discover configured files under data_dir.
        """

        base_dir = Path(self.config.data_dir)
        result: list[DataFileRef] = []

        for data_type in self.config.data_types:
            for symbol in self.config.symbols:
                if data_type == BacktestDataType.CANDLES:
                    for timeframe in self.config.timeframes:
                        result.extend(
                            self._discover_for(
                                data_type=data_type,
                                symbol=symbol,
                                timeframe=timeframe,
                            )
                        )
                    continue

                if data_type == BacktestDataType.OPEN_INTEREST:
                    # OI downloader may store period/timeframe subfolders.
                    found_any = False
                    for timeframe in self.config.timeframes:
                        refs = self._discover_for(
                            data_type=data_type,
                            symbol=symbol,
                            timeframe=timeframe,
                        )
                        if refs:
                            found_any = True
                            result.extend(refs)

                    refs_no_tf = self._discover_for(
                        data_type=data_type,
                        symbol=symbol,
                        timeframe=None,
                    )
                    if refs_no_tf:
                        found_any = True
                        result.extend(refs_no_tf)

                    if not found_any and self.config.require_open_interest:
                        self._raise_missing_file(data_type, symbol)
                    continue

                result.extend(
                    self._discover_for(
                        data_type=data_type,
                        symbol=symbol,
                        timeframe=None,
                    )
                )

        unique: dict[str, DataFileRef] = {}
        for item in result:
            unique[str(item.path)] = item

        return list(unique.values())

    def _discover_for(
        self,
        *,
        data_type: BacktestDataType,
        symbol: str,
        timeframe: str | None,
    ) -> list[DataFileRef]:
        symbol = symbol.upper()
        data_type_dir = self._data_type_dir_name(data_type)

        base = (
            Path(self.config.data_dir)
            / self.config.exchange
            / self.config.market_type
            / data_type_dir
            / symbol
        )

        if timeframe:
            search_dirs = [base / timeframe, base]
            filename_stems = [
                f"{symbol}_{timeframe}",
                symbol,
            ]
        else:
            search_dirs = [base]
            filename_stems = [symbol]

        refs: list[DataFileRef] = []

        for directory in search_dirs:
            for stem in filename_stems:
                for fmt in self._candidate_formats():
                    path = directory / f"{stem}.{self._format_extension(fmt)}"

                    if path.exists() and path.is_file():
                        refs.append(
                            DataFileRef(
                                path=path,
                                data_type=data_type,
                                exchange=self.config.exchange,
                                market_type=self.config.market_type,
                                symbol=symbol,
                                timeframe=timeframe,
                                format=fmt,
                            )
                        )

        # Flexible fallback: any matching file in expected folder.
        if not refs and base.exists():
            patterns = []

            if timeframe:
                patterns.extend(
                    [
                        f"**/*{symbol}*{timeframe}*",
                        f"**/{symbol}_{timeframe}*",
                    ]
                )
            else:
                patterns.append(f"**/*{symbol}*")

            for pattern in patterns:
                for path in base.glob(pattern):
                    if not path.is_file():
                        continue

                    fmt = self._format_from_suffix(path.suffix)
                    if fmt is None:
                        continue

                    refs.append(
                        DataFileRef(
                            path=path,
                            data_type=data_type,
                            exchange=self.config.exchange,
                            market_type=self.config.market_type,
                            symbol=symbol,
                            timeframe=timeframe,
                            format=fmt,
                        )
                    )

        if not refs and self._is_required(data_type):
            self._raise_missing_file(data_type, symbol, timeframe=timeframe)

        return refs

    def _raise_missing_file(
        self,
        data_type: BacktestDataType,
        symbol: str,
        *,
        timeframe: str | None = None,
    ) -> None:
        raise DataLoadError(
            "Required historical data file is missing.",
            details={
                "data_dir": str(self.config.data_dir),
                "exchange": self.config.exchange,
                "market_type": self.config.market_type,
                "data_type": data_type.value,
                "symbol": symbol,
                "timeframe": timeframe,
            },
        )

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def read_rows(
        self,
        path: Path,
        file_format: HistoricalDataFormat,
    ) -> list[dict[str, Any]]:
        """
        Read rows from local file.
        """

        if file_format == HistoricalDataFormat.PARQUET:
            return self._read_parquet(path)

        if file_format == HistoricalDataFormat.CSV:
            return self._read_csv(path)

        if file_format == HistoricalDataFormat.JSON:
            return self._read_json(path)

        if file_format == HistoricalDataFormat.JSONL:
            return self._read_jsonl(path)

        raise HistoricalDataFormatError(
            "Unsupported historical data input format.",
            details={
                "path": str(path),
                "format": file_format.value,
            },
        )

    @staticmethod
    def _read_parquet(path: Path) -> list[dict[str, Any]]:
        try:
            import pandas as pd
        except Exception as exc:
            raise HistoricalDataFormatError(
                "Reading parquet requires pandas and a parquet engine.",
                details={"path": str(path)},
            ) from exc

        try:
            dataframe = pd.read_parquet(path)
            return dataframe.to_dict(orient="records")
        except Exception as exc:
            raise DataLoadError(
                "Failed to read parquet file.",
                details={
                    "path": str(path),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            ) from exc

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, Any]]:
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
        except Exception as exc:
            raise DataLoadError(
                "Failed to read CSV file.",
                details={
                    "path": str(path),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            ) from exc

    @staticmethod
    def _read_json(path: Path) -> list[dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))

            if isinstance(payload, list):
                return [dict(item) for item in payload]

            if isinstance(payload, dict):
                for key in ("data", "rows", "items", "result", "records"):
                    value = payload.get(key)
                    if isinstance(value, list):
                        return [dict(item) for item in value]

                return [payload]

            raise DataLoadError(
                "JSON historical file must contain list or dict.",
                details={"path": str(path)},
            )

        except DataLoadError:
            raise
        except Exception as exc:
            raise DataLoadError(
                "Failed to read JSON file.",
                details={
                    "path": str(path),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            ) from exc

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))

            return rows

        except Exception as exc:
            raise DataLoadError(
                "Failed to read JSONL file.",
                details={
                    "path": str(path),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            ) from exc

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        data_type: BacktestDataType,
        symbol: str,
        timeframe: str | None,
        source_path: Path,
    ) -> list[Any]:
        """
        Normalize raw rows into Historical* records.
        """

        normalized: list[Any] = []

        for row in rows:
            try:
                if data_type == BacktestDataType.CANDLES:
                    normalized.append(
                        self._normalize_candle(row, symbol=symbol, timeframe=timeframe or self.config.timeframes[0])
                    )
                elif data_type == BacktestDataType.TRADES:
                    normalized.append(self._normalize_trade(row, symbol=symbol))
                elif data_type in {BacktestDataType.ORDERBOOK, BacktestDataType.ORDERBOOK_SNAPSHOT}:
                    normalized.append(self._normalize_orderbook(row, symbol=symbol))
                elif data_type == BacktestDataType.FUNDING:
                    normalized.append(self._normalize_funding(row, symbol=symbol))
                elif data_type == BacktestDataType.OPEN_INTEREST:
                    normalized.append(self._normalize_open_interest(row, symbol=symbol))
                elif data_type == BacktestDataType.LIQUIDATIONS:
                    normalized.append(self._normalize_liquidation(row, symbol=symbol))
                else:
                    continue

            except Exception as exc:
                if self.config.validation_level == DataValidationLevel.STRICT:
                    raise DataNormalizationError(
                        "Failed to normalize historical row.",
                        details={
                            "source_path": str(source_path),
                            "data_type": data_type.value,
                            "symbol": symbol,
                            "row": row,
                            "error": str(exc),
                            "error_type": exc.__class__.__name__,
                        },
                    ) from exc

                self.logger.warning(
                    "Skipping invalid historical row from %s: %s",
                    source_path,
                    exc,
                )

        return normalized

    def _normalize_candle(
        self,
        row: dict[str, Any],
        *,
        symbol: str,
        timeframe: str,
    ) -> HistoricalCandle:
        open_time_ms = self._int_from_keys(row, ["open_time_ms", "openTime", "open_time", "t", "timestamp"])
        close_time_ms = self._int_from_keys(
            row,
            ["close_time_ms", "closeTime", "close_time", "T"],
            default=open_time_ms,
        )

        return HistoricalCandle(
            exchange=str(row.get("exchange") or self.config.exchange),
            symbol=str(row.get("symbol") or symbol).upper(),
            market_type=str(row.get("market_type") or self.config.market_type),
            timeframe=str(row.get("timeframe") or timeframe),
            timestamp_ms=self._int_from_keys(row, ["timestamp_ms", "timestamp", "close_time_ms"], default=close_time_ms),
            received_at_ms=self._int_from_keys(row, ["received_at_ms", "timestamp_ms"], default=close_time_ms),
            open_time_ms=open_time_ms,
            close_time_ms=close_time_ms,
            open=self._float_from_keys(row, ["open", "o"]),
            high=self._float_from_keys(row, ["high", "h"]),
            low=self._float_from_keys(row, ["low", "l"]),
            close=self._float_from_keys(row, ["close", "c"]),
            volume=self._float_from_keys(row, ["volume", "v"], default=0.0),
            quote_volume=self._float_from_keys(row, ["quote_volume", "quoteVolume", "q"], default=0.0),
            trades_count=self._int_from_keys(row, ["trades_count", "numberOfTrades", "n"], default=0),
            is_closed=self._bool_from_keys(row, ["is_closed", "x"], default=True),
            source="data_loader",
            metadata=self._metadata(row),
        )

    def _normalize_trade(
        self,
        row: dict[str, Any],
        *,
        symbol: str,
    ) -> HistoricalTrade:
        timestamp = self._int_from_keys(row, ["timestamp_ms", "timestamp", "time", "T"])

        buyer_maker = self._optional_bool_from_keys(row, ["buyer_maker", "m"])
        aggressor_side = self._optional_str_from_keys(row, ["aggressor_side"])

        if aggressor_side is None and buyer_maker is not None:
            aggressor_side = "sell" if buyer_maker else "buy"

        return HistoricalTrade(
            exchange=str(row.get("exchange") or self.config.exchange),
            symbol=str(row.get("symbol") or symbol).upper(),
            market_type=str(row.get("market_type") or self.config.market_type),
            timestamp_ms=timestamp,
            received_at_ms=self._int_from_keys(row, ["received_at_ms", "timestamp_ms"], default=timestamp),
            trade_id=row.get("trade_id", row.get("id", row.get("a"))),
            price=self._float_from_keys(row, ["price", "p"]),
            quantity=self._float_from_keys(row, ["quantity", "qty", "q"]),
            side=self._optional_str_from_keys(row, ["side"]),
            aggressor_side=aggressor_side,
            buyer_maker=buyer_maker,
            source="data_loader",
            metadata=self._metadata(row),
        )

    def _normalize_orderbook(
        self,
        row: dict[str, Any],
        *,
        symbol: str,
    ) -> HistoricalOrderBookSnapshot:
        timestamp = self._int_from_keys(row, ["timestamp_ms", "timestamp", "time", "lastUpdateTime"])

        bids_raw = row.get("bids") or []
        asks_raw = row.get("asks") or []

        if isinstance(bids_raw, str):
            bids_raw = json.loads(bids_raw)

        if isinstance(asks_raw, str):
            asks_raw = json.loads(asks_raw)

        bids = [
            HistoricalOrderBookLevel(
                price=float(level[0] if isinstance(level, (list, tuple)) else level["price"]),
                quantity=float(level[1] if isinstance(level, (list, tuple)) else level["quantity"]),
            )
            for level in bids_raw
        ]
        asks = [
            HistoricalOrderBookLevel(
                price=float(level[0] if isinstance(level, (list, tuple)) else level["price"]),
                quantity=float(level[1] if isinstance(level, (list, tuple)) else level["quantity"]),
            )
            for level in asks_raw
        ]

        return HistoricalOrderBookSnapshot(
            exchange=str(row.get("exchange") or self.config.exchange),
            symbol=str(row.get("symbol") or symbol).upper(),
            market_type=str(row.get("market_type") or self.config.market_type),
            timestamp_ms=timestamp,
            received_at_ms=self._int_from_keys(row, ["received_at_ms", "timestamp_ms"], default=timestamp),
            bids=bids,
            asks=asks,
            sequence=self._optional_int_from_keys(row, ["sequence", "lastUpdateId", "u"]),
            depth=self._int_from_keys(row, ["depth"], default=max(len(bids), len(asks))),
            source="data_loader",
            metadata=self._metadata(row),
        )

    def _normalize_funding(
        self,
        row: dict[str, Any],
        *,
        symbol: str,
    ) -> HistoricalFundingRecord:
        timestamp = self._int_from_keys(row, ["timestamp_ms", "timestamp", "fundingTime", "funding_time", "time"])

        return HistoricalFundingRecord(
            exchange=str(row.get("exchange") or self.config.exchange),
            symbol=str(row.get("symbol") or symbol).upper(),
            market_type=str(row.get("market_type") or self.config.market_type),
            timestamp_ms=timestamp,
            received_at_ms=self._int_from_keys(row, ["received_at_ms", "timestamp_ms"], default=timestamp),
            funding_rate=self._float_from_keys(row, ["funding_rate", "fundingRate", "rate"], default=0.0),
            predicted_rate=self._optional_float_from_keys(row, ["predicted_rate", "predictedRate"]),
            mark_price=self._optional_float_from_keys(row, ["mark_price", "markPrice"]),
            index_price=self._optional_float_from_keys(row, ["index_price", "indexPrice"]),
            next_funding_time_ms=self._optional_int_from_keys(row, ["next_funding_time_ms", "nextFundingTime"]),
            source="data_loader",
            metadata=self._metadata(row),
        )

    def _normalize_open_interest(
        self,
        row: dict[str, Any],
        *,
        symbol: str,
    ) -> HistoricalOpenInterestRecord:
        timestamp = self._int_from_keys(row, ["timestamp_ms", "timestamp", "time", "sumOpenInterestTime"])

        return HistoricalOpenInterestRecord(
            exchange=str(row.get("exchange") or self.config.exchange),
            symbol=str(row.get("symbol") or symbol).upper(),
            market_type=str(row.get("market_type") or self.config.market_type),
            timestamp_ms=timestamp,
            received_at_ms=self._int_from_keys(row, ["received_at_ms", "timestamp_ms"], default=timestamp),
            open_interest=self._float_from_keys(
                row,
                ["open_interest", "openInterest", "sumOpenInterest"],
                default=0.0,
            ),
            open_interest_value=self._optional_float_from_keys(
                row,
                ["open_interest_value", "openInterestValue", "sumOpenInterestValue"],
            ),
            mark_price=self._optional_float_from_keys(row, ["mark_price", "markPrice"]),
            source="data_loader",
            metadata=self._metadata(row),
        )

    def _normalize_liquidation(
        self,
        row: dict[str, Any],
        *,
        symbol: str,
    ) -> HistoricalLiquidationRecord:
        timestamp = self._int_from_keys(row, ["timestamp_ms", "timestamp", "time", "updateTime", "T"])
        price = self._float_from_keys(row, ["price", "p", "avgPrice", "ap"])
        quantity = self._float_from_keys(row, ["quantity", "qty", "q", "origQty"])
        side = str(row.get("side") or row.get("S") or "").lower()

        return HistoricalLiquidationRecord(
            exchange=str(row.get("exchange") or self.config.exchange),
            symbol=str(row.get("symbol") or symbol).upper(),
            market_type=str(row.get("market_type") or self.config.market_type),
            timestamp_ms=timestamp,
            received_at_ms=self._int_from_keys(row, ["received_at_ms", "timestamp_ms"], default=timestamp),
            liquidation_id=row.get("liquidation_id", row.get("orderId", row.get("id"))),
            side=side,
            price=price,
            quantity=quantity,
            notional=self._float_from_keys(row, ["notional"], default=price * quantity),
            source="data_loader",
            metadata=self._metadata(row),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_bundle(
        self,
        bundle: LoadedDataBundle,
        *,
        period: BacktestPeriod | None = None,
    ) -> None:
        """
        Validate loaded data bundle.
        """

        if bundle.is_empty:
            raise DataValidationError("Loaded data bundle is empty.")

        if self.config.require_candles and not bundle.candles:
            raise DataValidationError("Required candle data is missing.")

        if self.config.require_trades and not bundle.trades:
            raise DataValidationError("Required trade data is missing.")

        if self.config.require_orderbook and not bundle.orderbooks:
            raise DataValidationError("Required orderbook data is missing.")

        if self.config.require_funding and not bundle.funding:
            raise DataValidationError("Required funding data is missing.")

        if self.config.require_open_interest and not bundle.open_interest:
            raise DataValidationError("Required open interest data is missing.")

        if self.config.validation_level in {DataValidationLevel.BASIC, DataValidationLevel.STRICT}:
            self._validate_record_ordering(bundle)
            self._validate_gaps(bundle)

        if self.config.validation_level == DataValidationLevel.STRICT:
            self._validate_symbols_and_timeframes(bundle)

    def _validate_record_ordering(self, bundle: LoadedDataBundle) -> None:
        for data_type, records in bundle.records_by_type().items():
            timestamps = [self._record_timestamp_ms(record) for record in records]
            if timestamps != sorted(timestamps):
                raise DataValidationError(
                    "Historical records are not sorted by timestamp.",
                    details={"data_type": data_type.value},
                )

    def _validate_gaps(self, bundle: LoadedDataBundle) -> None:
        if not bundle.candles:
            return

        grouped: dict[tuple[str, str, str, str], list[HistoricalCandle]] = {}

        for candle in bundle.candles:
            key = (candle.exchange, candle.market_type, candle.symbol, candle.timeframe)
            grouped.setdefault(key, []).append(candle)

        for key, candles in grouped.items():
            candles = sorted(candles, key=lambda item: item.open_time_ms)
            expected_gap_ms = self._timeframe_to_milliseconds(key[3])
            max_allowed_gap_ms = max(
                expected_gap_ms,
                self.config.max_allowed_gap_seconds * 1000,
            )

            for previous, current in zip(candles, candles[1:]):
                gap_ms = current.open_time_ms - previous.open_time_ms

                if gap_ms <= max_allowed_gap_ms:
                    continue

                message = (
                    f"Candle gap detected for {key}: "
                    f"{gap_ms / 1000:.0f}s gap between "
                    f"{previous.open_time_ms} and {current.open_time_ms}"
                )

                if self._is_gap_error_policy(self.config.gap_policy):
                    raise DataGapError(
                        message,
                        details={
                            "key": key,
                            "gap_ms": gap_ms,
                            "max_allowed_gap_ms": max_allowed_gap_ms,
                        },
                    )

                if self.config.gap_policy == DataGapPolicy.WARN:
                    bundle.warnings.append(message)
                    self.logger.warning(message)

                if self.config.gap_policy == DataGapPolicy.WARN:
                    bundle.warnings.append(message)
                    self.logger.warning(message)

    def _validate_symbols_and_timeframes(self, bundle: LoadedDataBundle) -> None:
        allowed_symbols = {symbol.upper() for symbol in self.config.symbols}
        allowed_timeframes = set(self.config.timeframes)

        for data_type, records in bundle.records_by_type().items():
            for record in records:
                symbol = getattr(record, "symbol", None)
                if symbol and symbol.upper() not in allowed_symbols:
                    raise DataValidationError(
                        "Record symbol is outside configured symbols.",
                        details={
                            "data_type": data_type.value,
                            "symbol": symbol,
                            "allowed_symbols": sorted(allowed_symbols),
                        },
                    )

                timeframe = getattr(record, "timeframe", None)
                if timeframe and data_type == BacktestDataType.CANDLES:
                    if timeframe not in allowed_timeframes:
                        raise DataValidationError(
                            "Candle timeframe is outside configured timeframes.",
                            details={
                                "timeframe": timeframe,
                                "allowed_timeframes": sorted(allowed_timeframes),
                            },
                        )

    # ------------------------------------------------------------------
    # Dataset info
    # ------------------------------------------------------------------

    def _build_dataset_info(
        self,
        *,
        dataset: BacktestDataset,
        bundle: LoadedDataBundle,
        period: BacktestPeriod | None,
    ) -> BacktestDatasetInfo:
        instruments = [
            BacktestInstrument(
                exchange=self.config.exchange,
                symbol=symbol,
                market_type=self.config.market_type,
            )
            for symbol in self.config.symbols
        ]

        data_sources = [
            BacktestDataSource(
                data_type=file_ref.data_type,
                format=file_ref.format,
                path=str(file_ref.path),
                exchange=file_ref.exchange,
                symbol=file_ref.symbol,
                market_type=file_ref.market_type,
                timeframe=file_ref.timeframe,
                metadata={
                    **dict(file_ref.metadata),
                    "source_id": str(index),
                    "name": file_ref.path.name,
                },
            )
            for index, file_ref in enumerate(bundle.files)
        ]


        first_event_time = dataset.events[0].event_time if dataset.events else None
        last_event_time = dataset.events[-1].event_time if dataset.events else None

        if period is None and first_event_time is not None and last_event_time is not None:
            period = BacktestPeriod(
                start=first_event_time,
                end=last_event_time,
            )

        return BacktestDatasetInfo(
            period=period,
            instruments=instruments,
            data_sources=data_sources,
            data_types=set(self.config.data_types),
            total_events=len(dataset.events),
            first_event_time=first_event_time,
            last_event_time=last_event_time,
            metadata={
                "loader": "DataLoader",
                "total_records": bundle.total_records,
                "warnings": list(bundle.warnings),
                "records": {
                    "candles": len(bundle.candles),
                    "trades": len(bundle.trades),
                    "orderbooks": len(bundle.orderbooks),
                    "funding": len(bundle.funding),
                    "open_interest": len(bundle.open_interest),
                    "liquidations": len(bundle.liquidations),
                },
            },
        )

    # ------------------------------------------------------------------
    # Sorting / dedupe / filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_bundle(bundle: LoadedDataBundle) -> None:
        bundle.candles.sort(key=lambda item: item.open_time_ms)
        bundle.trades.sort(key=lambda item: item.timestamp_ms)
        bundle.orderbooks.sort(key=lambda item: item.timestamp_ms)
        bundle.funding.sort(key=lambda item: item.timestamp_ms)
        bundle.open_interest.sort(key=lambda item: item.timestamp_ms)
        bundle.liquidations.sort(key=lambda item: item.timestamp_ms)

    def _dedupe_bundle(self, bundle: LoadedDataBundle) -> None:
        if self.config.drop_duplicate_events:
            bundle.candles = self._dedupe(
                bundle.candles,
                key=lambda item: (
                    item.exchange,
                    item.market_type,
                    item.symbol,
                    item.timeframe,
                    item.open_time_ms,
                ),
            )
            bundle.trades = self._dedupe(
                bundle.trades,
                key=lambda item: (
                    item.exchange,
                    item.market_type,
                    item.symbol,
                    item.trade_id or item.timestamp_ms,
                ),
            )
            bundle.orderbooks = self._dedupe(
                bundle.orderbooks,
                key=lambda item: (
                    item.exchange,
                    item.market_type,
                    item.symbol,
                    item.timestamp_ms,
                    item.sequence,
                ),
            )
            bundle.funding = self._dedupe(
                bundle.funding,
                key=lambda item: (
                    item.exchange,
                    item.market_type,
                    item.symbol,
                    item.timestamp_ms,
                ),
            )
            bundle.open_interest = self._dedupe(
                bundle.open_interest,
                key=lambda item: (
                    item.exchange,
                    item.market_type,
                    item.symbol,
                    item.timestamp_ms,
                ),
            )
            bundle.liquidations = self._dedupe(
                bundle.liquidations,
                key=lambda item: (
                    item.exchange,
                    item.market_type,
                    item.symbol,
                    item.liquidation_id or item.timestamp_ms,
                ),
            )

    @staticmethod
    def _is_gap_error_policy(policy: DataGapPolicy) -> bool:
        return str(getattr(policy, "value", policy)).lower() in {
            "error",
            "raise",
            "strict",
        }
    @staticmethod
    def _dedupe(records: Iterable[Any], *, key: Any) -> list[Any]:
        seen: set[Any] = set()
        result: list[Any] = []

        for record in records:
            value = key(record)
            if value in seen:
                continue
            seen.add(value)
            result.append(record)

        return result

    @staticmethod
    def _filter_records_by_period(
        records: list[Any],
        period: BacktestPeriod | None,
    ) -> list[Any]:
        if period is None:
            return records

        start_ms = period.warmup_start_ms
        end_ms = period.end_ms

        return [
            record
            for record in records
            if start_ms <= DataLoader._record_timestamp_ms(record) <= end_ms
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_required(self, data_type: BacktestDataType) -> bool:
        if data_type == BacktestDataType.CANDLES:
            return self.config.require_candles
        if data_type == BacktestDataType.TRADES:
            return self.config.require_trades
        if data_type in {BacktestDataType.ORDERBOOK, BacktestDataType.ORDERBOOK_SNAPSHOT}:
            return self.config.require_orderbook
        if data_type == BacktestDataType.FUNDING:
            return self.config.require_funding
        if data_type == BacktestDataType.OPEN_INTEREST:
            return self.config.require_open_interest
        return False

    @staticmethod
    def _data_type_dir_name(data_type: BacktestDataType) -> str:
        if data_type == BacktestDataType.ORDERBOOK_SNAPSHOT:
            return "orderbook_snapshot"
        return data_type.value

    def _candidate_formats(self) -> list[HistoricalDataFormat]:
        if self.config.input_format:
            # Prefer configured format, then try common fallbacks.
            formats = [
                self.config.input_format,
                HistoricalDataFormat.PARQUET,
                HistoricalDataFormat.CSV,
                HistoricalDataFormat.JSONL,
                HistoricalDataFormat.JSON,
            ]
        else:
            formats = [
                HistoricalDataFormat.PARQUET,
                HistoricalDataFormat.CSV,
                HistoricalDataFormat.JSONL,
                HistoricalDataFormat.JSON,
            ]

        unique: list[HistoricalDataFormat] = []
        for item in formats:
            if item not in unique:
                unique.append(item)

        return unique

    @staticmethod
    def _format_extension(value: HistoricalDataFormat) -> str:
        if value == HistoricalDataFormat.PARQUET:
            return "parquet"
        if value == HistoricalDataFormat.CSV:
            return "csv"
        if value == HistoricalDataFormat.JSONL:
            return "jsonl"
        if value == HistoricalDataFormat.JSON:
            return "json"

        raise HistoricalDataFormatError(
            "Unsupported historical data format.",
            details={"format": value.value},
        )

    @staticmethod
    def _format_from_suffix(suffix: str) -> HistoricalDataFormat | None:
        value = suffix.lower().lstrip(".")

        if value == "parquet":
            return HistoricalDataFormat.PARQUET
        if value == "csv":
            return HistoricalDataFormat.CSV
        if value == "jsonl":
            return HistoricalDataFormat.JSONL
        if value == "json":
            return HistoricalDataFormat.JSON

        return None

    @staticmethod
    def _record_timestamp_ms(record: Any) -> int:
        for attr in ("timestamp_ms", "close_time_ms", "open_time_ms"):
            value = getattr(record, attr, None)
            if value is not None:
                return int(value)

        raise DataNormalizationError(
            "Historical record has no timestamp.",
            details={"record_type": record.__class__.__name__},
        )

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

        raise DataNormalizationError(
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

        raise DataNormalizationError(
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

    @staticmethod
    def _bool_from_keys(
        row: dict[str, Any],
        keys: Sequence[str],
        *,
        default: bool,
    ) -> bool:
        value = None

        for key in keys:
            if key in row:
                value = row.get(key)
                break

        if value is None:
            return default

        if isinstance(value, bool):
            return value

        if isinstance(value, int | float):
            return bool(value)

        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False

        return default

    @staticmethod
    def _optional_bool_from_keys(
        row: dict[str, Any],
        keys: Sequence[str],
    ) -> bool | None:
        for key in keys:
            value = row.get(key)
            if value is None or value == "":
                continue

            if isinstance(value, bool):
                return value

            if isinstance(value, int | float):
                return bool(value)

            text = str(value).strip().lower()
            if text in {"true", "1", "yes", "y"}:
                return True
            if text in {"false", "0", "no", "n"}:
                return False

        return None

    @staticmethod
    def _optional_str_from_keys(
        row: dict[str, Any],
        keys: Sequence[str],
    ) -> str | None:
        for key in keys:
            value = row.get(key)
            if value is not None and value != "":
                return str(value)
        return None

    @staticmethod
    def _metadata(row: dict[str, Any]) -> dict[str, Any]:
        metadata = row.get("metadata")

        if isinstance(metadata, dict):
            return dict(metadata)

        if isinstance(metadata, str) and metadata:
            try:
                value = json.loads(metadata)
                if isinstance(value, dict):
                    return value
            except Exception:
                return {"raw_metadata": metadata}

        return {}

    @staticmethod
    def _timeframe_to_milliseconds(timeframe: str) -> int:
        value = timeframe.strip().lower()

        units = {
            "m": 60_000,
            "h": 60 * 60_000,
            "d": 24 * 60 * 60_000,
            "w": 7 * 24 * 60 * 60_000,
        }

        if len(value) < 2:
            raise DataValidationError(
                "Invalid timeframe.",
                details={"timeframe": timeframe},
            )

        amount = int(value[:-1])
        unit = value[-1]

        if unit not in units:
            raise DataValidationError(
                "Unsupported timeframe unit.",
                details={"timeframe": timeframe},
            )

        return amount * units[unit]

    def stats(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.config.data_dir),
            "input_format": self.config.input_format.value,
            "exchange": self.config.exchange,
            "market_type": self.config.market_type,
            "symbols": list(self.config.symbols),
            "timeframes": list(self.config.timeframes),
            "data_types": [item.value for item in self.config.data_types],
            "validation_level": self.config.validation_level.value,
            "gap_policy": self.config.gap_policy.value,
        }


# =============================================================================
# Convenience helpers
# =============================================================================


def load_backtest_dataset(
    config: DataLoaderConfig,
    *,
    period: BacktestPeriod | None = None,
    run_id: str | None = None,
) -> BacktestDataset:
    """
    Convenience helper for loading a BacktestDataset.
    """

    loader = DataLoader(config)
    return loader.load_dataset(period=period, run_id=run_id)


def load_backtest_bundle(
    config: DataLoaderConfig,
    *,
    period: BacktestPeriod | None = None,
) -> LoadedDataBundle:
    """
    Convenience helper for loading normalized historical records.
    """

    loader = DataLoader(config)
    return loader.load_bundle(period=period)


__all__ = [
    "DataFileRef",
    "LoadedDataBundle",
    "DataLoader",
    "load_backtest_dataset",
    "load_backtest_bundle",
]