from __future__ import annotations

import contextlib
import inspect
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.dataset as ds

try:
    from core.logger import get_logger
except Exception:  # pragma: no cover - keeps loader usable in offline scripts/tests.
    import logging

    def get_logger(name: str, **_: Any) -> logging.Logger:
        return logging.getLogger(name)


@dataclass(slots=True)
class ParquetLoaderConfig:
    """
    Read-side config for partitioned Parquet market data.

    This class intentionally mirrors the write-side layout produced by
    storage.parquet_storage. It does not subscribe to EventBus and does not
    write files; it only loads persisted records and can optionally feed them
    back through MarketIngestionService so analytics can rebuild MarketState.
    """

    root_dir: str = "data/parquet"

    candles_dataset: str = "candles"
    funding_dataset: str = "funding"
    analytics_dataset: str = "analytics_events"
    strategy_input_dataset: str = "strategy_input_events"
    strategy_events_dataset: str = "strategy_events"
    signal_events_dataset: str = "signal_events"
    risk_events_dataset: str = "risk_events"
    execution_events_dataset: str = "execution_events"
    position_events_dataset: str = "position_events"

    default_exchange: str = "binance"
    default_market_type: str = "usdm_futures"
    default_source: str = "parquet_loader"

    batch_size: int = 1_000
    evaluate_after_load: bool = True
    emit_loader_events: bool = True


@dataclass(slots=True)
class ParquetLoadResult:
    dataset: str
    exchange: str | None = None
    market_type: str | None = None
    symbols: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=list)
    requested_start_ms: int | None = None
    requested_end_ms: int | None = None
    rows_read: int = 0
    rows_loaded: int = 0
    batches_loaded: int = 0
    files_scanned: int = 0
    elapsed_ms: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbols": list(self.symbols),
            "timeframes": list(self.timeframes),
            "requested_start_ms": self.requested_start_ms,
            "requested_end_ms": self.requested_end_ms,
            "rows_read": self.rows_read,
            "rows_loaded": self.rows_loaded,
            "batches_loaded": self.batches_loaded,
            "files_scanned": self.files_scanned,
            "elapsed_ms": self.elapsed_ms,
            "errors": list(self.errors),
            "ok": self.ok,
        }


class ParquetMarketLoader:
    """
    Read-side companion for ParquetStorage.

    Typical runtime/replay path:

        ParquetMarketLoader.read_candles(...)
            -> MarketIngestionService.ingest_candles_batch(...)
            -> MarketStateStore
            -> MarketScheduler.evaluate_dirty_once()
            -> analytics process_market_snapshot()

    The loader supports the partitioned layout produced by ParquetStorage:

        candles/exchange=binance/symbol=BTCUSDT/market_type=usdm_futures/timeframe=1m/date=YYYY-MM-DD/*.parquet
        funding/exchange=binance/symbol=BTCUSDT/market_type=usdm_futures/date=YYYY-MM-DD/*.parquet

    It is deliberately tolerant around ingestion method signatures. If your
    MarketIngestionService exposes `ingest_candles_batch` / `ingest_funding_batch`,
    those are used. If the exact signature differs, kwargs are filtered through
    inspect.signature and several positional fallbacks are attempted.
    """

    DATASET_CANDLES = "candles"
    DATASET_FUNDING = "funding"
    DATASET_ANALYTICS = "analytics_events"
    DATASET_STRATEGY_INPUT = "strategy_input_events"
    DATASET_STRATEGY = "strategy_events"
    DATASET_SIGNAL = "signal_events"
    DATASET_RISK = "risk_events"
    DATASET_EXECUTION = "execution_events"
    DATASET_POSITION = "position_events"

    def __init__(
        self,
        *,
        config: ParquetLoaderConfig | None = None,
        root_dir: str | Path | None = None,
        market_ingestion: Any | None = None,
        market_scheduler: Any | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self.config = config or ParquetLoaderConfig()
        if root_dir is not None:
            self.config.root_dir = str(root_dir)

        self.root_dir = Path(self.config.root_dir)
        self.market_ingestion = market_ingestion
        self.market_scheduler = market_scheduler
        self.event_bus = event_bus
        self._logger = get_logger(
            __name__,
            service="parquet_loader",
            event_type="storage_parquet_loader",
        )

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def read_candles(
        self,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        symbol: str | None = None,
        symbols: Sequence[str] | None = None,
        timeframe: str | None = None,
        timeframes: Sequence[str] | None = None,
        start_ms: int | float | str | datetime | None = None,
        end_ms: int | float | str | datetime | None = None,
        columns: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        exchange = self._normalize_exchange(exchange)
        market_type = self._normalize_market_type(market_type)
        symbols_list = self._normalize_list(symbols or ([symbol] if symbol else []), upper=True)
        timeframes_list = self._normalize_list(timeframes or ([timeframe] if timeframe else []), upper=False)
        start = self._normalize_timestamp_ms(start_ms)
        end = self._normalize_timestamp_ms(end_ms)

        records = self._read_dataset_records(
            dataset=self.config.candles_dataset,
            exchange=exchange,
            market_type=market_type,
            symbols=symbols_list,
            timeframes=timeframes_list,
            start_ms=start,
            end_ms=end,
            timestamp_fields=("open_time_ms", "timestamp_ms", "close_time_ms"),
            columns=columns,
        )
        records.sort(key=lambda row: (str(row.get("symbol") or ""), str(row.get("timeframe") or ""), self._row_time_ms(row)))
        return [self._clean_candle_record(row) for row in records]

    def read_funding(
        self,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        symbol: str | None = None,
        symbols: Sequence[str] | None = None,
        start_ms: int | float | str | datetime | None = None,
        end_ms: int | float | str | datetime | None = None,
        columns: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        exchange = self._normalize_exchange(exchange)
        market_type = self._normalize_market_type(market_type)
        symbols_list = self._normalize_list(symbols or ([symbol] if symbol else []), upper=True)
        start = self._normalize_timestamp_ms(start_ms)
        end = self._normalize_timestamp_ms(end_ms)

        records = self._read_dataset_records(
            dataset=self.config.funding_dataset,
            exchange=exchange,
            market_type=market_type,
            symbols=symbols_list,
            start_ms=start,
            end_ms=end,
            timestamp_fields=("timestamp_ms", "funding_time_ms", "funding_time"),
            columns=columns,
        )
        records.sort(key=lambda row: (str(row.get("symbol") or ""), self._row_time_ms(row)))
        return [self._clean_funding_record(row) for row in records]

    def read_events(
        self,
        *,
        dataset: str,
        exchange: str | None = None,
        market_type: str | None = None,
        symbol: str | None = None,
        symbols: Sequence[str] | None = None,
        timeframe: str | None = None,
        timeframes: Sequence[str] | None = None,
        event_type: str | None = None,
        analytics_type: str | None = None,
        strategy_name: str | None = None,
        start_ms: int | float | str | datetime | None = None,
        end_ms: int | float | str | datetime | None = None,
        columns: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Read generic event datasets: analytics, strategy, signal, risk, execution, position."""
        records = self._read_dataset_records(
            dataset=dataset,
            exchange=self._normalize_exchange(exchange) if exchange else None,
            market_type=self._normalize_market_type(market_type) if market_type else None,
            symbols=self._normalize_list(symbols or ([symbol] if symbol else []), upper=True),
            timeframes=self._normalize_list(timeframes or ([timeframe] if timeframe else []), upper=False),
            event_type=event_type,
            analytics_type=analytics_type,
            strategy_name=strategy_name,
            start_ms=self._normalize_timestamp_ms(start_ms),
            end_ms=self._normalize_timestamp_ms(end_ms),
            timestamp_fields=("timestamp_ms", "ingested_at_ms"),
            columns=columns,
        )
        records.sort(key=self._row_time_ms)
        return records

    # ------------------------------------------------------------------
    # Public load/replay API
    # ------------------------------------------------------------------

    async def load_candles_to_ingestion(
        self,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        symbol: str | None = None,
        symbols: Sequence[str] | None = None,
        timeframe: str | None = None,
        timeframes: Sequence[str] | None = None,
        start_ms: int | float | str | datetime | None = None,
        end_ms: int | float | str | datetime | None = None,
        batch_size: int | None = None,
        evaluate_after_load: bool | None = None,
        source: str | None = None,
    ) -> ParquetLoadResult:
        started = time.time()
        exchange = self._normalize_exchange(exchange)
        market_type = self._normalize_market_type(market_type)
        symbols_list = self._normalize_list(symbols or ([symbol] if symbol else []), upper=True)
        timeframes_list = self._normalize_list(timeframes or ([timeframe] if timeframe else []), upper=False)
        start = self._normalize_timestamp_ms(start_ms)
        end = self._normalize_timestamp_ms(end_ms)
        source = source or self.config.default_source
        batch_size = max(1, int(batch_size or self.config.batch_size))

        result = ParquetLoadResult(
            dataset=self.config.candles_dataset,
            exchange=exchange,
            market_type=market_type,
            symbols=symbols_list,
            timeframes=timeframes_list,
            requested_start_ms=start,
            requested_end_ms=end,
        )

        try:
            rows = self.read_candles(
                exchange=exchange,
                market_type=market_type,
                symbols=symbols_list,
                timeframes=timeframes_list,
                start_ms=start,
                end_ms=end,
            )
            result.rows_read = len(rows)

            grouped = self._group_by(rows, "symbol", "timeframe")
            for (row_symbol, row_timeframe), group_rows in grouped.items():
                for batch in self._chunks(group_rows, batch_size):
                    await self._ingest_candles_batch(
                        exchange=exchange,
                        market_type=market_type,
                        symbol=str(row_symbol),
                        timeframe=str(row_timeframe),
                        candles=batch,
                        source=source,
                    )
                    result.rows_loaded += len(batch)
                    result.batches_loaded += 1

            if self._should_evaluate(evaluate_after_load):
                await self.evaluate_market_state_once(reason="parquet_candles_loaded")

        except Exception as exc:
            result.errors.append(str(exc))
            self._logger.exception("Parquet candle load failed")

        result.elapsed_ms = int((time.time() - started) * 1000)
        await self._emit_loader_event("storage.parquet_loader.candles_loaded", result.to_dict())
        return result

    async def load_funding_to_ingestion(
        self,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        symbol: str | None = None,
        symbols: Sequence[str] | None = None,
        start_ms: int | float | str | datetime | None = None,
        end_ms: int | float | str | datetime | None = None,
        batch_size: int | None = None,
        evaluate_after_load: bool | None = None,
        source: str | None = None,
    ) -> ParquetLoadResult:
        started = time.time()
        exchange = self._normalize_exchange(exchange)
        market_type = self._normalize_market_type(market_type)
        symbols_list = self._normalize_list(symbols or ([symbol] if symbol else []), upper=True)
        start = self._normalize_timestamp_ms(start_ms)
        end = self._normalize_timestamp_ms(end_ms)
        source = source or self.config.default_source
        batch_size = max(1, int(batch_size or self.config.batch_size))

        result = ParquetLoadResult(
            dataset=self.config.funding_dataset,
            exchange=exchange,
            market_type=market_type,
            symbols=symbols_list,
            requested_start_ms=start,
            requested_end_ms=end,
        )

        try:
            rows = self.read_funding(
                exchange=exchange,
                market_type=market_type,
                symbols=symbols_list,
                start_ms=start,
                end_ms=end,
            )
            result.rows_read = len(rows)

            grouped = self._group_by(rows, "symbol")
            for (row_symbol,), group_rows in grouped.items():
                for batch in self._chunks(group_rows, batch_size):
                    await self._ingest_funding_batch(
                        exchange=exchange,
                        market_type=market_type,
                        symbol=str(row_symbol),
                        funding=batch,
                        source=source,
                    )
                    result.rows_loaded += len(batch)
                    result.batches_loaded += 1

            if self._should_evaluate(evaluate_after_load):
                await self.evaluate_market_state_once(reason="parquet_funding_loaded")

        except Exception as exc:
            result.errors.append(str(exc))
            self._logger.exception("Parquet funding load failed")

        result.elapsed_ms = int((time.time() - started) * 1000)
        await self._emit_loader_event("storage.parquet_loader.funding_loaded", result.to_dict())
        return result

    async def load_market_data_to_ingestion(
        self,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        symbols: Sequence[str],
        timeframes: Sequence[str],
        start_ms: int | float | str | datetime | None = None,
        end_ms: int | float | str | datetime | None = None,
        load_candles: bool = True,
        load_funding: bool = True,
        batch_size: int | None = None,
        evaluate_after_load: bool | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Convenience method for bootstrapping analytics from Parquet before live WS starts."""
        results: dict[str, Any] = {}

        # Avoid evaluating twice; evaluate once at the end if requested.
        should_eval = self._should_evaluate(evaluate_after_load)

        if load_candles:
            candle_result = await self.load_candles_to_ingestion(
                exchange=exchange,
                market_type=market_type,
                symbols=symbols,
                timeframes=timeframes,
                start_ms=start_ms,
                end_ms=end_ms,
                batch_size=batch_size,
                evaluate_after_load=False,
                source=source,
            )
            results["candles"] = candle_result.to_dict()

        if load_funding:
            funding_result = await self.load_funding_to_ingestion(
                exchange=exchange,
                market_type=market_type,
                symbols=symbols,
                start_ms=start_ms,
                end_ms=end_ms,
                batch_size=batch_size,
                evaluate_after_load=False,
                source=source,
            )
            results["funding"] = funding_result.to_dict()

        if should_eval:
            results["market_state_evaluation"] = await self.evaluate_market_state_once(reason="parquet_market_data_loaded")

        return results

    async def evaluate_market_state_once(self, *, reason: str = "parquet_loader") -> Any:
        if self.market_scheduler is None:
            return None

        evaluate = getattr(self.market_scheduler, "evaluate_dirty_once", None)
        if not callable(evaluate):
            return None

        value = evaluate()
        if inspect.isawaitable(value):
            value = await value
        await self._emit_loader_event(
            "storage.parquet_loader.market_state_evaluated",
            {"reason": reason, "result": self._json_safe_value(value)},
        )
        return value

    # ------------------------------------------------------------------
    # Dataset reading
    # ------------------------------------------------------------------

    def dataset_path(self, dataset: str) -> Path:
        return self.root_dir / dataset

    def dataset_exists(self, dataset: str) -> bool:
        path = self.dataset_path(dataset)
        return path.exists() and any(path.rglob("*.parquet"))

    def count_files(self, dataset: str) -> int:
        path = self.dataset_path(dataset)
        if not path.exists():
            return 0
        return sum(1 for _ in path.rglob("*.parquet"))

    def _read_dataset_records(
        self,
        *,
        dataset: str,
        exchange: str | None = None,
        market_type: str | None = None,
        symbols: Sequence[str] | None = None,
        timeframes: Sequence[str] | None = None,
        event_type: str | None = None,
        analytics_type: str | None = None,
        strategy_name: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        timestamp_fields: Sequence[str] = ("timestamp_ms",),
        columns: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        path = self.dataset_path(dataset)
        if not path.exists():
            self._logger.warning("Parquet dataset path missing | dataset=%s path=%s", dataset, path)
            return []

        if not any(path.rglob("*.parquet")):
            return []

        filters = self._build_filter(
            exchange=exchange,
            market_type=market_type,
            symbols=symbols,
            timeframes=timeframes,
            event_type=event_type,
            analytics_type=analytics_type,
            strategy_name=strategy_name,
            start_ms=start_ms,
            end_ms=end_ms,
            timestamp_fields=timestamp_fields,
        )

        dataset_obj = ds.dataset(str(path), format="parquet", partitioning="hive")
        table = dataset_obj.to_table(columns=list(columns) if columns else None, filter=filters)
        records = table.to_pylist()

        # Fallback in-process time filtering is intentional: some older files may
        # have different timestamp columns, and PyArrow filters only the first
        # matching schema field robustly.
        if start_ms is not None or end_ms is not None:
            records = [
                row
                for row in records
                if self._time_in_range(self._row_time_ms(row), start_ms=start_ms, end_ms=end_ms)
            ]

        return [self._plain_record(row) for row in records]

    def _build_filter(
        self,
        *,
        exchange: str | None,
        market_type: str | None,
        symbols: Sequence[str] | None,
        timeframes: Sequence[str] | None,
        event_type: str | None,
        analytics_type: str | None,
        strategy_name: str | None,
        start_ms: int | None,
        end_ms: int | None,
        timestamp_fields: Sequence[str],
    ) -> ds.Expression | None:
        expr: ds.Expression | None = None

        def add(condition: ds.Expression) -> None:
            nonlocal expr
            expr = condition if expr is None else expr & condition

        if exchange:
            add(ds.field("exchange") == exchange)
        if market_type:
            add(ds.field("market_type") == market_type)
        if symbols:
            normalized = [str(s).upper() for s in symbols]
            add(ds.field("symbol").isin(normalized))
        if timeframes:
            add(ds.field("timeframe").isin([str(tf) for tf in timeframes]))
        if event_type:
            add(ds.field("event_type") == str(event_type))
        if analytics_type:
            add(ds.field("analytics_type") == str(analytics_type))
        if strategy_name:
            add(ds.field("strategy_name") == str(strategy_name))

        # Date partition pruning. We still apply precise timestamp filtering after read.
        date_expr = self._date_partition_filter(start_ms=start_ms, end_ms=end_ms)
        if date_expr is not None:
            add(date_expr)

        return expr

    def _date_partition_filter(self, *, start_ms: int | None, end_ms: int | None) -> ds.Expression | None:
        if start_ms is None and end_ms is None:
            return None

        start_date = self._partition_date(start_ms) if start_ms is not None else None
        end_date = self._partition_date(end_ms) if end_ms is not None else None

        if start_date and end_date:
            return (ds.field("date") >= start_date) & (ds.field("date") <= end_date)
        if start_date:
            return ds.field("date") >= start_date
        if end_date:
            return ds.field("date") <= end_date
        return None

    # ------------------------------------------------------------------
    # Ingestion bridges
    # ------------------------------------------------------------------

    async def _ingest_candles_batch(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
        candles: list[dict[str, Any]],
        source: str,
    ) -> None:
        if self.market_ingestion is None:
            raise RuntimeError("market_ingestion is required to load candles into analytics state")

        candidates = (
            "ingest_candles_batch",
            "ingest_candle_batch",
            "ingest_candles",
            "apply_candles_batch",
        )
        method = self._first_callable(self.market_ingestion, candidates)
        if method is None:
            # Last resort: call single-candle ingestion method repeatedly.
            single = self._first_callable(self.market_ingestion, ("ingest_candle", "apply_candle"))
            if single is None:
                raise RuntimeError("MarketIngestionService has no candle ingestion method")
            for candle in candles:
                await self._call_ingestion_method(
                    single,
                    exchange=exchange,
                    market_type=market_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    candle=candle,
                    source=source,
                )
            return

        await self._call_ingestion_method(
            method,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            items=candles,
            source=source,
        )

    async def _ingest_funding_batch(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        funding: list[dict[str, Any]],
        source: str,
    ) -> None:
        if self.market_ingestion is None:
            raise RuntimeError("market_ingestion is required to load funding into analytics state")

        candidates = (
            "ingest_funding_batch",
            "ingest_funding_rates_batch",
            "ingest_funding_snapshot",
            "ingest_funding_rates",
            "ingest_funding",
            "apply_funding_batch",
        )
        method = self._first_callable(self.market_ingestion, candidates)
        if method is None:
            raise RuntimeError("MarketIngestionService has no funding ingestion method")

        await self._call_ingestion_method(
            method,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            funding=funding,
            rates=funding,
            items=funding,
            source=source,
        )

    async def _call_ingestion_method(self, method: Any, **kwargs: Any) -> Any:
        filtered = self._filter_supported_kwargs(method, kwargs)

        attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            ((), filtered),
        ]

        # Positional fallbacks for common service signatures.
        if "candles" in kwargs:
            attempts.extend(
                [
                    ((kwargs["exchange"], kwargs["market_type"], kwargs["symbol"], kwargs["timeframe"], kwargs["candles"]), {"source": kwargs.get("source")}),
                    ((kwargs["symbol"], kwargs["timeframe"], kwargs["candles"]), {"exchange": kwargs["exchange"], "market_type": kwargs["market_type"], "source": kwargs.get("source")}),
                    ((kwargs["candles"],), filtered),
                ]
            )
        elif "funding" in kwargs:
            attempts.extend(
                [
                    ((kwargs["exchange"], kwargs["market_type"], kwargs["symbol"], kwargs["funding"]), {"source": kwargs.get("source")}),
                    ((kwargs["symbol"], kwargs["funding"]), {"exchange": kwargs["exchange"], "market_type": kwargs["market_type"], "source": kwargs.get("source")}),
                    ((kwargs["funding"],), filtered),
                ]
            )
        elif "candle" in kwargs:
            attempts.extend(
                [
                    ((kwargs["exchange"], kwargs["market_type"], kwargs["symbol"], kwargs["timeframe"], kwargs["candle"]), {"source": kwargs.get("source")}),
                    ((kwargs["candle"],), filtered),
                ]
            )

        last_error: Exception | None = None
        for args, call_kwargs in attempts:
            try:
                call_kwargs = self._filter_supported_kwargs(method, call_kwargs)
                value = method(*args, **call_kwargs)
                if inspect.isawaitable(value):
                    value = await value
                return value
            except TypeError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to call ingestion method {method!r}")

    # ------------------------------------------------------------------
    # Record shaping
    # ------------------------------------------------------------------

    def _clean_candle_record(self, row: Mapping[str, Any]) -> dict[str, Any]:
        open_time_ms = self._normalize_timestamp_ms(
            row.get("open_time_ms")
            or row.get("open_time")
            or row.get("timestamp_ms")
            or row.get("timestamp")
        )
        close_time_ms = self._normalize_timestamp_ms(
            row.get("close_time_ms")
            or row.get("close_time")
            or row.get("end_time_ms")
            or row.get("end_time")
        )
        timeframe = self._safe_str(row.get("timeframe")) or "1m"
        if close_time_ms is None and open_time_ms is not None:
            close_time_ms = open_time_ms + self._timeframe_to_ms(timeframe) - 1

        record = {
            "exchange": self._safe_str(row.get("exchange")) or self.config.default_exchange,
            "symbol": self._safe_str(row.get("symbol")),
            "market_type": self._safe_str(row.get("market_type")) or self.config.default_market_type,
            "timeframe": timeframe,
            "open_time_ms": open_time_ms,
            "close_time_ms": close_time_ms,
            "timestamp_ms": self._normalize_timestamp_ms(row.get("timestamp_ms")) or open_time_ms,
            "open": self._safe_float(row.get("open")),
            "high": self._safe_float(row.get("high")),
            "low": self._safe_float(row.get("low")),
            "close": self._safe_float(row.get("close")),
            "volume": self._safe_float(row.get("volume")),
            "quote_volume": self._safe_float(row.get("quote_volume")),
            "trades_count": self._safe_int(row.get("trades_count")),
            "is_closed": True,
            "source": row.get("source") or self.config.default_source,
        }
        # Preserve additional fields for downstream compatibility/debugging.
        for key, value in row.items():
            record.setdefault(str(key), self._json_safe_value(value))
        return record

    def _clean_funding_record(self, row: Mapping[str, Any]) -> dict[str, Any]:
        timestamp_ms = self._normalize_timestamp_ms(
            row.get("timestamp_ms")
            or row.get("funding_time_ms")
            or row.get("funding_time")
            or row.get("timestamp")
        )
        record = {
            "exchange": self._safe_str(row.get("exchange")) or self.config.default_exchange,
            "symbol": self._safe_str(row.get("symbol")),
            "market_type": self._safe_str(row.get("market_type")) or self.config.default_market_type,
            "timestamp_ms": timestamp_ms,
            "funding_time_ms": timestamp_ms,
            "funding_rate": self._safe_float(row.get("funding_rate") or row.get("rate")),
            "predicted_rate": self._safe_float(row.get("predicted_rate")),
            "next_funding_time_ms": self._normalize_timestamp_ms(row.get("next_funding_time_ms") or row.get("next_funding_time")),
            "mark_price": self._safe_float(row.get("mark_price")),
            "index_price": self._safe_float(row.get("index_price")),
            "source": row.get("source") or self.config.default_source,
        }
        for key, value in row.items():
            record.setdefault(str(key), self._json_safe_value(value))
        return record

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_callable(obj: Any, names: Iterable[str]) -> Any | None:
        for name in names:
            method = getattr(obj, name, None)
            if callable(method):
                return method
        return None

    @staticmethod
    def _supports_kw(callable_obj: Any, key: str) -> bool:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False
        for parameter in signature.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                return True
        return key in signature.parameters

    @classmethod
    def _filter_supported_kwargs(cls, callable_obj: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in kwargs.items() if cls._supports_kw(callable_obj, key)}

    @staticmethod
    def _chunks(items: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
        for index in range(0, len(items), size):
            yield list(items[index : index + size])

    @staticmethod
    def _group_by(rows: Sequence[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
        result: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            group_key = tuple(row.get(key) for key in keys)
            result.setdefault(group_key, []).append(row)
        return result

    def _should_evaluate(self, value: bool | None) -> bool:
        if value is None:
            return self.config.evaluate_after_load
        return bool(value)

    def _normalize_exchange(self, exchange: str | None) -> str:
        return str(exchange or self.config.default_exchange).strip().lower()

    def _normalize_market_type(self, market_type: str | None) -> str:
        return str(market_type or self.config.default_market_type).strip().lower()

    @staticmethod
    def _normalize_list(values: Sequence[str] | None, *, upper: bool) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values or []:
            text = str(value).strip()
            if not text:
                continue
            text = text.upper() if upper else text
            if text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _row_time_ms(self, row: Mapping[str, Any]) -> int:
        for key in ("open_time_ms", "timestamp_ms", "close_time_ms", "funding_time_ms", "ingested_at_ms"):
            value = self._normalize_timestamp_ms(row.get(key))
            if value is not None:
                return value
        return 0

    @staticmethod
    def _time_in_range(value_ms: int, *, start_ms: int | None, end_ms: int | None) -> bool:
        if start_ms is not None and value_ms < start_ms:
            return False
        if end_ms is not None and value_ms > end_ms:
            return False
        return True

    @staticmethod
    def _plain_record(row: Mapping[str, Any]) -> dict[str, Any]:
        return {str(key): value.as_py() if hasattr(value, "as_py") else value for key, value in row.items()}

    def _partition_date(self, timestamp_ms: int | None) -> str:
        timestamp = timestamp_ms or self._now_ms()
        return datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")

    def _normalize_timestamp_ms(self, value: int | float | str | datetime | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            with contextlib.suppress(ValueError, TypeError):
                value = float(text)
            if isinstance(value, str):
                with contextlib.suppress(ValueError):
                    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return int(dt.timestamp() * 1000)
                return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        absolute = abs(numeric)
        if absolute == 0:
            return 0
        if absolute < 10_000_000_000:
            return int(numeric * 1000)
        if absolute < 10_000_000_000_000:
            return int(numeric)
        if absolute < 10_000_000_000_000_000:
            return int(numeric / 1000)
        return int(numeric / 1_000_000)

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        with contextlib.suppress(TypeError, ValueError):
            return int(float(value))
        return None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        with contextlib.suppress(TypeError, ValueError):
            number = float(value)
            if math.isfinite(number):
                return number
        return None

    def _timeframe_to_ms(self, timeframe: str) -> int:
        value = str(timeframe or "1m").strip().lower()
        aliases = {
            "min1": "1m",
            "min5": "5m",
            "min15": "15m",
            "min30": "30m",
            "hour1": "1h",
            "day1": "1d",
        }
        value = aliases.get(value, value)
        if len(value) < 2:
            return 60_000
        unit = value[-1]
        with contextlib.suppress(ValueError):
            amount = int(value[:-1])
            if unit == "m":
                return amount * 60_000
            if unit == "h":
                return amount * 60 * 60_000
            if unit == "d":
                return amount * 24 * 60 * 60_000
            if unit == "w":
                return amount * 7 * 24 * 60 * 60_000
        return 60_000

    def _json_safe_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Mapping):
            return {str(key): self._json_safe_value(inner) for key, inner in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe_value(item) for item in value]
        if isinstance(value, pa.Scalar):
            return self._json_safe_value(value.as_py())
        return str(value)

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    async def _emit_loader_event(self, topic: str, payload: Mapping[str, Any]) -> None:
        if not self.config.emit_loader_events or self.event_bus is None:
            return
        emit = getattr(self.event_bus, "emit", None)
        if not callable(emit):
            return
        with contextlib.suppress(Exception):
            value = emit(topic, payload={"service": "parquet_loader", **dict(payload)}, source="storage.parquet_loader")
            if inspect.isawaitable(value):
                await value


__all__ = [
    "ParquetLoaderConfig",
    "ParquetLoadResult",
    "ParquetMarketLoader",
]