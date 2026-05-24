from __future__ import annotations

import contextlib
import inspect
import json
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

    Stage 3 responsibility:
    - read datasets written by ParquetStorage;
    - hydrate MarketStateStore through MarketIngestionService;
    - optionally trigger MarketScheduler once after restore;
    - never rewrite replayed records back into Parquet.
    """

    root_dir: str = "data/parquet"

    candles_dataset: str = "candles"
    trades_dataset: str = "trades"
    orderbook_snapshots_dataset: str = "orderbook_snapshots"
    funding_dataset: str = "funding"
    open_interest_dataset: str = "open_interest"
    liquidations_dataset: str = "liquidations"

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

    load_candles: bool = True
    load_trades: bool = False
    load_orderbook_snapshots: bool = True
    load_funding: bool = True
    load_open_interest: bool = True
    load_liquidations: bool = True

    suppress_persistable_on_replay: bool = True


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
    Restore/replay helper for state-driven market data.

    Correct startup path:

        ParquetMarketLoader.load_market_data_to_ingestion(...)
            -> MarketIngestionService.ingest_*_batch(..., suppress_persistable_events=True)
            -> MarketStateStore
            -> MarketScheduler.evaluate_dirty_once()
            -> analytics read consistent snapshots
    """

    DATASET_CANDLES = "candles"
    DATASET_TRADES = "trades"
    DATASET_ORDERBOOK_SNAPSHOTS = "orderbook_snapshots"
    DATASET_FUNDING = "funding"
    DATASET_OPEN_INTEREST = "open_interest"
    DATASET_LIQUIDATIONS = "liquidations"

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
        self._logger = get_logger(__name__, service="parquet_loader", event_type="storage_parquet_loader")

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def read_candles(self, **kwargs: Any) -> list[dict[str, Any]]:
        rows = self._read_market_dataset(self.config.candles_dataset, timestamp_fields=("open_time_ms", "timestamp_ms", "close_time_ms"), **kwargs)
        rows.sort(key=lambda row: (str(row.get("symbol") or ""), str(row.get("timeframe") or ""), self._row_time_ms(row)))
        return self._dedupe_records([self._clean_candle_record(row) for row in rows], keys=("exchange", "market_type", "symbol", "timeframe", "open_time_ms"))

    def read_trades(self, **kwargs: Any) -> list[dict[str, Any]]:
        rows = self._read_market_dataset(self.config.trades_dataset, timestamp_fields=("timestamp_ms", "trade_time_ms", "ingested_at_ms"), **kwargs)
        rows.sort(key=lambda row: (str(row.get("symbol") or ""), self._row_time_ms(row), str(row.get("trade_id") or "")))
        return self._dedupe_records([self._clean_trade_record(row) for row in rows], keys=("exchange", "market_type", "symbol", "trade_id", "timestamp_ms", "price", "quantity"))

    def read_orderbook_snapshots(self, **kwargs: Any) -> list[dict[str, Any]]:
        rows = self._read_market_dataset(self.config.orderbook_snapshots_dataset, timestamp_fields=("timestamp_ms", "ingested_at_ms"), **kwargs)
        rows.sort(key=lambda row: (str(row.get("symbol") or ""), self._row_time_ms(row), self._safe_int(row.get("sequence")) or 0))
        return self._dedupe_records([self._clean_orderbook_snapshot_record(row) for row in rows], keys=("exchange", "market_type", "symbol", "timestamp_ms", "last_update_id"))

    def read_funding(self, **kwargs: Any) -> list[dict[str, Any]]:
        rows = self._read_market_dataset(self.config.funding_dataset, timestamp_fields=("timestamp_ms", "funding_time_ms", "funding_time"), **kwargs)
        rows.sort(key=lambda row: (str(row.get("symbol") or ""), self._row_time_ms(row)))
        return self._dedupe_records([self._clean_funding_record(row) for row in rows], keys=("exchange", "market_type", "symbol", "timestamp_ms", "funding_rate"))

    def read_open_interest(self, **kwargs: Any) -> list[dict[str, Any]]:
        rows = self._read_market_dataset(self.config.open_interest_dataset, timestamp_fields=("timestamp_ms", "open_interest_time_ms", "ingested_at_ms"), **kwargs)
        rows.sort(key=lambda row: (str(row.get("symbol") or ""), self._row_time_ms(row)))
        return self._dedupe_records([self._clean_open_interest_record(row) for row in rows], keys=("exchange", "market_type", "symbol", "timestamp_ms", "open_interest"))

    def read_liquidations(self, **kwargs: Any) -> list[dict[str, Any]]:
        rows = self._read_market_dataset(self.config.liquidations_dataset, timestamp_fields=("timestamp_ms", "ingested_at_ms"), **kwargs)
        rows.sort(key=lambda row: (str(row.get("symbol") or ""), self._row_time_ms(row), str(row.get("order_id") or "")))
        return self._dedupe_records([self._clean_liquidation_record(row) for row in rows], keys=("exchange", "market_type", "symbol", "order_id", "timestamp_ms", "side", "price", "quantity"))

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
        rows = self._read_dataset_records(
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
        rows.sort(key=self._row_time_ms)
        return rows

    # ------------------------------------------------------------------
    # Public load/replay API
    # ------------------------------------------------------------------

    async def load_candles_to_ingestion(self, **kwargs: Any) -> ParquetLoadResult:
        return await self._load_rows_to_ingestion(
            dataset=self.config.candles_dataset,
            read_method=self.read_candles,
            group_keys=("symbol", "timeframe"),
            method_names=("ingest_candles_batch", "ingest_candles", "apply_candles_batch"),
            payload_key="candles",
            single_method_names=("ingest_candle", "apply_candle"),
            **kwargs,
        )

    async def load_trades_to_ingestion(self, **kwargs: Any) -> ParquetLoadResult:
        return await self._load_rows_to_ingestion(
            dataset=self.config.trades_dataset,
            read_method=self.read_trades,
            group_keys=("symbol",),
            method_names=("ingest_trades_batch", "ingest_trades", "apply_trades_batch"),
            payload_key="trades",
            single_method_names=("ingest_trade", "apply_trade"),
            **kwargs,
        )

    async def load_orderbook_snapshots_to_ingestion(self, **kwargs: Any) -> ParquetLoadResult:
        return await self._load_rows_to_ingestion(
            dataset=self.config.orderbook_snapshots_dataset,
            read_method=self.read_orderbook_snapshots,
            group_keys=("symbol",),
            method_names=("ingest_orderbook_snapshots_batch", "ingest_orderbook_snapshot_batch", "apply_orderbook_snapshots_batch"),
            payload_key="orderbook_snapshots",
            single_method_names=("ingest_orderbook_snapshot", "apply_snapshot"),
            **kwargs,
        )

    async def load_funding_to_ingestion(self, **kwargs: Any) -> ParquetLoadResult:
        return await self._load_rows_to_ingestion(
            dataset=self.config.funding_dataset,
            read_method=self.read_funding,
            group_keys=("symbol",),
            method_names=("ingest_funding_batch", "ingest_funding_rates_batch", "apply_funding_batch"),
            payload_key="funding",
            single_method_names=("ingest_funding", "apply_funding"),
            **kwargs,
        )

    async def load_open_interest_to_ingestion(self, **kwargs: Any) -> ParquetLoadResult:
        return await self._load_rows_to_ingestion(
            dataset=self.config.open_interest_dataset,
            read_method=self.read_open_interest,
            group_keys=("symbol",),
            method_names=("ingest_open_interest_batch", "ingest_open_interests_batch", "apply_open_interest_batch"),
            payload_key="open_interest",
            single_method_names=("ingest_open_interest", "apply_open_interest"),
            **kwargs,
        )

    async def load_liquidations_to_ingestion(self, **kwargs: Any) -> ParquetLoadResult:
        return await self._load_rows_to_ingestion(
            dataset=self.config.liquidations_dataset,
            read_method=self.read_liquidations,
            group_keys=("symbol",),
            method_names=("ingest_liquidations_batch", "ingest_liquidation_batch", "apply_liquidations_batch"),
            payload_key="liquidations",
            single_method_names=("ingest_liquidation", "apply_liquidation"),
            **kwargs,
        )

    async def load_market_data_to_ingestion(
        self,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        symbols: Sequence[str],
        timeframes: Sequence[str] | None = None,
        start_ms: int | float | str | datetime | None = None,
        end_ms: int | float | str | datetime | None = None,
        load_candles: bool | None = None,
        load_trades: bool | None = None,
        load_orderbook_snapshots: bool | None = None,
        load_funding: bool | None = None,
        load_open_interest: bool | None = None,
        load_liquidations: bool | None = None,
        batch_size: int | None = None,
        evaluate_after_load: bool | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Hydrate MarketStateStore from Parquet before live WS/REST warmup starts."""
        should_eval = self._should_evaluate(evaluate_after_load)
        common = {
            "exchange": exchange,
            "market_type": market_type,
            "symbols": symbols,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "batch_size": batch_size,
            "evaluate_after_load": False,
            "source": source,
        }
        results: dict[str, Any] = {}

        if self._choose(load_candles, self.config.load_candles):
            result = await self.load_candles_to_ingestion(timeframes=timeframes or [], **common)
            results["candles"] = result.to_dict()

        if self._choose(load_trades, self.config.load_trades):
            result = await self.load_trades_to_ingestion(**common)
            results["trades"] = result.to_dict()

        if self._choose(load_orderbook_snapshots, self.config.load_orderbook_snapshots):
            result = await self.load_orderbook_snapshots_to_ingestion(**common)
            results["orderbook_snapshots"] = result.to_dict()

        if self._choose(load_funding, self.config.load_funding):
            result = await self.load_funding_to_ingestion(**common)
            results["funding"] = result.to_dict()

        if self._choose(load_open_interest, self.config.load_open_interest):
            result = await self.load_open_interest_to_ingestion(**common)
            results["open_interest"] = result.to_dict()

        if self._choose(load_liquidations, self.config.load_liquidations):
            result = await self.load_liquidations_to_ingestion(**common)
            results["liquidations"] = result.to_dict()

        if should_eval:
            results["market_state_evaluation"] = await self.evaluate_market_state_once(reason="parquet_market_data_loaded")

        await self._emit_loader_event("storage.parquet_loader.market_data_loaded", {"results": results})
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
        await self._emit_loader_event("storage.parquet_loader.market_state_evaluated", {"reason": reason, "result": self._json_safe_value(value)})
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

    def _read_market_dataset(
        self,
        dataset: str,
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
        timestamp_fields: Sequence[str] = ("timestamp_ms",),
        **_: Any,
    ) -> list[dict[str, Any]]:
        return self._read_dataset_records(
            dataset=dataset,
            exchange=self._normalize_exchange(exchange),
            market_type=self._normalize_market_type(market_type),
            symbols=self._normalize_list(symbols or ([symbol] if symbol else []), upper=True),
            timeframes=self._normalize_list(timeframes or ([timeframe] if timeframe else []), upper=False),
            start_ms=self._normalize_timestamp_ms(start_ms),
            end_ms=self._normalize_timestamp_ms(end_ms),
            timestamp_fields=timestamp_fields,
            columns=columns,
        )

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
        if not path.exists() or not any(path.rglob("*.parquet")):
            return []

        try:
            dataset_obj = ds.dataset(str(path), format="parquet", partitioning="hive")
            table = dataset_obj.to_table(columns=list(columns) if columns else None)
            records = table.to_pylist()
        except Exception as exc:
            self._logger.warning("Parquet dataset read failed | dataset=%s path=%s error=%s", dataset, path, exc)
            return []

        records = [self._plain_record(row) for row in records]
        return [
            row
            for row in records
            if self._row_matches_filters(
                row,
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
        ]

    def _row_matches_filters(
        self,
        row: Mapping[str, Any],
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
    ) -> bool:
        if exchange and str(row.get("exchange") or "").lower() != exchange:
            return False
        if market_type and str(row.get("market_type") or "").lower() != market_type:
            return False
        if symbols and str(row.get("symbol") or "").upper() not in {str(item).upper() for item in symbols}:
            return False
        if timeframes and str(row.get("timeframe") or "") not in {str(item) for item in timeframes}:
            return False
        if event_type and str(row.get("event_type") or "") != str(event_type):
            return False
        if analytics_type and str(row.get("analytics_type") or "") != str(analytics_type):
            return False
        if strategy_name and str(row.get("strategy_name") or "") != str(strategy_name):
            return False
        ts = self._row_time_ms(row, timestamp_fields=timestamp_fields)
        return self._time_in_range(ts, start_ms=start_ms, end_ms=end_ms)

    # ------------------------------------------------------------------
    # Ingestion bridge
    # ------------------------------------------------------------------

    async def _load_rows_to_ingestion(
        self,
        *,
        dataset: str,
        read_method: Any,
        group_keys: tuple[str, ...],
        method_names: tuple[str, ...],
        payload_key: str,
        single_method_names: tuple[str, ...],
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
        **_: Any,
    ) -> ParquetLoadResult:
        started = time.time()
        exchange_norm = self._normalize_exchange(exchange)
        market_type_norm = self._normalize_market_type(market_type)
        symbols_list = self._normalize_list(symbols or ([symbol] if symbol else []), upper=True)
        timeframes_list = self._normalize_list(timeframes or ([timeframe] if timeframe else []), upper=False)
        start = self._normalize_timestamp_ms(start_ms)
        end = self._normalize_timestamp_ms(end_ms)
        source = source or self.config.default_source
        batch_size = max(1, int(batch_size or self.config.batch_size))
        result = ParquetLoadResult(
            dataset=dataset,
            exchange=exchange_norm,
            market_type=market_type_norm,
            symbols=symbols_list,
            timeframes=timeframes_list,
            requested_start_ms=start,
            requested_end_ms=end,
            files_scanned=self.count_files(dataset),
        )

        try:
            rows = read_method(
                exchange=exchange_norm,
                market_type=market_type_norm,
                symbols=symbols_list,
                timeframes=timeframes_list,
                start_ms=start,
                end_ms=end,
            )
            result.rows_read = len(rows)
            grouped = self._group_by(rows, *group_keys) if group_keys else {(): rows}
            for group_rows in grouped.values():
                for batch in self._chunks(group_rows, batch_size):
                    payload = {
                        payload_key: batch,
                        "items": batch,
                        "exchange": exchange_norm,
                        "market_type": market_type_norm,
                        "source": source,
                        "replay_source": source,
                        "suppress_persistable_events": self.config.suppress_persistable_on_replay,
                    }
                    first = batch[0] if batch else {}
                    if "symbol" in first:
                        payload["symbol"] = first.get("symbol")
                    if "timeframe" in first:
                        payload["timeframe"] = first.get("timeframe")
                    await self._ingest_batch_or_items(
                        payload=payload,
                        rows=batch,
                        method_names=method_names,
                        single_method_names=single_method_names,
                        source=source,
                    )
                    result.rows_loaded += len(batch)
                    result.batches_loaded += 1

            if self._should_evaluate(evaluate_after_load):
                await self.evaluate_market_state_once(reason=f"parquet_{dataset}_loaded")
        except Exception as exc:
            result.errors.append(str(exc))
            self._logger.exception("Parquet load failed | dataset=%s", dataset)

        result.elapsed_ms = int((time.time() - started) * 1000)
        await self._emit_loader_event(f"storage.parquet_loader.{dataset}_loaded", result.to_dict())
        return result

    async def _ingest_batch_or_items(
        self,
        *,
        payload: dict[str, Any],
        rows: list[dict[str, Any]],
        method_names: tuple[str, ...],
        single_method_names: tuple[str, ...],
        source: str,
    ) -> None:
        if self.market_ingestion is None:
            raise RuntimeError("market_ingestion is required to hydrate MarketStateStore from Parquet")
        method = self._first_callable(self.market_ingestion, method_names)
        if method is not None:
            await self._call_ingestion_method(
                method,
                payload=payload,
                source=source,
                suppress_persistable_events=self.config.suppress_persistable_on_replay,
            )
            return
        single = self._first_callable(self.market_ingestion, single_method_names)
        if single is None:
            raise RuntimeError(f"MarketIngestionService has no compatible ingestion method for {method_names!r}")
        for row in rows:
            await self._call_ingestion_method(
                single,
                payload=row,
                source=source,
                suppress_persistable_events=self.config.suppress_persistable_on_replay,
            )

    async def _call_ingestion_method(self, method: Any, **kwargs: Any) -> Any:
        filtered = self._filter_supported_kwargs(method, {key: value for key, value in kwargs.items() if key != "payload"})
        payload = kwargs.get("payload")
        attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        attempts.append(((payload,), filtered) if payload is not None else ((), filtered))
        attempts.append(((), {**filtered, **({"payload": payload} if payload is not None and self._supports_kw(method, "payload") else {})}))
        if isinstance(payload, Mapping):
            for key in ("candles", "trades", "orderbook_snapshots", "funding", "open_interest", "liquidations", "items"):
                if key in payload:
                    attempts.append(((payload[key],), filtered))
                    break

        last_error: Exception | None = None
        for args, call_kwargs in attempts:
            try:
                value = method(*args, **self._filter_supported_kwargs(method, call_kwargs))
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
        timeframe = self._safe_str(row.get("timeframe")) or "1m"
        open_time_ms = self._normalize_timestamp_ms(row.get("open_time_ms") or row.get("open_time") or row.get("timestamp_ms") or row.get("timestamp"))
        close_time_ms = self._normalize_timestamp_ms(row.get("close_time_ms") or row.get("close_time") or row.get("end_time_ms") or row.get("end_time"))
        if close_time_ms is None and open_time_ms is not None:
            close_time_ms = open_time_ms + self._timeframe_to_ms(timeframe) - 1
        return self._with_extra(row, {
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
            "source": self.config.default_source,
            "replay_source": self.config.default_source,
            "suppress_persistable_events": self.config.suppress_persistable_on_replay,
        })

    def _clean_trade_record(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return self._with_extra(row, {
            "exchange": self._safe_str(row.get("exchange")) or self.config.default_exchange,
            "symbol": self._safe_str(row.get("symbol")),
            "market_type": self._safe_str(row.get("market_type")) or self.config.default_market_type,
            "trade_id": self._safe_str(row.get("trade_id") or row.get("id")),
            "timestamp_ms": self._normalize_timestamp_ms(row.get("timestamp_ms") or row.get("trade_time_ms") or row.get("timestamp")),
            "price": self._safe_float(row.get("price")),
            "quantity": self._safe_float(row.get("quantity") or row.get("qty") or row.get("size")),
            "side": self._safe_str(row.get("side")),
            "aggressor_side": self._safe_str(row.get("aggressor_side") or row.get("taker_side") or row.get("side")),
            "source": self.config.default_source,
            "replay_source": self.config.default_source,
            "suppress_persistable_events": self.config.suppress_persistable_on_replay,
        })

    def _clean_orderbook_snapshot_record(self, row: Mapping[str, Any]) -> dict[str, Any]:
        bids = self._parse_levels(row.get("bids") or row.get("bids_json"))
        asks = self._parse_levels(row.get("asks") or row.get("asks_json"))
        return self._with_extra(row, {
            "exchange": self._safe_str(row.get("exchange")) or self.config.default_exchange,
            "symbol": self._safe_str(row.get("symbol")),
            "market_type": self._safe_str(row.get("market_type")) or self.config.default_market_type,
            "timestamp_ms": self._normalize_timestamp_ms(row.get("timestamp_ms") or row.get("ingested_at_ms")),
            "last_update_id": self._safe_int(row.get("last_update_id") or row.get("sequence") or row.get("update_id")),
            "bids": bids,
            "asks": asks,
            "source": self.config.default_source,
            "replay_source": self.config.default_source,
            "suppress_persistable_events": self.config.suppress_persistable_on_replay,
        })

    def _clean_funding_record(self, row: Mapping[str, Any]) -> dict[str, Any]:
        timestamp_ms = self._normalize_timestamp_ms(row.get("timestamp_ms") or row.get("funding_time_ms") or row.get("funding_time") or row.get("timestamp"))
        return self._with_extra(row, {
            "exchange": self._safe_str(row.get("exchange")) or self.config.default_exchange,
            "symbol": self._safe_str(row.get("symbol")),
            "market_type": self._safe_str(row.get("market_type")) or self.config.default_market_type,
            "timestamp_ms": timestamp_ms,
            "funding_time_ms": timestamp_ms,
            "funding_rate": self._safe_float(row.get("funding_rate") or row.get("rate")),
            "next_funding_time_ms": self._normalize_timestamp_ms(row.get("next_funding_time_ms") or row.get("next_funding_time")),
            "mark_price": self._safe_float(row.get("mark_price")),
            "index_price": self._safe_float(row.get("index_price")),
            "source": self.config.default_source,
            "replay_source": self.config.default_source,
            "suppress_persistable_events": self.config.suppress_persistable_on_replay,
        })

    def _clean_open_interest_record(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return self._with_extra(row, {
            "exchange": self._safe_str(row.get("exchange")) or self.config.default_exchange,
            "symbol": self._safe_str(row.get("symbol")),
            "market_type": self._safe_str(row.get("market_type")) or self.config.default_market_type,
            "timestamp_ms": self._normalize_timestamp_ms(row.get("timestamp_ms") or row.get("open_interest_time_ms") or row.get("timestamp")),
            "open_interest": self._safe_float(row.get("open_interest") or row.get("oi")),
            "open_interest_value": self._safe_float(row.get("open_interest_value")),
            "mark_price": self._safe_float(row.get("mark_price")),
            "index_price": self._safe_float(row.get("index_price")),
            "source": self.config.default_source,
            "replay_source": self.config.default_source,
            "suppress_persistable_events": self.config.suppress_persistable_on_replay,
        })

    def _clean_liquidation_record(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return self._with_extra(row, {
            "exchange": self._safe_str(row.get("exchange")) or self.config.default_exchange,
            "symbol": self._safe_str(row.get("symbol")),
            "market_type": self._safe_str(row.get("market_type")) or self.config.default_market_type,
            "timestamp_ms": self._normalize_timestamp_ms(row.get("timestamp_ms") or row.get("event_time") or row.get("timestamp")),
            "side": self._safe_str(row.get("side")),
            "price": self._safe_float(row.get("price")),
            "quantity": self._safe_float(row.get("quantity") or row.get("qty") or row.get("size")),
            "notional": self._safe_float(row.get("notional")),
            "order_id": self._safe_str(row.get("order_id")),
            "source": self.config.default_source,
            "replay_source": self.config.default_source,
            "suppress_persistable_events": self.config.suppress_persistable_on_replay,
        })

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

    @staticmethod
    def _choose(value: bool | None, default: bool) -> bool:
        return default if value is None else bool(value)

    def _should_evaluate(self, value: bool | None) -> bool:
        return self.config.evaluate_after_load if value is None else bool(value)

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

    def _row_time_ms(self, row: Mapping[str, Any], *, timestamp_fields: Sequence[str] | None = None) -> int:
        for key in timestamp_fields or ("open_time_ms", "timestamp_ms", "close_time_ms", "funding_time_ms", "ingested_at_ms"):
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

    def _parse_levels(self, value: Any) -> list[list[float]]:
        if value is None:
            return []
        if isinstance(value, str):
            with contextlib.suppress(Exception):
                value = json.loads(value)
        result: list[list[float]] = []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for level in value:
                if isinstance(level, Mapping):
                    price = self._safe_float(level.get("price"))
                    qty = self._safe_float(level.get("quantity") or level.get("qty") or level.get("size"))
                elif isinstance(level, Sequence) and not isinstance(level, (str, bytes, bytearray)) and len(level) >= 2:
                    price = self._safe_float(level[0])
                    qty = self._safe_float(level[1])
                else:
                    continue
                if price is not None and qty is not None:
                    result.append([price, qty])
        return result

    def _with_extra(self, row: Mapping[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        for key, value in row.items():
            record.setdefault(str(key), self._json_safe_value(value))
        return record

    def _dedupe_records(self, rows: list[dict[str, Any]], *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
        seen: set[tuple[Any, ...]] = set()
        result: list[dict[str, Any]] = []
        for row in rows:
            if not row.get("symbol"):
                continue
            key = tuple(row.get(item) for item in keys)
            if key in seen:
                continue
            seen.add(key)
            result.append(row)
        return result

    def _timeframe_to_ms(self, timeframe: str) -> int:
        value = str(timeframe or "1m").strip().lower()
        aliases = {"min1": "1m", "min5": "5m", "min15": "15m", "min30": "30m", "hour1": "1h", "day1": "1d"}
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
        if value is None or isinstance(value, (str, int, float, bool)):
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

    async def _emit_loader_event(self, topic: str, payload: Mapping[str, Any]) -> None:
        if not self.config.emit_loader_events or self.event_bus is None:
            return
        emit = getattr(self.event_bus, "emit", None)
        if not callable(emit):
            return
        with contextlib.suppress(Exception):
            value = emit(topic, {"service": "parquet_loader", **dict(payload)}, source="storage.parquet_loader")
            if inspect.isawaitable(value):
                await value


__all__ = ["ParquetLoaderConfig", "ParquetLoadResult", "ParquetMarketLoader"]
