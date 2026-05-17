from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from core.config import Config
from core.event_bus import Event, EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler


@dataclass(slots=True)
class ParquetStorageConfig:
    """
    Local Parquet storage config.

    Storage layer responsibility:
    - listen to EventBus topics;
    - buffer normalized/persistable market events;
    - write batches to partitioned Parquet datasets;
    - never call exchange/data/analytics/strategy/risk/execution directly.
    """

    root_dir: str = "data/parquet"

    flush_interval_seconds: float = 10.0
    max_records_per_dataset: int = 10_000
    max_total_buffer_records: int = 100_000

    compression: str = "zstd"
    row_group_size: int = 10_000

    enabled: bool = True

    store_trades: bool = True
    store_closed_candles: bool = True
    store_orderbook_snapshots: bool = True
    store_funding: bool = True
    store_open_interest: bool = True
    store_liquidations: bool = True
    store_analytics: bool = True

    emit_storage_events: bool = True

    @classmethod
    def from_core_config(cls, config: Config) -> "ParquetStorageConfig":
        storage = getattr(config, "storage", None)

        root_dir = getattr(storage, "parquet_dir", None)
        if root_dir is None:
            root_dir = getattr(storage, "data_dir", None)
        if root_dir is None:
            root_dir = cls.root_dir

        return cls(root_dir=str(root_dir))


@dataclass(slots=True)
class ParquetStorageMetrics:
    events_received: int = 0
    events_buffered: int = 0
    events_dropped: int = 0
    flush_runs: int = 0
    flush_errors: int = 0
    records_written: int = 0
    files_written: int = 0
    last_flush_at: float = 0.0
    last_error: str | None = None


class ParquetStorage:
    """
    EventBus-driven Parquet storage.

    Correct runtime pipeline:

        Exchange adapters
            -> EventBus: market.*
            -> Data caches
            -> EventBus: market.*.updated / market.candle.closed / persistable snapshots
            -> ParquetStorage
            -> partitioned Parquet files

    This class intentionally does NOT:
    - call exchange clients;
    - call cache classes;
    - call analytics/strategy/risk/execution;
    - store every raw websocket message;
    - store every unfinished candle update;
    - store every orderbook delta by default.
    """

    SERVICE = "parquet_storage"

    DATASET_CANDLES = "candles"
    DATASET_TRADES = "trades"
    DATASET_ORDERBOOK_SNAPSHOTS = "orderbook_snapshots"
    DATASET_FUNDING = "funding"
    DATASET_OPEN_INTEREST = "open_interest"
    DATASET_LIQUIDATIONS = "liquidations"
    DATASET_ANALYTICS = "analytics_events"

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        storage_config: ParquetStorageConfig | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._storage_config = storage_config or ParquetStorageConfig.from_core_config(config)

        self._logger = get_logger(
            __name__,
            service=self.SERVICE,
            event_type="storage_parquet",
        )

        self._root_dir = Path(self._storage_config.root_dir)

        self._buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._buffer_lock = asyncio.Lock()
        self._flush_lock = asyncio.Lock()

        self._registered = False
        self._started = False
        self._flush_job_id: str | None = None

        self._metrics = ParquetStorageMetrics()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Register EventBus subscriptions.

        ParquetStorage listens only to persistable events.
        It does not receive data through direct method calls from cache classes.
        """
        if self._registered:
            self._logger.warning("ParquetStorage already registered")
            return

        if not self._storage_config.enabled:
            self._logger.warning("ParquetStorage disabled, subscriptions skipped")
            self._registered = True
            return

        if self._storage_config.store_closed_candles:
            self._event_bus.subscribe("market.candle.closed", self._on_candle_closed)

        if self._storage_config.store_trades:
            self._event_bus.subscribe("market.trade", self._on_trade)

        if self._storage_config.store_orderbook_snapshots:
            self._event_bus.subscribe(
                "market.orderbook.snapshot.persistable",
                self._on_orderbook_snapshot,
            )

        if self._storage_config.store_funding:
            self._event_bus.subscribe("market.funding.updated", self._on_funding_updated)

        if self._storage_config.store_open_interest:
            self._event_bus.subscribe(
                "market.open_interest.updated",
                self._on_open_interest_updated,
            )

        if self._storage_config.store_liquidations:
            self._event_bus.subscribe("market.liquidation", self._on_liquidation)

        if self._storage_config.store_analytics:
            self._event_bus.subscribe("analytics.*", self._on_analytics_event)

        self._registered = True

        self._logger.info(
            "ParquetStorage registered | root_dir=%s",
            self._root_dir,
        )

    async def start(self) -> None:
        if self._started:
            self._logger.warning("ParquetStorage already started")
            return

        if not self._storage_config.enabled:
            self._logger.warning("ParquetStorage start skipped: disabled")
            self._started = True
            return

        self._root_dir.mkdir(parents=True, exist_ok=True)

        if not self._registered:
            self.register()

        self._register_flush_job()

        self._started = True

        self._logger.info(
            "ParquetStorage started | root_dir=%s flush_interval=%s",
            self._root_dir,
            self._storage_config.flush_interval_seconds,
        )

        await self._emit_storage_event(
            "storage.parquet.started",
            {
                "root_dir": str(self._root_dir),
                "flush_interval_seconds": self._storage_config.flush_interval_seconds,
            },
        )

    async def stop(self) -> None:
        if not self._started:
            self._logger.warning("ParquetStorage already stopped")
            return

        self._started = False

        self._remove_flush_job()

        await self.flush()

        self._logger.info("ParquetStorage stopped")

        await self._emit_storage_event(
            "storage.parquet.stopped",
            {
                "stats": self.stats(),
            },
        )

    def _register_flush_job(self) -> None:
        if self._scheduler is None:
            self._logger.warning("Scheduler not provided, automatic Parquet flush disabled")
            return

        existing = self._scheduler.get_job_by_name("parquet-storage-flush")
        if existing is not None:
            self._flush_job_id = existing.job_id
            self._logger.warning(
                "Parquet flush job already exists | job_id=%s",
                existing.job_id,
            )
            return

        self._flush_job_id = self._scheduler.add_interval_job(
            name="parquet-storage-flush",
            func=self.flush,
            interval=self._storage_config.flush_interval_seconds,
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=max(self._storage_config.flush_interval_seconds, 5.0),
            allow_overlap=False,
            enabled=True,
        )

        self._logger.info(
            "Parquet flush job registered | job_id=%s",
            self._flush_job_id,
        )

    def _remove_flush_job(self) -> None:
        if self._scheduler is None or self._flush_job_id is None:
            return

        with contextlib.suppress(Exception):
            self._scheduler.remove_job(self._flush_job_id)

        self._logger.info(
            "Parquet flush job removed | job_id=%s",
            self._flush_job_id,
        )

        self._flush_job_id = None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_candle_closed(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="market.candle.closed")

        record = self._normalize_candle(payload, topic=topic)
        if record is None:
            await self._drop_event(topic, payload, reason="invalid_candle")
            return

        await self._buffer_record(self.DATASET_CANDLES, record)

    async def _on_trade(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="market.trade")

        record = self._normalize_trade(payload, topic=topic)
        if record is None:
            await self._drop_event(topic, payload, reason="invalid_trade")
            return

        await self._buffer_record(self.DATASET_TRADES, record)

    async def _on_orderbook_snapshot(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="market.orderbook.snapshot.persistable")

        record = self._normalize_orderbook_snapshot(payload, topic=topic)
        if record is None:
            await self._drop_event(topic, payload, reason="invalid_orderbook_snapshot")
            return

        await self._buffer_record(self.DATASET_ORDERBOOK_SNAPSHOTS, record)

    async def _on_funding_updated(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="market.funding.updated")

        record = self._normalize_funding(payload, topic=topic)
        if record is None:
            await self._drop_event(topic, payload, reason="invalid_funding")
            return

        await self._buffer_record(self.DATASET_FUNDING, record)

    async def _on_open_interest_updated(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="market.open_interest.updated")

        record = self._normalize_open_interest(payload, topic=topic)
        if record is None:
            await self._drop_event(topic, payload, reason="invalid_open_interest")
            return

        await self._buffer_record(self.DATASET_OPEN_INTEREST, record)

    async def _on_liquidation(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="market.liquidation")

        record = self._normalize_liquidation(payload, topic=topic)
        if record is None:
            await self._drop_event(topic, payload, reason="invalid_liquidation")
            return

        await self._buffer_record(self.DATASET_LIQUIDATIONS, record)

    async def _on_analytics_event(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="analytics.unknown")

        record = self._normalize_analytics_event(payload, topic=topic)
        if record is None:
            await self._drop_event(topic, payload, reason="invalid_analytics_event")
            return

        await self._buffer_record(self.DATASET_ANALYTICS, record)

    # ------------------------------------------------------------------
    # Buffering
    # ------------------------------------------------------------------

    async def _buffer_record(self, dataset: str, record: dict[str, Any]) -> None:
        if not self._storage_config.enabled:
            return

        async with self._buffer_lock:
            self._metrics.events_received += 1

            total_buffered = self._total_buffered_records_unlocked()
            if total_buffered >= self._storage_config.max_total_buffer_records:
                self._metrics.events_dropped += 1
                self._metrics.last_error = "max_total_buffer_records_reached"

                self._logger.error(
                    "Parquet buffer overflow, dropping record | dataset=%s total_buffered=%s",
                    dataset,
                    total_buffered,
                )
                return

            self._buffers[dataset].append(record)
            self._metrics.events_buffered += 1

            should_flush = (
                len(self._buffers[dataset])
                >= self._storage_config.max_records_per_dataset
            )

        if should_flush:
            await self.flush_dataset(dataset)

    def _total_buffered_records_unlocked(self) -> int:
        return sum(len(records) for records in self._buffers.values())

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    async def flush(self) -> dict[str, Any]:
        if not self._storage_config.enabled:
            return {
                "enabled": False,
                "records_written": 0,
                "files_written": 0,
            }

        async with self._flush_lock:
            async with self._buffer_lock:
                batches = {
                    dataset: records[:]
                    for dataset, records in self._buffers.items()
                    if records
                }
                for dataset in batches:
                    self._buffers[dataset].clear()

            return await self._write_batches(batches)

    async def flush_dataset(self, dataset: str) -> dict[str, Any]:
        if not self._storage_config.enabled:
            return {
                "enabled": False,
                "dataset": dataset,
                "records_written": 0,
                "files_written": 0,
            }

        async with self._flush_lock:
            async with self._buffer_lock:
                records = self._buffers.get(dataset, [])
                if not records:
                    return {
                        "dataset": dataset,
                        "records_written": 0,
                        "files_written": 0,
                    }

                batch = {dataset: records[:]}
                self._buffers[dataset].clear()

            return await self._write_batches(batch)

    async def _write_batches(self, batches: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        if not batches:
            return {
                "records_written": 0,
                "files_written": 0,
                "datasets": {},
            }

        started_at = time.time()

        try:
            result = await asyncio.to_thread(self._write_batches_sync, batches)

            self._metrics.flush_runs += 1
            self._metrics.records_written += result["records_written"]
            self._metrics.files_written += result["files_written"]
            self._metrics.last_flush_at = time.time()
            self._metrics.last_error = None

            self._logger.info(
                "Parquet flush completed | records=%s files=%s elapsed_ms=%s",
                result["records_written"],
                result["files_written"],
                int((time.time() - started_at) * 1000),
            )

            await self._emit_storage_event(
                "storage.parquet.flushed",
                result,
            )

            return result

        except Exception as exc:
            self._metrics.flush_errors += 1
            self._metrics.last_error = str(exc)

            async with self._buffer_lock:
                for dataset, records in batches.items():
                    self._buffers[dataset] = records + self._buffers[dataset]

            self._logger.exception("Parquet flush failed")

            await self._emit_storage_event(
                "storage.parquet.error",
                {
                    "error": str(exc),
                    "datasets": list(batches.keys()),
                },
                priority=EventPriority.HIGH,
            )

            return {
                "records_written": 0,
                "files_written": 0,
                "error": str(exc),
                "datasets": list(batches.keys()),
            }

    def _write_batches_sync(self, batches: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "records_written": 0,
            "files_written": 0,
            "datasets": {},
        }

        for dataset, records in batches.items():
            grouped = self._group_records_for_partitioning(dataset, records)

            dataset_result = {
                "records_written": 0,
                "files_written": 0,
                "partitions": 0,
            }

            for partition_key, partition_records in grouped.items():
                partition_dir = self._build_partition_dir(dataset, partition_key)
                partition_dir.mkdir(parents=True, exist_ok=True)

                file_path = partition_dir / self._build_file_name(dataset)

                self._write_parquet_file(file_path, partition_records)

                written = len(partition_records)

                dataset_result["records_written"] += written
                dataset_result["files_written"] += 1
                dataset_result["partitions"] += 1

                result["records_written"] += written
                result["files_written"] += 1

            result["datasets"][dataset] = dataset_result

        return result

    def _write_parquet_file(self, file_path: Path, records: list[dict[str, Any]]) -> None:
        if not records:
            return

        normalized = [self._json_safe_record(record) for record in records]

        table = pa.Table.from_pylist(normalized)

        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")

        pq.write_table(
            table,
            tmp_path,
            compression=self._storage_config.compression,
            row_group_size=self._storage_config.row_group_size,
        )

        os.replace(tmp_path, file_path)

    # ------------------------------------------------------------------
    # Normalizers
    # ------------------------------------------------------------------

    def _normalize_candle(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        exchange = self._required_str(payload, "exchange")
        symbol = self._required_str(payload, "symbol")
        timeframe = self._required_str(payload, "timeframe")

        open_time_ms = self._first_int(
            payload,
            "open_time_ms",
            "open_time",
            "timestamp_ms",
            "timestamp",
        )

        if exchange is None or symbol is None or timeframe is None or open_time_ms is None:
            return None

        is_closed = self._safe_bool(payload.get("is_closed"))
        if not is_closed:
            return None

        close_time_ms = self._first_int(payload, "close_time_ms", "close_time")
        if close_time_ms is None:
            close_time_ms = self._infer_close_time_ms(open_time_ms, timeframe)

        return {
            "topic": topic,
            "dataset": self.DATASET_CANDLES,
            "exchange": exchange,
            "symbol": symbol,
            "market_type": self._market_type(payload),
            "timeframe": timeframe,
            "open_time_ms": open_time_ms,
            "close_time_ms": close_time_ms,
            "open": self._safe_float(payload.get("open")),
            "high": self._safe_float(payload.get("high")),
            "low": self._safe_float(payload.get("low")),
            "close": self._safe_float(payload.get("close")),
            "volume": self._safe_float(payload.get("volume")),
            "quote_volume": self._safe_float(payload.get("quote_volume")),
            "trades_count": self._safe_int(payload.get("trades_count")),
            "is_closed": True,
            "timestamp_ms": self._event_timestamp(payload, fallback=open_time_ms),
            "received_at_ms": self._received_at(payload),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_trade(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        exchange = self._required_str(payload, "exchange")
        symbol = self._required_str(payload, "symbol")

        timestamp_ms = self._first_int(
            payload,
            "timestamp_ms",
            "trade_time",
            "event_time",
            "time",
            "timestamp",
        )

        price = self._safe_float(payload.get("price"))
        quantity = self._first_float(payload, "quantity", "qty", "size", "amount")

        if exchange is None or symbol is None or timestamp_ms is None:
            return None
        if price is None or quantity is None:
            return None

        side = self._normalize_side(payload.get("side"))
        aggressor_side = self._normalize_side(
            payload.get("aggressor_side") or payload.get("taker_side") or side
        )

        return {
            "topic": topic,
            "dataset": self.DATASET_TRADES,
            "exchange": exchange,
            "symbol": symbol,
            "market_type": self._market_type(payload),
            "trade_id": self._safe_str(payload.get("trade_id") or payload.get("id")),
            "timestamp_ms": timestamp_ms,
            "received_at_ms": self._received_at(payload),
            "price": price,
            "quantity": quantity,
            "quote_quantity": self._first_float(payload, "quote_quantity", "quote_qty"),
            "side": side,
            "aggressor_side": aggressor_side,
            "buyer_is_maker": self._safe_optional_bool(payload.get("buyer_is_maker")),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_orderbook_snapshot(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        exchange = self._required_str(payload, "exchange")
        symbol = self._required_str(payload, "symbol")

        if exchange is None or symbol is None:
            return None

        bids = payload.get("bids") or []
        asks = payload.get("asks") or []

        if not isinstance(bids, list) or not isinstance(asks, list):
            return None

        timestamp_ms = self._event_timestamp(payload)

        best_bid = self._first_level_price(bids)
        best_ask = self._first_level_price(asks)

        return {
            "topic": topic,
            "dataset": self.DATASET_ORDERBOOK_SNAPSHOTS,
            "exchange": exchange,
            "symbol": symbol,
            "market_type": self._market_type(payload),
            "timestamp_ms": timestamp_ms,
            "received_at_ms": self._received_at(payload),
            "sequence": self._first_int(payload, "sequence", "update_id", "last_update_id"),
            "depth": max(len(bids), len(asks)),
            "bids_json": self._json_dumps(bids),
            "asks_json": self._json_dumps(asks),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": self._calc_spread(best_bid, best_ask),
            "mid_price": self._calc_mid_price(best_bid, best_ask),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_funding(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        exchange = self._required_str(payload, "exchange")
        symbol = self._required_str(payload, "symbol")

        timestamp_ms = self._first_int(
            payload,
            "timestamp_ms",
            "funding_time",
            "time",
            "timestamp",
        )

        funding_rate = self._safe_float(payload.get("funding_rate"))

        if exchange is None or symbol is None or timestamp_ms is None:
            return None
        if funding_rate is None:
            return None

        return {
            "topic": topic,
            "dataset": self.DATASET_FUNDING,
            "exchange": exchange,
            "symbol": symbol,
            "market_type": self._market_type(payload),
            "timestamp_ms": timestamp_ms,
            "received_at_ms": self._received_at(payload),
            "funding_rate": funding_rate,
            "predicted_rate": self._safe_float(payload.get("predicted_rate")),
            "next_funding_time_ms": self._first_int(
                payload,
                "next_funding_time_ms",
                "next_funding_time",
            ),
            "mark_price": self._safe_float(payload.get("mark_price")),
            "index_price": self._safe_float(payload.get("index_price")),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_open_interest(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        exchange = self._required_str(payload, "exchange")
        symbol = self._required_str(payload, "symbol")

        timestamp_ms = self._first_int(
            payload,
            "timestamp_ms",
            "open_interest_time",
            "time",
            "timestamp",
        )

        open_interest = self._safe_float(payload.get("open_interest"))

        if exchange is None or symbol is None or timestamp_ms is None:
            return None
        if open_interest is None:
            return None

        return {
            "topic": topic,
            "dataset": self.DATASET_OPEN_INTEREST,
            "exchange": exchange,
            "symbol": symbol,
            "market_type": self._market_type(payload),
            "timestamp_ms": timestamp_ms,
            "received_at_ms": self._received_at(payload),
            "open_interest": open_interest,
            "open_interest_value": self._safe_float(payload.get("open_interest_value")),
            "mark_price": self._safe_float(payload.get("mark_price")),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_liquidation(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        exchange = self._required_str(payload, "exchange")
        symbol = self._required_str(payload, "symbol")

        timestamp_ms = self._event_timestamp(payload)

        price = self._safe_float(payload.get("price"))
        quantity = self._first_float(payload, "quantity", "qty", "size", "amount")

        if exchange is None or symbol is None or timestamp_ms is None:
            return None

        return {
            "topic": topic,
            "dataset": self.DATASET_LIQUIDATIONS,
            "exchange": exchange,
            "symbol": symbol,
            "market_type": self._market_type(payload),
            "timestamp_ms": timestamp_ms,
            "received_at_ms": self._received_at(payload),
            "side": self._normalize_side(payload.get("side")),
            "price": price,
            "quantity": quantity,
            "notional": self._safe_float(payload.get("notional")),
            "order_id": self._safe_str(payload.get("order_id")),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_analytics_event(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        timestamp_ms = self._event_timestamp(payload)

        return {
            "topic": topic,
            "dataset": self.DATASET_ANALYTICS,
            "analytics_type": topic.removeprefix("analytics."),
            "exchange": self._safe_str(payload.get("exchange")),
            "symbol": self._safe_str(payload.get("symbol")),
            "market_type": self._market_type(payload),
            "timeframe": self._safe_str(payload.get("timeframe")),
            "timestamp_ms": timestamp_ms or self._now_ms(),
            "received_at_ms": self._received_at(payload),
            "direction": self._safe_str(payload.get("direction")),
            "score": self._safe_float(payload.get("score")),
            "confidence": self._safe_float(payload.get("confidence")),
            "features_json": self._json_dumps(payload.get("features", {})),
            "metadata_json": self._json_dumps(payload.get("metadata", {})),
            "payload_json": self._json_dumps(payload),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    # ------------------------------------------------------------------
    # Partitioning
    # ------------------------------------------------------------------

    def _group_records_for_partitioning(
        self,
        dataset: str,
        records: list[dict[str, Any]],
    ) -> dict[tuple[tuple[str, str], ...], list[dict[str, Any]]]:
        grouped: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = defaultdict(list)

        for record in records:
            partition = self._partition_key(dataset, record)
            grouped[partition].append(record)

        return grouped

    def _partition_key(
        self,
        dataset: str,
        record: Mapping[str, Any],
    ) -> tuple[tuple[str, str], ...]:
        date = self._partition_date(record.get("timestamp_ms") or record.get("ingested_at_ms"))

        exchange = self._partition_value(record.get("exchange"), default="unknown")
        symbol = self._partition_value(record.get("symbol"), default="unknown")
        market_type = self._partition_value(record.get("market_type"), default="unknown")

        if dataset == self.DATASET_CANDLES:
            timeframe = self._partition_value(record.get("timeframe"), default="unknown")
            return (
                ("exchange", exchange),
                ("symbol", symbol),
                ("market_type", market_type),
                ("timeframe", timeframe),
                ("date", date),
            )

        if dataset == self.DATASET_ANALYTICS:
            analytics_type = self._partition_value(record.get("analytics_type"), default="unknown")
            return (
                ("analytics_type", analytics_type),
                ("exchange", exchange),
                ("symbol", symbol),
                ("date", date),
            )

        if dataset == self.DATASET_ORDERBOOK_SNAPSHOTS:
            depth = self._partition_value(record.get("depth"), default="unknown")
            return (
                ("exchange", exchange),
                ("symbol", symbol),
                ("market_type", market_type),
                ("depth", depth),
                ("date", date),
            )

        return (
            ("exchange", exchange),
            ("symbol", symbol),
            ("market_type", market_type),
            ("date", date),
        )

    def _build_partition_dir(
        self,
        dataset: str,
        partition_key: tuple[tuple[str, str], ...],
    ) -> Path:
        path = self._root_dir / dataset
        for key, value in partition_key:
            path = path / f"{key}={value}"
        return path

    def _build_file_name(self, dataset: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        unique = uuid.uuid4().hex
        return f"{dataset}-{timestamp}-{unique}.parquet"

    # ------------------------------------------------------------------
    # Public inspection API
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        buffered_by_dataset = {
            dataset: len(records)
            for dataset, records in self._buffers.items()
        }

        return {
            "enabled": self._storage_config.enabled,
            "started": self._started,
            "registered": self._registered,
            "root_dir": str(self._root_dir),
            "buffered_total": sum(buffered_by_dataset.values()),
            "buffered_by_dataset": buffered_by_dataset,
            "events_received": self._metrics.events_received,
            "events_buffered": self._metrics.events_buffered,
            "events_dropped": self._metrics.events_dropped,
            "flush_runs": self._metrics.flush_runs,
            "flush_errors": self._metrics.flush_errors,
            "records_written": self._metrics.records_written,
            "files_written": self._metrics.files_written,
            "last_flush_at": self._metrics.last_flush_at,
            "last_error": self._metrics.last_error,
        }

    async def force_flush(self) -> dict[str, Any]:
        return await self.flush()

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    def _extract_payload(self, event: Event | Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(event, Event):
            payload = event.payload
            return payload if isinstance(payload, Mapping) else {}

        if isinstance(event, Mapping):
            payload = event.get("payload")
            if isinstance(payload, Mapping):
                return payload
            return event

        return {}

    def _extract_topic(self, event: Event | Mapping[str, Any], *, default: str) -> str:
        if isinstance(event, Event):
            return event.topic

        if isinstance(event, Mapping):
            topic = event.get("topic")
            if isinstance(topic, str) and topic:
                return topic

        return default

    async def _drop_event(
        self,
        topic: str,
        payload: Mapping[str, Any],
        *,
        reason: str,
    ) -> None:
        self._metrics.events_received += 1
        self._metrics.events_dropped += 1
        self._metrics.last_error = reason

        self._logger.warning(
            "Parquet event dropped | topic=%s reason=%s exchange=%s symbol=%s",
            topic,
            reason,
            payload.get("exchange"),
            payload.get("symbol"),
        )

        await self._emit_storage_event(
            "storage.parquet.event_dropped",
            {
                "topic": topic,
                "reason": reason,
                "exchange": payload.get("exchange"),
                "symbol": payload.get("symbol"),
            },
            priority=EventPriority.LOW,
        )

    async def _emit_storage_event(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.LOW,
    ) -> None:
        if not self._storage_config.emit_storage_events:
            return

        with contextlib.suppress(Exception):
            await self._event_bus.emit(
                topic,
                payload={
                    "service": self.SERVICE,
                    **payload,
                },
                priority=priority,
            )

    # ------------------------------------------------------------------
    # Value helpers
    # ------------------------------------------------------------------

    def _market_type(self, payload: Mapping[str, Any]) -> str:
        value = (
            payload.get("market_type")
            or payload.get("category")
            or payload.get("inst_type")
            or "perpetual"
        )
        return str(value).lower()

    def _event_timestamp(
        self,
        payload: Mapping[str, Any],
        *,
        fallback: int | None = None,
    ) -> int | None:
        return self._first_int(
            payload,
            "timestamp_ms",
            "timestamp",
            "event_time",
            "trade_time",
            "snapshot_time",
            "open_time_ms",
            fallback=fallback,
        )

    def _received_at(self, payload: Mapping[str, Any]) -> int:
        return (
            self._first_int(payload, "received_at_ms", "received_at")
            or self._now_ms()
        )

    def _required_str(self, payload: Mapping[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        value_str = str(value).strip()
        return value_str if value_str else None

    def _safe_str(self, value: Any) -> str | None:
        if value is None:
            return None
        value_str = str(value).strip()
        return value_str if value_str else None

    def _safe_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "closed"}
        return bool(value)

    def _safe_optional_bool(self, value: Any) -> bool | None:
        if value is None:
            return None
        return self._safe_bool(value)

    def _first_int(
        self,
        payload: Mapping[str, Any],
        *keys: str,
        fallback: int | None = None,
    ) -> int | None:
        for key in keys:
            value = self._safe_int(payload.get(key))
            if value is not None:
                return value
        return fallback

    def _first_float(self, payload: Mapping[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = self._safe_float(payload.get(key))
            if value is not None:
                return value
        return None

    def _normalize_side(self, value: Any) -> str:
        if value is None:
            return "unknown"

        side = str(value).strip().lower()

        if side in {"buy", "bid", "long"}:
            return "buy"
        if side in {"sell", "ask", "short"}:
            return "sell"

        return "unknown"

    def _first_level_price(self, levels: Any) -> float | None:
        if not isinstance(levels, list) or not levels:
            return None

        first = levels[0]

        if isinstance(first, Mapping):
            return self._safe_float(first.get("price"))

        if isinstance(first, (list, tuple)) and first:
            return self._safe_float(first[0])

        return None

    def _calc_spread(self, best_bid: float | None, best_ask: float | None) -> float | None:
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid

    def _calc_mid_price(self, best_bid: float | None, best_ask: float | None) -> float | None:
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2.0

    def _infer_close_time_ms(self, open_time_ms: int, timeframe: str) -> int:
        duration_ms = self._timeframe_to_ms(timeframe)
        return open_time_ms + duration_ms - 1

    def _timeframe_to_ms(self, timeframe: str) -> int:
        value = timeframe.strip().lower()

        aliases = {
            "min1": "1m",
            "min5": "5m",
            "min15": "15m",
            "min30": "30m",
            "hour1": "1h",
            "day1": "1d",
        }

        value = aliases.get(value, value)

        unit = value[-1]
        amount_raw = value[:-1]

        try:
            amount = int(amount_raw)
        except ValueError:
            return 60_000

        if unit == "m":
            return amount * 60_000
        if unit == "h":
            return amount * 60 * 60_000
        if unit == "d":
            return amount * 24 * 60 * 60_000
        if unit == "w":
            return amount * 7 * 24 * 60 * 60_000

        return 60_000

    def _partition_date(self, timestamp_ms: Any) -> str:
        timestamp = self._safe_int(timestamp_ms) or self._now_ms()
        dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")

    def _partition_value(self, value: Any, *, default: str) -> str:
        if value is None:
            return default

        text = str(value).strip()
        if not text:
            return default

        return (
            text.replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
            .replace("=", "_")
        )

    def _json_dumps(self, value: Any) -> str:
        return json.dumps(
            self._json_safe_value(value),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _json_safe_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(key): self._json_safe_value(value)
            for key, value in record.items()
        }

    def _json_safe_value(self, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Mapping):
            return {
                str(key): self._json_safe_value(inner_value)
                for key, inner_value in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [self._json_safe_value(item) for item in value]

        return str(value)

    def _now_ms(self) -> int:
        return int(time.time() * 1000)