from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
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
    store_strategy_events: bool = True
    store_signal_events: bool = True
    store_risk_events: bool = True
    store_execution_events: bool = True
    store_position_events: bool = True

    emit_storage_events: bool = True

    @classmethod
    def from_core_config(cls, config: Config) -> "ParquetStorageConfig":
        storage = getattr(config, "storage", None)

        root_dir = getattr(storage, "parquet_dir", None)
        if root_dir is None:
            root_dir = getattr(storage, "data_dir", None)
        if root_dir is None:
            app = getattr(config, "app", None)
            data_dir = getattr(app, "data_dir", None)
            root_dir = str(Path(data_dir) / "parquet") if data_dir is not None else cls.root_dir

        enabled = bool(getattr(storage, "parquet_enabled", cls.enabled)) if storage is not None else cls.enabled
        flush_interval = float(getattr(storage, "flush_interval_seconds", cls.flush_interval_seconds)) if storage is not None else cls.flush_interval_seconds
        batch_size = int(getattr(storage, "batch_size", cls.max_records_per_dataset)) if storage is not None else cls.max_records_per_dataset

        return cls(
            root_dir=str(root_dir),
            enabled=enabled,
            flush_interval_seconds=flush_interval,
            max_records_per_dataset=max(1, batch_size),
            store_trades=bool(getattr(storage, "store_trades", cls.store_trades)) if storage is not None else cls.store_trades,
            store_closed_candles=bool(getattr(storage, "store_closed_candles", cls.store_closed_candles)) if storage is not None else cls.store_closed_candles,
            store_orderbook_snapshots=bool(getattr(storage, "store_orderbook_snapshots", cls.store_orderbook_snapshots)) if storage is not None else cls.store_orderbook_snapshots,
            store_funding=bool(getattr(storage, "store_funding", cls.store_funding)) if storage is not None else cls.store_funding,
            store_open_interest=bool(getattr(storage, "store_open_interest", cls.store_open_interest)) if storage is not None else cls.store_open_interest,
            store_liquidations=bool(getattr(storage, "store_liquidations", cls.store_liquidations)) if storage is not None else cls.store_liquidations,
            store_analytics=bool(getattr(storage, "store_analytics", cls.store_analytics)) if storage is not None else cls.store_analytics,
            store_strategy_events=bool(getattr(storage, "store_strategy_events", cls.store_strategy_events)) if storage is not None else cls.store_strategy_events,
            store_signal_events=bool(getattr(storage, "store_signal_events", cls.store_signal_events)) if storage is not None else cls.store_signal_events,
            store_risk_events=bool(getattr(storage, "store_risk_events", cls.store_risk_events)) if storage is not None else cls.store_risk_events,
            store_execution_events=bool(getattr(storage, "store_execution_events", cls.store_execution_events)) if storage is not None else cls.store_execution_events,
            store_position_events=bool(getattr(storage, "store_position_events", cls.store_position_events)) if storage is not None else cls.store_position_events,
        )


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
    DATASET_STRATEGY_EVENTS = "strategy_events"
    DATASET_STRATEGY_INPUT_EVENTS = "strategy_input_events"
    DATASET_SIGNAL_EVENTS = "signal_events"
    DATASET_RISK_EVENTS = "risk_events"
    DATASET_EXECUTION_EVENTS = "execution_events"
    DATASET_POSITION_EVENTS = "position_events"

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
            self._event_bus.subscribe("market.candles.snapshot", self._on_candles_event)
            self._event_bus.subscribe("market.candles.updated", self._on_candles_event)
            self._event_bus.subscribe("market.candles.persistable", self._on_candles_event)

        if self._storage_config.store_trades:
            self._event_bus.subscribe("market.trade", self._on_trade)

        if self._storage_config.store_orderbook_snapshots:
            self._event_bus.subscribe(
                "market.orderbook.snapshot.persistable",
                self._on_orderbook_snapshot,
            )

        if self._storage_config.store_funding:
            self._event_bus.subscribe("market.funding.updated", self._on_funding_updated)
            self._event_bus.subscribe("market.funding.snapshot", self._on_funding_event)
            self._event_bus.subscribe("market.funding.persistable", self._on_funding_event)

        if self._storage_config.store_open_interest:
            self._event_bus.subscribe(
                "market.open_interest.updated",
                self._on_open_interest_updated,
            )

        if self._storage_config.store_liquidations:
            self._event_bus.subscribe("market.liquidation", self._on_liquidation)

        if self._storage_config.store_analytics:
            self._event_bus.subscribe("analytics.*", self._on_analytics_event)

        if self._storage_config.store_strategy_events:
            self._event_bus.subscribe("strategy.input.normalized", self._on_strategy_input_event)
            self._event_bus.subscribe("strategy.context.updated", self._on_strategy_input_event)
            self._event_bus.subscribe("strategy.*", self._on_strategy_event)

        if self._storage_config.store_signal_events:
            self._event_bus.subscribe("signal.*", self._on_signal_event)

        if self._storage_config.store_risk_events:
            self._event_bus.subscribe("risk.*", self._on_risk_event)

        if self._storage_config.store_execution_events:
            self._event_bus.subscribe("execution.*", self._on_execution_event)

        if self._storage_config.store_position_events:
            self._event_bus.subscribe("position.*", self._on_position_event)

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

    async def _on_candles_event(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="market.candles.updated")

        candles = self._extract_items(payload, "candles", "klines", "items", "data.candles", "data", "snapshot.candles")
        if not candles:
            record = self._normalize_candle(payload, topic=topic)
            if record is None:
                await self._drop_event(topic, payload, reason="invalid_candles_event")
                return
            await self._buffer_record(self.DATASET_CANDLES, record)
            return

        buffered = 0
        for candle in candles:
            candle_payload = self._merge_parent_scope(payload, candle)
            record = self._normalize_candle(candle_payload, topic=topic)
            if record is not None:
                await self._buffer_record(self.DATASET_CANDLES, record)
                buffered += 1

        if buffered <= 0:
            await self._drop_event(topic, payload, reason="invalid_candles_batch")

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

    async def _on_funding_event(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="market.funding.snapshot")

        items = self._extract_items(payload, "funding", "funding_rates", "items", "data.funding", "data", "snapshot.funding")
        if not items:
            record = self._normalize_funding(payload, topic=topic)
            if record is None:
                await self._drop_event(topic, payload, reason="invalid_funding_event")
                return
            await self._buffer_record(self.DATASET_FUNDING, record)
            return

        buffered = 0
        for item in items:
            funding_payload = self._merge_parent_scope(payload, item)
            record = self._normalize_funding(funding_payload, topic=topic)
            if record is not None:
                await self._buffer_record(self.DATASET_FUNDING, record)
                buffered += 1

        if buffered <= 0:
            await self._drop_event(topic, payload, reason="invalid_funding_batch")

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

    async def _on_strategy_input_event(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="strategy.input.normalized")

        record = self._normalize_strategy_input_event(payload, topic=topic)
        if record is None:
            await self._drop_event(topic, payload, reason="invalid_strategy_input_event")
            return

        await self._buffer_record(self.DATASET_STRATEGY_INPUT_EVENTS, record)

    async def _on_strategy_event(self, event: Event | Mapping[str, Any]) -> None:
        topic = self._extract_topic(event, default="strategy.unknown")
        if topic in {"strategy.input.normalized", "strategy.context.updated"}:
            return

        payload = self._extract_payload(event)
        record = self._normalize_strategy_event(payload, topic=topic)
        if record is None:
            await self._drop_event(topic, payload, reason="invalid_strategy_event")
            return

        await self._buffer_record(self.DATASET_STRATEGY_EVENTS, record)

    async def _on_signal_event(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="signal.unknown")

        record = self._normalize_signal_event(payload, topic=topic)
        if record is None:
            await self._drop_event(topic, payload, reason="invalid_signal_event")
            return

        await self._buffer_record(self.DATASET_SIGNAL_EVENTS, record)

    async def _on_risk_event(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="risk.unknown")

        record = self._normalize_risk_event(payload, topic=topic)
        if record is None:
            await self._drop_event(topic, payload, reason="invalid_risk_event")
            return

        await self._buffer_record(self.DATASET_RISK_EVENTS, record)

    async def _on_execution_event(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="execution.unknown")

        record = self._normalize_execution_event(payload, topic=topic)
        if record is None:
            await self._drop_event(topic, payload, reason="invalid_execution_event")
            return

        await self._buffer_record(self.DATASET_EXECUTION_EVENTS, record)

    async def _on_position_event(self, event: Event | Mapping[str, Any]) -> None:
        payload = self._extract_payload(event)
        topic = self._extract_topic(event, default="position.unknown")

        record = self._normalize_position_event(payload, topic=topic)
        if record is None:
            await self._drop_event(topic, payload, reason="invalid_position_event")
            return

        await self._buffer_record(self.DATASET_POSITION_EVENTS, record)

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


    def _normalize_entity_payload(
        self,
        payload: Mapping[str, Any],
        *candidate_paths: str,
    ) -> dict[str, Any]:
        """
        Build a storage-friendly payload from both legacy flat EventBus events
        and the newer state-driven payload shapes.

        The state-driven pipeline can publish low-frequency persistable events
        where the actual object is nested under keys like ``candle``, ``snapshot``
        or ``data``. Older storage code expected OHLCV/open-interest/funding
        fields on the top level. This helper merges top-level scope fields with
        the first matching nested entity so normalizers can support both shapes.
        """
        base = self._as_mapping(payload)
        if not base:
            return {}

        for path in candidate_paths:
            nested = self._get_path(base, path)
            nested_map = self._as_mapping(nested)
            if nested_map:
                merged = dict(base)
                merged.update(nested_map)
                return merged

        return dict(base)

    def _as_mapping(self, value: Any) -> dict[str, Any]:
        plain = self._plain_value(value)
        if isinstance(plain, Mapping):
            return {str(key): inner for key, inner in plain.items()}
        return {}

    def _plain_value(self, value: Any) -> Any:
        if value is None:
            return None

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            with contextlib.suppress(Exception):
                return self._plain_value(to_dict())

        if is_dataclass(value):
            with contextlib.suppress(Exception):
                return self._plain_value(asdict(value))

        if isinstance(value, Mapping):
            return {
                str(key): self._plain_value(inner)
                for key, inner in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [self._plain_value(item) for item in value]

        # Support simple slots-based DTOs that are not dataclasses.
        slots = getattr(value.__class__, "__slots__", None)
        if slots:
            result: dict[str, Any] = {}
            for slot in slots:
                if isinstance(slot, str) and hasattr(value, slot):
                    result[slot] = self._plain_value(getattr(value, slot))
            if result:
                return result

        if hasattr(value, "__dict__"):
            with contextlib.suppress(Exception):
                return {
                    str(key): self._plain_value(inner)
                    for key, inner in vars(value).items()
                    if not str(key).startswith("_")
                }

        return value

    def _get_path(self, payload: Mapping[str, Any], path: str) -> Any:
        current: Any = payload
        for part in path.split("."):
            current_map = self._as_mapping(current)
            if not current_map or part not in current_map:
                return None
            current = current_map.get(part)
        return current

    def _first_from(
        self,
        payload: Mapping[str, Any],
        *keys: str,
    ) -> Any:
        for key in keys:
            if "." in key:
                value = self._get_path(payload, key)
            else:
                value = payload.get(key)
            if value is not None:
                return value
        return None

    def _extract_items(self, payload: Mapping[str, Any], *candidate_paths: str) -> list[Any]:
        for path in candidate_paths:
            value = self._get_path(payload, path) if "." in path else payload.get(path)
            plain = self._plain_value(value)
            if isinstance(plain, list):
                return plain
            if isinstance(plain, tuple):
                return list(plain)
        return []

    def _merge_parent_scope(self, parent: Mapping[str, Any], item: Any) -> dict[str, Any]:
        item_map = self._as_mapping(item)
        if not item_map:
            return dict(parent)

        merged: dict[str, Any] = {}
        for key in (
            "exchange",
            "market_type",
            "symbol",
            "timeframe",
            "exchange_symbol",
            "scope",
            "scope_key",
            "source",
            "received_at",
            "received_at_ms",
        ):
            if key in parent and key not in item_map:
                merged[key] = parent[key]

        merged.update(item_map)
        return merged

    def _scope_key(self, payload: Mapping[str, Any]) -> str | None:
        explicit = self._safe_str(payload.get("scope_key"))
        if explicit:
            return explicit

        scope = self._as_mapping(payload.get("scope"))
        exchange = self._safe_str(payload.get("exchange") or scope.get("exchange"))
        market_type = self._safe_str(payload.get("market_type") or scope.get("market_type"))
        symbol = self._safe_str(payload.get("symbol") or scope.get("symbol"))
        timeframe = self._safe_str(payload.get("timeframe") or scope.get("timeframe"))

        if exchange and market_type and symbol and timeframe:
            return f"{exchange}:{market_type}:{symbol}:{timeframe}"
        return None

    def _normalize_candle(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        payload = self._normalize_entity_payload(
            payload,
            "candle",
            "closed_candle",
            "last_closed_candle",
            "data.candle",
            "data",
            "snapshot.last_closed_candle",
            "snapshot.candle",
            "snapshot",
            "state.last_closed_candle",
            "state.candle",
        )

        exchange = self._required_str(payload, "exchange")
        symbol = self._required_str(payload, "symbol")
        timeframe = self._required_str(payload, "timeframe")

        open_time_ms = self._first_timestamp_ms(
            payload,
            "open_time_ms",
            "open_time",
            "start_time_ms",
            "start_time",
            "timestamp_ms",
            "timestamp",
            "event_time",
        )

        close_time_ms = self._first_timestamp_ms(
            payload,
            "close_time_ms",
            "close_time",
            "end_time_ms",
            "end_time",
        )

        if open_time_ms is None and close_time_ms is not None and timeframe:
            open_time_ms = close_time_ms - self._timeframe_to_ms(timeframe) + 1

        if exchange is None or symbol is None or timeframe is None or open_time_ms is None:
            return None

        # ``market.candle.closed`` is already a closed-candle topic. In the
        # state-driven flow some emitters use this topic as a low-frequency
        # persistable trigger without carrying ``is_closed`` explicitly.
        is_closed = (
            self._safe_bool(payload.get("is_closed"))
            or self._safe_bool(payload.get("closed"))
            or topic.endswith(".closed")
        )
        if not is_closed:
            return None

        if close_time_ms is None:
            close_time_ms = self._infer_close_time_ms(open_time_ms, timeframe)

        open_price = self._safe_float(payload.get("open"))
        high_price = self._safe_float(payload.get("high"))
        low_price = self._safe_float(payload.get("low"))
        close_price = self._safe_float(payload.get("close"))

        if open_price is None or high_price is None or low_price is None or close_price is None:
            return None

        volume = self._first_float(payload, "volume", "base_volume", "qty", "quantity")
        quote_volume = self._first_float(payload, "quote_volume", "quote_asset_volume", "turnover")
        trades_count = self._first_int(payload, "trades_count", "trade_count", "num_trades", "number_of_trades")

        return {
            "topic": topic,
            "dataset": self.DATASET_CANDLES,
            "exchange": exchange,
            "symbol": symbol,
            "market_type": self._market_type(payload),
            "timeframe": timeframe,
            "open_time_ms": open_time_ms,
            "close_time_ms": close_time_ms,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "quote_volume": quote_volume,
            "trades_count": trades_count,
            "is_closed": True,
            "timestamp_ms": self._event_timestamp(payload, fallback=open_time_ms),
            "received_at_ms": self._received_at(payload),
            "source": payload.get("source") or payload.get("price_source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_trade(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        payload = self._normalize_entity_payload(
            payload,
            "trade",
            "data.trade",
            "data",
            "snapshot.trade",
            "event.trade",
        )

        exchange = self._required_str(payload, "exchange")
        symbol = self._required_str(payload, "symbol")

        timestamp_ms = self._first_timestamp_ms(
            payload,
            "timestamp_ms",
            "trade_time_ms",
            "trade_time",
            "event_time",
            "time",
            "timestamp",
        )

        price = self._safe_float(payload.get("price"))
        quantity = self._first_float(payload, "quantity", "qty", "size", "amount", "base_quantity")

        if exchange is None or symbol is None or timestamp_ms is None:
            return None
        if price is None or quantity is None:
            return None

        side = self._normalize_side(payload.get("side"))
        aggressor_side = self._normalize_side(
            payload.get("aggressor_side")
            or payload.get("taker_side")
            or payload.get("buyer_side")
            or side
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
            "quote_quantity": self._first_float(payload, "quote_quantity", "quote_qty", "quote_size", "turnover"),
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
        payload = self._normalize_entity_payload(
            payload,
            "orderbook",
            "book",
            "snapshot.orderbook",
            "snapshot.book",
            "snapshot",
            "data.orderbook",
            "data.book",
            "data",
        )

        exchange = self._required_str(payload, "exchange")
        symbol = self._required_str(payload, "symbol")

        if exchange is None or symbol is None:
            return None

        bids = payload.get("bids") or []
        asks = payload.get("asks") or []

        if not isinstance(bids, list) or not isinstance(asks, list):
            return None

        timestamp_ms = self._event_timestamp(payload)

        best_bid = (
            self._safe_float(payload.get("best_bid"))
            or self._first_level_price(bids)
        )
        best_ask = (
            self._safe_float(payload.get("best_ask"))
            or self._first_level_price(asks)
        )
        mid_price = (
            self._safe_float(payload.get("mid_price"))
            or self._calc_mid_price(best_bid, best_ask)
        )

        return {
            "topic": topic,
            "dataset": self.DATASET_ORDERBOOK_SNAPSHOTS,
            "exchange": exchange,
            "symbol": symbol,
            "market_type": self._market_type(payload),
            "timestamp_ms": timestamp_ms or self._now_ms(),
            "received_at_ms": self._received_at(payload),
            "sequence": self._first_int(payload, "sequence", "update_id", "last_update_id", "final_update_id"),
            "depth": max(len(bids), len(asks)),
            "bids_json": self._json_dumps(bids),
            "asks_json": self._json_dumps(asks),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": self._calc_spread(best_bid, best_ask),
            "mid_price": mid_price,
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_funding(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        payload = self._normalize_entity_payload(
            payload,
            "funding",
            "funding_snapshot",
            "snapshot.funding",
            "snapshot",
            "data.funding",
            "data",
        )

        exchange = self._required_str(payload, "exchange")
        symbol = self._required_str(payload, "symbol")

        timestamp_ms = self._first_timestamp_ms(
            payload,
            "timestamp_ms",
            "funding_time_ms",
            "funding_time",
            "time",
            "timestamp",
            "event_time",
        )

        funding_rate = self._first_float(payload, "funding_rate", "rate")

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
            "next_funding_time_ms": self._first_timestamp_ms(
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
        payload = self._normalize_entity_payload(
            payload,
            "open_interest",
            "open_interest_snapshot",
            "snapshot.open_interest",
            "snapshot",
            "data.open_interest",
            "data",
        )

        exchange = self._required_str(payload, "exchange")
        symbol = self._required_str(payload, "symbol")

        timestamp_ms = self._first_timestamp_ms(
            payload,
            "timestamp_ms",
            "open_interest_time_ms",
            "open_interest_time",
            "time",
            "timestamp",
            "event_time",
        )

        open_interest = self._first_float(payload, "open_interest", "oi")

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
            "index_price": self._safe_float(payload.get("index_price")),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_liquidation(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        payload = self._normalize_entity_payload(
            payload,
            "liquidation",
            "liquidation_event",
            "event.liquidation",
            "event",
            "data.liquidation",
            "data",
            "snapshot.liquidation",
        )

        exchange = self._required_str(payload, "exchange")
        symbol = self._required_str(payload, "symbol")

        timestamp_ms = self._event_timestamp(payload)

        price = self._safe_float(payload.get("price"))
        quantity = self._first_float(payload, "quantity", "qty", "size", "amount", "base_quantity")
        notional = self._safe_float(payload.get("notional"))
        if notional is None and price is not None and quantity is not None:
            notional = price * quantity

        if exchange is None or symbol is None or timestamp_ms is None:
            return None
        if price is None and notional is None:
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
            "notional": notional,
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
            "domain": self._safe_str(payload.get("domain")),
            "scope_key": self._scope_key(payload),
            "direction": self._safe_str(payload.get("direction") or payload.get("side")),
            "signal_type": self._safe_str(payload.get("signal_type") or payload.get("setup_type")),
            "reason": self._safe_str(payload.get("reason")),
            "score": self._safe_float(payload.get("score") or payload.get("strength")),
            "strength": self._safe_float(payload.get("strength")),
            "confidence": self._safe_float(payload.get("confidence")),
            "price": self._first_float(payload, "current_price", "entry_reference_price", "reference_price", "price", "last_price"),
            "features_json": self._json_dumps(payload.get("features", {})),
            "metadata_json": self._json_dumps(payload.get("metadata", {})),
            "payload_json": self._json_dumps(payload),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_strategy_input_event(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        timestamp_ms = self._event_timestamp(payload)
        context = self._as_mapping(payload.get("context"))
        normalized_payload = self._as_mapping(payload.get("normalized_payload"))

        exchange = self._safe_str(payload.get("exchange") or context.get("exchange") or normalized_payload.get("exchange"))
        symbol = self._safe_str(payload.get("symbol") or context.get("symbol") or normalized_payload.get("symbol"))
        timeframe = self._safe_str(payload.get("timeframe") or context.get("timeframe") or normalized_payload.get("timeframe"))

        return {
            "topic": topic,
            "dataset": self.DATASET_STRATEGY_INPUT_EVENTS,
            "event_type": topic,
            "source_topic": self._safe_str(payload.get("source_topic") or payload.get("event_name") or payload.get("analytics_topic")),
            "domain": self._safe_str(payload.get("domain") or normalized_payload.get("domain")),
            "feature_source": self._safe_str(payload.get("feature_source") or normalized_payload.get("feature_source")),
            "exchange": exchange,
            "symbol": symbol,
            "market_type": self._market_type(payload),
            "timeframe": timeframe,
            "scope_key": self._scope_key(payload) or self._scope_key(context) or self._scope_key(normalized_payload),
            "timestamp_ms": timestamp_ms or self._now_ms(),
            "received_at_ms": self._received_at(payload),
            "side": self._safe_str(payload.get("side") or payload.get("direction") or normalized_payload.get("side") or normalized_payload.get("direction")),
            "signal_type": self._safe_str(payload.get("signal_type") or payload.get("setup_type") or normalized_payload.get("signal_type") or normalized_payload.get("setup_type")),
            "reason": self._safe_str(payload.get("reason") or normalized_payload.get("reason")),
            "score": self._first_float(payload, "score", "strength", "confidence"),
            "strength": self._safe_float(payload.get("strength") or normalized_payload.get("strength")),
            "confidence": self._safe_float(payload.get("confidence") or normalized_payload.get("confidence")),
            "price": self._first_float(payload, "current_price", "entry_reference_price", "reference_price", "price", "last_price"),
            "contract_version": self._safe_str(
                payload.get("strategy_contract_version")
                or self._get_path(payload, "strategy_contract.version")
                or self._get_path(payload, "contract.version")
            ),
            "context_json": self._json_dumps(payload.get("context", {})),
            "features_json": self._json_dumps(payload.get("features", {})),
            "metadata_json": self._json_dumps(payload.get("metadata", {})),
            "contract_json": self._json_dumps(payload.get("strategy_contract") or payload.get("contract") or {}),
            "normalized_payload_json": self._json_dumps(payload.get("normalized_payload", {})),
            "payload_json": self._json_dumps(payload),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_strategy_event(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        timestamp_ms = self._event_timestamp(payload)
        return {
            "topic": topic,
            "dataset": self.DATASET_STRATEGY_EVENTS,
            "event_type": topic,
            "exchange": self._safe_str(payload.get("exchange")),
            "symbol": self._safe_str(payload.get("symbol")),
            "market_type": self._market_type(payload),
            "timeframe": self._safe_str(payload.get("timeframe")),
            "scope_key": self._scope_key(payload),
            "timestamp_ms": timestamp_ms or self._now_ms(),
            "received_at_ms": self._received_at(payload),
            "strategy_name": self._safe_str(payload.get("strategy_name") or payload.get("strategy")),
            "batch_id": self._safe_str(payload.get("batch_id")),
            "status": self._safe_str(payload.get("status")),
            "reason": self._safe_str(payload.get("reason")),
            "signals_count": self._first_int(payload, "signals_count", "signal_count", "total_signals"),
            "accepted_count": self._first_int(payload, "accepted_count", "accepted"),
            "rejected_count": self._first_int(payload, "rejected_count", "rejected"),
            "payload_json": self._json_dumps(payload),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_signal_event(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        timestamp_ms = self._event_timestamp(payload)
        return {
            "topic": topic,
            "dataset": self.DATASET_SIGNAL_EVENTS,
            "event_type": topic,
            "exchange": self._safe_str(payload.get("exchange")),
            "symbol": self._safe_str(payload.get("symbol")),
            "market_type": self._market_type(payload),
            "timeframe": self._safe_str(payload.get("timeframe")),
            "scope_key": self._scope_key(payload),
            "timestamp_ms": timestamp_ms or self._now_ms(),
            "received_at_ms": self._received_at(payload),
            "signal_id": self._safe_str(payload.get("signal_id") or payload.get("id")),
            "strategy_name": self._safe_str(payload.get("strategy_name") or payload.get("strategy")),
            "side": self._safe_str(payload.get("side") or payload.get("direction")),
            "status": self._safe_str(payload.get("status")),
            "reason": self._safe_str(payload.get("reason")),
            "confidence": self._safe_float(payload.get("confidence")),
            "score": self._safe_float(payload.get("score") or payload.get("strength")),
            "priority": self._safe_str(payload.get("priority")),
            "entry_price": self._first_float(payload, "entry_price", "current_price", "reference_price", "price"),
            "stop_loss": self._first_float(payload, "stop_loss", "sl"),
            "take_profit": self._first_float(payload, "take_profit", "tp"),
            "payload_json": self._json_dumps(payload),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_risk_event(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        timestamp_ms = self._event_timestamp(payload)
        return {
            "topic": topic,
            "dataset": self.DATASET_RISK_EVENTS,
            "event_type": topic,
            "exchange": self._safe_str(payload.get("exchange")),
            "symbol": self._safe_str(payload.get("symbol")),
            "market_type": self._market_type(payload),
            "timeframe": self._safe_str(payload.get("timeframe")),
            "timestamp_ms": timestamp_ms or self._now_ms(),
            "received_at_ms": self._received_at(payload),
            "signal_id": self._safe_str(payload.get("signal_id")),
            "strategy_name": self._safe_str(payload.get("strategy_name") or payload.get("strategy")),
            "decision": self._safe_str(payload.get("decision") or payload.get("risk_decision") or payload.get("status")),
            "reason": self._safe_str(payload.get("reason")),
            "side": self._safe_str(payload.get("side") or payload.get("position_side")),
            "final_size": self._safe_float(payload.get("final_size") or payload.get("size")),
            "final_leverage": self._safe_float(payload.get("final_leverage") or payload.get("leverage")),
            "final_margin": self._safe_float(payload.get("final_margin") or payload.get("margin")),
            "final_notional": self._safe_float(payload.get("final_notional") or payload.get("notional")),
            "final_risk_amount": self._safe_float(payload.get("final_risk_amount") or payload.get("risk_amount")),
            "reservation_id": self._safe_str(payload.get("reservation_id")),
            "payload_json": self._json_dumps(payload),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_execution_event(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        timestamp_ms = self._event_timestamp(payload)
        return {
            "topic": topic,
            "dataset": self.DATASET_EXECUTION_EVENTS,
            "event_type": topic,
            "exchange": self._safe_str(payload.get("exchange")),
            "symbol": self._safe_str(payload.get("symbol")),
            "market_type": self._market_type(payload),
            "timestamp_ms": timestamp_ms or self._now_ms(),
            "received_at_ms": self._received_at(payload),
            "order_id": self._safe_str(payload.get("order_id") or payload.get("exchange_order_id")),
            "client_order_id": self._safe_str(payload.get("client_order_id") or payload.get("new_client_order_id")),
            "position_id": self._safe_str(payload.get("position_id")),
            "signal_id": self._safe_str(payload.get("signal_id")),
            "strategy_name": self._safe_str(payload.get("strategy_name") or payload.get("strategy")),
            "side": self._safe_str(payload.get("side") or payload.get("order_side")),
            "order_type": self._safe_str(payload.get("order_type") or payload.get("type")),
            "status": self._safe_str(payload.get("status") or payload.get("order_status")),
            "price": self._first_float(payload, "price", "avg_price", "average_price"),
            "quantity": self._first_float(payload, "quantity", "qty", "size", "executed_quantity"),
            "filled_quantity": self._first_float(payload, "filled_quantity", "filled_qty", "executed_qty"),
            "notional": self._safe_float(payload.get("notional")),
            "reason": self._safe_str(payload.get("reason") or payload.get("error")),
            "payload_json": self._json_dumps(payload),
            "source": payload.get("source"),
            "ingested_at_ms": self._now_ms(),
        }

    def _normalize_position_event(
        self,
        payload: Mapping[str, Any],
        *,
        topic: str,
    ) -> dict[str, Any] | None:
        timestamp_ms = self._event_timestamp(payload)
        return {
            "topic": topic,
            "dataset": self.DATASET_POSITION_EVENTS,
            "event_type": topic,
            "exchange": self._safe_str(payload.get("exchange")),
            "symbol": self._safe_str(payload.get("symbol")),
            "market_type": self._market_type(payload),
            "timestamp_ms": timestamp_ms or self._now_ms(),
            "received_at_ms": self._received_at(payload),
            "position_id": self._safe_str(payload.get("position_id") or payload.get("id")),
            "signal_id": self._safe_str(payload.get("signal_id")),
            "strategy_name": self._safe_str(payload.get("strategy_name") or payload.get("strategy")),
            "side": self._safe_str(payload.get("side") or payload.get("position_side")),
            "status": self._safe_str(payload.get("status")),
            "size": self._first_float(payload, "size", "quantity", "position_amt", "position_amount"),
            "entry_price": self._safe_float(payload.get("entry_price")),
            "mark_price": self._safe_float(payload.get("mark_price")),
            "notional_value": self._safe_float(payload.get("notional_value") or payload.get("notional")),
            "leverage": self._safe_float(payload.get("leverage")),
            "margin_used": self._safe_float(payload.get("margin_used") or payload.get("margin")),
            "risk_amount": self._safe_float(payload.get("risk_amount")),
            "realized_pnl": self._safe_float(payload.get("realized_pnl")),
            "unrealized_pnl": self._safe_float(payload.get("unrealized_pnl")),
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

        if dataset == self.DATASET_STRATEGY_INPUT_EVENTS:
            domain = self._partition_value(record.get("domain"), default="unknown")
            timeframe = self._partition_value(record.get("timeframe"), default="unknown")
            return (
                ("domain", domain),
                ("exchange", exchange),
                ("symbol", symbol),
                ("timeframe", timeframe),
                ("date", date),
            )

        if dataset == self.DATASET_STRATEGY_EVENTS:
            event_type = self._partition_value(record.get("event_type"), default="unknown")
            strategy_name = self._partition_value(record.get("strategy_name"), default="unknown")
            return (
                ("event_type", event_type),
                ("strategy_name", strategy_name),
                ("symbol", symbol),
                ("date", date),
            )

        if dataset == self.DATASET_SIGNAL_EVENTS:
            event_type = self._partition_value(record.get("event_type"), default="unknown")
            strategy_name = self._partition_value(record.get("strategy_name"), default="unknown")
            return (
                ("event_type", event_type),
                ("strategy_name", strategy_name),
                ("symbol", symbol),
                ("date", date),
            )

        if dataset == self.DATASET_RISK_EVENTS:
            event_type = self._partition_value(record.get("event_type"), default="unknown")
            return (
                ("event_type", event_type),
                ("symbol", symbol),
                ("date", date),
            )

        if dataset == self.DATASET_EXECUTION_EVENTS:
            event_type = self._partition_value(record.get("event_type"), default="unknown")
            return (
                ("event_type", event_type),
                ("exchange", exchange),
                ("symbol", symbol),
                ("date", date),
            )

        if dataset == self.DATASET_POSITION_EVENTS:
            event_type = self._partition_value(record.get("event_type"), default="unknown")
            return (
                ("event_type", event_type),
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
            payload = self._plain_value(event.payload)
            return payload if isinstance(payload, Mapping) else {}

        if isinstance(event, Mapping):
            plain_event = self._plain_value(event)
            if not isinstance(plain_event, Mapping):
                return {}

            payload = plain_event.get("payload")
            payload = self._plain_value(payload)
            if isinstance(payload, Mapping):
                return payload

            return plain_event

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
            or payload.get("contract_type")
            or "usdm_futures"
        )
        return str(value).lower()

    def _event_timestamp(
        self,
        payload: Mapping[str, Any],
        *,
        fallback: int | None = None,
    ) -> int | None:
        return self._first_timestamp_ms(
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
            self._first_timestamp_ms(payload, "received_at_ms", "received_at")
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

        if isinstance(value, datetime):
            dt = value
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None

            # Numeric strings are common for exchange timestamps.
            with contextlib.suppress(ValueError, TypeError):
                return int(float(text))

            # ISO timestamps appear in state-driven DTOs and analytics payloads.
            normalized = text.replace("Z", "+00:00")
            with contextlib.suppress(ValueError):
                dt = datetime.fromisoformat(normalized)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)

            return None

        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _normalize_timestamp_ms(self, value: Any) -> int | None:
        """
        Normalize external timestamps to milliseconds.

        Exchange, analytics and storage DTOs in this project can carry time as:
        - seconds, e.g. 1760000000;
        - milliseconds, e.g. 1760000000000;
        - microseconds, e.g. 1760000000000000;
        - nanoseconds, e.g. 1760000000000000000;
        - datetime / ISO strings, which _safe_int already converts to ms.

        Returning a single millisecond representation prevents Parquet partitions
        such as date=1970-01-21 when a seconds timestamp is divided by 1000 again.
        """
        timestamp = self._safe_int(value)
        if timestamp is None:
            return None

        absolute = abs(timestamp)

        # Zero is technically valid but useless for live market data; keep it as
        # epoch ms rather than treating it as missing.
        if absolute == 0:
            return 0

        # Seconds: current timestamps are around 1.7e9.
        if absolute < 10_000_000_000:
            return timestamp * 1000

        # Milliseconds: current timestamps are around 1.7e12.
        if absolute < 10_000_000_000_000:
            return timestamp

        # Microseconds: current timestamps are around 1.7e15.
        if absolute < 10_000_000_000_000_000:
            return timestamp // 1000

        # Nanoseconds or larger precision.
        return timestamp // 1_000_000

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

    def _first_timestamp_ms(
        self,
        payload: Mapping[str, Any],
        *keys: str,
        fallback: int | None = None,
    ) -> int | None:
        for key in keys:
            value = self._normalize_timestamp_ms(payload.get(key))
            if value is not None:
                return value
        return self._normalize_timestamp_ms(fallback)

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
        timestamp = self._normalize_timestamp_ms(timestamp_ms) or self._now_ms()
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
        value = self._plain_value(value)

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