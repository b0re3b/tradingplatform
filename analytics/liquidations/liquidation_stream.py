from __future__ import annotations

import hashlib
import inspect
import time
from collections import deque
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import LiquidationStreamConfig
from .enums import LiquidationEventType, LiquidationSide
from .metrics import LiquidationMetrics
from .models import LiquidationEvent
from .state import LiquidationState
from .utils import (
    ensure_utc,
    is_stale_event,
    normalize_symbol,
    safe_decimal,
    utc_now,
)


@runtime_checkable
class LiquidationHistoryStoreProtocol(Protocol):
    """
    Optional storage hook for normalized liquidation history.

    Production implementation should live outside analytics, for example in
    storage/liquidation_parquet_store.py.  The stream only calls this protocol
    and does not know whether the backend is parquet, database, or in-memory.
    """

    async def append_event(self, event: LiquidationEvent) -> None:
        ...

    async def append_large_event(self, event: LiquidationEvent) -> None:
        ...

    async def flush(self) -> None:
        ...


class LiquidationStream:
    """
    Multi-exchange EventBus consumer/cache for liquidation events.

    Responsibilities:
    - subscribe to raw normalized-by-exchange liquidation payloads from EventBus;
    - accept data from many exchanges on the same topic without symbol conflicts;
    - normalize raw payloads into LiquidationEvent;
    - filter invalid/stale/duplicate events;
    - update shared LiquidationState using (exchange, symbol) keys;
    - update LiquidationMetrics;
    - optionally append normalized events to an injected history store;
    - publish market.liquidation.normalized / market.liquidation.large /
      market.liquidations.updated events;
    - register health/snapshot/cleanup jobs through core.Scheduler.

    This class intentionally does NOT connect to exchange WebSockets directly.
    Exchange adapters should publish raw liquidation payloads to EventBus, for
    example: BybitWS -> emit("market.liquidation", payload).  This stream is
    the data/cache layer between exchange adapters and analytics detectors.
    """

    DEFAULT_INPUT_TOPIC = "market.liquidation"
    DEFAULT_UPDATED_TOPIC = "market.liquidations.updated"

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: LiquidationStreamConfig,
        scheduler: Scheduler | None = None,
        state: LiquidationState | None = None,
        metrics: LiquidationMetrics | None = None,
        history_store: LiquidationHistoryStoreProtocol | None = None,
        service_name: str = "liquidation_stream",
    ) -> None:
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.config = config
        self.service_name = service_name
        self.history_store = history_store

        self.state = state or LiquidationState(
            max_events_per_symbol=self.config.max_buffer_size_per_symbol,
        )
        self.metrics = metrics or LiquidationMetrics()

        self.input_topic = str(
            getattr(self.config, "input_topic_raw", None)
            or getattr(self.config, "input_topic", None)
            or self.DEFAULT_INPUT_TOPIC
        )
        self.publish_topic_updated = str(
            getattr(self.config, "publish_topic_updated", None)
            or self.DEFAULT_UPDATED_TOPIC
        )

        self.logger = get_logger(
            __name__,
            event_type="analytics.liquidations.stream",
            service=service_name,
        )

        if state is None:
            self.logger.warning(
                "LiquidationStream initialized without shared LiquidationState. "
                "For production, pass the same LiquidationState instance to "
                "LiquidationStream and CascadeDetector."
            )

        self._running = False
        self._registered = False
        self._closed = False
        self._subscription: Subscription | None = None

        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None
        self._last_message_at: datetime | None = None
        self._last_event_at: datetime | None = None
        self._last_error_at: datetime | None = None
        self._last_error: str | None = None

        self._processed_messages = 0
        self._processed_events = 0
        self._dropped_invalid = 0
        self._dropped_stale = 0
        self._dropped_duplicates = 0
        self._storage_errors = 0

        self._published_raw = 0
        self._published_normalized = 0
        self._published_large = 0
        self._published_updated = 0
        self._published_health = 0
        self._published_snapshots = 0

        self._recent_payload_fingerprints: deque[str] = deque(
            maxlen=max(1, self.config.recent_payload_fingerprints_size),
        )
        self._recent_payload_fingerprint_set: set[str] = set()

        self._recent_large_events: deque[LiquidationEvent] = deque(
            maxlen=max(1, self.config.recent_large_events_size),
        )

        self._healthcheck_job_id: str | None = None
        self._snapshot_job_id: str | None = None
        self._cleanup_job_id: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """Register EventBus subscription and Scheduler jobs."""
        if self._registered:
            self.logger.warning("LiquidationStream already registered")
            return

        if self._closed:
            raise RuntimeError("Cannot register closed LiquidationStream")

        self._subscription = self.event_bus.subscribe(
            self.input_topic,
            self.on_raw_liquidation,
            name=f"{self.service_name}.on_raw_liquidation",
        )
        self._register_scheduler_jobs()
        self._registered = True

        self.logger.info(
            "LiquidationStream registered | input_topic=%s scheduler_enabled=%s",
            self.input_topic,
            self.scheduler is not None,
        )

    async def unregister(self) -> None:
        """Remove EventBus subscription and scheduler jobs."""
        if not self._registered:
            return

        if self._subscription is not None:
            self.event_bus.unsubscribe(self._subscription)
            self._subscription = None

        await self._remove_scheduler_jobs()
        self._registered = False

        self.logger.info("LiquidationStream unregistered")

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot start closed LiquidationStream")

        if self._running:
            self.logger.warning("LiquidationStream already running")
            return

        if not self.config.enabled:
            self.logger.warning("LiquidationStream is disabled by config")
            return

        if not self._registered:
            self.register()

        self._running = True
        self._started_at = utc_now()
        self._stopped_at = None
        self._last_error = None
        self._last_error_at = None

        self.logger.info(
            "LiquidationStream started | input_topic=%s symbols=%s exchanges=%s",
            self.input_topic,
            self.config.symbols,
            self.config.exchanges,
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._stopped_at = utc_now()

        if self.history_store is not None:
            await self._flush_history_store()

        self.logger.info("LiquidationStream stopped | stats=%s", self.get_stats())

    async def close(self) -> None:
        if self._closed:
            return

        await self.stop()
        await self.unregister()
        self._closed = True

        self.logger.info("LiquidationStream closed")

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    # ------------------------------------------------------------------
    # EventBus input handler
    # ------------------------------------------------------------------

    async def on_raw_liquidation(self, event: Event) -> None:
        """
        Consume raw liquidation payloads from exchange adapters.

        Expected topic: market.liquidation by default.  Payload must be a dict
        containing at least exchange, symbol, side, price, quantity/qty and a
        timestamp-like field.  The handler is multi-exchange safe because every
        normalized event carries its own exchange and symbol.
        """
        if not self._running:
            return

        started = time.perf_counter()
        payload = event.payload
        self._processed_messages += 1
        self._last_message_at = utc_now()

        if not isinstance(payload, dict):
            self._dropped_invalid += 1
            self.metrics.observe_invalid_event()
            self.logger.debug(
                "Non-dict liquidation payload dropped | topic=%s payload_type=%s",
                event.topic,
                type(payload).__name__,
            )
            return

        enriched_payload = dict(payload)
        enriched_payload.setdefault("correlation_id", event.correlation_id)
        enriched_payload.setdefault("source_topic", event.topic)

        try:
            await self.handle_raw_message(enriched_payload)
        finally:
            self.metrics.observe_latency_ms((time.perf_counter() - started) * 1000.0)

    async def handle_raw_message(self, payload: dict[str, Any]) -> LiquidationEvent | None:
        """
        Normalize, validate, deduplicate, store, and publish a raw liquidation payload.
        """
        if self.config.exchanges:
            exchange = self._extract_exchange(payload)
            if exchange and exchange not in {item.strip().lower() for item in self.config.exchanges}:
                self.logger.debug(
                    "Liquidation payload ignored due to exchange filter | exchange=%s",
                    exchange,
                )
                return None

        if self.config.symbols:
            symbol = self._extract_symbol(payload)
            allowed_symbols = {normalize_symbol(item) for item in self.config.symbols}
            if symbol and symbol not in allowed_symbols:
                self.logger.debug(
                    "Liquidation payload ignored due to symbol filter | symbol=%s",
                    symbol,
                )
                return None

        fingerprint = self._make_payload_fingerprint(payload)

        if self.config.deduplication_enabled:
            if self._is_duplicate_payload_fingerprint(fingerprint):
                self._dropped_duplicates += 1
                self.logger.debug(
                    "Duplicate liquidation payload dropped | fingerprint=%s",
                    fingerprint,
                )
                return None
            self._remember_payload_fingerprint(fingerprint)

        if self.config.emit_raw_events:
            await self._publish_raw_payload(payload, fingerprint=fingerprint)

        event = self.normalize_event(payload, raw_payload_hash=fingerprint)
        if event is None:
            self._dropped_invalid += 1
            self.metrics.observe_invalid_event(
                exchange=self._extract_exchange(payload) or None,
                symbol=self._extract_symbol(payload) or None,
            )
            return None

        is_large = event.is_large_at(self.config.large_liquidation_threshold_usd)

        if not event.is_valid:
            self._dropped_invalid += 1
            self.metrics.observe_event(event, is_valid=False, is_stale=False, is_large=False)
            return None

        if is_stale_event(event, stale_after_seconds=self.config.stale_event_threshold_seconds):
            self._dropped_stale += 1
            self.metrics.observe_event(event, is_valid=True, is_stale=True, is_large=is_large)
            self.logger.debug(
                "Stale liquidation event dropped | exchange=%s symbol=%s event_ts=%s",
                event.exchange,
                event.symbol,
                event.timestamp.isoformat(),
            )
            return None

        symbol_state = self.state.add_event(event)
        self._processed_events += 1
        self._last_event_at = ensure_utc(event.timestamp)

        self.metrics.observe_event(event, is_valid=True, is_stale=False, is_large=is_large)
        await self._append_history_event(event, is_large=is_large)

        await self.publish_event(event)
        await self.publish_updated_event(event, symbol_state_total=getattr(symbol_state, "total_buffered_events", None))

        if is_large and self.config.emit_large_events:
            self._recent_large_events.append(event)
            await self.publish_large_event(event)

        self.logger.debug(
            "Liquidation event processed | exchange=%s symbol=%s side=%s notional=%s buffered_events=%s",
            event.exchange,
            event.symbol,
            event.side.value,
            str(event.notional_usd),
            getattr(symbol_state, "total_buffered_events", None),
        )
        return event

    def normalize_event(
        self,
        payload: dict[str, Any],
        *,
        raw_payload_hash: str | None = None,
    ) -> LiquidationEvent | None:
        """
        Normalize a flat or exchange-shaped liquidation payload into LiquidationEvent.
        """
        try:
            exchange = self._extract_exchange(payload)
            symbol = self._extract_symbol(payload)
            side = self._extract_side(payload)
            price = self._extract_price(payload)
            quantity = self._extract_quantity(payload)
            notional_usd = self._extract_notional(payload, price=price, quantity=quantity)
            timestamp = self._extract_timestamp(payload)

            if (
                not exchange
                or not symbol
                or not side.is_known
                or price <= 0
                or quantity <= 0
                or notional_usd <= 0
            ):
                return None

            return LiquidationEvent(
                exchange=exchange,
                symbol=symbol,
                side=side,
                price=price,
                quantity=quantity,
                notional_usd=notional_usd,
                timestamp=timestamp,
                event_type=LiquidationEventType.NORMALIZED,
                trade_id=self._extract_trade_id(payload),
                order_id=self._extract_order_id(payload),
                event_id=self._extract_event_id(payload),
                correlation_id=self._extract_correlation_id(payload),
                source=self.service_name,
                raw_payload_hash=raw_payload_hash,
                metadata={
                    "raw_type": payload.get("type") or payload.get("e") or payload.get("event"),
                    "source_topic": payload.get("source_topic"),
                    "source_payload_keys": sorted(str(key) for key in payload.keys()),
                },
            )
        except Exception as exc:
            self._last_error = repr(exc)
            self._last_error_at = utc_now()
            self.logger.exception(
                "Failed to normalize liquidation payload | error=%s payload_preview=%s",
                repr(exc),
                self._safe_payload_preview(payload),
            )
            return None

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish_event(self, event: LiquidationEvent) -> bool:
        accepted = await self.event_bus.emit(
            self.config.publish_topic_normalized,
            event,
            priority=EventPriority.NORMAL,
            source=self.service_name,
            correlation_id=event.correlation_id,
            headers={
                "exchange": event.exchange,
                "symbol": event.symbol,
                "event_type": event.event_type.value,
            },
        )
        if accepted:
            self._published_normalized += 1
        return accepted

    async def publish_updated_event(self, event: LiquidationEvent, *, symbol_state_total: int | None = None) -> bool:
        payload = {
            "exchange": event.exchange,
            "symbol": event.symbol,
            "market_type": event.metadata.get("market_type") or "perpetual",
            "event": event.to_dict(serialize=True),
            "last_event_at": event.timestamp.isoformat(),
            "last_received_at": event.received_at.isoformat(),
            "symbol_state_total": symbol_state_total,
        }
        accepted = await self.event_bus.emit(
            self.publish_topic_updated,
            payload,
            priority=EventPriority.NORMAL,
            source=self.service_name,
            correlation_id=event.correlation_id,
            headers={
                "exchange": event.exchange,
                "symbol": event.symbol,
                "event_type": "updated",
            },
        )
        if accepted:
            self._published_updated += 1
        return accepted

    async def publish_large_event(self, event: LiquidationEvent) -> bool:
        accepted = await self.event_bus.emit(
            self.config.publish_topic_large,
            event,
            priority=EventPriority.HIGH,
            source=self.service_name,
            correlation_id=event.correlation_id,
            headers={
                "exchange": event.exchange,
                "symbol": event.symbol,
                "event_type": LiquidationEventType.LARGE.value,
            },
        )
        if accepted:
            self._published_large += 1
        return accepted

    async def _publish_raw_payload(self, payload: dict[str, Any], *, fingerprint: str) -> bool:
        exchange = self._extract_exchange(payload) or "unknown"
        raw_event = {
            "exchange": exchange,
            "symbol": self._extract_symbol(payload) or None,
            "received_at": utc_now().isoformat(),
            "fingerprint": fingerprint,
            "payload": payload,
        }
        accepted = await self.event_bus.emit(
            self.config.publish_topic_raw,
            raw_event,
            priority=EventPriority.LOW,
            source=self.service_name,
            headers={
                "exchange": exchange,
                "event_type": LiquidationEventType.RAW.value,
            },
        )
        if accepted:
            self._published_raw += 1
        return accepted

    async def emit_runtime_snapshot(self, topic: str | None = None) -> bool:
        snapshot = {
            "service": self.service_name,
            "input_topic": self.input_topic,
            "health": self.estimate_ingestion_health(),
            "stats": self.get_stats(),
            "metrics": self.metrics.to_dict(serialize=True),
            "state": [item.to_dict(serialize=True) for item in self.state.snapshots()],
            "emitted_at": utc_now().isoformat(),
        }
        accepted = await self.event_bus.emit(
            topic or self.config.publish_topic_snapshot,
            snapshot,
            priority=EventPriority.LOW,
            source=self.service_name,
            headers={"event_type": LiquidationEventType.SNAPSHOT.value},
        )
        if accepted:
            self._published_snapshots += 1
        return accepted

    async def emit_health(self) -> bool:
        health = self.estimate_ingestion_health()
        accepted = await self.event_bus.emit(
            self.config.publish_topic_health,
            health,
            priority=EventPriority.LOW,
            source=self.service_name,
            headers={"event_type": LiquidationEventType.HEALTH.value},
        )
        if accepted:
            self._published_health += 1
        return accepted

    # ------------------------------------------------------------------
    # Optional storage hook
    # ------------------------------------------------------------------

    async def _append_history_event(self, event: LiquidationEvent, *, is_large: bool) -> None:
        if self.history_store is None:
            return
        try:
            await self.history_store.append_event(event)
            if is_large:
                await self.history_store.append_large_event(event)
        except Exception as exc:
            self._storage_errors += 1
            self._last_error = repr(exc)
            self._last_error_at = utc_now()
            self.logger.exception(
                "Failed to append liquidation event to history store | exchange=%s symbol=%s error=%s",
                event.exchange,
                event.symbol,
                repr(exc),
            )

    async def _flush_history_store(self) -> None:
        if self.history_store is None:
            return
        try:
            await self.history_store.flush()
        except Exception as exc:
            self._storage_errors += 1
            self.logger.exception("Failed to flush liquidation history store | error=%s", repr(exc))

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _nested_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        for key in ("o", "order", "data", "payload"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        return {}

    def _pick(self, payload: dict[str, Any], *keys: str) -> Any:
        nested = self._nested_order(payload)
        for key in keys:
            if key in payload and payload[key] not in (None, ""):
                return payload[key]
        for key in keys:
            if key in nested and nested[key] not in (None, ""):
                return nested[key]
        return None

    def _extract_exchange(self, payload: dict[str, Any]) -> str:
        value = self._pick(payload, "exchange", "ex", "exchangeName", "source_exchange")
        if value is None and len(self.config.exchanges) == 1:
            value = self.config.exchanges[0]
        return str(value or "unknown").strip().lower()

    def _extract_symbol(self, payload: dict[str, Any]) -> str:
        value = self._pick(payload, "symbol", "s", "market", "instrument", "instId", "inst_id", "pair")
        if value is None:
            return ""
        return normalize_symbol(str(value))

    def _extract_side(self, payload: dict[str, Any]) -> LiquidationSide:
        value = self._pick(
            payload,
            "side",
            "S",
            "positionSide",
            "liquidation_side",
            "liquidationSide",
            "direction",
            "posSide",
        )
        return LiquidationSide.from_raw(value)

    def _extract_price(self, payload: dict[str, Any]) -> Decimal:
        value = self._pick(
            payload,
            "price",
            "p",
            "ap",
            "fillPrice",
            "avgPrice",
            "avg_price",
            "px",
            "triggerPx",
        )
        return safe_decimal(value)

    def _extract_quantity(self, payload: dict[str, Any]) -> Decimal:
        value = self._pick(
            payload,
            "quantity",
            "qty",
            "q",
            "size",
            "sz",
            "vol",
            "amount",
            "filledQty",
        )
        return safe_decimal(value)

    def _extract_notional(self, payload: dict[str, Any], *, price: Decimal, quantity: Decimal) -> Decimal:
        value = self._pick(
            payload,
            "notional_usd",
            "notional",
            "value",
            "usd_value",
            "turnover",
            "quoteQty",
        )
        notional = safe_decimal(value)
        if notional > Decimal("0"):
            return notional
        return price * quantity

    def _extract_timestamp(self, payload: dict[str, Any]) -> datetime:
        value = self._pick(
            payload,
            "timestamp",
            "timestamp_ms",
            "ts",
            "T",
            "E",
            "time",
            "createdAt",
            "updatedAt",
            "updated_time",
            "updatedTime",
            "execTime",
        )
        if value is None:
            return utc_now()
        return self._parse_datetime(value)

    def _extract_trade_id(self, payload: dict[str, Any]) -> str | None:
        value = self._pick(payload, "trade_id", "tradeId", "t", "execId", "dealId")
        return str(value) if value not in (None, "") else None

    def _extract_order_id(self, payload: dict[str, Any]) -> str | None:
        value = self._pick(payload, "order_id", "orderId", "i", "ordId")
        return str(value) if value not in (None, "") else None

    def _extract_event_id(self, payload: dict[str, Any]) -> str | None:
        value = self._pick(payload, "event_id", "eventId", "id")
        return str(value) if value not in (None, "") else None

    def _extract_correlation_id(self, payload: dict[str, Any]) -> str | None:
        value = self._pick(payload, "correlation_id", "correlationId", "trace_id", "traceId")
        return str(value) if value not in (None, "") else None

    def _parse_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return ensure_utc(value)
        if isinstance(value, (int, float)):
            # Most exchange timestamps are milliseconds; seconds are also accepted.
            numeric = float(value)
            if numeric > 1_000_000_000_000:
                numeric /= 1000.0
            return datetime.fromtimestamp(numeric, tz=utc_now().tzinfo)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return utc_now()
            if stripped.isdigit():
                return self._parse_datetime(int(stripped))
            return ensure_utc(datetime.fromisoformat(stripped.replace("Z", "+00:00")))
        return utc_now()

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_recent_events(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        side: LiquidationSide | None = None,
        limit: int = 100,
    ) -> list[LiquidationEvent]:
        return self.state.get_recent_events(
            exchange=exchange,
            symbol=symbol,
            side=side,
            limit=max(0, limit),
        )

    def get_recent_large_events(self, *, limit: int = 100) -> list[LiquidationEvent]:
        if limit <= 0:
            return []
        return list(reversed(self._recent_large_events))[:limit]

    def get_symbol_snapshot(self, *, exchange: str, symbol: str):
        return self.state.snapshot_by_symbol(exchange, symbol)

    def get_stats(self) -> dict[str, Any]:
        uptime_seconds = (
            max(0.0, (utc_now() - ensure_utc(self._started_at)).total_seconds())
            if self._started_at
            else 0.0
        )
        return {
            "service_name": self.service_name,
            "input_topic": self.input_topic,
            "running": self._running,
            "registered": self._registered,
            "closed": self._closed,
            "symbols_filter": list(self.config.symbols),
            "exchanges_filter": list(self.config.exchanges),
            "uptime_seconds": uptime_seconds,
            "processed_messages": self._processed_messages,
            "processed_events": self._processed_events,
            "dropped_invalid": self._dropped_invalid,
            "dropped_stale": self._dropped_stale,
            "dropped_duplicates": self._dropped_duplicates,
            "storage_errors": self._storage_errors,
            "published_raw": self._published_raw,
            "published_normalized": self._published_normalized,
            "published_large": self._published_large,
            "published_updated": self._published_updated,
            "published_health": self._published_health,
            "published_snapshots": self._published_snapshots,
            "last_message_at": self._last_message_at.isoformat() if self._last_message_at else None,
            "last_event_at": self._last_event_at.isoformat() if self._last_event_at else None,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at.isoformat() if self._last_error_at else None,
            "tracked_symbols": self.state.symbols_count,
            "state_total_buffered_events": self.state.total_buffered_events,
            "state_total_events_seen": self.state.total_events_seen,
            "large_events_buffered": len(self._recent_large_events),
            "healthcheck_job_id": self._healthcheck_job_id,
            "snapshot_job_id": self._snapshot_job_id,
            "cleanup_job_id": self._cleanup_job_id,
        }

    def get_health(self) -> dict[str, Any]:
        return self.estimate_ingestion_health()

    def estimate_ingestion_health(self) -> dict[str, Any]:
        now = utc_now()
        if not self._running:
            status = "stopped"
        elif self._last_message_at is None:
            status = "starting"
        else:
            age = (now - ensure_utc(self._last_message_at)).total_seconds()
            degraded_after = max(30.0, self.config.stale_event_threshold_seconds * 2.0)
            status = "healthy" if age <= degraded_after else "degraded"

        return {
            "service": self.service_name,
            "status": status,
            "running": self._running,
            "registered": self._registered,
            "input_topic": self.input_topic,
            "last_message_at": self._last_message_at.isoformat() if self._last_message_at else None,
            "last_event_at": self._last_event_at.isoformat() if self._last_event_at else None,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at.isoformat() if self._last_error_at else None,
            "processed_messages": self._processed_messages,
            "processed_events": self._processed_events,
            "dropped_invalid": self._dropped_invalid,
            "dropped_stale": self._dropped_stale,
            "dropped_duplicates": self._dropped_duplicates,
        }

    # ------------------------------------------------------------------
    # Scheduler jobs
    # ------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        if self.scheduler is None:
            return

        job_suffix = self.service_name

        if self._healthcheck_job_id is None:
            self._healthcheck_job_id = self._add_interval_job_once(
                name=f"{self.config.healthcheck_job_name}:{job_suffix}",
                func=self._scheduled_healthcheck,
                interval=self.config.healthcheck_interval_seconds,
            )

        if self._snapshot_job_id is None:
            self._snapshot_job_id = self._add_interval_job_once(
                name=f"{self.config.snapshot_job_name}:{job_suffix}",
                func=self._scheduled_snapshot,
                interval=self.config.snapshot_interval_seconds,
            )

        if self._cleanup_job_id is None:
            self._cleanup_job_id = self._add_interval_job_once(
                name=f"{self.config.cleanup_job_name}:{job_suffix}",
                func=self._scheduled_cleanup,
                interval=self.config.cleanup_interval_seconds,
            )

    def _add_interval_job_once(self, *, name: str, func: Any, interval: float) -> str:
        assert self.scheduler is not None
        existing_job = self.scheduler.get_job_by_name(name)
        if existing_job is not None:
            return existing_job.job_id
        return self.scheduler.add_interval_job(
            name=name,
            func=func,
            interval=interval,
            run_immediately=False,
            max_retries=self.config.scheduler_job_max_retries,
            retry_delay=self.config.scheduler_job_retry_delay_seconds,
            timeout=self.config.scheduler_job_timeout_seconds,
            allow_overlap=False,
            enabled=True,
        )

    async def _remove_scheduler_jobs(self) -> None:
        if self.scheduler is None:
            self._healthcheck_job_id = None
            self._snapshot_job_id = None
            self._cleanup_job_id = None
            return

        for job_id in (self._healthcheck_job_id, self._snapshot_job_id, self._cleanup_job_id):
            if job_id is None:
                continue
            try:
                result = self.scheduler.remove_job(job_id)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self.logger.warning(
                    "Failed to remove LiquidationStream scheduler job | job_id=%s error=%s",
                    job_id,
                    repr(exc),
                )

        self._healthcheck_job_id = None
        self._snapshot_job_id = None
        self._cleanup_job_id = None

    async def _scheduled_healthcheck(self) -> None:
        health = self.estimate_ingestion_health()
        await self.emit_health()
        if health["status"] == "degraded":
            self.logger.warning("LiquidationStream health degraded | health=%s", health)

    async def _scheduled_snapshot(self) -> None:
        await self.emit_runtime_snapshot()
        if self.history_store is not None:
            await self._flush_history_store()

    async def _scheduled_cleanup(self) -> None:
        min_timestamp = utc_now() - timedelta(
            seconds=max(
                self.config.stale_event_threshold_seconds,
                int(self.config.cleanup_interval_seconds),
            )
        )
        removed_events = self.state.prune_before(min_timestamp)
        removed_empty_states = self.state.remove_empty()
        if removed_events or removed_empty_states:
            self.logger.info(
                "LiquidationStream cleanup completed | removed_events=%s removed_empty_states=%s",
                removed_events,
                removed_empty_states,
            )

    # ------------------------------------------------------------------
    # Dedup helpers
    # ------------------------------------------------------------------

    def _make_payload_fingerprint(self, payload: dict[str, Any]) -> str:
        stable_parts = [
            self._extract_exchange(payload),
            self._extract_symbol(payload),
            self._extract_side(payload).value,
            str(self._extract_price(payload)),
            str(self._extract_quantity(payload)),
            str(self._extract_notional(
                payload,
                price=self._extract_price(payload),
                quantity=self._extract_quantity(payload),
            )),
            str(self._extract_timestamp(payload).timestamp()),
            self._extract_trade_id(payload) or "",
            self._extract_order_id(payload) or "",
            self._extract_event_id(payload) or "",
        ]
        return hashlib.sha256("|".join(stable_parts).encode("utf-8")).hexdigest()

    def _is_duplicate_payload_fingerprint(self, fingerprint: str) -> bool:
        return fingerprint in self._recent_payload_fingerprint_set

    def _remember_payload_fingerprint(self, fingerprint: str) -> None:
        if len(self._recent_payload_fingerprints) == self._recent_payload_fingerprints.maxlen:
            oldest = self._recent_payload_fingerprints.popleft()
            self._recent_payload_fingerprint_set.discard(oldest)
        self._recent_payload_fingerprints.append(fingerprint)
        self._recent_payload_fingerprint_set.add(fingerprint)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_connected(self) -> bool:
        # EventBus consumer has no exchange socket connection of its own.
        return self._running and self._registered

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _safe_payload_preview(self, payload: dict[str, Any], max_len: int = 1000) -> str:
        text = repr(payload)
        return text if len(text) <= max_len else text[:max_len] + "...<truncated>"