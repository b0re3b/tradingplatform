from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Protocol

from core.event_bus import EventBus, EventPriority
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


class LiquidationExchangeAdapterProtocol(Protocol):
    """
    Абстракція біржового адаптера для liquidation stream.

    Adapter відповідає лише за transport/exchange-specific частину:
    - connect liquidation feed;
    - disconnect liquidation feed;
    - отримання raw liquidation payload.
    """

    @property
    def name(self) -> str:
        ...

    async def connect_liquidations(self, symbols: tuple[str, ...]) -> None:
        ...

    async def disconnect_liquidations(self) -> None:
        ...

    async def recv_liquidation(self) -> dict[str, Any] | None:
        ...


class LiquidationStream:
    """
    Ingestion + normalization + publish layer для liquidation events.

    Відповідальність:
    - отримує raw liquidation payload-и від exchange adapter;
    - нормалізує payload у LiquidationEvent;
    - фільтрує invalid/stale/duplicate events;
    - оновлює LiquidationState;
    - оновлює LiquidationMetrics;
    - публікує market.liquidation.* події через core.EventBus;
    - реєструє health/snapshot/cleanup jobs через core.Scheduler.

    Цей клас НЕ:
    - не робить cascade detection;
    - не приймає торгових рішень;
    - не викликає strategy/risk/execution напряму;
    - не дублює core EventBus/Scheduler/logger.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        exchange_adapter: LiquidationExchangeAdapterProtocol,
        config: LiquidationStreamConfig,
        scheduler: Scheduler | None = None,
        state: LiquidationState | None = None,
        metrics: LiquidationMetrics | None = None,
        service_name: str = "liquidation_stream",
    ) -> None:
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.exchange_adapter = exchange_adapter
        self.config = config
        self.service_name = service_name

        self.state = state or LiquidationState(
            max_events_per_symbol=self.config.max_buffer_size_per_symbol,
        )
        self.metrics = metrics or LiquidationMetrics()

        self.logger = get_logger(
            __name__,
            event_type="analytics.liquidations.stream",
            exchange=self.exchange_name,
        )

        self._running = False
        self._registered = False
        self._connected = False

        self._consumer_task: asyncio.Task[None] | None = None

        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None
        self._last_message_at: datetime | None = None
        self._last_event_at: datetime | None = None
        self._last_error_at: datetime | None = None
        self._last_error: str | None = None
        self._last_reconnect_at: datetime | None = None

        self._processed_messages = 0
        self._processed_events = 0
        self._dropped_invalid = 0
        self._dropped_stale = 0
        self._dropped_duplicates = 0

        self._published_raw = 0
        self._published_normalized = 0
        self._published_large = 0
        self._published_health = 0
        self._published_snapshots = 0

        self._recent_payload_fingerprints: deque[str] = deque(
            maxlen=self.config.recent_payload_fingerprints_size,
        )
        self._recent_payload_fingerprint_set: set[str] = set()

        self._recent_large_events: deque[LiquidationEvent] = deque(
            maxlen=self.config.recent_large_events_size,
        )

        self._healthcheck_job_id: str | None = None
        self._snapshot_job_id: str | None = None
        self._cleanup_job_id: str | None = None

    # ---------------------------------------------------------------------
    # Lifecycle / registration
    # ---------------------------------------------------------------------

    def register(self) -> None:
        """
        Реєструє Scheduler jobs.

        LiquidationStream не підписується на EventBus, бо це ingestion source.
        Він тільки публікує market.liquidation.* події.
        """
        if self._registered:
            self.logger.warning("LiquidationStream already registered")
            return

        self._register_scheduler_jobs()
        self._registered = True

        self.logger.info(
            "LiquidationStream registered",
            extra={
                "exchange": self.exchange_name,
                "scheduler_enabled": self.scheduler is not None,
            },
        )

    async def start(self) -> None:
        if self._running:
            self.logger.warning("LiquidationStream already running")
            return

        if not self.config.enabled:
            self.logger.warning(
                "LiquidationStream is disabled by config",
                extra={"exchange": self.exchange_name},
            )
            return

        if not self._registered:
            self.register()

        self._running = True
        self._started_at = utc_now()
        self._stopped_at = None
        self._last_error = None
        self._last_error_at = None

        self.logger.info(
            "Starting LiquidationStream",
            extra={
                "exchange": self.exchange_name,
                "symbols": self.config.symbols,
            },
        )

        try:
            await self._connect()

            self._consumer_task = asyncio.create_task(
                self._consume_loop(),
                name=f"liquidation-stream:{self.exchange_name}",
            )

            self.logger.info(
                "LiquidationStream started",
                extra={
                    "exchange": self.exchange_name,
                    "symbols_count": len(self.config.symbols),
                },
            )

        except Exception as exc:
            self._running = False
            self._last_error = repr(exc)
            self._last_error_at = utc_now()

            self.logger.exception(
                "Failed to start LiquidationStream",
                extra={
                    "exchange": self.exchange_name,
                    "error": repr(exc),
                },
            )
            raise

    async def stop(self) -> None:
        if not self._running:
            return

        self.logger.info(
            "Stopping LiquidationStream",
            extra={"exchange": self.exchange_name},
        )

        self._running = False
        self._stopped_at = utc_now()

        if self._consumer_task is not None:
            self._consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consumer_task
            self._consumer_task = None

        await self._disconnect()

        self.logger.info(
            "LiquidationStream stopped",
            extra=self.get_stats(),
        )

    async def restart(self) -> None:
        self.logger.warning(
            "Restarting LiquidationStream",
            extra={"exchange": self.exchange_name},
        )
        await self.stop()
        await self.start()

    # ---------------------------------------------------------------------
    # Connection management
    # ---------------------------------------------------------------------

    async def _connect(self) -> None:
        await self.exchange_adapter.connect_liquidations(self.config.symbols)
        self._connected = True

        self.logger.info(
            "Connected liquidation feed",
            extra={
                "exchange": self.exchange_name,
                "symbols": self.config.symbols,
            },
        )

    async def _disconnect(self) -> None:
        try:
            await self.exchange_adapter.disconnect_liquidations()
        except Exception as exc:
            self.logger.warning(
                "Error during liquidation feed disconnect",
                extra={
                    "exchange": self.exchange_name,
                    "error": repr(exc),
                },
            )
        finally:
            self._connected = False

        self.logger.info(
            "Disconnected liquidation feed",
            extra={"exchange": self.exchange_name},
        )

    async def reconnect(self) -> None:
        now = utc_now()

        if self._last_reconnect_at is not None:
            elapsed = (now - self._last_reconnect_at).total_seconds()
            if elapsed < self.config.reconnect_cooldown_seconds:
                self.logger.warning(
                    "Reconnect skipped due to cooldown",
                    extra={
                        "exchange": self.exchange_name,
                        "elapsed_seconds": elapsed,
                        "cooldown_seconds": self.config.reconnect_cooldown_seconds,
                    },
                )
                return

        self._last_reconnect_at = now

        self.logger.warning(
            "Reconnecting liquidation feed",
            extra={"exchange": self.exchange_name},
        )

        await self._disconnect()
        await self._connect()

    # ---------------------------------------------------------------------
    # Main consumer loop
    # ---------------------------------------------------------------------

    async def _consume_loop(self) -> None:
        self.logger.info(
            "Liquidation consumer loop started",
            extra={"exchange": self.exchange_name},
        )

        while self._running:
            try:
                started = time.perf_counter()
                payload = await self.exchange_adapter.recv_liquidation()
                latency_ms = (time.perf_counter() - started) * 1000.0
                self.metrics.observe_latency_ms(latency_ms)

                if payload is None:
                    await asyncio.sleep(self.config.consumer_idle_sleep_seconds)
                    continue

                self._processed_messages += 1
                self._last_message_at = utc_now()

                await self.handle_raw_message(payload)

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                self._last_error = repr(exc)
                self._last_error_at = utc_now()

                self.logger.exception(
                    "Unhandled error in liquidation consume loop",
                    extra={
                        "exchange": self.exchange_name,
                        "error": repr(exc),
                    },
                )

                await asyncio.sleep(self.config.consumer_error_sleep_seconds)

        self.logger.info(
            "Liquidation consumer loop exited",
            extra={"exchange": self.exchange_name},
        )

    # ---------------------------------------------------------------------
    # Raw payload handling
    # ---------------------------------------------------------------------

    async def handle_raw_message(self, payload: dict[str, Any]) -> LiquidationEvent | None:
        """
        Приймає raw payload, дедуплікує, нормалізує, оновлює state/metrics
        і публікує EventBus події.
        """
        fingerprint = self._make_payload_fingerprint(payload)

        if self.config.deduplication_enabled:
            if self._is_duplicate_payload_fingerprint(fingerprint):
                self._dropped_duplicates += 1
                self.logger.debug(
                    "Duplicate liquidation payload dropped",
                    extra={
                        "exchange": self.exchange_name,
                        "fingerprint": fingerprint,
                    },
                )
                return None

            self._remember_payload_fingerprint(fingerprint)

        if self.config.emit_raw_events:
            await self._publish_raw_payload(payload, fingerprint=fingerprint)

        event = self.normalize_event(payload, raw_payload_hash=fingerprint)
        if event is None:
            self._dropped_invalid += 1
            self.metrics.observe_invalid_event(exchange=self.exchange_name)
            return None

        if not event.is_valid:
            self._dropped_invalid += 1
            self.metrics.observe_event(
                event,
                is_valid=False,
                is_stale=False,
                is_large=False,
            )
            self.logger.debug(
                "Invalid liquidation event dropped",
                extra={
                    "exchange": event.exchange,
                    "symbol": event.symbol,
                    "side": event.side.value,
                },
            )
            return None

        is_large = event.is_large_at(self.config.large_liquidation_threshold_usd)

        if is_stale_event(
            event,
            stale_after_seconds=self.config.stale_event_threshold_seconds,
        ):
            self._dropped_stale += 1
            self.metrics.observe_event(
                event,
                is_valid=True,
                is_stale=True,
                is_large=is_large,
            )

            self.logger.debug(
                "Stale liquidation event dropped",
                extra={
                    "exchange": event.exchange,
                    "symbol": event.symbol,
                    "event_ts": event.timestamp.isoformat(),
                },
            )
            return None

        symbol_state = self.state.add_event(event)

        self._processed_events += 1
        self._last_event_at = ensure_utc(event.timestamp)

        self.metrics.observe_event(
            event,
            is_valid=True,
            is_stale=False,
            is_large=is_large,
        )

        await self.publish_event(event)

        if is_large and self.config.emit_large_events:
            self._recent_large_events.append(event)
            await self.publish_large_event(event)

        self.logger.debug(
            "Liquidation event processed",
            extra={
                "exchange": event.exchange,
                "symbol": event.symbol,
                "side": event.side.value,
                "price": str(event.price),
                "quantity": str(event.quantity),
                "notional_usd": str(event.notional_usd),
                "buffered_events": symbol_state.total_buffered_events,
            },
        )

        return event

    def normalize_event(
        self,
        payload: dict[str, Any],
        *,
        raw_payload_hash: str | None = None,
    ) -> LiquidationEvent | None:
        """
        Нормалізує raw exchange payload у LiquidationEvent.

        Exchange-specific parsing бажано поступово переносити в exchange adapter,
        але stream лишається tolerant до різних payload-структур.
        """
        try:
            exchange = self._extract_exchange(payload)
            symbol = self._extract_symbol(payload)
            side = self._extract_side(payload)
            price = self._extract_price(payload)
            quantity = self._extract_quantity(payload)
            notional_usd = self._extract_notional(
                payload,
                price=price,
                quantity=quantity,
            )
            timestamp = self._extract_timestamp(payload)

            if not exchange or not symbol or price <= 0 or quantity <= 0 or notional_usd <= 0:
                return None

            event = LiquidationEvent(
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
                    "raw_type": payload.get("type"),
                    "raw_event": payload.get("event"),
                    "source_payload_keys": sorted(payload.keys()),
                    "exchange_adapter": self.exchange_name,
                },
            )

            return event

        except Exception as exc:
            self._last_error = repr(exc)
            self._last_error_at = utc_now()

            self.logger.exception(
                "Failed to normalize liquidation payload",
                extra={
                    "exchange": self.exchange_name,
                    "error": repr(exc),
                    "payload_preview": self._safe_payload_preview(payload),
                },
            )
            return None

    # ---------------------------------------------------------------------
    # Event publishing
    # ---------------------------------------------------------------------

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

    async def _publish_raw_payload(
        self,
        payload: dict[str, Any],
        *,
        fingerprint: str,
    ) -> bool:
        raw_event = {
            "exchange": self.exchange_name,
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
                "exchange": self.exchange_name,
                "event_type": LiquidationEventType.RAW.value,
            },
        )

        if accepted:
            self._published_raw += 1

        return accepted

    async def emit_runtime_snapshot(
        self,
        topic: str | None = None,
    ) -> bool:
        snapshot = {
            "service": self.service_name,
            "exchange": self.exchange_name,
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
            headers={
                "exchange": self.exchange_name,
                "event_type": LiquidationEventType.SNAPSHOT.value,
            },
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
            headers={
                "exchange": self.exchange_name,
                "event_type": LiquidationEventType.HEALTH.value,
            },
        )

        if accepted:
            self._published_health += 1

        return accepted

    # ---------------------------------------------------------------------
    # Extraction helpers
    # ---------------------------------------------------------------------

    def _extract_exchange(self, payload: dict[str, Any]) -> str:
        value = (
            payload.get("exchange")
            or payload.get("ex")
            or getattr(self.exchange_adapter, "name", None)
            or "unknown"
        )
        return str(value).strip().lower()

    def _extract_symbol(self, payload: dict[str, Any]) -> str:
        raw_symbol = (
            payload.get("symbol")
            or payload.get("s")
            or payload.get("market")
            or payload.get("instrument")
            or payload.get("instId")
            or payload.get("pair")
        )

        if raw_symbol is None:
            return ""

        return normalize_symbol(str(raw_symbol))

    def _extract_side(self, payload: dict[str, Any]) -> LiquidationSide:
        nested_order = payload.get("o") or payload.get("order") or {}

        raw_side = (
            payload.get("side")
            or payload.get("S")
            or payload.get("positionSide")
            or payload.get("liquidation_side")
            or payload.get("direction")
            or nested_order.get("side")
            or nested_order.get("S")
            or nested_order.get("positionSide")
        )

        return LiquidationSide.from_raw(raw_side)

    def _extract_price(self, payload: dict[str, Any]) -> Decimal:
        nested_order = payload.get("o") or payload.get("order") or {}

        value = (
            payload.get("price")
            or payload.get("p")
            or payload.get("ap")
            or payload.get("fillPrice")
            or payload.get("avgPrice")
            or nested_order.get("price")
            or nested_order.get("p")
            or nested_order.get("ap")
        )
        return safe_decimal(value)

    def _extract_quantity(self, payload: dict[str, Any]) -> Decimal:
        nested_order = payload.get("o") or payload.get("order") or {}

        value = (
            payload.get("quantity")
            or payload.get("qty")
            or payload.get("q")
            or payload.get("size")
            or payload.get("amount")
            or nested_order.get("quantity")
            or nested_order.get("qty")
            or nested_order.get("q")
            or nested_order.get("size")
        )
        return safe_decimal(value)

    def _extract_notional(
        self,
        payload: dict[str, Any],
        *,
        price: Decimal,
        quantity: Decimal,
    ) -> Decimal:
        nested_order = payload.get("o") or payload.get("order") or {}

        value = (
            payload.get("notional_usd")
            or payload.get("notional")
            or payload.get("usdValue")
            or payload.get("value")
            or nested_order.get("notional")
            or nested_order.get("value")
        )

        notional = safe_decimal(value)
        if notional > 0:
            return notional

        if price > 0 and quantity > 0:
            return price * quantity

        return Decimal("0")

    def _extract_timestamp(self, payload: dict[str, Any]) -> datetime:
        nested_order = payload.get("o") or payload.get("order") or {}

        raw_ts = (
            payload.get("timestamp")
            or payload.get("ts")
            or payload.get("T")
            or payload.get("time")
            or payload.get("updatedAt")
            or nested_order.get("timestamp")
            or nested_order.get("ts")
            or nested_order.get("T")
            or nested_order.get("time")
        )

        if raw_ts is None:
            return utc_now()

        if isinstance(raw_ts, datetime):
            return ensure_utc(raw_ts)

        if isinstance(raw_ts, (int, float)):
            if raw_ts > 10_000_000_000:
                return datetime.fromtimestamp(raw_ts / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(raw_ts, tz=timezone.utc)

        raw_ts_str = str(raw_ts).strip()

        if raw_ts_str.isdigit():
            ts_int = int(raw_ts_str)
            if ts_int > 10_000_000_000:
                return datetime.fromtimestamp(ts_int / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(ts_int, tz=timezone.utc)

        try:
            return ensure_utc(datetime.fromisoformat(raw_ts_str.replace("Z", "+00:00")))
        except Exception:
            return utc_now()

    def _extract_trade_id(self, payload: dict[str, Any]) -> str | None:
        nested_order = payload.get("o") or payload.get("order") or {}

        value = (
            payload.get("trade_id")
            or payload.get("tradeId")
            or payload.get("t")
            or nested_order.get("trade_id")
            or nested_order.get("tradeId")
            or nested_order.get("t")
        )
        return str(value) if value is not None else None

    def _extract_order_id(self, payload: dict[str, Any]) -> str | None:
        nested_order = payload.get("o") or payload.get("order") or {}

        value = (
            payload.get("order_id")
            or payload.get("orderId")
            or payload.get("i")
            or nested_order.get("order_id")
            or nested_order.get("orderId")
            or nested_order.get("i")
        )
        return str(value) if value is not None else None

    def _extract_event_id(self, payload: dict[str, Any]) -> str | None:
        value = (
            payload.get("event_id")
            or payload.get("eventId")
            or payload.get("e")
            or payload.get("id")
        )
        return str(value) if value is not None else None

    def _extract_correlation_id(self, payload: dict[str, Any]) -> str | None:
        value = (
            payload.get("correlation_id")
            or payload.get("correlationId")
            or payload.get("trace_id")
            or payload.get("traceId")
        )
        return str(value) if value is not None else None

    # ---------------------------------------------------------------------
    # Feature / query methods
    # ---------------------------------------------------------------------

    def get_recent_events(
        self,
        *,
        symbol: str | None = None,
        exchange: str | None = None,
        side: LiquidationSide | None = None,
        limit: int = 100,
    ) -> list[LiquidationEvent]:
        return self.state.get_recent_events(
            exchange=exchange,
            symbol=symbol,
            side=side,
            limit=limit,
        )

    def get_recent_large_events(
        self,
        *,
        symbol: str | None = None,
        side: LiquidationSide | None = None,
        limit: int = 50,
    ) -> list[LiquidationEvent]:
        if limit <= 0:
            return []

        target_symbol = normalize_symbol(symbol) if symbol else None
        result: list[LiquidationEvent] = []

        for event in reversed(self._recent_large_events):
            if target_symbol and event.symbol != target_symbol:
                continue
            if side is not None and event.side is not side:
                continue

            result.append(event)

            if len(result) >= limit:
                break

        return result

    def get_symbol_pressure_snapshot(self, symbol: str) -> dict[str, Any]:
        normalized_symbol = normalize_symbol(symbol)

        snapshots: list[dict[str, Any]] = []
        for (_, state_symbol), symbol_state in self.state.symbols.items():
            if state_symbol != normalized_symbol:
                continue

            snapshot = symbol_state.snapshot()
            snapshots.append(snapshot.to_dict(serialize=True))

        return {
            "symbol": normalized_symbol,
            "exchanges": snapshots,
            "total_exchanges": len(snapshots),
        }

    def get_top_symbols_by_liquidation_flow(self, limit: int = 10) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for symbol_state in self.state.symbols.values():
            rows.append(
                {
                    "exchange": symbol_state.exchange,
                    "symbol": symbol_state.symbol,
                    "buffered_events": symbol_state.total_buffered_events,
                    "long_events_count": symbol_state.long_events_count,
                    "short_events_count": symbol_state.short_events_count,
                    "last_event_at": (
                        symbol_state.last_event_at.isoformat()
                        if symbol_state.last_event_at
                        else None
                    ),
                }
            )

        rows.sort(key=lambda row: row["buffered_events"], reverse=True)
        return rows[: max(0, limit)]

    async def flush_state_for_symbol(
        self,
        symbol: str,
        exchange: str | None = None,
    ) -> int:
        normalized_symbol = normalize_symbol(symbol)
        removed = 0

        keys_to_remove: list[tuple[str, str]] = []

        for (state_exchange, state_symbol), symbol_state in self.state.symbols.items():
            if state_symbol != normalized_symbol:
                continue

            if exchange is not None and state_exchange != exchange.strip().lower():
                continue

            symbol_state.clear()
            keys_to_remove.append((state_exchange, state_symbol))
            removed += 1

        for key in keys_to_remove:
            self.state.remove(*key)

        self.logger.info(
            "Flushed liquidation state for symbol",
            extra={
                "symbol": normalized_symbol,
                "exchange": exchange,
                "removed_states": removed,
            },
        )

        return removed

    async def replay_events_to_bus(
        self,
        *,
        symbol: str | None = None,
        exchange: str | None = None,
        limit: int = 100,
        include_large_topic: bool = False,
    ) -> int:
        events = self.get_recent_events(
            symbol=symbol,
            exchange=exchange,
            limit=limit,
        )

        published = 0

        for event in reversed(events):
            if await self.publish_event(event):
                published += 1

            if (
                include_large_topic
                and event.is_large_at(self.config.large_liquidation_threshold_usd)
            ):
                await self.publish_large_event(event)

        self.logger.info(
            "Replayed liquidation events to EventBus",
            extra={
                "symbol": symbol,
                "exchange": exchange,
                "limit": limit,
                "published": published,
                "include_large_topic": include_large_topic,
            },
        )

        return published

    # ---------------------------------------------------------------------
    # Health / stats
    # ---------------------------------------------------------------------

    def estimate_ingestion_health(self) -> dict[str, Any]:
        now = utc_now()

        seconds_since_last_message = (
            (now - ensure_utc(self._last_message_at)).total_seconds()
            if self._last_message_at
            else None
        )
        seconds_since_last_event = (
            (now - ensure_utc(self._last_event_at)).total_seconds()
            if self._last_event_at
            else None
        )

        status = "healthy"

        if not self._running:
            status = "stopped"
        elif not self._connected:
            status = "disconnected"
        elif (
            seconds_since_last_message is not None
            and seconds_since_last_message > self.config.healthcheck_interval_seconds
        ):
            status = "degraded"

        return {
            "status": status,
            "running": self._running,
            "connected": self._connected,
            "exchange": self.exchange_name,
            "seconds_since_last_message": seconds_since_last_message,
            "seconds_since_last_event": seconds_since_last_event,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at.isoformat() if self._last_error_at else None,
            "last_reconnect_at": (
                self._last_reconnect_at.isoformat()
                if self._last_reconnect_at
                else None
            ),
        }

    def get_stats(self) -> dict[str, Any]:
        uptime_seconds = (
            max(0.0, (utc_now() - ensure_utc(self._started_at)).total_seconds())
            if self._started_at
            else 0.0
        )

        return {
            "service_name": self.service_name,
            "exchange": self.exchange_name,
            "running": self._running,
            "registered": self._registered,
            "connected": self._connected,
            "symbols": list(self.config.symbols),
            "uptime_seconds": uptime_seconds,
            "processed_messages": self._processed_messages,
            "processed_events": self._processed_events,
            "dropped_invalid": self._dropped_invalid,
            "dropped_stale": self._dropped_stale,
            "dropped_duplicates": self._dropped_duplicates,
            "published_raw": self._published_raw,
            "published_normalized": self._published_normalized,
            "published_large": self._published_large,
            "published_health": self._published_health,
            "published_snapshots": self._published_snapshots,
            "last_message_at": (
                self._last_message_at.isoformat()
                if self._last_message_at
                else None
            ),
            "last_event_at": (
                self._last_event_at.isoformat()
                if self._last_event_at
                else None
            ),
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

    # ---------------------------------------------------------------------
    # Scheduler jobs
    # ---------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        if self.scheduler is None:
            return

        if self._healthcheck_job_id is None:
            self._healthcheck_job_id = self.scheduler.add_interval_job(
                name=f"{self.config.healthcheck_job_name}:{self.exchange_name}",
                func=self._scheduled_healthcheck,
                interval=self.config.healthcheck_interval_seconds,
                run_immediately=False,
                max_retries=self.config.scheduler_job_max_retries,
                retry_delay=self.config.scheduler_job_retry_delay_seconds,
                timeout=self.config.scheduler_job_timeout_seconds,
                allow_overlap=False,
                enabled=True,
            )

        if self._snapshot_job_id is None:
            self._snapshot_job_id = self.scheduler.add_interval_job(
                name=f"{self.config.snapshot_job_name}:{self.exchange_name}",
                func=self._scheduled_snapshot,
                interval=self.config.snapshot_interval_seconds,
                run_immediately=False,
                max_retries=self.config.scheduler_job_max_retries,
                retry_delay=self.config.scheduler_job_retry_delay_seconds,
                timeout=self.config.scheduler_job_timeout_seconds,
                allow_overlap=False,
                enabled=True,
            )

        if self._cleanup_job_id is None:
            self._cleanup_job_id = self.scheduler.add_interval_job(
                name=f"{self.config.cleanup_job_name}:{self.exchange_name}",
                func=self._scheduled_cleanup,
                interval=self.config.cleanup_interval_seconds,
                run_immediately=False,
                max_retries=self.config.scheduler_job_max_retries,
                retry_delay=self.config.scheduler_job_retry_delay_seconds,
                timeout=self.config.scheduler_job_timeout_seconds,
                allow_overlap=False,
                enabled=True,
            )

    async def _scheduled_healthcheck(self) -> None:
        health = self.estimate_ingestion_health()
        await self.emit_health()

        if health["status"] == "degraded":
            self.logger.warning(
                "LiquidationStream health degraded",
                extra=health,
            )

            if self.config.reconnect_on_health_degraded and self._running:
                await self.reconnect()

        elif health["status"] in {"disconnected", "stopped"}:
            self.logger.warning(
                "LiquidationStream health warning",
                extra=health,
            )

    async def _scheduled_snapshot(self) -> None:
        await self.emit_runtime_snapshot()

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
                "LiquidationStream cleanup completed",
                extra={
                    "exchange": self.exchange_name,
                    "removed_events": removed_events,
                    "removed_empty_states": removed_empty_states,
                },
            )

    # ---------------------------------------------------------------------
    # Dedup helpers
    # ---------------------------------------------------------------------

    def _make_payload_fingerprint(self, payload: dict[str, Any]) -> str:
        nested_order = payload.get("o") or payload.get("order") or {}

        stable_parts = [
            str(payload.get("exchange") or self.exchange_name),
            str(payload.get("symbol") or payload.get("s") or payload.get("instId") or ""),
            str(payload.get("side") or payload.get("S") or nested_order.get("side") or ""),
            str(payload.get("price") or payload.get("p") or payload.get("ap") or ""),
            str(payload.get("quantity") or payload.get("qty") or payload.get("q") or ""),
            str(payload.get("timestamp") or payload.get("ts") or payload.get("T") or ""),
            str(payload.get("tradeId") or payload.get("t") or nested_order.get("tradeId") or ""),
            str(payload.get("orderId") or payload.get("i") or nested_order.get("orderId") or ""),
        ]

        raw = "|".join(stable_parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _is_duplicate_payload_fingerprint(self, fingerprint: str) -> bool:
        return fingerprint in self._recent_payload_fingerprint_set

    def _remember_payload_fingerprint(self, fingerprint: str) -> None:
        if len(self._recent_payload_fingerprints) == self._recent_payload_fingerprints.maxlen:
            oldest = self._recent_payload_fingerprints.popleft()
            self._recent_payload_fingerprint_set.discard(oldest)

        self._recent_payload_fingerprints.append(fingerprint)
        self._recent_payload_fingerprint_set.add(fingerprint)

    # ---------------------------------------------------------------------
    # Properties / internals
    # ---------------------------------------------------------------------

    @property
    def exchange_name(self) -> str:
        return str(getattr(self.exchange_adapter, "name", "unknown")).strip().lower()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _safe_payload_preview(self, payload: dict[str, Any], max_len: int = 1000) -> str:
        text = repr(payload)
        if len(text) <= max_len:
            return text
        return text[:max_len] + "...<truncated>"