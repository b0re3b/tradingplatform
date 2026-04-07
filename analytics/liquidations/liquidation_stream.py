from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Protocol

from core.logger import get_logger

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


class EventBusProtocol(Protocol):
    async def emit(self, topic: str, event: Any) -> None:
        ...


class SchedulerProtocol(Protocol):
    def add_interval_job(
        self,
        func: Any,
        seconds: int,
        *,
        name: str | None = None,
        enabled: bool = True,
        run_immediately: bool = False,
        max_retries: int = 0,
        timeout: int | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        ...

    async def run_job_now(self, job_id: str) -> None:
        ...


class LiquidationExchangeAdapterProtocol(Protocol):
    """
    Абстракція адаптера біржі для liquidation stream.

    Очікується, що exchange adapter:
    - уміє стартувати liquidation feed
    - уміє віддавати сирі liquidation payload-и через async iterator / queue / callback
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

    Задачі:
    - отримання сирих liquidation payload-ів від біржі
    - нормалізація до LiquidationEvent
    - фільтрація stale / invalid / duplicate payload-ів
    - збереження у liquidation state
    - оновлення runtime metrics
    - публікація подій в EventBus

    Цей клас НЕ:
    - будує liquidity zones
    - не шукає stop clusters
    - не робить cascade detection
    - не приймає торгові рішення
    """

    DEFAULT_HEALTH_DEGRADATION_SECONDS = 15
    DEFAULT_RECENT_FINGERPRINTS_SIZE = 10_000
    DEFAULT_RECENT_LARGE_EVENTS_SIZE = 500

    def __init__(
        self,
        *,
        event_bus: EventBusProtocol,
        exchange_adapter: LiquidationExchangeAdapterProtocol,
        config: LiquidationStreamConfig,
        state: LiquidationState | None = None,
        metrics: LiquidationMetrics | None = None,
        scheduler: SchedulerProtocol | None = None,
        service_name: str = "liquidation_stream",
    ) -> None:
        self.event_bus = event_bus
        self.exchange_adapter = exchange_adapter
        self.config = config
        self.scheduler = scheduler
        self.service_name = service_name

        self.logger = get_logger(
            __name__,
            service_name=service_name,
            component="analytics.liquidations.stream",
            exchange=getattr(exchange_adapter, "name", "unknown"),
        )

        self.state = state or LiquidationState(
            max_events_per_symbol=self.config.max_buffer_size_per_symbol
        )

        self.metrics = metrics or LiquidationMetrics()

        self._running = False
        self._connected = False
        self._consumer_task: asyncio.Task[None] | None = None
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
        self._published_raw = 0
        self._published_normalized = 0
        self._published_large = 0

        self._recent_payload_fingerprints: deque[str] = deque(
            maxlen=self.DEFAULT_RECENT_FINGERPRINTS_SIZE
        )
        self._recent_payload_fingerprint_set: set[str] = set()

        self._recent_large_events: deque[LiquidationEvent] = deque(
            maxlen=self.DEFAULT_RECENT_LARGE_EVENTS_SIZE
        )

        self._healthcheck_job_id: str | None = None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            self.logger.warning("LiquidationStream already running.")
            return

        self.logger.info(
            "Starting LiquidationStream.",
            extra={
                "exchange": self.exchange_name,
                "symbols": self.config.symbols,
                "enabled": self.config.enabled,
            },
        )

        if not self.config.enabled:
            self.logger.warning("LiquidationStream is disabled by config.")
            return

        self._running = True
        self._started_at = utc_now()
        self._stopped_at = None

        await self._connect()

        self._consumer_task = asyncio.create_task(
            self._consume_loop(),
            name=f"liquidation-stream:{self.exchange_name}",
        )

        self._register_scheduler_jobs()

        self.logger.info(
            "LiquidationStream started.",
            extra={
                "exchange": self.exchange_name,
                "symbols_count": len(self.config.symbols),
            },
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self.logger.info("Stopping LiquidationStream.", extra={"exchange": self.exchange_name})

        self._running = False
        self._stopped_at = utc_now()

        if self._consumer_task:
            self._consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consumer_task
            self._consumer_task = None

        await self._disconnect()

        self.logger.info(
            "LiquidationStream stopped.",
            extra=self.get_stats(),
        )

    async def restart(self) -> None:
        self.logger.info("Restarting LiquidationStream.", extra={"exchange": self.exchange_name})
        await self.stop()
        await self.start()

    # -------------------------------------------------------------------------
    # Connection management
    # -------------------------------------------------------------------------

    async def _connect(self) -> None:
        await self.exchange_adapter.connect_liquidations(self.config.symbols)
        self._connected = True
        self.logger.info(
            "Connected liquidation feed.",
            extra={
                "exchange": self.exchange_name,
                "symbols": self.config.symbols,
            },
        )

    async def _disconnect(self) -> None:
        with contextlib.suppress(Exception):
            await self.exchange_adapter.disconnect_liquidations()
        self._connected = False
        self.logger.info("Disconnected liquidation feed.", extra={"exchange": self.exchange_name})

    async def reconnect(self) -> None:
        self.logger.warning("Manual reconnect requested.", extra={"exchange": self.exchange_name})
        await self._disconnect()
        await self._connect()

    # -------------------------------------------------------------------------
    # Main consumer loop
    # -------------------------------------------------------------------------

    async def _consume_loop(self) -> None:
        self.logger.info("Liquidation consumer loop started.", extra={"exchange": self.exchange_name})

        while self._running:
            try:
                started = time.perf_counter()
                payload = await self.exchange_adapter.recv_liquidation()
                latency_ms = (time.perf_counter() - started) * 1000.0
                self.metrics.observe_latency_ms(latency_ms)

                if payload is None:
                    await asyncio.sleep(0.01)
                    continue

                self._processed_messages += 1
                self._last_message_at = utc_now()

                await self.handle_raw_message(payload)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error_at = utc_now()
                self._last_error = repr(exc)

                self.logger.exception(
                    "Unhandled error in liquidation consume loop.",
                    extra={
                        "exchange": self.exchange_name,
                        "error": repr(exc),
                    },
                )
                await asyncio.sleep(1.0)

        self.logger.info("Liquidation consumer loop exited.", extra={"exchange": self.exchange_name})

    # -------------------------------------------------------------------------
    # Raw payload handling
    # -------------------------------------------------------------------------

    async def handle_raw_message(self, payload: dict[str, Any]) -> LiquidationEvent | None:
        """
        Приймає сирий payload, валідовує, дедуплікує, нормалізує, оновлює state і публікує події.
        """
        fingerprint = self._make_payload_fingerprint(payload)
        if self._is_duplicate_payload_fingerprint(fingerprint):
            self._dropped_duplicates += 1
            self.logger.debug(
                "Duplicate liquidation payload dropped.",
                extra={"exchange": self.exchange_name, "fingerprint": fingerprint},
            )
            return None

        self._remember_payload_fingerprint(fingerprint)

        if self.config.emit_raw_events:
            await self._publish_raw_payload(payload)

        event = self.normalize_event(payload)
        if event is None:
            self._dropped_invalid += 1
            self.metrics.total_invalid_events += 1
            return None

        if not event.is_valid:
            self._dropped_invalid += 1
            self.metrics.observe_event(event, is_valid=False, is_stale=False, is_large=False)
            self.logger.debug(
                "Invalid liquidation event dropped.",
                extra={
                    "exchange": self.exchange_name,
                    "symbol": event.symbol,
                    "side": event.side.value,
                },
            )
            return None

        if is_stale_event(
            event,
            stale_after_seconds=self.config.stale_event_threshold_seconds,
        ):
            self._dropped_stale += 1
            self.metrics.observe_event(
                event,
                is_valid=True,
                is_stale=True,
                is_large=event.notional_usd >= self.config.large_liquidation_threshold_usd,
            )
            self.logger.debug(
                "Stale liquidation event dropped.",
                extra={
                    "exchange": self.exchange_name,
                    "symbol": event.symbol,
                    "event_ts": event.timestamp.isoformat(),
                },
            )
            return None

        symbol_state = self.state.add_event(event)
        self._processed_events += 1
        self._last_event_at = event.timestamp

        is_large = event.notional_usd >= self.config.large_liquidation_threshold_usd
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
            "Liquidation event processed.",
            extra={
                "exchange": event.exchange,
                "symbol": event.symbol,
                "side": event.side.value,
                "price": str(event.price),
                "quantity": str(event.quantity),
                "notional_usd": str(event.notional_usd),
                "buffered_events": len(symbol_state.events),
            },
        )

        return event

    def normalize_event(self, payload: dict[str, Any]) -> LiquidationEvent | None:
        """
        Нормалізація сирого payload у LiquidationEvent.

        Тут закладений максимально tolerant parsing.
        Якщо формат біржі відрізняється, краще виносити це в exchange adapter,
        але цей метод теж має вміти пережити різні payload-структури.
        """
        try:
            exchange = self._extract_exchange(payload)
            symbol = self._extract_symbol(payload)
            side = self._extract_side(payload)
            price = self._extract_price(payload)
            quantity = self._extract_quantity(payload)
            notional_usd = self._extract_notional(payload, price=price, quantity=quantity)
            timestamp = self._extract_timestamp(payload)

            if not exchange or not symbol or price <= 0 or quantity <= 0 or notional_usd <= 0:
                return None

            metadata = {
                "raw_type": payload.get("type"),
                "raw_event": payload.get("event"),
                "source_payload_keys": sorted(payload.keys()),
            }

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
                source="liquidation_stream",
                metadata=metadata,
            )

            return event

        except Exception as exc:
            self.logger.exception(
                "Failed to normalize liquidation payload.",
                extra={
                    "exchange": self.exchange_name,
                    "error": repr(exc),
                    "payload_preview": self._safe_payload_preview(payload),
                },
            )
            return None

    # -------------------------------------------------------------------------
    # Event publishing
    # -------------------------------------------------------------------------

    async def publish_event(self, event: LiquidationEvent) -> None:
        await self.event_bus.emit(self.config.publish_topic_normalized, event)
        self._published_normalized += 1

    async def publish_large_event(self, event: LiquidationEvent) -> None:
        await self.event_bus.emit(self.config.publish_topic_large, event)
        self._published_large += 1

    async def _publish_raw_payload(self, payload: dict[str, Any]) -> None:
        raw_event = {
            "exchange": self.exchange_name,
            "received_at": utc_now().isoformat(),
            "payload": payload,
        }
        await self.event_bus.emit(self.config.publish_topic_raw, raw_event)
        self._published_raw += 1

    # -------------------------------------------------------------------------
    # Extraction helpers
    # -------------------------------------------------------------------------

    def _extract_exchange(self, payload: dict[str, Any]) -> str:
        value = payload.get("exchange") or getattr(self.exchange_adapter, "name", None) or "unknown"
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
        raw_side = (
            payload.get("side")
            or payload.get("S")
            or payload.get("positionSide")
            or payload.get("liquidation_side")
            or payload.get("direction")
        )

        if raw_side is None:
            nested_order = payload.get("o") or payload.get("order") or {}
            raw_side = (
                nested_order.get("side")
                or nested_order.get("S")
                or nested_order.get("positionSide")
            )

        side = str(raw_side).strip().lower() if raw_side is not None else ""

        # Логіка для liquidation semantics:
        # liquidation of longs -> ринок тиснуть вниз
        # liquidation of shorts -> ринок виносить вгору
        if side in {"long", "buy_long", "longs"}:
            return LiquidationSide.LONG
        if side in {"short", "sell_short", "shorts"}:
            return LiquidationSide.SHORT

        # Біржі часто дають order side, а не position side:
        # SELL liquidation order зазвичай означає ліквідацію long
        # BUY liquidation order зазвичай означає ліквідацію short
        if side in {"sell", "ask"}:
            return LiquidationSide.LONG
        if side in {"buy", "bid"}:
            return LiquidationSide.SHORT

        return LiquidationSide.UNKNOWN

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
            # евристика: якщо timestamp дуже великий, це мс
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

    # -------------------------------------------------------------------------
    # Feature methods
    # -------------------------------------------------------------------------

    def get_recent_events(
        self,
        *,
        symbol: str | None = None,
        exchange: str | None = None,
        side: LiquidationSide | None = None,
        limit: int = 100,
    ) -> list[LiquidationEvent]:
        """
        Повертає останні liquidation events із локального state.
        """
        items: list[LiquidationEvent] = []

        target_symbol = normalize_symbol(symbol) if symbol else None
        target_exchange = exchange.lower() if exchange else None

        for (state_exchange, state_symbol), symbol_state in self.state.symbols.items():
            if target_exchange and state_exchange != target_exchange:
                continue
            if target_symbol and state_symbol != target_symbol:
                continue

            for event in reversed(symbol_state.events):
                if side and event.side != side:
                    continue
                items.append(event)
                if len(items) >= limit:
                    return items

        return items

    def get_recent_large_events(
        self,
        *,
        symbol: str | None = None,
        side: LiquidationSide | None = None,
        limit: int = 50,
    ) -> list[LiquidationEvent]:
        """
        Повертає останні великі liquidation events.
        Корисно для dashboard / debug / strategies context.
        """
        target_symbol = normalize_symbol(symbol) if symbol else None
        result: list[LiquidationEvent] = []

        for event in reversed(self._recent_large_events):
            if target_symbol and event.symbol != target_symbol:
                continue
            if side and event.side != side:
                continue
            result.append(event)
            if len(result) >= limit:
                break

        return result

    def get_symbol_pressure_snapshot(self, symbol: str) -> dict[str, Any]:
        """
        Дає швидкий snapshot тиску по символу:
        - скільки long/short liquidations у буфері
        - яка сторона домінує
        - останній event
        """
        normalized_symbol = normalize_symbol(symbol)
        snapshots: list[dict[str, Any]] = []

        for (exchange, state_symbol), symbol_state in self.state.symbols.items():
            if state_symbol != normalized_symbol:
                continue

            dominant_side = "unknown"
            if symbol_state.long_events_count > symbol_state.short_events_count:
                dominant_side = LiquidationSide.LONG.value
            elif symbol_state.short_events_count > symbol_state.long_events_count:
                dominant_side = LiquidationSide.SHORT.value

            snapshots.append(
                {
                    "exchange": exchange,
                    "symbol": state_symbol,
                    "buffered_events": len(symbol_state.events),
                    "long_events_count": symbol_state.long_events_count,
                    "short_events_count": symbol_state.short_events_count,
                    "dominant_side": dominant_side,
                    "last_event_at": symbol_state.last_event_at.isoformat()
                    if symbol_state.last_event_at
                    else None,
                    "last_cascade_at": symbol_state.last_cascade_at.isoformat()
                    if symbol_state.last_cascade_at
                    else None,
                    "cooldown_until": symbol_state.cooldown_until.isoformat()
                    if symbol_state.cooldown_until
                    else None,
                }
            )

        return {
            "symbol": normalized_symbol,
            "exchanges": snapshots,
            "total_exchanges": len(snapshots),
        }

    def get_top_symbols_by_liquidation_flow(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Дає топ символів за кількістю buffered liquidation events.
        Корисно для dashboard / alerting.
        """
        rows: list[dict[str, Any]] = []

        for (exchange, symbol), symbol_state in self.state.symbols.items():
            rows.append(
                {
                    "exchange": exchange,
                    "symbol": symbol,
                    "buffered_events": len(symbol_state.events),
                    "long_events_count": symbol_state.long_events_count,
                    "short_events_count": symbol_state.short_events_count,
                    "last_event_at": symbol_state.last_event_at.isoformat()
                    if symbol_state.last_event_at
                    else None,
                }
            )

        rows.sort(key=lambda row: row["buffered_events"], reverse=True)
        return rows[:limit]

    def estimate_ingestion_health(self) -> dict[str, Any]:
        """
        Оцінка стану ingestion pipeline.
        """
        now = utc_now()
        seconds_since_last_message = (
            (now - self._last_message_at).total_seconds()
            if self._last_message_at
            else None
        )
        seconds_since_last_event = (
            (now - self._last_event_at).total_seconds()
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
            and seconds_since_last_message > self.DEFAULT_HEALTH_DEGRADATION_SECONDS
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
        }

    async def flush_state_for_symbol(self, symbol: str, exchange: str | None = None) -> int:
        """
        Очищає state по конкретному символу.
        Корисно для reset/debug/reload.
        """
        normalized_symbol = normalize_symbol(symbol)
        removed = 0

        keys_to_remove: list[tuple[str, str]] = []
        for (state_exchange, state_symbol), symbol_state in self.state.symbols.items():
            if state_symbol != normalized_symbol:
                continue
            if exchange and state_exchange != exchange.lower():
                continue

            symbol_state.clear()
            keys_to_remove.append((state_exchange, state_symbol))
            removed += 1

        for key in keys_to_remove:
            self.state.remove(*key)

        self.logger.info(
            "Flushed liquidation state for symbol.",
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
        limit: int = 100,
        include_large_topic: bool = False,
    ) -> int:
        """
        Повторно публікує останні liquidation events в EventBus.
        Дуже зручно для:
        - recovery після restart detector-а
        - локального backfill для downstream consumers
        """
        events = self.get_recent_events(symbol=symbol, limit=limit)
        published = 0

        for event in reversed(events):
            await self.publish_event(event)
            published += 1

            if include_large_topic and event.notional_usd >= self.config.large_liquidation_threshold_usd:
                await self.publish_large_event(event)

        self.logger.info(
            "Replayed liquidation events to EventBus.",
            extra={
                "symbol": symbol,
                "limit": limit,
                "published": published,
                "include_large_topic": include_large_topic,
            },
        )
        return published

    async def emit_runtime_snapshot(self, topic: str = "analytics.liquidation.stream.snapshot") -> None:
        """
        Публікує поточний runtime snapshot у EventBus.
        Дуже корисно для dashboard / monitoring / admin tools.
        """
        snapshot = {
            "service": self.service_name,
            "exchange": self.exchange_name,
            "health": self.estimate_ingestion_health(),
            "stats": self.get_stats(),
            "metrics": asdict(self.metrics.snapshot()),
            "state": [asdict(item) for item in self.state.snapshots()],
            "emitted_at": utc_now().isoformat(),
        }
        await self.event_bus.emit(topic, snapshot)

    # -------------------------------------------------------------------------
    # Dedup helpers
    # -------------------------------------------------------------------------

    def _make_payload_fingerprint(self, payload: dict[str, Any]) -> str:
        stable_parts = [
            str(payload.get("symbol") or payload.get("s") or payload.get("instId") or ""),
            str(payload.get("side") or payload.get("S") or ""),
            str(payload.get("price") or payload.get("p") or payload.get("ap") or ""),
            str(payload.get("quantity") or payload.get("qty") or payload.get("q") or ""),
            str(payload.get("timestamp") or payload.get("ts") or payload.get("T") or ""),
            str(payload.get("tradeId") or payload.get("t") or ""),
            str(payload.get("orderId") or payload.get("i") or ""),
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

    # -------------------------------------------------------------------------
    # Scheduler / health
    # -------------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        if self.scheduler is None:
            return

        try:
            job = self.scheduler.add_interval_job(
                self._scheduled_healthcheck,
                seconds=10,
                name=f"liquidation_stream_healthcheck:{self.exchange_name}",
                enabled=True,
                run_immediately=False,
                max_retries=0,
                timeout=5,
                tags=["liquidations", "healthcheck", self.exchange_name],
            )
            self._healthcheck_job_id = getattr(job, "job_id", None)
        except Exception as exc:
            self.logger.exception(
                "Failed to register scheduler jobs for LiquidationStream.",
                extra={"error": repr(exc), "exchange": self.exchange_name},
            )

    async def _scheduled_healthcheck(self) -> None:
        health = self.estimate_ingestion_health()

        if health["status"] == "degraded":
            self.logger.warning(
                "LiquidationStream health degraded.",
                extra=health,
            )

    # -------------------------------------------------------------------------
    # Stats / diagnostics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        uptime_seconds = (
            max(0.0, (utc_now() - self._started_at).total_seconds())
            if self._started_at
            else 0.0
        )

        return {
            "service_name": self.service_name,
            "exchange": self.exchange_name,
            "running": self._running,
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
            "last_message_at": self._last_message_at.isoformat() if self._last_message_at else None,
            "last_event_at": self._last_event_at.isoformat() if self._last_event_at else None,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at.isoformat() if self._last_error_at else None,
            "tracked_symbols": len(self.state.symbols),
            "large_events_buffered": len(self._recent_large_events),
        }

    def get_health(self) -> dict[str, Any]:
        return self.estimate_ingestion_health()

    @property
    def exchange_name(self) -> str:
        return str(getattr(self.exchange_adapter, "name", "unknown")).lower()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _safe_payload_preview(self, payload: dict[str, Any], max_len: int = 1000) -> str:
        text = repr(payload)
        if len(text) <= max_len:
            return text
        return text[:max_len] + "...<truncated>"