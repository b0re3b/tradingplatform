from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from core.config import Config
from core.logger import get_logger


@dataclass(slots=True)
class OpenInterestRecord:
    exchange: str
    symbol: str
    market_type: str

    timestamp_ms: int
    received_at_ms: int

    open_interest: float
    open_interest_value: float | None = None
    mark_price: float | None = None


@dataclass(slots=True)
class OpenInterestState:
    exchange: str
    symbol: str
    market_type: str = "perpetual"

    history: list[OpenInterestRecord] = field(default_factory=list)

    last_timestamp_ms: int | None = None
    last_received_at_ms: int | None = None
    last_error: str | None = None

    total_updates: int = 0
    invalid_events: int = 0
    duplicate_events: int = 0
    trims_count: int = 0


class OpenInterestCache:
    """
    Локальний кеш open interest data.

    Відповідальність:
    - зберігати останні open interest records
    - підтримувати bounded history
    - віддавати latest / history / window stats
    - чистити застарілі записи
    """

    def __init__(
        self,
        *,
        config: Config,
        event_bus: Any | None = None,
        max_records_per_key: int = 5000,
        retention_ms: int = 30 * 24 * 60 * 60 * 1000,
        service_name: str = "open_interest_cache",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.max_records_per_key = max_records_per_key
        self.retention_ms = retention_ms
        self._service_name = service_name

        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="open_interest_cache",
        )

        self._states: dict[str, OpenInterestState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

        self._metrics: dict[str, int | float] = {
            "states_created": 0,
            "events_received": 0,
            "records_stored": 0,
            "invalid_events": 0,
            "duplicate_events": 0,
            "trimmed_records": 0,
            "cleanup_runs": 0,
            "cleanup_removed": 0,
            "last_cleanup_at": 0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(self, event: dict[str, Any]) -> None:
        """
        Очікує event формату:
        {
            "exchange": "...",
            "symbol": "...",
            "market_type": "...",
            "timestamp_ms": ...,
            "received_at_ms": ...,
            "open_interest": ...,
            # optional:
            "open_interest_value": ...,
            "mark_price": ...,
        }
        """
        state_key = self._build_state_key_from_event(event)
        lock = self._get_lock(state_key)

        async with lock:
            self._metrics["events_received"] += 1

            record = self._normalize_record(event)
            if record is None:
                self._metrics["invalid_events"] += 1

                state = self._states.get(state_key)
                if state is not None:
                    state.invalid_events += 1
                    state.last_error = "invalid_open_interest_event"

                await self._emit_event(
                    "system.open_interest_cache.invalid_open_interest",
                    {
                        "exchange": event.get("exchange"),
                        "symbol": event.get("symbol"),
                        "market_type": event.get("market_type", "perpetual"),
                    },
                )
                return

            state = self._get_or_create_state(record)

            if self._is_duplicate(state, record):
                state.duplicate_events += 1
                self._metrics["duplicate_events"] += 1

                self._logger.warning(
                    "Duplicate open interest event detected | exchange=%s symbol=%s timestamp_ms=%s",
                    record.exchange,
                    record.symbol,
                    record.timestamp_ms,
                )
                return

            state.history.append(record)
            state.last_timestamp_ms = record.timestamp_ms
            state.last_received_at_ms = record.received_at_ms
            state.last_error = None
            state.total_updates += 1

            self._metrics["records_stored"] += 1

            removed = self._trim_state(state)
            if removed > 0:
                state.trims_count += 1
                self._metrics["trimmed_records"] += removed

    async def get_latest(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
    ) -> dict[str, Any] | None:
        state_key = self._build_state_key(exchange, symbol, market_type)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._states.get(state_key)
            if state is None or not state.history:
                return None

            return self._serialize_record(state.history[-1])

    async def get_history(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        state_key = self._build_state_key(exchange, symbol, market_type)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._states.get(state_key)
            if state is None or limit <= 0:
                return []

            return [
                self._serialize_record(record)
                for record in state.history[-limit:]
            ]

    async def get_since(
        self,
        *,
        exchange: str,
        symbol: str,
        since_timestamp_ms: int,
        market_type: str = "perpetual",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        state_key = self._build_state_key(exchange, symbol, market_type)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._states.get(state_key)
            if state is None:
                return []

            filtered = [
                self._serialize_record(record)
                for record in state.history
                if record.timestamp_ms >= since_timestamp_ms
            ]

            if limit is not None and limit > 0:
                return filtered[-limit:]
            return filtered

    async def get_window_stats(
        self,
        *,
        exchange: str,
        symbol: str,
        window_ms: int,
        market_type: str = "perpetual",
    ) -> dict[str, Any] | None:
        if window_ms <= 0:
            raise ValueError("window_ms must be > 0")

        state_key = self._build_state_key(exchange, symbol, market_type)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._states.get(state_key)
            if state is None or not state.history:
                return None

            cutoff = self._now_ms() - window_ms
            records = [r for r in state.history if r.timestamp_ms >= cutoff]

            if not records:
                return {
                    "exchange": exchange,
                    "symbol": symbol,
                    "market_type": market_type,
                    "window_ms": window_ms,
                    "records_count": 0,
                    "latest_open_interest": None,
                    "min_open_interest": None,
                    "max_open_interest": None,
                    "avg_open_interest": None,
                    "delta_open_interest": None,
                    "latest_mark_price": None,
                    "latest_open_interest_value": None,
                }

            oi_values = [r.open_interest for r in records]
            first_oi = records[0].open_interest
            last_oi = records[-1].open_interest

            return {
                "exchange": exchange,
                "symbol": symbol,
                "market_type": market_type,
                "window_ms": window_ms,
                "records_count": len(records),
                "latest_open_interest": last_oi,
                "min_open_interest": min(oi_values),
                "max_open_interest": max(oi_values),
                "avg_open_interest": sum(oi_values) / len(oi_values),
                "delta_open_interest": last_oi - first_oi,
                "delta_open_interest_pct": ((last_oi - first_oi) / first_oi * 100.0) if first_oi > 0 else None,
                "latest_mark_price": records[-1].mark_price,
                "latest_open_interest_value": records[-1].open_interest_value,
                "first_timestamp_ms": records[0].timestamp_ms,
                "last_timestamp_ms": records[-1].timestamp_ms,
            }

    async def clear_symbol(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
        reason: str = "manual_clear",
    ) -> None:
        state_key = self._build_state_key(exchange, symbol, market_type)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._states.get(state_key)
            if state is None:
                return

            removed = len(state.history)
            state.history.clear()
            state.last_timestamp_ms = None
            state.last_received_at_ms = None
            state.last_error = reason

            self._logger.warning(
                "Open interest history cleared | exchange=%s symbol=%s removed=%s reason=%s",
                exchange,
                symbol,
                removed,
                reason,
            )

            await self._emit_event(
                "system.open_interest_cache.cleared",
                {
                    "exchange": exchange,
                    "symbol": symbol,
                    "market_type": market_type,
                    "removed": removed,
                    "reason": reason,
                },
            )

    async def cleanup_stale(self) -> int:
        total_removed = 0
        cutoff = self._now_ms() - self.retention_ms

        for state_key, state in list(self._states.items()):
            lock = self._get_lock(state_key)

            async with lock:
                original_len = len(state.history)
                if original_len == 0:
                    continue

                state.history = [r for r in state.history if r.timestamp_ms >= cutoff]
                removed_here = original_len - len(state.history)

                if removed_here > 0:
                    total_removed += removed_here
                    state.trims_count += 1

        self._metrics["cleanup_runs"] += 1
        self._metrics["cleanup_removed"] += total_removed
        self._metrics["last_cleanup_at"] = time.time()

        if total_removed > 0:
            self._logger.info(
                "Open interest cleanup completed | removed=%s retention_ms=%s",
                total_removed,
                self.retention_ms,
            )

        return total_removed

    async def has_data(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
    ) -> bool:
        state_key = self._build_state_key(exchange, symbol, market_type)
        state = self._states.get(state_key)
        return state is not None and len(state.history) > 0

    async def size(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
    ) -> int:
        state_key = self._build_state_key(exchange, symbol, market_type)
        state = self._states.get(state_key)
        if state is None:
            return 0
        return len(state.history)

    def stats(self) -> dict[str, Any]:
        active_states = sum(1 for state in self._states.values() if state.history)

        return {
            "states_total": len(self._states),
            "states_with_data": active_states,
            "states_created": self._metrics["states_created"],
            "events_received": self._metrics["events_received"],
            "records_stored": self._metrics["records_stored"],
            "invalid_events": self._metrics["invalid_events"],
            "duplicate_events": self._metrics["duplicate_events"],
            "trimmed_records": self._metrics["trimmed_records"],
            "cleanup_runs": self._metrics["cleanup_runs"],
            "cleanup_removed": self._metrics["cleanup_removed"],
            "last_cleanup_at": self._metrics["last_cleanup_at"],
            "max_records_per_key": self.max_records_per_key,
            "retention_ms": self.retention_ms,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create_state(self, record: OpenInterestRecord) -> OpenInterestState:
        state_key = self._build_state_key(
            record.exchange,
            record.symbol,
            record.market_type,
        )

        state = self._states.get(state_key)
        if state is not None:
            return state

        state = OpenInterestState(
            exchange=record.exchange,
            symbol=record.symbol,
            market_type=record.market_type,
        )
        self._states[state_key] = state
        self._metrics["states_created"] += 1

        self._logger.info(
            "Open interest state created | exchange=%s symbol=%s market_type=%s",
            record.exchange,
            record.symbol,
            record.market_type,
        )

        return state

    def _get_lock(self, state_key: str) -> asyncio.Lock:
        lock = self._locks.get(state_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[state_key] = lock
        return lock

    def _normalize_record(self, event: dict[str, Any]) -> OpenInterestRecord | None:
        exchange = event.get("exchange")
        symbol = event.get("symbol")
        market_type = event.get("market_type", "perpetual")

        timestamp_ms = self._safe_int(event.get("timestamp_ms"))
        received_at_ms = self._safe_int(event.get("received_at_ms")) or self._now_ms()
        open_interest = self._safe_float(event.get("open_interest"))

        if exchange is None or symbol is None:
            return None
        if timestamp_ms is None or open_interest is None:
            return None
        if open_interest < 0:
            return None

        return OpenInterestRecord(
            exchange=str(exchange),
            symbol=str(symbol),
            market_type=str(market_type),
            timestamp_ms=timestamp_ms,
            received_at_ms=received_at_ms,
            open_interest=open_interest,
            open_interest_value=self._safe_float(event.get("open_interest_value")),
            mark_price=self._safe_float(event.get("mark_price")),
        )

    def _is_duplicate(self, state: OpenInterestState, record: OpenInterestRecord) -> bool:
        if not state.history:
            return False

        last = state.history[-1]
        return (
            last.timestamp_ms == record.timestamp_ms
            and last.open_interest == record.open_interest
            and last.open_interest_value == record.open_interest_value
            and last.mark_price == record.mark_price
        )

    def _trim_state(self, state: OpenInterestState) -> int:
        removed = 0
        cutoff = self._now_ms() - self.retention_ms

        while len(state.history) > self.max_records_per_key:
            state.history.pop(0)
            removed += 1

        while state.history and state.history[0].timestamp_ms < cutoff:
            state.history.pop(0)
            removed += 1

        return removed

    async def _emit_event(self, topic: str, payload: dict[str, Any]) -> None:
        if self.event_bus is None:
            return

        try:
            await self.event_bus.emit(
                topic,
                payload,
                source="open_interest_cache",
            )
        except Exception:
            self._logger.exception(
                "Failed to emit open interest cache event | topic=%s",
                topic,
            )

    @staticmethod
    def _build_state_key(exchange: str, symbol: str, market_type: str) -> str:
        return f"{exchange}:{market_type}:{symbol}"

    def _build_state_key_from_event(self, event: dict[str, Any]) -> str:
        return self._build_state_key(
            str(event["exchange"]),
            str(event["symbol"]),
            str(event.get("market_type", "perpetual")),
        )

    @staticmethod
    def _serialize_record(record: OpenInterestRecord) -> dict[str, Any]:
        return {
            "exchange": record.exchange,
            "symbol": record.symbol,
            "market_type": record.market_type,
            "timestamp_ms": record.timestamp_ms,
            "received_at_ms": record.received_at_ms,
            "open_interest": record.open_interest,
            "open_interest_value": record.open_interest_value,
            "mark_price": record.mark_price,
        }

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(Decimal(str(value)))
        except (InvalidOperation, TypeError, ValueError):
            return None

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)