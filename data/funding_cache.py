from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from core.config import Config
from core.logger import get_logger


@dataclass(slots=True)
class FundingRecord:
    exchange: str
    symbol: str
    market_type: str

    timestamp_ms: int
    received_at_ms: int

    funding_rate: float
    next_funding_time_ms: int | None = None
    mark_price: float | None = None
    index_price: float | None = None
    predicted_rate: float | None = None


@dataclass(slots=True)
class FundingState:
    exchange: str
    symbol: str
    market_type: str = "perpetual"

    history: list[FundingRecord] = field(default_factory=list)

    last_timestamp_ms: int | None = None
    last_received_at_ms: int | None = None
    last_next_funding_time_ms: int | None = None
    last_error: str | None = None

    total_updates: int = 0
    invalid_events: int = 0
    duplicate_events: int = 0
    trims_count: int = 0


class FundingCache:
    """
    Локальний кеш funding data.

    Відповідальність:
    - зберігати останні funding records
    - підтримувати bounded history
    - віддавати latest / history / window stats
    - чистити застарілі записи
    """

    def __init__(
        self,
        *,
        config: Config,
        event_bus: Any | None = None,
        max_records_per_key: int = 1000,
        retention_ms: int = 30 * 24 * 60 * 60 * 1000,
        service_name: str = "funding_cache",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.max_records_per_key = max_records_per_key
        self.retention_ms = retention_ms
        self._service_name = service_name

        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="funding_cache",
        )

        self._states: dict[str, FundingState] = {}
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
            "funding_rate": ...,
            "next_funding_time_ms": ... | None,
            # optional:
            "mark_price": ...,
            "index_price": ...,
            "predicted_rate": ...,
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
                    state.last_error = "invalid_funding_event"

                await self._emit_event(
                    "system.funding_cache.invalid_funding",
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
                    "Duplicate funding event detected | exchange=%s symbol=%s timestamp_ms=%s",
                    record.exchange,
                    record.symbol,
                    record.timestamp_ms,
                )
                return

            state.history.append(record)
            state.last_timestamp_ms = record.timestamp_ms
            state.last_received_at_ms = record.received_at_ms
            state.last_next_funding_time_ms = record.next_funding_time_ms
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
                    "latest_rate": None,
                    "min_rate": None,
                    "max_rate": None,
                    "avg_rate": None,
                    "latest_next_funding_time_ms": None,
                }

            rates = [r.funding_rate for r in records]

            return {
                "exchange": exchange,
                "symbol": symbol,
                "market_type": market_type,
                "window_ms": window_ms,
                "records_count": len(records),
                "latest_rate": records[-1].funding_rate,
                "min_rate": min(rates),
                "max_rate": max(rates),
                "avg_rate": sum(rates) / len(rates),
                "latest_next_funding_time_ms": records[-1].next_funding_time_ms,
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
            state.last_next_funding_time_ms = None
            state.last_error = reason

            self._logger.warning(
                "Funding history cleared | exchange=%s symbol=%s removed=%s reason=%s",
                exchange,
                symbol,
                removed,
                reason,
            )

            await self._emit_event(
                "system.funding_cache.cleared",
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
                "Funding cleanup completed | removed=%s retention_ms=%s",
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

    def _get_or_create_state(self, record: FundingRecord) -> FundingState:
        state_key = self._build_state_key(
            record.exchange,
            record.symbol,
            record.market_type,
        )

        state = self._states.get(state_key)
        if state is not None:
            return state

        state = FundingState(
            exchange=record.exchange,
            symbol=record.symbol,
            market_type=record.market_type,
        )
        self._states[state_key] = state
        self._metrics["states_created"] += 1

        self._logger.info(
            "Funding state created | exchange=%s symbol=%s market_type=%s",
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

    def _normalize_record(self, event: dict[str, Any]) -> FundingRecord | None:
        exchange = event.get("exchange")
        symbol = event.get("symbol")
        market_type = event.get("market_type", "perpetual")

        timestamp_ms = self._safe_int(event.get("timestamp_ms"))
        received_at_ms = self._safe_int(event.get("received_at_ms")) or self._now_ms()
        funding_rate = self._safe_float(event.get("funding_rate"))

        if exchange is None or symbol is None:
            return None
        if timestamp_ms is None or funding_rate is None:
            return None

        return FundingRecord(
            exchange=str(exchange),
            symbol=str(symbol),
            market_type=str(market_type),
            timestamp_ms=timestamp_ms,
            received_at_ms=received_at_ms,
            funding_rate=funding_rate,
            next_funding_time_ms=self._safe_int(event.get("next_funding_time_ms")),
            mark_price=self._safe_float(event.get("mark_price")),
            index_price=self._safe_float(event.get("index_price")),
            predicted_rate=self._safe_float(event.get("predicted_rate")),
        )

    def _is_duplicate(self, state: FundingState, record: FundingRecord) -> bool:
        if not state.history:
            return False

        last = state.history[-1]
        return (
            last.timestamp_ms == record.timestamp_ms
            and last.funding_rate == record.funding_rate
            and last.next_funding_time_ms == record.next_funding_time_ms
        )

    def _trim_state(self, state: FundingState) -> int:
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
                source="funding_cache",
            )
        except Exception:
            self._logger.exception(
                "Failed to emit funding cache event | topic=%s",
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
    def _serialize_record(record: FundingRecord) -> dict[str, Any]:
        return {
            "exchange": record.exchange,
            "symbol": record.symbol,
            "market_type": record.market_type,
            "timestamp_ms": record.timestamp_ms,
            "received_at_ms": record.received_at_ms,
            "funding_rate": record.funding_rate,
            "next_funding_time_ms": record.next_funding_time_ms,
            "mark_price": record.mark_price,
            "index_price": record.index_price,
            "predicted_rate": record.predicted_rate,
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