from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from core.config import Config
from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler


@dataclass(slots=True)
class CandleRecord:
    exchange: str
    symbol: str
    market_type: str
    timeframe: str

    open_time_ms: int
    close_time_ms: int

    open: float
    high: float
    low: float
    close: float
    volume: float

    is_closed: bool
    timestamp_ms: int
    received_at_ms: int


@dataclass(slots=True)
class CandlesState:
    exchange: str
    symbol: str
    market_type: str
    timeframe: str

    candles: dict[int, CandleRecord] = field(default_factory=dict)

    last_candle_open_time_ms: int | None = None
    last_update_ts_ms: int | None = None
    last_received_at_ms: int | None = None
    last_error: str | None = None

    total_updates: int = 0
    total_closed: int = 0
    invalid_events: int = 0
    trims_count: int = 0


class CandlesCache:
    """
    Локальний кеш свічок.

    Відповідальність:
    - зберігати останні candles у пам'яті
    - оновлювати поточну свічку або додавати нову
    - підтримувати bounded history
    - віддавати recent / closed candles
    - чистити застарілі записи
    """

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        cleanup_interval_seconds: float = 60.0,
        max_candles_per_key: int = 2000,
        retention_ms: int = 7 * 24 * 60 * 60 * 1000,
        service_name: str = "candles_cache",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._cleanup_job_id: str | None = None
        self.max_candles_per_key = max_candles_per_key
        self.retention_ms = retention_ms
        self._service_name = service_name

        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="candles_cache",
        )

        self._states: dict[str, CandlesState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

        self._metrics: dict[str, int | float] = {
            "states_created": 0,
            "events_received": 0,
            "candles_upserted": 0,
            "candles_closed": 0,
            "invalid_events": 0,
            "trimmed_candles": 0,
            "cleanup_runs": 0,
            "cleanup_removed": 0,
            "last_cleanup_at": 0.0,
        }

    # ------------------------------------------------------------------
    # Lifecycle / EventBus integration
    # ------------------------------------------------------------------

    def register(self) -> None:
        """Підписує cache на всі candle-події від exchange adapters через EventBus."""
        if self.event_bus is None:
            self._logger.warning("CandlesCache register skipped: EventBus is not provided")
            return
        self.event_bus.subscribe("market.candle", self._on_market_candle)
        self.event_bus.subscribe("market.candles.snapshot", self._on_market_candles_snapshot)
        self._register_cleanup_job()
        self._logger.info("CandlesCache registered | topics=%s", ["market.candle", "market.candles.snapshot"])

    async def start(self) -> None:
        self.register()

    async def stop(self) -> None:
        if self.scheduler is not None and self._cleanup_job_id is not None:
            self.scheduler.remove_job(self._cleanup_job_id)
            self._cleanup_job_id = None

    async def _on_market_candle(self, event: Any) -> None:
        await self.update(self._normalize_inbound_payload(self._extract_payload(event)))

    async def _on_market_candles_snapshot(self, event: Any) -> None:
        payload = self._extract_payload(event)
        candles = payload.get("candles") or payload.get("items") or []
        if isinstance(candles, list):
            for candle in candles:
                if isinstance(candle, dict):
                    merged = {**payload, **candle}
                    merged.pop("candles", None)
                    merged.pop("items", None)
                    await self.update(self._normalize_inbound_payload(merged))

    def _register_cleanup_job(self) -> None:
        if self.scheduler is None or self._cleanup_job_id is not None:
            return
        self._cleanup_job_id = self.scheduler.add_interval_job(
            name="candles-cache-cleanup",
            func=self.cleanup_stale,
            interval=self.cleanup_interval_seconds,
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=30.0,
            allow_overlap=False,
            enabled=True,
        )

    @staticmethod
    def _extract_payload(event: Any) -> dict[str, Any]:
        payload = getattr(event, "payload", event)
        return payload if isinstance(payload, dict) else {}

    def _normalize_inbound_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = self._now_ms()
        normalized = dict(payload)
        normalized.setdefault("market_type", payload.get("category") or payload.get("market_type") or "perpetual")
        normalized["timestamp_ms"] = payload.get("timestamp_ms") or payload.get("event_time") or payload.get("timestamp") or payload.get("open_time") or now
        normalized["received_at_ms"] = payload.get("received_at_ms") or now
        normalized["open_time_ms"] = payload.get("open_time_ms") or payload.get("open_time") or payload.get("start")
        normalized["close_time_ms"] = payload.get("close_time_ms") or payload.get("close_time") or payload.get("end")
        return normalized

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
            "timeframe": "1m",
            "open": ...,
            "high": ...,
            "low": ...,
            "close": ...,
            "volume": ...,
            "is_closed": bool,
            # optional:
            "open_time_ms": ...,
            "close_time_ms": ...,
        }
        """
        state_key = self._build_state_key_from_event(event)
        lock = self._get_lock(state_key)

        async with lock:
            self._metrics["events_received"] += 1

            record = self._normalize_candle(event)
            if record is None:
                self._metrics["invalid_events"] += 1

                state = self._states.get(state_key)
                if state is not None:
                    state.invalid_events += 1
                    state.last_error = "invalid_candle_event"

                await self._emit_event(
                    "system.candles_cache.invalid_candle",
                    {
                        "exchange": event.get("exchange"),
                        "symbol": event.get("symbol"),
                        "market_type": event.get("market_type", "perpetual"),
                        "timeframe": event.get("timeframe"),
                    },
                )
                return

            state = self._get_or_create_state(record)

            existing = state.candles.get(record.open_time_ms)
            if existing is None:
                state.candles[record.open_time_ms] = record
            else:
                state.candles[record.open_time_ms] = self._merge_candles(existing, record)

            current = state.candles[record.open_time_ms]

            state.last_candle_open_time_ms = current.open_time_ms
            state.last_update_ts_ms = current.timestamp_ms
            state.last_received_at_ms = current.received_at_ms
            state.last_error = None
            state.total_updates += 1

            self._metrics["candles_upserted"] += 1

            if current.is_closed:
                state.total_closed += 1
                self._metrics["candles_closed"] += 1

            removed = self._trim_state(state)
            if removed > 0:
                state.trims_count += 1
                self._metrics["trimmed_candles"] += removed

            serialized = self._serialize_candle(current)
            await self._emit_event("market.candles.updated", serialized)
            if current.is_closed:
                await self._emit_event("market.candle.closed", serialized)

    async def get_recent_candles(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        market_type: str = "perpetual",
        limit: int = 100,
        only_closed: bool = False,
    ) -> list[dict[str, Any]]:
        state_key = self._build_state_key(exchange, symbol, market_type, timeframe)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._states.get(state_key)
            if state is None or limit <= 0:
                return []

            candles = self._sorted_candles(state)
            if only_closed:
                candles = [c for c in candles if c.is_closed]

            return [self._serialize_candle(c) for c in candles[-limit:]]

    async def get_last_candle(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        market_type: str = "perpetual",
        only_closed: bool = False,
    ) -> dict[str, Any] | None:
        candles = await self.get_recent_candles(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            market_type=market_type,
            limit=1 if not only_closed else max(2, self.max_candles_per_key),
            only_closed=only_closed,
        )
        if not candles:
            return None
        return candles[-1]

    async def get_candles_since(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        since_open_time_ms: int,
        market_type: str = "perpetual",
        limit: int | None = None,
        only_closed: bool = False,
    ) -> list[dict[str, Any]]:
        state_key = self._build_state_key(exchange, symbol, market_type, timeframe)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._states.get(state_key)
            if state is None:
                return []

            candles = [
                candle
                for candle in self._sorted_candles(state)
                if candle.open_time_ms >= since_open_time_ms
            ]

            if only_closed:
                candles = [c for c in candles if c.is_closed]

            if limit is not None and limit > 0:
                candles = candles[-limit:]

            return [self._serialize_candle(c) for c in candles]

    async def get_window_stats(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        market_type: str = "perpetual",
        limit: int = 50,
        only_closed: bool = True,
    ) -> dict[str, Any] | None:
        candles = await self.get_recent_candles(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            market_type=market_type,
            limit=limit,
            only_closed=only_closed,
        )
        if not candles:
            return None

        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        volumes = [float(c["volume"]) for c in candles]
        closes = [float(c["close"]) for c in candles]

        return {
            "exchange": exchange,
            "symbol": symbol,
            "market_type": market_type,
            "timeframe": timeframe,
            "candles_count": len(candles),
            "highest_high": max(highs),
            "lowest_low": min(lows),
            "total_volume": sum(volumes),
            "average_volume": sum(volumes) / len(volumes),
            "last_close": closes[-1],
            "first_open_time_ms": candles[0]["open_time_ms"],
            "last_open_time_ms": candles[-1]["open_time_ms"],
        }

    async def clear_symbol_timeframe(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        market_type: str = "perpetual",
        reason: str = "manual_clear",
    ) -> None:
        state_key = self._build_state_key(exchange, symbol, market_type, timeframe)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._states.get(state_key)
            if state is None:
                return

            removed = len(state.candles)
            state.candles.clear()
            state.last_candle_open_time_ms = None
            state.last_update_ts_ms = None
            state.last_received_at_ms = None
            state.last_error = reason

            self._logger.warning(
                "Candles cleared | exchange=%s symbol=%s timeframe=%s removed=%s reason=%s",
                exchange,
                symbol,
                timeframe,
                removed,
                reason,
            )

            await self._emit_event(
                "system.candles_cache.cleared",
                {
                    "exchange": exchange,
                    "symbol": symbol,
                    "market_type": market_type,
                    "timeframe": timeframe,
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
                to_delete = [
                    open_time_ms
                    for open_time_ms, candle in state.candles.items()
                    if candle.close_time_ms < cutoff
                ]

                for open_time_ms in to_delete:
                    state.candles.pop(open_time_ms, None)

                removed_here = len(to_delete)
                if removed_here > 0:
                    total_removed += removed_here
                    state.trims_count += 1

        self._metrics["cleanup_runs"] += 1
        self._metrics["cleanup_removed"] += total_removed
        self._metrics["last_cleanup_at"] = time.time()

        if total_removed > 0:
            self._logger.info(
                "Candles cleanup completed | removed=%s retention_ms=%s",
                total_removed,
                self.retention_ms,
            )

        return total_removed

    async def has_candles(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        market_type: str = "perpetual",
    ) -> bool:
        state_key = self._build_state_key(exchange, symbol, market_type, timeframe)
        state = self._states.get(state_key)
        return state is not None and len(state.candles) > 0

    async def size(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        market_type: str = "perpetual",
    ) -> int:
        state_key = self._build_state_key(exchange, symbol, market_type, timeframe)
        state = self._states.get(state_key)
        if state is None:
            return 0
        return len(state.candles)

    def stats(self) -> dict[str, Any]:
        active_states = sum(1 for state in self._states.values() if state.candles)

        return {
            "states_total": len(self._states),
            "states_with_candles": active_states,
            "states_created": self._metrics["states_created"],
            "events_received": self._metrics["events_received"],
            "candles_upserted": self._metrics["candles_upserted"],
            "candles_closed": self._metrics["candles_closed"],
            "invalid_events": self._metrics["invalid_events"],
            "trimmed_candles": self._metrics["trimmed_candles"],
            "cleanup_runs": self._metrics["cleanup_runs"],
            "cleanup_removed": self._metrics["cleanup_removed"],
            "last_cleanup_at": self._metrics["last_cleanup_at"],
            "max_candles_per_key": self.max_candles_per_key,
            "retention_ms": self.retention_ms,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create_state(self, record: CandleRecord) -> CandlesState:
        state_key = self._build_state_key(
            record.exchange,
            record.symbol,
            record.market_type,
            record.timeframe,
        )

        state = self._states.get(state_key)
        if state is not None:
            return state

        state = CandlesState(
            exchange=record.exchange,
            symbol=record.symbol,
            market_type=record.market_type,
            timeframe=record.timeframe,
        )
        self._states[state_key] = state
        self._metrics["states_created"] += 1

        self._logger.info(
            "Candles state created | exchange=%s symbol=%s timeframe=%s market_type=%s",
            record.exchange,
            record.symbol,
            record.timeframe,
            record.market_type,
        )

        return state

    def _get_lock(self, state_key: str) -> asyncio.Lock:
        lock = self._locks.get(state_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[state_key] = lock
        return lock

    def _normalize_candle(self, event: dict[str, Any]) -> CandleRecord | None:
        exchange = event.get("exchange")
        symbol = event.get("symbol")
        market_type = event.get("market_type", "perpetual")
        timeframe = event.get("timeframe")

        timestamp_ms = self._safe_int(event.get("timestamp_ms"))
        received_at_ms = self._safe_int(event.get("received_at_ms")) or self._now_ms()

        open_price = self._safe_float(event.get("open"))
        high_price = self._safe_float(event.get("high"))
        low_price = self._safe_float(event.get("low"))
        close_price = self._safe_float(event.get("close"))
        volume = self._safe_float(event.get("volume"))

        if exchange is None or symbol is None or timeframe is None:
            return None

        if (
            timestamp_ms is None
            or open_price is None
            or high_price is None
            or low_price is None
            or close_price is None
            or volume is None
        ):
            return None

        if (
            open_price <= 0
            or high_price <= 0
            or low_price <= 0
            or close_price <= 0
            or volume < 0
        ):
            return None

        if low_price > high_price:
            return None

        timeframe_ms = self._timeframe_to_ms(str(timeframe))
        if timeframe_ms is None or timeframe_ms <= 0:
            return None

        open_time_ms = self._safe_int(event.get("open_time_ms"))
        close_time_ms = self._safe_int(event.get("close_time_ms"))

        if open_time_ms is None:
            open_time_ms = (timestamp_ms // timeframe_ms) * timeframe_ms

        if close_time_ms is None:
            close_time_ms = open_time_ms + timeframe_ms - 1

        is_closed = bool(event.get("is_closed", False))

        return CandleRecord(
            exchange=str(exchange),
            symbol=str(symbol),
            market_type=str(market_type),
            timeframe=str(timeframe),
            open_time_ms=open_time_ms,
            close_time_ms=close_time_ms,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            is_closed=is_closed,
            timestamp_ms=timestamp_ms,
            received_at_ms=received_at_ms,
        )

    def _merge_candles(self, existing: CandleRecord, incoming: CandleRecord) -> CandleRecord:
        """
        Якщо це апдейт тієї ж свічки, оновлюємо значення новішими даними.
        """
        return CandleRecord(
            exchange=existing.exchange,
            symbol=existing.symbol,
            market_type=existing.market_type,
            timeframe=existing.timeframe,
            open_time_ms=existing.open_time_ms,
            close_time_ms=max(existing.close_time_ms, incoming.close_time_ms),
            open=incoming.open if incoming.timestamp_ms >= existing.timestamp_ms else existing.open,
            high=max(existing.high, incoming.high),
            low=min(existing.low, incoming.low),
            close=incoming.close if incoming.timestamp_ms >= existing.timestamp_ms else existing.close,
            volume=max(existing.volume, incoming.volume),
            is_closed=existing.is_closed or incoming.is_closed,
            timestamp_ms=max(existing.timestamp_ms, incoming.timestamp_ms),
            received_at_ms=max(existing.received_at_ms, incoming.received_at_ms),
        )

    def _trim_state(self, state: CandlesState) -> int:
        removed = 0
        cutoff = self._now_ms() - self.retention_ms

        sorted_open_times = sorted(state.candles.keys())

        while len(sorted_open_times) > self.max_candles_per_key:
            oldest = sorted_open_times.pop(0)
            state.candles.pop(oldest, None)
            removed += 1

        sorted_open_times = sorted(state.candles.keys())
        for open_time_ms in sorted_open_times:
            candle = state.candles.get(open_time_ms)
            if candle is None:
                continue
            if candle.close_time_ms < cutoff:
                state.candles.pop(open_time_ms, None)
                removed += 1

        return removed

    async def _emit_event(self, topic: str, payload: dict[str, Any]) -> None:
        if self.event_bus is None:
            return

        try:
            await self.event_bus.emit(
                topic,
                payload,
                source="candles_cache",
            )
        except Exception:
            self._logger.exception(
                "Failed to emit candles cache event | topic=%s",
                topic,
            )

    @staticmethod
    def _build_state_key(exchange: str, symbol: str, market_type: str, timeframe: str) -> str:
        return f"{exchange}:{market_type}:{symbol}:{timeframe}"

    def _build_state_key_from_event(self, event: dict[str, Any]) -> str:
        return self._build_state_key(
            str(event["exchange"]),
            str(event["symbol"]),
            str(event.get("market_type", "perpetual")),
            str(event["timeframe"]),
        )

    @staticmethod
    def _sorted_candles(state: CandlesState) -> list[CandleRecord]:
        return [state.candles[key] for key in sorted(state.candles.keys())]

    @staticmethod
    def _serialize_candle(candle: CandleRecord) -> dict[str, Any]:
        return {
            "exchange": candle.exchange,
            "symbol": candle.symbol,
            "market_type": candle.market_type,
            "timeframe": candle.timeframe,
            "open_time_ms": candle.open_time_ms,
            "close_time_ms": candle.close_time_ms,
            "timestamp_ms": candle.timestamp_ms,
            "received_at_ms": candle.received_at_ms,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "is_closed": candle.is_closed,
        }

    @staticmethod
    def _timeframe_to_ms(timeframe: str) -> int | None:
        normalized = timeframe.strip().lower()
        if not normalized:
            return None

        unit = normalized[-1]
        number_part = normalized[:-1]

        try:
            value = int(number_part)
        except ValueError:
            return None

        if value <= 0:
            return None

        if unit == "s":
            return value * 1000
        if unit == "m":
            return value * 60 * 1000
        if unit == "h":
            return value * 60 * 60 * 1000
        if unit == "d":
            return value * 24 * 60 * 60 * 1000

        return None

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