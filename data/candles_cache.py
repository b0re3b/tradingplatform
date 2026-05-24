from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from core.config import Config
from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler
from data.market_ingestion import MarketIngestionService
from data.market_models import CandleUpdate, MarketScope
from data.market_state import MarketStateStore


CandleRecord = CandleUpdate


@dataclass(slots=True)
class CandlesState:
    exchange: str
    symbol: str
    market_type: str
    timeframe: str
    last_candle_open_time_ms: int | None = None
    last_update_ts_ms: int | None = None
    total_updates: int = 0
    total_closed: int = 0
    invalid_events: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class CandlesCache:
    """
    State-driven candles cache facade.

    Direct apply only. It does not subscribe to market.candle / market.candles.snapshot.
    Closed candles may still create a low-frequency market.candle.closed trigger through
    MarketIngestionService, but high-frequency transport no longer uses EventBus.
    """

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        market_state: MarketStateStore | None = None,
        state_store: MarketStateStore | None = None,
        ingestion: MarketIngestionService | None = None,
        cleanup_interval_seconds: float = 60.0,
        max_candles_per_key: int = 2000,
        retention_ms: int = 7 * 24 * 60 * 60 * 1000,
        service_name: str = "candles_cache",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.cleanup_interval_seconds = float(cleanup_interval_seconds)
        self.max_candles_per_key = int(max_candles_per_key)
        self.retention_ms = int(retention_ms)
        self._service_name = service_name
        self._logger = get_logger(__name__, service=service_name, event_type="candles_cache")
        self.state_store = market_state or state_store or (ingestion.state_store if ingestion is not None else MarketStateStore())
        self.ingestion = ingestion or MarketIngestionService(state_store=self.state_store, event_bus=event_bus)
        self._cleanup_job_id: str | None = None
        self._registered = False
        self._metrics: dict[str, int | float] = {
            "candles_received": 0,
            "candles_closed": 0,
            "invalid_events": 0,
            "cleanup_runs": 0,
            "direct_apply_mode": 1,
        }

    def register(self) -> None:
        if self._registered:
            return
        self._registered = True
        self._register_cleanup_job()
        self._logger.info("CandlesCache registered in state-driven mode | raw_eventbus_subscriptions=0")

    async def start(self) -> None:
        self.register()

    async def stop(self) -> None:
        if self.scheduler is not None and self._cleanup_job_id is not None:
            with contextlib.suppress(Exception):
                self.scheduler.remove_job(self._cleanup_job_id)
        self._cleanup_job_id = None

    def _register_cleanup_job(self) -> None:
        if self.scheduler is None or self._cleanup_job_id is not None:
            return
        self._cleanup_job_id = self.scheduler.add_interval_job(
            name="candles-cache-state-cleanup",
            func=self.cleanup_stale,
            interval=self.cleanup_interval_seconds,
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=30.0,
            allow_overlap=False,
            enabled=True,
        )

    async def update(self, event: Mapping[str, Any] | CandleUpdate) -> None:
        update = event if isinstance(event, CandleUpdate) else CandleUpdate.from_payload(event)
        if update is None:
            self._metrics["invalid_events"] += 1
            return
        await self.ingestion.ingest_candle(update)
        self._metrics["candles_received"] += 1
        if update.is_closed:
            self._metrics["candles_closed"] += 1

    async def apply_candle(self, event: Mapping[str, Any] | CandleUpdate) -> None:
        await self.update(event)

    async def apply_candles_batch(self, payload: Mapping[str, Any] | list[Mapping[str, Any] | CandleUpdate]) -> int:
        count = await self.ingestion.ingest_candles_batch(payload)
        self._metrics["candles_received"] += int(count)
        if count <= 0:
            self._metrics["invalid_events"] += 1
        return count

    async def get_recent_candles(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        market_type: str = "usdm_futures",
        limit: int = 100,
        closed_only: bool = False,
    ) -> list[dict[str, Any]]:
        snapshot = await self.state_store.snapshot(MarketScope(exchange, market_type, symbol, timeframe))
        window = snapshot.candles.get(timeframe)
        if window is None:
            return []
        candles = list(window.candles)
        if closed_only:
            candles = [c for c in candles if c.is_closed]
        candles = candles[-max(1, int(limit)):]
        return [candle.to_dict() for candle in candles]

    async def get_last_candle(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        market_type: str = "usdm_futures",
        closed_only: bool = False,
    ) -> dict[str, Any] | None:
        candles = await self.get_recent_candles(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            market_type=market_type,
            limit=1_000 if closed_only else 1,
            closed_only=closed_only,
        )
        return candles[-1] if candles else None

    async def get_candles_since(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        since_ms: int,
        market_type: str = "usdm_futures",
    ) -> list[dict[str, Any]]:
        candles = await self.get_recent_candles(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            market_type=market_type,
            limit=self.max_candles_per_key,
        )
        return [c for c in candles if int(c.get("open_time_ms") or 0) >= since_ms]

    async def get_window_stats(self, *, exchange: str, symbol: str, timeframe: str, market_type: str = "usdm_futures", limit: int = 100) -> dict[str, Any]:
        candles = await self.get_recent_candles(exchange=exchange, symbol=symbol, timeframe=timeframe, market_type=market_type, limit=limit)
        if not candles:
            return {"candle_count": 0, "last_close": None, "high": None, "low": None, "volume": 0.0}
        return {
            "candle_count": len(candles),
            "last_close": candles[-1].get("close"),
            "high": max(float(c.get("high") or 0.0) for c in candles),
            "low": min(float(c.get("low") or 0.0) for c in candles),
            "volume": sum(float(c.get("volume") or 0.0) for c in candles),
            "first_open_time_ms": candles[0].get("open_time_ms"),
            "last_close_time_ms": candles[-1].get("close_time_ms"),
        }

    async def clear_symbol_timeframe(self, *, exchange: str, symbol: str, timeframe: str, market_type: str = "usdm_futures") -> int:
        key = MarketScope(exchange, market_type, symbol).symbol_key
        state = getattr(self.state_store, "_states", {}).get(key)
        if state is None:
            return 0
        removed = len(state.candles.get(timeframe, {}))
        state.candles.pop(timeframe, None)
        return removed

    async def cleanup_stale(self) -> int:
        self._metrics["cleanup_runs"] += 1
        result = await self.state_store.cleanup_stale()
        return int(result.get("removed", 0) or 0)

    async def has_candles(self, *, exchange: str, symbol: str, timeframe: str, market_type: str = "usdm_futures") -> bool:
        return bool(await self.get_recent_candles(exchange=exchange, symbol=symbol, timeframe=timeframe, market_type=market_type, limit=1))

    async def size(self, *, exchange: str | None = None, symbol: str | None = None, timeframe: str | None = None, market_type: str = "usdm_futures") -> int:
        if exchange and symbol and timeframe:
            return len(await self.get_recent_candles(exchange=exchange, symbol=symbol, timeframe=timeframe, market_type=market_type, limit=self.max_candles_per_key))
        stats = await self.state_store.stats()
        return int(stats.get("candles_ingested", 0) or 0)

    def stats(self) -> dict[str, Any]:
        return {**self._metrics, "ingestion": self.ingestion.stats()}
