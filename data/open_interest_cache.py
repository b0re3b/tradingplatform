from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from core.config import Config
from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler
from data.market_ingestion import MarketIngestionService
from data.market_models import MarketScope, OpenInterestUpdate
from data.market_state import MarketStateStore


OpenInterestRecord = OpenInterestUpdate


@dataclass(slots=True)
class OpenInterestState:
    exchange: str
    symbol: str
    market_type: str = "usdm_futures"
    total_updates: int = 0
    invalid_events: int = 0
    last_timestamp_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OpenInterestCache:
    """State-driven open-interest cache facade. Direct apply only; no EventBus subscriptions."""

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
        max_records_per_key: int = 1000,
        retention_ms: int = 30 * 24 * 60 * 60 * 1000,
        service_name: str = "open_interest_cache",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.cleanup_interval_seconds = float(cleanup_interval_seconds)
        self.max_records_per_key = int(max_records_per_key)
        self.retention_ms = int(retention_ms)
        self._service_name = service_name
        self._logger = get_logger(__name__, service=service_name, event_type="open_interest_cache")
        self.state_store = market_state or state_store or (ingestion.state_store if ingestion is not None else MarketStateStore())
        self.ingestion = ingestion or MarketIngestionService(state_store=self.state_store, event_bus=event_bus)
        self._cleanup_job_id: str | None = None
        self._registered = False
        self._metrics: dict[str, int | float] = {"records_stored": 0, "invalid_events": 0, "cleanup_runs": 0, "direct_apply_mode": 1}

    def register(self) -> None:
        if self._registered:
            return
        self._registered = True
        self._register_cleanup_job()
        self._logger.info("OpenInterestCache registered in state-driven mode | raw_eventbus_subscriptions=0")

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
            name="open-interest-cache-state-cleanup",
            func=self.cleanup_stale,
            interval=self.cleanup_interval_seconds,
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=30.0,
            allow_overlap=False,
            enabled=True,
        )

    async def update(self, event: Mapping[str, Any] | OpenInterestUpdate) -> None:
        ok = await self.ingestion.ingest_open_interest(event)
        if ok:
            self._metrics["records_stored"] += 1
        else:
            self._metrics["invalid_events"] += 1

    async def apply_open_interest(self, event: Mapping[str, Any] | OpenInterestUpdate) -> None:
        await self.update(event)

    async def get_latest(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures") -> dict[str, Any] | None:
        snapshot = await self.state_store.snapshot(MarketScope(exchange, market_type, symbol))
        if snapshot.open_interest is None:
            return None
        return {**snapshot.scope.to_dict(), **snapshot.open_interest.to_dict(), "current_price": snapshot.current_price}

    async def get_history(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures", limit: int = 100) -> list[dict[str, Any]]:
        latest = await self.get_latest(exchange=exchange, symbol=symbol, market_type=market_type)
        return [latest] if latest else []

    async def get_since(self, *, exchange: str, symbol: str, since_ms: int, market_type: str = "usdm_futures") -> list[dict[str, Any]]:
        latest = await self.get_latest(exchange=exchange, symbol=symbol, market_type=market_type)
        if latest and int(latest.get("timestamp_ms") or 0) >= since_ms:
            return [latest]
        return []

    async def get_window_stats(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures", **_: Any) -> dict[str, Any]:
        latest = await self.get_latest(exchange=exchange, symbol=symbol, market_type=market_type)
        return {"records": 1 if latest else 0, "latest": latest, "latest_open_interest": None if latest is None else latest.get("open_interest")}

    async def clear_symbol(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures") -> int:
        key = MarketScope(exchange, market_type, symbol).symbol_key
        state = getattr(self.state_store, "_states", {}).get(key)
        if state is None or state.open_interest is None:
            return 0
        state.open_interest = None
        return 1

    async def cleanup_stale(self) -> int:
        self._metrics["cleanup_runs"] += 1
        result = await self.state_store.cleanup_stale()
        return int(result.get("removed", 0) or 0)

    async def has_data(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures") -> bool:
        return await self.get_latest(exchange=exchange, symbol=symbol, market_type=market_type) is not None

    async def size(self, *, exchange: str | None = None, symbol: str | None = None, market_type: str = "usdm_futures") -> int:
        if exchange and symbol:
            return 1 if await self.has_data(exchange=exchange, symbol=symbol, market_type=market_type) else 0
        stats = await self.state_store.stats()
        return int(stats.get("open_interest_ingested", 0) or 0)

    def stats(self) -> dict[str, Any]:
        return {**self._metrics, "ingestion": self.ingestion.stats()}
