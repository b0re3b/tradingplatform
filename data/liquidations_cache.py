from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from core.config import Config
from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler
from data.market_ingestion import MarketIngestionService
from data.market_models import LiquidationUpdate, MarketScope, now_ms
from data.market_state import MarketStateStore


LiquidationRecord = LiquidationUpdate


@dataclass(slots=True)
class LiquidationsState:
    exchange: str
    symbol: str
    market_type: str = "usdm_futures"
    total_updates: int = 0
    invalid_events: int = 0
    last_timestamp_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LiquidationsCache:
    """State-driven liquidation cache. Direct apply only; no high-frequency EventBus transport."""

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
        max_records_per_symbol: int = 1000,
        retention_ms: int = 60 * 60 * 1000,
        service_name: str = "liquidations_cache",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.cleanup_interval_seconds = float(cleanup_interval_seconds)
        self.max_records_per_symbol = int(max_records_per_symbol)
        self.retention_ms = int(retention_ms)
        self._service_name = service_name
        self._logger = get_logger(__name__, service=service_name, event_type="liquidations_cache")
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
        self._logger.info("LiquidationsCache registered in state-driven mode | raw_eventbus_subscriptions=0")

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
            name="liquidations-cache-state-cleanup",
            func=self.cleanup_stale,
            interval=self.cleanup_interval_seconds,
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=30.0,
            allow_overlap=False,
            enabled=True,
        )

    async def update(self, event: Mapping[str, Any] | LiquidationUpdate) -> None:
        ok = await self.ingestion.ingest_liquidation(event)
        if ok:
            self._metrics["records_stored"] += 1
        else:
            self._metrics["invalid_events"] += 1

    async def apply_liquidation(self, event: Mapping[str, Any] | LiquidationUpdate) -> None:
        await self.update(event)

    async def get_recent_liquidations(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures", limit: int = 100, window_ms: int | None = None) -> list[dict[str, Any]]:
        snapshot = await self.state_store.snapshot(MarketScope(exchange, market_type, symbol))
        items = list(snapshot.liquidations)
        if window_ms is not None:
            cutoff = now_ms() - int(window_ms)
            items = [item for item in items if item.timestamp_ms >= cutoff]
        items = items[-max(1, int(limit)):]
        return [item.to_dict() for item in items]

    async def get_window_stats(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures", window_ms: int | None = None) -> dict[str, Any]:
        items = await self.get_recent_liquidations(exchange=exchange, symbol=symbol, market_type=market_type, limit=self.max_records_per_symbol, window_ms=window_ms)
        long_qty = sum(float(i.get("quantity") or 0.0) for i in items if str(i.get("side")).lower() in {"buy", "long"})
        short_qty = sum(float(i.get("quantity") or 0.0) for i in items if str(i.get("side")).lower() in {"sell", "short"})
        return {"count": len(items), "long_quantity": long_qty, "short_quantity": short_qty, "total_quantity": long_qty + short_qty, "latest": items[-1] if items else None}

    async def cleanup_stale(self) -> int:
        self._metrics["cleanup_runs"] += 1
        result = await self.state_store.cleanup_stale()
        return int(result.get("removed", 0) or 0)

    async def has_data(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures") -> bool:
        return bool(await self.get_recent_liquidations(exchange=exchange, symbol=symbol, market_type=market_type, limit=1))

    def stats(self) -> dict[str, Any]:
        return {**self._metrics, "ingestion": self.ingestion.stats()}


# Backward-compatible singular name for callers that use LiquidationCache.
LiquidationCache = LiquidationsCache
