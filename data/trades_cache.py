from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from core.config import Config
from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler
from data.market_ingestion import MarketIngestionService
from data.market_models import MarketScope, TradeUpdate, now_ms
from data.market_state import MarketStateStore


TradeRecord = TradeUpdate


@dataclass(slots=True)
class TradesState:
    exchange: str
    symbol: str
    market_type: str = "usdm_futures"
    timeframe: str | None = None
    total_received: int = 0
    invalid_trades: int = 0
    last_trade_ts_ms: int | None = None
    last_received_at_ms: int | None = None
    last_trade_id: str | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TradesCache:
    """
    State-driven trades cache facade.

    New responsibility:
    - accept direct apply calls from MarketIngestionService / exchange adapters;
    - mutate shared MarketStateStore;
    - never subscribe to raw EventBus topics;
    - never emit high-frequency market.trades.updated events.

    Analytics should read MarketStateStore snapshots through MarketScheduler.
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
        max_trades_per_book: int = 5000,
        retention_ms: int = 15 * 60 * 1000,
        trades_updated_emit_min_interval_ms: int = 0,
        service_name: str = "trades_cache",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.cleanup_interval_seconds = float(cleanup_interval_seconds)
        self.max_trades_per_book = int(max_trades_per_book)
        self.retention_ms = int(retention_ms)
        self.trades_updated_emit_min_interval_ms = 0
        self._service_name = service_name
        self._logger = get_logger(__name__, service=service_name, event_type="trades_cache")
        self.state_store = market_state or state_store or (ingestion.state_store if ingestion is not None else MarketStateStore())
        self.ingestion = ingestion or MarketIngestionService(state_store=self.state_store, event_bus=event_bus)
        self._cleanup_job_id: str | None = None
        self._registered = False
        self._metrics: dict[str, int | float] = {
            "trades_received": 0,
            "invalid_trades": 0,
            "cleanup_runs": 0,
            "direct_apply_mode": 1,
        }

    def register(self) -> None:
        """Register maintenance only. Raw EventBus subscriptions are intentionally disabled."""
        if self._registered:
            return
        self._registered = True
        self._register_cleanup_job()
        self._logger.info("TradesCache registered in state-driven mode | raw_eventbus_subscriptions=0")

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
            name="trades-cache-state-cleanup",
            func=self.cleanup_stale,
            interval=self.cleanup_interval_seconds,
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=30.0,
            allow_overlap=False,
            enabled=True,
        )

    async def update(self, event: Mapping[str, Any] | TradeUpdate) -> None:
        ok = await self.ingestion.ingest_trade(event)
        if ok:
            self._metrics["trades_received"] += 1
        else:
            self._metrics["invalid_trades"] += 1

    async def apply_trade(self, event: Mapping[str, Any] | TradeUpdate) -> None:
        await self.update(event)

    async def apply_trades_batch(self, payload: Mapping[str, Any] | list[Mapping[str, Any] | TradeUpdate]) -> int:
        count = await self.ingestion.ingest_trades_batch(payload)
        self._metrics["trades_received"] += int(count)
        if count <= 0:
            self._metrics["invalid_trades"] += 1
        return count

    async def get_recent_trades(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "usdm_futures",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        snapshot = await self.state_store.snapshot(MarketScope(exchange, market_type, symbol))
        trades = list(snapshot.trades.trades)[-max(1, int(limit)):]
        return [trade.to_dict() for trade in trades]

    async def get_trades_since(
        self,
        *,
        exchange: str,
        symbol: str,
        since_ms: int,
        market_type: str = "usdm_futures",
    ) -> list[dict[str, Any]]:
        snapshot = await self.state_store.snapshot(MarketScope(exchange, market_type, symbol))
        return [trade.to_dict() for trade in snapshot.trades.trades if trade.timestamp_ms >= since_ms]

    async def get_last_trade(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "usdm_futures",
    ) -> dict[str, Any] | None:
        snapshot = await self.state_store.snapshot(MarketScope(exchange, market_type, symbol))
        if not snapshot.trades.trades:
            return None
        return snapshot.trades.trades[-1].to_dict()

    async def get_window_stats(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "usdm_futures",
        window_ms: int | None = None,
    ) -> dict[str, Any]:
        snapshot = await self.state_store.snapshot(MarketScope(exchange, market_type, symbol))
        trades = list(snapshot.trades.trades)
        if window_ms is not None:
            cutoff = now_ms() - int(window_ms)
            trades = [trade for trade in trades if trade.timestamp_ms >= cutoff]
        buy_volume = sum(t.quantity for t in trades if t.aggressor_side == "buy" or t.side == "buy")
        sell_volume = sum(t.quantity for t in trades if t.aggressor_side == "sell" or t.side == "sell")
        total_volume = buy_volume + sell_volume
        return {
            "exchange": snapshot.scope.exchange,
            "symbol": snapshot.scope.symbol,
            "market_type": snapshot.scope.market_type,
            "trade_count": len(trades),
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "total_volume": total_volume,
            "delta_volume": buy_volume - sell_volume,
            "last_price": trades[-1].price if trades else snapshot.last_price,
            "first_timestamp_ms": trades[0].timestamp_ms if trades else None,
            "last_timestamp_ms": trades[-1].timestamp_ms if trades else None,
        }

    async def clear_symbol(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures") -> int:
        key = MarketScope(exchange, market_type, symbol).symbol_key
        removed = 1 if getattr(self.state_store, "_states", {}).pop(key, None) is not None else 0
        return removed

    async def cleanup_stale(self) -> int:
        self._metrics["cleanup_runs"] += 1
        result = await self.state_store.cleanup_stale()
        return int(result.get("removed", 0) or 0)

    async def has_trades(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures") -> bool:
        snapshot = await self.state_store.snapshot(MarketScope(exchange, market_type, symbol))
        return bool(snapshot.trades.trades)

    async def size(self, *, exchange: str | None = None, symbol: str | None = None, market_type: str = "usdm_futures") -> int:
        if exchange and symbol:
            snapshot = await self.state_store.snapshot(MarketScope(exchange, market_type, symbol))
            return len(snapshot.trades.trades)
        stats = await self.state_store.stats()
        return int(stats.get("trades_ingested", 0) or 0)

    def stats(self) -> dict[str, Any]:
        return {**self._metrics, "ingestion": self.ingestion.stats()}
