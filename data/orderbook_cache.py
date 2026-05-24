from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from core.config import Config
from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler
from data.market_ingestion import MarketIngestionService
from data.market_models import MarketScope, OrderBookDeltaUpdate, OrderBookSnapshotUpdate
from data.market_state import MarketStateStore


@dataclass(slots=True)
class OrderBookState:
    exchange: str
    symbol: str
    market_type: str = "usdm_futures"
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    sequence: int | None = None
    snapshot_received: bool = False
    resync_required: bool = False
    last_update_ms: int | None = None
    last_error: str | None = None


class OrderBookCache:
    """
    State-driven order book facade.

    Direct apply only. It does not subscribe to market.orderbook.* and does not emit
    market.orderbook.updated. Binance USD-M sequence validation is delegated to
    MarketStateStore, which applies snapshot lastUpdateId and delta U/u/pu rules.
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
        max_depth_per_side: int = 200,
        service_name: str = "orderbook_cache",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.max_depth_per_side = int(max_depth_per_side)
        self._service_name = service_name
        self._logger = get_logger(__name__, service=service_name, event_type="orderbook_cache")
        self.state_store = market_state or state_store or (ingestion.state_store if ingestion is not None else MarketStateStore())
        self.ingestion = ingestion or MarketIngestionService(state_store=self.state_store, event_bus=event_bus)
        self._registered = False
        self._metrics: dict[str, int | float] = {
            "snapshots_applied": 0,
            "deltas_applied": 0,
            "sequence_gaps": 0,
            "invalid_events": 0,
            "direct_apply_mode": 1,
        }

    def register(self) -> None:
        if self._registered:
            return
        self._registered = True
        self._logger.info("OrderBookCache registered in state-driven mode | raw_eventbus_subscriptions=0")

    async def start(self) -> None:
        self.register()

    async def stop(self) -> None:
        return None

    async def apply_snapshot(self, event: Mapping[str, Any] | OrderBookSnapshotUpdate) -> None:
        ok = await self.ingestion.ingest_orderbook_snapshot(event)
        if ok:
            self._metrics["snapshots_applied"] += 1
        else:
            self._metrics["invalid_events"] += 1

    async def apply_delta(self, event: Mapping[str, Any] | OrderBookDeltaUpdate) -> bool:
        valid = await self.ingestion.ingest_orderbook_delta(event)
        self._metrics["deltas_applied"] += 1
        if not valid:
            self._metrics["sequence_gaps"] += 1
        return valid

    async def reset_book(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures") -> bool:
        key = MarketScope(exchange, market_type, symbol).symbol_key
        state = getattr(self.state_store, "_states", {}).get(key)
        if state is None:
            return False
        state.orderbook.clear()
        return True

    async def mark_for_resync(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures", reason: str | None = None) -> None:
        key = MarketScope(exchange, market_type, symbol).symbol_key
        state = getattr(self.state_store, "_states", {}).get(key)
        if state is not None:
            state.orderbook.resync_required = True
            state.orderbook.last_error = reason or "manual_resync_required"

    async def get_book(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures", depth: int | None = None) -> dict[str, Any] | None:
        snapshot = await self.state_store.snapshot(MarketScope(exchange, market_type, symbol), depth=depth or self.max_depth_per_side)
        if snapshot.orderbook is None or not snapshot.orderbook.snapshot_received:
            return None
        return {
            **snapshot.scope.to_dict(),
            **snapshot.orderbook.to_dict(),
            "current_price": snapshot.current_price,
            "last_price": snapshot.last_price,
            "price": snapshot.current_price,
        }

    async def get_top_of_book(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures") -> dict[str, Any] | None:
        book = await self.get_book(exchange=exchange, symbol=symbol, market_type=market_type, depth=1)
        if book is None:
            return None
        return {
            "exchange": exchange,
            "symbol": symbol,
            "market_type": market_type,
            "best_bid": book.get("best_bid"),
            "best_ask": book.get("best_ask"),
            "mid_price": book.get("mid_price"),
            "spread": book.get("spread"),
            "sequence": book.get("sequence"),
            "resync_required": book.get("resync_required"),
        }

    async def has_book(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures") -> bool:
        book = await self.get_book(exchange=exchange, symbol=symbol, market_type=market_type, depth=1)
        return book is not None

    async def is_synced(self, *, exchange: str, symbol: str, market_type: str = "usdm_futures") -> bool:
        book = await self.get_book(exchange=exchange, symbol=symbol, market_type=market_type, depth=1)
        return bool(book and book.get("snapshot_received") and not book.get("resync_required"))

    def stats(self) -> dict[str, Any]:
        return {**self._metrics, "ingestion": self.ingestion.stats()}
