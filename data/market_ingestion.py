from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from data.market_models import (
    CandleUpdate,
    FundingUpdate,
    LiquidationUpdate,
    MarketScope,
    OpenInterestUpdate,
    OrderBookDeltaUpdate,
    OrderBookSnapshotUpdate,
    PriceUpdate,
    TradeUpdate,
    first_present,
    now_ms,
    safe_float,
    safe_int,
)
from data.market_state import MarketStateStore


@dataclass(slots=True)
class MarketIngestionConfig:
    emit_low_frequency_events: bool = True
    emit_candle_closed_event: bool = True
    emit_orderbook_resync_event: bool = True
    emit_state_health_events: bool = False
    default_exchange: str = "binance"
    default_market_type: str = "usdm_futures"
    default_timeframe: str = "1m"
    service_name: str = "market_ingestion"
    # When True, ingest_candles_batch() will NOT emit market.candle.closed for
    # any candle in the batch.  Set this during historical warmup to prevent
    # thousands of closed-candle events flooding the EventBus.
    suppress_batch_candle_events: bool = True


class MarketIngestionService:
    """
    Single write boundary for state-driven market data.

    Exchange WS/REST/warmup code should call this service instead of publishing
    high-frequency raw market events to EventBus. The service normalizes payloads,
    writes to MarketStateStore, and optionally emits only low-frequency triggers
    such as market.candle.closed or market.orderbook.resync_required.
    """

    def __init__(
        self,
        *,
        state_store: MarketStateStore,
        event_bus: EventBus | None = None,
        config: MarketIngestionConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        self.state_store = state_store
        self.event_bus = event_bus
        self.config = config or MarketIngestionConfig()
        self._service_name = service_name or self.config.service_name
        self._logger = get_logger(__name__, service=self._service_name, event_type="market_ingestion")
        self._metrics: dict[str, int] = {
            "trades": 0,
            "trade_batches": 0,
            "candles": 0,
            "candle_batches": 0,
            "orderbook_deltas": 0,
            "orderbook_snapshots": 0,
            "funding": 0,
            "open_interest": 0,
            "liquidations": 0,
            "invalid_payloads": 0,
            "resync_required": 0,
        }

    async def ingest_price(self, payload: Mapping[str, Any] | PriceUpdate) -> bool:
        update = payload if isinstance(payload, PriceUpdate) else self._price_from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_price(update)
        return True

    async def ingest_trade(self, payload: Mapping[str, Any] | TradeUpdate) -> bool:
        update = payload if isinstance(payload, TradeUpdate) else TradeUpdate.from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_trade(update)
        self._metrics["trades"] += 1
        return True

    async def ingest_trades_batch(self, payload: Mapping[str, Any] | Sequence[Mapping[str, Any] | TradeUpdate]) -> int:
        updates: list[TradeUpdate] = []
        if isinstance(payload, Mapping):
            items = payload.get("trades") or payload.get("items") or payload.get("data") or []
            if not isinstance(items, Sequence):
                items = []
            for item in items:
                if isinstance(item, TradeUpdate):
                    updates.append(item)
                elif isinstance(item, Mapping):
                    merged = {**payload, **item}
                    for key in ("trades", "items", "data"):
                        merged.pop(key, None)
                    update = TradeUpdate.from_payload(merged)
                    if update is not None:
                        updates.append(update)
        else:
            for item in payload:
                if isinstance(item, TradeUpdate):
                    updates.append(item)
                elif isinstance(item, Mapping):
                    update = TradeUpdate.from_payload(item)
                    if update is not None:
                        updates.append(update)
        if not updates:
            self._metrics["invalid_payloads"] += 1
            return 0
        await self.state_store.update_trades_batch(updates)
        self._metrics["trade_batches"] += 1
        self._metrics["trades"] += len(updates)
        return len(updates)

    async def ingest_candle(self, payload: Mapping[str, Any] | CandleUpdate) -> bool:
        update = payload if isinstance(payload, CandleUpdate) else CandleUpdate.from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_candle(update)
        self._metrics["candles"] += 1
        if update.is_closed and self.config.emit_low_frequency_events and self.config.emit_candle_closed_event:
            await self._emit_candle_closed(update)
        return True

    async def ingest_candles_batch(self, payload: Mapping[str, Any] | Sequence[Mapping[str, Any] | CandleUpdate]) -> int:
        updates: list[CandleUpdate] = []
        if isinstance(payload, Mapping):
            items = payload.get("candles") or payload.get("items") or payload.get("data") or []
            if not isinstance(items, Sequence):
                items = []
            for item in items:
                if isinstance(item, CandleUpdate):
                    updates.append(item)
                elif isinstance(item, Mapping):
                    merged = {**payload, **item}
                    for key in ("candles", "items", "data"):
                        merged.pop(key, None)
                    update = CandleUpdate.from_payload(merged)
                    if update is not None:
                        updates.append(update)
        else:
            for item in payload:
                if isinstance(item, CandleUpdate):
                    updates.append(item)
                elif isinstance(item, Mapping):
                    update = CandleUpdate.from_payload(item)
                    if update is not None:
                        updates.append(update)
        if not updates:
            self._metrics["invalid_payloads"] += 1
            return 0
        await self.state_store.update_candles_batch(updates)
        self._metrics["candle_batches"] += 1
        self._metrics["candles"] += len(updates)
        if (
            self.config.emit_low_frequency_events
            and self.config.emit_candle_closed_event
            and not self.config.suppress_batch_candle_events
        ):
            for update in updates:
                if update.is_closed:
                    await self._emit_candle_closed(update)
        return len(updates)

    async def ingest_orderbook_snapshot(self, payload: Mapping[str, Any] | OrderBookSnapshotUpdate) -> bool:
        update = payload if isinstance(payload, OrderBookSnapshotUpdate) else OrderBookSnapshotUpdate.from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_orderbook_snapshot(update)
        self._metrics["orderbook_snapshots"] += 1
        return True

    async def ingest_orderbook_delta(self, payload: Mapping[str, Any] | OrderBookDeltaUpdate) -> bool:
        update = payload if isinstance(payload, OrderBookDeltaUpdate) else OrderBookDeltaUpdate.from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        valid = await self.state_store.update_orderbook_delta(update)
        self._metrics["orderbook_deltas"] += 1
        if not valid:
            self._metrics["resync_required"] += 1
            if self.config.emit_low_frequency_events and self.config.emit_orderbook_resync_event:
                await self._emit_orderbook_resync_required(update)
        return valid

    async def ingest_funding(self, payload: Mapping[str, Any] | FundingUpdate) -> bool:
        update = payload if isinstance(payload, FundingUpdate) else FundingUpdate.from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_funding(update)
        self._metrics["funding"] += 1
        return True

    async def ingest_open_interest(self, payload: Mapping[str, Any] | OpenInterestUpdate) -> bool:
        update = payload if isinstance(payload, OpenInterestUpdate) else OpenInterestUpdate.from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_open_interest(update)
        self._metrics["open_interest"] += 1
        return True

    async def ingest_liquidation(self, payload: Mapping[str, Any] | LiquidationUpdate) -> bool:
        update = payload if isinstance(payload, LiquidationUpdate) else LiquidationUpdate.from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_liquidation(update)
        self._metrics["liquidations"] += 1
        return True

    async def snapshot(self, scope: MarketScope):
        return await self.state_store.snapshot(scope)

    def stats(self) -> dict[str, int]:
        return dict(self._metrics)

    def _price_from_payload(self, payload: Mapping[str, Any]) -> PriceUpdate | None:
        scope = MarketScope.from_payload(
            payload,
            default_exchange=self.config.default_exchange,
            default_market_type=self.config.default_market_type,
        )
        price = safe_float(first_present(payload, "current_price", "last_price", "price", "mark_price", "close"))
        if not scope.symbol or price is None or price <= 0:
            return None
        return PriceUpdate(
            scope=scope,
            price=price,
            source=str(first_present(payload, "price_source", "source") or "price"),
            timestamp_ms=safe_int(first_present(payload, "timestamp_ms", "event_time", "timestamp"), now_ms()) or now_ms(),
            received_at_ms=safe_int(payload.get("received_at_ms"), now_ms()) or now_ms(),
            mark_price=safe_float(first_present(payload, "mark_price", "markPrice")),
            index_price=safe_float(first_present(payload, "index_price", "indexPrice")),
            metadata=dict(payload.get("metadata") or {}),
        )

    async def _emit_candle_closed(self, update: CandleUpdate) -> None:
        if self.event_bus is None:
            return
        await self.event_bus.emit(
            "market.candle.closed",
            {
                **update.scope.to_dict(),
                "open_time_ms": update.open_time_ms,
                "close_time_ms": update.close_time_ms,
                "open": update.open,
                "high": update.high,
                "low": update.low,
                "close": update.close,
                "volume": update.volume,
                "timestamp_ms": update.timestamp_ms,
                "source": "market_ingestion",
            },
            priority=EventPriority.NORMAL,
            source=self._service_name,
        )

    async def _emit_orderbook_resync_required(self, update: OrderBookDeltaUpdate) -> None:
        if self.event_bus is None:
            return
        await self.event_bus.emit(
            "market.orderbook.resync_required",
            {
                **update.scope.to_dict(),
                "first_update_id": update.first_update_id,
                "final_update_id": update.final_update_id,
                "previous_final_update_id": update.previous_final_update_id,
                "timestamp_ms": update.timestamp_ms,
                "reason": "sequence_gap_or_delta_before_snapshot",
                "source": "market_ingestion",
            },
            priority=EventPriority.HIGH,
            source=self._service_name,
        )