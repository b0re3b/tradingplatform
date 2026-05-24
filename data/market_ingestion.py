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

    # Low-frequency storage bridge.  The state-driven ingestion path writes to
    # MarketStateStore directly, so storage must receive explicit persistable
    # batch events instead of relying on legacy raw market.* EventBus traffic.
    # These events are intentionally coarse grained and are suitable for
    # ParquetStorage; they must not be used as per-tick analytics triggers.
    emit_persistable_events: bool = True

    # Trades can be extremely high volume.  Keep persistence opt-in unless the
    # deployment has enough disk/IO budget or uses coalesced trade batches.
    emit_trade_persistable_events: bool = False

    # Orderbook snapshots can still be large, but they are snapshots rather than
    # deltas and are useful for liquidity reconstruction.
    emit_orderbook_snapshot_persistable_events: bool = True

    # Parquet/bootstrap restore feeds persisted data through this same ingestion
    # boundary to hydrate MarketStateStore. Those replayed records must not be
    # persisted again, otherwise every restart duplicates historical parquet rows.
    suppress_persistable_for_restore_sources: bool = True
    restore_sources: tuple[str, ...] = ("parquet_loader", "parquet_restore", "storage.parquet_loader")


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
            "persistable_events_emitted": 0,
            "persistable_events_failed": 0,
        }

    async def ingest_price(self, payload: Mapping[str, Any] | PriceUpdate) -> bool:
        update = payload if isinstance(payload, PriceUpdate) else self._price_from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_price(update)
        return True

    async def ingest_trade(
        self,
        payload: Mapping[str, Any] | TradeUpdate,
        *,
        source: str | None = None,
        suppress_persistable_events: bool | None = None,
    ) -> bool:
        suppress_persistable = self._should_suppress_persistable(payload, source, suppress_persistable_events)
        update = payload if isinstance(payload, TradeUpdate) else TradeUpdate.from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_trade(update)
        self._metrics["trades"] += 1
        if self.config.emit_trade_persistable_events and not suppress_persistable:
            await self._emit_trades_persistable([update], batch_source="single")
        return True

    async def ingest_trades_batch(
        self,
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any] | TradeUpdate],
        *,
        source: str | None = None,
        suppress_persistable_events: bool | None = None,
    ) -> int:
        suppress_persistable = self._should_suppress_persistable(payload, source, suppress_persistable_events)
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
        if self.config.emit_trade_persistable_events and not suppress_persistable:
            await self._emit_trades_persistable(updates, batch_source="batch")
        return len(updates)

    async def ingest_candle(
        self,
        payload: Mapping[str, Any] | CandleUpdate,
        *,
        source: str | None = None,
        suppress_persistable_events: bool | None = None,
    ) -> bool:
        suppress_persistable = self._should_suppress_persistable(payload, source, suppress_persistable_events)
        update = payload if isinstance(payload, CandleUpdate) else CandleUpdate.from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_candle(update)
        self._metrics["candles"] += 1
        if (
            update.is_closed
            and self.config.emit_low_frequency_events
            and self.config.emit_candle_closed_event
            and not self._is_restore_source(payload, source)
        ):
            await self._emit_candle_closed(update)
        return True

    async def ingest_candles_batch(
        self,
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any] | CandleUpdate],
        *,
        source: str | None = None,
        suppress_persistable_events: bool | None = None,
    ) -> int:
        suppress_persistable = self._should_suppress_persistable(payload, source, suppress_persistable_events)
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
            and not self._is_restore_source(payload, source)
        ):
            for update in updates:
                if update.is_closed:
                    await self._emit_candle_closed(update)
        if not suppress_persistable:
            await self._emit_candles_persistable(updates, batch_source="batch")
        return len(updates)

    async def ingest_orderbook_snapshot(
        self,
        payload: Mapping[str, Any] | OrderBookSnapshotUpdate,
        *,
        source: str | None = None,
        suppress_persistable_events: bool | None = None,
    ) -> bool:
        suppress_persistable = self._should_suppress_persistable(payload, source, suppress_persistable_events)
        update = payload if isinstance(payload, OrderBookSnapshotUpdate) else OrderBookSnapshotUpdate.from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_orderbook_snapshot(update)
        self._metrics["orderbook_snapshots"] += 1
        if self.config.emit_orderbook_snapshot_persistable_events and not suppress_persistable:
            await self._emit_orderbook_snapshot_persistable(update)
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

    async def ingest_funding(
        self,
        payload: Mapping[str, Any] | FundingUpdate,
        *,
        source: str | None = None,
        suppress_persistable_events: bool | None = None,
    ) -> bool:
        suppress_persistable = self._should_suppress_persistable(payload, source, suppress_persistable_events)
        update = payload if isinstance(payload, FundingUpdate) else FundingUpdate.from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_funding(update)
        self._metrics["funding"] += 1
        if not suppress_persistable:
            await self._emit_funding_persistable(update)
        return True

    async def ingest_open_interest(
        self,
        payload: Mapping[str, Any] | OpenInterestUpdate,
        *,
        source: str | None = None,
        suppress_persistable_events: bool | None = None,
    ) -> bool:
        suppress_persistable = self._should_suppress_persistable(payload, source, suppress_persistable_events)
        update = payload if isinstance(payload, OpenInterestUpdate) else OpenInterestUpdate.from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_open_interest(update)
        self._metrics["open_interest"] += 1
        if not suppress_persistable:
            await self._emit_open_interest_persistable(update)
        return True

    async def ingest_liquidation(
        self,
        payload: Mapping[str, Any] | LiquidationUpdate,
        *,
        source: str | None = None,
        suppress_persistable_events: bool | None = None,
    ) -> bool:
        suppress_persistable = self._should_suppress_persistable(payload, source, suppress_persistable_events)
        update = payload if isinstance(payload, LiquidationUpdate) else LiquidationUpdate.from_payload(payload)
        if update is None:
            self._metrics["invalid_payloads"] += 1
            return False
        await self.state_store.update_liquidation(update)
        self._metrics["liquidations"] += 1
        if not suppress_persistable:
            await self._emit_liquidations_persistable([update], batch_source="single")
        return True

    async def ingest_orderbook_snapshots_batch(
        self,
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any] | OrderBookSnapshotUpdate],
        *,
        source: str | None = None,
        suppress_persistable_events: bool | None = None,
    ) -> int:
        items = self._extract_batch_items(payload, "orderbooks", "orderbook_snapshots", "snapshots", "items", "data")
        count = 0
        for item in items:
            ok = await self.ingest_orderbook_snapshot(
                item,
                source=source,
                suppress_persistable_events=suppress_persistable_events,
            )
            if ok:
                count += 1
        if count <= 0:
            self._metrics["invalid_payloads"] += 1
        return count

    async def ingest_funding_batch(
        self,
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any] | FundingUpdate],
        *,
        source: str | None = None,
        suppress_persistable_events: bool | None = None,
    ) -> int:
        items = self._extract_batch_items(payload, "funding", "rates", "items", "data")
        count = 0
        for item in items:
            ok = await self.ingest_funding(
                item,
                source=source,
                suppress_persistable_events=suppress_persistable_events,
            )
            if ok:
                count += 1
        if count <= 0:
            self._metrics["invalid_payloads"] += 1
        return count

    async def ingest_open_interest_batch(
        self,
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any] | OpenInterestUpdate],
        *,
        source: str | None = None,
        suppress_persistable_events: bool | None = None,
    ) -> int:
        items = self._extract_batch_items(payload, "open_interest", "open_interests", "items", "data")
        count = 0
        for item in items:
            ok = await self.ingest_open_interest(
                item,
                source=source,
                suppress_persistable_events=suppress_persistable_events,
            )
            if ok:
                count += 1
        if count <= 0:
            self._metrics["invalid_payloads"] += 1
        return count

    async def ingest_liquidations_batch(
        self,
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any] | LiquidationUpdate],
        *,
        source: str | None = None,
        suppress_persistable_events: bool | None = None,
    ) -> int:
        items = self._extract_batch_items(payload, "liquidations", "items", "data")
        count = 0
        for item in items:
            ok = await self.ingest_liquidation(
                item,
                source=source,
                suppress_persistable_events=suppress_persistable_events,
            )
            if ok:
                count += 1
        if count <= 0:
            self._metrics["invalid_payloads"] += 1
        return count

    async def snapshot(self, scope: MarketScope):
        return await self.state_store.snapshot(scope)

    def stats(self) -> dict[str, int]:
        return dict(self._metrics)


    def _is_restore_source(self, payload: Any, source: str | None) -> bool:
        if not self.config.suppress_persistable_for_restore_sources:
            return False
        candidates: list[str] = []
        if source:
            candidates.append(str(source))
        if isinstance(payload, Mapping):
            for key in ("source", "batch_source", "replay_source", "restore_source"):
                value = payload.get(key)
                if value:
                    candidates.append(str(value))
            metadata = payload.get("metadata")
            if isinstance(metadata, Mapping):
                for key in ("source", "replay_source", "restore_source"):
                    value = metadata.get(key)
                    if value:
                        candidates.append(str(value))
        restore_sources = {str(item).lower() for item in self.config.restore_sources}
        return any(candidate.lower() in restore_sources for candidate in candidates)

    def _should_suppress_persistable(
        self,
        payload: Any,
        source: str | None,
        explicit: bool | None,
    ) -> bool:
        if explicit is not None:
            return bool(explicit)
        if not self.config.suppress_persistable_for_restore_sources:
            return False
        candidates: list[str] = []
        if source:
            candidates.append(str(source))
        if isinstance(payload, Mapping):
            for key in ("source", "batch_source", "replay_source", "restore_source"):
                value = payload.get(key)
                if value:
                    candidates.append(str(value))
            metadata = payload.get("metadata")
            if isinstance(metadata, Mapping):
                for key in ("source", "replay_source", "restore_source"):
                    value = metadata.get(key)
                    if value:
                        candidates.append(str(value))
        restore_sources = {str(item).lower() for item in self.config.restore_sources}
        return any(candidate.lower() in restore_sources for candidate in candidates)

    def _extract_batch_items(self, payload: Any, *keys: str) -> list[Any]:
        if isinstance(payload, Mapping):
            for key in keys:
                value = payload.get(key)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                    parent = dict(payload)
                    for remove_key in keys:
                        parent.pop(remove_key, None)
                    result: list[Any] = []
                    for item in value:
                        if isinstance(item, Mapping):
                            result.append({**parent, **dict(item)})
                        else:
                            result.append(item)
                    return result
            return [payload]
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            return list(payload)
        return []

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

    def _scope_payload(self, scope: MarketScope) -> dict[str, Any]:
        return scope.to_dict()

    def _candle_payload(self, update: CandleUpdate) -> dict[str, Any]:
        return {
            **self._scope_payload(update.scope),
            "open_time_ms": update.open_time_ms,
            "close_time_ms": update.close_time_ms,
            "open": update.open,
            "high": update.high,
            "low": update.low,
            "close": update.close,
            "volume": update.volume,
            "is_closed": update.is_closed,
            "timestamp_ms": update.timestamp_ms,
            "received_at_ms": update.received_at_ms,
            "metadata": dict(update.metadata),
        }

    def _trade_payload(self, update: TradeUpdate) -> dict[str, Any]:
        return {
            **self._scope_payload(update.scope),
            "trade_id": update.trade_id,
            "price": update.price,
            "quantity": update.quantity,
            "side": update.side,
            "aggressor_side": update.aggressor_side,
            "timestamp_ms": update.timestamp_ms,
            "received_at_ms": update.received_at_ms,
            "metadata": dict(update.metadata),
        }

    def _orderbook_snapshot_payload(self, update: OrderBookSnapshotUpdate) -> dict[str, Any]:
        return {
            **self._scope_payload(update.scope),
            "bids": list(update.bids),
            "asks": list(update.asks),
            "last_update_id": update.last_update_id,
            "timestamp_ms": update.timestamp_ms,
            "received_at_ms": update.received_at_ms,
            "metadata": dict(update.metadata),
        }

    def _funding_payload(self, update: FundingUpdate) -> dict[str, Any]:
        return {
            **self._scope_payload(update.scope),
            "funding_rate": update.funding_rate,
            "next_funding_time_ms": update.next_funding_time_ms,
            "mark_price": update.mark_price,
            "index_price": update.index_price,
            "timestamp_ms": update.timestamp_ms,
            "received_at_ms": update.received_at_ms,
            "metadata": dict(update.metadata),
        }

    def _open_interest_payload(self, update: OpenInterestUpdate) -> dict[str, Any]:
        return {
            **self._scope_payload(update.scope),
            "open_interest": update.open_interest,
            "open_interest_value": update.open_interest_value,
            "mark_price": update.mark_price,
            "index_price": update.index_price,
            "timestamp_ms": update.timestamp_ms,
            "received_at_ms": update.received_at_ms,
            "metadata": dict(update.metadata),
        }

    def _liquidation_payload(self, update: LiquidationUpdate) -> dict[str, Any]:
        notional = update.metadata.get("notional_usd")
        if notional is None and update.price is not None and update.quantity is not None:
            notional = float(update.price) * float(update.quantity)
        return {
            **self._scope_payload(update.scope),
            "order_id": update.order_id,
            "price": update.price,
            "quantity": update.quantity,
            "side": update.side,
            "liquidation_side": update.metadata.get("liquidation_side") or update.metadata.get("raw_side") or update.side,
            "notional_usd": notional,
            "notional": notional,
            "timestamp_ms": update.timestamp_ms,
            "received_at_ms": update.received_at_ms,
            "metadata": dict(update.metadata),
        }

    def _batch_scope_payload(self, updates: Sequence[Any]) -> dict[str, Any]:
        if not updates:
            return {}
        first = updates[0]
        scope = getattr(first, "scope", None)
        if isinstance(scope, MarketScope):
            return self._scope_payload(scope)
        return {}

    async def _emit_persistable_event(
        self,
        topic: str,
        payload: Mapping[str, Any],
        *,
        priority: EventPriority = EventPriority.LOW,
    ) -> None:
        if self.event_bus is None or not self.config.emit_persistable_events:
            return
        try:
            accepted = await self.event_bus.emit(
                topic,
                dict(payload),
                priority=priority,
                source=self._service_name,
            )
        except RuntimeError:
            # EventBus may not be started during unit tests or isolated warmup
            # probes. StateStore has already been updated, so persistence can be
            # retried by restore/warmup without breaking ingestion.
            self._metrics["persistable_events_failed"] += 1
            return
        except Exception:
            self._metrics["persistable_events_failed"] += 1
            self._logger.exception("Failed to emit persistable market event | topic=%s", topic)
            return
        if accepted:
            self._metrics["persistable_events_emitted"] += 1
        else:
            self._metrics["persistable_events_failed"] += 1

    async def _emit_candles_persistable(
        self,
        updates: Sequence[CandleUpdate],
        *,
        batch_source: str,
    ) -> None:
        closed = [update for update in updates if update.is_closed]
        if not closed:
            return
        payload = {
            **self._batch_scope_payload(closed),
            "candles": [self._candle_payload(update) for update in closed],
            "count": len(closed),
            "batch_source": batch_source,
            "timestamp_ms": max(update.timestamp_ms for update in closed),
            "received_at_ms": max(update.received_at_ms for update in closed),
            "source": "market_ingestion",
        }
        await self._emit_persistable_event("market.candles.persistable", payload)

    async def _emit_trades_persistable(
        self,
        updates: Sequence[TradeUpdate],
        *,
        batch_source: str,
    ) -> None:
        if not updates:
            return
        payload = {
            **self._batch_scope_payload(updates),
            "trades": [self._trade_payload(update) for update in updates],
            "count": len(updates),
            "batch_source": batch_source,
            "timestamp_ms": max(update.timestamp_ms for update in updates),
            "received_at_ms": max(update.received_at_ms for update in updates),
            "source": "market_ingestion",
        }
        await self._emit_persistable_event("market.trades.persistable", payload)

    async def _emit_orderbook_snapshot_persistable(self, update: OrderBookSnapshotUpdate) -> None:
        payload = {
            **self._orderbook_snapshot_payload(update),
            "orderbook": self._orderbook_snapshot_payload(update),
            "source": "market_ingestion",
        }
        await self._emit_persistable_event("market.orderbook.snapshot.persistable", payload)

    async def _emit_funding_persistable(self, update: FundingUpdate) -> None:
        payload = {
            **self._funding_payload(update),
            "funding": self._funding_payload(update),
            "source": "market_ingestion",
        }
        await self._emit_persistable_event("market.funding.persistable", payload)

    async def _emit_open_interest_persistable(self, update: OpenInterestUpdate) -> None:
        payload = {
            **self._open_interest_payload(update),
            "open_interest": self._open_interest_payload(update),
            "source": "market_ingestion",
        }
        await self._emit_persistable_event("market.open_interest.persistable", payload)

    async def _emit_liquidations_persistable(
        self,
        updates: Sequence[LiquidationUpdate],
        *,
        batch_source: str,
    ) -> None:
        if not updates:
            return
        payload = {
            **self._batch_scope_payload(updates),
            "liquidations": [self._liquidation_payload(update) for update in updates],
            "count": len(updates),
            "batch_source": batch_source,
            "timestamp_ms": max(update.timestamp_ms for update in updates),
            "received_at_ms": max(update.received_at_ms for update in updates),
            "source": "market_ingestion",
        }
        await self._emit_persistable_event("market.liquidations.persistable", payload)

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