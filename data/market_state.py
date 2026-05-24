from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field, fields
from typing import Any

from core.logger import get_logger
from data.dirty_registry import DirtySymbolRegistry
from data.market_models import (
    CandleUpdate,
    DirtyReason,
    FundingUpdate,
    LiquidationUpdate,
    MarketScope,
    OpenInterestUpdate,
    OrderBookDeltaUpdate,
    OrderBookSnapshotUpdate,
    PriceUpdate,
    TradeUpdate,
    now_ms,
)
from data.market_snapshots import (
    CandleSnapshot,
    CandlesWindowSnapshot,
    FundingSnapshot,
    LiquidationSnapshot,
    MarketSnapshot,
    OpenInterestSnapshot,
    OrderBookLevelSnapshot,
    OrderBookSnapshotView,
    TradeSnapshot,
    TradesWindowSnapshot,
)


@dataclass(slots=True)
class MarketStateConfig:
    max_trades_per_symbol: int = 5_000
    max_candles_per_scope: int = 2_000
    max_liquidations_per_symbol: int = 1_000
    max_orderbook_levels: int = 200
    trade_retention_ms: int = 15 * 60 * 1000
    liquidation_retention_ms: int = 60 * 60 * 1000
    default_market_type: str = "usdm_futures"
    service_name: str = "market_state_store"

    def validate(self) -> None:
        if self.max_trades_per_symbol <= 0:
            raise ValueError("max_trades_per_symbol must be > 0")
        if self.max_candles_per_scope <= 0:
            raise ValueError("max_candles_per_scope must be > 0")
        if self.max_orderbook_levels <= 0:
            raise ValueError("max_orderbook_levels must be > 0")


@dataclass(slots=True)
class MutableOrderBookState:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    sequence: int | None = None
    snapshot_received: bool = False
    resync_required: bool = False
    last_update_ms: int | None = None
    last_error: str | None = None
    sequence_gaps: int = 0
    updates_applied: int = 0

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.sequence = None
        self.snapshot_received = False
        self.resync_required = False
        self.last_error = None
        self.last_update_ms = None


@dataclass(slots=True)
class SymbolMarketState:
    scope: MarketScope
    trades: deque[TradeUpdate] = field(default_factory=deque)
    candles: dict[str, dict[int, CandleUpdate]] = field(default_factory=dict)
    orderbook: MutableOrderBookState = field(default_factory=MutableOrderBookState)
    funding: FundingUpdate | None = None
    open_interest: OpenInterestUpdate | None = None
    liquidations: deque[LiquidationUpdate] = field(default_factory=deque)
    last_price: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    price_source: str | None = None
    updated_at_ms: int | None = None
    dirty_reasons: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MarketStateStats:
    states_created: int = 0
    trades_ingested: int = 0
    candles_ingested: int = 0
    orderbook_deltas_ingested: int = 0
    orderbook_snapshots_ingested: int = 0
    funding_ingested: int = 0
    open_interest_ingested: int = 0
    liquidations_ingested: int = 0
    price_updates_ingested: int = 0
    orderbook_sequence_gaps: int = 0
    invalid_updates: int = 0
    snapshots_created: int = 0
    cleanup_runs: int = 0
    cleanup_removed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


class MarketStateStore:
    """
    Central state-driven market-data store.

    It is the replacement for raw market-data EventBus transport. Exchange WS
    and REST adapters should write here through MarketIngestionService. Analytics
    should read consistent snapshots at a controlled cadence and publish only
    analytics.* events through EventBus.
    """

    def __init__(
        self,
        *,
        config: MarketStateConfig | None = None,
        dirty_registry: DirtySymbolRegistry | None = None,
        service_name: str | None = None,
    ) -> None:
        self.config = config or MarketStateConfig()
        self.config.validate()
        self.dirty_registry = dirty_registry or DirtySymbolRegistry()
        self._service_name = service_name or self.config.service_name
        self._logger = get_logger(__name__, service=self._service_name, event_type="market_state")
        self._states: dict[str, SymbolMarketState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._stats = MarketStateStats()

    # ------------------------------------------------------------------
    # Update API
    # ------------------------------------------------------------------

    async def update_price(self, update: PriceUpdate) -> None:
        state = await self._get_or_create_state(update.scope)
        async with self._lock_for(update.scope.symbol_key):
            self._set_price(
                state,
                price=update.price,
                source=update.source,
                mark_price=update.mark_price,
                index_price=update.index_price,
                timestamp_ms=update.timestamp_ms,
            )
            self._stats.price_updates_ingested += 1
            state.dirty_reasons.add(DirtyReason.PRICE.value)
        await self._mark_dirty(update.scope, DirtyReason.PRICE, source=update.source)

    async def update_trade(self, update: TradeUpdate) -> None:
        state = await self._get_or_create_state(update.scope)
        async with self._lock_for(update.scope.symbol_key):
            state.trades.append(update)
            self._trim_deque_by_size(state.trades, self.config.max_trades_per_symbol)
            self._trim_deque_by_age(state.trades, self.config.trade_retention_ms)
            self._set_price(state, price=update.price, source="trade", timestamp_ms=update.timestamp_ms)
            self._stats.trades_ingested += 1
            state.dirty_reasons.add(DirtyReason.TRADE.value)
        await self._mark_dirty(update.scope, DirtyReason.TRADE, source="trade")

    async def update_trades_batch(self, updates: list[TradeUpdate]) -> None:
        for update in updates:
            await self.update_trade(update)

    async def update_candle(self, update: CandleUpdate) -> None:
        state = await self._get_or_create_state(update.scope)
        async with self._lock_for(update.scope.symbol_key):
            timeframe = update.scope.timeframe or "1m"
            bucket = state.candles.setdefault(timeframe, {})
            bucket[update.open_time_ms] = update
            self._trim_candles(bucket, self.config.max_candles_per_scope)
            source = "candle_closed" if update.is_closed else "candle"
            self._set_price(state, price=update.close, source=source, timestamp_ms=update.timestamp_ms)
            self._stats.candles_ingested += 1
            state.dirty_reasons.add(DirtyReason.CANDLE_CLOSED.value if update.is_closed else DirtyReason.CANDLE.value)
        await self._mark_dirty(
            update.scope,
            DirtyReason.CANDLE_CLOSED if update.is_closed else DirtyReason.CANDLE,
            source="candle",
            metadata={"timeframe": update.scope.timeframe, "is_closed": update.is_closed},
        )

    async def update_candles_batch(self, updates: list[CandleUpdate]) -> None:
        for update in updates:
            await self.update_candle(update)

    async def update_orderbook_snapshot(self, update: OrderBookSnapshotUpdate) -> None:
        state = await self._get_or_create_state(update.scope)
        async with self._lock_for(update.scope.symbol_key):
            book = state.orderbook
            book.bids = {price: qty for price, qty in update.bids if qty > 0}
            book.asks = {price: qty for price, qty in update.asks if qty > 0}
            book.sequence = update.last_update_id
            book.snapshot_received = True
            book.resync_required = False
            book.last_error = None
            book.last_update_ms = update.timestamp_ms
            self._trim_orderbook(book)
            mid = self._mid_price(book)
            if mid is not None:
                self._set_price(state, price=mid, source="orderbook_snapshot", timestamp_ms=update.timestamp_ms)
            self._stats.orderbook_snapshots_ingested += 1
            state.dirty_reasons.add(DirtyReason.REST_SNAPSHOT.value)
        await self._mark_dirty(update.scope, DirtyReason.REST_SNAPSHOT, source="orderbook_snapshot")

    async def update_orderbook_delta(self, update: OrderBookDeltaUpdate) -> bool:
        state = await self._get_or_create_state(update.scope)
        valid = True
        async with self._lock_for(update.scope.symbol_key):
            book = state.orderbook
            if not book.snapshot_received:
                book.resync_required = True
                book.last_error = "delta_before_snapshot"
                book.sequence_gaps += 1
                self._stats.orderbook_sequence_gaps += 1
                state.dirty_reasons.add(DirtyReason.ORDERBOOK_RESYNC_REQUIRED.value)
                valid = False
            elif not self._validate_orderbook_sequence(book, update):
                book.resync_required = True
                book.last_error = "sequence_gap"
                book.sequence_gaps += 1
                self._stats.orderbook_sequence_gaps += 1
                state.dirty_reasons.add(DirtyReason.ORDERBOOK_RESYNC_REQUIRED.value)
                valid = False
            else:
                for price, qty in update.bids:
                    if qty <= 0:
                        book.bids.pop(price, None)
                    else:
                        book.bids[price] = qty
                for price, qty in update.asks:
                    if qty <= 0:
                        book.asks.pop(price, None)
                    else:
                        book.asks[price] = qty
                if update.final_update_id is not None:
                    book.sequence = update.final_update_id
                book.last_update_ms = update.timestamp_ms
                book.last_error = None
                book.resync_required = False
                book.updates_applied += 1
                self._trim_orderbook(book)
                mid = self._mid_price(book)
                if mid is not None:
                    self._set_price(state, price=mid, source="orderbook", timestamp_ms=update.timestamp_ms)
                self._stats.orderbook_deltas_ingested += 1
                state.dirty_reasons.add(DirtyReason.ORDERBOOK.value)

        await self._mark_dirty(
            update.scope,
            DirtyReason.ORDERBOOK if valid else DirtyReason.ORDERBOOK_RESYNC_REQUIRED,
            source="orderbook",
            metadata={"valid": valid},
        )
        return valid

    async def update_funding(self, update: FundingUpdate) -> None:
        state = await self._get_or_create_state(update.scope)
        async with self._lock_for(update.scope.symbol_key):
            state.funding = update
            self._set_price(
                state,
                price=update.mark_price,
                source="funding_mark_price",
                mark_price=update.mark_price,
                index_price=update.index_price,
                timestamp_ms=update.timestamp_ms,
                allow_none_price=True,
            )
            self._stats.funding_ingested += 1
            state.dirty_reasons.add(DirtyReason.FUNDING.value)
        await self._mark_dirty(update.scope, DirtyReason.FUNDING, source="funding")

    async def update_open_interest(self, update: OpenInterestUpdate) -> None:
        state = await self._get_or_create_state(update.scope)
        async with self._lock_for(update.scope.symbol_key):
            state.open_interest = update
            self._set_price(
                state,
                price=update.mark_price,
                source="open_interest_mark_price",
                mark_price=update.mark_price,
                index_price=update.index_price,
                timestamp_ms=update.timestamp_ms,
                allow_none_price=True,
            )
            self._stats.open_interest_ingested += 1
            state.dirty_reasons.add(DirtyReason.OPEN_INTEREST.value)
        await self._mark_dirty(update.scope, DirtyReason.OPEN_INTEREST, source="open_interest")

    async def update_liquidation(self, update: LiquidationUpdate) -> None:
        state = await self._get_or_create_state(update.scope)
        async with self._lock_for(update.scope.symbol_key):
            state.liquidations.append(update)
            self._trim_deque_by_size(state.liquidations, self.config.max_liquidations_per_symbol)
            self._trim_deque_by_age(state.liquidations, self.config.liquidation_retention_ms)
            self._set_price(state, price=update.price, source="liquidation", timestamp_ms=update.timestamp_ms)
            self._stats.liquidations_ingested += 1
            state.dirty_reasons.add(DirtyReason.LIQUIDATION.value)
        await self._mark_dirty(update.scope, DirtyReason.LIQUIDATION, source="liquidation")

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    async def snapshot(self, scope: MarketScope, *, depth: int | None = None) -> MarketSnapshot:
        state = await self._get_or_create_state(scope)
        async with self._lock_for(scope.symbol_key):
            snapshot = self._build_snapshot_locked(state, depth=depth)
            self._stats.snapshots_created += 1
            return snapshot

    async def snapshots_for_dirty(self, *, limit: int | None = None, depth: int | None = None) -> list[MarketSnapshot]:
        dirty_items = await self.dirty_registry.pop_dirty(limit=limit)
        snapshots: list[MarketSnapshot] = []
        for item in dirty_items:
            snapshot = await self.snapshot(item.scope, depth=depth)
            snapshots.append(
                MarketSnapshot(
                    scope=snapshot.scope,
                    last_price=snapshot.last_price,
                    mark_price=snapshot.mark_price,
                    index_price=snapshot.index_price,
                    reference_price=snapshot.reference_price,
                    price_source=snapshot.price_source,
                    trades=snapshot.trades,
                    candles=snapshot.candles,
                    orderbook=snapshot.orderbook,
                    funding=snapshot.funding,
                    open_interest=snapshot.open_interest,
                    liquidations=snapshot.liquidations,
                    dirty_reasons=tuple(sorted(item.reasons or snapshot.dirty_reasons)),
                    updated_at_ms=snapshot.updated_at_ms,
                    metadata={**snapshot.metadata, "dirty": item.to_dict()},
                )
            )
        return snapshots

    async def latest_price(self, scope: MarketScope) -> float | None:
        state = await self._get_or_create_state(scope)
        async with self._lock_for(scope.symbol_key):
            return state.last_price or state.mark_price or state.index_price or self._mid_price(state.orderbook)

    async def cleanup_stale(self) -> dict[str, Any]:
        removed = 0
        async with self._global_lock:
            states = list(self._states.values())
        for state in states:
            async with self._lock_for(state.scope.symbol_key):
                before_trades = len(state.trades)
                before_liqs = len(state.liquidations)
                self._trim_deque_by_age(state.trades, self.config.trade_retention_ms)
                self._trim_deque_by_age(state.liquidations, self.config.liquidation_retention_ms)
                removed += before_trades - len(state.trades)
                removed += before_liqs - len(state.liquidations)
        self._stats.cleanup_runs += 1
        self._stats.cleanup_removed += removed
        return {"removed": removed, "states": len(self._states)}

    async def stats(self) -> dict[str, Any]:
        dirty_stats = await self.dirty_registry.stats()
        return {
            **self._stats.to_dict(),
            "states": len(self._states),
            "locks": len(self._locks),
            "dirty_registry": dirty_stats,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _get_or_create_state(self, scope: MarketScope) -> SymbolMarketState:
        key = scope.symbol_key
        state = self._states.get(key)
        if state is not None:
            return state
        async with self._global_lock:
            state = self._states.get(key)
            if state is None:
                state = SymbolMarketState(scope=scope.with_timeframe(None))
                self._states[key] = state
                self._stats.states_created += 1
            return state

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def _mark_dirty(self, scope: MarketScope, reason: DirtyReason, *, source: str, metadata: dict[str, Any] | None = None) -> None:
        await self.dirty_registry.mark_dirty(scope.with_timeframe(scope.timeframe), reason=reason, source=source, metadata=metadata)

    def _set_price(
        self,
        state: SymbolMarketState,
        *,
        price: float | None,
        source: str,
        timestamp_ms: int,
        mark_price: float | None = None,
        index_price: float | None = None,
        allow_none_price: bool = False,
    ) -> None:
        if price is not None and price > 0:
            state.last_price = price
            state.price_source = source
        elif not allow_none_price:
            return
        if mark_price is not None and mark_price > 0:
            state.mark_price = mark_price
        if index_price is not None and index_price > 0:
            state.index_price = index_price
        state.updated_at_ms = timestamp_ms or now_ms()

    def _validate_orderbook_sequence(self, book: MutableOrderBookState, update: OrderBookDeltaUpdate) -> bool:
        current = book.sequence
        final_update_id = update.final_update_id
        first_update_id = update.first_update_id
        previous_final_update_id = update.previous_final_update_id
        if current is None:
            return True
        if final_update_id is not None and final_update_id <= current:
            return True
        # Binance USD-M first delta after REST snapshot.
        if previous_final_update_id is None and first_update_id is not None and final_update_id is not None:
            return first_update_id <= current + 1 <= final_update_id
        # Binance USD-M continuous delta.
        if previous_final_update_id is not None:
            return previous_final_update_id == current
        return final_update_id is None or final_update_id >= current

    def _trim_orderbook(self, book: MutableOrderBookState) -> None:
        max_levels = self.config.max_orderbook_levels
        if len(book.bids) > max_levels:
            keep = dict(sorted(book.bids.items(), key=lambda item: item[0], reverse=True)[:max_levels])
            book.bids.clear()
            book.bids.update(keep)
        if len(book.asks) > max_levels:
            keep = dict(sorted(book.asks.items(), key=lambda item: item[0])[:max_levels])
            book.asks.clear()
            book.asks.update(keep)

    @staticmethod
    def _mid_price(book: MutableOrderBookState) -> float | None:
        if not book.bids or not book.asks:
            return None
        best_bid = max(book.bids)
        best_ask = min(book.asks)
        if best_bid <= 0 or best_ask <= 0:
            return None
        return (best_bid + best_ask) / 2.0

    @staticmethod
    def _trim_deque_by_size(items: deque[Any], max_items: int) -> None:
        while len(items) > max_items:
            items.popleft()

    @staticmethod
    def _trim_deque_by_age(items: deque[Any], retention_ms: int) -> None:
        cutoff = now_ms() - retention_ms
        while items:
            ts = getattr(items[0], "timestamp_ms", None)
            if ts is None or ts >= cutoff:
                break
            items.popleft()

    @staticmethod
    def _trim_candles(candles: dict[int, CandleUpdate], max_items: int) -> None:
        while len(candles) > max_items:
            oldest = min(candles)
            candles.pop(oldest, None)

    def _build_snapshot_locked(self, state: SymbolMarketState, *, depth: int | None = None) -> MarketSnapshot:
        trades = tuple(
            TradeSnapshot(
                price=item.price,
                quantity=item.quantity,
                side=item.side,
                aggressor_side=item.aggressor_side,
                timestamp_ms=item.timestamp_ms,
                trade_id=item.trade_id,
                metadata=dict(item.metadata),
            )
            for item in state.trades
        )
        buy_volume = sum(item.quantity for item in trades if item.aggressor_side == "buy" or item.side == "buy")
        sell_volume = sum(item.quantity for item in trades if item.aggressor_side == "sell" or item.side == "sell")
        trades_snapshot = TradesWindowSnapshot(
            trades=trades,
            last_price=trades[-1].price if trades else state.last_price,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            total_volume=buy_volume + sell_volume,
            trade_count=len(trades),
            first_timestamp_ms=trades[0].timestamp_ms if trades else None,
            last_timestamp_ms=trades[-1].timestamp_ms if trades else None,
        )

        candles_snapshot: dict[str, CandlesWindowSnapshot] = {}
        for timeframe, bucket in state.candles.items():
            ordered = [bucket[key] for key in sorted(bucket)]
            candles = tuple(
                CandleSnapshot(
                    timeframe=timeframe,
                    open_time_ms=item.open_time_ms,
                    close_time_ms=item.close_time_ms,
                    open=item.open,
                    high=item.high,
                    low=item.low,
                    close=item.close,
                    volume=item.volume,
                    is_closed=item.is_closed,
                    timestamp_ms=item.timestamp_ms,
                    metadata=dict(item.metadata),
                )
                for item in ordered
            )
            last_closed = next((item for item in reversed(candles) if item.is_closed), None)
            candles_snapshot[timeframe] = CandlesWindowSnapshot(
                timeframe=timeframe,
                candles=candles,
                last_close=candles[-1].close if candles else None,
                last_closed=last_closed,
                candle_count=len(candles),
            )

        depth_limit = depth or self.config.max_orderbook_levels
        bid_items = sorted(state.orderbook.bids.items(), key=lambda item: item[0], reverse=True)[:depth_limit]
        ask_items = sorted(state.orderbook.asks.items(), key=lambda item: item[0])[:depth_limit]
        best_bid = bid_items[0][0] if bid_items else None
        best_ask = ask_items[0][0] if ask_items else None
        mid_price = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else None
        orderbook_snapshot = OrderBookSnapshotView(
            bids=tuple(OrderBookLevelSnapshot(price=price, quantity=qty) for price, qty in bid_items),
            asks=tuple(OrderBookLevelSnapshot(price=price, quantity=qty) for price, qty in ask_items),
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=(best_ask - best_bid) if best_bid is not None and best_ask is not None else None,
            sequence=state.orderbook.sequence,
            snapshot_received=state.orderbook.snapshot_received,
            resync_required=state.orderbook.resync_required,
            last_update_ms=state.orderbook.last_update_ms,
            last_error=state.orderbook.last_error,
        )

        funding = None
        if state.funding is not None:
            funding = FundingSnapshot(
                funding_rate=state.funding.funding_rate,
                next_funding_time_ms=state.funding.next_funding_time_ms,
                mark_price=state.funding.mark_price,
                index_price=state.funding.index_price,
                timestamp_ms=state.funding.timestamp_ms,
                metadata=dict(state.funding.metadata),
            )
        open_interest = None
        if state.open_interest is not None:
            open_interest = OpenInterestSnapshot(
                open_interest=state.open_interest.open_interest,
                open_interest_value=state.open_interest.open_interest_value,
                mark_price=state.open_interest.mark_price,
                index_price=state.open_interest.index_price,
                timestamp_ms=state.open_interest.timestamp_ms,
                metadata=dict(state.open_interest.metadata),
            )
        liquidations = tuple(
            LiquidationSnapshot(
                price=item.price,
                quantity=item.quantity,
                side=item.side,
                timestamp_ms=item.timestamp_ms,
                order_id=item.order_id,
                metadata=dict(item.metadata),
            )
            for item in state.liquidations
        )
        reference_price = state.last_price or state.mark_price or mid_price or state.index_price
        return MarketSnapshot(
            scope=state.scope,
            last_price=state.last_price,
            mark_price=state.mark_price,
            index_price=state.index_price,
            reference_price=reference_price,
            price_source=state.price_source,
            trades=trades_snapshot,
            candles=candles_snapshot,
            orderbook=orderbook_snapshot,
            funding=funding,
            open_interest=open_interest,
            liquidations=liquidations,
            dirty_reasons=tuple(sorted(state.dirty_reasons)),
            updated_at_ms=state.updated_at_ms,
            metadata=dict(state.metadata),
        )