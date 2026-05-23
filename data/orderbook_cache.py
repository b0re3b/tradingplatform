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
class OrderBookLevel:
    price: float
    quantity: float


@dataclass(slots=True)
class OrderBookState:
    exchange: str
    symbol: str
    market_type: str = "perpetual"

    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)

    sequence: int | None = None
    prev_sequence: int | None = None

    is_synced: bool = False
    snapshot_received: bool = False
    last_update_ts_ms: int | None = None
    last_update_received_ms: int | None = None
    last_error: str | None = None

    updates_applied: int = 0
    snapshots_applied: int = 0
    sequence_gaps: int = 0
    dropped_updates: int = 0


class OrderBookCache:
    """
    Локальний кеш стакану.

    Відповідальність:
    - зберігати актуальний order book state по символах
    - приймати snapshot
    - застосовувати delta updates
    - перевіряти sequence continuity
    - підтримувати top-of-book / limited depth views
    - емінити системні події при gap/resync/reset
    """

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        cleanup_interval_seconds: float = 60.0,
        max_depth_per_side: int = 2000,
        service_name: str = "orderbook_cache",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._cleanup_job_id: str | None = None
        self.max_depth_per_side = max_depth_per_side
        self._service_name = service_name

        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="orderbook_cache",
        )

        self._books: dict[str, OrderBookState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

        self._metrics: dict[str, int | float] = {
            "snapshots_applied": 0,
            "updates_applied": 0,
            "sequence_gaps": 0,
            "invalid_levels": 0,
            "books_reset": 0,
            "books_created": 0,
            "last_reset_at": 0.0,
        }

    # ------------------------------------------------------------------
    # Lifecycle / EventBus integration
    # ------------------------------------------------------------------

    def register(self) -> None:
        """Підписує cache на orderbook-події від усіх exchange adapters."""
        if self.event_bus is None:
            self._logger.warning("OrderBookCache register skipped: EventBus is not provided")
            return
        self.event_bus.subscribe("market.orderbook", self._on_market_orderbook)
        self.event_bus.subscribe("market.orderbook.batch", self._on_market_orderbook_batch)
        self.event_bus.subscribe("market.orderbook.snapshot", self._on_market_orderbook_snapshot)
        self._logger.info(
            "OrderBookCache registered | topics=%s",
            ["market.orderbook", "market.orderbook.batch", "market.orderbook.snapshot"],
        )

    async def start(self) -> None:
        self.register()

    async def stop(self) -> None:
        return None

    async def _on_market_orderbook_snapshot(self, event: Any) -> None:
        await self.apply_snapshot(self._normalize_inbound_payload(self._extract_payload(event)))

    async def _on_market_orderbook(self, event: Any) -> None:
        payload = self._normalize_inbound_payload(self._extract_payload(event))
        key = self._build_book_key_from_event(payload)
        state = self._books.get(key)
        if state is None or not state.snapshot_received or payload.get("type") == "snapshot":
            await self.apply_snapshot(payload)
        else:
            await self.apply_delta(payload)

    async def _on_market_orderbook_batch(self, event: Any) -> None:
        """
        Handles coalesced orderbook updates emitted by WS adapters.

        The batch preserves the original update order. Each item is still applied
        through the same snapshot/delta path as a single market.orderbook event,
        so sequence validation and resync signaling remain centralized here.
        """
        payload = self._extract_payload(event)
        updates = payload.get("updates") or payload.get("deltas") or payload.get("items") or []

        if not isinstance(updates, list):
            self._logger.warning(
                "Orderbook batch ignored: updates is not a list | exchange=%s symbol=%s",
                payload.get("exchange"),
                payload.get("symbol"),
            )
            return

        for item in updates:
            if not isinstance(item, dict):
                continue

            merged = {**payload, **item}
            merged.pop("updates", None)
            merged.pop("deltas", None)
            merged.pop("items", None)

            normalized = self._normalize_inbound_payload(merged)
            key = self._build_book_key_from_event(normalized)
            state = self._books.get(key)

            if state is None or not state.snapshot_received or normalized.get("type") == "snapshot":
                await self.apply_snapshot(normalized)
            else:
                await self.apply_delta(normalized)


    @staticmethod
    def _extract_payload(event: Any) -> dict[str, Any]:
        payload = getattr(event, "payload", event)
        return payload if isinstance(payload, dict) else {}

    def _normalize_inbound_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = self._now_ms()
        normalized = dict(payload)
        normalized.setdefault("market_type", payload.get("category") or payload.get("market_type") or "perpetual")
        normalized["timestamp_ms"] = payload.get("timestamp_ms") or payload.get("event_time") or payload.get("timestamp") or payload.get("ts") or now
        normalized["received_at_ms"] = payload.get("received_at_ms") or now
        normalized["sequence"] = payload.get("sequence") or payload.get("final_update_id") or payload.get("update_id") or payload.get("seq_id") or payload.get("version")
        normalized["prev_sequence"] = payload.get("prev_sequence") or payload.get("first_update_id") or payload.get("prev_seq_id")
        return normalized

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def apply_snapshot(self, event: dict[str, Any]) -> None:
        """
        Очікує event формату:
        {
            "exchange": "...",
            "symbol": "...",
            "market_type": "...",
            "timestamp_ms": ...,
            "received_at_ms": ...,
            "bids": [[price, qty], ...],
            "asks": [[price, qty], ...],
            "sequence": int | None,
        }
        """
        book_key = self._build_book_key_from_event(event)
        lock = self._get_lock(book_key)

        async with lock:
            state = self._get_or_create_state(event)

            state.bids.clear()
            state.asks.clear()

            self._apply_side_snapshot(state.bids, event.get("bids", []), side="bids", state=state)
            self._apply_side_snapshot(state.asks, event.get("asks", []), side="asks", state=state)

            self._trim_depth(state)

            state.prev_sequence = state.sequence
            state.sequence = self._safe_int(event.get("sequence"))
            state.snapshot_received = True
            state.is_synced = True
            state.last_update_ts_ms = self._safe_int(event.get("timestamp_ms"))
            state.last_update_received_ms = self._safe_int(event.get("received_at_ms"))
            state.last_error = None
            state.snapshots_applied += 1

            self._metrics["snapshots_applied"] += 1

            self._logger.info(
                "Order book snapshot applied | exchange=%s symbol=%s bids=%s asks=%s sequence=%s",
                state.exchange,
                state.symbol,
                len(state.bids),
                len(state.asks),
                state.sequence,
            )

            await self._emit_event("market.orderbook.updated", self._serialize_book_state(state))

    async def apply_delta(self, event: dict[str, Any]) -> None:
        """
        Очікує event формату:
        {
            "exchange": "...",
            "symbol": "...",
            "market_type": "...",
            "timestamp_ms": ...,
            "received_at_ms": ...,
            "bids": [[price, qty], ...],
            "asks": [[price, qty], ...],
            "sequence": int | None,
            "prev_sequence": int | None,
        }
        """
        book_key = self._build_book_key_from_event(event)
        lock = self._get_lock(book_key)

        async with lock:
            state = self._get_or_create_state(event)

            if not state.snapshot_received:
                state.dropped_updates += 1
                state.last_error = "delta_before_snapshot"

                self._logger.warning(
                    "Delta dropped: snapshot not received yet | exchange=%s symbol=%s",
                    state.exchange,
                    state.symbol,
                )

                await self._emit_event(
                    "system.orderbook_cache.delta_dropped",
                    {
                        "exchange": state.exchange,
                        "symbol": state.symbol,
                        "market_type": state.market_type,
                        "reason": "snapshot_not_received",
                    },
                )
                return

            if not self._validate_sequence(state, event):
                state.sequence_gaps += 1
                state.is_synced = False
                state.last_error = "sequence_gap_detected"

                self._metrics["sequence_gaps"] += 1

                self._logger.warning(
                    "Order book sequence gap detected | exchange=%s symbol=%s state_sequence=%s event_prev_sequence=%s event_sequence=%s",
                    state.exchange,
                    state.symbol,
                    state.sequence,
                    event.get("prev_sequence"),
                    event.get("sequence"),
                )

                await self._emit_event(
                    "system.orderbook_cache.sequence_gap",
                    {
                        "exchange": state.exchange,
                        "symbol": state.symbol,
                        "market_type": state.market_type,
                        "current_sequence": state.sequence,
                        "event_prev_sequence": self._safe_int(event.get("prev_sequence")),
                        "event_sequence": self._safe_int(event.get("sequence")),
                    },
                )
                return

            self._apply_side_delta(state.bids, event.get("bids", []), side="bids", state=state)
            self._apply_side_delta(state.asks, event.get("asks", []), side="asks", state=state)

            self._trim_depth(state)

            state.prev_sequence = state.sequence
            state.sequence = self._safe_int(event.get("sequence")) or state.sequence
            state.last_update_ts_ms = self._safe_int(event.get("timestamp_ms"))
            state.last_update_received_ms = self._safe_int(event.get("received_at_ms"))
            state.last_error = None
            state.is_synced = True
            state.updates_applied += 1

            self._metrics["updates_applied"] += 1

            await self._emit_event("market.orderbook.updated", self._serialize_book_state(state))

    async def reset_book(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
        reason: str = "manual_reset",
    ) -> None:
        book_key = self._build_book_key(exchange, symbol, market_type)
        lock = self._get_lock(book_key)

        async with lock:
            state = self._books.get(book_key)
            if state is None:
                return

            state.bids.clear()
            state.asks.clear()
            state.sequence = None
            state.prev_sequence = None
            state.snapshot_received = False
            state.is_synced = False
            state.last_error = reason
            state.last_update_ts_ms = None
            state.last_update_received_ms = None

            self._metrics["books_reset"] += 1
            self._metrics["last_reset_at"] = time.time()

            self._logger.warning(
                "Order book reset | exchange=%s symbol=%s reason=%s",
                exchange,
                symbol,
                reason,
            )

            await self._emit_event(
                "system.orderbook_cache.reset",
                {
                    "exchange": exchange,
                    "symbol": symbol,
                    "market_type": market_type,
                    "reason": reason,
                },
            )

    async def mark_for_resync(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
        reason: str = "resync_required",
    ) -> None:
        book_key = self._build_book_key(exchange, symbol, market_type)
        lock = self._get_lock(book_key)

        async with lock:
            state = self._books.get(book_key)
            if state is None:
                return

            state.is_synced = False
            state.last_error = reason

            self._logger.warning(
                "Order book marked for resync | exchange=%s symbol=%s reason=%s",
                exchange,
                symbol,
                reason,
            )

            await self._emit_event(
                "system.orderbook_cache.resync_required",
                {
                    "exchange": exchange,
                    "symbol": symbol,
                    "market_type": market_type,
                    "reason": reason,
                    "sequence": state.sequence,
                },
            )

    async def get_book(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
        depth: int | None = None,
    ) -> dict[str, Any] | None:
        book_key = self._build_book_key(exchange, symbol, market_type)
        lock = self._get_lock(book_key)

        async with lock:
            state = self._books.get(book_key)
            if state is None:
                return None

            bids = self._sorted_bids(state.bids)
            asks = self._sorted_asks(state.asks)

            if depth is not None and depth > 0:
                bids = bids[:depth]
                asks = asks[:depth]

            return {
                "exchange": state.exchange,
                "symbol": state.symbol,
                "market_type": state.market_type,
                "sequence": state.sequence,
                "prev_sequence": state.prev_sequence,
                "is_synced": state.is_synced,
                "snapshot_received": state.snapshot_received,
                "last_update_ts_ms": state.last_update_ts_ms,
                "last_update_received_ms": state.last_update_received_ms,
                "last_error": state.last_error,
                "bids": bids,
                "asks": asks,
                "best_bid": bids[0] if bids else None,
                "best_ask": asks[0] if asks else None,
                "spread": self._calc_spread(bids, asks),
                "mid_price": self._calc_mid_price(bids, asks),
                "updates_applied": state.updates_applied,
                "snapshots_applied": state.snapshots_applied,
                "sequence_gaps": state.sequence_gaps,
                "dropped_updates": state.dropped_updates,
            }

    async def get_top_of_book(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
    ) -> dict[str, Any] | None:
        book = await self.get_book(
            exchange=exchange,
            symbol=symbol,
            market_type=market_type,
            depth=1,
        )
        if book is None:
            return None

        return {
            "exchange": book["exchange"],
            "symbol": book["symbol"],
            "market_type": book["market_type"],
            "sequence": book["sequence"],
            "is_synced": book["is_synced"],
            "best_bid": book["best_bid"],
            "best_ask": book["best_ask"],
            "spread": book["spread"],
            "mid_price": book["mid_price"],
            "last_update_ts_ms": book["last_update_ts_ms"],
        }

    async def has_book(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
    ) -> bool:
        book_key = self._build_book_key(exchange, symbol, market_type)
        return book_key in self._books

    async def is_synced(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
    ) -> bool:
        book_key = self._build_book_key(exchange, symbol, market_type)
        state = self._books.get(book_key)
        if state is None:
            return False
        return state.is_synced and state.snapshot_received

    def stats(self) -> dict[str, Any]:
        synced_books = sum(1 for state in self._books.values() if state.is_synced)
        snapshot_books = sum(1 for state in self._books.values() if state.snapshot_received)

        return {
            "books_total": len(self._books),
            "books_synced": synced_books,
            "books_with_snapshot": snapshot_books,
            "snapshots_applied": self._metrics["snapshots_applied"],
            "updates_applied": self._metrics["updates_applied"],
            "sequence_gaps": self._metrics["sequence_gaps"],
            "invalid_levels": self._metrics["invalid_levels"],
            "books_reset": self._metrics["books_reset"],
            "books_created": self._metrics["books_created"],
            "last_reset_at": self._metrics["last_reset_at"],
            "max_depth_per_side": self.max_depth_per_side,
        }

    # ------------------------------------------------------------------
    # Internal book operations
    # ------------------------------------------------------------------

    def _apply_side_snapshot(
        self,
        side_map: dict[float, float],
        levels: Any,
        *,
        side: str,
        state: OrderBookState,
    ) -> None:
        for level in levels or []:
            parsed = self._parse_level(level)
            if parsed is None:
                self._metrics["invalid_levels"] += 1
                continue

            price, quantity = parsed
            if quantity <= 0:
                continue

            side_map[price] = quantity

    def _apply_side_delta(
        self,
        side_map: dict[float, float],
        levels: Any,
        *,
        side: str,
        state: OrderBookState,
    ) -> None:
        for level in levels or []:
            parsed = self._parse_level(level)
            if parsed is None:
                self._metrics["invalid_levels"] += 1
                continue

            price, quantity = parsed

            if quantity <= 0:
                side_map.pop(price, None)
                continue

            side_map[price] = quantity

    def _validate_sequence(self, state: OrderBookState, event: dict[str, Any]) -> bool:
        current_sequence = state.sequence
        event_sequence = self._safe_int(event.get("sequence"))
        event_prev_sequence = self._safe_int(event.get("prev_sequence"))

        if current_sequence is None:
            return True

        if event_prev_sequence is not None:
            return event_prev_sequence == current_sequence

        if event_sequence is not None:
            return event_sequence >= current_sequence

        return True

    def _trim_depth(self, state: OrderBookState) -> None:
        if self.max_depth_per_side <= 0:
            return

        if len(state.bids) > self.max_depth_per_side:
            sorted_prices = sorted(state.bids.keys(), reverse=True)
            for price in sorted_prices[self.max_depth_per_side:]:
                state.bids.pop(price, None)

        if len(state.asks) > self.max_depth_per_side:
            sorted_prices = sorted(state.asks.keys())
            for price in sorted_prices[self.max_depth_per_side:]:
                state.asks.pop(price, None)

    def _get_or_create_state(self, event: dict[str, Any]) -> OrderBookState:
        exchange = str(event["exchange"])
        symbol = str(event["symbol"])
        market_type = str(event.get("market_type", "perpetual"))

        book_key = self._build_book_key(exchange, symbol, market_type)
        state = self._books.get(book_key)
        if state is not None:
            return state

        state = OrderBookState(
            exchange=exchange,
            symbol=symbol,
            market_type=market_type,
        )
        self._books[book_key] = state
        self._metrics["books_created"] += 1

        self._logger.info(
            "Order book state created | exchange=%s symbol=%s market_type=%s",
            exchange,
            symbol,
            market_type,
        )

        return state

    def _get_lock(self, book_key: str) -> asyncio.Lock:
        lock = self._locks.get(book_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[book_key] = lock
        return lock

    async def _emit_event(self, topic: str, payload: dict[str, Any]) -> None:
        if self.event_bus is None:
            return

        try:
            await self.event_bus.emit(
                topic,
                payload,
                source="orderbook_cache",
            )
        except Exception:
            self._logger.exception(
                "Failed to emit orderbook cache event | topic=%s",
                topic,
            )

    def _serialize_book_state(self, state: OrderBookState) -> dict[str, Any]:
        bids = self._sorted_bids(state.bids)
        asks = self._sorted_asks(state.asks)
        return {
            "exchange": state.exchange,
            "symbol": state.symbol,
            "market_type": state.market_type,
            "sequence": state.sequence,
            "prev_sequence": state.prev_sequence,
            "is_synced": state.is_synced,
            "snapshot_received": state.snapshot_received,
            "last_update_ts_ms": state.last_update_ts_ms,
            "last_update_received_ms": state.last_update_received_ms,
            "bids": bids[: self.max_depth_per_side],
            "asks": asks[: self.max_depth_per_side],
            "best_bid": bids[0] if bids else None,
            "best_ask": asks[0] if asks else None,
            "spread": self._calc_spread(bids, asks),
            "mid_price": self._calc_mid_price(bids, asks),
        }

    # ------------------------------------------------------------------
    # Serialization / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_book_key(exchange: str, symbol: str, market_type: str) -> str:
        return f"{exchange}:{market_type}:{symbol}"

    def _build_book_key_from_event(self, event: dict[str, Any]) -> str:
        return self._build_book_key(
            str(event["exchange"]),
            str(event["symbol"]),
            str(event.get("market_type", "perpetual")),
        )

    @staticmethod
    def _parse_level(level: Any) -> tuple[float, float] | None:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            return None

        try:
            price = float(Decimal(str(level[0])))
            quantity = float(Decimal(str(level[1])))
        except (InvalidOperation, TypeError, ValueError):
            return None

        if price <= 0:
            return None

        return price, quantity

    @staticmethod
    def _sorted_bids(bids: dict[float, float]) -> list[list[float]]:
        return [[price, qty] for price, qty in sorted(bids.items(), key=lambda x: x[0], reverse=True)]

    @staticmethod
    def _sorted_asks(asks: dict[float, float]) -> list[list[float]]:
        return [[price, qty] for price, qty in sorted(asks.items(), key=lambda x: x[0])]

    @staticmethod
    def _calc_spread(bids: list[list[float]], asks: list[list[float]]) -> float | None:
        if not bids or not asks:
            return None
        return asks[0][0] - bids[0][0]

    @staticmethod
    def _calc_mid_price(bids: list[list[float]], asks: list[list[float]]) -> float | None:
        if not bids or not asks:
            return None
        return (bids[0][0] + asks[0][0]) / 2.0

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)