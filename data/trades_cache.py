from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from core.config import Config
from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler


@dataclass(slots=True)
class TradeRecord:
    exchange: str
    symbol: str
    market_type: str
    timestamp_ms: int
    received_at_ms: int
    trade_id: str | None
    price: float
    quantity: float
    side: str
    aggressor_side: str


@dataclass(slots=True)
class TradesState:
    exchange: str
    symbol: str
    market_type: str = "perpetual"

    trades: deque[TradeRecord] = field(default_factory=deque)

    last_trade_ts_ms: int | None = None
    last_received_at_ms: int | None = None
    last_trade_id: str | None = None
    last_error: str | None = None

    total_received: int = 0
    total_dropped: int = 0
    invalid_trades: int = 0
    duplicate_trade_ids: int = 0
    trims_count: int = 0


class TradesCache:
    """
    Локальний кеш останніх трейдів.

    Відповідальність:
    - зберігати останні trades у пам'яті
    - обмежувати обсяг кешу
    - віддавати recent trades
    - рахувати базові агрегати по вікну
    - підтримувати cleanup старих записів
    """

    def __init__(
        self,
        *,
        config: Config,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        cleanup_interval_seconds: float = 60.0,
        max_trades_per_book: int = 5000,
        retention_ms: int = 15 * 60 * 1000,
        service_name: str = "trades_cache",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._cleanup_job_id: str | None = None
        self.max_trades_per_book = max_trades_per_book
        self.retention_ms = retention_ms
        self._service_name = service_name

        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="trades_cache",
        )

        self._states: dict[str, TradesState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

        self._metrics: dict[str, int | float] = {
            "states_created": 0,
            "trades_received": 0,
            "trades_stored": 0,
            "trades_dropped": 0,
            "invalid_trades": 0,
            "duplicate_trade_ids": 0,
            "cleanup_runs": 0,
            "cleanup_removed": 0,
            "last_cleanup_at": 0.0,
        }

    # ------------------------------------------------------------------
    # Lifecycle / EventBus integration
    # ------------------------------------------------------------------

    def register(self) -> None:
        """Підписує cache на trade-події від усіх exchange adapters через EventBus."""
        if self.event_bus is None:
            self._logger.warning("TradesCache register skipped: EventBus is not provided")
            return
        self.event_bus.subscribe("market.trade", self._on_market_trade)
        self.event_bus.subscribe("market.trades.snapshot", self._on_market_trades_snapshot)
        self._register_cleanup_job()
        self._logger.info("TradesCache registered | topics=%s", ["market.trade", "market.trades.snapshot"])

    async def start(self) -> None:
        self.register()

    async def stop(self) -> None:
        if self.scheduler is not None and self._cleanup_job_id is not None:
            self.scheduler.remove_job(self._cleanup_job_id)
            self._cleanup_job_id = None

    async def _on_market_trade(self, event: Any) -> None:
        await self.update(self._normalize_inbound_payload(self._extract_payload(event)))

    async def _on_market_trades_snapshot(self, event: Any) -> None:
        payload = self._extract_payload(event)
        trades = payload.get("trades") or payload.get("items") or []
        if isinstance(trades, list):
            for trade in trades:
                if isinstance(trade, dict):
                    merged = {**payload, **trade}
                    merged.pop("trades", None)
                    merged.pop("items", None)
                    await self.update(self._normalize_inbound_payload(merged))

    def _register_cleanup_job(self) -> None:
        if self.scheduler is None or self._cleanup_job_id is not None:
            return
        self._cleanup_job_id = self.scheduler.add_interval_job(
            name="trades-cache-cleanup",
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
        normalized["timestamp_ms"] = payload.get("timestamp_ms") or payload.get("trade_time") or payload.get("event_time") or payload.get("timestamp") or now
        normalized["received_at_ms"] = payload.get("received_at_ms") or now
        normalized["quantity"] = payload.get("quantity") if payload.get("quantity") is not None else payload.get("qty")
        normalized["aggressor_side"] = payload.get("aggressor_side") or payload.get("side") or "unknown"
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
            "trade_id": "...",
            "price": ...,
            "quantity": ...,
            "side": "buy" | "sell" | "unknown",
            "aggressor_side": "buy" | "sell" | "unknown",
        }
        """
        state_key = self._build_state_key_from_event(event)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._get_or_create_state(event)
            self._metrics["trades_received"] += 1
            state.total_received += 1

            record = self._normalize_trade(event)
            if record is None:
                state.invalid_trades += 1
                state.last_error = "invalid_trade_event"
                self._metrics["invalid_trades"] += 1

                await self._emit_event(
                    "system.trades_cache.invalid_trade",
                    {
                        "exchange": event.get("exchange"),
                        "symbol": event.get("symbol"),
                        "market_type": event.get("market_type", "perpetual"),
                    },
                )
                return

            if self._is_duplicate_trade(state, record):
                state.duplicate_trade_ids += 1
                self._metrics["duplicate_trade_ids"] += 1

                self._logger.warning(
                    "Duplicate trade detected | exchange=%s symbol=%s trade_id=%s",
                    record.exchange,
                    record.symbol,
                    record.trade_id,
                )
                return

            state.trades.append(record)
            state.last_trade_ts_ms = record.timestamp_ms
            state.last_received_at_ms = record.received_at_ms
            state.last_trade_id = record.trade_id
            state.last_error = None

            self._metrics["trades_stored"] += 1

            removed = self._trim_state(state)
            if removed > 0:
                state.trims_count += 1
                state.total_dropped += removed
                self._metrics["trades_dropped"] += removed

            await self._emit_event(
                "market.trades.updated",
                {
                    "exchange": record.exchange,
                    "symbol": record.symbol,
                    "market_type": record.market_type,
                    "trade": self._serialize_trade(record),
                    "last_trade_ts_ms": record.timestamp_ms,
                },
            )

    async def get_recent_trades(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        state_key = self._build_state_key(exchange, symbol, market_type)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._states.get(state_key)
            if state is None:
                return []

            if limit <= 0:
                return []

            trades = list(state.trades)[-limit:]
            return [self._serialize_trade(trade) for trade in trades]

    async def get_trades_since(
        self,
        *,
        exchange: str,
        symbol: str,
        since_timestamp_ms: int,
        market_type: str = "perpetual",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        state_key = self._build_state_key(exchange, symbol, market_type)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._states.get(state_key)
            if state is None:
                return []

            filtered = [
                self._serialize_trade(trade)
                for trade in state.trades
                if trade.timestamp_ms >= since_timestamp_ms
            ]

            if limit is not None and limit > 0:
                return filtered[-limit:]
            return filtered

    async def get_last_trade(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
    ) -> dict[str, Any] | None:
        state_key = self._build_state_key(exchange, symbol, market_type)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._states.get(state_key)
            if state is None or not state.trades:
                return None

            return self._serialize_trade(state.trades[-1])

    async def get_window_stats(
        self,
        *,
        exchange: str,
        symbol: str,
        window_ms: int,
        market_type: str = "perpetual",
    ) -> dict[str, Any] | None:
        state_key = self._build_state_key(exchange, symbol, market_type)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._states.get(state_key)
            if state is None or not state.trades:
                return None

            if window_ms <= 0:
                raise ValueError("window_ms must be > 0")

            cutoff = self._now_ms() - window_ms
            trades = [trade for trade in state.trades if trade.timestamp_ms >= cutoff]

            if not trades:
                return {
                    "exchange": exchange,
                    "symbol": symbol,
                    "market_type": market_type,
                    "window_ms": window_ms,
                    "trades_count": 0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "buy_volume": 0.0,
                    "sell_volume": 0.0,
                    "total_volume": 0.0,
                    "notional": 0.0,
                    "vwap": None,
                    "min_price": None,
                    "max_price": None,
                    "first_timestamp_ms": None,
                    "last_timestamp_ms": None,
                }

            buy_count = 0
            sell_count = 0
            buy_volume = 0.0
            sell_volume = 0.0
            total_volume = 0.0
            notional = 0.0
            min_price = None
            max_price = None

            for trade in trades:
                total_volume += trade.quantity
                notional += trade.price * trade.quantity

                if min_price is None or trade.price < min_price:
                    min_price = trade.price
                if max_price is None or trade.price > max_price:
                    max_price = trade.price

                side = trade.aggressor_side if trade.aggressor_side != "unknown" else trade.side
                if side == "buy":
                    buy_count += 1
                    buy_volume += trade.quantity
                elif side == "sell":
                    sell_count += 1
                    sell_volume += trade.quantity

            vwap = notional / total_volume if total_volume > 0 else None

            return {
                "exchange": exchange,
                "symbol": symbol,
                "market_type": market_type,
                "window_ms": window_ms,
                "trades_count": len(trades),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "total_volume": total_volume,
                "delta_volume": buy_volume - sell_volume,
                "notional": notional,
                "vwap": vwap,
                "min_price": min_price,
                "max_price": max_price,
                "first_timestamp_ms": trades[0].timestamp_ms,
                "last_timestamp_ms": trades[-1].timestamp_ms,
            }

    async def clear_symbol(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
        reason: str = "manual_clear",
    ) -> None:
        state_key = self._build_state_key(exchange, symbol, market_type)
        lock = self._get_lock(state_key)

        async with lock:
            state = self._states.get(state_key)
            if state is None:
                return

            removed = len(state.trades)
            state.trades.clear()
            state.last_trade_ts_ms = None
            state.last_received_at_ms = None
            state.last_trade_id = None
            state.last_error = reason

            self._logger.warning(
                "Trades cleared | exchange=%s symbol=%s removed=%s reason=%s",
                exchange,
                symbol,
                removed,
                reason,
            )

            await self._emit_event(
                "system.trades_cache.cleared",
                {
                    "exchange": exchange,
                    "symbol": symbol,
                    "market_type": market_type,
                    "removed": removed,
                    "reason": reason,
                },
            )

    async def cleanup_stale(self) -> int:
        total_removed = 0
        now_ms = self._now_ms()
        cutoff = now_ms - self.retention_ms

        for state_key, state in list(self._states.items()):
            lock = self._get_lock(state_key)

            async with lock:
                removed_here = 0
                while state.trades and state.trades[0].timestamp_ms < cutoff:
                    state.trades.popleft()
                    removed_here += 1

                if removed_here > 0:
                    state.total_dropped += removed_here
                    state.trims_count += 1
                    total_removed += removed_here

        self._metrics["cleanup_runs"] += 1
        self._metrics["cleanup_removed"] += total_removed
        self._metrics["last_cleanup_at"] = time.time()

        if total_removed > 0:
            self._logger.info(
                "Trades cleanup completed | removed=%s retention_ms=%s",
                total_removed,
                self.retention_ms,
            )

        return total_removed

    async def has_trades(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
    ) -> bool:
        state_key = self._build_state_key(exchange, symbol, market_type)
        state = self._states.get(state_key)
        return state is not None and len(state.trades) > 0

    async def size(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = "perpetual",
    ) -> int:
        state_key = self._build_state_key(exchange, symbol, market_type)
        state = self._states.get(state_key)
        if state is None:
            return 0
        return len(state.trades)

    def stats(self) -> dict[str, Any]:
        active_states = sum(1 for state in self._states.values() if state.trades)

        return {
            "states_total": len(self._states),
            "states_with_trades": active_states,
            "states_created": self._metrics["states_created"],
            "trades_received": self._metrics["trades_received"],
            "trades_stored": self._metrics["trades_stored"],
            "trades_dropped": self._metrics["trades_dropped"],
            "invalid_trades": self._metrics["invalid_trades"],
            "duplicate_trade_ids": self._metrics["duplicate_trade_ids"],
            "cleanup_runs": self._metrics["cleanup_runs"],
            "cleanup_removed": self._metrics["cleanup_removed"],
            "last_cleanup_at": self._metrics["last_cleanup_at"],
            "max_trades_per_book": self.max_trades_per_book,
            "retention_ms": self.retention_ms,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create_state(self, event: dict[str, Any]) -> TradesState:
        exchange = str(event["exchange"])
        symbol = str(event["symbol"])
        market_type = str(event.get("market_type", "perpetual"))

        state_key = self._build_state_key(exchange, symbol, market_type)
        state = self._states.get(state_key)
        if state is not None:
            return state

        state = TradesState(
            exchange=exchange,
            symbol=symbol,
            market_type=market_type,
        )
        self._states[state_key] = state
        self._metrics["states_created"] += 1

        self._logger.info(
            "Trades state created | exchange=%s symbol=%s market_type=%s",
            exchange,
            symbol,
            market_type,
        )

        return state

    def _get_lock(self, state_key: str) -> asyncio.Lock:
        lock = self._locks.get(state_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[state_key] = lock
        return lock

    def _normalize_trade(self, event: dict[str, Any]) -> TradeRecord | None:
        exchange = event.get("exchange")
        symbol = event.get("symbol")
        market_type = event.get("market_type", "perpetual")

        timestamp_ms = self._safe_int(event.get("timestamp_ms"))
        received_at_ms = self._safe_int(event.get("received_at_ms")) or self._now_ms()

        price = self._safe_float(event.get("price"))
        quantity = self._safe_float(event.get("quantity"))

        if exchange is None or symbol is None:
            return None
        if timestamp_ms is None or price is None or quantity is None:
            return None
        if price <= 0 or quantity <= 0:
            return None

        side = self._normalize_side(event.get("side"))
        aggressor_side = self._normalize_side(event.get("aggressor_side"))

        return TradeRecord(
            exchange=str(exchange),
            symbol=str(symbol),
            market_type=str(market_type),
            timestamp_ms=timestamp_ms,
            received_at_ms=received_at_ms,
            trade_id=self._safe_str(event.get("trade_id")),
            price=price,
            quantity=quantity,
            side=side,
            aggressor_side=aggressor_side,
        )

    def _is_duplicate_trade(self, state: TradesState, trade: TradeRecord) -> bool:
        if trade.trade_id is None:
            return False
        if not state.trades:
            return False

        last_trade = state.trades[-1]
        return (
            last_trade.trade_id == trade.trade_id
            and last_trade.timestamp_ms == trade.timestamp_ms
            and last_trade.price == trade.price
            and last_trade.quantity == trade.quantity
        )

    def _trim_state(self, state: TradesState) -> int:
        removed = 0
        now_ms = self._now_ms()
        cutoff = now_ms - self.retention_ms

        while state.trades and len(state.trades) > self.max_trades_per_book:
            state.trades.popleft()
            removed += 1

        while state.trades and state.trades[0].timestamp_ms < cutoff:
            state.trades.popleft()
            removed += 1

        return removed

    async def _emit_event(self, topic: str, payload: dict[str, Any]) -> None:
        if self.event_bus is None:
            return

        try:
            await self.event_bus.emit(
                topic,
                payload,
                source="trades_cache",
            )
        except Exception:
            self._logger.exception(
                "Failed to emit trades cache event | topic=%s",
                topic,
            )

    @staticmethod
    def _build_state_key(exchange: str, symbol: str, market_type: str) -> str:
        return f"{exchange}:{market_type}:{symbol}"

    def _build_state_key_from_event(self, event: dict[str, Any]) -> str:
        return self._build_state_key(
            str(event["exchange"]),
            str(event["symbol"]),
            str(event.get("market_type", "perpetual")),
        )

    @staticmethod
    def _serialize_trade(trade: TradeRecord) -> dict[str, Any]:
        return {
            "exchange": trade.exchange,
            "symbol": trade.symbol,
            "market_type": trade.market_type,
            "timestamp_ms": trade.timestamp_ms,
            "received_at_ms": trade.received_at_ms,
            "trade_id": trade.trade_id,
            "price": trade.price,
            "quantity": trade.quantity,
            "side": trade.side,
            "aggressor_side": trade.aggressor_side,
        }

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
    def _safe_str(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _normalize_side(value: Any) -> str:
        if value is None:
            return "unknown"

        normalized = str(value).strip().lower()
        if normalized in {"buy", "bid", "b"}:
            return "buy"
        if normalized in {"sell", "ask", "s"}:
            return "sell"
        return "unknown"

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)