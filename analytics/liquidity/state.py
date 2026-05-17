from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .models import (
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    EqualLevel,
    LiquidityLevel,
    LiquidityMapSnapshot,
    StopCluster,
)


def utcnow() -> datetime:
    """
    Єдиний timezone-aware UTC timestamp для liquidity state.

    State layer не імпортує core/utils.time_utils, щоб не створювати
    зайвих залежностей. Якщо пізніше в проєкті буде глобальний time helper,
    цю функцію можна буде замінити на нього.
    """
    return datetime.now(timezone.utc)


def _normalize_scope_value(value: Any, default: str) -> str:
    normalized = str(value or default).strip()
    return normalized if normalized else default


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_timeframe(value: Any) -> str:
    return str(value or "").strip()


@dataclass(slots=True)
class LiquidityTimeframeState:
    """
    In-memory state для конкретного exchange + market_type + symbol + timeframe.

    Це чистий state container:
    - не має EventBus;
    - не має Scheduler;
    - не має logger;
    - не виконує IO;
    - не запускає periodic tasks.

    Оновлюється LiquidityService після побудови LiquidityMapSnapshot.
    """

    exchange: str
    market_type: str
    symbol: str
    timeframe: str

    active_levels: list[LiquidityLevel] = field(default_factory=list)
    equal_levels: list[EqualLevel] = field(default_factory=list)
    stop_clusters: list[StopCluster] = field(default_factory=list)

    last_snapshot: LiquidityMapSnapshot | None = None

    last_candle_open_time: datetime | None = None
    last_candle_close_time: datetime | None = None
    last_orderbook_update_at: datetime | None = None
    last_price_update_at: datetime | None = None
    last_snapshot_at: datetime | None = None
    last_update_at: datetime | None = None

    processed_candles: int = 0
    processed_orderbook_updates: int = 0
    processed_price_updates: int = 0
    snapshots_built: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.exchange = _normalize_scope_value(self.exchange, DEFAULT_EXCHANGE)
        self.market_type = _normalize_scope_value(self.market_type, DEFAULT_MARKET_TYPE)
        self.symbol = _normalize_symbol(self.symbol)
        self.timeframe = _normalize_timeframe(self.timeframe)

    @property
    def key(self) -> str:
        return LiquidityState.make_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    def touch(self, ts: datetime | None = None) -> None:
        self.last_update_at = self._normalize_timestamp(ts) or utcnow()

    def apply_snapshot(
        self,
        snapshot: LiquidityMapSnapshot,
        *,
        ts: datetime | None = None,
    ) -> None:
        """
        Застосовує новий LiquidityMapSnapshot до state.

        Викликається LiquidityService після успішного rebuild_snapshot().
        """
        self.exchange = _normalize_scope_value(snapshot.exchange, self.exchange)
        self.market_type = _normalize_scope_value(snapshot.market_type, self.market_type)
        self.symbol = _normalize_symbol(snapshot.symbol)
        self.timeframe = _normalize_timeframe(snapshot.timeframe)

        self.last_snapshot = snapshot

        self.active_levels = list(snapshot.active_levels)
        self.equal_levels = list(snapshot.equal_levels)
        self.stop_clusters = list(snapshot.stop_clusters)

        self.last_snapshot_at = self._normalize_timestamp(snapshot.timestamp)
        self.snapshots_built += 1
        self.touch(ts or snapshot.timestamp)

    def record_candle_processed(
        self,
        *,
        open_time: datetime | None = None,
        close_time: datetime | None = None,
        ts: datetime | None = None,
    ) -> None:
        self.processed_candles += 1

        normalized_open_time = self._normalize_timestamp(open_time)
        normalized_close_time = self._normalize_timestamp(close_time)

        if normalized_open_time is not None:
            self.last_candle_open_time = normalized_open_time

        if normalized_close_time is not None:
            self.last_candle_close_time = normalized_close_time

        self.touch(ts)

    def record_orderbook_processed(
        self,
        *,
        ts: datetime | None = None,
    ) -> None:
        event_ts = self._normalize_timestamp(ts) or utcnow()
        self.processed_orderbook_updates += 1
        self.last_orderbook_update_at = event_ts
        self.touch(event_ts)

    def record_price_processed(
        self,
        *,
        ts: datetime | None = None,
    ) -> None:
        event_ts = self._normalize_timestamp(ts) or utcnow()
        self.processed_price_updates += 1
        self.last_price_update_at = event_ts
        self.touch(event_ts)

    def prune(
        self,
        *,
        max_active_levels: int,
        max_active_clusters: int,
    ) -> None:
        """
        Обрізає state до заданих retention limits.

        Не запускається самостійно. Виклик має робити LiquidityService
        через Scheduler.add_interval_job().
        """
        if max_active_levels > 0 and len(self.active_levels) > max_active_levels:
            self.active_levels = sorted(
                self.active_levels,
                key=lambda level: level.confidence,
                reverse=True,
            )[:max_active_levels]

        if max_active_levels > 0 and len(self.equal_levels) > max_active_levels:
            self.equal_levels = sorted(
                self.equal_levels,
                key=lambda level: level.confidence,
                reverse=True,
            )[:max_active_levels]

        if max_active_clusters > 0 and len(self.stop_clusters) > max_active_clusters:
            self.stop_clusters = sorted(
                self.stop_clusters,
                key=lambda cluster: cluster.confidence,
                reverse=True,
            )[:max_active_clusters]

        self.touch()

    def remove_inactive_levels(self) -> int:
        """
        Видаляє terminal/inactive liquidity levels.

        Returns
        -------
        int
            Кількість видалених рівнів.
        """
        before_active = len(self.active_levels)
        before_equal = len(self.equal_levels)

        self.active_levels = [
            level
            for level in self.active_levels
            if level.is_active()
        ]

        self.equal_levels = [
            level
            for level in self.equal_levels
            if level.is_active()
        ]

        removed = (before_active - len(self.active_levels)) + (
            before_equal - len(self.equal_levels)
        )

        if removed:
            self.touch()

        return removed

    def has_snapshot(self) -> bool:
        return self.last_snapshot is not None

    def has_levels(self) -> bool:
        return bool(
            self.active_levels
            or self.equal_levels
            or self.stop_clusters
        )

    def to_metrics_payload(self) -> dict[str, Any]:
        """
        Compact metrics payload для analytics.liquidity.state.metrics.
        """
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "key": self.key,
            "active_levels": len(self.active_levels),
            "equal_levels": len(self.equal_levels),
            "stop_clusters": len(self.stop_clusters),
            "has_snapshot": self.has_snapshot(),
            "processed_candles": self.processed_candles,
            "processed_orderbook_updates": self.processed_orderbook_updates,
            "processed_price_updates": self.processed_price_updates,
            "snapshots_built": self.snapshots_built,
            "last_candle_open_time": (
                self.last_candle_open_time.isoformat()
                if self.last_candle_open_time
                else None
            ),
            "last_candle_close_time": (
                self.last_candle_close_time.isoformat()
                if self.last_candle_close_time
                else None
            ),
            "last_orderbook_update_at": (
                self.last_orderbook_update_at.isoformat()
                if self.last_orderbook_update_at
                else None
            ),
            "last_price_update_at": (
                self.last_price_update_at.isoformat()
                if self.last_price_update_at
                else None
            ),
            "last_snapshot_at": (
                self.last_snapshot_at.isoformat()
                if self.last_snapshot_at
                else None
            ),
            "last_update_at": (
                self.last_update_at.isoformat()
                if self.last_update_at
                else None
            ),
            "metadata": dict(self.metadata),
        }

    def reset(self) -> None:
        self.active_levels.clear()
        self.equal_levels.clear()
        self.stop_clusters.clear()

        self.last_snapshot = None

        self.last_candle_open_time = None
        self.last_candle_close_time = None
        self.last_orderbook_update_at = None
        self.last_price_update_at = None
        self.last_snapshot_at = None
        self.last_update_at = None

        self.processed_candles = 0
        self.processed_orderbook_updates = 0
        self.processed_price_updates = 0
        self.snapshots_built = 0

        self.metadata.clear()

    @staticmethod
    def _normalize_timestamp(value: datetime | None) -> datetime | None:
        if value is None:
            return None

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)


@dataclass(slots=True)
class LiquidityState:
    """
    Загальний in-memory state liquidity-модуля.

    Ключ:
        "{exchange}:{market_type}:{symbol}:{timeframe}"

    Цей клас не керує lifecycle. Його використовує LiquidityService.
    """

    states: dict[str, LiquidityTimeframeState] = field(default_factory=dict)

    @staticmethod
    def make_key(
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str = "",
        timeframe: str = "",
    ) -> str:
        normalized_exchange = _normalize_scope_value(exchange, DEFAULT_EXCHANGE).lower()
        normalized_market_type = _normalize_scope_value(
            market_type,
            DEFAULT_MARKET_TYPE,
        ).lower()
        normalized_symbol = _normalize_symbol(symbol)
        normalized_timeframe = _normalize_timeframe(timeframe)

        return (
            f"{normalized_exchange}:"
            f"{normalized_market_type}:"
            f"{normalized_symbol}:"
            f"{normalized_timeframe}"
        )

    @staticmethod
    def make_market_prefix(
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str = "",
    ) -> str:
        normalized_exchange = _normalize_scope_value(exchange, DEFAULT_EXCHANGE).lower()
        normalized_market_type = _normalize_scope_value(
            market_type,
            DEFAULT_MARKET_TYPE,
        ).lower()
        normalized_symbol = _normalize_symbol(symbol)

        return f"{normalized_exchange}:{normalized_market_type}:{normalized_symbol}:"

    def get(
        self,
        symbol: str,
        timeframe: str,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
    ) -> LiquidityTimeframeState | None:
        return self.states.get(
            self.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    def get_or_create(
        self,
        symbol: str,
        timeframe: str,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
    ) -> LiquidityTimeframeState:
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        if key not in self.states:
            self.states[key] = LiquidityTimeframeState(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )

        return self.states[key]

    def get_for_market(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
    ) -> list[LiquidityTimeframeState]:
        prefix = self.make_market_prefix(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
        )
        return [
            state
            for key, state in self.states.items()
            if key.startswith(prefix)
        ]

    def apply_snapshot(
        self,
        snapshot: LiquidityMapSnapshot,
    ) -> LiquidityTimeframeState:
        state = self.get_or_create(
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
        )
        state.apply_snapshot(snapshot)
        return state

    def remove(
        self,
        symbol: str,
        timeframe: str,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
    ) -> None:
        self.states.pop(
            self.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            ),
            None,
        )

    def remove_market(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
    ) -> int:
        prefix = self.make_market_prefix(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
        )
        keys_to_remove = [
            key
            for key in self.states
            if key.startswith(prefix)
        ]

        for key in keys_to_remove:
            self.states.pop(key, None)

        return len(keys_to_remove)

    def clear(self) -> None:
        self.states.clear()

    def keys(self) -> list[str]:
        return list(self.states.keys())

    def values(self) -> list[LiquidityTimeframeState]:
        return list(self.states.values())

    def items(self) -> list[tuple[str, LiquidityTimeframeState]]:
        return list(self.states.items())

    def count(self) -> int:
        return len(self.states)

    def prune_all(
        self,
        *,
        max_active_levels: int,
        max_active_clusters: int,
    ) -> None:
        """
        Prune для всіх timeframe-state.

        Запускати з LiquidityService через Scheduler.
        """
        for state in self.states.values():
            state.prune(
                max_active_levels=max_active_levels,
                max_active_clusters=max_active_clusters,
            )

    def remove_empty_states(self) -> int:
        """
        Видаляє порожні states без snapshot і без рівнів.

        Returns
        -------
        int
            Кількість видалених states.
        """
        empty_keys = [
            key
            for key, state in self.states.items()
            if not state.has_snapshot() and not state.has_levels()
        ]

        for key in empty_keys:
            self.states.pop(key, None)

        return len(empty_keys)

    def remove_inactive_levels(self) -> int:
        """
        Видаляє inactive/terminal levels у всіх states.
        """
        removed = 0

        for state in self.states.values():
            removed += state.remove_inactive_levels()

        return removed

    def to_metrics_payload(self) -> dict[str, Any]:
        """
        Compact metrics payload для service-level event.
        """
        states = list(self.states.values())

        return {
            "states_count": len(states),
            "exchanges": sorted({state.exchange for state in states}),
            "market_types": sorted({state.market_type for state in states}),
            "symbols": sorted({state.symbol for state in states}),
            "timeframes": sorted({state.timeframe for state in states}),
            "total_active_levels": sum(len(state.active_levels) for state in states),
            "total_equal_levels": sum(len(state.equal_levels) for state in states),
            "total_stop_clusters": sum(len(state.stop_clusters) for state in states),
            "total_processed_candles": sum(
                state.processed_candles
                for state in states
            ),
            "total_processed_orderbook_updates": sum(
                state.processed_orderbook_updates
                for state in states
            ),
            "total_processed_price_updates": sum(
                state.processed_price_updates
                for state in states
            ),
            "total_snapshots_built": sum(
                state.snapshots_built
                for state in states
            ),
            "states": [
                state.to_metrics_payload()
                for state in states
            ],
        }