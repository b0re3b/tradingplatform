from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .models import (
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    EqualLevel,
    LiquidityKey,
    LiquidityLevel,
    LiquidityMapSnapshot,
    StopCluster,
    ensure_utc,
    liquidity_key_to_dict,
    liquidity_key_to_string,
    make_liquidity_key,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    utc_now,
)


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

    Canonical scope:
        exchange + market_type + symbol + timeframe
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
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.timeframe = normalize_timeframe(self.timeframe)

        self.last_candle_open_time = self._normalize_timestamp(
            self.last_candle_open_time
        )
        self.last_candle_close_time = self._normalize_timestamp(
            self.last_candle_close_time
        )
        self.last_orderbook_update_at = self._normalize_timestamp(
            self.last_orderbook_update_at
        )
        self.last_price_update_at = self._normalize_timestamp(
            self.last_price_update_at
        )
        self.last_snapshot_at = self._normalize_timestamp(self.last_snapshot_at)
        self.last_update_at = self._normalize_timestamp(self.last_update_at)

        self.processed_candles = max(0, int(self.processed_candles))
        self.processed_orderbook_updates = max(
            0,
            int(self.processed_orderbook_updates),
        )
        self.processed_price_updates = max(0, int(self.processed_price_updates))
        self.snapshots_built = max(0, int(self.snapshots_built))

        self.metadata = dict(self.metadata or {})
        self.metadata.setdefault("scope", self.scope)

        self._apply_scope_to_children()

    @property
    def liquidity_key(self) -> LiquidityKey:
        return make_liquidity_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def key(self) -> LiquidityKey:
        """
        Canonical typed key.

        Новий код має використовувати саме цей key, а не string key.
        """
        return self.liquidity_key

    @property
    def key_string(self) -> str:
        return liquidity_key_to_string(self.key)

    @property
    def scope_key(self) -> str:
        return self.key_string

    @property
    def scope(self) -> dict[str, str]:
        return liquidity_key_to_dict(self.key)

    def touch(self, ts: datetime | None = None) -> None:
        self.last_update_at = self._normalize_timestamp(ts) or utc_now()

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
        snapshot_key = make_liquidity_key(
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
        )

        if snapshot_key != self.key:
            raise ValueError(
                "LiquidityMapSnapshot scope mismatch: "
                f"state={self.scope}, snapshot={liquidity_key_to_dict(snapshot_key)}"
            )

        self.last_snapshot = snapshot

        self.active_levels = list(snapshot.active_levels)
        self.equal_levels = list(snapshot.equal_levels)
        self.stop_clusters = list(snapshot.stop_clusters)

        self._apply_scope_to_children()

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
        event_ts = self._normalize_timestamp(ts) or utc_now()
        self.processed_orderbook_updates += 1
        self.last_orderbook_update_at = event_ts
        self.touch(event_ts)

    def record_price_processed(
        self,
        *,
        ts: datetime | None = None,
    ) -> None:
        event_ts = self._normalize_timestamp(ts) or utc_now()
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

        self._apply_scope_to_children()
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
            "scope": self.scope,
            "scope_key": self.scope_key,
            "liquidity_key": self.key,
            "active_levels": len(self.active_levels),
            "equal_levels": len(self.equal_levels),
            "stop_clusters": len(self.stop_clusters),
            "has_snapshot": self.has_snapshot(),
            "has_levels": self.has_levels(),
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
        self.metadata["scope"] = self.scope

    def _apply_scope_to_children(self) -> None:
        """
        Гарантує, що всі дочірні liquidity-моделі мають той самий scope.
        """
        for level in self.active_levels:
            self._scope_level(level)

        for level in self.equal_levels:
            self._scope_level(level)

        for cluster in self.stop_clusters:
            self._scope_cluster(cluster)

        if self.last_snapshot is not None:
            if self.last_snapshot.liquidity_key != self.key:
                raise ValueError(
                    "Last snapshot scope mismatch: "
                    f"state={self.scope}, snapshot={self.last_snapshot.scope}"
                )

    def _scope_level(self, level: LiquidityLevel) -> None:
        level.exchange = self.exchange
        level.market_type = self.market_type
        level.symbol = self.symbol
        level.timeframe = self.timeframe
        level.metadata.setdefault("scope", self.scope)

    def _scope_cluster(self, cluster: StopCluster) -> None:
        cluster.exchange = self.exchange
        cluster.market_type = self.market_type
        cluster.symbol = self.symbol
        cluster.timeframe = self.timeframe
        cluster.metadata.setdefault("scope", self.scope)

        for source_level in cluster.source_levels:
            self._scope_level(source_level)

    @staticmethod
    def _normalize_timestamp(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return ensure_utc(value)


@dataclass(slots=True)
class LiquidityState:
    """
    Загальний in-memory state liquidity-модуля.

    Canonical key:
        LiquidityKey = exchange + market_type + symbol + timeframe

    Цей клас не керує lifecycle. Його використовує LiquidityService.
    """

    states: dict[LiquidityKey, LiquidityTimeframeState] = field(default_factory=dict)

    @staticmethod
    def make_key(
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str = "",
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> LiquidityKey:
        return make_liquidity_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    @staticmethod
    def make_key_string(
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str = "",
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> str:
        return liquidity_key_to_string(
            LiquidityState.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    @staticmethod
    def make_market_prefix(
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str = "",
    ) -> tuple[str, str, str]:
        """
        Prefix для пошуку всіх timeframe state конкретного ринку.

        Повертає tuple, а не string, бо canonical key тепер typed LiquidityKey.
        """
        return (
            normalize_exchange(exchange),
            normalize_market_type(market_type),
            normalize_symbol(symbol),
        )

    def get(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
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

    def get_key(self, key: LiquidityKey) -> LiquidityTimeframeState | None:
        return self.states.get(key)

    def get_or_create(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
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
            scope = liquidity_key_to_dict(key)
            self.states[key] = LiquidityTimeframeState(
                exchange=scope["exchange"],
                market_type=scope["market_type"],
                symbol=scope["symbol"],
                timeframe=scope["timeframe"],
            )

        return self.states[key]

    def get_or_create_key(self, key: LiquidityKey) -> LiquidityTimeframeState:
        scope = liquidity_key_to_dict(key)
        return self.get_or_create(
            exchange=scope["exchange"],
            market_type=scope["market_type"],
            symbol=scope["symbol"],
            timeframe=scope["timeframe"],
        )

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
            if key[:3] == prefix
        ]

    def get_for_symbol(
        self,
        *,
        symbol: str,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> list[LiquidityTimeframeState]:
        """
        Гнучкий scoped lookup.

        Якщо exchange/market_type/timeframe не передані, повертає всі matching states.
        """
        normalized_symbol = normalize_symbol(symbol)
        normalized_exchange = normalize_exchange(exchange) if exchange is not None else None
        normalized_market_type = (
            normalize_market_type(market_type)
            if market_type is not None
            else None
        )
        normalized_timeframe = (
            normalize_timeframe(timeframe)
            if timeframe is not None
            else None
        )

        result: list[LiquidityTimeframeState] = []

        for key, state in self.states.items():
            scope = liquidity_key_to_dict(key)

            if scope["symbol"] != normalized_symbol:
                continue

            if normalized_exchange is not None and scope["exchange"] != normalized_exchange:
                continue

            if (
                normalized_market_type is not None
                and scope["market_type"] != normalized_market_type
            ):
                continue

            if normalized_timeframe is not None and scope["timeframe"] != normalized_timeframe:
                continue

            result.append(state)

        return result

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
        timeframe: str = DEFAULT_TIMEFRAME,
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

    def remove_key(self, key: LiquidityKey) -> None:
        self.states.pop(key, None)

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
            if key[:3] == prefix
        ]

        for key in keys_to_remove:
            self.states.pop(key, None)

        return len(keys_to_remove)

    def clear(self) -> None:
        self.states.clear()

    def keys(self) -> list[LiquidityKey]:
        return list(self.states.keys())

    def key_strings(self) -> list[str]:
        return [liquidity_key_to_string(key) for key in self.states]

    def values(self) -> list[LiquidityTimeframeState]:
        return list(self.states.values())

    def items(self) -> list[tuple[LiquidityKey, LiquidityTimeframeState]]:
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
            "scope": "exchange:market_type:symbol:timeframe",
            "exchanges": sorted({state.exchange for state in states}),
            "market_types": sorted({state.market_type for state in states}),
            "symbols": sorted({state.symbol for state in states}),
            "timeframes": sorted({state.timeframe for state in states}),
            "scope_keys": sorted(state.scope_key for state in states),
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


__all__ = [
    "LiquidityTimeframeState",
    "LiquidityState",
]