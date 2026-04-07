from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from .base import BaseSpoofingTracker
from .config import SpoofingConfig
from .enums import (
    LiquidityEventType,
    OrderbookWallState,
    SpoofingComponent,
    SpoofingSide,
)
from .models import (
    LiquidityLifecycleEvent,
    OrderbookLevelSnapshot,
    TrackedWall,
)


class PersistenceTracker(BaseSpoofingTracker):
    """
    Stateful tracker життєвого циклу великих рівнів ліквідності в стакані.

    Основні задачі:
    - створювати та оновлювати tracked walls
    - відстежувати lifetime, size evolution, pull/fill dynamics
    - генерувати lifecycle events
    - очищати прострочений стан
    - надавати API для детекторів spoofing-пакета

    Очікуване використання:
    1. analyzer отримує orderbook update
    2. analyzer / wall detector передає сюди relevant level snapshots
    3. tracker повертає список lifecycle events та актуальні tracked walls
    4. інші детектори аналізують цей state

    Важливо:
    - Tracker сам по собі не вирішує, чи це spoofing.
    - Він лише веде життєвий цикл стінок/рівнів.
    """

    component = SpoofingComponent.PERSISTENCE_TRACKER

    def __init__(
        self,
        event_bus: Any | None,
        config: SpoofingConfig,
    ) -> None:
        super().__init__(event_bus=event_bus, config=config)

        self._walls_by_id: dict[str, TrackedWall] = {}
        self._wall_ids_by_symbol: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._history_by_level: dict[str, list[LiquidityLifecycleEvent]] = defaultdict(list)

        self._last_cleanup_at: datetime | None = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_wall(self, wall_id: str) -> TrackedWall | None:
        return self._walls_by_id.get(wall_id)

    def get_wall_by_level(
        self,
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide,
        price: float,
    ) -> TrackedWall | None:
        wall_id = self.build_wall_id(
            exchange=exchange,
            symbol=symbol,
            side=side,
            price=price,
        )
        return self._walls_by_id.get(wall_id)

    def get_walls_for_symbol(
        self,
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide | None = None,
        state: OrderbookWallState | None = None,
    ) -> list[TrackedWall]:
        key = (exchange, symbol)
        wall_ids = self._wall_ids_by_symbol.get(key, set())
        walls: list[TrackedWall] = []

        for wall_id in wall_ids:
            wall = self._walls_by_id.get(wall_id)
            if wall is None:
                continue
            if side is not None and wall.side != side:
                continue
            if state is not None and wall.state != state:
                continue
            walls.append(wall)

        walls.sort(key=lambda item: item.last_seen_at, reverse=True)
        return walls

    def get_recent_history(
        self,
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide,
        price: float,
        limit: int = 50,
    ) -> list[LiquidityLifecycleEvent]:
        level_key = self.build_level_key(
            exchange=exchange,
            symbol=symbol,
            side=side,
            price=price,
        )
        events = self._history_by_level.get(level_key, [])
        return events[-limit:]

    def upsert_snapshot(
        self,
        snapshot: OrderbookLevelSnapshot,
    ) -> tuple[TrackedWall, list[LiquidityLifecycleEvent]]:
        """
        Створює або оновлює tracked wall на основі snapshot рівня стакана.

        Якщо рівень новий -> створює wall + CREATED event.
        Якщо рівень вже трекається -> оновлює wall + UPDATED / PARTIALLY_FILLED event.
        """
        wall_id = self.build_wall_id(
            exchange=snapshot.exchange,
            symbol=snapshot.symbol,
            side=snapshot.side,
            price=snapshot.price,
        )

        existing = self._walls_by_id.get(wall_id)
        if existing is None:
            wall, events = self._create_wall(snapshot)
            self._maybe_enforce_symbol_limit(snapshot.exchange, snapshot.symbol)
            return wall, events

        wall, events = self._update_wall(existing, snapshot)
        return wall, events

    def upsert_many(
        self,
        snapshots: Iterable[OrderbookLevelSnapshot],
    ) -> tuple[list[TrackedWall], list[LiquidityLifecycleEvent]]:
        """
        Batch upsert для кількох snapshot-рівнів.
        """
        walls: list[TrackedWall] = []
        events: list[LiquidityLifecycleEvent] = []

        for snapshot in snapshots:
            wall, wall_events = self.upsert_snapshot(snapshot)
            walls.append(wall)
            events.extend(wall_events)

        return walls, events

    def mark_pulled(
        self,
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide,
        price: float,
        timestamp: datetime | None = None,
        removed_size: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[TrackedWall | None, LiquidityLifecycleEvent | None]:
        """
        Позначає рівень як знятий з orderbook без повного виконання.
        """
        wall = self.get_wall_by_level(
            exchange=exchange,
            symbol=symbol,
            side=side,
            price=price,
        )
        if wall is None:
            return None, None

        ts = self.ensure_utc(timestamp)
        size_before = wall.current_size
        inferred_removed = size_before if removed_size is None else max(0.0, removed_size)
        size_after = max(0.0, size_before - inferred_removed)

        wall.last_seen_at = ts
        wall.current_size = size_after
        wall.min_size = min(wall.min_size, wall.current_size)
        wall.total_removed_size += max(0.0, size_before - size_after)
        wall.estimated_pulled_size += max(0.0, size_before - size_after)
        wall.state = OrderbookWallState.PULLED

        event = self._make_lifecycle_event(
            wall=wall,
            event_type=LiquidityEventType.PULLED,
            size_before=size_before,
            size_after=size_after,
            timestamp=ts,
            metadata=metadata,
        )
        self._store_history_event(wall, event)

        self.log_debug(
            "Wall marked as pulled",
            wall_id=wall.wall_id,
            symbol=wall.symbol,
            exchange=wall.exchange,
            side=wall.side.value,
            price=wall.price,
            removed_size=inferred_removed,
            lifetime_ms=wall.lifetime_ms,
        )
        return wall, event

    def mark_filled(
        self,
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide,
        price: float,
        timestamp: datetime | None = None,
        filled_size: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[TrackedWall | None, LiquidityLifecycleEvent | None]:
        """
        Позначає рівень як повністю або майже повністю виконаний.
        """
        wall = self.get_wall_by_level(
            exchange=exchange,
            symbol=symbol,
            side=side,
            price=price,
        )
        if wall is None:
            return None, None

        ts = self.ensure_utc(timestamp)
        size_before = wall.current_size
        inferred_filled = size_before if filled_size is None else max(0.0, filled_size)
        size_after = max(0.0, size_before - inferred_filled)

        wall.last_seen_at = ts
        wall.current_size = size_after
        wall.min_size = min(wall.min_size, wall.current_size)
        wall.total_removed_size += max(0.0, size_before - size_after)
        wall.estimated_filled_size += max(0.0, size_before - size_after)
        wall.state = OrderbookWallState.FILLED if size_after <= self.config.persistence.size_update_epsilon else OrderbookWallState.WEAKENING

        event_type = (
            LiquidityEventType.FULLY_FILLED
            if size_after <= self.config.persistence.size_update_epsilon
            else LiquidityEventType.PARTIALLY_FILLED
        )

        event = self._make_lifecycle_event(
            wall=wall,
            event_type=event_type,
            size_before=size_before,
            size_after=size_after,
            timestamp=ts,
            metadata=metadata,
        )
        self._store_history_event(wall, event)

        self.log_debug(
            "Wall marked as filled",
            wall_id=wall.wall_id,
            symbol=wall.symbol,
            exchange=wall.exchange,
            side=wall.side.value,
            price=wall.price,
            filled_size=inferred_filled,
            lifetime_ms=wall.lifetime_ms,
        )
        return wall, event

    def apply_trade_execution(
        self,
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide,
        price: float,
        trade_size: float,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[TrackedWall | None, LiquidityLifecycleEvent | None]:
        """
        Евристично застосовує trade execution до рівня.
        Це корисно, якщо ти хочеш частково враховувати fills через trade flow.

        side тут — сторона ЛІКВІДНОСТІ, яка стояла в стакані.
        Наприклад:
        - якщо aggressive buyer зняв ask wall, тоді side = ASK
        """
        wall = self.get_wall_by_level(
            exchange=exchange,
            symbol=symbol,
            side=side,
            price=price,
        )
        if wall is None:
            return None, None

        ts = self.ensure_utc(timestamp)
        size_before = wall.current_size
        filled = max(0.0, min(trade_size, size_before))
        size_after = max(0.0, size_before - filled)

        wall.last_seen_at = ts
        wall.current_size = size_after
        wall.min_size = min(wall.min_size, wall.current_size)
        wall.total_removed_size += filled
        wall.estimated_filled_size += filled
        wall.touch_count += 1

        if size_after <= self.config.persistence.size_update_epsilon:
            wall.state = OrderbookWallState.FILLED
            event_type = LiquidityEventType.FULLY_FILLED
        else:
            wall.state = OrderbookWallState.WEAKENING
            event_type = LiquidityEventType.PARTIALLY_FILLED

        event = self._make_lifecycle_event(
            wall=wall,
            event_type=event_type,
            size_before=size_before,
            size_after=size_after,
            timestamp=ts,
            metadata=metadata,
        )
        self._store_history_event(wall, event)

        return wall, event

    def cleanup(self, now: datetime | None = None) -> int:
        """
        Видаляє прострочені tracked walls.

        Стінка вважається простроченою, якщо її не оновлювали довше за wall_ttl_ms.
        Перед видаленням їй ставиться state=EXPIRED і записується EXPIRED event.
        """
        current_time = self.ensure_utc(now)
        ttl_ms = self.config.persistence.wall_ttl_ms

        expired_ids: list[str] = []
        expired_events: list[LiquidityLifecycleEvent] = []

        for wall_id, wall in list(self._walls_by_id.items()):
            age_ms = (current_time - wall.last_seen_at).total_seconds() * 1000.0
            if age_ms <= ttl_ms:
                continue

            wall.state = OrderbookWallState.EXPIRED
            event = self._make_lifecycle_event(
                wall=wall,
                event_type=LiquidityEventType.EXPIRED,
                size_before=wall.current_size,
                size_after=wall.current_size,
                timestamp=current_time,
                metadata={"reason": "ttl_expired"},
            )
            self._store_history_event(wall, event)
            expired_events.append(event)
            expired_ids.append(wall_id)

        for wall_id in expired_ids:
            self._remove_wall_by_id(wall_id)

        self._last_cleanup_at = current_time

        if expired_ids:
            self.log_debug(
                "Persistence tracker cleanup completed",
                expired_count=len(expired_ids),
            )

        return len(expired_ids)

    def maybe_cleanup(self, now: datetime | None = None) -> int:
        """
        Cleanup only if cleanup interval elapsed.
        """
        current_time = self.ensure_utc(now)
        interval_ms = self.config.persistence.cleanup_interval_ms

        if self._last_cleanup_at is None:
            return self.cleanup(current_time)

        elapsed_ms = (current_time - self._last_cleanup_at).total_seconds() * 1000.0
        if elapsed_ms < interval_ms:
            return 0

        return self.cleanup(current_time)

    def prune_history(self, *, max_events_per_level: int = 200) -> int:
        """
        Обрізає надто довгу історію lifecycle events для кожного рівня.
        """
        removed = 0
        for level_key, events in list(self._history_by_level.items()):
            if len(events) <= max_events_per_level:
                continue
            to_remove = len(events) - max_events_per_level
            self._history_by_level[level_key] = events[-max_events_per_level:]
            removed += to_remove
        return removed

    def build_features_from_wall(
        self,
        wall: TrackedWall,
        *,
        current_mid_price: float | None = None,
        repetition_count: int = 0,
    ) -> dict[str, float | int | bool]:
        """
        Базовий helper для майбутніх detector-ів.
        Повертає сирі persistence-related features.
        """
        reference_mid = current_mid_price or wall.mid_price_at_creation or 0.0
        distance_bps = self.bps_distance(wall.price, reference_mid) if reference_mid > 0 else 0.0

        return {
            "wall_size": wall.current_size,
            "max_wall_size": wall.max_size,
            "initial_wall_size": wall.initial_size,
            "lifetime_ms": wall.lifetime_ms,
            "fill_ratio": wall.fill_ratio,
            "pull_ratio": wall.pull_ratio,
            "current_to_max_ratio": wall.current_to_max_ratio,
            "updates_count": wall.updates_count,
            "touch_count": wall.touch_count,
            "near_touch_count": wall.near_touch_count,
            "distance_from_mid_bps": distance_bps,
            "repetition_count": repetition_count,
            "is_active": wall.state == OrderbookWallState.ACTIVE,
            "is_pulled": wall.state == OrderbookWallState.PULLED,
            "is_filled": wall.state == OrderbookWallState.FILLED,
            "is_weakening": wall.state == OrderbookWallState.WEAKENING,
        }

    # -------------------------------------------------------------------------
    # Internal wall lifecycle logic
    # -------------------------------------------------------------------------

    def _create_wall(
        self,
        snapshot: OrderbookLevelSnapshot,
    ) -> tuple[TrackedWall, list[LiquidityLifecycleEvent]]:
        wall_id = self.build_wall_id(
            exchange=snapshot.exchange,
            symbol=snapshot.symbol,
            side=snapshot.side,
            price=snapshot.price,
        )

        wall = TrackedWall(
            wall_id=wall_id,
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            side=snapshot.side,
            price=snapshot.price,
            first_seen_at=snapshot.timestamp,
            last_seen_at=snapshot.timestamp,
            initial_size=snapshot.size,
            current_size=snapshot.size,
            max_size=snapshot.size,
            min_size=snapshot.size,
            best_bid_at_creation=snapshot.best_bid,
            best_ask_at_creation=snapshot.best_ask,
            mid_price_at_creation=snapshot.mid_price,
            metadata=dict(snapshot.metadata),
        )

        self._walls_by_id[wall.wall_id] = wall
        self._wall_ids_by_symbol[(wall.exchange, wall.symbol)].add(wall.wall_id)

        event = self._make_lifecycle_event(
            wall=wall,
            event_type=LiquidityEventType.CREATED,
            size_before=0.0,
            size_after=wall.current_size,
            timestamp=snapshot.timestamp,
            metadata={
                "best_bid": snapshot.best_bid,
                "best_ask": snapshot.best_ask,
                "mid_price": snapshot.mid_price,
                "sequence_id": snapshot.sequence_id,
            },
        )
        self._store_history_event(wall, event)

        self.log_debug(
            "Tracked wall created",
            wall_id=wall.wall_id,
            symbol=wall.symbol,
            exchange=wall.exchange,
            side=wall.side.value,
            price=wall.price,
            size=wall.current_size,
        )

        return wall, [event]

    def _update_wall(
        self,
        wall: TrackedWall,
        snapshot: OrderbookLevelSnapshot,
    ) -> tuple[TrackedWall, list[LiquidityLifecycleEvent]]:
        size_before = wall.current_size
        size_after = snapshot.size
        delta = size_after - size_before
        ts = snapshot.timestamp

        wall.last_seen_at = ts
        wall.updates_count += 1
        wall.current_size = size_after
        wall.max_size = max(wall.max_size, size_after)
        wall.min_size = min(wall.min_size, size_after)

        events: list[LiquidityLifecycleEvent] = []

        if delta > self.config.persistence.size_update_epsilon:
            wall.total_added_size += delta
            wall.state = OrderbookWallState.ACTIVE

        elif delta < -self.config.persistence.size_update_epsilon:
            removed = abs(delta)
            wall.total_removed_size += removed

            if self._is_partial_fill_candidate(wall=wall, snapshot=snapshot, removed_size=removed):
                wall.estimated_filled_size += removed
                wall.state = OrderbookWallState.WEAKENING

                event = self._make_lifecycle_event(
                    wall=wall,
                    event_type=LiquidityEventType.PARTIALLY_FILLED,
                    size_before=size_before,
                    size_after=size_after,
                    timestamp=ts,
                    metadata={
                        "best_bid": snapshot.best_bid,
                        "best_ask": snapshot.best_ask,
                        "mid_price": snapshot.mid_price,
                        "reason": "heuristic_partial_fill",
                        "sequence_id": snapshot.sequence_id,
                    },
                )
                events.append(event)
                self._store_history_event(wall, event)
            else:
                wall.estimated_pulled_size += removed
                wall.state = (
                    OrderbookWallState.PULLED
                    if size_after <= self.config.persistence.size_update_epsilon
                    else OrderbookWallState.WEAKENING
                )

                event = self._make_lifecycle_event(
                    wall=wall,
                    event_type=LiquidityEventType.UPDATED,
                    size_before=size_before,
                    size_after=size_after,
                    timestamp=ts,
                    metadata={
                        "best_bid": snapshot.best_bid,
                        "best_ask": snapshot.best_ask,
                        "mid_price": snapshot.mid_price,
                        "reason": "size_reduction",
                        "sequence_id": snapshot.sequence_id,
                    },
                )
                events.append(event)
                self._store_history_event(wall, event)

        else:
            wall.state = wall.state if wall.state != OrderbookWallState.PULLED else OrderbookWallState.ACTIVE

            event = self._make_lifecycle_event(
                wall=wall,
                event_type=LiquidityEventType.UPDATED,
                size_before=size_before,
                size_after=size_after,
                timestamp=ts,
                metadata={
                    "best_bid": snapshot.best_bid,
                    "best_ask": snapshot.best_ask,
                    "mid_price": snapshot.mid_price,
                    "reason": "heartbeat_update",
                    "sequence_id": snapshot.sequence_id,
                },
            )
            events.append(event)
            self._store_history_event(wall, event)

        if self._is_near_touch(wall, snapshot):
            wall.near_touch_count += 1

        if size_after <= self.config.persistence.size_update_epsilon:
            # Не видаляємо одразу, щоб детектори ще могли побачити стан.
            # Cleanup прибере його пізніше.
            if wall.estimated_filled_size >= wall.estimated_pulled_size:
                wall.state = OrderbookWallState.FILLED
            else:
                wall.state = OrderbookWallState.PULLED

        self.log_debug(
            "Tracked wall updated",
            wall_id=wall.wall_id,
            symbol=wall.symbol,
            exchange=wall.exchange,
            side=wall.side.value,
            price=wall.price,
            size_before=size_before,
            size_after=size_after,
            state=wall.state.value,
            updates_count=wall.updates_count,
        )

        return wall, events

    def _is_partial_fill_candidate(
        self,
        *,
        wall: TrackedWall,
        snapshot: OrderbookLevelSnapshot,
        removed_size: float,
    ) -> bool:
        """
        Проста евристика:
        - якщо увімкнено estimate_fill_on_touch_only, то reduction розглядаємо як fill
          лише коли рівень був близько до best quote
        - інакше reduction скоріше трактуємо як pull
        """
        if removed_size <= self.config.persistence.size_update_epsilon:
            return False

        if not self.config.persistence.estimate_fill_on_touch_only:
            return self.config.persistence.estimate_fill_from_trade_flow

        return self._is_near_touch(wall, snapshot)

    def _is_near_touch(
        self,
        wall: TrackedWall,
        snapshot: OrderbookLevelSnapshot,
    ) -> bool:
        """
        Визначає, чи рівень знаходився достатньо близько до best quote,
        щоб reduction size міг бути fill, а не pull.
        """
        if wall.side == SpoofingSide.BID:
            if snapshot.best_bid is None:
                return False
            return abs(wall.price - snapshot.best_bid) <= self._price_epsilon(wall.price)

        if wall.side == SpoofingSide.ASK:
            if snapshot.best_ask is None:
                return False
            return abs(wall.price - snapshot.best_ask) <= self._price_epsilon(wall.price)

        return False

    def _price_epsilon(self, price: float) -> float:
        decimals = self.config.persistence.price_rounding_decimals
        return 10 ** (-decimals) if decimals > 0 else max(price * 1e-8, 1e-12)

    def _make_lifecycle_event(
        self,
        *,
        wall: TrackedWall,
        event_type: LiquidityEventType,
        size_before: float,
        size_after: float,
        timestamp: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> LiquidityLifecycleEvent:
        return LiquidityLifecycleEvent(
            wall_id=wall.wall_id,
            symbol=wall.symbol,
            exchange=wall.exchange,
            side=wall.side,
            event_type=event_type,
            price=wall.price,
            size_before=size_before,
            size_after=size_after,
            delta_size=size_after - size_before,
            timestamp=timestamp,
            metadata=metadata or {},
        )

    def _store_history_event(
        self,
        wall: TrackedWall,
        event: LiquidityLifecycleEvent,
    ) -> None:
        level_key = self.build_level_key(
            exchange=wall.exchange,
            symbol=wall.symbol,
            side=wall.side,
            price=wall.price,
        )
        self._history_by_level[level_key].append(event)

    def _remove_wall_by_id(self, wall_id: str) -> None:
        wall = self._walls_by_id.pop(wall_id, None)
        if wall is None:
            return

        key = (wall.exchange, wall.symbol)
        wall_ids = self._wall_ids_by_symbol.get(key)
        if wall_ids is not None:
            wall_ids.discard(wall_id)
            if not wall_ids:
                self._wall_ids_by_symbol.pop(key, None)

    def _maybe_enforce_symbol_limit(self, exchange: str, symbol: str) -> None:
        """
        Обмежує кількість tracked walls на символ.
        Видаляє найстаріші / найменш релевантні.
        """
        limit = self.config.analyzer.max_tracked_walls_per_symbol
        key = (exchange, symbol)
        wall_ids = self._wall_ids_by_symbol.get(key, set())

        if len(wall_ids) <= limit:
            return

        walls = [
            self._walls_by_id[wall_id]
            for wall_id in wall_ids
            if wall_id in self._walls_by_id
        ]
        walls.sort(key=lambda item: (item.last_seen_at, item.first_seen_at, item.max_size))

        overflow = len(walls) - limit
        for wall in walls[:overflow]:
            self._remove_wall_by_id(wall.wall_id)

    # -------------------------------------------------------------------------
    # Static helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def build_wall_id(
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide,
        price: float,
    ) -> str:
        return f"{exchange}:{symbol}:{side.value}:{price:.12f}"

    @staticmethod
    def build_level_key(
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide,
        price: float,
    ) -> str:
        return f"{exchange}:{symbol}:{side.value}:{price:.12f}"

    # -------------------------------------------------------------------------
    # Debug / inspection helpers
    # -------------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        by_state: dict[str, int] = defaultdict(int)
        by_symbol: dict[str, int] = defaultdict(int)

        for wall in self._walls_by_id.values():
            by_state[wall.state.value] += 1
            by_symbol[f"{wall.exchange}:{wall.symbol}"] += 1

        return {
            "tracked_walls": len(self._walls_by_id),
            "symbols": dict(by_symbol),
            "states": dict(by_state),
            "history_levels": len(self._history_by_level),
            "last_cleanup_at": self._last_cleanup_at.isoformat() if self._last_cleanup_at else None,
        }

    def snapshot_state(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
    ) -> list[TrackedWall]:
        """
        Повертає копії tracked walls для безпечного читання зовнішніми модулями.
        """
        items: list[TrackedWall] = []

        for wall in self._walls_by_id.values():
            if exchange is not None and wall.exchange != exchange:
                continue
            if symbol is not None and wall.symbol != symbol:
                continue
            items.append(replace(wall))

        items.sort(key=lambda item: item.last_seen_at, reverse=True)
        return items