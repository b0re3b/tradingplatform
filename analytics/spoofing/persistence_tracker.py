from __future__ import annotations
from core.logger import get_logger

from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from typing import Any, Iterable

from core.event_bus import EventBus
from core.scheduler import Scheduler

from .base import BaseSpoofingTracker
from .config import SpoofingConfig
from .enums import (
    LiquidityEventType,
    OrderbookWallState,
    SpoofingComponent,
    SpoofingSide,
)
from .models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    LiquidityLifecycleEvent,
    OrderbookLevelSnapshot,
    SpoofingFeatures,
    SpoofingKey,
    TrackedWall,
    make_spoofing_key,
    spoofing_key_to_dict,
)


class PersistenceTracker(BaseSpoofingTracker):
    """
    Stateful tracker життєвого циклу великих рівнів ліквідності в стакані.

    Відповідає тільки за in-memory state та lifecycle events:
    - створення / оновлення tracked walls;
    - lifetime, size evolution, pull/fill dynamics;
    - lifecycle history;
    - cleanup прострочених walls;
    - scoped read API для detector-ів.

    Correct scope:
        exchange + market_type + symbol + timeframe

    Correct input flow:
        exchange adapters
            -> market.orderbook
            -> OrderBookCache
            -> market.orderbook.updated
            -> SpoofingAnalyzer
            -> PersistenceTracker

    Важливо:
    - не визначає spoofing самостійно;
    - не підписується на EventBus;
    - не публікує EventBus events;
    - не запускає власні asyncio loops;
    - periodic cleanup має реєструвати SpoofingAnalyzer через Scheduler.add_interval_job().
    """

    component = SpoofingComponent.PERSISTENCE_TRACKER

    def __init__(
        self,
        *,
        event_bus: EventBus | None,
        scheduler: Scheduler | None,
        config: SpoofingConfig,
    ) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__init__", _analytics_args)
        except Exception:
            pass
        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config,
        )

        self._walls_by_id: dict[str, TrackedWall] = {}

        # New canonical index:
        # key = (exchange, market_type, symbol, timeframe)
        self._wall_ids_by_key: dict[SpoofingKey, set[str]] = defaultdict(set)

        # Lifecycle history per scoped price level:
        # exchange:market_type:symbol:timeframe:side:price
        self._history_by_level: dict[str, list[LiquidityLifecycleEvent]] = defaultdict(list)

        self._last_cleanup_at: datetime | None = None

    # -------------------------------------------------------------------------
    # Public read API
    # -------------------------------------------------------------------------

    def get_wall(self, wall_id: str) -> TrackedWall | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_wall", _analytics_args)
        except Exception:
            pass
        return self._walls_by_id.get(wall_id)

    def get_wall_by_level(
        self,
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide,
        price: float,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> TrackedWall | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_wall_by_level", _analytics_args)
        except Exception:
            pass
        wall_id = self.build_wall_id(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            price=price,
        )
        return self._walls_by_id.get(wall_id)

    def get_wall_by_snapshot(
        self,
        snapshot: OrderbookLevelSnapshot,
    ) -> TrackedWall | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_wall_by_snapshot", _analytics_args)
        except Exception:
            pass
        return self.get_wall_by_level(
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            side=snapshot.side,
            price=snapshot.price,
        )

    def get_walls_for_key(
        self,
        key: SpoofingKey,
        *,
        side: SpoofingSide | None = None,
        state: OrderbookWallState | None = None,
    ) -> list[TrackedWall]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_walls_for_key", _analytics_args)
        except Exception:
            pass
        wall_ids = self._wall_ids_by_key.get(key, set())

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

    def get_walls_for_scope(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        side: SpoofingSide | None = None,
        state: OrderbookWallState | None = None,
    ) -> list[TrackedWall]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_walls_for_scope", _analytics_args)
        except Exception:
            pass
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        return self.get_walls_for_key(key, side=side, state=state)

    def get_walls_for_symbol(
        self,
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide | None = None,
        state: OrderbookWallState | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> list[TrackedWall]:
        """
        Backward-compatible helper.

        New code should use get_walls_for_key() або get_walls_for_scope().
        Якщо market_type/timeframe не передані, повертає всі scope-и для
        exchange + symbol.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_walls_for_symbol", _analytics_args)
        except Exception:
            pass
        normalized_exchange = self.normalize_exchange(exchange)
        normalized_symbol = self.normalize_symbol(symbol)
        normalized_market_type = (
            self.normalize_market_type(market_type)
            if market_type is not None
            else None
        )
        normalized_timeframe = (
            self.normalize_timeframe(timeframe)
            if timeframe is not None
            else None
        )

        walls: list[TrackedWall] = []

        for key, wall_ids in self._wall_ids_by_key.items():
            key_exchange, key_market_type, key_symbol, key_timeframe = key

            if key_exchange != normalized_exchange:
                continue
            if key_symbol != normalized_symbol:
                continue
            if normalized_market_type is not None and key_market_type != normalized_market_type:
                continue
            if normalized_timeframe is not None and key_timeframe != normalized_timeframe:
                continue

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
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        limit: int = 50,
    ) -> list[LiquidityLifecycleEvent]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_recent_history", _analytics_args)
        except Exception:
            pass
        level_key = self.build_level_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            price=price,
        )
        events = self._history_by_level.get(level_key, [])

        if limit <= 0:
            return []

        return events[-limit:]

    def get_recent_history_for_wall(
        self,
        wall: TrackedWall,
        *,
        limit: int = 50,
    ) -> list[LiquidityLifecycleEvent]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_recent_history_for_wall", _analytics_args)
        except Exception:
            pass
        return self.get_recent_history(
            exchange=wall.exchange,
            market_type=wall.market_type,
            symbol=wall.symbol,
            timeframe=wall.timeframe,
            side=wall.side,
            price=wall.price,
            limit=limit,
        )

    def snapshot_state(
        self,
        *,
        key: SpoofingKey | None = None,
        exchange: str | None = None,
        symbol: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> list[TrackedWall]:
        """
        Повертає копії tracked walls для безпечного читання зовнішніми модулями.

        New code:
            snapshot_state(key=...)

        Legacy-compatible filters:
            exchange/symbol/market_type/timeframe
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "snapshot_state", _analytics_args)
        except Exception:
            pass
        items: list[TrackedWall] = []

        if key is not None:
            for wall in self.get_walls_for_key(key):
                items.append(replace(wall))

            items.sort(key=lambda item: item.last_seen_at, reverse=True)
            return items

        normalized_exchange = (
            self.normalize_exchange(exchange)
            if exchange is not None
            else None
        )
        normalized_symbol = (
            self.normalize_symbol(symbol)
            if symbol is not None
            else None
        )
        normalized_market_type = (
            self.normalize_market_type(market_type)
            if market_type is not None
            else None
        )
        normalized_timeframe = (
            self.normalize_timeframe(timeframe)
            if timeframe is not None
            else None
        )

        for wall in self._walls_by_id.values():
            if normalized_exchange is not None and wall.exchange != normalized_exchange:
                continue
            if normalized_symbol is not None and wall.symbol != normalized_symbol:
                continue
            if normalized_market_type is not None and wall.market_type != normalized_market_type:
                continue
            if normalized_timeframe is not None and wall.timeframe != normalized_timeframe:
                continue

            items.append(replace(wall))

        items.sort(key=lambda item: item.last_seen_at, reverse=True)
        return items

    # -------------------------------------------------------------------------
    # Public write/update API
    # -------------------------------------------------------------------------

    def upsert_snapshot(
        self,
        snapshot: OrderbookLevelSnapshot,
    ) -> tuple[TrackedWall, list[LiquidityLifecycleEvent]]:
        """
        Створює або оновлює tracked wall на основі normalized snapshot рівня стакана.

        Production source:
            OrderBookCache -> market.orderbook.updated -> SpoofingAnalyzer
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "upsert_snapshot", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.persistence.enabled:
            raise RuntimeError("PersistenceTracker is disabled by config")

        wall_id = self.build_wall_id(
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            side=snapshot.side,
            price=snapshot.price,
        )

        existing = self._walls_by_id.get(wall_id)
        if existing is None:
            wall, events = self._create_wall(snapshot)
            self._maybe_enforce_key_limit(wall.key)
            return wall, events

        return self._update_wall(existing, snapshot)

    def upsert_many(
        self,
        snapshots: Iterable[OrderbookLevelSnapshot],
    ) -> tuple[list[TrackedWall], list[LiquidityLifecycleEvent]]:
        """
        Batch upsert для кількох snapshot-рівнів.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "upsert_many", _analytics_args)
        except Exception:
            pass
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
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        timestamp: datetime | None = None,
        removed_size: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[TrackedWall | None, LiquidityLifecycleEvent | None]:
        """
        Позначає рівень як знятий з orderbook без повного виконання.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "mark_pulled", _analytics_args)
        except Exception:
            pass
        wall = self.get_wall_by_level(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            price=price,
        )
        if wall is None:
            return None, None

        ts = self.ensure_utc(timestamp)
        size_before = wall.current_size
        inferred_removed = size_before if removed_size is None else max(0.0, removed_size)
        size_after = max(0.0, size_before - inferred_removed)

        removed_delta = max(0.0, size_before - size_after)

        wall.last_seen_at = ts
        wall.current_size = size_after
        wall.min_size = min(wall.min_size, wall.current_size)
        wall.total_removed_size += removed_delta
        wall.estimated_pulled_size += removed_delta
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
            market_type=wall.market_type,
            timeframe=wall.timeframe,
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
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        timestamp: datetime | None = None,
        filled_size: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[TrackedWall | None, LiquidityLifecycleEvent | None]:
        """
        Позначає рівень як повністю або частково виконаний.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "mark_filled", _analytics_args)
        except Exception:
            pass
        wall = self.get_wall_by_level(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            price=price,
        )
        if wall is None:
            return None, None

        ts = self.ensure_utc(timestamp)
        size_before = wall.current_size
        inferred_filled = size_before if filled_size is None else max(0.0, filled_size)
        size_after = max(0.0, size_before - inferred_filled)

        filled_delta = max(0.0, size_before - size_after)

        wall.last_seen_at = ts
        wall.current_size = size_after
        wall.min_size = min(wall.min_size, wall.current_size)
        wall.total_removed_size += filled_delta
        wall.estimated_filled_size += filled_delta

        is_fully_filled = size_after <= self.config.persistence.size_update_epsilon
        wall.state = (
            OrderbookWallState.FILLED
            if is_fully_filled
            else OrderbookWallState.WEAKENING
        )

        event = self._make_lifecycle_event(
            wall=wall,
            event_type=LiquidityEventType.FULLY_FILLED
            if is_fully_filled
            else LiquidityEventType.PARTIALLY_FILLED,
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
            market_type=wall.market_type,
            timeframe=wall.timeframe,
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
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[TrackedWall | None, LiquidityLifecycleEvent | None]:
        """
        Евристично застосовує trade execution до tracked wall.

        side тут — сторона ліквідності, яка стояла в стакані:
        - aggressive buyer зняв ask wall -> side = ASK;
        - aggressive seller зняв bid wall -> side = BID.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "apply_trade_execution", _analytics_args)
        except Exception:
            pass
        wall = self.get_wall_by_level(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            price=price,
        )
        if wall is None:
            return None, None

        ts = self.ensure_utc(timestamp)
        size_before = wall.current_size
        filled = max(0.0, min(float(trade_size), size_before))
        size_after = max(0.0, size_before - filled)

        wall.last_seen_at = ts
        wall.current_size = size_after
        wall.min_size = min(wall.min_size, wall.current_size)
        wall.total_removed_size += filled
        wall.estimated_filled_size += filled
        wall.touch_count += 1

        is_fully_filled = size_after <= self.config.persistence.size_update_epsilon
        wall.state = (
            OrderbookWallState.FILLED
            if is_fully_filled
            else OrderbookWallState.WEAKENING
        )

        event = self._make_lifecycle_event(
            wall=wall,
            event_type=LiquidityEventType.FULLY_FILLED
            if is_fully_filled
            else LiquidityEventType.PARTIALLY_FILLED,
            size_before=size_before,
            size_after=size_after,
            timestamp=ts,
            metadata=metadata,
        )
        self._store_history_event(wall, event)

        return wall, event

    # -------------------------------------------------------------------------
    # Cleanup API
    # -------------------------------------------------------------------------

    def cleanup(self, now: datetime | None = None) -> int:
        """
        Видаляє прострочені tracked walls.

        Стінка вважається простроченою, якщо її не оновлювали довше за wall_ttl_ms.
        Перед видаленням їй ставиться state=EXPIRED і записується EXPIRED event.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "cleanup", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.persistence.enabled:
            return 0

        current_time = self.ensure_utc(now)
        ttl_ms = self.config.persistence.wall_ttl_ms

        expired_ids: list[str] = []

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
            expired_ids.append(wall_id)

        for wall_id in expired_ids:
            self._remove_wall_by_id(wall_id)

        self._last_cleanup_at = current_time

        pruned_count = self.prune_history(
            max_events_per_level=self.config.persistence.max_history_events_per_level,
        )

        if expired_ids or pruned_count:
            self.log_debug(
                "Persistence tracker cleanup completed",
                expired_count=len(expired_ids),
                pruned_history_events=pruned_count,
            )

        return len(expired_ids)

    def cleanup_expired(self, now: datetime | None = None) -> int:
        """
        Явний alias для cleanup expired walls.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "cleanup_expired", _analytics_args)
        except Exception:
            pass
        return self.cleanup(now)

    def maybe_cleanup(self, now: datetime | None = None) -> int:
        """
        Виконує cleanup тільки якщо минув persistence.cleanup_interval_ms.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "maybe_cleanup", _analytics_args)
        except Exception:
            pass
        current_time = self.ensure_utc(now)
        interval_ms = self.config.persistence.cleanup_interval_ms

        if self._last_cleanup_at is None:
            return self.cleanup(current_time)

        elapsed_ms = (current_time - self._last_cleanup_at).total_seconds() * 1000.0
        if elapsed_ms < interval_ms:
            return 0

        return self.cleanup(current_time)

    def prune_history(self, *, max_events_per_level: int | None = None) -> int:
        """
        Обрізає надто довгу історію lifecycle events для кожного рівня.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "prune_history", _analytics_args)
        except Exception:
            pass
        limit = (
            self.config.persistence.max_history_events_per_level
            if max_events_per_level is None
            else max_events_per_level
        )

        if limit <= 0:
            return 0

        removed = 0
        for level_key, events in list(self._history_by_level.items()):
            if len(events) <= limit:
                continue

            to_remove = len(events) - limit
            self._history_by_level[level_key] = events[-limit:]
            removed += to_remove

        return removed

    # -------------------------------------------------------------------------
    # Feature helpers
    # -------------------------------------------------------------------------

    def build_features_from_wall(
        self,
        wall: TrackedWall,
        *,
        current_mid_price: float | None = None,
        repetition_count: int = 0,
    ) -> SpoofingFeatures:
        """
        Будує базові persistence-related features для detector-ів.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_features_from_wall", _analytics_args)
        except Exception:
            pass
        reference_mid = current_mid_price or wall.mid_price_at_creation or 0.0
        distance_bps = (
            self.bps_distance(wall.price, reference_mid)
            if reference_mid > 0
            else 0.0
        )

        cancel_to_fill_ratio = (
            wall.estimated_pulled_size / wall.estimated_filled_size
            if wall.estimated_filled_size > 0
            else wall.estimated_pulled_size
        )

        return SpoofingFeatures(
            symbol=wall.symbol,
            exchange=wall.exchange,
            market_type=wall.market_type,
            timeframe=wall.timeframe,
            exchange_symbol=wall.exchange_symbol,
            side=wall.side,
            price=wall.price,
            wall_size=wall.current_size,
            wall_size_ratio=wall.current_to_max_ratio,
            distance_from_mid_bps=distance_bps,
            lifetime_ms=wall.lifetime_ms,
            updates_count=wall.updates_count,
            repetition_count=repetition_count,
            fill_ratio=wall.fill_ratio,
            pull_ratio=wall.pull_ratio,
            cancel_to_fill_ratio=cancel_to_fill_ratio,
            is_fast_pull=wall.state == OrderbookWallState.PULLED,
            timestamp=wall.last_seen_at,
            metadata={
                "wall_id": wall.wall_id,
                "scope": spoofing_key_to_dict(wall.key),
                "max_wall_size": wall.max_size,
                "initial_wall_size": wall.initial_size,
                "current_to_max_ratio": wall.current_to_max_ratio,
                "touch_count": wall.touch_count,
                "near_touch_count": wall.near_touch_count,
                "state": wall.state.value,
            },
        )

    # -------------------------------------------------------------------------
    # Internal wall lifecycle logic
    # -------------------------------------------------------------------------

    def _create_wall(
        self,
        snapshot: OrderbookLevelSnapshot,
    ) -> tuple[TrackedWall, list[LiquidityLifecycleEvent]]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_create_wall", _analytics_args)
        except Exception:
            pass
        normalized_price = self._normalize_price(snapshot.price)

        wall_id = self.build_wall_id(
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            side=snapshot.side,
            price=normalized_price,
        )

        wall = TrackedWall(
            wall_id=wall_id,
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            timeframe=snapshot.timeframe,
            exchange_symbol=snapshot.exchange_symbol,
            side=snapshot.side,
            price=normalized_price,
            first_seen_at=snapshot.timestamp,
            last_seen_at=snapshot.timestamp,
            initial_size=snapshot.size,
            current_size=snapshot.size,
            max_size=snapshot.size,
            min_size=snapshot.size,
            best_bid_at_creation=snapshot.best_bid,
            best_ask_at_creation=snapshot.best_ask,
            mid_price_at_creation=snapshot.mid_price,
            metadata={
                **dict(snapshot.metadata),
                "scope": spoofing_key_to_dict(snapshot.key),
                "sequence_id": snapshot.sequence_id,
            },
        )

        self._walls_by_id[wall.wall_id] = wall
        self._wall_ids_by_key[wall.key].add(wall.wall_id)

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
            market_type=wall.market_type,
            timeframe=wall.timeframe,
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
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_update_wall", _analytics_args)
        except Exception:
            pass
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
                    "reason": "size_increase",
                    "sequence_id": snapshot.sequence_id,
                },
            )
            events.append(event)
            self._store_history_event(wall, event)

        elif delta < -self.config.persistence.size_update_epsilon:
            event = self._handle_size_reduction(
                wall=wall,
                snapshot=snapshot,
                size_before=size_before,
                size_after=size_after,
                removed=abs(delta),
            )
            events.append(event)

        else:
            if wall.state == OrderbookWallState.PULLED:
                wall.state = OrderbookWallState.ACTIVE

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
            if wall.estimated_filled_size >= wall.estimated_pulled_size:
                wall.state = OrderbookWallState.FILLED
            else:
                wall.state = OrderbookWallState.PULLED

        self.log_debug(
            "Tracked wall updated",
            wall_id=wall.wall_id,
            symbol=wall.symbol,
            exchange=wall.exchange,
            market_type=wall.market_type,
            timeframe=wall.timeframe,
            side=wall.side.value,
            price=wall.price,
            size_before=size_before,
            size_after=size_after,
            state=wall.state.value,
            updates_count=wall.updates_count,
        )

        return wall, events

    def _handle_size_reduction(
        self,
        *,
        wall: TrackedWall,
        snapshot: OrderbookLevelSnapshot,
        size_before: float,
        size_after: float,
        removed: float,
    ) -> LiquidityLifecycleEvent:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_handle_size_reduction", _analytics_args)
        except Exception:
            pass
        wall.total_removed_size += removed

        if self._is_partial_fill_candidate(
            wall=wall,
            snapshot=snapshot,
            removed_size=removed,
        ):
            wall.estimated_filled_size += removed
            wall.state = OrderbookWallState.WEAKENING
            event_type = LiquidityEventType.PARTIALLY_FILLED
            reason = "heuristic_partial_fill"
        else:
            wall.estimated_pulled_size += removed
            wall.state = (
                OrderbookWallState.PULLED
                if size_after <= self.config.persistence.size_update_epsilon
                else OrderbookWallState.WEAKENING
            )
            event_type = (
                LiquidityEventType.PULLED
                if wall.state == OrderbookWallState.PULLED
                else LiquidityEventType.UPDATED
            )
            reason = "size_reduction"

        event = self._make_lifecycle_event(
            wall=wall,
            event_type=event_type,
            size_before=size_before,
            size_after=size_after,
            timestamp=snapshot.timestamp,
            metadata={
                "best_bid": snapshot.best_bid,
                "best_ask": snapshot.best_ask,
                "mid_price": snapshot.mid_price,
                "reason": reason,
                "sequence_id": snapshot.sequence_id,
            },
        )
        self._store_history_event(wall, event)
        return event

    def _is_partial_fill_candidate(
        self,
        *,
        wall: TrackedWall,
        snapshot: OrderbookLevelSnapshot,
        removed_size: float,
    ) -> bool:
        """
        Евристика fill-vs-pull:
        - якщо estimate_fill_on_touch_only=True, reduction вважається fill
          тільки коли рівень був біля best quote;
        - якщо estimate_fill_on_touch_only=False, використовуємо
          estimate_fill_from_trade_flow.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_partial_fill_candidate", _analytics_args)
        except Exception:
            pass
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
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_near_touch", _analytics_args)
        except Exception:
            pass
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
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_price_epsilon", _analytics_args)
        except Exception:
            pass
        decimals = self.config.persistence.price_rounding_decimals
        if decimals > 0:
            return 10 ** (-decimals)
        return max(price * 1e-8, 1e-12)

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
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_make_lifecycle_event", _analytics_args)
        except Exception:
            pass
        return LiquidityLifecycleEvent(
            wall_id=wall.wall_id,
            symbol=wall.symbol,
            exchange=wall.exchange,
            market_type=wall.market_type,
            timeframe=wall.timeframe,
            exchange_symbol=wall.exchange_symbol,
            side=wall.side,
            event_type=event_type,
            price=wall.price,
            size_before=size_before,
            size_after=size_after,
            delta_size=size_after - size_before,
            timestamp=timestamp,
            metadata={
                **(metadata or {}),
                "scope": spoofing_key_to_dict(wall.key),
            },
        )

    def _store_history_event(
        self,
        wall: TrackedWall,
        event: LiquidityLifecycleEvent,
    ) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_store_history_event", _analytics_args)
        except Exception:
            pass
        level_key = self.build_level_key(
            exchange=wall.exchange,
            market_type=wall.market_type,
            symbol=wall.symbol,
            timeframe=wall.timeframe,
            side=wall.side,
            price=wall.price,
        )

        history = self._history_by_level[level_key]
        history.append(event)

        limit = self.config.persistence.max_history_events_per_level
        if limit > 0 and len(history) > limit:
            del history[: len(history) - limit]

    def _remove_wall_by_id(self, wall_id: str) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_remove_wall_by_id", _analytics_args)
        except Exception:
            pass
        wall = self._walls_by_id.pop(wall_id, None)
        if wall is None:
            return

        wall_ids = self._wall_ids_by_key.get(wall.key)
        if wall_ids is not None:
            wall_ids.discard(wall_id)
            if not wall_ids:
                self._wall_ids_by_key.pop(wall.key, None)

    def _maybe_enforce_key_limit(self, key: SpoofingKey) -> None:
        """
        Обмежує кількість tracked walls на scoped key.
        Видаляє найстаріші / найменш релевантні.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_maybe_enforce_key_limit", _analytics_args)
        except Exception:
            pass
        limit = getattr(
            self.config.persistence,
            "max_walls_per_key",
            getattr(self.config.persistence, "max_walls_per_symbol", 500),
        )
        if limit <= 0:
            return

        wall_ids = self._wall_ids_by_key.get(key, set())

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

    def _maybe_enforce_symbol_limit(self, exchange: str, symbol: str) -> None:
        """
        Legacy-compatible wrapper.

        New code should use _maybe_enforce_key_limit().
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_maybe_enforce_symbol_limit", _analytics_args)
        except Exception:
            pass
        for key in list(self._wall_ids_by_key.keys()):
            key_exchange, _, key_symbol, _ = key
            if key_exchange == self.normalize_exchange(exchange) and key_symbol == self.normalize_symbol(symbol):
                self._maybe_enforce_key_limit(key)

    # -------------------------------------------------------------------------
    # Key helpers
    # -------------------------------------------------------------------------

    def build_wall_id(
        self,
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide,
        price: float,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> str:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_wall_id", _analytics_args)
        except Exception:
            pass
        normalized_price = self._normalize_price(price)
        key = make_spoofing_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        scope = spoofing_key_to_dict(key)
        parsed_side = self.parse_spoofing_side(side)

        return (
            f"{scope['exchange']}:{scope['market_type']}:{scope['symbol']}:"
            f"{scope['timeframe']}:{parsed_side.value}:{normalized_price:.12f}"
        )

    def build_level_key(
        self,
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide,
        price: float,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> str:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_level_key", _analytics_args)
        except Exception:
            pass
        return self.build_wall_id(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            price=price,
        )

    def _normalize_price(self, price: float) -> float:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_price", _analytics_args)
        except Exception:
            pass
        decimals = max(0, int(self.config.persistence.price_rounding_decimals))
        return round(float(price), decimals)

    # -------------------------------------------------------------------------
    # Debug / inspection helpers
    # -------------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "stats", _analytics_args)
        except Exception:
            pass
        by_state: dict[str, int] = defaultdict(int)
        by_key: dict[str, int] = defaultdict(int)
        by_market: dict[str, int] = defaultdict(int)

        for wall in self._walls_by_id.values():
            by_state[wall.state.value] += 1

            key_dict = spoofing_key_to_dict(wall.key)
            key_label = (
                f"{key_dict['exchange']}:{key_dict['market_type']}:"
                f"{key_dict['symbol']}:{key_dict['timeframe']}"
            )
            market_label = (
                f"{key_dict['exchange']}:{key_dict['market_type']}:{key_dict['symbol']}"
            )

            by_key[key_label] += 1
            by_market[market_label] += 1

        return {
            "tracked_walls": len(self._walls_by_id),
            "keys": dict(by_key),
            "markets": dict(by_market),
            "states": dict(by_state),
            "history_levels": len(self._history_by_level),
            "last_cleanup_at": (
                self._last_cleanup_at.isoformat()
                if self._last_cleanup_at
                else None
            ),
            "scope": "exchange:market_type:symbol:timeframe",
        }


__all__ = ["PersistenceTracker"]