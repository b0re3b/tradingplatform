from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from core.logger import get_logger

from .config import LiquidityConfig
from .enums import LiquidityStatus, SweepStatus
from .liquidity_map import LiquidityMap
from .models import EqualLevel, LiquidityLevel, LiquidityMapSnapshot, StopCluster
from .state import LiquidityState, LiquidityTimeframeState


class LiquidityTopics:
    MARKET_CANDLE_CLOSED = "market.candle.closed"
    MARKET_ORDERBOOK_UPDATED = "market.orderbook.updated"
    MARKET_PRICE_UPDATED = "market.price.updated"

    LIQUIDITY_MAP_UPDATED = "liquidity.map.updated"
    LIQUIDITY_LEVEL_DETECTED = "liquidity.level.detected"
    LIQUIDITY_LEVEL_SWEPT = "liquidity.level.swept"
    LIQUIDITY_STOP_CLUSTER_DETECTED = "liquidity.stop_cluster.detected"


@dataclass(slots=True)
class LiquidityServiceStats:
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    snapshots_built: int = 0
    candle_events_processed: int = 0
    orderbook_events_processed: int = 0
    price_events_processed: int = 0
    emitted_map_updates: int = 0
    emitted_level_events: int = 0
    emitted_cluster_events: int = 0
    errors_count: int = 0
    last_error: str | None = None


@dataclass(slots=True)
class LiquidityServiceContext:
    symbol: str
    timeframe: str
    candles: list[Any] = field(default_factory=list)
    orderbook: dict[str, list[Any]] = field(default_factory=lambda: {"bids": [], "asks": []})
    current_price: float | None = None
    last_snapshot: LiquidityMapSnapshot | None = None
    last_rebuild_at: datetime | None = None


class LiquidityService:
    """
    Оркестратор liquidity-домену.

    Відповідає за:
    - підписку на EventBus
    - накопичення market context per symbol/timeframe
    - виклик LiquidityMap
    - оновлення LiquidityState
    - публікацію liquidity events
    """

    def __init__(
        self,
        config: LiquidityConfig,
        event_bus: Any,
        liquidity_map: LiquidityMap,
        scheduler: Any | None = None,
        max_candles_per_context: int = 500,
    ) -> None:
        self._config = config
        self._config.validate()

        self._event_bus = event_bus
        self._liquidity_map = liquidity_map
        self._scheduler = scheduler
        self._max_candles_per_context = max_candles_per_context

        self._state = LiquidityState()
        self._contexts: dict[str, LiquidityServiceContext] = {}
        self._locks: dict[str, asyncio.Lock] = {}

        self._stats = LiquidityServiceStats()
        self._running = False

        self._logger = get_logger(__name__, service_name="liquidity")

    async def start(self) -> None:
        if self._running:
            self._logger.warning("LiquidityService is already running")
            return

        await self._subscribe_events()
        self._running = True
        self._stats.started_at = self._utcnow()

        self._logger.info("LiquidityService started")

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("LiquidityService is not running")
            return

        await self._unsubscribe_events()
        self._running = False
        self._stats.stopped_at = self._utcnow()

        self._logger.info(
            "LiquidityService stopped",
            extra={
                "snapshots_built": self._stats.snapshots_built,
                "candle_events_processed": self._stats.candle_events_processed,
                "orderbook_events_processed": self._stats.orderbook_events_processed,
                "price_events_processed": self._stats.price_events_processed,
                "emitted_map_updates": self._stats.emitted_map_updates,
                "emitted_level_events": self._stats.emitted_level_events,
                "emitted_cluster_events": self._stats.emitted_cluster_events,
                "errors_count": self._stats.errors_count,
            },
        )

    async def rebuild_snapshot(
        self,
        symbol: str,
        timeframe: str,
        extra_levels: Sequence[LiquidityLevel] | None = None,
        extra_clusters: Sequence[StopCluster] | None = None,
        force: bool = False,
    ) -> LiquidityMapSnapshot | None:
        """
        Явна перебудова snapshot-а для symbol/timeframe.
        """
        key = self._make_key(symbol, timeframe)
        lock = self._get_lock(key)

        async with lock:
            context = self._contexts.get(key)
            if context is None:
                self._logger.debug(
                    "Skip rebuild: context not found",
                    extra={"symbol": symbol, "timeframe": timeframe},
                )
                return None

            if not force and not self._can_build_snapshot(context):
                self._logger.debug(
                    "Skip rebuild: insufficient context",
                    extra={"symbol": symbol, "timeframe": timeframe},
                )
                return None

            try:
                snapshot = self._liquidity_map.build_snapshot(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=context.candles,
                    current_price=float(context.current_price),
                    orderbook=context.orderbook,
                    extra_levels=extra_levels,
                    extra_clusters=extra_clusters,
                )

                await self._apply_snapshot(
                    symbol=symbol,
                    timeframe=timeframe,
                    snapshot=snapshot,
                )

                return snapshot

            except Exception as exc:
                self._handle_error(
                    "Failed to rebuild liquidity snapshot",
                    exc,
                    extra={"symbol": symbol, "timeframe": timeframe},
                )
                return None

    def get_state(self) -> LiquidityState:
        return self._state

    def get_stats(self) -> LiquidityServiceStats:
        return self._stats

    def get_last_snapshot(
        self,
        symbol: str,
        timeframe: str,
    ) -> LiquidityMapSnapshot | None:
        state = self._state.get(symbol, timeframe)
        if state is None:
            return None
        return state.last_snapshot

    def get_context(
        self,
        symbol: str,
        timeframe: str,
    ) -> LiquidityServiceContext | None:
        return self._contexts.get(self._make_key(symbol, timeframe))

    async def on_candle_closed(self, event: Any) -> None:
        await self._on_candle_closed(event)

    async def on_orderbook_updated(self, event: Any) -> None:
        await self._on_orderbook_updated(event)

    async def on_price_updated(self, event: Any) -> None:
        await self._on_price_updated(event)

    async def _subscribe_events(self) -> None:
        if self._event_bus is None:
            raise ValueError("event_bus is required for LiquidityService")

        await self._event_bus.subscribe(
            LiquidityTopics.MARKET_CANDLE_CLOSED,
            self._on_candle_closed,
        )
        await self._event_bus.subscribe(
            LiquidityTopics.MARKET_ORDERBOOK_UPDATED,
            self._on_orderbook_updated,
        )
        await self._event_bus.subscribe(
            LiquidityTopics.MARKET_PRICE_UPDATED,
            self._on_price_updated,
        )

    async def _unsubscribe_events(self) -> None:
        if self._event_bus is None:
            return

        unsubscribe = getattr(self._event_bus, "unsubscribe", None)
        if unsubscribe is None:
            return

        await unsubscribe(LiquidityTopics.MARKET_CANDLE_CLOSED, self._on_candle_closed)
        await unsubscribe(LiquidityTopics.MARKET_ORDERBOOK_UPDATED, self._on_orderbook_updated)
        await unsubscribe(LiquidityTopics.MARKET_PRICE_UPDATED, self._on_price_updated)

    async def _on_candle_closed(self, event: Any) -> None:
        self._stats.candle_events_processed += 1

        try:
            symbol = self._extract_required(event, "symbol")
            timeframe = self._extract_required(event, "timeframe")
            candle = self._extract_required(event, "candle")
            current_price = self._extract_optional(event, "current_price")
            if current_price is None:
                current_price = self._extract_price_from_candle(candle)

            key = self._make_key(symbol, timeframe)
            lock = self._get_lock(key)

            async with lock:
                context = self._get_or_create_context(symbol, timeframe)
                context.candles.append(candle)
                context.candles = context.candles[-self._max_candles_per_context :]
                context.current_price = float(current_price) if current_price is not None else context.current_price

                state = self._state.get_or_create(symbol, timeframe)
                state.processed_candles += 1
                state.last_candle_close_time = self._extract_event_timestamp(event) or self._utcnow()
                state.touch()

                if not self._can_build_snapshot(context):
                    return

                snapshot = self._liquidity_map.build_snapshot(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=context.candles,
                    current_price=float(context.current_price),
                    orderbook=context.orderbook,
                )

                await self._apply_snapshot(
                    symbol=symbol,
                    timeframe=timeframe,
                    snapshot=snapshot,
                )

        except Exception as exc:
            self._handle_error("Failed to process candle closed event", exc)

    async def _on_orderbook_updated(self, event: Any) -> None:
        self._stats.orderbook_events_processed += 1

        try:
            symbol = self._extract_required(event, "symbol")
            timeframe = self._extract_optional(event, "timeframe", default="default")
            bids = self._extract_optional(event, "bids", default=[])
            asks = self._extract_optional(event, "asks", default=[])
            current_price = self._extract_optional(event, "current_price")

            key = self._make_key(symbol, timeframe)
            lock = self._get_lock(key)

            async with lock:
                context = self._get_or_create_context(symbol, timeframe)
                context.orderbook = {
                    "bids": list(bids or []),
                    "asks": list(asks or []),
                }
                if current_price is not None:
                    context.current_price = float(current_price)

                state = self._state.get_or_create(symbol, timeframe)
                state.processed_orderbook_updates += 1
                state.touch()

                if not self._can_build_snapshot(context):
                    return

                if not self._should_rebuild_on_orderbook_update(context):
                    return

                snapshot = self._liquidity_map.build_snapshot(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=context.candles,
                    current_price=float(context.current_price),
                    orderbook=context.orderbook,
                )

                await self._apply_snapshot(
                    symbol=symbol,
                    timeframe=timeframe,
                    snapshot=snapshot,
                )

        except Exception as exc:
            self._handle_error("Failed to process orderbook update event", exc)

    async def _on_price_updated(self, event: Any) -> None:
        self._stats.price_events_processed += 1

        try:
            symbol = self._extract_required(event, "symbol")
            price = float(self._extract_required(event, "price"))

            timeframe = self._extract_optional(event, "timeframe")
            if timeframe is not None:
                keys = [self._make_key(symbol, timeframe)]
            else:
                keys = [key for key in self._contexts.keys() if key.startswith(f"{symbol}:")]

            for key in keys:
                lock = self._get_lock(key)

                async with lock:
                    context = self._contexts.get(key)
                    if context is None:
                        continue

                    context.current_price = price

        except Exception as exc:
            self._handle_error("Failed to process price update event", exc)

    async def _apply_snapshot(
        self,
        symbol: str,
        timeframe: str,
        snapshot: LiquidityMapSnapshot,
    ) -> None:
        key = self._make_key(symbol, timeframe)
        context = self._get_or_create_context(symbol, timeframe)
        previous_snapshot = context.last_snapshot

        context.last_snapshot = snapshot
        context.last_rebuild_at = snapshot.timestamp

        state = self._state.get_or_create(symbol, timeframe)
        self._update_state_from_snapshot(state, snapshot)

        self._stats.snapshots_built += 1

        if self._config.publish_events:
            await self._emit_snapshot_events(
                symbol=symbol,
                timeframe=timeframe,
                snapshot=snapshot,
                previous_snapshot=previous_snapshot,
            )

    def _update_state_from_snapshot(
        self,
        state: LiquidityTimeframeState,
        snapshot: LiquidityMapSnapshot,
    ) -> None:
        state.active_levels = list(snapshot.active_levels)
        state.equal_levels = list(snapshot.equal_levels)
        state.stop_clusters = list(snapshot.stop_clusters)
        state.last_snapshot = snapshot
        state.last_update_at = snapshot.timestamp

    async def _emit_snapshot_events(
        self,
        symbol: str,
        timeframe: str,
        snapshot: LiquidityMapSnapshot,
        previous_snapshot: LiquidityMapSnapshot | None,
    ) -> None:
        await self._emit_map_updated(snapshot)
        await self._emit_level_events(snapshot, previous_snapshot)
        await self._emit_cluster_events(snapshot, previous_snapshot)

    async def _emit_map_updated(self, snapshot: LiquidityMapSnapshot) -> None:
        await self._safe_emit(
            LiquidityTopics.LIQUIDITY_MAP_UPDATED,
            snapshot,
        )
        self._stats.emitted_map_updates += 1

    async def _emit_level_events(
        self,
        snapshot: LiquidityMapSnapshot,
        previous_snapshot: LiquidityMapSnapshot | None,
    ) -> None:
        previous_levels = self._index_levels(previous_snapshot.active_levels if previous_snapshot else [])
        current_levels = self._index_levels(snapshot.active_levels)

        for level_key, level in current_levels.items():
            if level_key not in previous_levels:
                await self._safe_emit(
                    LiquidityTopics.LIQUIDITY_LEVEL_DETECTED,
                    level,
                )
                self._stats.emitted_level_events += 1
                continue

            prev_level = previous_levels[level_key]
            if prev_level.sweep_status != level.sweep_status and level.sweep_status in {
                SweepStatus.PARTIALLY_SWEPT,
                SweepStatus.SWEPT,
            }:
                await self._safe_emit(
                    LiquidityTopics.LIQUIDITY_LEVEL_SWEPT,
                    level,
                )
                self._stats.emitted_level_events += 1

    async def _emit_cluster_events(
        self,
        snapshot: LiquidityMapSnapshot,
        previous_snapshot: LiquidityMapSnapshot | None,
    ) -> None:
        previous_clusters = self._index_clusters(previous_snapshot.stop_clusters if previous_snapshot else [])
        current_clusters = self._index_clusters(snapshot.stop_clusters)

        for cluster_key, cluster in current_clusters.items():
            if cluster_key not in previous_clusters:
                await self._safe_emit(
                    LiquidityTopics.LIQUIDITY_STOP_CLUSTER_DETECTED,
                    cluster,
                )
                self._stats.emitted_cluster_events += 1

    async def _safe_emit(
        self,
        topic: str,
        payload: Any,
    ) -> None:
        if self._event_bus is None:
            return

        await self._event_bus.emit(topic, payload)

    def _index_levels(
        self,
        levels: Sequence[LiquidityLevel],
    ) -> dict[str, LiquidityLevel]:
        result: dict[str, LiquidityLevel] = {}

        for level in levels:
            key = (
                f"{level.symbol}|{level.timeframe}|{level.level_type.value}|"
                f"{level.side.value}|{round(level.price, 8)}"
            )
            result[key] = level

        return result

    def _index_clusters(
        self,
        clusters: Sequence[StopCluster],
    ) -> dict[str, StopCluster]:
        result: dict[str, StopCluster] = {}

        for cluster in clusters:
            key = (
                f"{cluster.symbol}|{cluster.timeframe}|{cluster.side.value}|"
                f"{round(cluster.low_price, 8)}|{round(cluster.high_price, 8)}"
            )
            result[key] = cluster

        return result

    def _can_build_snapshot(self, context: LiquidityServiceContext) -> bool:
        if context.current_price is None or context.current_price <= 0:
            return False

        min_candles = self._config.pivot_lookback + self._config.pivot_lookforward + 3
        return len(context.candles) >= min_candles

    def _should_rebuild_on_orderbook_update(
        self,
        context: LiquidityServiceContext,
    ) -> bool:
        """
        Не перебудовуємо snapshot на кожен orderbook tick безконтрольно.
        """
        if context.last_rebuild_at is None:
            return True

        delta = self._utcnow() - context.last_rebuild_at
        return delta.total_seconds() >= 2.0

    def _get_or_create_context(
        self,
        symbol: str,
        timeframe: str,
    ) -> LiquidityServiceContext:
        key = self._make_key(symbol, timeframe)
        if key not in self._contexts:
            self._contexts[key] = LiquidityServiceContext(
                symbol=symbol,
                timeframe=timeframe,
            )
        return self._contexts[key]

    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _make_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}:{timeframe}"

    def _extract_required(self, event: Any, field: str) -> Any:
        value = self._extract_optional(event, field)
        if value is None:
            raise ValueError(f"Event must contain '{field}'")
        return value

    def _extract_optional(
        self,
        event: Any,
        field: str,
        default: Any = None,
    ) -> Any:
        if isinstance(event, dict):
            return event.get(field, default)
        return getattr(event, field, default)

    def _extract_event_timestamp(self, event: Any) -> datetime | None:
        for field in ("timestamp", "time", "event_time", "close_time"):
            value = self._extract_optional(event, field)
            if value is None:
                continue

            if isinstance(value, datetime):
                if value.tzinfo is not None:
                    return value.astimezone(timezone.utc).replace(tzinfo=None)
                return value

            if isinstance(value, (int, float)):
                try:
                    return datetime.utcfromtimestamp(value / 1000 if value > 1e12 else value)
                except Exception:
                    return None

            if isinstance(value, str):
                try:
                    dt = datetime.fromisoformat(value)
                    if dt.tzinfo is not None:
                        return dt.astimezone(timezone.utc).replace(tzinfo=None)
                    return dt
                except Exception:
                    return None

        return None

    def _extract_price_from_candle(self, candle: Any) -> float | None:
        value = None
        if isinstance(candle, dict):
            value = candle.get("close")
        else:
            value = getattr(candle, "close", None)

        if value is None:
            return None

        return float(value)

    def _handle_error(
        self,
        message: str,
        exc: Exception,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._stats.errors_count += 1
        self._stats.last_error = str(exc)

        payload = {
            "error": str(exc),
        }
        if extra:
            payload.update(extra)

        self._logger.exception(message, extra=payload)

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None)