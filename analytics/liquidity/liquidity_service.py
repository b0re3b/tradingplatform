from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import LiquidityConfig
from .enums import SweepStatus
from .liquidity_map import LiquidityMap
from .models import LiquidityLevel, LiquidityMapSnapshot, StopCluster
from .state import LiquidityState
from .utils import get_candle_close, get_first_value, safe_float


class LiquidityTopics:
    """
    Event topics для analytics/liquidity integration layer.

    Market topics приходять із data layer.
    Analytics topics публікуються для strategy/dashboard/storage/bots.
    """

    MARKET_CANDLE_CLOSED = "market.candle.closed"
    MARKET_ORDERBOOK_UPDATED = "market.orderbook.updated"
    MARKET_PRICE_UPDATED = "market.price.updated"

    ANALYTICS_LIQUIDITY_MAP_UPDATED = "analytics.liquidity.map.updated"
    ANALYTICS_LIQUIDITY_LEVEL_DETECTED = "analytics.liquidity.level.detected"
    ANALYTICS_LIQUIDITY_LEVEL_SWEPT = "analytics.liquidity.level.swept"
    ANALYTICS_LIQUIDITY_STOP_CLUSTER_DETECTED = (
        "analytics.liquidity.stop_cluster.detected"
    )
    ANALYTICS_LIQUIDITY_SIGNAL_UPDATED = "analytics.liquidity.signal.updated"
    ANALYTICS_LIQUIDITY_STATE_METRICS = "analytics.liquidity.state.metrics"
    ANALYTICS_LIQUIDITY_HEALTHCHECK = "analytics.liquidity.healthcheck"


@dataclass(slots=True)
class LiquidityServiceStats:
    """
    Runtime stats для LiquidityService.
    """

    started_at: datetime | None = None
    stopped_at: datetime | None = None

    snapshots_built: int = 0

    candle_events_processed: int = 0
    orderbook_events_processed: int = 0
    price_events_processed: int = 0

    emitted_map_updates: int = 0
    emitted_level_events: int = 0
    emitted_cluster_events: int = 0
    emitted_signal_events: int = 0
    emitted_metrics_events: int = 0
    emitted_healthcheck_events: int = 0

    cleanup_runs: int = 0
    removed_empty_contexts: int = 0
    removed_empty_states: int = 0
    removed_inactive_levels: int = 0

    errors_count: int = 0
    last_error: str | None = None
    last_error_at: datetime | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "snapshots_built": self.snapshots_built,
            "candle_events_processed": self.candle_events_processed,
            "orderbook_events_processed": self.orderbook_events_processed,
            "price_events_processed": self.price_events_processed,
            "emitted_map_updates": self.emitted_map_updates,
            "emitted_level_events": self.emitted_level_events,
            "emitted_cluster_events": self.emitted_cluster_events,
            "emitted_signal_events": self.emitted_signal_events,
            "emitted_metrics_events": self.emitted_metrics_events,
            "emitted_healthcheck_events": self.emitted_healthcheck_events,
            "cleanup_runs": self.cleanup_runs,
            "removed_empty_contexts": self.removed_empty_contexts,
            "removed_empty_states": self.removed_empty_states,
            "removed_inactive_levels": self.removed_inactive_levels,
            "errors_count": self.errors_count,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at.isoformat()
            if self.last_error_at
            else None,
        }


@dataclass(slots=True)
class LiquidityServiceContext:
    """
    Runtime market context для конкретного exchange + market_type + symbol + timeframe.

    Важливо:
    - context не може бути тільки symbol/timeframe, бо BTCUSDT з Binance,
      Bybit, OKX і MEXC не є одним і тим самим market stream;
    - orderbook updates зазвичай не мають timeframe, тому вони застосовуються
      тільки до context-ів того самого exchange + market_type + symbol.
    """

    exchange: str
    market_type: str
    symbol: str
    timeframe: str

    candles: list[Any] = field(default_factory=list)
    orderbook: dict[str, list[Any]] = field(
        default_factory=lambda: {"bids": [], "asks": []}
    )
    current_price: float | None = None

    last_snapshot: LiquidityMapSnapshot | None = None
    last_rebuild_at: datetime | None = None
    last_update_at: datetime | None = None

    def __post_init__(self) -> None:
        self.exchange = self._normalize_scope_value(self.exchange)
        self.market_type = self._normalize_scope_value(self.market_type)
        self.symbol = str(self.symbol or "").strip().upper()
        self.timeframe = str(self.timeframe or "").strip()

    @property
    def key(self) -> str:
        return LiquidityService.make_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    def touch(self, ts: datetime | None = None) -> None:
        self.last_update_at = ts or datetime.now(timezone.utc)

    def can_build(self, min_candles: int) -> bool:
        return (
            self.current_price is not None
            and self.current_price > 0
            and len(self.candles) >= min_candles
        )

    @staticmethod
    def _normalize_scope_value(value: Any) -> str:
        normalized = str(value or "").strip()
        return normalized if normalized else "unknown"


class LiquidityService:
    """
    Production-ready orchestration layer для analytics/liquidity.

    Відповідальність:
    - приймає EventBus / Scheduler / Config через dependency injection;
    - підписується на market.* події через register();
    - накопичує market context per exchange/market_type/symbol/timeframe;
    - викликає LiquidityMap;
    - оновлює LiquidityState;
    - публікує analytics.liquidity.* події;
    - запускає cleanup / metrics / healthcheck через Scheduler.

    Важливо:
    - detectors і LiquidityMap залишаються чистими domain-компонентами;
    - вся інтеграція з core.EventBus і core.Scheduler живе тут;
    - цей service не читає біржі напряму і не звертається напряму до exchange adapters.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None,
        config: LiquidityConfig,
        liquidity_map: LiquidityMap,
    ) -> None:
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._config = config
        self._config.validate()

        self._liquidity_map = liquidity_map

        self._state = LiquidityState()
        self._contexts: dict[str, LiquidityServiceContext] = {}
        self._locks: dict[str, asyncio.Lock] = {}

        self._subscriptions: list[Subscription] = []
        self._scheduler_job_ids: list[str] = []

        self._stats = LiquidityServiceStats()

        self._registered = False
        self._running = False

        self._logger = get_logger(
            __name__,
            service_name="analytics_liquidity",
            event_type="liquidity_service",
        )

    # ------------------------------------------------------------------
    # Lifecycle / registration
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Реєструє EventBus subscriptions і Scheduler jobs.

        Викликати один раз під час bootstrap перед start().
        """
        if self._registered:
            self._logger.warning("LiquidityService already registered")
            return

        self._register_event_subscriptions()
        self._register_scheduler_jobs()

        self._registered = True

        self._logger.info(
            "LiquidityService registered",
            extra={
                "subscriptions": len(self._subscriptions),
                "scheduler_jobs": len(self._scheduler_job_ids),
            },
        )

    async def start(self) -> None:
        """
        Запускає runtime-state сервісу.

        EventBus має бути вже started зовнішнім bootstrap/main.
        Scheduler також має запускатися централізовано, не тут.
        """
        if self._running:
            self._logger.warning("LiquidityService already started")
            return

        if not self._registered:
            self.register()

        self._running = True
        self._stats.started_at = self._utcnow()
        self._stats.stopped_at = None

        self._logger.info("LiquidityService started")

    async def stop(self) -> None:
        """
        Зупиняє сервіс і відписує EventBus subscriptions.

        Scheduler jobs прибираються з Scheduler, якщо scheduler передано.
        """
        if not self._running and not self._registered:
            self._logger.warning("LiquidityService already stopped")
            return

        self._unregister_event_subscriptions()
        self._unregister_scheduler_jobs()

        self._running = False
        self._registered = False
        self._stats.stopped_at = self._utcnow()

        self._logger.info(
            "LiquidityService stopped",
            extra=self._stats.to_payload(),
        )

    def _register_event_subscriptions(self) -> None:
        self._subscriptions.append(
            self._event_bus.subscribe(
                LiquidityTopics.MARKET_CANDLE_CLOSED,
                self._on_candle_closed,
                name="analytics_liquidity.on_candle_closed",
            )
        )
        self._subscriptions.append(
            self._event_bus.subscribe(
                LiquidityTopics.MARKET_ORDERBOOK_UPDATED,
                self._on_orderbook_updated,
                name="analytics_liquidity.on_orderbook_updated",
            )
        )
        self._subscriptions.append(
            self._event_bus.subscribe(
                LiquidityTopics.MARKET_PRICE_UPDATED,
                self._on_price_updated,
                name="analytics_liquidity.on_price_updated",
            )
        )

    def _unregister_event_subscriptions(self) -> None:
        for subscription in self._subscriptions:
            self._event_bus.unsubscribe(subscription)

        self._subscriptions.clear()

    def _register_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            return

        if self._config.cleanup_enabled:
            self._scheduler_job_ids.append(
                self._scheduler.add_interval_job(
                    name="analytics_liquidity.cleanup",
                    func=self._cleanup,
                    interval=self._config.cleanup_interval_seconds,
                    run_immediately=False,
                    max_retries=self._config.scheduler_job_max_retries,
                    retry_delay=self._config.scheduler_job_retry_delay_seconds,
                    timeout=self._config.scheduler_job_timeout_seconds,
                    allow_overlap=False,
                    enabled=True,
                )
            )

        if self._config.emit_state_metrics:
            self._scheduler_job_ids.append(
                self._scheduler.add_interval_job(
                    name="analytics_liquidity.emit_state_metrics",
                    func=self._emit_state_metrics,
                    interval=self._config.state_metrics_interval_seconds,
                    run_immediately=False,
                    max_retries=self._config.scheduler_job_max_retries,
                    retry_delay=self._config.scheduler_job_retry_delay_seconds,
                    timeout=self._config.scheduler_job_timeout_seconds,
                    allow_overlap=False,
                    enabled=True,
                )
            )

        self._scheduler_job_ids.append(
            self._scheduler.add_interval_job(
                name="analytics_liquidity.healthcheck",
                func=self._emit_healthcheck,
                interval=self._config.healthcheck_interval_seconds,
                run_immediately=False,
                max_retries=self._config.scheduler_job_max_retries,
                retry_delay=self._config.scheduler_job_retry_delay_seconds,
                timeout=self._config.scheduler_job_timeout_seconds,
                allow_overlap=False,
                enabled=True,
            )
        )

    def _unregister_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            self._scheduler_job_ids.clear()
            return

        for job_id in self._scheduler_job_ids:
            try:
                self._scheduler.remove_job(job_id)
            except KeyError:
                self._logger.warning(
                    "Scheduler job already removed",
                    extra={"job_id": job_id},
                )

        self._scheduler_job_ids.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rebuild_snapshot(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
        extra_levels: Sequence[LiquidityLevel] | None = None,
        extra_clusters: Sequence[StopCluster] | None = None,
        force: bool = False,
    ) -> LiquidityMapSnapshot | None:
        """
        Явна перебудова snapshot-а для exchange/market_type/symbol/timeframe.
        """
        exchange = self._normalize_exchange(exchange)
        market_type = self._normalize_market_type_value(market_type)
        symbol = self._normalize_symbol(symbol)
        timeframe = self._normalize_timeframe(timeframe)

        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        lock = self._get_lock(key)

        async with lock:
            context = self._contexts.get(key)
            if context is None:
                self._logger.debug(
                    "Skip rebuild: context not found",
                    extra={
                        "exchange": exchange,
                        "market_type": market_type,
                        "symbol": symbol,
                        "timeframe": timeframe,
                    },
                )
                return None

            return await self._rebuild_context_snapshot_locked(
                context=context,
                extra_levels=extra_levels,
                extra_clusters=extra_clusters,
                force=force,
            )

    def get_state(self) -> LiquidityState:
        return self._state

    def get_stats(self) -> LiquidityServiceStats:
        return self._stats

    def get_last_snapshot(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
    ) -> LiquidityMapSnapshot | None:
        state = self._state.get(
            exchange=self._normalize_exchange(exchange),
            market_type=self._normalize_market_type_value(market_type),
            symbol=self._normalize_symbol(symbol),
            timeframe=self._normalize_timeframe(timeframe),
        )

        if state is not None:
            return state.last_snapshot

        context = self.get_context(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        return context.last_snapshot if context else None

    def get_context(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
    ) -> LiquidityServiceContext | None:
        return self._contexts.get(
            self.make_key(
                exchange=self._normalize_exchange(exchange),
                market_type=self._normalize_market_type_value(market_type),
                symbol=self._normalize_symbol(symbol),
                timeframe=self._normalize_timeframe(timeframe),
            )
        )

    async def on_candle_closed(self, event: Event | dict[str, Any]) -> None:
        await self._on_candle_closed(event)

    async def on_orderbook_updated(self, event: Event | dict[str, Any]) -> None:
        await self._on_orderbook_updated(event)

    async def on_price_updated(self, event: Event | dict[str, Any]) -> None:
        await self._on_price_updated(event)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_candle_closed(self, event: Event | dict[str, Any]) -> None:
        if not self._running or not self._config.enabled:
            return

        self._stats.candle_events_processed += 1

        try:
            payload = self._event_payload(event)

            exchange = self._normalize_exchange(
                self._extract_required_str(payload, "exchange")
            )
            market_type = self._normalize_market_type_value(
                self._extract_market_type(payload)
            )
            symbol = self._normalize_symbol(
                self._extract_required_str(payload, "symbol")
            )
            timeframe = self._normalize_timeframe(
                self._extract_required_str(payload, "timeframe")
            )

            # CandlesCache емiтить serialized candle напряму.
            # Деякі інтеграційні шари можуть емiтити {"candle": {...}}.
            candle = payload.get("candle")
            if candle is None:
                candle = payload

            current_price = self._extract_optional(payload, "current_price")
            if current_price is None:
                current_price = self._extract_price_from_candle(candle)

            event_ts = self._extract_event_timestamp(payload) or self._utcnow()

            key = self.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
            lock = self._get_lock(key)

            async with lock:
                context = self._get_or_create_context(
                    exchange=exchange,
                    market_type=market_type,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                context.candles.append(candle)
                context.candles = context.candles[
                    -self._config.max_candles_per_context :
                ]

                if current_price is not None:
                    price = safe_float(current_price)
                    if price > 0:
                        context.current_price = price

                context.touch(event_ts)

                state = self._state.get_or_create(
                    exchange=context.exchange,
                    market_type=context.market_type,
                    symbol=context.symbol,
                    timeframe=context.timeframe,
                )
                state.record_candle_processed(
                    close_time=event_ts,
                    ts=event_ts,
                )

                await self._rebuild_context_snapshot_locked(
                    context=context,
                    force=False,
                )

        except Exception as exc:
            self._handle_error(
                "Failed to process candle closed event",
                exc,
                extra=self._error_extra(event),
            )

    async def _on_orderbook_updated(self, event: Event | dict[str, Any]) -> None:
        if not self._running or not self._config.enabled:
            return

        self._stats.orderbook_events_processed += 1

        try:
            payload = self._event_payload(event)

            exchange = self._normalize_exchange(
                self._extract_required_str(payload, "exchange")
            )
            market_type = self._normalize_market_type_value(
                self._extract_market_type(payload)
            )
            symbol = self._normalize_symbol(
                self._extract_required_str(payload, "symbol")
            )

            bids = self._extract_optional(payload, "bids", default=[])
            asks = self._extract_optional(payload, "asks", default=[])
            current_price = self._extract_optional(payload, "current_price")

            if current_price is None:
                current_price = self._extract_mid_price(payload)

            event_ts = self._extract_event_timestamp(payload) or self._utcnow()

            # Orderbook не має природного timeframe, тому оновлюємо тільки ті
            # liquidity contexts, які вже існують для цього exchange/market/symbol.
            matching_keys = self._find_context_keys_for_market(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
            )

            for key in matching_keys:
                lock = self._get_lock(key)

                async with lock:
                    context = self._contexts.get(key)
                    if context is None:
                        continue

                    context.orderbook = {
                        "bids": list(bids or []),
                        "asks": list(asks or []),
                    }

                    if current_price is not None:
                        price = safe_float(current_price)
                        if price > 0:
                            context.current_price = price

                    context.touch(event_ts)

                    state = self._state.get_or_create(
                        exchange=context.exchange,
                        market_type=context.market_type,
                        symbol=context.symbol,
                        timeframe=context.timeframe,
                    )
                    state.record_orderbook_processed(ts=event_ts)

                    if not self._config.rebuild_on_orderbook_updates:
                        continue

                    if not self._should_rebuild_context(context):
                        continue

                    await self._rebuild_context_snapshot_locked(
                        context=context,
                        force=False,
                    )

        except Exception as exc:
            self._handle_error(
                "Failed to process orderbook update event",
                exc,
                extra=self._error_extra(event),
            )

    async def _on_price_updated(self, event: Event | dict[str, Any]) -> None:
        if not self._running or not self._config.enabled:
            return

        self._stats.price_events_processed += 1

        try:
            payload = self._event_payload(event)

            exchange = self._normalize_exchange(
                self._extract_required_str(payload, "exchange")
            )
            market_type = self._normalize_market_type_value(
                self._extract_market_type(payload)
            )
            symbol = self._normalize_symbol(
                self._extract_required_str(payload, "symbol")
            )

            price = safe_float(self._extract_required(payload, "price"))
            if price <= 0:
                raise ValueError("price must be > 0")

            timeframe = self._extract_optional(payload, "timeframe")
            event_ts = self._extract_event_timestamp(payload) or self._utcnow()

            if timeframe is not None:
                keys = [
                    self.make_key(
                        exchange=exchange,
                        market_type=market_type,
                        symbol=symbol,
                        timeframe=self._normalize_timeframe(timeframe),
                    )
                ]
            else:
                keys = self._find_context_keys_for_market(
                    exchange=exchange,
                    market_type=market_type,
                    symbol=symbol,
                )

            for key in keys:
                lock = self._get_lock(key)

                async with lock:
                    context = self._contexts.get(key)
                    if context is None:
                        continue

                    context.current_price = price
                    context.touch(event_ts)

                    state = self._state.get_or_create(
                        exchange=context.exchange,
                        market_type=context.market_type,
                        symbol=context.symbol,
                        timeframe=context.timeframe,
                    )
                    state.record_price_processed(ts=event_ts)

                    if not self._config.rebuild_on_price_updates:
                        continue

                    if not self._should_rebuild_context(context):
                        continue

                    await self._rebuild_context_snapshot_locked(
                        context=context,
                        force=False,
                    )

        except Exception as exc:
            self._handle_error(
                "Failed to process price update event",
                exc,
                extra=self._error_extra(event),
            )

    # ------------------------------------------------------------------
    # Snapshot application
    # ------------------------------------------------------------------

    async def _rebuild_context_snapshot_locked(
        self,
        context: LiquidityServiceContext,
        extra_levels: Sequence[LiquidityLevel] | None = None,
        extra_clusters: Sequence[StopCluster] | None = None,
        force: bool = False,
    ) -> LiquidityMapSnapshot | None:
        """
        Rebuild snapshot.

        Має викликатися тільки всередині lock для відповідного context.
        """
        if not force and not self._can_build_snapshot(context):
            return None

        if not force and not self._should_rebuild_context(context):
            return None

        if context.current_price is None or context.current_price <= 0:
            return None

        try:
            snapshot = self._liquidity_map.build_snapshot(
                exchange=context.exchange,
                market_type=context.market_type,
                symbol=context.symbol,
                timeframe=context.timeframe,
                candles=context.candles,
                current_price=context.current_price,
                orderbook=context.orderbook,
                extra_levels=extra_levels,
                extra_clusters=extra_clusters,
            )

            await self._apply_snapshot(
                context=context,
                snapshot=snapshot,
            )

            return snapshot

        except Exception as exc:
            self._handle_error(
                "Failed to rebuild liquidity snapshot",
                exc,
                extra={
                    "exchange": context.exchange,
                    "market_type": context.market_type,
                    "symbol": context.symbol,
                    "timeframe": context.timeframe,
                },
            )
            return None

    async def _apply_snapshot(
        self,
        context: LiquidityServiceContext,
        snapshot: LiquidityMapSnapshot,
    ) -> None:
        previous_snapshot = context.last_snapshot

        context.last_snapshot = snapshot
        context.last_rebuild_at = snapshot.timestamp
        context.touch(snapshot.timestamp)

        self._state.apply_snapshot(snapshot)

        self._stats.snapshots_built += 1

        if self._config.publish_events:
            await self._emit_snapshot_events(
                context=context,
                snapshot=snapshot,
                previous_snapshot=previous_snapshot,
            )

    async def _emit_snapshot_events(
        self,
        *,
        context: LiquidityServiceContext,
        snapshot: LiquidityMapSnapshot,
        previous_snapshot: LiquidityMapSnapshot | None,
    ) -> None:
        if self._config.emit_map_updates:
            await self._emit_map_updated(context, snapshot)

        if self._config.emit_level_events or self._config.emit_sweep_events:
            await self._emit_level_events(context, snapshot, previous_snapshot)

        if self._config.emit_cluster_events:
            await self._emit_cluster_events(context, snapshot, previous_snapshot)

        if self._config.emit_signal_events and snapshot.signal is not None:
            await self._emit_signal_updated(context, snapshot)

    async def _emit_map_updated(
        self,
        context: LiquidityServiceContext,
        snapshot: LiquidityMapSnapshot,
    ) -> None:
        await self._safe_emit(
            topic=LiquidityTopics.ANALYTICS_LIQUIDITY_MAP_UPDATED,
            payload=self._scoped_snapshot_payload(context, snapshot),
            priority=EventPriority.NORMAL,
        )
        self._stats.emitted_map_updates += 1

    async def _emit_signal_updated(
        self,
        context: LiquidityServiceContext,
        snapshot: LiquidityMapSnapshot,
    ) -> None:
        if snapshot.signal is None:
            return

        payload = snapshot.signal.to_event_payload()
        payload.update(
            {
                "exchange": context.exchange,
                "market_type": context.market_type,
                "symbol": context.symbol,
                "timeframe": context.timeframe,
                "context_key": context.key,
            }
        )

        await self._safe_emit(
            topic=LiquidityTopics.ANALYTICS_LIQUIDITY_SIGNAL_UPDATED,
            payload=payload,
            priority=EventPriority.NORMAL,
        )
        self._stats.emitted_signal_events += 1

    async def _emit_level_events(
        self,
        context: LiquidityServiceContext,
        snapshot: LiquidityMapSnapshot,
        previous_snapshot: LiquidityMapSnapshot | None,
    ) -> None:
        previous_levels = self._index_levels(
            previous_snapshot.active_levels if previous_snapshot else []
        )
        current_levels = self._index_levels(snapshot.active_levels)

        for level_key, level in current_levels.items():
            previous = previous_levels.get(level_key)

            if previous is None and self._config.emit_level_events:
                payload = level.to_event_payload()
                payload.update(
                    {
                        "exchange": context.exchange,
                        "market_type": context.market_type,
                        "symbol": context.symbol,
                        "timeframe": context.timeframe,
                        "context_key": context.key,
                    }
                )

                await self._safe_emit(
                    topic=LiquidityTopics.ANALYTICS_LIQUIDITY_LEVEL_DETECTED,
                    payload=payload,
                    priority=EventPriority.NORMAL,
                )
                self._stats.emitted_level_events += 1
                continue

            if previous is None:
                continue

            sweep_changed = previous.sweep_status != level.sweep_status
            swept_now = level.sweep_status in {
                SweepStatus.PARTIALLY_SWEPT,
                SweepStatus.SWEPT,
            }

            if (
                self._config.emit_sweep_events
                and sweep_changed
                and swept_now
            ):
                payload = level.to_event_payload()
                payload.update(
                    {
                        "exchange": context.exchange,
                        "market_type": context.market_type,
                        "symbol": context.symbol,
                        "timeframe": context.timeframe,
                        "context_key": context.key,
                    }
                )

                await self._safe_emit(
                    topic=LiquidityTopics.ANALYTICS_LIQUIDITY_LEVEL_SWEPT,
                    payload=payload,
                    priority=EventPriority.HIGH,
                )
                self._stats.emitted_level_events += 1

    async def _emit_cluster_events(
        self,
        context: LiquidityServiceContext,
        snapshot: LiquidityMapSnapshot,
        previous_snapshot: LiquidityMapSnapshot | None,
    ) -> None:
        previous_clusters = self._index_clusters(
            previous_snapshot.stop_clusters if previous_snapshot else []
        )
        current_clusters = self._index_clusters(snapshot.stop_clusters)

        for cluster_key, cluster in current_clusters.items():
            if cluster_key in previous_clusters:
                continue

            payload = cluster.to_event_payload()
            payload.update(
                {
                    "exchange": context.exchange,
                    "market_type": context.market_type,
                    "symbol": context.symbol,
                    "timeframe": context.timeframe,
                    "context_key": context.key,
                }
            )

            await self._safe_emit(
                topic=LiquidityTopics.ANALYTICS_LIQUIDITY_STOP_CLUSTER_DETECTED,
                payload=payload,
                priority=EventPriority.NORMAL,
            )
            self._stats.emitted_cluster_events += 1

    async def _safe_emit(
        self,
        *,
        topic: str,
        payload: dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        try:
            await self._event_bus.emit(
                topic,
                payload,
                priority=priority,
                source="analytics_liquidity",
            )
        except Exception as exc:
            self._handle_error(
                "Failed to emit liquidity event",
                exc,
                extra={"topic": topic},
            )

    # ------------------------------------------------------------------
    # Scheduler jobs
    # ------------------------------------------------------------------

    async def _cleanup(self) -> None:
        """
        Periodic cleanup. Запускається тільки через Scheduler.
        """
        self._stats.cleanup_runs += 1

        removed_inactive = self._state.remove_inactive_levels()
        self._state.prune_all(
            max_active_levels=self._config.max_active_levels,
            max_active_clusters=self._config.max_active_clusters,
        )

        removed_states = self._state.remove_empty_states()
        removed_contexts = self._remove_excess_or_empty_contexts()

        self._stats.removed_inactive_levels += removed_inactive
        self._stats.removed_empty_states += removed_states
        self._stats.removed_empty_contexts += removed_contexts

        self._logger.debug(
            "LiquidityService cleanup completed",
            extra={
                "removed_inactive_levels": removed_inactive,
                "removed_empty_states": removed_states,
                "removed_contexts": removed_contexts,
                "contexts": len(self._contexts),
                "states": self._state.count(),
            },
        )

    async def _emit_state_metrics(self) -> None:
        if not self._running or not self._config.publish_events:
            return

        payload = {
            "service": "analytics_liquidity",
            "timestamp": self._utcnow().isoformat(),
            "stats": self._stats.to_payload(),
            "state": self._state.to_metrics_payload(),
            "contexts_count": len(self._contexts),
            "context_keys": list(self._contexts.keys()),
        }

        await self._safe_emit(
            topic=LiquidityTopics.ANALYTICS_LIQUIDITY_STATE_METRICS,
            payload=payload,
            priority=EventPriority.LOW,
        )
        self._stats.emitted_metrics_events += 1

    async def _emit_healthcheck(self) -> None:
        if not self._running or not self._config.publish_events:
            return

        payload = {
            "service": "analytics_liquidity",
            "timestamp": self._utcnow().isoformat(),
            "running": self._running,
            "registered": self._registered,
            "contexts_count": len(self._contexts),
            "states_count": self._state.count(),
            "subscriptions": len(self._subscriptions),
            "scheduler_jobs": len(self._scheduler_job_ids),
            "errors_count": self._stats.errors_count,
            "last_error": self._stats.last_error,
            "context_keys": list(self._contexts.keys()),
        }

        await self._safe_emit(
            topic=LiquidityTopics.ANALYTICS_LIQUIDITY_HEALTHCHECK,
            payload=payload,
            priority=EventPriority.LOW,
        )
        self._stats.emitted_healthcheck_events += 1

    def _remove_excess_or_empty_contexts(self) -> int:
        removed = 0

        empty_keys = [
            key
            for key, context in self._contexts.items()
            if not context.candles and context.last_snapshot is None
        ]

        for key in empty_keys:
            self._contexts.pop(key, None)
            self._locks.pop(key, None)
            removed += 1

        if len(self._contexts) <= self._config.max_contexts:
            return removed

        sorted_items = sorted(
            self._contexts.items(),
            key=lambda item: item[1].last_update_at
            or datetime.min.replace(tzinfo=timezone.utc),
        )

        excess = len(self._contexts) - self._config.max_contexts
        for key, _ in sorted_items[:excess]:
            self._contexts.pop(key, None)
            self._locks.pop(key, None)
            removed += 1

        return removed

    # ------------------------------------------------------------------
    # Indexing / rebuild guards
    # ------------------------------------------------------------------

    def _index_levels(
        self,
        levels: Sequence[LiquidityLevel],
    ) -> dict[str, LiquidityLevel]:
        return {level.key: level for level in levels}

    def _index_clusters(
        self,
        clusters: Sequence[StopCluster],
    ) -> dict[str, StopCluster]:
        return {cluster.key: cluster for cluster in clusters}

    def _can_build_snapshot(
        self,
        context: LiquidityServiceContext,
    ) -> bool:
        return context.can_build(self._config.min_candles_for_snapshot)

    def _should_rebuild_context(
        self,
        context: LiquidityServiceContext,
    ) -> bool:
        if context.last_rebuild_at is None:
            return True

        delta = self._utcnow() - context.last_rebuild_at
        return delta.total_seconds() >= self._config.snapshot_rebuild_min_interval_seconds

    # ------------------------------------------------------------------
    # Context / locks
    # ------------------------------------------------------------------

    def _get_or_create_context(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
    ) -> LiquidityServiceContext:
        exchange = self._normalize_exchange(exchange)
        market_type = self._normalize_market_type_value(market_type)
        symbol = self._normalize_symbol(symbol)
        timeframe = self._normalize_timeframe(timeframe)

        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        if key not in self._contexts:
            self._contexts[key] = LiquidityServiceContext(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )

        return self._contexts[key]

    def _find_context_keys_for_market(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
    ) -> list[str]:
        prefix = self.make_market_prefix(
            exchange=self._normalize_exchange(exchange),
            market_type=self._normalize_market_type_value(market_type),
            symbol=self._normalize_symbol(symbol),
        )
        return [
            key
            for key in self._contexts.keys()
            if key.startswith(prefix)
        ]

    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        return self._locks[key]

    @staticmethod
    def make_key(
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
    ) -> str:
        return (
            f"{LiquidityService._normalize_key_part(exchange)}:"
            f"{LiquidityService._normalize_key_part(market_type)}:"
            f"{LiquidityService._normalize_key_part(symbol)}:"
            f"{LiquidityService._normalize_key_part(timeframe)}"
        )

    @staticmethod
    def make_market_prefix(
        *,
        exchange: str,
        market_type: str,
        symbol: str,
    ) -> str:
        return (
            f"{LiquidityService._normalize_key_part(exchange)}:"
            f"{LiquidityService._normalize_key_part(market_type)}:"
            f"{LiquidityService._normalize_key_part(symbol)}:"
        )

    @staticmethod
    def _normalize_key_part(value: Any) -> str:
        return str(value).strip().lower()

    # Backward-compatible private alias, якщо десь у тестах викликається _make_key.
    @staticmethod
    def _make_key(
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
    ) -> str:
        return LiquidityService.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    # ------------------------------------------------------------------
    # Payload helpers
    # ------------------------------------------------------------------

    def _scoped_snapshot_payload(
        self,
        context: LiquidityServiceContext,
        snapshot: LiquidityMapSnapshot,
    ) -> dict[str, Any]:
        payload = snapshot.to_event_payload()
        payload.update(
            {
                "exchange": context.exchange,
                "market_type": context.market_type,
                "symbol": context.symbol,
                "timeframe": context.timeframe,
                "context_key": context.key,
            }
        )
        return payload

    def _event_payload(self, event: Event | dict[str, Any]) -> dict[str, Any]:
        if isinstance(event, Event):
            if not isinstance(event.payload, dict):
                raise ValueError("Event payload must be dict")
            return event.payload

        if isinstance(event, dict):
            return event

        raise TypeError(f"Unsupported event type: {type(event)!r}")

    def _extract_required(
        self,
        payload: dict[str, Any],
        field_name: str,
    ) -> Any:
        value = self._extract_optional(payload, field_name)

        if value is None:
            raise ValueError(f"Event payload must contain '{field_name}'")

        return value

    def _extract_required_str(
        self,
        payload: dict[str, Any],
        field_name: str,
    ) -> str:
        value = self._extract_required(payload, field_name)
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"Event payload field '{field_name}' must not be empty")
        return normalized

    @staticmethod
    def _extract_optional(
        payload: dict[str, Any],
        field_name: str,
        default: Any = None,
    ) -> Any:
        return payload.get(field_name, default)

    def _extract_market_type(self, payload: dict[str, Any]) -> str:
        value = (
            payload.get("market_type")
            or payload.get("category")
            or payload.get("inst_type")
            or "perpetual"
        )
        return self._normalize_market_type_value(value)

    def _extract_event_timestamp(
        self,
        payload: dict[str, Any],
    ) -> datetime | None:
        value = get_first_value(
            payload,
            (
                "timestamp_ms",
                "received_at_ms",
                "last_update_ts_ms",
                "last_update_received_ms",
                "close_time_ms",
                "open_time_ms",
                "timestamp",
                "time",
                "event_time",
                "close_time",
                "open_time",
                "ts",
            ),
        )
        return self._parse_datetime(value)

    def _extract_price_from_candle(self, candle: Any) -> float | None:
        price = safe_float(get_candle_close(candle), default=0.0)
        return price if price > 0 else None

    def _extract_mid_price(self, payload: dict[str, Any]) -> float | None:
        mid_price = safe_float(payload.get("mid_price"), default=0.0)
        if mid_price > 0:
            return mid_price

        best_bid = payload.get("best_bid")
        best_ask = payload.get("best_ask")

        bid_price = self._extract_level_price(best_bid)
        ask_price = self._extract_level_price(best_ask)

        if bid_price > 0 and ask_price > 0:
            return (bid_price + ask_price) / 2.0

        bids = payload.get("bids") or []
        asks = payload.get("asks") or []

        bid_price = self._extract_level_price(bids[0]) if bids else 0.0
        ask_price = self._extract_level_price(asks[0]) if asks else 0.0

        if bid_price > 0 and ask_price > 0:
            return (bid_price + ask_price) / 2.0

        return None

    @staticmethod
    def _extract_level_price(level: Any) -> float:
        if level is None:
            return 0.0

        if isinstance(level, dict):
            return safe_float(
                level.get("price")
                or level.get("p")
                or level.get(0),
                default=0.0,
            )

        if isinstance(level, (list, tuple)) and level:
            return safe_float(level[0], default=0.0)

        return safe_float(getattr(level, "price", None), default=0.0)

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            return self._normalize_timestamp(value)

        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(
                    value / 1000 if value > 1e12 else value,
                    tz=timezone.utc,
                )
            except (OSError, OverflowError, ValueError):
                return None

        if isinstance(value, str):
            try:
                normalized = value.replace("Z", "+00:00")
                return self._normalize_timestamp(
                    datetime.fromisoformat(normalized)
                )
            except ValueError:
                return None

        return None

    @staticmethod
    def _normalize_timestamp(value: datetime | None) -> datetime | None:
        if value is None:
            return None

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)

    @staticmethod
    def _normalize_exchange(value: Any) -> str:
        normalized = str(value or "").strip()
        return normalized if normalized else "unknown"

    @staticmethod
    def _normalize_market_type_value(value: Any) -> str:
        normalized = str(value or "").strip()
        return normalized if normalized else "perpetual"

    @staticmethod
    def _normalize_symbol(value: Any) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _normalize_timeframe(value: Any) -> str:
        return str(value or "").strip()

    # ------------------------------------------------------------------
    # Error / time
    # ------------------------------------------------------------------

    def _error_extra(self, event: Event | dict[str, Any]) -> dict[str, Any]:
        if isinstance(event, Event):
            return {
                "topic": getattr(event, "topic", None),
                "event_id": getattr(event, "event_id", None),
            }

        return {
            "topic": None,
            "event_id": None,
        }

    def _handle_error(
        self,
        message: str,
        exc: Exception,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._stats.errors_count += 1
        self._stats.last_error = str(exc)
        self._stats.last_error_at = self._utcnow()

        payload: dict[str, Any] = {"error": str(exc)}
        if extra:
            payload.update(extra)

        self._logger.exception(message, extra=payload)

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)