from __future__ import annotations

from analytics.strategy_contract import ensure_strategy_payload_contract
import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler
from analytics.market_state_contract import MarketStateSnapshotSource, snapshot_candles

from analytics.liquidity.config import LiquidityConfig
from analytics.liquidity.enums import SweepStatus
from analytics.liquidity.liquidity_map import LiquidityMap
from analytics.liquidity.models import (
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
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
from .state import LiquidityState
from .utils import get_candle_close, get_first_value, safe_float


@dataclass(slots=True)
class LiquidityServiceStats:
    """
    Runtime stats для LiquidityService.
    """

    started_at: datetime | None = None
    stopped_at: datetime | None = None

    snapshots_built: int = 0

    candle_events_processed: int = 0
    candles_updated_events_processed: int = 0
    orderbook_events_processed: int = 0
    price_events_processed: int = 0

    skipped_by_scope_filter: int = 0
    skipped_no_context: int = 0
    skipped_not_enough_data: int = 0

    emitted_map_updates: int = 0
    emitted_level_events: int = 0
    emitted_sweep_events: int = 0
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
            "candles_updated_events_processed": self.candles_updated_events_processed,
            "orderbook_events_processed": self.orderbook_events_processed,
            "price_events_processed": self.price_events_processed,
            "skipped_by_scope_filter": self.skipped_by_scope_filter,
            "skipped_no_context": self.skipped_no_context,
            "skipped_not_enough_data": self.skipped_not_enough_data,
            "emitted_map_updates": self.emitted_map_updates,
            "emitted_level_events": self.emitted_level_events,
            "emitted_sweep_events": self.emitted_sweep_events,
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
            "last_error_at": self.last_error_at.isoformat() if self.last_error_at else None,
        }


@dataclass(slots=True)
class LiquidityServiceContext:
    """
    Runtime market context для конкретного exchange + market_type + symbol + timeframe.

    Context не читає біржі і не публікує події.
    Він тільки тримає локальний rolling context для LiquidityMap.
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
    last_emit_at: datetime | None = None

    def __post_init__(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.timeframe = normalize_timeframe(self.timeframe)

        if self.last_rebuild_at is not None:
            self.last_rebuild_at = ensure_utc(self.last_rebuild_at)
        if self.last_update_at is not None:
            self.last_update_at = ensure_utc(self.last_update_at)

    @property
    def key(self) -> LiquidityKey:
        return make_liquidity_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def scope(self) -> dict[str, str]:
        return liquidity_key_to_dict(self.key)

    @property
    def scope_key(self) -> str:
        return liquidity_key_to_string(self.key)

    def touch(self, ts: datetime | None = None) -> None:
        self.last_update_at = ensure_utc(ts or utc_now())

    def can_build(self, min_candles: int) -> bool:
        return (
            self.current_price is not None
            and self.current_price > 0
            and len(self.candles) >= min_candles
        )


class LiquidityService:
    """
    Production-ready orchestration layer для analytics/liquidity.

    Correct production flow:
        data/candles_cache.py
            -> market.candle.closed / market.candles.updated
            -> LiquidityService
            -> LiquidityMap
            -> LiquidityState
            -> analytics.liquidity.*

        data/orderbook_cache.py
            -> market.orderbook.updated
            -> LiquidityService
            -> LiquidityMap
            -> LiquidityState
            -> analytics.liquidity.*

    Відповідальність:
    - приймає EventBus / Scheduler / Config через constructor dependency injection;
    - підписується тільки на config-driven data-layer topics;
    - накопичує context per exchange + market_type + symbol + timeframe;
    - викликає pure LiquidityMap;
    - оновлює pure LiquidityState;
    - публікує analytics.liquidity.* через EventBus;
    - запускає cleanup / metrics / healthcheck через core.Scheduler.

    Цей service НЕ:
    - не читає біржі напряму;
    - не підключається до WebSocket;
    - не викликає strategy/risk/execution напряму;
    - не запускає власні uncontrolled asyncio loops.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None,
        config: LiquidityConfig,
        liquidity_map: LiquidityMap,
        state: LiquidityState | None = None,
        market_state_store: Any | None = None,
        service_name: str = "analytics_liquidity",
    ) -> None:
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._config = config
        self._config.validate()
        self._config.assert_production_topics_allowed()

        self._liquidity_map = liquidity_map
        self._state = state or LiquidityState()
        self._market_state_store = market_state_store
        self._state_snapshot_source = (
            MarketStateSnapshotSource(market_state_store) if market_state_store is not None else None
        )

        self._service_name = service_name

        self._contexts: dict[LiquidityKey, LiquidityServiceContext] = {}
        self._locks: dict[LiquidityKey, asyncio.Lock] = {}

        self._subscriptions: list[Subscription] = []
        self._scheduler_job_ids: list[str] = []

        self._stats = LiquidityServiceStats()

        self._registered = False
        self._running = False

        self._logger = get_logger(
            __name__,
            service_name=self._service_name,
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

        if not self._config.enabled:
            self._logger.info("LiquidityService registration skipped: disabled by config")
            return

        self._register_event_subscriptions()
        self._register_scheduler_jobs()

        self._registered = True

        self._logger.info(
            "LiquidityService registered",
            extra={
                "subscriptions": len(self._subscriptions),
                "scheduler_jobs": len(self._scheduler_job_ids),
                "input_topics": list(self._config.production_input_topics),
                "output_topics": list(self._config.output_topics),
                "scope": "exchange:market_type:symbol:timeframe",
            },
        )

    async def start(self) -> None:
        """
        Запускає runtime-state сервісу.

        EventBus і Scheduler мають запускатися централізовано bootstrap/main.
        """
        if self._running:
            self._logger.warning("LiquidityService already started")
            return

        if not self._config.enabled:
            self._logger.warning("LiquidityService start skipped: disabled by config")
            return

        if not self._registered:
            self.register()

        self._running = True
        self._stats.started_at = utc_now()
        self._stats.stopped_at = None

        self._logger.info(
            "LiquidityService started",
            extra={
                "scope": "exchange:market_type:symbol:timeframe",
                "input_topics": list(self._config.production_input_topics),
            },
        )

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
        self._stats.stopped_at = utc_now()

        self._logger.info(
            "LiquidityService stopped",
            extra=self._stats.to_payload(),
        )

    def _register_event_subscriptions(self) -> None:
        """
        Реєструє тільки topics із LiquidityConfig.
        Жодних hardcoded raw market topics у service.
        """
        for topic in self._config.candle_topics:
            self._config.assert_input_topic_allowed(topic)
            self._subscriptions.append(
                self._event_bus.subscribe(
                    topic,
                    self._on_candle_closed,
                    name=f"{self._service_name}.on_candle_closed",
                )
            )

        for topic in self._config.candles_updated_topics:
            self._config.assert_input_topic_allowed(topic)
            self._subscriptions.append(
                self._event_bus.subscribe(
                    topic,
                    self._on_candles_updated,
                    name=f"{self._service_name}.on_candles_updated",
                )
            )

        for topic in self._config.orderbook_topics:
            self._config.assert_input_topic_allowed(topic)
            self._subscriptions.append(
                self._event_bus.subscribe(
                    topic,
                    self._on_orderbook_updated,
                    name=f"{self._service_name}.on_orderbook_updated",
                )
            )

        for topic in self._config.price_topics:
            self._config.assert_input_topic_allowed(topic)
            self._subscriptions.append(
                self._event_bus.subscribe(
                    topic,
                    self._on_price_updated,
                    name=f"{self._service_name}.on_price_updated",
                )
            )

    def _unregister_event_subscriptions(self) -> None:
        for subscription in list(self._subscriptions):
            try:
                self._event_bus.unsubscribe(subscription)
            except Exception as exc:
                self._logger.warning(
                    "Failed to unsubscribe LiquidityService subscription",
                    extra={"subscription": repr(subscription), "error": repr(exc)},
                )

        self._subscriptions.clear()

    def _register_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            return

        if self._config.cleanup_enabled:
            self._scheduler_job_ids.append(
                self._add_interval_job_once(
                    name=self._config.cleanup_job_name,
                    func=self._cleanup,
                    interval=self._config.cleanup_interval_seconds,
                )
            )

        if self._config.emit_state_metrics:
            self._scheduler_job_ids.append(
                self._add_interval_job_once(
                    name=self._config.state_metrics_job_name,
                    func=self._emit_state_metrics,
                    interval=self._config.state_metrics_interval_seconds,
                )
            )

        self._scheduler_job_ids.append(
            self._add_interval_job_once(
                name=self._config.healthcheck_job_name,
                func=self._emit_healthcheck,
                interval=self._config.healthcheck_interval_seconds,
            )
        )

    def _add_interval_job_once(
        self,
        *,
        name: str,
        func: Any,
        interval: float,
    ) -> str:
        assert self._scheduler is not None

        existing = self._scheduler.get_job_by_name(name)
        if existing is not None:
            return existing.job_id

        return self._scheduler.add_interval_job(
            name=name,
            func=func,
            interval=interval,
            run_immediately=False,
            max_retries=self._config.scheduler_job_max_retries,
            retry_delay=self._config.scheduler_job_retry_delay_seconds,
            timeout=self._config.scheduler_job_timeout_seconds,
            allow_overlap=False,
            enabled=True,
        )

    def _unregister_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            self._scheduler_job_ids.clear()
            return

        for job_id in list(self._scheduler_job_ids):
            try:
                result = self._scheduler.remove_job(job_id)
                if inspect.isawaitable(result):
                    self._logger.warning(
                        "Scheduler.remove_job returned awaitable in sync cleanup; "
                        "job may need explicit async cleanup by caller",
                        extra={"job_id": job_id},
                    )
            except KeyError:
                self._logger.warning(
                    "Scheduler job already removed",
                    extra={"job_id": job_id},
                )
            except Exception as exc:
                self._logger.warning(
                    "Failed to remove scheduler job",
                    extra={"job_id": job_id, "error": repr(exc)},
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
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        if not self._config.should_process_key(key):
            self._stats.skipped_by_scope_filter += 1
            return None

        lock = self._get_lock(key)

        async with lock:
            context = self._contexts.get(key)
            if context is None:
                self._stats.skipped_no_context += 1
                self._logger.debug(
                    "Skip rebuild: context not found",
                    extra={
                        "scope": liquidity_key_to_dict(key),
                        "scope_key": liquidity_key_to_string(key),
                    },
                )
                return None

            return await self._rebuild_context_snapshot_locked(
                context=context,
                extra_levels=extra_levels,
                extra_clusters=extra_clusters,
                force=force,
            )

    async def process_market_state_snapshot(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
        force: bool = False,
    ) -> LiquidityMapSnapshot | None:
        """Evaluate liquidity from MarketStateStore candles/orderbook snapshot.

        This is the state-driven input contract and replaces direct reliance on
        market.candles.updated / market.orderbook.updated EventBus input.
        """
        source = getattr(self, "_state_snapshot_source", None)
        if source is None:
            self._stats.skipped_no_context += 1
            return None
        snapshot = await source.snapshot(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        if snapshot is None:
            self._stats.skipped_no_context += 1
            return None
        candles = snapshot_candles(snapshot, timeframe=timeframe)
        orderbook = getattr(snapshot, "orderbook", None)
        current_price = (
            getattr(snapshot, "last_price", None)
            or getattr(snapshot, "mark_price", None)
            or getattr(orderbook, "mid_price", None)
        )
        await self._on_candles_updated({
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": list(candles or []),
            "current_price": current_price,
            "source_topic": "market_state.snapshot",
        })
        return await self.rebuild_snapshot(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            force=force,
        )

    async def process_market_snapshot(self, snapshot: Any) -> LiquidityMapSnapshot | None:
        """MarketScheduler-compatible evaluator callback."""
        if snapshot is None:
            return None

        cfg = getattr(self, "_config", None) or getattr(self, "config", None)
        scope = getattr(snapshot, "scope", None)
        exchange = (
            getattr(scope, "exchange", None)
            or getattr(snapshot, "exchange", None)
            or getattr(cfg, "default_exchange", None)
            or "binance"
        )
        market_type = (
            getattr(scope, "market_type", None)
            or getattr(snapshot, "market_type", None)
            or getattr(cfg, "default_market_type", None)
            or "usdm_futures"
        )
        symbol = getattr(scope, "symbol", None) or getattr(snapshot, "symbol", None)
        timeframe = (
            getattr(scope, "timeframe", None)
            or getattr(snapshot, "timeframe", None)
            or getattr(cfg, "default_timeframe", None)
            or "1m"
        )
        if not symbol:
            return None
        return await self.process_market_state_snapshot(
            exchange=str(exchange),
            market_type=str(market_type),
            symbol=str(symbol).upper(),
            timeframe=str(timeframe),
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
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        state = self._state.get_key(key)
        if state is not None:
            return state.last_snapshot

        context = self._contexts.get(key)
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
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    async def on_candle_closed(self, event: Event | dict[str, Any]) -> None:
        await self._on_candle_closed(event)

    async def on_candles_updated(self, event: Event | dict[str, Any]) -> None:
        await self._on_candles_updated(event)

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

        try:
            payload = self._event_payload(event)

            candle = payload.get("candle") if isinstance(payload.get("candle"), dict) else payload
            exchange = self._extract_exchange(payload, fallback=candle)
            market_type = self._extract_market_type(payload, fallback=candle)
            symbol = self._extract_symbol(payload, fallback=candle)
            timeframe = self._extract_timeframe(payload, fallback=candle)

            key = self.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )

            if not self._config.should_process_key(key):
                self._stats.skipped_by_scope_filter += 1
                return

            self._stats.candle_events_processed += 1

            current_price = self._extract_optional(payload, "current_price")
            if current_price is None:
                current_price = self._extract_price_from_candle(candle)

            event_ts = self._extract_event_timestamp(payload) or self._extract_event_timestamp(candle) or utc_now()

            lock = self._get_lock(key)

            async with lock:
                context = self._get_or_create_context_from_key(key)
                context.candles.append(candle)
                context.candles = context.candles[-self._config.max_candles_per_context :]

                if current_price is not None:
                    price = safe_float(current_price)
                    if price > 0:
                        context.current_price = price

                context.touch(event_ts)

                state = self._state.get_or_create_key(key)
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

    async def _on_candles_updated(self, event: Event | dict[str, Any]) -> None:
        """
        Optional handler для market.candles.updated.

        Очікує payload:
            {
                exchange, market_type, symbol, timeframe,
                candles: [...]
            }

        Якщо payload містить тільки latest candle — обробляється як candle.closed/update.
        """
        if not self._running or not self._config.enabled:
            return

        try:
            payload = self._event_payload(event)

            candles_payload = (
                payload.get("candles")
                or payload.get("recent_candles")
                or payload.get("items")
            )

            if not candles_payload:
                await self._on_candle_closed(event)
                return

            if not isinstance(candles_payload, Sequence):
                raise ValueError("candles payload must be a sequence")

            exchange = self._extract_exchange(payload)
            market_type = self._extract_market_type(payload)
            symbol = self._extract_symbol(payload)
            timeframe = self._extract_timeframe(payload)

            key = self.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )

            if not self._config.should_process_key(key):
                self._stats.skipped_by_scope_filter += 1
                return

            self._stats.candles_updated_events_processed += 1

            current_price = self._extract_optional(payload, "current_price")
            if current_price is None and candles_payload:
                current_price = self._extract_price_from_candle(candles_payload[-1])

            event_ts = self._extract_event_timestamp(payload) or utc_now()

            lock = self._get_lock(key)

            async with lock:
                context = self._get_or_create_context_from_key(key)

                context.candles = list(candles_payload)[-self._config.max_candles_per_context :]

                if current_price is not None:
                    price = safe_float(current_price)
                    if price > 0:
                        context.current_price = price

                context.touch(event_ts)

                state = self._state.get_or_create_key(key)
                state.record_candle_processed(ts=event_ts)

                await self._rebuild_context_snapshot_locked(
                    context=context,
                    force=False,
                )

        except Exception as exc:
            self._handle_error(
                "Failed to process candles updated event",
                exc,
                extra=self._error_extra(event),
            )

    async def _on_orderbook_updated(self, event: Event | dict[str, Any]) -> None:
        if not self._running or not self._config.enabled:
            return

        try:
            payload = self._event_payload(event)

            exchange = self._extract_exchange(payload)
            market_type = self._extract_market_type(payload)
            symbol = self._extract_symbol(payload)

            if not self._config.should_process_scope(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=self._config.default_timeframe,
            ):
                self._stats.skipped_by_scope_filter += 1
                return

            self._stats.orderbook_events_processed += 1

            bids = self._extract_optional(payload, "bids", default=[])
            asks = self._extract_optional(payload, "asks", default=[])
            current_price = self._extract_optional(payload, "current_price")

            if current_price is None:
                current_price = self._extract_mid_price(payload)

            event_ts = self._extract_event_timestamp(payload) or utc_now()

            # Orderbook не має природного timeframe, тому оновлюємо тільки ті
            # liquidity contexts, які вже існують для exchange + market_type + symbol.
            matching_keys = self._find_context_keys_for_market(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
            )

            if not matching_keys:
                self._stats.skipped_no_context += 1
                return

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

                    state = self._state.get_or_create_key(key)
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
        """
        Optional non-canonical price handler.

        За замовчуванням price topics вимкнені в LiquidityConfig.
        Canonical price source — candle close / candles cache.
        """
        if not self._running or not self._config.enabled:
            return

        if not self._config.allow_price_input_topics:
            return

        try:
            payload = self._event_payload(event)

            exchange = self._extract_exchange(payload)
            market_type = self._extract_market_type(payload)
            symbol = self._extract_symbol(payload)

            price = safe_float(self._extract_required(payload, "price"))
            if price <= 0:
                raise ValueError("price must be > 0")

            timeframe_value = self._extract_optional(payload, "timeframe")
            event_ts = self._extract_event_timestamp(payload) or utc_now()

            if timeframe_value is not None:
                key = self.make_key(
                    exchange=exchange,
                    market_type=market_type,
                    symbol=symbol,
                    timeframe=normalize_timeframe(timeframe_value),
                )
                keys = [key]
            else:
                keys = self._find_context_keys_for_market(
                    exchange=exchange,
                    market_type=market_type,
                    symbol=symbol,
                )

            if not keys:
                self._stats.skipped_no_context += 1
                return

            self._stats.price_events_processed += 1

            for key in keys:
                if not self._config.should_process_key(key):
                    self._stats.skipped_by_scope_filter += 1
                    continue

                lock = self._get_lock(key)

                async with lock:
                    context = self._contexts.get(key)
                    if context is None:
                        continue

                    context.current_price = price
                    context.touch(event_ts)

                    state = self._state.get_or_create_key(key)
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
            self._stats.skipped_not_enough_data += 1
            return None

        if not force and not self._should_rebuild_context(context):
            return None

        if context.current_price is None or context.current_price <= 0:
            self._stats.skipped_not_enough_data += 1
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
                    "scope": context.scope,
                    "scope_key": context.scope_key,
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
        if snapshot.liquidity_key != context.key:
            raise ValueError(
                "LiquidityMapSnapshot scope mismatch: "
                f"context={context.scope}, snapshot={snapshot.scope}"
            )

        previous_snapshot = context.last_snapshot

        context.last_snapshot = snapshot
        context.last_rebuild_at = ensure_utc(snapshot.timestamp)
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
        now = utc_now()
        if context.last_emit_at is not None:
            delta = (now - ensure_utc(context.last_emit_at)).total_seconds()
            if delta < self._config.snapshot_rebuild_min_interval_seconds:
                return
        context.last_emit_at = now

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
        accepted = await self._safe_emit(
            topic=self._config.map_updated_topic,
            payload=self._scoped_snapshot_payload(context, snapshot),
            priority=EventPriority.NORMAL,
            context=context,
            event_type="map_updated",
        )
        if accepted:
            self._stats.emitted_map_updates += 1

    async def _emit_signal_updated(
        self,
        context: LiquidityServiceContext,
        snapshot: LiquidityMapSnapshot,
    ) -> None:
        if snapshot.signal is None:
            return

        payload = snapshot.signal.to_event_payload()
        payload.update(self._scope_payload(context))

        accepted = await self._safe_emit(
            topic=self._config.signal_updated_topic,
            payload=payload,
            priority=EventPriority.NORMAL,
            context=context,
            event_type="signal_updated",
        )
        if accepted:
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
                payload.update(self._scope_payload(context))

                accepted = await self._safe_emit(
                    topic=self._config.level_detected_topic,
                    payload=payload,
                    priority=EventPriority.NORMAL,
                    context=context,
                    event_type="level_detected",
                )
                if accepted:
                    self._stats.emitted_level_events += 1
                continue

            if previous is None:
                continue

            sweep_changed = previous.sweep_status != level.sweep_status
            swept_now = level.sweep_status in {
                SweepStatus.PARTIALLY_SWEPT,
                SweepStatus.SWEPT,
            }

            if self._config.emit_sweep_events and sweep_changed and swept_now:
                payload = level.to_event_payload()
                payload.update(self._scope_payload(context))

                accepted = await self._safe_emit(
                    topic=self._config.level_swept_topic,
                    payload=payload,
                    priority=EventPriority.HIGH,
                    context=context,
                    event_type="level_swept",
                )
                if accepted:
                    self._stats.emitted_sweep_events += 1

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
            payload.update(self._scope_payload(context))

            accepted = await self._safe_emit(
                topic=self._config.stop_cluster_detected_topic,
                payload=payload,
                priority=EventPriority.NORMAL,
                context=context,
                event_type="stop_cluster_detected",
            )
            if accepted:
                self._stats.emitted_cluster_events += 1

    async def _safe_emit(
        self,
        *,
        topic: str,
        payload: dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        context: LiquidityServiceContext | None = None,
        event_type: str | None = None,
    ) -> bool:
        try:
            headers: dict[str, str] = {}
            if context is not None:
                headers.update(
                    {
                        "exchange": context.exchange,
                        "market_type": context.market_type,
                        "symbol": context.symbol,
                        "timeframe": context.timeframe,
                        "scope_key": context.scope_key,
                    }
                )
            if event_type:
                headers["event_type"] = event_type

            strategy_payload = ensure_strategy_payload_contract(
                payload,
                topic=topic,
                source=self._service_name,
                domain="liquidity",
            )
            return await self._event_bus.emit(
                topic,
                strategy_payload,
                priority=priority,
                source=self._service_name,
                headers=headers,
            )
        except Exception as exc:
            self._handle_error(
                "Failed to emit liquidity event",
                exc,
                extra={"topic": topic},
            )
            return False

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
            "service": self._service_name,
            "timestamp": utc_now().isoformat(),
            "scope": "exchange:market_type:symbol:timeframe",
            "stats": self._stats.to_payload(),
            "state": self._state.to_metrics_payload(),
            "contexts_count": len(self._contexts),
            "context_keys": [
                liquidity_key_to_string(key)
                for key in self._contexts.keys()
            ],
        }

        accepted = await self._safe_emit(
            topic=self._config.state_metrics_topic,
            payload=payload,
            priority=EventPriority.LOW,
            event_type="state_metrics",
        )
        if accepted:
            self._stats.emitted_metrics_events += 1

    async def _emit_healthcheck(self) -> None:
        if not self._running or not self._config.publish_events:
            return

        payload = {
            "service": self._service_name,
            "timestamp": utc_now().isoformat(),
            "running": self._running,
            "registered": self._registered,
            "scope": "exchange:market_type:symbol:timeframe",
            "contexts_count": len(self._contexts),
            "states_count": self._state.count(),
            "subscriptions": len(self._subscriptions),
            "scheduler_jobs": len(self._scheduler_job_ids),
            "input_topics": list(self._config.production_input_topics),
            "output_topics": list(self._config.output_topics),
            "errors_count": self._stats.errors_count,
            "last_error": self._stats.last_error,
            "context_keys": [
                liquidity_key_to_string(key)
                for key in self._contexts.keys()
            ],
        }

        accepted = await self._safe_emit(
            topic=self._config.healthcheck_topic,
            payload=payload,
            priority=EventPriority.LOW,
            event_type="healthcheck",
        )
        if accepted:
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
            key=lambda item: item[1].last_update_at or datetime.min.replace(tzinfo=utc_now().tzinfo),
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

        delta = utc_now() - ensure_utc(context.last_rebuild_at)
        return delta.total_seconds() >= self._config.snapshot_rebuild_min_interval_seconds

    # ------------------------------------------------------------------
    # Context / locks
    # ------------------------------------------------------------------

    def _get_or_create_context_from_key(
        self,
        key: LiquidityKey,
    ) -> LiquidityServiceContext:
        if key not in self._contexts:
            scope = liquidity_key_to_dict(key)
            self._contexts[key] = LiquidityServiceContext(
                exchange=scope["exchange"],
                market_type=scope["market_type"],
                symbol=scope["symbol"],
                timeframe=scope["timeframe"],
            )

        return self._contexts[key]

    def _find_context_keys_for_market(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
    ) -> list[LiquidityKey]:
        prefix = (
            normalize_exchange(exchange),
            normalize_market_type(market_type),
            normalize_symbol(symbol),
        )

        return [
            key
            for key in self._contexts.keys()
            if key[:3] == prefix
        ]

    def _get_lock(self, key: LiquidityKey) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        return self._locks[key]

    @staticmethod
    def make_key(
        *,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str,
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
        *,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> str:
        return liquidity_key_to_string(
            LiquidityService.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    # Backward-compatible private alias.
    @staticmethod
    def _make_key(
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
    ) -> LiquidityKey:
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
        payload.update(self._scope_payload(context))
        return payload

    @staticmethod
    def _scope_payload(context: LiquidityServiceContext) -> dict[str, Any]:
        return {
            "exchange": context.exchange,
            "market_type": context.market_type,
            "symbol": context.symbol,
            "timeframe": context.timeframe,
            "scope": context.scope,
            "scope_key": context.scope_key,
            "liquidity_key": context.key,
            "context_key": context.scope_key,
        }

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

    def _extract_exchange(
        self,
        payload: dict[str, Any],
        *,
        fallback: Any | None = None,
    ) -> str:
        value = (
            payload.get("exchange")
            or payload.get("ex")
            or payload.get("source_exchange")
        )

        if value is None and isinstance(fallback, dict):
            value = (
                fallback.get("exchange")
                or fallback.get("ex")
                or fallback.get("source_exchange")
            )

        return normalize_exchange(value or self._config.default_exchange)

    def _extract_market_type(
        self,
        payload: dict[str, Any],
        *,
        fallback: Any | None = None,
    ) -> str:
        value = (
            payload.get("market_type")
            or payload.get("category")
            or payload.get("inst_type")
            or payload.get("market")
        )

        if value is None and isinstance(fallback, dict):
            value = (
                fallback.get("market_type")
                or fallback.get("category")
                or fallback.get("inst_type")
                or fallback.get("market")
            )

        return normalize_market_type(value or self._config.default_market_type)

    def _extract_symbol(
        self,
        payload: dict[str, Any],
        *,
        fallback: Any | None = None,
    ) -> str:
        value = (
            payload.get("symbol")
            or payload.get("s")
            or payload.get("instrument")
            or payload.get("instId")
            or payload.get("pair")
        )

        if value is None and isinstance(fallback, dict):
            value = (
                fallback.get("symbol")
                or fallback.get("s")
                or fallback.get("instrument")
                or fallback.get("instId")
                or fallback.get("pair")
            )

        return normalize_symbol(value)

    def _extract_timeframe(
        self,
        payload: dict[str, Any],
        *,
        fallback: Any | None = None,
    ) -> str:
        value = (
            payload.get("timeframe")
            or payload.get("tf")
            or payload.get("interval")
        )

        if value is None and isinstance(fallback, dict):
            value = (
                fallback.get("timeframe")
                or fallback.get("tf")
                or fallback.get("interval")
            )

        return normalize_timeframe(value or self._config.default_timeframe)

    def _extract_event_timestamp(
        self,
        payload: Any,
    ) -> datetime | None:
        if not isinstance(payload, dict):
            return None

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
                    tz=utc_now().tzinfo,
                )
            except (OSError, OverflowError, ValueError):
                return None

        if isinstance(value, str):
            try:
                normalized = value.replace("Z", "+00:00")
                return self._normalize_timestamp(datetime.fromisoformat(normalized))
            except ValueError:
                return None

        return None

    @staticmethod
    def _normalize_timestamp(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return ensure_utc(value)

    # ------------------------------------------------------------------
    # Error / diagnostics
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
        self._stats.last_error_at = utc_now()

        payload: dict[str, Any] = {"error": str(exc)}
        if extra:
            payload.update(extra)

        self._logger.exception(message, extra=payload)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_registered(self) -> bool:
        return self._registered


__all__ = [
    "LiquidityServiceStats",
    "LiquidityServiceContext",
    "LiquidityService",
]