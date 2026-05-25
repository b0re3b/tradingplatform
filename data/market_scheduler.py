from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler
from data.market_models import DirtyReason, normalize_exchange, normalize_market_type, normalize_symbol, normalize_timeframe
from data.market_snapshots import MarketSnapshot
from data.market_state import MarketStateStore


SnapshotCallback = Callable[[MarketSnapshot], Awaitable[None] | None]
BatchSnapshotCallback = Callable[[list[MarketSnapshot]], Awaitable[None] | None]
SnapshotPredicate = Callable[[MarketSnapshot], bool]


@dataclass(slots=True)
class MarketSchedulerConfig:
    enabled: bool = True
    interval_seconds: float = 1.0
    batch_size: int = 100
    snapshot_depth: int = 50
    run_immediately: bool = False
    job_timeout_seconds: float = 30.0
    emit_snapshot_ready_events: bool = False
    emit_empty_ticks: bool = False
    service_name: str = "market_scheduler"
    eventbus_skip_utilization_threshold: float = 0.80
    infer_scope_filters_from_name: bool = True
    infer_dirty_reason_filters_from_name: bool = True
    max_snapshots_per_evaluator_per_tick: int | None = None

    def validate(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.snapshot_depth <= 0:
            raise ValueError("snapshot_depth must be > 0")
        if not 0 <= self.eventbus_skip_utilization_threshold <= 1:
            raise ValueError("eventbus_skip_utilization_threshold must be between 0 and 1")
        if self.max_snapshots_per_evaluator_per_tick is not None and self.max_snapshots_per_evaluator_per_tick <= 0:
            raise ValueError("max_snapshots_per_evaluator_per_tick must be > 0 when set")


@dataclass(slots=True)
class MarketSchedulerStats:
    ticks: int = 0
    empty_ticks: int = 0
    snapshots_evaluated: int = 0
    snapshots_dispatched: int = 0
    callback_errors: int = 0
    skipped_overlap: int = 0
    skipped_eventbus_saturated: int = 0
    evaluator_invocations: int = 0
    batch_callback_invocations: int = 0
    snapshot_callback_invocations: int = 0
    filtered_snapshot_candidates: int = 0
    last_tick_snapshot_count: int = 0
    last_tick_dispatch_count: int = 0
    last_tick_evaluator_invocations: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


@dataclass(slots=True)
class RegisteredEvaluator:
    name: str
    callback: SnapshotCallback | None = None
    batch_callback: BatchSnapshotCallback | None = None
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    exchanges: frozenset[str] | None = None
    market_types: frozenset[str] | None = None
    symbols: frozenset[str] | None = None
    timeframes: frozenset[str] | None = None
    dirty_reasons: frozenset[str] | None = None
    predicate: SnapshotPredicate | None = None
    max_snapshots_per_tick: int | None = None

    def matches(self, snapshot: MarketSnapshot) -> bool:
        scope = snapshot.scope
        if self.exchanges is not None and scope.exchange not in self.exchanges:
            return False
        if self.market_types is not None and scope.market_type not in self.market_types:
            return False
        if self.symbols is not None and scope.symbol not in self.symbols:
            return False
        if self.timeframes is not None:
            timeframe = scope.timeframe or ""
            if timeframe not in self.timeframes:
                # Trades, orderbook, funding, OI, liquidations and price updates
                # are symbol-level state updates and legitimately arrive without
                # a candle timeframe.  Do not drop those snapshots only because a
                # consumer has a default timeframe for its own rolling window.
                reasons = {str(reason) for reason in snapshot.dirty_reasons}
                timeframe_neutral_reasons = {
                    DirtyReason.TRADE.value,
                    DirtyReason.TRADES_BATCH.value,
                    DirtyReason.ORDERBOOK.value,
                    DirtyReason.ORDERBOOK_RESYNC_REQUIRED.value,
                    DirtyReason.REST_SNAPSHOT.value,
                    DirtyReason.FUNDING.value,
                    DirtyReason.OPEN_INTEREST.value,
                    DirtyReason.LIQUIDATION.value,
                    DirtyReason.PRICE.value,
                }
                if timeframe or not reasons.intersection(timeframe_neutral_reasons):
                    return False
        if self.dirty_reasons is not None:
            reasons = {str(reason) for reason in snapshot.dirty_reasons}
            if not reasons.intersection(self.dirty_reasons):
                return False
        if self.predicate is not None and not self.predicate(snapshot):
            return False
        return True

    def filtered(self, snapshots: list[MarketSnapshot], *, default_limit: int | None = None) -> list[MarketSnapshot]:
        limit = self.max_snapshots_per_tick or default_limit
        matched: list[MarketSnapshot] = []
        for snapshot in snapshots:
            if self.matches(snapshot):
                matched.append(snapshot)
                if limit is not None and len(matched) >= limit:
                    break
        return matched

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
            "exchanges": sorted(self.exchanges) if self.exchanges is not None else None,
            "market_types": sorted(self.market_types) if self.market_types is not None else None,
            "symbols": sorted(self.symbols) if self.symbols is not None else None,
            "timeframes": sorted(self.timeframes) if self.timeframes is not None else None,
            "dirty_reasons": sorted(self.dirty_reasons) if self.dirty_reasons is not None else None,
            "has_callback": self.callback is not None,
            "has_batch_callback": self.batch_callback is not None,
            "has_predicate": self.predicate is not None,
            "max_snapshots_per_tick": self.max_snapshots_per_tick,
        }


class MarketScheduler:
    """
    Controlled snapshot evaluator for state-driven analytics.

    The scheduler drains dirty scopes from MarketStateStore at a bounded cadence,
    builds copy-on-read snapshots and dispatches each evaluator only to matching
    scopes/reasons.  This avoids the high-load anti-pattern where every analyzer
    receives every dirty snapshot and then discards most of them itself.
    """

    _DOMAIN_REASON_FILTERS: dict[str, frozenset[str]] = {
        "price_action": frozenset({DirtyReason.CANDLE.value, DirtyReason.CANDLE_CLOSED.value, DirtyReason.WARMUP.value}),
        "liquidity": frozenset({
            DirtyReason.CANDLE.value,
            DirtyReason.CANDLE_CLOSED.value,
            DirtyReason.WARMUP.value,
        }),
        "orderflow": frozenset({DirtyReason.TRADE.value, DirtyReason.TRADES_BATCH.value, DirtyReason.ORDERBOOK.value}),
        "funding": frozenset({DirtyReason.FUNDING.value}),
        "open_interest": frozenset({DirtyReason.OPEN_INTEREST.value}),
        "oi": frozenset({DirtyReason.OPEN_INTEREST.value}),
        "liquidations": frozenset({DirtyReason.LIQUIDATION.value}),
        "liquidation": frozenset({DirtyReason.LIQUIDATION.value}),
        "whales": frozenset({DirtyReason.TRADE.value, DirtyReason.TRADES_BATCH.value, DirtyReason.LIQUIDATION.value}),
        "whale": frozenset({DirtyReason.TRADE.value, DirtyReason.TRADES_BATCH.value, DirtyReason.LIQUIDATION.value}),
        "spoofing": frozenset({
            DirtyReason.ORDERBOOK.value,
            DirtyReason.ORDERBOOK_RESYNC_REQUIRED.value,
            DirtyReason.REST_SNAPSHOT.value,
            DirtyReason.TRADE.value,
            DirtyReason.TRADES_BATCH.value,
        }),
        "spreads": frozenset({
            DirtyReason.ORDERBOOK.value,
            DirtyReason.ORDERBOOK_RESYNC_REQUIRED.value,
            DirtyReason.REST_SNAPSHOT.value,
            DirtyReason.PRICE.value,
            DirtyReason.FUNDING.value,
            DirtyReason.OPEN_INTEREST.value,
            DirtyReason.TRADE.value,
            DirtyReason.TRADES_BATCH.value,
        }),
        "spread": frozenset({
            DirtyReason.ORDERBOOK.value,
            DirtyReason.ORDERBOOK_RESYNC_REQUIRED.value,
            DirtyReason.REST_SNAPSHOT.value,
            DirtyReason.PRICE.value,
            DirtyReason.FUNDING.value,
            DirtyReason.OPEN_INTEREST.value,
            DirtyReason.TRADE.value,
            DirtyReason.TRADES_BATCH.value,
        }),
    }

    def __init__(
        self,
        *,
        state_store: MarketStateStore,
        scheduler: Scheduler,
        event_bus: EventBus | None = None,
        config: MarketSchedulerConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        self.state_store = state_store
        self.scheduler = scheduler
        self.event_bus = event_bus
        self.config = config or MarketSchedulerConfig()
        self.config.validate()
        self._service_name = service_name or self.config.service_name
        self._logger = get_logger(__name__, service=self._service_name, event_type="market_scheduler")
        self._evaluators: dict[str, RegisteredEvaluator] = {}
        self._job_id: str | None = None
        self._running = False
        self._tick_lock = asyncio.Lock()
        self._stats = MarketSchedulerStats()

    def register_evaluator(
        self,
        name: str,
        callback: SnapshotCallback | None = None,
        *,
        batch_callback: BatchSnapshotCallback | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
        exchange: str | Iterable[str] | None = None,
        market_type: str | Iterable[str] | None = None,
        symbol: str | Iterable[str] | None = None,
        timeframe: str | Iterable[str] | None = None,
        dirty_reasons: str | DirtyReason | Iterable[str | DirtyReason] | None = None,
        predicate: SnapshotPredicate | None = None,
        max_snapshots_per_tick: int | None = None,
    ) -> None:
        if callback is None and batch_callback is None:
            raise ValueError("callback or batch_callback is required")
        if max_snapshots_per_tick is not None and max_snapshots_per_tick <= 0:
            raise ValueError("max_snapshots_per_tick must be > 0 when set")

        meta = dict(metadata or {})
        inferred = self._infer_filters_from_name_and_metadata(name=name, metadata=meta)
        evaluator = RegisteredEvaluator(
            name=name,
            callback=callback,
            batch_callback=batch_callback,
            enabled=enabled,
            metadata=meta,
            exchanges=self._normalize_exchanges(exchange) or inferred.get("exchanges"),
            market_types=self._normalize_market_types(market_type) or inferred.get("market_types"),
            symbols=self._normalize_symbols(symbol) or inferred.get("symbols"),
            timeframes=self._normalize_timeframes(timeframe) or inferred.get("timeframes"),
            dirty_reasons=self._normalize_dirty_reasons(dirty_reasons) or inferred.get("dirty_reasons"),
            predicate=predicate,
            max_snapshots_per_tick=max_snapshots_per_tick,
        )
        self._evaluators[name] = evaluator
        self._logger.info(
            "Market snapshot evaluator registered | name=%s enabled=%s filters=%s",
            name,
            enabled,
            evaluator.diagnostics(),
        )

    def unregister_evaluator(self, name: str) -> None:
        self._evaluators.pop(name, None)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self.config.enabled:
            self._job_id = self.scheduler.add_interval_job(
                name="market-snapshot-evaluation",
                func=self.evaluate_dirty_once,
                interval=self.config.interval_seconds,
                run_immediately=self.config.run_immediately,
                max_retries=0,
                retry_delay=1.0,
                timeout=self.config.job_timeout_seconds,
                allow_overlap=False,
                enabled=True,
            )
        self._logger.info(
            "MarketScheduler started | interval=%s batch_size=%s evaluators=%s job_id=%s",
            self.config.interval_seconds,
            self.config.batch_size,
            len(self._evaluators),
            self._job_id,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._job_id is not None:
            self.scheduler.remove_job(self._job_id)
            self._job_id = None
        self._logger.info("MarketScheduler stopped | stats=%s", self._stats.to_dict())

    async def evaluate_dirty_once(self) -> dict[str, Any]:
        if self._tick_lock.locked():
            self._stats.skipped_overlap += 1
            return {"skipped": True, "reason": "previous_tick_running"}

        if self.event_bus is not None:
            utilization = self.event_bus.stats().get("queue_utilization", 0.0)
            threshold = self.config.eventbus_skip_utilization_threshold
            if utilization > threshold:
                self._stats.skipped_eventbus_saturated += 1
                self._logger.warning(
                    "Skipping evaluation tick — EventBus saturated | utilization=%.2f threshold=%.2f skipped_total=%s",
                    utilization,
                    threshold,
                    self._stats.skipped_eventbus_saturated,
                )
                return {"skipped": True, "reason": "eventbus_saturated", "utilization": utilization}

        async with self._tick_lock:
            self._stats.ticks += 1
            snapshots = await self.state_store.snapshots_for_dirty(
                limit=self.config.batch_size,
                depth=self.config.snapshot_depth,
            )
            self._stats.last_tick_snapshot_count = len(snapshots)
            self._stats.last_tick_dispatch_count = 0
            self._stats.last_tick_evaluator_invocations = 0

            if not snapshots:
                self._stats.empty_ticks += 1
                if self.config.emit_empty_ticks:
                    await self._emit_snapshot_ready([])
                return {"snapshots": 0, "evaluators": len(self._evaluators), "dispatched": 0}

            enabled = [item for item in self._evaluators.values() if item.enabled]
            dispatches: dict[str, int] = {}
            for evaluator in enabled:
                matched = evaluator.filtered(
                    snapshots,
                    default_limit=self.config.max_snapshots_per_evaluator_per_tick,
                )
                self._stats.filtered_snapshot_candidates += len(snapshots)
                if not matched:
                    dispatches[evaluator.name] = 0
                    continue

                dispatches[evaluator.name] = len(matched)
                self._stats.last_tick_dispatch_count += len(matched)
                self._stats.snapshots_dispatched += len(matched)
                try:
                    invoked = False
                    if evaluator.batch_callback is not None:
                        result = evaluator.batch_callback(matched)
                        if inspect.isawaitable(result):
                            await result
                        self._stats.batch_callback_invocations += 1
                        invoked = True
                    if evaluator.callback is not None:
                        for snapshot in matched:
                            result = evaluator.callback(snapshot)
                            if inspect.isawaitable(result):
                                await result
                            self._stats.snapshot_callback_invocations += 1
                            invoked = True
                    if invoked:
                        self._stats.evaluator_invocations += 1
                        self._stats.last_tick_evaluator_invocations += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._stats.callback_errors += 1
                    self._stats.last_error = str(exc)
                    self._logger.exception(
                        "Market snapshot evaluator failed | evaluator=%s matched_snapshots=%s total_snapshots=%s",
                        evaluator.name,
                        len(matched),
                        len(snapshots),
                    )

            self._stats.snapshots_evaluated += len(snapshots)
            await self._emit_snapshot_ready(snapshots)
            return {
                "snapshots": len(snapshots),
                "evaluators": len(enabled),
                "dispatched": self._stats.last_tick_dispatch_count,
                "evaluator_invocations": self._stats.last_tick_evaluator_invocations,
                "callback_errors": self._stats.callback_errors,
                "dispatches": dispatches,
            }

    async def _emit_snapshot_ready(self, snapshots: list[MarketSnapshot]) -> None:
        if not self.config.emit_snapshot_ready_events or self.event_bus is None:
            return
        await self.event_bus.emit(
            "market.state.snapshot_ready",
            {
                "snapshot_count": len(snapshots),
                "symbols": [snapshot.scope.symbol for snapshot in snapshots[:50]],
                "scopes": [snapshot.scope.to_dict() for snapshot in snapshots[:50]],
                "stats": self._stats.to_dict(),
            },
            priority=EventPriority.LOW,
            source=self._service_name,
        )

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats.to_dict(),
            "running": self._running,
            "job_id": self._job_id,
            "evaluators": {name: evaluator.diagnostics() for name, evaluator in self._evaluators.items()},
        }

    def evaluator_diagnostics(self) -> dict[str, dict[str, Any]]:
        return {name: evaluator.diagnostics() for name, evaluator in self._evaluators.items()}

    def _infer_filters_from_name_and_metadata(self, *, name: str, metadata: dict[str, Any]) -> dict[str, frozenset[str] | None]:
        inferred: dict[str, frozenset[str] | None] = {
            "exchanges": None,
            "market_types": None,
            "symbols": None,
            "timeframes": None,
            "dirty_reasons": None,
        }

        metadata_exchange = metadata.get("exchange") or metadata.get("exchanges")
        metadata_market_type = metadata.get("market_type") or metadata.get("market_types")
        metadata_symbol = metadata.get("symbol") or metadata.get("symbols")
        metadata_timeframe = metadata.get("timeframe") or metadata.get("timeframes")
        metadata_reasons = metadata.get("dirty_reasons") or metadata.get("reasons")

        inferred["exchanges"] = self._normalize_exchanges(metadata_exchange)
        inferred["market_types"] = self._normalize_market_types(metadata_market_type)
        inferred["symbols"] = self._normalize_symbols(metadata_symbol)
        inferred["timeframes"] = self._normalize_timeframes(metadata_timeframe)
        inferred["dirty_reasons"] = self._normalize_dirty_reasons(metadata_reasons)

        if self.config.infer_scope_filters_from_name:
            parts = [part for part in name.split(":") if part]
            if len(parts) >= 3 and "price_action" in parts[0]:
                inferred["symbols"] = inferred["symbols"] or self._normalize_symbols(parts[1])
                inferred["timeframes"] = inferred["timeframes"] or self._normalize_timeframes(parts[2])

        if self.config.infer_dirty_reason_filters_from_name and inferred["dirty_reasons"] is None:
            lower_name = name.lower()
            domain = str(metadata.get("domain") or metadata.get("feature_source") or "").lower()
            for candidate, reasons in self._DOMAIN_REASON_FILTERS.items():
                if candidate and (candidate in lower_name or candidate == domain):
                    inferred["dirty_reasons"] = reasons
                    break

        return inferred

    @staticmethod
    def _as_iterable(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, DirtyReason):
            return [value]
        if isinstance(value, Iterable):
            return list(value)
        return [value]

    @classmethod
    def _normalize_exchanges(cls, value: Any) -> frozenset[str] | None:
        items = [normalize_exchange(item) for item in cls._as_iterable(value) if item is not None and str(item).strip()]
        return frozenset(items) or None

    @classmethod
    def _normalize_market_types(cls, value: Any) -> frozenset[str] | None:
        items = [normalize_market_type(item) for item in cls._as_iterable(value) if item is not None and str(item).strip()]
        return frozenset(items) or None

    @classmethod
    def _normalize_symbols(cls, value: Any) -> frozenset[str] | None:
        items = [normalize_symbol(item) for item in cls._as_iterable(value) if item is not None and str(item).strip()]
        return frozenset(items) or None

    @classmethod
    def _normalize_timeframes(cls, value: Any) -> frozenset[str] | None:
        items: list[str] = []
        for item in cls._as_iterable(value):
            normalized = normalize_timeframe(item)
            if normalized:
                items.append(normalized)
        return frozenset(items) or None

    @classmethod
    def _normalize_dirty_reasons(cls, value: Any) -> frozenset[str] | None:
        items: list[str] = []
        for item in cls._as_iterable(value):
            if item is None:
                continue
            if isinstance(item, DirtyReason):
                items.append(item.value)
            else:
                text = str(item).strip().lower()
                if text:
                    items.append(text)
        return frozenset(items) or None
